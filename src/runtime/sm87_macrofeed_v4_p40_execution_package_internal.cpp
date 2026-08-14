#include "sm87_macrofeed_v4_p40_execution_package_internal.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail {
namespace {

namespace events = sm87_macrofeed_v4_execution_events_detail;
namespace startup = sm87_macrofeed_v4_p40_startup_package_detail;
using Error = Sm87MacroFeedV4P40ExecutionPackageError;
using Status = Sm87MacroFeedV4P40ExecutionPackageStatus;

std::atomic<std::uint64_t> g_next_execution_identity{1U};
std::atomic<std::uint64_t> g_next_front_half_receipt_identity{1U};
std::atomic<std::uint64_t> g_next_complete_layer_receipt_identity{1U};

[[nodiscard]] std::uint64_t next_nonzero(
    std::atomic<std::uint64_t>* const source) noexcept {
  std::uint64_t value = source->fetch_add(1U, std::memory_order_relaxed);
  if (value == 0U) {
    value = source->fetch_add(1U, std::memory_order_relaxed);
  }
  return value;
}

[[nodiscard]] constexpr std::uint64_t mix64(std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xbf58'476d'1ce4'e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d0'49bb'1331'11ebULL;
  value ^= value >> 31U;
  return value;
}

[[nodiscard]] constexpr Status ok() noexcept { return {}; }

[[nodiscard]] float fp32_from_bits(const std::uint32_t bits) noexcept {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] Status failure(
    const Error error, const char* const context, const int cuda_error = 0,
    const std::size_t layer = kSm87MacroFeedV4LayerCount,
    const bool post_attention_norm = false,
    const events::Sm87MacroFeedV4ExecutionStatus& event_status = {}) noexcept {
  Status status;
  status.error = error;
  status.context = context;
  status.cuda_error = cuda_error;
  status.layer = layer;
  status.post_attention_norm = post_attention_norm;
  status.event_status = event_status;
  return status;
}

[[nodiscard]] Status event_failure(
    const char* const context,
    const events::Sm87MacroFeedV4ExecutionStatus& event_status) noexcept {
  return failure(Error::kExecutionEvent, context, event_status.cuda_error,
                 event_status.panel, false, event_status);
}

[[nodiscard]] bool pointer_aligned(const void* const pointer,
                                   const std::size_t alignment) noexcept {
  return pointer != nullptr && alignment != 0U &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] bool exact_live_synthetic_fp8_asset(
    const kernels::Sm87TargetAotFp8CudaAssetView& asset,
    const kernels::Sm87TargetAotProjectionRole expected_role,
    const std::int32_t expected_device, int* const cuda_error) noexcept {
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(expected_role);
  const auto& upload = asset.device_upload_receipt;
  if (cuda_error == nullptr || expected_device < 0 || !layout.valid() ||
      !kernels::sm87_target_aot_fp8_cuda_asset_valid(asset) ||
      asset.payload.role != expected_role ||
      asset.payload.bytes != layout.payload_bytes ||
      asset.payload.begin == 0U || asset.payload.end <= asset.payload.begin ||
      asset.payload.end - asset.payload.begin != asset.payload.bytes ||
      upload.role != expected_role || upload.device_ordinal != expected_device ||
      upload.device_payload_begin != asset.payload.begin ||
      upload.device_payload_end != asset.payload.end ||
      upload.device_payload_bytes != asset.payload.bytes ||
      upload.device_allocation_begin == 0U ||
      upload.device_allocation_end <= upload.device_allocation_begin ||
      upload.device_allocation_end - upload.device_allocation_begin !=
          upload.device_allocation_bytes ||
      upload.device_allocation_owner_identity == 0U ||
      upload.device_allocation_identity == 0U) {
    if (cuda_error != nullptr) {
      *cuda_error = static_cast<int>(cudaErrorInvalidValue);
    }
    return false;
  }

  int current_device = -1;
  cudaError_t runtime_status = cudaGetDevice(&current_device);
  if (runtime_status != cudaSuccess || current_device != expected_device) {
    *cuda_error = runtime_status == cudaSuccess
                      ? static_cast<int>(cudaErrorInvalidDevice)
                      : static_cast<int>(runtime_status);
    return false;
  }

  const void* const begin =
      reinterpret_cast<const void*>(asset.payload.begin);
  const void* const end =
      reinterpret_cast<const void*>(asset.payload.end - 1U);
  cudaPointerAttributes begin_attributes{};
  cudaPointerAttributes end_attributes{};
  runtime_status = cudaPointerGetAttributes(&begin_attributes, begin);
  if (runtime_status != cudaSuccess) {
    *cuda_error = static_cast<int>(runtime_status);
    return false;
  }
  runtime_status = cudaPointerGetAttributes(&end_attributes, end);
  if (runtime_status != cudaSuccess ||
      begin_attributes.type != cudaMemoryTypeDevice ||
      end_attributes.type != cudaMemoryTypeDevice ||
      begin_attributes.device != expected_device ||
      end_attributes.device != expected_device) {
    *cuda_error = runtime_status == cudaSuccess
                      ? static_cast<int>(cudaErrorInvalidDevicePointer)
                      : static_cast<int>(runtime_status);
    return false;
  }

  CUdeviceptr allocation_begin = 0U;
  CUdeviceptr end_allocation_begin = 0U;
  std::size_t allocation_bytes = 0U;
  std::size_t end_allocation_bytes = 0U;
  const CUresult driver_status = cuMemGetAddressRange(
      &allocation_begin, &allocation_bytes,
      static_cast<CUdeviceptr>(asset.payload.begin));
  const CUresult end_driver_status = cuMemGetAddressRange(
      &end_allocation_begin, &end_allocation_bytes,
      static_cast<CUdeviceptr>(asset.payload.end - 1U));
  const bool overflows =
      allocation_bytes > std::numeric_limits<std::uintptr_t>::max() -
                             static_cast<std::uintptr_t>(allocation_begin);
  const std::uintptr_t allocation_end =
      overflows ? 0U
                : static_cast<std::uintptr_t>(allocation_begin) +
                      allocation_bytes;
  if (driver_status != CUDA_SUCCESS || end_driver_status != CUDA_SUCCESS ||
      allocation_begin == 0U || allocation_bytes == 0U || overflows ||
      end_allocation_begin != allocation_begin ||
      end_allocation_bytes != allocation_bytes ||
      static_cast<std::uintptr_t>(allocation_begin) !=
          upload.device_allocation_begin ||
      allocation_bytes != upload.device_allocation_bytes ||
      allocation_end != upload.device_allocation_end) {
    *cuda_error =
        driver_status != CUDA_SUCCESS
            ? static_cast<int>(driver_status)
            : (end_driver_status != CUDA_SUCCESS
                   ? static_cast<int>(end_driver_status)
                   : static_cast<int>(cudaErrorInvalidDevicePointer));
    return false;
  }
  return true;
}

[[nodiscard]] bool exact_live_device_range(
    const void* const pointer, const std::uint64_t bytes,
    const std::int32_t expected_device,
    const std::uintptr_t expected_allocation_begin,
    const std::uint64_t expected_allocation_bytes,
    int* const cuda_error) noexcept {
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  if (cuda_error == nullptr || pointer == nullptr || bytes == 0U ||
      expected_device < 0 ||
      bytes > std::numeric_limits<std::uintptr_t>::max() -
                  reinterpret_cast<std::uintptr_t>(pointer)) {
    if (cuda_error != nullptr) {
      *cuda_error = static_cast<int>(cudaErrorInvalidValue);
    }
    return false;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  const auto end = begin + static_cast<std::uintptr_t>(bytes);
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess || current_device != expected_device) {
    *cuda_error = status == cudaSuccess
                      ? static_cast<int>(cudaErrorInvalidDevice)
                      : static_cast<int>(status);
    return false;
  }
  cudaPointerAttributes begin_attributes{};
  cudaPointerAttributes end_attributes{};
  status = cudaPointerGetAttributes(&begin_attributes, pointer);
  if (status == cudaSuccess) {
    status = cudaPointerGetAttributes(
        &end_attributes, reinterpret_cast<const void*>(end - 1U));
  }
  if (status != cudaSuccess ||
      begin_attributes.type != cudaMemoryTypeDevice ||
      end_attributes.type != cudaMemoryTypeDevice ||
      begin_attributes.device != expected_device ||
      end_attributes.device != expected_device) {
    *cuda_error = status == cudaSuccess
                      ? static_cast<int>(cudaErrorInvalidDevicePointer)
                      : static_cast<int>(status);
    return false;
  }
  CUdeviceptr allocation_begin = 0U;
  CUdeviceptr end_allocation_begin = 0U;
  std::size_t allocation_bytes = 0U;
  std::size_t end_allocation_bytes = 0U;
  const CUresult driver_status = cuMemGetAddressRange(
      &allocation_begin, &allocation_bytes, static_cast<CUdeviceptr>(begin));
  const CUresult end_driver_status = cuMemGetAddressRange(
      &end_allocation_begin, &end_allocation_bytes,
      static_cast<CUdeviceptr>(end - 1U));
  const bool allocation_overflows =
      allocation_bytes > std::numeric_limits<std::uintptr_t>::max() -
                             static_cast<std::uintptr_t>(allocation_begin);
  const auto allocation_end =
      allocation_overflows
          ? std::uintptr_t{0U}
          : static_cast<std::uintptr_t>(allocation_begin) + allocation_bytes;
  const bool expected_allocation_matches =
      expected_allocation_begin == 0U
          ? expected_allocation_bytes == 0U
          : expected_allocation_bytes == allocation_bytes &&
                expected_allocation_begin ==
                    static_cast<std::uintptr_t>(allocation_begin);
  if (driver_status != CUDA_SUCCESS || end_driver_status != CUDA_SUCCESS ||
      allocation_begin == 0U || allocation_bytes == 0U ||
      allocation_overflows || end_allocation_begin != allocation_begin ||
      end_allocation_bytes != allocation_bytes || begin < allocation_begin ||
      end > allocation_end || !expected_allocation_matches) {
    *cuda_error =
        driver_status != CUDA_SUCCESS
            ? static_cast<int>(driver_status)
            : (end_driver_status != CUDA_SUCCESS
                   ? static_cast<int>(end_driver_status)
                   : static_cast<int>(cudaErrorInvalidDevicePointer));
    return false;
  }
  return true;
}

[[nodiscard]] bool exact_live_synthetic_nvfp4_asset(
    const kernels::Sm87TargetAotNvFp4CudaAssetView& asset,
    const kernels::Sm87TargetAotProjectionRole expected_role,
    const std::int32_t expected_device, int* const cuda_error) noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(expected_role);
  const auto& upload = asset.device_upload_receipt;
  if (!layout.valid() ||
      !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(asset) ||
      asset.payload.role != expected_role ||
      asset.payload.bytes != layout.payload_bytes ||
      upload.role != expected_role || upload.device_ordinal != expected_device ||
      upload.device_payload_begin != asset.payload.begin ||
      upload.device_payload_end != asset.payload.end ||
      upload.device_payload_bytes != asset.payload.bytes ||
      upload.device_allocation_owner_identity == 0U ||
      upload.device_allocation_identity == 0U) {
    if (cuda_error != nullptr) {
      *cuda_error = static_cast<int>(cudaErrorInvalidValue);
    }
    return false;
  }
  return exact_live_device_range(
      reinterpret_cast<const void*>(asset.payload.begin), asset.payload.bytes,
      expected_device, upload.device_allocation_begin,
      upload.device_allocation_bytes, cuda_error);
}

}  // namespace

Sm87MacroFeedV4P40ExecutionPackage::Sm87MacroFeedV4P40ExecutionPackage(
    ProjectionCatalog projection_catalog, Bf16AbCatalog bf16_ab_catalog,
    LayerNormCatalog layer_norm_catalog, GdnQkvZCatalog gdn_qkvz_catalog,
    MlpPairCatalog mlp_pair_catalog,
    GdnLayer0ExecutionSource gdn_layer0_source,
    std::optional<CompleteGdnLayer0ExecutionSource>
        complete_gdn_layer0_source,
    kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot
        norm_resources,
    kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot
        bf16_ab_resources,
    kernels::Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot gdn_resources,
    kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources gate_up_resources,
    kernels::Sm87MacroFeedV4NvFp4DownCudaResources down_resources,
    void* const transient_allocation, void* const recurrent_allocation,
    std::unique_ptr<Sm87MacroFeedV4RequestState> request_state,
    std::shared_ptr<EventsOwner> events_owner,
    std::unique_ptr<EventsDriver> events_driver,
    Sm87MacroFeedV4P40ExecutionPackageAudit audit) noexcept
    : projection_catalog_(std::move(projection_catalog)),
      bf16_ab_catalog_(std::move(bf16_ab_catalog)),
      layer_norm_catalog_(std::move(layer_norm_catalog)),
      gdn_qkvz_catalog_(std::move(gdn_qkvz_catalog)),
      mlp_pair_catalog_(std::move(mlp_pair_catalog)),
      gdn_layer0_source_(gdn_layer0_source),
      complete_gdn_layer0_source_(std::move(complete_gdn_layer0_source)),
      norm_resources_(norm_resources),
      bf16_ab_resources_(bf16_ab_resources),
      gdn_resources_(gdn_resources),
      gate_up_resources_(gate_up_resources),
      down_resources_(down_resources),
      transient_allocation_(transient_allocation),
      recurrent_allocation_(recurrent_allocation),
      ping_(reinterpret_cast<std::uint16_t*>(
          static_cast<std::uint8_t*>(transient_allocation_) +
          kSm87MacroFeedV4P40ExecutionPingOffset)),
      pong_(reinterpret_cast<std::uint16_t*>(
          static_cast<std::uint8_t*>(transient_allocation_) +
          kSm87MacroFeedV4P40ExecutionPongOffset)),
      scratch_(reinterpret_cast<std::uint16_t*>(
          static_cast<std::uint8_t*>(transient_allocation_) +
          kSm87MacroFeedV4P40ExecutionScratchOffset)),
      request_state_(std::move(request_state)),
      events_owner_(std::move(events_owner)),
      events_driver_(std::move(events_driver)),
      audit_(audit),
      construction_postconditions_sealed_(true) {}

Sm87MacroFeedV4P40ExecutionPackage::~Sm87MacroFeedV4P40ExecutionPackage()
    noexcept {
  release();
}

std::uint64_t Sm87MacroFeedV4P40ExecutionPackage::
    compute_gdn_layer_catalog_fold_identity(
        const GdnQkvZCatalog& catalog) noexcept {
  std::uint64_t identity = 0x5133'4d46'5634'4743ULL;
  identity = mix64(identity ^ catalog.size());
  for (std::size_t ordinal = 0U; ordinal < catalog.size(); ++ordinal) {
    const std::uint64_t binding_identity =
        StartupPackage::compute_gdn_qkvz_execution_binding_identity(
            catalog[ordinal]);
    if (binding_identity == 0U ||
        binding_identity != catalog[ordinal].binding_identity) {
      return 0U;
    }
    identity = mix64(identity ^ (ordinal + 1U));
    identity = mix64(identity ^ binding_identity);
  }
  return identity == 0U ? 1U : identity;
}

std::uint64_t Sm87MacroFeedV4P40ExecutionPackage::
    compute_mlp_pair_catalog_fold_identity(
        const MlpPairCatalog& catalog) noexcept {
  std::uint64_t identity = 0x5133'4d46'5634'4d43ULL;
  identity = mix64(identity ^ catalog.size());
  for (std::size_t layer = 0U; layer < catalog.size(); ++layer) {
    const std::uint64_t binding_identity =
        StartupPackage::compute_mlp_pair_execution_binding_identity(
            catalog[layer]);
    if (binding_identity == 0U ||
        binding_identity != catalog[layer].binding_identity) {
      return 0U;
    }
    identity = mix64(identity ^ (layer + 1U));
    identity = mix64(identity ^ binding_identity);
  }
  return identity == 0U ? 1U : identity;
}

std::uint64_t Sm87MacroFeedV4P40ExecutionPackage::
    compute_complete_layer0_source_identity(
        const CompleteGdnLayer0ExecutionSource& source) noexcept {
  const auto& gdn = source.gdn_layer;
  const auto& mlp = source.mlp_pair;
  const std::array<std::uint64_t, 4U> asset_identities{{
      StartupPackage::compute_gdn_qkvz_asset_value_identity(gdn.asset),
      StartupPackage::compute_gdn_qkvz_asset_value_identity(
          gdn.gdn_output.asset),
      StartupPackage::compute_nvfp4_asset_value_identity(mlp.gate_up.asset),
      StartupPackage::compute_nvfp4_asset_value_identity(mlp.down.asset),
  }};
  for (const std::uint64_t identity : asset_identities) {
    if (identity == 0U) {
      return 0U;
    }
  }
  if (!kernels::sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
          source.gate_up_receipt) ||
      !kernels::sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
          source.down_receipt)) {
    return 0U;
  }

  std::uint64_t identity = source.synthetic_t1
                               ? 0x5133'4d46'5634'5359ULL
                               : 0x5133'4d46'5634'524cULL;
  const auto add = [&identity](const std::uint64_t value) noexcept {
    identity = mix64(identity ^ value);
  };
  add(source.synthetic_t1);
  add(gdn.binding_identity);
  add(mlp.binding_identity);
  for (const std::uint64_t asset_identity : asset_identities) {
    add(asset_identity);
  }
  add(static_cast<std::uint64_t>(gdn.resources.identity));
  add(static_cast<std::uint64_t>(gdn.gdn_output.resources.identity));
  add(gdn.gdn_output.binding_identity);
  add(gdn.continuation.conv_weight_identity);
  add(gdn.continuation.a_log_identity);
  add(gdn.continuation.dt_bias_identity);
  add(gdn.continuation.norm_weight_identity);
  add(gdn.continuation.aggregate_identity);
  add(reinterpret_cast<std::uintptr_t>(gdn.continuation.conv_weight));
  add(reinterpret_cast<std::uintptr_t>(gdn.continuation.a_log));
  add(reinterpret_cast<std::uintptr_t>(gdn.continuation.dt_bias));
  add(reinterpret_cast<std::uintptr_t>(gdn.continuation.norm_weight));
  add(source.gate_up_receipt.receipt_identity);
  add(source.gate_up_receipt.payload_identity);
  add(source.gate_up_receipt.gate_source_identity);
  add(source.gate_up_receipt.up_source_identity);
  add(source.down_receipt.receipt_identity);
  add(source.down_receipt.payload_identity);
  add(mlp.gate_up.tactic_identity);
  add(mlp.down.tactic_identity);
  return identity == 0U ? 1U : identity;
}

Sm87MacroFeedV4P40ExecutionPackageCreateResult::operator bool()
    const noexcept {
  return package != nullptr && static_cast<bool>(status) && audit.valid() &&
         package->valid() &&
         package->audit().package_identity == audit.package_identity;
}

Sm87MacroFeedV4P40ExecutionPackageCreateResult
Sm87MacroFeedV4P40ExecutionPackage::create(
    const StartupPackage& startup_package) noexcept {
  return create_impl(startup_package, nullptr, nullptr);
}

Sm87MacroFeedV4P40ExecutionPackageCreateResult
Sm87MacroFeedV4P40ExecutionPackage::create_impl(
    const StartupPackage& startup_package,
    const kernels::Sm87TargetAotFp8CudaAssetView* const
        synthetic_t1_gdn_layer0_asset,
    const SyntheticCompleteGdnLayer0Source* const
        synthetic_complete_gdn_layer0_source) noexcept {
  Sm87MacroFeedV4P40ExecutionPackageCreateResult result;
#if !defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_EXECUTION_EVENTS_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_NORM_RESIDUAL_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_GDN_C8000_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_NVFP4_GATE_UP_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_NVFP4_DOWN_ADMISSION)
  (void)startup_package;
  (void)synthetic_t1_gdn_layer0_asset;
  (void)synthetic_complete_gdn_layer0_source;
  result.status = failure(Error::kAdmissionDisabled, "admission_disabled");
  return result;
#else
  if (!startup_package.valid() || !startup_package.audit().valid()) {
    result.status = failure(Error::kStartupPackage,
                            "live_startup_package_required");
    return result;
  }

  // Copy every immutable binding by value.  The execution package never
  // retains a pointer into StartupPackage storage and remains destruction-safe
  // if it escapes the Engine composition root's ownership chain.
  ProjectionCatalog projection_catalog = startup_package.projection_bindings_;
  for (const auto& binding : projection_catalog) {
    if (!binding.has_value() || binding->binding_identity() == 0U ||
        binding->package_identity() !=
            startup_package.audit().package_identity) {
      result.status = failure(Error::kProjectionCatalog,
                              "projection_binding_identity");
      return result;
    }
  }

  Bf16AbCatalog bf16_ab_catalog{};
  if (!startup_package
           .seal_bf16_ab_execution_catalog_for_execution_package(
               &bf16_ab_catalog)) {
    result.status = failure(Error::kBf16AbCatalog,
                            "complete_bf16_ab_catalog_seal");
    return result;
  }

  LayerNormCatalog layer_norm_catalog{};
  std::uint64_t layer_norm_catalog_identity = 0U;
  std::size_t norm_failure_layer = kSm87MacroFeedV4LayerCount;
  bool norm_failure_post_attention = false;
  int norm_cuda_error = 0;
  if (!startup_package
           .seal_layer_norm_execution_catalog_for_execution_package(
               &layer_norm_catalog, &layer_norm_catalog_identity,
               &norm_failure_layer, &norm_failure_post_attention,
               &norm_cuda_error)) {
    result.status = failure(Error::kLayerNormCatalog,
                            "complete_layer_norm_catalog_seal",
                            norm_cuda_error, norm_failure_layer,
                            norm_failure_post_attention);
    return result;
  }

  // Production construction is all-or-nothing over the 48 natural GDN
  // layers.  The only exception is the private synthetic-T1 seam used to
  // exercise this one-layer composition against an honest live allocation;
  // it may proceed only when the known non-executable host catalog fails on
  // ordinal zero and it remains explicitly distinguishable in the audit.
  GdnQkvZCatalog gdn_qkvz_catalog{};
  std::uint64_t gdn_qkvz_catalog_identity = 0U;
  std::size_t gdn_failure_ordinal = kSm87MacroFeedV4StateLayerCount;
  int gdn_cuda_error = 0;
  const bool real_gdn_catalog =
      startup_package
          .seal_gdn_qkvz_execution_catalog_for_execution_package(
              &gdn_qkvz_catalog, &gdn_qkvz_catalog_identity,
              &gdn_failure_ordinal, &gdn_cuda_error);
  const bool synthetic_complete_requested =
      synthetic_complete_gdn_layer0_source != nullptr;
  const auto* const selected_synthetic_qkvz_asset =
      synthetic_complete_requested
          ? &synthetic_complete_gdn_layer0_source->gdn_qkvz_asset
          : synthetic_t1_gdn_layer0_asset;
  if (synthetic_complete_requested && synthetic_t1_gdn_layer0_asset != nullptr) {
    result.status = failure(Error::kGdnQkvZCatalog,
                            "ambiguous_synthetic_layer0_source");
    return result;
  }
  GdnLayer0ExecutionSource gdn_layer0_source;
  if (selected_synthetic_qkvz_asset == nullptr) {
    if (!real_gdn_catalog || gdn_qkvz_catalog_identity == 0U ||
        gdn_failure_ordinal != gdn_qkvz_catalog.size()) {
      const std::size_t failure_layer =
          gdn_failure_ordinal < gdn_qkvz_catalog.size()
              ? gdn_failure_ordinal + gdn_failure_ordinal / 3U
              : kSm87MacroFeedV4LayerCount;
      result.status = failure(Error::kGdnQkvZCatalog,
                              "complete_gdn_qkvz_catalog_seal",
                              gdn_cuda_error, failure_layer);
      return result;
    }
    const auto& binding = gdn_qkvz_catalog[0U];
    gdn_layer0_source.asset = binding.asset;
    gdn_layer0_source.resources = binding.resources;
    gdn_layer0_source.identity = binding.binding_identity;
    gdn_layer0_source.synthetic_t1 = false;
  } else {
    if (real_gdn_catalog || gdn_failure_ordinal != 0U ||
        gdn_cuda_error == 0) {
      result.status = failure(
          Error::kGdnQkvZCatalog,
          "synthetic_t1_requires_ordinal0_nonexecutable_catalog_failure",
          gdn_cuda_error);
      return result;
    }
    // cudaPointerGetAttributes intentionally failed on the fake catalog.
    // Clear only that host-thread diagnostic before inspecting the honest
    // synthetic allocation; no device work has been submitted at this point.
    (void)cudaGetLastError();

    const kernels::Sm87MacroFeedV4Fp8CudaResources resources =
        startup_package.gdn_qkvz_startup_seal().resources;
    constexpr auto kRole =
        kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
    constexpr auto kLayout =
        kernels::Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1;
    int synthetic_cuda_error = 0;
    const std::uint64_t asset_identity =
        StartupPackage::compute_gdn_qkvz_asset_value_identity(
            *selected_synthetic_qkvz_asset);
    if (!resources.static_resource_gate_passed ||
        !kernels::sm87_macrofeed_v4_fp8_resource_gate(resources) ||
        resources.role != kRole || resources.input_layout != kLayout ||
        resources.identity !=
            kernels::Sm87MacroFeedV4Fp8Identity::
                kGdnQkvZM64N128K64OrdinaryGridV1 ||
        resources.device_ordinal != startup_package.audit().device_ordinal ||
        asset_identity == 0U ||
        !exact_live_synthetic_fp8_asset(
            *selected_synthetic_qkvz_asset,
            kRole,
            startup_package.audit().device_ordinal,
            &synthetic_cuda_error)) {
      result.status = failure(
          Error::kGdnQkvZCatalog,
          "synthetic_t1_live_gdn_qkvz_source_required",
          synthetic_cuda_error);
      return result;
    }
    gdn_layer0_source.asset = *selected_synthetic_qkvz_asset;
    gdn_layer0_source.resources = resources;
    std::uint64_t source_identity =
        mix64(0x5133'4d46'5634'5431ULL ^ asset_identity);
    source_identity = mix64(
        source_identity ^ static_cast<std::uint64_t>(resources.identity));
    source_identity = mix64(
        source_identity ^ resources.static_resource_gate_passed);
    gdn_layer0_source.identity = source_identity == 0U ? 1U : source_identity;
    gdn_layer0_source.synthetic_t1 = true;
    gdn_qkvz_catalog.fill({});
    gdn_qkvz_catalog_identity = 0U;
  }

  MlpPairCatalog mlp_pair_catalog{};
  std::uint64_t mlp_pair_catalog_identity = 0U;
  std::size_t mlp_failure_layer = kSm87MacroFeedV4LayerCount;
  int mlp_cuda_error = 0;
  const bool real_mlp_catalog =
      startup_package.seal_mlp_pair_execution_catalog_for_execution_package(
          &mlp_pair_catalog, &mlp_pair_catalog_identity, &mlp_failure_layer,
          &mlp_cuda_error);
  const bool synthetic_requested = selected_synthetic_qkvz_asset != nullptr;
  if (!synthetic_requested) {
    if (!real_mlp_catalog || mlp_pair_catalog_identity == 0U ||
        mlp_failure_layer != mlp_pair_catalog.size()) {
      result.status = failure(Error::kMlpPairCatalog,
                              "complete_mlp_pair_catalog_seal",
                              mlp_cuda_error, mlp_failure_layer);
      return result;
    }
  } else {
    if (real_mlp_catalog || mlp_failure_layer != 0U || mlp_cuda_error == 0) {
      result.status = failure(
          Error::kMlpPairCatalog,
          "synthetic_t1_requires_layer0_nonexecutable_mlp_catalog_failure",
          mlp_cuda_error, mlp_failure_layer);
      return result;
    }
    (void)cudaGetLastError();
    mlp_pair_catalog.fill({});
    mlp_pair_catalog_identity = 0U;
  }

  const std::uint64_t retained_gdn_layer_catalog_fold_identity =
      synthetic_requested
          ? 0U
          : compute_gdn_layer_catalog_fold_identity(gdn_qkvz_catalog);
  const std::uint64_t retained_mlp_pair_catalog_fold_identity =
      synthetic_requested
          ? 0U
          : compute_mlp_pair_catalog_fold_identity(mlp_pair_catalog);
  if (!synthetic_requested &&
      (retained_gdn_layer_catalog_fold_identity == 0U ||
       retained_mlp_pair_catalog_fold_identity == 0U)) {
    result.status = failure(Error::kCompleteLayerBinding,
                            "retained_complete_catalog_identity_fold");
    return result;
  }

  std::optional<CompleteGdnLayer0ExecutionSource>
      complete_gdn_layer0_source;
  if (!synthetic_requested) {
    CompleteGdnLayer0ExecutionSource source;
    source.gdn_layer = gdn_qkvz_catalog[0U];
    source.mlp_pair = mlp_pair_catalog[0U];
    source.gate_up_receipt = source.mlp_pair.gate_up.payload_receipt;
    source.down_receipt = source.mlp_pair.down.payload_receipt;
    source.synthetic_t1 = false;
    source.identity = compute_complete_layer0_source_identity(source);
    if (source.identity == 0U) {
      result.status = failure(Error::kCompleteLayerBinding,
                              "real_complete_layer0_source_identity");
      return result;
    }
    complete_gdn_layer0_source.emplace(std::move(source));
  } else if (synthetic_complete_requested) {
    const auto& synthetic = *synthetic_complete_gdn_layer0_source;
    constexpr auto kOutputRole =
        kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput;
    constexpr auto kGateRole =
        kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp;
    constexpr auto kDownRole =
        kernels::Sm87TargetAotProjectionRole::kNvFp4Down;
    int synthetic_cuda_error = 0;
    const std::array<const void*, 4U> continuation_weights{{
        synthetic.conv_weight, synthetic.a_log, synthetic.dt_bias,
        synthetic.norm_weight}};
    const std::array<std::uint64_t, 4U> continuation_bytes{{
        kernels::kSm87MacroFeedV4GdnConvWeightBytes,
        kernels::kSm87MacroFeedV4GdnHeadVectorBytes,
        kernels::kSm87MacroFeedV4GdnHeadVectorBytes,
        kernels::kSm87MacroFeedV4GdnNormWeightBytes}};
    bool continuation_live = true;
    for (std::size_t index = 0U; index < continuation_weights.size();
         ++index) {
      continuation_live =
          continuation_live &&
          reinterpret_cast<std::uintptr_t>(continuation_weights[index]) %
                  kernels::kSm87MacroFeedV4GdnPointerAlignment ==
              0U &&
          exact_live_device_range(
              continuation_weights[index], continuation_bytes[index],
              startup_package.audit().device_ordinal, 0U, 0U,
              &synthetic_cuda_error);
    }
    for (std::size_t left = 0U; left < continuation_weights.size(); ++left) {
      const auto left_begin =
          reinterpret_cast<std::uintptr_t>(continuation_weights[left]);
      const auto left_end = left_begin + continuation_bytes[left];
      for (std::size_t right = left + 1U;
           right < continuation_weights.size(); ++right) {
        const auto right_begin =
            reinterpret_cast<std::uintptr_t>(continuation_weights[right]);
        const auto right_end = right_begin + continuation_bytes[right];
        continuation_live = continuation_live &&
                            (left_end <= right_begin || right_end <= left_begin);
      }
    }
    if (!exact_live_synthetic_fp8_asset(
            synthetic.gdn_output_asset, kOutputRole,
            startup_package.audit().device_ordinal,
            &synthetic_cuda_error) ||
        !exact_live_synthetic_nvfp4_asset(
            synthetic.gate_up_asset, kGateRole,
            startup_package.audit().device_ordinal,
            &synthetic_cuda_error) ||
        !exact_live_synthetic_nvfp4_asset(
            synthetic.down_asset, kDownRole,
            startup_package.audit().device_ordinal,
            &synthetic_cuda_error) ||
        !continuation_live ||
        !kernels::sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
            synthetic.gate_up_receipt) ||
        !kernels::sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
            synthetic.down_receipt) ||
        synthetic.gate_up_receipt.device_ordinal !=
            startup_package.audit().device_ordinal ||
        synthetic.gate_up_receipt.payload_identity !=
            synthetic.gate_up_asset.artifact_identity ||
        synthetic.gate_up_receipt.payload_begin !=
            synthetic.gate_up_asset.payload.begin ||
        synthetic.gate_up_receipt.payload_end !=
            synthetic.gate_up_asset.payload.end ||
        synthetic.gate_up_receipt.payload_bytes !=
            synthetic.gate_up_asset.payload.bytes ||
        synthetic.down_receipt.device_ordinal !=
            startup_package.audit().device_ordinal ||
        synthetic.down_receipt.payload_identity !=
            synthetic.down_asset.artifact_identity ||
        synthetic.down_receipt.payload_begin !=
            synthetic.down_asset.payload.begin ||
        synthetic.down_receipt.payload_end != synthetic.down_asset.payload.end ||
        synthetic.down_receipt.payload_bytes !=
            synthetic.down_asset.payload.bytes) {
      result.status = failure(Error::kCompleteLayerBinding,
                              "synthetic_complete_layer0_live_sources",
                              synthetic_cuda_error, 0U);
      return result;
    }

    CompleteGdnLayer0ExecutionSource source;
    source.gdn_layer.gdn_ordinal = 0U;
    source.gdn_layer.model_layer = 0U;
    source.gdn_layer.role =
        kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
    source.gdn_layer.input_layout =
        kernels::Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1;
    source.gdn_layer.tactic_identity =
        kernels::Sm87MacroFeedV4Fp8Identity::
            kGdnQkvZM64N128K64OrdinaryGridV1;
    source.gdn_layer.asset = synthetic.gdn_qkvz_asset;
    source.gdn_layer.resources = gdn_layer0_source.resources;
    source.gdn_layer.gdn_output.role = kOutputRole;
    source.gdn_layer.gdn_output.input_layout =
        kernels::Sm87MacroFeedV4Fp8InputLayout::kGdnContiguousVScratchV1;
    source.gdn_layer.gdn_output.tactic_identity =
        kernels::Sm87MacroFeedV4Fp8Identity::
            kGdnAttentionOutputM64N128K64OrdinaryGridV1;
    source.gdn_layer.gdn_output.asset = synthetic.gdn_output_asset;
    source.gdn_layer.gdn_output.resources =
        startup_package.gdn_qkvz_startup_seal().output_resources;
    source.gdn_layer.gdn_output.live_cuda_payload_range_validated = true;
    source.gdn_layer.continuation.conv_weight = synthetic.conv_weight;
    source.gdn_layer.continuation.a_log = synthetic.a_log;
    source.gdn_layer.continuation.dt_bias = synthetic.dt_bias;
    source.gdn_layer.continuation.norm_weight = synthetic.norm_weight;
    source.gdn_layer.continuation.exact_shapes = true;
    source.gdn_layer.continuation.live_cuda_weight_ranges_validated = true;
    source.gdn_layer.live_cuda_payload_range_validated = true;
    source.mlp_pair.model_layer = 0U;
    source.mlp_pair.gate_up.asset = synthetic.gate_up_asset;
    source.mlp_pair.gate_up.payload_receipt = synthetic.gate_up_receipt;
    source.mlp_pair.gate_up.tactic_identity = static_cast<std::uint64_t>(
        kernels::kSm87MacroFeedV4NvFp4GateUpIdentity);
    source.mlp_pair.down.asset = synthetic.down_asset;
    source.mlp_pair.down.payload_receipt = synthetic.down_receipt;
    source.mlp_pair.down.tactic_identity = static_cast<std::uint64_t>(
        kernels::kSm87MacroFeedV4NvFp4DownIdentity);
    source.mlp_pair.live_cuda_payload_ranges_validated = true;
    source.gate_up_receipt = synthetic.gate_up_receipt;
    source.down_receipt = synthetic.down_receipt;
    source.synthetic_t1 = true;
    source.identity = compute_complete_layer0_source_identity(source);
    if (source.identity == 0U) {
      result.status = failure(Error::kCompleteLayerBinding,
                              "synthetic_complete_layer0_source_identity");
      return result;
    }
    complete_gdn_layer0_source.emplace(std::move(source));
  }

  kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot
      norm_resources{};
  const int norm_resource_status =
      kernels::query_sm87_macrofeed_v4_norm_residual_admission_resources_cuda(
          &norm_resources);
  if (norm_resource_status != static_cast<int>(cudaSuccess) ||
      !kernels::sm87_macrofeed_v4_norm_residual_resource_gate(
          norm_resources) ||
      norm_resources.device_ordinal !=
          startup_package.audit().device_ordinal) {
    result.status = failure(Error::kNormResources,
                            "exact_norm_resource_seal",
                            norm_resource_status);
    return result;
  }
  const auto bf16_ab_resources =
      startup_package.bf16_ab_startup_seal().resources;
  if (!kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
          bf16_ab_resources) ||
      bf16_ab_resources.device_ordinal !=
          startup_package.audit().device_ordinal) {
    result.status = failure(Error::kBf16AbCatalog,
                            "exact_bf16_ab_resource_seal");
    return result;
  }
  kernels::Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot gdn_resources{};
  const int gdn_resource_status =
      kernels::query_sm87_macrofeed_v4_gdn_c8000_admission_resource_snapshot_cuda(
          &gdn_resources);
  if (gdn_resource_status != static_cast<int>(cudaSuccess) ||
      !kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
          gdn_resources) ||
      gdn_resources.device_ordinal !=
          startup_package.audit().device_ordinal) {
    result.status = failure(Error::kGdnResources,
                            "exact_gdn_resource_seal",
                            gdn_resource_status);
    return result;
  }
  const auto gate_up_resources =
      startup_package.gate_up_startup_seal().resources;
  if (!kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(
          gate_up_resources) ||
      gate_up_resources.device_ordinal !=
          startup_package.audit().device_ordinal) {
    result.status = failure(Error::kGateUpResources,
                            "exact_gate_up_resource_seal");
    return result;
  }
  const auto down_resources = startup_package.down_startup_seal().resources;
  if (!kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(down_resources) ||
      down_resources.device_ordinal !=
          startup_package.audit().device_ordinal) {
    result.status = failure(Error::kDownResources,
                            "exact_down_resource_seal");
    return result;
  }

  auto events_created =
      events::create_sm87_macrofeed_v4_execution_events_owner();
  if (!events_created) {
    result.status = failure(Error::kExecutionEvents,
                            "execution_event_owner_create", 0,
                            kSm87MacroFeedV4LayerCount, false,
                            events_created.status);
    return result;
  }
  std::shared_ptr<EventsOwner> events_owner;
  try {
    events_owner =
        std::shared_ptr<EventsOwner>(std::move(events_created.owner));
  } catch (const std::bad_alloc&) {
    result.status = failure(Error::kExecutionEvents,
                            "execution_event_owner_lifetime_allocation");
    return result;
  }
  auto events_driver =
      events::bind_sm87_macrofeed_v4_execution_events_driver(events_owner);
  if (events_driver == nullptr) {
    result.status = failure(Error::kExecutionEvents,
                            "execution_event_driver_bind");
    return result;
  }
  const std::uint64_t events_owner_identity =
      events_driver->owner_identity();
  if (events_owner_identity == 0U ||
      events_driver->device_ordinal() !=
          startup_package.audit().device_ordinal) {
    result.status = failure(Error::kExecutionEvents,
                            "execution_event_owner_device_identity");
    return result;
  }

  void* transient = nullptr;
  cudaError_t cuda_status = cudaMalloc(
      &transient,
      static_cast<std::size_t>(
          kSm87MacroFeedV4P40ExecutionTransientBytes));
  if (cuda_status != cudaSuccess || transient == nullptr) {
    result.status = failure(Error::kTransientAllocation,
                            "transient_allocation",
                            static_cast<int>(cuda_status));
    return result;
  }

  void* recurrent = nullptr;
  cuda_status = cudaMalloc(
      &recurrent,
      static_cast<std::size_t>(kSm87MacroFeedV4RecurrentStorageBytes));
  if (cuda_status != cudaSuccess || recurrent == nullptr) {
    (void)cudaFree(transient);
    result.status = failure(Error::kRecurrentAllocation,
                            "recurrent_allocation",
                            static_cast<int>(cuda_status));
    return result;
  }

  const std::uint64_t transient_identity =
      next_nonzero(&g_next_execution_identity);
  const std::uint64_t recurrent_identity =
      next_nonzero(&g_next_execution_identity);
  const std::uint64_t bank_a_identity =
      next_nonzero(&g_next_execution_identity);
  const std::uint64_t bank_b_identity =
      next_nonzero(&g_next_execution_identity);
  const auto cold_initialized = events_driver->initialize_cold_recurrent_storage(
      recurrent,
      static_cast<std::size_t>(kSm87MacroFeedV4RecurrentStorageBytes),
      recurrent_identity);
  if (!cold_initialized) {
    events_driver.reset();
    events_owner.reset();
    (void)cudaFree(recurrent);
    (void)cudaFree(transient);
    result.status = failure(Error::kColdRecurrentInitialization,
                            "cold_recurrent_storage_zero_and_seal",
                            cold_initialized.cuda_error,
                            kSm87MacroFeedV4LayerCount, false,
                            cold_initialized);
    return result;
  }
  const auto cold_snapshot = events_driver->snapshot();
  if (cold_snapshot.cold_recurrent_initializations != 1U ||
      cold_snapshot.cold_recurrent_allocation_identity !=
          recurrent_identity ||
      cold_snapshot.cold_recurrent_allocation_begin !=
          reinterpret_cast<std::uintptr_t>(recurrent) ||
      cold_snapshot.cold_recurrent_zero_bytes !=
          kSm87MacroFeedV4RecurrentStorageBytes) {
    events_driver.reset();
    events_owner.reset();
    (void)cudaFree(recurrent);
    (void)cudaFree(transient);
    result.status = failure(Error::kColdRecurrentInitialization,
                            "cold_recurrent_storage_seal_postcondition");
    return result;
  }
  const auto admission = make_sm87_macrofeed_v4_request_state_admission(
      events_owner_identity, recurrent_identity, bank_a_identity,
      bank_b_identity);
  auto request_created = Sm87MacroFeedV4RequestState::create(admission);
  if (!request_created) {
    (void)cudaFree(recurrent);
    (void)cudaFree(transient);
    result.status = failure(Error::kRequestState,
                            "request_state_owner_binding");
    return result;
  }

  Sm87MacroFeedV4P40ExecutionPackageAudit audit;
  audit.startup_package_identity = startup_package.audit().package_identity;
  audit.projection_catalog_identity = startup_package.audit().catalog_identity;
  audit.bf16_ab_catalog_identity =
      startup_package.audit().bf16_ab_binding_catalog_identity;
  audit.layer_norm_catalog_identity = layer_norm_catalog_identity;
  audit.gdn_qkvz_catalog_identity = gdn_qkvz_catalog_identity;
  audit.mlp_pair_catalog_identity = mlp_pair_catalog_identity;
  audit.retained_gdn_layer_catalog_fold_identity =
      retained_gdn_layer_catalog_fold_identity;
  audit.retained_mlp_pair_catalog_fold_identity =
      retained_mlp_pair_catalog_fold_identity;
  audit.gdn_layer0_source_identity =
      complete_gdn_layer0_source.has_value()
          ? complete_gdn_layer0_source->identity
          : gdn_layer0_source.identity;
  audit.transient_allocation_identity = transient_identity;
  audit.recurrent_allocation_identity = recurrent_identity;
  audit.execution_events_owner_identity = events_owner_identity;
  audit.device_ordinal = startup_package.audit().device_ordinal;
  audit.projection_bindings = projection_catalog.size();
  audit.bf16_ab_pairs = bf16_ab_catalog.size();
  audit.layer_norm_pairs = layer_norm_catalog.size();
  audit.gdn_qkvz_bindings =
      gdn_layer0_source.synthetic_t1 ? 1U : gdn_qkvz_catalog.size();
  audit.mlp_pair_bindings =
      gdn_layer0_source.synthetic_t1
          ? (complete_gdn_layer0_source.has_value() ? 1U : 0U)
          : mlp_pair_catalog.size();
  audit.transient_bytes = kSm87MacroFeedV4P40ExecutionTransientBytes;
  audit.recurrent_bytes = kSm87MacroFeedV4RecurrentStorageBytes;
  audit.cold_recurrent_zero_bytes = cold_snapshot.cold_recurrent_zero_bytes;
  audit.cold_recurrent_initializations =
      cold_snapshot.cold_recurrent_initializations;
  audit.fixed_gdn_layer0_front_half_bound = true;
  audit.fixed_gdn_layer0_complete_bound =
      complete_gdn_layer0_source.has_value();
  audit.qkvz_ab_ready_transaction_bound = true;
  audit.synthetic_t1_gdn_layer0_source = gdn_layer0_source.synthetic_t1;
  audit.whole_layer_executor_bound =
      complete_gdn_layer0_source.has_value();
  audit.whole_model_executor_bound = false;
  audit.selector_bound = false;
  audit.api_route_bound = false;
  audit.default_off = true;
  audit.jit_present = false;
  audit.request_time_repack_present = false;
  audit.request_time_autotune_present = false;
  audit.fallback_present = false;
  audit.cublaslt_present = false;
  audit.mtp_present = false;
  audit.production_dispatch_eligible = false;
  std::uint64_t package_identity = 0x5133'4d46'5634'4501ULL;
  package_identity = mix64(package_identity ^ audit.startup_package_identity);
  package_identity = mix64(package_identity ^ audit.projection_catalog_identity);
  package_identity = mix64(package_identity ^ audit.bf16_ab_catalog_identity);
  package_identity = mix64(package_identity ^ audit.layer_norm_catalog_identity);
  package_identity = mix64(package_identity ^ audit.gdn_qkvz_catalog_identity);
  package_identity = mix64(package_identity ^ audit.mlp_pair_catalog_identity);
  package_identity = mix64(
      package_identity ^ audit.retained_gdn_layer_catalog_fold_identity);
  package_identity = mix64(
      package_identity ^ audit.retained_mlp_pair_catalog_fold_identity);
  package_identity = mix64(package_identity ^ audit.gdn_layer0_source_identity);
  package_identity = mix64(package_identity ^ audit.transient_allocation_identity);
  package_identity = mix64(package_identity ^ audit.recurrent_allocation_identity);
  package_identity = mix64(package_identity ^ audit.cold_recurrent_zero_bytes);
  package_identity =
      mix64(package_identity ^ audit.execution_events_owner_identity);
  audit.package_identity = package_identity == 0U ? 1U : package_identity;
  if (!audit.valid()) {
    (void)cudaFree(recurrent);
    (void)cudaFree(transient);
    result.status = failure(Error::kPackageAllocation,
                            "execution_package_audit");
    return result;
  }

  result.package.reset(new (std::nothrow) Sm87MacroFeedV4P40ExecutionPackage(
      std::move(projection_catalog), std::move(bf16_ab_catalog),
      std::move(layer_norm_catalog), std::move(gdn_qkvz_catalog),
      std::move(mlp_pair_catalog), gdn_layer0_source,
      std::move(complete_gdn_layer0_source), norm_resources,
      bf16_ab_resources, gdn_resources, gate_up_resources, down_resources,
      transient, recurrent, std::move(request_created.state),
      std::move(events_owner), std::move(events_driver), audit));
  if (result.package == nullptr) {
    (void)cudaFree(recurrent);
    (void)cudaFree(transient);
    result.status = failure(Error::kPackageAllocation,
                            "execution_package_host_allocation");
    return result;
  }
  if (!result.package->valid()) {
    result.package.reset();
    result.status = failure(Error::kPackageAllocation,
                            "execution_package_postcondition");
    return result;
  }
  result.audit = audit;
  result.status = ok();
  return result;
#endif
}

bool Sm87MacroFeedV4P40ExecutionPackage::front_half_bindings_valid()
    const noexcept {
  if (ping_ == nullptr || pong_ == nullptr || scratch_ == nullptr ||
      bf16_ab_catalog_.empty() || layer_norm_catalog_.empty() ||
      gdn_layer0_source_.identity == 0U) {
    return false;
  }
  constexpr std::size_t kQkvzRowBegin = 0U;
  constexpr std::size_t kQkvzRowEnd =
      kernels::kSm87MacroFeedV4Fp8GdnZOffset +
      kernels::kSm87MacroFeedV4Fp8GdnZFeatures;
  constexpr std::size_t kAbRowBegin =
      kernels::kSm87MacroFeedV4Bf16AbAOffset;
  constexpr std::size_t kAbRowEnd =
      kernels::kSm87MacroFeedV4Bf16AbBOffset +
      kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection;
  static_assert(kQkvzRowBegin == 0U && kQkvzRowEnd == 16'384U);
  static_assert(kAbRowBegin == 16'384U && kAbRowEnd == 16'480U);
  static_assert(kQkvzRowEnd <= kAbRowBegin);
  static_assert(kAbRowEnd <= kernels::kSm87MacroFeedV4Fp8ScratchRowStride);

  const auto& norm = layer_norm_catalog_[0U];
  const auto& ab = bf16_ab_catalog_[0U];
  const auto& gdn = gdn_layer0_source_;
  const auto ping_range = kernels::sm87_macrofeed_v4_norm_residual_byte_range(
      ping_, kernels::kSm87MacroFeedV4NormResidualHiddenBytes);
  const auto pong_range = kernels::sm87_macrofeed_v4_norm_residual_byte_range(
      pong_, kernels::kSm87MacroFeedV4NormResidualHiddenBytes);
  const auto scratch_range = kernels::sm87_macrofeed_v4_bf16_ab_byte_range(
      scratch_, kernels::kSm87MacroFeedV4Bf16AbScratchBytes);
  return norm.model_layer == 0U && norm.input_layernorm != nullptr &&
         norm.input_layernorm_identity != 0U && norm.pair_identity != 0U &&
         norm.epsilon_fp32_bits ==
             kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits &&
         ab.model_layer == 0U && ab.a_weights != nullptr &&
         ab.b_weights != nullptr && ab.pair_identity != 0U &&
         kernels::sm87_target_aot_fp8_cuda_asset_valid(gdn.asset) &&
         gdn.asset.payload.role ==
             kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ &&
         kernels::sm87_macrofeed_v4_fp8_resource_gate(gdn.resources) &&
         gdn.resources.role ==
             kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ &&
         gdn.resources.input_layout ==
             kernels::Sm87MacroFeedV4Fp8InputLayout::
                 kHiddenContiguousH5120V1 &&
         gdn.resources.identity ==
             kernels::Sm87MacroFeedV4Fp8Identity::
                 kGdnQkvZM64N128K64OrdinaryGridV1 &&
         gdn.resources.device_ordinal == audit_.device_ordinal &&
         gdn.synthetic_t1 == audit_.synthetic_t1_gdn_layer0_source &&
         pointer_aligned(ping_,
                         kernels::kSm87MacroFeedV4NormResidualPointerAlignment) &&
         pointer_aligned(pong_,
                         kernels::kSm87MacroFeedV4NormResidualPointerAlignment) &&
         pointer_aligned(scratch_,
                         kernels::kSm87MacroFeedV4Bf16AbPointerAlignment) &&
         ping_range.valid && pong_range.valid && scratch_range.valid &&
         kernels::sm87_macrofeed_v4_norm_residual_ranges_disjoint(
             ping_range, pong_range) &&
         ping_range.end <= scratch_range.begin &&
         pong_range.end <= scratch_range.begin;
}

bool Sm87MacroFeedV4P40ExecutionPackage::complete_layer_bindings_valid()
    const noexcept {
  if (!complete_gdn_layer0_source_.has_value() ||
      ping_ == nullptr || pong_ == nullptr || scratch_ == nullptr ||
      recurrent_allocation_ == nullptr || bf16_ab_catalog_.empty() ||
      layer_norm_catalog_.empty()) {
    return false;
  }
  const auto& source = *complete_gdn_layer0_source_;
  const auto& gdn = source.gdn_layer;
  const auto& output = gdn.gdn_output;
  const auto& continuation = gdn.continuation;
  const auto& mlp = source.mlp_pair;
  const auto& gate = mlp.gate_up;
  const auto& down = mlp.down;
  const bool production_binding_identities =
      gdn.binding_identity != 0U &&
      gdn.binding_identity ==
          StartupPackage::compute_gdn_qkvz_execution_binding_identity(gdn) &&
      mlp.binding_identity != 0U &&
      mlp.binding_identity ==
          StartupPackage::compute_mlp_pair_execution_binding_identity(mlp);
  const bool synthetic_launch_only_identities =
      gdn.binding_identity == 0U && gdn.package_identity == 0U &&
      gdn.deployment_plan_identity == 0U && gdn.owner_identity == 0U &&
      gdn.allocation_identity == 0U &&
      gdn.projection_catalog_identity == 0U && gdn.device_identity == 0U &&
      gdn.resource_seal_identity == 0U &&
      gdn.projection_binding_identity == 0U &&
      gdn.asset_value_identity == 0U &&
      output.binding_identity == 0U &&
      output.projection_binding_identity == 0U &&
      output.asset_value_identity == 0U &&
      continuation.conv_weight_identity == 0U &&
      continuation.a_log_identity == 0U &&
      continuation.dt_bias_identity == 0U &&
      continuation.norm_weight_identity == 0U &&
      continuation.aggregate_identity == 0U && mlp.binding_identity == 0U &&
      mlp.package_identity == 0U && mlp.deployment_plan_identity == 0U &&
      mlp.owner_identity == 0U && mlp.allocation_identity == 0U &&
      mlp.projection_catalog_identity == 0U && mlp.device_identity == 0U &&
      gate.projection_binding_identity == 0U && gate.asset_value_identity == 0U &&
      down.projection_binding_identity == 0U && down.asset_value_identity == 0U;
  const bool source_authority_matches =
      source.synthetic_t1 ? synthetic_launch_only_identities
                          : production_binding_identities;
  return source.identity != 0U &&
         source.identity == audit_.gdn_layer0_source_identity &&
         source.identity == compute_complete_layer0_source_identity(source) &&
         source.synthetic_t1 ==
             audit_.synthetic_t1_gdn_layer0_source &&
         source_authority_matches &&
         gdn.gdn_ordinal == 0U && gdn.model_layer == 0U &&
         gdn.role == kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ &&
         gdn.input_layout ==
             kernels::Sm87MacroFeedV4Fp8InputLayout::
                 kHiddenContiguousH5120V1 &&
         gdn.tactic_identity ==
             kernels::Sm87MacroFeedV4Fp8Identity::
                 kGdnQkvZM64N128K64OrdinaryGridV1 &&
         kernels::sm87_target_aot_fp8_cuda_asset_valid(gdn.asset) &&
         kernels::sm87_macrofeed_v4_fp8_resource_gate(gdn.resources) &&
         gdn.resources.role ==
             kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ &&
         gdn.resources.input_layout ==
             kernels::Sm87MacroFeedV4Fp8InputLayout::
                 kHiddenContiguousH5120V1 &&
         gdn.resources.identity ==
             kernels::Sm87MacroFeedV4Fp8Identity::
                 kGdnQkvZM64N128K64OrdinaryGridV1 &&
         gdn.resources.device_ordinal == audit_.device_ordinal &&
         gdn.live_cuda_payload_range_validated &&
         output.role ==
             kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput &&
         output.input_layout ==
             kernels::Sm87MacroFeedV4Fp8InputLayout::
                 kGdnContiguousVScratchV1 &&
         output.tactic_identity ==
             kernels::Sm87MacroFeedV4Fp8Identity::
                 kGdnAttentionOutputM64N128K64OrdinaryGridV1 &&
         kernels::sm87_target_aot_fp8_cuda_asset_valid(output.asset) &&
         kernels::sm87_macrofeed_v4_fp8_resource_gate(output.resources) &&
         output.resources.role ==
             kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput &&
         output.resources.input_layout ==
             kernels::Sm87MacroFeedV4Fp8InputLayout::
                 kGdnContiguousVScratchV1 &&
         output.resources.identity ==
             kernels::Sm87MacroFeedV4Fp8Identity::
                 kGdnAttentionOutputM64N128K64OrdinaryGridV1 &&
         output.resources.device_ordinal == audit_.device_ordinal &&
         output.live_cuda_payload_range_validated &&
         continuation.conv_weight != nullptr &&
         continuation.a_log != nullptr && continuation.dt_bias != nullptr &&
         continuation.norm_weight != nullptr && continuation.exact_shapes &&
         continuation.live_cuda_weight_ranges_validated &&
         mlp.model_layer == 0U &&
         kernels::sm87_target_aot_nvfp4_cuda_asset_valid(gate.asset) &&
         gate.asset.payload.role ==
             kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp &&
         kernels::sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
             source.gate_up_receipt) &&
         source.gate_up_receipt.payload_identity ==
             gate.asset.artifact_identity &&
         source.gate_up_receipt.receipt_identity ==
             gate.payload_receipt.receipt_identity &&
         source.gate_up_receipt.payload_begin == gate.asset.payload.begin &&
         source.gate_up_receipt.payload_end == gate.asset.payload.end &&
         kernels::sm87_target_aot_nvfp4_cuda_asset_valid(down.asset) &&
         down.asset.payload.role ==
             kernels::Sm87TargetAotProjectionRole::kNvFp4Down &&
         kernels::sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
             source.down_receipt) &&
         source.down_receipt.payload_identity ==
             down.asset.artifact_identity &&
         source.down_receipt.receipt_identity ==
             down.payload_receipt.receipt_identity &&
         source.down_receipt.payload_begin == down.asset.payload.begin &&
         source.down_receipt.payload_end == down.asset.payload.end &&
         mlp.live_cuda_payload_ranges_validated &&
         layer_norm_catalog_[0U].post_attention_layernorm != nullptr &&
         kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
             gdn_resources_) &&
         gdn_resources_.device_ordinal == audit_.device_ordinal &&
         kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(
             gate_up_resources_) &&
         gate_up_resources_.device_ordinal == audit_.device_ordinal &&
         kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(
             down_resources_) &&
         down_resources_.device_ordinal == audit_.device_ordinal;
}

bool Sm87MacroFeedV4P40ExecutionPackage::valid() const noexcept {
  if (!construction_postconditions_sealed_ ||
      transient_allocation_ == nullptr ||
      recurrent_allocation_ == nullptr || request_state_ == nullptr ||
      events_owner_ == nullptr || events_driver_ == nullptr ||
      !audit_.valid() ||
      events_driver_->owner_identity() !=
          audit_.execution_events_owner_identity ||
      events_driver_->device_ordinal() != audit_.device_ordinal ||
      !kernels::sm87_macrofeed_v4_norm_residual_resource_gate(
          norm_resources_) ||
      norm_resources_.device_ordinal != audit_.device_ordinal ||
      !kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
          bf16_ab_resources_) ||
      bf16_ab_resources_.device_ordinal != audit_.device_ordinal ||
      request_state_->admission().owner_identity !=
          audit_.execution_events_owner_identity ||
      request_state_->admission().allocation_identity !=
          audit_.recurrent_allocation_identity ||
      request_state_->admission().allocation_bytes !=
          kSm87MacroFeedV4RecurrentStorageBytes ||
      projection_catalog_.size() != audit_.projection_bindings ||
      bf16_ab_catalog_.size() != audit_.bf16_ab_pairs ||
      layer_norm_catalog_.size() != audit_.layer_norm_pairs ||
      complete_gdn_layer0_source_.has_value() !=
          audit_.fixed_gdn_layer0_complete_bound ||
      !front_half_bindings_valid()) {
    return false;
  }
  for (const auto& binding : projection_catalog_) {
    if (!binding.has_value() || binding->binding_identity() == 0U ||
        binding->package_identity() != audit_.startup_package_identity) {
      return false;
    }
  }
  if (audit_.synthetic_t1_gdn_layer0_source) {
    for (const auto& binding : gdn_qkvz_catalog_) {
      if (binding.binding_identity != 0U) {
        return false;
      }
    }
    for (const auto& binding : mlp_pair_catalog_) {
      if (binding.binding_identity != 0U) {
        return false;
      }
    }
    return !audit_.fixed_gdn_layer0_complete_bound ||
           complete_layer_bindings_valid();
  }
  if (compute_gdn_layer_catalog_fold_identity(gdn_qkvz_catalog_) !=
          audit_.retained_gdn_layer_catalog_fold_identity ||
      compute_mlp_pair_catalog_fold_identity(mlp_pair_catalog_) !=
          audit_.retained_mlp_pair_catalog_fold_identity) {
    return false;
  }
  for (std::size_t ordinal = 0U; ordinal < gdn_qkvz_catalog_.size();
       ++ordinal) {
    const auto& binding = gdn_qkvz_catalog_[ordinal];
    if (binding.gdn_ordinal != ordinal ||
        binding.model_layer != ordinal + ordinal / 3U ||
        binding.role !=
            kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ ||
        binding.input_layout !=
            kernels::Sm87MacroFeedV4Fp8InputLayout::
                kHiddenContiguousH5120V1 ||
        binding.tactic_identity !=
            kernels::Sm87MacroFeedV4Fp8Identity::
                kGdnQkvZM64N128K64OrdinaryGridV1 ||
        binding.package_identity != audit_.startup_package_identity ||
        binding.projection_catalog_identity !=
            audit_.projection_catalog_identity ||
        binding.binding_identity == 0U ||
        !binding.live_cuda_payload_range_validated ||
        binding.request_selectable || binding.launcher_authority ||
        binding.production_dispatch_eligible) {
      return false;
    }
  }
  const auto& first = gdn_qkvz_catalog_[0U];
  for (std::size_t layer = 0U; layer < mlp_pair_catalog_.size(); ++layer) {
    const auto& binding = mlp_pair_catalog_[layer];
    if (binding.model_layer != layer || binding.binding_identity == 0U ||
        binding.package_identity != audit_.startup_package_identity ||
        binding.projection_catalog_identity !=
            audit_.projection_catalog_identity ||
        !binding.live_cuda_payload_ranges_validated ||
        binding.request_selectable || binding.launcher_authority ||
        binding.production_dispatch_eligible ||
        !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(
            binding.gate_up.asset) ||
        !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(
            binding.down.asset) ||
        !kernels::sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
            binding.gate_up.payload_receipt) ||
        !kernels::sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
            binding.down.payload_receipt)) {
      return false;
    }
  }
  return gdn_layer0_source_.identity == first.binding_identity &&
         gdn_layer0_source_.asset.payload.begin == first.asset.payload.begin &&
         gdn_layer0_source_.asset.payload.end == first.asset.payload.end &&
         !gdn_layer0_source_.synthetic_t1 &&
         complete_layer_bindings_valid();
}

Sm87MacroFeedV4P40ExecutionPackageStatus
Sm87MacroFeedV4P40ExecutionPackage::abort_request_state() noexcept {
  if (request_state_ == nullptr) {
    return failure(Error::kRequestAbort, "missing_request_state");
  }
  const auto access = request_state_->issue_sealed_access();
  const auto status = request_state_->abort_unpublished_request(
      access, Sm87MacroFeedV4RequestDiscardReason::kFailed);
  return status ? ok()
                : failure(Error::kRequestAbort,
                          "request_state_abort_unpublished");
}

Sm87MacroFeedV4P40ExecutionPackageStatus
Sm87MacroFeedV4P40ExecutionPackage::terminalize_event_failure(
    const char* const context,
    const events::Sm87MacroFeedV4ExecutionStatus& event_status,
    const Sm87MacroFeedV4RequestStateSealedAccess*
        const request_state_access) noexcept {
  if (events_driver_ == nullptr) {
    return failure(Error::kPhysicalDrain,
                   "missing_event_driver_during_terminalization", 0,
                   kSm87MacroFeedV4LayerCount, false, event_status);
  }
  const auto snapshot = events_driver_->snapshot();
  if (snapshot.state !=
      events::Sm87MacroFeedV4ExecutionOwnerState::kPoisoned) {
    return event_failure(context, event_status);
  }

  if (request_state_access != nullptr && request_state_ == nullptr) {
    return failure(Error::kPhysicalDrain,
                   "missing_request_state_during_terminalization", 0,
                   kSm87MacroFeedV4LayerCount, false, event_status);
  }
  const auto poison = request_state_access == nullptr
                          ? events_driver_->drain_poisoned_request()
                          : events_driver_->drain_poisoned_request_and_discard(
                                *request_state_, *request_state_access,
                                Sm87MacroFeedV4RequestDiscardReason::kFailed);
  const auto& original =
      event_status.error != events::Sm87MacroFeedV4ExecutionError::kNone
          ? event_status
          : poison.poison_cause;
  if (!poison.physical_quiescence_attested) {
    return failure(Error::kPhysicalDrain,
                   "poisoned_request_terminal_drain_failed",
                   poison.drain_status.cuda_error,
                   poison.drain_status.panel, false, original);
  }
  if (request_state_access != nullptr &&
      !poison.request_state_discarded) {
    return failure(Error::kPhysicalDrain,
                   "poisoned_request_state_discard_failed",
                   poison.drain_status.cuda_error,
                   poison.drain_status.panel, false,
                   poison.drain_status);
  }
  return failure(Error::kExecutionEvent, context, original.cuda_error,
                 original.panel, false, original);
}

Sm87MacroFeedV4P40ExecutionPackageStatus
Sm87MacroFeedV4P40ExecutionPackage::drain_and_discard_active_panel(
    const PanelAccess& panel_access,
    std::uint64_t* const owner_drain_receipt_identity,
    const Sm87MacroFeedV4RequestStateSealedAccess*
        const request_state_access) noexcept {
  if (owner_drain_receipt_identity != nullptr) {
    *owner_drain_receipt_identity = 0U;
  }
  if (events_owner_ == nullptr || events_driver_ == nullptr) {
    return failure(Error::kPhysicalDrain, "missing_execution_event_owner");
  }
  auto snapshot = events_driver_->snapshot();
  if (snapshot.state == events::Sm87MacroFeedV4ExecutionOwnerState::kPoisoned) {
    return terminalize_event_failure(
        "poisoned_request_before_tail_drain", snapshot.poison_cause,
        request_state_access);
  }

  auto enqueue = events_driver_->record_event(
      panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kMainTail);
  if (!enqueue) {
    return terminalize_event_failure("main_tail_record", enqueue.status,
                                     request_state_access);
  }
  enqueue = events_driver_->record_event(
      panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kAbAux,
      events::Sm87MacroFeedV4ExecutionEvent::kAbTail);
  if (!enqueue) {
    return terminalize_event_failure("ab_tail_record", enqueue.status,
                                     request_state_access);
  }
  enqueue = events_driver_->wait_event(
      panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kMainTail);
  if (!enqueue) {
    return terminalize_event_failure("main_tail_control_join",
                                     enqueue.status, request_state_access);
  }
  enqueue = events_driver_->wait_event(
      panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kAbTail);
  if (!enqueue) {
    return terminalize_event_failure("ab_tail_control_join",
                                     enqueue.status, request_state_access);
  }
  enqueue = events_driver_->record_event(
      panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kOwnerDrained);
  if (!enqueue) {
    return terminalize_event_failure("owner_drained_record",
                                     enqueue.status, request_state_access);
  }
  const auto observed = events_driver_->observe_event_synchronize(
      panel_access,
      events::Sm87MacroFeedV4ExecutionEvent::kOwnerDrained);
  if (!observed ||
      !events_driver_->completion_receipt_matches(
          panel_access,
          events::Sm87MacroFeedV4ExecutionEvent::kOwnerDrained,
          observed.receipt)) {
    return terminalize_event_failure(
        "owner_drained_physical_observation", observed.status,
        request_state_access);
  }
  const auto discarded = request_state_access == nullptr
                             ? events_driver_->discard_after_drain(
                                   panel_access, observed.receipt)
                             : events_driver_->discard_request_state_after_drain(
                                   panel_access, observed.receipt,
                                   *request_state_, *request_state_access,
                                   Sm87MacroFeedV4RequestDiscardReason::kFailed);
  if (!discarded) {
    return terminalize_event_failure("owner_discard_after_drain",
                                     discarded, request_state_access);
  }
  if (owner_drain_receipt_identity != nullptr) {
    *owner_drain_receipt_identity = observed.receipt.receipt_identity();
  }
  return ok();
}

Sm87MacroFeedV4GdnLayer0FrontHalfResult
Sm87MacroFeedV4P40ExecutionPackage::execute_gdn_layer0_front_half_once()
    noexcept {
  Sm87MacroFeedV4GdnLayer0FrontHalfResult result;
  if (execution_attempted_) {
    result.status = failure(Error::kAlreadyExecuted,
                            "front_half_is_one_shot");
    return result;
  }
  execution_attempted_ = true;
  // The complete 256-entry projection and CUDA allocation validation ran once
  // in create().  Request execution deliberately does not call valid() or
  // rescan any construction-sealed catalog.
  if (!construction_postconditions_sealed_ ||
      !front_half_bindings_valid()) {
    result.status = failure(Error::kFrontHalfBinding,
                            "sealed_front_half_binding_invalid");
    return result;
  }

  const auto request_access = request_state_->issue_sealed_access();
  auto event_status = events_driver_->begin_request(
      *request_state_, request_access);
  if (!event_status) {
    result.status = event_failure("execution_begin_request", event_status);
    (void)abort_request_state();
    return result;
  }
  auto panel = events_driver_->begin_panel(0U);
  if (!panel) {
    result.status = event_failure("execution_begin_panel", panel.status);
    (void)abort_request_state();
    return result;
  }

  kernels::Sm87MacroFeedV4InputNormArguments norm_arguments;
  norm_arguments.input_hidden = ping_;
  norm_arguments.centered_weight =
      layer_norm_catalog_[0U].input_layernorm;
  norm_arguments.output_hidden = pong_;
  norm_arguments.token_count =
      kernels::kSm87MacroFeedV4NormResidualTokens;
  norm_arguments.hidden_size =
      kernels::kSm87MacroFeedV4NormResidualHidden;
  norm_arguments.epsilon_fp32_bits =
      layer_norm_catalog_[0U].epsilon_fp32_bits;
  norm_arguments.cuda_stream = nullptr;

  auto enqueue = events_driver_->submit_input_norm_and_record_ready(
      *panel.panel_access, norm_arguments, norm_resources_);
  if (!enqueue) {
    result.status = event_failure("input_norm_submit_and_record", enqueue.status);
    const auto drain = drain_and_discard_active_panel(*panel.panel_access);
    if (drain.error == Error::kPhysicalDrain) {
      result.status = drain;
    }
    (void)abort_request_state();
    return result;
  }
  enqueue = events_driver_->wait_event(
      *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kAbAux,
      events::Sm87MacroFeedV4ExecutionEvent::kNormReady);
  if (!enqueue) {
    result.status = event_failure("norm_ready_ab_wait", enqueue.status);
    const auto drain = drain_and_discard_active_panel(*panel.panel_access);
    if (drain.error == Error::kPhysicalDrain) {
      result.status = drain;
    }
    (void)abort_request_state();
    return result;
  }

  kernels::Sm87MacroFeedV4Bf16AbArguments ab_arguments;
  ab_arguments.a_weights = bf16_ab_catalog_[0U].a_weights;
  ab_arguments.b_weights = bf16_ab_catalog_[0U].b_weights;
  ab_arguments.input = pong_;
  ab_arguments.scratch = scratch_;
  ab_arguments.token_count = kernels::kSm87MacroFeedV4Bf16AbTokens;
  ab_arguments.scratch_row_stride =
      kernels::kSm87MacroFeedV4Bf16AbScratchRowStride;
  ab_arguments.cuda_stream = nullptr;
  enqueue = events_driver_->submit_bf16_ab_and_record_ready(
      *panel.panel_access, ab_arguments,
      bf16_ab_resources_);
  if (!enqueue) {
    result.status = event_failure("bf16_ab_submit_and_record", enqueue.status);
    const auto drain = drain_and_discard_active_panel(*panel.panel_access);
    if (drain.error == Error::kPhysicalDrain) {
      result.status = drain;
    }
    (void)abort_request_state();
    return result;
  }

  // Main already contains InputNorm.  This owner-locked transaction appends
  // exactly one fixed QKVZ body followed immediately by Main<-AbReady.  AbAux
  // has independently queued A/B after NormReady, so the two projections may
  // overlap physically without exposing either stream or permitting a caller
  // to interleave work between QKVZ and the wait.
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4GdnQkvzC8000Arguments gdn_arguments;
  gdn_arguments.hidden_input = pong_;
  gdn_arguments.asset = gdn_layer0_source_.asset;
  gdn_arguments.phase_scratch = scratch_;
  enqueue = events_driver_->submit_gdn_qkvz_c8000_then_wait_ab_ready(
      *panel.panel_access, gdn_arguments, gdn_layer0_source_.resources);
  if (!enqueue) {
    result.status = event_failure(
        "gdn_qkvz_submit_then_ab_ready_wait", enqueue.status);
    const auto drain = drain_and_discard_active_panel(*panel.panel_access);
    if (drain.error == Error::kPhysicalDrain) {
      result.status = drain;
    }
    (void)abort_request_state();
    return result;
  }

  result.status = drain_and_discard_active_panel(*panel.panel_access);
  if (!result.status) {
    (void)abort_request_state();
    return result;
  }
  result.status = abort_request_state();
  if (!result.status) {
    return result;
  }

  const auto event_snapshot = events_driver_->snapshot();
  const auto request_snapshot = request_state_->snapshot();
  Sm87MacroFeedV4GdnLayer0FrontHalfReceipt receipt;
  receipt.receipt_identity =
      next_nonzero(&g_next_front_half_receipt_identity);
  receipt.package_identity = audit_.package_identity;
  receipt.gdn_layer0_source_identity = audit_.gdn_layer0_source_identity;
  receipt.gdn_qkvz_catalog_identity = audit_.gdn_qkvz_catalog_identity;
  receipt.request_epoch = request_access.request_epoch();
  receipt.panel = 0U;
  receipt.model_layer = 0U;
  receipt.input_norm_launches = event_snapshot.input_norm_submissions;
  receipt.gdn_qkvz_launches = event_snapshot.gdn_qkvz_c8000_submissions;
  receipt.bf16_ab_launches = event_snapshot.bf16_ab_submissions;
  receipt.bound_kernel_submissions =
      event_snapshot.bound_kernel_submissions;
  receipt.physical_completion_receipts =
      event_snapshot.physical_completion_receipts_issued;
  receipt.norm_ready_recorded =
      event_snapshot.event_generations[static_cast<std::size_t>(
          events::Sm87MacroFeedV4ExecutionEvent::kNormReady)] == 1U;
  receipt.norm_ready_waited_by_ab = true;
  receipt.ab_ready_recorded =
      event_snapshot.event_generations[static_cast<std::size_t>(
          events::Sm87MacroFeedV4ExecutionEvent::kAbReady)] == 1U;
  receipt.ab_ready_waited_by_main =
      event_snapshot.gdn_qkvz_ab_ready_wait_transactions == 1U;
  receipt.owner_drained_physically =
      event_snapshot.state ==
          events::Sm87MacroFeedV4ExecutionOwnerState::kRequestDiscarded &&
      event_snapshot.owner_drained_recorded;
  receipt.request_discarded_without_publication =
      request_snapshot.phase == Sm87MacroFeedV4RequestStatePhase::kFailed &&
      !request_snapshot.canonical_state_published &&
      !request_snapshot.logical_sequence_fence_published &&
      !request_snapshot.decode_access_issued;
  receipt.gdn_layer0_front_half_only = true;
  receipt.synthetic_t1_gdn_layer0_source =
      audit_.synthetic_t1_gdn_layer0_source;
  receipt.layer_complete = false;
  receipt.panel_complete = false;
  receipt.model_complete = false;
  receipt.production_dispatch_eligible = false;
  result.receipt = receipt;
  if (!result.receipt.valid()) {
    result.status = failure(Error::kPhysicalDrain,
                            "front_half_receipt_postcondition");
  }
  return result;
}

Sm87MacroFeedV4GdnLayer0CompleteResult
Sm87MacroFeedV4P40ExecutionPackage::execute_gdn_layer0_complete_once()
    noexcept {
  Sm87MacroFeedV4GdnLayer0CompleteResult result;
  if (execution_attempted_) {
    result.status = failure(Error::kAlreadyExecuted,
                            "complete_gdn_layer0_is_one_shot");
    return result;
  }
  execution_attempted_ = true;
  // All expensive catalog/range/resource checks ran during create().  The
  // request path consumes only the immutable construction seal.
  if (!construction_postconditions_sealed_ ||
      !audit_.fixed_gdn_layer0_complete_bound ||
      !complete_gdn_layer0_source_.has_value()) {
    result.status = failure(Error::kCompleteLayerBinding,
                            "sealed_complete_gdn_layer0_binding_required");
    return result;
  }

  const auto request_access = request_state_->issue_sealed_access();
  auto event_status = events_driver_->begin_request(
      *request_state_, request_access);
  if (!event_status) {
    result.status = event_failure("complete_layer_begin_request",
                                  event_status);
    (void)abort_request_state();
    return result;
  }
  auto panel = events_driver_->begin_panel(0U);
  if (!panel) {
    result.status = event_failure("complete_layer_event_begin_panel",
                                  panel.status);
    (void)abort_request_state();
    return result;
  }
  const auto request_panel_status =
      request_state_->begin_panel(request_access, 0U);
  if (!request_panel_status) {
    result.status = failure(Error::kGdnLayerStateGrant,
                            "complete_layer_request_state_begin_panel", 0,
                            request_panel_status.layer);
    const auto drain =
        drain_and_discard_active_panel(*panel.panel_access);
    if (!drain) {
      result.status = drain;
    }
    (void)abort_request_state();
    return result;
  }
  auto authorized = request_state_->authorize_gdn_layer_state(
      request_access, 0U, 0U);
  if (!authorized) {
    result.status = failure(Error::kGdnLayerStateGrant,
                            "complete_layer_state_authorization", 0,
                            authorized.status.layer);
    const auto drain = drain_and_discard_active_panel(
        *panel.panel_access, nullptr, &request_access);
    if (!drain) {
      result.status = drain;
    }
    return result;
  }

  auto grant = std::move(*authorized.grant);
  const std::uint64_t grant_identity = grant.grant_identity();
  const std::uint64_t state_epoch_before = grant.state_epoch();
  const std::size_t active_bank_before = grant.active_bank_index();
  const std::size_t candidate_bank_before = grant.candidate_bank_index();
  const auto active_conv_offset = grant.active_conv_allocation_offset();
  const auto candidate_conv_offset = grant.candidate_conv_allocation_offset();
  const auto active_state_offset =
      grant.active_gdn_state_allocation_offset();
  const auto candidate_state_offset =
      grant.candidate_gdn_state_allocation_offset();
  const auto exact_range = [](const std::uint64_t offset,
                              const std::uint64_t bytes) noexcept {
    return offset <= kSm87MacroFeedV4RecurrentStorageBytes &&
           bytes <= kSm87MacroFeedV4RecurrentStorageBytes - offset;
  };
  if (grant.model_layer() != 0U || grant.state_layer_ordinal() != 0U ||
      grant.conv_bytes() != kernels::kSm87MacroFeedV4GdnConvHistoryBytes ||
      grant.gdn_state_bytes() != kernels::kSm87MacroFeedV4GdnStateBytes ||
      !exact_range(active_conv_offset, grant.conv_bytes()) ||
      !exact_range(candidate_conv_offset, grant.conv_bytes()) ||
      !exact_range(active_state_offset, grant.gdn_state_bytes()) ||
      !exact_range(candidate_state_offset, grant.gdn_state_bytes())) {
    result.status = failure(Error::kGdnLayerStateGrant,
                            "complete_layer_state_grant_offsets", 0, 0U);
    const auto drain = drain_and_discard_active_panel(
        *panel.panel_access, nullptr, &request_access);
    if (!drain) {
      result.status = drain;
    }
    return result;
  }

  const auto& source = *complete_gdn_layer0_source_;
  const auto& gdn = source.gdn_layer;
  const auto& mlp = source.mlp_pair;
  const auto* const recurrent =
      static_cast<const std::uint8_t*>(recurrent_allocation_);
  auto* const recurrent_mutable =
      static_cast<std::uint8_t*>(recurrent_allocation_);
  events::Sm87MacroFeedV4CompleteGdnLayerC8000Submission submission;
  submission.input_norm.input_hidden = ping_;
  submission.input_norm.centered_weight =
      layer_norm_catalog_[0U].input_layernorm;
  submission.input_norm.output_hidden = pong_;
  submission.input_norm.token_count =
      kernels::kSm87MacroFeedV4NormResidualTokens;
  submission.input_norm.hidden_size =
      kernels::kSm87MacroFeedV4NormResidualHidden;
  submission.input_norm.epsilon_fp32_bits =
      layer_norm_catalog_[0U].epsilon_fp32_bits;
  submission.bf16_ab.a_weights = bf16_ab_catalog_[0U].a_weights;
  submission.bf16_ab.b_weights = bf16_ab_catalog_[0U].b_weights;
  submission.bf16_ab.input = pong_;
  submission.bf16_ab.scratch = scratch_;
  submission.bf16_ab.token_count = kernels::kSm87MacroFeedV4Bf16AbTokens;
  submission.bf16_ab.scratch_row_stride =
      kernels::kSm87MacroFeedV4Bf16AbScratchRowStride;
  submission.gdn_qkvz.hidden_input = pong_;
  submission.gdn_qkvz.asset = gdn.asset;
  submission.gdn_qkvz.phase_scratch = scratch_;
  submission.gdn_continuation.phase_scratch = scratch_;
  submission.gdn_continuation.conv_weight = gdn.continuation.conv_weight;
  submission.gdn_continuation.a_log = gdn.continuation.a_log;
  submission.gdn_continuation.dt_bias = gdn.continuation.dt_bias;
  submission.gdn_continuation.norm_weight = gdn.continuation.norm_weight;
  submission.gdn_continuation.active_conv_history =
      reinterpret_cast<const std::uint16_t*>(recurrent + active_conv_offset);
  submission.gdn_continuation.candidate_conv_history =
      reinterpret_cast<std::uint16_t*>(recurrent_mutable +
                                       candidate_conv_offset);
  submission.gdn_continuation.active_recurrent_state =
      reinterpret_cast<const std::uint16_t*>(recurrent + active_state_offset);
  submission.gdn_continuation.candidate_recurrent_state =
      reinterpret_cast<std::uint16_t*>(recurrent_mutable +
                                       candidate_state_offset);
  submission.gdn_continuation.cancellation_signal = nullptr;
  submission.gdn_continuation.l2_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  submission.gdn_continuation.norm_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  submission.gdn_output.phase_scratch = scratch_;
  submission.gdn_output.asset = gdn.gdn_output.asset;
  submission.gdn_output.branch_output = pong_;
  submission.residual_post_norm.left_residual_then_normalized = ping_;
  submission.residual_post_norm.right_branch_then_residual = pong_;
  submission.residual_post_norm.centered_weight =
      layer_norm_catalog_[0U].post_attention_layernorm;
  submission.gate_up.normalized_input = ping_;
  submission.gate_up.payload = reinterpret_cast<const std::uint8_t*>(
      mlp.gate_up.asset.payload.begin);
  submission.gate_up.payload_bytes = mlp.gate_up.asset.payload.bytes;
  submission.gate_up.gate_tensor_scale =
      fp32_from_bits(mlp.gate_up.asset.tensor_scale_bits[0U]);
  submission.gate_up.up_tensor_scale =
      fp32_from_bits(mlp.gate_up.asset.tensor_scale_bits[1U]);
  submission.gate_up.intermediate_output = scratch_;
  submission.gate_up.canonical_v3_payload_receipt =
      source.gate_up_receipt;
  submission.down.intermediate_input = scratch_;
  submission.down.payload = reinterpret_cast<const std::uint8_t*>(
      mlp.down.asset.payload.begin);
  submission.down.payload_bytes = mlp.down.asset.payload.bytes;
  submission.down.tensor_scale =
      fp32_from_bits(mlp.down.asset.tensor_scale_bits[0U]);
  submission.down.residual_output = pong_;
  submission.down.payload_receipt = source.down_receipt;
  submission.norm_resources = norm_resources_;
  submission.bf16_ab_resources = bf16_ab_resources_;
  submission.gdn_qkvz_resources = gdn.resources;
  submission.gdn_continuation_resources = gdn_resources_;
  submission.gdn_output_resources = gdn.gdn_output.resources;
  submission.gate_up_resources = gate_up_resources_;
  submission.down_resources = down_resources_;

  const auto enqueued =
      events_driver_->submit_complete_gdn_layer_c8000_prevalidated(
          *panel.panel_access, submission);
  if (!enqueued) {
    result.status = event_failure("complete_gdn_layer_enqueue",
                                  enqueued.status);
    const auto drain = drain_and_discard_active_panel(
        *panel.panel_access, nullptr, &request_access);
    if (!drain) {
      result.status = drain;
    }
    return result;
  }

  const auto committed = request_state_->commit_gdn_layer_candidate_enqueued(
      request_access, std::move(grant));
  if (!committed) {
    result.status = failure(Error::kGdnLayerStateGrant,
                            "complete_layer_state_commit", 0,
                            committed.layer);
    const auto drain = drain_and_discard_active_panel(
        *panel.panel_access, nullptr, &request_access);
    if (!drain) {
      result.status = drain;
    }
    return result;
  }
  const auto committed_snapshot = request_state_->snapshot();
  std::uint64_t physical_owner_drain_receipt_identity = 0U;
  result.status = drain_and_discard_active_panel(
      *panel.panel_access, &physical_owner_drain_receipt_identity,
      &request_access);
  if (!result.status) {
    return result;
  }

  const auto event_snapshot = events_driver_->snapshot();
  const auto request_snapshot = request_state_->snapshot();
  Sm87MacroFeedV4GdnLayer0CompleteReceipt receipt;
  receipt.receipt_identity =
      next_nonzero(&g_next_complete_layer_receipt_identity);
  receipt.package_identity = audit_.package_identity;
  receipt.gdn_layer0_source_identity = audit_.gdn_layer0_source_identity;
  receipt.gdn_qkvz_catalog_identity = audit_.gdn_qkvz_catalog_identity;
  receipt.mlp_pair_catalog_identity = audit_.mlp_pair_catalog_identity;
  receipt.request_epoch = request_access.request_epoch();
  receipt.state_epoch_before = state_epoch_before;
  receipt.state_epoch_after = committed_snapshot.state_epoch;
  receipt.state_grant_identity = grant_identity;
  receipt.panel = 0U;
  receipt.model_layer = 0U;
  receipt.active_bank_before = active_bank_before;
  receipt.active_bank_after = committed_snapshot.active_bank_index;
  receipt.candidate_bank_before = candidate_bank_before;
  receipt.candidate_bank_after = committed_snapshot.candidate_bank_index;
  receipt.input_norm_launches = enqueued.receipt.input_norm_launches;
  receipt.bf16_ab_launches = enqueued.receipt.bf16_ab_launches;
  receipt.gdn_qkvz_launches = enqueued.receipt.gdn_qkvz_launches;
  receipt.gdn_continuation_launches =
      enqueued.receipt.gdn_continuation_launches;
  receipt.gdn_output_launches = enqueued.receipt.gdn_output_launches;
  receipt.residual_post_norm_launches =
      enqueued.receipt.residual_post_norm_launches;
  receipt.gate_up_launches = enqueued.receipt.gate_up_launches;
  receipt.down_launches = enqueued.receipt.down_launches;
  receipt.bound_kernel_submissions =
      enqueued.receipt.bound_kernel_submissions;
  receipt.asynchronous_d2d_copies =
      enqueued.receipt.asynchronous_d2d_copies;
  receipt.conv_history_copy_bytes =
      enqueued.receipt.conv_history_copy_bytes;
  receipt.physical_owner_drain_receipt_identity =
      physical_owner_drain_receipt_identity;
  receipt.physical_completion_receipts =
      event_snapshot.physical_completion_receipts_issued;
  receipt.norm_ready_waited_by_ab =
      enqueued.receipt.norm_ready_waited_by_ab;
  receipt.ab_ready_waited_by_main =
      enqueued.receipt.ab_ready_waited_by_main;
  receipt.layer_complete = enqueued.receipt.complete_layer_enqueued;
  receipt.state_candidate_recorded =
      committed_snapshot.next_model_layer == 1U &&
      committed_snapshot.panel_conv_layers_prepared == 1U &&
      committed_snapshot.panel_gdn_layers_assigned == 1U &&
      committed_snapshot.panel_swap_count == 0U;
  receipt.owner_drained_physically =
      event_snapshot.state ==
          events::Sm87MacroFeedV4ExecutionOwnerState::kRequestDiscarded &&
      event_snapshot.owner_drained_recorded &&
      request_snapshot.physical_owner_drain_receipt_identity ==
          physical_owner_drain_receipt_identity;
  receipt.physical_execution_receipt_issued =
      request_snapshot.physical_execution_receipt_issued;
  receipt.candidate_discarded_without_publication =
      request_snapshot.phase == Sm87MacroFeedV4RequestStatePhase::kFailed &&
      request_snapshot.candidate_discard_count == 1U &&
      !request_snapshot.canonical_state_published &&
      !request_snapshot.logical_sequence_fence_published &&
      !request_snapshot.decode_access_issued;
  receipt.synthetic_t1_gdn_layer0_source =
      audit_.synthetic_t1_gdn_layer0_source;
  receipt.panel_complete = false;
  receipt.model_complete = false;
  receipt.production_dispatch_eligible = false;
  result.receipt = receipt;
  if (!result.receipt.valid()) {
    result.status = failure(Error::kPhysicalDrain,
                            "complete_gdn_layer0_receipt_postcondition");
  }
  return result;
}

void Sm87MacroFeedV4P40ExecutionPackage::release() noexcept {
  // Events own all in-flight device work and therefore die first.  Their
  // destructor synchronizes every private stream before request metadata or
  // either allocation is released.
  events_driver_.reset();
  events_owner_.reset();
  request_state_.reset();
  if (recurrent_allocation_ != nullptr) {
    (void)cudaFree(recurrent_allocation_);
    recurrent_allocation_ = nullptr;
  }
  if (transient_allocation_ != nullptr) {
    (void)cudaFree(transient_allocation_);
    transient_allocation_ = nullptr;
  }
  ping_ = nullptr;
  pong_ = nullptr;
  scratch_ = nullptr;
  for (auto& binding : projection_catalog_) {
    binding.reset();
  }
  bf16_ab_catalog_.fill({});
  layer_norm_catalog_.fill({});
  gdn_qkvz_catalog_.fill({});
  mlp_pair_catalog_.fill({});
  gdn_layer0_source_ = {};
  complete_gdn_layer0_source_.reset();
  norm_resources_ = {};
  bf16_ab_resources_ = {};
  gdn_resources_ = {};
  gate_up_resources_ = {};
  down_resources_ = {};
  audit_ = {};
  construction_postconditions_sealed_ = false;
}

}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail
