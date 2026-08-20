#include "../src/runtime/sm87_macrofeed_v4_p40_execution_package_internal.h"
#include "support/sm87_macrofeed_v4_live_fp8_asset_fixture.h"
#include "support/sm87_target_aot_complete_host_fixture.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail {

class Sm87MacroFeedV4P40StartupPackageHostTestFixture final {
 public:
  [[nodiscard]] static Sm87MacroFeedV4P40StartupPackageCreateResult create(
      const ModelWeights& model_weights) noexcept {
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
    auto access = target_aot_complete_execution_detail::
        Sm87TargetAotCompleteProjectionExecutionAccess::
            bind_complete_host_test_fixture(
                model_weights,
                static_cast<std::uintptr_t>(resident_arena_begin),
                resident_arena_bytes);
    if (!access) {
      Sm87MacroFeedV4P40StartupPackageCreateResult result;
      result.status.error =
          Sm87MacroFeedV4P40StartupPackageError::kProjectionAccessBind;
      result.status.context = "host_test_resident_access_bind";
      return result;
    }
    return Sm87MacroFeedV4P40StartupPackage::
        create_from_host_test_authority(model_weights, std::move(*access));
  }
};

}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail

namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail {

class Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture final {
 public:
  struct Samples final {
    std::uint16_t normalized = 0U;
    std::uint16_t gdn_qkvz = 0U;
    std::uint16_t projection_a = 0U;
    std::uint16_t projection_b = 0U;
    std::uint16_t scratch_gap = 0U;
  };

  struct RequestOutcome final {
    Sm87MacroFeedV4RequestStatePhase phase =
        Sm87MacroFeedV4RequestStatePhase::kInvalid;
    bool canonical_state_published = true;
    bool logical_sequence_fence_published = true;
    bool decode_access_issued = true;
  };

  [[nodiscard]] static Sm87MacroFeedV4GdnLayer0FrontHalfResult execute_once(
      Sm87MacroFeedV4P40ExecutionPackage& package) noexcept {
    return package.execute_gdn_layer0_front_half_once();
  }

  [[nodiscard]] static Sm87MacroFeedV4GdnLayer0CompleteResult
  execute_complete_once(
      Sm87MacroFeedV4P40ExecutionPackage& package) noexcept {
    return package.execute_gdn_layer0_complete_once();
  }

  [[nodiscard]] static Sm87MacroFeedV4P40ExecutionPackageCreateResult
  create(
      const Sm87MacroFeedV4P40ExecutionPackage::StartupPackage& startup)
      noexcept {
    // Negative-catalog fixture only: it fails before normal Full/RoPE owner
    // admission and therefore must not fabricate an Engine binding.
    return Sm87MacroFeedV4P40ExecutionPackage::create(startup, {}, 0U);
  }

  [[nodiscard]] static Sm87MacroFeedV4P40ExecutionPackageCreateResult
  create_with_synthetic_t1_gdn_layer0(
      const Sm87MacroFeedV4P40ExecutionPackage::StartupPackage& startup,
      const kernels::Sm87TargetAotFp8CudaAssetView& asset) noexcept {
    return Sm87MacroFeedV4P40ExecutionPackage::create_impl(
        startup, nullptr, 0U, &asset, nullptr);
  }

  [[nodiscard]] static Sm87MacroFeedV4P40ExecutionPackageCreateResult
  create_with_synthetic_t1_complete_gdn_layer0(
      const Sm87MacroFeedV4P40ExecutionPackage::StartupPackage& startup,
      const kernels::Sm87TargetAotFp8CudaAssetView& gdn_qkvz_asset,
      const kernels::Sm87TargetAotFp8CudaAssetView& gdn_output_asset,
      const kernels::Sm87TargetAotNvFp4CudaAssetView& gate_up_asset,
      const kernels::Sm87TargetAotNvFp4CudaAssetView& down_asset,
      const std::uint16_t* const conv_weight,
      const std::uint16_t* const a_log,
      const std::uint16_t* const dt_bias,
      const std::uint16_t* const norm_weight,
      const kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt&
          gate_up_receipt,
      const kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt&
          down_receipt) noexcept {
    Sm87MacroFeedV4P40ExecutionPackage::SyntheticCompleteGdnLayer0Source
        source;
    source.gdn_qkvz_asset = gdn_qkvz_asset;
    source.gdn_output_asset = gdn_output_asset;
    source.gate_up_asset = gate_up_asset;
    source.down_asset = down_asset;
    source.conv_weight = conv_weight;
    source.a_log = a_log;
    source.dt_bias = dt_bias;
    source.norm_weight = norm_weight;
    source.gate_up_receipt = gate_up_receipt;
    source.down_receipt = down_receipt;
    return Sm87MacroFeedV4P40ExecutionPackage::create_impl(
        startup, nullptr, 0U, nullptr, &source);
  }

  [[nodiscard]] static bool exercise_terminal_poison_drain(
      Sm87MacroFeedV4P40ExecutionPackage& package) noexcept {
    if (package.request_state_ == nullptr || package.events_owner_ == nullptr ||
        package.events_driver_ == nullptr) {
      return false;
    }
    const auto request_access = package.request_state_->issue_sealed_access();
    const auto begin_request = package.events_driver_->begin_request(
        *package.request_state_, request_access);
    if (!begin_request) {
      return false;
    }
    auto panel = package.events_driver_->begin_panel(0U);
    if (!panel) {
      return false;
    }
    const auto injected =
        sm87_macrofeed_v4_execution_events_detail::
            Sm87MacroFeedV4ExecutionEventsCudaTestFixture::
                inject_poison_without_drain(
                    *package.events_owner_,
                    sm87_macrofeed_v4_execution_events_detail::
                        Sm87MacroFeedV4ExecutionError::kCudaSubmission);
    if (injected.error !=
        sm87_macrofeed_v4_execution_events_detail::
            Sm87MacroFeedV4ExecutionError::kCudaSubmission) {
      return false;
    }
    const auto terminalized = package.terminalize_event_failure(
        "test_tail_record_cuda_failure", injected);
    const auto snapshot = package.events_driver_->snapshot();
    const auto aborted = package.abort_request_state();
    return terminalized.error ==
               Sm87MacroFeedV4P40ExecutionPackageError::kExecutionEvent &&
           terminalized.event_status.error ==
               sm87_macrofeed_v4_execution_events_detail::
                   Sm87MacroFeedV4ExecutionError::kCudaSubmission &&
           snapshot.state ==
               sm87_macrofeed_v4_execution_events_detail::
                   Sm87MacroFeedV4ExecutionOwnerState::kPoisoned &&
           snapshot.poison_drain_all_stream_synchronizations_attempted &&
           snapshot.poisoned_terminal_quiescence_attested && aborted;
  }

  [[nodiscard]] static bool
  exercise_pending_gdn_grant_poison_drain(
      Sm87MacroFeedV4P40ExecutionPackage& package) noexcept {
    if (package.request_state_ == nullptr || package.events_owner_ == nullptr ||
        package.events_driver_ == nullptr) {
      return false;
    }
    const auto request_access = package.request_state_->issue_sealed_access();
    const auto begin_request = package.events_driver_->begin_request(
        *package.request_state_, request_access);
    auto panel = package.events_driver_->begin_panel(0U);
    const auto begin_state_panel =
        package.request_state_->begin_panel(request_access, 0U);
    auto authorized = package.request_state_->authorize_gdn_layer_state(
        request_access, 0U, 0U);
    if (!begin_request || !panel || !begin_state_panel || !authorized) {
      return false;
    }
    const std::uint64_t grant_identity =
        authorized.grant->grant_identity();
    const auto before = package.request_state_->snapshot();
    const auto injected =
        sm87_macrofeed_v4_execution_events_detail::
            Sm87MacroFeedV4ExecutionEventsCudaTestFixture::
                inject_poison_without_drain(
                    *package.events_owner_,
                    sm87_macrofeed_v4_execution_events_detail::
                        Sm87MacroFeedV4ExecutionError::kCudaSubmission);
    const auto terminalized = package.terminalize_event_failure(
        "test_pending_grant_cuda_failure", injected, &request_access);
    const auto event_snapshot = package.events_driver_->snapshot();
    const auto after = package.request_state_->snapshot();
    const auto replay =
        package.request_state_->commit_gdn_layer_candidate_enqueued(
            request_access, std::move(*authorized.grant));
    return terminalized.error ==
               Sm87MacroFeedV4P40ExecutionPackageError::kExecutionEvent &&
           event_snapshot.state ==
               sm87_macrofeed_v4_execution_events_detail::
                   Sm87MacroFeedV4ExecutionOwnerState::kPoisoned &&
           event_snapshot.poisoned_terminal_quiescence_attested &&
           after.phase == Sm87MacroFeedV4RequestStatePhase::kFailed &&
           after.pending_gdn_layer_grant_identity == 0U &&
           after.last_invalidated_gdn_layer_grant_identity == grant_identity &&
           after.physical_execution_receipt_issued &&
           after.physical_owner_drain_was_poison_terminal &&
           after.candidate_discard_count == 1U &&
           after.active_bank_identity == before.active_bank_identity &&
           after.candidate_bank_identity == before.candidate_bank_identity &&
           after.state_epoch == before.state_epoch &&
           !after.canonical_state_published &&
           !after.logical_sequence_fence_published &&
           !after.decode_access_issued && !replay;
  }

  [[nodiscard]] static bool
  exercise_synthetic_full_attention_composer_fail_closed(
      Sm87MacroFeedV4P40ExecutionPackage& package) noexcept {
    if (package.request_state_ == nullptr || package.events_owner_ == nullptr ||
        package.events_driver_ == nullptr ||
        package.full_attention_composer_authority_sealed() ||
        !package.audit_.synthetic_t1_gdn_layer0_source ||
        package.audit_.full_attention_catalog_identity != 0U ||
        package.audit_.retained_full_attention_catalog_fold_identity != 0U ||
        package.audit_.full_attention_resource_bundle_identity != 0U ||
        package.audit_.full_attention_bindings != 0U ||
        package.audit_.kv_allocation_identity != 0U ||
        package.audit_.engine_rope_binding_identity != 0U) {
      return false;
    }
    const auto request_access = package.request_state_->issue_sealed_access();
    const auto begin_request = package.events_driver_->begin_request(
        *package.request_state_, request_access);
    auto panel = package.events_driver_->begin_panel(0U);
    const auto begin_state_panel =
        package.request_state_->begin_panel(request_access, 0U);
    if (!begin_request || !panel || !begin_state_panel) {
      return false;
    }
    const auto before = package.events_driver_->snapshot();
    const auto attempted =
        package.submit_complete_full_attention_layer_c8000(
            request_access, *panel.panel_access, 0U);
    const auto events_after = package.events_driver_->snapshot();
    const auto request_after = package.request_state_->snapshot();
    return !attempted && !attempted.receipt.valid() &&
           attempted.receipt.authority_domain ==
               sm87_macrofeed_v4_execution_events_detail::
                   Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain::
                       kInvalid &&
           attempted.receipt.package_identity == 0U &&
           attempted.receipt.full_attention_catalog_identity == 0U &&
           attempted.receipt.full_attention_binding_identity == 0U &&
           attempted.receipt.mlp_binding_identity == 0U &&
           attempted.receipt.input_norm_binding_identity == 0U &&
           attempted.receipt.post_norm_binding_identity == 0U &&
           attempted.receipt.rope_binding_identity == 0U &&
           attempted.receipt.resource_bundle_identity == 0U &&
           attempted.receipt.synthetic_source_identity == 0U &&
           attempted.receipt.enqueue_receipt.authority_domain() ==
               sm87_macrofeed_v4_execution_events_detail::
                   Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain::
                       kInvalid &&
           attempted.receipt.enqueue_receipt.execution_package_identity() ==
               0U &&
           attempted.receipt.enqueue_receipt
                   .full_attention_catalog_identity() == 0U &&
           attempted.receipt.enqueue_receipt
                   .full_attention_binding_identity() == 0U &&
           attempted.receipt.enqueue_receipt.mlp_binding_identity() == 0U &&
           attempted.receipt.enqueue_receipt
                   .input_norm_binding_identity() == 0U &&
           attempted.receipt.enqueue_receipt
                   .post_norm_binding_identity() == 0U &&
           attempted.receipt.enqueue_receipt.rope_binding_identity() == 0U &&
           attempted.receipt.enqueue_receipt.resource_bundle_identity() ==
               0U &&
           attempted.receipt.enqueue_receipt.synthetic_source_identity() ==
               0U &&
           attempted.status.error ==
               Sm87MacroFeedV4P40ExecutionPackageError::
                   kFullAttentionCatalog &&
           before.bound_kernel_submissions == 0U &&
           events_after.bound_kernel_submissions == 0U &&
           events_after.full_qkv_c8000_submissions == 0U &&
           events_after.full_attention_preprocess_c8000_submissions == 0U &&
           events_after.attention_c8000_submissions == 0U &&
           events_after.full_attention_output_c8000_submissions == 0U &&
           events_after.residual_post_norm_submissions == 0U &&
           events_after.gate_up_c8000_submissions == 0U &&
           events_after.down_c8000_submissions == 0U &&
           events_after.complete_full_attention_layers_submitted == 0U &&
           events_after.accepted_full_attention_grants == 0U &&
           !events_after.last_full_attention_accepted_prefix.valid_prefix() &&
           events_after.physical_completion_receipts_issued == 1U &&
           events_after.state ==
               sm87_macrofeed_v4_execution_events_detail::
                   Sm87MacroFeedV4ExecutionOwnerState::kRequestDiscarded &&
           events_after.owner_drained_recorded &&
           request_after.phase == Sm87MacroFeedV4RequestStatePhase::kFailed &&
           request_after.pending_full_attention_kv_grant_identity == 0U &&
           request_after.panel_kv_layers_staged == 0U &&
           request_after.next_model_layer == 0U &&
           request_after.candidate_discard_count == 1U &&
           request_after.physical_execution_receipt_issued &&
           !request_after.physical_owner_drain_was_poison_terminal &&
           !request_after.canonical_state_published &&
           !request_after.logical_sequence_fence_published &&
           !request_after.decode_access_issued;
  }

  [[nodiscard]] static bool
  exercise_synthetic_normal_gdn_composer_fail_closed(
      Sm87MacroFeedV4P40ExecutionPackage& package) noexcept {
    if (package.request_state_ == nullptr || package.events_owner_ == nullptr ||
        package.events_driver_ == nullptr ||
        package.gdn_composer_authority_sealed()) {
      return false;
    }
    const auto request_access = package.request_state_->issue_sealed_access();
    const auto begin_request = package.events_driver_->begin_request(
        *package.request_state_, request_access);
    auto panel = package.events_driver_->begin_panel(0U);
    const auto begin_state_panel =
        package.request_state_->begin_panel(request_access, 0U);
    if (!begin_request || !panel || !begin_state_panel) {
      return false;
    }
    const auto before = package.events_driver_->snapshot();
    const auto attempted = package.submit_complete_gdn_layer_c8000(
        request_access, *panel.panel_access, 0U);
    const auto events_after = package.events_driver_->snapshot();
    const auto request_after = package.request_state_->snapshot();
    return !attempted && !attempted.receipt.valid() &&
           attempted.status.error ==
               Sm87MacroFeedV4P40ExecutionPackageError::kGdnQkvZCatalog &&
           before.bound_kernel_submissions == 0U &&
           events_after.bound_kernel_submissions == 0U &&
           events_after.input_norm_submissions == 0U &&
           events_after.bf16_ab_submissions == 0U &&
           events_after.gdn_qkvz_c8000_submissions == 0U &&
           events_after.gdn_continuation_c8000_submissions == 0U &&
           events_after.gdn_history_d2d_copies == 0U &&
           events_after.gdn_history_d2d_bytes == 0U &&
           events_after.gdn_output_c8000_submissions == 0U &&
           events_after.residual_post_norm_submissions == 0U &&
           events_after.gate_up_c8000_submissions == 0U &&
           events_after.down_c8000_submissions == 0U &&
           events_after.complete_gdn_layers_submitted == 0U &&
           events_after.accepted_gdn_grants == 0U &&
           !events_after.last_gdn_accepted_prefix.valid_prefix() &&
           events_after.physical_completion_receipts_issued == 1U &&
           events_after.state ==
               sm87_macrofeed_v4_execution_events_detail::
                   Sm87MacroFeedV4ExecutionOwnerState::kRequestDiscarded &&
           events_after.owner_drained_recorded &&
           request_after.phase == Sm87MacroFeedV4RequestStatePhase::kFailed &&
           request_after.pending_gdn_layer_grant_identity == 0U &&
           request_after.panel_conv_layers_prepared == 0U &&
           request_after.panel_gdn_layers_assigned == 0U &&
           request_after.next_model_layer == 0U &&
           request_after.candidate_discard_count == 1U &&
           request_after.physical_execution_receipt_issued &&
           !request_after.physical_owner_drain_was_poison_terminal &&
           !request_after.canonical_state_published &&
           !request_after.logical_sequence_fence_published &&
           !request_after.decode_access_issued;
  }

  [[nodiscard]] static bool seed(
      Sm87MacroFeedV4P40ExecutionPackage& package) noexcept {
    if (package.ping_ == nullptr || package.pong_ == nullptr ||
        package.scratch_ == nullptr) {
      return false;
    }
    if (cudaMemset(package.ping_, 0x3f,
                   kernels::kSm87MacroFeedV4NormResidualHiddenBytes) !=
            cudaSuccess ||
        cudaMemset(package.pong_, 0,
                   kernels::kSm87MacroFeedV4NormResidualHiddenBytes) !=
            cudaSuccess ||
        cudaMemset(package.scratch_, 0x5a,
                   kernels::kSm87MacroFeedV4Bf16AbScratchBytes) !=
            cudaSuccess) {
      return false;
    }
    // The package streams are deliberately non-blocking.  Establish fixture
    // initialization physically before handing execution to those private
    // streams instead of relying on legacy-default-stream ordering.
    return cudaDeviceSynchronize() == cudaSuccess;
  }

  [[nodiscard]] static bool read_samples(
      const Sm87MacroFeedV4P40ExecutionPackage& package,
      Samples* const samples) noexcept {
    if (samples == nullptr || package.pong_ == nullptr ||
        package.scratch_ == nullptr) {
      return false;
    }
    *samples = {};
    return cudaMemcpy(&samples->normalized, package.pong_,
                      sizeof(samples->normalized),
                      cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(&samples->gdn_qkvz, package.scratch_,
                      sizeof(samples->gdn_qkvz),
                      cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(&samples->projection_a,
                      package.scratch_ +
                          kernels::kSm87MacroFeedV4Bf16AbAOffset,
                      sizeof(samples->projection_a),
                      cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(&samples->projection_b,
                      package.scratch_ +
                          kernels::kSm87MacroFeedV4Bf16AbBOffset,
                      sizeof(samples->projection_b),
                      cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(&samples->scratch_gap,
                      package.scratch_ + 16'480U,
                      sizeof(samples->scratch_gap),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  [[nodiscard]] static bool read_request_outcome(
      const Sm87MacroFeedV4P40ExecutionPackage& package,
      RequestOutcome* const outcome) noexcept {
    if (outcome == nullptr || package.request_state_ == nullptr) {
      return false;
    }
    const auto snapshot = package.request_state_->snapshot();
    outcome->phase = snapshot.phase;
    outcome->canonical_state_published =
        snapshot.canonical_state_published;
    outcome->logical_sequence_fence_published =
        snapshot.logical_sequence_fence_published;
    outcome->decode_access_issued = snapshot.decode_access_issued;
    return true;
  }

  [[nodiscard]] static bool request_boundary_is_explicitly_unbound(
      const Sm87MacroFeedV4P40ExecutionPackage& package) noexcept {
    if (package.request_boundary_catalog_identity_ != 0U ||
        package.request_boundary_host_staging_ != nullptr ||
        package.request_boundary_host_token_ids_ != nullptr ||
        package.request_boundary_host_result_ != nullptr ||
        package.request_boundary_device_token_ids_ != nullptr ||
        package.request_boundary_final_norm_ != nullptr ||
        package.request_boundary_logits_ != nullptr ||
        package.request_boundary_greedy_workspace_ != nullptr ||
        package.request_boundary_final_result_ != nullptr) {
      return false;
    }
    for (const auto& binding : package.request_boundary_catalog_) {
      if (binding.binding_identity != 0U ||
          binding.source_catalog_identity != 0U ||
          binding.resource_bundle_identity != 0U ||
          binding.normal_resident_authority ||
          binding.host_test_resident_authority) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] static bool scratch_gap_untouched(
      const Sm87MacroFeedV4P40ExecutionPackage& package) {
    if (package.scratch_ == nullptr) {
      return false;
    }
    constexpr std::size_t kGapBegin = 16'480U;
    constexpr std::size_t kGapWidth =
        kernels::kSm87MacroFeedV4Fp8ScratchRowStride - kGapBegin;
    constexpr std::size_t kRows = kernels::kSm87MacroFeedV4Fp8Tokens;
    static_assert(kGapWidth == 928U);
    std::vector<std::uint16_t> gap(kGapWidth * kRows, 0U);
    if (cudaMemcpy2D(
            gap.data(), kGapWidth * sizeof(std::uint16_t),
            package.scratch_ + kGapBegin,
            kernels::kSm87MacroFeedV4Fp8ScratchRowStride *
                sizeof(std::uint16_t),
            kGapWidth * sizeof(std::uint16_t), kRows,
            cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    for (const std::uint16_t value : gap) {
      if (value != 0x5a5aU) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail

namespace {

namespace execution =
    q3x::runtime::sm87_macrofeed_v4_p40_execution_detail;
namespace events =
    q3x::runtime::sm87_macrofeed_v4_execution_events_detail;
namespace bound_launch =
    q3x::kernels::sm87_macrofeed_v4_bound_launch_detail;
namespace startup =
    q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail;
namespace target_aot =
    q3x::runtime::target_aot_complete_execution_detail;
namespace kernels = q3x::kernels;

using Access = target_aot::Sm87TargetAotCompleteProjectionExecutionAccess;
using Bf16AbPair = target_aot::Sm87TargetAotCompleteHostTestBf16AbPair;
using LayerNormPair =
    target_aot::Sm87TargetAotCompleteHostTestLayerNormPair;
using FullQkNormPair =
    target_aot::Sm87TargetAotCompleteHostTestFullQkNormPair;
using RequestBoundary =
    target_aot::Sm87TargetAotCompleteHostTestRequestBoundary;
using Owner = q3x::runtime::Sm87TargetAotCompleteProjectionDeviceAssets;
using q3x::runtime::Bf16LinearWeight;
using q3x::runtime::Bf16VectorWeight;
using q3x::runtime::ModelWeights;
using q3x::runtime::NvFp4LinearWeight;

template <typename T, typename = void>
struct HasPublicExecutionCreate : std::false_type {};

template <typename T>
struct HasPublicExecutionCreate<
    T, std::void_t<decltype(T::create(
           std::declval<const typename T::StartupPackage&>(),
           std::declval<
               const execution::Sm87MacroFeedV4P40EngineRopeBinding&>(),
           std::declval<std::uint64_t>()))>>
    : std::true_type {};

static_assert(!HasPublicExecutionCreate<
              execution::Sm87MacroFeedV4P40ExecutionPackage>::value);
static_assert(
    execution::kSm87MacroFeedV4P40RequestBoundaryHostTokenBytes == 160'000U);
static_assert(
    execution::kSm87MacroFeedV4P40RequestBoundaryHostResultOffset ==
    160'000U);
static_assert(
    execution::kSm87MacroFeedV4P40RequestBoundaryHostStagingBytes ==
    160'008U);
static_assert(execution::kSm87MacroFeedV4P40ExecutionTotalOwnedBytes ==
              3'220'861'192U);
static_assert(
    execution::kSm87MacroFeedV4P40RequestBoundaryTokenIdsScratchBegin == 0U &&
    execution::kSm87MacroFeedV4P40RequestBoundaryTokenIdsScratchEnd ==
        32'000U);
static_assert(
    execution::kSm87MacroFeedV4P40RequestBoundaryFinalNormScratchBegin == 0U &&
    execution::kSm87MacroFeedV4P40RequestBoundaryFinalNormScratchEnd ==
        10'240U);
static_assert(
    execution::kSm87MacroFeedV4P40RequestBoundaryLogitsScratchBegin ==
        10'240U &&
    execution::kSm87MacroFeedV4P40RequestBoundaryLogitsScratchEnd ==
        506'880U);
static_assert(
    execution::kSm87MacroFeedV4P40RequestBoundaryGreedyScratchBegin ==
        506'880U &&
    execution::kSm87MacroFeedV4P40RequestBoundaryGreedyScratchEnd ==
        507'144U);
static_assert(
    execution::kSm87MacroFeedV4P40RequestBoundaryFinalResultScratchBegin ==
        507'136U &&
    execution::kSm87MacroFeedV4P40RequestBoundaryFinalResultScratchEnd ==
        507'144U);

void require_test(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

class FixedBoundaryDeviceAllocation final {
 public:
  FixedBoundaryDeviceAllocation() = default;
  FixedBoundaryDeviceAllocation(const FixedBoundaryDeviceAllocation&) =
      delete;
  FixedBoundaryDeviceAllocation& operator=(
      const FixedBoundaryDeviceAllocation&) = delete;

  ~FixedBoundaryDeviceAllocation() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
      pointer_ = nullptr;
    }
  }

  [[nodiscard]] bool allocate(const std::size_t bytes) noexcept {
    if (pointer_ != nullptr || bytes == 0U ||
        cudaMalloc(&pointer_, bytes) != cudaSuccess || pointer_ == nullptr) {
      pointer_ = nullptr;
      return false;
    }
    bytes_ = bytes;
    return true;
  }

  template <typename T>
  [[nodiscard]] T* as() noexcept {
    return static_cast<T*>(pointer_);
  }

  template <typename T>
  [[nodiscard]] const T* as() const noexcept {
    return static_cast<const T*>(pointer_);
  }

  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0U;
};

void seed_and_require_stale_cuda_error() {
  const cudaError_t injected =
      cudaMemcpy(nullptr, nullptr, 1U, cudaMemcpyHostToDevice);
  require_test(injected == cudaErrorInvalidValue &&
                   cudaPeekAtLastError() == cudaErrorInvalidValue,
               "could not seed exact stale CUDA invalid-value state");
}

void clear_and_require_stale_cuda_error() {
  require_test(cudaGetLastError() == cudaErrorInvalidValue &&
                   cudaPeekAtLastError() == cudaSuccess,
               "fixed boundary stale CUDA state was not retained fail-closed");
}

class LiveExecutionWeights final {
 public:
  LiveExecutionWeights() = default;
  LiveExecutionWeights(const LiveExecutionWeights&) = delete;
  LiveExecutionWeights& operator=(const LiveExecutionWeights&) = delete;

  ~LiveExecutionWeights() {
    if (request_boundary_allocation_ != nullptr) {
      (void)cudaFree(request_boundary_allocation_);
      request_boundary_allocation_ = nullptr;
    }
    if (full_qk_norm_allocation_ != nullptr) {
      (void)cudaFree(full_qk_norm_allocation_);
      full_qk_norm_allocation_ = nullptr;
    }
    if (layer_norm_allocation_ != nullptr) {
      (void)cudaFree(layer_norm_allocation_);
      layer_norm_allocation_ = nullptr;
    }
    if (bf16_ab_allocation_ != nullptr) {
      (void)cudaFree(bf16_ab_allocation_);
      bf16_ab_allocation_ = nullptr;
    }
  }

  [[nodiscard]] bool allocate() noexcept {
    if (bf16_ab_allocation_ != nullptr ||
        layer_norm_allocation_ != nullptr ||
        full_qk_norm_allocation_ != nullptr ||
        request_boundary_allocation_ != nullptr) {
      return false;
    }
    if (cudaMalloc(&bf16_ab_allocation_, kBf16AbAllocationBytes) !=
            cudaSuccess ||
        bf16_ab_allocation_ == nullptr) {
      bf16_ab_allocation_ = nullptr;
      return false;
    }
    if (cudaMemset(bf16_ab_allocation_, 0, kBf16AbAllocationBytes) !=
        cudaSuccess) {
      return false;
    }
    if (cudaMalloc(&full_qk_norm_allocation_,
                   kFullQkNormAllocationBytes) != cudaSuccess ||
        full_qk_norm_allocation_ == nullptr ||
        cudaMemset(full_qk_norm_allocation_, 0,
                   kFullQkNormAllocationBytes) != cudaSuccess) {
      return false;
    }
    if (cudaMalloc(&request_boundary_allocation_,
                   kRequestBoundaryAllocationBytes) != cudaSuccess ||
        request_boundary_allocation_ == nullptr) {
      request_boundary_allocation_ = nullptr;
      return false;
    }
    auto* const request_bytes =
        static_cast<std::uint8_t*>(request_boundary_allocation_);
    request_boundary_.embedding = Bf16LinearWeight{
        reinterpret_cast<const std::uint16_t*>(
            request_bytes + kRequestEmbeddingOffset),
        bound_launch::kSm87MacroFeedV4EmbeddingVocabulary,
        bound_launch::kSm87MacroFeedV4Hidden};
    request_boundary_.final_norm = Bf16VectorWeight{
        reinterpret_cast<const std::uint16_t*>(
            request_bytes + kRequestFinalNormOffset),
        bound_launch::kSm87MacroFeedV4Hidden};
    request_boundary_.lm_head = NvFp4LinearWeight{
        request_bytes + kRequestLmPackedOffset,
        request_bytes + kRequestLmBlockScaleOffset,
        reinterpret_cast<const float*>(request_bytes +
                                       kRequestWeightScaleOffset),
        reinterpret_cast<const float*>(request_bytes +
                                       kRequestInputScaleOffset),
        1.0F,
        1.0F,
        bound_launch::kSm87MacroFeedV4LmHeadRows,
        bound_launch::kSm87MacroFeedV4LmHeadColumns};
    constexpr float kScale = 1.0F;
    if (cudaMemcpy(request_bytes + kRequestWeightScaleOffset, &kScale,
                   sizeof(kScale), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(request_bytes + kRequestInputScaleOffset, &kScale,
                   sizeof(kScale), cudaMemcpyHostToDevice) != cudaSuccess) {
      return false;
    }
    // Keep LayerNorm last: the one-past-final-post-norm negative below must
    // not alias the beginning of a later live fixture allocation.
    if (cudaMalloc(&layer_norm_allocation_, kLayerNormAllocationBytes) !=
            cudaSuccess ||
        layer_norm_allocation_ == nullptr) {
      layer_norm_allocation_ = nullptr;
      return false;
    }
    if (cudaMemset(layer_norm_allocation_, 0,
                   kLayerNormAllocationBytes) != cudaSuccess) {
      return false;
    }

    auto* const bf16_bytes =
        static_cast<std::uint8_t*>(bf16_ab_allocation_);
    for (std::size_t ordinal = 0U; ordinal < bf16_ab_pairs_.size();
         ++ordinal) {
      bf16_ab_pairs_[ordinal].a = Bf16LinearWeight{
          reinterpret_cast<const std::uint16_t*>(
              bf16_bytes + (2U * ordinal) * kBf16AbWeightBytes),
          kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection,
          kernels::kSm87MacroFeedV4Bf16AbInputFeatures};
      bf16_ab_pairs_[ordinal].b = Bf16LinearWeight{
          reinterpret_cast<const std::uint16_t*>(
              bf16_bytes + (2U * ordinal + 1U) * kBf16AbWeightBytes),
          kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection,
          kernels::kSm87MacroFeedV4Bf16AbInputFeatures};
    }

    auto* const norm_bytes =
        static_cast<std::uint8_t*>(layer_norm_allocation_);
    for (std::size_t layer = 0U; layer < layer_norm_pairs_.size(); ++layer) {
      layer_norm_pairs_[layer].input_layernorm = Bf16VectorWeight{
          reinterpret_cast<const std::uint16_t*>(
              norm_bytes + (2U * layer) * kLayerNormWeightBytes),
          kernels::kSm87MacroFeedV4NormResidualHidden};
      layer_norm_pairs_[layer].post_attention_layernorm = Bf16VectorWeight{
          reinterpret_cast<const std::uint16_t*>(
              norm_bytes + (2U * layer + 1U) * kLayerNormWeightBytes),
          kernels::kSm87MacroFeedV4NormResidualHidden};
    }
    auto* const full_norm_bytes =
        static_cast<std::uint8_t*>(full_qk_norm_allocation_);
    for (std::size_t ordinal = 0U; ordinal < full_qk_norm_pairs_.size();
         ++ordinal) {
      full_qk_norm_pairs_[ordinal].q_norm = Bf16VectorWeight{
          reinterpret_cast<const std::uint16_t*>(
              full_norm_bytes + (2U * ordinal) * kFullQkNormWeightBytes),
          kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension};
      full_qk_norm_pairs_[ordinal].k_norm = Bf16VectorWeight{
          reinterpret_cast<const std::uint16_t*>(
              full_norm_bytes +
              (2U * ordinal + 1U) * kFullQkNormWeightBytes),
          kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension};
    }
    return true;
  }

  [[nodiscard]] bool install(ModelWeights& model_weights) const noexcept {
    return bf16_ab_allocation_ != nullptr &&
           layer_norm_allocation_ != nullptr &&
           full_qk_norm_allocation_ != nullptr &&
           request_boundary_allocation_ != nullptr &&
           Access::install_complete_host_test_bf16_ab_pairs(
               model_weights, bf16_ab_pairs_.data(),
               bf16_ab_pairs_.size()) &&
           Access::install_complete_host_test_layer_norm_pairs(
               model_weights, layer_norm_pairs_.data(),
               layer_norm_pairs_.size()) &&
           Access::install_complete_host_test_full_qk_norm_pairs(
               model_weights, full_qk_norm_pairs_.data(),
               full_qk_norm_pairs_.size()) &&
           Access::install_complete_host_test_request_boundary(
               model_weights, request_boundary_);
  }

  [[nodiscard]] bool install_one_past_final_post_norm(
      ModelWeights& model_weights) noexcept {
    if (layer_norm_allocation_ == nullptr) {
      return false;
    }
    saved_final_post_norm_ = layer_norm_pairs_.back().post_attention_layernorm;
    layer_norm_pairs_.back().post_attention_layernorm = Bf16VectorWeight{
        reinterpret_cast<const std::uint16_t*>(
            static_cast<const std::uint8_t*>(layer_norm_allocation_) +
            kLayerNormAllocationBytes),
        kernels::kSm87MacroFeedV4NormResidualHidden};
    final_post_norm_poisoned_ = true;
    return Access::install_complete_host_test_layer_norm_pairs(
        model_weights, layer_norm_pairs_.data(), layer_norm_pairs_.size());
  }

  [[nodiscard]] bool restore_final_post_norm(
      ModelWeights& model_weights) noexcept {
    if (!final_post_norm_poisoned_) {
      return false;
    }
    layer_norm_pairs_.back().post_attention_layernorm = saved_final_post_norm_;
    saved_final_post_norm_ = {};
    final_post_norm_poisoned_ = false;
    return Access::install_complete_host_test_layer_norm_pairs(
        model_weights, layer_norm_pairs_.data(), layer_norm_pairs_.size());
  }

 private:
  static constexpr std::size_t kBf16AbWeightBytes =
      kernels::kSm87MacroFeedV4Bf16AbWeightBytes;
  static constexpr std::size_t kBf16AbAllocationBytes =
      2U * q3x::runtime::kQwen36LinearAttentionLayerCount *
      kBf16AbWeightBytes;
  static constexpr std::size_t kLayerNormWeightBytes =
      kernels::kSm87MacroFeedV4NormResidualWeightBytes;
  static constexpr std::size_t kLayerNormAllocationBytes =
      2U * q3x::runtime::kQwen36DenseLayerCount *
      kLayerNormWeightBytes;
  static constexpr std::size_t kFullQkNormWeightBytes =
      kernels::kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes;
  static constexpr std::size_t kFullQkNormAllocationBytes =
      2U * q3x::runtime::kQwen36FullAttentionLayerCount *
      kFullQkNormWeightBytes;
  static constexpr std::size_t kRequestEmbeddingOffset = 0U;
  static constexpr std::size_t kRequestFinalNormOffset =
      kRequestEmbeddingOffset +
      bound_launch::kSm87MacroFeedV4EmbeddingTableBytes;
  static constexpr std::size_t kRequestLmPackedOffset =
      kRequestFinalNormOffset + bound_launch::kSm87MacroFeedV4FinalNormBytes;
  static constexpr std::size_t kRequestLmBlockScaleOffset =
      kRequestLmPackedOffset +
      bound_launch::kSm87MacroFeedV4LmHeadPackedWeightBytes;
  static constexpr std::size_t kRequestWeightScaleOffset =
      kRequestLmBlockScaleOffset +
      bound_launch::kSm87MacroFeedV4LmHeadBlockScaleBytes;
  static constexpr std::size_t kRequestInputScaleOffset =
      kRequestWeightScaleOffset + 16U;
  static constexpr std::size_t kRequestBoundaryAllocationBytes =
      kRequestInputScaleOffset + 16U;

  static_assert(kBf16AbAllocationBytes == 47'185'920U);
  static_assert(kLayerNormAllocationBytes == 1'310'720U);
  static_assert(kFullQkNormAllocationBytes == 16'384U);
  static_assert(kBf16AbWeightBytes % 256U == 0U);
  static_assert(kLayerNormWeightBytes % 256U == 0U);

  void* bf16_ab_allocation_ = nullptr;
  void* layer_norm_allocation_ = nullptr;
  void* full_qk_norm_allocation_ = nullptr;
  void* request_boundary_allocation_ = nullptr;
  std::array<Bf16AbPair, q3x::runtime::kQwen36LinearAttentionLayerCount>
      bf16_ab_pairs_{};
  std::array<LayerNormPair, q3x::runtime::kQwen36DenseLayerCount>
      layer_norm_pairs_{};
  std::array<FullQkNormPair,
             q3x::runtime::kQwen36FullAttentionLayerCount>
      full_qk_norm_pairs_{};
  RequestBoundary request_boundary_{};
  Bf16VectorWeight saved_final_post_norm_{};
  bool final_post_norm_poisoned_ = false;
};

class LiveGdnContinuationWeights final {
 public:
  LiveGdnContinuationWeights() = default;
  LiveGdnContinuationWeights(const LiveGdnContinuationWeights&) = delete;
  LiveGdnContinuationWeights& operator=(
      const LiveGdnContinuationWeights&) = delete;

  ~LiveGdnContinuationWeights() {
    if (allocation_ != nullptr) {
      (void)cudaFree(allocation_);
      allocation_ = nullptr;
    }
  }

  [[nodiscard]] bool allocate() noexcept {
    if (allocation_ != nullptr ||
        cudaMalloc(&allocation_, kAllocationBytes) != cudaSuccess ||
        allocation_ == nullptr) {
      return false;
    }
    return cudaMemset(allocation_, 0, kAllocationBytes) == cudaSuccess &&
           cudaDeviceSynchronize() == cudaSuccess;
  }

  [[nodiscard]] const std::uint16_t* conv_weight() const noexcept {
    return pointer_at(kConvWeightOffset);
  }
  [[nodiscard]] const std::uint16_t* a_log() const noexcept {
    return pointer_at(kALogOffset);
  }
  [[nodiscard]] const std::uint16_t* dt_bias() const noexcept {
    return pointer_at(kDtBiasOffset);
  }
  [[nodiscard]] const std::uint16_t* norm_weight() const noexcept {
    return pointer_at(kNormWeightOffset);
  }
  [[nodiscard]] constexpr std::size_t bytes() const noexcept {
    return kAllocationBytes;
  }

 private:
  static constexpr std::size_t kConvWeightOffset = 0U;
  static constexpr std::size_t kALogOffset =
      kConvWeightOffset + kernels::kSm87MacroFeedV4GdnConvWeightBytes;
  static constexpr std::size_t kDtBiasOffset =
      kALogOffset + kernels::kSm87MacroFeedV4GdnHeadVectorBytes;
  static constexpr std::size_t kNormWeightOffset =
      kDtBiasOffset + kernels::kSm87MacroFeedV4GdnHeadVectorBytes;
  static constexpr std::size_t kAllocationBytes =
      kNormWeightOffset + kernels::kSm87MacroFeedV4GdnNormWeightBytes;

  static_assert(kernels::kSm87MacroFeedV4GdnConvWeightBytes == 81'920U);
  static_assert(kernels::kSm87MacroFeedV4GdnHeadVectorBytes == 96U);
  static_assert(kernels::kSm87MacroFeedV4GdnNormWeightBytes == 256U);
  static_assert(kALogOffset % kernels::kSm87MacroFeedV4GdnPointerAlignment ==
                0U);
  static_assert(kDtBiasOffset %
                    kernels::kSm87MacroFeedV4GdnPointerAlignment ==
                0U);
  static_assert(kNormWeightOffset %
                    kernels::kSm87MacroFeedV4GdnPointerAlignment ==
                0U);
  static_assert(kAllocationBytes == 82'368U);

  [[nodiscard]] const std::uint16_t* pointer_at(
      const std::size_t offset) const noexcept {
    return allocation_ == nullptr
               ? nullptr
               : reinterpret_cast<const std::uint16_t*>(
                     static_cast<const std::uint8_t*>(allocation_) + offset);
  }

  void* allocation_ = nullptr;
};

[[nodiscard]] kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt
make_gate_up_payload_receipt(
    const q3x::tests::support::Sm87MacroFeedV4LiveNvFp4AssetFixture&
        fixture) noexcept {
  const auto layout = kernels::sm87_target_aot_projection_packed_layout(
      kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp);
  kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt receipt;
  receipt.plan_identity = kernels::kSm87MacroFeedV3NvFp4GateUpIdentity;
  receipt.payload_identity = fixture.asset.artifact_identity;
  receipt.gate_source_identity = fixture.inventory.sources[0U].tensor_identity;
  receipt.up_source_identity = fixture.inventory.sources[1U].tensor_identity;
  receipt.device_ordinal = fixture.upload.device_ordinal;
  receipt.payload_begin = fixture.asset.payload.begin;
  receipt.payload_end = fixture.asset.payload.end;
  receipt.payload_bytes = fixture.asset.payload.bytes;
  receipt.gate_partition_bytes = layout.partitions[0U].payload_bytes;
  receipt.up_partition_bytes = layout.partitions[1U].payload_bytes;
  receipt.canonical_consumer_n64_k16_lane_component_v1 =
      fixture.asset.transform_identity ==
      kernels::Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.canonical_gate_then_up_partition_order =
      layout.partitions[0U].logical_role ==
          kernels::Sm87TargetAotLogicalRole::kNvFp4Gate &&
      layout.partitions[1U].logical_role ==
          kernels::Sm87TargetAotLogicalRole::kNvFp4Up;
  receipt.independent_tensor_scales =
      layout.partitions[0U].independent_tensor_scale &&
      layout.partitions[1U].independent_tensor_scale;
  receipt.host_bytes_authenticated_before_copy =
      fixture.upload.host_payload_digest_verified_before_copy &&
      fixture.upload.host_payload_immutable_until_completion;
  receipt.device_readback_authenticated =
      fixture.upload.verification_event_observed &&
      fixture.upload.verification_completed &&
      fixture.upload.device_payload_matches_host_payload;
  receipt.allocation_retained_for_launch =
      fixture.upload.allocation_retained_for_asset_lifetime;
  receipt.receipt_identity = kernels::
      sm87_macrofeed_v3_nvfp4_gate_up_compute_payload_receipt_identity(
          receipt);
  return receipt;
}

[[nodiscard]] kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt
make_down_payload_receipt(
    const q3x::tests::support::Sm87MacroFeedV4LiveNvFp4AssetFixture&
        fixture) noexcept {
  kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt receipt;
  receipt.plan_identity = kernels::kSm87MacroFeedV3NvFp4DownIdentity;
  receipt.payload_identity = fixture.asset.artifact_identity;
  receipt.device_ordinal = fixture.upload.device_ordinal;
  receipt.payload_begin = fixture.asset.payload.begin;
  receipt.payload_end = fixture.asset.payload.end;
  receipt.payload_bytes = fixture.asset.payload.bytes;
  receipt.canonical_consumer_n64_k16_lane_component_v1 =
      fixture.asset.transform_identity ==
      kernels::Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.host_bytes_authenticated_before_copy =
      fixture.upload.host_payload_digest_verified_before_copy &&
      fixture.upload.host_payload_immutable_until_completion;
  receipt.device_readback_authenticated =
      fixture.upload.verification_event_observed &&
      fixture.upload.verification_completed &&
      fixture.upload.device_payload_matches_host_payload;
  receipt.allocation_retained_for_launch =
      fixture.upload.allocation_retained_for_asset_lifetime;
  receipt.receipt_identity =
      kernels::sm87_macrofeed_v3_nvfp4_down_compute_payload_receipt_identity(
          receipt);
  return receipt;
}

[[nodiscard]] bool exact_sm87_device_available() noexcept {
  int device = -1;
  cudaDeviceProp properties{};
  return cudaGetDevice(&device) == cudaSuccess && device == 0 &&
         cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
         properties.major == 8 && properties.minor == 7 &&
         properties.multiProcessorCount == 16;
}

[[nodiscard]] bool clear_fixture(std::optional<ModelWeights>& model_weights,
                                 Owner& owner) noexcept {
  model_weights.reset();
  return Access::clear_host_test_fixture(owner) && owner.empty();
}

void test_fixed_outer_request_boundaries() {
  using EventsFixture = events::Sm87MacroFeedV4ExecutionEventsCudaTestFixture;
  static_assert(bound_launch::kSm87MacroFeedV4EmbeddingTokenCount == 8'000U);
  static_assert(bound_launch::kSm87MacroFeedV4EmbeddingVocabulary ==
                248'320U);
  static_assert(bound_launch::kSm87MacroFeedV4Hidden == 5'120U);
  static_assert(bound_launch::kSm87MacroFeedV4FinalNormEpsilonFp32Bits ==
                0x3586'37bdU);
  static_assert(bound_launch::kSm87MacroFeedV4GreedyWorkspaceResults == 33U);
  static_assert(bound_launch::kSm87MacroFeedV4LmHeadRows == 248'320U);
  static_assert(bound_launch::kSm87MacroFeedV4LmHeadColumns == 5'120U);

  bound_launch::Sm87MacroFeedV4EmbeddingC8000ResourceSnapshot
      embedding_resources{};
  bound_launch::Sm87MacroFeedV4FinalNormM1ResourceSnapshot norm_resources{};
  bound_launch::Sm87MacroFeedV4GreedyArgmaxM1ResourceSnapshot
      greedy_resources{};
  bound_launch::Sm87MacroFeedV4LmHeadM1ResourceSnapshot lm_resources{};
  require_test(
      bound_launch::query_embedding_c8000_resources_cuda(
          &embedding_resources) == static_cast<int>(cudaSuccess) &&
          embedding_resources.identity != 0U &&
          embedding_resources.exact_geometry &&
          embedding_resources.static_resource_gate_passed &&
          embedding_resources.gather.threads == 256 &&
          embedding_resources.gather.grid_ctas == 8'000 &&
          embedding_resources.gather.static_shared_bytes == 0U &&
          embedding_resources.gather.local_bytes == 0U,
      "fixed C8000 embedding construction snapshot failed");
  require_test(
      bound_launch::query_final_norm_m1_resources_cuda(&norm_resources) ==
              static_cast<int>(cudaSuccess) &&
          norm_resources.identity != 0U && norm_resources.exact_geometry &&
          norm_resources.static_resource_gate_passed &&
          norm_resources.centered_rms_norm.threads == 256 &&
          norm_resources.centered_rms_norm.grid_ctas == 1 &&
          norm_resources.centered_rms_norm.static_shared_bytes == 1'024U &&
          norm_resources.centered_rms_norm.local_bytes == 0U,
      "fixed final centered RMSNorm construction snapshot failed");
  require_test(
      bound_launch::query_greedy_argmax_m1_resources_cuda(
          &greedy_resources) == static_cast<int>(cudaSuccess) &&
          greedy_resources.identity != 0U &&
          greedy_resources.exact_geometry &&
          greedy_resources.static_resource_gate_passed &&
          greedy_resources.partial.threads == 256 &&
          greedy_resources.partial.grid_ctas == 32 &&
          greedy_resources.partial.static_shared_bytes == 3'072U &&
          greedy_resources.finalize.threads == 32 &&
          greedy_resources.finalize.grid_ctas == 1,
      "fixed BF16 greedy construction snapshot failed");
  require_test(
      bound_launch::query_lm_head_m1_resources_cuda(&lm_resources) ==
              static_cast<int>(cudaSuccess) &&
          lm_resources.identity != 0U && lm_resources.exact_geometry &&
          lm_resources.static_resource_gate_passed &&
          lm_resources.activation_staged.registers_per_thread == 64 &&
          lm_resources.activation_staged.static_shared_bytes == 11'328U &&
          lm_resources.activation_staged.local_bytes == 0U &&
          lm_resources.activation_staged.active_blocks_per_sm == 4 &&
          lm_resources.activation_staged.threads == 256 &&
          lm_resources.activation_staged.grid_ctas == 64,
      "fixed NVFP4 LM-head construction snapshot failed");
  require_test(
      bound_launch::query_embedding_c8000_resources_cuda(nullptr) ==
              static_cast<int>(cudaErrorInvalidValue) &&
          bound_launch::query_final_norm_m1_resources_cuda(nullptr) ==
              static_cast<int>(cudaErrorInvalidValue) &&
          bound_launch::query_greedy_argmax_m1_resources_cuda(nullptr) ==
              static_cast<int>(cudaErrorInvalidValue) &&
          bound_launch::query_lm_head_m1_resources_cuda(nullptr) ==
              static_cast<int>(cudaErrorInvalidValue),
      "fixed boundary construction queries accepted null snapshots");

  auto owner_result = events::create_sm87_macrofeed_v4_execution_events_owner();
  require_test(static_cast<bool>(owner_result) && owner_result.owner != nullptr,
               "could not create fixed-boundary EventsOwner");
  auto& boundary_owner = *owner_result.owner;

  {
    FixedBoundaryDeviceAllocation table;
    FixedBoundaryDeviceAllocation token_ids;
    FixedBoundaryDeviceAllocation hidden_output;
    require_test(
        table.allocate(bound_launch::kSm87MacroFeedV4EmbeddingTableBytes) &&
            token_ids.allocate(
                bound_launch::kSm87MacroFeedV4EmbeddingTokenIdBytes) &&
            hidden_output.allocate(
                bound_launch::kSm87MacroFeedV4EmbeddingOutputBytes),
        "could not allocate exact C8000 embedding boundary buffers");

    constexpr std::uint32_t kFirstTokenId = 17U;
    constexpr std::uint32_t kLastTokenId = 19U;
    std::vector<std::uint32_t> host_token_ids(
        bound_launch::kSm87MacroFeedV4EmbeddingTokenCount, kFirstTokenId);
    host_token_ids.back() = kLastTokenId;
    std::vector<std::uint16_t> first_row(
        bound_launch::kSm87MacroFeedV4Hidden);
    std::vector<std::uint16_t> last_row(
        bound_launch::kSm87MacroFeedV4Hidden);
    for (std::size_t hidden = 0U;
         hidden < bound_launch::kSm87MacroFeedV4Hidden; ++hidden) {
      first_row[hidden] =
          static_cast<std::uint16_t>(0x1000U + hidden % 0x0400U);
      last_row[hidden] =
          static_cast<std::uint16_t>(0x5000U + hidden % 0x0400U);
    }
    require_test(
        cudaMemcpy(token_ids.as<std::uint32_t>(), host_token_ids.data(),
                   bound_launch::kSm87MacroFeedV4EmbeddingTokenIdBytes,
                   cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(
                table.as<std::uint16_t>() +
                    static_cast<std::size_t>(kFirstTokenId) *
                        bound_launch::kSm87MacroFeedV4Hidden,
                first_row.data(), bound_launch::kSm87MacroFeedV4FinalNormBytes,
                cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(
                table.as<std::uint16_t>() +
                    static_cast<std::size_t>(kLastTokenId) *
                        bound_launch::kSm87MacroFeedV4Hidden,
                last_row.data(), bound_launch::kSm87MacroFeedV4FinalNormBytes,
                cudaMemcpyHostToDevice) == cudaSuccess,
        "could not install synthetic embedding rows and IDs");
    require_test(cudaDeviceSynchronize() == cudaSuccess,
                 "embedding fixture seed did not physically complete");

    const bound_launch::Sm87MacroFeedV4EmbeddingC8000Arguments arguments{
        table.as<std::uint16_t>(), token_ids.as<std::uint32_t>(),
        hidden_output.as<std::uint16_t>()};
    const auto invoke_embedding =
        [&](const bound_launch::Sm87MacroFeedV4EmbeddingC8000Arguments& args,
            const bound_launch::Sm87MacroFeedV4EmbeddingC8000ResourceSnapshot&
                resources,
            bound_launch::Sm87MacroFeedV4FixedSubmitLedger* const ledger) {
          return EventsFixture::submit_embedding_c8000_constituent(
              boundary_owner, args, resources, ledger);
        };
    bound_launch::Sm87MacroFeedV4FixedSubmitLedger ledger{7U};
    auto tampered_resources = embedding_resources;
    tampered_resources.identity ^= 1U;
    require_test(
        invoke_embedding(arguments, tampered_resources, &ledger) ==
                static_cast<int>(cudaErrorLaunchOutOfResources) &&
            ledger.accepted_kernel_launches == 0U,
        "embedding seam accepted a tampered construction snapshot");
    auto alias_arguments = arguments;
    alias_arguments.hidden_output =
        const_cast<std::uint16_t*>(arguments.embedding_table);
    ledger.accepted_kernel_launches = 7U;
    require_test(
        invoke_embedding(alias_arguments, embedding_resources, &ledger) ==
                static_cast<int>(cudaErrorInvalidValue) &&
            ledger.accepted_kernel_launches == 0U,
        "embedding seam accepted overlapping table/output extents");
    seed_and_require_stale_cuda_error();
    ledger.accepted_kernel_launches = 7U;
    require_test(
        invoke_embedding(arguments, embedding_resources, &ledger) ==
                static_cast<int>(cudaErrorInvalidValue) &&
            ledger.accepted_kernel_launches == 0U,
        "embedding seam consumed stale error or accepted work after it");
    clear_and_require_stale_cuda_error();
    require_test(
        invoke_embedding(arguments, embedding_resources, &ledger) ==
                static_cast<int>(cudaSuccess) &&
            ledger.accepted_kernel_launches == 1U &&
            cudaDeviceSynchronize() == cudaSuccess,
        "fixed C8000 embedding launch failed");
    std::uint16_t first_sample = 0U;
    std::uint16_t middle_sample = 0U;
    std::uint16_t last_sample = 0U;
    require_test(
        cudaMemcpy(&first_sample, hidden_output.as<std::uint16_t>(),
                   sizeof(first_sample), cudaMemcpyDeviceToHost) ==
                cudaSuccess &&
            cudaMemcpy(
                &middle_sample,
                hidden_output.as<std::uint16_t>() +
                    4'000U * bound_launch::kSm87MacroFeedV4Hidden + 2'557U,
                sizeof(middle_sample), cudaMemcpyDeviceToHost) ==
                cudaSuccess &&
            cudaMemcpy(
                &last_sample,
                hidden_output.as<std::uint16_t>() +
                    (bound_launch::kSm87MacroFeedV4EmbeddingTokenCount - 1U) *
                        bound_launch::kSm87MacroFeedV4Hidden +
                    (bound_launch::kSm87MacroFeedV4Hidden - 1U),
                sizeof(last_sample), cudaMemcpyDeviceToHost) == cudaSuccess &&
            first_sample == first_row.front() &&
            middle_sample == first_row[2'557U] &&
            last_sample == last_row.back(),
        "fixed C8000 embedding did not preserve exact BF16 row payloads");
  }

  {
    FixedBoundaryDeviceAllocation hidden;
    FixedBoundaryDeviceAllocation centered_weight;
    require_test(
        hidden.allocate(bound_launch::kSm87MacroFeedV4FinalNormBytes) &&
            centered_weight.allocate(
                bound_launch::kSm87MacroFeedV4FinalNormBytes),
        "could not allocate fixed final RMSNorm buffers");
    const std::vector<std::uint16_t> host_hidden(
        bound_launch::kSm87MacroFeedV4Hidden, 0x4000U);
    const std::vector<std::uint16_t> host_weight(
        bound_launch::kSm87MacroFeedV4Hidden, 0x3f00U);
    require_test(
        cudaMemcpy(hidden.as<std::uint16_t>(), host_hidden.data(),
                   bound_launch::kSm87MacroFeedV4FinalNormBytes,
                   cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(centered_weight.as<std::uint16_t>(),
                       host_weight.data(),
                       bound_launch::kSm87MacroFeedV4FinalNormBytes,
                       cudaMemcpyHostToDevice) == cudaSuccess,
        "could not seed final RMSNorm fixture");
    require_test(cudaDeviceSynchronize() == cudaSuccess,
                 "final RMSNorm fixture seed did not physically complete");
    const bound_launch::Sm87MacroFeedV4FinalNormM1Arguments arguments{
        hidden.as<std::uint16_t>(), centered_weight.as<std::uint16_t>(),
        hidden.as<std::uint16_t>()};
    const auto invoke_norm =
        [&](const bound_launch::Sm87MacroFeedV4FinalNormM1Arguments& args,
            const bound_launch::Sm87MacroFeedV4FinalNormM1ResourceSnapshot&
                resources,
            bound_launch::Sm87MacroFeedV4FixedSubmitLedger* const ledger) {
          return EventsFixture::submit_final_norm_m1_constituent(
              boundary_owner, args, resources, ledger);
        };
    bound_launch::Sm87MacroFeedV4FixedSubmitLedger ledger{7U};
    auto tampered_resources = norm_resources;
    tampered_resources.static_resource_gate_passed = false;
    require_test(
        invoke_norm(arguments, tampered_resources, &ledger) ==
                static_cast<int>(cudaErrorLaunchOutOfResources) &&
            ledger.accepted_kernel_launches == 0U,
        "final RMSNorm seam accepted a failed resource gate");
    auto partial_alias = arguments;
    partial_alias.normalized_output =
        const_cast<std::uint16_t*>(arguments.hidden_input) + 1U;
    ledger.accepted_kernel_launches = 7U;
    require_test(
        invoke_norm(partial_alias, norm_resources, &ledger) ==
                static_cast<int>(cudaErrorInvalidValue) &&
            ledger.accepted_kernel_launches == 0U,
        "final RMSNorm seam accepted a shifted partial alias");
    seed_and_require_stale_cuda_error();
    ledger.accepted_kernel_launches = 7U;
    require_test(
        invoke_norm(arguments, norm_resources, &ledger) ==
                static_cast<int>(cudaErrorInvalidValue) &&
            ledger.accepted_kernel_launches == 0U,
        "final RMSNorm seam accepted work behind a stale error");
    clear_and_require_stale_cuda_error();
    require_test(
        invoke_norm(arguments, norm_resources, &ledger) ==
                static_cast<int>(cudaSuccess) &&
            ledger.accepted_kernel_launches == 1U &&
            cudaDeviceSynchronize() == cudaSuccess,
        "fixed in-place final centered RMSNorm launch failed");
    std::vector<std::uint16_t> normalized(
        bound_launch::kSm87MacroFeedV4Hidden);
    require_test(
        cudaMemcpy(normalized.data(), hidden.as<std::uint16_t>(),
                   bound_launch::kSm87MacroFeedV4FinalNormBytes,
                   cudaMemcpyDeviceToHost) == cudaSuccess,
        "could not read final RMSNorm output");
    for (const std::uint16_t value : normalized) {
      require_test(value == 0x3fc0U,
                   "final centered RMSNorm lost exact BF16 result");
    }
  }

  {
    FixedBoundaryDeviceAllocation packed_weights;
    FixedBoundaryDeviceAllocation block_scales;
    FixedBoundaryDeviceAllocation activation;
    FixedBoundaryDeviceAllocation logits;
    FixedBoundaryDeviceAllocation greedy_workspace;
    require_test(
        packed_weights.allocate(
            bound_launch::kSm87MacroFeedV4LmHeadPackedWeightBytes) &&
            block_scales.allocate(
                bound_launch::kSm87MacroFeedV4LmHeadBlockScaleBytes) &&
            activation.allocate(
                bound_launch::kSm87MacroFeedV4LmHeadActivationBytes) &&
            logits.allocate(bound_launch::kSm87MacroFeedV4LmHeadOutputBytes) &&
            greedy_workspace.allocate(
                bound_launch::kSm87MacroFeedV4GreedyWorkspaceBytes),
        "could not allocate fixed LM-head/greedy boundary buffers");

    constexpr std::size_t kSelectedRow = 123'456U;
    constexpr std::size_t kPackedBytesPerRow =
        bound_launch::kSm87MacroFeedV4LmHeadColumns / 2U;
    constexpr std::size_t kScaleBytesPerRow =
        bound_launch::kSm87MacroFeedV4LmHeadColumns / 16U;
    const std::vector<std::uint8_t> selected_packed_row(
        kPackedBytesPerRow, 0x22U);
    const std::vector<std::uint8_t> selected_scale_row(
        kScaleBytesPerRow, 0x38U);
    const std::vector<std::uint16_t> host_activation(
        bound_launch::kSm87MacroFeedV4LmHeadColumns, 0x3f80U);
    require_test(
        cudaMemset(block_scales.as<std::uint8_t>(), 0,
                   block_scales.bytes()) == cudaSuccess &&
            cudaMemcpy(
                packed_weights.as<std::uint8_t>() +
                    kSelectedRow * kPackedBytesPerRow,
                selected_packed_row.data(), selected_packed_row.size(),
                cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(
                block_scales.as<std::uint8_t>() +
                    kSelectedRow * kScaleBytesPerRow,
                selected_scale_row.data(), selected_scale_row.size(),
                cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(
                activation.as<std::uint16_t>(), host_activation.data(),
                bound_launch::kSm87MacroFeedV4LmHeadActivationBytes,
                cudaMemcpyHostToDevice) == cudaSuccess,
        "could not seed fixed NVFP4 LM-head fixture");
    require_test(cudaDeviceSynchronize() == cudaSuccess,
                 "LM-head fixture seed did not physically complete");

    const bound_launch::Sm87MacroFeedV4LmHeadM1Arguments lm_arguments{
        packed_weights.as<std::uint8_t>(), block_scales.as<std::uint8_t>(),
        0.5F, activation.as<std::uint16_t>(), logits.as<std::uint16_t>()};
    const auto invoke_lm =
        [&](const bound_launch::Sm87MacroFeedV4LmHeadM1Arguments& args,
            const bound_launch::Sm87MacroFeedV4LmHeadM1ResourceSnapshot&
                resources,
            bound_launch::Sm87MacroFeedV4FixedSubmitLedger* const ledger) {
          return EventsFixture::submit_lm_head_m1_constituent(
              boundary_owner, args, resources, ledger);
        };
    bound_launch::Sm87MacroFeedV4FixedSubmitLedger lm_ledger{7U};
    auto tampered_lm_resources = lm_resources;
    tampered_lm_resources.activation_staged.grid_ctas = 63;
    require_test(
        invoke_lm(lm_arguments, tampered_lm_resources, &lm_ledger) ==
                static_cast<int>(cudaErrorLaunchOutOfResources) &&
            lm_ledger.accepted_kernel_launches == 0U,
        "LM-head seam accepted non-fixed launch geometry");
    std::array<bound_launch::Sm87MacroFeedV4LmHeadM1Arguments, 4U>
        null_role_arguments{lm_arguments, lm_arguments, lm_arguments,
                            lm_arguments};
    null_role_arguments[0].packed_weights = nullptr;
    null_role_arguments[1].block_scales = nullptr;
    null_role_arguments[2].activation = nullptr;
    null_role_arguments[3].logits_output = nullptr;
    for (const auto& null_role : null_role_arguments) {
      lm_ledger.accepted_kernel_launches = 7U;
      require_test(
          invoke_lm(null_role, lm_resources, &lm_ledger) ==
                  static_cast<int>(cudaErrorInvalidValue) &&
              lm_ledger.accepted_kernel_launches == 0U,
          "LM-head seam accepted a null fixed-role pointer");
    }
    std::array<bound_launch::Sm87MacroFeedV4LmHeadM1Arguments, 2U>
        invalid_scale_arguments{lm_arguments, lm_arguments};
    invalid_scale_arguments[0].weight_scale_2 = -0.5F;
    invalid_scale_arguments[1].weight_scale_2 =
        std::numeric_limits<float>::quiet_NaN();
    for (const auto& invalid_scale : invalid_scale_arguments) {
      lm_ledger.accepted_kernel_launches = 7U;
      require_test(
          invoke_lm(invalid_scale, lm_resources, &lm_ledger) ==
                  static_cast<int>(cudaErrorInvalidValue) &&
              lm_ledger.accepted_kernel_launches == 0U,
          "LM-head seam accepted an invalid fixed weight scale");
    }
    auto aliased_lm_arguments = lm_arguments;
    aliased_lm_arguments.logits_output =
        const_cast<std::uint16_t*>(lm_arguments.activation);
    lm_ledger.accepted_kernel_launches = 7U;
    require_test(
        invoke_lm(aliased_lm_arguments, lm_resources, &lm_ledger) ==
                static_cast<int>(cudaErrorInvalidValue) &&
            lm_ledger.accepted_kernel_launches == 0U,
        "LM-head seam accepted activation/output aliasing");
    seed_and_require_stale_cuda_error();
    lm_ledger.accepted_kernel_launches = 7U;
    require_test(
        invoke_lm(lm_arguments, lm_resources, &lm_ledger) ==
                static_cast<int>(cudaErrorInvalidValue) &&
            lm_ledger.accepted_kernel_launches == 0U,
        "LM-head seam accepted work behind a stale error");
    clear_and_require_stale_cuda_error();
    require_test(
        invoke_lm(lm_arguments, lm_resources, &lm_ledger) ==
                static_cast<int>(cudaSuccess) &&
            lm_ledger.accepted_kernel_launches == 1U &&
            cudaDeviceSynchronize() == cudaSuccess,
        "fixed activation-staged NVFP4 LM-head launch failed");
    std::uint16_t zero_logit = 1U;
    std::uint16_t selected_logit = 0U;
    require_test(
        cudaMemcpy(&zero_logit, logits.as<std::uint16_t>(),
                   sizeof(zero_logit), cudaMemcpyDeviceToHost) ==
                cudaSuccess &&
            cudaMemcpy(&selected_logit,
                       logits.as<std::uint16_t>() + kSelectedRow,
                       sizeof(selected_logit), cudaMemcpyDeviceToHost) ==
                cudaSuccess &&
            zero_logit == 0U && selected_logit == 0x4520U,
        "fixed LM-head did not preserve exact scale2-only BF16 output");

    const bound_launch::Sm87MacroFeedV4GreedyArgmaxM1Arguments
        greedy_arguments{
            logits.as<std::uint16_t>(),
            greedy_workspace.as<q3x::runtime::Bf16GreedyArgmaxResult>()};
    const auto invoke_greedy =
        [&](const bound_launch::Sm87MacroFeedV4GreedyArgmaxM1Arguments& args,
            const bound_launch::Sm87MacroFeedV4GreedyArgmaxM1ResourceSnapshot&
                resources,
            bound_launch::Sm87MacroFeedV4FixedSubmitLedger* const ledger) {
          return EventsFixture::submit_greedy_argmax_m1_constituent(
              boundary_owner, args, resources, ledger);
        };
    bound_launch::Sm87MacroFeedV4FixedSubmitLedger greedy_ledger{7U};
    auto tampered_greedy_resources = greedy_resources;
    tampered_greedy_resources.partial.static_shared_bytes = 0U;
    require_test(
        invoke_greedy(greedy_arguments, tampered_greedy_resources,
                      &greedy_ledger) ==
                static_cast<int>(cudaErrorLaunchOutOfResources) &&
            greedy_ledger.accepted_kernel_launches == 0U,
        "greedy seam accepted tampered partial resources");
    auto aliased_greedy_arguments = greedy_arguments;
    aliased_greedy_arguments.result_workspace =
        reinterpret_cast<q3x::runtime::Bf16GreedyArgmaxResult*>(
            logits.as<std::uint16_t>());
    greedy_ledger.accepted_kernel_launches = 7U;
    require_test(
        invoke_greedy(aliased_greedy_arguments, greedy_resources,
                      &greedy_ledger) ==
                static_cast<int>(cudaErrorInvalidValue) &&
            greedy_ledger.accepted_kernel_launches == 0U,
        "greedy seam accepted logits/workspace aliasing");
    seed_and_require_stale_cuda_error();
    greedy_ledger.accepted_kernel_launches = 7U;
    require_test(
        invoke_greedy(greedy_arguments, greedy_resources, &greedy_ledger) ==
                static_cast<int>(cudaErrorInvalidValue) &&
            greedy_ledger.accepted_kernel_launches == 0U,
        "greedy seam accepted work behind a stale error");
    clear_and_require_stale_cuda_error();
    require_test(
        invoke_greedy(greedy_arguments, greedy_resources, &greedy_ledger) ==
                static_cast<int>(cudaSuccess) &&
            greedy_ledger.accepted_kernel_launches == 2U &&
            cudaDeviceSynchronize() == cudaSuccess,
        "fixed BF16 greedy launch failed");
    q3x::runtime::Bf16GreedyArgmaxResult result{};
    require_test(
        cudaMemcpy(
            &result,
            greedy_workspace.as<q3x::runtime::Bf16GreedyArgmaxResult>() +
                (bound_launch::kSm87MacroFeedV4GreedyWorkspaceResults - 1U),
            sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess &&
            result.index == kSelectedRow && result.value_bits == 0x4520U &&
            result.has_nonfinite == 0U,
        "greedy seam did not select the unique exact LM-head maximum");

    constexpr std::size_t kEarlierTie = 123U;
    constexpr std::size_t kNonfinite = 7U;
    const std::uint16_t tie_bits = 0x4520U;
    const std::uint16_t infinity_bits = 0x7f80U;
    require_test(
        cudaMemcpy(logits.as<std::uint16_t>() + kEarlierTie, &tie_bits,
                   sizeof(tie_bits), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(logits.as<std::uint16_t>() + kNonfinite,
                       &infinity_bits, sizeof(infinity_bits),
                       cudaMemcpyHostToDevice) == cudaSuccess,
        "could not install greedy tie/nonfinite fixture");
    require_test(cudaDeviceSynchronize() == cudaSuccess,
                 "greedy tie/nonfinite fixture did not physically complete");
    require_test(
        invoke_greedy(greedy_arguments, greedy_resources, &greedy_ledger) ==
                static_cast<int>(cudaSuccess) &&
            greedy_ledger.accepted_kernel_launches == 2U &&
            cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(
                &result,
                greedy_workspace.as<q3x::runtime::Bf16GreedyArgmaxResult>() +
                    (bound_launch::kSm87MacroFeedV4GreedyWorkspaceResults -
                     1U),
                sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess &&
            result.index == kEarlierTie && result.value_bits == tie_bits &&
            result.has_nonfinite == 1U,
        "greedy seam lost earliest-tie or nonfinite semantics");
  }
}

void test_real_cuda_front_half() {
  static_assert(
      execution::kSm87MacroFeedV4P40ExecutionPackageCompiled);
  static_assert(startup::kSm87MacroFeedV4P40StartupPackageCompiled);
  static_assert(!std::is_default_constructible_v<
                execution::Sm87MacroFeedV4P40ExecutionPackage>);
  static_assert(!std::is_copy_constructible_v<
                execution::Sm87MacroFeedV4P40ExecutionPackage>);
  static_assert(!std::is_move_constructible_v<
                execution::Sm87MacroFeedV4P40ExecutionPackage>);
  static_assert(!std::is_default_constructible_v<
                bound_launch::Sm87MacroFeedV4LockedSubmitToken>);
  static_assert(!std::is_copy_constructible_v<
                bound_launch::Sm87MacroFeedV4LockedSubmitToken>);
  static_assert(!std::is_move_constructible_v<
                bound_launch::Sm87MacroFeedV4LockedSubmitToken>);
  static_assert(!std::is_default_constructible_v<
                events::Sm87MacroFeedV4ExecutionEventsDriver>);
  static_assert(!std::is_copy_constructible_v<
                events::Sm87MacroFeedV4ExecutionEventsDriver>);
  static_assert(!std::is_move_constructible_v<
                events::Sm87MacroFeedV4ExecutionEventsDriver>);

  Owner owner;
  LiveExecutionWeights live_weights;
  q3x::tests::support::Sm87MacroFeedV4LiveFp8AssetFixture
      live_gdn_qkvz;
  q3x::tests::support::Sm87MacroFeedV4LiveFp8AssetFixture
      live_gdn_output;
  q3x::tests::support::Sm87MacroFeedV4LiveNvFp4AssetFixture
      live_gate_up;
  q3x::tests::support::Sm87MacroFeedV4LiveNvFp4AssetFixture live_down;
  LiveGdnContinuationWeights live_gdn_continuation;
  require_test(live_weights.allocate(),
               "could not allocate real BF16/LayerNorm CUDA fixtures");
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_test(model_weights.has_value(),
               "could not construct complete target-AOT fixture");
  require_test(live_weights.install(*model_weights),
               "could not install real execution weights");

  require_test(
      live_weights.install_one_past_final_post_norm(*model_weights),
      "could not install one-past LayerNorm negative fixture");
  auto invalid_startup =
      startup::Sm87MacroFeedV4P40StartupPackageHostTestFixture::create(
          *model_weights);
  if (!invalid_startup) {
    std::cerr << "negative startup error="
              << static_cast<unsigned>(invalid_startup.status.error)
              << " context=" << invalid_startup.status.context
              << " layer=" << invalid_startup.status.layer
              << " cuda=" << invalid_startup.status.cuda_error << '\n';
  }
  require_test(static_cast<bool>(invalid_startup),
               "negative fixture could not create its startup package");
  auto invalid_execution =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::create(
          *invalid_startup.package);
  const bool invalid_norm_rejected =
      !invalid_execution && invalid_execution.package == nullptr &&
      invalid_execution.status.error ==
          execution::Sm87MacroFeedV4P40ExecutionPackageError::
              kLayerNormCatalog &&
      invalid_execution.status.layer ==
          q3x::runtime::kQwen36DenseLayerCount - 1U &&
      invalid_execution.status.post_attention_norm &&
      invalid_execution.status.cuda_error != 0;
  if (!invalid_norm_rejected) {
    std::cerr << "negative execution error="
              << static_cast<unsigned>(invalid_execution.status.error)
              << " context=" << invalid_execution.status.context
              << " layer=" << invalid_execution.status.layer
              << " post_norm="
              << invalid_execution.status.post_attention_norm
              << " cuda=" << invalid_execution.status.cuda_error
              << " package=" << (invalid_execution.package != nullptr)
              << '\n';
  }
  require_test(
      invalid_norm_rejected,
      "one-past LayerNorm device range did not fail closed");
  invalid_startup.package.reset();
  require_test(live_weights.restore_final_post_norm(*model_weights),
               "could not restore final LayerNorm binding");

  auto startup_created =
      startup::Sm87MacroFeedV4P40StartupPackageHostTestFixture::create(
          *model_weights);
  if (!startup_created) {
    std::cerr << "startup package error="
              << static_cast<unsigned>(startup_created.status.error)
              << " context=" << startup_created.status.context
              << " cuda=" << startup_created.status.cuda_error << '\n';
  }
  require_test(static_cast<bool>(startup_created),
               "actual-kernel startup package creation failed");
  require_test(
      !startup_created.audit.request_boundary_normal_resident_authority &&
          startup_created.audit.request_boundary_host_test_resident_authority &&
          startup_created.audit.request_boundary_source_catalog_identity !=
              0U &&
          startup_created.audit.request_boundary_resident_root_identity !=
              0U &&
          startup_created.audit.request_boundary_live_device_ranges_complete &&
          !startup_created.audit
               .request_boundary_observed_resource_execution_catalog_sealed,
      "host Startup fixture did not retain a live, non-upgradable boundary source");

  auto fake_catalog_execution =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::create(
          *startup_created.package);
  require_test(
      !fake_catalog_execution && fake_catalog_execution.package == nullptr &&
          fake_catalog_execution.status.error ==
              execution::Sm87MacroFeedV4P40ExecutionPackageError::
                  kGdnQkvZCatalog &&
          fake_catalog_execution.status.layer == 0U &&
          fake_catalog_execution.status.cuda_error != 0,
      "fake target-AOT catalog acquired executable QKVZ authority");

  // Keep these live allocations after the one-past LayerNorm negative.  CUDA
  // is otherwise free to place a later allocation exactly at that one-past
  // address, turning the intended invalid pointer into an unrelated live
  // range and making the negative allocator-order dependent.
  require_test(
      live_gdn_qkvz.initialize(
          kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ, 0),
      "could not allocate honest synthetic-T1 GDN-QKVZ CUDA asset");
  require_test(
      live_gdn_qkvz.payload_allocation.bytes() == 83'886'080U,
      "synthetic-T1 GDN-QKVZ payload did not use the exact real shape");
  require_test(
      live_gdn_output.initialize(
          kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput, 0),
      "could not allocate honest synthetic-T1 GDN-output CUDA asset");
  require_test(
      live_gdn_output.payload_allocation.bytes() == 31'457'280U,
      "synthetic-T1 GDN-output payload did not use the exact real shape");
  require_test(
      live_gate_up.initialize(
          kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp, 0),
      "could not allocate honest synthetic-T1 Gate/Up CUDA asset");
  require_test(
      live_gate_up.payload_allocation.bytes() == 100'270'080U,
      "synthetic-T1 Gate/Up payload did not use the exact real shape");
  require_test(
      live_down.initialize(
          kernels::Sm87TargetAotProjectionRole::kNvFp4Down, 0),
      "could not allocate honest synthetic-T1 Down CUDA asset");
  require_test(
      live_down.payload_allocation.bytes() == 50'135'040U,
      "synthetic-T1 Down payload did not use the exact real shape");
  require_test(live_gdn_continuation.allocate() &&
                   live_gdn_continuation.bytes() == 82'368U,
               "could not allocate exact complete-layer GDN weights");

  constexpr std::uint64_t kNvFp4ValuesPerPartition =
      17'408ULL * 5'120ULL;
  constexpr std::uint64_t kNvFp4WeightBytesPerPartition =
      kNvFp4ValuesPerPartition / 2U;
  constexpr std::uint64_t kNvFp4BlockScalesPerPartition =
      kNvFp4ValuesPerPartition / 16U;
  const auto& gate_transform = live_gate_up.transform.partitions[0U];
  const auto& down_transform = live_down.transform.partitions[0U];
  require_test(
      gate_transform.source_weight_bytes_hashed ==
              kNvFp4WeightBytesPerPartition &&
          gate_transform.repacked_weight_values ==
              kNvFp4ValuesPerPartition &&
          gate_transform.repacked_block_scale_values ==
              kNvFp4BlockScalesPerPartition &&
          gate_transform.source_block_scale_e4m3fn_bytes_scanned ==
              kNvFp4BlockScalesPerPartition &&
          gate_transform.payload_block_scale_e4m3fn_bytes_scanned ==
              kNvFp4BlockScalesPerPartition &&
          gate_transform.bit_exact_block_scale_permutation &&
          down_transform.source_weight_bytes_hashed ==
              kNvFp4WeightBytesPerPartition &&
          down_transform.repacked_weight_values ==
              kNvFp4ValuesPerPartition &&
          down_transform.repacked_block_scale_values ==
              kNvFp4BlockScalesPerPartition &&
          down_transform.bit_exact_block_scale_permutation,
      "synthetic NVFP4 transform ledger lost exact weight/scale counts");

  const auto gate_up_receipt = make_gate_up_payload_receipt(live_gate_up);
  const auto down_receipt = make_down_payload_receipt(live_down);
  require_test(
      kernels::sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
          gate_up_receipt) &&
          kernels::sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
              down_receipt),
      "could not seal canonical V3 Gate/Up and Down payload receipts");

  auto terminal_drain_created =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          create_with_synthetic_t1_gdn_layer0(
              *startup_created.package, live_gdn_qkvz.asset);
  require_test(static_cast<bool>(terminal_drain_created),
               "terminal-drain execution package creation failed");
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          exercise_terminal_poison_drain(*terminal_drain_created.package),
      "tail-stage CUDA poison did not immediately drain all private streams");
  terminal_drain_created.package.reset();

  auto synthetic_full_composer_created =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          create_with_synthetic_t1_gdn_layer0(
              *startup_created.package, live_gdn_qkvz.asset);
  require_test(static_cast<bool>(synthetic_full_composer_created),
               "synthetic Full-composer negative package creation failed");
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          exercise_synthetic_full_attention_composer_fail_closed(
              *synthetic_full_composer_created.package),
      "synthetic package crossed the real-owner Full composer boundary");
  synthetic_full_composer_created.package.reset();

  auto synthetic_gdn_composer_created =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          create_with_synthetic_t1_gdn_layer0(
              *startup_created.package, live_gdn_qkvz.asset);
  require_test(static_cast<bool>(synthetic_gdn_composer_created),
               "synthetic GDN-composer negative package creation failed");
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          exercise_synthetic_normal_gdn_composer_fail_closed(
              *synthetic_gdn_composer_created.package),
      "synthetic package crossed the normal sealed-catalog GDN composer");
  synthetic_gdn_composer_created.package.reset();

  auto execution_created =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          create_with_synthetic_t1_gdn_layer0(
              *startup_created.package, live_gdn_qkvz.asset);
  if (!execution_created) {
    std::cerr << "execution package error="
              << static_cast<unsigned>(execution_created.status.error)
              << " context=" << execution_created.status.context
              << " layer=" << execution_created.status.layer
              << " post_norm="
              << execution_created.status.post_attention_norm
              << " cuda=" << execution_created.status.cuda_error << '\n';
  }
  require_test(static_cast<bool>(execution_created),
               "real CUDA execution package creation failed");

  const auto& audit = execution_created.audit;
  require_test(
      audit.valid() && audit.fixed_gdn_layer0_front_half_bound &&
          audit.qkvz_ab_ready_transaction_bound &&
          audit.synthetic_t1_gdn_layer0_source &&
          audit.request_boundary_catalog_identity == 0U &&
          audit.retained_request_boundary_catalog_fold_identity == 0U &&
          audit.request_boundary_source_catalog_identity == 0U &&
          audit.request_boundary_resource_bundle_identity == 0U &&
          audit.request_boundary_binding_identity == 0U &&
          audit.request_boundary_bindings == 0U &&
          audit.request_boundary_resource_queries == 0U &&
          !audit.request_boundary_execution_catalog_bound &&
          !audit.request_boundary_source_private_resource_queries &&
          !audit.request_boundary_normal_resident_authority &&
          !audit.request_boundary_host_test_resident_authority &&
          audit.request_boundary_synthetic_unbound &&
          audit.request_boundary_host_staging_allocation_identity == 0U &&
          audit.request_boundary_scratch_alias_identity == 0U &&
          audit.request_boundary_host_staging_begin == 0U &&
          audit.request_boundary_host_staging_bytes == 0U &&
          audit.request_boundary_host_owned_bytes == 0U &&
          audit.total_owned_bytes == audit.execution_owned_bytes &&
          audit.request_boundary_scratch_alias_span_bytes == 0U &&
          audit.request_boundary_host_staging_flags == 0U &&
          !audit.request_boundary_host_staging_pinned &&
          !audit.request_boundary_host_staging_construction_zero_initialized &&
          !audit.request_boundary_scratch_aliases_exact &&
          !audit.request_boundary_request_selectable &&
          !audit.request_boundary_launcher_authority &&
          !audit.request_boundary_production_dispatch_eligible &&
          audit.gdn_qkvz_catalog_identity == 0U &&
          audit.gdn_layer0_source_identity != 0U &&
          audit.gdn_qkvz_bindings == 1U &&
          !audit.whole_layer_executor_bound &&
          !audit.whole_model_executor_bound && !audit.selector_bound &&
          !audit.api_route_bound && audit.default_off &&
          !audit.jit_present && !audit.request_time_repack_present &&
          !audit.request_time_autotune_present && !audit.fallback_present &&
          !audit.cublaslt_present && !audit.mtp_present &&
          !audit.production_dispatch_eligible,
      "execution audit overstated the layer-0 front-half slice");
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          request_boundary_is_explicitly_unbound(
              *execution_created.package),
      "synthetic package acquired request-boundary catalog or storage authority");

  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::seed(
          *execution_created.package),
      "could not seed private execution-package buffers");
  const auto front_half =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          execute_once(*execution_created.package);
  if (!front_half) {
    std::cerr << "front-half error="
              << static_cast<unsigned>(front_half.status.error)
              << " context=" << front_half.status.context
              << " cuda=" << front_half.status.cuda_error << '\n';
  }
  require_test(static_cast<bool>(front_half) && front_half.receipt.valid(),
               "real CUDA front-half execution did not close its receipt");
  require_test(
      front_half.receipt.gdn_layer0_source_identity ==
              audit.gdn_layer0_source_identity &&
          front_half.receipt.gdn_qkvz_catalog_identity == 0U &&
          front_half.receipt.synthetic_t1_gdn_layer0_source &&
          front_half.receipt.input_norm_launches == 1U &&
          front_half.receipt.gdn_qkvz_launches == 1U &&
          front_half.receipt.bf16_ab_launches == 1U &&
          front_half.receipt.bound_kernel_submissions == 3U &&
          front_half.receipt.physical_completion_receipts == 1U &&
          front_half.receipt.norm_ready_recorded &&
          front_half.receipt.norm_ready_waited_by_ab &&
          front_half.receipt.ab_ready_recorded &&
          front_half.receipt.ab_ready_waited_by_main &&
          front_half.receipt.owner_drained_physically &&
          front_half.receipt.request_discarded_without_publication &&
          front_half.receipt.gdn_layer0_front_half_only &&
          !front_half.receipt.layer_complete &&
          !front_half.receipt.panel_complete &&
          !front_half.receipt.model_complete &&
          !front_half.receipt.production_dispatch_eligible,
      "front-half receipt overstated ordering, completion, or authority");

  execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::Samples
      samples{};
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          read_samples(*execution_created.package, &samples),
      "could not read physically drained CUDA samples");
  require_test(samples.normalized != 0U,
               "input normalization did not publish a nonzero sample");
  require_test(
      samples.gdn_qkvz == 0U && samples.projection_a == 0U &&
          samples.projection_b == 0U,
      "zero QKVZ/A/B weights produced a nonzero projection");
  require_test(samples.scratch_gap == 0x5a5aU,
               "QKVZ or A/B wrote outside its row-local scratch interval");
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          scratch_gap_untouched(*execution_created.package),
      "QKVZ or A/B modified an element in the 8000-row scratch gap");

  execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
      RequestOutcome request_outcome{};
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          read_request_outcome(*execution_created.package,
                               &request_outcome),
      "could not inspect the private request-state outcome");
  require_test(
      request_outcome.phase ==
              q3x::runtime::Sm87MacroFeedV4RequestStatePhase::kFailed &&
          !request_outcome.canonical_state_published &&
          !request_outcome.logical_sequence_fence_published &&
          !request_outcome.decode_access_issued,
      "front-half admission published canonical or Decode-visible state");

  const auto repeated =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          execute_once(*execution_created.package);
  require_test(
      !repeated &&
          repeated.status.error ==
              execution::Sm87MacroFeedV4P40ExecutionPackageError::
                  kAlreadyExecuted &&
          !repeated.receipt.valid(),
      "one-shot front-half package accepted a repeated execution");

  execution_created.package.reset();

  auto mismatched_gate_receipt = gate_up_receipt;
  ++mismatched_gate_receipt.payload_identity;
  mismatched_gate_receipt.receipt_identity =
      kernels::sm87_macrofeed_v3_nvfp4_gate_up_compute_payload_receipt_identity(
          mismatched_gate_receipt);
  require_test(
      kernels::sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
          mismatched_gate_receipt),
      "asset-mismatched Gate/Up negative receipt was not self-consistent");
  auto mismatched_receipt_execution =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          create_with_synthetic_t1_complete_gdn_layer0(
              *startup_created.package, live_gdn_qkvz.asset,
              live_gdn_output.asset, live_gate_up.asset, live_down.asset,
              live_gdn_continuation.conv_weight(),
              live_gdn_continuation.a_log(),
              live_gdn_continuation.dt_bias(),
              live_gdn_continuation.norm_weight(), mismatched_gate_receipt,
              down_receipt);
  require_test(
      !mismatched_receipt_execution &&
          mismatched_receipt_execution.package == nullptr &&
          mismatched_receipt_execution.status.error ==
              execution::Sm87MacroFeedV4P40ExecutionPackageError::
                  kCompleteLayerBinding,
      "synthetic Gate/Up receipt detached from its asset was admitted");

  auto overlapping_continuation_execution =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          create_with_synthetic_t1_complete_gdn_layer0(
              *startup_created.package, live_gdn_qkvz.asset,
              live_gdn_output.asset, live_gate_up.asset, live_down.asset,
              live_gdn_continuation.conv_weight(),
              live_gdn_continuation.a_log(),
              live_gdn_continuation.a_log(),
              live_gdn_continuation.norm_weight(), gate_up_receipt,
              down_receipt);
  require_test(
      !overlapping_continuation_execution &&
          overlapping_continuation_execution.package == nullptr &&
          overlapping_continuation_execution.status.error ==
              execution::Sm87MacroFeedV4P40ExecutionPackageError::
                  kCompleteLayerBinding,
      "overlapping synthetic GDN continuation roles were admitted");

  auto pending_grant_poison_execution =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          create_with_synthetic_t1_complete_gdn_layer0(
              *startup_created.package, live_gdn_qkvz.asset,
              live_gdn_output.asset, live_gate_up.asset, live_down.asset,
              live_gdn_continuation.conv_weight(),
              live_gdn_continuation.a_log(),
              live_gdn_continuation.dt_bias(),
              live_gdn_continuation.norm_weight(), gate_up_receipt,
              down_receipt);
  require_test(static_cast<bool>(pending_grant_poison_execution),
               "pending-grant poison package creation failed");
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          exercise_pending_gdn_grant_poison_drain(
              *pending_grant_poison_execution.package),
      "poison drain did not invalidate a pending GDN grant without publication");
  pending_grant_poison_execution.package.reset();

  auto complete_execution_created =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          create_with_synthetic_t1_complete_gdn_layer0(
              *startup_created.package, live_gdn_qkvz.asset,
              live_gdn_output.asset, live_gate_up.asset, live_down.asset,
              live_gdn_continuation.conv_weight(),
              live_gdn_continuation.a_log(),
              live_gdn_continuation.dt_bias(),
              live_gdn_continuation.norm_weight(), gate_up_receipt,
              down_receipt);
  if (!complete_execution_created) {
    std::cerr << "complete execution package error="
              << static_cast<unsigned>(
                     complete_execution_created.status.error)
              << " context=" << complete_execution_created.status.context
              << " layer=" << complete_execution_created.status.layer
              << " cuda=" << complete_execution_created.status.cuda_error
              << '\n';
  }
  require_test(static_cast<bool>(complete_execution_created),
               "complete GDN layer-0 package creation failed");
  const auto& complete_audit = complete_execution_created.audit;
  require_test(
      complete_audit.valid() &&
          complete_audit.fixed_gdn_layer0_front_half_bound &&
          complete_audit.fixed_gdn_layer0_complete_bound &&
          complete_audit.qkvz_ab_ready_transaction_bound &&
          complete_audit.synthetic_t1_gdn_layer0_source &&
          complete_audit.gdn_qkvz_catalog_identity == 0U &&
          complete_audit.mlp_pair_catalog_identity == 0U &&
          complete_audit.gdn_qkvz_bindings == 1U &&
          complete_audit.mlp_pair_bindings == 1U &&
          complete_audit.whole_layer_executor_bound &&
          !complete_audit.whole_model_executor_bound &&
          !complete_audit.selector_bound && !complete_audit.api_route_bound &&
          complete_audit.default_off && !complete_audit.jit_present &&
          !complete_audit.request_time_repack_present &&
          !complete_audit.request_time_autotune_present &&
          !complete_audit.fallback_present && !complete_audit.cublaslt_present &&
          !complete_audit.mtp_present &&
          !complete_audit.production_dispatch_eligible,
      "complete-layer audit overstated source or production authority");
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::seed(
          *complete_execution_created.package),
      "could not seed complete-layer private buffers");

  const auto complete =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          execute_complete_once(*complete_execution_created.package);
  if (!complete) {
    std::cerr << "complete layer error="
              << static_cast<unsigned>(complete.status.error)
              << " context=" << complete.status.context
              << " cuda=" << complete.status.cuda_error << '\n';
  }
  require_test(static_cast<bool>(complete) && complete.receipt.valid(),
               "complete GDN layer-0 execution did not close its receipt");
  require_test(
      complete.receipt.gdn_layer0_source_identity ==
              complete_audit.gdn_layer0_source_identity &&
          complete.receipt.gdn_qkvz_catalog_identity == 0U &&
          complete.receipt.mlp_pair_catalog_identity == 0U &&
          complete.receipt.synthetic_t1_gdn_layer0_source &&
          complete.receipt.input_norm_launches == 1U &&
          complete.receipt.bf16_ab_launches == 1U &&
          complete.receipt.gdn_qkvz_launches == 1U &&
          complete.receipt.gdn_continuation_launches == 2U &&
          complete.receipt.gdn_output_launches == 1U &&
          complete.receipt.residual_post_norm_launches == 1U &&
          complete.receipt.gate_up_launches == 1U &&
          complete.receipt.down_launches == 1U &&
          complete.receipt.bound_kernel_submissions == 9U &&
          complete.receipt.asynchronous_d2d_copies == 1U &&
          complete.receipt.conv_history_copy_bytes == 61'440U &&
          complete.receipt.enqueue_receipt_owner_matched &&
          complete.receipt.enqueue_receipt.authority_domain() ==
              events::Sm87MacroFeedV4GdnSubmissionAuthorityDomain::
                  kSyntheticT1 &&
          complete.receipt.enqueue_receipt.execution_package_identity() ==
              complete.receipt.package_identity &&
          complete.receipt.enqueue_receipt.gdn_catalog_identity() == 0U &&
          complete.receipt.enqueue_receipt.gdn_binding_identity() == 0U &&
          complete.receipt.enqueue_receipt.mlp_catalog_identity() == 0U &&
          complete.receipt.enqueue_receipt.mlp_binding_identity() == 0U &&
          complete.receipt.enqueue_receipt.resource_bundle_identity() == 0U &&
          complete.receipt.enqueue_receipt.synthetic_source_identity() ==
              complete_audit.gdn_layer0_source_identity &&
          complete.receipt.enqueue_receipt.bf16_ab_catalog_identity() != 0U &&
          complete.receipt.enqueue_receipt.bf16_ab_pair_identity() != 0U &&
          complete.receipt.enqueue_receipt.layer_norm_catalog_identity() !=
              0U &&
          complete.receipt.enqueue_receipt.layer_norm_pair_identity() != 0U &&
          complete.receipt.physical_owner_drain_receipt_identity != 0U &&
          complete.receipt.physical_completion_receipts == 1U &&
          complete.receipt.norm_ready_waited_by_ab &&
          complete.receipt.ab_ready_waited_by_main &&
          complete.receipt.layer_complete &&
          complete.receipt.state_candidate_recorded &&
          complete.receipt.owner_drained_physically &&
          complete.receipt.physical_execution_receipt_issued &&
          complete.receipt.candidate_discarded_without_publication &&
          complete.receipt.state_epoch_after ==
              complete.receipt.state_epoch_before &&
          complete.receipt.active_bank_after ==
              complete.receipt.active_bank_before &&
          complete.receipt.candidate_bank_after ==
              complete.receipt.candidate_bank_before &&
          !complete.receipt.panel_complete &&
          !complete.receipt.model_complete &&
          !complete.receipt.production_dispatch_eligible,
      "complete-layer receipt did not prove exactly 9 kernels plus 1 D2D");

  execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
      RequestOutcome complete_request_outcome{};
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          read_request_outcome(*complete_execution_created.package,
                               &complete_request_outcome),
      "could not inspect complete-layer request-state outcome");
  require_test(
      complete_request_outcome.phase ==
              q3x::runtime::Sm87MacroFeedV4RequestStatePhase::kFailed &&
          !complete_request_outcome.canonical_state_published &&
          !complete_request_outcome.logical_sequence_fence_published &&
          !complete_request_outcome.decode_access_issued,
      "complete synthetic layer published canonical or Decode-visible state");

  const auto complete_repeated =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          execute_complete_once(*complete_execution_created.package);
  require_test(
      !complete_repeated &&
          complete_repeated.status.error ==
              execution::Sm87MacroFeedV4P40ExecutionPackageError::
                  kAlreadyExecuted &&
          !complete_repeated.receipt.valid(),
      "one-shot complete-layer package accepted a repeated execution");
  complete_execution_created.package.reset();

  auto startup_independent_snapshot =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          create_with_synthetic_t1_gdn_layer0(
              *startup_created.package, live_gdn_qkvz.asset);
  require_test(static_cast<bool>(startup_independent_snapshot),
               "startup-independent projection snapshot creation failed");
  startup_created.package.reset();
  require_test(startup_independent_snapshot.package->valid(),
               "execution package retained startup-owned catalog pointers");
  startup_independent_snapshot.package.reset();
  require_test(clear_fixture(model_weights, owner),
               "complete target-AOT fixture cleanup failed");
}

}  // namespace

int main() {
  if (!exact_sm87_device_available()) {
    std::cout << "sm87_macrofeed_v4_p40_execution_package_cuda_test: "
                 "SKIP (requires device 0 with SM87 and exactly 16 SMs)\n";
    return 77;
  }
  test_real_cuda_front_half();
  // Keep the multi-gigabyte constituent fixture last so its allocator history
  // cannot perturb the execution package's construction-time reserve gates.
  test_fixed_outer_request_boundaries();
  std::cout <<
      "sm87_macrofeed_v4_p40_execution_package_cuda_test: PASS\n";
  return 0;
}
