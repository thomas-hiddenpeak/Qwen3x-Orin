#pragma once

#include "sm87_macrofeed_v4_execution_events_internal.h"
#include "sm87_macrofeed_v4_p40_startup_package_internal.h"
#include "sm87_macrofeed_v4_request_state_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail {

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_EXECUTION_EVENTS_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_NORM_RESIDUAL_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FULL_ATTENTION_PREPROCESS_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_ATTENTION_C8000_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_GDN_C8000_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_NVFP4_GATE_UP_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_NVFP4_DOWN_ADMISSION)
inline constexpr bool kSm87MacroFeedV4P40ExecutionPackageCompiled = true;
#else
inline constexpr bool kSm87MacroFeedV4P40ExecutionPackageCompiled = false;
#endif

inline constexpr std::uint64_t
    kSm87MacroFeedV4P40ExecutionTransientBytes =
        2U * kernels::kSm87MacroFeedV4NormResidualHiddenBytes +
        kernels::kSm87MacroFeedV4Bf16AbScratchBytes;
inline constexpr std::uint64_t kSm87MacroFeedV4P40ExecutionPingOffset = 0U;
inline constexpr std::uint64_t kSm87MacroFeedV4P40ExecutionPongOffset =
    kernels::kSm87MacroFeedV4NormResidualHiddenBytes;
inline constexpr std::uint64_t kSm87MacroFeedV4P40ExecutionScratchOffset =
    2U * kernels::kSm87MacroFeedV4NormResidualHiddenBytes;
inline constexpr std::uint64_t kSm87MacroFeedV4P40ExecutionOwnedBytes =
    kSm87MacroFeedV4P40ExecutionTransientBytes +
    kSm87MacroFeedV4RecurrentStorageBytes +
    kSm87MacroFeedV4AttentionKvArenaBytes;
inline constexpr std::size_t kSm87MacroFeedV4P40EngineRopePositions =
    262'144U;
inline constexpr std::size_t kSm87MacroFeedV4P40EngineRopePairs = 32U;
inline constexpr std::uint64_t kSm87MacroFeedV4P40EngineRopeTableBytes =
    static_cast<std::uint64_t>(kSm87MacroFeedV4P40EngineRopePositions) *
    kSm87MacroFeedV4P40EngineRopePairs * sizeof(float);
inline constexpr std::uint64_t kSm87MacroFeedV4P40EngineRopeAllocationBytes =
    2U * kSm87MacroFeedV4P40EngineRopeTableBytes;

static_assert(kSm87MacroFeedV4P40ExecutionTransientBytes == 442'368'000U);
static_assert(kSm87MacroFeedV4P40ExecutionScratchOffset == 163'840'000U);
static_assert(kSm87MacroFeedV4P40ExecutionOwnedBytes == 3'220'701'184U);
static_assert(kSm87MacroFeedV4P40EngineRopeTableBytes == 33'554'432U);
static_assert(kSm87MacroFeedV4P40EngineRopeAllocationBytes == 67'108'864U);
static_assert(kSm87MacroFeedV4P40EngineRopePositions >=
              kSm87MacroFeedV4P40Tokens);

// V4-private, non-owning view of the Engine RoPE owner.  The normal factory
// accepts this value only from its Engine composition-root friend, validates
// the complete live contiguous owner allocation, and then retains the sealed
// value for Full-Attention execution.  Copying this POD grants no ownership,
// launch authority, selector, or API route.
struct Sm87MacroFeedV4P40EngineRopeBinding final {
  const float* cosines = nullptr;
  const float* sines = nullptr;
  std::size_t position_count = 0U;
  std::size_t rotary_pairs = 0U;
  std::uint64_t owner_identity = 0U;
  std::uintptr_t allocation_begin = 0U;
  std::uint64_t allocation_bytes = 0U;
  std::int32_t device_ordinal = -1;
};

enum class Sm87MacroFeedV4P40ExecutionPackageError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kStartupPackage,
  kProjectionCatalog,
  kBf16AbCatalog,
  kLayerNormCatalog,
  kGdnQkvZCatalog,
  kMlpPairCatalog,
  kFullAttentionResources,
  kFullAttentionCatalog,
  kNormResources,
  kGdnResources,
  kGateUpResources,
  kDownResources,
  kExecutionEvents,
  kEngineRope,
  kMemoryReserve,
  kKvAllocation,
  kTransientAllocation,
  kRecurrentAllocation,
  kColdRecurrentInitialization,
  kRequestState,
  kPackageAllocation,
  kAlreadyExecuted,
  kFrontHalfBinding,
  kCompleteLayerBinding,
  kGdnLayerStateGrant,
  kFullAttentionKvGrant,
  kExecutionEvent,
  kPhysicalDrain,
  kRequestAbort,
};

struct Sm87MacroFeedV4P40ExecutionPackageStatus final {
  Sm87MacroFeedV4P40ExecutionPackageError error =
      Sm87MacroFeedV4P40ExecutionPackageError::kNone;
  const char* context = "none";
  int cuda_error = 0;
  std::size_t layer = kSm87MacroFeedV4LayerCount;
  bool post_attention_norm = false;
  sm87_macrofeed_v4_execution_events_detail::
      Sm87MacroFeedV4ExecutionStatus event_status{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return error == Sm87MacroFeedV4P40ExecutionPackageError::kNone;
  }
};

struct Sm87MacroFeedV4P40ExecutionPackageAudit final {
  std::uint64_t package_identity = 0U;
  std::uint64_t startup_package_identity = 0U;
  std::uint64_t projection_catalog_identity = 0U;
  std::uint64_t bf16_ab_catalog_identity = 0U;
  std::uint64_t layer_norm_catalog_identity = 0U;
  std::uint64_t gdn_qkvz_catalog_identity = 0U;
  std::uint64_t mlp_pair_catalog_identity = 0U;
  std::uint64_t full_attention_catalog_identity = 0U;
  std::uint64_t retained_gdn_layer_catalog_fold_identity = 0U;
  std::uint64_t retained_mlp_pair_catalog_fold_identity = 0U;
  std::uint64_t retained_full_attention_catalog_fold_identity = 0U;
  std::uint64_t full_attention_resource_bundle_identity = 0U;
  std::uint64_t gdn_layer0_source_identity = 0U;
  std::uint64_t transient_allocation_identity = 0U;
  std::uint64_t recurrent_allocation_identity = 0U;
  std::uint64_t kv_allocation_identity = 0U;
  std::uint64_t request_state_kv_allocation_identity = 0U;
  std::uint64_t engine_rope_owner_identity = 0U;
  std::uint64_t engine_rope_binding_identity = 0U;
  std::uint64_t execution_events_owner_identity = 0U;
  std::uintptr_t kv_allocation_begin = 0U;
  std::uintptr_t engine_rope_allocation_begin = 0U;
  std::int32_t device_ordinal = -1;
  std::size_t projection_bindings = 0U;
  std::size_t bf16_ab_pairs = 0U;
  std::size_t layer_norm_pairs = 0U;
  std::size_t gdn_qkvz_bindings = 0U;
  std::size_t mlp_pair_bindings = 0U;
  std::size_t full_attention_bindings = 0U;
  std::size_t engine_rope_positions = 0U;
  std::size_t engine_rope_pairs = 0U;
  std::uint64_t transient_bytes = 0U;
  std::uint64_t recurrent_bytes = 0U;
  std::uint64_t kv_allocation_bytes = 0U;
  std::uint64_t request_state_kv_allocation_bytes = 0U;
  std::uint64_t engine_rope_allocation_bytes = 0U;
  std::uint64_t execution_owned_bytes = 0U;
  std::uint64_t required_device_allocation_bytes = 0U;
  std::uint64_t minimum_free_bytes_after_create = 0U;
  std::uint64_t device_free_bytes_before_allocations = 0U;
  std::uint64_t device_free_bytes_after_allocations = 0U;
  std::uint64_t cold_recurrent_zero_bytes = 0U;
  std::size_t cold_recurrent_initializations = 0U;
  bool request_state_kv_physical_owner_bound = false;
  bool aggregate_memory_gate_passed = false;
  bool fixed_gdn_layer0_front_half_bound = false;
  bool fixed_gdn_layer0_complete_bound = false;
  bool qkvz_ab_ready_transaction_bound = false;
  bool synthetic_t1_gdn_layer0_source = false;
  bool whole_layer_executor_bound = true;
  bool whole_model_executor_bound = true;
  bool selector_bound = true;
  bool api_route_bound = true;
  bool default_off = false;
  bool jit_present = true;
  bool request_time_repack_present = true;
  bool request_time_autotune_present = true;
  bool fallback_present = true;
  bool cublaslt_present = true;
  bool mtp_present = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    const bool real_complete_catalogs =
        !synthetic_t1_gdn_layer0_source &&
        gdn_qkvz_catalog_identity != 0U &&
        retained_gdn_layer_catalog_fold_identity != 0U &&
        gdn_qkvz_bindings == kSm87MacroFeedV4StateLayerCount &&
        mlp_pair_catalog_identity != 0U &&
        retained_mlp_pair_catalog_fold_identity != 0U &&
        mlp_pair_bindings == kSm87MacroFeedV4LayerCount &&
        fixed_gdn_layer0_complete_bound;
    const bool real_full_attention_owner =
        !synthetic_t1_gdn_layer0_source &&
        full_attention_catalog_identity != 0U &&
        retained_full_attention_catalog_fold_identity != 0U &&
        full_attention_resource_bundle_identity != 0U &&
        full_attention_bindings == kSm87MacroFeedV4FullAttentionLayerCount &&
        kv_allocation_identity != 0U && kv_allocation_begin != 0U &&
        kv_allocation_bytes == kSm87MacroFeedV4AttentionKvArenaBytes &&
        request_state_kv_allocation_identity == kv_allocation_identity &&
        request_state_kv_allocation_bytes == kv_allocation_bytes &&
        request_state_kv_physical_owner_bound &&
        engine_rope_owner_identity != 0U &&
        engine_rope_binding_identity != 0U &&
        engine_rope_allocation_begin != 0U &&
        engine_rope_positions == kSm87MacroFeedV4P40EngineRopePositions &&
        engine_rope_pairs == kSm87MacroFeedV4P40EngineRopePairs &&
        engine_rope_allocation_bytes ==
            kSm87MacroFeedV4P40EngineRopeAllocationBytes &&
        execution_owned_bytes == kSm87MacroFeedV4P40ExecutionOwnedBytes &&
        required_device_allocation_bytes == execution_owned_bytes &&
        aggregate_memory_gate_passed &&
        device_free_bytes_before_allocations >=
            required_device_allocation_bytes &&
        minimum_free_bytes_after_create <=
            device_free_bytes_before_allocations -
                required_device_allocation_bytes &&
        device_free_bytes_after_allocations >=
            minimum_free_bytes_after_create;
    const bool synthetic_no_full_attention_owner =
        synthetic_t1_gdn_layer0_source &&
        full_attention_catalog_identity == 0U &&
        retained_full_attention_catalog_fold_identity == 0U &&
        full_attention_resource_bundle_identity == 0U &&
        full_attention_bindings == 0U && kv_allocation_identity == 0U &&
        kv_allocation_begin == 0U && kv_allocation_bytes == 0U &&
        request_state_kv_allocation_identity != 0U &&
        request_state_kv_allocation_bytes ==
            kSm87MacroFeedV4AttentionKvArenaBytes &&
        !request_state_kv_physical_owner_bound &&
        engine_rope_owner_identity == 0U &&
        engine_rope_binding_identity == 0U &&
        engine_rope_allocation_begin == 0U && engine_rope_positions == 0U &&
        engine_rope_pairs == 0U && engine_rope_allocation_bytes == 0U &&
        execution_owned_bytes ==
            kSm87MacroFeedV4P40ExecutionTransientBytes +
                kSm87MacroFeedV4RecurrentStorageBytes &&
        required_device_allocation_bytes == 0U &&
        minimum_free_bytes_after_create == 0U &&
        device_free_bytes_before_allocations == 0U &&
        device_free_bytes_after_allocations == 0U &&
        !aggregate_memory_gate_passed;
    const bool synthetic_front_half_source =
        synthetic_t1_gdn_layer0_source &&
        gdn_qkvz_catalog_identity == 0U && gdn_qkvz_bindings == 1U &&
        mlp_pair_catalog_identity == 0U &&
        retained_gdn_layer_catalog_fold_identity == 0U &&
        retained_mlp_pair_catalog_fold_identity == 0U &&
        mlp_pair_bindings == 0U &&
        !fixed_gdn_layer0_complete_bound;
    const bool synthetic_complete_source =
        synthetic_t1_gdn_layer0_source &&
        gdn_qkvz_catalog_identity == 0U && gdn_qkvz_bindings == 1U &&
        mlp_pair_catalog_identity == 0U &&
        retained_gdn_layer_catalog_fold_identity == 0U &&
        retained_mlp_pair_catalog_fold_identity == 0U &&
        mlp_pair_bindings == 1U &&
        fixed_gdn_layer0_complete_bound;
    return package_identity != 0U && startup_package_identity != 0U &&
           projection_catalog_identity != 0U &&
           bf16_ab_catalog_identity != 0U &&
           layer_norm_catalog_identity != 0U &&
           gdn_layer0_source_identity != 0U &&
           (real_complete_catalogs || synthetic_front_half_source ||
            synthetic_complete_source) &&
           (real_full_attention_owner ||
            synthetic_no_full_attention_owner) &&
           transient_allocation_identity != 0U &&
           recurrent_allocation_identity != 0U &&
           execution_events_owner_identity != 0U && device_ordinal >= 0 &&
           projection_bindings ==
               sm87_macrofeed_v4_p40_startup_package_detail::
                   kSm87MacroFeedV4P40StartupPackageArtifacts &&
           bf16_ab_pairs == kSm87MacroFeedV4StateLayerCount &&
           layer_norm_pairs == kSm87MacroFeedV4LayerCount &&
           transient_bytes == kSm87MacroFeedV4P40ExecutionTransientBytes &&
           recurrent_bytes == kSm87MacroFeedV4RecurrentStorageBytes &&
           cold_recurrent_zero_bytes == recurrent_bytes &&
           cold_recurrent_initializations == 1U &&
           fixed_gdn_layer0_front_half_bound &&
           qkvz_ab_ready_transaction_bound &&
           whole_layer_executor_bound == fixed_gdn_layer0_complete_bound &&
           !whole_model_executor_bound &&
           !selector_bound && !api_route_bound && default_off &&
           !jit_present && !request_time_repack_present &&
           !request_time_autotune_present && !fallback_present &&
           !cublaslt_present && !mtp_present &&
           !production_dispatch_eligible;
  }
};

struct Sm87MacroFeedV4GdnLayer0FrontHalfReceipt final {
  std::uint64_t receipt_identity = 0U;
  std::uint64_t package_identity = 0U;
  std::uint64_t gdn_layer0_source_identity = 0U;
  std::uint64_t gdn_qkvz_catalog_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::size_t model_layer = kSm87MacroFeedV4LayerCount;
  std::size_t input_norm_launches = 0U;
  std::size_t gdn_qkvz_launches = 0U;
  std::size_t bf16_ab_launches = 0U;
  std::size_t bound_kernel_submissions = 0U;
  std::size_t physical_completion_receipts = 0U;
  bool norm_ready_recorded = false;
  bool norm_ready_waited_by_ab = false;
  bool ab_ready_recorded = false;
  bool ab_ready_waited_by_main = true;
  bool owner_drained_physically = false;
  bool request_discarded_without_publication = false;
  bool gdn_layer0_front_half_only = false;
  bool synthetic_t1_gdn_layer0_source = false;
  bool layer_complete = true;
  bool panel_complete = true;
  bool model_complete = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    const bool source_provenance_valid =
        gdn_layer0_source_identity != 0U &&
        ((synthetic_t1_gdn_layer0_source &&
          gdn_qkvz_catalog_identity == 0U) ||
         (!synthetic_t1_gdn_layer0_source &&
          gdn_qkvz_catalog_identity != 0U));
    return receipt_identity != 0U && package_identity != 0U &&
           source_provenance_valid && request_epoch != 0U && panel == 0U &&
           model_layer == 0U &&
           input_norm_launches == 1U && gdn_qkvz_launches == 1U &&
           bf16_ab_launches == 1U && bound_kernel_submissions == 3U &&
           physical_completion_receipts == 1U && norm_ready_recorded &&
           norm_ready_waited_by_ab && ab_ready_recorded &&
           ab_ready_waited_by_main && owner_drained_physically &&
           request_discarded_without_publication &&
           gdn_layer0_front_half_only && !layer_complete && !panel_complete &&
           !model_complete && !production_dispatch_eligible;
  }
};

struct Sm87MacroFeedV4GdnLayer0FrontHalfResult final {
  Sm87MacroFeedV4P40ExecutionPackageStatus status{};
  Sm87MacroFeedV4GdnLayer0FrontHalfReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.valid();
  }
};

// One physically drained, exact layer-0 GDN decoder layer.  This receipt is
// intentionally narrower than a panel/model completion: it proves the fixed
// nine-kernel plus one-D2D enqueue, state-grant commit, terminal drain, and
// candidate discard, but grants no publication or production-route authority.
struct Sm87MacroFeedV4GdnLayer0CompleteReceipt final {
  std::uint64_t receipt_identity = 0U;
  std::uint64_t package_identity = 0U;
  std::uint64_t gdn_layer0_source_identity = 0U;
  std::uint64_t gdn_qkvz_catalog_identity = 0U;
  std::uint64_t mlp_pair_catalog_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::uint64_t state_epoch_before = 0U;
  std::uint64_t state_epoch_after = 0U;
  std::uint64_t state_grant_identity = 0U;
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::size_t model_layer = kSm87MacroFeedV4LayerCount;
  std::size_t active_bank_before = 2U;
  std::size_t active_bank_after = 2U;
  std::size_t candidate_bank_before = 2U;
  std::size_t candidate_bank_after = 2U;
  std::size_t input_norm_launches = 0U;
  std::size_t bf16_ab_launches = 0U;
  std::size_t gdn_qkvz_launches = 0U;
  std::size_t gdn_continuation_launches = 0U;
  std::size_t gdn_output_launches = 0U;
  std::size_t residual_post_norm_launches = 0U;
  std::size_t gate_up_launches = 0U;
  std::size_t down_launches = 0U;
  std::size_t bound_kernel_submissions = 0U;
  std::size_t asynchronous_d2d_copies = 0U;
  std::uint64_t conv_history_copy_bytes = 0U;
  sm87_macrofeed_v4_execution_events_detail::
      Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt enqueue_receipt{};
  bool enqueue_receipt_owner_matched = false;
  std::uint64_t physical_owner_drain_receipt_identity = 0U;
  std::size_t physical_completion_receipts = 0U;
  bool norm_ready_waited_by_ab = false;
  bool ab_ready_waited_by_main = false;
  bool layer_complete = false;
  bool state_candidate_recorded = false;
  bool owner_drained_physically = false;
  bool physical_execution_receipt_issued = false;
  bool candidate_discarded_without_publication = false;
  bool synthetic_t1_gdn_layer0_source = false;
  bool panel_complete = false;
  bool model_complete = false;
  bool production_dispatch_eligible = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    const bool source_provenance_valid =
        gdn_layer0_source_identity != 0U &&
        ((synthetic_t1_gdn_layer0_source &&
          gdn_qkvz_catalog_identity == 0U &&
          mlp_pair_catalog_identity == 0U) ||
         (!synthetic_t1_gdn_layer0_source &&
          gdn_qkvz_catalog_identity != 0U &&
          mlp_pair_catalog_identity != 0U));
    return receipt_identity != 0U && package_identity != 0U &&
           source_provenance_valid && request_epoch != 0U &&
           state_epoch_after == state_epoch_before &&
           state_grant_identity != 0U && panel == 0U && model_layer == 0U &&
           active_bank_before < 2U && active_bank_after == active_bank_before &&
           candidate_bank_before < 2U &&
           candidate_bank_after == candidate_bank_before &&
           candidate_bank_before != active_bank_before &&
           input_norm_launches == 1U && bf16_ab_launches == 1U &&
           gdn_qkvz_launches == 1U && gdn_continuation_launches == 2U &&
           gdn_output_launches == 1U &&
           residual_post_norm_launches == 1U && gate_up_launches == 1U &&
           down_launches == 1U && bound_kernel_submissions == 9U &&
           asynchronous_d2d_copies == 1U &&
           conv_history_copy_bytes ==
               kernels::kSm87MacroFeedV4GdnConvHistoryBytes &&
           enqueue_receipt.valid_shape() &&
           enqueue_receipt.authority_domain() ==
               sm87_macrofeed_v4_execution_events_detail::
                   Sm87MacroFeedV4GdnSubmissionAuthorityDomain::kSyntheticT1 &&
           enqueue_receipt.execution_package_identity() == package_identity &&
           enqueue_receipt.gdn_catalog_identity() == 0U &&
           enqueue_receipt.gdn_binding_identity() == 0U &&
           enqueue_receipt.mlp_catalog_identity() == 0U &&
           enqueue_receipt.mlp_binding_identity() == 0U &&
           enqueue_receipt.resource_bundle_identity() == 0U &&
           enqueue_receipt.synthetic_source_identity() ==
               gdn_layer0_source_identity &&
           enqueue_receipt.request_epoch() == request_epoch &&
           enqueue_receipt.panel() == panel &&
           enqueue_receipt.grant_identity() == state_grant_identity &&
           enqueue_receipt.grant_state_epoch() == state_epoch_before &&
           enqueue_receipt.recurrent_allocation_identity() != 0U &&
           enqueue_receipt.gdn_ordinal() == 0U &&
           enqueue_receipt.model_layer() == model_layer &&
           enqueue_receipt.active_bank_index() == active_bank_before &&
           enqueue_receipt.candidate_bank_index() ==
               candidate_bank_before &&
           enqueue_receipt.active_conv_allocation_offset() ==
               static_cast<std::uint64_t>(active_bank_before) *
                   kSm87MacroFeedV4RecurrentEpochBytes &&
           enqueue_receipt.candidate_conv_allocation_offset() ==
               static_cast<std::uint64_t>(candidate_bank_before) *
                   kSm87MacroFeedV4RecurrentEpochBytes &&
           enqueue_receipt.conv_bytes() == conv_history_copy_bytes &&
           enqueue_receipt.active_gdn_state_allocation_offset() ==
               static_cast<std::uint64_t>(active_bank_before) *
                       kSm87MacroFeedV4RecurrentEpochBytes +
                   kSm87MacroFeedV4ConvEpochBytes &&
           enqueue_receipt.candidate_gdn_state_allocation_offset() ==
               static_cast<std::uint64_t>(candidate_bank_before) *
                       kSm87MacroFeedV4RecurrentEpochBytes +
                   kSm87MacroFeedV4ConvEpochBytes &&
           enqueue_receipt.gdn_state_bytes() ==
               kernels::kSm87MacroFeedV4GdnStateBytes &&
           enqueue_receipt.input_norm_launches() == input_norm_launches &&
           enqueue_receipt.bf16_ab_launches() == bf16_ab_launches &&
           enqueue_receipt.gdn_qkvz_launches() == gdn_qkvz_launches &&
           enqueue_receipt.gdn_continuation_launches() ==
               gdn_continuation_launches &&
           enqueue_receipt.gdn_output_launches() == gdn_output_launches &&
           enqueue_receipt.residual_post_norm_launches() ==
               residual_post_norm_launches &&
           enqueue_receipt.gate_up_launches() == gate_up_launches &&
           enqueue_receipt.down_launches() == down_launches &&
           enqueue_receipt.bound_kernel_submissions() == 9U &&
           enqueue_receipt.asynchronous_d2d_copies() == 1U &&
           enqueue_receipt.conv_history_copy_bytes() ==
               conv_history_copy_bytes &&
           enqueue_receipt.norm_ready_waited_by_ab() &&
           enqueue_receipt.ab_ready_waited_by_main() &&
           enqueue_receipt.complete_layer_enqueued() &&
           !enqueue_receipt.physical_device_completion_attested() &&
           !enqueue_receipt.panel_complete() &&
           !enqueue_receipt.production_receipt_eligible() &&
           enqueue_receipt_owner_matched &&
           physical_owner_drain_receipt_identity != 0U &&
           physical_completion_receipts == 1U &&
           norm_ready_waited_by_ab && ab_ready_waited_by_main &&
           layer_complete && state_candidate_recorded &&
           owner_drained_physically && physical_execution_receipt_issued &&
           candidate_discarded_without_publication && !panel_complete &&
           !model_complete && !production_dispatch_eligible;
  }
};

struct Sm87MacroFeedV4GdnLayer0CompleteResult final {
  Sm87MacroFeedV4P40ExecutionPackageStatus status{};
  Sm87MacroFeedV4GdnLayer0CompleteReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.valid();
  }
};

// Package-composed enqueue-and-commit evidence for one naturally ordered GDN
// layer.  The nested opaque Events receipt is the owner-authenticated fact;
// these outer fields only bind it to the construction-sealed package catalogs
// and the RequestState move-only grant commit.  Success neither drains the
// streams nor publishes recurrent, panel, model, Decode, or production state.
struct Sm87MacroFeedV4GdnLayerCommitReceipt final {
  std::uint64_t package_identity = 0U;
  std::uint64_t gdn_catalog_identity = 0U;
  std::uint64_t gdn_binding_identity = 0U;
  std::uint64_t bf16_ab_catalog_identity = 0U;
  std::uint64_t bf16_ab_pair_identity = 0U;
  std::uint64_t layer_norm_catalog_identity = 0U;
  std::uint64_t layer_norm_pair_identity = 0U;
  std::uint64_t input_norm_binding_identity = 0U;
  std::uint64_t post_norm_binding_identity = 0U;
  std::uint64_t mlp_catalog_identity = 0U;
  std::uint64_t mlp_binding_identity = 0U;
  std::uint64_t resource_bundle_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::uint64_t grant_identity = 0U;
  std::uint64_t grant_state_epoch = 0U;
  std::uint64_t recurrent_allocation_identity = 0U;
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::size_t gdn_ordinal = kSm87MacroFeedV4StateLayerCount;
  std::size_t model_layer = kSm87MacroFeedV4LayerCount;
  std::size_t active_bank_index = 2U;
  std::size_t candidate_bank_index = 2U;
  std::uint64_t active_conv_allocation_offset = 0U;
  std::uint64_t candidate_conv_allocation_offset = 0U;
  std::uint64_t conv_bytes = 0U;
  std::uint64_t active_gdn_state_allocation_offset = 0U;
  std::uint64_t candidate_gdn_state_allocation_offset = 0U;
  std::uint64_t gdn_state_bytes = 0U;
  std::size_t next_model_layer_after_commit = 0U;
  std::size_t panel_conv_layers_prepared_after_commit = 0U;
  std::size_t panel_gdn_layers_assigned_after_commit = 0U;
  sm87_macrofeed_v4_execution_events_detail::
      Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt enqueue_receipt{};
  bool enqueue_receipt_owner_matched = false;
  bool state_candidate_commit_recorded = false;
  bool physical_device_completion_attested = true;
  bool panel_complete = true;
  bool model_complete = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] bool valid() const noexcept {
    using AuthorityDomain = sm87_macrofeed_v4_execution_events_detail::
        Sm87MacroFeedV4GdnSubmissionAuthorityDomain;
    const std::uint64_t active_bank_offset =
        active_bank_index < 2U
            ? static_cast<std::uint64_t>(active_bank_index) *
                  kSm87MacroFeedV4RecurrentEpochBytes
            : kSm87MacroFeedV4RecurrentStorageBytes;
    const std::uint64_t candidate_bank_offset =
        candidate_bank_index < 2U
            ? static_cast<std::uint64_t>(candidate_bank_index) *
                  kSm87MacroFeedV4RecurrentEpochBytes
            : kSm87MacroFeedV4RecurrentStorageBytes;
    const std::uint64_t conv_slice_offset =
        static_cast<std::uint64_t>(gdn_ordinal) *
        kSm87MacroFeedV4ConvLayerBytes;
    const std::uint64_t gdn_slice_offset =
        kSm87MacroFeedV4ConvEpochBytes +
        static_cast<std::uint64_t>(gdn_ordinal) *
            kSm87MacroFeedV4GdnStateLayerBytes;
    return package_identity != 0U && gdn_catalog_identity != 0U &&
           gdn_binding_identity != 0U && bf16_ab_catalog_identity != 0U &&
           bf16_ab_pair_identity != 0U &&
           layer_norm_catalog_identity != 0U &&
           layer_norm_pair_identity != 0U &&
           input_norm_binding_identity != 0U &&
           post_norm_binding_identity != 0U && mlp_catalog_identity != 0U &&
           mlp_binding_identity != 0U && resource_bundle_identity != 0U &&
           request_epoch != 0U && grant_identity != 0U &&
           recurrent_allocation_identity != 0U &&
           panel < kSm87MacroFeedV4PanelCount &&
           gdn_ordinal < kSm87MacroFeedV4StateLayerCount &&
           model_layer == gdn_ordinal + gdn_ordinal / 3U &&
           active_bank_index < 2U && candidate_bank_index < 2U &&
           active_bank_index != candidate_bank_index &&
           conv_bytes == kernels::kSm87MacroFeedV4GdnConvHistoryBytes &&
           gdn_state_bytes == kernels::kSm87MacroFeedV4GdnStateBytes &&
           active_conv_allocation_offset ==
               active_bank_offset + conv_slice_offset &&
           candidate_conv_allocation_offset ==
               candidate_bank_offset + conv_slice_offset &&
           active_gdn_state_allocation_offset ==
               active_bank_offset + gdn_slice_offset &&
           candidate_gdn_state_allocation_offset ==
               candidate_bank_offset + gdn_slice_offset &&
           next_model_layer_after_commit == model_layer + 1U &&
           panel_conv_layers_prepared_after_commit == gdn_ordinal + 1U &&
           panel_gdn_layers_assigned_after_commit == gdn_ordinal + 1U &&
           enqueue_receipt.valid_shape() &&
           enqueue_receipt.authority_domain() ==
               AuthorityDomain::kNormalSealedCatalog &&
           enqueue_receipt.execution_package_identity() == package_identity &&
           enqueue_receipt.gdn_catalog_identity() == gdn_catalog_identity &&
           enqueue_receipt.gdn_binding_identity() == gdn_binding_identity &&
           enqueue_receipt.bf16_ab_catalog_identity() ==
               bf16_ab_catalog_identity &&
           enqueue_receipt.bf16_ab_pair_identity() ==
               bf16_ab_pair_identity &&
           enqueue_receipt.layer_norm_catalog_identity() ==
               layer_norm_catalog_identity &&
           enqueue_receipt.layer_norm_pair_identity() ==
               layer_norm_pair_identity &&
           enqueue_receipt.input_norm_binding_identity() ==
               input_norm_binding_identity &&
           enqueue_receipt.post_norm_binding_identity() ==
               post_norm_binding_identity &&
           enqueue_receipt.mlp_catalog_identity() == mlp_catalog_identity &&
           enqueue_receipt.mlp_binding_identity() == mlp_binding_identity &&
           enqueue_receipt.resource_bundle_identity() ==
               resource_bundle_identity &&
           enqueue_receipt.synthetic_source_identity() == 0U &&
           enqueue_receipt.request_epoch() == request_epoch &&
           enqueue_receipt.panel() == panel &&
           enqueue_receipt.grant_identity() == grant_identity &&
           enqueue_receipt.grant_state_epoch() == grant_state_epoch &&
           enqueue_receipt.recurrent_allocation_identity() ==
               recurrent_allocation_identity &&
           enqueue_receipt.gdn_ordinal() == gdn_ordinal &&
           enqueue_receipt.model_layer() == model_layer &&
           enqueue_receipt.active_bank_index() == active_bank_index &&
           enqueue_receipt.candidate_bank_index() == candidate_bank_index &&
           enqueue_receipt.active_conv_allocation_offset() ==
               active_conv_allocation_offset &&
           enqueue_receipt.candidate_conv_allocation_offset() ==
               candidate_conv_allocation_offset &&
           enqueue_receipt.conv_bytes() == conv_bytes &&
           enqueue_receipt.active_gdn_state_allocation_offset() ==
               active_gdn_state_allocation_offset &&
           enqueue_receipt.candidate_gdn_state_allocation_offset() ==
               candidate_gdn_state_allocation_offset &&
           enqueue_receipt.gdn_state_bytes() == gdn_state_bytes &&
           enqueue_receipt.input_norm_launches() == 1U &&
           enqueue_receipt.bf16_ab_launches() == 1U &&
           enqueue_receipt.gdn_qkvz_launches() == 1U &&
           enqueue_receipt.gdn_continuation_launches() == 2U &&
           enqueue_receipt.gdn_output_launches() == 1U &&
           enqueue_receipt.residual_post_norm_launches() == 1U &&
           enqueue_receipt.gate_up_launches() == 1U &&
           enqueue_receipt.down_launches() == 1U &&
           enqueue_receipt.bound_kernel_submissions() == 9U &&
           enqueue_receipt.asynchronous_d2d_copies() == 1U &&
           enqueue_receipt.conv_history_copy_bytes() == conv_bytes &&
           enqueue_receipt.norm_ready_waited_by_ab() &&
           enqueue_receipt.ab_ready_waited_by_main() &&
           enqueue_receipt.complete_layer_enqueued() &&
           !enqueue_receipt.physical_device_completion_attested() &&
           !enqueue_receipt.panel_complete() &&
           !enqueue_receipt.production_receipt_eligible() &&
           enqueue_receipt_owner_matched && state_candidate_commit_recorded &&
           !physical_device_completion_attested && !panel_complete &&
           !model_complete && !production_dispatch_eligible;
  }
};

struct Sm87MacroFeedV4GdnLayerCommitResult final {
  Sm87MacroFeedV4P40ExecutionPackageStatus status{};
  Sm87MacroFeedV4GdnLayerCommitReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.valid();
  }
};

// Package-composed evidence for one naturally ordered Full-Attention layer.
// The independently forgeable outer fields make no authority claim; the live
// owner match of the nested opaque Events receipt is the authentication fact.
// Together they record only that the fixed eight-kernel transaction was
// accepted and RequestState consumed the corresponding move-only K/V grant.
// They do not attest device completion, panel publication, model completion,
// or a production route.
struct Sm87MacroFeedV4FullAttentionLayerCommitReceipt final {
  std::uint64_t package_identity = 0U;
  std::uint64_t full_attention_catalog_identity = 0U;
  std::uint64_t full_attention_binding_identity = 0U;
  std::uint64_t mlp_binding_identity = 0U;
  std::uint64_t resource_bundle_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::uint64_t grant_identity = 0U;
  std::uint64_t grant_state_epoch = 0U;
  std::uint64_t kv_allocation_identity = 0U;
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::size_t full_attention_ordinal =
      kSm87MacroFeedV4FullAttentionLayerCount;
  std::size_t model_layer = kSm87MacroFeedV4LayerCount;
  std::size_t previous_valid_end = kSm87MacroFeedV4P40Tokens;
  std::size_t candidate_end = kSm87MacroFeedV4P40Tokens;
  std::size_t next_model_layer_after_commit = 0U;
  std::size_t panel_kv_layers_staged_after_commit = 0U;
  sm87_macrofeed_v4_execution_events_detail::
      Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt
          enqueue_receipt{};
  bool enqueue_receipt_owner_matched = false;
  bool kv_layer_commit_recorded = false;
  bool physical_device_completion_attested = true;
  bool panel_complete = true;
  bool model_complete = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] bool valid() const noexcept {
    return package_identity != 0U &&
           full_attention_catalog_identity != 0U &&
           full_attention_binding_identity != 0U &&
           mlp_binding_identity != 0U && resource_bundle_identity != 0U &&
           request_epoch != 0U && grant_identity != 0U &&
           kv_allocation_identity != 0U &&
           panel < kSm87MacroFeedV4PanelCount &&
           full_attention_ordinal <
               kSm87MacroFeedV4FullAttentionLayerCount &&
           model_layer == 4U * full_attention_ordinal + 3U &&
           previous_valid_end ==
               panel * kSm87MacroFeedV4PanelTokens &&
           candidate_end ==
               previous_valid_end + kSm87MacroFeedV4PanelTokens &&
           next_model_layer_after_commit == model_layer + 1U &&
           panel_kv_layers_staged_after_commit ==
               full_attention_ordinal + 1U &&
           enqueue_receipt.valid_shape() &&
           enqueue_receipt.request_epoch() == request_epoch &&
           enqueue_receipt.panel() == panel &&
           enqueue_receipt.grant_identity() == grant_identity &&
           enqueue_receipt.kv_allocation_identity() ==
               kv_allocation_identity &&
           enqueue_receipt.full_attention_ordinal() ==
               full_attention_ordinal &&
           enqueue_receipt.model_layer() == model_layer &&
           enqueue_receipt.full_attention_catalog_identity() ==
               full_attention_catalog_identity &&
           enqueue_receipt.resource_bundle_identity() ==
               resource_bundle_identity &&
           enqueue_receipt.bound_kernel_submissions() == 8U &&
           enqueue_receipt.asynchronous_d2d_copies() == 0U &&
           enqueue_receipt.asynchronous_d2d_copy_bytes() == 0U &&
           enqueue_receipt.complete_layer_enqueued() &&
           !enqueue_receipt.physical_device_completion_attested() &&
           !enqueue_receipt.panel_complete() &&
           !enqueue_receipt.production_receipt_eligible() &&
           enqueue_receipt_owner_matched && kv_layer_commit_recorded &&
           !physical_device_completion_attested && !panel_complete &&
           !model_complete && !production_dispatch_eligible;
  }
};

struct Sm87MacroFeedV4FullAttentionLayerCommitResult final {
  Sm87MacroFeedV4P40ExecutionPackageStatus status{};
  Sm87MacroFeedV4FullAttentionLayerCommitReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.valid();
  }
};

class Sm87MacroFeedV4P40ExecutionPackage;
class Sm87MacroFeedV4P40ExecutionCompositionRoot;

struct Sm87MacroFeedV4P40ExecutionPackageCreateResult final {
  std::unique_ptr<Sm87MacroFeedV4P40ExecutionPackage> package;
  Sm87MacroFeedV4P40ExecutionPackageStatus status{};
  Sm87MacroFeedV4P40ExecutionPackageAudit audit{};

  [[nodiscard]] explicit operator bool() const noexcept;
};

class Sm87MacroFeedV4P40ExecutionPackage final {
 public:
  using StartupPackage = sm87_macrofeed_v4_p40_startup_package_detail::
      Sm87MacroFeedV4P40StartupPackage;

  Sm87MacroFeedV4P40ExecutionPackage() = delete;
  Sm87MacroFeedV4P40ExecutionPackage(
      const Sm87MacroFeedV4P40ExecutionPackage&) = delete;
  Sm87MacroFeedV4P40ExecutionPackage& operator=(
      const Sm87MacroFeedV4P40ExecutionPackage&) = delete;
  Sm87MacroFeedV4P40ExecutionPackage(
      Sm87MacroFeedV4P40ExecutionPackage&&) = delete;
  Sm87MacroFeedV4P40ExecutionPackage& operator=(
      Sm87MacroFeedV4P40ExecutionPackage&&) = delete;
  ~Sm87MacroFeedV4P40ExecutionPackage() noexcept;

  // Construction/lifetime diagnostic only.  It performs the complete sealed
  // catalog postcondition scan and must never be called from request execution.
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const Sm87MacroFeedV4P40ExecutionPackageAudit& audit()
      const noexcept {
    return audit_;
  }

 private:
  // Only the Engine composition root and the named CUDA fixture may mint an
  // execution package.  Keeping this normal factory private prevents a
  // caller from detaching stream/transient ownership from the complete
  // ModelWeights -> StartupPackage -> ExecutionPackage lifetime root.
  [[nodiscard]] static Sm87MacroFeedV4P40ExecutionPackageCreateResult create(
      const StartupPackage& startup_package,
      const Sm87MacroFeedV4P40EngineRopeBinding& engine_rope,
      std::uint64_t minimum_free_bytes_after_create) noexcept;

  using ProjectionBinding = sm87_macrofeed_v4_p40_startup_package_detail::
      Sm87MacroFeedV4ProjectionStartupBinding;
  using ProjectionCatalog =
      std::array<std::optional<ProjectionBinding>,
                 sm87_macrofeed_v4_p40_startup_package_detail::
                     kSm87MacroFeedV4P40StartupPackageArtifacts>;
  using Bf16AbCatalog = StartupPackage::Bf16AbExecutionBindingCatalog;
  using LayerNormCatalog = StartupPackage::LayerNormExecutionBindingCatalog;
  using GdnQkvZCatalog = StartupPackage::GdnQkvZExecutionBindingCatalog;
  using MlpPairCatalog = StartupPackage::MlpPairExecutionBindingCatalog;
  using FullAttentionCatalog =
      StartupPackage::FullAttentionLayerExecutionBindingCatalog;
  using EventsOwner = sm87_macrofeed_v4_execution_events_detail::
      Sm87MacroFeedV4ExecutionEventsOwner;
  using EventsDriver = sm87_macrofeed_v4_execution_events_detail::
      Sm87MacroFeedV4ExecutionEventsDriver;
  using PanelAccess = sm87_macrofeed_v4_execution_events_detail::
      Sm87MacroFeedV4ExecutionPanelAccess;

  struct GdnLayer0ExecutionSource final {
    kernels::Sm87TargetAotFp8CudaAssetView asset{};
    kernels::Sm87MacroFeedV4Fp8CudaResources resources{};
    std::uint64_t identity = 0U;
    bool synthetic_t1 = false;
  };

  struct CompleteGdnLayer0ExecutionSource final {
    StartupPackage::GdnLayerExecutionBinding gdn_layer{};
    StartupPackage::MlpPairExecutionBinding mlp_pair{};
    kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt gate_up_receipt{};
    kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt down_receipt{};
    std::uint64_t identity = 0U;
    bool synthetic_t1 = false;
  };

  // BUILD_TESTING friend-fixture input only.  It can substitute one honest
  // live CUDA source for the fake complete-catalog fixture, but it never
  // becomes a selector, public launcher, or production execution capability.
  struct SyntheticCompleteGdnLayer0Source final {
    kernels::Sm87TargetAotFp8CudaAssetView gdn_qkvz_asset{};
    kernels::Sm87TargetAotFp8CudaAssetView gdn_output_asset{};
    kernels::Sm87TargetAotNvFp4CudaAssetView gate_up_asset{};
    kernels::Sm87TargetAotNvFp4CudaAssetView down_asset{};
    const std::uint16_t* conv_weight = nullptr;
    const std::uint16_t* a_log = nullptr;
    const std::uint16_t* dt_bias = nullptr;
    const std::uint16_t* norm_weight = nullptr;
    kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt gate_up_receipt{};
    kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt down_receipt{};
  };

  Sm87MacroFeedV4P40ExecutionPackage(
      ProjectionCatalog projection_catalog, Bf16AbCatalog bf16_ab_catalog,
      LayerNormCatalog layer_norm_catalog, GdnQkvZCatalog gdn_qkvz_catalog,
      MlpPairCatalog mlp_pair_catalog,
      FullAttentionCatalog full_attention_catalog,
      GdnLayer0ExecutionSource gdn_layer0_source,
      std::optional<CompleteGdnLayer0ExecutionSource>
          complete_gdn_layer0_source,
      kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot
          norm_resources,
      kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot
          bf16_ab_resources,
      kernels::Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot
          gdn_resources,
      kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources gate_up_resources,
      kernels::Sm87MacroFeedV4NvFp4DownCudaResources down_resources,
      Sm87MacroFeedV4P40EngineRopeBinding engine_rope,
      void* kv_allocation, void* transient_allocation,
      void* recurrent_allocation,
      std::unique_ptr<Sm87MacroFeedV4RequestState> request_state,
      std::shared_ptr<EventsOwner> events_owner,
      std::unique_ptr<EventsDriver> events_driver,
      Sm87MacroFeedV4P40ExecutionPackageAudit audit) noexcept;

  [[nodiscard]] static Sm87MacroFeedV4P40ExecutionPackageCreateResult
  create_impl(
      const StartupPackage& startup_package,
      const Sm87MacroFeedV4P40EngineRopeBinding* engine_rope,
      std::uint64_t minimum_free_bytes_after_create,
      const kernels::Sm87TargetAotFp8CudaAssetView*
          synthetic_t1_gdn_layer0_asset,
      const SyntheticCompleteGdnLayer0Source*
          synthetic_complete_gdn_layer0_source) noexcept;

  // This is deliberately a one-shot admission slice, not a model executor.
  // Only the Engine composition root and the CUDA fixture may invoke it;
  // a package cannot independently launch against expired weight ownership.
  [[nodiscard]] Sm87MacroFeedV4GdnLayer0FrontHalfResult
  execute_gdn_layer0_front_half_once() noexcept;
  [[nodiscard]] Sm87MacroFeedV4GdnLayer0CompleteResult
  execute_gdn_layer0_complete_once() noexcept;
  // Reusable normal-domain GDN composition seam for a future natural-layer
  // cohort.  The caller supplies only already-live owner capabilities and the
  // fixed GDN ordinal; every pointer, identity, resource and layer is derived
  // in O(1) from the construction-sealed package.
  [[nodiscard]] Sm87MacroFeedV4GdnLayerCommitResult
  submit_complete_gdn_layer_c8000(
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      const PanelAccess& panel_access, std::size_t gdn_ordinal) noexcept;
  // First reusable Full-layer composition slice.  A future natural-layer
  // cohort is the only intended caller: it supplies the already active
  // package-owned request/panel capabilities and a fixed Full ordinal.  No
  // pointer, tactic, stream, role, layer kind, or K/V offset crosses this
  // boundary as a caller choice.
  [[nodiscard]] Sm87MacroFeedV4FullAttentionLayerCommitResult
  submit_complete_full_attention_layer_c8000(
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      const PanelAccess& panel_access,
      std::size_t full_attention_ordinal) noexcept;
  [[nodiscard]] bool front_half_bindings_valid() const noexcept;
  [[nodiscard]] bool complete_layer_bindings_valid() const noexcept;
  [[nodiscard]] bool gdn_composer_authority_sealed() const noexcept;
  [[nodiscard]] bool compose_complete_gdn_submission(
      const Sm87MacroFeedV4GdnLayerStateGrant& grant,
      sm87_macrofeed_v4_execution_events_detail::
          Sm87MacroFeedV4CompleteGdnLayerC8000Submission* submission)
      const noexcept;
  [[nodiscard]] bool full_attention_composer_authority_sealed()
      const noexcept;
  [[nodiscard]] bool compose_complete_full_attention_submission(
      const Sm87MacroFeedV4FullAttentionKvGrant& grant,
      sm87_macrofeed_v4_execution_events_detail::
          Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission*
              submission) const noexcept;
  [[nodiscard]] static std::uint64_t compute_gdn_layer_catalog_fold_identity(
      const GdnQkvZCatalog& catalog) noexcept;
  [[nodiscard]] static std::uint64_t compute_mlp_pair_catalog_fold_identity(
      const MlpPairCatalog& catalog) noexcept;
  [[nodiscard]] static std::uint64_t
  compute_full_attention_catalog_fold_identity(
      const FullAttentionCatalog& catalog) noexcept;
  [[nodiscard]] static std::uint64_t compute_complete_layer0_source_identity(
      const CompleteGdnLayer0ExecutionSource& source) noexcept;
  [[nodiscard]] Sm87MacroFeedV4P40ExecutionPackageStatus
  drain_and_discard_active_panel(
      const PanelAccess& panel_access,
      std::uint64_t* owner_drain_receipt_identity = nullptr,
      const Sm87MacroFeedV4RequestStateSealedAccess*
          request_state_access = nullptr) noexcept;
  [[nodiscard]] Sm87MacroFeedV4P40ExecutionPackageStatus
  terminalize_event_failure(
      const char* context,
      const sm87_macrofeed_v4_execution_events_detail::
          Sm87MacroFeedV4ExecutionStatus& event_status,
      const Sm87MacroFeedV4RequestStateSealedAccess*
          request_state_access = nullptr) noexcept;
  [[nodiscard]] Sm87MacroFeedV4P40ExecutionPackageStatus
  abort_request_state() noexcept;
  void release() noexcept;

  ProjectionCatalog projection_catalog_{};
  Bf16AbCatalog bf16_ab_catalog_{};
  LayerNormCatalog layer_norm_catalog_{};
  GdnQkvZCatalog gdn_qkvz_catalog_{};
  MlpPairCatalog mlp_pair_catalog_{};
  FullAttentionCatalog full_attention_catalog_{};
  GdnLayer0ExecutionSource gdn_layer0_source_{};
  std::optional<CompleteGdnLayer0ExecutionSource>
      complete_gdn_layer0_source_{};
  kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot
      norm_resources_{};
  kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot
      bf16_ab_resources_{};
  kernels::Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot
      gdn_resources_{};
  kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources gate_up_resources_{};
  kernels::Sm87MacroFeedV4NvFp4DownCudaResources down_resources_{};
  Sm87MacroFeedV4P40EngineRopeBinding engine_rope_{};
  void* kv_allocation_ = nullptr;
  void* transient_allocation_ = nullptr;
  void* recurrent_allocation_ = nullptr;
  std::uint16_t* ping_ = nullptr;
  std::uint16_t* pong_ = nullptr;
  std::uint16_t* scratch_ = nullptr;
  std::unique_ptr<Sm87MacroFeedV4RequestState> request_state_;
  std::shared_ptr<EventsOwner> events_owner_;
  std::unique_ptr<EventsDriver> events_driver_;
  Sm87MacroFeedV4P40ExecutionPackageAudit audit_{};
  bool construction_postconditions_sealed_ = false;
  bool execution_attempted_ = false;

  friend struct Sm87MacroFeedV4P40ExecutionPackageCreateResult;
  friend class Sm87MacroFeedV4P40ExecutionCompositionRoot;
  friend class Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture;
};

}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail
