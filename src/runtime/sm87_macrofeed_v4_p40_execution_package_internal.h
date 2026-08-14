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

static_assert(kSm87MacroFeedV4P40ExecutionTransientBytes == 442'368'000U);
static_assert(kSm87MacroFeedV4P40ExecutionScratchOffset == 163'840'000U);

enum class Sm87MacroFeedV4P40ExecutionPackageError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kStartupPackage,
  kProjectionCatalog,
  kBf16AbCatalog,
  kLayerNormCatalog,
  kGdnQkvZCatalog,
  kMlpPairCatalog,
  kNormResources,
  kGdnResources,
  kGateUpResources,
  kDownResources,
  kExecutionEvents,
  kTransientAllocation,
  kRecurrentAllocation,
  kColdRecurrentInitialization,
  kRequestState,
  kPackageAllocation,
  kAlreadyExecuted,
  kFrontHalfBinding,
  kCompleteLayerBinding,
  kGdnLayerStateGrant,
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
  std::uint64_t retained_gdn_layer_catalog_fold_identity = 0U;
  std::uint64_t retained_mlp_pair_catalog_fold_identity = 0U;
  std::uint64_t gdn_layer0_source_identity = 0U;
  std::uint64_t transient_allocation_identity = 0U;
  std::uint64_t recurrent_allocation_identity = 0U;
  std::uint64_t execution_events_owner_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::size_t projection_bindings = 0U;
  std::size_t bf16_ab_pairs = 0U;
  std::size_t layer_norm_pairs = 0U;
  std::size_t gdn_qkvz_bindings = 0U;
  std::size_t mlp_pair_bindings = 0U;
  std::uint64_t transient_bytes = 0U;
  std::uint64_t recurrent_bytes = 0U;
  std::uint64_t cold_recurrent_zero_bytes = 0U;
  std::size_t cold_recurrent_initializations = 0U;
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
      const StartupPackage& startup_package) noexcept;

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
      void* transient_allocation, void* recurrent_allocation,
      std::unique_ptr<Sm87MacroFeedV4RequestState> request_state,
      std::shared_ptr<EventsOwner> events_owner,
      std::unique_ptr<EventsDriver> events_driver,
      Sm87MacroFeedV4P40ExecutionPackageAudit audit) noexcept;

  [[nodiscard]] static Sm87MacroFeedV4P40ExecutionPackageCreateResult
  create_impl(
      const StartupPackage& startup_package,
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
  [[nodiscard]] bool front_half_bindings_valid() const noexcept;
  [[nodiscard]] bool complete_layer_bindings_valid() const noexcept;
  [[nodiscard]] static std::uint64_t compute_gdn_layer_catalog_fold_identity(
      const GdnQkvZCatalog& catalog) noexcept;
  [[nodiscard]] static std::uint64_t compute_mlp_pair_catalog_fold_identity(
      const MlpPairCatalog& catalog) noexcept;
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
