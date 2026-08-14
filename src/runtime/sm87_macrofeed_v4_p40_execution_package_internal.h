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
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)
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
  kNormResources,
  kExecutionEvents,
  kTransientAllocation,
  kRecurrentAllocation,
  kRequestState,
  kPackageAllocation,
  kAlreadyExecuted,
  kFrontHalfBinding,
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
  std::uint64_t gdn_layer0_source_identity = 0U;
  std::uint64_t transient_allocation_identity = 0U;
  std::uint64_t recurrent_allocation_identity = 0U;
  std::uint64_t execution_events_owner_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::size_t projection_bindings = 0U;
  std::size_t bf16_ab_pairs = 0U;
  std::size_t layer_norm_pairs = 0U;
  std::size_t gdn_qkvz_bindings = 0U;
  std::uint64_t transient_bytes = 0U;
  std::uint64_t recurrent_bytes = 0U;
  bool fixed_gdn_layer0_front_half_bound = false;
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
    const bool real_gdn_catalog =
        !synthetic_t1_gdn_layer0_source &&
        gdn_qkvz_catalog_identity != 0U &&
        gdn_qkvz_bindings == kSm87MacroFeedV4StateLayerCount;
    const bool synthetic_t1_source =
        synthetic_t1_gdn_layer0_source &&
        gdn_qkvz_catalog_identity == 0U && gdn_qkvz_bindings == 1U;
    return package_identity != 0U && startup_package_identity != 0U &&
           projection_catalog_identity != 0U &&
           bf16_ab_catalog_identity != 0U &&
           layer_norm_catalog_identity != 0U &&
           gdn_layer0_source_identity != 0U &&
           (real_gdn_catalog || synthetic_t1_source) &&
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
           fixed_gdn_layer0_front_half_bound &&
           qkvz_ab_ready_transaction_bound &&
           !whole_layer_executor_bound && !whole_model_executor_bound &&
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

  [[nodiscard]] static Sm87MacroFeedV4P40ExecutionPackageCreateResult create(
      const StartupPackage& startup_package) noexcept;

  // Construction/lifetime diagnostic only.  It performs the complete sealed
  // catalog postcondition scan and must never be called from request execution.
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const Sm87MacroFeedV4P40ExecutionPackageAudit& audit()
      const noexcept {
    return audit_;
  }

 private:
  using ProjectionBinding = sm87_macrofeed_v4_p40_startup_package_detail::
      Sm87MacroFeedV4ProjectionStartupBinding;
  using ProjectionCatalog =
      std::array<std::optional<ProjectionBinding>,
                 sm87_macrofeed_v4_p40_startup_package_detail::
                     kSm87MacroFeedV4P40StartupPackageArtifacts>;
  using Bf16AbCatalog = StartupPackage::Bf16AbExecutionBindingCatalog;
  using LayerNormCatalog = StartupPackage::LayerNormExecutionBindingCatalog;
  using GdnQkvZCatalog = StartupPackage::GdnQkvZExecutionBindingCatalog;
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

  Sm87MacroFeedV4P40ExecutionPackage(
      ProjectionCatalog projection_catalog, Bf16AbCatalog bf16_ab_catalog,
      LayerNormCatalog layer_norm_catalog, GdnQkvZCatalog gdn_qkvz_catalog,
      GdnLayer0ExecutionSource gdn_layer0_source,
      kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot
          norm_resources,
      kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot
          bf16_ab_resources,
      void* transient_allocation, void* recurrent_allocation,
      std::unique_ptr<Sm87MacroFeedV4RequestState> request_state,
      std::shared_ptr<EventsOwner> events_owner,
      std::unique_ptr<EventsDriver> events_driver,
      Sm87MacroFeedV4P40ExecutionPackageAudit audit) noexcept;

  [[nodiscard]] static Sm87MacroFeedV4P40ExecutionPackageCreateResult
  create_impl(
      const StartupPackage& startup_package,
      const kernels::Sm87TargetAotFp8CudaAssetView*
          synthetic_t1_gdn_layer0_asset) noexcept;

  // CUDA-fixture-only composition seam.  It cannot grant authority to the
  // fake complete-catalog fixture: the supplied typed asset must own one
  // honest live CUDA allocation, the resulting package is explicitly marked
  // synthetic/non-production, and the normal create() path still requires
  // all 48 construction-sealed real-model bindings.
  [[nodiscard]] static Sm87MacroFeedV4P40ExecutionPackageCreateResult
  create_with_synthetic_t1_gdn_layer0_for_cuda_test(
      const StartupPackage& startup_package,
      const kernels::Sm87TargetAotFp8CudaAssetView& asset) noexcept;

  // This is deliberately a one-shot admission slice, not a model executor.
  // Only the future Engine composition root and the CUDA fixture may invoke it;
  // a package cannot independently launch against expired weight ownership.
  [[nodiscard]] Sm87MacroFeedV4GdnLayer0FrontHalfResult
  execute_gdn_layer0_front_half_once() noexcept;
  [[nodiscard]] bool front_half_bindings_valid() const noexcept;
  [[nodiscard]] Sm87MacroFeedV4P40ExecutionPackageStatus
  drain_and_discard_active_panel(const PanelAccess& panel_access) noexcept;
  [[nodiscard]] Sm87MacroFeedV4P40ExecutionPackageStatus
  terminalize_event_failure(
      const char* context,
      const sm87_macrofeed_v4_execution_events_detail::
          Sm87MacroFeedV4ExecutionStatus& event_status) noexcept;
  [[nodiscard]] Sm87MacroFeedV4P40ExecutionPackageStatus
  abort_request_state() noexcept;
  void release() noexcept;

  ProjectionCatalog projection_catalog_{};
  Bf16AbCatalog bf16_ab_catalog_{};
  LayerNormCatalog layer_norm_catalog_{};
  GdnQkvZCatalog gdn_qkvz_catalog_{};
  GdnLayer0ExecutionSource gdn_layer0_source_{};
  kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot
      norm_resources_{};
  kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot
      bf16_ab_resources_{};
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
