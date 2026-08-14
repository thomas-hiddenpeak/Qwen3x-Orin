#pragma once

#include "sm87_macrofeed_v4_request_state_internal.h"
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
#include "../kernels/sm87/sm87_macrofeed_v4_bound_launch_internal.h"
#endif

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

namespace q3x::runtime::sm87_macrofeed_v4_execution_events_detail {

inline constexpr std::size_t kSm87MacroFeedV4ExecutionStreamCount = 3U;
inline constexpr std::size_t kSm87MacroFeedV4ExecutionEventCount = 9U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbCyclesPerPanel =
    kSm87MacroFeedV4StateLayerCount;

enum class Sm87MacroFeedV4ExecutionStream : std::uint8_t {
  kMain = 0U,
  kAbAux,
  kControl,
  kCount,
};

enum class Sm87MacroFeedV4ExecutionEvent : std::uint8_t {
  kNormReady = 0U,
  kAbReady,
  kPanelDone,
  kMainTail,
  kAbTail,
  kOwnerDrained,
  kFinalRepresentationReady,
  kCanonicalCopyDone,
  kFinalPublish,
  kCount,
};

enum class Sm87MacroFeedV4ExecutionOwnerState : std::uint8_t {
  kEmpty = 0U,
  kReady,
  kRequestActive,
  kRequestCompleted,
  kRequestDiscarded,
  kPoisoned,
  kDestroyed,
};

enum class Sm87MacroFeedV4ExecutionError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kDeviceQuery,
  kWrongDevice,
  kStreamCreate,
  kStreamValidation,
  kEventCreate,
  kForeignOwnerAccess,
  kForeignRequestAccess,
  kInvalidRequestEpoch,
  kInvalidOwnerState,
  kInvalidPanel,
  kStalePanelGeneration,
  kWrongStream,
  kInvalidEventOrder,
  kEventNotRecorded,
  kEventGenerationUnobserved,
  kEventNotComplete,
  kEventAlreadyObserved,
  kPhysicalObservationForbidden,
  kCudaSubmission,
  kCudaObservation,
  kKernelSubmitContract,
  kReceiptInvalid,
  kPanelIncomplete,
  kDrainIncomplete,
  kFinalPublicationIncomplete,
};

struct Sm87MacroFeedV4ExecutionStatus final {
  Sm87MacroFeedV4ExecutionError error =
      Sm87MacroFeedV4ExecutionError::kNone;
  const char* context = "none";
  int cuda_error = 0;
  Sm87MacroFeedV4ExecutionStream stream =
      Sm87MacroFeedV4ExecutionStream::kCount;
  Sm87MacroFeedV4ExecutionEvent event =
      Sm87MacroFeedV4ExecutionEvent::kCount;
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation = 0U;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return error == Sm87MacroFeedV4ExecutionError::kNone;
  }
};

class Sm87MacroFeedV4ExecutionEventsOwner;
class Sm87MacroFeedV4ExecutionEventsDriver;
class Sm87MacroFeedV4ExecutionEventsCudaTestFixture;

// The only caller-visible owner capability.  It exposes identities but no raw
// CUDA stream or event handle.  Object identity and a private seal are checked
// on every operation; caller-filled evidence cannot construct a substitute.
class Sm87MacroFeedV4ExecutionEventsAccess final {
 public:
  Sm87MacroFeedV4ExecutionEventsAccess() = delete;
  Sm87MacroFeedV4ExecutionEventsAccess(
      const Sm87MacroFeedV4ExecutionEventsAccess&) = delete;
  Sm87MacroFeedV4ExecutionEventsAccess& operator=(
      const Sm87MacroFeedV4ExecutionEventsAccess&) = delete;
  Sm87MacroFeedV4ExecutionEventsAccess(
      Sm87MacroFeedV4ExecutionEventsAccess&&) = delete;
  Sm87MacroFeedV4ExecutionEventsAccess& operator=(
      Sm87MacroFeedV4ExecutionEventsAccess&&) = delete;

  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }

 private:
  Sm87MacroFeedV4ExecutionEventsAccess(
      const Sm87MacroFeedV4ExecutionEventsOwner* owner,
      std::uint64_t owner_identity, std::uint64_t seal_nonce,
      std::int32_t device_ordinal) noexcept;

  const Sm87MacroFeedV4ExecutionEventsOwner* owner_ = nullptr;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t seal_nonce_ = 0U;
  std::int32_t device_ordinal_ = -1;

  friend class Sm87MacroFeedV4ExecutionEventsOwner;
};

// One owner-minted panel capability.  A capability becomes stale as soon as
// its panel completes, is discarded, or another request begins.
class Sm87MacroFeedV4ExecutionPanelAccess final {
 public:
  Sm87MacroFeedV4ExecutionPanelAccess() = delete;
  Sm87MacroFeedV4ExecutionPanelAccess(
      const Sm87MacroFeedV4ExecutionPanelAccess&) = delete;
  Sm87MacroFeedV4ExecutionPanelAccess& operator=(
      const Sm87MacroFeedV4ExecutionPanelAccess&) = delete;
  Sm87MacroFeedV4ExecutionPanelAccess(
      Sm87MacroFeedV4ExecutionPanelAccess&&) = delete;
  Sm87MacroFeedV4ExecutionPanelAccess& operator=(
      Sm87MacroFeedV4ExecutionPanelAccess&&) = delete;

  [[nodiscard]] std::uint64_t request_epoch() const noexcept {
    return request_epoch_;
  }
  [[nodiscard]] std::size_t panel() const noexcept { return panel_; }
  [[nodiscard]] std::uint64_t panel_generation() const noexcept {
    return panel_generation_;
  }

 private:
  Sm87MacroFeedV4ExecutionPanelAccess(
      const Sm87MacroFeedV4ExecutionEventsOwner* owner,
      std::uint64_t owner_identity, std::uint64_t seal_nonce,
      std::uint64_t request_epoch, std::size_t panel,
      std::uint64_t panel_generation) noexcept;

  const Sm87MacroFeedV4ExecutionEventsOwner* owner_ = nullptr;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t seal_nonce_ = 0U;
  std::uint64_t request_epoch_ = 0U;
  std::size_t panel_ = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation_ = 0U;

  friend class Sm87MacroFeedV4ExecutionEventsOwner;
};

enum class Sm87MacroFeedV4EnqueueOperation : std::uint8_t {
  kInvalid = 0U,
  kRecord,
  kWait,
};

// Enqueue evidence is intentionally a different type from physical completion
// evidence.  Its authority bits are permanently false.
struct Sm87MacroFeedV4EventEnqueueReceipt final {
  std::uint64_t enqueue_identity = 0U;
  std::uint64_t owner_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation = 0U;
  Sm87MacroFeedV4ExecutionStream stream =
      Sm87MacroFeedV4ExecutionStream::kCount;
  Sm87MacroFeedV4ExecutionEvent event =
      Sm87MacroFeedV4ExecutionEvent::kCount;
  std::uint64_t event_generation = 0U;
  Sm87MacroFeedV4EnqueueOperation operation =
      Sm87MacroFeedV4EnqueueOperation::kInvalid;
  bool cuda_enqueue_accepted = false;
  bool physical_device_completion_attested = false;
  bool production_receipt_eligible = false;
};

struct Sm87MacroFeedV4EventEnqueueResult final {
  Sm87MacroFeedV4ExecutionStatus status{};
  Sm87MacroFeedV4EventEnqueueReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.enqueue_identity != 0U &&
           receipt.cuda_enqueue_accepted &&
           !receipt.physical_device_completion_attested &&
           !receipt.production_receipt_eligible;
  }
};

// Copyable opaque completion evidence.  The default object is deliberately an
// invalid forged receipt.  Valid objects can only be minted after a successful
// cudaEventQuery or cudaEventSynchronize and carry a private owner seal.
class Sm87MacroFeedV4PhysicalCompletionReceipt final {
 public:
  Sm87MacroFeedV4PhysicalCompletionReceipt() = default;
  Sm87MacroFeedV4PhysicalCompletionReceipt(
      const Sm87MacroFeedV4PhysicalCompletionReceipt&) = default;
  Sm87MacroFeedV4PhysicalCompletionReceipt& operator=(
      const Sm87MacroFeedV4PhysicalCompletionReceipt&) = default;

  [[nodiscard]] std::uint64_t receipt_identity() const noexcept {
    return receipt_identity_;
  }
  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::uint64_t request_epoch() const noexcept {
    return request_epoch_;
  }
  [[nodiscard]] std::size_t panel() const noexcept { return panel_; }
  [[nodiscard]] std::uint64_t panel_generation() const noexcept {
    return panel_generation_;
  }
  [[nodiscard]] Sm87MacroFeedV4ExecutionStream producer() const noexcept {
    return producer_;
  }
  [[nodiscard]] Sm87MacroFeedV4ExecutionEvent event() const noexcept {
    return event_;
  }
  [[nodiscard]] std::uint64_t event_generation() const noexcept {
    return event_generation_;
  }
  [[nodiscard]] std::uint64_t main_tail_generation() const noexcept {
    return main_tail_generation_;
  }
  [[nodiscard]] std::uint64_t ab_tail_generation() const noexcept {
    return ab_tail_generation_;
  }
  [[nodiscard]] bool observed_by_query() const noexcept {
    return observed_by_query_;
  }
  [[nodiscard]] bool observed_by_synchronize() const noexcept {
    return observed_by_synchronize_;
  }
  [[nodiscard]] bool physical_device_completion_attested() const noexcept {
    return physical_device_completion_attested_;
  }
  [[nodiscard]] bool production_receipt_eligible() const noexcept {
    return production_receipt_eligible_;
  }

 private:
  std::uint64_t receipt_identity_ = 0U;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t request_epoch_ = 0U;
  std::size_t panel_ = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation_ = 0U;
  Sm87MacroFeedV4ExecutionStream producer_ =
      Sm87MacroFeedV4ExecutionStream::kCount;
  Sm87MacroFeedV4ExecutionEvent event_ =
      Sm87MacroFeedV4ExecutionEvent::kCount;
  std::uint64_t event_generation_ = 0U;
  std::uint64_t main_tail_generation_ = 0U;
  std::uint64_t ab_tail_generation_ = 0U;
  std::uint64_t authenticator_ = 0U;
  bool observed_by_query_ = false;
  bool observed_by_synchronize_ = false;
  bool physical_device_completion_attested_ = false;
  bool production_receipt_eligible_ = false;

  friend class Sm87MacroFeedV4ExecutionEventsOwner;
};

struct Sm87MacroFeedV4PhysicalObservationResult final {
  Sm87MacroFeedV4ExecutionStatus status{};
  Sm87MacroFeedV4PhysicalCompletionReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.receipt_identity() != 0U &&
           receipt.physical_device_completion_attested() &&
           !receipt.production_receipt_eligible();
  }
};

struct Sm87MacroFeedV4PoisonDrainResult final {
  Sm87MacroFeedV4ExecutionStatus drain_status{};
  Sm87MacroFeedV4ExecutionStatus poison_cause{};
  std::array<int, kSm87MacroFeedV4ExecutionStreamCount>
      stream_cuda_status{};
  bool all_stream_synchronizations_attempted = false;
  bool physical_quiescence_attested = false;
  bool discard_required = false;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(drain_status) &&
           poison_cause.error != Sm87MacroFeedV4ExecutionError::kNone &&
           physical_quiescence_attested;
  }
};

struct Sm87MacroFeedV4PanelBeginResult final {
  std::unique_ptr<Sm87MacroFeedV4ExecutionPanelAccess> panel_access;
  Sm87MacroFeedV4ExecutionStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return panel_access != nullptr && static_cast<bool>(status);
  }
};

struct Sm87MacroFeedV4ExecutionEventsSnapshot final {
  Sm87MacroFeedV4ExecutionOwnerState state =
      Sm87MacroFeedV4ExecutionOwnerState::kEmpty;
  std::uint64_t owner_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::uint64_t request_owner_identity = 0U;
  std::uint64_t request_allocation_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::size_t completed_panels = 0U;
  std::size_t active_panel = kSm87MacroFeedV4PanelCount;
  std::uint64_t active_panel_generation = 0U;
  std::size_t bf16_ab_cycles_completed = 0U;
  std::array<std::uint64_t, kSm87MacroFeedV4ExecutionEventCount>
      event_generations{};
  std::size_t enqueue_receipts_issued = 0U;
  std::size_t physical_completion_receipts_issued = 0U;
  std::size_t bound_kernel_submissions = 0U;
  std::size_t input_norm_submissions = 0U;
  std::size_t bf16_ab_submissions = 0U;
  bool streams_nonblocking = false;
  bool panel_done_recorded = false;
  bool main_tail_recorded = false;
  bool ab_tail_recorded = false;
  bool main_tail_joined = false;
  bool ab_tail_joined = false;
  bool owner_drained_recorded = false;
  bool final_representation_ready_recorded = false;
  bool final_representation_joined = false;
  bool canonical_copy_done_recorded = false;
  bool canonical_copy_joined = false;
  bool final_publish_recorded = false;
  Sm87MacroFeedV4ExecutionStatus poison_cause{};
  std::array<int, kSm87MacroFeedV4ExecutionStreamCount>
      poison_drain_stream_cuda_status{};
  bool poison_drain_all_stream_synchronizations_attempted = false;
  bool poisoned_terminal_quiescence_attested = false;
  bool default_off = true;
  bool selector_bound = false;
  bool launcher_bound = false;
  bool jit_present = false;
  bool request_time_repack_present = false;
  bool request_time_autotune_present = false;
  bool cublaslt_present = false;
  bool mtp_present = false;
  bool production_dispatch_eligible = false;
};

struct Sm87MacroFeedV4ExecutionEventsCreateResult final {
  std::unique_ptr<Sm87MacroFeedV4ExecutionEventsOwner> owner;
  Sm87MacroFeedV4ExecutionStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept;
};

class Sm87MacroFeedV4ExecutionEventsOwner final {
 public:
  Sm87MacroFeedV4ExecutionEventsOwner(
      const Sm87MacroFeedV4ExecutionEventsOwner&) = delete;
  Sm87MacroFeedV4ExecutionEventsOwner& operator=(
      const Sm87MacroFeedV4ExecutionEventsOwner&) = delete;
  Sm87MacroFeedV4ExecutionEventsOwner(
      Sm87MacroFeedV4ExecutionEventsOwner&&) = delete;
  Sm87MacroFeedV4ExecutionEventsOwner& operator=(
      Sm87MacroFeedV4ExecutionEventsOwner&&) = delete;
  ~Sm87MacroFeedV4ExecutionEventsOwner();

  [[nodiscard]] Sm87MacroFeedV4ExecutionEventsSnapshot snapshot()
      const noexcept;

 private:
  [[nodiscard]] const Sm87MacroFeedV4ExecutionEventsAccess* access()
      const noexcept {
    return access_.get();
  }

  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus begin_request(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PanelBeginResult begin_panel(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      std::size_t panel) noexcept;

  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult record_event(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream producer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult wait_event(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream consumer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  // The exact body enqueue and its ready-event record are one owner-locked
  // transaction.  The package never receives or stores a raw CUDA stream.
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_input_norm_and_record_ready(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::Sm87MacroFeedV4InputNormArguments& arguments,
      const kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&
          resources) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_bf16_ab_and_record_ready(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::Sm87MacroFeedV4Bf16AbArguments& arguments,
      const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
          resources) noexcept;
#endif

  [[nodiscard]] Sm87MacroFeedV4PhysicalObservationResult
  observe_event_query(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PhysicalObservationResult
  observe_event_synchronize(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;

  [[nodiscard]] bool completion_receipt_matches(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent expected_event,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const noexcept;

  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus close_panel(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access) noexcept;

  // A candidate epoch may be discarded only after both producer streams have
  // recorded tails, Control has waited for both, OwnerDrained has been
  // recorded, and that exact OwnerDrained generation has physically completed.
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus discard_after_drain(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained) noexcept;

  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus complete_request(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& final_panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& final_publish) noexcept;

  // Terminal-only recovery for a CUDA submission/observation failure.  It
  // physically synchronizes every owned stream, requires the request data to
  // be discarded, and keeps this CUDA owner permanently poisoned.  The
  // original cause remains separate from drain success so failure evidence
  // cannot be swallowed or cleared by a later request.
  [[nodiscard]] Sm87MacroFeedV4PoisonDrainResult drain_poisoned_request(
      const Sm87MacroFeedV4ExecutionEventsAccess& access) noexcept;

  struct EventState final {
    std::uint64_t generation = 0U;
    std::uint64_t request_epoch = 0U;
    std::size_t panel = kSm87MacroFeedV4PanelCount;
    std::uint64_t panel_generation = 0U;
    Sm87MacroFeedV4ExecutionStream producer =
        Sm87MacroFeedV4ExecutionStream::kCount;
    bool recorded = false;
    bool dependency_observed = false;
    bool physical_observed = false;
  };

  enum class AbCyclePhase : std::uint8_t {
    kExpectNormRecord = 0U,
    kExpectNormWait,
    kExpectAbRecord,
    kExpectAbWait,
  };

  Sm87MacroFeedV4ExecutionEventsOwner() noexcept;

  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus initialize() noexcept;
  [[nodiscard]] bool owner_access_matches(
      const Sm87MacroFeedV4ExecutionEventsAccess& access) const noexcept;
  [[nodiscard]] bool panel_access_matches(
      const Sm87MacroFeedV4ExecutionPanelAccess& access) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus validate_operation_access(
      const Sm87MacroFeedV4ExecutionEventsAccess& owner_access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus validate_record_order(
      Sm87MacroFeedV4ExecutionStream producer,
      Sm87MacroFeedV4ExecutionEvent event) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus validate_wait_order(
      Sm87MacroFeedV4ExecutionStream consumer,
      Sm87MacroFeedV4ExecutionEvent event) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult record_event_locked(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream producer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  void advance_record_order(Sm87MacroFeedV4ExecutionEvent event) noexcept;
  void advance_wait_order(Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueReceipt mint_enqueue_receipt(
      Sm87MacroFeedV4EnqueueOperation operation,
      Sm87MacroFeedV4ExecutionStream stream,
      Sm87MacroFeedV4ExecutionEvent event,
      std::uint64_t event_generation) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PhysicalCompletionReceipt
  mint_completion_receipt(Sm87MacroFeedV4ExecutionEvent event,
                          bool observed_by_query,
                          bool observed_by_synchronize) noexcept;
  [[nodiscard]] std::uint64_t completion_authenticator(
      const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const noexcept;
  [[nodiscard]] bool completion_receipt_matches_locked(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent expected_event,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const noexcept;
  void record_poison_cause(
      const Sm87MacroFeedV4ExecutionStatus& cause) noexcept;
  void reset_request_ledger() noexcept;
  void release_resources() noexcept;

  Sm87MacroFeedV4ExecutionOwnerState state_ =
      Sm87MacroFeedV4ExecutionOwnerState::kEmpty;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t seal_nonce_ = 0U;
  std::uint64_t receipt_secret_ = 0U;
  std::int32_t device_ordinal_ = -1;
  std::array<void*, kSm87MacroFeedV4ExecutionStreamCount> streams_{};
  std::array<void*, kSm87MacroFeedV4ExecutionEventCount> events_{};
  std::unique_ptr<Sm87MacroFeedV4ExecutionEventsAccess> access_;
  mutable std::mutex mutex_;
  std::uint64_t request_owner_identity_ = 0U;
  std::uint64_t request_allocation_identity_ = 0U;
  std::uint64_t request_epoch_ = 0U;
  std::uint64_t last_request_epoch_ = 0U;
  std::size_t completed_panels_ = 0U;
  std::size_t active_panel_ = kSm87MacroFeedV4PanelCount;
  std::uint64_t next_panel_generation_ = 1U;
  std::uint64_t active_panel_generation_ = 0U;
  std::array<EventState, kSm87MacroFeedV4ExecutionEventCount> event_state_{};
  AbCyclePhase ab_cycle_phase_ = AbCyclePhase::kExpectNormRecord;
  std::size_t bf16_ab_cycles_completed_ = 0U;
  std::size_t enqueue_receipts_issued_ = 0U;
  std::size_t physical_completion_receipts_issued_ = 0U;
  std::size_t bound_kernel_submissions_ = 0U;
  std::size_t input_norm_submissions_ = 0U;
  std::size_t bf16_ab_submissions_ = 0U;
  bool streams_nonblocking_ = false;
  bool panel_done_recorded_ = false;
  bool draining_ = false;
  bool main_tail_recorded_ = false;
  bool ab_tail_recorded_ = false;
  bool main_tail_joined_ = false;
  bool ab_tail_joined_ = false;
  bool owner_drained_recorded_ = false;
  bool final_representation_ready_recorded_ = false;
  bool final_representation_joined_ = false;
  bool canonical_copy_done_recorded_ = false;
  bool canonical_copy_joined_ = false;
  bool final_publish_recorded_ = false;
  Sm87MacroFeedV4ExecutionStatus poison_cause_{};
  std::array<int, kSm87MacroFeedV4ExecutionStreamCount>
      poison_drain_stream_cuda_status_{};
  bool poison_drain_all_stream_synchronizations_attempted_ = false;
  bool poisoned_terminal_quiescence_attested_ = false;

  friend struct Sm87MacroFeedV4ExecutionEventsCreateResult;
  friend Sm87MacroFeedV4ExecutionEventsCreateResult
  create_sm87_macrofeed_v4_execution_events_owner() noexcept;
  friend class Sm87MacroFeedV4ExecutionEventsDriver;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  friend std::unique_ptr<Sm87MacroFeedV4ExecutionEventsDriver>
  bind_sm87_macrofeed_v4_execution_events_driver(
      const std::shared_ptr<Sm87MacroFeedV4ExecutionEventsOwner>& owner)
      noexcept;
#endif
  friend class Sm87MacroFeedV4ExecutionEventsCudaTestFixture;
};

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
// Narrow owner-issued execution capability.  It forwards only the fixed V4
// state machine and typed kernel submissions; no raw stream/event member or
// generic callback is reachable from the execution package.
class Sm87MacroFeedV4ExecutionEventsDriver final {
 public:
  Sm87MacroFeedV4ExecutionEventsDriver() = delete;
  Sm87MacroFeedV4ExecutionEventsDriver(
      const Sm87MacroFeedV4ExecutionEventsDriver&) = delete;
  Sm87MacroFeedV4ExecutionEventsDriver& operator=(
      const Sm87MacroFeedV4ExecutionEventsDriver&) = delete;
  Sm87MacroFeedV4ExecutionEventsDriver(
      Sm87MacroFeedV4ExecutionEventsDriver&&) = delete;
  Sm87MacroFeedV4ExecutionEventsDriver& operator=(
      Sm87MacroFeedV4ExecutionEventsDriver&&) = delete;
  ~Sm87MacroFeedV4ExecutionEventsDriver() = default;

  [[nodiscard]] std::uint64_t owner_identity() const noexcept;
  [[nodiscard]] std::int32_t device_ordinal() const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionEventsSnapshot snapshot()
      const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus begin_request(
      const Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PanelBeginResult begin_panel(
      std::size_t panel) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult record_event(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream producer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult wait_event(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream consumer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_input_norm_and_record_ready(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::Sm87MacroFeedV4InputNormArguments& arguments,
      const kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&
          resources) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_bf16_ab_and_record_ready(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::Sm87MacroFeedV4Bf16AbArguments& arguments,
      const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
          resources) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PhysicalObservationResult
  observe_event_synchronize(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] bool completion_receipt_matches(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent expected_event,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus discard_after_drain(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PoisonDrainResult drain_poisoned_request()
      noexcept;

 private:
  explicit Sm87MacroFeedV4ExecutionEventsDriver(
      std::shared_ptr<Sm87MacroFeedV4ExecutionEventsOwner> owner,
      const Sm87MacroFeedV4ExecutionEventsAccess* access) noexcept
      : owner_(std::move(owner)), access_(access) {}

  std::shared_ptr<Sm87MacroFeedV4ExecutionEventsOwner> owner_;
  const Sm87MacroFeedV4ExecutionEventsAccess* access_ = nullptr;

  friend std::unique_ptr<Sm87MacroFeedV4ExecutionEventsDriver>
  bind_sm87_macrofeed_v4_execution_events_driver(
      const std::shared_ptr<Sm87MacroFeedV4ExecutionEventsOwner>& owner)
      noexcept;
};

[[nodiscard]] std::unique_ptr<Sm87MacroFeedV4ExecutionEventsDriver>
bind_sm87_macrofeed_v4_execution_events_driver(
    const std::shared_ptr<Sm87MacroFeedV4ExecutionEventsOwner>& owner)
    noexcept;
#endif

// Default-off admission factory.  No selector, launcher, public CUDA handle,
// production receipt, JIT, repack, autotune, cuBLASLt, or MTP route is exposed.
[[nodiscard]] Sm87MacroFeedV4ExecutionEventsCreateResult
create_sm87_macrofeed_v4_execution_events_owner() noexcept;

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_EXECUTION_EVENTS_TEST_FIXTURE)
// Device-correctness-only wrappers for private transitions.  They expose no
// CUDA handle and are absent from non-test compilation.
class Sm87MacroFeedV4ExecutionEventsCudaTestFixture final {
 public:
  [[nodiscard]] static std::uint64_t owner_identity(
      const Sm87MacroFeedV4ExecutionEventsOwner& owner) noexcept {
    std::lock_guard<std::mutex> lock(owner.mutex_);
    return owner.access_ == nullptr ? 0U : owner.owner_identity_;
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus begin_request(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept {
    return owner.begin_request(*owner.access_, request_owner, request_access);
  }
  [[nodiscard]] static Sm87MacroFeedV4PanelBeginResult begin_panel(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const std::size_t panel) noexcept {
    return owner.begin_panel(*owner.access_, panel);
  }
  [[nodiscard]] static Sm87MacroFeedV4EventEnqueueResult record_event(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream producer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept {
    return owner.record_event(*owner.access_, panel_access, producer, event);
  }
  [[nodiscard]] static Sm87MacroFeedV4EventEnqueueResult wait_event(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream consumer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept {
    return owner.wait_event(*owner.access_, panel_access, consumer, event);
  }
  [[nodiscard]] static Sm87MacroFeedV4PhysicalObservationResult
  observe_event_query(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent event) noexcept {
    return owner.observe_event_query(*owner.access_, panel_access, event);
  }
  [[nodiscard]] static Sm87MacroFeedV4PhysicalObservationResult
  observe_event_synchronize(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent event) noexcept {
    return owner.observe_event_synchronize(*owner.access_, panel_access,
                                           event);
  }
  [[nodiscard]] static bool completion_receipt_matches(
      const Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent expected_event,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) noexcept {
    return owner.completion_receipt_matches(*owner.access_, panel_access,
                                            expected_event, receipt);
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus close_panel(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access) noexcept {
    return owner.close_panel(*owner.access_, panel_access);
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus discard_after_drain(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained) noexcept {
    return owner.discard_after_drain(*owner.access_, panel_access,
                                     owner_drained);
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus complete_request(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& final_panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& final_publish) noexcept {
    return owner.complete_request(*owner.access_, final_panel_access,
                                  final_publish);
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus
  inject_poison_without_drain(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionError poison_error) noexcept {
    std::lock_guard<std::mutex> lock(owner.mutex_);
    Sm87MacroFeedV4ExecutionStatus cause;
    if (owner.state_ !=
            Sm87MacroFeedV4ExecutionOwnerState::kRequestActive ||
        (poison_error != Sm87MacroFeedV4ExecutionError::kCudaSubmission &&
         poison_error != Sm87MacroFeedV4ExecutionError::kCudaObservation)) {
      cause.error = Sm87MacroFeedV4ExecutionError::kInvalidOwnerState;
      cause.context =
          "test_poison_requires_active_request_and_cuda_error";
      return cause;
    }
    cause.error = poison_error;
    cause.context = "test_only_injected_cuda_failure_without_drain";
    cause.cuda_error = 1;
    owner.record_poison_cause(cause);
    return cause;
  }
  [[nodiscard]] static Sm87MacroFeedV4PoisonDrainResult
  inject_poison_and_drain(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionError poison_error) noexcept {
    {
      std::lock_guard<std::mutex> lock(owner.mutex_);
      if (owner.state_ !=
              Sm87MacroFeedV4ExecutionOwnerState::kRequestActive ||
          (poison_error !=
               Sm87MacroFeedV4ExecutionError::kCudaSubmission &&
           poison_error !=
               Sm87MacroFeedV4ExecutionError::kCudaObservation)) {
        Sm87MacroFeedV4PoisonDrainResult result;
        result.drain_status.error =
            Sm87MacroFeedV4ExecutionError::kInvalidOwnerState;
        result.drain_status.context =
            "test_poison_requires_active_request_and_cuda_error";
        return result;
      }
      Sm87MacroFeedV4ExecutionStatus cause;
      cause.error = poison_error;
      cause.context = "test_only_injected_cuda_failure";
      owner.record_poison_cause(cause);
    }
    return owner.drain_poisoned_request(*owner.access_);
  }
};
#endif

}  // namespace q3x::runtime::sm87_macrofeed_v4_execution_events_detail
