#include "sm87_macrofeed_v4_execution_events_internal.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace q3x::runtime::sm87_macrofeed_v4_execution_events_detail {
namespace {

std::atomic<std::uint64_t> g_next_owner_identity{1U};
std::atomic<std::uint64_t> g_next_seal_nonce{1U};
std::atomic<std::uint64_t> g_next_enqueue_identity{1U};
std::atomic<std::uint64_t> g_next_completion_identity{1U};

[[nodiscard]] std::uint64_t next_nonzero(
    std::atomic<std::uint64_t>* const source) noexcept {
  std::uint64_t value = source->fetch_add(1U, std::memory_order_relaxed);
  if (value == 0U) {
    value = source->fetch_add(1U, std::memory_order_relaxed);
  }
  return value;
}

[[nodiscard]] constexpr std::size_t stream_index(
    const Sm87MacroFeedV4ExecutionStream stream) noexcept {
  return static_cast<std::size_t>(stream);
}

[[nodiscard]] constexpr std::size_t event_index(
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  return static_cast<std::size_t>(event);
}

[[nodiscard]] constexpr bool valid_stream(
    const Sm87MacroFeedV4ExecutionStream stream) noexcept {
  return stream_index(stream) < kSm87MacroFeedV4ExecutionStreamCount;
}

[[nodiscard]] constexpr bool valid_event(
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  return event_index(event) < kSm87MacroFeedV4ExecutionEventCount;
}

[[nodiscard]] constexpr bool physical_observation_allowed(
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  return event == Sm87MacroFeedV4ExecutionEvent::kOwnerDrained ||
         event == Sm87MacroFeedV4ExecutionEvent::kFinalPublish;
}

[[nodiscard]] constexpr bool terminal_stream_boundary_observed(
    const cudaError_t status) noexcept {
  switch (status) {
    case cudaSuccess:
    // cudaStreamSynchronize may report a prior asynchronous execution or
    // launch failure after it has reached the stream's terminal observation
    // boundary.  These statuses prove drain completion but never execution
    // success; every raw status remains in the terminal receipt.
    case cudaErrorInvalidConfiguration:
    case cudaErrorInvalidDeviceFunction:
    case cudaErrorLaunchOutOfResources:
    case cudaErrorLaunchTimeout:
    case cudaErrorAssert:
    case cudaErrorHardwareStackError:
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] constexpr Sm87MacroFeedV4ExecutionStatus ok() noexcept {
  return {};
}

[[nodiscard]] constexpr Sm87MacroFeedV4ExecutionStatus fail(
    const Sm87MacroFeedV4ExecutionError error, const char* const context,
    const int cuda_error = 0,
    const Sm87MacroFeedV4ExecutionStream stream =
        Sm87MacroFeedV4ExecutionStream::kCount,
    const Sm87MacroFeedV4ExecutionEvent event =
        Sm87MacroFeedV4ExecutionEvent::kCount,
    const std::size_t panel = kSm87MacroFeedV4PanelCount,
    const std::uint64_t panel_generation = 0U) noexcept {
  return {error, context, cuda_error, stream, event, panel,
          panel_generation};
}

[[nodiscard]] constexpr Sm87MacroFeedV4ExecutionStream expected_producer(
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  switch (event) {
    case Sm87MacroFeedV4ExecutionEvent::kNormReady:
    case Sm87MacroFeedV4ExecutionEvent::kPanelDone:
    case Sm87MacroFeedV4ExecutionEvent::kMainTail:
    case Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady:
    case Sm87MacroFeedV4ExecutionEvent::kFinalPublish:
      return Sm87MacroFeedV4ExecutionStream::kMain;
    case Sm87MacroFeedV4ExecutionEvent::kAbReady:
    case Sm87MacroFeedV4ExecutionEvent::kAbTail:
      return Sm87MacroFeedV4ExecutionStream::kAbAux;
    case Sm87MacroFeedV4ExecutionEvent::kOwnerDrained:
    case Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone:
      return Sm87MacroFeedV4ExecutionStream::kControl;
    case Sm87MacroFeedV4ExecutionEvent::kCount:
      break;
  }
  return Sm87MacroFeedV4ExecutionStream::kCount;
}

[[nodiscard]] constexpr Sm87MacroFeedV4ExecutionStream expected_consumer(
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  switch (event) {
    case Sm87MacroFeedV4ExecutionEvent::kNormReady:
      return Sm87MacroFeedV4ExecutionStream::kAbAux;
    case Sm87MacroFeedV4ExecutionEvent::kAbReady:
    case Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone:
      return Sm87MacroFeedV4ExecutionStream::kMain;
    case Sm87MacroFeedV4ExecutionEvent::kMainTail:
    case Sm87MacroFeedV4ExecutionEvent::kAbTail:
    case Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady:
      return Sm87MacroFeedV4ExecutionStream::kControl;
    case Sm87MacroFeedV4ExecutionEvent::kPanelDone:
    case Sm87MacroFeedV4ExecutionEvent::kOwnerDrained:
    case Sm87MacroFeedV4ExecutionEvent::kFinalPublish:
    case Sm87MacroFeedV4ExecutionEvent::kCount:
      break;
  }
  return Sm87MacroFeedV4ExecutionStream::kCount;
}

[[nodiscard]] constexpr std::uint64_t mix64(std::uint64_t value) noexcept {
  value += 0x9e37'79b9'7f4a'7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58'476d'1ce4'e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d0'49bb'1331'11ebULL;
  return value ^ (value >> 31U);
}

}  // namespace

Sm87MacroFeedV4ExecutionEventsAccess::
    Sm87MacroFeedV4ExecutionEventsAccess(
        const Sm87MacroFeedV4ExecutionEventsOwner* const owner,
        const std::uint64_t owner_identity, const std::uint64_t seal_nonce,
        const std::int32_t device_ordinal) noexcept
    : owner_(owner),
      owner_identity_(owner_identity),
      seal_nonce_(seal_nonce),
      device_ordinal_(device_ordinal) {}

Sm87MacroFeedV4ExecutionPanelAccess::Sm87MacroFeedV4ExecutionPanelAccess(
    const Sm87MacroFeedV4ExecutionEventsOwner* const owner,
    const std::uint64_t owner_identity, const std::uint64_t seal_nonce,
    const std::uint64_t request_epoch, const std::size_t panel,
    const std::uint64_t panel_generation) noexcept
    : owner_(owner),
      owner_identity_(owner_identity),
      seal_nonce_(seal_nonce),
      request_epoch_(request_epoch),
      panel_(panel),
      panel_generation_(panel_generation) {}

Sm87MacroFeedV4ExecutionEventsCreateResult::operator bool() const noexcept {
  return owner != nullptr && static_cast<bool>(status) &&
         owner->access() != nullptr &&
         owner->snapshot().state == Sm87MacroFeedV4ExecutionOwnerState::kReady;
}

Sm87MacroFeedV4ExecutionEventsOwner::
    Sm87MacroFeedV4ExecutionEventsOwner() noexcept
    : owner_identity_(next_nonzero(&g_next_owner_identity)),
      seal_nonce_(next_nonzero(&g_next_seal_nonce)),
      receipt_secret_(mix64(owner_identity_ ^ (seal_nonce_ << 1U) ^
                            0x5133'4d46'5634'4556ULL)) {}

Sm87MacroFeedV4ExecutionEventsOwner::~Sm87MacroFeedV4ExecutionEventsOwner() {
  release_resources();
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsOwner::initialize() noexcept {
#if !defined(Q3X_ENABLE_SM87_MACROFEED_V4_EXECUTION_EVENTS_ADMISSION)
  return fail(Sm87MacroFeedV4ExecutionError::kAdmissionDisabled,
              "macrofeed_v4_execution_events_admission_disabled");
#else
  if (state_ != Sm87MacroFeedV4ExecutionOwnerState::kEmpty) {
    return fail(Sm87MacroFeedV4ExecutionError::kInvalidOwnerState,
                "execution_events_initialize_once");
  }

  cudaError_t cuda_status = cudaGetDevice(&device_ordinal_);
  if (cuda_status != cudaSuccess) {
    return fail(Sm87MacroFeedV4ExecutionError::kDeviceQuery,
                "cudaGetDevice", static_cast<int>(cuda_status));
  }
  cudaDeviceProp properties{};
  cuda_status = cudaGetDeviceProperties(&properties, device_ordinal_);
  if (cuda_status != cudaSuccess) {
    return fail(Sm87MacroFeedV4ExecutionError::kDeviceQuery,
                "cudaGetDeviceProperties", static_cast<int>(cuda_status));
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    return fail(Sm87MacroFeedV4ExecutionError::kWrongDevice,
                "exact_sm87_16sm_required");
  }

  for (std::size_t index = 0U; index < streams_.size(); ++index) {
    cudaStream_t stream = nullptr;
    cuda_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess || stream == nullptr) {
      return fail(Sm87MacroFeedV4ExecutionError::kStreamCreate,
                  "cudaStreamCreateWithFlags", static_cast<int>(cuda_status),
                  static_cast<Sm87MacroFeedV4ExecutionStream>(index));
    }
    streams_[index] = reinterpret_cast<void*>(stream);
    unsigned int flags = 0U;
    cuda_status = cudaStreamGetFlags(stream, &flags);
    if (cuda_status != cudaSuccess ||
        (flags & cudaStreamNonBlocking) == 0U) {
      return fail(Sm87MacroFeedV4ExecutionError::kStreamValidation,
                  "exact_nonblocking_stream_required",
                  static_cast<int>(cuda_status),
                  static_cast<Sm87MacroFeedV4ExecutionStream>(index));
    }
  }
  streams_nonblocking_ = true;

  for (std::size_t index = 0U; index < events_.size(); ++index) {
    cudaEvent_t event = nullptr;
    cuda_status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (cuda_status != cudaSuccess || event == nullptr) {
      return fail(Sm87MacroFeedV4ExecutionError::kEventCreate,
                  "cudaEventCreateWithFlags", static_cast<int>(cuda_status),
                  Sm87MacroFeedV4ExecutionStream::kCount,
                  static_cast<Sm87MacroFeedV4ExecutionEvent>(index));
    }
    events_[index] = reinterpret_cast<void*>(event);
  }

  access_.reset(new (std::nothrow) Sm87MacroFeedV4ExecutionEventsAccess(
      this, owner_identity_, seal_nonce_, device_ordinal_));
  if (access_ == nullptr) {
    return fail(Sm87MacroFeedV4ExecutionError::kInvalidOwnerState,
                "execution_access_allocation");
  }
  state_ = Sm87MacroFeedV4ExecutionOwnerState::kReady;
  return ok();
#endif
}

bool Sm87MacroFeedV4ExecutionEventsOwner::owner_access_matches(
    const Sm87MacroFeedV4ExecutionEventsAccess& access) const noexcept {
  return access_ != nullptr && &access == access_.get() &&
         access.owner_ == this && access.owner_identity_ == owner_identity_ &&
         access.seal_nonce_ == seal_nonce_ &&
         access.device_ordinal_ == device_ordinal_;
}

bool Sm87MacroFeedV4ExecutionEventsOwner::panel_access_matches(
    const Sm87MacroFeedV4ExecutionPanelAccess& access) const noexcept {
  return access.owner_ == this && access.owner_identity_ == owner_identity_ &&
         access.seal_nonce_ == seal_nonce_ &&
         access.request_epoch_ == request_epoch_ && request_epoch_ != 0U &&
         access.panel_ == active_panel_ &&
         access.panel_generation_ == active_panel_generation_ &&
         active_panel_ < kSm87MacroFeedV4PanelCount &&
         active_panel_generation_ != 0U;
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsOwner::validate_operation_access(
    const Sm87MacroFeedV4ExecutionEventsAccess& owner_access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access) const noexcept {
  if (!owner_access_matches(owner_access)) {
    return fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                "owner_issued_execution_access_required");
  }
  if (state_ != Sm87MacroFeedV4ExecutionOwnerState::kRequestActive) {
    return fail(Sm87MacroFeedV4ExecutionError::kInvalidOwnerState,
                "active_request_required");
  }
  if (!panel_access_matches(panel_access)) {
    return fail(Sm87MacroFeedV4ExecutionError::kStalePanelGeneration,
                "owner_issued_current_panel_generation_required", 0,
                Sm87MacroFeedV4ExecutionStream::kCount,
                Sm87MacroFeedV4ExecutionEvent::kCount, panel_access.panel_,
                panel_access.panel_generation_);
  }
  return ok();
}

void Sm87MacroFeedV4ExecutionEventsOwner::reset_request_ledger() noexcept {
  completed_panels_ = 0U;
  active_panel_ = kSm87MacroFeedV4PanelCount;
  next_panel_generation_ = 1U;
  active_panel_generation_ = 0U;
  event_state_.fill({});
  ab_cycle_phase_ = AbCyclePhase::kExpectNormRecord;
  bf16_ab_cycles_completed_ = 0U;
  enqueue_receipts_issued_ = 0U;
  physical_completion_receipts_issued_ = 0U;
  panel_done_recorded_ = false;
  draining_ = false;
  main_tail_recorded_ = false;
  ab_tail_recorded_ = false;
  main_tail_joined_ = false;
  ab_tail_joined_ = false;
  owner_drained_recorded_ = false;
  final_representation_ready_recorded_ = false;
  final_representation_joined_ = false;
  canonical_copy_done_recorded_ = false;
  canonical_copy_joined_ = false;
  final_publish_recorded_ = false;
  poison_cause_ = {};
  poison_drain_stream_cuda_status_.fill(0);
  poison_drain_all_stream_synchronizations_attempted_ = false;
  poisoned_terminal_quiescence_attested_ = false;
}

void Sm87MacroFeedV4ExecutionEventsOwner::record_poison_cause(
    const Sm87MacroFeedV4ExecutionStatus& cause) noexcept {
  if (poison_cause_.error == Sm87MacroFeedV4ExecutionError::kNone) {
    poison_cause_ = cause;
  }
  state_ = Sm87MacroFeedV4ExecutionOwnerState::kPoisoned;
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsOwner::begin_request(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4RequestState& request_owner,
    const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!owner_access_matches(access)) {
    return fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                "begin_request_owner_issued_access");
  }
  if (state_ != Sm87MacroFeedV4ExecutionOwnerState::kReady &&
      state_ != Sm87MacroFeedV4ExecutionOwnerState::kRequestCompleted &&
      state_ != Sm87MacroFeedV4ExecutionOwnerState::kRequestDiscarded) {
    return fail(Sm87MacroFeedV4ExecutionError::kInvalidOwnerState,
                "begin_request_requires_quiesced_owner");
  }
  std::uint64_t allocation_identity = 0U;
  std::uint64_t request_epoch = 0U;
  const auto request_validation =
      request_owner.validate_execution_begin_access(
          request_access, owner_identity_, &allocation_identity,
          &request_epoch);
  if (!request_validation) {
    return fail(Sm87MacroFeedV4ExecutionError::kForeignRequestAccess,
                "live_admitted_request_state_and_sealed_access_required");
  }
  if (request_epoch == 0U || request_epoch <= last_request_epoch_) {
    return fail(Sm87MacroFeedV4ExecutionError::kInvalidRequestEpoch,
                "fresh_monotonic_request_epoch_required");
  }

  reset_request_ledger();
  request_owner_identity_ = owner_identity_;
  request_allocation_identity_ = allocation_identity;
  request_epoch_ = request_epoch;
  last_request_epoch_ = request_epoch_;
  state_ = Sm87MacroFeedV4ExecutionOwnerState::kRequestActive;
  return ok();
}

Sm87MacroFeedV4PanelBeginResult
Sm87MacroFeedV4ExecutionEventsOwner::begin_panel(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const std::size_t panel) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4PanelBeginResult result;
  if (!owner_access_matches(access)) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                         "begin_panel_owner_issued_access");
    return result;
  }
  if (state_ != Sm87MacroFeedV4ExecutionOwnerState::kRequestActive ||
      active_panel_ != kSm87MacroFeedV4PanelCount || draining_) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kInvalidOwnerState,
                         "begin_panel_requires_idle_active_request");
    return result;
  }
  if (panel >= kSm87MacroFeedV4PanelCount || panel != completed_panels_) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kInvalidPanel,
                         "panels_must_begin_in_fixed_zero_to_four_order", 0,
                         Sm87MacroFeedV4ExecutionStream::kCount,
                         Sm87MacroFeedV4ExecutionEvent::kCount, panel);
    return result;
  }

  // PanelDone is a device-ordered marker, not a per-panel host barrier.  No
  // consumer waits on it and no physical receipt is required to advance the
  // enqueue ledger.  CUDA permits re-recording the same event; invalidate the
  // retired host generation here while preserving its monotonic generation
  // counter.  Stale panel capabilities cannot observe the replacement.
  EventState& panel_done =
      event_state_[event_index(Sm87MacroFeedV4ExecutionEvent::kPanelDone)];
  panel_done.request_epoch = 0U;
  panel_done.panel = kSm87MacroFeedV4PanelCount;
  panel_done.panel_generation = 0U;
  panel_done.producer = Sm87MacroFeedV4ExecutionStream::kCount;
  panel_done.recorded = false;
  panel_done.dependency_observed = false;
  panel_done.physical_observed = false;

  active_panel_ = panel;
  active_panel_generation_ = next_panel_generation_++;
  ab_cycle_phase_ = AbCyclePhase::kExpectNormRecord;
  bf16_ab_cycles_completed_ = 0U;
  panel_done_recorded_ = false;
  draining_ = false;
  main_tail_recorded_ = false;
  ab_tail_recorded_ = false;
  main_tail_joined_ = false;
  ab_tail_joined_ = false;
  owner_drained_recorded_ = false;

  result.panel_access.reset(
      new (std::nothrow) Sm87MacroFeedV4ExecutionPanelAccess(
          this, owner_identity_, seal_nonce_, request_epoch_, active_panel_,
          active_panel_generation_));
  if (result.panel_access == nullptr) {
    active_panel_ = kSm87MacroFeedV4PanelCount;
    active_panel_generation_ = 0U;
    result.status = fail(Sm87MacroFeedV4ExecutionError::kInvalidOwnerState,
                         "panel_access_allocation");
    return result;
  }
  result.status = ok();
  return result;
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsOwner::validate_record_order(
    const Sm87MacroFeedV4ExecutionStream producer,
    const Sm87MacroFeedV4ExecutionEvent event) const noexcept {
  if (!valid_stream(producer) || !valid_event(event) ||
      producer != expected_producer(event)) {
    return fail(Sm87MacroFeedV4ExecutionError::kWrongStream,
                "event_has_one_fixed_producer_stream", 0, producer, event,
                active_panel_, active_panel_generation_);
  }

  switch (event) {
    case Sm87MacroFeedV4ExecutionEvent::kNormReady:
      if (draining_ || completed_panels_ >= kSm87MacroFeedV4PanelCount ||
          ab_cycle_phase_ != AbCyclePhase::kExpectNormRecord ||
          bf16_ab_cycles_completed_ >= kSm87MacroFeedV4Bf16AbCyclesPerPanel) {
        break;
      }
      return ok();
    case Sm87MacroFeedV4ExecutionEvent::kAbReady:
      if (!draining_ && ab_cycle_phase_ == AbCyclePhase::kExpectAbRecord) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kPanelDone:
      if (!draining_ && !panel_done_recorded_ &&
          bf16_ab_cycles_completed_ ==
              kSm87MacroFeedV4Bf16AbCyclesPerPanel &&
          ab_cycle_phase_ == AbCyclePhase::kExpectNormRecord) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kMainTail:
      if (!main_tail_recorded_ && !owner_drained_recorded_) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kAbTail:
      if (!ab_tail_recorded_ && !owner_drained_recorded_) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kOwnerDrained:
      if (draining_ && main_tail_joined_ && ab_tail_joined_ &&
          !owner_drained_recorded_) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady:
      if (!draining_ && completed_panels_ == kSm87MacroFeedV4PanelCount &&
          !final_representation_ready_recorded_) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone:
      if (!draining_ && final_representation_joined_ &&
          !canonical_copy_done_recorded_) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kFinalPublish:
      if (!draining_ && canonical_copy_joined_ && !final_publish_recorded_) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kCount:
      break;
  }
  return fail(Sm87MacroFeedV4ExecutionError::kInvalidEventOrder,
              "record_event_violates_fixed_v4_order", 0, producer, event,
              active_panel_, active_panel_generation_);
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsOwner::validate_wait_order(
    const Sm87MacroFeedV4ExecutionStream consumer,
    const Sm87MacroFeedV4ExecutionEvent event) const noexcept {
  if (!valid_stream(consumer) || !valid_event(event) ||
      consumer != expected_consumer(event)) {
    return fail(Sm87MacroFeedV4ExecutionError::kWrongStream,
                "event_has_one_fixed_consumer_stream", 0, consumer, event,
                active_panel_, active_panel_generation_);
  }
  switch (event) {
    case Sm87MacroFeedV4ExecutionEvent::kNormReady:
      if (!draining_ && ab_cycle_phase_ == AbCyclePhase::kExpectNormWait) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kAbReady:
      if (!draining_ && ab_cycle_phase_ == AbCyclePhase::kExpectAbWait) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kMainTail:
      if (draining_ && main_tail_recorded_ && !main_tail_joined_) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kAbTail:
      if (draining_ && ab_tail_recorded_ && !ab_tail_joined_) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady:
      if (!draining_ && final_representation_ready_recorded_ &&
          !final_representation_joined_) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone:
      if (!draining_ && canonical_copy_done_recorded_ &&
          !canonical_copy_joined_) {
        return ok();
      }
      break;
    case Sm87MacroFeedV4ExecutionEvent::kPanelDone:
    case Sm87MacroFeedV4ExecutionEvent::kOwnerDrained:
    case Sm87MacroFeedV4ExecutionEvent::kFinalPublish:
    case Sm87MacroFeedV4ExecutionEvent::kCount:
      break;
  }
  return fail(Sm87MacroFeedV4ExecutionError::kInvalidEventOrder,
              "wait_event_violates_fixed_v4_order", 0, consumer, event,
              active_panel_, active_panel_generation_);
}

void Sm87MacroFeedV4ExecutionEventsOwner::advance_record_order(
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  switch (event) {
    case Sm87MacroFeedV4ExecutionEvent::kNormReady:
      ab_cycle_phase_ = AbCyclePhase::kExpectNormWait;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kAbReady:
      ab_cycle_phase_ = AbCyclePhase::kExpectAbWait;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kPanelDone:
      panel_done_recorded_ = true;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kMainTail:
      draining_ = true;
      main_tail_recorded_ = true;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kAbTail:
      draining_ = true;
      ab_tail_recorded_ = true;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kOwnerDrained:
      owner_drained_recorded_ = true;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady:
      final_representation_ready_recorded_ = true;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone:
      canonical_copy_done_recorded_ = true;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kFinalPublish:
      final_publish_recorded_ = true;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kCount:
      break;
  }
}

void Sm87MacroFeedV4ExecutionEventsOwner::advance_wait_order(
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  switch (event) {
    case Sm87MacroFeedV4ExecutionEvent::kNormReady:
      ab_cycle_phase_ = AbCyclePhase::kExpectAbRecord;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kAbReady:
      ab_cycle_phase_ = AbCyclePhase::kExpectNormRecord;
      ++bf16_ab_cycles_completed_;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kMainTail:
      main_tail_joined_ = true;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kAbTail:
      ab_tail_joined_ = true;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady:
      final_representation_joined_ = true;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone:
      canonical_copy_joined_ = true;
      break;
    case Sm87MacroFeedV4ExecutionEvent::kPanelDone:
    case Sm87MacroFeedV4ExecutionEvent::kOwnerDrained:
    case Sm87MacroFeedV4ExecutionEvent::kFinalPublish:
    case Sm87MacroFeedV4ExecutionEvent::kCount:
      break;
  }
}

Sm87MacroFeedV4EventEnqueueReceipt
Sm87MacroFeedV4ExecutionEventsOwner::mint_enqueue_receipt(
    const Sm87MacroFeedV4EnqueueOperation operation,
    const Sm87MacroFeedV4ExecutionStream stream,
    const Sm87MacroFeedV4ExecutionEvent event,
    const std::uint64_t event_generation) noexcept {
  Sm87MacroFeedV4EventEnqueueReceipt receipt;
  receipt.enqueue_identity = next_nonzero(&g_next_enqueue_identity);
  receipt.owner_identity = owner_identity_;
  receipt.request_epoch = request_epoch_;
  receipt.panel = active_panel_;
  receipt.panel_generation = active_panel_generation_;
  receipt.stream = stream;
  receipt.event = event;
  receipt.event_generation = event_generation;
  receipt.operation = operation;
  receipt.cuda_enqueue_accepted = true;
  receipt.physical_device_completion_attested = false;
  receipt.production_receipt_eligible = false;
  ++enqueue_receipts_issued_;
  return receipt;
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::record_event(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionStream producer,
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4EventEnqueueResult result;
  result.status = validate_operation_access(access, panel_access);
  if (!result.status) {
    return result;
  }
  if (!valid_event(event)) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kInvalidEventOrder,
                         "record_requires_known_event", 0, producer, event,
                         active_panel_, active_panel_generation_);
    return result;
  }
  EventState& event_state = event_state_[event_index(event)];
  if (event_state.recorded && !event_state.dependency_observed &&
      !event_state.physical_observed) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kEventGenerationUnobserved,
        "event_generation_must_be_observed_before_rerecord", 0, producer,
        event, active_panel_, active_panel_generation_);
    return result;
  }
  result.status = validate_record_order(producer, event);
  if (!result.status) {
    return result;
  }

  const cudaError_t cuda_status = cudaEventRecord(
      reinterpret_cast<cudaEvent_t>(events_[event_index(event)]),
      reinterpret_cast<cudaStream_t>(streams_[stream_index(producer)]));
  if (cuda_status != cudaSuccess) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kCudaSubmission,
                         "cudaEventRecord", static_cast<int>(cuda_status),
                         producer, event, active_panel_,
                         active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }

  ++event_state.generation;
  event_state.request_epoch = request_epoch_;
  event_state.panel = active_panel_;
  event_state.panel_generation = active_panel_generation_;
  event_state.producer = producer;
  event_state.recorded = true;
  event_state.dependency_observed = false;
  event_state.physical_observed = false;
  advance_record_order(event);
  result.receipt = mint_enqueue_receipt(
      Sm87MacroFeedV4EnqueueOperation::kRecord, producer, event,
      event_state.generation);
  result.status = ok();
  return result;
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::wait_event(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionStream consumer,
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4EventEnqueueResult result;
  result.status = validate_operation_access(access, panel_access);
  if (!result.status) {
    return result;
  }
  if (!valid_event(event)) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kEventNotRecorded,
                         "wait_requires_known_event", 0, consumer, event,
                         active_panel_, active_panel_generation_);
    return result;
  }
  EventState& event_state = event_state_[event_index(event)];
  if (!event_state.recorded || event_state.request_epoch != request_epoch_ ||
      event_state.panel != active_panel_ ||
      event_state.panel_generation != active_panel_generation_) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kEventNotRecorded,
                         "wait_requires_current_panel_event_generation", 0,
                         consumer, event, active_panel_,
                         active_panel_generation_);
    return result;
  }
  result.status = validate_wait_order(consumer, event);
  if (!result.status) {
    return result;
  }

  const cudaError_t cuda_status = cudaStreamWaitEvent(
      reinterpret_cast<cudaStream_t>(streams_[stream_index(consumer)]),
      reinterpret_cast<cudaEvent_t>(events_[event_index(event)]), 0U);
  if (cuda_status != cudaSuccess) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kCudaSubmission,
                         "cudaStreamWaitEvent",
                         static_cast<int>(cuda_status), consumer, event,
                         active_panel_, active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  event_state.dependency_observed = true;
  advance_wait_order(event);
  result.receipt = mint_enqueue_receipt(
      Sm87MacroFeedV4EnqueueOperation::kWait, consumer, event,
      event_state.generation);
  result.status = ok();
  return result;
}

std::uint64_t Sm87MacroFeedV4ExecutionEventsOwner::completion_authenticator(
    const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const noexcept {
  std::uint64_t value = receipt_secret_;
  value = mix64(value ^ receipt.receipt_identity_);
  value = mix64(value ^ receipt.owner_identity_);
  value = mix64(value ^ receipt.request_epoch_);
  value = mix64(value ^ static_cast<std::uint64_t>(receipt.panel_));
  value = mix64(value ^ receipt.panel_generation_);
  value = mix64(value ^ static_cast<std::uint64_t>(receipt.producer_));
  value = mix64(value ^ static_cast<std::uint64_t>(receipt.event_));
  value = mix64(value ^ receipt.event_generation_);
  value = mix64(value ^ receipt.main_tail_generation_);
  value = mix64(value ^ receipt.ab_tail_generation_);
  value = mix64(value ^ static_cast<std::uint64_t>(
                            receipt.observed_by_query_));
  value = mix64(value ^ static_cast<std::uint64_t>(
                            receipt.observed_by_synchronize_));
  return value;
}

Sm87MacroFeedV4PhysicalCompletionReceipt
Sm87MacroFeedV4ExecutionEventsOwner::mint_completion_receipt(
    const Sm87MacroFeedV4ExecutionEvent event,
    const bool observed_by_query,
    const bool observed_by_synchronize) noexcept {
  const EventState& event_state = event_state_[event_index(event)];
  Sm87MacroFeedV4PhysicalCompletionReceipt receipt;
  receipt.receipt_identity_ = next_nonzero(&g_next_completion_identity);
  receipt.owner_identity_ = owner_identity_;
  receipt.request_epoch_ = request_epoch_;
  receipt.panel_ = active_panel_;
  receipt.panel_generation_ = active_panel_generation_;
  receipt.producer_ = event_state.producer;
  receipt.event_ = event;
  receipt.event_generation_ = event_state.generation;
  if (event == Sm87MacroFeedV4ExecutionEvent::kOwnerDrained) {
    receipt.main_tail_generation_ =
        event_state_[event_index(Sm87MacroFeedV4ExecutionEvent::kMainTail)]
            .generation;
    receipt.ab_tail_generation_ =
        event_state_[event_index(Sm87MacroFeedV4ExecutionEvent::kAbTail)]
            .generation;
  }
  receipt.observed_by_query_ = observed_by_query;
  receipt.observed_by_synchronize_ = observed_by_synchronize;
  receipt.physical_device_completion_attested_ = true;
  receipt.production_receipt_eligible_ = false;
  receipt.authenticator_ = completion_authenticator(receipt);
  ++physical_completion_receipts_issued_;
  return receipt;
}

Sm87MacroFeedV4PhysicalObservationResult
Sm87MacroFeedV4ExecutionEventsOwner::observe_event_query(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4PhysicalObservationResult result;
  result.status = validate_operation_access(access, panel_access);
  if (!result.status) {
    return result;
  }
  if (!valid_event(event)) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kEventNotRecorded,
                         "query_requires_known_event", 0,
                         Sm87MacroFeedV4ExecutionStream::kCount, event,
                         active_panel_, active_panel_generation_);
    return result;
  }
  if (!physical_observation_allowed(event)) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kPhysicalObservationForbidden,
        "query_forbidden_for_device_order_only_event", 0,
        Sm87MacroFeedV4ExecutionStream::kCount, event, active_panel_,
        active_panel_generation_);
    return result;
  }
  EventState& event_state = event_state_[event_index(event)];
  if (!event_state.recorded || event_state.request_epoch != request_epoch_ ||
      event_state.panel != active_panel_ ||
      event_state.panel_generation != active_panel_generation_) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kEventNotRecorded,
                         "query_requires_current_panel_event_generation", 0,
                         Sm87MacroFeedV4ExecutionStream::kCount, event,
                         active_panel_, active_panel_generation_);
    return result;
  }
  if (event_state.physical_observed) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kEventAlreadyObserved,
                         "one_completion_receipt_per_event_generation", 0,
                         event_state.producer, event, active_panel_,
                         active_panel_generation_);
    return result;
  }
  const cudaError_t cuda_status = cudaEventQuery(
      reinterpret_cast<cudaEvent_t>(events_[event_index(event)]));
  if (cuda_status == cudaErrorNotReady) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kEventNotComplete,
                         "cudaEventQuery_not_ready",
                         static_cast<int>(cuda_status), event_state.producer,
                         event, active_panel_, active_panel_generation_);
    return result;
  }
  if (cuda_status != cudaSuccess) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kCudaObservation,
                         "cudaEventQuery", static_cast<int>(cuda_status),
                         event_state.producer, event, active_panel_,
                         active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  event_state.physical_observed = true;
  result.receipt = mint_completion_receipt(event, true, false);
  result.status = ok();
  return result;
}

Sm87MacroFeedV4PhysicalObservationResult
Sm87MacroFeedV4ExecutionEventsOwner::observe_event_synchronize(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4PhysicalObservationResult result;
  result.status = validate_operation_access(access, panel_access);
  if (!result.status) {
    return result;
  }
  if (!valid_event(event)) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kEventNotRecorded,
                         "synchronize_requires_known_event", 0,
                         Sm87MacroFeedV4ExecutionStream::kCount, event,
                         active_panel_, active_panel_generation_);
    return result;
  }
  if (!physical_observation_allowed(event)) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kPhysicalObservationForbidden,
        "synchronize_forbidden_for_device_order_only_event", 0,
        Sm87MacroFeedV4ExecutionStream::kCount, event, active_panel_,
        active_panel_generation_);
    return result;
  }
  EventState& event_state = event_state_[event_index(event)];
  if (!event_state.recorded || event_state.request_epoch != request_epoch_ ||
      event_state.panel != active_panel_ ||
      event_state.panel_generation != active_panel_generation_) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kEventNotRecorded,
        "synchronize_requires_current_panel_event_generation", 0,
        Sm87MacroFeedV4ExecutionStream::kCount, event, active_panel_,
        active_panel_generation_);
    return result;
  }
  if (event_state.physical_observed) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kEventAlreadyObserved,
                         "one_completion_receipt_per_event_generation", 0,
                         event_state.producer, event, active_panel_,
                         active_panel_generation_);
    return result;
  }
  const cudaError_t cuda_status = cudaEventSynchronize(
      reinterpret_cast<cudaEvent_t>(events_[event_index(event)]));
  if (cuda_status != cudaSuccess) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kCudaObservation,
                         "cudaEventSynchronize",
                         static_cast<int>(cuda_status), event_state.producer,
                         event, active_panel_, active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  event_state.physical_observed = true;
  result.receipt = mint_completion_receipt(event, false, true);
  result.status = ok();
  return result;
}

bool Sm87MacroFeedV4ExecutionEventsOwner::
    completion_receipt_matches_locked(
        const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
        const Sm87MacroFeedV4ExecutionEvent expected_event,
        const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const
    noexcept {
  if (!panel_access_matches(panel_access) || !valid_event(expected_event)) {
    return false;
  }
  const EventState& event_state = event_state_[event_index(expected_event)];
  return receipt.receipt_identity_ != 0U &&
         receipt.owner_identity_ == owner_identity_ &&
         receipt.request_epoch_ == request_epoch_ &&
         receipt.panel_ == active_panel_ &&
         receipt.panel_generation_ == active_panel_generation_ &&
         receipt.producer_ == event_state.producer &&
         receipt.event_ == expected_event &&
         receipt.event_generation_ == event_state.generation &&
         receipt.observed_by_query_ != receipt.observed_by_synchronize_ &&
         receipt.physical_device_completion_attested_ &&
         !receipt.production_receipt_eligible_ && event_state.recorded &&
         event_state.physical_observed &&
         receipt.authenticator_ == completion_authenticator(receipt);
}

bool Sm87MacroFeedV4ExecutionEventsOwner::completion_receipt_matches(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionEvent expected_event,
    const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return owner_access_matches(access) &&
         state_ == Sm87MacroFeedV4ExecutionOwnerState::kRequestActive &&
         completion_receipt_matches_locked(panel_access, expected_event,
                                           receipt);
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsOwner::close_panel(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4ExecutionStatus status =
      validate_operation_access(access, panel_access);
  if (!status) {
    return status;
  }
  if (completed_panels_ >= kSm87MacroFeedV4PanelCount ||
      active_panel_ != completed_panels_ || !panel_done_recorded_ ||
      draining_ ||
      bf16_ab_cycles_completed_ != kSm87MacroFeedV4Bf16AbCyclesPerPanel ||
      ab_cycle_phase_ != AbCyclePhase::kExpectNormRecord) {
    return fail(Sm87MacroFeedV4ExecutionError::kPanelIncomplete,
                "panel_requires_all_48_ab_cycles_and_panel_done", 0,
                Sm87MacroFeedV4ExecutionStream::kMain,
                Sm87MacroFeedV4ExecutionEvent::kPanelDone, active_panel_,
                active_panel_generation_);
  }

  ++completed_panels_;
  if (completed_panels_ < kSm87MacroFeedV4PanelCount) {
    active_panel_ = kSm87MacroFeedV4PanelCount;
    active_panel_generation_ = 0U;
  }
  return ok();
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsOwner::discard_after_drain(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4ExecutionStatus status =
      validate_operation_access(access, panel_access);
  if (!status) {
    return status;
  }
  const std::uint64_t main_generation =
      event_state_[event_index(Sm87MacroFeedV4ExecutionEvent::kMainTail)]
          .generation;
  const std::uint64_t ab_generation =
      event_state_[event_index(Sm87MacroFeedV4ExecutionEvent::kAbTail)]
          .generation;
  if (!draining_ || !main_tail_recorded_ || !ab_tail_recorded_ ||
      !main_tail_joined_ || !ab_tail_joined_ ||
      !owner_drained_recorded_ || main_generation == 0U ||
      ab_generation == 0U) {
    return fail(Sm87MacroFeedV4ExecutionError::kDrainIncomplete,
                "discard_requires_both_tail_joins_and_owner_drained", 0,
                Sm87MacroFeedV4ExecutionStream::kControl,
                Sm87MacroFeedV4ExecutionEvent::kOwnerDrained, active_panel_,
                active_panel_generation_);
  }
  if (!completion_receipt_matches_locked(
          panel_access, Sm87MacroFeedV4ExecutionEvent::kOwnerDrained,
          owner_drained) ||
      owner_drained.main_tail_generation_ != main_generation ||
      owner_drained.ab_tail_generation_ != ab_generation) {
    return fail(Sm87MacroFeedV4ExecutionError::kReceiptInvalid,
                "discard_requires_exact_dual_stream_drain_receipt", 0,
                Sm87MacroFeedV4ExecutionStream::kControl,
                Sm87MacroFeedV4ExecutionEvent::kOwnerDrained, active_panel_,
                active_panel_generation_);
  }

  state_ = Sm87MacroFeedV4ExecutionOwnerState::kRequestDiscarded;
  active_panel_ = kSm87MacroFeedV4PanelCount;
  active_panel_generation_ = 0U;
  request_epoch_ = 0U;
  return ok();
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsOwner::complete_request(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& final_panel_access,
    const Sm87MacroFeedV4PhysicalCompletionReceipt& final_publish) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4ExecutionStatus status =
      validate_operation_access(access, final_panel_access);
  if (!status) {
    return status;
  }
  if (completed_panels_ != kSm87MacroFeedV4PanelCount ||
      active_panel_ != kSm87MacroFeedV4PanelCount - 1U || draining_ ||
      !final_representation_ready_recorded_ ||
      !final_representation_joined_ || !canonical_copy_done_recorded_ ||
      !canonical_copy_joined_ || !final_publish_recorded_) {
    return fail(
        Sm87MacroFeedV4ExecutionError::kFinalPublicationIncomplete,
        "complete_requires_fixed_final_representation_copy_publish_chain", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kFinalPublish, active_panel_,
        active_panel_generation_);
  }
  if (!completion_receipt_matches_locked(
          final_panel_access, Sm87MacroFeedV4ExecutionEvent::kFinalPublish,
          final_publish)) {
    return fail(Sm87MacroFeedV4ExecutionError::kReceiptInvalid,
                "complete_requires_physical_final_publish_receipt", 0,
                Sm87MacroFeedV4ExecutionStream::kMain,
                Sm87MacroFeedV4ExecutionEvent::kFinalPublish, active_panel_,
                active_panel_generation_);
  }
  state_ = Sm87MacroFeedV4ExecutionOwnerState::kRequestCompleted;
  active_panel_ = kSm87MacroFeedV4PanelCount;
  active_panel_generation_ = 0U;
  request_epoch_ = 0U;
  return ok();
}

Sm87MacroFeedV4PoisonDrainResult
Sm87MacroFeedV4ExecutionEventsOwner::drain_poisoned_request(
    const Sm87MacroFeedV4ExecutionEventsAccess& access) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4PoisonDrainResult result;
  result.poison_cause = poison_cause_;
  if (!owner_access_matches(access)) {
    result.drain_status =
        fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
             "poison_drain_owner_issued_access");
    return result;
  }
  if (state_ != Sm87MacroFeedV4ExecutionOwnerState::kPoisoned ||
      poison_cause_.error == Sm87MacroFeedV4ExecutionError::kNone ||
      poisoned_terminal_quiescence_attested_) {
    result.drain_status =
        fail(Sm87MacroFeedV4ExecutionError::kInvalidOwnerState,
             "poison_drain_requires_retained_cuda_failure");
    return result;
  }

  // This is deliberately a terminal host wait, never a steady-state panel
  // boundary.  Synchronizing all three streams is the physical quiescence
  // proof when an event record/wait/observation path can no longer be trusted.
  bool synchronization_execution_clean = true;
  bool all_terminal_boundaries_observed = true;
  for (std::size_t index = 0U; index < streams_.size(); ++index) {
    const cudaError_t cuda_status = cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(streams_[index]));
    result.stream_cuda_status[index] = static_cast<int>(cuda_status);
    poison_drain_stream_cuda_status_[index] =
        static_cast<int>(cuda_status);
    if (cuda_status != cudaSuccess) {
      synchronization_execution_clean = false;
      if (result.drain_status.error ==
          Sm87MacroFeedV4ExecutionError::kNone) {
        result.drain_status = fail(
            Sm87MacroFeedV4ExecutionError::kCudaObservation,
            "poison_terminal_cudaStreamSynchronize_async_error",
            static_cast<int>(cuda_status),
            static_cast<Sm87MacroFeedV4ExecutionStream>(index));
      }
    }
    all_terminal_boundaries_observed =
        all_terminal_boundaries_observed &&
        terminal_stream_boundary_observed(cuda_status);
  }

  result.all_stream_synchronizations_attempted = true;
  poison_drain_all_stream_synchronizations_attempted_ = true;
  // cudaStreamSynchronize returning an asynchronous execution error still
  // establishes the terminal observation boundary for that stream.  It does
  // not turn the request into success: the per-stream statuses and the first
  // drain error are retained and the request is unconditionally discarded.
  if (!all_terminal_boundaries_observed) {
    if (result.drain_status.error ==
        Sm87MacroFeedV4ExecutionError::kNone) {
      result.drain_status = fail(
          Sm87MacroFeedV4ExecutionError::kCudaObservation,
          "poison_terminal_stream_boundary_unverified");
    }
    return result;
  }
  poisoned_terminal_quiescence_attested_ = true;
  // A terminal stream boundary makes request-owned storage safe to release;
  // it does not make a CUDA owner/context healthy again.  Keep the owner in
  // its non-reusable poisoned state so begin_request() can never clear the
  // retained asynchronous failure and admit another request.
  state_ = Sm87MacroFeedV4ExecutionOwnerState::kPoisoned;
  active_panel_ = kSm87MacroFeedV4PanelCount;
  active_panel_generation_ = 0U;
  request_epoch_ = 0U;
  if (synchronization_execution_clean) {
    result.drain_status = ok();
  }
  result.physical_quiescence_attested = true;
  result.discard_required = true;
  return result;
}

Sm87MacroFeedV4ExecutionEventsSnapshot
Sm87MacroFeedV4ExecutionEventsOwner::snapshot() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4ExecutionEventsSnapshot snapshot;
  snapshot.state = state_;
  snapshot.owner_identity = owner_identity_;
  snapshot.device_ordinal = device_ordinal_;
  snapshot.request_owner_identity = request_owner_identity_;
  snapshot.request_allocation_identity = request_allocation_identity_;
  snapshot.request_epoch = request_epoch_;
  snapshot.completed_panels = completed_panels_;
  snapshot.active_panel = active_panel_;
  snapshot.active_panel_generation = active_panel_generation_;
  snapshot.bf16_ab_cycles_completed = bf16_ab_cycles_completed_;
  for (std::size_t index = 0U; index < event_state_.size(); ++index) {
    snapshot.event_generations[index] = event_state_[index].generation;
  }
  snapshot.enqueue_receipts_issued = enqueue_receipts_issued_;
  snapshot.physical_completion_receipts_issued =
      physical_completion_receipts_issued_;
  snapshot.streams_nonblocking = streams_nonblocking_;
  snapshot.panel_done_recorded = panel_done_recorded_;
  snapshot.main_tail_recorded = main_tail_recorded_;
  snapshot.ab_tail_recorded = ab_tail_recorded_;
  snapshot.main_tail_joined = main_tail_joined_;
  snapshot.ab_tail_joined = ab_tail_joined_;
  snapshot.owner_drained_recorded = owner_drained_recorded_;
  snapshot.final_representation_ready_recorded =
      final_representation_ready_recorded_;
  snapshot.final_representation_joined = final_representation_joined_;
  snapshot.canonical_copy_done_recorded = canonical_copy_done_recorded_;
  snapshot.canonical_copy_joined = canonical_copy_joined_;
  snapshot.final_publish_recorded = final_publish_recorded_;
  snapshot.poison_cause = poison_cause_;
  snapshot.poison_drain_stream_cuda_status =
      poison_drain_stream_cuda_status_;
  snapshot.poison_drain_all_stream_synchronizations_attempted =
      poison_drain_all_stream_synchronizations_attempted_;
  snapshot.poisoned_terminal_quiescence_attested =
      poisoned_terminal_quiescence_attested_;
  return snapshot;
}

void Sm87MacroFeedV4ExecutionEventsOwner::release_resources() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == Sm87MacroFeedV4ExecutionOwnerState::kDestroyed) {
    return;
  }
  for (void* const raw_stream : streams_) {
    if (raw_stream != nullptr) {
      (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(raw_stream));
    }
  }
  for (void*& raw_event : events_) {
    if (raw_event != nullptr) {
      (void)cudaEventDestroy(reinterpret_cast<cudaEvent_t>(raw_event));
      raw_event = nullptr;
    }
  }
  for (void*& raw_stream : streams_) {
    if (raw_stream != nullptr) {
      (void)cudaStreamDestroy(reinterpret_cast<cudaStream_t>(raw_stream));
      raw_stream = nullptr;
    }
  }
  access_.reset();
  state_ = Sm87MacroFeedV4ExecutionOwnerState::kDestroyed;
}

Sm87MacroFeedV4ExecutionEventsCreateResult
create_sm87_macrofeed_v4_execution_events_owner() noexcept {
  Sm87MacroFeedV4ExecutionEventsCreateResult result;
  result.owner.reset(
      new (std::nothrow) Sm87MacroFeedV4ExecutionEventsOwner());
  if (result.owner == nullptr) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kInvalidOwnerState,
                         "execution_events_owner_allocation");
    return result;
  }
  result.status = result.owner->initialize();
  if (!result.status) {
    result.owner.reset();
  }
  return result;
}

}  // namespace q3x::runtime::sm87_macrofeed_v4_execution_events_detail
