#include "sm87_macrofeed_v4_execution_events_internal.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace q3x::runtime::sm87_macrofeed_v4_execution_events_detail {
namespace {

std::atomic<std::uint64_t> g_next_owner_identity{1U};
std::atomic<std::uint64_t> g_next_seal_nonce{1U};
std::atomic<std::uint64_t> g_next_enqueue_identity{1U};
std::atomic<std::uint64_t> g_next_completion_identity{1U};
std::atomic<std::uint64_t> g_next_gdn_layer_transaction_identity{1U};
std::atomic<std::uint64_t> g_next_full_attention_layer_transaction_identity{
    1U};
std::atomic<std::uint64_t> g_next_poison_quiescence_identity{1U};

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

struct ByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr ByteRange byte_range(
    const std::uintptr_t begin, const std::uint64_t bytes) noexcept {
  if (begin == 0U || bytes == 0U ||
      bytes > std::numeric_limits<std::uintptr_t>::max()) {
    return {};
  }
  const auto width = static_cast<std::uintptr_t>(bytes);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - width) {
    return {};
  }
  return {begin, begin + width, true};
}

[[nodiscard]] constexpr ByteRange byte_range(
    const void* const begin, const std::uint64_t bytes) noexcept {
  return byte_range(reinterpret_cast<std::uintptr_t>(begin), bytes);
}

template <std::size_t kCount>
[[nodiscard]] constexpr bool pairwise_disjoint(
    const std::array<ByteRange, kCount>& ranges) noexcept {
  for (std::size_t first = 0U; first < kCount; ++first) {
    if (!ranges[first].valid) {
      return false;
    }
    for (std::size_t second = first + 1U; second < kCount; ++second) {
      if (!(ranges[first].end <= ranges[second].begin ||
            ranges[second].end <= ranges[first].begin)) {
        return false;
      }
    }
  }
  return true;
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
  bound_kernel_submissions_ = 0U;
  input_norm_submissions_ = 0U;
  bf16_ab_submissions_ = 0U;
  gdn_qkvz_c8000_submissions_ = 0U;
  gdn_qkvz_ab_ready_wait_transactions_ = 0U;
  gdn_continuation_c8000_submissions_ = 0U;
  gdn_history_d2d_copies_ = 0U;
  gdn_history_d2d_bytes_ = 0U;
  gdn_output_c8000_submissions_ = 0U;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  full_qkv_c8000_submissions_ = 0U;
  full_attention_preprocess_c8000_submissions_ = 0U;
  attention_c8000_submissions_ = 0U;
  full_attention_output_c8000_submissions_ = 0U;
#endif
  residual_post_norm_submissions_ = 0U;
  gate_up_c8000_submissions_ = 0U;
  down_c8000_submissions_ = 0U;
  complete_gdn_layers_submitted_ = 0U;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  accepted_gdn_grant_identities_.fill(0U);
  accepted_gdn_grant_count_ = 0U;
  last_gdn_accepted_prefix_ = {};
  complete_full_attention_layers_submitted_ = 0U;
  accepted_full_attention_grant_identities_.fill(0U);
  accepted_full_attention_grant_count_ = 0U;
  last_full_attention_accepted_prefix_ = {};
#endif
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
  poisoned_terminal_quiescence_identity_ = 0U;
  test_fail_next_bound_ab_wait_ = false;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  test_fail_gdn_after_accepted_operation_ =
      std::numeric_limits<std::size_t>::max();
  test_fail_full_after_accepted_prefix_ =
      std::numeric_limits<std::size_t>::max();
#endif
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
  if (cold_recurrent_initializations_ != 1U ||
      cold_recurrent_allocation_identity_ == 0U ||
      cold_recurrent_allocation_begin_ == 0U ||
      cold_recurrent_zero_bytes_ != kSm87MacroFeedV4RecurrentStorageBytes ||
      allocation_identity != cold_recurrent_allocation_identity_) {
    return fail(Sm87MacroFeedV4ExecutionError::kInvalidOwnerState,
                "request_recurrent_allocation_must_match_cold_zero_seal");
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
  return record_event_locked(access, panel_access, producer, event);
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::record_event_locked(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionStream producer,
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
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

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::submit_input_norm_and_record_ready(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const kernels::Sm87MacroFeedV4InputNormArguments& arguments,
    const kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&
        resources) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return submit_input_norm_and_record_ready_locked(
      access, panel_access, arguments, resources);
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::
    submit_input_norm_and_record_ready_locked(
        const Sm87MacroFeedV4ExecutionEventsAccess& access,
        const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
        const kernels::Sm87MacroFeedV4InputNormArguments& arguments,
        const kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&
            resources) noexcept {
  Sm87MacroFeedV4EventEnqueueResult result;
  result.status = validate_operation_access(access, panel_access);
  if (!result.status) {
    return result;
  }
  auto bound_arguments = arguments;
  bound_arguments.cuda_stream =
      streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kMain)];
  if (!kernels::sm87_macrofeed_v4_input_norm_arguments_valid(
          bound_arguments) ||
      !kernels::sm87_macrofeed_v4_norm_residual_resource_gate(resources) ||
      arguments.cuda_stream != nullptr ||
      resources.device_ordinal != device_ordinal_) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
        "sealed_input_norm_binding_required", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kNormReady, active_panel_,
        active_panel_generation_);
    return result;
  }
  result.status = validate_record_order(
      Sm87MacroFeedV4ExecutionStream::kMain,
      Sm87MacroFeedV4ExecutionEvent::kNormReady);
  if (!result.status) {
    return result;
  }
  const EventState& norm_state =
      event_state_[event_index(Sm87MacroFeedV4ExecutionEvent::kNormReady)];
  if (norm_state.recorded && !norm_state.dependency_observed &&
      !norm_state.physical_observed) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
        "norm_ready_generation_must_be_consumed_before_kernel_submit", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kNormReady, active_panel_,
        active_panel_generation_);
    return result;
  }

  const kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4LockedSubmitToken token(
          streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kMain)]);
  std::size_t submitted = 0U;
  const int cuda_status = kernels::sm87_macrofeed_v4_bound_launch_detail::
      enqueue_input_norm_prevalidated(token, arguments, resources,
                                      &submitted);
  if (cuda_status != static_cast<int>(cudaSuccess)) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kCudaSubmission,
                         "bound_input_norm_enqueue", cuda_status,
                         Sm87MacroFeedV4ExecutionStream::kMain,
                         Sm87MacroFeedV4ExecutionEvent::kNormReady,
                         active_panel_, active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  if (submitted != 1U) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kReceiptInvalid,
                         "bound_input_norm_exactly_one_launch_required", 0,
                         Sm87MacroFeedV4ExecutionStream::kMain,
                         Sm87MacroFeedV4ExecutionEvent::kNormReady,
                         active_panel_, active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  ++bound_kernel_submissions_;
  ++input_norm_submissions_;
  result = record_event_locked(
      access, panel_access, Sm87MacroFeedV4ExecutionStream::kMain,
      Sm87MacroFeedV4ExecutionEvent::kNormReady);
  if (!result && state_ != Sm87MacroFeedV4ExecutionOwnerState::kPoisoned) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kReceiptInvalid,
        "submitted_input_norm_requires_ready_event_receipt", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kNormReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
  }
  return result;
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::submit_bf16_ab_and_record_ready(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const kernels::Sm87MacroFeedV4Bf16AbArguments& arguments,
    const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
        resources) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return submit_bf16_ab_and_record_ready_locked(
      access, panel_access, arguments, resources);
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::
    submit_bf16_ab_and_record_ready_locked(
        const Sm87MacroFeedV4ExecutionEventsAccess& access,
        const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
        const kernels::Sm87MacroFeedV4Bf16AbArguments& arguments,
        const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
            resources) noexcept {
  Sm87MacroFeedV4EventEnqueueResult result;
  result.status = validate_operation_access(access, panel_access);
  if (!result.status) {
    return result;
  }
  auto bound_arguments = arguments;
  bound_arguments.cuda_stream =
      streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kAbAux)];
  if (!kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(
          bound_arguments) ||
      !kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(resources) ||
      arguments.cuda_stream != nullptr ||
      resources.device_ordinal != device_ordinal_) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
        "sealed_bf16_ab_binding_required", 0,
        Sm87MacroFeedV4ExecutionStream::kAbAux,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    return result;
  }
  result.status = validate_record_order(
      Sm87MacroFeedV4ExecutionStream::kAbAux,
      Sm87MacroFeedV4ExecutionEvent::kAbReady);
  if (!result.status) {
    return result;
  }
  const EventState& ab_state =
      event_state_[event_index(Sm87MacroFeedV4ExecutionEvent::kAbReady)];
  if (ab_state.recorded && !ab_state.dependency_observed &&
      !ab_state.physical_observed) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
        "ab_ready_generation_must_be_consumed_before_kernel_submit", 0,
        Sm87MacroFeedV4ExecutionStream::kAbAux,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    return result;
  }

  const kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4LockedSubmitToken token(
          streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kAbAux)]);
  std::size_t submitted = 0U;
  const int cuda_status = kernels::sm87_macrofeed_v4_bound_launch_detail::
      enqueue_bf16_ab_prevalidated(token, arguments, resources, &submitted);
  if (cuda_status != static_cast<int>(cudaSuccess)) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kCudaSubmission,
                         "bound_bf16_ab_enqueue", cuda_status,
                         Sm87MacroFeedV4ExecutionStream::kAbAux,
                         Sm87MacroFeedV4ExecutionEvent::kAbReady,
                         active_panel_, active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  if (submitted != 1U) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kReceiptInvalid,
                         "bound_bf16_ab_exactly_one_launch_required", 0,
                         Sm87MacroFeedV4ExecutionStream::kAbAux,
                         Sm87MacroFeedV4ExecutionEvent::kAbReady,
                         active_panel_, active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  ++bound_kernel_submissions_;
  ++bf16_ab_submissions_;
  result = record_event_locked(
      access, panel_access, Sm87MacroFeedV4ExecutionStream::kAbAux,
      Sm87MacroFeedV4ExecutionEvent::kAbReady);
  if (!result && state_ != Sm87MacroFeedV4ExecutionOwnerState::kPoisoned) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kReceiptInvalid,
        "submitted_bf16_ab_requires_ready_event_receipt", 0,
        Sm87MacroFeedV4ExecutionStream::kAbAux,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
  }
  return result;
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::
    submit_gdn_qkvz_c8000_then_wait_ab_ready(
        const Sm87MacroFeedV4ExecutionEventsAccess& access,
        const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
        const kernels::sm87_macrofeed_v4_bound_launch_detail::
            Sm87MacroFeedV4GdnQkvzC8000Arguments& arguments,
        const kernels::Sm87MacroFeedV4Fp8CudaResources& resources) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return submit_gdn_qkvz_c8000_then_wait_ab_ready_locked(
      access, panel_access, arguments, resources);
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::
    submit_gdn_qkvz_c8000_then_wait_ab_ready_locked(
        const Sm87MacroFeedV4ExecutionEventsAccess& access,
        const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
        const kernels::sm87_macrofeed_v4_bound_launch_detail::
            Sm87MacroFeedV4GdnQkvzC8000Arguments& arguments,
        const kernels::Sm87MacroFeedV4Fp8CudaResources& resources) noexcept {
  Sm87MacroFeedV4EventEnqueueResult result;
  result.status = validate_operation_access(access, panel_access);
  if (!result.status) {
    // Once a request is active, a stale/forged panel capability is a
    // transaction failure, not a harmless pre-admission rejection: earlier
    // Main/AbAux work may already be in flight.  Preserve the first cause and
    // force the caller through the terminal three-stream poison drain.  A
    // foreign owner capability is the only case known to be outside this
    // owner's submission boundary.
    if (result.status.error !=
            Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess &&
        owner_access_matches(access) &&
        state_ == Sm87MacroFeedV4ExecutionOwnerState::kRequestActive) {
      record_poison_cause(result.status);
    }
    return result;
  }
  constexpr auto kRole = kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
  constexpr auto kLayout =
      kernels::Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1;
  const kernels::Sm87MacroFeedV4Fp8Arguments fixed{
      kRole,
      arguments.hidden_input,
      kernels::kSm87MacroFeedV4Fp8HiddenRowStride,
      arguments.asset,
      kernels::kSm87MacroFeedV4Fp8Tokens,
      arguments.phase_scratch,
      kernels::kSm87MacroFeedV4Fp8ScratchRowStride,
      nullptr,
      0U,
      nullptr,
      0U,
      streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kMain)],
      kLayout};
  if (!kernels::sm87_macrofeed_v4_fp8_arguments_valid(fixed) ||
      !resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(resources) ||
      resources.role != kRole || resources.input_layout != kLayout ||
      resources.identity !=
          kernels::sm87_macrofeed_v4_fp8_identity(kRole, kLayout) ||
      resources.device_ordinal != device_ordinal_) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
        "sealed_gdn_qkvz_c8000_binding_required", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }

  EventState& ab_state =
      event_state_[event_index(Sm87MacroFeedV4ExecutionEvent::kAbReady)];
  if (!ab_state.recorded || ab_state.request_epoch != request_epoch_ ||
      ab_state.panel != active_panel_ ||
      ab_state.panel_generation != active_panel_generation_ ||
      ab_state.dependency_observed || ab_state.physical_observed) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kEventNotRecorded,
        "gdn_qkvz_requires_current_unobserved_ab_ready_generation", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  result.status = validate_wait_order(
      Sm87MacroFeedV4ExecutionStream::kMain,
      Sm87MacroFeedV4ExecutionEvent::kAbReady);
  if (!result.status) {
    record_poison_cause(result.status);
    return result;
  }

  const kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4LockedSubmitToken token(
          streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kMain)]);
  std::size_t submitted = 0U;
  const int enqueue_status =
      kernels::sm87_macrofeed_v4_bound_launch_detail::
          enqueue_gdn_qkvz_c8000_prevalidated(token, arguments, resources,
                                               &submitted);
  if (enqueue_status != static_cast<int>(cudaSuccess)) {
    result.status = fail(Sm87MacroFeedV4ExecutionError::kCudaSubmission,
                         "bound_gdn_qkvz_c8000_enqueue", enqueue_status,
                         Sm87MacroFeedV4ExecutionStream::kMain,
                         Sm87MacroFeedV4ExecutionEvent::kAbReady,
                         active_panel_, active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  if (submitted != 1U) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kReceiptInvalid,
        "bound_gdn_qkvz_c8000_exactly_one_launch_required", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  ++bound_kernel_submissions_;
  ++gdn_qkvz_c8000_submissions_;

  void* wait_event_handle =
      events_[event_index(Sm87MacroFeedV4ExecutionEvent::kAbReady)];
  if (test_fail_next_bound_ab_wait_) {
    test_fail_next_bound_ab_wait_ = false;
    wait_event_handle = nullptr;
  }
  const cudaError_t wait_status = cudaStreamWaitEvent(
      reinterpret_cast<cudaStream_t>(
          streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kMain)]),
      reinterpret_cast<cudaEvent_t>(wait_event_handle), 0U);
  if (wait_status != cudaSuccess) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kCudaSubmission,
        "bound_gdn_qkvz_c8000_cudaStreamWaitEvent",
        static_cast<int>(wait_status), Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }

  ab_state.dependency_observed = true;
  advance_wait_order(Sm87MacroFeedV4ExecutionEvent::kAbReady);
  ++gdn_qkvz_ab_ready_wait_transactions_;
  result.receipt = mint_enqueue_receipt(
      Sm87MacroFeedV4EnqueueOperation::kWait,
      Sm87MacroFeedV4ExecutionStream::kMain,
      Sm87MacroFeedV4ExecutionEvent::kAbReady, ab_state.generation);
  result.status = ok();
  return result;
}

Sm87MacroFeedV4CompleteGdnLayerEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::
    submit_complete_gdn_layer_c8000_prevalidated(
        const Sm87MacroFeedV4ExecutionEventsAccess& access,
        const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
        const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
        const Sm87MacroFeedV4CompleteGdnLayerC8000Submission& submission)
        noexcept {
  namespace bound =
      kernels::sm87_macrofeed_v4_bound_launch_detail;
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4CompleteGdnLayerEnqueueResult result;
  result.status = validate_operation_access(access, panel_access);
  if (!result.status) {
    return result;
  }

  Sm87MacroFeedV4GdnAcceptedPrefixLedger ledger;
  ledger.transaction_identity =
      next_nonzero(&g_next_gdn_layer_transaction_identity);
  ledger.owner_identity = owner_identity_;
  ledger.request_epoch = request_epoch_;
  ledger.panel = active_panel_;
  ledger.panel_generation = active_panel_generation_;
  ledger.grant_identity = gdn_grant.grant_identity();
  ledger.grant_state_epoch = gdn_grant.state_epoch();
  ledger.recurrent_allocation_identity = gdn_grant.allocation_identity();
  ledger.gdn_ordinal = gdn_grant.state_layer_ordinal();
  ledger.model_layer = gdn_grant.model_layer();
  ledger.active_bank_index = gdn_grant.active_bank_index();
  ledger.candidate_bank_index = gdn_grant.candidate_bank_index();
  ledger.active_conv_allocation_offset =
      gdn_grant.active_conv_allocation_offset();
  ledger.candidate_conv_allocation_offset =
      gdn_grant.candidate_conv_allocation_offset();
  ledger.conv_bytes = gdn_grant.conv_bytes();
  ledger.active_gdn_state_allocation_offset =
      gdn_grant.active_gdn_state_allocation_offset();
  ledger.candidate_gdn_state_allocation_offset =
      gdn_grant.candidate_gdn_state_allocation_offset();
  ledger.gdn_state_bytes = gdn_grant.gdn_state_bytes();
  last_gdn_accepted_prefix_ = ledger;

  const void* const main_stream =
      streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kMain)];
  const void* const ab_stream =
      streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kAbAux)];
  auto checked_input_norm = submission.input_norm;
  checked_input_norm.cuda_stream = const_cast<void*>(main_stream);
  auto checked_bf16_ab = submission.bf16_ab;
  checked_bf16_ab.cuda_stream = const_cast<void*>(ab_stream);
  const kernels::Sm87MacroFeedV4Fp8Arguments checked_gdn_qkvz{
      kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
      submission.gdn_qkvz.hidden_input,
      kernels::kSm87MacroFeedV4Fp8HiddenRowStride,
      submission.gdn_qkvz.asset,
      kernels::kSm87MacroFeedV4Fp8Tokens,
      submission.gdn_qkvz.phase_scratch,
      kernels::kSm87MacroFeedV4Fp8ScratchRowStride,
      nullptr,
      0U,
      nullptr,
      0U,
      const_cast<void*>(main_stream),
      kernels::Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1};
  const kernels::Sm87MacroFeedV4GdnC8000Arguments checked_continuation{
      submission.gdn_continuation.phase_scratch,
      kernels::kSm87MacroFeedV4GdnC8000Tokens,
      kernels::kSm87MacroFeedV4GdnScratchRowStride,
      submission.gdn_continuation.conv_weight,
      submission.gdn_continuation.a_log,
      submission.gdn_continuation.dt_bias,
      submission.gdn_continuation.norm_weight,
      submission.gdn_continuation.active_conv_history,
      submission.gdn_continuation.candidate_conv_history,
      submission.gdn_continuation.active_recurrent_state,
      submission.gdn_continuation.candidate_recurrent_state,
      submission.gdn_continuation.cancellation_signal,
      submission.gdn_continuation.l2_epsilon_fp32_bits,
      submission.gdn_continuation.norm_epsilon_fp32_bits,
      const_cast<void*>(main_stream)};
  const kernels::Sm87MacroFeedV4Fp8Arguments checked_gdn_output{
      kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
      submission.gdn_output.phase_scratch,
      kernels::kSm87MacroFeedV4Fp8ScratchRowStride,
      submission.gdn_output.asset,
      kernels::kSm87MacroFeedV4Fp8Tokens,
      submission.gdn_output.branch_output,
      kernels::kSm87MacroFeedV4Fp8HiddenRowStride,
      nullptr,
      0U,
      nullptr,
      0U,
      const_cast<void*>(main_stream),
      kernels::Sm87MacroFeedV4Fp8InputLayout::kGdnContiguousVScratchV1};
  const kernels::Sm87MacroFeedV4ResidualPostNormArguments
      checked_residual_post_norm{
          submission.residual_post_norm.left_residual_then_normalized,
          submission.residual_post_norm.right_branch_then_residual,
          submission.residual_post_norm.centered_weight,
          kernels::kSm87MacroFeedV4NormResidualTokens,
          kernels::kSm87MacroFeedV4NormResidualHidden,
          kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits,
          const_cast<void*>(main_stream)};
  const kernels::Sm87MacroFeedV4NvFp4GateUpArguments checked_gate_up{
      submission.gate_up.normalized_input,
      submission.gate_up.payload,
      submission.gate_up.payload_bytes,
      submission.gate_up.gate_tensor_scale,
      submission.gate_up.up_tensor_scale,
      kernels::kSm87MacroFeedV4NvFp4GateUpTokens,
      submission.gate_up.intermediate_output,
      const_cast<void*>(main_stream),
      submission.gate_up.canonical_v3_payload_receipt};
  const kernels::Sm87MacroFeedV4NvFp4DownArguments checked_down{
      submission.down.intermediate_input,
      submission.down.payload,
      submission.down.payload_bytes,
      submission.down.tensor_scale,
      kernels::kSm87MacroFeedV4NvFp4DownTokens,
      submission.down.residual_output,
      const_cast<void*>(main_stream),
      submission.down.payload_receipt};

  constexpr auto kGdnQkvzIdentity =
      kernels::Sm87MacroFeedV4Fp8Identity::
          kGdnQkvZM64N128K64OrdinaryGridV1;
  constexpr auto kGdnOutputIdentity =
      kernels::Sm87MacroFeedV4Fp8Identity::
          kGdnAttentionOutputM64N128K64OrdinaryGridV1;
  const bool normal_authority =
      submission.authority_domain ==
          Sm87MacroFeedV4GdnSubmissionAuthorityDomain::kNormalSealedCatalog &&
      submission.gdn_catalog_identity != 0U &&
      submission.gdn_binding_identity != 0U &&
      submission.mlp_catalog_identity != 0U &&
      submission.mlp_binding_identity != 0U &&
      submission.resource_bundle_identity != 0U &&
      submission.synthetic_source_identity == 0U;
  const bool synthetic_authority =
      submission.authority_domain ==
          Sm87MacroFeedV4GdnSubmissionAuthorityDomain::kSyntheticT1 &&
      submission.gdn_catalog_identity == 0U &&
      submission.gdn_binding_identity == 0U &&
      submission.mlp_catalog_identity == 0U &&
      submission.mlp_binding_identity == 0U &&
      submission.resource_bundle_identity == 0U &&
      submission.synthetic_source_identity != 0U;
  const bool exact_submission_identities =
      submission.execution_package_identity != 0U &&
      submission.bf16_ab_catalog_identity != 0U &&
      submission.bf16_ab_pair_identity != 0U &&
      submission.layer_norm_catalog_identity != 0U &&
      submission.layer_norm_pair_identity != 0U &&
      submission.input_norm_binding_identity != 0U &&
      submission.post_norm_binding_identity != 0U &&
      (normal_authority || synthetic_authority);

  const bool exact_grant =
      gdn_grant.grant_identity() != 0U &&
      gdn_grant.owner_identity() == owner_identity_ &&
      gdn_grant.allocation_identity() == request_allocation_identity_ &&
      gdn_grant.request_epoch() == request_epoch_ &&
      gdn_grant.panel() == active_panel_ &&
      gdn_grant.state_layer_ordinal() == submission.gdn_ordinal &&
      gdn_grant.model_layer() == submission.model_layer &&
      submission.gdn_ordinal < kSm87MacroFeedV4StateLayerCount &&
      submission.model_layer ==
          submission.gdn_ordinal + submission.gdn_ordinal / 3U &&
      gdn_grant.active_bank_index() < 2U &&
      gdn_grant.candidate_bank_index() < 2U &&
      gdn_grant.active_bank_index() != gdn_grant.candidate_bank_index() &&
      gdn_grant.conv_bytes() ==
          kernels::kSm87MacroFeedV4GdnConvHistoryBytes &&
      gdn_grant.gdn_state_bytes() ==
          kernels::kSm87MacroFeedV4GdnStateBytes;

  const auto bank_origin = [](const std::size_t bank) noexcept {
    return static_cast<std::uint64_t>(bank) *
           kSm87MacroFeedV4RecurrentEpochBytes;
  };
  const std::uint64_t ordinal_conv_offset =
      static_cast<std::uint64_t>(submission.gdn_ordinal) *
      kSm87MacroFeedV4ConvLayerBytes;
  const std::uint64_t ordinal_state_offset =
      kSm87MacroFeedV4ConvEpochBytes +
      static_cast<std::uint64_t>(submission.gdn_ordinal) *
          kSm87MacroFeedV4GdnStateLayerBytes;
  const bool exact_grant_offsets =
      exact_grant &&
      gdn_grant.active_conv_allocation_offset() ==
          bank_origin(gdn_grant.active_bank_index()) + ordinal_conv_offset &&
      gdn_grant.candidate_conv_allocation_offset() ==
          bank_origin(gdn_grant.candidate_bank_index()) +
              ordinal_conv_offset &&
      gdn_grant.active_gdn_state_allocation_offset() ==
          bank_origin(gdn_grant.active_bank_index()) + ordinal_state_offset &&
      gdn_grant.candidate_gdn_state_allocation_offset() ==
          bank_origin(gdn_grant.candidate_bank_index()) +
              ordinal_state_offset;

  const auto recurrent_pointer =
      [&](const std::uint64_t offset) noexcept -> const void* {
    if (cold_recurrent_allocation_begin_ == 0U ||
        offset > kSm87MacroFeedV4RecurrentStorageBytes ||
        cold_recurrent_allocation_begin_ >
            std::numeric_limits<std::uintptr_t>::max() - offset) {
      return nullptr;
    }
    return reinterpret_cast<const void*>(
        cold_recurrent_allocation_begin_ +
        static_cast<std::uintptr_t>(offset));
  };
  const bool exact_recurrent_pointers =
      exact_grant_offsets &&
      submission.gdn_continuation.active_conv_history ==
          recurrent_pointer(gdn_grant.active_conv_allocation_offset()) &&
      submission.gdn_continuation.candidate_conv_history ==
          recurrent_pointer(gdn_grant.candidate_conv_allocation_offset()) &&
      submission.gdn_continuation.active_recurrent_state ==
          recurrent_pointer(
              gdn_grant.active_gdn_state_allocation_offset()) &&
      submission.gdn_continuation.candidate_recurrent_state ==
          recurrent_pointer(
              gdn_grant.candidate_gdn_state_allocation_offset());

  constexpr std::uint64_t kHiddenBytes =
      kernels::kSm87MacroFeedV4NormResidualHiddenBytes;
  constexpr std::uint64_t kScratchBytes =
      kernels::kSm87MacroFeedV4GdnScratchBytes;
  constexpr std::uint64_t kNormWeightBytes =
      kernels::kSm87MacroFeedV4NormResidualWeightBytes;
  const std::array<ByteRange, 15U> non_recurrent_ranges{{
      byte_range(submission.input_norm.input_hidden, kHiddenBytes),
      byte_range(submission.input_norm.output_hidden, kHiddenBytes),
      byte_range(submission.bf16_ab.scratch, kScratchBytes),
      byte_range(submission.input_norm.centered_weight, kNormWeightBytes),
      byte_range(submission.residual_post_norm.centered_weight,
                 kNormWeightBytes),
      byte_range(submission.bf16_ab.a_weights,
                 kernels::kSm87MacroFeedV4Bf16AbWeightBytes),
      byte_range(submission.bf16_ab.b_weights,
                 kernels::kSm87MacroFeedV4Bf16AbWeightBytes),
      byte_range(submission.gdn_qkvz.asset.payload.begin,
                 submission.gdn_qkvz.asset.payload.bytes),
      byte_range(submission.gdn_output.asset.payload.begin,
                 submission.gdn_output.asset.payload.bytes),
      byte_range(submission.gdn_continuation.conv_weight,
                 kernels::kSm87MacroFeedV4GdnConvWeightBytes),
      byte_range(submission.gdn_continuation.a_log,
                 kernels::kSm87MacroFeedV4GdnHeadVectorBytes),
      byte_range(submission.gdn_continuation.dt_bias,
                 kernels::kSm87MacroFeedV4GdnHeadVectorBytes),
      byte_range(submission.gdn_continuation.norm_weight,
                 kernels::kSm87MacroFeedV4GdnNormWeightBytes),
      byte_range(submission.gate_up.payload,
                 submission.gate_up.payload_bytes),
      byte_range(submission.down.payload, submission.down.payload_bytes),
  }};
  const ByteRange recurrent_arena =
      byte_range(cold_recurrent_allocation_begin_,
                 kSm87MacroFeedV4RecurrentStorageBytes);
  bool recurrent_disjoint = recurrent_arena.valid;
  for (const ByteRange& range : non_recurrent_ranges) {
    recurrent_disjoint =
        recurrent_disjoint && range.valid &&
        (range.end <= recurrent_arena.begin ||
         recurrent_arena.end <= range.begin);
  }
  const bool exact_alias_graph =
      submission.input_norm.input_hidden ==
          submission.residual_post_norm.left_residual_then_normalized &&
      submission.input_norm.output_hidden == submission.bf16_ab.input &&
      submission.input_norm.output_hidden ==
          submission.gdn_qkvz.hidden_input &&
      submission.input_norm.output_hidden ==
          submission.gdn_output.branch_output &&
      submission.input_norm.output_hidden ==
          submission.residual_post_norm.right_branch_then_residual &&
      submission.bf16_ab.scratch == submission.gdn_qkvz.phase_scratch &&
      submission.bf16_ab.scratch ==
          submission.gdn_continuation.phase_scratch &&
      submission.bf16_ab.scratch == submission.gdn_output.phase_scratch &&
      submission.bf16_ab.scratch ==
          submission.gate_up.intermediate_output &&
      submission.bf16_ab.scratch == submission.down.intermediate_input &&
      submission.gate_up.normalized_input ==
          submission.residual_post_norm.left_residual_then_normalized &&
      submission.down.residual_output ==
          submission.residual_post_norm.right_branch_then_residual &&
      pairwise_disjoint(non_recurrent_ranges) && recurrent_disjoint;

  if (cold_recurrent_initializations_ != 1U ||
      cold_recurrent_allocation_identity_ != request_allocation_identity_ ||
      cold_recurrent_allocation_begin_ == 0U ||
      cold_recurrent_zero_bytes_ != kSm87MacroFeedV4RecurrentStorageBytes ||
      !exact_submission_identities || !exact_grant_offsets ||
      !exact_recurrent_pointers || !exact_alias_graph || !ledger.valid_prefix() ||
      submission.input_norm.cuda_stream != nullptr ||
      submission.bf16_ab.cuda_stream != nullptr ||
      submission.gdn_continuation.cancellation_signal != nullptr ||
      !kernels::sm87_macrofeed_v4_input_norm_arguments_valid(
          checked_input_norm) ||
      !kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(checked_bf16_ab) ||
      !kernels::sm87_macrofeed_v4_fp8_arguments_valid(checked_gdn_qkvz) ||
      !kernels::sm87_macrofeed_v4_gdn_c8000_arguments_valid(
          checked_continuation) ||
      !kernels::sm87_macrofeed_v4_fp8_arguments_valid(checked_gdn_output) ||
      !kernels::sm87_macrofeed_v4_residual_post_norm_arguments_valid(
          checked_residual_post_norm) ||
      !kernels::sm87_macrofeed_v4_nvfp4_gate_up_arguments_valid(
          checked_gate_up) ||
      !kernels::sm87_macrofeed_v4_nvfp4_down_arguments_valid(checked_down) ||
      !kernels::sm87_macrofeed_v4_norm_residual_resource_gate(
          submission.norm_resources) ||
      !kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
          submission.bf16_ab_resources) ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(
          submission.gdn_qkvz_resources) ||
      submission.gdn_qkvz_resources.identity != kGdnQkvzIdentity ||
      submission.gdn_qkvz_resources.role !=
          kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ ||
      submission.gdn_qkvz_resources.input_layout !=
          kernels::Sm87MacroFeedV4Fp8InputLayout::
              kHiddenContiguousH5120V1 ||
      !kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
          submission.gdn_continuation_resources) ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(
          submission.gdn_output_resources) ||
      submission.gdn_output_resources.identity != kGdnOutputIdentity ||
      submission.gdn_output_resources.role !=
          kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput ||
      submission.gdn_output_resources.input_layout !=
          kernels::Sm87MacroFeedV4Fp8InputLayout::
              kGdnContiguousVScratchV1 ||
      !kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(
          submission.gate_up_resources) ||
      !kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(
          submission.down_resources) ||
      submission.norm_resources.device_ordinal != device_ordinal_ ||
      submission.bf16_ab_resources.device_ordinal != device_ordinal_ ||
      submission.gdn_qkvz_resources.device_ordinal != device_ordinal_ ||
      submission.gdn_continuation_resources.device_ordinal !=
          device_ordinal_ ||
      submission.gdn_output_resources.device_ordinal != device_ordinal_ ||
      submission.gate_up_resources.device_ordinal != device_ordinal_ ||
      submission.down_resources.device_ordinal != device_ordinal_) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
        "complete_gdn_layer_all_bindings_prevalidated_before_enqueue", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kNormReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }

  static_assert(std::tuple_size<
                    decltype(accepted_gdn_grant_identities_)>::value ==
                kSm87MacroFeedV4PanelCount *
                    kSm87MacroFeedV4StateLayerCount);
  const std::size_t grant_slot =
      active_panel_ * kSm87MacroFeedV4StateLayerCount +
      submission.gdn_ordinal;
  if (accepted_gdn_grant_identities_[grant_slot] != 0U) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
        "complete_gdn_layer_grant_at_most_once", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kCount, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  accepted_gdn_grant_identities_[grant_slot] =
      gdn_grant.grant_identity();
  ++accepted_gdn_grant_count_;

  const auto inject_failure_if_armed = [&]() noexcept {
    if (test_fail_gdn_after_accepted_operation_ !=
        ledger.accepted_operations()) {
      return false;
    }
    test_fail_gdn_after_accepted_operation_ =
        std::numeric_limits<std::size_t>::max();
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kCudaSubmission,
        "test_gdn_failure_after_accepted_operation", 1,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kCount, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return true;
  };
  if (inject_failure_if_armed()) {
    return result;
  }

  std::size_t before_submissions = input_norm_submissions_;
  auto enqueue = submit_input_norm_and_record_ready_locked(
      access, panel_access, submission.input_norm,
      submission.norm_resources);
  ledger.input_norm_launches =
      input_norm_submissions_ - before_submissions;
  ledger.accepted_kernel_launches += ledger.input_norm_launches;
  last_gdn_accepted_prefix_ = ledger;
  if (!enqueue) {
    result.status = enqueue.status;
    if (state_ != Sm87MacroFeedV4ExecutionOwnerState::kPoisoned) {
      record_poison_cause(result.status);
    }
    return result;
  }
  if (inject_failure_if_armed()) {
    return result;
  }

  enqueue = wait_event_locked(
      access, panel_access, Sm87MacroFeedV4ExecutionStream::kAbAux,
      Sm87MacroFeedV4ExecutionEvent::kNormReady);
  if (!enqueue) {
    result.status = enqueue.status;
    if (state_ != Sm87MacroFeedV4ExecutionOwnerState::kPoisoned) {
      record_poison_cause(result.status);
    }
    return result;
  }

  before_submissions = bf16_ab_submissions_;
  enqueue = submit_bf16_ab_and_record_ready_locked(
      access, panel_access, submission.bf16_ab,
      submission.bf16_ab_resources);
  ledger.bf16_ab_launches =
      bf16_ab_submissions_ - before_submissions;
  ledger.accepted_kernel_launches += ledger.bf16_ab_launches;
  last_gdn_accepted_prefix_ = ledger;
  if (!enqueue) {
    result.status = enqueue.status;
    if (state_ != Sm87MacroFeedV4ExecutionOwnerState::kPoisoned) {
      record_poison_cause(result.status);
    }
    return result;
  }
  if (inject_failure_if_armed()) {
    return result;
  }

  before_submissions = gdn_qkvz_c8000_submissions_;
  enqueue = submit_gdn_qkvz_c8000_then_wait_ab_ready_locked(
      access, panel_access, submission.gdn_qkvz,
      submission.gdn_qkvz_resources);
  ledger.gdn_qkvz_launches =
      gdn_qkvz_c8000_submissions_ - before_submissions;
  ledger.accepted_kernel_launches += ledger.gdn_qkvz_launches;
  last_gdn_accepted_prefix_ = ledger;
  if (!enqueue) {
    result.status = enqueue.status;
    if (state_ != Sm87MacroFeedV4ExecutionOwnerState::kPoisoned) {
      record_poison_cause(result.status);
    }
    return result;
  }
  if (inject_failure_if_armed()) {
    return result;
  }

  const bound::Sm87MacroFeedV4LockedSubmitToken main_token(
      streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kMain)]);
  bound::Sm87MacroFeedV4GdnContinuationSubmitLedger continuation_ledger{};
  int cuda_status = bound::enqueue_gdn_continuation_c8000_prevalidated(
      main_token, submission.gdn_continuation,
      submission.gdn_continuation_resources, &continuation_ledger);
  bound_kernel_submissions_ +=
      continuation_ledger.accepted_kernel_launches;
  gdn_continuation_c8000_submissions_ +=
      continuation_ledger.accepted_kernel_launches;
  gdn_history_d2d_copies_ += continuation_ledger.asynchronous_d2d_copies;
  gdn_history_d2d_bytes_ += continuation_ledger.conv_history_copy_bytes;
  ledger.gdn_continuation_launches =
      continuation_ledger.accepted_kernel_launches;
  ledger.accepted_kernel_launches +=
      continuation_ledger.accepted_kernel_launches;
  ledger.asynchronous_d2d_copies =
      continuation_ledger.asynchronous_d2d_copies;
  ledger.conv_history_copy_bytes =
      continuation_ledger.conv_history_copy_bytes;
  last_gdn_accepted_prefix_ = ledger;
  if (cuda_status != static_cast<int>(cudaSuccess) ||
      continuation_ledger.accepted_kernel_launches != 2U ||
      continuation_ledger.asynchronous_d2d_copies != 1U ||
      continuation_ledger.conv_history_copy_bytes !=
          kernels::kSm87MacroFeedV4GdnConvHistoryBytes) {
    result.status = fail(
        cuda_status == static_cast<int>(cudaSuccess)
            ? Sm87MacroFeedV4ExecutionError::kReceiptInvalid
            : Sm87MacroFeedV4ExecutionError::kCudaSubmission,
        "complete_gdn_layer_continuation_exact_two_launches", cuda_status,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  if (inject_failure_if_armed()) {
    return result;
  }

  std::size_t submitted = 0U;
  cuda_status = bound::enqueue_gdn_o_c8000_prevalidated(
      main_token, submission.gdn_output, submission.gdn_output_resources,
      &submitted);
  ledger.gdn_output_launches += submitted;
  ledger.accepted_kernel_launches += submitted;
  bound_kernel_submissions_ += submitted;
  gdn_output_c8000_submissions_ += submitted;
  last_gdn_accepted_prefix_ = ledger;
  if (cuda_status != static_cast<int>(cudaSuccess) || submitted != 1U) {
    result.status = fail(
        cuda_status == static_cast<int>(cudaSuccess)
            ? Sm87MacroFeedV4ExecutionError::kReceiptInvalid
            : Sm87MacroFeedV4ExecutionError::kCudaSubmission,
        "complete_gdn_layer_output_exact_one_launch", cuda_status,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  if (inject_failure_if_armed()) {
    return result;
  }

  submitted = 0U;
  cuda_status = bound::enqueue_residual_post_norm_prevalidated(
      main_token, submission.residual_post_norm,
      submission.norm_resources, &submitted);
  ledger.residual_post_norm_launches += submitted;
  ledger.accepted_kernel_launches += submitted;
  bound_kernel_submissions_ += submitted;
  residual_post_norm_submissions_ += submitted;
  last_gdn_accepted_prefix_ = ledger;
  if (cuda_status != static_cast<int>(cudaSuccess) || submitted != 1U) {
    result.status = fail(
        cuda_status == static_cast<int>(cudaSuccess)
            ? Sm87MacroFeedV4ExecutionError::kReceiptInvalid
            : Sm87MacroFeedV4ExecutionError::kCudaSubmission,
        "complete_gdn_layer_residual_post_norm_exact_one_launch",
        cuda_status, Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  if (inject_failure_if_armed()) {
    return result;
  }

  submitted = 0U;
  cuda_status = bound::enqueue_gate_up_c8000_prevalidated(
      main_token, submission.gate_up, submission.gate_up_resources,
      &submitted);
  ledger.gate_up_launches += submitted;
  ledger.accepted_kernel_launches += submitted;
  bound_kernel_submissions_ += submitted;
  gate_up_c8000_submissions_ += submitted;
  last_gdn_accepted_prefix_ = ledger;
  if (cuda_status != static_cast<int>(cudaSuccess) || submitted != 1U) {
    result.status = fail(
        cuda_status == static_cast<int>(cudaSuccess)
            ? Sm87MacroFeedV4ExecutionError::kReceiptInvalid
            : Sm87MacroFeedV4ExecutionError::kCudaSubmission,
        "complete_gdn_layer_gate_up_exact_one_launch", cuda_status,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  if (inject_failure_if_armed()) {
    return result;
  }

  submitted = 0U;
  cuda_status = bound::enqueue_down_c8000_prevalidated(
      main_token, submission.down, submission.down_resources, &submitted);
  ledger.down_launches += submitted;
  ledger.accepted_kernel_launches += submitted;
  bound_kernel_submissions_ += submitted;
  down_c8000_submissions_ += submitted;
  last_gdn_accepted_prefix_ = ledger;
  if (cuda_status != static_cast<int>(cudaSuccess) || submitted != 1U) {
    result.status = fail(
        cuda_status == static_cast<int>(cudaSuccess)
            ? Sm87MacroFeedV4ExecutionError::kReceiptInvalid
            : Sm87MacroFeedV4ExecutionError::kCudaSubmission,
        "complete_gdn_layer_down_exact_one_launch", cuda_status,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kAbReady, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }

  Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt receipt;
  ledger.complete = true;
  last_gdn_accepted_prefix_ = ledger;
  receipt.transaction_identity_ = ledger.transaction_identity;
  receipt.owner_identity_ = owner_identity_;
  receipt.request_epoch_ = request_epoch_;
  receipt.panel_ = active_panel_;
  receipt.panel_generation_ = active_panel_generation_;
  receipt.grant_identity_ = gdn_grant.grant_identity();
  receipt.grant_state_epoch_ = gdn_grant.state_epoch();
  receipt.recurrent_allocation_identity_ =
      gdn_grant.allocation_identity();
  receipt.gdn_ordinal_ = submission.gdn_ordinal;
  receipt.model_layer_ = submission.model_layer;
  receipt.active_bank_index_ = gdn_grant.active_bank_index();
  receipt.candidate_bank_index_ = gdn_grant.candidate_bank_index();
  receipt.active_conv_allocation_offset_ =
      gdn_grant.active_conv_allocation_offset();
  receipt.candidate_conv_allocation_offset_ =
      gdn_grant.candidate_conv_allocation_offset();
  receipt.conv_bytes_ = gdn_grant.conv_bytes();
  receipt.active_gdn_state_allocation_offset_ =
      gdn_grant.active_gdn_state_allocation_offset();
  receipt.candidate_gdn_state_allocation_offset_ =
      gdn_grant.candidate_gdn_state_allocation_offset();
  receipt.gdn_state_bytes_ = gdn_grant.gdn_state_bytes();
  receipt.authority_domain_ = submission.authority_domain;
  receipt.execution_package_identity_ =
      submission.execution_package_identity;
  receipt.gdn_catalog_identity_ = submission.gdn_catalog_identity;
  receipt.gdn_binding_identity_ = submission.gdn_binding_identity;
  receipt.bf16_ab_catalog_identity_ = submission.bf16_ab_catalog_identity;
  receipt.bf16_ab_pair_identity_ = submission.bf16_ab_pair_identity;
  receipt.layer_norm_catalog_identity_ =
      submission.layer_norm_catalog_identity;
  receipt.layer_norm_pair_identity_ = submission.layer_norm_pair_identity;
  receipt.input_norm_binding_identity_ =
      submission.input_norm_binding_identity;
  receipt.post_norm_binding_identity_ =
      submission.post_norm_binding_identity;
  receipt.mlp_catalog_identity_ = submission.mlp_catalog_identity;
  receipt.mlp_binding_identity_ = submission.mlp_binding_identity;
  receipt.resource_bundle_identity_ = submission.resource_bundle_identity;
  receipt.synthetic_source_identity_ = submission.synthetic_source_identity;
  receipt.submission_digest_ = gdn_submission_digest(submission);
  receipt.input_norm_launches_ = ledger.input_norm_launches;
  receipt.bf16_ab_launches_ = ledger.bf16_ab_launches;
  receipt.gdn_qkvz_launches_ = ledger.gdn_qkvz_launches;
  receipt.gdn_continuation_launches_ =
      ledger.gdn_continuation_launches;
  receipt.gdn_output_launches_ = ledger.gdn_output_launches;
  receipt.residual_post_norm_launches_ =
      ledger.residual_post_norm_launches;
  receipt.gate_up_launches_ = ledger.gate_up_launches;
  receipt.down_launches_ = ledger.down_launches;
  receipt.bound_kernel_submissions_ = ledger.accepted_kernel_launches;
  receipt.asynchronous_d2d_copies_ = ledger.asynchronous_d2d_copies;
  receipt.conv_history_copy_bytes_ = ledger.conv_history_copy_bytes;
  receipt.norm_ready_waited_by_ab_ = true;
  receipt.ab_ready_waited_by_main_ = true;
  receipt.complete_layer_enqueued_ = true;
  receipt.physical_device_completion_attested_ = false;
  receipt.panel_complete_ = false;
  receipt.production_receipt_eligible_ = false;
  receipt.authenticator_ = gdn_receipt_authenticator(receipt);
  if (!gdn_receipt_matches_locked(panel_access, gdn_grant, submission,
                                  receipt)) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kReceiptInvalid,
        "complete_gdn_layer_owner_authenticated_receipt", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kCount, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  ++complete_gdn_layers_submitted_;
  result.receipt = receipt;
  result.status = ok();
  return result;
}

Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::
    submit_complete_full_attention_layer_c8000_prevalidated(
        const Sm87MacroFeedV4ExecutionEventsAccess& access,
        const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
        const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
        const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
            submission) noexcept {
  namespace bound =
      kernels::sm87_macrofeed_v4_bound_launch_detail;
  std::lock_guard<std::mutex> lock(mutex_);
  Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueResult result;
  result.status = validate_operation_access(access, panel_access);
  if (!result.status) {
    return result;
  }

  Sm87MacroFeedV4FullAttentionAcceptedPrefixLedger ledger;
  ledger.transaction_identity =
      next_nonzero(&g_next_full_attention_layer_transaction_identity);
  ledger.owner_identity = owner_identity_;
  ledger.request_epoch = request_epoch_;
  ledger.panel = active_panel_;
  ledger.panel_generation = active_panel_generation_;
  ledger.first_position =
      active_panel_ * kernels::kSm87MacroFeedV4Fp8Tokens;
  ledger.grant_identity = kv_grant.grant_identity();
  ledger.grant_state_epoch = kv_grant.state_epoch();
  ledger.kv_allocation_identity = kv_grant.kv_allocation_identity();
  ledger.full_attention_ordinal = kv_grant.attention_layer_ordinal();
  ledger.model_layer = kv_grant.model_layer();
  last_full_attention_accepted_prefix_ = ledger;

  const void* const main_stream =
      streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kMain)];
  auto checked_input_norm = submission.input_norm;
  checked_input_norm.cuda_stream = const_cast<void*>(main_stream);
  const kernels::Sm87MacroFeedV4Fp8Arguments checked_full_qkv{
      kernels::Sm87TargetAotProjectionRole::kFp8FullQkv,
      submission.full_qkv.hidden_input,
      kernels::kSm87MacroFeedV4Fp8HiddenRowStride,
      submission.full_qkv.asset,
      kernels::kSm87MacroFeedV4Fp8Tokens,
      submission.full_qkv.q_gate_scratch,
      kernels::kSm87MacroFeedV4Fp8ScratchRowStride,
      submission.full_qkv.key_panel_output,
      kernels::kSm87MacroFeedV4Fp8KvNhdRowStride,
      submission.full_qkv.value_panel_output,
      kernels::kSm87MacroFeedV4Fp8KvNhdRowStride,
      const_cast<void*>(main_stream),
      kernels::Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1};
  const kernels::Sm87MacroFeedV4FullAttentionPreprocessArguments
      checked_preprocess{
          submission.preprocess.q_gate_scratch,
          kernels::kSm87MacroFeedV4FullAttentionPreprocessTokens,
          kernels::kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride,
          submission.preprocess.key_cache_origin,
          kernels::kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions,
          kernels::kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride,
          submission.preprocess.q_norm_weight,
          submission.preprocess.k_norm_weight,
          submission.preprocess.cosines,
          submission.preprocess.sines,
          kernels::kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions,
          kernels::kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf,
          submission.preprocess.first_position,
          kernels::kSm87MacroFeedV4FullAttentionPreprocessEpsilonFp32Bits,
          const_cast<void*>(main_stream)};
  const kernels::Sm87MacroFeedV4AttentionC8000Arguments checked_attention{
      submission.attention.q_gate_scratch,
      kernels::kSm87MacroFeedV4AttentionC8000Tokens,
      kernels::kSm87MacroFeedV4AttentionC8000ScratchRowStride,
      submission.attention.key_cache_origin,
      submission.attention.value_cache_origin,
      kernels::kSm87MacroFeedV4AttentionC8000MaximumPositions,
      kernels::kSm87MacroFeedV4AttentionC8000KvRowStride,
      submission.attention.first_position,
      const_cast<void*>(main_stream)};
  const kernels::Sm87MacroFeedV4Fp8Arguments checked_full_output{
      kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
      submission.full_output.q_gate_scratch,
      kernels::kSm87MacroFeedV4Fp8ScratchRowStride,
      submission.full_output.asset,
      kernels::kSm87MacroFeedV4Fp8Tokens,
      submission.full_output.branch_output,
      kernels::kSm87MacroFeedV4Fp8HiddenRowStride,
      nullptr,
      0U,
      nullptr,
      0U,
      const_cast<void*>(main_stream),
      kernels::Sm87MacroFeedV4Fp8InputLayout::
          kFullAttentionInterleavedQScratchV1};
  const kernels::Sm87MacroFeedV4ResidualPostNormArguments
      checked_residual_post_norm{
          submission.residual_post_norm.left_residual_then_normalized,
          submission.residual_post_norm.right_branch_then_residual,
          submission.residual_post_norm.centered_weight,
          kernels::kSm87MacroFeedV4NormResidualTokens,
          kernels::kSm87MacroFeedV4NormResidualHidden,
          kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits,
          const_cast<void*>(main_stream)};
  const kernels::Sm87MacroFeedV4NvFp4GateUpArguments checked_gate_up{
      submission.gate_up.normalized_input,
      submission.gate_up.payload,
      submission.gate_up.payload_bytes,
      submission.gate_up.gate_tensor_scale,
      submission.gate_up.up_tensor_scale,
      kernels::kSm87MacroFeedV4NvFp4GateUpTokens,
      submission.gate_up.intermediate_output,
      const_cast<void*>(main_stream),
      submission.gate_up.canonical_v3_payload_receipt};
  const kernels::Sm87MacroFeedV4NvFp4DownArguments checked_down{
      submission.down.intermediate_input,
      submission.down.payload,
      submission.down.payload_bytes,
      submission.down.tensor_scale,
      kernels::kSm87MacroFeedV4NvFp4DownTokens,
      submission.down.residual_output,
      const_cast<void*>(main_stream),
      submission.down.payload_receipt};

  constexpr auto kFullQkvIdentity =
      kernels::Sm87MacroFeedV4Fp8Identity::
          kFullQkvM64N128K64OrdinaryGridV1;
  constexpr auto kFullOutputIdentity =
      kernels::Sm87MacroFeedV4Fp8Identity::
          kAttentionOutputM64N128K64OrdinaryGridV1;
  static_assert(static_cast<std::uint64_t>(kFullQkvIdentity) ==
                0x5133'4d46'5634'4602ULL);
  static_assert(static_cast<std::uint64_t>(kFullOutputIdentity) ==
                0x5133'4d46'5634'4603ULL);

  const auto address = [](const void* const pointer) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer);
  };
  constexpr std::uint64_t kHiddenBytes =
      kernels::kSm87MacroFeedV4Fp8Tokens *
      kernels::kSm87MacroFeedV4Fp8HiddenRowStride *
      sizeof(std::uint16_t);
  constexpr std::uint64_t kScratchBytes =
      kernels::kSm87MacroFeedV4Fp8Tokens *
      kernels::kSm87MacroFeedV4Fp8ScratchRowStride *
      sizeof(std::uint16_t);
  constexpr std::uint64_t kKvBytes =
      kernels::kSm87MacroFeedV4AttentionC8000MaximumPositions *
      kernels::kSm87MacroFeedV4AttentionC8000KvRowStride *
      sizeof(std::uint16_t);
  constexpr std::uint64_t kNormWeightBytes =
      kernels::kSm87MacroFeedV4NormResidualHidden * sizeof(std::uint16_t);
  constexpr std::uint64_t kQkNormWeightBytes =
      kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension *
      sizeof(std::uint16_t);
  constexpr std::uint64_t kRopeBytes =
      kernels::kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions *
      kernels::kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf *
      sizeof(float);
  const std::array<ByteRange, 15U> non_aliasing_ranges{{
      byte_range(submission.input_norm.input_hidden, kHiddenBytes),
      byte_range(submission.input_norm.output_hidden, kHiddenBytes),
      byte_range(submission.full_qkv.q_gate_scratch, kScratchBytes),
      byte_range(submission.preprocess.key_cache_origin, kKvBytes),
      byte_range(submission.attention.value_cache_origin, kKvBytes),
      byte_range(submission.input_norm.centered_weight, kNormWeightBytes),
      byte_range(submission.residual_post_norm.centered_weight,
                 kNormWeightBytes),
      byte_range(submission.preprocess.q_norm_weight, kQkNormWeightBytes),
      byte_range(submission.preprocess.k_norm_weight, kQkNormWeightBytes),
      byte_range(submission.preprocess.cosines, kRopeBytes),
      byte_range(submission.preprocess.sines, kRopeBytes),
      byte_range(submission.full_qkv.asset.payload.begin,
                 submission.full_qkv.asset.payload.bytes),
      byte_range(submission.full_output.asset.payload.begin,
                 submission.full_output.asset.payload.bytes),
      byte_range(submission.gate_up.payload, submission.gate_up.payload_bytes),
      byte_range(submission.down.payload, submission.down.payload_bytes),
  }};
  const std::uintptr_t key_full_pointer =
      address(submission.preprocess.key_cache_origin);
  const std::uintptr_t value_full_pointer =
      address(submission.attention.value_cache_origin);
  const bool grant_offsets_fit_pointer =
      kv_grant.key_full_allocation_origin() <=
          std::numeric_limits<std::uintptr_t>::max() &&
      kv_grant.value_full_allocation_origin() <=
          std::numeric_limits<std::uintptr_t>::max() &&
      kv_grant.key_panel_allocation_offset() <=
          std::numeric_limits<std::uintptr_t>::max() &&
      kv_grant.value_panel_allocation_offset() <=
          std::numeric_limits<std::uintptr_t>::max() &&
      key_full_pointer >= kv_grant.key_full_allocation_origin();
  const std::uintptr_t kv_arena_pointer =
      grant_offsets_fit_pointer
          ? key_full_pointer - static_cast<std::uintptr_t>(
                                   kv_grant.key_full_allocation_origin())
          : 0U;
  const auto arena_offset_fits =
      [&](const std::uint64_t offset) noexcept {
        return offset <= std::numeric_limits<std::uintptr_t>::max() -
                             kv_arena_pointer;
      };
  const bool grant_pointer_arithmetic_valid =
      grant_offsets_fit_pointer &&
      arena_offset_fits(kv_grant.key_full_allocation_origin()) &&
      arena_offset_fits(kv_grant.value_full_allocation_origin()) &&
      arena_offset_fits(kv_grant.key_panel_allocation_offset()) &&
      arena_offset_fits(kv_grant.value_panel_allocation_offset());
  const bool exact_grant =
      kv_grant.grant_identity() != 0U &&
      kv_grant.owner_identity() == owner_identity_ &&
      kv_grant.request_epoch() == request_epoch_ &&
      kv_grant.kv_allocation_identity() != 0U &&
      kv_grant.panel() == active_panel_ &&
      kv_grant.attention_layer_ordinal() ==
          submission.full_attention_ordinal &&
      kv_grant.model_layer() == submission.model_layer &&
      submission.full_attention_ordinal <
          kSm87MacroFeedV4FullAttentionLayerCount &&
      submission.model_layer ==
          4U * submission.full_attention_ordinal + 3U &&
      kv_grant.key_full_allocation_origin() ==
          submission.full_attention_ordinal *
              kSm87MacroFeedV4AttentionKvLayerBytes &&
      kv_grant.value_full_allocation_origin() ==
          kv_grant.key_full_allocation_origin() +
              kSm87MacroFeedV4AttentionKvPlaneBytes &&
      kv_grant.key_panel_allocation_offset() ==
          kv_grant.key_full_allocation_origin() +
              active_panel_ * kSm87MacroFeedV4AttentionKvPanelBytes &&
      kv_grant.value_panel_allocation_offset() ==
          kv_grant.value_full_allocation_origin() +
              active_panel_ * kSm87MacroFeedV4AttentionKvPanelBytes &&
      kv_grant.panel_bytes() == kSm87MacroFeedV4AttentionKvPanelBytes &&
      kv_grant.first_position() == ledger.first_position &&
      kv_grant.previous_valid_end() == ledger.first_position &&
      kv_grant.candidate_end() ==
          ledger.first_position + kSm87MacroFeedV4PanelTokens;
  const bool normal_authority =
      submission.authority_domain ==
          Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain::
              kNormalSealedCatalog &&
      submission.execution_package_identity != 0U &&
      submission.full_attention_catalog_identity != 0U &&
      submission.full_attention_binding_identity != 0U &&
      submission.mlp_binding_identity != 0U &&
      submission.input_norm_binding_identity != 0U &&
      submission.post_norm_binding_identity != 0U &&
      submission.rope_binding_identity != 0U &&
      submission.resource_bundle_identity != 0U &&
      submission.synthetic_source_identity == 0U;
  const bool synthetic_authority =
      submission.authority_domain ==
          Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain::
              kSyntheticT1 &&
      submission.execution_package_identity == 0U &&
      submission.full_attention_catalog_identity == 0U &&
      submission.full_attention_binding_identity == 0U &&
      submission.mlp_binding_identity == 0U &&
      submission.input_norm_binding_identity == 0U &&
      submission.post_norm_binding_identity == 0U &&
      submission.rope_binding_identity == 0U &&
      submission.resource_bundle_identity == 0U &&
      submission.synthetic_source_identity != 0U;
  const bool exact_submission_identities =
      normal_authority || synthetic_authority;
  const bool exact_alias_graph =
      submission.input_norm.input_hidden ==
          submission.residual_post_norm.left_residual_then_normalized &&
      submission.input_norm.output_hidden ==
          submission.full_qkv.hidden_input &&
      submission.input_norm.output_hidden ==
          submission.full_output.branch_output &&
      submission.input_norm.output_hidden ==
          submission.residual_post_norm.right_branch_then_residual &&
      submission.full_qkv.q_gate_scratch ==
          submission.preprocess.q_gate_scratch &&
      submission.full_qkv.q_gate_scratch ==
          submission.attention.q_gate_scratch &&
      submission.full_qkv.q_gate_scratch ==
          submission.full_output.q_gate_scratch &&
      address(submission.preprocess.key_cache_origin) ==
          address(submission.attention.key_cache_origin) &&
      submission.preprocess.first_position ==
          submission.attention.first_position &&
      submission.preprocess.first_position == ledger.first_position &&
      grant_pointer_arithmetic_valid &&
      key_full_pointer ==
          kv_arena_pointer + static_cast<std::uintptr_t>(
                                 kv_grant.key_full_allocation_origin()) &&
      value_full_pointer ==
          kv_arena_pointer + static_cast<std::uintptr_t>(
                                 kv_grant.value_full_allocation_origin()) &&
      address(submission.full_qkv.key_panel_output) ==
          kv_arena_pointer + static_cast<std::uintptr_t>(
                                 kv_grant.key_panel_allocation_offset()) &&
      address(submission.full_qkv.value_panel_output) ==
          kv_arena_pointer + static_cast<std::uintptr_t>(
                                 kv_grant.value_panel_allocation_offset()) &&
      submission.gate_up.normalized_input ==
          submission.residual_post_norm.left_residual_then_normalized &&
      submission.gate_up.intermediate_output ==
          submission.down.intermediate_input &&
      submission.gate_up.intermediate_output ==
          submission.full_qkv.q_gate_scratch &&
      submission.down.residual_output ==
          submission.residual_post_norm.right_branch_then_residual &&
      pairwise_disjoint(non_aliasing_ranges);

  const bool all_resources_and_arguments_valid =
      !draining_ && ab_cycle_phase_ == AbCyclePhase::kExpectNormRecord &&
      submission.input_norm.cuda_stream == nullptr &&
      kernels::sm87_macrofeed_v4_input_norm_arguments_valid(
          checked_input_norm) &&
      kernels::sm87_macrofeed_v4_fp8_arguments_valid(checked_full_qkv) &&
      kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          checked_preprocess) &&
      kernels::sm87_macrofeed_v4_attention_c8000_arguments_valid(
          checked_attention) &&
      kernels::sm87_macrofeed_v4_fp8_arguments_valid(checked_full_output) &&
      kernels::sm87_macrofeed_v4_residual_post_norm_arguments_valid(
          checked_residual_post_norm) &&
      kernels::sm87_macrofeed_v4_nvfp4_gate_up_arguments_valid(
          checked_gate_up) &&
      kernels::sm87_macrofeed_v4_nvfp4_down_arguments_valid(checked_down) &&
      kernels::sm87_macrofeed_v4_norm_residual_resource_gate(
          submission.norm_resources) &&
      submission.norm_resources.static_resource_gate_passed &&
      kernels::sm87_macrofeed_v4_fp8_resource_gate(
          submission.full_qkv_resources) &&
      submission.full_qkv_resources.static_resource_gate_passed &&
      submission.full_qkv_resources.identity == kFullQkvIdentity &&
      submission.full_qkv_resources.role ==
          kernels::Sm87TargetAotProjectionRole::kFp8FullQkv &&
      submission.full_qkv_resources.input_layout ==
          kernels::Sm87MacroFeedV4Fp8InputLayout::
              kHiddenContiguousH5120V1 &&
      kernels::
          sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
              submission.preprocess_resources) &&
      submission.preprocess_resources.static_resource_gate_passed &&
      kernels::sm87_macrofeed_v4_attention_c8000_admission_resource_gate(
          submission.attention_resources) &&
      submission.attention_resources.static_resource_gate_passed &&
      kernels::sm87_macrofeed_v4_fp8_resource_gate(
          submission.full_output_resources) &&
      submission.full_output_resources.static_resource_gate_passed &&
      submission.full_output_resources.identity == kFullOutputIdentity &&
      submission.full_output_resources.role ==
          kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput &&
      submission.full_output_resources.input_layout ==
          kernels::Sm87MacroFeedV4Fp8InputLayout::
              kFullAttentionInterleavedQScratchV1 &&
      kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(
          submission.gate_up_resources) &&
      submission.gate_up_resources.static_resource_gate_passed &&
      kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(
          submission.down_resources) &&
      submission.down_resources.static_resource_gate_passed &&
      submission.norm_resources.device_ordinal == device_ordinal_ &&
      submission.full_qkv_resources.device_ordinal == device_ordinal_ &&
      submission.preprocess_resources.device_ordinal == device_ordinal_ &&
      submission.attention_resources.device_ordinal == device_ordinal_ &&
      submission.full_output_resources.device_ordinal == device_ordinal_ &&
      submission.gate_up_resources.device_ordinal == device_ordinal_ &&
      submission.down_resources.device_ordinal == device_ordinal_ &&
      exact_grant && exact_submission_identities && exact_alias_graph;
  if (!all_resources_and_arguments_valid) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
        "complete_full_attention_all_bindings_prevalidated_before_enqueue",
        0, Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kCount, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }

  static_assert(
      std::tuple_size<
          decltype(accepted_full_attention_grant_identities_)>::value ==
      kSm87MacroFeedV4PanelCount *
          kSm87MacroFeedV4FullAttentionLayerCount);
  const std::size_t grant_slot =
      active_panel_ * kSm87MacroFeedV4FullAttentionLayerCount +
      submission.full_attention_ordinal;
  if (accepted_full_attention_grant_identities_[grant_slot] != 0U) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
        "complete_full_attention_grant_at_most_once", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kCount, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  accepted_full_attention_grant_identities_[grant_slot] =
      kv_grant.grant_identity();
  ++accepted_full_attention_grant_count_;

  if (test_fail_full_after_accepted_prefix_ == 0U) {
    test_fail_full_after_accepted_prefix_ =
        std::numeric_limits<std::size_t>::max();
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kCudaSubmission,
        "test_full_attention_failure_after_zero_accepted", 1,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kCount, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }

  const bound::Sm87MacroFeedV4LockedSubmitToken main_token(
      streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kMain)]);
  const auto retain_step =
      [&](const int cuda_status, const std::size_t submitted,
          std::size_t* const operation_ledger,
          std::size_t* const global_operation_ledger,
          const char* const context) noexcept {
        *operation_ledger += submitted;
        ledger.accepted_kernel_launches += submitted;
        bound_kernel_submissions_ += submitted;
        *global_operation_ledger += submitted;
        last_full_attention_accepted_prefix_ = ledger;
        if (cuda_status != static_cast<int>(cudaSuccess) || submitted != 1U) {
          result.status = fail(
              cuda_status == static_cast<int>(cudaSuccess)
                  ? Sm87MacroFeedV4ExecutionError::kReceiptInvalid
                  : Sm87MacroFeedV4ExecutionError::kCudaSubmission,
              context, cuda_status, Sm87MacroFeedV4ExecutionStream::kMain,
              Sm87MacroFeedV4ExecutionEvent::kCount, active_panel_,
              active_panel_generation_);
          record_poison_cause(result.status);
          return false;
        }
        if (test_fail_full_after_accepted_prefix_ ==
            ledger.accepted_kernel_launches) {
          test_fail_full_after_accepted_prefix_ =
              std::numeric_limits<std::size_t>::max();
          result.status = fail(
              Sm87MacroFeedV4ExecutionError::kCudaSubmission,
              "test_full_attention_failure_after_accepted_prefix", 1,
              Sm87MacroFeedV4ExecutionStream::kMain,
              Sm87MacroFeedV4ExecutionEvent::kCount, active_panel_,
              active_panel_generation_);
          record_poison_cause(result.status);
          return false;
        }
        return true;
      };

  std::size_t submitted = 0U;
  int cuda_status = bound::enqueue_input_norm_prevalidated(
      main_token, submission.input_norm, submission.norm_resources,
      &submitted);
  if (!retain_step(cuda_status, submitted, &ledger.input_norm_launches,
                   &input_norm_submissions_,
                   "complete_full_attention_input_norm_exact_one_launch")) {
    return result;
  }

  submitted = 0U;
  cuda_status = bound::enqueue_full_qkv_c8000_prevalidated(
      main_token, submission.full_qkv, submission.full_qkv_resources,
      &submitted);
  if (!retain_step(cuda_status, submitted, &ledger.full_qkv_launches,
                   &full_qkv_c8000_submissions_,
                   "complete_full_attention_qkv_exact_one_launch")) {
    return result;
  }

  submitted = 0U;
  cuda_status = bound::enqueue_full_attention_preprocess_c8000_prevalidated(
      main_token, submission.preprocess, submission.preprocess_resources,
      &submitted);
  if (!retain_step(
          cuda_status, submitted, &ledger.preprocess_launches,
          &full_attention_preprocess_c8000_submissions_,
          "complete_full_attention_preprocess_exact_one_launch")) {
    return result;
  }

  submitted = 0U;
  cuda_status = bound::enqueue_attention_c8000_prevalidated(
      main_token, submission.attention, submission.attention_resources,
      &submitted);
  if (!retain_step(cuda_status, submitted, &ledger.attention_launches,
                   &attention_c8000_submissions_,
                   "complete_full_attention_core_exact_one_launch")) {
    return result;
  }

  submitted = 0U;
  cuda_status = bound::enqueue_full_attention_o_c8000_prevalidated(
      main_token, submission.full_output, submission.full_output_resources,
      &submitted);
  if (!retain_step(
          cuda_status, submitted, &ledger.full_output_launches,
          &full_attention_output_c8000_submissions_,
          "complete_full_attention_output_exact_one_launch")) {
    return result;
  }

  submitted = 0U;
  cuda_status = bound::enqueue_residual_post_norm_prevalidated(
      main_token, submission.residual_post_norm, submission.norm_resources,
      &submitted);
  if (!retain_step(
          cuda_status, submitted, &ledger.residual_post_norm_launches,
          &residual_post_norm_submissions_,
          "complete_full_attention_residual_post_norm_exact_one_launch")) {
    return result;
  }

  submitted = 0U;
  cuda_status = bound::enqueue_gate_up_c8000_prevalidated(
      main_token, submission.gate_up, submission.gate_up_resources,
      &submitted);
  if (!retain_step(cuda_status, submitted, &ledger.gate_up_launches,
                   &gate_up_c8000_submissions_,
                   "complete_full_attention_gate_up_exact_one_launch")) {
    return result;
  }

  submitted = 0U;
  cuda_status = bound::enqueue_down_c8000_prevalidated(
      main_token, submission.down, submission.down_resources, &submitted);
  if (!retain_step(cuda_status, submitted, &ledger.down_launches,
                   &down_c8000_submissions_,
                   "complete_full_attention_down_exact_one_launch")) {
    return result;
  }

  ledger.complete = true;
  last_full_attention_accepted_prefix_ = ledger;
  Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt receipt;
  receipt.transaction_identity_ = ledger.transaction_identity;
  receipt.owner_identity_ = owner_identity_;
  receipt.request_epoch_ = request_epoch_;
  receipt.panel_ = active_panel_;
  receipt.panel_generation_ = active_panel_generation_;
  receipt.first_position_ = ledger.first_position;
  receipt.grant_identity_ = kv_grant.grant_identity();
  receipt.grant_state_epoch_ = kv_grant.state_epoch();
  receipt.kv_allocation_identity_ = kv_grant.kv_allocation_identity();
  receipt.key_full_allocation_origin_ =
      kv_grant.key_full_allocation_origin();
  receipt.value_full_allocation_origin_ =
      kv_grant.value_full_allocation_origin();
  receipt.key_panel_allocation_offset_ =
      kv_grant.key_panel_allocation_offset();
  receipt.value_panel_allocation_offset_ =
      kv_grant.value_panel_allocation_offset();
  receipt.kv_panel_bytes_ = kv_grant.panel_bytes();
  receipt.previous_valid_end_ = kv_grant.previous_valid_end();
  receipt.candidate_end_ = kv_grant.candidate_end();
  receipt.full_attention_ordinal_ = submission.full_attention_ordinal;
  receipt.model_layer_ = submission.model_layer;
  receipt.authority_domain_ = submission.authority_domain;
  receipt.execution_package_identity_ =
      submission.execution_package_identity;
  receipt.full_attention_catalog_identity_ =
      submission.full_attention_catalog_identity;
  receipt.full_attention_binding_identity_ =
      submission.full_attention_binding_identity;
  receipt.mlp_binding_identity_ = submission.mlp_binding_identity;
  receipt.input_norm_binding_identity_ =
      submission.input_norm_binding_identity;
  receipt.post_norm_binding_identity_ =
      submission.post_norm_binding_identity;
  receipt.rope_binding_identity_ = submission.rope_binding_identity;
  receipt.resource_bundle_identity_ = submission.resource_bundle_identity;
  receipt.synthetic_source_identity_ =
      submission.synthetic_source_identity;
  receipt.submission_digest_ = full_attention_submission_digest(submission);
  receipt.input_norm_launches_ = ledger.input_norm_launches;
  receipt.full_qkv_launches_ = ledger.full_qkv_launches;
  receipt.preprocess_launches_ = ledger.preprocess_launches;
  receipt.attention_launches_ = ledger.attention_launches;
  receipt.full_output_launches_ = ledger.full_output_launches;
  receipt.residual_post_norm_launches_ =
      ledger.residual_post_norm_launches;
  receipt.gate_up_launches_ = ledger.gate_up_launches;
  receipt.down_launches_ = ledger.down_launches;
  receipt.bound_kernel_submissions_ = ledger.accepted_kernel_launches;
  receipt.asynchronous_d2d_copies_ = ledger.asynchronous_d2d_copies;
  receipt.asynchronous_d2d_copy_bytes_ =
      ledger.asynchronous_d2d_copy_bytes;
  receipt.complete_layer_enqueued_ = true;
  receipt.physical_device_completion_attested_ = false;
  receipt.panel_complete_ = false;
  receipt.production_receipt_eligible_ = false;
  receipt.authenticator_ = full_attention_receipt_authenticator(receipt);
  if (!full_attention_receipt_matches_locked(panel_access, kv_grant,
                                             submission, receipt)) {
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kReceiptInvalid,
        "complete_full_attention_owner_authenticated_receipt", 0,
        Sm87MacroFeedV4ExecutionStream::kMain,
        Sm87MacroFeedV4ExecutionEvent::kCount, active_panel_,
        active_panel_generation_);
    record_poison_cause(result.status);
    return result;
  }
  ++complete_full_attention_layers_submitted_;
  result.receipt = receipt;
  result.status = ok();
  return result;
}
#endif

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsOwner::initialize_cold_recurrent_storage(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    void* const recurrent_allocation,
    const std::size_t recurrent_bytes,
    const std::uint64_t recurrent_allocation_identity) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!owner_access_matches(access)) {
    return fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                "cold_recurrent_zero_owner_issued_access");
  }
  if (state_ != Sm87MacroFeedV4ExecutionOwnerState::kReady ||
      recurrent_allocation == nullptr ||
      reinterpret_cast<std::uintptr_t>(recurrent_allocation) % 256U != 0U ||
      recurrent_bytes != kSm87MacroFeedV4RecurrentStorageBytes ||
      recurrent_allocation_identity == 0U ||
      cold_recurrent_initializations_ != 0U ||
      cold_recurrent_allocation_identity_ != 0U ||
      cold_recurrent_allocation_begin_ != 0U ||
      cold_recurrent_zero_bytes_ != 0U) {
    return fail(Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
                "cold_recurrent_zero_exact_once_before_request");
  }

  const auto control = reinterpret_cast<cudaStream_t>(
      streams_[stream_index(Sm87MacroFeedV4ExecutionStream::kControl)]);
  cudaError_t cuda_status =
      cudaMemsetAsync(recurrent_allocation, 0, recurrent_bytes, control);
  if (cuda_status != cudaSuccess) {
    const auto status = fail(
        Sm87MacroFeedV4ExecutionError::kCudaSubmission,
        "cold_recurrent_zero_memset_enqueue", static_cast<int>(cuda_status),
        Sm87MacroFeedV4ExecutionStream::kControl);
    record_poison_cause(status);
    return status;
  }
  // Construction-only physical fence.  No request or panel is active and no
  // request-hot path may repeat this whole-arena operation.
  cuda_status = cudaStreamSynchronize(control);
  if (cuda_status != cudaSuccess) {
    const auto status = fail(
        Sm87MacroFeedV4ExecutionError::kCudaObservation,
        "cold_recurrent_zero_control_observation",
        static_cast<int>(cuda_status),
        Sm87MacroFeedV4ExecutionStream::kControl);
    record_poison_cause(status);
    return status;
  }
  cold_recurrent_initializations_ = 1U;
  cold_recurrent_allocation_identity_ = recurrent_allocation_identity;
  cold_recurrent_allocation_begin_ =
      reinterpret_cast<std::uintptr_t>(recurrent_allocation);
  cold_recurrent_zero_bytes_ = recurrent_bytes;
  return ok();
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::wait_event(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionStream consumer,
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return wait_event_locked(access, panel_access, consumer, event);
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsOwner::wait_event_locked(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionStream consumer,
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
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

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
std::uint64_t Sm87MacroFeedV4ExecutionEventsOwner::gdn_submission_digest(
    const Sm87MacroFeedV4CompleteGdnLayerC8000Submission& submission)
    const noexcept {
  // Mix every semantic field explicitly; never inspect struct padding.
  std::uint64_t value = mix64(0x5133'4d46'5634'4744ULL);
  const auto add = [&value](const std::uint64_t fact) noexcept {
    value = mix64(value ^ fact);
  };
  const auto add_size = [&add](const std::size_t fact) noexcept {
    add(static_cast<std::uint64_t>(fact));
  };
  const auto add_i32 = [&add](const std::int32_t fact) noexcept {
    add(static_cast<std::uint64_t>(static_cast<std::int64_t>(fact)));
  };
  const auto add_bool = [&add](const bool fact) noexcept {
    add(static_cast<std::uint64_t>(fact));
  };
  const auto add_pointer = [&add](const void* const pointer) noexcept {
    add(static_cast<std::uint64_t>(
        reinterpret_cast<std::uintptr_t>(pointer)));
  };
  const auto add_float = [&add](const float fact) noexcept {
    std::uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(fact));
    std::memcpy(&bits, &fact, sizeof(bits));
    add(bits);
  };
  const auto add_fp8_asset = [&](const auto& asset) noexcept {
    add(static_cast<std::uint64_t>(asset.payload.role));
    add(static_cast<std::uint64_t>(asset.payload.plan_identity));
    add(static_cast<std::uint64_t>(asset.payload.layout_identity));
    add(asset.payload.begin);
    add(asset.payload.end);
    add(asset.payload.bytes);
    add_bool(asset.payload.valid);
    add(asset.artifact_identity);
    add(asset.source_inventory_identity);
    add(static_cast<std::uint64_t>(asset.transform_identity));
    for (const std::uint8_t byte : asset.host_payload_digest.bytes) {
      add(byte);
    }
    add(asset.host_manifest_seal.value);
    const auto& upload = asset.device_upload_receipt;
    add(upload.receipt_identity);
    add(upload.artifact_identity);
    add(upload.source_inventory_identity);
    add(static_cast<std::uint64_t>(upload.role));
    add(static_cast<std::uint64_t>(upload.plan_identity));
    add(static_cast<std::uint64_t>(upload.layout_identity));
    add(static_cast<std::uint64_t>(upload.transform_identity));
    add(upload.device_allocation_identity);
    add(upload.device_allocation_owner_identity);
    add_i32(upload.device_ordinal);
    add(upload.device_allocation_begin);
    add(upload.device_allocation_end);
    add(upload.device_allocation_bytes);
    add(upload.device_payload_begin);
    add(upload.device_payload_end);
    add(upload.device_payload_bytes);
    for (const std::uint32_t bits : asset.tensor_scale_bits) {
      add(bits);
    }
    for (const std::uint16_t bits :
         asset.compensated_tensor_scale_bf16_bits) {
      add(bits);
    }
    add(asset.tensor_scale_count);
    add_bool(asset.no_request_time_repacking);
    add_bool(asset.no_request_time_scale_conversion);
    add_bool(asset.valid);
  };
  const auto add_gate_up_receipt = [&](const auto& receipt) noexcept {
    add(receipt.receipt_identity);
    add(static_cast<std::uint64_t>(receipt.plan_identity));
    add(receipt.payload_identity);
    add(receipt.gate_source_identity);
    add(receipt.up_source_identity);
    add_i32(receipt.device_ordinal);
    add(receipt.payload_begin);
    add(receipt.payload_end);
    add(receipt.payload_bytes);
    add(receipt.gate_partition_bytes);
    add(receipt.up_partition_bytes);
    add_bool(receipt.canonical_consumer_n64_k16_lane_component_v1);
    add_bool(receipt.canonical_gate_then_up_partition_order);
    add_bool(receipt.independent_tensor_scales);
    add_bool(receipt.host_bytes_authenticated_before_copy);
    add_bool(receipt.device_readback_authenticated);
    add_bool(receipt.allocation_retained_for_launch);
  };
  const auto add_down_receipt = [&](const auto& receipt) noexcept {
    add(receipt.receipt_identity);
    add(static_cast<std::uint64_t>(receipt.plan_identity));
    add(receipt.payload_identity);
    add_i32(receipt.device_ordinal);
    add(receipt.payload_begin);
    add(receipt.payload_end);
    add(receipt.payload_bytes);
    add_bool(receipt.canonical_consumer_n64_k16_lane_component_v1);
    add_bool(receipt.host_bytes_authenticated_before_copy);
    add_bool(receipt.device_readback_authenticated);
    add_bool(receipt.allocation_retained_for_launch);
  };
  const auto add_kernel_resources = [&](const auto& kernel) noexcept {
    add_i32(kernel.registers_per_thread);
    add_size(kernel.static_shared_bytes);
    add_size(kernel.local_bytes);
    add_i32(kernel.maximum_threads_per_block);
    add_i32(kernel.active_blocks_per_sm);
    add_i32(kernel.threads_per_block);
    add_i32(kernel.physical_grid_ctas);
  };
  const auto add_norm_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_kernel_resources(resources.input_norm);
    add_kernel_resources(resources.fused_residual_norm);
    add_bool(resources.kernels_compiled);
    add_bool(resources.exact_geometry);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
    add_bool(resources.startup_package_unbound);
    add_bool(resources.execution_capability);
    add_bool(resources.caller_snapshot_grants_production_authority);
  };
  const auto add_bf16_ab_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_i32(resources.registers_per_thread);
    add_size(resources.static_shared_bytes);
    add_size(resources.dynamic_shared_bytes);
    add_size(resources.local_bytes);
    add_i32(resources.maximum_threads_per_block);
    add_i32(resources.active_blocks_per_sm);
    add_i32(resources.threads_per_block);
    add_i32(resources.physical_grid_ctas);
    add_bool(resources.kernel_compiled);
    add_bool(resources.exact_geometry);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
    add_bool(resources.startup_package_unbound);
    add_bool(resources.execution_capability);
    add_bool(resources.caller_snapshot_grants_production_authority);
  };
  const auto add_fp8_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add(static_cast<std::uint64_t>(resources.role));
    add(static_cast<std::uint64_t>(resources.input_layout));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_i32(resources.registers_per_thread);
    add_size(resources.static_shared_bytes);
    add_size(resources.dynamic_shared_bytes);
    add_size(resources.local_bytes);
    add_i32(resources.maximum_threads_per_block);
    add_i32(resources.active_blocks_per_sm);
    add_size(resources.shared_bytes_per_sm);
    add_size(resources.optin_shared_bytes_per_block);
    add_bool(resources.kernel_compiled);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
  };
  const auto add_gdn_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_kernel_resources(resources.convolution);
    add_kernel_resources(resources.recurrence_epilogue);
    add_bool(resources.kernels_compiled);
    add_bool(resources.exact_geometry);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
    add_bool(resources.startup_package_unbound);
    add_bool(resources.execution_capability);
    add_bool(resources.caller_snapshot_grants_production_authority);
  };
  const auto add_gate_up_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_i32(resources.registers_per_thread);
    add_size(resources.static_shared_bytes);
    add_size(resources.dynamic_shared_bytes);
    add_size(resources.local_bytes);
    add_i32(resources.maximum_threads_per_block);
    add_i32(resources.active_blocks_per_sm);
    add_bool(resources.kernel_compiled);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
  };
  const auto add_down_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_i32(resources.registers_per_thread);
    add_size(resources.static_shared_bytes);
    add_size(resources.dynamic_shared_bytes);
    add_size(resources.local_bytes);
    add_i32(resources.maximum_threads_per_block);
    add_i32(resources.active_blocks_per_sm);
    add_size(resources.shared_bytes_per_sm);
    add_size(resources.optin_shared_bytes_per_block);
    add_bool(resources.kernel_compiled);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
  };

  add(static_cast<std::uint64_t>(submission.authority_domain));
  add(submission.execution_package_identity);
  add(submission.gdn_catalog_identity);
  add(submission.gdn_binding_identity);
  add(submission.bf16_ab_catalog_identity);
  add(submission.bf16_ab_pair_identity);
  add(submission.layer_norm_catalog_identity);
  add(submission.layer_norm_pair_identity);
  add(submission.input_norm_binding_identity);
  add(submission.post_norm_binding_identity);
  add(submission.mlp_catalog_identity);
  add(submission.mlp_binding_identity);
  add(submission.resource_bundle_identity);
  add(submission.synthetic_source_identity);
  add_size(submission.gdn_ordinal);
  add_size(submission.model_layer);

  add(1U);
  add_pointer(submission.input_norm.input_hidden);
  add_pointer(submission.input_norm.centered_weight);
  add_pointer(submission.input_norm.output_hidden);
  add_size(submission.input_norm.token_count);
  add_size(submission.input_norm.hidden_size);
  add(submission.input_norm.epsilon_fp32_bits);
  add_pointer(submission.input_norm.cuda_stream);

  add(2U);
  add_pointer(submission.bf16_ab.a_weights);
  add_pointer(submission.bf16_ab.b_weights);
  add_pointer(submission.bf16_ab.input);
  add_pointer(submission.bf16_ab.scratch);
  add_size(submission.bf16_ab.token_count);
  add_size(submission.bf16_ab.scratch_row_stride);
  add_pointer(submission.bf16_ab.cuda_stream);

  add(3U);
  add_pointer(submission.gdn_qkvz.hidden_input);
  add_fp8_asset(submission.gdn_qkvz.asset);
  add_pointer(submission.gdn_qkvz.phase_scratch);

  add(4U);
  add_pointer(submission.gdn_continuation.phase_scratch);
  add_pointer(submission.gdn_continuation.conv_weight);
  add_pointer(submission.gdn_continuation.a_log);
  add_pointer(submission.gdn_continuation.dt_bias);
  add_pointer(submission.gdn_continuation.norm_weight);
  add_pointer(submission.gdn_continuation.active_conv_history);
  add_pointer(submission.gdn_continuation.candidate_conv_history);
  add_pointer(submission.gdn_continuation.active_recurrent_state);
  add_pointer(submission.gdn_continuation.candidate_recurrent_state);
  add_pointer(submission.gdn_continuation.cancellation_signal);
  add(submission.gdn_continuation.l2_epsilon_fp32_bits);
  add(submission.gdn_continuation.norm_epsilon_fp32_bits);

  add(5U);
  add_pointer(submission.gdn_output.phase_scratch);
  add_fp8_asset(submission.gdn_output.asset);
  add_pointer(submission.gdn_output.branch_output);

  add(6U);
  add_pointer(submission.residual_post_norm.left_residual_then_normalized);
  add_pointer(submission.residual_post_norm.right_branch_then_residual);
  add_pointer(submission.residual_post_norm.centered_weight);

  add(7U);
  add_pointer(submission.gate_up.normalized_input);
  add_pointer(submission.gate_up.payload);
  add_size(submission.gate_up.payload_bytes);
  add_float(submission.gate_up.gate_tensor_scale);
  add_float(submission.gate_up.up_tensor_scale);
  add_pointer(submission.gate_up.intermediate_output);
  add_gate_up_receipt(submission.gate_up.canonical_v3_payload_receipt);

  add(8U);
  add_pointer(submission.down.intermediate_input);
  add_pointer(submission.down.payload);
  add_size(submission.down.payload_bytes);
  add_float(submission.down.tensor_scale);
  add_pointer(submission.down.residual_output);
  add_down_receipt(submission.down.payload_receipt);

  add(9U);
  add_norm_resources(submission.norm_resources);
  add_bf16_ab_resources(submission.bf16_ab_resources);
  add_fp8_resources(submission.gdn_qkvz_resources);
  add_gdn_resources(submission.gdn_continuation_resources);
  add_fp8_resources(submission.gdn_output_resources);
  add_gate_up_resources(submission.gate_up_resources);
  add_down_resources(submission.down_resources);
  return value == 0U ? 0x5133'4d46'5634'4744ULL : value;
}

std::uint64_t
Sm87MacroFeedV4ExecutionEventsOwner::gdn_receipt_authenticator(
    const Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt& receipt)
    const noexcept {
  std::uint64_t value =
      mix64(receipt_secret_ ^ 0x5133'4d46'5634'4741ULL);
  const auto add = [&value](const std::uint64_t fact) noexcept {
    value = mix64(value ^ fact);
  };
  add(receipt.transaction_identity_);
  add(receipt.owner_identity_);
  add(receipt.request_epoch_);
  add(receipt.panel_);
  add(receipt.panel_generation_);
  add(receipt.grant_identity_);
  add(receipt.grant_state_epoch_);
  add(receipt.recurrent_allocation_identity_);
  add(receipt.gdn_ordinal_);
  add(receipt.model_layer_);
  add(receipt.active_bank_index_);
  add(receipt.candidate_bank_index_);
  add(receipt.active_conv_allocation_offset_);
  add(receipt.candidate_conv_allocation_offset_);
  add(receipt.conv_bytes_);
  add(receipt.active_gdn_state_allocation_offset_);
  add(receipt.candidate_gdn_state_allocation_offset_);
  add(receipt.gdn_state_bytes_);
  add(static_cast<std::uint64_t>(receipt.authority_domain_));
  add(receipt.execution_package_identity_);
  add(receipt.gdn_catalog_identity_);
  add(receipt.gdn_binding_identity_);
  add(receipt.bf16_ab_catalog_identity_);
  add(receipt.bf16_ab_pair_identity_);
  add(receipt.layer_norm_catalog_identity_);
  add(receipt.layer_norm_pair_identity_);
  add(receipt.input_norm_binding_identity_);
  add(receipt.post_norm_binding_identity_);
  add(receipt.mlp_catalog_identity_);
  add(receipt.mlp_binding_identity_);
  add(receipt.resource_bundle_identity_);
  add(receipt.synthetic_source_identity_);
  add(receipt.submission_digest_);
  add(receipt.input_norm_launches_);
  add(receipt.bf16_ab_launches_);
  add(receipt.gdn_qkvz_launches_);
  add(receipt.gdn_continuation_launches_);
  add(receipt.gdn_output_launches_);
  add(receipt.residual_post_norm_launches_);
  add(receipt.gate_up_launches_);
  add(receipt.down_launches_);
  add(receipt.bound_kernel_submissions_);
  add(receipt.asynchronous_d2d_copies_);
  add(receipt.conv_history_copy_bytes_);
  add(receipt.norm_ready_waited_by_ab_);
  add(receipt.ab_ready_waited_by_main_);
  add(receipt.complete_layer_enqueued_);
  add(receipt.physical_device_completion_attested_);
  add(receipt.panel_complete_);
  add(receipt.production_receipt_eligible_);
  return value;
}

bool Sm87MacroFeedV4ExecutionEventsOwner::gdn_receipt_matches_locked(
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
    const Sm87MacroFeedV4CompleteGdnLayerC8000Submission& expected_submission,
    const Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt& receipt)
    const noexcept {
  const bool slot_valid =
      receipt.gdn_ordinal_ < kSm87MacroFeedV4StateLayerCount &&
      receipt.panel_ < kSm87MacroFeedV4PanelCount;
  const std::size_t slot =
      slot_valid
          ? receipt.panel_ * kSm87MacroFeedV4StateLayerCount +
                receipt.gdn_ordinal_
          : accepted_gdn_grant_identities_.size();
  return panel_access_matches(panel_access) && receipt.valid_shape() &&
         slot < accepted_gdn_grant_identities_.size() &&
         accepted_gdn_grant_identities_[slot] ==
             gdn_grant.grant_identity() &&
         receipt.owner_identity_ == owner_identity_ &&
         receipt.request_epoch_ == request_epoch_ &&
         receipt.panel_ == active_panel_ &&
         receipt.panel_generation_ == active_panel_generation_ &&
         gdn_grant.owner_identity() == owner_identity_ &&
         gdn_grant.allocation_identity() == request_allocation_identity_ &&
         gdn_grant.request_epoch() == request_epoch_ &&
         gdn_grant.panel() == active_panel_ &&
         receipt.grant_identity_ == gdn_grant.grant_identity() &&
         receipt.grant_state_epoch_ == gdn_grant.state_epoch() &&
         receipt.recurrent_allocation_identity_ ==
             gdn_grant.allocation_identity() &&
         receipt.gdn_ordinal_ == gdn_grant.state_layer_ordinal() &&
         receipt.model_layer_ == gdn_grant.model_layer() &&
         receipt.active_bank_index_ == gdn_grant.active_bank_index() &&
         receipt.candidate_bank_index_ ==
             gdn_grant.candidate_bank_index() &&
         receipt.active_conv_allocation_offset_ ==
             gdn_grant.active_conv_allocation_offset() &&
         receipt.candidate_conv_allocation_offset_ ==
             gdn_grant.candidate_conv_allocation_offset() &&
         receipt.conv_bytes_ == gdn_grant.conv_bytes() &&
         receipt.active_gdn_state_allocation_offset_ ==
             gdn_grant.active_gdn_state_allocation_offset() &&
         receipt.candidate_gdn_state_allocation_offset_ ==
             gdn_grant.candidate_gdn_state_allocation_offset() &&
         receipt.gdn_state_bytes_ == gdn_grant.gdn_state_bytes() &&
         receipt.authority_domain_ == expected_submission.authority_domain &&
         receipt.execution_package_identity_ ==
             expected_submission.execution_package_identity &&
         receipt.gdn_catalog_identity_ ==
             expected_submission.gdn_catalog_identity &&
         receipt.gdn_binding_identity_ ==
             expected_submission.gdn_binding_identity &&
         receipt.bf16_ab_catalog_identity_ ==
             expected_submission.bf16_ab_catalog_identity &&
         receipt.bf16_ab_pair_identity_ ==
             expected_submission.bf16_ab_pair_identity &&
         receipt.layer_norm_catalog_identity_ ==
             expected_submission.layer_norm_catalog_identity &&
         receipt.layer_norm_pair_identity_ ==
             expected_submission.layer_norm_pair_identity &&
         receipt.input_norm_binding_identity_ ==
             expected_submission.input_norm_binding_identity &&
         receipt.post_norm_binding_identity_ ==
             expected_submission.post_norm_binding_identity &&
         receipt.mlp_catalog_identity_ ==
             expected_submission.mlp_catalog_identity &&
         receipt.mlp_binding_identity_ ==
             expected_submission.mlp_binding_identity &&
         receipt.resource_bundle_identity_ ==
             expected_submission.resource_bundle_identity &&
         receipt.synthetic_source_identity_ ==
             expected_submission.synthetic_source_identity &&
         receipt.gdn_ordinal_ == expected_submission.gdn_ordinal &&
         receipt.model_layer_ == expected_submission.model_layer &&
         receipt.submission_digest_ ==
             gdn_submission_digest(expected_submission) &&
         receipt.authenticator_ != 0U &&
         receipt.authenticator_ == gdn_receipt_authenticator(receipt);
}

bool Sm87MacroFeedV4ExecutionEventsOwner::gdn_receipt_matches(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
    const Sm87MacroFeedV4CompleteGdnLayerC8000Submission& expected_submission,
    const Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt& receipt)
    const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return owner_access_matches(access) &&
         state_ == Sm87MacroFeedV4ExecutionOwnerState::kRequestActive &&
         gdn_receipt_matches_locked(panel_access, gdn_grant,
                                    expected_submission, receipt);
}

std::uint64_t
Sm87MacroFeedV4ExecutionEventsOwner::full_attention_submission_digest(
    const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
        submission) const noexcept {
  // Every fact is mixed explicitly.  In particular, no object representation
  // or struct padding is read, so the digest is stable across compilers while
  // still binding the exact package-to-owner submission.
  std::uint64_t value = mix64(0x5133'4d46'5634'5344ULL);
  const auto add = [&value](const std::uint64_t fact) noexcept {
    value = mix64(value ^ fact);
  };
  const auto add_size = [&add](const std::size_t fact) noexcept {
    add(static_cast<std::uint64_t>(fact));
  };
  const auto add_i32 = [&add](const std::int32_t fact) noexcept {
    add(static_cast<std::uint64_t>(static_cast<std::int64_t>(fact)));
  };
  const auto add_bool = [&add](const bool fact) noexcept {
    add(static_cast<std::uint64_t>(fact));
  };
  const auto add_pointer = [&add](const void* const pointer) noexcept {
    add(static_cast<std::uint64_t>(
        reinterpret_cast<std::uintptr_t>(pointer)));
  };
  const auto add_float = [&add](const float fact) noexcept {
    std::uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(fact));
    std::memcpy(&bits, &fact, sizeof(bits));
    add(bits);
  };
  const auto add_fp8_asset = [&](const auto& asset) noexcept {
    add(static_cast<std::uint64_t>(asset.payload.role));
    add(static_cast<std::uint64_t>(asset.payload.plan_identity));
    add(static_cast<std::uint64_t>(asset.payload.layout_identity));
    add(asset.payload.begin);
    add(asset.payload.end);
    add(asset.payload.bytes);
    add_bool(asset.payload.valid);
    add(asset.artifact_identity);
    add(asset.source_inventory_identity);
    add(static_cast<std::uint64_t>(asset.transform_identity));
    for (const std::uint8_t byte : asset.host_payload_digest.bytes) {
      add(byte);
    }
    add(asset.host_manifest_seal.value);
    const auto& upload = asset.device_upload_receipt;
    add(upload.receipt_identity);
    add(upload.artifact_identity);
    add(upload.source_inventory_identity);
    add(static_cast<std::uint64_t>(upload.role));
    add(static_cast<std::uint64_t>(upload.plan_identity));
    add(static_cast<std::uint64_t>(upload.layout_identity));
    add(static_cast<std::uint64_t>(upload.transform_identity));
    add(upload.device_allocation_identity);
    add(upload.device_allocation_owner_identity);
    add_i32(upload.device_ordinal);
    add(upload.device_allocation_begin);
    add(upload.device_allocation_end);
    add(upload.device_allocation_bytes);
    add(upload.device_payload_begin);
    add(upload.device_payload_end);
    add(upload.device_payload_bytes);
    for (const std::uint32_t bits : asset.tensor_scale_bits) {
      add(bits);
    }
    for (const std::uint16_t bits :
         asset.compensated_tensor_scale_bf16_bits) {
      add(bits);
    }
    add(asset.tensor_scale_count);
    add_bool(asset.no_request_time_repacking);
    add_bool(asset.no_request_time_scale_conversion);
    add_bool(asset.valid);
  };
  const auto add_gate_up_receipt = [&](const auto& receipt) noexcept {
    add(receipt.receipt_identity);
    add(static_cast<std::uint64_t>(receipt.plan_identity));
    add(receipt.payload_identity);
    add(receipt.gate_source_identity);
    add(receipt.up_source_identity);
    add_i32(receipt.device_ordinal);
    add(receipt.payload_begin);
    add(receipt.payload_end);
    add(receipt.payload_bytes);
    add(receipt.gate_partition_bytes);
    add(receipt.up_partition_bytes);
    add_bool(receipt.canonical_consumer_n64_k16_lane_component_v1);
    add_bool(receipt.canonical_gate_then_up_partition_order);
    add_bool(receipt.independent_tensor_scales);
    add_bool(receipt.host_bytes_authenticated_before_copy);
    add_bool(receipt.device_readback_authenticated);
    add_bool(receipt.allocation_retained_for_launch);
  };
  const auto add_down_receipt = [&](const auto& receipt) noexcept {
    add(receipt.receipt_identity);
    add(static_cast<std::uint64_t>(receipt.plan_identity));
    add(receipt.payload_identity);
    add_i32(receipt.device_ordinal);
    add(receipt.payload_begin);
    add(receipt.payload_end);
    add(receipt.payload_bytes);
    add_bool(receipt.canonical_consumer_n64_k16_lane_component_v1);
    add_bool(receipt.host_bytes_authenticated_before_copy);
    add_bool(receipt.device_readback_authenticated);
    add_bool(receipt.allocation_retained_for_launch);
  };
  const auto add_norm_kernel = [&](const auto& kernel) noexcept {
    add_i32(kernel.registers_per_thread);
    add_size(kernel.static_shared_bytes);
    add_size(kernel.local_bytes);
    add_i32(kernel.maximum_threads_per_block);
    add_i32(kernel.active_blocks_per_sm);
    add_i32(kernel.threads_per_block);
    add_i32(kernel.physical_grid_ctas);
  };
  const auto add_norm_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_norm_kernel(resources.input_norm);
    add_norm_kernel(resources.fused_residual_norm);
    add_bool(resources.kernels_compiled);
    add_bool(resources.exact_geometry);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
    add_bool(resources.startup_package_unbound);
    add_bool(resources.execution_capability);
    add_bool(resources.caller_snapshot_grants_production_authority);
  };
  const auto add_fp8_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add(static_cast<std::uint64_t>(resources.role));
    add(static_cast<std::uint64_t>(resources.input_layout));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_i32(resources.registers_per_thread);
    add_size(resources.static_shared_bytes);
    add_size(resources.dynamic_shared_bytes);
    add_size(resources.local_bytes);
    add_i32(resources.maximum_threads_per_block);
    add_i32(resources.active_blocks_per_sm);
    add_size(resources.shared_bytes_per_sm);
    add_size(resources.optin_shared_bytes_per_block);
    add_bool(resources.kernel_compiled);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
  };
  const auto add_preprocess_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_i32(resources.kernel.registers_per_thread);
    add_size(resources.kernel.static_shared_bytes);
    add_size(resources.kernel.local_bytes);
    add_i32(resources.kernel.maximum_threads_per_block);
    add_i32(resources.kernel.active_blocks_per_sm);
    add_i32(resources.kernel.threads_per_block);
    add_i32(resources.kernel.grid_x);
    add_i32(resources.kernel.grid_y);
    add_i32(resources.kernel.physical_grid_ctas);
    add_bool(resources.kernel_compiled);
    add_bool(resources.exact_geometry);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
    add_bool(resources.startup_package_unbound);
    add_bool(resources.execution_capability);
    add_bool(resources.caller_snapshot_grants_production_authority);
  };
  const auto add_attention_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_i32(resources.kernel.registers_per_thread);
    add_size(resources.kernel.static_shared_bytes);
    add_size(resources.kernel.dynamic_shared_bytes);
    add_size(resources.kernel.local_bytes);
    add_i32(resources.kernel.maximum_threads_per_block);
    add_i32(resources.kernel.active_blocks_per_sm);
    add_i32(resources.kernel.threads_per_block);
    add_i32(resources.kernel.grid_x);
    add_i32(resources.kernel.grid_y);
    add_i32(resources.kernel.grid_z);
    add_i32(resources.kernel.physical_grid_ctas);
    add_bool(resources.kernel_compiled);
    add_bool(resources.exact_geometry);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
    add_bool(resources.startup_package_unbound);
    add_bool(resources.execution_capability);
    add_bool(resources.caller_snapshot_grants_production_authority);
  };
  const auto add_gate_up_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_i32(resources.registers_per_thread);
    add_size(resources.static_shared_bytes);
    add_size(resources.dynamic_shared_bytes);
    add_size(resources.local_bytes);
    add_i32(resources.maximum_threads_per_block);
    add_i32(resources.active_blocks_per_sm);
    add_bool(resources.kernel_compiled);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
  };
  const auto add_down_resources = [&](const auto& resources) noexcept {
    add(static_cast<std::uint64_t>(resources.identity));
    add_i32(resources.device_ordinal);
    add_i32(resources.compute_major);
    add_i32(resources.compute_minor);
    add_i32(resources.sm_count);
    add_i32(resources.binary_version);
    add_i32(resources.registers_per_thread);
    add_size(resources.static_shared_bytes);
    add_size(resources.dynamic_shared_bytes);
    add_size(resources.local_bytes);
    add_i32(resources.maximum_threads_per_block);
    add_i32(resources.active_blocks_per_sm);
    add_size(resources.shared_bytes_per_sm);
    add_size(resources.optin_shared_bytes_per_block);
    add_bool(resources.kernel_compiled);
    add_bool(resources.static_resource_gate_passed);
    add_bool(resources.numerical_contract_qualified);
    add_bool(resources.production_dispatch_eligible);
  };

  add(static_cast<std::uint64_t>(submission.authority_domain));
  add(submission.execution_package_identity);
  add(submission.full_attention_catalog_identity);
  add(submission.full_attention_binding_identity);
  add(submission.mlp_binding_identity);
  add(submission.input_norm_binding_identity);
  add(submission.post_norm_binding_identity);
  add(submission.rope_binding_identity);
  add(submission.resource_bundle_identity);
  add(submission.synthetic_source_identity);
  add_size(submission.full_attention_ordinal);
  add_size(submission.model_layer);

  add(1U);
  add_pointer(submission.input_norm.input_hidden);
  add_pointer(submission.input_norm.centered_weight);
  add_pointer(submission.input_norm.output_hidden);
  add_size(submission.input_norm.token_count);
  add_size(submission.input_norm.hidden_size);
  add(submission.input_norm.epsilon_fp32_bits);
  add_pointer(submission.input_norm.cuda_stream);

  add(2U);
  add_pointer(submission.full_qkv.hidden_input);
  add_fp8_asset(submission.full_qkv.asset);
  add_pointer(submission.full_qkv.q_gate_scratch);
  add_pointer(submission.full_qkv.key_panel_output);
  add_pointer(submission.full_qkv.value_panel_output);

  add(3U);
  add_pointer(submission.preprocess.q_gate_scratch);
  add_pointer(submission.preprocess.key_cache_origin);
  add_pointer(submission.preprocess.q_norm_weight);
  add_pointer(submission.preprocess.k_norm_weight);
  add_pointer(submission.preprocess.cosines);
  add_pointer(submission.preprocess.sines);
  add_size(submission.preprocess.first_position);

  add(4U);
  add_pointer(submission.attention.q_gate_scratch);
  add_pointer(submission.attention.key_cache_origin);
  add_pointer(submission.attention.value_cache_origin);
  add_size(submission.attention.first_position);

  add(5U);
  add_pointer(submission.full_output.q_gate_scratch);
  add_fp8_asset(submission.full_output.asset);
  add_pointer(submission.full_output.branch_output);

  add(6U);
  add_pointer(submission.residual_post_norm.left_residual_then_normalized);
  add_pointer(submission.residual_post_norm.right_branch_then_residual);
  add_pointer(submission.residual_post_norm.centered_weight);

  add(7U);
  add_pointer(submission.gate_up.normalized_input);
  add_pointer(submission.gate_up.payload);
  add_size(submission.gate_up.payload_bytes);
  add_float(submission.gate_up.gate_tensor_scale);
  add_float(submission.gate_up.up_tensor_scale);
  add_pointer(submission.gate_up.intermediate_output);
  add_gate_up_receipt(submission.gate_up.canonical_v3_payload_receipt);

  add(8U);
  add_pointer(submission.down.intermediate_input);
  add_pointer(submission.down.payload);
  add_size(submission.down.payload_bytes);
  add_float(submission.down.tensor_scale);
  add_pointer(submission.down.residual_output);
  add_down_receipt(submission.down.payload_receipt);

  add(9U);
  add_norm_resources(submission.norm_resources);
  add_fp8_resources(submission.full_qkv_resources);
  add_preprocess_resources(submission.preprocess_resources);
  add_attention_resources(submission.attention_resources);
  add_fp8_resources(submission.full_output_resources);
  add_gate_up_resources(submission.gate_up_resources);
  add_down_resources(submission.down_resources);
  return value == 0U ? 0x5133'4d46'5634'5344ULL : value;
}

std::uint64_t
Sm87MacroFeedV4ExecutionEventsOwner::full_attention_receipt_authenticator(
    const Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt& receipt)
    const noexcept {
  std::uint64_t value =
      mix64(receipt_secret_ ^ 0x5133'4d46'5634'4641ULL);
  value = mix64(value ^ receipt.transaction_identity_);
  value = mix64(value ^ receipt.owner_identity_);
  value = mix64(value ^ receipt.request_epoch_);
  value = mix64(value ^ static_cast<std::uint64_t>(receipt.panel_));
  value = mix64(value ^ receipt.panel_generation_);
  value = mix64(value ^ static_cast<std::uint64_t>(receipt.first_position_));
  value = mix64(value ^ receipt.grant_identity_);
  value = mix64(value ^ receipt.grant_state_epoch_);
  value = mix64(value ^ receipt.kv_allocation_identity_);
  value = mix64(value ^ receipt.key_full_allocation_origin_);
  value = mix64(value ^ receipt.value_full_allocation_origin_);
  value = mix64(value ^ receipt.key_panel_allocation_offset_);
  value = mix64(value ^ receipt.value_panel_allocation_offset_);
  value = mix64(value ^ receipt.kv_panel_bytes_);
  value = mix64(
      value ^ static_cast<std::uint64_t>(receipt.previous_valid_end_));
  value =
      mix64(value ^ static_cast<std::uint64_t>(receipt.candidate_end_));
  value = mix64(value ^
                static_cast<std::uint64_t>(receipt.full_attention_ordinal_));
  value =
      mix64(value ^ static_cast<std::uint64_t>(receipt.model_layer_));
  value = mix64(value ^
                static_cast<std::uint64_t>(receipt.authority_domain_));
  value = mix64(value ^ receipt.execution_package_identity_);
  value = mix64(value ^ receipt.full_attention_catalog_identity_);
  value = mix64(value ^ receipt.full_attention_binding_identity_);
  value = mix64(value ^ receipt.mlp_binding_identity_);
  value = mix64(value ^ receipt.input_norm_binding_identity_);
  value = mix64(value ^ receipt.post_norm_binding_identity_);
  value = mix64(value ^ receipt.rope_binding_identity_);
  value = mix64(value ^ receipt.resource_bundle_identity_);
  value = mix64(value ^ receipt.synthetic_source_identity_);
  value = mix64(value ^ receipt.submission_digest_);
  value = mix64(
      value ^ static_cast<std::uint64_t>(receipt.input_norm_launches_));
  value = mix64(
      value ^ static_cast<std::uint64_t>(receipt.full_qkv_launches_));
  value = mix64(
      value ^ static_cast<std::uint64_t>(receipt.preprocess_launches_));
  value = mix64(
      value ^ static_cast<std::uint64_t>(receipt.attention_launches_));
  value = mix64(
      value ^ static_cast<std::uint64_t>(receipt.full_output_launches_));
  value = mix64(value ^ static_cast<std::uint64_t>(
                            receipt.residual_post_norm_launches_));
  value = mix64(
      value ^ static_cast<std::uint64_t>(receipt.gate_up_launches_));
  value = mix64(
      value ^ static_cast<std::uint64_t>(receipt.down_launches_));
  value = mix64(value ^ static_cast<std::uint64_t>(
                            receipt.bound_kernel_submissions_));
  value = mix64(value ^ static_cast<std::uint64_t>(
                            receipt.asynchronous_d2d_copies_));
  value = mix64(value ^ receipt.asynchronous_d2d_copy_bytes_);
  value = mix64(value ^ static_cast<std::uint64_t>(
                            receipt.complete_layer_enqueued_));
  value = mix64(value ^ static_cast<std::uint64_t>(
                            receipt.physical_device_completion_attested_));
  value = mix64(
      value ^ static_cast<std::uint64_t>(receipt.panel_complete_));
  return mix64(value ^ static_cast<std::uint64_t>(
                           receipt.production_receipt_eligible_));
}

bool Sm87MacroFeedV4ExecutionEventsOwner::
    full_attention_receipt_matches_locked(
        const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
        const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
        const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
            expected_submission,
        const Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt&
            receipt) const noexcept {
  const bool valid_grant_slot =
      active_panel_ < kSm87MacroFeedV4PanelCount &&
      expected_submission.full_attention_ordinal <
          kSm87MacroFeedV4FullAttentionLayerCount;
  const std::size_t grant_slot =
      valid_grant_slot
          ? active_panel_ * kSm87MacroFeedV4FullAttentionLayerCount +
                expected_submission.full_attention_ordinal
          : accepted_full_attention_grant_identities_.size();
  return valid_grant_slot &&
         accepted_full_attention_grant_identities_[grant_slot] ==
             kv_grant.grant_identity() &&
         panel_access_matches(panel_access) && receipt.valid_shape() &&
         receipt.owner_identity_ == owner_identity_ &&
         receipt.request_epoch_ == request_epoch_ &&
         receipt.panel_ == active_panel_ &&
         receipt.panel_generation_ == active_panel_generation_ &&
         kv_grant.owner_identity() == owner_identity_ &&
         kv_grant.request_epoch() == request_epoch_ &&
         kv_grant.panel() == active_panel_ &&
         receipt.grant_identity_ == kv_grant.grant_identity() &&
         receipt.grant_state_epoch_ == kv_grant.state_epoch() &&
         receipt.kv_allocation_identity_ ==
             kv_grant.kv_allocation_identity() &&
         receipt.key_full_allocation_origin_ ==
             kv_grant.key_full_allocation_origin() &&
         receipt.value_full_allocation_origin_ ==
             kv_grant.value_full_allocation_origin() &&
         receipt.key_panel_allocation_offset_ ==
             kv_grant.key_panel_allocation_offset() &&
         receipt.value_panel_allocation_offset_ ==
             kv_grant.value_panel_allocation_offset() &&
         receipt.kv_panel_bytes_ == kv_grant.panel_bytes() &&
         receipt.first_position_ == kv_grant.first_position() &&
         receipt.previous_valid_end_ == kv_grant.previous_valid_end() &&
         receipt.candidate_end_ == kv_grant.candidate_end() &&
         receipt.full_attention_ordinal_ ==
             kv_grant.attention_layer_ordinal() &&
         receipt.model_layer_ == kv_grant.model_layer() &&
         receipt.full_attention_ordinal_ ==
             expected_submission.full_attention_ordinal &&
         receipt.model_layer_ == expected_submission.model_layer &&
         receipt.authority_domain_ == expected_submission.authority_domain &&
         receipt.execution_package_identity_ ==
             expected_submission.execution_package_identity &&
         receipt.full_attention_catalog_identity_ ==
             expected_submission.full_attention_catalog_identity &&
         receipt.full_attention_binding_identity_ ==
             expected_submission.full_attention_binding_identity &&
         receipt.mlp_binding_identity_ ==
             expected_submission.mlp_binding_identity &&
         receipt.input_norm_binding_identity_ ==
             expected_submission.input_norm_binding_identity &&
         receipt.post_norm_binding_identity_ ==
             expected_submission.post_norm_binding_identity &&
         receipt.rope_binding_identity_ ==
             expected_submission.rope_binding_identity &&
         receipt.resource_bundle_identity_ ==
             expected_submission.resource_bundle_identity &&
         receipt.synthetic_source_identity_ ==
             expected_submission.synthetic_source_identity &&
         receipt.submission_digest_ ==
             full_attention_submission_digest(expected_submission) &&
         receipt.authenticator_ != 0U &&
         receipt.authenticator_ ==
             full_attention_receipt_authenticator(receipt);
}

bool Sm87MacroFeedV4ExecutionEventsOwner::full_attention_receipt_matches(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
    const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
        expected_submission,
    const Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt& receipt)
    const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return owner_access_matches(access) &&
         state_ == Sm87MacroFeedV4ExecutionOwnerState::kRequestActive &&
         full_attention_receipt_matches_locked(
             panel_access, kv_grant, expected_submission, receipt);
}
#endif

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
Sm87MacroFeedV4ExecutionEventsOwner::discard_request_state_after_drain(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained,
    Sm87MacroFeedV4RequestState& request_owner,
    const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
    const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
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
                "combined_discard_requires_dual_tail_owner_drain", 0,
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
                "combined_discard_requires_exact_physical_drain_receipt", 0,
                Sm87MacroFeedV4ExecutionStream::kControl,
                Sm87MacroFeedV4ExecutionEvent::kOwnerDrained, active_panel_,
                active_panel_generation_);
  }

  // Lock ordering is EventsOwner -> RequestState, identical to begin_request().
  // RequestState is terminalized before this owner retires the identities that
  // authenticate the physical observation.
  const auto request_discard =
      request_owner.discard_active_panel_after_physical_execution_drain(
          request_access, owner_identity_, request_allocation_identity_,
          request_epoch_, active_panel_, active_panel_generation_,
          owner_drained.receipt_identity_, false, reason);
  if (!request_discard) {
    const auto failure = fail(
        Sm87MacroFeedV4ExecutionError::kRequestStateDiscard,
        request_discard.context, 0, Sm87MacroFeedV4ExecutionStream::kControl,
        Sm87MacroFeedV4ExecutionEvent::kOwnerDrained, active_panel_,
        active_panel_generation_);
    // Do not leave an owner whose physical and logical ledgers disagree in the
    // reusable RequestActive state.  Preserve the live request/panel identities
    // and poison it so the package can perform the terminal all-stream drain
    // and retry the same owner-mediated RequestState discard exactly once.
    record_poison_cause(failure);
    return failure;
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
  return drain_poisoned_request_locked(
      access, nullptr, nullptr, Sm87MacroFeedV4RequestDiscardReason::kInvalid);
}

Sm87MacroFeedV4PoisonDrainResult
Sm87MacroFeedV4ExecutionEventsOwner::drain_poisoned_request_and_discard(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    Sm87MacroFeedV4RequestState& request_owner,
    const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
    const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return drain_poisoned_request_locked(access, &request_owner, &request_access,
                                       reason);
}

Sm87MacroFeedV4PoisonDrainResult
Sm87MacroFeedV4ExecutionEventsOwner::drain_poisoned_request_locked(
    const Sm87MacroFeedV4ExecutionEventsAccess& access,
    Sm87MacroFeedV4RequestState* const request_owner,
    const Sm87MacroFeedV4RequestStateSealedAccess* const request_access,
    const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
  Sm87MacroFeedV4PoisonDrainResult result;
  result.poison_cause = poison_cause_;
  if (!owner_access_matches(access)) {
    result.drain_status =
        fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
             "poison_drain_owner_issued_access");
    return result;
  }
  const bool combined_discard = request_owner != nullptr;
  if (combined_discard != (request_access != nullptr) ||
      (combined_discard &&
       (reason != Sm87MacroFeedV4RequestDiscardReason::kCancelled &&
        reason != Sm87MacroFeedV4RequestDiscardReason::kFailed))) {
    result.drain_status =
        fail(Sm87MacroFeedV4ExecutionError::kRequestStateDiscard,
             "poison_combined_discard_requires_exact_request_authority");
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
  if (combined_discard &&
      (request_allocation_identity_ == 0U || request_epoch_ == 0U ||
       active_panel_ >= kSm87MacroFeedV4PanelCount ||
       active_panel_generation_ == 0U)) {
    result.drain_status = fail(
        Sm87MacroFeedV4ExecutionError::kRequestStateDiscard,
        "poison_combined_discard_requires_live_panel_generation", 0,
        Sm87MacroFeedV4ExecutionStream::kCount,
        Sm87MacroFeedV4ExecutionEvent::kCount, active_panel_,
        active_panel_generation_);
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
  poisoned_terminal_quiescence_identity_ =
      next_nonzero(&g_next_poison_quiescence_identity);
  result.physical_quiescence_attested = true;
  result.quiescence_identity = poisoned_terminal_quiescence_identity_;
  result.discard_required = true;

  if (combined_discard) {
    // The audit identity above is never consumed as caller authority.  This
    // EventsOwner still holds its exact poisoned request/panel generation and
    // directly performs the only RequestState transition that may consume it.
    result.request_state_status =
        request_owner->discard_active_panel_after_physical_execution_drain(
            *request_access, owner_identity_, request_allocation_identity_,
            request_epoch_, active_panel_, active_panel_generation_,
            poisoned_terminal_quiescence_identity_, true, reason);
    if (!result.request_state_status) {
      if (result.drain_status.error ==
          Sm87MacroFeedV4ExecutionError::kNone) {
        result.drain_status = fail(
            Sm87MacroFeedV4ExecutionError::kRequestStateDiscard,
            result.request_state_status.context, 0,
            Sm87MacroFeedV4ExecutionStream::kCount,
            Sm87MacroFeedV4ExecutionEvent::kCount, active_panel_,
            active_panel_generation_);
      }
      // Physical quiescence is an immutable fact, but retain the live logical
      // identities for diagnosis when RequestState rejects the terminal
      // transition.  The poisoned owner remains permanently non-reusable.
      return result;
    }
    result.request_state_discarded = true;
  }
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
  snapshot.bound_kernel_submissions = bound_kernel_submissions_;
  snapshot.input_norm_submissions = input_norm_submissions_;
  snapshot.bf16_ab_submissions = bf16_ab_submissions_;
  snapshot.gdn_qkvz_c8000_submissions = gdn_qkvz_c8000_submissions_;
  snapshot.gdn_qkvz_ab_ready_wait_transactions =
      gdn_qkvz_ab_ready_wait_transactions_;
  snapshot.gdn_continuation_c8000_submissions =
      gdn_continuation_c8000_submissions_;
  snapshot.gdn_history_d2d_copies = gdn_history_d2d_copies_;
  snapshot.gdn_history_d2d_bytes = gdn_history_d2d_bytes_;
  snapshot.gdn_output_c8000_submissions = gdn_output_c8000_submissions_;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  snapshot.full_qkv_c8000_submissions = full_qkv_c8000_submissions_;
  snapshot.full_attention_preprocess_c8000_submissions =
      full_attention_preprocess_c8000_submissions_;
  snapshot.attention_c8000_submissions = attention_c8000_submissions_;
  snapshot.full_attention_output_c8000_submissions =
      full_attention_output_c8000_submissions_;
#endif
  snapshot.residual_post_norm_submissions =
      residual_post_norm_submissions_;
  snapshot.gate_up_c8000_submissions = gate_up_c8000_submissions_;
  snapshot.down_c8000_submissions = down_c8000_submissions_;
  snapshot.complete_gdn_layers_submitted = complete_gdn_layers_submitted_;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  snapshot.accepted_gdn_grants = accepted_gdn_grant_count_;
  snapshot.last_gdn_accepted_prefix = last_gdn_accepted_prefix_;
  snapshot.complete_full_attention_layers_submitted =
      complete_full_attention_layers_submitted_;
  snapshot.accepted_full_attention_grants =
      accepted_full_attention_grant_count_;
  snapshot.last_full_attention_accepted_prefix =
      last_full_attention_accepted_prefix_;
#endif
  snapshot.cold_recurrent_initializations = cold_recurrent_initializations_;
  snapshot.cold_recurrent_allocation_identity =
      cold_recurrent_allocation_identity_;
  snapshot.cold_recurrent_allocation_begin =
      cold_recurrent_allocation_begin_;
  snapshot.cold_recurrent_zero_bytes = cold_recurrent_zero_bytes_;
  snapshot.streams_nonblocking = streams_nonblocking_;
  snapshot.bf16_ab_cycle_at_norm_boundary =
      ab_cycle_phase_ == AbCyclePhase::kExpectNormRecord;
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
  snapshot.poisoned_terminal_quiescence_identity =
      poisoned_terminal_quiescence_identity_;
  return snapshot;
}

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
std::uint64_t Sm87MacroFeedV4ExecutionEventsDriver::owner_identity()
    const noexcept {
  return owner_ != nullptr && access_ != nullptr ? access_->owner_identity()
                                                  : 0U;
}

std::int32_t Sm87MacroFeedV4ExecutionEventsDriver::device_ordinal()
    const noexcept {
  return owner_ != nullptr && access_ != nullptr ? access_->device_ordinal()
                                                  : -1;
}

Sm87MacroFeedV4ExecutionEventsSnapshot
Sm87MacroFeedV4ExecutionEventsDriver::snapshot() const noexcept {
  return owner_ == nullptr ? Sm87MacroFeedV4ExecutionEventsSnapshot{}
                           : owner_->snapshot();
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsDriver::begin_request(
    const Sm87MacroFeedV4RequestState& request_owner,
    const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept {
  return owner_ == nullptr || access_ == nullptr
             ? fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                    "execution_driver_unbound")
             : owner_->begin_request(*access_, request_owner, request_access);
}

Sm87MacroFeedV4PanelBeginResult
Sm87MacroFeedV4ExecutionEventsDriver::begin_panel(
    const std::size_t panel) noexcept {
  if (owner_ == nullptr || access_ == nullptr) {
    Sm87MacroFeedV4PanelBeginResult result;
    result.status = fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                         "execution_driver_unbound");
    return result;
  }
  return owner_->begin_panel(*access_, panel);
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsDriver::record_event(
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionStream producer,
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  if (owner_ == nullptr || access_ == nullptr) {
    Sm87MacroFeedV4EventEnqueueResult result;
    result.status = fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                         "execution_driver_unbound");
    return result;
  }
  if (event == Sm87MacroFeedV4ExecutionEvent::kNormReady ||
      event == Sm87MacroFeedV4ExecutionEvent::kAbReady) {
    Sm87MacroFeedV4EventEnqueueResult result;
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
        "driver_ready_event_requires_bound_kernel_transaction", 0,
        producer, event);
    return result;
  }
  return owner_->record_event(*access_, panel_access, producer, event);
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsDriver::wait_event(
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionStream consumer,
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  if (owner_ == nullptr || access_ == nullptr) {
    Sm87MacroFeedV4EventEnqueueResult result;
    result.status = fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                         "execution_driver_unbound");
    return result;
  }
  if (consumer == Sm87MacroFeedV4ExecutionStream::kMain &&
      event == Sm87MacroFeedV4ExecutionEvent::kAbReady) {
    Sm87MacroFeedV4EventEnqueueResult result;
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kKernelSubmitContract,
        "driver_ab_ready_wait_requires_fixed_gdn_qkvz_transaction", 0,
        consumer, event);
    return result;
  }
  return owner_->wait_event(*access_, panel_access, consumer, event);
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsDriver::submit_input_norm_and_record_ready(
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const kernels::Sm87MacroFeedV4InputNormArguments& arguments,
    const kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&
        resources) noexcept {
  if (owner_ == nullptr || access_ == nullptr) {
    Sm87MacroFeedV4EventEnqueueResult result;
    result.status = fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                         "execution_driver_unbound");
    return result;
  }
  return owner_->submit_input_norm_and_record_ready(
      *access_, panel_access, arguments, resources);
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsDriver::submit_bf16_ab_and_record_ready(
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const kernels::Sm87MacroFeedV4Bf16AbArguments& arguments,
    const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
        resources) noexcept {
  if (owner_ == nullptr || access_ == nullptr) {
    Sm87MacroFeedV4EventEnqueueResult result;
    result.status = fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                         "execution_driver_unbound");
    return result;
  }
  return owner_->submit_bf16_ab_and_record_ready(
      *access_, panel_access, arguments, resources);
}

Sm87MacroFeedV4EventEnqueueResult
Sm87MacroFeedV4ExecutionEventsDriver::
    submit_gdn_qkvz_c8000_then_wait_ab_ready(
        const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
        const kernels::sm87_macrofeed_v4_bound_launch_detail::
            Sm87MacroFeedV4GdnQkvzC8000Arguments& arguments,
        const kernels::Sm87MacroFeedV4Fp8CudaResources& resources) noexcept {
  if (owner_ == nullptr || access_ == nullptr) {
    Sm87MacroFeedV4EventEnqueueResult result;
    result.status = fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                         "execution_driver_unbound");
    return result;
  }
  return owner_->submit_gdn_qkvz_c8000_then_wait_ab_ready(
      *access_, panel_access, arguments, resources);
}

Sm87MacroFeedV4CompleteGdnLayerEnqueueResult
Sm87MacroFeedV4ExecutionEventsDriver::
    submit_complete_gdn_layer_c8000_prevalidated(
        const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
        const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
        const Sm87MacroFeedV4CompleteGdnLayerC8000Submission& submission)
        noexcept {
  if (owner_ == nullptr || access_ == nullptr) {
    Sm87MacroFeedV4CompleteGdnLayerEnqueueResult result;
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
        "execution_driver_unbound");
    return result;
  }
  return owner_->submit_complete_gdn_layer_c8000_prevalidated(
      *access_, panel_access, gdn_grant, submission);
}

bool Sm87MacroFeedV4ExecutionEventsDriver::gdn_receipt_matches(
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
    const Sm87MacroFeedV4CompleteGdnLayerC8000Submission& expected_submission,
    const Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt& receipt)
    const noexcept {
  return owner_ != nullptr && access_ != nullptr &&
         owner_->gdn_receipt_matches(*access_, panel_access, gdn_grant,
                                     expected_submission, receipt);
}

Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueResult
Sm87MacroFeedV4ExecutionEventsDriver::
    submit_complete_full_attention_layer_c8000_prevalidated(
        const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
        const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
        const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
            submission) noexcept {
  if (owner_ == nullptr || access_ == nullptr) {
    Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueResult result;
    result.status = fail(
        Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
        "execution_driver_unbound");
    return result;
  }
  return owner_->submit_complete_full_attention_layer_c8000_prevalidated(
      *access_, panel_access, kv_grant, submission);
}

bool Sm87MacroFeedV4ExecutionEventsDriver::full_attention_receipt_matches(
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
    const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
        expected_submission,
    const Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt& receipt)
    const noexcept {
  return owner_ != nullptr && access_ != nullptr &&
         owner_->full_attention_receipt_matches(*access_, panel_access,
                                                kv_grant,
                                                expected_submission,
                                                receipt);
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsDriver::initialize_cold_recurrent_storage(
    void* const recurrent_allocation,
    const std::size_t recurrent_bytes,
    const std::uint64_t recurrent_allocation_identity) noexcept {
  return owner_ == nullptr || access_ == nullptr
             ? fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                    "execution_driver_unbound")
             : owner_->initialize_cold_recurrent_storage(
                   *access_, recurrent_allocation, recurrent_bytes,
                   recurrent_allocation_identity);
}

Sm87MacroFeedV4PhysicalObservationResult
Sm87MacroFeedV4ExecutionEventsDriver::observe_event_synchronize(
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionEvent event) noexcept {
  if (owner_ == nullptr || access_ == nullptr) {
    Sm87MacroFeedV4PhysicalObservationResult result;
    result.status = fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                         "execution_driver_unbound");
    return result;
  }
  return owner_->observe_event_synchronize(*access_, panel_access, event);
}

bool Sm87MacroFeedV4ExecutionEventsDriver::completion_receipt_matches(
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4ExecutionEvent expected_event,
    const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const noexcept {
  return owner_ != nullptr && access_ != nullptr &&
         owner_->completion_receipt_matches(*access_, panel_access,
                                            expected_event, receipt);
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsDriver::discard_after_drain(
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained) noexcept {
  return owner_ == nullptr || access_ == nullptr
             ? fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                    "execution_driver_unbound")
             : owner_->discard_after_drain(*access_, panel_access,
                                           owner_drained);
}

Sm87MacroFeedV4ExecutionStatus
Sm87MacroFeedV4ExecutionEventsDriver::discard_request_state_after_drain(
    const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained,
    Sm87MacroFeedV4RequestState& request_owner,
    const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
    const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
  return owner_ == nullptr || access_ == nullptr
             ? fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
                    "execution_driver_unbound")
             : owner_->discard_request_state_after_drain(
                   *access_, panel_access, owner_drained, request_owner,
                   request_access, reason);
}

Sm87MacroFeedV4PoisonDrainResult
Sm87MacroFeedV4ExecutionEventsDriver::drain_poisoned_request() noexcept {
  if (owner_ == nullptr || access_ == nullptr) {
    Sm87MacroFeedV4PoisonDrainResult result;
    result.drain_status =
        fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
             "execution_driver_unbound");
    return result;
  }
  return owner_->drain_poisoned_request(*access_);
}

Sm87MacroFeedV4PoisonDrainResult
Sm87MacroFeedV4ExecutionEventsDriver::drain_poisoned_request_and_discard(
    Sm87MacroFeedV4RequestState& request_owner,
    const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
    const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
  if (owner_ == nullptr || access_ == nullptr) {
    Sm87MacroFeedV4PoisonDrainResult result;
    result.drain_status =
        fail(Sm87MacroFeedV4ExecutionError::kForeignOwnerAccess,
             "execution_driver_unbound");
    return result;
  }
  return owner_->drain_poisoned_request_and_discard(
      *access_, request_owner, request_access, reason);
}

std::unique_ptr<Sm87MacroFeedV4ExecutionEventsDriver>
bind_sm87_macrofeed_v4_execution_events_driver(
    const std::shared_ptr<Sm87MacroFeedV4ExecutionEventsOwner>& owner)
    noexcept {
  if (owner == nullptr) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(owner->mutex_);
  if (owner->state_ != Sm87MacroFeedV4ExecutionOwnerState::kReady ||
      owner->access_ == nullptr ||
      !owner->owner_access_matches(*owner->access_)) {
    return nullptr;
  }
  return std::unique_ptr<Sm87MacroFeedV4ExecutionEventsDriver>(
      new (std::nothrow) Sm87MacroFeedV4ExecutionEventsDriver(
          owner, owner->access_.get()));
}
#endif

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
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  accepted_gdn_grant_identities_.fill(0U);
  accepted_gdn_grant_count_ = 0U;
  last_gdn_accepted_prefix_ = {};
  test_fail_gdn_after_accepted_operation_ =
      std::numeric_limits<std::size_t>::max();
  accepted_full_attention_grant_identities_.fill(0U);
  accepted_full_attention_grant_count_ = 0U;
  last_full_attention_accepted_prefix_ = {};
  test_fail_full_after_accepted_prefix_ =
      std::numeric_limits<std::size_t>::max();
#endif
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
