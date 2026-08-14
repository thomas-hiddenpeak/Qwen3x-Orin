#include "sm87_macrofeed_v4_p40_execution_package_internal.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
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

}  // namespace

Sm87MacroFeedV4P40ExecutionPackage::Sm87MacroFeedV4P40ExecutionPackage(
    ProjectionCatalog projection_catalog, Bf16AbCatalog bf16_ab_catalog,
    LayerNormCatalog layer_norm_catalog,
    kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot
        norm_resources,
    kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot
        bf16_ab_resources,
    void* const transient_allocation, void* const recurrent_allocation,
    std::unique_ptr<Sm87MacroFeedV4RequestState> request_state,
    std::shared_ptr<EventsOwner> events_owner,
    std::unique_ptr<EventsDriver> events_driver,
    Sm87MacroFeedV4P40ExecutionPackageAudit audit) noexcept
    : projection_catalog_(std::move(projection_catalog)),
      bf16_ab_catalog_(std::move(bf16_ab_catalog)),
      layer_norm_catalog_(std::move(layer_norm_catalog)),
      norm_resources_(norm_resources),
      bf16_ab_resources_(bf16_ab_resources),
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

Sm87MacroFeedV4P40ExecutionPackageCreateResult::operator bool()
    const noexcept {
  return package != nullptr && static_cast<bool>(status) && audit.valid() &&
         package->valid() &&
         package->audit().package_identity == audit.package_identity;
}

Sm87MacroFeedV4P40ExecutionPackageCreateResult
Sm87MacroFeedV4P40ExecutionPackage::create(
    const StartupPackage& startup_package) noexcept {
  Sm87MacroFeedV4P40ExecutionPackageCreateResult result;
#if !defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_EXECUTION_EVENTS_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_NORM_RESIDUAL_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION)
  (void)startup_package;
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
  // if it escapes before the future Engine composition root is introduced.
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
  audit.transient_allocation_identity = transient_identity;
  audit.recurrent_allocation_identity = recurrent_identity;
  audit.execution_events_owner_identity = events_owner_identity;
  audit.device_ordinal = startup_package.audit().device_ordinal;
  audit.projection_bindings = projection_catalog.size();
  audit.bf16_ab_pairs = bf16_ab_catalog.size();
  audit.layer_norm_pairs = layer_norm_catalog.size();
  audit.transient_bytes = kSm87MacroFeedV4P40ExecutionTransientBytes;
  audit.recurrent_bytes = kSm87MacroFeedV4RecurrentStorageBytes;
  audit.fixed_gdn_layer0_front_half_bound = true;
  audit.whole_layer_executor_bound = false;
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
  package_identity = mix64(package_identity ^ audit.transient_allocation_identity);
  package_identity = mix64(package_identity ^ audit.recurrent_allocation_identity);
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
      std::move(layer_norm_catalog), norm_resources, bf16_ab_resources,
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
      bf16_ab_catalog_.empty() || layer_norm_catalog_.empty()) {
    return false;
  }
  const auto& norm = layer_norm_catalog_[0U];
  const auto& ab = bf16_ab_catalog_[0U];
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
      !front_half_bindings_valid()) {
    return false;
  }
  for (const auto& binding : projection_catalog_) {
    if (!binding.has_value() || binding->binding_identity() == 0U ||
        binding->package_identity() != audit_.startup_package_identity) {
      return false;
    }
  }
  return true;
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
    const events::Sm87MacroFeedV4ExecutionStatus& event_status) noexcept {
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

  const auto poison = events_driver_->drain_poisoned_request();
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
  return failure(Error::kExecutionEvent, context, original.cuda_error,
                 original.panel, false, original);
}

Sm87MacroFeedV4P40ExecutionPackageStatus
Sm87MacroFeedV4P40ExecutionPackage::drain_and_discard_active_panel(
    const PanelAccess& panel_access) noexcept {
  if (events_owner_ == nullptr || events_driver_ == nullptr) {
    return failure(Error::kPhysicalDrain, "missing_execution_event_owner");
  }
  auto snapshot = events_driver_->snapshot();
  if (snapshot.state == events::Sm87MacroFeedV4ExecutionOwnerState::kPoisoned) {
    return terminalize_event_failure(
        "poisoned_request_before_tail_drain", snapshot.poison_cause);
  }

  auto enqueue = events_driver_->record_event(
      panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kMainTail);
  if (!enqueue) {
    return terminalize_event_failure("main_tail_record", enqueue.status);
  }
  enqueue = events_driver_->record_event(
      panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kAbAux,
      events::Sm87MacroFeedV4ExecutionEvent::kAbTail);
  if (!enqueue) {
    return terminalize_event_failure("ab_tail_record", enqueue.status);
  }
  enqueue = events_driver_->wait_event(
      panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kMainTail);
  if (!enqueue) {
    return terminalize_event_failure("main_tail_control_join",
                                     enqueue.status);
  }
  enqueue = events_driver_->wait_event(
      panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kAbTail);
  if (!enqueue) {
    return terminalize_event_failure("ab_tail_control_join",
                                     enqueue.status);
  }
  enqueue = events_driver_->record_event(
      panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kOwnerDrained);
  if (!enqueue) {
    return terminalize_event_failure("owner_drained_record",
                                     enqueue.status);
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
        "owner_drained_physical_observation", observed.status);
  }
  const auto discarded = events_driver_->discard_after_drain(
      panel_access, observed.receipt);
  if (!discarded) {
    return terminalize_event_failure("owner_discard_after_drain",
                                     discarded);
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

  // Main's AbReady wait is intentionally absent: no QKVZ projection has been
  // admitted in this slice, so closing the cycle would falsely attest the
  // QKVZ/A-B overlap point.  The dual-tail Control drain still proves both
  // submitted kernels are physically quiescent before discard.
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
  receipt.request_epoch = request_access.request_epoch();
  receipt.panel = 0U;
  receipt.model_layer = 0U;
  receipt.input_norm_launches = event_snapshot.input_norm_submissions;
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
  receipt.ab_ready_waited_by_main = false;
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
  norm_resources_ = {};
  bf16_ab_resources_ = {};
  audit_ = {};
  construction_postconditions_sealed_ = false;
}

}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail
