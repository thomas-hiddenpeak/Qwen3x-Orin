#include "sm87_bulk_dataflow_v2_p40_request_state_internal.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail {
namespace {

std::atomic<std::uint64_t> g_next_v2_request_state_identity{1U};

[[nodiscard]] std::uint64_t next_identity() noexcept {
  for (;;) {
    const std::uint64_t value = g_next_v2_request_state_identity.fetch_add(
        1U, std::memory_order_relaxed);
    if (value != 0U) {
      return value;
    }
  }
}

[[nodiscard]] constexpr Sm87BulkV2P40RequestStateStatus ok() noexcept {
  return {};
}

[[nodiscard]] constexpr Sm87BulkV2P40RequestStateStatus error(
    const Sm87BulkV2P40RequestStateError code, const char* const context,
    const int cuda_error = 0,
    const std::size_t resource_index =
        std::numeric_limits<std::size_t>::max()) noexcept {
  return {code, context, cuda_error, resource_index};
}

[[nodiscard]] constexpr std::size_t stream_index(
    const Sm87BulkV2P40Stream stream) noexcept {
  return static_cast<std::size_t>(stream);
}

[[nodiscard]] constexpr bool valid_stream(
    const Sm87BulkV2P40Stream stream) noexcept {
  return stream_index(stream) < kSm87BulkV2P40StreamCount;
}

[[nodiscard]] constexpr std::size_t binding_index(
    const Sm87BulkV2P40RequestArenaRole role) noexcept {
  switch (role) {
    case Sm87BulkV2P40RequestArenaRole::kPersistent:
      return 0U;
    case Sm87BulkV2P40RequestArenaRole::kGdnColdStateAndHistory:
      return 1U;
    case Sm87BulkV2P40RequestArenaRole::kResidual:
      return 2U;
    case Sm87BulkV2P40RequestArenaRole::kFamily:
      return 3U;
    case Sm87BulkV2P40RequestArenaRole::kFinalHidden:
      return 4U;
    case Sm87BulkV2P40RequestArenaRole::kInvalid:
      break;
  }
  return kSm87BulkV2P40RequestArenaBindingCount;
}

[[nodiscard]] constexpr Sm87BulkV2P40Range
final_greedy_workspace_family_range() noexcept {
  constexpr auto family = sm87_bulk_v2_p40_family_arena_plan();
  for (const auto& binding : family.bindings) {
    if (binding.role == Sm87BulkV2P40BufferRole::kFinalGreedyWorkspace) {
      return binding.range;
    }
  }
  return {};
}

[[nodiscard]] void* offset_pointer(void* const base,
                                   const std::uint64_t offset) noexcept {
  if (base == nullptr ||
      offset > std::numeric_limits<std::uintptr_t>::max() -
                   reinterpret_cast<std::uintptr_t>(base)) {
    return nullptr;
  }
  return reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(base) +
                                 static_cast<std::uintptr_t>(offset));
}

[[nodiscard]] bool identity_equal(
    const Sm87BulkV2P40RequestStateIdentity& left,
    const Sm87BulkV2P40RequestStateIdentity& right) noexcept {
  return sm87_bulk_v2_p40_magic_equal(left.plan_magic, right.plan_magic) &&
         left.abi_major == right.abi_major &&
         left.abi_minor == right.abi_minor &&
         left.owner_identity == right.owner_identity &&
         left.seal_nonce == right.seal_nonce &&
         left.allocation_identity == right.allocation_identity &&
         left.stream_event_owner_identity ==
             right.stream_event_owner_identity &&
         left.pinned_handoff_identity == right.pinned_handoff_identity &&
         left.device_ordinal == right.device_ordinal &&
         left.execution_class == right.execution_class &&
         left.one_exact_device_allocation ==
             right.one_exact_device_allocation &&
         left.control_plane_owned_elsewhere ==
             right.control_plane_owned_elsewhere &&
         left.whole_arena_reset_forbidden ==
             right.whole_arena_reset_forbidden &&
         left.production_dispatch_eligible ==
             right.production_dispatch_eligible;
}

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_REQUEST_STATE_ADMISSION)
class CudartSm87BulkV2P40RequestStateRuntime final
    : public Sm87BulkV2P40RequestStateCudaRuntime {
 public:
  [[nodiscard]] int get_current_device(
      std::int32_t* const device_ordinal) noexcept override {
    return static_cast<int>(cudaGetDevice(device_ordinal));
  }

  [[nodiscard]] int get_device_properties(
      const std::int32_t device_ordinal,
      Sm87BulkV2P40RequestDeviceProperties* const properties)
      noexcept override {
    if (properties == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaDeviceProp raw{};
    const cudaError_t status = cudaGetDeviceProperties(&raw, device_ordinal);
    if (status == cudaSuccess) {
      properties->major = raw.major;
      properties->minor = raw.minor;
      properties->multiprocessor_count = raw.multiProcessorCount;
    }
    return static_cast<int>(status);
  }

  [[nodiscard]] int get_stream_flags(
      void* const stream, unsigned int* const flags) noexcept override {
    return static_cast<int>(cudaStreamGetFlags(
        reinterpret_cast<cudaStream_t>(stream), flags));
  }

  [[nodiscard]] int allocate_device(void** const pointer,
                                    const std::size_t bytes) noexcept override {
    return static_cast<int>(cudaMalloc(pointer, bytes));
  }

  [[nodiscard]] int free_device(void* const pointer) noexcept override {
    return static_cast<int>(cudaFree(pointer));
  }

  [[nodiscard]] int allocate_pinned_host(
      void** const pointer, const std::size_t bytes) noexcept override {
    return static_cast<int>(
        cudaHostAlloc(pointer, bytes, cudaHostAllocPortable));
  }

  [[nodiscard]] int free_pinned_host(void* const pointer) noexcept override {
    return static_cast<int>(cudaFreeHost(pointer));
  }

  [[nodiscard]] int query_pointer(
      const void* const pointer,
      Sm87BulkV2P40RequestPointerAttributes* const attributes)
      noexcept override {
    if (attributes == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaPointerAttributes raw{};
    const cudaError_t status = cudaPointerGetAttributes(&raw, pointer);
    if (status == cudaSuccess) {
      attributes->kind =
          raw.type == cudaMemoryTypeHost
              ? Sm87BulkV2P40RequestPointerKind::kHost
              : (raw.type == cudaMemoryTypeDevice
                     ? Sm87BulkV2P40RequestPointerKind::kDevice
                     : Sm87BulkV2P40RequestPointerKind::kUnknown);
      attributes->host_pointer = raw.hostPointer;
      attributes->device_pointer = raw.devicePointer;
      attributes->device_ordinal = raw.device;
    }
    return static_cast<int>(status);
  }

  [[nodiscard]] int memset_async(void* const pointer, const int value,
                                 const std::size_t bytes,
                                 void* const stream) noexcept override {
    return static_cast<int>(cudaMemsetAsync(
        pointer, value, bytes, reinterpret_cast<cudaStream_t>(stream)));
  }

  [[nodiscard]] int copy_device_to_host_async(
      void* const host_destination, const void* const device_source,
      const std::size_t bytes, void* const stream) noexcept override {
    return static_cast<int>(cudaMemcpyAsync(
        host_destination, device_source, bytes, cudaMemcpyDeviceToHost,
        reinterpret_cast<cudaStream_t>(stream)));
  }

  [[nodiscard]] int synchronize_stream(
      void* const stream) noexcept override {
    return static_cast<int>(
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
  }
};
#endif

}  // namespace

bool Sm87BulkV2P40RequestStateIdentity::valid() const noexcept {
  const bool common =
      sm87_bulk_v2_p40_magic_equal(plan_magic,
                                   kSm87BulkV2P40PlanMagic) &&
      abi_major == kSm87BulkV2P40PlanAbiMajor &&
      abi_minor == kSm87BulkV2P40PlanAbiMinor && owner_identity != 0U &&
      seal_nonce != 0U && allocation_identity != 0U &&
      stream_event_owner_identity != 0U &&
      pinned_handoff_identity != 0U && device_ordinal >= 0 &&
      one_exact_device_allocation && control_plane_owned_elsewhere &&
      whole_arena_reset_forbidden && !production_dispatch_eligible;
  return common && (default_off_development_resource_valid() ||
                    synthetic_host_contract_valid());
}

bool Sm87BulkV2P40RequestStateIdentity::
    default_off_development_resource_valid() const noexcept {
  return execution_class == Sm87BulkV2P40RequestStateExecutionClass::
                                kDefaultOffDevelopmentResource &&
         !production_dispatch_eligible;
}

bool Sm87BulkV2P40RequestStateIdentity::synthetic_host_contract_valid()
    const noexcept {
  return execution_class == Sm87BulkV2P40RequestStateExecutionClass::
                                kSyntheticHostContract &&
         !production_dispatch_eligible;
}

Sm87BulkV2P40RequestStateSealedAccess::
    Sm87BulkV2P40RequestStateSealedAccess(
        const Sm87BulkV2P40RequestState* const owner,
        const Sm87BulkV2P40RequestStateIdentity& identity,
        const Sm87BulkV2P40RequestArenaLayout& layout,
        const std::array<void*, kSm87BulkV2P40StreamCount>& streams,
        void* const arena_base,
        void* const pinned_handoff_destination) noexcept
    : owner_(owner),
      identity_(identity),
      layout_(layout),
      streams_(streams),
      arena_base_(arena_base),
      pinned_handoff_destination_(pinned_handoff_destination) {}

bool Sm87BulkV2P40RequestStateSealedAccess::valid() const noexcept {
  if (owner_ == nullptr || !identity_.valid() ||
      !sm87_bulk_v2_p40_request_arena_layout_valid(layout_) ||
      arena_base_ == nullptr || pinned_handoff_destination_ == nullptr ||
      reinterpret_cast<std::uintptr_t>(arena_base_) %
              kSm87BulkV2P40ArenaAlignment !=
          0U ||
      reinterpret_cast<std::uintptr_t>(pinned_handoff_destination_) %
              alignof(Sm87BulkV2P40PinnedHandoff) !=
          0U) {
    return false;
  }
  for (void* const stream : streams_) {
    if (stream == nullptr) {
      return false;
    }
  }
  for (const auto& binding : layout_.bindings) {
    if (offset_pointer(arena_base_, binding.range.offset) == nullptr) {
      return false;
    }
  }
  return true;
}

bool Sm87BulkV2P40RequestStateSealedAccess::
    default_off_development_resource_valid() const noexcept {
  return valid() && identity_.default_off_development_resource_valid();
}

void* Sm87BulkV2P40RequestStateSealedAccess::cuda_stream(
    const Sm87BulkV2P40Stream stream) const noexcept {
  return valid_stream(stream) ? streams_[stream_index(stream)] : nullptr;
}

void* Sm87BulkV2P40RequestStateSealedAccess::arena_span(
    const Sm87BulkV2P40RequestArenaRole role) const noexcept {
  const std::size_t index = binding_index(role);
  return index < layout_.bindings.size()
             ? offset_pointer(arena_base_,
                              layout_.bindings[index].range.offset)
             : nullptr;
}

std::uint64_t Sm87BulkV2P40RequestStateSealedAccess::arena_span_bytes(
    const Sm87BulkV2P40RequestArenaRole role) const noexcept {
  const std::size_t index = binding_index(role);
  return index < layout_.bindings.size()
             ? layout_.bindings[index].range.bytes
             : 0U;
}

struct Sm87BulkV2P40RequestState::Impl final {
  Sm87BulkV2P40RequestStateCudaRuntime* cuda = nullptr;
  Sm87BulkV2P40RequestArenaLayout layout{};
  Sm87BulkV2P40RequestStateIdentity identity{};
  std::array<void*, kSm87BulkV2P40StreamCount> streams{};
  void* arena = nullptr;
  Sm87BulkV2P40PinnedHandoff* pinned_handoff = nullptr;
  Sm87BulkV2P40RequestStateLifecycle lifecycle =
      Sm87BulkV2P40RequestStateLifecycle::kInvalid;
  std::uint64_t reset_epoch = 0U;
  std::uint64_t request_epoch = 0U;
  std::uint64_t last_request_epoch = 0U;
  int first_error = 0;
  bool reset_pending_on_main = false;
  bool terminal_main_synchronized = false;
  bool handoff_d2h_enqueued = false;
  bool handoff_observed = false;
  bool state_committed = false;
};

Sm87BulkV2P40RequestState::Sm87BulkV2P40RequestState(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

Sm87BulkV2P40RequestState::~Sm87BulkV2P40RequestState() {
  sealed_access_.reset();
  if (impl_ == nullptr) {
    return;
  }
  auto& impl = *impl_;
  const bool potentially_live =
      impl.lifecycle == Sm87BulkV2P40RequestStateLifecycle::kActive ||
      impl.lifecycle == Sm87BulkV2P40RequestStateLifecycle::kPoisoned ||
      impl.reset_pending_on_main;
  if (potentially_live && impl.cuda != nullptr) {
    // The state owns no streams.  It drains every borrowed v2 stream before
    // releasing data or pinned storage so cancellation/failure cannot turn
    // into a use-after-free.  Normal completed/cancelled teardown is already
    // quiescent and does not repeat the terminal wait.
    for (void* const stream : impl.streams) {
      if (stream != nullptr) {
        (void)impl.cuda->synchronize_stream(stream);
      }
    }
  }
  if (impl.cuda != nullptr && impl.pinned_handoff != nullptr) {
    (void)impl.cuda->free_pinned_host(impl.pinned_handoff);
    impl.pinned_handoff = nullptr;
  }
  if (impl.cuda != nullptr && impl.arena != nullptr) {
    (void)impl.cuda->free_device(impl.arena);
    impl.arena = nullptr;
  }
  impl.lifecycle = Sm87BulkV2P40RequestStateLifecycle::kDestroyed;
}

Sm87BulkV2P40RequestStateCreateResult
Sm87BulkV2P40RequestState::create_bound(
    Sm87BulkV2P40RequestStateCudaRuntime* const cuda,
    const std::array<void*, kSm87BulkV2P40StreamCount>& streams,
    const std::uint64_t owner_identity,
    const std::int32_t expected_device_ordinal,
    const Sm87BulkV2P40RequestStateExecutionClass execution_class) noexcept {
  Sm87BulkV2P40RequestStateCreateResult result;
  constexpr auto layout = sm87_bulk_v2_p40_request_arena_layout();
  if (cuda == nullptr || owner_identity == 0U ||
      expected_device_ordinal < 0 ||
      !sm87_bulk_v2_p40_request_arena_layout_valid(layout) ||
      !sm87_bulk_v2_p40_execution_plan_valid(
          kSm87BulkV2P40FrozenExecutionPlan)) {
    result.status = error(Sm87BulkV2P40RequestStateError::kInvalidPlan,
                          "v2_p40_request_state_frozen_plan");
    return result;
  }
  for (std::size_t index = 0U; index < streams.size(); ++index) {
    if (streams[index] == nullptr) {
      result.status = error(
          Sm87BulkV2P40RequestStateError::kInvalidOwner,
          "all_five_owner_streams_required", 0, index);
      return result;
    }
  }

  std::int32_t current_device = -1;
  int cuda_status = cuda->get_current_device(&current_device);
  if (cuda_status != 0) {
    result.status = error(Sm87BulkV2P40RequestStateError::kDeviceQuery,
                          "get_current_device", cuda_status);
    return result;
  }
  Sm87BulkV2P40RequestDeviceProperties properties{};
  cuda_status = cuda->get_device_properties(current_device, &properties);
  if (cuda_status != 0) {
    result.status = error(Sm87BulkV2P40RequestStateError::kDeviceQuery,
                          "get_device_properties", cuda_status);
    return result;
  }
  if (current_device != expected_device_ordinal || properties.major != 8 ||
      properties.minor != 7 || properties.multiprocessor_count != 16) {
    result.status = error(Sm87BulkV2P40RequestStateError::kWrongDevice,
                          "exact_owner_bound_sm87_16sm_required");
    return result;
  }
  for (std::size_t index = 0U; index < streams.size(); ++index) {
    unsigned int flags = 0U;
    cuda_status = cuda->get_stream_flags(streams[index], &flags);
    if (cuda_status != 0 ||
        (flags & kSm87BulkV2P40NonBlockingStreamFlag) == 0U) {
      result.status = error(
          Sm87BulkV2P40RequestStateError::kStreamValidation,
          "validate_borrowed_nonblocking_owner_stream", cuda_status,
          index);
      return result;
    }
  }

  void* arena = nullptr;
  Sm87BulkV2P40PinnedHandoff* pinned_handoff = nullptr;
  const auto release_locals = [&]() noexcept {
    if (pinned_handoff != nullptr) {
      (void)cuda->free_pinned_host(pinned_handoff);
      pinned_handoff = nullptr;
    }
    if (arena != nullptr) {
      (void)cuda->free_device(arena);
      arena = nullptr;
    }
  };

  cuda_status = cuda->allocate_device(
      &arena, static_cast<std::size_t>(layout.arena_bytes));
  if (cuda_status != 0 || arena == nullptr) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kDeviceAllocation,
        "allocate_exact_5075652608_byte_request_arena", cuda_status);
    release_locals();
    return result;
  }
  Sm87BulkV2P40RequestPointerAttributes arena_attributes{};
  cuda_status = cuda->query_pointer(arena, &arena_attributes);
  if (cuda_status != 0 ||
      arena_attributes.kind != Sm87BulkV2P40RequestPointerKind::kDevice ||
      arena_attributes.device_pointer != arena ||
      arena_attributes.device_ordinal != current_device ||
      reinterpret_cast<std::uintptr_t>(arena) %
              kSm87BulkV2P40ArenaAlignment !=
          0U ||
      offset_pointer(arena, layout.arena_bytes - 1U) == nullptr) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kDevicePointerValidation,
        "validate_exact_request_arena_pointer_and_extent", cuda_status);
    release_locals();
    return result;
  }

  void* pinned = nullptr;
  cuda_status = cuda->allocate_pinned_host(
      &pinned, sizeof(Sm87BulkV2P40PinnedHandoff));
  pinned_handoff = static_cast<Sm87BulkV2P40PinnedHandoff*>(pinned);
  if (cuda_status != 0 || pinned_handoff == nullptr) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kPinnedHandoffAllocation,
        "allocate_owner_bound_pinned_8_byte_handoff", cuda_status);
    release_locals();
    return result;
  }
  Sm87BulkV2P40RequestPointerAttributes handoff_attributes{};
  cuda_status = cuda->query_pointer(pinned_handoff, &handoff_attributes);
  if (cuda_status != 0 ||
      handoff_attributes.kind != Sm87BulkV2P40RequestPointerKind::kHost ||
      handoff_attributes.host_pointer != pinned_handoff ||
      reinterpret_cast<std::uintptr_t>(pinned_handoff) %
              alignof(Sm87BulkV2P40PinnedHandoff) !=
          0U) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kPinnedHandoffValidation,
        "validate_owner_bound_pinned_8_byte_handoff", cuda_status);
    release_locals();
    return result;
  }
  *pinned_handoff = {};

  cuda_status = cuda->memset_async(
      arena, 0, static_cast<std::size_t>(layout.cold_reset_bytes),
      streams[stream_index(Sm87BulkV2P40Stream::kMain)]);
  if (cuda_status != 0) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kColdResetSubmission,
        "initialize_only_gdn_persistent_state_and_history", cuda_status);
    release_locals();
    return result;
  }
  cuda_status = cuda->synchronize_stream(
      streams[stream_index(Sm87BulkV2P40Stream::kMain)]);
  if (cuda_status != 0) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kColdResetSynchronize,
        "synchronize_initial_gdn_persistent_cold_reset", cuda_status);
    release_locals();
    return result;
  }

  std::unique_ptr<Impl> impl(new (std::nothrow) Impl());
  if (impl == nullptr) {
    result.status = error(Sm87BulkV2P40RequestStateError::kAccessAllocation,
                          "allocate_request_state_impl");
    release_locals();
    return result;
  }
  impl->cuda = cuda;
  impl->layout = layout;
  impl->identity.plan_magic = kSm87BulkV2P40PlanMagic;
  impl->identity.abi_major = kSm87BulkV2P40PlanAbiMajor;
  impl->identity.abi_minor = kSm87BulkV2P40PlanAbiMinor;
  impl->identity.owner_identity = owner_identity;
  impl->identity.seal_nonce = next_identity();
  impl->identity.allocation_identity = next_identity();
  // This v2 owner is the physical owner of all five borrowed streams/events;
  // no caller-provided stream namespace is accepted here.
  impl->identity.stream_event_owner_identity = owner_identity;
  impl->identity.pinned_handoff_identity = next_identity();
  impl->identity.device_ordinal = current_device;
  impl->identity.execution_class = execution_class;
  impl->identity.one_exact_device_allocation = true;
  impl->identity.control_plane_owned_elsewhere = true;
  impl->identity.whole_arena_reset_forbidden = true;
  impl->identity.production_dispatch_eligible = false;
  impl->streams = streams;
  impl->arena = arena;
  impl->pinned_handoff = pinned_handoff;
  impl->lifecycle = Sm87BulkV2P40RequestStateLifecycle::kReady;
  impl->reset_epoch = next_identity();
  impl->reset_pending_on_main = false;

  std::unique_ptr<Sm87BulkV2P40RequestState> state(
      new (std::nothrow) Sm87BulkV2P40RequestState(std::move(impl)));
  if (state == nullptr) {
    result.status = error(Sm87BulkV2P40RequestStateError::kAccessAllocation,
                          "allocate_request_state_owner");
    release_locals();
    return result;
  }
  arena = nullptr;
  pinned_handoff = nullptr;
  std::unique_ptr<Sm87BulkV2P40RequestStateSealedAccess> access(
      new (std::nothrow) Sm87BulkV2P40RequestStateSealedAccess(
          state.get(), state->impl_->identity, state->impl_->layout,
          state->impl_->streams, state->impl_->arena,
          state->impl_->pinned_handoff));
  if (access == nullptr || !access->valid()) {
    result.status = error(Sm87BulkV2P40RequestStateError::kAccessAllocation,
                          "mint_immutable_request_state_access");
    return result;
  }
  state->sealed_access_ = std::move(access);
  result.state = std::move(state);
  result.status = ok();
  return result;
}

Sm87BulkV2P40RequestStateLifecycle
Sm87BulkV2P40RequestState::lifecycle() const noexcept {
  return impl_ != nullptr ? impl_->lifecycle
                          : Sm87BulkV2P40RequestStateLifecycle::kInvalid;
}

const Sm87BulkV2P40RequestStateSealedAccess*
Sm87BulkV2P40RequestState::sealed_access() const noexcept {
  return sealed_access_.get();
}

std::uint64_t Sm87BulkV2P40RequestState::request_epoch() const noexcept {
  return impl_ != nullptr ? impl_->request_epoch : 0U;
}

bool Sm87BulkV2P40RequestState::access_matches(
    const Sm87BulkV2P40RequestStateSealedAccess& access) const noexcept {
  return impl_ != nullptr && sealed_access_ != nullptr &&
         sealed_access_.get() == &access && access.owner_ == this &&
         access.valid() && identity_equal(access.identity_, impl_->identity) &&
         access.layout_.arena_bytes == impl_->layout.arena_bytes &&
         access.streams_ == impl_->streams &&
         access.arena_base_ == impl_->arena &&
         access.pinned_handoff_destination_ == impl_->pinned_handoff;
}

Sm87BulkV2P40RequestStateStatus Sm87BulkV2P40RequestState::begin_request(
    const Sm87BulkV2P40RequestStateSealedAccess& access,
    const std::uint64_t request_epoch) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40RequestStateError::kForeignAccess,
                 "begin_requires_owner_issued_request_state_access");
  }
  if (impl_->lifecycle != Sm87BulkV2P40RequestStateLifecycle::kReady) {
    return error(Sm87BulkV2P40RequestStateError::kInvalidLifecycle,
                 "begin_requires_cold_ready_request_state");
  }
  if (request_epoch == 0U || request_epoch <= impl_->last_request_epoch) {
    return error(Sm87BulkV2P40RequestStateError::kInvalidRequestEpoch,
                 "begin_requires_fresh_strictly_increasing_epoch");
  }
  impl_->request_epoch = request_epoch;
  impl_->last_request_epoch = request_epoch;
  impl_->first_error = 0;
  impl_->reset_pending_on_main = false;
  impl_->terminal_main_synchronized = false;
  impl_->handoff_d2h_enqueued = false;
  impl_->handoff_observed = false;
  impl_->state_committed = false;
  impl_->lifecycle = Sm87BulkV2P40RequestStateLifecycle::kActive;
  return ok();
}

Sm87BulkV2P40RequestStateStatus
Sm87BulkV2P40RequestState::enqueue_handoff_d2h(
    const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40RequestStateError::kForeignAccess,
                 "handoff_d2h_requires_owner_issued_request_state_access");
  }
  if (impl_->lifecycle != Sm87BulkV2P40RequestStateLifecycle::kActive ||
      impl_->handoff_d2h_enqueued) {
    return error(Sm87BulkV2P40RequestStateError::kInvalidLifecycle,
                 "exactly_one_active_handoff_d2h_required");
  }
  constexpr auto source_range = final_greedy_workspace_family_range();
  static_assert(source_range.bytes >= sizeof(Sm87BulkV2P40PinnedHandoff));
  const std::uint64_t source_offset =
      kSm87BulkV2P40FamilyArenaOffset + source_range.offset;
  const void* const device_handoff_source =
      offset_pointer(impl_->arena, source_offset);
  if (device_handoff_source == nullptr ||
      source_offset > impl_->layout.arena_bytes ||
      sizeof(Sm87BulkV2P40PinnedHandoff) >
          impl_->layout.arena_bytes - source_offset) {
    impl_->first_error = static_cast<int>(cudaErrorInvalidValue);
    impl_->lifecycle = Sm87BulkV2P40RequestStateLifecycle::kPoisoned;
    return error(Sm87BulkV2P40RequestStateError::kInvalidPlan,
                 "frozen_final_greedy_workspace_handoff_source");
  }
  const int cuda_status = impl_->cuda->copy_device_to_host_async(
      impl_->pinned_handoff, device_handoff_source,
      sizeof(Sm87BulkV2P40PinnedHandoff),
      impl_->streams[stream_index(Sm87BulkV2P40Stream::kMain)]);
  if (cuda_status != 0) {
    impl_->first_error = cuda_status;
    impl_->lifecycle = Sm87BulkV2P40RequestStateLifecycle::kPoisoned;
    impl_->state_committed = false;
    return error(Sm87BulkV2P40RequestStateError::kHandoffSubmission,
                 "enqueue_owner_bound_8_byte_handoff_d2h", cuda_status);
  }
  impl_->handoff_d2h_enqueued = true;
  return ok();
}

Sm87BulkV2P40RequestStateRearmResult
Sm87BulkV2P40RequestState::rearm_for_cold_request(
    const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept {
  Sm87BulkV2P40RequestStateRearmResult result;
  if (!access_matches(access)) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kForeignAccess,
        "rearm_requires_owner_issued_request_state_access");
    return result;
  }
  result.source_lifecycle = impl_->lifecycle;
  result.result_lifecycle = impl_->lifecycle;
  result.owner_identity = impl_->identity.owner_identity;
  result.allocation_identity = impl_->identity.allocation_identity;
  result.previous_reset_epoch = impl_->reset_epoch;
  result.device_allocations = 1U;
  result.request_hot_static_cuda_queries = 0U;

  if (impl_->lifecycle != Sm87BulkV2P40RequestStateLifecycle::kReady &&
      impl_->lifecycle != Sm87BulkV2P40RequestStateLifecycle::kCompleted &&
      impl_->lifecycle != Sm87BulkV2P40RequestStateLifecycle::kCancelled) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kInvalidLifecycle,
        "rearm_rejects_active_poisoned_or_invalid_state");
    return result;
  }

  void* const arena_before = impl_->arena;
  auto* const handoff_before = impl_->pinned_handoff;
  const auto streams_before = impl_->streams;
  const auto identity_before = impl_->identity;
  *impl_->pinned_handoff = {};
  const int cuda_status = impl_->cuda->memset_async(
      access.arena_span(
          Sm87BulkV2P40RequestArenaRole::kGdnColdStateAndHistory),
      0, static_cast<std::size_t>(kSm87BulkV2P40ColdResetBytes),
      impl_->streams[stream_index(Sm87BulkV2P40Stream::kMain)]);
  if (cuda_status != 0) {
    impl_->first_error = cuda_status;
    impl_->lifecycle = Sm87BulkV2P40RequestStateLifecycle::kPoisoned;
    result.result_lifecycle = impl_->lifecycle;
    result.status = error(
        Sm87BulkV2P40RequestStateError::kColdResetSubmission,
        "rearm_only_gdn_persistent_state_and_history", cuda_status);
    return result;
  }

  impl_->reset_epoch = next_identity();
  impl_->request_epoch = 0U;
  impl_->first_error = 0;
  impl_->reset_pending_on_main = true;
  impl_->terminal_main_synchronized = false;
  impl_->handoff_d2h_enqueued = false;
  impl_->handoff_observed = false;
  impl_->state_committed = false;
  impl_->lifecycle = Sm87BulkV2P40RequestStateLifecycle::kReady;

  result.status = ok();
  result.result_lifecycle = impl_->lifecycle;
  result.reset_epoch = impl_->reset_epoch;
  result.zeroed_device_bytes = kSm87BulkV2P40ColdResetBytes;
  result.reset_enqueued_on_main = true;
  result.whole_arena_reset = false;
  result.control_plane_touched = false;
  result.addresses_and_identities_preserved =
      arena_before == impl_->arena && handoff_before == impl_->pinned_handoff &&
      streams_before == impl_->streams &&
      identity_equal(identity_before, impl_->identity);
  return result;
}

Sm87BulkV2P40RequestStateStatus
Sm87BulkV2P40RequestState::mark_cancelled_after_owner_drain(
    const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40RequestStateError::kForeignAccess,
                 "cancel_requires_owner_issued_request_state_access");
  }
  if (impl_->lifecycle != Sm87BulkV2P40RequestStateLifecycle::kActive) {
    return error(Sm87BulkV2P40RequestStateError::kInvalidLifecycle,
                 "cancel_requires_active_unpublished_request");
  }
  impl_->reset_pending_on_main = false;
  impl_->terminal_main_synchronized = false;
  impl_->handoff_d2h_enqueued = false;
  impl_->handoff_observed = false;
  impl_->state_committed = false;
  impl_->lifecycle = Sm87BulkV2P40RequestStateLifecycle::kCancelled;
  return ok();
}

Sm87BulkV2P40RequestStateStatus
Sm87BulkV2P40RequestState::poison_after_owner_drain(
    const Sm87BulkV2P40RequestStateSealedAccess& access,
    const int first_error) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40RequestStateError::kForeignAccess,
                 "poison_requires_owner_issued_request_state_access");
  }
  if ((impl_->lifecycle != Sm87BulkV2P40RequestStateLifecycle::kActive &&
       impl_->lifecycle != Sm87BulkV2P40RequestStateLifecycle::kReady) ||
      first_error == 0) {
    return error(Sm87BulkV2P40RequestStateError::kInvalidLifecycle,
                 "poison_requires_failed_uncommitted_owner_drain");
  }
  impl_->first_error = first_error;
  impl_->reset_pending_on_main = false;
  impl_->terminal_main_synchronized = false;
  impl_->handoff_d2h_enqueued = false;
  impl_->handoff_observed = false;
  impl_->state_committed = false;
  impl_->lifecycle = Sm87BulkV2P40RequestStateLifecycle::kPoisoned;
  return ok();
}

Sm87BulkV2P40RequestStateHandoffResult
Sm87BulkV2P40RequestState::synchronize_terminal_main_and_observe_handoff(
    const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept {
  Sm87BulkV2P40RequestStateHandoffResult result;
  if (!access_matches(access)) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kForeignAccess,
        "handoff_requires_owner_issued_request_state_access");
    return result;
  }
  result.request_epoch = impl_->request_epoch;
  if (impl_->lifecycle != Sm87BulkV2P40RequestStateLifecycle::kActive) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kInvalidLifecycle,
        "handoff_requires_active_unpublished_request");
    return result;
  }
  if (!impl_->handoff_d2h_enqueued) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kHandoffNotEnqueued,
        "terminal_observation_requires_owner_bound_handoff_d2h");
    return result;
  }

  const int cuda_status = impl_->cuda->synchronize_stream(
      impl_->streams[stream_index(Sm87BulkV2P40Stream::kMain)]);
  if (cuda_status != 0) {
    impl_->first_error = cuda_status;
    impl_->lifecycle = Sm87BulkV2P40RequestStateLifecycle::kPoisoned;
    impl_->state_committed = false;
    result.status = error(
        Sm87BulkV2P40RequestStateError::kTerminalSynchronize,
        "terminal_main_sync_before_pinned_handoff_observation",
        cuda_status);
    return result;
  }

  impl_->reset_pending_on_main = false;
  impl_->terminal_main_synchronized = true;
  const Sm87BulkV2P40PinnedHandoff observed = *impl_->pinned_handoff;
  impl_->handoff_observed = true;
  result.token_id = observed.token_id;
  result.nonfinite = observed.nonfinite;
  result.terminal_main_synchronized = true;
  result.handoff_observed = true;
  if (observed.token_id >= kSm87BulkV2P40Vocabulary ||
      observed.nonfinite != 0U) {
    impl_->first_error = static_cast<int>(cudaErrorInvalidValue);
    impl_->state_committed = false;
    impl_->lifecycle = Sm87BulkV2P40RequestStateLifecycle::kPoisoned;
    result.status = error(
        Sm87BulkV2P40RequestStateError::kInvalidHandoff,
        "finite_in_vocabulary_terminal_handoff_required",
        static_cast<int>(cudaErrorInvalidValue));
    return result;
  }

  impl_->first_error = 0;
  impl_->state_committed = true;
  impl_->lifecycle = Sm87BulkV2P40RequestStateLifecycle::kCompleted;
  result.status = ok();
  result.state_committed = true;
  return result;
}

Sm87BulkV2P40RequestStateCreateResult
create_sm87_bulk_dataflow_v2_p40_request_state(
    Sm87BulkV2P40Owner& owner) noexcept {
  Sm87BulkV2P40RequestStateCreateResult result;
#if !defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_REQUEST_STATE_ADMISSION)
  (void)owner;
  result.status = error(
      Sm87BulkV2P40RequestStateError::kAdmissionDisabled,
      "Q3X_BUILD_SM87_BULK_DATAFLOW_V2_P40_OWNER_ADMISSION");
  return result;
#else
  if (owner.state_ != Sm87BulkV2P40OwnerState::kResourcesReady ||
      owner.execution_access_ != nullptr || owner.owner_identity_ == 0U ||
      owner.device_ordinal_ < 0) {
    result.status = error(
        Sm87BulkV2P40RequestStateError::kInvalidOwner,
        "request_state_requires_unsealed_v2_resource_owner");
    return result;
  }
  static CudartSm87BulkV2P40RequestStateRuntime cuda_runtime;
  return Sm87BulkV2P40RequestState::create_bound(
      &cuda_runtime, owner.streams_, owner.owner_identity_,
      owner.device_ordinal_,
      Sm87BulkV2P40RequestStateExecutionClass::
          kDefaultOffDevelopmentResource);
#endif
}

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_REQUEST_STATE_HOST_FIXTURE)
Sm87BulkV2P40RequestStateCreateResult
Sm87BulkV2P40RequestStateHostFixture::create(
    Sm87BulkV2P40RequestStateCudaRuntime* const cuda,
    const std::array<void*, kSm87BulkV2P40StreamCount>& streams,
    const std::uint64_t owner_identity,
    const std::int32_t expected_device_ordinal) noexcept {
  return Sm87BulkV2P40RequestState::create_bound(
      cuda, streams, owner_identity, expected_device_ordinal,
      Sm87BulkV2P40RequestStateExecutionClass::kSyntheticHostContract);
}

Sm87BulkV2P40RequestStateStatus
Sm87BulkV2P40RequestStateHostFixture::mark_cancelled_after_owner_drain(
    Sm87BulkV2P40RequestState& state,
    const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept {
  return state.mark_cancelled_after_owner_drain(access);
}

Sm87BulkV2P40RequestStateStatus
Sm87BulkV2P40RequestStateHostFixture::poison_after_owner_drain(
    Sm87BulkV2P40RequestState& state,
    const Sm87BulkV2P40RequestStateSealedAccess& access,
    const int first_error) noexcept {
  return state.poison_after_owner_drain(access, first_error);
}

Sm87BulkV2P40RequestStateHandoffResult
Sm87BulkV2P40RequestStateHostFixture::
    synchronize_terminal_main_and_observe_handoff(
        Sm87BulkV2P40RequestState& state,
        const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept {
  return state.synchronize_terminal_main_and_observe_handoff(access);
}

void Sm87BulkV2P40RequestStateHostFixture::emulate_completed_handoff_d2h(
    Sm87BulkV2P40RequestState& state, const std::uint32_t token_id,
    const std::uint32_t nonfinite) noexcept {
  if (state.impl_ != nullptr && state.impl_->pinned_handoff != nullptr &&
      state.impl_->lifecycle ==
          Sm87BulkV2P40RequestStateLifecycle::kActive) {
    state.impl_->pinned_handoff->token_id = token_id;
    state.impl_->pinned_handoff->nonfinite = nonfinite;
    state.impl_->handoff_d2h_enqueued = true;
  }
}
#endif

const char* to_string(const Sm87BulkV2P40RequestStateError value) noexcept {
  switch (value) {
    case Sm87BulkV2P40RequestStateError::kNone:
      return "none";
    case Sm87BulkV2P40RequestStateError::kAdmissionDisabled:
      return "admission_disabled";
    case Sm87BulkV2P40RequestStateError::kInvalidOwner:
      return "invalid_owner";
    case Sm87BulkV2P40RequestStateError::kInvalidPlan:
      return "invalid_plan";
    case Sm87BulkV2P40RequestStateError::kDeviceQuery:
      return "device_query";
    case Sm87BulkV2P40RequestStateError::kWrongDevice:
      return "wrong_device";
    case Sm87BulkV2P40RequestStateError::kStreamValidation:
      return "stream_validation";
    case Sm87BulkV2P40RequestStateError::kDeviceAllocation:
      return "device_allocation";
    case Sm87BulkV2P40RequestStateError::kDevicePointerValidation:
      return "device_pointer_validation";
    case Sm87BulkV2P40RequestStateError::kPinnedHandoffAllocation:
      return "pinned_handoff_allocation";
    case Sm87BulkV2P40RequestStateError::kPinnedHandoffValidation:
      return "pinned_handoff_validation";
    case Sm87BulkV2P40RequestStateError::kColdResetSubmission:
      return "cold_reset_submission";
    case Sm87BulkV2P40RequestStateError::kColdResetSynchronize:
      return "cold_reset_synchronize";
    case Sm87BulkV2P40RequestStateError::kAccessAllocation:
      return "access_allocation";
    case Sm87BulkV2P40RequestStateError::kForeignAccess:
      return "foreign_access";
    case Sm87BulkV2P40RequestStateError::kInvalidLifecycle:
      return "invalid_lifecycle";
    case Sm87BulkV2P40RequestStateError::kInvalidRequestEpoch:
      return "invalid_request_epoch";
    case Sm87BulkV2P40RequestStateError::kHandoffSubmission:
      return "handoff_submission";
    case Sm87BulkV2P40RequestStateError::kHandoffNotEnqueued:
      return "handoff_not_enqueued";
    case Sm87BulkV2P40RequestStateError::kTerminalSynchronize:
      return "terminal_synchronize";
    case Sm87BulkV2P40RequestStateError::kInvalidHandoff:
      return "invalid_handoff";
  }
  return "unknown";
}

}  // namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail
