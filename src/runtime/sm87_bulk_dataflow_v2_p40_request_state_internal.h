#pragma once

#include "sm87_bulk_dataflow_v2_p40_owner_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail {

// Independent v2 request-data owner.  It borrows the v2 owner's five streams
// but owns no stream, event, control word, or device-control allocation.  Its
// only device allocation is the exact 5,075,652,608-byte data plane frozen by
// sm87_bulk_dataflow_v2_p40_plan.h.  This interface is default-off development
// infrastructure and carries no production-dispatch qualification.

struct alignas(8) Sm87BulkV2P40PinnedHandoff final {
  std::uint32_t token_id = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t nonfinite = 1U;
};

static_assert(sizeof(Sm87BulkV2P40PinnedHandoff) == 8U);
static_assert(alignof(Sm87BulkV2P40PinnedHandoff) == 8U);

enum class Sm87BulkV2P40RequestArenaRole : std::uint8_t {
  kInvalid = 0U,
  kPersistent,
  kGdnColdStateAndHistory,
  kResidual,
  kFamily,
  kFinalHidden,
};

struct Sm87BulkV2P40RequestArenaBinding final {
  Sm87BulkV2P40RequestArenaRole role =
      Sm87BulkV2P40RequestArenaRole::kInvalid;
  Sm87BulkV2P40Range range{};
};

inline constexpr std::size_t kSm87BulkV2P40RequestArenaBindingCount = 5U;

struct Sm87BulkV2P40RequestArenaLayout final {
  std::uint64_t arena_bytes = 0U;
  std::array<Sm87BulkV2P40RequestArenaBinding,
             kSm87BulkV2P40RequestArenaBindingCount>
      bindings{};
  std::uint64_t cold_reset_bytes = 0U;
  std::uint64_t separately_owned_control_bytes = 0U;
  std::uint64_t pinned_handoff_bytes = 0U;
  bool one_device_allocation = false;
  bool control_plane_is_external = false;
  bool whole_arena_reset_forbidden = false;
};

[[nodiscard]] constexpr Sm87BulkV2P40RequestArenaLayout
sm87_bulk_v2_p40_request_arena_layout() noexcept {
  return {
      kSm87BulkV2P40RequestArenaBytes,
      {{
          {Sm87BulkV2P40RequestArenaRole::kPersistent,
           {0U, kSm87BulkV2P40PersistentBytes}},
          {Sm87BulkV2P40RequestArenaRole::kGdnColdStateAndHistory,
           {0U, kSm87BulkV2P40ColdResetBytes}},
          {Sm87BulkV2P40RequestArenaRole::kResidual,
           {kSm87BulkV2P40PersistentBytes,
            kSm87BulkV2P40ResidualBytes}},
          {Sm87BulkV2P40RequestArenaRole::kFamily,
           {kSm87BulkV2P40FamilyArenaOffset,
            kSm87BulkV2P40FamilyArenaBytes}},
          {Sm87BulkV2P40RequestArenaRole::kFinalHidden,
           {kSm87BulkV2P40FamilyArenaOffset +
                kSm87BulkV2P40FamilyArenaBytes,
            kSm87BulkV2P40FinalHiddenBytes}},
      }},
      kSm87BulkV2P40ColdResetBytes,
      kSm87BulkV2P40ControlArenaBytes,
      sizeof(Sm87BulkV2P40PinnedHandoff),
      true,
      true,
      true,
  };
}

[[nodiscard]] constexpr bool sm87_bulk_v2_p40_request_arena_layout_valid(
    const Sm87BulkV2P40RequestArenaLayout& layout) noexcept {
  constexpr auto expected = sm87_bulk_v2_p40_request_arena_layout();
  if (layout.arena_bytes != kSm87BulkV2P40RequestArenaBytes ||
      layout.cold_reset_bytes != kSm87BulkV2P40ColdResetBytes ||
      layout.separately_owned_control_bytes !=
          kSm87BulkV2P40ControlArenaBytes ||
      layout.pinned_handoff_bytes !=
          sizeof(Sm87BulkV2P40PinnedHandoff) ||
      !layout.one_device_allocation || !layout.control_plane_is_external ||
      !layout.whole_arena_reset_forbidden) {
    return false;
  }
  for (std::size_t index = 0U; index < layout.bindings.size(); ++index) {
    const auto& binding = layout.bindings[index];
    const auto& canonical = expected.bindings[index];
    if (binding.role != canonical.role ||
        binding.range.offset != canonical.range.offset ||
        binding.range.bytes != canonical.range.bytes ||
        !binding.range.valid(layout.arena_bytes) ||
        binding.range.offset % kSm87BulkV2P40ArenaAlignment != 0U) {
      return false;
    }
  }
  const auto& persistent = layout.bindings[0U].range;
  const auto& cold = layout.bindings[1U].range;
  const auto& residual = layout.bindings[2U].range;
  const auto& family = layout.bindings[3U].range;
  const auto& final_hidden = layout.bindings[4U].range;
  return persistent.offset == 0U && cold.offset == persistent.offset &&
         cold.bytes < persistent.bytes && cold.end() <= persistent.end() &&
         persistent.end() == residual.offset &&
         residual.end() == family.offset &&
         family.end() == final_hidden.offset &&
         final_hidden.end() == layout.arena_bytes &&
         !sm87_bulk_v2_p40_ranges_overlap(persistent, residual) &&
         !sm87_bulk_v2_p40_ranges_overlap(persistent, family) &&
         !sm87_bulk_v2_p40_ranges_overlap(persistent, final_hidden) &&
         !sm87_bulk_v2_p40_ranges_overlap(residual, family) &&
         !sm87_bulk_v2_p40_ranges_overlap(residual, final_hidden) &&
         !sm87_bulk_v2_p40_ranges_overlap(family, final_hidden);
}

enum class Sm87BulkV2P40RequestPointerKind : std::uint8_t {
  kUnknown = 0U,
  kHost,
  kDevice,
};

struct Sm87BulkV2P40RequestPointerAttributes final {
  Sm87BulkV2P40RequestPointerKind kind =
      Sm87BulkV2P40RequestPointerKind::kUnknown;
  void* host_pointer = nullptr;
  void* device_pointer = nullptr;
  std::int32_t device_ordinal = -1;
};

struct Sm87BulkV2P40RequestDeviceProperties final {
  std::int32_t major = 0;
  std::int32_t minor = 0;
  std::int32_t multiprocessor_count = 0;
};

// Narrow CUDA seam for host lifecycle tests.  Static query methods are used
// only while minting the immutable startup capability; the request hot path
// has no route back to them.
class Sm87BulkV2P40RequestStateCudaRuntime {
 public:
  virtual ~Sm87BulkV2P40RequestStateCudaRuntime() = default;

  [[nodiscard]] virtual int get_current_device(
      std::int32_t* device_ordinal) noexcept = 0;
  [[nodiscard]] virtual int get_device_properties(
      std::int32_t device_ordinal,
      Sm87BulkV2P40RequestDeviceProperties* properties) noexcept = 0;
  [[nodiscard]] virtual int get_stream_flags(
      void* stream, unsigned int* flags) noexcept = 0;
  [[nodiscard]] virtual int allocate_device(void** pointer,
                                            std::size_t bytes) noexcept = 0;
  [[nodiscard]] virtual int free_device(void* pointer) noexcept = 0;
  [[nodiscard]] virtual int allocate_pinned_host(void** pointer,
                                                 std::size_t bytes) noexcept = 0;
  [[nodiscard]] virtual int free_pinned_host(void* pointer) noexcept = 0;
  [[nodiscard]] virtual int query_pointer(
      const void* pointer,
      Sm87BulkV2P40RequestPointerAttributes* attributes) noexcept = 0;
  [[nodiscard]] virtual int memset_async(void* pointer, int value,
                                         std::size_t bytes,
                                         void* stream) noexcept = 0;
  [[nodiscard]] virtual int copy_device_to_host_async(
      void* host_destination, const void* device_source, std::size_t bytes,
      void* stream) noexcept = 0;
  [[nodiscard]] virtual int synchronize_stream(void* stream) noexcept = 0;
};

enum class Sm87BulkV2P40RequestStateLifecycle : std::uint8_t {
  kInvalid = 0U,
  kReady,
  kActive,
  kCompleted,
  kCancelled,
  kPoisoned,
  kDestroyed,
};

enum class Sm87BulkV2P40RequestStateExecutionClass : std::uint8_t {
  kInvalid = 0U,
  kDefaultOffDevelopmentResource,
  kSyntheticHostContract,
};

enum class Sm87BulkV2P40RequestStateError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kInvalidOwner,
  kInvalidPlan,
  kDeviceQuery,
  kWrongDevice,
  kStreamValidation,
  kDeviceAllocation,
  kDevicePointerValidation,
  kPinnedHandoffAllocation,
  kPinnedHandoffValidation,
  kColdResetSubmission,
  kColdResetSynchronize,
  kAccessAllocation,
  kForeignAccess,
  kInvalidLifecycle,
  kInvalidRequestEpoch,
  kHandoffSubmission,
  kHandoffNotEnqueued,
  kTerminalSynchronize,
  kInvalidHandoff,
};

struct Sm87BulkV2P40RequestStateStatus final {
  Sm87BulkV2P40RequestStateError error =
      Sm87BulkV2P40RequestStateError::kNone;
  const char* context = "none";
  int cuda_error = 0;
  std::size_t resource_index = std::numeric_limits<std::size_t>::max();

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == Sm87BulkV2P40RequestStateError::kNone;
  }
};

struct Sm87BulkV2P40RequestStateIdentity final {
  std::array<std::uint8_t, 8U> plan_magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  std::uint64_t owner_identity = 0U;
  std::uint64_t seal_nonce = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t stream_event_owner_identity = 0U;
  std::uint64_t pinned_handoff_identity = 0U;
  std::int32_t device_ordinal = -1;
  Sm87BulkV2P40RequestStateExecutionClass execution_class =
      Sm87BulkV2P40RequestStateExecutionClass::kInvalid;
  bool one_exact_device_allocation = false;
  bool control_plane_owned_elsewhere = false;
  bool whole_arena_reset_forbidden = false;
  bool production_dispatch_eligible = true;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool default_off_development_resource_valid()
      const noexcept;
  [[nodiscard]] bool synthetic_host_contract_valid() const noexcept;
};

class Sm87BulkV2P40RequestState;

// Immutable owner-issued capability.  The executor may obtain arena spans,
// but never the pinned host address; it can request only the fixed 8-byte D2H
// through the Owner transaction.  Neither executor nor caller can enqueue the
// fixed D2H directly.  Observation remains a private state transition after
// terminal Main sync.
class Sm87BulkV2P40RequestStateSealedAccess final {
 public:
  Sm87BulkV2P40RequestStateSealedAccess(
      const Sm87BulkV2P40RequestStateSealedAccess&) = delete;
  Sm87BulkV2P40RequestStateSealedAccess& operator=(
      const Sm87BulkV2P40RequestStateSealedAccess&) = delete;
  Sm87BulkV2P40RequestStateSealedAccess(
      Sm87BulkV2P40RequestStateSealedAccess&&) = delete;
  Sm87BulkV2P40RequestStateSealedAccess& operator=(
      Sm87BulkV2P40RequestStateSealedAccess&&) = delete;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool default_off_development_resource_valid()
      const noexcept;
  [[nodiscard]] const Sm87BulkV2P40RequestStateIdentity& identity()
      const noexcept {
    return identity_;
  }
  [[nodiscard]] const Sm87BulkV2P40RequestArenaLayout& layout()
      const noexcept {
    return layout_;
  }
  [[nodiscard]] void* cuda_stream(Sm87BulkV2P40Stream stream) const noexcept;
  [[nodiscard]] void* arena_base() const noexcept { return arena_base_; }
  [[nodiscard]] std::uint64_t arena_bytes() const noexcept {
    return layout_.arena_bytes;
  }
  [[nodiscard]] void* arena_span(
      Sm87BulkV2P40RequestArenaRole role) const noexcept;
  [[nodiscard]] std::uint64_t arena_span_bytes(
      Sm87BulkV2P40RequestArenaRole role) const noexcept;
 private:
  Sm87BulkV2P40RequestStateSealedAccess(
      const Sm87BulkV2P40RequestState* owner,
      const Sm87BulkV2P40RequestStateIdentity& identity,
      const Sm87BulkV2P40RequestArenaLayout& layout,
      const std::array<void*, kSm87BulkV2P40StreamCount>& streams,
      void* arena_base, void* pinned_handoff_destination) noexcept;

  const Sm87BulkV2P40RequestState* owner_ = nullptr;
  Sm87BulkV2P40RequestStateIdentity identity_{};
  Sm87BulkV2P40RequestArenaLayout layout_{};
  std::array<void*, kSm87BulkV2P40StreamCount> streams_{};
  void* arena_base_ = nullptr;
  void* pinned_handoff_destination_ = nullptr;

  friend class Sm87BulkV2P40RequestState;
};

struct Sm87BulkV2P40RequestStateRearmResult final {
  Sm87BulkV2P40RequestStateStatus status{};
  Sm87BulkV2P40RequestStateLifecycle source_lifecycle =
      Sm87BulkV2P40RequestStateLifecycle::kInvalid;
  Sm87BulkV2P40RequestStateLifecycle result_lifecycle =
      Sm87BulkV2P40RequestStateLifecycle::kInvalid;
  std::uint64_t owner_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t previous_reset_epoch = 0U;
  std::uint64_t reset_epoch = 0U;
  std::uint64_t zeroed_device_bytes = 0U;
  std::size_t device_allocations = 0U;
  std::size_t request_hot_static_cuda_queries = 0U;
  bool reset_enqueued_on_main = false;
  bool whole_arena_reset = true;
  bool control_plane_touched = true;
  bool addresses_and_identities_preserved = false;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) &&
           result_lifecycle == Sm87BulkV2P40RequestStateLifecycle::kReady &&
           owner_identity != 0U && allocation_identity != 0U &&
           reset_epoch != 0U && reset_epoch != previous_reset_epoch &&
           zeroed_device_bytes == kSm87BulkV2P40ColdResetBytes &&
           device_allocations == 1U &&
           request_hot_static_cuda_queries == 0U &&
           reset_enqueued_on_main && !whole_arena_reset &&
           !control_plane_touched && addresses_and_identities_preserved;
  }
};

struct Sm87BulkV2P40RequestStateHandoffResult final {
  Sm87BulkV2P40RequestStateStatus status{};
  std::uint64_t request_epoch = 0U;
  std::uint32_t token_id = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t nonfinite = 1U;
  bool terminal_main_synchronized = false;
  bool handoff_observed = false;
  bool state_committed = false;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && request_epoch != 0U &&
           token_id < kSm87BulkV2P40Vocabulary && nonfinite == 0U &&
           terminal_main_synchronized && handoff_observed &&
           state_committed;
  }
};

class Sm87BulkV2P40RequestState final {
 public:
  Sm87BulkV2P40RequestState(const Sm87BulkV2P40RequestState&) = delete;
  Sm87BulkV2P40RequestState& operator=(
      const Sm87BulkV2P40RequestState&) = delete;
  Sm87BulkV2P40RequestState(Sm87BulkV2P40RequestState&&) = delete;
  Sm87BulkV2P40RequestState& operator=(
      Sm87BulkV2P40RequestState&&) = delete;
  ~Sm87BulkV2P40RequestState();

  [[nodiscard]] Sm87BulkV2P40RequestStateLifecycle lifecycle()
      const noexcept;
  [[nodiscard]] const Sm87BulkV2P40RequestStateSealedAccess* sealed_access()
      const noexcept;
  [[nodiscard]] std::uint64_t request_epoch() const noexcept;
  [[nodiscard]] Sm87BulkV2P40RequestStateRearmResult
  rearm_for_cold_request(
      const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept;

 private:
  struct Impl;
  explicit Sm87BulkV2P40RequestState(std::unique_ptr<Impl> impl) noexcept;
  [[nodiscard]] static Sm87BulkV2P40RequestStateCreateResult create_bound(
      Sm87BulkV2P40RequestStateCudaRuntime* cuda,
      const std::array<void*, kSm87BulkV2P40StreamCount>& streams,
      std::uint64_t owner_identity, std::int32_t expected_device_ordinal,
      Sm87BulkV2P40RequestStateExecutionClass execution_class) noexcept;

  [[nodiscard]] bool access_matches(
      const Sm87BulkV2P40RequestStateSealedAccess& access) const noexcept;
  [[nodiscard]] Sm87BulkV2P40RequestStateStatus begin_request(
      const Sm87BulkV2P40RequestStateSealedAccess& access,
      std::uint64_t request_epoch) noexcept;
  // Enqueues exactly one 8-byte D2H on the owner-bound Main stream. Both ends
  // are fixed and this method is private: only the bound Owner transaction (or
  // an explicitly compiled host fixture) may invoke it.
  [[nodiscard]] Sm87BulkV2P40RequestStateStatus enqueue_handoff_d2h(
      const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept;
  [[nodiscard]] bool owner_begin_binding_valid(
      const Sm87BulkV2P40RequestStateSealedAccess& access,
      std::uint64_t owner_identity,
      std::uint64_t request_allocation_identity,
      std::uint64_t stream_event_owner_identity,
      std::int32_t device_ordinal) const noexcept;
  // Owner-only preflight for the terminal transaction.  It proves the exact
  // capability object, physical owner, request allocation, stream namespace,
  // device and epoch before the owner records its terminal event.  It exposes
  // neither the pinned address nor the observed handoff value.
  [[nodiscard]] bool owner_completion_binding_valid(
      const Sm87BulkV2P40RequestStateSealedAccess& access,
      std::uint64_t owner_identity, std::uint64_t request_epoch,
      std::uint64_t request_allocation_identity,
      std::uint64_t stream_event_owner_identity,
      std::int32_t device_ordinal) const noexcept;
  [[nodiscard]] Sm87BulkV2P40RequestStateStatus
  mark_cancelled_after_owner_drain(
      const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept;
  [[nodiscard]] Sm87BulkV2P40RequestStateStatus
  poison_after_owner_drain(
      const Sm87BulkV2P40RequestStateSealedAccess& access,
      int first_error) noexcept;
  [[nodiscard]] Sm87BulkV2P40RequestStateHandoffResult
  synchronize_terminal_main_and_observe_handoff(
      const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept;

  std::unique_ptr<Impl> impl_;
  std::unique_ptr<Sm87BulkV2P40RequestStateSealedAccess> sealed_access_;

  friend Sm87BulkV2P40RequestStateCreateResult
  create_sm87_bulk_dataflow_v2_p40_request_state(
      Sm87BulkV2P40Owner&) noexcept;
  friend class Sm87BulkV2P40Owner;
#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_REQUEST_STATE_HOST_FIXTURE) || \
    defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_OWNER_HOST_FIXTURE)
  friend class Sm87BulkV2P40RequestStateHostFixture;
#endif
};

struct Sm87BulkV2P40RequestStateCreateResult final {
  std::unique_ptr<Sm87BulkV2P40RequestState> state;
  Sm87BulkV2P40RequestStateStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return state != nullptr && static_cast<bool>(status) &&
           state->lifecycle() == Sm87BulkV2P40RequestStateLifecycle::kReady &&
           state->sealed_access() != nullptr &&
           state->sealed_access()->valid();
  }
};

// Production-shaped resource factory.  The caller supplies an actual v2
// resource owner, never a stream pointer, device ordinal, allocation address,
// or identity.  The owner must still be unsealed so this capability can later
// participate in the composite constituent seal.
[[nodiscard]] Sm87BulkV2P40RequestStateCreateResult
create_sm87_bulk_dataflow_v2_p40_request_state(
    Sm87BulkV2P40Owner& owner) noexcept;

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_REQUEST_STATE_HOST_FIXTURE) || \
    defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_OWNER_HOST_FIXTURE)
class Sm87BulkV2P40RequestStateHostFixture final {
 public:
  [[nodiscard]] static Sm87BulkV2P40RequestStateCreateResult create(
      Sm87BulkV2P40RequestStateCudaRuntime* cuda,
      const std::array<void*, kSm87BulkV2P40StreamCount>& streams,
      std::uint64_t owner_identity = 1U,
      std::int32_t expected_device_ordinal = 0) noexcept;
  [[nodiscard]] static Sm87BulkV2P40RequestStateStatus begin_request(
      Sm87BulkV2P40RequestState& state,
      const Sm87BulkV2P40RequestStateSealedAccess& access,
      std::uint64_t request_epoch) noexcept;
  [[nodiscard]] static Sm87BulkV2P40RequestStateStatus enqueue_handoff_d2h(
      Sm87BulkV2P40RequestState& state,
      const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept;
  [[nodiscard]] static Sm87BulkV2P40RequestStateStatus
  mark_cancelled_after_owner_drain(
      Sm87BulkV2P40RequestState& state,
      const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept;
  [[nodiscard]] static Sm87BulkV2P40RequestStateStatus
  poison_after_owner_drain(
      Sm87BulkV2P40RequestState& state,
      const Sm87BulkV2P40RequestStateSealedAccess& access,
      int first_error) noexcept;
  [[nodiscard]] static Sm87BulkV2P40RequestStateHandoffResult
  synchronize_terminal_main_and_observe_handoff(
      Sm87BulkV2P40RequestState& state,
      const Sm87BulkV2P40RequestStateSealedAccess& access) noexcept;
  static void emulate_completed_handoff_d2h(
      Sm87BulkV2P40RequestState& state, std::uint32_t token_id,
      std::uint32_t nonfinite) noexcept;
};
#endif

[[nodiscard]] const char* to_string(
    Sm87BulkV2P40RequestStateError error) noexcept;

inline constexpr auto kSm87BulkV2P40FrozenRequestArenaLayout =
    sm87_bulk_v2_p40_request_arena_layout();

static_assert(sm87_bulk_v2_p40_request_arena_layout_valid(
    kSm87BulkV2P40FrozenRequestArenaLayout));
static_assert(kSm87BulkV2P40ColdResetBytes == 78'446'592ULL);
static_assert(kSm87BulkV2P40RequestArenaBytes == 5'075'652'608ULL);
static_assert(kSm87BulkV2P40ControlArenaBytes == 1'152ULL);

}  // namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail
