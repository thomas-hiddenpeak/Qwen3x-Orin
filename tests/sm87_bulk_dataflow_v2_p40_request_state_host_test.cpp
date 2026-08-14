#include "sm87_bulk_dataflow_v2_p40_request_state_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace owner =
    q3x::runtime::sm87_bulk_v2_p40_owner_detail;
namespace runtime = q3x::runtime;

static_assert(!std::is_default_constructible_v<
              owner::Sm87BulkV2P40RequestStateSealedAccess>);
static_assert(!std::is_copy_constructible_v<
              owner::Sm87BulkV2P40RequestStateSealedAccess>);
static_assert(!std::is_move_constructible_v<
              owner::Sm87BulkV2P40RequestStateSealedAccess>);
static_assert(!std::is_default_constructible_v<
              owner::Sm87BulkV2P40RequestState>);
static_assert(!std::is_copy_constructible_v<
              owner::Sm87BulkV2P40RequestState>);
static_assert(!std::is_move_constructible_v<
              owner::Sm87BulkV2P40RequestState>);
namespace {

struct TestContext final {
  int failures = 0;

  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  }
};

struct MemsetCall final {
  void* pointer = nullptr;
  int value = -1;
  std::size_t bytes = 0U;
  void* stream = nullptr;
};

class FakeRequestStateCudaRuntime final
    : public owner::Sm87BulkV2P40RequestStateCudaRuntime {
 public:
  std::int32_t current_device = 0;
  owner::Sm87BulkV2P40RequestDeviceProperties properties{8, 7, 16};
  void* device_pointer = reinterpret_cast<void*>(0x1'0000'0000ULL);
  unsigned int stream_flags =
      owner::kSm87BulkV2P40NonBlockingStreamFlag;
  int get_device_error = 0;
  int get_properties_error = 0;
  int get_stream_flags_error = 0;
  int allocate_device_error = 0;
  int allocate_pinned_error = 0;
  int query_device_error = 0;
  int query_pinned_error = 0;
  int next_memset_error = 0;
  int next_copy_error = 0;
  int next_synchronize_error = 0;
  bool report_device_as_host = false;
  bool report_pinned_as_device = false;

  std::size_t get_device_calls = 0U;
  std::size_t get_properties_calls = 0U;
  std::size_t get_stream_flags_calls = 0U;
  std::size_t allocate_device_calls = 0U;
  std::size_t free_device_calls = 0U;
  std::size_t allocate_pinned_calls = 0U;
  std::size_t free_pinned_calls = 0U;
  std::size_t query_pointer_calls = 0U;
  std::size_t synchronize_calls = 0U;
  std::size_t copy_device_to_host_calls = 0U;
  const void* last_copy_device_source = nullptr;
  std::vector<std::size_t> device_allocation_bytes;
  std::vector<std::size_t> pinned_allocation_bytes;
  std::vector<MemsetCall> memset_calls;
  std::vector<void*> synchronized_streams;
  owner::Sm87BulkV2P40PinnedHandoff* pinned_pointer = nullptr;

  [[nodiscard]] std::size_t static_query_calls() const noexcept {
    return get_device_calls + get_properties_calls +
           get_stream_flags_calls + query_pointer_calls;
  }

  [[nodiscard]] int get_current_device(
      std::int32_t* const device_ordinal) noexcept override {
    ++get_device_calls;
    if (get_device_error == 0 && device_ordinal != nullptr) {
      *device_ordinal = current_device;
    }
    return get_device_error;
  }

  [[nodiscard]] int get_device_properties(
      const std::int32_t,
      owner::Sm87BulkV2P40RequestDeviceProperties* const output)
      noexcept override {
    ++get_properties_calls;
    if (get_properties_error == 0 && output != nullptr) {
      *output = properties;
    }
    return get_properties_error;
  }

  [[nodiscard]] int get_stream_flags(
      void* const, unsigned int* const flags) noexcept override {
    ++get_stream_flags_calls;
    if (get_stream_flags_error == 0 && flags != nullptr) {
      *flags = stream_flags;
    }
    return get_stream_flags_error;
  }

  [[nodiscard]] int allocate_device(
      void** const pointer, const std::size_t bytes) noexcept override {
    ++allocate_device_calls;
    device_allocation_bytes.push_back(bytes);
    if (allocate_device_error == 0 && pointer != nullptr) {
      *pointer = device_pointer;
    }
    return allocate_device_error;
  }

  [[nodiscard]] int free_device(void* const pointer) noexcept override {
    if (pointer == device_pointer) {
      ++free_device_calls;
    }
    return 0;
  }

  [[nodiscard]] int allocate_pinned_host(
      void** const pointer, const std::size_t bytes) noexcept override {
    ++allocate_pinned_calls;
    pinned_allocation_bytes.push_back(bytes);
    if (allocate_pinned_error != 0 || pointer == nullptr) {
      return allocate_pinned_error;
    }
    pinned_pointer =
        new (std::nothrow) owner::Sm87BulkV2P40PinnedHandoff();
    if (pinned_pointer == nullptr) {
      return 2;
    }
    *pointer = pinned_pointer;
    return 0;
  }

  [[nodiscard]] int free_pinned_host(void* const pointer) noexcept override {
    if (pointer == pinned_pointer && pinned_pointer != nullptr) {
      ++free_pinned_calls;
      delete pinned_pointer;
      pinned_pointer = nullptr;
    }
    return 0;
  }

  [[nodiscard]] int query_pointer(
      const void* const pointer,
      owner::Sm87BulkV2P40RequestPointerAttributes* const attributes)
      noexcept override {
    ++query_pointer_calls;
    if (attributes == nullptr) {
      return 2;
    }
    if (pointer == device_pointer) {
      if (query_device_error != 0) {
        return query_device_error;
      }
      attributes->kind =
          report_device_as_host
              ? owner::Sm87BulkV2P40RequestPointerKind::kHost
              : owner::Sm87BulkV2P40RequestPointerKind::kDevice;
      attributes->device_pointer = device_pointer;
      attributes->device_ordinal = current_device;
      return 0;
    }
    if (pointer == pinned_pointer) {
      if (query_pinned_error != 0) {
        return query_pinned_error;
      }
      attributes->kind =
          report_pinned_as_device
              ? owner::Sm87BulkV2P40RequestPointerKind::kDevice
              : owner::Sm87BulkV2P40RequestPointerKind::kHost;
      attributes->host_pointer = pinned_pointer;
      return 0;
    }
    return 2;
  }

  [[nodiscard]] int memset_async(void* const pointer, const int value,
                                 const std::size_t bytes,
                                 void* const stream) noexcept override {
    memset_calls.push_back({pointer, value, bytes, stream});
    const int result = next_memset_error;
    next_memset_error = 0;
    return result;
  }

  [[nodiscard]] int copy_device_to_host_async(
      void* const host_destination, const void* const device_source,
      const std::size_t bytes, void* const stream) noexcept override {
    ++copy_device_to_host_calls;
    last_copy_device_source = device_source;
    if (host_destination != pinned_pointer || device_source == nullptr ||
        bytes != sizeof(owner::Sm87BulkV2P40PinnedHandoff) ||
        stream == nullptr) {
      return 2;
    }
    const int result = next_copy_error;
    next_copy_error = 0;
    return result;
  }

  [[nodiscard]] int synchronize_stream(
      void* const stream) noexcept override {
    ++synchronize_calls;
    synchronized_streams.push_back(stream);
    const int result = next_synchronize_error;
    next_synchronize_error = 0;
    return result;
  }
};

[[nodiscard]] std::array<void*, runtime::kSm87BulkV2P40StreamCount>
streams(const std::uintptr_t base = 0x20'0000ULL) noexcept {
  std::array<void*, runtime::kSm87BulkV2P40StreamCount> result{};
  for (std::size_t index = 0U; index < result.size(); ++index) {
    result[index] = reinterpret_cast<void*>(base + (index + 1U) * 0x100U);
  }
  return result;
}

void test_frozen_layout(TestContext& test) {
  constexpr auto layout = owner::sm87_bulk_v2_p40_request_arena_layout();
  test.expect(owner::sm87_bulk_v2_p40_request_arena_layout_valid(layout),
              "canonical request arena layout validates");
  test.expect(layout.arena_bytes == 5'075'652'608ULL &&
                  layout.cold_reset_bytes == 78'446'592ULL &&
                  layout.separately_owned_control_bytes == 1'152ULL &&
                  layout.pinned_handoff_bytes == 8U,
              "data, cold-reset, external-control, and handoff sizes are exact");
  test.expect(layout.bindings[0U].range.offset == 0U &&
                  layout.bindings[0U].range.bytes ==
                      runtime::kSm87BulkV2P40PersistentBytes &&
                  layout.bindings[1U].range.offset == 0U &&
                  layout.bindings[1U].range.bytes ==
                      runtime::kSm87BulkV2P40ColdResetBytes &&
                  layout.bindings[2U].range.offset ==
                      runtime::kSm87BulkV2P40PersistentBytes &&
                  layout.bindings[3U].range.offset ==
                      runtime::kSm87BulkV2P40FamilyArenaOffset &&
                  layout.bindings[4U].range.end() == layout.arena_bytes,
              "top-level ranges exactly tile the frozen v2 data allocation");
  test.expect(layout.cold_reset_bytes ==
                  runtime::kSm87BulkV2P40GdnLayers *
                      (runtime::kSm87BulkV2P40GdnStateBytesPerLayer +
                       runtime::kSm87BulkV2P40GdnHistoryBytesPerLayer),
              "cold reset is exactly 48 GDN state/history pairs");

  auto changed = layout;
  changed.bindings[4U].range.offset -= 256U;
  test.expect(!owner::sm87_bulk_v2_p40_request_arena_layout_valid(changed),
              "overlap or drift from the frozen range map fails closed");
  changed = layout;
  changed.cold_reset_bytes = layout.arena_bytes;
  test.expect(!owner::sm87_bulk_v2_p40_request_arena_layout_valid(changed),
              "a whole-arena reset cannot be relabeled as cold state reset");
}

void test_startup_single_allocation_and_partial_clear(TestContext& test) {
  FakeRequestStateCudaRuntime cuda;
  const auto owned_streams = streams();
  {
    auto created = owner::Sm87BulkV2P40RequestStateHostFixture::create(
        &cuda, owned_streams, 71U, 0);
    test.expect(static_cast<bool>(created),
                "synthetic host fixture mints one immutable request state");
    if (!created) {
      return;
    }
    const auto* const access = created.state->sealed_access();
    test.expect(access != nullptr && access->valid() &&
                    !access->default_off_development_resource_valid() &&
                    access->identity().synthetic_host_contract_valid(),
                "host fixture stays synthetic and cannot claim real development admission");
    test.expect(cuda.allocate_device_calls == 1U &&
                    cuda.device_allocation_bytes.size() == 1U &&
                    cuda.device_allocation_bytes[0U] ==
                        runtime::kSm87BulkV2P40RequestArenaBytes &&
                    cuda.allocate_pinned_calls == 1U &&
                    cuda.pinned_allocation_bytes[0U] == 8U,
                "startup owns exactly one 5.075-GB device allocation and one 8-byte pinned handoff");
    test.expect(cuda.device_allocation_bytes[0U] !=
                    runtime::kSm87BulkV2P40ControlArenaBytes,
                "request state does not duplicate the owner-managed 1152-byte control plane");
    test.expect(cuda.memset_calls.size() == 1U &&
                    cuda.memset_calls[0U].pointer == cuda.device_pointer &&
                    cuda.memset_calls[0U].value == 0 &&
                    cuda.memset_calls[0U].bytes ==
                        runtime::kSm87BulkV2P40ColdResetBytes &&
                    cuda.memset_calls[0U].bytes !=
                        runtime::kSm87BulkV2P40RequestArenaBytes &&
                    cuda.memset_calls[0U].stream == owned_streams[0U] &&
                    cuda.synchronize_calls == 1U,
                "startup clears and waits only for the GDN persistent prefix on Main");
    test.expect(access->arena_span(
                    owner::Sm87BulkV2P40RequestArenaRole::kPersistent) ==
                    cuda.device_pointer &&
                    access->arena_span_bytes(
                        owner::Sm87BulkV2P40RequestArenaRole::kFamily) ==
                        runtime::kSm87BulkV2P40FamilyArenaBytes &&
                    access->cuda_stream(runtime::Sm87BulkV2P40Stream::kMain) ==
                        owned_streams[0U],
                "sealed access binds exact arena spans and borrowed Main without exposing pinned host storage");
  }
  test.expect(cuda.free_device_calls == 1U && cuda.free_pinned_calls == 1U,
              "quiescent owner teardown releases exactly its two owned resources");
}

void test_hot_rearm_and_terminal_handoff(TestContext& test) {
  FakeRequestStateCudaRuntime cuda;
  auto created = owner::Sm87BulkV2P40RequestStateHostFixture::create(
      &cuda, streams(), 81U, 0);
  test.expect(static_cast<bool>(created), "hot lifecycle fixture creates");
  if (!created) {
    return;
  }
  auto& state = *created.state;
  const auto& access = *state.sealed_access();
  const auto identity = access.identity();
  const std::size_t startup_queries = cuda.static_query_calls();

  test.expect(static_cast<bool>(
                  owner::Sm87BulkV2P40RequestStateHostFixture::begin_request(
                      state, access, 100U)),
              "first fresh request epoch begins");
  const std::size_t memset_before_active_rearm = cuda.memset_calls.size();
  const auto active_rearm = state.rearm_for_cold_request(access);
  test.expect(!active_rearm &&
                  active_rearm.status.error ==
                      owner::Sm87BulkV2P40RequestStateError::
                          kInvalidLifecycle &&
                  cuda.memset_calls.size() == memset_before_active_rearm,
              "active request rearm fails without mutating device state");

  test.expect(static_cast<bool>(
                  owner::Sm87BulkV2P40RequestStateHostFixture::
                      mark_cancelled_after_owner_drain(state, access)),
              "owner-drained cancellation keeps state unpublished");
  const std::size_t sync_before_rearm = cuda.synchronize_calls;
  const auto rearmed = state.rearm_for_cold_request(access);
  test.expect(static_cast<bool>(rearmed) &&
                  cuda.memset_calls.size() == 2U &&
                  cuda.memset_calls.back().bytes ==
                      runtime::kSm87BulkV2P40ColdResetBytes &&
                  cuda.synchronize_calls == sync_before_rearm &&
                  cuda.static_query_calls() == startup_queries &&
                  rearmed.request_hot_static_cuda_queries == 0U &&
                  rearmed.zeroed_device_bytes == 78'446'592ULL &&
                  !rearmed.whole_arena_reset &&
                  !rearmed.control_plane_touched &&
                  rearmed.allocation_identity ==
                      identity.allocation_identity,
              "hot rearm enqueues only the local GDN reset with zero static CUDA queries or host wait");
  test.expect(!owner::Sm87BulkV2P40RequestStateHostFixture::begin_request(
                  state, access, 100U) &&
                  state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kReady,
              "stale request epoch cannot consume the rearmed state");
  test.expect(static_cast<bool>(
                  owner::Sm87BulkV2P40RequestStateHostFixture::begin_request(
                      state, access, 101U)),
              "strictly newer owner request epoch begins");

  const std::size_t sync_before_missing_d2h = cuda.synchronize_calls;
  const auto missing_d2h = owner::Sm87BulkV2P40RequestStateHostFixture::
      synchronize_terminal_main_and_observe_handoff(state, access);
  test.expect(!missing_d2h &&
                  missing_d2h.status.error ==
                      owner::Sm87BulkV2P40RequestStateError::
                          kHandoffNotEnqueued &&
                  cuda.synchronize_calls == sync_before_missing_d2h &&
                  state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kActive,
              "terminal observation fails before sync when no owner-bound D2H was enqueued");
  test.expect(static_cast<bool>(owner::Sm87BulkV2P40RequestStateHostFixture::
                                    enqueue_handoff_d2h(state, access)) &&
                  cuda.copy_device_to_host_calls == 1U &&
                  cuda.last_copy_device_source == reinterpret_cast<void*>(
                      reinterpret_cast<std::uintptr_t>(cuda.device_pointer) +
                      runtime::kSm87BulkV2P40FamilyArenaOffset +
                      runtime::kSm87BulkV2P40FinalLogitsBytes),
              "executor can enqueue one fixed 8-byte D2H whose source is the frozen final-greedy range");
  test.expect(!owner::Sm87BulkV2P40RequestStateHostFixture::
                  enqueue_handoff_d2h(state, access) &&
                  cuda.copy_device_to_host_calls == 1U,
              "a request cannot enqueue a second handoff into the same pinned slot");
  owner::Sm87BulkV2P40RequestStateHostFixture::
      emulate_completed_handoff_d2h(state, 42U, 0U, 0x3f80U);
  const std::size_t sync_before_handoff = cuda.synchronize_calls;
  const auto handoff = owner::Sm87BulkV2P40RequestStateHostFixture::
      synchronize_terminal_main_and_observe_handoff(state, access);
  test.expect(static_cast<bool>(handoff) && handoff.token_id == 42U &&
                  handoff.value_bits == 0x3f80U &&
                  handoff.nonfinite == 0U &&
                  handoff.terminal_main_synchronized &&
                  handoff.handoff_observed && handoff.state_committed &&
                  cuda.synchronize_calls == sync_before_handoff + 1U &&
                  cuda.synchronized_streams.back() == streams()[0U] &&
                  state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kCompleted,
              "8-byte handoff is first observed after exactly one terminal Main synchronization");
  test.expect(cuda.static_query_calls() == startup_queries,
              "begin, rearm, and terminal observation issue no static CUDA query");
}

void test_invalid_handoff_and_sync_failure_poison(TestContext& test) {
  const auto exercise_invalid = [&test](const std::uint32_t token,
                                        const std::uint32_t nonfinite,
                                        const char* const message) {
    FakeRequestStateCudaRuntime cuda;
    auto created = owner::Sm87BulkV2P40RequestStateHostFixture::create(
        &cuda, streams(), 91U, 0);
    if (!created) {
      test.expect(false, "invalid handoff fixture creates");
      return;
    }
    auto& state = *created.state;
    const auto& access = *state.sealed_access();
    test.expect(static_cast<bool>(
                    owner::Sm87BulkV2P40RequestStateHostFixture::begin_request(
                        state, access, 1U)),
                "invalid handoff fixture begins");
    test.expect(static_cast<bool>(owner::Sm87BulkV2P40RequestStateHostFixture::
                                      enqueue_handoff_d2h(state, access)),
                "invalid handoff fixture enqueues fixed D2H");
    owner::Sm87BulkV2P40RequestStateHostFixture::
        emulate_completed_handoff_d2h(state, token, nonfinite);
    const auto observed = owner::Sm87BulkV2P40RequestStateHostFixture::
        synchronize_terminal_main_and_observe_handoff(state, access);
    test.expect(!observed && observed.terminal_main_synchronized &&
                    observed.handoff_observed && !observed.state_committed &&
                    state.lifecycle() ==
                        owner::Sm87BulkV2P40RequestStateLifecycle::kPoisoned &&
                    !state.rearm_for_cold_request(access),
                message);
  };
  exercise_invalid(runtime::kSm87BulkV2P40Vocabulary, 0U,
                   "out-of-vocabulary handoff poisons only after terminal observation");
  exercise_invalid(1U, 1U,
                   "non-finite handoff poisons only after terminal observation");

  FakeRequestStateCudaRuntime cuda;
  auto created = owner::Sm87BulkV2P40RequestStateHostFixture::create(
      &cuda, streams(), 92U, 0);
  test.expect(static_cast<bool>(created), "terminal failure fixture creates");
  if (!created) {
    return;
  }
  auto& state = *created.state;
  const auto& access = *state.sealed_access();
  test.expect(static_cast<bool>(
                  owner::Sm87BulkV2P40RequestStateHostFixture::begin_request(
                      state, access, 2U)),
              "terminal failure fixture begins");
  test.expect(static_cast<bool>(owner::Sm87BulkV2P40RequestStateHostFixture::
                                    enqueue_handoff_d2h(state, access)),
              "terminal failure fixture enqueues fixed D2H");
  owner::Sm87BulkV2P40RequestStateHostFixture::
      emulate_completed_handoff_d2h(state, 2U, 0U);
  cuda.next_synchronize_error = 77;
  const auto failed = owner::Sm87BulkV2P40RequestStateHostFixture::
      synchronize_terminal_main_and_observe_handoff(state, access);
  test.expect(!failed &&
                  failed.status.error ==
                      owner::Sm87BulkV2P40RequestStateError::
                          kTerminalSynchronize &&
                  !failed.terminal_main_synchronized &&
                  !failed.handoff_observed && !failed.state_committed &&
                  state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kPoisoned,
              "failed terminal Main sync never reads or commits pinned storage");
}

void test_foreign_access_and_startup_fail_closed(TestContext& test) {
  FakeRequestStateCudaRuntime cuda_a;
  FakeRequestStateCudaRuntime cuda_b;
  auto first = owner::Sm87BulkV2P40RequestStateHostFixture::create(
      &cuda_a, streams(0x30'0000ULL), 101U, 0);
  auto second = owner::Sm87BulkV2P40RequestStateHostFixture::create(
      &cuda_b, streams(0x40'0000ULL), 102U, 0);
  test.expect(static_cast<bool>(first) && static_cast<bool>(second),
              "two independent request owners create");
  if (first && second) {
    const auto status =
        owner::Sm87BulkV2P40RequestStateHostFixture::begin_request(
            *first.state, *second.state->sealed_access(), 1U);
    test.expect(!status &&
                    status.error ==
                        owner::Sm87BulkV2P40RequestStateError::kForeignAccess &&
                    first.state->lifecycle() ==
                        owner::Sm87BulkV2P40RequestStateLifecycle::kReady,
                "numeric identities cannot substitute a foreign capability object");
  }

  FakeRequestStateCudaRuntime wrong_device;
  wrong_device.properties.minor = 6;
  auto rejected = owner::Sm87BulkV2P40RequestStateHostFixture::create(
      &wrong_device, streams(), 103U, 0);
  test.expect(!rejected &&
                  rejected.status.error ==
                      owner::Sm87BulkV2P40RequestStateError::kWrongDevice &&
                  wrong_device.allocate_device_calls == 0U,
              "non-SM87 startup fails before allocation");

  FakeRequestStateCudaRuntime wrong_stream;
  wrong_stream.stream_flags = 0U;
  rejected = owner::Sm87BulkV2P40RequestStateHostFixture::create(
      &wrong_stream, streams(), 104U, 0);
  test.expect(!rejected &&
                  rejected.status.error ==
                      owner::Sm87BulkV2P40RequestStateError::
                          kStreamValidation &&
                  wrong_stream.allocate_device_calls == 0U,
              "blocking or foreign stream contract fails before allocation");

  FakeRequestStateCudaRuntime wrong_pointer;
  wrong_pointer.report_device_as_host = true;
  rejected = owner::Sm87BulkV2P40RequestStateHostFixture::create(
      &wrong_pointer, streams(), 105U, 0);
  test.expect(!rejected &&
                  rejected.status.error ==
                      owner::Sm87BulkV2P40RequestStateError::
                          kDevicePointerValidation &&
                  wrong_pointer.free_device_calls == 1U,
              "wrong device pointer kind releases the single allocation and fails closed");

  FakeRequestStateCudaRuntime reset_failure;
  reset_failure.next_memset_error = 88;
  rejected = owner::Sm87BulkV2P40RequestStateHostFixture::create(
      &reset_failure, streams(), 106U, 0);
  test.expect(!rejected &&
                  rejected.status.error ==
                      owner::Sm87BulkV2P40RequestStateError::
                          kColdResetSubmission &&
                  reset_failure.free_device_calls == 1U &&
                  reset_failure.free_pinned_calls == 1U,
              "failed partial reset releases both owned resources without exposing access");
}

void test_active_destruction_drains_borrowed_streams(TestContext& test) {
  FakeRequestStateCudaRuntime cuda;
  const auto owned_streams = streams(0x50'0000ULL);
  {
    auto created = owner::Sm87BulkV2P40RequestStateHostFixture::create(
        &cuda, owned_streams, 111U, 0);
    test.expect(static_cast<bool>(created),
                "active destruction fixture creates");
    if (!created) {
      return;
    }
    test.expect(static_cast<bool>(
                    owner::Sm87BulkV2P40RequestStateHostFixture::begin_request(
                        *created.state, *created.state->sealed_access(), 1U)),
                "active destruction fixture begins");
  }
  test.expect(cuda.synchronize_calls ==
                  1U + runtime::kSm87BulkV2P40StreamCount &&
                  cuda.free_pinned_calls == 1U &&
                  cuda.free_device_calls == 1U,
              "active teardown drains all five borrowed streams before releasing request-owned memory");
}

}  // namespace

int main() {
  TestContext test;
  test_frozen_layout(test);
  test_startup_single_allocation_and_partial_clear(test);
  test_hot_rearm_and_terminal_handoff(test);
  test_invalid_handoff_and_sync_failure_poison(test);
  test_foreign_access_and_startup_fail_closed(test);
  test_active_destruction_drains_borrowed_streams(test);
  if (test.failures != 0) {
    std::cerr << test.failures << " request-state host checks failed\n";
    return 1;
  }
  std::cout << "SM87 bulk-dataflow v2 P40 request-state host checks passed\n";
  return 0;
}
