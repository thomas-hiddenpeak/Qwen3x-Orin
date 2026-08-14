#include "../src/runtime/sm87_macrofeed_v4_execution_events_internal.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>

namespace events =
    q3x::runtime::sm87_macrofeed_v4_execution_events_detail;
namespace runtime = q3x::runtime;
using Fixture = events::Sm87MacroFeedV4ExecutionEventsCudaTestFixture;

static_assert(!std::is_default_constructible_v<
              events::Sm87MacroFeedV4ExecutionEventsAccess>);
static_assert(!std::is_copy_constructible_v<
              events::Sm87MacroFeedV4ExecutionEventsAccess>);
static_assert(!std::is_move_constructible_v<
              events::Sm87MacroFeedV4ExecutionEventsAccess>);
static_assert(!std::is_default_constructible_v<
              events::Sm87MacroFeedV4ExecutionPanelAccess>);
static_assert(!std::is_copy_constructible_v<
              events::Sm87MacroFeedV4ExecutionPanelAccess>);
static_assert(!std::is_convertible_v<
              events::Sm87MacroFeedV4EventEnqueueReceipt,
              events::Sm87MacroFeedV4PhysicalCompletionReceipt>);

template <typename T, typename = void>
struct HasPublicAccess : std::false_type {};

template <typename T>
struct HasPublicAccess<
    T, std::void_t<decltype(std::declval<const T&>().access())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasPublicBeginPanel : std::false_type {};

template <typename T>
struct HasPublicBeginPanel<
    T, std::void_t<decltype(std::declval<T&>().begin_panel(
           std::declval<const events::Sm87MacroFeedV4ExecutionEventsAccess&>(),
           std::size_t{}))>> : std::true_type {};

template <typename T, typename = void>
struct HasPublicRecordEvent : std::false_type {};

template <typename T>
struct HasPublicRecordEvent<
    T, std::void_t<decltype(std::declval<T&>().record_event(
           std::declval<const events::Sm87MacroFeedV4ExecutionEventsAccess&>(),
           std::declval<const events::Sm87MacroFeedV4ExecutionPanelAccess&>(),
           events::Sm87MacroFeedV4ExecutionStream::kMain,
           events::Sm87MacroFeedV4ExecutionEvent::kPanelDone))>>
    : std::true_type {};

static_assert(!HasPublicAccess<
              events::Sm87MacroFeedV4ExecutionEventsOwner>::value);
static_assert(!HasPublicBeginPanel<
              events::Sm87MacroFeedV4ExecutionEventsOwner>::value);
static_assert(!HasPublicRecordEvent<
              events::Sm87MacroFeedV4ExecutionEventsOwner>::value);

namespace {

struct Test final {
  int failures = 0;

  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  }
};

struct BoundOwner final {
  events::Sm87MacroFeedV4ExecutionEventsCreateResult execution{};
  runtime::Sm87MacroFeedV4RequestStateCreateResult request{};
  runtime::Sm87MacroFeedV4RequestStateSealedAccess request_access;
  void* recurrent_allocation = nullptr;

  BoundOwner(events::Sm87MacroFeedV4ExecutionEventsCreateResult execution_in,
             runtime::Sm87MacroFeedV4RequestStateCreateResult request_in,
             runtime::Sm87MacroFeedV4RequestStateSealedAccess access_in,
             void* const recurrent_allocation_in)
      : execution(std::move(execution_in)),
        request(std::move(request_in)),
        request_access(access_in),
        recurrent_allocation(recurrent_allocation_in) {}

  ~BoundOwner() {
    execution.owner.reset();
    request.state.reset();
    if (recurrent_allocation != nullptr) {
      (void)cudaFree(recurrent_allocation);
      recurrent_allocation = nullptr;
    }
  }
};

[[nodiscard]] std::unique_ptr<BoundOwner> make_bound_owner(
    Test& test, const std::uint64_t allocation_identity) {
  auto execution =
      events::create_sm87_macrofeed_v4_execution_events_owner();
  test.expect(static_cast<bool>(execution),
              "physical execution-event owner creates on exact SM87/16SM");
  if (!execution) {
    return nullptr;
  }
  const std::uint64_t owner_identity =
      Fixture::owner_identity(*execution.owner);
  test.expect(owner_identity != 0U,
              "test fixture observes a nonzero private owner identity");
  if (owner_identity == 0U) {
    return nullptr;
  }

  void* recurrent_allocation = nullptr;
  const cudaError_t allocation_status = cudaMalloc(
      &recurrent_allocation, runtime::kSm87MacroFeedV4RecurrentStorageBytes);
  test.expect(allocation_status == cudaSuccess &&
                  recurrent_allocation != nullptr,
              "test owner allocates exact dual-epoch recurrent storage");
  if (allocation_status != cudaSuccess || recurrent_allocation == nullptr) {
    return nullptr;
  }
  const auto cold_initialized = Fixture::initialize_cold_recurrent_storage(
      *execution.owner, recurrent_allocation,
      runtime::kSm87MacroFeedV4RecurrentStorageBytes, allocation_identity);
  test.expect(static_cast<bool>(cold_initialized),
              "test owner cold-initializes exact recurrent allocation once");
  if (!cold_initialized) {
    execution.owner.reset();
    (void)cudaFree(recurrent_allocation);
    return nullptr;
  }

  const auto admission =
      runtime::make_sm87_macrofeed_v4_request_state_admission(
          owner_identity, allocation_identity, allocation_identity + 1U,
          allocation_identity + 2U);
  auto request = runtime::Sm87MacroFeedV4RequestState::create(admission);
  test.expect(static_cast<bool>(request),
              "host RequestState binds the same Engine identity");
  if (!request) {
    execution.owner.reset();
    (void)cudaFree(recurrent_allocation);
    return nullptr;
  }
  auto request_access = request.state->issue_sealed_access();
  return std::make_unique<BoundOwner>(std::move(execution),
                                      std::move(request), request_access,
                                      recurrent_allocation);
}

void expect_physical_observation_forbidden(
    Test& test, events::Sm87MacroFeedV4ExecutionEventsOwner& owner,
    const events::Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const events::Sm87MacroFeedV4ExecutionEvent event) {
  const auto query = Fixture::observe_event_query(owner, panel_access, event);
  test.expect(
      !query &&
          query.status.error ==
              events::Sm87MacroFeedV4ExecutionError::
                  kPhysicalObservationForbidden &&
          query.status.event == event,
      "device-order-only event rejects host query observation");
  const auto synchronize =
      Fixture::observe_event_synchronize(owner, panel_access, event);
  test.expect(
      !synchronize &&
          synchronize.status.error ==
              events::Sm87MacroFeedV4ExecutionError::
                  kPhysicalObservationForbidden &&
          synchronize.status.event == event,
      "device-order-only event rejects host synchronize observation");
}

[[nodiscard]] bool run_ab_cycle(
    Test& test, events::Sm87MacroFeedV4ExecutionEventsOwner& owner,
    const events::Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const bool verify_observation_boundary) {
  const auto norm_record = Fixture::record_event(
      owner, panel_access, events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kNormReady);
  test.expect(static_cast<bool>(norm_record), "Main records NormReady");
  if (!norm_record) {
    return false;
  }
  test.expect(!norm_record.receipt.physical_device_completion_attested &&
                  !norm_record.receipt.production_receipt_eligible,
              "record enqueue receipt has no completion authority");
  if (verify_observation_boundary) {
    expect_physical_observation_forbidden(
        test, owner, panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kNormReady);
  }

  const auto norm_wait = Fixture::wait_event(
      owner, panel_access, events::Sm87MacroFeedV4ExecutionStream::kAbAux,
      events::Sm87MacroFeedV4ExecutionEvent::kNormReady);
  const auto ab_record = Fixture::record_event(
      owner, panel_access, events::Sm87MacroFeedV4ExecutionStream::kAbAux,
      events::Sm87MacroFeedV4ExecutionEvent::kAbReady);
  if (verify_observation_boundary && ab_record) {
    expect_physical_observation_forbidden(
        test, owner, panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kAbReady);
  }
  const auto ab_wait = Fixture::wait_event(
      owner, panel_access, events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kAbReady);
  test.expect(static_cast<bool>(norm_wait) && static_cast<bool>(ab_record) &&
                  static_cast<bool>(ab_wait),
              "one fixed NormReady/AbReady device-order cycle enqueues");
  return static_cast<bool>(norm_wait) && static_cast<bool>(ab_record) &&
         static_cast<bool>(ab_wait);
}

[[nodiscard]] bool enqueue_and_close_panel(
    Test& test, events::Sm87MacroFeedV4ExecutionEventsOwner& owner,
    const events::Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const std::size_t expected_completed_panels) {
  for (std::size_t cycle = 0U;
       cycle < events::kSm87MacroFeedV4Bf16AbCyclesPerPanel; ++cycle) {
    if (!run_ab_cycle(test, owner, panel_access,
                      expected_completed_panels == 1U && cycle == 0U)) {
      return false;
    }
  }
  const auto panel_done = Fixture::record_event(
      owner, panel_access, events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kPanelDone);
  test.expect(static_cast<bool>(panel_done) &&
                  !panel_done.receipt.physical_device_completion_attested,
              "PanelDone is a device enqueue marker, not a host receipt");
  if (!panel_done) {
    return false;
  }
  if (expected_completed_panels == 1U) {
    expect_physical_observation_forbidden(
        test, owner, panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kPanelDone);
  }

  const auto closed = Fixture::close_panel(owner, panel_access);
  test.expect(static_cast<bool>(closed),
              "PanelDone enqueue advances host ledger without CUDA wait");
  const auto snapshot = owner.snapshot();
  test.expect(snapshot.completed_panels == expected_completed_panels &&
                  snapshot.completed_panels <=
                      runtime::kSm87MacroFeedV4PanelCount &&
                  snapshot.physical_completion_receipts_issued == 0U,
              "panel close is bounded and issues no physical receipt");

  const auto duplicate = Fixture::close_panel(owner, panel_access);
  test.expect(!duplicate &&
                  owner.snapshot().completed_panels ==
                      expected_completed_panels,
              "one panel generation is consumed exactly once");
  return static_cast<bool>(closed);
}

void test_five_panel_enqueue_without_host_barriers(Test& test) {
  auto bound = make_bound_owner(test, 0x2000U);
  if (bound == nullptr) {
    return;
  }
  auto& owner = *bound->execution.owner;
  test.expect(static_cast<bool>(Fixture::begin_request(
                  owner, *bound->request.state, bound->request_access)),
              "live admitted RequestState begins physical owner request");

  std::unique_ptr<events::Sm87MacroFeedV4ExecutionPanelAccess>
      final_panel_access;
  for (std::size_t panel = 0U;
       panel < runtime::kSm87MacroFeedV4PanelCount; ++panel) {
    auto panel_begin = Fixture::begin_panel(owner, panel);
    test.expect(static_cast<bool>(panel_begin) &&
                    panel_begin.panel_access->panel() == panel,
                "panels begin in fixed zero-to-four order");
    if (!panel_begin) {
      return;
    }
    if (!enqueue_and_close_panel(test, owner, *panel_begin.panel_access,
                                 panel + 1U)) {
      return;
    }
    if (panel + 1U == runtime::kSm87MacroFeedV4PanelCount) {
      final_panel_access = std::move(panel_begin.panel_access);
    }
  }

  test.expect(final_panel_access != nullptr &&
                  owner.snapshot().physical_completion_receipts_issued == 0U,
              "all five panels enqueue and close with no query/synchronize");
  if (final_panel_access == nullptr) {
    return;
  }

  const auto representation = Fixture::record_event(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady);
  const auto representation_join = Fixture::wait_event(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady);
  const auto copy_done = Fixture::record_event(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone);
  const auto copy_join = Fixture::wait_event(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone);
  const auto publish = Fixture::record_event(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kFinalPublish);
  test.expect(static_cast<bool>(representation) &&
                  static_cast<bool>(representation_join) &&
                  static_cast<bool>(copy_done) &&
                  static_cast<bool>(copy_join) &&
                  static_cast<bool>(publish),
              "fixed final representation/copy/publication chain enqueues");
  if (representation) {
    expect_physical_observation_forbidden(
        test, owner, *final_panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady);
  }
  if (copy_done) {
    expect_physical_observation_forbidden(
        test, owner, *final_panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone);
  }

  const auto final_observation = Fixture::observe_event_synchronize(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionEvent::kFinalPublish);
  test.expect(static_cast<bool>(final_observation) &&
                  final_observation.receipt.observed_by_synchronize() &&
                  final_observation.receipt
                      .physical_device_completion_attested() &&
                  !final_observation.receipt.production_receipt_eligible(),
              "FinalPublish alone crosses the physical completion boundary");
  test.expect(Fixture::completion_receipt_matches(
                  owner, *final_panel_access,
                  events::Sm87MacroFeedV4ExecutionEvent::kFinalPublish,
                  final_observation.receipt),
              "owner authenticates exact physical FinalPublish receipt");
  test.expect(static_cast<bool>(Fixture::complete_request(
                  owner, *final_panel_access, final_observation.receipt)),
              "physical FinalPublish receipt completes request");

  const auto snapshot = owner.snapshot();
  test.expect(snapshot.state ==
                  events::Sm87MacroFeedV4ExecutionOwnerState::
                      kRequestCompleted &&
                  snapshot.completed_panels ==
                      runtime::kSm87MacroFeedV4PanelCount &&
                  snapshot.physical_completion_receipts_issued == 1U,
              "complete request retains five panels and one physical receipt");
}

void test_dual_stream_discard_drain(Test& test) {
  auto bound = make_bound_owner(test, 0x4000U);
  if (bound == nullptr) {
    return;
  }
  auto& owner = *bound->execution.owner;
  test.expect(static_cast<bool>(Fixture::begin_request(
                  owner, *bound->request.state, bound->request_access)),
              "discard test begins request");
  auto panel = Fixture::begin_panel(owner, 0U);
  if (!panel) {
    test.expect(false, "discard test begins panel");
    return;
  }

  const auto main_tail = Fixture::record_event(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kMainTail);
  const auto ab_tail = Fixture::record_event(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kAbAux,
      events::Sm87MacroFeedV4ExecutionEvent::kAbTail);
  if (main_tail) {
    expect_physical_observation_forbidden(
        test, owner, *panel.panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kMainTail);
  }
  if (ab_tail) {
    expect_physical_observation_forbidden(
        test, owner, *panel.panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kAbTail);
  }
  const auto main_join = Fixture::wait_event(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kMainTail);
  const auto ab_join = Fixture::wait_event(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kAbTail);
  const auto drained_record = Fixture::record_event(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kOwnerDrained);
  test.expect(static_cast<bool>(main_tail) && static_cast<bool>(ab_tail) &&
                  static_cast<bool>(main_join) &&
                  static_cast<bool>(ab_join) &&
                  static_cast<bool>(drained_record),
              "dual-stream tails join on Control before OwnerDrained");

  const events::Sm87MacroFeedV4PhysicalCompletionReceipt forged;
  test.expect(!Fixture::discard_after_drain(
                  owner, *panel.panel_access, forged),
              "forged completion cannot discard request");
  const auto drained = Fixture::observe_event_synchronize(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionEvent::kOwnerDrained);
  test.expect(static_cast<bool>(drained) &&
                  drained.receipt.main_tail_generation() != 0U &&
                  drained.receipt.ab_tail_generation() != 0U,
              "physical OwnerDrained receipt binds both producer tails");
  test.expect(static_cast<bool>(Fixture::discard_after_drain(
                  owner, *panel.panel_access, drained.receipt)) &&
                  owner.snapshot().state ==
                      events::Sm87MacroFeedV4ExecutionOwnerState::
                          kRequestDiscarded,
              "physically drained request discards safely");
}

void test_request_owner_phase_and_identity_binding(Test& test) {
  auto first = make_bound_owner(test, 0x5000U);
  auto second = make_bound_owner(test, 0x6000U);
  if (first == nullptr || second == nullptr) {
    return;
  }

  const auto foreign_request = Fixture::begin_request(
      *second->execution.owner, *first->request.state, first->request_access);
  test.expect(!foreign_request &&
                  foreign_request.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kForeignRequestAccess,
              "request owner identity must match execution owner");

  const auto copied_access = first->request_access;
  const auto detached_copy = Fixture::begin_request(
      *second->execution.owner, *second->request.state, copied_access);
  test.expect(!detached_copy &&
                  detached_copy.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kForeignRequestAccess,
              "copied sealed access cannot detach from its RequestState");

  test.expect(static_cast<bool>(second->request.state->begin_panel(
                  second->request_access, 0U)),
              "host fixture moves RequestState out of admitted phase");
  const auto wrong_phase = Fixture::begin_request(
      *second->execution.owner, *second->request.state,
      second->request_access);
  test.expect(!wrong_phase &&
                  wrong_phase.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kForeignRequestAccess,
              "execution begin rejects non-admitted RequestState phase");
}

void test_poison_terminal_drain(Test& test) {
  auto bound = make_bound_owner(test, 0x7000U);
  if (bound == nullptr) {
    return;
  }
  auto& owner = *bound->execution.owner;
  test.expect(static_cast<bool>(Fixture::begin_request(
                  owner, *bound->request.state, bound->request_access)),
              "poison-drain test begins request");
  const auto drained = Fixture::inject_poison_and_drain(
      owner, events::Sm87MacroFeedV4ExecutionError::kCudaSubmission);
  const auto snapshot = owner.snapshot();
  test.expect(static_cast<bool>(drained) &&
                  drained.poison_cause.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kCudaSubmission &&
                  snapshot.state ==
                      events::Sm87MacroFeedV4ExecutionOwnerState::kPoisoned &&
                  snapshot.poisoned_terminal_quiescence_attested &&
                  snapshot.poison_cause.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kCudaSubmission &&
                  drained.all_stream_synchronizations_attempted &&
                  drained.discard_required &&
                  snapshot
                      .poison_drain_all_stream_synchronizations_attempted &&
                  snapshot.poison_drain_stream_cuda_status ==
                      drained.stream_cuda_status,
              "terminal three-stream drain attests quiescence and preserves "
              "original CUDA failure");
  const auto forbidden_reentry = Fixture::begin_request(
      owner, *bound->request.state, bound->request_access);
  test.expect(!forbidden_reentry &&
                  forbidden_reentry.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kInvalidOwnerState,
              "physically drained poisoned owner can never admit a new "
              "request");
  const auto repeated_drain = Fixture::inject_poison_and_drain(
      owner, events::Sm87MacroFeedV4ExecutionError::kCudaSubmission);
  test.expect(!repeated_drain &&
                  repeated_drain.drain_status.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kInvalidOwnerState,
              "terminal poison drain is one-shot");
}

[[nodiscard]] bool exact_target_device_available() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return false;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return false;
  }
  return properties.major == 8 && properties.minor == 7 &&
         properties.multiProcessorCount == 16;
}

}  // namespace

int main() {
  if (!exact_target_device_available()) {
    std::cout << "SKIP: exact SM87/16SM device unavailable\n";
    return 77;
  }

  Test test;
  test_five_panel_enqueue_without_host_barriers(test);
  test_dual_stream_discard_drain(test);
  test_request_owner_phase_and_identity_binding(test);
  test_poison_terminal_drain(test);
  if (test.failures != 0) {
    std::cerr << "sm87_macrofeed_v4_execution_events_cuda_test: "
              << test.failures << " failure(s)\n";
    return 1;
  }
  std::cout << "sm87_macrofeed_v4_execution_events_cuda_test: PASS\n";
  return 0;
}
