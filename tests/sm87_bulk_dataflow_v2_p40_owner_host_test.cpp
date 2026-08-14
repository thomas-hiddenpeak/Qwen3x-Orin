#include "sm87_bulk_dataflow_v2_p40_constituent_seal_internal.h"
#include "sm87_bulk_dataflow_v2_p40_request_state_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

namespace owner =
    q3x::runtime::sm87_bulk_v2_p40_owner_detail;
namespace runtime = q3x::runtime;
namespace kernels = q3x::kernels;

namespace {

static_assert(!std::is_default_constructible_v<
              owner::Sm87BulkV2P40ConstituentSealAccess>);
static_assert(!std::is_copy_constructible_v<
              owner::Sm87BulkV2P40ConstituentSealAccess>);
static_assert(!std::is_default_constructible_v<
              owner::Sm87BulkV2P40ExecutionAccess>);
static_assert(!std::is_copy_constructible_v<
              owner::Sm87BulkV2P40ExecutionAccess>);
static_assert(!std::is_constructible_v<
              owner::Sm87BulkV2P40ExecutionAccess,
              runtime::Sm87BulkV2P40SealedExecutionAccess>);
static_assert(!std::is_constructible_v<
              owner::Sm87BulkV2P40ConstituentSealAccess,
              owner::Sm87BulkV2P40DevelopmentAdmissionEvidence>);
static_assert(!std::is_default_constructible_v<
              owner::Sm87BulkV2P40WholeProjectionStartupAccess>);
static_assert(!std::is_copy_constructible_v<
              owner::Sm87BulkV2P40WholeProjectionStartupAccess>);
static_assert(!std::is_default_constructible_v<
              owner::Sm87BulkV2P40WholeProjectionStartupRoot>);
static_assert(!std::is_constructible_v<
              owner::Sm87BulkV2P40WholeProjectionStartupAccess,
              owner::Sm87BulkV2P40WholeProjectionStartupObservations>);
static_assert(static_cast<std::uint8_t>(
                  owner::Sm87BulkV2P40ExecutionClass::
                      kDefaultOffDevelopmentCandidate) == 1U);
static_assert(static_cast<std::uint8_t>(
                  owner::Sm87BulkV2P40ExecutionClass::
                      kSyntheticHostContract) == 2U);
static_assert(static_cast<std::uint8_t>(
                  owner::Sm87BulkV2P40ExecutionClass::
                      kDefaultOffDirectionWitness) == 3U);
using ProductionCompleteMethod = owner::Sm87BulkV2P40OwnerStatus (
    owner::Sm87BulkV2P40Owner::*)(
        const owner::Sm87BulkV2P40ExecutionAccess&) noexcept;
static_assert(std::is_same_v<
              decltype(static_cast<ProductionCompleteMethod>(
                  &owner::Sm87BulkV2P40Owner::complete_request)),
              ProductionCompleteMethod>);
using OwnerBoundCompleteMethod = owner::Sm87BulkV2P40OwnerStatus (
    owner::Sm87BulkV2P40Owner::*)(
        const owner::Sm87BulkV2P40ExecutionAccess&,
        owner::Sm87BulkV2P40RequestState&,
        const owner::Sm87BulkV2P40RequestStateSealedAccess&) noexcept;
static_assert(std::is_same_v<
              decltype(static_cast<OwnerBoundCompleteMethod>(
                  &owner::Sm87BulkV2P40Owner::complete_request)),
              OwnerBoundCompleteMethod>);

class TestContext final {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

class FakeCudaRuntime final : public owner::Sm87BulkV2P40CudaRuntime {
 public:
  ~FakeCudaRuntime() override {
    delete[] device_allocation_;
    delete cancellation_host_;
    for (void* const stream : live_streams_) {
      delete static_cast<std::uint8_t*>(stream);
    }
    for (void* const event : live_events_) {
      delete static_cast<std::uint8_t*>(event);
    }
  }

  [[nodiscard]] int get_current_device(
      std::int32_t* const device_ordinal) noexcept override {
    ++static_query_calls;
    if (device_ordinal == nullptr) {
      return kInvalid;
    }
    *device_ordinal = 0;
    return 0;
  }

  [[nodiscard]] int get_device_properties(
      const std::int32_t device_ordinal,
      owner::Sm87BulkV2P40DeviceProperties* const properties)
      noexcept override {
    ++static_query_calls;
    if (device_ordinal != 0 || properties == nullptr) {
      return kInvalid;
    }
    *properties = {8, 7, 16};
    return 0;
  }

  [[nodiscard]] int create_nonblocking_stream(
      void** const stream) noexcept override {
    if (stream == nullptr) {
      return kInvalid;
    }
    auto* const handle = new (std::nothrow) std::uint8_t{0U};
    if (handle == nullptr) {
      return kOutOfMemory;
    }
    *stream = handle;
    live_streams_[stream_create_calls] = handle;
    ++stream_create_calls;
    return 0;
  }

  [[nodiscard]] int get_stream_flags(
      void* const stream, unsigned int* const flags) noexcept override {
    ++static_query_calls;
    if (stream == nullptr || flags == nullptr) {
      return kInvalid;
    }
    *flags = owner::kSm87BulkV2P40NonBlockingStreamFlag;
    return 0;
  }

  [[nodiscard]] int destroy_stream(void* const stream) noexcept override {
    if (stream == nullptr) {
      return kInvalid;
    }
    for (void*& live : live_streams_) {
      if (live == stream) {
        delete static_cast<std::uint8_t*>(live);
        live = nullptr;
        ++stream_destroy_calls;
        return 0;
      }
    }
    return kInvalid;
  }

  [[nodiscard]] int create_disable_timing_event(
      void** const event) noexcept override {
    if (event == nullptr) {
      return kInvalid;
    }
    if (event_create_calls == fail_event_create_ordinal) {
      return kInjected;
    }
    auto* const handle = new (std::nothrow) std::uint8_t{0U};
    if (handle == nullptr) {
      return kOutOfMemory;
    }
    *event = handle;
    live_events_[event_create_calls] = handle;
    ++event_create_calls;
    return 0;
  }

  [[nodiscard]] int destroy_event(void* const event) noexcept override {
    if (event == nullptr) {
      return kInvalid;
    }
    for (void*& live : live_events_) {
      if (live == event) {
        delete static_cast<std::uint8_t*>(live);
        live = nullptr;
        ++event_destroy_calls;
        return 0;
      }
    }
    return kInvalid;
  }

  [[nodiscard]] int allocate_device(
      void** const pointer, const std::size_t bytes) noexcept override {
    if (pointer == nullptr || bytes != runtime::kSm87BulkV2P40ControlArenaBytes ||
        device_allocation_ != nullptr) {
      return kInvalid;
    }
    device_allocation_ = new (std::nothrow) std::uint8_t[bytes];
    if (device_allocation_ == nullptr) {
      return kOutOfMemory;
    }
    *pointer = device_allocation_;
    ++device_allocate_calls;
    return 0;
  }

  [[nodiscard]] int free_device(void* const pointer) noexcept override {
    if (pointer != device_allocation_) {
      return kInvalid;
    }
    delete[] device_allocation_;
    device_allocation_ = nullptr;
    ++device_free_calls;
    return 0;
  }

  [[nodiscard]] int allocate_mapped_host(
      void** const pointer, const std::size_t bytes) noexcept override {
    if (pointer == nullptr || bytes != sizeof(std::uint32_t) ||
        cancellation_host_ != nullptr) {
      return kInvalid;
    }
    cancellation_host_ = new (std::nothrow) std::uint32_t{0U};
    if (cancellation_host_ == nullptr) {
      return kOutOfMemory;
    }
    *pointer = cancellation_host_;
    ++host_allocate_calls;
    return 0;
  }

  [[nodiscard]] int mapped_device_alias(
      void** const device_alias, void* const host_pointer) noexcept override {
    if (device_alias == nullptr || host_pointer != cancellation_host_) {
      return kInvalid;
    }
    *device_alias = cancellation_host_;
    return 0;
  }

  [[nodiscard]] int free_mapped_host(void* const pointer) noexcept override {
    if (pointer != cancellation_host_) {
      return kInvalid;
    }
    delete cancellation_host_;
    cancellation_host_ = nullptr;
    ++host_free_calls;
    return 0;
  }

  [[nodiscard]] int query_pointer(
      const void* const pointer,
      owner::Sm87BulkV2P40PointerAttributes* const attributes)
      noexcept override {
    ++static_query_calls;
    if (attributes == nullptr) {
      return kInvalid;
    }
    if (pointer == device_allocation_) {
      *attributes = {owner::Sm87BulkV2P40PointerKind::kDevice, nullptr,
                     device_allocation_, 0};
      return 0;
    }
    if (pointer == cancellation_host_) {
      *attributes = {owner::Sm87BulkV2P40PointerKind::kHost,
                     cancellation_host_,
                     bad_mapped_pair ? nullptr : cancellation_host_, 0};
      return 0;
    }
    return kInvalid;
  }

  [[nodiscard]] int memset_async(void* const pointer, const int value,
                                 const std::size_t bytes,
                                 void* const stream) noexcept override {
    if (pointer != device_allocation_ || bytes !=
            runtime::kSm87BulkV2P40ControlArenaBytes ||
        stream == nullptr || value != 0) {
      return kInvalid;
    }
    for (std::size_t index = 0U; index < bytes; ++index) {
      device_allocation_[index] = 0U;
    }
    ++memset_calls;
    return fail_next_memset ? (fail_next_memset = false, kInjected) : 0;
  }

  [[nodiscard]] int record_event(void* const event,
                                 void* const stream) noexcept override {
    if (event == nullptr || stream == nullptr) {
      return kInvalid;
    }
    ++event_record_calls;
    return 0;
  }

  [[nodiscard]] int stream_wait_event(void* const stream,
                                      void* const event) noexcept override {
    if (stream == nullptr || event == nullptr) {
      return kInvalid;
    }
    ++event_wait_calls;
    return 0;
  }

  [[nodiscard]] int synchronize_stream(void* const stream) noexcept override {
    if (stream == nullptr) {
      return kInvalid;
    }
    ++stream_sync_calls;
    return 0;
  }

  [[nodiscard]] std::uint32_t cancellation_value() const noexcept {
    return cancellation_host_ == nullptr ? 0U : *cancellation_host_;
  }

  static constexpr int kInvalid = 1;
  static constexpr int kOutOfMemory = 2;
  static constexpr int kInjected = 719;

  std::size_t fail_event_create_ordinal =
      std::numeric_limits<std::size_t>::max();
  bool bad_mapped_pair = false;
  bool fail_next_memset = false;
  std::size_t static_query_calls = 0U;
  std::size_t stream_create_calls = 0U;
  std::size_t stream_destroy_calls = 0U;
  std::size_t event_create_calls = 0U;
  std::size_t event_destroy_calls = 0U;
  std::size_t device_allocate_calls = 0U;
  std::size_t device_free_calls = 0U;
  std::size_t host_allocate_calls = 0U;
  std::size_t host_free_calls = 0U;
  std::size_t memset_calls = 0U;
  std::size_t event_record_calls = 0U;
  std::size_t event_wait_calls = 0U;
  std::size_t stream_sync_calls = 0U;

 private:
  std::array<void*, runtime::kSm87BulkV2P40StreamCount> live_streams_{};
  std::array<void*, runtime::kSm87BulkV2P40ReusableEventCount> live_events_{};
  std::uint8_t* device_allocation_ = nullptr;
  std::uint32_t* cancellation_host_ = nullptr;
};

class FakeRequestStateCudaRuntime final
    : public owner::Sm87BulkV2P40RequestStateCudaRuntime {
 public:
  ~FakeRequestStateCudaRuntime() override { delete pinned_handoff_; }

  [[nodiscard]] int get_current_device(
      std::int32_t* const device_ordinal) noexcept override {
    ++static_query_calls;
    if (device_ordinal == nullptr) {
      return 1;
    }
    *device_ordinal = 0;
    return 0;
  }

  [[nodiscard]] int get_device_properties(
      const std::int32_t device_ordinal,
      owner::Sm87BulkV2P40RequestDeviceProperties* const properties)
      noexcept override {
    ++static_query_calls;
    if (device_ordinal != 0 || properties == nullptr) {
      return 1;
    }
    *properties = {8, 7, 16};
    return 0;
  }

  [[nodiscard]] int get_stream_flags(
      void* const stream, unsigned int* const flags) noexcept override {
    ++static_query_calls;
    if (stream == nullptr || flags == nullptr) {
      return 1;
    }
    *flags = owner::kSm87BulkV2P40NonBlockingStreamFlag;
    return 0;
  }

  [[nodiscard]] int allocate_device(
      void** const pointer, const std::size_t bytes) noexcept override {
    if (pointer == nullptr ||
        bytes != runtime::kSm87BulkV2P40RequestArenaBytes ||
        arena_live_) {
      return 1;
    }
    arena_live_ = true;
    *pointer = arena_pointer();
    return 0;
  }

  [[nodiscard]] int free_device(void* const pointer) noexcept override {
    if (!arena_live_ || pointer != arena_pointer()) {
      return 1;
    }
    arena_live_ = false;
    return 0;
  }

  [[nodiscard]] int allocate_pinned_host(
      void** const pointer, const std::size_t bytes) noexcept override {
    if (pointer == nullptr ||
        bytes != sizeof(owner::Sm87BulkV2P40PinnedHandoff) ||
        pinned_handoff_ != nullptr) {
      return 1;
    }
    pinned_handoff_ =
        new (std::nothrow) owner::Sm87BulkV2P40PinnedHandoff();
    if (pinned_handoff_ == nullptr) {
      return 2;
    }
    *pointer = pinned_handoff_;
    return 0;
  }

  [[nodiscard]] int free_pinned_host(void* const pointer) noexcept override {
    if (pointer != pinned_handoff_) {
      return 1;
    }
    delete pinned_handoff_;
    pinned_handoff_ = nullptr;
    return 0;
  }

  [[nodiscard]] int query_pointer(
      const void* const pointer,
      owner::Sm87BulkV2P40RequestPointerAttributes* const attributes)
      noexcept override {
    ++static_query_calls;
    if (attributes == nullptr) {
      return 1;
    }
    if (arena_live_ && pointer == arena_pointer()) {
      *attributes = {owner::Sm87BulkV2P40RequestPointerKind::kDevice,
                     nullptr, arena_pointer(), 0};
      return 0;
    }
    if (pointer == pinned_handoff_) {
      *attributes = {owner::Sm87BulkV2P40RequestPointerKind::kHost,
                     pinned_handoff_, nullptr, 0};
      return 0;
    }
    return 1;
  }

  [[nodiscard]] int memset_async(void* const pointer, const int value,
                                 const std::size_t bytes,
                                 void* const stream) noexcept override {
    if (!arena_live_ || pointer != arena_pointer() || value != 0 ||
        bytes != runtime::kSm87BulkV2P40ColdResetBytes || stream == nullptr) {
      return 1;
    }
    ++memset_calls;
    return 0;
  }

  [[nodiscard]] int copy_device_to_host_async(
      void* const host_destination, const void* const device_source,
      const std::size_t bytes, void* const stream) noexcept override {
    if (host_destination != pinned_handoff_ || device_source == nullptr ||
        bytes != sizeof(owner::Sm87BulkV2P40PinnedHandoff) ||
        stream == nullptr) {
      return 1;
    }
    ++copy_calls;
    return 0;
  }

  [[nodiscard]] int synchronize_stream(
      void* const stream) noexcept override {
    if (stream == nullptr) {
      return 1;
    }
    ++stream_sync_calls;
    return 0;
  }

  std::size_t static_query_calls = 0U;
  std::size_t memset_calls = 0U;
  std::size_t copy_calls = 0U;
  std::size_t stream_sync_calls = 0U;

 private:
  [[nodiscard]] static void* arena_pointer() noexcept {
    return reinterpret_cast<void*>(0x1'0000'0000ULL);
  }

  bool arena_live_ = false;
  owner::Sm87BulkV2P40PinnedHandoff* pinned_handoff_ = nullptr;
};

template <class T>
[[nodiscard]] T* fake_pointer(const std::uintptr_t address) noexcept {
  return reinterpret_cast<T*>(address);
}

[[nodiscard]] kernels::Sm87BulkV2GdnKernelResources passing_gdn_kernel(
    const int threads, const int grid) noexcept {
  kernels::Sm87BulkV2GdnKernelResources resources;
  resources.registers_per_thread = 64;
  resources.static_shared_bytes = 34'056U;
  resources.local_bytes = 0U;
  resources.maximum_threads_per_block = 1'024;
  resources.active_blocks_per_sm = 4;
  resources.threads_per_block = threads;
  resources.physical_grid_ctas = grid;
  return resources;
}

[[nodiscard]] kernels::Sm87BulkV2GdnC64Resources
passing_gdn_resources() noexcept {
  kernels::Sm87BulkV2GdnC64Resources resources;
  resources.binary_version = 87;
  resources.producer = passing_gdn_kernel(
      static_cast<int>(kernels::kSm87BulkV2GdnProducerThreads),
      static_cast<int>(kernels::kSm87BulkV2GdnProducerCtas));
  resources.recurrence = passing_gdn_kernel(
      static_cast<int>(kernels::kSm87BulkV2GdnRecurrenceThreads),
      static_cast<int>(kernels::kSm87BulkV2GdnRecurrenceCtas));
  resources.epilogue = passing_gdn_kernel(
      static_cast<int>(kernels::kSm87BulkV2GdnEpilogueThreads),
      static_cast<int>(kernels::kSm87BulkV2GdnEpilogueCtas));
  resources.kernels_compiled = true;
  resources.exact_geometry = true;
  resources.resource_gate_passed = true;
  return resources;
}

[[nodiscard]] kernels::Sm87BulkV2GdnP40Arguments
make_gdn_arguments_for_owner_contract(
    kernels::Sm87BulkV2GdnP40SubmissionReceipt* const receipt) noexcept {
  kernels::Sm87BulkV2GdnP40Arguments arguments;
  arguments.raw_qkvz =
      fake_pointer<const std::uint16_t>(0x0000'0010'0000'0000ULL);
  arguments.interleaved_ab =
      fake_pointer<const std::uint16_t>(0x0000'0010'6000'0000ULL);
  arguments.conv_weight =
      fake_pointer<const std::uint16_t>(0x0000'0010'7000'0000ULL);
  arguments.initial_conv_history =
      fake_pointer<const std::uint16_t>(0x0000'0010'7020'0000ULL);
  arguments.a_log =
      fake_pointer<const std::uint16_t>(0x0000'0010'7040'0000ULL);
  arguments.dt_bias =
      fake_pointer<const std::uint16_t>(0x0000'0010'7040'1000ULL);
  arguments.norm_weight =
      fake_pointer<const std::uint16_t>(0x0000'0010'7040'2000ULL);
  arguments.initial_recurrent_state =
      fake_pointer<const std::uint16_t>(0x0000'0010'8000'0000ULL);
  arguments.l2_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.norm_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.normalized_q = {
      fake_pointer<float>(0x0000'0010'9000'0000ULL),
      fake_pointer<float>(0x0000'0010'9010'0000ULL)};
  arguments.normalized_k = {
      fake_pointer<float>(0x0000'0010'9020'0000ULL),
      fake_pointer<float>(0x0000'0010'9030'0000ULL)};
  arguments.prepared_v = {
      fake_pointer<std::uint16_t>(0x0000'0010'9040'0000ULL),
      fake_pointer<std::uint16_t>(0x0000'0010'9050'0000ULL)};
  arguments.alpha = {
      fake_pointer<float>(0x0000'0010'9060'0000ULL),
      fake_pointer<float>(0x0000'0010'9061'0000ULL)};
  arguments.beta = {
      fake_pointer<float>(0x0000'0010'9062'0000ULL),
      fake_pointer<float>(0x0000'0010'9063'0000ULL)};
  arguments.raw_output = {
      fake_pointer<std::uint16_t>(0x0000'0010'9070'0000ULL),
      fake_pointer<std::uint16_t>(0x0000'0010'9080'0000ULL)};
  arguments.output =
      fake_pointer<std::uint16_t>(0x0000'0011'0000'0000ULL);
  arguments.conv_history = {
      fake_pointer<std::uint16_t>(0x0000'0011'3000'0000ULL),
      fake_pointer<std::uint16_t>(0x0000'0011'3020'0000ULL)};
  arguments.transactional_recurrent_state =
      fake_pointer<std::uint16_t>(0x0000'0011'4000'0000ULL);
  arguments.cancellation_snapshot = {
      fake_pointer<std::uint32_t>(0x0000'0011'4020'0000ULL),
      fake_pointer<std::uint32_t>(0x0000'0011'4020'1000ULL)};
  arguments.submission_receipt = receipt;
  return arguments;
}

[[nodiscard]] kernels::Sm87BulkV2GdnP40Session
make_terminal_gdn_session_for_owner_contract(
    const owner::Sm87BulkV2P40Owner& execution_owner,
    kernels::Sm87BulkV2GdnP40SubmissionReceipt* const receipt) noexcept {
  kernels::Sm87BulkV2GdnP40SessionPlan plan;
  const auto arguments = make_gdn_arguments_for_owner_contract(receipt);
  for (auto& layer : plan.layers) {
    layer = arguments;
  }
  owner::Sm87BulkV2P40OwnerHostFixture::
      bind_gdn_owner_handles_for_host_contract(execution_owner, &plan);

  kernels::Sm87BulkV2GdnP40Session session;
  session.sealed_plan = plan;
  session.sealed_resources = passing_gdn_resources();
  session.sealed_device = execution_owner.device_ordinal();
  session.lifecycle =
      kernels::Sm87BulkV2GdnP40SessionLifecycle::kAwaitingDrain;
  session.next_epoch = kernels::kSm87BulkV2GdnP40SessionLayerCount;
  session.bridged_epochs = kernels::kSm87BulkV2GdnP40SessionLayerCount;
  session.bridge_pending = false;
  return session;
}

[[nodiscard]] owner::Sm87BulkV2P40OwnerIdentity evidence_identity() noexcept {
  owner::Sm87BulkV2P40OwnerIdentity identity;
  identity.deployment_identity = 101U;
  identity.model_identity = 102U;
  identity.request_allocation_identity = 103U;
  identity.stream_event_owner_identity = 104U;
  identity.asset_catalog_identity = 105U;
  identity.binary_evidence_identity = 106U;
  identity.fp8_oracle_evidence_identity = 107U;
  identity.attention_oracle_evidence_identity = 108U;
  identity.gdn_oracle_evidence_identity = 109U;
  identity.nvfp4_oracle_evidence_identity = 110U;
  return identity;
}

[[nodiscard]] owner::Sm87BulkV2P40DevelopmentAdmissionEvidence
development_admission_semantics() noexcept {
  owner::Sm87BulkV2P40DevelopmentAdmissionEvidence evidence;
  evidence.identity = evidence_identity();
  evidence.identity.plan_magic = runtime::kSm87BulkV2P40PlanMagic;
  evidence.identity.abi_major = runtime::kSm87BulkV2P40PlanAbiMajor;
  evidence.identity.abi_minor = runtime::kSm87BulkV2P40PlanAbiMinor;
  evidence.identity.owner_identity = 201U;
  evidence.identity.seal_nonce = 202U;
  evidence.identity.device_ordinal = 0;
  evidence.identity.execution_class = owner::Sm87BulkV2P40ExecutionClass::
      kDefaultOffDevelopmentCandidate;
  evidence.identity.authenticated_real_constituents = true;
  evidence.identity.exact_numerical_contract_qualified = true;
  evidence.identity.development_execution_eligible = true;
  evidence.identity.production_dispatch_eligible = false;
  evidence.constituent_findings.fill(
      owner::Sm87BulkV2P40ConstituentFinding::kReady);
  evidence.all_static_resource_checks_complete = true;
  evidence.authenticated_real_constituents = true;
  evidence.exact_numerical_contract_qualified = true;
  evidence.default_off_candidate_eligible = true;
  evidence.production_dispatch_eligible = false;
  evidence.production_selector_bound = false;
  evidence.synthetic_host_contract = false;
  return evidence;
}

void test_development_admission_is_not_production_admission(
    TestContext& test) {
  const auto valid = development_admission_semantics();
  test.expect(valid.semantics_valid(),
              "real-qualified default-off evidence admits development while production remains false");

  auto production = valid;
  production.production_dispatch_eligible = true;
  production.identity.production_dispatch_eligible = true;
  test.expect(!production.semantics_valid(),
              "production eligibility is not required and cannot be smuggled into the development gate");

  auto synthetic = valid;
  synthetic.synthetic_host_contract = true;
  test.expect(!synthetic.semantics_valid(),
              "synthetic host-contract evidence cannot cross the real development gate");

  auto unqualified = valid;
  unqualified.constituent_findings[static_cast<std::size_t>(
      owner::Sm87BulkV2P40Constituent::kNvFp4Projection)] =
      owner::Sm87BulkV2P40ConstituentFinding::
          kNumericalContractUnqualified;
  test.expect(!unqualified.semantics_valid(),
              "one unqualified real constituent keeps development execution closed");

  FakeCudaRuntime cuda;
  auto created = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda);
  test.expect(static_cast<bool>(created),
              "real-composite audit fixture owns startup resources");
  if (!created) {
    return;
  }
  owner::Sm87BulkV2P40RealConstituentSealRequest request;
  request.identities = {301U, 302U, 303U, 304U, 305U,
                        306U, 307U, 308U, 309U, 310U};
  const auto audited =
      owner::seal_sm87_bulk_dataflow_v2_p40_real_constituents(
          *created.owner, request);
  test.expect(!audited &&
                  audited.status.error ==
                      owner::Sm87BulkV2P40OwnerError::kMissingConstituentSeal &&
                  created.owner->state() ==
                      owner::Sm87BulkV2P40OwnerState::kResourcesReady &&
                  created.owner->execution_access() == nullptr,
              "missing real capabilities are reported without minting execution authority");
  test.expect(
      audited.audit.constituent_findings[static_cast<std::size_t>(
          owner::Sm87BulkV2P40Constituent::kNvFp4Projection)] ==
          owner::Sm87BulkV2P40ConstituentFinding::
              kMissingUnforgeableQualificationInterface &&
          audited.audit.constituent_findings[static_cast<std::size_t>(
              owner::Sm87BulkV2P40Constituent::kRequestArena)] ==
              owner::Sm87BulkV2P40ConstituentFinding::
                  kMissingUnforgeableQualificationInterface &&
          audited.audit.constituent_findings[static_cast<std::size_t>(
              owner::Sm87BulkV2P40Constituent::kPinnedHandoff)] ==
              owner::Sm87BulkV2P40ConstituentFinding::
                  kMissingUnforgeableQualificationInterface,
      "NVFP4, request-arena, and pinned-handoff interface gaps stay explicit and fail closed");
}

void test_direction_witness_is_disjoint_from_qualified_execution(
    TestContext& test) {
  auto identity = evidence_identity();
  identity.plan_magic = runtime::kSm87BulkV2P40PlanMagic;
  identity.abi_major = runtime::kSm87BulkV2P40PlanAbiMajor;
  identity.abi_minor = runtime::kSm87BulkV2P40PlanAbiMinor;
  identity.owner_identity = 211U;
  identity.seal_nonce = 212U;
  identity.device_ordinal = 0;
  identity.execution_class = owner::Sm87BulkV2P40ExecutionClass::
      kDefaultOffDirectionWitness;
  identity.authenticated_real_constituents = true;
  identity.exact_numerical_contract_qualified = false;
  identity.development_execution_eligible = true;
  identity.production_dispatch_eligible = false;
  test.expect(identity.valid() && identity.direction_witness_valid() &&
                  !identity.development_candidate_valid() &&
                  !identity.synthetic_host_contract_valid(),
              "an authenticated accuracy-unqualified identity is valid only as a default-off direction witness");

  auto exact = identity;
  exact.exact_numerical_contract_qualified = true;
  test.expect(!exact.valid() && !exact.direction_witness_valid(),
              "numerical qualification cannot be asserted while retaining direction-witness identity");
  auto production = identity;
  production.production_dispatch_eligible = true;
  test.expect(!production.valid() && !production.direction_witness_valid(),
              "a direction witness can never carry production-dispatch eligibility");
  auto unauthenticated = identity;
  unauthenticated.authenticated_real_constituents = false;
  test.expect(!unauthenticated.valid() &&
                  !unauthenticated.direction_witness_valid(),
              "caller-shaped untrusted constituents cannot mint a direction witness");

  FakeCudaRuntime cuda;
  auto created = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda);
  test.expect(static_cast<bool>(created),
              "direction-witness fixture owns startup resources");
  if (!created) {
    return;
  }
  auto witness = owner::Sm87BulkV2P40OwnerHostFixture::
      mint_direction_witness_constituent_seal(*created.owner,
                                              evidence_identity());
  test.expect(witness != nullptr,
              "the test-only fixture can form the direction-witness topology");
  if (witness == nullptr) {
    return;
  }

  const auto development = created.owner->
      seal_for_default_off_development_execution(*witness);
  test.expect(!development &&
                  development.error ==
                      owner::Sm87BulkV2P40OwnerError::
                          kInvalidConstituentSeal &&
                  created.owner->state() ==
                      owner::Sm87BulkV2P40OwnerState::kResourcesReady,
              "an accuracy-unqualified witness cannot enter the numerically qualified development gate");
  const auto synthetic =
      created.owner->seal_synthetic_for_host_contract(*witness);
  test.expect(!synthetic &&
                  synthetic.error == owner::Sm87BulkV2P40OwnerError::
                                         kInvalidConstituentSeal &&
                  created.owner->state() ==
                      owner::Sm87BulkV2P40OwnerState::kResourcesReady,
              "the synthetic host-contract gate rejects a direction witness");
  const auto admitted =
      created.owner->seal_for_default_off_direction_witness(*witness);
  const auto* const access = created.owner->execution_access();
  test.expect(static_cast<bool>(admitted) && access != nullptr &&
                  access->identity().direction_witness_valid() &&
                  !access->identity().development_candidate_valid() &&
                  !access->identity().exact_numerical_contract_qualified &&
                  !access->identity().production_dispatch_eligible,
              "only the explicit default-off direction gate mints the accuracy-unqualified execution capability");
}

void test_whole_projection_startup_capability(TestContext& test) {
  FakeCudaRuntime cuda_a;
  FakeCudaRuntime cuda_b;
  auto first = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda_a);
  auto second = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda_b);
  test.expect(static_cast<bool>(first) && static_cast<bool>(second),
              "two startup resource owners exist for capability isolation");
  if (!first || !second) {
    return;
  }

  const auto passing = owner::
      Sm87BulkV2P40WholeProjectionStartupHostFixture::passing_observations();
  auto forged = passing;
  // These are the authority-looking fields a caller could set in the public
  // observation records.  They must not help, even when every byte and
  // resource value otherwise matches the frozen catalog.
  forged.fp8.admission_capability_issued = true;
  forged.gate_up.resource_gate_passed = true;
  forged.down.resource_gate_passed = true;
  const auto public_attempt = owner::
      Sm87BulkV2P40WholeProjectionStartupHostFixture::
          attempt_from_caller_filled_observations(*first.owner, forged);
  test.expect(
      !public_attempt && public_attempt.root == nullptr &&
          public_attempt.status.error ==
              owner::Sm87BulkV2P40WholeProjectionStartupError::
                  kCallerFilledObservationIsNotAuthority &&
          public_attempt.audit
              .caller_filled_public_observation_used_as_authority,
      "caller-filled public records cannot mint the private startup access");

  const auto synthetic = owner::
      Sm87BulkV2P40WholeProjectionStartupHostFixture::
          mint_from_synthetic_startup_query(*first.owner, passing);
  test.expect(
      !synthetic && synthetic.root != nullptr &&
          synthetic.audit.resource_qualification_valid() &&
          !synthetic.audit.configured_source_sha256_gate_passed &&
          synthetic.audit.synthetic_host_query &&
          owner::Sm87BulkV2P40WholeProjectionStartupHostFixture::
              synthetic_access_valid(synthetic, *first.owner),
      "the host fixture exercises resource/catalog checks without impersonating the source-hash-gated fixed-AOT path");
  if (synthetic.root != nullptr && synthetic.root->access() != nullptr) {
    const auto* const access = synthetic.root->access();
    test.expect(
        access->bound_to(*first.owner) && !access->bound_to(*second.owner) &&
            !access->default_off_fixed_aot_resource_valid() &&
            !access->numerical_contract_qualified() &&
            !access->performance_qualified() &&
            !access->production_dispatch_eligible(),
        "startup access is owner-bound and grants no numerical, performance, or production qualification");
  }

  auto wrong_resources = passing;
  ++wrong_resources.gate_up.registers_per_thread;
  const auto wrong_resource_result = owner::
      Sm87BulkV2P40WholeProjectionStartupHostFixture::
          mint_from_synthetic_startup_query(*first.owner, wrong_resources);
  test.expect(
      !wrong_resource_result && wrong_resource_result.root == nullptr &&
          wrong_resource_result.status.error ==
              owner::Sm87BulkV2P40WholeProjectionStartupError::
                  kResourceMismatch &&
          wrong_resource_result.status.resource_index == 3U,
      "one wrong exact resource value fails closed before capability issue");

  auto wrong_catalog = passing;
  wrong_catalog.fp8.roles[1U].code.sass_identity ^= 1U;
  const auto wrong_catalog_result = owner::
      Sm87BulkV2P40WholeProjectionStartupHostFixture::
          mint_from_synthetic_startup_query(*first.owner, wrong_catalog);
  test.expect(
      !wrong_catalog_result && wrong_catalog_result.root == nullptr &&
          wrong_catalog_result.status.error ==
              owner::Sm87BulkV2P40WholeProjectionStartupError::
                  kRetainedEvidenceCatalogMismatch,
      "one mismatched retained evidence record fails the catalog check without claiming loaded-binary authentication");

  auto missing_successor = passing;
  missing_successor.down_successor_linked = false;
  const auto missing_result = owner::
      Sm87BulkV2P40WholeProjectionStartupHostFixture::
          mint_from_synthetic_startup_query(*first.owner,
                                            missing_successor);
  test.expect(
      !missing_result && missing_result.root == nullptr &&
          missing_result.status.error ==
              owner::Sm87BulkV2P40WholeProjectionStartupError::
                  kMissingWholeSuccessor &&
          missing_result.status.resource_index == 4U,
      "a partial whole-projection successor set fails before execution");

  const auto production_shaped = owner::
      create_sm87_bulk_dataflow_v2_p40_whole_projection_startup_root(
          *first.owner);
  test.expect(
      !production_shaped && production_shaped.root == nullptr &&
          production_shaped.status.error ==
              owner::Sm87BulkV2P40WholeProjectionStartupError::
                  kMissingWholeSuccessor,
      "a build without the complete linked CUDA successor set fails closed without using the host fixture");
}

[[nodiscard]] const owner::Sm87BulkV2P40ExecutionAccess* create_and_seal(
    TestContext& test, owner::Sm87BulkV2P40Owner* const execution_owner) {
  const auto evidence = evidence_identity();
  auto seal = owner::Sm87BulkV2P40OwnerHostFixture::
      mint_synthetic_constituent_seal(*execution_owner, evidence);
  test.expect(seal != nullptr,
              "the host fixture can mint only its T0 constituent proof");
  if (seal == nullptr) {
    return nullptr;
  }
  const auto real_status = execution_owner->
      seal_for_default_off_development_execution(*seal);
  test.expect(!real_status &&
                  real_status.error ==
                      owner::Sm87BulkV2P40OwnerError::
                          kInvalidConstituentSeal,
              "a synthetic fixture can never pass the real default-off development gate");
  const auto status =
      execution_owner->seal_synthetic_for_host_contract(*seal);
  test.expect(static_cast<bool>(status),
              "the owner accepts its complete owner-bound fixture proof");
  return execution_owner->execution_access();
}

[[nodiscard]] const owner::Sm87BulkV2P40ExecutionAccess*
create_and_seal_for_request_state(
    TestContext& test, owner::Sm87BulkV2P40Owner* const execution_owner,
    const owner::Sm87BulkV2P40RequestStateSealedAccess& request_access) {
  auto evidence = evidence_identity();
  evidence.request_allocation_identity =
      request_access.identity().allocation_identity;
  evidence.stream_event_owner_identity =
      request_access.identity().stream_event_owner_identity;
  auto seal = owner::Sm87BulkV2P40OwnerHostFixture::
      mint_synthetic_constituent_seal(*execution_owner, evidence);
  test.expect(seal != nullptr,
              "request-state completion fixture can bind its exact allocation identity");
  if (seal == nullptr) {
    return nullptr;
  }
  const auto status =
      execution_owner->seal_synthetic_for_host_contract(*seal);
  test.expect(static_cast<bool>(status),
              "owner seals the request-allocation-bound synthetic capability");
  return execution_owner->execution_access();
}

[[nodiscard]] bool submit_layer_work(
    owner::Sm87BulkV2P40Owner& execution_owner,
    const owner::Sm87BulkV2P40ExecutionAccess& access,
    const std::size_t model_layer) {
  const bool full = runtime::sm87_bulk_v2_p40_is_full_layer(model_layer);
  const auto note = [&](const runtime::Sm87BulkV2P40Stream stream,
                        const owner::Sm87BulkV2P40SubmissionCounter counter,
                        const std::size_t count,
                        const runtime::Sm87BulkV2P40FamilyPhase family,
                        const std::size_t constituent) {
    return static_cast<bool>(execution_owner.note_submission(
        access, stream, counter, count, model_layer, family, 0U,
        constituent));
  };
  const auto projection =
      runtime::Sm87BulkV2P40Stream::kProjectionAndGdnProducer;
  if (full) {
    if (!note(projection,
              owner::Sm87BulkV2P40SubmissionCounter::
                  kFp8FullInputWholeRoleLaunch,
              1U, runtime::Sm87BulkV2P40FamilyPhase::kFullInput, 0U) ||
        !note(runtime::Sm87BulkV2P40Stream::kMain,
              owner::Sm87BulkV2P40SubmissionCounter::
                  kAttentionPreprocessPanel,
              5U, runtime::Sm87BulkV2P40FamilyPhase::kFullPreprocess, 1U) ||
        !note(runtime::Sm87BulkV2P40Stream::kMain,
              owner::Sm87BulkV2P40SubmissionCounter::kAttentionLaunch,
              q3x::kernels::kSm87BulkV2AttentionKernelLaunches,
              runtime::Sm87BulkV2P40FamilyPhase::kFullAttentionCore, 2U)) {
      return false;
    }
  } else {
    if (!note(projection,
              owner::Sm87BulkV2P40SubmissionCounter::
                  kFp8GdnInputWholeRoleLaunch,
              1U, runtime::Sm87BulkV2P40FamilyPhase::kGdnInput, 0U) ||
        !note(runtime::Sm87BulkV2P40Stream::kBf16Ab,
              owner::Sm87BulkV2P40SubmissionCounter::kBf16AbLaunch, 1U,
              runtime::Sm87BulkV2P40FamilyPhase::kGdnInput, 1U) ||
        !note(projection,
              owner::Sm87BulkV2P40SubmissionCounter::kGdnProducerChunk,
              q3x::kernels::kSm87BulkV2GdnP40Chunks,
              runtime::Sm87BulkV2P40FamilyPhase::kGdnCore, 2U) ||
        !note(runtime::Sm87BulkV2P40Stream::kGdnRecurrence,
              owner::Sm87BulkV2P40SubmissionCounter::kGdnRecurrenceChunk,
              q3x::kernels::kSm87BulkV2GdnP40Chunks,
              runtime::Sm87BulkV2P40FamilyPhase::kGdnCore, 3U) ||
        !note(runtime::Sm87BulkV2P40Stream::kGdnEpilogue,
              owner::Sm87BulkV2P40SubmissionCounter::kGdnEpilogueChunk,
              q3x::kernels::kSm87BulkV2GdnP40Chunks,
              runtime::Sm87BulkV2P40FamilyPhase::kGdnCore, 4U) ||
        !note(projection,
              owner::Sm87BulkV2P40SubmissionCounter::kGdnPersistentCopy,
              2U, runtime::Sm87BulkV2P40FamilyPhase::kGdnStatePublish,
              5U)) {
      return false;
    }
  }
  return note(projection,
              owner::Sm87BulkV2P40SubmissionCounter::
                  kFp8OutputWholeRoleLaunch,
              1U,
              full ? runtime::Sm87BulkV2P40FamilyPhase::kFullOutputProjection
                   : runtime::Sm87BulkV2P40FamilyPhase::kGdnOutputProjection,
              6U) &&
         note(projection,
              owner::Sm87BulkV2P40SubmissionCounter::
                  kNvFp4GateUpWholeRoleLaunch,
              1U, runtime::Sm87BulkV2P40FamilyPhase::kMlp, 7U) &&
         note(projection,
              owner::Sm87BulkV2P40SubmissionCounter::
                  kNvFp4DownWholeRoleLaunch,
              1U, runtime::Sm87BulkV2P40FamilyPhase::kMlp, 8U);
}

[[nodiscard]] bool submit_and_close_layer(
    owner::Sm87BulkV2P40Owner& execution_owner,
    const owner::Sm87BulkV2P40ExecutionAccess& access,
    const std::size_t model_layer) {
  if (!submit_layer_work(execution_owner, access, model_layer)) {
    return false;
  }
  return static_cast<bool>(execution_owner.close_layer(
      access, model_layer,
      runtime::sm87_bulk_v2_p40_is_full_layer(model_layer)
          ? owner::Sm87BulkV2P40LayerKind::kFull
          : owner::Sm87BulkV2P40LayerKind::kGdn));
}

[[nodiscard]] bool join_all_latest_auxiliary_generations(
    owner::Sm87BulkV2P40Owner& execution_owner,
    const owner::Sm87BulkV2P40ExecutionAccess& access) {
  for (std::size_t auxiliary = 1U;
       auxiliary < runtime::kSm87BulkV2P40StreamCount; ++auxiliary) {
    const auto stream =
        static_cast<runtime::Sm87BulkV2P40Stream>(auxiliary);
    const auto event =
        static_cast<runtime::Sm87BulkV2P40ReusableEvent>(auxiliary - 1U);
    if (!execution_owner.record_event(access, stream, event) ||
        !execution_owner.wait_event(
            access, runtime::Sm87BulkV2P40Stream::kMain, event)) {
      return false;
    }
  }
  return true;
}

struct BoundOwnerRequestFixture final {
  FakeCudaRuntime cuda;
  FakeRequestStateCudaRuntime request_cuda;
  owner::Sm87BulkV2P40OwnerCreateResult owner_created;
  owner::Sm87BulkV2P40RequestStateCreateResult request_created;
  const owner::Sm87BulkV2P40ExecutionAccess* access = nullptr;
};

[[nodiscard]] bool initialize_bound_owner_request_fixture(
    TestContext& test, BoundOwnerRequestFixture* const fixture) {
  if (fixture == nullptr) {
    return false;
  }
  fixture->owner_created =
      owner::Sm87BulkV2P40OwnerHostFixture::create(&fixture->cuda);
  test.expect(static_cast<bool>(fixture->owner_created),
              "bound fixture creates an exact resource owner");
  if (!fixture->owner_created) {
    return false;
  }
  const auto borrowed = owner::Sm87BulkV2P40OwnerHostFixture::
      borrow_streams_for_request_state(*fixture->owner_created.owner);
  fixture->request_created =
      owner::Sm87BulkV2P40RequestStateHostFixture::create(
          &fixture->request_cuda, borrowed,
          fixture->owner_created.owner->owner_identity(),
          fixture->owner_created.owner->device_ordinal());
  test.expect(static_cast<bool>(fixture->request_created),
              "bound fixture creates an exact request state");
  if (!fixture->request_created) {
    return false;
  }
  fixture->access = create_and_seal_for_request_state(
      test, fixture->owner_created.owner.get(),
      *fixture->request_created.state->sealed_access());
  return fixture->access != nullptr;
}

void test_resource_creation_and_private_authority(TestContext& test) {
  FakeCudaRuntime cuda;
  auto created = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda);
  test.expect(static_cast<bool>(created),
              "host creation admits the exact resource owner");
  if (!created) {
    return;
  }
  test.expect(cuda.stream_create_calls == 5U &&
                  cuda.event_create_calls == 12U &&
                  cuda.device_allocate_calls == 1U &&
                  cuda.host_allocate_calls == 1U,
              "startup owns five streams, twelve events, one 1152B control arena, and one mapped word");
  test.expect(created.owner->execution_access() == nullptr &&
                  created.owner->state() ==
                      owner::Sm87BulkV2P40OwnerState::kResourcesReady,
              "resource creation alone grants no execution authority");

  const auto* const access = create_and_seal(test, created.owner.get());
  test.expect(access != nullptr && access->identity().valid(),
              "the private execution capability binds the complete identity");
  if (access == nullptr) {
    return;
  }
  test.expect(access->identity().binary_evidence_identity == 106U &&
                  access->identity().fp8_oracle_evidence_identity == 107U &&
                  access->identity().attention_oracle_evidence_identity ==
                      108U &&
                  access->identity().gdn_oracle_evidence_identity == 109U &&
                  access->identity().nvfp4_oracle_evidence_identity == 110U,
              "the capability retains every binary and numerical evidence identity");

  const std::size_t queries_before_request = cuda.static_query_calls;
  const auto begin = created.owner->begin_request(*access, 501U);
  test.expect(static_cast<bool>(begin) &&
                  cuda.static_query_calls == queries_before_request &&
                  cuda.cancellation_value() == 0U,
              "request rearm clears the epoch without a CUDA static query");
  const auto note = created.owner->note_submission(
      *access, runtime::Sm87BulkV2P40Stream::kProjectionAndGdnProducer,
      owner::Sm87BulkV2P40SubmissionCounter::kFp8GdnInputWholeRoleLaunch,
      1U, 0U,
      runtime::Sm87BulkV2P40FamilyPhase::kGdnInput, 0U, 0U);
  test.expect(static_cast<bool>(note),
              "the owner records a first partial constituent submission");
  test.expect(
      created.owner->receipt().aggregate.projection_successor
                  .fp8_gdn_input_whole_launches == 1U &&
          created.owner->receipt().aggregate.projection_successor
                  .fp8_whole_role_launches == 1U &&
          created.owner->receipt().aggregate.projection_successor
                  .fp8_exact_control_launches == 0U,
      "a physical submission increments the whole-role receipt and never the old exact-control ledger");
  const std::size_t sync_before_failure = cuda.stream_sync_calls;
  const auto failed = created.owner->poison_after_submission_failure(
      *access, FakeCudaRuntime::kInjected, 0U,
      runtime::Sm87BulkV2P40FamilyPhase::kGdnInput, 0U, 0U);
  test.expect(!failed &&
                  failed.error ==
                      owner::Sm87BulkV2P40OwnerError::kCudaSubmission &&
                  cuda.stream_sync_calls == sync_before_failure + 5U,
              "a partial error cancels and drains all five streams once");
  test.expect(created.owner->state() ==
                  owner::Sm87BulkV2P40OwnerState::kPoisoned &&
                  cuda.cancellation_value() == 1U &&
                  created.owner->receipt().aggregate.lifecycle ==
                      runtime::Sm87BulkV2P40OwnerLifecycle::kPoisoned &&
                  created.owner->receipt().aggregate.all_streams_drained &&
                  !created.owner->receipt().aggregate.state_committed,
              "partial work is poisoned and can never publish Prefill state");
  test.expect(created.owner->receipt().identity_valid() &&
                  created.owner->receipt().request_hot_static_cuda_queries ==
                      0U &&
                  !created.owner->receipt()
                       .public_aggregate_used_as_authority,
              "the poisoned receipt remains bound to the private capability identity");

  created.owner.reset();
  test.expect(cuda.event_destroy_calls == 12U &&
                  cuda.stream_destroy_calls == 5U &&
                  cuda.host_free_calls == 1U &&
                  cuda.device_free_calls == 1U,
              "destruction releases every startup-owned resource");
}

void test_foreign_seal_rejected(TestContext& test) {
  FakeCudaRuntime cuda_a;
  FakeCudaRuntime cuda_b;
  auto first = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda_a);
  auto second = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda_b);
  test.expect(static_cast<bool>(first) && static_cast<bool>(second),
              "two independent resource owners can be constructed");
  if (!first || !second) {
    return;
  }
  auto foreign = owner::Sm87BulkV2P40OwnerHostFixture::
      mint_synthetic_constituent_seal(*first.owner, evidence_identity());
  test.expect(foreign != nullptr, "the first owner fixture seal exists");
  if (foreign == nullptr) {
    return;
  }
  const auto status = second.owner->
      seal_for_default_off_development_execution(*foreign);
  test.expect(!status && status.error ==
                             owner::Sm87BulkV2P40OwnerError::
                                 kForeignConstituentSeal &&
                  second.owner->execution_access() == nullptr,
              "a complete seal cannot cross its physical owner");
}

void test_natural_layer_close_transaction(TestContext& test) {
  FakeCudaRuntime cuda;
  auto created = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda);
  test.expect(static_cast<bool>(created),
              "natural-layer closure fixture owner exists");
  if (!created) {
    return;
  }
  const auto* const access = create_and_seal(test, created.owner.get());
  if (access == nullptr || !created.owner->begin_request(*access, 551U)) {
    test.expect(false, "natural-layer closure request begins");
    return;
  }

  const auto early = created.owner->close_layer(
      *access, 0U, owner::Sm87BulkV2P40LayerKind::kGdn);
  test.expect(!early && early.error ==
                            owner::Sm87BulkV2P40OwnerError::
                                kIncompleteLayerWork,
              "layer zero cannot close before its exact constituent prefix");
  const auto wrong_kind = created.owner->close_layer(
      *access, 0U, owner::Sm87BulkV2P40LayerKind::kFull);
  test.expect(!wrong_kind &&
                  wrong_kind.error ==
                      owner::Sm87BulkV2P40OwnerError::kWrongLayerKind,
              "caller-declared Full cannot override natural GDN layer zero");
  test.expect(submit_layer_work(*created.owner, *access, 0U),
              "layer zero exact constituent work is recorded");
  const auto skipped = created.owner->close_layer(
      *access, 1U, owner::Sm87BulkV2P40LayerKind::kGdn);
  test.expect(!skipped &&
                  skipped.error ==
                      owner::Sm87BulkV2P40OwnerError::kInvalidLayerOrder,
              "a natural layer cannot be skipped");
  test.expect(static_cast<bool>(created.owner->close_layer(
                  *access, 0U, owner::Sm87BulkV2P40LayerKind::kGdn)) &&
                  created.owner->receipt().aggregate.completed_layers == 1U &&
                  created.owner->receipt().aggregate.completed_gdn_layers ==
                      1U &&
                  created.owner->receipt().aggregate
                          .closed_gdn_state_publications == 1U &&
                  created.owner->receipt().aggregate
                          .logical_projection_roles == 8U &&
                  created.owner->receipt().aggregate.fused_outer_operations ==
                      5U,
              "owner derives and commits the complete GDN layer-zero receipt");
  const auto duplicate = created.owner->close_layer(
      *access, 0U, owner::Sm87BulkV2P40LayerKind::kGdn);
  test.expect(!duplicate &&
                  duplicate.error ==
                      owner::Sm87BulkV2P40OwnerError::kInvalidLayerOrder,
              "an already-closed natural layer cannot close twice");

  const auto wrong_counter = created.owner->note_submission(
      *access,
      runtime::Sm87BulkV2P40Stream::kProjectionAndGdnProducer,
      owner::Sm87BulkV2P40SubmissionCounter::
          kFp8FullInputWholeRoleLaunch,
      1U, 1U, runtime::Sm87BulkV2P40FamilyPhase::kFullInput, 0U, 0U);
  test.expect(!wrong_counter &&
                  wrong_counter.error ==
                      owner::Sm87BulkV2P40OwnerError::kWrongLayerKind,
              "a Full-only launch cannot be attributed to natural GDN layer one");
  test.expect(submit_layer_work(*created.owner, *access, 1U),
              "layer one exact constituent work is recorded");
  const auto second_wrong_kind = created.owner->close_layer(
      *access, 1U, owner::Sm87BulkV2P40LayerKind::kFull);
  test.expect(!second_wrong_kind &&
                  second_wrong_kind.error ==
                      owner::Sm87BulkV2P40OwnerError::kWrongLayerKind,
              "wrong caller kind cannot close a fully submitted GDN layer");
  test.expect(static_cast<bool>(created.owner->close_layer(
                  *access, 1U, owner::Sm87BulkV2P40LayerKind::kGdn)),
              "the exact next natural GDN layer closes once");
  test.expect(static_cast<bool>(created.owner->cancel_request(*access)),
              "partial natural-layer test ends through owner-wide drain");
}

void test_owner_bound_request_state_completion(TestContext& test) {
  FakeCudaRuntime cuda;
  FakeRequestStateCudaRuntime request_cuda;
  auto created = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda);
  test.expect(static_cast<bool>(created),
              "owner-bound terminal transaction fixture exists");
  if (!created) {
    return;
  }
  const auto borrowed =
      owner::Sm87BulkV2P40OwnerHostFixture::borrow_streams_for_request_state(
          *created.owner);
  auto request_created = owner::Sm87BulkV2P40RequestStateHostFixture::create(
      &request_cuda, borrowed, created.owner->owner_identity(),
      created.owner->device_ordinal());
  test.expect(static_cast<bool>(request_created),
              "request state binds the owner's exact five stream handles");
  if (!request_created) {
    return;
  }
  auto& request_state = *request_created.state;
  const auto& request_access = *request_state.sealed_access();
  const auto* const access = create_and_seal_for_request_state(
      test, created.owner.get(), request_access);
  if (access == nullptr) {
    return;
  }
  constexpr std::uint64_t request_epoch = 801U;
  test.expect(static_cast<bool>(
                  created.owner->begin_request(
                      *access, request_state, request_access, request_epoch)),
              "owner and request state begin one identical request epoch");
  const auto early_handoff =
      created.owner->enqueue_owner_bound_handoff_d2h(*access);
  test.expect(!early_handoff &&
                  early_handoff.error ==
                      owner::Sm87BulkV2P40OwnerError::kInvalidFinalOrder &&
                  request_cuda.copy_calls == 0U,
              "fixed D2H is impossible before all layers and final operators close");

  bool all_layers_closed = true;
  for (std::size_t layer = 0U; layer < runtime::kSm87BulkV2P40Layers;
       ++layer) {
    if (!submit_and_close_layer(*created.owner, *access, layer)) {
      all_layers_closed = false;
      break;
    }
  }
  test.expect(all_layers_closed &&
                  created.owner->receipt().aggregate.completed_layers ==
                      runtime::kSm87BulkV2P40Layers &&
                  created.owner->receipt().aggregate.completed_gdn_layers ==
                      runtime::kSm87BulkV2P40GdnLayers &&
                  created.owner->receipt().aggregate.completed_full_layers ==
                      runtime::kSm87BulkV2P40FullLayers,
              "all 64 layers close only through their natural exact prefixes");
  if (!all_layers_closed) {
    return;
  }
  test.expect(join_all_latest_auxiliary_generations(
                  *created.owner, *access),
              "Main transitively joins every latest auxiliary generation");

  const auto final_note = [&](const owner::Sm87BulkV2P40SubmissionCounter counter,
                              const std::size_t constituent) {
    return created.owner->note_submission(
        *access, runtime::Sm87BulkV2P40Stream::kMain, counter, 1U,
        runtime::kSm87BulkV2P40Layers - 1U,
        runtime::Sm87BulkV2P40FamilyPhase::kFinalHandoff, 0U,
        constituent);
  };
  const auto early_lm =
      final_note(owner::Sm87BulkV2P40SubmissionCounter::kLmHead, 1U);
  const auto direct_handoff_note =
      final_note(owner::Sm87BulkV2P40SubmissionCounter::kHandoffD2h, 3U);
  test.expect(!early_lm &&
                  early_lm.error ==
                      owner::Sm87BulkV2P40OwnerError::kInvalidFinalOrder &&
                  !direct_handoff_note &&
                  direct_handoff_note.error ==
                      owner::Sm87BulkV2P40OwnerError::
                          kMissingOwnerBoundHandoff &&
                  request_cuda.copy_calls == 0U,
              "final counters enforce Norm-to-LM-to-Argmax and reject caller-recorded D2H");
  test.expect(final_note(owner::Sm87BulkV2P40SubmissionCounter::kFinalNorm,
                         0U) &&
                  final_note(owner::Sm87BulkV2P40SubmissionCounter::kLmHead,
                             1U) &&
                  final_note(owner::Sm87BulkV2P40SubmissionCounter::kArgmax,
                             2U) &&
                  static_cast<bool>(
                      created.owner->enqueue_owner_bound_handoff_d2h(
                          *access)),
              "final norm, LM head, argmax, and fixed owner-bound D2H close the exact work receipt");
  owner::Sm87BulkV2P40RequestStateHostFixture::
      emulate_completed_handoff_d2h(request_state, 123U, 0U);

  auto gdn_receipt = kernels::sm87_bulk_v2_gdn_p40_submission_receipt();
  gdn_receipt.lifecycle =
      kernels::Sm87BulkV2GdnP40OwnerLifecycle::kSubmitted;
  gdn_receipt.generation = 7U;
  gdn_receipt.successful_submission_calls = 90'000U;
  gdn_receipt.submission_started = true;
  gdn_receipt.reusable = false;
  auto gdn_session = make_terminal_gdn_session_for_owner_contract(
      *created.owner, &gdn_receipt);
  test.expect(
      kernels::sm87_bulk_v2_gdn_p40_session_hot_rearm_candidate(
          gdn_session),
      "the completed 48-layer GDN generation awaits whole-Owner retirement authority");
  const auto premature_rearm =
      owner::Sm87BulkV2P40OwnerHostFixture::
          hot_rearm_gdn_session_after_completed_request(
              *created.owner, *access, gdn_session);
  test.expect(!premature_rearm &&
                  premature_rearm.error ==
                      owner::Sm87BulkV2P40OwnerError::kInvalidOwnerState &&
                  gdn_session.lifecycle ==
                      kernels::Sm87BulkV2GdnP40SessionLifecycle::
                          kAwaitingDrain &&
                  gdn_receipt.lifecycle ==
                      kernels::Sm87BulkV2GdnP40OwnerLifecycle::kSubmitted,
              "a complete submission cannot hot rearm before the sole terminal Main sync");

  const std::size_t owner_sync_before = cuda.stream_sync_calls;
  const std::size_t state_sync_before = request_cuda.stream_sync_calls;
  const auto completed = created.owner->complete_request(
      *access, request_state, request_access);
  const auto& receipt = created.owner->receipt().aggregate;
  test.expect(static_cast<bool>(completed) &&
                  created.owner->state() ==
                      owner::Sm87BulkV2P40OwnerState::kCompleted &&
                  request_state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kCompleted,
              "the exact owner-bound terminal transaction commits both owners");
  test.expect(cuda.stream_sync_calls == owner_sync_before &&
                  request_cuda.stream_sync_calls == state_sync_before + 1U &&
                  receipt.terminal_host_waits == 1U &&
                  receipt.terminal_host_drains == 1U &&
                  receipt.all_streams_drained,
              "normal completion performs only RequestState's one transitive Main host sync");
  test.expect(receipt.lifecycle ==
                  runtime::Sm87BulkV2P40OwnerLifecycle::kCompleted &&
                  receipt.state_committed && receipt.handoff_observed &&
                  receipt.handoff_token_id == 123U &&
                  receipt.handoff_nonfinite == 0U &&
                  created.owner->receipt().identity_valid(),
              "completed receipt publishes only the post-sync private 8-byte handoff observation");

  const std::size_t queries_before_rearm = cuda.static_query_calls;
  const std::size_t owner_syncs_before_rearm = cuda.stream_sync_calls;
  const std::size_t state_syncs_before_rearm =
      request_cuda.stream_sync_calls;
  const std::size_t event_records_before_rearm = cuda.event_record_calls;
  const std::size_t event_waits_before_rearm = cuda.event_wait_calls;
  auto foreign_gdn_session = gdn_session;
  for (auto& layer : foreign_gdn_session.sealed_plan.layers) {
    layer.streams[0U] =
        fake_pointer<void>(0x0000'0000'ffff'1000ULL);
  }
  test.expect(
      kernels::sm87_bulk_v2_gdn_p40_session_hot_rearm_candidate(
          foreign_gdn_session),
      "a structurally valid foreign session is distinguishable from this Owner's physical binding");
  const auto foreign_rearm =
      owner::Sm87BulkV2P40OwnerHostFixture::
          hot_rearm_gdn_session_after_completed_request(
              *created.owner, *access, foreign_gdn_session);
  test.expect(!foreign_rearm &&
                  foreign_rearm.error ==
                      owner::Sm87BulkV2P40OwnerError::kForeignGdnSession &&
                  gdn_receipt.lifecycle ==
                      kernels::Sm87BulkV2GdnP40OwnerLifecycle::kSubmitted,
              "a completed Owner cannot retire another session's CUDA owner");
  const auto rearmed =
      owner::Sm87BulkV2P40OwnerHostFixture::
          hot_rearm_gdn_session_after_completed_request(
              *created.owner, *access, gdn_session);
  test.expect(static_cast<bool>(rearmed) &&
                  gdn_session.lifecycle ==
                      kernels::Sm87BulkV2GdnP40SessionLifecycle::kReady &&
                  gdn_session.next_epoch == 0U &&
                  gdn_session.bridged_epochs == 0U &&
                  !gdn_session.bridge_pending &&
                  gdn_receipt.lifecycle ==
                      kernels::Sm87BulkV2GdnP40OwnerLifecycle::kReady &&
                  gdn_receipt.generation == 7U &&
                  gdn_receipt.drain_attempted &&
                  gdn_receipt.drain_completed && gdn_receipt.reusable,
              "the exact terminal proof rearms only host session/receipt state for the next request");
  test.expect(cuda.static_query_calls == queries_before_rearm &&
                  cuda.stream_sync_calls == owner_syncs_before_rearm &&
                  request_cuda.stream_sync_calls ==
                      state_syncs_before_rearm &&
                  cuda.event_record_calls == event_records_before_rearm &&
                  cuda.event_wait_calls == event_waits_before_rearm,
              "hot rearm performs no CUDA query, submission, event operation, or host wait");
  const auto duplicate_rearm =
      owner::Sm87BulkV2P40OwnerHostFixture::
          hot_rearm_gdn_session_after_completed_request(
              *created.owner, *access, gdn_session);
  test.expect(!duplicate_rearm &&
                  duplicate_rearm.error ==
                      owner::Sm87BulkV2P40OwnerError::
                          kGdnSessionNotRearmable,
              "one terminal proof cannot rearm the same GDN generation twice");
  const auto state_rearmed =
      request_state.rearm_for_cold_request(request_access);
  const auto second_begin = created.owner->begin_request(
      *access, request_state, request_access, request_epoch + 1U);
  test.expect(static_cast<bool>(state_rearmed) &&
                  static_cast<bool>(second_begin) &&
                  created.owner->state() ==
                      owner::Sm87BulkV2P40OwnerState::kActive &&
                  request_state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kActive &&
                  gdn_session.lifecycle ==
                      kernels::Sm87BulkV2GdnP40SessionLifecycle::kReady &&
                  gdn_receipt.lifecycle ==
                      kernels::Sm87BulkV2GdnP40OwnerLifecycle::kReady,
              "the same sealed Owner, RequestState allocation, and GDN CUDA owner admit a second request epoch");
  test.expect(static_cast<bool>(created.owner->cancel_request(*access)),
              "the second-request admission fixture retires through the exact owner drain");
}

void test_owner_bound_cancel_rearms_request_state(TestContext& test) {
  BoundOwnerRequestFixture fixture;
  if (!initialize_bound_owner_request_fixture(test, &fixture)) {
    return;
  }
  auto& execution_owner = *fixture.owner_created.owner;
  auto& request_state = *fixture.request_created.state;
  const auto& request_access = *request_state.sealed_access();
  test.expect(static_cast<bool>(execution_owner.begin_request(
                  *fixture.access, request_state, request_access, 901U)),
              "cancel fixture begins through the owner-bound transaction");
  const std::size_t sync_before = fixture.cuda.stream_sync_calls;
  const auto cancelled = execution_owner.cancel_request(*fixture.access);
  test.expect(static_cast<bool>(cancelled) &&
                  execution_owner.state() ==
                      owner::Sm87BulkV2P40OwnerState::kCancelled &&
                  request_state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kCancelled &&
                  fixture.cuda.stream_sync_calls == sync_before +
                      runtime::kSm87BulkV2P40StreamCount,
              "one owner-wide cancel drain transitions the exact bound request state");
  const auto rearmed = request_state.rearm_for_cold_request(request_access);
  test.expect(static_cast<bool>(rearmed) &&
                  request_state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kReady,
              "a request cancelled by its owner can rearm its cold GDN prefix");
  test.expect(static_cast<bool>(execution_owner.begin_request(
                  *fixture.access, request_state, request_access, 902U)),
              "the rearmed allocation can bind a fresh owner epoch");
  test.expect(static_cast<bool>(execution_owner.cancel_request(
                  *fixture.access)) &&
                  request_state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kCancelled,
              "the second bound epoch also cancels without a state leak");
}

void test_owner_bound_partial_failure_poisons_request_state(
    TestContext& test) {
  BoundOwnerRequestFixture fixture;
  if (!initialize_bound_owner_request_fixture(test, &fixture)) {
    return;
  }
  auto& execution_owner = *fixture.owner_created.owner;
  auto& request_state = *fixture.request_created.state;
  const auto& request_access = *request_state.sealed_access();
  test.expect(static_cast<bool>(execution_owner.begin_request(
                  *fixture.access, request_state, request_access, 911U)),
              "partial-failure fixture begins with an exact state binding");
  test.expect(static_cast<bool>(execution_owner.note_submission(
                  *fixture.access,
                  runtime::Sm87BulkV2P40Stream::
                      kProjectionAndGdnProducer,
                  owner::Sm87BulkV2P40SubmissionCounter::
                      kFp8GdnInputWholeRoleLaunch,
                  1U, 0U, runtime::Sm87BulkV2P40FamilyPhase::kGdnInput,
                  0U, 0U)),
              "partial-failure fixture records one real submission");
  const auto failed = execution_owner.poison_after_submission_failure(
      *fixture.access, FakeCudaRuntime::kInjected, 0U,
      runtime::Sm87BulkV2P40FamilyPhase::kGdnInput, 0U, 0U);
  test.expect(!failed &&
                  failed.error ==
                      owner::Sm87BulkV2P40OwnerError::kCudaSubmission &&
                  execution_owner.state() ==
                      owner::Sm87BulkV2P40OwnerState::kPoisoned &&
                  request_state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kPoisoned &&
                  !request_state.rearm_for_cold_request(request_access),
              "partial CUDA failure drains once and atomically poisons the bound request state");
}

void test_foreign_request_state_is_never_polluted(TestContext& test) {
  BoundOwnerRequestFixture fixture;
  if (!initialize_bound_owner_request_fixture(test, &fixture)) {
    return;
  }
  FakeRequestStateCudaRuntime foreign_cuda;
  std::array<void*, runtime::kSm87BulkV2P40StreamCount> foreign_streams{};
  for (std::size_t index = 0U; index < foreign_streams.size(); ++index) {
    foreign_streams[index] = reinterpret_cast<void*>(
        0x8'0000ULL + (index + 1U) * 0x100U);
  }
  auto foreign = owner::Sm87BulkV2P40RequestStateHostFixture::create(
      &foreign_cuda, foreign_streams,
      fixture.owner_created.owner->owner_identity() + 1000U, 0);
  test.expect(static_cast<bool>(foreign),
              "foreign-state isolation fixture creates independently");
  if (!foreign) {
    return;
  }
  auto& execution_owner = *fixture.owner_created.owner;
  auto& bound_state = *fixture.request_created.state;
  const auto& bound_access = *bound_state.sealed_access();
  auto& foreign_state = *foreign.state;
  const auto& foreign_access = *foreign_state.sealed_access();
  const auto foreign_begin = execution_owner.begin_request(
      *fixture.access, foreign_state, foreign_access, 921U);
  test.expect(!foreign_begin &&
                  foreign_begin.error ==
                      owner::Sm87BulkV2P40OwnerError::kForeignRequestState &&
                  execution_owner.state() ==
                      owner::Sm87BulkV2P40OwnerState::kSealed &&
                  bound_state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kReady &&
                  foreign_state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kReady,
              "foreign begin fails before either request-state lifecycle mutates");
  test.expect(static_cast<bool>(execution_owner.begin_request(
                  *fixture.access, bound_state, bound_access, 921U)),
              "the exact state still binds after a rejected foreign attempt");
  const auto foreign_complete = execution_owner.complete_request(
      *fixture.access, foreign_state, foreign_access);
  test.expect(!foreign_complete &&
                  foreign_complete.error ==
                      owner::Sm87BulkV2P40OwnerError::kForeignRequestState &&
                  execution_owner.state() ==
                      owner::Sm87BulkV2P40OwnerState::kPoisoned &&
                  bound_state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kPoisoned &&
                  foreign_state.lifecycle() ==
                      owner::Sm87BulkV2P40RequestStateLifecycle::kReady,
              "foreign completion poisons only the owner's exact active state and never the foreign object");
}

void test_hot_ordering_and_terminal_wait(TestContext& test) {
  FakeCudaRuntime cuda;
  auto created = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda);
  test.expect(static_cast<bool>(created), "completion fixture owner exists");
  if (!created) {
    return;
  }
  const auto* const access = create_and_seal(test, created.owner.get());
  if (access == nullptr) {
    return;
  }
  const std::size_t queries_before_request = cuda.static_query_calls;
  test.expect(static_cast<bool>(created.owner->begin_request(*access, 601U)),
              "a sealed owner begins a fresh epoch");

  for (std::size_t auxiliary = 1U;
       auxiliary < runtime::kSm87BulkV2P40StreamCount; ++auxiliary) {
    const auto stream =
        static_cast<runtime::Sm87BulkV2P40Stream>(auxiliary);
    const auto event =
        static_cast<runtime::Sm87BulkV2P40ReusableEvent>(auxiliary - 1U);
    test.expect(static_cast<bool>(created.owner->note_submission(
                    *access, stream,
                    owner::Sm87BulkV2P40SubmissionCounter::
                        kFp8GdnInputWholeRoleLaunch,
                    1U, 0U,
                    runtime::Sm87BulkV2P40FamilyPhase::kGdnInput,
                    auxiliary, 0U)),
                "an auxiliary stream records its latest submitted generation");
    test.expect(static_cast<bool>(created.owner->record_event(
                    *access, stream, event)),
                "an auxiliary producer records a device event");
    test.expect(static_cast<bool>(created.owner->wait_event(
                    *access, runtime::Sm87BulkV2P40Stream::kMain, event)),
                "the main stream consumes the exact device dependency");
  }
  test.expect(cuda.static_query_calls == queries_before_request,
              "device ordering performs zero request-time static queries");

  owner::Sm87BulkV2P40OwnerHostFixture::populate_complete_work_receipt(
      created.owner.get());
  const std::size_t sync_before_terminal = cuda.stream_sync_calls;
  const auto completed =
      created.owner->complete_synthetic_for_host_contract(*access, 42U, 0U);
  test.expect(static_cast<bool>(completed) &&
                  cuda.stream_sync_calls == sync_before_terminal + 1U &&
                  created.owner->state() ==
                      owner::Sm87BulkV2P40OwnerState::kCompleted,
              "the fully joined graph uses one terminal host wait");
  test.expect(created.owner->receipt().aggregate.lifecycle ==
                  runtime::Sm87BulkV2P40OwnerLifecycle::kCompleted &&
                  created.owner->receipt().aggregate.state_committed &&
                  created.owner->receipt().aggregate.handoff_observed &&
                  created.owner->receipt().aggregate.terminal_host_waits ==
                      1U &&
                  created.owner->receipt().aggregate.terminal_host_drains ==
                      1U &&
                  created.owner->receipt().identity_valid(),
              "terminal publication retains a complete owner-bound receipt");
  test.expect(
      created.owner->receipt().aggregate.projection_successor
                  .fp8_whole_role_launches == 128U &&
          created.owner->receipt().aggregate.projection_successor
                  .nvfp4_whole_role_launches == 128U &&
          created.owner->receipt().aggregate.projection_successor
                  .fp8_exact_control_launches == 0U &&
          created.owner->receipt().aggregate.projection_successor
                  .nvfp4_exact_control_launches == 0U,
      "the completed owner receipt closes 128 plus 128 successor launches with no control substitution");

  test.expect(static_cast<bool>(created.owner->begin_request(*access, 602U)),
              "a completed owner rearms only under a fresh epoch");
  const auto stale_wait = created.owner->wait_event(
      *access, runtime::Sm87BulkV2P40Stream::kMain,
      runtime::Sm87BulkV2P40ReusableEvent::kNormalizedReady);
  test.expect(!stale_wait &&
                  stale_wait.error ==
                      owner::Sm87BulkV2P40OwnerError::kEventNotRecorded &&
                  created.owner->receipt().joined_auxiliary_stream_mask ==
                      0U,
              "a new epoch clears logical event generations so a prior-request CUDA event cannot satisfy its joins");
  const std::size_t sync_before_cancel = cuda.stream_sync_calls;
  const auto cancelled = created.owner->cancel_request(*access);
  test.expect(static_cast<bool>(cancelled) &&
                  cuda.stream_sync_calls == sync_before_cancel + 5U &&
                  created.owner->state() ==
                      owner::Sm87BulkV2P40OwnerState::kCancelled,
              "explicit cancellation also drains every owned stream");

  const auto replay = created.owner->begin_request(*access, 601U);
  test.expect(!replay && replay.error ==
                             owner::Sm87BulkV2P40OwnerError::
                                 kInvalidRequestEpoch,
              "an older request epoch cannot replay after cancellation");
}

void test_real_completion_requires_owner_bound_handoff(TestContext& test) {
  FakeCudaRuntime cuda;
  auto created = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda);
  test.expect(static_cast<bool>(created),
              "real-completion boundary fixture owner exists");
  if (!created) {
    return;
  }
  const auto* const access = create_and_seal(test, created.owner.get());
  if (access == nullptr || !created.owner->begin_request(*access, 650U)) {
    test.expect(false, "real-completion boundary request begins");
    return;
  }
  for (std::size_t auxiliary = 1U;
       auxiliary < runtime::kSm87BulkV2P40StreamCount; ++auxiliary) {
    const auto stream =
        static_cast<runtime::Sm87BulkV2P40Stream>(auxiliary);
    const auto event =
        static_cast<runtime::Sm87BulkV2P40ReusableEvent>(auxiliary - 1U);
    (void)created.owner->note_submission(
        *access, stream,
        owner::Sm87BulkV2P40SubmissionCounter::
            kFp8GdnInputWholeRoleLaunch,
        1U, 0U,
        runtime::Sm87BulkV2P40FamilyPhase::kGdnInput, auxiliary, 0U);
    (void)created.owner->record_event(*access, stream, event);
    (void)created.owner->wait_event(
        *access, runtime::Sm87BulkV2P40Stream::kMain, event);
  }
  owner::Sm87BulkV2P40OwnerHostFixture::populate_complete_work_receipt(
      created.owner.get());
  const std::size_t sync_before = cuda.stream_sync_calls;
  const auto completed = created.owner->complete_request(*access);
  test.expect(!completed &&
                  completed.error == owner::Sm87BulkV2P40OwnerError::
                                         kMissingOwnerBoundHandoff &&
                  cuda.stream_sync_calls == sync_before + 5U &&
                  created.owner->state() ==
                      owner::Sm87BulkV2P40OwnerState::kPoisoned &&
                  !created.owner->receipt().aggregate.state_committed &&
                  !created.owner->receipt().aggregate.handoff_observed,
              "real completion accepts no caller handoff and fails closed until owner-bound pinned storage exists");
}

void test_invalid_synthetic_handoff_poisoned_after_terminal_wait(
    TestContext& test) {
  const auto exercise = [&test](const std::uint32_t token,
                                const std::uint32_t nonfinite,
                                const char* const message) {
    FakeCudaRuntime cuda;
    auto created = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda);
    test.expect(static_cast<bool>(created),
                "invalid-handoff fixture owner exists");
    if (!created) {
      return;
    }
    const auto* const access = create_and_seal(test, created.owner.get());
    if (access == nullptr ||
        !created.owner->begin_request(*access, 701U)) {
      test.expect(false, "invalid-handoff request begins");
      return;
    }
    for (std::size_t auxiliary = 1U;
         auxiliary < runtime::kSm87BulkV2P40StreamCount; ++auxiliary) {
      const auto stream =
          static_cast<runtime::Sm87BulkV2P40Stream>(auxiliary);
      const auto event =
          static_cast<runtime::Sm87BulkV2P40ReusableEvent>(auxiliary - 1U);
      (void)created.owner->note_submission(
          *access, stream,
          owner::Sm87BulkV2P40SubmissionCounter::
              kFp8GdnInputWholeRoleLaunch,
          1U, 0U,
          runtime::Sm87BulkV2P40FamilyPhase::kGdnInput, auxiliary, 0U);
      (void)created.owner->record_event(*access, stream, event);
      (void)created.owner->wait_event(
          *access, runtime::Sm87BulkV2P40Stream::kMain, event);
    }
    owner::Sm87BulkV2P40OwnerHostFixture::populate_complete_work_receipt(
        created.owner.get());
    const std::size_t sync_before = cuda.stream_sync_calls;
    const auto completed =
        created.owner->complete_synthetic_for_host_contract(
            *access, token, nonfinite);
    test.expect(!completed &&
                    completed.error ==
                        owner::Sm87BulkV2P40OwnerError::kInvalidHandoff &&
                    created.owner->state() ==
                        owner::Sm87BulkV2P40OwnerState::kPoisoned &&
                    cuda.stream_sync_calls == sync_before + 6U &&
                    created.owner->receipt().aggregate.terminal_host_waits ==
                        1U &&
                    cuda.cancellation_value() == 1U &&
                    created.owner->receipt().aggregate.lifecycle ==
                        runtime::Sm87BulkV2P40OwnerLifecycle::kPoisoned &&
                    !created.owner->receipt().aggregate.state_committed &&
                    !created.owner->receipt().aggregate.handoff_observed &&
                    created.owner->receipt().aggregate.first_error != 0,
                message);
  };

  exercise(static_cast<std::uint32_t>(runtime::kSm87BulkV2P40Vocabulary),
           0U,
           "an out-of-vocabulary handoff cancels, drains, and poisons before state commit");
  exercise(17U, 1U,
           "a nonfinite handoff cancels, drains, and poisons before state commit");
}

void test_failure_cleanup(TestContext& test) {
  {
    FakeCudaRuntime cuda;
    cuda.fail_event_create_ordinal = 3U;
    const auto created = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda);
    test.expect(!created &&
                    created.status.error ==
                        owner::Sm87BulkV2P40OwnerError::kEventCreate,
                "partial event construction fails closed");
    test.expect(cuda.stream_destroy_calls == 5U &&
                    cuda.event_destroy_calls == 3U,
                "partial construction is unwound exactly");
  }
  {
    FakeCudaRuntime cuda;
    cuda.bad_mapped_pair = true;
    const auto created = owner::Sm87BulkV2P40OwnerHostFixture::create(&cuda);
    test.expect(!created &&
                    created.status.error ==
                        owner::Sm87BulkV2P40OwnerError::
                            kCancellationValidation,
                "a non-exact mapped host/device pair is rejected");
    test.expect(cuda.host_free_calls == 1U &&
                    cuda.device_free_calls == 1U &&
                    cuda.event_destroy_calls == 12U &&
                    cuda.stream_destroy_calls == 5U,
                "mapped-pair rejection leaves no owned resources");
  }
}

}  // namespace

int main() {
  TestContext test;
  test_development_admission_is_not_production_admission(test);
  test_direction_witness_is_disjoint_from_qualified_execution(test);
  test_whole_projection_startup_capability(test);
  test_resource_creation_and_private_authority(test);
  test_foreign_seal_rejected(test);
  test_natural_layer_close_transaction(test);
  test_owner_bound_request_state_completion(test);
  test_owner_bound_cancel_rearms_request_state(test);
  test_owner_bound_partial_failure_poisons_request_state(test);
  test_foreign_request_state_is_never_polluted(test);
  test_hot_ordering_and_terminal_wait(test);
  test_real_completion_requires_owner_bound_handoff(test);
  test_invalid_synthetic_handoff_poisoned_after_terminal_wait(test);
  test_failure_cleanup(test);
  if (test.failures() != 0) {
    return 1;
  }
  std::cout << "SM87 bulk-dataflow-v2 whole-P40 ownership/lifecycle skeleton host contract passed\n";
  return 0;
}
