#include "sm87_bulk_dataflow_v2_p40_constituent_seal_internal.h"

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
using ProductionCompleteMethod = owner::Sm87BulkV2P40OwnerStatus (
    owner::Sm87BulkV2P40Owner::*)(
        const owner::Sm87BulkV2P40ExecutionAccess&) noexcept;
static_assert(std::is_same_v<
              decltype(&owner::Sm87BulkV2P40Owner::complete_request),
              ProductionCompleteMethod>);

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
              "startup owns five streams, twelve events, one 1280B control arena, and one mapped word");
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
  test_resource_creation_and_private_authority(test);
  test_foreign_seal_rejected(test);
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
