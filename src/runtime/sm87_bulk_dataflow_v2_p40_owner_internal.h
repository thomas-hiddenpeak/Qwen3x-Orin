#pragma once

#include "q3x/runtime/sm87_bulk_dataflow_v2_p40_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail {

inline constexpr unsigned int kSm87BulkV2P40NonBlockingStreamFlag = 1U;

// CUDA is hidden behind this narrow interface so the owner lifecycle can be
// tested without opening a device context.  The production implementation is
// source-private and forwards one-for-one to cudart.  No query method is used
// after startup sealing.
enum class Sm87BulkV2P40PointerKind : std::uint8_t {
  kUnknown = 0U,
  kHost,
  kDevice,
};

struct Sm87BulkV2P40PointerAttributes final {
  Sm87BulkV2P40PointerKind kind = Sm87BulkV2P40PointerKind::kUnknown;
  void* host_pointer = nullptr;
  void* device_pointer = nullptr;
  std::int32_t device_ordinal = -1;
};

struct Sm87BulkV2P40DeviceProperties final {
  std::int32_t major = 0;
  std::int32_t minor = 0;
  std::int32_t multiprocessor_count = 0;
};

class Sm87BulkV2P40CudaRuntime {
 public:
  virtual ~Sm87BulkV2P40CudaRuntime() = default;

  [[nodiscard]] virtual int get_current_device(
      std::int32_t* device_ordinal) noexcept = 0;
  [[nodiscard]] virtual int get_device_properties(
      std::int32_t device_ordinal,
      Sm87BulkV2P40DeviceProperties* properties) noexcept = 0;
  [[nodiscard]] virtual int create_nonblocking_stream(
      void** stream) noexcept = 0;
  [[nodiscard]] virtual int get_stream_flags(
      void* stream, unsigned int* flags) noexcept = 0;
  [[nodiscard]] virtual int destroy_stream(void* stream) noexcept = 0;
  [[nodiscard]] virtual int create_disable_timing_event(
      void** event) noexcept = 0;
  [[nodiscard]] virtual int destroy_event(void* event) noexcept = 0;
  [[nodiscard]] virtual int allocate_device(void** pointer,
                                            std::size_t bytes) noexcept = 0;
  [[nodiscard]] virtual int free_device(void* pointer) noexcept = 0;
  [[nodiscard]] virtual int allocate_mapped_host(void** pointer,
                                                 std::size_t bytes) noexcept = 0;
  [[nodiscard]] virtual int mapped_device_alias(
      void** device_alias, void* host_pointer) noexcept = 0;
  [[nodiscard]] virtual int free_mapped_host(void* pointer) noexcept = 0;
  [[nodiscard]] virtual int query_pointer(
      const void* pointer,
      Sm87BulkV2P40PointerAttributes* attributes) noexcept = 0;

  [[nodiscard]] virtual int memset_async(void* pointer, int value,
                                         std::size_t bytes,
                                         void* stream) noexcept = 0;
  [[nodiscard]] virtual int record_event(void* event,
                                         void* stream) noexcept = 0;
  [[nodiscard]] virtual int stream_wait_event(void* stream,
                                              void* event) noexcept = 0;
  [[nodiscard]] virtual int synchronize_stream(void* stream) noexcept = 0;
};

enum class Sm87BulkV2P40OwnerState : std::uint8_t {
  kEmpty = 0U,
  kResourcesReady,
  kSealed,
  kActive,
  kDraining,
  kCompleted,
  kCancelled,
  kPoisoned,
  kDestroyed,
};

// Development admission is deliberately not production admission.  A real
// candidate must run before production qualification can be measured, while
// the host-contract class is a compile-time-only test seam that can never be
// presented to the real gate.
enum class Sm87BulkV2P40ExecutionClass : std::uint8_t {
  kInvalid = 0U,
  kDefaultOffDevelopmentCandidate,
  kSyntheticHostContract,
};

enum class Sm87BulkV2P40OwnerError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kInvalidPlan,
  kDeviceQuery,
  kWrongDevice,
  kStreamCreate,
  kStreamValidation,
  kEventCreate,
  kDeviceControlAllocation,
  kDeviceControlValidation,
  kCancellationAllocation,
  kCancellationMapping,
  kCancellationValidation,
  kMissingConstituentSeal,
  kForeignConstituentSeal,
  kInvalidConstituentSeal,
  kInvalidOwnerState,
  kForeignExecutionAccess,
  kInvalidRequestEpoch,
  kCudaSubmission,
  kEventNotRecorded,
  kIncompleteDeviceJoin,
  kIncompleteWorkReceipt,
  kMissingOwnerBoundHandoff,
  kInvalidHandoff,
  kDrainFailure,
};

struct Sm87BulkV2P40OwnerStatus final {
  Sm87BulkV2P40OwnerError error = Sm87BulkV2P40OwnerError::kNone;
  const char* context = "none";
  int cuda_error = 0;
  std::size_t resource_index = std::numeric_limits<std::size_t>::max();

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == Sm87BulkV2P40OwnerError::kNone;
  }
};

// Unlike Sm87BulkV2P40SealedExecutionAccess in the public plan header, this
// identity is never accepted from a caller.  It is copied into the private
// owner-bound capability and every request receipt after constituent sealing.
struct Sm87BulkV2P40OwnerIdentity final {
  std::array<std::uint8_t, 8U> plan_magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  std::uint64_t owner_identity = 0U;
  std::uint64_t seal_nonce = 0U;
  std::uint64_t deployment_identity = 0U;
  std::uint64_t model_identity = 0U;
  std::uint64_t request_allocation_identity = 0U;
  std::uint64_t stream_event_owner_identity = 0U;
  std::uint64_t asset_catalog_identity = 0U;
  std::uint64_t binary_evidence_identity = 0U;
  std::uint64_t fp8_oracle_evidence_identity = 0U;
  std::uint64_t attention_oracle_evidence_identity = 0U;
  std::uint64_t gdn_oracle_evidence_identity = 0U;
  std::uint64_t nvfp4_oracle_evidence_identity = 0U;
  std::int32_t device_ordinal = -1;
  Sm87BulkV2P40ExecutionClass execution_class =
      Sm87BulkV2P40ExecutionClass::kInvalid;
  bool authenticated_real_constituents = false;
  bool exact_numerical_contract_qualified = false;
  bool development_execution_eligible = false;
  bool production_dispatch_eligible = false;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool development_candidate_valid() const noexcept;
  [[nodiscard]] bool synthetic_host_contract_valid() const noexcept;
};

struct Sm87BulkV2P40OwnerReceipt final {
  Sm87BulkV2P40RequestReceipt aggregate{};
  Sm87BulkV2P40OwnerIdentity identity{};
  std::size_t device_ordering_operations = 0U;
  std::size_t request_hot_static_cuda_queries = 0U;
  std::uint32_t joined_auxiliary_stream_mask = 0U;
  bool owner_bound_capability_used = false;
  bool public_aggregate_used_as_authority = true;

  [[nodiscard]] bool identity_valid() const noexcept;
};

class Sm87BulkV2P40Owner;
class Sm87BulkV2P40OwnerHostFixture;
struct Sm87BulkV2P40OwnerCreateResult;
class Sm87BulkV2P40RequestState;
struct Sm87BulkV2P40RequestStateCreateResult;
struct Sm87BulkV2P40RealConstituentSealRequest;
struct Sm87BulkV2P40RealConstituentSealResult;

[[nodiscard]] Sm87BulkV2P40RequestStateCreateResult
create_sm87_bulk_dataflow_v2_p40_request_state(
    Sm87BulkV2P40Owner& owner) noexcept;

// This token represents the still-required composition of the real FP8,
// Attention, BF16 A/B, GDN, NVFP4, request-arena, pinned-handoff and
// authenticated-asset
// startup seals.  There is intentionally no public constructor and this
// adapter cannot complete: until all real constituent capabilities exist,
// even default-off development execution remains fail-closed.  Production
// selection is a later, separate qualification and has no factory here.
class Sm87BulkV2P40ConstituentSealAccess final {
 public:
  Sm87BulkV2P40ConstituentSealAccess(
      const Sm87BulkV2P40ConstituentSealAccess&) = delete;
  Sm87BulkV2P40ConstituentSealAccess& operator=(
      const Sm87BulkV2P40ConstituentSealAccess&) = delete;
  Sm87BulkV2P40ConstituentSealAccess(
      Sm87BulkV2P40ConstituentSealAccess&&) = delete;
  Sm87BulkV2P40ConstituentSealAccess& operator=(
      Sm87BulkV2P40ConstituentSealAccess&&) = delete;

 private:
  Sm87BulkV2P40ConstituentSealAccess() = default;

  Sm87BulkV2P40OwnerIdentity identity_{};
  std::uint64_t bound_owner_identity_ = 0U;
  std::array<void*, kSm87BulkV2P40StreamCount> streams_{};
  std::array<void*, kSm87BulkV2P40ReusableEventCount> events_{};
  void* device_control_arena_ = nullptr;
  std::uint32_t* cancellation_host_word_ = nullptr;
  const std::uint32_t* cancellation_device_alias_ = nullptr;
  bool real_fp8_binding_seal = false;
  bool real_attention_binding_seal = false;
  bool real_bf16_ab_binding_seal = false;
  bool real_gdn_session_seal = false;
  bool real_nvfp4_binding_seal = false;
  bool real_request_arena_seal = false;
  bool real_pinned_handoff_seal = false;
  bool all_static_resource_checks_complete = false;
  bool authenticated_real_constituents = false;
  bool exact_numerical_contract_qualified = false;
  bool default_off_candidate_eligible = false;
  bool production_dispatch_eligible = false;
  bool synthetic_host_contract_only = false;

  friend class Sm87BulkV2P40Owner;
  friend class Sm87BulkV2P40OwnerHostFixture;
  friend Sm87BulkV2P40RealConstituentSealResult
  seal_sm87_bulk_dataflow_v2_p40_real_constituents(
      Sm87BulkV2P40Owner&,
      const Sm87BulkV2P40RealConstituentSealRequest&) noexcept;
};

// The only execution authority.  Its constructor is private, its identity is
// owner-issued, and every hot-path entry checks object identity as well as the
// complete sealed identity.  Caller-filled public evidence cannot construct
// or substitute this capability.
class Sm87BulkV2P40ExecutionAccess final {
 public:
  Sm87BulkV2P40ExecutionAccess(const Sm87BulkV2P40ExecutionAccess&) = delete;
  Sm87BulkV2P40ExecutionAccess& operator=(
      const Sm87BulkV2P40ExecutionAccess&) = delete;
  Sm87BulkV2P40ExecutionAccess(Sm87BulkV2P40ExecutionAccess&&) = delete;
  Sm87BulkV2P40ExecutionAccess& operator=(
      Sm87BulkV2P40ExecutionAccess&&) = delete;

  [[nodiscard]] const Sm87BulkV2P40OwnerIdentity& identity() const noexcept {
    return identity_;
  }
  [[nodiscard]] void* cuda_stream(
      Sm87BulkV2P40Stream stream) const noexcept;
  [[nodiscard]] void* cuda_event(
      Sm87BulkV2P40ReusableEvent event) const noexcept;
  [[nodiscard]] void* device_control_arena() const noexcept {
    return device_control_arena_;
  }
  [[nodiscard]] const std::uint32_t* device_cancellation_alias()
      const noexcept {
    return cancellation_device_alias_;
  }

 private:
  Sm87BulkV2P40ExecutionAccess(
      const Sm87BulkV2P40Owner* owner,
      const Sm87BulkV2P40OwnerIdentity& identity,
      const std::array<void*, kSm87BulkV2P40StreamCount>& streams,
      const std::array<void*, kSm87BulkV2P40ReusableEventCount>& events,
      void* device_control_arena,
      const std::uint32_t* cancellation_device_alias) noexcept;

  const Sm87BulkV2P40Owner* owner_ = nullptr;
  Sm87BulkV2P40OwnerIdentity identity_{};
  std::array<void*, kSm87BulkV2P40StreamCount> streams_{};
  std::array<void*, kSm87BulkV2P40ReusableEventCount> events_{};
  void* device_control_arena_ = nullptr;
  const std::uint32_t* cancellation_device_alias_ = nullptr;

  friend class Sm87BulkV2P40Owner;
};

enum class Sm87BulkV2P40SubmissionCounter : std::uint8_t {
  kFp8GdnInputWholeRoleLaunch = 0U,
  kFp8FullInputWholeRoleLaunch,
  kFp8OutputWholeRoleLaunch,
  kNvFp4GateUpWholeRoleLaunch,
  kNvFp4DownWholeRoleLaunch,
  kAttentionLaunch,
  kAttentionPreprocessPanel,
  kBf16AbLaunch,
  kGdnProducerChunk,
  kGdnRecurrenceChunk,
  kGdnEpilogueChunk,
  kGdnPersistentCopy,
  kFinalNorm,
  kLmHead,
  kArgmax,
  kHandoffD2h,
};

class Sm87BulkV2P40Owner final {
 public:
  Sm87BulkV2P40Owner(const Sm87BulkV2P40Owner&) = delete;
  Sm87BulkV2P40Owner& operator=(const Sm87BulkV2P40Owner&) = delete;
  Sm87BulkV2P40Owner(Sm87BulkV2P40Owner&&) = delete;
  Sm87BulkV2P40Owner& operator=(Sm87BulkV2P40Owner&&) = delete;
  ~Sm87BulkV2P40Owner();

  [[nodiscard]] Sm87BulkV2P40OwnerState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }
  [[nodiscard]] const Sm87BulkV2P40ExecutionAccess* execution_access()
      const noexcept {
    return execution_access_.get();
  }
  [[nodiscard]] const Sm87BulkV2P40OwnerReceipt& receipt() const noexcept {
    return receipt_;
  }

  // Real, authenticated, numerically qualified development admission only.
  // production_dispatch_eligible must remain false and no selector is bound.
  [[nodiscard]] Sm87BulkV2P40OwnerStatus
  seal_for_default_off_development_execution(
      const Sm87BulkV2P40ConstituentSealAccess& constituent_seal) noexcept;
#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_OWNER_HOST_FIXTURE)
  [[nodiscard]] Sm87BulkV2P40OwnerStatus seal_synthetic_for_host_contract(
      const Sm87BulkV2P40ConstituentSealAccess& constituent_seal) noexcept;
#endif
  [[nodiscard]] Sm87BulkV2P40OwnerStatus begin_request(
      const Sm87BulkV2P40ExecutionAccess& access,
      std::uint64_t request_epoch) noexcept;
  [[nodiscard]] Sm87BulkV2P40OwnerStatus record_event(
      const Sm87BulkV2P40ExecutionAccess& access,
      Sm87BulkV2P40Stream producer,
      Sm87BulkV2P40ReusableEvent event) noexcept;
  [[nodiscard]] Sm87BulkV2P40OwnerStatus wait_event(
      const Sm87BulkV2P40ExecutionAccess& access,
      Sm87BulkV2P40Stream consumer,
      Sm87BulkV2P40ReusableEvent event) noexcept;
  [[nodiscard]] Sm87BulkV2P40OwnerStatus note_submission(
      const Sm87BulkV2P40ExecutionAccess& access,
      Sm87BulkV2P40Stream producer,
      Sm87BulkV2P40SubmissionCounter counter, std::size_t count,
      std::size_t layer, Sm87BulkV2P40FamilyPhase family,
      std::size_t segment, std::size_t constituent) noexcept;
  [[nodiscard]] Sm87BulkV2P40OwnerStatus poison_after_submission_failure(
      const Sm87BulkV2P40ExecutionAccess& access, int cuda_error,
      std::size_t layer, Sm87BulkV2P40FamilyPhase family,
      std::size_t segment, std::size_t constituent) noexcept;
  [[nodiscard]] Sm87BulkV2P40OwnerStatus cancel_request(
      const Sm87BulkV2P40ExecutionAccess& access) noexcept;
  // Real completion intentionally accepts no caller-provided handoff value.
  // It remains fail-closed until the owner has an authenticated pinned
  // handoff capability that can be read only after terminal GPU completion.
  [[nodiscard]] Sm87BulkV2P40OwnerStatus complete_request(
      const Sm87BulkV2P40ExecutionAccess& access) noexcept;
#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_OWNER_HOST_FIXTURE)
  // Test-only state-machine seam.  The supplied value is observed only after
  // the terminal event/main-stream wait and is absent from production builds.
  [[nodiscard]] Sm87BulkV2P40OwnerStatus
  complete_synthetic_for_host_contract(
      const Sm87BulkV2P40ExecutionAccess& access,
      std::uint32_t handoff_token_id,
      std::uint32_t handoff_nonfinite) noexcept;
#endif

 private:
  explicit Sm87BulkV2P40Owner(Sm87BulkV2P40CudaRuntime* cuda) noexcept;

  [[nodiscard]] Sm87BulkV2P40OwnerStatus initialize_resources() noexcept;
  [[nodiscard]] Sm87BulkV2P40OwnerStatus install_execution_access(
      const Sm87BulkV2P40ConstituentSealAccess& constituent_seal,
      bool require_real_development_qualification) noexcept;
  [[nodiscard]] bool access_matches(
      const Sm87BulkV2P40ExecutionAccess& access) const noexcept;
  [[nodiscard]] Sm87BulkV2P40OwnerStatus cancel_drain_and_transition(
      Sm87BulkV2P40OwnerLifecycle terminal_lifecycle,
      int first_error) noexcept;
  [[nodiscard]] Sm87BulkV2P40OwnerStatus drain_all_streams() noexcept;
  [[nodiscard]] bool work_receipt_complete() const noexcept;
  void publish_cancellation(std::uint32_t value) noexcept;
  void release_resources() noexcept;

  Sm87BulkV2P40CudaRuntime* cuda_ = nullptr;
  Sm87BulkV2P40OwnerState state_ = Sm87BulkV2P40OwnerState::kEmpty;
  std::uint64_t owner_identity_ = 0U;
  std::int32_t device_ordinal_ = -1;
  std::array<void*, kSm87BulkV2P40StreamCount> streams_{};
  std::array<void*, kSm87BulkV2P40ReusableEventCount> events_{};
  void* device_control_arena_ = nullptr;
  std::uint32_t* cancellation_host_word_ = nullptr;
  const std::uint32_t* cancellation_device_alias_ = nullptr;
  std::unique_ptr<Sm87BulkV2P40ExecutionAccess> execution_access_;
  Sm87BulkV2P40OwnerReceipt receipt_{};
  std::uint64_t last_request_epoch_ = 0U;
  std::array<std::size_t, kSm87BulkV2P40ReusableEventCount>
      event_generations_{};
  std::array<Sm87BulkV2P40Stream, kSm87BulkV2P40ReusableEventCount>
      event_producers_{};
  std::array<std::size_t, kSm87BulkV2P40StreamCount>
      stream_submission_generations_{};
  std::array<std::size_t, kSm87BulkV2P40ReusableEventCount>
      event_stream_generations_{};
  std::array<std::size_t, kSm87BulkV2P40StreamCount>
      main_joined_stream_generations_{};

  friend struct Sm87BulkV2P40OwnerCreateResult;
  friend class Sm87BulkV2P40OwnerHostFixture;
  friend Sm87BulkV2P40OwnerCreateResult
  create_sm87_bulk_dataflow_v2_p40_owner_resources() noexcept;
  friend Sm87BulkV2P40RealConstituentSealResult
  seal_sm87_bulk_dataflow_v2_p40_real_constituents(
      Sm87BulkV2P40Owner&,
      const Sm87BulkV2P40RealConstituentSealRequest&) noexcept;
  friend Sm87BulkV2P40RequestStateCreateResult
  create_sm87_bulk_dataflow_v2_p40_request_state(
      Sm87BulkV2P40Owner&) noexcept;
};

struct Sm87BulkV2P40OwnerCreateResult final {
  std::unique_ptr<Sm87BulkV2P40Owner> owner;
  Sm87BulkV2P40OwnerStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner != nullptr && static_cast<bool>(status) &&
           owner->state() == Sm87BulkV2P40OwnerState::kResourcesReady &&
           owner->execution_access() == nullptr;
  }
};

// Resource creation performs the CUDA startup checks but cannot mint
// execution authority.  The concrete constituent adapter remains blocked on
// missing exact interfaces/qualifications, and no production factory exists.
[[nodiscard]] Sm87BulkV2P40OwnerCreateResult
create_sm87_bulk_dataflow_v2_p40_owner_resources() noexcept;

// Source-private host fixture.  It exists only in the explicitly test-only
// owner build to exercise this same state machine with a fake CUDA runtime.
// Its synthetic constituent token has T0 lifecycle authority only and is not
// present in a production compilation.
#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_OWNER_HOST_FIXTURE)
class Sm87BulkV2P40OwnerHostFixture final {
 public:
  [[nodiscard]] static Sm87BulkV2P40OwnerCreateResult create(
      Sm87BulkV2P40CudaRuntime* cuda) noexcept;
  [[nodiscard]] static std::unique_ptr<Sm87BulkV2P40ConstituentSealAccess>
  mint_synthetic_constituent_seal(
      const Sm87BulkV2P40Owner& owner,
      const Sm87BulkV2P40OwnerIdentity& evidence) noexcept;
  static void populate_complete_work_receipt(
      Sm87BulkV2P40Owner* owner) noexcept;
};
#endif

}  // namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail
