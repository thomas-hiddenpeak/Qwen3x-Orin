#pragma once

#include "sm87_macrofeed_v4_request_state_internal.h"
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
#include "../kernels/sm87/sm87_macrofeed_v4_bound_launch_internal.h"
#endif

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail {
class Sm87MacroFeedV4P40ExecutionPackage;
}

namespace q3x::runtime::sm87_macrofeed_v4_execution_events_detail {

inline constexpr std::size_t kSm87MacroFeedV4ExecutionStreamCount = 3U;
inline constexpr std::size_t kSm87MacroFeedV4ExecutionEventCount = 9U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbCyclesPerPanel =
    kSm87MacroFeedV4StateLayerCount;

enum class Sm87MacroFeedV4ExecutionStream : std::uint8_t {
  kMain = 0U,
  kAbAux,
  kControl,
  kCount,
};

enum class Sm87MacroFeedV4ExecutionEvent : std::uint8_t {
  kNormReady = 0U,
  kAbReady,
  kPanelDone,
  kMainTail,
  kAbTail,
  kOwnerDrained,
  kFinalRepresentationReady,
  kCanonicalCopyDone,
  kFinalPublish,
  kCount,
};

enum class Sm87MacroFeedV4ExecutionOwnerState : std::uint8_t {
  kEmpty = 0U,
  kReady,
  kRequestActive,
  kRequestCompleted,
  kRequestDiscarded,
  kPoisoned,
  kDestroyed,
};

enum class Sm87MacroFeedV4ExecutionError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kDeviceQuery,
  kWrongDevice,
  kStreamCreate,
  kStreamValidation,
  kEventCreate,
  kForeignOwnerAccess,
  kForeignRequestAccess,
  kInvalidRequestEpoch,
  kInvalidOwnerState,
  kInvalidPanel,
  kStalePanelGeneration,
  kWrongStream,
  kInvalidEventOrder,
  kEventNotRecorded,
  kEventGenerationUnobserved,
  kEventNotComplete,
  kEventAlreadyObserved,
  kPhysicalObservationForbidden,
  kCudaSubmission,
  kCudaObservation,
  kKernelSubmitContract,
  kReceiptInvalid,
  kPanelIncomplete,
  kDrainIncomplete,
  kFinalPublicationIncomplete,
  kRequestStateDiscard,
  kRequestStateRearm,
  kRequestStatePanelBegin,
  kRequestStatePanelCommit,
};

struct Sm87MacroFeedV4ExecutionStatus final {
  Sm87MacroFeedV4ExecutionError error =
      Sm87MacroFeedV4ExecutionError::kNone;
  const char* context = "none";
  int cuda_error = 0;
  Sm87MacroFeedV4ExecutionStream stream =
      Sm87MacroFeedV4ExecutionStream::kCount;
  Sm87MacroFeedV4ExecutionEvent event =
      Sm87MacroFeedV4ExecutionEvent::kCount;
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation = 0U;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return error == Sm87MacroFeedV4ExecutionError::kNone;
  }
};

class Sm87MacroFeedV4ExecutionEventsOwner;
class Sm87MacroFeedV4ExecutionEventsDriver;
class Sm87MacroFeedV4ExecutionEventsCudaTestFixture;

// The only caller-visible owner capability.  It exposes identities but no raw
// CUDA stream or event handle.  Object identity and a private seal are checked
// on every operation; caller-filled evidence cannot construct a substitute.
class Sm87MacroFeedV4ExecutionEventsAccess final {
 public:
  Sm87MacroFeedV4ExecutionEventsAccess() = delete;
  Sm87MacroFeedV4ExecutionEventsAccess(
      const Sm87MacroFeedV4ExecutionEventsAccess&) = delete;
  Sm87MacroFeedV4ExecutionEventsAccess& operator=(
      const Sm87MacroFeedV4ExecutionEventsAccess&) = delete;
  Sm87MacroFeedV4ExecutionEventsAccess(
      Sm87MacroFeedV4ExecutionEventsAccess&&) = delete;
  Sm87MacroFeedV4ExecutionEventsAccess& operator=(
      Sm87MacroFeedV4ExecutionEventsAccess&&) = delete;

  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }

 private:
  Sm87MacroFeedV4ExecutionEventsAccess(
      const Sm87MacroFeedV4ExecutionEventsOwner* owner,
      std::uint64_t owner_identity, std::uint64_t seal_nonce,
      std::int32_t device_ordinal) noexcept;

  const Sm87MacroFeedV4ExecutionEventsOwner* owner_ = nullptr;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t seal_nonce_ = 0U;
  std::int32_t device_ordinal_ = -1;

  friend class Sm87MacroFeedV4ExecutionEventsOwner;
};

// One owner-minted panel capability.  A capability becomes stale as soon as
// its panel completes, is discarded, or another request begins.
class Sm87MacroFeedV4ExecutionPanelAccess final {
 public:
  Sm87MacroFeedV4ExecutionPanelAccess() = delete;
  Sm87MacroFeedV4ExecutionPanelAccess(
      const Sm87MacroFeedV4ExecutionPanelAccess&) = delete;
  Sm87MacroFeedV4ExecutionPanelAccess& operator=(
      const Sm87MacroFeedV4ExecutionPanelAccess&) = delete;
  Sm87MacroFeedV4ExecutionPanelAccess(
      Sm87MacroFeedV4ExecutionPanelAccess&& other) noexcept;
  Sm87MacroFeedV4ExecutionPanelAccess& operator=(
      Sm87MacroFeedV4ExecutionPanelAccess&& other) noexcept;

  [[nodiscard]] std::uint64_t request_epoch() const noexcept {
    return request_epoch_;
  }
  [[nodiscard]] std::size_t panel() const noexcept { return panel_; }
  [[nodiscard]] std::uint64_t panel_generation() const noexcept {
    return panel_generation_;
  }

 private:
  Sm87MacroFeedV4ExecutionPanelAccess(
      const Sm87MacroFeedV4ExecutionEventsOwner* owner,
      std::uint64_t owner_identity, std::uint64_t seal_nonce,
      std::uint64_t request_epoch, std::size_t panel,
      std::uint64_t panel_generation) noexcept;

  const Sm87MacroFeedV4ExecutionEventsOwner* owner_ = nullptr;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t seal_nonce_ = 0U;
  std::uint64_t request_epoch_ = 0U;
  std::size_t panel_ = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation_ = 0U;

  friend class Sm87MacroFeedV4ExecutionEventsOwner;
};

enum class Sm87MacroFeedV4EnqueueOperation : std::uint8_t {
  kInvalid = 0U,
  kRecord,
  kWait,
};

// Enqueue evidence is intentionally a different type from physical completion
// evidence.  Its authority bits are permanently false.
struct Sm87MacroFeedV4EventEnqueueReceipt final {
  std::uint64_t enqueue_identity = 0U;
  std::uint64_t owner_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation = 0U;
  Sm87MacroFeedV4ExecutionStream stream =
      Sm87MacroFeedV4ExecutionStream::kCount;
  Sm87MacroFeedV4ExecutionEvent event =
      Sm87MacroFeedV4ExecutionEvent::kCount;
  std::uint64_t event_generation = 0U;
  Sm87MacroFeedV4EnqueueOperation operation =
      Sm87MacroFeedV4EnqueueOperation::kInvalid;
  bool cuda_enqueue_accepted = false;
  bool physical_device_completion_attested = false;
  bool production_receipt_eligible = false;
};

struct Sm87MacroFeedV4EventEnqueueResult final {
  Sm87MacroFeedV4ExecutionStatus status{};
  Sm87MacroFeedV4EventEnqueueReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.enqueue_identity != 0U &&
           receipt.cuda_enqueue_accepted &&
           !receipt.physical_device_completion_attested &&
           !receipt.production_receipt_eligible;
  }
};

// Copyable opaque completion evidence.  The default object is deliberately an
// invalid forged receipt.  Valid objects can only be minted after a successful
// cudaEventQuery or cudaEventSynchronize and carry a private owner seal.
class Sm87MacroFeedV4PhysicalCompletionReceipt final {
 public:
  Sm87MacroFeedV4PhysicalCompletionReceipt() = default;
  Sm87MacroFeedV4PhysicalCompletionReceipt(
      const Sm87MacroFeedV4PhysicalCompletionReceipt&) = default;
  Sm87MacroFeedV4PhysicalCompletionReceipt& operator=(
      const Sm87MacroFeedV4PhysicalCompletionReceipt&) = default;

  [[nodiscard]] std::uint64_t receipt_identity() const noexcept {
    return receipt_identity_;
  }
  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::uint64_t request_epoch() const noexcept {
    return request_epoch_;
  }
  [[nodiscard]] std::size_t panel() const noexcept { return panel_; }
  [[nodiscard]] std::uint64_t panel_generation() const noexcept {
    return panel_generation_;
  }
  [[nodiscard]] Sm87MacroFeedV4ExecutionStream producer() const noexcept {
    return producer_;
  }
  [[nodiscard]] Sm87MacroFeedV4ExecutionEvent event() const noexcept {
    return event_;
  }
  [[nodiscard]] std::uint64_t event_generation() const noexcept {
    return event_generation_;
  }
  [[nodiscard]] std::uint64_t main_tail_generation() const noexcept {
    return main_tail_generation_;
  }
  [[nodiscard]] std::uint64_t ab_tail_generation() const noexcept {
    return ab_tail_generation_;
  }
  [[nodiscard]] bool observed_by_query() const noexcept {
    return observed_by_query_;
  }
  [[nodiscard]] bool observed_by_synchronize() const noexcept {
    return observed_by_synchronize_;
  }
  [[nodiscard]] bool physical_device_completion_attested() const noexcept {
    return physical_device_completion_attested_;
  }
  [[nodiscard]] bool production_receipt_eligible() const noexcept {
    return production_receipt_eligible_;
  }

 private:
  std::uint64_t receipt_identity_ = 0U;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t request_epoch_ = 0U;
  std::size_t panel_ = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation_ = 0U;
  Sm87MacroFeedV4ExecutionStream producer_ =
      Sm87MacroFeedV4ExecutionStream::kCount;
  Sm87MacroFeedV4ExecutionEvent event_ =
      Sm87MacroFeedV4ExecutionEvent::kCount;
  std::uint64_t event_generation_ = 0U;
  std::uint64_t main_tail_generation_ = 0U;
  std::uint64_t ab_tail_generation_ = 0U;
  std::uint64_t authenticator_ = 0U;
  bool observed_by_query_ = false;
  bool observed_by_synchronize_ = false;
  bool physical_device_completion_attested_ = false;
  bool production_receipt_eligible_ = false;

  friend class Sm87MacroFeedV4ExecutionEventsOwner;
};

struct Sm87MacroFeedV4PhysicalObservationResult final {
  Sm87MacroFeedV4ExecutionStatus status{};
  Sm87MacroFeedV4PhysicalCompletionReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.receipt_identity() != 0U &&
           receipt.physical_device_completion_attested() &&
           !receipt.production_receipt_eligible();
  }
};

struct Sm87MacroFeedV4PoisonDrainResult final {
  Sm87MacroFeedV4ExecutionStatus drain_status{};
  Sm87MacroFeedV4ExecutionStatus poison_cause{};
  Sm87MacroFeedV4RequestStateStatus request_state_status{};
  std::array<int, kSm87MacroFeedV4ExecutionStreamCount>
      stream_cuda_status{};
  // Audit identity only.  It is deliberately not accepted by RequestState as
  // authority: the EventsOwner performs the state discard while it still owns
  // the lock and the exact poisoned request generation.
  std::uint64_t quiescence_identity = 0U;
  bool all_stream_synchronizations_attempted = false;
  bool physical_quiescence_attested = false;
  bool request_state_discarded = false;
  bool discard_required = false;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(drain_status) &&
           poison_cause.error != Sm87MacroFeedV4ExecutionError::kNone &&
           quiescence_identity != 0U && physical_quiescence_attested;
  }
};

struct Sm87MacroFeedV4PanelBeginResult final {
  std::optional<Sm87MacroFeedV4ExecutionPanelAccess> panel_access;
  Sm87MacroFeedV4ExecutionStatus status{};
  Sm87MacroFeedV4RequestStateStatus request_state_status{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return panel_access.has_value() && static_cast<bool>(status) &&
           static_cast<bool>(request_state_status);
  }
};

// Runtime request admission is one owner-locked physical/logical transaction.
// A successful result carries both the newly minted RequestState epoch and its
// live panel-0 capability; every failure is capability-empty.  This closes the
// otherwise-undrainable interval between a recurrent zero/rearm and the first
// panel.  The construction-only cold-zero seal is deliberately accounted
// separately in the owner snapshot.
struct Sm87MacroFeedV4ColdRequestRearmResult final {
  Sm87MacroFeedV4ExecutionStatus status{};
  Sm87MacroFeedV4RequestStateStatus request_state_status{};
  std::optional<Sm87MacroFeedV4RequestStateSealedAccess> request_access{};
  std::optional<Sm87MacroFeedV4ExecutionPanelAccess> panel_access{};
  std::uint64_t previous_request_epoch = 0U;
  std::uint64_t request_epoch = 0U;
  std::uint64_t enqueued_recurrent_zero_bytes = 0U;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) &&
           static_cast<bool>(request_state_status) &&
           request_access.has_value() && panel_access.has_value() &&
           previous_request_epoch != 0U &&
           request_epoch > previous_request_epoch &&
           request_access->request_epoch() == request_epoch &&
           panel_access->request_epoch() == request_epoch &&
           panel_access->panel() == 0U &&
           enqueued_recurrent_zero_bytes ==
               kSm87MacroFeedV4RecurrentStorageBytes;
  }
};

// Private authenticated audit evidence for a panel transaction that has
// already been committed by EventsOwner.  No RequestState API consumes this
// type, so copying or replaying it can never authorize another bank swap.
class Sm87MacroFeedV4PanelCommitAuditReceipt final {
 public:
  Sm87MacroFeedV4PanelCommitAuditReceipt() = default;
  Sm87MacroFeedV4PanelCommitAuditReceipt(
      const Sm87MacroFeedV4PanelCommitAuditReceipt&) = default;
  Sm87MacroFeedV4PanelCommitAuditReceipt& operator=(
      const Sm87MacroFeedV4PanelCommitAuditReceipt&) = default;

  [[nodiscard]] std::uint64_t receipt_identity() const noexcept {
    return receipt_identity_;
  }
  [[nodiscard]] std::uint64_t request_epoch() const noexcept {
    return request_epoch_;
  }
  [[nodiscard]] std::size_t panel() const noexcept { return panel_; }
  [[nodiscard]] std::uint64_t panel_generation() const noexcept {
    return panel_generation_;
  }
  [[nodiscard]] std::size_t accepted_gdn_layers() const noexcept {
    return accepted_gdn_layers_;
  }
  [[nodiscard]] std::size_t accepted_full_attention_layers() const noexcept {
    return accepted_full_attention_layers_;
  }
  [[nodiscard]] std::size_t accepted_kernel_submissions() const noexcept {
    return accepted_kernel_submissions_;
  }
  [[nodiscard]] std::size_t accepted_d2d_copies() const noexcept {
    return accepted_d2d_copies_;
  }
  [[nodiscard]] std::uint64_t accepted_d2d_copy_bytes() const noexcept {
    return accepted_d2d_copy_bytes_;
  }

 private:
  std::uint64_t receipt_identity_ = 0U;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t request_epoch_ = 0U;
  std::size_t panel_ = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation_ = 0U;
  std::uint64_t panel_done_generation_ = 0U;
  std::size_t accepted_gdn_layers_ = 0U;
  std::size_t accepted_full_attention_layers_ = 0U;
  std::size_t accepted_kernel_submissions_ = 0U;
  std::size_t accepted_d2d_copies_ = 0U;
  std::uint64_t accepted_d2d_copy_bytes_ = 0U;
  std::uint64_t authenticator_ = 0U;
  bool request_state_commit_succeeded_ = false;
  bool mutation_authority_ = false;

  friend class Sm87MacroFeedV4ExecutionEventsOwner;
};

struct Sm87MacroFeedV4PanelCloseCommitResult final {
  Sm87MacroFeedV4ExecutionStatus status{};
  Sm87MacroFeedV4RequestStateStatus request_state_status{};
  Sm87MacroFeedV4PanelCommitAuditReceipt audit_receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) &&
           static_cast<bool>(request_state_status) &&
           audit_receipt.receipt_identity() != 0U;
  }
};

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
enum class Sm87MacroFeedV4GdnSubmissionAuthorityDomain : std::uint8_t {
  kInvalid = 0U,
  kNormalSealedCatalog,
  kSyntheticT1,
};

enum class Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain
    : std::uint8_t {
  kInvalid = 0U,
  kNormalSealedCatalog,
  kSyntheticT1,
};

// One package-bound whole-GDN-layer submission.  Every constituent is fixed
// to C8000 and all CUDA resources were sealed during package construction.
// This type crosses only the private package -> EventsOwner boundary; it is
// not a caller-fillable production API or a physical completion receipt.
struct Sm87MacroFeedV4CompleteGdnLayerC8000Submission final {
  Sm87MacroFeedV4GdnSubmissionAuthorityDomain authority_domain =
      Sm87MacroFeedV4GdnSubmissionAuthorityDomain::kInvalid;
  std::uint64_t execution_package_identity = 0U;
  std::uint64_t gdn_catalog_identity = 0U;
  std::uint64_t gdn_binding_identity = 0U;
  std::uint64_t bf16_ab_catalog_identity = 0U;
  std::uint64_t bf16_ab_pair_identity = 0U;
  std::uint64_t layer_norm_catalog_identity = 0U;
  std::uint64_t layer_norm_pair_identity = 0U;
  std::uint64_t input_norm_binding_identity = 0U;
  std::uint64_t post_norm_binding_identity = 0U;
  std::uint64_t mlp_catalog_identity = 0U;
  std::uint64_t mlp_binding_identity = 0U;
  // In the normal domain this is the construction-sealed GDN resource-seal
  // identity.  It is deliberately zero for Synthetic-T1; the semantic digest
  // below still binds all seven concrete resource snapshots directly.
  std::uint64_t resource_bundle_identity = 0U;
  std::uint64_t synthetic_source_identity = 0U;
  std::size_t gdn_ordinal = kSm87MacroFeedV4StateLayerCount;
  std::size_t model_layer = kSm87MacroFeedV4LayerCount;
  kernels::Sm87MacroFeedV4InputNormArguments input_norm{};
  kernels::Sm87MacroFeedV4Bf16AbArguments bf16_ab{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4GdnQkvzC8000Arguments gdn_qkvz{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4GdnContinuationC8000Arguments gdn_continuation{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4GdnOC8000Arguments gdn_output{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4ResidualPostNormC8000Arguments residual_post_norm{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4GateUpC8000Arguments gate_up{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4DownC8000Arguments down{};
  kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot
      norm_resources{};
  kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot
      bf16_ab_resources{};
  kernels::Sm87MacroFeedV4Fp8CudaResources gdn_qkvz_resources{};
  kernels::Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot
      gdn_continuation_resources{};
  kernels::Sm87MacroFeedV4Fp8CudaResources gdn_output_resources{};
  kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources gate_up_resources{};
  kernels::Sm87MacroFeedV4NvFp4DownCudaResources down_resources{};
};

// Owner-retained physical acceptance evidence.  Kernel and D2D counters are
// separate because the continuation can accept its history copy and zero,
// one or two kernels before a later CUDA error is observed.
struct Sm87MacroFeedV4GdnAcceptedPrefixLedger final {
  std::uint64_t transaction_identity = 0U;
  std::uint64_t owner_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation = 0U;
  std::uint64_t grant_identity = 0U;
  std::uint64_t grant_state_epoch = 0U;
  std::uint64_t recurrent_allocation_identity = 0U;
  std::size_t gdn_ordinal = kSm87MacroFeedV4StateLayerCount;
  std::size_t model_layer = kSm87MacroFeedV4LayerCount;
  std::size_t active_bank_index = 2U;
  std::size_t candidate_bank_index = 2U;
  std::uint64_t active_conv_allocation_offset = 0U;
  std::uint64_t candidate_conv_allocation_offset = 0U;
  std::uint64_t conv_bytes = 0U;
  std::uint64_t active_gdn_state_allocation_offset = 0U;
  std::uint64_t candidate_gdn_state_allocation_offset = 0U;
  std::uint64_t gdn_state_bytes = 0U;
  std::size_t input_norm_launches = 0U;
  std::size_t bf16_ab_launches = 0U;
  std::size_t gdn_qkvz_launches = 0U;
  std::size_t gdn_continuation_launches = 0U;
  std::size_t gdn_output_launches = 0U;
  std::size_t residual_post_norm_launches = 0U;
  std::size_t gate_up_launches = 0U;
  std::size_t down_launches = 0U;
  std::size_t accepted_kernel_launches = 0U;
  std::size_t asynchronous_d2d_copies = 0U;
  std::uint64_t conv_history_copy_bytes = 0U;
  bool complete = false;

  [[nodiscard]] constexpr std::size_t accepted_operations() const noexcept {
    return accepted_kernel_launches + asynchronous_d2d_copies;
  }

  [[nodiscard]] constexpr bool valid_prefix() const noexcept {
    const bool exact_copy =
        (asynchronous_d2d_copies == 0U && conv_history_copy_bytes == 0U) ||
        (asynchronous_d2d_copies == 1U &&
         conv_history_copy_bytes ==
             kernels::kSm87MacroFeedV4GdnConvHistoryBytes);
    const bool ordered_prefix =
        bf16_ab_launches <= input_norm_launches &&
        gdn_qkvz_launches <= bf16_ab_launches &&
        asynchronous_d2d_copies <= gdn_qkvz_launches &&
        (gdn_continuation_launches == 0U ||
         asynchronous_d2d_copies == 1U) &&
        gdn_output_launches <=
            (gdn_continuation_launches == 2U ? 1U : 0U) &&
        residual_post_norm_launches <= gdn_output_launches &&
        gate_up_launches <= residual_post_norm_launches &&
        down_launches <= gate_up_launches;
    return transaction_identity != 0U && owner_identity != 0U &&
           request_epoch != 0U && panel < kSm87MacroFeedV4PanelCount &&
           panel_generation != 0U && grant_identity != 0U &&
           recurrent_allocation_identity != 0U &&
           gdn_ordinal < kSm87MacroFeedV4StateLayerCount &&
           model_layer == gdn_ordinal + gdn_ordinal / 3U &&
           active_bank_index < 2U && candidate_bank_index < 2U &&
           active_bank_index != candidate_bank_index &&
           conv_bytes == kernels::kSm87MacroFeedV4GdnConvHistoryBytes &&
           gdn_state_bytes == kernels::kSm87MacroFeedV4GdnStateBytes &&
           input_norm_launches <= 1U && bf16_ab_launches <= 1U &&
           gdn_qkvz_launches <= 1U && gdn_continuation_launches <= 2U &&
           gdn_output_launches <= 1U &&
           residual_post_norm_launches <= 1U && gate_up_launches <= 1U &&
           down_launches <= 1U &&
           accepted_kernel_launches ==
               input_norm_launches + bf16_ab_launches +
                   gdn_qkvz_launches + gdn_continuation_launches +
                   gdn_output_launches + residual_post_norm_launches +
                   gate_up_launches + down_launches &&
           accepted_kernel_launches <= 9U && exact_copy && ordered_prefix &&
           accepted_operations() <= 10U &&
           (!complete || (accepted_kernel_launches == 9U &&
                          asynchronous_d2d_copies == 1U));
  }
};

// Copyable but opaque enqueue evidence.  Only the issuing owner can mint or
// authenticate it against the still-live grant and exact expected submission.
class Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt final {
 public:
  Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt() = default;
  Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt(
      const Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt&) = default;
  Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt& operator=(
      const Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt&) = default;

  [[nodiscard]] std::uint64_t transaction_identity() const noexcept {
    return transaction_identity_;
  }
  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::uint64_t request_epoch() const noexcept {
    return request_epoch_;
  }
  [[nodiscard]] std::size_t panel() const noexcept { return panel_; }
  [[nodiscard]] std::uint64_t panel_generation() const noexcept {
    return panel_generation_;
  }
  [[nodiscard]] std::uint64_t grant_identity() const noexcept {
    return grant_identity_;
  }
  [[nodiscard]] std::uint64_t grant_state_epoch() const noexcept {
    return grant_state_epoch_;
  }
  [[nodiscard]] std::uint64_t recurrent_allocation_identity() const noexcept {
    return recurrent_allocation_identity_;
  }
  [[nodiscard]] std::size_t gdn_ordinal() const noexcept {
    return gdn_ordinal_;
  }
  [[nodiscard]] std::size_t model_layer() const noexcept {
    return model_layer_;
  }
  [[nodiscard]] std::size_t active_bank_index() const noexcept {
    return active_bank_index_;
  }
  [[nodiscard]] std::size_t candidate_bank_index() const noexcept {
    return candidate_bank_index_;
  }
  [[nodiscard]] std::uint64_t active_conv_allocation_offset() const noexcept {
    return active_conv_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t candidate_conv_allocation_offset()
      const noexcept {
    return candidate_conv_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t conv_bytes() const noexcept {
    return conv_bytes_;
  }
  [[nodiscard]] std::uint64_t active_gdn_state_allocation_offset()
      const noexcept {
    return active_gdn_state_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t candidate_gdn_state_allocation_offset()
      const noexcept {
    return candidate_gdn_state_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t gdn_state_bytes() const noexcept {
    return gdn_state_bytes_;
  }
  [[nodiscard]] Sm87MacroFeedV4GdnSubmissionAuthorityDomain authority_domain()
      const noexcept {
    return authority_domain_;
  }
  [[nodiscard]] std::uint64_t execution_package_identity() const noexcept {
    return execution_package_identity_;
  }
  [[nodiscard]] std::uint64_t gdn_catalog_identity() const noexcept {
    return gdn_catalog_identity_;
  }
  [[nodiscard]] std::uint64_t gdn_binding_identity() const noexcept {
    return gdn_binding_identity_;
  }
  [[nodiscard]] std::uint64_t bf16_ab_catalog_identity() const noexcept {
    return bf16_ab_catalog_identity_;
  }
  [[nodiscard]] std::uint64_t bf16_ab_pair_identity() const noexcept {
    return bf16_ab_pair_identity_;
  }
  [[nodiscard]] std::uint64_t layer_norm_catalog_identity() const noexcept {
    return layer_norm_catalog_identity_;
  }
  [[nodiscard]] std::uint64_t layer_norm_pair_identity() const noexcept {
    return layer_norm_pair_identity_;
  }
  [[nodiscard]] std::uint64_t input_norm_binding_identity() const noexcept {
    return input_norm_binding_identity_;
  }
  [[nodiscard]] std::uint64_t post_norm_binding_identity() const noexcept {
    return post_norm_binding_identity_;
  }
  [[nodiscard]] std::uint64_t mlp_catalog_identity() const noexcept {
    return mlp_catalog_identity_;
  }
  [[nodiscard]] std::uint64_t mlp_binding_identity() const noexcept {
    return mlp_binding_identity_;
  }
  [[nodiscard]] std::uint64_t resource_bundle_identity() const noexcept {
    return resource_bundle_identity_;
  }
  [[nodiscard]] std::uint64_t synthetic_source_identity() const noexcept {
    return synthetic_source_identity_;
  }
  [[nodiscard]] std::uint64_t submission_digest() const noexcept {
    return submission_digest_;
  }
  [[nodiscard]] std::size_t input_norm_launches() const noexcept {
    return input_norm_launches_;
  }
  [[nodiscard]] std::size_t bf16_ab_launches() const noexcept {
    return bf16_ab_launches_;
  }
  [[nodiscard]] std::size_t gdn_qkvz_launches() const noexcept {
    return gdn_qkvz_launches_;
  }
  [[nodiscard]] std::size_t gdn_continuation_launches() const noexcept {
    return gdn_continuation_launches_;
  }
  [[nodiscard]] std::size_t gdn_output_launches() const noexcept {
    return gdn_output_launches_;
  }
  [[nodiscard]] std::size_t residual_post_norm_launches() const noexcept {
    return residual_post_norm_launches_;
  }
  [[nodiscard]] std::size_t gate_up_launches() const noexcept {
    return gate_up_launches_;
  }
  [[nodiscard]] std::size_t down_launches() const noexcept {
    return down_launches_;
  }
  [[nodiscard]] std::size_t bound_kernel_submissions() const noexcept {
    return bound_kernel_submissions_;
  }
  [[nodiscard]] std::size_t asynchronous_d2d_copies() const noexcept {
    return asynchronous_d2d_copies_;
  }
  [[nodiscard]] std::uint64_t conv_history_copy_bytes() const noexcept {
    return conv_history_copy_bytes_;
  }
  [[nodiscard]] bool norm_ready_waited_by_ab() const noexcept {
    return norm_ready_waited_by_ab_;
  }
  [[nodiscard]] bool ab_ready_waited_by_main() const noexcept {
    return ab_ready_waited_by_main_;
  }
  [[nodiscard]] bool complete_layer_enqueued() const noexcept {
    return complete_layer_enqueued_;
  }
  [[nodiscard]] bool physical_device_completion_attested() const noexcept {
    return physical_device_completion_attested_;
  }
  [[nodiscard]] bool panel_complete() const noexcept {
    return panel_complete_;
  }
  [[nodiscard]] bool production_receipt_eligible() const noexcept {
    return production_receipt_eligible_;
  }

  [[nodiscard]] constexpr bool valid_shape() const noexcept {
    const bool normal_authority =
        authority_domain_ ==
            Sm87MacroFeedV4GdnSubmissionAuthorityDomain::kNormalSealedCatalog &&
        gdn_catalog_identity_ != 0U && gdn_binding_identity_ != 0U &&
        mlp_catalog_identity_ != 0U && mlp_binding_identity_ != 0U &&
        resource_bundle_identity_ != 0U && synthetic_source_identity_ == 0U;
    const bool synthetic_authority =
        authority_domain_ ==
            Sm87MacroFeedV4GdnSubmissionAuthorityDomain::kSyntheticT1 &&
        gdn_catalog_identity_ == 0U && gdn_binding_identity_ == 0U &&
        mlp_catalog_identity_ == 0U && mlp_binding_identity_ == 0U &&
        resource_bundle_identity_ == 0U && synthetic_source_identity_ != 0U;
    return transaction_identity_ != 0U && owner_identity_ != 0U &&
           request_epoch_ != 0U && panel_ < kSm87MacroFeedV4PanelCount &&
           panel_generation_ != 0U && grant_identity_ != 0U &&
           recurrent_allocation_identity_ != 0U &&
           gdn_ordinal_ < kSm87MacroFeedV4StateLayerCount &&
           model_layer_ == gdn_ordinal_ + gdn_ordinal_ / 3U &&
           active_bank_index_ < 2U && candidate_bank_index_ < 2U &&
           active_bank_index_ != candidate_bank_index_ &&
           conv_bytes_ == kernels::kSm87MacroFeedV4GdnConvHistoryBytes &&
           gdn_state_bytes_ == kernels::kSm87MacroFeedV4GdnStateBytes &&
           execution_package_identity_ != 0U &&
           bf16_ab_catalog_identity_ != 0U &&
           bf16_ab_pair_identity_ != 0U &&
           layer_norm_catalog_identity_ != 0U &&
           layer_norm_pair_identity_ != 0U &&
           input_norm_binding_identity_ != 0U &&
           post_norm_binding_identity_ != 0U &&
           (normal_authority || synthetic_authority) &&
           submission_digest_ != 0U && input_norm_launches_ == 1U &&
           bf16_ab_launches_ == 1U && gdn_qkvz_launches_ == 1U &&
           gdn_continuation_launches_ == 2U &&
           gdn_output_launches_ == 1U &&
           residual_post_norm_launches_ == 1U &&
           gate_up_launches_ == 1U && down_launches_ == 1U &&
           bound_kernel_submissions_ == 9U &&
           asynchronous_d2d_copies_ == 1U &&
           conv_history_copy_bytes_ ==
               kernels::kSm87MacroFeedV4GdnConvHistoryBytes &&
           norm_ready_waited_by_ab_ && ab_ready_waited_by_main_ &&
           complete_layer_enqueued_ &&
           !physical_device_completion_attested_ && !panel_complete_ &&
           !production_receipt_eligible_;
  }

 private:
  std::uint64_t transaction_identity_ = 0U;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t request_epoch_ = 0U;
  std::size_t panel_ = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation_ = 0U;
  std::uint64_t grant_identity_ = 0U;
  std::uint64_t grant_state_epoch_ = 0U;
  std::uint64_t recurrent_allocation_identity_ = 0U;
  std::size_t gdn_ordinal_ = kSm87MacroFeedV4StateLayerCount;
  std::size_t model_layer_ = kSm87MacroFeedV4LayerCount;
  std::size_t active_bank_index_ = 2U;
  std::size_t candidate_bank_index_ = 2U;
  std::uint64_t active_conv_allocation_offset_ = 0U;
  std::uint64_t candidate_conv_allocation_offset_ = 0U;
  std::uint64_t conv_bytes_ = 0U;
  std::uint64_t active_gdn_state_allocation_offset_ = 0U;
  std::uint64_t candidate_gdn_state_allocation_offset_ = 0U;
  std::uint64_t gdn_state_bytes_ = 0U;
  Sm87MacroFeedV4GdnSubmissionAuthorityDomain authority_domain_ =
      Sm87MacroFeedV4GdnSubmissionAuthorityDomain::kInvalid;
  std::uint64_t execution_package_identity_ = 0U;
  std::uint64_t gdn_catalog_identity_ = 0U;
  std::uint64_t gdn_binding_identity_ = 0U;
  std::uint64_t bf16_ab_catalog_identity_ = 0U;
  std::uint64_t bf16_ab_pair_identity_ = 0U;
  std::uint64_t layer_norm_catalog_identity_ = 0U;
  std::uint64_t layer_norm_pair_identity_ = 0U;
  std::uint64_t input_norm_binding_identity_ = 0U;
  std::uint64_t post_norm_binding_identity_ = 0U;
  std::uint64_t mlp_catalog_identity_ = 0U;
  std::uint64_t mlp_binding_identity_ = 0U;
  std::uint64_t resource_bundle_identity_ = 0U;
  std::uint64_t synthetic_source_identity_ = 0U;
  std::uint64_t submission_digest_ = 0U;
  std::size_t input_norm_launches_ = 0U;
  std::size_t bf16_ab_launches_ = 0U;
  std::size_t gdn_qkvz_launches_ = 0U;
  std::size_t gdn_continuation_launches_ = 0U;
  std::size_t gdn_output_launches_ = 0U;
  std::size_t residual_post_norm_launches_ = 0U;
  std::size_t gate_up_launches_ = 0U;
  std::size_t down_launches_ = 0U;
  std::size_t bound_kernel_submissions_ = 0U;
  std::size_t asynchronous_d2d_copies_ = 0U;
  std::uint64_t conv_history_copy_bytes_ = 0U;
  std::uint64_t authenticator_ = 0U;
  bool norm_ready_waited_by_ab_ = false;
  bool ab_ready_waited_by_main_ = false;
  bool complete_layer_enqueued_ = false;
  bool physical_device_completion_attested_ = false;
  bool panel_complete_ = false;
  bool production_receipt_eligible_ = false;

  friend class Sm87MacroFeedV4ExecutionEventsOwner;
};

struct Sm87MacroFeedV4CompleteGdnLayerEnqueueResult final {
  Sm87MacroFeedV4ExecutionStatus status{};
  Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.valid_shape();
  }
};

// One package-bound Full-Attention layer.  Geometry, role, layout, tactic,
// stream and K/V capacity never cross this boundary as caller choices: the
// EventsOwner reconstructs and validates all public argument forms before it
// permits the first enqueue.
struct Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission final {
  Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain authority_domain =
      Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain::kInvalid;
  std::uint64_t execution_package_identity = 0U;
  std::uint64_t full_attention_catalog_identity = 0U;
  std::uint64_t full_attention_binding_identity = 0U;
  std::uint64_t mlp_binding_identity = 0U;
  std::uint64_t input_norm_binding_identity = 0U;
  std::uint64_t post_norm_binding_identity = 0U;
  std::uint64_t rope_binding_identity = 0U;
  std::uint64_t resource_bundle_identity = 0U;
  std::uint64_t synthetic_source_identity = 0U;
  std::size_t full_attention_ordinal =
      kSm87MacroFeedV4FullAttentionLayerCount;
  std::size_t model_layer = kSm87MacroFeedV4LayerCount;
  kernels::Sm87MacroFeedV4InputNormArguments input_norm{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4FullQkvC8000Arguments full_qkv{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4FullAttentionPreprocessC8000Arguments preprocess{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4AttentionCoreC8000Arguments attention{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4FullAttentionOC8000Arguments full_output{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4ResidualPostNormC8000Arguments residual_post_norm{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4GateUpC8000Arguments gate_up{};
  kernels::sm87_macrofeed_v4_bound_launch_detail::
      Sm87MacroFeedV4DownC8000Arguments down{};
  kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot
      norm_resources{};
  kernels::Sm87MacroFeedV4Fp8CudaResources full_qkv_resources{};
  kernels::Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot
      preprocess_resources{};
  kernels::Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot
      attention_resources{};
  kernels::Sm87MacroFeedV4Fp8CudaResources full_output_resources{};
  kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources gate_up_resources{};
  kernels::Sm87MacroFeedV4NvFp4DownCudaResources down_resources{};
};

// Owner-retained physical acceptance evidence.  The ledger is updated with
// each seam's submitted count before that seam's CUDA status is interpreted,
// so a later failure cannot erase the exact prefix that must be drained.
struct Sm87MacroFeedV4FullAttentionAcceptedPrefixLedger final {
  std::uint64_t transaction_identity = 0U;
  std::uint64_t owner_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation = 0U;
  std::size_t first_position = kSm87MacroFeedV4P40Tokens;
  std::uint64_t grant_identity = 0U;
  std::uint64_t grant_state_epoch = 0U;
  std::uint64_t kv_allocation_identity = 0U;
  std::size_t full_attention_ordinal =
      kSm87MacroFeedV4FullAttentionLayerCount;
  std::size_t model_layer = kSm87MacroFeedV4LayerCount;
  std::size_t input_norm_launches = 0U;
  std::size_t full_qkv_launches = 0U;
  std::size_t preprocess_launches = 0U;
  std::size_t attention_launches = 0U;
  std::size_t full_output_launches = 0U;
  std::size_t residual_post_norm_launches = 0U;
  std::size_t gate_up_launches = 0U;
  std::size_t down_launches = 0U;
  std::size_t accepted_kernel_launches = 0U;
  std::size_t asynchronous_d2d_copies = 0U;
  std::uint64_t asynchronous_d2d_copy_bytes = 0U;
  bool complete = false;

  [[nodiscard]] constexpr bool valid_prefix() const noexcept {
    return transaction_identity != 0U && owner_identity != 0U &&
           request_epoch != 0U && panel < kSm87MacroFeedV4PanelCount &&
           panel_generation != 0U &&
           first_position == panel * kernels::kSm87MacroFeedV4Fp8Tokens &&
           grant_identity != 0U &&
           kv_allocation_identity != 0U &&
           full_attention_ordinal < kSm87MacroFeedV4FullAttentionLayerCount &&
           model_layer == 4U * full_attention_ordinal + 3U &&
           input_norm_launches <= 1U && full_qkv_launches <= 1U &&
           preprocess_launches <= 1U && attention_launches <= 1U &&
           full_output_launches <= 1U &&
           residual_post_norm_launches <= 1U &&
           gate_up_launches <= 1U && down_launches <= 1U &&
           accepted_kernel_launches ==
               input_norm_launches + full_qkv_launches +
                   preprocess_launches + attention_launches +
                   full_output_launches + residual_post_norm_launches +
                   gate_up_launches + down_launches &&
           accepted_kernel_launches <= 8U &&
           asynchronous_d2d_copies == 0U &&
           asynchronous_d2d_copy_bytes == 0U &&
           (!complete || accepted_kernel_launches == 8U);
  }
};

// Copyable but opaque enqueue evidence.  Shape validation is intentionally
// weaker than authority validation: only the issuing owner can recompute the
// private authenticator and bind it to the live request/panel generation.
class Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt final {
 public:
  Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt() = default;
  Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt(
      const Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt&) =
      default;
  Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt& operator=(
      const Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt&) =
      default;

  [[nodiscard]] std::uint64_t transaction_identity() const noexcept {
    return transaction_identity_;
  }
  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::uint64_t request_epoch() const noexcept {
    return request_epoch_;
  }
  [[nodiscard]] std::size_t panel() const noexcept { return panel_; }
  [[nodiscard]] std::uint64_t panel_generation() const noexcept {
    return panel_generation_;
  }
  [[nodiscard]] std::size_t first_position() const noexcept {
    return first_position_;
  }
  [[nodiscard]] std::uint64_t grant_identity() const noexcept {
    return grant_identity_;
  }
  [[nodiscard]] std::uint64_t grant_state_epoch() const noexcept {
    return grant_state_epoch_;
  }
  [[nodiscard]] std::uint64_t kv_allocation_identity() const noexcept {
    return kv_allocation_identity_;
  }
  [[nodiscard]] std::uint64_t key_full_allocation_origin() const noexcept {
    return key_full_allocation_origin_;
  }
  [[nodiscard]] std::uint64_t value_full_allocation_origin() const noexcept {
    return value_full_allocation_origin_;
  }
  [[nodiscard]] std::uint64_t key_panel_allocation_offset() const noexcept {
    return key_panel_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t value_panel_allocation_offset() const noexcept {
    return value_panel_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t kv_panel_bytes() const noexcept {
    return kv_panel_bytes_;
  }
  [[nodiscard]] std::size_t previous_valid_end() const noexcept {
    return previous_valid_end_;
  }
  [[nodiscard]] std::size_t candidate_end() const noexcept {
    return candidate_end_;
  }
  [[nodiscard]] std::size_t full_attention_ordinal() const noexcept {
    return full_attention_ordinal_;
  }
  [[nodiscard]] std::size_t model_layer() const noexcept {
    return model_layer_;
  }
  [[nodiscard]] Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain
  authority_domain() const noexcept {
    return authority_domain_;
  }
  [[nodiscard]] std::uint64_t execution_package_identity() const noexcept {
    return execution_package_identity_;
  }
  [[nodiscard]] std::uint64_t full_attention_catalog_identity()
      const noexcept {
    return full_attention_catalog_identity_;
  }
  [[nodiscard]] std::uint64_t full_attention_binding_identity()
      const noexcept {
    return full_attention_binding_identity_;
  }
  [[nodiscard]] std::uint64_t mlp_binding_identity() const noexcept {
    return mlp_binding_identity_;
  }
  [[nodiscard]] std::uint64_t input_norm_binding_identity() const noexcept {
    return input_norm_binding_identity_;
  }
  [[nodiscard]] std::uint64_t post_norm_binding_identity() const noexcept {
    return post_norm_binding_identity_;
  }
  [[nodiscard]] std::uint64_t rope_binding_identity() const noexcept {
    return rope_binding_identity_;
  }
  [[nodiscard]] std::uint64_t resource_bundle_identity() const noexcept {
    return resource_bundle_identity_;
  }
  [[nodiscard]] std::uint64_t synthetic_source_identity() const noexcept {
    return synthetic_source_identity_;
  }
  [[nodiscard]] std::uint64_t submission_digest() const noexcept {
    return submission_digest_;
  }
  [[nodiscard]] std::size_t input_norm_launches() const noexcept {
    return input_norm_launches_;
  }
  [[nodiscard]] std::size_t full_qkv_launches() const noexcept {
    return full_qkv_launches_;
  }
  [[nodiscard]] std::size_t preprocess_launches() const noexcept {
    return preprocess_launches_;
  }
  [[nodiscard]] std::size_t attention_launches() const noexcept {
    return attention_launches_;
  }
  [[nodiscard]] std::size_t full_output_launches() const noexcept {
    return full_output_launches_;
  }
  [[nodiscard]] std::size_t residual_post_norm_launches() const noexcept {
    return residual_post_norm_launches_;
  }
  [[nodiscard]] std::size_t gate_up_launches() const noexcept {
    return gate_up_launches_;
  }
  [[nodiscard]] std::size_t down_launches() const noexcept {
    return down_launches_;
  }
  [[nodiscard]] std::size_t bound_kernel_submissions() const noexcept {
    return bound_kernel_submissions_;
  }
  [[nodiscard]] std::size_t asynchronous_d2d_copies() const noexcept {
    return asynchronous_d2d_copies_;
  }
  [[nodiscard]] std::uint64_t asynchronous_d2d_copy_bytes() const noexcept {
    return asynchronous_d2d_copy_bytes_;
  }
  [[nodiscard]] bool complete_layer_enqueued() const noexcept {
    return complete_layer_enqueued_;
  }
  [[nodiscard]] bool physical_device_completion_attested() const noexcept {
    return physical_device_completion_attested_;
  }
  [[nodiscard]] bool panel_complete() const noexcept {
    return panel_complete_;
  }
  [[nodiscard]] bool production_receipt_eligible() const noexcept {
    return production_receipt_eligible_;
  }
  [[nodiscard]] constexpr bool valid_shape() const noexcept {
    const bool normal_authority =
        authority_domain_ ==
            Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain::
                kNormalSealedCatalog &&
        execution_package_identity_ != 0U &&
        full_attention_catalog_identity_ != 0U &&
        full_attention_binding_identity_ != 0U &&
        mlp_binding_identity_ != 0U && input_norm_binding_identity_ != 0U &&
        post_norm_binding_identity_ != 0U && rope_binding_identity_ != 0U &&
        resource_bundle_identity_ != 0U && synthetic_source_identity_ == 0U;
    const bool synthetic_authority =
        authority_domain_ ==
            Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain::
                kSyntheticT1 &&
        execution_package_identity_ == 0U &&
        full_attention_catalog_identity_ == 0U &&
        full_attention_binding_identity_ == 0U &&
        mlp_binding_identity_ == 0U && input_norm_binding_identity_ == 0U &&
        post_norm_binding_identity_ == 0U && rope_binding_identity_ == 0U &&
        resource_bundle_identity_ == 0U && synthetic_source_identity_ != 0U;
    return transaction_identity_ != 0U && owner_identity_ != 0U &&
           request_epoch_ != 0U && panel_ < kSm87MacroFeedV4PanelCount &&
           panel_generation_ != 0U &&
           first_position_ ==
               panel_ * kernels::kSm87MacroFeedV4Fp8Tokens &&
           grant_identity_ != 0U &&
           kv_allocation_identity_ != 0U &&
           full_attention_ordinal_ <
               kSm87MacroFeedV4FullAttentionLayerCount &&
           model_layer_ == 4U * full_attention_ordinal_ + 3U &&
           key_full_allocation_origin_ ==
               full_attention_ordinal_ *
                   kSm87MacroFeedV4AttentionKvLayerBytes &&
           value_full_allocation_origin_ ==
               key_full_allocation_origin_ +
                   kSm87MacroFeedV4AttentionKvPlaneBytes &&
           key_panel_allocation_offset_ ==
               key_full_allocation_origin_ +
                   panel_ * kSm87MacroFeedV4AttentionKvPanelBytes &&
           value_panel_allocation_offset_ ==
               value_full_allocation_origin_ +
                   panel_ * kSm87MacroFeedV4AttentionKvPanelBytes &&
           kv_panel_bytes_ == kSm87MacroFeedV4AttentionKvPanelBytes &&
           previous_valid_end_ == first_position_ &&
           candidate_end_ ==
               first_position_ + kSm87MacroFeedV4PanelTokens &&
           (normal_authority || synthetic_authority) &&
           submission_digest_ != 0U &&
           input_norm_launches_ == 1U && full_qkv_launches_ == 1U &&
           preprocess_launches_ == 1U && attention_launches_ == 1U &&
           full_output_launches_ == 1U &&
           residual_post_norm_launches_ == 1U &&
           gate_up_launches_ == 1U && down_launches_ == 1U &&
           bound_kernel_submissions_ == 8U &&
           asynchronous_d2d_copies_ == 0U &&
           asynchronous_d2d_copy_bytes_ == 0U &&
           complete_layer_enqueued_ &&
           !physical_device_completion_attested_ && !panel_complete_ &&
           !production_receipt_eligible_;
  }

 private:
  std::uint64_t transaction_identity_ = 0U;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t request_epoch_ = 0U;
  std::size_t panel_ = kSm87MacroFeedV4PanelCount;
  std::uint64_t panel_generation_ = 0U;
  std::size_t first_position_ = kSm87MacroFeedV4P40Tokens;
  std::uint64_t grant_identity_ = 0U;
  std::uint64_t grant_state_epoch_ = 0U;
  std::uint64_t kv_allocation_identity_ = 0U;
  std::uint64_t key_full_allocation_origin_ = 0U;
  std::uint64_t value_full_allocation_origin_ = 0U;
  std::uint64_t key_panel_allocation_offset_ = 0U;
  std::uint64_t value_panel_allocation_offset_ = 0U;
  std::uint64_t kv_panel_bytes_ = 0U;
  std::size_t previous_valid_end_ = kSm87MacroFeedV4P40Tokens;
  std::size_t candidate_end_ = kSm87MacroFeedV4P40Tokens;
  std::size_t full_attention_ordinal_ =
      kSm87MacroFeedV4FullAttentionLayerCount;
  std::size_t model_layer_ = kSm87MacroFeedV4LayerCount;
  Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain authority_domain_ =
      Sm87MacroFeedV4FullAttentionSubmissionAuthorityDomain::kInvalid;
  std::uint64_t execution_package_identity_ = 0U;
  std::uint64_t full_attention_catalog_identity_ = 0U;
  std::uint64_t full_attention_binding_identity_ = 0U;
  std::uint64_t mlp_binding_identity_ = 0U;
  std::uint64_t input_norm_binding_identity_ = 0U;
  std::uint64_t post_norm_binding_identity_ = 0U;
  std::uint64_t rope_binding_identity_ = 0U;
  std::uint64_t resource_bundle_identity_ = 0U;
  std::uint64_t synthetic_source_identity_ = 0U;
  std::uint64_t submission_digest_ = 0U;
  std::size_t input_norm_launches_ = 0U;
  std::size_t full_qkv_launches_ = 0U;
  std::size_t preprocess_launches_ = 0U;
  std::size_t attention_launches_ = 0U;
  std::size_t full_output_launches_ = 0U;
  std::size_t residual_post_norm_launches_ = 0U;
  std::size_t gate_up_launches_ = 0U;
  std::size_t down_launches_ = 0U;
  std::size_t bound_kernel_submissions_ = 0U;
  std::size_t asynchronous_d2d_copies_ = 0U;
  std::uint64_t asynchronous_d2d_copy_bytes_ = 0U;
  std::uint64_t authenticator_ = 0U;
  bool complete_layer_enqueued_ = false;
  bool physical_device_completion_attested_ = false;
  bool panel_complete_ = false;
  bool production_receipt_eligible_ = false;

  friend class Sm87MacroFeedV4ExecutionEventsOwner;
};

struct Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueResult final {
  Sm87MacroFeedV4ExecutionStatus status{};
  Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.valid_shape();
  }
};
#endif

struct Sm87MacroFeedV4ExecutionEventsSnapshot final {
  Sm87MacroFeedV4ExecutionOwnerState state =
      Sm87MacroFeedV4ExecutionOwnerState::kEmpty;
  std::uint64_t owner_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::uint64_t request_owner_identity = 0U;
  std::uint64_t request_allocation_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::size_t completed_panels = 0U;
  std::size_t active_panel = kSm87MacroFeedV4PanelCount;
  std::uint64_t active_panel_generation = 0U;
  std::size_t bf16_ab_cycles_completed = 0U;
  std::array<std::uint64_t, kSm87MacroFeedV4ExecutionEventCount>
      event_generations{};
  std::size_t enqueue_receipts_issued = 0U;
  std::size_t physical_completion_receipts_issued = 0U;
  std::size_t bound_kernel_submissions = 0U;
  std::size_t input_norm_submissions = 0U;
  std::size_t bf16_ab_submissions = 0U;
  std::size_t gdn_qkvz_c8000_submissions = 0U;
  std::size_t gdn_qkvz_ab_ready_wait_transactions = 0U;
  std::size_t gdn_continuation_c8000_submissions = 0U;
  std::size_t gdn_history_d2d_copies = 0U;
  std::uint64_t gdn_history_d2d_bytes = 0U;
  std::size_t gdn_output_c8000_submissions = 0U;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  std::size_t full_qkv_c8000_submissions = 0U;
  std::size_t full_attention_preprocess_c8000_submissions = 0U;
  std::size_t attention_c8000_submissions = 0U;
  std::size_t full_attention_output_c8000_submissions = 0U;
#endif
  std::size_t residual_post_norm_submissions = 0U;
  std::size_t gate_up_c8000_submissions = 0U;
  std::size_t down_c8000_submissions = 0U;
  std::size_t complete_gdn_layers_submitted = 0U;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  // Numbers of per-panel/layer GDN and Full grant slots reserved before each
  // transaction's first enqueue.  A failed transaction retains its reservation
  // because the request is then terminally poisoned; these are at-most-once
  // authority counts, not success counts.
  std::size_t accepted_gdn_grants = 0U;
  Sm87MacroFeedV4GdnAcceptedPrefixLedger last_gdn_accepted_prefix{};
  std::size_t complete_full_attention_layers_submitted = 0U;
  std::size_t accepted_full_attention_grants = 0U;
  Sm87MacroFeedV4FullAttentionAcceptedPrefixLedger
      last_full_attention_accepted_prefix{};
#endif
  std::size_t cold_recurrent_initializations = 0U;
  std::uint64_t cold_recurrent_allocation_identity = 0U;
  std::uintptr_t cold_recurrent_allocation_begin = 0U;
  std::uint64_t cold_recurrent_zero_bytes = 0U;
  std::size_t runtime_cold_rearms = 0U;
  std::uint64_t runtime_recurrent_zero_bytes = 0U;
  std::uint64_t last_rearmed_previous_request_epoch = 0U;
  std::uint64_t last_rearmed_request_epoch = 0U;
  std::size_t panel_commit_audit_receipts_issued = 0U;
  bool streams_nonblocking = false;
  bool bf16_ab_cycle_at_norm_boundary = true;
  bool panel_done_recorded = false;
  bool main_tail_recorded = false;
  bool ab_tail_recorded = false;
  bool main_tail_joined = false;
  bool ab_tail_joined = false;
  bool owner_drained_recorded = false;
  bool final_representation_ready_recorded = false;
  bool final_representation_joined = false;
  bool canonical_copy_done_recorded = false;
  bool canonical_copy_joined = false;
  bool final_publish_recorded = false;
  Sm87MacroFeedV4ExecutionStatus poison_cause{};
  std::array<int, kSm87MacroFeedV4ExecutionStreamCount>
      poison_drain_stream_cuda_status{};
  bool poison_drain_all_stream_synchronizations_attempted = false;
  bool poisoned_terminal_quiescence_attested = false;
  std::uint64_t poisoned_terminal_quiescence_identity = 0U;
  bool default_off = true;
  bool selector_bound = false;
  bool launcher_bound = false;
  bool jit_present = false;
  bool request_time_repack_present = false;
  bool request_time_autotune_present = false;
  bool cublaslt_present = false;
  bool mtp_present = false;
  bool production_dispatch_eligible = false;
};

struct Sm87MacroFeedV4ExecutionEventsCreateResult final {
  std::unique_ptr<Sm87MacroFeedV4ExecutionEventsOwner> owner;
  Sm87MacroFeedV4ExecutionStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept;
};

class Sm87MacroFeedV4ExecutionEventsOwner final {
 public:
  Sm87MacroFeedV4ExecutionEventsOwner(
      const Sm87MacroFeedV4ExecutionEventsOwner&) = delete;
  Sm87MacroFeedV4ExecutionEventsOwner& operator=(
      const Sm87MacroFeedV4ExecutionEventsOwner&) = delete;
  Sm87MacroFeedV4ExecutionEventsOwner(
      Sm87MacroFeedV4ExecutionEventsOwner&&) = delete;
  Sm87MacroFeedV4ExecutionEventsOwner& operator=(
      Sm87MacroFeedV4ExecutionEventsOwner&&) = delete;
  ~Sm87MacroFeedV4ExecutionEventsOwner();

  [[nodiscard]] Sm87MacroFeedV4ExecutionEventsSnapshot snapshot()
      const noexcept;

 private:
  [[nodiscard]] const Sm87MacroFeedV4ExecutionEventsAccess* access()
      const noexcept {
    return access_.get();
  }

  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus begin_request(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept;
  [[nodiscard]] Sm87MacroFeedV4ColdRequestRearmResult rearm_cold_request(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& previous_request_access)
      noexcept;
  [[nodiscard]] Sm87MacroFeedV4PanelBeginResult begin_panel(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      std::size_t panel) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PanelBeginResult begin_panel_with_state(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      std::size_t panel) noexcept;

  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult record_event(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream producer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult wait_event(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream consumer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus
  initialize_cold_recurrent_storage(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      void* recurrent_allocation, std::size_t recurrent_bytes,
      std::uint64_t recurrent_allocation_identity) noexcept;

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  // The exact body enqueue and its ready-event record are one owner-locked
  // transaction.  The package never receives or stores a raw CUDA stream.
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_input_norm_and_record_ready(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::Sm87MacroFeedV4InputNormArguments& arguments,
      const kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&
          resources) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_bf16_ab_and_record_ready(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::Sm87MacroFeedV4Bf16AbArguments& arguments,
      const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
          resources) noexcept;
  // At ExpectAbWait, Main submits exactly one fixed GDN-QKVZ C8000 kernel and
  // then waits on the current AbReady generation under this same owner lock.
  // No caller can interleave stream work between those two CUDA submissions.
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_gdn_qkvz_c8000_then_wait_ab_ready(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::sm87_macrofeed_v4_bound_launch_detail::
          Sm87MacroFeedV4GdnQkvzC8000Arguments& arguments,
      const kernels::Sm87MacroFeedV4Fp8CudaResources& resources) noexcept;
  [[nodiscard]] Sm87MacroFeedV4CompleteGdnLayerEnqueueResult
  submit_complete_gdn_layer_c8000_prevalidated(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
      const Sm87MacroFeedV4CompleteGdnLayerC8000Submission& submission)
      noexcept;
  [[nodiscard]] bool gdn_receipt_matches(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
      const Sm87MacroFeedV4CompleteGdnLayerC8000Submission&
          expected_submission,
      const Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt& receipt)
      const noexcept;
  [[nodiscard]] Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueResult
  submit_complete_full_attention_layer_c8000_prevalidated(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
      const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
          submission) noexcept;
  [[nodiscard]] bool full_attention_receipt_matches(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
      const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
          expected_submission,
      const Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt& receipt)
      const noexcept;
#endif

  [[nodiscard]] Sm87MacroFeedV4PhysicalObservationResult
  observe_event_query(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PhysicalObservationResult
  observe_event_synchronize(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;

  [[nodiscard]] bool completion_receipt_matches(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent expected_event,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const noexcept;

  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus close_panel(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access) noexcept;

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  [[nodiscard]] Sm87MacroFeedV4PanelCloseCommitResult
  close_panel_and_commit_state(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept;
#endif

  // A candidate epoch may be discarded only after both producer streams have
  // recorded tails, Control has waited for both, OwnerDrained has been
  // recorded, and that exact OwnerDrained generation has physically completed.
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus discard_after_drain(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained) noexcept;

  // Whole-owner terminal transition.  The opaque physical receipt is checked
  // while the EventsOwner lock is held, then this owner directly discards the
  // matching RequestState candidate before retiring its own live generation.
  // No caller-fillable receipt identity crosses that authority boundary.
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus
  discard_request_state_after_drain(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;

  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus
  discard_final_request_state_after_drain(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& final_panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;

  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus complete_request(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& final_panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& final_publish) noexcept;

  // Terminal-only recovery for a CUDA submission/observation failure.  It
  // physically synchronizes every owned stream, requires the request data to
  // be discarded, and keeps this CUDA owner permanently poisoned.  The
  // original cause remains separate from drain success so failure evidence
  // cannot be swallowed or cleared by a later request.
  [[nodiscard]] Sm87MacroFeedV4PoisonDrainResult drain_poisoned_request(
      const Sm87MacroFeedV4ExecutionEventsAccess& access) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PoisonDrainResult
  drain_poisoned_request_and_discard(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;

  struct EventState final {
    std::uint64_t generation = 0U;
    std::uint64_t request_epoch = 0U;
    std::size_t panel = kSm87MacroFeedV4PanelCount;
    std::uint64_t panel_generation = 0U;
    Sm87MacroFeedV4ExecutionStream producer =
        Sm87MacroFeedV4ExecutionStream::kCount;
    bool recorded = false;
    bool dependency_observed = false;
    bool physical_observed = false;
  };

  enum class AbCyclePhase : std::uint8_t {
    kExpectNormRecord = 0U,
    kExpectNormWait,
    kExpectAbRecord,
    kExpectAbWait,
  };

  struct PanelLedgerBaseline final {
    std::size_t bf16_ab_cycles = 0U;
    std::size_t bound_kernels = 0U;
    std::size_t input_norm = 0U;
    std::size_t bf16_ab = 0U;
    std::size_t gdn_qkvz = 0U;
    std::size_t gdn_qkvz_waits = 0U;
    std::size_t gdn_continuation = 0U;
    std::size_t gdn_copies = 0U;
    std::uint64_t gdn_copy_bytes = 0U;
    std::size_t gdn_output = 0U;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
    std::size_t full_qkv = 0U;
    std::size_t full_preprocess = 0U;
    std::size_t attention = 0U;
    std::size_t full_output = 0U;
#endif
    std::size_t residual_post_norm = 0U;
    std::size_t gate_up = 0U;
    std::size_t down = 0U;
    std::size_t complete_gdn = 0U;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
    std::size_t accepted_gdn = 0U;
    std::size_t complete_full = 0U;
    std::size_t accepted_full = 0U;
#endif
  };

  Sm87MacroFeedV4ExecutionEventsOwner() noexcept;

  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus initialize() noexcept;
  [[nodiscard]] Sm87MacroFeedV4PanelBeginResult begin_panel_locked(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      std::size_t panel) noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionPanelAccess
  activate_panel_locked(std::size_t panel) noexcept;
  [[nodiscard]] bool owner_access_matches(
      const Sm87MacroFeedV4ExecutionEventsAccess& access) const noexcept;
  [[nodiscard]] bool panel_access_matches(
      const Sm87MacroFeedV4ExecutionPanelAccess& access) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus validate_operation_access(
      const Sm87MacroFeedV4ExecutionEventsAccess& owner_access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus validate_record_order(
      Sm87MacroFeedV4ExecutionStream producer,
      Sm87MacroFeedV4ExecutionEvent event) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus validate_wait_order(
      Sm87MacroFeedV4ExecutionStream consumer,
      Sm87MacroFeedV4ExecutionEvent event) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult record_event_locked(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream producer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult wait_event_locked(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream consumer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_input_norm_and_record_ready_locked(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::Sm87MacroFeedV4InputNormArguments& arguments,
      const kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&
          resources) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_bf16_ab_and_record_ready_locked(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::Sm87MacroFeedV4Bf16AbArguments& arguments,
      const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
          resources) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_gdn_qkvz_c8000_then_wait_ab_ready_locked(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::sm87_macrofeed_v4_bound_launch_detail::
          Sm87MacroFeedV4GdnQkvzC8000Arguments& arguments,
      const kernels::Sm87MacroFeedV4Fp8CudaResources& resources) noexcept;
#endif
  void advance_record_order(Sm87MacroFeedV4ExecutionEvent event) noexcept;
  void advance_wait_order(Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueReceipt mint_enqueue_receipt(
      Sm87MacroFeedV4EnqueueOperation operation,
      Sm87MacroFeedV4ExecutionStream stream,
      Sm87MacroFeedV4ExecutionEvent event,
      std::uint64_t event_generation) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PhysicalCompletionReceipt
  mint_completion_receipt(Sm87MacroFeedV4ExecutionEvent event,
                          bool observed_by_query,
                          bool observed_by_synchronize) noexcept;
  [[nodiscard]] std::uint64_t completion_authenticator(
      const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const noexcept;
  [[nodiscard]] bool completion_receipt_matches_locked(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent expected_event,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const noexcept;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus
  validate_exact_panel_ledger_locked() const noexcept;
  [[nodiscard]] Sm87MacroFeedV4PanelCommitAuditReceipt
  mint_panel_commit_audit_receipt_locked(
      std::uint64_t panel_done_generation) noexcept;
  [[nodiscard]] std::uint64_t panel_commit_audit_authenticator(
      const Sm87MacroFeedV4PanelCommitAuditReceipt& receipt) const noexcept;
#endif
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  [[nodiscard]] std::uint64_t gdn_submission_digest(
      const Sm87MacroFeedV4CompleteGdnLayerC8000Submission& submission)
      const noexcept;
  [[nodiscard]] std::uint64_t gdn_receipt_authenticator(
      const Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt& receipt)
      const noexcept;
  [[nodiscard]] bool gdn_receipt_matches_locked(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
      const Sm87MacroFeedV4CompleteGdnLayerC8000Submission&
          expected_submission,
      const Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt& receipt)
      const noexcept;
  [[nodiscard]] std::uint64_t full_attention_submission_digest(
      const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
          submission) const noexcept;
  [[nodiscard]] std::uint64_t full_attention_receipt_authenticator(
      const Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt& receipt)
      const noexcept;
  [[nodiscard]] bool full_attention_receipt_matches_locked(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
      const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
          expected_submission,
      const Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt& receipt)
      const noexcept;
#endif
  void record_poison_cause(
      const Sm87MacroFeedV4ExecutionStatus& cause) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PoisonDrainResult
  drain_poisoned_request_locked(
      const Sm87MacroFeedV4ExecutionEventsAccess& access,
      Sm87MacroFeedV4RequestState* request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess* request_access,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;
  void reset_request_ledger() noexcept;
  void release_resources() noexcept;

  Sm87MacroFeedV4ExecutionOwnerState state_ =
      Sm87MacroFeedV4ExecutionOwnerState::kEmpty;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t seal_nonce_ = 0U;
  std::uint64_t receipt_secret_ = 0U;
  std::int32_t device_ordinal_ = -1;
  std::array<void*, kSm87MacroFeedV4ExecutionStreamCount> streams_{};
  std::array<void*, kSm87MacroFeedV4ExecutionEventCount> events_{};
  std::unique_ptr<Sm87MacroFeedV4ExecutionEventsAccess> access_;
  mutable std::mutex mutex_;
  std::uint64_t request_owner_identity_ = 0U;
  std::uint64_t request_allocation_identity_ = 0U;
  std::uint64_t request_epoch_ = 0U;
  std::uint64_t last_request_epoch_ = 0U;
  std::size_t completed_panels_ = 0U;
  std::size_t active_panel_ = kSm87MacroFeedV4PanelCount;
  std::uint64_t next_panel_generation_ = 1U;
  std::uint64_t active_panel_generation_ = 0U;
  std::array<EventState, kSm87MacroFeedV4ExecutionEventCount> event_state_{};
  AbCyclePhase ab_cycle_phase_ = AbCyclePhase::kExpectNormRecord;
  std::size_t bf16_ab_cycles_completed_ = 0U;
  std::size_t enqueue_receipts_issued_ = 0U;
  std::size_t physical_completion_receipts_issued_ = 0U;
  std::size_t bound_kernel_submissions_ = 0U;
  std::size_t input_norm_submissions_ = 0U;
  std::size_t bf16_ab_submissions_ = 0U;
  std::size_t gdn_qkvz_c8000_submissions_ = 0U;
  std::size_t gdn_qkvz_ab_ready_wait_transactions_ = 0U;
  std::size_t gdn_continuation_c8000_submissions_ = 0U;
  std::size_t gdn_history_d2d_copies_ = 0U;
  std::uint64_t gdn_history_d2d_bytes_ = 0U;
  std::size_t gdn_output_c8000_submissions_ = 0U;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  std::size_t full_qkv_c8000_submissions_ = 0U;
  std::size_t full_attention_preprocess_c8000_submissions_ = 0U;
  std::size_t attention_c8000_submissions_ = 0U;
  std::size_t full_attention_output_c8000_submissions_ = 0U;
#endif
  std::size_t residual_post_norm_submissions_ = 0U;
  std::size_t gate_up_c8000_submissions_ = 0U;
  std::size_t down_c8000_submissions_ = 0U;
  std::size_t complete_gdn_layers_submitted_ = 0U;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  std::array<std::uint64_t,
             kSm87MacroFeedV4PanelCount * kSm87MacroFeedV4StateLayerCount>
      accepted_gdn_grant_identities_{};
  std::size_t accepted_gdn_grant_count_ = 0U;
  Sm87MacroFeedV4GdnAcceptedPrefixLedger last_gdn_accepted_prefix_{};
  std::size_t complete_full_attention_layers_submitted_ = 0U;
  std::array<std::uint64_t,
             kSm87MacroFeedV4PanelCount *
                 kSm87MacroFeedV4FullAttentionLayerCount>
      accepted_full_attention_grant_identities_{};
  std::size_t accepted_full_attention_grant_count_ = 0U;
  Sm87MacroFeedV4FullAttentionAcceptedPrefixLedger
      last_full_attention_accepted_prefix_{};
#endif
  std::size_t cold_recurrent_initializations_ = 0U;
  std::uint64_t cold_recurrent_allocation_identity_ = 0U;
  std::uintptr_t cold_recurrent_allocation_begin_ = 0U;
  std::uint64_t cold_recurrent_zero_bytes_ = 0U;
  std::size_t runtime_cold_rearms_ = 0U;
  std::uint64_t runtime_recurrent_zero_bytes_ = 0U;
  std::uint64_t last_rearmed_previous_request_epoch_ = 0U;
  std::uint64_t last_rearmed_request_epoch_ = 0U;
  std::size_t panel_commit_audit_receipts_issued_ = 0U;
  PanelLedgerBaseline panel_ledger_baseline_{};
  bool streams_nonblocking_ = false;
  bool panel_done_recorded_ = false;
  bool draining_ = false;
  bool main_tail_recorded_ = false;
  bool ab_tail_recorded_ = false;
  bool main_tail_joined_ = false;
  bool ab_tail_joined_ = false;
  bool owner_drained_recorded_ = false;
  bool final_representation_ready_recorded_ = false;
  bool final_representation_joined_ = false;
  bool canonical_copy_done_recorded_ = false;
  bool canonical_copy_joined_ = false;
  bool final_publish_recorded_ = false;
  Sm87MacroFeedV4ExecutionStatus poison_cause_{};
  std::array<int, kSm87MacroFeedV4ExecutionStreamCount>
      poison_drain_stream_cuda_status_{};
  bool poison_drain_all_stream_synchronizations_attempted_ = false;
  bool poisoned_terminal_quiescence_attested_ = false;
  std::uint64_t poisoned_terminal_quiescence_identity_ = 0U;
  // Test fixture fault injection only.  It is unreachable without the
  // header-only test capability and never changes a production route.
  bool test_fail_next_bound_ab_wait_ = false;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  std::size_t test_fail_gdn_after_accepted_operation_ =
      std::numeric_limits<std::size_t>::max();
  std::size_t test_fail_full_after_accepted_prefix_ =
      std::numeric_limits<std::size_t>::max();
#endif

  friend struct Sm87MacroFeedV4ExecutionEventsCreateResult;
  friend Sm87MacroFeedV4ExecutionEventsCreateResult
  create_sm87_macrofeed_v4_execution_events_owner() noexcept;
  friend class Sm87MacroFeedV4ExecutionEventsDriver;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  friend std::unique_ptr<Sm87MacroFeedV4ExecutionEventsDriver>
  bind_sm87_macrofeed_v4_execution_events_driver(
      const std::shared_ptr<Sm87MacroFeedV4ExecutionEventsOwner>& owner)
      noexcept;
#endif
  friend class Sm87MacroFeedV4ExecutionEventsCudaTestFixture;
};

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
// Narrow owner-issued execution capability.  It forwards only the fixed V4
// state machine and typed kernel submissions; no raw stream/event member or
// generic callback is reachable from the execution package.
class Sm87MacroFeedV4ExecutionEventsDriver final {
 public:
  Sm87MacroFeedV4ExecutionEventsDriver() = delete;
  Sm87MacroFeedV4ExecutionEventsDriver(
      const Sm87MacroFeedV4ExecutionEventsDriver&) = delete;
  Sm87MacroFeedV4ExecutionEventsDriver& operator=(
      const Sm87MacroFeedV4ExecutionEventsDriver&) = delete;
  Sm87MacroFeedV4ExecutionEventsDriver(
      Sm87MacroFeedV4ExecutionEventsDriver&&) = delete;
  Sm87MacroFeedV4ExecutionEventsDriver& operator=(
      Sm87MacroFeedV4ExecutionEventsDriver&&) = delete;
  ~Sm87MacroFeedV4ExecutionEventsDriver() = default;

  [[nodiscard]] std::uint64_t owner_identity() const noexcept;
  [[nodiscard]] std::int32_t device_ordinal() const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionEventsSnapshot snapshot()
      const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus begin_request(
      const Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PanelBeginResult begin_panel(
      std::size_t panel) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult record_event(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream producer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult wait_event(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream consumer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_input_norm_and_record_ready(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::Sm87MacroFeedV4InputNormArguments& arguments,
      const kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&
          resources) noexcept;
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_bf16_ab_and_record_ready(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::Sm87MacroFeedV4Bf16AbArguments& arguments,
      const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
          resources) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PhysicalObservationResult
  observe_event_synchronize(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent event) noexcept;
  [[nodiscard]] bool completion_receipt_matches(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent expected_event,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus discard_after_drain(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PoisonDrainResult drain_poisoned_request()
      noexcept;

 private:
  [[nodiscard]] Sm87MacroFeedV4ColdRequestRearmResult rearm_cold_request(
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& previous_request_access)
      noexcept;
  [[nodiscard]] Sm87MacroFeedV4PanelBeginResult begin_panel_with_state(
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      std::size_t panel) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PanelCloseCommitResult
  close_panel_and_commit_state(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus
  discard_final_request_state_after_drain(
      const Sm87MacroFeedV4ExecutionPanelAccess& final_panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;

  // Package-authority-only transaction.  Raw asset/resource values cannot be
  // submitted through the caller-visible driver surface: the sole non-test
  // friend copies them from the construction-sealed 48-layer catalog.
  [[nodiscard]] Sm87MacroFeedV4EventEnqueueResult
  submit_gdn_qkvz_c8000_then_wait_ab_ready(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::sm87_macrofeed_v4_bound_launch_detail::
          Sm87MacroFeedV4GdnQkvzC8000Arguments& arguments,
      const kernels::Sm87MacroFeedV4Fp8CudaResources& resources) noexcept;
  [[nodiscard]] Sm87MacroFeedV4CompleteGdnLayerEnqueueResult
  submit_complete_gdn_layer_c8000_prevalidated(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
      const Sm87MacroFeedV4CompleteGdnLayerC8000Submission& submission)
      noexcept;
  [[nodiscard]] bool gdn_receipt_matches(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
      const Sm87MacroFeedV4CompleteGdnLayerC8000Submission&
          expected_submission,
      const Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt& receipt)
      const noexcept;
  [[nodiscard]] Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueResult
  submit_complete_full_attention_layer_c8000_prevalidated(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
      const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
          submission) noexcept;
  [[nodiscard]] bool full_attention_receipt_matches(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
      const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
          expected_submission,
      const Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt& receipt)
      const noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus
  initialize_cold_recurrent_storage(
      void* recurrent_allocation, std::size_t recurrent_bytes,
      std::uint64_t recurrent_allocation_identity) noexcept;
  [[nodiscard]] Sm87MacroFeedV4ExecutionStatus
  discard_request_state_after_drain(
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;
  [[nodiscard]] Sm87MacroFeedV4PoisonDrainResult
  drain_poisoned_request_and_discard(
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;

  explicit Sm87MacroFeedV4ExecutionEventsDriver(
      std::shared_ptr<Sm87MacroFeedV4ExecutionEventsOwner> owner,
      const Sm87MacroFeedV4ExecutionEventsAccess* access) noexcept
      : owner_(std::move(owner)), access_(access) {}

  std::shared_ptr<Sm87MacroFeedV4ExecutionEventsOwner> owner_;
  const Sm87MacroFeedV4ExecutionEventsAccess* access_ = nullptr;

  friend std::unique_ptr<Sm87MacroFeedV4ExecutionEventsDriver>
  bind_sm87_macrofeed_v4_execution_events_driver(
      const std::shared_ptr<Sm87MacroFeedV4ExecutionEventsOwner>& owner)
      noexcept;
  friend class ::q3x::runtime::sm87_macrofeed_v4_p40_execution_detail::
      Sm87MacroFeedV4P40ExecutionPackage;
};

[[nodiscard]] std::unique_ptr<Sm87MacroFeedV4ExecutionEventsDriver>
bind_sm87_macrofeed_v4_execution_events_driver(
    const std::shared_ptr<Sm87MacroFeedV4ExecutionEventsOwner>& owner)
    noexcept;
#endif

// Default-off admission factory.  No selector, launcher, public CUDA handle,
// production receipt, JIT, repack, autotune, cuBLASLt, or MTP route is exposed.
[[nodiscard]] Sm87MacroFeedV4ExecutionEventsCreateResult
create_sm87_macrofeed_v4_execution_events_owner() noexcept;

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_EXECUTION_EVENTS_TEST_FIXTURE)
// Device-correctness-only wrappers for private transitions.  They expose no
// CUDA handle and are absent from non-test compilation.
class Sm87MacroFeedV4ExecutionEventsCudaTestFixture final {
 public:
  [[nodiscard]] static std::uint64_t owner_identity(
      const Sm87MacroFeedV4ExecutionEventsOwner& owner) noexcept {
    std::lock_guard<std::mutex> lock(owner.mutex_);
    return owner.access_ == nullptr ? 0U : owner.owner_identity_;
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus begin_request(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept {
    return owner.begin_request(*owner.access_, request_owner, request_access);
  }
  [[nodiscard]] static Sm87MacroFeedV4ColdRequestRearmResult
  rearm_cold_request(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& previous_request_access)
      noexcept {
    return owner.rearm_cold_request(*owner.access_, request_owner,
                                    previous_request_access);
  }
  [[nodiscard]] static Sm87MacroFeedV4PanelBeginResult begin_panel(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const std::size_t panel) noexcept {
    return owner.begin_panel(*owner.access_, panel);
  }
  [[nodiscard]] static Sm87MacroFeedV4PanelBeginResult
  begin_panel_with_state(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      const std::size_t panel) noexcept {
    return owner.begin_panel_with_state(*owner.access_, request_owner,
                                        request_access, panel);
  }
  [[nodiscard]] static Sm87MacroFeedV4EventEnqueueResult record_event(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream producer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept {
    return owner.record_event(*owner.access_, panel_access, producer, event);
  }
  [[nodiscard]] static Sm87MacroFeedV4EventEnqueueResult wait_event(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionStream consumer,
      Sm87MacroFeedV4ExecutionEvent event) noexcept {
    return owner.wait_event(*owner.access_, panel_access, consumer, event);
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus
  initialize_cold_recurrent_storage(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      void* recurrent_allocation, const std::size_t recurrent_bytes,
      const std::uint64_t recurrent_allocation_identity) noexcept {
    return owner.initialize_cold_recurrent_storage(
        *owner.access_, recurrent_allocation, recurrent_bytes,
        recurrent_allocation_identity);
  }
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  // Focused lifecycle-test seam.  Existing transaction tests prove each
  // physical counter increment; this bounded helper composes their exact
  // successful panel ledger without enqueueing 560 expensive kernels again.
  // It is absent from production and cannot mint a panel receipt or mutate
  // RequestState.
  [[nodiscard]] static bool seed_exact_panel_ledger_for_atomic_close(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const bool omit_one_bound_kernel = false) noexcept {
    std::lock_guard<std::mutex> lock(owner.mutex_);
    if (owner.state_ !=
            Sm87MacroFeedV4ExecutionOwnerState::kRequestActive ||
        owner.active_panel_ >= kSm87MacroFeedV4PanelCount ||
        owner.panel_done_recorded_ || owner.draining_) {
      return false;
    }
    constexpr std::size_t kGdn = kSm87MacroFeedV4StateLayerCount;
    constexpr std::size_t kFull = kSm87MacroFeedV4FullAttentionLayerCount;
    constexpr std::size_t kLayers = kSm87MacroFeedV4LayerCount;
    constexpr std::size_t kKernels = 560U;
    const auto& baseline = owner.panel_ledger_baseline_;
    owner.bf16_ab_cycles_completed_ = kGdn;
    owner.ab_cycle_phase_ =
        Sm87MacroFeedV4ExecutionEventsOwner::AbCyclePhase::kExpectNormRecord;
    owner.bound_kernel_submissions_ =
        baseline.bound_kernels + kKernels -
        static_cast<std::size_t>(omit_one_bound_kernel);
    owner.input_norm_submissions_ = baseline.input_norm + kLayers;
    owner.bf16_ab_submissions_ = baseline.bf16_ab + kGdn;
    owner.gdn_qkvz_c8000_submissions_ = baseline.gdn_qkvz + kGdn;
    owner.gdn_qkvz_ab_ready_wait_transactions_ =
        baseline.gdn_qkvz_waits + kGdn;
    owner.gdn_continuation_c8000_submissions_ =
        baseline.gdn_continuation + 2U * kGdn;
    owner.gdn_history_d2d_copies_ = baseline.gdn_copies + kGdn;
    owner.gdn_history_d2d_bytes_ =
        baseline.gdn_copy_bytes + kSm87MacroFeedV4ConvEpochBytes;
    owner.gdn_output_c8000_submissions_ = baseline.gdn_output + kGdn;
    owner.full_qkv_c8000_submissions_ = baseline.full_qkv + kFull;
    owner.full_attention_preprocess_c8000_submissions_ =
        baseline.full_preprocess + kFull;
    owner.attention_c8000_submissions_ = baseline.attention + kFull;
    owner.full_attention_output_c8000_submissions_ =
        baseline.full_output + kFull;
    owner.residual_post_norm_submissions_ =
        baseline.residual_post_norm + kLayers;
    owner.gate_up_c8000_submissions_ = baseline.gate_up + kLayers;
    owner.down_c8000_submissions_ = baseline.down + kLayers;
    owner.complete_gdn_layers_submitted_ = baseline.complete_gdn + kGdn;
    owner.accepted_gdn_grant_count_ = baseline.accepted_gdn + kGdn;
    owner.complete_full_attention_layers_submitted_ =
        baseline.complete_full + kFull;
    owner.accepted_full_attention_grant_count_ =
        baseline.accepted_full + kFull;

    const std::size_t gdn_begin = owner.active_panel_ * kGdn;
    for (std::size_t ordinal = 0U; ordinal < kGdn; ++ordinal) {
      owner.accepted_gdn_grant_identities_[gdn_begin + ordinal] =
          1U + static_cast<std::uint64_t>(gdn_begin + ordinal);
    }
    const std::size_t full_begin = owner.active_panel_ * kFull;
    for (std::size_t ordinal = 0U; ordinal < kFull; ++ordinal) {
      owner.accepted_full_attention_grant_identities_[full_begin + ordinal] =
          1U + static_cast<std::uint64_t>(full_begin + ordinal);
    }

    auto& gdn = owner.last_gdn_accepted_prefix_;
    gdn = {};
    gdn.transaction_identity = 1U;
    gdn.owner_identity = owner.owner_identity_;
    gdn.request_epoch = owner.request_epoch_;
    gdn.panel = owner.active_panel_;
    gdn.panel_generation = owner.active_panel_generation_;
    gdn.grant_identity = owner.accepted_gdn_grant_identities_[
        gdn_begin + kGdn - 1U];
    gdn.recurrent_allocation_identity =
        owner.request_allocation_identity_;
    gdn.gdn_ordinal = kGdn - 1U;
    gdn.model_layer = kLayers - 2U;
    gdn.active_bank_index = owner.active_panel_ % 2U;
    gdn.candidate_bank_index = 1U - gdn.active_bank_index;
    gdn.conv_bytes = kernels::kSm87MacroFeedV4GdnConvHistoryBytes;
    gdn.gdn_state_bytes = kernels::kSm87MacroFeedV4GdnStateBytes;
    gdn.input_norm_launches = 1U;
    gdn.bf16_ab_launches = 1U;
    gdn.gdn_qkvz_launches = 1U;
    gdn.gdn_continuation_launches = 2U;
    gdn.gdn_output_launches = 1U;
    gdn.residual_post_norm_launches = 1U;
    gdn.gate_up_launches = 1U;
    gdn.down_launches = 1U;
    gdn.accepted_kernel_launches = 9U;
    gdn.asynchronous_d2d_copies = 1U;
    gdn.conv_history_copy_bytes =
        kernels::kSm87MacroFeedV4GdnConvHistoryBytes;
    gdn.complete = true;

    auto& full = owner.last_full_attention_accepted_prefix_;
    full = {};
    full.transaction_identity = 1U;
    full.owner_identity = owner.owner_identity_;
    full.request_epoch = owner.request_epoch_;
    full.panel = owner.active_panel_;
    full.panel_generation = owner.active_panel_generation_;
    full.first_position =
        owner.active_panel_ * kernels::kSm87MacroFeedV4Fp8Tokens;
    full.grant_identity = owner.accepted_full_attention_grant_identities_[
        full_begin + kFull - 1U];
    full.kv_allocation_identity = 1U;
    full.full_attention_ordinal = kFull - 1U;
    full.model_layer = kLayers - 1U;
    full.input_norm_launches = 1U;
    full.full_qkv_launches = 1U;
    full.preprocess_launches = 1U;
    full.attention_launches = 1U;
    full.full_output_launches = 1U;
    full.residual_post_norm_launches = 1U;
    full.gate_up_launches = 1U;
    full.down_launches = 1U;
    full.accepted_kernel_launches = 8U;
    full.complete = true;
    return true;
  }

  // Explicit test-only bypass for exercising transaction failures.  This is
  // deliberately absent from the execution driver and from non-test builds.
  [[nodiscard]] static Sm87MacroFeedV4EventEnqueueResult
  submit_gdn_qkvz_c8000_then_wait_ab_ready(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const kernels::sm87_macrofeed_v4_bound_launch_detail::
          Sm87MacroFeedV4GdnQkvzC8000Arguments& arguments,
      const kernels::Sm87MacroFeedV4Fp8CudaResources& resources) noexcept {
    return owner.submit_gdn_qkvz_c8000_then_wait_ab_ready(
        *owner.access_, panel_access, arguments, resources);
  }
  [[nodiscard]] static Sm87MacroFeedV4CompleteGdnLayerEnqueueResult
  submit_complete_gdn_layer_c8000_prevalidated(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
      const Sm87MacroFeedV4CompleteGdnLayerC8000Submission& submission)
      noexcept {
    return owner.submit_complete_gdn_layer_c8000_prevalidated(
        *owner.access_, panel_access, gdn_grant, submission);
  }
  [[nodiscard]] static bool gdn_receipt_matches(
      const Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4GdnLayerStateGrant& gdn_grant,
      const Sm87MacroFeedV4CompleteGdnLayerC8000Submission&
          expected_submission,
      const Sm87MacroFeedV4CompleteGdnLayerEnqueueReceipt& receipt) noexcept {
    return owner.gdn_receipt_matches(*owner.access_, panel_access, gdn_grant,
                                     expected_submission, receipt);
  }
  [[nodiscard]] static
      Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueResult
  submit_complete_full_attention_layer_c8000_prevalidated(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
      const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
          submission) noexcept {
    return owner.submit_complete_full_attention_layer_c8000_prevalidated(
        *owner.access_, panel_access, kv_grant, submission);
  }
  [[nodiscard]] static bool full_attention_receipt_matches(
      const Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4FullAttentionKvGrant& kv_grant,
      const Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
          expected_submission,
      const Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt& receipt)
      noexcept {
    return owner.full_attention_receipt_matches(
        *owner.access_, panel_access, kv_grant, expected_submission, receipt);
  }
#endif
  [[nodiscard]] static Sm87MacroFeedV4PhysicalObservationResult
  observe_event_query(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent event) noexcept {
    return owner.observe_event_query(*owner.access_, panel_access, event);
  }
  [[nodiscard]] static Sm87MacroFeedV4PhysicalObservationResult
  observe_event_synchronize(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent event) noexcept {
    return owner.observe_event_synchronize(*owner.access_, panel_access,
                                           event);
  }
  [[nodiscard]] static bool completion_receipt_matches(
      const Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4ExecutionEvent expected_event,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& receipt) noexcept {
    return owner.completion_receipt_matches(*owner.access_, panel_access,
                                            expected_event, receipt);
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus close_panel(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access) noexcept {
    return owner.close_panel(*owner.access_, panel_access);
  }
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  [[nodiscard]] static Sm87MacroFeedV4PanelCloseCommitResult
  close_panel_and_commit_state(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept {
    return owner.close_panel_and_commit_state(
        *owner.access_, panel_access, request_owner, request_access);
  }
#endif
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus discard_after_drain(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained) noexcept {
    return owner.discard_after_drain(*owner.access_, panel_access,
                                     owner_drained);
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus
  discard_request_state_after_drain(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
    return owner.discard_request_state_after_drain(
        *owner.access_, panel_access, owner_drained, request_owner,
        request_access, reason);
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus
  discard_final_request_state_after_drain(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& final_panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& owner_drained,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access,
      const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
    return owner.discard_final_request_state_after_drain(
        *owner.access_, final_panel_access, owner_drained, request_owner,
        request_access, reason);
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus complete_request(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionPanelAccess& final_panel_access,
      const Sm87MacroFeedV4PhysicalCompletionReceipt& final_publish) noexcept {
    return owner.complete_request(*owner.access_, final_panel_access,
                                  final_publish);
  }
  [[nodiscard]] static Sm87MacroFeedV4ExecutionStatus
  inject_poison_without_drain(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionError poison_error) noexcept {
    std::lock_guard<std::mutex> lock(owner.mutex_);
    Sm87MacroFeedV4ExecutionStatus cause;
    if (owner.state_ !=
            Sm87MacroFeedV4ExecutionOwnerState::kRequestActive ||
        (poison_error != Sm87MacroFeedV4ExecutionError::kCudaSubmission &&
         poison_error != Sm87MacroFeedV4ExecutionError::kCudaObservation)) {
      cause.error = Sm87MacroFeedV4ExecutionError::kInvalidOwnerState;
      cause.context =
          "test_poison_requires_active_request_and_cuda_error";
      return cause;
    }
    cause.error = poison_error;
    cause.context = "test_only_injected_cuda_failure_without_drain";
    cause.cuda_error = 1;
    owner.record_poison_cause(cause);
    return cause;
  }
  [[nodiscard]] static Sm87MacroFeedV4PoisonDrainResult
  inject_poison_and_drain(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const Sm87MacroFeedV4ExecutionError poison_error) noexcept {
    {
      std::lock_guard<std::mutex> lock(owner.mutex_);
      if (owner.state_ !=
              Sm87MacroFeedV4ExecutionOwnerState::kRequestActive ||
          (poison_error !=
               Sm87MacroFeedV4ExecutionError::kCudaSubmission &&
           poison_error !=
               Sm87MacroFeedV4ExecutionError::kCudaObservation)) {
        Sm87MacroFeedV4PoisonDrainResult result;
        result.drain_status.error =
            Sm87MacroFeedV4ExecutionError::kInvalidOwnerState;
        result.drain_status.context =
            "test_poison_requires_active_request_and_cuda_error";
        return result;
      }
      Sm87MacroFeedV4ExecutionStatus cause;
      cause.error = poison_error;
      cause.context = "test_only_injected_cuda_failure";
      owner.record_poison_cause(cause);
    }
    return owner.drain_poisoned_request(*owner.access_);
  }
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  [[nodiscard]] static bool fail_next_bound_ab_wait(
      Sm87MacroFeedV4ExecutionEventsOwner& owner) noexcept {
    std::lock_guard<std::mutex> lock(owner.mutex_);
    if (owner.state_ !=
            Sm87MacroFeedV4ExecutionOwnerState::kRequestActive ||
        owner.test_fail_next_bound_ab_wait_) {
      return false;
    }
    owner.test_fail_next_bound_ab_wait_ = true;
    return true;
  }
  [[nodiscard]] static bool fail_full_after_accepted_prefix(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const std::size_t accepted_prefix) noexcept {
    std::lock_guard<std::mutex> lock(owner.mutex_);
    if (owner.state_ !=
            Sm87MacroFeedV4ExecutionOwnerState::kRequestActive ||
        accepted_prefix > 8U ||
        owner.test_fail_full_after_accepted_prefix_ !=
            std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    owner.test_fail_full_after_accepted_prefix_ = accepted_prefix;
    return true;
  }
  [[nodiscard]] static bool fail_gdn_after_accepted_operation(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      const std::size_t accepted_operation) noexcept {
    std::lock_guard<std::mutex> lock(owner.mutex_);
    // Four and five are continuation-internal physical prefixes.  The bound
    // launcher retains them on a genuine CUDA failure, but Events cannot
    // manufacture such a failure without lying about accepted device work.
    if (owner.state_ !=
            Sm87MacroFeedV4ExecutionOwnerState::kRequestActive ||
        accepted_operation > 9U || accepted_operation == 4U ||
        accepted_operation == 5U ||
        owner.test_fail_gdn_after_accepted_operation_ !=
            std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    owner.test_fail_gdn_after_accepted_operation_ = accepted_operation;
    return true;
  }
  [[nodiscard]] static Sm87MacroFeedV4PoisonDrainResult
  drain_poisoned_request(
      Sm87MacroFeedV4ExecutionEventsOwner& owner) noexcept {
    return owner.drain_poisoned_request(*owner.access_);
  }
  [[nodiscard]] static Sm87MacroFeedV4PoisonDrainResult
  drain_poisoned_request_and_discard(
      Sm87MacroFeedV4ExecutionEventsOwner& owner,
      Sm87MacroFeedV4RequestState& request_owner,
      const Sm87MacroFeedV4RequestStateSealedAccess& request_access) noexcept {
    return owner.drain_poisoned_request_and_discard(
        *owner.access_, request_owner, request_access,
        Sm87MacroFeedV4RequestDiscardReason::kFailed);
  }
#endif
};
#endif

}  // namespace q3x::runtime::sm87_macrofeed_v4_execution_events_detail
