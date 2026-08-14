#pragma once

#include "q3x/runtime/model_weights.h"
#include "sm87_bulk_dataflow_v2_p40_constituent_seal_internal.h"
#include "sm87_target_aot_projection_complete_execution_access_internal.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace q3x::runtime::sm87_bulk_v2_p40_composition_detail {

namespace owner_detail =
    q3x::runtime::sm87_bulk_v2_p40_owner_detail;

// This is a source-private development composition.  Its identity is minted
// exclusively from live ModelWeights attachment, request-allocation, owner,
// stream/event, linked whole-kernel and constituent-seal objects.  It is an
// accuracy-unqualified direction witness and can never authorize production
// dispatch or claim numerical qualification.
enum class Sm87BulkV2P40CompositionError : std::uint8_t {
  kNone = 0U,
  kProjectionAttachment,
  kProjectionCatalog,
  kOwnerResources,
  kOwnerDeviceMismatch,
  kRequestState,
  kWholeProjectionStartup,
  kModelBinding,
  kBf16AbSeal,
  kAttentionSeal,
  kGdnSessionSeal,
  kIdentityDerivation,
  kConstituentSealAllocation,
  kDirectionWitnessOwnerSeal,
  kRootAllocation,
};

struct Sm87BulkV2P40CompositionStatus final {
  Sm87BulkV2P40CompositionError error =
      Sm87BulkV2P40CompositionError::kNone;
  const char* context = "none";
  int cuda_error = 0;
  std::size_t constituent_index = 0U;

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == Sm87BulkV2P40CompositionError::kNone;
  }
};

// Capability-free report.  These fields are useful for startup diagnostics,
// but copying this structure grants no owner or enqueue authority.
struct Sm87BulkV2P40CompositionAudit final {
  owner_detail::Sm87BulkV2P40OwnerIdentity identity{};
  std::size_t projection_artifacts = 0U;
  std::size_t bf16_ab_bindings = 0U;
  std::size_t attention_bindings = 0U;
  std::size_t gdn_bindings = 0U;
  bool complete_projection_attachment_live = false;
  bool whole_projection_source_resource_root_live = false;
  bool request_state_owner_bound = false;
  bool bf16_ab_access_sealed = false;
  bool attention_access_sealed = false;
  bool gdn_session_sealed = false;
  bool direction_witness_constituent_seal_root_minted = false;
  bool owner_execution_access_issued = false;
  bool caller_identity_claims_accepted = true;
  bool caller_raw_capability_set_accepted = true;
  bool exact_numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] bool valid() const noexcept;
};

class Sm87BulkV2P40CompositionRoot;

struct Sm87BulkV2P40CompositionCreateResult final {
  std::unique_ptr<Sm87BulkV2P40CompositionRoot> root;
  Sm87BulkV2P40CompositionAudit audit{};
  Sm87BulkV2P40CompositionStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept;
};

// Sole lifetime owner for the V2 P40 startup composition.  ModelWeights is an
// engine-owned immutable borrow and must outlive this object; every capability
// derived from it, plus Owner, RequestState, streams/events/control, whole-
// projection startup root, BF16/Attention accesses and GDN session/receipt,
// is retained here for the complete borrow interval.
class Sm87BulkV2P40CompositionRoot final {
 public:
  using ProjectionAccess =
      target_aot_complete_execution_detail::
          Sm87TargetAotCompleteProjectionExecutionAccess;
  using WholeProjectionAccess =
      owner_detail::Sm87BulkV2P40WholeProjectionStartupAccess;
  using Bf16AbAccess =
      q3x::kernels::sm87_target_aot_bf16_ab_execution_detail::
          SealedInterleavedP40Access;
  using AttentionAccess =
      q3x::kernels::sm87_bulk_v2_attention_execution_detail::
          Sm87BulkV2AttentionSealedAccess;

  Sm87BulkV2P40CompositionRoot(
      const Sm87BulkV2P40CompositionRoot&) = delete;
  Sm87BulkV2P40CompositionRoot& operator=(
      const Sm87BulkV2P40CompositionRoot&) = delete;
  Sm87BulkV2P40CompositionRoot(Sm87BulkV2P40CompositionRoot&&) = delete;
  Sm87BulkV2P40CompositionRoot& operator=(
      Sm87BulkV2P40CompositionRoot&&) = delete;
  ~Sm87BulkV2P40CompositionRoot();

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const ModelWeights* model_weights() const noexcept {
    return model_weights_;
  }
  [[nodiscard]] owner_detail::Sm87BulkV2P40Owner* owner() noexcept {
    return owner_.get();
  }
  [[nodiscard]] const owner_detail::Sm87BulkV2P40Owner* owner()
      const noexcept {
    return owner_.get();
  }
  [[nodiscard]] owner_detail::Sm87BulkV2P40RequestState* request_state()
      noexcept {
    return request_state_.get();
  }
  [[nodiscard]] const owner_detail::Sm87BulkV2P40RequestState* request_state()
      const noexcept {
    return request_state_.get();
  }
  [[nodiscard]] const owner_detail::Sm87BulkV2P40RequestStateSealedAccess*
  request_access() const noexcept;
  [[nodiscard]] const owner_detail::Sm87BulkV2P40ExecutionAccess*
  execution_access() const noexcept;
  [[nodiscard]] const ProjectionAccess* projection_access() const noexcept {
    return projection_access_ ? &*projection_access_ : nullptr;
  }
  [[nodiscard]] const WholeProjectionAccess* whole_projection_access()
      const noexcept;
  [[nodiscard]] const Bf16AbAccess* bf16_ab_access() const noexcept {
    return bf16_ab_access_ ? &*bf16_ab_access_ : nullptr;
  }
  [[nodiscard]] const AttentionAccess* attention_access() const noexcept {
    return attention_access_.get();
  }
  [[nodiscard]] q3x::kernels::Sm87BulkV2GdnP40Session* gdn_session()
      noexcept {
    return &gdn_session_;
  }
  [[nodiscard]] const q3x::kernels::Sm87BulkV2GdnP40Session* gdn_session()
      const noexcept {
    return &gdn_session_;
  }
  [[nodiscard]] q3x::kernels::Sm87BulkV2GdnP40SubmissionReceipt*
  gdn_receipt() noexcept {
    return &gdn_receipt_;
  }
  [[nodiscard]] const q3x::kernels::Sm87BulkV2GdnP40SubmissionReceipt*
  gdn_receipt() const noexcept {
    return &gdn_receipt_;
  }
  // The executor cannot mutate a completed GDN generation directly.  This
  // root-only transition asks the physical Owner to prove its exact terminal
  // Main synchronization and this root's stream/event/cancellation binding
  // before rearming the retained session without another CUDA query.
  [[nodiscard]] owner_detail::Sm87BulkV2P40OwnerStatus
  hot_rearm_gdn_session_after_completed_request() noexcept;
  [[nodiscard]] const owner_detail::Sm87BulkV2P40OwnerIdentity& identity()
      const noexcept {
    return identity_;
  }
  [[nodiscard]] const Sm87BulkV2P40CompositionAudit& audit() const noexcept {
    return audit_;
  }

 private:
  Sm87BulkV2P40CompositionRoot(
      const ModelWeights& model_weights,
      std::unique_ptr<owner_detail::Sm87BulkV2P40Owner> owner,
      std::unique_ptr<owner_detail::Sm87BulkV2P40RequestState> request_state,
      std::unique_ptr<
          owner_detail::Sm87BulkV2P40WholeProjectionStartupRoot>
          whole_projection_root,
      ProjectionAccess projection_access) noexcept;

  [[nodiscard]] Sm87BulkV2P40CompositionStatus initialize() noexcept;
  [[nodiscard]] Sm87BulkV2P40CompositionStatus derive_identity() noexcept;
  [[nodiscard]] Sm87BulkV2P40CompositionStatus seal_bf16_ab() noexcept;
  [[nodiscard]] Sm87BulkV2P40CompositionStatus seal_attention() noexcept;
  [[nodiscard]] Sm87BulkV2P40CompositionStatus seal_gdn_session() noexcept;
  [[nodiscard]] Sm87BulkV2P40CompositionStatus
  mint_and_install_direction_witness() noexcept;

  // Declaration order is lifetime order: reverse destruction retires every
  // borrower/session before RequestState and finally the physical Owner.
  const ModelWeights* model_weights_ = nullptr;
  std::unique_ptr<owner_detail::Sm87BulkV2P40Owner> owner_;
  std::unique_ptr<owner_detail::Sm87BulkV2P40RequestState> request_state_;
  std::unique_ptr<
      owner_detail::Sm87BulkV2P40WholeProjectionStartupRoot>
      whole_projection_root_;
  std::optional<ProjectionAccess> projection_access_;
  std::optional<Bf16AbAccess> bf16_ab_access_;
  std::unique_ptr<AttentionAccess> attention_access_;
  q3x::kernels::Sm87BulkV2GdnP40SubmissionReceipt gdn_receipt_{};
  q3x::kernels::Sm87BulkV2GdnP40Session gdn_session_{};
  std::unique_ptr<owner_detail::Sm87BulkV2P40ConstituentSealAccess>
      constituent_seal_;
  owner_detail::Sm87BulkV2P40OwnerIdentity identity_{};
  Sm87BulkV2P40CompositionAudit audit_{};

  friend Sm87BulkV2P40CompositionCreateResult
  create_sm87_bulk_dataflow_v2_p40_composition_root(
      const ModelWeights&) noexcept;
};

// The only factory.  It accepts the authenticated model object and nothing
// else: no caller IDs, streams, events, allocations, receipts, manifests,
// capability pointers, numerical claims or production selectors.
[[nodiscard]] Sm87BulkV2P40CompositionCreateResult
create_sm87_bulk_dataflow_v2_p40_composition_root(
    const ModelWeights& model_weights) noexcept;

[[nodiscard]] const char* to_string(
    Sm87BulkV2P40CompositionError error) noexcept;

}  // namespace q3x::runtime::sm87_bulk_v2_p40_composition_detail
