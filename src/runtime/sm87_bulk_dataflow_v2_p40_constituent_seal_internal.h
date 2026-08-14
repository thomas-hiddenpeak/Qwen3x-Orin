#pragma once

#include "sm87_bulk_dataflow_v2_p40_owner_internal.h"
#include "sm87_bulk_dataflow_v2_p40_request_state_internal.h"

#include "q3x/kernels/sm87_bulk_dataflow_v2_gdn_p40_plan.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_whole_p40.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_down_whole_p40.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40.h"
#include "sm87_bulk_dataflow_v2_attention_l2_cohort_launch_internal.h"
#include "sm87_bulk_dataflow_v2_fp8_projection_launch_internal.h"
#include "sm87_target_aot_bf16_ab_launch_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail {

// The public whole-kernel resource structs are deliberately caller-fillable
// observations.  This separate startup root is the only object that may turn
// observations made by the linked CUDA query functions into an owner-bound
// source/resource-envelope capability.  In the real path, CMake has already
// authenticated the six fixed source files before compiling this issuer; the
// runtime queries then bind the actual linked symbols' resource envelopes.
// The retained cubin/SASS records below are catalog metadata only: this code
// does not hash or authenticate the currently loaded ELF, cubin, or SASS.
// Numerical, performance, release and production qualification are all
// outside this boundary.
enum class Sm87BulkV2P40WholeProjectionStartupError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kInvalidOwner,
  kMissingWholeSuccessor,
  kFp8Query,
  kGateUpQuery,
  kDownQuery,
  kWrongDevice,
  kResourceMismatch,
  kRetainedEvidenceCatalogMismatch,
  kCallerFilledObservationIsNotAuthority,
  kAllocation,
};

enum class Sm87BulkV2P40WholeProjectionStartupExecutionClass
    : std::uint8_t {
  kInvalid = 0U,
  kDefaultOffFixedAot,
  kSyntheticHostQuery,
};

// Retained offline evidence identities used only to detect catalog drift in
// this source-private issuer.  They do not describe or authenticate the
// currently loaded executable image.
struct Sm87BulkV2P40WholeProjectionRetainedEvidenceCatalog final {
  std::array<std::uint8_t, 8U> magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  std::array<std::uint8_t, 32U> fp8_retained_sass_record_sha256{};
  std::array<std::uint8_t, 32U> gate_up_retained_cubin_record_sha256{};
  std::array<std::uint8_t, 32U> gate_up_retained_sass_record_sha256{};
  std::array<std::uint8_t, 32U> down_retained_sass_record_sha256{};

  [[nodiscard]] bool valid() const noexcept;
};

struct Sm87BulkV2P40WholeProjectionStartupStatus final {
  Sm87BulkV2P40WholeProjectionStartupError error =
      Sm87BulkV2P40WholeProjectionStartupError::kNone;
  const char* context = "none";
  int cuda_error = 0;
  std::size_t resource_index = std::numeric_limits<std::size_t>::max();

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == Sm87BulkV2P40WholeProjectionStartupError::kNone;
  }
};

// Capability-free startup report.  Its booleans can be copied or forged and
// therefore have no enqueue or owner-seal authority.  A successful report is
// useful only together with the private root-owned Access object below.
struct Sm87BulkV2P40WholeProjectionStartupAudit final {
  Sm87BulkV2P40WholeProjectionRetainedEvidenceCatalog retained_catalog{};
  std::uint64_t owner_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::size_t cuda_resource_queries = 0U;
  bool fp8_whole_successor_linked = false;
  bool gate_up_whole_successor_linked = false;
  bool down_whole_successor_linked = false;
  bool all_required_dynamic_shared_attributes_configured = false;
  bool all_linked_symbol_resource_envelopes_exact = false;
  bool retained_evidence_catalog_consistent = false;
  bool configured_source_sha256_gate_passed = false;
  bool synthetic_host_query = false;
  bool private_startup_query_used = false;
  bool caller_filled_public_observation_used_as_authority = true;
  bool default_off = false;
  bool numerical_contract_qualified = true;
  bool performance_qualified = true;
  bool release_qualified = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] bool resource_qualification_valid() const noexcept;
};

class Sm87BulkV2P40WholeProjectionStartupRoot;
class Sm87BulkV2P40WholeProjectionStartupIssuer;
struct Sm87BulkV2P40WholeProjectionStartupResult;

// Immutable in-process capability.  It exposes neither the public resource
// records nor a constructor.  Object identity, root identity and the exact
// physical v2 owner must all match at the composition boundary.
class Sm87BulkV2P40WholeProjectionStartupAccess final {
 public:
  Sm87BulkV2P40WholeProjectionStartupAccess(
      const Sm87BulkV2P40WholeProjectionStartupAccess&) = delete;
  Sm87BulkV2P40WholeProjectionStartupAccess& operator=(
      const Sm87BulkV2P40WholeProjectionStartupAccess&) = delete;
  Sm87BulkV2P40WholeProjectionStartupAccess(
      Sm87BulkV2P40WholeProjectionStartupAccess&&) = delete;
  Sm87BulkV2P40WholeProjectionStartupAccess& operator=(
      Sm87BulkV2P40WholeProjectionStartupAccess&&) = delete;

  // Validates only the CMake source-hash-gated, linked-symbol resource
  // envelope.  It deliberately carries no loaded-binary/SASS, numerical,
  // performance, release, or production qualification.
  [[nodiscard]] bool default_off_fixed_aot_resource_valid() const noexcept;
  [[nodiscard]] bool bound_to(
      const Sm87BulkV2P40Owner& owner) const noexcept;
  [[nodiscard]] const Sm87BulkV2P40WholeProjectionRetainedEvidenceCatalog&
  retained_evidence_catalog() const noexcept {
    return retained_catalog_;
  }
  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }
  [[nodiscard]] bool numerical_contract_qualified() const noexcept {
    return false;
  }
  [[nodiscard]] bool performance_qualified() const noexcept { return false; }
  [[nodiscard]] bool production_dispatch_eligible() const noexcept {
    return false;
  }

 private:
  Sm87BulkV2P40WholeProjectionStartupAccess(
      const Sm87BulkV2P40WholeProjectionStartupRoot* issuer,
      const Sm87BulkV2P40Owner* owner,
      const Sm87BulkV2P40WholeProjectionRetainedEvidenceCatalog&
          retained_catalog,
      Sm87BulkV2P40WholeProjectionStartupExecutionClass execution_class,
      std::uint64_t owner_identity, std::int32_t device_ordinal) noexcept;

  const Sm87BulkV2P40WholeProjectionStartupRoot* issuer_ = nullptr;
  const Sm87BulkV2P40Owner* owner_ = nullptr;
  Sm87BulkV2P40WholeProjectionRetainedEvidenceCatalog retained_catalog_{};
  Sm87BulkV2P40WholeProjectionStartupExecutionClass execution_class_ =
      Sm87BulkV2P40WholeProjectionStartupExecutionClass::kInvalid;
  std::uint64_t owner_identity_ = 0U;
  std::int32_t device_ordinal_ = -1;
  bool whole_fp8_bound_ = false;
  bool whole_gate_up_bound_ = false;
  bool whole_down_bound_ = false;
  bool required_dynamic_shared_attributes_configured_ = false;
  bool linked_symbol_resource_envelopes_exact_ = false;
  bool retained_evidence_catalog_consistent_ = false;
  bool configured_source_sha256_gate_passed_ = false;
  bool default_off_ = false;

  friend class Sm87BulkV2P40WholeProjectionStartupRoot;
#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_STARTUP_SEAL_HOST_FIXTURE)
  friend class Sm87BulkV2P40WholeProjectionStartupHostFixture;
#endif
};

// Lifetime owner and sole issuer.  Destroying this root destroys the access;
// an executor may borrow the pointer only while the startup root remains
// alive.  No public resource record can instantiate either class.
class Sm87BulkV2P40WholeProjectionStartupRoot final {
 public:
  Sm87BulkV2P40WholeProjectionStartupRoot(
      const Sm87BulkV2P40WholeProjectionStartupRoot&) = delete;
  Sm87BulkV2P40WholeProjectionStartupRoot& operator=(
      const Sm87BulkV2P40WholeProjectionStartupRoot&) = delete;
  Sm87BulkV2P40WholeProjectionStartupRoot(
      Sm87BulkV2P40WholeProjectionStartupRoot&&) = delete;
  Sm87BulkV2P40WholeProjectionStartupRoot& operator=(
      Sm87BulkV2P40WholeProjectionStartupRoot&&) = delete;

  [[nodiscard]] const Sm87BulkV2P40WholeProjectionStartupAccess* access()
      const noexcept {
    return access_.get();
  }

 private:
  explicit Sm87BulkV2P40WholeProjectionStartupRoot(
      const Sm87BulkV2P40Owner* owner) noexcept;
  [[nodiscard]] bool install_access(
      const Sm87BulkV2P40WholeProjectionRetainedEvidenceCatalog&
          retained_catalog,
      Sm87BulkV2P40WholeProjectionStartupExecutionClass execution_class,
      std::uint64_t owner_identity, std::int32_t device_ordinal) noexcept;

  const Sm87BulkV2P40Owner* owner_ = nullptr;
  std::unique_ptr<Sm87BulkV2P40WholeProjectionStartupAccess> access_;

  friend struct Sm87BulkV2P40WholeProjectionStartupResult;
  friend class Sm87BulkV2P40WholeProjectionStartupIssuer;
  friend Sm87BulkV2P40WholeProjectionStartupResult
  create_sm87_bulk_dataflow_v2_p40_whole_projection_startup_root(
      Sm87BulkV2P40Owner&) noexcept;
#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_STARTUP_SEAL_HOST_FIXTURE)
  friend class Sm87BulkV2P40WholeProjectionStartupHostFixture;
#endif
};

struct Sm87BulkV2P40WholeProjectionStartupResult final {
  std::unique_ptr<Sm87BulkV2P40WholeProjectionStartupRoot> root;
  Sm87BulkV2P40WholeProjectionStartupAudit audit{};
  Sm87BulkV2P40WholeProjectionStartupStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return root != nullptr && root->access() != nullptr &&
           root->access()->default_off_fixed_aot_resource_valid() &&
           audit.resource_qualification_valid() &&
           static_cast<bool>(status);
  }
};

// Production-shaped, default-off startup factory.  The call accepts only the
// physical v2 owner.  The official CMake path source-hash-gates the six fixed
// inputs, and this factory queries the exact linked AOT symbols' resource
// envelopes.  Retained code-evidence values are internal catalog records, not
// authentication of the loaded ELF/cubin/SASS.  When any whole successor is
// absent at build time it returns kMissingWholeSuccessor without a CUDA call.
[[nodiscard]] Sm87BulkV2P40WholeProjectionStartupResult
create_sm87_bulk_dataflow_v2_p40_whole_projection_startup_root(
    Sm87BulkV2P40Owner& owner) noexcept;

struct Sm87BulkV2P40WholeProjectionStartupObservations final {
  q3x::kernels::Sm87BulkV2Fp8WholeP40FamilyResources fp8{};
  q3x::kernels::Sm87BulkV2NvFp4GateUpWholeP40Resources gate_up{};
  q3x::kernels::Sm87BulkV2NvFp4DownWholeP40Resources down{};
  int fp8_query_error = 0;
  int gate_up_query_error = 0;
  int down_query_error = 0;
  bool fp8_successor_linked = false;
  bool gate_up_successor_linked = false;
  bool down_successor_linked = false;
};

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_STARTUP_SEAL_HOST_FIXTURE)
// Test-only query provenance seam.  The caller-filled path always rejects;
// the synthetic query path can create only kSyntheticHostQuery access, which
// is never valid at the real default-off factory boundary.
class Sm87BulkV2P40WholeProjectionStartupHostFixture final {
 public:
  [[nodiscard]] static Sm87BulkV2P40WholeProjectionStartupObservations
  passing_observations(std::int32_t device_ordinal = 0) noexcept;
  [[nodiscard]] static Sm87BulkV2P40WholeProjectionStartupResult
  attempt_from_caller_filled_observations(
      Sm87BulkV2P40Owner& owner,
      const Sm87BulkV2P40WholeProjectionStartupObservations& observations)
      noexcept;
  [[nodiscard]] static Sm87BulkV2P40WholeProjectionStartupResult
  mint_from_synthetic_startup_query(
      Sm87BulkV2P40Owner& owner,
      const Sm87BulkV2P40WholeProjectionStartupObservations& observations)
      noexcept;
  [[nodiscard]] static bool synthetic_access_valid(
      const Sm87BulkV2P40WholeProjectionStartupResult& result,
      const Sm87BulkV2P40Owner& owner) noexcept;
};
#endif

enum class Sm87BulkV2P40Constituent : std::uint8_t {
  kFp8Projection = 0U,
  kFullAttention,
  kBf16Ab,
  kGdn,
  kNvFp4Projection,
  kRequestArena,
  kPinnedHandoff,
  kCount,
};

inline constexpr std::size_t kSm87BulkV2P40ConstituentCount =
    static_cast<std::size_t>(Sm87BulkV2P40Constituent::kCount);

enum class Sm87BulkV2P40ConstituentFinding : std::uint8_t {
  kUninspected = 0U,
  kReady,
  kMissingSealedAccess,
  kInvalidSealedAccess,
  kForeignOwner,
  kForeignAllocation,
  kForeignDeployment,
  kForeignDevice,
  kForeignStreamOwner,
  kWrongOwnedStream,
  kNumericalContractUnqualified,
  kProductionDispatchMustRemainFalse,
  kMissingAuthenticatedNumericalEvidenceBinding,
  kMissingUnforgeableQualificationInterface,
  kMutableSessionIsNotCapability,
};

// This is a capability-free description of the development-admission rule.
// It exists so the rule can be host-tested without forging CUDA startup
// capabilities.  A caller-filled instance is evidence for diagnostics only;
// it can never construct Sm87BulkV2P40ConstituentSealAccess or execution
// authority.
struct Sm87BulkV2P40DevelopmentAdmissionEvidence final {
  Sm87BulkV2P40OwnerIdentity identity{};
  std::array<Sm87BulkV2P40ConstituentFinding,
             kSm87BulkV2P40ConstituentCount>
      constituent_findings{};
  bool all_static_resource_checks_complete = false;
  bool authenticated_real_constituents = false;
  bool exact_numerical_contract_qualified = false;
  bool default_off_candidate_eligible = false;
  bool production_dispatch_eligible = false;
  bool production_selector_bound = false;
  bool synthetic_host_contract = false;
  bool mtp_used = false;
  bool cublaslt_production_path_used = false;
  bool request_time_jit_used = false;
  bool request_time_repack_used = false;
  bool accuracy_relaxation_used = false;

  [[nodiscard]] bool semantics_valid() const noexcept;
};

// Identities supplied by the real startup composition root.  The adapter
// overwrites all owner-issued fields (magic, ABI, owner, nonce, device and
// execution class); a caller cannot choose those fields.
struct Sm87BulkV2P40RealConstituentIdentityClaims final {
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

  [[nodiscard]] bool complete() const noexcept;
};

struct Sm87BulkV2P40RealConstituentSealRequest final {
  Sm87BulkV2P40RealConstituentIdentityClaims identities{};
  const q3x::kernels::sm87_bulk_v2_fp8_execution_detail::
      Sm87BulkV2Fp8SealedAccess* fp8 = nullptr;
  const q3x::kernels::sm87_bulk_v2_attention_execution_detail::
      Sm87BulkV2AttentionSealedAccess* attention = nullptr;
  const q3x::kernels::sm87_target_aot_bf16_ab_execution_detail::
      SealedInterleavedP40Access* bf16_ab = nullptr;
  const q3x::kernels::Sm87BulkV2GdnP40Session* gdn = nullptr;
  const Sm87BulkV2P40RequestStateSealedAccess* request_state = nullptr;
};

struct Sm87BulkV2P40RealConstituentSealResult final {
  Sm87BulkV2P40DevelopmentAdmissionEvidence audit{};
  Sm87BulkV2P40OwnerStatus status{};
  const Sm87BulkV2P40ExecutionAccess* access = nullptr;

  [[nodiscard]] explicit operator bool() const noexcept {
    return audit.semantics_valid() && static_cast<bool>(status) &&
           access != nullptr;
  }
};

// Audits the currently available real startup capabilities and attempts to
// seal the default-off candidate.  As of this slice, BF16 A/B and GDN do not
// expose authenticated numerical evidence capabilities, and NVFP4 exposes
// no unforgeable startup access. Request arena and pinned handoff share one
// owner-bound immutable access, but the adapter still leaves the owner
// kResourcesReady.  It does not accept manifests, booleans, or the public
// aggregate as substitutes.
[[nodiscard]] Sm87BulkV2P40RealConstituentSealResult
seal_sm87_bulk_dataflow_v2_p40_real_constituents(
    Sm87BulkV2P40Owner& owner,
    const Sm87BulkV2P40RealConstituentSealRequest& request) noexcept;

}  // namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail
