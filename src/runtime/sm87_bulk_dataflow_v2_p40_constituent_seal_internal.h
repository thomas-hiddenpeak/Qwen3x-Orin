#pragma once

#include "sm87_bulk_dataflow_v2_p40_owner_internal.h"
#include "sm87_bulk_dataflow_v2_p40_request_state_internal.h"

#include "q3x/kernels/sm87_bulk_dataflow_v2_gdn_p40_plan.h"
#include "sm87_bulk_dataflow_v2_attention_l2_cohort_launch_internal.h"
#include "sm87_bulk_dataflow_v2_fp8_projection_launch_internal.h"
#include "sm87_target_aot_bf16_ab_launch_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail {

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
