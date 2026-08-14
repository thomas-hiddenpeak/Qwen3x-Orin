#include "sm87_bulk_dataflow_v2_p40_constituent_seal_internal.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail {
namespace {

[[nodiscard]] constexpr std::size_t constituent_index(
    const Sm87BulkV2P40Constituent constituent) noexcept {
  return static_cast<std::size_t>(constituent);
}

[[nodiscard]] bool all_ready(
    const std::array<Sm87BulkV2P40ConstituentFinding,
                     kSm87BulkV2P40ConstituentCount>& findings) noexcept {
  return std::all_of(
      findings.begin(), findings.end(), [](const auto finding) noexcept {
        return finding == Sm87BulkV2P40ConstituentFinding::kReady;
      });
}

[[nodiscard]] Sm87BulkV2P40ConstituentFinding audit_fp8(
    const std::int32_t owner_device, void* const expected_stream,
    const Sm87BulkV2P40RealConstituentSealRequest& request) noexcept {
  if (request.fp8 == nullptr) {
    return Sm87BulkV2P40ConstituentFinding::kMissingSealedAccess;
  }
  if (!request.fp8->valid()) {
    return Sm87BulkV2P40ConstituentFinding::kInvalidSealedAccess;
  }
  const auto& receipt = request.fp8->receipt();
  if (receipt.deployment_identity != request.identities.deployment_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignDeployment;
  }
  if (receipt.device_ordinal != owner_device) {
    return Sm87BulkV2P40ConstituentFinding::kForeignDevice;
  }
  if (receipt.cuda_stream_owner_identity !=
      request.identities.stream_event_owner_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignStreamOwner;
  }
  if (receipt.cuda_stream != expected_stream) {
    return Sm87BulkV2P40ConstituentFinding::kWrongOwnedStream;
  }
  if (!receipt.numerical_contract_qualified) {
    return Sm87BulkV2P40ConstituentFinding::kNumericalContractUnqualified;
  }
  if (receipt.production_dispatch_eligible) {
    return Sm87BulkV2P40ConstituentFinding::
        kProductionDispatchMustRemainFalse;
  }
  // The current FP8 receipt carries no identity binding the numerical oracle
  // to this exact binary/access.  A true flag alone is not sufficient.
  return Sm87BulkV2P40ConstituentFinding::
      kMissingAuthenticatedNumericalEvidenceBinding;
}

[[nodiscard]] Sm87BulkV2P40ConstituentFinding audit_attention(
    const std::int32_t owner_device, void* const expected_stream,
    const Sm87BulkV2P40RealConstituentSealRequest& request) noexcept {
  if (request.attention == nullptr) {
    return Sm87BulkV2P40ConstituentFinding::kMissingSealedAccess;
  }
  if (!request.attention->valid()) {
    return Sm87BulkV2P40ConstituentFinding::kInvalidSealedAccess;
  }
  const auto& receipt = request.attention->receipt();
  if (receipt.deployment_identity != request.identities.deployment_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignDeployment;
  }
  if (receipt.device_ordinal != owner_device) {
    return Sm87BulkV2P40ConstituentFinding::kForeignDevice;
  }
  if (receipt.cuda_stream_owner_identity !=
      request.identities.stream_event_owner_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignStreamOwner;
  }
  if (receipt.cuda_stream != expected_stream) {
    return Sm87BulkV2P40ConstituentFinding::kWrongOwnedStream;
  }
  if (!receipt.numerical_contract_qualified) {
    return Sm87BulkV2P40ConstituentFinding::kNumericalContractUnqualified;
  }
  if (receipt.production_dispatch_eligible) {
    return Sm87BulkV2P40ConstituentFinding::
        kProductionDispatchMustRemainFalse;
  }
  return Sm87BulkV2P40ConstituentFinding::
      kMissingAuthenticatedNumericalEvidenceBinding;
}

[[nodiscard]] Sm87BulkV2P40ConstituentFinding audit_request_state(
    const std::uint64_t owner_identity, const std::int32_t owner_device,
    void* const owner_main_stream,
    const Sm87BulkV2P40RealConstituentSealRequest& request) noexcept {
  if (request.request_state == nullptr) {
    return Sm87BulkV2P40ConstituentFinding::
        kMissingUnforgeableQualificationInterface;
  }
  if (!request.request_state->default_off_development_resource_valid()) {
    return Sm87BulkV2P40ConstituentFinding::kInvalidSealedAccess;
  }
  const auto& identity = request.request_state->identity();
  if (identity.owner_identity != owner_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignOwner;
  }
  if (identity.device_ordinal != owner_device) {
    return Sm87BulkV2P40ConstituentFinding::kForeignDevice;
  }
  if (identity.stream_event_owner_identity !=
      request.identities.stream_event_owner_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignStreamOwner;
  }
  if (request.request_state->cuda_stream(Sm87BulkV2P40Stream::kMain) !=
      owner_main_stream) {
    return Sm87BulkV2P40ConstituentFinding::kWrongOwnedStream;
  }
  if (identity.allocation_identity !=
      request.identities.request_allocation_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignAllocation;
  }
  return Sm87BulkV2P40ConstituentFinding::kReady;
}

}  // namespace

bool Sm87BulkV2P40DevelopmentAdmissionEvidence::semantics_valid()
    const noexcept {
  return identity.valid() && identity.development_candidate_valid() &&
         all_ready(constituent_findings) &&
         all_static_resource_checks_complete &&
         authenticated_real_constituents &&
         exact_numerical_contract_qualified &&
         default_off_candidate_eligible &&
         !production_dispatch_eligible && !production_selector_bound &&
         !synthetic_host_contract && !mtp_used &&
         !cublaslt_production_path_used && !request_time_jit_used &&
         !request_time_repack_used && !accuracy_relaxation_used;
}

bool Sm87BulkV2P40RealConstituentIdentityClaims::complete() const noexcept {
  return deployment_identity != 0U && model_identity != 0U &&
         request_allocation_identity != 0U &&
         stream_event_owner_identity != 0U &&
         asset_catalog_identity != 0U && binary_evidence_identity != 0U &&
         fp8_oracle_evidence_identity != 0U &&
         attention_oracle_evidence_identity != 0U &&
         gdn_oracle_evidence_identity != 0U &&
         nvfp4_oracle_evidence_identity != 0U;
}

Sm87BulkV2P40RealConstituentSealResult
seal_sm87_bulk_dataflow_v2_p40_real_constituents(
    Sm87BulkV2P40Owner& owner,
    const Sm87BulkV2P40RealConstituentSealRequest& request) noexcept {
  Sm87BulkV2P40RealConstituentSealResult result;
  result.audit.constituent_findings.fill(
      Sm87BulkV2P40ConstituentFinding::kMissingSealedAccess);

  if (owner.state_ != Sm87BulkV2P40OwnerState::kResourcesReady ||
      owner.execution_access_ != nullptr) {
    result.status = {Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                     "real_composite_requires_resources_ready", 0, 0U};
    return result;
  }
  if (!request.identities.complete()) {
    result.status = {Sm87BulkV2P40OwnerError::kMissingConstituentSeal,
                     "real_composite_identity_claims_incomplete", 0, 0U};
    return result;
  }

  auto& identity = result.audit.identity;
  identity.plan_magic = kSm87BulkV2P40PlanMagic;
  identity.abi_major = kSm87BulkV2P40PlanAbiMajor;
  identity.abi_minor = kSm87BulkV2P40PlanAbiMinor;
  identity.owner_identity = owner.owner_identity_;
  // One owner can be sealed only once.  Its owner-issued nonce is therefore
  // also a unique seal nonce without accepting caller-selected authority.
  identity.seal_nonce = owner.owner_identity_;
  identity.deployment_identity = request.identities.deployment_identity;
  identity.model_identity = request.identities.model_identity;
  identity.request_allocation_identity =
      request.identities.request_allocation_identity;
  identity.stream_event_owner_identity =
      request.identities.stream_event_owner_identity;
  identity.asset_catalog_identity =
      request.identities.asset_catalog_identity;
  identity.binary_evidence_identity =
      request.identities.binary_evidence_identity;
  identity.fp8_oracle_evidence_identity =
      request.identities.fp8_oracle_evidence_identity;
  identity.attention_oracle_evidence_identity =
      request.identities.attention_oracle_evidence_identity;
  identity.gdn_oracle_evidence_identity =
      request.identities.gdn_oracle_evidence_identity;
  identity.nvfp4_oracle_evidence_identity =
      request.identities.nvfp4_oracle_evidence_identity;
  identity.device_ordinal = owner.device_ordinal_;
  identity.execution_class =
      Sm87BulkV2P40ExecutionClass::kDefaultOffDevelopmentCandidate;
  identity.production_dispatch_eligible = false;

  auto& findings = result.audit.constituent_findings;
  findings[constituent_index(Sm87BulkV2P40Constituent::kFp8Projection)] =
      audit_fp8(
          owner.device_ordinal_,
          owner.streams_[static_cast<std::size_t>(
              Sm87BulkV2P40Stream::kProjectionAndGdnProducer)],
          request);
  findings[constituent_index(Sm87BulkV2P40Constituent::kFullAttention)] =
      audit_attention(
          owner.device_ordinal_,
          owner.streams_[static_cast<std::size_t>(Sm87BulkV2P40Stream::kMain)],
          request);
  findings[constituent_index(Sm87BulkV2P40Constituent::kBf16Ab)] =
      request.bf16_ab == nullptr
          ? Sm87BulkV2P40ConstituentFinding::kMissingSealedAccess
          : Sm87BulkV2P40ConstituentFinding::
                kMissingUnforgeableQualificationInterface;
  if (request.gdn == nullptr) {
    findings[constituent_index(Sm87BulkV2P40Constituent::kGdn)] =
        Sm87BulkV2P40ConstituentFinding::kMissingSealedAccess;
  } else if (!q3x::kernels::sm87_bulk_v2_gdn_p40_session_state_valid(
                 *request.gdn)) {
    findings[constituent_index(Sm87BulkV2P40Constituent::kGdn)] =
        Sm87BulkV2P40ConstituentFinding::kInvalidSealedAccess;
  } else {
    findings[constituent_index(Sm87BulkV2P40Constituent::kGdn)] =
        Sm87BulkV2P40ConstituentFinding::kMutableSessionIsNotCapability;
  }
  // Confirmed interface gap: NVFP4 currently exposes a manifest, resource
  // query and admission-only numerical launcher, but no startup-sealed access.
  findings[constituent_index(Sm87BulkV2P40Constituent::kNvFp4Projection)] =
      Sm87BulkV2P40ConstituentFinding::
          kMissingUnforgeableQualificationInterface;
  const auto request_state_finding = audit_request_state(
      owner.owner_identity_, owner.device_ordinal_,
      owner.streams_[static_cast<std::size_t>(
          Sm87BulkV2P40Stream::kMain)],
      request);
  findings[constituent_index(Sm87BulkV2P40Constituent::kRequestArena)] =
      request_state_finding;
  findings[constituent_index(Sm87BulkV2P40Constituent::kPinnedHandoff)] =
      request_state_finding;

  const bool constituents_ready = all_ready(findings);
  result.audit.all_static_resource_checks_complete = constituents_ready;
  result.audit.authenticated_real_constituents = constituents_ready;
  result.audit.exact_numerical_contract_qualified = constituents_ready;
  result.audit.default_off_candidate_eligible = constituents_ready;
  result.audit.production_dispatch_eligible = false;
  result.audit.production_selector_bound = false;
  result.audit.synthetic_host_contract = false;
  result.audit.mtp_used = false;
  result.audit.cublaslt_production_path_used = false;
  result.audit.request_time_jit_used = false;
  result.audit.request_time_repack_used = false;
  result.audit.accuracy_relaxation_used = false;
  identity.authenticated_real_constituents = constituents_ready;
  identity.exact_numerical_contract_qualified = constituents_ready;
  identity.development_execution_eligible = constituents_ready;

  // No current combination can reach this branch until every exact interface
  // above exists and binds its own numerical evidence.  Keeping the branch
  // fail-closed is intentional; manifests and public aggregates are not
  // accepted as shortcuts.
  if (!result.audit.semantics_valid()) {
    const auto blocker = std::find_if(
        findings.begin(), findings.end(), [](const auto finding) noexcept {
          return finding != Sm87BulkV2P40ConstituentFinding::kReady;
        });
    const std::size_t blocker_index =
        blocker == findings.end()
            ? kSm87BulkV2P40ConstituentCount
            : static_cast<std::size_t>(blocker - findings.begin());
    result.status = {Sm87BulkV2P40OwnerError::kMissingConstituentSeal,
                     "real_constituent_capability_or_qualification_missing",
                     0, blocker_index};
    return result;
  }

  result.status = {Sm87BulkV2P40OwnerError::kMissingConstituentSeal,
                   "real_constituent_adapter_not_yet_complete", 0, 0U};
  return result;
}

}  // namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail
