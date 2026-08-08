#pragma once

#include "q3x/runtime/prefill_execution_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime {

// Typed host contract for the indivisible operator boundary of
// AC-PREFILL-LAYERMAJOR-8K-v1. It contains opaque DeploymentPlan identities,
// never device pointers, CUDA handles, function pointers, or a route selector.
enum class PrefillBindingRole : std::uint8_t {
  kNvfp4GateUp = 0,
  kNvfp4Down,
  kLinearFp8Qkv,
  kLinearFp8Z,
  kLinearFp8O,
  kFullFp8Q,
  kFullFp8K,
  kFullFp8V,
  kFullFp8O,
  kLinearBf16A,
  kLinearBf16B,
  kExactGdn,
  kExactCausalAttention,
  kResidual,
  kNormalization,
  kEmbedding,
  kFinalHandoff,
  kCount,
  kInvalid = 0xffU,
};

inline constexpr std::size_t kLayerMajorPrefillRequiredOperatorRoleCount =
    static_cast<std::size_t>(PrefillBindingRole::kCount);

static_assert(kLayerMajorPrefillRequiredOperatorRoleCount == 17U);

enum class PrefillNumericalMode : std::uint8_t {
  kUnbound = 0,
  kExact,
  kApproximate,
};

enum class PrefillOperatorProvider : std::uint8_t {
  kUnbound = 0,
  kNative,
  kCuBlasLtReference,
  kExternalRuntime,
};

enum class PrefillTacticMode : std::uint8_t {
  kUnbound = 0,
  kAheadOfTime,
  kJit,
  kRequestTimeSelected,
};

// A role can describe an unbound C8192 target or the current C512 inventory
// without claiming that either has executed. Only kAuthenticatedC8192 can
// satisfy the binding contract, and even that cannot connect a production
// route through this host-only type.
enum class PrefillBindingAttestation : std::uint8_t {
  kUnboundDesign = 0,
  kLegacyC512InventoryOnly,
  kAuthenticatedC8192,
};

enum class PrefillResourceBindingState : std::uint8_t {
  kNotRequired = 0,
  kUnbound,
  kBound,
};

// identity is an opaque nonzero index into a future authenticated
// DeploymentPlan. It is deliberately not an address or callable handle.
struct PrefillResourceBinding {
  PrefillResourceBindingState state =
      PrefillResourceBindingState::kUnbound;
  std::uint64_t identity = 0U;

  [[nodiscard]] constexpr bool bound() const noexcept {
    return state == PrefillResourceBindingState::kBound && identity != 0U;
  }
};

struct PrefillTacticBinding {
  // The authenticated tactic artifact owns the physical tile, pipeline,
  // residency, and launch geometry. Those properties deliberately remain
  // opaque here so an unbound design cannot freeze an unmeasured tactic.
  PrefillTacticMode mode = PrefillTacticMode::kUnbound;
  std::uint64_t identity = 0U;

  [[nodiscard]] constexpr bool aot_bound() const noexcept {
    return mode == PrefillTacticMode::kAheadOfTime && identity != 0U;
  }
};

struct PrefillOperatorRoleRequirements {
  bool weight = false;
  bool sidecar = false;
  bool workspace = false;
  bool launcher = true;
  bool completion_event = true;
};

struct PrefillOperatorBinding {
  PrefillBindingRole role = PrefillBindingRole::kInvalid;
  std::uint32_t maximum_logical_panel_m = 0U;
  PrefillNumericalMode numerical_mode = PrefillNumericalMode::kUnbound;
  PrefillOperatorProvider provider = PrefillOperatorProvider::kUnbound;
  PrefillTacticBinding tactic;
  PrefillResourceBinding weight;
  PrefillResourceBinding sidecar;
  PrefillResourceBinding workspace;
  PrefillResourceBinding launcher;
  PrefillResourceBinding completion_event;
  PrefillBindingAttestation attestation =
      PrefillBindingAttestation::kUnboundDesign;
  bool uses_mtp = false;
};

struct PrefillOperatorBindingSet {
  std::array<PrefillOperatorBinding,
             kLayerMajorPrefillRequiredOperatorRoleCount>
      bindings{};
  std::size_t binding_count = 0U;
};

enum class PrefillOperatorBindingIssue : std::uint64_t {
  kNone = 0U,
  kInvalidRole = 1ULL << 0U,
  kMissingRole = 1ULL << 1U,
  kDuplicateRole = 1ULL << 2U,
  kInsufficientLogicalPanelCapacity = 1ULL << 3U,
  kNumericalModeUnbound = 1ULL << 4U,
  kApproximateNumericsForbidden = 1ULL << 5U,
  kProviderUnbound = 1ULL << 6U,
  kNonNativeProvider = 1ULL << 7U,
  kCuBlasLtForbidden = 1ULL << 8U,
  kMtpForbidden = 1ULL << 9U,
  kTacticNotAot = 1ULL << 10U,
  kTacticIdentityUnbound = 1ULL << 11U,
  kGateDownTacticAlias = 1ULL << 12U,
  kWeightBindingIncomplete = 1ULL << 13U,
  kSidecarBindingIncomplete = 1ULL << 14U,
  kWorkspaceBindingIncomplete = 1ULL << 15U,
  kLauncherBindingIncomplete = 1ULL << 16U,
  kCompletionEventBindingIncomplete = 1ULL << 17U,
  kMalformedBindingIdentity = 1ULL << 18U,
  kC8192AttestationUnbound = 1ULL << 19U,
  kLegacyC512InventoryOnly = 1ULL << 20U,
  kMalformedEnumEncoding = 1ULL << 21U,
};

struct PrefillOperatorBindingValidation {
  std::uint64_t issue_mask = 0U;
  std::array<std::uint8_t,
             kLayerMajorPrefillRequiredOperatorRoleCount>
      role_counts{};

  bool role_coverage_complete = false;
  bool logical_capacity_complete = false;
  bool exact_numerics_complete = false;
  bool native_provider_complete = false;
  bool aot_tactics_complete = false;
  bool artifact_bindings_complete = false;
  bool workspace_bindings_complete = false;
  bool launcher_bindings_complete = false;
  bool completion_events_complete = false;
  bool forbidden_boundaries_satisfied = false;
  bool attestation_complete = false;
  bool operator_bindings_complete = false;

  [[nodiscard]] constexpr bool complete() const noexcept {
    return operator_bindings_complete;
  }

  // This slice validates a descriptor only. A later BoundPrefillExecutionPlan
  // must authenticate the installed binary, bind real handles, and connect
  // engine/runner state before any execution gate can exist.
  [[nodiscard]] constexpr bool executable() const noexcept { return false; }
};

[[nodiscard]] constexpr bool has_prefill_operator_binding_issue(
    const PrefillOperatorBindingValidation& validation,
    const PrefillOperatorBindingIssue issue) noexcept {
  return (validation.issue_mask & static_cast<std::uint64_t>(issue)) != 0U;
}

[[nodiscard]] PrefillOperatorRoleRequirements
prefill_operator_role_requirements(PrefillBindingRole role) noexcept;

// Complete role/logical-capacity target for C8192. Exact/native/AOT describe
// the required route, while zero tactic/resource identities and
// kUnboundDesign explicitly prove that no physical tactic, launcher, or event
// is bound yet.
[[nodiscard]] PrefillOperatorBindingSet
make_unbound_layer_major_c8192_operator_bindings() noexcept;

// Inventory view of the existing public C512 mechanisms. Every role is capped
// at M512 and every candidate launcher/event remains unbound; this function
// cannot promote the legacy route into AC-PREFILL-LAYERMAJOR-8K-v1.
[[nodiscard]] PrefillOperatorBindingSet
make_legacy_c512_prefill_launcher_inventory() noexcept;

[[nodiscard]] PrefillOperatorBindingValidation
validate_layer_major_c8192_operator_bindings(
    const PrefillOperatorBindingSet& binding_set) noexcept;

}  // namespace q3x::runtime
