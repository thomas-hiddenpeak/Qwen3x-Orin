#include "q3x/runtime/prefill_operator_bindings.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime {
namespace {

[[nodiscard]] constexpr std::uint64_t issue_bit(
    const PrefillOperatorBindingIssue issue) noexcept {
  return static_cast<std::uint64_t>(issue);
}

void add_issue(PrefillOperatorBindingValidation& validation,
               const PrefillOperatorBindingIssue issue) noexcept {
  validation.issue_mask |= issue_bit(issue);
}

[[nodiscard]] constexpr bool valid_role(
    const PrefillBindingRole role) noexcept {
  return static_cast<std::uint8_t>(role) <
         static_cast<std::uint8_t>(PrefillBindingRole::kCount);
}

[[nodiscard]] constexpr std::size_t role_index(
    const PrefillBindingRole role) noexcept {
  return static_cast<std::size_t>(role);
}

[[nodiscard]] constexpr PrefillResourceBinding unbound_resource(
    const bool required) noexcept {
  return {required ? PrefillResourceBindingState::kUnbound
                   : PrefillResourceBindingState::kNotRequired,
          0U};
}

[[nodiscard]] PrefillOperatorBindingSet make_inventory(
    const std::uint32_t maximum_logical_panel_m,
    const PrefillBindingAttestation attestation) noexcept {
  PrefillOperatorBindingSet result;
  result.binding_count = result.bindings.size();
  for (std::size_t index = 0U; index < result.bindings.size(); ++index) {
    const auto role = static_cast<PrefillBindingRole>(index);
    const PrefillOperatorRoleRequirements requirements =
        prefill_operator_role_requirements(role);
    PrefillOperatorBinding& binding = result.bindings[index];
    binding.role = role;
    binding.maximum_logical_panel_m = maximum_logical_panel_m;
    binding.numerical_mode = PrefillNumericalMode::kExact;
    binding.provider = PrefillOperatorProvider::kNative;
    // AOT is the required specialization boundary. identity=0 makes it an
    // explicit unbound requirement rather than an execution claim.
    binding.tactic = {PrefillTacticMode::kAheadOfTime, 0U};
    binding.weight = unbound_resource(requirements.weight);
    binding.sidecar = unbound_resource(requirements.sidecar);
    binding.workspace = unbound_resource(requirements.workspace);
    binding.launcher = unbound_resource(requirements.launcher);
    binding.completion_event =
        unbound_resource(requirements.completion_event);
    binding.attestation = attestation;
    binding.uses_mtp = false;
  }
  return result;
}

[[nodiscard]] constexpr bool valid_binding_encoding(
    const PrefillResourceBinding& binding) noexcept {
  switch (binding.state) {
    case PrefillResourceBindingState::kNotRequired:
    case PrefillResourceBindingState::kUnbound:
      return binding.identity == 0U;
    case PrefillResourceBindingState::kBound:
      return binding.identity != 0U;
  }
  return false;
}

[[nodiscard]] constexpr bool satisfies_resource_requirement(
    const PrefillResourceBinding& binding, const bool required) noexcept {
  if (!valid_binding_encoding(binding)) {
    return false;
  }
  if (required) {
    return binding.bound();
  }
  return binding.state == PrefillResourceBindingState::kNotRequired;
}

void validate_resource(
    PrefillOperatorBindingValidation& validation,
    const PrefillResourceBinding& binding, const bool required,
    const PrefillOperatorBindingIssue incomplete_issue,
    bool& family_complete) noexcept {
  if (!valid_binding_encoding(binding)) {
    add_issue(validation,
              PrefillOperatorBindingIssue::kMalformedBindingIdentity);
  }
  if (!satisfies_resource_requirement(binding, required)) {
    add_issue(validation, incomplete_issue);
    family_complete = false;
  }
}

}  // namespace

PrefillOperatorRoleRequirements prefill_operator_role_requirements(
    const PrefillBindingRole role) noexcept {
  switch (role) {
    case PrefillBindingRole::kNvfp4GateUp:
    case PrefillBindingRole::kNvfp4Down:
    case PrefillBindingRole::kLinearFp8Qkv:
    case PrefillBindingRole::kLinearFp8Z:
    case PrefillBindingRole::kLinearFp8O:
    case PrefillBindingRole::kFullFp8Q:
    case PrefillBindingRole::kFullFp8K:
    case PrefillBindingRole::kFullFp8V:
    case PrefillBindingRole::kFullFp8O:
      return {/*weight=*/true, /*sidecar=*/true, /*workspace=*/true,
              /*launcher=*/true, /*completion_event=*/true};
    case PrefillBindingRole::kLinearBf16A:
    case PrefillBindingRole::kLinearBf16B:
    case PrefillBindingRole::kExactGdn:
    case PrefillBindingRole::kNormalization:
    case PrefillBindingRole::kEmbedding:
      return {/*weight=*/true, /*sidecar=*/false, /*workspace=*/true,
              /*launcher=*/true, /*completion_event=*/true};
    case PrefillBindingRole::kExactCausalAttention:
    case PrefillBindingRole::kResidual:
      return {/*weight=*/false, /*sidecar=*/false, /*workspace=*/true,
              /*launcher=*/true, /*completion_event=*/true};
    case PrefillBindingRole::kFinalHandoff:
      return {/*weight=*/false, /*sidecar=*/false, /*workspace=*/false,
              /*launcher=*/true, /*completion_event=*/true};
    case PrefillBindingRole::kCount:
    case PrefillBindingRole::kInvalid:
      return {/*weight=*/false, /*sidecar=*/false, /*workspace=*/false,
              /*launcher=*/false, /*completion_event=*/false};
  }
  return {/*weight=*/false, /*sidecar=*/false, /*workspace=*/false,
          /*launcher=*/false, /*completion_event=*/false};
}

PrefillOperatorBindingSet
make_unbound_layer_major_c8192_operator_bindings() noexcept {
  return make_inventory(kLayerMajorPrefillOperatorPanelTokens,
                        PrefillBindingAttestation::kUnboundDesign);
}

PrefillOperatorBindingSet
make_legacy_c512_prefill_launcher_inventory() noexcept {
  return make_inventory(kLayerMajorPrefillLegacyPublicTileTokens,
                        PrefillBindingAttestation::
                            kLegacyC512InventoryOnly);
}

PrefillOperatorBindingValidation
validate_layer_major_c8192_operator_bindings(
    const PrefillOperatorBindingSet& binding_set) noexcept {
  PrefillOperatorBindingValidation validation;
  const std::size_t inspected_count =
      std::min(binding_set.binding_count, binding_set.bindings.size());
  if (binding_set.binding_count > binding_set.bindings.size()) {
    add_issue(validation, PrefillOperatorBindingIssue::kInvalidRole);
  }

  for (std::size_t index = 0U; index < inspected_count; ++index) {
    const PrefillBindingRole role = binding_set.bindings[index].role;
    if (!valid_role(role)) {
      add_issue(validation, PrefillOperatorBindingIssue::kInvalidRole);
      continue;
    }
    std::uint8_t& count = validation.role_counts[role_index(role)];
    if (count != 0U) {
      add_issue(validation, PrefillOperatorBindingIssue::kDuplicateRole);
    }
    if (count != 0xffU) {
      ++count;
    }
  }

  bool exact_role_counts =
      binding_set.binding_count == validation.role_counts.size();
  for (const std::uint8_t count : validation.role_counts) {
    if (count == 0U) {
      add_issue(validation, PrefillOperatorBindingIssue::kMissingRole);
      exact_role_counts = false;
    } else if (count != 1U) {
      exact_role_counts = false;
    }
  }
  validation.role_coverage_complete = exact_role_counts;

  bool capacities_valid = true;
  bool exact_numerics = true;
  bool native_provider = true;
  bool aot_tactics = true;
  bool artifacts = true;
  bool workspaces = true;
  bool launchers = true;
  bool completion_events = true;
  bool forbidden_boundaries = true;
  bool attestations = true;
  const PrefillOperatorBinding* gate_up = nullptr;
  const PrefillOperatorBinding* down = nullptr;

  for (std::size_t index = 0U; index < inspected_count; ++index) {
    const PrefillOperatorBinding& binding = binding_set.bindings[index];
    if (!valid_role(binding.role)) {
      capacities_valid = false;
      exact_numerics = false;
      native_provider = false;
      aot_tactics = false;
      artifacts = false;
      workspaces = false;
      launchers = false;
      completion_events = false;
      forbidden_boundaries = false;
      attestations = false;
      continue;
    }
    if (binding.role == PrefillBindingRole::kNvfp4GateUp) {
      gate_up = &binding;
    } else if (binding.role == PrefillBindingRole::kNvfp4Down) {
      down = &binding;
    }

    if (binding.maximum_logical_panel_m <
        kLayerMajorPrefillOperatorPanelTokens) {
      add_issue(validation, PrefillOperatorBindingIssue::
                                kInsufficientLogicalPanelCapacity);
      capacities_valid = false;
    }
    switch (binding.numerical_mode) {
      case PrefillNumericalMode::kExact:
        break;
      case PrefillNumericalMode::kUnbound:
        add_issue(validation,
                  PrefillOperatorBindingIssue::kNumericalModeUnbound);
        exact_numerics = false;
        break;
      case PrefillNumericalMode::kApproximate:
        add_issue(validation, PrefillOperatorBindingIssue::
                                  kApproximateNumericsForbidden);
        exact_numerics = false;
        forbidden_boundaries = false;
        break;
      default:
        add_issue(validation,
                  PrefillOperatorBindingIssue::kMalformedEnumEncoding);
        exact_numerics = false;
        forbidden_boundaries = false;
        break;
    }

    switch (binding.provider) {
      case PrefillOperatorProvider::kNative:
        break;
      case PrefillOperatorProvider::kUnbound:
        add_issue(validation,
                  PrefillOperatorBindingIssue::kProviderUnbound);
        native_provider = false;
        break;
      case PrefillOperatorProvider::kCuBlasLtReference:
        add_issue(validation,
                  PrefillOperatorBindingIssue::kCuBlasLtForbidden);
        add_issue(validation,
                  PrefillOperatorBindingIssue::kNonNativeProvider);
        native_provider = false;
        forbidden_boundaries = false;
        break;
      case PrefillOperatorProvider::kExternalRuntime:
        add_issue(validation,
                  PrefillOperatorBindingIssue::kNonNativeProvider);
        native_provider = false;
        forbidden_boundaries = false;
        break;
      default:
        add_issue(validation,
                  PrefillOperatorBindingIssue::kMalformedEnumEncoding);
        native_provider = false;
        forbidden_boundaries = false;
        break;
    }

    if (binding.uses_mtp) {
      add_issue(validation, PrefillOperatorBindingIssue::kMtpForbidden);
      forbidden_boundaries = false;
    }

    if (binding.tactic.mode != PrefillTacticMode::kAheadOfTime) {
      add_issue(validation, PrefillOperatorBindingIssue::kTacticNotAot);
      aot_tactics = false;
    }
    if (binding.tactic.identity == 0U) {
      add_issue(validation,
                PrefillOperatorBindingIssue::kTacticIdentityUnbound);
      aot_tactics = false;
    }

    const PrefillOperatorRoleRequirements requirements =
        prefill_operator_role_requirements(binding.role);
    validate_resource(validation, binding.weight, requirements.weight,
                      PrefillOperatorBindingIssue::kWeightBindingIncomplete,
                      artifacts);
    validate_resource(validation, binding.sidecar, requirements.sidecar,
                      PrefillOperatorBindingIssue::kSidecarBindingIncomplete,
                      artifacts);
    validate_resource(
        validation, binding.workspace, requirements.workspace,
        PrefillOperatorBindingIssue::kWorkspaceBindingIncomplete,
        workspaces);
    validate_resource(validation, binding.launcher, requirements.launcher,
                      PrefillOperatorBindingIssue::kLauncherBindingIncomplete,
                      launchers);
    validate_resource(
        validation, binding.completion_event,
        requirements.completion_event,
        PrefillOperatorBindingIssue::kCompletionEventBindingIncomplete,
        completion_events);

    switch (binding.attestation) {
      case PrefillBindingAttestation::kAuthenticatedC8192:
        break;
      case PrefillBindingAttestation::kUnboundDesign:
        add_issue(validation,
                  PrefillOperatorBindingIssue::kC8192AttestationUnbound);
        attestations = false;
        break;
      case PrefillBindingAttestation::kLegacyC512InventoryOnly:
        add_issue(validation,
                  PrefillOperatorBindingIssue::kLegacyC512InventoryOnly);
        attestations = false;
        break;
      default:
        add_issue(validation,
                  PrefillOperatorBindingIssue::kMalformedEnumEncoding);
        attestations = false;
        break;
    }
  }

  bool gate_down_tactics_distinct = true;
  if (gate_up != nullptr && down != nullptr &&
      gate_up->tactic.identity != 0U &&
      gate_up->tactic.identity == down->tactic.identity) {
    add_issue(validation,
              PrefillOperatorBindingIssue::kGateDownTacticAlias);
    gate_down_tactics_distinct = false;
  }

  validation.logical_capacity_complete =
      validation.role_coverage_complete && capacities_valid;
  validation.exact_numerics_complete =
      validation.role_coverage_complete && exact_numerics;
  validation.native_provider_complete =
      validation.role_coverage_complete && native_provider;
  validation.aot_tactics_complete =
      validation.role_coverage_complete && aot_tactics &&
      gate_down_tactics_distinct;
  validation.artifact_bindings_complete =
      validation.role_coverage_complete && artifacts;
  validation.workspace_bindings_complete =
      validation.role_coverage_complete && workspaces;
  validation.launcher_bindings_complete =
      validation.role_coverage_complete && launchers;
  validation.completion_events_complete =
      validation.role_coverage_complete && completion_events;
  validation.forbidden_boundaries_satisfied =
      validation.role_coverage_complete && forbidden_boundaries;
  validation.attestation_complete =
      validation.role_coverage_complete && attestations;
  validation.operator_bindings_complete =
      validation.role_coverage_complete &&
      validation.logical_capacity_complete &&
      validation.exact_numerics_complete &&
      validation.native_provider_complete &&
      validation.aot_tactics_complete &&
      validation.artifact_bindings_complete &&
      validation.workspace_bindings_complete &&
      validation.launcher_bindings_complete &&
      validation.completion_events_complete &&
      validation.forbidden_boundaries_satisfied &&
      validation.attestation_complete && validation.issue_mask == 0U;
  return validation;
}

}  // namespace q3x::runtime
