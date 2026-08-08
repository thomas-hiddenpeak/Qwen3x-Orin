#include "q3x/runtime/prefill_operator_bindings.h"
#include "q3x/runtime/prefill_route_evidence.h"

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

namespace runtime = q3x::runtime;

class TestContext {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

[[nodiscard]] runtime::PrefillOperatorBinding& binding_for(
    runtime::PrefillOperatorBindingSet& set,
    const runtime::PrefillBindingRole role) {
  return set.bindings[static_cast<std::size_t>(role)];
}

void bind_resource(runtime::PrefillResourceBinding& binding,
                   const bool required, const std::uint64_t identity) {
  binding = required
                ? runtime::PrefillResourceBinding{
                      runtime::PrefillResourceBindingState::kBound, identity}
                : runtime::PrefillResourceBinding{
                      runtime::PrefillResourceBindingState::kNotRequired, 0U};
}

void bind_all_contract_fields(runtime::PrefillOperatorBindingSet& set) {
  for (std::size_t index = 0U; index < set.binding_count; ++index) {
    runtime::PrefillOperatorBinding& binding = set.bindings[index];
    const runtime::PrefillOperatorRoleRequirements requirements =
        runtime::prefill_operator_role_requirements(binding.role);
    const std::uint64_t base = 100U + index * 10U;
    binding.numerical_mode = runtime::PrefillNumericalMode::kExact;
    binding.provider = runtime::PrefillOperatorProvider::kNative;
    binding.tactic = {runtime::PrefillTacticMode::kAheadOfTime, base + 1U};
    bind_resource(binding.weight, requirements.weight, base + 2U);
    bind_resource(binding.sidecar, requirements.sidecar, base + 3U);
    bind_resource(binding.workspace, requirements.workspace, base + 4U);
    bind_resource(binding.launcher, requirements.launcher, base + 5U);
    bind_resource(binding.completion_event, requirements.completion_event,
                  base + 6U);
    binding.attestation =
        runtime::PrefillBindingAttestation::kAuthenticatedC8192;
    binding.uses_mtp = false;
  }
}

void test_c8192_shape_is_complete_but_unbound(TestContext& test) {
  const runtime::PrefillOperatorBindingSet set =
      runtime::make_unbound_layer_major_c8192_operator_bindings();
  const runtime::PrefillOperatorBindingValidation validation =
      runtime::validate_layer_major_c8192_operator_bindings(set);

  bool all_roles_are_explicitly_unbound =
      set.binding_count == runtime::kLayerMajorPrefillRequiredOperatorRoleCount;
  for (std::size_t index = 0U; index < set.binding_count; ++index) {
    const runtime::PrefillOperatorBinding& binding = set.bindings[index];
    all_roles_are_explicitly_unbound =
        all_roles_are_explicitly_unbound &&
        binding.role == static_cast<runtime::PrefillBindingRole>(index) &&
        binding.maximum_logical_panel_m == 8'192U &&
        binding.numerical_mode == runtime::PrefillNumericalMode::kExact &&
        binding.provider == runtime::PrefillOperatorProvider::kNative &&
        binding.tactic.mode == runtime::PrefillTacticMode::kAheadOfTime &&
        binding.tactic.identity == 0U &&
        !binding.launcher.bound() && !binding.completion_event.bound() &&
        binding.attestation ==
            runtime::PrefillBindingAttestation::kUnboundDesign &&
        !binding.uses_mtp;
  }
  test.expect(all_roles_are_explicitly_unbound,
              "all 17 C8192 target roles carry logical capacity and unbound "
              "state");
  test.expect(validation.role_coverage_complete &&
                  validation.logical_capacity_complete &&
                  validation.exact_numerics_complete &&
                  validation.native_provider_complete &&
                  validation.forbidden_boundaries_satisfied,
              "C8192 target capacity and hard numerical boundaries are "
              "complete");
  test.expect(!validation.aot_tactics_complete &&
                  !validation.artifact_bindings_complete &&
                  !validation.workspace_bindings_complete &&
                  !validation.launcher_bindings_complete &&
                  !validation.completion_events_complete &&
                  !validation.attestation_complete &&
                  !validation.complete() && !validation.executable(),
              "capacity completeness never promotes unbound target "
              "descriptors");
  test.expect(runtime::has_prefill_operator_binding_issue(
                  validation,
                  runtime::PrefillOperatorBindingIssue::
                      kTacticIdentityUnbound) &&
                  runtime::has_prefill_operator_binding_issue(
                      validation,
                      runtime::PrefillOperatorBindingIssue::
                          kLauncherBindingIncomplete) &&
                  runtime::has_prefill_operator_binding_issue(
                      validation,
                      runtime::PrefillOperatorBindingIssue::
                          kCompletionEventBindingIncomplete) &&
                  runtime::has_prefill_operator_binding_issue(
                      validation,
                      runtime::PrefillOperatorBindingIssue::
                          kC8192AttestationUnbound),
              "unbound tactic, launcher, event, and attestation are visible");
}

void test_missing_and_duplicate_roles_fail_closed(TestContext& test) {
  runtime::PrefillOperatorBindingSet missing =
      runtime::make_unbound_layer_major_c8192_operator_bindings();
  --missing.binding_count;
  const runtime::PrefillOperatorBindingValidation missing_validation =
      runtime::validate_layer_major_c8192_operator_bindings(missing);
  test.expect(!missing_validation.role_coverage_complete &&
                  runtime::has_prefill_operator_binding_issue(
                      missing_validation,
                      runtime::PrefillOperatorBindingIssue::kMissingRole) &&
                  !missing_validation.complete(),
              "a missing final handoff role invalidates the whole contract");

  runtime::PrefillOperatorBindingSet duplicate =
      runtime::make_unbound_layer_major_c8192_operator_bindings();
  duplicate.bindings.back().role =
      runtime::PrefillBindingRole::kNvfp4GateUp;
  const runtime::PrefillOperatorBindingValidation duplicate_validation =
      runtime::validate_layer_major_c8192_operator_bindings(duplicate);
  test.expect(!duplicate_validation.role_coverage_complete &&
                  runtime::has_prefill_operator_binding_issue(
                      duplicate_validation,
                      runtime::PrefillOperatorBindingIssue::kDuplicateRole) &&
                  runtime::has_prefill_operator_binding_issue(
                      duplicate_validation,
                      runtime::PrefillOperatorBindingIssue::kMissingRole) &&
                  !duplicate_validation.complete(),
              "a duplicate cannot substitute for another required role");
}

void test_gate_up_and_down_require_distinct_tactics(TestContext& test) {
  runtime::PrefillOperatorBindingSet set =
      runtime::make_unbound_layer_major_c8192_operator_bindings();
  bind_all_contract_fields(set);
  runtime::PrefillOperatorBinding& gate_up =
      binding_for(set, runtime::PrefillBindingRole::kNvfp4GateUp);
  runtime::PrefillOperatorBinding& down =
      binding_for(set, runtime::PrefillBindingRole::kNvfp4Down);
  down.tactic.identity = gate_up.tactic.identity;

  const runtime::PrefillOperatorBindingValidation validation =
      runtime::validate_layer_major_c8192_operator_bindings(set);
  test.expect(runtime::has_prefill_operator_binding_issue(
                  validation,
                  runtime::PrefillOperatorBindingIssue::
                      kGateDownTacticAlias) &&
                  !validation.aot_tactics_complete &&
                  !validation.complete(),
              "GateUp and Down cannot alias one shape-tactic identity");
}

void test_legacy_c512_inventory_is_incapable_and_unbound(TestContext& test) {
  const runtime::PrefillOperatorBindingSet inventory =
      runtime::make_legacy_c512_prefill_launcher_inventory();
  const runtime::PrefillOperatorBindingValidation validation =
      runtime::validate_layer_major_c8192_operator_bindings(inventory);

  bool explicit_inventory_only = true;
  for (std::size_t index = 0U; index < inventory.binding_count; ++index) {
    const runtime::PrefillOperatorBinding& binding = inventory.bindings[index];
    explicit_inventory_only =
        explicit_inventory_only && binding.maximum_logical_panel_m == 512U &&
        binding.attestation ==
            runtime::PrefillBindingAttestation::kLegacyC512InventoryOnly &&
        !binding.launcher.bound() && !binding.completion_event.bound();
  }
  test.expect(explicit_inventory_only,
              "legacy C512 inventory never appears as a bound C8192 route");
  test.expect(!validation.logical_capacity_complete &&
                  !validation.launcher_bindings_complete &&
                  !validation.completion_events_complete &&
                  runtime::has_prefill_operator_binding_issue(
                      validation,
                      runtime::PrefillOperatorBindingIssue::
                          kInsufficientLogicalPanelCapacity) &&
                  runtime::has_prefill_operator_binding_issue(
                      validation,
                      runtime::PrefillOperatorBindingIssue::
                          kLegacyC512InventoryOnly) &&
                  !validation.complete() && !validation.executable(),
              "C512 inventory is explicitly incapable of the C8192 contract");
}

void test_forbidden_routes_are_independent_failures(TestContext& test) {
  runtime::PrefillOperatorBindingSet set =
      runtime::make_unbound_layer_major_c8192_operator_bindings();
  bind_all_contract_fields(set);
  binding_for(set, runtime::PrefillBindingRole::kNvfp4GateUp).provider =
      runtime::PrefillOperatorProvider::kCuBlasLtReference;
  binding_for(set, runtime::PrefillBindingRole::kExactGdn).uses_mtp = true;
  binding_for(set, runtime::PrefillBindingRole::kExactCausalAttention)
      .numerical_mode = runtime::PrefillNumericalMode::kApproximate;

  const runtime::PrefillOperatorBindingValidation validation =
      runtime::validate_layer_major_c8192_operator_bindings(set);
  test.expect(runtime::has_prefill_operator_binding_issue(
                  validation,
                  runtime::PrefillOperatorBindingIssue::kCuBlasLtForbidden) &&
                  runtime::has_prefill_operator_binding_issue(
                      validation,
                      runtime::PrefillOperatorBindingIssue::kMtpForbidden) &&
                  runtime::has_prefill_operator_binding_issue(
                      validation,
                      runtime::PrefillOperatorBindingIssue::
                          kApproximateNumericsForbidden) &&
                  !validation.native_provider_complete &&
                  !validation.exact_numerics_complete &&
                  !validation.forbidden_boundaries_satisfied &&
                  !validation.complete(),
              "cuBLASLt, MTP, and approximate numerics each fail closed");
}

void test_complete_host_contract_still_cannot_execute(TestContext& test) {
  runtime::PrefillOperatorBindingSet set =
      runtime::make_unbound_layer_major_c8192_operator_bindings();
  bind_all_contract_fields(set);
  const runtime::PrefillOperatorBindingValidation validation =
      runtime::validate_layer_major_c8192_operator_bindings(set);

  test.expect(validation.issue_mask == 0U &&
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
                  validation.attestation_complete && validation.complete(),
              "only the fully bound 17-role C8192 descriptor is complete");
  test.expect(!validation.executable(),
              "host contract completion cannot connect engine or runner");
}

void test_malformed_enums_and_external_provider_fail_closed(
    TestContext& test) {
  runtime::PrefillOperatorBindingSet malformed =
      runtime::make_unbound_layer_major_c8192_operator_bindings();
  bind_all_contract_fields(malformed);
  binding_for(malformed, runtime::PrefillBindingRole::kNvfp4GateUp)
      .numerical_mode = static_cast<runtime::PrefillNumericalMode>(0xffU);
  binding_for(malformed, runtime::PrefillBindingRole::kNvfp4Down).provider =
      static_cast<runtime::PrefillOperatorProvider>(0xffU);
  binding_for(malformed, runtime::PrefillBindingRole::kExactGdn).attestation =
      static_cast<runtime::PrefillBindingAttestation>(0xffU);
  const runtime::PrefillOperatorBindingValidation malformed_validation =
      runtime::validate_layer_major_c8192_operator_bindings(malformed);
  test.expect(runtime::has_prefill_operator_binding_issue(
                  malformed_validation,
                  runtime::PrefillOperatorBindingIssue::
                      kMalformedEnumEncoding) &&
                  !malformed_validation.exact_numerics_complete &&
                  !malformed_validation.native_provider_complete &&
                  !malformed_validation.forbidden_boundaries_satisfied &&
                  !malformed_validation.attestation_complete &&
                  !malformed_validation.complete(),
              "malformed enum encodings cannot pass a bound descriptor");

  runtime::PrefillOperatorBindingSet external =
      runtime::make_unbound_layer_major_c8192_operator_bindings();
  bind_all_contract_fields(external);
  binding_for(external, runtime::PrefillBindingRole::kLinearFp8Qkv).provider =
      runtime::PrefillOperatorProvider::kExternalRuntime;
  const runtime::PrefillOperatorBindingValidation external_validation =
      runtime::validate_layer_major_c8192_operator_bindings(external);
  test.expect(!external_validation.native_provider_complete &&
                  !external_validation.forbidden_boundaries_satisfied &&
                  !external_validation.complete(),
              "an external runtime is neither native nor an allowed boundary");
}

}  // namespace

int main() {
  TestContext test;
  test_c8192_shape_is_complete_but_unbound(test);
  test_missing_and_duplicate_roles_fail_closed(test);
  test_gate_up_and_down_require_distinct_tactics(test);
  test_legacy_c512_inventory_is_incapable_and_unbound(test);
  test_forbidden_routes_are_independent_failures(test);
  test_complete_host_contract_still_cannot_execute(test);
  test_malformed_enums_and_external_provider_fail_closed(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " prefill operator binding test(s) failed\n";
    return 1;
  }
  std::cout << "All prefill operator binding tests passed\n";
  return 0;
}
