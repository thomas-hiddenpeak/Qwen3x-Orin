#include "support/p40_projection_work_manifest.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace manifest = q3x::tests::p40_work_manifest;

namespace {

static_assert(manifest::kLogicalRoleCount == 496U);
static_assert(manifest::kOuterOperationCount == 304U);
static_assert(manifest::kPhysicalLaunchCount == 1'216U);
static_assert(manifest::kFp8LogicalRoleCount == 208U);
static_assert(manifest::kNvFp4LogicalRoleCount == 192U);
static_assert(manifest::kBf16LogicalRoleCount == 96U);
static_assert(manifest::kFp8OuterOperationCount == 128U);
static_assert(manifest::kNvFp4OuterOperationCount == 128U);
static_assert(manifest::kBf16OuterOperationCount == 48U);
static_assert(manifest::kFp8PhysicalLaunchCount == 1'040U);
static_assert(manifest::kNvFp4PhysicalLaunchCount == 128U);
static_assert(manifest::kBf16PhysicalLaunchCount == 48U);
static_assert(manifest::kProjectionConventionalOperations ==
              1'948'044'492'800'000ULL);

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

[[nodiscard]] const manifest::LogicalProjectionRole* find_role(
    const std::size_t layer,
    const manifest::CheckpointSubrole subrole) noexcept {
  for (const auto& role : manifest::kManifest.logical_roles) {
    if (role.layer == layer && role.subrole == subrole) {
      return &role;
    }
  }
  return nullptr;
}

void test_three_level_inventory(TestContext& test) {
  const auto result = manifest::check_manifest(manifest::kManifest);
  test.expect(result.valid && result.issue == manifest::ManifestIssue::kNone,
              "canonical v10 P40 manifest closes");
  test.expect(result.totals.fp8.logical_roles == 208U &&
                  result.totals.nvfp4.logical_roles == 192U &&
                  result.totals.bf16.logical_roles == 96U &&
                  result.totals.fp8.outer_operations == 128U &&
                  result.totals.nvfp4.outer_operations == 128U &&
                  result.totals.bf16.outer_operations == 48U &&
                  result.totals.fp8.physical_launches == 1'040U &&
                  result.totals.nvfp4.physical_launches == 128U &&
                  result.totals.bf16.physical_launches == 48U &&
                  result.totals.logical_conventional_operations ==
                      1'948'044'492'800'000ULL &&
                  result.totals.physical_conventional_operations ==
                      1'948'044'492'800'000ULL,
              "all family, outer, launch, and operation totals close");
  test.expect(!manifest::kManifest.production_reachable &&
                  manifest::kManifest.diagnostic_only,
              "manifest is diagnostic-only and not production-reachable");
  test.expect(manifest::kSchemaIdentity ==
                  std::string_view("q3x.p40.projection-work-manifest.v1") &&
                  manifest::kRouteIdentity ==
                      std::string_view(
                          "q3x.sm87.ac-prefill-prompt-wide-v2.native-p40-"
                          "whole-core.v1"),
              "schema and current-v10 route identities are fixed");

  std::size_t linear_layers = 0U;
  std::size_t full_layers = 0U;
  for (std::size_t layer = 0U; layer < manifest::kLayerCount; ++layer) {
    manifest::is_full_attention_layer(layer) ? ++full_layers
                                              : ++linear_layers;
  }
  test.expect(linear_layers == 48U && full_layers == 16U,
              "64-layer schedule is 48 linear plus 16 full attention");

  const auto* linear_qkv =
      find_role(0U, manifest::CheckpointSubrole::kLinearQkv);
  const auto* linear_z = find_role(0U, manifest::CheckpointSubrole::kLinearZ);
  const auto* linear_a = find_role(0U, manifest::CheckpointSubrole::kLinearA);
  const auto* full_q = find_role(3U, manifest::CheckpointSubrole::kFullQGate);
  const auto* full_k = find_role(3U, manifest::CheckpointSubrole::kFullK);
  const auto* gate = find_role(0U, manifest::CheckpointSubrole::kMlpGate);
  const auto* up = find_role(0U, manifest::CheckpointSubrole::kMlpUp);
  const auto* down = find_role(0U, manifest::CheckpointSubrole::kMlpDown);
  test.expect(linear_qkv != nullptr && linear_qkv->m == 40'000U &&
                  linear_qkv->n == 10'240U && linear_qkv->k == 5'120U,
              "linear QKV checkpoint/output role has exact M/N/K");
  test.expect(linear_z != nullptr && linear_z->n == 6'144U &&
                  linear_z->k == 5'120U &&
                  linear_z->outer_operation == linear_qkv->outer_operation,
              "linear QKV and Z map to one fused outer operation");
  test.expect(linear_a != nullptr && linear_a->n == 48U &&
                  linear_a->k == 5'120U &&
                  linear_a->family == manifest::ProjectionFamily::kBf16,
              "linear A/B role shape and BF16 family are explicit");
  test.expect(full_q != nullptr && full_k != nullptr &&
                  full_q->n == 12'288U && full_k->n == 1'024U &&
                  full_q->outer_operation == full_k->outer_operation,
              "full Q+gate/K/V roles map to the same outer operation");
  test.expect(gate != nullptr && up != nullptr && down != nullptr &&
                  gate->n == 17'408U && gate->k == 5'120U &&
                  up->outer_operation == gate->outer_operation &&
                  down->n == 5'120U && down->k == 17'408U,
              "Gate/Up and K-heavy Down shapes are distinguished");
  test.expect(gate != nullptr && up != nullptr &&
                  gate->scale.source_partition !=
                      up->scale.source_partition &&
                  gate->scale.physical_artifact ==
                      up->scale.physical_artifact &&
                  gate->scale.physical_output_offset == 0U &&
                  up->scale.physical_output_offset == 17'408U,
              "Gate/Up preserve source scale partitions inside one artifact");

  const auto& linear_input_outer =
      manifest::kManifest.outer_operations[linear_qkv->outer_operation];
  const auto& full_input_outer =
      manifest::kManifest.outer_operations[full_q->outer_operation];
  const auto& gate_up_outer =
      manifest::kManifest.outer_operations[gate->outer_operation];
  test.expect(linear_input_outer.logical_count == 2U &&
                  linear_input_outer.physical_count == 10U,
              "linear QKV+Z outer operation lowers to 2x5 P8000 launches");
  test.expect(full_input_outer.logical_count == 3U &&
                  full_input_outer.physical_count == 15U,
              "full Q/K/V outer operation lowers to 3x5 P8000 launches");
  test.expect(gate_up_outer.logical_count == 2U &&
                  gate_up_outer.physical_count == 1U,
              "Gate+Up outer operation lowers to one P40000 launch");

  for (const auto& role : manifest::kManifest.logical_roles) {
    test.expect(!manifest::name(role.family).empty() &&
                    !manifest::name(role.layer_kind).empty() &&
                    !manifest::name(role.subrole).empty() &&
                    !manifest::name(role.algorithm).empty() &&
                    !manifest::name(role.activation_dtype).empty() &&
                    !manifest::name(role.scale.kind).empty(),
                "every logical role has printable diagnostic identities");
  }
}

void test_fail_closed_mutations(TestContext& test) {
  {
    auto changed = manifest::kManifest;
    ++changed.physical_launches[0U].token_offset;
    const auto result = manifest::check_manifest(changed);
    test.expect(!result.valid &&
                    result.issue == manifest::ManifestIssue::kPhysicalLaunch,
                "a shifted P8000 launch fails closed");
  }
  {
    auto changed = manifest::kManifest;
    const auto* gate =
        find_role(0U, manifest::CheckpointSubrole::kMlpGate);
    const auto* up = find_role(0U, manifest::CheckpointSubrole::kMlpUp);
    if (gate != nullptr && up != nullptr) {
      changed.logical_roles[up->ordinal].scale.source_partition =
          changed.logical_roles[gate->ordinal].scale.source_partition;
    }
    const auto result = manifest::check_manifest(changed);
    test.expect(!result.valid &&
                    result.issue == manifest::ManifestIssue::kScalePartition,
                "Gate/Up source-scale aliasing fails closed");
  }
  {
    auto changed = manifest::kManifest;
    ++changed.logical_roles[0U].n;
    const auto result = manifest::check_manifest(changed);
    test.expect(!result.valid &&
                    result.issue == manifest::ManifestIssue::kLogicalRole,
                "M/N/K drift fails closed before totals are accepted");
  }
  {
    auto changed = manifest::kManifest;
    changed.outer_operations[0U].physical_count = 9U;
    const auto result = manifest::check_manifest(changed);
    test.expect(!result.valid &&
                    result.issue ==
                        manifest::ManifestIssue::kPhysicalLaunch,
                "outer-to-physical launch range drift fails closed");
  }
}

}  // namespace

int main() {
  TestContext test;
  test_three_level_inventory(test);
  test_fail_closed_mutations(test);
  if (test.failures() != 0) {
    return 1;
  }
  std::cout << "P40 v10 projection work manifest: 496 logical roles, 304 "
               "outer operations, 1216 physical CUDA launches, "
               "1948044492800000 conventional operations\n";
  return 0;
}
