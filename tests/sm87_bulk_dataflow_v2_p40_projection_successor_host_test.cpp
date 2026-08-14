#include "q3x/runtime/sm87_bulk_dataflow_v2_p40_projection_successor.h"

#include <cstddef>
#include <iostream>

namespace runtime = q3x::runtime;

namespace {

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

[[nodiscard]] constexpr runtime::Sm87BulkV2P40ProjectionSuccessorReceipt
complete_successor_receipt() noexcept {
  auto receipt = runtime::sm87_bulk_v2_p40_projection_successor_receipt();
  receipt.fp8_gdn_input_whole_launches =
      runtime::kSm87BulkV2P40Fp8GdnInputOuterOperations;
  receipt.fp8_full_input_whole_launches =
      runtime::kSm87BulkV2P40Fp8FullInputOuterOperations;
  receipt.fp8_output_whole_launches =
      runtime::kSm87BulkV2P40Fp8OutputOuterOperations;
  receipt.fp8_whole_role_launches =
      runtime::kSm87BulkV2P40Fp8WholeRoleLaunches;
  receipt.nvfp4_gate_up_whole_launches =
      runtime::kSm87BulkV2P40NvFp4GateUpOuterOperations;
  receipt.nvfp4_down_whole_launches =
      runtime::kSm87BulkV2P40NvFp4DownOuterOperations;
  receipt.nvfp4_whole_role_launches =
      runtime::kSm87BulkV2P40NvFp4WholeRoleLaunches;
  receipt.bf16_ab_physical_launches =
      runtime::kSm87BulkV2P40Bf16AbPhysicalLaunches;
  return receipt;
}

void test_manifest(TestContext& test) {
  constexpr auto manifest =
      runtime::kSm87BulkV2P40FrozenProjectionSuccessorManifest;
  static_assert(
      runtime::sm87_bulk_v2_p40_projection_successor_manifest_valid(
          manifest));
  test.expect(
      runtime::sm87_bulk_v2_p40_projection_successor_manifest_valid(
          manifest),
      "the frozen whole-P40000 projection successor manifest validates");
  test.expect(manifest.fp8_logical_roles == 208U &&
                  manifest.fp8_fused_outer_operations == 128U &&
                  manifest.fp8_whole_role_launches == 128U,
              "FP8 keeps 208 checkpoint roles while 128 fused roles become 128 whole launches");
  test.expect(manifest.nvfp4_logical_roles == 192U &&
                  manifest.nvfp4_fused_outer_operations == 128U &&
                  manifest.nvfp4_whole_role_launches == 128U,
              "NVFP4 keeps 192 checkpoint roles while 128 fused roles become 128 whole launches");
  test.expect(manifest.logical_projection_roles == 496U &&
                  manifest.fused_outer_operations == 304U &&
                  manifest.conventional_operations ==
                      1'948'044'492'800'000ULL,
              "whole-successor topology preserves the matched logical and arithmetic ledger");
  test.expect(manifest.fp8_exact_control_launches == 5'120U &&
                  manifest.nvfp4_exact_control_launches == 2'560U &&
                  manifest.exact_controls_are_oracle_only &&
                  !manifest.exact_controls_can_satisfy_successor_receipt,
              "the old segmented controls are retained only as exact oracles");
  test.expect(manifest.default_off &&
                  !manifest.numerical_contract_qualified &&
                  !manifest.static_resource_contract_qualified &&
                  !manifest.production_dispatch_eligible,
              "the topology manifest grants no numerical, resource, or production qualification");

  auto wrong = manifest;
  wrong.fp8_whole_role_launches = 5'120U;
  test.expect(
      !runtime::sm87_bulk_v2_p40_projection_successor_manifest_valid(
          wrong),
      "the FP8 exact-control launch count cannot relabel the successor");
  wrong = manifest;
  wrong.nvfp4_whole_role_launches = 2'560U;
  test.expect(
      !runtime::sm87_bulk_v2_p40_projection_successor_manifest_valid(
          wrong),
      "the NVFP4 exact-control launch count cannot relabel the successor");
  wrong = manifest;
  wrong.numerical_contract_qualified = true;
  test.expect(
      !runtime::sm87_bulk_v2_p40_projection_successor_manifest_valid(
          wrong),
      "host topology cannot forge numerical qualification");
}

void test_receipt(TestContext& test) {
  constexpr auto completed = complete_successor_receipt();
  static_assert(
      runtime::sm87_bulk_v2_p40_projection_successor_receipt_complete(
          completed));
  test.expect(
      runtime::sm87_bulk_v2_p40_projection_successor_receipt_complete(
          completed),
      "all five whole-role classes close one successor receipt");

  auto partial = completed;
  --partial.fp8_gdn_input_whole_launches;
  test.expect(
      !runtime::sm87_bulk_v2_p40_projection_successor_receipt_complete(
          partial),
      "one missing GDN input whole launch rejects completion");
  partial = completed;
  --partial.nvfp4_down_whole_launches;
  test.expect(
      !runtime::sm87_bulk_v2_p40_projection_successor_receipt_complete(
          partial),
      "one missing Down whole launch rejects completion");

  auto exact_control = completed;
  exact_control.route =
      runtime::Sm87BulkV2P40ProjectionRoute::kExactControlSteppingStones;
  exact_control.fp8_gdn_input_whole_launches = 0U;
  exact_control.fp8_full_input_whole_launches = 0U;
  exact_control.fp8_output_whole_launches = 0U;
  exact_control.fp8_whole_role_launches = 0U;
  exact_control.nvfp4_gate_up_whole_launches = 0U;
  exact_control.nvfp4_down_whole_launches = 0U;
  exact_control.nvfp4_whole_role_launches = 0U;
  exact_control.fp8_exact_control_launches =
      runtime::kSm87BulkV2P40Fp8ExactControlLaunches;
  exact_control.nvfp4_exact_control_launches =
      runtime::kSm87BulkV2P40NvFp4ExactControlLaunches;
  test.expect(
      !runtime::sm87_bulk_v2_p40_projection_successor_receipt_complete(
          exact_control),
      "a complete old-control execution cannot satisfy the successor receipt");

  auto mixed = completed;
  mixed.fp8_exact_control_launches = 1U;
  test.expect(
      !runtime::sm87_bulk_v2_p40_projection_successor_receipt_complete(
          mixed),
      "a mixed successor/control execution fails closed");
  mixed = completed;
  mixed.production_dispatch_eligible = true;
  test.expect(
      !runtime::sm87_bulk_v2_p40_projection_successor_receipt_complete(
          mixed),
      "a host receipt cannot grant production dispatch");
}

}  // namespace

int main() {
  TestContext test;
  test_manifest(test);
  test_receipt(test);
  if (test.failures() != 0) {
    return 1;
  }
  std::cout << "SM87 bulk-dataflow-v2 P40 projection successor host contract passed\n";
  return 0;
}
