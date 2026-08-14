#include "sm87_bulk_dataflow_v2_fp8_projection_launch_internal.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

namespace kernels = q3x::kernels;
namespace execution =
    q3x::kernels::sm87_bulk_v2_fp8_execution_detail;

namespace {

using Role = kernels::Sm87TargetAotProjectionRole;

static_assert(!std::is_default_constructible_v<
              execution::Sm87BulkV2Fp8SealedAccess>);
static_assert(!std::is_copy_constructible_v<
              execution::Sm87BulkV2Fp8SealedAccess>);

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

[[nodiscard]] void* fake_stream() noexcept {
  return reinterpret_cast<void*>(0x0000'0000'0010'0000ULL);
}

[[nodiscard]] execution::Sm87BulkV2Fp8SealReceipt
passing_seal_receipt() noexcept {
  execution::Sm87BulkV2Fp8SealReceipt receipt;
  receipt.magic = execution::kSm87BulkV2Fp8SealReceiptMagic;
  receipt.version = execution::kSm87BulkV2Fp8SealReceiptVersion;
  receipt.seal_nonce = 11U;
  receipt.deployment_identity = 12U;
  receipt.binding_catalog_identity = 13U;
  receipt.binding_lifetime_owner_identity = 14U;
  receipt.cuda_stream_owner_identity = 15U;
  receipt.authenticated_weight_owner_identity = 16U;
  receipt.device_ordinal = 0;
  receipt.cuda_stream = fake_stream();
  receipt.sealed_role_bindings = kernels::kSm87BulkV2Fp8LogicalRoleCount;
  receipt.sealed_resource_roles = 3U;
  receipt.hot_path_static_cuda_queries = 0U;
  receipt.exact_sm87_device_validated = true;
  receipt.nonblocking_stream_validated = true;
  receipt.complete_device_ranges_validated = true;
  receipt.all_authenticated_weight_owners_match = true;
  receipt.static_resources_validated_at_startup = true;
  receipt.request_hot_path_prevalidated = true;
  receipt.numerical_contract_qualified = false;
  receipt.production_dispatch_eligible = false;
  return receipt;
}

[[nodiscard]] execution::Sm87BulkV2Fp8SubmissionReceipt
passing_submission_receipt() noexcept {
  execution::Sm87BulkV2Fp8SubmissionReceipt receipt;
  receipt.magic = execution::kSm87BulkV2Fp8SubmissionReceiptMagic;
  receipt.version = execution::kSm87BulkV2Fp8SubmissionReceiptVersion;
  receipt.seal_nonce = 11U;
  receipt.request_epoch = 29U;
  receipt.layer = 3U;
  receipt.role = Role::kFp8FullQkv;
  receipt.expected_launches = kernels::kSm87BulkV2Fp8SegmentsPerRole;
  receipt.attempted_launches = receipt.expected_launches;
  receipt.submitted_launches = receipt.expected_launches;
  receipt.failed_launch_ordinal = std::numeric_limits<std::size_t>::max();
  receipt.cuda_error = 0;
  receipt.state = execution::Sm87BulkV2Fp8SubmissionState::kSubmitted;
  receipt.prevalidated_hot_path_used = true;
  receipt.static_cuda_query_issued = false;
  receipt.owner_drain_required = false;
  return receipt;
}

void test_canonical_role_mapping(TestContext& test) {
  std::size_t observed = 0U;
  for (std::size_t layer = 0U;
       layer < kernels::kSm87BulkV2Fp8LayerCount; ++layer) {
    const Role input = (layer + 1U) % 4U == 0U
                           ? Role::kFp8FullQkv
                           : Role::kFp8GdnQkvZ;
    test.expect(execution::sm87_bulk_v2_fp8_role_ordinal(layer, input) ==
                    layer * 2U,
                "every layer has exactly one canonical FP8 input role");
    test.expect(
        execution::sm87_bulk_v2_fp8_role_ordinal(
            layer, Role::kFp8AttentionOutput) == layer * 2U + 1U,
        "every layer has one canonical FP8 output role");
    const Role wrong_input = input == Role::kFp8FullQkv
                                 ? Role::kFp8GdnQkvZ
                                 : Role::kFp8FullQkv;
    test.expect(execution::sm87_bulk_v2_fp8_role_ordinal(
                    layer, wrong_input) ==
                    kernels::kSm87BulkV2Fp8LogicalRoleCount,
                "the non-model input role fails closed");
    observed += 2U;
  }
  test.expect(observed == kernels::kSm87BulkV2Fp8LogicalRoleCount,
              "the startup catalog closes all 128 bindings");
}

void test_seal_receipt(TestContext& test) {
  const auto receipt = passing_seal_receipt();
  test.expect(execution::sm87_bulk_v2_fp8_seal_receipt_valid(receipt),
              "a complete startup-only FP8 seal validates");

  auto changed = receipt;
  changed.sealed_role_bindings = 127U;
  test.expect(!execution::sm87_bulk_v2_fp8_seal_receipt_valid(changed),
              "a partial role catalog cannot be sealed");
  changed = receipt;
  changed.hot_path_static_cuda_queries = 1U;
  test.expect(!execution::sm87_bulk_v2_fp8_seal_receipt_valid(changed),
              "the contract rejects any hot static CUDA query");
  changed = receipt;
  changed.complete_device_ranges_validated = false;
  test.expect(!execution::sm87_bulk_v2_fp8_seal_receipt_valid(changed),
              "endpoint-only pointer validation cannot form a seal");
  changed = receipt;
  changed.nonblocking_stream_validated = false;
  test.expect(!execution::sm87_bulk_v2_fp8_seal_receipt_valid(changed),
              "an unchecked stream cannot form a seal");
  changed = receipt;
  changed.numerical_contract_qualified = true;
  test.expect(!execution::sm87_bulk_v2_fp8_seal_receipt_valid(changed),
              "a dispatch seal cannot forge numerical qualification");
  changed = receipt;
  changed.production_dispatch_eligible = true;
  test.expect(!execution::sm87_bulk_v2_fp8_seal_receipt_valid(changed),
              "the internal seam remains default-off");
}

void test_submission_receipt(TestContext& test) {
  const auto success = passing_submission_receipt();
  test.expect(
      execution::sm87_bulk_v2_fp8_submission_receipt_valid(success),
      "40 accepted launches form one submitted role receipt");

  auto changed = success;
  changed.static_cuda_query_issued = true;
  test.expect(
      !execution::sm87_bulk_v2_fp8_submission_receipt_valid(changed),
      "a static CUDA query invalidates hot-path evidence");
  changed = success;
  changed.role = Role::kFp8GdnQkvZ;
  test.expect(
      !execution::sm87_bulk_v2_fp8_submission_receipt_valid(changed),
      "the receipt enforces the model layer/role identity");

  auto first_failure = success;
  first_failure.state =
      execution::Sm87BulkV2Fp8SubmissionState::kFailedBeforeSubmission;
  first_failure.attempted_launches = 1U;
  first_failure.submitted_launches = 0U;
  first_failure.failed_launch_ordinal = 0U;
  first_failure.cuda_error = 9;
  first_failure.owner_drain_required = false;
  test.expect(execution::sm87_bulk_v2_fp8_submission_receipt_valid(
                  first_failure),
              "a first-launch failure proves no constituent work submitted");

  auto stale_error = first_failure;
  stale_error.attempted_launches = 0U;
  stale_error.failed_launch_ordinal =
      std::numeric_limits<std::size_t>::max();
  test.expect(execution::sm87_bulk_v2_fp8_submission_receipt_valid(
                  stale_error),
              "a preexisting CUDA error fails before any FP8 launch attempt");

  auto partial = success;
  partial.state = execution::Sm87BulkV2Fp8SubmissionState::
      kFailedAfterPartialSubmission;
  partial.attempted_launches = 18U;
  partial.submitted_launches = 17U;
  partial.failed_launch_ordinal = 17U;
  partial.cuda_error = 9;
  partial.owner_drain_required = true;
  test.expect(execution::sm87_bulk_v2_fp8_submission_receipt_valid(partial),
              "partial FP8 submission requires owner drain/poison");
  partial.owner_drain_required = false;
  test.expect(
      !execution::sm87_bulk_v2_fp8_submission_receipt_valid(partial),
      "partial submission cannot silently return to reusable state");
}

}  // namespace

int main() {
  TestContext test;
  test_canonical_role_mapping(test);
  test_seal_receipt(test);
  test_submission_receipt(test);
  if (test.failures() != 0) {
    return 1;
  }
  std::cout << "SM87 bulk-v2 FP8 startup-seal/prevalidated-enqueue host "
               "contract checks passed\n";
  return 0;
}
