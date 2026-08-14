#include "sm87_bulk_dataflow_v2_attention_l2_cohort_launch_internal.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

namespace kernels = q3x::kernels;
namespace execution =
    q3x::kernels::sm87_bulk_v2_attention_execution_detail;

namespace {

static_assert(!std::is_default_constructible_v<
              execution::Sm87BulkV2AttentionSealedAccess>);
static_assert(!std::is_copy_constructible_v<
              execution::Sm87BulkV2AttentionSealedAccess>);
static_assert(execution::kSm87BulkV2AttentionFullLayerBindings == 16U);

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
  return reinterpret_cast<void*>(0x0000'0000'0020'0000ULL);
}

[[nodiscard]] execution::Sm87BulkV2AttentionSealReceipt
passing_seal_receipt() noexcept {
  execution::Sm87BulkV2AttentionSealReceipt receipt;
  receipt.magic = execution::kSm87BulkV2AttentionSealReceiptMagic;
  receipt.version = execution::kSm87BulkV2AttentionSealReceiptVersion;
  receipt.seal_nonce = 21U;
  receipt.deployment_identity = 22U;
  receipt.binding_catalog_identity = 23U;
  receipt.binding_lifetime_owner_identity = 24U;
  receipt.cuda_stream_owner_identity = 25U;
  receipt.device_ordinal = 0;
  receipt.cuda_stream = fake_stream();
  receipt.sealed_layer_bindings =
      execution::kSm87BulkV2AttentionFullLayerBindings;
  receipt.sealed_resource_kernels = 1U;
  receipt.hot_path_static_cuda_queries = 0U;
  receipt.exact_sm87_device_validated = true;
  receipt.nonblocking_stream_validated = true;
  receipt.complete_device_ranges_validated = true;
  receipt.static_resources_validated_at_startup = true;
  receipt.request_hot_path_prevalidated = true;
  receipt.numerical_contract_qualified = false;
  receipt.production_dispatch_eligible = false;
  return receipt;
}

[[nodiscard]] execution::Sm87BulkV2AttentionSubmissionReceipt
passing_submission_receipt() noexcept {
  execution::Sm87BulkV2AttentionSubmissionReceipt receipt;
  receipt.magic = execution::kSm87BulkV2AttentionSubmissionReceiptMagic;
  receipt.version = execution::kSm87BulkV2AttentionSubmissionReceiptVersion;
  receipt.seal_nonce = 21U;
  receipt.request_epoch = 31U;
  receipt.model_layer = 63U;
  receipt.expected_launches = kernels::kSm87BulkV2AttentionKernelLaunches;
  receipt.attempted_launches = receipt.expected_launches;
  receipt.submitted_launches = receipt.expected_launches;
  receipt.failed_launch_ordinal = std::numeric_limits<std::size_t>::max();
  receipt.cuda_error = 0;
  receipt.state = execution::Sm87BulkV2AttentionSubmissionState::kSubmitted;
  receipt.prevalidated_hot_path_used = true;
  receipt.static_cuda_query_issued = false;
  receipt.owner_drain_required = false;
  return receipt;
}

void test_full_layer_mapping(TestContext& test) {
  for (std::size_t ordinal = 0U;
       ordinal < execution::kSm87BulkV2AttentionFullLayerBindings;
       ++ordinal) {
    const std::size_t layer = ordinal * 4U + 3U;
    test.expect(execution::sm87_bulk_v2_attention_full_ordinal(layer) ==
                    ordinal,
                "all 16 full-Attention layers map canonically");
  }
  test.expect(execution::sm87_bulk_v2_attention_full_ordinal(0U) ==
                  execution::kSm87BulkV2AttentionFullLayerBindings &&
                  execution::sm87_bulk_v2_attention_full_ordinal(62U) ==
                      execution::kSm87BulkV2AttentionFullLayerBindings &&
                  execution::sm87_bulk_v2_attention_full_ordinal(64U) ==
                      execution::kSm87BulkV2AttentionFullLayerBindings,
              "GDN and out-of-model layers fail closed");
}

void test_seal_receipt(TestContext& test) {
  const auto receipt = passing_seal_receipt();
  test.expect(execution::sm87_bulk_v2_attention_seal_receipt_valid(receipt),
              "a complete 16-layer Attention startup seal validates");

  auto changed = receipt;
  changed.sealed_layer_bindings = 15U;
  test.expect(
      !execution::sm87_bulk_v2_attention_seal_receipt_valid(changed),
      "a partial Attention binding catalog cannot be sealed");
  changed = receipt;
  changed.hot_path_static_cuda_queries = 1U;
  test.expect(
      !execution::sm87_bulk_v2_attention_seal_receipt_valid(changed),
      "the hot Attention contract admits zero static CUDA queries");
  changed = receipt;
  changed.complete_device_ranges_validated = false;
  test.expect(
      !execution::sm87_bulk_v2_attention_seal_receipt_valid(changed),
      "endpoint-only pointer checks cannot form an Attention seal");
  changed = receipt;
  changed.static_resources_validated_at_startup = false;
  test.expect(
      !execution::sm87_bulk_v2_attention_seal_receipt_valid(changed),
      "resource queries must close before request admission");
  changed = receipt;
  changed.numerical_contract_qualified = true;
  test.expect(
      !execution::sm87_bulk_v2_attention_seal_receipt_valid(changed),
      "the launch seam cannot forge numerical qualification");
  changed = receipt;
  changed.production_dispatch_eligible = true;
  test.expect(
      !execution::sm87_bulk_v2_attention_seal_receipt_valid(changed),
      "the launch seam remains default-off");
}

void test_submission_receipt(TestContext& test) {
  const auto success = passing_submission_receipt();
  test.expect(
      execution::sm87_bulk_v2_attention_submission_receipt_valid(success),
      "four accepted same-head launches form a submitted receipt");

  auto changed = success;
  changed.static_cuda_query_issued = true;
  test.expect(
      !execution::sm87_bulk_v2_attention_submission_receipt_valid(changed),
      "a static CUDA query invalidates hot-path evidence");
  changed = success;
  changed.model_layer = 62U;
  test.expect(
      !execution::sm87_bulk_v2_attention_submission_receipt_valid(changed),
      "an Attention receipt cannot name a GDN layer");

  auto first_failure = success;
  first_failure.state = execution::Sm87BulkV2AttentionSubmissionState::
      kFailedBeforeSubmission;
  first_failure.attempted_launches = 1U;
  first_failure.submitted_launches = 0U;
  first_failure.failed_launch_ordinal = 0U;
  first_failure.cuda_error = 9;
  first_failure.owner_drain_required = false;
  test.expect(
      execution::sm87_bulk_v2_attention_submission_receipt_valid(
          first_failure),
      "a first-launch failure proves no Attention work submitted");

  auto stale_error = first_failure;
  stale_error.attempted_launches = 0U;
  stale_error.failed_launch_ordinal =
      std::numeric_limits<std::size_t>::max();
  test.expect(
      execution::sm87_bulk_v2_attention_submission_receipt_valid(
          stale_error),
      "a preexisting CUDA error fails before any Attention launch attempt");

  auto partial = success;
  partial.state = execution::Sm87BulkV2AttentionSubmissionState::
      kFailedAfterPartialSubmission;
  partial.attempted_launches = 3U;
  partial.submitted_launches = 2U;
  partial.failed_launch_ordinal = 2U;
  partial.cuda_error = 9;
  partial.owner_drain_required = true;
  test.expect(
      execution::sm87_bulk_v2_attention_submission_receipt_valid(partial),
      "partial Attention submission requires owner drain/poison");
  partial.owner_drain_required = false;
  test.expect(
      !execution::sm87_bulk_v2_attention_submission_receipt_valid(partial),
      "partial Attention work cannot silently return to reusable state");
}

}  // namespace

int main() {
  TestContext test;
  test_full_layer_mapping(test);
  test_seal_receipt(test);
  test_submission_receipt(test);
  if (test.failures() != 0) {
    return 1;
  }
  std::cout << "SM87 bulk-v2 Attention startup-seal/prevalidated-enqueue "
               "host contract checks passed\n";
  return 0;
}
