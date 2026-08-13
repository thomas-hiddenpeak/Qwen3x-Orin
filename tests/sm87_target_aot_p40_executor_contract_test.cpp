#include "../src/runtime/sm87_target_aot_p40_executor_internal.h"

#include <cstddef>
#include <iostream>
#include <type_traits>

namespace executor =
    q3x::runtime::sm87_target_aot_p40_executor_detail;
namespace runtime = q3x::runtime;

namespace {

constexpr auto kContract =
    executor::sm87_target_aot_p40_execution_contract();

static_assert(kContract.valid());
static_assert(kContract.prompt_tokens == 40'000U);
static_assert(kContract.handoff_tokens == 1U);
static_assert(kContract.layers == 64U);
static_assert(kContract.gdn_layers == 48U);
static_assert(kContract.full_attention_layers == 16U);
static_assert(kContract.full_attention_panel_tokens == 8'000U);
static_assert(kContract.full_attention_panels_per_layer == 5U);
static_assert(kContract.projection_assets == 256U);
static_assert(kContract.layer_completion_events == 512U);
static_assert(kContract.global_completion_events == 7U);
static_assert(kContract.producer_lanes == 2U);
static_assert(kContract.producer_join_event_slots == 2U);
static_assert(kContract.producer_launches_initially_serial);
static_assert(!kContract.producer_parallelism_claimed);
static_assert(!kContract.mtp_permitted);
static_assert(!kContract.fallback_permitted);
static_assert(!kContract.cublaslt_permitted);
static_assert(!kContract.jit_permitted);
static_assert(kContract.final_handoff_callable);
static_assert(!std::is_default_constructible_v<
              executor::Sm87TargetAotP40Executor>);
static_assert(!std::is_copy_constructible_v<
              executor::Sm87TargetAotP40Executor>);
static_assert(!std::is_move_constructible_v<
              executor::Sm87TargetAotP40Executor>);

#if defined(Q3X_EXPECT_SM87_TARGET_AOT_P40_EXECUTOR_V1_ADMISSION)
static_assert(executor::kSm87TargetAotP40ExecutorAdmissionCompiled);
#else
static_assert(!executor::kSm87TargetAotP40ExecutorAdmissionCompiled);
#endif

class TestContext final {
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

void test_exact_request_boundary(TestContext& test) {
  test.expect(executor::sm87_target_aot_p40_exact_request_shape(40'000U, 1U),
              "the only admitted shape is P40000 plus one-token handoff");
  test.expect(!executor::sm87_target_aot_p40_exact_request_shape(39'999U, 1U),
              "a shorter prompt cannot enter this exact executor");
  test.expect(!executor::sm87_target_aot_p40_exact_request_shape(40'001U, 1U),
              "request capacity is not a second prompt shape");
  test.expect(!executor::sm87_target_aot_p40_exact_request_shape(40'000U, 0U),
              "zero-token handoff is rejected");
  test.expect(!executor::sm87_target_aot_p40_exact_request_shape(40'000U, 2U),
              "MTP/multi-token handoff is rejected");
}

void test_canonical_layer_and_panel_schedule(TestContext& test) {
  std::size_t gdn = 0U;
  std::size_t full = 0U;
  std::size_t panels = 0U;
  std::size_t layer_events = 0U;
  for (std::size_t layer = 0U; layer < kContract.layers; ++layer) {
    if ((layer + 1U) % 4U == 0U) {
      ++full;
      panels += kContract.full_attention_panels_per_layer;
    } else {
      ++gdn;
    }
    layer_events += runtime::kSm87TargetAotP40LayerEventCount;
  }
  test.expect(gdn == 48U && full == 16U,
              "the executor follows the model's exact 48 GDN / 16 Full map");
  test.expect(panels == 80U,
              "all Full layers contain exactly five real P8000 preprocesses");
  test.expect(layer_events == 512U &&
                  layer_events == kContract.layer_completion_events,
              "all eight ordered owner events exist for every layer");
}

void test_final_handoff_scratch_lifetime(TestContext& test) {
  constexpr auto scratch =
      executor::sm87_target_aot_p40_final_handoff_scratch_contract();
  static_assert(scratch.valid());
  test.expect(scratch.logits_offset_bytes == 0U &&
                  scratch.logits_bytes == 496'640U,
              "the dead family prefix owns exact [248320] BF16 logits");
  test.expect(scratch.greedy_workspace_offset_bytes == 496'640U &&
                  scratch.greedy_workspace_bytes == 264U &&
                  scratch.required_bytes == 496'904U,
              "the 33-result greedy workspace follows logits without padding");
  test.expect(scratch.required_bytes <= scratch.family_span_bytes &&
                  scratch.begins_at_family_arena &&
                  scratch.live_only_after_all_layers &&
                  !scratch.overlaps_final_hidden,
              "handoff reuses only dead family storage and excludes final_hidden");
}

void test_committed_handoff_is_the_only_success(TestContext& test) {
  executor::Sm87TargetAotP40ExecutionResult result;
  result.receipt.prompt_tokens = 40'000U;
  result.receipt.requested_handoff_tokens = 1U;
  result.receipt.completed_layers = 64U;
  result.receipt.completed_gdn_layers = 48U;
  result.receipt.completed_full_attention_layers = 16U;
  result.receipt.completed_attention_panels = 80U;
  result.receipt.recorded_layer_events = 512U;
  result.receipt.recorded_global_events = 7U;
  result.receipt.handoff_token_id = 42U;
  result.receipt.handoff_value_bits = 0x3F80U;
  result.receipt.handoff_has_nonfinite = 0U;
  result.receipt.finalization =
      executor::Sm87TargetAotP40Finalization::kCommitted;
  result.receipt.transaction_started = true;
  result.receipt.transaction_committed = true;
  result.receipt.handoff_result_observed = true;
  result.receipt.handoff_complete = true;
  test.expect(static_cast<bool>(result),
              "a finite observed token plus owner commit is success");
  test.expect(result.receipt.recorded_global_events ==
                  runtime::kSm87TargetAotP40GlobalEventCount,
              "the receipt includes the owner-issued request commit event");

  result.receipt.handoff_has_nonfinite = 1U;
  result.receipt.transaction_committed = false;
  result.receipt.handoff_complete = false;
  result.receipt.transaction_cancelled = true;
  result.receipt.finalization =
      executor::Sm87TargetAotP40Finalization::kLogitsReady;
  result.status.code =
      executor::Sm87TargetAotP40ExecutorError::kInvalidHandoffResult;
  result.status.context = "nonfinite_bf16_logits";
  test.expect(!static_cast<bool>(result),
              "an observed nonfinite logit cancels instead of publishing");
}

void test_admission_binary_links_executor(TestContext& test) {
#if defined(Q3X_EXPECT_SM87_TARGET_AOT_P40_EXECUTOR_V1_ADMISSION)
  // Taking the address of this non-inline definition forces the admission
  // test binary to resolve the real executor object from q3x_engine.  This is
  // a link smoke, not a substitute for the real-weight P40000 API run.
  using BindFunction = decltype(&executor::Sm87TargetAotP40Executor::bind);
  volatile BindFunction bind_symbol =
      &executor::Sm87TargetAotP40Executor::bind;
  test.expect(bind_symbol != nullptr,
              "the admission binary links the real executor bind symbol");
#else
  test.expect(!executor::kSm87TargetAotP40ExecutorAdmissionCompiled,
              "the default binary keeps the target executor closed");
#endif
}

}  // namespace

int main() {
  TestContext test;
  test_exact_request_boundary(test);
  test_canonical_layer_and_panel_schedule(test);
  test_final_handoff_scratch_lifetime(test);
  test_committed_handoff_is_the_only_success(test);
  test_admission_binary_links_executor(test);
  if (test.failures() != 0) {
    return 1;
  }
  std::cout << "SM87 target-AOT P40000 executor host contract checks passed\n";
  return 0;
}
