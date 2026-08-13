#include "q3x/kernels/sm87_bulk_dataflow_v2_gdn_c64.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_gdn_p40_plan.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace kernels = q3x::kernels;

namespace {

static_assert(kernels::kSm87BulkV2GdnProducerCtas == 1'024U);
static_assert(kernels::kSm87BulkV2GdnProducerThreads == 128U);
static_assert(kernels::kSm87BulkV2GdnRecurrenceCtas == 48U);
static_assert(kernels::kSm87BulkV2GdnRecurrenceThreads == 256U);
static_assert(kernels::kSm87BulkV2GdnEpilogueCtas == 384U);
static_assert(kernels::kSm87BulkV2GdnEpilogueThreads == 256U);
static_assert(kernels::kSm87BulkV2GdnWorkspaceBytes == 2'646'016U);
static_assert(kernels::kSm87BulkV2GdnStateTraceBytes == 100'663'296U);
static_assert(
    (kernels::kSm87BulkV2GdnC64RequiredPolicy &
     kernels::kSm87BulkV2GdnExactPerTokenBf16State) != 0U);
static_assert(
    (kernels::kSm87BulkV2GdnC64RequiredPolicy &
     kernels::kSm87BulkV2GdnNoWyKktSsdScan) != 0U);
static_assert(
    (kernels::kSm87BulkV2GdnC64RequiredPolicy &
     kernels::kSm87BulkV2GdnNoProductionSelector) != 0U);
static_assert(
    kernels::kSm87BulkV2GdnQkContract ==
    kernels::Sm87TargetAotGdnQkNormalizationContract::
        kFp32Pair0_64Pair32_96Shuffle16To1RsqrtfQInvSqrt128);
static_assert(
    kernels::kSm87BulkV2GdnRecurrenceContract ==
    kernels::Sm87TargetAotGdnRecurrenceExecutionContract::
        kAlphaScalePredictionUpdateOutputKeyAscendingFmafPerTokenBf16);
static_assert(
    kernels::kSm87BulkV2GdnNormGateContract ==
    kernels::Sm87TargetAotGdnNormGateContract::
        kRawBf16PairShuffleRmsRsqrtfPlainWeightSiluBf16Rne);
static_assert(std::is_same_v<
              decltype(kernels::Sm87BulkV2GdnC64Arguments{}.normalized_q),
              float*>);
static_assert(std::is_same_v<
              decltype(kernels::Sm87BulkV2GdnC64Arguments{}.
                           initial_recurrent_state),
              const std::uint16_t*>);

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

template <class T>
[[nodiscard]] T* fake_pointer(const std::uintptr_t address) noexcept {
  return reinterpret_cast<T*>(address);
}

[[nodiscard]] kernels::Sm87BulkV2GdnC64Arguments make_arguments() noexcept {
  kernels::Sm87BulkV2GdnC64Arguments arguments;
  arguments.raw_qkvz =
      fake_pointer<const std::uint16_t>(0x0000'0001'0000'0000ULL);
  arguments.interleaved_ab =
      fake_pointer<const std::uint16_t>(0x0000'0001'0100'0000ULL);
  arguments.conv_weight =
      fake_pointer<const std::uint16_t>(0x0000'0001'0200'0000ULL);
  arguments.initial_conv_history =
      fake_pointer<const std::uint16_t>(0x0000'0001'0300'0000ULL);
  arguments.a_log =
      fake_pointer<const std::uint16_t>(0x0000'0001'0400'0000ULL);
  arguments.dt_bias =
      fake_pointer<const std::uint16_t>(0x0000'0001'0400'1000ULL);
  arguments.norm_weight =
      fake_pointer<const std::uint16_t>(0x0000'0001'0400'2000ULL);
  arguments.initial_recurrent_state =
      fake_pointer<const std::uint16_t>(0x0000'0001'0500'0000ULL);
  arguments.l2_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.norm_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.first_position = 0U;
  arguments.token_count = kernels::kSm87BulkV2GdnC64Tokens;
  arguments.normalized_q =
      fake_pointer<float>(0x0000'0001'0700'0000ULL);
  arguments.normalized_k =
      fake_pointer<float>(0x0000'0001'0800'0000ULL);
  arguments.prepared_v =
      fake_pointer<std::uint16_t>(0x0000'0001'0900'0000ULL);
  arguments.alpha = fake_pointer<float>(0x0000'0001'0a00'0000ULL);
  arguments.beta = fake_pointer<float>(0x0000'0001'0a10'0000ULL);
  arguments.raw_output =
      fake_pointer<std::uint16_t>(0x0000'0001'0b00'0000ULL);
  arguments.output =
      fake_pointer<std::uint16_t>(0x0000'0001'0c00'0000ULL);
  arguments.final_conv_history =
      fake_pointer<std::uint16_t>(0x0000'0001'0d00'0000ULL);
  arguments.final_recurrent_state =
      fake_pointer<std::uint16_t>(0x0000'0001'0e00'0000ULL);
  arguments.cuda_stream = fake_pointer<void>(0x0000'0000'0000'0100ULL);
  return arguments;
}

void test_arguments(TestContext& test) {
  const auto arguments = make_arguments();
  test.expect(kernels::sm87_bulk_v2_gdn_c64_arguments_valid(arguments),
              "one exact C64 cell validates");

  auto changed = arguments;
  changed.token_count = 32U;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "the first cell admits only C64");
  changed = arguments;
  changed.first_position = 64U;
  test.expect(kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "the exact C64 cell admits an aligned continuation boundary");
  changed = arguments;
  changed.first_position = 1U;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "continuation boundaries remain C64 aligned");
  changed = arguments;
  changed.first_position = 129'920U;
  test.expect(kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "the final complete C64 inside P130 validates");
  changed = arguments;
  changed.first_position = 129'984U;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "a complete C64 cannot cross the P130 capacity");
  changed = arguments;
  ++changed.l2_epsilon_fp32_bits;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "QK epsilon preserves raw FP32 bits");
  changed = arguments;
  ++changed.norm_epsilon_fp32_bits;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "RMS epsilon preserves raw FP32 bits");
  changed = arguments;
  changed.cuda_stream = nullptr;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "the default stream is not an owner identity");
  changed = arguments;
  changed.normalized_q = nullptr;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "producer Q publication is mandatory");
  changed = arguments;
  changed.normalized_k = changed.normalized_q = arguments.normalized_q;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "Q and K FP32 publications are disjoint");
  changed = arguments;
  changed.raw_output = reinterpret_cast<std::uint16_t*>(arguments.output);
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "raw BF16 and final BF16 publication remain distinct");
  changed = arguments;
  changed.final_conv_history =
      const_cast<std::uint16_t*>(arguments.initial_conv_history);
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "conv history cannot race in place across token tiles");
  changed = arguments;
  changed.final_recurrent_state =
      const_cast<std::uint16_t*>(arguments.initial_recurrent_state);
  test.expect(kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "one value-head owner may publish its C64 BF16 state in place");
  changed = arguments;
  changed.final_recurrent_state = fake_pointer<std::uint16_t>(
      reinterpret_cast<std::uintptr_t>(arguments.initial_recurrent_state) +
      16U);
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "partial state aliases are rejected");
  changed = arguments;
  changed.prepared_v = fake_pointer<std::uint16_t>(
      reinterpret_cast<std::uintptr_t>(arguments.prepared_v) + 2U);
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(changed),
              "workspace bindings retain 16-byte alignment");

  const auto trace_range = kernels::sm87_bulk_v2_gdn_byte_range(
      fake_pointer<std::uint16_t>(0x0000'0002'0000'0000ULL),
      kernels::kSm87BulkV2GdnStateTraceBytes);
  test.expect(trace_range.valid &&
                  trace_range.end - trace_range.begin ==
                      kernels::kSm87BulkV2GdnStateTraceBytes,
              "correctness trace covers every token/head state word");
  test.expect(kernels::sm87_bulk_v2_gdn_c64_state_trace_valid(
                  arguments,
                  fake_pointer<std::uint16_t>(0x0000'0002'0000'0000ULL)),
              "a disjoint aligned 96-MiB state trace validates");
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_state_trace_valid(
                  arguments,
                  const_cast<std::uint16_t*>(
                      arguments.initial_recurrent_state)),
              "the state trace cannot alias the incoming authoritative state");
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_state_trace_valid(
                  arguments, arguments.output),
              "the state trace cannot overwrite final output");
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_state_trace_valid(
                  arguments,
                  fake_pointer<std::uint16_t>(0x0000'0002'0000'0002ULL)),
              "the state trace retains 16-byte alignment");
}

[[nodiscard]] kernels::Sm87BulkV2GdnKernelResources passing_kernel(
    const int threads, const int grid) noexcept {
  kernels::Sm87BulkV2GdnKernelResources resources;
  resources.registers_per_thread = 64;
  resources.static_shared_bytes = 34'056U;
  resources.local_bytes = 0U;
  resources.maximum_threads_per_block = 1'024;
  resources.active_blocks_per_sm = 4;
  resources.threads_per_block = threads;
  resources.physical_grid_ctas = grid;
  return resources;
}

[[nodiscard]] kernels::Sm87BulkV2GdnC64Resources passing_resources() noexcept {
  kernels::Sm87BulkV2GdnC64Resources resources;
  resources.binary_version = 87;
  resources.producer = passing_kernel(
      static_cast<int>(kernels::kSm87BulkV2GdnProducerThreads),
      static_cast<int>(kernels::kSm87BulkV2GdnProducerCtas));
  resources.recurrence = passing_kernel(
      static_cast<int>(kernels::kSm87BulkV2GdnRecurrenceThreads),
      static_cast<int>(kernels::kSm87BulkV2GdnRecurrenceCtas));
  resources.epilogue = passing_kernel(
      static_cast<int>(kernels::kSm87BulkV2GdnEpilogueThreads),
      static_cast<int>(kernels::kSm87BulkV2GdnEpilogueCtas));
  resources.kernels_compiled = true;
  resources.exact_geometry = true;
  resources.resource_gate_passed = true;
  return resources;
}

void test_resources(TestContext& test) {
  kernels::Sm87BulkV2GdnC64Resources unavailable;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_resources_valid(unavailable),
              "an uncompiled default-off record is not runnable");

  const auto resources = passing_resources();
  test.expect(kernels::sm87_bulk_v2_gdn_c64_resources_valid(resources),
              "all three kernels pass the frozen resource and geometry gate");

  auto changed = resources;
  changed.recurrence.registers_per_thread = 86;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_resources_valid(changed),
              "86 recurrence registers fail closed");
  changed = resources;
  changed.recurrence.local_bytes = 4U;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_resources_valid(changed),
              "any recurrence local memory fails closed");
  changed = resources;
  changed.recurrence.active_blocks_per_sm = 2;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_resources_valid(changed),
              "two recurrence CTAs per SM cannot cover 48 heads in one wave");
  changed = resources;
  changed.producer.physical_grid_ctas = 512;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_resources_valid(changed),
              "producer geometry is exactly 1024x128");
  changed = resources;
  changed.epilogue.physical_grid_ctas = 48;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_resources_valid(changed),
              "epilogue geometry is exactly 384x256");
  changed = resources;
  changed.resource_gate_passed = false;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_resources_valid(changed),
              "a query cannot suppress a failed hard gate");
  changed = resources;
  changed.numerical_contract_qualified = true;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_resources_valid(changed),
              "the first cell cannot forge completed numerical qualification");
  changed = resources;
  changed.production_dispatch_eligible = true;
  test.expect(!kernels::sm87_bulk_v2_gdn_c64_resources_valid(changed),
              "the first cell can never claim production dispatch");
}

void test_p40_plan(TestContext& test) {
  constexpr auto plan = kernels::sm87_bulk_v2_gdn_p40_plan();
  static_assert(plan.valid());
  static_assert(plan.chunk_count == 625U && plan.tail_tokens == 0U);
  static_assert(plan.reusable_events == 6U);
  static_assert(plan.private_bytes == 6'987'776U);

  constexpr auto first = kernels::sm87_bulk_v2_gdn_p40_chunk(plan, 0U);
  constexpr auto second = kernels::sm87_bulk_v2_gdn_p40_chunk(plan, 1U);
  constexpr auto third = kernels::sm87_bulk_v2_gdn_p40_chunk(plan, 2U);
  constexpr auto last = kernels::sm87_bulk_v2_gdn_p40_chunk(plan, 624U);
  static_assert(first.valid(plan) && second.valid(plan) && third.valid(plan) &&
                last.valid(plan));
  test.expect(first.first_position == 0U &&
                  first.input_history_is_request_initial &&
                  !first.producer_waits_for_recurrence_slot &&
                  !first.recurrence_waits_for_epilogue_slot,
              "the first C64 consumes request-initial history without reuse waits");
  test.expect(second.first_position == 64U &&
                  second.input_history_slot == 0U &&
                  second.output_history_slot == 1U &&
                  !second.producer_waits_for_recurrence_slot,
              "the second C64 consumes the first producer history in stream order");
  test.expect(third.prepared_slot == 0U && third.raw_output_slot == 0U &&
                  third.producer_waits_for_recurrence_slot &&
                  third.recurrence_waits_for_epilogue_slot,
              "the third C64 waits before reusing both bounded slots");
  test.expect(last.first_position + last.token_count ==
                      kernels::kSm87BulkV2GdnP40Tokens &&
                  last.output_history_slot == 0U,
              "the final C64 covers P40 exactly without a tail");
  test.expect(!kernels::sm87_bulk_v2_gdn_p40_chunk(plan, 625U).valid(plan),
              "the P40 plan rejects an out-of-range chunk");
}

}  // namespace

int main() {
  TestContext test;
  test_arguments(test);
  test_resources(test);
  test_p40_plan(test);
  if (test.failures() != 0) {
    return 1;
  }
  std::cout << "SM87 bulk-dataflow-v2 exact single-C64 GDN host contract "
               "checks passed\n";
  return 0;
}
