#include "q3x/kernels/sm87_bulk_dataflow_v2_attention_l2_cohort.h"

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace kernels = q3x::kernels;

namespace {

constexpr auto kTargetP40AttentionPlan =
    kernels::sm87_target_aot_attention_plan(
        kernels::kSm87BulkV2AttentionTokens);

static_assert(kTargetP40AttentionPlan.valid());
static_assert(kTargetP40AttentionPlan.query_tiles_per_kv_head == 1'875U);
static_assert(kTargetP40AttentionPlan.total_ctas == 7'500U);
static_assert(kTargetP40AttentionPlan.topology ==
              kernels::Sm87TargetAotAttentionTopology::kQ128Kv32TwoStage);
static_assert(kernels::sm87_target_aot_same_attention_numerical_contract(
    kernels::kSm87BulkV2AttentionNumericalContract,
    kTargetP40AttentionPlan.numerical_execution));
static_assert(kernels::sm87_bulk_v2_attention_mapping_is_bijective());

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

[[nodiscard]] kernels::Sm87BulkV2AttentionArguments
make_arguments() noexcept {
  kernels::Sm87BulkV2AttentionArguments arguments;
  arguments.processed_query =
      fake_pointer<const std::uint16_t>(0x0000'0001'0000'0000ULL);
  arguments.processed_key =
      fake_pointer<const std::uint16_t>(0x0000'0001'2000'0000ULL);
  arguments.processed_value =
      fake_pointer<const std::uint16_t>(0x0000'0001'3000'0000ULL);
  arguments.processed_gate =
      fake_pointer<const std::uint16_t>(0x0000'0001'4000'0000ULL);
  arguments.gated_output =
      fake_pointer<std::uint16_t>(0x0000'0001'6000'0000ULL);
  arguments.token_count = kernels::kSm87BulkV2AttentionTokens;
  arguments.device_ordinal = 0;
  arguments.cuda_stream = fake_pointer<void>(0x100ULL);
  return arguments;
}

void test_mapping(TestContext& test) {
  std::size_t stores = 0U;
  std::size_t suppressed = 0U;
  for (std::size_t kv_head = 0U;
       kv_head < kernels::kSm87BulkV2AttentionKvHeads; ++kv_head) {
    for (std::size_t epoch = 0U;
         epoch < kernels::kSm87BulkV2AttentionSnakeEpochs; ++epoch) {
      for (std::size_t lane = 0U;
           lane < kernels::kSm87BulkV2AttentionPersistentLanes; ++lane) {
        const auto item = kernels::sm87_bulk_v2_attention_work_item(
            kv_head, lane, epoch);
        test.expect(item.valid, "every persistent lane owns every epoch");
        if (item.store_enabled) {
          ++stores;
        } else {
          ++suppressed;
          test.expect(
              epoch == 117U && lane < 13U && item.query_tile == 1'874U,
              "only final lanes 0..12 repeat tile 1874 without stores");
        }
      }
    }
  }
  test.expect(stores == 7'500U,
              "all four KV heads publish exactly 7500 real tiles");
  test.expect(suppressed == 52U,
              "four same-head launches suppress exactly 4x13 repeats");

  const auto even_first =
      kernels::sm87_bulk_v2_attention_work_item(0U, 0U, 0U);
  const auto even_last =
      kernels::sm87_bulk_v2_attention_work_item(0U, 15U, 0U);
  const auto odd_first =
      kernels::sm87_bulk_v2_attention_work_item(0U, 0U, 1U);
  const auto odd_last =
      kernels::sm87_bulk_v2_attention_work_item(0U, 15U, 1U);
  test.expect(even_first.query_tile == 0U &&
                  even_last.query_tile == 15U,
              "even epochs map 16e+l");
  test.expect(odd_first.query_tile == 31U &&
                  odd_last.query_tile == 16U,
              "odd epochs map 16e+15-l");

  const auto final_repeat =
      kernels::sm87_bulk_v2_attention_work_item(2U, 12U, 117U);
  const auto final_owner =
      kernels::sm87_bulk_v2_attention_work_item(2U, 13U, 117U);
  const auto final_second =
      kernels::sm87_bulk_v2_attention_work_item(2U, 14U, 117U);
  const auto final_third =
      kernels::sm87_bulk_v2_attention_work_item(2U, 15U, 117U);
  test.expect(final_repeat.query_tile == 1'874U &&
                  !final_repeat.store_enabled,
              "the last suppressed lane repeats the final real tile");
  test.expect(final_owner.query_tile == 1'874U &&
                  final_owner.store_enabled &&
                  final_second.query_tile == 1'873U &&
                  final_second.store_enabled &&
                  final_third.query_tile == 1'872U &&
                  final_third.store_enabled,
              "the final snake epoch has three unique store owners");

  test.expect(
      !kernels::sm87_bulk_v2_attention_work_item(4U, 0U, 0U).valid &&
          !kernels::sm87_bulk_v2_attention_work_item(0U, 16U, 0U).valid &&
          !kernels::sm87_bulk_v2_attention_work_item(0U, 0U, 118U).valid,
      "out-of-contract heads, lanes, and epochs fail closed");
}

void test_arguments(TestContext& test) {
  const auto arguments = make_arguments();
  test.expect(kernels::sm87_bulk_v2_attention_arguments_valid(arguments),
              "one disjoint exact-P40000 binding validates");

  auto changed = arguments;
  changed.token_count = 39'999U;
  test.expect(!kernels::sm87_bulk_v2_attention_arguments_valid(changed),
              "the first cell admits only exact P40000");
  changed = arguments;
  changed.device_ordinal = -1;
  test.expect(!kernels::sm87_bulk_v2_attention_arguments_valid(changed),
              "a concrete device owner is mandatory");
  changed = arguments;
  changed.cuda_stream = nullptr;
  test.expect(!kernels::sm87_bulk_v2_attention_arguments_valid(changed),
              "the implicit stream is not an owner identity");
  changed = arguments;
  changed.processed_query = fake_pointer<const std::uint16_t>(
      reinterpret_cast<std::uintptr_t>(arguments.processed_query) + 2U);
  test.expect(!kernels::sm87_bulk_v2_attention_arguments_valid(changed),
              "every operand retains vector alignment");
  changed = arguments;
  changed.gated_output = const_cast<std::uint16_t*>(
      arguments.processed_gate);
  test.expect(!kernels::sm87_bulk_v2_attention_arguments_valid(changed),
              "gate input and gated publication cannot alias");
  changed = arguments;
  changed.processed_value = arguments.processed_key;
  test.expect(!kernels::sm87_bulk_v2_attention_arguments_valid(changed),
              "ordered K and V spans remain disjoint");
}

[[nodiscard]] kernels::Sm87BulkV2AttentionResources
passing_resources() noexcept {
  kernels::Sm87BulkV2AttentionResources resources;
  resources.binary_version = 87;
  resources.registers_per_thread = 254;
  resources.static_shared_bytes = 0U;
  resources.dynamic_shared_bytes =
      kernels::kSm87BulkV2AttentionDynamicSharedBytes;
  resources.local_bytes = 0U;
  resources.maximum_threads_per_block = 256;
  resources.active_blocks_per_sm = 1;
  resources.device_sm_count = 16;
  resources.device_optin_shared_bytes = 163'840U;
  resources.threads_per_block = 256;
  resources.physical_grid_ctas_per_launch = 16;
  resources.physical_launches = 4;
  resources.query_tiles_per_kv_head = 1'875U;
  resources.snake_epochs = 118U;
  resources.store_disabled_bodies = 52U;
  resources.kernel_compiled = true;
  resources.exact_p40000_only = true;
  resources.same_kv_head_per_launch = true;
  resources.mapping_bijective = true;
  resources.no_cooperative_launch = true;
  resources.no_cross_cta_barrier_or_lock = true;
  resources.persistent_cta_residency_capacity = true;
  resources.resource_gate_passed = true;
  return resources;
}

void test_resources(TestContext& test) {
  kernels::Sm87BulkV2AttentionResources unavailable;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(unavailable),
              "the uncompiled default-off record is not runnable");

  const auto resources = passing_resources();
  test.expect(kernels::sm87_bulk_v2_attention_resources_valid(resources),
              "the device has capacity for all 16 one-CTA/SM lanes");

  auto changed = resources;
  changed.registers_per_thread = 256;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "more than 255 registers fails closed");
  changed = resources;
  changed.local_bytes = 8U;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "any local-memory frame fails closed");
  changed = resources;
  changed.device_sm_count = 15;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "the cell is exact 16-SM SM87 only");
  changed = resources;
  changed.active_blocks_per_sm = 0;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "zero queried CTA residency cannot provide 16-lane capacity");
  changed = resources;
  changed.persistent_cta_residency_capacity = false;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "the admission record cannot omit 16-CTA residency capacity");
  changed = resources;
  changed.physical_grid_ctas_per_launch = 64;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "one launch contains one same-KV-head 16-CTA cohort");
  changed = resources;
  changed.physical_launches = 1;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "four KV heads require four stream-ordered cohorts");
  changed = resources;
  changed.store_disabled_bodies = 13U;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "the full mapping records all four heads' suppressed stores");
  changed = resources;
  changed.mapping_bijective = false;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "resource admission cannot bypass the mapping proof");
  changed = resources;
  changed.resource_gate_passed = false;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "a failed queried hard gate remains failed");
  changed = resources;
  changed.numerical_contract_qualified = true;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "the first cell cannot forge numerical qualification");
  changed = resources;
  changed.production_dispatch_eligible = true;
  test.expect(!kernels::sm87_bulk_v2_attention_resources_valid(changed),
              "the first cell cannot claim production dispatch");
}

}  // namespace

int main() {
  TestContext test;
  test_mapping(test);
  test_arguments(test);
  test_resources(test);
  if (test.failures() != 0) {
    return 1;
  }
  std::cout << "SM87 bulk-dataflow-v2 exact P40 Attention persistent-L2-"
               "cohort host contract checks passed\n";
  return 0;
}
