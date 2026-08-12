#include "q3x/kernels/sm87_target_aot_projection_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

using kernels::Sm87TargetAotLogicalRole;
using kernels::Sm87TargetAotProjectionEncoding;
using kernels::Sm87TargetAotProjectionPlan;
using kernels::Sm87TargetAotProjectionRole;

constexpr auto kGateP40 = kernels::sm87_target_aot_projection_plan(
    Sm87TargetAotProjectionRole::kNvFp4GateUp, 40'000U);
constexpr auto kDownP40 = kernels::sm87_target_aot_projection_plan(
    Sm87TargetAotProjectionRole::kNvFp4Down, 40'000U);
constexpr auto kGdnP40 = kernels::sm87_target_aot_projection_plan(
    Sm87TargetAotProjectionRole::kFp8GdnQkvZ, 40'000U);
constexpr auto kFullP40 = kernels::sm87_target_aot_projection_plan(
    Sm87TargetAotProjectionRole::kFp8FullQkv, 40'000U);
constexpr auto kOutputP40 = kernels::sm87_target_aot_projection_plan(
    Sm87TargetAotProjectionRole::kFp8AttentionOutput, 40'000U);

static_assert(kGateP40.valid() && kDownP40.valid() && kGdnP40.valid() &&
              kFullP40.valid() && kOutputP40.valid());
static_assert(kGateP40.encoding ==
                  Sm87TargetAotProjectionEncoding::
                      kNvFp4E2M1Block16E4M3FnScale &&
              kGateP40.input_features == 5'120U &&
              kGateP40.projected_output_features == 34'816U &&
              kGateP40.published_output_features == 17'408U &&
              kGateP40.partition_count == 2U &&
              kGateP40.grid_m == 313U && kGateP40.grid_n == 68U &&
              kGateP40.tail_rows == 64U && kGateP40.k_tiles == 80U &&
              kGateP40.logical_tasks == 21'284U &&
              kGateP40.mma_tile_tasks == 42'568U &&
              kGateP40.mma_partitions_per_task == 2U &&
              kGateP40.kernel_launches == 1U &&
              kGateP40.same_cta_partition_pair &&
              kGateP40.gate_bf16_ready_before_up_consumer &&
              kGateP40.raster_group_m == 2U && !kGateP40.stream_k &&
              kGateP40.physical_ctas == 16U &&
              kGateP40.ctas_per_sm == 1U &&
              kGateP40.split_k_workspace_bytes == 0U &&
              kGateP40.stream_k_lock_bytes == 0U &&
              !kGateP40.cuda_implementation_present &&
              !kGateP40.static_resources_qualified &&
              !kGateP40.numerical_contract_qualified &&
              !kGateP40.production_dispatch_eligible);
static_assert(kDownP40.input_features == 17'408U &&
              kDownP40.projected_output_features == 5'120U &&
              kDownP40.grid_m == 313U && kDownP40.grid_n == 20U &&
              kDownP40.k_tiles == 272U &&
              kDownP40.raster_group_m == 1U);
static_assert(kGdnP40.projected_output_features == 16'384U &&
              kGdnP40.grid_n == 64U && kGdnP40.partition_count == 2U &&
              kGdnP40.raster_group_m == 2U);
static_assert(kFullP40.projected_output_features == 14'336U &&
              kFullP40.grid_n == 56U && kFullP40.partition_count == 3U &&
              kFullP40.raster_group_m == 2U);
static_assert(kOutputP40.input_features == 6'144U &&
              kOutputP40.grid_n == 20U && kOutputP40.k_tiles == 96U &&
              kOutputP40.raster_group_m == 1U);
static_assert(kGateP40.partitions[0U].role ==
                  Sm87TargetAotLogicalRole::kNvFp4Gate &&
              kGateP40.partitions[1U].role ==
                  Sm87TargetAotLogicalRole::kNvFp4Up &&
              kGateP40.partitions[1U].output_offset == 17'408U);
static_assert(kFullP40.partitions[0U].output_features == 12'288U &&
              kFullP40.partitions[1U].output_offset == 12'288U &&
              kFullP40.partitions[2U].output_offset == 13'312U);
static_assert(
    (kGateP40.policy & kernels::kSm87TargetAotAccuracyUnqualified) != 0U &&
    (kGateP40.policy & kernels::kSm87TargetAotNoCuBlasLt) != 0U &&
    (kGateP40.policy & kernels::kSm87TargetAotFullKSingleCta) != 0U &&
    (kGateP40.policy & kernels::kSm87TargetAotCpAsyncCg) != 0U &&
    (kGateP40.policy &
     kernels::kSm87TargetAotSharedToRegisterDoubleBuffer) != 0U &&
    (kGateP40.policy &
     kernels::kSm87TargetAotPrefetchBeforeStageDrain) != 0U);
static_assert(
    (kGateP40.policy & kernels::kSm87TargetAotGateUpSameCtaPartitionPair) !=
        0U &&
    (kGateP40.policy &
     kernels::kSm87TargetAotGateBf16ReadyBeforeUpConsumer) != 0U);
static_assert(!kernels::sm87_target_aot_projection_plan(
                   Sm87TargetAotProjectionRole::kNvFp4GateUp, 39'999U)
                   .valid());
static_assert(!kernels::sm87_target_aot_projection_plan(
                   static_cast<Sm87TargetAotProjectionRole>(0xffU), 40'000U)
                   .valid());

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

[[nodiscard]] bool exhaustive_persistent_bijection(
    const Sm87TargetAotProjectionPlan& plan) {
  std::vector<std::uint8_t> visited(plan.logical_tasks, 0U);
  for (std::size_t cta = 0U;
       cta < kernels::kSm87TargetAotProjectionPersistentCtas; ++cta) {
    for (std::size_t iteration = 0U;; ++iteration) {
      const auto task = kernels::sm87_target_aot_projection_persistent_task(
          plan, cta, iteration);
      if (!task.valid) {
        break;
      }
      if (task.m_tile >= plan.grid_m || task.n_tile >= plan.grid_n) {
        return false;
      }
      if (plan.role == Sm87TargetAotProjectionRole::kNvFp4GateUp) {
        if (task.first_partition != 0U || task.partition_count != 2U) {
          return false;
        }
      } else if (task.partition_count != 1U ||
                 task.first_partition >= plan.partition_count) {
        return false;
      }
      const std::size_t canonical = task.m_tile * plan.grid_n + task.n_tile;
      if (visited[canonical] != 0U) {
        return false;
      }
      visited[canonical] = 1U;
    }
  }
  for (const std::uint8_t count : visited) {
    if (count != 1U) {
      return false;
    }
  }
  return true;
}

void test_target_buckets_and_role_separation(TestContext& test) {
  constexpr std::array<Sm87TargetAotProjectionRole, 5U> kRoles{{
      Sm87TargetAotProjectionRole::kNvFp4GateUp,
      Sm87TargetAotProjectionRole::kNvFp4Down,
      Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
      Sm87TargetAotProjectionRole::kFp8FullQkv,
      Sm87TargetAotProjectionRole::kFp8AttentionOutput,
  }};
  constexpr std::array<std::size_t, 3U> kExpectedGridM{{313U, 469U,
                                                        1'016U}};
  constexpr std::array<std::size_t, 3U> kExpectedTail{{64U, 96U, 80U}};
  for (std::size_t bucket = 0U;
       bucket < kernels::kSm87TargetAotWitnessTokenCounts.size();
       ++bucket) {
    for (const Sm87TargetAotProjectionRole role : kRoles) {
      const auto plan = kernels::sm87_target_aot_projection_plan(
          role, kernels::kSm87TargetAotWitnessTokenCounts[bucket]);
      test.expect(plan.valid(), "every target bucket has every role plan");
      test.expect(plan.grid_m == kExpectedGridM[bucket] &&
                      plan.tail_rows == kExpectedTail[bucket],
                  "target bucket owns the expected M128 tail geometry");
      test.expect(!plan.stream_k && !plan.production_dispatch_eligible,
                  "v1 is full-K DP and cannot enter production dispatch");
      test.expect(exhaustive_persistent_bijection(plan),
                  "persistent DP raster covers every output tile once");
    }
  }

  test.expect(kGateP40.tactic != kDownP40.tactic &&
                  kGdnP40.tactic != kFullP40.tactic &&
                  kFullP40.tactic != kOutputP40.tactic,
              "same base tile never collapses role-specific tactic identity");
}

void test_plan_mutation_fails_closed(TestContext& test) {
  Sm87TargetAotProjectionPlan changed = kGateP40;
  changed.raster_group_m = 1U;
  test.expect(!changed.valid(), "Gate raster mutation fails closed");
  changed = kGateP40;
  changed.stream_k = true;
  test.expect(!changed.valid(), "unproved Stream-K mutation fails closed");
  changed = kGateP40;
  changed.cp_async_cache_global = false;
  test.expect(!changed.valid(), "L1-polluting cp.async mutation fails closed");
  changed = kGateP40;
  changed.prefetch_before_stage_drain = false;
  test.expect(!changed.valid(), "producer/consumer overlap mutation fails closed");
  changed = kGateP40;
  changed.partitions[1U].independent_tensor_scale = false;
  test.expect(!changed.valid(), "merged Up scale mutation fails closed");
  changed = kGateP40;
  changed.same_cta_partition_pair = false;
  test.expect(!changed.valid(),
              "Gate/Up cannot be scheduled to unrelated persistent CTAs");
  changed = kGateP40;
  changed.gate_bf16_ready_before_up_consumer = false;
  test.expect(!changed.valid(),
              "Up must consume the Gate BF16 publication, not accumulator");
  changed = kGateP40;
  changed.production_dispatch_eligible = true;
  test.expect(!changed.valid(), "production eligibility cannot be self-declared");
}

}  // namespace

int main() {
  TestContext test;
  test_target_buckets_and_role_separation(test);
  test_plan_mutation_fails_closed(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "sm87 target AOT projection plan checks passed\n";
  return 0;
}
