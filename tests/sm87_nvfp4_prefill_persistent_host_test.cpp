#include "q3x/kernels/sm87_nvfp4_prefill_persistent.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

namespace {

using q3x::kernels::Sm87NvFp4PersistentPrefillRaster;
using q3x::kernels::Sm87NvFp4PersistentPrefillRole;
using q3x::kernels::Sm87NvFp4PersistentPrefillTailPolicy;
using q3x::kernels::sm87_nvfp4_persistent_prefill_plan;
using q3x::kernels::sm87_nvfp4_persistent_prefill_shape_contract;
using q3x::kernels::sm87_nvfp4_persistent_prefill_task;

constexpr auto kGate = sm87_nvfp4_persistent_prefill_plan(
    Sm87NvFp4PersistentPrefillRole::kGateUpPaired, 40'000U);
constexpr auto kDown = sm87_nvfp4_persistent_prefill_plan(
    Sm87NvFp4PersistentPrefillRole::kDown, 40'000U);
constexpr auto kP60 =
    sm87_nvfp4_persistent_prefill_shape_contract(60'000U);
constexpr auto kM5424 =
    sm87_nvfp4_persistent_prefill_shape_contract(5'424U);

static_assert(kGate.valid() && kDown.valid());
static_assert(kGate.raster ==
              Sm87NvFp4PersistentPrefillRaster::kGateUpGroupedM4NMajor);
static_assert(kDown.raster ==
              Sm87NvFp4PersistentPrefillRaster::kDownBStationaryN256);
static_assert(kGate.m_tiles == 625U && kGate.n_tiles == 136U &&
              kGate.task_count == 85'000U);
static_assert(kDown.m_tiles == 625U && kDown.n_tiles == 20U &&
              kDown.task_count == 12'500U);
static_assert(kGate.persistent_ctas == 16U &&
              kDown.persistent_ctas == 16U);
static_assert(kGate.tasks_per_cta_upper_bound == 5'313U);
static_assert(kDown.tasks_per_cta_upper_bound == 782U);
static_assert(kGate.tile_m == 64U && kGate.tile_n == 256U &&
              kGate.tile_k == 64U && kGate.pipeline_stages == 4U);
static_assert(!kGate.split_k && !kGate.stream_k &&
              kGate.fp32_accumulation &&
              kGate.bf16_publication_boundary);
static_assert(kGate.requires_interleaved_gate_up_sidecar &&
              !kGate.fused_down_residual);
static_assert(!kDown.requires_interleaved_gate_up_sidecar &&
              kDown.fused_down_residual);

static_assert(kP60.valid() && !kP60.admitted &&
              kP60.aligned_prefix_tokens == 59'968U &&
              kP60.tail_tokens == 32U &&
              kP60.tail_policy ==
                  Sm87NvFp4PersistentPrefillTailPolicy::
                      kNativeM32CompanionRequired);
static_assert(kM5424.valid() && !kM5424.admitted &&
              kM5424.aligned_prefix_tokens == 5'376U &&
              kM5424.tail_tokens == 48U &&
              kM5424.tail_policy ==
                  Sm87NvFp4PersistentPrefillTailPolicy::
                      kGeneralTailCompanionRequired);

static_assert(q3x::kernels::
                  kSm87NvFp4PersistentPrefillGateOneCohortFullSlabBytes ==
              3'358'720U);
static_assert(q3x::kernels::
                  kSm87NvFp4PersistentPrefillGateConcurrentFullSlabBytes ==
              5'570'560U);
static_assert(q3x::kernels::
                  kSm87NvFp4PersistentPrefillGateConcurrentFullSlabBytes >
              q3x::kernels::kSm87NvFp4PersistentPrefillL2BudgetBytes);
static_assert(q3x::kernels::
                  kSm87NvFp4PersistentPrefillGateConcurrentK64StageL2Bytes ==
              69'632U);
static_assert(q3x::kernels::
                  kSm87NvFp4PersistentPrefillGateFourStageL2Bytes ==
              278'528U);
static_assert(q3x::kernels::
                  kSm87NvFp4PersistentPrefillDownFourStageL2Bytes ==
              561'152U);
static_assert(q3x::kernels::
                  kSm87NvFp4PersistentPrefillGateMaximumLocalBytes == 32U);
static_assert(q3x::kernels::
                  kSm87NvFp4PersistentPrefillDownMaximumLocalBytes == 16U);

static_assert(sm87_nvfp4_persistent_prefill_task(kGate, 0U).m_tile == 0U &&
              sm87_nvfp4_persistent_prefill_task(kGate, 0U).n_tile == 0U);
static_assert(sm87_nvfp4_persistent_prefill_task(kGate, 3U).m_tile == 3U &&
              sm87_nvfp4_persistent_prefill_task(kGate, 3U).n_tile == 0U);
static_assert(sm87_nvfp4_persistent_prefill_task(kGate, 4U).m_tile == 0U &&
              sm87_nvfp4_persistent_prefill_task(kGate, 4U).n_tile == 1U);
static_assert(sm87_nvfp4_persistent_prefill_task(kGate, 543U).m_tile == 3U &&
              sm87_nvfp4_persistent_prefill_task(kGate, 543U).n_tile == 135U);
static_assert(sm87_nvfp4_persistent_prefill_task(kGate, 544U).m_tile == 4U &&
              sm87_nvfp4_persistent_prefill_task(kGate, 544U).n_tile == 0U);
static_assert(
    sm87_nvfp4_persistent_prefill_task(kGate, 84'864U).m_tile == 624U &&
    sm87_nvfp4_persistent_prefill_task(kGate, 84'864U).n_tile == 0U);
static_assert(
    sm87_nvfp4_persistent_prefill_task(kGate, 84'999U).m_tile == 624U &&
    sm87_nvfp4_persistent_prefill_task(kGate, 84'999U).n_tile == 135U);

static_assert(sm87_nvfp4_persistent_prefill_task(kDown, 0U).m_tile == 0U &&
              sm87_nvfp4_persistent_prefill_task(kDown, 0U).n_tile == 0U);
static_assert(sm87_nvfp4_persistent_prefill_task(kDown, 624U).m_tile == 624U &&
              sm87_nvfp4_persistent_prefill_task(kDown, 624U).n_tile == 0U);
static_assert(sm87_nvfp4_persistent_prefill_task(kDown, 625U).m_tile == 0U &&
              sm87_nvfp4_persistent_prefill_task(kDown, 625U).n_tile == 1U);
static_assert(
    sm87_nvfp4_persistent_prefill_task(kDown, 12'499U).m_tile == 624U &&
    sm87_nvfp4_persistent_prefill_task(kDown, 12'499U).n_tile == 19U);

using GateLauncher = int (*)(const std::uint16_t*, const std::uint8_t*,
                             const std::uint8_t*, const float*, std::size_t,
                             std::uint16_t*, void*) noexcept;
using DownLauncher = int (*)(const std::uint16_t*, const std::uint8_t*,
                             const std::uint8_t*, const float*, std::size_t,
                             std::uint16_t*, void*) noexcept;
using CapabilityQuery = int (*)(Sm87NvFp4PersistentPrefillRole, std::size_t,
                                q3x::kernels::
                                    Sm87NvFp4PersistentPrefillCapability*)
    noexcept;
using ResourceQuery = int (*)(Sm87NvFp4PersistentPrefillRole, std::size_t,
                              q3x::kernels::
                                  Sm87NvFp4PersistentPrefillResources*)
    noexcept;
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_nvfp4_persistent_prefill_gate_up_cuda),
              GateLauncher>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_nvfp4_persistent_prefill_down_residual_cuda),
              DownLauncher>);

// Volatile namespace-scope initializers retain relocations to every optional
// CUDA entry point. The host test therefore fails at link time if CMake omits
// the persistent source inventory, without launching or querying the GPU.
[[maybe_unused]] CapabilityQuery volatile kLinkedCapabilityQuery =
    &q3x::kernels::query_sm87_nvfp4_persistent_prefill_capability_cuda;
[[maybe_unused]] ResourceQuery volatile kLinkedResourceQuery =
    &q3x::kernels::query_sm87_nvfp4_persistent_prefill_resources_cuda;
[[maybe_unused]] GateLauncher volatile kLinkedGateLauncher =
    &q3x::kernels::launch_sm87_nvfp4_persistent_prefill_gate_up_cuda;
[[maybe_unused]] DownLauncher volatile kLinkedDownLauncher =
    &q3x::kernels::launch_sm87_nvfp4_persistent_prefill_down_residual_cuda;

[[nodiscard]] bool check(const bool condition, const char* const message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

[[nodiscard]] bool exhaustive_bijection(
    const q3x::kernels::Sm87NvFp4PersistentPrefillPlan& plan) {
  std::vector<std::uint8_t> visited(plan.task_count, 0U);
  for (std::size_t linear = 0U; linear < plan.task_count; ++linear) {
    const auto task = sm87_nvfp4_persistent_prefill_task(plan, linear);
    if (!task.valid || task.m_tile >= plan.m_tiles ||
        task.n_tile >= plan.n_tiles) {
      return false;
    }
    const std::size_t physical = task.m_tile * plan.n_tiles + task.n_tile;
    if (visited[physical] != 0U) {
      return false;
    }
    visited[physical] = 1U;
  }
  for (const auto count : visited) {
    if (count != 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool persistent_cta_coverage(
    const q3x::kernels::Sm87NvFp4PersistentPrefillPlan& plan) {
  std::vector<std::uint8_t> visited(plan.task_count, 0U);
  for (std::size_t cta = 0U; cta < plan.persistent_ctas; ++cta) {
    std::size_t owned = 0U;
    for (std::size_t linear = cta; linear < plan.task_count;
         linear += plan.persistent_ctas) {
      ++visited[linear];
      ++owned;
    }
    if (owned > plan.tasks_per_cta_upper_bound) {
      return false;
    }
  }
  for (const auto count : visited) {
    if (count != 1U) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= check(exhaustive_bijection(kGate),
              "Gate grouped-M4 raster must be a complete bijection");
  ok &= check(exhaustive_bijection(kDown),
              "Down B-stationary raster must be a complete bijection");
  ok &= check(persistent_cta_coverage(kGate),
              "Gate persistent CTA ownership must cover every task once");
  ok &= check(persistent_cta_coverage(kDown),
              "Down persistent CTA ownership must cover every task once");
  ok &= check(!sm87_nvfp4_persistent_prefill_plan(
                   Sm87NvFp4PersistentPrefillRole::kGateUpPaired, 60'000U)
                   .valid(),
              "P60 must fail closed until the M32 companion exists");
  ok &= check(!sm87_nvfp4_persistent_prefill_plan(
                   Sm87NvFp4PersistentPrefillRole::kDown, 5'424U)
                   .valid(),
              "M5424 must fail closed until the general tail exists");
  return ok ? 0 : 1;
}
