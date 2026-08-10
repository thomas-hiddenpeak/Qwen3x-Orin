#include "q3x/kernels/sm87_p40_projection_reset.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

namespace {

using q3x::kernels::Sm87P40ProjectionResetNvFp4Plan;
using q3x::kernels::Sm87P40ProjectionResetNvFp4Role;
using q3x::kernels::Sm87P40ProjectionResetFp8Plan;
using q3x::kernels::Sm87P40ProjectionResetFp8Role;
using q3x::kernels::Sm87P40ProjectionResetRaster;
using q3x::kernels::sm87_p40_projection_reset_fp8_plan;
using q3x::kernels::sm87_p40_projection_reset_fp8_task;
using q3x::kernels::sm87_p40_projection_reset_nvfp4_plan;
using q3x::kernels::sm87_p40_projection_reset_nvfp4_task;

constexpr auto kGate = sm87_p40_projection_reset_nvfp4_plan(
    Sm87P40ProjectionResetNvFp4Role::kInterleavedGateUpSilu, 40'000U);
constexpr auto kDown = sm87_p40_projection_reset_nvfp4_plan(
    Sm87P40ProjectionResetNvFp4Role::kDownResidual, 40'000U);
constexpr auto kLinearFp8 = sm87_p40_projection_reset_fp8_plan(
    Sm87P40ProjectionResetFp8Role::kLinearQkvZInput, 40'000U);
constexpr auto kFullFp8 = sm87_p40_projection_reset_fp8_plan(
    Sm87P40ProjectionResetFp8Role::kFullQkvInput, 40'000U);
constexpr auto kOutputFp8 = sm87_p40_projection_reset_fp8_plan(
    Sm87P40ProjectionResetFp8Role::kAttentionOutput, 40'000U);

static_assert(kGate.valid() && kDown.valid());
static_assert(kLinearFp8.valid() && kFullFp8.valid() && kOutputFp8.valid());
static_assert(kGate.raster ==
              Sm87P40ProjectionResetRaster::kGateUpGroupedM4NMajor);
static_assert(kDown.raster ==
              Sm87P40ProjectionResetRaster::kDownAMajorNFast);
static_assert(kGate.input_features == 5'120U &&
              kGate.packed_output_features == 34'816U &&
              kGate.published_output_features == 17'408U);
static_assert(kDown.input_features == 17'408U &&
              kDown.packed_output_features == 5'120U &&
              kDown.published_output_features == 5'120U);
static_assert(kGate.grid_m == 625U && kGate.grid_n == 272U &&
              kGate.logical_tasks == 170'000U);
static_assert(kDown.grid_m == 625U && kDown.grid_n == 40U &&
              kDown.logical_tasks == 25'000U);
static_assert(kGate.fused_silu && !kGate.fused_residual);
static_assert(!kDown.fused_silu && kDown.fused_residual);
static_assert(kLinearFp8.input_features == 5'120U &&
              kLinearFp8.partition_count == 2U &&
              kLinearFp8.total_output_features == 16'384U &&
              kLinearFp8.grid_m == 313U && kLinearFp8.grid_n == 64U &&
              kLinearFp8.group_m == 2U &&
              kLinearFp8.logical_tasks == 20'032U);
static_assert(kFullFp8.input_features == 5'120U &&
              kFullFp8.partition_count == 3U &&
              kFullFp8.total_output_features == 14'336U &&
              kFullFp8.grid_m == 313U && kFullFp8.grid_n == 56U &&
              kFullFp8.group_m == 2U &&
              kFullFp8.logical_tasks == 17'528U);
static_assert(kOutputFp8.input_features == 6'144U &&
              kOutputFp8.partition_count == 1U &&
              kOutputFp8.total_output_features == 5'120U &&
              kOutputFp8.grid_m == 313U && kOutputFp8.grid_n == 20U &&
              kOutputFp8.group_m == 1U &&
              kOutputFp8.logical_tasks == 6'260U);
static_assert(kLinearFp8.raster ==
              Sm87P40ProjectionResetRaster::kFp8InputGroupedM2NMajor);
static_assert(kOutputFp8.raster ==
              Sm87P40ProjectionResetRaster::kFp8OutputAMajorNFast);

static_assert(q3x::kernels::kSm87P40ProjectionResetTileM == 64U);
static_assert(q3x::kernels::kSm87P40ProjectionResetTileN == 128U);
static_assert(q3x::kernels::kSm87P40ProjectionResetTileK == 32U);
static_assert(q3x::kernels::kSm87P40ProjectionResetThreads == 128U);
static_assert(q3x::kernels::kSm87P40ProjectionResetWarps == 4U);
static_assert(q3x::kernels::kSm87P40ProjectionResetPipelineStages == 3U);
static_assert(q3x::kernels::kSm87P40ProjectionResetPersistentCtas == 32U);
static_assert(q3x::kernels::kSm87P40ProjectionResetPersistentCtas ==
              2U * q3x::kernels::kSm87P40ProjectionResetSmCount);
static_assert(q3x::kernels::kSm87P40ProjectionResetDynamicSharedBytes ==
              19'456U);
static_assert(q3x::kernels::kSm87P40ProjectionResetFp8TileM == 128U);
static_assert(q3x::kernels::kSm87P40ProjectionResetFp8TileN == 256U);
static_assert(q3x::kernels::kSm87P40ProjectionResetFp8TileK == 64U);
static_assert(q3x::kernels::kSm87P40ProjectionResetFp8Threads == 256U);
static_assert(q3x::kernels::kSm87P40ProjectionResetFp8PersistentCtas == 16U);
static_assert(q3x::kernels::kSm87P40ProjectionResetFp8DynamicSharedBytes ==
              98'304U);

static_assert(sm87_p40_projection_reset_nvfp4_task(kGate, 0U).m_tile == 0U &&
              sm87_p40_projection_reset_nvfp4_task(kGate, 0U).n_tile == 0U);
static_assert(sm87_p40_projection_reset_nvfp4_task(kGate, 3U).m_tile == 3U &&
              sm87_p40_projection_reset_nvfp4_task(kGate, 3U).n_tile == 0U);
static_assert(sm87_p40_projection_reset_nvfp4_task(kGate, 4U).m_tile == 0U &&
              sm87_p40_projection_reset_nvfp4_task(kGate, 4U).n_tile == 1U);
static_assert(sm87_p40_projection_reset_nvfp4_task(kGate, 1'087U).m_tile ==
                  3U &&
              sm87_p40_projection_reset_nvfp4_task(kGate, 1'087U).n_tile ==
                  271U);
static_assert(sm87_p40_projection_reset_nvfp4_task(kGate, 1'088U).m_tile ==
                  4U &&
              sm87_p40_projection_reset_nvfp4_task(kGate, 1'088U).n_tile ==
                  0U);
static_assert(sm87_p40_projection_reset_nvfp4_task(kGate, 169'999U).m_tile ==
                  624U &&
              sm87_p40_projection_reset_nvfp4_task(kGate, 169'999U).n_tile ==
                  271U);
static_assert(sm87_p40_projection_reset_nvfp4_task(kDown, 39U).m_tile == 0U &&
              sm87_p40_projection_reset_nvfp4_task(kDown, 39U).n_tile == 39U);
static_assert(sm87_p40_projection_reset_nvfp4_task(kDown, 40U).m_tile == 1U &&
              sm87_p40_projection_reset_nvfp4_task(kDown, 40U).n_tile == 0U);
static_assert(sm87_p40_projection_reset_nvfp4_task(kDown, 24'999U).m_tile ==
                  624U &&
              sm87_p40_projection_reset_nvfp4_task(kDown, 24'999U).n_tile ==
                  39U);
static_assert(!sm87_p40_projection_reset_nvfp4_task(
                   kGate, kGate.logical_tasks)
                   .valid);
static_assert(sm87_p40_projection_reset_fp8_task(kLinearFp8, 0U).m_tile ==
                  0U &&
              sm87_p40_projection_reset_fp8_task(kLinearFp8, 0U).n_tile ==
                  0U);
static_assert(sm87_p40_projection_reset_fp8_task(kLinearFp8, 1U).m_tile ==
                  1U &&
              sm87_p40_projection_reset_fp8_task(kLinearFp8, 1U).n_tile ==
                  0U);
static_assert(sm87_p40_projection_reset_fp8_task(kLinearFp8, 2U).m_tile ==
                  0U &&
              sm87_p40_projection_reset_fp8_task(kLinearFp8, 2U).n_tile ==
                  1U);
static_assert(sm87_p40_projection_reset_fp8_task(kLinearFp8, 127U).m_tile ==
                  1U &&
              sm87_p40_projection_reset_fp8_task(kLinearFp8, 127U).n_tile ==
                  63U);
static_assert(sm87_p40_projection_reset_fp8_task(kLinearFp8, 20'031U)
                      .m_tile == 312U &&
              sm87_p40_projection_reset_fp8_task(kLinearFp8, 20'031U)
                      .n_tile == 63U);
static_assert(sm87_p40_projection_reset_fp8_task(kOutputFp8, 19U).m_tile ==
                  0U &&
              sm87_p40_projection_reset_fp8_task(kOutputFp8, 19U).n_tile ==
                  19U);
static_assert(sm87_p40_projection_reset_fp8_task(kOutputFp8, 20U).m_tile ==
                  1U &&
              sm87_p40_projection_reset_fp8_task(kOutputFp8, 20U).n_tile ==
                  0U);

using GateLauncher = int (*)(const std::uint16_t*, const std::uint8_t*,
                             const std::uint8_t*, const float*, std::size_t,
                             std::uint16_t*, void*) noexcept;
using DownLauncher = GateLauncher;
using CapabilityQuery = int (*)(
    Sm87P40ProjectionResetNvFp4Role, std::size_t,
    q3x::kernels::Sm87P40ProjectionResetCapability*) noexcept;
using ResourceQuery = int (*)(
    Sm87P40ProjectionResetNvFp4Role, std::size_t,
    q3x::kernels::Sm87P40ProjectionResetResources*) noexcept;
using Fp8Launcher = int (*)(
    const q3x::kernels::Sm87Fp8PrefillSupermatrixPartition*, std::size_t,
    const std::uint16_t*, std::size_t, std::size_t, void*) noexcept;
using Fp8CapabilityQuery = int (*)(
    Sm87P40ProjectionResetFp8Role, std::size_t,
    q3x::kernels::Sm87P40ProjectionResetFp8Capability*) noexcept;
using Fp8ResourceQuery = int (*)(
    Sm87P40ProjectionResetFp8Role, std::size_t,
    q3x::kernels::Sm87P40ProjectionResetResources*) noexcept;

static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_p40_projection_reset_nvfp4_gate_up_silu_cuda),
              GateLauncher>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_p40_projection_reset_nvfp4_down_residual_cuda),
              DownLauncher>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_p40_projection_reset_fp8_supermatrix_cuda),
              Fp8Launcher>);

// Retain relocations to the optional CUDA inventory so an incomplete CMake
// admission fails at link time without querying or launching the GPU.
[[maybe_unused]] CapabilityQuery volatile kLinkedCapabilityQuery =
    &q3x::kernels::
        query_sm87_p40_projection_reset_nvfp4_capability_cuda;
[[maybe_unused]] ResourceQuery volatile kLinkedResourceQuery =
    &q3x::kernels::query_sm87_p40_projection_reset_nvfp4_resources_cuda;
[[maybe_unused]] GateLauncher volatile kLinkedGateLauncher =
    &q3x::kernels::
        launch_sm87_p40_projection_reset_nvfp4_gate_up_silu_cuda;
[[maybe_unused]] DownLauncher volatile kLinkedDownLauncher =
    &q3x::kernels::
        launch_sm87_p40_projection_reset_nvfp4_down_residual_cuda;
[[maybe_unused]] Fp8CapabilityQuery volatile kLinkedFp8CapabilityQuery =
    &q3x::kernels::query_sm87_p40_projection_reset_fp8_capability_cuda;
[[maybe_unused]] Fp8ResourceQuery volatile kLinkedFp8ResourceQuery =
    &q3x::kernels::query_sm87_p40_projection_reset_fp8_resources_cuda;
[[maybe_unused]] Fp8Launcher volatile kLinkedFp8Launcher =
    &q3x::kernels::
        launch_sm87_p40_projection_reset_fp8_supermatrix_cuda;

[[nodiscard]] bool check(const bool condition, const char* const message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

[[nodiscard]] bool exhaustive_bijection(
    const Sm87P40ProjectionResetNvFp4Plan& plan) {
  std::vector<std::uint8_t> visited(plan.logical_tasks, 0U);
  for (std::size_t linear = 0U; linear < plan.logical_tasks; ++linear) {
    const auto task = sm87_p40_projection_reset_nvfp4_task(plan, linear);
    if (!task.valid || task.m_tile >= plan.grid_m ||
        task.n_tile >= plan.grid_n) {
      return false;
    }
    const std::size_t canonical = task.m_tile * plan.grid_n + task.n_tile;
    if (visited[canonical] != 0U) {
      return false;
    }
    visited[canonical] = 1U;
  }
  for (const auto count : visited) {
    if (count != 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool exhaustive_fp8_bijection(
    const Sm87P40ProjectionResetFp8Plan& plan) {
  std::vector<std::uint8_t> visited(plan.logical_tasks, 0U);
  for (std::size_t linear = 0U; linear < plan.logical_tasks; ++linear) {
    const auto task = sm87_p40_projection_reset_fp8_task(plan, linear);
    if (!task.valid || task.m_tile >= plan.grid_m ||
        task.n_tile >= plan.grid_n) {
      return false;
    }
    const std::size_t canonical = task.m_tile * plan.grid_n + task.n_tile;
    if (visited[canonical] != 0U) {
      return false;
    }
    visited[canonical] = 1U;
  }
  for (const auto count : visited) {
    if (count != 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool persistent_coverage(
    const Sm87P40ProjectionResetNvFp4Plan& plan) {
  std::vector<std::uint8_t> visited(plan.logical_tasks, 0U);
  for (std::size_t cta = 0U;
       cta < q3x::kernels::kSm87P40ProjectionResetPersistentCtas; ++cta) {
    for (std::size_t task = cta; task < plan.logical_tasks;
         task += q3x::kernels::kSm87P40ProjectionResetPersistentCtas) {
      ++visited[task];
    }
  }
  for (const auto count : visited) {
    if (count != 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool fp8_persistent_coverage(
    const Sm87P40ProjectionResetFp8Plan& plan) {
  std::vector<std::uint8_t> visited(plan.logical_tasks, 0U);
  for (std::size_t cta = 0U;
       cta < q3x::kernels::kSm87P40ProjectionResetFp8PersistentCtas;
       ++cta) {
    for (std::size_t task = cta; task < plan.logical_tasks;
         task += q3x::kernels::kSm87P40ProjectionResetFp8PersistentCtas) {
      ++visited[task];
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
              "Gate A-major raster must be a complete bijection");
  ok &= check(exhaustive_bijection(kDown),
              "Down B-major raster must be a complete bijection");
  ok &= check(persistent_coverage(kGate),
              "Gate persistent CTAs must cover every task exactly once");
  ok &= check(persistent_coverage(kDown),
              "Down persistent CTAs must cover every task exactly once");
  ok &= check(exhaustive_fp8_bijection(kLinearFp8),
              "Linear FP8 M2 raster must be a complete bijection");
  ok &= check(exhaustive_fp8_bijection(kFullFp8),
              "Full FP8 M2 raster must be a complete bijection");
  ok &= check(exhaustive_fp8_bijection(kOutputFp8),
              "Output FP8 M1 raster must be a complete bijection");
  ok &= check(fp8_persistent_coverage(kLinearFp8) &&
                  fp8_persistent_coverage(kFullFp8) &&
                  fp8_persistent_coverage(kOutputFp8),
              "FP8 persistent CTAs must cover every task exactly once");
  ok &= check(!sm87_p40_projection_reset_nvfp4_plan(
                   Sm87P40ProjectionResetNvFp4Role::kInterleavedGateUpSilu,
                   39'999U)
                   .valid(),
              "P39999 must fail closed");
  ok &= check(!sm87_p40_projection_reset_nvfp4_plan(
                   Sm87P40ProjectionResetNvFp4Role::kDownResidual, 40'001U)
                   .valid(),
              "P40001 must fail closed");
  ok &= check(!sm87_p40_projection_reset_nvfp4_plan(
                   Sm87P40ProjectionResetNvFp4Role::kDownResidual, 60'000U)
                   .valid(),
              "P60000 must fail closed until its native tail is implemented");
  ok &= check(!sm87_p40_projection_reset_fp8_plan(
                   Sm87P40ProjectionResetFp8Role::kLinearQkvZInput, 60'000U)
                   .valid(),
              "FP8 P60000 must fail closed until its native plan exists");
  ok &= check(!sm87_p40_projection_reset_nvfp4_plan(
                   static_cast<Sm87P40ProjectionResetNvFp4Role>(0xffU),
                   40'000U)
                   .valid(),
              "Unknown roles must fail closed");
  return ok ? 0 : 1;
}
