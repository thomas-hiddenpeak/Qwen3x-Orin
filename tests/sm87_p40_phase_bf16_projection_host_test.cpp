#include "q3x/kernels/sm87_p40_phase_bf16_projection.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {

using q3x::kernels::Sm87P40PhaseBf16ProjectionPlan;
using q3x::kernels::Sm87P40PhaseBf16ProjectionResources;
using q3x::kernels::Sm87P40PhaseBf16ProjectionRole;
using q3x::kernels::make_sm87_p40_phase_bf16_projection_plan;
using q3x::kernels::sm87_p40_phase_bf16_projection_tile;

constexpr auto kGateOrUp = make_sm87_p40_phase_bf16_projection_plan(
    Sm87P40PhaseBf16ProjectionRole::kGateOrUpK5120N17408, 40'000U);
constexpr auto kDown = make_sm87_p40_phase_bf16_projection_plan(
    Sm87P40PhaseBf16ProjectionRole::kDownK17408N5120, 40'000U);
constexpr auto kFp8N1024 = make_sm87_p40_phase_bf16_projection_plan(
    Sm87P40PhaseBf16ProjectionRole::kFp8K5120N1024, 40'000U);
constexpr auto kFp8N5120 = make_sm87_p40_phase_bf16_projection_plan(
    Sm87P40PhaseBf16ProjectionRole::kFp8K5120N5120, 40'000U);
constexpr auto kFp8N6144 = make_sm87_p40_phase_bf16_projection_plan(
    Sm87P40PhaseBf16ProjectionRole::kFp8K5120N6144, 40'000U);
constexpr auto kFp8N10240 = make_sm87_p40_phase_bf16_projection_plan(
    Sm87P40PhaseBf16ProjectionRole::kFp8K5120N10240, 40'000U);
constexpr auto kFp8N12288 = make_sm87_p40_phase_bf16_projection_plan(
    Sm87P40PhaseBf16ProjectionRole::kFp8K5120N12288, 40'000U);
constexpr auto kFp8Output = make_sm87_p40_phase_bf16_projection_plan(
    Sm87P40PhaseBf16ProjectionRole::kFp8K6144N5120, 40'000U);

static_assert(kGateOrUp.valid() && kDown.valid());
static_assert(kFp8N1024.valid() && kFp8N5120.valid() &&
              kFp8N6144.valid() && kFp8N10240.valid() &&
              kFp8N12288.valid() && kFp8Output.valid());
static_assert(q3x::kernels::kSm87P40PhaseBf16TileM == 128U);
static_assert(q3x::kernels::kSm87P40PhaseBf16TileN == 128U);
static_assert(q3x::kernels::kSm87P40PhaseBf16TileK == 64U);
static_assert(q3x::kernels::kSm87P40PhaseBf16Threads == 256U);
static_assert(q3x::kernels::kSm87P40PhaseBf16PipelineStages == 2U);
static_assert(q3x::kernels::kSm87P40PhaseBf16DynamicSharedBytes == 73'728U);

static_assert(kGateOrUp.input_features == 5'120U &&
              kGateOrUp.output_features == 17'408U &&
              kGateOrUp.grid_m == 313U && kGateOrUp.grid_n == 136U &&
              kGateOrUp.k_stages == 80U &&
              kGateOrUp.logical_ctas == 42'568U &&
              kGateOrUp.minimum_sm_waves == 2'661U);
static_assert(kDown.input_features == 17'408U &&
              kDown.output_features == 5'120U && kDown.grid_m == 313U &&
              kDown.grid_n == 40U && kDown.k_stages == 272U &&
              kDown.logical_ctas == 12'520U &&
              kDown.minimum_sm_waves == 783U);
static_assert(kFp8N1024.grid_n == 8U &&
              kFp8N1024.logical_ctas == 2'504U &&
              kFp8N1024.minimum_sm_waves == 157U);
static_assert(kFp8N5120.grid_n == 40U &&
              kFp8N5120.logical_ctas == 12'520U);
static_assert(kFp8N6144.grid_n == 48U &&
              kFp8N6144.logical_ctas == 15'024U &&
              kFp8N6144.minimum_sm_waves == 939U);
static_assert(kFp8N10240.grid_n == 80U &&
              kFp8N10240.logical_ctas == 25'040U &&
              kFp8N10240.minimum_sm_waves == 1'565U);
static_assert(kFp8N12288.grid_n == 96U &&
              kFp8N12288.logical_ctas == 30'048U &&
              kFp8N12288.minimum_sm_waves == 1'878U);
static_assert(kFp8Output.input_features == 6'144U &&
              kFp8Output.output_features == 5'120U &&
              kFp8Output.k_stages == 96U &&
              kFp8Output.logical_ctas == 12'520U);

constexpr auto kFirstGateTile =
    sm87_p40_phase_bf16_projection_tile(kGateOrUp, 0U, 0U);
constexpr auto kLastGateTile = sm87_p40_phase_bf16_projection_tile(
    kGateOrUp, kGateOrUp.grid_n - 1U, kGateOrUp.grid_m - 1U);
static_assert(kFirstGateTile.valid && kFirstGateTile.first_token == 0U &&
              kFirstGateTile.token_count == 128U &&
              kFirstGateTile.first_output_feature == 0U);
static_assert(kLastGateTile.valid &&
              kLastGateTile.first_token == 39'936U &&
              kLastGateTile.token_count == 64U &&
              kLastGateTile.first_output_feature == 17'280U &&
              kLastGateTile.output_feature_count == 128U);
static_assert(!sm87_p40_phase_bf16_projection_tile(
                   kGateOrUp, kGateOrUp.grid_n, 0U)
                   .valid);
static_assert(!sm87_p40_phase_bf16_projection_tile(
                   kGateOrUp, 0U, kGateOrUp.grid_m)
                   .valid);
static_assert(!make_sm87_p40_phase_bf16_projection_plan(
                   Sm87P40PhaseBf16ProjectionRole::kFp8K5120N5120,
                   39'999U)
                   .valid());
static_assert(!make_sm87_p40_phase_bf16_projection_plan(
                   static_cast<Sm87P40PhaseBf16ProjectionRole>(255U),
                   40'000U)
                   .valid());

constexpr Sm87P40PhaseBf16ProjectionResources kAdmittedResources{
    8, 7, 16, 87,
    q3x::kernels::kSm87P40PhaseBf16ExpectedRegistersPerThread, 0U,
    q3x::kernels::kSm87P40PhaseBf16DynamicSharedBytes, 0U, 1, true};
constexpr auto kSpilledResources = [] {
  auto resources = kAdmittedResources;
  resources.local_bytes = 16U;
  return resources;
}();
constexpr auto kWrongBinaryResources = [] {
  auto resources = kAdmittedResources;
  resources.binary_version = 86;
  return resources;
}();
constexpr auto kNoOccupancyResources = [] {
  auto resources = kAdmittedResources;
  resources.active_blocks_per_sm = 0;
  return resources;
}();
constexpr auto kWrongSmCountResources = [] {
  auto resources = kAdmittedResources;
  resources.sm_count = 8;
  return resources;
}();
static_assert(kAdmittedResources.valid());
static_assert(!kSpilledResources.valid());
static_assert(!kWrongBinaryResources.valid());
static_assert(!kNoOccupancyResources.valid());
static_assert(!kWrongSmCountResources.valid());

using ResourceQuery = int (*)(
    Sm87P40PhaseBf16ProjectionRole, std::size_t,
    Sm87P40PhaseBf16ProjectionResources*) noexcept;
using Launcher = int (*)(Sm87P40PhaseBf16ProjectionRole,
                         const std::uint16_t*, const std::uint16_t*,
                         std::size_t, float, std::uint16_t*, void*) noexcept;
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           query_sm87_p40_phase_bf16_projection_resources_cuda),
              ResourceQuery>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_p40_phase_bf16_projection_cuda),
              Launcher>);

// Force link-time inventory validation without touching a CUDA device.
[[maybe_unused]] ResourceQuery volatile kLinkedResourceQuery =
    &q3x::kernels::query_sm87_p40_phase_bf16_projection_resources_cuda;
[[maybe_unused]] Launcher volatile kLinkedLauncher =
    &q3x::kernels::launch_sm87_p40_phase_bf16_projection_cuda;

[[nodiscard]] bool check(const bool condition, const char* const message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

[[nodiscard]] bool exhaustive_grid_coverage(
    const Sm87P40PhaseBf16ProjectionPlan& plan) {
  std::size_t visited = 0U;
  for (std::size_t block_y = 0U; block_y < plan.grid_m; ++block_y) {
    for (std::size_t block_x = 0U; block_x < plan.grid_n; ++block_x) {
      const auto tile = sm87_p40_phase_bf16_projection_tile(
          plan, block_x, block_y);
      if (!tile.valid || tile.block_x != block_x ||
          tile.block_y != block_y ||
          tile.first_output_feature !=
              block_x * q3x::kernels::kSm87P40PhaseBf16TileN ||
          tile.first_token !=
              block_y * q3x::kernels::kSm87P40PhaseBf16TileM) {
        return false;
      }
      ++visited;
    }
  }
  return visited == plan.logical_ctas;
}

}  // namespace

int main() {
  constexpr std::array<Sm87P40PhaseBf16ProjectionPlan, 8U> kPlans{
      kGateOrUp, kDown, kFp8N1024, kFp8N5120,
      kFp8N6144, kFp8N10240, kFp8N12288, kFp8Output};
  bool ok = true;
  for (const auto& plan : kPlans) {
    ok &= check(plan.valid(), "an admitted role produced an invalid plan");
    ok &= check(plan.logical_ctas >
                    q3x::kernels::kSm87P40PhaseBf16SmCount,
                "the plan regressed to a tiny persistent grid");
    ok &= check(exhaustive_grid_coverage(plan),
                "ordinary two-dimensional grid coverage failed");
  }
  return ok ? 0 : 1;
}
