#include "q3x/kernels/sm87_nvfp4_prefill_shape_wide.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

namespace {

using q3x::kernels::Sm87NvFp4PrefillShapeWideDataflow;
using q3x::kernels::Sm87NvFp4PrefillShapeWideRole;
using q3x::kernels::sm87_nvfp4_prefill_shape_wide_plan;
using q3x::kernels::sm87_nvfp4_prefill_shape_wide_supports;
using q3x::kernels::sm87_nvfp4_prefill_shape_wide_tile_coordinate;

constexpr auto kGate = sm87_nvfp4_prefill_shape_wide_plan(
    Sm87NvFp4PrefillShapeWideRole::kGateUp, 40'000U);
constexpr auto kDown = sm87_nvfp4_prefill_shape_wide_plan(
    Sm87NvFp4PrefillShapeWideRole::kDown, 40'000U);

static_assert(kGate.valid() && kDown.valid());
static_assert(kGate.input_features == 5'120U);
static_assert(kGate.weight_output_features == 34'816U);
static_assert(kGate.published_output_features == 17'408U);
static_assert(kGate.tile_m == 128U && kGate.branch_tile_n == 64U &&
              kGate.physical_tile_n == 128U && kGate.tile_k == 64U);
static_assert(kGate.pipeline_stages == 3U);
static_assert(kGate.grid_m == 313U && kGate.grid_n == 272U &&
              kGate.tail_rows == 64U);
static_assert(kGate.dynamic_shared_bytes == 62'976U);
static_assert(kGate.group_m == 2U && !kGate.a_major && kGate.fused_silu &&
              !kGate.bf16_branch_boundary &&
              !kGate.fused_in_place_residual);
static_assert(
    kGate.dataflow == Sm87NvFp4PrefillShapeWideDataflow::
                          kGateUpM128N64PairGroupM2);

static_assert(kDown.input_features == 17'408U);
static_assert(kDown.weight_output_features == 5'120U);
static_assert(kDown.published_output_features == 5'120U);
static_assert(kDown.tile_m == 128U && kDown.branch_tile_n == 128U &&
              kDown.physical_tile_n == 128U && kDown.tile_k == 64U);
static_assert(kDown.pipeline_stages == 3U);
static_assert(kDown.grid_m == 313U && kDown.grid_n == 40U &&
              kDown.tail_rows == 64U);
static_assert(kDown.dynamic_shared_bytes == 62'976U);
static_assert(kDown.group_m == 1U && kDown.a_major && !kDown.fused_silu &&
              kDown.bf16_branch_boundary &&
              kDown.fused_in_place_residual);
static_assert(kDown.dataflow ==
              Sm87NvFp4PrefillShapeWideDataflow::
                  kDownM128N128GroupM1Amajor);

static_assert(sm87_nvfp4_prefill_shape_wide_tile_coordinate(
                  Sm87NvFp4PrefillShapeWideRole::kGateUp, 0U)
                      .m_tile == 0U);
static_assert(sm87_nvfp4_prefill_shape_wide_tile_coordinate(
                  Sm87NvFp4PrefillShapeWideRole::kGateUp, 1U)
                      .m_tile == 1U);
static_assert(sm87_nvfp4_prefill_shape_wide_tile_coordinate(
                  Sm87NvFp4PrefillShapeWideRole::kGateUp, 2U)
                      .n_tile == 1U);
static_assert(sm87_nvfp4_prefill_shape_wide_tile_coordinate(
                  Sm87NvFp4PrefillShapeWideRole::kGateUp, 85'135U)
                      .n_tile == 271U);
static_assert(sm87_nvfp4_prefill_shape_wide_tile_coordinate(
                  Sm87NvFp4PrefillShapeWideRole::kDown, 40U)
                      .m_tile == 1U);

static_assert(q3x::kernels::kSm87NvFp4PrefillShapeWideThreads == 256U);
static_assert(
    q3x::kernels::kSm87NvFp4PrefillShapeWideMinimumBlocksPerSm == 2U);
static_assert(q3x::kernels::kSm87NvFp4PrefillShapeWideGateDynamicSharedBytes <=
              q3x::kernels::kSm87NvFp4PrefillShapeWideSharedLimitBytes);
static_assert(q3x::kernels::kSm87NvFp4PrefillShapeWideDownDynamicSharedBytes <=
              q3x::kernels::kSm87NvFp4PrefillShapeWideSharedLimitBytes);

static_assert(!sm87_nvfp4_prefill_shape_wide_supports(
    Sm87NvFp4PrefillShapeWideRole::kGateUp, 0U));
static_assert(!sm87_nvfp4_prefill_shape_wide_supports(
    Sm87NvFp4PrefillShapeWideRole::kDown, 39'999U));
static_assert(!sm87_nvfp4_prefill_shape_wide_supports(
    Sm87NvFp4PrefillShapeWideRole::kGateUp, 40'001U));
static_assert(!sm87_nvfp4_prefill_shape_wide_plan(
                   static_cast<Sm87NvFp4PrefillShapeWideRole>(255U),
                   40'000U)
                   .valid());

constexpr bool forged_plans_fail_closed() {
  auto wrong_gate_grid = kGate;
  wrong_gate_grid.grid_m = 312U;
  auto wrong_down_raster = kDown;
  wrong_down_raster.group_m = 2U;
  auto wrong_down_boundary = kDown;
  wrong_down_boundary.bf16_branch_boundary = false;
  auto wrong_gate_shared = kGate;
  wrong_gate_shared.dynamic_shared_bytes = 70'144U;
  return !wrong_gate_grid.valid() && !wrong_down_raster.valid() &&
         !wrong_down_boundary.valid() && !wrong_gate_shared.valid();
}

static_assert(forged_plans_fail_closed());

using GateLauncher = int (*)(const std::uint16_t*, const std::uint8_t*,
                             const std::uint8_t*, const float*, std::size_t,
                             std::uint16_t*, void*) noexcept;
using DownLauncher = int (*)(const std::uint16_t*, const std::uint8_t*,
                             const std::uint8_t*, const float*,
                             const std::uint16_t*, std::size_t,
                             std::uint16_t*, void*) noexcept;
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_nvfp4_prefill_shape_wide_gate_up_cuda),
              GateLauncher>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_nvfp4_prefill_shape_wide_down_cuda),
              DownLauncher>);

[[nodiscard]] bool check(const bool condition, const char* const message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

[[nodiscard]] bool raster_is_bijection(
    const Sm87NvFp4PrefillShapeWideRole role,
    const q3x::kernels::Sm87NvFp4PrefillShapeWidePlan& plan) {
  std::vector<std::uint8_t> seen(plan.grid_m * plan.grid_n, 0U);
  for (std::size_t block = 0U; block < seen.size(); ++block) {
    const auto coordinate =
        sm87_nvfp4_prefill_shape_wide_tile_coordinate(role, block);
    if (!coordinate.mapped || coordinate.m_tile >= plan.grid_m ||
        coordinate.n_tile >= plan.grid_n) {
      return false;
    }
    const std::size_t logical =
        coordinate.m_tile * plan.grid_n + coordinate.n_tile;
    if (seen[logical] != 0U) {
      return false;
    }
    seen[logical] = 1U;
  }
  for (const std::uint8_t hit : seen) {
    if (hit != 1U) {
      return false;
    }
  }
  return !sm87_nvfp4_prefill_shape_wide_tile_coordinate(role, seen.size())
              .mapped;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= check(kGate.valid() && kDown.valid(),
              "exact P40000 shape-wide plans must be valid");
  ok &= check(kGate.grid_m * kGate.grid_n == 85'136U,
              "Gate group-M=2 physical grid mismatch");
  ok &= check(kDown.grid_m * kDown.grid_n == 12'520U,
              "Down A-major physical grid mismatch");
  ok &= check(raster_is_bijection(
                  Sm87NvFp4PrefillShapeWideRole::kGateUp, kGate),
              "Gate group-M=2 raster must cover every tile exactly once");
  ok &= check(raster_is_bijection(
                  Sm87NvFp4PrefillShapeWideRole::kDown, kDown),
              "Down A-major raster must cover every tile exactly once");
  for (const std::size_t unsupported :
       {1U, 512U, 8'000U, 8'192U, 39'999U, 40'001U, 60'000U}) {
    ok &= check(!sm87_nvfp4_prefill_shape_wide_supports(
                    Sm87NvFp4PrefillShapeWideRole::kGateUp, unsupported),
                "unsupported Gate M must fail closed");
    ok &= check(!sm87_nvfp4_prefill_shape_wide_supports(
                    Sm87NvFp4PrefillShapeWideRole::kDown, unsupported),
                "unsupported Down M must fail closed");
  }
  return ok ? 0 : 1;
}
