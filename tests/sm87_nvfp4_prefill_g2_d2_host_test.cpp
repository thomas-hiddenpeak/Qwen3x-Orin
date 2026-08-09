#include "q3x/kernels/sm87_nvfp4_prefill_g2_d2.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {

using q3x::kernels::Sm87NvFp4PrefillG2D2Dataflow;
using q3x::kernels::Sm87NvFp4PrefillG2D2Role;
using q3x::kernels::sm87_nvfp4_prefill_g2_d2_plan;
using q3x::kernels::sm87_nvfp4_prefill_g2_d2_supports;

constexpr auto kGate8192 = sm87_nvfp4_prefill_g2_d2_plan(
    Sm87NvFp4PrefillG2D2Role::kGateUpG2, 8'192U);
constexpr auto kGate7712 = sm87_nvfp4_prefill_g2_d2_plan(
    Sm87NvFp4PrefillG2D2Role::kGateUpG2, 7'712U);
constexpr auto kDown8192 = sm87_nvfp4_prefill_g2_d2_plan(
    Sm87NvFp4PrefillG2D2Role::kDownD2, 8'192U);
constexpr auto kDown7712 = sm87_nvfp4_prefill_g2_d2_plan(
    Sm87NvFp4PrefillG2D2Role::kDownD2, 7'712U);

static_assert(kGate8192.valid() && kGate7712.valid());
static_assert(kDown8192.valid() && kDown7712.valid());
static_assert(kGate8192.input_features == 5'120U);
static_assert(kGate8192.weight_output_features == 34'816U);
static_assert(kGate8192.published_output_features == 17'408U);
static_assert(kGate8192.branch_tile_n == 64U);
static_assert(kGate8192.grid_m == 64U && kGate8192.grid_n == 272U);
static_assert(kGate8192.tail_rows == 0U);
static_assert(kGate7712.grid_m == 61U && kGate7712.grid_n == 272U);
static_assert(kGate7712.tail_rows == 32U);
static_assert(
    kGate8192.dataflow ==
    Sm87NvFp4PrefillG2D2Dataflow::kGateUpM128PairRaster);

static_assert(kDown8192.input_features == 17'408U);
static_assert(kDown8192.weight_output_features == 5'120U);
static_assert(kDown8192.published_output_features == 5'120U);
static_assert(kDown8192.branch_tile_n == 128U);
static_assert(kDown8192.grid_m == 64U && kDown8192.grid_n == 40U);
static_assert(kDown8192.tail_rows == 0U);
static_assert(kDown7712.grid_m == 61U && kDown7712.grid_n == 40U);
static_assert(kDown7712.tail_rows == 32U);
static_assert(
    kDown8192.dataflow ==
    Sm87NvFp4PrefillG2D2Dataflow::kDownM128N128BStationaryRaster);

static_assert(kGate8192.tile_m == 128U && kDown8192.tile_m == 128U);
static_assert(kGate8192.tile_k == 64U && kDown8192.tile_k == 64U);
static_assert(kGate8192.threads == 256U && kDown8192.threads == 256U);
static_assert(kGate8192.pipeline_stages == 2U &&
              kDown8192.pipeline_stages == 2U);
static_assert(kGate8192.dynamic_shared_bytes == 41'984U &&
              kDown8192.dynamic_shared_bytes == 41'984U);
static_assert(q3x::kernels::kSm87NvFp4PrefillG2D2SharedABytesPerStage ==
              16'384U);
static_assert(q3x::kernels::kSm87NvFp4PrefillG2D2SharedBBytesPerStage ==
              4'096U);
static_assert(q3x::kernels::kSm87NvFp4PrefillG2D2SharedScaleBytesPerStage ==
              512U);

static_assert(!sm87_nvfp4_prefill_g2_d2_supports(
    Sm87NvFp4PrefillG2D2Role::kGateUpG2, 0U));
static_assert(!sm87_nvfp4_prefill_g2_d2_supports(
    Sm87NvFp4PrefillG2D2Role::kDownD2, 8'191U));
static_assert(!sm87_nvfp4_prefill_g2_d2_plan(
                   static_cast<Sm87NvFp4PrefillG2D2Role>(255U), 8'192U)
                   .valid());

constexpr bool forged_plan_is_rejected() {
  auto wrong_shape = kGate8192;
  wrong_shape.published_output_features = 34'816U;
  auto wrong_grid = kDown7712;
  wrong_grid.grid_n = 41U;
  auto wrong_dataflow = kGate8192;
  wrong_dataflow.dataflow =
      Sm87NvFp4PrefillG2D2Dataflow::kDownM128N128BStationaryRaster;
  auto wrong_tail = kDown7712;
  wrong_tail.tail_rows = 0U;
  return !wrong_shape.valid() && !wrong_grid.valid() &&
         !wrong_dataflow.valid() && !wrong_tail.valid();
}

static_assert(forged_plan_is_rejected());

using GateLauncher = int (*)(const std::uint16_t*, const std::uint8_t*,
                             const std::uint8_t*, const float*, std::size_t,
                             std::uint16_t*, void*) noexcept;
using DownLauncher = int (*)(const std::uint16_t*, const std::uint8_t*,
                             const std::uint8_t*, const float*,
                             const std::uint16_t*, std::size_t,
                             std::uint16_t*, void*) noexcept;
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_nvfp4_prefill_gate_up_g2_cuda),
              GateLauncher>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::launch_sm87_nvfp4_prefill_down_d2_cuda),
              DownLauncher>);

[[nodiscard]] bool check(const bool condition, const char* const message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

}  // namespace

int main() {
  bool ok = true;
  for (const auto token_count : {std::size_t{8'192U},
                                 std::size_t{7'712U}}) {
    const auto gate = sm87_nvfp4_prefill_g2_d2_plan(
        Sm87NvFp4PrefillG2D2Role::kGateUpG2, token_count);
    const auto down = sm87_nvfp4_prefill_g2_d2_plan(
        Sm87NvFp4PrefillG2D2Role::kDownD2, token_count);
    ok &= check(gate.valid() && down.valid(),
                "supported G2/D2 plans must be valid");
    ok &= check(gate.grid_n == 272U && down.grid_n == 40U,
                "G2/D2 grid-N contract mismatch");
    ok &= check(gate.dynamic_shared_bytes == 41'984U &&
                    down.dynamic_shared_bytes == 41'984U,
                "G2/D2 shared-memory contract mismatch");
  }

  for (const auto token_count :
       {std::size_t{0U}, std::size_t{512U}, std::size_t{513U},
        std::size_t{7'711U}, std::size_t{8'191U}, std::size_t{8'193U}}) {
    ok &= check(!sm87_nvfp4_prefill_g2_d2_supports(
                    Sm87NvFp4PrefillG2D2Role::kGateUpG2, token_count),
                "unsupported G2 M must fail closed");
    ok &= check(!sm87_nvfp4_prefill_g2_d2_supports(
                    Sm87NvFp4PrefillG2D2Role::kDownD2, token_count),
                "unsupported D2 M must fail closed");
  }
  return ok ? 0 : 1;
}
