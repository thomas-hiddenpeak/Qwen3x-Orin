#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta.h"

#include <cstddef>
#include <iostream>

namespace {

namespace kernels = q3x::kernels;

static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileM ==
    128U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileN ==
    64U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N641CtaWarpM ==
    128U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N641CtaWarpN ==
    8U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N641CtaThreads ==
    256U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N641CtaAStages ==
    2U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N641CtaSharedBytes ==
    66'560U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N641CtaMaximumRegisters ==
    255U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N641CtaCtasPerSm ==
    1U);

[[nodiscard]] bool verify_v2_reuse() {
  constexpr std::size_t n = 128U;
  constexpr std::size_t k = 1'024U;
  constexpr std::size_t groups = k / 512U;
  const std::size_t code_capacity =
      kernels::sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
          n, k);
  const std::size_t scale_capacity =
      kernels::sm87_a4w4_gateup_k512_fragment_native_scale_capacity_elements(
          n, k);
  if (code_capacity != n * k ||
      scale_capacity != 2U * n * groups) {
    return false;
  }
  for (std::size_t warp = 0U; warp < 8U; ++warp) {
    for (std::size_t lane = 0U; lane < 32U; ++lane) {
      const std::size_t slot =
          kernels::sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
              warp * 8U, 1U, 7U, lane, groups);
      const std::size_t expected =
          (((1U * 8U + 7U) * 8U + warp) * 32U + lane) * 16U;
      if (slot != expected || slot % 16U != 0U ||
          slot + 16U > code_capacity) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!verify_v2_reuse()) {
    std::cerr << "v2 paired ABI mismatch\n";
    return 1;
  }
  const auto window0 =
      kernels::
          sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_plan(
              1'920U, 17'408U, 5'120U, 0U, 12'288U);
  const auto window1 =
      kernels::
          sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_plan(
              1'920U, 17'408U, 5'120U, 12'288U, 5'120U);
  if (!kernels::
          sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_is_model_plan(
              window0) ||
      !kernels::
          sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_is_model_plan(
              window1) ||
      window0.m_tiles != 15U || window0.n_tiles != 192U ||
      window0.work_cells != 2'880U || window0.launch_ctas != 16U ||
      window0.minimum_cells_per_cta != 180U ||
      window0.maximum_cells_per_cta != 180U ||
      window1.m_tiles != 15U || window1.n_tiles != 80U ||
      window1.work_cells != 1'200U || window1.launch_ctas != 16U ||
      window1.minimum_cells_per_cta != 75U ||
      window1.maximum_cells_per_cta != 75U) {
    std::cerr << "model plan mismatch\n";
    return 1;
  }
  const auto minimum =
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_plan(
          128U, 64U, 512U, 0U, 64U);
  const auto maximum =
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_plan(
          4'096U, 128U, 512U, 0U, 128U);
  if (minimum.work_cells != 1U || minimum.launch_ctas != 1U ||
      maximum.m_tiles != 32U || maximum.n_tiles != 2U ||
      maximum.work_cells != 64U || maximum.launch_ctas != 16U ||
      maximum.minimum_cells_per_cta != 4U ||
      maximum.maximum_cells_per_cta != 4U ||
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_plan(
          64U, 64U, 512U, 0U, 64U)
              .launch_ctas != 0U ||
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_plan(
          4'224U, 64U, 512U, 0U, 64U)
              .launch_ctas != 0U ||
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_plan(
          128U, 128U, 512U, 32U, 64U)
              .launch_ctas != 0U) {
    std::cerr << "shape admission mismatch\n";
    return 1;
  }
  std::cout << "PASS: M128N64/1CTA plan is N-major, v2-native, and "
               "model-window exact\n";
  return 0;
}
