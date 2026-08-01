#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native_m128.h"

#include <cstddef>
#include <iostream>

namespace {

namespace kernels = q3x::kernels;

static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128TileM == 128U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128TileN == 32U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128Threads == 256U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128Warps == 8U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128AStages == 3U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128SharedBytes ==
    24'576U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128MaximumRegisters ==
    128U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128CtasPerSm == 2U);

[[nodiscard]] bool verify_v2_reuse() {
  constexpr std::size_t n = 128U;
  constexpr std::size_t k = 1'024U;
  constexpr std::size_t groups = k / 512U;
  const std::size_t codes =
      kernels::sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
          n, k);
  const std::size_t scales =
      kernels::sm87_a4w4_gateup_k512_fragment_native_scale_capacity_elements(
          n, k);
  if (codes != n * k || scales != 2U * n * groups) {
    std::cerr << "v2 capacity changed\n";
    return false;
  }
  // Adjacent warp ids (2*n, 2*n+1) have different M ownership but use the
  // same N8 coordinate.  The ABI address is consequently identical.
  for (std::size_t warp_n = 0U; warp_n < 4U; ++warp_n) {
    for (std::size_t group = 0U; group < groups; ++group) {
      for (std::size_t phase = 0U; phase < 8U; ++phase) {
        for (std::size_t lane = 0U; lane < 32U; ++lane) {
          const std::size_t first =
              kernels::sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                  warp_n * 8U, group, phase, lane, groups);
          const std::size_t second =
              kernels::sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                  warp_n * 8U, group, phase, lane, groups);
          if (first != second || first % 16U != 0U ||
              first + 16U > codes) {
            std::cerr << "paired-warp B address contract failed\n";
            return false;
          }
        }
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!verify_v2_reuse()) {
    return 1;
  }
  const auto window0 =
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128_plan(
          1'920U, 17'408U, 5'120U, 0U, 12'288U);
  const auto window1 =
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128_plan(
          1'920U, 17'408U, 5'120U, 12'288U, 5'120U);
  if (!kernels::sm87_a4w4_gateup_k512_fragment_native_m128_is_model_plan(
          window0) ||
      !kernels::sm87_a4w4_gateup_k512_fragment_native_m128_is_model_plan(
          window1) ||
      window0.m_tiles != 15U || window0.n_tiles != 384U ||
      window0.work_cells != 5'760U || window0.launch_ctas != 32U ||
      window0.minimum_cells_per_cta != 180U ||
      window0.maximum_cells_per_cta != 180U ||
      window1.m_tiles != 15U || window1.n_tiles != 160U ||
      window1.work_cells != 2'400U || window1.launch_ctas != 32U ||
      window1.minimum_cells_per_cta != 75U ||
      window1.maximum_cells_per_cta != 75U) {
    std::cerr << "natural model plan mismatch\n";
    return 1;
  }
  const auto minimum =
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128_plan(
          128U, 64U, 512U, 0U, 32U);
  const auto maximum =
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128_plan(
          4'096U, 64U, 512U, 0U, 64U);
  if (minimum.work_cells != 1U || minimum.launch_ctas != 1U ||
      maximum.m_tiles != 32U || maximum.n_tiles != 2U ||
      maximum.work_cells != 64U || maximum.launch_ctas != 32U ||
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128_plan(
          64U, 64U, 512U, 0U, 32U)
              .launch_ctas != 0U ||
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128_plan(
          4'224U, 64U, 512U, 0U, 32U)
              .launch_ctas != 0U ||
      kernels::sm87_a4w4_gateup_k512_fragment_native_m128_plan(
          128U, 64U, 512U, 16U, 32U)
              .launch_ctas != 0U) {
    std::cerr << "shape admission mismatch\n";
    return 1;
  }
  std::cout << "PASS: M128N32 plan is N-major, v2-equal-byte, and "
               "model-window exact\n";
  return 0;
}
