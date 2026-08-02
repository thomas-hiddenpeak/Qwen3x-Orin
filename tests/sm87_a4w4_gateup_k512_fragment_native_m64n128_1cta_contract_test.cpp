#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM64N1281CtaTileM ==
    64U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM64N1281CtaTileN ==
    128U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM64N1281CtaThreads ==
    512U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM64N1281CtaWarps ==
    16U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM64N1281CtaWarpN ==
    8U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM64N1281CtaSharedBytes ==
    12'288U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM64N1281CtaMaximumRegisters ==
    128U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM64N1281CtaCtasPerSm ==
    1U);

[[nodiscard]] bool verify_two_block_v2_coverage() {
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
    std::cerr << "v2 paired capacity changed\n";
    return false;
  }

  // Warp 0..7 consume the first N64 block and warp 8..15 consume the
  // adjacent block.  Every aligned 16-byte Gate+Up slot must be covered
  // exactly once by the CTA.
  std::vector<std::uint8_t> seen(code_capacity / 16U, 0U);
  for (std::size_t group = 0U; group < groups; ++group) {
    for (std::size_t phase = 0U; phase < 8U; ++phase) {
      for (std::size_t warp = 0U; warp < 16U; ++warp) {
        for (std::size_t lane = 0U; lane < 32U; ++lane) {
          const std::size_t slot =
              kernels::
                  sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                      warp * 8U, group, phase, lane, groups);
          const std::size_t block = warp / 8U;
          const std::size_t fragment = warp % 8U;
          const std::size_t expected =
              (((((block * groups + group) * 8U + phase) * 8U +
                  fragment) *
                     32U +
                 lane) *
                16U);
          if (slot != expected || slot % 16U != 0U ||
              slot + 16U > code_capacity || seen[slot / 16U] != 0U) {
            std::cerr << "two-block warp-to-payload mapping mismatch\n";
            return false;
          }
          seen[slot / 16U] = 1U;
        }
      }
    }
  }
  for (const std::uint8_t value : seen) {
    if (value != 1U) {
      std::cerr << "two-block payload coverage has a hole\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!verify_two_block_v2_coverage()) {
    return 1;
  }

  const auto window0 =
      kernels::
          sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
              1'920U, 17'408U, 5'120U, 0U, 12'288U);
  const auto window1 =
      kernels::
          sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
              1'920U, 17'408U, 5'120U, 12'288U, 5'120U);
  if (!kernels::
          sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_is_model_plan(
              window0) ||
      !kernels::
          sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_is_model_plan(
              window1) ||
      window0.m_tiles != 30U || window0.n_tiles != 96U ||
      window0.k512_groups != 10U ||
      window0.physical_k64_groups != 80U ||
      window0.work_cells != 2'880U || window0.launch_ctas != 16U ||
      window1.m_tiles != 30U || window1.n_tiles != 40U ||
      window1.k512_groups != 10U ||
      window1.physical_k64_groups != 80U ||
      window1.work_cells != 1'200U || window1.launch_ctas != 16U) {
    std::cerr << "model plan mismatch\n";
    return 1;
  }

  const auto minimum =
      kernels::
          sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
              64U, 128U, 512U, 0U, 128U);
  const auto persistent =
      kernels::
          sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
              1'088U, 256U, 1'024U, 128U, 128U);
  if (minimum.m_tiles != 1U || minimum.n_tiles != 1U ||
      minimum.work_cells != 1U || minimum.launch_ctas != 1U ||
      persistent.m_tiles != 17U || persistent.n_tiles != 1U ||
      persistent.work_cells != 17U || persistent.launch_ctas != 16U ||
      kernels::
              sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
                  32U, 128U, 512U, 0U, 128U)
              .launch_ctas != 0U ||
      kernels::
              sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
                  64U, 256U, 512U, 64U, 128U)
              .launch_ctas != 0U ||
      kernels::
              sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
                  64U, 128U, 512U, 0U, 64U)
              .launch_ctas != 0U ||
      kernels::
          sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_is_model_plan(
              kernels::
                  sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
                      1'920U, 17'408U, 5'120U, 0U, 128U))) {
    std::cerr << "shape admission mismatch\n";
    return 1;
  }

  std::cout << "PASS: M64N128/1CTA plan reuses the complete two-block v2 "
               "payload and admits exact model windows\n";
  return 0;
}
