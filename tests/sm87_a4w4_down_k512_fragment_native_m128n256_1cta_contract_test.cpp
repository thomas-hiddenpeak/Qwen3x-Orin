#include "q3x/kernels/sm87_a4w4_down_k512_fragment_native_m128n256_1cta.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace {

class Test final {
 public:
  void expect(const bool condition, const char* const description) {
    if (!condition) {
      std::cerr << "FAIL: " << description << '\n';
      ++failures_;
    }
  }
  [[nodiscard]] int result() const noexcept {
    return failures_ == 0 ? 0 : 1;
  }

 private:
  int failures_{};
};

void verify_schedule(
    Test& test,
    const q3x::kernels::
        Sm87A4W4DownK512FragmentNativeM128N2561CtaPlan& plan) {
  namespace kernels = q3x::kernels;
  std::vector<unsigned int> visits(plan.work_cells, 0U);
  for (std::size_t cta = 0U; cta < plan.launch_ctas; ++cta) {
    for (std::size_t ni = 0U;
         ni < plan.maximum_n_stripes_per_group; ++ni) {
      for (std::size_t mi = 0U;
           mi < plan.maximum_m_tiles_per_owner; ++mi) {
        const auto tile =
            kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_tile(
                plan, cta, ni, mi);
        if (!tile.valid) {
          continue;
        }
        ++visits[tile.n_stripe * plan.m_tiles + tile.m_tile];
      }
    }
  }
  for (const unsigned int count : visits) {
    test.expect(count == 1U,
                "persistent schedule did not visit a cell exactly once");
  }
}

}  // namespace

int main() {
  namespace kernels = q3x::kernels;
  Test test;

  test.expect(
      kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaTileM == 128U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaTileN ==
              256U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaWarpM ==
              128U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaWarpN ==
              32U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaThreads ==
              256U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaWarps == 8U,
      "M128N256 CTA or M128N32 warp ownership changed");
  test.expect(
      kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp ==
              8U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaN16Phases ==
              2U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaN8PerPhase ==
              2U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaK64PerScale ==
              8U,
      "one B record -> eight M16 reuse or N16 phase contract changed");
  test.expect(
      kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaAStages == 2U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaAStageBytes ==
              32'768U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaScaleSlotBytes ==
              768U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes ==
              67'072U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaCtasPerSm ==
              1U &&
          kernels::kSm87A4W4DownK512FragmentNativeM128N2561CtaMaximumRegisters ==
              255U,
      "two-stage M128K512 pipeline or one-CTA resource contract changed");

  constexpr auto p128 =
      kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_plan(
          128U, 5'120U, 17'408U);
  test.expect(p128.m_tiles == 1U && p128.n_stripes == 20U &&
                  p128.k512_groups == 34U &&
                  p128.physical_k64_groups == 272U &&
                  p128.work_cells == 20U && p128.m_owner_ctas == 1U &&
                  p128.n_wave_groups == 16U && p128.launch_ctas == 16U,
              "P128 short-M plan does not occupy sixteen SMs");
  constexpr auto p512 =
      kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_plan(
          512U, 5'120U, 17'408U);
  test.expect(p512.m_tiles == 4U && p512.n_stripes == 20U &&
                  p512.m_owner_ctas == 4U && p512.n_wave_groups == 4U &&
                  p512.launch_ctas == 16U &&
                  p512.maximum_n_stripes_per_group == 5U,
              "P512 short-M four-wave schedule changed");
  constexpr auto p1920 =
      kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_plan(
          1'920U, 5'120U, 17'408U);
  test.expect(p1920.m_tiles == 15U && p1920.n_stripes == 20U &&
                  p1920.work_cells == 300U &&
                  p1920.m_owner_ctas == 15U &&
                  p1920.n_wave_groups == 1U &&
                  p1920.launch_ctas == 15U,
              "P1920 is not a synchronous fifteen-CTA B wave");
  constexpr auto p2048 =
      kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_plan(
          2'048U, 5'120U, 17'408U);
  test.expect(p2048.m_tiles == 16U && p2048.n_stripes == 20U &&
                  p2048.work_cells == 320U &&
                  p2048.launch_ctas == 16U,
              "P2048 model plan changed");
  constexpr auto p4096 =
      kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_plan(
          4'096U, 5'120U, 17'408U);
  test.expect(p4096.m_tiles == 32U && p4096.n_stripes == 20U &&
                  p4096.work_cells == 640U &&
                  p4096.m_owner_ctas == 16U &&
                  p4096.n_wave_groups == 1U &&
                  p4096.launch_ctas == 16U &&
                  p4096.maximum_m_tiles_per_owner == 2U,
              "P4096 two-M-tile owner schedule changed");

  verify_schedule(test, p128);
  verify_schedule(test, p512);
  verify_schedule(test, p1920);
  verify_schedule(test, p2048);
  verify_schedule(test, p4096);

  constexpr auto k512 =
      kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_test_plan(
          128U, 256U, 512U);
  constexpr auto k1024 =
      kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_test_plan(
          256U, 512U, 1'024U);
  test.expect(k512.work_cells == 1U && k512.k512_groups == 1U &&
                  k512.launch_ctas == 1U &&
                  k1024.work_cells == 4U &&
                  k1024.k512_groups == 2U && k1024.launch_ctas == 4U,
              "generic K512/K1024 complete-cell plan changed");
  test.expect(
      kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_plan(
          64U, 5'120U, 17'408U)
              .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_plan(
              1'984U, 5'120U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_plan(
              2'048U, 4'864U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_test_plan(
              128U, 128U, 512U)
                  .launch_ctas == 0U,
      "production/test shapes did not fail closed");
  test.expect(
      kernels::query_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_resources_cuda(
          nullptr) == static_cast<int>(cudaErrorInvalidValue),
      "resource query did not reject null before CUDA state access");

  if (test.result() == 0) {
    std::cout << "SM87 Down K512 fragment-native M128N256/1CTA contract passed\n";
  }
  return test.result();
}
