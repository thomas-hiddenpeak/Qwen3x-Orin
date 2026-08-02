#include "q3x/kernels/sm87_a4w4_down_k512_m16n64_v2.h"

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

void check_schedule(
    Test& test,
    const q3x::kernels::Sm87A4W4DownK512M16N64V2Plan& plan,
    const char* const description) {
  namespace kernels = q3x::kernels;
  std::vector<unsigned int> visits(plan.work_tiles, 0U);
  for (std::size_t cta = 0U; cta < plan.launch_ctas; ++cta) {
    for (std::size_t iteration = 0U;; ++iteration) {
      const auto tile =
          kernels::sm87_a4w4_down_k512_m16n64_v2_work_tile(
              plan, cta, iteration);
      if (!tile.valid) {
        break;
      }
      test.expect(tile.m_tile < plan.m_tiles && tile.n_tile < plan.n_tiles,
                  "scheduler emitted an out-of-range tile");
      if (tile.m_tile < plan.m_tiles && tile.n_tile < plan.n_tiles) {
        ++visits[tile.n_tile * plan.m_tiles + tile.m_tile];
      }
    }
  }
  bool exact = true;
  for (const unsigned int count : visits) {
    exact = exact && count == 1U;
  }
  test.expect(exact, description);
}

}  // namespace

int main() {
  namespace kernels = q3x::kernels;
  Test test;

  test.expect(
      kernels::kSm87A4W4DownK512M16N64V2TileM == 128U &&
          kernels::kSm87A4W4DownK512M16N64V2TileN == 128U &&
          kernels::kSm87A4W4DownK512M16N64V2WarpTileM == 16U &&
          kernels::kSm87A4W4DownK512M16N64V2WarpTileN == 64U &&
          kernels::kSm87A4W4DownK512M16N64V2WarpRows == 8U &&
          kernels::kSm87A4W4DownK512M16N64V2WarpColumns == 2U &&
          kernels::kSm87A4W4DownK512M16N64V2Threads == 512U &&
          kernels::kSm87A4W4DownK512M16N64V2Warps == 16U,
      "M128N128 CTA and 8x2 M16N64 warp ownership changed");
  test.expect(
      kernels::kSm87A4W4DownK512M16N64V2CopyK == 256U &&
          kernels::kSm87A4W4DownK512M16N64V2ScaleK == 512U &&
          kernels::kSm87A4W4DownK512M16N64V2K64PerStage == 4U &&
          kernels::kSm87A4W4DownK512M16N64V2StagesPerScale == 2U &&
          kernels::kSm87A4W4DownK512M16N64V2Stages == 4U,
      "four-stage K256 ring or K512 boundary changed");
  test.expect(
      kernels::kSm87A4W4DownK512M16N64V2StageABytes == 16U * 1'024U &&
          kernels::kSm87A4W4DownK512M16N64V2StageBBytes ==
              16U * 1'024U &&
          kernels::kSm87A4W4DownK512M16N64V2StageBytes ==
              32U * 1'024U &&
          kernels::kSm87A4W4DownK512M16N64V2DynamicSharedBytes ==
              128U * 1'024U &&
          kernels::kSm87A4W4DownK512M16N64V2MaximumRegisters == 128U &&
          kernels::kSm87A4W4DownK512M16N64V2CtasPerSm == 1U,
      "full-residency resource envelope changed");

  constexpr auto p1920 =
      kernels::sm87_a4w4_down_k512_m16n64_v2_plan(
          1'920U, 5'120U, 17'408U);
  test.expect(p1920.m_tiles == 15U && p1920.n_tiles == 40U &&
                  p1920.k512_groups == 34U &&
                  p1920.physical_k256_stages == 68U &&
                  p1920.physical_k64_groups == 272U &&
                  p1920.work_tiles == 600U &&
                  p1920.launch_ctas == 15U &&
                  p1920.base_waves == 1U &&
                  p1920.residual_m_tiles == 0U,
              "P1853 launch extent is not fifteen fixed M owners");
  for (std::size_t cta = 0U; cta < p1920.launch_ctas; ++cta) {
    for (std::size_t n = 0U; n < p1920.n_tiles; ++n) {
      const auto tile =
          kernels::sm87_a4w4_down_k512_m16n64_v2_work_tile(
              p1920, cta, n);
      test.expect(tile.valid && tile.m_tile == cta && tile.n_tile == n,
                  "P1920 CTA did not retain one M owner across N");
    }
    test.expect(
        !kernels::sm87_a4w4_down_k512_m16n64_v2_work_tile(
             p1920, cta, p1920.n_tiles)
             .valid,
        "P1920 CTA emitted work after its complete N sweep");
  }
  check_schedule(test, p1920,
                 "P1920 scheduler did not cover every tile once");

  constexpr auto p2176 =
      kernels::sm87_a4w4_down_k512_m16n64_v2_plan(
          2'176U, 5'120U, 17'408U);
  test.expect(p2176.m_tiles == 17U && p2176.launch_ctas == 16U &&
                  p2176.base_waves == 1U &&
                  p2176.base_m_tiles == 16U &&
                  p2176.residual_m_tiles == 1U &&
                  p2176.residual_work_tiles == 40U,
              "P2176 did not isolate one residual M wave");
  check_schedule(test, p2176,
                 "P2176 complete-cell residual schedule duplicated work");

  constexpr auto p3840 =
      kernels::sm87_a4w4_down_k512_m16n64_v2_plan(
          3'840U, 5'120U, 17'408U);
  test.expect(p3840.m_tiles == 30U && p3840.launch_ctas == 16U &&
                  p3840.base_waves == 1U &&
                  p3840.residual_m_tiles == 14U &&
                  p3840.residual_work_tiles == 560U,
              "P3840 residual wave decomposition changed");
  check_schedule(test, p3840,
                 "P3840 complete-cell residual schedule duplicated work");

  constexpr auto p4096 =
      kernels::sm87_a4w4_down_k512_m16n64_v2_plan(
          4'096U, 5'120U, 17'408U);
  test.expect(p4096.m_tiles == 32U && p4096.launch_ctas == 16U &&
                  p4096.base_waves == 2U &&
                  p4096.residual_m_tiles == 0U,
              "P4096 is not two complete fixed-owner waves");
  check_schedule(test, p4096,
                 "P4096 scheduler did not cover every tile once");

  test.expect(
      kernels::sm87_a4w4_down_k512_m16n64_v2_plan(
          1'984U, 5'120U, 17'408U)
              .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_m16n64_v2_plan(
              4'224U, 5'120U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_m16n64_v2_plan(
              2'048U, 4'992U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_m16n64_v2_plan(
              2'048U, 5'120U, 16'896U)
                  .launch_ctas == 0U,
      "production admission did not enforce complete M128 and fixed N/K");
  test.expect(
      kernels::sm87_a4w4_down_k512_m16n64_v2_test_plan(
          128U, 128U, 512U)
                  .work_tiles == 1U &&
          kernels::sm87_a4w4_down_k512_m16n64_v2_test_plan(
              127U, 128U, 512U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_m16n64_v2_test_plan(
              128U, 64U, 512U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_m16n64_v2_test_plan(
              128U, 128U, 256U)
                  .launch_ctas == 0U,
      "correctness-only plan did not fail closed on M/N/K tails");

  constexpr std::size_t a_bytes =
      kernels::sm87_a4w4_down_k512_packed_capacity_bytes(
          1'920U, 17'408U);
  constexpr std::size_t a_scales =
      kernels::sm87_a4w4_down_k512_scale_capacity_elements(
          1'920U, 17'408U);
  test.expect(a_bytes == 16'711'680U && a_scales == 65'280U,
              "P1920 inherited K512 sidecar capacities changed");

  test.expect(
      kernels::query_sm87_a4w4_down_k512_m16n64_v2_resources_cuda(
          nullptr) == static_cast<int>(cudaErrorInvalidValue),
      "resource query did not reject null before touching CUDA state");

  if (test.result() == 0) {
    std::cout << "SM87 Down K512 M16N64 v2 contract passed\n";
  }
  return test.result();
}
