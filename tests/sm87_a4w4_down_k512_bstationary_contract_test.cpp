#include "q3x/kernels/sm87_a4w4_down_k512_bstationary.h"

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
    const q3x::kernels::Sm87A4W4DownK512BStationaryPlan& plan) {
  namespace kernels = q3x::kernels;
  std::vector<unsigned int> visits(plan.work_tiles, 0U);
  std::vector<std::size_t> cta_work(plan.launch_ctas, 0U);
  if (kernels::sm87_a4w4_down_k512_bstationary_uses_exact_owner_groups(
          plan)) {
    for (std::size_t group = 0U; group < plan.owner_groups; ++group) {
      for (std::size_t cta = 0U; cta < plan.launch_ctas; ++cta) {
        std::size_t expected_n = 0U;
        bool owner_valid = false;
        std::size_t owner_m = 0U;
        for (std::size_t n_wave = 0U; n_wave < plan.n_tiles; ++n_wave) {
          const auto tile = kernels::sm87_a4w4_down_k512_bstationary_tile(
              plan, cta, group, n_wave);
          if (!tile.valid) {
            continue;
          }
          if (!owner_valid) {
            owner_valid = true;
            owner_m = tile.m_tile;
          }
          test.expect(tile.m_tile == owner_m,
                      "CTA changed M owner inside one exact B wave");
          test.expect(tile.n_tile == expected_n,
                      "CTA skipped or reordered an exact N wave");
          ++expected_n;
          ++cta_work[cta];
          ++visits[tile.n_tile * plan.m_tiles + tile.m_tile];
        }
        if (owner_valid) {
          test.expect(expected_n == plan.n_tiles,
                      "exact owner did not visit every N tile");
        }
      }
    }
  } else {
    const std::size_t maximum_iterations =
        (plan.work_tiles + plan.launch_ctas - 1U) / plan.launch_ctas;
    for (std::size_t cta = 0U; cta < plan.launch_ctas; ++cta) {
      for (std::size_t iteration = 0U; iteration < maximum_iterations;
           ++iteration) {
        const auto tile =
            kernels::sm87_a4w4_down_k512_bstationary_balanced_tile(
                plan, cta, iteration);
        if (!tile.valid) {
          continue;
        }
        ++cta_work[cta];
        ++visits[tile.n_tile * plan.m_tiles + tile.m_tile];
      }
    }
    std::size_t minimum_work = cta_work.front();
    std::size_t maximum_work = cta_work.front();
    for (const std::size_t count : cta_work) {
      minimum_work = count < minimum_work ? count : minimum_work;
      maximum_work = count > maximum_work ? count : maximum_work;
    }
    test.expect(maximum_work - minimum_work <= 1U,
                "balanced schedule differs by more than one cell per CTA");
    test.expect(
        maximum_work ==
                (plan.work_tiles + plan.launch_ctas - 1U) /
                    plan.launch_ctas &&
            minimum_work == plan.work_tiles / plan.launch_ctas,
        "balanced schedule did not realize floor/ceil cell ownership");
  }
  for (const unsigned int count : visits) {
    test.expect(count == 1U, "hybrid schedule did not visit a cell once");
  }
}

}  // namespace

int main() {
  namespace kernels = q3x::kernels;
  Test test;

  test.expect(
      kernels::kSm87A4W4DownK512BStationaryTileM == 128U &&
          kernels::kSm87A4W4DownK512BStationaryTileN == 128U &&
          kernels::kSm87A4W4DownK512BStationaryWarpTileM == 16U &&
          kernels::kSm87A4W4DownK512BStationaryWarpTileN == 64U &&
          kernels::kSm87A4W4DownK512BStationaryThreads == 512U &&
          kernels::kSm87A4W4DownK512BStationaryWarps == 16U,
      "M128N128 CTA or 8x2 M16N64 warp ownership changed");
  test.expect(
      kernels::kSm87A4W4DownK512BStationaryStageK == 256U &&
          kernels::kSm87A4W4DownK512BStationaryScaleK == 512U &&
          kernels::kSm87A4W4DownK512BStationaryK64PerStage == 4U &&
          kernels::kSm87A4W4DownK512BStationaryStagesPerScale == 2U &&
          kernels::kSm87A4W4DownK512BStationaryStages == 3U,
      "three-stage K256 ring or K512 numerical boundary changed");
  test.expect(
      kernels::kSm87A4W4DownK512BStationaryStageABytes == 16U * 1'024U &&
          kernels::kSm87A4W4DownK512BStationaryStageBBytes ==
              16U * 1'024U &&
          kernels::kSm87A4W4DownK512BStationaryDynamicSharedBytes ==
              96U * 1'024U &&
          kernels::kSm87A4W4DownK512BStationaryCtasPerSm == 1U &&
          kernels::kSm87A4W4DownK512BStationaryMaximumRegisters == 128U,
      "16-warp resource model changed");

  constexpr auto p1920 =
      kernels::sm87_a4w4_down_k512_bstationary_plan(
          1'920U, 5'120U, 17'408U);
  test.expect(p1920.m_tiles == 15U && p1920.n_tiles == 40U &&
                  p1920.k512_groups == 34U &&
                  p1920.physical_k256_stages == 68U &&
                  p1920.physical_k64_groups == 272U &&
                  p1920.work_tiles == 600U &&
                  p1920.launch_ctas == 15U && p1920.owner_groups == 1U,
              "P1920 plan is not one exact 15-CTA B-wave");
  test.expect(
      kernels::sm87_a4w4_down_k512_bstationary_uses_exact_owner_groups(
          p1920),
      "P1920 did not select the exact-owner B-wave");
  constexpr auto p2176 =
      kernels::sm87_a4w4_down_k512_bstationary_plan(
          2'176U, 5'120U, 17'408U);
  test.expect(p2176.m_tiles == 17U && p2176.n_tiles == 40U &&
                  p2176.work_tiles == 680U &&
                  p2176.launch_ctas == 16U && p2176.owner_groups == 2U,
              "P2176 plan did not isolate the one-owner tail group");
  test.expect(
      !kernels::sm87_a4w4_down_k512_bstationary_uses_exact_owner_groups(
          p2176),
      "P2176 did not select balanced N-major scheduling");
  constexpr auto p4096 =
      kernels::sm87_a4w4_down_k512_bstationary_plan(
          4'096U, 5'120U, 17'408U);
  test.expect(p4096.m_tiles == 32U && p4096.launch_ctas == 16U &&
                  p4096.owner_groups == 2U,
              "P4096 plan is not two complete owner groups");

  test.expect(
      kernels::sm87_a4w4_down_k512_bstationary_plan(
          1'984U, 5'120U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_bstationary_plan(
              4'224U, 5'120U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_bstationary_plan(
              2'048U, 4'992U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_bstationary_plan(
              2'048U, 5'120U, 16'896U)
                  .launch_ctas == 0U,
      "production plan did not enforce padded M128 and fixed N/K");
  test.expect(
      kernels::sm87_a4w4_down_k512_bstationary_test_plan(
          128U, 128U, 512U)
                  .work_tiles == 1U &&
          kernels::sm87_a4w4_down_k512_bstationary_test_plan(
              127U, 128U, 512U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_bstationary_test_plan(
              64U, 64U, 512U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_bstationary_test_plan(
              128U, 128U, 256U)
                  .launch_ctas == 0U,
      "correctness plan did not fail closed on M/N/K tails");

  verify_schedule(test, p1920);
  verify_schedule(test, p2176);
  verify_schedule(test, p4096);
  constexpr auto p1920_last =
      kernels::sm87_a4w4_down_k512_bstationary_tile(
          p1920, 14U, 0U, 39U);
  constexpr auto p2176_remainder =
      kernels::sm87_a4w4_down_k512_bstationary_balanced_tile(
          p2176, 0U, 1U);
  constexpr auto p2176_last =
      kernels::sm87_a4w4_down_k512_bstationary_balanced_tile(
          p2176, 7U, 42U);
  test.expect(p1920_last.valid && p1920_last.m_tile == 14U &&
                  p1920_last.n_tile == 39U,
              "P1920 exact B-wave terminal coordinate changed");
  test.expect(p2176_remainder.valid &&
                  p2176_remainder.m_tile == 16U &&
                  p2176_remainder.n_tile == 0U &&
                  p2176_last.valid && p2176_last.m_tile == 16U &&
                  p2176_last.n_tile == 39U,
              "P2176 balanced remainder distribution changed");
  test.expect(
      !kernels::sm87_a4w4_down_k512_bstationary_tile(
          p2176, 0U, 0U, 0U)
           .valid &&
          !kernels::sm87_a4w4_down_k512_bstationary_balanced_tile(
              p1920, 0U, 0U)
                  .valid,
      "schedule-specific helpers did not fail closed");

  test.expect(
      kernels::query_sm87_a4w4_down_k512_bstationary_resources_cuda(
          nullptr) == static_cast<int>(cudaErrorInvalidValue),
      "resource query did not reject null before CUDA state access");

  if (test.result() == 0) {
    std::cout << "SM87 Down K512 B-stationary standalone contract passed\n";
  }
  return test.result();
}
