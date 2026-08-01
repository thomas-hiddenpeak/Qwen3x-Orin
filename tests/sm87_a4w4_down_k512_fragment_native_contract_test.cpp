#include "q3x/kernels/sm87_a4w4_down_k512_fragment_native.h"

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

void verify_b_layout_bijection(Test& test) {
  namespace kernels = q3x::kernels;
  constexpr std::size_t panel_bytes = 128U * 64U / 2U;
  std::vector<unsigned int> logical_visits(panel_bytes, 0U);
  std::vector<unsigned int> physical_visits(panel_bytes, 0U);
  for (std::size_t warp = 0U;
       warp < kernels::kSm87A4W4DownK512FragmentWarps; ++warp) {
    for (std::size_t lane = 0U; lane < 32U; ++lane) {
      const std::size_t vector_offset =
          kernels::sm87_a4w4_down_k512_fragment_b_vector_offset(
              0U, 0U, 0U, warp, lane, 1U);
      test.expect(vector_offset % 16U == 0U,
                  "B lane vector is not aligned to 16 bytes");
      for (std::size_t word = 0U; word < 4U; ++word) {
        const auto coordinate =
            kernels::sm87_a4w4_down_k512_fragment_b_word_coordinate(
                warp, lane, word);
        test.expect(coordinate.valid && coordinate.n < 128U &&
                        coordinate.byte_in_k64 + 4U <= 32U,
                    "B word coordinate escaped one N128/K64 panel");
        for (std::size_t byte = 0U; byte < 4U; ++byte) {
          ++logical_visits[coordinate.n * 32U +
                           coordinate.byte_in_k64 + byte];
          ++physical_visits[vector_offset + word * 4U + byte];
        }
      }
    }
  }
  for (const unsigned int visits : logical_visits) {
    test.expect(visits == 1U,
                "fragment layout is not a logical byte bijection");
  }
  for (const unsigned int visits : physical_visits) {
    test.expect(visits == 1U,
                "fragment layout does not densely fill the B panel");
  }
}

void verify_schedule(
    Test& test,
    const q3x::kernels::Sm87A4W4DownK512FragmentPlan& plan) {
  namespace kernels = q3x::kernels;
  std::vector<unsigned int> visits(plan.work_tiles, 0U);
  for (std::size_t group = 0U; group < plan.owner_groups; ++group) {
    for (std::size_t cta = 0U;
         cta < kernels::kSm87A4W4DownK512FragmentBWaveCtas; ++cta) {
      for (std::size_t n = 0U; n < plan.n_tiles; ++n) {
        const auto tile =
            kernels::sm87_a4w4_down_k512_fragment_native_tile(
                plan, cta, group, n);
        if (!tile.valid) {
          continue;
        }
        test.expect(tile.n_tile == n,
                    "CTA escaped the common B-stationary N wave");
        ++visits[tile.n_tile * plan.m_tiles + tile.m_tile];
      }
    }
  }
  for (const unsigned int count : visits) {
    test.expect(count == 1U, "grouped schedule did not visit a cell once");
  }
  if (plan.launch_ctas >= 2U && plan.n_tiles != 0U) {
    const auto even =
        kernels::sm87_a4w4_down_k512_fragment_native_tile(
            plan, 0U, 0U, plan.n_tiles - 1U);
    const auto odd =
        kernels::sm87_a4w4_down_k512_fragment_native_tile(
            plan, 1U, 0U, plan.n_tiles - 1U);
    test.expect(even.valid && odd.valid &&
                    even.m_tile + 1U == odd.m_tile &&
                    even.n_tile == odd.n_tile,
                "paired M64 CTAs did not share one B panel wave");
  }
}

}  // namespace

int main() {
  namespace kernels = q3x::kernels;
  Test test;

  test.expect(
      kernels::kSm87A4W4DownK512FragmentTileM == 64U &&
          kernels::kSm87A4W4DownK512FragmentTileN == 128U &&
          kernels::kSm87A4W4DownK512FragmentWarpTileM == 64U &&
          kernels::kSm87A4W4DownK512FragmentWarpTileN == 16U &&
          kernels::kSm87A4W4DownK512FragmentThreads == 256U &&
          kernels::kSm87A4W4DownK512FragmentWarps == 8U,
      "M64N128 CTA or M64N16 warp ownership changed");
  test.expect(
      kernels::kSm87A4W4DownK512FragmentStageK == 256U &&
          kernels::kSm87A4W4DownK512FragmentScaleK == 512U &&
          kernels::kSm87A4W4DownK512FragmentStages == 3U &&
          kernels::kSm87A4W4DownK512FragmentDynamicSharedBytes ==
              24'960U &&
          kernels::kSm87A4W4DownK512FragmentCtasPerSm == 2U &&
          kernels::kSm87A4W4DownK512FragmentMaximumRegisters == 128U,
      "A-only three-stage ring or two-CTA resource contract changed");
  test.expect(
      kernels::kSm87A4W4DownK512FragmentLaneVectorBytes == 16U &&
          kernels::kSm87A4W4DownK512FragmentN8PerWarp == 2U &&
          kernels::kSm87A4W4DownK512FragmentM16PerWarp == 4U,
      "one LDG.128 -> two N8 -> four M16 reuse contract changed");

  verify_b_layout_bijection(test);
  test.expect(
      kernels::sm87_a4w4_down_k512_fragment_b_capacity_bytes(
          5'120U, 17'408U) ==
              kernels::sm87_a4w4_down_k512_packed_capacity_bytes(
                  5'120U, 17'408U) &&
          kernels::sm87_a4w4_down_k512_fragment_b_capacity_bytes(
              128U, 512U) == 32'768U &&
          kernels::sm87_a4w4_down_k512_fragment_b_capacity_bytes(
              64U, 512U) == 0U,
      "fragment-native B layout changed payload byte count");

  constexpr auto p1920 =
      kernels::sm87_a4w4_down_k512_fragment_native_plan(
          1'920U, 5'120U, 17'408U);
  test.expect(p1920.m_tiles == 30U && p1920.n_tiles == 40U &&
                  p1920.k512_groups == 34U &&
                  p1920.physical_k256_stages == 68U &&
                  p1920.physical_k64_groups == 272U &&
                  p1920.work_tiles == 1'200U &&
                  p1920.launch_ctas == 1'200U &&
                  p1920.owner_groups == 1U,
              "P1920 plan is not one 30-CTA B wave");
  constexpr auto p2176 =
      kernels::sm87_a4w4_down_k512_fragment_native_plan(
          2'176U, 5'120U, 17'408U);
  test.expect(p2176.m_tiles == 34U && p2176.n_tiles == 40U &&
                  p2176.launch_ctas == 1'360U &&
                  p2176.owner_groups == 2U,
              "P2176 plan did not isolate its paired-M64 tail group");
  constexpr auto p4096 =
      kernels::sm87_a4w4_down_k512_fragment_native_plan(
          4'096U, 5'120U, 17'408U);
  test.expect(p4096.m_tiles == 64U && p4096.launch_ctas == 2'560U &&
                  p4096.owner_groups == 2U,
              "P4096 plan is not two complete 32-CTA owner groups");
  verify_schedule(test, p1920);
  verify_schedule(test, p2176);
  verify_schedule(test, p4096);

  test.expect(
      kernels::sm87_a4w4_down_k512_fragment_native_plan(
          1'984U, 5'120U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_fragment_native_plan(
              2'048U, 4'992U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_fragment_native_test_plan(
              64U, 128U, 512U)
                  .work_tiles == 1U &&
          kernels::sm87_a4w4_down_k512_fragment_native_test_plan(
              63U, 128U, 512U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_fragment_native_test_plan(
              64U, 64U, 512U)
                  .launch_ctas == 0U,
      "production/test shapes did not fail closed");

  test.expect(
      kernels::query_sm87_a4w4_down_k512_fragment_native_resources_cuda(
          nullptr) == static_cast<int>(cudaErrorInvalidValue),
      "resource query did not reject null before CUDA state access");

  if (test.result() == 0) {
    std::cout << "SM87 Down K512 fragment-native standalone contract passed\n";
  }
  return test.result();
}
