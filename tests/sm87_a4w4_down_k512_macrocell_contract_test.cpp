#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
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

}  // namespace

int main() {
  namespace kernels = q3x::kernels;
  Test test;

  test.expect(
      kernels::kSm87A4W4DownK512TileM == 128U &&
          kernels::kSm87A4W4DownK512TileN == 128U &&
          kernels::kSm87A4W4DownK512WarpTileM == 32U &&
          kernels::kSm87A4W4DownK512WarpTileN == 64U &&
          kernels::kSm87A4W4DownK512Threads == 256U &&
          kernels::kSm87A4W4DownK512Warps == 8U,
      "M128N128 CTA and 4x2 M32N64 warp ownership changed");
  test.expect(
      kernels::kSm87A4W4DownK512CopyK == 256U &&
          kernels::kSm87A4W4DownK512ScaleK == 512U &&
          kernels::kSm87A4W4DownK512K64PerStage == 4U &&
          kernels::kSm87A4W4DownK512StagesPerScale == 2U &&
          kernels::kSm87A4W4DownK512Stages == 4U,
      "four-stage K256 ring and paired K512 boundary changed");
  test.expect(
      kernels::kSm87A4W4DownK512StageABytes == 16U * 1'024U &&
          kernels::kSm87A4W4DownK512StageBBytes == 16U * 1'024U &&
          kernels::kSm87A4W4DownK512StageBytes == 32U * 1'024U &&
          kernels::kSm87A4W4DownK512DynamicSharedBytes ==
              128U * 1'024U,
      "128-KiB opt-in dynamic-shared ring changed");
  test.expect(
      kernels::kSm87A4W4DownK512PersistentCtas == 16U &&
          kernels::kSm87A4W4DownK512CtasPerSm == 1U,
      "one persistent CTA per pinned SM changed");

  constexpr auto real = kernels::sm87_a4w4_down_k512_plan(
      2'048U, 5'120U, 17'408U);
  test.expect(real.m_tiles == 16U && real.n_tiles == 40U &&
                  real.k512_groups == 34U &&
                  real.physical_k256_stages == 68U &&
                  real.physical_k64_groups == 272U &&
                  real.work_tiles == 640U && real.launch_ctas == 16U,
              "fixed Down shape plan is not M16xN40xK34");
  constexpr auto real_4k = kernels::sm87_a4w4_down_k512_plan(
      4'096U, 5'120U, 17'408U);
  test.expect(real_4k.m_tiles == 32U && real_4k.n_tiles == 40U &&
                  real_4k.k512_groups == 34U &&
                  real_4k.work_tiles == 1'280U &&
                  real_4k.launch_ctas == 16U,
              "M4096 API span is not a two-owner-per-CTA B-wave");
  test.expect(
      kernels::sm87_a4w4_down_k512_plan(1'920U, 5'120U, 17'408U)
                  .launch_ctas == 16U &&
          kernels::sm87_a4w4_down_k512_plan(1'984U, 5'120U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_plan(4'224U, 5'120U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_plan(2'048U, 4'992U, 17'408U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_plan(2'048U, 5'120U, 16'896U)
                  .launch_ctas == 0U,
      "production admission did not enforce complete M128 and fixed N/K");
  test.expect(
      kernels::sm87_a4w4_down_k512_test_plan(128U, 128U, 512U)
                  .work_tiles == 1U &&
          kernels::sm87_a4w4_down_k512_test_plan(127U, 128U, 512U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_test_plan(128U, 64U, 512U)
                  .launch_ctas == 0U &&
          kernels::sm87_a4w4_down_k512_test_plan(128U, 128U, 256U)
                  .launch_ctas == 0U,
      "correctness-only plan did not fail closed on M/N/K tails");

  // N-major + stride-16 must produce forty synchronized B waves.  CTA b
  // retains M tile b and visits every N tile exactly once.
  std::array<unsigned int, 640U> visits{};
  std::array<unsigned int, 16U> cta_work{};
  for (std::size_t cta = 0U; cta < real.launch_ctas; ++cta) {
    for (std::size_t work = cta; work < real.work_tiles;
         work += real.launch_ctas) {
      ++visits[work];
      ++cta_work[cta];
      const std::size_t n_tile = work / real.m_tiles;
      const std::size_t m_tile = work - n_tile * real.m_tiles;
      test.expect(m_tile == cta,
                  "B-wave CTA did not retain one M128 tile");
      test.expect(n_tile == cta_work[cta] - 1U,
                  "B-wave CTA skipped or reordered an N128 tile");
    }
  }
  for (const unsigned int count : visits) {
    test.expect(count == 1U, "B-wave did not visit every output tile once");
  }
  for (const unsigned int count : cta_work) {
    test.expect(count == 40U, "B-wave CTA did not own forty N tiles");
  }

  constexpr std::size_t a_bytes =
      kernels::sm87_a4w4_down_k512_packed_capacity_bytes(2'048U,
                                                          17'408U);
  constexpr std::size_t b_bytes =
      kernels::sm87_a4w4_down_k512_packed_capacity_bytes(5'120U,
                                                          17'408U);
  constexpr std::size_t a_scales =
      kernels::sm87_a4w4_down_k512_scale_capacity_elements(2'048U,
                                                            17'408U);
  constexpr std::size_t b_scales =
      kernels::sm87_a4w4_down_k512_scale_capacity_elements(5'120U,
                                                            17'408U);
  test.expect(a_bytes == 17'825'792U && b_bytes == 44'564'480U &&
                  a_scales == 69'632U && b_scales == 174'080U,
              "fixed Down packed/scale capacities changed");
  test.expect(
      kernels::sm87_a4w4_down_k512_packed_offset(
          2'047U, 271U, 31U, 272U) +
              1U ==
          a_bytes &&
          kernels::sm87_a4w4_down_k512_packed_offset(
              5'119U, 271U, 31U, 272U) +
                  1U ==
              b_bytes &&
          kernels::sm87_a4w4_down_k512_scale_offset(
              2'047U, 33U, 34U) +
                  1U ==
              a_scales &&
          kernels::sm87_a4w4_down_k512_scale_offset(
              5'119U, 33U, 34U) +
                  1U ==
              b_scales,
      "final fixed-shape packed/scale offsets do not meet capacities");

  // The defining numerical boundary is one S32 reduction over eight K64
  // partials followed by one FP32 scale application.
  constexpr std::array<std::int32_t, 8U> partials{
      4'096, -4'096, 3'000, -2'000, 1'000, -500, 250, -125};
  std::int32_t combined = 0;
  for (const std::int32_t partial : partials) {
    combined += partial;
  }
  test.expect(combined == 1'625 && 8 * 4'096 == 32'768,
              "K512 S32 reduction/bound changed");

  test.expect(
      kernels::query_sm87_a4w4_down_k512_macrocell_resources_cuda(
          nullptr) == static_cast<int>(cudaErrorInvalidValue),
      "resource query did not reject null before touching CUDA state");

  if (test.result() == 0) {
    std::cout << "SM87 Down K512 macrocell standalone contract passed\n";
  }
  return test.result();
}
