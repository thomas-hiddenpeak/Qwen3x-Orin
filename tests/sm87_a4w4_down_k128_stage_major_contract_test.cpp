#include "q3x/kernels/sm87_a4w4_down_k128_stage_major.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iostream>

namespace {

namespace kernels = q3x::kernels;

struct TestContext final {
  int failures{};

  void expect(const bool condition, const char* const description) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << description << '\n';
    }
  }
};

}  // namespace

int main() {
  TestContext test;
  test.expect(kernels::kSm87A4W4DownK128StageMajorTileM == 128U &&
                  kernels::kSm87A4W4DownK128StageMajorTileN == 256U &&
                  kernels::kSm87A4W4DownK128StageMajorTileK == 128U,
              "Down owns the M128N256K128 tile");
  test.expect(kernels::kSm87A4W4DownK128StageMajorThreads == 256U &&
                  kernels::kSm87A4W4DownK128StageMajorPersistentCtas ==
                      16U,
              "Down owns its 256-thread persistent-16 launch");
  test.expect(kernels::kSm87A4W4DownK128StageMajorSharedBytes == 46'464U,
              "two full slots plus one upper-A scratch fit 48 KiB");

  const auto down = kernels::sm87_a4w4_down_k128_stage_major_plan(
      2'048U, 5'120U, 17'408U);
  test.expect(down.m_tiles == 16U && down.n_stripes == 20U &&
                  down.k128_groups == 136U &&
                  down.physical_k64_groups == 272U &&
                  down.work_tiles == 320U && down.launch_ctas == 16U,
              "real Down plan is shape-specific and exact");

  const auto single = kernels::sm87_a4w4_down_k128_stage_major_plan(
      128U, 256U, 256U);
  test.expect(single.m_tiles == 1U && single.n_stripes == 1U &&
                  single.k128_groups == 2U && single.launch_ctas == 1U,
              "minimal exact tile has one CTA");
  test.expect(kernels::sm87_a4w4_down_k128_stage_major_plan(
                  127U, 256U, 256U)
                      .launch_ctas == 0U &&
                  kernels::sm87_a4w4_down_k128_stage_major_plan(
                      128U, 128U, 256U)
                      .launch_ctas == 0U &&
                  kernels::sm87_a4w4_down_k128_stage_major_plan(
                      128U, 256U, 192U)
                      .launch_ctas == 0U,
              "M/N/K tails fail closed");
  test.expect(
      kernels::query_sm87_a4w4_down_k128_stage_major_resources_cuda(
          nullptr) == static_cast<int>(cudaErrorInvalidValue),
      "resource query rejects null before touching CUDA state");

  if (test.failures == 0) {
    std::cout << "SM87 Down K128 stage-major host contract passed\n";
  }
  return test.failures == 0 ? 0 : 1;
}
