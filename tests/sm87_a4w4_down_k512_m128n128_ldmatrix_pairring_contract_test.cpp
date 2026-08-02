#include "q3x/kernels/sm87_a4w4_down_k512_m128n128_ldmatrix_pairring.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iostream>

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
  test.expect(kernels::kSm87A4W4DownK512TileM == 128U &&
                  kernels::kSm87A4W4DownK512TileN == 128U &&
                  kernels::kSm87A4W4DownK512Threads == 256U &&
                  kernels::kSm87A4W4DownK512Warps == 8U,
              "M128N128/256-thread/8-warp shape changed");
  test.expect(kernels::kSm87A4W4DownK512Stages == 4U &&
                  kernels::kSm87A4W4DownK512StagesPerScale == 2U &&
                  kernels::kSm87A4W4DownK512DynamicSharedBytes ==
                      128U * 1'024U,
              "four-K256-stage 128-KiB ring changed");
  test.expect(
      kernels::kSm87A4W4DownK512M128N128LdmatrixPairringLdmatrixX4PerGroup ==
              16U &&
          kernels::kSm87A4W4DownK512M128N128LdmatrixPairringLdmatrixX2PerGroup ==
              64U &&
          kernels::kSm87A4W4DownK512M128N128LdmatrixPairringImmaPerGroup ==
              128U,
      "LDSM/IMMA per-K512 structural contract changed");
  test.expect(kernels::sm87_a4w4_down_k512_test_plan(
                  128U, 128U, 512U)
                      .k512_groups == 1U &&
                  kernels::sm87_a4w4_down_k512_test_plan(
                      256U, 256U, 1'024U)
                          .k512_groups == 2U &&
                  kernels::sm87_a4w4_down_k512_test_plan(
                      256U, 256U, 17'408U)
                          .k512_groups == 34U,
              "odd/even and production-K complete-cell plans changed");
  test.expect(
      kernels::query_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_resources_cuda(
          nullptr) == static_cast<int>(cudaErrorInvalidValue),
      "resource query did not reject null before CUDA access");

  if (test.result() == 0) {
    std::cout << "Down M128N128 LDSM pair-ring contract passed\n";
  }
  return test.result();
}
