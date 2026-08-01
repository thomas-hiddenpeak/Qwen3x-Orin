#include "q3x/kernels/sm87_a4w4_gateup_k512_macrocell.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

class Test final {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      std::cerr << message << '\n';
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
  using namespace q3x::kernels;
  Test test;

  test.expect(kSm87A4W4GateUpK512MacroThreads == 512U &&
                  kSm87A4W4GateUpK512MacroWarps == 16U &&
                  kSm87A4W4GateUpK512MacroProjectionWarps == 8U &&
                  kSm87A4W4GateUpK512MacroCtasPerSm == 1U,
              "512-thread/dual-crew ownership contract failed");
  test.expect(kSm87A4W4GateUpK512MacroAStageBytes == 8'192U &&
                  kSm87A4W4GateUpK512MacroBStageBytes == 16'384U &&
                  kSm87A4W4GateUpK512MacroStageBytes == 40'960U &&
                  kSm87A4W4GateUpK512MacroScaleSlotBytes == 640U &&
                  kSm87A4W4GateUpK512MacroSharedBytes == 83'200U,
              "K256 raw-stage/dynamic-shared contract failed");

  constexpr auto full = sm87_a4w4_gateup_k512_macro_plan(
      2'048U, 17'408U, 5'120U, 0U, 17'408U);
  constexpr auto primary = sm87_a4w4_gateup_k512_macro_plan(
      2'048U, 17'408U, 5'120U, 0U, 12'288U);
  constexpr auto secondary = sm87_a4w4_gateup_k512_macro_plan(
      2'048U, 17'408U, 5'120U, 12'288U, 5'120U);
  test.expect(sm87_a4w4_gateup_k512_macro_is_model_plan(full) &&
                  full.m_tiles == 32U && full.n_tiles == 136U &&
                  full.k512_groups == 10U &&
                  full.physical_k256_groups == 20U &&
                  full.physical_k64_groups == 80U &&
                  full.work_cells == 4'352U && full.launch_ctas == 16U,
              "real full-N plan failed");
  test.expect(sm87_a4w4_gateup_k512_macro_is_model_plan(primary) &&
                  primary.n_start == 0U &&
                  primary.n_count == 12'288U &&
                  primary.n_tiles == 96U &&
                  primary.work_cells == 3'072U,
              "primary workspace window plan failed");
  test.expect(sm87_a4w4_gateup_k512_macro_is_model_plan(secondary) &&
                  secondary.n_start == 12'288U &&
                  secondary.n_count == 5'120U &&
                  secondary.n_tiles == 40U &&
                  secondary.work_cells == 1'280U,
              "secondary workspace window plan failed");

  constexpr std::size_t a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(2'048U, 5'120U);
  constexpr std::size_t b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(17'408U, 5'120U);
  constexpr std::size_t a_scales =
      sm87_a4w4_gateup_k512_macro_scale_capacity_elements(
          2'048U, 5'120U);
  constexpr std::size_t b_scales =
      sm87_a4w4_gateup_k512_macro_scale_capacity_elements(
          17'408U, 5'120U);
  test.expect(a_bytes == 5'242'880U && b_bytes == 44'564'480U &&
                  a_scales == 20'480U && b_scales == 174'080U,
              "full input capacity contract failed");
  test.expect(
      sm87_a4w4_consumer_packed_offset(
          2'047U, full.physical_k64_groups - 1U, 31U,
          full.physical_k64_groups) == a_bytes - 1U &&
          sm87_a4w4_consumer_packed_offset(
              17'407U, full.physical_k64_groups - 1U, 31U,
              full.physical_k64_groups) == b_bytes - 1U &&
          sm87_a4w4_gateup_k512_macro_scale_offset(
              2'047U, full.k512_groups - 1U,
              full.k512_groups) == a_scales - 1U &&
          sm87_a4w4_gateup_k512_macro_scale_offset(
              17'407U, full.k512_groups - 1U,
              full.k512_groups) == b_scales - 1U,
      "consumer packed/K512 scale last-offset contract failed");

  // Exercise the scale map as a bijection over a partial final outer block.
  constexpr std::size_t outer = 129U;
  constexpr std::size_t logical_k = 1'024U;
  constexpr std::size_t groups = logical_k / 512U;
  const std::size_t scale_capacity =
      sm87_a4w4_gateup_k512_macro_scale_capacity_elements(outer,
                                                           logical_k);
  std::vector<std::uint8_t> seen(scale_capacity, 0U);
  for (std::size_t row = 0U; row < outer; ++row) {
    for (std::size_t group = 0U; group < groups; ++group) {
      const std::size_t offset =
          sm87_a4w4_gateup_k512_macro_scale_offset(row, group,
                                                   groups);
      test.expect(offset < scale_capacity,
                  "K512 scale offset exceeded capacity");
      if (offset < scale_capacity) {
        test.expect(seen[offset] == 0U,
                    "K512 scale offset alias detected");
        seen[offset] = 1U;
      }
    }
  }

  test.expect(
      sm87_a4w4_gateup_k512_macro_plan(
          63U, 17'408U, 5'120U, 0U, 12'288U)
              .launch_ctas == 0U &&
          sm87_a4w4_gateup_k512_macro_plan(
              64U, 17'408U, 5'120U, 64U, 12'288U)
                  .launch_ctas == 0U &&
          sm87_a4w4_gateup_k512_macro_plan(
              64U, 17'408U, 5'120U, 0U, 12'224U)
                  .launch_ctas == 0U &&
          sm87_a4w4_gateup_k512_macro_plan(
              64U, 17'408U, 5'120U, 12'288U, 5'248U)
                  .launch_ctas == 0U &&
          sm87_a4w4_gateup_k512_macro_plan(
              64U, 17'408U, 4'864U, 0U, 12'288U)
                  .launch_ctas == 0U,
      "M/N-window/K tail rejection failed");

  test.expect(
      query_sm87_a4w4_gateup_k512_macrocell_resources_cuda(nullptr) ==
              static_cast<int>(cudaErrorInvalidValue) &&
          launch_sm87_a4w4_gateup_k512_macrocell_bf16_cuda(
              nullptr, a_bytes, nullptr, a_scales, nullptr, b_bytes,
              nullptr, b_scales, nullptr, b_bytes, nullptr, b_scales,
              2'048U, 17'408U, 5'120U, 0U, 12'288U, nullptr,
              12'288U, 2'048U * 12'288U) ==
              static_cast<int>(cudaErrorInvalidValue),
      "null pointer fail-closed contract failed");

  auto* volatile query =
      &query_sm87_a4w4_gateup_k512_macrocell_resources_cuda;
  auto* volatile production =
      &launch_sm87_a4w4_gateup_k512_macrocell_bf16_cuda;
  auto* volatile correctness =
      &launch_sm87_a4w4_gateup_k512_macrocell_test_bf16_cuda;
  test.expect(query != nullptr && production != nullptr &&
                  correctness != nullptr,
              "link surface contract failed");

  if (test.result() == 0) {
    std::cout << "SM87 Gate+Up K512 macrocell contract passed\n";
  }
  return test.result();
}
