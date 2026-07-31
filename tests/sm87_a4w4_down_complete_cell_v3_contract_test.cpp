#include "q3x/kernels/sm87_a4w4_down_complete_cell_v3.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime_api.h>

#include <array>
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
  test.expect(kernels::kSm87A4W4DownCellV3TileM == 128U &&
                  kernels::kSm87A4W4DownCellV3TileN == 128U &&
                  kernels::kSm87A4W4DownCellV3TileK == 128U,
              "Down complete cell owns M128N128K128");
  test.expect(kernels::kSm87A4W4DownCellV3Threads == 256U &&
                  kernels::kSm87A4W4DownCellV3PersistentCtas == 32U &&
                  kernels::kSm87A4W4DownCellV3CtasPerSm == 2U,
              "Down complete cell owns its two-CTA persistent launch");
  test.expect(kernels::kSm87A4W4DownCellV3WarpTileM == 32U &&
                  kernels::kSm87A4W4DownCellV3WarpTileN == 64U &&
                  kernels::kSm87A4W4DownCellV3WarpRows == 4U &&
                  kernels::kSm87A4W4DownCellV3WarpColumns == 2U &&
                  kernels::kSm87A4W4DownCellV3WarpRows *
                          kernels::kSm87A4W4DownCellV3WarpColumns ==
                      kernels::kSm87A4W4DownCellV3Warps,
              "eight warps form the complete 4x2 M32N64 ownership grid");
  test.expect(kernels::kSm87A4W4DownCellV3AStages == 2U &&
                  kernels::kSm87A4W4DownCellV3BStages == 3U &&
                  kernels::kSm87A4W4DownCellV3SharedBytes == 42'240U &&
                  2U * kernels::kSm87A4W4DownCellV3SharedBytes <=
                      96U * 1'024U,
              "asymmetric A2/B3 ring fits two CTA per SM");

  constexpr std::size_t m_count = 2'048U;
  constexpr std::size_t n_count = 5'120U;
  constexpr std::size_t k_count = 17'408U;
  constexpr auto plan = kernels::sm87_a4w4_down_complete_cell_v3_plan(
      m_count, n_count, k_count);
  test.expect(plan.m_tiles == 16U && plan.n_tiles == 40U &&
                  plan.k128_groups == 136U &&
                  plan.physical_k64_groups == 272U &&
                  plan.work_tiles == 640U && plan.launch_ctas == 32U,
              "real Down plan covers 640 cells with 32 CTAs");

  std::array<unsigned int, 640U> visits{};
  std::array<unsigned int, 32U> cta_work{};
  for (std::size_t cta = 0U; cta < plan.launch_ctas; ++cta) {
    for (std::size_t work = cta; work < plan.work_tiles;
         work += plan.launch_ctas) {
      ++visits[work];
      ++cta_work[cta];
      const std::size_t n_tile = work / plan.m_tiles;
      const std::size_t m_tile = work - n_tile * plan.m_tiles;
      test.expect(m_tile == cta % plan.m_tiles,
                  "real mapping keeps one M tile per CTA");
      test.expect(n_tile % 2U == cta / plan.m_tiles,
                  "real mapping keeps one N parity per CTA");
    }
  }
  for (const unsigned int count : visits) {
    test.expect(count == 1U, "persistent mapping visits each cell once");
  }
  for (const unsigned int count : cta_work) {
    test.expect(count == 20U, "real mapping balances twenty cells per CTA");
  }

  constexpr std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count, k_count);
  constexpr std::size_t b_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(n_count, k_count);
  constexpr std::size_t a_scales =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(m_count,
                                                               k_count);
  constexpr std::size_t b_scales =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(n_count,
                                                               k_count);
  constexpr std::size_t final_a_byte =
      kernels::sm87_a4w4_consumer_packed_offset(
          m_count - 1U, plan.physical_k64_groups - 1U, 31U,
          plan.physical_k64_groups);
  constexpr std::size_t final_b_byte =
      kernels::sm87_a4w4_consumer_packed_offset(
          n_count - 1U, plan.physical_k64_groups - 1U, 31U,
          plan.physical_k64_groups);
  constexpr std::size_t final_a_scale =
      kernels::sm87_a4w4_consumer_k128_scale_offset(
          m_count - 1U, plan.k128_groups - 1U, plan.k128_groups);
  constexpr std::size_t final_b_scale =
      kernels::sm87_a4w4_consumer_k128_scale_offset(
          n_count - 1U, plan.k128_groups - 1U, plan.k128_groups);
  test.expect(final_a_byte + 1U == a_bytes &&
                  final_b_byte + 1U == b_bytes &&
                  final_a_scale + 1U == a_scales &&
                  final_b_scale + 1U == b_scales,
              "real-shape final packed and scale addresses match capacity");
  test.expect((m_count - 1U) * n_count + (n_count - 1U) ==
                  m_count * n_count - 1U,
              "real-shape final BF16 output address is in capacity");

  const auto one = kernels::sm87_a4w4_down_complete_cell_v3_plan(
      128U, 128U, 128U);
  test.expect(one.work_tiles == 1U && one.k128_groups == 1U &&
                  one.launch_ctas == 1U,
              "one logical K128 group covers ring fill and drain");
  test.expect(kernels::sm87_a4w4_down_complete_cell_v3_plan(
                  127U, 128U, 128U)
                      .launch_ctas == 0U &&
                  kernels::sm87_a4w4_down_complete_cell_v3_plan(
                      128U, 64U, 128U)
                      .launch_ctas == 0U &&
                  kernels::sm87_a4w4_down_complete_cell_v3_plan(
                      128U, 128U, 64U)
                      .launch_ctas == 0U,
              "M/N/K tails fail closed");
  test.expect(
      kernels::query_sm87_a4w4_down_complete_cell_v3_resources_cuda(
          nullptr) == static_cast<int>(cudaErrorInvalidValue),
      "resource query rejects null without touching CUDA state");

  if (test.failures == 0) {
    std::cout << "SM87 Down complete-cell v3 host contract passed\n";
  }
  return test.failures == 0 ? 0 : 1;
}
