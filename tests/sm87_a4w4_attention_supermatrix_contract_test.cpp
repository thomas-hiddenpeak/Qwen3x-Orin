#include "q3x/kernels/sm87_a4w4_attention_supermatrix_cell.h"
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
  test.expect(kernels::kSm87A4W4AttentionCellTileM == 128U &&
                  kernels::kSm87A4W4AttentionCellPanelN == 64U &&
                  kernels::kSm87A4W4AttentionCellPanelsPerCell == 2U &&
                  kernels::kSm87A4W4AttentionCellTileK == 128U,
              "attention supermatrix owns M128 x (N64+N64) x K128");
  test.expect(kernels::kSm87A4W4AttentionCellThreads == 256U &&
                  kernels::kSm87A4W4AttentionCellPersistentCtas == 32U &&
                  kernels::kSm87A4W4AttentionCellCtasPerSm == 2U,
              "attention supermatrix owns a two-CTA persistent launch");
  test.expect(kernels::kSm87A4W4AttentionCellAStages == 2U &&
                  kernels::kSm87A4W4AttentionCellBPairStages == 3U &&
                  kernels::kSm87A4W4AttentionCellSharedBytes == 42'240U &&
                  2U * kernels::kSm87A4W4AttentionCellSharedBytes <=
                      96U * 1'024U,
              "asymmetric A2/B3 ring fits two CTAs per SM");

  constexpr auto linear = kernels::sm87_a4w4_attention_supermatrix_plan(
      2'048U, kernels::kQwen36LinearQkvOutputSize,
      kernels::kQwen36LinearZOutputSize,
      kernels::kQwen36AttentionInputSize);
  test.expect(linear.m_tiles == 16U && linear.first_panels == 160U &&
                  linear.second_panels == 96U &&
                  linear.paired_panels == 96U &&
                  linear.pair_cells == 128U &&
                  linear.k128_groups == 40U &&
                  linear.physical_k64_groups == 80U &&
                  linear.work_cells == 2'048U &&
                  linear.launch_ctas == 32U,
              "P2048 Linear QKV+Z plan fixes the real topology");

  std::array<unsigned int, kernels::kQwen36LinearQkvPanels> qkv_visits{};
  std::array<unsigned int, kernels::kQwen36LinearZPanels> z_visits{};
  for (std::size_t cell = 0U; cell < linear.pair_cells; ++cell) {
    for (std::size_t slot = 0U;
         slot < kernels::kSm87A4W4AttentionCellPanelsPerCell; ++slot) {
      const auto panel =
          kernels::sm87_a4w4_attention_supermatrix_panel(linear, cell,
                                                          slot);
      if (panel.projection == 0U) {
        test.expect(panel.panel < qkv_visits.size(),
                    "QKV panel index stays in bounds");
        if (panel.panel < qkv_visits.size()) {
          ++qkv_visits[panel.panel];
        }
      } else if (panel.projection == 1U) {
        test.expect(panel.panel < z_visits.size(),
                    "Z panel index stays in bounds");
        if (panel.panel < z_visits.size()) {
          ++z_visits[panel.panel];
        }
      } else {
        test.expect(false, "valid cell maps to a valid projection");
      }
    }
  }
  for (const unsigned int count : qkv_visits) {
    test.expect(count == 1U, "every QKV N64 panel is visited once");
  }
  for (const unsigned int count : z_visits) {
    test.expect(count == 1U, "every Z N64 panel is visited once");
  }
  for (std::size_t cell = 0U; cell < 96U; ++cell) {
    const auto first =
        kernels::sm87_a4w4_attention_supermatrix_panel(linear, cell, 0U);
    const auto second =
        kernels::sm87_a4w4_attention_supermatrix_panel(linear, cell, 1U);
    test.expect(first.projection == 0U && first.panel == cell &&
                    second.projection == 1U && second.panel == cell,
                "first 96 cells pair QKV and Z panels sharing A");
  }

  std::array<unsigned int, 2'048U> work_visits{};
  std::array<unsigned int, 32U> cta_work{};
  for (std::size_t cta = 0U; cta < linear.launch_ctas; ++cta) {
    for (std::size_t work = cta; work < linear.work_cells;
         work += linear.launch_ctas) {
      ++work_visits[work];
      ++cta_work[cta];
      const std::size_t cell = work / linear.m_tiles;
      const std::size_t m_tile = work - cell * linear.m_tiles;
      test.expect(m_tile == cta % linear.m_tiles,
                  "persistent mapping keeps one M tile per CTA");
      test.expect(cell % 2U == cta / linear.m_tiles,
                  "persistent mapping keeps one pair-cell parity per CTA");
    }
  }
  for (const unsigned int count : work_visits) {
    test.expect(count == 1U, "persistent mapping visits each work cell once");
  }
  for (const unsigned int count : cta_work) {
    test.expect(count == 64U, "P2048 Linear work is balanced across 32 CTAs");
  }

  // Full Q+K+V reuses this physical cell as Q plus a virtual concatenated
  // K/V panel space. The eventual three-output launcher splits those 32
  // panels back into the independent K and V BF16 ABIs at store time.
  constexpr auto full = kernels::sm87_a4w4_attention_supermatrix_plan(
      2'048U, kernels::kQwen36FullQOutputSize,
      2U * kernels::kQwen36FullKvOutputSize,
      kernels::kQwen36AttentionInputSize);
  test.expect(full.first_panels == 192U && full.second_panels == 32U &&
                  full.pair_cells ==
                      kernels::kQwen36FullAttentionPairCells &&
                  full.work_cells == 1'792U,
              "Full Q+K+V global design has 112 cells per M tile");
  constexpr std::size_t o_n64_panels =
      kernels::kQwen36AttentionOOutputSize /
      kernels::kSm87A4W4AttentionCellPanelN;
  test.expect(o_n64_panels == 80U &&
                  o_n64_panels /
                          kernels::kSm87A4W4AttentionCellPanelsPerCell ==
                      40U &&
                  kernels::kQwen36AttentionOInputSize == 6'144U,
              "O shape remains a dedicated one-output 40-N128-cell family");

  constexpr std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          linear.token_count, linear.input_size);
  constexpr std::size_t qkv_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          linear.first_output_size, linear.input_size);
  constexpr std::size_t z_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          linear.second_output_size, linear.input_size);
  constexpr std::size_t final_a = kernels::sm87_a4w4_consumer_packed_offset(
      linear.token_count - 1U, linear.physical_k64_groups - 1U, 31U,
      linear.physical_k64_groups);
  constexpr std::size_t final_qkv =
      kernels::sm87_a4w4_consumer_packed_offset(
          linear.first_output_size - 1U,
          linear.physical_k64_groups - 1U, 31U,
          linear.physical_k64_groups);
  constexpr std::size_t final_z = kernels::sm87_a4w4_consumer_packed_offset(
      linear.second_output_size - 1U,
      linear.physical_k64_groups - 1U, 31U,
      linear.physical_k64_groups);
  test.expect(final_a + 1U == a_bytes && final_qkv + 1U == qkv_bytes &&
                  final_z + 1U == z_bytes,
              "real-shape final packed addresses match capacities");

  test.expect(kernels::sm87_a4w4_attention_supermatrix_plan(
                  127U, 256U, 256U, 128U)
                      .launch_ctas == 0U &&
                  kernels::sm87_a4w4_attention_supermatrix_plan(
                      128U, 255U, 256U, 128U)
                      .launch_ctas == 0U &&
                  kernels::sm87_a4w4_attention_supermatrix_plan(
                      128U, 256U, 192U, 128U)
                      .launch_ctas == 0U &&
                  kernels::sm87_a4w4_attention_supermatrix_plan(
                      128U, 256U, 256U, 64U)
                      .launch_ctas == 0U,
              "M/N/K tails and an odd unpaired panel fail closed");
  test.expect(
      kernels::query_sm87_a4w4_linear_qkv_z_supermatrix_resources_cuda(
          nullptr) == static_cast<int>(cudaErrorInvalidValue),
      "resource query rejects null without touching CUDA state");

  if (test.failures == 0) {
    std::cout << "SM87 attention supermatrix host contract passed\n";
  }
  return test.failures == 0 ? 0 : 1;
}
