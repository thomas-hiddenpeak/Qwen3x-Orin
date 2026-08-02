#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

namespace kernels = q3x::kernels;

[[nodiscard]] bool check_real_layout() {
  constexpr std::size_t m = 2'048U;
  constexpr std::size_t n = 17'408U;
  constexpr std::size_t k = 5'120U;
  constexpr auto plan =
      kernels::sm87_a4w4_gateup_down_edge_plan(m, m, n, k);
  constexpr std::size_t a_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(m, k);
  constexpr std::size_t a_scales =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(m, k);
  constexpr std::size_t b_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(n, k);
  constexpr std::size_t b_scales =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(n, k);
  constexpr std::size_t output_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(m, n);
  constexpr std::size_t output_scales =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(m, n);

  return plan.logical_token_count == m && plan.launch_token_count == m &&
         plan.m_tiles == 32U && plan.edge_groups == 34U &&
         plan.compute_cells_per_edge == 4U &&
         plan.input_k512_groups == 10U &&
         plan.input_physical_k64_groups == 80U &&
         plan.output_physical_k64_groups == 272U &&
         plan.work_edge_cells == 1'088U && plan.launch_ctas == 16U &&
         plan.base_waves == 2U && plan.residual_m_tiles == 0U &&
         plan.residual_edge_cells == 0U &&
         a_bytes == 5'242'880U && a_scales == 20'480U &&
         b_bytes == 44'564'480U && b_scales == 174'080U &&
         output_bytes == 17'825'792U && output_scales == 69'632U &&
         kernels::sm87_a4w4_gateup_down_edge_packed_offset(
             m - 1U, plan.output_physical_k64_groups - 1U, 31U,
             plan.output_physical_k64_groups) == output_bytes - 1U &&
         kernels::sm87_a4w4_gateup_down_edge_scale_offset(
             m - 1U, plan.edge_groups - 1U, plan.edge_groups) ==
             output_scales - 1U;
}

[[nodiscard]] bool check_scheduler() {
  constexpr auto p128 = kernels::sm87_a4w4_gateup_down_edge_plan(
      128U, 128U, 17'408U, 5'120U);
  constexpr auto p512 = kernels::sm87_a4w4_gateup_down_edge_plan(
      512U, 512U, 17'408U, 5'120U);
  constexpr auto p1920 = kernels::sm87_a4w4_gateup_down_edge_plan(
      1'920U, 1'920U, 17'408U, 5'120U);
  constexpr auto p2048 = kernels::sm87_a4w4_gateup_down_edge_plan(
      2'048U, 2'048U, 17'408U, 5'120U);
  constexpr auto p2176 = kernels::sm87_a4w4_gateup_down_edge_plan(
      2'148U, 2'176U, 17'408U, 5'120U);
  constexpr auto p4096 = kernels::sm87_a4w4_gateup_down_edge_plan(
      4'096U, 4'096U, 17'408U, 5'120U);
  constexpr auto residual_test =
      kernels::sm87_a4w4_gateup_down_edge_test_plan(
          129U, 256U, 2'048U, 1'024U, 3U);

  return p128.m_tiles == 2U && p128.launch_ctas == 16U &&
         p128.base_waves == 0U && p128.residual_m_tiles == 2U &&
         p128.residual_edge_cells == 68U &&
         p512.m_tiles == 8U && p512.launch_ctas == 16U &&
         p512.base_waves == 0U && p512.residual_m_tiles == 8U &&
         p512.residual_edge_cells == 272U &&
         p1920.m_tiles == 30U && p1920.launch_ctas == 16U &&
         p1920.base_waves == 1U && p1920.residual_m_tiles == 14U &&
         p1920.residual_edge_cells == 476U &&
         p2048.m_tiles == 32U && p2048.launch_ctas == 16U &&
         p2048.base_waves == 2U && p2048.residual_m_tiles == 0U &&
         p2176.m_tiles == 34U && p2176.launch_ctas == 16U &&
         p2176.base_waves == 2U && p2176.residual_m_tiles == 2U &&
         p2176.residual_edge_cells == 68U &&
         p4096.m_tiles == 64U && p4096.launch_ctas == 16U &&
         p4096.base_waves == 4U && p4096.residual_m_tiles == 0U &&
         residual_test.m_tiles == 4U && residual_test.edge_groups == 4U &&
         residual_test.work_edge_cells == 16U &&
         residual_test.launch_ctas == 3U &&
         residual_test.base_waves == 1U &&
         residual_test.residual_m_tiles == 1U &&
         residual_test.residual_edge_cells == 4U;
}

[[nodiscard]] bool check_structure() {
  return kernels::kSm87A4W4GateUpDownEdgeThreads == 512U &&
         kernels::kSm87A4W4GateUpDownEdgeWarps == 16U &&
         kernels::kSm87A4W4GateUpDownEdgeWarpRows == 4U &&
         kernels::kSm87A4W4GateUpDownEdgeWarpColumns == 4U &&
         kernels::kSm87A4W4GateUpDownEdgeWarpTileM == 16U &&
         kernels::kSm87A4W4GateUpDownEdgeWarpTileN == 32U &&
         kernels::kSm87A4W4GateUpDownEdgeCellsPerScale == 4U &&
         kernels::kSm87A4W4GateUpDownEdgePipelineBytes == 83'200U &&
         kernels::kSm87A4W4GateUpDownEdgePlaneBytes == 65'536U &&
         kernels::kSm87A4W4GateUpDownEdgeDynamicSharedBytes == 148'736U &&
         kernels::kSm87A4W4GateUpDownEdgeMaximumRegisters == 128U &&
         kernels::kSm87A4W4GateUpDownEdgeCtasPerSm == 1U;
}

[[nodiscard]] bool check_fail_closed() {
  const auto bad_launch =
      kernels::sm87_a4w4_gateup_down_edge_test_plan(
          129U, 128U, 2'048U, 1'024U, 3U);
  const auto bad_n = kernels::sm87_a4w4_gateup_down_edge_test_plan(
      128U, 128U, 1'920U, 1'024U, 3U);
  const auto bad_k = kernels::sm87_a4w4_gateup_down_edge_test_plan(
      128U, 128U, 2'048U, 768U, 3U);
  const auto bad_zero_ctas =
      kernels::sm87_a4w4_gateup_down_edge_test_plan(
          128U, 128U, 2'048U, 1'024U, 0U);
  const auto bad_many_ctas =
      kernels::sm87_a4w4_gateup_down_edge_test_plan(
          128U, 128U, 2'048U, 1'024U, 17U);
  const auto bad_model = kernels::sm87_a4w4_gateup_down_edge_plan(
      128U, 128U, 2'048U, 1'024U);
  const int null_query =
      kernels::query_sm87_a4w4_gateup_down_k512_edge_resources_cuda(
          nullptr);
  const int null_launch =
      kernels::launch_sm87_a4w4_gateup_down_k512_edge_test_cuda(
          nullptr, 65'536U, nullptr, 256U, nullptr, 1'048'576U,
          nullptr, 4'096U, nullptr, 1'048'576U, nullptr, 4'096U,
          129U, 256U, 2'048U, 1'024U, 1.0F, nullptr, 262'144U,
          nullptr, 1'024U, 3U);
  return bad_launch.launch_ctas == 0U && bad_n.launch_ctas == 0U &&
         bad_k.launch_ctas == 0U && bad_zero_ctas.launch_ctas == 0U &&
         bad_many_ctas.launch_ctas == 0U &&
         bad_model.launch_ctas == 0U &&
         null_query == static_cast<int>(cudaErrorInvalidValue) &&
         null_launch == static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] bool check_link_surface() {
  auto* volatile query =
      &kernels::query_sm87_a4w4_gateup_down_k512_edge_resources_cuda;
  auto* volatile production =
      &kernels::launch_sm87_a4w4_gateup_down_k512_edge_cuda;
  auto* volatile synthetic =
      &kernels::launch_sm87_a4w4_gateup_down_k512_edge_test_cuda;
  return query != nullptr && production != nullptr && synthetic != nullptr;
}

}  // namespace

int main() {
  if (!check_real_layout()) {
    std::cerr << "real GateUp->Down K512 edge layout failed\n";
    return 1;
  }
  if (!check_scheduler()) {
    std::cerr << "base-wave/residual edge-cell schedule failed\n";
    return 1;
  }
  if (!check_structure()) {
    std::cerr << "M64N128/K512 edge structure failed\n";
    return 1;
  }
  if (!check_fail_closed()) {
    std::cerr << "fail-closed admission failed\n";
    return 1;
  }
  if (!check_link_surface()) {
    std::cerr << "edge link surface failed\n";
    return 1;
  }
  std::cout << "GateUp->Down K512 edge contract passed\n";
  return 0;
}
