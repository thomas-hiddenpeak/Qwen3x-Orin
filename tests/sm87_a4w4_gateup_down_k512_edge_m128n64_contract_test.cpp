#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m128n64.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iostream>

namespace {

namespace kernels = q3x::kernels;

[[nodiscard]] bool check_real_layout() {
  constexpr std::size_t m = 2'048U;
  constexpr std::size_t n = 17'408U;
  constexpr std::size_t k = 5'120U;
  constexpr auto plan =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_plan(m, m, n, k);
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
  return plan.m_tiles == 16U && plan.edge_groups == 34U &&
         plan.strips_per_edge == 8U && plan.input_k512_groups == 10U &&
         plan.input_physical_k64_groups == 80U &&
         plan.output_physical_k64_groups == 272U &&
         plan.work_edge_cells == 544U && plan.launch_ctas == 16U &&
         plan.maximum_iterations == 34U &&
         a_bytes == 5'242'880U && a_scales == 20'480U &&
         b_bytes == 44'564'480U && b_scales == 174'080U &&
         output_bytes == 17'825'792U && output_scales == 69'632U;
}

[[nodiscard]] bool check_scheduler() {
  constexpr auto p2048 =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_plan(
          2'048U, 2'048U, 17'408U, 5'120U);
  constexpr auto first =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_work_cell(
          p2048, 7U, 0U);
  constexpr auto last =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_work_cell(
          p2048, 7U, 33U);
  constexpr auto exhausted =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_work_cell(
          p2048, 7U, 34U);
  constexpr auto residual =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_test_plan(
          129U, 256U, 1'024U, 1'024U, 3U);
  constexpr auto residual0 =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_work_cell(
          residual, 2U, 0U);
  constexpr auto residual1 =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_work_cell(
          residual, 2U, 1U);
  return first.valid && first.m_tile == 7U && first.edge_group == 0U &&
         last.valid && last.m_tile == 7U && last.edge_group == 33U &&
         !exhausted.valid && residual.m_tiles == 2U &&
         residual.edge_groups == 2U && residual.work_edge_cells == 4U &&
         residual.launch_ctas == 3U && residual.maximum_iterations == 2U &&
         residual0.valid && residual0.m_tile == 0U &&
         residual0.edge_group == 1U && !residual1.valid;
}

[[nodiscard]] bool check_structure() {
  return kernels::kSm87A4W4GateUpDownEdgeM128N64Threads == 512U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64Warps == 16U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64WarpRows == 8U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64WarpColumns == 2U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64WarpTileM == 16U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64WarpTileN == 32U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64Stages == 4U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64ScaleSlots == 2U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64PipelineBytes == 33'792U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64PlaneBytes == 131'072U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes ==
             164'864U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64MaximumRegisters == 128U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N64CtasPerSm == 1U;
}

[[nodiscard]] bool check_fail_closed() {
  const auto bad_launch =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_test_plan(
          129U, 128U, 1'024U, 1'024U, 3U);
  const auto bad_n =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_test_plan(
          128U, 128U, 768U, 1'024U, 3U);
  const auto bad_k =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_test_plan(
          128U, 128U, 1'024U, 768U, 3U);
  const auto bad_ctas =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_test_plan(
          128U, 128U, 1'024U, 1'024U, 17U);
  const auto bad_model =
      kernels::sm87_a4w4_gateup_down_edge_m128n64_plan(
          128U, 128U, 1'024U, 1'024U);
  return bad_launch.launch_ctas == 0U && bad_n.launch_ctas == 0U &&
         bad_k.launch_ctas == 0U && bad_ctas.launch_ctas == 0U &&
         bad_model.launch_ctas == 0U &&
         kernels::query_sm87_a4w4_gateup_down_k512_edge_m128n64_resources_cuda(
             nullptr) == static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] bool check_link_surface() {
  auto* volatile query =
      &kernels::query_sm87_a4w4_gateup_down_k512_edge_m128n64_resources_cuda;
  auto* volatile production =
      &kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n64_cuda;
  auto* volatile synthetic =
      &kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n64_test_cuda;
  return query != nullptr && production != nullptr && synthetic != nullptr;
}

}  // namespace

int main() {
  if (!check_real_layout() || !check_scheduler() || !check_structure() ||
      !check_fail_closed() || !check_link_surface()) {
    std::cerr << "M128N64 GateUp->Down K512 edge contract failed\n";
    return 1;
  }
  std::cout << "M128N64 GateUp->Down K512 edge contract passed\n";
  return 0;
}
