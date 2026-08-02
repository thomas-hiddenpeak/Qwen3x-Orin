#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating.h"

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
  return plan.launch_ctas == 16U && plan.m_tiles == 32U &&
         plan.edge_groups == 34U && plan.input_k512_groups == 10U &&
         plan.output_physical_k64_groups == 272U &&
         a_bytes == 5'242'880U && a_scales == 20'480U &&
         b_bytes == 44'564'480U && b_scales == 174'080U &&
         output_bytes == 17'825'792U && output_scales == 69'632U;
}

[[nodiscard]] bool check_scheduler() {
  constexpr auto m64 = kernels::sm87_a4w4_gateup_down_edge_test_plan(
      64U, 128U, 1'024U, 1'536U, 3U);
  constexpr auto m65 = kernels::sm87_a4w4_gateup_down_edge_test_plan(
      65U, 128U, 1'024U, 1'536U, 3U);
  constexpr auto residual = kernels::sm87_a4w4_gateup_down_edge_test_plan(
      129U, 256U, 2'048U, 1'536U, 3U);
  return m64.m_tiles == 2U && m64.input_k512_groups == 3U &&
         m64.launch_ctas == 3U && m64.base_waves == 0U &&
         m64.residual_m_tiles == 2U && m64.residual_edge_cells == 4U &&
         m65.m_tiles == 2U && m65.launch_ctas == 3U &&
         m65.residual_m_tiles == 2U && m65.residual_edge_cells == 4U &&
         residual.m_tiles == 4U && residual.edge_groups == 4U &&
         residual.input_k512_groups == 3U && residual.launch_ctas == 3U &&
         residual.base_waves == 1U && residual.residual_m_tiles == 1U &&
         residual.residual_edge_cells == 4U;
}

[[nodiscard]] bool check_structure() {
  return kernels::kSm87A4W4GateUpDownEdgeM64N128K256AlternatingThreads ==
             512U &&
         kernels::kSm87A4W4GateUpDownEdgeM64N128K256AlternatingCtasPerSm ==
             1U &&
         kernels::kSm87A4W4GateUpDownEdgeM64N128K256AlternatingMaximumRegisters ==
             128U &&
         kernels::kSm87A4W4GateUpDownEdgeM64N128K256AlternatingPreferredRegisters ==
             125U &&
         kernels::kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes ==
             148'736U &&
         kernels::kSm87A4W4GateUpDownEdgeM64N128K256AlternatingBarriersPerEdge ==
             85U;
}

[[nodiscard]] bool check_fail_closed() {
  const int null_query =
      kernels::query_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_resources_cuda(
          nullptr);
  const int null_launch =
      kernels::launch_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_test_cuda(
          nullptr, 98'304U, nullptr, 384U, nullptr, 1'572'864U,
          nullptr, 6'144U, nullptr, 1'572'864U, nullptr, 6'144U,
          65U, 128U, 2'048U, 1'536U, 1.0F, nullptr, 131'072U,
          nullptr, 512U, 3U);
  const int null_production =
      kernels::launch_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_cuda(
          nullptr, 1'310'720U, nullptr, 5'120U, nullptr, 44'564'480U,
          nullptr, 174'080U, nullptr, 44'564'480U, nullptr, 174'080U,
          512U, 512U, 17'408U, 5'120U, 1.0F, nullptr, 4'456'448U,
          nullptr, 17'408U);
  return null_query == static_cast<int>(cudaErrorInvalidValue) &&
         null_launch == static_cast<int>(cudaErrorInvalidValue) &&
         null_production == static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] bool check_link_surface() {
  auto* volatile query =
      &kernels::query_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_resources_cuda;
  auto* volatile production =
      &kernels::launch_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_cuda;
  auto* volatile synthetic =
      &kernels::launch_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_test_cuda;
  return query != nullptr && production != nullptr && synthetic != nullptr;
}

}  // namespace

int main() {
  if (!check_real_layout() || !check_scheduler() || !check_structure() ||
      !check_fail_closed() || !check_link_surface()) {
    std::cerr << "alternating K256 GateUp edge contract failed\n";
    return 1;
  }
  std::cout << "alternating K256 GateUp edge contract passed\n";
  return 0;
}
