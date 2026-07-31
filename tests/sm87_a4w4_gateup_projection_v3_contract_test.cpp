#include "q3x/kernels/sm87_a4w4_gateup_projection_v3.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

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
      kernels::sm87_a4w4_gateup_projection_v3_plan(m, n, k);
  constexpr std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m, k);
  constexpr std::size_t a_scales =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(m, k);
  constexpr std::size_t b_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(n, k);
  constexpr std::size_t b_scales =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(n, k);
  constexpr std::size_t output_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m, n);
  constexpr std::size_t output_scales =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(m, n);

  return plan.m_tiles == 32U && plan.n_tiles == 136U &&
         plan.k128_groups == 40U && plan.physical_k64_groups == 80U &&
         plan.work_cells == 4'352U && plan.launch_ctas == 32U &&
         plan.output_physical_k64_groups == 272U &&
         a_bytes == 5'242'880U && a_scales == 81'920U &&
         b_bytes == 44'564'480U && b_scales == 696'320U &&
         output_bytes == 17'825'792U && output_scales == 278'528U &&
         kernels::sm87_a4w4_consumer_packed_offset(
             m - 1U, plan.physical_k64_groups - 1U, 31U,
             plan.physical_k64_groups) == a_bytes - 1U &&
         kernels::sm87_a4w4_consumer_packed_offset(
             n - 1U, plan.physical_k64_groups - 1U, 31U,
             plan.physical_k64_groups) == b_bytes - 1U &&
         kernels::sm87_a4w4_consumer_packed_offset(
             m - 1U, plan.output_physical_k64_groups - 1U, 31U,
             plan.output_physical_k64_groups) == output_bytes - 1U &&
         kernels::sm87_a4w4_consumer_k128_scale_offset(
             m - 1U, plan.k128_groups - 1U, plan.k128_groups) ==
             a_scales - 1U &&
         kernels::sm87_a4w4_consumer_k128_scale_offset(
             n - 1U, plan.k128_groups - 1U, plan.k128_groups) ==
             b_scales - 1U &&
         kernels::sm87_a4w4_consumer_k128_scale_offset(
             m - 1U, plan.n_tiles - 1U, plan.n_tiles) ==
             output_scales - 1U;
}

[[nodiscard]] bool check_rejections() {
  return kernels::sm87_a4w4_gateup_projection_v3_plan(
             65U, 128U, 256U)
                 .launch_ctas == 0U &&
         kernels::sm87_a4w4_gateup_projection_v3_plan(
             64U, 192U, 256U)
                 .launch_ctas == 0U &&
         kernels::sm87_a4w4_gateup_projection_v3_plan(
             64U, 128U, 192U)
                 .launch_ctas == 0U &&
         kernels::query_sm87_a4w4_gateup_projection_v3_resources_cuda(
             nullptr) == static_cast<int>(cudaErrorInvalidValue) &&
         kernels::launch_sm87_a4w4_gateup_projection_v3_cuda(
             nullptr, 8'192U, nullptr, 128U, nullptr, 16'384U, nullptr,
             256U, nullptr, 16'384U, nullptr, 256U, 64U, 128U, 256U,
             1.0F, nullptr, 4'096U, nullptr, 64U) ==
             static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] bool check_link_surface() {
  auto* volatile query =
      &kernels::query_sm87_a4w4_gateup_projection_v3_resources_cuda;
  auto* volatile launch =
      &kernels::launch_sm87_a4w4_gateup_projection_v3_cuda;
  return query != nullptr && launch != nullptr;
}

}  // namespace

int main() {
  if (!check_real_layout()) {
    std::cerr << "v3 real M2048/N17408/K5120 layout contract failed\n";
    return 1;
  }
  if (!check_rejections()) {
    std::cerr << "v3 fail-closed contract failed\n";
    return 1;
  }
  if (!check_link_surface()) {
    std::cerr << "v3 link surface contract failed\n";
    return 1;
  }
  std::cout << "Gate+Up projection v3 real-shape layout contract passed\n";
  return 0;
}
