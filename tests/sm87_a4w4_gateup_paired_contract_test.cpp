#include "q3x/kernels/sm87_a4w4_gateup_paired.h"
#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

namespace kernels = q3x::kernels;

[[nodiscard]] bool check_real_shapes() {
  const kernels::Sm87A4W4GateUpPairedPlan p512 =
      kernels::sm87_a4w4_gateup_paired_plan(512U, 17'408U, 5'120U);
  const kernels::Sm87A4W4GateUpPairedPlan p3847 =
      kernels::sm87_a4w4_gateup_paired_plan(3'847U, 17'408U, 5'120U);
  const kernels::Sm87A4W4GateUpPairedPlan p1024 =
      kernels::sm87_a4w4_gateup_paired_plan(1'024U, 17'408U, 5'120U);
  const kernels::Sm87A4W4GateUpPairedPlan p2048 =
      kernels::sm87_a4w4_gateup_paired_plan(2'048U, 17'408U, 5'120U);
  const kernels::Sm87A4W4GateUpPairedPlan p4096 =
      kernels::sm87_a4w4_gateup_paired_plan(4'096U, 17'408U, 5'120U);
  return p512.m_tiles == 16U && p512.n_tiles == 136U &&
         p512.k64_groups == 80U && p512.work_tiles == 2'176U &&
         p512.launch_ctas == 32U &&
         p512.tile_m == 32U && p512.tile_n == 128U &&
         p512.kernel ==
             kernels::Sm87A4W4GateUpPairedKernel::kM32N128K64 &&
         p512.packed_output_row_bytes == 8'704U &&
         p512.output_scale_row_elements == 272U &&
         p3847.m_tiles == 121U && p3847.n_tiles == 136U &&
         p3847.work_tiles == 16'456U && p3847.launch_ctas == 32U &&
         p3847.kernel ==
             kernels::Sm87A4W4GateUpPairedKernel::kM32N128K64 &&
         p1024.m_tiles == 16U && p1024.n_tiles == 272U &&
         p1024.work_tiles == 4'352U && p1024.tile_m == 64U &&
         p1024.tile_n == 64U &&
         p1024.kernel ==
             kernels::Sm87A4W4GateUpPairedKernel::kM64N64K64 &&
         p2048.m_tiles == 32U && p2048.n_tiles == 136U &&
         p2048.work_tiles == 4'352U && p2048.tile_m == 64U &&
         p2048.tile_n == 128U &&
         p2048.kernel ==
             kernels::Sm87A4W4GateUpPairedKernel::kM64N128K64 &&
         p4096.m_tiles == 64U && p4096.n_tiles == 136U &&
         p4096.work_tiles == 8'704U &&
         p4096.kernel ==
             kernels::Sm87A4W4GateUpPairedKernel::kM64N128K64;
}

[[nodiscard]] bool check_n_major_work_tile_planner() {
  const kernels::Sm87A4W4GateUpPairedPlan plan =
      kernels::sm87_a4w4_gateup_paired_plan(2'048U, 17'408U, 5'120U);
  const kernels::Sm87A4W4GateUpPairedWorkTile first =
      kernels::sm87_a4w4_gateup_paired_n_major_work_tile(
          0U, plan.m_tiles, plan.work_tiles);
  const kernels::Sm87A4W4GateUpPairedWorkTile last_m =
      kernels::sm87_a4w4_gateup_paired_n_major_work_tile(
          31U, plan.m_tiles, plan.work_tiles);
  const kernels::Sm87A4W4GateUpPairedWorkTile next_n =
      kernels::sm87_a4w4_gateup_paired_n_major_work_tile(
          32U, plan.m_tiles, plan.work_tiles);
  const kernels::Sm87A4W4GateUpPairedWorkTile last =
      kernels::sm87_a4w4_gateup_paired_n_major_work_tile(
          plan.work_tiles - 1U, plan.m_tiles, plan.work_tiles);
  const kernels::Sm87A4W4GateUpPairedWorkTile rejected =
      kernels::sm87_a4w4_gateup_paired_n_major_work_tile(
          plan.work_tiles, plan.m_tiles, plan.work_tiles);
  return first.valid && first.m_tile == 0U && first.n_tile == 0U &&
         last_m.valid && last_m.m_tile == 31U && last_m.n_tile == 0U &&
         next_n.valid && next_n.m_tile == 0U && next_n.n_tile == 1U &&
         last.valid && last.m_tile == 31U && last.n_tile == 135U &&
         !rejected.valid;
}

[[nodiscard]] bool check_tail_and_rejection_plans() {
  const kernels::Sm87A4W4GateUpPairedPlan tail =
      kernels::sm87_a4w4_gateup_paired_plan(65U, 128U, 256U);
  return tail.m_tiles == 3U && tail.n_tiles == 1U &&
         tail.k64_groups == 4U && tail.work_tiles == 3U &&
         tail.launch_ctas == 32U &&
         kernels::sm87_a4w4_gateup_paired_plan(0U, 128U, 64U)
                 .launch_ctas == 0U &&
         kernels::sm87_a4w4_gateup_paired_plan(1U, 129U, 64U)
                 .launch_ctas == 0U &&
         kernels::sm87_a4w4_gateup_paired_plan(1U, 128U, 65U)
                 .launch_ctas == 0U;
}

[[nodiscard]] bool check_k64_composite_plans() {
  const auto generic65 =
      kernels::sm87_a4w4_prefill_gemm_k64_composite_plan(
          65U, 5'120U, 17'408U);
  const auto generic1025 =
      kernels::sm87_a4w4_prefill_gemm_k64_composite_plan(
          1'025U, 5'120U, 17'408U);
  const auto generic1804 =
      kernels::sm87_a4w4_prefill_gemm_k64_composite_plan(
          1'804U, 5'120U, 17'408U);
  const auto generic3987 =
      kernels::sm87_a4w4_prefill_gemm_k64_composite_plan(
          3'987U, 5'120U, 17'408U);
  const auto paired65 =
      kernels::sm87_a4w4_gateup_paired_k64_composite_plan(
          65U, 17'408U, 5'120U);
  const auto paired1025 =
      kernels::sm87_a4w4_gateup_paired_k64_composite_plan(
          1'025U, 17'408U, 5'120U);
  const auto paired1804 =
      kernels::sm87_a4w4_gateup_paired_k64_composite_plan(
          1'804U, 17'408U, 5'120U);
  const auto paired3987 =
      kernels::sm87_a4w4_gateup_paired_k64_composite_plan(
          3'987U, 17'408U, 5'120U);

  return generic65.valid && generic65.prefix_token_count == 0U &&
         generic65.tail_token_count == 65U &&
         generic65.tail_plan.launch_ctas != 0U &&
         generic1025.prefix_token_count == 1'024U &&
         generic1025.tail_token_count == 1U &&
         generic1025.prefix_kernel ==
             kernels::Sm87A4W4PrefillK64CompositePrefixKernel::kM64N64K64 &&
         generic1804.prefix_token_count == 1'792U &&
         generic1804.tail_token_count == 12U &&
         generic1804.prefix_kernel ==
             kernels::Sm87A4W4PrefillK64CompositePrefixKernel::kM64N256K64 &&
         generic3987.prefix_token_count == 3'968U &&
         generic3987.tail_token_count == 19U &&
         generic3987.prefix_kernel ==
             kernels::Sm87A4W4PrefillK64CompositePrefixKernel::kM64N256K64 &&
         paired65.valid && paired65.prefix_token_count == 0U &&
         paired65.tail_token_count == 65U &&
         paired65.tail_plan.kernel ==
             kernels::Sm87A4W4GateUpPairedKernel::kM32N128K64 &&
         paired1025.prefix_token_count == 1'024U &&
         paired1025.tail_token_count == 1U &&
         paired1025.prefix_plan.kernel ==
             kernels::Sm87A4W4GateUpPairedKernel::kM64N64K64 &&
         paired1804.prefix_token_count == 1'792U &&
         paired1804.tail_token_count == 12U &&
         paired1804.prefix_plan.kernel ==
             kernels::Sm87A4W4GateUpPairedKernel::kM64N128K64 &&
         paired3987.prefix_token_count == 3'968U &&
         paired3987.tail_token_count == 19U &&
         paired3987.prefix_plan.kernel ==
             kernels::Sm87A4W4GateUpPairedKernel::kM64N128K64;
}

[[nodiscard]] bool check_k128_plans() {
  const kernels::Sm87A4W4GateUpPairedK128Plan tiny =
      kernels::sm87_a4w4_gateup_paired_k128_plan(64U, 128U, 256U);
  const kernels::Sm87A4W4GateUpPairedK128Plan real =
      kernels::sm87_a4w4_gateup_paired_k128_plan(
          2'048U, 17'408U, 5'120U);
  return tiny.m_tiles == 1U && tiny.n_tiles == 1U &&
         tiny.k128_groups == 2U && tiny.physical_k64_groups == 4U &&
         tiny.work_tiles == 1U && tiny.launch_ctas == 1U &&
         tiny.packed_output_row_bytes == 64U &&
         tiny.output_scale_row_elements == 1U &&
         real.m_tiles == 32U && real.n_tiles == 136U &&
         real.k128_groups == 40U && real.physical_k64_groups == 80U &&
         real.work_tiles == 4'352U && real.launch_ctas == 32U &&
         real.output_scale_row_elements == 136U &&
         kernels::sm87_a4w4_gateup_paired_k128_plan(
                     65U, 128U, 256U)
                 .launch_ctas == 0U &&
         kernels::sm87_a4w4_gateup_paired_k128_plan(
                     64U, 192U, 256U)
                 .launch_ctas == 0U &&
         kernels::sm87_a4w4_gateup_paired_k128_plan(
                     64U, 128U, 192U)
                 .launch_ctas == 0U;
}

[[nodiscard]] bool check_fail_closed_host_surface() {
  if (kernels::query_sm87_a4w4_gateup_paired_resources_cuda(nullptr) !=
          static_cast<int>(cudaErrorInvalidValue) ||
      kernels::query_sm87_a4w4_gateup_paired_large_m_resources_cuda(
          nullptr) != static_cast<int>(cudaErrorInvalidValue) ||
      kernels::query_sm87_a4w4_gateup_paired_wide_large_m_resources_cuda(
          nullptr) != static_cast<int>(cudaErrorInvalidValue) ||
      kernels::query_sm87_a4w4_gateup_paired_k128_resources_cuda(nullptr) !=
          static_cast<int>(cudaErrorInvalidValue)) {
    return false;
  }

  // All pointers are intentionally null.  Argument validation must reject
  // this call before touching CUDA device state, keeping the compile/ABI test
  // runnable on a host with no available GPU.
  const int status = kernels::launch_sm87_a4w4_gateup_paired_cuda(
      nullptr, 32U, nullptr, 1U, nullptr, 32U, nullptr, 1U, nullptr, 32U,
      nullptr, 1U, 1U, 64U, 64U, 1.0F, nullptr, 32U, nullptr, 1U);
  const int k128_status =
      kernels::launch_sm87_a4w4_gateup_paired_k128_cuda(
          nullptr, 64U, nullptr, 1U, nullptr, 64U, nullptr, 1U, nullptr,
          64U, nullptr, 1U, 64U, 128U, 128U, 1.0F, nullptr, 64U,
          nullptr, 1U);
  return status == static_cast<int>(cudaErrorInvalidValue) &&
         k128_status == static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] bool check_link_contract() {
  auto* volatile query =
      &kernels::query_sm87_a4w4_gateup_paired_resources_cuda;
  auto* volatile large_m_query =
      &kernels::query_sm87_a4w4_gateup_paired_large_m_resources_cuda;
  auto* volatile wide_large_m_query =
      &kernels::query_sm87_a4w4_gateup_paired_wide_large_m_resources_cuda;
  auto* volatile k128_query =
      &kernels::query_sm87_a4w4_gateup_paired_k128_resources_cuda;
  auto* volatile launch = &kernels::launch_sm87_a4w4_gateup_paired_cuda;
  auto* volatile k128_launch =
      &kernels::launch_sm87_a4w4_gateup_paired_k128_cuda;
  return query != nullptr && large_m_query != nullptr &&
         wide_large_m_query != nullptr && k128_query != nullptr &&
         launch != nullptr && k128_launch != nullptr;
}

}  // namespace

int main() {
  if (!check_real_shapes()) {
    std::cerr << "real Gate+Up plan contract failed\n";
    return 1;
  }
  if (!check_tail_and_rejection_plans()) {
    std::cerr << "tail/rejection plan contract failed\n";
    return 1;
  }
  if (!check_n_major_work_tile_planner()) {
    std::cerr << "N-major work-tile planner contract failed\n";
    return 1;
  }
  if (!check_k64_composite_plans()) {
    std::cerr << "K64 composite plan contract failed\n";
    return 1;
  }
  if (!check_k128_plans()) {
    std::cerr << "shared-K128 Gate+Up plan contract failed\n";
    return 1;
  }
  if (!check_fail_closed_host_surface()) {
    std::cerr << "fail-closed host surface contract failed\n";
    return 1;
  }
  if (!check_link_contract()) {
    std::cerr << "paired Gate+Up admission link contract failed\n";
    return 1;
  }
  std::cout << "SM87 A4W4 paired Gate+Up host contracts passed\n";
  return 0;
}
