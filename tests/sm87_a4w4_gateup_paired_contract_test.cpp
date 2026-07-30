#include "q3x/kernels/sm87_a4w4_gateup_paired.h"

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
  return p512.m_tiles == 16U && p512.n_tiles == 136U &&
         p512.k64_groups == 80U && p512.work_tiles == 2'176U &&
         p512.launch_ctas == 32U &&
         p512.packed_output_row_bytes == 8'704U &&
         p512.output_scale_row_elements == 272U &&
         p3847.m_tiles == 121U && p3847.n_tiles == 136U &&
         p3847.work_tiles == 16'456U && p3847.launch_ctas == 32U;
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

[[nodiscard]] bool check_fail_closed_host_surface() {
  if (kernels::query_sm87_a4w4_gateup_paired_resources_cuda(nullptr) !=
      static_cast<int>(cudaErrorInvalidValue)) {
    return false;
  }

  // All pointers are intentionally null.  Argument validation must reject
  // this call before touching CUDA device state, keeping the compile/ABI test
  // runnable on a host with no available GPU.
  const int status = kernels::launch_sm87_a4w4_gateup_paired_cuda(
      nullptr, 32U, nullptr, 1U, nullptr, 32U, nullptr, 1U, nullptr, 32U,
      nullptr, 1U, 1U, 64U, 64U, 1.0F, nullptr, 32U, nullptr, 1U);
  return status == static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] bool check_link_contract() {
  auto* volatile query =
      &kernels::query_sm87_a4w4_gateup_paired_resources_cuda;
  auto* volatile launch = &kernels::launch_sm87_a4w4_gateup_paired_cuda;
  return query != nullptr && launch != nullptr;
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
