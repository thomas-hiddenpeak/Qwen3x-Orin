#include "q3x/kernels/sm87_a4w4_gateup_factorized_r4_m64n64_2cta.h"

#include <cuda_runtime_api.h>

#include <iostream>

namespace {

[[nodiscard]] int target_status() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      count == 0) {
    (void)cudaGetLastError();
    return 77;
  }
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount: " << cudaGetErrorName(status) << '\n';
    return 1;
  }
  int device = 0;
  status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDevice: " << cudaGetErrorName(status) << '\n';
    return 1;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties: " << cudaGetErrorName(status)
              << '\n';
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: requires the pinned 16-SM SM87 target\n";
    return 77;
  }
  return 0;
}

}  // namespace

int main() {
  namespace kernels = q3x::kernels;
  constexpr auto plan =
      kernels::sm87_a4w4_gateup_factorized_r4_m64n64_2cta_plan(
          1'853U, 1'920U, 17'408U, 5'120U);
  constexpr auto bad_launch =
      kernels::sm87_a4w4_gateup_factorized_r4_m64n64_2cta_plan(
          1'853U, 1'856U, 17'408U, 5'120U);
  const bool static_contract =
      kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileM == 64U &&
      kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileN == 64U &&
      kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaThreads == 256U &&
      kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWarps == 8U &&
      kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaCopyK == 256U &&
      kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedStages == 2U &&
      kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaCtasPerSm == 2U &&
      kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPersistentCtas == 32U &&
      kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaMaximumRegisters ==
          128U &&
      kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedCrossBytes ==
          16'384U &&
      kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes ==
          65'536U &&
      plan.work_tiles == 8'160U && plan.launch_ctas == 32U &&
      bad_launch.launch_ctas == 0U &&
      kernels::
              query_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_resources_cuda(
                  nullptr) == static_cast<int>(cudaErrorInvalidValue);
  if (!static_contract) {
    std::cerr << "R4 M64N64 two-CTA Gate+Up static contract failed\n";
    return 1;
  }

  const int target = target_status();
  if (target != 0) {
    return target;
  }

  kernels::Sm87A4W4GateUpFactorizedR4M64N64TwoCtaResources resources{};
  const int status = kernels::
      query_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_resources_cuda(
          &resources);
  std::cout << "R4 M64N64 two-CTA Gate+Up resources: status=" << status
            << " registers_per_thread=" << resources.registers_per_thread
            << " static_shared_bytes=" << resources.static_shared_bytes
            << " dynamic_shared_bytes=" << resources.dynamic_shared_bytes
            << " localSizeBytes=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << " resident_blocks=" << resources.resident_blocks
            << " max_dynamic_shared="
            << resources.configured_dynamic_shared_limit_bytes << '\n';
  if (status != static_cast<int>(cudaSuccess) ||
      resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 128 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 65'536U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 256 ||
      resources.active_blocks_per_sm < 2 ||
      resources.resident_blocks < 32) {
    std::cerr << "R4 M64N64 two-CTA Gate+Up hard resource gate failed\n";
    return 1;
  }

  std::cout << "R4 M64N64 two-CTA Gate+Up resource contract passed\n";
  return 0;
}
