#include "q3x/kernels/sm87_a4w4_down_factorized_r4_m128n64_2cta.h"

#include <cuda_runtime_api.h>

#include <iostream>

namespace {

namespace kernels = q3x::kernels;

[[nodiscard]] constexpr bool contract_holds() noexcept {
  const auto plan = kernels::sm87_a4w4_down_factorized_r4_m128n64_plan(
      1'853U, 1'920U, 5'120U, 17'408U);
  return kernels::kSm87A4W4DownFactorizedR4M128N64TileM == 128U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64TileN == 64U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64Threads == 256U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64Warps == 8U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64CopyK == 256U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64SharedStages == 3U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64AStageBytes == 16'384U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64BStageBytes == 8'192U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64DynamicSharedBytes ==
             73'728U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64MaximumDynamicSharedBytes ==
             83'968U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64MaximumRegisters == 128U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64CtasPerSm == 2U &&
         kernels::kSm87A4W4DownFactorizedR4M128N64PersistentCtas == 32U &&
         plan.m_owners == 15U && plan.n_tiles == 80U &&
         plan.work_tiles == 1'200U && plan.launch_ctas == 32U;
}

static_assert(contract_holds());

}  // namespace

int main() {
  if (!contract_holds() ||
      kernels::query_sm87_a4w4_down_factorized_r4_m128n64_resources_cuda(
          nullptr) != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "R4 Down M128N64 2-CTA static contract failed\n";
    return 1;
  }

  kernels::Sm87A4W4DownFactorizedR4M128N64Resources resources{};
  const int status =
      kernels::query_sm87_a4w4_down_factorized_r4_m128n64_resources_cuda(
          &resources);
  if (status == static_cast<int>(cudaErrorNotSupported) ||
      status == static_cast<int>(cudaErrorNoDevice) ||
      status == static_cast<int>(cudaErrorInsufficientDriver)) {
    std::cout << "SKIP: R4 Down M128N64 2-CTA requires 16-SM SM87\n";
    return 77;
  }

  const bool passes =
      status == static_cast<int>(cudaSuccess) &&
      resources.registers_per_thread > 0 &&
      resources.registers_per_thread <= 128 &&
      resources.static_shared_bytes == 0U &&
      resources.dynamic_shared_bytes == 73'728U &&
      resources.dynamic_shared_bytes <= 83'968U &&
      resources.configured_dynamic_shared_limit_bytes >= 73'728U &&
      resources.device_optin_shared_limit_bytes >= 73'728U &&
      resources.local_bytes == 0U &&
      resources.maximum_threads_per_block >= 256 &&
      resources.active_blocks_per_sm == 2 &&
      resources.resident_blocks >= 32 &&
      resources.multiprocessor_count == 16 &&
      resources.compute_major == 8 && resources.compute_minor == 7;

  std::cout << "R4 Down M128N64 2-CTA resources: status=" << status
            << " registers=" << resources.registers_per_thread
            << " static_shared=" << resources.static_shared_bytes
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " max_threads=" << resources.maximum_threads_per_block
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << " resident_blocks=" << resources.resident_blocks
            << " gate=" << (passes ? "PASS" : "FAIL") << '\n';
  if (!passes) {
    if (status != static_cast<int>(cudaSuccess)) {
      std::cerr << "resource query status: "
                << cudaGetErrorName(static_cast<cudaError_t>(status)) << '\n';
    }
    return 1;
  }
  return 0;
}
