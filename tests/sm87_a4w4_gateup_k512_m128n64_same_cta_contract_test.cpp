#include "q3x/kernels/sm87_a4w4_gateup_k512_m128n64_same_cta.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

#if defined(Q3X_TEST_M128N64_SAME_CTA_RESOURCES)
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
  cudaDeviceProp properties{};
  status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDevice: " << cudaGetErrorName(status) << '\n';
    return 1;
  }
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
#endif

}  // namespace

int main() {
  namespace kernels = q3x::kernels;
  constexpr auto primary =
      kernels::sm87_a4w4_gateup_k512_m128n64_same_cta_plan(
          1'853U, 1'920U, 17'408U, 5'120U, 0U, 12'288U);
  constexpr auto secondary =
      kernels::sm87_a4w4_gateup_k512_m128n64_same_cta_plan(
          2'048U, 2'048U, 17'408U, 5'120U, 12'288U, 5'120U);
  constexpr auto bad_launch =
      kernels::sm87_a4w4_gateup_k512_m128n64_same_cta_plan(
          1'853U, 1'856U, 17'408U, 5'120U, 0U, 12'288U);
  constexpr auto noncanonical_padding =
      kernels::sm87_a4w4_gateup_k512_m128n64_same_cta_plan(
          1U, 256U, 17'408U, 5'120U, 0U, 12'288U);

  using Query = int (*)(
      kernels::Sm87A4W4GateUpK512M128N64SameCtaResources*) noexcept;
  using Launch = int (*)(
      const std::uint8_t*, std::size_t, const std::uint16_t*,
      std::size_t, const std::uint8_t*, std::size_t,
      const std::uint16_t*, std::size_t, const std::uint8_t*,
      std::size_t, const std::uint16_t*, std::size_t, std::size_t,
      std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
      std::uint16_t*, std::size_t, std::size_t, void*) noexcept;
  Query query = &kernels::
      query_sm87_a4w4_gateup_k512_m128n64_same_cta_resources_cuda;
  Launch launch = &kernels::
      launch_sm87_a4w4_gateup_k512_m128n64_same_cta_bf16_cuda;

  const bool valid =
      query != nullptr && launch != nullptr &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaTileM == 128U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaTileN == 64U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaThreads == 256U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaWarps == 8U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaWarpRows == 4U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaWarpColumns == 2U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaStages == 2U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaCtasPerSm == 2U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaPersistentCtas == 32U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaPipelineBytes == 49'920U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaGatePlaneBytes == 32'768U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaDynamicSharedBytes ==
          82'688U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaSecondaryWidth == 5'120U &&
      kernels::kSm87A4W4GateUpK512M128N64SameCtaSecondaryRowStride ==
          6'144U &&
      primary.m_tiles == 15U && primary.n_tiles == 192U &&
      primary.work_cells == 2'880U && primary.k512_groups == 10U &&
      primary.launch_ctas == 32U && primary.minimum_cells_per_cta == 90U &&
      primary.maximum_cells_per_cta == 90U &&
      kernels::sm87_a4w4_gateup_k512_m128n64_same_cta_is_model_plan(
          primary) &&
      secondary.m_tiles == 16U && secondary.n_tiles == 80U &&
      secondary.launch_ctas == 32U &&
      kernels::sm87_a4w4_gateup_k512_m128n64_same_cta_is_model_plan(
          secondary) &&
      bad_launch.launch_ctas == 0U &&
      noncanonical_padding.launch_ctas == 32U &&
      !kernels::sm87_a4w4_gateup_k512_m128n64_same_cta_is_model_plan(
          noncanonical_padding) &&
      kernels::sm87_a4w4_gateup_k512_m128n64_same_cta_scale_capacity(
          1'920U, 5'120U) == 19'200U &&
      kernels::
              query_sm87_a4w4_gateup_k512_m128n64_same_cta_resources_cuda(
                  nullptr) == static_cast<int>(cudaErrorInvalidValue);
  if (!valid) {
    std::cerr << "M128N64 same-CTA Gate+Up contract failed\n";
    return 1;
  }

#if defined(Q3X_TEST_M128N64_SAME_CTA_RESOURCES)
  const int target = target_status();
  if (target != 0) {
    return target;
  }
  kernels::Sm87A4W4GateUpK512M128N64SameCtaResources resources{};
  const int status = kernels::
      query_sm87_a4w4_gateup_k512_m128n64_same_cta_resources_cuda(
          &resources);
  std::cout << "M128N64 same-CTA resources: status=" << status
            << " registers_per_thread=" << resources.registers_per_thread
            << " static_shared_bytes=" << resources.static_shared_bytes
            << " dynamic_shared_bytes=" << resources.dynamic_shared_bytes
            << " localSizeBytes=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << " max_dynamic_shared="
            << resources.configured_dynamic_shared_limit_bytes
            << " shared_per_sm=" << resources.device_shared_per_sm_bytes
            << '\n';
  if (status != static_cast<int>(cudaSuccess) ||
      resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 128 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 82'688U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 256 ||
      resources.active_blocks_per_sm < 2) {
    std::cerr << "M128N64 same-CTA hard resource gate failed\n";
    return 1;
  }
#endif

  std::cout << "M128N64 same-CTA Gate+Up contract passed\n";
  return 0;
}
