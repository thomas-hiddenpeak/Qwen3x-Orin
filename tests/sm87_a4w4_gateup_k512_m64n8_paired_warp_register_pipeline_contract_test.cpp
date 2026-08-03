#include "q3x/kernels/sm87_a4w4_gateup_k512_m64n8_paired_warp_register_pipeline.h"

#if defined(Q3X_TEST_M64N8_PAIRED_WARP_PIPELINE_RESOURCES)
#include <cuda_runtime_api.h>
#endif

#include <cstddef>
#include <iostream>
#include <type_traits>

namespace {

namespace kernels = q3x::kernels;

using Plan = kernels::Sm87A4W4GateUpDownEdgePlan;

static_assert(std::is_standard_layout_v<Plan>);
static_assert(std::is_same_v<
              decltype(kernels::sm87_a4w4_gateup_down_edge_test_plan(
                  64U, 128U, 512U, 512U, 1U)),
              Plan>);

[[nodiscard]] constexpr bool constants_hold() noexcept {
  return kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineTileM ==
             64U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineComputeTileN ==
             128U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineEdgeN ==
             512U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineThreads ==
             512U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineWarps ==
             16U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineWarpN ==
             8U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineProjections ==
             2U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelinePanelsPerWarp ==
             4U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineK64PerGroup ==
             8U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineAStages ==
             2U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineAStageBytes ==
             16'384U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineScaleSlotBytes ==
             640U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineSlotBytes ==
             17'024U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelinePipelineBytes ==
             34'048U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineEdgeBytes ==
             65'536U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineDynamicSharedBytes ==
             99'584U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineMaximumRegisters ==
             128U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineCtasPerSm ==
             1U &&
         kernels::kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelinePersistentCtas ==
             16U;
}

[[nodiscard]] constexpr bool plans_hold() noexcept {
  const Plan one = kernels::sm87_a4w4_gateup_down_edge_test_plan(
      64U, 128U, 512U, 512U, 3U);
  const Plan residual = kernels::sm87_a4w4_gateup_down_edge_test_plan(
      513U, 640U, 1'024U, 1'536U, 7U);
  const Plan model = kernels::sm87_a4w4_gateup_down_edge_plan(
      1'853U, 1'920U, 17'408U, 5'120U);
  const Plan bad_launch = kernels::sm87_a4w4_gateup_down_edge_test_plan(
      129U, 128U, 512U, 512U, 1U);
  return one.m_tiles == 2U && one.edge_groups == 1U &&
         one.input_k512_groups == 1U && one.work_edge_cells == 2U &&
         one.launch_ctas == 2U && residual.m_tiles == 10U &&
         residual.edge_groups == 2U && residual.input_k512_groups == 3U &&
         residual.work_edge_cells == 20U && residual.launch_ctas == 7U &&
         model.m_tiles == 30U && model.edge_groups == 34U &&
         model.input_k512_groups == 10U && model.work_edge_cells == 1'020U &&
         model.launch_ctas == 16U && bad_launch.launch_ctas == 0U;
}

static_assert(constants_hold());
static_assert(plans_hold());

#if defined(Q3X_TEST_M64N8_PAIRED_WARP_PIPELINE_RESOURCES)

[[nodiscard]] int run_resource_gate() {
  kernels::Sm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineResources
      resources{};
  const int status = kernels::
      query_sm87_a4w4_gateup_k512_m64n8_paired_warp_register_pipeline_resources_cuda(
          &resources);
  if (status == static_cast<int>(cudaErrorNotSupported) ||
      status == static_cast<int>(cudaErrorNoDevice) ||
      status == static_cast<int>(cudaErrorInsufficientDriver)) {
    std::cout << "SKIP: paired-warp pipeline resources require 16-SM SM87\n";
    return 77;
  }
  if (status != static_cast<int>(cudaSuccess)) {
    std::cerr << "paired-warp pipeline resource query failed: "
              << cudaGetErrorName(static_cast<cudaError_t>(status)) << '\n';
    return 1;
  }
  const bool pass =
      resources.registers_per_thread > 0 &&
      resources.registers_per_thread <= 128 &&
      resources.static_shared_bytes == 0U &&
      resources.dynamic_shared_bytes == 99'584U &&
      resources.configured_dynamic_shared_limit_bytes >= 99'584U &&
      resources.device_optin_shared_limit_bytes >= 99'584U &&
      resources.local_bytes == 0U &&
      resources.maximum_threads_per_block >= 512 &&
      resources.active_blocks_per_sm == 1 &&
      resources.compute_major == 8 && resources.compute_minor == 7;
  std::cout << "paired-warp pipeline resources: registers="
            << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << " gate=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass ? 0 : 1;
}

#endif

}  // namespace

int main() {
#if defined(Q3X_TEST_M64N8_PAIRED_WARP_PIPELINE_RESOURCES)
  return run_resource_gate();
#else
  if (!constants_hold() || !plans_hold()) {
    std::cerr << "paired-warp register-pipeline contract failed\n";
    return 1;
  }
  std::cout << "paired-warp register-pipeline contract passed\n";
  return 0;
#endif
}
