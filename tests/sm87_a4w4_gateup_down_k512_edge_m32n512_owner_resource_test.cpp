#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m32n512_owner.h"

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
  const bool contract =
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerTileM == 32U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerTileN == 512U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN == 64U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerCellsPerEdge == 8U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerCopyK == 128U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerK64PerStage == 2U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerStages == 4U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlots == 2U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerThreads == 256U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerWarps == 8U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerPersistentCtas == 32U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerTeams == 16U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerCtasPerSm == 2U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerStageBytes == 10'240U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerPipelineBytes == 41'600U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerEdgePlaneBytes == 32'768U &&
      kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes ==
          74'368U &&
      kernels::sm87_a4w4_gateup_down_k512_edge_m32n512_owner_test_plan(
          117U, 128U, 512U, 512U)
              .launch_ctas == 32U &&
      kernels::sm87_a4w4_gateup_down_k512_edge_m32n512_owner_test_plan(
          117U, 128U, 1'024U, 1'536U)
              .residual_edge_cells == 4U &&
      kernels::sm87_a4w4_gateup_down_k512_edge_m32n512_owner_test_plan(
          117U, 256U, 512U, 512U)
              .launch_ctas == 0U &&
      kernels::sm87_a4w4_gateup_down_k512_edge_m32n512_owner_plan(
          117U, 128U, 1'024U, 1'536U)
              .launch_ctas == 0U &&
      kernels::sm87_a4w4_gateup_down_k512_edge_m32n512_owner_plan(
          512U, 512U, 17'408U, 5'120U)
              .launch_ctas == 32U &&
      kernels::
              query_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_resources_cuda(
                  nullptr) == static_cast<int>(cudaErrorInvalidValue);
  if (!contract) {
    std::cerr << "M32N512 edge-owner resource contract failed\n";
    return 1;
  }

  const int target = target_status();
  if (target != 0) {
    return target;
  }

  kernels::Sm87A4W4GateUpDownEdgeM32N512OwnerResources resources{};
  const int status = kernels::
      query_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_resources_cuda(
          &resources);
  std::cout << "M32N512 edge-owner resources: status=" << status
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
      resources.dynamic_shared_bytes != 74'368U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 256 ||
      resources.active_blocks_per_sm != 2) {
    std::cerr << "M32N512 edge-owner hard resource gate failed\n";
    return 1;
  }

  std::cout << "M32N512 edge-owner resource contract passed\n";
  return 0;
}
