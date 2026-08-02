#include "q3x/kernels/sm87_a4w4_gateup_k512_m128n512_fused_quantize.h"

#if defined(Q3X_TEST_M128N512_FUSED_QUANTIZE_RESOURCES)
#include <cuda_runtime_api.h>
#endif

#include <cstddef>
#include <iostream>
#include <type_traits>

namespace {

namespace kernels = q3x::kernels;

using Plan = kernels::Sm87A4W4GateUpK512M128N512FusedQuantizePlan;

static_assert(std::is_standard_layout_v<Plan>);
static_assert(std::is_same_v<
              decltype(kernels::
                           sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_plan(
                               128U, 128U, 512U, 512U, 1U)),
              Plan>);
static_assert(std::is_same_v<
              decltype(kernels::
                           sm87_a4w4_gateup_k512_m128n512_fused_quantize_plan(
                               1'853U, 1'920U, 17'408U, 5'120U)),
              Plan>);

[[nodiscard]] constexpr bool constants_hold() noexcept {
  return kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeTileM == 128U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeTileN == 512U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeComputeTileN ==
             128U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeSubcells == 4U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeThreads == 512U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeWarps == 16U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeWarpM == 16U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeWarpN == 64U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeStages == 2U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizePipelineBytes ==
             99'840U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeProductBytes ==
             65'536U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes ==
             165'376U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeMaximumRegisters ==
             128U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizeCtasPerSm == 1U &&
         kernels::kSm87A4W4GateUpK512M128N512FusedQuantizePersistentCtas ==
             16U;
}

[[nodiscard]] constexpr bool plans_hold() noexcept {
  const Plan one =
      kernels::sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_plan(
          128U, 128U, 512U, 512U, 3U);
  const Plan residual =
      kernels::sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_plan(
          129U, 256U, 1'024U, 1'024U, 3U);
  const Plan model =
      kernels::sm87_a4w4_gateup_k512_m128n512_fused_quantize_plan(
          1'853U, 1'920U, 17'408U, 5'120U);
  const Plan bad_launch =
      kernels::sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_plan(
          129U, 128U, 512U, 512U, 1U);
  const Plan bad_n =
      kernels::sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_plan(
          128U, 128U, 768U, 512U, 1U);
  const Plan bad_k =
      kernels::sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_plan(
          128U, 128U, 512U, 768U, 1U);
  const Plan no_ctas =
      kernels::sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_plan(
          128U, 128U, 512U, 512U, 0U);
  return one.m_tiles == 1U && one.edge_groups == 1U &&
         one.input_k512_groups == 1U && one.work_cells == 1U &&
         one.launch_ctas == 1U && residual.m_tiles == 2U &&
         residual.edge_groups == 2U && residual.input_k512_groups == 2U &&
         residual.work_cells == 4U && residual.launch_ctas == 3U &&
         model.m_tiles == 15U && model.edge_groups == 34U &&
         model.input_k512_groups == 10U &&
         model.input_physical_k64_groups == 80U &&
         model.output_physical_k64_groups == 272U &&
         model.work_cells == 510U && model.launch_ctas == 16U &&
         bad_launch.launch_ctas == 0U && bad_n.launch_ctas == 0U &&
         bad_k.launch_ctas == 0U && no_ctas.launch_ctas == 0U;
}

static_assert(constants_hold());
static_assert(plans_hold());

#if defined(Q3X_TEST_M128N512_FUSED_QUANTIZE_RESOURCES)

[[nodiscard]] int run_resource_gate() {
  kernels::Sm87A4W4GateUpK512M128N512FusedQuantizeResources resources{};
  const int status = kernels::
      query_sm87_a4w4_gateup_k512_m128n512_fused_quantize_resources_cuda(
          &resources);
  if (status == static_cast<int>(cudaErrorNotSupported) ||
      status == static_cast<int>(cudaErrorNoDevice) ||
      status == static_cast<int>(cudaErrorInsufficientDriver)) {
    std::cout << "SKIP: M128N512 fused-quantize resources require SM87\n";
    return 77;
  }
  if (status != static_cast<int>(cudaSuccess)) {
    std::cerr << "M128N512 fused-quantize resource query failed: "
              << cudaGetErrorName(static_cast<cudaError_t>(status)) << '\n';
    return 1;
  }
  const bool pass =
      resources.registers_per_thread > 0 &&
      resources.registers_per_thread <=
          static_cast<int>(
              kernels::
                  kSm87A4W4GateUpK512M128N512FusedQuantizeMaximumRegisters) &&
      resources.dynamic_shared_bytes ==
          kernels::
              kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes &&
      resources.local_bytes == 0U && resources.active_blocks_per_sm >= 1 &&
      resources.compute_major == 8 && resources.compute_minor == 7;
  std::cout << "M128N512 fused-quantize resources: registers="
            << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << " compute=" << resources.compute_major << '.'
            << resources.compute_minor << " gate="
            << (pass ? "PASS" : "FAIL") << '\n';
  if (!pass) {
    std::cerr << "M128N512 fused-quantize resource hard gate failed\n";
    return 1;
  }
  return 0;
}

#endif

}  // namespace

int main() {
#if defined(Q3X_TEST_M128N512_FUSED_QUANTIZE_RESOURCES)
  return run_resource_gate();
#else
  if (!constants_hold() || !plans_hold()) {
    std::cerr << "M128N512 fused-quantize Gate+Up contract failed\n";
    return 1;
  }
  std::cout << "M128N512 fused-quantize Gate+Up contract passed\n";
  return 0;
#endif
}
