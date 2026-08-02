#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {

namespace kernels = q3x::kernels;

using Launch = int (*)(
    const std::uint8_t*, std::size_t, const std::uint16_t*, std::size_t,
    const std::uint8_t*, std::size_t, const std::uint16_t*, std::size_t,
    const std::uint8_t*, std::size_t, const std::uint16_t*, std::size_t,
    std::size_t, std::size_t, std::size_t, std::size_t, float,
    std::uint8_t*, std::size_t, std::uint8_t*, std::size_t,
    std::uint16_t*, std::size_t, void*) noexcept;

using TestLaunch = int (*)(
    const std::uint8_t*, std::size_t, const std::uint16_t*, std::size_t,
    const std::uint8_t*, std::size_t, const std::uint16_t*, std::size_t,
    const std::uint8_t*, std::size_t, const std::uint16_t*, std::size_t,
    std::size_t, std::size_t, std::size_t, std::size_t, float,
    std::uint8_t*, std::size_t, std::uint8_t*, std::size_t,
    std::uint16_t*, std::size_t, unsigned int, void*) noexcept;

static_assert(std::is_same_v<
              decltype(&kernels::
                           launch_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_cuda),
              Launch>);
static_assert(std::is_same_v<
              decltype(&kernels::
                           launch_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_test_cuda),
              TestLaunch>);

[[nodiscard]] constexpr bool constants_hold() noexcept {
  return kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixTileM == 128U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixTileN == 512U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixComputeTileN ==
             64U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixCellsPerEdge ==
             8U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixThreads == 256U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixWarps == 8U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixM16Panels == 8U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixAStageBytes ==
             32'768U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixScaleSlotBytes ==
             512U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixPipelineBytes ==
             66'560U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixEdgePlaneBytes ==
             65'536U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixDynamicSharedBytes ==
             132'096U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixScratchBytesPerCta ==
             65'536U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixMaximumRegisters ==
             255U &&
         kernels::kSm87A4W4GateUpDownEdgeM128N512LdmatrixCtasPerSm == 1U;
}

[[nodiscard]] constexpr bool plans_hold() noexcept {
  const auto one =
      kernels::sm87_a4w4_gateup_down_edge_m128n512_ldmatrix_test_plan(
          128U, 128U, 512U, 512U, 3U);
  const auto residual =
      kernels::sm87_a4w4_gateup_down_edge_m128n512_ldmatrix_test_plan(
          129U, 256U, 1'024U, 1'024U, 3U);
  const auto model =
      kernels::sm87_a4w4_gateup_down_edge_m128n512_ldmatrix_plan(
          1'853U, 1'920U, 17'408U, 5'120U);
  const auto bad_launch =
      kernels::sm87_a4w4_gateup_down_edge_m128n512_ldmatrix_test_plan(
          129U, 128U, 512U, 512U, 1U);
  const auto bad_k =
      kernels::sm87_a4w4_gateup_down_edge_m128n512_ldmatrix_test_plan(
          128U, 128U, 512U, 768U, 1U);
  return one.m_tiles == 1U && one.edge_groups == 1U &&
         one.input_k512_groups == 1U && one.work_cells == 1U &&
         one.launch_ctas == 1U &&
         one.required_scratch_bytes == 65'536U &&
         residual.m_tiles == 2U && residual.edge_groups == 2U &&
         residual.work_cells == 4U && residual.launch_ctas == 3U &&
         residual.base_waves == 0U && residual.residual_m_tiles == 2U &&
         residual.residual_cells == 4U &&
         residual.required_scratch_bytes == 3U * 65'536U &&
         model.m_tiles == 15U && model.edge_groups == 34U &&
         model.input_k512_groups == 10U && model.launch_ctas == 16U &&
         model.required_scratch_bytes == 16U * 65'536U &&
         bad_launch.launch_ctas == 0U && bad_k.launch_ctas == 0U;
}

static_assert(constants_hold());
static_assert(plans_hold());

}  // namespace

int main() {
  if (!constants_hold() || !plans_hold()) {
    std::cerr << "M128N512 LDSM GateUp edge contract failed\n";
    return 1;
  }
  std::cout << "M128N512 LDSM GateUp edge contract passed\n";
  return 0;
}
