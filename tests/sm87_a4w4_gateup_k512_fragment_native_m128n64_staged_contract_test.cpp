#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native_m128n64_staged.h"

#include <cstddef>
#include <iostream>

namespace {

namespace kernels = q3x::kernels;

static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedTileM == 128U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedTileN == 64U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedWarpM == 64U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedWarpN == 8U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedThreads == 512U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedWarps == 16U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedCopyK == 256U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedK64PerCopy ==
    4U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedAStageBytes ==
    16'384U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedBStageBytes ==
    16'384U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedStageBytes ==
    32'768U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedStages == 2U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedSharedBytes ==
    66'560U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedMaximumRegisters ==
    128U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedTargetRegisters ==
    120U);
static_assert(
    kernels::kSm87A4W4GateUpK512FragmentNativeM128N64StagedCtasPerSm == 1U);

[[nodiscard]] bool verify_v2_stage_bijection() {
  constexpr std::size_t n = 128U;
  constexpr std::size_t k = 1'024U;
  constexpr std::size_t groups = k / 512U;
  const std::size_t capacity =
      kernels::sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
          n, k);
  if (capacity != n * k) {
    return false;
  }
  std::size_t vectors = 0U;
  for (std::size_t phase = 0U; phase < 4U; ++phase) {
    for (std::size_t n8 = 0U; n8 < 8U; ++n8) {
      for (std::size_t lane = 0U; lane < 32U; ++lane) {
        const std::size_t offset =
            kernels::
                sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                    n8 * 8U, 1U, 4U + phase, lane, groups);
        const std::size_t expected =
            ((((1U * 8U + 4U + phase) * 8U + n8) * 32U + lane) *
             16U);
        if (offset != expected || offset % 16U != 0U ||
            offset + 16U > capacity) {
          return false;
        }
        ++vectors;
      }
    }
  }
  return vectors == 1'024U;
}

}  // namespace

int main() {
  if (!verify_v2_stage_bijection()) {
    std::cerr << "paired-B K256 stage is not a 16 KiB v2 bijection\n";
    return 1;
  }
  const auto window0 =
      kernels::
          sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_plan(
              1'920U, 17'408U, 5'120U, 0U, 12'288U);
  const auto window1 =
      kernels::
          sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_plan(
              1'920U, 17'408U, 5'120U, 12'288U, 5'120U);
  if (!kernels::
          sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_is_model_plan(
              window0) ||
      !kernels::
          sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_is_model_plan(
              window1) ||
      window0.m_tiles != 15U || window0.n_tiles != 192U ||
      window0.work_cells != 2'880U || window0.launch_ctas != 15U ||
      window0.minimum_cells_per_cta != 192U ||
      window0.maximum_cells_per_cta != 192U ||
      window1.n_tiles != 80U || window1.work_cells != 1'200U ||
      window1.launch_ctas != 15U ||
      window1.minimum_cells_per_cta != 80U ||
      window1.maximum_cells_per_cta != 80U) {
    std::cerr << "fixed-M model schedule mismatch\n";
    return 1;
  }
  const auto full =
      kernels::
          sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_plan(
              2'048U, 17'408U, 5'120U, 0U, 17'408U);
  const auto maximum =
      kernels::
          sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_plan(
              4'096U, 128U, 512U, 0U, 128U);
  if (full.m_tiles != 16U || full.n_tiles != 272U ||
      full.k512_groups != 10U || full.work_cells != 4'352U ||
      full.launch_ctas != 16U || full.minimum_cells_per_cta != 272U ||
      kernels::
              sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_barriers_per_cell(
                  full.k512_groups) != 41U ||
      maximum.m_tiles != 32U || maximum.n_tiles != 2U ||
      maximum.launch_ctas != 16U ||
      maximum.minimum_cells_per_cta != 4U ||
      maximum.maximum_cells_per_cta != 4U ||
      kernels::
              sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_plan(
                  64U, 64U, 512U, 0U, 64U)
                  .launch_ctas != 0U ||
      kernels::
              sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_plan(
                  128U, 128U, 512U, 32U, 64U)
                  .launch_ctas != 0U) {
    std::cerr << "shape or fixed-M admission mismatch\n";
    return 1;
  }
  auto* volatile query =
      &kernels::
          query_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_resources_cuda;
  auto* volatile launch =
      &kernels::
          launch_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_test_bf16_cuda;
  if (query == nullptr || launch == nullptr) {
    std::cerr << "link surface missing\n";
    return 1;
  }
  std::cout << "PASS: M128N64 staged v2 paired-B contract\n";
  return 0;
}
