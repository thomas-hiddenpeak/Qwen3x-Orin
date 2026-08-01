#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Standalone M128-balanced Gate+Up experiment over the existing v2 paired
// payload.  It is deliberately disconnected from CMake and dispatch.
//
// A CTA owns M128 x N32.  Its eight warps form a 2(M64) x 4(N8) grid:
//
//   warp_n8 = warp >> 1, warp_m64 = warp & 1.
//
// Every warp owns one complete M64 x N8 Gate+Up strip and applies SwiGLU
// locally.  Adjacent warps are the two M64 owners of the same N8 fragment;
// they issue an aligned ld.global.ca.v4.u32 for the identical v2 B record in
// the same phase, enabling L1 reuse.  B is never copied to shared memory and
// the v2 publication needs no permutation.
//
// A uses the proven three-stage K128 cp.async ring enlarged to M128: 3 x
// 8,192 = 24,576 shared bytes.  Eight K64 MMA terms remain S32 until one
// K512 scale boundary, and K512 groups accumulate in FP32 in the same order
// as the authenticated M64 kernel.  Two M128N32 cells cover M128N64 with
// equal aggregate presented bytes to two M64N64 cells, but move B reuse from
// cross-CTA L2 timing into same-CTA, same-phase L1 timing.  This is the
// spill-free fallback selected after the direct M128N64 form compiled with
// 128 registers, a 96-byte stack, and 176-byte spill loads/stores.
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128TileM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128TileN = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128WarpM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128WarpN = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128Threads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128Warps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128PhysicalK = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128ScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128K64PerScale = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128K128PerScale = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128AStages = 3U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128AStageK = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128AStageBytes =
        kSm87A4W4GateUpK512FragmentNativeM128TileM *
        kSm87A4W4GateUpK512FragmentNativeM128AStageK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128SharedBytes =
        kSm87A4W4GateUpK512FragmentNativeM128AStages *
        kSm87A4W4GateUpK512FragmentNativeM128AStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128MaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128CtasPerSm = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128PersistentCtas = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128MinimumTokens = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128MaximumTokens = 4'096U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128ModelIntermediate = 17'408U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128ModelInput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128Window0Start = 0U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128Window0Count = 12'288U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128Window1Start = 12'288U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128Window1Count = 5'120U;

static_assert(kSm87A4W4GateUpK512FragmentNativeM128AStageBytes ==
              8'192U);
static_assert(kSm87A4W4GateUpK512FragmentNativeM128SharedBytes ==
              24'576U);

struct Sm87A4W4GateUpK512FragmentNativeM128Plan final {
  std::size_t token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t n_start{};
  std::size_t n_count{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k512_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_cells{};
  std::size_t launch_ctas{};
  std::size_t minimum_cells_per_cta{};
  std::size_t maximum_cells_per_cta{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpK512FragmentNativeM128Plan
sm87_a4w4_gateup_k512_fragment_native_m128_plan(
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count) noexcept {
  if (token_count <
          kSm87A4W4GateUpK512FragmentNativeM128MinimumTokens ||
      token_count >
          kSm87A4W4GateUpK512FragmentNativeM128MaximumTokens ||
      token_count % kSm87A4W4GateUpK512FragmentNativeM128TileM != 0U ||
      intermediate_size == 0U ||
      intermediate_size % 64U != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpK512FragmentNativeM128ScaleK != 0U ||
      n_start % kSm87A4W4GateUpK512FragmentNativeM128TileN != 0U ||
      n_count == 0U ||
      n_count % kSm87A4W4GateUpK512FragmentNativeM128TileN != 0U ||
      n_start > intermediate_size ||
      n_count > intermediate_size - n_start) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4GateUpK512FragmentNativeM128TileM;
  const std::size_t n_tiles =
      n_count / kSm87A4W4GateUpK512FragmentNativeM128TileN;
  if (!sm87_a4w4_gateup_k512_fragment_native_product_fits(
          m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      work_cells <
              kSm87A4W4GateUpK512FragmentNativeM128PersistentCtas
          ? work_cells
          : kSm87A4W4GateUpK512FragmentNativeM128PersistentCtas;
  return {token_count,
          intermediate_size,
          input_size,
          n_start,
          n_count,
          m_tiles,
          n_tiles,
          input_size /
              kSm87A4W4GateUpK512FragmentNativeM128ScaleK,
          input_size /
              kSm87A4W4GateUpK512FragmentNativeM128PhysicalK,
          work_cells,
          launch_ctas,
          work_cells / launch_ctas,
          work_cells / launch_ctas +
              (work_cells % launch_ctas != 0U ? 1U : 0U)};
}

[[nodiscard]] constexpr bool
sm87_a4w4_gateup_k512_fragment_native_m128_is_model_plan(
    const Sm87A4W4GateUpK512FragmentNativeM128Plan& plan) noexcept {
  const bool model_window =
      (plan.n_start ==
           kSm87A4W4GateUpK512FragmentNativeM128Window0Start &&
       plan.n_count ==
           kSm87A4W4GateUpK512FragmentNativeM128Window0Count) ||
      (plan.n_start ==
           kSm87A4W4GateUpK512FragmentNativeM128Window1Start &&
       plan.n_count ==
           kSm87A4W4GateUpK512FragmentNativeM128Window1Count);
  return plan.launch_ctas != 0U &&
         plan.intermediate_size ==
             kSm87A4W4GateUpK512FragmentNativeM128ModelIntermediate &&
         plan.input_size ==
             kSm87A4W4GateUpK512FragmentNativeM128ModelInput &&
         model_window;
}

struct Sm87A4W4GateUpK512FragmentNativeM128Resources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int
query_sm87_a4w4_gateup_k512_fragment_native_m128_resources_cuda(
    Sm87A4W4GateUpK512FragmentNativeM128Resources* resources) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_fragment_native_m128_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* paired_b_codes,
    std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* paired_b_scales_bf16,
    std::size_t paired_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::size_t n_start,
    std::size_t n_count,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_fragment_native_m128_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* paired_b_codes,
    std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* paired_b_scales_bf16,
    std::size_t paired_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::size_t n_start,
    std::size_t n_count,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
