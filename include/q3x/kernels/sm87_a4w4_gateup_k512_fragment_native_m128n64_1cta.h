#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off one-CTA/SM Gate+Up experiment over the existing v2 paired
// payload.  CMake and runtime dispatch expose it only through the explicit
// M128N64 one-CTA admission build; it never replaces the production default.
//
// A CTA owns M128 x N64.  Warp w owns the complete M128 x N8 strip at
// n=8*w:
//
//   warp_n8 = warp.
//
// The warp issues one ld.global.ca.v4.u32 v2 B record per K64 plane and
// reuses those Gate/Up fragments across eight M16 panels.  Gate/Up FP32
// accumulators and K512 S32 partials remain register-resident through
// same-warp SwiGLU.  Relative to two M64N64 CTAs covering M128N64, this
// ownership halves B record loads and register feeds.  B never enters
// shared memory.
//
// Each M128 x K512 A stage is 32 KiB.  A two-stage pipeline plus two
// 512-byte K512 scale slots uses 66,560 dynamic shared bytes.  While plane p
// of group g computes, plane p of group g+1 is prefetched into the other
// stage before the group swap.
// The kernel intentionally trades occupancy for a spill-free wide logical
// cell: launch_bounds is 256 threads / one CTA per SM and the admission cap
// is 255 registers/thread, matching the strategy used by stock Marlin-class
// large-M kernels rather than forcing a two-CTA 128-register ceiling.
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaWarpM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaWarpN = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaMmaN = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaPhysicalK = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaK64PerScale = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaAStages = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaAStageBytes =
        kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileM *
        kSm87A4W4GateUpK512FragmentNativeM128N641CtaScaleK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaScaleSlotBytes = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaSharedBytes =
        kSm87A4W4GateUpK512FragmentNativeM128N641CtaAStages *
            (kSm87A4W4GateUpK512FragmentNativeM128N641CtaAStageBytes +
             kSm87A4W4GateUpK512FragmentNativeM128N641CtaScaleSlotBytes);
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaMaximumRegisters = 255U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaPersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaMinimumTokens = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaMaximumTokens = 4'096U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaModelIntermediate =
        17'408U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaModelInput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaWindow0Start = 0U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaWindow0Count = 12'288U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaWindow1Start = 12'288U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaWindow1Count = 5'120U;

static_assert(
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaAStageBytes ==
    32'768U);
static_assert(
    kSm87A4W4GateUpK512FragmentNativeM128N641CtaSharedBytes ==
    66'560U);

struct Sm87A4W4GateUpK512FragmentNativeM128N641CtaPlan final {
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

[[nodiscard]] constexpr Sm87A4W4GateUpK512FragmentNativeM128N641CtaPlan
sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_plan(
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count) noexcept {
  if (token_count <
          kSm87A4W4GateUpK512FragmentNativeM128N641CtaMinimumTokens ||
      token_count >
          kSm87A4W4GateUpK512FragmentNativeM128N641CtaMaximumTokens ||
      token_count %
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileM !=
          0U ||
      intermediate_size == 0U || intermediate_size % 64U != 0U ||
      input_size == 0U ||
      input_size %
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaScaleK !=
          0U ||
      n_start %
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileN !=
          0U ||
      n_count == 0U ||
      n_count %
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileN !=
          0U ||
      n_start > intermediate_size ||
      n_count > intermediate_size - n_start) {
    return {};
  }
  const std::size_t m_tiles =
      token_count /
      kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileM;
  const std::size_t n_tiles =
      n_count /
      kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileN;
  if (!sm87_a4w4_gateup_k512_fragment_native_product_fits(
          m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      work_cells <
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaPersistentCtas
          ? work_cells
          : kSm87A4W4GateUpK512FragmentNativeM128N641CtaPersistentCtas;
  return {token_count,
          intermediate_size,
          input_size,
          n_start,
          n_count,
          m_tiles,
          n_tiles,
          input_size /
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaScaleK,
          input_size /
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaPhysicalK,
          work_cells,
          launch_ctas,
          work_cells / launch_ctas,
          work_cells / launch_ctas +
              (work_cells % launch_ctas != 0U ? 1U : 0U)};
}

[[nodiscard]] constexpr bool
sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_is_model_plan(
    const Sm87A4W4GateUpK512FragmentNativeM128N641CtaPlan& plan) noexcept {
  const bool model_window =
      (plan.n_start ==
           kSm87A4W4GateUpK512FragmentNativeM128N641CtaWindow0Start &&
       plan.n_count ==
           kSm87A4W4GateUpK512FragmentNativeM128N641CtaWindow0Count) ||
      (plan.n_start ==
           kSm87A4W4GateUpK512FragmentNativeM128N641CtaWindow1Start &&
       plan.n_count ==
           kSm87A4W4GateUpK512FragmentNativeM128N641CtaWindow1Count);
  return plan.launch_ctas != 0U &&
         plan.intermediate_size ==
             kSm87A4W4GateUpK512FragmentNativeM128N641CtaModelIntermediate &&
         plan.input_size ==
             kSm87A4W4GateUpK512FragmentNativeM128N641CtaModelInput &&
         model_window;
}

struct Sm87A4W4GateUpK512FragmentNativeM128N641CtaResources final {
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
query_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_resources_cuda(
    Sm87A4W4GateUpK512FragmentNativeM128N641CtaResources* resources) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_bf16_cuda(
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
launch_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_test_bf16_cuda(
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
