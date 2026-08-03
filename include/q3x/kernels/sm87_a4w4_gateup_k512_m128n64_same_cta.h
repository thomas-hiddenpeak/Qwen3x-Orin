#pragma once

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_GATEUP_M128N64_SAME_CTA_HOST_DEVICE \
  __host__ __device__
#else
#define Q3X_SM87_A4W4_GATEUP_M128N64_SAME_CTA_HOST_DEVICE
#endif

namespace q3x::kernels {

// Default-off large-M Gate+Up experiment over the authenticated canonical
// K512 v1 payload.  One 256-thread CTA owns M128N64.  Its 4x2 warp grid first
// computes a complete Gate projection, publishes the exact FP32 Gate plane in
// shared memory, then reuses the same registers and two-stage K256 A+B ring
// for Up.  The epilogue emits BF16-RNE SiLU(Gate_fp32)*Up_fp32 into one of the
// existing primary/secondary split output windows; there is no global Gate
// scratch and no intermediate projection rounding.
//
// The dynamic shared footprint is deliberately below half of the pinned
// SM87 per-SM opt-in limit:
//
//   2 * (M128 A K256 + N64 B K256) = 49,152 bytes
//   2 * (M128 A scales + N64 B scales) = 768 bytes
//   M128N64 exact FP32 Gate handoff = 32,768 bytes
//                                      ----------------
//                                      82,688 bytes
//
// Each warp owns M32N32 and retains 32 FP32 projection accumulators/thread.
// Eight K64 integer MMA terms remain S32 until the K512 boundary, where one
// rounded FP32 BF16-scale product and one FP32 FMA are applied in ascending
// group order.  Production dispatch remains default-off behind an exact
// selector and fails closed unless the hard 2-CTA/SM resource gate passes.
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaTileM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaTileN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaWarpM = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaWarpN = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaWarpRows = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaWarpColumns = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaStages = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaScaleSlots = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaCopyK = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaK64PerStage = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaPersistentCtas = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaCtasPerSm = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaPackedK64Bytes = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaAStageBytes = 16'384U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaBStageBytes = 8'192U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaStageBytes = 24'576U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaScaleSlotBytes = 384U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaPipelineBytes = 49'920U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaGatePlaneBytes = 32'768U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaDynamicSharedBytes = 82'688U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaModelIntermediate = 17'408U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaModelInput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaPrimaryWidth = 12'288U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaSecondaryWidth = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaSecondaryRowStride = 6'144U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64SameCtaMaximumTokens = 4'096U;

[[nodiscard]] constexpr bool
sm87_a4w4_gateup_k512_m128n64_same_cta_product_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_m128n64_same_cta_outer_blocks(
    const std::size_t outer) noexcept {
  return outer == 0U ? 0U : 1U + (outer - 1U) / 64U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_m128n64_same_cta_scale_capacity(
    const std::size_t outer, const std::size_t logical_k) noexcept {
  if (logical_k == 0U ||
      logical_k % kSm87A4W4GateUpK512M128N64SameCtaScaleK != 0U) {
    return 0U;
  }
  const std::size_t blocks =
      sm87_a4w4_gateup_k512_m128n64_same_cta_outer_blocks(outer);
  const std::size_t groups =
      logical_k / kSm87A4W4GateUpK512M128N64SameCtaScaleK;
  if (blocks == 0U ||
      !sm87_a4w4_gateup_k512_m128n64_same_cta_product_fits(blocks,
                                                             groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  return sm87_a4w4_gateup_k512_m128n64_same_cta_product_fits(
             block_groups, 64U)
             ? block_groups * 64U
             : 0U;
}

[[nodiscard]]
Q3X_SM87_A4W4_GATEUP_M128N64_SAME_CTA_HOST_DEVICE constexpr std::size_t
sm87_a4w4_gateup_k512_m128n64_same_cta_scale_offset(
    const std::size_t outer_coordinate, const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  return ((outer_coordinate / 64U) * k512_group_count + k512_group) *
             64U +
         outer_coordinate % 64U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_m128n64_same_cta_canonical_launch_tokens(
    const std::size_t logical_token_count) noexcept {
  if (logical_token_count == 0U ||
      logical_token_count >
          kSm87A4W4GateUpK512M128N64SameCtaMaximumTokens ||
      logical_token_count >
          std::numeric_limits<std::size_t>::max() -
              (kSm87A4W4GateUpK512M128N64SameCtaTileM - 1U)) {
    return 0U;
  }
  return ((logical_token_count +
           kSm87A4W4GateUpK512M128N64SameCtaTileM - 1U) /
          kSm87A4W4GateUpK512M128N64SameCtaTileM) *
         kSm87A4W4GateUpK512M128N64SameCtaTileM;
}

struct Sm87A4W4GateUpK512M128N64SameCtaPlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
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

[[nodiscard]] constexpr Sm87A4W4GateUpK512M128N64SameCtaPlan
sm87_a4w4_gateup_k512_m128n64_same_cta_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size, const std::size_t input_size,
    const std::size_t n_start, const std::size_t n_count) noexcept {
  if (logical_token_count == 0U ||
      logical_token_count > launch_token_count ||
      launch_token_count %
              kSm87A4W4GateUpK512M128N64SameCtaTileM !=
          0U ||
      intermediate_size == 0U ||
      intermediate_size %
              kSm87A4W4GateUpK512M128N64SameCtaTileN !=
          0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpK512M128N64SameCtaScaleK != 0U ||
      n_start % kSm87A4W4GateUpK512M128N64SameCtaTileN != 0U ||
      n_count == 0U ||
      n_count % kSm87A4W4GateUpK512M128N64SameCtaTileN != 0U ||
      n_start > intermediate_size ||
      n_count > intermediate_size - n_start) {
    return {};
  }
  const std::size_t m_tiles =
      launch_token_count / kSm87A4W4GateUpK512M128N64SameCtaTileM;
  const std::size_t n_tiles =
      n_count / kSm87A4W4GateUpK512M128N64SameCtaTileN;
  if (!sm87_a4w4_gateup_k512_m128n64_same_cta_product_fits(m_tiles,
                                                             n_tiles)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      work_cells < kSm87A4W4GateUpK512M128N64SameCtaPersistentCtas
          ? work_cells
          : kSm87A4W4GateUpK512M128N64SameCtaPersistentCtas;
  const std::size_t minimum_cells_per_cta = work_cells / launch_ctas;
  const std::size_t maximum_cells_per_cta =
      minimum_cells_per_cta + (work_cells % launch_ctas != 0U ? 1U : 0U);
  return {logical_token_count,
          launch_token_count,
          intermediate_size,
          input_size,
          n_start,
          n_count,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4GateUpK512M128N64SameCtaScaleK,
          input_size / 64U,
          work_cells,
          launch_ctas,
          minimum_cells_per_cta,
          maximum_cells_per_cta};
}

[[nodiscard]] constexpr bool
sm87_a4w4_gateup_k512_m128n64_same_cta_is_model_plan(
    const Sm87A4W4GateUpK512M128N64SameCtaPlan& plan) noexcept {
  const bool split_window =
      (plan.n_start == 0U &&
       plan.n_count ==
           kSm87A4W4GateUpK512M128N64SameCtaPrimaryWidth) ||
      (plan.n_start ==
               kSm87A4W4GateUpK512M128N64SameCtaPrimaryWidth &&
       plan.n_count ==
           kSm87A4W4GateUpK512M128N64SameCtaSecondaryWidth);
  return plan.launch_ctas != 0U &&
         plan.launch_token_count ==
             sm87_a4w4_gateup_k512_m128n64_same_cta_canonical_launch_tokens(
                 plan.logical_token_count) &&
         plan.intermediate_size ==
             kSm87A4W4GateUpK512M128N64SameCtaModelIntermediate &&
         plan.input_size ==
             kSm87A4W4GateUpK512M128N64SameCtaModelInput &&
         split_window;
}

struct Sm87A4W4GateUpK512M128N64SameCtaResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t configured_dynamic_shared_limit_bytes{};
  std::size_t device_optin_shared_limit_bytes{};
  std::size_t device_shared_per_sm_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int
query_sm87_a4w4_gateup_k512_m128n64_same_cta_resources_cuda(
    Sm87A4W4GateUpK512M128N64SameCtaResources* resources) noexcept;

// output_bf16 points at the selected split plane.  Its row stride is the
// physical width of that plane (12,288 primary or 6,144 secondary in the
// model path); the secondary logical n_count remains 5,120.  n_start is the
// absolute weight-row coordinate.
[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_m128n64_same_cta_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_gate_b,
    std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* gate_b_k512_scales_bf16,
    std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* packed_up_b,
    std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* up_b_k512_scales_bf16,
    std::size_t up_b_scale_capacity_elements,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::size_t n_start,
    std::size_t n_count,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_m128n64_same_cta_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_gate_b,
    std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* gate_b_k512_scales_bf16,
    std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* packed_up_b,
    std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* up_b_k512_scales_bf16,
    std::size_t up_b_scale_capacity_elements,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::size_t n_start,
    std::size_t n_count,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4GateUpK512M128N64SameCtaWarps ==
              kSm87A4W4GateUpK512M128N64SameCtaWarpRows *
                  kSm87A4W4GateUpK512M128N64SameCtaWarpColumns);
static_assert(kSm87A4W4GateUpK512M128N64SameCtaStageBytes ==
              kSm87A4W4GateUpK512M128N64SameCtaAStageBytes +
                  kSm87A4W4GateUpK512M128N64SameCtaBStageBytes);
static_assert(kSm87A4W4GateUpK512M128N64SameCtaPipelineBytes ==
              kSm87A4W4GateUpK512M128N64SameCtaStages *
                      kSm87A4W4GateUpK512M128N64SameCtaStageBytes +
                  kSm87A4W4GateUpK512M128N64SameCtaScaleSlots *
                      kSm87A4W4GateUpK512M128N64SameCtaScaleSlotBytes);
static_assert(kSm87A4W4GateUpK512M128N64SameCtaDynamicSharedBytes ==
              kSm87A4W4GateUpK512M128N64SameCtaPipelineBytes +
                  kSm87A4W4GateUpK512M128N64SameCtaGatePlaneBytes);
static_assert(
    sm87_a4w4_gateup_k512_m128n64_same_cta_plan(
        1'853U, 1'920U, 17'408U, 5'120U, 0U, 12'288U)
        .launch_ctas == 32U);
static_assert(
    sm87_a4w4_gateup_k512_m128n64_same_cta_plan(
        2'048U, 2'048U, 17'408U, 5'120U, 12'288U, 5'120U)
        .n_tiles == 80U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_GATEUP_M128N64_SAME_CTA_HOST_DEVICE
