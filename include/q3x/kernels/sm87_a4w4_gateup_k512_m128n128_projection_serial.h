#pragma once

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_GATEUP_M128N128_SERIAL_HOST_DEVICE \
  __host__ __device__
#else
#define Q3X_SM87_A4W4_GATEUP_M128N128_SERIAL_HOST_DEVICE
#endif

// Isolated large-M Gate+Up experiment over the authenticated v1 consumer
// payload.  One persistent 512-thread CTA owns M128N128.  Its 4x4 warp grid
// first computes the complete Gate projection, publishes the exact FP32 Gate
// plane, then reuses the same accumulator registers and three-stage A+B K256
// ring for Up.  The Up epilogue applies SiLU(Gate_fp32)*Up_fp32 and emits the
// same BF16 pre-quantization seam as the production M64N128 edge.
//
// Projection serialization intentionally reloads A once.  In exchange, one
// CTA presents each Gate/Up B tile and its scales once for M128 instead of
// once per M64 CTA.  The 165,376-byte shared footprint is:
//
//   3 * (M128 A K256 + N128 B K256) = 98,304 bytes
//   3 * (M128 A scales + N128 B scales) = 1,536 bytes
//   M128N128 exact FP32 Gate handoff = 65,536 bytes
//
// The two projections retain the exact K512 numerical boundary: eight K64
// S32 MMA terms, one rounded FP32 scale product, then one FP32 FMA in
// ascending K512-group order.  This file is a default-off experiment and is
// deliberately disconnected from runtime dispatch.
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialTileM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialTileN = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialWarpM = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialWarpN = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialThreads = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialWarps = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialStages = 3U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialScaleSlots = 3U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialCopyK = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialK64PerStage = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialPersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialPackedK64Bytes = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialStageBytes = 32'768U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialScaleSlotBytes = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialGatePlaneBytes = 65'536U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialDynamicSharedBytes =
        kSm87A4W4GateUpK512M128N128ProjectionSerialStages *
            kSm87A4W4GateUpK512M128N128ProjectionSerialStageBytes +
        kSm87A4W4GateUpK512M128N128ProjectionSerialScaleSlots *
            kSm87A4W4GateUpK512M128N128ProjectionSerialScaleSlotBytes +
        kSm87A4W4GateUpK512M128N128ProjectionSerialGatePlaneBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialModelIntermediate =
        17'408U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialModelInput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N128ProjectionSerialMaximumTokens = 4'096U;

[[nodiscard]] constexpr bool
sm87_a4w4_gateup_k512_m128n128_projection_serial_product_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_m128n128_projection_serial_outer_blocks(
    const std::size_t outer) noexcept {
  return outer == 0U ? 0U : 1U + (outer - 1U) / 64U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_m128n128_projection_serial_scale_capacity(
    const std::size_t outer, const std::size_t logical_k) noexcept {
  if (logical_k == 0U ||
      logical_k %
              kSm87A4W4GateUpK512M128N128ProjectionSerialScaleK !=
          0U) {
    return 0U;
  }
  const std::size_t blocks =
      sm87_a4w4_gateup_k512_m128n128_projection_serial_outer_blocks(
          outer);
  const std::size_t groups =
      logical_k /
      kSm87A4W4GateUpK512M128N128ProjectionSerialScaleK;
  if (blocks == 0U ||
      !sm87_a4w4_gateup_k512_m128n128_projection_serial_product_fits(
          blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  return sm87_a4w4_gateup_k512_m128n128_projection_serial_product_fits(
             block_groups, 64U)
             ? block_groups * 64U
             : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_GATEUP_M128N128_SERIAL_HOST_DEVICE constexpr
    std::size_t
sm87_a4w4_gateup_k512_m128n128_projection_serial_scale_offset(
    const std::size_t outer_coordinate, const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  return ((outer_coordinate / 64U) * k512_group_count + k512_group) *
             64U +
         outer_coordinate % 64U;
}

struct Sm87A4W4GateUpK512M128N128ProjectionSerialPlan final {
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
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_m128n128_projection_serial_canonical_launch_tokens(
    const std::size_t logical_token_count) noexcept {
  if (logical_token_count == 0U ||
      logical_token_count >
          kSm87A4W4GateUpK512M128N128ProjectionSerialMaximumTokens ||
      logical_token_count >
          std::numeric_limits<std::size_t>::max() -
              (kSm87A4W4GateUpK512M128N128ProjectionSerialTileM - 1U)) {
    return 0U;
  }
  return ((logical_token_count +
           kSm87A4W4GateUpK512M128N128ProjectionSerialTileM - 1U) /
          kSm87A4W4GateUpK512M128N128ProjectionSerialTileM) *
         kSm87A4W4GateUpK512M128N128ProjectionSerialTileM;
}

[[nodiscard]] constexpr
    Sm87A4W4GateUpK512M128N128ProjectionSerialPlan
sm87_a4w4_gateup_k512_m128n128_projection_serial_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size, const std::size_t input_size,
    const std::size_t n_start, const std::size_t n_count) noexcept {
  if (logical_token_count == 0U ||
      logical_token_count > launch_token_count ||
      launch_token_count %
              kSm87A4W4GateUpK512M128N128ProjectionSerialTileM !=
          0U ||
      intermediate_size == 0U ||
      intermediate_size %
              kSm87A4W4GateUpK512M128N128ProjectionSerialTileN !=
          0U ||
      input_size == 0U ||
      input_size %
              kSm87A4W4GateUpK512M128N128ProjectionSerialScaleK !=
          0U ||
      n_start %
              kSm87A4W4GateUpK512M128N128ProjectionSerialTileN !=
          0U ||
      n_count == 0U ||
      n_count %
              kSm87A4W4GateUpK512M128N128ProjectionSerialTileN !=
          0U ||
      n_start > intermediate_size ||
      n_count > intermediate_size - n_start) {
    return {};
  }
  const std::size_t m_tiles =
      launch_token_count /
      kSm87A4W4GateUpK512M128N128ProjectionSerialTileM;
  const std::size_t n_tiles =
      n_count /
      kSm87A4W4GateUpK512M128N128ProjectionSerialTileN;
  if (!sm87_a4w4_gateup_k512_m128n128_projection_serial_product_fits(
          m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      work_cells <
              kSm87A4W4GateUpK512M128N128ProjectionSerialPersistentCtas
          ? work_cells
          : kSm87A4W4GateUpK512M128N128ProjectionSerialPersistentCtas;
  return {logical_token_count,
          launch_token_count,
          intermediate_size,
          input_size,
          n_start,
          n_count,
          m_tiles,
          n_tiles,
          input_size /
              kSm87A4W4GateUpK512M128N128ProjectionSerialScaleK,
          input_size / 64U,
          launch_ctas};
}

[[nodiscard]] constexpr bool
sm87_a4w4_gateup_k512_m128n128_projection_serial_is_model_plan(
    const Sm87A4W4GateUpK512M128N128ProjectionSerialPlan& plan) noexcept {
  const bool split_window =
      (plan.n_start == 0U && plan.n_count == 12'288U) ||
      (plan.n_start == 12'288U && plan.n_count == 5'120U);
  return plan.launch_ctas != 0U &&
         plan.launch_token_count ==
             sm87_a4w4_gateup_k512_m128n128_projection_serial_canonical_launch_tokens(
                 plan.logical_token_count) &&
         plan.intermediate_size ==
             kSm87A4W4GateUpK512M128N128ProjectionSerialModelIntermediate &&
         plan.input_size ==
             kSm87A4W4GateUpK512M128N128ProjectionSerialModelInput &&
         split_window;
}

struct Sm87A4W4GateUpK512M128N128ProjectionSerialResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t configured_dynamic_shared_limit_bytes{};
  std::size_t device_optin_shared_limit_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int
query_sm87_a4w4_gateup_k512_m128n128_projection_serial_resources_cuda(
    Sm87A4W4GateUpK512M128N128ProjectionSerialResources* resources)
    noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_m128n128_projection_serial_bf16_cuda(
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
launch_sm87_a4w4_gateup_k512_m128n128_projection_serial_test_bf16_cuda(
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

static_assert(
    kSm87A4W4GateUpK512M128N128ProjectionSerialDynamicSharedBytes ==
    165'376U);
static_assert(
    sm87_a4w4_gateup_k512_m128n128_projection_serial_plan(
        1'853U, 1'920U, 17'408U, 5'120U, 0U, 12'288U)
        .m_tiles == 15U);
static_assert(
    sm87_a4w4_gateup_k512_m128n128_projection_serial_plan(
        2'048U, 2'048U, 17'408U, 5'120U, 12'288U, 5'120U)
        .n_tiles == 40U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_GATEUP_M128N128_SERIAL_HOST_DEVICE
