#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_GATEUP_K512_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_GATEUP_K512_HOST_DEVICE
#endif

namespace q3x::kernels {

// Standalone throughput macrocell for the Qwen3.6-27B Gate+Up projection.
// It consumes independently quantized signed-S4/K512 A, Gate, and Up
// operands and produces one row-major BF16 SiLU(Gate)*Up matrix.  The BF16
// output is intentionally left unquantized for the existing K512 activation
// quantizer; this cell never emits a second copy of the product.
//
// One 512-thread CTA owns M64 x (Gate N128 + Up N128).  Warps 0..7 form the
// Gate crew and warps 8..15 form the matching Up crew.  Within each crew,
// four M16 strips and two N64 phases cover the complete tile.  Gate and Up
// share the single A load.  A physical stage contains four K64 code planes
// (K256): 8 KiB of A plus 16 KiB each of Gate and Up.  The two-stage ring is
// 80 KiB.  Two 640-byte scale slots allow the following K512 scale group to
// travel with its even K256 stage, for a total dynamic allocation of 83,200
// bytes and exactly one resident CTA/SM on the pinned 16-SM SM87 target.
//
// The numerical boundary, in ascending K512 group order, is:
//
//   I_projection(m,n,g) = sum_{k in K512(g)} A_s4(m,k) * B_s4(n,k)
//   scale_product = round_fp32(A_bf16_scale * B_bf16_scale)
//   projection = fma_rn(float(I_projection), scale_product, projection)
//   output_bf16 = bf16_rne(SiLU(gate_projection) * up_projection)
//
// All eight K64 MMA results are accumulated in S32 before the single FP32
// apply.  Splitting the K512 group at K64/K128/K256 is not ABI-compatible.
inline constexpr std::size_t kSm87A4W4GateUpK512MacroTileM = 64U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroTileN = 128U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroCopyK = 256U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroScaleK = 512U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroPhysicalK64 = 64U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroK64PerCopy = 4U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroCopiesPerScale = 2U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroThreads = 512U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroWarps = 16U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroProjectionWarps = 8U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroStages = 2U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroScaleSlots = 2U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroPersistentCtas = 16U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroCtasPerSm = 1U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroMaximumRegisters = 128U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroPackedK64Bytes = 32U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroOuterBlock = 64U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroAStageBytes =
    kSm87A4W4GateUpK512MacroTileM *
    kSm87A4W4GateUpK512MacroCopyK / 2U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroBStageBytes =
    kSm87A4W4GateUpK512MacroTileN *
    kSm87A4W4GateUpK512MacroCopyK / 2U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroStageBytes =
    kSm87A4W4GateUpK512MacroAStageBytes +
    2U * kSm87A4W4GateUpK512MacroBStageBytes;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroScaleSlotBytes =
    (kSm87A4W4GateUpK512MacroTileM +
     2U * kSm87A4W4GateUpK512MacroTileN) * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87A4W4GateUpK512MacroSharedBytes =
    kSm87A4W4GateUpK512MacroStages *
        kSm87A4W4GateUpK512MacroStageBytes +
    kSm87A4W4GateUpK512MacroScaleSlots *
        kSm87A4W4GateUpK512MacroScaleSlotBytes;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroModelIntermediate =
    17'408U;
inline constexpr std::size_t kSm87A4W4GateUpK512MacroModelInput = 5'120U;

[[nodiscard]] constexpr bool sm87_a4w4_gateup_k512_macro_product_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_macro_outer_block_count(
    const std::size_t outer_count) noexcept {
  return outer_count == 0U
             ? 0U
             : 1U + (outer_count - 1U) /
                        kSm87A4W4GateUpK512MacroOuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_macro_scale_group_count(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4GateUpK512MacroScaleK == 0U
             ? logical_k / kSm87A4W4GateUpK512MacroScaleK
             : 0U;
}

// K512 scale layout: [ceil(outer/64), K/512, 64] BF16 elements.  It follows
// the consumer outer-block convention but is independent data, not a pooled
// view of a K64/K128 scale plane.
[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_macro_scale_capacity_elements(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t blocks =
      sm87_a4w4_gateup_k512_macro_outer_block_count(outer_count);
  const std::size_t groups =
      sm87_a4w4_gateup_k512_macro_scale_group_count(logical_k);
  if (blocks == 0U || groups == 0U ||
      !sm87_a4w4_gateup_k512_macro_product_fits(blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  return sm87_a4w4_gateup_k512_macro_product_fits(
             block_groups, kSm87A4W4GateUpK512MacroOuterBlock)
             ? block_groups * kSm87A4W4GateUpK512MacroOuterBlock
             : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_GATEUP_K512_HOST_DEVICE constexpr std::size_t
sm87_a4w4_gateup_k512_macro_scale_offset(
    const std::size_t outer_coordinate, const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  return ((outer_coordinate / kSm87A4W4GateUpK512MacroOuterBlock) *
              k512_group_count +
          k512_group) *
             kSm87A4W4GateUpK512MacroOuterBlock +
         outer_coordinate % kSm87A4W4GateUpK512MacroOuterBlock;
}

struct Sm87A4W4GateUpK512MacroPlan final {
  std::size_t token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t n_start{};
  std::size_t n_count{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k512_groups{};
  std::size_t physical_k256_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_cells{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpK512MacroPlan
sm87_a4w4_gateup_k512_macro_plan(
    const std::size_t token_count, const std::size_t intermediate_size,
    const std::size_t input_size, const std::size_t n_start,
    const std::size_t n_count) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4GateUpK512MacroTileM != 0U ||
      intermediate_size == 0U ||
      intermediate_size % kSm87A4W4GateUpK512MacroTileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpK512MacroScaleK != 0U ||
      n_start % kSm87A4W4GateUpK512MacroTileN != 0U ||
      n_count == 0U ||
      n_count % kSm87A4W4GateUpK512MacroTileN != 0U ||
      n_start > intermediate_size ||
      n_count > intermediate_size - n_start) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4GateUpK512MacroTileM;
  const std::size_t n_tiles =
      n_count / kSm87A4W4GateUpK512MacroTileN;
  if (!sm87_a4w4_gateup_k512_macro_product_fits(m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * n_tiles;
  return {token_count,
          intermediate_size,
          input_size,
          n_start,
          n_count,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4GateUpK512MacroScaleK,
          input_size / kSm87A4W4GateUpK512MacroCopyK,
          input_size / kSm87A4W4GateUpK512MacroPhysicalK64,
          work_cells,
          m_tiles < kSm87A4W4GateUpK512MacroPersistentCtas
              ? m_tiles
              : kSm87A4W4GateUpK512MacroPersistentCtas};
}

[[nodiscard]] constexpr bool sm87_a4w4_gateup_k512_macro_is_model_plan(
    const Sm87A4W4GateUpK512MacroPlan& plan) noexcept {
  return plan.launch_ctas != 0U &&
         plan.intermediate_size ==
             kSm87A4W4GateUpK512MacroModelIntermediate &&
         plan.input_size == kSm87A4W4GateUpK512MacroModelInput;
}

struct Sm87A4W4GateUpK512MacroResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

// Success requires <=128 registers/thread, zero local storage, exactly 83,200
// launch-time dynamic shared bytes, and one active 512-thread CTA/SM on the
// pinned 16-SM SM87 target.  The query performs the real opt-in and occupancy
// calls; it does not report a compile-time estimate.
[[nodiscard]] int query_sm87_a4w4_gateup_k512_macrocell_resources_cuda(
    Sm87A4W4GateUpK512MacroResources* resources) noexcept;

// Production surface.  M may be any complete M64 token tile, while the full
// weight shape is pinned to Qwen3.6-27B N=17,408/K=5,120.  n_start/n_count
// select an N128-aligned window from those complete weight rows.  Output
// column zero corresponds to absolute weight row n_start, which lets callers
// place [0,12,288) and [12,288,17,408) directly in the existing primary and
// secondary workspaces without allocating a full Mx17,408 BF16 matrix.  Every
// storage extent is explicit; output may have an even padded row stride but
// may not alias any input range.
[[nodiscard]] int launch_sm87_a4w4_gateup_k512_macrocell_bf16_cuda(
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
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::size_t n_start,
    std::size_t n_count,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Synthetic-correctness-only surface.  It accepts complete M64/N128/K512
// cells and an explicit persistent CTA cap; its capacity, target, alignment,
// aliasing, and resource gates are otherwise identical to production.
[[nodiscard]] int launch_sm87_a4w4_gateup_k512_macrocell_test_bf16_cuda(
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

static_assert(kSm87A4W4GateUpK512MacroAStageBytes == 8'192U);
static_assert(kSm87A4W4GateUpK512MacroBStageBytes == 16'384U);
static_assert(kSm87A4W4GateUpK512MacroStageBytes == 40'960U);
static_assert(kSm87A4W4GateUpK512MacroScaleSlotBytes == 640U);
static_assert(kSm87A4W4GateUpK512MacroSharedBytes == 83'200U);
static_assert(kSm87A4W4GateUpK512MacroSharedBytes <= 96U * 1'024U);
static_assert(sm87_a4w4_gateup_k512_macro_plan(
                  2'048U, 17'408U, 5'120U, 0U, 17'408U)
                  .m_tiles == 32U);
static_assert(sm87_a4w4_gateup_k512_macro_plan(
                  2'048U, 17'408U, 5'120U, 0U, 17'408U)
                  .n_tiles == 136U);
static_assert(sm87_a4w4_gateup_k512_macro_plan(
                  2'048U, 17'408U, 5'120U, 0U, 17'408U)
                  .k512_groups == 10U);
static_assert(sm87_a4w4_gateup_k512_macro_plan(
                  2'048U, 17'408U, 5'120U, 0U, 17'408U)
                  .work_cells == 4'352U);
static_assert(sm87_a4w4_gateup_k512_macro_plan(
                  2'048U, 17'408U, 5'120U, 0U, 17'408U)
                  .launch_ctas == 16U);
static_assert(sm87_a4w4_gateup_k512_macro_plan(
                  2'048U, 17'408U, 5'120U, 0U, 12'288U)
                  .n_tiles == 96U);
static_assert(sm87_a4w4_gateup_k512_macro_plan(
                  2'048U, 17'408U, 5'120U, 12'288U, 5'120U)
                  .n_tiles == 40U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_GATEUP_K512_HOST_DEVICE
