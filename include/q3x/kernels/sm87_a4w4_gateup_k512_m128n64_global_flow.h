#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_GATEUP_K512_M128N64_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_GATEUP_K512_M128N64_HOST_DEVICE
#endif

namespace q3x::kernels {

// Standalone, default-off Gate+Up candidate for the Qwen3.6-27B Prefill
// shape.  A 512-thread CTA logically owns M128 x (Gate N128 + Up N128),
// executes that region as two complete-K N64 phases, shares each phase's A
// presentation between the two projections, and writes only the final BF16
// SiLU(Gate)*Up product.
//
// The shape is selected from a global traffic model, not from an occupancy
// micro-parameter sweep.  For one complete K=5120 cell, packed-code bytes
// presented by a CTA are:
//
//   M64 x (Gate N128 + Up N128): A=163,840, B=655,360, total=819,200
//   M128 x (Gate N64  + Up N64): A=327,680, B=327,680, total=655,360
//   M64 x (Gate N64  + Up N64): A=163,840, B=327,680, total=491,520
//
// Two incumbent cells and one complete candidate logical cell cover the same
// M128N128 output region.  The incumbent pair presents A=327,680 and
// B=1,310,720 bytes; the two candidate phases present A=655,360 and
// B=655,360 bytes.  This candidate therefore halves B presentation, doubles
// A presentation, and reduces aggregate packed-code presentation by 20%.
// M64N64 would require 8,704 cells after doubling the N cell count;
// its aggregate presentation is 20% above the incumbent and its 2 CTA/SM do
// not increase active warps (2*8 versus 1*16).  M128N128 cannot retain both
// K512 S32 partials and FP32 accumulators within the 128-register/thread
// block-level limit of a 512-thread CTA.  Two-phase M128N128 is thus the
// widest logical cell that improves global flow without spills or an
// intermediate projection.
//
// A raw stage holds K256: 16 KiB A plus 8 KiB each Gate and Up.  A two-stage
// ring plus two 512-byte K512 scale slots is 66,560 bytes.  Each projection
// crew has eight warps.  Warp p owns one M32N32 region: p/2 selects one of
// four M32 strips and p%2 one of two N32 halves.  This doubles A-fragment LDS
// but halves B-fragment LDS relative to M16N64, reducing aggregate fragment
// loads by 20% while preserving 32 outputs/thread.  The numerical boundary
// is identical to the authenticated K512 route:
//
//   I(m,n,g) = sum_{k in K512(g)} A_s4(m,k) * B_s4(n,k)
//   projection = fma_rn(float(I),
//                       round_fp32(A_bf16_scale*B_bf16_scale), projection)
//   output_bf16 = bf16_rne(SiLU(gate_projection) * up_projection)
//
// All eight K64 terms remain S32 until the one K512 dequantization boundary.
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64TileM = 128U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64TileN = 128U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64PhaseN = 64U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64PhasesPerTile = 2U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64CopyK = 256U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64ScaleK = 512U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64PhysicalK64 = 64U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64K64PerCopy = 4U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64Threads = 512U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64Warps = 16U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64ProjectionWarps = 8U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64Stages = 2U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64ScaleSlots = 2U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64PersistentCtas = 16U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64CtasPerSm = 1U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64MaximumRegisters = 128U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64PackedK64Bytes = 32U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64OuterBlock = 64U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64AStageBytes =
    kSm87A4W4GateUpK512M128N64TileM *
    kSm87A4W4GateUpK512M128N64CopyK / 2U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64BStageBytes =
    kSm87A4W4GateUpK512M128N64PhaseN *
    kSm87A4W4GateUpK512M128N64CopyK / 2U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64StageBytes =
    kSm87A4W4GateUpK512M128N64AStageBytes +
    2U * kSm87A4W4GateUpK512M128N64BStageBytes;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64ScaleSlotBytes =
    (kSm87A4W4GateUpK512M128N64TileM +
     2U * kSm87A4W4GateUpK512M128N64PhaseN) * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64SharedBytes =
    kSm87A4W4GateUpK512M128N64Stages *
        kSm87A4W4GateUpK512M128N64StageBytes +
    kSm87A4W4GateUpK512M128N64ScaleSlots *
        kSm87A4W4GateUpK512M128N64ScaleSlotBytes;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64ModelIntermediate =
    17'408U;
inline constexpr std::size_t kSm87A4W4GateUpK512M128N64ModelInput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64IncumbentPairABytes =
        2U * 64U * kSm87A4W4GateUpK512M128N64ModelInput / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64IncumbentPairBBytes =
        2U * 2U * 128U * kSm87A4W4GateUpK512M128N64ModelInput / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64CandidateLogicalABytes =
        kSm87A4W4GateUpK512M128N64PhasesPerTile *
        kSm87A4W4GateUpK512M128N64TileM *
        kSm87A4W4GateUpK512M128N64ModelInput / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N64CandidateLogicalBBytes =
        kSm87A4W4GateUpK512M128N64PhasesPerTile * 2U *
        kSm87A4W4GateUpK512M128N64PhaseN *
        kSm87A4W4GateUpK512M128N64ModelInput / 2U;

[[nodiscard]] constexpr bool sm87_a4w4_gateup_k512_m128n64_product_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_m128n64_outer_block_count(
    const std::size_t outer_count) noexcept {
  return outer_count == 0U
             ? 0U
             : 1U + (outer_count - 1U) /
                        kSm87A4W4GateUpK512M128N64OuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_m128n64_scale_group_count(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4GateUpK512M128N64ScaleK == 0U
             ? logical_k / kSm87A4W4GateUpK512M128N64ScaleK
             : 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_m128n64_scale_capacity_elements(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t blocks =
      sm87_a4w4_gateup_k512_m128n64_outer_block_count(outer_count);
  const std::size_t groups =
      sm87_a4w4_gateup_k512_m128n64_scale_group_count(logical_k);
  if (blocks == 0U || groups == 0U ||
      !sm87_a4w4_gateup_k512_m128n64_product_fits(blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  return sm87_a4w4_gateup_k512_m128n64_product_fits(
             block_groups, kSm87A4W4GateUpK512M128N64OuterBlock)
             ? block_groups * kSm87A4W4GateUpK512M128N64OuterBlock
             : 0U;
}

[[nodiscard]]
Q3X_SM87_A4W4_GATEUP_K512_M128N64_HOST_DEVICE constexpr std::size_t
sm87_a4w4_gateup_k512_m128n64_scale_offset(
    const std::size_t outer_coordinate, const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  return ((outer_coordinate / kSm87A4W4GateUpK512M128N64OuterBlock) *
              k512_group_count +
          k512_group) *
             kSm87A4W4GateUpK512M128N64OuterBlock +
         outer_coordinate % kSm87A4W4GateUpK512M128N64OuterBlock;
}

struct Sm87A4W4GateUpK512M128N64Plan final {
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
  std::size_t minimum_cells_per_cta{};
  std::size_t maximum_cells_per_cta{};
  std::size_t b_wave_n_span_bound{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpK512M128N64Plan
sm87_a4w4_gateup_k512_m128n64_plan(
    const std::size_t token_count, const std::size_t intermediate_size,
    const std::size_t input_size, const std::size_t n_start,
    const std::size_t n_count) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4GateUpK512M128N64TileM != 0U ||
      intermediate_size == 0U ||
      intermediate_size % kSm87A4W4GateUpK512M128N64TileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpK512M128N64ScaleK != 0U ||
      n_start % kSm87A4W4GateUpK512M128N64TileN != 0U ||
      n_count == 0U ||
      n_count % kSm87A4W4GateUpK512M128N64TileN != 0U ||
      n_start > intermediate_size ||
      n_count > intermediate_size - n_start) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4GateUpK512M128N64TileM;
  const std::size_t n_tiles =
      n_count / kSm87A4W4GateUpK512M128N64TileN;
  if (!sm87_a4w4_gateup_k512_m128n64_product_fits(m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      work_cells < kSm87A4W4GateUpK512M128N64PersistentCtas
          ? work_cells
          : kSm87A4W4GateUpK512M128N64PersistentCtas;
  const std::size_t minimum_cells_per_cta = work_cells / launch_ctas;
  const std::size_t maximum_cells_per_cta =
      minimum_cells_per_cta + (work_cells % launch_ctas != 0U ? 1U : 0U);
  // Work is N-major.  A contiguous CTA round therefore stays inside this
  // conservative number of adjacent N tiles.  Natural P1920/P2176 both
  // bound the B wave to two adjacent windows while avoiding an M-tail sweep.
  const std::size_t b_wave_n_span_bound =
      1U + (launch_ctas - 1U + m_tiles - 1U) / m_tiles;
  return {token_count,
          intermediate_size,
          input_size,
          n_start,
          n_count,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4GateUpK512M128N64ScaleK,
          input_size / kSm87A4W4GateUpK512M128N64CopyK,
          input_size / kSm87A4W4GateUpK512M128N64PhysicalK64,
          work_cells,
          launch_ctas,
          minimum_cells_per_cta,
          maximum_cells_per_cta,
          b_wave_n_span_bound};
}

[[nodiscard]] constexpr bool
sm87_a4w4_gateup_k512_m128n64_is_model_plan(
    const Sm87A4W4GateUpK512M128N64Plan& plan) noexcept {
  return plan.launch_ctas != 0U &&
         plan.intermediate_size ==
             kSm87A4W4GateUpK512M128N64ModelIntermediate &&
         plan.input_size == kSm87A4W4GateUpK512M128N64ModelInput;
}

struct Sm87A4W4GateUpK512M128N64Resources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

// Success requires <=128 registers/thread, zero local storage, exactly 66,560
// dynamic shared bytes, and at least one active 512-thread CTA/SM on the
// pinned 16-SM SM87 target.  This is a runtime query, not an estimate.
[[nodiscard]] int query_sm87_a4w4_gateup_k512_m128n64_resources_cuda(
    Sm87A4W4GateUpK512M128N64Resources* resources) noexcept;

// Default-off production-shape launch surface.  The model dimensions are
// pinned to N=17,408/K=5,120; M may be any complete M128 tile in [128,4096].
// n_start/n_count select an N128-aligned output window; each N128 cell is
// internally executed as two N64 phases without retaining both accumulators.
[[nodiscard]] int launch_sm87_a4w4_gateup_k512_m128n64_bf16_cuda(
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

// Synthetic-correctness-only surface.  It accepts complete M128/N128/K512
// cells and an explicit CTA cap, but applies the same capacity, alignment,
// alias, target, and resource gates as the production-shape surface.
[[nodiscard]] int launch_sm87_a4w4_gateup_k512_m128n64_test_bf16_cuda(
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

static_assert(kSm87A4W4GateUpK512M128N64AStageBytes == 16'384U);
static_assert(kSm87A4W4GateUpK512M128N64BStageBytes == 8'192U);
static_assert(kSm87A4W4GateUpK512M128N64StageBytes == 32'768U);
static_assert(kSm87A4W4GateUpK512M128N64ScaleSlotBytes == 512U);
static_assert(kSm87A4W4GateUpK512M128N64SharedBytes == 66'560U);
static_assert(kSm87A4W4GateUpK512M128N64SharedBytes <= 80U * 1'024U);
static_assert(kSm87A4W4GateUpK512M128N64TileN ==
              kSm87A4W4GateUpK512M128N64PhasesPerTile *
                  kSm87A4W4GateUpK512M128N64PhaseN);
static_assert(kSm87A4W4GateUpK512M128N64IncumbentPairABytes == 327'680U);
static_assert(kSm87A4W4GateUpK512M128N64IncumbentPairBBytes == 1'310'720U);
static_assert(kSm87A4W4GateUpK512M128N64CandidateLogicalABytes == 655'360U);
static_assert(kSm87A4W4GateUpK512M128N64CandidateLogicalBBytes == 655'360U);
static_assert(
    5U * (kSm87A4W4GateUpK512M128N64CandidateLogicalABytes +
          kSm87A4W4GateUpK512M128N64CandidateLogicalBBytes) ==
    4U * (kSm87A4W4GateUpK512M128N64IncumbentPairABytes +
          kSm87A4W4GateUpK512M128N64IncumbentPairBBytes));
static_assert(sm87_a4w4_gateup_k512_m128n64_plan(
                  2'048U, 17'408U, 5'120U, 0U, 17'408U)
                  .m_tiles == 16U);
static_assert(sm87_a4w4_gateup_k512_m128n64_plan(
                  2'048U, 17'408U, 5'120U, 0U, 17'408U)
                  .n_tiles == 136U);
static_assert(sm87_a4w4_gateup_k512_m128n64_plan(
                  2'048U, 17'408U, 5'120U, 0U, 17'408U)
                  .work_cells == 2'176U);
static_assert(sm87_a4w4_gateup_k512_m128n64_plan(
                  128U, 17'408U, 5'120U, 0U, 17'408U)
                  .launch_ctas == 16U);
static_assert(sm87_a4w4_gateup_k512_m128n64_plan(
                  1'920U, 17'408U, 5'120U, 0U, 17'408U)
                  .maximum_cells_per_cta == 128U);
static_assert(sm87_a4w4_gateup_k512_m128n64_plan(
                  2'176U, 17'408U, 5'120U, 0U, 17'408U)
                  .maximum_cells_per_cta == 145U);
static_assert(sm87_a4w4_gateup_k512_m128n64_plan(
                  2'176U, 17'408U, 5'120U, 0U, 17'408U)
                  .b_wave_n_span_bound == 2U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_GATEUP_K512_M128N64_HOST_DEVICE
