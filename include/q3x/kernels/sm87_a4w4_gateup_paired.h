#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_GATEUP_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_GATEUP_HOST_DEVICE
#endif

namespace q3x::kernels {

// Test-admission-only paired Gate+Up projection for the Qwen3.6 Prefill
// plane.  The kernel consumes one consumer-order A4 activation and two W4
// matrices, computes SiLU(Gate) * Up without materializing either BF16
// projection, and writes the same consumer-order A4/K64 activation ABI used
// directly by the Down projection.
//
//   A:             signed A4 [ceil(M/64),K/64,64,32]
//   A scales:      BF16      [ceil(M/64),K/64,64]
//   Gate/Up B:     signed W4 [N/64,K/64,64,32]
//   Gate/Up scales BF16      [N/64,K/64,64]
//   output:        signed A4 [ceil(M/64),N/64,64,32]
//   output scales: BF16      [ceil(M/64),N/64,64]
//
// This surface is intentionally absent from q3x_kernels, the runner, and all
// production selectors.  Synthetic inputs establish correctness only; they
// are not a performance admission payload.
inline constexpr std::size_t kSm87A4W4GateUpTileM = 32U;
inline constexpr std::size_t kSm87A4W4GateUpTileN = 128U;
inline constexpr std::size_t kSm87A4W4GateUpTileK = 64U;
inline constexpr std::size_t kSm87A4W4GateUpThreads = 256U;
inline constexpr std::size_t kSm87A4W4GateUpWarps = 8U;
inline constexpr std::size_t kSm87A4W4GateUpPipelineStages = 3U;
inline constexpr std::size_t kSm87A4W4GateUpPersistentCtas = 32U;
inline constexpr std::size_t kSm87A4W4GateUpCtasPerSm = 2U;
inline constexpr std::size_t kSm87A4W4GateUpPackedOutputTileRowBytes =
    kSm87A4W4GateUpTileN / 2U;

// Large-M candidate.  Keep the per-warp M16xN32 ownership and accumulator
// footprint unchanged while exchanging CTA-level N reuse for twice as much M
// reuse of each staged Gate/Up row.  The launcher admits this shape only for
// complete M64 prefixes of at least 1024 tokens; any residual 1..63 rows stay
// on the established M32N128 kernel in the same public launch.
inline constexpr std::size_t kSm87A4W4GateUpLargeMTileM = 64U;
inline constexpr std::size_t kSm87A4W4GateUpLargeMTileN = 64U;
inline constexpr std::size_t kSm87A4W4GateUpLargeMTileK = 64U;
inline constexpr std::size_t kSm87A4W4GateUpLargeMMinimumTokens = 1'024U;

// Whole-M Gate+Up candidate for spans at and above P2048.  Eight warps own a
// 2x4 array of M32xN32 warp tiles.  Relative to M64N64 this keeps each A K64
// plane resident while consuming twice as many Gate/Up rows, reducing the
// work-tile count without changing the consumer-order A4/K64 ABI.
inline constexpr std::size_t kSm87A4W4GateUpWideLargeMTileM = 64U;
inline constexpr std::size_t kSm87A4W4GateUpWideLargeMTileN = 128U;
inline constexpr std::size_t kSm87A4W4GateUpWideLargeMTileK = 64U;
inline constexpr std::size_t kSm87A4W4GateUpWideLargeMLogicalTileK = 128U;
inline constexpr std::size_t kSm87A4W4GateUpWideLargeMPipelineSlots = 2U;
inline constexpr std::size_t kSm87A4W4GateUpWideLargeMMinimumTokens = 2'048U;

// Independent K128 shared-scale Gate+Up vertical slice.  The packed-code
// planes retain the two physical K64 blocks per logical K128 group, while
// A, Gate, Up, and the fused Down-input output each own one BF16 scale per
// K128 group.  In particular, the epilogue reduces all 128 SiLU(Gate)*Up
// values before it emits the one scale consumed by the K128 Down GEMM; it
// does not reuse either of the established per-K64 output scales.
//
// This API is additive and is never selected by the K64 planner/launcher.
inline constexpr std::size_t kSm87A4W4GateUpK128TileM = 64U;
inline constexpr std::size_t kSm87A4W4GateUpK128TileN = 128U;
inline constexpr std::size_t kSm87A4W4GateUpK128TileK = 128U;
inline constexpr std::size_t kSm87A4W4GateUpK128PipelineSlots = 2U;
inline constexpr std::size_t kSm87A4W4GateUpK128MaximumRegisters = 128U;

enum class Sm87A4W4GateUpPairedKernel : std::uint8_t {
  kM32N128K64 = 0U,
  kM64N64K64 = 1U,
  kM64N128K64 = 2U,
};

struct Sm87A4W4GateUpPairedWorkTile final {
  std::size_t m_tile{};
  std::size_t n_tile{};
  bool valid{};
};

// Persistent CTAs traverse M tiles consecutively inside one N tile.  The
// mapping is deliberately independent of any kernel body so contract tests
// can pin the N-major schedule and every candidate consumes the same planner.
[[nodiscard]] Q3X_SM87_A4W4_GATEUP_HOST_DEVICE constexpr
Sm87A4W4GateUpPairedWorkTile sm87_a4w4_gateup_paired_n_major_work_tile(
    const std::size_t work_tile, const std::size_t m_tile_count,
    const std::size_t work_tile_count) noexcept {
  if (m_tile_count == 0U || work_tile >= work_tile_count) {
    return {};
  }
  const std::size_t n_tile = work_tile / m_tile_count;
  return {work_tile - n_tile * m_tile_count, n_tile, true};
}

struct Sm87A4W4GateUpPairedPlan final {
  std::size_t token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
  std::size_t packed_output_row_bytes{};
  std::size_t output_scale_row_elements{};
  std::size_t tile_m{};
  std::size_t tile_n{};
  Sm87A4W4GateUpPairedKernel kernel{
      Sm87A4W4GateUpPairedKernel::kM32N128K64};
};

struct Sm87A4W4GateUpPairedK64CompositePlan final {
  bool valid{};
  std::size_t token_count{};
  std::size_t prefix_token_count{};
  std::size_t tail_token_count{};
  Sm87A4W4GateUpPairedPlan prefix_plan{};
  Sm87A4W4GateUpPairedPlan tail_plan{};
};

struct Sm87A4W4GateUpPairedK128Plan final {
  std::size_t token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k128_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
  std::size_t packed_output_row_bytes{};
  std::size_t output_scale_row_elements{};
};

[[nodiscard]] constexpr std::size_t sm87_a4w4_gateup_ceil_div(
    const std::size_t numerator, const std::size_t denominator) noexcept {
  return denominator == 0U || numerator == 0U
             ? 0U
             : 1U + (numerator - 1U) / denominator;
}

[[nodiscard]] constexpr Sm87A4W4GateUpPairedPlan
sm87_a4w4_gateup_paired_plan(const std::size_t token_count,
                             const std::size_t intermediate_size,
                             const std::size_t input_size) noexcept {
  const bool use_wide_large_m =
      token_count >= kSm87A4W4GateUpWideLargeMMinimumTokens &&
      token_count % kSm87A4W4GateUpWideLargeMTileM == 0U &&
      intermediate_size % kSm87A4W4GateUpWideLargeMTileN == 0U;
  const bool use_large_m = !use_wide_large_m &&
      token_count >= kSm87A4W4GateUpLargeMMinimumTokens &&
      token_count % kSm87A4W4GateUpLargeMTileM == 0U;
  const std::size_t tile_m =
      use_wide_large_m ? kSm87A4W4GateUpWideLargeMTileM
                       : (use_large_m ? kSm87A4W4GateUpLargeMTileM
                                      : kSm87A4W4GateUpTileM);
  const std::size_t tile_n =
      use_wide_large_m ? kSm87A4W4GateUpWideLargeMTileN
                       : (use_large_m ? kSm87A4W4GateUpLargeMTileN
                                      : kSm87A4W4GateUpTileN);
  if (token_count == 0U || intermediate_size == 0U || input_size == 0U ||
      intermediate_size % tile_n != 0U ||
      input_size % kSm87A4W4GateUpTileK != 0U) {
    return {};
  }
  const std::size_t m_tiles = sm87_a4w4_gateup_ceil_div(
      token_count, tile_m);
  const std::size_t n_tiles = intermediate_size / tile_n;
  if (m_tiles == 0U || n_tiles == 0U ||
      m_tiles > static_cast<std::size_t>(-1) / n_tiles) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {token_count,
          intermediate_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4GateUpTileK,
          work_tiles,
          kSm87A4W4GateUpPersistentCtas,
          intermediate_size / 2U,
          intermediate_size / kSm87A4W4GateUpTileK,
          tile_m,
          tile_n,
          use_wide_large_m
              ? Sm87A4W4GateUpPairedKernel::kM64N128K64
              : (use_large_m
                     ? Sm87A4W4GateUpPairedKernel::kM64N64K64
                     : Sm87A4W4GateUpPairedKernel::kM32N128K64)};
}

// K64-only composition plan.  The largest threshold-qualified M64 prefix
// retains the existing M64N128 or M64N64 kernel selected by its exact plan;
// the residual consumer block uses the existing M32N128 tail kernel.
[[nodiscard]] constexpr Sm87A4W4GateUpPairedK64CompositePlan
sm87_a4w4_gateup_paired_k64_composite_plan(
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  Sm87A4W4GateUpPairedK64CompositePlan result{};
  const Sm87A4W4GateUpPairedPlan full_plan =
      sm87_a4w4_gateup_paired_plan(token_count, intermediate_size,
                                  input_size);
  if (full_plan.launch_ctas == 0U) {
    return result;
  }

  result.valid = true;
  result.token_count = token_count;
  const std::size_t maximum_m64_prefix =
      token_count / kSm87A4W4GateUpLargeMTileM *
      kSm87A4W4GateUpLargeMTileM;
  if (maximum_m64_prefix >= kSm87A4W4GateUpLargeMMinimumTokens) {
    result.prefix_plan = sm87_a4w4_gateup_paired_plan(
        maximum_m64_prefix, intermediate_size, input_size);
    if (result.prefix_plan.launch_ctas != 0U &&
        result.prefix_plan.tile_m == kSm87A4W4GateUpLargeMTileM) {
      result.prefix_token_count = maximum_m64_prefix;
      result.tail_token_count = token_count - maximum_m64_prefix;
    }
  }
  if (result.prefix_token_count == 0U) {
    result.tail_token_count = token_count;
    result.tail_plan = full_plan;
    return result;
  }
  if (result.tail_token_count != 0U) {
    result.tail_plan = sm87_a4w4_gateup_paired_plan(
        result.tail_token_count, intermediate_size, input_size);
    if (result.tail_plan.launch_ctas == 0U ||
        result.tail_plan.kernel !=
            Sm87A4W4GateUpPairedKernel::kM32N128K64) {
      return {};
    }
  }
  return result;
}

[[nodiscard]] constexpr Sm87A4W4GateUpPairedK128Plan
sm87_a4w4_gateup_paired_k128_plan(
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4GateUpK128TileM != 0U ||
      intermediate_size == 0U ||
      intermediate_size % kSm87A4W4GateUpK128TileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpK128TileK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4GateUpK128TileM;
  const std::size_t n_tiles =
      intermediate_size / kSm87A4W4GateUpK128TileN;
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  if (m_tiles == 0U || n_tiles == 0U || m_tiles > maximum / n_tiles) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {token_count,
          intermediate_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4GateUpK128TileK,
          input_size / kSm87A4W4GateUpTileK,
          work_tiles,
          work_tiles < kSm87A4W4GateUpPersistentCtas
              ? work_tiles
              : kSm87A4W4GateUpPersistentCtas,
          intermediate_size / 2U,
          intermediate_size / kSm87A4W4GateUpK128TileK};
}

struct Sm87A4W4GateUpPairedResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

// The query is an admission gate, not an informational best-effort query.  It
// fails with cudaErrorLaunchOutOfResources unless the compiled kernel has no
// local-memory/stack frame and admits at least two resident CTAs per SM.
[[nodiscard]] int query_sm87_a4w4_gateup_paired_resources_cuda(
    Sm87A4W4GateUpPairedResources* resources) noexcept;

// Independent resource gate for the large-M candidate.  It has the same hard
// requirements as the established path: no local-memory frame/spill and at
// least two resident 256-thread CTAs per SM87 SM.
[[nodiscard]] int query_sm87_a4w4_gateup_paired_large_m_resources_cuda(
    Sm87A4W4GateUpPairedResources* resources) noexcept;

// Independent admission query for the P2048+ M64N128 candidate.  The kernel
// is rejected unless its compiled image has no local frame/spill and admits
// two resident 256-thread CTAs per SM87 SM.
[[nodiscard]] int
query_sm87_a4w4_gateup_paired_wide_large_m_resources_cuda(
    Sm87A4W4GateUpPairedResources* resources) noexcept;

// Independent admission query for the M64N128 shared-scale K128 kernel.
// Success requires <=128 registers/thread, <=48 KiB shared memory, no local
// frame/spill, and at least two resident 256-thread CTAs per SM87 SM.
[[nodiscard]] int query_sm87_a4w4_gateup_paired_k128_resources_cuda(
    Sm87A4W4GateUpPairedResources* resources) noexcept;

// output_clip_ratio is calibration-owned and must be in (0, 1].  Quantized
// output uses symmetric [-7, 7] codes.  M tails are zero-filled during async
// staging and are never written to the output buffers.
[[nodiscard]] int launch_sm87_a4w4_gateup_paired_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k64_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_gate_b,
    std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* gate_b_k64_scales_bf16,
    std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* packed_up_b,
    std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* up_b_k64_scales_bf16,
    std::size_t up_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    float output_clip_ratio,
    std::uint8_t* packed_output,
    std::size_t packed_output_capacity_bytes,
    std::uint16_t* output_k64_scales_bf16,
    std::size_t output_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// M64N128 shared-scale K128 Gate+Up projection and Down-input producer.
// Packed codes preserve the established physical K64 consumer layout.  A,
// Gate, and Up scale capacities use
// sm87_a4w4_consumer_k128_scale_capacity_elements(outer,K); output scale
// capacity uses the same helper for (M,N).  Each logical K128 contribution
// performs two native K64 MMAs into one S32 partial followed by one I2F and
// one scale-product application.  The epilogue publishes one output scale
// for both adjacent physical K64 blocks, ready for the K128 Down projection.
[[nodiscard]] int launch_sm87_a4w4_gateup_paired_k128_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k128_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_gate_b,
    std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* gate_b_k128_scales_bf16,
    std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* packed_up_b,
    std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* up_b_k128_scales_bf16,
    std::size_t up_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    float output_clip_ratio,
    std::uint8_t* packed_output,
    std::size_t packed_output_capacity_bytes,
    std::uint16_t* output_k128_scales_bf16,
    std::size_t output_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

static_assert(sm87_a4w4_gateup_paired_plan(512U, 17'408U, 5'120U)
                  .work_tiles == 2'176U);
static_assert(sm87_a4w4_gateup_paired_plan(512U, 17'408U, 5'120U)
                  .launch_ctas == 32U);
static_assert(sm87_a4w4_gateup_paired_plan(1'024U, 17'408U, 5'120U)
                  .kernel == Sm87A4W4GateUpPairedKernel::kM64N64K64);
static_assert(sm87_a4w4_gateup_paired_plan(1'024U, 17'408U, 5'120U)
                  .m_tiles == 16U);
static_assert(sm87_a4w4_gateup_paired_plan(1'024U, 17'408U, 5'120U)
                  .n_tiles == 272U);
static_assert(sm87_a4w4_gateup_paired_plan(1'088U, 17'408U, 5'120U)
                  .kernel == Sm87A4W4GateUpPairedKernel::kM64N64K64);
static_assert(sm87_a4w4_gateup_paired_plan(1'025U, 17'408U, 5'120U)
                  .kernel == Sm87A4W4GateUpPairedKernel::kM32N128K64);
static_assert(sm87_a4w4_gateup_paired_k64_composite_plan(
                  1'025U, 17'408U, 5'120U)
                  .prefix_plan.kernel ==
              Sm87A4W4GateUpPairedKernel::kM64N64K64);
static_assert(sm87_a4w4_gateup_paired_k64_composite_plan(
                  2'049U, 17'408U, 5'120U)
                  .prefix_plan.kernel ==
              Sm87A4W4GateUpPairedKernel::kM64N128K64);
static_assert(sm87_a4w4_gateup_paired_k64_composite_plan(
                  3'987U, 17'408U, 5'120U)
                  .prefix_token_count == 3'968U);
static_assert(sm87_a4w4_gateup_paired_k64_composite_plan(
                  3'987U, 17'408U, 5'120U)
                  .tail_token_count == 19U);
static_assert(sm87_a4w4_gateup_paired_plan(2'048U, 17'408U, 5'120U)
                  .kernel == Sm87A4W4GateUpPairedKernel::kM64N128K64);
static_assert(sm87_a4w4_gateup_paired_plan(2'048U, 17'408U, 5'120U)
                  .work_tiles == 4'352U);
static_assert(sm87_a4w4_gateup_paired_plan(2'048U, 17'344U, 5'120U)
                  .kernel == Sm87A4W4GateUpPairedKernel::kM64N64K64);
static_assert(sm87_a4w4_gateup_paired_n_major_work_tile(0U, 32U, 4'352U)
                  .m_tile == 0U);
static_assert(sm87_a4w4_gateup_paired_n_major_work_tile(31U, 32U, 4'352U)
                  .n_tile == 0U);
static_assert(sm87_a4w4_gateup_paired_n_major_work_tile(32U, 32U, 4'352U)
                  .n_tile == 1U);
static_assert(!sm87_a4w4_gateup_paired_n_major_work_tile(
                   4'352U, 32U, 4'352U)
                   .valid);
static_assert(sm87_a4w4_gateup_paired_plan(65U, 64U, 192U)
                  .launch_ctas == 0U);
static_assert(sm87_a4w4_gateup_paired_plan(65U, 128U, 192U)
                  .m_tiles == 3U);
static_assert(sm87_a4w4_gateup_paired_plan(1U, 128U, 64U)
                  .launch_ctas == 32U);
static_assert(sm87_a4w4_gateup_paired_plan(0U, 128U, 64U)
                  .launch_ctas == 0U);
static_assert(sm87_a4w4_gateup_paired_k128_plan(64U, 128U, 128U)
                  .work_tiles == 1U);
static_assert(sm87_a4w4_gateup_paired_k128_plan(
                  2'048U, 17'408U, 5'120U)
                  .k128_groups == 40U);
static_assert(sm87_a4w4_gateup_paired_k128_plan(
                  2'048U, 17'408U, 5'120U)
                  .physical_k64_groups == 80U);
static_assert(sm87_a4w4_gateup_paired_k128_plan(
                  2'048U, 17'408U, 5'120U)
                  .work_tiles == 4'352U);
static_assert(sm87_a4w4_gateup_paired_k128_plan(65U, 128U, 128U)
                  .launch_ctas == 0U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_GATEUP_HOST_DEVICE
