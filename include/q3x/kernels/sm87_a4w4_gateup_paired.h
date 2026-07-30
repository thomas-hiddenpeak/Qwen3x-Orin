#pragma once

#include <cstddef>
#include <cstdint>

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
// complete M64 spans of at least 1024 tokens; C512 and all tails stay on the
// established M32N128 kernel.
inline constexpr std::size_t kSm87A4W4GateUpLargeMTileM = 64U;
inline constexpr std::size_t kSm87A4W4GateUpLargeMTileN = 64U;
inline constexpr std::size_t kSm87A4W4GateUpLargeMTileK = 64U;
inline constexpr std::size_t kSm87A4W4GateUpLargeMMinimumTokens = 1'024U;

enum class Sm87A4W4GateUpPairedKernel : std::uint8_t {
  kM32N128K64 = 0U,
  kM64N64K64 = 1U,
};

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
  const bool use_large_m =
      token_count >= kSm87A4W4GateUpLargeMMinimumTokens &&
      token_count % kSm87A4W4GateUpLargeMTileM == 0U;
  const std::size_t tile_m = use_large_m
                                 ? kSm87A4W4GateUpLargeMTileM
                                 : kSm87A4W4GateUpTileM;
  const std::size_t tile_n = use_large_m
                                 ? kSm87A4W4GateUpLargeMTileN
                                 : kSm87A4W4GateUpTileN;
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
          use_large_m ? Sm87A4W4GateUpPairedKernel::kM64N64K64
                      : Sm87A4W4GateUpPairedKernel::kM32N128K64};
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
static_assert(sm87_a4w4_gateup_paired_plan(65U, 64U, 192U)
                  .launch_ctas == 0U);
static_assert(sm87_a4w4_gateup_paired_plan(65U, 128U, 192U)
                  .m_tiles == 3U);
static_assert(sm87_a4w4_gateup_paired_plan(1U, 128U, 64U)
                  .launch_ctas == 32U);
static_assert(sm87_a4w4_gateup_paired_plan(0U, 128U, 64U)
                  .launch_ctas == 0U);

}  // namespace q3x::kernels
