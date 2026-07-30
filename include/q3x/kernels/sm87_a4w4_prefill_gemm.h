#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Experimental full-tile A4W4 Prefill GEMM.  This is the first complete
// consumer of the native SM87 m16n8k64 primitive; it deliberately owns only
// the integer projection boundary and does not select itself in the runner.
//
// A and B use the signed-nibble consumer-block ABI shared with the calibrated
// full-model sidecar:
//
//   A: [ceil(M/64), K/64, 64, 32] bytes
//   B: [N/64, K/64, 64, 32] bytes
//   A scales: BF16 [ceil(M/64), K/64, 64]
//   B scales: BF16 [N/64, K/64, 64]
//   D: BF16 [M, N]
//
// M64/N64 are physical blocks only. The CTA combines one M32 half-block and
// two adjacent N64 blocks into M32N128, reducing activation rescans while
// retaining the two-CTA/SM resource contract. Shared packed rows use the
// primitive's XOR half-row swizzle; global consumer blocks stay contiguous.
//
// Every K64 integer partial is dequantized before it is accumulated in FP32;
// scales from distinct groups are never incorrectly factored across K.
inline constexpr std::size_t kSm87A4W4PrefillTileM = 32U;
inline constexpr std::size_t kSm87A4W4PrefillTileN = 128U;
inline constexpr std::size_t kSm87A4W4PrefillTileK = 64U;
inline constexpr std::size_t kSm87A4W4PrefillThreads = 256U;
inline constexpr std::size_t kSm87A4W4PrefillWarps = 8U;
inline constexpr std::size_t kSm87A4W4PrefillPipelineStages = 3U;
inline constexpr std::size_t kSm87A4W4PrefillPersistentCtas = 32U;
inline constexpr std::size_t kSm87A4W4PrefillCtasPerSm = 2U;

// Whole-span large-M candidate.  It preserves the public consumer/output ABI
// and the 256-thread persistent launch contract, but assigns the eight warps
// as 4x2 M16N32 warp tiles.  Relative to M32N128, M64N64 halves packed-B and
// B-scale staging per output while accepting the corresponding A rescan.  It
// is deliberately selected only for exact M64 spans: C512 and all tails stay
// on the established M32N128 kernel.
inline constexpr std::size_t kSm87A4W4PrefillLargeMTileM = 64U;
inline constexpr std::size_t kSm87A4W4PrefillLargeMTileN = 64U;
inline constexpr std::size_t kSm87A4W4PrefillLargeMMinimumTokens = 1'024U;

// Whole-M wide-N candidate.  Each CTA owns M64N256 and assigns the eight
// warps as 2x4 M32N64 warp tiles.  Two adjacent K64 groups share one logical
// pipeline buffer without changing their accumulation order.  The candidate
// is admitted only for exact M64/N256 shapes at P>=2048; P1024 remains on the
// M64N64 kernel and C512/tails remain on M32N128.
inline constexpr std::size_t kSm87A4W4PrefillWideTileM = 64U;
inline constexpr std::size_t kSm87A4W4PrefillWideTileN = 256U;
inline constexpr std::size_t kSm87A4W4PrefillWideMinimumTokens = 2'048U;
inline constexpr std::size_t kSm87A4W4PrefillWideLogicalTileK = 128U;
inline constexpr std::size_t kSm87A4W4PrefillWidePipelineStages = 2U;

[[nodiscard]] constexpr bool sm87_a4w4_prefill_uses_large_m_candidate(
    const std::size_t token_count) noexcept {
  return token_count >= kSm87A4W4PrefillLargeMMinimumTokens &&
         token_count % kSm87A4W4PrefillLargeMTileM == 0U;
}

[[nodiscard]] constexpr bool sm87_a4w4_prefill_uses_m64n256_candidate(
    const std::size_t token_count,
    const std::size_t output_size) noexcept {
  return token_count >= kSm87A4W4PrefillWideMinimumTokens &&
         token_count % kSm87A4W4PrefillWideTileM == 0U &&
         output_size != 0U &&
         output_size % kSm87A4W4PrefillWideTileN == 0U;
}

struct Sm87A4W4PrefillGemmPlan final {
  std::size_t token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr std::size_t sm87_a4w4_ceil_div(
    const std::size_t numerator, const std::size_t denominator) noexcept {
  return denominator == 0U || numerator == 0U
             ? 0U
             : 1U + (numerator - 1U) / denominator;
}

[[nodiscard]] constexpr Sm87A4W4PrefillGemmPlan
sm87_a4w4_prefill_gemm_plan(const std::size_t token_count,
                            const std::size_t output_size,
                            const std::size_t input_size) noexcept {
  if (token_count == 0U || output_size == 0U || input_size == 0U ||
      output_size % kSm87A4W4PrefillTileN != 0U ||
      input_size % kSm87A4W4PrefillTileK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      sm87_a4w4_ceil_div(token_count, kSm87A4W4PrefillTileM);
  const std::size_t n_tiles = output_size / kSm87A4W4PrefillTileN;
  if (m_tiles == 0U || n_tiles == 0U ||
      m_tiles > static_cast<std::size_t>(-1) / n_tiles) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4PrefillTileK,
          work_tiles,
          work_tiles < kSm87A4W4PrefillPersistentCtas
              ? work_tiles
              : kSm87A4W4PrefillPersistentCtas};
}

// The wide tile changes the number and ordering of persistent work items, so
// it deliberately owns a separate planner instead of reinterpreting the
// baseline M32N128 plan in the launcher.
[[nodiscard]] constexpr Sm87A4W4PrefillGemmPlan
sm87_a4w4_prefill_gemm_m64n256_plan(
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (!sm87_a4w4_prefill_uses_m64n256_candidate(token_count, output_size) ||
      input_size == 0U || input_size % kSm87A4W4PrefillTileK != 0U) {
    return {};
  }
  const std::size_t m_tiles = token_count / kSm87A4W4PrefillWideTileM;
  const std::size_t n_tiles = output_size / kSm87A4W4PrefillWideTileN;
  if (m_tiles == 0U || n_tiles == 0U ||
      m_tiles > static_cast<std::size_t>(-1) / n_tiles) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4PrefillTileK,
          work_tiles,
          work_tiles < kSm87A4W4PrefillPersistentCtas
              ? work_tiles
              : kSm87A4W4PrefillPersistentCtas};
}

struct Sm87A4W4PrefillGemmResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

// Bring-up activation producer.  Eight warps independently quantize eight
// [row,K64] groups per CTA. clip_ratio is calibration-controlled and must be
// in (0,1]. The producer writes the consumer-block layout above with an
// addressable final M64 block; unused tail rows are never consumed. Capacity
// arguments cover that padded representation and are checked before launch.
[[nodiscard]] int launch_sm87_a4_quantize_bf16_cuda(
    const std::uint16_t* input_bf16,
    std::size_t input_row_stride_elements,
    std::size_t token_count,
    std::size_t input_size,
    float clip_ratio,
    std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    std::uint16_t* a_k64_scales_bf16,
    std::size_t a_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int query_sm87_a4w4_prefill_gemm_resources_cuda(
    Sm87A4W4PrefillGemmResources* resources) noexcept;

// Queries the independently compiled M64N64 candidate.  Keeping this separate
// from the baseline query makes the >=2 CTA/SM and zero-local-spill admission
// auditable without changing the established resource-query ABI.
[[nodiscard]] int query_sm87_a4w4_prefill_gemm_m64n64_resources_cuda(
    Sm87A4W4PrefillGemmResources* resources) noexcept;

// Queries the independently compiled M64N256/K128-logical candidate.
[[nodiscard]] int query_sm87_a4w4_prefill_gemm_m64n256_resources_cuda(
    Sm87A4W4PrefillGemmResources* resources) noexcept;

// Launches one persistent projection.  The current admission accepts arbitrary
// positive M, all fixed Qwen3.6 projection N values (multiples of 128), and K
// values divisible by 64.  Tail M rows are zero-filled in the async staging
// path and never written outside D.
[[nodiscard]] int launch_sm87_a4w4_prefill_gemm_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k64_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k64_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    void* cuda_stream = nullptr) noexcept;

static_assert(sm87_a4w4_prefill_gemm_plan(512U, 17'408U, 5'120U)
                  .work_tiles == 2'176U);
static_assert(sm87_a4w4_prefill_gemm_plan(512U, 5'120U, 17'408U)
                  .work_tiles == 640U);
static_assert(sm87_a4w4_prefill_gemm_plan(3'847U, 12'288U, 5'120U)
                  .launch_ctas == kSm87A4W4PrefillPersistentCtas);
static_assert(sm87_a4w4_prefill_gemm_plan(1U, 1'024U, 5'120U)
                  .m_tiles == 1U);
static_assert(sm87_a4w4_prefill_gemm_plan(0U, 128U, 64U).launch_ctas == 0U);
static_assert(!sm87_a4w4_prefill_uses_large_m_candidate(512U));
static_assert(!sm87_a4w4_prefill_uses_large_m_candidate(1'023U));
static_assert(sm87_a4w4_prefill_uses_large_m_candidate(1'024U));
static_assert(sm87_a4w4_prefill_uses_large_m_candidate(4'096U));
static_assert(!sm87_a4w4_prefill_uses_large_m_candidate(4'097U));
static_assert(!sm87_a4w4_prefill_uses_m64n256_candidate(1'024U, 5'120U));
static_assert(sm87_a4w4_prefill_uses_m64n256_candidate(2'048U, 5'120U));
static_assert(!sm87_a4w4_prefill_uses_m64n256_candidate(2'048U, 5'184U));
static_assert(!sm87_a4w4_prefill_uses_m64n256_candidate(2'049U, 5'120U));
static_assert(sm87_a4w4_prefill_gemm_m64n256_plan(
                  2'048U, 17'408U, 5'120U)
                  .work_tiles == 2'176U);
static_assert(sm87_a4w4_prefill_gemm_m64n256_plan(
                  4'096U, 5'120U, 17'408U)
                  .work_tiles == 1'280U);

}  // namespace q3x::kernels
