#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Experimental full-tile A4W4 Prefill GEMM.  This is the first complete
// consumer of the native SM87 m16n8k64 primitive; it deliberately owns only
// the integer projection boundary and does not select itself in the runner.
//
// A and B use the canonical signed-nibble ABI shared with
// sm87_a4w4_prefill_primitive.h:
//
//   A: [M, K/2] bytes, row-major, dynamically quantized per [row, K64]
//   B: [N, K/2] bytes, row-major, calibrated offline per [row, K64]
//   A scales: BF16 [M, K/64]
//   B scales: BF16 [N, K/64]
//   D: BF16 [M, N]
//
// Every K64 integer partial is dequantized before it is accumulated in FP32;
// scales from distinct groups are never incorrectly factored across K.
inline constexpr std::size_t kSm87A4W4PrefillTileM = 64U;
inline constexpr std::size_t kSm87A4W4PrefillTileN = 64U;
inline constexpr std::size_t kSm87A4W4PrefillTileK = 64U;
inline constexpr std::size_t kSm87A4W4PrefillThreads = 256U;
inline constexpr std::size_t kSm87A4W4PrefillWarps = 8U;
inline constexpr std::size_t kSm87A4W4PrefillPipelineStages = 3U;
inline constexpr std::size_t kSm87A4W4PrefillPersistentCtas = 32U;
inline constexpr std::size_t kSm87A4W4PrefillCtasPerSm = 2U;

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
// [row,K64] groups per CTA.  clip_ratio is a calibration-controlled value in
// (0,1]; the packed output uses [-7,7] symmetric codes and one BF16 scale per
// group.  Target fused producers may bypass this launch while preserving the
// exact same ABI.
[[nodiscard]] int launch_sm87_a4_quantize_bf16_cuda(
    const std::uint16_t* input_bf16,
    std::size_t input_row_stride_elements,
    std::size_t token_count,
    std::size_t input_size,
    float clip_ratio,
    std::uint8_t* packed_a,
    std::size_t packed_a_row_stride_bytes,
    std::uint16_t* a_k64_scales_bf16,
    std::size_t scale_row_stride_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int query_sm87_a4w4_prefill_gemm_resources_cuda(
    Sm87A4W4PrefillGemmResources* resources) noexcept;

// Launches one persistent projection.  The current admission accepts arbitrary
// positive M, all fixed Qwen3.6 projection N values (multiples of 64), and K
// values divisible by 64.  Tail M rows are zero-filled in the async staging
// path and never written outside D.
[[nodiscard]] int launch_sm87_a4w4_prefill_gemm_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_row_stride_bytes,
    const std::uint16_t* a_k64_scales_bf16,
    const std::uint8_t* packed_b,
    std::size_t packed_b_row_stride_bytes,
    const std::uint16_t* b_k64_scales_bf16,
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
static_assert(sm87_a4w4_prefill_gemm_plan(0U, 64U, 64U).launch_ctas == 0U);

}  // namespace q3x::kernels
