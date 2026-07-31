#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Prefill-only Down projection experiment for the pinned Qwen3.6 shape.
//
// The physical packed-code ABI remains two adjacent K64 consumer blocks per
// shared K128 scale.  One CTA owns M128N256 and keeps one N256 stripe while it
// walks every M128 tile in ascending order.  Sixteen persistent CTAs are a
// shape-specific choice for Down (N=5120 => twenty N stripes), not a selector
// shared with Gate/Up.
inline constexpr std::size_t kSm87A4W4DownK128StageMajorTileM = 128U;
inline constexpr std::size_t kSm87A4W4DownK128StageMajorTileN = 256U;
inline constexpr std::size_t kSm87A4W4DownK128StageMajorTileK = 128U;
inline constexpr std::size_t kSm87A4W4DownK128StageMajorThreads = 256U;
inline constexpr std::size_t kSm87A4W4DownK128StageMajorWarps = 8U;
inline constexpr std::size_t kSm87A4W4DownK128StageMajorPipelineSlots = 2U;
inline constexpr std::size_t kSm87A4W4DownK128StageMajorPersistentCtas = 16U;
inline constexpr std::size_t kSm87A4W4DownK128StageMajorMaximumRegisters =
    255U;
inline constexpr std::size_t kSm87A4W4DownK128StageMajorSharedBytes =
    46'464U;

struct Sm87A4W4DownK128StageMajorPlan final {
  std::size_t token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_stripes{};
  std::size_t k128_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr Sm87A4W4DownK128StageMajorPlan
sm87_a4w4_down_k128_stage_major_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4DownK128StageMajorTileM != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4DownK128StageMajorTileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4DownK128StageMajorTileK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4DownK128StageMajorTileM;
  const std::size_t n_stripes =
      output_size / kSm87A4W4DownK128StageMajorTileN;
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  if (m_tiles == 0U || n_stripes == 0U ||
      m_tiles > maximum / n_stripes) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_stripes;
  const std::size_t launch_ctas =
      n_stripes < kSm87A4W4DownK128StageMajorPersistentCtas
          ? n_stripes
          : kSm87A4W4DownK128StageMajorPersistentCtas;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_stripes,
          input_size / kSm87A4W4DownK128StageMajorTileK,
          input_size / 64U,
          work_tiles,
          launch_ctas};
}

struct Sm87A4W4DownK128StageMajorResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int query_sm87_a4w4_down_k128_stage_major_resources_cuda(
    Sm87A4W4DownK128StageMajorResources* resources) noexcept;

// Both operands use the existing consumer-block packed-code layout.  Scale
// capacities use sm87_a4w4_consumer_k128_scale_capacity_elements().  M, N,
// and K are deliberately exact tile multiples: this isolated candidate owns
// no tail policy and cannot silently reinterpret a K64 sidecar.
[[nodiscard]] int launch_sm87_a4w4_down_k128_stage_major_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k128_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k128_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    void* cuda_stream = nullptr) noexcept;

static_assert(sm87_a4w4_down_k128_stage_major_plan(128U, 256U, 256U)
                  .work_tiles == 1U);
static_assert(sm87_a4w4_down_k128_stage_major_plan(
                  2'048U, 5'120U, 17'408U)
                  .m_tiles == 16U);
static_assert(sm87_a4w4_down_k128_stage_major_plan(
                  2'048U, 5'120U, 17'408U)
                  .n_stripes == 20U);
static_assert(sm87_a4w4_down_k128_stage_major_plan(
                  2'048U, 5'120U, 17'408U)
                  .k128_groups == 136U);
static_assert(sm87_a4w4_down_k128_stage_major_plan(
                  2'048U, 5'120U, 17'408U)
                  .launch_ctas == 16U);
static_assert(sm87_a4w4_down_k128_stage_major_plan(64U, 256U, 256U)
                  .launch_ctas == 0U);

}  // namespace q3x::kernels
