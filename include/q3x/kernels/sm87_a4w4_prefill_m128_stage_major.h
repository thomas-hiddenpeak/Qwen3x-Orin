#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Isolated next-generation Prefill structural candidate.  One 256-thread
// CTA owns M128N256 and assigns one M16N256 output strip to each of its eight
// warps.  A logical K128 group remains two physical K64 consumer-code planes;
// the two native MMAs accumulate into one S32 fragment before the single
// shared K128 scale product is applied.
//
// Shared memory is stage-major: slot[stage] owns both physical A/B code
// planes and the one A/B scale plane for that K128 group.  Two 25,344-byte
// slots form a 50,688-byte ping-pong pipeline.  This exceeds the legacy
// 48-KiB static allocation ceiling intentionally and is admitted through the
// SM87 opt-in dynamic-shared-memory contract, one CTA per SM.
inline constexpr std::size_t kSm87A4W4M128StageMajorTileM = 128U;
inline constexpr std::size_t kSm87A4W4M128StageMajorTileN = 256U;
inline constexpr std::size_t kSm87A4W4M128StageMajorTileK = 128U;
inline constexpr std::size_t kSm87A4W4M128StageMajorThreads = 256U;
inline constexpr std::size_t kSm87A4W4M128StageMajorWarps = 8U;
inline constexpr std::size_t kSm87A4W4M128StageMajorPipelineSlots = 2U;
inline constexpr std::size_t kSm87A4W4M128StageMajorPersistentCtas = 16U;
inline constexpr std::size_t kSm87A4W4M128StageMajorCtasPerSm = 1U;
inline constexpr std::size_t kSm87A4W4M128StageMajorMaximumRegisters = 255U;
inline constexpr std::size_t kSm87A4W4M128StageMajorStageBytes = 25'344U;
inline constexpr std::size_t kSm87A4W4M128StageMajorSharedBytes = 50'688U;
inline constexpr std::size_t kSm87A4W4M128StageMajorPairedTileN = 128U;
inline constexpr std::size_t kSm87A4W4M128StageMajorPairedSharedBytes =
    65'536U;

struct Sm87A4W4M128StageMajorPlan final {
  std::size_t token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k128_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr Sm87A4W4M128StageMajorPlan
sm87_a4w4_m128_stage_major_plan(const std::size_t token_count,
                                const std::size_t output_size,
                                const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4M128StageMajorTileM != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4M128StageMajorTileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4M128StageMajorTileK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4M128StageMajorTileM;
  const std::size_t n_tiles =
      output_size / kSm87A4W4M128StageMajorTileN;
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  if (m_tiles == 0U || n_tiles == 0U || m_tiles > maximum / n_tiles) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      work_tiles < kSm87A4W4M128StageMajorPersistentCtas
          ? work_tiles
          : kSm87A4W4M128StageMajorPersistentCtas;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4M128StageMajorTileK,
          input_size / 64U,
          work_tiles,
          launch_ctas};
}

struct Sm87A4W4M128StageMajorResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

struct Sm87A4W4M128StageMajorPairedPlan final {
  std::size_t token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k128_groups{};
  std::size_t physical_k64_groups{};
  std::size_t output_physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr Sm87A4W4M128StageMajorPairedPlan
sm87_a4w4_m128_stage_major_paired_plan(
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4M128StageMajorTileM != 0U ||
      intermediate_size == 0U ||
      intermediate_size % kSm87A4W4M128StageMajorPairedTileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4M128StageMajorTileK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4M128StageMajorTileM;
  const std::size_t n_tiles =
      intermediate_size / kSm87A4W4M128StageMajorPairedTileN;
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  if (m_tiles == 0U || n_tiles == 0U || m_tiles > maximum / n_tiles) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      work_tiles < kSm87A4W4M128StageMajorPersistentCtas
          ? work_tiles
          : kSm87A4W4M128StageMajorPersistentCtas;
  return {token_count,
          intermediate_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4M128StageMajorTileK,
          input_size / 64U,
          intermediate_size / 64U,
          work_tiles,
          launch_ctas};
}

// Both surfaces reject every target except the 16-SM SM87 deployment.  The
// launcher accepts the existing consumer-block packed-code ABI and shared
// K128 BF16 scale ABI; no conversion or alternate weight representation is
// hidden behind this candidate.
[[nodiscard]] int query_sm87_a4w4_m128_stage_major_resources_cuda(
    Sm87A4W4M128StageMajorResources* resources) noexcept;

[[nodiscard]] int launch_sm87_a4w4_m128_stage_major_bf16_cuda(
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

// Paired Gate+Up structural surface.  Its K loop shares the M128 activation
// slot across two N128 weight planes, computes SiLU(Gate)*Up, then directly
// emits the established two-physical-K64/one-K128-scale activation ABI for
// the Down projection.  The dead pipeline allocation is reused by a
// 64-KiB M128N128 FP32 product tile during output quantization.
[[nodiscard]] int query_sm87_a4w4_m128_stage_major_paired_resources_cuda(
    Sm87A4W4M128StageMajorResources* resources) noexcept;

[[nodiscard]] int launch_sm87_a4w4_m128_stage_major_paired_cuda(
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

static_assert(kSm87A4W4M128StageMajorStageBytes *
                  kSm87A4W4M128StageMajorPipelineSlots ==
              kSm87A4W4M128StageMajorSharedBytes);
static_assert(sm87_a4w4_m128_stage_major_plan(128U, 256U, 128U)
                  .launch_ctas == 1U);
static_assert(sm87_a4w4_m128_stage_major_plan(2'048U, 256U, 5'120U)
                  .launch_ctas == 16U);
static_assert(sm87_a4w4_m128_stage_major_plan(127U, 256U, 128U)
                  .launch_ctas == 0U);
static_assert(sm87_a4w4_m128_stage_major_plan(128U, 128U, 128U)
                  .launch_ctas == 0U);
static_assert(sm87_a4w4_m128_stage_major_plan(128U, 256U, 64U)
                  .launch_ctas == 0U);
static_assert(sm87_a4w4_m128_stage_major_paired_plan(128U, 128U, 128U)
                  .launch_ctas == 1U);
static_assert(sm87_a4w4_m128_stage_major_paired_plan(
                  2'048U, 17'408U, 5'120U)
                  .launch_ctas == 16U);
static_assert(sm87_a4w4_m128_stage_major_paired_plan(64U, 128U, 128U)
                  .launch_ctas == 0U);

}  // namespace q3x::kernels
