#pragma once

#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_DOWN_K512_FRAGMENT_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_DOWN_K512_FRAGMENT_HOST_DEVICE
#endif

namespace q3x::kernels {

// Standalone Down-projection dataflow prototype. No runtime selects this API.
//
// One 256-thread CTA owns M64N128. Its eight warps own disjoint M64N16
// stripes. A is the authenticated canonical outer64 K512 payload and enters
// a three-slot K256 cp.async ring. B uses an offline, byte-preserving consumer
// layout:
//
//   [N128 panel][K512 group][K64][N16 warp][lane][16-byte vector].
//
// A lane vector is exactly the two m16n8k64 B fragments used by that warp:
// {N[0:8].x0, N[0:8].x1, N[8:16].x0, N[8:16].x1}. The kernel therefore
// issues one aligned 128-bit global load per lane and K64, does no B shared
// store or LDS, and reuses those fragments across all four M16 rows.
inline constexpr std::size_t kSm87A4W4DownK512FragmentMinimumTokens = 128U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentMaximumTokens = 4'096U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentOutputSize = 5'120U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentInputSize = 17'408U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentTileM = 64U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentTileN = 128U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentWarpTileM = 64U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentWarpTileN = 16U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentStageK = 256U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentScaleK = 512U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentPhysicalK64 = 64U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentK64PerStage = 4U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentStagesPerScale = 2U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentStages = 3U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentThreads = 256U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentWarps = 8U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentM16PerWarp = 4U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentN8PerWarp = 2U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentLaneVectorBytes = 16U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentPackedRowK64Bytes =
    32U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentStageABytes =
    kSm87A4W4DownK512FragmentTileM *
    kSm87A4W4DownK512FragmentStageK / 2U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentScaleSlotBytes =
    (kSm87A4W4DownK512FragmentTileM +
     kSm87A4W4DownK512FragmentTileN) *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87A4W4DownK512FragmentDynamicSharedBytes =
    kSm87A4W4DownK512FragmentStages *
        kSm87A4W4DownK512FragmentStageABytes +
    kSm87A4W4DownK512FragmentScaleSlotBytes;
inline constexpr std::size_t kSm87A4W4DownK512FragmentBWaveCtas = 32U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentMaximumLaunchCtas =
    (kSm87A4W4DownK512FragmentMaximumTokens /
     kSm87A4W4DownK512FragmentTileM) *
    (kSm87A4W4DownK512FragmentOutputSize /
     kSm87A4W4DownK512FragmentTileN);
inline constexpr std::size_t kSm87A4W4DownK512FragmentCtasPerSm = 2U;
inline constexpr std::size_t kSm87A4W4DownK512FragmentMaximumRegisters =
    128U;

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_k512_fragment_b_capacity_bytes(
    const std::size_t output_size, const std::size_t input_size) noexcept {
  if (output_size == 0U || input_size == 0U ||
      output_size % kSm87A4W4DownK512FragmentTileN != 0U ||
      input_size % kSm87A4W4DownK512FragmentScaleK != 0U ||
      !sm87_a4w4_down_k512_product_fits(output_size, input_size / 2U)) {
    return 0U;
  }
  return output_size * (input_size / 2U);
}

// Offset of one aligned lane vector. Every valid tuple is 16-byte aligned.
[[nodiscard]] Q3X_SM87_A4W4_DOWN_K512_FRAGMENT_HOST_DEVICE constexpr
    std::size_t
sm87_a4w4_down_k512_fragment_b_vector_offset(
    const std::size_t n128_panel, const std::size_t k512_group,
    const std::size_t k64_in_group, const std::size_t n16_warp,
    const std::size_t lane, const std::size_t k512_group_count) noexcept {
  return (((((n128_panel * k512_group_count + k512_group) *
                  (kSm87A4W4DownK512FragmentScaleK /
                   kSm87A4W4DownK512FragmentPhysicalK64) +
              k64_in_group) *
                 kSm87A4W4DownK512FragmentWarps +
             n16_warp) *
                32U +
            lane) *
           kSm87A4W4DownK512FragmentLaneVectorBytes);
}

struct Sm87A4W4DownK512FragmentBWordCoordinate final {
  std::size_t n{};
  std::size_t byte_in_k64{};
  bool valid{};
};

// Logical source coordinate for one u32 word in a lane's 16-byte vector.
// word 0/1 is the first N8 fragment; word 2/3 is the adjacent N8 fragment.
[[nodiscard]] constexpr Sm87A4W4DownK512FragmentBWordCoordinate
sm87_a4w4_down_k512_fragment_b_word_coordinate(
    const std::size_t n16_warp, const std::size_t lane,
    const std::size_t word) noexcept {
  if (n16_warp >= kSm87A4W4DownK512FragmentWarps || lane >= 32U ||
      word >= 4U) {
    return {};
  }
  return {n16_warp * kSm87A4W4DownK512FragmentWarpTileN +
              (word / 2U) * 8U + lane / 4U,
          (word % 2U) * 16U + 4U * (lane % 4U), true};
}

struct Sm87A4W4DownK512FragmentPlan final {
  std::size_t token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k512_groups{};
  std::size_t physical_k256_stages{};
  std::size_t physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
  std::size_t owner_groups{};
};

struct Sm87A4W4DownK512FragmentTile final {
  std::size_t m_tile{};
  std::size_t n_tile{};
  bool valid{};
};

[[nodiscard]] constexpr Sm87A4W4DownK512FragmentPlan
sm87_a4w4_down_k512_fragment_native_test_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4DownK512FragmentTileM != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4DownK512FragmentTileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4DownK512FragmentScaleK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4DownK512FragmentTileM;
  const std::size_t n_tiles =
      output_size / kSm87A4W4DownK512FragmentTileN;
  if (!sm87_a4w4_down_k512_product_fits(m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  const std::size_t launch_ctas = work_tiles;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4DownK512FragmentScaleK,
          input_size / kSm87A4W4DownK512FragmentStageK,
          input_size / kSm87A4W4DownK512FragmentPhysicalK64,
          work_tiles,
          launch_ctas,
          (m_tiles + kSm87A4W4DownK512FragmentBWaveCtas - 1U) /
              kSm87A4W4DownK512FragmentBWaveCtas};
}

[[nodiscard]] constexpr Sm87A4W4DownK512FragmentPlan
sm87_a4w4_down_k512_fragment_native_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  // The real runner supplies ceil128(M), even though the cell itself is M64.
  return token_count >= kSm87A4W4DownK512FragmentMinimumTokens &&
                 token_count <= kSm87A4W4DownK512FragmentMaximumTokens &&
                 token_count % 128U == 0U &&
                 output_size == kSm87A4W4DownK512FragmentOutputSize &&
                 input_size == kSm87A4W4DownK512FragmentInputSize
             ? sm87_a4w4_down_k512_fragment_native_test_plan(
                   token_count, output_size, input_size)
             : Sm87A4W4DownK512FragmentPlan{};
}

[[nodiscard]] constexpr Sm87A4W4DownK512FragmentTile
sm87_a4w4_down_k512_fragment_native_tile(
    const Sm87A4W4DownK512FragmentPlan& plan, const std::size_t cta,
    const std::size_t owner_group, const std::size_t n_wave) noexcept {
  if (plan.launch_ctas == 0U ||
      cta >= kSm87A4W4DownK512FragmentBWaveCtas ||
      owner_group >= plan.owner_groups || n_wave >= plan.n_tiles) {
    return {};
  }
  const std::size_t m_tile =
      owner_group * kSm87A4W4DownK512FragmentBWaveCtas + cta;
  return m_tile < plan.m_tiles
             ? Sm87A4W4DownK512FragmentTile{m_tile, n_wave, true}
             : Sm87A4W4DownK512FragmentTile{};
}

struct Sm87A4W4DownK512FragmentResources final {
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
query_sm87_a4w4_down_k512_fragment_native_resources_cuda(
    Sm87A4W4DownK512FragmentResources* resources) noexcept;

[[nodiscard]] int launch_sm87_a4w4_down_k512_fragment_native_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* fragment_native_b,
    std::size_t fragment_native_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Correctness-only generic complete-cell launcher. Synthetic input through
// this API is never a performance result.
[[nodiscard]] int
launch_sm87_a4w4_down_k512_fragment_native_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* fragment_native_b,
    std::size_t fragment_native_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4DownK512FragmentStageABytes == 8U * 1'024U);
static_assert(kSm87A4W4DownK512FragmentScaleSlotBytes == 384U);
static_assert(kSm87A4W4DownK512FragmentDynamicSharedBytes == 24'960U);
static_assert(kSm87A4W4DownK512FragmentWarpTileM ==
              4U * 16U);
static_assert(kSm87A4W4DownK512FragmentWarpTileN ==
              2U * 8U);
static_assert(kSm87A4W4DownK512FragmentTileN ==
              kSm87A4W4DownK512FragmentWarps *
                  kSm87A4W4DownK512FragmentWarpTileN);
static_assert(kSm87A4W4DownK512FragmentWarps * 32U *
                      kSm87A4W4DownK512FragmentLaneVectorBytes ==
                  kSm87A4W4DownK512FragmentTileN *
                      kSm87A4W4DownK512FragmentPhysicalK64 / 2U);
static_assert(sm87_a4w4_down_k512_fragment_b_capacity_bytes(
                  5'120U, 17'408U) ==
              sm87_a4w4_down_k512_packed_capacity_bytes(
                  5'120U, 17'408U));

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_DOWN_K512_FRAGMENT_HOST_DEVICE
