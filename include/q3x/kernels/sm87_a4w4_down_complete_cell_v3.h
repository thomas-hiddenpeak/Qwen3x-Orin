#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Down-specific complete cell for the authenticated K128 consumer ABI.  Its
// runtime vertical slice is build-time optional and remains default-off.
//
// One 256-thread CTA owns M128N128.  Its eight warps own a 4x2 grid of M32N64
// output cells, keeping 64 FP32 accumulators/thread for the complete K loop.
// Within each warp a B fragment and B scale feed both resident M16 partials;
// A and B scales are decoded once at their natural reuse scope.
// Two A stages and three B stages form an asymmetric K128 ring: the cold B
// plane receives two groups of lookahead while the fixed-M A plane receives
// one.  The resulting 42,240-byte cell preserves two-CTA/SM residency.
inline constexpr std::size_t kSm87A4W4DownCellV3TileM = 128U;
inline constexpr std::size_t kSm87A4W4DownCellV3TileN = 128U;
inline constexpr std::size_t kSm87A4W4DownCellV3TileK = 128U;
inline constexpr std::size_t kSm87A4W4DownCellV3Threads = 256U;
inline constexpr std::size_t kSm87A4W4DownCellV3Warps = 8U;
inline constexpr std::size_t kSm87A4W4DownCellV3WarpTileM = 32U;
inline constexpr std::size_t kSm87A4W4DownCellV3WarpTileN = 64U;
inline constexpr std::size_t kSm87A4W4DownCellV3WarpRows = 4U;
inline constexpr std::size_t kSm87A4W4DownCellV3WarpColumns = 2U;
inline constexpr std::size_t kSm87A4W4DownCellV3AStages = 2U;
inline constexpr std::size_t kSm87A4W4DownCellV3BStages = 3U;
inline constexpr std::size_t kSm87A4W4DownCellV3PersistentCtas = 32U;
inline constexpr std::size_t kSm87A4W4DownCellV3CtasPerSm = 2U;
inline constexpr std::size_t kSm87A4W4DownCellV3MaximumRegisters = 128U;
inline constexpr std::size_t kSm87A4W4DownCellV3StageBytes = 8'448U;
inline constexpr std::size_t kSm87A4W4DownCellV3SharedBytes = 42'240U;

struct Sm87A4W4DownCellV3Plan final {
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

[[nodiscard]] constexpr Sm87A4W4DownCellV3Plan
sm87_a4w4_down_complete_cell_v3_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4DownCellV3TileM != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4DownCellV3TileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4DownCellV3TileK != 0U) {
    return {};
  }
  const std::size_t m_tiles = token_count / kSm87A4W4DownCellV3TileM;
  const std::size_t n_tiles = output_size / kSm87A4W4DownCellV3TileN;
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  if (m_tiles == 0U || n_tiles == 0U || m_tiles > maximum / n_tiles) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4DownCellV3TileK,
          input_size / 64U,
          work_tiles,
          work_tiles < kSm87A4W4DownCellV3PersistentCtas
              ? work_tiles
              : kSm87A4W4DownCellV3PersistentCtas};
}

struct Sm87A4W4DownCellV3Resources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int query_sm87_a4w4_down_complete_cell_v3_resources_cuda(
    Sm87A4W4DownCellV3Resources* resources) noexcept;

// Both operands use the existing consumer-block packed-code layout.  Scale
// capacities use sm87_a4w4_consumer_k128_scale_capacity_elements().  This
// isolated cell has no tail policy and fails closed on non-tile-multiple M/N/K.
[[nodiscard]] int launch_sm87_a4w4_down_complete_cell_v3_bf16_cuda(
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

static_assert(kSm87A4W4DownCellV3StageBytes *
                      (kSm87A4W4DownCellV3AStages +
                       kSm87A4W4DownCellV3BStages) ==
                  kSm87A4W4DownCellV3SharedBytes);
static_assert(kSm87A4W4DownCellV3SharedBytes *
                  kSm87A4W4DownCellV3CtasPerSm <=
              96U * 1'024U);
static_assert(kSm87A4W4DownCellV3WarpTileM *
                      kSm87A4W4DownCellV3WarpRows ==
                  kSm87A4W4DownCellV3TileM &&
              kSm87A4W4DownCellV3WarpTileN *
                      kSm87A4W4DownCellV3WarpColumns ==
                  kSm87A4W4DownCellV3TileN &&
              kSm87A4W4DownCellV3WarpRows *
                      kSm87A4W4DownCellV3WarpColumns ==
                  kSm87A4W4DownCellV3Warps);
static_assert(sm87_a4w4_down_complete_cell_v3_plan(
                  2'048U, 5'120U, 17'408U)
                  .m_tiles == 16U);
static_assert(sm87_a4w4_down_complete_cell_v3_plan(
                  2'048U, 5'120U, 17'408U)
                  .n_tiles == 40U);
static_assert(sm87_a4w4_down_complete_cell_v3_plan(
                  2'048U, 5'120U, 17'408U)
                  .k128_groups == 136U);
static_assert(sm87_a4w4_down_complete_cell_v3_plan(
                  2'048U, 5'120U, 17'408U)
                  .work_tiles == 640U);
static_assert(sm87_a4w4_down_complete_cell_v3_plan(
                  2'048U, 5'120U, 17'408U)
                  .launch_ctas == 32U);

}  // namespace q3x::kernels
