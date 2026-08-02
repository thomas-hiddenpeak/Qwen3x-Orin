#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_GATEUP_DOWN_EDGE_M128N64_HOST_DEVICE \
  __host__ __device__
#else
#define Q3X_SM87_A4W4_GATEUP_DOWN_EDGE_M128N64_HOST_DEVICE
#endif

namespace q3x::kernels {

// Structural successor to the retained producer-owned K512 edge.  A CTA
// owns a complete (M128,K512) product/quantization cell and visits the eight
// N64 Gate+Up strips that form it.  Sixteen warps form an 8x2 grid of paired
// M16N32 Gate/Up warp tiles.  This doubles B-weight reuse across M while
// retaining the exact signed-A4/K512-scale arithmetic and Down-input ABI.
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64TileM = 128U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64TileN = 64U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64ScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N64StripsPerScale = 8U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64PhysicalK64 =
    64U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64Stages = 4U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64ScaleSlots = 2U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64Threads = 512U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64Warps = 16U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64WarpTileM = 16U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64WarpTileN = 32U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64WarpRows = 8U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64WarpColumns = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N64PackedK64Bytes = 32U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64OuterBlock = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N64LaunchAlignment = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N64PersistentCtas = 16U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64CtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N64MaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N64ModelIntermediate = 17'408U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N64ModelInput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N64MaximumLaunchTokens = 4'096U;

inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64AStageBytes =
    kSm87A4W4GateUpDownEdgeM128N64TileM *
    kSm87A4W4GateUpDownEdgeM128N64PhysicalK64 / 2U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64BStageBytes =
    kSm87A4W4GateUpDownEdgeM128N64TileN *
    kSm87A4W4GateUpDownEdgeM128N64PhysicalK64 / 2U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64StageBytes =
    kSm87A4W4GateUpDownEdgeM128N64AStageBytes +
    2U * kSm87A4W4GateUpDownEdgeM128N64BStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N64ScaleSlotBytes =
        (kSm87A4W4GateUpDownEdgeM128N64TileM +
         2U * kSm87A4W4GateUpDownEdgeM128N64TileN) *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N64PipelineBytes =
        kSm87A4W4GateUpDownEdgeM128N64Stages *
            kSm87A4W4GateUpDownEdgeM128N64StageBytes +
        kSm87A4W4GateUpDownEdgeM128N64ScaleSlots *
            kSm87A4W4GateUpDownEdgeM128N64ScaleSlotBytes;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeM128N64PlaneBytes =
    kSm87A4W4GateUpDownEdgeM128N64TileM *
    kSm87A4W4GateUpDownEdgeM128N64ScaleK * sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes =
        kSm87A4W4GateUpDownEdgeM128N64PipelineBytes +
        kSm87A4W4GateUpDownEdgeM128N64PlaneBytes;

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_down_edge_m128n64_launch_token_count(
    const std::size_t logical_token_count) noexcept {
  const std::size_t tiles = sm87_a4w4_gateup_down_edge_ceil_div(
      logical_token_count,
      kSm87A4W4GateUpDownEdgeM128N64LaunchAlignment);
  return tiles != 0U &&
                 sm87_a4w4_gateup_down_edge_product_fits(
                     tiles,
                     kSm87A4W4GateUpDownEdgeM128N64LaunchAlignment)
             ? tiles * kSm87A4W4GateUpDownEdgeM128N64LaunchAlignment
             : 0U;
}

struct Sm87A4W4GateUpDownEdgeM128N64Plan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t edge_groups{};
  std::size_t strips_per_edge{};
  std::size_t input_k512_groups{};
  std::size_t input_physical_k64_groups{};
  std::size_t output_physical_k64_groups{};
  std::size_t work_edge_cells{};
  std::size_t launch_ctas{};
  std::size_t maximum_iterations{};
};

struct Sm87A4W4GateUpDownEdgeM128N64WorkCell final {
  std::size_t m_tile{};
  std::size_t edge_group{};
  bool valid{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgeM128N64Plan
sm87_a4w4_gateup_down_edge_m128n64_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t maximum_launch_ctas =
        kSm87A4W4GateUpDownEdgeM128N64PersistentCtas) noexcept {
  if (logical_token_count == 0U ||
      launch_token_count !=
          sm87_a4w4_gateup_down_edge_m128n64_launch_token_count(
              logical_token_count) ||
      launch_token_count % kSm87A4W4GateUpDownEdgeM128N64TileM != 0U ||
      intermediate_size == 0U ||
      intermediate_size % kSm87A4W4GateUpDownEdgeM128N64ScaleK != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpDownEdgeM128N64ScaleK != 0U ||
      maximum_launch_ctas == 0U ||
      maximum_launch_ctas >
          kSm87A4W4GateUpDownEdgeM128N64PersistentCtas) {
    return {};
  }
  const std::size_t m_tiles =
      launch_token_count / kSm87A4W4GateUpDownEdgeM128N64TileM;
  const std::size_t edge_groups =
      intermediate_size / kSm87A4W4GateUpDownEdgeM128N64ScaleK;
  if (!sm87_a4w4_gateup_down_edge_product_fits(m_tiles, edge_groups)) {
    return {};
  }
  const std::size_t work_edge_cells = m_tiles * edge_groups;
  const std::size_t launch_ctas =
      work_edge_cells < maximum_launch_ctas ? work_edge_cells
                                            : maximum_launch_ctas;
  return {logical_token_count,
          launch_token_count,
          intermediate_size,
          input_size,
          m_tiles,
          edge_groups,
          kSm87A4W4GateUpDownEdgeM128N64StripsPerScale,
          input_size / kSm87A4W4GateUpDownEdgeM128N64ScaleK,
          input_size / kSm87A4W4GateUpDownEdgeM128N64PhysicalK64,
          intermediate_size / kSm87A4W4GateUpDownEdgeM128N64PhysicalK64,
          work_edge_cells,
          launch_ctas,
          sm87_a4w4_gateup_down_edge_ceil_div(work_edge_cells,
                                               launch_ctas)};
}

[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgeM128N64Plan
sm87_a4w4_gateup_down_edge_m128n64_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return launch_token_count <=
                     kSm87A4W4GateUpDownEdgeM128N64MaximumLaunchTokens &&
                 intermediate_size ==
                     kSm87A4W4GateUpDownEdgeM128N64ModelIntermediate &&
                 input_size == kSm87A4W4GateUpDownEdgeM128N64ModelInput
             ? sm87_a4w4_gateup_down_edge_m128n64_test_plan(
                   logical_token_count, launch_token_count,
                   intermediate_size, input_size)
             : Sm87A4W4GateUpDownEdgeM128N64Plan{};
}

// Work is N-major.  When M has sixteen M128 tiles (the P2048 authority),
// every resident CTA retains one M owner while all CTAs advance through the
// same N512 edge group, maximizing L2 weight reuse without imbalance.
[[nodiscard]]
    Q3X_SM87_A4W4_GATEUP_DOWN_EDGE_M128N64_HOST_DEVICE constexpr
    Sm87A4W4GateUpDownEdgeM128N64WorkCell
sm87_a4w4_gateup_down_edge_m128n64_work_cell(
    const Sm87A4W4GateUpDownEdgeM128N64Plan& plan,
    const std::size_t cta, const std::size_t iteration) noexcept {
  constexpr std::size_t maximum = ~std::size_t{0};
  if (plan.launch_ctas == 0U || cta >= plan.launch_ctas ||
      iteration > (maximum - cta) / plan.launch_ctas) {
    return {};
  }
  const std::size_t ordinal = cta + iteration * plan.launch_ctas;
  if (ordinal >= plan.work_edge_cells) {
    return {};
  }
  const std::size_t edge_group = ordinal / plan.m_tiles;
  return {ordinal - edge_group * plan.m_tiles, edge_group, true};
}

struct Sm87A4W4GateUpDownEdgeM128N64Resources final {
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
query_sm87_a4w4_gateup_down_k512_edge_m128n64_resources_cuda(
    Sm87A4W4GateUpDownEdgeM128N64Resources* resources) noexcept;

[[nodiscard]] int launch_sm87_a4w4_gateup_down_k512_edge_m128n64_cuda(
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
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    float output_clip_ratio,
    std::uint8_t* packed_output,
    std::size_t packed_output_capacity_bytes,
    std::uint16_t* output_k512_scales_bf16,
    std::size_t output_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k512_edge_m128n64_test_cuda(
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
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    float output_clip_ratio,
    std::uint8_t* packed_output,
    std::size_t packed_output_capacity_bytes,
    std::uint16_t* output_k512_scales_bf16,
    std::size_t output_scale_capacity_elements,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4GateUpDownEdgeM128N64AStageBytes == 4'096U);
static_assert(kSm87A4W4GateUpDownEdgeM128N64BStageBytes == 2'048U);
static_assert(kSm87A4W4GateUpDownEdgeM128N64StageBytes == 8'192U);
static_assert(kSm87A4W4GateUpDownEdgeM128N64ScaleSlotBytes == 512U);
static_assert(kSm87A4W4GateUpDownEdgeM128N64PipelineBytes == 33'792U);
static_assert(kSm87A4W4GateUpDownEdgeM128N64PlaneBytes == 131'072U);
static_assert(kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes ==
              164'864U);
static_assert(kSm87A4W4GateUpDownEdgeM128N64WarpRows *
                      kSm87A4W4GateUpDownEdgeM128N64WarpColumns ==
                  kSm87A4W4GateUpDownEdgeM128N64Warps);
static_assert(kSm87A4W4GateUpDownEdgeM128N64WarpTileM *
                      kSm87A4W4GateUpDownEdgeM128N64WarpRows ==
                  kSm87A4W4GateUpDownEdgeM128N64TileM);
static_assert(kSm87A4W4GateUpDownEdgeM128N64WarpTileN *
                      kSm87A4W4GateUpDownEdgeM128N64WarpColumns ==
                  kSm87A4W4GateUpDownEdgeM128N64TileN);
static_assert(kSm87A4W4GateUpDownEdgeM128N64TileN *
                      kSm87A4W4GateUpDownEdgeM128N64StripsPerScale ==
                  kSm87A4W4GateUpDownEdgeM128N64ScaleK);
static_assert(sm87_a4w4_gateup_down_edge_m128n64_plan(
                  2'048U, 2'048U, 17'408U, 5'120U)
                  .work_edge_cells == 544U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_GATEUP_DOWN_EDGE_M128N64_HOST_DEVICE
