#pragma once

#include "q3x/kernels/sm87_nvfp4_marlin.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off, fixed-shape surface for reconstructing the frozen stock-vLLM
// BF16 x NVFP4 Marlin host schedule at exact M40000.  It deliberately does
// not widen any historical M<=8192 API or change its selector semantics.
enum class Sm87NvFp4MarlinP40ParityRole : std::uint8_t {
  kGateUp = 0U,
  kDown = 1U,
};

enum class Sm87NvFp4MarlinP40ParitySegmentKind : std::uint8_t {
  // Frozen stock host policy: one M1024 launch, using the unmodified
  // LegacyStripe scheduler and original Marlin epilogue.  Every output tile
  // has one full-K owner at this M.
  kLegacyStripeFullKM1024 = 0U,
  // Frozen stock tail policy: the same LegacyStripe kernel runs at M64.  Its
  // part-2 stripe has role-dependent cross-CTA FP32 reductions through C_tmp
  // and ordered locks; this is not interchangeable with a full-K raster in
  // finite precision.
  kLegacyStripeSplitKM64 = 1U,
};

inline constexpr std::size_t kSm87NvFp4MarlinP40ParityTokens = 40'000U;
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityLegacySegmentTokens = 1'024U;
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityLegacySegmentCount = 39U;
inline constexpr std::size_t kSm87NvFp4MarlinP40ParityTailTokens = 64U;
inline constexpr std::size_t kSm87NvFp4MarlinP40ParitySegmentCount = 40U;
inline constexpr std::size_t kSm87NvFp4MarlinP40ParityPersistentCtas = 16U;
inline constexpr std::size_t kSm87NvFp4MarlinP40ParityLockCount =
    kSm87NvFp4MarlinP40ParityPersistentCtas;
inline constexpr std::size_t kSm87NvFp4MarlinP40ParityLockBytes =
    kSm87NvFp4MarlinP40ParityLockCount * sizeof(std::int32_t);
inline constexpr std::size_t kSm87NvFp4MarlinP40ParityReductionElements =
    kSm87NvFp4MarlinP40ParityPersistentCtas *
    kSm87NvFp4MarlinThreadM * kSm87NvFp4MarlinThreadN;
inline constexpr std::size_t kSm87NvFp4MarlinP40ParityReductionBytes =
    kSm87NvFp4MarlinP40ParityReductionElements * sizeof(float);
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityGateTailPart2Tiles = 8U;
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityGateTailSplitOutputTiles = 8U;
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityDownTailPart2Tiles = 20U;
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityDownTailSplitOutputTiles = 12U;
inline constexpr std::size_t kSm87NvFp4MarlinP40ParitySplitWays = 2U;
// Stock Marlin publishes one contiguous row-major C[M,N]. Canonical GateThenUp
// packing assigns columns [0,17408) to Gate and [17408,34816) to Up inside
// every token row; it does not publish two contiguous [M,17408] matrices.
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityGateUpRowFeatures =
        kSm87NvFp4MarlinGateUpOutput;
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityGateColumnOffset = 0U;
inline constexpr std::size_t kSm87NvFp4MarlinP40ParityUpColumnOffset =
    kSm87NvFp4MarlinIntermediate;
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityGateUpRowStrideElements =
        kSm87NvFp4MarlinP40ParityGateUpRowFeatures;
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityGateUpRowStrideBytes =
        kSm87NvFp4MarlinP40ParityGateUpRowStrideElements *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityUpColumnByteOffset =
        kSm87NvFp4MarlinP40ParityUpColumnOffset * sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityMergedGateUpElements =
        kSm87NvFp4MarlinP40ParityTokens *
        kSm87NvFp4MarlinP40ParityGateUpRowStrideElements;
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityMergedGateUpBytes =
        kSm87NvFp4MarlinP40ParityMergedGateUpElements *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityActivatedElements =
        kSm87NvFp4MarlinP40ParityTokens * kSm87NvFp4MarlinIntermediate;
inline constexpr std::size_t
    kSm87NvFp4MarlinP40ParityActivatedBytes =
        kSm87NvFp4MarlinP40ParityActivatedElements * sizeof(std::uint16_t);

static_assert(kSm87NvFp4MarlinP40ParityLegacySegmentCount *
                      kSm87NvFp4MarlinP40ParityLegacySegmentTokens +
                  kSm87NvFp4MarlinP40ParityTailTokens ==
              kSm87NvFp4MarlinP40ParityTokens);
static_assert(kSm87NvFp4MarlinP40ParityLegacySegmentTokens ==
              kSm87NvFp4MarlinMaximumKernelSegmentTokens);
static_assert(kSm87NvFp4MarlinP40ParityTailTokens ==
              kSm87NvFp4MarlinM64Tokens);
static_assert(kSm87NvFp4MarlinP40ParityLockBytes == 64U);
static_assert(kSm87NvFp4MarlinP40ParityReductionBytes == 1'048'576U);
static_assert(kSm87NvFp4MarlinP40ParityLockBytes ==
              kSm87NvFp4MarlinLockBytes);
static_assert(kSm87NvFp4MarlinP40ParityReductionBytes ==
              kSm87NvFp4MarlinReductionBytes);
static_assert(kSm87NvFp4MarlinP40ParityGateUpRowStrideBytes == 69'632U);
static_assert(kSm87NvFp4MarlinP40ParityUpColumnByteOffset == 34'816U);
static_assert(kSm87NvFp4MarlinP40ParityMergedGateUpBytes ==
              2'785'280'000U);
static_assert(kSm87NvFp4MarlinP40ParityActivatedBytes == 1'392'640'000U);

struct Sm87NvFp4MarlinP40ParityGateUpLayout {
  std::size_t token_count = kSm87NvFp4MarlinP40ParityTokens;
  std::size_t row_features =
      kSm87NvFp4MarlinP40ParityGateUpRowFeatures;
  std::size_t row_stride_elements =
      kSm87NvFp4MarlinP40ParityGateUpRowStrideElements;
  std::size_t gate_column_offset =
      kSm87NvFp4MarlinP40ParityGateColumnOffset;
  std::size_t up_column_offset =
      kSm87NvFp4MarlinP40ParityUpColumnOffset;
  std::size_t merged_bytes =
      kSm87NvFp4MarlinP40ParityMergedGateUpBytes;
  std::size_t activated_bytes =
      kSm87NvFp4MarlinP40ParityActivatedBytes;
  bool halves_are_per_token_columns = true;
  bool activated_is_independent = true;

  [[nodiscard]] constexpr std::size_t gate_linear_index(
      const std::size_t token, const std::size_t column) const noexcept {
    return token * row_stride_elements + gate_column_offset + column;
  }

  [[nodiscard]] constexpr std::size_t up_linear_index(
      const std::size_t token, const std::size_t column) const noexcept {
    return token * row_stride_elements + up_column_offset + column;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return token_count == kSm87NvFp4MarlinP40ParityTokens &&
           row_features == kSm87NvFp4MarlinGateUpOutput &&
           row_stride_elements == row_features &&
           gate_column_offset == 0U &&
           up_column_offset == kSm87NvFp4MarlinIntermediate &&
           up_column_offset + kSm87NvFp4MarlinIntermediate == row_features &&
           merged_bytes == token_count * row_stride_elements *
                               sizeof(std::uint16_t) &&
           activated_bytes == token_count * kSm87NvFp4MarlinIntermediate *
                                  sizeof(std::uint16_t) &&
           halves_are_per_token_columns && activated_is_independent;
  }
};

inline constexpr Sm87NvFp4MarlinP40ParityGateUpLayout
    kSm87NvFp4MarlinP40ParityGateUpLayout{};
static_assert(kSm87NvFp4MarlinP40ParityGateUpLayout.valid());

struct Sm87NvFp4MarlinP40ParitySegment {
  std::size_t index = 0U;
  std::size_t token_offset = 0U;
  std::size_t token_count = 0U;
  std::size_t m_tiles = 0U;
  std::size_t n_tiles = 0U;
  std::size_t k_tiles = 0U;
  std::size_t output_tile_count = 0U;
  std::size_t physical_tile_slice_count = 0U;
  std::size_t full_k_output_tile_count = 0U;
  std::size_t split_k_output_tile_count = 0U;
  std::size_t split_k_partial_slice_count = 0U;
  std::size_t persistent_ctas = 0U;
  // M1024 retains exactly 16 part-2 tiles, one full-K tile per CTA.  At M64,
  // stock LegacyStripe uses 8 GateUp or 20 Down part-2 tiles; 8 or 12 of those
  // logical output tiles respectively have two physical K-partial slices.
  std::size_t legacy_part2_tiles = 0U;
  Sm87NvFp4MarlinP40ParitySegmentKind kind =
      Sm87NvFp4MarlinP40ParitySegmentKind::kLegacyStripeFullKM1024;
  bool all_output_tiles_full_k_owned = false;
  bool split_k = true;
  bool uses_locks = true;
  bool uses_reduction_workspace = true;
  bool atomic_add = true;
  bool fp32_reduce = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    const bool bulk =
        kind ==
        Sm87NvFp4MarlinP40ParitySegmentKind::kLegacyStripeFullKM1024;
    const bool tail =
        kind ==
        Sm87NvFp4MarlinP40ParitySegmentKind::kLegacyStripeSplitKM64;
    const bool gate_shape =
        n_tiles == kSm87NvFp4MarlinGateUpOutput /
                       kSm87NvFp4MarlinThreadN &&
        k_tiles == kSm87NvFp4MarlinHidden / kSm87NvFp4MarlinThreadK;
    const bool down_shape =
        n_tiles ==
                      kSm87NvFp4MarlinHidden / kSm87NvFp4MarlinThreadN &&
        k_tiles == kSm87NvFp4MarlinIntermediate /
                       kSm87NvFp4MarlinThreadK;
    const std::size_t expected_tail_part2 =
        gate_shape ? kSm87NvFp4MarlinP40ParityGateTailPart2Tiles
                   : kSm87NvFp4MarlinP40ParityDownTailPart2Tiles;
    const std::size_t expected_tail_split =
        gate_shape ? kSm87NvFp4MarlinP40ParityGateTailSplitOutputTiles
                   : kSm87NvFp4MarlinP40ParityDownTailSplitOutputTiles;
    return index < kSm87NvFp4MarlinP40ParitySegmentCount &&
           token_offset + token_count <=
               kSm87NvFp4MarlinP40ParityTokens &&
           (gate_shape || down_shape) &&
           ((bulk &&
             index < kSm87NvFp4MarlinP40ParityLegacySegmentCount &&
             token_count ==
                 kSm87NvFp4MarlinP40ParityLegacySegmentTokens &&
             m_tiles == 16U && legacy_part2_tiles == 16U &&
             full_k_output_tile_count == output_tile_count &&
             split_k_output_tile_count == 0U &&
             split_k_partial_slice_count == 0U &&
             physical_tile_slice_count == output_tile_count &&
             all_output_tiles_full_k_owned && !split_k && !uses_locks &&
             !uses_reduction_workspace) ||
            (tail &&
             index == kSm87NvFp4MarlinP40ParityLegacySegmentCount &&
             token_count == kSm87NvFp4MarlinP40ParityTailTokens &&
             m_tiles == 1U && legacy_part2_tiles == expected_tail_part2 &&
             split_k_output_tile_count == expected_tail_split &&
             split_k_partial_slice_count ==
                 expected_tail_split *
                     kSm87NvFp4MarlinP40ParitySplitWays &&
             full_k_output_tile_count + split_k_output_tile_count ==
                 output_tile_count &&
             physical_tile_slice_count ==
                 full_k_output_tile_count + split_k_partial_slice_count &&
             !all_output_tiles_full_k_owned && split_k && uses_locks &&
             uses_reduction_workspace)) &&
           n_tiles != 0U && k_tiles != 0U &&
           output_tile_count == m_tiles * n_tiles &&
           persistent_ctas == kSm87NvFp4MarlinP40ParityPersistentCtas &&
           !atomic_add && fp32_reduce;
  }
};

struct Sm87NvFp4MarlinP40ParityPlan {
  Sm87NvFp4MarlinP40ParityRole role =
      Sm87NvFp4MarlinP40ParityRole::kGateUp;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t weight_output_features = 0U;
  std::size_t published_output_features = 0U;
  std::size_t n_tiles = 0U;
  std::size_t k_tiles = 0U;
  std::size_t segment_count = 0U;
  std::size_t legacy_full_k_m1024_launches = 0U;
  std::size_t legacy_split_k_m64_launches = 0U;
  std::size_t physical_projection_launches = 0U;
  std::size_t tail_part2_tiles = 0U;
  std::size_t tail_full_k_output_tiles = 0U;
  std::size_t tail_split_k_output_tiles = 0U;
  std::size_t tail_split_k_partial_slices = 0U;
  std::size_t required_reduction_workspace_bytes = 0U;
  std::size_t required_lock_bytes = 0U;
  std::size_t required_standalone_silu_launches = 0U;
  bool requires_canonical_gate_then_up = false;
  bool bf16_projection_publication = false;
  bool fused_silu = true;
  bool fused_residual = true;
  bool split_k = true;
  bool uses_locks = true;
  bool uses_reduction_workspace = true;
  bool requires_zero_initialized_locks = false;
  bool atomic_add = true;
  bool fp32_reduce = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    const bool gate = role == Sm87NvFp4MarlinP40ParityRole::kGateUp;
    const bool down = role == Sm87NvFp4MarlinP40ParityRole::kDown;
    return (gate || down) && token_count == kSm87NvFp4MarlinP40ParityTokens &&
           input_features ==
               (gate ? kSm87NvFp4MarlinHidden
                     : kSm87NvFp4MarlinIntermediate) &&
           weight_output_features ==
               (gate ? kSm87NvFp4MarlinGateUpOutput
                     : kSm87NvFp4MarlinHidden) &&
           published_output_features == weight_output_features &&
           n_tiles ==
               weight_output_features / kSm87NvFp4MarlinThreadN &&
           k_tiles == input_features / kSm87NvFp4MarlinThreadK &&
           segment_count == kSm87NvFp4MarlinP40ParitySegmentCount &&
           legacy_full_k_m1024_launches ==
               kSm87NvFp4MarlinP40ParityLegacySegmentCount &&
           legacy_split_k_m64_launches == 1U &&
           physical_projection_launches ==
               legacy_full_k_m1024_launches +
                   legacy_split_k_m64_launches &&
           tail_part2_tiles ==
               (gate ? kSm87NvFp4MarlinP40ParityGateTailPart2Tiles
                     : kSm87NvFp4MarlinP40ParityDownTailPart2Tiles) &&
           tail_split_k_output_tiles ==
               (gate
                    ? kSm87NvFp4MarlinP40ParityGateTailSplitOutputTiles
                    : kSm87NvFp4MarlinP40ParityDownTailSplitOutputTiles) &&
           tail_full_k_output_tiles + tail_split_k_output_tiles == n_tiles &&
           tail_split_k_partial_slices ==
               tail_split_k_output_tiles *
                   kSm87NvFp4MarlinP40ParitySplitWays &&
           required_reduction_workspace_bytes ==
               kSm87NvFp4MarlinP40ParityReductionBytes &&
           required_lock_bytes == kSm87NvFp4MarlinP40ParityLockBytes &&
           required_standalone_silu_launches == (gate ? 1U : 0U) &&
           requires_canonical_gate_then_up == gate &&
           bf16_projection_publication && !fused_silu && !fused_residual &&
           split_k && uses_locks && uses_reduction_workspace &&
           requires_zero_initialized_locks && !atomic_add && fp32_reduce;
  }
};

[[nodiscard]] constexpr Sm87NvFp4MarlinP40ParityPlan
sm87_nvfp4_marlin_p40_parity_plan(
    const Sm87NvFp4MarlinP40ParityRole role,
    const std::size_t token_count) noexcept {
  if (token_count != kSm87NvFp4MarlinP40ParityTokens) {
    return {};
  }
  const bool gate = role == Sm87NvFp4MarlinP40ParityRole::kGateUp;
  const bool down = role == Sm87NvFp4MarlinP40ParityRole::kDown;
  if (!gate && !down) {
    return {};
  }
  const std::size_t input_features =
      gate ? kSm87NvFp4MarlinHidden : kSm87NvFp4MarlinIntermediate;
  const std::size_t output_features =
      gate ? kSm87NvFp4MarlinGateUpOutput : kSm87NvFp4MarlinHidden;
  const std::size_t n_tiles =
      output_features / kSm87NvFp4MarlinThreadN;
  const std::size_t tail_split_tiles =
      gate ? kSm87NvFp4MarlinP40ParityGateTailSplitOutputTiles
           : kSm87NvFp4MarlinP40ParityDownTailSplitOutputTiles;
  Sm87NvFp4MarlinP40ParityPlan plan{};
  plan.role = role;
  plan.token_count = token_count;
  plan.input_features = input_features;
  plan.weight_output_features = output_features;
  plan.published_output_features = output_features;
  plan.n_tiles = n_tiles;
  plan.k_tiles = input_features / kSm87NvFp4MarlinThreadK;
  plan.segment_count = kSm87NvFp4MarlinP40ParitySegmentCount;
  plan.legacy_full_k_m1024_launches =
      kSm87NvFp4MarlinP40ParityLegacySegmentCount;
  plan.legacy_split_k_m64_launches = 1U;
  plan.physical_projection_launches = plan.segment_count;
  plan.tail_part2_tiles =
      gate ? kSm87NvFp4MarlinP40ParityGateTailPart2Tiles
           : kSm87NvFp4MarlinP40ParityDownTailPart2Tiles;
  plan.tail_full_k_output_tiles = n_tiles - tail_split_tiles;
  plan.tail_split_k_output_tiles = tail_split_tiles;
  plan.tail_split_k_partial_slices =
      tail_split_tiles * kSm87NvFp4MarlinP40ParitySplitWays;
  plan.required_reduction_workspace_bytes =
      kSm87NvFp4MarlinP40ParityReductionBytes;
  plan.required_lock_bytes = kSm87NvFp4MarlinP40ParityLockBytes;
  plan.required_standalone_silu_launches = gate ? 1U : 0U;
  plan.requires_canonical_gate_then_up = gate;
  plan.bf16_projection_publication = true;
  plan.fused_silu = false;
  plan.fused_residual = false;
  plan.split_k = true;
  plan.uses_locks = true;
  plan.uses_reduction_workspace = true;
  plan.requires_zero_initialized_locks = true;
  plan.atomic_add = false;
  plan.fp32_reduce = true;
  return plan;
}

[[nodiscard]] constexpr Sm87NvFp4MarlinP40ParitySegment
sm87_nvfp4_marlin_p40_parity_segment(
    const Sm87NvFp4MarlinP40ParityPlan& plan,
    const std::size_t index) noexcept {
  if (!plan.valid() || index >= plan.segment_count) {
    return {};
  }
  const bool tail =
      index == kSm87NvFp4MarlinP40ParityLegacySegmentCount;
  const std::size_t token_count =
      tail ? kSm87NvFp4MarlinP40ParityTailTokens
           : kSm87NvFp4MarlinP40ParityLegacySegmentTokens;
  const std::size_t m_tiles = token_count / kSm87NvFp4MarlinThreadM;
  const std::size_t output_tiles = m_tiles * plan.n_tiles;
  const std::size_t split_tiles =
      tail ? plan.tail_split_k_output_tiles : 0U;
  const std::size_t split_partials =
      split_tiles * kSm87NvFp4MarlinP40ParitySplitWays;
  Sm87NvFp4MarlinP40ParitySegment segment{};
  segment.index = index;
  segment.token_offset =
      index * kSm87NvFp4MarlinP40ParityLegacySegmentTokens;
  segment.token_count = token_count;
  segment.m_tiles = m_tiles;
  segment.n_tiles = plan.n_tiles;
  segment.k_tiles = plan.k_tiles;
  segment.output_tile_count = output_tiles;
  segment.full_k_output_tile_count = output_tiles - split_tiles;
  segment.split_k_output_tile_count = split_tiles;
  segment.split_k_partial_slice_count = split_partials;
  segment.physical_tile_slice_count =
      segment.full_k_output_tile_count + split_partials;
  segment.persistent_ctas = kSm87NvFp4MarlinP40ParityPersistentCtas;
  segment.legacy_part2_tiles =
      tail ? plan.tail_part2_tiles
           : kSm87NvFp4MarlinP40ParityPersistentCtas;
  segment.kind =
      tail ? Sm87NvFp4MarlinP40ParitySegmentKind::kLegacyStripeSplitKM64
           : Sm87NvFp4MarlinP40ParitySegmentKind::kLegacyStripeFullKM1024;
  segment.all_output_tiles_full_k_owned = !tail;
  segment.split_k = tail;
  segment.uses_locks = tail;
  segment.uses_reduction_workspace = tail;
  segment.atomic_add = false;
  segment.fp32_reduce = true;
  return segment;
}

struct Sm87NvFp4MarlinP40ParityLaunchCounters {
  std::size_t legacy_full_k_m1024_launches = 0U;
  std::size_t legacy_split_k_m64_launches = 0U;
  std::size_t physical_projection_launches = 0U;
  std::size_t tail_split_k_output_tiles = 0U;
  std::size_t tail_split_k_partial_slices = 0U;
  bool complete = false;

  [[nodiscard]] constexpr bool counts_match(
      const Sm87NvFp4MarlinP40ParityPlan& plan) const noexcept {
    return plan.valid() &&
           legacy_full_k_m1024_launches ==
               plan.legacy_full_k_m1024_launches &&
           legacy_split_k_m64_launches ==
               plan.legacy_split_k_m64_launches &&
           physical_projection_launches ==
               plan.physical_projection_launches &&
           tail_split_k_output_tiles == plan.tail_split_k_output_tiles &&
           tail_split_k_partial_slices ==
               plan.tail_split_k_partial_slices;
  }

  [[nodiscard]] constexpr bool matches(
      const Sm87NvFp4MarlinP40ParityPlan& plan) const noexcept {
    return complete && counts_match(plan);
  }
};

struct Sm87NvFp4MarlinP40ParityKernelResources {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

struct Sm87NvFp4MarlinP40ParityResources {
  Sm87NvFp4MarlinP40ParityRole role =
      Sm87NvFp4MarlinP40ParityRole::kGateUp;
  Sm87NvFp4MarlinP40ParityKernelResources legacy_stripe{};
  std::size_t reduction_workspace_bytes = 0U;
  std::size_t lock_bytes = 0U;
  std::size_t tail_split_k_output_tiles = 0U;
  std::size_t tail_split_k_partial_slices = 0U;
  bool bulk_and_tail_share_kernel = false;
  bool requires_zero_initialized_locks = false;
  bool atomic_add = true;
  bool fp32_reduce = false;
  bool supported = false;
};

// Dedicated canonical pack entry point.  Unlike the historical generic
// prepare API it exposes no interleave switch, so a P40 parity caller cannot
// accidentally bind the v10 Gate/Up artifact.
[[nodiscard]] int prepare_sm87_nvfp4_marlin_p40_parity_gate_up_cuda(
    const std::uint8_t* canonical_gate_weight,
    const std::uint8_t* canonical_up_weight,
    const std::uint8_t* canonical_gate_scales,
    const std::uint8_t* canonical_up_scales,
    const float* canonical_shared_weight_scale_2_device,
    float scale_factor, std::uint8_t* marlin_weight,
    std::uint8_t* marlin_scales, float* marlin_global_scale,
    void* transpose_scratch, std::size_t transpose_scratch_bytes,
    void* cuda_stream = nullptr) noexcept;

// GateUp publishes one merged row-major BF16 [40000,34816] matrix requiring
// exactly 2,785,280,000 bytes. Each token row is [Gate(17408), Up(17408)]; the
// Up half begins 34,816 bytes into each 69,632-byte row. It is not valid to
// expose Gate@base and Up@base+1,392,640,000 as contiguous matrix views.
// Stock M64 LegacyStripe requires exactly 1 MiB of FP32 C_tmp and 64 bytes of
// int32 locks.  All 16 locks must be zero before the first call on an ordered
// stream; a successful stock tail returns every lock it used to zero.  C_tmp
// need not be initialized but must be 16-byte aligned; locks must be int32
// aligned. The byte capacities are explicit and must equal the constants
// above; aliasing any input, weight, output, or each other is rejected. Atomic
// reduction is disabled and cross-CTA reduction is FP32.
[[nodiscard]] int launch_sm87_nvfp4_marlin_p40_parity_gate_up_cuda(
    const std::uint16_t* input, const std::uint8_t* canonical_marlin_weight,
    const std::uint8_t* canonical_marlin_scales,
    const float* marlin_global_scale,
    std::uint16_t* merged_gate_up_row_major_bf16,
    float* reduction_workspace, std::size_t reduction_workspace_bytes,
    std::int32_t* locks, std::size_t lock_bytes,
    Sm87NvFp4MarlinP40ParityLaunchCounters* counters,
    void* cuda_stream = nullptr) noexcept;

// This is a distinct launch and consumes the already-published BF16 Gate and
// Up columns. It writes an independent contiguous [40000,17408] Activated
// matrix requiring 1,392,640,000 bytes and preserves the projection rounding
// boundary.
[[nodiscard]] int launch_sm87_nvfp4_marlin_p40_parity_silu_cuda(
    const std::uint16_t* merged_gate_up_row_major_bf16,
    std::uint16_t* activated_bf16,
    void* cuda_stream = nullptr) noexcept;

// Publishes only the BF16 Down branch. Residual addition is deliberately not
// part of this kernel surface.
[[nodiscard]] int launch_sm87_nvfp4_marlin_p40_parity_down_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales, const float* marlin_global_scale,
    std::uint16_t* branch_bf16,
    float* reduction_workspace, std::size_t reduction_workspace_bytes,
    std::int32_t* locks, std::size_t lock_bytes,
    Sm87NvFp4MarlinP40ParityLaunchCounters* counters,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int query_sm87_nvfp4_marlin_p40_parity_resources_cuda(
    Sm87NvFp4MarlinP40ParityRole role,
    Sm87NvFp4MarlinP40ParityResources* resources) noexcept;

}  // namespace q3x::kernels
