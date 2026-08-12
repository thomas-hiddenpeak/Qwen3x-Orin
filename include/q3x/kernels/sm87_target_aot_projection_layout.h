#pragma once

#include "q3x/kernels/sm87_target_aot_projection_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Host-only packed-operand layout schema for AC-PREFILL-SM87-AOT-SYSTEM-v1.
// Sm87TargetAotProjectionPackedManifest is an in-memory sidecar schema, not
// an on-disk header and never safe to serialize with a struct byte copy. The
// 4096-byte header reservation and canonical little-endian encoding remain a
// future loader contract and must be frozen before an asset is authenticated.
// The
// layout is derived for the M128N256K64, eight-warp projection plan.  It is
// independent of prompt M: P40, P60, and P130 use the same engine-lifetime
// artifact.  This header exposes no CUDA launcher and grants no resource,
// numerical, or production qualification.
inline constexpr std::array<std::uint8_t, 8U>
    kSm87TargetAotProjectionPackedMagic{{'Q', '3', 'X', 'A',
                                         'O', 'T', 'P', '1'}};
inline constexpr std::uint16_t kSm87TargetAotProjectionPackedAbiMajor = 1U;
inline constexpr std::uint16_t kSm87TargetAotProjectionPackedAbiMinor = 0U;
inline constexpr std::uint32_t kSm87TargetAotProjectionPackedHeaderBytes =
    4'096U;
inline constexpr std::uint32_t kSm87TargetAotProjectionPackedAlignment =
    256U;
inline constexpr std::uint32_t kSm87TargetAotProjectionPackedVectorBytes =
    16U;
inline constexpr std::uint32_t kSm87TargetAotProjectionPackedNWarpCount =
    4U;
inline constexpr std::uint32_t kSm87TargetAotProjectionPackedMWarpCount =
    2U;
inline constexpr std::uint32_t kSm87TargetAotProjectionPackedNPerWarp = 64U;
inline constexpr std::uint32_t kSm87TargetAotProjectionPackedN8Panels = 8U;
inline constexpr std::uint32_t kSm87TargetAotProjectionPackedK16s = 4U;
inline constexpr std::uint32_t kSm87TargetAotProjectionPackedLanes = 32U;
inline constexpr std::size_t kSm87TargetAotProjectionPackedMaxPartitions =
    kSm87TargetAotProjectionMaximumPartitions;

enum class Sm87TargetAotProjectionPackedPlanIdentity : std::uint16_t {
  kInvalid = 0U,
  kNvFp4GateUpM128N256K64V1,
  kNvFp4DownM128N256K64V1,
  kFp8GdnQkvZM128N256K64V1,
  kFp8FullQkvM128N256K64V1,
  kFp8AttentionOutputM128N256K64V1,
};

enum class Sm87TargetAotProjectionPackedLayoutIdentity : std::uint16_t {
  kInvalid = 0U,
  // One cell owns N256 x K64.  Within the B region the order is
  // [K16][N-warp][N8-panel][lane][component].  NVFP4 stores four lane
  // components in two bytes: even component ordinals occupy the low nibble
  // and odd component ordinals occupy the high nibble. FP8 stores the four
  // components in four bytes. Both FP8 payload and block scales are E4M3FN;
  // the future numerical contract must explicitly handle terminal 0x7f/0xff
  // encodings before qualification. This layout schema only preserves their
  // bytes. NVFP4 scales
  // follow B in [K16][N-warp][N8-panel][row] order.
  kConsumerN64K16LaneComponentV1,
};

enum class Sm87TargetAotProjectionPackedRegion : std::uint8_t {
  kInvalid = 0U,
  kWeight,
  kBlockScale,
};

struct Sm87TargetAotProjectionPackedPartition {
  Sm87TargetAotLogicalRole logical_role =
      Sm87TargetAotLogicalRole::kInvalid;
  std::uint32_t partition_index = 0U;
  std::uint32_t global_n_offset = 0U;
  std::uint32_t output_features = 0U;
  std::uint32_t input_features = 0U;
  std::uint32_t n_tiles = 0U;
  std::uint32_t k_tiles = 0U;
  std::uint32_t weight_bits = 0U;
  std::uint32_t block_scale_group_k = 0U;
  std::uint32_t weight_bytes_per_cell = 0U;
  std::uint32_t block_scale_bytes_per_cell = 0U;
  std::uint32_t cell_bytes = 0U;
  std::uint64_t payload_offset = 0U;
  std::uint64_t payload_bytes = 0U;
  bool independent_source = false;
  bool independent_tensor_scale = false;
};

struct Sm87TargetAotProjectionPackedLayout {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  Sm87TargetAotProjectionPackedPlanIdentity plan_identity =
      Sm87TargetAotProjectionPackedPlanIdentity::kInvalid;
  Sm87TargetAotProjectionPackedLayoutIdentity layout_identity =
      Sm87TargetAotProjectionPackedLayoutIdentity::kInvalid;
  Sm87TargetAotProjectionEncoding encoding =
      Sm87TargetAotProjectionEncoding::kInvalid;
  std::uint32_t projected_output_features = 0U;
  std::uint32_t input_features = 0U;
  std::uint32_t partition_count = 0U;
  std::array<Sm87TargetAotProjectionPackedPartition,
             kSm87TargetAotProjectionPackedMaxPartitions>
      partitions{};
  std::uint64_t payload_bytes = 0U;
  std::uint32_t payload_alignment = 0U;
  std::uint32_t g2s_vector_bytes = 0U;
  std::uint32_t block_m = 0U;
  std::uint32_t block_n = 0U;
  std::uint32_t block_k = 0U;
  // The engine streams consecutive persisted K64 cells through this
  // three-entry shared-memory ring; pipeline stages never duplicate payload.
  std::uint32_t pipeline_stages = 0U;
  std::uint32_t cta_threads = 0U;
  std::uint32_t n_warps = 0U;
  std::uint32_t m_warps = 0U;
  bool token_count_independent = false;
  bool cuda_implementation_present = false;
  bool static_resources_qualified = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;

  [[nodiscard]] constexpr bool valid() const noexcept;
};

[[nodiscard]] constexpr bool sm87_target_aot_packed_add_fits(
    const std::uint64_t left, const std::uint64_t right) noexcept {
  return left <= std::numeric_limits<std::uint64_t>::max() - right;
}

[[nodiscard]] constexpr bool sm87_target_aot_packed_mul_fits(
    const std::uint64_t left, const std::uint64_t right) noexcept {
  return right == 0U ||
         left <= std::numeric_limits<std::uint64_t>::max() / right;
}

[[nodiscard]] constexpr bool sm87_target_aot_packed_aligned(
    const std::uint64_t value, const std::uint64_t alignment) noexcept {
  return alignment != 0U && value % alignment == 0U;
}

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedPlanIdentity
sm87_target_aot_projection_packed_plan_identity(
    const Sm87TargetAotProjectionRole role) noexcept {
  switch (role) {
    case Sm87TargetAotProjectionRole::kNvFp4GateUp:
      return Sm87TargetAotProjectionPackedPlanIdentity::
          kNvFp4GateUpM128N256K64V1;
    case Sm87TargetAotProjectionRole::kNvFp4Down:
      return Sm87TargetAotProjectionPackedPlanIdentity::
          kNvFp4DownM128N256K64V1;
    case Sm87TargetAotProjectionRole::kFp8GdnQkvZ:
      return Sm87TargetAotProjectionPackedPlanIdentity::
          kFp8GdnQkvZM128N256K64V1;
    case Sm87TargetAotProjectionRole::kFp8FullQkv:
      return Sm87TargetAotProjectionPackedPlanIdentity::
          kFp8FullQkvM128N256K64V1;
    case Sm87TargetAotProjectionRole::kFp8AttentionOutput:
      return Sm87TargetAotProjectionPackedPlanIdentity::
          kFp8AttentionOutputM128N256K64V1;
    case Sm87TargetAotProjectionRole::kInvalid:
    case Sm87TargetAotProjectionRole::kCount:
    default:
      return Sm87TargetAotProjectionPackedPlanIdentity::kInvalid;
  }
}

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedLayout
sm87_target_aot_projection_packed_layout(
    const Sm87TargetAotProjectionRole role) noexcept {
  // Projection dimensions and partition boundaries do not depend on M.  P40
  // is used only as a canonical way to obtain the already frozen role plan.
  const auto projection = sm87_target_aot_projection_plan(role, 40'000U);
  if (!projection.valid()) {
    return {};
  }

  Sm87TargetAotProjectionPackedLayout layout;
  layout.role = role;
  layout.plan_identity =
      sm87_target_aot_projection_packed_plan_identity(role);
  layout.layout_identity = Sm87TargetAotProjectionPackedLayoutIdentity::
      kConsumerN64K16LaneComponentV1;
  layout.encoding = projection.encoding;
  layout.projected_output_features =
      static_cast<std::uint32_t>(projection.projected_output_features);
  layout.input_features =
      static_cast<std::uint32_t>(projection.input_features);
  layout.partition_count =
      static_cast<std::uint32_t>(projection.partition_count);
  layout.payload_alignment = kSm87TargetAotProjectionPackedAlignment;
  layout.g2s_vector_bytes = kSm87TargetAotProjectionPackedVectorBytes;
  layout.block_m =
      static_cast<std::uint32_t>(kSm87TargetAotProjectionBlockM);
  layout.block_n =
      static_cast<std::uint32_t>(kSm87TargetAotProjectionBlockN);
  layout.block_k =
      static_cast<std::uint32_t>(kSm87TargetAotProjectionBlockK);
  layout.pipeline_stages =
      static_cast<std::uint32_t>(kSm87TargetAotProjectionPipelineStages);
  layout.cta_threads =
      static_cast<std::uint32_t>(kSm87TargetAotProjectionThreads);
  layout.n_warps = kSm87TargetAotProjectionPackedNWarpCount;
  layout.m_warps = kSm87TargetAotProjectionPackedMWarpCount;
  layout.token_count_independent = true;

  const bool nvfp4 = projection.encoding ==
      Sm87TargetAotProjectionEncoding::kNvFp4E2M1Block16E4M3FnScale;
  const std::uint32_t weight_bits = nvfp4 ? 4U : 8U;
  const std::uint32_t weight_bytes_per_cell =
      static_cast<std::uint32_t>(
          kSm87TargetAotProjectionBlockN *
          kSm87TargetAotProjectionBlockK * weight_bits / 8U);
  const std::uint32_t scale_bytes_per_cell =
      nvfp4 ? static_cast<std::uint32_t>(
                    kSm87TargetAotProjectionBlockN *
                    (kSm87TargetAotProjectionBlockK / 16U))
             : 0U;
  const std::uint32_t cell_bytes =
      weight_bytes_per_cell + scale_bytes_per_cell;

  std::uint64_t cursor = 0U;
  for (std::uint32_t index = 0U; index < layout.partition_count; ++index) {
    const auto& source = projection.partitions[index];
    auto& partition = layout.partitions[index];
    partition.logical_role = source.role;
    partition.partition_index = index;
    partition.global_n_offset =
        static_cast<std::uint32_t>(source.output_offset);
    partition.output_features =
        static_cast<std::uint32_t>(source.output_features);
    partition.input_features = layout.input_features;
    partition.n_tiles = partition.output_features /
                        static_cast<std::uint32_t>(
                            kSm87TargetAotProjectionBlockN);
    partition.k_tiles = partition.input_features /
                        static_cast<std::uint32_t>(
                            kSm87TargetAotProjectionBlockK);
    partition.weight_bits = weight_bits;
    partition.block_scale_group_k = nvfp4 ? 16U : 0U;
    partition.weight_bytes_per_cell = weight_bytes_per_cell;
    partition.block_scale_bytes_per_cell = scale_bytes_per_cell;
    partition.cell_bytes = cell_bytes;
    partition.payload_offset = cursor;
    const std::uint64_t cells =
        static_cast<std::uint64_t>(partition.n_tiles) * partition.k_tiles;
    partition.payload_bytes = cells * cell_bytes;
    partition.independent_source = true;
    partition.independent_tensor_scale = true;
    cursor += partition.payload_bytes;
  }
  layout.payload_bytes = cursor;
  return layout;
}

[[nodiscard]] constexpr bool sm87_target_aot_same_packed_partition(
    const Sm87TargetAotProjectionPackedPartition& left,
    const Sm87TargetAotProjectionPackedPartition& right) noexcept {
  return left.logical_role == right.logical_role &&
         left.partition_index == right.partition_index &&
         left.global_n_offset == right.global_n_offset &&
         left.output_features == right.output_features &&
         left.input_features == right.input_features &&
         left.n_tiles == right.n_tiles && left.k_tiles == right.k_tiles &&
         left.weight_bits == right.weight_bits &&
         left.block_scale_group_k == right.block_scale_group_k &&
         left.weight_bytes_per_cell == right.weight_bytes_per_cell &&
         left.block_scale_bytes_per_cell ==
             right.block_scale_bytes_per_cell &&
         left.cell_bytes == right.cell_bytes &&
         left.payload_offset == right.payload_offset &&
         left.payload_bytes == right.payload_bytes &&
         left.independent_source == right.independent_source &&
         left.independent_tensor_scale == right.independent_tensor_scale;
}

[[nodiscard]] constexpr bool sm87_target_aot_same_packed_layout(
    const Sm87TargetAotProjectionPackedLayout& left,
    const Sm87TargetAotProjectionPackedLayout& right) noexcept {
  if (left.role != right.role ||
      left.plan_identity != right.plan_identity ||
      left.layout_identity != right.layout_identity ||
      left.encoding != right.encoding ||
      left.projected_output_features != right.projected_output_features ||
      left.input_features != right.input_features ||
      left.partition_count != right.partition_count ||
      left.payload_bytes != right.payload_bytes ||
      left.payload_alignment != right.payload_alignment ||
      left.g2s_vector_bytes != right.g2s_vector_bytes ||
      left.block_m != right.block_m || left.block_n != right.block_n ||
      left.block_k != right.block_k ||
      left.pipeline_stages != right.pipeline_stages ||
      left.cta_threads != right.cta_threads ||
      left.n_warps != right.n_warps ||
      left.m_warps != right.m_warps ||
      left.token_count_independent != right.token_count_independent ||
      left.cuda_implementation_present !=
          right.cuda_implementation_present ||
      left.static_resources_qualified != right.static_resources_qualified ||
      left.numerical_contract_qualified !=
          right.numerical_contract_qualified ||
      left.production_dispatch_eligible !=
          right.production_dispatch_eligible) {
    return false;
  }
  for (std::size_t index = 0U; index < left.partitions.size(); ++index) {
    if (!sm87_target_aot_same_packed_partition(left.partitions[index],
                                               right.partitions[index])) {
      return false;
    }
  }
  return true;
}

constexpr bool Sm87TargetAotProjectionPackedLayout::valid() const noexcept {
  if (role == Sm87TargetAotProjectionRole::kInvalid ||
      role == Sm87TargetAotProjectionRole::kCount) {
    return false;
  }
  return sm87_target_aot_same_packed_layout(
      *this, sm87_target_aot_projection_packed_layout(role));
}

struct Sm87TargetAotProjectionPackedCell {
  std::uint32_t partition_index = 0U;
  std::uint32_t n_tile = 0U;
  std::uint32_t k_tile = 0U;
  std::uint64_t payload_offset = 0U;
  std::uint64_t weight_offset = 0U;
  std::uint32_t weight_bytes = 0U;
  std::uint64_t block_scale_offset = 0U;
  std::uint32_t block_scale_bytes = 0U;
  std::uint32_t cell_bytes = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedCell
sm87_target_aot_projection_packed_cell(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::size_t partition_index, const std::size_t n_tile,
    const std::size_t k_tile) noexcept {
  if (!layout.valid() || partition_index >= layout.partition_count) {
    return {};
  }
  const auto& partition = layout.partitions[partition_index];
  if (n_tile >= partition.n_tiles || k_tile >= partition.k_tiles) {
    return {};
  }
  const std::uint64_t linear_cell =
      static_cast<std::uint64_t>(n_tile) * partition.k_tiles + k_tile;
  if (!sm87_target_aot_packed_mul_fits(linear_cell,
                                       partition.cell_bytes)) {
    return {};
  }
  const std::uint64_t relative = linear_cell * partition.cell_bytes;
  if (!sm87_target_aot_packed_add_fits(partition.payload_offset,
                                       relative)) {
    return {};
  }
  const std::uint64_t cell_offset = partition.payload_offset + relative;
  if (!sm87_target_aot_packed_add_fits(
          cell_offset, partition.weight_bytes_per_cell)) {
    return {};
  }
  return {static_cast<std::uint32_t>(partition_index),
          static_cast<std::uint32_t>(n_tile),
          static_cast<std::uint32_t>(k_tile),
          cell_offset,
          cell_offset,
          partition.weight_bytes_per_cell,
          cell_offset + partition.weight_bytes_per_cell,
          partition.block_scale_bytes_per_cell,
          partition.cell_bytes,
          true};
}

struct Sm87TargetAotProjectionPackedFragment {
  std::uint32_t partition_index = 0U;
  std::uint32_t n_tile = 0U;
  std::uint32_t k_tile = 0U;
  std::uint8_t k16 = 0U;
  std::uint8_t n_warp = 0U;
  std::uint8_t n8_panel = 0U;
  std::uint8_t consumer_warp0 = 0U;
  std::uint8_t consumer_warp1 = 0U;
  std::uint64_t weight_offset = 0U;
  std::uint32_t weight_bytes = 0U;
  std::uint64_t block_scale_offset = 0U;
  std::uint32_t block_scale_bytes = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedFragment
sm87_target_aot_projection_packed_fragment(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::size_t partition_index, const std::size_t n_tile,
    const std::size_t k_tile, const std::size_t k16,
    const std::size_t n_warp, const std::size_t n8_panel) noexcept {
  const auto cell = sm87_target_aot_projection_packed_cell(
      layout, partition_index, n_tile, k_tile);
  if (!cell.valid || k16 >= kSm87TargetAotProjectionPackedK16s ||
      n_warp >= kSm87TargetAotProjectionPackedNWarpCount ||
      n8_panel >= kSm87TargetAotProjectionPackedN8Panels) {
    return {};
  }
  const auto& partition = layout.partitions[partition_index];
  const std::uint32_t weight_fragment_bytes =
      partition.weight_bits == 4U ? 64U : 128U;
  const std::uint64_t fragment =
      (static_cast<std::uint64_t>(k16) *
           kSm87TargetAotProjectionPackedNWarpCount +
       n_warp) *
          kSm87TargetAotProjectionPackedN8Panels +
      n8_panel;
  const std::uint64_t weight_offset =
      cell.weight_offset + fragment * weight_fragment_bytes;
  const std::uint64_t scale_offset =
      cell.block_scale_offset + fragment * 8U;
  return {static_cast<std::uint32_t>(partition_index),
          static_cast<std::uint32_t>(n_tile),
          static_cast<std::uint32_t>(k_tile),
          static_cast<std::uint8_t>(k16),
          static_cast<std::uint8_t>(n_warp),
          static_cast<std::uint8_t>(n8_panel),
          static_cast<std::uint8_t>(n_warp),
          static_cast<std::uint8_t>(
              n_warp + kSm87TargetAotProjectionPackedNWarpCount),
          weight_offset,
          weight_fragment_bytes,
          partition.block_scale_bytes_per_cell == 0U ? 0U : scale_offset,
          partition.block_scale_bytes_per_cell == 0U ? 0U : 8U,
          true};
}

[[nodiscard]] constexpr std::uint8_t
sm87_target_aot_projection_packed_component_for_k16(
    const std::uint32_t k_in_16) noexcept {
  if (k_in_16 < 8U) {
    return static_cast<std::uint8_t>((k_in_16 & 1U) == 0U ? 0U : 2U);
  }
  return static_cast<std::uint8_t>((k_in_16 & 1U) == 0U ? 1U : 3U);
}

[[nodiscard]] constexpr std::uint8_t
sm87_target_aot_projection_packed_lane_in_group_for_k16(
    const std::uint32_t k_in_16) noexcept {
  return static_cast<std::uint8_t>((k_in_16 % 8U) / 2U);
}

[[nodiscard]] constexpr std::uint8_t
sm87_target_aot_projection_packed_k16_for_lane_component(
    const std::uint32_t lane_in_group,
    const std::uint32_t component) noexcept {
  return static_cast<std::uint8_t>(
      2U * lane_in_group + (component >= 2U ? 1U : 0U) +
      ((component & 1U) != 0U ? 8U : 0U));
}

struct Sm87TargetAotProjectionPackedWeightAddress {
  std::uint32_t partition_index = 0U;
  std::uint32_t n = 0U;
  std::uint32_t k = 0U;
  std::uint64_t byte_offset = 0U;
  std::uint8_t nibble = 0U;
  std::uint8_t s2r_lane = 0U;
  std::uint8_t s2r_component = 0U;
  std::uint8_t n_warp = 0U;
  std::uint8_t consumer_warp0 = 0U;
  std::uint8_t consumer_warp1 = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedWeightAddress
sm87_target_aot_projection_packed_weight_address(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::size_t partition_index, const std::size_t n,
    const std::size_t k) noexcept {
  if (!layout.valid() || partition_index >= layout.partition_count) {
    return {};
  }
  const auto& partition = layout.partitions[partition_index];
  if (n >= partition.output_features || k >= partition.input_features) {
    return {};
  }
  const std::uint32_t n_in_tile = static_cast<std::uint32_t>(
      n % kSm87TargetAotProjectionBlockN);
  const std::uint32_t k_in_tile = static_cast<std::uint32_t>(
      k % kSm87TargetAotProjectionBlockK);
  const std::uint32_t n_warp =
      n_in_tile / kSm87TargetAotProjectionPackedNPerWarp;
  const std::uint32_t n_in_warp =
      n_in_tile % kSm87TargetAotProjectionPackedNPerWarp;
  const std::uint32_t n8_panel = n_in_warp / 8U;
  const std::uint32_t row = n_in_warp % 8U;
  const std::uint32_t k16 = k_in_tile / 16U;
  const std::uint32_t k_in_16 = k_in_tile % 16U;
  const std::uint32_t lane_in_group =
      sm87_target_aot_projection_packed_lane_in_group_for_k16(k_in_16);
  const std::uint32_t component =
      sm87_target_aot_projection_packed_component_for_k16(k_in_16);
  const std::uint32_t lane = row * 4U + lane_in_group;
  const auto fragment = sm87_target_aot_projection_packed_fragment(
      layout, partition_index, n / kSm87TargetAotProjectionBlockN,
      k / kSm87TargetAotProjectionBlockK, k16, n_warp, n8_panel);
  if (!fragment.valid) {
    return {};
  }

  std::uint64_t byte_offset = fragment.weight_offset;
  std::uint8_t nibble = 0U;
  if (partition.weight_bits == 4U) {
    byte_offset += lane * 2U + component / 2U;
    nibble = static_cast<std::uint8_t>(component & 1U);
  } else {
    byte_offset += lane * 4U + component;
  }
  return {static_cast<std::uint32_t>(partition_index),
          static_cast<std::uint32_t>(n),
          static_cast<std::uint32_t>(k),
          byte_offset,
          nibble,
          static_cast<std::uint8_t>(lane),
          static_cast<std::uint8_t>(component),
          static_cast<std::uint8_t>(n_warp),
          fragment.consumer_warp0,
          fragment.consumer_warp1,
          true};
}

struct Sm87TargetAotProjectionPackedScaleAddress {
  std::uint32_t partition_index = 0U;
  std::uint32_t n = 0U;
  std::uint32_t scale_group = 0U;
  std::uint64_t byte_offset = 0U;
  std::uint8_t n_warp = 0U;
  std::uint8_t consumer_warp0 = 0U;
  std::uint8_t consumer_warp1 = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedScaleAddress
sm87_target_aot_projection_packed_scale_address(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::size_t partition_index, const std::size_t n,
    const std::size_t scale_group) noexcept {
  if (!layout.valid() || partition_index >= layout.partition_count) {
    return {};
  }
  const auto& partition = layout.partitions[partition_index];
  if (partition.block_scale_group_k != 16U ||
      n >= partition.output_features ||
      scale_group >= partition.input_features / 16U) {
    return {};
  }
  const std::uint32_t n_in_tile = static_cast<std::uint32_t>(
      n % kSm87TargetAotProjectionBlockN);
  const std::uint32_t n_warp =
      n_in_tile / kSm87TargetAotProjectionPackedNPerWarp;
  const std::uint32_t n_in_warp =
      n_in_tile % kSm87TargetAotProjectionPackedNPerWarp;
  const std::uint32_t n8_panel = n_in_warp / 8U;
  const std::uint32_t row = n_in_warp % 8U;
  const std::uint32_t k_tile = static_cast<std::uint32_t>(scale_group / 4U);
  const std::uint32_t k16 = static_cast<std::uint32_t>(scale_group % 4U);
  const auto fragment = sm87_target_aot_projection_packed_fragment(
      layout, partition_index, n / kSm87TargetAotProjectionBlockN,
      k_tile, k16, n_warp, n8_panel);
  if (!fragment.valid || fragment.block_scale_bytes != 8U) {
    return {};
  }
  return {static_cast<std::uint32_t>(partition_index),
          static_cast<std::uint32_t>(n),
          static_cast<std::uint32_t>(scale_group),
          fragment.block_scale_offset + row,
          static_cast<std::uint8_t>(n_warp),
          fragment.consumer_warp0,
          fragment.consumer_warp1,
          true};
}

[[nodiscard]] constexpr std::size_t
sm87_target_aot_projection_packed_partition_for_offset(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::uint64_t byte_offset) noexcept {
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    const auto& partition = layout.partitions[index];
    if (byte_offset >= partition.payload_offset &&
        byte_offset - partition.payload_offset < partition.payload_bytes) {
      return index;
    }
  }
  return kSm87TargetAotProjectionPackedMaxPartitions;
}

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedWeightAddress
sm87_target_aot_projection_packed_reverse_weight(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::uint64_t byte_offset, const std::uint8_t nibble) noexcept {
  if (!layout.valid()) {
    return {};
  }
  const std::size_t partition_index =
      sm87_target_aot_projection_packed_partition_for_offset(layout,
                                                             byte_offset);
  if (partition_index >= layout.partition_count) {
    return {};
  }
  const auto& partition = layout.partitions[partition_index];
  const std::uint64_t relative = byte_offset - partition.payload_offset;
  const std::uint64_t cell_index = relative / partition.cell_bytes;
  const std::uint32_t byte_in_cell =
      static_cast<std::uint32_t>(relative % partition.cell_bytes);
  if (byte_in_cell >= partition.weight_bytes_per_cell ||
      (partition.weight_bits == 8U && nibble != 0U) || nibble > 1U) {
    return {};
  }
  const std::uint32_t n_tile =
      static_cast<std::uint32_t>(cell_index / partition.k_tiles);
  const std::uint32_t k_tile =
      static_cast<std::uint32_t>(cell_index % partition.k_tiles);
  const std::uint32_t fragment_bytes =
      partition.weight_bits == 4U ? 64U : 128U;
  const std::uint32_t fragment_index = byte_in_cell / fragment_bytes;
  const std::uint32_t byte_in_fragment = byte_in_cell % fragment_bytes;
  const std::uint32_t n8_panel =
      fragment_index % kSm87TargetAotProjectionPackedN8Panels;
  const std::uint32_t fragment_outer =
      fragment_index / kSm87TargetAotProjectionPackedN8Panels;
  const std::uint32_t n_warp =
      fragment_outer % kSm87TargetAotProjectionPackedNWarpCount;
  const std::uint32_t k16 =
      fragment_outer / kSm87TargetAotProjectionPackedNWarpCount;
  std::uint32_t lane = 0U;
  std::uint32_t component = 0U;
  if (partition.weight_bits == 4U) {
    lane = byte_in_fragment / 2U;
    component = (byte_in_fragment % 2U) * 2U + nibble;
  } else {
    lane = byte_in_fragment / 4U;
    component = byte_in_fragment % 4U;
  }
  if (lane >= kSm87TargetAotProjectionPackedLanes || component >= 4U ||
      k16 >= kSm87TargetAotProjectionPackedK16s) {
    return {};
  }
  const std::uint32_t row = lane / 4U;
  const std::uint32_t lane_in_group = lane % 4U;
  const std::uint32_t k_in_16 =
      sm87_target_aot_projection_packed_k16_for_lane_component(
          lane_in_group, component);
  const std::uint32_t n =
      n_tile * static_cast<std::uint32_t>(kSm87TargetAotProjectionBlockN) +
      n_warp * kSm87TargetAotProjectionPackedNPerWarp + n8_panel * 8U + row;
  const std::uint32_t k =
      k_tile * static_cast<std::uint32_t>(kSm87TargetAotProjectionBlockK) +
      k16 * 16U + k_in_16;
  const auto result = sm87_target_aot_projection_packed_weight_address(
      layout, partition_index, n, k);
  return result.valid && result.byte_offset == byte_offset &&
                 result.nibble == nibble
             ? result
             : Sm87TargetAotProjectionPackedWeightAddress{};
}

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedScaleAddress
sm87_target_aot_projection_packed_reverse_scale(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::uint64_t byte_offset) noexcept {
  if (!layout.valid()) {
    return {};
  }
  const std::size_t partition_index =
      sm87_target_aot_projection_packed_partition_for_offset(layout,
                                                             byte_offset);
  if (partition_index >= layout.partition_count) {
    return {};
  }
  const auto& partition = layout.partitions[partition_index];
  if (partition.block_scale_group_k != 16U) {
    return {};
  }
  const std::uint64_t relative = byte_offset - partition.payload_offset;
  const std::uint64_t cell_index = relative / partition.cell_bytes;
  const std::uint32_t byte_in_cell =
      static_cast<std::uint32_t>(relative % partition.cell_bytes);
  if (byte_in_cell < partition.weight_bytes_per_cell ||
      byte_in_cell >= partition.cell_bytes) {
    return {};
  }
  const std::uint32_t scale_index =
      byte_in_cell - partition.weight_bytes_per_cell;
  const std::uint32_t row = scale_index % 8U;
  const std::uint32_t scale_fragment = scale_index / 8U;
  const std::uint32_t n8_panel =
      scale_fragment % kSm87TargetAotProjectionPackedN8Panels;
  const std::uint32_t fragment_outer =
      scale_fragment / kSm87TargetAotProjectionPackedN8Panels;
  const std::uint32_t n_warp =
      fragment_outer % kSm87TargetAotProjectionPackedNWarpCount;
  const std::uint32_t k16 =
      fragment_outer / kSm87TargetAotProjectionPackedNWarpCount;
  if (k16 >= kSm87TargetAotProjectionPackedK16s) {
    return {};
  }
  const std::uint32_t n_tile =
      static_cast<std::uint32_t>(cell_index / partition.k_tiles);
  const std::uint32_t k_tile =
      static_cast<std::uint32_t>(cell_index % partition.k_tiles);
  const std::uint32_t n =
      n_tile * static_cast<std::uint32_t>(kSm87TargetAotProjectionBlockN) +
      n_warp * kSm87TargetAotProjectionPackedNPerWarp + n8_panel * 8U + row;
  const std::uint32_t scale_group = k_tile * 4U + k16;
  const auto result = sm87_target_aot_projection_packed_scale_address(
      layout, partition_index, n, scale_group);
  return result.valid && result.byte_offset == byte_offset
             ? result
             : Sm87TargetAotProjectionPackedScaleAddress{};
}

struct Sm87TargetAotProjectionPackedG2SVector {
  Sm87TargetAotProjectionPackedRegion region =
      Sm87TargetAotProjectionPackedRegion::kInvalid;
  std::uint32_t partition_index = 0U;
  std::uint32_t n_tile = 0U;
  std::uint32_t k_tile = 0U;
  std::uint32_t vector_index = 0U;
  std::uint16_t loader_thread = 0U;
  std::uint16_t loader_pass = 0U;
  std::uint64_t global_byte_offset = 0U;
  std::uint32_t shared_byte_offset = 0U;
  std::uint32_t bytes = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedG2SVector
sm87_target_aot_projection_packed_g2s_vector(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::size_t partition_index, const std::size_t n_tile,
    const std::size_t k_tile,
    const Sm87TargetAotProjectionPackedRegion region,
    const std::size_t vector_index) noexcept {
  const auto cell = sm87_target_aot_projection_packed_cell(
      layout, partition_index, n_tile, k_tile);
  if (!cell.valid ||
      (region != Sm87TargetAotProjectionPackedRegion::kWeight &&
       region != Sm87TargetAotProjectionPackedRegion::kBlockScale)) {
    return {};
  }
  const std::uint64_t region_offset =
      region == Sm87TargetAotProjectionPackedRegion::kWeight
          ? cell.weight_offset
          : cell.block_scale_offset;
  const std::uint32_t region_bytes =
      region == Sm87TargetAotProjectionPackedRegion::kWeight
          ? cell.weight_bytes
          : cell.block_scale_bytes;
  if (region_bytes == 0U ||
      vector_index >= region_bytes / kSm87TargetAotProjectionPackedVectorBytes) {
    return {};
  }
  const std::uint64_t vector_byte =
      vector_index * kSm87TargetAotProjectionPackedVectorBytes;
  return {region,
          static_cast<std::uint32_t>(partition_index),
          static_cast<std::uint32_t>(n_tile),
          static_cast<std::uint32_t>(k_tile),
          static_cast<std::uint32_t>(vector_index),
          static_cast<std::uint16_t>(vector_index % layout.cta_threads),
          static_cast<std::uint16_t>(vector_index / layout.cta_threads),
          region_offset + vector_byte,
          static_cast<std::uint32_t>(vector_byte),
          kSm87TargetAotProjectionPackedVectorBytes,
          true};
}

struct Sm87TargetAotProjectionPackedPayloadView {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  Sm87TargetAotProjectionPackedPlanIdentity plan_identity =
      Sm87TargetAotProjectionPackedPlanIdentity::kInvalid;
  Sm87TargetAotProjectionPackedLayoutIdentity layout_identity =
      Sm87TargetAotProjectionPackedLayoutIdentity::kInvalid;
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  std::uint64_t bytes = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedPayloadView
sm87_target_aot_projection_bind_packed_payload(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::uintptr_t address, const std::uint64_t bytes) noexcept {
  if (!layout.valid() || address == 0U || bytes != layout.payload_bytes ||
      !sm87_target_aot_packed_aligned(address, layout.payload_alignment) ||
      bytes > std::numeric_limits<std::uintptr_t>::max() ||
      address > std::numeric_limits<std::uintptr_t>::max() -
                    static_cast<std::uintptr_t>(bytes)) {
    return {};
  }
  return {layout.role,
          layout.plan_identity,
          layout.layout_identity,
          address,
          address + static_cast<std::uintptr_t>(bytes),
          bytes,
          true};
}

[[nodiscard]] constexpr bool sm87_target_aot_projection_payload_aliases(
    const Sm87TargetAotProjectionPackedPayloadView& left,
    const Sm87TargetAotProjectionPackedPayloadView& right) noexcept {
  return left.valid && right.valid && left.begin < right.end &&
         right.begin < left.end;
}

[[nodiscard]] constexpr bool
sm87_target_aot_projection_payload_pair_admissible(
    const Sm87TargetAotProjectionPackedPayloadView& left,
    const Sm87TargetAotProjectionPackedPayloadView& right) noexcept {
  return left.valid && right.valid &&
         !sm87_target_aot_projection_payload_aliases(left, right);
}

struct Sm87TargetAotProjectionSha256Digest {
  std::array<std::uint8_t, 32U> bytes{};

  [[nodiscard]] friend constexpr bool operator==(
      const Sm87TargetAotProjectionSha256Digest& left,
      const Sm87TargetAotProjectionSha256Digest& right) noexcept {
    for (std::size_t index = 0U; index < left.bytes.size(); ++index) {
      if (left.bytes[index] != right.bytes[index]) {
        return false;
      }
    }
    return true;
  }
  [[nodiscard]] friend constexpr bool operator!=(
      const Sm87TargetAotProjectionSha256Digest& left,
      const Sm87TargetAotProjectionSha256Digest& right) noexcept {
    return !(left == right);
  }
};

[[nodiscard]] constexpr bool sm87_target_aot_projection_digest_is_zero(
    const Sm87TargetAotProjectionSha256Digest& digest) noexcept {
  for (const std::uint8_t byte : digest.bytes) {
    if (byte != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool sm87_target_aot_projection_scale_bits_valid(
    const std::uint32_t bits) noexcept {
  return (bits & 0x8000'0000U) == 0U &&
         (bits & 0x7fff'ffffU) != 0U &&
         (bits & 0x7f80'0000U) != 0x7f80'0000U;
}

struct Sm87TargetAotProjectionPackedSourceBinding {
  Sm87TargetAotLogicalRole logical_role =
      Sm87TargetAotLogicalRole::kInvalid;
  std::uint32_t partition_index = 0U;
  std::uint64_t tensor_identity = 0U;
  // weight_digest is SHA-256 of the complete canonical packed-weight tensor.
  // scale_digest is SHA-256 of the complete block-scale tensor (when present)
  // followed by the four little-endian bytes of tensor_scale_bits. Neither
  // digest may be synthesized from names, shapes, or other metadata.
  Sm87TargetAotProjectionSha256Digest weight_digest{};
  Sm87TargetAotProjectionSha256Digest scale_digest{};
  std::uint32_t output_features = 0U;
  std::uint32_t input_features = 0U;
  std::uint32_t tensor_scale_bits = 0U;
  std::uint64_t payload_offset = 0U;
  std::uint64_t payload_bytes = 0U;
};

[[nodiscard]] constexpr bool
sm87_target_aot_projection_source_binding_is_zero(
    const Sm87TargetAotProjectionPackedSourceBinding& source) noexcept {
  return source.logical_role == Sm87TargetAotLogicalRole::kInvalid &&
         source.partition_index == 0U && source.tensor_identity == 0U &&
         sm87_target_aot_projection_digest_is_zero(source.weight_digest) &&
         sm87_target_aot_projection_digest_is_zero(source.scale_digest) &&
         source.output_features == 0U && source.input_features == 0U &&
         source.tensor_scale_bits == 0U && source.payload_offset == 0U &&
         source.payload_bytes == 0U;
}

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedSourceBinding
sm87_target_aot_projection_packed_source_binding(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::size_t partition_index, const std::uint64_t tensor_identity,
    const Sm87TargetAotProjectionSha256Digest& weight_digest,
    const Sm87TargetAotProjectionSha256Digest& scale_digest,
    const std::uint32_t tensor_scale_bits) noexcept {
  if (!layout.valid() || partition_index >= layout.partition_count) {
    return {};
  }
  const auto& partition = layout.partitions[partition_index];
  return {partition.logical_role,
          static_cast<std::uint32_t>(partition_index),
          tensor_identity,
          weight_digest,
          scale_digest,
          partition.output_features,
          partition.input_features,
          tensor_scale_bits,
          partition.payload_offset,
          partition.payload_bytes};
}

struct Sm87TargetAotProjectionPackedSourceInventory {
  std::uint64_t identity = 0U;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::uint32_t source_count = 0U;
  std::array<Sm87TargetAotProjectionPackedSourceBinding,
             kSm87TargetAotProjectionPackedMaxPartitions>
      sources{};

  [[nodiscard]] constexpr bool valid(
      const Sm87TargetAotProjectionPackedLayout& layout) const noexcept {
    if (!layout.valid() || identity == 0U || role != layout.role ||
        source_count != layout.partition_count) {
      return false;
    }
    for (std::size_t index = 0U; index < sources.size(); ++index) {
      const auto& source = sources[index];
      if (index >= source_count) {
        if (!sm87_target_aot_projection_source_binding_is_zero(source)) {
          return false;
        }
        continue;
      }
      const auto& partition = layout.partitions[index];
      if (source.logical_role != partition.logical_role ||
          source.partition_index != index ||
          source.tensor_identity == 0U ||
          sm87_target_aot_projection_digest_is_zero(source.weight_digest) ||
          sm87_target_aot_projection_digest_is_zero(source.scale_digest) ||
          source.output_features != partition.output_features ||
          source.input_features != partition.input_features ||
          !sm87_target_aot_projection_scale_bits_valid(
              source.tensor_scale_bits) ||
          source.payload_offset != partition.payload_offset ||
          source.payload_bytes != partition.payload_bytes) {
        return false;
      }
      for (std::size_t prior = 0U; prior < index; ++prior) {
        if (source.tensor_identity == sources[prior].tensor_identity) {
          return false;
        }
      }
    }
    return true;
  }
};

struct Sm87TargetAotProjectionPackedManifestSeal {
  std::uint64_t value = 0U;

  [[nodiscard]] friend constexpr bool operator==(
      const Sm87TargetAotProjectionPackedManifestSeal& left,
      const Sm87TargetAotProjectionPackedManifestSeal& right) noexcept {
    return left.value == right.value;
  }
  [[nodiscard]] friend constexpr bool operator!=(
      const Sm87TargetAotProjectionPackedManifestSeal& left,
      const Sm87TargetAotProjectionPackedManifestSeal& right) noexcept {
    return !(left == right);
  }
};

struct Sm87TargetAotProjectionPackedManifest {
  std::array<std::uint8_t, 8U> magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  std::uint32_t header_bytes = 0U;
  std::uint64_t artifact_identity = 0U;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  Sm87TargetAotProjectionPackedPlanIdentity plan_identity =
      Sm87TargetAotProjectionPackedPlanIdentity::kInvalid;
  Sm87TargetAotProjectionPackedLayoutIdentity layout_identity =
      Sm87TargetAotProjectionPackedLayoutIdentity::kInvalid;
  Sm87TargetAotProjectionEncoding encoding =
      Sm87TargetAotProjectionEncoding::kInvalid;
  std::uint64_t source_inventory_identity = 0U;
  std::uint32_t source_count = 0U;
  std::array<Sm87TargetAotProjectionPackedSourceBinding,
             kSm87TargetAotProjectionPackedMaxPartitions>
      sources{};
  std::uint64_t payload_offset = 0U;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t artifact_bytes = 0U;
  std::uint32_t payload_alignment = 0U;
  Sm87TargetAotProjectionSha256Digest payload_digest{};
  bool token_count_independent = false;
  bool cuda_implementation_present = false;
  bool static_resources_qualified = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
  Sm87TargetAotProjectionPackedManifestSeal seal{};
};

// A future loader creates this receipt only after hashing the actual payload
// byte interval. Manifest validation alone authenticates schema/source keys;
// it deliberately cannot authenticate bytes that were not supplied to it.
struct Sm87TargetAotProjectionPackedPayloadReceipt {
  std::uint64_t artifact_identity = 0U;
  std::uint64_t observed_payload_offset = 0U;
  std::uint64_t observed_payload_bytes = 0U;
  Sm87TargetAotProjectionSha256Digest observed_payload_digest{};
  bool digest_computed_from_payload_bytes = false;
};

enum class Sm87TargetAotProjectionPackedTransformIdentity : std::uint16_t {
  kInvalid = 0U,
  // A bit-preserving permutation from each canonical [N,K] source into the
  // ConsumerN64K16LaneComponentV1 address map. NVFP4 nibbles and E4M3FN
  // block-scale bytes are copied exactly; FP8 E4M3FN weight bytes are copied
  // exactly. The independent FP32 tensor scale remains outside the payload.
  kCanonicalNkToConsumerN64K16LaneComponentV1,
};

struct Sm87TargetAotProjectionPackedPartitionTransformReceipt {
  Sm87TargetAotLogicalRole logical_role =
      Sm87TargetAotLogicalRole::kInvalid;
  std::uint32_t partition_index = 0U;
  std::uint64_t tensor_identity = 0U;
  Sm87TargetAotProjectionSha256Digest observed_source_weight_digest{};
  Sm87TargetAotProjectionSha256Digest observed_source_scale_digest{};
  std::uint64_t source_weight_bytes_hashed = 0U;
  std::uint64_t source_scale_bytes_hashed = 0U;
  std::uint64_t repacked_weight_values = 0U;
  std::uint64_t repacked_block_scale_values = 0U;
  // Only NVFP4's E4M3FN block-scale tensor is an admission domain that
  // rejects terminal NaN encodings. FP8 weight bytes are authenticated by
  // digest and retain admitted Marlin's raw 0x7f/0xff -> +/-480 semantics.
  std::uint64_t source_block_scale_e4m3fn_bytes_scanned = 0U;
  std::uint64_t payload_block_scale_e4m3fn_bytes_scanned = 0U;
  std::uint64_t source_forbidden_block_scale_codes = 0U;
  std::uint64_t payload_forbidden_block_scale_codes = 0U;
  std::uint64_t payload_offset = 0U;
  std::uint64_t payload_bytes = 0U;
  bool source_digests_computed_from_tensor_bytes = false;
  bool canonical_address_bijection_applied = false;
  bool bit_exact_weight_permutation = false;
  bool bit_exact_block_scale_permutation = false;
  bool tensor_scale_kept_external = false;
};

[[nodiscard]] constexpr bool
sm87_target_aot_projection_partition_transform_receipt_is_zero(
    const Sm87TargetAotProjectionPackedPartitionTransformReceipt&
        receipt) noexcept {
  return receipt.logical_role == Sm87TargetAotLogicalRole::kInvalid &&
         receipt.partition_index == 0U && receipt.tensor_identity == 0U &&
         sm87_target_aot_projection_digest_is_zero(
             receipt.observed_source_weight_digest) &&
         sm87_target_aot_projection_digest_is_zero(
             receipt.observed_source_scale_digest) &&
         receipt.source_weight_bytes_hashed == 0U &&
         receipt.source_scale_bytes_hashed == 0U &&
         receipt.repacked_weight_values == 0U &&
         receipt.repacked_block_scale_values == 0U &&
         receipt.source_block_scale_e4m3fn_bytes_scanned == 0U &&
         receipt.payload_block_scale_e4m3fn_bytes_scanned == 0U &&
         receipt.source_forbidden_block_scale_codes == 0U &&
         receipt.payload_forbidden_block_scale_codes == 0U &&
         receipt.payload_offset == 0U && receipt.payload_bytes == 0U &&
         !receipt.source_digests_computed_from_tensor_bytes &&
         !receipt.canonical_address_bijection_applied &&
         !receipt.bit_exact_weight_permutation &&
         !receipt.bit_exact_block_scale_permutation &&
         !receipt.tensor_scale_kept_external;
}

struct Sm87TargetAotProjectionPackedTransformReceipt {
  std::uint64_t artifact_identity = 0U;
  std::uint64_t source_inventory_identity = 0U;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  Sm87TargetAotProjectionPackedPlanIdentity plan_identity =
      Sm87TargetAotProjectionPackedPlanIdentity::kInvalid;
  Sm87TargetAotProjectionPackedLayoutIdentity layout_identity =
      Sm87TargetAotProjectionPackedLayoutIdentity::kInvalid;
  Sm87TargetAotProjectionEncoding encoding =
      Sm87TargetAotProjectionEncoding::kInvalid;
  Sm87TargetAotProjectionPackedTransformIdentity transform_identity =
      Sm87TargetAotProjectionPackedTransformIdentity::kInvalid;
  std::uint32_t partition_count = 0U;
  std::array<Sm87TargetAotProjectionPackedPartitionTransformReceipt,
             kSm87TargetAotProjectionPackedMaxPartitions>
      partitions{};
  Sm87TargetAotProjectionPackedPayloadReceipt payload{};
  bool deterministic_transform = false;
  bool no_arithmetic_conversion = false;
  bool no_request_time_repacking = false;
};

[[nodiscard]] constexpr bool
sm87_target_aot_projection_block_scale_e4m3fn_code_is_forbidden(
    const std::uint8_t code) noexcept {
  const bool terminal_nan = code == 0x7fU || code == 0xffU;
  const bool negative_nonzero =
      (code & 0x80U) != 0U && (code & 0x7fU) != 0U;
  return terminal_nan || negative_nonzero;
}

[[nodiscard]] constexpr std::uint64_t
sm87_target_aot_projection_manifest_hash_byte(std::uint64_t hash,
                                               const std::uint8_t byte) noexcept {
  return (hash ^ byte) * 1'099'511'628'211ULL;
}

[[nodiscard]] constexpr std::uint64_t
sm87_target_aot_projection_manifest_hash_u64(std::uint64_t hash,
                                              std::uint64_t value,
                                              const std::size_t bytes) noexcept {
  for (std::size_t index = 0U; index < bytes; ++index) {
    hash = sm87_target_aot_projection_manifest_hash_byte(
        hash, static_cast<std::uint8_t>(value >> (8U * index)));
  }
  return hash;
}

[[nodiscard]] constexpr Sm87TargetAotProjectionPackedManifestSeal
sm87_target_aot_projection_compute_manifest_seal(
    const Sm87TargetAotProjectionPackedManifest& manifest) noexcept {
  std::uint64_t hash = 14'695'981'039'346'656'037ULL;
  for (const std::uint8_t byte : manifest.magic) {
    hash = sm87_target_aot_projection_manifest_hash_byte(hash, byte);
  }
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.abi_major, sizeof(manifest.abi_major));
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.abi_minor, sizeof(manifest.abi_minor));
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.header_bytes, sizeof(manifest.header_bytes));
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.artifact_identity, sizeof(manifest.artifact_identity));
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, static_cast<std::uint8_t>(manifest.role), 1U);
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, static_cast<std::uint16_t>(manifest.plan_identity), 2U);
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, static_cast<std::uint16_t>(manifest.layout_identity), 2U);
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, static_cast<std::uint8_t>(manifest.encoding), 1U);
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.source_inventory_identity,
      sizeof(manifest.source_inventory_identity));
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.source_count, sizeof(manifest.source_count));
  for (const auto& source : manifest.sources) {
    hash = sm87_target_aot_projection_manifest_hash_u64(
        hash, static_cast<std::uint8_t>(source.logical_role), 1U);
    hash = sm87_target_aot_projection_manifest_hash_u64(
        hash, source.partition_index, sizeof(source.partition_index));
    hash = sm87_target_aot_projection_manifest_hash_u64(
        hash, source.tensor_identity, sizeof(source.tensor_identity));
    for (const std::uint8_t byte : source.weight_digest.bytes) {
      hash = sm87_target_aot_projection_manifest_hash_byte(hash, byte);
    }
    for (const std::uint8_t byte : source.scale_digest.bytes) {
      hash = sm87_target_aot_projection_manifest_hash_byte(hash, byte);
    }
    hash = sm87_target_aot_projection_manifest_hash_u64(
        hash, source.output_features, sizeof(source.output_features));
    hash = sm87_target_aot_projection_manifest_hash_u64(
        hash, source.input_features, sizeof(source.input_features));
    hash = sm87_target_aot_projection_manifest_hash_u64(
        hash, source.tensor_scale_bits, sizeof(source.tensor_scale_bits));
    hash = sm87_target_aot_projection_manifest_hash_u64(
        hash, source.payload_offset, sizeof(source.payload_offset));
    hash = sm87_target_aot_projection_manifest_hash_u64(
        hash, source.payload_bytes, sizeof(source.payload_bytes));
  }
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.payload_offset, sizeof(manifest.payload_offset));
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.payload_bytes, sizeof(manifest.payload_bytes));
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.artifact_bytes, sizeof(manifest.artifact_bytes));
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.payload_alignment, sizeof(manifest.payload_alignment));
  for (const std::uint8_t byte : manifest.payload_digest.bytes) {
    hash = sm87_target_aot_projection_manifest_hash_byte(hash, byte);
  }
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.token_count_independent ? 1U : 0U, 1U);
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.cuda_implementation_present ? 1U : 0U, 1U);
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.static_resources_qualified ? 1U : 0U, 1U);
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.numerical_contract_qualified ? 1U : 0U, 1U);
  hash = sm87_target_aot_projection_manifest_hash_u64(
      hash, manifest.production_dispatch_eligible ? 1U : 0U, 1U);
  return {hash};
}

[[nodiscard]] constexpr bool sm87_target_aot_projection_same_source_binding(
    const Sm87TargetAotProjectionPackedSourceBinding& left,
    const Sm87TargetAotProjectionPackedSourceBinding& right) noexcept {
  return left.logical_role == right.logical_role &&
         left.partition_index == right.partition_index &&
         left.tensor_identity == right.tensor_identity &&
         left.weight_digest == right.weight_digest &&
         left.scale_digest == right.scale_digest &&
         left.output_features == right.output_features &&
         left.input_features == right.input_features &&
         left.tensor_scale_bits == right.tensor_scale_bits &&
         left.payload_offset == right.payload_offset &&
         left.payload_bytes == right.payload_bytes;
}

[[nodiscard]] constexpr bool
sm87_target_aot_projection_packed_manifest_structurally_valid(
    const Sm87TargetAotProjectionPackedManifest& manifest) noexcept {
  for (std::size_t index = 0U; index < manifest.magic.size(); ++index) {
    if (manifest.magic[index] !=
        kSm87TargetAotProjectionPackedMagic[index]) {
      return false;
    }
  }
  const auto layout =
      sm87_target_aot_projection_packed_layout(manifest.role);
  if (!layout.valid() ||
      manifest.abi_major != kSm87TargetAotProjectionPackedAbiMajor ||
      manifest.abi_minor != kSm87TargetAotProjectionPackedAbiMinor ||
      manifest.header_bytes != kSm87TargetAotProjectionPackedHeaderBytes ||
      manifest.artifact_identity == 0U ||
      manifest.plan_identity != layout.plan_identity ||
      manifest.layout_identity != layout.layout_identity ||
      manifest.encoding != layout.encoding ||
      manifest.source_inventory_identity == 0U ||
      manifest.source_count != layout.partition_count ||
      manifest.payload_offset != kSm87TargetAotProjectionPackedHeaderBytes ||
      manifest.payload_bytes != layout.payload_bytes ||
      manifest.payload_alignment != layout.payload_alignment ||
      !sm87_target_aot_packed_aligned(manifest.payload_offset,
                                      manifest.payload_alignment) ||
      !sm87_target_aot_packed_add_fits(manifest.payload_offset,
                                       manifest.payload_bytes) ||
      manifest.artifact_bytes !=
          manifest.payload_offset + manifest.payload_bytes ||
      sm87_target_aot_projection_digest_is_zero(manifest.payload_digest) ||
      !manifest.token_count_independent ||
      manifest.cuda_implementation_present ||
      manifest.static_resources_qualified ||
      manifest.numerical_contract_qualified ||
      manifest.production_dispatch_eligible || manifest.seal.value == 0U ||
      manifest.seal !=
          sm87_target_aot_projection_compute_manifest_seal(manifest)) {
    return false;
  }
  Sm87TargetAotProjectionPackedSourceInventory inventory;
  inventory.identity = manifest.source_inventory_identity;
  inventory.role = manifest.role;
  inventory.source_count = manifest.source_count;
  inventory.sources = manifest.sources;
  return inventory.valid(layout);
}

[[nodiscard]] constexpr bool
sm87_target_aot_projection_validate_packed_manifest(
    const Sm87TargetAotProjectionPackedManifest& manifest,
    const Sm87TargetAotProjectionPackedSourceInventory& expected) noexcept {
  if (!sm87_target_aot_projection_packed_manifest_structurally_valid(
          manifest)) {
    return false;
  }
  const auto layout =
      sm87_target_aot_projection_packed_layout(manifest.role);
  if (!expected.valid(layout) ||
      expected.identity != manifest.source_inventory_identity ||
      expected.role != manifest.role ||
      expected.source_count != manifest.source_count) {
    return false;
  }
  for (std::size_t index = 0U; index < expected.sources.size(); ++index) {
    if (!sm87_target_aot_projection_same_source_binding(
            expected.sources[index], manifest.sources[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool
sm87_target_aot_projection_validate_payload_receipt(
    const Sm87TargetAotProjectionPackedManifest& manifest,
    const Sm87TargetAotProjectionPackedPayloadReceipt& receipt) noexcept {
  return sm87_target_aot_projection_packed_manifest_structurally_valid(
             manifest) &&
         receipt.digest_computed_from_payload_bytes &&
         receipt.artifact_identity == manifest.artifact_identity &&
         receipt.observed_payload_offset == manifest.payload_offset &&
         receipt.observed_payload_bytes == manifest.payload_bytes &&
         !sm87_target_aot_projection_digest_is_zero(
             receipt.observed_payload_digest) &&
         receipt.observed_payload_digest == manifest.payload_digest;
}

[[nodiscard]] constexpr bool
sm87_target_aot_projection_validate_transform_receipt(
    const Sm87TargetAotProjectionPackedManifest& manifest,
    const Sm87TargetAotProjectionPackedSourceInventory& expected,
    const Sm87TargetAotProjectionPackedTransformReceipt& receipt) noexcept {
  if (!sm87_target_aot_projection_validate_packed_manifest(manifest,
                                                           expected) ||
      receipt.artifact_identity != manifest.artifact_identity ||
      receipt.source_inventory_identity !=
          manifest.source_inventory_identity ||
      receipt.role != manifest.role ||
      receipt.plan_identity != manifest.plan_identity ||
      receipt.layout_identity != manifest.layout_identity ||
      receipt.encoding != manifest.encoding ||
      receipt.transform_identity !=
          Sm87TargetAotProjectionPackedTransformIdentity::
              kCanonicalNkToConsumerN64K16LaneComponentV1 ||
      receipt.partition_count != manifest.source_count ||
      !receipt.deterministic_transform ||
      !receipt.no_arithmetic_conversion ||
      !receipt.no_request_time_repacking ||
      !sm87_target_aot_projection_validate_payload_receipt(
          manifest, receipt.payload)) {
    return false;
  }

  const auto layout =
      sm87_target_aot_projection_packed_layout(manifest.role);
  for (std::size_t index = 0U; index < receipt.partitions.size(); ++index) {
    const auto& observed = receipt.partitions[index];
    if (index >= receipt.partition_count) {
      if (!sm87_target_aot_projection_partition_transform_receipt_is_zero(
              observed)) {
        return false;
      }
      continue;
    }

    const auto& partition = layout.partitions[index];
    const auto& source = expected.sources[index];
    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    const std::uint64_t source_weight_bytes =
        values * partition.weight_bits / 8U;
    const std::uint64_t block_scale_values =
        partition.block_scale_group_k == 0U
            ? 0U
            : values / partition.block_scale_group_k;
    // scale_digest authenticates all scale sources: the optional E4M3FN
    // block-scale tensor followed by the independent FP32 tensor scale.
    const std::uint64_t source_scale_bytes =
        block_scale_values + sizeof(std::uint32_t);
    const std::uint64_t block_scale_e4m3fn_bytes = block_scale_values;
    if (observed.logical_role != partition.logical_role ||
        observed.partition_index != index ||
        observed.tensor_identity != source.tensor_identity ||
        observed.observed_source_weight_digest != source.weight_digest ||
        observed.observed_source_scale_digest != source.scale_digest ||
        observed.source_weight_bytes_hashed != source_weight_bytes ||
        observed.source_scale_bytes_hashed != source_scale_bytes ||
        observed.repacked_weight_values != values ||
        observed.repacked_block_scale_values != block_scale_values ||
        observed.source_block_scale_e4m3fn_bytes_scanned !=
            block_scale_e4m3fn_bytes ||
        observed.payload_block_scale_e4m3fn_bytes_scanned !=
            block_scale_e4m3fn_bytes ||
        observed.source_forbidden_block_scale_codes != 0U ||
        observed.payload_forbidden_block_scale_codes != 0U ||
        observed.payload_offset != partition.payload_offset ||
        observed.payload_bytes != partition.payload_bytes ||
        !observed.source_digests_computed_from_tensor_bytes ||
        !observed.canonical_address_bijection_applied ||
        !observed.bit_exact_weight_permutation ||
        observed.bit_exact_block_scale_permutation !=
            (block_scale_values != 0U) ||
        !observed.tensor_scale_kept_external) {
      return false;
    }
  }
  return true;
}

inline bool sm87_target_aot_projection_seal_packed_manifest(
    Sm87TargetAotProjectionPackedManifest* const manifest) noexcept {
  if (manifest == nullptr) {
    return false;
  }
  manifest->seal = {};
  manifest->seal =
      sm87_target_aot_projection_compute_manifest_seal(*manifest);
  if (!sm87_target_aot_projection_packed_manifest_structurally_valid(
          *manifest)) {
    manifest->seal = {};
    return false;
  }
  return true;
}

[[nodiscard]] inline Sm87TargetAotProjectionPackedManifest
sm87_target_aot_projection_make_packed_manifest(
    const Sm87TargetAotProjectionRole role,
    const std::uint64_t artifact_identity,
    const Sm87TargetAotProjectionPackedSourceInventory& sources,
    const Sm87TargetAotProjectionSha256Digest& payload_digest) noexcept {
  const auto layout = sm87_target_aot_projection_packed_layout(role);
  if (!layout.valid() || !sources.valid(layout) ||
      sm87_target_aot_projection_digest_is_zero(payload_digest) ||
      artifact_identity == 0U ||
      !sm87_target_aot_packed_add_fits(
          kSm87TargetAotProjectionPackedHeaderBytes,
          layout.payload_bytes)) {
    return {};
  }
  Sm87TargetAotProjectionPackedManifest manifest;
  manifest.magic = kSm87TargetAotProjectionPackedMagic;
  manifest.abi_major = kSm87TargetAotProjectionPackedAbiMajor;
  manifest.abi_minor = kSm87TargetAotProjectionPackedAbiMinor;
  manifest.header_bytes = kSm87TargetAotProjectionPackedHeaderBytes;
  manifest.artifact_identity = artifact_identity;
  manifest.role = role;
  manifest.plan_identity = layout.plan_identity;
  manifest.layout_identity = layout.layout_identity;
  manifest.encoding = layout.encoding;
  manifest.source_inventory_identity = sources.identity;
  manifest.source_count = sources.source_count;
  manifest.sources = sources.sources;
  manifest.payload_offset = kSm87TargetAotProjectionPackedHeaderBytes;
  manifest.payload_bytes = layout.payload_bytes;
  manifest.artifact_bytes = manifest.payload_offset + manifest.payload_bytes;
  manifest.payload_alignment = layout.payload_alignment;
  manifest.payload_digest = payload_digest;
  manifest.token_count_independent = true;
  if (!sm87_target_aot_projection_seal_packed_manifest(&manifest)) {
    return {};
  }
  return manifest;
}

inline constexpr auto kSm87TargetAotNvFp4GateUpPackedLayout =
    sm87_target_aot_projection_packed_layout(
        Sm87TargetAotProjectionRole::kNvFp4GateUp);
inline constexpr auto kSm87TargetAotNvFp4DownPackedLayout =
    sm87_target_aot_projection_packed_layout(
        Sm87TargetAotProjectionRole::kNvFp4Down);
inline constexpr auto kSm87TargetAotFp8GdnQkvZPackedLayout =
    sm87_target_aot_projection_packed_layout(
        Sm87TargetAotProjectionRole::kFp8GdnQkvZ);
inline constexpr auto kSm87TargetAotFp8FullQkvPackedLayout =
    sm87_target_aot_projection_packed_layout(
        Sm87TargetAotProjectionRole::kFp8FullQkv);
inline constexpr auto kSm87TargetAotFp8AttentionOutputPackedLayout =
    sm87_target_aot_projection_packed_layout(
        Sm87TargetAotProjectionRole::kFp8AttentionOutput);

static_assert(kSm87TargetAotNvFp4GateUpPackedLayout.valid());
static_assert(kSm87TargetAotNvFp4DownPackedLayout.valid());
static_assert(kSm87TargetAotFp8GdnQkvZPackedLayout.valid());
static_assert(kSm87TargetAotFp8FullQkvPackedLayout.valid());
static_assert(kSm87TargetAotFp8AttentionOutputPackedLayout.valid());
static_assert(kSm87TargetAotNvFp4GateUpPackedLayout.payload_bytes ==
              100'270'080U);
static_assert(kSm87TargetAotNvFp4DownPackedLayout.payload_bytes ==
              50'135'040U);
static_assert(kSm87TargetAotFp8GdnQkvZPackedLayout.payload_bytes ==
              83'886'080U);
static_assert(kSm87TargetAotFp8FullQkvPackedLayout.payload_bytes ==
              73'400'320U);
static_assert(kSm87TargetAotFp8AttentionOutputPackedLayout.payload_bytes ==
              31'457'280U);
static_assert(kSm87TargetAotNvFp4GateUpPackedLayout
                      .partitions[0U]
                      .weight_bytes_per_cell == 8'192U);
static_assert(kSm87TargetAotNvFp4GateUpPackedLayout
                      .partitions[0U]
                      .block_scale_bytes_per_cell == 1'024U);
static_assert(kSm87TargetAotFp8GdnQkvZPackedLayout
                      .partitions[0U]
                      .weight_bytes_per_cell == 16'384U);
static_assert(!kSm87TargetAotNvFp4GateUpPackedLayout
                   .cuda_implementation_present &&
              !kSm87TargetAotNvFp4GateUpPackedLayout
                   .static_resources_qualified &&
              !kSm87TargetAotNvFp4GateUpPackedLayout
                   .numerical_contract_qualified &&
              !kSm87TargetAotNvFp4GateUpPackedLayout
                   .production_dispatch_eligible);

}  // namespace q3x::kernels
