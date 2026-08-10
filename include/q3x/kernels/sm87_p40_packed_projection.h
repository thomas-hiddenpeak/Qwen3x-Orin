#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Host/device-neutral ABI for WP-P40-PACKED-PROJECTION-v1.  This header
// describes authenticated artifacts and fixed SM87 work ownership; it does
// not select a runtime route or authorize a fallback.
inline constexpr std::array<std::uint8_t, 8U>
    kSm87P40PackedProjectionMagic{{'Q', '3', 'X', 'P', '4', '0', 'P', '1'}};
inline constexpr std::uint16_t kSm87P40PackedProjectionAbiMajor = 1U;
inline constexpr std::uint16_t kSm87P40PackedProjectionAbiMinor = 0U;
inline constexpr std::uint32_t kSm87P40PackedProjectionHeaderBytes = 4'096U;
inline constexpr std::uint32_t kSm87P40PackedProjectionPayloadAlignment =
    256U;
inline constexpr std::uint32_t kSm87P40PackedProjectionTokens = 40'000U;
inline constexpr std::uint32_t kSm87P40PackedProjectionHidden = 5'120U;
inline constexpr std::uint32_t kSm87P40PackedProjectionIntermediate = 17'408U;
inline constexpr std::uint32_t kSm87P40PackedProjectionLayerCount = 64U;
inline constexpr std::uint32_t kSm87P40PackedProjectionLinearLayerCount =
    48U;
inline constexpr std::uint32_t kSm87P40PackedProjectionFullLayerCount = 16U;
inline constexpr std::uint32_t kSm87P40PackedProjectionFp8LogicalRoleCount =
    208U;
inline constexpr std::uint32_t
    kSm87P40PackedProjectionNvFp4LogicalRoleCount = 192U;
inline constexpr std::uint32_t kSm87P40PackedProjectionSourceIdentityCount =
    kSm87P40PackedProjectionFp8LogicalRoleCount +
    kSm87P40PackedProjectionNvFp4LogicalRoleCount;
inline constexpr std::uint32_t kSm87P40PackedProjectionArtifactCount = 256U;
inline constexpr std::uint32_t kSm87P40PackedProjectionPersistentCtas = 32U;
inline constexpr std::uint32_t kSm87P40PackedProjectionSmCount = 16U;
inline constexpr std::uint32_t kSm87P40PackedProjectionTileM = 64U;
inline constexpr std::uint32_t kSm87P40PackedProjectionGridM = 625U;
inline constexpr std::size_t kSm87P40PackedProjectionMaximumSources = 3U;

enum class Sm87P40PackedProjectionRole : std::uint8_t {
  kNvFp4GateUp = 0U,
  kNvFp4Down,
  kFp8LinearQkvZ,
  kFp8FullQkv,
  kFp8AttentionOutput,
  kCount,
  kInvalid = 0xffU,
};

enum class Sm87P40PackedLogicalRole : std::uint8_t {
  kNvFp4Gate = 0U,
  kNvFp4Up,
  kNvFp4Down,
  kFp8LinearQkv,
  kFp8LinearZ,
  kFp8FullQ,
  kFp8FullK,
  kFp8FullV,
  kFp8AttentionOutput,
  kInvalid = 0xffU,
};

enum class Sm87P40PackedEncoding : std::uint8_t {
  kNvFp4E2M1Block16 = 0U,
  kFp8E4M3TensorScale,
  kInvalid = 0xffU,
};

enum class Sm87P40PackedRaster : std::uint8_t {
  kGroupedMNMajor = 0U,
  kAMajorNFast,
  kInvalid = 0xffU,
};

enum class Sm87P40PackedTactic : std::uint8_t {
  kNvFp4GateUpM64N256K64Gm4 = 0U,
  kNvFp4DownM64N128K64AMajor,
  kFp8WideM64N128K64Gm4,
  kFp8FullQkvMixedPersistent,
  kFp8SmallKvM64N64K128Gm4,
  kFp8OutputM64N128K64Gm3,
  kInvalid = 0xffU,
};

enum class Sm87P40PackedProvider : std::uint8_t {
  kNativeSm87 = 0U,
  kInvalid = 0xffU,
};

enum Sm87P40PackedPolicy : std::uint32_t {
  kSm87P40PackedExactNumerics = 1U << 0U,
  kSm87P40PackedAheadOfTimeLayout = 1U << 1U,
  kSm87P40PackedNoFallback = 1U << 2U,
  kSm87P40PackedNoMtp = 1U << 3U,
  kSm87P40PackedNoCuBlasLt = 1U << 4U,
  kSm87P40PackedFullKAccumulation = 1U << 5U,
};

inline constexpr std::uint32_t kSm87P40PackedRequiredPolicy =
    kSm87P40PackedExactNumerics |
    kSm87P40PackedAheadOfTimeLayout |
    kSm87P40PackedNoFallback | kSm87P40PackedNoMtp |
    kSm87P40PackedNoCuBlasLt | kSm87P40PackedFullKAccumulation;

struct Sm87P40PackedDigest {
  std::array<std::uint8_t, 32U> bytes{};

  [[nodiscard]] friend constexpr bool operator==(
      const Sm87P40PackedDigest& left,
      const Sm87P40PackedDigest& right) noexcept {
    for (std::size_t index = 0U; index < left.bytes.size(); ++index) {
      if (left.bytes[index] != right.bytes[index]) {
        return false;
      }
    }
    return true;
  }
  [[nodiscard]] friend constexpr bool operator!=(
      const Sm87P40PackedDigest& left,
      const Sm87P40PackedDigest& right) noexcept {
    return !(left == right);
  }
};

[[nodiscard]] constexpr bool sm87_p40_packed_digest_is_zero(
    const Sm87P40PackedDigest& digest) noexcept {
  for (const std::uint8_t byte : digest.bytes) {
    if (byte != 0U) {
      return false;
    }
  }
  return true;
}

struct Sm87P40PackedSourceIdentity {
  Sm87P40PackedLogicalRole role = Sm87P40PackedLogicalRole::kInvalid;
  std::uint64_t tensor_identity = 0U;
  Sm87P40PackedDigest weight_digest{};
  Sm87P40PackedDigest scale_digest{};
  // Exact IEEE-754 bits of the original tensor/global output scale. Gate and
  // Up always occupy independent entries even when their values are equal.
  std::uint32_t global_scale_bits = 0U;
};

struct Sm87P40PackedPartitionPlan {
  Sm87P40PackedLogicalRole role = Sm87P40PackedLogicalRole::kInvalid;
  Sm87P40PackedTactic tactic = Sm87P40PackedTactic::kInvalid;
  std::uint32_t output_features = 0U;
  std::uint32_t input_features = 0U;
  std::uint32_t first_task_n_tile = 0U;
  std::uint32_t task_n_tiles = 0U;
  std::uint32_t tile_n = 0U;
  std::uint32_t tile_k = 0U;
  std::uint32_t threads = 0U;
  std::uint32_t warps = 0U;
  std::uint32_t pipeline_stages = 0U;
  std::uint32_t k_tiles = 0U;
  std::uint32_t fragment_weight_bytes = 0U;
  std::uint32_t fragment_scale_bytes = 0U;
  std::uint32_t cell_bytes = 0U;
  std::uint64_t payload_offset = 0U;
  std::uint64_t payload_bytes = 0U;
};

struct Sm87P40PackedProjectionPlan {
  Sm87P40PackedProjectionRole role =
      Sm87P40PackedProjectionRole::kInvalid;
  Sm87P40PackedEncoding encoding = Sm87P40PackedEncoding::kInvalid;
  Sm87P40PackedTactic tactic = Sm87P40PackedTactic::kInvalid;
  Sm87P40PackedRaster raster = Sm87P40PackedRaster::kInvalid;
  std::uint32_t token_count = 0U;
  std::uint32_t tile_m = 0U;
  std::uint32_t grid_m = 0U;
  std::uint32_t grid_n = 0U;
  std::uint32_t group_m = 0U;
  std::uint32_t persistent_ctas = 0U;
  std::uint32_t source_count = 0U;
  std::uint32_t policy = 0U;
  std::uint64_t logical_tasks = 0U;
  std::uint64_t payload_bytes = 0U;
  std::array<Sm87P40PackedPartitionPlan,
             kSm87P40PackedProjectionMaximumSources>
      partitions{};

  [[nodiscard]] constexpr bool valid() const noexcept;
};

[[nodiscard]] constexpr bool sm87_p40_packed_is_full_layer(
    const std::size_t layer_index) noexcept {
  return layer_index < kSm87P40PackedProjectionLayerCount &&
         ((layer_index + 1U) % 4U) == 0U;
}

[[nodiscard]] constexpr Sm87P40PackedPartitionPlan
sm87_p40_packed_partition(
    const Sm87P40PackedLogicalRole role,
    const Sm87P40PackedTactic tactic, const std::uint32_t output_features,
    const std::uint32_t input_features,
    const std::uint32_t first_task_n_tile,
    const std::uint32_t tile_n, const std::uint32_t tile_k,
    const std::uint32_t threads, const std::uint32_t warps,
    const std::uint32_t pipeline_stages,
    const std::uint64_t payload_offset,
    const std::uint32_t weight_bits,
    const std::uint32_t scale_group_size) noexcept {
  const std::uint32_t task_n_tiles = output_features / tile_n;
  const std::uint32_t k_tiles = input_features / tile_k;
  const std::uint32_t columns_per_warp = tile_n / warps;
  const std::uint32_t fragment_weight_bytes =
      columns_per_warp * 16U * weight_bits / 8U;
  const std::uint32_t fragment_scale_bytes =
      scale_group_size == 0U ? 0U : columns_per_warp;
  const std::uint32_t cell_bytes =
      (tile_k / 16U) * warps *
      (fragment_weight_bytes + fragment_scale_bytes);
  const std::uint64_t payload_bytes =
      static_cast<std::uint64_t>(task_n_tiles) * k_tiles * cell_bytes;
  return {role,
          tactic,
          output_features,
          input_features,
          first_task_n_tile,
          task_n_tiles,
          tile_n,
          tile_k,
          threads,
          warps,
          pipeline_stages,
          k_tiles,
          fragment_weight_bytes,
          fragment_scale_bytes,
          cell_bytes,
          payload_offset,
          payload_bytes};
}

[[nodiscard]] constexpr Sm87P40PackedProjectionPlan
sm87_p40_packed_projection_plan(
    const Sm87P40PackedProjectionRole role) noexcept {
  Sm87P40PackedProjectionPlan plan;
  plan.role = role;
  plan.token_count = kSm87P40PackedProjectionTokens;
  plan.tile_m = kSm87P40PackedProjectionTileM;
  plan.grid_m = kSm87P40PackedProjectionGridM;
  plan.persistent_ctas = kSm87P40PackedProjectionPersistentCtas;
  plan.policy = kSm87P40PackedRequiredPolicy;

  if (role == Sm87P40PackedProjectionRole::kNvFp4GateUp) {
    plan.encoding = Sm87P40PackedEncoding::kNvFp4E2M1Block16;
    plan.tactic = Sm87P40PackedTactic::kNvFp4GateUpM64N256K64Gm4;
    plan.raster = Sm87P40PackedRaster::kGroupedMNMajor;
    plan.grid_n = 136U;
    plan.group_m = 4U;
    plan.source_count = 2U;
    // One cell contains Gate128 followed by Up128 at warp-fragment
    // granularity: warps 0..3 own Gate and 4..7 own Up.
    const auto gate = sm87_p40_packed_partition(
        Sm87P40PackedLogicalRole::kNvFp4Gate, plan.tactic,
        kSm87P40PackedProjectionIntermediate,
        kSm87P40PackedProjectionHidden, 0U, 128U, 64U, 256U, 4U, 4U,
        0U, 4U, 16U);
    auto up = gate;
    up.role = Sm87P40PackedLogicalRole::kNvFp4Up;
    // The paired physical cell is twice one logical branch fragment stream.
    const std::uint64_t paired_payload = 2U * gate.payload_bytes;
    plan.partitions[0U] = gate;
    plan.partitions[0U].payload_bytes = paired_payload;
    plan.partitions[1U] = up;
    plan.partitions[1U].payload_bytes = paired_payload;
    plan.payload_bytes = paired_payload;
  } else if (role == Sm87P40PackedProjectionRole::kNvFp4Down) {
    plan.encoding = Sm87P40PackedEncoding::kNvFp4E2M1Block16;
    plan.tactic = Sm87P40PackedTactic::kNvFp4DownM64N128K64AMajor;
    plan.raster = Sm87P40PackedRaster::kAMajorNFast;
    plan.grid_n = 40U;
    plan.group_m = 1U;
    plan.source_count = 1U;
    plan.partitions[0U] = sm87_p40_packed_partition(
        Sm87P40PackedLogicalRole::kNvFp4Down, plan.tactic,
        kSm87P40PackedProjectionHidden,
        kSm87P40PackedProjectionIntermediate, 0U, 128U, 64U, 128U, 4U,
        4U, 0U, 4U, 16U);
    plan.payload_bytes = plan.partitions[0U].payload_bytes;
  } else if (role == Sm87P40PackedProjectionRole::kFp8LinearQkvZ) {
    plan.encoding = Sm87P40PackedEncoding::kFp8E4M3TensorScale;
    plan.tactic = Sm87P40PackedTactic::kFp8WideM64N128K64Gm4;
    plan.raster = Sm87P40PackedRaster::kGroupedMNMajor;
    plan.grid_n = 128U;
    plan.group_m = 4U;
    plan.source_count = 2U;
    plan.partitions[0U] = sm87_p40_packed_partition(
        Sm87P40PackedLogicalRole::kFp8LinearQkv, plan.tactic, 10'240U,
        5'120U, 0U, 128U, 64U, 128U, 4U, 4U, 0U, 8U, 0U);
    plan.partitions[1U] = sm87_p40_packed_partition(
        Sm87P40PackedLogicalRole::kFp8LinearZ, plan.tactic, 6'144U,
        5'120U, 80U, 128U, 64U, 128U, 4U, 4U,
        plan.partitions[0U].payload_bytes, 8U, 0U);
    plan.payload_bytes = plan.partitions[0U].payload_bytes +
                         plan.partitions[1U].payload_bytes;
  } else if (role == Sm87P40PackedProjectionRole::kFp8FullQkv) {
    plan.encoding = Sm87P40PackedEncoding::kFp8E4M3TensorScale;
    plan.tactic = Sm87P40PackedTactic::kFp8FullQkvMixedPersistent;
    plan.raster = Sm87P40PackedRaster::kGroupedMNMajor;
    plan.grid_n = 128U;
    plan.group_m = 4U;
    plan.source_count = 3U;
    plan.partitions[0U] = sm87_p40_packed_partition(
        Sm87P40PackedLogicalRole::kFp8FullQ,
        Sm87P40PackedTactic::kFp8WideM64N128K64Gm4, 12'288U, 5'120U,
        0U, 128U, 64U, 128U, 4U, 4U, 0U, 8U, 0U);
    plan.partitions[1U] = sm87_p40_packed_partition(
        Sm87P40PackedLogicalRole::kFp8FullK,
        Sm87P40PackedTactic::kFp8SmallKvM64N64K128Gm4, 1'024U, 5'120U,
        96U, 64U, 128U, 128U, 4U, 3U,
        plan.partitions[0U].payload_bytes, 8U, 0U);
    plan.partitions[2U] = sm87_p40_packed_partition(
        Sm87P40PackedLogicalRole::kFp8FullV,
        Sm87P40PackedTactic::kFp8SmallKvM64N64K128Gm4, 1'024U, 5'120U,
        112U, 64U, 128U, 128U, 4U, 3U,
        plan.partitions[0U].payload_bytes +
            plan.partitions[1U].payload_bytes,
        8U, 0U);
    plan.payload_bytes = plan.partitions[0U].payload_bytes +
                         plan.partitions[1U].payload_bytes +
                         plan.partitions[2U].payload_bytes;
  } else if (role == Sm87P40PackedProjectionRole::kFp8AttentionOutput) {
    plan.encoding = Sm87P40PackedEncoding::kFp8E4M3TensorScale;
    plan.tactic = Sm87P40PackedTactic::kFp8OutputM64N128K64Gm3;
    plan.raster = Sm87P40PackedRaster::kGroupedMNMajor;
    plan.grid_n = 40U;
    plan.group_m = 3U;
    plan.source_count = 1U;
    plan.partitions[0U] = sm87_p40_packed_partition(
        Sm87P40PackedLogicalRole::kFp8AttentionOutput, plan.tactic, 5'120U,
        6'144U, 0U, 128U, 64U, 128U, 4U, 4U, 0U, 8U, 0U);
    plan.payload_bytes = plan.partitions[0U].payload_bytes;
  } else {
    return {};
  }
  plan.logical_tasks =
      static_cast<std::uint64_t>(plan.grid_m) * plan.grid_n;
  return plan;
}

// Explicit token-count admission surface for callers that have not yet
// proven P40 ownership. This ABI has no generic/tail route.
[[nodiscard]] constexpr Sm87P40PackedProjectionPlan
sm87_p40_packed_projection_plan(
    const Sm87P40PackedProjectionRole role,
    const std::size_t token_count) noexcept {
  return token_count == kSm87P40PackedProjectionTokens
             ? sm87_p40_packed_projection_plan(role)
             : Sm87P40PackedProjectionPlan{};
}

[[nodiscard]] constexpr bool sm87_p40_packed_same_partition(
    const Sm87P40PackedPartitionPlan& left,
    const Sm87P40PackedPartitionPlan& right) noexcept {
  return left.role == right.role && left.tactic == right.tactic &&
         left.output_features == right.output_features &&
         left.input_features == right.input_features &&
         left.first_task_n_tile == right.first_task_n_tile &&
         left.task_n_tiles == right.task_n_tiles &&
         left.tile_n == right.tile_n && left.tile_k == right.tile_k &&
         left.threads == right.threads && left.warps == right.warps &&
         left.pipeline_stages == right.pipeline_stages &&
         left.k_tiles == right.k_tiles &&
         left.fragment_weight_bytes == right.fragment_weight_bytes &&
         left.fragment_scale_bytes == right.fragment_scale_bytes &&
         left.cell_bytes == right.cell_bytes &&
         left.payload_offset == right.payload_offset &&
         left.payload_bytes == right.payload_bytes;
}

[[nodiscard]] constexpr bool sm87_p40_packed_same_plan(
    const Sm87P40PackedProjectionPlan& left,
    const Sm87P40PackedProjectionPlan& right) noexcept {
  if (left.role != right.role || left.encoding != right.encoding ||
      left.tactic != right.tactic || left.raster != right.raster ||
      left.token_count != right.token_count ||
      left.tile_m != right.tile_m || left.grid_m != right.grid_m ||
      left.grid_n != right.grid_n || left.group_m != right.group_m ||
      left.persistent_ctas != right.persistent_ctas ||
      left.source_count != right.source_count ||
      left.policy != right.policy ||
      left.logical_tasks != right.logical_tasks ||
      left.payload_bytes != right.payload_bytes) {
    return false;
  }
  for (std::size_t index = 0U; index < left.partitions.size(); ++index) {
    if (!sm87_p40_packed_same_partition(left.partitions[index],
                                        right.partitions[index])) {
      return false;
    }
  }
  return true;
}

constexpr bool Sm87P40PackedProjectionPlan::valid() const noexcept {
  if (role == Sm87P40PackedProjectionRole::kInvalid ||
      role == Sm87P40PackedProjectionRole::kCount) {
    return false;
  }
  return sm87_p40_packed_same_plan(
      *this, sm87_p40_packed_projection_plan(role));
}

struct Sm87P40PackedTask {
  std::uint32_t m_tile = 0U;
  std::uint32_t n_tile = 0U;
  std::uint8_t partition_index = 0xffU;
  std::uint8_t source_partition_mask = 0U;
  std::uint32_t local_n_tile = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr std::uint8_t sm87_p40_packed_partition_for_n_tile(
    const Sm87P40PackedProjectionPlan& plan,
    const std::uint32_t n_tile) noexcept {
  if (plan.role == Sm87P40PackedProjectionRole::kNvFp4GateUp) {
    return 0xffU;
  }
  for (std::uint8_t index = 0U; index < plan.source_count; ++index) {
    const auto& partition = plan.partitions[index];
    if (n_tile >= partition.first_task_n_tile &&
        n_tile - partition.first_task_n_tile < partition.task_n_tiles) {
      return index;
    }
  }
  return 0xffU;
}

[[nodiscard]] constexpr Sm87P40PackedTask sm87_p40_packed_task(
    const Sm87P40PackedProjectionPlan& plan,
    const std::uint64_t linear_task) noexcept {
  if (!plan.valid() || linear_task >= plan.logical_tasks) {
    return {};
  }
  const std::uint64_t group_span =
      static_cast<std::uint64_t>(plan.group_m) * plan.grid_n;
  const std::uint64_t group = linear_task / group_span;
  const std::uint32_t first_m =
      static_cast<std::uint32_t>(group * plan.group_m);
  const std::uint32_t remaining_m = plan.grid_m - first_m;
  const std::uint32_t active_m =
      remaining_m < plan.group_m ? remaining_m : plan.group_m;
  const std::uint64_t offset = linear_task % group_span;
  const std::uint32_t m_tile =
      first_m + static_cast<std::uint32_t>(offset % active_m);
  const std::uint32_t n_tile =
      static_cast<std::uint32_t>(offset / active_m);
  if (plan.role == Sm87P40PackedProjectionRole::kNvFp4GateUp) {
    return {m_tile, n_tile, 0xffU, 0x03U, n_tile, true};
  }
  const std::uint8_t partition_index =
      sm87_p40_packed_partition_for_n_tile(plan, n_tile);
  if (partition_index >= plan.source_count) {
    return {};
  }
  return {m_tile,
          n_tile,
          partition_index,
          static_cast<std::uint8_t>(1U << partition_index),
          n_tile - plan.partitions[partition_index].first_task_n_tile,
          true};
}

struct Sm87P40PackedLayoutCell {
  std::uint8_t partition_index = 0xffU;
  std::uint8_t source_partition_mask = 0U;
  std::uint32_t local_n_tile = 0U;
  std::uint32_t k_tile = 0U;
  std::uint64_t payload_offset = 0U;
  std::uint32_t cell_bytes = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87P40PackedLayoutCell
sm87_p40_packed_layout_cell(
    const Sm87P40PackedProjectionPlan& plan,
    const std::uint32_t n_tile, const std::uint32_t k_tile) noexcept {
  if (!plan.valid() || n_tile >= plan.grid_n) {
    return {};
  }
  if (plan.role == Sm87P40PackedProjectionRole::kNvFp4GateUp) {
    const auto& branch = plan.partitions[0U];
    if (k_tile >= branch.k_tiles) {
      return {};
    }
    const std::uint32_t paired_cell_bytes = 2U * branch.cell_bytes;
    return {0xffU,
            0x03U,
            n_tile,
            k_tile,
            (static_cast<std::uint64_t>(n_tile) * branch.k_tiles + k_tile) *
                paired_cell_bytes,
            paired_cell_bytes,
            true};
  }
  const std::uint8_t partition_index =
      sm87_p40_packed_partition_for_n_tile(plan, n_tile);
  if (partition_index >= plan.source_count) {
    return {};
  }
  const auto& partition = plan.partitions[partition_index];
  if (k_tile >= partition.k_tiles) {
    return {};
  }
  const std::uint32_t local_n_tile =
      n_tile - partition.first_task_n_tile;
  return {partition_index,
          static_cast<std::uint8_t>(1U << partition_index),
          local_n_tile,
          k_tile,
          partition.payload_offset +
              (static_cast<std::uint64_t>(local_n_tile) *
                   partition.k_tiles +
               k_tile) *
                  partition.cell_bytes,
          partition.cell_bytes,
          true};
}

struct Sm87P40PackedFragment {
  std::uint8_t partition_index = 0xffU;
  std::uint32_t output_column = 0U;
  std::uint32_t k16 = 0U;
  std::uint64_t weight_offset = 0U;
  std::uint32_t weight_bytes = 0U;
  std::uint64_t scale_offset = 0U;
  std::uint32_t scale_bytes = 0U;
  bool valid = false;
};

// Returns the consumer-order fragment owned by one warp for one global K16.
// NVFP4 cells are [K16][warp][packed-weight][block-scale]. FP8 cells use the
// same order with a zero block-scale extent; tensor scales remain in the
// independent authenticated source identities.
[[nodiscard]] constexpr Sm87P40PackedFragment sm87_p40_packed_fragment(
    const Sm87P40PackedProjectionPlan& plan,
    const std::uint32_t n_tile, const std::uint32_t global_k16,
    const std::uint32_t warp) noexcept {
  if (!plan.valid() || n_tile >= plan.grid_n) {
    return {};
  }
  std::uint8_t partition_index =
      sm87_p40_packed_partition_for_n_tile(plan, n_tile);
  const Sm87P40PackedPartitionPlan* partition = nullptr;
  std::uint32_t local_warp = warp;
  if (plan.role == Sm87P40PackedProjectionRole::kNvFp4GateUp) {
    if (warp >= 8U) {
      return {};
    }
    partition_index = warp < 4U ? 0U : 1U;
    partition = &plan.partitions[partition_index];
    local_warp = warp % 4U;
  } else {
    if (partition_index >= plan.source_count) {
      return {};
    }
    partition = &plan.partitions[partition_index];
    if (warp >= partition->warps) {
      return {};
    }
  }
  const std::uint32_t total_k16 = partition->input_features / 16U;
  if (global_k16 >= total_k16) {
    return {};
  }
  const std::uint32_t k16_per_tile = partition->tile_k / 16U;
  const std::uint32_t k_tile = global_k16 / k16_per_tile;
  const std::uint32_t local_k16 = global_k16 % k16_per_tile;
  const auto cell = sm87_p40_packed_layout_cell(plan, n_tile, k_tile);
  if (!cell.valid) {
    return {};
  }
  const std::uint32_t physical_warps =
      plan.role == Sm87P40PackedProjectionRole::kNvFp4GateUp
          ? 8U
          : partition->warps;
  const std::uint32_t physical_warp =
      plan.role == Sm87P40PackedProjectionRole::kNvFp4GateUp
          ? warp
          : local_warp;
  const std::uint32_t fragment_bytes =
      partition->fragment_weight_bytes + partition->fragment_scale_bytes;
  const std::uint64_t fragment_offset =
      cell.payload_offset +
      (static_cast<std::uint64_t>(local_k16) * physical_warps +
       physical_warp) *
          fragment_bytes;
  const std::uint32_t local_n_tile =
      plan.role == Sm87P40PackedProjectionRole::kNvFp4GateUp
          ? n_tile
          : n_tile - partition->first_task_n_tile;
  return {partition_index,
          local_n_tile * partition->tile_n +
              local_warp * (partition->tile_n / partition->warps),
          global_k16,
          fragment_offset,
          partition->fragment_weight_bytes,
          fragment_offset + partition->fragment_weight_bytes,
          partition->fragment_scale_bytes,
          true};
}

struct Sm87P40PackedArtifactManifest {
  std::array<std::uint8_t, 8U> magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  std::uint32_t header_bytes = 0U;
  std::uint32_t layer_index = 0U;
  std::uint64_t artifact_identity = 0U;
  Sm87P40PackedProjectionRole role =
      Sm87P40PackedProjectionRole::kInvalid;
  Sm87P40PackedTactic tactic = Sm87P40PackedTactic::kInvalid;
  Sm87P40PackedProvider provider = Sm87P40PackedProvider::kInvalid;
  std::uint32_t policy = 0U;
  std::uint32_t source_count = 0U;
  std::uint64_t payload_offset = 0U;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t artifact_bytes = 0U;
  Sm87P40PackedDigest model_digest{};
  Sm87P40PackedDigest checkpoint_digest{};
  Sm87P40PackedDigest payload_digest{};
  std::array<Sm87P40PackedSourceIdentity,
             kSm87P40PackedProjectionMaximumSources>
      sources{};
  Sm87P40PackedDigest manifest_digest{};
};

enum class Sm87P40PackedManifestIssue : std::uint64_t {
  kNone = 0U,
  kMagic = 1ULL << 0U,
  kVersion = 1ULL << 1U,
  kHeader = 1ULL << 2U,
  kLayer = 1ULL << 3U,
  kRole = 1ULL << 4U,
  kTactic = 1ULL << 5U,
  kProvider = 1ULL << 6U,
  kPolicy = 1ULL << 7U,
  kSourceCount = 1ULL << 8U,
  kSourceRole = 1ULL << 9U,
  kSourceIdentity = 1ULL << 10U,
  kSourceDigest = 1ULL << 11U,
  kScale = 1ULL << 12U,
  kPayloadRange = 1ULL << 13U,
  kPayloadDigest = 1ULL << 14U,
  kModelDigest = 1ULL << 15U,
  kManifestDigest = 1ULL << 16U,
  kArtifactIdentity = 1ULL << 17U,
};

struct Sm87P40PackedManifestValidation {
  std::uint64_t issue_mask = 0U;
  [[nodiscard]] constexpr bool valid() const noexcept {
    return issue_mask == 0U;
  }
};

enum class Sm87P40PackedInventoryIssue : std::uint64_t {
  kNone = 0U,
  kNull = 1ULL << 0U,
  kCount = 1ULL << 1U,
  kManifest = 1ULL << 2U,
  kDuplicateArtifact = 1ULL << 3U,
  kDuplicateSource = 1ULL << 4U,
  kDuplicateRoleLayer = 1ULL << 5U,
  kMissingRoleLayer = 1ULL << 6U,
  kWrongLayerType = 1ULL << 7U,
  kModelMismatch = 1ULL << 8U,
  kCheckpointMismatch = 1ULL << 9U,
  kFp8LogicalCount = 1ULL << 10U,
  kSourceIdentityCount = 1ULL << 11U,
};

struct Sm87P40PackedInventoryValidation {
  std::uint64_t issue_mask = 0U;
  std::size_t artifact_count = 0U;
  std::size_t gate_up_artifacts = 0U;
  std::size_t down_artifacts = 0U;
  std::size_t fp8_artifacts = 0U;
  std::size_t fp8_logical_roles = 0U;
  std::size_t source_identities = 0U;
  [[nodiscard]] constexpr bool valid() const noexcept {
    return issue_mask == 0U &&
           artifact_count == kSm87P40PackedProjectionArtifactCount &&
           gate_up_artifacts == kSm87P40PackedProjectionLayerCount &&
           down_artifacts == kSm87P40PackedProjectionLayerCount &&
           fp8_artifacts == 2U * kSm87P40PackedProjectionLayerCount &&
           fp8_logical_roles ==
               kSm87P40PackedProjectionFp8LogicalRoleCount &&
           source_identities ==
               kSm87P40PackedProjectionSourceIdentityCount;
  }
};

[[nodiscard]] Sm87P40PackedArtifactManifest
make_sm87_p40_packed_artifact_manifest(
    Sm87P40PackedProjectionRole role,
    std::size_t layer_index) noexcept;

[[nodiscard]] bool compute_sm87_p40_packed_manifest_digest(
    const Sm87P40PackedArtifactManifest& manifest,
    Sm87P40PackedDigest* digest) noexcept;

[[nodiscard]] bool seal_sm87_p40_packed_artifact_manifest(
    Sm87P40PackedArtifactManifest* manifest) noexcept;

[[nodiscard]] Sm87P40PackedManifestValidation
validate_sm87_p40_packed_artifact_manifest(
    const Sm87P40PackedArtifactManifest& manifest) noexcept;

[[nodiscard]] Sm87P40PackedInventoryValidation
validate_sm87_p40_packed_artifact_inventory(
    const Sm87P40PackedArtifactManifest* manifests,
    std::size_t manifest_count) noexcept;

// Runtime-facing non-owning view. scalar_scales preserves one independent
// scalar per source partition; the artifact payload never embeds host/device
// addresses and remains reusable by an offline writer and a load-time mapper.
struct Sm87P40PackedProjectionDeviceView {
  const std::uint8_t* payload = nullptr;
  std::size_t payload_bytes = 0U;
  std::uint64_t artifact_identity = 0U;
  Sm87P40PackedProjectionRole role =
      Sm87P40PackedProjectionRole::kInvalid;
  Sm87P40PackedTactic tactic = Sm87P40PackedTactic::kInvalid;
  std::uint32_t source_count = 0U;
  std::array<float, kSm87P40PackedProjectionMaximumSources> scalar_scales{};
};

struct Sm87P40PackedProjectionResources {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

// Load-time/offline pack surface. sources are canonical checkpoint tensors;
// destination is exactly plan.payload_bytes. No request path may call it.
struct Sm87P40PackedCanonicalSource {
  Sm87P40PackedLogicalRole role = Sm87P40PackedLogicalRole::kInvalid;
  const std::uint8_t* weight = nullptr;
  const std::uint8_t* block_scale = nullptr;
  const float* global_scale_device = nullptr;
  std::size_t output_features = 0U;
  std::size_t input_features = 0U;
};

int prepare_sm87_p40_packed_projection_cuda(
    Sm87P40PackedProjectionRole role,
    const Sm87P40PackedCanonicalSource* sources,
    std::size_t source_count, std::uint8_t* destination,
    std::size_t destination_bytes, void* cuda_stream = nullptr) noexcept;

int query_sm87_p40_packed_projection_resources_cuda(
    Sm87P40PackedProjectionRole role,
    Sm87P40PackedProjectionResources* resources) noexcept;

int launch_sm87_p40_packed_nvfp4_gate_up_cuda(
    const std::uint16_t* input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    std::size_t token_count, std::uint16_t* activated_output,
    void* cuda_stream = nullptr) noexcept;

int launch_sm87_p40_packed_nvfp4_down_cuda(
    const std::uint16_t* input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    std::size_t token_count, std::uint16_t* residual_in_out,
    void* cuda_stream = nullptr) noexcept;

int launch_sm87_p40_packed_fp8_cuda(
    const std::uint16_t* input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    std::size_t token_count,
    const std::array<std::uint16_t*,
                     kSm87P40PackedProjectionMaximumSources>& outputs,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87P40PackedProjectionGridM *
                      kSm87P40PackedProjectionTileM ==
                  kSm87P40PackedProjectionTokens);
static_assert(kSm87P40PackedProjectionPersistentCtas ==
              2U * kSm87P40PackedProjectionSmCount);
static_assert(sm87_p40_packed_projection_plan(
                  Sm87P40PackedProjectionRole::kNvFp4GateUp)
                  .payload_bytes == 100'270'080ULL);
static_assert(sm87_p40_packed_projection_plan(
                  Sm87P40PackedProjectionRole::kNvFp4Down)
                  .payload_bytes == 50'135'040ULL);
static_assert(sm87_p40_packed_projection_plan(
                  Sm87P40PackedProjectionRole::kFp8LinearQkvZ)
                  .payload_bytes == 83'886'080ULL);
static_assert(sm87_p40_packed_projection_plan(
                  Sm87P40PackedProjectionRole::kFp8FullQkv)
                  .payload_bytes == 73'400'320ULL);
static_assert(sm87_p40_packed_projection_plan(
                  Sm87P40PackedProjectionRole::kFp8AttentionOutput)
                  .payload_bytes == 31'457'280ULL);

}  // namespace q3x::kernels
