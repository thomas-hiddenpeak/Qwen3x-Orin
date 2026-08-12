#include "q3x/runtime/sm87_target_aot_projection_assets.h"

#include "q3x/core/sha256.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

namespace kernels = q3x::kernels;

constexpr std::size_t kCellWeightValues =
    kernels::kSm87TargetAotProjectionBlockN *
    kernels::kSm87TargetAotProjectionBlockK;
constexpr std::size_t kCellScaleValues =
    kernels::kSm87TargetAotProjectionBlockN *
    (kernels::kSm87TargetAotProjectionBlockK / 16U);

[[nodiscard]] kernels::Sm87TargetAotProjectionSha256Digest copy_digest(
    const core::Sha256Digest& source) noexcept {
  kernels::Sm87TargetAotProjectionSha256Digest digest;
  digest.bytes = source.bytes;
  return digest;
}

[[nodiscard]] bool span_well_formed(
    const Sm87TargetAotProjectionConstBytes span) noexcept {
  if (span.bytes == 0U) {
    return true;
  }
  if (span.data == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(span.data);
  return span.bytes <=
         static_cast<std::size_t>(
             std::numeric_limits<std::uintptr_t>::max() - begin);
}

[[nodiscard]] bool span_well_formed(
    const Sm87TargetAotProjectionMutableBytes span) noexcept {
  return span.data != nullptr && span.bytes != 0U &&
         span.bytes <= static_cast<std::size_t>(
                           std::numeric_limits<std::uintptr_t>::max() -
                           reinterpret_cast<std::uintptr_t>(span.data));
}

struct AddressRange {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool empty = true;
};

[[nodiscard]] AddressRange address_range(
    const Sm87TargetAotProjectionConstBytes span) noexcept {
  if (span.bytes == 0U) {
    return {};
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(span.data);
  return {begin, begin + span.bytes, false};
}

[[nodiscard]] AddressRange address_range(
    const Sm87TargetAotProjectionMutableBytes span) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(span.data);
  return {begin, begin + span.bytes, false};
}

[[nodiscard]] bool ranges_overlap(const AddressRange& left,
                                  const AddressRange& right) noexcept {
  return !left.empty && !right.empty && left.begin < right.end &&
         right.begin < left.end;
}

[[nodiscard]] bool source_is_zero(
    const Sm87TargetAotProjectionSourceBytes& source) noexcept {
  return source.logical_role == kernels::Sm87TargetAotLogicalRole::kInvalid &&
         source.tensor_identity == 0U && source.output_features == 0U &&
         source.input_features == 0U && source.packed_weight.data == nullptr &&
         source.packed_weight.bytes == 0U &&
         source.block_scale.data == nullptr && source.block_scale.bytes == 0U &&
         source.tensor_scale_bits == 0U;
}

[[nodiscard]] bool u64_to_size(const std::uint64_t value,
                               std::size_t* const output) noexcept {
  if (output == nullptr ||
      value > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  *output = static_cast<std::size_t>(value);
  return true;
}

[[nodiscard]] bool scale_digest(
    const Sm87TargetAotProjectionConstBytes block_scale,
    const std::uint32_t tensor_scale_bits,
    kernels::Sm87TargetAotProjectionSha256Digest* const digest) noexcept {
  if (digest == nullptr) {
    return false;
  }
  core::Sha256 hasher;
  if (!hasher.update(block_scale.data, block_scale.bytes)) {
    return false;
  }
  const std::array<std::uint8_t, 4U> scale_bytes{{
      static_cast<std::uint8_t>(tensor_scale_bits),
      static_cast<std::uint8_t>(tensor_scale_bits >> 8U),
      static_cast<std::uint8_t>(tensor_scale_bits >> 16U),
      static_cast<std::uint8_t>(tensor_scale_bits >> 24U),
  }};
  if (!hasher.update(scale_bytes.data(), scale_bytes.size())) {
    return false;
  }
  *digest = copy_digest(hasher.finalize());
  return true;
}

struct InspectedSources {
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory{};
  std::array<std::uint64_t,
             kernels::kSm87TargetAotProjectionPackedMaxPartitions>
      weight_bytes{};
  std::array<std::uint64_t,
             kernels::kSm87TargetAotProjectionPackedMaxPartitions>
      block_scale_bytes{};
};

[[nodiscard]] Sm87TargetAotProjectionAssetError inspect_sources(
    const Sm87TargetAotProjectionSourceSet& sources,
    InspectedSources* const inspected) noexcept {
  if (inspected == nullptr || sources.inventory_identity == 0U) {
    return Sm87TargetAotProjectionAssetError::kInvalidArgument;
  }
  *inspected = {};
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(sources.role);
  if (!layout.valid()) {
    return Sm87TargetAotProjectionAssetError::kInvalidLayout;
  }
  if (sources.source_count != layout.partition_count ||
      sources.source_count > sources.sources.size()) {
    return Sm87TargetAotProjectionAssetError::kSourceCountMismatch;
  }

  std::array<AddressRange,
             2U * kernels::kSm87TargetAotProjectionPackedMaxPartitions>
      ranges{};
  std::size_t range_count = 0U;
  inspected->inventory.identity = sources.inventory_identity;
  inspected->inventory.role = sources.role;
  inspected->inventory.source_count = sources.source_count;

  for (std::size_t index = 0U; index < sources.sources.size(); ++index) {
    const auto& source = sources.sources[index];
    if (index >= sources.source_count) {
      if (!source_is_zero(source)) {
        return Sm87TargetAotProjectionAssetError::kSourceCountMismatch;
      }
      continue;
    }
    const auto& partition = layout.partitions[index];
    if (source.logical_role != partition.logical_role ||
        source.tensor_identity == 0U ||
        source.output_features != partition.output_features ||
        source.input_features != partition.input_features ||
        !kernels::sm87_target_aot_projection_scale_bits_valid(
            source.tensor_scale_bits)) {
      return Sm87TargetAotProjectionAssetError::kSourceMetadataMismatch;
    }
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (source.tensor_identity == sources.sources[prior].tensor_identity) {
        return Sm87TargetAotProjectionAssetError::kSourceMetadataMismatch;
      }
    }
    if (!span_well_formed(source.packed_weight) ||
        !span_well_formed(source.block_scale)) {
      return Sm87TargetAotProjectionAssetError::kSourceRangeOverflow;
    }

    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    const std::uint64_t expected_weight_bytes =
        values * partition.weight_bits / 8U;
    const std::uint64_t expected_scale_bytes =
        partition.block_scale_group_k == 0U
            ? 0U
            : values / partition.block_scale_group_k;
    std::size_t weight_bytes = 0U;
    std::size_t block_scale_bytes = 0U;
    if (!u64_to_size(expected_weight_bytes, &weight_bytes) ||
        !u64_to_size(expected_scale_bytes, &block_scale_bytes)) {
      return Sm87TargetAotProjectionAssetError::kSizeOverflow;
    }
    if (source.packed_weight.bytes != weight_bytes ||
        source.block_scale.bytes != block_scale_bytes ||
        source.packed_weight.data == nullptr ||
        (block_scale_bytes != 0U && source.block_scale.data == nullptr) ||
        (block_scale_bytes == 0U && source.block_scale.data != nullptr)) {
      return Sm87TargetAotProjectionAssetError::kSourceSizeMismatch;
    }
    inspected->weight_bytes[index] = expected_weight_bytes;
    inspected->block_scale_bytes[index] = expected_scale_bytes;

    ranges[range_count++] = address_range(source.packed_weight);
    if (source.block_scale.bytes != 0U) {
      ranges[range_count++] = address_range(source.block_scale);
    }
    for (std::size_t right = 0U; right < range_count; ++right) {
      for (std::size_t left = 0U; left < right; ++left) {
        if (ranges_overlap(ranges[left], ranges[right])) {
          return Sm87TargetAotProjectionAssetError::kSourceAliasing;
        }
      }
    }

    for (std::size_t scale = 0U; scale < source.block_scale.bytes; ++scale) {
      if (kernels::
              sm87_target_aot_projection_block_scale_e4m3fn_code_is_forbidden(
                  source.block_scale.data[scale])) {
        return Sm87TargetAotProjectionAssetError::
            kForbiddenNvFp4BlockScale;
      }
    }

    kernels::Sm87TargetAotProjectionSha256Digest weight_digest{};
    if (!sm87_target_aot_projection_sha256(source.packed_weight,
                                            &weight_digest)) {
      return Sm87TargetAotProjectionAssetError::kInvalidArgument;
    }
    kernels::Sm87TargetAotProjectionSha256Digest scales{};
    if (!scale_digest(source.block_scale, source.tensor_scale_bits, &scales)) {
      return Sm87TargetAotProjectionAssetError::kInvalidArgument;
    }
    inspected->inventory.sources[index] =
        kernels::sm87_target_aot_projection_packed_source_binding(
            layout, index, source.tensor_identity, weight_digest, scales,
            source.tensor_scale_bits);
  }
  if (!inspected->inventory.valid(layout)) {
    return Sm87TargetAotProjectionAssetError::kSourceMetadataMismatch;
  }
  return Sm87TargetAotProjectionAssetError::kSuccess;
}

[[nodiscard]] Sm87TargetAotProjectionAssetError compare_inventory(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& observed,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& expected)
    noexcept {
  if (!expected.valid(layout) || expected.identity != observed.identity ||
      expected.role != observed.role ||
      expected.source_count != observed.source_count) {
    return Sm87TargetAotProjectionAssetError::kSourceMetadataMismatch;
  }
  for (std::size_t index = 0U; index < expected.sources.size(); ++index) {
    const auto& left = observed.sources[index];
    const auto& right = expected.sources[index];
    if (left.weight_digest != right.weight_digest ||
        left.scale_digest != right.scale_digest) {
      return Sm87TargetAotProjectionAssetError::kSourceDigestMismatch;
    }
    auto without_digests = left;
    without_digests.weight_digest = right.weight_digest;
    without_digests.scale_digest = right.scale_digest;
    if (!kernels::sm87_target_aot_projection_same_source_binding(
            without_digests, right)) {
      return Sm87TargetAotProjectionAssetError::kSourceMetadataMismatch;
    }
  }
  return Sm87TargetAotProjectionAssetError::kSuccess;
}

struct LocalWeightTarget {
  std::uint32_t byte_offset = 0U;
  std::uint8_t nibble = 0U;
};

struct LocalMaps {
  std::array<LocalWeightTarget, kCellWeightValues> weights{};
  std::array<std::uint32_t, kCellScaleValues> scales{};
};

[[nodiscard]] bool build_local_maps(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const std::size_t partition_index, LocalMaps* const maps) noexcept {
  if (maps == nullptr || partition_index >= layout.partition_count) {
    return false;
  }
  *maps = {};
  const auto& partition = layout.partitions[partition_index];
  const auto cell = kernels::sm87_target_aot_projection_packed_cell(
      layout, partition_index, 0U, 0U);
  if (!cell.valid) {
    return false;
  }
  std::array<std::uint8_t, kCellWeightValues> visited_weights{};
  for (std::size_t n = 0U;
       n < kernels::kSm87TargetAotProjectionBlockN; ++n) {
    for (std::size_t k = 0U;
         k < kernels::kSm87TargetAotProjectionBlockK; ++k) {
      const auto address =
          kernels::sm87_target_aot_projection_packed_weight_address(
              layout, partition_index, n, k);
      if (!address.valid || address.byte_offset < cell.weight_offset) {
        return false;
      }
      const std::uint64_t delta = address.byte_offset - cell.weight_offset;
      const std::uint64_t slot =
          partition.weight_bits == 4U ? delta * 2U + address.nibble : delta;
      if (delta >= cell.weight_bytes || slot >= visited_weights.size() ||
          visited_weights[static_cast<std::size_t>(slot)] != 0U) {
        return false;
      }
      visited_weights[static_cast<std::size_t>(slot)] = 1U;
      maps->weights[n * kernels::kSm87TargetAotProjectionBlockK + k] = {
          static_cast<std::uint32_t>(delta), address.nibble};
    }
  }
  for (const std::uint8_t visited : visited_weights) {
    if (visited != 1U) {
      return false;
    }
  }

  if (partition.block_scale_group_k == 0U) {
    return cell.block_scale_bytes == 0U;
  }
  std::array<std::uint8_t, kCellScaleValues> visited_scales{};
  for (std::size_t n = 0U;
       n < kernels::kSm87TargetAotProjectionBlockN; ++n) {
    for (std::size_t group = 0U;
         group < kernels::kSm87TargetAotProjectionBlockK / 16U; ++group) {
      const auto address =
          kernels::sm87_target_aot_projection_packed_scale_address(
              layout, partition_index, n, group);
      if (!address.valid ||
          address.byte_offset < cell.block_scale_offset) {
        return false;
      }
      const std::uint64_t delta =
          address.byte_offset - cell.block_scale_offset;
      if (delta >= cell.block_scale_bytes ||
          delta >= visited_scales.size() ||
          visited_scales[static_cast<std::size_t>(delta)] != 0U) {
        return false;
      }
      visited_scales[static_cast<std::size_t>(delta)] = 1U;
      maps->scales[n *
                       (kernels::kSm87TargetAotProjectionBlockK / 16U) +
                   group] = static_cast<std::uint32_t>(delta);
    }
  }
  for (const std::uint8_t visited : visited_scales) {
    if (visited != 1U) {
      return false;
    }
  }
  return true;
}

void write_nibble(std::uint8_t* const destination,
                  const std::uint8_t nibble_index,
                  const std::uint8_t value) noexcept {
  const unsigned int shift = 4U * nibble_index;
  const std::uint8_t mask = static_cast<std::uint8_t>(0x0fU << shift);
  *destination = static_cast<std::uint8_t>(
      (*destination & static_cast<std::uint8_t>(~mask)) |
      static_cast<std::uint8_t>((value & 0x0fU) << shift));
}

[[nodiscard]] std::uint8_t read_nibble(const std::uint8_t value,
                                       const std::uint8_t nibble_index)
    noexcept {
  return static_cast<std::uint8_t>((value >> (4U * nibble_index)) & 0x0fU);
}

[[nodiscard]] Sm87TargetAotProjectionAssetError transform_payload(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const Sm87TargetAotProjectionSourceSet& sources,
    const Sm87TargetAotProjectionMutableBytes payload) noexcept {
  std::fill_n(payload.data, payload.bytes, std::uint8_t{0U});
  for (std::size_t partition_index = 0U;
       partition_index < layout.partition_count; ++partition_index) {
    const auto& partition = layout.partitions[partition_index];
    const auto& source = sources.sources[partition_index];
    LocalMaps maps;
    if (!build_local_maps(layout, partition_index, &maps)) {
      return Sm87TargetAotProjectionAssetError::kPayloadBijectionMismatch;
    }
    const std::size_t n_tiles = partition.n_tiles;
    const std::size_t k_tiles = partition.k_tiles;
    for (std::size_t n_tile = 0U; n_tile < n_tiles; ++n_tile) {
      for (std::size_t k_tile = 0U; k_tile < k_tiles; ++k_tile) {
        const auto cell = kernels::sm87_target_aot_projection_packed_cell(
            layout, partition_index, n_tile, k_tile);
        if (!cell.valid || cell.payload_offset + cell.cell_bytes >
                               static_cast<std::uint64_t>(payload.bytes)) {
          return Sm87TargetAotProjectionAssetError::kPayloadSizeMismatch;
        }
        for (std::size_t local_n = 0U;
             local_n < kernels::kSm87TargetAotProjectionBlockN; ++local_n) {
          const std::size_t global_n =
              n_tile * kernels::kSm87TargetAotProjectionBlockN + local_n;
          if (partition.weight_bits == 4U) {
            const std::size_t source_row =
                global_n * (partition.input_features / 2U);
            for (std::size_t local_k = 0U;
                 local_k < kernels::kSm87TargetAotProjectionBlockK;
                 local_k += 2U) {
              const std::size_t global_k =
                  k_tile * kernels::kSm87TargetAotProjectionBlockK + local_k;
              const std::uint8_t packed =
                  source.packed_weight.data[source_row + global_k / 2U];
              for (std::size_t pair = 0U; pair < 2U; ++pair) {
                const auto& target =
                    maps.weights[local_n *
                                     kernels::kSm87TargetAotProjectionBlockK +
                                 local_k + pair];
                write_nibble(payload.data + cell.weight_offset +
                                 target.byte_offset,
                             target.nibble,
                             static_cast<std::uint8_t>(packed >> (4U * pair)));
              }
            }
          } else {
            const std::size_t source_row =
                global_n * partition.input_features;
            for (std::size_t local_k = 0U;
                 local_k < kernels::kSm87TargetAotProjectionBlockK;
                 ++local_k) {
              const std::size_t global_k =
                  k_tile * kernels::kSm87TargetAotProjectionBlockK + local_k;
              const auto& target =
                  maps.weights[local_n *
                                   kernels::kSm87TargetAotProjectionBlockK +
                               local_k];
              payload.data[cell.weight_offset + target.byte_offset] =
                  source.packed_weight.data[source_row + global_k];
            }
          }
        }
        if (partition.block_scale_group_k != 0U) {
          for (std::size_t local_n = 0U;
               local_n < kernels::kSm87TargetAotProjectionBlockN; ++local_n) {
            const std::size_t global_n =
                n_tile * kernels::kSm87TargetAotProjectionBlockN + local_n;
            const std::size_t source_row =
                global_n *
                (partition.input_features / partition.block_scale_group_k);
            for (std::size_t group = 0U;
                 group < kernels::kSm87TargetAotProjectionBlockK / 16U;
                 ++group) {
              const std::size_t global_group =
                  k_tile * (kernels::kSm87TargetAotProjectionBlockK / 16U) +
                  group;
              payload.data[cell.block_scale_offset +
                           maps.scales[local_n *
                                           (kernels::
                                                kSm87TargetAotProjectionBlockK /
                                            16U) +
                                       group]] =
                  source.block_scale.data[source_row + global_group];
            }
          }
        }
      }
    }
  }
  return Sm87TargetAotProjectionAssetError::kSuccess;
}

[[nodiscard]] Sm87TargetAotProjectionAssetError scan_payload_scales(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const Sm87TargetAotProjectionConstBytes payload) noexcept {
  for (std::size_t partition_index = 0U;
       partition_index < layout.partition_count; ++partition_index) {
    const auto& partition = layout.partitions[partition_index];
    if (partition.block_scale_group_k == 0U) {
      continue;
    }
    for (std::size_t n_tile = 0U; n_tile < partition.n_tiles; ++n_tile) {
      for (std::size_t k_tile = 0U; k_tile < partition.k_tiles; ++k_tile) {
        const auto cell = kernels::sm87_target_aot_projection_packed_cell(
            layout, partition_index, n_tile, k_tile);
        if (!cell.valid ||
            cell.block_scale_offset + cell.block_scale_bytes >
                static_cast<std::uint64_t>(payload.bytes)) {
          return Sm87TargetAotProjectionAssetError::kPayloadSizeMismatch;
        }
        for (std::size_t byte = 0U; byte < cell.block_scale_bytes; ++byte) {
          if (kernels::
                  sm87_target_aot_projection_block_scale_e4m3fn_code_is_forbidden(
                      payload.data[cell.block_scale_offset + byte])) {
            return Sm87TargetAotProjectionAssetError::
                kForbiddenNvFp4BlockScale;
          }
        }
      }
    }
  }
  return Sm87TargetAotProjectionAssetError::kSuccess;
}

[[nodiscard]] Sm87TargetAotProjectionAssetError verify_payload_bijection(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const Sm87TargetAotProjectionSourceSet& sources,
    const Sm87TargetAotProjectionConstBytes payload) noexcept {
  for (std::size_t partition_index = 0U;
       partition_index < layout.partition_count; ++partition_index) {
    const auto& partition = layout.partitions[partition_index];
    const auto& source = sources.sources[partition_index];
    LocalMaps maps;
    if (!build_local_maps(layout, partition_index, &maps)) {
      return Sm87TargetAotProjectionAssetError::kPayloadBijectionMismatch;
    }
    for (std::size_t n_tile = 0U; n_tile < partition.n_tiles; ++n_tile) {
      for (std::size_t k_tile = 0U; k_tile < partition.k_tiles; ++k_tile) {
        const auto cell = kernels::sm87_target_aot_projection_packed_cell(
            layout, partition_index, n_tile, k_tile);
        if (!cell.valid || cell.payload_offset + cell.cell_bytes >
                               static_cast<std::uint64_t>(payload.bytes)) {
          return Sm87TargetAotProjectionAssetError::kPayloadSizeMismatch;
        }
        for (std::size_t local_n = 0U;
             local_n < kernels::kSm87TargetAotProjectionBlockN; ++local_n) {
          const std::size_t global_n =
              n_tile * kernels::kSm87TargetAotProjectionBlockN + local_n;
          if (partition.weight_bits == 4U) {
            const std::size_t source_row =
                global_n * (partition.input_features / 2U);
            for (std::size_t local_k = 0U;
                 local_k < kernels::kSm87TargetAotProjectionBlockK;
                 ++local_k) {
              const std::size_t global_k =
                  k_tile * kernels::kSm87TargetAotProjectionBlockK + local_k;
              const std::uint8_t expected = read_nibble(
                  source.packed_weight.data[source_row + global_k / 2U],
                  static_cast<std::uint8_t>(global_k & 1U));
              const auto& target =
                  maps.weights[local_n *
                                   kernels::kSm87TargetAotProjectionBlockK +
                               local_k];
              const std::uint8_t observed = read_nibble(
                  payload.data[cell.weight_offset + target.byte_offset],
                  target.nibble);
              if (expected != observed) {
                return Sm87TargetAotProjectionAssetError::
                    kPayloadBijectionMismatch;
              }
            }
          } else {
            const std::size_t source_row =
                global_n * partition.input_features;
            for (std::size_t local_k = 0U;
                 local_k < kernels::kSm87TargetAotProjectionBlockK;
                 ++local_k) {
              const std::size_t global_k =
                  k_tile * kernels::kSm87TargetAotProjectionBlockK + local_k;
              const auto& target =
                  maps.weights[local_n *
                                   kernels::kSm87TargetAotProjectionBlockK +
                               local_k];
              if (source.packed_weight.data[source_row + global_k] !=
                  payload.data[cell.weight_offset + target.byte_offset]) {
                return Sm87TargetAotProjectionAssetError::
                    kPayloadBijectionMismatch;
              }
            }
          }
        }
        if (partition.block_scale_group_k != 0U) {
          for (std::size_t local_n = 0U;
               local_n < kernels::kSm87TargetAotProjectionBlockN; ++local_n) {
            const std::size_t global_n =
                n_tile * kernels::kSm87TargetAotProjectionBlockN + local_n;
            const std::size_t source_row =
                global_n *
                (partition.input_features / partition.block_scale_group_k);
            for (std::size_t group = 0U;
                 group < kernels::kSm87TargetAotProjectionBlockK / 16U;
                 ++group) {
              const std::size_t global_group =
                  k_tile * (kernels::kSm87TargetAotProjectionBlockK / 16U) +
                  group;
              if (source.block_scale.data[source_row + global_group] !=
                  payload.data[cell.block_scale_offset +
                               maps.scales[local_n *
                                               (kernels::
                                                    kSm87TargetAotProjectionBlockK /
                                                16U) +
                                           group]]) {
                return Sm87TargetAotProjectionAssetError::
                    kPayloadBijectionMismatch;
              }
            }
          }
        }
      }
    }
  }
  return Sm87TargetAotProjectionAssetError::kSuccess;
}

[[nodiscard]] bool payload_aliases_sources(
    const Sm87TargetAotProjectionSourceSet& sources,
    const AddressRange& payload) noexcept {
  for (std::size_t index = 0U; index < sources.source_count; ++index) {
    if (ranges_overlap(payload, address_range(sources.sources[index]
                                                  .packed_weight)) ||
        ranges_overlap(payload,
                       address_range(sources.sources[index].block_scale))) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedTransformReceipt
make_transform_receipt(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& inventory)
    noexcept {
  kernels::Sm87TargetAotProjectionPackedTransformReceipt receipt;
  receipt.artifact_identity = manifest.artifact_identity;
  receipt.source_inventory_identity = inventory.identity;
  receipt.role = layout.role;
  receipt.plan_identity = layout.plan_identity;
  receipt.layout_identity = layout.layout_identity;
  receipt.encoding = layout.encoding;
  receipt.transform_identity =
      kernels::Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.partition_count = layout.partition_count;
  receipt.deterministic_transform = true;
  receipt.no_arithmetic_conversion = true;
  receipt.no_request_time_repacking = true;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    const auto& partition = layout.partitions[index];
    const auto& source = inventory.sources[index];
    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    const std::uint64_t weight_bytes =
        values * partition.weight_bits / 8U;
    const std::uint64_t scale_values =
        partition.block_scale_group_k == 0U
            ? 0U
            : values / partition.block_scale_group_k;
    receipt.partitions[index] = {
        partition.logical_role,
        static_cast<std::uint32_t>(index),
        source.tensor_identity,
        source.weight_digest,
        source.scale_digest,
        weight_bytes,
        scale_values + sizeof(std::uint32_t),
        values,
        scale_values,
        scale_values,
        scale_values,
        0U,
        0U,
        partition.payload_offset,
        partition.payload_bytes,
        true,
        true,
        true,
        scale_values != 0U,
        true,
    };
  }
  receipt.payload.artifact_identity = manifest.artifact_identity;
  receipt.payload.observed_payload_offset = manifest.payload_offset;
  receipt.payload.observed_payload_bytes = manifest.payload_bytes;
  receipt.payload.observed_payload_digest = manifest.payload_digest;
  receipt.payload.digest_computed_from_payload_bytes = true;
  return receipt;
}

}  // namespace

const char* sm87_target_aot_projection_asset_error_string(
    const Sm87TargetAotProjectionAssetError error) noexcept {
  switch (error) {
    case Sm87TargetAotProjectionAssetError::kSuccess:
      return "success";
    case Sm87TargetAotProjectionAssetError::kInvalidArgument:
      return "invalid argument";
    case Sm87TargetAotProjectionAssetError::kInvalidLayout:
      return "invalid target-AOT projection layout";
    case Sm87TargetAotProjectionAssetError::kSizeOverflow:
      return "asset size overflow";
    case Sm87TargetAotProjectionAssetError::kSourceCountMismatch:
      return "source count mismatch";
    case Sm87TargetAotProjectionAssetError::kSourceMetadataMismatch:
      return "source metadata mismatch";
    case Sm87TargetAotProjectionAssetError::kSourceSizeMismatch:
      return "source byte size mismatch";
    case Sm87TargetAotProjectionAssetError::kSourceRangeOverflow:
      return "source byte range overflow";
    case Sm87TargetAotProjectionAssetError::kSourceAliasing:
      return "source byte ranges alias";
    case Sm87TargetAotProjectionAssetError::kSourceDigestMismatch:
      return "source digest mismatch";
    case Sm87TargetAotProjectionAssetError::kForbiddenNvFp4BlockScale:
      return "forbidden NVFP4 block-scale encoding";
    case Sm87TargetAotProjectionAssetError::kPayloadSizeMismatch:
      return "payload byte size mismatch";
    case Sm87TargetAotProjectionAssetError::kPayloadRangeOverflow:
      return "payload byte range overflow";
    case Sm87TargetAotProjectionAssetError::kPayloadAliasing:
      return "payload aliases source bytes";
    case Sm87TargetAotProjectionAssetError::kPayloadDigestMismatch:
      return "payload digest mismatch";
    case Sm87TargetAotProjectionAssetError::kPayloadBijectionMismatch:
      return "payload bijection mismatch";
    case Sm87TargetAotProjectionAssetError::kManifestMismatch:
      return "packed manifest mismatch";
    case Sm87TargetAotProjectionAssetError::kTransformReceiptMismatch:
      return "transform receipt mismatch";
  }
  return "unknown target-AOT projection asset error";
}

bool sm87_target_aot_projection_sha256(
    const Sm87TargetAotProjectionConstBytes bytes,
    kernels::Sm87TargetAotProjectionSha256Digest* const digest) noexcept {
  if (digest == nullptr || !span_well_formed(bytes)) {
    return false;
  }
  core::Sha256 hasher;
  if (!hasher.update(bytes.data, bytes.bytes)) {
    return false;
  }
  *digest = copy_digest(hasher.finalize());
  return true;
}

Sm87TargetAotProjectionSourceInspection
sm87_target_aot_projection_inspect_sources(
    const Sm87TargetAotProjectionSourceSet& sources) noexcept {
  Sm87TargetAotProjectionSourceInspection result;
  InspectedSources inspected;
  result.error = inspect_sources(sources, &inspected);
  if (result.error == Sm87TargetAotProjectionAssetError::kSuccess) {
    result.inventory = inspected.inventory;
  }
  return result;
}

Sm87TargetAotProjectionAssetBuildResult
sm87_target_aot_projection_build_asset(
    const std::uint64_t artifact_identity,
    const Sm87TargetAotProjectionSourceSet& sources,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory&
        expected_inventory,
    const Sm87TargetAotProjectionMutableBytes payload) noexcept {
  Sm87TargetAotProjectionAssetBuildResult result;
  if (artifact_identity == 0U) {
    result.error = Sm87TargetAotProjectionAssetError::kInvalidArgument;
    return result;
  }
  InspectedSources inspected;
  result.error = inspect_sources(sources, &inspected);
  if (result.error != Sm87TargetAotProjectionAssetError::kSuccess) {
    return result;
  }
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(sources.role);
  result.error = compare_inventory(layout, inspected.inventory,
                                   expected_inventory);
  if (result.error != Sm87TargetAotProjectionAssetError::kSuccess) {
    return result;
  }
  if (!span_well_formed(payload)) {
    result.error = payload.data == nullptr || payload.bytes == 0U
                       ? Sm87TargetAotProjectionAssetError::kPayloadSizeMismatch
                       : Sm87TargetAotProjectionAssetError::
                             kPayloadRangeOverflow;
    return result;
  }
  std::size_t expected_payload_bytes = 0U;
  if (!u64_to_size(layout.payload_bytes, &expected_payload_bytes)) {
    result.error = Sm87TargetAotProjectionAssetError::kSizeOverflow;
    return result;
  }
  if (payload.bytes != expected_payload_bytes) {
    result.error = Sm87TargetAotProjectionAssetError::kPayloadSizeMismatch;
    return result;
  }
  if (payload_aliases_sources(sources, address_range(payload))) {
    result.error = Sm87TargetAotProjectionAssetError::kPayloadAliasing;
    return result;
  }
  result.error = transform_payload(layout, sources, payload);
  if (result.error != Sm87TargetAotProjectionAssetError::kSuccess) {
    std::fill_n(payload.data, payload.bytes, std::uint8_t{0U});
    return result;
  }
  const Sm87TargetAotProjectionConstBytes payload_view{payload.data,
                                                       payload.bytes};
  result.error = scan_payload_scales(layout, payload_view);
  if (result.error == Sm87TargetAotProjectionAssetError::kSuccess) {
    result.error = verify_payload_bijection(layout, sources, payload_view);
  }
  if (result.error != Sm87TargetAotProjectionAssetError::kSuccess) {
    std::fill_n(payload.data, payload.bytes, std::uint8_t{0U});
    return result;
  }
  kernels::Sm87TargetAotProjectionSha256Digest digest{};
  if (!sm87_target_aot_projection_sha256(payload_view, &digest)) {
    result.error = Sm87TargetAotProjectionAssetError::kInvalidArgument;
    std::fill_n(payload.data, payload.bytes, std::uint8_t{0U});
    return result;
  }
  result.manifest = kernels::sm87_target_aot_projection_make_packed_manifest(
      sources.role, artifact_identity, expected_inventory, digest);
  if (!kernels::sm87_target_aot_projection_validate_packed_manifest(
          result.manifest, expected_inventory)) {
    std::fill_n(payload.data, payload.bytes, std::uint8_t{0U});
    result = {};
    result.error = Sm87TargetAotProjectionAssetError::kManifestMismatch;
    return result;
  }
  result.transform_receipt =
      make_transform_receipt(layout, result.manifest, expected_inventory);
  if (!kernels::sm87_target_aot_projection_validate_transform_receipt(
          result.manifest, expected_inventory, result.transform_receipt)) {
    std::fill_n(payload.data, payload.bytes, std::uint8_t{0U});
    result = {};
    result.error =
        Sm87TargetAotProjectionAssetError::kTransformReceiptMismatch;
    return result;
  }
  result.error = Sm87TargetAotProjectionAssetError::kSuccess;
  return result;
}

Sm87TargetAotProjectionAssetError
sm87_target_aot_projection_validate_asset(
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const kernels::Sm87TargetAotProjectionPackedTransformReceipt& receipt,
    const Sm87TargetAotProjectionSourceSet& sources,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory&
        expected_inventory,
    const Sm87TargetAotProjectionConstBytes payload) noexcept {
  InspectedSources inspected;
  Sm87TargetAotProjectionAssetError error =
      inspect_sources(sources, &inspected);
  if (error != Sm87TargetAotProjectionAssetError::kSuccess) {
    return error;
  }
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(sources.role);
  error = compare_inventory(layout, inspected.inventory, expected_inventory);
  if (error != Sm87TargetAotProjectionAssetError::kSuccess) {
    return error;
  }
  if (!kernels::sm87_target_aot_projection_validate_packed_manifest(
          manifest, expected_inventory)) {
    return Sm87TargetAotProjectionAssetError::kManifestMismatch;
  }
  if (!kernels::sm87_target_aot_projection_validate_transform_receipt(
          manifest, expected_inventory, receipt)) {
    return Sm87TargetAotProjectionAssetError::kTransformReceiptMismatch;
  }
  if (!span_well_formed(payload)) {
    return payload.data == nullptr || payload.bytes == 0U
               ? Sm87TargetAotProjectionAssetError::kPayloadSizeMismatch
               : Sm87TargetAotProjectionAssetError::kPayloadRangeOverflow;
  }
  std::size_t expected_payload_bytes = 0U;
  if (!u64_to_size(layout.payload_bytes, &expected_payload_bytes)) {
    return Sm87TargetAotProjectionAssetError::kSizeOverflow;
  }
  if (payload.bytes != expected_payload_bytes) {
    return Sm87TargetAotProjectionAssetError::kPayloadSizeMismatch;
  }
  if (payload_aliases_sources(sources, address_range(payload))) {
    return Sm87TargetAotProjectionAssetError::kPayloadAliasing;
  }
  kernels::Sm87TargetAotProjectionSha256Digest digest{};
  if (!sm87_target_aot_projection_sha256(payload, &digest)) {
    return Sm87TargetAotProjectionAssetError::kInvalidArgument;
  }
  if (digest != manifest.payload_digest ||
      digest != receipt.payload.observed_payload_digest) {
    return Sm87TargetAotProjectionAssetError::kPayloadDigestMismatch;
  }
  error = scan_payload_scales(layout, payload);
  if (error != Sm87TargetAotProjectionAssetError::kSuccess) {
    return error;
  }
  return verify_payload_bijection(layout, sources, payload);
}

}  // namespace q3x::runtime
