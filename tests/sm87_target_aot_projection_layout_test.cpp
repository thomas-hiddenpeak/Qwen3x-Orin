#include "q3x/kernels/sm87_target_aot_projection_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>

namespace {

namespace kernels = q3x::kernels;

using kernels::Sm87TargetAotProjectionPackedLayout;
using kernels::Sm87TargetAotProjectionPackedRegion;
using kernels::Sm87TargetAotProjectionPackedSourceInventory;
using kernels::Sm87TargetAotProjectionRole;
using kernels::Sm87TargetAotProjectionSha256Digest;

constexpr std::array<Sm87TargetAotProjectionRole, 5U> kRoles{{
    Sm87TargetAotProjectionRole::kNvFp4GateUp,
    Sm87TargetAotProjectionRole::kNvFp4Down,
    Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
    Sm87TargetAotProjectionRole::kFp8FullQkv,
    Sm87TargetAotProjectionRole::kFp8AttentionOutput,
}};

constexpr auto kGate = kernels::kSm87TargetAotNvFp4GateUpPackedLayout;
constexpr auto kDown = kernels::kSm87TargetAotNvFp4DownPackedLayout;
constexpr auto kGdn = kernels::kSm87TargetAotFp8GdnQkvZPackedLayout;
constexpr auto kFull = kernels::kSm87TargetAotFp8FullQkvPackedLayout;
constexpr auto kOutput =
    kernels::kSm87TargetAotFp8AttentionOutputPackedLayout;

static_assert(kGate.valid() && kDown.valid() && kGdn.valid() &&
              kFull.valid() && kOutput.valid());
static_assert(kGate.partition_count == 2U &&
              kGate.partitions[0U].output_features == 17'408U &&
              kGate.partitions[1U].output_features == 17'408U &&
              kGate.partitions[1U].payload_offset == 50'135'040U);
static_assert(kDown.partition_count == 1U &&
              kDown.partitions[0U].n_tiles == 20U &&
              kDown.partitions[0U].k_tiles == 272U);
static_assert(kGdn.partition_count == 2U &&
              kGdn.partitions[0U].n_tiles == 40U &&
              kGdn.partitions[1U].n_tiles == 24U);
static_assert(kFull.partition_count == 3U &&
              kFull.partitions[0U].n_tiles == 48U &&
              kFull.partitions[1U].n_tiles == 4U &&
              kFull.partitions[2U].n_tiles == 4U);
static_assert(kOutput.partition_count == 1U &&
              kOutput.partitions[0U].k_tiles == 96U);
static_assert(kGate.block_m == 128U && kGate.block_n == 256U &&
              kGate.block_k == 64U && kGate.cta_threads == 256U &&
              kGate.m_warps * kGate.n_warps == 8U &&
              kGate.pipeline_stages == 3U);
static_assert(kernels::sm87_target_aot_projection_packed_k16_for_lane_component(
                  0U, 0U) == 0U &&
              kernels::sm87_target_aot_projection_packed_k16_for_lane_component(
                  3U, 2U) == 7U &&
              kernels::sm87_target_aot_projection_packed_k16_for_lane_component(
                  0U, 1U) == 8U &&
              kernels::sm87_target_aot_projection_packed_k16_for_lane_component(
                  3U, 3U) == 15U);
static_assert(
    kernels::sm87_target_aot_projection_mma_b_register_component(0U) == 0U &&
    kernels::sm87_target_aot_projection_mma_b_register_component(1U) == 2U &&
    kernels::sm87_target_aot_projection_mma_b_register_component(2U) == 1U &&
    kernels::sm87_target_aot_projection_mma_b_register_component(3U) == 3U &&
    kernels::sm87_target_aot_projection_mma_b_register_component(4U) ==
        0xffU &&
    kernels::sm87_target_aot_projection_packed_k16_for_lane_component(
        2U,
        kernels::sm87_target_aot_projection_mma_b_register_component(0U)) ==
        4U &&
    kernels::sm87_target_aot_projection_packed_k16_for_lane_component(
        2U,
        kernels::sm87_target_aot_projection_mma_b_register_component(1U)) ==
        5U &&
    kernels::sm87_target_aot_projection_packed_k16_for_lane_component(
        2U,
        kernels::sm87_target_aot_projection_mma_b_register_component(2U)) ==
        12U &&
    kernels::sm87_target_aot_projection_packed_k16_for_lane_component(
        2U,
        kernels::sm87_target_aot_projection_mma_b_register_component(3U)) ==
        13U);
static_assert(
    kernels::sm87_target_aot_projection_nvfp4_physical_warp(
        Sm87TargetAotProjectionRole::kNvFp4GateUp, 0U, 0U, 0U)
            .physical_warp == 0U &&
    kernels::sm87_target_aot_projection_nvfp4_physical_warp(
        Sm87TargetAotProjectionRole::kNvFp4GateUp, 0U, 0U, 2U)
            .physical_warp == 0U &&
    kernels::sm87_target_aot_projection_nvfp4_physical_warp(
        Sm87TargetAotProjectionRole::kNvFp4GateUp, 1U, 1U, 3U)
            .physical_warp == 7U &&
    kernels::sm87_target_aot_projection_nvfp4_physical_warp(
        Sm87TargetAotProjectionRole::kNvFp4Down, 0U, 1U, 3U)
            .physical_warp == 7U &&
    !kernels::sm87_target_aot_projection_nvfp4_physical_warp(
         Sm87TargetAotProjectionRole::kFp8FullQkv, 0U, 0U, 0U)
         .valid);

class TestContext {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

[[nodiscard]] Sm87TargetAotProjectionSha256Digest make_digest(
    const std::uint64_t seed) noexcept {
  Sm87TargetAotProjectionSha256Digest digest;
  std::uint64_t state = seed | 1U;
  for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    digest.bytes[index] = static_cast<std::uint8_t>(state >> 24U);
  }
  if (kernels::sm87_target_aot_projection_digest_is_zero(digest)) {
    digest.bytes[0U] = 1U;
  }
  return digest;
}

[[nodiscard]] Sm87TargetAotProjectionPackedSourceInventory make_inventory(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::uint64_t seed) noexcept {
  Sm87TargetAotProjectionPackedSourceInventory inventory;
  inventory.identity = 0x5100'0000'0000'0000ULL + seed;
  inventory.role = layout.role;
  inventory.source_count = layout.partition_count;
  constexpr std::array<std::uint32_t, 3U> kScaleBits{{
      0x3f80'0000U,  // 1.0
      0x3f00'0000U,  // 0.5
      0x3e80'0000U,  // 0.25
  }};
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    inventory.sources[index] =
        kernels::sm87_target_aot_projection_packed_source_binding(
            layout, index, 0x6100'0000'0000'0000ULL + seed * 4U + index,
            make_digest(seed * 17U + index * 2U + 1U),
            make_digest(seed * 17U + index * 2U + 2U),
            kScaleBits[index]);
  }
  return inventory;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedTransformReceipt
make_transform_receipt(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const Sm87TargetAotProjectionPackedSourceInventory& inventory,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest) noexcept {
  kernels::Sm87TargetAotProjectionPackedTransformReceipt receipt;
  receipt.artifact_identity = manifest.artifact_identity;
  receipt.source_inventory_identity = inventory.identity;
  receipt.role = layout.role;
  receipt.plan_identity = layout.plan_identity;
  receipt.layout_identity = layout.layout_identity;
  receipt.encoding = layout.encoding;
  receipt.transform_identity = kernels::
      Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.partition_count = layout.partition_count;
  receipt.payload = {manifest.artifact_identity,
                     manifest.payload_offset,
                     manifest.payload_bytes,
                     manifest.payload_digest,
                     true};
  receipt.deterministic_transform = true;
  receipt.no_arithmetic_conversion = true;
  receipt.no_request_time_repacking = true;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    const auto& partition = layout.partitions[index];
    const auto& source = inventory.sources[index];
    auto& observed = receipt.partitions[index];
    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    const std::uint64_t weight_bytes =
        values * partition.weight_bits / 8U;
    const std::uint64_t block_scale_values =
        partition.block_scale_group_k == 0U
            ? 0U
            : values / partition.block_scale_group_k;
    observed.logical_role = partition.logical_role;
    observed.partition_index = static_cast<std::uint32_t>(index);
    observed.tensor_identity = source.tensor_identity;
    observed.observed_source_weight_digest = source.weight_digest;
    observed.observed_source_scale_digest = source.scale_digest;
    observed.source_weight_bytes_hashed = weight_bytes;
    observed.source_scale_bytes_hashed =
        block_scale_values + sizeof(std::uint32_t);
    observed.repacked_weight_values = values;
    observed.repacked_block_scale_values = block_scale_values;
    observed.source_block_scale_e4m3fn_bytes_scanned = block_scale_values;
    observed.payload_block_scale_e4m3fn_bytes_scanned = block_scale_values;
    observed.payload_offset = partition.payload_offset;
    observed.payload_bytes = partition.payload_bytes;
    observed.source_digests_computed_from_tensor_bytes = true;
    observed.canonical_address_bijection_applied = true;
    observed.bit_exact_weight_permutation = true;
    observed.bit_exact_block_scale_permutation =
        block_scale_values != 0U;
    observed.tensor_scale_kept_external = true;
  }
  return receipt;
}

[[nodiscard]] bool same_logical_weight(
    const kernels::Sm87TargetAotProjectionPackedWeightAddress& left,
    const kernels::Sm87TargetAotProjectionPackedWeightAddress& right) {
  return left.valid && right.valid &&
         left.partition_index == right.partition_index && left.n == right.n &&
         left.k == right.k && left.byte_offset == right.byte_offset &&
         left.nibble == right.nibble &&
         left.s2r_lane == right.s2r_lane &&
         left.s2r_component == right.s2r_component &&
         left.virtual_m_warp_consumer0 ==
             right.virtual_m_warp_consumer0 &&
         left.virtual_m_warp_consumer1 ==
             right.virtual_m_warp_consumer1;
}

[[nodiscard]] bool same_logical_scale(
    const kernels::Sm87TargetAotProjectionPackedScaleAddress& left,
    const kernels::Sm87TargetAotProjectionPackedScaleAddress& right) {
  return left.valid && right.valid &&
         left.partition_index == right.partition_index && left.n == right.n &&
         left.scale_group == right.scale_group &&
         left.byte_offset == right.byte_offset &&
         left.virtual_m_warp_consumer0 ==
             right.virtual_m_warp_consumer0 &&
         left.virtual_m_warp_consumer1 ==
             right.virtual_m_warp_consumer1;
}

// Exhaustive proof is factorized instead of allocating a payload-sized set:
// every physical cell is visited, and the complete N256xK64 inner permutation
// is enumerated once per partition.  Since cell translation is affine and
// checked for every cell, the Cartesian product proves full-layout coverage.
[[nodiscard]] bool exhaustive_factorized_bijection(
    const Sm87TargetAotProjectionPackedLayout& layout) {
  if (!layout.valid()) {
    return false;
  }
  std::uint64_t payload_cursor = 0U;
  for (std::size_t partition_index = 0U;
       partition_index < layout.partition_count; ++partition_index) {
    const auto& partition = layout.partitions[partition_index];
    if (partition.payload_offset != payload_cursor ||
        partition.payload_offset % layout.payload_alignment != 0U) {
      return false;
    }
    const std::uint64_t cell_count =
        static_cast<std::uint64_t>(partition.n_tiles) * partition.k_tiles;
    const std::uint64_t expected_weight_bytes =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features * partition.weight_bits / 8U;
    const std::uint64_t expected_scale_bytes =
        partition.block_scale_group_k == 0U
            ? 0U
            : static_cast<std::uint64_t>(partition.output_features) *
                  (partition.input_features /
                   partition.block_scale_group_k);
    if (cell_count * partition.cell_bytes != partition.payload_bytes ||
        cell_count * partition.weight_bytes_per_cell !=
            expected_weight_bytes ||
        cell_count * partition.block_scale_bytes_per_cell !=
            expected_scale_bytes ||
        partition.payload_bytes != expected_weight_bytes +
                                       expected_scale_bytes) {
      return false;
    }

    // Visit every cell and prove there are no gaps between cell ranges.
    for (std::uint64_t linear = 0U; linear < cell_count; ++linear) {
      const std::size_t n_tile = linear / partition.k_tiles;
      const std::size_t k_tile = linear % partition.k_tiles;
      const auto cell = kernels::sm87_target_aot_projection_packed_cell(
          layout, partition_index, n_tile, k_tile);
      const std::uint64_t expected =
          partition.payload_offset + linear * partition.cell_bytes;
      if (!cell.valid || cell.payload_offset != expected ||
          cell.weight_offset != expected ||
          cell.block_scale_offset !=
              expected + partition.weight_bytes_per_cell ||
          cell.cell_bytes != partition.cell_bytes) {
        return false;
      }

      // The first and last logical coordinates attest translation for every
      // affine cell base, including each role's terminal N/K cell.
      const std::uint32_t first_n = static_cast<std::uint32_t>(
          n_tile * kernels::kSm87TargetAotProjectionBlockN);
      const std::uint32_t first_k = static_cast<std::uint32_t>(
          k_tile * kernels::kSm87TargetAotProjectionBlockK);
      const auto first =
          kernels::sm87_target_aot_projection_packed_weight_address(
              layout, partition_index, first_n, first_k);
      const auto last =
          kernels::sm87_target_aot_projection_packed_weight_address(
              layout, partition_index,
              first_n + kernels::kSm87TargetAotProjectionBlockN - 1U,
              first_k + kernels::kSm87TargetAotProjectionBlockK - 1U);
      if (!first.valid || !last.valid ||
          !same_logical_weight(
              first, kernels::sm87_target_aot_projection_packed_reverse_weight(
                         layout, first.byte_offset, first.nibble)) ||
          !same_logical_weight(
              last, kernels::sm87_target_aot_projection_packed_reverse_weight(
                        layout, last.byte_offset, last.nibble))) {
        return false;
      }
      if (partition.block_scale_group_k != 0U) {
        const std::uint32_t first_group =
            static_cast<std::uint32_t>(k_tile * 4U);
        const auto first_scale =
            kernels::sm87_target_aot_projection_packed_scale_address(
                layout, partition_index, first_n, first_group);
        const auto last_scale =
            kernels::sm87_target_aot_projection_packed_scale_address(
                layout, partition_index,
                first_n + kernels::kSm87TargetAotProjectionBlockN - 1U,
                first_group + 3U);
        if (!same_logical_scale(
                first_scale,
                kernels::sm87_target_aot_projection_packed_reverse_scale(
                    layout, first_scale.byte_offset)) ||
            !same_logical_scale(
                last_scale,
                kernels::sm87_target_aot_projection_packed_reverse_scale(
                    layout, last_scale.byte_offset))) {
          return false;
        }
      }
    }

    // Enumerate the full cell-local physical address space.  The largest set
    // is 32 KiB: one byte per possible FP4 nibble slot.
    std::array<std::uint8_t, 32'768U> weight_slots{};
    std::array<std::uint8_t, 1'024U> scale_slots{};
    std::uint32_t weight_slot_count = 0U;
    std::uint32_t scale_slot_count = 0U;
    for (std::uint32_t n = 0U;
         n < kernels::kSm87TargetAotProjectionBlockN; ++n) {
      for (std::uint32_t k = 0U;
           k < kernels::kSm87TargetAotProjectionBlockK; ++k) {
        const auto address =
            kernels::sm87_target_aot_projection_packed_weight_address(
                layout, partition_index, n, k);
        if (!address.valid || address.virtual_m_warp_consumer0 != 0U ||
            address.virtual_m_warp_consumer1 != 1U ||
            !same_logical_weight(
                address,
                kernels::sm87_target_aot_projection_packed_reverse_weight(
                    layout, address.byte_offset, address.nibble))) {
          return false;
        }
        const std::uint64_t relative =
            address.byte_offset - partition.payload_offset;
        const std::uint32_t slot =
            partition.weight_bits == 4U
                ? static_cast<std::uint32_t>(relative * 2U + address.nibble)
                : static_cast<std::uint32_t>(relative);
        if (slot >= weight_slots.size() || weight_slots[slot] != 0U) {
          return false;
        }
        weight_slots[slot] = 1U;
        ++weight_slot_count;
      }
    }
    const std::uint32_t expected_weight_slots =
        static_cast<std::uint32_t>(
            kernels::kSm87TargetAotProjectionBlockN *
            kernels::kSm87TargetAotProjectionBlockK);
    if (weight_slot_count != expected_weight_slots) {
      return false;
    }
    for (std::uint32_t index = 0U; index < expected_weight_slots; ++index) {
      if (weight_slots[index] != 1U) {
        return false;
      }
    }

    if (partition.block_scale_group_k != 0U) {
      for (std::uint32_t n = 0U;
           n < kernels::kSm87TargetAotProjectionBlockN; ++n) {
        for (std::uint32_t group = 0U; group < 4U; ++group) {
          const auto address =
              kernels::sm87_target_aot_projection_packed_scale_address(
                  layout, partition_index, n, group);
          if (!address.valid ||
              !same_logical_scale(
                  address,
                  kernels::sm87_target_aot_projection_packed_reverse_scale(
                      layout, address.byte_offset))) {
            return false;
          }
          const std::uint32_t slot = static_cast<std::uint32_t>(
              address.byte_offset - partition.payload_offset -
              partition.weight_bytes_per_cell);
          if (slot >= scale_slots.size() || scale_slots[slot] != 0U) {
            return false;
          }
          scale_slots[slot] = 1U;
          ++scale_slot_count;
        }
      }
      if (scale_slot_count != partition.block_scale_bytes_per_cell) {
        return false;
      }
      for (std::uint32_t index = 0U;
           index < partition.block_scale_bytes_per_cell; ++index) {
        if (scale_slots[index] != 1U) {
          return false;
        }
      }
    }

    // Exhaust the 128 physical S2R fragments and both M-warp consumers.
    std::uint32_t fragment_count = 0U;
    for (std::uint32_t k16 = 0U; k16 < 4U; ++k16) {
      for (std::uint32_t n_warp = 0U; n_warp < 4U; ++n_warp) {
        for (std::uint32_t panel = 0U; panel < 8U; ++panel) {
          const auto fragment =
              kernels::sm87_target_aot_projection_packed_fragment(
                  layout, partition_index, 0U, 0U, k16, n_warp, panel);
          const std::uint32_t linear_fragment =
              (k16 * 4U + n_warp) * 8U + panel;
          const std::uint32_t fragment_bytes =
              partition.weight_bits == 4U ? 64U : 128U;
          if (!fragment.valid ||
              fragment.weight_offset !=
                  partition.payload_offset +
                      static_cast<std::uint64_t>(linear_fragment) *
                          fragment_bytes ||
              fragment.virtual_m_warp_consumer0 != 0U ||
              fragment.virtual_m_warp_consumer1 != 1U ||
              (partition.block_scale_group_k != 0U &&
               fragment.block_scale_offset !=
                   partition.payload_offset +
                       partition.weight_bytes_per_cell +
                       static_cast<std::uint64_t>(linear_fragment) * 8U)) {
            return false;
          }
          ++fragment_count;
        }
      }
    }
    if (fragment_count != 128U) {
      return false;
    }

    // Exhaust the 16-byte G2S vector ownership for one cell.  Every loader
    // thread receives two NVFP4-B passes or four FP8-B passes; scale has one
    // pass on exactly the first 64 threads.
    std::array<std::uint8_t, 256U> owner_counts{};
    const std::uint32_t weight_vectors =
        partition.weight_bytes_per_cell /
        kernels::kSm87TargetAotProjectionPackedVectorBytes;
    for (std::uint32_t vector = 0U; vector < weight_vectors; ++vector) {
      const auto owner =
          kernels::sm87_target_aot_projection_packed_g2s_vector(
              layout, partition_index, 0U, 0U,
              Sm87TargetAotProjectionPackedRegion::kWeight, vector);
      if (!owner.valid || owner.loader_thread != vector % 256U ||
          owner.loader_pass != vector / 256U ||
          owner.shared_byte_offset != vector * 16U ||
          owner.global_byte_offset !=
              partition.payload_offset + vector * 16U) {
        return false;
      }
      ++owner_counts[owner.loader_thread];
    }
    const std::uint8_t expected_passes =
        static_cast<std::uint8_t>(weight_vectors / 256U);
    for (const std::uint8_t count : owner_counts) {
      if (count != expected_passes) {
        return false;
      }
    }
    if (kernels::sm87_target_aot_projection_packed_g2s_vector(
            layout, partition_index, 0U, 0U,
            Sm87TargetAotProjectionPackedRegion::kWeight, weight_vectors)
            .valid) {
      return false;
    }
    if (partition.block_scale_bytes_per_cell != 0U) {
      for (std::uint32_t vector = 0U; vector < 64U; ++vector) {
        const auto owner =
            kernels::sm87_target_aot_projection_packed_g2s_vector(
                layout, partition_index, 0U, 0U,
                Sm87TargetAotProjectionPackedRegion::kBlockScale, vector);
        if (!owner.valid || owner.loader_thread != vector ||
            owner.loader_pass != 0U ||
            owner.global_byte_offset !=
                partition.payload_offset +
                    partition.weight_bytes_per_cell + vector * 16U) {
          return false;
        }
      }
    } else if (kernels::sm87_target_aot_projection_packed_g2s_vector(
                   layout, partition_index, 0U, 0U,
                   Sm87TargetAotProjectionPackedRegion::kBlockScale, 0U)
                   .valid) {
      return false;
    }
    payload_cursor += partition.payload_bytes;
  }
  return payload_cursor == layout.payload_bytes;
}

void test_nvfp4_physical_warp_maps(TestContext& test) {
  for (std::size_t n_half = 0U; n_half < 2U; ++n_half) {
    std::array<std::uint8_t, 8U> seen{};
    for (std::size_t partition_branch = 0U; partition_branch < 2U;
         ++partition_branch) {
      for (std::size_t m_warp = 0U; m_warp < 2U; ++m_warp) {
        for (std::size_t n_warp_in_half = 0U; n_warp_in_half < 2U;
             ++n_warp_in_half) {
          const std::size_t n_warp = n_half * 2U + n_warp_in_half;
          const auto mapping =
              kernels::sm87_target_aot_projection_nvfp4_physical_warp(
                  Sm87TargetAotProjectionRole::kNvFp4GateUp,
                  partition_branch, m_warp, n_warp);
          const std::size_t expected =
              partition_branch * 4U + m_warp * 2U + n_warp_in_half;
          if (!mapping.valid ||
              mapping.partition_branch != partition_branch ||
              mapping.n_half != n_half ||
              mapping.virtual_m_warp != m_warp ||
              mapping.virtual_n_warp != n_warp_in_half ||
              mapping.physical_warp != expected ||
              !mapping.physical_warp_reused_across_n_halves ||
              expected >= seen.size() || seen[expected] != 0U) {
            test.expect(false,
                        "Gate/Up branch and N-half warp map is exhaustive");
            return;
          }
          seen[expected] = 1U;

          const auto reused =
              kernels::sm87_target_aot_projection_nvfp4_physical_warp(
                  Sm87TargetAotProjectionRole::kNvFp4GateUp,
                  partition_branch, m_warp,
                  (1U - n_half) * 2U + n_warp_in_half);
          if (!reused.valid || reused.physical_warp != mapping.physical_warp) {
            test.expect(false,
                        "Gate/Up physical warps are reused across N halves");
            return;
          }
        }
      }
    }
    for (const auto count : seen) {
      if (count != 1U) {
        test.expect(false,
                    "each Gate/Up N half owns all eight physical warps once");
        return;
      }
    }
  }

  std::array<std::uint8_t, 8U> seen_down{};
  for (std::size_t m_warp = 0U; m_warp < 2U; ++m_warp) {
    for (std::size_t n_warp = 0U; n_warp < 4U; ++n_warp) {
      const auto mapping =
          kernels::sm87_target_aot_projection_nvfp4_physical_warp(
              Sm87TargetAotProjectionRole::kNvFp4Down, 0U, m_warp,
              n_warp);
      const std::size_t expected = m_warp * 4U + n_warp;
      if (!mapping.valid || mapping.partition_branch != 0U ||
          mapping.n_half != 0U || mapping.virtual_m_warp != m_warp ||
          mapping.virtual_n_warp != n_warp ||
          mapping.physical_warp != expected ||
          mapping.physical_warp_reused_across_n_halves ||
          expected >= seen_down.size() || seen_down[expected] != 0U) {
        test.expect(false, "Down physical warp map is m*4+n");
        return;
      }
      seen_down[expected] = 1U;
    }
  }
  for (const auto count : seen_down) {
    if (count != 1U) {
      test.expect(false, "Down owns all eight physical warps once");
      return;
    }
  }

  test.expect(
      !kernels::sm87_target_aot_projection_nvfp4_physical_warp(
           Sm87TargetAotProjectionRole::kNvFp4GateUp, 2U, 0U, 0U)
           .valid &&
          !kernels::sm87_target_aot_projection_nvfp4_physical_warp(
               Sm87TargetAotProjectionRole::kNvFp4Down, 1U, 0U, 0U)
               .valid &&
          !kernels::sm87_target_aot_projection_nvfp4_physical_warp(
               Sm87TargetAotProjectionRole::kNvFp4Down, 0U, 2U, 0U)
               .valid &&
          !kernels::sm87_target_aot_projection_nvfp4_physical_warp(
               Sm87TargetAotProjectionRole::kNvFp4Down, 0U, 0U, 4U)
               .valid &&
          !kernels::sm87_target_aot_projection_nvfp4_physical_warp(
               Sm87TargetAotProjectionRole::kFp8FullQkv, 0U, 0U, 0U)
               .valid,
      "NVFP4 physical warp map fails closed outside its exact role domain");
}

void test_all_roles_and_token_independence(TestContext& test) {
  for (const auto role : kRoles) {
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(role);
    test.expect(layout.valid(), "every projection role has a packed layout");
    test.expect(exhaustive_factorized_bijection(layout),
                "factorized exhaustive layout bijection and coverage");
    for (const std::size_t tokens :
         kernels::kSm87TargetAotWitnessTokenCounts) {
      const auto plan =
          kernels::sm87_target_aot_projection_plan(role, tokens);
      bool same = plan.valid() &&
                  plan.input_features == layout.input_features &&
                  plan.projected_output_features ==
                      layout.projected_output_features &&
                  plan.partition_count == layout.partition_count;
      for (std::size_t index = 0U;
           same && index < layout.partition_count; ++index) {
        same = plan.partitions[index].role ==
                   layout.partitions[index].logical_role &&
               plan.partitions[index].output_features ==
                   layout.partitions[index].output_features;
      }
      test.expect(same,
                  "P40/P60/P130 share one token-independent operand ABI");
    }
    test.expect(!layout.cuda_implementation_present &&
                    !layout.static_resources_qualified &&
                    !layout.numerical_contract_qualified &&
                    !layout.production_dispatch_eligible,
                "host ABI never claims CUDA/resource/numerical/production qualification");
  }
}

void test_mapping_bounds_and_wrong_offsets(TestContext& test) {
  for (const auto role : kRoles) {
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(role);
    const auto& partition = layout.partitions[0U];
    test.expect(
        !kernels::sm87_target_aot_projection_packed_weight_address(
             layout, layout.partition_count, 0U, 0U)
             .valid &&
            !kernels::sm87_target_aot_projection_packed_weight_address(
                 layout, 0U, partition.output_features, 0U)
                 .valid &&
            !kernels::sm87_target_aot_projection_packed_weight_address(
                 layout, 0U, 0U, partition.input_features)
                 .valid,
        "weight coordinate bounds fail closed");
    test.expect(
        !kernels::sm87_target_aot_projection_packed_cell(
             layout, 0U, partition.n_tiles, 0U)
             .valid &&
            !kernels::sm87_target_aot_projection_packed_cell(
                 layout, 0U, 0U, partition.k_tiles)
                 .valid &&
            !kernels::sm87_target_aot_projection_packed_reverse_weight(
                 layout, layout.payload_bytes, 0U)
                 .valid,
        "cell and reverse offset bounds fail closed");
    const auto cell = kernels::sm87_target_aot_projection_packed_cell(
        layout, 0U, 0U, 0U);
    if (partition.block_scale_bytes_per_cell != 0U) {
      test.expect(
          !kernels::sm87_target_aot_projection_packed_reverse_weight(
               layout, cell.block_scale_offset, 0U)
               .valid &&
              !kernels::sm87_target_aot_projection_packed_reverse_scale(
               layout, cell.weight_offset)
               .valid &&
              !kernels::sm87_target_aot_projection_packed_scale_address(
                   layout, 0U, partition.output_features, 0U)
                   .valid &&
              !kernels::sm87_target_aot_projection_packed_scale_address(
                   layout, 0U, 0U, partition.input_features / 16U)
                   .valid,
          "block-scale bytes cannot alias weights and coordinates fail closed");
    } else {
      test.expect(
          !kernels::sm87_target_aot_projection_packed_scale_address(
               layout, 0U, 0U, 0U)
               .valid,
          "FP8 tensor scale is external and has no per-cell offset");
    }
  }
}

void test_layout_mutations_fail_closed(TestContext& test) {
  for (const auto role : kRoles) {
    const auto canonical =
        kernels::sm87_target_aot_projection_packed_layout(role);
    auto changed = canonical;
    ++changed.input_features;
    test.expect(!changed.valid(), "wrong layout K fails closed");
    changed = canonical;
    ++changed.projected_output_features;
    test.expect(!changed.valid(), "wrong layout N fails closed");
    changed = canonical;
    changed.plan_identity =
        kernels::Sm87TargetAotProjectionPackedPlanIdentity::kInvalid;
    test.expect(!changed.valid(), "wrong typed plan identity fails closed");
    changed = canonical;
    ++changed.pipeline_stages;
    test.expect(!changed.valid(), "wrong pipeline-stage binding fails closed");
    changed = canonical;
    ++changed.payload_bytes;
    test.expect(!changed.valid(), "wrong payload size fails closed");
    changed = canonical;
    changed.partitions[0U].payload_offset += 256U;
    test.expect(!changed.valid(), "payload hole fails closed");
    if (changed.partition_count > 1U) {
      changed = canonical;
      changed.partitions[1U].payload_offset =
          changed.partitions[0U].payload_offset;
      test.expect(!changed.valid(), "partition alias fails closed");
    }
    changed = canonical;
    changed.cuda_implementation_present = true;
    test.expect(!changed.valid(), "host ABI cannot acquire CUDA status");
    changed = canonical;
    changed.numerical_contract_qualified = true;
    test.expect(!changed.valid(), "host ABI cannot acquire numerical status");
  }
  test.expect(!kernels::sm87_target_aot_packed_add_fits(
                  std::numeric_limits<std::uint64_t>::max(), 1U) &&
                  !kernels::sm87_target_aot_packed_mul_fits(
                      std::numeric_limits<std::uint64_t>::max(), 2U),
              "size arithmetic overflow fails closed");
}

void test_payload_binding_and_alias(TestContext& test) {
  const auto layout = kGate;
  constexpr std::uintptr_t kBase = 0x1000'0000U;
  const auto first = kernels::sm87_target_aot_projection_bind_packed_payload(
      layout, kBase, layout.payload_bytes);
  const auto overlapping =
      kernels::sm87_target_aot_projection_bind_packed_payload(
          layout, kBase + 256U, layout.payload_bytes);
  const auto adjacent = kernels::sm87_target_aot_projection_bind_packed_payload(
      layout, first.end, layout.payload_bytes);
  test.expect(first.valid && overlapping.valid && adjacent.valid,
              "aligned exact-size payload views bind");
  test.expect(
      kernels::sm87_target_aot_projection_payload_aliases(first, overlapping) &&
          !kernels::sm87_target_aot_projection_payload_pair_admissible(
              first, overlapping) &&
          kernels::sm87_target_aot_projection_payload_pair_admissible(
              first, adjacent),
      "payload alias is rejected while exact adjacency is admitted");
  test.expect(
      !kernels::sm87_target_aot_projection_bind_packed_payload(
           layout, 0U, layout.payload_bytes)
           .valid &&
          !kernels::sm87_target_aot_projection_bind_packed_payload(
               layout, kBase + 1U, layout.payload_bytes)
               .valid &&
          !kernels::sm87_target_aot_projection_bind_packed_payload(
               layout, kBase, layout.payload_bytes - 1U)
               .valid,
      "null, misaligned, and wrong-size payload views fail closed");
  const std::uintptr_t high_aligned =
      std::numeric_limits<std::uintptr_t>::max() &
      ~static_cast<std::uintptr_t>(255U);
  test.expect(
      !kernels::sm87_target_aot_projection_bind_packed_payload(
           layout, high_aligned, layout.payload_bytes)
           .valid,
      "wrapped payload address range fails closed");
}

void test_typed_manifests(TestContext& test) {
  for (std::size_t role_index = 0U; role_index < kRoles.size(); ++role_index) {
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(kRoles[role_index]);
    const auto inventory = make_inventory(layout, role_index + 1U);
    const auto manifest =
        kernels::sm87_target_aot_projection_make_packed_manifest(
            layout.role, 0x7100'0000'0000'0000ULL + role_index,
            inventory, make_digest(0x8000U + role_index));
    test.expect(inventory.valid(layout),
                "typed source inventory binds exact role shapes and scales");
    test.expect(
        kernels::sm87_target_aot_projection_packed_manifest_structurally_valid(
            manifest) &&
            kernels::sm87_target_aot_projection_validate_packed_manifest(
                manifest, inventory),
        "sidecar manifest schema validates against DeploymentPlan sources");
    const kernels::Sm87TargetAotProjectionPackedPayloadReceipt receipt{
        manifest.artifact_identity,
        manifest.payload_offset,
        manifest.payload_bytes,
        manifest.payload_digest,
        true};
    test.expect(
        kernels::sm87_target_aot_projection_validate_payload_receipt(
            manifest, receipt),
        "loader receipt must bind a digest computed from actual payload bytes");
    const auto transform =
        make_transform_receipt(layout, inventory, manifest);
    test.expect(
        kernels::sm87_target_aot_projection_validate_transform_receipt(
            manifest, inventory, transform),
        "source bytes, exact permutation, E4M3FN scan, and payload digest "
        "close one authenticated transform receipt");
    auto forbidden_source = transform;
    forbidden_source.partitions[0U]
        .source_forbidden_block_scale_codes = 1U;
    test.expect(
        !kernels::sm87_target_aot_projection_validate_transform_receipt(
            manifest, inventory, forbidden_source),
        "forbidden source E4M3FN terminal code fails closed");
    auto forbidden_payload = transform;
    forbidden_payload.partitions[0U]
        .payload_forbidden_block_scale_codes = 1U;
    test.expect(
        !kernels::sm87_target_aot_projection_validate_transform_receipt(
            manifest, inventory, forbidden_payload),
        "forbidden packed E4M3FN terminal code fails closed");
    auto converted = transform;
    converted.no_arithmetic_conversion = false;
    test.expect(
        !kernels::sm87_target_aot_projection_validate_transform_receipt(
            manifest, inventory, converted),
        "arithmetic conversion cannot impersonate a bit-exact permutation");
    auto unobserved_source = transform;
    unobserved_source.partitions[0U]
        .source_digests_computed_from_tensor_bytes = false;
    test.expect(
        !kernels::sm87_target_aot_projection_validate_transform_receipt(
            manifest, inventory, unobserved_source),
        "metadata-derived source digests cannot authenticate tensor bytes");
    auto incomplete_scan = transform;
    --incomplete_scan.partitions[0U]
          .payload_block_scale_e4m3fn_bytes_scanned;
    test.expect(
        !kernels::sm87_target_aot_projection_validate_transform_receipt(
            manifest, inventory, incomplete_scan),
        "partial E4M3FN payload scan fails closed");
    auto corrupted_payload_receipt = receipt;
    corrupted_payload_receipt.observed_payload_digest.bytes[7U] ^= 1U;
    test.expect(
        !kernels::sm87_target_aot_projection_validate_payload_receipt(
            manifest, corrupted_payload_receipt),
        "observed payload corruption fails digest authentication");
    auto unobserved_payload_receipt = receipt;
    unobserved_payload_receipt.digest_computed_from_payload_bytes = false;
    test.expect(
        !kernels::sm87_target_aot_projection_validate_payload_receipt(
            manifest, unobserved_payload_receipt),
        "manifest digest cannot impersonate a loader payload observation");

    auto changed = manifest;
    ++changed.abi_major;
    test.expect(
        !kernels::sm87_target_aot_projection_seal_packed_manifest(&changed),
        "wrong ABI fails closed even after reseal attempt");
    changed = manifest;
    changed.plan_identity =
        kernels::Sm87TargetAotProjectionPackedPlanIdentity::kInvalid;
    test.expect(
        !kernels::sm87_target_aot_projection_seal_packed_manifest(&changed),
        "wrong manifest plan identity fails closed");
    changed = manifest;
    changed.payload_offset += 1U;
    test.expect(
        !kernels::sm87_target_aot_projection_seal_packed_manifest(&changed),
        "wrong or misaligned payload offset fails closed");
    changed = manifest;
    changed.payload_digest = {};
    test.expect(
        !kernels::sm87_target_aot_projection_seal_packed_manifest(&changed),
        "zero payload digest fails closed");
    changed = manifest;
    changed.sources[0U].input_features += 1U;
    test.expect(
        !kernels::sm87_target_aot_projection_seal_packed_manifest(&changed),
        "wrong source K fails closed");
    changed = manifest;
    changed.sources[0U].output_features += 1U;
    test.expect(
        !kernels::sm87_target_aot_projection_seal_packed_manifest(&changed),
        "wrong source N fails closed");
    changed = manifest;
    changed.sources[0U].tensor_scale_bits = 0x7f80'0000U;
    test.expect(
        !kernels::sm87_target_aot_projection_seal_packed_manifest(&changed),
        "non-finite tensor scale bits fail closed");
    changed = manifest;
    changed.seal.value ^= 1U;
    test.expect(
        !kernels::sm87_target_aot_projection_packed_manifest_structurally_valid(
            changed),
        "manifest mutation without reseal fails closed");

    auto duplicate = inventory;
    if (duplicate.source_count > 1U) {
      duplicate.sources[1U].tensor_identity =
          duplicate.sources[0U].tensor_identity;
      test.expect(!duplicate.valid(layout),
                  "source identity alias fails closed");
    }
  }

  const auto expected = make_inventory(kGate, 0x91U);
  const auto manifest =
      kernels::sm87_target_aot_projection_make_packed_manifest(
          kGate.role, 0x7200'0000'0000'0001ULL, expected,
          make_digest(0x9200U));
  auto source_swap = manifest;
  std::swap(source_swap.sources[0U].tensor_identity,
            source_swap.sources[1U].tensor_identity);
  std::swap(source_swap.sources[0U].weight_digest,
            source_swap.sources[1U].weight_digest);
  test.expect(
      kernels::sm87_target_aot_projection_seal_packed_manifest(&source_swap) &&
          !kernels::sm87_target_aot_projection_validate_packed_manifest(
              source_swap, expected),
      "Gate/Up source swap is rejected by the expected typed bindings");

  auto scale_swap = manifest;
  std::swap(scale_swap.sources[0U].scale_digest,
            scale_swap.sources[1U].scale_digest);
  std::swap(scale_swap.sources[0U].tensor_scale_bits,
            scale_swap.sources[1U].tensor_scale_bits);
  test.expect(
      kernels::sm87_target_aot_projection_seal_packed_manifest(&scale_swap) &&
          !kernels::sm87_target_aot_projection_validate_packed_manifest(
              scale_swap, expected),
      "Gate/Up scale swap is rejected after a valid structural reseal");

  auto partition_swap = manifest;
  std::swap(partition_swap.sources[0U], partition_swap.sources[1U]);
  test.expect(
      !kernels::sm87_target_aot_projection_seal_packed_manifest(
          &partition_swap),
      "Gate/Up semantic partition swap fails structural validation");
  test.expect(!kernels::sm87_target_aot_projection_seal_packed_manifest(nullptr),
              "null manifest seal fails closed");
  test.expect(
      kernels::sm87_target_aot_projection_block_scale_e4m3fn_code_is_forbidden(
          0x7fU) &&
          kernels::
              sm87_target_aot_projection_block_scale_e4m3fn_code_is_forbidden(
              0xffU) &&
          kernels::
              sm87_target_aot_projection_block_scale_e4m3fn_code_is_forbidden(
                  0x81U) &&
          !kernels::
              sm87_target_aot_projection_block_scale_e4m3fn_code_is_forbidden(
                  0x80U) &&
          !kernels::
              sm87_target_aot_projection_block_scale_e4m3fn_code_is_forbidden(
              0x7eU),
      "NVFP4 block scales reject NaN and negative nonzero but accept signed "
      "zero");
}

}  // namespace

int main() {
  TestContext test;
  test_all_roles_and_token_independence(test);
  test_nvfp4_physical_warp_maps(test);
  test_mapping_bounds_and_wrong_offsets(test);
  test_layout_mutations_fail_closed(test);
  test_payload_binding_and_alias(test);
  test_typed_manifests(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " SM87 target AOT projection layout checks failed\n";
    return 1;
  }
    std::cout << "SM87 target AOT projection packed-layout schema, factorized "
               "bijection, G2S/S2R ownership, and manifest checks passed\n";
  return 0;
}
