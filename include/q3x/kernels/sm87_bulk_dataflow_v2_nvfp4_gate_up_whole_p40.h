#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_projection.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Independent whole-layer Gate+Up successor for
// AC-PREFILL-SM87-BULK-DATAFLOW-v2.  It deliberately does not reuse the
// M256/M1024 work-stealing identity: one cooperative launch owns the complete
// P40000 x J17408 Gate+Up publication and preserves the authenticated target-
// AOT payload bytes.  This remains a default-off CUDA admission with no
// runner selector or production authority.
inline constexpr char kSm87BulkV2NvFp4GateUpWholeP40Identity[] =
    "q3x.sm87.bulk-v2.nvfp4-gate-up.whole-p40000."
    "m64n64k64-raster-m4n8.v1";

inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40Tokens = 40'000U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40TileM = 64U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40TileN = 64U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40TileK = 64U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40MTiles =
        kSm87BulkV2NvFp4GateUpWholeP40Tokens /
        kSm87BulkV2NvFp4GateUpWholeP40TileM;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40NTiles =
        kSm87BulkV2NvFp4Intermediate /
        kSm87BulkV2NvFp4GateUpWholeP40TileN;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40KTiles =
        kSm87BulkV2NvFp4Hidden /
        kSm87BulkV2NvFp4GateUpWholeP40TileK;

inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40RasterM = 4U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40RasterN = 8U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40PersistentCtas = 32U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40Threads = 256U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40PipelineStages = 3U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40RegisterStages = 2U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40RequiredCtasPerSm = 2U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40MaximumRegisters = 128U;

inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40MGroups =
        (kSm87BulkV2NvFp4GateUpWholeP40MTiles +
         kSm87BulkV2NvFp4GateUpWholeP40RasterM - 1U) /
        kSm87BulkV2NvFp4GateUpWholeP40RasterM;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40NGroups =
        (kSm87BulkV2NvFp4GateUpWholeP40NTiles +
         kSm87BulkV2NvFp4GateUpWholeP40RasterN - 1U) /
        kSm87BulkV2NvFp4GateUpWholeP40RasterN;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40Cohorts =
        kSm87BulkV2NvFp4GateUpWholeP40MGroups *
        kSm87BulkV2NvFp4GateUpWholeP40NGroups;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40LogicalCells =
        kSm87BulkV2NvFp4GateUpWholeP40MTiles *
        kSm87BulkV2NvFp4GateUpWholeP40NTiles;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40ScheduledCells =
        kSm87BulkV2NvFp4GateUpWholeP40Cohorts *
        kSm87BulkV2NvFp4GateUpWholeP40PersistentCtas;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40MaskedCells =
        kSm87BulkV2NvFp4GateUpWholeP40ScheduledCells -
        kSm87BulkV2NvFp4GateUpWholeP40LogicalCells;

inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40ActivationBytesPerStage =
        kSm87BulkV2NvFp4GateUpWholeP40TileM *
        kSm87BulkV2NvFp4GateUpWholeP40TileK *
        sizeof(std::uint16_t);
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40BranchWeightBytesPerStage =
        kSm87BulkV2NvFp4GateUpWholeP40TileN *
        kSm87BulkV2NvFp4GateUpWholeP40TileK / 2U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40BranchScaleBytesPerStage =
        kSm87BulkV2NvFp4GateUpWholeP40TileN *
        (kSm87BulkV2NvFp4GateUpWholeP40TileK / 16U);
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40BytesPerStage =
        kSm87BulkV2NvFp4GateUpWholeP40ActivationBytesPerStage +
        2U *
            (kSm87BulkV2NvFp4GateUpWholeP40BranchWeightBytesPerStage +
             kSm87BulkV2NvFp4GateUpWholeP40BranchScaleBytesPerStage);
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40PipelineSharedBytes =
        kSm87BulkV2NvFp4GateUpWholeP40PipelineStages *
        kSm87BulkV2NvFp4GateUpWholeP40BytesPerStage;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4GateUpWholeP40DynamicSharedBytes =
        kSm87BulkV2NvFp4GateUpWholeP40PipelineSharedBytes;
inline constexpr std::uint64_t
    kSm87BulkV2NvFp4GateUpWholeP40InputBytes =
        static_cast<std::uint64_t>(
            kSm87BulkV2NvFp4GateUpWholeP40Tokens) *
        kSm87BulkV2NvFp4Hidden * sizeof(std::uint16_t);
inline constexpr std::uint64_t
    kSm87BulkV2NvFp4GateUpWholeP40HBytes =
        static_cast<std::uint64_t>(
            kSm87BulkV2NvFp4GateUpWholeP40Tokens) *
        kSm87BulkV2NvFp4Intermediate * sizeof(std::uint16_t);

[[nodiscard]] constexpr std::uint32_t
sm87_bulk_v2_nvfp4_gate_up_whole_p40_m_groups(
    const std::uint32_t m_tiles) noexcept {
  return (m_tiles + kSm87BulkV2NvFp4GateUpWholeP40RasterM - 1U) /
         kSm87BulkV2NvFp4GateUpWholeP40RasterM;
}

[[nodiscard]] constexpr std::uint32_t
sm87_bulk_v2_nvfp4_gate_up_whole_p40_n_groups(
    const std::uint32_t n_tiles) noexcept {
  return (n_tiles + kSm87BulkV2NvFp4GateUpWholeP40RasterN - 1U) /
         kSm87BulkV2NvFp4GateUpWholeP40RasterN;
}

struct Sm87BulkV2NvFp4GateUpWholeP40Cell final {
  std::uint32_t cohort = 0U;
  std::uint32_t cta = 0U;
  std::uint32_t m_group = 0U;
  std::uint32_t n_epoch = 0U;
  std::uint32_t n_group = 0U;
  std::uint32_t m_lane = 0U;
  std::uint32_t n_lane = 0U;
  std::uint32_t m_tile = 0U;
  std::uint32_t n_tile = 0U;
  std::uint32_t first_m = 0U;
  std::uint32_t first_n = 0U;
  std::uint32_t logical_ordinal = 0U;
  bool active = false;
};

// Pure mapping used by both the host proof and the CUDA body.  Cohorts are
// M-group-major: all N cohorts for one four-row activation group complete
// before the next M group begins.  CTA lanes are a fixed 4M x 8N Cartesian
// product, so four CTAs request the same Gate/Up B pair while eight CTAs
// request the same A tile at each full cohort.
[[nodiscard]] constexpr Sm87BulkV2NvFp4GateUpWholeP40Cell
sm87_bulk_v2_nvfp4_gate_up_whole_p40_cell(
    const std::uint32_t cohort, const std::uint32_t cta,
    const std::uint32_t m_tiles =
        kSm87BulkV2NvFp4GateUpWholeP40MTiles,
    const std::uint32_t n_tiles =
        kSm87BulkV2NvFp4GateUpWholeP40NTiles) noexcept {
  Sm87BulkV2NvFp4GateUpWholeP40Cell result;
  const std::uint32_t n_groups =
      sm87_bulk_v2_nvfp4_gate_up_whole_p40_n_groups(n_tiles);
  const std::uint32_t m_groups =
      sm87_bulk_v2_nvfp4_gate_up_whole_p40_m_groups(m_tiles);
  if (m_tiles == 0U || n_tiles == 0U || n_groups == 0U ||
      cohort >= m_groups * n_groups ||
      cta >= kSm87BulkV2NvFp4GateUpWholeP40PersistentCtas) {
    return result;
  }
  result.cohort = cohort;
  result.cta = cta;
  result.m_group = cohort / n_groups;
  result.n_epoch = cohort % n_groups;
  result.n_group = (result.m_group & 1U) == 0U
                       ? result.n_epoch
                       : n_groups - 1U - result.n_epoch;
  result.m_lane = cta / kSm87BulkV2NvFp4GateUpWholeP40RasterN;
  result.n_lane = cta % kSm87BulkV2NvFp4GateUpWholeP40RasterN;
  result.m_tile =
      result.m_group * kSm87BulkV2NvFp4GateUpWholeP40RasterM +
      result.m_lane;
  result.n_tile =
      result.n_group * kSm87BulkV2NvFp4GateUpWholeP40RasterN +
      result.n_lane;
  result.first_m =
      result.m_tile * kSm87BulkV2NvFp4GateUpWholeP40TileM;
  result.first_n =
      result.n_tile * kSm87BulkV2NvFp4GateUpWholeP40TileN;
  result.active = result.m_tile < m_tiles && result.n_tile < n_tiles;
  if (result.active) {
    result.logical_ordinal = result.m_tile * n_tiles + result.n_tile;
  }
  return result;
}

struct Sm87BulkV2NvFp4GateUpWholeP40Traffic final {
  std::uint32_t physical_launches = 0U;
  std::uint32_t legacy_m1024_segments = 0U;
  std::uint32_t m_groups = 0U;
  std::uint32_t n_groups = 0U;
  std::uint32_t cohorts = 0U;
  std::uint32_t logical_cells = 0U;
  std::uint32_t scheduled_cells = 0U;
  std::uint32_t masked_cells = 0U;
  std::uint32_t full_cohort_b_requesters = 0U;
  std::uint32_t full_cohort_distinct_b_pairs = 0U;
  std::uint32_t theoretical_full_cohort_b_reuse = 0U;
  std::uint32_t grid_syncs_per_launch = 0U;
  std::uint64_t activation_group_footprint_bytes = 0U;
  std::uint64_t distinct_activation_bytes = 0U;
  std::uint64_t branch_b_bytes_per_n_tile = 0U;
  std::uint64_t paired_b_bytes_per_n_tile = 0U;
  std::uint64_t logical_cta_b_request_bytes = 0U;
  std::uint64_t theoretical_l2_b_service_bytes = 0U;
  bool m_group_outer = false;
  bool one_a_g2s_per_k64_stage = false;
  bool independent_gate_up_b_and_scales = false;
  bool theoretical_not_measured = false;
  bool activation_group_l2_residency_unproven = false;
};

[[nodiscard]] constexpr Sm87BulkV2NvFp4GateUpWholeP40Traffic
sm87_bulk_v2_nvfp4_gate_up_whole_p40_traffic() noexcept {
  constexpr std::uint64_t kBranchBytesPerNTile =
      static_cast<std::uint64_t>(
          kSm87BulkV2NvFp4GateUpWholeP40TileN) *
      kSm87BulkV2NvFp4Hidden / 2U +
      static_cast<std::uint64_t>(
          kSm87BulkV2NvFp4GateUpWholeP40TileN) *
          (kSm87BulkV2NvFp4Hidden / 16U);
  constexpr std::uint64_t kPairBytesPerNTile =
      2U * kBranchBytesPerNTile;
  Sm87BulkV2NvFp4GateUpWholeP40Traffic result;
  result.physical_launches = 1U;
  result.legacy_m1024_segments = 0U;
  result.m_groups = kSm87BulkV2NvFp4GateUpWholeP40MGroups;
  result.n_groups = kSm87BulkV2NvFp4GateUpWholeP40NGroups;
  result.cohorts = kSm87BulkV2NvFp4GateUpWholeP40Cohorts;
  result.logical_cells = kSm87BulkV2NvFp4GateUpWholeP40LogicalCells;
  result.scheduled_cells = kSm87BulkV2NvFp4GateUpWholeP40ScheduledCells;
  result.masked_cells = kSm87BulkV2NvFp4GateUpWholeP40MaskedCells;
  result.full_cohort_b_requesters =
      kSm87BulkV2NvFp4GateUpWholeP40RasterM;
  result.full_cohort_distinct_b_pairs = 1U;
  result.theoretical_full_cohort_b_reuse =
      kSm87BulkV2NvFp4GateUpWholeP40RasterM;
  // One entry synchronization publishes initialized control state and one
  // terminal synchronization closes the receipt.  There is deliberately no
  // grid barrier in any of the 5,338 cohorts.
  result.grid_syncs_per_launch = 2U;
  result.activation_group_footprint_bytes =
      static_cast<std::uint64_t>(
          kSm87BulkV2NvFp4GateUpWholeP40RasterM) *
      kSm87BulkV2NvFp4GateUpWholeP40TileM *
      kSm87BulkV2NvFp4Hidden * sizeof(std::uint16_t);
  result.distinct_activation_bytes =
      static_cast<std::uint64_t>(
          kSm87BulkV2NvFp4GateUpWholeP40Tokens) *
      kSm87BulkV2NvFp4Hidden * sizeof(std::uint16_t);
  result.branch_b_bytes_per_n_tile = kBranchBytesPerNTile;
  result.paired_b_bytes_per_n_tile = kPairBytesPerNTile;
  result.logical_cta_b_request_bytes =
      static_cast<std::uint64_t>(
          kSm87BulkV2NvFp4GateUpWholeP40LogicalCells) *
      kPairBytesPerNTile;
  result.theoretical_l2_b_service_bytes =
      static_cast<std::uint64_t>(
          kSm87BulkV2NvFp4GateUpWholeP40MGroups) *
      kSm87BulkV2NvFp4GateUpWholeP40NTiles * kPairBytesPerNTile;
  result.m_group_outer = true;
  result.one_a_g2s_per_k64_stage = true;
  result.independent_gate_up_b_and_scales = true;
  result.theoretical_not_measured = true;
  // M-group-major traversal creates the 2.5-MiB reuse opportunity, but does
  // not establish that A survives all 34 N cohorts.  Only a retained NCU
  // capture may promote that hypothesis to measured residency.
  result.activation_group_l2_residency_unproven = true;
  return result;
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_nvfp4_gate_up_whole_p40_traffic_valid(
    const Sm87BulkV2NvFp4GateUpWholeP40Traffic& traffic) noexcept {
  return traffic.physical_launches == 1U &&
         traffic.legacy_m1024_segments == 0U &&
         traffic.m_groups == 157U && traffic.n_groups == 34U &&
         traffic.cohorts == 5'338U &&
         traffic.logical_cells == 170'000U &&
         traffic.scheduled_cells == 170'816U &&
         traffic.masked_cells == 816U &&
         traffic.full_cohort_b_requesters == 4U &&
         traffic.full_cohort_distinct_b_pairs == 1U &&
         traffic.theoretical_full_cohort_b_reuse == 4U &&
         traffic.grid_syncs_per_launch == 2U &&
         traffic.activation_group_footprint_bytes == 2'621'440ULL &&
         traffic.distinct_activation_bytes == 409'600'000ULL &&
         traffic.branch_b_bytes_per_n_tile == 184'320ULL &&
         traffic.paired_b_bytes_per_n_tile == 368'640ULL &&
         traffic.logical_cta_b_request_bytes == 62'668'800'000ULL &&
         traffic.theoretical_l2_b_service_bytes ==
             static_cast<std::uint64_t>(157U) *
                 kSm87BulkV2NvFp4GateUpPayloadBytes &&
         traffic.m_group_outer && traffic.one_a_g2s_per_k64_stage &&
         traffic.independent_gate_up_b_and_scales &&
         traffic.theoretical_not_measured &&
         traffic.activation_group_l2_residency_unproven;
}

struct Sm87BulkV2NvFp4GateUpWholeP40HostContract final {
  const char* identity = nullptr;
  Sm87BulkV2NvFp4GateUpWholeP40Traffic traffic{};
  std::uint32_t persistent_ctas = 0U;
  std::uint32_t threads = 0U;
  std::uint32_t pipeline_stages = 0U;
  std::uint32_t register_stages = 0U;
  std::uint32_t dynamic_shared_bytes = 0U;
  bool deterministic_full_grid_mapping = false;
  bool snake_n_group_order = false;
  bool no_cohort_grid_barrier = false;
  bool no_split_k = false;
  bool no_global_partial_c = false;
  bool no_activation_quantization = false;
  bool no_cublaslt = false;
  bool no_mtp = false;
  bool no_request_time_jit_repack_query = false;
  bool independent_default_off_identity = false;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr Sm87BulkV2NvFp4GateUpWholeP40HostContract
sm87_bulk_v2_nvfp4_gate_up_whole_p40_host_contract() noexcept {
  return {kSm87BulkV2NvFp4GateUpWholeP40Identity,
          sm87_bulk_v2_nvfp4_gate_up_whole_p40_traffic(),
          kSm87BulkV2NvFp4GateUpWholeP40PersistentCtas,
          kSm87BulkV2NvFp4GateUpWholeP40Threads,
          kSm87BulkV2NvFp4GateUpWholeP40PipelineStages,
          kSm87BulkV2NvFp4GateUpWholeP40RegisterStages,
          kSm87BulkV2NvFp4GateUpWholeP40DynamicSharedBytes,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          false};
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_nvfp4_gate_up_whole_p40_host_contract_valid(
    const Sm87BulkV2NvFp4GateUpWholeP40HostContract& contract) noexcept {
  return contract.identity == kSm87BulkV2NvFp4GateUpWholeP40Identity &&
         sm87_bulk_v2_nvfp4_gate_up_whole_p40_traffic_valid(
             contract.traffic) &&
         contract.persistent_ctas == 32U && contract.threads == 256U &&
         contract.pipeline_stages == 3U &&
         contract.register_stages == 2U &&
         contract.dynamic_shared_bytes == 38'400U &&
         contract.deterministic_full_grid_mapping &&
         contract.snake_n_group_order &&
         contract.no_cohort_grid_barrier && contract.no_split_k &&
         contract.no_global_partial_c &&
         contract.no_activation_quantization && contract.no_cublaslt &&
         contract.no_mtp &&
         contract.no_request_time_jit_repack_query &&
         contract.independent_default_off_identity &&
         !contract.production_dispatch_eligible;
}

struct alignas(64) Sm87BulkV2NvFp4GateUpWholeP40DeviceControl final {
  std::uint64_t transaction_epoch = 0U;
  std::uint32_t expected_cells = 0U;
  std::uint32_t started_cells = 0U;
  std::uint32_t completed_cells = 0U;
  std::uint32_t completed_ctas = 0U;
  std::uint32_t cancellation_observed = 0U;
  std::uint32_t launch_completed = 0U;
  std::uint32_t first_incomplete_cohort = 0U;
  std::uint32_t reserved[6U]{};
};

static_assert(sizeof(Sm87BulkV2NvFp4GateUpWholeP40DeviceControl) == 64U);
static_assert(alignof(Sm87BulkV2NvFp4GateUpWholeP40DeviceControl) == 64U);

struct Sm87BulkV2NvFp4GateUpWholeP40Arguments final {
  std::uint64_t transaction_epoch = 0U;
  const std::uint16_t* normalized_input = nullptr;
  std::uint16_t* h = nullptr;
  Sm87BulkV2NvFp4GateUpWholeP40DeviceControl* device_control = nullptr;
  const std::uint32_t* cancellation_signal = nullptr;
  Sm87TargetAotNvFp4CudaAssetView gate_up_asset{};
  void* cuda_stream = nullptr;
};

struct Sm87BulkV2NvFp4GateUpWholeP40PublicByteRanges final {
  Sm87TargetAotNvFp4CudaByteRange input{};
  Sm87TargetAotNvFp4CudaByteRange h{};
  Sm87TargetAotNvFp4CudaByteRange payload{};
  Sm87TargetAotNvFp4CudaByteRange control{};
  Sm87TargetAotNvFp4CudaByteRange cancellation{};
  bool cancellation_present = false;
};

// Pure public-ABI guard.  It proves only fixed byte spans, address arithmetic,
// alignment and pairwise non-aliasing; it performs no CUDA query and grants no
// ownership, residency or production authority.
[[nodiscard]] constexpr Sm87BulkV2NvFp4GateUpWholeP40PublicByteRanges
sm87_bulk_v2_nvfp4_gate_up_whole_p40_public_byte_ranges(
    const std::uint16_t* const input, std::uint16_t* const h,
    const std::uintptr_t payload_begin, const std::uintptr_t payload_end,
    const bool payload_valid,
    Sm87BulkV2NvFp4GateUpWholeP40DeviceControl* const control,
    const std::uint32_t* const cancellation) noexcept {
  Sm87BulkV2NvFp4GateUpWholeP40PublicByteRanges ranges;
  ranges.input = sm87_target_aot_nvfp4_cuda_byte_range(
      input, kSm87BulkV2NvFp4GateUpWholeP40InputBytes);
  ranges.h = sm87_target_aot_nvfp4_cuda_byte_range(
      h, kSm87BulkV2NvFp4GateUpWholeP40HBytes);
  ranges.payload = {
      payload_begin, payload_end,
      payload_valid && payload_begin != 0U && payload_end > payload_begin &&
          payload_end - payload_begin ==
              kSm87BulkV2NvFp4GateUpPayloadBytes};
  ranges.control = sm87_target_aot_nvfp4_cuda_byte_range(
      control, sizeof(Sm87BulkV2NvFp4GateUpWholeP40DeviceControl));
  ranges.cancellation_present = cancellation != nullptr;
  if (ranges.cancellation_present) {
    ranges.cancellation = sm87_target_aot_nvfp4_cuda_byte_range(
        cancellation, sizeof(std::uint32_t));
  }
  return ranges;
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_nvfp4_gate_up_whole_p40_public_byte_ranges_valid(
    const Sm87BulkV2NvFp4GateUpWholeP40PublicByteRanges& ranges) noexcept {
  const bool required_alignment =
      ranges.input.valid && ranges.h.valid && ranges.payload.valid &&
      ranges.control.valid && ranges.input.begin % 16U == 0U &&
      ranges.h.begin % 16U == 0U && ranges.payload.begin % 16U == 0U &&
      ranges.control.begin %
              alignof(Sm87BulkV2NvFp4GateUpWholeP40DeviceControl) ==
          0U;
  if (!required_alignment ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(ranges.input, ranges.h) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(ranges.input,
                                                ranges.payload) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(ranges.input,
                                                ranges.control) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(ranges.h, ranges.payload) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(ranges.h, ranges.control) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(ranges.payload,
                                                ranges.control)) {
    return false;
  }
  if (!ranges.cancellation_present) {
    return true;
  }
  return ranges.cancellation.valid && ranges.cancellation.begin % 4U == 0U &&
         !sm87_target_aot_nvfp4_cuda_ranges_overlap(ranges.cancellation,
                                                    ranges.input) &&
         !sm87_target_aot_nvfp4_cuda_ranges_overlap(ranges.cancellation,
                                                    ranges.h) &&
         !sm87_target_aot_nvfp4_cuda_ranges_overlap(ranges.cancellation,
                                                    ranges.payload) &&
         !sm87_target_aot_nvfp4_cuda_ranges_overlap(ranges.cancellation,
                                                    ranges.control);
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_nvfp4_gate_up_whole_p40_public_arguments_ranges_valid(
    const Sm87BulkV2NvFp4GateUpWholeP40Arguments& arguments) noexcept {
  return sm87_bulk_v2_nvfp4_gate_up_whole_p40_public_byte_ranges_valid(
      sm87_bulk_v2_nvfp4_gate_up_whole_p40_public_byte_ranges(
          arguments.normalized_input, arguments.h,
          arguments.gate_up_asset.payload.begin,
          arguments.gate_up_asset.payload.end,
          arguments.gate_up_asset.payload.valid, arguments.device_control,
          arguments.cancellation_signal));
}

struct Sm87BulkV2NvFp4GateUpWholeP40CodeEvidence final {
  std::uint64_t elf_identity = 0U;
  std::uint64_t canonical_sass_hash = 0U;
  std::uint32_t instruction_rows = 0U;
  std::uint32_t text_bytes = 0U;
  std::size_t stack_bytes = 0U;
  std::size_t spill_store_bytes = 0U;
  std::size_t spill_load_bytes = 0U;
  std::size_t local_bytes = 0U;
  bool contains_cp_async_cg = false;
  bool contains_ldmatrix = false;
  bool contains_bf16_mma = false;
  bool one_cooperative_kernel_symbol = false;
  bool same_elf_exact_oracle = false;
  bool valid = false;
};

[[nodiscard]] constexpr bool
sm87_bulk_v2_nvfp4_gate_up_whole_p40_code_evidence_valid(
    const Sm87BulkV2NvFp4GateUpWholeP40CodeEvidence& evidence) noexcept {
  return evidence.elf_identity != 0U &&
         evidence.canonical_sass_hash != 0U &&
         evidence.instruction_rows != 0U && evidence.text_bytes != 0U &&
         evidence.stack_bytes == 0U && evidence.spill_store_bytes == 0U &&
         evidence.spill_load_bytes == 0U && evidence.local_bytes == 0U &&
         evidence.contains_cp_async_cg && evidence.contains_ldmatrix &&
         evidence.contains_bf16_mma &&
         evidence.one_cooperative_kernel_symbol &&
         evidence.same_elf_exact_oracle && evidence.valid;
}

struct Sm87BulkV2NvFp4GateUpWholeP40Resources final {
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int cooperative_grid_capacity = 0;
  Sm87BulkV2NvFp4GateUpWholeP40CodeEvidence code{};
  bool cooperative_launch_supported = false;
  bool exact_oracle_attached = false;
  bool resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr bool
sm87_bulk_v2_nvfp4_gate_up_whole_p40_resources_valid(
    const Sm87BulkV2NvFp4GateUpWholeP40Resources& resources) noexcept {
  return resources.binary_version == 87 &&
         resources.registers_per_thread > 0 &&
         resources.registers_per_thread <=
             static_cast<int>(
                 kSm87BulkV2NvFp4GateUpWholeP40MaximumRegisters) &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kSm87BulkV2NvFp4GateUpWholeP40DynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >= 256 &&
         resources.active_blocks_per_sm >= 2 &&
         resources.cooperative_grid_capacity >= 32 &&
         sm87_bulk_v2_nvfp4_gate_up_whole_p40_code_evidence_valid(
             resources.code) &&
         resources.cooperative_launch_supported &&
         resources.exact_oracle_attached &&
         resources.resource_gate_passed &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

// Read-only admission query.  It is never called from the request path and
// cannot promote the candidate. CodeEvidence is a forgeable value record, not
// a capability: the query may echo it for inspection but never sets
// resource_gate_passed from caller-provided fields. Only an external admission
// step that authenticates retained ELF/SASS hashes may combine those facts.
[[nodiscard]] int
query_sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40_resources_cuda(
    const Sm87BulkV2NvFp4GateUpWholeP40CodeEvidence* code_evidence,
    Sm87BulkV2NvFp4GateUpWholeP40Resources* resources) noexcept;

// Prevalidated, default-off whole-P40 enqueue.  The function performs no
// device, pointer, occupancy, function-resource, JIT, repack, or tactic query.
// It submits exactly one 32-CTA cooperative kernel.  A future private owner
// must seal the asset/pointers before exposing this seam to an executor.
[[nodiscard]] int
launch_sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40_cuda(
    const Sm87BulkV2NvFp4GateUpWholeP40Arguments& arguments) noexcept;

inline constexpr auto kSm87BulkV2NvFp4GateUpWholeP40FrozenContract =
    sm87_bulk_v2_nvfp4_gate_up_whole_p40_host_contract();

static_assert(kSm87BulkV2NvFp4GateUpWholeP40MTiles == 625U);
static_assert(kSm87BulkV2NvFp4GateUpWholeP40NTiles == 272U);
static_assert(kSm87BulkV2NvFp4GateUpWholeP40KTiles == 80U);
static_assert(kSm87BulkV2NvFp4GateUpWholeP40PersistentCtas ==
              kSm87BulkV2NvFp4GateUpWholeP40RasterM *
                  kSm87BulkV2NvFp4GateUpWholeP40RasterN);
static_assert(kSm87BulkV2NvFp4GateUpWholeP40PipelineSharedBytes == 38'400U);
static_assert(kSm87BulkV2NvFp4GateUpWholeP40DynamicSharedBytes == 38'400U);
static_assert(kSm87BulkV2NvFp4GateUpWholeP40InputBytes == 409'600'000ULL);
static_assert(kSm87BulkV2NvFp4GateUpWholeP40HBytes == 1'392'640'000ULL);
static_assert(sm87_bulk_v2_nvfp4_gate_up_whole_p40_host_contract_valid(
    kSm87BulkV2NvFp4GateUpWholeP40FrozenContract));

}  // namespace q3x::kernels
