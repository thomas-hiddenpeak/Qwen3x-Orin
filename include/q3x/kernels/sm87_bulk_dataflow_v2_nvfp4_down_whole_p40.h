#pragma once

#include "q3x/kernels/sm87_target_aot_projection_cuda.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off K-heavy Down successor for
// AC-PREFILL-SM87-BULK-DATAFLOW-v2.  Unlike the closed joint M256 control,
// one cooperative launch owns the complete P40000 Down role.  The 32
// resident CTAs form an explicit 8M x 4N L2 service cohort; every CTA keeps
// one complete, ascending-K output tile and no K slice is ever published.
inline constexpr std::array<std::uint8_t, 8U>
    kSm87BulkV2NvFp4DownWholeP40Magic{
        {'Q', '3', 'X', 'D', 'W', 'P', '4', '0'}};
inline constexpr std::uint16_t
    kSm87BulkV2NvFp4DownWholeP40AbiMajor = 1U;
inline constexpr std::uint16_t
    kSm87BulkV2NvFp4DownWholeP40AbiMinor = 0U;

inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40Tokens = 40'000U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40Intermediate = 17'408U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40Hidden = 5'120U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DownWholeP40TileM = 64U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DownWholeP40TileN = 256U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DownWholeP40TileK = 64U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DownWholeP40MTiles = 625U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DownWholeP40NTiles = 20U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DownWholeP40KTiles = 272U;

inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40CohortM = 8U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40CohortN = 4U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40PersistentCtas = 32U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40Threads = 256U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40RequiredCtasPerSm = 2U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40SmCount = 16U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40MaximumRegisters = 128U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40PipelineStages = 3U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40RegisterStages = 2U;
// ABI identity of the exact CUDA kernel symbol whose >48-KiB dynamic-shared
// opt-in is established during startup resource inspection.  It is not an
// address and grants no authority by itself; the startup capability below
// binds the attribute receipt, SASS evidence, device and launch resource
// envelope to one symbol identity.
inline constexpr std::uint64_t
    kSm87BulkV2NvFp4DownWholeP40KernelSymbolIdentity =
        0x4457'5034'304b'3031ULL;

inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40MCohorts = 79U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40NCohorts = 5U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40SuperWaves = 395U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40FullMCohorts = 78U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40TailMRows = 1U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40LogicalOutputTiles = 12'500U;

inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40ActivationBytesPerStage = 8'192U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40WeightBytesPerStage = 8'192U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40ScaleBytesPerStage = 1'024U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40BytesPerStage = 17'408U;
inline constexpr std::uint32_t
    kSm87BulkV2NvFp4DownWholeP40DynamicSharedBytes = 52'224U;
inline constexpr std::uint64_t
    kSm87BulkV2NvFp4DownWholeP40PayloadBytes = 50'135'040ULL;
inline constexpr std::uint64_t kSm87BulkV2NvFp4DownWholeP40HBytes =
    static_cast<std::uint64_t>(kSm87BulkV2NvFp4DownWholeP40Tokens) *
    kSm87BulkV2NvFp4DownWholeP40Intermediate * sizeof(std::uint16_t);
inline constexpr std::uint64_t
    kSm87BulkV2NvFp4DownWholeP40ResidualBytes =
        static_cast<std::uint64_t>(kSm87BulkV2NvFp4DownWholeP40Tokens) *
        kSm87BulkV2NvFp4DownWholeP40Hidden * sizeof(std::uint16_t);

// Useful H requests have exact four-way address equivalence inside each 4N
// cohort. Full 8M cohorts have exact eight-way B address equivalence. The
// final one-row M cohort masks seven CTAs before load/MMA/store; it performs
// no padded projection work and is accounted separately.
inline constexpr std::uint64_t
    kSm87BulkV2NvFp4DownWholeP40UsefulHRequests =
        static_cast<std::uint64_t>(
            kSm87BulkV2NvFp4DownWholeP40LogicalOutputTiles) *
        kSm87BulkV2NvFp4DownWholeP40KTiles;
inline constexpr std::uint64_t
    kSm87BulkV2NvFp4DownWholeP40UniqueHServices =
        static_cast<std::uint64_t>(
            kSm87BulkV2NvFp4DownWholeP40MTiles) *
        kSm87BulkV2NvFp4DownWholeP40NCohorts *
        kSm87BulkV2NvFp4DownWholeP40KTiles;
inline constexpr std::uint64_t
    kSm87BulkV2NvFp4DownWholeP40FullCohortBRequests =
        static_cast<std::uint64_t>(
            kSm87BulkV2NvFp4DownWholeP40FullMCohorts) *
        kSm87BulkV2NvFp4DownWholeP40CohortM *
        kSm87BulkV2NvFp4DownWholeP40NTiles *
        kSm87BulkV2NvFp4DownWholeP40KTiles;
inline constexpr std::uint64_t
    kSm87BulkV2NvFp4DownWholeP40FullCohortUniqueBServices =
        static_cast<std::uint64_t>(
            kSm87BulkV2NvFp4DownWholeP40FullMCohorts) *
        kSm87BulkV2NvFp4DownWholeP40NTiles *
        kSm87BulkV2NvFp4DownWholeP40KTiles;
inline constexpr std::uint64_t
    kSm87BulkV2NvFp4DownWholeP40TailBRequests =
        static_cast<std::uint64_t>(
            kSm87BulkV2NvFp4DownWholeP40TailMRows) *
        kSm87BulkV2NvFp4DownWholeP40NTiles *
        kSm87BulkV2NvFp4DownWholeP40KTiles;

static_assert(kSm87BulkV2NvFp4DownWholeP40MTiles *
                  kSm87BulkV2NvFp4DownWholeP40TileM ==
              kSm87BulkV2NvFp4DownWholeP40Tokens);
static_assert(kSm87BulkV2NvFp4DownWholeP40NTiles *
                  kSm87BulkV2NvFp4DownWholeP40TileN ==
              kSm87BulkV2NvFp4DownWholeP40Hidden);
static_assert(kSm87BulkV2NvFp4DownWholeP40KTiles *
                  kSm87BulkV2NvFp4DownWholeP40TileK ==
              kSm87BulkV2NvFp4DownWholeP40Intermediate);
static_assert(kSm87BulkV2NvFp4DownWholeP40CohortM *
                  kSm87BulkV2NvFp4DownWholeP40CohortN ==
              kSm87BulkV2NvFp4DownWholeP40PersistentCtas);
static_assert(kSm87BulkV2NvFp4DownWholeP40UsefulHRequests ==
              4ULL * kSm87BulkV2NvFp4DownWholeP40UniqueHServices);
static_assert(kSm87BulkV2NvFp4DownWholeP40FullCohortBRequests ==
              8ULL *
                  kSm87BulkV2NvFp4DownWholeP40FullCohortUniqueBServices);

enum class Sm87BulkV2NvFp4DownWholeP40ExecutionIdentity : std::uint16_t {
  kInvalid = 0U,
  kPersistent32CtaM64N256K64Cohort8M4NExactV1,
};

enum Sm87BulkV2NvFp4DownWholeP40Policy : std::uint64_t {
  kSm87BulkV2DownWholeP40Bf16Activation = 1ULL << 0U,
  kSm87BulkV2DownWholeP40Fp32Accumulation = 1ULL << 1U,
  kSm87BulkV2DownWholeP40AscendingFullK = 1ULL << 2U,
  kSm87BulkV2DownWholeP40NoSplitK = 1ULL << 3U,
  kSm87BulkV2DownWholeP40TensorScaleThenBf16Rne = 1ULL << 4U,
  kSm87BulkV2DownWholeP40Bf16ResidualThenBf16Rne = 1ULL << 5U,
  kSm87BulkV2DownWholeP40ThreeStageCpAsyncCg = 1ULL << 6U,
  kSm87BulkV2DownWholeP40TwoStageS2R = 1ULL << 7U,
  kSm87BulkV2DownWholeP40Cohort8M4N = 1ULL << 8U,
  kSm87BulkV2DownWholeP40OneWholeLayerLaunch = 1ULL << 9U,
  kSm87BulkV2DownWholeP40NoActivationQuantization = 1ULL << 10U,
  kSm87BulkV2DownWholeP40NoRequestJit = 1ULL << 11U,
  kSm87BulkV2DownWholeP40NoRequestRepack = 1ULL << 12U,
  kSm87BulkV2DownWholeP40NoHotCudaQuery = 1ULL << 13U,
  kSm87BulkV2DownWholeP40NoCublasLt = 1ULL << 14U,
  kSm87BulkV2DownWholeP40NoMtp = 1ULL << 15U,
  kSm87BulkV2DownWholeP40NoProductionSelector = 1ULL << 16U,
};

inline constexpr std::uint64_t
    kSm87BulkV2NvFp4DownWholeP40RequiredPolicy =
        kSm87BulkV2DownWholeP40Bf16Activation |
        kSm87BulkV2DownWholeP40Fp32Accumulation |
        kSm87BulkV2DownWholeP40AscendingFullK |
        kSm87BulkV2DownWholeP40NoSplitK |
        kSm87BulkV2DownWholeP40TensorScaleThenBf16Rne |
        kSm87BulkV2DownWholeP40Bf16ResidualThenBf16Rne |
        kSm87BulkV2DownWholeP40ThreeStageCpAsyncCg |
        kSm87BulkV2DownWholeP40TwoStageS2R |
        kSm87BulkV2DownWholeP40Cohort8M4N |
        kSm87BulkV2DownWholeP40OneWholeLayerLaunch |
        kSm87BulkV2DownWholeP40NoActivationQuantization |
        kSm87BulkV2DownWholeP40NoRequestJit |
        kSm87BulkV2DownWholeP40NoRequestRepack |
        kSm87BulkV2DownWholeP40NoHotCudaQuery |
        kSm87BulkV2DownWholeP40NoCublasLt |
        kSm87BulkV2DownWholeP40NoMtp |
        kSm87BulkV2DownWholeP40NoProductionSelector;

struct Sm87BulkV2NvFp4DownWholeP40WorkItem final {
  std::uint32_t super_wave = 0U;
  std::uint32_t block = 0U;
  std::uint32_t m_cohort = 0U;
  std::uint32_t n_cohort = 0U;
  std::uint32_t m_lane = 0U;
  std::uint32_t n_lane = 0U;
  std::uint32_t m_tile = 0U;
  std::uint32_t service_m_tile = 0U;
  std::uint32_t n_tile = 0U;
  bool output_owner = false;
  bool h_request_valid = false;
  bool b_service_participant = false;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2NvFp4DownWholeP40WorkItem
sm87_bulk_v2_nvfp4_down_whole_p40_work_item(
    const std::size_t super_wave, const std::size_t block) noexcept {
  if (super_wave >= kSm87BulkV2NvFp4DownWholeP40SuperWaves ||
      block >= kSm87BulkV2NvFp4DownWholeP40PersistentCtas) {
    return {};
  }
  const auto wave = static_cast<std::uint32_t>(super_wave);
  const auto cta = static_cast<std::uint32_t>(block);
  const std::uint32_t m_cohort =
      wave / kSm87BulkV2NvFp4DownWholeP40NCohorts;
  const std::uint32_t n_cohort =
      wave % kSm87BulkV2NvFp4DownWholeP40NCohorts;
  const std::uint32_t m_lane =
      cta / kSm87BulkV2NvFp4DownWholeP40CohortN;
  const std::uint32_t n_lane =
      cta % kSm87BulkV2NvFp4DownWholeP40CohortN;
  const std::uint32_t m_tile =
      m_cohort * kSm87BulkV2NvFp4DownWholeP40CohortM + m_lane;
  const std::uint32_t n_tile =
      n_cohort * kSm87BulkV2NvFp4DownWholeP40CohortN + n_lane;
  const bool output_owner =
      m_tile < kSm87BulkV2NvFp4DownWholeP40MTiles &&
      n_tile < kSm87BulkV2NvFp4DownWholeP40NTiles;
  // Only the final M cohort has invalid lanes. They remain in the fixed outer
  // schedule but issue no loads, MMA, or stores.
  const std::uint32_t service_m = m_tile;
  return {wave, cta, m_cohort, n_cohort, m_lane, n_lane, m_tile,
          service_m, n_tile, output_owner, output_owner, output_owner, true};
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_nvfp4_down_whole_p40_same_h_service(
    const Sm87BulkV2NvFp4DownWholeP40WorkItem& left,
    const Sm87BulkV2NvFp4DownWholeP40WorkItem& right) noexcept {
  return left.valid && right.valid && left.super_wave == right.super_wave &&
         left.service_m_tile == right.service_m_tile;
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_nvfp4_down_whole_p40_same_b_service(
    const Sm87BulkV2NvFp4DownWholeP40WorkItem& left,
    const Sm87BulkV2NvFp4DownWholeP40WorkItem& right) noexcept {
  return left.valid && right.valid && left.super_wave == right.super_wave &&
         left.n_tile == right.n_tile;
}

struct Sm87BulkV2NvFp4DownWholeP40TrafficContract final {
  std::uint64_t useful_output_tiles = 0U;
  std::uint64_t useful_h_requests = 0U;
  std::uint64_t unique_h_services = 0U;
  std::uint64_t full_cohort_b_requests = 0U;
  std::uint64_t full_cohort_unique_b_services = 0U;
  std::uint64_t tail_b_requests = 0U;
  std::uint64_t masked_tail_lanes = 0U;
  std::uint32_t physical_launches = 0U;
  std::uint32_t cooperative_grid_barriers = 0U;
  std::uint32_t cooperative_grid_barriers_per_super_wave = 0U;
  bool exact_unique_coverage = false;
  bool four_way_h_cohort = false;
  bool eight_way_b_cohort = false;
  bool tail_publication_safe = false;
  bool theoretical_l2_service_only = false;
  bool measured_cross_cta_residency = false;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2NvFp4DownWholeP40TrafficContract
sm87_bulk_v2_nvfp4_down_whole_p40_traffic_contract() noexcept {
  return {kSm87BulkV2NvFp4DownWholeP40LogicalOutputTiles,
          kSm87BulkV2NvFp4DownWholeP40UsefulHRequests,
          kSm87BulkV2NvFp4DownWholeP40UniqueHServices,
          kSm87BulkV2NvFp4DownWholeP40FullCohortBRequests,
          kSm87BulkV2NvFp4DownWholeP40FullCohortUniqueBServices,
          kSm87BulkV2NvFp4DownWholeP40TailBRequests,
          static_cast<std::uint64_t>(
              kSm87BulkV2NvFp4DownWholeP40CohortM -
              kSm87BulkV2NvFp4DownWholeP40TailMRows),
          1U,
          2U,
          0U,
          true,
          kSm87BulkV2NvFp4DownWholeP40UsefulHRequests ==
              4ULL * kSm87BulkV2NvFp4DownWholeP40UniqueHServices,
          kSm87BulkV2NvFp4DownWholeP40FullCohortBRequests ==
              8ULL *
                  kSm87BulkV2NvFp4DownWholeP40FullCohortUniqueBServices,
          true,
          true,
          false,
          true};
}

struct alignas(64) Sm87BulkV2NvFp4DownWholeP40DeviceControl final {
  std::uint64_t transaction_epoch = 0U;
  std::uint32_t requested_m_tiles = 0U;
  std::uint32_t requested_n_tiles = 0U;
  std::uint32_t completed_super_waves = 0U;
  std::uint32_t completed_output_tiles = 0U;
  std::uint32_t cancellation_observed = 0U;
  std::uint32_t wave_cancelled = 0U;
  std::uint32_t first_unfinished_super_wave = 0U;
  std::uint32_t error_code = 0U;
  std::uint64_t policy = 0U;
  std::array<std::uint64_t, 2U> reserved{};
  std::array<std::uint32_t,
             kSm87BulkV2NvFp4DownWholeP40PersistentCtas>
      cta_completed_super_waves{};
};

static_assert(sizeof(Sm87BulkV2NvFp4DownWholeP40DeviceControl) == 192U);

struct Sm87BulkV2NvFp4DownWholeP40CodeEvidence final {
  std::uint64_t elf_identity = 0U;
  std::uint64_t canonical_sass_hash = 0U;
  std::uint32_t instruction_rows = 0U;
  std::uint32_t text_bytes = 0U;
  std::uint32_t stack_frame_bytes = 0U;
  std::uint32_t spill_store_bytes = 0U;
  std::uint32_t spill_load_bytes = 0U;
  std::uint32_t local_load_store_rows = 0U;
  bool launch_bounds_256_2 = false;
  bool cooperative_grid_sync_present = false;
  bool cp_async_cg_present = false;
  bool two_stage_s2r_present = false;
  bool full_k_accumulator_present = false;
  bool split_k_or_partial_c_absent = false;
};

[[nodiscard]] constexpr bool
sm87_bulk_v2_nvfp4_down_whole_p40_code_evidence_valid(
    const Sm87BulkV2NvFp4DownWholeP40CodeEvidence& evidence) noexcept {
  return evidence.elf_identity != 0U &&
         evidence.canonical_sass_hash != 0U &&
         evidence.instruction_rows != 0U && evidence.text_bytes != 0U &&
         evidence.stack_frame_bytes == 0U &&
         evidence.spill_store_bytes == 0U &&
         evidence.spill_load_bytes == 0U &&
         evidence.local_load_store_rows == 0U &&
         evidence.launch_bounds_256_2 &&
         evidence.cooperative_grid_sync_present &&
         evidence.cp_async_cg_present && evidence.two_stage_s2r_present &&
         evidence.full_k_accumulator_present &&
         evidence.split_k_or_partial_c_absent;
}

struct Sm87BulkV2NvFp4DownWholeP40Resources final {
  std::uint64_t kernel_symbol_identity = 0U;
  int device_ordinal = -1;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int cooperative_grid_capacity = 0;
  Sm87BulkV2NvFp4DownWholeP40CodeEvidence code{};
  bool kernel_compiled = false;
  bool cooperative_launch_supported = false;
  bool dynamic_shared_attribute_configured = false;
  bool resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool
sm87_bulk_v2_nvfp4_down_whole_p40_resources_valid(
    const Sm87BulkV2NvFp4DownWholeP40Resources& resources) noexcept {
  return resources.kernel_symbol_identity ==
             kSm87BulkV2NvFp4DownWholeP40KernelSymbolIdentity &&
         resources.device_ordinal >= 0 && resources.binary_version == 87 &&
         resources.registers_per_thread > 0 &&
         resources.registers_per_thread <=
             static_cast<int>(
                 kSm87BulkV2NvFp4DownWholeP40MaximumRegisters) &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kSm87BulkV2NvFp4DownWholeP40DynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<int>(kSm87BulkV2NvFp4DownWholeP40Threads) &&
         resources.active_blocks_per_sm >=
             static_cast<int>(
                 kSm87BulkV2NvFp4DownWholeP40RequiredCtasPerSm) &&
         resources.cooperative_grid_capacity >=
             static_cast<int>(
                 kSm87BulkV2NvFp4DownWholeP40PersistentCtas) &&
         sm87_bulk_v2_nvfp4_down_whole_p40_code_evidence_valid(
             resources.code) &&
         resources.kernel_compiled &&
         resources.cooperative_launch_supported &&
         resources.dynamic_shared_attribute_configured &&
         resources.resource_gate_passed &&
         !resources.production_dispatch_eligible;
}

// Startup/offline resource inspection only. This call sets the 52,224-byte
// dynamic-shared opt-in on the exact kernel symbol and records that fact in
// Resources after checking the same ELF's attributes and occupancy.  This is
// still only a forgeable observation record: caller-provided CodeEvidence can
// never mint resource_gate_passed.  A future private startup owner must bind
// the retained ELF/SASS evidence to the configured symbol.  The execution
// launcher below never calls this query and therefore performs no request-time
// CUDA property, occupancy, function-attribute, or pointer query.
[[nodiscard]] int
query_sm87_bulk_dataflow_v2_nvfp4_down_whole_p40_resources_cuda(
    const Sm87BulkV2NvFp4DownWholeP40CodeEvidence* code_evidence,
    Sm87BulkV2NvFp4DownWholeP40Resources* resources) noexcept;

// Default-off launch arguments. The asset is the existing authenticated
// target-AOT Down byte view; this successor consumes it in-place and does not
// repack or materialize decoded weights.  The process must have configured the
// exact kernel symbol's >48-KiB dynamic-shared opt-in during private startup;
// the public enqueue does not accept a forgeable Resources value as proof.
struct Sm87BulkV2NvFp4DownWholeP40Arguments final {
  std::uint64_t transaction_epoch = 0U;
  const std::uint16_t* h = nullptr;
  std::uint16_t* residual = nullptr;
  Sm87BulkV2NvFp4DownWholeP40DeviceControl* device_control = nullptr;
  const std::uint32_t* cancellation_signal = nullptr;
  Sm87TargetAotNvFp4CudaAssetView down_asset{};
  void* cuda_stream = nullptr;
};

// One and only one whole-layer cooperative launch. This default-off seam is
// not a production selector and does not grant owner/seal authority.
[[nodiscard]] int
launch_sm87_bulk_dataflow_v2_nvfp4_down_whole_p40_cuda(
    const Sm87BulkV2NvFp4DownWholeP40Arguments& arguments) noexcept;

}  // namespace q3x::kernels
