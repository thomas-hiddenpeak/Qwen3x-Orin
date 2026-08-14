#pragma once

#include "q3x/kernels/sm87_target_aot_projection_fp8_cuda.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Independent whole-prompt FP8 successor for
// AC-PREFILL-SM87-BULK-DATAFLOW-v2.  Each fixed model role owns one complete
// P40000 cooperative launch.  This admission consumes the authenticated
// target-AOT FP8 payload unchanged and is deliberately unreachable from every
// production selector.
inline constexpr char kSm87BulkV2Fp8WholeP40Identity[] =
    "q3x.sm87.bulk-v2.fp8.whole-p40000."
    "m64n128k64-raster-m4n8.v1";

inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40Tokens = 40'000U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40TileM = 64U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40TileN = 128U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40TileK = 64U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40MTiles = 625U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40CohortM = 4U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40CohortN = 8U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40PersistentCtas = 32U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40Threads = 256U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40Warps = 8U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40PipelineStages = 3U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40RegisterStages = 2U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40RequiredCtasPerSm = 2U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40MaximumRegisters = 128U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40SmCount = 16U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40LogicalRoles = 128U;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40PhysicalLaunches = 128U;

inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40ABytesPerStage =
    kSm87BulkV2Fp8WholeP40TileM * kSm87BulkV2Fp8WholeP40TileK *
    sizeof(std::uint16_t);
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40BBytesPerStage =
    kSm87BulkV2Fp8WholeP40TileN * kSm87BulkV2Fp8WholeP40TileK;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40BytesPerStage =
    kSm87BulkV2Fp8WholeP40ABytesPerStage +
    kSm87BulkV2Fp8WholeP40BBytesPerStage;
inline constexpr std::uint32_t kSm87BulkV2Fp8WholeP40DynamicSharedBytes =
    kSm87BulkV2Fp8WholeP40PipelineStages *
    kSm87BulkV2Fp8WholeP40BytesPerStage;

static_assert(kSm87BulkV2Fp8WholeP40PersistentCtas ==
              kSm87BulkV2Fp8WholeP40CohortM *
                  kSm87BulkV2Fp8WholeP40CohortN);
static_assert(kSm87BulkV2Fp8WholeP40DynamicSharedBytes == 49'152U);

enum class Sm87BulkV2Fp8WholeP40ExecutionIdentity : std::uint16_t {
  kInvalid = 0U,
  kPersistent32CtaM64N128K64Cohort4M8NExactV1,
};

enum Sm87BulkV2Fp8WholeP40Policy : std::uint64_t {
  kSm87BulkV2Fp8WholeP40Bf16Activation = 1ULL << 0U,
  kSm87BulkV2Fp8WholeP40RawMarlinFp8Payload = 1ULL << 1U,
  kSm87BulkV2Fp8WholeP40Fp32Accumulation = 1ULL << 2U,
  kSm87BulkV2Fp8WholeP40AscendingFullK = 1ULL << 3U,
  kSm87BulkV2Fp8WholeP40PartitionPrivateScale = 1ULL << 4U,
  kSm87BulkV2Fp8WholeP40Bf16RnePublication = 1ULL << 5U,
  kSm87BulkV2Fp8WholeP40NoSplitK = 1ULL << 6U,
  kSm87BulkV2Fp8WholeP40NoGlobalPartialC = 1ULL << 7U,
  kSm87BulkV2Fp8WholeP40ThreeStageCpAsyncCg = 1ULL << 8U,
  kSm87BulkV2Fp8WholeP40TwoStageS2R = 1ULL << 9U,
  kSm87BulkV2Fp8WholeP40Cohort4M8N = 1ULL << 10U,
  kSm87BulkV2Fp8WholeP40OneWholeRoleLaunch = 1ULL << 11U,
  kSm87BulkV2Fp8WholeP40MaskedTailNoWork = 1ULL << 12U,
  kSm87BulkV2Fp8WholeP40NoRequestJit = 1ULL << 13U,
  kSm87BulkV2Fp8WholeP40NoRequestRepack = 1ULL << 14U,
  kSm87BulkV2Fp8WholeP40NoHotCudaQuery = 1ULL << 15U,
  kSm87BulkV2Fp8WholeP40NoActivationQuantization = 1ULL << 16U,
  kSm87BulkV2Fp8WholeP40NoCublasLt = 1ULL << 17U,
  kSm87BulkV2Fp8WholeP40NoMtp = 1ULL << 18U,
  kSm87BulkV2Fp8WholeP40NoProductionSelector = 1ULL << 19U,
  kSm87BulkV2Fp8WholeP40NoResidencyClaim = 1ULL << 20U,
};

inline constexpr std::uint64_t kSm87BulkV2Fp8WholeP40RequiredPolicy =
    kSm87BulkV2Fp8WholeP40Bf16Activation |
    kSm87BulkV2Fp8WholeP40RawMarlinFp8Payload |
    kSm87BulkV2Fp8WholeP40Fp32Accumulation |
    kSm87BulkV2Fp8WholeP40AscendingFullK |
    kSm87BulkV2Fp8WholeP40PartitionPrivateScale |
    kSm87BulkV2Fp8WholeP40Bf16RnePublication |
    kSm87BulkV2Fp8WholeP40NoSplitK |
    kSm87BulkV2Fp8WholeP40NoGlobalPartialC |
    kSm87BulkV2Fp8WholeP40ThreeStageCpAsyncCg |
    kSm87BulkV2Fp8WholeP40TwoStageS2R |
    kSm87BulkV2Fp8WholeP40Cohort4M8N |
    kSm87BulkV2Fp8WholeP40OneWholeRoleLaunch |
    kSm87BulkV2Fp8WholeP40MaskedTailNoWork |
    kSm87BulkV2Fp8WholeP40NoRequestJit |
    kSm87BulkV2Fp8WholeP40NoRequestRepack |
    kSm87BulkV2Fp8WholeP40NoHotCudaQuery |
    kSm87BulkV2Fp8WholeP40NoActivationQuantization |
    kSm87BulkV2Fp8WholeP40NoCublasLt |
    kSm87BulkV2Fp8WholeP40NoMtp |
    kSm87BulkV2Fp8WholeP40NoProductionSelector |
    kSm87BulkV2Fp8WholeP40NoResidencyClaim;

#if defined(__CUDACC__)
#define Q3X_SM87_BULK_V2_FP8_WHOLE_HD __host__ __device__
#else
#define Q3X_SM87_BULK_V2_FP8_WHOLE_HD
#endif

[[nodiscard]] Q3X_SM87_BULK_V2_FP8_WHOLE_HD constexpr std::uint16_t
sm87_bulk_v2_fp8_whole_p40_raw_code_to_biased_bf16_bits(
    const std::uint8_t code) noexcept {
  return static_cast<std::uint16_t>(
      (static_cast<std::uint16_t>(code & 0x80U) << 8U) |
      (static_cast<std::uint16_t>(code & 0x7fU) << 4U));
}

#undef Q3X_SM87_BULK_V2_FP8_WHOLE_HD

struct Sm87BulkV2Fp8WholeP40RolePlan final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::uint32_t input_features = 0U;
  std::uint32_t projected_output_features = 0U;
  std::uint32_t primary_output_features = 0U;
  std::uint32_t secondary_output_features = 0U;
  std::uint32_t tertiary_output_features = 0U;
  std::uint32_t partition_count = 0U;
  std::array<std::uint32_t, 3U> partition_n128_tiles{};
  std::array<std::uint64_t, 3U> partition_payload_offsets{};
  std::uint32_t k_tiles = 0U;
  std::uint32_t n_tiles = 0U;
  std::uint32_t m_groups = 0U;
  std::uint32_t n_groups = 0U;
  std::uint32_t cohorts = 0U;
  std::uint32_t logical_cells = 0U;
  std::uint32_t scheduled_cells = 0U;
  std::uint32_t masked_cells = 0U;
  std::uint64_t payload_bytes = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2Fp8WholeP40RolePlan
sm87_bulk_v2_fp8_whole_p40_role_plan(
    const Sm87TargetAotProjectionRole role) noexcept {
  Sm87BulkV2Fp8WholeP40RolePlan result;
  result.role = role;
  if (role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    result.input_features = 5'120U;
    result.projected_output_features = 16'384U;
    result.primary_output_features = 16'384U;
    result.partition_count = 2U;
    result.partition_n128_tiles = {80U, 48U, 0U};
    result.partition_payload_offsets = {0U, 52'428'800U, 0U};
    result.k_tiles = 80U;
    result.n_tiles = 128U;
    result.payload_bytes = 83'886'080ULL;
  } else if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    result.input_features = 5'120U;
    result.projected_output_features = 14'336U;
    result.primary_output_features = 12'288U;
    result.secondary_output_features = 1'024U;
    result.tertiary_output_features = 1'024U;
    result.partition_count = 3U;
    result.partition_n128_tiles = {96U, 8U, 8U};
    result.partition_payload_offsets =
        {0U, 62'914'560U, 68'157'440U};
    result.k_tiles = 80U;
    result.n_tiles = 112U;
    result.payload_bytes = 73'400'320ULL;
  } else if (role ==
             Sm87TargetAotProjectionRole::kFp8AttentionOutput) {
    result.input_features = 6'144U;
    result.projected_output_features = 5'120U;
    result.primary_output_features = 5'120U;
    result.partition_count = 1U;
    result.partition_n128_tiles = {40U, 0U, 0U};
    result.partition_payload_offsets = {0U, 0U, 0U};
    result.k_tiles = 96U;
    result.n_tiles = 40U;
    result.payload_bytes = 31'457'280ULL;
  } else {
    return result;
  }
  result.m_groups =
      (kSm87BulkV2Fp8WholeP40MTiles +
       kSm87BulkV2Fp8WholeP40CohortM - 1U) /
      kSm87BulkV2Fp8WholeP40CohortM;
  result.n_groups =
      (result.n_tiles + kSm87BulkV2Fp8WholeP40CohortN - 1U) /
      kSm87BulkV2Fp8WholeP40CohortN;
  result.cohorts = result.m_groups * result.n_groups;
  result.logical_cells =
      kSm87BulkV2Fp8WholeP40MTiles * result.n_tiles;
  result.scheduled_cells =
      result.cohorts * kSm87BulkV2Fp8WholeP40PersistentCtas;
  result.masked_cells = result.scheduled_cells - result.logical_cells;
  result.valid = true;
  return result;
}

struct Sm87BulkV2Fp8WholeP40FamilyRole final {
  std::uint32_t ordinal = 0U;
  std::uint32_t layer = 0U;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::uint32_t physical_launches = 0U;
  std::uint32_t logical_cells = 0U;
};

struct Sm87BulkV2Fp8WholeP40FamilyManifest final {
  std::array<Sm87BulkV2Fp8WholeP40FamilyRole,
             kSm87BulkV2Fp8WholeP40LogicalRoles>
      roles{};
  std::uint32_t role_count = 0U;
  std::uint32_t gdn_input_roles = 0U;
  std::uint32_t full_input_roles = 0U;
  std::uint32_t output_roles = 0U;
  std::uint32_t physical_launches = 0U;
  std::uint64_t logical_cells = 0U;
  bool one_launch_per_outer_role = false;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr Sm87BulkV2Fp8WholeP40FamilyManifest
sm87_bulk_v2_fp8_whole_p40_family_manifest() noexcept {
  Sm87BulkV2Fp8WholeP40FamilyManifest result;
  for (std::uint32_t layer = 0U; layer < 64U; ++layer) {
    const bool full = (layer + 1U) % 4U == 0U;
    const auto input_role =
        full ? Sm87TargetAotProjectionRole::kFp8FullQkv
             : Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
    const auto input_plan =
        sm87_bulk_v2_fp8_whole_p40_role_plan(input_role);
    result.roles[result.role_count] =
        {result.role_count, layer, input_role, 1U,
         input_plan.logical_cells};
    ++result.role_count;
    ++result.physical_launches;
    result.logical_cells += input_plan.logical_cells;
    if (full) {
      ++result.full_input_roles;
    } else {
      ++result.gdn_input_roles;
    }
    const auto output_plan = sm87_bulk_v2_fp8_whole_p40_role_plan(
        Sm87TargetAotProjectionRole::kFp8AttentionOutput);
    result.roles[result.role_count] =
        {result.role_count, layer,
         Sm87TargetAotProjectionRole::kFp8AttentionOutput, 1U,
         output_plan.logical_cells};
    ++result.role_count;
    ++result.output_roles;
    ++result.physical_launches;
    result.logical_cells += output_plan.logical_cells;
  }
  result.one_launch_per_outer_role = true;
  result.production_dispatch_eligible = false;
  return result;
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_fp8_whole_p40_family_manifest_valid(
    const Sm87BulkV2Fp8WholeP40FamilyManifest& manifest) noexcept {
  if (manifest.role_count != 128U || manifest.gdn_input_roles != 48U ||
      manifest.full_input_roles != 16U || manifest.output_roles != 64U ||
      manifest.physical_launches != 128U ||
      manifest.logical_cells != 6'560'000ULL ||
      !manifest.one_launch_per_outer_role ||
      manifest.production_dispatch_eligible) {
    return false;
  }
  std::uint32_t gdn = 0U;
  std::uint32_t full = 0U;
  std::uint32_t output = 0U;
  std::uint32_t launches = 0U;
  std::uint64_t cells = 0U;
  for (std::size_t ordinal = 0U; ordinal < manifest.roles.size();
       ++ordinal) {
    const auto& descriptor = manifest.roles[ordinal];
    const auto expected_role =
        ordinal % 2U == 1U
            ? Sm87TargetAotProjectionRole::kFp8AttentionOutput
            : ((((ordinal / 2U) + 1U) % 4U == 0U)
                   ? Sm87TargetAotProjectionRole::kFp8FullQkv
                   : Sm87TargetAotProjectionRole::kFp8GdnQkvZ);
    const auto plan =
        sm87_bulk_v2_fp8_whole_p40_role_plan(descriptor.role);
    if (!plan.valid || descriptor.ordinal != ordinal ||
        descriptor.layer != ordinal / 2U ||
        descriptor.role != expected_role ||
        descriptor.physical_launches != 1U ||
        descriptor.logical_cells != plan.logical_cells) {
      return false;
    }
    launches += descriptor.physical_launches;
    cells += descriptor.logical_cells;
    if (descriptor.role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
      ++gdn;
    } else if (descriptor.role ==
               Sm87TargetAotProjectionRole::kFp8FullQkv) {
      ++full;
    } else {
      ++output;
    }
  }
  return gdn == manifest.gdn_input_roles &&
         full == manifest.full_input_roles &&
         output == manifest.output_roles &&
         launches == manifest.physical_launches &&
         cells == manifest.logical_cells;
}

struct Sm87BulkV2Fp8WholeP40ByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2Fp8WholeP40ByteRange
sm87_bulk_v2_fp8_whole_p40_byte_range(
    const void* const pointer, const std::uint64_t bytes) noexcept {
  if (pointer == nullptr || bytes == 0U ||
      bytes > std::numeric_limits<std::uintptr_t>::max()) {
    return {};
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() -
                  static_cast<std::uintptr_t>(bytes)) {
    return {};
  }
  return {begin, begin + static_cast<std::uintptr_t>(bytes), true};
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_fp8_whole_p40_ranges_overlap(
    const Sm87BulkV2Fp8WholeP40ByteRange& left,
    const Sm87BulkV2Fp8WholeP40ByteRange& right) noexcept {
  return left.valid && right.valid && left.begin < right.end &&
         right.begin < left.end;
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_fp8_whole_p40_output_shape_valid(
    const Sm87TargetAotProjectionRole role,
    const std::uint16_t* const primary,
    const std::uint16_t* const secondary,
    const std::uint16_t* const tertiary) noexcept {
  if (!sm87_bulk_v2_fp8_whole_p40_role_plan(role).valid ||
      primary == nullptr) {
    return false;
  }
  if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    return secondary != nullptr && tertiary != nullptr &&
           primary != secondary && primary != tertiary &&
           secondary != tertiary;
  }
  return secondary == nullptr && tertiary == nullptr;
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_fp8_whole_p40_ranges_valid(
    const Sm87TargetAotProjectionRole role, const void* const input,
    const std::uintptr_t payload_begin, const std::uintptr_t payload_end,
    const void* const primary, const void* const secondary,
    const void* const tertiary, const void* const control,
    const void* const cancellation) noexcept {
  const auto plan = sm87_bulk_v2_fp8_whole_p40_role_plan(role);
  if (!plan.valid || payload_begin == 0U || payload_end <= payload_begin ||
      payload_end - payload_begin != plan.payload_bytes) {
    return false;
  }
  std::array<Sm87BulkV2Fp8WholeP40ByteRange, 7U> ranges{};
  std::size_t count = 0U;
  ranges[count++] = sm87_bulk_v2_fp8_whole_p40_byte_range(
      input, static_cast<std::uint64_t>(kSm87BulkV2Fp8WholeP40Tokens) *
                 plan.input_features * sizeof(std::uint16_t));
  ranges[count++] = {payload_begin, payload_end, true};
  ranges[count++] = sm87_bulk_v2_fp8_whole_p40_byte_range(
      primary, static_cast<std::uint64_t>(kSm87BulkV2Fp8WholeP40Tokens) *
                   plan.primary_output_features * sizeof(std::uint16_t));
  if (plan.secondary_output_features != 0U) {
    ranges[count++] = sm87_bulk_v2_fp8_whole_p40_byte_range(
        secondary,
        static_cast<std::uint64_t>(kSm87BulkV2Fp8WholeP40Tokens) *
            plan.secondary_output_features * sizeof(std::uint16_t));
    ranges[count++] = sm87_bulk_v2_fp8_whole_p40_byte_range(
        tertiary,
        static_cast<std::uint64_t>(kSm87BulkV2Fp8WholeP40Tokens) *
            plan.tertiary_output_features * sizeof(std::uint16_t));
  }
  ranges[count++] = sm87_bulk_v2_fp8_whole_p40_byte_range(
      control, 64U);
  if (cancellation != nullptr) {
    ranges[count++] = sm87_bulk_v2_fp8_whole_p40_byte_range(
        cancellation, sizeof(std::uint32_t));
  }
  for (std::size_t first = 0U; first < count; ++first) {
    if (!ranges[first].valid) {
      return false;
    }
    for (std::size_t second = first + 1U; second < count; ++second) {
      if (sm87_bulk_v2_fp8_whole_p40_ranges_overlap(
              ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

struct Sm87BulkV2Fp8WholeP40WorkItem final {
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
  bool valid = false;
};

// M-group-major with a snake N order.  Full cohorts provide eight identical
// A addresses and four identical B addresses.  This is only a same-address
// service opportunity; CUDA does not guarantee the CTAs remain phase aligned
// or that either operand remains resident in L2.
[[nodiscard]] constexpr Sm87BulkV2Fp8WholeP40WorkItem
sm87_bulk_v2_fp8_whole_p40_work_item(
    const Sm87TargetAotProjectionRole role, const std::uint32_t cohort,
    const std::uint32_t cta,
    const std::uint32_t m_tiles = kSm87BulkV2Fp8WholeP40MTiles,
    const std::uint32_t n_tiles = 0U) noexcept {
  Sm87BulkV2Fp8WholeP40WorkItem result;
  const auto plan = sm87_bulk_v2_fp8_whole_p40_role_plan(role);
  const std::uint32_t actual_n_tiles = n_tiles == 0U ? plan.n_tiles : n_tiles;
  if (!plan.valid || m_tiles == 0U || actual_n_tiles == 0U ||
      m_tiles > kSm87BulkV2Fp8WholeP40MTiles ||
      actual_n_tiles > plan.n_tiles ||
      cta >= kSm87BulkV2Fp8WholeP40PersistentCtas) {
    return result;
  }
  const std::uint32_t m_groups =
      (m_tiles + kSm87BulkV2Fp8WholeP40CohortM - 1U) /
      kSm87BulkV2Fp8WholeP40CohortM;
  const std::uint32_t n_groups =
      (actual_n_tiles + kSm87BulkV2Fp8WholeP40CohortN - 1U) /
      kSm87BulkV2Fp8WholeP40CohortN;
  if (cohort >= m_groups * n_groups) {
    return result;
  }
  result.cohort = cohort;
  result.cta = cta;
  result.m_group = cohort / n_groups;
  result.n_epoch = cohort % n_groups;
  result.n_group = (result.m_group & 1U) == 0U
                       ? result.n_epoch
                       : n_groups - 1U - result.n_epoch;
  result.m_lane = cta / kSm87BulkV2Fp8WholeP40CohortN;
  result.n_lane = cta % kSm87BulkV2Fp8WholeP40CohortN;
  result.m_tile =
      result.m_group * kSm87BulkV2Fp8WholeP40CohortM + result.m_lane;
  result.n_tile =
      result.n_group * kSm87BulkV2Fp8WholeP40CohortN + result.n_lane;
  result.first_m = result.m_tile * kSm87BulkV2Fp8WholeP40TileM;
  result.first_n = result.n_tile * kSm87BulkV2Fp8WholeP40TileN;
  result.active = result.m_tile < m_tiles && result.n_tile < actual_n_tiles;
  result.valid = true;
  if (result.active) {
    result.logical_ordinal = result.m_tile * actual_n_tiles + result.n_tile;
  }
  return result;
}

struct Sm87BulkV2Fp8WholeP40Traffic final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::uint32_t physical_launches = 0U;
  std::uint32_t legacy_segments = 0U;
  std::uint32_t logical_cells = 0U;
  std::uint32_t scheduled_cells = 0U;
  std::uint32_t masked_cells = 0U;
  std::uint32_t grid_barriers = 0U;
  std::uint32_t grid_barriers_per_cohort = 1U;
  std::uint32_t full_cohort_a_requesters = 0U;
  std::uint32_t full_cohort_b_requesters = 0U;
  std::uint64_t a_cell_bytes = 0U;
  std::uint64_t b_cell_bytes = 0U;
  std::uint64_t logical_a_request_bytes = 0U;
  std::uint64_t logical_b_request_bytes = 0U;
  std::uint64_t theoretical_same_address_a_service_bytes = 0U;
  std::uint64_t theoretical_same_address_b_service_bytes = 0U;
  std::uint64_t input_footprint_bytes = 0U;
  std::uint64_t payload_footprint_bytes = 0U;
  std::uint64_t stationary_a_group_footprint_bytes = 0U;
  std::uint64_t stationary_hypothesis_total_service_bytes = 0U;
  bool m_group_outer = false;
  bool snake_n = false;
  bool same_address_service_is_theoretical_only = false;
  bool complete_role_footprint_exceeds_l2 = false;
  bool measured_cross_cta_residency = true;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2Fp8WholeP40Traffic
sm87_bulk_v2_fp8_whole_p40_traffic(
    const Sm87TargetAotProjectionRole role) noexcept {
  const auto plan = sm87_bulk_v2_fp8_whole_p40_role_plan(role);
  if (!plan.valid) {
    return {};
  }
  const std::uint64_t a_cell =
      static_cast<std::uint64_t>(kSm87BulkV2Fp8WholeP40TileM) *
      plan.input_features * sizeof(std::uint16_t);
  const std::uint64_t b_cell =
      static_cast<std::uint64_t>(kSm87BulkV2Fp8WholeP40TileN) *
      plan.input_features;
  const std::uint64_t theoretical_a_cells =
      static_cast<std::uint64_t>(kSm87BulkV2Fp8WholeP40MTiles) *
      plan.n_groups;
  const std::uint64_t theoretical_b_cells =
      static_cast<std::uint64_t>(plan.m_groups) * plan.n_tiles;
  const std::uint64_t input_bytes =
      static_cast<std::uint64_t>(kSm87BulkV2Fp8WholeP40Tokens) *
      plan.input_features * sizeof(std::uint16_t);
  // If the four-row A group survives every N cohort, A is serviced once and
  // B is serviced once per M group.  This is a falsifiable scheduler
  // hypothesis, never a residency or measured-traffic claim.
  const std::uint64_t stationary_hypothesis =
      input_bytes + static_cast<std::uint64_t>(plan.m_groups) *
                        plan.payload_bytes;
  return {role,
          1U,
          0U,
          plan.logical_cells,
          plan.scheduled_cells,
          plan.masked_cells,
          2U,
          0U,
          kSm87BulkV2Fp8WholeP40CohortN,
          kSm87BulkV2Fp8WholeP40CohortM,
          a_cell,
          b_cell,
          static_cast<std::uint64_t>(plan.logical_cells) * a_cell,
          static_cast<std::uint64_t>(plan.logical_cells) * b_cell,
          theoretical_a_cells * a_cell,
          theoretical_b_cells * b_cell,
          input_bytes,
          plan.payload_bytes,
          static_cast<std::uint64_t>(kSm87BulkV2Fp8WholeP40CohortM) *
              a_cell,
          stationary_hypothesis,
          true,
          true,
          true,
          input_bytes + plan.payload_bytes > 4ULL * 1'024ULL * 1'024ULL,
          false,
          true};
}

[[nodiscard]] constexpr bool sm87_bulk_v2_fp8_whole_p40_traffic_valid(
    const Sm87BulkV2Fp8WholeP40Traffic& traffic) noexcept {
  const auto plan =
      sm87_bulk_v2_fp8_whole_p40_role_plan(traffic.role);
  return plan.valid && traffic.physical_launches == 1U &&
         traffic.legacy_segments == 0U &&
         traffic.logical_cells == plan.logical_cells &&
         traffic.scheduled_cells == plan.scheduled_cells &&
         traffic.masked_cells == plan.masked_cells &&
         traffic.grid_barriers == 2U &&
         traffic.grid_barriers_per_cohort == 0U &&
         traffic.full_cohort_a_requesters == 8U &&
         traffic.full_cohort_b_requesters == 4U &&
         traffic.a_cell_bytes == traffic.b_cell_bytes &&
         traffic.logical_a_request_bytes ==
             static_cast<std::uint64_t>(plan.logical_cells) *
                 traffic.a_cell_bytes &&
         traffic.logical_b_request_bytes ==
             static_cast<std::uint64_t>(plan.logical_cells) *
                 traffic.b_cell_bytes &&
         traffic.stationary_a_group_footprint_bytes <=
             4ULL * 1'024ULL * 1'024ULL &&
         traffic.m_group_outer && traffic.snake_n &&
         traffic.same_address_service_is_theoretical_only &&
         traffic.complete_role_footprint_exceeds_l2 &&
         !traffic.measured_cross_cta_residency && traffic.valid;
}

struct Sm87BulkV2Fp8WholeP40FamilyContract final {
  Sm87BulkV2Fp8WholeP40ExecutionIdentity identity =
      Sm87BulkV2Fp8WholeP40ExecutionIdentity::kInvalid;
  std::array<Sm87BulkV2Fp8WholeP40Traffic, 3U> roles{};
  std::uint32_t logical_roles = 0U;
  std::uint32_t physical_launches = 0U;
  std::uint64_t required_policy = 0U;
  bool authenticated_payload_bytes_unchanged = false;
  bool independent_partition_scales = false;
  bool no_request_time_discovery = false;
  bool default_off = false;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr Sm87BulkV2Fp8WholeP40FamilyContract
sm87_bulk_v2_fp8_whole_p40_family_contract() noexcept {
  return {
      Sm87BulkV2Fp8WholeP40ExecutionIdentity::
          kPersistent32CtaM64N128K64Cohort4M8NExactV1,
      {{sm87_bulk_v2_fp8_whole_p40_traffic(
            Sm87TargetAotProjectionRole::kFp8GdnQkvZ),
        sm87_bulk_v2_fp8_whole_p40_traffic(
            Sm87TargetAotProjectionRole::kFp8FullQkv),
        sm87_bulk_v2_fp8_whole_p40_traffic(
            Sm87TargetAotProjectionRole::kFp8AttentionOutput)}},
      kSm87BulkV2Fp8WholeP40LogicalRoles,
      kSm87BulkV2Fp8WholeP40PhysicalLaunches,
      kSm87BulkV2Fp8WholeP40RequiredPolicy,
      true,
      true,
      true,
      true,
      false,
      false};
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_fp8_whole_p40_family_contract_valid(
    const Sm87BulkV2Fp8WholeP40FamilyContract& contract) noexcept {
  return contract.identity == Sm87BulkV2Fp8WholeP40ExecutionIdentity::
                                  kPersistent32CtaM64N128K64Cohort4M8NExactV1 &&
         sm87_bulk_v2_fp8_whole_p40_traffic_valid(contract.roles[0U]) &&
         sm87_bulk_v2_fp8_whole_p40_traffic_valid(contract.roles[1U]) &&
         sm87_bulk_v2_fp8_whole_p40_traffic_valid(contract.roles[2U]) &&
         contract.logical_roles == 128U &&
         contract.physical_launches == 128U &&
         contract.required_policy == kSm87BulkV2Fp8WholeP40RequiredPolicy &&
         contract.authenticated_payload_bytes_unchanged &&
         contract.independent_partition_scales &&
         contract.no_request_time_discovery && contract.default_off &&
         !contract.numerical_contract_qualified &&
         !contract.production_dispatch_eligible;
}

struct alignas(64) Sm87BulkV2Fp8WholeP40DeviceControl final {
  std::uint64_t transaction_epoch = 0U;
  std::uint32_t expected_cells = 0U;
  std::uint32_t started_cells = 0U;
  std::uint32_t completed_cells = 0U;
  std::uint32_t completed_ctas = 0U;
  std::uint32_t cancellation_observed = 0U;
  std::uint32_t launch_completed = 0U;
  std::uint32_t first_incomplete_cohort = 0U;
  std::uint32_t role = 0U;
  std::uint64_t policy = 0U;
  std::array<std::uint32_t, 4U> reserved{};
};

static_assert(sizeof(Sm87BulkV2Fp8WholeP40DeviceControl) == 64U);
static_assert(alignof(Sm87BulkV2Fp8WholeP40DeviceControl) == 64U);

struct Sm87BulkV2Fp8WholeP40Arguments final {
  std::uint64_t transaction_epoch = 0U;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  const std::uint16_t* input = nullptr;
  Sm87TargetAotFp8CudaAssetView authenticated_asset{};
  std::uint16_t* primary_output = nullptr;
  std::uint16_t* secondary_output = nullptr;
  std::uint16_t* tertiary_output = nullptr;
  Sm87BulkV2Fp8WholeP40DeviceControl* device_control = nullptr;
  const std::uint32_t* cancellation_signal = nullptr;
  void* cuda_stream = nullptr;
};

struct Sm87BulkV2Fp8WholeP40CodeEvidence final {
  std::uint64_t elf_identity = 0U;
  std::uint64_t sass_identity = 0U;
  std::size_t stack_bytes = 0U;
  std::size_t spill_store_bytes = 0U;
  std::size_t spill_load_bytes = 0U;
  std::size_t local_bytes = 0U;
  bool launch_bounds_256_2 = false;
  bool contains_cp_async_cg = false;
  bool contains_ldmatrix = false;
  bool contains_bf16_mma = false;
  bool same_kernel_exact_oracle = false;
  bool no_partial_c_symbol = false;
  bool valid = false;
};

[[nodiscard]] constexpr bool
sm87_bulk_v2_fp8_whole_p40_code_evidence_valid(
    const Sm87BulkV2Fp8WholeP40CodeEvidence& evidence) noexcept {
  return evidence.elf_identity != 0U && evidence.sass_identity != 0U &&
         evidence.stack_bytes == 0U && evidence.spill_store_bytes == 0U &&
         evidence.spill_load_bytes == 0U && evidence.local_bytes == 0U &&
         evidence.launch_bounds_256_2 && evidence.contains_cp_async_cg &&
         evidence.contains_ldmatrix && evidence.contains_bf16_mma &&
         evidence.same_kernel_exact_oracle && evidence.no_partial_c_symbol &&
         evidence.valid;
}

struct Sm87BulkV2Fp8WholeP40KernelResources final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int cooperative_grid_capacity = 0;
  Sm87BulkV2Fp8WholeP40CodeEvidence code{};
  bool cooperative_launch_supported = false;
  bool runtime_envelope_observed = false;
  bool external_static_record_consistent = false;
  bool admission_capability_issued = true;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
};

// A read-only observation validator.  The code record is caller supplied and
// therefore cannot mint launch authority.  A private startup owner must bind
// the independently hash-pinned ELF/SASS record to the exact loaded binary
// before it can issue any future admission capability.
[[nodiscard]] constexpr bool
sm87_bulk_v2_fp8_whole_p40_resource_observation_consistent(
    const Sm87BulkV2Fp8WholeP40KernelResources& resources) noexcept {
  return sm87_bulk_v2_fp8_whole_p40_role_plan(resources.role).valid &&
         resources.binary_version == 87 &&
         resources.registers_per_thread > 0 &&
         resources.registers_per_thread <=
             static_cast<int>(kSm87BulkV2Fp8WholeP40MaximumRegisters) &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kSm87BulkV2Fp8WholeP40DynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >= 256 &&
         resources.active_blocks_per_sm >= 2 &&
         resources.cooperative_grid_capacity >= 32 &&
         sm87_bulk_v2_fp8_whole_p40_code_evidence_valid(resources.code) &&
         resources.cooperative_launch_supported &&
         resources.runtime_envelope_observed &&
         resources.external_static_record_consistent &&
         !resources.admission_capability_issued &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

struct Sm87BulkV2Fp8WholeP40FamilyResources final {
  std::array<Sm87BulkV2Fp8WholeP40KernelResources, 3U> roles{};
  bool all_runtime_envelopes_observed = false;
  bool all_external_static_records_consistent = false;
  bool admission_capability_issued = true;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr bool
sm87_bulk_v2_fp8_whole_p40_family_resource_observation_consistent(
    const Sm87BulkV2Fp8WholeP40FamilyResources& resources) noexcept {
  return resources.roles[0U].role ==
             Sm87TargetAotProjectionRole::kFp8GdnQkvZ &&
         resources.roles[1U].role ==
             Sm87TargetAotProjectionRole::kFp8FullQkv &&
         resources.roles[2U].role ==
             Sm87TargetAotProjectionRole::kFp8AttentionOutput &&
         sm87_bulk_v2_fp8_whole_p40_resource_observation_consistent(
             resources.roles[0U]) &&
         sm87_bulk_v2_fp8_whole_p40_resource_observation_consistent(
             resources.roles[1U]) &&
         sm87_bulk_v2_fp8_whole_p40_resource_observation_consistent(
             resources.roles[2U]) &&
         resources.all_runtime_envelopes_observed &&
         resources.all_external_static_records_consistent &&
         !resources.admission_capability_issued &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

// Startup-only resource observation.  This is not reachable from the request
// enqueue and requires independently retained same-ELF SASS evidence.  Its
// result is deliberately incapable of authorizing a launch.
[[nodiscard]] int query_sm87_bulk_dataflow_v2_fp8_whole_p40_resources_cuda(
    const std::array<Sm87BulkV2Fp8WholeP40CodeEvidence, 3U>* code_evidence,
    Sm87BulkV2Fp8WholeP40FamilyResources* resources) noexcept;

// Prevalidated hot enqueue: exactly one cooperative kernel for one complete
// P40000 role.  Startup must already have applied the 49,152-byte dynamic-
// shared attribute and privately bound resource/SASS evidence to the loaded
// ELF.  This function performs no device, occupancy, pointer-allocation, JIT,
// repack, tactic, attribute, or function-resource query.  A future private
// executor must seal that startup state and pointer ownership before exposing
// this seam.
[[nodiscard]] int launch_sm87_bulk_dataflow_v2_fp8_whole_p40_cuda(
    const Sm87BulkV2Fp8WholeP40Arguments& arguments) noexcept;

inline constexpr auto kSm87BulkV2Fp8WholeP40FrozenFamilyContract =
    sm87_bulk_v2_fp8_whole_p40_family_contract();
inline constexpr auto kSm87BulkV2Fp8WholeP40FrozenFamilyManifest =
    sm87_bulk_v2_fp8_whole_p40_family_manifest();

static_assert(sm87_bulk_v2_fp8_whole_p40_raw_code_to_biased_bf16_bits(
                  0x7fU) == 0x07f0U);
static_assert(sm87_bulk_v2_fp8_whole_p40_raw_code_to_biased_bf16_bits(
                  0xffU) == 0x87f0U);
static_assert(sm87_bulk_v2_fp8_whole_p40_family_contract_valid(
    kSm87BulkV2Fp8WholeP40FrozenFamilyContract));
static_assert(sm87_bulk_v2_fp8_whole_p40_family_manifest_valid(
    kSm87BulkV2Fp8WholeP40FrozenFamilyManifest));

}  // namespace q3x::kernels
