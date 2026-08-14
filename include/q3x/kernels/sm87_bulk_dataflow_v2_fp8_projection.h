#pragma once

#include "q3x/kernels/sm87_target_aot_projection_fp8_cuda.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off first executable projection constituent for
// AC-PREFILL-SM87-BULK-DATAFLOW-v2.  It deliberately consumes the already
// authenticated target-AOT FP8 payload bytes through a new execution identity;
// it does not inherit the rejected v1 M128 plan identity, bind a runner
// selector, or claim numerical/performance/production qualification.
inline constexpr std::size_t kSm87BulkV2Fp8P40Tokens = 40'000U;
inline constexpr std::size_t kSm87BulkV2Fp8MacroTokens = 1'024U;
inline constexpr std::size_t kSm87BulkV2Fp8TailTokens = 64U;
inline constexpr std::size_t kSm87BulkV2Fp8MacroSegments = 39U;
inline constexpr std::size_t kSm87BulkV2Fp8SegmentsPerRole = 40U;
inline constexpr std::size_t kSm87BulkV2Fp8LayerCount = 64U;
inline constexpr std::size_t kSm87BulkV2Fp8LogicalRoleCount = 128U;
inline constexpr std::size_t kSm87BulkV2Fp8GdnRoleCount = 48U;
inline constexpr std::size_t kSm87BulkV2Fp8FullRoleCount = 16U;
inline constexpr std::size_t kSm87BulkV2Fp8OutputRoleCount = 64U;
inline constexpr std::size_t kSm87BulkV2Fp8PhysicalLaunches = 5'120U;

inline constexpr std::size_t kSm87BulkV2Fp8TileM = 64U;
inline constexpr std::size_t kSm87BulkV2Fp8TileN = 256U;
inline constexpr std::size_t kSm87BulkV2Fp8TileK = 64U;
inline constexpr std::size_t kSm87BulkV2Fp8WarpM = 64U;
inline constexpr std::size_t kSm87BulkV2Fp8WarpN = 32U;
inline constexpr std::size_t kSm87BulkV2Fp8WarpK = 64U;
inline constexpr std::size_t kSm87BulkV2Fp8Warps = 8U;
inline constexpr std::size_t kSm87BulkV2Fp8Threads = 256U;
inline constexpr std::size_t kSm87BulkV2Fp8PersistentCtas = 16U;
inline constexpr std::size_t kSm87BulkV2Fp8PipelineStages = 4U;
inline constexpr std::size_t kSm87BulkV2Fp8RegisterStages = 2U;
inline constexpr std::size_t kSm87BulkV2Fp8ABytesPerStage =
    kSm87BulkV2Fp8TileM * kSm87BulkV2Fp8TileK * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87BulkV2Fp8BBytesPerStage =
    kSm87BulkV2Fp8TileN * kSm87BulkV2Fp8TileK;
inline constexpr std::size_t kSm87BulkV2Fp8DynamicSharedBytes =
    kSm87BulkV2Fp8PipelineStages *
    (kSm87BulkV2Fp8ABytesPerStage + kSm87BulkV2Fp8BBytesPerStage);
inline constexpr int kSm87BulkV2Fp8PreferredMaximumRegisters = 192;
inline constexpr int kSm87BulkV2Fp8HardMaximumRegisters = 255;
inline constexpr int kSm87BulkV2Fp8RequiredActiveCtasPerSm = 1;

enum class Sm87BulkV2Fp8ExecutionIdentity : std::uint16_t {
  kInvalid = 0U,
  kM64N256K64FourStageFullKRegisterDecodeP40V2,
};

enum class Sm87BulkV2Fp8PayloadViewIdentity : std::uint16_t {
  kInvalid = 0U,
  // The bytes are the authenticated target-AOT
  // [K16][N64][N8][lane][component] payload.  Only the execution consumer is
  // new; this identity performs no repack or arithmetic conversion.
  kTargetAotAuthenticatedByteViewV2,
};

enum Sm87BulkV2Fp8Policy : std::uint64_t {
  kSm87BulkV2Fp8Bf16Activation = 1ULL << 0U,
  kSm87BulkV2Fp8Fp32MmaAccumulation = 1ULL << 1U,
  kSm87BulkV2Fp8AscendingFullKOwnership = 1ULL << 2U,
  kSm87BulkV2Fp8PartitionPrivateScale = 1ULL << 3U,
  kSm87BulkV2Fp8Bf16RnePublication = 1ULL << 4U,
  kSm87BulkV2Fp8MarlinTerminalCodes = 1ULL << 5U,
  kSm87BulkV2Fp8NoSplitK = 1ULL << 6U,
  kSm87BulkV2Fp8NoRequestRepack = 1ULL << 7U,
  kSm87BulkV2Fp8NoRequestScaleConversion = 1ULL << 8U,
  kSm87BulkV2Fp8NoProductionSelector = 1ULL << 9U,
  kSm87BulkV2Fp8NoCublasLt = 1ULL << 10U,
  kSm87BulkV2Fp8NoMtp = 1ULL << 11U,
  kSm87BulkV2Fp8NoActivationQuantization = 1ULL << 12U,
  kSm87BulkV2Fp8NoL2ResidencyClaim = 1ULL << 13U,
  kSm87BulkV2Fp8AXorSharedLayout = 1ULL << 14U,
  kSm87BulkV2Fp8TwoSlotS2R = 1ULL << 15U,
  kSm87BulkV2Fp8NMajorM64Cohort = 1ULL << 16U,
};

inline constexpr std::uint64_t kSm87BulkV2Fp8RequiredPolicy =
    kSm87BulkV2Fp8Bf16Activation |
    kSm87BulkV2Fp8Fp32MmaAccumulation |
    kSm87BulkV2Fp8AscendingFullKOwnership |
    kSm87BulkV2Fp8PartitionPrivateScale |
    kSm87BulkV2Fp8Bf16RnePublication |
    kSm87BulkV2Fp8MarlinTerminalCodes |
    kSm87BulkV2Fp8NoSplitK |
    kSm87BulkV2Fp8NoRequestRepack |
    kSm87BulkV2Fp8NoRequestScaleConversion |
    kSm87BulkV2Fp8NoProductionSelector |
    kSm87BulkV2Fp8NoCublasLt |
    kSm87BulkV2Fp8NoMtp |
    kSm87BulkV2Fp8NoActivationQuantization |
    kSm87BulkV2Fp8NoL2ResidencyClaim |
    kSm87BulkV2Fp8AXorSharedLayout |
    kSm87BulkV2Fp8TwoSlotS2R |
    kSm87BulkV2Fp8NMajorM64Cohort;

#if defined(__CUDACC__)
#define Q3X_SM87_BULK_V2_FP8_HD __host__ __device__
#else
#define Q3X_SM87_BULK_V2_FP8_HD
#endif

// Marlin W8A16 moves the raw E4M3FN sign/magnitude bits into BF16 without
// canonicalizing terminal encodings.  The separately compensated 2^120 scale
// restores the value domain, making 0x7f/0xff exactly +/-480 rather than NaN.
[[nodiscard]] Q3X_SM87_BULK_V2_FP8_HD constexpr std::uint16_t
sm87_bulk_v2_fp8_raw_code_to_biased_bf16_bits(
    const std::uint8_t code) noexcept {
  return static_cast<std::uint16_t>(
      (static_cast<std::uint16_t>(code & 0x80U) << 8U) |
      (static_cast<std::uint16_t>(code & 0x7fU) << 4U));
}

#undef Q3X_SM87_BULK_V2_FP8_HD

struct Sm87BulkV2Fp8RolePlan final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::uint32_t input_features = 0U;
  std::uint32_t projected_output_features = 0U;
  std::uint32_t primary_output_features = 0U;
  std::uint32_t secondary_output_features = 0U;
  std::uint32_t tertiary_output_features = 0U;
  std::uint32_t partition_count = 0U;
  std::array<std::uint32_t, 3U> partition_n_tiles{};
  std::array<std::uint64_t, 3U> partition_payload_offsets{};
  std::uint32_t k_tiles = 0U;
  std::uint32_t n_tiles = 0U;
  std::uint32_t segments = 0U;
  std::uint64_t logical_cta_tiles = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2Fp8RolePlan
sm87_bulk_v2_fp8_role_plan(
    const Sm87TargetAotProjectionRole role) noexcept {
  Sm87BulkV2Fp8RolePlan result;
  result.role = role;
  result.segments = kSm87BulkV2Fp8SegmentsPerRole;
  if (role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    result.input_features = 5'120U;
    result.projected_output_features = 16'384U;
    result.primary_output_features = 16'384U;
    result.partition_count = 2U;
    result.partition_n_tiles = {40U, 24U, 0U};
    result.partition_payload_offsets = {0U, 52'428'800U, 0U};
    result.k_tiles = 80U;
    result.n_tiles = 64U;
  } else if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    result.input_features = 5'120U;
    result.projected_output_features = 14'336U;
    result.primary_output_features = 12'288U;
    result.secondary_output_features = 1'024U;
    result.tertiary_output_features = 1'024U;
    result.partition_count = 3U;
    result.partition_n_tiles = {48U, 4U, 4U};
    result.partition_payload_offsets =
        {0U, 62'914'560U, 68'157'440U};
    result.k_tiles = 80U;
    result.n_tiles = 56U;
  } else if (role ==
             Sm87TargetAotProjectionRole::kFp8AttentionOutput) {
    result.input_features = 6'144U;
    result.projected_output_features = 5'120U;
    result.primary_output_features = 5'120U;
    result.partition_count = 1U;
    result.partition_n_tiles = {20U, 0U, 0U};
    result.partition_payload_offsets = {0U, 0U, 0U};
    result.k_tiles = 96U;
    result.n_tiles = 20U;
  } else {
    return result;
  }
  result.logical_cta_tiles =
      (kSm87BulkV2Fp8P40Tokens / kSm87BulkV2Fp8TileM) *
      result.n_tiles;
  result.valid = true;
  return result;
}

struct Sm87BulkV2Fp8FamilyRole final {
  std::uint32_t ordinal = 0U;
  std::uint32_t layer = 0U;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::uint32_t physical_launches = 0U;
  std::uint64_t logical_cta_tiles = 0U;
};

struct Sm87BulkV2Fp8FamilyManifest final {
  Sm87BulkV2Fp8ExecutionIdentity execution_identity =
      Sm87BulkV2Fp8ExecutionIdentity::kInvalid;
  Sm87BulkV2Fp8PayloadViewIdentity payload_view_identity =
      Sm87BulkV2Fp8PayloadViewIdentity::kInvalid;
  std::array<Sm87BulkV2Fp8FamilyRole,
             kSm87BulkV2Fp8LogicalRoleCount>
      roles{};
  std::uint32_t role_count = 0U;
  std::uint32_t gdn_roles = 0U;
  std::uint32_t full_roles = 0U;
  std::uint32_t output_roles = 0U;
  std::uint32_t physical_launches = 0U;
  std::uint64_t logical_cta_tiles = 0U;
  std::uint64_t required_policy = 0U;
  bool reuses_authenticated_payload_bytes = false;
  bool request_time_repack = true;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr Sm87BulkV2Fp8FamilyManifest
sm87_bulk_v2_fp8_family_manifest() noexcept {
  Sm87BulkV2Fp8FamilyManifest result;
  result.execution_identity = Sm87BulkV2Fp8ExecutionIdentity::
      kM64N256K64FourStageFullKRegisterDecodeP40V2;
  result.payload_view_identity = Sm87BulkV2Fp8PayloadViewIdentity::
      kTargetAotAuthenticatedByteViewV2;
  for (std::size_t layer = 0U; layer < kSm87BulkV2Fp8LayerCount; ++layer) {
    const bool full = (layer + 1U) % 4U == 0U;
    const auto input_role =
        full ? Sm87TargetAotProjectionRole::kFp8FullQkv
             : Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
    const auto input_plan = sm87_bulk_v2_fp8_role_plan(input_role);
    result.roles[result.role_count] = {
        result.role_count, static_cast<std::uint32_t>(layer), input_role,
        input_plan.segments, input_plan.logical_cta_tiles};
    ++result.role_count;
    result.physical_launches += input_plan.segments;
    result.logical_cta_tiles += input_plan.logical_cta_tiles;
    if (full) {
      ++result.full_roles;
    } else {
      ++result.gdn_roles;
    }

    const auto output_plan = sm87_bulk_v2_fp8_role_plan(
        Sm87TargetAotProjectionRole::kFp8AttentionOutput);
    result.roles[result.role_count] = {
        result.role_count, static_cast<std::uint32_t>(layer),
        Sm87TargetAotProjectionRole::kFp8AttentionOutput,
        output_plan.segments, output_plan.logical_cta_tiles};
    ++result.role_count;
    ++result.output_roles;
    result.physical_launches += output_plan.segments;
    result.logical_cta_tiles += output_plan.logical_cta_tiles;
  }
  result.required_policy = kSm87BulkV2Fp8RequiredPolicy;
  result.reuses_authenticated_payload_bytes = true;
  result.request_time_repack = false;
  result.production_dispatch_eligible = false;
  return result;
}

[[nodiscard]] constexpr bool sm87_bulk_v2_fp8_family_manifest_valid(
    const Sm87BulkV2Fp8FamilyManifest& manifest) noexcept {
  if (manifest.execution_identity != Sm87BulkV2Fp8ExecutionIdentity::
          kM64N256K64FourStageFullKRegisterDecodeP40V2 ||
      manifest.payload_view_identity != Sm87BulkV2Fp8PayloadViewIdentity::
          kTargetAotAuthenticatedByteViewV2 ||
      manifest.role_count != kSm87BulkV2Fp8LogicalRoleCount ||
      manifest.gdn_roles != kSm87BulkV2Fp8GdnRoleCount ||
      manifest.full_roles != kSm87BulkV2Fp8FullRoleCount ||
      manifest.output_roles != kSm87BulkV2Fp8OutputRoleCount ||
      manifest.physical_launches != kSm87BulkV2Fp8PhysicalLaunches ||
      manifest.logical_cta_tiles != 3'280'000U ||
      manifest.required_policy != kSm87BulkV2Fp8RequiredPolicy ||
      !manifest.reuses_authenticated_payload_bytes ||
      manifest.request_time_repack || manifest.production_dispatch_eligible) {
    return false;
  }
  for (std::size_t index = 0U; index < manifest.roles.size(); ++index) {
    const auto& descriptor = manifest.roles[index];
    const auto plan = sm87_bulk_v2_fp8_role_plan(descriptor.role);
    if (!plan.valid || descriptor.ordinal != index ||
        descriptor.layer != index / 2U ||
        descriptor.physical_launches != plan.segments ||
        descriptor.logical_cta_tiles != plan.logical_cta_tiles ||
        (index % 2U == 0U && descriptor.role !=
             (((descriptor.layer + 1U) % 4U == 0U)
                  ? Sm87TargetAotProjectionRole::kFp8FullQkv
                  : Sm87TargetAotProjectionRole::kFp8GdnQkvZ)) ||
        (index % 2U == 1U && descriptor.role !=
             Sm87TargetAotProjectionRole::kFp8AttentionOutput)) {
      return false;
    }
  }
  return true;
}

struct Sm87BulkV2Fp8AssetView final {
  Sm87TargetAotFp8CudaAssetView authenticated{};
  Sm87BulkV2Fp8ExecutionIdentity execution_identity =
      Sm87BulkV2Fp8ExecutionIdentity::kInvalid;
  Sm87BulkV2Fp8PayloadViewIdentity payload_view_identity =
      Sm87BulkV2Fp8PayloadViewIdentity::kInvalid;
  bool same_authenticated_payload_bytes = false;
  bool no_request_time_repack = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr Sm87BulkV2Fp8AssetView
sm87_bulk_v2_fp8_bind_authenticated_asset(
    const Sm87TargetAotFp8CudaAssetView& authenticated) noexcept {
  if (!sm87_target_aot_fp8_cuda_asset_valid(authenticated)) {
    return {};
  }
  Sm87BulkV2Fp8AssetView result;
  result.authenticated = authenticated;
  result.execution_identity = Sm87BulkV2Fp8ExecutionIdentity::
      kM64N256K64FourStageFullKRegisterDecodeP40V2;
  result.payload_view_identity = Sm87BulkV2Fp8PayloadViewIdentity::
      kTargetAotAuthenticatedByteViewV2;
  result.same_authenticated_payload_bytes = true;
  result.no_request_time_repack = true;
  result.production_dispatch_eligible = false;
  return result;
}

[[nodiscard]] constexpr bool sm87_bulk_v2_fp8_asset_valid(
    const Sm87BulkV2Fp8AssetView& asset,
    const Sm87TargetAotProjectionRole role) noexcept {
  return sm87_bulk_v2_fp8_role_plan(role).valid &&
         asset.execution_identity == Sm87BulkV2Fp8ExecutionIdentity::
             kM64N256K64FourStageFullKRegisterDecodeP40V2 &&
         asset.payload_view_identity == Sm87BulkV2Fp8PayloadViewIdentity::
             kTargetAotAuthenticatedByteViewV2 &&
         asset.same_authenticated_payload_bytes &&
         asset.no_request_time_repack &&
         !asset.production_dispatch_eligible &&
         asset.authenticated.payload.role == role &&
         sm87_target_aot_fp8_cuda_asset_valid(asset.authenticated);
}

struct Sm87BulkV2Fp8RoleArguments final {
  std::uint32_t layer = 0U;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  const std::uint16_t* input = nullptr;
  Sm87BulkV2Fp8AssetView asset{};
  std::uint16_t* primary_output = nullptr;
  std::uint16_t* secondary_output = nullptr;
  std::uint16_t* tertiary_output = nullptr;
};

struct Sm87BulkV2Fp8SegmentArguments final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  const std::uint16_t* input = nullptr;
  Sm87BulkV2Fp8AssetView asset{};
  std::size_t token_count = 0U;
  std::uint16_t* primary_output = nullptr;
  std::uint16_t* secondary_output = nullptr;
  std::uint16_t* tertiary_output = nullptr;
  void* cuda_stream = nullptr;
};

struct Sm87BulkV2Fp8FamilyArguments final {
  const Sm87BulkV2Fp8RoleArguments* roles = nullptr;
  std::size_t role_count = 0U;
  std::size_t token_count = 0U;
  void* cuda_stream = nullptr;
};

[[nodiscard]] constexpr bool sm87_bulk_v2_fp8_segment_token_count_valid(
    const std::size_t token_count) noexcept {
  return token_count == kSm87BulkV2Fp8MacroTokens ||
         token_count == kSm87BulkV2Fp8TailTokens;
}

[[nodiscard]] constexpr bool sm87_bulk_v2_fp8_output_shape_valid(
    const Sm87TargetAotProjectionRole role,
    const std::uint16_t* const primary,
    const std::uint16_t* const secondary,
    const std::uint16_t* const tertiary) noexcept {
  if (!sm87_bulk_v2_fp8_role_plan(role).valid || primary == nullptr) {
    return false;
  }
  if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    return secondary != nullptr && tertiary != nullptr &&
           primary != secondary && primary != tertiary &&
           secondary != tertiary;
  }
  return secondary == nullptr && tertiary == nullptr;
}

[[nodiscard]] constexpr bool sm87_bulk_v2_fp8_segment_ranges_valid(
    const Sm87BulkV2Fp8SegmentArguments& arguments) noexcept {
  const auto plan = sm87_bulk_v2_fp8_role_plan(arguments.role);
  if (!plan.valid || arguments.token_count == 0U) {
    return false;
  }
  std::array<Sm87TargetAotFp8CudaByteRange, 5U> ranges{};
  std::size_t count = 0U;
  ranges[count++] = sm87_target_aot_fp8_cuda_byte_range(
      arguments.input,
      static_cast<std::uint64_t>(arguments.token_count) *
          plan.input_features * sizeof(std::uint16_t));
  ranges[count++] = sm87_target_aot_fp8_cuda_byte_range(
      arguments.primary_output,
      static_cast<std::uint64_t>(arguments.token_count) *
          plan.primary_output_features * sizeof(std::uint16_t));
  if (plan.secondary_output_features != 0U) {
    ranges[count++] = sm87_target_aot_fp8_cuda_byte_range(
        arguments.secondary_output,
        static_cast<std::uint64_t>(arguments.token_count) *
            plan.secondary_output_features * sizeof(std::uint16_t));
    ranges[count++] = sm87_target_aot_fp8_cuda_byte_range(
        arguments.tertiary_output,
        static_cast<std::uint64_t>(arguments.token_count) *
            plan.tertiary_output_features * sizeof(std::uint16_t));
  }
  ranges[count++] = {
      arguments.asset.authenticated.payload.begin,
      arguments.asset.authenticated.payload.end,
      arguments.asset.authenticated.payload.valid};
  for (std::size_t first = 0U; first < count; ++first) {
    if (!ranges[first].valid) {
      return false;
    }
    for (std::size_t second = first + 1U; second < count; ++second) {
      if (sm87_target_aot_fp8_cuda_ranges_overlap(ranges[first],
                                                   ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] constexpr bool sm87_bulk_v2_fp8_segment_arguments_valid(
    const Sm87BulkV2Fp8SegmentArguments& arguments) noexcept {
  const auto plan = sm87_bulk_v2_fp8_role_plan(arguments.role);
  return plan.valid &&
         sm87_bulk_v2_fp8_segment_token_count_valid(
             arguments.token_count) &&
         arguments.input != nullptr && arguments.cuda_stream != nullptr &&
         reinterpret_cast<std::uintptr_t>(arguments.input) % 16U == 0U &&
         reinterpret_cast<std::uintptr_t>(arguments.primary_output) % 16U ==
             0U &&
         (arguments.secondary_output == nullptr ||
          reinterpret_cast<std::uintptr_t>(arguments.secondary_output) %
                  16U ==
              0U) &&
         (arguments.tertiary_output == nullptr ||
          reinterpret_cast<std::uintptr_t>(arguments.tertiary_output) % 16U ==
              0U) &&
         sm87_bulk_v2_fp8_output_shape_valid(
             arguments.role, arguments.primary_output,
             arguments.secondary_output, arguments.tertiary_output) &&
         sm87_bulk_v2_fp8_asset_valid(arguments.asset, arguments.role) &&
         sm87_bulk_v2_fp8_segment_ranges_valid(arguments);
}

struct Sm87BulkV2Fp8KernelResources final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  bool kernel_compiled = false;
  bool resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool sm87_bulk_v2_fp8_kernel_resources_valid(
    const Sm87BulkV2Fp8KernelResources& resources) noexcept {
  return sm87_bulk_v2_fp8_role_plan(resources.role).valid &&
         resources.binary_version == 87 && resources.registers_per_thread > 0 &&
         resources.registers_per_thread <=
             kSm87BulkV2Fp8HardMaximumRegisters &&
         resources.dynamic_shared_bytes ==
             kSm87BulkV2Fp8DynamicSharedBytes &&
         resources.static_shared_bytes == 0U &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<int>(kSm87BulkV2Fp8Threads) &&
         resources.active_blocks_per_sm ==
             kSm87BulkV2Fp8RequiredActiveCtasPerSm &&
         resources.kernel_compiled && resources.resource_gate_passed &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

struct Sm87BulkV2Fp8FamilyResources final {
  std::array<Sm87BulkV2Fp8KernelResources, 3U> roles{};
  bool all_compiled = false;
  bool resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool sm87_bulk_v2_fp8_family_resources_valid(
    const Sm87BulkV2Fp8FamilyResources& resources) noexcept {
  return resources.all_compiled && resources.resource_gate_passed &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible &&
         resources.roles[0U].role ==
             Sm87TargetAotProjectionRole::kFp8GdnQkvZ &&
         resources.roles[1U].role ==
             Sm87TargetAotProjectionRole::kFp8FullQkv &&
         resources.roles[2U].role ==
             Sm87TargetAotProjectionRole::kFp8AttentionOutput &&
         sm87_bulk_v2_fp8_kernel_resources_valid(resources.roles[0U]) &&
         sm87_bulk_v2_fp8_kernel_resources_valid(resources.roles[1U]) &&
         sm87_bulk_v2_fp8_kernel_resources_valid(resources.roles[2U]);
}

// Correctness-only raw-payload seam for the CUDA oracle.  It does not accept
// P40000, cannot construct a family executor, and has no authentication or
// performance authority.  Production-like role/family launches below accept
// only Sm87BulkV2Fp8AssetView.
struct Sm87BulkV2Fp8OracleArguments final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  const std::uint16_t* input = nullptr;
  const std::uint8_t* payload = nullptr;
  std::array<std::uint16_t, 3U> compensated_scale_bf16_bits{};
  std::size_t token_count = 0U;
  std::uint16_t* primary_output = nullptr;
  std::uint16_t* secondary_output = nullptr;
  std::uint16_t* tertiary_output = nullptr;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int query_sm87_bulk_dataflow_v2_fp8_family_resources_cuda(
    Sm87BulkV2Fp8FamilyResources* resources) noexcept;

[[nodiscard]] int launch_sm87_bulk_dataflow_v2_fp8_segment_cuda(
    const Sm87BulkV2Fp8SegmentArguments& arguments) noexcept;

[[nodiscard]] int launch_sm87_bulk_dataflow_v2_fp8_role_p40_cuda(
    const Sm87BulkV2Fp8RoleArguments& arguments, void* cuda_stream,
    std::size_t* enqueued_launches) noexcept;

[[nodiscard]] int launch_sm87_bulk_dataflow_v2_fp8_family_p40_cuda(
    const Sm87BulkV2Fp8FamilyArguments& arguments,
    std::size_t* enqueued_launches) noexcept;

[[nodiscard]] int
launch_sm87_bulk_dataflow_v2_fp8_oracle_segment_cuda(
    const Sm87BulkV2Fp8OracleArguments& arguments) noexcept;

inline constexpr auto kSm87BulkV2Fp8FrozenFamilyManifest =
    sm87_bulk_v2_fp8_family_manifest();

static_assert(kSm87BulkV2Fp8P40Tokens ==
              kSm87BulkV2Fp8MacroSegments *
                      kSm87BulkV2Fp8MacroTokens +
                  kSm87BulkV2Fp8TailTokens);
static_assert(kSm87BulkV2Fp8DynamicSharedBytes == 98'304U);
static_assert(kSm87BulkV2Fp8PhysicalLaunches ==
              kSm87BulkV2Fp8LogicalRoleCount *
                  kSm87BulkV2Fp8SegmentsPerRole);
static_assert(sm87_bulk_v2_fp8_raw_code_to_biased_bf16_bits(0x7fU) ==
              0x07f0U);
static_assert(sm87_bulk_v2_fp8_raw_code_to_biased_bf16_bits(0xffU) ==
              0x87f0U);
static_assert(sm87_bulk_v2_fp8_family_manifest_valid(
    kSm87BulkV2Fp8FrozenFamilyManifest));

}  // namespace q3x::kernels
