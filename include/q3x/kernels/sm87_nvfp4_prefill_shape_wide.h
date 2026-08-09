#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// A test-only, fail-closed Qwen3.6-27B/SM87 surface for the exact P40000
// shape-wide NVFP4 MLP experiment.  Gate+Up and Down deliberately do not
// share one tile configuration: their N/K aspect ratios require different
// ownership and raster ordering.  No production dispatcher or fallback is
// reachable through this API.
enum class Sm87NvFp4PrefillShapeWideRole : std::uint8_t {
  kGateUp = 0U,
  kDown = 1U,
};

enum class Sm87NvFp4PrefillShapeWideDataflow : std::uint8_t {
  kGateUpM128N64PairGroupM2 = 0U,
  kDownM128N128GroupM1Amajor = 1U,
};

inline constexpr std::size_t kSm87NvFp4PrefillShapeWideTokens = 40'000U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideHidden = 5'120U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideIntermediate =
    17'408U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideMergedGateUp =
    34'816U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideTileK = 64U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideThreads = 256U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideSmCount = 16U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideMaximumRegisters =
    128U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideMinimumBlocksPerSm =
    2U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideSharedLimitBytes =
    72U * 1'024U;

inline constexpr std::size_t kSm87NvFp4PrefillShapeWideGateTileM = 128U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideGateBranchTileN = 64U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideGatePhysicalTileN =
    128U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideGatePipelineStages =
    3U;
inline constexpr std::size_t
    kSm87NvFp4PrefillShapeWideGateSharedABytesPerStage =
        kSm87NvFp4PrefillShapeWideGateTileM *
        kSm87NvFp4PrefillShapeWideTileK * sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87NvFp4PrefillShapeWideGateSharedBBytesPerStage =
        kSm87NvFp4PrefillShapeWideGatePhysicalTileN *
        kSm87NvFp4PrefillShapeWideTileK / 2U;
inline constexpr std::size_t
    kSm87NvFp4PrefillShapeWideGateSharedScaleBytesPerStage =
        kSm87NvFp4PrefillShapeWideGatePhysicalTileN *
        kSm87NvFp4PrefillShapeWideTileK / 16U;
inline constexpr std::size_t
    kSm87NvFp4PrefillShapeWideGateDynamicSharedBytes =
        kSm87NvFp4PrefillShapeWideGatePipelineStages *
        (kSm87NvFp4PrefillShapeWideGateSharedABytesPerStage +
         kSm87NvFp4PrefillShapeWideGateSharedBBytesPerStage +
         kSm87NvFp4PrefillShapeWideGateSharedScaleBytesPerStage);

inline constexpr std::size_t kSm87NvFp4PrefillShapeWideDownTileM = 128U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideDownTileN = 128U;
inline constexpr std::size_t kSm87NvFp4PrefillShapeWideDownPipelineStages =
    3U;
inline constexpr std::size_t
    kSm87NvFp4PrefillShapeWideDownSharedABytesPerStage =
        kSm87NvFp4PrefillShapeWideDownTileM *
        kSm87NvFp4PrefillShapeWideTileK * sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87NvFp4PrefillShapeWideDownSharedBBytesPerStage =
        kSm87NvFp4PrefillShapeWideDownTileN *
        kSm87NvFp4PrefillShapeWideTileK / 2U;
inline constexpr std::size_t
    kSm87NvFp4PrefillShapeWideDownSharedScaleBytesPerStage =
        kSm87NvFp4PrefillShapeWideDownTileN *
        kSm87NvFp4PrefillShapeWideTileK / 16U;
inline constexpr std::size_t
    kSm87NvFp4PrefillShapeWideDownDynamicSharedBytes =
        kSm87NvFp4PrefillShapeWideDownPipelineStages *
        (kSm87NvFp4PrefillShapeWideDownSharedABytesPerStage +
         kSm87NvFp4PrefillShapeWideDownSharedBBytesPerStage +
         kSm87NvFp4PrefillShapeWideDownSharedScaleBytesPerStage);

static_assert(kSm87NvFp4PrefillShapeWideGateSharedABytesPerStage ==
              16'384U);
static_assert(kSm87NvFp4PrefillShapeWideGateSharedBBytesPerStage == 4'096U);
static_assert(kSm87NvFp4PrefillShapeWideGateSharedScaleBytesPerStage == 512U);
static_assert(kSm87NvFp4PrefillShapeWideGateDynamicSharedBytes == 62'976U);
static_assert(kSm87NvFp4PrefillShapeWideDownSharedABytesPerStage ==
              16'384U);
static_assert(kSm87NvFp4PrefillShapeWideDownSharedBBytesPerStage == 4'096U);
static_assert(kSm87NvFp4PrefillShapeWideDownSharedScaleBytesPerStage == 512U);
static_assert(kSm87NvFp4PrefillShapeWideDownDynamicSharedBytes == 62'976U);
static_assert(kSm87NvFp4PrefillShapeWideGateDynamicSharedBytes <=
              kSm87NvFp4PrefillShapeWideSharedLimitBytes);
static_assert(kSm87NvFp4PrefillShapeWideDownDynamicSharedBytes <=
              kSm87NvFp4PrefillShapeWideSharedLimitBytes);
static_assert(kSm87NvFp4PrefillShapeWideTokens %
                      kSm87NvFp4PrefillShapeWideGateTileM ==
                  64U);
static_assert(kSm87NvFp4PrefillShapeWideTokens %
                      kSm87NvFp4PrefillShapeWideDownTileM ==
                  64U);

struct Sm87NvFp4PrefillShapeWidePlan {
  Sm87NvFp4PrefillShapeWideRole role =
      Sm87NvFp4PrefillShapeWideRole::kGateUp;
  Sm87NvFp4PrefillShapeWideDataflow dataflow =
      Sm87NvFp4PrefillShapeWideDataflow::kGateUpM128N64PairGroupM2;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t weight_output_features = 0U;
  std::size_t published_output_features = 0U;
  std::size_t tile_m = 0U;
  std::size_t branch_tile_n = 0U;
  std::size_t physical_tile_n = 0U;
  std::size_t tile_k = 0U;
  std::size_t threads = 0U;
  std::size_t pipeline_stages = 0U;
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t tail_rows = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t group_m = 0U;
  bool a_major = false;
  bool fused_silu = false;
  bool bf16_branch_boundary = false;
  bool fused_in_place_residual = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    const bool gate = role == Sm87NvFp4PrefillShapeWideRole::kGateUp;
    const bool down = role == Sm87NvFp4PrefillShapeWideRole::kDown;
    return (gate || down) && token_count == kSm87NvFp4PrefillShapeWideTokens &&
           dataflow ==
               (gate ? Sm87NvFp4PrefillShapeWideDataflow::
                           kGateUpM128N64PairGroupM2
                     : Sm87NvFp4PrefillShapeWideDataflow::
                           kDownM128N128GroupM1Amajor) &&
           input_features ==
               (gate ? kSm87NvFp4PrefillShapeWideHidden
                     : kSm87NvFp4PrefillShapeWideIntermediate) &&
           weight_output_features ==
               (gate ? kSm87NvFp4PrefillShapeWideMergedGateUp
                     : kSm87NvFp4PrefillShapeWideHidden) &&
           published_output_features ==
               (gate ? kSm87NvFp4PrefillShapeWideIntermediate
                     : kSm87NvFp4PrefillShapeWideHidden) &&
           tile_m ==
               (gate ? kSm87NvFp4PrefillShapeWideGateTileM
                     : kSm87NvFp4PrefillShapeWideDownTileM) &&
           branch_tile_n ==
               (gate ? kSm87NvFp4PrefillShapeWideGateBranchTileN
                     : kSm87NvFp4PrefillShapeWideDownTileN) &&
           physical_tile_n ==
               (gate ? kSm87NvFp4PrefillShapeWideGatePhysicalTileN
                     : kSm87NvFp4PrefillShapeWideDownTileN) &&
           tile_k == kSm87NvFp4PrefillShapeWideTileK &&
           threads == kSm87NvFp4PrefillShapeWideThreads &&
           pipeline_stages ==
               (gate ? kSm87NvFp4PrefillShapeWideGatePipelineStages
                     : kSm87NvFp4PrefillShapeWideDownPipelineStages) &&
           grid_m == 313U && grid_n == (gate ? 272U : 40U) &&
           tail_rows == 64U &&
           dynamic_shared_bytes ==
               (gate ? kSm87NvFp4PrefillShapeWideGateDynamicSharedBytes
                     : kSm87NvFp4PrefillShapeWideDownDynamicSharedBytes) &&
           group_m == (gate ? 2U : 1U) && a_major == down &&
           fused_silu == gate &&
           bf16_branch_boundary == down &&
           fused_in_place_residual == down;
  }
};

[[nodiscard]] constexpr bool sm87_nvfp4_prefill_shape_wide_supports(
    const Sm87NvFp4PrefillShapeWideRole role,
    const std::size_t token_count) noexcept {
  const bool valid_role =
      role == Sm87NvFp4PrefillShapeWideRole::kGateUp ||
      role == Sm87NvFp4PrefillShapeWideRole::kDown;
  return valid_role && token_count == kSm87NvFp4PrefillShapeWideTokens;
}

[[nodiscard]] constexpr Sm87NvFp4PrefillShapeWidePlan
sm87_nvfp4_prefill_shape_wide_plan(
    const Sm87NvFp4PrefillShapeWideRole role,
    const std::size_t token_count) noexcept {
  if (!sm87_nvfp4_prefill_shape_wide_supports(role, token_count)) {
    return {};
  }
  const bool gate = role == Sm87NvFp4PrefillShapeWideRole::kGateUp;
  return {
      role,
      gate ? Sm87NvFp4PrefillShapeWideDataflow::
                 kGateUpM128N64PairGroupM2
           : Sm87NvFp4PrefillShapeWideDataflow::
                 kDownM128N128GroupM1Amajor,
      token_count,
      gate ? kSm87NvFp4PrefillShapeWideHidden
           : kSm87NvFp4PrefillShapeWideIntermediate,
      gate ? kSm87NvFp4PrefillShapeWideMergedGateUp
           : kSm87NvFp4PrefillShapeWideHidden,
      gate ? kSm87NvFp4PrefillShapeWideIntermediate
           : kSm87NvFp4PrefillShapeWideHidden,
      gate ? kSm87NvFp4PrefillShapeWideGateTileM
           : kSm87NvFp4PrefillShapeWideDownTileM,
      gate ? kSm87NvFp4PrefillShapeWideGateBranchTileN
           : kSm87NvFp4PrefillShapeWideDownTileN,
      gate ? kSm87NvFp4PrefillShapeWideGatePhysicalTileN
           : kSm87NvFp4PrefillShapeWideDownTileN,
      kSm87NvFp4PrefillShapeWideTileK,
      kSm87NvFp4PrefillShapeWideThreads,
      gate ? kSm87NvFp4PrefillShapeWideGatePipelineStages
           : kSm87NvFp4PrefillShapeWideDownPipelineStages,
      313U,
      gate ? 272U : 40U,
      64U,
      gate ? kSm87NvFp4PrefillShapeWideGateDynamicSharedBytes
           : kSm87NvFp4PrefillShapeWideDownDynamicSharedBytes,
      gate ? 2U : 1U,
      !gate,
      gate,
      !gate,
      !gate};
}

struct Sm87NvFp4PrefillShapeWideTileCoordinate {
  std::size_t m_tile = 0U;
  std::size_t n_tile = 0U;
  bool mapped = false;
};

// Host-visible copy of the sealed physical raster. Tests exhaust the complete
// P40000 grid to prove that grouped Gate ordering and A-major Down ordering
// are both bijections, including Gate's final one-M group.
[[nodiscard]] constexpr Sm87NvFp4PrefillShapeWideTileCoordinate
sm87_nvfp4_prefill_shape_wide_tile_coordinate(
    const Sm87NvFp4PrefillShapeWideRole role,
    const std::size_t physical_block) noexcept {
  const auto plan = sm87_nvfp4_prefill_shape_wide_plan(
      role, kSm87NvFp4PrefillShapeWideTokens);
  if (!plan.valid() || physical_block >= plan.grid_m * plan.grid_n) {
    return {};
  }
  if (role == Sm87NvFp4PrefillShapeWideRole::kGateUp) {
    constexpr std::size_t kGroupM = 2U;
    constexpr std::size_t kGridN = 272U;
    const std::size_t group_blocks = kGroupM * kGridN;
    const std::size_t group = physical_block / group_blocks;
    const std::size_t first_group_m = group * kGroupM;
    const std::size_t remaining_m = 313U - first_group_m;
    const std::size_t active_group_m =
        remaining_m < kGroupM ? remaining_m : kGroupM;
    const std::size_t group_offset = physical_block % group_blocks;
    return {first_group_m + group_offset % active_group_m,
            group_offset / active_group_m, true};
  }
  return {physical_block / 40U, physical_block % 40U, true};
}

struct Sm87NvFp4PrefillShapeWideCapability {
  Sm87NvFp4PrefillShapeWidePlan plan{};
  int device = -1;
  int compute_major = 0;
  int compute_minor = 0;
  int sm_count = 0;
  std::size_t optin_shared_bytes_per_block = 0U;
  std::size_t shared_bytes_per_sm = 0U;
  bool supported = false;
};

struct Sm87NvFp4PrefillShapeWideResources {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

// CUDA-facing functions return cudaError_t encoded as int so this header is
// CUDA-header neutral.  Unsupported admission/device/shape/resource states
// return cudaErrorNotSupported. Pointer, alignment, and alias violations
// return cudaErrorInvalidValue.
int query_sm87_nvfp4_prefill_shape_wide_capability_cuda(
    Sm87NvFp4PrefillShapeWideRole role, std::size_t token_count,
    Sm87NvFp4PrefillShapeWideCapability* capability) noexcept;

int query_sm87_nvfp4_prefill_shape_wide_resources_cuda(
    Sm87NvFp4PrefillShapeWideRole role, std::size_t token_count,
    Sm87NvFp4PrefillShapeWideResources* resources) noexcept;

int launch_sm87_nvfp4_prefill_shape_wide_gate_up_cuda(
    const std::uint16_t* input, const std::uint8_t* merged_marlin_weight,
    const std::uint8_t* merged_marlin_scales,
    const float* marlin_global_scale, std::size_t token_count,
    std::uint16_t* activated, void* cuda_stream) noexcept;

int launch_sm87_nvfp4_prefill_shape_wide_down_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales,
    const float* marlin_global_scale, const std::uint16_t* residual,
    std::size_t token_count, std::uint16_t* output,
    void* cuda_stream) noexcept;

}  // namespace q3x::kernels
