#pragma once

#include "q3x/kernels/sm87_fp8_prefill_supermatrix.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Test-only physical surface for the exact-P40000 projection reset.  The
// kernels are deliberately Qwen3.6-27B/SM87-specific and have no compatibility
// fallback.  Gate/Up consumes the already authenticated, losslessly
// interleaved Marlin sidecar and publishes the BF16 SiLU product.  Down
// consumes that product and updates the BF16 residual in place.
enum class Sm87P40ProjectionResetNvFp4Role : std::uint8_t {
  kInterleavedGateUpSilu = 0U,
  kDownResidual = 1U,
};

enum class Sm87P40ProjectionResetRaster : std::uint8_t {
  // Four M64 A slabs (2.5 MiB) are retained while N advances.  Each N128
  // weight/scale slab is consumed by four adjacent CTAs before the raster
  // moves on, keeping the working cohort below Orin's 4 MiB L2 budget.
  kGateUpGroupedM4NMajor = 0U,
  // One Down M64 A slab is 2.125 MiB; adding a second slab would evict the
  // 1.196 MiB N128 weight/scale slab from the 4 MiB L2.  Down therefore keeps
  // one A slab resident and sweeps N before advancing M.
  kDownAMajorNFast = 1U,
  // The K5120 FP8 input projections retain two M128 slabs while sweeping the
  // grouped QKV/Z N256 supermatrix.  Two A slabs plus one B slab stay below
  // the 4 MiB Orin L2 budget.
  kFp8InputGroupedM2NMajor = 2U,
  // K6144 attention output retains one M128 slab.  A second slab would exceed
  // the 4 MiB L2 budget once the N256 B slab is included.
  kFp8OutputAMajorNFast = 3U,
};

enum class Sm87P40ProjectionResetFp8Role : std::uint8_t {
  kLinearQkvZInput = 0U,
  kFullQkvInput = 1U,
  kAttentionOutput = 2U,
};

inline constexpr std::size_t kSm87P40ProjectionResetTokens = 40'000U;
inline constexpr std::size_t kSm87P40ProjectionResetHidden = 5'120U;
inline constexpr std::size_t kSm87P40ProjectionResetIntermediate = 17'408U;
inline constexpr std::size_t kSm87P40ProjectionResetMergedGateUp = 34'816U;
inline constexpr std::size_t kSm87P40ProjectionResetTileM = 64U;
inline constexpr std::size_t kSm87P40ProjectionResetTileN = 128U;
inline constexpr std::size_t kSm87P40ProjectionResetTileK = 32U;
inline constexpr std::size_t kSm87P40ProjectionResetThreads = 128U;
inline constexpr std::size_t kSm87P40ProjectionResetWarps = 4U;
inline constexpr std::size_t kSm87P40ProjectionResetPipelineStages = 3U;
inline constexpr std::size_t kSm87P40ProjectionResetPersistentCtas = 32U;
inline constexpr std::size_t kSm87P40ProjectionResetSmCount = 16U;
inline constexpr std::size_t kSm87P40ProjectionResetSharedABytesPerStage =
    kSm87P40ProjectionResetTileM * kSm87P40ProjectionResetTileK *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87P40ProjectionResetSharedBBytesPerStage =
    kSm87P40ProjectionResetTileN * kSm87P40ProjectionResetTileK / 2U;
inline constexpr std::size_t kSm87P40ProjectionResetSharedScaleBytesPerStage =
    kSm87P40ProjectionResetTileN * kSm87P40ProjectionResetTileK / 16U;
inline constexpr std::size_t kSm87P40ProjectionResetMainloopSharedBytes =
    kSm87P40ProjectionResetPipelineStages *
    (kSm87P40ProjectionResetSharedABytesPerStage +
     kSm87P40ProjectionResetSharedBBytesPerStage +
     kSm87P40ProjectionResetSharedScaleBytesPerStage);
inline constexpr std::size_t
    kSm87P40ProjectionResetAlignedMainloopSharedBytes =
        (kSm87P40ProjectionResetMainloopSharedBytes + 1'023U) & ~1'023U;
inline constexpr std::size_t kSm87P40ProjectionResetGateEpilogueSharedBytes =
    kSm87P40ProjectionResetTileM * kSm87P40ProjectionResetTileN *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87P40ProjectionResetDynamicSharedBytes =
    kSm87P40ProjectionResetAlignedMainloopSharedBytes >
            kSm87P40ProjectionResetGateEpilogueSharedBytes
        ? kSm87P40ProjectionResetAlignedMainloopSharedBytes
        : kSm87P40ProjectionResetGateEpilogueSharedBytes;

inline constexpr std::size_t kSm87P40ProjectionResetFp8TileM = 128U;
inline constexpr std::size_t kSm87P40ProjectionResetFp8TileN = 256U;
inline constexpr std::size_t kSm87P40ProjectionResetFp8TileK = 64U;
inline constexpr std::size_t kSm87P40ProjectionResetFp8Threads = 256U;
inline constexpr std::size_t kSm87P40ProjectionResetFp8Warps = 8U;
inline constexpr std::size_t kSm87P40ProjectionResetFp8PipelineStages = 3U;
inline constexpr std::size_t kSm87P40ProjectionResetFp8PersistentCtas = 16U;
inline constexpr std::size_t kSm87P40ProjectionResetFp8SharedABytesPerStage =
    kSm87P40ProjectionResetFp8TileM *
    kSm87P40ProjectionResetFp8TileK * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87P40ProjectionResetFp8SharedBBytesPerStage =
    kSm87P40ProjectionResetFp8TileN *
    kSm87P40ProjectionResetFp8TileK;
inline constexpr std::size_t kSm87P40ProjectionResetFp8DynamicSharedBytes =
    kSm87P40ProjectionResetFp8PipelineStages *
    (kSm87P40ProjectionResetFp8SharedABytesPerStage +
     kSm87P40ProjectionResetFp8SharedBBytesPerStage);

static_assert(kSm87P40ProjectionResetTokens %
                      kSm87P40ProjectionResetTileM ==
                  0U);
static_assert(kSm87P40ProjectionResetIntermediate % 64U == 0U);
static_assert(kSm87P40ProjectionResetHidden %
                      kSm87P40ProjectionResetTileN ==
                  0U);
static_assert(kSm87P40ProjectionResetThreads ==
              32U * kSm87P40ProjectionResetWarps);
static_assert(kSm87P40ProjectionResetSharedABytesPerStage == 4'096U);
static_assert(kSm87P40ProjectionResetSharedBBytesPerStage == 2'048U);
static_assert(kSm87P40ProjectionResetSharedScaleBytesPerStage == 256U);
static_assert(kSm87P40ProjectionResetMainloopSharedBytes == 19'200U);
static_assert(kSm87P40ProjectionResetAlignedMainloopSharedBytes == 19'456U);
static_assert(kSm87P40ProjectionResetGateEpilogueSharedBytes == 16'384U);
static_assert(kSm87P40ProjectionResetDynamicSharedBytes == 19'456U);
static_assert(kSm87P40ProjectionResetFp8Threads ==
              32U * kSm87P40ProjectionResetFp8Warps);
static_assert(kSm87P40ProjectionResetFp8SharedABytesPerStage == 16'384U);
static_assert(kSm87P40ProjectionResetFp8SharedBBytesPerStage == 16'384U);
static_assert(kSm87P40ProjectionResetFp8DynamicSharedBytes == 98'304U);

struct Sm87P40ProjectionResetNvFp4Plan {
  Sm87P40ProjectionResetNvFp4Role role =
      Sm87P40ProjectionResetNvFp4Role::kInterleavedGateUpSilu;
  Sm87P40ProjectionResetRaster raster =
      Sm87P40ProjectionResetRaster::kGateUpGroupedM4NMajor;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t packed_output_features = 0U;
  std::size_t published_output_features = 0U;
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t logical_tasks = 0U;
  bool fused_silu = false;
  bool fused_residual = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    const bool gate = role ==
                      Sm87P40ProjectionResetNvFp4Role::
                          kInterleavedGateUpSilu;
    const bool down =
        role == Sm87P40ProjectionResetNvFp4Role::kDownResidual;
    return (gate || down) && token_count == kSm87P40ProjectionResetTokens &&
           raster ==
               (gate ? Sm87P40ProjectionResetRaster::kGateUpGroupedM4NMajor
                     : Sm87P40ProjectionResetRaster::kDownAMajorNFast) &&
           input_features ==
               (gate ? kSm87P40ProjectionResetHidden
                     : kSm87P40ProjectionResetIntermediate) &&
           packed_output_features ==
               (gate ? kSm87P40ProjectionResetMergedGateUp
                     : kSm87P40ProjectionResetHidden) &&
           published_output_features ==
               (gate ? kSm87P40ProjectionResetIntermediate
                     : kSm87P40ProjectionResetHidden) &&
           grid_m == 625U && grid_n == (gate ? 272U : 40U) &&
           logical_tasks == grid_m * grid_n && fused_silu == gate &&
           fused_residual == down;
  }
};

struct Sm87P40ProjectionResetTask {
  std::size_t m_tile = 0U;
  std::size_t n_tile = 0U;
  bool valid = false;
};

struct Sm87P40ProjectionResetFp8Plan {
  Sm87P40ProjectionResetFp8Role role =
      Sm87P40ProjectionResetFp8Role::kLinearQkvZInput;
  Sm87P40ProjectionResetRaster raster =
      Sm87P40ProjectionResetRaster::kFp8InputGroupedM2NMajor;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t partition_count = 0U;
  std::size_t total_output_features = 0U;
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t group_m = 0U;
  std::size_t logical_tasks = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    const bool linear =
        role == Sm87P40ProjectionResetFp8Role::kLinearQkvZInput;
    const bool full = role == Sm87P40ProjectionResetFp8Role::kFullQkvInput;
    const bool output =
        role == Sm87P40ProjectionResetFp8Role::kAttentionOutput;
    return (linear || full || output) &&
           token_count == kSm87P40ProjectionResetTokens &&
           input_features == (output ? 6'144U : 5'120U) &&
           partition_count == (linear ? 2U : (full ? 3U : 1U)) &&
           total_output_features ==
               (linear ? 16'384U : (full ? 14'336U : 5'120U)) &&
           grid_m == 313U &&
           grid_n == total_output_features /
                         kSm87P40ProjectionResetFp8TileN &&
           group_m == (output ? 1U : 2U) &&
           raster ==
               (output ? Sm87P40ProjectionResetRaster::kFp8OutputAMajorNFast
                       : Sm87P40ProjectionResetRaster::
                             kFp8InputGroupedM2NMajor) &&
           logical_tasks == grid_m * grid_n;
  }
};

[[nodiscard]] constexpr Sm87P40ProjectionResetNvFp4Plan
sm87_p40_projection_reset_nvfp4_plan(
    const Sm87P40ProjectionResetNvFp4Role role,
    const std::size_t token_count) noexcept {
  if (token_count != kSm87P40ProjectionResetTokens) {
    return {};
  }
  const bool gate = role ==
                    Sm87P40ProjectionResetNvFp4Role::
                        kInterleavedGateUpSilu;
  if (!gate && role != Sm87P40ProjectionResetNvFp4Role::kDownResidual) {
    return {};
  }
  const std::size_t grid_n = gate ? 272U : 40U;
  return {role,
          gate ? Sm87P40ProjectionResetRaster::kGateUpGroupedM4NMajor
               : Sm87P40ProjectionResetRaster::kDownAMajorNFast,
          token_count,
          gate ? kSm87P40ProjectionResetHidden
               : kSm87P40ProjectionResetIntermediate,
          gate ? kSm87P40ProjectionResetMergedGateUp
               : kSm87P40ProjectionResetHidden,
          gate ? kSm87P40ProjectionResetIntermediate
               : kSm87P40ProjectionResetHidden,
          625U,
          grid_n,
          625U * grid_n,
          gate,
          !gate};
}

[[nodiscard]] constexpr Sm87P40ProjectionResetTask
sm87_p40_projection_reset_nvfp4_task(
    const Sm87P40ProjectionResetNvFp4Plan& plan,
    const std::size_t linear_task) noexcept {
  if (!plan.valid() || linear_task >= plan.logical_tasks) {
    return {};
  }
  if (plan.raster ==
      Sm87P40ProjectionResetRaster::kGateUpGroupedM4NMajor) {
    constexpr std::size_t kGroupM = 4U;
    const std::size_t group_span = kGroupM * plan.grid_n;
    const std::size_t group = linear_task / group_span;
    const std::size_t first_m = group * kGroupM;
    const std::size_t active_m =
        plan.grid_m - first_m < kGroupM ? plan.grid_m - first_m : kGroupM;
    const std::size_t group_offset = linear_task % group_span;
    return {first_m + group_offset % active_m,
            group_offset / active_m, true};
  }
  return {linear_task / plan.grid_n, linear_task % plan.grid_n, true};
}

[[nodiscard]] constexpr Sm87P40ProjectionResetFp8Plan
sm87_p40_projection_reset_fp8_plan(
    const Sm87P40ProjectionResetFp8Role role,
    const std::size_t token_count) noexcept {
  if (token_count != kSm87P40ProjectionResetTokens) {
    return {};
  }
  if (role == Sm87P40ProjectionResetFp8Role::kLinearQkvZInput) {
    return {role,
            Sm87P40ProjectionResetRaster::kFp8InputGroupedM2NMajor,
            token_count, 5'120U, 2U, 16'384U, 313U, 64U, 2U,
            313U * 64U};
  }
  if (role == Sm87P40ProjectionResetFp8Role::kFullQkvInput) {
    return {role,
            Sm87P40ProjectionResetRaster::kFp8InputGroupedM2NMajor,
            token_count, 5'120U, 3U, 14'336U, 313U, 56U, 2U,
            313U * 56U};
  }
  if (role == Sm87P40ProjectionResetFp8Role::kAttentionOutput) {
    return {role,
            Sm87P40ProjectionResetRaster::kFp8OutputAMajorNFast,
            token_count, 6'144U, 1U, 5'120U, 313U, 20U, 1U,
            313U * 20U};
  }
  return {};
}

[[nodiscard]] constexpr Sm87P40ProjectionResetTask
sm87_p40_projection_reset_fp8_task(
    const Sm87P40ProjectionResetFp8Plan& plan,
    const std::size_t linear_task) noexcept {
  if (!plan.valid() || linear_task >= plan.logical_tasks) {
    return {};
  }
  const std::size_t group_span = plan.group_m * plan.grid_n;
  const std::size_t group = linear_task / group_span;
  const std::size_t first_m = group * plan.group_m;
  const std::size_t active_m =
      plan.grid_m - first_m < plan.group_m ? plan.grid_m - first_m
                                           : plan.group_m;
  const std::size_t group_offset = linear_task % group_span;
  return {first_m + group_offset % active_m,
          group_offset / active_m, true};
}

struct Sm87P40ProjectionResetCapability {
  Sm87P40ProjectionResetNvFp4Plan plan{};
  int device = -1;
  int compute_major = 0;
  int compute_minor = 0;
  int sm_count = 0;
  std::size_t shared_bytes_per_sm = 0U;
  bool supported = false;
};

struct Sm87P40ProjectionResetFp8Capability {
  Sm87P40ProjectionResetFp8Plan plan{};
  int device = -1;
  int compute_major = 0;
  int compute_minor = 0;
  int sm_count = 0;
  std::size_t shared_bytes_per_sm = 0U;
  std::size_t opt_in_shared_bytes_per_block = 0U;
  bool supported = false;
};

struct Sm87P40ProjectionResetResources {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

int query_sm87_p40_projection_reset_nvfp4_capability_cuda(
    Sm87P40ProjectionResetNvFp4Role role, std::size_t token_count,
    Sm87P40ProjectionResetCapability* capability) noexcept;

int query_sm87_p40_projection_reset_nvfp4_resources_cuda(
    Sm87P40ProjectionResetNvFp4Role role, std::size_t token_count,
    Sm87P40ProjectionResetResources* resources) noexcept;

int query_sm87_p40_projection_reset_fp8_capability_cuda(
    Sm87P40ProjectionResetFp8Role role, std::size_t token_count,
    Sm87P40ProjectionResetFp8Capability* capability) noexcept;

int query_sm87_p40_projection_reset_fp8_resources_cuda(
    Sm87P40ProjectionResetFp8Role role, std::size_t token_count,
    Sm87P40ProjectionResetResources* resources) noexcept;

int launch_sm87_p40_projection_reset_nvfp4_gate_up_silu_cuda(
    const std::uint16_t* input,
    const std::uint8_t* interleaved_marlin_weight,
    const std::uint8_t* interleaved_marlin_scales,
    const float* marlin_global_scale, std::size_t token_count,
    std::uint16_t* activated_output, void* cuda_stream) noexcept;

int launch_sm87_p40_projection_reset_nvfp4_down_residual_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales, const float* marlin_global_scale,
    std::size_t token_count, std::uint16_t* residual_in_out,
    void* cuda_stream) noexcept;

int launch_sm87_p40_projection_reset_fp8_supermatrix_cuda(
    const Sm87Fp8PrefillSupermatrixPartition* partitions,
    std::size_t partition_count, const std::uint16_t* activations,
    std::size_t token_count, std::size_t columns,
    void* cuda_stream) noexcept;

}  // namespace q3x::kernels
