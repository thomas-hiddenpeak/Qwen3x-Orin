#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// A deliberately model- and device-specific admission surface for the second
// true-large-M NVFP4 Prefill architecture. There is no generic fallback behind
// this API. Gate+Up consumes the canonical GateThenUp Marlin sidecar and Down
// publishes its residual result in place.
enum class Sm87NvFp4PrefillG2D2Role : std::uint8_t {
  kGateUpG2 = 0U,
  kDownD2 = 1U,
};

enum class Sm87NvFp4PrefillG2D2Dataflow : std::uint8_t {
  kGateUpM128PairRaster = 0U,
  kDownM128N128BStationaryRaster = 1U,
};

inline constexpr std::size_t kSm87NvFp4PrefillG2D2MaximumTokens = 8'192U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2ShortPanelTokens = 7'712U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2Hidden = 5'120U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2Intermediate = 17'408U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2MergedGateUp = 34'816U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2TileM = 128U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2PhysicalTileN = 128U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2GateUpBranchTileN = 64U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2DownBranchTileN = 128U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2TileK = 64U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2Threads = 256U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2PipelineStages = 2U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2SmCount = 16U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2MaximumRegisters = 128U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2MinimumBlocksPerSm = 2U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2SharedLimitBytes = 50U * 1'024U;

inline constexpr std::size_t kSm87NvFp4PrefillG2D2SharedABytesPerStage =
    kSm87NvFp4PrefillG2D2TileM * kSm87NvFp4PrefillG2D2TileK *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87NvFp4PrefillG2D2SharedBBytesPerStage =
    kSm87NvFp4PrefillG2D2PhysicalTileN *
    kSm87NvFp4PrefillG2D2TileK / 2U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2SharedScaleBytesPerStage =
    kSm87NvFp4PrefillG2D2PhysicalTileN *
    kSm87NvFp4PrefillG2D2TileK / 16U;
inline constexpr std::size_t kSm87NvFp4PrefillG2D2DynamicSharedBytes =
    kSm87NvFp4PrefillG2D2PipelineStages *
    (kSm87NvFp4PrefillG2D2SharedABytesPerStage +
     kSm87NvFp4PrefillG2D2SharedBBytesPerStage +
     kSm87NvFp4PrefillG2D2SharedScaleBytesPerStage);

static_assert(kSm87NvFp4PrefillG2D2SharedABytesPerStage == 16'384U);
static_assert(kSm87NvFp4PrefillG2D2SharedBBytesPerStage == 4'096U);
static_assert(kSm87NvFp4PrefillG2D2SharedScaleBytesPerStage == 512U);
static_assert(kSm87NvFp4PrefillG2D2DynamicSharedBytes == 41'984U);
static_assert(kSm87NvFp4PrefillG2D2DynamicSharedBytes <=
              kSm87NvFp4PrefillG2D2SharedLimitBytes);
static_assert(kSm87NvFp4PrefillG2D2ShortPanelTokens %
                      kSm87NvFp4PrefillG2D2TileM ==
                  32U,
              "M7712 must use the native masked M32 tail");

struct Sm87NvFp4PrefillG2D2Plan {
  Sm87NvFp4PrefillG2D2Role role =
      Sm87NvFp4PrefillG2D2Role::kGateUpG2;
  Sm87NvFp4PrefillG2D2Dataflow dataflow =
      Sm87NvFp4PrefillG2D2Dataflow::kGateUpM128PairRaster;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t weight_output_features = 0U;
  std::size_t published_output_features = 0U;
  std::size_t tile_m = 0U;
  std::size_t branch_tile_n = 0U;
  std::size_t tile_k = 0U;
  std::size_t threads = 0U;
  std::size_t pipeline_stages = 0U;
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t tail_rows = 0U;
  std::size_t dynamic_shared_bytes = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    const bool gate = role == Sm87NvFp4PrefillG2D2Role::kGateUpG2;
    const bool down = role == Sm87NvFp4PrefillG2D2Role::kDownD2;
    const bool supported_tokens =
        token_count == kSm87NvFp4PrefillG2D2MaximumTokens ||
        token_count == kSm87NvFp4PrefillG2D2ShortPanelTokens;
    const std::size_t expected_grid_m =
        supported_tokens
            ? (token_count + kSm87NvFp4PrefillG2D2TileM - 1U) /
                  kSm87NvFp4PrefillG2D2TileM
            : 0U;
    return (gate || down) && supported_tokens &&
           dataflow ==
               (gate ? Sm87NvFp4PrefillG2D2Dataflow::
                           kGateUpM128PairRaster
                     : Sm87NvFp4PrefillG2D2Dataflow::
                           kDownM128N128BStationaryRaster) &&
           input_features ==
               (gate ? kSm87NvFp4PrefillG2D2Hidden
                     : kSm87NvFp4PrefillG2D2Intermediate) &&
           weight_output_features ==
               (gate ? kSm87NvFp4PrefillG2D2MergedGateUp
                     : kSm87NvFp4PrefillG2D2Hidden) &&
           published_output_features ==
               (gate ? kSm87NvFp4PrefillG2D2Intermediate
                     : kSm87NvFp4PrefillG2D2Hidden) &&
           tile_m == kSm87NvFp4PrefillG2D2TileM &&
           branch_tile_n ==
               (gate ? kSm87NvFp4PrefillG2D2GateUpBranchTileN
                     : kSm87NvFp4PrefillG2D2DownBranchTileN) &&
           tile_k == kSm87NvFp4PrefillG2D2TileK &&
           threads == kSm87NvFp4PrefillG2D2Threads &&
           pipeline_stages == kSm87NvFp4PrefillG2D2PipelineStages &&
           grid_m == expected_grid_m && grid_n == (gate ? 272U : 40U) &&
           tail_rows == token_count % kSm87NvFp4PrefillG2D2TileM &&
           dynamic_shared_bytes == kSm87NvFp4PrefillG2D2DynamicSharedBytes;
  }
};

[[nodiscard]] constexpr bool sm87_nvfp4_prefill_g2_d2_supports(
    const Sm87NvFp4PrefillG2D2Role role,
    const std::size_t token_count) noexcept {
  const bool valid_role = role == Sm87NvFp4PrefillG2D2Role::kGateUpG2 ||
                          role == Sm87NvFp4PrefillG2D2Role::kDownD2;
  return valid_role &&
         (token_count == kSm87NvFp4PrefillG2D2MaximumTokens ||
          token_count == kSm87NvFp4PrefillG2D2ShortPanelTokens);
}

[[nodiscard]] constexpr Sm87NvFp4PrefillG2D2Plan
sm87_nvfp4_prefill_g2_d2_plan(
    const Sm87NvFp4PrefillG2D2Role role,
    const std::size_t token_count) noexcept {
  if (!sm87_nvfp4_prefill_g2_d2_supports(role, token_count)) {
    return {};
  }
  const bool gate = role == Sm87NvFp4PrefillG2D2Role::kGateUpG2;
  const std::size_t input_features =
      gate ? kSm87NvFp4PrefillG2D2Hidden
           : kSm87NvFp4PrefillG2D2Intermediate;
  const std::size_t weight_output_features =
      gate ? kSm87NvFp4PrefillG2D2MergedGateUp
           : kSm87NvFp4PrefillG2D2Hidden;
  const std::size_t published_output_features =
      gate ? kSm87NvFp4PrefillG2D2Intermediate
           : kSm87NvFp4PrefillG2D2Hidden;
  const std::size_t branch_tile_n =
      gate ? kSm87NvFp4PrefillG2D2GateUpBranchTileN
           : kSm87NvFp4PrefillG2D2DownBranchTileN;
  return {role,
          gate ? Sm87NvFp4PrefillG2D2Dataflow::kGateUpM128PairRaster
               : Sm87NvFp4PrefillG2D2Dataflow::
                     kDownM128N128BStationaryRaster,
          token_count,
          input_features,
          weight_output_features,
          published_output_features,
          kSm87NvFp4PrefillG2D2TileM,
          branch_tile_n,
          kSm87NvFp4PrefillG2D2TileK,
          kSm87NvFp4PrefillG2D2Threads,
          kSm87NvFp4PrefillG2D2PipelineStages,
          (token_count + kSm87NvFp4PrefillG2D2TileM - 1U) /
              kSm87NvFp4PrefillG2D2TileM,
          published_output_features / branch_tile_n,
          token_count % kSm87NvFp4PrefillG2D2TileM,
          kSm87NvFp4PrefillG2D2DynamicSharedBytes};
}

struct Sm87NvFp4PrefillG2D2Capability {
  Sm87NvFp4PrefillG2D2Plan plan{};
  int device = -1;
  int compute_major = 0;
  int compute_minor = 0;
  int sm_count = 0;
  std::size_t optin_shared_bytes_per_block = 0U;
  bool supported = false;
};

struct Sm87NvFp4PrefillG2D2Resources {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

// CUDA-facing functions return cudaError_t encoded as int so this public
// header remains CUDA-header neutral. Unsupported shapes/devices, a disabled
// admission build, or a kernel outside the fixed resource envelope return
// cudaErrorNotSupported. Pointer, alignment, or alias violations return
// cudaErrorInvalidValue.
int query_sm87_nvfp4_prefill_g2_d2_capability_cuda(
    Sm87NvFp4PrefillG2D2Role role, std::size_t token_count,
    Sm87NvFp4PrefillG2D2Capability* capability) noexcept;

int query_sm87_nvfp4_prefill_g2_d2_resources_cuda(
    Sm87NvFp4PrefillG2D2Role role, std::size_t token_count,
    Sm87NvFp4PrefillG2D2Resources* resources) noexcept;

int launch_sm87_nvfp4_prefill_gate_up_g2_cuda(
    const std::uint16_t* input,
    const std::uint8_t* merged_marlin_weight,
    const std::uint8_t* merged_marlin_scales,
    const float* marlin_global_scale, std::size_t token_count,
    std::uint16_t* activated, void* cuda_stream) noexcept;

int launch_sm87_nvfp4_prefill_down_d2_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales,
    const float* marlin_global_scale, const std::uint16_t* residual,
    std::size_t token_count, std::uint16_t* output,
    void* cuda_stream) noexcept;

}  // namespace q3x::kernels
