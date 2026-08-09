#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// This is a deliberately narrow Qwen3.6-27B/SM87 admission surface.  It is
// not a generic GEMM API and it never falls back to the Marlin M64/K-split
// path.  Only the two production prefill panels are representable.
enum class Sm87NvFp4PrefillLargeMRole : std::uint8_t {
  kGateUp = 0U,
  kDown = 1U,
};

enum class Sm87NvFp4PrefillLargeMDataflow : std::uint8_t {
  kGateUpAStationaryRaster = 0U,
  kDownBStationaryRaster = 1U,
};

inline constexpr std::size_t kSm87NvFp4PrefillLargeMMaximumTokens = 8'192U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMShortPanelTokens = 7'712U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMHidden = 5'120U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMIntermediate = 17'408U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMGateUpOutput = 34'816U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMTileM = 128U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMTileN = 256U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMTileK = 64U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMThreads = 256U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMPipelineStages = 3U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMSmCount = 16U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMSharedABytesPerStage =
    kSm87NvFp4PrefillLargeMTileM * 72U * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87NvFp4PrefillLargeMSharedBBytesPerStage =
    kSm87NvFp4PrefillLargeMTileN * kSm87NvFp4PrefillLargeMTileK / 2U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMSharedScaleBytesPerStage =
    kSm87NvFp4PrefillLargeMTileN * kSm87NvFp4PrefillLargeMTileK / 16U;
inline constexpr std::size_t kSm87NvFp4PrefillLargeMDynamicSharedBytes =
    kSm87NvFp4PrefillLargeMPipelineStages *
    (kSm87NvFp4PrefillLargeMSharedABytesPerStage +
     kSm87NvFp4PrefillLargeMSharedBBytesPerStage +
     kSm87NvFp4PrefillLargeMSharedScaleBytesPerStage);

static_assert(kSm87NvFp4PrefillLargeMDynamicSharedBytes == 82'944U);
static_assert(kSm87NvFp4PrefillLargeMShortPanelTokens %
                      kSm87NvFp4PrefillLargeMTileM ==
                  32U,
              "M7712 must use the native masked M32 tail");

struct Sm87NvFp4PrefillLargeMPlan {
  Sm87NvFp4PrefillLargeMRole role =
      Sm87NvFp4PrefillLargeMRole::kGateUp;
  Sm87NvFp4PrefillLargeMDataflow dataflow =
      Sm87NvFp4PrefillLargeMDataflow::kGateUpAStationaryRaster;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t output_features = 0U;
  std::size_t tile_m = 0U;
  std::size_t tile_n = 0U;
  std::size_t tile_k = 0U;
  std::size_t threads = 0U;
  std::size_t pipeline_stages = 0U;
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t tail_rows = 0U;
  std::size_t dynamic_shared_bytes = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return token_count != 0U && input_features != 0U &&
           output_features != 0U && tile_m == kSm87NvFp4PrefillLargeMTileM &&
           tile_n == kSm87NvFp4PrefillLargeMTileN &&
           tile_k == kSm87NvFp4PrefillLargeMTileK &&
           threads == kSm87NvFp4PrefillLargeMThreads &&
           pipeline_stages == kSm87NvFp4PrefillLargeMPipelineStages &&
           grid_m != 0U && grid_n != 0U &&
           dynamic_shared_bytes ==
               kSm87NvFp4PrefillLargeMDynamicSharedBytes;
  }
};

[[nodiscard]] constexpr bool sm87_nvfp4_prefill_large_m_supports(
    const Sm87NvFp4PrefillLargeMRole role,
    const std::size_t token_count) noexcept {
  const bool valid_role = role == Sm87NvFp4PrefillLargeMRole::kGateUp ||
                          role == Sm87NvFp4PrefillLargeMRole::kDown;
  return valid_role &&
         (token_count == kSm87NvFp4PrefillLargeMMaximumTokens ||
          token_count == kSm87NvFp4PrefillLargeMShortPanelTokens);
}

[[nodiscard]] constexpr Sm87NvFp4PrefillLargeMPlan
sm87_nvfp4_prefill_large_m_plan(
    const Sm87NvFp4PrefillLargeMRole role,
    const std::size_t token_count) noexcept {
  if (!sm87_nvfp4_prefill_large_m_supports(role, token_count)) {
    return {};
  }
  const std::size_t input_features =
      role == Sm87NvFp4PrefillLargeMRole::kGateUp
          ? kSm87NvFp4PrefillLargeMHidden
          : kSm87NvFp4PrefillLargeMIntermediate;
  const std::size_t output_features =
      role == Sm87NvFp4PrefillLargeMRole::kGateUp
          ? kSm87NvFp4PrefillLargeMGateUpOutput
          : kSm87NvFp4PrefillLargeMHidden;
  return {role,
          role == Sm87NvFp4PrefillLargeMRole::kGateUp
              ? Sm87NvFp4PrefillLargeMDataflow::kGateUpAStationaryRaster
              : Sm87NvFp4PrefillLargeMDataflow::kDownBStationaryRaster,
          token_count,
          input_features,
          output_features,
          kSm87NvFp4PrefillLargeMTileM,
          kSm87NvFp4PrefillLargeMTileN,
          kSm87NvFp4PrefillLargeMTileK,
          kSm87NvFp4PrefillLargeMThreads,
          kSm87NvFp4PrefillLargeMPipelineStages,
          (token_count + kSm87NvFp4PrefillLargeMTileM - 1U) /
              kSm87NvFp4PrefillLargeMTileM,
          output_features / kSm87NvFp4PrefillLargeMTileN,
          token_count % kSm87NvFp4PrefillLargeMTileM,
          kSm87NvFp4PrefillLargeMDynamicSharedBytes};
}

struct Sm87NvFp4PrefillLargeMCapability {
  Sm87NvFp4PrefillLargeMPlan plan{};
  int device = -1;
  int compute_major = 0;
  int compute_minor = 0;
  int sm_count = 0;
  std::size_t optin_shared_bytes_per_block = 0U;
  bool supported = false;
};

struct Sm87NvFp4PrefillLargeMResources {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

// All CUDA-facing functions return a cudaError_t encoded as int so this
// public header stays CUDA-header neutral. Unsupported devices/shapes or an
// incomplete admission cell return cudaErrorNotSupported. Pointer, alias, or
// alignment violations return cudaErrorInvalidValue.
int query_sm87_nvfp4_prefill_large_m_capability_cuda(
    Sm87NvFp4PrefillLargeMRole role, std::size_t token_count,
    Sm87NvFp4PrefillLargeMCapability* capability) noexcept;

int query_sm87_nvfp4_prefill_large_m_resources_cuda(
    Sm87NvFp4PrefillLargeMRole role, std::size_t token_count,
    Sm87NvFp4PrefillLargeMResources* resources) noexcept;

int launch_sm87_nvfp4_prefill_large_m_gate_up_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales, const float* marlin_global_scale,
    std::size_t token_count, std::uint16_t* output,
    void* cuda_stream) noexcept;

int launch_sm87_nvfp4_prefill_large_m_down_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales, const float* marlin_global_scale,
    std::size_t token_count, std::uint16_t* output,
    void* cuda_stream) noexcept;

}  // namespace q3x::kernels
