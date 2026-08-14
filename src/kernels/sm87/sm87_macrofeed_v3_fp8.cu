#include "q3x/kernels/sm87_macrofeed_v3_fp8.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87TargetAotProjectionThreads);
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kTileM =
    static_cast<unsigned int>(kSm87TargetAotProjectionBlockM);
constexpr unsigned int kTileN =
    static_cast<unsigned int>(kSm87TargetAotProjectionBlockN);
constexpr unsigned int kTileK =
    static_cast<unsigned int>(kSm87TargetAotProjectionBlockK);
constexpr unsigned int kStages =
    static_cast<unsigned int>(kSm87TargetAotProjectionPipelineStages);
constexpr unsigned int kPersistentCtas =
    static_cast<unsigned int>(kSm87TargetAotProjectionPersistentCtas);
constexpr unsigned int kWarpM = 128U;
constexpr unsigned int kWarpN = 32U;
constexpr unsigned int kM16Panels = kWarpM / 16U;
constexpr unsigned int kN8Panels = kWarpN / 8U;
constexpr unsigned int kN8PanelsPerN64 = 8U;
constexpr unsigned int kK16Panels = kTileK / 16U;
constexpr unsigned int kActivationVectors = kTileM * kTileK / 8U;
constexpr unsigned int kWeightBytes = kTileN * kTileK;
constexpr unsigned int kWeightVectors = kWeightBytes / sizeof(uint4);
constexpr auto kGdnLayout = sm87_target_aot_projection_packed_layout(
    Sm87TargetAotProjectionRole::kFp8GdnQkvZ);
constexpr auto kFullLayout = sm87_target_aot_projection_packed_layout(
    Sm87TargetAotProjectionRole::kFp8FullQkv);
constexpr auto kOutputLayout = sm87_target_aot_projection_packed_layout(
    Sm87TargetAotProjectionRole::kFp8AttentionOutput);

static_assert(kThreads == 256U);
static_assert(kTileM == 128U && kTileN == 256U && kTileK == 64U);
static_assert(kStages == 3U && kPersistentCtas == 16U);
static_assert(kWarpM == 128U && kWarpN == 32U &&
              kM16Panels == 8U && kN8Panels == 4U);
static_assert(kActivationVectors == 1'024U && kWeightVectors == 1'024U);
static_assert(kGdnLayout.valid() && kGdnLayout.partition_count == 2U &&
              kGdnLayout.partitions[0U].n_tiles == 40U &&
              kGdnLayout.partitions[1U].n_tiles == 24U &&
              kGdnLayout.partitions[1U].payload_offset == 52'428'800U);
static_assert(kFullLayout.valid() && kFullLayout.partition_count == 3U &&
              kFullLayout.partitions[0U].n_tiles == 48U &&
              kFullLayout.partitions[1U].n_tiles == 4U &&
              kFullLayout.partitions[2U].n_tiles == 4U &&
              kFullLayout.partitions[1U].payload_offset == 62'914'560U &&
              kFullLayout.partitions[2U].payload_offset == 68'157'440U);
static_assert(kOutputLayout.valid() &&
              kOutputLayout.partition_count == 1U &&
              kOutputLayout.partitions[0U].n_tiles == 20U);

struct alignas(32) Fp8PipelineStorage final {
  uint4 activations[kStages][kActivationVectors];
  uint4 weights[kStages][kWeightVectors];
};

static_assert(sizeof(Fp8PipelineStorage) == kSm87TargetAotFp8SharedBytes);

struct M16K16Activation final {
  std::uint32_t x0;
  std::uint32_t x1;
  std::uint32_t x2;
  std::uint32_t x3;
};

struct K16N8Weight final {
  std::uint32_t x0;
  std::uint32_t x1;
};

struct M16N8Accumulator final {
  float x0;
  float x1;
  float x2;
  float x3;
};

using WarpAccumulator = M16N8Accumulator[kM16Panels][kN8Panels];
using DecodedWeightStage = K16N8Weight[kN8Panels];

template <bool kPredicate>
__device__ __forceinline__ void cp_async_cg_16(
    void* const shared_destination, const void* const global_source,
    const bool valid = true) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));
  if constexpr (kPredicate) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :
                 : "r"(shared_address), "l"(global_source),
                   "r"(valid ? 16U : 0U)
                 : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :
                 : "r"(shared_address), "l"(global_source)
                 : "memory");
  }
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      valid ? *reinterpret_cast<const uint4*>(global_source)
            : make_uint4(0U, 0U, 0U, 0U);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <unsigned int kOutstanding>
__device__ __forceinline__ void cp_async_wait_group() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;"
               :
               : "n"(kOutstanding)
               : "memory");
#endif
}

__device__ __forceinline__ void load_activation_fragment(
    M16K16Activation& fragment,
    const std::uint16_t* const shared_activations,
    const unsigned int m16, const unsigned int k16,
    const unsigned int lane) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row = m16 * 16U + lane % 8U +
                           (quadrant & 1U) * 8U;
  const unsigned int column = k16 * 16U + (quadrant >> 1U) * 8U;
  const auto* const source = shared_activations + row * kTileK + column;
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1), "=r"(fragment.x2),
        "=r"(fragment.x3)
      : "r"(shared_address)
      : "memory");
#else
  (void)fragment;
  (void)shared_activations;
  (void)m16;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    M16N8Accumulator& accumulator,
    const M16K16Activation& activation,
    const K16N8Weight& weight) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(activation.x0), "r"(activation.x1), "r"(activation.x2),
        "r"(activation.x3), "r"(weight.x0), "r"(weight.x1));
#else
  (void)accumulator;
  (void)activation;
  (void)weight;
#endif
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_rne(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ K16N8Weight
decode_weight_fragment(
    const std::uint8_t* const shared_weight, const unsigned int k16,
    const unsigned int n64_warp, const unsigned int n8_from_n64,
    const unsigned int lane) noexcept {
  const unsigned int fragment =
      (k16 * 4U + n64_warp) * kN8PanelsPerN64 + n8_from_n64;
  const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
      shared_weight + fragment * 128U + lane * 4U);
  // Identical raw Marlin E4M3FN -> BF16 bias-shift feed.  The byte is moved
  // into BF16 sign/exponent/mantissa position without canonical FP8
  // conversion, so terminal codes 0x7f/0xff remain +/-480 after the final
  // 2^120-compensated tensor scale rather than becoming NaN.
  const auto raw_bf16 = [packed](const unsigned int packed_component) {
    const std::uint8_t code =
        static_cast<std::uint8_t>(packed >> (8U * packed_component));
    return static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(code & 0x80U) << 8U) |
        (static_cast<std::uint16_t>(code & 0x7fU) << 4U));
  };
  // Persisted components [K0,K8,K1,K9] become MMA register halves
  // [K0,K1] and [K8,K9].
  const std::uint16_t component0 = raw_bf16(0U);
  const std::uint16_t component1 = raw_bf16(2U);
  const std::uint16_t component2 = raw_bf16(1U);
  const std::uint16_t component3 = raw_bf16(3U);
  return {static_cast<std::uint32_t>(component0) |
              (static_cast<std::uint32_t>(component1) << 16U),
          static_cast<std::uint32_t>(component2) |
              (static_cast<std::uint32_t>(component3) << 16U)};
}

__device__ __forceinline__ void clear_accumulators(
    WarpAccumulator& accumulators) noexcept {
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      accumulators[m16][n8] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }
}

template <unsigned int kInputFeatures, unsigned int kKTiles>
__device__ __forceinline__ void issue_pipeline_stage(
    Fp8PipelineStorage* const storage, const unsigned int slot,
    const std::uint16_t* const input, const std::uint8_t* const payload,
    const unsigned int rows, const unsigned int first_m,
    const unsigned int partition_offset, const unsigned int partition_n_tile,
    const unsigned int k_tile) noexcept {
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const unsigned int row = vector_index / (kTileK / 8U);
    const unsigned int vector = vector_index % (kTileK / 8U);
    const bool row_valid = first_m + row < rows;
    const unsigned int source_row = row_valid ? first_m + row : 0U;
    const auto* const source = reinterpret_cast<const uint4*>(
                                   input +
                                   static_cast<std::size_t>(source_row) *
                                       kInputFeatures +
                                   k_tile * kTileK) +
                               vector;
    cp_async_cg_16<true>(&storage->activations[slot][vector_index], source,
                         row_valid);
  }
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const std::uint64_t cell =
        static_cast<std::uint64_t>(partition_offset) +
        (static_cast<std::uint64_t>(partition_n_tile) * kKTiles + k_tile) *
            kWeightBytes;
    const auto* const source =
        reinterpret_cast<const uint4*>(payload + cell) + vector_index;
    cp_async_cg_16<false>(&storage->weights[slot][vector_index], source);
  }
  cp_async_commit_group();
}

template <unsigned int kInputFeatures, unsigned int kKTiles>
__device__ __forceinline__ void run_full_k(
    Fp8PipelineStorage* const storage, const std::uint16_t* const input,
    const std::uint8_t* const payload, const unsigned int rows,
    const unsigned int first_m, const unsigned int partition_offset,
    const unsigned int partition_n_tile, WarpAccumulator& accumulators)
    noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  // Two phase-independent physical warps divide each N64 owner into N32
  // halves.  Every B fragment is decoded exactly once while each warp retains
  // the complete M128 accumulator lifetime.
  const unsigned int n64_warp = warp >> 1U;
  const unsigned int first_n8 = (warp & 1U) * kN8Panels;
  clear_accumulators(accumulators);
  issue_pipeline_stage<kInputFeatures, kKTiles>(
      storage, 0U, input, payload, rows, first_m, partition_offset,
      partition_n_tile, 0U);
  issue_pipeline_stage<kInputFeatures, kKTiles>(
      storage, 1U, input, payload, rows, first_m, partition_offset,
      partition_n_tile, 1U);
  issue_pipeline_stage<kInputFeatures, kKTiles>(
      storage, 2U, input, payload, rows, first_m, partition_offset,
      partition_n_tile, 2U);

#pragma unroll 1
  for (unsigned int k_tile = 0U; k_tile < kKTiles; ++k_tile) {
    if (k_tile + 2U < kKTiles) {
      cp_async_wait_group<2U>();
    } else if (k_tile + 1U < kKTiles) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();
    const unsigned int slot = k_tile % kStages;
    const auto* const shared_activations =
        reinterpret_cast<const std::uint16_t*>(storage->activations[slot]);
    const auto* const shared_weight =
        reinterpret_cast<const std::uint8_t*>(storage->weights[slot]);

    DecodedWeightStage decoded[2U];
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      decoded[0U][n8] = decode_weight_fragment(
          shared_weight, 0U, n64_warp, first_n8 + n8, lane);
    }
#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16Panels; ++k16) {
      const unsigned int current = k16 & 1U;
      const unsigned int next = current ^ 1U;
      if (k16 + 1U < kK16Panels) {
#pragma unroll
        for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
          decoded[next][n8] = decode_weight_fragment(
              shared_weight, k16 + 1U, n64_warp,
              first_n8 + n8, lane);
        }
      }
      M16K16Activation activation[kM16Panels];
#pragma unroll
      for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
        load_activation_fragment(activation[m16], shared_activations,
                                 m16, k16, lane);
      }
      if (k16 + 1U == kK16Panels) {
        __syncthreads();
        if (k_tile + kStages < kKTiles) {
          issue_pipeline_stage<kInputFeatures, kKTiles>(
              storage, slot, input, payload, rows, first_m, partition_offset,
              partition_n_tile, k_tile + kStages);
        }
      }
#pragma unroll
      for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
#pragma unroll
        for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
          mma_m16n8k16_bf16(accumulators[m16][n8], activation[m16],
                            decoded[current][n8]);
        }
      }
    }
  }
  cp_async_wait_group<0U>();
  __syncthreads();
}

__device__ __forceinline__ void publish_output(
    const WarpAccumulator& accumulators, const unsigned int rows,
    const unsigned int output_features, const unsigned int first_m,
    const unsigned int first_n, const float compensated_scale,
    std::uint16_t* const output) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int n64_warp = warp >> 1U;
  const unsigned int first_n8 = (warp & 1U) * kN8Panels;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
    const unsigned int local_row0 =
        m16 * 16U + lane_group;
    const unsigned int local_row1 = local_row0 + 8U;
    const unsigned int global_row0 = first_m + local_row0;
    const unsigned int global_row1 = first_m + local_row1;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      const unsigned int column =
          first_n + n64_warp * 64U + (first_n8 + n8) * 8U +
          lane_in_group * 2U;
      const auto& value = accumulators[m16][n8];
      if (global_row0 < rows) {
        const std::uint32_t packed =
            static_cast<std::uint32_t>(
                encode_bf16_rne(value.x0 * compensated_scale)) |
            (static_cast<std::uint32_t>(
                 encode_bf16_rne(value.x1 * compensated_scale))
             << 16U);
        *reinterpret_cast<std::uint32_t*>(
            output + static_cast<std::size_t>(global_row0) * output_features +
            column) = packed;
      }
      if (global_row1 < rows) {
        const std::uint32_t packed =
            static_cast<std::uint32_t>(
                encode_bf16_rne(value.x2 * compensated_scale)) |
            (static_cast<std::uint32_t>(
                 encode_bf16_rne(value.x3 * compensated_scale))
             << 16U);
        *reinterpret_cast<std::uint32_t*>(
            output + static_cast<std::size_t>(global_row1) * output_features +
            column) = packed;
      }
    }
  }
}

__device__ __forceinline__ void projection_task(
    const unsigned int linear_task, const unsigned int grid_m,
    const unsigned int grid_n, const unsigned int raster_group_m,
    unsigned int& m_tile, unsigned int& n_tile) noexcept {
  const unsigned int group_span = raster_group_m * grid_n;
  const unsigned int group = linear_task / group_span;
  const unsigned int first_m = group * raster_group_m;
  const unsigned int remaining_m = grid_m - first_m;
  const unsigned int active_m =
      remaining_m < raster_group_m ? remaining_m : raster_group_m;
  const unsigned int group_offset = linear_task % group_span;
  n_tile = group_offset / active_m;
  m_tile = first_m + group_offset % active_m;
}

template <Sm87TargetAotProjectionRole kRole,
          unsigned int kInputFeatures, unsigned int kOutputFeatures,
          unsigned int kKTiles, unsigned int kRasterGroupM,
          unsigned int kPartitionCount>
__global__ __launch_bounds__(kThreads, 1)
void sm87_macrofeed_v3_fp8_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const unsigned int rows,
    const float scale0, const float scale1, const float scale2,
    std::uint16_t* __restrict__ output,
    std::uint16_t* __restrict__ secondary_output,
    std::uint16_t* __restrict__ tertiary_output,
    const bool scatter_partitions) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage = reinterpret_cast<Fp8PipelineStorage*>(dynamic_storage);
  const unsigned int grid_m = (rows + kTileM - 1U) / kTileM;
  constexpr unsigned int kGridN = kOutputFeatures / kTileN;
  const unsigned int logical_tasks = grid_m * kGridN;
  for (unsigned int linear_task = blockIdx.x; linear_task < logical_tasks;
       linear_task += kPersistentCtas) {
    unsigned int m_tile = 0U;
    unsigned int n_tile = 0U;
    projection_task(linear_task, grid_m, kGridN, kRasterGroupM, m_tile,
                    n_tile);
    unsigned int partition = 0U;
    unsigned int partition_first_tile = 0U;
    unsigned int partition_n_tiles = kGridN;
    unsigned int partition_offset = 0U;
    float compensated_scale = scale0;
    if constexpr (kRole == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
      if (n_tile >= 40U) {
        partition = 1U;
        partition_first_tile = 40U;
        partition_n_tiles = 24U;
        partition_offset = 52'428'800U;
        compensated_scale = scale1;
      } else {
        partition_n_tiles = 40U;
      }
    } else if constexpr (kRole ==
                         Sm87TargetAotProjectionRole::kFp8FullQkv) {
      if (n_tile >= 52U) {
        partition = 2U;
        partition_first_tile = 52U;
        partition_n_tiles = 4U;
        partition_offset = 68'157'440U;
        compensated_scale = scale2;
      } else if (n_tile >= 48U) {
        partition = 1U;
        partition_first_tile = 48U;
        partition_n_tiles = 4U;
        partition_offset = 62'914'560U;
        compensated_scale = scale1;
      } else {
        partition_n_tiles = 48U;
      }
    }
    const unsigned int partition_n_tile =
        n_tile - partition_first_tile;
    if (partition >= kPartitionCount ||
        partition_n_tile >= partition_n_tiles) {
      continue;
    }
    WarpAccumulator accumulators;
    run_full_k<kInputFeatures, kKTiles>(
        storage, input, payload, rows, m_tile * kTileM,
        partition_offset,
        partition_n_tile, accumulators);
    std::uint16_t* destination = output;
    unsigned int destination_features = kOutputFeatures;
    unsigned int destination_first_n = n_tile * kTileN;
    if (scatter_partitions) {
      destination = partition == 0U
                        ? output
                        : (partition == 1U ? secondary_output
                                           : tertiary_output);
      if constexpr (kRole ==
                    Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
        destination_features = partition == 0U ? 10'240U : 6'144U;
      } else if constexpr (kRole ==
                           Sm87TargetAotProjectionRole::kFp8FullQkv) {
        destination_features = partition == 0U ? 12'288U : 1'024U;
      } else {
        destination_features = 5'120U;
      }
      destination_first_n = partition_n_tile * kTileN;
    }
    publish_output(accumulators, rows, destination_features,
                   m_tile * kTileM, destination_first_n,
                   compensated_scale, destination);
    __syncthreads();
  }
}


__global__ __launch_bounds__(kThreads, 1)
void sm87_macrofeed_v3_fp8_tile_test_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const unsigned int valid_rows,
    const float compensated_scale, std::uint16_t* __restrict__ output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage = reinterpret_cast<Fp8PipelineStorage*>(dynamic_storage);
  WarpAccumulator accumulators;
  run_full_k<static_cast<unsigned int>(
                 kSm87MacroFeedV3Fp8TestInputFeatures),
             static_cast<unsigned int>(kSm87MacroFeedV3Fp8TestKTiles)>(
      storage, input, payload, valid_rows, 0U, 0U, 0U, accumulators);
  publish_output(accumulators, valid_rows,
                 static_cast<unsigned int>(kSm87MacroFeedV3Fp8BlockN), 0U,
                 0U, compensated_scale, output);
}

__global__ void sm87_macrofeed_v3_fp8_code_test_kernel(
    const std::uint8_t* __restrict__ codes,
    std::uint16_t* __restrict__ decoded) {
  const unsigned int index = threadIdx.x;
  const std::uint8_t code = codes[index];
  decoded[index] = static_cast<std::uint16_t>(
      (static_cast<std::uint16_t>(code & 0x80U) << 8U) |
      (static_cast<std::uint16_t>(code & 0x7fU) << 4U));
}

[[nodiscard]] cudaError_t set_dynamic_shared_attribute(
    const Sm87TargetAotProjectionRole role) noexcept {
  cudaError_t status = cudaErrorInvalidValue;
  if (role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    status = cudaFuncSetAttribute(
        sm87_macrofeed_v3_fp8_kernel<
            Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
            5'120U, 16'384U, 80U, 2U, 2U>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSm87MacroFeedV3Fp8DynamicSharedBytes));
  } else if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    status = cudaFuncSetAttribute(
        sm87_macrofeed_v3_fp8_kernel<
            Sm87TargetAotProjectionRole::kFp8FullQkv,
            5'120U, 14'336U, 80U, 2U, 3U>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSm87MacroFeedV3Fp8DynamicSharedBytes));
  } else if (role ==
             Sm87TargetAotProjectionRole::kFp8AttentionOutput) {
    status = cudaFuncSetAttribute(
        sm87_macrofeed_v3_fp8_kernel<
            Sm87TargetAotProjectionRole::kFp8AttentionOutput,
            6'144U, 5'120U, 96U, 1U, 1U>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSm87MacroFeedV3Fp8DynamicSharedBytes));
  }
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      sm87_macrofeed_v3_fp8_tile_test_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87MacroFeedV3Fp8DynamicSharedBytes));
}

[[nodiscard]] cudaError_t validate_fixed_device(
    int* const device_ordinal, cudaDeviceProp* const properties) noexcept {
  if (device_ordinal == nullptr || properties == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaGetDevice(device_ordinal);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaGetDeviceProperties(properties, *device_ordinal);
  if (status != cudaSuccess) {
    return status;
  }
  return properties->major == 8 && properties->minor == 7 &&
                 properties->multiProcessorCount ==
                     static_cast<int>(kSm87MacroFeedV3Fp8SmCount) &&
                 properties->sharedMemPerBlockOptin >=
                     kSm87MacroFeedV3Fp8DynamicSharedBytes
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] bool exact_device_pointer(const void* const pointer,
                                        const int device_ordinal) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  return status == cudaSuccess && attributes.type == cudaMemoryTypeDevice &&
         attributes.device == device_ordinal;
}

template <typename Kernel>
[[nodiscard]] cudaError_t query_resources_body(
    Kernel kernel, const Sm87TargetAotProjectionRole role,
    Sm87MacroFeedV3Fp8CudaResources* const resources) noexcept {
  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status = validate_fixed_device(&device, &properties);
  if (status != cudaSuccess) {
    return status;
  }
  status = set_dynamic_shared_attribute(role);
  if (status != cudaSuccess) {
    return status;
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, kThreads,
      kSm87MacroFeedV3Fp8DynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  resources->identity = sm87_macrofeed_v3_fp8_identity(role);
  resources->role = role;
  resources->device_ordinal = device;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  resources->sm_count = properties.multiProcessorCount;
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = kSm87MacroFeedV3Fp8DynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->optin_shared_bytes_per_block =
      properties.sharedMemPerBlockOptin;
  resources->kernel_compiled = true;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  resources->static_resource_gate_passed =
      sm87_macrofeed_v3_fp8_resource_gate(*resources);
  return cudaSuccess;
}

[[nodiscard]] cudaError_t query_role_resources(
    const Sm87TargetAotProjectionRole role,
    Sm87MacroFeedV3Fp8CudaResources* const resources) noexcept {
  if (role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    return query_resources_body(
        sm87_macrofeed_v3_fp8_kernel<
            Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
            5'120U, 16'384U, 80U, 2U, 2U>,
        role, resources);
  }
  if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    return query_resources_body(
        sm87_macrofeed_v3_fp8_kernel<
            Sm87TargetAotProjectionRole::kFp8FullQkv,
            5'120U, 14'336U, 80U, 2U, 3U>,
        role, resources);
  }
  if (role == Sm87TargetAotProjectionRole::kFp8AttentionOutput) {
    return query_resources_body(
        sm87_macrofeed_v3_fp8_kernel<
            Sm87TargetAotProjectionRole::kFp8AttentionOutput,
            6'144U, 5'120U, 96U, 1U, 1U>,
        role, resources);
  }
  return cudaErrorNotSupported;
}

struct PointerRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] PointerRange pointer_range(const void* const pointer,
                                         const std::uint64_t bytes) noexcept {
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

[[nodiscard]] bool disjoint(const PointerRange& left,
                            const PointerRange& right) noexcept {
  return left.valid && right.valid &&
         (left.end <= right.begin || right.end <= left.begin);
}

[[nodiscard]] float decode_compensated_scale(
    const std::uint16_t bits) noexcept {
  const std::uint32_t raw = static_cast<std::uint32_t>(bits) << 16U;
  float scale = 0.0F;
  std::memcpy(&scale, &raw, sizeof(scale));
  return scale;
}

[[nodiscard]] int enqueue_sealed(
    const Sm87MacroFeedV3Fp8Arguments& arguments,
    const Sm87MacroFeedV3Fp8StartupSeal& startup_seal,
    Sm87MacroFeedV3Fp8LaunchReceipt* const receipt) noexcept {
  const auto plan =
      sm87_macrofeed_v3_fp8_plan(arguments.role, arguments.token_count);
  std::array<float, 3U> scales{};
  for (std::size_t index = 0U; index < plan.partition_count; ++index) {
    scales[index] = decode_compensated_scale(
        arguments.asset.compensated_tensor_scale_bf16_bits[index]);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  const auto* const payload = reinterpret_cast<const std::uint8_t*>(
      arguments.asset.payload.begin);
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    sm87_macrofeed_v3_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
        5'120U, 16'384U, 80U, 2U, 2U>
        <<<kPersistentCtas, kThreads,
           kSm87MacroFeedV3Fp8DynamicSharedBytes, stream>>>(
            arguments.input, payload,
            static_cast<unsigned int>(arguments.token_count), scales[0U],
            scales[1U], scales[2U], arguments.partition_outputs[0U],
            arguments.partition_outputs[1U], nullptr, true);
  } else if (arguments.role ==
             Sm87TargetAotProjectionRole::kFp8FullQkv) {
    sm87_macrofeed_v3_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8FullQkv,
        5'120U, 14'336U, 80U, 2U, 3U>
        <<<kPersistentCtas, kThreads,
           kSm87MacroFeedV3Fp8DynamicSharedBytes, stream>>>(
            arguments.input, payload,
            static_cast<unsigned int>(arguments.token_count), scales[0U],
            scales[1U], scales[2U], arguments.partition_outputs[0U],
            arguments.partition_outputs[1U],
            arguments.partition_outputs[2U], true);
  } else {
    sm87_macrofeed_v3_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8AttentionOutput,
        6'144U, 5'120U, 96U, 1U, 1U>
        <<<kPersistentCtas, kThreads,
           kSm87MacroFeedV3Fp8DynamicSharedBytes, stream>>>(
            arguments.input, payload,
            static_cast<unsigned int>(arguments.token_count), scales[0U],
            scales[1U], scales[2U], arguments.partition_outputs[0U], nullptr,
            nullptr, true);
  }
  const cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *receipt = {plan.identity,
              arguments.asset.artifact_identity,
              arguments.token_count,
              plan.logical_tasks,
              kSm87MacroFeedV3Fp8TailRows,
              1U,
              0U,
              true,
              true,
              false,
              true,
              false};
  (void)startup_seal;
  return static_cast<int>(cudaSuccess);
}

}  // namespace

bool sm87_macrofeed_v3_fp8_arguments_valid(
    const Sm87MacroFeedV3Fp8Arguments& arguments) noexcept {
  const auto plan =
      sm87_macrofeed_v3_fp8_plan(arguments.role, arguments.token_count);
  if (!plan.valid() || arguments.input == nullptr ||
      arguments.asset.payload.role != arguments.role ||
      !sm87_target_aot_fp8_cuda_asset_valid(arguments.asset) ||
      reinterpret_cast<std::uintptr_t>(arguments.input) % alignof(uint4) !=
          0U) {
    return false;
  }
  const std::uint64_t input_bytes =
      static_cast<std::uint64_t>(plan.token_count) * plan.input_features *
      sizeof(std::uint16_t);
  std::array<PointerRange, 5U> ranges{};
  ranges[0U] = pointer_range(arguments.input, input_bytes);
  ranges[1U] = PointerRange{arguments.asset.payload.begin,
                            arguments.asset.payload.end,
                            arguments.asset.payload.valid};
  for (std::size_t index = 0U; index < 3U; ++index) {
    if (index < plan.partition_count) {
      if (arguments.partition_outputs[index] == nullptr ||
          reinterpret_cast<std::uintptr_t>(
              arguments.partition_outputs[index]) %
                  alignof(std::uint32_t) !=
              0U) {
        return false;
      }
      const std::uint64_t output_bytes =
          static_cast<std::uint64_t>(plan.token_count) *
          plan.partition_features[index] * sizeof(std::uint16_t);
      ranges[2U + index] =
          pointer_range(arguments.partition_outputs[index], output_bytes);
    } else if (arguments.partition_outputs[index] != nullptr) {
      return false;
    }
  }
  for (std::size_t first = 0U; first < 2U + plan.partition_count; ++first) {
    if (!ranges[first].valid) {
      return false;
    }
    for (std::size_t second = first + 1U;
         second < 2U + plan.partition_count; ++second) {
      if (!disjoint(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

int query_sm87_macrofeed_v3_fp8_cuda_resources(
    const Sm87TargetAotProjectionRole role,
    Sm87MacroFeedV3Fp8CudaResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  if (!sm87_macrofeed_v3_fp8_plan(role, kSm87MacroFeedV3Fp8Tokens).valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(query_role_resources(role, resources));
}

int seal_sm87_macrofeed_v3_fp8_startup_cuda(
    const Sm87TargetAotProjectionRole role,
    Sm87MacroFeedV3Fp8StartupSeal* const seal) noexcept {
  if (seal == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *seal = {};
  Sm87MacroFeedV3Fp8CudaResources resources{};
  const cudaError_t status = query_role_resources(role, &resources);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (!resources.static_resource_gate_passed ||
      !sm87_macrofeed_v3_fp8_resource_gate(resources)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  seal->resources = resources;
  seal->dynamic_shared_attribute_set = true;
  seal->tactic_frozen_before_requests = true;
  seal->no_hot_device_queries = true;
  seal->no_hot_function_queries = true;
  seal->no_hot_occupancy_queries = true;
  seal->no_hot_pointer_queries = true;
  seal->no_hot_error_state_clear = true;
  seal->t0_t1_only = true;
  seal->production_dispatch_eligible = false;
  seal->seal_identity =
      sm87_macrofeed_v3_fp8_compute_startup_seal_identity(*seal);
  return sm87_macrofeed_v3_fp8_startup_seal_valid(*seal)
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorNotSupported);
}

int launch_sm87_macrofeed_v3_fp8_cuda(
    const Sm87MacroFeedV3Fp8Arguments& arguments,
    Sm87MacroFeedV3Fp8LaunchReceipt* const receipt) noexcept {
  if (receipt == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *receipt = {};
  if (!sm87_macrofeed_v3_fp8_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  Sm87MacroFeedV3Fp8StartupSeal seal{};
  const int seal_status =
      seal_sm87_macrofeed_v3_fp8_startup_cuda(arguments.role, &seal);
  if (seal_status != static_cast<int>(cudaSuccess)) {
    return seal_status;
  }
  const int device = seal.resources.device_ordinal;
  if (arguments.asset.device_upload_receipt.device_ordinal != device ||
      !exact_device_pointer(arguments.input, device) ||
      !exact_device_pointer(
          reinterpret_cast<const void*>(arguments.asset.payload.begin),
          device)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  const auto plan = sm87_macrofeed_v3_fp8_plan(
      arguments.role, arguments.token_count);
  for (std::size_t index = 0U; index < plan.partition_count; ++index) {
    if (!exact_device_pointer(arguments.partition_outputs[index], device)) {
      return static_cast<int>(cudaErrorInvalidDevicePointer);
    }
  }
  return launch_sm87_macrofeed_v3_fp8_sealed_cuda(arguments, seal, receipt);
}

int launch_sm87_macrofeed_v3_fp8_sealed_cuda(
    const Sm87MacroFeedV3Fp8Arguments& arguments,
    const Sm87MacroFeedV3Fp8StartupSeal& startup_seal,
    Sm87MacroFeedV3Fp8LaunchReceipt* const receipt) noexcept {
  if (receipt == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *receipt = {};
  if (!sm87_macrofeed_v3_fp8_arguments_valid(arguments) ||
      !sm87_macrofeed_v3_fp8_startup_seal_valid(startup_seal) ||
      startup_seal.resources.role != arguments.role ||
      startup_seal.resources.device_ordinal !=
          arguments.asset.device_upload_receipt.device_ordinal) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return enqueue_sealed(arguments, startup_seal, receipt);
}

int launch_sm87_macrofeed_v3_fp8_tile_test_cuda(
    const Sm87TargetAotProjectionRole role,
    const std::size_t partition_index,
    const std::uint16_t* const input_m128_k256,
    const std::uint8_t* const canonical_payload_four_cells,
    const std::uint16_t compensated_scale_bf16_bits,
    const std::size_t valid_rows, std::uint16_t* const output_m128_n256,
    void* const cuda_stream) noexcept {
  const auto plan = sm87_macrofeed_v3_fp8_plan(
      role, kSm87MacroFeedV3Fp8Tokens);
  if (!plan.valid() || partition_index >= plan.partition_count ||
      input_m128_k256 == nullptr ||
      canonical_payload_four_cells == nullptr ||
      output_m128_n256 == nullptr ||
      (valid_rows != 64U && valid_rows != 128U) ||
      compensated_scale_bf16_bits == 0U ||
      reinterpret_cast<std::uintptr_t>(input_m128_k256) % alignof(uint4) !=
          0U ||
      reinterpret_cast<std::uintptr_t>(canonical_payload_four_cells) %
              alignof(uint4) !=
          0U ||
      reinterpret_cast<std::uintptr_t>(output_m128_n256) %
              alignof(std::uint32_t) !=
          0U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status = validate_fixed_device(&device, &properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = set_dynamic_shared_attribute(role);
  if (status != cudaSuccess ||
      !exact_device_pointer(input_m128_k256, device) ||
      !exact_device_pointer(canonical_payload_four_cells, device) ||
      !exact_device_pointer(output_m128_n256, device)) {
    return static_cast<int>(status != cudaSuccess ? status
                                                  : cudaErrorInvalidDevicePointer);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_macrofeed_v3_fp8_tile_test_kernel
      <<<1U, kThreads, kSm87MacroFeedV3Fp8DynamicSharedBytes, stream>>>(
          input_m128_k256, canonical_payload_four_cells,
          static_cast<unsigned int>(valid_rows),
          decode_compensated_scale(compensated_scale_bf16_bits),
          output_m128_n256);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_macrofeed_v3_fp8_code_test_cuda(
    const std::uint8_t* const codes_256,
    std::uint16_t* const bias_shift_bits_256,
    void* const cuda_stream) noexcept {
  if (codes_256 == nullptr || bias_shift_bits_256 == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status = validate_fixed_device(&device, &properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (!exact_device_pointer(codes_256, device) ||
      !exact_device_pointer(bias_shift_bits_256, device)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_macrofeed_v3_fp8_code_test_kernel<<<1U, 256U, 0U, stream>>>(
      codes_256, bias_shift_bits_256);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
