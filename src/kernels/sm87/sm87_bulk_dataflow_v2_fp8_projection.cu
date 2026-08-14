#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_projection.h"

#include "sm87_bulk_dataflow_v2_fp8_projection_launch_internal.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace q3x::kernels {
namespace {

constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87BulkV2Fp8Threads);
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kTileM =
    static_cast<unsigned int>(kSm87BulkV2Fp8TileM);
constexpr unsigned int kTileN =
    static_cast<unsigned int>(kSm87BulkV2Fp8TileN);
constexpr unsigned int kTileK =
    static_cast<unsigned int>(kSm87BulkV2Fp8TileK);
constexpr unsigned int kStages =
    static_cast<unsigned int>(kSm87BulkV2Fp8PipelineStages);
constexpr unsigned int kPersistentCtas =
    static_cast<unsigned int>(kSm87BulkV2Fp8PersistentCtas);
constexpr unsigned int kM16Panels = kTileM / 16U;
constexpr unsigned int kN8PanelsPerWarp =
    static_cast<unsigned int>(kSm87BulkV2Fp8WarpN / 8U);
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

static_assert(kThreads == 256U && kTileM == 64U && kTileN == 256U &&
              kTileK == 64U);
static_assert(kStages == 4U && kPersistentCtas == 16U);
static_assert(kActivationVectors == 512U && kWeightVectors == 1'024U);
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

static_assert(sizeof(Fp8PipelineStorage) ==
              kSm87BulkV2Fp8DynamicSharedBytes);

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

using WarpAccumulator =
    M16N8Accumulator[kM16Panels][kN8PanelsPerWarp];
using DecodedWeightStage = K16N8Weight[kN8PanelsPerWarp];

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

// A uses a row-dependent XOR of its eight 16-byte K chunks.  This is a pure
// shared-memory address permutation: global input bytes and arithmetic order
// are unchanged.
[[nodiscard]] __device__ __forceinline__ unsigned int
activation_swizzled_vector(const unsigned int row,
                            const unsigned int vector) noexcept {
  return row * (kTileK / 8U) + (vector ^ (row & 7U));
}

__device__ __forceinline__ void load_activation_fragment(
    M16K16Activation& fragment,
    const std::uint16_t* const shared_activations,
    const unsigned int m16, const unsigned int k16,
    const unsigned int lane) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row =
      m16 * 16U + lane % 8U + (quadrant & 1U) * 8U;
  const unsigned int column = k16 * 16U + (quadrant >> 1U) * 8U;
  const unsigned int vector = column / 8U;
  const auto* const source =
      shared_activations + activation_swizzled_vector(row, vector) * 8U;
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
decode_weight_fragment(const std::uint8_t* const shared_weight,
                       const unsigned int k16,
                       const unsigned int global_n8_panel,
                       const unsigned int lane) noexcept {
  // The authenticated payload is [K16][N64][N8][lane][component].
  // Flattening N64/N8 gives 32 N8 panels per N256 cell.
  const unsigned int fragment = k16 * 32U + global_n8_panel;
  const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
      shared_weight + fragment * 128U + lane * 4U);
  const std::uint8_t code0 = static_cast<std::uint8_t>(packed);
  const std::uint8_t code1 = static_cast<std::uint8_t>(packed >> 16U);
  const std::uint8_t code2 = static_cast<std::uint8_t>(packed >> 8U);
  const std::uint8_t code3 = static_cast<std::uint8_t>(packed >> 24U);
  const std::uint16_t component0 =
      sm87_bulk_v2_fp8_raw_code_to_biased_bf16_bits(code0);
  const std::uint16_t component1 =
      sm87_bulk_v2_fp8_raw_code_to_biased_bf16_bits(code1);
  const std::uint16_t component2 =
      sm87_bulk_v2_fp8_raw_code_to_biased_bf16_bits(code2);
  const std::uint16_t component3 =
      sm87_bulk_v2_fp8_raw_code_to_biased_bf16_bits(code3);
  // Persisted [K0,K8,K1,K9] -> MMA register halves [K0,K1]/[K8,K9].
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
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      accumulators[m16][n8] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }
}

template <unsigned int kInputFeatures, unsigned int kKTiles>
__device__ __forceinline__ void issue_pipeline_stage(
    Fp8PipelineStorage* const storage, const unsigned int slot,
    const std::uint16_t* const input, const std::uint8_t* const payload,
    const unsigned int first_m, const std::uint64_t partition_offset,
    const unsigned int partition_n_tile,
    const unsigned int k_tile) noexcept {
#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const unsigned int row = vector_index / (kTileK / 8U);
    const unsigned int vector = vector_index % (kTileK / 8U);
    const auto* const source = reinterpret_cast<const uint4*>(
                                   input +
                                   static_cast<std::size_t>(first_m + row) *
                                       kInputFeatures +
                                   k_tile * kTileK) +
                               vector;
    cp_async_cg_16<false>(
        &storage->activations[slot]
                             [activation_swizzled_vector(row, vector)],
        source);
  }
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const std::uint64_t cell =
        partition_offset +
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
    const std::uint8_t* const payload, const unsigned int first_m,
    const std::uint64_t partition_offset,
    const unsigned int partition_n_tile,
    WarpAccumulator& accumulators) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  clear_accumulators(accumulators);

#pragma unroll
  for (unsigned int stage = 0U; stage < kStages; ++stage) {
    issue_pipeline_stage<kInputFeatures, kKTiles>(
        storage, stage, input, payload, first_m, partition_offset,
        partition_n_tile, stage);
  }

#pragma unroll 1
  for (unsigned int k_tile = 0U; k_tile < kKTiles; ++k_tile) {
    if (k_tile + 3U < kKTiles) {
      cp_async_wait_group<3U>();
    } else if (k_tile + 2U < kKTiles) {
      cp_async_wait_group<2U>();
    } else if (k_tile + 1U < kKTiles) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();
    const unsigned int slot = k_tile & (kStages - 1U);
    const auto* const shared_activations =
        reinterpret_cast<const std::uint16_t*>(storage->activations[slot]);
    const auto* const shared_weight =
        reinterpret_cast<const std::uint8_t*>(storage->weights[slot]);

    DecodedWeightStage decoded[2U];
    const unsigned int first_n8_panel = warp * kN8PanelsPerWarp;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      decoded[0U][n8] = decode_weight_fragment(
          shared_weight, 0U, first_n8_panel + n8, lane);
    }

#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16Panels; ++k16) {
      const unsigned int current = k16 & 1U;
      const unsigned int next = current ^ 1U;
      if (k16 + 1U < kK16Panels) {
#pragma unroll
        for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
          decoded[next][n8] = decode_weight_fragment(
              shared_weight, k16 + 1U, first_n8_panel + n8, lane);
        }
      }
      M16K16Activation activation[kM16Panels];
#pragma unroll
      for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
        load_activation_fragment(activation[m16], shared_activations, m16,
                                 k16, lane);
      }
      // The last shared reads have reached registers.  All warps rendezvous
      // before the same slot is recycled for K+4; the last MMA quartet then
      // overlaps that global-to-shared group.
      if (k16 + 1U == kK16Panels) {
        __syncthreads();
        if (k_tile + kStages < kKTiles) {
          issue_pipeline_stage<kInputFeatures, kKTiles>(
              storage, slot, input, payload, first_m, partition_offset,
              partition_n_tile, k_tile + kStages);
        }
      }
#pragma unroll
      for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
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
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
    const unsigned int local_row0 = m16 * 16U + lane_group;
    const unsigned int local_row1 = local_row0 + 8U;
    const unsigned int global_row0 = first_m + local_row0;
    const unsigned int global_row1 = first_m + local_row1;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      const unsigned int column =
          first_n + warp * 32U + n8 * 8U + lane_in_group * 2U;
      const auto& value = accumulators[m16][n8];
      if (global_row0 < rows) {
        const std::uint32_t packed =
            static_cast<std::uint32_t>(encode_bf16_rne(
                __fmul_rn(value.x0, compensated_scale))) |
            (static_cast<std::uint32_t>(encode_bf16_rne(
                 __fmul_rn(value.x1, compensated_scale)))
             << 16U);
        *reinterpret_cast<std::uint32_t*>(
            output + static_cast<std::size_t>(global_row0) * output_features +
            column) = packed;
      }
      if (global_row1 < rows) {
        const std::uint32_t packed =
            static_cast<std::uint32_t>(encode_bf16_rne(
                __fmul_rn(value.x2, compensated_scale))) |
            (static_cast<std::uint32_t>(encode_bf16_rne(
                 __fmul_rn(value.x3, compensated_scale)))
             << 16U);
        *reinterpret_cast<std::uint32_t*>(
            output + static_cast<std::size_t>(global_row1) * output_features +
            column) = packed;
      }
    }
  }
}

template <Sm87TargetAotProjectionRole kRole,
          unsigned int kInputFeatures, unsigned int kOutputFeatures,
          unsigned int kKTiles, unsigned int kPartitionCount>
__global__ __launch_bounds__(kThreads, 1)
void sm87_bulk_v2_fp8_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const unsigned int rows,
    const float scale0, const float scale1, const float scale2,
    std::uint16_t* __restrict__ primary_output,
    std::uint16_t* __restrict__ secondary_output,
    std::uint16_t* __restrict__ tertiary_output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage = reinterpret_cast<Fp8PipelineStorage*>(dynamic_storage);
  const unsigned int grid_m = rows / kTileM;
  constexpr unsigned int kGridN = kOutputFeatures / kTileN;
  const unsigned int logical_tasks = grid_m * kGridN;

  // For M1024, blockIdx is the M64 cohort member and +=16 advances N.  All
  // sixteen CTAs therefore request the same B tile together while keeping
  // private A rows.  This is a temporal L2-cohort hint, not a persistence or
  // residency guarantee.  For M64, the same loop assigns one N tile per CTA.
  for (unsigned int linear_task = blockIdx.x; linear_task < logical_tasks;
       linear_task += kPersistentCtas) {
    const unsigned int m_tile = linear_task % grid_m;
    const unsigned int n_tile = linear_task / grid_m;
    unsigned int partition = 0U;
    unsigned int partition_first_tile = 0U;
    unsigned int partition_n_tiles = kGridN;
    std::uint64_t partition_offset = 0U;
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
        storage, input, payload, m_tile * kTileM, partition_offset,
        partition_n_tile, accumulators);

    std::uint16_t* destination = primary_output;
    unsigned int destination_features = kOutputFeatures;
    unsigned int destination_first_n = n_tile * kTileN;
    if constexpr (kRole == Sm87TargetAotProjectionRole::kFp8FullQkv) {
      destination = partition == 0U
                        ? primary_output
                        : (partition == 1U ? secondary_output
                                           : tertiary_output);
      destination_features = partition == 0U ? 12'288U : 1'024U;
      destination_first_n = partition_n_tile * kTileN;
    }
    publish_output(accumulators, rows, destination_features,
                   m_tile * kTileM, destination_first_n,
                   compensated_scale, destination);
    __syncthreads();
  }
}

template <typename Kernel>
[[nodiscard]] cudaError_t set_dynamic_shared_for_kernel(
    Kernel kernel) noexcept {
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87BulkV2Fp8DynamicSharedBytes));
}

[[nodiscard]] cudaError_t set_dynamic_shared(
    const Sm87TargetAotProjectionRole role) noexcept {
  if (role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    return set_dynamic_shared_for_kernel(sm87_bulk_v2_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
        5'120U, 16'384U, 80U, 2U>);
  }
  if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    return set_dynamic_shared_for_kernel(sm87_bulk_v2_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8FullQkv,
        5'120U, 14'336U, 80U, 3U>);
  }
  if (role == Sm87TargetAotProjectionRole::kFp8AttentionOutput) {
    return set_dynamic_shared_for_kernel(sm87_bulk_v2_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8AttentionOutput,
        6'144U, 5'120U, 96U, 1U>);
  }
  return cudaErrorInvalidValue;
}

[[nodiscard]] cudaError_t validate_fixed_device() noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return status;
  }
  return properties.major == 8 && properties.minor == 7 &&
                 properties.multiProcessorCount == 16 &&
                 properties.sharedMemPerBlockOptin >=
                     kSm87BulkV2Fp8DynamicSharedBytes
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
[[nodiscard]] cudaError_t query_kernel_resources(
    Kernel kernel, const Sm87TargetAotProjectionRole role,
    Sm87BulkV2Fp8KernelResources* const resources) noexcept {
  cudaError_t status = set_dynamic_shared_for_kernel(kernel);
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
      &active_blocks, kernel, kThreads, kSm87BulkV2Fp8DynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  resources->role = role;
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = kSm87BulkV2Fp8DynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->kernel_compiled = true;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  resources->resource_gate_passed =
      resources->binary_version == 87 &&
      resources->registers_per_thread > 0 &&
      resources->registers_per_thread <=
          kSm87BulkV2Fp8HardMaximumRegisters &&
      resources->local_bytes == 0U &&
      resources->maximum_threads_per_block >= static_cast<int>(kThreads) &&
      resources->active_blocks_per_sm ==
          kSm87BulkV2Fp8RequiredActiveCtasPerSm;
  return cudaSuccess;
}

[[nodiscard]] cudaError_t query_family_resources_impl(
    Sm87BulkV2Fp8FamilyResources* const resources) noexcept {
  cudaError_t status = query_kernel_resources(
      sm87_bulk_v2_fp8_kernel<
          Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
          5'120U, 16'384U, 80U, 2U>,
      Sm87TargetAotProjectionRole::kFp8GdnQkvZ, &resources->roles[0U]);
  if (status != cudaSuccess) {
    return status;
  }
  status = query_kernel_resources(
      sm87_bulk_v2_fp8_kernel<
          Sm87TargetAotProjectionRole::kFp8FullQkv,
          5'120U, 14'336U, 80U, 3U>,
      Sm87TargetAotProjectionRole::kFp8FullQkv, &resources->roles[1U]);
  if (status != cudaSuccess) {
    return status;
  }
  status = query_kernel_resources(
      sm87_bulk_v2_fp8_kernel<
          Sm87TargetAotProjectionRole::kFp8AttentionOutput,
          6'144U, 5'120U, 96U, 1U>,
      Sm87TargetAotProjectionRole::kFp8AttentionOutput,
      &resources->roles[2U]);
  if (status != cudaSuccess) {
    return status;
  }
  resources->all_compiled = true;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  resources->resource_gate_passed =
      sm87_bulk_v2_fp8_kernel_resources_valid(resources->roles[0U]) &&
      sm87_bulk_v2_fp8_kernel_resources_valid(resources->roles[1U]) &&
      sm87_bulk_v2_fp8_kernel_resources_valid(resources->roles[2U]);
  return cudaSuccess;
}

[[nodiscard]] bool p40_role_arguments_valid(
    const Sm87BulkV2Fp8RoleArguments& arguments) noexcept {
  const auto plan = sm87_bulk_v2_fp8_role_plan(arguments.role);
  if (!plan.valid || arguments.layer >= kSm87BulkV2Fp8LayerCount ||
      arguments.input == nullptr ||
      reinterpret_cast<std::uintptr_t>(arguments.input) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(arguments.primary_output) % 16U !=
          0U ||
      (arguments.secondary_output != nullptr &&
       reinterpret_cast<std::uintptr_t>(arguments.secondary_output) % 16U !=
           0U) ||
      (arguments.tertiary_output != nullptr &&
       reinterpret_cast<std::uintptr_t>(arguments.tertiary_output) % 16U !=
           0U) ||
      !sm87_bulk_v2_fp8_output_shape_valid(
          arguments.role, arguments.primary_output,
          arguments.secondary_output, arguments.tertiary_output) ||
      !sm87_bulk_v2_fp8_asset_valid(arguments.asset, arguments.role)) {
    return false;
  }
  const std::array<const void*, 4U> pointers{{
      arguments.input, arguments.primary_output, arguments.secondary_output,
      arguments.tertiary_output}};
  const std::array<std::uint64_t, 4U> widths{{
      plan.input_features, plan.primary_output_features,
      plan.secondary_output_features, plan.tertiary_output_features}};
  std::array<Sm87TargetAotFp8CudaByteRange, 5U> ranges{};
  std::size_t range_count = 0U;
  for (std::size_t index = 0U; index < pointers.size(); ++index) {
    if (widths[index] == 0U) {
      continue;
    }
    ranges[range_count++] = sm87_target_aot_fp8_cuda_byte_range(
        pointers[index], kSm87BulkV2Fp8P40Tokens * widths[index] *
                             sizeof(std::uint16_t));
  }
  ranges[range_count++] = {
      arguments.asset.authenticated.payload.begin,
      arguments.asset.authenticated.payload.end,
      arguments.asset.authenticated.payload.valid};
  for (std::size_t first = 0U; first < range_count; ++first) {
    if (!ranges[first].valid) {
      return false;
    }
    for (std::size_t second = first + 1U; second < range_count; ++second) {
      if (sm87_target_aot_fp8_cuda_ranges_overlap(ranges[first],
                                                   ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

struct DeviceAllocationRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  std::size_t bytes = 0U;
};

// Startup-only, complete-range validation.  Pointer attributes alone prove
// only the queried byte.  cudaMemGetAddressRange on both endpoints proves the
// whole logical tensor remains inside one live CUDA allocation.
[[nodiscard]] bool exact_device_allocation_range(
    const Sm87TargetAotFp8CudaByteRange& range,
    const int device_ordinal,
    DeviceAllocationRange* const allocation = nullptr) noexcept {
  if (!range.valid || range.begin == 0U || range.end <= range.begin) {
    return false;
  }
  const std::array<const void*, 2U> endpoints{{
      reinterpret_cast<const void*>(range.begin),
      reinterpret_cast<const void*>(range.end - 1U),
  }};
  std::array<CUdeviceptr, 2U> bases{{0U, 0U}};
  std::array<std::size_t, 2U> bytes{{0U, 0U}};
  for (std::size_t index = 0U; index < endpoints.size(); ++index) {
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, endpoints[index]) !=
            cudaSuccess ||
        attributes.type != cudaMemoryTypeDevice ||
        attributes.device != device_ordinal ||
        cuMemGetAddressRange(
            &bases[index], &bytes[index],
            static_cast<CUdeviceptr>(
                reinterpret_cast<std::uintptr_t>(endpoints[index]))) !=
            CUDA_SUCCESS ||
        bases[index] == 0U || bytes[index] == 0U) {
      return false;
    }
  }
  if (bases[0U] != bases[1U] || bytes[0U] != bytes[1U]) {
    return false;
  }
  const auto allocation_begin = static_cast<std::uintptr_t>(bases[0U]);
  if (bytes[0U] > std::numeric_limits<std::uintptr_t>::max() ||
      allocation_begin > std::numeric_limits<std::uintptr_t>::max() -
                             static_cast<std::uintptr_t>(bytes[0U])) {
    return false;
  }
  const auto allocation_end =
      allocation_begin + static_cast<std::uintptr_t>(bytes[0U]);
  if (range.begin < allocation_begin || range.end > allocation_end) {
    return false;
  }
  if (allocation != nullptr) {
    *allocation = {allocation_begin, allocation_end, bytes[0U]};
  }
  return true;
}

[[nodiscard]] bool p40_role_complete_device_ranges_valid(
    const Sm87BulkV2Fp8RoleArguments& arguments,
    const int device_ordinal) noexcept {
  const auto plan = sm87_bulk_v2_fp8_role_plan(arguments.role);
  if (!plan.valid) {
    return false;
  }
  const std::array<const void*, 4U> pointers{{
      arguments.input, arguments.primary_output, arguments.secondary_output,
      arguments.tertiary_output,
  }};
  const std::array<std::uint64_t, 4U> widths{{
      plan.input_features, plan.primary_output_features,
      plan.secondary_output_features, plan.tertiary_output_features,
  }};
  for (std::size_t index = 0U; index < pointers.size(); ++index) {
    if (widths[index] == 0U) {
      continue;
    }
    const auto range = sm87_target_aot_fp8_cuda_byte_range(
        pointers[index], kSm87BulkV2Fp8P40Tokens * widths[index] *
                             sizeof(std::uint16_t));
    if (!exact_device_allocation_range(range, device_ordinal)) {
      return false;
    }
  }

  const auto& upload =
      arguments.asset.authenticated.device_upload_receipt;
  const Sm87TargetAotFp8CudaByteRange payload_range{
      arguments.asset.authenticated.payload.begin,
      arguments.asset.authenticated.payload.end,
      arguments.asset.authenticated.payload.valid,
  };
  DeviceAllocationRange actual_allocation{};
  return exact_device_allocation_range(payload_range, device_ordinal,
                                       &actual_allocation) &&
         actual_allocation.begin == upload.device_allocation_begin &&
         actual_allocation.end == upload.device_allocation_end &&
         actual_allocation.bytes == upload.device_allocation_bytes;
}

[[nodiscard]] bool oracle_arguments_valid(
    const Sm87BulkV2Fp8OracleArguments& arguments) noexcept {
  const auto plan = sm87_bulk_v2_fp8_role_plan(arguments.role);
  if (!plan.valid || arguments.input == nullptr ||
      arguments.payload == nullptr || arguments.cuda_stream == nullptr ||
      !sm87_bulk_v2_fp8_segment_token_count_valid(arguments.token_count) ||
      reinterpret_cast<std::uintptr_t>(arguments.input) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(arguments.payload) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(arguments.primary_output) % 16U !=
          0U ||
      (arguments.secondary_output != nullptr &&
       reinterpret_cast<std::uintptr_t>(arguments.secondary_output) % 16U !=
           0U) ||
      (arguments.tertiary_output != nullptr &&
       reinterpret_cast<std::uintptr_t>(arguments.tertiary_output) % 16U !=
           0U) ||
      !sm87_bulk_v2_fp8_output_shape_valid(
          arguments.role, arguments.primary_output,
          arguments.secondary_output, arguments.tertiary_output)) {
    return false;
  }
  for (std::size_t index = 0U; index < plan.partition_count; ++index) {
    if (arguments.compensated_scale_bf16_bits[index] == 0U) {
      return false;
    }
  }
  const auto layout =
      sm87_target_aot_projection_packed_layout(arguments.role);
  const std::array<const void*, 4U> pointers{{
      arguments.input, arguments.primary_output, arguments.secondary_output,
      arguments.tertiary_output}};
  const std::array<std::uint64_t, 4U> widths{{
      plan.input_features, plan.primary_output_features,
      plan.secondary_output_features, plan.tertiary_output_features}};
  std::array<Sm87TargetAotFp8CudaByteRange, 5U> ranges{};
  std::size_t range_count = 0U;
  for (std::size_t index = 0U; index < pointers.size(); ++index) {
    if (widths[index] == 0U) {
      continue;
    }
    ranges[range_count++] = sm87_target_aot_fp8_cuda_byte_range(
        pointers[index], arguments.token_count * widths[index] *
                             sizeof(std::uint16_t));
  }
  ranges[range_count++] = sm87_target_aot_fp8_cuda_byte_range(
      arguments.payload, layout.payload_bytes);
  for (std::size_t first = 0U; first < range_count; ++first) {
    if (!ranges[first].valid) {
      return false;
    }
    for (std::size_t second = first + 1U; second < range_count; ++second) {
      if (sm87_target_aot_fp8_cuda_ranges_overlap(ranges[first],
                                                   ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] cudaError_t launch_kernel(
    const Sm87BulkV2Fp8OracleArguments& arguments,
    const bool clear_prior_error = true) noexcept {
  float scales[3U]{};
  for (std::size_t index = 0U; index < 3U; ++index) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(
            arguments.compensated_scale_bf16_bits[index])
        << 16U;
    std::memcpy(&scales[index], &bits, sizeof(float));
  }
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  const dim3 grid(kPersistentCtas);
  const dim3 block(kThreads);
  if (clear_prior_error) {
    (void)cudaGetLastError();
  }
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    sm87_bulk_v2_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
        5'120U, 16'384U, 80U, 2U>
        <<<grid, block, kSm87BulkV2Fp8DynamicSharedBytes, stream>>>(
            arguments.input, arguments.payload,
            static_cast<unsigned int>(arguments.token_count), scales[0U],
            scales[1U], scales[2U], arguments.primary_output,
            arguments.secondary_output, arguments.tertiary_output);
  } else if (arguments.role ==
             Sm87TargetAotProjectionRole::kFp8FullQkv) {
    sm87_bulk_v2_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8FullQkv,
        5'120U, 14'336U, 80U, 3U>
        <<<grid, block, kSm87BulkV2Fp8DynamicSharedBytes, stream>>>(
            arguments.input, arguments.payload,
            static_cast<unsigned int>(arguments.token_count), scales[0U],
            scales[1U], scales[2U], arguments.primary_output,
            arguments.secondary_output, arguments.tertiary_output);
  } else {
    sm87_bulk_v2_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8AttentionOutput,
        6'144U, 5'120U, 96U, 1U>
        <<<grid, block, kSm87BulkV2Fp8DynamicSharedBytes, stream>>>(
            arguments.input, arguments.payload,
            static_cast<unsigned int>(arguments.token_count), scales[0U],
            scales[1U], scales[2U], arguments.primary_output,
            arguments.secondary_output, arguments.tertiary_output);
  }
  return cudaPeekAtLastError();
}

[[nodiscard]] int launch_segment_impl(
    const Sm87BulkV2Fp8SegmentArguments& arguments,
    const bool enforce_resource_gate) noexcept {
  if (!sm87_bulk_v2_fp8_segment_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t device_status = validate_fixed_device();
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  if (enforce_resource_gate) {
    Sm87BulkV2Fp8FamilyResources resources{};
    const cudaError_t query_status = query_family_resources_impl(&resources);
    if (query_status != cudaSuccess) {
      return static_cast<int>(query_status);
    }
    if (!sm87_bulk_v2_fp8_family_resources_valid(resources)) {
      return static_cast<int>(cudaErrorNotSupported);
    }
  }
  const auto& upload = arguments.asset.authenticated.device_upload_receipt;
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const std::array<const void*, 5U> pointers{{
      reinterpret_cast<const void*>(arguments.asset.authenticated.payload.begin),
      arguments.input, arguments.primary_output, arguments.secondary_output,
      arguments.tertiary_output}};
  if (current_device != upload.device_ordinal) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  for (const void* const pointer : pointers) {
    if (pointer != nullptr && !exact_device_pointer(pointer, current_device)) {
      return static_cast<int>(cudaErrorInvalidDevicePointer);
    }
  }
  status = set_dynamic_shared(arguments.role);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  Sm87BulkV2Fp8OracleArguments launch{};
  launch.role = arguments.role;
  launch.input = arguments.input;
  launch.payload = reinterpret_cast<const std::uint8_t*>(
      arguments.asset.authenticated.payload.begin);
  launch.compensated_scale_bf16_bits =
      arguments.asset.authenticated.compensated_tensor_scale_bf16_bits;
  launch.token_count = arguments.token_count;
  launch.primary_output = arguments.primary_output;
  launch.secondary_output = arguments.secondary_output;
  launch.tertiary_output = arguments.tertiary_output;
  launch.cuda_stream = arguments.cuda_stream;
  return static_cast<int>(launch_kernel(launch));
}

[[nodiscard]] int launch_role_p40_impl(
    const Sm87BulkV2Fp8RoleArguments& arguments, void* const cuda_stream,
    const bool enforce_resource_gate,
    std::size_t* const enqueued_launches) noexcept {
  if (enqueued_launches == nullptr || cuda_stream == nullptr ||
      !p40_role_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *enqueued_launches = 0U;
  if (enforce_resource_gate) {
    Sm87BulkV2Fp8FamilyResources resources{};
    const int status =
        query_sm87_bulk_dataflow_v2_fp8_family_resources_cuda(&resources);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    if (!sm87_bulk_v2_fp8_family_resources_valid(resources)) {
      return static_cast<int>(cudaErrorNotSupported);
    }
  }
  const cudaError_t device_status = validate_fixed_device();
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const auto& upload = arguments.asset.authenticated.device_upload_receipt;
  const std::array<const void*, 5U> pointers{{
      reinterpret_cast<const void*>(
          arguments.asset.authenticated.payload.begin),
      arguments.input, arguments.primary_output, arguments.secondary_output,
      arguments.tertiary_output}};
  if (current_device != upload.device_ordinal) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  for (const void* const pointer : pointers) {
    if (pointer != nullptr && !exact_device_pointer(pointer, current_device)) {
      return static_cast<int>(cudaErrorInvalidDevicePointer);
    }
  }
  status = set_dynamic_shared(arguments.role);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const auto plan = sm87_bulk_v2_fp8_role_plan(arguments.role);
  std::size_t first_token = 0U;
  for (std::size_t segment = 0U;
       segment < kSm87BulkV2Fp8SegmentsPerRole; ++segment) {
    const std::size_t tokens =
        segment < kSm87BulkV2Fp8MacroSegments
            ? kSm87BulkV2Fp8MacroTokens
            : kSm87BulkV2Fp8TailTokens;
    Sm87BulkV2Fp8OracleArguments current{};
    current.role = arguments.role;
    current.input = arguments.input + first_token * plan.input_features;
    current.payload = reinterpret_cast<const std::uint8_t*>(
        arguments.asset.authenticated.payload.begin);
    current.compensated_scale_bf16_bits =
        arguments.asset.authenticated.compensated_tensor_scale_bf16_bits;
    current.token_count = tokens;
    current.primary_output =
        arguments.primary_output + first_token * plan.primary_output_features;
    if (plan.secondary_output_features != 0U) {
      current.secondary_output = arguments.secondary_output +
                                 first_token * plan.secondary_output_features;
      current.tertiary_output = arguments.tertiary_output +
                                first_token * plan.tertiary_output_features;
    }
    current.cuda_stream = cuda_stream;
    const cudaError_t launch_status = launch_kernel(current);
    if (launch_status != cudaSuccess) {
      return static_cast<int>(launch_status);
    }
    ++*enqueued_launches;
    first_token += tokens;
  }
  return first_token == kSm87BulkV2Fp8P40Tokens
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorInvalidValue);
}

}  // namespace

int query_sm87_bulk_dataflow_v2_fp8_family_resources_cuda(
    Sm87BulkV2Fp8FamilyResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  const cudaError_t device_status = validate_fixed_device();
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  return static_cast<int>(query_family_resources_impl(resources));
}

int launch_sm87_bulk_dataflow_v2_fp8_segment_cuda(
    const Sm87BulkV2Fp8SegmentArguments& arguments) noexcept {
  return launch_segment_impl(arguments, true);
}

int launch_sm87_bulk_dataflow_v2_fp8_role_p40_cuda(
    const Sm87BulkV2Fp8RoleArguments& arguments, void* const cuda_stream,
    std::size_t* const enqueued_launches) noexcept {
  return launch_role_p40_impl(arguments, cuda_stream, true,
                              enqueued_launches);
}

int launch_sm87_bulk_dataflow_v2_fp8_family_p40_cuda(
    const Sm87BulkV2Fp8FamilyArguments& arguments,
    std::size_t* const enqueued_launches) noexcept {
  if (enqueued_launches == nullptr || arguments.roles == nullptr ||
      arguments.role_count != kSm87BulkV2Fp8LogicalRoleCount ||
      arguments.token_count != kSm87BulkV2Fp8P40Tokens ||
      arguments.cuda_stream == nullptr ||
      !sm87_bulk_v2_fp8_family_manifest_valid(
          kSm87BulkV2Fp8FrozenFamilyManifest)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *enqueued_launches = 0U;
  // Close every host-side identity/range check before the first enqueue.  A
  // malformed later role therefore cannot leave a partially admitted family
  // in the stream.
  for (std::size_t ordinal = 0U;
       ordinal < kSm87BulkV2Fp8LogicalRoleCount; ++ordinal) {
    const auto& expected = kSm87BulkV2Fp8FrozenFamilyManifest.roles[ordinal];
    const auto& role = arguments.roles[ordinal];
    if (role.layer != expected.layer || role.role != expected.role ||
        !p40_role_arguments_valid(role)) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  Sm87BulkV2Fp8FamilyResources resources{};
  const int resource_status =
      query_sm87_bulk_dataflow_v2_fp8_family_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }
  if (!sm87_bulk_v2_fp8_family_resources_valid(resources)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  for (std::size_t ordinal = 0U;
       ordinal < kSm87BulkV2Fp8LogicalRoleCount; ++ordinal) {
    const auto& role = arguments.roles[ordinal];
    std::size_t role_launches = 0U;
    const int status = launch_role_p40_impl(
        role, arguments.cuda_stream, false, &role_launches);
    if (status != static_cast<int>(cudaSuccess) ||
        role_launches != kSm87BulkV2Fp8SegmentsPerRole) {
      return status == static_cast<int>(cudaSuccess)
                 ? static_cast<int>(cudaErrorInvalidValue)
                 : status;
    }
    *enqueued_launches += role_launches;
  }
  return *enqueued_launches == kSm87BulkV2Fp8PhysicalLaunches
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorInvalidValue);
}

int launch_sm87_bulk_dataflow_v2_fp8_oracle_segment_cuda(
    const Sm87BulkV2Fp8OracleArguments& arguments) noexcept {
  if (!oracle_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t device_status = validate_fixed_device();
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  Sm87BulkV2Fp8FamilyResources resources{};
  const cudaError_t resource_status = query_family_resources_impl(&resources);
  if (resource_status != cudaSuccess) {
    return static_cast<int>(resource_status);
  }
  if (!sm87_bulk_v2_fp8_family_resources_valid(resources)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const std::array<const void*, 5U> pointers{{
      arguments.input, arguments.payload, arguments.primary_output,
      arguments.secondary_output, arguments.tertiary_output}};
  for (const void* const pointer : pointers) {
    if (pointer != nullptr && !exact_device_pointer(pointer, device)) {
      return static_cast<int>(cudaErrorInvalidDevicePointer);
    }
  }
  status = set_dynamic_shared(arguments.role);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(launch_kernel(arguments));
}

namespace sm87_bulk_v2_fp8_execution_detail {
namespace {

[[nodiscard]] std::uint64_t mix_seal_nonce(
    std::uint64_t nonce, const std::uint64_t value) noexcept {
  nonce ^= value + 0x9e37'79b9'7f4a'7c15ULL + (nonce << 6U) +
           (nonce >> 2U);
  return nonce;
}

[[nodiscard]] Sm87BulkV2Fp8SealResult fp8_seal_failure(
    const cudaError_t error,
    const Sm87BulkV2Fp8SealFailure failure) noexcept {
  Sm87BulkV2Fp8SealResult result;
  result.cuda_error = static_cast<int>(error);
  result.failure = failure;
  return result;
}

}  // namespace

Sm87BulkV2Fp8SealResult seal_sm87_bulk_v2_fp8_p40_cuda(
    const Sm87BulkV2Fp8SealRequest& request) noexcept {
  if (request.role_bindings == nullptr ||
      request.role_binding_count != kSm87BulkV2Fp8LogicalRoleCount ||
      request.device_ordinal < 0 || request.cuda_stream == nullptr ||
      request.deployment_identity == 0U ||
      request.binding_catalog_identity == 0U ||
      request.binding_lifetime_owner_identity == 0U ||
      request.cuda_stream_owner_identity == 0U ||
      request.authenticated_weight_owner_identity == 0U ||
      !sm87_bulk_v2_fp8_family_manifest_valid(
          kSm87BulkV2Fp8FrozenFamilyManifest)) {
    return fp8_seal_failure(cudaErrorInvalidValue,
                            Sm87BulkV2Fp8SealFailure::kInvalidRequest);
  }

  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess || current_device != request.device_ordinal) {
    return fp8_seal_failure(
        status == cudaSuccess ? cudaErrorInvalidDevice : status,
        Sm87BulkV2Fp8SealFailure::kDevice);
  }
  status = validate_fixed_device();
  if (status != cudaSuccess) {
    return fp8_seal_failure(status, Sm87BulkV2Fp8SealFailure::kDevice);
  }

  unsigned int stream_flags = 0U;
  status = cudaStreamGetFlags(
      reinterpret_cast<cudaStream_t>(request.cuda_stream), &stream_flags);
  if (status != cudaSuccess ||
      (stream_flags & cudaStreamNonBlocking) == 0U) {
    return fp8_seal_failure(
        status == cudaSuccess ? cudaErrorInvalidResourceHandle : status,
        Sm87BulkV2Fp8SealFailure::kStream);
  }

  Sm87BulkV2Fp8FamilyResources resources{};
  status = query_family_resources_impl(&resources);
  if (status != cudaSuccess ||
      !sm87_bulk_v2_fp8_family_resources_valid(resources)) {
    return fp8_seal_failure(
        status == cudaSuccess ? cudaErrorNotSupported : status,
        Sm87BulkV2Fp8SealFailure::kResources);
  }

  std::array<Sm87BulkV2Fp8RoleArguments,
             kSm87BulkV2Fp8LogicalRoleCount>
      sealed_bindings{};
  for (std::size_t ordinal = 0U; ordinal < sealed_bindings.size();
       ++ordinal) {
    const auto& expected = kSm87BulkV2Fp8FrozenFamilyManifest.roles[ordinal];
    const auto& binding = request.role_bindings[ordinal];
    const auto& upload =
        binding.asset.authenticated.device_upload_receipt;
    if (binding.layer != expected.layer || binding.role != expected.role ||
        !p40_role_arguments_valid(binding) ||
        upload.device_ordinal != request.device_ordinal ||
        upload.device_allocation_owner_identity !=
            request.authenticated_weight_owner_identity) {
      return fp8_seal_failure(
          cudaErrorInvalidDevicePointer,
          Sm87BulkV2Fp8SealFailure::kBindingIdentity);
    }
    if (!p40_role_complete_device_ranges_valid(binding,
                                                request.device_ordinal)) {
      return fp8_seal_failure(cudaErrorInvalidDevicePointer,
                              Sm87BulkV2Fp8SealFailure::kBindingRange);
    }
    sealed_bindings[ordinal] = binding;
  }

  static std::atomic<std::uint64_t> seal_serial{1U};
  std::uint64_t nonce = 0x5133'5832'5034'3046ULL;
  nonce = mix_seal_nonce(nonce, request.deployment_identity);
  nonce = mix_seal_nonce(nonce, request.binding_catalog_identity);
  nonce = mix_seal_nonce(nonce, request.binding_lifetime_owner_identity);
  nonce = mix_seal_nonce(nonce, request.cuda_stream_owner_identity);
  nonce = mix_seal_nonce(nonce,
                         request.authenticated_weight_owner_identity);
  nonce = mix_seal_nonce(
      nonce, reinterpret_cast<std::uintptr_t>(request.cuda_stream));
  nonce = mix_seal_nonce(
      nonce, seal_serial.fetch_add(1U, std::memory_order_relaxed));
  if (nonce == 0U) {
    nonce = 0x5133'5832'5034'3046ULL;
  }

  Sm87BulkV2Fp8SealReceipt receipt{};
  receipt.magic = kSm87BulkV2Fp8SealReceiptMagic;
  receipt.version = kSm87BulkV2Fp8SealReceiptVersion;
  receipt.seal_nonce = nonce;
  receipt.deployment_identity = request.deployment_identity;
  receipt.binding_catalog_identity = request.binding_catalog_identity;
  receipt.binding_lifetime_owner_identity =
      request.binding_lifetime_owner_identity;
  receipt.cuda_stream_owner_identity = request.cuda_stream_owner_identity;
  receipt.authenticated_weight_owner_identity =
      request.authenticated_weight_owner_identity;
  receipt.device_ordinal = request.device_ordinal;
  receipt.cuda_stream = request.cuda_stream;
  receipt.sealed_role_bindings = sealed_bindings.size();
  receipt.sealed_resource_roles = resources.roles.size();
  receipt.hot_path_static_cuda_queries = 0U;
  receipt.exact_sm87_device_validated = true;
  receipt.nonblocking_stream_validated = true;
  receipt.complete_device_ranges_validated = true;
  receipt.all_authenticated_weight_owners_match = true;
  receipt.static_resources_validated_at_startup = true;
  receipt.request_hot_path_prevalidated = true;
  receipt.numerical_contract_qualified = false;
  receipt.production_dispatch_eligible = false;
  if (!sm87_bulk_v2_fp8_seal_receipt_valid(receipt)) {
    return fp8_seal_failure(cudaErrorInvalidValue,
                            Sm87BulkV2Fp8SealFailure::kInvalidRequest);
  }

  auto access = std::unique_ptr<Sm87BulkV2Fp8SealedAccess>(
      new (std::nothrow) Sm87BulkV2Fp8SealedAccess(
          sealed_bindings, request.cuda_stream, receipt));
  if (access == nullptr) {
    return fp8_seal_failure(cudaErrorMemoryAllocation,
                            Sm87BulkV2Fp8SealFailure::kAllocation);
  }
  Sm87BulkV2Fp8SealResult result;
  result.access = std::move(access);
  result.receipt = receipt;
  result.cuda_error = static_cast<int>(cudaSuccess);
  result.failure = Sm87BulkV2Fp8SealFailure::kNone;
  return result;
}

int enqueue_sm87_bulk_v2_fp8_role_p40_prevalidated_cuda(
    const Sm87BulkV2Fp8SealedAccess& access,
    const std::uint64_t request_epoch, const std::size_t layer,
    const Sm87TargetAotProjectionRole role,
    Sm87BulkV2Fp8SubmissionReceipt* const receipt) noexcept {
  if (receipt == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *receipt = {};
  const std::size_t ordinal = sm87_bulk_v2_fp8_role_ordinal(layer, role);
  if (!access.valid() || request_epoch == 0U ||
      ordinal >= kSm87BulkV2Fp8LogicalRoleCount) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  receipt->magic = kSm87BulkV2Fp8SubmissionReceiptMagic;
  receipt->version = kSm87BulkV2Fp8SubmissionReceiptVersion;
  receipt->seal_nonce = access.receipt_.seal_nonce;
  receipt->request_epoch = request_epoch;
  receipt->layer = static_cast<std::uint32_t>(layer);
  receipt->role = role;
  receipt->expected_launches = kSm87BulkV2Fp8SegmentsPerRole;
  receipt->failed_launch_ordinal =
      std::numeric_limits<std::size_t>::max();
  receipt->prevalidated_hot_path_used = true;
  receipt->static_cuda_query_issued = false;

  const cudaError_t prior_error = cudaGetLastError();
  if (prior_error != cudaSuccess) {
    receipt->cuda_error = static_cast<int>(prior_error);
    receipt->state =
        Sm87BulkV2Fp8SubmissionState::kFailedBeforeSubmission;
    return receipt->cuda_error;
  }

  const auto& binding = access.bindings_[ordinal];
  const auto plan = sm87_bulk_v2_fp8_role_plan(role);
  std::size_t first_token = 0U;
  for (std::size_t segment = 0U;
       segment < kSm87BulkV2Fp8SegmentsPerRole; ++segment) {
    const std::size_t tokens =
        segment < kSm87BulkV2Fp8MacroSegments
            ? kSm87BulkV2Fp8MacroTokens
            : kSm87BulkV2Fp8TailTokens;
    Sm87BulkV2Fp8OracleArguments launch{};
    launch.role = role;
    launch.input = binding.input + first_token * plan.input_features;
    launch.payload = reinterpret_cast<const std::uint8_t*>(
        binding.asset.authenticated.payload.begin);
    launch.compensated_scale_bf16_bits =
        binding.asset.authenticated.compensated_tensor_scale_bf16_bits;
    launch.token_count = tokens;
    launch.primary_output = binding.primary_output +
                            first_token * plan.primary_output_features;
    if (plan.secondary_output_features != 0U) {
      launch.secondary_output = binding.secondary_output +
                                first_token * plan.secondary_output_features;
      launch.tertiary_output = binding.tertiary_output +
                               first_token * plan.tertiary_output_features;
    }
    launch.cuda_stream = access.cuda_stream_;

    ++receipt->attempted_launches;
    const cudaError_t launch_status = launch_kernel(launch, false);
    if (launch_status != cudaSuccess) {
      receipt->failed_launch_ordinal = segment;
      receipt->cuda_error = static_cast<int>(launch_status);
      receipt->owner_drain_required = receipt->submitted_launches != 0U;
      receipt->state =
          receipt->owner_drain_required
              ? Sm87BulkV2Fp8SubmissionState::kFailedAfterPartialSubmission
              : Sm87BulkV2Fp8SubmissionState::kFailedBeforeSubmission;
      return receipt->cuda_error;
    }
    ++receipt->submitted_launches;
    first_token += tokens;
  }

  receipt->cuda_error = static_cast<int>(cudaSuccess);
  receipt->state = Sm87BulkV2Fp8SubmissionState::kSubmitted;
  receipt->owner_drain_required = false;
  return static_cast<int>(cudaSuccess);
}

}  // namespace sm87_bulk_v2_fp8_execution_detail

}  // namespace q3x::kernels
