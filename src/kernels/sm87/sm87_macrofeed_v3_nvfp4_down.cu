#include "q3x/kernels/sm87_macrofeed_v3_nvfp4_down.h"

#include "q3x/kernels/sm87_target_aot_projection_layout.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87MacroFeedV3NvFp4DownThreads);
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kTileM =
    static_cast<unsigned int>(kSm87MacroFeedV3NvFp4DownBlockM);
constexpr unsigned int kTileN =
    static_cast<unsigned int>(kSm87MacroFeedV3NvFp4DownBlockN);
constexpr unsigned int kTileK =
    static_cast<unsigned int>(kSm87MacroFeedV3NvFp4DownBlockK);
constexpr unsigned int kStages =
    static_cast<unsigned int>(kSm87MacroFeedV3NvFp4DownPipelineStages);
constexpr unsigned int kPersistentCtas =
    static_cast<unsigned int>(kSm87MacroFeedV3NvFp4DownPersistentCtas);
constexpr unsigned int kM16Panels = 8U;
constexpr unsigned int kN8PanelsPerWarp = 4U;
constexpr unsigned int kN8PanelsPerN64 = 8U;
constexpr unsigned int kK16Panels = 4U;
constexpr unsigned int kActivationVectors = kTileM * kTileK / 8U;
constexpr unsigned int kBranchWeightVectors =
    static_cast<unsigned int>(
        kSm87MacroFeedV3NvFp4DownWeightBytesPerCell / 2U /
        sizeof(uint4));
constexpr unsigned int kBranchScaleVectors =
    static_cast<unsigned int>(
        kSm87MacroFeedV3NvFp4DownScaleBytesPerCell / 2U /
        sizeof(uint4));

static_assert(kThreads == 8U * kWarpSize);
static_assert(kTileM == 128U && kTileN == 256U && kTileK == 64U);
static_assert(kStages == 3U && kPersistentCtas == 16U);
static_assert(kM16Panels == 8U && kN8PanelsPerWarp == 4U);
static_assert(kActivationVectors == 1'024U);
static_assert(kBranchWeightVectors == 256U);
static_assert(kBranchScaleVectors == 32U);

struct alignas(32) PipelineStorage final {
  uint4 activations[kStages][kTileM][kTileK / 8U];
  uint4 weights[kStages][2U][kBranchWeightVectors];
  uint4 scales[kStages][2U][kBranchScaleVectors];
};

static_assert(sizeof(PipelineStorage) ==
              kSm87MacroFeedV3NvFp4DownDynamicSharedBytes);

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

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ float decode_e2m1(
    const std::uint8_t code) noexcept {
  const std::uint8_t magnitude = code & 0x07U;
  float value = 0.0F;
  switch (magnitude) {
    case 1U:
      value = 0.5F;
      break;
    case 2U:
      value = 1.0F;
      break;
    case 3U:
      value = 1.5F;
      break;
    case 4U:
      value = 2.0F;
      break;
    case 5U:
      value = 3.0F;
      break;
    case 6U:
      value = 4.0F;
      break;
    case 7U:
      value = 6.0F;
      break;
    case 0U:
    default:
      break;
  }
  return (code & 0x08U) != 0U ? -value : value;
}

[[nodiscard]] __device__ __forceinline__ float decode_e4m3fn_scale(
    const std::uint8_t code) noexcept {
  const unsigned int magnitude = code & 0x7fU;
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  float value = 0.0F;
  if (exponent == 0U) {
    value = ldexpf(static_cast<float>(mantissa), -9);
  } else {
    value = ldexpf(static_cast<float>(8U + mantissa),
                   static_cast<int>(exponent) - 10);
  }
  return (code & 0x80U) != 0U ? -value : value;
}

[[nodiscard]] __device__ __forceinline__ K16N8Weight
decode_weight_fragment(const std::uint8_t* const shared_weight,
                       const std::uint8_t* const shared_scale,
                       const unsigned int k16,
                       const unsigned int local_n_warp,
                       const unsigned int n8_panel,
                       const unsigned int lane) noexcept {
  const unsigned int fragment =
      (k16 * 2U + local_n_warp) * kN8PanelsPerN64 + n8_panel;
  const std::uint16_t packed = *reinterpret_cast<const std::uint16_t*>(
      shared_weight + fragment * 64U + lane * 2U);
  const std::uint8_t scale_code =
      shared_scale[fragment * 8U + lane / 4U];
  const float scale = decode_e4m3fn_scale(scale_code);
  const auto packed_component = [packed](const unsigned int component) {
    return static_cast<std::uint8_t>((packed >> (4U * component)) & 0x0fU);
  };
  const std::uint16_t component0 = encode_bf16_rne(
      decode_e2m1(packed_component(
          sm87_target_aot_projection_mma_b_register_component(0U))) *
      scale);
  const std::uint16_t component1 = encode_bf16_rne(
      decode_e2m1(packed_component(
          sm87_target_aot_projection_mma_b_register_component(1U))) *
      scale);
  const std::uint16_t component2 = encode_bf16_rne(
      decode_e2m1(packed_component(
          sm87_target_aot_projection_mma_b_register_component(2U))) *
      scale);
  const std::uint16_t component3 = encode_bf16_rne(
      decode_e2m1(packed_component(
          sm87_target_aot_projection_mma_b_register_component(3U))) *
      scale);
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
    PipelineStorage* const storage, const unsigned int slot,
    const std::uint16_t* const input, const std::uint8_t* const payload,
    const unsigned int rows, const unsigned int first_m,
    const unsigned int n_tile, const unsigned int k_tile) noexcept {
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
    cp_async_cg_16<true>(&storage->activations[slot][row][vector], source,
                         row_valid);
  }

#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int combined = threadIdx.x + pass * kThreads;
    const unsigned int branch = combined / kBranchWeightVectors;
    const unsigned int local_vector = combined % kBranchWeightVectors;
    const unsigned int vector_in_fragment = local_vector & 3U;
    const unsigned int fragment = local_vector >> 2U;
    const unsigned int n8_panel = fragment % kN8PanelsPerN64;
    const unsigned int outer = fragment / kN8PanelsPerN64;
    const unsigned int local_n_warp = outer & 1U;
    const unsigned int k16 = outer >> 1U;
    const unsigned int global_n_warp = branch * 2U + local_n_warp;
    const std::uint64_t cell =
        (static_cast<std::uint64_t>(n_tile) * kKTiles + k_tile) *
        kSm87MacroFeedV3NvFp4DownCellBytes;
    const std::uint64_t global_fragment =
        (static_cast<std::uint64_t>(k16) * 4U + global_n_warp) *
            kN8PanelsPerN64 +
        n8_panel;
    const auto* const source = reinterpret_cast<const uint4*>(
        payload + cell + global_fragment * 64U) + vector_in_fragment;
    cp_async_cg_16<false>(
        &storage->weights[slot][branch][local_vector], source);
  }

  if (threadIdx.x < 2U * kBranchScaleVectors) {
    const unsigned int combined = threadIdx.x;
    const unsigned int branch = combined / kBranchScaleVectors;
    const unsigned int local_vector = combined % kBranchScaleVectors;
    const unsigned int n8_pair = local_vector & 3U;
    const unsigned int outer = local_vector >> 2U;
    const unsigned int local_n_warp = outer & 1U;
    const unsigned int k16 = outer >> 1U;
    const unsigned int global_n_warp = branch * 2U + local_n_warp;
    const std::uint64_t cell =
        (static_cast<std::uint64_t>(n_tile) * kKTiles + k_tile) *
        kSm87MacroFeedV3NvFp4DownCellBytes;
    const std::uint64_t scale_byte =
        (static_cast<std::uint64_t>(k16) * 4U + global_n_warp) * 64U +
        n8_pair * 16U;
    const auto* const source = reinterpret_cast<const uint4*>(
        payload + cell + kSm87MacroFeedV3NvFp4DownWeightBytesPerCell +
        scale_byte);
    cp_async_cg_16<false>(
        &storage->scales[slot][branch][local_vector], source);
  }
  cp_async_commit_group();
}

template <unsigned int kInputFeatures, unsigned int kKTiles>
__device__ __forceinline__ void run_full_k(
    PipelineStorage* const storage, const std::uint16_t* const input,
    const std::uint8_t* const payload, const unsigned int rows,
    const unsigned int first_m, const unsigned int n_tile,
    WarpAccumulator& accumulators) noexcept {
  static_assert(kKTiles >= kStages);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int branch = warp / 4U;
  const unsigned int branch_quarter = warp % 4U;
  const unsigned int local_n_warp = branch_quarter / 2U;
  const unsigned int first_n8 =
      (branch_quarter % 2U) * kN8PanelsPerWarp;

  clear_accumulators(accumulators);
  issue_pipeline_stage<kInputFeatures, kKTiles>(
      storage, 0U, input, payload, rows, first_m, n_tile, 0U);
  issue_pipeline_stage<kInputFeatures, kKTiles>(
      storage, 1U, input, payload, rows, first_m, n_tile, 1U);
  issue_pipeline_stage<kInputFeatures, kKTiles>(
      storage, 2U, input, payload, rows, first_m, n_tile, 2U);

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
        reinterpret_cast<const std::uint16_t*>(
            storage->activations[slot]);
    const auto* const shared_weight = reinterpret_cast<const std::uint8_t*>(
        storage->weights[slot][branch]);
    const auto* const shared_scale = reinterpret_cast<const std::uint8_t*>(
        storage->scales[slot][branch]);

    DecodedWeightStage decoded[2U];
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      decoded[0U][n8] = decode_weight_fragment(
          shared_weight, shared_scale, 0U, local_n_warp, first_n8 + n8,
          lane);
    }

#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16Panels; ++k16) {
      const unsigned int current = k16 & 1U;
      const unsigned int next = current ^ 1U;
      if (k16 + 1U < kK16Panels) {
#pragma unroll
        for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
          decoded[next][n8] = decode_weight_fragment(
              shared_weight, shared_scale, k16 + 1U, local_n_warp,
              first_n8 + n8, lane);
        }
      }

      M16K16Activation activation[kM16Panels];
#pragma unroll
      for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
        load_activation_fragment(activation[m16], shared_activations, m16,
                                 k16, lane);
      }

      if (k16 + 1U == kK16Panels) {
        __syncthreads();
        if (k_tile + kStages < kKTiles) {
          issue_pipeline_stage<kInputFeatures, kKTiles>(
              storage, slot, input, payload, rows, first_m, n_tile,
              k_tile + kStages);
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

[[nodiscard]] __device__ __forceinline__ std::uint32_t
pack_scaled_bf16_pair(const float low, const float high,
                      const float tensor_scale) noexcept {
  return static_cast<std::uint32_t>(encode_bf16_rne(low * tensor_scale)) |
         (static_cast<std::uint32_t>(
              encode_bf16_rne(high * tensor_scale))
          << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t add_residual_pair(
    const std::uint32_t branch_bits,
    const std::uint32_t residual_bits) noexcept {
  const float branch0 =
      decode_bf16(static_cast<std::uint16_t>(branch_bits));
  const float branch1 =
      decode_bf16(static_cast<std::uint16_t>(branch_bits >> 16U));
  const float residual0 =
      decode_bf16(static_cast<std::uint16_t>(residual_bits));
  const float residual1 =
      decode_bf16(static_cast<std::uint16_t>(residual_bits >> 16U));
  return static_cast<std::uint32_t>(
             encode_bf16_rne(branch0 + residual0)) |
         (static_cast<std::uint32_t>(
              encode_bf16_rne(branch1 + residual1))
          << 16U);
}

template <unsigned int kOutputFeatures>
__device__ __forceinline__ void publish_down_residual(
    const WarpAccumulator& accumulators, const unsigned int rows,
    const unsigned int first_m, const unsigned int first_n,
    const unsigned int warp, const unsigned int lane,
    const float tensor_scale, std::uint16_t* const residual) noexcept {
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
          first_n + warp * kSm87MacroFeedV3NvFp4DownWarpN + n8 * 8U +
          lane_in_group * 2U;
      const auto& value = accumulators[m16][n8];
      if (global_row0 < rows) {
        auto* const destination = reinterpret_cast<std::uint32_t*>(
            residual + static_cast<std::size_t>(global_row0) *
                           kOutputFeatures +
            column);
        *destination = add_residual_pair(
            pack_scaled_bf16_pair(value.x0, value.x1, tensor_scale),
            *destination);
      }
      if (global_row1 < rows) {
        auto* const destination = reinterpret_cast<std::uint32_t*>(
            residual + static_cast<std::size_t>(global_row1) *
                           kOutputFeatures +
            column);
        *destination = add_residual_pair(
            pack_scaled_bf16_pair(value.x2, value.x3, tensor_scale),
            *destination);
      }
    }
  }
}

__global__ __launch_bounds__(kThreads, 1)
void sm87_macrofeed_v3_nvfp4_down_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const float tensor_scale,
    std::uint16_t* __restrict__ residual) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage = reinterpret_cast<PipelineStorage*>(dynamic_storage);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;

  for (unsigned int linear_task = blockIdx.x;
       linear_task < kSm87MacroFeedV3NvFp4DownLogicalTasks;
       linear_task += kPersistentCtas) {
    const unsigned int m_tile =
        linear_task % kSm87MacroFeedV3NvFp4DownGridM;
    const unsigned int n_tile =
        linear_task / kSm87MacroFeedV3NvFp4DownGridM;
    const unsigned int first_m = m_tile * kTileM;
    WarpAccumulator accumulators;
    run_full_k<kSm87MacroFeedV3NvFp4DownInputFeatures,
               kSm87MacroFeedV3NvFp4DownKTiles>(
        storage, input, payload, kSm87MacroFeedV3NvFp4DownTokens, first_m,
        n_tile, accumulators);
    publish_down_residual<kSm87MacroFeedV3NvFp4DownOutputFeatures>(
        accumulators, kSm87MacroFeedV3NvFp4DownTokens, first_m,
        n_tile * kTileN, warp, lane, tensor_scale, residual);
    __syncthreads();
  }
}

__global__ __launch_bounds__(kThreads, 1)
void sm87_macrofeed_v3_nvfp4_down_tile_test_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const unsigned int valid_rows,
    const float tensor_scale, std::uint16_t* __restrict__ residual) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage = reinterpret_cast<PipelineStorage*>(dynamic_storage);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  WarpAccumulator accumulators;
  run_full_k<kSm87MacroFeedV3NvFp4DownTestInputFeatures,
             kSm87MacroFeedV3NvFp4DownTestKTiles>(
      storage, input, payload, valid_rows, 0U, 0U, accumulators);
  publish_down_residual<kTileN>(accumulators, valid_rows, 0U, 0U, warp,
                                lane, tensor_scale, residual);
}

[[nodiscard]] cudaError_t set_dynamic_shared_attribute() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      sm87_macrofeed_v3_nvfp4_down_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87MacroFeedV3NvFp4DownDynamicSharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      sm87_macrofeed_v3_nvfp4_down_tile_test_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87MacroFeedV3NvFp4DownDynamicSharedBytes));
}

[[nodiscard]] cudaError_t validate_device(int* const device_ordinal,
                                          cudaDeviceProp* const properties)
    noexcept {
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
                     static_cast<int>(kSm87MacroFeedV3NvFp4DownSmCount) &&
                 properties->sharedMemPerBlockOptin >=
                     kSm87MacroFeedV3NvFp4DownDynamicSharedBytes
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] bool device_pointer(const void* const pointer,
                                  const int device_ordinal) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  return status == cudaSuccess && attributes.type == cudaMemoryTypeDevice &&
         attributes.device == device_ordinal;
}

struct PointerRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] PointerRange pointer_range(const void* const pointer,
                                         const std::size_t bytes) noexcept {
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin == 0U || bytes == 0U ||
      begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return {};
  }
  return {begin, begin + bytes, true};
}

[[nodiscard]] bool disjoint(const PointerRange& left,
                            const PointerRange& right) noexcept {
  return left.valid && right.valid &&
         (left.end <= right.begin || right.end <= left.begin);
}

[[nodiscard]] cudaError_t query_resources_body(
    Sm87MacroFeedV3NvFp4DownCudaResources* const resources) noexcept {
  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status = validate_device(&device, &properties);
  if (status != cudaSuccess) {
    return status;
  }
  status = set_dynamic_shared_attribute();
  if (status != cudaSuccess) {
    return status;
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes,
                                 sm87_macrofeed_v3_nvfp4_down_kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, sm87_macrofeed_v3_nvfp4_down_kernel, kThreads,
      kSm87MacroFeedV3NvFp4DownDynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }

  resources->identity = kSm87MacroFeedV3NvFp4DownIdentity;
  resources->device_ordinal = device;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  resources->sm_count = properties.multiProcessorCount;
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87MacroFeedV3NvFp4DownDynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->optin_shared_bytes_per_block =
      properties.sharedMemPerBlockOptin;
  resources->kernel_compiled = true;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  resources->static_resource_gate_passed =
      sm87_macrofeed_v3_nvfp4_down_resource_gate(*resources);
  return cudaSuccess;
}

}  // namespace

bool sm87_macrofeed_v3_nvfp4_down_arguments_valid(
    const Sm87MacroFeedV3NvFp4DownArguments& arguments) noexcept {
  const auto plan =
      sm87_macrofeed_v3_nvfp4_down_plan(arguments.token_count);
  if (!plan.valid() || arguments.input == nullptr ||
      arguments.payload == nullptr || arguments.residual == nullptr ||
      arguments.payload_bytes != plan.payload_bytes ||
      !std::isfinite(arguments.tensor_scale) ||
      arguments.tensor_scale <= 0.0F ||
      reinterpret_cast<std::uintptr_t>(arguments.input) % alignof(uint4) !=
          0U ||
      reinterpret_cast<std::uintptr_t>(arguments.payload) %
              kSm87MacroFeedV3NvFp4DownPayloadAlignment !=
          0U ||
      reinterpret_cast<std::uintptr_t>(arguments.residual) %
              alignof(std::uint32_t) !=
          0U ||
      !sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
          arguments.payload_receipt) ||
      arguments.payload_receipt.payload_begin !=
          reinterpret_cast<std::uintptr_t>(arguments.payload) ||
      arguments.payload_receipt.payload_bytes != arguments.payload_bytes) {
    return false;
  }

  constexpr std::size_t kInputBytes =
      kSm87MacroFeedV3NvFp4DownTokens *
      kSm87MacroFeedV3NvFp4DownInputFeatures * sizeof(std::uint16_t);
  constexpr std::size_t kResidualBytes =
      kSm87MacroFeedV3NvFp4DownTokens *
      kSm87MacroFeedV3NvFp4DownOutputFeatures * sizeof(std::uint16_t);
  const auto input = pointer_range(arguments.input, kInputBytes);
  const auto payload = pointer_range(arguments.payload, arguments.payload_bytes);
  const auto residual = pointer_range(arguments.residual, kResidualBytes);
  return disjoint(input, payload) && disjoint(input, residual) &&
         disjoint(payload, residual);
}

int query_sm87_macrofeed_v3_nvfp4_down_cuda_resources(
    Sm87MacroFeedV3NvFp4DownCudaResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  return static_cast<int>(query_resources_body(resources));
}

int launch_sm87_macrofeed_v3_nvfp4_down_cuda(
    const Sm87MacroFeedV3NvFp4DownArguments& arguments,
    Sm87MacroFeedV3NvFp4DownLaunchReceipt* const receipt) noexcept {
  if (receipt == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *receipt = {};
  if (!sm87_macrofeed_v3_nvfp4_down_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  Sm87MacroFeedV3NvFp4DownCudaResources resources{};
  const cudaError_t resource_status = query_resources_body(&resources);
  if (resource_status != cudaSuccess) {
    return static_cast<int>(resource_status);
  }
  if (!resources.static_resource_gate_passed ||
      !sm87_macrofeed_v3_nvfp4_down_resource_gate(resources)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (resources.device_ordinal != arguments.payload_receipt.device_ordinal ||
      !device_pointer(arguments.input, resources.device_ordinal) ||
      !device_pointer(arguments.payload, resources.device_ordinal) ||
      !device_pointer(arguments.residual, resources.device_ordinal)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  (void)cudaGetLastError();
  sm87_macrofeed_v3_nvfp4_down_kernel
      <<<kPersistentCtas, kThreads,
         kSm87MacroFeedV3NvFp4DownDynamicSharedBytes, stream>>>(
          arguments.input, arguments.payload, arguments.tensor_scale,
          arguments.residual);
  const cudaError_t launch_status = cudaPeekAtLastError();
  if (launch_status != cudaSuccess) {
    return static_cast<int>(launch_status);
  }
  *receipt = {kSm87MacroFeedV3NvFp4DownIdentity,
              arguments.payload_receipt.payload_identity,
              arguments.token_count,
              kSm87MacroFeedV3NvFp4DownLogicalTasks,
              kSm87MacroFeedV3NvFp4DownTailRows,
              1U,
              0U,
              true,
              false,
              true,
              false};
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_macrofeed_v3_nvfp4_down_tile_test_cuda(
    const std::uint16_t* const input_m128_k256,
    const std::uint8_t* const canonical_payload_four_cells,
    const float tensor_scale, const std::size_t valid_rows,
    std::uint16_t* const residual_m128_n256,
    void* const cuda_stream) noexcept {
  if (input_m128_k256 == nullptr ||
      canonical_payload_four_cells == nullptr ||
      residual_m128_n256 == nullptr ||
      (valid_rows != 64U && valid_rows != 128U) ||
      !std::isfinite(tensor_scale) || tensor_scale <= 0.0F ||
      reinterpret_cast<std::uintptr_t>(input_m128_k256) % alignof(uint4) !=
          0U ||
      reinterpret_cast<std::uintptr_t>(canonical_payload_four_cells) %
              alignof(uint4) !=
          0U ||
      reinterpret_cast<std::uintptr_t>(residual_m128_n256) %
              alignof(std::uint32_t) !=
          0U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  Sm87MacroFeedV3NvFp4DownCudaResources resources{};
  const cudaError_t resource_status = query_resources_body(&resources);
  if (resource_status != cudaSuccess) {
    return static_cast<int>(resource_status);
  }
  if (!resources.static_resource_gate_passed ||
      !device_pointer(input_m128_k256, resources.device_ordinal) ||
      !device_pointer(canonical_payload_four_cells,
                      resources.device_ordinal) ||
      !device_pointer(residual_m128_n256, resources.device_ordinal)) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  sm87_macrofeed_v3_nvfp4_down_tile_test_kernel
      <<<1U, kThreads, kSm87MacroFeedV3NvFp4DownDynamicSharedBytes,
         stream>>>(input_m128_k256, canonical_payload_four_cells,
                   static_cast<unsigned int>(valid_rows), tensor_scale,
                   residual_m128_n256);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
