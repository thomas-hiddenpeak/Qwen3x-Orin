#include "q3x/kernels/sm87_macrofeed_v4_fp8.h"

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
#include "sm87_macrofeed_v4_bound_launch_internal.h"
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace q3x::kernels {
namespace {

constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87MacroFeedV4Fp8Threads);
constexpr unsigned int kTileM =
    static_cast<unsigned int>(kSm87MacroFeedV4Fp8BlockM);
constexpr unsigned int kTileN =
    static_cast<unsigned int>(kSm87MacroFeedV4Fp8BlockN);
constexpr unsigned int kTileK =
    static_cast<unsigned int>(kSm87MacroFeedV4Fp8BlockK);
constexpr unsigned int kStages =
    static_cast<unsigned int>(kSm87MacroFeedV4Fp8PipelineStages);
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kM16PanelsPerWarp = 2U;
constexpr unsigned int kN8PanelsPerWarp = 4U;
constexpr unsigned int kK16Panels = 4U;
constexpr unsigned int kActivationVectors = kTileM * kTileK / 8U;
constexpr unsigned int kWeightVectors = kTileN * kTileK / 16U;
constexpr unsigned int kAuthenticatedN256CellBytes = 16'384U;
constexpr unsigned int kAuthenticatedK16Bytes = 4'096U;
constexpr unsigned int kN128K16Bytes = 2'048U;

static_assert(kActivationVectors == 512U && kWeightVectors == 512U);
static_assert(kSm87MacroFeedV4Fp8ActivationBytesPerStage == 8'192U);
static_assert(kSm87MacroFeedV4Fp8WeightBytesPerStage == 8'192U);

struct alignas(32) Fp8Pipeline final {
  uint4 activations[kStages][kActivationVectors];
  uint4 weights[kStages][kWeightVectors];
};

static_assert(sizeof(Fp8Pipeline) ==
              kSm87MacroFeedV4Fp8DynamicSharedBytes);

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
    M16N8Accumulator[kM16PanelsPerWarp][kN8PanelsPerWarp];
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
  const unsigned int column =
      k16 * 16U + (quadrant >> 1U) * 8U;
  const auto* const source =
      shared_activations +
      activation_swizzled_vector(row, column / 8U) * 8U;
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
                       const unsigned int local_n8_panel,
                       const unsigned int lane) noexcept {
  const unsigned int fragment = k16 * 16U + local_n8_panel;
  const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
      shared_weight + fragment * 128U + lane * 4U);
  const auto decode = [packed](const unsigned int component) {
    const auto code =
        static_cast<std::uint8_t>(packed >> (8U * component));
    return static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(code & 0x80U) << 8U) |
        (static_cast<std::uint16_t>(code & 0x7fU) << 4U));
  };
  // Persisted [K0,K8,K1,K9] becomes MMA [K0,K1] and [K8,K9].
  const std::uint16_t component0 = decode(0U);
  const std::uint16_t component1 = decode(2U);
  const std::uint16_t component2 = decode(1U);
  const std::uint16_t component3 = decode(3U);
  return {static_cast<std::uint32_t>(component0) |
              (static_cast<std::uint32_t>(component1) << 16U),
          static_cast<std::uint32_t>(component2) |
              (static_cast<std::uint32_t>(component3) << 16U)};
}

__device__ __forceinline__ void clear_accumulators(
    WarpAccumulator& accumulators) noexcept {
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16PanelsPerWarp; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      accumulators[m16][n8] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }
}

template <unsigned int kInputFeatures, unsigned int kKTiles,
          bool kPredicatedRows,
          Sm87MacroFeedV4Fp8InputLayout kInputLayout>
__device__ __forceinline__ void issue_pipeline_stage(
    Fp8Pipeline* const pipeline, const unsigned int slot,
    const std::uint16_t* const input, const std::size_t input_row_stride,
    const std::uint8_t* const payload, const unsigned int rows,
    const unsigned int first_m, const std::uint64_t partition_offset,
    const unsigned int partition_n128_tile,
    const unsigned int input_first_k_tile,
    const unsigned int k_tile) noexcept {
  static_assert(kInputFeatures == kKTiles * kTileK);
  const unsigned int logical_first_k =
      (input_first_k_tile + k_tile) * kTileK;
  unsigned int physical_first_k = logical_first_k;
  if constexpr (kInputLayout ==
                Sm87MacroFeedV4Fp8InputLayout::
                    kFullAttentionInterleavedQScratchV1) {
    physical_first_k =
        (logical_first_k /
         static_cast<unsigned int>(
             kSm87MacroFeedV4Fp8AttentionHeadFeatures)) *
            static_cast<unsigned int>(kSm87MacroFeedV4Fp8QGateHeadStride) +
        logical_first_k %
            static_cast<unsigned int>(
                kSm87MacroFeedV4Fp8AttentionHeadFeatures);
  } else if constexpr (kInputLayout ==
                       Sm87MacroFeedV4Fp8InputLayout::
                           kGdnContiguousVScratchV1) {
    physical_first_k =
        static_cast<unsigned int>(
            kSm87MacroFeedV4Fp8GdnAttentionOutputPhysicalOffset) +
        logical_first_k;
  }
#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const unsigned int row = vector_index / (kTileK / 8U);
    const unsigned int vector = vector_index % (kTileK / 8U);
    const bool row_valid = !kPredicatedRows || first_m + row < rows;
    const unsigned int source_row = row_valid ? first_m + row : 0U;
    const auto* const source = reinterpret_cast<const uint4*>(
                                   input +
                                   static_cast<std::size_t>(source_row) *
                                       input_row_stride +
                                   physical_first_k) +
                               vector;
    cp_async_cg_16<kPredicatedRows>(
        &pipeline->activations[slot]
                              [activation_swizzled_vector(row, vector)],
        source, row_valid);
  }
  const unsigned int parent_n256_tile = partition_n128_tile / 2U;
  const unsigned int n128_half = partition_n128_tile & 1U;
  const std::uint64_t cell =
      partition_offset +
      (static_cast<std::uint64_t>(parent_n256_tile) * kKTiles + k_tile) *
          kAuthenticatedN256CellBytes;
#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const unsigned int k16 = vector_index / 128U;
    const unsigned int local_vector = vector_index % 128U;
    const auto* const source = reinterpret_cast<const uint4*>(
        payload + cell + k16 * kAuthenticatedK16Bytes +
        n128_half * kN128K16Bytes + local_vector * sizeof(uint4));
    cp_async_cg_16<false>(&pipeline->weights[slot][vector_index], source);
  }
  cp_async_commit_group();
}

template <unsigned int kInputFeatures, unsigned int kKTiles,
          bool kPredicatedRows,
          Sm87MacroFeedV4Fp8InputLayout kInputLayout>
__device__ __forceinline__ void run_full_k(
    Fp8Pipeline* const pipeline, const std::uint16_t* const input,
    const std::size_t input_row_stride, const std::uint8_t* const payload,
    const unsigned int rows, const unsigned int first_m,
    const std::uint64_t partition_offset,
    const unsigned int partition_n128_tile,
    const unsigned int input_first_k_tile,
    WarpAccumulator& accumulators) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_m = warp / 4U;
  const unsigned int warp_n = warp % 4U;
  clear_accumulators(accumulators);

#pragma unroll
  for (unsigned int stage = 0U; stage < kStages; ++stage) {
    issue_pipeline_stage<kInputFeatures, kKTiles, kPredicatedRows,
                         kInputLayout>(
        pipeline, stage, input, input_row_stride, payload, rows, first_m,
        partition_offset, partition_n128_tile, input_first_k_tile, stage);
  }

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
            pipeline->activations[slot]);
    const auto* const shared_weight =
        reinterpret_cast<const std::uint8_t*>(pipeline->weights[slot]);

    DecodedWeightStage decoded[2U];
    const unsigned int first_n8 = warp_n * kN8PanelsPerWarp;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      decoded[0U][n8] = decode_weight_fragment(
          shared_weight, 0U, first_n8 + n8, lane);
    }

#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16Panels; ++k16) {
      const unsigned int current = k16 & 1U;
      const unsigned int next = current ^ 1U;
      if (k16 + 1U < kK16Panels) {
#pragma unroll
        for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
          decoded[next][n8] = decode_weight_fragment(
              shared_weight, k16 + 1U, first_n8 + n8, lane);
        }
      }
      M16K16Activation activation[kM16PanelsPerWarp];
#pragma unroll
      for (unsigned int m16 = 0U; m16 < kM16PanelsPerWarp; ++m16) {
        load_activation_fragment(activation[m16], shared_activations,
                                 warp_m * kM16PanelsPerWarp + m16,
                                 k16, lane);
      }
      if (k16 + 1U == kK16Panels) {
        __syncthreads();
        if (k_tile + kStages < kKTiles) {
          issue_pipeline_stage<kInputFeatures, kKTiles, kPredicatedRows,
                               kInputLayout>(
              pipeline, slot, input, input_row_stride, payload, rows,
              first_m, partition_offset, partition_n128_tile,
              input_first_k_tile, k_tile + kStages);
        }
      }
#pragma unroll
      for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
#pragma unroll
        for (unsigned int m16 = 0U; m16 < kM16PanelsPerWarp; ++m16) {
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
    const unsigned int first_m, const unsigned int first_n,
    const std::size_t output_row_stride, const float compensated_scale,
    std::uint16_t* const output) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_m = warp / 4U;
  const unsigned int warp_n = warp % 4U;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16PanelsPerWarp; ++m16) {
    const unsigned int local_row0 =
        warp_m * 32U + m16 * 16U + lane_group;
    const unsigned int local_row1 = local_row0 + 8U;
    const unsigned int global_row0 = first_m + local_row0;
    const unsigned int global_row1 = first_m + local_row1;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      const unsigned int column =
          first_n + warp_n * 32U + n8 * 8U + lane_in_group * 2U;
      const auto& value = accumulators[m16][n8];
      if (global_row0 < rows) {
        const std::uint32_t packed =
            static_cast<std::uint32_t>(encode_bf16_rne(
                __fmul_rn(value.x0, compensated_scale))) |
            (static_cast<std::uint32_t>(encode_bf16_rne(
                 __fmul_rn(value.x1, compensated_scale)))
             << 16U);
        *reinterpret_cast<std::uint32_t*>(
            output + static_cast<std::size_t>(global_row0) *
                         output_row_stride +
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
            output + static_cast<std::size_t>(global_row1) *
                         output_row_stride +
            column) = packed;
      }
    }
  }
}

template <Sm87TargetAotProjectionRole kRole,
          unsigned int kInputFeatures, unsigned int kKTiles,
          Sm87MacroFeedV4Fp8InputLayout kInputLayout>
__global__ __launch_bounds__(256, 2)
void sm87_macrofeed_v4_fp8_kernel(
    const std::uint16_t* const input, const std::size_t input_row_stride,
    const std::uint8_t* const payload, const float scale0,
    const float scale1, const float scale2,
    std::uint16_t* const primary_output,
    const std::size_t primary_output_row_stride,
    std::uint16_t* const key_output,
    const std::size_t key_output_row_stride,
    std::uint16_t* const value_output,
    const std::size_t value_output_row_stride,
    const unsigned int n_tiles) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const pipeline = reinterpret_cast<Fp8Pipeline*>(dynamic_storage);
  const unsigned int m_tile = blockIdx.x / n_tiles;
  const unsigned int n_tile = blockIdx.x % n_tiles;

  unsigned int partition = 0U;
  unsigned int partition_first_n128 = 0U;
  std::uint64_t partition_offset = 0U;
  float compensated_scale = scale0;
  if constexpr (kRole == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    if (n_tile >= 80U) {
      partition = 1U;
      partition_first_n128 = 80U;
      partition_offset = 52'428'800U;
      compensated_scale = scale1;
    }
  } else if constexpr (kRole ==
                       Sm87TargetAotProjectionRole::kFp8FullQkv) {
    if (n_tile >= 104U) {
      partition = 2U;
      partition_first_n128 = 104U;
      partition_offset = 68'157'440U;
      compensated_scale = scale2;
    } else if (n_tile >= 96U) {
      partition = 1U;
      partition_first_n128 = 96U;
      partition_offset = 62'914'560U;
      compensated_scale = scale1;
    }
  }

  WarpAccumulator accumulators;
  run_full_k<kInputFeatures, kKTiles, false, kInputLayout>(
      pipeline, input, input_row_stride, payload,
      static_cast<unsigned int>(kSm87MacroFeedV4Fp8Tokens),
      m_tile * kTileM, partition_offset,
      n_tile - partition_first_n128, 0U, accumulators);

  std::uint16_t* destination = primary_output;
  std::size_t destination_row_stride = primary_output_row_stride;
  unsigned int destination_first_n = n_tile * kTileN;
  if constexpr (kRole == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    if (partition == 0U) {
      const unsigned int logical_first_n = n_tile * kTileN;
      const bool gate =
          logical_first_n >=
          static_cast<unsigned int>(kSm87MacroFeedV4Fp8FullQFeatures);
      const unsigned int local =
          gate ? logical_first_n -
                     static_cast<unsigned int>(
                         kSm87MacroFeedV4Fp8FullQFeatures)
               : logical_first_n;
      destination_first_n =
          (local /
           static_cast<unsigned int>(
               kSm87MacroFeedV4Fp8AttentionHeadFeatures)) *
              static_cast<unsigned int>(
                  kSm87MacroFeedV4Fp8QGateHeadStride) +
          local %
              static_cast<unsigned int>(
                  kSm87MacroFeedV4Fp8AttentionHeadFeatures) +
          (gate ? static_cast<unsigned int>(
                      kSm87MacroFeedV4Fp8AttentionHeadFeatures)
                : 0U);
    } else if (partition == 1U) {
      destination = key_output;
      destination_row_stride = key_output_row_stride;
      destination_first_n = (n_tile - partition_first_n128) * kTileN;
    } else if (partition == 2U) {
      destination = value_output;
      destination_row_stride = value_output_row_stride;
      destination_first_n = (n_tile - partition_first_n128) * kTileN;
    }
  }
  publish_output(accumulators,
                 static_cast<unsigned int>(kSm87MacroFeedV4Fp8Tokens),
                 m_tile * kTileM, destination_first_n,
                 destination_row_stride, compensated_scale, destination);
}

template <Sm87TargetAotProjectionRole kRole,
          Sm87MacroFeedV4Fp8InputLayout kInputLayout>
__global__ __launch_bounds__(256, 2)
void sm87_macrofeed_v4_fp8_tile_test_kernel(
    const std::uint16_t* const input, const std::size_t input_row_stride,
    const std::uint8_t* const payload, const float compensated_scale,
    const unsigned int valid_rows, const unsigned int partition,
    const unsigned int partition_n256_tile,
    const unsigned int canonical_n128_half,
    const unsigned int logical_input_first_k,
    std::uint16_t* const primary_output,
    const std::size_t primary_output_row_stride,
    std::uint16_t* const key_output,
    const std::size_t key_output_row_stride,
    std::uint16_t* const value_output,
    const std::size_t value_output_row_stride) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const pipeline = reinterpret_cast<Fp8Pipeline*>(dynamic_storage);
  WarpAccumulator accumulators;
  run_full_k<static_cast<unsigned int>(
                 kSm87MacroFeedV4Fp8TestInputFeatures),
             static_cast<unsigned int>(kSm87MacroFeedV4Fp8TestKTiles), true,
             kInputLayout>(
      pipeline, input, input_row_stride, payload, valid_rows, 0U, 0U,
      canonical_n128_half, logical_input_first_k / kTileK, accumulators);

  std::uint16_t* destination = primary_output;
  std::size_t destination_row_stride = primary_output_row_stride;
  const unsigned int partition_local_first_n =
      partition_n256_tile * 2U * kTileN + canonical_n128_half * kTileN;
  unsigned int destination_first_n = partition_local_first_n;
  if constexpr (kRole == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    if (partition == 1U) {
      destination_first_n =
          static_cast<unsigned int>(kSm87MacroFeedV4Fp8GdnZOffset) +
          partition_local_first_n;
    }
  } else if constexpr (kRole ==
                       Sm87TargetAotProjectionRole::kFp8FullQkv) {
    if (partition == 0U) {
      const bool gate =
          partition_local_first_n >=
          static_cast<unsigned int>(kSm87MacroFeedV4Fp8FullQFeatures);
      const unsigned int local =
          gate ? partition_local_first_n -
                     static_cast<unsigned int>(
                         kSm87MacroFeedV4Fp8FullQFeatures)
               : partition_local_first_n;
      destination_first_n =
          (local /
           static_cast<unsigned int>(
               kSm87MacroFeedV4Fp8AttentionHeadFeatures)) *
              static_cast<unsigned int>(
                  kSm87MacroFeedV4Fp8QGateHeadStride) +
          local %
              static_cast<unsigned int>(
                  kSm87MacroFeedV4Fp8AttentionHeadFeatures) +
          (gate ? static_cast<unsigned int>(
                      kSm87MacroFeedV4Fp8AttentionHeadFeatures)
                : 0U);
    } else if (partition == 1U) {
      destination = key_output;
      destination_row_stride = key_output_row_stride;
    } else if (partition == 2U) {
      destination = value_output;
      destination_row_stride = value_output_row_stride;
    }
  }
  publish_output(accumulators, valid_rows, 0U, destination_first_n,
                 destination_row_stride, compensated_scale, destination);
}

template <typename Kernel>
[[nodiscard]] cudaError_t set_dynamic_shared(Kernel kernel) noexcept {
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87MacroFeedV4Fp8DynamicSharedBytes));
}

template <typename Kernel>
[[nodiscard]] cudaError_t query_kernel_resources(
    Kernel kernel, const Sm87TargetAotProjectionRole role,
    const Sm87MacroFeedV4Fp8InputLayout input_layout,
    const int device_ordinal, const cudaDeviceProp& properties,
    Sm87MacroFeedV4Fp8CudaResources* const resources) noexcept {
  cudaError_t status = set_dynamic_shared(kernel);
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
      &active_blocks, kernel, static_cast<int>(kThreads),
      kSm87MacroFeedV4Fp8DynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  resources->identity = sm87_macrofeed_v4_fp8_identity(role, input_layout);
  resources->role = role;
  resources->input_layout = input_layout;
  resources->device_ordinal = device_ordinal;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  resources->sm_count = properties.multiProcessorCount;
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87MacroFeedV4Fp8DynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->shared_bytes_per_sm = properties.sharedMemPerMultiprocessor;
  resources->optin_shared_bytes_per_block = properties.sharedMemPerBlockOptin;
  resources->kernel_compiled = true;
  resources->static_resource_gate_passed = false;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  return cudaSuccess;
}

[[nodiscard]] cudaError_t query_resources(
    const Sm87TargetAotProjectionRole role,
    const Sm87MacroFeedV4Fp8InputLayout input_layout,
    Sm87MacroFeedV4Fp8CudaResources* const resources) noexcept {
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
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(kSm87MacroFeedV4Fp8SmCount)) {
    return cudaErrorNotSupported;
  }
  if (!sm87_macrofeed_v4_fp8_input_layout(role, input_layout)) {
    return cudaErrorInvalidValue;
  }
  if (role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    return query_kernel_resources(
        sm87_macrofeed_v4_fp8_kernel<
            Sm87TargetAotProjectionRole::kFp8GdnQkvZ, 5'120U, 80U,
            Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1>,
        role, input_layout, device, properties, resources);
  }
  if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    return query_kernel_resources(
        sm87_macrofeed_v4_fp8_kernel<
            Sm87TargetAotProjectionRole::kFp8FullQkv, 5'120U, 80U,
            Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1>,
        role, input_layout, device, properties, resources);
  }
  if (role == Sm87TargetAotProjectionRole::kFp8AttentionOutput) {
    if (input_layout == Sm87MacroFeedV4Fp8InputLayout::
                            kGdnContiguousVScratchV1) {
      return query_kernel_resources(
          sm87_macrofeed_v4_fp8_kernel<
              Sm87TargetAotProjectionRole::kFp8AttentionOutput, 6'144U, 96U,
              Sm87MacroFeedV4Fp8InputLayout::kGdnContiguousVScratchV1>,
          role, input_layout, device, properties, resources);
    }
    return query_kernel_resources(
        sm87_macrofeed_v4_fp8_kernel<
            Sm87TargetAotProjectionRole::kFp8AttentionOutput, 6'144U, 96U,
            Sm87MacroFeedV4Fp8InputLayout::
                kFullAttentionInterleavedQScratchV1>,
        role, input_layout, device, properties, resources);
  }
  return cudaErrorInvalidValue;
}

[[nodiscard]] float bf16_to_float(const std::uint16_t bits) noexcept {
  const std::uint32_t bits32 = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &bits32, sizeof(value));
  return value;
}

[[nodiscard]] bool scale_valid(const float scale) noexcept {
  return std::isfinite(scale) && scale > 0.0F;
}

// Pure fixed-geometry enqueue body shared by the public T1 probe and the
// owner-locked execution seam.  Every device/resource/range observation is a
// construction/admission responsibility; this body only rechecks the fixed
// structural contract, decodes already-authenticated scale bits, and submits
// exactly one GDN-QKVZ kernel.
[[nodiscard]] cudaError_t enqueue_gdn_qkvz_c8000_body(
    const std::uint16_t* const hidden_input,
    const Sm87TargetAotFp8CudaAssetView& asset,
    std::uint16_t* const phase_scratch,
    const cudaStream_t stream) noexcept {
  constexpr auto plan = sm87_macrofeed_v4_fp8_plan(
      Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
      kSm87MacroFeedV4Fp8Tokens,
      Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1);
  static_assert(plan.valid() && plan.input_features == 5'120U &&
                plan.input_row_stride == 5'120U &&
                plan.primary_output_row_stride == 17'408U &&
                plan.key_output_features == 0U &&
                plan.value_output_features == 0U);
  const Sm87MacroFeedV4Fp8Arguments fixed{
      Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
      hidden_input,
      kSm87MacroFeedV4Fp8HiddenRowStride,
      asset,
      kSm87MacroFeedV4Fp8Tokens,
      phase_scratch,
      kSm87MacroFeedV4Fp8ScratchRowStride,
      nullptr,
      0U,
      nullptr,
      0U,
      reinterpret_cast<void*>(stream),
      Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1};
  if (stream == nullptr || !sm87_macrofeed_v4_fp8_arguments_valid(fixed)) {
    return cudaErrorInvalidValue;
  }

  std::array<float, 3U> scales{};
  for (std::size_t index = 0U; index < plan.partition_count; ++index) {
    scales[index] =
        bf16_to_float(asset.compensated_tensor_scale_bf16_bits[index]);
    if (!scale_valid(scales[index])) {
      return cudaErrorInvalidValue;
    }
  }
  const auto* const payload = reinterpret_cast<const std::uint8_t*>(
      asset.payload.begin);
  // Do not clear this host thread's prior CUDA error.  In the composed path a
  // stale failure is evidence, not launch hygiene: cudaPeekAtLastError below
  // must surface it so the EventsOwner poisons and physically drains all
  // streams instead of erasing the first observable cause.
  sm87_macrofeed_v4_fp8_kernel<
      Sm87TargetAotProjectionRole::kFp8GdnQkvZ, 5'120U, 80U,
      Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1>
      <<<static_cast<unsigned int>(plan.logical_tasks),
         static_cast<unsigned int>(kSm87MacroFeedV4Fp8Threads),
         kSm87MacroFeedV4Fp8DynamicSharedBytes, stream>>>(
          hidden_input, kSm87MacroFeedV4Fp8HiddenRowStride, payload,
          scales[0U], scales[1U], scales[2U], phase_scratch,
          kSm87MacroFeedV4Fp8ScratchRowStride, nullptr, 0U, nullptr, 0U,
          static_cast<unsigned int>(plan.grid_n));
  return cudaPeekAtLastError();
}

[[nodiscard]] bool tile_test_arguments_valid(
    const Sm87MacroFeedV4Fp8TileTestArguments& arguments) noexcept {
  const auto input_layout = sm87_macrofeed_v4_fp8_resolve_input_layout(
      arguments.role, arguments.input_layout);
  const auto plan = sm87_macrofeed_v4_fp8_plan(
      arguments.role, kSm87MacroFeedV4Fp8Tokens, input_layout);
  if (!plan.valid() || arguments.partition_index >= plan.partition_count ||
      arguments.partition_n256_tile >=
          plan.partition_features[arguments.partition_index] / 256U ||
      arguments.input == nullptr ||
      arguments.canonical_payload_four_k64_cells == nullptr ||
      arguments.primary_output == nullptr || arguments.cuda_stream == nullptr ||
      arguments.valid_rows == 0U ||
      arguments.valid_rows > kSm87MacroFeedV4Fp8BlockM ||
      arguments.canonical_n128_half > 1U ||
      arguments.logical_input_first_k % kTileK != 0U ||
      arguments.logical_input_first_k +
              kSm87MacroFeedV4Fp8TestInputFeatures >
          plan.input_features ||
      arguments.input_row_stride < kSm87MacroFeedV4Fp8TestInputFeatures ||
      arguments.input_row_stride % 8U != 0U ||
      reinterpret_cast<std::uintptr_t>(arguments.input) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(
          arguments.canonical_payload_four_k64_cells) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(arguments.primary_output) % 16U !=
          0U ||
      arguments.primary_output_row_stride % 8U != 0U) {
    return false;
  }
  const float scale = bf16_to_float(arguments.compensated_scale_bf16_bits);
  if (!scale_valid(scale)) {
    return false;
  }
  if (arguments.role ==
      Sm87TargetAotProjectionRole::kFp8AttentionOutput) {
    if (arguments.input_row_stride !=
        kSm87MacroFeedV4Fp8ScratchRowStride) {
      return false;
    }
    if (input_layout == Sm87MacroFeedV4Fp8InputLayout::
                            kGdnContiguousVScratchV1) {
      if (arguments.logical_input_first_k !=
              kSm87MacroFeedV4Fp8TestGdnAttentionOutputLogicalFirstK &&
          arguments.logical_input_first_k !=
              kSm87MacroFeedV4Fp8TestGdnAttentionOutputLogicalTailFirstK) {
        return false;
      }
    } else if (arguments.logical_input_first_k !=
               kSm87MacroFeedV4Fp8TestAttentionOutputLogicalFirstK) {
      return false;
    }
  } else if (arguments.logical_input_first_k != 0U) {
    return false;
  }
  const std::size_t partition_local_first_n =
      arguments.partition_n256_tile * 256U +
      arguments.canonical_n128_half * kTileN;
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    const std::size_t required =
        arguments.partition_index == 0U
            ? partition_local_first_n + kTileN
            : kSm87MacroFeedV4Fp8GdnZOffset +
                  partition_local_first_n + kTileN;
    return arguments.primary_output_row_stride >= required &&
           arguments.key_output == nullptr && arguments.value_output == nullptr &&
           arguments.key_output_row_stride == 0U &&
           arguments.value_output_row_stride == 0U;
  }
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    const std::size_t primary_first_n =
        arguments.partition_index == 0U
            ? sm87_macrofeed_v4_fp8_interleaved_q_gate_physical_offset(
                  partition_local_first_n)
            : 0U;
    return arguments.primary_output_row_stride >= primary_first_n + kTileN &&
           arguments.key_output != nullptr && arguments.value_output != nullptr &&
           arguments.key_output_row_stride >=
               kSm87MacroFeedV4Fp8KvNhdRowStride &&
           arguments.value_output_row_stride >=
               kSm87MacroFeedV4Fp8KvNhdRowStride &&
           arguments.key_output_row_stride % 8U == 0U &&
           arguments.value_output_row_stride % 8U == 0U &&
           reinterpret_cast<std::uintptr_t>(arguments.key_output) % 16U ==
               0U &&
           reinterpret_cast<std::uintptr_t>(arguments.value_output) % 16U ==
               0U;
  }
  return arguments.partition_index == 0U &&
         arguments.primary_output_row_stride >=
             partition_local_first_n + kTileN &&
         arguments.key_output == nullptr && arguments.value_output == nullptr &&
         arguments.key_output_row_stride == 0U &&
         arguments.value_output_row_stride == 0U;
}

[[nodiscard]] bool cuda_device_range_owned(
    const Sm87MacroFeedV4Fp8ByteRange& range,
    const int expected_device_ordinal,
    std::uintptr_t* const allocation_begin = nullptr,
    std::size_t* const allocation_bytes = nullptr) noexcept {
  if (!range.valid || range.begin == 0U || range.end <= range.begin ||
      expected_device_ordinal < 0) {
    return false;
  }
  cudaPointerAttributes attributes{};
  const auto* const pointer = reinterpret_cast<const void*>(range.begin);
  cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess || attributes.type != cudaMemoryTypeDevice ||
      attributes.device != expected_device_ordinal) {
    return false;
  }
  CUdeviceptr base_pointer = 0U;
  std::size_t bytes = 0U;
  if (cuMemGetAddressRange(&base_pointer, &bytes,
                           static_cast<CUdeviceptr>(range.begin)) !=
          CUDA_SUCCESS ||
      base_pointer == 0U || bytes == 0U) {
    return false;
  }
  const std::uintptr_t begin = static_cast<std::uintptr_t>(base_pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  const std::uintptr_t end = begin + bytes;
  if (range.begin < begin || range.end > end) {
    return false;
  }
  if (allocation_begin != nullptr) {
    *allocation_begin = begin;
  }
  if (allocation_bytes != nullptr) {
    *allocation_bytes = bytes;
  }
  return true;
}

[[nodiscard]] constexpr bool resource_observations_exactly_match(
    const Sm87MacroFeedV4Fp8CudaResources& expected,
    const Sm87MacroFeedV4Fp8CudaResources& observed) noexcept {
  return expected.identity == observed.identity &&
         expected.role == observed.role &&
         expected.input_layout == observed.input_layout &&
         expected.device_ordinal == observed.device_ordinal &&
         expected.compute_major == observed.compute_major &&
         expected.compute_minor == observed.compute_minor &&
         expected.sm_count == observed.sm_count &&
         expected.binary_version == observed.binary_version &&
         expected.registers_per_thread == observed.registers_per_thread &&
         expected.static_shared_bytes == observed.static_shared_bytes &&
         expected.dynamic_shared_bytes == observed.dynamic_shared_bytes &&
         expected.local_bytes == observed.local_bytes &&
         expected.maximum_threads_per_block ==
             observed.maximum_threads_per_block &&
         expected.active_blocks_per_sm == observed.active_blocks_per_sm &&
         expected.shared_bytes_per_sm == observed.shared_bytes_per_sm &&
         expected.optin_shared_bytes_per_block ==
             observed.optin_shared_bytes_per_block &&
         expected.kernel_compiled == observed.kernel_compiled &&
         expected.static_resource_gate_passed ==
             observed.static_resource_gate_passed &&
         expected.numerical_contract_qualified ==
             observed.numerical_contract_qualified &&
         expected.production_dispatch_eligible ==
             observed.production_dispatch_eligible;
}

[[nodiscard]] bool live_t1_device_bindings_valid(
    const Sm87MacroFeedV4Fp8Arguments& arguments,
    const Sm87MacroFeedV4Fp8Plan& plan,
    const Sm87MacroFeedV4Fp8T1AdmissionSnapshot& snapshot,
    int* const current_device_ordinal) noexcept {
  if (current_device_ordinal == nullptr) {
    return false;
  }
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device < 0 ||
      snapshot.resources.device_ordinal != current_device ||
      arguments.asset.device_upload_receipt.device_ordinal !=
          current_device) {
    return false;
  }

  std::array<Sm87MacroFeedV4Fp8ByteRange, 4U> ranges{};
  std::size_t count = 0U;
  ranges[count++] = sm87_macrofeed_v4_fp8_strided_subrange(
      arguments.input, arguments.token_count, arguments.input_row_stride,
      plan.input_physical_offset, plan.input_physical_span);
  ranges[count++] = sm87_macrofeed_v4_fp8_strided_range(
      arguments.primary_output, arguments.token_count,
      arguments.primary_output_row_stride, plan.primary_output_features);
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    ranges[count++] = sm87_macrofeed_v4_fp8_strided_range(
        arguments.key_output, arguments.token_count,
        arguments.key_output_row_stride, plan.key_output_features);
    ranges[count++] = sm87_macrofeed_v4_fp8_strided_range(
        arguments.value_output, arguments.token_count,
        arguments.value_output_row_stride, plan.value_output_features);
  }
  for (std::size_t index = 0U; index < count; ++index) {
    if (!cuda_device_range_owned(ranges[index], current_device)) {
      return false;
    }
  }

  const Sm87MacroFeedV4Fp8ByteRange payload{
      arguments.asset.payload.begin, arguments.asset.payload.end,
      arguments.asset.payload.valid};
  std::uintptr_t observed_allocation_begin = 0U;
  std::size_t observed_allocation_bytes = 0U;
  if (!cuda_device_range_owned(payload, current_device,
                               &observed_allocation_begin,
                               &observed_allocation_bytes)) {
    return false;
  }
  const auto& upload = arguments.asset.device_upload_receipt;
  if (observed_allocation_begin != upload.device_allocation_begin ||
      observed_allocation_bytes != upload.device_allocation_bytes ||
      upload.device_allocation_end < upload.device_allocation_begin ||
      upload.device_allocation_end - upload.device_allocation_begin !=
          upload.device_allocation_bytes) {
    return false;
  }
  *current_device_ordinal = current_device;
  return true;
}

}  // namespace

bool sm87_macrofeed_v4_fp8_arguments_valid(
    const Sm87MacroFeedV4Fp8Arguments& arguments) noexcept {
  const auto input_layout = sm87_macrofeed_v4_fp8_resolve_input_layout(
      arguments.role, arguments.input_layout);
  const auto plan = sm87_macrofeed_v4_fp8_plan(
      arguments.role, arguments.token_count, input_layout);
  const Sm87MacroFeedV4Fp8LayoutBinding binding{
      arguments.role,
      arguments.token_count,
      arguments.input,
      arguments.input_row_stride,
      arguments.primary_output,
      arguments.primary_output_row_stride,
      arguments.key_output,
      arguments.key_output_row_stride,
      arguments.value_output,
      arguments.value_output_row_stride,
      input_layout};
  if (!plan.valid() || arguments.cuda_stream == nullptr ||
      !sm87_macrofeed_v4_fp8_layout_valid(binding) ||
      !sm87_target_aot_fp8_cuda_asset_valid(arguments.asset) ||
      arguments.asset.payload.role != arguments.role ||
      arguments.asset.payload.bytes != plan.payload_bytes) {
    return false;
  }
  const Sm87MacroFeedV4Fp8ByteRange payload{
      arguments.asset.payload.begin, arguments.asset.payload.end,
      arguments.asset.payload.valid};
  std::array<Sm87MacroFeedV4Fp8ByteRange, 4U> ranges{};
  std::size_t count = 0U;
  ranges[count++] = sm87_macrofeed_v4_fp8_strided_subrange(
      arguments.input, arguments.token_count, arguments.input_row_stride,
      plan.input_physical_offset, plan.input_physical_span);
  ranges[count++] = sm87_macrofeed_v4_fp8_strided_range(
      arguments.primary_output, arguments.token_count,
      arguments.primary_output_row_stride, plan.primary_output_features);
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    ranges[count++] = sm87_macrofeed_v4_fp8_strided_range(
        arguments.key_output, arguments.token_count,
        arguments.key_output_row_stride, plan.key_output_features);
    ranges[count++] = sm87_macrofeed_v4_fp8_strided_range(
        arguments.value_output, arguments.token_count,
        arguments.value_output_row_stride, plan.value_output_features);
  }
  if (!payload.valid) {
    return false;
  }
  for (std::size_t index = 0U; index < count; ++index) {
    if (!ranges[index].valid ||
        sm87_macrofeed_v4_fp8_ranges_overlap(ranges[index], payload)) {
      return false;
    }
  }
  return true;
}

int query_sm87_macrofeed_v4_fp8_cuda_resources(
    const Sm87TargetAotProjectionRole role,
    Sm87MacroFeedV4Fp8CudaResources* const resources) noexcept {
  return query_sm87_macrofeed_v4_fp8_cuda_resources(
      role, sm87_macrofeed_v4_fp8_default_input_layout(role), resources);
}

int query_sm87_macrofeed_v4_fp8_cuda_resources(
    const Sm87TargetAotProjectionRole role,
    const Sm87MacroFeedV4Fp8InputLayout input_layout,
    Sm87MacroFeedV4Fp8CudaResources* const resources) noexcept {
  if (resources == nullptr || !sm87_macrofeed_v4_fp8_role(role)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  return static_cast<int>(query_resources(role, input_layout, resources));
}

int capture_sm87_macrofeed_v4_fp8_t1_admission_snapshot_cuda(
    const Sm87TargetAotProjectionRole role,
    Sm87MacroFeedV4Fp8T1AdmissionSnapshot* const snapshot) noexcept {
  return capture_sm87_macrofeed_v4_fp8_t1_admission_snapshot_cuda(
      role, sm87_macrofeed_v4_fp8_default_input_layout(role), snapshot);
}

int capture_sm87_macrofeed_v4_fp8_t1_admission_snapshot_cuda(
    const Sm87TargetAotProjectionRole role,
    const Sm87MacroFeedV4Fp8InputLayout input_layout,
    Sm87MacroFeedV4Fp8T1AdmissionSnapshot* const snapshot) noexcept {
  if (snapshot == nullptr || !sm87_macrofeed_v4_fp8_role(role)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *snapshot = {};
  const int status = query_sm87_macrofeed_v4_fp8_cuda_resources(
      role, input_layout, &snapshot->resources);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  snapshot->resources.static_resource_gate_passed =
      sm87_macrofeed_v4_fp8_resource_gate(snapshot->resources);
  if (!snapshot->resources.static_resource_gate_passed) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  snapshot->dynamic_shared_attribute_observed = true;
  snapshot->resource_query_completed = true;
  snapshot->caller_constructible = true;
  snapshot->startup_package_bound = false;
  snapshot->execution_capability = false;
  snapshot->admission_only = true;
  snapshot->default_off = true;
  snapshot->selector_present = false;
  snapshot->production_dispatch_eligible = false;
  snapshot->snapshot_identity =
      sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
          *snapshot);
  return sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(*snapshot)
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorNotSupported);
}

int launch_sm87_macrofeed_v4_fp8_t1_admission_cuda(
    const Sm87MacroFeedV4Fp8Arguments& arguments,
    const Sm87MacroFeedV4Fp8T1AdmissionSnapshot& snapshot,
    Sm87MacroFeedV4Fp8T1AdmissionLaunchReceipt* const receipt) noexcept {
  if (receipt == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *receipt = {};
  const auto input_layout = sm87_macrofeed_v4_fp8_resolve_input_layout(
      arguments.role, arguments.input_layout);
  const auto plan = sm87_macrofeed_v4_fp8_plan(
      arguments.role, arguments.token_count, input_layout);
  if (!plan.valid() || !sm87_macrofeed_v4_fp8_arguments_valid(arguments) ||
      !sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(snapshot) ||
      snapshot.resources.role != arguments.role ||
      snapshot.resources.input_layout != input_layout ||
      snapshot.resources.identity != plan.identity) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  Sm87MacroFeedV4Fp8CudaResources observed_resources{};
  const int resource_status = query_sm87_macrofeed_v4_fp8_cuda_resources(
      arguments.role, input_layout, &observed_resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }
  observed_resources.static_resource_gate_passed =
      sm87_macrofeed_v4_fp8_resource_gate(observed_resources);
  if (!observed_resources.static_resource_gate_passed ||
      !resource_observations_exactly_match(snapshot.resources,
                                           observed_resources)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int current_device_ordinal = -1;
  if (!live_t1_device_bindings_valid(arguments, plan, snapshot,
                                      &current_device_ordinal)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  std::array<float, 3U> scales{};
  for (std::size_t index = 0U; index < plan.partition_count; ++index) {
    scales[index] = bf16_to_float(
        arguments.asset.compensated_tensor_scale_bf16_bits[index]);
    if (!scale_valid(scales[index])) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  const dim3 grid(static_cast<unsigned int>(plan.logical_tasks));
  const dim3 block(static_cast<unsigned int>(kSm87MacroFeedV4Fp8Threads));
  const auto* const payload = reinterpret_cast<const std::uint8_t*>(
      arguments.asset.payload.begin);
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    const cudaError_t enqueue_status = enqueue_gdn_qkvz_c8000_body(
        arguments.input, arguments.asset, arguments.primary_output, stream);
    if (enqueue_status != cudaSuccess) {
      return static_cast<int>(enqueue_status);
    }
  } else if (arguments.role ==
             Sm87TargetAotProjectionRole::kFp8FullQkv) {
    sm87_macrofeed_v4_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8FullQkv, 5'120U, 80U,
        Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1>
        <<<grid, block, kSm87MacroFeedV4Fp8DynamicSharedBytes, stream>>>(
            arguments.input, arguments.input_row_stride, payload, scales[0U],
            scales[1U], scales[2U], arguments.primary_output,
            arguments.primary_output_row_stride, arguments.key_output,
            arguments.key_output_row_stride, arguments.value_output,
            arguments.value_output_row_stride,
            static_cast<unsigned int>(plan.grid_n));
  } else if (input_layout == Sm87MacroFeedV4Fp8InputLayout::
                                 kGdnContiguousVScratchV1) {
    sm87_macrofeed_v4_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8AttentionOutput, 6'144U, 96U,
        Sm87MacroFeedV4Fp8InputLayout::kGdnContiguousVScratchV1>
        <<<grid, block, kSm87MacroFeedV4Fp8DynamicSharedBytes, stream>>>(
            arguments.input, arguments.input_row_stride, payload, scales[0U],
            scales[1U], scales[2U], arguments.primary_output,
            arguments.primary_output_row_stride, arguments.key_output,
            arguments.key_output_row_stride, arguments.value_output,
            arguments.value_output_row_stride,
            static_cast<unsigned int>(plan.grid_n));
  } else {
    sm87_macrofeed_v4_fp8_kernel<
        Sm87TargetAotProjectionRole::kFp8AttentionOutput, 6'144U, 96U,
        Sm87MacroFeedV4Fp8InputLayout::
            kFullAttentionInterleavedQScratchV1>
        <<<grid, block, kSm87MacroFeedV4Fp8DynamicSharedBytes, stream>>>(
            arguments.input, arguments.input_row_stride, payload, scales[0U],
            scales[1U], scales[2U], arguments.primary_output,
            arguments.primary_output_row_stride, arguments.key_output,
            arguments.key_output_row_stride, arguments.value_output,
            arguments.value_output_row_stride,
            static_cast<unsigned int>(plan.grid_n));
  }
  const cudaError_t launch_status = cudaPeekAtLastError();
  if (launch_status != cudaSuccess) {
    return static_cast<int>(launch_status);
  }
  receipt->identity = plan.identity;
  receipt->role = arguments.role;
  receipt->input_layout = input_layout;
  receipt->artifact_identity = arguments.asset.artifact_identity;
  receipt->device_ordinal = current_device_ordinal;
  receipt->token_count = arguments.token_count;
  receipt->logical_tasks = plan.logical_tasks;
  receipt->physical_kernel_launches = 1U;
  receipt->fallback_launches = 0U;
  receipt->ordinary_full_grid = true;
  receipt->role_specific_direct_scatter = true;
  receipt->private_nhd_kv = plan.private_nhd_kv;
  receipt->authenticated_asset_zero_copy = true;
  receipt->launch_enqueued = true;
  receipt->completion_observed = false;
  receipt->admission_only = true;
  receipt->caller_constructible_snapshot = true;
  receipt->startup_package_bound = false;
  receipt->execution_capability = false;
  receipt->current_device_matches_snapshot = true;
  receipt->asset_upload_device_matches_current = true;
  receipt->live_resource_snapshot_verified = true;
  receipt->caller_stream_non_null = true;
  receipt->stream_owner_verified = false;
  receipt->live_cuda_ranges_verified = true;
  receipt->production_dispatch_eligible = false;
  return static_cast<int>(cudaSuccess);
}

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
namespace sm87_macrofeed_v4_bound_launch_detail {

int enqueue_gdn_qkvz_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4GdnQkvzC8000Arguments& arguments,
    const Sm87MacroFeedV4Fp8CudaResources& resources,
    std::size_t* const submitted_launches) noexcept {
  if (submitted_launches == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *submitted_launches = 0U;
  constexpr auto kRole = Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
  constexpr auto kLayout =
      Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1;
  const Sm87MacroFeedV4Fp8Arguments fixed{
      kRole,
      arguments.hidden_input,
      kSm87MacroFeedV4Fp8HiddenRowStride,
      arguments.asset,
      kSm87MacroFeedV4Fp8Tokens,
      arguments.phase_scratch,
      kSm87MacroFeedV4Fp8ScratchRowStride,
      nullptr,
      0U,
      nullptr,
      0U,
      token.cuda_stream_,
      kLayout};
  if (!sm87_macrofeed_v4_fp8_arguments_valid(fixed) ||
      !resources.static_resource_gate_passed ||
      !sm87_macrofeed_v4_fp8_resource_gate(resources) ||
      resources.role != kRole || resources.input_layout != kLayout ||
      resources.identity != sm87_macrofeed_v4_fp8_identity(kRole, kLayout)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t status = enqueue_gdn_qkvz_c8000_body(
      arguments.hidden_input, arguments.asset, arguments.phase_scratch,
      reinterpret_cast<cudaStream_t>(token.cuda_stream_));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *submitted_launches = 1U;
  return static_cast<int>(cudaSuccess);
}

}  // namespace sm87_macrofeed_v4_bound_launch_detail
#endif

int launch_sm87_macrofeed_v4_fp8_tile_test_cuda(
    const Sm87MacroFeedV4Fp8TileTestArguments& arguments) noexcept {
  if (!tile_test_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const float scale = bf16_to_float(arguments.compensated_scale_bf16_bits);
  const auto input_layout = sm87_macrofeed_v4_fp8_resolve_input_layout(
      arguments.role, arguments.input_layout);
  cudaError_t status = cudaSuccess;
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    status = set_dynamic_shared(sm87_macrofeed_v4_fp8_tile_test_kernel<
        Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
        Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1>);
  } else if (arguments.role ==
             Sm87TargetAotProjectionRole::kFp8FullQkv) {
    status = set_dynamic_shared(sm87_macrofeed_v4_fp8_tile_test_kernel<
        Sm87TargetAotProjectionRole::kFp8FullQkv,
        Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1>);
  } else if (input_layout == Sm87MacroFeedV4Fp8InputLayout::
                                 kGdnContiguousVScratchV1) {
    status = set_dynamic_shared(sm87_macrofeed_v4_fp8_tile_test_kernel<
        Sm87TargetAotProjectionRole::kFp8AttentionOutput,
        Sm87MacroFeedV4Fp8InputLayout::kGdnContiguousVScratchV1>);
  } else {
    status = set_dynamic_shared(sm87_macrofeed_v4_fp8_tile_test_kernel<
        Sm87TargetAotProjectionRole::kFp8AttentionOutput,
        Sm87MacroFeedV4Fp8InputLayout::
            kFullAttentionInterleavedQScratchV1>);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    sm87_macrofeed_v4_fp8_tile_test_kernel<
        Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
        Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1>
        <<<1U, kThreads, kSm87MacroFeedV4Fp8DynamicSharedBytes, stream>>>(
            arguments.input, arguments.input_row_stride,
            arguments.canonical_payload_four_k64_cells, scale,
            static_cast<unsigned int>(arguments.valid_rows),
            static_cast<unsigned int>(arguments.partition_index),
            static_cast<unsigned int>(arguments.partition_n256_tile),
            static_cast<unsigned int>(arguments.canonical_n128_half),
            static_cast<unsigned int>(arguments.logical_input_first_k),
            arguments.primary_output, arguments.primary_output_row_stride,
            arguments.key_output, arguments.key_output_row_stride,
            arguments.value_output, arguments.value_output_row_stride);
  } else if (arguments.role ==
             Sm87TargetAotProjectionRole::kFp8FullQkv) {
    sm87_macrofeed_v4_fp8_tile_test_kernel<
        Sm87TargetAotProjectionRole::kFp8FullQkv,
        Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1>
        <<<1U, kThreads, kSm87MacroFeedV4Fp8DynamicSharedBytes, stream>>>(
            arguments.input, arguments.input_row_stride,
            arguments.canonical_payload_four_k64_cells, scale,
            static_cast<unsigned int>(arguments.valid_rows),
            static_cast<unsigned int>(arguments.partition_index),
            static_cast<unsigned int>(arguments.partition_n256_tile),
            static_cast<unsigned int>(arguments.canonical_n128_half),
            static_cast<unsigned int>(arguments.logical_input_first_k),
            arguments.primary_output, arguments.primary_output_row_stride,
            arguments.key_output, arguments.key_output_row_stride,
            arguments.value_output, arguments.value_output_row_stride);
  } else if (input_layout == Sm87MacroFeedV4Fp8InputLayout::
                                 kGdnContiguousVScratchV1) {
    sm87_macrofeed_v4_fp8_tile_test_kernel<
        Sm87TargetAotProjectionRole::kFp8AttentionOutput,
        Sm87MacroFeedV4Fp8InputLayout::kGdnContiguousVScratchV1>
        <<<1U, kThreads, kSm87MacroFeedV4Fp8DynamicSharedBytes, stream>>>(
            arguments.input, arguments.input_row_stride,
            arguments.canonical_payload_four_k64_cells, scale,
            static_cast<unsigned int>(arguments.valid_rows),
            static_cast<unsigned int>(arguments.partition_index),
            static_cast<unsigned int>(arguments.partition_n256_tile),
            static_cast<unsigned int>(arguments.canonical_n128_half),
            static_cast<unsigned int>(arguments.logical_input_first_k),
            arguments.primary_output, arguments.primary_output_row_stride,
            arguments.key_output, arguments.key_output_row_stride,
            arguments.value_output, arguments.value_output_row_stride);
  } else {
    sm87_macrofeed_v4_fp8_tile_test_kernel<
        Sm87TargetAotProjectionRole::kFp8AttentionOutput,
        Sm87MacroFeedV4Fp8InputLayout::
            kFullAttentionInterleavedQScratchV1>
        <<<1U, kThreads, kSm87MacroFeedV4Fp8DynamicSharedBytes, stream>>>(
            arguments.input, arguments.input_row_stride,
            arguments.canonical_payload_four_k64_cells, scale,
            static_cast<unsigned int>(arguments.valid_rows),
            static_cast<unsigned int>(arguments.partition_index),
            static_cast<unsigned int>(arguments.partition_n256_tile),
            static_cast<unsigned int>(arguments.canonical_n128_half),
            static_cast<unsigned int>(arguments.logical_input_first_k),
            arguments.primary_output, arguments.primary_output_row_stride,
            arguments.key_output, arguments.key_output_row_stride,
            arguments.value_output, arguments.value_output_row_stride);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
