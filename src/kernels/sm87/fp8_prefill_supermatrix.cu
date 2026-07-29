#include "q3x/kernels/sm87_fp8_prefill_supermatrix.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kWarpsPerBlock = 8U;
constexpr unsigned int kThreads = kWarpSize * kWarpsPerBlock;
constexpr unsigned int kTokenCount = 512U;
constexpr unsigned int kResidentTokenCount = 64U;
constexpr unsigned int kTokenTiles = kTokenCount / kResidentTokenCount;
constexpr unsigned int kOutputColumnsPerBlock = 256U;
constexpr unsigned int kColumnsPerStage = 64U;
constexpr unsigned int kK16Groups = 4U;
constexpr unsigned int kN8PanelsPerWarp = 4U;
constexpr unsigned int kPipelineStages = 4U;
constexpr unsigned int kSharedLeadingDimension = 72U;
constexpr unsigned int kPersistentBlocks = 16U;
constexpr unsigned int kSidecarWordsPerStage =
    kK16Groups * kN8PanelsPerWarp * kThreads;
constexpr unsigned int kSidecarVectorsPerStage =
    kSidecarWordsPerStage * sizeof(std::uint32_t) / sizeof(uint4);
constexpr unsigned int kActivationVectorsPerStage =
    kResidentTokenCount * kSharedLeadingDimension *
    sizeof(std::uint16_t) / sizeof(uint4);
constexpr std::size_t kDynamicSharedBytes =
    kPipelineStages *
    (kActivationVectorsPerStage + kSidecarVectorsPerStage) * sizeof(uint4);

static_assert(kTokenTiles == 8U);
static_assert(kSidecarWordsPerStage == 4'096U);
static_assert(kSidecarVectorsPerStage == 1'024U);
static_assert(kActivationVectorsPerStage == 576U);
static_assert(kDynamicSharedBytes == 102'400U);

struct alignas(32) Fp8SupermatrixPipelineStorage {
  uint4 activations[kPipelineStages][kActivationVectorsPerStage];
  uint4 weights[kPipelineStages][kSidecarVectorsPerStage];
};

static_assert(sizeof(Fp8SupermatrixPipelineStorage) ==
              kDynamicSharedBytes);

struct InlineM16N8Accumulator {
  float x0;
  float x1;
  float x2;
  float x3;
};

struct DevicePartition {
  const uint4* sidecar;
  std::uint16_t* output;
  float weight_scale;
  unsigned int rows;
  unsigned int nblocks;
};

[[nodiscard]] bool byte_range_overflows(const void* const pointer,
                                        const std::size_t bytes) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return pointer == nullptr ||
         bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

[[nodiscard]] bool byte_ranges_overlap(const void* const first,
                                       const std::size_t first_bytes,
                                       const void* const second,
                                       const std::size_t second_bytes) noexcept {
  if (byte_range_overflows(first, first_bytes) ||
      byte_range_overflows(second, second_bytes)) {
    return true;
  }
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  return first_begin < second_begin + second_bytes &&
         second_begin < first_begin + first_bytes;
}

[[nodiscard]] bool pointer_is_aligned(const void* const pointer,
                                      const std::size_t alignment) noexcept {
  return pointer != nullptr && alignment != 0U &&
         (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
}

[[nodiscard]] int invalid_value() noexcept {
  return static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t
decode_e4m3fn_to_biased_bf16_bits(const std::uint8_t code) {
  const std::uint16_t sign =
      static_cast<std::uint16_t>(code & 0x80U) << 8U;
  const std::uint16_t magnitude =
      static_cast<std::uint16_t>(code & 0x7fU);
  return magnitude == 0x7fU
             ? static_cast<std::uint16_t>(sign | 0x7fc0U)
             : static_cast<std::uint16_t>(sign | (magnitude << 4U));
}

__device__ __forceinline__ void cp_async_cg_shared_global_16(
    void* const shared_destination, const void* const global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :
               : "r"(shared_address), "l"(global_source));
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_0() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_1() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_2() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 2;" ::: "memory");
#endif
}

[[nodiscard]] __device__ __forceinline__ uint2
decode_fp8x4_to_biased_bf16x4(const std::uint32_t packed) {
  // The sidecar byte order is [v0,v2,v1,v3]. The two mask/shift operations
  // therefore produce exactly the two m16n8 B registers [v0,v1] and [v2,v3]
  // without a PRMT transpose. Finite E4M3FN values are represented as BF16
  // value/2^120; the common 2^120 factor is restored once in the output
  // scale. This is exact for normal, subnormal, signed-zero, and sign bits.
  constexpr std::uint32_t kMagnitudeMask = 0x7f00'7f00U;
  const std::uint32_t odd =
      (packed & 0x8000'8000U) | ((packed & kMagnitudeMask) >> 4U);
  const std::uint32_t shifted = packed << 8U;
  const std::uint32_t even =
      (shifted & 0x8000'8000U) |
      ((shifted & kMagnitudeMask) >> 4U);

  // E4M3FN has one NaN magnitude. The authenticated checkpoint contains no
  // such code, but retain defined synthetic correctness without burdening
  // the real common path with four scalar decoders.
  const std::uint32_t magnitudes = packed & 0x7f7f'7f7fU;
  const std::uint32_t nan_candidates = magnitudes ^ 0x7f7f'7f7fU;
  const std::uint32_t has_nan =
      (nan_candidates - 0x0101'0101U) & ~nan_candidates & 0x8080'8080U;
  if (has_nan == 0U) {
    return uint2{even, odd};
  }

  const std::uint16_t value0 =
      decode_e4m3fn_to_biased_bf16_bits(
          static_cast<std::uint8_t>(packed));
  const std::uint16_t value2 =
      decode_e4m3fn_to_biased_bf16_bits(
          static_cast<std::uint8_t>(packed >> 8U));
  const std::uint16_t value1 =
      decode_e4m3fn_to_biased_bf16_bits(
          static_cast<std::uint8_t>(packed >> 16U));
  const std::uint16_t value3 =
      decode_e4m3fn_to_biased_bf16_bits(
          static_cast<std::uint8_t>(packed >> 24U));
  return uint2{static_cast<std::uint32_t>(value0) |
                   (static_cast<std::uint32_t>(value1) << 16U),
               static_cast<std::uint32_t>(value2) |
                   (static_cast<std::uint32_t>(value3) << 16U)};
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t
encode_bf16_rne(const float value) {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
pack_bf16_fragment_pair(const __nv_bfloat16 low,
                        const __nv_bfloat16 high) {
  return static_cast<std::uint32_t>(__bfloat16_as_ushort(low)) |
         (static_cast<std::uint32_t>(__bfloat16_as_ushort(high)) << 16U);
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    InlineM16N8Accumulator& accumulator, const std::uint32_t a0,
    const std::uint32_t a1, const std::uint32_t a2,
    const std::uint32_t a3, const std::uint32_t b0,
    const std::uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
pack_scaled_bf16_output_pair(const float low, const float high,
                             const float scale) {
  return static_cast<std::uint32_t>(encode_bf16_rne(low * scale)) |
         (static_cast<std::uint32_t>(encode_bf16_rne(high * scale)) << 16U);
}

template <unsigned int kColumns>
__device__ __forceinline__ void issue_pipeline_stage(
    Fp8SupermatrixPipelineStorage* const pipeline,
    const unsigned int shared_slot, const uint4* const sidecar_stage,
    const std::uint16_t* const tile_activations,
    const unsigned int first_k) {
  constexpr unsigned int kActivationChunksPerToken =
      kColumnsPerStage * sizeof(std::uint16_t) / sizeof(uint4);
  constexpr unsigned int kSharedActivationChunksPerToken =
      kSharedLeadingDimension * sizeof(std::uint16_t) / sizeof(uint4);
  constexpr unsigned int kActivationChunkCount =
      kResidentTokenCount * kActivationChunksPerToken;
  static_assert(kActivationChunksPerToken == 8U);
  static_assert(kSharedActivationChunksPerToken == 9U);
  static_assert(kActivationChunkCount == 2U * kThreads);

#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int index = threadIdx.x + pass * kThreads;
    const unsigned int token = index / kActivationChunksPerToken;
    const unsigned int chunk = index % kActivationChunksPerToken;
    cp_async_cg_shared_global_16(
        pipeline->activations[shared_slot] +
            token * kSharedActivationChunksPerToken + chunk,
        reinterpret_cast<const uint4*>(
            tile_activations + static_cast<std::size_t>(token) * kColumns +
            first_k) +
            chunk);
  }
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int index = threadIdx.x + pass * kThreads;
    cp_async_cg_shared_global_16(
        pipeline->weights[shared_slot] + index,
        sidecar_stage + index);
  }
  cp_async_commit_group();
}

__global__ __launch_bounds__(kThreads) void fp8_supermatrix_pack_kernel(
    const std::uint8_t* const canonical_weights,
    std::uint32_t* const sidecar_weights, const unsigned int rows,
    const unsigned int columns, const unsigned int kstage_count) {
  const unsigned int tile = blockIdx.x;
  const unsigned int nblock = tile / kstage_count;
  const unsigned int kstage = tile % kstage_count;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  std::uint32_t* const sidecar_stage =
      sidecar_weights +
      static_cast<std::size_t>(tile) * kSidecarWordsPerStage;

#pragma unroll
  for (unsigned int k16 = 0U; k16 < kK16Groups; ++k16) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp;
         ++n_panel) {
      const unsigned int row =
          nblock * kOutputColumnsPerBlock + warp * 32U +
          n_panel * 8U + lane_group;
      const unsigned int column =
          kstage * kColumnsPerStage + k16 * 16U +
          2U * lane_in_group;
      const std::uint8_t* const source =
          canonical_weights + static_cast<std::size_t>(row) * columns +
          column;
      const std::uint32_t packed =
          static_cast<std::uint32_t>(source[0]) |
          (static_cast<std::uint32_t>(source[8]) << 8U) |
          (static_cast<std::uint32_t>(source[1]) << 16U) |
          (static_cast<std::uint32_t>(source[9]) << 24U);
      sidecar_stage[(k16 * kN8PanelsPerWarp + n_panel) * kThreads +
                    thread] = packed;
    }
  }
  (void)rows;
}

template <unsigned int kColumns, unsigned int kPartitionCount>
__global__ __launch_bounds__(kThreads, 1) void
fp8_prefill_supermatrix_m64_n256_k64_kernel(
    const DevicePartition partition0, const DevicePartition partition1,
    const DevicePartition partition2,
    const std::uint16_t* const activations,
    const unsigned int total_nblocks) {
  constexpr unsigned int kKStageCount = kColumns / kColumnsPerStage;
  static_assert(kColumns == 5'120U || kColumns == 6'144U);
  static_assert(kKStageCount == 80U || kKStageCount == 96U);
  static_assert(kPartitionCount >= 1U && kPartitionCount <= 3U);

  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const pipeline =
      reinterpret_cast<Fp8SupermatrixPipelineStorage*>(dynamic_storage);
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  const unsigned int token_tile = blockIdx.x % kTokenTiles;
  const unsigned int nblock_phase = blockIdx.x / kTokenTiles;
  const std::size_t first_token =
      static_cast<std::size_t>(token_tile) * kResidentTokenCount;
  const std::uint16_t* const tile_activations =
      activations + first_token * kColumns;

  for (unsigned int global_nblock = nblock_phase;
       global_nblock < total_nblocks; global_nblock += 2U) {
    DevicePartition selected = partition0;
    unsigned int local_nblock = global_nblock;
    if constexpr (kPartitionCount >= 2U) {
      if (local_nblock >= partition0.nblocks) {
        local_nblock -= partition0.nblocks;
        selected = partition1;
        if constexpr (kPartitionCount == 3U) {
          if (local_nblock >= partition1.nblocks) {
            local_nblock -= partition1.nblocks;
            selected = partition2;
          }
        }
      }
    }

    const uint4* const sidecar_tile =
        selected.sidecar +
        static_cast<std::size_t>(local_nblock) * kKStageCount *
            kSidecarVectorsPerStage;
    std::uint16_t* const tile_output =
        selected.output + first_token * selected.rows;
    const unsigned int first_output_column =
        local_nblock * kOutputColumnsPerBlock;

    InlineM16N8Accumulator accumulators[4][4];
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < 4U; ++m_panel) {
#pragma unroll
      for (unsigned int n_panel = 0U; n_panel < 4U; ++n_panel) {
        accumulators[m_panel][n_panel] =
            InlineM16N8Accumulator{0.0F, 0.0F, 0.0F, 0.0F};
      }
    }

    issue_pipeline_stage<kColumns>(pipeline, 0U, sidecar_tile,
                                   tile_activations, 0U);
    issue_pipeline_stage<kColumns>(
        pipeline, 1U, sidecar_tile + kSidecarVectorsPerStage,
        tile_activations, kColumnsPerStage);
    issue_pipeline_stage<kColumns>(
        pipeline, 2U, sidecar_tile + 2U * kSidecarVectorsPerStage,
        tile_activations, 2U * kColumnsPerStage);

#pragma unroll 1
    for (unsigned int stage = 0U; stage < kKStageCount; ++stage) {
      if (stage + 2U < kKStageCount) {
        cp_async_wait_group_2();
      } else if (stage + 1U < kKStageCount) {
        cp_async_wait_group_1();
      } else {
        cp_async_wait_group_0();
      }
      __syncthreads();

      if (stage + 3U < kKStageCount) {
        const unsigned int future_stage = stage + 3U;
        issue_pipeline_stage<kColumns>(
            pipeline, future_stage % kPipelineStages,
            sidecar_tile +
                static_cast<std::size_t>(future_stage) *
                    kSidecarVectorsPerStage,
            tile_activations, future_stage * kColumnsPerStage);
      }

      const unsigned int shared_slot = stage % kPipelineStages;
      const auto* const shared_a =
          reinterpret_cast<const __nv_bfloat16*>(
              pipeline->activations[shared_slot]);
      const auto* const shared_b =
          reinterpret_cast<const std::uint32_t*>(
              pipeline->weights[shared_slot]);

#pragma unroll
      for (unsigned int k16 = 0U; k16 < kK16Groups; ++k16) {
        uint2 decoded_b[4];
#pragma unroll
        for (unsigned int n_panel = 0U; n_panel < 4U; ++n_panel) {
          const std::uint32_t packed =
              shared_b[(k16 * kN8PanelsPerWarp + n_panel) * kThreads +
                       thread];
          decoded_b[n_panel] = decode_fp8x4_to_biased_bf16x4(packed);
        }

#pragma unroll
        for (unsigned int m_panel = 0U; m_panel < 4U; ++m_panel) {
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                 __nv_bfloat16,
                                 nvcuda::wmma::row_major>
              activation_fragment;
          nvcuda::wmma::load_matrix_sync(
              activation_fragment,
              shared_a + m_panel * 16U * kSharedLeadingDimension +
                  k16 * 16U,
              kSharedLeadingDimension);
          const std::uint32_t a0 = pack_bf16_fragment_pair(
              activation_fragment.x[0], activation_fragment.x[1]);
          const std::uint32_t a1 = pack_bf16_fragment_pair(
              activation_fragment.x[2], activation_fragment.x[3]);
          const std::uint32_t a2 = pack_bf16_fragment_pair(
              activation_fragment.x[4], activation_fragment.x[5]);
          const std::uint32_t a3 = pack_bf16_fragment_pair(
              activation_fragment.x[6], activation_fragment.x[7]);
#pragma unroll
          for (unsigned int n_panel = 0U; n_panel < 4U; ++n_panel) {
            mma_m16n8k16_bf16(
                accumulators[m_panel][n_panel], a0, a1, a2, a3,
                decoded_b[n_panel].x, decoded_b[n_panel].y);
          }
        }
      }
    }
    cp_async_wait_group_0();
    __syncthreads();

    constexpr float kFp8ToBiasedBf16Scale = 0x1p120F;
    const float output_scale =
        selected.weight_scale * kFp8ToBiasedBf16Scale;
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < 4U; ++m_panel) {
#pragma unroll
      for (unsigned int n_panel = 0U; n_panel < 4U; ++n_panel) {
        const InlineM16N8Accumulator accumulator =
            accumulators[m_panel][n_panel];
        const unsigned int token0 = m_panel * 16U + lane_group;
        const unsigned int token1 = token0 + 8U;
        const unsigned int local_column =
            warp * 32U + n_panel * 8U + 2U * lane_in_group;
        *reinterpret_cast<std::uint32_t*>(
            tile_output + static_cast<std::size_t>(token0) * selected.rows +
            first_output_column + local_column) =
            pack_scaled_bf16_output_pair(
                accumulator.x0, accumulator.x1, output_scale);
        *reinterpret_cast<std::uint32_t*>(
            tile_output + static_cast<std::size_t>(token1) * selected.rows +
            first_output_column + local_column) =
            pack_scaled_bf16_output_pair(
                accumulator.x2, accumulator.x3, output_scale);
      }
    }
    __syncthreads();
  }
}

[[nodiscard]] bool has_supported_topology(
    const Sm87Fp8PrefillSupermatrixPartition* const partitions,
    const std::size_t partition_count,
    const std::size_t columns) noexcept {
  if (columns == 5'120U && partition_count == 2U) {
    return partitions[0].rows == 10'240U &&
           partitions[1].rows == 6'144U;
  }
  if (columns == 5'120U && partition_count == 3U) {
    return partitions[0].rows == 12'288U &&
           partitions[1].rows == 1'024U &&
           partitions[2].rows == 1'024U;
  }
  return columns == 6'144U && partition_count == 1U &&
         partitions[0].rows == 5'120U;
}

template <unsigned int kColumns, unsigned int kPartitionCount>
[[nodiscard]] int launch_supermatrix_kernel(
    const DevicePartition& partition0,
    const DevicePartition& partition1,
    const DevicePartition& partition2,
    const std::uint16_t* const activations,
    const unsigned int total_nblocks,
    const cudaStream_t stream) noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      fp8_prefill_supermatrix_m64_n256_k64_kernel<kColumns,
                                                  kPartitionCount>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDynamicSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  fp8_prefill_supermatrix_m64_n256_k64_kernel<kColumns, kPartitionCount>
      <<<kPersistentBlocks, kThreads, kDynamicSharedBytes, stream>>>(
          partition0, partition1, partition2, activations, total_nblocks);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

int launch_sm87_fp8_prefill_supermatrix_pack_cuda(
    const std::uint8_t* const canonical_weights,
    std::uint8_t* const sidecar_weights, const std::size_t rows,
    const std::size_t columns, void* const cuda_stream) noexcept {
  if (canonical_weights == nullptr || sidecar_weights == nullptr ||
      rows == 0U || (rows % kOutputColumnsPerBlock) != 0U ||
      (columns != 5'120U && columns != 6'144U) ||
      !pointer_is_aligned(canonical_weights, alignof(uint4)) ||
      !pointer_is_aligned(sidecar_weights, alignof(uint4)) ||
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    return invalid_value();
  }
  const std::size_t bytes = rows * columns;
  if (byte_ranges_overlap(canonical_weights, bytes, sidecar_weights,
                          bytes)) {
    return invalid_value();
  }
  const std::size_t nblocks = rows / kOutputColumnsPerBlock;
  const std::size_t kstage_count = columns / kColumnsPerStage;
  if (nblocks > std::numeric_limits<unsigned int>::max() / kstage_count) {
    return invalid_value();
  }
  const unsigned int blocks =
      static_cast<unsigned int>(nblocks * kstage_count);
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_supermatrix_pack_kernel<<<blocks, kThreads, 0U, stream>>>(
      canonical_weights, reinterpret_cast<std::uint32_t*>(sidecar_weights),
      static_cast<unsigned int>(rows), static_cast<unsigned int>(columns),
      static_cast<unsigned int>(kstage_count));
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
    const Sm87Fp8PrefillSupermatrixPartition* const partitions,
    const std::size_t partition_count,
    const std::uint16_t* const activations,
    const std::size_t token_count, const std::size_t columns,
    void* const cuda_stream) noexcept {
  if (partitions == nullptr || partition_count == 0U ||
      partition_count > 3U || token_count != kTokenCount ||
      !has_supported_topology(partitions, partition_count, columns) ||
      !pointer_is_aligned(activations, alignof(uint4))) {
    return invalid_value();
  }
  const std::size_t activation_bytes =
      token_count * columns * sizeof(std::uint16_t);
  DevicePartition device_partitions[3]{};
  unsigned int total_nblocks = 0U;
  for (std::size_t index = 0U; index < partition_count; ++index) {
    const Sm87Fp8PrefillSupermatrixPartition& partition =
        partitions[index];
    if (partition.register_feed_sidecar == nullptr ||
        partition.output == nullptr ||
        !std::isfinite(partition.weight_scale) ||
        partition.weight_scale < 0.0F || partition.rows == 0U ||
        (partition.rows % kOutputColumnsPerBlock) != 0U ||
        !pointer_is_aligned(partition.register_feed_sidecar,
                            alignof(uint4)) ||
        !pointer_is_aligned(partition.output,
                            alignof(std::uint32_t)) ||
        partition.rows >
            std::numeric_limits<std::size_t>::max() / columns ||
        partition.rows >
            std::numeric_limits<std::size_t>::max() /
                (token_count * sizeof(std::uint16_t))) {
      return invalid_value();
    }
    const std::size_t sidecar_bytes = partition.rows * columns;
    const std::size_t output_bytes =
        token_count * partition.rows * sizeof(std::uint16_t);
    if (byte_ranges_overlap(partition.register_feed_sidecar, sidecar_bytes,
                            activations, activation_bytes) ||
        byte_ranges_overlap(partition.register_feed_sidecar, sidecar_bytes,
                            partition.output, output_bytes) ||
        byte_ranges_overlap(activations, activation_bytes,
                            partition.output, output_bytes)) {
      return invalid_value();
    }
    for (std::size_t prior = 0U; prior < index; ++prior) {
      const std::size_t prior_sidecar_bytes =
          partitions[prior].rows * columns;
      const std::size_t prior_output_bytes =
          token_count * partitions[prior].rows * sizeof(std::uint16_t);
      if (byte_ranges_overlap(partition.register_feed_sidecar,
                              sidecar_bytes,
                              partitions[prior].register_feed_sidecar,
                              prior_sidecar_bytes) ||
          byte_ranges_overlap(partition.output, output_bytes,
                              partitions[prior].output,
                              prior_output_bytes)) {
        return invalid_value();
      }
    }
    const std::size_t nblocks =
        partition.rows / kOutputColumnsPerBlock;
    if (nblocks > std::numeric_limits<unsigned int>::max() -
                      total_nblocks) {
      return invalid_value();
    }
    device_partitions[index] =
        DevicePartition{reinterpret_cast<const uint4*>(
                            partition.register_feed_sidecar),
                        partition.output, partition.weight_scale,
                        static_cast<unsigned int>(partition.rows),
                        static_cast<unsigned int>(nblocks)};
    total_nblocks += static_cast<unsigned int>(nblocks);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  if (columns == 5'120U && partition_count == 2U) {
    return launch_supermatrix_kernel<5'120U, 2U>(
        device_partitions[0], device_partitions[1], device_partitions[2],
        activations, total_nblocks, stream);
  }
  if (columns == 5'120U && partition_count == 3U) {
    return launch_supermatrix_kernel<5'120U, 3U>(
        device_partitions[0], device_partitions[1], device_partitions[2],
        activations, total_nblocks, stream);
  }
  return launch_supermatrix_kernel<6'144U, 1U>(
      device_partitions[0], device_partitions[1], device_partitions[2],
      activations, total_nblocks, stream);
}

}  // namespace q3x::kernels
