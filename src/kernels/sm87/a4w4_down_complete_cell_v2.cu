#include "q3x/kernels/sm87_a4w4_down_complete_cell_v2.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kPackedK64Bytes = 32U;
inline constexpr unsigned int kPhysicalK64PerK128 = 2U;
inline constexpr unsigned int kCodePlaneBytes =
    kSm87A4W4DownCellV2TileM * kPackedK64Bytes;

struct alignas(16) Sm87A4W4DownCellV2AStage final {
  std::uint8_t code[kPhysicalK64PerK128][kCodePlaneBytes];
  std::uint16_t scale[kSm87A4W4DownCellV2TileM];
};

struct alignas(16) Sm87A4W4DownCellV2BStage final {
  std::uint8_t code[kPhysicalK64PerK128][kCodePlaneBytes];
  std::uint16_t scale[kSm87A4W4DownCellV2TileN];
};

struct alignas(16) Sm87A4W4DownCellV2Shared final {
  Sm87A4W4DownCellV2AStage a[kSm87A4W4DownCellV2AStages];
  Sm87A4W4DownCellV2BStage b[kSm87A4W4DownCellV2BStages];
};

static_assert(sizeof(Sm87A4W4DownCellV2AStage) ==
              kSm87A4W4DownCellV2StageBytes);
static_assert(sizeof(Sm87A4W4DownCellV2BStage) ==
              kSm87A4W4DownCellV2StageBytes);
static_assert(sizeof(Sm87A4W4DownCellV2Shared) ==
              kSm87A4W4DownCellV2SharedBytes);
static_assert(sizeof(Sm87A4W4DownCellV2Shared) *
                  kSm87A4W4DownCellV2CtasPerSm <=
              96U * 1'024U);

[[nodiscard]] constexpr bool aligned(const void* const pointer,
                                     const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] constexpr bool product_fits(const std::size_t first,
                                          const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr bool consumer_capacity_fits(
    const std::size_t outer_count,
    const std::size_t physical_k64_group_count) noexcept {
  const std::size_t outer_blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return outer_blocks != 0U && physical_k64_group_count != 0U &&
         product_fits(outer_blocks, physical_k64_group_count) &&
         product_fits(outer_blocks * physical_k64_group_count,
                      kSm87A4W4ConsumerOuterBlock) &&
         product_fits(outer_blocks * physical_k64_group_count *
                          kSm87A4W4ConsumerOuterBlock,
                      kSm87A4W4ConsumerPackedKBlockBytes);
}

[[nodiscard]] constexpr bool shared_scale_capacity_fits(
    const std::size_t outer_count,
    const std::size_t k128_group_count) noexcept {
  const std::size_t outer_blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return outer_blocks != 0U && k128_group_count != 0U &&
         product_fits(outer_blocks, k128_group_count) &&
         product_fits(outer_blocks * k128_group_count,
                      kSm87A4W4ConsumerOuterBlock);
}

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ void cp_async_16(
    void* const destination, const void* const source) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(source)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cp_async_commit() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" : : : "memory");
#else
  asm volatile("trap;");
#endif
}

template <unsigned int Remaining>
__device__ __forceinline__ void cp_async_wait() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;"
               :
               : "n"(Remaining)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void issue_a_stage(
    Sm87A4W4DownCellV2AStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  constexpr unsigned int kVectorsPerPhysicalK64 = kCodePlaneBytes / 16U;
  constexpr unsigned int kCodeVectors =
      kPhysicalK64PerK128 * kVectorsPerPhysicalK64;
  constexpr unsigned int kScaleVectors =
      kSm87A4W4DownCellV2TileM * sizeof(std::uint16_t) / 16U;
  static_assert(kCodeVectors == 2U * kSm87A4W4DownCellV2Threads);
  static_assert(kScaleVectors == 16U);

  for (unsigned int vector = threadIdx.x; vector < kCodeVectors;
       vector += blockDim.x) {
    const unsigned int half = vector / kVectorsPerPhysicalK64;
    const unsigned int half_vector =
        vector - half * kVectorsPerPhysicalK64;
    const unsigned int row = half_vector / 2U;
    const unsigned int row_vector = half_vector % 2U;
    const unsigned int physical_group = 2U * k128_group + half;
    cp_async_16(
        stage.code[half] + sm87_a4w4_swizzled_k64_byte_offset(
                               row, 16U * row_vector),
        packed_a + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_group, 16U * row_vector,
                       physical_k64_group_count));
  }
  if (threadIdx.x < kScaleVectors) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage.scale + first_row,
        a_k128_scales_bf16 + sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(m_tile_start) + first_row,
            k128_group, k128_group_count));
  }
}

__device__ __forceinline__ void issue_b_stage(
    Sm87A4W4DownCellV2BStage& stage,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k128_scales_bf16,
    const unsigned int n_tile_start,
    const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  constexpr unsigned int kVectorsPerPhysicalK64 = kCodePlaneBytes / 16U;
  constexpr unsigned int kCodeVectors =
      kPhysicalK64PerK128 * kVectorsPerPhysicalK64;
  constexpr unsigned int kScaleVectors =
      kSm87A4W4DownCellV2TileN * sizeof(std::uint16_t) / 16U;
  static_assert(kCodeVectors == 2U * kSm87A4W4DownCellV2Threads);
  static_assert(kScaleVectors == 16U);

  for (unsigned int vector = threadIdx.x; vector < kCodeVectors;
       vector += blockDim.x) {
    const unsigned int half = vector / kVectorsPerPhysicalK64;
    const unsigned int half_vector =
        vector - half * kVectorsPerPhysicalK64;
    const unsigned int row = half_vector / 2U;
    const unsigned int row_vector = half_vector % 2U;
    const unsigned int physical_group = 2U * k128_group + half;
    cp_async_16(
        stage.code[half] + sm87_a4w4_swizzled_k64_byte_offset(
                               row, 16U * row_vector),
        packed_b + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(n_tile_start) + row,
                       physical_group, 16U * row_vector,
                       physical_k64_group_count));
  }
  if (threadIdx.x < kScaleVectors) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage.scale + first_row,
        b_k128_scales_bf16 + sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(n_tile_start) + first_row,
            k128_group, k128_group_count));
  }
}

__device__ __forceinline__ void issue_pair(
    Sm87A4W4DownCellV2Shared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k128_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  issue_a_stage(shared.a[group % kSm87A4W4DownCellV2AStages], packed_a,
                a_k128_scales_bf16, m_tile_start, group,
                physical_k64_group_count, k128_group_count);
  issue_b_stage(shared.b[group % kSm87A4W4DownCellV2BStages], packed_b,
                b_k128_scales_bf16, n_tile_start, group,
                physical_k64_group_count, k128_group_count);
  cp_async_commit();
}

__device__ __forceinline__ void accumulate_group(
    const Sm87A4W4DownCellV2AStage& a_stage,
    const Sm87A4W4DownCellV2BStage& b_stage,
    float (&accumulators)[16U][4U]) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int fragment_m_start = warp * 16U;
  const Sm87A4W4AFragment a_fragments[kPhysicalK64PerK128] = {
      sm87_a4w4_load_a_fragment_swizzled_shared(
          a_stage.code[0U] + fragment_m_start * kPackedK64Bytes, lane),
      sm87_a4w4_load_a_fragment_swizzled_shared(
          a_stage.code[1U] + fragment_m_start * kPackedK64Bytes, lane)};

#pragma unroll
  for (unsigned int fragment_n = 0U; fragment_n < 16U; ++fragment_n) {
    const unsigned int fragment_n_start = fragment_n * 8U;
    Sm87A4W4Accumulator partial{};
#pragma unroll
    for (unsigned int half = 0U; half < kPhysicalK64PerK128; ++half) {
      const Sm87A4W4BFragment b_fragment =
          sm87_a4w4_load_b_fragment_swizzled_shared(
              b_stage.code[half] + fragment_n_start * kPackedK64Bytes,
              lane);
      sm87_a4w4_mma_m16n8k64(partial, a_fragments[half], b_fragment);
    }
    const std::int32_t integer_partial[4U] = {
        partial.x0, partial.x1, partial.x2, partial.x3};
#pragma unroll
    for (unsigned int output = 0U; output < 4U; ++output) {
      const Sm87A4W4AccumulatorCoordinate coordinate =
          sm87_a4w4_accumulator_coordinate(lane, output);
      const unsigned int local_m = fragment_m_start + coordinate.m;
      const unsigned int local_n = fragment_n_start + coordinate.n;
      const float scale_product =
          decode_bf16(a_stage.scale[local_m]) *
          decode_bf16(b_stage.scale[local_n]);
      accumulators[fragment_n][output] +=
          static_cast<float>(integer_partial[output]) * scale_product;
    }
  }
}

__device__ __forceinline__ void wait_for_group(
    const unsigned int group,
    const unsigned int group_count) noexcept {
  if (group == 0U) {
    if (group_count >= 3U) {
      cp_async_wait<2U>();
    } else if (group_count == 2U) {
      cp_async_wait<1U>();
    } else {
      cp_async_wait<0U>();
    }
  } else if (group == 1U) {
    if (group + 2U < group_count) {
      cp_async_wait<3U>();
    } else if (group + 1U < group_count) {
      cp_async_wait<2U>();
    } else {
      cp_async_wait<0U>();
    }
  } else if (group + 2U < group_count) {
    cp_async_wait<3U>();
  } else if (group + 1U < group_count) {
    cp_async_wait<2U>();
  } else {
    cp_async_wait<0U>();
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownCellV2Threads,
                      kSm87A4W4DownCellV2CtasPerSm)
void q3x_sm87_a4w4_down_complete_cell_v2_m128n128k128_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k128_scales_bf16,
    const unsigned int k128_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count,
    const unsigned int work_tile_count) {
  __shared__ Sm87A4W4DownCellV2Shared shared;

  // N-major work plus a 32-CTA stride gives the target shape a balanced,
  // fixed-M schedule: m_tiles=16, so each CTA visits exactly one M tile and
  // one of the two N parities for all 40 N tiles.
  for (unsigned int work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    const unsigned int n_tile = work_tile / m_tile_count;
    const unsigned int m_tile = work_tile - n_tile * m_tile_count;
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4DownCellV2TileM;
    const unsigned int n_tile_start =
        n_tile * kSm87A4W4DownCellV2TileN;
    float accumulators[16U][4U]{};

    issue_pair(shared, packed_a, a_k128_scales_bf16, packed_b,
               b_k128_scales_bf16, m_tile_start, n_tile_start, 0U,
               physical_k64_group_count, k128_group_count);
    if (k128_group_count > 1U) {
      issue_pair(shared, packed_a, a_k128_scales_bf16, packed_b,
                 b_k128_scales_bf16, m_tile_start, n_tile_start, 1U,
                 physical_k64_group_count, k128_group_count);
    }
    if (k128_group_count > 2U) {
      issue_b_stage(shared.b[2U], packed_b, b_k128_scales_bf16,
                    n_tile_start, 2U, physical_k64_group_count,
                    k128_group_count);
      cp_async_commit();
    }

    for (unsigned int group = 0U; group < k128_group_count; ++group) {
      wait_for_group(group, k128_group_count);
      __syncthreads();
      accumulate_group(shared.a[group % kSm87A4W4DownCellV2AStages],
                       shared.b[group % kSm87A4W4DownCellV2BStages],
                       accumulators);

      // Every warp must release the current slots before their group+2 and
      // group+3 occupants can begin overwriting them.
      __syncthreads();
      if (group + kSm87A4W4DownCellV2AStages < k128_group_count) {
        const unsigned int future_a =
            group + kSm87A4W4DownCellV2AStages;
        issue_a_stage(
            shared.a[future_a % kSm87A4W4DownCellV2AStages], packed_a,
            a_k128_scales_bf16, m_tile_start, future_a,
            physical_k64_group_count, k128_group_count);
        cp_async_commit();
      }
      if (group + kSm87A4W4DownCellV2BStages < k128_group_count) {
        const unsigned int future_b =
            group + kSm87A4W4DownCellV2BStages;
        issue_b_stage(
            shared.b[future_b % kSm87A4W4DownCellV2BStages], packed_b,
            b_k128_scales_bf16, n_tile_start, future_b,
            physical_k64_group_count, k128_group_count);
        cp_async_commit();
      }
    }

    const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
    const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
#pragma unroll
    for (unsigned int fragment_n = 0U; fragment_n < 16U; ++fragment_n) {
#pragma unroll
      for (unsigned int output = 0U; output < 4U; ++output) {
        const Sm87A4W4AccumulatorCoordinate coordinate =
            sm87_a4w4_accumulator_coordinate(lane, output);
        const unsigned int global_m =
            m_tile_start + warp * 16U + coordinate.m;
        const unsigned int global_n =
            n_tile_start + fragment_n * 8U + coordinate.n;
        output_bf16[static_cast<std::size_t>(global_m) *
                        output_row_stride_elements +
                    global_n] =
            encode_bf16(accumulators[fragment_n][output]);
      }
    }
    __syncthreads();
  }
}

[[nodiscard]] int validate_sm87(
    cudaDeviceProp* const properties = nullptr) noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp local{};
  status = cudaGetDeviceProperties(&local, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (local.major != kSm87A4W4RequiredComputeMajor ||
      local.minor != kSm87A4W4RequiredComputeMinor ||
      local.multiProcessorCount != static_cast<int>(kRequiredSmCount)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local;
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int query_sm87_a4w4_down_complete_cell_v2_resources_cuda(
    Sm87A4W4DownCellV2Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4DownCellV2Resources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_down_complete_cell_v2_m128n128k128_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_complete_cell_v2_m128n128k128_kernel,
      static_cast<int>(kSm87A4W4DownCellV2Threads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = attributes.maxDynamicSharedSizeBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->registers_per_thread >
          static_cast<int>(kSm87A4W4DownCellV2MaximumRegisters) ||
      resources->local_bytes != 0U ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4DownCellV2CtasPerSm) ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4DownCellV2Threads) ||
      resources->static_shared_bytes != kSm87A4W4DownCellV2SharedBytes) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_down_complete_cell_v2_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k128_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4DownCellV2Plan plan =
      sm87_a4w4_down_complete_cell_v2_plan(token_count, output_size,
                                           input_size);
  if (plan.launch_ctas == 0U || !aligned(packed_a, 16U) ||
      !aligned(a_k128_scales_bf16, 16U) || !aligned(packed_b, 16U) ||
      !aligned(b_k128_scales_bf16, 16U) ||
      !aligned(output_bf16, alignof(std::uint16_t)) ||
      output_row_stride_elements < output_size ||
      !product_fits(token_count, output_row_stride_elements) ||
      !consumer_capacity_fits(token_count, plan.physical_k64_groups) ||
      !consumer_capacity_fits(output_size, plan.physical_k64_groups) ||
      !shared_scale_capacity_fits(token_count, plan.k128_groups) ||
      !shared_scale_capacity_fits(output_size, plan.k128_groups) ||
      plan.k128_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_tiles > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count, input_size);
  const std::size_t required_a_scale_elements =
      sm87_a4w4_consumer_k128_scale_capacity_elements(token_count,
                                                       input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(output_size, input_size);
  const std::size_t required_b_scale_elements =
      sm87_a4w4_consumer_k128_scale_capacity_elements(output_size,
                                                       input_size);
  if (required_a_bytes == 0U || required_a_scale_elements == 0U ||
      required_b_bytes == 0U || required_b_scale_elements == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scale_elements ||
      packed_b_capacity_bytes < required_b_bytes ||
      b_scale_capacity_elements < required_b_scale_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  Sm87A4W4DownCellV2Resources resources{};
  const int resource_status =
      query_sm87_a4w4_down_complete_cell_v2_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_down_complete_cell_v2_m128n128k128_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas),
      static_cast<unsigned int>(kSm87A4W4DownCellV2Threads), 0U, stream>>>(
      packed_a, a_k128_scales_bf16, packed_b, b_k128_scales_bf16,
      static_cast<unsigned int>(plan.k128_groups),
      static_cast<unsigned int>(plan.physical_k64_groups), output_bf16,
      static_cast<unsigned int>(output_row_stride_elements),
      static_cast<unsigned int>(plan.m_tiles),
      static_cast<unsigned int>(plan.work_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
