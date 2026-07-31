#include "q3x/kernels/sm87_a4w4_down_k128_stage_major.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr std::size_t kPackedK64Bytes = 32U;
inline constexpr std::size_t kM64PackedK64Bytes = 64U * kPackedK64Bytes;
inline constexpr std::size_t kN256PackedK64Bytes =
    kSm87A4W4DownK128StageMajorTileN * kPackedK64Bytes;
inline constexpr int kRequiredSmCount = 16;

// The full slots hold B plus the lower M64 activation half.  The upper M64
// activation half has a single scratch slot: it is prefetched while the lower
// half computes, and the current B stage remains live for both halves.
struct alignas(16) Sm87A4W4DownK128FullStage final {
  std::uint8_t a[kSm87A4W4PhysicalK64BlocksPerSharedScale]
                [kM64PackedK64Bytes];
  std::uint8_t b[kSm87A4W4PhysicalK64BlocksPerSharedScale]
                [kN256PackedK64Bytes];
  std::uint16_t a_scales[64U];
  std::uint16_t b_scales[kSm87A4W4DownK128StageMajorTileN];
};

struct alignas(16) Sm87A4W4DownK128UpperAScratch final {
  std::uint8_t a[kSm87A4W4PhysicalK64BlocksPerSharedScale]
                [kM64PackedK64Bytes];
  std::uint16_t a_scales[64U];
};

struct alignas(16) Sm87A4W4DownK128SharedStorage final {
  Sm87A4W4DownK128FullStage
      pipeline[kSm87A4W4DownK128StageMajorPipelineSlots];
  Sm87A4W4DownK128UpperAScratch upper_a;
};

static_assert(sizeof(Sm87A4W4DownK128FullStage) == 21'120U);
static_assert(sizeof(Sm87A4W4DownK128UpperAScratch) == 4'224U);
static_assert(sizeof(Sm87A4W4DownK128SharedStorage) ==
              kSm87A4W4DownK128StageMajorSharedBytes);
static_assert(sizeof(Sm87A4W4DownK128SharedStorage) <= 48U * 1'024U);

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

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;"
               :
               : "n"(PendingGroups)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void prefetch_full_stage(
    Sm87A4W4DownK128FullStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k128_scales_bf16,
    const unsigned int m_tile_start, const unsigned int n_tile_start,
    const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  constexpr std::size_t kVectorsPerPhysicalK64 =
      (kM64PackedK64Bytes + kN256PackedK64Bytes) / 16U;
  constexpr std::size_t kVectorCount =
      kSm87A4W4PhysicalK64BlocksPerSharedScale *
      kVectorsPerPhysicalK64;
  static_assert(kVectorsPerPhysicalK64 == 640U);
  static_assert(kVectorCount ==
                5U * kSm87A4W4DownK128StageMajorThreads);

  for (std::size_t vector = threadIdx.x; vector < kVectorCount;
       vector += blockDim.x) {
    const std::size_t half = vector / kVectorsPerPhysicalK64;
    const std::size_t group_vector =
        vector - half * kVectorsPerPhysicalK64;
    const std::size_t physical_group =
        static_cast<std::size_t>(k128_group) *
            kSm87A4W4PhysicalK64BlocksPerSharedScale +
        half;
    if (group_vector < kM64PackedK64Bytes / 16U) {
      const std::size_t row = group_vector / 2U;
      const std::size_t row_vector = group_vector % 2U;
      const std::uint8_t* const source =
          packed_a + sm87_a4w4_consumer_packed_offset(
                         static_cast<std::size_t>(m_tile_start) + row,
                         physical_group, row_vector * 16U,
                         physical_k64_group_count);
      cp_async_16(stage.a[half] +
                      sm87_a4w4_swizzled_k64_byte_offset(
                          row, row_vector * 16U),
                  source);
    } else {
      const std::size_t b_vector =
          group_vector - kM64PackedK64Bytes / 16U;
      const std::size_t row = b_vector / 2U;
      const std::size_t row_vector = b_vector % 2U;
      const std::uint8_t* const source =
          packed_b + sm87_a4w4_consumer_packed_offset(
                         static_cast<std::size_t>(n_tile_start) + row,
                         physical_group, row_vector * 16U,
                         physical_k64_group_count);
      cp_async_16(stage.b[half] +
                      sm87_a4w4_swizzled_k64_byte_offset(
                          row, row_vector * 16U),
                  source);
    }
  }
  if (threadIdx.x < 64U) {
    stage.a_scales[threadIdx.x] =
        a_k128_scales_bf16[sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(m_tile_start) + threadIdx.x,
            k128_group, k128_group_count)];
  }
  stage.b_scales[threadIdx.x] =
      b_k128_scales_bf16[sm87_a4w4_consumer_k128_scale_offset(
          static_cast<std::size_t>(n_tile_start) + threadIdx.x,
          k128_group, k128_group_count)];
  cp_async_commit();
}

__device__ __forceinline__ void prefetch_upper_a(
    Sm87A4W4DownK128UpperAScratch& scratch,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const unsigned int m_tile_start, const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  constexpr std::size_t kVectorsPerPhysicalK64 =
      kM64PackedK64Bytes / 16U;
  constexpr std::size_t kVectorCount =
      kSm87A4W4PhysicalK64BlocksPerSharedScale *
      kVectorsPerPhysicalK64;
  static_assert(kVectorCount ==
                kSm87A4W4DownK128StageMajorThreads);

  const std::size_t vector = threadIdx.x;
  const std::size_t half = vector / kVectorsPerPhysicalK64;
  const std::size_t half_vector =
      vector - half * kVectorsPerPhysicalK64;
  const std::size_t row = half_vector / 2U;
  const std::size_t row_vector = half_vector % 2U;
  const std::size_t physical_group =
      static_cast<std::size_t>(k128_group) *
          kSm87A4W4PhysicalK64BlocksPerSharedScale +
      half;
  const std::uint8_t* const source =
      packed_a + sm87_a4w4_consumer_packed_offset(
                     static_cast<std::size_t>(m_tile_start) + 64U + row,
                     physical_group, row_vector * 16U,
                     physical_k64_group_count);
  cp_async_16(scratch.a[half] +
                  sm87_a4w4_swizzled_k64_byte_offset(
                      row, row_vector * 16U),
              source);
  if (threadIdx.x < 64U) {
    scratch.a_scales[threadIdx.x] =
        a_k128_scales_bf16[sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(m_tile_start) + 64U + threadIdx.x,
            k128_group, k128_group_count)];
  }
  cp_async_commit();
}

__device__ __forceinline__ void accumulate_m64n256_k128(
    const std::uint8_t a[kSm87A4W4PhysicalK64BlocksPerSharedScale]
                        [kM64PackedK64Bytes],
    const std::uint8_t b[kSm87A4W4PhysicalK64BlocksPerSharedScale]
                        [kN256PackedK64Bytes],
    const std::uint16_t a_scales[64U],
    const std::uint16_t b_scales[kSm87A4W4DownK128StageMajorTileN],
    const unsigned int lane, const unsigned int warp_m,
    const unsigned int warp_n, float (&accumulators)[16U][4U]) noexcept {
  const unsigned int fragment_m_start = warp_m * 16U;
  const Sm87A4W4AFragment a_fragments[2U] = {
      sm87_a4w4_load_a_fragment_swizzled_shared(
          a[0U] + fragment_m_start * kPackedK64Bytes, lane),
      sm87_a4w4_load_a_fragment_swizzled_shared(
          a[1U] + fragment_m_start * kPackedK64Bytes, lane)};

#pragma unroll
  for (unsigned int fragment_n = 0U; fragment_n < 16U; ++fragment_n) {
    const unsigned int fragment_n_start =
        warp_n * 128U + fragment_n * 8U;
    Sm87A4W4Accumulator partial{};
#pragma unroll
    for (unsigned int half = 0U;
         half < kSm87A4W4PhysicalK64BlocksPerSharedScale; ++half) {
      const Sm87A4W4BFragment b_fragment =
          sm87_a4w4_load_b_fragment_swizzled_shared(
              b[half] + fragment_n_start * kPackedK64Bytes, lane);
      // Preserve the admitted K128 contract exactly: low physical K64 first,
      // high physical K64 second into the same S32 accumulator.
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
          decode_bf16(a_scales[local_m]) *
          decode_bf16(b_scales[local_n]);
      accumulators[fragment_n][output] +=
          static_cast<float>(integer_partial[output]) * scale_product;
    }
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownK128StageMajorThreads, 1)
void q3x_sm87_a4w4_down_k128_stage_major_m128n256_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k128_scales_bf16,
    const unsigned int k128_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count,
    const unsigned int n_stripe_count) {
  __shared__ Sm87A4W4DownK128SharedStorage shared;

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m = warp / 2U;
  const unsigned int warp_n = warp % 2U;

  // A CTA stays on one N stripe for all ascending M128 tiles.  Only after the
  // complete stripe is emitted can it take stripe + gridDim.x.  This is the
  // Down-specific B-stationary ordering; no Gate-shaped M-stripe scheduler is
  // reused here.
  for (unsigned int n_stripe = blockIdx.x; n_stripe < n_stripe_count;
       n_stripe += gridDim.x) {
    const unsigned int n_tile_start =
        n_stripe * kSm87A4W4DownK128StageMajorTileN;
    for (unsigned int m_tile = 0U; m_tile < m_tile_count; ++m_tile) {
      const unsigned int m_tile_start =
          m_tile * kSm87A4W4DownK128StageMajorTileM;
      float accumulators[2U][16U][4U]{};

      prefetch_full_stage(
          shared.pipeline[0U], packed_a, a_k128_scales_bf16, packed_b,
          b_k128_scales_bf16, m_tile_start, n_tile_start, 0U,
          physical_k64_group_count, k128_group_count);
      cp_async_wait<0>();
      __syncthreads();

      for (unsigned int group = 0U; group < k128_group_count; ++group) {
        const bool has_next = group + 1U < k128_group_count;
        if (has_next) {
          prefetch_full_stage(
              shared.pipeline[(group + 1U) %
                              kSm87A4W4DownK128StageMajorPipelineSlots],
              packed_a, a_k128_scales_bf16, packed_b,
              b_k128_scales_bf16, m_tile_start, n_tile_start, group + 1U,
              physical_k64_group_count, k128_group_count);
        }
        prefetch_upper_a(shared.upper_a, packed_a, a_k128_scales_bf16,
                         m_tile_start, group, physical_k64_group_count,
                         k128_group_count);

        const Sm87A4W4DownK128FullStage& current =
            shared.pipeline[group %
                            kSm87A4W4DownK128StageMajorPipelineSlots];
        accumulate_m64n256_k128(
            current.a, current.b, current.a_scales, current.b_scales,
            lane, warp_m, warp_n, accumulators[0U]);

        // Both the next complete K128 stage and upper-M64 A scratch were
        // issued before lower-M64 compute.  Waiting here extends the current
        // B stage across the full M128 output tile without a second B load.
        cp_async_wait<0>();
        __syncthreads();
        accumulate_m64n256_k128(
            shared.upper_a.a, current.b, shared.upper_a.a_scales,
            current.b_scales, lane, warp_m, warp_n, accumulators[1U]);
        __syncthreads();
      }

#pragma unroll
      for (unsigned int m_half = 0U; m_half < 2U; ++m_half) {
#pragma unroll
        for (unsigned int fragment_n = 0U; fragment_n < 16U;
             ++fragment_n) {
#pragma unroll
          for (unsigned int output = 0U; output < 4U; ++output) {
            const Sm87A4W4AccumulatorCoordinate coordinate =
                sm87_a4w4_accumulator_coordinate(lane, output);
            const unsigned int global_m =
                m_tile_start + m_half * 64U + warp_m * 16U +
                coordinate.m;
            const unsigned int global_n =
                n_tile_start + warp_n * 128U + fragment_n * 8U +
                coordinate.n;
            output_bf16[static_cast<std::size_t>(global_m) *
                            output_row_stride_elements +
                        global_n] =
                encode_bf16(accumulators[m_half][fragment_n][output]);
          }
        }
      }
      __syncthreads();
    }
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
      local.multiProcessorCount != kRequiredSmCount) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local;
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int query_sm87_a4w4_down_k128_stage_major_resources_cuda(
    Sm87A4W4DownK128StageMajorResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4DownK128StageMajorResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_down_k128_stage_major_m128n256_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_k128_stage_major_m128n256_kernel,
      static_cast<int>(kSm87A4W4DownK128StageMajorThreads), 0U);
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
          static_cast<int>(
              kSm87A4W4DownK128StageMajorMaximumRegisters) ||
      resources->local_bytes != 0U ||
      resources->active_blocks_per_sm < 1 ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4DownK128StageMajorThreads) ||
      resources->static_shared_bytes !=
          kSm87A4W4DownK128StageMajorSharedBytes) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_down_k128_stage_major_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k128_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size, std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4DownK128StageMajorPlan plan =
      sm87_a4w4_down_k128_stage_major_plan(token_count, output_size,
                                           input_size);
  if (plan.launch_ctas == 0U || !aligned(packed_a, 16U) ||
      !aligned(a_k128_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_b, 16U) ||
      !aligned(b_k128_scales_bf16, alignof(std::uint16_t)) ||
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
      plan.n_stripes > std::numeric_limits<unsigned int>::max()) {
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
  Sm87A4W4DownK128StageMajorResources resources{};
  const int resource_status =
      query_sm87_a4w4_down_k128_stage_major_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_down_k128_stage_major_m128n256_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas),
      static_cast<unsigned int>(kSm87A4W4DownK128StageMajorThreads), 0U,
      stream>>>(
      packed_a, a_k128_scales_bf16, packed_b, b_k128_scales_bf16,
      static_cast<unsigned int>(plan.k128_groups),
      static_cast<unsigned int>(plan.physical_k64_groups), output_bf16,
      static_cast<unsigned int>(output_row_stride_elements),
      static_cast<unsigned int>(plan.m_tiles),
      static_cast<unsigned int>(plan.n_stripes));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
