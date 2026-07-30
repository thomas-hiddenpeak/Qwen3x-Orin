#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr std::size_t kPackedK64Bytes =
    kSm87A4W4PrefillTileK / 2U;
inline constexpr std::size_t kStageABytes =
    kSm87A4W4PrefillTileM * kPackedK64Bytes;
inline constexpr std::size_t kStageBBytes =
    kSm87A4W4PrefillTileN * kPackedK64Bytes;
inline constexpr int kRequiredSmCount = 16;

struct alignas(16) Sm87A4W4PipelineStage final {
  std::uint8_t a[kStageABytes];
  std::uint8_t b[kStageBBytes];
};

static_assert(sizeof(Sm87A4W4PipelineStage) == 4'096U);
static_assert(kSm87A4W4PrefillThreads * 16U ==
              sizeof(Sm87A4W4PipelineStage));

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

extern "C" __global__ __launch_bounds__(kSm87A4W4PrefillThreads)
void q3x_sm87_a4_quantize_bf16_kernel(
    const std::uint16_t* const input_bf16,
    const std::size_t input_row_stride_elements,
    const std::size_t k64_group_count,
    const std::size_t group_count,
    const float clip_ratio,
    std::uint8_t* const packed_a,
    const std::size_t packed_a_row_stride_bytes,
    std::uint16_t* const a_k64_scales_bf16,
    const std::size_t scale_row_stride_elements) {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const std::size_t group_ordinal =
      static_cast<std::size_t>(blockIdx.x) * kSm87A4W4PrefillWarps + warp;
  if (group_ordinal >= group_count) {
    return;
  }
  const std::size_t row = group_ordinal / k64_group_count;
  const std::size_t group = group_ordinal - row * k64_group_count;
  const std::size_t input_offset =
      row * input_row_stride_elements + group * kSm87A4W4PrefillTileK;
  float even = decode_bf16(input_bf16[input_offset + 2U * lane]);
  float odd = decode_bf16(input_bf16[input_offset + 2U * lane + 1U]);
  float maximum = fmaxf(fabsf(even), fabsf(odd));
#pragma unroll
  for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
    maximum = fmaxf(maximum,
                    __shfl_down_sync(0xffffffffU, maximum, delta));
  }
  maximum = __shfl_sync(0xffffffffU, maximum, 0U);
  const float clipped_maximum = maximum * clip_ratio;
  const float scale = clipped_maximum > 0.0F ? clipped_maximum / 7.0F : 1.0F;
  const float inverse_scale = 1.0F / scale;
  even = fminf(fmaxf(even, -clipped_maximum), clipped_maximum);
  odd = fminf(fmaxf(odd, -clipped_maximum), clipped_maximum);
  const int even_rounded = __float2int_rn(even * inverse_scale);
  const int odd_rounded = __float2int_rn(odd * inverse_scale);
  const int even_code = even_rounded < -7 ? -7 :
                        (even_rounded > 7 ? 7 : even_rounded);
  const int odd_code = odd_rounded < -7 ? -7 :
                       (odd_rounded > 7 ? 7 : odd_rounded);
  packed_a[row * packed_a_row_stride_bytes + group * kPackedK64Bytes + lane] =
      sm87_a4w4_pack_signed_pair(even_code, odd_code);
  if (lane == 0U) {
    a_k64_scales_bf16[row * scale_row_stride_elements + group] =
        encode_bf16(scale);
  }
}

__device__ __forceinline__ void cp_async_16(
    void* const destination, const void* const source,
    const unsigned int source_bytes) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :
               : "r"(shared_address), "l"(source), "r"(source_bytes)
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

__device__ __forceinline__ void prefetch_stage(
    Sm87A4W4PipelineStage& stage,
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_row_stride_bytes,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_row_stride_bytes,
    const std::size_t token_count,
    const std::size_t m_tile_start,
    const std::size_t n_tile_start,
    const std::size_t k64_group) noexcept {
  const std::size_t vector = threadIdx.x;
  if (vector < kStageABytes / 16U) {
    const std::size_t row = vector / 2U;
    const std::size_t row_vector = vector % 2U;
    const std::size_t global_row = m_tile_start + row;
    const bool valid = global_row < token_count;
    const std::uint8_t* const source =
        valid ? packed_a + global_row * packed_a_row_stride_bytes +
                    k64_group * kPackedK64Bytes + row_vector * 16U
              : packed_a;
    cp_async_16(stage.a + row * kPackedK64Bytes + row_vector * 16U,
                source, valid ? 16U : 0U);
  } else {
    const std::size_t b_vector = vector - kStageABytes / 16U;
    const std::size_t row = b_vector / 2U;
    const std::size_t row_vector = b_vector % 2U;
    const std::size_t global_row = n_tile_start + row;
    const std::uint8_t* const source =
        packed_b + global_row * packed_b_row_stride_bytes +
        k64_group * kPackedK64Bytes + row_vector * 16U;
    cp_async_16(stage.b + row * kPackedK64Bytes + row_vector * 16U,
                source, 16U);
  }
  cp_async_commit();
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4PrefillThreads,
                      kSm87A4W4PrefillCtasPerSm)
void q3x_sm87_a4w4_prefill_gemm_kernel(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_row_stride_bytes,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::size_t a_scale_row_stride_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_row_stride_bytes,
    const std::uint16_t* const b_k64_scales_bf16,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t k64_group_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t m_tile_count,
    const std::size_t work_tile_count) {
  __shared__ Sm87A4W4PipelineStage
      pipeline[kSm87A4W4PrefillPipelineStages];

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const std::size_t warp_m = warp % 4U;
  const std::size_t warp_n = warp / 4U;

  for (std::size_t work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    // N-major ordering groups every M tile of the same B tile contiguously.
    // At P512, each 32-CTA wave therefore works on four N64 tiles and reuses
    // each B tile across eight M64 tiles through L2.
    const std::size_t n_tile = work_tile / m_tile_count;
    const std::size_t m_tile = work_tile - n_tile * m_tile_count;
    const std::size_t m_tile_start = m_tile * kSm87A4W4PrefillTileM;
    const std::size_t n_tile_start = n_tile * kSm87A4W4PrefillTileN;

    float accumulators[4U][4U]{};

    prefetch_stage(pipeline[0U], packed_a,
                   packed_a_row_stride_bytes, packed_b,
                   packed_b_row_stride_bytes, token_count, m_tile_start,
                   n_tile_start, 0U);
    if (k64_group_count > 1U) {
      prefetch_stage(pipeline[1U], packed_a,
                     packed_a_row_stride_bytes, packed_b,
                     packed_b_row_stride_bytes, token_count, m_tile_start,
                     n_tile_start, 1U);
    }

    for (std::size_t group = 0U; group < k64_group_count; ++group) {
      if (group + 2U < k64_group_count) {
        prefetch_stage(
            pipeline[(group + 2U) % kSm87A4W4PrefillPipelineStages],
            packed_a, packed_a_row_stride_bytes, packed_b,
            packed_b_row_stride_bytes, token_count, m_tile_start,
            n_tile_start, group + 2U);
      }
      if (group + 1U == k64_group_count) {
        cp_async_wait<0>();
      } else {
        cp_async_wait<1>();
      }
      __syncthreads();

      const Sm87A4W4PipelineStage& stage =
          pipeline[group % kSm87A4W4PrefillPipelineStages];
      const std::uint8_t* const warp_a =
          stage.a + warp_m * 16U * kPackedK64Bytes;
      const Sm87A4W4AFragment a = sm87_a4w4_load_a_fragment(
          warp_a, kPackedK64Bytes, 0U, lane);

#pragma unroll
      for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
        const std::size_t fragment_n = warp_n * 32U + fragment * 8U;
        const std::uint8_t* const warp_b =
            stage.b + fragment_n * kPackedK64Bytes;
        const Sm87A4W4BFragment b = sm87_a4w4_load_b_fragment(
            warp_b, kPackedK64Bytes, 0U, lane);
        Sm87A4W4Accumulator partial{};
        sm87_a4w4_mma_m16n8k64(partial, a, b);
        const std::int32_t integer_partial[4U] = {
            partial.x0, partial.x1, partial.x2, partial.x3};

#pragma unroll
        for (unsigned int output = 0U; output < 4U; ++output) {
          const Sm87A4W4AccumulatorCoordinate coordinate =
              sm87_a4w4_accumulator_coordinate(lane, output);
          const std::size_t local_m = warp_m * 16U + coordinate.m;
          const std::size_t local_n = fragment_n + coordinate.n;
          const std::size_t global_m = m_tile_start + local_m;
          const std::size_t global_n = n_tile_start + local_n;
          // The final M tile may contain zero-filled rows.  Its first row is
          // always valid, so it is a safe scale address for masked rows.
          const std::size_t scale_m =
              global_m < token_count ? global_m : m_tile_start;
          const float a_scale = decode_bf16(
              a_k64_scales_bf16[
                  scale_m * a_scale_row_stride_elements + group]);
          const float b_scale = decode_bf16(
              b_k64_scales_bf16[global_n * k64_group_count + group]);
          accumulators[fragment][output] +=
              static_cast<float>(integer_partial[output]) * a_scale *
              b_scale;
        }
      }
      // All warps must finish reading this stage before the next iteration
      // can recycle it for group+3.
      __syncthreads();
    }

#pragma unroll
    for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
#pragma unroll
      for (unsigned int output = 0U; output < 4U; ++output) {
        const Sm87A4W4AccumulatorCoordinate coordinate =
            sm87_a4w4_accumulator_coordinate(lane, output);
        const std::size_t global_m =
            m_tile_start + warp_m * 16U + coordinate.m;
        const std::size_t global_n =
            n_tile_start + warp_n * 32U + fragment * 8U + coordinate.n;
        if (global_m < token_count && global_n < output_size) {
          output_bf16[global_m * output_row_stride_elements + global_n] =
              encode_bf16(accumulators[fragment][output]);
        }
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
      local.multiProcessorCount != kRequiredSmCount) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local;
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int launch_sm87_a4_quantize_bf16_cuda(
    const std::uint16_t* const input_bf16,
    const std::size_t input_row_stride_elements,
    const std::size_t token_count,
    const std::size_t input_size,
    const float clip_ratio,
    std::uint8_t* const packed_a,
    const std::size_t packed_a_row_stride_bytes,
    std::uint16_t* const a_k64_scales_bf16,
    const std::size_t scale_row_stride_elements,
    void* const cuda_stream) noexcept {
  if (token_count == 0U || input_size == 0U ||
      input_size % kSm87A4W4PrefillTileK != 0U ||
      !(clip_ratio > 0.0F && clip_ratio <= 1.0F) ||
      !aligned(input_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k64_scales_bf16, alignof(std::uint16_t)) ||
      input_row_stride_elements < input_size ||
      packed_a_row_stride_bytes < input_size / 2U ||
      packed_a_row_stride_bytes % 16U != 0U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t k64_groups = input_size / kSm87A4W4PrefillTileK;
  if (scale_row_stride_elements < k64_groups ||
      !product_fits(token_count, k64_groups)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_sm87();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  const std::size_t group_count = token_count * k64_groups;
  const std::size_t blocks =
      sm87_a4w4_ceil_div(group_count, kSm87A4W4PrefillWarps);
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4_quantize_bf16_kernel<<<
      static_cast<unsigned int>(blocks),
      static_cast<unsigned int>(kSm87A4W4PrefillThreads), 0U, stream>>>(
      input_bf16, input_row_stride_elements, k64_groups, group_count,
      clip_ratio, packed_a, packed_a_row_stride_bytes,
      a_k64_scales_bf16, scale_row_stride_elements);
  return static_cast<int>(cudaPeekAtLastError());
}

int query_sm87_a4w4_prefill_gemm_resources_cuda(
    Sm87A4W4PrefillGemmResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4PrefillGemmResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_prefill_gemm_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_prefill_gemm_kernel,
      static_cast<int>(kSm87A4W4PrefillThreads), 0U);
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
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_prefill_gemm_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_row_stride_bytes,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::size_t a_scale_row_stride_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_row_stride_bytes,
    const std::uint16_t* const b_k64_scales_bf16,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4PrefillGemmPlan plan =
      sm87_a4w4_prefill_gemm_plan(token_count, output_size, input_size);
  if (plan.launch_ctas == 0U || !aligned(packed_a, 16U) ||
      !aligned(packed_b, 16U) ||
      !aligned(a_k64_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(b_k64_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(output_bf16, alignof(std::uint16_t)) ||
      !product_fits(plan.k64_groups, kPackedK64Bytes) ||
      packed_a_row_stride_bytes < input_size / 2U ||
      packed_b_row_stride_bytes < input_size / 2U ||
      a_scale_row_stride_elements < plan.k64_groups ||
      packed_a_row_stride_bytes % 16U != 0U ||
      packed_b_row_stride_bytes % 16U != 0U ||
      output_row_stride_elements < output_size) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_sm87();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  Sm87A4W4PrefillGemmResources resources{};
  const int resource_status =
      query_sm87_a4w4_prefill_gemm_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }
  if (resources.active_blocks_per_sm <
          static_cast<int>(kSm87A4W4PrefillCtasPerSm) ||
      resources.local_bytes != 0U) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_prefill_gemm_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas),
      static_cast<unsigned int>(kSm87A4W4PrefillThreads), 0U, stream>>>(
      packed_a, packed_a_row_stride_bytes, a_k64_scales_bf16,
      a_scale_row_stride_elements, packed_b, packed_b_row_stride_bytes,
      b_k64_scales_bf16, token_count, output_size, plan.k64_groups, output_bf16,
      output_row_stride_elements, plan.m_tiles, plan.work_tiles);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
