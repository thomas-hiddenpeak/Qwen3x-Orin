#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

[[nodiscard]] constexpr bool pointer_is_aligned(
    const void* const pointer, const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] constexpr bool multiplication_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

extern "C" __global__ __launch_bounds__(kSm87A4W4WarpThreads)
void q3x_sm87_a4w4_k64_primitive_kernel(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_row_stride_bytes,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_row_stride_bytes,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint16_t* const b_k64_scales_bf16,
    const std::size_t k64_group_count,
    const std::size_t k64_group,
    float* const output,
    const std::size_t output_row_stride_elements) {
  const unsigned int lane = threadIdx.x;
  const Sm87A4W4AFragment a = sm87_a4w4_load_a_fragment(
      packed_a, packed_a_row_stride_bytes, k64_group, lane);
  const Sm87A4W4BFragment b = sm87_a4w4_load_b_fragment(
      packed_b, packed_b_row_stride_bytes, k64_group, lane);
  Sm87A4W4Accumulator accumulator{};
  sm87_a4w4_mma_m16n8k64(accumulator, a, b);

  const std::int32_t values[4] = {
      accumulator.x0,
      accumulator.x1,
      accumulator.x2,
      accumulator.x3,
  };
#pragma unroll
  for (unsigned int register_index = 0U; register_index < 4U;
       ++register_index) {
    const Sm87A4W4AccumulatorCoordinate coordinate =
        sm87_a4w4_accumulator_coordinate(lane, register_index);
    const float a_scale = decode_bf16(
        a_k64_scales_bf16[sm87_a4w4_k64_scale_offset(
            coordinate.m, k64_group, k64_group_count)]);
    const float b_scale = decode_bf16(
        b_k64_scales_bf16[sm87_a4w4_k64_scale_offset(
            coordinate.n, k64_group, k64_group_count)]);
    output[static_cast<std::size_t>(coordinate.m) *
               output_row_stride_elements +
           coordinate.n] =
        static_cast<float>(values[register_index]) * a_scale * b_scale;
  }
}

[[nodiscard]] int reject_non_sm87(
    cudaDeviceProp* const properties = nullptr) noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp local_properties{};
  status = cudaGetDeviceProperties(&local_properties, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (local_properties.major != kSm87A4W4RequiredComputeMajor ||
      local_properties.minor != kSm87A4W4RequiredComputeMinor) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local_properties;
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int query_sm87_a4w4_primitive_resources_cuda(
    Sm87A4W4PrimitiveResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4PrimitiveResources{};

  cudaDeviceProp properties{};
  int status = reject_non_sm87(&properties);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  cudaFuncAttributes attributes{};
  const cudaError_t attribute_status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_k64_primitive_kernel);
  if (attribute_status != cudaSuccess) {
    return static_cast<int>(attribute_status);
  }
  int active_blocks = 0;
  const cudaError_t occupancy_status =
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, q3x_sm87_a4w4_k64_primitive_kernel,
          static_cast<int>(kSm87A4W4WarpThreads), 0U);
  if (occupancy_status != cudaSuccess) {
    return static_cast<int>(occupancy_status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = 0U;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_k64_primitive_smoke_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_row_stride_bytes,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_row_stride_bytes,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint16_t* const b_k64_scales_bf16,
    const std::size_t k64_group_count,
    const std::size_t k64_group,
    float* const output,
    const std::size_t output_row_stride_elements,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kPackedAlignment = alignof(std::uint32_t);
  if (!pointer_is_aligned(packed_a, kPackedAlignment) ||
      !pointer_is_aligned(packed_b, kPackedAlignment) ||
      !pointer_is_aligned(a_k64_scales_bf16, alignof(std::uint16_t)) ||
      !pointer_is_aligned(b_k64_scales_bf16, alignof(std::uint16_t)) ||
      !pointer_is_aligned(output, alignof(float)) ||
      packed_a_row_stride_bytes < kSm87A4W4MmaK / 2U ||
      packed_b_row_stride_bytes < kSm87A4W4MmaK / 2U ||
      packed_a_row_stride_bytes % kPackedAlignment != 0U ||
      packed_b_row_stride_bytes % kPackedAlignment != 0U ||
      output_row_stride_elements < kSm87A4W4MmaN ||
      k64_group_count == 0U || k64_group >= k64_group_count ||
      !multiplication_fits(k64_group_count, kSm87A4W4MmaK / 2U) ||
      packed_a_row_stride_bytes <
          k64_group_count * (kSm87A4W4MmaK / 2U) ||
      packed_b_row_stride_bytes <
          k64_group_count * (kSm87A4W4MmaK / 2U)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int architecture_status = reject_non_sm87();
  if (architecture_status != static_cast<int>(cudaSuccess)) {
    return architecture_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_k64_primitive_kernel<<<
      1U, static_cast<unsigned int>(kSm87A4W4WarpThreads), 0U, stream>>>(
      packed_a, packed_a_row_stride_bytes, packed_b,
      packed_b_row_stride_bytes, a_k64_scales_bf16, b_k64_scales_bf16,
      k64_group_count, k64_group, output, output_row_stride_elements);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
