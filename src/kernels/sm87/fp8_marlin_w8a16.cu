#include "q3x/kernels/sm87_fp8_marlin_w8a16.h"

#include "third_party/vllm_marlin/gptq_marlin_repack_kernel.cuh"
#include "third_party/vllm_marlin/marlin_template.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

constexpr int kStages = 4;
constexpr int kGroupBlocks = -1;

[[nodiscard]] bool aligned(const void* const pointer,
                           const std::size_t alignment) noexcept {
  return pointer != nullptr && alignment != 0U &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

// Exact word view constructed by vLLM's pack_fp8_to_int32 for a canonical
// project checkpoint tensor: [N,K] U8 -> [N,K/4] U32 -> [K/4,N] U32.
__global__ void transpose_fp8_words_kernel(
    const std::uint32_t* const canonical, const int output_size,
    const int k_words, std::uint32_t* const transposed) {
  __shared__ std::uint32_t tile[32][33];
  const int input_k = static_cast<int>(blockIdx.x) * 32 + threadIdx.x;
  const int input_n = static_cast<int>(blockIdx.y) * 32 + threadIdx.y;

#pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    const int row = input_n + offset;
    if (input_k < k_words && row < output_size) {
      tile[threadIdx.y + offset][threadIdx.x] =
          canonical[static_cast<std::size_t>(row) * k_words + input_k];
    }
  }
  __syncthreads();

  const int output_n = static_cast<int>(blockIdx.y) * 32 + threadIdx.x;
  const int output_k = static_cast<int>(blockIdx.x) * 32 + threadIdx.y;
#pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    const int row = output_k + offset;
    if (output_n < output_size && row < k_words) {
      transposed[static_cast<std::size_t>(row) * output_size + output_n] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_rne(
    const float value) {
  std::uint32_t bits = __float_as_uint(value);
  const std::uint32_t magnitude = bits & 0x7FFF'FFFFU;
  if (magnitude > 0x7F80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

// vLLM first converts the checkpoint F32 scale to the layer's BF16 dtype and
// only then multiplies by the exactly representable power 2^120.  The scalar
// checkpoint contract makes every pre-permutation channel equal, so the
// single-group Marlin scale permutation cannot change this expanded vector.
__global__ void expand_fp8_scale_kernel(
    const float* const canonical_scale, const int output_size,
    std::uint16_t* const marlin_scales) {
  const std::uint16_t rounded = encode_bf16_rne(canonical_scale[0]);
  const std::uint16_t fused = encode_bf16_rne(ldexpf(decode_bf16(rounded), 120));
  for (int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < output_size;
       index += static_cast<int>(blockDim.x) * gridDim.x) {
    marlin_scales[index] = fused;
  }
}

[[nodiscard]] int validate_fixed_device() noexcept {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(kSm87Fp8MarlinSmCount) ||
      properties.sharedMemPerBlockOptin <
          kSm87Fp8MarlinDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaSuccess);
}

template <int Threads, int ThreadMBlocks, int ThreadNBlocks,
          int ThreadKBlocks, bool MBlockSize8 = false>
[[nodiscard]] int launch_marlin_specialization(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint16_t* const marlin_scales,
    const std::size_t token_count, const int output_size,
    const int input_size, std::uint16_t* const output, float* const c_tmp,
    std::int32_t* const locks, void* const cuda_stream) noexcept {
  const auto kernel =
      marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                     vllm::kBFloat16.id(), vllm::kBFloat16.id(), Threads,
                     ThreadMBlocks, ThreadNBlocks, ThreadKBlocks, MBlockSize8,
                     kStages, kGroupBlocks, false>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87Fp8MarlinDynamicSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  kernel<<<static_cast<unsigned int>(kSm87Fp8MarlinSmCount), Threads,
           kSm87Fp8MarlinDynamicSharedBytes, stream>>>(
      reinterpret_cast<const int4*>(input),
      reinterpret_cast<const int4*>(marlin_weight),
      reinterpret_cast<int4*>(output), reinterpret_cast<int4*>(c_tmp),
      nullptr, nullptr, reinterpret_cast<const int4*>(marlin_scales), nullptr,
      nullptr, nullptr, 1, static_cast<int>(token_count), output_size,
      input_size, input_size, locks, false, false, true,
      static_cast<int>(kSm87Fp8MarlinDynamicSharedBytes));
  return static_cast<int>(cudaPeekAtLastError());
}

[[nodiscard]] int launch_marlin_segment(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint16_t* const marlin_scales,
    const std::size_t token_count, const int output_size,
    const int input_size, std::uint16_t* const output, float* const c_tmp,
    std::int32_t* const locks, void* const cuda_stream) noexcept {
  if (token_count <= 8U) {
    return launch_marlin_specialization<256, 1, 8, 8, true>(
        input, marlin_weight, marlin_scales, token_count, output_size,
        input_size, output, c_tmp, locks, cuda_stream);
  }
  if (token_count <= 16U) {
    return launch_marlin_specialization<256, 1, 8, 8>(
        input, marlin_weight, marlin_scales, token_count, output_size,
        input_size, output, c_tmp, locks, cuda_stream);
  }
  if (output_size == 1'024 &&
      token_count <= kSm87Fp8MarlinC64Tokens) {
    if (token_count <= kSm87Fp8MarlinC32Tokens) {
      return launch_marlin_specialization<128, 2, 4, 8>(
          input, marlin_weight, marlin_scales, token_count, output_size,
          input_size, output, c_tmp, locks, cuda_stream);
    }
    if (token_count <= 48U) {
      return launch_marlin_specialization<128, 3, 4, 8>(
          input, marlin_weight, marlin_scales, token_count, output_size,
          input_size, output, c_tmp, locks, cuda_stream);
    }
    return launch_marlin_specialization<128, 4, 4, 8>(
        input, marlin_weight, marlin_scales, token_count, output_size,
        input_size, output, c_tmp, locks, cuda_stream);
  }
  if (token_count <= kSm87Fp8MarlinC32Tokens) {
    return launch_marlin_specialization<256, 2, 16, 4>(
        input, marlin_weight, marlin_scales, token_count, output_size,
        input_size, output, c_tmp, locks, cuda_stream);
  }
  if (token_count <= 48U) {
    return launch_marlin_specialization<256, 3, 16, 4>(
        input, marlin_weight, marlin_scales, token_count, output_size,
        input_size, output, c_tmp, locks, cuda_stream);
  }
  return launch_marlin_specialization<256, 4, 16, 4>(
      input, marlin_weight, marlin_scales, token_count, output_size,
      input_size, output, c_tmp, locks, cuda_stream);
}

}  // namespace

int prepare_sm87_fp8_marlin_projection_cuda(
    const std::uint8_t* const canonical_weight,
    const float* const canonical_weight_scale_device,
    const std::size_t output_size, const std::size_t input_size,
    std::uint8_t* const marlin_weight,
    std::uint16_t* const marlin_scales, void* const transpose_scratch,
    const std::size_t transpose_scratch_bytes,
    void* const cuda_stream) noexcept {
  const std::size_t required_weight_bytes =
      sm87_fp8_marlin_weight_bytes(output_size, input_size);
  if (!sm87_fp8_marlin_supports_shape(output_size, input_size) ||
      !aligned(canonical_weight, alignof(std::uint32_t)) ||
      !aligned(canonical_weight_scale_device, alignof(float)) ||
      !aligned(marlin_weight, 16U) ||
      !aligned(marlin_scales, 16U) ||
      !aligned(transpose_scratch, 16U) ||
      transpose_scratch_bytes < required_weight_bytes) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_fixed_device();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }

  const int output_size_int = static_cast<int>(output_size);
  const int input_size_int = static_cast<int>(input_size);
  const int k_words = input_size_int / 4;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const dim3 transpose_block(32U, 8U, 1U);
  const dim3 transpose_grid(
      static_cast<unsigned int>((k_words + 31) / 32),
      static_cast<unsigned int>((output_size_int + 31) / 32), 1U);
  transpose_fp8_words_kernel<<<transpose_grid, transpose_block, 0U, stream>>>(
      reinterpret_cast<const std::uint32_t*>(canonical_weight),
      output_size_int, k_words,
      reinterpret_cast<std::uint32_t*>(transpose_scratch));
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  const auto repack =
      marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 8, false,
                                        false>;
  status = cudaFuncSetAttribute(
      repack, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87Fp8MarlinDynamicSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  repack<<<static_cast<unsigned int>(kSm87Fp8MarlinSmCount),
           marlin::repack_threads, kSm87Fp8MarlinDynamicSharedBytes, stream>>>(
      reinterpret_cast<const std::uint32_t*>(transpose_scratch), nullptr,
      reinterpret_cast<std::uint32_t*>(marlin_weight), input_size_int,
      output_size_int);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  constexpr unsigned int kScaleThreads = 256U;
  const unsigned int scale_blocks = static_cast<unsigned int>(
      (output_size + kScaleThreads - 1U) / kScaleThreads);
  expand_fp8_scale_kernel<<<scale_blocks, kScaleThreads, 0U, stream>>>(
      canonical_weight_scale_device, output_size_int, marlin_scales);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_fp8_marlin_projection_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint16_t* const marlin_scales,
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size, std::uint16_t* const output,
    float* const c_tmp, std::int32_t* const locks,
    void* const cuda_stream) noexcept {
  if (!sm87_fp8_marlin_supports_token_count(token_count) ||
      !sm87_fp8_marlin_supports_shape(output_size, input_size) ||
      !aligned(input, 16U) || !aligned(marlin_weight, 16U) ||
      !aligned(marlin_scales, 16U) || !aligned(output, 16U) ||
      !aligned(c_tmp, 16U) || !aligned(locks, alignof(std::int32_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_fixed_device();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  const int output_size_int = static_cast<int>(output_size);
  const int input_size_int = static_cast<int>(input_size);
  const Sm87Fp8MarlinExecutionPlan plan =
      sm87_fp8_marlin_execution_plan(token_count);
  const int primary_status = launch_marlin_segment(
      input, marlin_weight, marlin_scales, plan.primary_tokens,
      output_size_int, input_size_int, output, c_tmp, locks, cuda_stream);
  if (primary_status != static_cast<int>(cudaSuccess) ||
      plan.remainder_tokens == 0U) {
    return primary_status;
  }
  const std::size_t input_offset = plan.primary_tokens * input_size;
  const std::size_t output_offset = plan.primary_tokens * output_size;
  return launch_marlin_segment(
      input + input_offset, marlin_weight, marlin_scales,
      plan.remainder_tokens, output_size_int, input_size_int,
      output + output_offset, c_tmp, locks, cuda_stream);
}

}  // namespace q3x::kernels
