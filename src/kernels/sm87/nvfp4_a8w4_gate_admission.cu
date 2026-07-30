#include "q3x/kernels/sm87_nvfp4_a8w4_gate_admission.h"

#include "third_party/vllm_marlin/gptq_marlin_repack_kernel.cuh"
#include "third_party/vllm_marlin/marlin_template.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

constexpr int kThreads =
    static_cast<int>(kSm87NvFp4A8W4GateAdmissionThreads);
constexpr int kThreadMBlocks = 4;
constexpr int kThreadNBlocks = 16;
constexpr int kThreadKBlocks = 4;
constexpr int kStages = 4;
constexpr int kGroupBlocks = 2;
constexpr unsigned int kPrepareBlocks = 4096U;

// Keep the documented dynamic allocation anchored to the exact vendored
// specialization.  All units below are int4 slots.
constexpr std::size_t kInputScalePrefix = 4U * kThreadMBlocks;
constexpr std::size_t kBStage =
    ((kThreadNBlocks * 16U) * 16U / 8U) / 2U * kThreadKBlocks / 2U;
constexpr std::size_t kBPipeline = kStages * kBStage;
constexpr std::size_t kReduction =
    (2U * kThreadNBlocks + 1U) * 16U * kThreadMBlocks;
constexpr std::size_t kBias = kThreadNBlocks * 16U / 8U;
constexpr std::size_t kBReductionBias =
    (kReduction > kBPipeline ? kReduction : kBPipeline) >
            ((kReduction < kBPipeline ? kReduction : kBPipeline) + kBias)
        ? (kReduction > kBPipeline ? kReduction : kBPipeline)
        : ((kReduction < kBPipeline ? kReduction : kBPipeline) + kBias);
constexpr std::size_t kScalePipeline =
    kStages * (kThreadKBlocks / kGroupBlocks) *
    (16U * kThreadNBlocks / 8U);
constexpr std::size_t kActivationPipeline =
    kStages * (16U * kThreadKBlocks / 16U) * (16U * kThreadMBlocks);
static_assert((kInputScalePrefix + kBReductionBias + kScalePipeline +
               kActivationPipeline) *
                  sizeof(int4) ==
              kSm87NvFp4A8W4GateAdmissionDynamicSharedBytes);

[[nodiscard]] bool aligned(const void* const pointer,
                           const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
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
          static_cast<int>(kSm87NvFp4A8W4GateAdmissionSmCount) ||
      properties.sharedMemPerBlockOptin <
          kSm87NvFp4A8W4GateAdmissionDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaSuccess);
}

__device__ __forceinline__ float decode_e4m3fn(
    const std::uint8_t bits) {
  const unsigned int sign =
      static_cast<unsigned int>(bits & 0x80U) << 24U;
  const unsigned int magnitude = static_cast<unsigned int>(bits & 0x7fU);
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  if (magnitude == 0x7fU) {
    return __uint_as_float(sign | 0x7fc0'0000U);
  }
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return __uint_as_float(sign);
    }
    const unsigned int leading =
        mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    const unsigned int fp32_exponent = 118U + leading;
    const unsigned int fp32_mantissa =
        (mantissa - (1U << leading)) << (23U - leading);
    return __uint_as_float(sign | (fp32_exponent << 23U) | fp32_mantissa);
  }
  return __uint_as_float(sign | ((120U + exponent) << 23U) |
                         (mantissa << 20U));
}

__device__ __forceinline__ float decode_e2m1(
    const std::uint8_t nibble) {
  const unsigned int sign =
      static_cast<unsigned int>(nibble & 0x08U) << 28U;
  const unsigned int magnitude = static_cast<unsigned int>(nibble & 0x07U);
  const unsigned int nonzero_mask =
      0U - static_cast<unsigned int>(magnitude != 0U);
  const unsigned int mantissa =
      ((magnitude & 1U) & static_cast<unsigned int>(magnitude > 1U)) << 22U;
  const unsigned int finite_bits =
      ((126U + (magnitude >> 1U)) << 23U) | mantissa;
  return __uint_as_float(sign | (finite_bits & nonzero_mask));
}

__device__ __forceinline__ float decode_bf16_bits(
    const std::uint16_t bits) {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

__device__ __forceinline__ float warp_max(float value) {
  constexpr unsigned int kMask = 0xffff'ffffU;
  value = fmaxf(value, __shfl_down_sync(kMask, value, 16));
  value = fmaxf(value, __shfl_down_sync(kMask, value, 8));
  value = fmaxf(value, __shfl_down_sync(kMask, value, 4));
  value = fmaxf(value, __shfl_down_sync(kMask, value, 2));
  value = fmaxf(value, __shfl_down_sync(kMask, value, 1));
  return value;
}

__global__ void recode_canonical_nvfp4_to_symmetric_w4_kernel(
    const std::uint8_t* const canonical_weight,
    const std::uint8_t* const canonical_block_scales,
    const float* const canonical_weight_scale_2,
    std::uint32_t* const canonical_qweight,
    float* const beta,
    float* const maximum_beta) {
  constexpr std::size_t kRows = kSm87NvFp4A8W4GateAdmissionRows;
  constexpr std::size_t kColumns = kSm87NvFp4A8W4GateAdmissionColumns;
  constexpr std::size_t kGroups = kSm87NvFp4A8W4GateAdmissionGroups;
  constexpr std::size_t kPackedRow = kColumns / 2U;
  constexpr std::size_t kScaleRow = kColumns / 16U;
  constexpr std::size_t kTotalGroups = kRows * kGroups;

  const std::size_t linear_thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t linear_stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const float global_scale = canonical_weight_scale_2[0];
  float thread_maximum = 0.0F;

  for (std::size_t linear_group = linear_thread;
       linear_group < kTotalGroups; linear_group += linear_stride) {
    const std::size_t n = linear_group / kGroups;
    const std::size_t group = linear_group - n * kGroups;
    const std::size_t first_k = group * 32U;
    const float scale0 =
        decode_e4m3fn(canonical_block_scales[n * kScaleRow + group * 2U]) *
        global_scale;
    const float scale1 =
        decode_e4m3fn(
            canonical_block_scales[n * kScaleRow + group * 2U + 1U]) *
        global_scale;

    float maximum = 0.0F;
#pragma unroll
    for (int lane = 0; lane < 32; ++lane) {
      const std::size_t k = first_k + static_cast<std::size_t>(lane);
      const std::uint8_t packed =
          canonical_weight[n * kPackedRow + k / 2U];
      const std::uint8_t code =
          (k & 1U) == 0U ? static_cast<std::uint8_t>(packed & 0x0fU)
                         : static_cast<std::uint8_t>(packed >> 4U);
      const float scale = lane < 16 ? scale0 : scale1;
      maximum = fmaxf(maximum, fabsf(decode_e2m1(code) * scale));
    }
    const float group_beta = maximum * (1.0F / 7.0F);
    beta[group * kRows + n] = group_beta;
    thread_maximum = fmaxf(thread_maximum, group_beta);

#pragma unroll
    for (int word = 0; word < 4; ++word) {
      std::uint32_t qword = 0U;
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        const int group_lane = word * 8 + lane;
        const std::size_t k =
            first_k + static_cast<std::size_t>(group_lane);
        const std::uint8_t packed =
            canonical_weight[n * kPackedRow + k / 2U];
        const std::uint8_t code =
            (k & 1U) == 0U ? static_cast<std::uint8_t>(packed & 0x0fU)
                           : static_cast<std::uint8_t>(packed >> 4U);
        const float scale = group_lane < 16 ? scale0 : scale1;
        const float value = decode_e2m1(code) * scale;
        int quantized =
            group_beta == 0.0F ? 0 : __float2int_rn(value / group_beta);
        quantized = max(-7, min(7, quantized));
        const unsigned int biased =
            static_cast<unsigned int>(quantized + 8);
        qword |= biased << (static_cast<unsigned int>(lane) * 4U);
      }
      canonical_qweight[(group * 4U + static_cast<std::size_t>(word)) *
                            kRows +
                        n] = qword;
    }
  }

  thread_maximum = warp_max(thread_maximum);
  __shared__ float warp_maxima[8];
  if ((threadIdx.x & 31U) == 0U) {
    warp_maxima[threadIdx.x / 32U] = thread_maximum;
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    float block_maximum = threadIdx.x < 8U ? warp_maxima[threadIdx.x] : 0.0F;
    block_maximum = warp_max(block_maximum);
    if (threadIdx.x == 0U) {
      atomicMax(reinterpret_cast<unsigned int*>(maximum_beta),
                __float_as_uint(block_maximum));
    }
  }
}

__device__ __forceinline__ unsigned int scale_source_lane(
    const unsigned int destination_lane) {
  const unsigned int outer = destination_lane / 8U;
  const unsigned int inner = destination_lane % 8U;
  return 2U * outer + 8U * (inner / 2U) + inner % 2U;
}

__global__ void compress_and_permute_scales_kernel(
    const float* const beta,
    const float* const maximum_beta,
    std::uint16_t* const integer_scales,
    float* const rho) {
  constexpr std::size_t kRows = kSm87NvFp4A8W4GateAdmissionRows;
  constexpr std::size_t kScaleElements =
      kSm87NvFp4A8W4GateAdmissionScaleElements;
  const float maximum = maximum_beta[0];
  const float scale_factor = maximum == 0.0F ? 1.0F : maximum / 4096.0F;

  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    rho[0] = scale_factor;
  }
  for (std::size_t destination =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       destination < kScaleElements;
       destination += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t group = destination / kRows;
    const std::size_t destination_n = destination - group * kRows;
    const std::size_t chunk = destination_n / 32U;
    const unsigned int destination_lane =
        static_cast<unsigned int>(destination_n % 32U);
    const std::size_t source_n =
        chunk * 32U + scale_source_lane(destination_lane);
    int encoded =
        maximum == 0.0F
            ? 0
            : __float2int_rn(beta[group * kRows + source_n] / scale_factor);
    encoded = max(0, min(4096, encoded));
    integer_scales[destination] = static_cast<std::uint16_t>(encoded);
  }
}

__global__ void dynamic_per_row_a8_kernel(
    const std::uint16_t* const activation,
    const float* const rho,
    std::int8_t* const quantized_activation,
    float* const row_scales) {
  constexpr std::size_t kColumns = kSm87NvFp4A8W4GateAdmissionColumns;
  const std::size_t row = blockIdx.x;
  const std::size_t row_offset = row * kColumns;
  float maximum = 0.0F;
  for (std::size_t column = threadIdx.x; column < kColumns;
       column += blockDim.x) {
    maximum = fmaxf(maximum,
                    fabsf(decode_bf16_bits(activation[row_offset + column])));
  }
  maximum = warp_max(maximum);
  __shared__ float warp_maxima[8];
  if ((threadIdx.x & 31U) == 0U) {
    warp_maxima[threadIdx.x / 32U] = maximum;
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    float block_maximum = threadIdx.x < 8U ? warp_maxima[threadIdx.x] : 0.0F;
    block_maximum = warp_max(block_maximum);
    if (threadIdx.x == 0U) {
      warp_maxima[0] = block_maximum;
    }
  }
  __syncthreads();
  const float alpha = warp_maxima[0] == 0.0F
                          ? 1.0F
                          : warp_maxima[0] * (1.0F / 127.0F);
  if (threadIdx.x == 0U) {
    row_scales[row] = alpha * rho[0];
  }
  for (std::size_t column = threadIdx.x; column < kColumns;
       column += blockDim.x) {
    const float value = decode_bf16_bits(activation[row_offset + column]);
    int quantized = __float2int_rn(value / alpha);
    quantized = max(-127, min(127, quantized));
    quantized_activation[row_offset + column] =
        static_cast<std::int8_t>(quantized);
  }
}

[[nodiscard]] auto a8w4_gate_kernel() noexcept {
  return marlin::Marlin<vllm::kS8.id(), vllm::kU4B8.id(),
                        vllm::kBFloat16.id(), vllm::kBFloat16.id(), kThreads,
                        kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                        kStages, kGroupBlocks, false>;
}

}  // namespace

int prepare_sm87_nvfp4_a8w4_gate_m512_admission_cuda(
    const std::uint8_t* const canonical_weight,
    const std::uint8_t* const canonical_block_scales,
    const float* const canonical_weight_scale_2,
    std::uint8_t* const marlin_weight,
    std::uint16_t* const marlin_integer_scales,
    float* const rho,
    void* const scratch,
    const std::size_t scratch_bytes,
    const bool enable_approximate_admission,
    void* const cuda_stream) noexcept {
  if (!enable_approximate_admission) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!aligned(canonical_weight, 4U) || canonical_block_scales == nullptr ||
      !aligned(canonical_weight_scale_2, alignof(float)) ||
      !aligned(marlin_weight, 16U) || !aligned(marlin_integer_scales, 16U) ||
      !aligned(rho, alignof(float)) || !aligned(scratch, 16U) ||
      scratch_bytes < kSm87NvFp4A8W4GateAdmissionScratchBytes) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_fixed_device();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }

  auto* const scratch_bytes_pointer = static_cast<std::uint8_t*>(scratch);
  auto* const canonical_qweight =
      reinterpret_cast<std::uint32_t*>(scratch_bytes_pointer);
  auto* const beta = reinterpret_cast<float*>(
      scratch_bytes_pointer + kSm87NvFp4A8W4GateAdmissionWeightBytes);
  auto* const maximum_beta =
      beta + kSm87NvFp4A8W4GateAdmissionScaleElements;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  cudaError_t status =
      cudaMemsetAsync(maximum_beta, 0, sizeof(float), stream);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  recode_canonical_nvfp4_to_symmetric_w4_kernel
      <<<kPrepareBlocks, kThreads, 0U, stream>>>(
          canonical_weight, canonical_block_scales,
          canonical_weight_scale_2, canonical_qweight, beta, maximum_beta);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  compress_and_permute_scales_kernel
      <<<kPrepareBlocks, kThreads, 0U, stream>>>(
          beta, maximum_beta, marlin_integer_scales, rho);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, false, true>
      <<<static_cast<unsigned int>(kSm87NvFp4A8W4GateAdmissionSmCount),
         marlin::repack_threads,
         kSm87NvFp4A8W4GateAdmissionRepackSharedBytes, stream>>>(
          canonical_qweight, nullptr,
          reinterpret_cast<std::uint32_t*>(marlin_weight),
          static_cast<int>(kSm87NvFp4A8W4GateAdmissionColumns),
          static_cast<int>(kSm87NvFp4A8W4GateAdmissionRows));
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_dynamic_a8_gate_m512_admission_cuda(
    const std::uint16_t* const activation,
    const float* const rho,
    std::int8_t* const quantized_activation,
    float* const row_scales,
    const bool enable_approximate_admission,
    void* const cuda_stream) noexcept {
  if (!enable_approximate_admission) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!aligned(activation, alignof(std::uint16_t)) ||
      !aligned(rho, alignof(float)) ||
      !aligned(quantized_activation, 16U) ||
      !aligned(row_scales, alignof(float))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_fixed_device();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  dynamic_per_row_a8_kernel
      <<<static_cast<unsigned int>(kSm87NvFp4A8W4GateAdmissionTokens),
         kThreads, 0U, stream>>>(activation, rho, quantized_activation,
                                row_scales);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_nvfp4_a8w4_gate_m512_admission_cuda(
    const std::int8_t* const quantized_activation,
    const float* const row_scales,
    const std::uint8_t* const marlin_weight,
    const std::uint16_t* const marlin_integer_scales,
    std::uint16_t* const output,
    float* const reduction_workspace,
    std::int32_t* const locks,
    const bool enable_approximate_admission,
    void* const cuda_stream) noexcept {
  if (!enable_approximate_admission) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!aligned(quantized_activation, 16U) ||
      !aligned(row_scales, alignof(float)) ||
      !aligned(marlin_weight, 16U) ||
      !aligned(marlin_integer_scales, 16U) || !aligned(output, 16U) ||
      !aligned(reduction_workspace, 16U) ||
      !aligned(locks, alignof(std::int32_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_fixed_device();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }

  const auto kernel = a8w4_gate_kernel();
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87NvFp4A8W4GateAdmissionDynamicSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  kernel<<<static_cast<unsigned int>(kSm87NvFp4A8W4GateAdmissionSmCount),
           kThreads, kSm87NvFp4A8W4GateAdmissionDynamicSharedBytes, stream>>>(
      reinterpret_cast<const int4*>(quantized_activation),
      reinterpret_cast<const int4*>(marlin_weight),
      reinterpret_cast<int4*>(output),
      reinterpret_cast<int4*>(reduction_workspace), nullptr, row_scales,
      reinterpret_cast<const int4*>(marlin_integer_scales), nullptr, nullptr,
      nullptr, static_cast<int>(kSm87NvFp4A8W4GateAdmissionGroups),
      static_cast<int>(kSm87NvFp4A8W4GateAdmissionTokens),
      static_cast<int>(kSm87NvFp4A8W4GateAdmissionRows),
      static_cast<int>(kSm87NvFp4A8W4GateAdmissionColumns),
      static_cast<int>(kSm87NvFp4A8W4GateAdmissionColumns), locks, false,
      false, true,
      static_cast<int>(kSm87NvFp4A8W4GateAdmissionDynamicSharedBytes));
  return static_cast<int>(cudaPeekAtLastError());
}

int query_sm87_nvfp4_a8w4_gate_m512_admission_resources_cuda(
    Sm87NvFp4A8W4GateAdmissionResources* const resources,
    const bool enable_approximate_admission) noexcept {
  if (!enable_approximate_admission || resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_fixed_device();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  const auto kernel = a8w4_gate_kernel();
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87NvFp4A8W4GateAdmissionDynamicSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, kThreads,
      kSm87NvFp4A8W4GateAdmissionDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes =
      static_cast<int>(attributes.sharedSizeBytes);
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
