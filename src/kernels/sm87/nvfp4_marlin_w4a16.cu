#include "q3x/kernels/sm87_nvfp4_marlin.h"

#include "third_party/vllm_marlin/gptq_marlin_repack_kernel.cuh"
#include "third_party/vllm_marlin/marlin_template.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kThreadMBlocks = 4;
constexpr int kThreadNBlocks = 16;
constexpr int kThreadKBlocks = 4;
constexpr int kStages = 4;
constexpr int kGroupBlocks = 1;
constexpr int kGroupSize = 16;
[[nodiscard]] bool aligned(const void* const pointer,
                           const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] float decode_e4m3fn_host(const std::uint8_t bits) noexcept {
  const std::uint8_t magnitude = bits & 0x7FU;
  const int exponent = static_cast<int>((magnitude >> 3U) & 0x0FU);
  const int mantissa = static_cast<int>(magnitude & 0x07U);
  if (exponent == 0x0F && mantissa == 0x07) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float value = exponent == 0
                          ? std::ldexp(static_cast<float>(mantissa), -9)
                          : std::ldexp(1.0F +
                                         static_cast<float>(mantissa) / 8.0F,
                                     exponent - 7);
  return (bits & 0x80U) != 0U ? -value : value;
}

__device__ __forceinline__ float decode_e4m3fn_device(
    const std::uint8_t bits) {
  const std::uint8_t magnitude = bits & 0x7FU;
  const int exponent = static_cast<int>((magnitude >> 3U) & 0x0FU);
  const int mantissa = static_cast<int>(magnitude & 0x07U);
  if (exponent == 0x0F && mantissa == 0x07) {
    return nanf("");
  }
  const float value = exponent == 0
                          ? ldexpf(static_cast<float>(mantissa), -9)
                          : ldexpf(1.0F +
                                      static_cast<float>(mantissa) / 8.0F,
                                  exponent - 7);
  return (bits & 0x80U) != 0U ? -value : value;
}

// Coalesced [N,K/8] -> [K/8,N] uint32 transpose. With a second matrix the N
// rows are concatenated before transpose, exactly matching the tensor that
// vLLM builds with view(int32).T.contiguous() for a fused Gate+Up projection.
__global__ void transpose_modelopt_words_kernel(
    const std::uint32_t* const first,
    const std::uint32_t* const second, const int first_n, const int total_n,
    const int k_words, const bool interleave_gate_up,
    std::uint32_t* const output) {
  __shared__ std::uint32_t tile[32][33];
  const int input_k = static_cast<int>(blockIdx.x) * 32 + threadIdx.x;
  const int input_n = static_cast<int>(blockIdx.y) * 32 + threadIdx.y;

#pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    const int row = input_n + offset;
    if (input_k < k_words && row < total_n) {
      const bool select_second =
          interleave_gate_up ? (row & 1) != 0 : row >= first_n;
      const std::uint32_t* const selected = select_second ? second : first;
      const int selected_row = interleave_gate_up
                                   ? row / 2
                                   : (select_second ? row - first_n : row);
      tile[threadIdx.y + offset][threadIdx.x] =
          selected[static_cast<std::size_t>(selected_row) * k_words + input_k];
    }
  }
  __syncthreads();

  const int output_n = static_cast<int>(blockIdx.y) * 32 + threadIdx.x;
  const int output_k = static_cast<int>(blockIdx.x) * 32 + threadIdx.y;
#pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    const int row = output_k + offset;
    if (output_n < total_n && row < k_words) {
      output[static_cast<std::size_t>(row) * total_n + output_n] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

__global__ void process_modelopt_scales_kernel(
    const std::uint8_t* const first,
    const std::uint8_t* const second, const int first_n, const int total_n,
    const int groups, const float scale_factor,
    const bool interleave_gate_up,
    std::uint8_t* const output) {
  const std::size_t total =
      static_cast<std::size_t>(total_n) * static_cast<std::size_t>(groups);
  for (std::size_t destination =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       destination < total;
       destination += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t group = destination / static_cast<std::size_t>(total_n);
    const int destination_n =
        static_cast<int>(destination % static_cast<std::size_t>(total_n));
    const int four_lane = destination_n & 3;
    const int before_four =
        (destination_n & ~3) + ((four_lane & 1) << 1) +
        ((four_lane & 2) >> 1);
    const int chunk = before_four / 64;
    const int permutation_lane = before_four & 63;
    const int source_n = chunk * 64 + (permutation_lane & 7) * 8 +
                         permutation_lane / 8;
    const bool select_second = interleave_gate_up
                                   ? (source_n & 1) != 0
                                   : source_n >= first_n;
    const std::uint8_t* const selected = select_second ? second : first;
    const int selected_n = interleave_gate_up
                               ? source_n / 2
                               : (select_second ? source_n - first_n
                                                : source_n);
    const std::uint8_t code =
        selected[static_cast<std::size_t>(selected_n) * groups + group];

    __half scaled = __float2half_rn(decode_e4m3fn_device(code) * scale_factor);
    scaled = __hmul_rn(scaled, __float2half_rn(128.0F));
    if (__hlt(scaled, __float2half_rn(2.0F))) {
      scaled = __float2half_rn(0.0F);
    }
    const __half_raw raw = scaled;
    output[destination] =
        static_cast<std::uint8_t>((static_cast<std::uint16_t>(raw.x) >> 7U) &
                                  0xFFU);
  }
}

__global__ void process_global_scale_kernel(
    const float* const canonical, const float scale_factor,
    float* const output) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    output[0] = ldexpf(canonical[0], 119) / scale_factor;
  }
}

__device__ __forceinline__ float decode_bf16_bits(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_bits(
    const float value) {
  std::uint32_t bits = __float_as_uint(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__global__ void merged_gate_up_silu_kernel(
    const std::uint16_t* const merged, const std::size_t elements,
    std::uint16_t* const output) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t token = index / kSm87NvFp4MarlinIntermediate;
    const std::size_t column = index % kSm87NvFp4MarlinIntermediate;
    const std::size_t gate_index =
        token * kSm87NvFp4MarlinGateUpOutput + column;
    const float gate = decode_bf16_bits(merged[gate_index]);
    const float up = decode_bf16_bits(
        merged[gate_index + kSm87NvFp4MarlinIntermediate]);
    output[index] = encode_bf16_bits(gate / (1.0F + expf(-gate)) * up);
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
          static_cast<int>(kSm87NvFp4MarlinSmCount) ||
      properties.sharedMemPerBlockOptin <
          kSm87NvFp4MarlinDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int prepare_projection(
    const std::uint8_t* const first_weight,
    const std::uint8_t* const second_weight,
    const std::uint8_t* const first_scales,
    const std::uint8_t* const second_scales,
    const float* const canonical_global_scale, const float scale_factor,
    const int first_n, const int total_n, const int input_size,
    std::uint8_t* const marlin_weight, std::uint8_t* const marlin_scales,
    float* const marlin_global_scale, void* const transpose_scratch,
    const std::size_t transpose_scratch_bytes,
    void* const cuda_stream, const bool interleave_gate_up) noexcept {
  const std::size_t required_weight_bytes =
      sm87_nvfp4_marlin_weight_bytes(static_cast<std::size_t>(total_n),
                                     static_cast<std::size_t>(input_size));
  if (!aligned(first_weight, 4U) ||
      (second_weight != nullptr && !aligned(second_weight, 4U)) ||
      first_scales == nullptr ||
      !aligned(canonical_global_scale, alignof(float)) ||
      !std::isfinite(scale_factor) || scale_factor < 1.0F ||
      !aligned(marlin_weight, 16U) || !aligned(marlin_scales, 16U) ||
      !aligned(marlin_global_scale, alignof(float)) ||
      !aligned(transpose_scratch, 16U) ||
      transpose_scratch_bytes < required_weight_bytes || first_n <= 0 ||
      total_n < first_n || input_size <= 0 || total_n % 64 != 0 ||
      input_size % 16 != 0 ||
      (interleave_gate_up && total_n != 2 * first_n)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((total_n != first_n) != (second_weight != nullptr) ||
      (second_weight != nullptr) != (second_scales != nullptr)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_fixed_device();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int k_words = input_size / 8;
  const dim3 transpose_block(32U, 8U, 1U);
  const dim3 transpose_grid(
      static_cast<unsigned int>((k_words + 31) / 32),
      static_cast<unsigned int>((total_n + 31) / 32), 1U);
  transpose_modelopt_words_kernel<<<transpose_grid, transpose_block, 0U,
                                    stream>>>(
      reinterpret_cast<const std::uint32_t*>(first_weight),
      reinterpret_cast<const std::uint32_t*>(second_weight), first_n, total_n,
      k_words, interleave_gate_up,
      reinterpret_cast<std::uint32_t*>(transpose_scratch));
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  status = cudaFuncSetAttribute(
      marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, false,
                                        false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87NvFp4MarlinDynamicSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, false, false>
      <<<static_cast<unsigned int>(kSm87NvFp4MarlinSmCount),
         marlin::repack_threads, kSm87NvFp4MarlinDynamicSharedBytes, stream>>>(
          reinterpret_cast<const std::uint32_t*>(transpose_scratch), nullptr,
          reinterpret_cast<std::uint32_t*>(marlin_weight), input_size,
          total_n);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  constexpr int kScaleThreads = 256;
  const std::size_t scale_count =
      sm87_nvfp4_marlin_scale_bytes(static_cast<std::size_t>(total_n),
                                    static_cast<std::size_t>(input_size));
  const unsigned int scale_blocks = static_cast<unsigned int>(
      std::min<std::size_t>((scale_count + kScaleThreads - 1U) /
                                kScaleThreads,
                            4096U));
  process_modelopt_scales_kernel<<<scale_blocks, kScaleThreads, 0U, stream>>>(
      first_scales, second_scales, first_n, total_n, input_size / kGroupSize,
      scale_factor, interleave_gate_up, marlin_scales);
  process_global_scale_kernel<<<1U, 1U, 0U, stream>>>(
      canonical_global_scale, scale_factor, marlin_global_scale);
  return static_cast<int>(cudaPeekAtLastError());
}

template <bool FusedGateUp, int ThreadMBlocks, int ThreadNBlocks,
          int ThreadKBlocks, bool MBlockSize8>
[[nodiscard]] int launch_marlin_specialization(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale, const std::size_t token_count,
    const int output_size, const int input_size, std::uint16_t* const output,
    std::uint16_t* const fused_gate_up_output, float* const c_tmp,
    std::int32_t* const locks,
    void* const cuda_stream) noexcept {
  const auto kernel =
      marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                     vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                     ThreadMBlocks, ThreadNBlocks, ThreadKBlocks, MBlockSize8,
                     kStages, kGroupBlocks, false, FusedGateUp>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87NvFp4MarlinDynamicSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  kernel<<<static_cast<unsigned int>(kSm87NvFp4MarlinSmCount), kThreads,
           kSm87NvFp4MarlinDynamicSharedBytes, stream>>>(
      reinterpret_cast<const int4*>(input),
      reinterpret_cast<const int4*>(marlin_weight),
      reinterpret_cast<int4*>(output), reinterpret_cast<int4*>(c_tmp),
      reinterpret_cast<const int4*>(fused_gate_up_output), nullptr,
      reinterpret_cast<const int4*>(marlin_scales),
      marlin_global_scale, nullptr, nullptr, input_size / kGroupSize,
      static_cast<int>(token_count), output_size, input_size, input_size, locks,
      false, false, true,
      static_cast<int>(kSm87NvFp4MarlinDynamicSharedBytes));
  return static_cast<int>(cudaPeekAtLastError());
}

template <bool FusedGateUp>
[[nodiscard]] int launch_marlin_segment(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale, const std::size_t token_count,
    const int output_size, const int input_size, std::uint16_t* const output,
    std::uint16_t* const fused_gate_up_output, float* const c_tmp,
    std::int32_t* const locks,
    void* const cuda_stream) noexcept {
  if (token_count <= 8U) {
    return launch_marlin_specialization<FusedGateUp, 1, 8, 8, true>(
        input, marlin_weight, marlin_scales, marlin_global_scale, token_count,
        output_size, input_size, output, fused_gate_up_output, c_tmp, locks,
        cuda_stream);
  }
  if (token_count <= 16U) {
    return launch_marlin_specialization<FusedGateUp, 1, 8, 8, false>(
        input, marlin_weight, marlin_scales, marlin_global_scale, token_count,
        output_size, input_size, output, fused_gate_up_output, c_tmp, locks,
        cuda_stream);
  }
  if (token_count <= kSm87NvFp4MarlinTailMaximumTokens) {
    return launch_marlin_specialization<FusedGateUp, 2, 16, 4, false>(
        input, marlin_weight, marlin_scales, marlin_global_scale, token_count,
        output_size, input_size, output, fused_gate_up_output, c_tmp, locks,
        cuda_stream);
  }
  return launch_marlin_specialization<FusedGateUp, 4, 16, 4, false>(
      input, marlin_weight, marlin_scales, marlin_global_scale, token_count,
      output_size, input_size, output, fused_gate_up_output, c_tmp, locks,
      cuda_stream);
}

template <bool FusedGateUp>
[[nodiscard]] int launch_fixed_marlin(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale, const std::size_t token_count,
    const int output_size, const int input_size, std::uint16_t* const output,
    std::uint16_t* const fused_gate_up_output, float* const c_tmp,
    std::int32_t* const locks,
    void* const cuda_stream) noexcept {
  if (!aligned(input, 16U) || !aligned(marlin_weight, 16U) ||
      !aligned(marlin_scales, 16U) ||
      !aligned(marlin_global_scale, alignof(float)) ||
      !aligned(output, 16U) || !aligned(c_tmp, 16U) ||
      !aligned(locks, alignof(std::int32_t)) ||
      (FusedGateUp && !aligned(fused_gate_up_output, 8U)) ||
      !sm87_nvfp4_marlin_supports_token_count(token_count)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_fixed_device();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }

  const Sm87NvFp4MarlinExecutionPlan plan =
      sm87_nvfp4_marlin_execution_plan(token_count);
  const int primary_status = launch_marlin_segment<FusedGateUp>(
      input, marlin_weight, marlin_scales, marlin_global_scale,
      plan.primary_tokens, output_size, input_size, output,
      fused_gate_up_output, c_tmp, locks, cuda_stream);
  if (primary_status != static_cast<int>(cudaSuccess) ||
      plan.remainder_tokens == 0U) {
    return primary_status;
  }

  const std::size_t input_offset =
      plan.primary_tokens * static_cast<std::size_t>(input_size);
  const std::size_t output_offset =
      plan.primary_tokens * static_cast<std::size_t>(output_size);
  return launch_marlin_segment<FusedGateUp>(
      input + input_offset, marlin_weight, marlin_scales,
      marlin_global_scale, plan.remainder_tokens, output_size, input_size,
      output + output_offset,
      fused_gate_up_output +
          plan.primary_tokens * static_cast<std::size_t>(output_size / 2),
      c_tmp, locks, cuda_stream);
}

}  // namespace

bool derive_sm87_nvfp4_marlin_scale_factor(
    const std::uint8_t* const first_scales,
    const std::size_t first_scale_bytes,
    const std::uint8_t* const second_scales,
    const std::size_t second_scale_bytes, float* const scale_factor) noexcept {
  if (first_scales == nullptr || first_scale_bytes == 0U ||
      scale_factor == nullptr ||
      ((second_scales == nullptr) != (second_scale_bytes == 0U))) {
    return false;
  }
  float maximum = 0.0F;
  const auto scan = [&maximum](const std::uint8_t* const scales,
                               const std::size_t bytes) {
    for (std::size_t index = 0U; index < bytes; ++index) {
      const float value = decode_e4m3fn_host(scales[index]);
      if (!std::isfinite(value) || value < 0.0F) {
        return false;
      }
      maximum = std::max(maximum, value);
    }
    return true;
  };
  if (!scan(first_scales, first_scale_bytes) ||
      (second_scales != nullptr &&
       !scan(second_scales, second_scale_bytes))) {
    return false;
  }
  float factor = 1.0F;
  if (maximum > 0.0F && maximum < 448.0F) {
    factor = std::exp2(std::floor(std::log2(448.0F / maximum)));
  }
  if (!std::isfinite(factor) || factor < 1.0F) {
    return false;
  }
  *scale_factor = factor;
  return true;
}

int prepare_sm87_nvfp4_marlin_gate_up_cuda(
    const std::uint8_t* const canonical_gate_weight,
    const std::uint8_t* const canonical_up_weight,
    const std::uint8_t* const canonical_gate_scales,
    const std::uint8_t* const canonical_up_scales,
    const float* const canonical_shared_weight_scale_2_device,
    const float scale_factor, std::uint8_t* const marlin_weight,
    std::uint8_t* const marlin_scales, float* const marlin_global_scale,
    void* const transpose_scratch,
    const std::size_t transpose_scratch_bytes,
    void* const cuda_stream, const bool interleave_gate_up) noexcept {
  return prepare_projection(
      canonical_gate_weight, canonical_up_weight, canonical_gate_scales,
      canonical_up_scales, canonical_shared_weight_scale_2_device,
      scale_factor, static_cast<int>(kSm87NvFp4MarlinIntermediate),
      static_cast<int>(kSm87NvFp4MarlinGateUpOutput),
      static_cast<int>(kSm87NvFp4MarlinHidden), marlin_weight, marlin_scales,
      marlin_global_scale, transpose_scratch, transpose_scratch_bytes,
      cuda_stream, interleave_gate_up);
}

int prepare_sm87_nvfp4_marlin_down_cuda(
    const std::uint8_t* const canonical_weight,
    const std::uint8_t* const canonical_scales,
    const float* const canonical_weight_scale_2_device,
    const float scale_factor, std::uint8_t* const marlin_weight,
    std::uint8_t* const marlin_scales, float* const marlin_global_scale,
    void* const transpose_scratch,
    const std::size_t transpose_scratch_bytes,
    void* const cuda_stream) noexcept {
  return prepare_projection(
      canonical_weight, nullptr, canonical_scales, nullptr,
      canonical_weight_scale_2_device, scale_factor,
      static_cast<int>(kSm87NvFp4MarlinHidden),
      static_cast<int>(kSm87NvFp4MarlinHidden),
      static_cast<int>(kSm87NvFp4MarlinIntermediate), marlin_weight,
      marlin_scales, marlin_global_scale, transpose_scratch,
      transpose_scratch_bytes, cuda_stream, false);
}

int launch_sm87_nvfp4_marlin_gate_up_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* const merged_gate_up_output, float* const c_tmp,
    std::int32_t* const locks, void* const cuda_stream) noexcept {
  return launch_fixed_marlin<false>(
      input, marlin_weight, marlin_scales, marlin_global_scale,
      token_count,
      static_cast<int>(kSm87NvFp4MarlinGateUpOutput),
      static_cast<int>(kSm87NvFp4MarlinHidden), merged_gate_up_output, nullptr,
      c_tmp, locks, cuda_stream);
}

int launch_sm87_nvfp4_marlin_gate_up_epilogue_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* const merged_gate_up_workspace,
    std::uint16_t* const output, float* const c_tmp,
    std::int32_t* const locks, void* const cuda_stream) noexcept {
  return launch_fixed_marlin<true>(
      input, marlin_weight, marlin_scales, marlin_global_scale, token_count,
      static_cast<int>(kSm87NvFp4MarlinGateUpOutput),
      static_cast<int>(kSm87NvFp4MarlinHidden), merged_gate_up_workspace,
      output, c_tmp, locks, cuda_stream);
}

int launch_sm87_nvfp4_marlin_down_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale, const std::size_t token_count,
    std::uint16_t* const output, float* const c_tmp,
    std::int32_t* const locks,
    void* const cuda_stream) noexcept {
  return launch_fixed_marlin<false>(
      input, marlin_weight, marlin_scales, marlin_global_scale,
      token_count,
      static_cast<int>(kSm87NvFp4MarlinHidden),
      static_cast<int>(kSm87NvFp4MarlinIntermediate), output, nullptr, c_tmp,
      locks, cuda_stream);
}

int launch_sm87_nvfp4_marlin_gate_up_silu_cuda(
    const std::uint16_t* const merged_gate_up,
    const std::size_t token_count, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!aligned(merged_gate_up, alignof(std::uint16_t)) ||
      !aligned(output, alignof(std::uint16_t)) ||
      !sm87_nvfp4_marlin_supports_token_count(token_count)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned int kBlockThreads = 256U;
  const std::size_t elements =
      token_count * kSm87NvFp4MarlinIntermediate;
  const unsigned int required_blocks = static_cast<unsigned int>(
      (elements + kBlockThreads - 1U) / kBlockThreads);
  const unsigned int blocks = required_blocks < 4096U ? required_blocks : 4096U;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  merged_gate_up_silu_kernel<<<blocks, kBlockThreads, 0U, stream>>>(
      merged_gate_up, elements, output);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_nvfp4_marlin_gate_up_m512_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    std::uint16_t* const merged_gate_up_output, float* const c_tmp,
    std::int32_t* const locks, void* const cuda_stream) noexcept {
  return launch_sm87_nvfp4_marlin_gate_up_cuda(
      input, marlin_weight, marlin_scales, marlin_global_scale,
      kSm87NvFp4MarlinTokens, merged_gate_up_output, c_tmp, locks,
      cuda_stream);
}

int launch_sm87_nvfp4_marlin_down_m512_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale, std::uint16_t* const output,
    float* const c_tmp, std::int32_t* const locks,
    void* const cuda_stream) noexcept {
  return launch_sm87_nvfp4_marlin_down_cuda(
      input, marlin_weight, marlin_scales, marlin_global_scale,
      kSm87NvFp4MarlinTokens, output, c_tmp, locks, cuda_stream);
}

int launch_sm87_nvfp4_marlin_gate_up_silu_m512_cuda(
    const std::uint16_t* const merged_gate_up,
    std::uint16_t* const output, void* const cuda_stream) noexcept {
  return launch_sm87_nvfp4_marlin_gate_up_silu_cuda(
      merged_gate_up, kSm87NvFp4MarlinTokens, output, cuda_stream);
}

int query_sm87_nvfp4_marlin_m512_resources_cuda(
    Sm87NvFp4MarlinKernelResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto kernel =
      marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                     vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                     kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                     kStages, kGroupBlocks, false>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87NvFp4MarlinDynamicSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, kernel, kThreads, kSm87NvFp4MarlinDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes =
      static_cast<int>(attributes.sharedSizeBytes);
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active;
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
