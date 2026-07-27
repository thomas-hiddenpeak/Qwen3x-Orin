#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime {
namespace {

constexpr unsigned int kWholeSpanThreads = 256U;
constexpr unsigned int kWholeSpanC256 = 256U;
constexpr unsigned int kWholeSpanC512 = 512U;

// Test-only exact whole-span canary. The arithmetic body is deliberately kept
// in lockstep with the production exact-M16 register-state kernel, whose device
// function has internal linkage and therefore cannot be called from this
// isolated translation unit. Only the fixed token-loop extent differs. Keeping
// this copy outside q3x_kernels prevents the experiment from changing any
// production CUDA section, resource count, selector, or SASS.

[[nodiscard]] bool valid_gdn_dimensions(
    const GdnDimensions dimensions) noexcept {
  return dimensions.qk_head_count == kGdnQkHeadCount &&
         dimensions.value_head_count == kGdnValueHeadCount &&
         dimensions.head_dimension == kGdnHeadDimension;
}

[[nodiscard]] bool invalid_gdn_alias(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    const std::uint16_t* const state_output,
    const std::uint16_t* const output) noexcept {
  const bool output_alias =
      output == conv_qkv || output == a || output == b || output == A_log ||
      output == dt_bias || output == state_input || output == state_output;
  const bool state_input_alias =
      state_input == conv_qkv || state_input == a || state_input == b ||
      state_input == A_log || state_input == dt_bias;
  const bool state_output_alias =
      state_output == conv_qkv || state_output == a || state_output == b ||
      state_output == A_log || state_output == dt_bias;
  return output_alias || state_input_alias || state_output_alias;
}

__device__ __forceinline__ float decode_bf16_device(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_device(
    const float value) {
  unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ std::uint32_t encode_bf16_pair_device(
    const float first,
    const float second) {
  const unsigned int first_bits = __float_as_uint(first);
  const unsigned int second_bits = __float_as_uint(second);
  const __nv_bfloat162_raw converted = static_cast<__nv_bfloat162_raw>(
      __floats2bfloat162_rn(first, second));
  std::uint32_t packed = static_cast<std::uint32_t>(converted.x) |
                         (static_cast<std::uint32_t>(converted.y) << 16U);
  if ((first_bits & 0x7fffffffU) > 0x7f800000U) {
    const std::uint32_t nan_bits = (first_bits >> 16U) | 0x0040U;
    packed = (packed & 0xffff0000U) | nan_bits;
  }
  if ((second_bits & 0x7fffffffU) > 0x7f800000U) {
    const std::uint32_t nan_bits = (second_bits >> 16U) | 0x0040U;
    packed = (packed & 0x0000ffffU) | (nan_bits << 16U);
  }
  return packed;
}

__device__ __forceinline__ float stable_softplus_device(const float value) {
  return value > 20.0F ? value : log1pf(expf(value));
}

__device__ __forceinline__ float stable_sigmoid_device(const float value) {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + expf(-value));
  }
  const float exponential = expf(value);
  return exponential / (1.0F + exponential);
}

template <unsigned int TokenCount>
__launch_bounds__(kWholeSpanThreads, 4) __global__ void
gated_delta_net_update_whole_span_register_state_kernel(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    std::uint16_t* const output) {
  static_assert(TokenCount == kWholeSpanC256 || TokenCount == kWholeSpanC512);
  constexpr unsigned int kWarpSize = 32U;
  constexpr unsigned int kWarpsPerBlock = 8U;
  constexpr unsigned int kRowsPerWarpBatch = 8U;
  constexpr unsigned int kBatchRowOffset = 64U;
  constexpr unsigned int kFullWarpMask = 0xffffffffU;
  constexpr std::size_t kKeysPerLane =
      kGdnHeadDimension / static_cast<std::size_t>(kWarpSize);
  constexpr std::size_t kScratchRowStride = kGdnHeadDimension + 1U;
  static_assert(kKeysPerLane == 4U);
  static_assert(kWarpsPerBlock * kRowsPerWarpBatch * 2U ==
                kGdnHeadDimension);
  static_assert(kScratchRowStride == 129U);

  __shared__ float normalized_q[kGdnHeadDimension];
  __shared__ float normalized_k[kGdnHeadDimension];
  __shared__ float recurrence_scalars[2];
  __shared__ float
      row_scratch[kWarpsPerBlock * kRowsPerWarpBatch * kScratchRowStride];

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const std::size_t value_head = blockIdx.x;
  const std::size_t qk_head = value_head / 3U;
  const std::size_t head_state_offset =
      value_head * kGdnHeadDimension * kGdnHeadDimension;
  const std::size_t warp_first_row =
      static_cast<std::size_t>(warp) * kRowsPerWarpBatch;
  constexpr std::size_t kKOffset = kGdnQElements;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

  constexpr std::size_t kKeyPairsPerLane = kKeysPerLane / 2U;
  static_assert(kKeyPairsPerLane == 2U);
  std::uint32_t lower_state_words[kRowsPerWarpBatch][kKeyPairsPerLane];
  std::uint32_t upper_state_words[kRowsPerWarpBatch][kKeyPairsPerLane];
#pragma unroll
  for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
#pragma unroll
    for (std::size_t pair = 0U; pair < kKeyPairsPerLane; ++pair) {
      const std::size_t first_item = pair * 2U;
      const std::size_t first_key =
          lane + first_item * static_cast<std::size_t>(kWarpSize);
      const std::size_t second_key = first_key + kWarpSize;
      const std::size_t lower_row_offset =
          head_state_offset +
          (warp_first_row + static_cast<std::size_t>(row)) *
              kGdnHeadDimension;
      const std::size_t upper_row_offset =
          lower_row_offset +
          static_cast<std::size_t>(kBatchRowOffset) * kGdnHeadDimension;
      const std::uint32_t lower_first =
          state_input[lower_row_offset + first_key];
      const std::uint32_t lower_second =
          state_input[lower_row_offset + second_key];
      const std::uint32_t upper_first =
          state_input[upper_row_offset + first_key];
      const std::uint32_t upper_second =
          state_input[upper_row_offset + second_key];
      lower_state_words[row][pair] =
          lower_first | (lower_second << 16U);
      upper_state_words[row][pair] =
          upper_first | (upper_second << 16U);
    }
  }

#pragma unroll 1
  for (unsigned int token = 0U; token < TokenCount; ++token) {
    if (token != 0U) {
      __syncthreads();
    }
    const std::size_t qkv_token_offset =
        static_cast<std::size_t>(token) * kGdnQkvChannels;
    const std::size_t scalar_token_offset =
        static_cast<std::size_t>(token) * kGdnValueHeadCount;
    const std::size_t output_token_offset =
        static_cast<std::size_t>(token) * kGdnVElements;

    if (thread < kGdnHeadDimension) {
      const std::size_t q_index =
          qkv_token_offset + qk_head * kGdnHeadDimension + thread;
      const std::size_t k_index =
          qkv_token_offset + kKOffset +
          qk_head * kGdnHeadDimension + thread;
      normalized_q[thread] = decode_bf16_device(conv_qkv[q_index]);
      normalized_k[thread] = decode_bf16_device(conv_qkv[k_index]);
    }
    __syncthreads();

    if (warp == 0U) {
      const float first = normalized_q[lane];
      const float second = normalized_q[lane + 32U];
      const float third = normalized_q[lane + 64U];
      const float fourth = normalized_q[lane + 96U];
      const float first_square = first * first;
      const float second_square = second * second;
      const float third_square = third * third;
      const float fourth_square = fourth * fourth;
      const float first_pair = first_square + third_square;
      const float second_pair = second_square + fourth_square;
      float warp_sum = first_pair + second_pair;
#pragma unroll
      for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
        const float shuffled =
            __shfl_down_sync(kFullWarpMask, warp_sum, stride);
        warp_sum += shuffled;
      }
      float q_scale = 0.0F;
      if (lane == 0U) {
        q_scale =
            rsqrtf(warp_sum + l2_epsilon) *
            rsqrtf(static_cast<float>(kGdnHeadDimension));
      }
      q_scale = __shfl_sync(kFullWarpMask, q_scale, 0U);
      normalized_q[lane] = first * q_scale;
      normalized_q[lane + 32U] = second * q_scale;
      normalized_q[lane + 64U] = third * q_scale;
      normalized_q[lane + 96U] = fourth * q_scale;
    } else if (warp == 1U) {
      const float first = normalized_k[lane];
      const float second = normalized_k[lane + 32U];
      const float third = normalized_k[lane + 64U];
      const float fourth = normalized_k[lane + 96U];
      const float first_square = first * first;
      const float second_square = second * second;
      const float third_square = third * third;
      const float fourth_square = fourth * fourth;
      const float first_pair = first_square + third_square;
      const float second_pair = second_square + fourth_square;
      float warp_sum = first_pair + second_pair;
#pragma unroll
      for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
        const float shuffled =
            __shfl_down_sync(kFullWarpMask, warp_sum, stride);
        warp_sum += shuffled;
      }
      float k_scale = 0.0F;
      if (lane == 0U) {
        k_scale = rsqrtf(warp_sum + l2_epsilon);
      }
      k_scale = __shfl_sync(kFullWarpMask, k_scale, 0U);
      normalized_k[lane] = first * k_scale;
      normalized_k[lane + 32U] = second * k_scale;
      normalized_k[lane + 64U] = third * k_scale;
      normalized_k[lane + 96U] = fourth * k_scale;
    }
    if (thread == 64U) {
      const float gate_input =
          decode_bf16_device(a[scalar_token_offset + value_head]) +
          decode_bf16_device(dt_bias[value_head]);
      const float g = -expf(decode_bf16_device(A_log[value_head])) *
                      stable_softplus_device(gate_input);
      recurrence_scalars[0] = expf(g);
      recurrence_scalars[1] = stable_sigmoid_device(
          decode_bf16_device(b[scalar_token_offset + value_head]));
    }
    __syncthreads();

    const float alpha = recurrence_scalars[0];
    const float beta = recurrence_scalars[1];
    float* const warp_scratch =
        row_scratch +
        static_cast<std::size_t>(warp * kRowsPerWarpBatch) *
            kScratchRowStride;

#pragma unroll 1
    for (unsigned int batch = 0U; batch < 2U; ++batch) {
      const std::size_t first_row =
          warp_first_row +
          static_cast<std::size_t>(batch * kBatchRowOffset);

#pragma unroll
      for (std::size_t pair = 0U; pair < kKeyPairsPerLane; ++pair) {
        const std::size_t first_item = pair * 2U;
        const std::size_t first_key =
            lane + first_item * static_cast<std::size_t>(kWarpSize);
        const std::size_t second_key = first_key + kWarpSize;
#pragma unroll
        for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
          const std::uint32_t word =
              batch == 0U ? lower_state_words[row][pair]
                          : upper_state_words[row][pair];
          const std::size_t scratch_row_offset =
              static_cast<std::size_t>(row) * kScratchRowStride;
          warp_scratch[scratch_row_offset + first_key] =
              alpha * decode_bf16_device(
                          static_cast<std::uint16_t>(word));
          warp_scratch[scratch_row_offset + second_key] =
              alpha * decode_bf16_device(
                          static_cast<std::uint16_t>(word >> 16U));
        }
      }
      __syncwarp(kFullWarpMask);

      float lane_prediction = 0.0F;
      if (lane < kRowsPerWarpBatch) {
        const float* const lane_scratch =
            warp_scratch + static_cast<std::size_t>(lane) *
                               kScratchRowStride;
#pragma unroll
        for (std::size_t key_dimension = 0U;
             key_dimension < kGdnHeadDimension; ++key_dimension) {
          lane_prediction =
              fmaf(lane_scratch[key_dimension], normalized_k[key_dimension],
                   lane_prediction);
        }
      }

      float lane_delta = 0.0F;
      if (lane < kRowsPerWarpBatch) {
        const std::size_t value_offset =
            qkv_token_offset + kVOffset +
            value_head * kGdnHeadDimension;
        const float lane_value = decode_bf16_device(
            conv_qkv[value_offset + first_row + lane]);
        lane_delta = (lane_value - lane_prediction) * beta;
      }
      float deltas[kRowsPerWarpBatch];
#pragma unroll
      for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
        deltas[row] = __shfl_sync(kFullWarpMask, lane_delta, row);
      }

#pragma unroll
      for (std::size_t pair = 0U; pair < kKeyPairsPerLane; ++pair) {
        const std::size_t first_item = pair * 2U;
        const std::size_t first_key =
            lane + first_item * static_cast<std::size_t>(kWarpSize);
        const std::size_t second_key = first_key + kWarpSize;
#pragma unroll
        for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
          const std::size_t scratch_row_offset =
              static_cast<std::size_t>(row) * kScratchRowStride;
          const std::size_t first_scratch_offset =
              scratch_row_offset + first_key;
          const std::size_t second_scratch_offset =
              scratch_row_offset + second_key;
          const float first_updated =
              fmaf(deltas[row], normalized_k[first_key],
                   warp_scratch[first_scratch_offset]);
          const float second_updated =
              fmaf(deltas[row], normalized_k[second_key],
                   warp_scratch[second_scratch_offset]);
          const std::uint32_t rounded =
              encode_bf16_pair_device(first_updated, second_updated);
          if (batch == 0U) {
            lower_state_words[row][pair] = rounded;
          } else {
            upper_state_words[row][pair] = rounded;
          }
          warp_scratch[first_scratch_offset] = first_updated;
          warp_scratch[second_scratch_offset] = second_updated;
        }
      }
      __syncwarp(kFullWarpMask);

      if (lane < kRowsPerWarpBatch) {
        const float* const lane_scratch =
            warp_scratch + static_cast<std::size_t>(lane) *
                               kScratchRowStride;
        float lane_result = 0.0F;
#pragma unroll
        for (std::size_t key_dimension = 0U;
             key_dimension < kGdnHeadDimension; ++key_dimension) {
          lane_result =
              fmaf(lane_scratch[key_dimension], normalized_q[key_dimension],
                   lane_result);
        }
        const std::size_t output_offset =
            output_token_offset + value_head * kGdnHeadDimension;
        output[output_offset + first_row + lane] =
            encode_bf16_device(lane_result);
      }
      __syncwarp(kFullWarpMask);
    }
  }

#pragma unroll
  for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
#pragma unroll
    for (std::size_t pair = 0U; pair < kKeyPairsPerLane; ++pair) {
      const std::size_t first_item = pair * 2U;
      const std::size_t first_key =
          lane + first_item * static_cast<std::size_t>(kWarpSize);
      const std::size_t second_key = first_key + kWarpSize;
      const std::size_t lower_row_offset =
          head_state_offset +
          (warp_first_row + static_cast<std::size_t>(row)) *
              kGdnHeadDimension;
      const std::size_t upper_row_offset =
          lower_row_offset +
          static_cast<std::size_t>(kBatchRowOffset) * kGdnHeadDimension;
      const std::uint32_t lower_word = lower_state_words[row][pair];
      const std::uint32_t upper_word = upper_state_words[row][pair];
      state_output[lower_row_offset + first_key] =
          static_cast<std::uint16_t>(lower_word);
      state_output[lower_row_offset + second_key] =
          static_cast<std::uint16_t>(lower_word >> 16U);
      state_output[upper_row_offset + first_key] =
          static_cast<std::uint16_t>(upper_word);
      state_output[upper_row_offset + second_key] =
          static_cast<std::uint16_t>(upper_word >> 16U);
    }
  }
}

template <unsigned int TokenCount>
int launch_whole_span(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_whole_span_register_state_kernel<TokenCount><<<
      static_cast<unsigned int>(kGdnValueHeadCount), kWholeSpanThreads, 0U,
      stream>>>(conv_qkv, a, b, A_log, dt_bias, state_input, state_output,
                l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

template <unsigned int TokenCount>
int query_whole_span_resources(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      gated_delta_net_update_whole_span_register_state_kernel<TokenCount>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      gated_delta_net_update_whole_span_register_state_kernel<TokenCount>,
      static_cast<int>(kWholeSpanThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int launch_gated_delta_net_update_whole_span_register_state_test_cuda(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    std::uint16_t* const output,
    const GdnDimensions dimensions,
    void* const cuda_stream) noexcept {
  if ((token_count != kWholeSpanC256 && token_count != kWholeSpanC512) ||
      !valid_gdn_dimensions(dimensions) || !std::isfinite(l2_epsilon) ||
      l2_epsilon <= 0.0F || conv_qkv == nullptr || a == nullptr ||
      b == nullptr || A_log == nullptr || dt_bias == nullptr ||
      state_input == nullptr || state_output == nullptr || output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (token_count == kWholeSpanC256) {
    return launch_whole_span<kWholeSpanC256>(
        conv_qkv, a, b, A_log, dt_bias, state_input, state_output, l2_epsilon,
        output, cuda_stream);
  }
  return launch_whole_span<kWholeSpanC512>(
      conv_qkv, a, b, A_log, dt_bias, state_input, state_output, l2_epsilon,
      output, cuda_stream);
}

int query_gated_delta_net_update_whole_span_register_state_resources_test_cuda(
    const std::size_t token_count,
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if ((token_count != kWholeSpanC256 && token_count != kWholeSpanC512) ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (token_count == kWholeSpanC256) {
    return query_whole_span_resources<kWholeSpanC256>(
        registers_per_thread, static_shared_bytes, local_bytes,
        maximum_threads_per_block, active_blocks_per_sm);
  }
  return query_whole_span_resources<kWholeSpanC512>(
      registers_per_thread, static_shared_bytes, local_bytes,
      maximum_threads_per_block, active_blocks_per_sm);
}

}  // namespace q3x::runtime
