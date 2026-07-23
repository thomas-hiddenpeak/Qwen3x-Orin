#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime {
namespace {

constexpr unsigned int kRegisterStateThreads = 256U;
constexpr unsigned int kRegisterStateTokenCount = 16U;

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

// Exact-M16 experiment: each lane owns four keys from the same eight rows in
// both row batches assigned to its warp. The low/high BF16 halves of each word
// hold matching (row,key) elements from the lower/upper batch respectively.
// State is loaded once, rounded back into these words at every token boundary,
// and written to global memory only after token 15.
__launch_bounds__(kRegisterStateThreads, 4)
__global__ void gated_delta_net_update_warp_eight_row_register_state_m16_kernel(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    std::uint16_t* const output) {
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

  std::uint32_t state_words[kRowsPerWarpBatch][kKeysPerLane];
#pragma unroll
  for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
#pragma unroll
    for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
      const std::size_t key_dimension =
          lane + item * static_cast<std::size_t>(kWarpSize);
      const std::size_t lower_offset =
          head_state_offset +
          (warp_first_row + static_cast<std::size_t>(row)) *
              kGdnHeadDimension +
          key_dimension;
      const std::size_t upper_offset =
          lower_offset +
          static_cast<std::size_t>(kBatchRowOffset) * kGdnHeadDimension;
      const std::uint32_t lower = state_input[lower_offset];
      const std::uint32_t upper = state_input[upper_offset];
      state_words[row][item] = lower | (upper << 16U);
    }
  }

#pragma unroll 1
  for (unsigned int token = 0U; token < kRegisterStateTokenCount; ++token) {
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
      for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
        const std::size_t key_dimension =
            lane + item * static_cast<std::size_t>(kWarpSize);
#pragma unroll
        for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
          const std::uint32_t word = state_words[row][item];
          const std::uint16_t state_bits = static_cast<std::uint16_t>(
              batch == 0U ? word : (word >> 16U));
          const std::size_t scratch_offset =
              static_cast<std::size_t>(row) * kScratchRowStride +
              key_dimension;
          warp_scratch[scratch_offset] =
              alpha * decode_bf16_device(state_bits);
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
      for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
        const std::size_t key_dimension =
            lane + item * static_cast<std::size_t>(kWarpSize);
#pragma unroll
        for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
          const std::size_t scratch_offset =
              static_cast<std::size_t>(row) * kScratchRowStride +
              key_dimension;
          const float updated =
              fmaf(deltas[row], normalized_k[key_dimension],
                   warp_scratch[scratch_offset]);
          const std::uint32_t rounded = encode_bf16_device(updated);
          const std::uint32_t old_word = state_words[row][item];
          state_words[row][item] =
              batch == 0U ? (old_word & 0xffff0000U) | rounded
                          : (old_word & 0x0000ffffU) | (rounded << 16U);
          warp_scratch[scratch_offset] = updated;
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
    for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
      const std::size_t key_dimension =
          lane + item * static_cast<std::size_t>(kWarpSize);
      const std::size_t lower_offset =
          head_state_offset +
          (warp_first_row + static_cast<std::size_t>(row)) *
              kGdnHeadDimension +
          key_dimension;
      const std::size_t upper_offset =
          lower_offset +
          static_cast<std::size_t>(kBatchRowOffset) * kGdnHeadDimension;
      const std::uint32_t word = state_words[row][item];
      state_output[lower_offset] = static_cast<std::uint16_t>(word);
      state_output[upper_offset] =
          static_cast<std::uint16_t>(word >> 16U);
    }
  }
}

}  // namespace

int launch_gated_delta_net_update_tile_warp_eight_row_register_state_m16_test_cuda(
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
    void* const cuda_stream) noexcept {
  if (token_count != kRegisterStateTokenCount ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr ||
      A_log == nullptr || dt_bias == nullptr || state_input == nullptr ||
      state_output == nullptr || output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_warp_eight_row_register_state_m16_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kRegisterStateThreads,
      0U, stream>>>(conv_qkv, a, b, A_log, dt_bias, state_input,
                    state_output, l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int query_gated_delta_net_update_warp_eight_row_register_state_m16_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      gated_delta_net_update_warp_eight_row_register_state_m16_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      gated_delta_net_update_warp_eight_row_register_state_m16_kernel,
      static_cast<int>(kRegisterStateThreads), 0U);
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

}  // namespace q3x::runtime
