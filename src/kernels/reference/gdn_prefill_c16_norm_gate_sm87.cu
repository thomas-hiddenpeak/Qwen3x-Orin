#include "gdn_prefill_c16_norm_gate_sm87.h"

#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace q3x::runtime {

#if defined(Q3X_GDN_C16_NORM_GATE_STANDALONE)
// Existing test-only production resource query. This candidate translation
// unit is linked into its own executable and does not enter q3x_kernels.
[[nodiscard]] int
query_gated_delta_net_update_warp_eight_row_register_state_m16_resources_test_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;
#endif

#if !defined(Q3X_GDN_C16_NORM_GATE_USE_LINKED_IMPLEMENTATION)
namespace gdn_prefill_c16_norm_gate_test_detail {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kWarpsPerBlock = kThreads / kWarpSize;
constexpr unsigned int kRowsPerWarpBatch = 8U;
constexpr unsigned int kBatchRowOffset = 64U;
constexpr unsigned int kTokenCount = 16U;
constexpr unsigned int kFullWarpMask = 0xffffffffU;
constexpr std::size_t kKeysPerLane = kGdnHeadDimension / kWarpSize;
constexpr std::size_t kKeyPairsPerLane = kKeysPerLane / 2U;
constexpr std::size_t kScratchRowStride = kGdnHeadDimension + 1U;
constexpr std::size_t kRawRowStride = kGdnHeadDimension + 1U;
constexpr std::size_t kRawTileElements = kTokenCount * kRawRowStride;
constexpr std::size_t kProductionSharedBytes =
    (2U * kGdnHeadDimension + 2U +
     kWarpsPerBlock * kRowsPerWarpBatch * kScratchRowStride) *
    sizeof(float);
constexpr std::size_t kSharedBoundaryLogicalBytes =
    kProductionSharedBytes + kRawTileElements * sizeof(std::uint16_t);

static_assert(kWarpsPerBlock * kRowsPerWarpBatch * 2U ==
              kGdnHeadDimension);
static_assert(kKeysPerLane == 4U);
static_assert(kKeyPairsPerLane == 2U);
static_assert(kScratchRowStride == 129U);
static_assert(kRawRowStride == 129U);
static_assert(kRawTileElements * sizeof(std::uint16_t) == 4128U);
static_assert(kProductionSharedBytes == 34056U);
static_assert(kSharedBoundaryLogicalBytes == 38184U);

[[nodiscard]] bool invalid_alias(
    const std::uint16_t* const conv_qkv, const std::uint16_t* const a,
    const std::uint16_t* const b, const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    const std::uint16_t* const state_output,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    const std::uint16_t* const output) noexcept {
  const bool output_alias =
      output == conv_qkv || output == a || output == b ||
      output == A_log || output == dt_bias || output == state_input ||
      output == state_output || output == norm_weight ||
      output == silu_gate;
  const bool state_input_alias =
      state_input == conv_qkv || state_input == a || state_input == b ||
      state_input == A_log || state_input == dt_bias ||
      state_input == norm_weight || state_input == silu_gate;
  const bool state_output_alias =
      state_output == conv_qkv || state_output == a || state_output == b ||
      state_output == A_log || state_output == dt_bias ||
      state_output == norm_weight || state_output == silu_gate;
  return output_alias || state_input_alias || state_output_alias;
}

__device__ __forceinline__ float decode_bf16_device(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_device(
    const float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ std::uint32_t encode_bf16_pair_device(
    const float first, const float second) {
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

// The arithmetic through raw-output BF16 encoding is a literal test-only copy
// of the production exact-C16 register-state kernel. The template changes only
// where that rounded raw output resides before the already-defined plain
// RMSNorm/SiLU epilogue. kStoreRawDebug is compiled only for correctness and
// has no performance or resource authority.
template <bool kUseSharedBoundary, bool kStoreRawDebug>
__launch_bounds__(kThreads, 4)
__global__ void gated_delta_net_update_c16_plain_rms_norm_silu_gate_kernel(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    const float norm_epsilon,
    std::uint16_t* const output,
    std::uint16_t* const raw_debug) {
  __shared__ float normalized_q[kGdnHeadDimension];
  __shared__ float normalized_k[kGdnHeadDimension];
  __shared__ float recurrence_scalars[2];
  __shared__ float
      row_scratch[kWarpsPerBlock * kRowsPerWarpBatch * kScratchRowStride];
  __shared__ std::uint16_t
      raw_output[kUseSharedBoundary ? kRawTileElements : 1U];

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
  for (unsigned int token = 0U; token < kTokenCount; ++token) {
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
        q_scale = rsqrtf(warp_sum + l2_epsilon) *
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
            warp_scratch +
            static_cast<std::size_t>(lane) * kScratchRowStride;
#pragma unroll
        for (std::size_t key_dimension = 0U;
             key_dimension < kGdnHeadDimension; ++key_dimension) {
          lane_prediction =
              fmaf(lane_scratch[key_dimension],
                   normalized_k[key_dimension], lane_prediction);
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
            warp_scratch +
            static_cast<std::size_t>(lane) * kScratchRowStride;
        float lane_result = 0.0F;
#pragma unroll
        for (std::size_t key_dimension = 0U;
             key_dimension < kGdnHeadDimension; ++key_dimension) {
          lane_result =
              fmaf(lane_scratch[key_dimension],
                   normalized_q[key_dimension], lane_result);
        }
        const std::size_t output_offset =
            output_token_offset + value_head * kGdnHeadDimension +
            first_row + lane;
        const std::uint16_t rounded = encode_bf16_device(lane_result);
        if constexpr (kUseSharedBoundary) {
          raw_output[static_cast<std::size_t>(token) * kRawRowStride +
                     first_row + lane] = rounded;
        } else {
          output[output_offset] = rounded;
        }
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
  __syncthreads();

#pragma unroll 1
  for (unsigned int token = 0U; token < kTokenCount; ++token) {
    const std::size_t output_offset =
        static_cast<std::size_t>(token) * kGdnVElements +
        value_head * kGdnHeadDimension;
    float raw_value = 0.0F;
    std::uint16_t raw_word = 0U;
    if (thread < kGdnHeadDimension) {
      if constexpr (kUseSharedBoundary) {
        raw_word = raw_output[static_cast<std::size_t>(token) *
                                  kRawRowStride +
                              thread];
      } else {
        raw_word = output[output_offset + thread];
      }
      raw_value = decode_bf16_device(raw_word);
    }
    float sum = 0.0F;
    if (thread < kGdnHeadDimension) {
      sum = fmaf(raw_value, raw_value, sum);
    }
    row_scratch[thread] = sum;
    __syncthreads();
#pragma unroll
    for (unsigned int stride = kThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (thread < stride) {
        row_scratch[thread] += row_scratch[thread + stride];
      }
      __syncthreads();
    }
    const float inverse_rms =
        rsqrtf(row_scratch[0] /
                   static_cast<float>(kGdnHeadDimension) +
               norm_epsilon);
    if (thread < kGdnHeadDimension) {
      if constexpr (kStoreRawDebug) {
        raw_debug[output_offset + thread] = raw_word;
      }
      float value = raw_value * inverse_rms *
                    decode_bf16_device(norm_weight[thread]);
      const float gate_value =
          decode_bf16_device(silu_gate[output_offset + thread]);
      value *= gate_value / (1.0F + expf(-gate_value));
      output[output_offset + thread] = encode_bf16_device(value);
    }
    __syncthreads();
  }
}

template <bool kUseSharedBoundary, bool kStoreRawDebug>
[[nodiscard]] int launch_impl(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    const float norm_epsilon,
    std::uint16_t* const output,
    std::uint16_t* const raw_debug,
    void* const cuda_stream) noexcept {
  if (token_count != kTokenCount ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      !std::isfinite(norm_epsilon) || norm_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr ||
      A_log == nullptr || dt_bias == nullptr || state_input == nullptr ||
      state_output == nullptr || norm_weight == nullptr ||
      silu_gate == nullptr || output == nullptr ||
      (kStoreRawDebug && raw_debug == nullptr) ||
      invalid_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                    state_output, norm_weight, silu_gate, output) ||
      (kStoreRawDebug &&
       (raw_debug == conv_qkv || raw_debug == a || raw_debug == b ||
        raw_debug == A_log || raw_debug == dt_bias ||
        raw_debug == state_input || raw_debug == state_output ||
        raw_debug == norm_weight || raw_debug == silu_gate ||
        raw_debug == output))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_c16_plain_rms_norm_silu_gate_kernel<
      kUseSharedBoundary, kStoreRawDebug>
      <<<static_cast<unsigned int>(kGdnValueHeadCount), kThreads, 0U,
         stream>>>(conv_qkv, a, b, A_log, dt_bias, state_input,
                   state_output, l2_epsilon, norm_weight, silu_gate,
                   norm_epsilon, output, raw_debug);
  return static_cast<int>(cudaGetLastError());
}

template <bool kUseSharedBoundary>
[[nodiscard]] int query_resources_impl(
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
      gated_delta_net_update_c16_plain_rms_norm_silu_gate_kernel<
          kUseSharedBoundary, false>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      gated_delta_net_update_c16_plain_rms_norm_silu_gate_kernel<
          kUseSharedBoundary, false>,
      static_cast<int>(kThreads), 0U);
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

[[nodiscard]] int launch_shared_boundary(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    const float norm_epsilon,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  return launch_impl<true, false>(
      conv_qkv, token_count, a, b, A_log, dt_bias, state_input,
      state_output, l2_epsilon, norm_weight, silu_gate, norm_epsilon,
      output, nullptr, cuda_stream);
}

[[nodiscard]] int launch_shared_boundary_debug(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    const float norm_epsilon,
    std::uint16_t* const output,
    std::uint16_t* const raw_debug,
    void* const cuda_stream) noexcept {
  return launch_impl<true, true>(
      conv_qkv, token_count, a, b, A_log, dt_bias, state_input,
      state_output, l2_epsilon, norm_weight, silu_gate, norm_epsilon,
      output, raw_debug, cuda_stream);
}

[[nodiscard]] int launch_global_boundary(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    const float norm_epsilon,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  return launch_impl<false, false>(
      conv_qkv, token_count, a, b, A_log, dt_bias, state_input,
      state_output, l2_epsilon, norm_weight, silu_gate, norm_epsilon,
      output, nullptr, cuda_stream);
}

[[nodiscard]] int query_shared_resources(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  return query_resources_impl<true>(
      registers_per_thread, static_shared_bytes, local_bytes,
      maximum_threads_per_block, active_blocks_per_sm);
}

[[nodiscard]] int query_global_resources(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  return query_resources_impl<false>(
      registers_per_thread, static_shared_bytes, local_bytes,
      maximum_threads_per_block, active_blocks_per_sm);
}

}  // namespace gdn_prefill_c16_norm_gate_test_detail
#endif
}  // namespace q3x::runtime

#if defined(Q3X_GDN_C16_NORM_GATE_STANDALONE)
namespace {

constexpr std::size_t kHostTokenCount = 16U;
constexpr std::size_t kOutputElements =
    kHostTokenCount * q3x::runtime::kGdnVElements;
constexpr std::size_t kNormHeadCount =
    kHostTokenCount * q3x::runtime::kGdnValueHeadCount;
constexpr float kL2Epsilon = 1.0e-6F;
constexpr float kNormEpsilon = 1.0e-5F;

class TestContext {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] bool cuda_ok(const cudaError_t status,
                             const std::string& operation) {
    expect(status == cudaSuccess,
           operation + ": " + cudaGetErrorString(status));
    return status == cudaSuccess;
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <typename T>
class ManagedBuffer {
 public:
  ManagedBuffer() = default;
  ManagedBuffer(const ManagedBuffer&) = delete;
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  ~ManagedBuffer() {
    if (allocation_ != nullptr) {
      (void)cudaFree(allocation_);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t count) {
    count_ = count;
    const cudaError_t status = cudaMallocManaged(
        reinterpret_cast<void**>(&allocation_),
        (count + 2U * kGuardElements) * sizeof(T));
    if (status != cudaSuccess) {
      return status;
    }
    data_ = allocation_ + kGuardElements;
    std::fill_n(allocation_, kGuardElements, kGuardValue);
    std::fill_n(data_ + count_, kGuardElements, kGuardValue);
    return cudaSuccess;
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] T& operator[](const std::size_t index) noexcept {
    return data_[index];
  }
  [[nodiscard]] bool guards_intact() const noexcept {
    if (allocation_ == nullptr) {
      return false;
    }
    for (std::size_t index = 0U; index < kGuardElements; ++index) {
      if (allocation_[index] != kGuardValue ||
          data_[count_ + index] != kGuardValue) {
        return false;
      }
    }
    return true;
  }

 private:
  static constexpr std::size_t kGuardElements = 64U;
  static constexpr T kGuardValue = static_cast<T>(0x5a5aU);
  T* allocation_ = nullptr;
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

void fill_inputs(ManagedBuffer<std::uint16_t>& conv_qkv,
                 ManagedBuffer<std::uint16_t>& a,
                 ManagedBuffer<std::uint16_t>& b,
                 ManagedBuffer<std::uint16_t>& A_log,
                 ManagedBuffer<std::uint16_t>& dt_bias,
                 ManagedBuffer<std::uint16_t>& norm_weight,
                 ManagedBuffer<std::uint16_t>& silu_gate) {
  constexpr std::size_t kKOffset = q3x::runtime::kGdnQElements;
  constexpr std::size_t kVOffset =
      q3x::runtime::kGdnQElements + q3x::runtime::kGdnKElements;
  for (std::size_t token = 0U; token < kHostTokenCount; ++token) {
    const std::size_t qkv_offset = token * q3x::runtime::kGdnQkvChannels;
    const std::size_t scalar_offset =
        token * q3x::runtime::kGdnValueHeadCount;
    for (std::size_t head = 0U;
         head < q3x::runtime::kGdnQkHeadCount; ++head) {
      for (std::size_t dimension = 0U;
           dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
        const int q_centered =
            static_cast<int>((head * 11U + dimension * 3U + token * 5U) %
                             29U) -
            14;
        const int k_centered =
            static_cast<int>((head * 7U + dimension * 5U + token * 2U) %
                             31U) -
            15;
        conv_qkv[qkv_offset +
                 head * q3x::runtime::kGdnHeadDimension + dimension] =
            encode_bf16(static_cast<float>(q_centered) / 16.0F);
        conv_qkv[qkv_offset + kKOffset +
                 head * q3x::runtime::kGdnHeadDimension + dimension] =
            encode_bf16(static_cast<float>(k_centered) / 16.0F);
      }
    }
    for (std::size_t index = 0U;
         index < q3x::runtime::kGdnVElements; ++index) {
      const int centered =
          static_cast<int>((index * 13U + token * 7U) % 37U) - 18;
      conv_qkv[qkv_offset + kVOffset + index] =
          encode_bf16(static_cast<float>(centered) / 16.0F);
      const int gate_centered =
          static_cast<int>((index * 17U + token * 11U) % 41U) - 20;
      silu_gate[token * q3x::runtime::kGdnVElements + index] =
          encode_bf16(static_cast<float>(gate_centered) / 8.0F);
    }
    for (std::size_t head = 0U;
         head < q3x::runtime::kGdnValueHeadCount; ++head) {
      a[scalar_offset + head] = encode_bf16(
          static_cast<float>(static_cast<int>((head + token) % 9U) - 4) *
          0.25F);
      b[scalar_offset + head] = encode_bf16(
          static_cast<float>(static_cast<int>((head + 2U * token) % 11U) -
                             5) *
          0.5F);
    }
  }
  for (std::size_t head = 0U;
       head < q3x::runtime::kGdnValueHeadCount; ++head) {
    A_log[head] = encode_bf16(
        -1.5F + static_cast<float>(head % 5U) * 0.375F);
    dt_bias[head] = encode_bf16(
        -0.75F + static_cast<float>(head % 7U) * 0.125F);
  }
  for (std::size_t dimension = 0U;
       dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
    const int centered = static_cast<int>((dimension * 7U) % 23U) - 11;
    norm_weight[dimension] =
        encode_bf16(1.0F + static_cast<float>(centered) / 64.0F);
  }
  b[0U] = encode_bf16(20.0F);
  b[q3x::runtime::kGdnValueHeadCount + 1U] = encode_bf16(-20.0F);
  a[2U * q3x::runtime::kGdnValueHeadCount + 2U] = encode_bf16(25.0F);
  A_log[2U] = encode_bf16(4.0F);
  silu_gate[0U] = encode_bf16(20.0F);
  silu_gate[q3x::runtime::kGdnHeadDimension] = encode_bf16(-20.0F);
}

void fill_state(std::uint16_t* const state, const bool inject_nan) {
  for (std::size_t index = 0U;
       index < q3x::runtime::kGdnStateElements; ++index) {
    const int centered = static_cast<int>((index * 5U) % 23U) - 11;
    state[index] = encode_bf16(static_cast<float>(centered) / 512.0F);
  }
  if (!inject_nan) {
    return;
  }
  constexpr std::array<std::uint16_t, 4U> kNanWords{
      0x7f81U, 0xff81U, 0x7fc1U, 0xffc1U};
  constexpr std::array<std::size_t, 4U> kRows{0U, 1U, 64U, 65U};
  constexpr std::array<std::size_t, 4U> kKeys{0U, 32U, 64U, 96U};
  for (std::size_t head = 0U;
       head < q3x::runtime::kGdnValueHeadCount; ++head) {
    const std::size_t head_offset =
        head * q3x::runtime::kGdnHeadDimension *
        q3x::runtime::kGdnHeadDimension;
    for (std::size_t item = 0U; item < kNanWords.size(); ++item) {
      state[head_offset +
            kRows[item] * q3x::runtime::kGdnHeadDimension + kKeys[item]] =
          kNanWords[item];
    }
  }
}

void expect_bitwise(TestContext& test, const std::uint16_t* const actual,
                    const std::uint16_t* const expected,
                    const std::size_t count, const std::string& label) {
  std::size_t mismatches = 0U;
  std::size_t first = count;
  for (std::size_t index = 0U; index < count; ++index) {
    if (actual[index] != expected[index]) {
      first = first == count ? index : first;
      ++mismatches;
    }
  }
  test.expect(mismatches == 0U,
              label + " mismatches=" + std::to_string(mismatches) +
                  (first == count ? "" : " first=" + std::to_string(first)));
}

struct KernelTopology {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  std::size_t total_nodes = 0U;
  std::size_t kernel_nodes = 0U;
  cudaKernelNodeParams kernel{};

  KernelTopology() = default;
  KernelTopology(const KernelTopology&) = delete;
  KernelTopology& operator=(const KernelTopology&) = delete;

  ~KernelTopology() {
    if (executable != nullptr) {
      (void)cudaGraphExecDestroy(executable);
    }
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
  }
};

template <typename Launch>
void capture_one_kernel(TestContext& test, cudaStream_t stream,
                        const std::string& label, Launch&& launch,
                        KernelTopology& topology) {
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
      label + " begin capture");
  if (!ready) {
    return;
  }
  test.expect(static_cast<cudaError_t>(launch()) == cudaSuccess,
              label + " captured launch");
  ready = test.cuda_ok(cudaStreamEndCapture(stream, &topology.graph),
                       label + " end capture");
  if (!ready || topology.graph == nullptr) {
    return;
  }
  ready = test.cuda_ok(
      cudaGraphGetNodes(topology.graph, nullptr, &topology.total_nodes),
      label + " count nodes");
  if (!ready) {
    return;
  }
  std::vector<cudaGraphNode_t> nodes(topology.total_nodes);
  std::size_t copied = nodes.size();
  ready = test.cuda_ok(
      cudaGraphGetNodes(topology.graph, nodes.data(), &copied),
      label + " get nodes");
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    ready = test.cuda_ok(cudaGraphNodeGetType(node, &type),
                         label + " node type") &&
            ready;
    if (type == cudaGraphNodeTypeKernel) {
      ++topology.kernel_nodes;
      ready = test.cuda_ok(cudaGraphKernelNodeGetParams(node, &topology.kernel),
                           label + " kernel params") &&
              ready;
    }
  }
  test.expect(topology.total_nodes == 1U && topology.kernel_nodes == 1U,
              label + " captures exactly one kernel");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      cudaGraphInstantiate(&topology.executable, topology.graph, nullptr,
                           nullptr, 0U),
      label + " instantiate");
  if (ready) {
    ready = test.cuda_ok(cudaGraphLaunch(topology.executable, stream),
                         label + " replay");
    (void)(ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                 label + " replay synchronize"));
  }
}

template <typename Launch>
void expect_empty_invalid_capture(TestContext& test, cudaStream_t stream,
                                  const std::string& label,
                                  Launch&& launch) {
  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
      label + " begin capture");
  if (!ready) {
    return;
  }
  test.expect(static_cast<cudaError_t>(launch()) == cudaErrorInvalidValue,
              label + " rejects before enqueue");
  ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                       label + " end capture");
  if (ready && graph != nullptr) {
    std::size_t nodes = 0U;
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &nodes),
                         label + " count nodes");
    test.expect(ready && nodes == 0U, label + " captures zero nodes");
    (void)cudaGraphDestroy(graph);
  }
}

void test_resources(TestContext& test) {
  int shared_registers = 0;
  std::size_t shared_bytes = 0U;
  std::size_t shared_local = 0U;
  int shared_threads = 0;
  int shared_active = 0;
  bool ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          gdn_prefill_c16_norm_gate_test_detail::query_shared_resources(
              &shared_registers, &shared_bytes, &shared_local,
              &shared_threads, &shared_active)),
      "shared-boundary query resources");

  int global_registers = 0;
  std::size_t global_bytes = 0U;
  std::size_t global_local = 0U;
  int global_threads = 0;
  int global_active = 0;
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           gdn_prefill_c16_norm_gate_test_detail::
                               query_global_resources(
                                   &global_registers, &global_bytes,
                                   &global_local, &global_threads,
                                   &global_active)),
                       "global-boundary query resources");

  int production_registers = 0;
  std::size_t production_shared = 0U;
  std::size_t production_local = 0U;
  int production_threads = 0;
  int production_active = 0;
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           query_gated_delta_net_update_warp_eight_row_register_state_m16_resources_test_cuda(
                               &production_registers, &production_shared,
                               &production_local, &production_threads,
                               &production_active)),
                       "production exact-C16 query resources");
  if (!ready) {
    return;
  }

  const bool shared_gate =
      shared_registers <= 85 && shared_bytes == 38184U &&
      shared_local == 0U && shared_threads == 256 && shared_active >= 3;
  const bool global_control_gate =
      global_registers <= 85 && global_bytes >= 34056U &&
      global_bytes <= 34064U && global_local == 0U &&
      global_threads == 256 && global_active >= 3;
  const bool production_frozen =
      production_registers == 64 && production_shared == 34056U &&
      production_local == 0U && production_threads == 256 &&
      production_active == 4;
  std::cout << "GDN_C16_NORM_GATE_RESOURCES: shared_registers="
            << shared_registers << " shared_bytes=" << shared_bytes
            << " shared_local=" << shared_local
            << " shared_active=" << shared_active
            << " global_registers=" << global_registers
            << " global_bytes=" << global_bytes
            << " global_local=" << global_local
            << " global_active=" << global_active
            << " production_registers=" << production_registers
            << " production_shared=" << production_shared
            << " production_active=" << production_active
            << " minimum_active=3 target_active=4"
            << " shared_gate=" << (shared_gate ? "PASS" : "FAIL")
            << " global_control_gate="
            << (global_control_gate ? "PASS" : "FAIL")
            << " production_frozen="
            << (production_frozen ? "PASS" : "FAIL") << '\n';
  test.expect(shared_gate, "shared-boundary clears P0 resource gate");
  test.expect(global_control_gate,
              "global-boundary control clears P0 resource gate");
  test.expect(production_frozen, "production exact-C16 resources stay frozen");
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          gdn_prefill_c16_norm_gate_test_detail::query_shared_resources(
              nullptr, &shared_bytes, &shared_local, &shared_threads,
              &shared_active)) == cudaErrorInvalidValue,
      "resource query rejects null output");
}

void test_correctness_and_contract(TestContext& test, cudaStream_t stream) {
  constexpr std::uint16_t kPoison = 0x7fc1U;
  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> norm_weight;
  ManagedBuffer<std::uint16_t> silu_gate;
  ManagedBuffer<std::uint16_t> immutable_state;
  ManagedBuffer<std::uint16_t> baseline_state;
  ManagedBuffer<std::uint16_t> shared_state;
  ManagedBuffer<std::uint16_t> global_state;
  ManagedBuffer<std::uint16_t> debug_state;
  ManagedBuffer<std::uint16_t> shared_graph_state;
  ManagedBuffer<std::uint16_t> global_graph_state;
  ManagedBuffer<std::uint16_t> baseline_raw;
  ManagedBuffer<std::uint16_t> baseline_final;
  ManagedBuffer<std::uint16_t> shared_final;
  ManagedBuffer<std::uint16_t> global_final;
  ManagedBuffer<std::uint16_t> debug_raw;
  ManagedBuffer<std::uint16_t> debug_final;
  ManagedBuffer<std::uint16_t> shared_graph_final;
  ManagedBuffer<std::uint16_t> global_graph_final;

  bool ready = test.cuda_ok(
      conv_qkv.allocate(kHostTokenCount * q3x::runtime::kGdnQkvChannels),
      "allocate conv QKV");
  ready = ready && test.cuda_ok(
                       a.allocate(kHostTokenCount *
                                  q3x::runtime::kGdnValueHeadCount),
                       "allocate a");
  ready = ready && test.cuda_ok(
                       b.allocate(kHostTokenCount *
                                  q3x::runtime::kGdnValueHeadCount),
                       "allocate b");
  ready = ready && test.cuda_ok(
                       A_log.allocate(q3x::runtime::kGdnValueHeadCount),
                       "allocate A_log");
  ready = ready && test.cuda_ok(
                       dt_bias.allocate(q3x::runtime::kGdnValueHeadCount),
                       "allocate dt_bias");
  ready = ready && test.cuda_ok(
                       norm_weight.allocate(q3x::runtime::kGdnHeadDimension),
                       "allocate norm weight");
  ready = ready && test.cuda_ok(silu_gate.allocate(kOutputElements),
                                "allocate SiLU gate");
  const auto allocate_state = [&](ManagedBuffer<std::uint16_t>& buffer,
                                  const std::string& label) {
    return test.cuda_ok(buffer.allocate(q3x::runtime::kGdnStateElements),
                        label);
  };
  ready = ready && allocate_state(immutable_state, "allocate immutable state");
  ready = ready && allocate_state(baseline_state, "allocate baseline state");
  ready = ready && allocate_state(shared_state, "allocate shared state");
  ready = ready && allocate_state(global_state, "allocate global state");
  ready = ready && allocate_state(debug_state, "allocate debug state");
  ready = ready &&
          allocate_state(shared_graph_state, "allocate shared Graph state");
  ready = ready &&
          allocate_state(global_graph_state, "allocate global Graph state");
  const auto allocate_output = [&](ManagedBuffer<std::uint16_t>& buffer,
                                   const std::string& label) {
    return test.cuda_ok(buffer.allocate(kOutputElements), label);
  };
  ready = ready && allocate_output(baseline_raw, "allocate baseline raw");
  ready = ready && allocate_output(baseline_final, "allocate baseline final");
  ready = ready && allocate_output(shared_final, "allocate shared final");
  ready = ready && allocate_output(global_final, "allocate global final");
  ready = ready && allocate_output(debug_raw, "allocate debug raw");
  ready = ready && allocate_output(debug_final, "allocate debug final");
  ready = ready &&
          allocate_output(shared_graph_final, "allocate shared Graph final");
  ready = ready &&
          allocate_output(global_graph_final, "allocate global Graph final");
  if (!ready) {
    return;
  }

  fill_inputs(conv_qkv, a, b, A_log, dt_bias, norm_weight, silu_gate);
  fill_state(immutable_state.data(), false);
  const std::vector<std::uint16_t> original_conv(
      conv_qkv.data(), conv_qkv.data() + conv_qkv.size());
  const std::vector<std::uint16_t> original_a(a.data(), a.data() + a.size());
  const std::vector<std::uint16_t> original_b(b.data(), b.data() + b.size());
  const std::vector<std::uint16_t> original_A_log(
      A_log.data(), A_log.data() + A_log.size());
  const std::vector<std::uint16_t> original_dt_bias(
      dt_bias.data(), dt_bias.data() + dt_bias.size());
  const std::vector<std::uint16_t> original_norm(
      norm_weight.data(), norm_weight.data() + norm_weight.size());
  const std::vector<std::uint16_t> original_gate(
      silu_gate.data(), silu_gate.data() + silu_gate.size());
  const std::vector<std::uint16_t> original_state(
      immutable_state.data(), immutable_state.data() + immutable_state.size());
  const auto reset_state = [&](ManagedBuffer<std::uint16_t>& buffer) {
    std::copy_n(immutable_state.data(), q3x::runtime::kGdnStateElements,
                buffer.data());
  };
  reset_state(baseline_state);
  reset_state(shared_state);
  reset_state(global_state);
  reset_state(shared_graph_state);
  reset_state(global_graph_state);
  std::fill_n(debug_state.data(), debug_state.size(), kPoison);
  std::fill_n(baseline_raw.data(), baseline_raw.size(), kPoison);
  std::fill_n(baseline_final.data(), baseline_final.size(), kPoison);
  std::fill_n(shared_final.data(), shared_final.size(), kPoison);
  std::fill_n(global_final.data(), global_final.size(), kPoison);
  std::fill_n(debug_raw.data(), debug_raw.size(), kPoison);
  std::fill_n(debug_final.data(), debug_final.size(), kPoison);
  std::fill_n(shared_graph_final.data(), shared_graph_final.size(), kPoison);
  std::fill_n(global_graph_final.data(), global_graph_final.size(), kPoison);

  const auto launch_shared =
      [&](const std::uint16_t* input_state, std::uint16_t* output_state,
          std::uint16_t* output, cudaStream_t target_stream) {
        return q3x::runtime::gdn_prefill_c16_norm_gate_test_detail::
            launch_shared_boundary(
                conv_qkv.data(), kHostTokenCount, a.data(), b.data(),
                A_log.data(), dt_bias.data(), input_state, output_state,
                kL2Epsilon, norm_weight.data(), silu_gate.data(),
                kNormEpsilon, output, static_cast<void*>(target_stream));
      };
  const auto launch_global =
      [&](const std::uint16_t* input_state, std::uint16_t* output_state,
          std::uint16_t* output, cudaStream_t target_stream) {
        return q3x::runtime::gdn_prefill_c16_norm_gate_test_detail::
            launch_global_boundary(
                conv_qkv.data(), kHostTokenCount, a.data(), b.data(),
                A_log.data(), dt_bias.data(), input_state, output_state,
                kL2Epsilon, norm_weight.data(), silu_gate.data(),
                kNormEpsilon, output, static_cast<void*>(target_stream));
      };

  ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::runtime::launch_gated_delta_net_update_tile_warp_parallel_cuda(
              conv_qkv.data(), kHostTokenCount, a.data(), b.data(),
              A_log.data(), dt_bias.data(), baseline_state.data(),
              baseline_state.data(), kL2Epsilon, baseline_raw.data(), {},
              static_cast<void*>(stream))),
      "launch production exact-C16 raw baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                               baseline_raw.data(), norm_weight.data(),
                               silu_gate.data(), kNormHeadCount,
                               q3x::runtime::kGdnHeadDimension, kNormEpsilon,
                               baseline_final.data(),
                               static_cast<void*>(stream))),
                       "launch production norm/gate baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_shared(
                           shared_state.data(), shared_state.data(),
                           shared_final.data(), stream)),
                       "launch shared-boundary in-place candidate");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_global(
                           global_state.data(), global_state.data(),
                           global_final.data(), stream)),
                       "launch global-boundary in-place control");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           gdn_prefill_c16_norm_gate_test_detail::
                               launch_shared_boundary_debug(
                                   conv_qkv.data(), kHostTokenCount,
                                   a.data(), b.data(), A_log.data(),
                                   dt_bias.data(), immutable_state.data(),
                                   debug_state.data(), kL2Epsilon,
                                   norm_weight.data(), silu_gate.data(),
                                   kNormEpsilon, debug_final.data(),
                                   debug_raw.data(),
                                   static_cast<void*>(stream))),
                       "launch shared-boundary raw-debug candidate");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "correctness synchronize");
  if (!ready) {
    return;
  }

  expect_bitwise(test, shared_state.data(), baseline_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "shared-boundary state equals production");
  expect_bitwise(test, global_state.data(), baseline_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "global-boundary state equals production");
  expect_bitwise(test, debug_state.data(), baseline_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "disjoint debug state equals production");
  expect_bitwise(test, debug_raw.data(), baseline_raw.data(), kOutputElements,
                 "shared raw BF16 boundary equals production");
  expect_bitwise(test, shared_final.data(), baseline_final.data(),
                 kOutputElements, "shared-boundary final equals baseline");
  expect_bitwise(test, global_final.data(), baseline_final.data(),
                 kOutputElements, "global-boundary final equals baseline");
  expect_bitwise(test, debug_final.data(), baseline_final.data(),
                 kOutputElements, "debug final equals baseline");

  KernelTopology shared_topology;
  capture_one_kernel(
      test, stream, "shared-boundary Graph",
      [&]() {
        return launch_shared(immutable_state.data(), shared_graph_state.data(),
                             shared_graph_final.data(), stream);
      },
      shared_topology);
  KernelTopology global_topology;
  capture_one_kernel(
      test, stream, "global-boundary Graph",
      [&]() {
        return launch_global(immutable_state.data(), global_graph_state.data(),
                             global_graph_final.data(), stream);
      },
      global_topology);
  if (shared_topology.kernel_nodes == 1U) {
    test.expect(shared_topology.kernel.gridDim.x ==
                        q3x::runtime::kGdnValueHeadCount &&
                    shared_topology.kernel.blockDim.x == 256U &&
                    shared_topology.kernel.sharedMemBytes == 0U,
                "shared-boundary Graph locks 48x256 static-shared topology");
  }
  if (shared_topology.kernel_nodes == 1U &&
      global_topology.kernel_nodes == 1U) {
    test.expect(shared_topology.kernel.func != nullptr &&
                    global_topology.kernel.func != nullptr &&
                    shared_topology.kernel.func != global_topology.kernel.func,
                "shared and global boundary controls remain distinct kernels");
  }
  expect_bitwise(test, shared_graph_state.data(), baseline_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "shared Graph state equals baseline");
  expect_bitwise(test, global_graph_state.data(), baseline_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "global Graph state equals baseline");
  expect_bitwise(test, shared_graph_final.data(), baseline_final.data(),
                 kOutputElements, "shared Graph final equals baseline");
  expect_bitwise(test, global_graph_final.data(), baseline_final.data(),
                 kOutputElements, "global Graph final equals baseline");

  std::size_t finite_final_values = 0U;
  std::size_t nonzero_final_values = 0U;
  for (std::size_t index = 0U; index < baseline_final.size(); ++index) {
    const std::uint16_t bits = baseline_final[index];
    if ((bits & 0x7f80U) != 0x7f80U) {
      ++finite_final_values;
    }
    if ((bits & 0x7fffU) != 0U) {
      ++nonzero_final_values;
    }
  }
  test.expect(finite_final_values == baseline_final.size(),
              "finite-state composite produces only finite final values");
  test.expect(nonzero_final_values > baseline_final.size() / 2U,
              "finite-state composite exercises nonzero norm/gate values");

  // Retain a separate NaN-boundary case without allowing it to mask the
  // finite RMSNorm/SiLU proof above.
  fill_state(immutable_state.data(), true);
  reset_state(baseline_state);
  std::fill_n(debug_state.data(), debug_state.size(), kPoison);
  std::fill_n(baseline_raw.data(), baseline_raw.size(), kPoison);
  std::fill_n(baseline_final.data(), baseline_final.size(), kPoison);
  std::fill_n(debug_raw.data(), debug_raw.size(), kPoison);
  std::fill_n(debug_final.data(), debug_final.size(), kPoison);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::runtime::launch_gated_delta_net_update_tile_warp_parallel_cuda(
              conv_qkv.data(), kHostTokenCount, a.data(), b.data(),
              A_log.data(), dt_bias.data(), baseline_state.data(),
              baseline_state.data(), kL2Epsilon, baseline_raw.data(), {},
              static_cast<void*>(stream))),
      "launch NaN production exact-C16 raw baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                               baseline_raw.data(), norm_weight.data(),
                               silu_gate.data(), kNormHeadCount,
                               q3x::runtime::kGdnHeadDimension, kNormEpsilon,
                               baseline_final.data(),
                               static_cast<void*>(stream))),
                       "launch NaN production norm/gate baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           gdn_prefill_c16_norm_gate_test_detail::
                               launch_shared_boundary_debug(
                                   conv_qkv.data(), kHostTokenCount,
                                   a.data(), b.data(), A_log.data(),
                                   dt_bias.data(), immutable_state.data(),
                                   debug_state.data(), kL2Epsilon,
                                   norm_weight.data(), silu_gate.data(),
                                   kNormEpsilon, debug_final.data(),
                                   debug_raw.data(),
                                   static_cast<void*>(stream))),
                       "launch NaN shared-boundary debug candidate");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "NaN-boundary synchronize");
  if (!ready) {
    return;
  }
  expect_bitwise(test, debug_state.data(), baseline_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "NaN-boundary state equals production");
  expect_bitwise(test, debug_raw.data(), baseline_raw.data(), kOutputElements,
                 "NaN shared raw BF16 boundary equals production");
  expect_bitwise(test, debug_final.data(), baseline_final.data(),
                 kOutputElements, "NaN-boundary final equals baseline");
  std::copy(original_state.begin(), original_state.end(),
            immutable_state.data());

  using q3x::runtime::gdn_prefill_c16_norm_gate_test_detail::
      launch_shared_boundary;
  const auto expect_invalid = [&](const int status, const std::string& label) {
    test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
                label);
  };
  expect_invalid(
      launch_shared_boundary(
          conv_qkv.data(), kHostTokenCount - 1U, a.data(), b.data(),
          A_log.data(), dt_bias.data(), immutable_state.data(),
          shared_graph_state.data(), kL2Epsilon, norm_weight.data(),
          silu_gate.data(), kNormEpsilon, shared_graph_final.data(),
          static_cast<void*>(stream)),
      "rejects non-C16 token count");
  expect_invalid(
      launch_shared_boundary(
          conv_qkv.data(), kHostTokenCount, a.data(), b.data(), A_log.data(),
          dt_bias.data(), immutable_state.data(), shared_graph_state.data(),
          0.0F, norm_weight.data(), silu_gate.data(), kNormEpsilon,
          shared_graph_final.data(), static_cast<void*>(stream)),
      "rejects zero L2 epsilon");
  expect_invalid(
      launch_shared_boundary(
          conv_qkv.data(), kHostTokenCount, a.data(), b.data(), A_log.data(),
          dt_bias.data(), immutable_state.data(), shared_graph_state.data(),
          kL2Epsilon, norm_weight.data(), silu_gate.data(),
          std::numeric_limits<float>::quiet_NaN(), shared_graph_final.data(),
          static_cast<void*>(stream)),
      "rejects NaN norm epsilon");
  expect_invalid(
      launch_shared_boundary(
          conv_qkv.data(), kHostTokenCount, a.data(), b.data(), A_log.data(),
          dt_bias.data(), immutable_state.data(), shared_graph_state.data(),
          kL2Epsilon, nullptr, silu_gate.data(), kNormEpsilon,
          shared_graph_final.data(), static_cast<void*>(stream)),
      "rejects null norm weight");
  expect_invalid(
      launch_shared_boundary(
          conv_qkv.data(), kHostTokenCount, a.data(), b.data(), A_log.data(),
          dt_bias.data(), immutable_state.data(), shared_graph_state.data(),
          kL2Epsilon, norm_weight.data(), nullptr, kNormEpsilon,
          shared_graph_final.data(), static_cast<void*>(stream)),
      "rejects null SiLU gate");
  expect_invalid(
      launch_shared_boundary(
          conv_qkv.data(), kHostTokenCount, a.data(), b.data(), A_log.data(),
          dt_bias.data(), immutable_state.data(), shared_graph_state.data(),
          kL2Epsilon, norm_weight.data(), silu_gate.data(), kNormEpsilon,
          conv_qkv.data(), static_cast<void*>(stream)),
      "rejects output/input alias");
  expect_invalid(
      q3x::runtime::gdn_prefill_c16_norm_gate_test_detail::
          launch_shared_boundary_debug(
              conv_qkv.data(), kHostTokenCount, a.data(), b.data(),
              A_log.data(), dt_bias.data(), immutable_state.data(),
              debug_state.data(), kL2Epsilon, norm_weight.data(),
              silu_gate.data(), kNormEpsilon, debug_final.data(), nullptr,
              static_cast<void*>(stream)),
      "raw-debug candidate rejects null debug boundary");
  expect_empty_invalid_capture(
      test, stream, "invalid shared-boundary Graph",
      [&]() {
        return launch_shared_boundary(
            conv_qkv.data(), kHostTokenCount - 1U, a.data(), b.data(),
            A_log.data(), dt_bias.data(), immutable_state.data(),
            shared_graph_state.data(), kL2Epsilon, norm_weight.data(),
            silu_gate.data(), kNormEpsilon, shared_graph_final.data(),
            static_cast<void*>(stream));
      });

  expect_bitwise(test, conv_qkv.data(), original_conv.data(),
                 original_conv.size(), "conv QKV stays immutable");
  expect_bitwise(test, a.data(), original_a.data(), original_a.size(),
                 "a stays immutable");
  expect_bitwise(test, b.data(), original_b.data(), original_b.size(),
                 "b stays immutable");
  expect_bitwise(test, A_log.data(), original_A_log.data(),
                 original_A_log.size(), "A_log stays immutable");
  expect_bitwise(test, dt_bias.data(), original_dt_bias.data(),
                 original_dt_bias.size(), "dt_bias stays immutable");
  expect_bitwise(test, norm_weight.data(), original_norm.data(),
                 original_norm.size(), "norm weight stays immutable");
  expect_bitwise(test, silu_gate.data(), original_gate.data(),
                 original_gate.size(), "SiLU gate stays immutable");
  expect_bitwise(test, immutable_state.data(), original_state.data(),
                 original_state.size(), "disjoint input state stays immutable");
  const auto expect_guards = [&](const ManagedBuffer<std::uint16_t>& buffer,
                                 const std::string& label) {
    test.expect(buffer.guards_intact(), label + " redzones stay intact");
  };
  expect_guards(conv_qkv, "conv QKV");
  expect_guards(a, "a");
  expect_guards(b, "b");
  expect_guards(A_log, "A_log");
  expect_guards(dt_bias, "dt_bias");
  expect_guards(norm_weight, "norm weight");
  expect_guards(silu_gate, "SiLU gate");
  expect_guards(immutable_state, "immutable state");
  expect_guards(baseline_state, "baseline state");
  expect_guards(shared_state, "shared state");
  expect_guards(global_state, "global state");
  expect_guards(debug_state, "debug state");
  expect_guards(shared_graph_state, "shared Graph state");
  expect_guards(global_graph_state, "global Graph state");
  expect_guards(baseline_raw, "baseline raw");
  expect_guards(baseline_final, "baseline final");
  expect_guards(shared_final, "shared final");
  expect_guards(global_final, "global final");
  expect_guards(debug_raw, "debug raw");
  expect_guards(debug_final, "debug final");
  expect_guards(shared_graph_final, "shared Graph final");
  expect_guards(global_graph_final, "global Graph final");
  std::cout << "GDN_C16_NORM_GATE_BITWISE: raw_elements="
            << kOutputElements
            << " final_elements=" << kOutputElements
            << " state_elements=" << q3x::runtime::kGdnStateElements
            << " shared_boundary=true global_control=true"
            << " in_place=true disjoint=true graph=true redzones=true"
            << " immutable_inputs=true finite_epilogue=true"
            << " nan_boundary=true invalid_cases=7\n";
}

}  // namespace

int q3x_gdn_prefill_c16_norm_gate_sm87_candidate_main() {
  TestContext test;
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: GDN C16 norm/gate candidate requires CUDA\n";
    return 77;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                    "query CUDA device")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: GDN C16 norm/gate candidate requires SM87; got sm_"
              << properties.major << properties.minor << '\n';
    return 77;
  }
  std::cout << "GDN_C16_NORM_GATE_DEVICE: name=" << properties.name
            << " sm=" << properties.major << properties.minor << '\n';
  std::cout << "GDN_C16_NORM_GATE_EVIDENCE: tier=T1 payload=synthetic"
            << " authority=correctness_only performance_decision=NOT_RUN"
            << " production_promotion=NOT_RUN\n";

  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create test stream")) {
    return 1;
  }
  test_resources(test);
  test_correctness_and_contract(test, stream);
  (void)cudaStreamDestroy(stream);

  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " GDN C16 norm/gate candidate assertion(s) failed\n";
    return 1;
  }
  std::cout << "GDN C16 shared-boundary norm/gate SM87 candidate passed\n";
  return 0;
}
#endif
