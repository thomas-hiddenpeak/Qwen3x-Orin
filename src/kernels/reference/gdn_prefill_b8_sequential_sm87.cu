#include "gdn_prefill_b8_sequential_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kWarpsPerBlock = kThreads / kWarpSize;
constexpr unsigned int kBlockTokens = 8U;
constexpr unsigned int kRowsPerWarp = 8U;
constexpr unsigned int kSubgroupWidth = 4U;
constexpr unsigned int kFullWarpMask = 0xffffffffU;
constexpr std::size_t kFirstExactTokenCount = 256U;
constexpr std::size_t kSecondExactTokenCount = 512U;

static_assert(kWarpsPerBlock * kRowsPerWarp * 2U == kGdnHeadDimension);
static_assert(kWarpSize / kSubgroupWidth == kRowsPerWarp);
static_assert((kFirstExactTokenCount % kBlockTokens) == 0U);
static_assert((kSecondExactTokenCount % kBlockTokens) == 0U);

[[nodiscard]] bool byte_range_overflows(const void* const pointer,
                                        const std::size_t bytes) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

[[nodiscard]] bool ranges_overlap(const void* const first,
                                  const std::size_t first_bytes,
                                  const void* const second,
                                  const std::size_t second_bytes) noexcept {
  if (byte_range_overflows(first, first_bytes) ||
      byte_range_overflows(second, second_bytes)) {
    return true;
  }
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  return first_begin < second_begin + second_bytes &&
         second_begin < first_begin + first_bytes;
}

[[nodiscard]] bool aligned_bf16_pointer(
    const std::uint16_t* const pointer) noexcept {
  return (reinterpret_cast<std::uintptr_t>(pointer) %
          alignof(std::uint16_t)) == 0U;
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

// Admission-only Prefill candidate selected by the B8 screen. One CTA owns a
// value head. Every four-lane subgroup keeps one complete state row in FP32
// across eight ordered recurrence steps, then publishes one BF16 boundary.
// No WY transformation is used, so the update remains the screened sequential
// recurrence. Exact C256/C512 comprise 32/64 ordered B8 boundaries.
__launch_bounds__(kThreads, 1) __global__ void
gated_delta_net_update_sequential_fp32_b8_kernel(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    std::uint16_t* const output) {
  __shared__ float normalized_q[kBlockTokens][kGdnHeadDimension];
  __shared__ float normalized_k[kBlockTokens][kGdnHeadDimension];
  __shared__ float alpha[kBlockTokens];
  __shared__ float beta[kBlockTokens];

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const unsigned int row_in_warp = lane / kSubgroupWidth;
  const unsigned int subgroup_lane = lane % kSubgroupWidth;
  const std::size_t value_head = blockIdx.x;
  const std::size_t qk_head = value_head / 3U;
  constexpr std::size_t kKOffset = kGdnQElements;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;
  constexpr unsigned int kItemsPerLane =
      kGdnHeadDimension / kSubgroupWidth;
  static_assert(kItemsPerLane == 32U);

#pragma unroll 1
  for (std::size_t block_start = 0U; block_start < token_count;
       block_start += kBlockTokens) {
    const std::size_t remaining_tokens = token_count - block_start;
    const unsigned int block_token_count = static_cast<unsigned int>(
        remaining_tokens < static_cast<std::size_t>(kBlockTokens)
            ? remaining_tokens
            : static_cast<std::size_t>(kBlockTokens));

    if (warp < block_token_count) {
      const std::size_t token = block_start + warp;
      const std::size_t qkv_token_offset = token * kGdnQkvChannels;
      float q_values[4];
      float k_values[4];
#pragma unroll
      for (unsigned int item = 0U; item < 4U; ++item) {
        const unsigned int dimension = lane + item * kWarpSize;
        q_values[item] = decode_bf16_device(
            conv_qkv[qkv_token_offset + qk_head * kGdnHeadDimension +
                     dimension]);
        k_values[item] = decode_bf16_device(
            conv_qkv[qkv_token_offset + kKOffset +
                     qk_head * kGdnHeadDimension + dimension]);
      }
      const float q_first_pair =
          q_values[0] * q_values[0] + q_values[2] * q_values[2];
      const float q_second_pair =
          q_values[1] * q_values[1] + q_values[3] * q_values[3];
      const float k_first_pair =
          k_values[0] * k_values[0] + k_values[2] * k_values[2];
      const float k_second_pair =
          k_values[1] * k_values[1] + k_values[3] * k_values[3];
      float q_sum = q_first_pair + q_second_pair;
      float k_sum = k_first_pair + k_second_pair;
#pragma unroll
      for (unsigned int stride = kWarpSize / 2U; stride != 0U;
           stride >>= 1U) {
        q_sum += __shfl_down_sync(kFullWarpMask, q_sum, stride);
        k_sum += __shfl_down_sync(kFullWarpMask, k_sum, stride);
      }
      const float q_scale =
          __shfl_sync(kFullWarpMask,
                      lane == 0U
                          ? rsqrtf(q_sum + l2_epsilon) *
                                rsqrtf(static_cast<float>(kGdnHeadDimension))
                          : 0.0F,
                      0U);
      const float k_scale = __shfl_sync(
          kFullWarpMask, lane == 0U ? rsqrtf(k_sum + l2_epsilon) : 0.0F,
          0U);
#pragma unroll
      for (unsigned int item = 0U; item < 4U; ++item) {
        const unsigned int dimension = lane + item * kWarpSize;
        normalized_q[warp][dimension] = q_values[item] * q_scale;
        normalized_k[warp][dimension] = k_values[item] * k_scale;
      }
      if (lane == 0U) {
        const std::size_t scalar_offset =
            token * kGdnValueHeadCount + value_head;
        const float gate_input = decode_bf16_device(a[scalar_offset]) +
                                 decode_bf16_device(dt_bias[value_head]);
        const float g = -expf(decode_bf16_device(A_log[value_head])) *
                        stable_softplus_device(gate_input);
        alpha[warp] = expf(g);
        beta[warp] =
            stable_sigmoid_device(decode_bf16_device(b[scalar_offset]));
      }
    }
    __syncthreads();

    const std::uint16_t* const recurrent_state =
        block_start == 0U ? state_input : state_output;
    const std::size_t head_state_offset =
        value_head * kGdnHeadDimension * kGdnHeadDimension;

#pragma unroll
    for (unsigned int batch = 0U; batch < 2U; ++batch) {
      const std::size_t row =
          static_cast<std::size_t>(warp * kRowsPerWarp + row_in_warp) +
          static_cast<std::size_t>(batch) * 64U;
      const std::size_t state_row_offset =
          head_state_offset + row * kGdnHeadDimension;
      float state_items[kItemsPerLane];

#pragma unroll
      for (unsigned int item = 0U; item < kItemsPerLane; ++item) {
        const unsigned int dimension =
            subgroup_lane + item * kSubgroupWidth;
        state_items[item] = decode_bf16_device(
            recurrent_state[state_row_offset + dimension]);
      }
#pragma unroll
      for (unsigned int token = 0U; token < kBlockTokens; ++token) {
        if (token < block_token_count) {
          float prediction = 0.0F;
#pragma unroll
          for (unsigned int item = 0U; item < kItemsPerLane; ++item) {
            const unsigned int dimension =
                subgroup_lane + item * kSubgroupWidth;
            prediction = fmaf(alpha[token] * state_items[item],
                              normalized_k[token][dimension], prediction);
          }
          prediction += __shfl_down_sync(kFullWarpMask, prediction, 2U,
                                         kSubgroupWidth);
          prediction += __shfl_down_sync(kFullWarpMask, prediction, 1U,
                                         kSubgroupWidth);
          float delta = 0.0F;
          if (subgroup_lane == 0U) {
            const std::size_t value_offset =
                (block_start + token) * kGdnQkvChannels + kVOffset +
                value_head * kGdnHeadDimension + row;
            delta =
                (decode_bf16_device(conv_qkv[value_offset]) - prediction) *
                beta[token];
          }
          delta = __shfl_sync(kFullWarpMask, delta, 0U, kSubgroupWidth);
          float result = 0.0F;
#pragma unroll
          for (unsigned int item = 0U; item < kItemsPerLane; ++item) {
            const unsigned int dimension =
                subgroup_lane + item * kSubgroupWidth;
            state_items[item] =
                fmaf(delta, normalized_k[token][dimension],
                     alpha[token] * state_items[item]);
            result = fmaf(state_items[item], normalized_q[token][dimension],
                          result);
          }
          result += __shfl_down_sync(kFullWarpMask, result, 2U,
                                     kSubgroupWidth);
          result += __shfl_down_sync(kFullWarpMask, result, 1U,
                                     kSubgroupWidth);
          if (subgroup_lane == 0U) {
            const std::size_t output_offset =
                (block_start + token) * kGdnVElements +
                value_head * kGdnHeadDimension + row;
            output[output_offset] = encode_bf16_device(result);
          }
        }
      }
#pragma unroll
      for (unsigned int item = 0U; item < kItemsPerLane; ++item) {
        const unsigned int dimension =
            subgroup_lane + item * kSubgroupWidth;
        state_output[state_row_offset + dimension] =
            encode_bf16_device(state_items[item]);
      }
    }
    // The next B8 block reads this CTA's just-published BF16 head state.
    __syncthreads();
  }
}

[[nodiscard]] bool invalid_launch_arguments(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    const std::uint16_t* const state_output,
    const float l2_epsilon,
    const std::uint16_t* const output) noexcept {
  if ((token_count != kFirstExactTokenCount &&
       token_count != kSecondExactTokenCount) ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr ||
      A_log == nullptr || dt_bias == nullptr || state_input == nullptr ||
      state_output == nullptr || output == nullptr ||
      !aligned_bf16_pointer(conv_qkv) || !aligned_bf16_pointer(a) ||
      !aligned_bf16_pointer(b) || !aligned_bf16_pointer(A_log) ||
      !aligned_bf16_pointer(dt_bias) ||
      !aligned_bf16_pointer(state_input) ||
      !aligned_bf16_pointer(state_output) ||
      !aligned_bf16_pointer(output)) {
    return true;
  }

  constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);
  const std::size_t qkv_bytes =
      token_count * kGdnQkvChannels * kBf16Bytes;
  const std::size_t scalar_bytes =
      token_count * kGdnValueHeadCount * kBf16Bytes;
  constexpr std::size_t kHeadScalarBytes =
      kGdnValueHeadCount * kBf16Bytes;
  constexpr std::size_t kStateBytes = kGdnStateElements * kBf16Bytes;
  const std::size_t output_bytes =
      token_count * kGdnVElements * kBf16Bytes;

  if (byte_range_overflows(conv_qkv, qkv_bytes) ||
      byte_range_overflows(a, scalar_bytes) ||
      byte_range_overflows(b, scalar_bytes) ||
      byte_range_overflows(A_log, kHeadScalarBytes) ||
      byte_range_overflows(dt_bias, kHeadScalarBytes) ||
      byte_range_overflows(state_input, kStateBytes) ||
      byte_range_overflows(state_output, kStateBytes) ||
      byte_range_overflows(output, output_bytes)) {
    return true;
  }

  struct Span {
    const void* pointer;
    std::size_t bytes;
  };
  constexpr std::size_t kStateInputIndex = 5U;
  constexpr std::size_t kStateOutputIndex = 6U;
  const std::array<Span, 8U> spans{{
      {conv_qkv, qkv_bytes},
      {a, scalar_bytes},
      {b, scalar_bytes},
      {A_log, kHeadScalarBytes},
      {dt_bias, kHeadScalarBytes},
      {state_input, kStateBytes},
      {state_output, kStateBytes},
      {output, output_bytes},
  }};
  for (std::size_t first = 0U; first < spans.size(); ++first) {
    for (std::size_t second = first + 1U; second < spans.size(); ++second) {
      const bool exact_in_place_state =
          first == kStateInputIndex && second == kStateOutputIndex &&
          state_input == state_output;
      if (!exact_in_place_state &&
          ranges_overlap(spans[first].pointer, spans[first].bytes,
                         spans[second].pointer, spans[second].bytes)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

namespace gdn_prefill_b8_detail {

int launch_gated_delta_net_update_sequential_fp32_b8_exact_cuda(
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
  if (invalid_launch_arguments(conv_qkv, token_count, a, b, A_log, dt_bias,
                               state_input, state_output, l2_epsilon,
                               output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_sequential_fp32_b8_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kThreads, 0U, stream>>>(
      conv_qkv, token_count, a, b, A_log, dt_bias, state_input, state_output,
      l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int query_gated_delta_net_update_sequential_fp32_b8_resources_cuda(
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
      &attributes, gated_delta_net_update_sequential_fp32_b8_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, gated_delta_net_update_sequential_fp32_b8_kernel,
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

}  // namespace gdn_prefill_b8_detail
}  // namespace q3x::runtime
