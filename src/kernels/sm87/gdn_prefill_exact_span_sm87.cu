#include "gdn_prefill_exact_span_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_exact_span_detail {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kWarps = kThreads / kWarpSize;
constexpr unsigned int kRowsPerWarp = 16U;
constexpr unsigned int kLanesPerRow = 2U;
constexpr unsigned int kKeysPerLane = 64U;
constexpr unsigned int kKeysPerSlice = 8U;
constexpr unsigned int kSlicesPerLane = kKeysPerLane / kKeysPerSlice;
constexpr unsigned int kWordsPerSlice = kKeysPerSlice / 2U;
constexpr unsigned int kStateWordsPerLane =
    kSlicesPerLane * kWordsPerSlice;
constexpr unsigned int kMaximumTokenCount = 512U;
constexpr unsigned int kFullWarpMask = 0xffffffffU;
constexpr std::size_t kQOffset = 0U;
constexpr std::size_t kKOffset = kGdnQElements;
constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

static_assert(kWarps * kRowsPerWarp == kGdnHeadDimension);
static_assert(kLanesPerRow * kRowsPerWarp == kWarpSize);
static_assert(kKeysPerLane * kLanesPerRow == kGdnHeadDimension);
static_assert(kSlicesPerLane == 8U);
static_assert(kWordsPerSlice == 4U);
static_assert(kStateWordsPerLane == 32U);

__device__ __forceinline__ float decode_bf16(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16(
    const float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ std::uint32_t encode_bf16_pair(
    const float first, const float second) {
  const unsigned int first_bits = __float_as_uint(first);
  const unsigned int second_bits = __float_as_uint(second);
  const __nv_bfloat162_raw converted = static_cast<__nv_bfloat162_raw>(
      __floats2bfloat162_rn(first, second));
  std::uint32_t packed = static_cast<std::uint32_t>(converted.x) |
                         (static_cast<std::uint32_t>(converted.y) << 16U);
  if ((first_bits & 0x7fffffffU) > 0x7f800000U) {
    packed = (packed & 0xffff0000U) |
             static_cast<std::uint32_t>((first_bits >> 16U) | 0x0040U);
  }
  if ((second_bits & 0x7fffffffU) > 0x7f800000U) {
    packed = (packed & 0x0000ffffU) |
             (static_cast<std::uint32_t>((second_bits >> 16U) | 0x0040U)
              << 16U);
  }
  return packed;
}

__device__ __forceinline__ float stable_softplus(const float value) {
  return value > 20.0F ? value : log1pf(expf(value));
}

__device__ __forceinline__ float stable_sigmoid(const float value) {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + expf(-value));
  }
  const float exponential = expf(value);
  return exponential / (1.0F + exponential);
}

// This is the long-span form of the independently screened exact M1 row16
// register-baton topology.  Relative to the exact-C16 incumbent it removes
// the FP32 row_scratch write/prediction-read/update-read/update-write/output-
// read cycle.  Packed BF16 state remains authoritative in registers after
// every token; extending the token loop never changes that numerical
// boundary.
__launch_bounds__(kThreads, 3) __global__ void
gated_delta_net_update_exact_span_row16_kernel(
    const std::uint16_t* const conv_qkv,
    const unsigned int token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    std::uint16_t* const output) {
  __shared__ float normalized_q[kGdnHeadDimension];
  __shared__ float normalized_k[kGdnHeadDimension];
  __shared__ float recurrence_scalars[2];

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const unsigned int row_in_warp = lane / kLanesPerRow;
  const unsigned int row_half = lane % kLanesPerRow;
  const unsigned int pair_first_lane = lane & ~1U;
  const unsigned int pair_second_lane = pair_first_lane + 1U;
  const std::size_t value_head = blockIdx.x;
  const std::size_t qk_head = value_head / 3U;
  const std::size_t value_row =
      static_cast<std::size_t>(warp) * kRowsPerWarp + row_in_warp;
  const std::size_t state_row_offset =
      (value_head * kGdnHeadDimension + value_row) * kGdnHeadDimension;

  std::uint32_t state_words[kStateWordsPerLane];
#pragma unroll
  for (unsigned int slice = 0U; slice < kSlicesPerLane; ++slice) {
    const std::size_t first_key =
        static_cast<std::size_t>(slice) * 2U * kKeysPerSlice +
        static_cast<std::size_t>(row_half) * kKeysPerSlice;
    const auto packed = *reinterpret_cast<const uint4*>(
        state_input + state_row_offset + first_key);
    state_words[slice * kWordsPerSlice + 0U] = packed.x;
    state_words[slice * kWordsPerSlice + 1U] = packed.y;
    state_words[slice * kWordsPerSlice + 2U] = packed.z;
    state_words[slice * kWordsPerSlice + 3U] = packed.w;
  }

#pragma unroll 1
  for (unsigned int token = 0U; token < token_count; ++token) {
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
      normalized_q[thread] = decode_bf16(
          conv_qkv[qkv_token_offset + kQOffset +
                   qk_head * kGdnHeadDimension + thread]);
      normalized_k[thread] = decode_bf16(
          conv_qkv[qkv_token_offset + kKOffset +
                   qk_head * kGdnHeadDimension + thread]);
    }
    __syncthreads();

    // Preserve the exact incumbent four-item normalization tree.
    if (warp == 0U || warp == 1U) {
      float* const values = warp == 0U ? normalized_q : normalized_k;
      const float first = values[lane];
      const float second = values[lane + 32U];
      const float third = values[lane + 64U];
      const float fourth = values[lane + 96U];
      const float first_pair = first * first + third * third;
      const float second_pair = second * second + fourth * fourth;
      float square_sum = first_pair + second_pair;
#pragma unroll
      for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
        square_sum +=
            __shfl_down_sync(kFullWarpMask, square_sum, stride);
      }
      float scale = 0.0F;
      if (lane == 0U) {
        scale = rsqrtf(square_sum + l2_epsilon);
        if (warp == 0U) {
          scale *= rsqrtf(static_cast<float>(kGdnHeadDimension));
        }
      }
      scale = __shfl_sync(kFullWarpMask, scale, 0U);
      values[lane] = first * scale;
      values[lane + 32U] = second * scale;
      values[lane + 64U] = third * scale;
      values[lane + 96U] = fourth * scale;
    }
    if (thread == 64U) {
      const float gate_input =
          decode_bf16(a[scalar_token_offset + value_head]) +
          decode_bf16(dt_bias[value_head]);
      const float g = -expf(decode_bf16(A_log[value_head])) *
                      stable_softplus(gate_input);
      recurrence_scalars[0] = expf(g);
      recurrence_scalars[1] =
          stable_sigmoid(decode_bf16(b[scalar_token_offset + value_head]));
    }
    __syncthreads();

    const float alpha = recurrence_scalars[0];
    const float beta = recurrence_scalars[1];
    float prediction = 0.0F;
#pragma unroll
    for (unsigned int slice = 0U; slice < kSlicesPerLane; ++slice) {
      const unsigned int first_key =
          slice * 2U * kKeysPerSlice + row_half * kKeysPerSlice;
      if (row_half == 0U) {
#pragma unroll
        for (unsigned int word_index = 0U;
             word_index < kWordsPerSlice; ++word_index) {
          const std::uint32_t word =
              state_words[slice * kWordsPerSlice + word_index];
          const unsigned int key = first_key + word_index * 2U;
          const float first_state = alpha * decode_bf16(
              static_cast<std::uint16_t>(word));
          const float second_state = alpha * decode_bf16(
              static_cast<std::uint16_t>(word >> 16U));
          prediction =
              fmaf(first_state, normalized_k[key], prediction);
          prediction =
              fmaf(second_state, normalized_k[key + 1U], prediction);
        }
      }
      prediction = __shfl_sync(kFullWarpMask, prediction, pair_first_lane);
      if (row_half == 1U) {
#pragma unroll
        for (unsigned int word_index = 0U;
             word_index < kWordsPerSlice; ++word_index) {
          const std::uint32_t word =
              state_words[slice * kWordsPerSlice + word_index];
          const unsigned int key = first_key + word_index * 2U;
          const float first_state = alpha * decode_bf16(
              static_cast<std::uint16_t>(word));
          const float second_state = alpha * decode_bf16(
              static_cast<std::uint16_t>(word >> 16U));
          prediction =
              fmaf(first_state, normalized_k[key], prediction);
          prediction =
              fmaf(second_state, normalized_k[key + 1U], prediction);
        }
      }
      prediction = __shfl_sync(kFullWarpMask, prediction, pair_second_lane);
    }

    float delta = 0.0F;
    if (row_half == 0U) {
      const std::size_t value_offset =
          qkv_token_offset + kVOffset +
          value_head * kGdnHeadDimension + value_row;
      delta = (decode_bf16(conv_qkv[value_offset]) - prediction) * beta;
    }
    delta = __shfl_sync(kFullWarpMask, delta, pair_first_lane);

    float result = 0.0F;
#pragma unroll
    for (unsigned int slice = 0U; slice < kSlicesPerLane; ++slice) {
      const unsigned int first_key =
          slice * 2U * kKeysPerSlice + row_half * kKeysPerSlice;
      if (row_half == 0U) {
#pragma unroll
        for (unsigned int word_index = 0U;
             word_index < kWordsPerSlice; ++word_index) {
          const unsigned int state_index =
              slice * kWordsPerSlice + word_index;
          const std::uint32_t word = state_words[state_index];
          const unsigned int key = first_key + word_index * 2U;
          const float first_updated = fmaf(
              delta, normalized_k[key],
              alpha * decode_bf16(static_cast<std::uint16_t>(word)));
          const float second_updated = fmaf(
              delta, normalized_k[key + 1U],
              alpha * decode_bf16(static_cast<std::uint16_t>(word >> 16U)));
          state_words[state_index] =
              encode_bf16_pair(first_updated, second_updated);
          result = fmaf(first_updated, normalized_q[key], result);
          result = fmaf(second_updated, normalized_q[key + 1U], result);
        }
      }
      result = __shfl_sync(kFullWarpMask, result, pair_first_lane);
      if (row_half == 1U) {
#pragma unroll
        for (unsigned int word_index = 0U;
             word_index < kWordsPerSlice; ++word_index) {
          const unsigned int state_index =
              slice * kWordsPerSlice + word_index;
          const std::uint32_t word = state_words[state_index];
          const unsigned int key = first_key + word_index * 2U;
          const float first_updated = fmaf(
              delta, normalized_k[key],
              alpha * decode_bf16(static_cast<std::uint16_t>(word)));
          const float second_updated = fmaf(
              delta, normalized_k[key + 1U],
              alpha * decode_bf16(static_cast<std::uint16_t>(word >> 16U)));
          state_words[state_index] =
              encode_bf16_pair(first_updated, second_updated);
          result = fmaf(first_updated, normalized_q[key], result);
          result = fmaf(second_updated, normalized_q[key + 1U], result);
        }
      }
      result = __shfl_sync(kFullWarpMask, result, pair_second_lane);
    }
    if (row_half == 1U) {
      output[output_token_offset + value_head * kGdnHeadDimension +
             value_row] = encode_bf16(result);
    }
  }

#pragma unroll
  for (unsigned int slice = 0U; slice < kSlicesPerLane; ++slice) {
    const std::size_t first_key =
        static_cast<std::size_t>(slice) * 2U * kKeysPerSlice +
        static_cast<std::size_t>(row_half) * kKeysPerSlice;
    const uint4 packed{
        state_words[slice * kWordsPerSlice + 0U],
        state_words[slice * kWordsPerSlice + 1U],
        state_words[slice * kWordsPerSlice + 2U],
        state_words[slice * kWordsPerSlice + 3U]};
    *reinterpret_cast<uint4*>(state_output + state_row_offset + first_key) =
        packed;
  }
}

[[nodiscard]] bool invalid_alias(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    const std::uint16_t* const state_output,
    const std::uint16_t* const output) noexcept {
  const bool output_alias =
      output == conv_qkv || output == a || output == b ||
      output == A_log || output == dt_bias || output == state_input ||
      output == state_output;
  const bool state_input_alias =
      state_input == conv_qkv || state_input == a || state_input == b ||
      state_input == A_log || state_input == dt_bias;
  const bool state_output_alias =
      state_output == conv_qkv || state_output == a || state_output == b ||
      state_output == A_log || state_output == dt_bias;
  return output_alias || state_input_alias || state_output_alias;
}

}  // namespace

int launch_row16_register_baton(
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
  if (token_count == 0U || token_count > kMaximumTokenCount ||
      token_count % 16U != 0U || conv_qkv == nullptr || a == nullptr ||
      b == nullptr || A_log == nullptr || dt_bias == nullptr ||
      state_input == nullptr || state_output == nullptr || output == nullptr ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      (reinterpret_cast<std::uintptr_t>(state_input) & 0xfU) != 0U ||
      (reinterpret_cast<std::uintptr_t>(state_output) & 0xfU) != 0U ||
      invalid_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                    state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_exact_span_row16_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kThreads, 0U, stream>>>(
      conv_qkv, static_cast<unsigned int>(token_count), a, b, A_log,
      dt_bias, state_input, state_output, l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int query_row16_register_baton_resources(
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
      &attributes, gated_delta_net_update_exact_span_row16_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, gated_delta_net_update_exact_span_row16_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active;
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::runtime::gdn_prefill_exact_span_detail

