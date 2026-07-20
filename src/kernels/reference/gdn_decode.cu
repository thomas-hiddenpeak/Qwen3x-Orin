#include "q3x/runtime/gdn_decode.h"
#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

constexpr unsigned int kConvThreads = 256U;
constexpr unsigned int kGdnThreads =
    static_cast<unsigned int>(kGdnHeadDimension);

[[nodiscard]] bool multiply_overflows(const std::size_t left,
                                      const std::size_t right) noexcept {
  return right != 0U &&
         left > std::numeric_limits<std::size_t>::max() / right;
}

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

[[nodiscard]] GdnStatus validate_dimensions(
    const GdnDimensions dimensions) noexcept {
  if (multiply_overflows(dimensions.qk_head_count,
                         dimensions.head_dimension) ||
      multiply_overflows(dimensions.value_head_count,
                         dimensions.head_dimension)) {
    return GdnStatus::kSizeOverflow;
  }
  const std::size_t q_elements =
      dimensions.qk_head_count * dimensions.head_dimension;
  const std::size_t v_elements =
      dimensions.value_head_count * dimensions.head_dimension;
  if (q_elements >
          (std::numeric_limits<std::size_t>::max() - v_elements) / 2U ||
      multiply_overflows(v_elements, dimensions.head_dimension)) {
    return GdnStatus::kSizeOverflow;
  }
  if (dimensions.qk_head_count != kGdnQkHeadCount ||
      dimensions.value_head_count != kGdnValueHeadCount ||
      dimensions.head_dimension != kGdnHeadDimension) {
    return GdnStatus::kInvalidDimension;
  }
  return GdnStatus::kSuccess;
}

[[nodiscard]] bool invalid_conv_alias(
    const std::uint16_t* const raw_qkv,
    const std::uint16_t* const conv_weight,
    const std::uint16_t* const history,
    const std::uint16_t* const output) noexcept {
  return raw_qkv == conv_weight || raw_qkv == history ||
         conv_weight == history || conv_weight == output ||
         history == output;
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

[[nodiscard]] bool invalid_gdn_norm_gate_alias(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    const std::size_t norm_head_dimension,
    std::uint16_t* const output) noexcept {
  constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);
  constexpr std::size_t kQkvBytes = kGdnQkvChannels * kBf16Bytes;
  constexpr std::size_t kScalarBytes =
      kGdnValueHeadCount * kBf16Bytes;
  constexpr std::size_t kStateBytes = kGdnStateElements * kBf16Bytes;
  constexpr std::size_t kOutputBytes = kGdnVElements * kBf16Bytes;
  const std::size_t norm_bytes = norm_head_dimension * kBf16Bytes;

  const bool output_overlap =
      ranges_overlap(output, kOutputBytes, conv_qkv, kQkvBytes) ||
      ranges_overlap(output, kOutputBytes, a, kScalarBytes) ||
      ranges_overlap(output, kOutputBytes, b, kScalarBytes) ||
      ranges_overlap(output, kOutputBytes, A_log, kScalarBytes) ||
      ranges_overlap(output, kOutputBytes, dt_bias, kScalarBytes) ||
      ranges_overlap(output, kOutputBytes, state_input, kStateBytes) ||
      ranges_overlap(output, kOutputBytes, state_output, kStateBytes) ||
      ranges_overlap(output, kOutputBytes, norm_weight, norm_bytes) ||
      ranges_overlap(output, kOutputBytes, silu_gate, kOutputBytes);
  const bool state_overlap =
      (state_input != state_output &&
       ranges_overlap(state_input, kStateBytes, state_output,
                      kStateBytes)) ||
      ranges_overlap(state_output, kStateBytes, conv_qkv, kQkvBytes) ||
      ranges_overlap(state_output, kStateBytes, a, kScalarBytes) ||
      ranges_overlap(state_output, kStateBytes, b, kScalarBytes) ||
      ranges_overlap(state_output, kStateBytes, A_log, kScalarBytes) ||
      ranges_overlap(state_output, kStateBytes, dt_bias, kScalarBytes) ||
      ranges_overlap(state_output, kStateBytes, norm_weight, norm_bytes) ||
      ranges_overlap(state_output, kStateBytes, silu_gate, kOutputBytes);
  return output_overlap || state_overlap;
}

[[nodiscard]] bool aligned_bf16_pointer(
    const std::uint16_t* const pointer) noexcept {
  return (reinterpret_cast<std::uintptr_t>(pointer) %
          alignof(std::uint16_t)) == 0U;
}

[[nodiscard]] int validate_gdn_plain_norm_silu_gate_composite(
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
    const std::size_t norm_head_count,
    const std::size_t norm_head_dimension,
    const float norm_epsilon,
    std::uint16_t* const output,
    const GdnDimensions dimensions,
    bool* const exact_fusion) noexcept {
  if (exact_fusion == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *exact_fusion = false;
  if (validate_dimensions(dimensions) != GdnStatus::kSuccess ||
      norm_head_count == 0U || norm_head_dimension == 0U ||
      multiply_overflows(norm_head_count, norm_head_dimension) ||
      norm_head_count * norm_head_dimension != kGdnVElements ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      !std::isfinite(norm_epsilon) || norm_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr ||
      A_log == nullptr || dt_bias == nullptr || state_input == nullptr ||
      state_output == nullptr || norm_weight == nullptr ||
      silu_gate == nullptr || output == nullptr ||
      !aligned_bf16_pointer(conv_qkv) || !aligned_bf16_pointer(a) ||
      !aligned_bf16_pointer(b) || !aligned_bf16_pointer(A_log) ||
      !aligned_bf16_pointer(dt_bias) ||
      !aligned_bf16_pointer(state_input) ||
      !aligned_bf16_pointer(state_output) ||
      !aligned_bf16_pointer(norm_weight) ||
      !aligned_bf16_pointer(silu_gate) ||
      !aligned_bf16_pointer(output) ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output) ||
      invalid_gdn_norm_gate_alias(
          conv_qkv, a, b, A_log, dt_bias, state_input, state_output,
          norm_weight, silu_gate, norm_head_dimension, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *exact_fusion =
      supports_gated_delta_net_update_plain_rms_norm_silu_gate_fusion(
          1U, dimensions, norm_head_count, norm_head_dimension);
  return static_cast<int>(cudaSuccess);
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
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
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

__global__ void causal_conv1d_silu_update_kernel(
    const std::uint16_t* const raw_qkv,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history,
    std::uint16_t* const output) {
  for (std::size_t channel =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       channel < kGdnQkvChannels;
       channel += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t history_offset = channel * kGdnConvHistoryWidth;
    const std::size_t weight_offset = channel * kGdnConvKernelWidth;
    const std::uint16_t current_bits = raw_qkv[channel];
    float convolution = 0.0F;
    convolution = fmaf(decode_bf16_device(history[history_offset]),
                       decode_bf16_device(conv_weight[weight_offset]),
                       convolution);
    convolution = fmaf(decode_bf16_device(history[history_offset + 1U]),
                       decode_bf16_device(conv_weight[weight_offset + 1U]),
                       convolution);
    convolution = fmaf(decode_bf16_device(history[history_offset + 2U]),
                       decode_bf16_device(conv_weight[weight_offset + 2U]),
                       convolution);
    convolution = fmaf(decode_bf16_device(current_bits),
                       decode_bf16_device(conv_weight[weight_offset + 3U]),
                       convolution);
    output[channel] = encode_bf16_device(
        convolution / (1.0F + expf(-convolution)));
    history[history_offset] = history[history_offset + 1U];
    history[history_offset + 1U] = history[history_offset + 2U];
    history[history_offset + 2U] = current_bits;
  }
}

__global__ void causal_conv1d_silu_update_tile_kernel(
    const std::uint16_t* const raw_qkv,
    const std::size_t token_count,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history,
    std::uint16_t* const output) {
  for (std::size_t channel =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       channel < kGdnQkvChannels;
       channel += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t history_offset = channel * kGdnConvHistoryWidth;
    const std::size_t weight_offset = channel * kGdnConvKernelWidth;
    std::uint16_t history_0 = history[history_offset];
    std::uint16_t history_1 = history[history_offset + 1U];
    std::uint16_t history_2 = history[history_offset + 2U];
    for (std::size_t token = 0U; token < token_count; ++token) {
      const std::size_t token_offset = token * kGdnQkvChannels;
      const std::uint16_t current_bits = raw_qkv[token_offset + channel];
      float convolution = 0.0F;
      convolution = fmaf(decode_bf16_device(history_0),
                         decode_bf16_device(conv_weight[weight_offset]),
                         convolution);
      convolution = fmaf(
          decode_bf16_device(history_1),
          decode_bf16_device(conv_weight[weight_offset + 1U]), convolution);
      convolution = fmaf(
          decode_bf16_device(history_2),
          decode_bf16_device(conv_weight[weight_offset + 2U]), convolution);
      convolution = fmaf(
          decode_bf16_device(current_bits),
          decode_bf16_device(conv_weight[weight_offset + 3U]), convolution);
      output[token_offset + channel] = encode_bf16_device(
          convolution / (1.0F + expf(-convolution)));
      history_0 = history_1;
      history_1 = history_2;
      history_2 = current_bits;
    }
    history[history_offset] = history_0;
    history[history_offset + 1U] = history_1;
    history[history_offset + 2U] = history_2;
  }
}

__global__ void gated_delta_net_update_kernel(
    const std::uint16_t* const conv_qkv,
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
  __shared__ float partial[kGdnHeadDimension];
  __shared__ float recurrence_scalars[2];

  const std::size_t value_head = blockIdx.x;
  const std::size_t dimension = threadIdx.x;
  const std::size_t qk_head = value_head / 3U;
  constexpr std::size_t kKOffset = kGdnQElements;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;
  const std::size_t q_index = qk_head * kGdnHeadDimension + dimension;
  const std::size_t k_index =
      kKOffset + qk_head * kGdnHeadDimension + dimension;
  const float q_value = decode_bf16_device(conv_qkv[q_index]);
  const float k_value = decode_bf16_device(conv_qkv[k_index]);

  partial[dimension] = q_value * q_value;
  __syncthreads();
  for (unsigned int stride = kGdnThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float q_scale =
      rsqrtf(partial[0] + l2_epsilon) *
      rsqrtf(static_cast<float>(kGdnHeadDimension));
  normalized_q[dimension] = q_value * q_scale;
  // Do not reuse partial until every thread has consumed the Q reduction.
  __syncthreads();

  partial[dimension] = k_value * k_value;
  __syncthreads();
  for (unsigned int stride = kGdnThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  normalized_k[dimension] =
      k_value * rsqrtf(partial[0] + l2_epsilon);

  if (threadIdx.x == 0U) {
    const float gate_input = decode_bf16_device(a[value_head]) +
                             decode_bf16_device(dt_bias[value_head]);
    const float g = -expf(decode_bf16_device(A_log[value_head])) *
                    stable_softplus_device(gate_input);
    recurrence_scalars[0] = expf(g);
    recurrence_scalars[1] =
        stable_sigmoid_device(decode_bf16_device(b[value_head]));
  }
  __syncthreads();

  const float alpha = recurrence_scalars[0];
  const float beta = recurrence_scalars[1];
  const std::size_t state_row_offset =
      value_head * kGdnHeadDimension * kGdnHeadDimension +
      dimension * kGdnHeadDimension;
  float prediction = 0.0F;
  for (std::size_t key_dimension = 0; key_dimension < kGdnHeadDimension;
       ++key_dimension) {
    const float decayed_state =
        alpha * decode_bf16_device(state_input[state_row_offset + key_dimension]);
    prediction = fmaf(decayed_state, normalized_k[key_dimension], prediction);
  }
  const float value = decode_bf16_device(
      conv_qkv[kVOffset + value_head * kGdnHeadDimension + dimension]);
  const float delta = (value - prediction) * beta;
  float result = 0.0F;
  for (std::size_t key_dimension = 0; key_dimension < kGdnHeadDimension;
       ++key_dimension) {
    const std::size_t state_index = state_row_offset + key_dimension;
    const float decayed_state =
        alpha * decode_bf16_device(state_input[state_index]);
    const float updated_state =
        fmaf(delta, normalized_k[key_dimension], decayed_state);
    state_output[state_index] = encode_bf16_device(updated_state);
    result = fmaf(updated_state, normalized_q[key_dimension], result);
  }
  output[value_head * kGdnHeadDimension + dimension] =
      encode_bf16_device(result);
}

__global__ void gated_delta_net_update_tile_kernel(
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
  __shared__ float normalized_q[kGdnHeadDimension];
  __shared__ float normalized_k[kGdnHeadDimension];
  __shared__ float partial[kGdnHeadDimension];
  __shared__ float recurrence_scalars[2];

  const std::size_t value_head = blockIdx.x;
  const std::size_t dimension = threadIdx.x;
  const std::size_t qk_head = value_head / 3U;
  constexpr std::size_t kKOffset = kGdnQElements;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;
  const std::size_t state_row_offset =
      value_head * kGdnHeadDimension * kGdnHeadDimension +
      dimension * kGdnHeadDimension;

  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t qkv_token_offset = token * kGdnQkvChannels;
    const std::size_t scalar_token_offset = token * kGdnValueHeadCount;
    const std::size_t output_token_offset = token * kGdnVElements;
    const std::size_t q_index =
        qkv_token_offset + qk_head * kGdnHeadDimension + dimension;
    const std::size_t k_index = qkv_token_offset + kKOffset +
                                qk_head * kGdnHeadDimension + dimension;
    const float q_value = decode_bf16_device(conv_qkv[q_index]);
    const float k_value = decode_bf16_device(conv_qkv[k_index]);

    partial[dimension] = q_value * q_value;
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    const float q_scale =
        rsqrtf(partial[0] + l2_epsilon) *
        rsqrtf(static_cast<float>(kGdnHeadDimension));
    normalized_q[dimension] = q_value * q_scale;
    // Match the single-token boundary before reusing partial for K.
    __syncthreads();

    partial[dimension] = k_value * k_value;
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    normalized_k[dimension] =
        k_value * rsqrtf(partial[0] + l2_epsilon);

    if (threadIdx.x == 0U) {
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
    const std::uint16_t* const recurrent_state =
        token == 0U ? state_input : state_output;
    float prediction = 0.0F;
    for (std::size_t key_dimension = 0;
         key_dimension < kGdnHeadDimension; ++key_dimension) {
      const float decayed_state = alpha * decode_bf16_device(
                                              recurrent_state[
                                                  state_row_offset +
                                                  key_dimension]);
      prediction =
          fmaf(decayed_state, normalized_k[key_dimension], prediction);
    }
    const float value = decode_bf16_device(
        conv_qkv[qkv_token_offset + kVOffset +
                 value_head * kGdnHeadDimension + dimension]);
    const float delta = (value - prediction) * beta;
    float result = 0.0F;
    for (std::size_t key_dimension = 0;
         key_dimension < kGdnHeadDimension; ++key_dimension) {
      const std::size_t state_index = state_row_offset + key_dimension;
      const float decayed_state =
          alpha * decode_bf16_device(recurrent_state[state_index]);
      const float updated_state =
          fmaf(delta, normalized_k[key_dimension], decayed_state);
      state_output[state_index] = encode_bf16_device(updated_state);
      result = fmaf(updated_state, normalized_q[key_dimension], result);
    }
    output[output_token_offset + value_head * kGdnHeadDimension + dimension] =
        encode_bf16_device(result);
    if (token + 1U < token_count) {
      __syncthreads();
    }
  }
}

// Preserve the reference kernel's scalar accumulation order while fixing its
// pathological state access pattern. A reference warp visits 32 different
// rows for a fixed key dimension, which turns each state load/store into 32
// widely separated memory transactions. Here one warp owns a row: all lanes
// cooperatively stage its 128 key dimensions, while lane zero performs the
// two left-to-right dot products. The arithmetic dependency chains therefore
// remain identical, but state traffic is fully coalesced.
__global__ void gated_delta_net_update_warp_parallel_kernel(
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
  constexpr unsigned int kWarpSize = 32U;
  constexpr unsigned int kWarpsPerBlock = 8U;
  constexpr unsigned int kFullWarpMask = 0xffffffffU;
  constexpr std::size_t kKeysPerLane =
      kGdnHeadDimension / static_cast<std::size_t>(kWarpSize);
  static_assert(kKeysPerLane == 4U);

  __shared__ float normalized_q[kGdnHeadDimension];
  __shared__ float normalized_k[kGdnHeadDimension];
  __shared__ float partial[kGdnHeadDimension];
  __shared__ float recurrence_scalars[2];
  __shared__ float row_scratch[kWarpsPerBlock * kGdnHeadDimension];

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const std::size_t value_head = blockIdx.x;
  const std::size_t qk_head = value_head / 3U;
  constexpr std::size_t kKOffset = kGdnQElements;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t qkv_token_offset = token * kGdnQkvChannels;
    const std::size_t scalar_token_offset = token * kGdnValueHeadCount;
    const std::size_t output_token_offset = token * kGdnVElements;

    float q_value = 0.0F;
    float k_value = 0.0F;
    if (thread < kGdnHeadDimension) {
      const std::size_t q_index =
          qkv_token_offset + qk_head * kGdnHeadDimension + thread;
      const std::size_t k_index =
          qkv_token_offset + kKOffset +
          qk_head * kGdnHeadDimension + thread;
      q_value = decode_bf16_device(conv_qkv[q_index]);
      k_value = decode_bf16_device(conv_qkv[k_index]);
      partial[thread] = q_value * q_value;
    }
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (thread < stride) {
        partial[thread] += partial[thread + stride];
      }
      __syncthreads();
    }
    if (thread < kGdnHeadDimension) {
      const float q_scale =
          rsqrtf(partial[0] + l2_epsilon) *
          rsqrtf(static_cast<float>(kGdnHeadDimension));
      normalized_q[thread] = q_value * q_scale;
    }
    // Match the reference boundary before reusing partial for K.
    __syncthreads();

    if (thread < kGdnHeadDimension) {
      partial[thread] = k_value * k_value;
    }
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (thread < stride) {
        partial[thread] += partial[thread + stride];
      }
      __syncthreads();
    }
    if (thread < kGdnHeadDimension) {
      normalized_k[thread] =
          k_value * rsqrtf(partial[0] + l2_epsilon);
    }
    if (thread == 0U) {
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
    const std::uint16_t* const recurrent_state =
        token == 0U ? state_input : state_output;
    float* const warp_scratch =
        row_scratch + static_cast<std::size_t>(warp) * kGdnHeadDimension;

    for (std::size_t row = warp; row < kGdnHeadDimension;
         row += kWarpsPerBlock) {
      const std::size_t state_row_offset =
          value_head * kGdnHeadDimension * kGdnHeadDimension +
          row * kGdnHeadDimension;
#pragma unroll
      for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
        const std::size_t key_dimension =
            lane + item * static_cast<std::size_t>(kWarpSize);
        warp_scratch[key_dimension] =
            alpha * decode_bf16_device(
                        recurrent_state[state_row_offset + key_dimension]);
      }
      __syncwarp(kFullWarpMask);

      float prediction = 0.0F;
      if (lane == 0U) {
#pragma unroll
        for (std::size_t key_dimension = 0U;
             key_dimension < kGdnHeadDimension; ++key_dimension) {
          prediction = fmaf(warp_scratch[key_dimension],
                            normalized_k[key_dimension], prediction);
        }
      }
      prediction = __shfl_sync(kFullWarpMask, prediction, 0);
      float delta = 0.0F;
      if (lane == 0U) {
        const float value = decode_bf16_device(
            conv_qkv[qkv_token_offset + kVOffset +
                     value_head * kGdnHeadDimension + row]);
        delta = (value - prediction) * beta;
      }
      delta = __shfl_sync(kFullWarpMask, delta, 0);

#pragma unroll
      for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
        const std::size_t key_dimension =
            lane + item * static_cast<std::size_t>(kWarpSize);
        const std::size_t state_index =
            state_row_offset + key_dimension;
        const float updated_state =
            fmaf(delta, normalized_k[key_dimension],
                 warp_scratch[key_dimension]);
        state_output[state_index] = encode_bf16_device(updated_state);
        warp_scratch[key_dimension] = updated_state;
      }
      __syncwarp(kFullWarpMask);

      if (lane == 0U) {
        float result = 0.0F;
#pragma unroll
        for (std::size_t key_dimension = 0U;
             key_dimension < kGdnHeadDimension; ++key_dimension) {
          result = fmaf(warp_scratch[key_dimension],
                        normalized_q[key_dimension], result);
        }
        output[output_token_offset +
               value_head * kGdnHeadDimension + row] =
            encode_bf16_device(result);
      }
      // Keep the cooperative writes for the next row from clobbering the
      // scratch values while lane zero is still consuming this row.
      __syncwarp(kFullWarpMask);
    }
    // Every row must be persisted before the next token uses state_output as
    // its recurrent input.
    __syncthreads();
  }
}

// Preserved two-row software-pipelining baseline used by same-binary tests.
// Each row visits key dimensions 0..127 in the reference order.
__global__ void gated_delta_net_update_warp_row_pair_kernel(
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
  constexpr unsigned int kWarpSize = 32U;
  constexpr unsigned int kWarpsPerBlock = 8U;
  constexpr unsigned int kRowsPerWarpBatch = 2U;
  constexpr unsigned int kFullWarpMask = 0xffffffffU;
  constexpr std::size_t kKeysPerLane =
      kGdnHeadDimension / static_cast<std::size_t>(kWarpSize);
  static_assert(kKeysPerLane == 4U);
  static_assert((kGdnHeadDimension %
                 (kWarpsPerBlock * kRowsPerWarpBatch)) == 0U);

  __shared__ float normalized_q[kGdnHeadDimension];
  __shared__ float normalized_k[kGdnHeadDimension];
  __shared__ float partial[kGdnHeadDimension];
  __shared__ float recurrence_scalars[2];
  __shared__ float
      row_scratch[kWarpsPerBlock * kRowsPerWarpBatch * kGdnHeadDimension];

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const std::size_t value_head = blockIdx.x;
  const std::size_t qk_head = value_head / 3U;
  constexpr std::size_t kKOffset = kGdnQElements;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t qkv_token_offset = token * kGdnQkvChannels;
    const std::size_t scalar_token_offset = token * kGdnValueHeadCount;
    const std::size_t output_token_offset = token * kGdnVElements;

    float q_value = 0.0F;
    float k_value = 0.0F;
    if (thread < kGdnHeadDimension) {
      const std::size_t q_index =
          qkv_token_offset + qk_head * kGdnHeadDimension + thread;
      const std::size_t k_index =
          qkv_token_offset + kKOffset +
          qk_head * kGdnHeadDimension + thread;
      q_value = decode_bf16_device(conv_qkv[q_index]);
      k_value = decode_bf16_device(conv_qkv[k_index]);
      partial[thread] = q_value * q_value;
    }
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (thread < stride) {
        partial[thread] += partial[thread + stride];
      }
      __syncthreads();
    }
    if (thread < kGdnHeadDimension) {
      const float q_scale =
          rsqrtf(partial[0] + l2_epsilon) *
          rsqrtf(static_cast<float>(kGdnHeadDimension));
      normalized_q[thread] = q_value * q_scale;
    }
    __syncthreads();

    if (thread < kGdnHeadDimension) {
      partial[thread] = k_value * k_value;
    }
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (thread < stride) {
        partial[thread] += partial[thread + stride];
      }
      __syncthreads();
    }
    if (thread < kGdnHeadDimension) {
      normalized_k[thread] =
          k_value * rsqrtf(partial[0] + l2_epsilon);
    }
    if (thread == 0U) {
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
    const std::uint16_t* const recurrent_state =
        token == 0U ? state_input : state_output;
    float* const first_scratch =
        row_scratch +
        static_cast<std::size_t>(warp * kRowsPerWarpBatch) *
            kGdnHeadDimension;
    float* const second_scratch = first_scratch + kGdnHeadDimension;

    for (std::size_t first_row =
             static_cast<std::size_t>(warp) * kRowsPerWarpBatch;
         first_row < kGdnHeadDimension;
         first_row += kWarpsPerBlock * kRowsPerWarpBatch) {
      const std::size_t second_row = first_row + 1U;
      const std::size_t first_state_row_offset =
          value_head * kGdnHeadDimension * kGdnHeadDimension +
          first_row * kGdnHeadDimension;
      const std::size_t second_state_row_offset =
          first_state_row_offset + kGdnHeadDimension;
#pragma unroll
      for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
        const std::size_t key_dimension =
            lane + item * static_cast<std::size_t>(kWarpSize);
        first_scratch[key_dimension] =
            alpha * decode_bf16_device(
                        recurrent_state[first_state_row_offset +
                                        key_dimension]);
        second_scratch[key_dimension] =
            alpha * decode_bf16_device(
                        recurrent_state[second_state_row_offset +
                                        key_dimension]);
      }
      __syncwarp(kFullWarpMask);

      float first_prediction = 0.0F;
      float second_prediction = 0.0F;
      if (lane == 0U) {
#pragma unroll
        for (std::size_t key_dimension = 0U;
             key_dimension < kGdnHeadDimension; ++key_dimension) {
          first_prediction =
              fmaf(first_scratch[key_dimension], normalized_k[key_dimension],
                   first_prediction);
          second_prediction =
              fmaf(second_scratch[key_dimension], normalized_k[key_dimension],
                   second_prediction);
        }
      }
      first_prediction =
          __shfl_sync(kFullWarpMask, first_prediction, 0);
      second_prediction =
          __shfl_sync(kFullWarpMask, second_prediction, 0);
      float first_delta = 0.0F;
      float second_delta = 0.0F;
      if (lane == 0U) {
        const std::size_t value_offset =
            qkv_token_offset + kVOffset +
            value_head * kGdnHeadDimension;
        const float first_value =
            decode_bf16_device(conv_qkv[value_offset + first_row]);
        const float second_value =
            decode_bf16_device(conv_qkv[value_offset + second_row]);
        first_delta = (first_value - first_prediction) * beta;
        second_delta = (second_value - second_prediction) * beta;
      }
      first_delta = __shfl_sync(kFullWarpMask, first_delta, 0);
      second_delta = __shfl_sync(kFullWarpMask, second_delta, 0);

#pragma unroll
      for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
        const std::size_t key_dimension =
            lane + item * static_cast<std::size_t>(kWarpSize);
        const float first_updated =
            fmaf(first_delta, normalized_k[key_dimension],
                 first_scratch[key_dimension]);
        const float second_updated =
            fmaf(second_delta, normalized_k[key_dimension],
                 second_scratch[key_dimension]);
        state_output[first_state_row_offset + key_dimension] =
            encode_bf16_device(first_updated);
        state_output[second_state_row_offset + key_dimension] =
            encode_bf16_device(second_updated);
        first_scratch[key_dimension] = first_updated;
        second_scratch[key_dimension] = second_updated;
      }
      __syncwarp(kFullWarpMask);

      if (lane == 0U) {
        float first_result = 0.0F;
        float second_result = 0.0F;
#pragma unroll
        for (std::size_t key_dimension = 0U;
             key_dimension < kGdnHeadDimension; ++key_dimension) {
          first_result =
              fmaf(first_scratch[key_dimension], normalized_q[key_dimension],
                   first_result);
          second_result =
              fmaf(second_scratch[key_dimension], normalized_q[key_dimension],
                   second_result);
        }
        const std::size_t output_offset =
            output_token_offset + value_head * kGdnHeadDimension;
        output[output_offset + first_row] =
            encode_bf16_device(first_result);
        output[output_offset + second_row] =
            encode_bf16_device(second_result);
      }
      __syncwarp(kFullWarpMask);
    }
    __syncthreads();
  }
}

// Preserved four-row software-pipeline baseline for same-binary measurements.
// Four independent state-row FMA chains execute in lane 0 while every row
// retains the exact K=0..127 accumulation order of the two-row baseline.
__global__ void gated_delta_net_update_warp_four_row_kernel(
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
  constexpr unsigned int kWarpSize = 32U;
  constexpr unsigned int kWarpsPerBlock = 8U;
  constexpr unsigned int kRowsPerWarpBatch = 4U;
  constexpr unsigned int kFullWarpMask = 0xffffffffU;
  constexpr std::size_t kKeysPerLane =
      kGdnHeadDimension / static_cast<std::size_t>(kWarpSize);
  static_assert(kKeysPerLane == 4U);
  static_assert((kGdnHeadDimension %
                 (kWarpsPerBlock * kRowsPerWarpBatch)) == 0U);

  __shared__ float normalized_q[kGdnHeadDimension];
  __shared__ float normalized_k[kGdnHeadDimension];
  __shared__ float partial[kGdnHeadDimension];
  __shared__ float recurrence_scalars[2];
  __shared__ float
      row_scratch[kWarpsPerBlock * kRowsPerWarpBatch * kGdnHeadDimension];

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const std::size_t value_head = blockIdx.x;
  const std::size_t qk_head = value_head / 3U;
  constexpr std::size_t kKOffset = kGdnQElements;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t qkv_token_offset = token * kGdnQkvChannels;
    const std::size_t scalar_token_offset = token * kGdnValueHeadCount;
    const std::size_t output_token_offset = token * kGdnVElements;

    float q_value = 0.0F;
    float k_value = 0.0F;
    if (thread < kGdnHeadDimension) {
      const std::size_t q_index =
          qkv_token_offset + qk_head * kGdnHeadDimension + thread;
      const std::size_t k_index =
          qkv_token_offset + kKOffset +
          qk_head * kGdnHeadDimension + thread;
      q_value = decode_bf16_device(conv_qkv[q_index]);
      k_value = decode_bf16_device(conv_qkv[k_index]);
      partial[thread] = q_value * q_value;
    }
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (thread < stride) {
        partial[thread] += partial[thread + stride];
      }
      __syncthreads();
    }
    if (thread < kGdnHeadDimension) {
      const float q_scale =
          rsqrtf(partial[0] + l2_epsilon) *
          rsqrtf(static_cast<float>(kGdnHeadDimension));
      normalized_q[thread] = q_value * q_scale;
    }
    __syncthreads();

    if (thread < kGdnHeadDimension) {
      partial[thread] = k_value * k_value;
    }
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (thread < stride) {
        partial[thread] += partial[thread + stride];
      }
      __syncthreads();
    }
    if (thread < kGdnHeadDimension) {
      normalized_k[thread] =
          k_value * rsqrtf(partial[0] + l2_epsilon);
    }
    if (thread == 0U) {
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
    const std::uint16_t* const recurrent_state =
        token == 0U ? state_input : state_output;
    float* const first_scratch =
        row_scratch +
        static_cast<std::size_t>(warp * kRowsPerWarpBatch) *
            kGdnHeadDimension;
    float* const second_scratch = first_scratch + kGdnHeadDimension;
    float* const third_scratch = second_scratch + kGdnHeadDimension;
    float* const fourth_scratch = third_scratch + kGdnHeadDimension;

    for (std::size_t first_row =
             static_cast<std::size_t>(warp) * kRowsPerWarpBatch;
         first_row < kGdnHeadDimension;
         first_row += kWarpsPerBlock * kRowsPerWarpBatch) {
      const std::size_t second_row = first_row + 1U;
      const std::size_t third_row = first_row + 2U;
      const std::size_t fourth_row = first_row + 3U;
      const std::size_t first_state_row_offset =
          value_head * kGdnHeadDimension * kGdnHeadDimension +
          first_row * kGdnHeadDimension;
      const std::size_t second_state_row_offset =
          first_state_row_offset + kGdnHeadDimension;
      const std::size_t third_state_row_offset =
          second_state_row_offset + kGdnHeadDimension;
      const std::size_t fourth_state_row_offset =
          third_state_row_offset + kGdnHeadDimension;
#pragma unroll
      for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
        const std::size_t key_dimension =
            lane + item * static_cast<std::size_t>(kWarpSize);
        first_scratch[key_dimension] =
            alpha * decode_bf16_device(
                        recurrent_state[first_state_row_offset +
                                        key_dimension]);
        second_scratch[key_dimension] =
            alpha * decode_bf16_device(
                        recurrent_state[second_state_row_offset +
                                        key_dimension]);
        third_scratch[key_dimension] =
            alpha * decode_bf16_device(
                        recurrent_state[third_state_row_offset +
                                        key_dimension]);
        fourth_scratch[key_dimension] =
            alpha * decode_bf16_device(
                        recurrent_state[fourth_state_row_offset +
                                        key_dimension]);
      }
      __syncwarp(kFullWarpMask);

      float first_prediction = 0.0F;
      float second_prediction = 0.0F;
      float third_prediction = 0.0F;
      float fourth_prediction = 0.0F;
      if (lane == 0U) {
#pragma unroll
        for (std::size_t key_dimension = 0U;
             key_dimension < kGdnHeadDimension; ++key_dimension) {
          first_prediction =
              fmaf(first_scratch[key_dimension], normalized_k[key_dimension],
                   first_prediction);
          second_prediction =
              fmaf(second_scratch[key_dimension], normalized_k[key_dimension],
                   second_prediction);
          third_prediction =
              fmaf(third_scratch[key_dimension], normalized_k[key_dimension],
                   third_prediction);
          fourth_prediction =
              fmaf(fourth_scratch[key_dimension], normalized_k[key_dimension],
                   fourth_prediction);
        }
      }
      first_prediction =
          __shfl_sync(kFullWarpMask, first_prediction, 0);
      second_prediction =
          __shfl_sync(kFullWarpMask, second_prediction, 0);
      third_prediction =
          __shfl_sync(kFullWarpMask, third_prediction, 0);
      fourth_prediction =
          __shfl_sync(kFullWarpMask, fourth_prediction, 0);

      float first_delta = 0.0F;
      float second_delta = 0.0F;
      float third_delta = 0.0F;
      float fourth_delta = 0.0F;
      if (lane == 0U) {
        const std::size_t value_offset =
            qkv_token_offset + kVOffset +
            value_head * kGdnHeadDimension;
        const float first_value =
            decode_bf16_device(conv_qkv[value_offset + first_row]);
        const float second_value =
            decode_bf16_device(conv_qkv[value_offset + second_row]);
        const float third_value =
            decode_bf16_device(conv_qkv[value_offset + third_row]);
        const float fourth_value =
            decode_bf16_device(conv_qkv[value_offset + fourth_row]);
        first_delta = (first_value - first_prediction) * beta;
        second_delta = (second_value - second_prediction) * beta;
        third_delta = (third_value - third_prediction) * beta;
        fourth_delta = (fourth_value - fourth_prediction) * beta;
      }
      first_delta = __shfl_sync(kFullWarpMask, first_delta, 0);
      second_delta = __shfl_sync(kFullWarpMask, second_delta, 0);
      third_delta = __shfl_sync(kFullWarpMask, third_delta, 0);
      fourth_delta = __shfl_sync(kFullWarpMask, fourth_delta, 0);

#pragma unroll
      for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
        const std::size_t key_dimension =
            lane + item * static_cast<std::size_t>(kWarpSize);
        const float first_updated =
            fmaf(first_delta, normalized_k[key_dimension],
                 first_scratch[key_dimension]);
        const float second_updated =
            fmaf(second_delta, normalized_k[key_dimension],
                 second_scratch[key_dimension]);
        const float third_updated =
            fmaf(third_delta, normalized_k[key_dimension],
                 third_scratch[key_dimension]);
        const float fourth_updated =
            fmaf(fourth_delta, normalized_k[key_dimension],
                 fourth_scratch[key_dimension]);
        state_output[first_state_row_offset + key_dimension] =
            encode_bf16_device(first_updated);
        state_output[second_state_row_offset + key_dimension] =
            encode_bf16_device(second_updated);
        state_output[third_state_row_offset + key_dimension] =
            encode_bf16_device(third_updated);
        state_output[fourth_state_row_offset + key_dimension] =
            encode_bf16_device(fourth_updated);
        first_scratch[key_dimension] = first_updated;
        second_scratch[key_dimension] = second_updated;
        third_scratch[key_dimension] = third_updated;
        fourth_scratch[key_dimension] = fourth_updated;
      }
      __syncwarp(kFullWarpMask);

      if (lane == 0U) {
        float first_result = 0.0F;
        float second_result = 0.0F;
        float third_result = 0.0F;
        float fourth_result = 0.0F;
#pragma unroll
        for (std::size_t key_dimension = 0U;
             key_dimension < kGdnHeadDimension; ++key_dimension) {
          first_result =
              fmaf(first_scratch[key_dimension], normalized_q[key_dimension],
                   first_result);
          second_result =
              fmaf(second_scratch[key_dimension], normalized_q[key_dimension],
                   second_result);
          third_result =
              fmaf(third_scratch[key_dimension], normalized_q[key_dimension],
                   third_result);
          fourth_result =
              fmaf(fourth_scratch[key_dimension], normalized_q[key_dimension],
                   fourth_result);
        }
        const std::size_t output_offset =
            output_token_offset + value_head * kGdnHeadDimension;
        output[output_offset + first_row] =
            encode_bf16_device(first_result);
        output[output_offset + second_row] =
            encode_bf16_device(second_result);
        output[output_offset + third_row] =
            encode_bf16_device(third_result);
        output[output_offset + fourth_row] =
            encode_bf16_device(fourth_result);
      }
      __syncwarp(kFullWarpMask);
    }
    __syncthreads();
  }
}

// Preserved four-row production predecessor. Lanes 0..3 each own one complete
// K=128 prediction/result chain, preserving the scalar FMA operand order
// within every row while exposing the independent chains to the warp scheduler.
// The extra scratch column rotates equal-key accesses across shared-memory
// banks for the four participating lanes.
__global__ void gated_delta_net_update_warp_four_row_lane_striped_kernel(
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
  constexpr unsigned int kWarpSize = 32U;
  constexpr unsigned int kWarpsPerBlock = 8U;
  constexpr unsigned int kRowsPerWarpBatch = 4U;
  constexpr unsigned int kFullWarpMask = 0xffffffffU;
  constexpr std::size_t kKeysPerLane =
      kGdnHeadDimension / static_cast<std::size_t>(kWarpSize);
  constexpr std::size_t kScratchRowStride = kGdnHeadDimension + 1U;
  static_assert(kKeysPerLane == 4U);
  static_assert((kGdnHeadDimension %
                 (kWarpsPerBlock * kRowsPerWarpBatch)) == 0U);
  static_assert(kScratchRowStride == 129U);

  __shared__ float normalized_q[kGdnHeadDimension];
  __shared__ float normalized_k[kGdnHeadDimension];
  __shared__ float partial[kGdnHeadDimension];
  __shared__ float recurrence_scalars[2];
  __shared__ float
      row_scratch[kWarpsPerBlock * kRowsPerWarpBatch * kScratchRowStride];

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const std::size_t value_head = blockIdx.x;
  const std::size_t qk_head = value_head / 3U;
  constexpr std::size_t kKOffset = kGdnQElements;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t qkv_token_offset = token * kGdnQkvChannels;
    const std::size_t scalar_token_offset = token * kGdnValueHeadCount;
    const std::size_t output_token_offset = token * kGdnVElements;

    float q_value = 0.0F;
    float k_value = 0.0F;
    if (thread < kGdnHeadDimension) {
      const std::size_t q_index =
          qkv_token_offset + qk_head * kGdnHeadDimension + thread;
      const std::size_t k_index =
          qkv_token_offset + kKOffset +
          qk_head * kGdnHeadDimension + thread;
      q_value = decode_bf16_device(conv_qkv[q_index]);
      k_value = decode_bf16_device(conv_qkv[k_index]);
      partial[thread] = q_value * q_value;
    }
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (thread < stride) {
        partial[thread] += partial[thread + stride];
      }
      __syncthreads();
    }
    if (thread < kGdnHeadDimension) {
      const float q_scale =
          rsqrtf(partial[0] + l2_epsilon) *
          rsqrtf(static_cast<float>(kGdnHeadDimension));
      normalized_q[thread] = q_value * q_scale;
    }
    __syncthreads();

    if (thread < kGdnHeadDimension) {
      partial[thread] = k_value * k_value;
    }
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (thread < stride) {
        partial[thread] += partial[thread + stride];
      }
      __syncthreads();
    }
    if (thread < kGdnHeadDimension) {
      normalized_k[thread] =
          k_value * rsqrtf(partial[0] + l2_epsilon);
    }
    if (thread == 0U) {
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
    const std::uint16_t* const recurrent_state =
        token == 0U ? state_input : state_output;
    float* const first_scratch =
        row_scratch +
        static_cast<std::size_t>(warp * kRowsPerWarpBatch) *
            kScratchRowStride;
    float* const second_scratch = first_scratch + kScratchRowStride;
    float* const third_scratch = second_scratch + kScratchRowStride;
    float* const fourth_scratch = third_scratch + kScratchRowStride;

    for (std::size_t first_row =
             static_cast<std::size_t>(warp) * kRowsPerWarpBatch;
         first_row < kGdnHeadDimension;
         first_row += kWarpsPerBlock * kRowsPerWarpBatch) {
      const std::size_t first_state_row_offset =
          value_head * kGdnHeadDimension * kGdnHeadDimension +
          first_row * kGdnHeadDimension;
      const std::size_t second_state_row_offset =
          first_state_row_offset + kGdnHeadDimension;
      const std::size_t third_state_row_offset =
          second_state_row_offset + kGdnHeadDimension;
      const std::size_t fourth_state_row_offset =
          third_state_row_offset + kGdnHeadDimension;
#pragma unroll
      for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
        const std::size_t key_dimension =
            lane + item * static_cast<std::size_t>(kWarpSize);
        first_scratch[key_dimension] =
            alpha * decode_bf16_device(
                        recurrent_state[first_state_row_offset +
                                        key_dimension]);
        second_scratch[key_dimension] =
            alpha * decode_bf16_device(
                        recurrent_state[second_state_row_offset +
                                        key_dimension]);
        third_scratch[key_dimension] =
            alpha * decode_bf16_device(
                        recurrent_state[third_state_row_offset +
                                        key_dimension]);
        fourth_scratch[key_dimension] =
            alpha * decode_bf16_device(
                        recurrent_state[fourth_state_row_offset +
                                        key_dimension]);
      }
      __syncwarp(kFullWarpMask);

      float lane_prediction = 0.0F;
      if (lane < kRowsPerWarpBatch) {
        const float* const lane_scratch =
            first_scratch + static_cast<std::size_t>(lane) *
                                kScratchRowStride;
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
      const float first_delta =
          __shfl_sync(kFullWarpMask, lane_delta, 0);
      const float second_delta =
          __shfl_sync(kFullWarpMask, lane_delta, 1);
      const float third_delta =
          __shfl_sync(kFullWarpMask, lane_delta, 2);
      const float fourth_delta =
          __shfl_sync(kFullWarpMask, lane_delta, 3);

#pragma unroll
      for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
        const std::size_t key_dimension =
            lane + item * static_cast<std::size_t>(kWarpSize);
        const float first_updated =
            fmaf(first_delta, normalized_k[key_dimension],
                 first_scratch[key_dimension]);
        const float second_updated =
            fmaf(second_delta, normalized_k[key_dimension],
                 second_scratch[key_dimension]);
        const float third_updated =
            fmaf(third_delta, normalized_k[key_dimension],
                 third_scratch[key_dimension]);
        const float fourth_updated =
            fmaf(fourth_delta, normalized_k[key_dimension],
                 fourth_scratch[key_dimension]);
        state_output[first_state_row_offset + key_dimension] =
            encode_bf16_device(first_updated);
        state_output[second_state_row_offset + key_dimension] =
            encode_bf16_device(second_updated);
        state_output[third_state_row_offset + key_dimension] =
            encode_bf16_device(third_updated);
        state_output[fourth_state_row_offset + key_dimension] =
            encode_bf16_device(fourth_updated);
        first_scratch[key_dimension] = first_updated;
        second_scratch[key_dimension] = second_updated;
        third_scratch[key_dimension] = third_updated;
        fourth_scratch[key_dimension] = fourth_updated;
      }
      __syncwarp(kFullWarpMask);

      if (lane < kRowsPerWarpBatch) {
        const float* const lane_scratch =
            first_scratch + static_cast<std::size_t>(lane) *
                                kScratchRowStride;
        float lane_result = 0.0F;
#pragma unroll
        for (std::size_t key_dimension = 0U;
             key_dimension < kGdnHeadDimension; ++key_dimension) {
          lane_result =
              fmaf(lane_scratch[key_dimension],
                   normalized_q[key_dimension], lane_result);
        }
        const std::size_t output_offset =
            output_token_offset + value_head * kGdnHeadDimension;
        output[output_offset + first_row + lane] =
            encode_bf16_device(lane_result);
      }
      __syncwarp(kFullWarpMask);
    }
    __syncthreads();
  }
}

// Production eight-row continuation of the lane-striped pipeline.
// Lanes 0..7 retain eight independent scalar FMA chains, while all lanes keep
// the coalesced state load/update phase. Doubling the rows per warp halves the
// outer state-row loop without changing any per-row arithmetic order.
__global__ void gated_delta_net_update_warp_eight_row_lane_striped_kernel(
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
  constexpr unsigned int kWarpSize = 32U;
  constexpr unsigned int kWarpsPerBlock = 8U;
  constexpr unsigned int kRowsPerWarpBatch = 8U;
  constexpr unsigned int kFullWarpMask = 0xffffffffU;
  constexpr std::size_t kKeysPerLane =
      kGdnHeadDimension / static_cast<std::size_t>(kWarpSize);
  constexpr std::size_t kScratchRowStride = kGdnHeadDimension + 1U;
  static_assert(kKeysPerLane == 4U);
  static_assert((kGdnHeadDimension %
                 (kWarpsPerBlock * kRowsPerWarpBatch)) == 0U);
  static_assert(kScratchRowStride == 129U);

  __shared__ float normalized_q[kGdnHeadDimension];
  __shared__ float normalized_k[kGdnHeadDimension];
  __shared__ float partial[kGdnHeadDimension];
  __shared__ float recurrence_scalars[2];
  __shared__ float
      row_scratch[kWarpsPerBlock * kRowsPerWarpBatch * kScratchRowStride];

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const std::size_t value_head = blockIdx.x;
  const std::size_t qk_head = value_head / 3U;
  constexpr std::size_t kKOffset = kGdnQElements;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t qkv_token_offset = token * kGdnQkvChannels;
    const std::size_t scalar_token_offset = token * kGdnValueHeadCount;
    const std::size_t output_token_offset = token * kGdnVElements;

    float q_value = 0.0F;
    float k_value = 0.0F;
    if (thread < kGdnHeadDimension) {
      const std::size_t q_index =
          qkv_token_offset + qk_head * kGdnHeadDimension + thread;
      const std::size_t k_index =
          qkv_token_offset + kKOffset +
          qk_head * kGdnHeadDimension + thread;
      q_value = decode_bf16_device(conv_qkv[q_index]);
      k_value = decode_bf16_device(conv_qkv[k_index]);
      partial[thread] = q_value * q_value;
    }
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (thread < stride) {
        partial[thread] += partial[thread + stride];
      }
      __syncthreads();
    }
    if (thread < kGdnHeadDimension) {
      const float q_scale =
          rsqrtf(partial[0] + l2_epsilon) *
          rsqrtf(static_cast<float>(kGdnHeadDimension));
      normalized_q[thread] = q_value * q_scale;
    }
    __syncthreads();

    if (thread < kGdnHeadDimension) {
      partial[thread] = k_value * k_value;
    }
    __syncthreads();
    for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
         stride >>= 1U) {
      if (thread < stride) {
        partial[thread] += partial[thread + stride];
      }
      __syncthreads();
    }
    if (thread < kGdnHeadDimension) {
      normalized_k[thread] =
          k_value * rsqrtf(partial[0] + l2_epsilon);
    }
    if (thread == 0U) {
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
    const std::uint16_t* const recurrent_state =
        token == 0U ? state_input : state_output;
    float* const warp_scratch =
        row_scratch +
        static_cast<std::size_t>(warp * kRowsPerWarpBatch) *
            kScratchRowStride;

    for (std::size_t first_row =
             static_cast<std::size_t>(warp) * kRowsPerWarpBatch;
         first_row < kGdnHeadDimension;
         first_row += kWarpsPerBlock * kRowsPerWarpBatch) {
      const std::size_t first_state_row_offset =
          value_head * kGdnHeadDimension * kGdnHeadDimension +
          first_row * kGdnHeadDimension;
#pragma unroll
      for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
        const std::size_t key_dimension =
            lane + item * static_cast<std::size_t>(kWarpSize);
#pragma unroll
        for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
          const std::size_t scratch_offset =
              static_cast<std::size_t>(row) * kScratchRowStride +
              key_dimension;
          const std::size_t state_offset =
              first_state_row_offset +
              static_cast<std::size_t>(row) * kGdnHeadDimension +
              key_dimension;
          warp_scratch[scratch_offset] =
              alpha * decode_bf16_device(recurrent_state[state_offset]);
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
          const std::size_t state_offset =
              first_state_row_offset +
              static_cast<std::size_t>(row) * kGdnHeadDimension +
              key_dimension;
          state_output[state_offset] = encode_bf16_device(updated);
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
    __syncthreads();
  }
}

// Exact M=1 composite. The GDN phase is intentionally a literal
// continuation of the production eight-row kernel. Each CTA owns one complete
// value head, so the following headwise normalization needs only a block
// barrier. The intermediate is first rounded to and stored as BF16 in output,
// then read back exactly as the ordered two-kernel reference does.
__global__ void
gated_delta_net_update_warp_eight_row_plain_rms_norm_silu_gate_kernel(
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
    std::uint16_t* const output) {
  constexpr unsigned int kWarpSize = 32U;
  constexpr unsigned int kWarpsPerBlock = 8U;
  constexpr unsigned int kRowsPerWarpBatch = 8U;
  constexpr unsigned int kFullWarpMask = 0xffffffffU;
  constexpr std::size_t kKeysPerLane =
      kGdnHeadDimension / static_cast<std::size_t>(kWarpSize);
  constexpr std::size_t kScratchRowStride = kGdnHeadDimension + 1U;
  static_assert(kKeysPerLane == 4U);
  static_assert((kGdnHeadDimension %
                 (kWarpsPerBlock * kRowsPerWarpBatch)) == 0U);
  static_assert(kScratchRowStride == 129U);

  __shared__ float normalized_q[kGdnHeadDimension];
  __shared__ float normalized_k[kGdnHeadDimension];
  __shared__ float partial[kGdnHeadDimension];
  __shared__ float recurrence_scalars[2];
  __shared__ float
      row_scratch[kWarpsPerBlock * kRowsPerWarpBatch * kScratchRowStride];

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const std::size_t value_head = blockIdx.x;
  const std::size_t qk_head = value_head / 3U;
  constexpr std::size_t kKOffset = kGdnQElements;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

  float q_value = 0.0F;
  float k_value = 0.0F;
  if (thread < kGdnHeadDimension) {
    const std::size_t q_index =
        qk_head * kGdnHeadDimension + thread;
    const std::size_t k_index =
        kKOffset + qk_head * kGdnHeadDimension + thread;
    q_value = decode_bf16_device(conv_qkv[q_index]);
    k_value = decode_bf16_device(conv_qkv[k_index]);
    partial[thread] = q_value * q_value;
  }
  __syncthreads();
  for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
       stride >>= 1U) {
    if (thread < stride) {
      partial[thread] += partial[thread + stride];
    }
    __syncthreads();
  }
  if (thread < kGdnHeadDimension) {
    const float q_scale =
        rsqrtf(partial[0] + l2_epsilon) *
        rsqrtf(static_cast<float>(kGdnHeadDimension));
    normalized_q[thread] = q_value * q_scale;
  }
  __syncthreads();

  if (thread < kGdnHeadDimension) {
    partial[thread] = k_value * k_value;
  }
  __syncthreads();
  for (unsigned int stride = kGdnThreads / 2U; stride != 0U;
       stride >>= 1U) {
    if (thread < stride) {
      partial[thread] += partial[thread + stride];
    }
    __syncthreads();
  }
  if (thread < kGdnHeadDimension) {
    normalized_k[thread] =
        k_value * rsqrtf(partial[0] + l2_epsilon);
  }
  if (thread == 0U) {
    const float gate_input =
        decode_bf16_device(a[value_head]) +
        decode_bf16_device(dt_bias[value_head]);
    const float g = -expf(decode_bf16_device(A_log[value_head])) *
                    stable_softplus_device(gate_input);
    recurrence_scalars[0] = expf(g);
    recurrence_scalars[1] =
        stable_sigmoid_device(decode_bf16_device(b[value_head]));
  }
  __syncthreads();

  const float alpha = recurrence_scalars[0];
  const float beta = recurrence_scalars[1];
  float* const warp_scratch =
      row_scratch +
      static_cast<std::size_t>(warp * kRowsPerWarpBatch) *
          kScratchRowStride;

  for (std::size_t first_row =
           static_cast<std::size_t>(warp) * kRowsPerWarpBatch;
       first_row < kGdnHeadDimension;
       first_row += kWarpsPerBlock * kRowsPerWarpBatch) {
    const std::size_t first_state_row_offset =
        value_head * kGdnHeadDimension * kGdnHeadDimension +
        first_row * kGdnHeadDimension;
#pragma unroll
    for (std::size_t item = 0U; item < kKeysPerLane; ++item) {
      const std::size_t key_dimension =
          lane + item * static_cast<std::size_t>(kWarpSize);
#pragma unroll
      for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
        const std::size_t scratch_offset =
            static_cast<std::size_t>(row) * kScratchRowStride +
            key_dimension;
        const std::size_t state_offset =
            first_state_row_offset +
            static_cast<std::size_t>(row) * kGdnHeadDimension +
            key_dimension;
        warp_scratch[scratch_offset] =
            alpha * decode_bf16_device(state_input[state_offset]);
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
          kVOffset + value_head * kGdnHeadDimension;
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
        const std::size_t state_offset =
            first_state_row_offset +
            static_cast<std::size_t>(row) * kGdnHeadDimension +
            key_dimension;
        state_output[state_offset] = encode_bf16_device(updated);
        warp_scratch[scratch_offset] = updated;
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
          value_head * kGdnHeadDimension;
      output[output_offset + first_row + lane] =
          encode_bf16_device(lane_result);
    }
    __syncwarp(kFullWarpMask);
  }
  __syncthreads();

  const std::size_t output_offset = value_head * kGdnHeadDimension;
  float sum = 0.0F;
  if (thread < kGdnHeadDimension) {
    const float value =
        decode_bf16_device(output[output_offset + thread]);
    sum = fmaf(value, value, sum);
  }
  // The state-row scratch is dead after the GDN phase. Reusing it here keeps
  // the prototype's static shared-memory footprint equal to production GDN.
  row_scratch[thread] = sum;
  __syncthreads();
  for (unsigned int stride = 256U / 2U; stride != 0U; stride >>= 1U) {
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
    float value =
        decode_bf16_device(output[output_offset + thread]) * inverse_rms *
        decode_bf16_device(norm_weight[thread]);
    const float gate_value =
        decode_bf16_device(silu_gate[output_offset + thread]);
    value *= gate_value / (1.0F + expf(-gate_value));
    output[output_offset + thread] = encode_bf16_device(value);
  }
}

}  // namespace

int launch_causal_conv1d_silu_update_reference_cuda(
    const std::uint16_t* const raw_qkv,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history_in_out,
    std::uint16_t* const conv_qkv_output,
    const GdnDimensions dimensions,
    void* const cuda_stream) noexcept {
  if (validate_dimensions(dimensions) != GdnStatus::kSuccess ||
      raw_qkv == nullptr || conv_weight == nullptr ||
      history_in_out == nullptr || conv_qkv_output == nullptr ||
      invalid_conv_alias(raw_qkv, conv_weight, history_in_out,
                         conv_qkv_output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned int kBlocks = static_cast<unsigned int>(
      kGdnQkvChannels / kConvThreads +
      (kGdnQkvChannels % kConvThreads != 0U ? 1U : 0U));
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  causal_conv1d_silu_update_kernel<<<kBlocks, kConvThreads, 0U, stream>>>(
      raw_qkv, conv_weight, history_in_out, conv_qkv_output);
  return static_cast<int>(cudaGetLastError());
}

int launch_causal_conv1d_silu_update_tile_reference_cuda(
    const std::uint16_t* const raw_qkv,
    const std::size_t token_count,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history_in_out,
    std::uint16_t* const conv_qkv_output,
    const GdnDimensions dimensions,
    void* const cuda_stream) noexcept {
  if (token_count == 1U) {
    return launch_causal_conv1d_silu_update_reference_cuda(
        raw_qkv, conv_weight, history_in_out, conv_qkv_output, dimensions,
        cuda_stream);
  }
  if (token_count == 0U || token_count > kGdnMaximumTileTokenCount ||
      validate_dimensions(dimensions) != GdnStatus::kSuccess ||
      raw_qkv == nullptr || conv_weight == nullptr ||
      history_in_out == nullptr || conv_qkv_output == nullptr ||
      invalid_conv_alias(raw_qkv, conv_weight, history_in_out,
                         conv_qkv_output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned int kBlocks = static_cast<unsigned int>(
      kGdnQkvChannels / kConvThreads +
      (kGdnQkvChannels % kConvThreads != 0U ? 1U : 0U));
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  causal_conv1d_silu_update_tile_kernel<<<kBlocks, kConvThreads, 0U, stream>>>(
      raw_qkv, token_count, conv_weight, history_in_out, conv_qkv_output);
  return static_cast<int>(cudaGetLastError());
}

int launch_gated_delta_net_update_reference_cuda(
    const std::uint16_t* const conv_qkv,
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
  if (validate_dimensions(dimensions) != GdnStatus::kSuccess ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr || A_log == nullptr ||
      dt_bias == nullptr || state_input == nullptr || state_output == nullptr ||
      output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kGdnThreads, 0U, stream>>>(
      conv_qkv, a, b, A_log, dt_bias, state_input, state_output, l2_epsilon,
      output);
  return static_cast<int>(cudaGetLastError());
}

int launch_gated_delta_net_update_tile_reference_cuda(
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
  if (token_count == 1U) {
    return launch_gated_delta_net_update_reference_cuda(
        conv_qkv, a, b, A_log, dt_bias, state_input, state_output, l2_epsilon,
        output, dimensions, cuda_stream);
  }
  if (token_count == 0U || token_count > kGdnMaximumTileTokenCount ||
      validate_dimensions(dimensions) != GdnStatus::kSuccess ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr || A_log == nullptr ||
      dt_bias == nullptr || state_input == nullptr || state_output == nullptr ||
      output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_tile_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kGdnThreads, 0U, stream>>>(
      conv_qkv, token_count, a, b, A_log, dt_bias, state_input, state_output,
      l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_gated_delta_net_update_warp_parallel_cuda(
    const std::uint16_t* const conv_qkv,
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
  if (validate_dimensions(dimensions) != GdnStatus::kSuccess ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr || A_log == nullptr ||
      dt_bias == nullptr || state_input == nullptr || state_output == nullptr ||
      output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned int kWarpParallelThreads = 256U;
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_warp_eight_row_lane_striped_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kWarpParallelThreads,
      0U, stream>>>(conv_qkv, 1U, a, b, A_log, dt_bias, state_input,
                    state_output, l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_gated_delta_net_update_tile_warp_parallel_cuda(
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
  if (token_count == 1U) {
    return launch_gated_delta_net_update_warp_parallel_cuda(
        conv_qkv, a, b, A_log, dt_bias, state_input, state_output, l2_epsilon,
        output, dimensions, cuda_stream);
  }
  if (token_count == 0U || token_count > kGdnMaximumTileTokenCount ||
      validate_dimensions(dimensions) != GdnStatus::kSuccess ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr || A_log == nullptr ||
      dt_bias == nullptr || state_input == nullptr || state_output == nullptr ||
      output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned int kWarpParallelThreads = 256U;
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_warp_eight_row_lane_striped_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kWarpParallelThreads,
      0U, stream>>>(conv_qkv, token_count, a, b, A_log, dt_bias, state_input,
                    state_output, l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

// Test-only entry points for same-binary mirrored performance measurements.
// They intentionally remain outside the public header.
int launch_gated_delta_net_update_tile_warp_baseline_test_cuda(
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
  if (token_count == 0U || token_count > kGdnMaximumTileTokenCount ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr || A_log == nullptr ||
      dt_bias == nullptr || state_input == nullptr || state_output == nullptr ||
      output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned int kWarpParallelThreads = 256U;
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_warp_parallel_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kWarpParallelThreads,
      0U, stream>>>(conv_qkv, token_count, a, b, A_log, dt_bias, state_input,
                    state_output, l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_gated_delta_net_update_tile_warp_row_pair_test_cuda(
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
  if (token_count == 0U || token_count > kGdnMaximumTileTokenCount ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr || A_log == nullptr ||
      dt_bias == nullptr || state_input == nullptr || state_output == nullptr ||
      output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned int kWarpParallelThreads = 256U;
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_warp_row_pair_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kWarpParallelThreads,
      0U, stream>>>(conv_qkv, token_count, a, b, A_log, dt_bias, state_input,
                    state_output, l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_gated_delta_net_update_tile_warp_four_row_test_cuda(
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
  if (token_count == 0U || token_count > kGdnMaximumTileTokenCount ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr || A_log == nullptr ||
      dt_bias == nullptr || state_input == nullptr || state_output == nullptr ||
      output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned int kWarpParallelThreads = 256U;
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_warp_four_row_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kWarpParallelThreads,
      0U, stream>>>(conv_qkv, token_count, a, b, A_log, dt_bias, state_input,
                    state_output, l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_gated_delta_net_update_tile_warp_four_row_lane_striped_test_cuda(
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
  if (token_count == 0U || token_count > kGdnMaximumTileTokenCount ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr || A_log == nullptr ||
      dt_bias == nullptr || state_input == nullptr || state_output == nullptr ||
      output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned int kWarpParallelThreads = 256U;
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_warp_four_row_lane_striped_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kWarpParallelThreads,
      0U, stream>>>(conv_qkv, token_count, a, b, A_log, dt_bias, state_input,
                    state_output, l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_gated_delta_net_update_tile_warp_eight_row_lane_striped_test_cuda(
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
  if (token_count == 0U || token_count > kGdnMaximumTileTokenCount ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr || A_log == nullptr ||
      dt_bias == nullptr || state_input == nullptr || state_output == nullptr ||
      output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned int kWarpParallelThreads = 256U;
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_warp_eight_row_lane_striped_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kWarpParallelThreads,
      0U, stream>>>(conv_qkv, token_count, a, b, A_log, dt_bias, state_input,
                    state_output, l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

namespace gdn_decode_detail {

int launch_gated_delta_net_update_plain_rms_norm_silu_gate_exact_cuda(
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
    void* const cuda_stream) noexcept {
  bool exact_fusion = false;
  const int validation = validate_gdn_plain_norm_silu_gate_composite(
      conv_qkv, a, b, A_log, dt_bias, state_input, state_output,
      l2_epsilon, norm_weight, silu_gate, kGdnValueHeadCount,
      kGdnHeadDimension, norm_epsilon, output, {}, &exact_fusion);
  if (validation != static_cast<int>(cudaSuccess) || !exact_fusion) {
    return validation != static_cast<int>(cudaSuccess)
               ? validation
               : static_cast<int>(cudaErrorInvalidValue);
  }

  constexpr unsigned int kWarpParallelThreads = 256U;
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_warp_eight_row_plain_rms_norm_silu_gate_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kWarpParallelThreads,
      0U, stream>>>(conv_qkv, a, b, A_log, dt_bias, state_input,
                    state_output, l2_epsilon, norm_weight, silu_gate,
                    norm_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace gdn_decode_detail

int launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
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
    const std::size_t norm_head_count,
    const std::size_t norm_head_dimension,
    const float norm_epsilon,
    std::uint16_t* const output,
    const GdnDimensions dimensions,
    void* const cuda_stream) noexcept {
  bool exact_fusion = false;
  const int validation = validate_gdn_plain_norm_silu_gate_composite(
      conv_qkv, a, b, A_log, dt_bias, state_input, state_output,
      l2_epsilon, norm_weight, silu_gate, norm_head_count,
      norm_head_dimension, norm_epsilon, output, dimensions, &exact_fusion);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (exact_fusion) {
    return gdn_decode_detail::
        launch_gated_delta_net_update_plain_rms_norm_silu_gate_exact_cuda(
            conv_qkv, a, b, A_log, dt_bias, state_input, state_output,
            l2_epsilon, norm_weight, silu_gate, norm_epsilon, output,
            cuda_stream);
  }

  int status = launch_gated_delta_net_update_warp_parallel_cuda(
      conv_qkv, a, b, A_log, dt_bias, state_input, state_output,
      l2_epsilon, output, dimensions, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
      output, norm_weight, silu_gate, norm_head_count, norm_head_dimension,
      norm_epsilon, output, cuda_stream);
  return status;
}

int query_gated_delta_net_update_warp_eight_row_plain_rms_norm_silu_gate_resources_test_cuda(
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
      gated_delta_net_update_warp_eight_row_plain_rms_norm_silu_gate_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      gated_delta_net_update_warp_eight_row_plain_rms_norm_silu_gate_kernel,
      256, 0U);
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
