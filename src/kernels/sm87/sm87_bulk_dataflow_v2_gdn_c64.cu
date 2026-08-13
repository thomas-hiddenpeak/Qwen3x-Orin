#include "q3x/kernels/sm87_bulk_dataflow_v2_gdn_c64.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kFullWarpMask = 0xffff'ffffU;
constexpr unsigned int kProducerWarps =
    static_cast<unsigned int>(kSm87BulkV2GdnProducerThreads / kWarpSize);
constexpr unsigned int kRecurrenceWarps =
    static_cast<unsigned int>(kSm87BulkV2GdnRecurrenceThreads / kWarpSize);
constexpr unsigned int kRowsPerRecurrenceWarp = 8U;
constexpr unsigned int kUpperRowOffset = 64U;
constexpr unsigned int kScratchRowStride = 129U;
constexpr unsigned int kKeyPairsPerLane = 2U;
constexpr unsigned int kConvWeightsPerQkRole =
    2U * static_cast<unsigned int>(kSm87TargetAotGdnStateKeyDimension) * 4U;
constexpr unsigned int kConvWeightsPerValueRole =
    static_cast<unsigned int>(kSm87TargetAotGdnStateValueDimension) * 4U;

static_assert(kProducerWarps == 4U);
static_assert(kRecurrenceWarps == 8U);
static_assert(kConvWeightsPerQkRole == 1'024U);
static_assert(kConvWeightsPerValueRole == 512U);

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t value) noexcept {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_rne(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
encode_bf16_pair_rne(const float first, const float second) noexcept {
  const unsigned int first_bits = __float_as_uint(first);
  const unsigned int second_bits = __float_as_uint(second);
  const __nv_bfloat162_raw converted = static_cast<__nv_bfloat162_raw>(
      __floats2bfloat162_rn(first, second));
  std::uint32_t packed = static_cast<std::uint32_t>(converted.x) |
                         (static_cast<std::uint32_t>(converted.y) << 16U);
  if ((first_bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    packed = (packed & 0xffff'0000U) |
             static_cast<std::uint32_t>((first_bits >> 16U) | 0x0040U);
  }
  if ((second_bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    packed =
        (packed & 0x0000'ffffU) |
        (static_cast<std::uint32_t>((second_bits >> 16U) | 0x0040U)
         << 16U);
  }
  return packed;
}

[[nodiscard]] __device__ __forceinline__ float stable_softplus(
    const float value) noexcept {
  return value > 20.0F ? value : log1pf(expf(value));
}

[[nodiscard]] __device__ __forceinline__ float stable_sigmoid(
    const float value) noexcept {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + expf(-value));
  }
  const float exponential = expf(value);
  return exponential / (1.0F + exponential);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t causal_raw_or_history(
    const std::uint16_t* const raw_qkvz,
    const std::uint16_t* const initial_history,
    const unsigned int channel,
    const unsigned int token,
    const unsigned int tap) noexcept {
  const int source_token = static_cast<int>(token) + static_cast<int>(tap) - 3;
  if (source_token >= 0) {
    return raw_qkvz[static_cast<std::size_t>(source_token) *
                           kSm87TargetAotGdnRawQkvZChannels +
                       channel];
  }
  return initial_history[
      static_cast<std::size_t>(channel) * kSm87TargetAotGdnConvHistory +
      static_cast<unsigned int>(source_token + 3)];
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t exact_conv_silu_bf16(
    const std::uint16_t* const raw_qkvz,
    const std::uint16_t* const initial_history,
    const std::uint16_t* const staged_weights,
    const unsigned int staged_weight_offset,
    const unsigned int channel,
    const unsigned int token) noexcept {
  float convolution = 0.0F;
#pragma unroll
  for (unsigned int tap = 0U; tap < 4U; ++tap) {
    convolution = fmaf(
        decode_bf16(causal_raw_or_history(raw_qkvz, initial_history, channel,
                                          token, tap)),
        decode_bf16(staged_weights[staged_weight_offset + tap]), convolution);
  }
  return encode_bf16_rne(convolution / (1.0F + expf(-convolution)));
}

// One CTA owns four token rows and one of 64 producer roles.  Roles 0..15
// publish one normalized Q/K pair each; roles 16..63 publish one V head and
// its exact FP32 alpha/beta scalars.  Token rows are independent because the
// width-four convolution consumes raw projection history, never a prior
// convolved output.
__global__ __launch_bounds__(kSm87BulkV2GdnProducerThreads, 3)
void prepare_exact_c64_kernel(
    const std::uint16_t* const raw_qkvz,
    const std::uint16_t* const interleaved_ab,
    const std::uint16_t* const conv_weight,
    const std::uint16_t* const initial_conv_history,
    const std::uint16_t* const a_log,
    const std::uint16_t* const dt_bias,
    const std::uint32_t l2_epsilon_bits,
    float* const normalized_q,
    float* const normalized_k,
    std::uint16_t* const prepared_v,
    float* const alpha,
    float* const beta,
    std::uint16_t* const final_conv_history) {
  __shared__ std::uint16_t staged_weights[kConvWeightsPerQkRole];
  const unsigned int role = blockIdx.y;
  const bool qk_role = role < kSm87BulkV2GdnQkProducerRoles;
  const unsigned int staged_weight_count =
      qk_role ? kConvWeightsPerQkRole : kConvWeightsPerValueRole;

  for (unsigned int item = threadIdx.x; item < staged_weight_count;
       item += blockDim.x) {
    const unsigned int local_channel = item / 4U;
    const unsigned int tap = item % 4U;
    unsigned int canonical_channel = 0U;
    if (qk_role) {
      canonical_channel =
          local_channel < kSm87TargetAotGdnStateKeyDimension
              ? role * kSm87TargetAotGdnStateKeyDimension + local_channel
              : static_cast<unsigned int>(kSm87TargetAotGdnRawKOffset) +
                    role * kSm87TargetAotGdnStateKeyDimension +
                    local_channel - kSm87TargetAotGdnStateKeyDimension;
    } else {
      const unsigned int value_head =
          role - static_cast<unsigned int>(kSm87BulkV2GdnQkProducerRoles);
      canonical_channel =
          static_cast<unsigned int>(kSm87TargetAotGdnRawVOffset) +
          value_head * kSm87TargetAotGdnStateValueDimension + local_channel;
    }
    staged_weights[item] =
        conv_weight[static_cast<std::size_t>(canonical_channel) * 4U + tap];
  }
  __syncthreads();

  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int token =
      blockIdx.x * static_cast<unsigned int>(kSm87BulkV2GdnProducerRowsPerCta) +
      warp;
  const float l2_epsilon = __uint_as_float(l2_epsilon_bits);

  if (qk_role) {
    float q_values[4U];
    float k_values[4U];
#pragma unroll
    for (unsigned int item = 0U; item < 4U; ++item) {
      const unsigned int dimension = lane + item * kWarpSize;
      const unsigned int q_channel =
          role * kSm87TargetAotGdnStateKeyDimension + dimension;
      const unsigned int k_channel =
          static_cast<unsigned int>(kSm87TargetAotGdnRawKOffset) + q_channel;
      q_values[item] = decode_bf16(exact_conv_silu_bf16(
          raw_qkvz, initial_conv_history, staged_weights, dimension * 4U,
          q_channel, token));
      k_values[item] = decode_bf16(exact_conv_silu_bf16(
          raw_qkvz, initial_conv_history, staged_weights,
          (kSm87TargetAotGdnStateKeyDimension + dimension) * 4U, k_channel,
          token));
    }

    // Keep the target-AOT source-level operation boundaries: four rounded
    // FP32 multiplies, then (0+64)/(32+96), then the shuffle tree.  Writing
    // these as two multiply-add expressions would authorize contraction not
    // present in the pinned contract.
    const float q_first_square = q_values[0U] * q_values[0U];
    const float q_second_square = q_values[1U] * q_values[1U];
    const float q_third_square = q_values[2U] * q_values[2U];
    const float q_fourth_square = q_values[3U] * q_values[3U];
    const float q_first_pair = q_first_square + q_third_square;
    const float q_second_pair = q_second_square + q_fourth_square;
    float q_sum = q_first_pair + q_second_pair;
    const float k_first_square = k_values[0U] * k_values[0U];
    const float k_second_square = k_values[1U] * k_values[1U];
    const float k_third_square = k_values[2U] * k_values[2U];
    const float k_fourth_square = k_values[3U] * k_values[3U];
    const float k_first_pair = k_first_square + k_third_square;
    const float k_second_pair = k_second_square + k_fourth_square;
    float k_sum = k_first_pair + k_second_pair;
#pragma unroll
    for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
      q_sum += __shfl_down_sync(kFullWarpMask, q_sum, stride);
      k_sum += __shfl_down_sync(kFullWarpMask, k_sum, stride);
    }
    float q_scale = 0.0F;
    float k_scale = 0.0F;
    if (lane == 0U) {
      q_scale = rsqrtf(q_sum + l2_epsilon);
      q_scale *= rsqrtf(
          static_cast<float>(kSm87TargetAotGdnStateKeyDimension));
      k_scale = rsqrtf(k_sum + l2_epsilon);
    }
    q_scale = __shfl_sync(kFullWarpMask, q_scale, 0U);
    k_scale = __shfl_sync(kFullWarpMask, k_scale, 0U);
    const std::size_t base =
        (static_cast<std::size_t>(role) * kSm87BulkV2GdnC64Tokens + token) *
        kSm87TargetAotGdnStateKeyDimension;
#pragma unroll
    for (unsigned int item = 0U; item < 4U; ++item) {
      const unsigned int dimension = lane + item * kWarpSize;
      normalized_q[base + dimension] = q_values[item] * q_scale;
      normalized_k[base + dimension] = k_values[item] * k_scale;
    }
  } else {
    const unsigned int value_head =
        role - static_cast<unsigned int>(kSm87BulkV2GdnQkProducerRoles);
    const std::size_t base =
        (static_cast<std::size_t>(value_head) * kSm87BulkV2GdnC64Tokens +
         token) *
        kSm87TargetAotGdnStateValueDimension;
#pragma unroll
    for (unsigned int item = 0U; item < 4U; ++item) {
      const unsigned int dimension = lane + item * kWarpSize;
      const unsigned int channel =
          static_cast<unsigned int>(kSm87TargetAotGdnRawVOffset) +
          value_head * kSm87TargetAotGdnStateValueDimension + dimension;
      prepared_v[base + dimension] = exact_conv_silu_bf16(
          raw_qkvz, initial_conv_history, staged_weights, dimension * 4U,
          channel, token);
    }
    if (lane == 0U) {
      const std::size_t ab_row =
          static_cast<std::size_t>(token) * kSm87TargetAotGdnAbChannels;
      const float gate_input =
          decode_bf16(interleaved_ab[ab_row + value_head]) +
          decode_bf16(dt_bias[value_head]);
      const float g = -expf(decode_bf16(a_log[value_head])) *
                      stable_softplus(gate_input);
      const std::size_t scalar =
          static_cast<std::size_t>(value_head) *
              kSm87BulkV2GdnC64Tokens +
          token;
      alpha[scalar] = expf(g);
      beta[scalar] = stable_sigmoid(decode_bf16(interleaved_ab[
          ab_row + kSm87TargetAotGdnValueHeads + value_head]));
    }
  }

  // Only the final token tile publishes history.  Its addresses are disjoint
  // across all 64 roles, and the source is the immutable raw projection.
  if (blockIdx.x + 1U ==
      kSm87BulkV2GdnC64Tokens / kSm87BulkV2GdnProducerRowsPerCta) {
    if (qk_role && threadIdx.x < kSm87TargetAotGdnStateKeyDimension) {
      const unsigned int q_channel =
          role * kSm87TargetAotGdnStateKeyDimension + threadIdx.x;
      const unsigned int k_channel =
          static_cast<unsigned int>(kSm87TargetAotGdnRawKOffset) + q_channel;
#pragma unroll
      for (unsigned int history = 0U;
           history < kSm87TargetAotGdnConvHistory; ++history) {
        const unsigned int history_token =
            static_cast<unsigned int>(kSm87BulkV2GdnC64Tokens -
                                      kSm87TargetAotGdnConvHistory) +
            history;
        final_conv_history[static_cast<std::size_t>(q_channel) *
                               kSm87TargetAotGdnConvHistory +
                           history] =
            raw_qkvz[static_cast<std::size_t>(history_token) *
                         kSm87TargetAotGdnRawQkvZChannels +
                     q_channel];
        final_conv_history[static_cast<std::size_t>(k_channel) *
                               kSm87TargetAotGdnConvHistory +
                           history] =
            raw_qkvz[static_cast<std::size_t>(history_token) *
                         kSm87TargetAotGdnRawQkvZChannels +
                     k_channel];
      }
    } else if (!qk_role &&
               threadIdx.x < kSm87TargetAotGdnStateValueDimension) {
      const unsigned int value_head =
          role - static_cast<unsigned int>(kSm87BulkV2GdnQkProducerRoles);
      const unsigned int channel =
          static_cast<unsigned int>(kSm87TargetAotGdnRawVOffset) +
          value_head * kSm87TargetAotGdnStateValueDimension + threadIdx.x;
#pragma unroll
      for (unsigned int history = 0U;
           history < kSm87TargetAotGdnConvHistory; ++history) {
        const unsigned int history_token =
            static_cast<unsigned int>(kSm87BulkV2GdnC64Tokens -
                                      kSm87TargetAotGdnConvHistory) +
            history;
        final_conv_history[static_cast<std::size_t>(channel) *
                               kSm87TargetAotGdnConvHistory +
                           history] =
            raw_qkvz[static_cast<std::size_t>(history_token) *
                         kSm87TargetAotGdnRawQkvZChannels +
                     channel];
      }
    }
  }
}

struct HeadRegisterState final {
  std::uint32_t lower[kRowsPerRecurrenceWarp][kKeyPairsPerLane];
  std::uint32_t upper[kRowsPerRecurrenceWarp][kKeyPairsPerLane];
};

struct RecurrenceSharedStorage final {
  float normalized_q[kSm87TargetAotGdnStateKeyDimension];
  float normalized_k[kSm87TargetAotGdnStateKeyDimension];
  float scalars[2U];
  float row_scratch[kRecurrenceWarps * kRowsPerRecurrenceWarp *
                    kScratchRowStride];
};

static_assert(sizeof(RecurrenceSharedStorage) == 34'056U);

__device__ __forceinline__ void load_head_state(
    HeadRegisterState& state,
    const unsigned int value_head,
    const std::uint16_t* const initial_state) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int first_row = warp * kRowsPerRecurrenceWarp;
  const std::size_t head_offset =
      static_cast<std::size_t>(value_head) *
      kSm87TargetAotGdnStateValuesPerHead;
#pragma unroll
  for (unsigned int row = 0U; row < kRowsPerRecurrenceWarp; ++row) {
#pragma unroll
    for (unsigned int pair = 0U; pair < kKeyPairsPerLane; ++pair) {
      const unsigned int first_key = lane + pair * 2U * kWarpSize;
      const unsigned int second_key = first_key + kWarpSize;
      const unsigned int lower_row = first_row + row;
      const unsigned int upper_row = lower_row + kUpperRowOffset;
      const std::size_t lower =
          head_offset + static_cast<std::size_t>(lower_row) *
                            kSm87TargetAotGdnStateKeyDimension;
      const std::size_t upper =
          head_offset + static_cast<std::size_t>(upper_row) *
                            kSm87TargetAotGdnStateKeyDimension;
      state.lower[row][pair] =
          static_cast<std::uint32_t>(initial_state[lower + first_key]) |
          (static_cast<std::uint32_t>(initial_state[lower + second_key])
           << 16U);
      state.upper[row][pair] =
          static_cast<std::uint32_t>(initial_state[upper + first_key]) |
          (static_cast<std::uint32_t>(initial_state[upper + second_key])
           << 16U);
    }
  }
}

__device__ __forceinline__ void store_head_state(
    const HeadRegisterState& state,
    const unsigned int value_head,
    std::uint16_t* const destination) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int first_row = warp * kRowsPerRecurrenceWarp;
  const std::size_t head_offset =
      static_cast<std::size_t>(value_head) *
      kSm87TargetAotGdnStateValuesPerHead;
#pragma unroll
  for (unsigned int row = 0U; row < kRowsPerRecurrenceWarp; ++row) {
#pragma unroll
    for (unsigned int pair = 0U; pair < kKeyPairsPerLane; ++pair) {
      const unsigned int first_key = lane + pair * 2U * kWarpSize;
      const unsigned int second_key = first_key + kWarpSize;
      const unsigned int lower_row = first_row + row;
      const unsigned int upper_row = lower_row + kUpperRowOffset;
      const std::size_t lower =
          head_offset + static_cast<std::size_t>(lower_row) *
                            kSm87TargetAotGdnStateKeyDimension;
      const std::size_t upper =
          head_offset + static_cast<std::size_t>(upper_row) *
                            kSm87TargetAotGdnStateKeyDimension;
      const std::uint32_t lower_word = state.lower[row][pair];
      const std::uint32_t upper_word = state.upper[row][pair];
      destination[lower + first_key] =
          static_cast<std::uint16_t>(lower_word);
      destination[lower + second_key] =
          static_cast<std::uint16_t>(lower_word >> 16U);
      destination[upper + first_key] =
          static_cast<std::uint16_t>(upper_word);
      destination[upper + second_key] =
          static_cast<std::uint16_t>(upper_word >> 16U);
    }
  }
}

template <bool kStoreStateTrace>
__global__ __launch_bounds__(kSm87BulkV2GdnRecurrenceThreads, 3)
void recur_exact_c64_kernel(
    const float* const prepared_q,
    const float* const prepared_k,
    const std::uint16_t* const prepared_v,
    const float* const alpha,
    const float* const beta,
    const std::uint16_t* const initial_state,
    std::uint16_t* const raw_output,
    std::uint16_t* const final_state,
    std::uint16_t* const state_trace) {
  __shared__ RecurrenceSharedStorage storage;
  const unsigned int value_head = blockIdx.x;
  const unsigned int qk_group =
      value_head / kSm87TargetAotGdnValueHeadsPerQkGroup;
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_first_row = warp * kRowsPerRecurrenceWarp;
  HeadRegisterState state;
  load_head_state(state, value_head, initial_state);

#pragma unroll 1
  for (unsigned int token = 0U; token < kSm87BulkV2GdnC64Tokens; ++token) {
    if (threadIdx.x < kSm87TargetAotGdnStateKeyDimension) {
      const std::size_t qk =
          (static_cast<std::size_t>(qk_group) *
               kSm87BulkV2GdnC64Tokens +
           token) *
              kSm87TargetAotGdnStateKeyDimension +
          threadIdx.x;
      storage.normalized_q[threadIdx.x] = prepared_q[qk];
      storage.normalized_k[threadIdx.x] = prepared_k[qk];
    }
    if (threadIdx.x == 0U) {
      const std::size_t scalar =
          static_cast<std::size_t>(value_head) *
              kSm87BulkV2GdnC64Tokens +
          token;
      storage.scalars[0U] = alpha[scalar];
      storage.scalars[1U] = beta[scalar];
    }
    __syncthreads();

    const float token_alpha = storage.scalars[0U];
    const float token_beta = storage.scalars[1U];
    float* const warp_scratch =
        storage.row_scratch +
        static_cast<std::size_t>(warp * kRowsPerRecurrenceWarp) *
            kScratchRowStride;

#pragma unroll 1
    for (unsigned int batch = 0U; batch < 2U; ++batch) {
      const unsigned int first_row =
          warp_first_row + batch * kUpperRowOffset;
#pragma unroll
      for (unsigned int pair = 0U; pair < kKeyPairsPerLane; ++pair) {
        const unsigned int first_key = lane + pair * 2U * kWarpSize;
        const unsigned int second_key = first_key + kWarpSize;
#pragma unroll
        for (unsigned int row = 0U; row < kRowsPerRecurrenceWarp; ++row) {
          const std::uint32_t word =
              batch == 0U ? state.lower[row][pair] : state.upper[row][pair];
          const std::size_t scratch =
              static_cast<std::size_t>(row) * kScratchRowStride;
          warp_scratch[scratch + first_key] =
              token_alpha *
              decode_bf16(static_cast<std::uint16_t>(word));
          warp_scratch[scratch + second_key] =
              token_alpha *
              decode_bf16(static_cast<std::uint16_t>(word >> 16U));
        }
      }
      __syncwarp(kFullWarpMask);

      float lane_prediction = 0.0F;
      if (lane < kRowsPerRecurrenceWarp) {
        const float* const row =
            warp_scratch + static_cast<std::size_t>(lane) * kScratchRowStride;
#pragma unroll
        for (unsigned int key = 0U;
             key < kSm87TargetAotGdnStateKeyDimension; ++key) {
          lane_prediction =
              fmaf(row[key], storage.normalized_k[key], lane_prediction);
        }
      }

      float lane_delta = 0.0F;
      if (lane < kRowsPerRecurrenceWarp) {
        const unsigned int value_dimension = first_row + lane;
        const std::size_t value =
            (static_cast<std::size_t>(value_head) *
                 kSm87BulkV2GdnC64Tokens +
             token) *
                kSm87TargetAotGdnStateValueDimension +
            value_dimension;
        lane_delta =
            (decode_bf16(prepared_v[value]) - lane_prediction) * token_beta;
      }
      float deltas[kRowsPerRecurrenceWarp];
#pragma unroll
      for (unsigned int row = 0U; row < kRowsPerRecurrenceWarp; ++row) {
        deltas[row] = __shfl_sync(kFullWarpMask, lane_delta, row);
      }

#pragma unroll
      for (unsigned int pair = 0U; pair < kKeyPairsPerLane; ++pair) {
        const unsigned int first_key = lane + pair * 2U * kWarpSize;
        const unsigned int second_key = first_key + kWarpSize;
#pragma unroll
        for (unsigned int row = 0U; row < kRowsPerRecurrenceWarp; ++row) {
          const std::size_t scratch =
              static_cast<std::size_t>(row) * kScratchRowStride;
          const float first_updated =
              fmaf(deltas[row], storage.normalized_k[first_key],
                   warp_scratch[scratch + first_key]);
          const float second_updated =
              fmaf(deltas[row], storage.normalized_k[second_key],
                   warp_scratch[scratch + second_key]);
          const std::uint32_t rounded =
              encode_bf16_pair_rne(first_updated, second_updated);
          if (batch == 0U) {
            state.lower[row][pair] = rounded;
          } else {
            state.upper[row][pair] = rounded;
          }
          // Same-token output consumes the pre-round FP32 updates.
          warp_scratch[scratch + first_key] = first_updated;
          warp_scratch[scratch + second_key] = second_updated;
        }
      }
      __syncwarp(kFullWarpMask);

      if (lane < kRowsPerRecurrenceWarp) {
        const float* const row =
            warp_scratch + static_cast<std::size_t>(lane) * kScratchRowStride;
        float result = 0.0F;
#pragma unroll
        for (unsigned int key = 0U;
             key < kSm87TargetAotGdnStateKeyDimension; ++key) {
          result = fmaf(row[key], storage.normalized_q[key], result);
        }
        const unsigned int value_dimension = first_row + lane;
        const std::size_t destination =
            (static_cast<std::size_t>(token) *
                 kSm87TargetAotGdnValueHeads +
             value_head) *
                kSm87TargetAotGdnStateValueDimension +
            value_dimension;
        raw_output[destination] = encode_bf16_rne(result);
      }
      __syncwarp(kFullWarpMask);
    }
    __syncthreads();

    if constexpr (kStoreStateTrace) {
      store_head_state(
          state, value_head,
          state_trace + static_cast<std::size_t>(token) *
                            kSm87TargetAotGdnTotalStateBytes /
                            kSm87TargetAotGdnBf16Bytes);
    }
  }
  store_head_state(state, value_head, final_state);
}

__global__ __launch_bounds__(kSm87BulkV2GdnEpilogueThreads, 3)
void rms_norm_silu_rows8_exact_kernel(
    const std::uint16_t* const raw_output,
    const std::uint16_t* const raw_qkvz,
    const std::uint16_t* const norm_weight,
    const std::uint32_t norm_epsilon_bits,
    std::uint16_t* const output) {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int row =
      blockIdx.x * static_cast<unsigned int>(kSm87BulkV2GdnEpilogueRowsPerCta) +
      warp;
  const unsigned int token =
      row / static_cast<unsigned int>(kSm87TargetAotGdnValueHeads);
  const unsigned int value_head =
      row % static_cast<unsigned int>(kSm87TargetAotGdnValueHeads);
  const std::size_t raw_base =
      static_cast<std::size_t>(row) * kSm87TargetAotGdnStateValueDimension;
  float raw_values[4U];
  float squares[4U];
#pragma unroll
  for (unsigned int item = 0U; item < 4U; ++item) {
    const unsigned int dimension = lane + item * kWarpSize;
    raw_values[item] = decode_bf16(raw_output[raw_base + dimension]);
    squares[item] = fmaf(raw_values[item], raw_values[item], 0.0F);
  }
  const float first_pair = squares[0U] + squares[2U];
  const float second_pair = squares[1U] + squares[3U];
  float warp_sum = first_pair + second_pair;
#pragma unroll
  for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
    warp_sum += __shfl_down_sync(kFullWarpMask, warp_sum, stride);
  }
  float inverse_rms = 0.0F;
  if (lane == 0U) {
    inverse_rms = rsqrtf(
        warp_sum /
                static_cast<float>(kSm87TargetAotGdnStateValueDimension) +
            __uint_as_float(norm_epsilon_bits));
  }
  inverse_rms = __shfl_sync(kFullWarpMask, inverse_rms, 0U);
#pragma unroll
  for (unsigned int item = 0U; item < 4U; ++item) {
    const unsigned int dimension = lane + item * kWarpSize;
    float value = raw_values[item] * inverse_rms;
    value *= decode_bf16(norm_weight[dimension]);
    const std::size_t z_index =
        static_cast<std::size_t>(token) *
            kSm87TargetAotGdnRawQkvZChannels +
        kSm87TargetAotGdnRawZOffset +
        value_head * kSm87TargetAotGdnStateValueDimension + dimension;
    const float z = decode_bf16(raw_qkvz[z_index]);
    value *= z / (1.0F + expf(-z));
    const std::size_t destination =
        static_cast<std::size_t>(token) * kSm87TargetAotGdnOutputChannels +
        value_head * kSm87TargetAotGdnStateValueDimension + dimension;
    output[destination] = encode_bf16_rne(value);
  }
}

[[nodiscard]] cudaError_t validate_fixed_device(int* const device) noexcept {
  if (device == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaGetDevice(device);
  if (status != cudaSuccess) {
    return status;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, *device);
  if (status != cudaSuccess) {
    return status;
  }
  return properties.major == 8 && properties.minor == 7 &&
                 properties.multiProcessorCount == 16
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] bool exact_device_pointer(const void* const pointer,
                                        const int device) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  return status == cudaSuccess && attributes.type == cudaMemoryTypeDevice &&
         attributes.device == device;
}

template <class Kernel>
[[nodiscard]] cudaError_t inspect_kernel(
    const Kernel kernel,
    const int threads,
    const int grid_ctas,
    Sm87BulkV2GdnKernelResources* const resources) noexcept {
  if (resources == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, threads, 0U);
  if (status != cudaSuccess) {
    return status;
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->threads_per_block = threads;
  resources->physical_grid_ctas = grid_ctas;
  return cudaSuccess;
}

[[nodiscard]] bool all_device_pointers(
    const Sm87BulkV2GdnC64Arguments& arguments,
    const int device) noexcept {
  const void* const pointers[] = {
      arguments.raw_qkvz,
      arguments.interleaved_ab,
      arguments.conv_weight,
      arguments.initial_conv_history,
      arguments.a_log,
      arguments.dt_bias,
      arguments.norm_weight,
      arguments.initial_recurrent_state,
      arguments.normalized_q,
      arguments.normalized_k,
      arguments.prepared_v,
      arguments.alpha,
      arguments.beta,
      arguments.raw_output,
      arguments.output,
      arguments.final_conv_history,
      arguments.final_recurrent_state,
  };
  for (const void* const pointer : pointers) {
    if (!exact_device_pointer(pointer, device)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] int launch_impl(
    const Sm87BulkV2GdnC64Arguments& arguments,
    std::uint16_t* const state_trace) noexcept {
  if (!sm87_bulk_v2_gdn_c64_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int device = -1;
  cudaError_t status = validate_fixed_device(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (!all_device_pointers(arguments, device) ||
      (state_trace != nullptr &&
       (!sm87_bulk_v2_gdn_pointer_aligned(state_trace) ||
        !exact_device_pointer(state_trace, device)))) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }

  if (state_trace != nullptr &&
      !sm87_bulk_v2_gdn_c64_state_trace_valid(arguments, state_trace)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  Sm87BulkV2GdnC64Resources resources{};
  const int query_status =
      query_sm87_bulk_dataflow_v2_gdn_c64_resources_cuda(&resources);
  if (query_status != static_cast<int>(cudaSuccess)) {
    return query_status;
  }
  if (state_trace == nullptr) {
    if (!sm87_bulk_v2_gdn_c64_resources_valid(resources)) {
      return static_cast<int>(cudaErrorLaunchOutOfResources);
    }
  } else {
    // The 96-MiB oracle has a separate recurrence specialization and never
    // borrows recur<false>'s resource record.  Producer and epilogue retain
    // their hard gates; the trace kernel is inspected independently but has
    // correctness-only resource authority.
    Sm87BulkV2GdnKernelResources trace_resources{};
    status = inspect_kernel(
        recur_exact_c64_kernel<true>,
        static_cast<int>(kSm87BulkV2GdnRecurrenceThreads),
        static_cast<int>(kSm87BulkV2GdnRecurrenceCtas), &trace_resources);
    const bool boundary_kernels_pass =
        sm87_bulk_v2_gdn_kernel_resources_pass(
            resources.producer,
            static_cast<int>(kSm87BulkV2GdnProducerThreads),
            static_cast<int>(kSm87BulkV2GdnProducerCtas)) &&
        sm87_bulk_v2_gdn_kernel_resources_pass(
            resources.epilogue,
            static_cast<int>(kSm87BulkV2GdnEpilogueThreads),
            static_cast<int>(kSm87BulkV2GdnEpilogueCtas));
    if (status != cudaSuccess || !boundary_kernels_pass ||
        trace_resources.threads_per_block !=
            static_cast<int>(kSm87BulkV2GdnRecurrenceThreads) ||
        trace_resources.physical_grid_ctas !=
            static_cast<int>(kSm87BulkV2GdnRecurrenceCtas)) {
      return static_cast<int>(status != cudaSuccess
                                  ? status
                                  : cudaErrorLaunchOutOfResources);
    }
  }

  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  (void)cudaGetLastError();
  const dim3 producer_grid(
      static_cast<unsigned int>(kSm87BulkV2GdnC64Tokens /
                                kSm87BulkV2GdnProducerRowsPerCta),
      static_cast<unsigned int>(kSm87BulkV2GdnProducerRoles));
  prepare_exact_c64_kernel<<<producer_grid, kSm87BulkV2GdnProducerThreads, 0U,
                             stream>>>(
      arguments.raw_qkvz, arguments.interleaved_ab, arguments.conv_weight,
      arguments.initial_conv_history, arguments.a_log, arguments.dt_bias,
      arguments.l2_epsilon_fp32_bits, arguments.normalized_q,
      arguments.normalized_k, arguments.prepared_v, arguments.alpha,
      arguments.beta, arguments.final_conv_history);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  if (state_trace == nullptr) {
    recur_exact_c64_kernel<false>
        <<<kSm87BulkV2GdnRecurrenceCtas, kSm87BulkV2GdnRecurrenceThreads, 0U,
           stream>>>(
            arguments.normalized_q, arguments.normalized_k,
            arguments.prepared_v, arguments.alpha, arguments.beta,
            arguments.initial_recurrent_state, arguments.raw_output,
            arguments.final_recurrent_state, nullptr);
  } else {
    recur_exact_c64_kernel<true>
        <<<kSm87BulkV2GdnRecurrenceCtas, kSm87BulkV2GdnRecurrenceThreads, 0U,
           stream>>>(
            arguments.normalized_q, arguments.normalized_k,
            arguments.prepared_v, arguments.alpha, arguments.beta,
            arguments.initial_recurrent_state, arguments.raw_output,
            arguments.final_recurrent_state, state_trace);
  }
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  rms_norm_silu_rows8_exact_kernel
      <<<kSm87BulkV2GdnEpilogueCtas, kSm87BulkV2GdnEpilogueThreads, 0U,
         stream>>>(arguments.raw_output, arguments.raw_qkvz,
                   arguments.norm_weight, arguments.norm_epsilon_fp32_bits,
                   arguments.output);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_bulk_dataflow_v2_gdn_c64_resources_cuda(
    Sm87BulkV2GdnC64Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  int device = -1;
  cudaError_t status = validate_fixed_device(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  (void)device;
  cudaFuncAttributes producer_attributes{};
  status = cudaFuncGetAttributes(&producer_attributes,
                                 prepare_exact_c64_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->binary_version = producer_attributes.binaryVersion;
  status = inspect_kernel(
      prepare_exact_c64_kernel,
      static_cast<int>(kSm87BulkV2GdnProducerThreads),
      static_cast<int>(kSm87BulkV2GdnProducerCtas), &resources->producer);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = inspect_kernel(
      recur_exact_c64_kernel<false>,
      static_cast<int>(kSm87BulkV2GdnRecurrenceThreads),
      static_cast<int>(kSm87BulkV2GdnRecurrenceCtas), &resources->recurrence);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = inspect_kernel(
      rms_norm_silu_rows8_exact_kernel,
      static_cast<int>(kSm87BulkV2GdnEpilogueThreads),
      static_cast<int>(kSm87BulkV2GdnEpilogueCtas), &resources->epilogue);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->kernels_compiled = true;
  resources->exact_geometry = true;
  resources->resource_gate_passed =
      sm87_bulk_v2_gdn_kernel_resources_pass(
          resources->producer,
          static_cast<int>(kSm87BulkV2GdnProducerThreads),
          static_cast<int>(kSm87BulkV2GdnProducerCtas)) &&
      sm87_bulk_v2_gdn_kernel_resources_pass(
          resources->recurrence,
          static_cast<int>(kSm87BulkV2GdnRecurrenceThreads),
          static_cast<int>(kSm87BulkV2GdnRecurrenceCtas)) &&
      sm87_bulk_v2_gdn_kernel_resources_pass(
          resources->epilogue,
          static_cast<int>(kSm87BulkV2GdnEpilogueThreads),
          static_cast<int>(kSm87BulkV2GdnEpilogueCtas));
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_bulk_dataflow_v2_gdn_c64_cuda(
    const Sm87BulkV2GdnC64Arguments& arguments) noexcept {
  return launch_impl(arguments, nullptr);
}

int launch_sm87_bulk_dataflow_v2_gdn_c64_state_trace_cuda(
    const Sm87BulkV2GdnC64Arguments& arguments,
    std::uint16_t* const state_after_token) noexcept {
  if (state_after_token == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return launch_impl(arguments, state_after_token);
}

}  // namespace q3x::kernels
