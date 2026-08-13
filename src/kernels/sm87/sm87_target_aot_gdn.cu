#include "q3x/kernels/sm87_target_aot_gdn_cuda.h"

#include "sm87_target_aot_gdn_launch_internal.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87TargetAotGdnThreadsPerCta);
constexpr unsigned int kOwnerCtas =
    static_cast<unsigned int>(kSm87TargetAotGdnOwnerCtas);
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kWarps =
    static_cast<unsigned int>(kSm87TargetAotGdnWarpsPerCta);
constexpr unsigned int kRowsPerWarpBatch = 8U;
constexpr unsigned int kBatchRowOffset = 64U;
constexpr unsigned int kScratchRowStride = 129U;
constexpr unsigned int kFullWarpMask = 0xffff'ffffU;
constexpr unsigned int kTokensPerBlock =
    static_cast<unsigned int>(kSm87TargetAotGdnExactRecurrenceTokens);
constexpr unsigned int kC16Blocks =
    static_cast<unsigned int>(kSm87TargetAotGdnCudaTokenCount /
                              kSm87TargetAotGdnExactRecurrenceTokens);
constexpr unsigned int kChannelsPerOwner =
    static_cast<unsigned int>(kSm87TargetAotGdnConvChannelsPerOwner);
constexpr unsigned int kValueChannelsPerOwner =
    static_cast<unsigned int>(kSm87TargetAotGdnValueHeadsPerQkGroup *
                              kSm87TargetAotGdnStateValueDimension);
constexpr unsigned int kChannelItemsPerThread =
    (kChannelsPerOwner + kThreads - 1U) / kThreads;
constexpr unsigned int kKeyPairsPerLane =
    static_cast<unsigned int>(kSm87TargetAotGdnStateKeyDimension /
                              kWarpSize / 2U);

static_assert(kThreads == 256U && kOwnerCtas == 16U && kWarps == 8U);
static_assert(kTokensPerBlock == 16U && kC16Blocks == 2'500U);
static_assert(kChannelsPerOwner == 640U && kValueChannelsPerOwner == 384U);
static_assert(kChannelItemsPerThread == 3U && kKeyPairsPerLane == 2U);

struct PayloadSlot final {
  std::uint16_t q[kSm87TargetAotGdnExactRecurrenceTokens *
                  kSm87TargetAotGdnStateKeyDimension];
  std::uint16_t k[kSm87TargetAotGdnExactRecurrenceTokens *
                  kSm87TargetAotGdnStateKeyDimension];
  std::uint16_t v[kSm87TargetAotGdnExactRecurrenceTokens *
                  kSm87TargetAotGdnValueHeadsPerQkGroup *
                  kSm87TargetAotGdnStateValueDimension];
  std::uint16_t z[kSm87TargetAotGdnExactRecurrenceTokens *
                  kSm87TargetAotGdnValueHeadsPerQkGroup *
                  kSm87TargetAotGdnStateValueDimension];
  std::uint16_t a[kSm87TargetAotGdnExactRecurrenceTokens *
                  kSm87TargetAotGdnValueHeadsPerQkGroup];
  std::uint16_t b[kSm87TargetAotGdnExactRecurrenceTokens *
                  kSm87TargetAotGdnValueHeadsPerQkGroup];
};

struct SharedStorage final {
  PayloadSlot payload[kSm87TargetAotGdnPreparationSlots];
  float row_scratch[kWarps * kRowsPerWarpBatch * kScratchRowStride];
  float normalized_q[kSm87TargetAotGdnStateKeyDimension];
  float normalized_k[kSm87TargetAotGdnStateKeyDimension];
  float recurrence_scalars[kSm87TargetAotGdnValueHeadsPerQkGroup * 2U];
  std::uint16_t raw_output[kSm87TargetAotGdnStateValueDimension];
  std::uint32_t cancellation_observed;
};

static_assert(sizeof(PayloadSlot) == kSm87TargetAotGdnPayloadBytesPerSlot);
static_assert(offsetof(SharedStorage, row_scratch) ==
              kSm87TargetAotGdnPrivateSharedPayloadBytes);
static_assert(sizeof(SharedStorage) ==
              kSm87TargetAotGdnCudaDynamicSharedBytes);

struct HeadRegisterState final {
  std::uint32_t lower[kRowsPerWarpBatch][kKeyPairsPerLane];
  std::uint32_t upper[kRowsPerWarpBatch][kKeyPairsPerLane];
};

struct ConvChannelRegisters final {
  float weight[4U];
  std::uint16_t history[3U];
  unsigned int canonical_channel;
  unsigned int local_channel;
  bool valid;
};

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

[[nodiscard]] __device__ __forceinline__ unsigned int canonical_channel(
    const unsigned int owner, const unsigned int local_channel) noexcept {
  if (local_channel < kSm87TargetAotGdnStateKeyDimension) {
    return owner * kSm87TargetAotGdnStateKeyDimension + local_channel;
  }
  if (local_channel < 2U * kSm87TargetAotGdnStateKeyDimension) {
    return static_cast<unsigned int>(kSm87TargetAotGdnRawKOffset) +
           owner * kSm87TargetAotGdnStateKeyDimension +
           local_channel - kSm87TargetAotGdnStateKeyDimension;
  }
  return static_cast<unsigned int>(kSm87TargetAotGdnRawVOffset) +
         owner * kValueChannelsPerOwner +
         local_channel - 2U * kSm87TargetAotGdnStateKeyDimension;
}

__device__ __forceinline__ void initialize_head_state(
    HeadRegisterState& state) noexcept {
#pragma unroll
  for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
#pragma unroll
    for (unsigned int pair = 0U; pair < kKeyPairsPerLane; ++pair) {
      state.lower[row][pair] = 0U;
      state.upper[row][pair] = 0U;
    }
  }
}

__device__ __forceinline__ void initialize_conv_channels(
    ConvChannelRegisters (&channels)[kChannelItemsPerThread],
    const unsigned int owner,
    const std::uint16_t* const conv_weight) noexcept {
#pragma unroll
  for (unsigned int item = 0U; item < kChannelItemsPerThread; ++item) {
    auto& channel = channels[item];
    channel.local_channel = threadIdx.x + item * kThreads;
    channel.valid = channel.local_channel < kChannelsPerOwner;
    channel.canonical_channel =
        channel.valid
            ? canonical_channel(owner, channel.local_channel)
            : 0U;
#pragma unroll
    for (unsigned int history = 0U; history < 3U; ++history) {
      channel.history[history] = 0U;
    }
#pragma unroll
    for (unsigned int tap = 0U; tap < 4U; ++tap) {
      channel.weight[tap] =
          channel.valid
              ? decode_bf16(conv_weight[
                    static_cast<std::size_t>(channel.canonical_channel) * 4U +
                    tap])
              : 0.0F;
    }
  }
}

__device__ __forceinline__ void prepare_c16_payload(
    PayloadSlot* const slot,
    ConvChannelRegisters (&channels)[kChannelItemsPerThread],
    const unsigned int owner, const unsigned int first_token,
    const std::uint16_t* const raw_qkvz,
    const std::uint16_t* const interleaved_ab) noexcept {
  // Q/K/V convolution: every thread owns two or three complete channel
  // histories and advances them from oldest history to the current token.
#pragma unroll
  for (unsigned int item = 0U; item < kChannelItemsPerThread; ++item) {
    auto& channel = channels[item];
    if (channel.valid) {
#pragma unroll
      for (unsigned int token = 0U; token < kTokensPerBlock; ++token) {
        const std::size_t source =
            static_cast<std::size_t>(first_token + token) *
                kSm87TargetAotGdnRawQkvZChannels +
            channel.canonical_channel;
        const std::uint16_t current = raw_qkvz[source];
        float convolution = 0.0F;
        convolution = fmaf(decode_bf16(channel.history[0U]),
                           channel.weight[0U], convolution);
        convolution = fmaf(decode_bf16(channel.history[1U]),
                           channel.weight[1U], convolution);
        convolution = fmaf(decode_bf16(channel.history[2U]),
                           channel.weight[2U], convolution);
        convolution =
            fmaf(decode_bf16(current), channel.weight[3U], convolution);
        const std::uint16_t published = encode_bf16_rne(
            convolution / (1.0F + expf(-convolution)));
        if (channel.local_channel <
            kSm87TargetAotGdnStateKeyDimension) {
          slot->q[token * kSm87TargetAotGdnStateKeyDimension +
                  channel.local_channel] = published;
        } else if (channel.local_channel <
                   2U * kSm87TargetAotGdnStateKeyDimension) {
          slot->k[token * kSm87TargetAotGdnStateKeyDimension +
                  channel.local_channel -
                      kSm87TargetAotGdnStateKeyDimension] = published;
        } else {
          slot->v[token * kValueChannelsPerOwner +
                  channel.local_channel -
                      2U * kSm87TargetAotGdnStateKeyDimension] = published;
        }
        channel.history[0U] = channel.history[1U];
        channel.history[1U] = channel.history[2U];
        channel.history[2U] = current;
      }
    }
  }

  // Z bypasses convolution and retains the raw projection bits.
  constexpr unsigned int kZValues = kTokensPerBlock * kValueChannelsPerOwner;
  for (unsigned int element = threadIdx.x; element < kZValues;
       element += kThreads) {
    const unsigned int token = element / kValueChannelsPerOwner;
    const unsigned int channel = element % kValueChannelsPerOwner;
    const std::size_t source =
        static_cast<std::size_t>(first_token + token) *
            kSm87TargetAotGdnRawQkvZChannels +
        kSm87TargetAotGdnRawZOffset + owner * kValueChannelsPerOwner +
        channel;
    slot->z[element] = raw_qkvz[source];
  }

  // The producer owns one [T,96] span whose row is [A48,B48].
  constexpr unsigned int kAbValues =
      kTokensPerBlock * kSm87TargetAotGdnValueHeadsPerQkGroup;
  if (threadIdx.x < kAbValues) {
    const unsigned int token =
        threadIdx.x / kSm87TargetAotGdnValueHeadsPerQkGroup;
    const unsigned int head =
        threadIdx.x % kSm87TargetAotGdnValueHeadsPerQkGroup;
    const std::size_t row =
        static_cast<std::size_t>(first_token + token) *
            kSm87TargetAotGdnAbChannels;
    const std::size_t head_index =
        owner * kSm87TargetAotGdnValueHeadsPerQkGroup + head;
    slot->a[threadIdx.x] = interleaved_ab[row + head_index];
    slot->b[threadIdx.x] = interleaved_ab[
        row + kSm87TargetAotGdnValueHeads + head_index];
  }
}

__device__ __forceinline__ void prepare_normalized_qk(
    SharedStorage* const storage, const PayloadSlot* const slot,
    const unsigned int token, const float l2_epsilon) noexcept {
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  if (thread < kSm87TargetAotGdnStateKeyDimension) {
    storage->normalized_q[thread] = decode_bf16(
        slot->q[token * kSm87TargetAotGdnStateKeyDimension + thread]);
    storage->normalized_k[thread] = decode_bf16(
        slot->k[token * kSm87TargetAotGdnStateKeyDimension + thread]);
  }
  __syncthreads();

  if (warp == 0U || warp == 1U) {
    float* const values = warp == 0U ? storage->normalized_q
                                     : storage->normalized_k;
    const float first = values[lane];
    const float second = values[lane + 32U];
    const float third = values[lane + 64U];
    const float fourth = values[lane + 96U];
    const float first_square = first * first;
    const float second_square = second * second;
    const float third_square = third * third;
    const float fourth_square = fourth * fourth;
    const float first_pair = first_square + third_square;
    const float second_pair = second_square + fourth_square;
    float warp_sum = first_pair + second_pair;
#pragma unroll
    for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
      warp_sum += __shfl_down_sync(kFullWarpMask, warp_sum, stride);
    }
    float scale = 0.0F;
    if (lane == 0U) {
      scale = rsqrtf(warp_sum + l2_epsilon);
      if (warp == 0U) {
        scale *= rsqrtf(
            static_cast<float>(kSm87TargetAotGdnStateKeyDimension));
      }
    }
    scale = __shfl_sync(kFullWarpMask, scale, 0U);
    values[lane] = first * scale;
    values[lane + 32U] = second * scale;
    values[lane + 64U] = third * scale;
    values[lane + 96U] = fourth * scale;
  }
  __syncthreads();
}

__device__ __forceinline__ void prepare_recurrence_scalars(
    SharedStorage* const storage, const PayloadSlot* const slot,
    const unsigned int owner, const unsigned int token,
    const std::uint16_t* const a_log,
    const std::uint16_t* const dt_bias) noexcept {
  if (threadIdx.x < kSm87TargetAotGdnValueHeadsPerQkGroup) {
    const unsigned int ordinal = threadIdx.x;
    const unsigned int value_head =
        owner * kSm87TargetAotGdnValueHeadsPerQkGroup + ordinal;
    const unsigned int scalar =
        token * kSm87TargetAotGdnValueHeadsPerQkGroup + ordinal;
    const float gate_input =
        decode_bf16(slot->a[scalar]) + decode_bf16(dt_bias[value_head]);
    const float g = -expf(decode_bf16(a_log[value_head])) *
                    stable_softplus(gate_input);
    storage->recurrence_scalars[ordinal * 2U] = expf(g);
    storage->recurrence_scalars[ordinal * 2U + 1U] =
        stable_sigmoid(decode_bf16(slot->b[scalar]));
  }
  __syncthreads();
}

template <unsigned int kHeadOrdinal>
__device__ __forceinline__ void update_head_and_publish(
    HeadRegisterState& state, SharedStorage* const storage,
    const PayloadSlot* const slot, const unsigned int owner,
    const unsigned int local_token, const unsigned int global_token,
    const float norm_epsilon, const std::uint16_t* const norm_weight,
    std::uint16_t* const output) noexcept {
  static_assert(kHeadOrdinal < kSm87TargetAotGdnValueHeadsPerQkGroup);
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const unsigned int warp_first_row = warp * kRowsPerWarpBatch;
  const float alpha = storage->recurrence_scalars[kHeadOrdinal * 2U];
  const float beta = storage->recurrence_scalars[kHeadOrdinal * 2U + 1U];
  float* const warp_scratch =
      storage->row_scratch +
      static_cast<std::size_t>(warp * kRowsPerWarpBatch) *
          kScratchRowStride;

#pragma unroll 1
  for (unsigned int batch = 0U; batch < 2U; ++batch) {
    const unsigned int first_row = warp_first_row + batch * kBatchRowOffset;
#pragma unroll
    for (unsigned int pair = 0U; pair < kKeyPairsPerLane; ++pair) {
      const unsigned int first_key = lane + pair * 2U * kWarpSize;
      const unsigned int second_key = first_key + kWarpSize;
#pragma unroll
      for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
        const std::uint32_t word =
            batch == 0U ? state.lower[row][pair]
                        : state.upper[row][pair];
        const std::size_t scratch =
            static_cast<std::size_t>(row) * kScratchRowStride;
        warp_scratch[scratch + first_key] =
            alpha * decode_bf16(static_cast<std::uint16_t>(word));
        warp_scratch[scratch + second_key] =
            alpha * decode_bf16(static_cast<std::uint16_t>(word >> 16U));
      }
    }
    __syncwarp(kFullWarpMask);

    float lane_prediction = 0.0F;
    if (lane < kRowsPerWarpBatch) {
      const float* const row =
          warp_scratch + static_cast<std::size_t>(lane) * kScratchRowStride;
#pragma unroll
      for (unsigned int key = 0U;
           key < kSm87TargetAotGdnStateKeyDimension; ++key) {
        lane_prediction = fmaf(row[key], storage->normalized_k[key],
                               lane_prediction);
      }
    }

    float lane_delta = 0.0F;
    if (lane < kRowsPerWarpBatch) {
      const unsigned int value_dimension = first_row + lane;
      const std::size_t value =
          static_cast<std::size_t>(local_token) * kValueChannelsPerOwner +
          kHeadOrdinal * kSm87TargetAotGdnStateValueDimension +
          value_dimension;
      lane_delta =
          (decode_bf16(slot->v[value]) - lane_prediction) * beta;
    }
    float deltas[kRowsPerWarpBatch];
#pragma unroll
    for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
      deltas[row] = __shfl_sync(kFullWarpMask, lane_delta, row);
    }

#pragma unroll
    for (unsigned int pair = 0U; pair < kKeyPairsPerLane; ++pair) {
      const unsigned int first_key = lane + pair * 2U * kWarpSize;
      const unsigned int second_key = first_key + kWarpSize;
#pragma unroll
      for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
        const std::size_t scratch =
            static_cast<std::size_t>(row) * kScratchRowStride;
        const float first_updated =
            fmaf(deltas[row], storage->normalized_k[first_key],
                 warp_scratch[scratch + first_key]);
        const float second_updated =
            fmaf(deltas[row], storage->normalized_k[second_key],
                 warp_scratch[scratch + second_key]);
        const std::uint32_t rounded =
            encode_bf16_pair_rne(first_updated, second_updated);
        if (batch == 0U) {
          state.lower[row][pair] = rounded;
        } else {
          state.upper[row][pair] = rounded;
        }
        // Same-token output consumes the unrounded updates.
        warp_scratch[scratch + first_key] = first_updated;
        warp_scratch[scratch + second_key] = second_updated;
      }
    }
    __syncwarp(kFullWarpMask);

    if (lane < kRowsPerWarpBatch) {
      const float* const row =
          warp_scratch + static_cast<std::size_t>(lane) * kScratchRowStride;
      float result = 0.0F;
#pragma unroll
      for (unsigned int key = 0U;
           key < kSm87TargetAotGdnStateKeyDimension; ++key) {
        result = fmaf(row[key], storage->normalized_q[key], result);
      }
      storage->raw_output[first_row + lane] = encode_bf16_rne(result);
    }
    __syncwarp(kFullWarpMask);
  }
  __syncthreads();

  // Preserve the exact standalone epilogue tree: each lane owns dimensions
  // [lane,lane+32,lane+64,lane+96], forms (0+64)/(32+96), then shuffle-adds
  // 16,8,4,2,1.  Only warp zero participates because one token/head row is
  // published at a time.
  if (warp == 0U) {
    float raw_values[4U];
    float squares[4U];
#pragma unroll
    for (unsigned int item = 0U; item < 4U; ++item) {
      const unsigned int dimension = lane + item * kWarpSize;
      raw_values[item] = decode_bf16(storage->raw_output[dimension]);
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
              norm_epsilon);
    }
    inverse_rms = __shfl_sync(kFullWarpMask, inverse_rms, 0U);
#pragma unroll
    for (unsigned int item = 0U; item < 4U; ++item) {
      const unsigned int dimension = lane + item * kWarpSize;
      float value = raw_values[item] * inverse_rms;
      value *= decode_bf16(norm_weight[dimension]);
      const std::size_t z_index =
          static_cast<std::size_t>(local_token) * kValueChannelsPerOwner +
          kHeadOrdinal * kSm87TargetAotGdnStateValueDimension + dimension;
      const float z = decode_bf16(slot->z[z_index]);
      value *= z / (1.0F + expf(-z));
      const std::size_t destination =
          static_cast<std::size_t>(global_token) *
              kSm87TargetAotGdnOutputChannels +
          (owner * kSm87TargetAotGdnValueHeadsPerQkGroup + kHeadOrdinal) *
              kSm87TargetAotGdnStateValueDimension +
          dimension;
      output[destination] = encode_bf16_rne(value);
    }
  }
  __syncthreads();
}

template <unsigned int kHeadOrdinal>
__device__ __forceinline__ void store_head_state(
    const HeadRegisterState& state, const unsigned int owner,
    std::uint16_t* const final_recurrent_state) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_first_row = warp * kRowsPerWarpBatch;
  const unsigned int value_head =
      owner * kSm87TargetAotGdnValueHeadsPerQkGroup + kHeadOrdinal;
  const std::size_t head_offset =
      static_cast<std::size_t>(value_head) *
      kSm87TargetAotGdnStateValuesPerHead;
#pragma unroll
  for (unsigned int row = 0U; row < kRowsPerWarpBatch; ++row) {
#pragma unroll
    for (unsigned int pair = 0U; pair < kKeyPairsPerLane; ++pair) {
      const unsigned int first_key = lane + pair * 2U * kWarpSize;
      const unsigned int second_key = first_key + kWarpSize;
      const unsigned int lower_row = warp_first_row + row;
      const unsigned int upper_row = lower_row + kBatchRowOffset;
      const std::uint32_t lower = state.lower[row][pair];
      const std::uint32_t upper = state.upper[row][pair];
      const std::size_t lower_offset =
          head_offset +
          static_cast<std::size_t>(lower_row) *
              kSm87TargetAotGdnStateKeyDimension;
      const std::size_t upper_offset =
          head_offset +
          static_cast<std::size_t>(upper_row) *
              kSm87TargetAotGdnStateKeyDimension;
      final_recurrent_state[lower_offset + first_key] =
          static_cast<std::uint16_t>(lower);
      final_recurrent_state[lower_offset + second_key] =
          static_cast<std::uint16_t>(lower >> 16U);
      final_recurrent_state[upper_offset + first_key] =
          static_cast<std::uint16_t>(upper);
      final_recurrent_state[upper_offset + second_key] =
          static_cast<std::uint16_t>(upper >> 16U);
    }
  }
}

__device__ __forceinline__ void store_conv_history(
    const ConvChannelRegisters (&channels)[kChannelItemsPerThread],
    std::uint16_t* const final_conv_history) noexcept {
#pragma unroll
  for (unsigned int item = 0U; item < kChannelItemsPerThread; ++item) {
    const auto& channel = channels[item];
    if (channel.valid) {
      const std::size_t destination =
          static_cast<std::size_t>(channel.canonical_channel) *
          kSm87TargetAotGdnConvHistory;
      final_conv_history[destination] = channel.history[0U];
      final_conv_history[destination + 1U] = channel.history[1U];
      final_conv_history[destination + 2U] = channel.history[2U];
    }
  }
}

// One ordinary, non-cooperative kernel.  blockIdx.x is the sole physical
// owner of QK group x, its three value heads, 96 KiB of final recurrent state,
// and 640 width-four convolution histories for the complete prompt.
__global__ __launch_bounds__(kThreads, 1)
void sm87_target_aot_gdn_p40000_exact_kernel(
    const std::uint16_t* const raw_qkvz,
    const std::uint16_t* const interleaved_ab,
    const std::uint16_t* const conv_weight,
    const std::uint16_t* const a_log,
    const std::uint16_t* const dt_bias,
    const std::uint32_t l2_epsilon_bits,
    const std::uint16_t* const norm_weight,
    const std::uint32_t norm_epsilon_bits,
    const std::uint32_t* const cancellation_signal,
    std::uint16_t* const output,
    std::uint16_t* const final_conv_history,
    std::uint16_t* const final_recurrent_state) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto* const storage = reinterpret_cast<SharedStorage*>(shared_bytes);
  const unsigned int owner = blockIdx.x;
  const float l2_epsilon = __uint_as_float(l2_epsilon_bits);
  const float norm_epsilon = __uint_as_float(norm_epsilon_bits);

  HeadRegisterState state0{};
  HeadRegisterState state1{};
  HeadRegisterState state2{};
  initialize_head_state(state0);
  initialize_head_state(state1);
  initialize_head_state(state2);

  ConvChannelRegisters channels[kChannelItemsPerThread];
  initialize_conv_channels(channels, owner, conv_weight);

#pragma unroll 1
  for (unsigned int c16 = 0U; c16 < kC16Blocks; ++c16) {
    // Same-CTA consumer completion is the only reuse barrier.  No cross-CTA
    // state, barrier, or cooperative-grid primitive exists.
    __syncthreads();
    if (threadIdx.x == 0U) {
      storage->cancellation_observed =
          *reinterpret_cast<const volatile std::uint32_t*>(
              cancellation_signal);
    }
    __syncthreads();
    if (storage->cancellation_observed != 0U) {
      return;
    }
    PayloadSlot* const slot =
        &storage->payload[c16 % kSm87TargetAotGdnPreparationSlots];
    const unsigned int first_token = c16 * kTokensPerBlock;
    prepare_c16_payload(slot, channels, owner, first_token, raw_qkvz,
                        interleaved_ab);
    __syncthreads();

#pragma unroll 1
    for (unsigned int token = 0U; token < kTokensPerBlock; ++token) {
      prepare_normalized_qk(storage, slot, token, l2_epsilon);
      prepare_recurrence_scalars(storage, slot, owner, token, a_log, dt_bias);
      const unsigned int global_token = first_token + token;
      update_head_and_publish<0U>(state0, storage, slot, owner, token,
                                  global_token, norm_epsilon, norm_weight,
                                  output);
      update_head_and_publish<1U>(state1, storage, slot, owner, token,
                                  global_token, norm_epsilon, norm_weight,
                                  output);
      update_head_and_publish<2U>(state2, storage, slot, owner, token,
                                  global_token, norm_epsilon, norm_weight,
                                  output);
    }
    if (threadIdx.x == 0U) {
      storage->cancellation_observed =
          *reinterpret_cast<const volatile std::uint32_t*>(
              cancellation_signal);
    }
    __syncthreads();
    if (storage->cancellation_observed != 0U) {
      return;
    }
  }

  store_head_state<0U>(state0, owner, final_recurrent_state);
  store_head_state<1U>(state1, owner, final_recurrent_state);
  store_head_state<2U>(state2, owner, final_recurrent_state);
  store_conv_history(channels, final_conv_history);
}

[[nodiscard]] cudaError_t validate_fixed_device() noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return status;
  }
  return properties.major == 8 && properties.minor == 7 &&
                 properties.multiProcessorCount ==
                     static_cast<int>(kSm87TargetAotGdnOwnerCtas) &&
                 properties.sharedMemPerBlockOptin >=
                     kSm87TargetAotGdnCudaDynamicSharedBytes
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

[[nodiscard]] bool cancellation_pointer_accessible(
    const void* const pointer, const int current_device) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess || attributes.devicePointer == nullptr) {
    return false;
  }
  if (attributes.type == cudaMemoryTypeDevice) {
    return attributes.device == current_device;
  }
  if (attributes.type == cudaMemoryTypeHost) {
    // cudaHostAllocMapped/cudaHostRegisterMapped exposes devicePointer only
    // when the host control allocation is mapped into this CUDA context.
    return true;
  }
  if (attributes.type == cudaMemoryTypeManaged) {
    return attributes.device == current_device ||
           attributes.device == cudaCpuDeviceId;
  }
  return false;
}

[[nodiscard]] cudaError_t set_dynamic_shared_attribute() noexcept {
  return cudaFuncSetAttribute(
      sm87_target_aot_gdn_p40000_exact_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87TargetAotGdnCudaDynamicSharedBytes));
}

[[nodiscard, maybe_unused]] int launch_authenticated_body(
    const Sm87TargetAotGdnCudaArguments& arguments) noexcept {
  if (!sm87_target_aot_gdn_cuda_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t device_status = validate_fixed_device();
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  int device = -1;
  const cudaError_t current_status = cudaGetDevice(&device);
  if (current_status != cudaSuccess) {
    return static_cast<int>(current_status);
  }
  const void* const pointers[] = {
      arguments.raw_qkvz, arguments.interleaved_ab, arguments.conv_weight,
      arguments.a_log, arguments.dt_bias, arguments.norm_weight,
      arguments.output, arguments.final_conv_history,
      arguments.final_recurrent_state};
  for (const void* const pointer : pointers) {
    if (!exact_device_pointer(pointer, device)) {
      return static_cast<int>(cudaErrorInvalidDevicePointer);
    }
  }
  if (!cancellation_pointer_accessible(arguments.cancellation_signal,
                                       device)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  const cudaError_t attribute_status = set_dynamic_shared_attribute();
  if (attribute_status != cudaSuccess) {
    return static_cast<int>(attribute_status);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  (void)cudaGetLastError();
  sm87_target_aot_gdn_p40000_exact_kernel<<<
      kOwnerCtas, kThreads, kSm87TargetAotGdnCudaDynamicSharedBytes, stream>>>(
      arguments.raw_qkvz, arguments.interleaved_ab, arguments.conv_weight,
      arguments.a_log, arguments.dt_bias, arguments.l2_epsilon_fp32_bits,
      arguments.norm_weight, arguments.norm_epsilon_fp32_bits,
      arguments.cancellation_signal, arguments.output,
      arguments.final_conv_history,
      arguments.final_recurrent_state);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_target_aot_gdn_cuda_resources(
    const std::size_t token_count,
    Sm87TargetAotGdnCudaResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  if (token_count != kSm87TargetAotGdnCudaTokenCount ||
      !sm87_target_aot_gdn_plan(token_count).valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  resources->token_count = token_count;
  const cudaError_t device_status = validate_fixed_device();
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  cudaError_t status = set_dynamic_shared_attribute();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes,
                                 sm87_target_aot_gdn_p40000_exact_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, sm87_target_aot_gdn_p40000_exact_kernel,
      static_cast<int>(kThreads),
      kSm87TargetAotGdnCudaDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = kSm87TargetAotGdnCudaDynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->physical_grid_ctas = static_cast<int>(kOwnerCtas);
  resources->kernel_compiled = true;
  resources->exact_owner_geometry = true;
  resources->static_resources_qualified = false;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_target_aot_gdn_cuda(
    const Sm87TargetAotGdnCudaArguments& arguments) noexcept {
  if (!sm87_target_aot_gdn_cuda_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaErrorNotSupported);
}

namespace sm87_target_aot_gdn_execution_detail {

int launch_authenticated(
    const Sm87TargetAotGdnCudaArguments& arguments) noexcept {
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_REQUEST_STATE_V1_ADMISSION)
  return launch_authenticated_body(arguments);
#else
  (void)arguments;
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

}  // namespace sm87_target_aot_gdn_execution_detail

}  // namespace q3x::kernels
