#include "q3x/kernels/sm87_macrofeed_v3_gdn_p40.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kFullWarpMask = 0xffff'ffffU;
constexpr unsigned int kRecurrenceWarps = 8U;
constexpr unsigned int kRowsPerRecurrenceWarp = 8U;
constexpr unsigned int kUpperRowOffset = 64U;
constexpr unsigned int kScratchRowStride = 129U;
constexpr unsigned int kKeyPairsPerLane = 2U;

static_assert(kRecurrenceWarps * kRowsPerRecurrenceWarp * 2U ==
              kSm87TargetAotGdnStateValueDimension);

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

__device__ __forceinline__ void sample_cancellation(
    const std::uint32_t* const signal,
    std::uint32_t& observed) noexcept {
  if (threadIdx.x == 0U) {
    observed =
        signal == nullptr
            ? 0U
            : *reinterpret_cast<const volatile std::uint32_t*>(signal);
  }
  __syncthreads();
}

// Forty independent channel CTAs keep the complete width-four filter and raw
// history in registers.  The immutable raw tensor and disjoint convolved
// tensor make the P40 loop race-free, while a final cancellation sample keeps
// persistent history publication transactional at this kernel boundary.
__global__ __launch_bounds__(kSm87MacrofeedV3GdnConvThreads)
void causal_conv1d_silu_p40_kernel(
    const std::uint16_t* const raw_qkv,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const conv_history,
    std::uint16_t* const conv_qkv,
    const unsigned int token_count,
    const std::uint32_t* const cancellation_signal) {
  __shared__ std::uint32_t cancellation_observed;
  const unsigned int channel =
      blockIdx.x * static_cast<unsigned int>(kSm87MacrofeedV3GdnConvThreads) +
      threadIdx.x;
  const std::size_t history_offset =
      static_cast<std::size_t>(channel) * kSm87TargetAotGdnConvHistory;
  const std::size_t weight_offset =
      static_cast<std::size_t>(channel) * kSm87TargetAotGdnConvWidth;

  const float weight_0 = decode_bf16(conv_weight[weight_offset]);
  const float weight_1 = decode_bf16(conv_weight[weight_offset + 1U]);
  const float weight_2 = decode_bf16(conv_weight[weight_offset + 2U]);
  const float weight_3 = decode_bf16(conv_weight[weight_offset + 3U]);
  std::uint16_t history_0 = conv_history[history_offset];
  std::uint16_t history_1 = conv_history[history_offset + 1U];
  std::uint16_t history_2 = conv_history[history_offset + 2U];

  std::size_t element = channel;
#pragma unroll 1
  for (unsigned int token = 0U; token < token_count; ++token) {
    if (token % kSm87MacrofeedV3GdnCancellationQuantum == 0U) {
      sample_cancellation(cancellation_signal, cancellation_observed);
      if (cancellation_observed != 0U) {
        return;
      }
    }
    const std::uint16_t current = raw_qkv[element];
    float convolution = 0.0F;
    convolution = fmaf(decode_bf16(history_0), weight_0, convolution);
    convolution = fmaf(decode_bf16(history_1), weight_1, convolution);
    convolution = fmaf(decode_bf16(history_2), weight_2, convolution);
    convolution = fmaf(decode_bf16(current), weight_3, convolution);
    conv_qkv[element] = encode_bf16_rne(
        convolution / (1.0F + expf(-convolution)));
    history_0 = history_1;
    history_1 = history_2;
    history_2 = current;
    element += kSm87TargetAotGdnTotalConvChannels;
  }

  sample_cancellation(cancellation_signal, cancellation_observed);
  if (cancellation_observed == 0U) {
    conv_history[history_offset] = history_0;
    conv_history[history_offset + 1U] = history_1;
    conv_history[history_offset + 2U] = history_2;
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
  std::uint16_t raw_output[kSm87TargetAotGdnStateValueDimension];
  std::uint32_t cancellation_observed;
};

static_assert(sizeof(RecurrenceSharedStorage) == 34'316U);

__device__ __forceinline__ void load_head_state(
    HeadRegisterState& state,
    const unsigned int value_head,
    const std::uint16_t* const source) noexcept {
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
          static_cast<std::uint32_t>(source[lower + first_key]) |
          (static_cast<std::uint32_t>(source[lower + second_key]) << 16U);
      state.upper[row][pair] =
          static_cast<std::uint32_t>(source[upper + first_key]) |
          (static_cast<std::uint32_t>(source[upper + second_key]) << 16U);
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

// One CTA owns one value head for a C5000 epoch.  The exact BF16 state is
// retained in registers, while Q/K and row scratch are CTA-private.  Raw GDN
// results are rounded into shared BF16 before warp 0 reproduces the incumbent
// four-values-per-lane RMS reduction tree and SiLU gate.
__global__ __launch_bounds__(kSm87MacrofeedV3GdnRecurrenceThreads, 3)
void exact_gdn_norm_gate_macrochunk_kernel(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const a_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const z,
    std::uint16_t* const recurrent_state,
    std::uint16_t* const output,
    const unsigned int first_token,
    const unsigned int token_count,
    const std::uint32_t l2_epsilon_bits,
    const std::uint32_t norm_epsilon_bits,
    const std::uint32_t* const cancellation_signal) {
  __shared__ RecurrenceSharedStorage storage;
  const unsigned int value_head = blockIdx.x;
  const unsigned int qk_group =
      value_head / kSm87TargetAotGdnValueHeadsPerQkGroup;
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_first_row = warp * kRowsPerRecurrenceWarp;
  HeadRegisterState state;

  sample_cancellation(cancellation_signal, storage.cancellation_observed);
  if (storage.cancellation_observed != 0U) {
    return;
  }
  load_head_state(state, value_head, recurrent_state);

#pragma unroll 1
  for (unsigned int local_token = 0U; local_token < token_count;
       ++local_token) {
    if (local_token != 0U &&
        local_token % kSm87MacrofeedV3GdnCancellationQuantum == 0U) {
      sample_cancellation(cancellation_signal,
                          storage.cancellation_observed);
      if (storage.cancellation_observed != 0U) {
        return;
      }
    }
    const unsigned int token = first_token + local_token;
    const std::size_t qkv_token =
        static_cast<std::size_t>(token) *
        kSm87TargetAotGdnTotalConvChannels;
    if (threadIdx.x < kSm87TargetAotGdnStateKeyDimension) {
      const std::size_t q =
          qkv_token + static_cast<std::size_t>(qk_group) *
                          kSm87TargetAotGdnStateKeyDimension +
          threadIdx.x;
      const std::size_t k =
          qkv_token + kSm87TargetAotGdnRawKOffset +
          static_cast<std::size_t>(qk_group) *
              kSm87TargetAotGdnStateKeyDimension +
          threadIdx.x;
      storage.normalized_q[threadIdx.x] = decode_bf16(conv_qkv[q]);
      storage.normalized_k[threadIdx.x] = decode_bf16(conv_qkv[k]);
    }
    __syncthreads();

    if (warp == 0U || warp == 1U) {
      float* const values = warp == 0U ? storage.normalized_q
                                      : storage.normalized_k;
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
        scale = rsqrtf(warp_sum + __uint_as_float(l2_epsilon_bits));
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
    if (threadIdx.x == 64U) {
      const std::size_t scalar =
          static_cast<std::size_t>(token) * kSm87TargetAotGdnValueHeads +
          value_head;
      const float gate_input =
          decode_bf16(a[scalar]) + decode_bf16(dt_bias[value_head]);
      const float g = -expf(decode_bf16(a_log[value_head])) *
                      stable_softplus(gate_input);
      storage.scalars[0U] = expf(g);
      storage.scalars[1U] = stable_sigmoid(decode_bf16(b[scalar]));
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
              batch == 0U ? state.lower[row][pair]
                          : state.upper[row][pair];
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
            warp_scratch + static_cast<std::size_t>(lane) *
                               kScratchRowStride;
#pragma unroll
        for (unsigned int key = 0U;
             key < kSm87TargetAotGdnStateKeyDimension; ++key) {
          lane_prediction = fmaf(row[key], storage.normalized_k[key],
                                 lane_prediction);
        }
      }

      float lane_delta = 0.0F;
      if (lane < kRowsPerRecurrenceWarp) {
        const unsigned int value_dimension = first_row + lane;
        const std::size_t value =
            qkv_token + kSm87TargetAotGdnRawVOffset +
            static_cast<std::size_t>(value_head) *
                kSm87TargetAotGdnStateValueDimension +
            value_dimension;
        lane_delta =
            (decode_bf16(conv_qkv[value]) - lane_prediction) * token_beta;
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
          // The same-token output consumes the pre-round FP32 update.
          warp_scratch[scratch + first_key] = first_updated;
          warp_scratch[scratch + second_key] = second_updated;
        }
      }
      __syncwarp(kFullWarpMask);

      if (lane < kRowsPerRecurrenceWarp) {
        const float* const row =
            warp_scratch + static_cast<std::size_t>(lane) *
                               kScratchRowStride;
        float result = 0.0F;
#pragma unroll
        for (unsigned int key = 0U;
             key < kSm87TargetAotGdnStateKeyDimension; ++key) {
          result = fmaf(row[key], storage.normalized_q[key], result);
        }
        storage.raw_output[first_row + lane] = encode_bf16_rne(result);
      }
      __syncwarp(kFullWarpMask);
    }
    __syncthreads();

    if (warp == 0U) {
      float raw_values[4U];
      float squares[4U];
#pragma unroll
      for (unsigned int item = 0U; item < 4U; ++item) {
        const unsigned int dimension = lane + item * kWarpSize;
        raw_values[item] = decode_bf16(storage.raw_output[dimension]);
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
                    static_cast<float>(
                        kSm87TargetAotGdnStateValueDimension) +
                __uint_as_float(norm_epsilon_bits));
      }
      inverse_rms = __shfl_sync(kFullWarpMask, inverse_rms, 0U);
#pragma unroll
      for (unsigned int item = 0U; item < 4U; ++item) {
        const unsigned int dimension = lane + item * kWarpSize;
        float value = raw_values[item] * inverse_rms;
        value *= decode_bf16(norm_weight[dimension]);
        const std::size_t element =
            static_cast<std::size_t>(token) *
                kSm87TargetAotGdnOutputChannels +
            static_cast<std::size_t>(value_head) *
                kSm87TargetAotGdnStateValueDimension +
            dimension;
        const float gate = decode_bf16(z[element]);
        value *= gate / (1.0F + expf(-gate));
        output[element] = encode_bf16_rne(value);
      }
    }
    // Warp 0 must finish consuming the shared raw BF16 row before another
    // recurrence step overwrites it.
    __syncthreads();
  }

  sample_cancellation(cancellation_signal, storage.cancellation_observed);
  if (storage.cancellation_observed == 0U) {
    store_head_state(state, value_head, recurrent_state);
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

template <class Kernel>
[[nodiscard]] cudaError_t inspect_kernel(
    const Kernel kernel,
    const int threads,
    const int grid_ctas,
    Sm87MacrofeedV3GdnKernelResources* const resources) noexcept {
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

}  // namespace

int query_sm87_macrofeed_v3_gdn_p40_resources_cuda(
    Sm87MacrofeedV3GdnResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  int device = -1;
  cudaError_t status = validate_fixed_device(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->binary_version = 87;
  status = inspect_kernel(
      causal_conv1d_silu_p40_kernel,
      static_cast<int>(kSm87MacrofeedV3GdnConvThreads),
      static_cast<int>(kSm87MacrofeedV3GdnConvCtas),
      &resources->convolution);
  if (status == cudaSuccess) {
    status = inspect_kernel(
        exact_gdn_norm_gate_macrochunk_kernel,
        static_cast<int>(kSm87MacrofeedV3GdnRecurrenceThreads),
        static_cast<int>(kSm87MacrofeedV3GdnRecurrenceCtas),
        &resources->recurrence_epilogue);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->kernels_compiled = true;
  resources->exact_geometry = true;
  resources->resource_gate_passed =
      resources->convolution.local_bytes == 0U &&
      resources->convolution.active_blocks_per_sm >= 2 &&
      resources->recurrence_epilogue.local_bytes == 0U &&
      resources->recurrence_epilogue.active_blocks_per_sm >= 3 &&
      resources->recurrence_epilogue.static_shared_bytes <=
          properties.sharedMemPerMultiprocessor / 3U;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_macrofeed_v3_gdn_p40_cuda(
    const Sm87MacrofeedV3GdnP40Arguments& arguments) noexcept {
  Sm87MacrofeedV3GdnResources resources{};
  const int query =
      query_sm87_macrofeed_v3_gdn_p40_resources_cuda(&resources);
  if (query != static_cast<int>(cudaSuccess)) {
    return query;
  }
  return launch_sm87_macrofeed_v3_gdn_p40_sealed_cuda(arguments, resources);
}

int launch_sm87_macrofeed_v3_gdn_p40_sealed_cuda(
    const Sm87MacrofeedV3GdnP40Arguments& arguments,
    const Sm87MacrofeedV3GdnResources& sealed_resources) noexcept {
  if (!sm87_macrofeed_v3_gdn_p40_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!sm87_macrofeed_v3_gdn_resources_valid(sealed_resources)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  causal_conv1d_silu_p40_kernel<<<
      kSm87MacrofeedV3GdnConvCtas, kSm87MacrofeedV3GdnConvThreads, 0U,
      stream>>>(arguments.raw_qkv, arguments.conv_weight,
                arguments.conv_history, arguments.conv_qkv,
                static_cast<unsigned int>(kSm87MacrofeedV3GdnP40Tokens),
                arguments.cancellation_signal);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  for (unsigned int chunk = 0U;
       chunk < static_cast<unsigned int>(kSm87MacrofeedV3GdnMacrochunks);
       ++chunk) {
    const unsigned int first_token =
        chunk *
        static_cast<unsigned int>(kSm87MacrofeedV3GdnMacrochunkTokens);
    exact_gdn_norm_gate_macrochunk_kernel<<<
        kSm87MacrofeedV3GdnRecurrenceCtas,
        kSm87MacrofeedV3GdnRecurrenceThreads, 0U, stream>>>(
        arguments.conv_qkv, arguments.a, arguments.b, arguments.a_log,
        arguments.dt_bias, arguments.norm_weight, arguments.z,
        arguments.recurrent_state, arguments.output, first_token,
        static_cast<unsigned int>(kSm87MacrofeedV3GdnMacrochunkTokens),
        arguments.l2_epsilon_fp32_bits, arguments.norm_epsilon_fp32_bits,
        arguments.cancellation_signal);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_macrofeed_v3_gdn_c64_oracle_cuda(
    const Sm87MacrofeedV3GdnC64OracleArguments& arguments) noexcept {
  const void* const pointers[] = {
      arguments.raw_qkv,      arguments.a,
      arguments.b,            arguments.conv_weight,
      arguments.a_log,        arguments.dt_bias,
      arguments.norm_weight,  arguments.z,
      arguments.conv_history, arguments.recurrent_state,
      arguments.conv_qkv,     arguments.output,
      arguments.cuda_stream,
  };
  for (const void* const pointer : pointers) {
    if (pointer == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  if (arguments.l2_epsilon_fp32_bits !=
          kSm87TargetAotGdnEpsilonFp32Bits ||
      arguments.norm_epsilon_fp32_bits !=
          kSm87TargetAotGdnEpsilonFp32Bits) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  Sm87MacrofeedV3GdnResources resources{};
  const int query =
      query_sm87_macrofeed_v3_gdn_p40_resources_cuda(&resources);
  if (query != static_cast<int>(cudaSuccess)) {
    return query;
  }
  if (!sm87_macrofeed_v3_gdn_resources_valid(resources)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }

  constexpr unsigned int kOracleTokens = 64U;
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  (void)cudaGetLastError();
  causal_conv1d_silu_p40_kernel<<<
      kSm87MacrofeedV3GdnConvCtas, kSm87MacrofeedV3GdnConvThreads, 0U,
      stream>>>(arguments.raw_qkv, arguments.conv_weight,
                arguments.conv_history, arguments.conv_qkv, kOracleTokens,
                nullptr);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  exact_gdn_norm_gate_macrochunk_kernel<<<
      kSm87MacrofeedV3GdnRecurrenceCtas,
      kSm87MacrofeedV3GdnRecurrenceThreads, 0U, stream>>>(
      arguments.conv_qkv, arguments.a, arguments.b, arguments.a_log,
      arguments.dt_bias, arguments.norm_weight, arguments.z,
      arguments.recurrent_state, arguments.output, 0U, kOracleTokens,
      arguments.l2_epsilon_fp32_bits, arguments.norm_epsilon_fp32_bits,
      nullptr);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_macrofeed_v3_gdn_c128_two_epoch_oracle_cuda(
    const Sm87MacrofeedV3GdnC64OracleArguments& arguments) noexcept {
  const void* const pointers[] = {
      arguments.raw_qkv,      arguments.a,
      arguments.b,            arguments.conv_weight,
      arguments.a_log,        arguments.dt_bias,
      arguments.norm_weight,  arguments.z,
      arguments.conv_history, arguments.recurrent_state,
      arguments.conv_qkv,     arguments.output,
      arguments.cuda_stream,
  };
  for (const void* const pointer : pointers) {
    if (pointer == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  if (arguments.l2_epsilon_fp32_bits !=
          kSm87TargetAotGdnEpsilonFp32Bits ||
      arguments.norm_epsilon_fp32_bits !=
          kSm87TargetAotGdnEpsilonFp32Bits) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  Sm87MacrofeedV3GdnResources resources{};
  const int query =
      query_sm87_macrofeed_v3_gdn_p40_resources_cuda(&resources);
  if (query != static_cast<int>(cudaSuccess)) {
    return query;
  }
  if (!sm87_macrofeed_v3_gdn_resources_valid(resources)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }

  constexpr unsigned int kOracleTokens = 128U;
  constexpr unsigned int kOracleEpochTokens = 64U;
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  (void)cudaGetLastError();
  causal_conv1d_silu_p40_kernel<<<
      kSm87MacrofeedV3GdnConvCtas, kSm87MacrofeedV3GdnConvThreads, 0U,
      stream>>>(arguments.raw_qkv, arguments.conv_weight,
                arguments.conv_history, arguments.conv_qkv, kOracleTokens,
                nullptr);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  for (unsigned int first_token = 0U; first_token < kOracleTokens;
       first_token += kOracleEpochTokens) {
    exact_gdn_norm_gate_macrochunk_kernel<<<
        kSm87MacrofeedV3GdnRecurrenceCtas,
        kSm87MacrofeedV3GdnRecurrenceThreads, 0U, stream>>>(
        arguments.conv_qkv, arguments.a, arguments.b, arguments.a_log,
        arguments.dt_bias, arguments.norm_weight, arguments.z,
        arguments.recurrent_state, arguments.output, first_token,
        kOracleEpochTokens, arguments.l2_epsilon_fp32_bits,
        arguments.norm_epsilon_fp32_bits, nullptr);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
