#include "q3x/kernels/sm87_macrofeed_v4_gdn_c8000.h"

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
#include "sm87_macrofeed_v4_bound_launch_internal.h"
#endif

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
constexpr unsigned int kScratchRowPitch = 129U;
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

// One CTA thread owns one convolution channel for the complete C8000 panel.
// The row-strided Q/K/V projection is overwritten only after its raw current
// token has been consumed.  History starts in the already-copied candidate
// bank, remains register-resident through the panel, and is published only at
// successful kernel completion.
__global__ __launch_bounds__(kSm87MacroFeedV4GdnConvThreads)
void causal_conv1d_silu_c8000_in_place_kernel(
    std::uint16_t* const scratch,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const candidate_conv_history,
    const unsigned int token_count,
    const std::uint32_t* const cancellation_signal) {
  __shared__ std::uint32_t cancellation_observed;
  const unsigned int channel =
      blockIdx.x * static_cast<unsigned int>(
                       kSm87MacroFeedV4GdnConvThreads) +
      threadIdx.x;
  const std::size_t history_offset =
      static_cast<std::size_t>(channel) * kSm87TargetAotGdnConvHistory;
  const std::size_t weight_offset =
      static_cast<std::size_t>(channel) * kSm87TargetAotGdnConvWidth;

  const float weight_0 = decode_bf16(conv_weight[weight_offset]);
  const float weight_1 = decode_bf16(conv_weight[weight_offset + 1U]);
  const float weight_2 = decode_bf16(conv_weight[weight_offset + 2U]);
  const float weight_3 = decode_bf16(conv_weight[weight_offset + 3U]);
  std::uint16_t history_0 = candidate_conv_history[history_offset];
  std::uint16_t history_1 = candidate_conv_history[history_offset + 1U];
  std::uint16_t history_2 = candidate_conv_history[history_offset + 2U];

  std::size_t element = kSm87MacroFeedV4GdnQkvOffset + channel;
#pragma unroll 1
  for (unsigned int token = 0U; token < token_count; ++token) {
    if (token % kSm87MacroFeedV4GdnCancellationQuantum == 0U) {
      sample_cancellation(cancellation_signal, cancellation_observed);
      if (cancellation_observed != 0U) {
        return;
      }
    }
    const std::uint16_t current = scratch[element];
    float convolution = 0.0F;
    convolution = fmaf(decode_bf16(history_0), weight_0, convolution);
    convolution = fmaf(decode_bf16(history_1), weight_1, convolution);
    convolution = fmaf(decode_bf16(history_2), weight_2, convolution);
    convolution = fmaf(decode_bf16(current), weight_3, convolution);
    scratch[element] = encode_bf16_rne(
        convolution / (1.0F + expf(-convolution)));
    history_0 = history_1;
    history_1 = history_2;
    history_2 = current;
    element += kSm87MacroFeedV4GdnScratchRowStride;
  }

  sample_cancellation(cancellation_signal, cancellation_observed);
  if (cancellation_observed == 0U) {
    candidate_conv_history[history_offset] = history_0;
    candidate_conv_history[history_offset + 1U] = history_1;
    candidate_conv_history[history_offset + 2U] = history_2;
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
                    kScratchRowPitch];
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

// One CTA owns one value head for the complete C8000 continuation.  The
// incoming active epoch is const.  Its BF16 bits are loaded once into the
// candidate register state; every later token consumes the preceding rounded
// candidate bits.  Same-token output still consumes the FP32 update before
// that rounding, and only the final complete candidate is published globally.
__global__ __launch_bounds__(kSm87MacroFeedV4GdnRecurrenceThreads, 3)
void exact_gdn_norm_gate_c8000_in_place_kernel(
    std::uint16_t* const scratch,
    const std::uint16_t* const a_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const active_recurrent_state,
    std::uint16_t* const candidate_recurrent_state,
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
  load_head_state(state, value_head, active_recurrent_state);

#pragma unroll 1
  for (unsigned int token = 0U; token < token_count; ++token) {
    if (token != 0U &&
        token % kSm87MacroFeedV4GdnCancellationQuantum == 0U) {
      sample_cancellation(cancellation_signal,
                          storage.cancellation_observed);
      if (storage.cancellation_observed != 0U) {
        return;
      }
    }
    const std::size_t token_base =
        static_cast<std::size_t>(token) *
        kSm87MacroFeedV4GdnScratchRowStride;
    if (threadIdx.x < kSm87TargetAotGdnStateKeyDimension) {
      const std::size_t q =
          token_base + kSm87MacroFeedV4GdnQkvOffset +
          static_cast<std::size_t>(qk_group) *
              kSm87TargetAotGdnStateKeyDimension +
          threadIdx.x;
      const std::size_t k =
          token_base + kSm87MacroFeedV4GdnQkvOffset +
          kSm87TargetAotGdnRawKOffset +
          static_cast<std::size_t>(qk_group) *
              kSm87TargetAotGdnStateKeyDimension +
          threadIdx.x;
      storage.normalized_q[threadIdx.x] = decode_bf16(scratch[q]);
      storage.normalized_k[threadIdx.x] = decode_bf16(scratch[k]);
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
      const float gate_input =
          decode_bf16(
              scratch[token_base + kSm87MacroFeedV4GdnAOffset +
                      value_head]) +
          decode_bf16(dt_bias[value_head]);
      const float g = -expf(decode_bf16(a_log[value_head])) *
                      stable_softplus(gate_input);
      storage.scalars[0U] = expf(g);
      storage.scalars[1U] = stable_sigmoid(decode_bf16(
          scratch[token_base + kSm87MacroFeedV4GdnBOffset + value_head]));
    }
    __syncthreads();

    const float token_alpha = storage.scalars[0U];
    const float token_beta = storage.scalars[1U];
    float* const warp_scratch =
        storage.row_scratch +
        static_cast<std::size_t>(warp * kRowsPerRecurrenceWarp) *
            kScratchRowPitch;

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
          const std::size_t scratch_offset =
              static_cast<std::size_t>(row) * kScratchRowPitch;
          warp_scratch[scratch_offset + first_key] =
              token_alpha *
              decode_bf16(static_cast<std::uint16_t>(word));
          warp_scratch[scratch_offset + second_key] =
              token_alpha *
              decode_bf16(static_cast<std::uint16_t>(word >> 16U));
        }
      }
      __syncwarp(kFullWarpMask);

      float lane_prediction = 0.0F;
      if (lane < kRowsPerRecurrenceWarp) {
        const float* const row =
            warp_scratch +
            static_cast<std::size_t>(lane) * kScratchRowPitch;
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
            token_base + kSm87MacroFeedV4GdnOutputOffset +
            static_cast<std::size_t>(value_head) *
                kSm87TargetAotGdnStateValueDimension +
            value_dimension;
        lane_delta =
            (decode_bf16(scratch[value]) - lane_prediction) * token_beta;
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
          const std::size_t scratch_offset =
              static_cast<std::size_t>(row) * kScratchRowPitch;
          const float first_updated =
              fmaf(deltas[row], storage.normalized_k[first_key],
                   warp_scratch[scratch_offset + first_key]);
          const float second_updated =
              fmaf(deltas[row], storage.normalized_k[second_key],
                   warp_scratch[scratch_offset + second_key]);
          const std::uint32_t rounded =
              encode_bf16_pair_rne(first_updated, second_updated);
          if (batch == 0U) {
            state.lower[row][pair] = rounded;
          } else {
            state.upper[row][pair] = rounded;
          }
          // Preserve the authoritative incumbent boundary: this token's
          // output consumes the update before BF16 state publication.
          warp_scratch[scratch_offset + first_key] = first_updated;
          warp_scratch[scratch_offset + second_key] = second_updated;
        }
      }
      __syncwarp(kFullWarpMask);

      if (lane < kRowsPerRecurrenceWarp) {
        const float* const row =
            warp_scratch +
            static_cast<std::size_t>(lane) * kScratchRowPitch;
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
            token_base + kSm87MacroFeedV4GdnOutputOffset +
            static_cast<std::size_t>(value_head) *
                kSm87TargetAotGdnStateValueDimension +
            dimension;
        const float gate = decode_bf16(
            scratch[token_base + kSm87MacroFeedV4GdnZOffset +
                    static_cast<std::size_t>(value_head) *
                        kSm87TargetAotGdnStateValueDimension +
                    dimension]);
        value *= gate / (1.0F + expf(-gate));
        scratch[element] = encode_bf16_rne(value);
      }
    }
    // Warp 0 must finish consuming raw_output before the next token reuses it.
    __syncthreads();
  }

  sample_cancellation(cancellation_signal, storage.cancellation_observed);
  if (storage.cancellation_observed == 0U) {
    store_head_state(state, value_head, candidate_recurrent_state);
  }
}

[[nodiscard]] cudaError_t validate_fixed_device(
    int* const device,
    cudaDeviceProp* const properties) noexcept {
  if (device == nullptr || properties == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaGetDevice(device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaGetDeviceProperties(properties, *device);
  if (status != cudaSuccess) {
    return status;
  }
  return properties->major == 8 && properties->minor == 7 &&
                 properties->multiProcessorCount ==
                     static_cast<int>(kSm87MacroFeedV4GdnSmCount)
             ? cudaSuccess
             : cudaErrorNotSupported;
}

template <class Kernel>
[[nodiscard]] cudaError_t inspect_kernel(
    const Kernel kernel,
    const int threads,
    const int grid_ctas,
    Sm87MacroFeedV4GdnKernelResourceSnapshot* const resources) noexcept {
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

[[nodiscard]] bool device_allocation_range_owned(
    const Sm87MacroFeedV4GdnByteRange& range,
    const int device_ordinal) noexcept {
  if (!range.valid || range.begin == 0U || range.end <= range.begin ||
      device_ordinal < 0) {
    return false;
  }
  cudaPointerAttributes attributes{};
  const auto* const pointer = reinterpret_cast<const void*>(range.begin);
  const cudaError_t pointer_status =
      cudaPointerGetAttributes(&attributes, pointer);
  if (pointer_status != cudaSuccess ||
      attributes.type != cudaMemoryTypeDevice ||
      attributes.device != device_ordinal) {
    return false;
  }
  CUdeviceptr allocation_base = 0U;
  std::size_t allocation_bytes = 0U;
  const CUresult range_status = cuMemGetAddressRange(
      &allocation_base, &allocation_bytes,
      static_cast<CUdeviceptr>(range.begin));
  if (range_status != CUDA_SUCCESS || allocation_base == 0U ||
      allocation_bytes == 0U) {
    return false;
  }
  const auto allocation_begin =
      static_cast<std::uintptr_t>(allocation_base);
  if (allocation_begin > std::numeric_limits<std::uintptr_t>::max() -
                             allocation_bytes) {
    return false;
  }
  const auto allocation_end = allocation_begin + allocation_bytes;
  return range.begin >= allocation_begin && range.end <= allocation_end;
}

template <class Arguments>
[[nodiscard]] bool argument_device_allocation_ranges_owned(
    const Arguments& arguments, const int device_ordinal) noexcept {
  const Sm87MacroFeedV4GdnByteRange ranges[] = {
      sm87_macrofeed_v4_gdn_byte_range(
          arguments.scratch,
          arguments.token_count * arguments.scratch_row_stride *
              kSm87TargetAotGdnBf16Bytes),
      sm87_macrofeed_v4_gdn_byte_range(
          arguments.conv_weight, kSm87MacroFeedV4GdnConvWeightBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          arguments.a_log, kSm87MacroFeedV4GdnHeadVectorBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          arguments.dt_bias, kSm87MacroFeedV4GdnHeadVectorBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          arguments.norm_weight, kSm87MacroFeedV4GdnNormWeightBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          arguments.active_conv_history,
          kSm87MacroFeedV4GdnConvHistoryBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          arguments.candidate_conv_history,
          kSm87MacroFeedV4GdnConvHistoryBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          arguments.active_recurrent_state, kSm87MacroFeedV4GdnStateBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          arguments.candidate_recurrent_state,
          kSm87MacroFeedV4GdnStateBytes),
  };
  for (const auto& range : ranges) {
    if (!device_allocation_range_owned(range, device_ordinal)) {
      return false;
    }
  }
  return arguments.cancellation_signal == nullptr ||
         device_allocation_range_owned(
             sm87_macrofeed_v4_gdn_byte_range(
                 arguments.cancellation_signal, sizeof(std::uint32_t)),
             device_ordinal);
}

[[nodiscard]] constexpr bool kernel_resource_snapshot_equal(
    const Sm87MacroFeedV4GdnKernelResourceSnapshot& left,
    const Sm87MacroFeedV4GdnKernelResourceSnapshot& right) noexcept {
  return left.registers_per_thread == right.registers_per_thread &&
         left.static_shared_bytes == right.static_shared_bytes &&
         left.local_bytes == right.local_bytes &&
         left.maximum_threads_per_block == right.maximum_threads_per_block &&
         left.active_blocks_per_sm == right.active_blocks_per_sm &&
         left.threads_per_block == right.threads_per_block &&
         left.physical_grid_ctas == right.physical_grid_ctas;
}

[[nodiscard]] constexpr bool admission_resource_snapshot_equal(
    const Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot& left,
    const Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot& right) noexcept {
  return left.identity == right.identity &&
         left.device_ordinal == right.device_ordinal &&
         left.compute_major == right.compute_major &&
         left.compute_minor == right.compute_minor &&
         left.sm_count == right.sm_count &&
         left.binary_version == right.binary_version &&
         kernel_resource_snapshot_equal(left.convolution, right.convolution) &&
         kernel_resource_snapshot_equal(left.recurrence_epilogue,
                                        right.recurrence_epilogue) &&
         left.kernels_compiled == right.kernels_compiled &&
         left.exact_geometry == right.exact_geometry &&
         left.static_resource_gate_passed ==
             right.static_resource_gate_passed &&
         left.numerical_contract_qualified ==
             right.numerical_contract_qualified &&
         left.production_dispatch_eligible ==
             right.production_dispatch_eligible &&
         left.startup_package_unbound == right.startup_package_unbound &&
         left.execution_capability == right.execution_capability &&
         left.caller_snapshot_grants_production_authority ==
             right.caller_snapshot_grants_production_authority;
}

[[nodiscard]] bool structural_arguments_valid(
    const std::uint16_t* const scratch,
    const std::size_t token_count,
    const std::size_t scratch_row_stride,
    const std::uint16_t* const conv_weight,
    const std::uint16_t* const a_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const active_conv_history,
    const std::uint16_t* const candidate_conv_history,
    const std::uint16_t* const active_recurrent_state,
    const std::uint16_t* const candidate_recurrent_state,
    const std::uint32_t* const cancellation_signal,
    const std::uint32_t l2_epsilon_fp32_bits,
    const std::uint32_t norm_epsilon_fp32_bits,
    const void* const cuda_stream,
    const bool production_extent) noexcept {
  if ((production_extent &&
       token_count != kSm87MacroFeedV4GdnC8000Tokens) ||
      (!production_extent &&
       (token_count == 0U ||
        token_count > kSm87MacroFeedV4GdnOracleMaximumTokens)) ||
      scratch_row_stride != kSm87MacroFeedV4GdnScratchRowStride ||
      l2_epsilon_fp32_bits != kSm87TargetAotGdnEpsilonFp32Bits ||
      norm_epsilon_fp32_bits != kSm87TargetAotGdnEpsilonFp32Bits ||
      cuda_stream == nullptr) {
    return false;
  }
  const void* const aligned_pointers[] = {
      scratch,
      conv_weight,
      a_log,
      dt_bias,
      norm_weight,
      active_conv_history,
      candidate_conv_history,
      active_recurrent_state,
      candidate_recurrent_state,
  };
  for (const void* const pointer : aligned_pointers) {
    if (!sm87_macrofeed_v4_gdn_pointer_aligned(pointer)) {
      return false;
    }
  }
  if (cancellation_signal != nullptr &&
      reinterpret_cast<std::uintptr_t>(cancellation_signal) %
              alignof(std::uint32_t) !=
          0U) {
    return false;
  }

  const Sm87MacroFeedV4GdnByteRange ranges[] = {
      sm87_macrofeed_v4_gdn_byte_range(
          scratch, token_count * scratch_row_stride *
                       kSm87TargetAotGdnBf16Bytes),
      sm87_macrofeed_v4_gdn_byte_range(
          conv_weight, kSm87MacroFeedV4GdnConvWeightBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          a_log, kSm87MacroFeedV4GdnHeadVectorBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          dt_bias, kSm87MacroFeedV4GdnHeadVectorBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          norm_weight, kSm87MacroFeedV4GdnNormWeightBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          active_conv_history, kSm87MacroFeedV4GdnConvHistoryBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          candidate_conv_history, kSm87MacroFeedV4GdnConvHistoryBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          active_recurrent_state, kSm87MacroFeedV4GdnStateBytes),
      sm87_macrofeed_v4_gdn_byte_range(
          candidate_recurrent_state, kSm87MacroFeedV4GdnStateBytes),
  };
  for (std::size_t left = 0U; left < 9U; ++left) {
    if (!ranges[left].valid) {
      return false;
    }
    for (std::size_t right = left + 1U; right < 9U; ++right) {
      if (!sm87_macrofeed_v4_gdn_ranges_disjoint(ranges[left],
                                                 ranges[right])) {
        return false;
      }
    }
  }
  if (cancellation_signal != nullptr) {
    const auto cancellation = sm87_macrofeed_v4_gdn_byte_range(
        cancellation_signal, sizeof(std::uint32_t));
    if (!cancellation.valid) {
      return false;
    }
    for (const auto& range : ranges) {
      if (!sm87_macrofeed_v4_gdn_ranges_disjoint(cancellation, range)) {
        return false;
      }
    }
  }
  return true;
}

struct ConstituentSubmitLedger final {
  std::size_t accepted_kernel_launches = 0U;
  std::size_t asynchronous_d2d_copies = 0U;
  std::size_t conv_history_copy_bytes = 0U;
};

static_assert(kSm87MacroFeedV4GdnPhysicalKernelLaunches == 2U &&
              kSm87MacroFeedV4GdnHistoryCopies == 1U &&
              kSm87MacroFeedV4GdnConvHistoryBytes == 61'440U);

[[nodiscard]] cudaError_t enqueue_constituent(
    std::uint16_t* const scratch,
    const std::uint16_t* const conv_weight,
    const std::uint16_t* const a_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const active_conv_history,
    std::uint16_t* const candidate_conv_history,
    const std::uint16_t* const active_recurrent_state,
    std::uint16_t* const candidate_recurrent_state,
    const std::uint32_t* const cancellation_signal,
    const unsigned int token_count,
    const std::uint32_t l2_epsilon_fp32_bits,
    const std::uint32_t norm_epsilon_fp32_bits,
    ConstituentSubmitLedger* const submit_ledger,
    const cudaStream_t stream) noexcept {
  if (submit_ledger == nullptr) {
    return cudaErrorInvalidValue;
  }
  *submit_ledger = {};
  cudaError_t status = cudaMemcpyAsync(
      candidate_conv_history, active_conv_history,
      kSm87MacroFeedV4GdnConvHistoryBytes, cudaMemcpyDeviceToDevice, stream);
  if (status != cudaSuccess) {
    return status;
  }
  submit_ledger->asynchronous_d2d_copies = 1U;
  submit_ledger->conv_history_copy_bytes =
      kSm87MacroFeedV4GdnConvHistoryBytes;
  causal_conv1d_silu_c8000_in_place_kernel<<<
      kSm87MacroFeedV4GdnConvCtas, kSm87MacroFeedV4GdnConvThreads, 0U,
      stream>>>(scratch, conv_weight, candidate_conv_history, token_count,
                cancellation_signal);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  submit_ledger->accepted_kernel_launches = 1U;
  exact_gdn_norm_gate_c8000_in_place_kernel<<<
      kSm87MacroFeedV4GdnRecurrenceCtas,
      kSm87MacroFeedV4GdnRecurrenceThreads, 0U, stream>>>(
      scratch, a_log, dt_bias, norm_weight, active_recurrent_state,
      candidate_recurrent_state, token_count, l2_epsilon_fp32_bits,
      norm_epsilon_fp32_bits, cancellation_signal);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  submit_ledger->accepted_kernel_launches = 2U;
  return cudaSuccess;
}

}  // namespace

bool sm87_macrofeed_v4_gdn_c8000_arguments_valid(
    const Sm87MacroFeedV4GdnC8000Arguments& arguments) noexcept {
  return sm87_macrofeed_v4_gdn_c8000_plan(
             arguments.token_count, arguments.scratch_row_stride)
             .valid() &&
         structural_arguments_valid(
             arguments.scratch, arguments.token_count,
             arguments.scratch_row_stride, arguments.conv_weight,
             arguments.a_log, arguments.dt_bias, arguments.norm_weight,
             arguments.active_conv_history,
             arguments.candidate_conv_history,
             arguments.active_recurrent_state,
             arguments.candidate_recurrent_state,
             arguments.cancellation_signal, arguments.l2_epsilon_fp32_bits,
             arguments.norm_epsilon_fp32_bits, arguments.cuda_stream, true);
}

int query_sm87_macrofeed_v4_gdn_c8000_admission_resource_snapshot_cuda(
    Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot* const
        resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status = validate_fixed_device(&device, &properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->identity = kSm87MacroFeedV4GdnC8000Identity;
  resources->device_ordinal = device;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  resources->sm_count = properties.multiProcessorCount;
  status = inspect_kernel(
      causal_conv1d_silu_c8000_in_place_kernel,
      static_cast<int>(kSm87MacroFeedV4GdnConvThreads),
      static_cast<int>(kSm87MacroFeedV4GdnConvCtas),
      &resources->convolution);
  if (status == cudaSuccess) {
    status = inspect_kernel(
        exact_gdn_norm_gate_c8000_in_place_kernel,
        static_cast<int>(kSm87MacroFeedV4GdnRecurrenceThreads),
        static_cast<int>(kSm87MacroFeedV4GdnRecurrenceCtas),
        &resources->recurrence_epilogue);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes convolution_attributes{};
  cudaFuncAttributes recurrence_attributes{};
  status = cudaFuncGetAttributes(&convolution_attributes,
                                 causal_conv1d_silu_c8000_in_place_kernel);
  if (status == cudaSuccess) {
    status = cudaFuncGetAttributes(
        &recurrence_attributes,
        exact_gdn_norm_gate_c8000_in_place_kernel);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->binary_version =
      convolution_attributes.binaryVersion ==
              recurrence_attributes.binaryVersion
          ? convolution_attributes.binaryVersion
          : 0;
  resources->kernels_compiled = true;
  resources->exact_geometry = true;
  resources->startup_package_unbound = true;
  resources->execution_capability = false;
  resources->caller_snapshot_grants_production_authority = false;
  resources->static_resource_gate_passed = true;
  resources->static_resource_gate_passed =
      sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(*resources);
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_macrofeed_v4_gdn_c8000_admission_cuda(
    const Sm87MacroFeedV4GdnC8000Arguments& arguments,
    const Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot& resources,
    Sm87MacroFeedV4GdnC8000AdmissionLaunchReceipt* const receipt) noexcept {
  if (receipt == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *receipt = {};
  if (!sm87_macrofeed_v4_gdn_c8000_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(resources)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot observed_resources{};
  const int query_status =
      query_sm87_macrofeed_v4_gdn_c8000_admission_resource_snapshot_cuda(
          &observed_resources);
  if (query_status != static_cast<int>(cudaSuccess)) {
    return query_status;
  }
  if (resources.device_ordinal != observed_resources.device_ordinal) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }
  if (!admission_resource_snapshot_equal(resources, observed_resources)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  int stream_device = -1;
  cudaError_t status = cudaStreamGetDevice(stream, &stream_device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (stream_device != observed_resources.device_ordinal) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }
  if (!argument_device_allocation_ranges_owned(
          arguments, observed_resources.device_ordinal)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  ConstituentSubmitLedger submit_ledger{};
  status = enqueue_constituent(
      arguments.scratch, arguments.conv_weight, arguments.a_log,
      arguments.dt_bias, arguments.norm_weight,
      arguments.active_conv_history, arguments.candidate_conv_history,
      arguments.active_recurrent_state,
      arguments.candidate_recurrent_state, arguments.cancellation_signal,
      static_cast<unsigned int>(arguments.token_count),
      arguments.l2_epsilon_fp32_bits, arguments.norm_epsilon_fp32_bits,
      &submit_ledger, stream);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  receipt->identity = kSm87MacroFeedV4GdnC8000Identity;
  receipt->token_count = arguments.token_count;
  receipt->scratch_row_stride = arguments.scratch_row_stride;
  receipt->conv_history_copy_bytes = kSm87MacroFeedV4GdnConvHistoryBytes;
  receipt->whole_recurrent_epoch_copy_bytes = 0U;
  receipt->physical_kernel_launches = static_cast<std::uint32_t>(
      kSm87MacroFeedV4GdnPhysicalKernelLaunches);
  receipt->asynchronous_d2d_copies =
      static_cast<std::uint32_t>(kSm87MacroFeedV4GdnHistoryCopies);
  receipt->active_state_const = true;
  receipt->candidate_state_full_assignment_required = true;
  receipt->in_place_qkv_convolution = true;
  receipt->in_place_v_output = true;
  receipt->current_device_revalidated = true;
  receipt->caller_snapshot_exact_observed_match = true;
  receipt->caller_supplied_live_stream_required = true;
  receipt->live_stream_device_observed = true;
  receipt->device_allocation_ranges_owned = true;
  receipt->launch_enqueued = true;
  receipt->numerical_contract_qualified = false;
  receipt->production_dispatch_eligible = false;
  receipt->startup_package_unbound = true;
  receipt->execution_capability = false;
  receipt->caller_snapshot_grants_production_authority = false;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_macrofeed_v4_gdn_oracle_cuda(
    const Sm87MacroFeedV4GdnOracleArguments& arguments) noexcept {
  if (!structural_arguments_valid(
          arguments.scratch, arguments.token_count,
          arguments.scratch_row_stride, arguments.conv_weight,
          arguments.a_log, arguments.dt_bias, arguments.norm_weight,
          arguments.active_conv_history,
          arguments.candidate_conv_history,
          arguments.active_recurrent_state,
          arguments.candidate_recurrent_state,
          arguments.cancellation_signal,
          arguments.l2_epsilon_fp32_bits,
          arguments.norm_epsilon_fp32_bits, arguments.cuda_stream, false)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot resources{};
  const int query =
      query_sm87_macrofeed_v4_gdn_c8000_admission_resource_snapshot_cuda(
          &resources);
  if (query != static_cast<int>(cudaSuccess)) {
    return query;
  }
  if (!sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(resources)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  if (!argument_device_allocation_ranges_owned(arguments,
                                                resources.device_ordinal)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  (void)cudaGetLastError();
  ConstituentSubmitLedger submit_ledger{};
  return static_cast<int>(enqueue_constituent(
      arguments.scratch, arguments.conv_weight, arguments.a_log,
      arguments.dt_bias, arguments.norm_weight,
      arguments.active_conv_history, arguments.candidate_conv_history,
      arguments.active_recurrent_state,
      arguments.candidate_recurrent_state,
      arguments.cancellation_signal,
      static_cast<unsigned int>(arguments.token_count),
      arguments.l2_epsilon_fp32_bits, arguments.norm_epsilon_fp32_bits,
      &submit_ledger,
      reinterpret_cast<cudaStream_t>(arguments.cuda_stream)));
}

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
namespace sm87_macrofeed_v4_bound_launch_detail {

int enqueue_gdn_continuation_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4GdnContinuationC8000Arguments& arguments,
    const Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot& resources,
    Sm87MacroFeedV4GdnContinuationSubmitLedger* const
        submit_ledger) noexcept {
  if (submit_ledger == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *submit_ledger = {};
  const Sm87MacroFeedV4GdnC8000Arguments fixed{
      arguments.phase_scratch,
      kSm87MacroFeedV4GdnC8000Tokens,
      kSm87MacroFeedV4GdnScratchRowStride,
      arguments.conv_weight,
      arguments.a_log,
      arguments.dt_bias,
      arguments.norm_weight,
      arguments.active_conv_history,
      arguments.candidate_conv_history,
      arguments.active_recurrent_state,
      arguments.candidate_recurrent_state,
      arguments.cancellation_signal,
      arguments.l2_epsilon_fp32_bits,
      arguments.norm_epsilon_fp32_bits,
      token.cuda_stream_};
  if (!sm87_macrofeed_v4_gdn_c8000_arguments_valid(fixed) ||
      !resources.static_resource_gate_passed ||
      !sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(resources)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t prior_status = cudaPeekAtLastError();
  if (prior_status != cudaSuccess) {
    return static_cast<int>(prior_status);
  }
  ConstituentSubmitLedger accepted{};
  const cudaError_t status = enqueue_constituent(
      fixed.scratch, fixed.conv_weight, fixed.a_log, fixed.dt_bias,
      fixed.norm_weight, fixed.active_conv_history,
      fixed.candidate_conv_history, fixed.active_recurrent_state,
      fixed.candidate_recurrent_state, fixed.cancellation_signal,
      static_cast<unsigned int>(fixed.token_count),
      fixed.l2_epsilon_fp32_bits, fixed.norm_epsilon_fp32_bits,
      &accepted,
      reinterpret_cast<cudaStream_t>(token.cuda_stream_));
  submit_ledger->accepted_kernel_launches =
      accepted.accepted_kernel_launches;
  submit_ledger->asynchronous_d2d_copies =
      accepted.asynchronous_d2d_copies;
  submit_ledger->conv_history_copy_bytes =
      accepted.conv_history_copy_bytes;
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace sm87_macrofeed_v4_bound_launch_detail
#endif

}  // namespace q3x::kernels
