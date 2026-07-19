#include "q3x/runtime/decode_ops.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr std::size_t kMaximumBlocks = 65535U;
constexpr std::size_t kResidualRmsHiddenSize = 5120U;
constexpr unsigned int kResidualRmsBlocks =
    kResidualRmsHiddenSize / kThreads;
constexpr std::size_t kResidualRmsBytes =
    kResidualRmsHiddenSize * sizeof(std::uint16_t);
constexpr std::size_t kFusedGqaQueryHeads = 24U;
constexpr std::size_t kFusedGqaKvHeads = 4U;
constexpr std::size_t kFusedGqaHeadDimension = 256U;
constexpr std::size_t kQkRopeQueryHeads = 24U;
constexpr std::size_t kQkRopeKvHeads = 4U;
constexpr std::size_t kFullPreprocessQueryHeads = 24U;
constexpr std::size_t kFullPreprocessKvHeads = 4U;
constexpr std::size_t kFullPreprocessHeadDimension = 256U;
constexpr unsigned int kBf16GreedyArgmaxBlocks =
    static_cast<unsigned int>(kBf16GreedyArgmaxWorkspaceResults - 1U);
constexpr std::size_t kBf16GreedyArgmaxStride =
    kBf16GreedyArgmaxBlocks * kThreads;
static_assert((kThreads & (kThreads - 1U)) == 0U);
static_assert(kBf16GreedyArgmaxBlocks != 0U &&
              kBf16GreedyArgmaxBlocks <= 32U);
static_assert(kResidualRmsHiddenSize % kThreads == 0U);
static_assert(kFusedGqaHeadDimension == kThreads);

[[nodiscard]] bool multiply_overflows(const std::size_t left,
                                      const std::size_t right) noexcept {
  return right != 0U &&
         left > std::numeric_limits<std::size_t>::max() / right;
}

[[nodiscard]] bool product3_overflows(const std::size_t first,
                                      const std::size_t second,
                                      const std::size_t third) noexcept {
  return multiply_overflows(first, second) ||
         multiply_overflows(first * second, third);
}

[[nodiscard]] bool valid_epsilon(const float epsilon) noexcept {
  return std::isfinite(epsilon) && epsilon > 0.0F;
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

[[nodiscard]] bool partially_overlaps(const void* const first,
                                      const std::size_t first_bytes,
                                      const void* const second,
                                      const std::size_t second_bytes) noexcept {
  return first != second &&
         ranges_overlap(first, first_bytes, second, second_bytes);
}

[[nodiscard]] unsigned int block_count(const std::size_t work_items) noexcept {
  const std::size_t needed =
      work_items / kThreads + (work_items % kThreads != 0U ? 1U : 0U);
  return static_cast<unsigned int>(needed < kMaximumBlocks ? needed
                                                           : kMaximumBlocks);
}

[[nodiscard]] unsigned int row_block_count(const std::size_t rows) noexcept {
  return static_cast<unsigned int>(rows < kMaximumBlocks ? rows
                                                          : kMaximumBlocks);
}

__device__ __forceinline__ float decode_bf16_device(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_device(
    const float value) {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fffffffU;
  if (magnitude > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ bool finite_bf16_device(
    const std::uint16_t value) {
  return (value & 0x7f80U) != 0x7f80U;
}

__global__ __launch_bounds__(kThreads, 1) void bf16_greedy_argmax_kernel(
    const std::uint16_t* const input, const std::size_t element_count,
    Bf16GreedyArgmaxResult* const partial_results) {
  __shared__ float maxima[kThreads];
  __shared__ std::uint32_t indices[kThreads];
  __shared__ std::uint16_t value_bits[kThreads];
  __shared__ std::uint16_t nonfinite[kThreads];

  std::uint32_t local_index = std::numeric_limits<std::uint32_t>::max();
  std::uint16_t local_bits = 0U;
  float local_maximum = __uint_as_float(0xff80'0000U);
  std::uint16_t local_nonfinite = 0U;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count; index += kBf16GreedyArgmaxStride) {
    const std::uint16_t bits = input[index];
    if (!finite_bf16_device(bits)) {
      local_nonfinite = 1U;
      continue;
    }
    const float value = decode_bf16_device(bits);
    if (local_index == std::numeric_limits<std::uint32_t>::max() ||
        value > local_maximum) {
      local_index = static_cast<std::uint32_t>(index);
      local_bits = bits;
      local_maximum = value;
    }
  }

  maxima[threadIdx.x] = local_maximum;
  indices[threadIdx.x] = local_index;
  value_bits[threadIdx.x] = local_bits;
  nonfinite[threadIdx.x] = local_nonfinite;
  __syncthreads();

  for (unsigned int offset = kThreads / 2U; offset != 0U; offset /= 2U) {
    if (threadIdx.x < offset) {
      const unsigned int other = threadIdx.x + offset;
      const std::uint32_t other_index = indices[other];
      const bool take_other =
          other_index != std::numeric_limits<std::uint32_t>::max() &&
          (indices[threadIdx.x] ==
               std::numeric_limits<std::uint32_t>::max() ||
           maxima[other] > maxima[threadIdx.x] ||
           (maxima[other] == maxima[threadIdx.x] &&
            other_index < indices[threadIdx.x]));
      if (take_other) {
        maxima[threadIdx.x] = maxima[other];
        indices[threadIdx.x] = other_index;
        value_bits[threadIdx.x] = value_bits[other];
      }
      nonfinite[threadIdx.x] = static_cast<std::uint16_t>(
          nonfinite[threadIdx.x] | nonfinite[other]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0U) {
    Bf16GreedyArgmaxResult& partial = partial_results[blockIdx.x];
    partial.index = indices[0];
    partial.value_bits = value_bits[0];
    partial.has_nonfinite = nonfinite[0];
  }
}

__global__ void bf16_greedy_argmax_finalize_kernel(
    const std::uint16_t* const input,
    const Bf16GreedyArgmaxResult* const partial_results,
    Bf16GreedyArgmaxResult* const result) {
  const unsigned int lane = threadIdx.x;
  std::uint32_t index = std::numeric_limits<std::uint32_t>::max();
  std::uint16_t bits = 0U;
  std::uint16_t has_nonfinite = 0U;
  float maximum = __uint_as_float(0xff80'0000U);
  if (lane < kBf16GreedyArgmaxBlocks) {
    const Bf16GreedyArgmaxResult partial = partial_results[lane];
    index = partial.index;
    bits = partial.value_bits;
    has_nonfinite = partial.has_nonfinite;
    if (index != std::numeric_limits<std::uint32_t>::max()) {
      maximum = decode_bf16_device(bits);
    }
  }

  constexpr unsigned int kMask = 0xffff'ffffU;
  for (unsigned int offset = 16U; offset != 0U; offset /= 2U) {
    const std::uint32_t other_index =
        __shfl_down_sync(kMask, index, offset);
    const std::uint16_t other_bits = static_cast<std::uint16_t>(
        __shfl_down_sync(kMask, static_cast<unsigned int>(bits), offset));
    const std::uint16_t other_nonfinite = static_cast<std::uint16_t>(
        __shfl_down_sync(kMask, static_cast<unsigned int>(has_nonfinite),
                         offset));
    const float other_maximum =
        __shfl_down_sync(kMask, maximum, offset);
    if (other_index != std::numeric_limits<std::uint32_t>::max() &&
        (index == std::numeric_limits<std::uint32_t>::max() ||
         other_maximum > maximum ||
         (other_maximum == maximum && other_index < index))) {
      index = other_index;
      bits = other_bits;
      maximum = other_maximum;
    }
    has_nonfinite = static_cast<std::uint16_t>(has_nonfinite |
                                               other_nonfinite);
  }
  if (lane == 0U) {
    result->index = index;
    result->value_bits =
        index == std::numeric_limits<std::uint32_t>::max() ? 0U
                                                           : input[index];
    result->has_nonfinite = has_nonfinite;
  }
}

__global__ void embedding_gather_kernel(
    const std::uint16_t* const table,
    const std::size_t offset,
    const std::size_t hidden_size,
    std::uint16_t* const output) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < hidden_size;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    output[index] = table[offset + index];
  }
}

template <bool kCentered>
__global__ void rms_norm_kernel(const std::uint16_t* const input,
                                const std::uint16_t* const weight,
                                const std::size_t hidden_size,
                                const float epsilon,
                                std::uint16_t* const output) {
  __shared__ float partial[kThreads];
  float sum = 0.0F;
  for (std::size_t index = threadIdx.x; index < hidden_size;
       index += blockDim.x) {
    const float value = decode_bf16_device(input[index]);
    sum = fmaf(value, value, sum);
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(partial[0] / static_cast<float>(hidden_size) + epsilon);
  for (std::size_t index = threadIdx.x; index < hidden_size;
       index += blockDim.x) {
    const float gamma = decode_bf16_device(weight[index]) +
                        (kCentered ? 1.0F : 0.0F);
    output[index] = encode_bf16_device(
        decode_bf16_device(input[index]) * inverse_rms * gamma);
  }
}

template <bool kCentered, bool kApplySiluGate>
__global__ void headwise_rms_norm_kernel(
    const std::uint16_t* const input,
    const std::uint16_t* const shared_weight,
    const std::uint16_t* const gate,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output) {
  __shared__ float partial[kThreads];
  for (std::size_t head = static_cast<std::size_t>(blockIdx.x);
       head < head_count; head += static_cast<std::size_t>(gridDim.x)) {
    const std::size_t offset = head * head_dimension;
    float sum = 0.0F;
    for (std::size_t dimension = threadIdx.x; dimension < head_dimension;
         dimension += blockDim.x) {
      const float value = decode_bf16_device(input[offset + dimension]);
      sum = fmaf(value, value, sum);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    const float inverse_rms =
        rsqrtf(partial[0] / static_cast<float>(head_dimension) + epsilon);
    for (std::size_t dimension = threadIdx.x; dimension < head_dimension;
         dimension += blockDim.x) {
      const float gamma = decode_bf16_device(shared_weight[dimension]) +
                          (kCentered ? 1.0F : 0.0F);
      float value = decode_bf16_device(input[offset + dimension]) *
                    inverse_rms * gamma;
      if constexpr (kApplySiluGate) {
        const float gate_value =
            decode_bf16_device(gate[offset + dimension]);
        value *= gate_value / (1.0F + expf(-gate_value));
      }
      output[offset + dimension] = encode_bf16_device(value);
    }
    __syncthreads();
  }
}

// One block owns one Q or K head for one token. Q blocks also perform the raw
// interleaved Q/gate split. The reduction, centered RMSNorm, BF16 boundary,
// and RoPE instruction order match the corresponding reference kernels.
__global__ void full_attention_preprocess_24_4_256_64_kernel(
    const std::uint16_t* const interleaved_q_gate,
    std::uint16_t* const key,
    const std::uint16_t* const q_weight,
    const std::uint16_t* const k_weight,
    const float epsilon,
    std::uint16_t* const query_output,
    std::uint16_t* const gate_output,
    const float* const cosines,
    const float* const sines,
    const std::size_t first_position) {
  constexpr std::size_t kCombinedHeads =
      kFullPreprocessQueryHeads + kFullPreprocessKvHeads;
  __shared__ float partial[kThreads];

  const std::size_t combined_head = static_cast<std::size_t>(blockIdx.x);
  const std::size_t token = combined_head / kCombinedHeads;
  const std::size_t token_head = combined_head - token * kCombinedHeads;
  const bool is_query = token_head < kFullPreprocessQueryHeads;
  const std::size_t head =
      is_query ? token_head : token_head - kFullPreprocessQueryHeads;
  const std::size_t dimension = threadIdx.x;

  std::size_t packed_offset = 0U;
  const std::uint16_t* weight = nullptr;
  float value = 0.0F;
  if (is_query) {
    packed_offset =
        (token * kFullPreprocessQueryHeads + head) *
            kFullPreprocessHeadDimension +
        dimension;
    const std::size_t interleaved_offset =
        (token * kFullPreprocessQueryHeads + head) *
            (2U * kFullPreprocessHeadDimension) +
        dimension;
    value = decode_bf16_device(interleaved_q_gate[interleaved_offset]);
    gate_output[packed_offset] =
        interleaved_q_gate[interleaved_offset +
                           kFullPreprocessHeadDimension];
    weight = q_weight;
  } else {
    packed_offset =
        (token * kFullPreprocessKvHeads + head) *
            kFullPreprocessHeadDimension +
        dimension;
    value = decode_bf16_device(key[packed_offset]);
    weight = k_weight;
  }

  float sum = 0.0F;
  sum = fmaf(value, value, sum);
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(partial[0] /
                 static_cast<float>(kFullPreprocessHeadDimension) +
             epsilon);
  const float gamma = decode_bf16_device(weight[dimension]) + 1.0F;
  const std::uint16_t normalized =
      encode_bf16_device(value * inverse_rms * gamma);
  std::uint16_t* const normalized_output = is_query ? query_output : key;
  normalized_output[packed_offset] = normalized;

  constexpr std::size_t kHalfRotary = kQwenRotaryDimension / 2U;
  __syncthreads();
  if (dimension < kHalfRotary) {
    std::uint16_t* const head_output =
        normalized_output + packed_offset - dimension;
    const std::size_t table_offset =
        (first_position + token) * kHalfRotary + dimension;
    const float cosine = cosines[table_offset];
    const float sine = sines[table_offset];
    const float first = decode_bf16_device(head_output[dimension]);
    const float second =
        decode_bf16_device(head_output[dimension + kHalfRotary]);
    // Spell out which product is rounded before the FFMA. This is the exact
    // instruction order emitted for qk_partial_neox_rope_tile above.
    const float rotated_first = fmaf(first, cosine, -(second * sine));
    const float rotated_second = fmaf(second, cosine, first * sine);
    head_output[dimension] = encode_bf16_device(rotated_first);
    head_output[dimension + kHalfRotary] =
        encode_bf16_device(rotated_second);
  }
}

__global__ void residual_add_kernel(const std::uint16_t* const left,
                                    const std::uint16_t* const right,
                                    const std::size_t element_count,
                                    std::uint16_t* const output) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    output[index] = encode_bf16_device(decode_bf16_device(left[index]) +
                                       decode_bf16_device(right[index]));
  }
}

__global__ void residual_add_centered_rms_norm_5120_kernel(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::uint16_t* const weight,
    const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output) {
  const std::size_t residual_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  residual_output[residual_index] =
      encode_bf16_device(decode_bf16_device(left[residual_index]) +
                         decode_bf16_device(right[residual_index]));

  cooperative_groups::this_grid().sync();
  if (blockIdx.x != 0U) {
    return;
  }

  __shared__ float partial[kThreads];
  float sum = 0.0F;
  for (std::size_t index = threadIdx.x; index < kResidualRmsHiddenSize;
       index += blockDim.x) {
    const float value = decode_bf16_device(residual_output[index]);
    sum = fmaf(value, value, sum);
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(partial[0] / static_cast<float>(kResidualRmsHiddenSize) +
             epsilon);
  for (std::size_t index = threadIdx.x; index < kResidualRmsHiddenSize;
       index += blockDim.x) {
    const float gamma = decode_bf16_device(weight[index]) + 1.0F;
    normalized_output[index] = encode_bf16_device(
        decode_bf16_device(residual_output[index]) * inverse_rms * gamma);
  }
}

__global__ void fp32_to_bf16_kernel(const float* const input,
                                    const std::size_t element_count,
                                    std::uint16_t* const output) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    output[index] = encode_bf16_device(input[index]);
  }
}

__global__ void silu_mul_kernel(const std::uint16_t* const gate,
                                const std::uint16_t* const up,
                                const std::size_t element_count,
                                std::uint16_t* const output) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const float gate_value = decode_bf16_device(gate[index]);
    const float silu = gate_value / (1.0F + expf(-gate_value));
    output[index] =
        encode_bf16_device(silu * decode_bf16_device(up[index]));
  }
}

__global__ void sigmoid_gate_kernel(const std::uint16_t* const value,
                                    const std::uint16_t* const gate,
                                    const std::size_t element_count,
                                    std::uint16_t* const output) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const float gate_value = decode_bf16_device(gate[index]);
    const float sigmoid =
        gate_value >= 0.0F
            ? 1.0F / (1.0F + expf(-gate_value))
            : expf(gate_value) / (1.0F + expf(gate_value));
    output[index] =
        encode_bf16_device(decode_bf16_device(value[index]) * sigmoid);
  }
}

__global__ void l2_normalize_heads_kernel(
    const std::uint16_t* const input,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output) {
  __shared__ float partial[kThreads];
  for (std::size_t head = static_cast<std::size_t>(blockIdx.x);
       head < head_count; head += static_cast<std::size_t>(gridDim.x)) {
    const std::size_t offset = head * head_dimension;
    float sum = 0.0F;
    for (std::size_t dimension = threadIdx.x; dimension < head_dimension;
         dimension += blockDim.x) {
      const float value = decode_bf16_device(input[offset + dimension]);
      sum = fmaf(value, value, sum);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    const float inverse_norm = rsqrtf(partial[0] + epsilon);
    for (std::size_t dimension = threadIdx.x; dimension < head_dimension;
         dimension += blockDim.x) {
      output[offset + dimension] = encode_bf16_device(
          decode_bf16_device(input[offset + dimension]) * inverse_norm);
    }
    __syncthreads();
  }
}

__global__ void partial_neox_rope_kernel(
    const std::uint16_t* const input,
    const float* const cosines,
    const float* const sines,
    const std::size_t head_count,
    std::uint16_t* const output) {
  constexpr std::size_t kHalfRotary = kQwenRotaryDimension / 2U;
  constexpr std::size_t kTasksPerHead =
      kHalfRotary +
      (kFullAttentionHeadDimension - kQwenRotaryDimension);
  const std::size_t total_tasks = head_count * kTasksPerHead;
  for (std::size_t task =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < total_tasks;
       task += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t head = task / kTasksPerHead;
    const std::size_t local = task - head * kTasksPerHead;
    const std::size_t offset = head * kFullAttentionHeadDimension;
    if (local < kHalfRotary) {
      const float first = decode_bf16_device(input[offset + local]);
      const float second =
          decode_bf16_device(input[offset + local + kHalfRotary]);
      output[offset + local] = encode_bf16_device(
          first * cosines[local] - second * sines[local]);
      output[offset + local + kHalfRotary] = encode_bf16_device(
          second * cosines[local] + first * sines[local]);
    } else {
      const std::size_t dimension =
          kQwenRotaryDimension + (local - kHalfRotary);
      output[offset + dimension] = input[offset + dimension];
    }
  }
}

// One warp owns one Q or K head for one token. Its per-pair arithmetic remains
// bitwise equivalent to the reference kernel while non-rotary BF16 payload is
// already in place and does not need to be copied.
__global__ void qk_partial_neox_rope_tile_24_4_256_64_kernel(
    std::uint16_t* const query,
    std::uint16_t* const key,
    const float* const cosines,
    const float* const sines,
    const std::size_t first_position) {
  constexpr std::size_t kCombinedHeads =
      kQkRopeQueryHeads + kQkRopeKvHeads;
  constexpr std::size_t kHalfRotary = kQwenRotaryDimension / 2U;
  const std::size_t combined_head = static_cast<std::size_t>(blockIdx.x);
  const std::size_t token = combined_head / kCombinedHeads;
  const std::size_t token_head = combined_head - token * kCombinedHeads;
  const bool is_query = token_head < kQkRopeQueryHeads;
  const std::size_t head =
      is_query ? token_head : token_head - kQkRopeQueryHeads;
  std::uint16_t* const token_data =
      is_query
          ? query + token * kQkRopeQueryHeads *
                        kFullAttentionHeadDimension
          : key + token * kQkRopeKvHeads *
                      kFullAttentionHeadDimension;
  std::uint16_t* const head_data =
      token_data + head * kFullAttentionHeadDimension;
  const std::size_t position = first_position + token;
  const float* const token_cosines =
      cosines + position * kHalfRotary;
  const float* const token_sines = sines + position * kHalfRotary;

  if (threadIdx.x < kHalfRotary) {
    const std::size_t pair = threadIdx.x;
    const float first = decode_bf16_device(head_data[pair]);
    const float second =
        decode_bf16_device(head_data[pair + kHalfRotary]);
    head_data[pair] = encode_bf16_device(
        first * token_cosines[pair] - second * token_sines[pair]);
    head_data[pair + kHalfRotary] = encode_bf16_device(
        second * token_cosines[pair] + first * token_sines[pair]);
  }
}

__global__ void softmax_kernel(const float* const input,
                               const std::size_t rows,
                               const std::size_t columns,
                               float* const output) {
  __shared__ float partial[kThreads];
  for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < rows;
       row += static_cast<std::size_t>(gridDim.x)) {
    const std::size_t offset = row * columns;
    float maximum = -__int_as_float(0x7f800000);
    for (std::size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
      maximum = fmaxf(maximum, input[offset + column]);
    }
    partial[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] =
            fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    maximum = partial[0];

    float denominator = 0.0F;
    for (std::size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
      denominator += expf(input[offset + column] - maximum);
    }
    partial[threadIdx.x] = denominator;
    __syncthreads();
    for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    denominator = partial[0];
    for (std::size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
      output[offset + column] =
          expf(input[offset + column] - maximum) / denominator;
    }
    __syncthreads();
  }
}

__global__ void attention_scores_kernel(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    const float attention_scale,
    float* const scores) {
  __shared__ float partial[kThreads];
  const std::size_t queries_per_kv = query_head_count / kv_head_count;
  for (std::size_t query_head = static_cast<std::size_t>(blockIdx.x);
       query_head < query_head_count;
       query_head += static_cast<std::size_t>(gridDim.x)) {
    const std::size_t kv_head = query_head / queries_per_kv;
    const std::size_t query_offset = query_head * head_dimension;
    for (std::size_t position = 0; position < sequence_length; ++position) {
      const std::size_t key_offset =
          (position * kv_head_count + kv_head) * head_dimension;
      float score = 0.0F;
      for (std::size_t dimension = threadIdx.x; dimension < head_dimension;
           dimension += blockDim.x) {
        score = fmaf(decode_bf16_device(query[query_offset + dimension]),
                     decode_bf16_device(key_cache[key_offset + dimension]),
                     score);
      }
      partial[threadIdx.x] = score;
      __syncthreads();
      for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
          partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0U) {
        scores[query_head * sequence_length + position] =
            partial[0] * attention_scale;
      }
      __syncthreads();
    }
  }
}

__global__ void attention_values_kernel(
    const std::uint16_t* const value_cache,
    const float* const probabilities,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    std::uint16_t* const output) {
  const std::size_t queries_per_kv = query_head_count / kv_head_count;
  for (std::size_t query_head = static_cast<std::size_t>(blockIdx.x);
       query_head < query_head_count;
       query_head += static_cast<std::size_t>(gridDim.x)) {
    const std::size_t kv_head = query_head / queries_per_kv;
    const std::size_t probability_offset = query_head * sequence_length;
    const std::size_t output_offset = query_head * head_dimension;
    for (std::size_t dimension = threadIdx.x; dimension < head_dimension;
         dimension += blockDim.x) {
      float value = 0.0F;
      for (std::size_t position = 0; position < sequence_length; ++position) {
        const std::size_t value_offset =
            (position * kv_head_count + kv_head) * head_dimension;
        value = fmaf(probabilities[probability_offset + position],
                     decode_bf16_device(value_cache[value_offset + dimension]),
                     value);
      }
      output[output_offset + dimension] = encode_bf16_device(value);
    }
  }
}

// Each CTA owns one query head, so the score, softmax, value, and gate phases
// retain the reference kernels' thread mapping and reduction order. The
// explicit BF16 encode/decode below preserves the observable rounding boundary
// between attention_values_kernel and sigmoid_gate_kernel.
__global__ void gqa_sigmoid_gate_24_4_256_kernel(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::size_t sequence_length,
    const float attention_scale,
    float* const probabilities_scratch,
    const std::uint16_t* const gate,
    std::uint16_t* const output) {
  __shared__ float partial[kThreads];
  __shared__ float probabilities[kFusedGqaMaximumSequenceLength];

  const std::size_t query_head = static_cast<std::size_t>(blockIdx.x);
  const std::size_t queries_per_kv =
      kFusedGqaQueryHeads / kFusedGqaKvHeads;
  const std::size_t kv_head = query_head / queries_per_kv;
  const std::size_t query_offset =
      query_head * kFusedGqaHeadDimension;

  for (std::size_t position = 0U; position < sequence_length; ++position) {
    const std::size_t key_offset =
        (position * kFusedGqaKvHeads + kv_head) * kFusedGqaHeadDimension;
    float score = 0.0F;
    for (std::size_t dimension = threadIdx.x;
         dimension < kFusedGqaHeadDimension;
         dimension += blockDim.x) {
      score = fmaf(decode_bf16_device(query[query_offset + dimension]),
                   decode_bf16_device(key_cache[key_offset + dimension]),
                   score);
    }
    partial[threadIdx.x] = score;
    __syncthreads();
    for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0U) {
      probabilities[position] = partial[0] * attention_scale;
    }
    __syncthreads();
  }

  float maximum = -__int_as_float(0x7f800000);
  for (std::size_t position = threadIdx.x; position < sequence_length;
       position += blockDim.x) {
    maximum = fmaxf(maximum, probabilities[position]);
  }
  partial[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] =
          fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  maximum = partial[0];

  float denominator = 0.0F;
  for (std::size_t position = threadIdx.x; position < sequence_length;
       position += blockDim.x) {
    denominator += expf(probabilities[position] - maximum);
  }
  partial[threadIdx.x] = denominator;
  __syncthreads();
  for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  denominator = partial[0];

  const std::size_t probability_offset = query_head * sequence_length;
  for (std::size_t position = threadIdx.x; position < sequence_length;
       position += blockDim.x) {
    const float probability =
        expf(probabilities[position] - maximum) / denominator;
    probabilities[position] = probability;
    probabilities_scratch[probability_offset + position] = probability;
  }
  __syncthreads();

  for (std::size_t dimension = threadIdx.x;
       dimension < kFusedGqaHeadDimension;
       dimension += blockDim.x) {
    float value = 0.0F;
    for (std::size_t position = 0U; position < sequence_length; ++position) {
      const std::size_t value_offset =
          (position * kFusedGqaKvHeads + kv_head) * kFusedGqaHeadDimension;
      value = fmaf(probabilities[position],
                   decode_bf16_device(
                       value_cache[value_offset + dimension]),
                   value);
    }
    const std::uint16_t encoded_value = encode_bf16_device(value);
    const float gate_value =
        decode_bf16_device(gate[query_offset + dimension]);
    const float sigmoid =
        gate_value >= 0.0F
            ? 1.0F / (1.0F + expf(-gate_value))
            : expf(gate_value) / (1.0F + expf(gate_value));
    output[query_offset + dimension] = encode_bf16_device(
        decode_bf16_device(encoded_value) * sigmoid);
  }
}

[[nodiscard]] bool valid_attention_dimensions(
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension) noexcept {
  return query_head_count != 0U && kv_head_count != 0U &&
         sequence_length != 0U && head_dimension != 0U &&
         query_head_count % kv_head_count == 0U;
}

template <bool kCentered, bool kApplySiluGate>
[[nodiscard]] int launch_headwise_rms_norm(
    const std::uint16_t* const input,
    const std::uint16_t* const shared_weight,
    const std::uint16_t* const gate,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!valid_epsilon(epsilon) ||
      multiply_overflows(head_count, head_dimension)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (head_count == 0U || head_dimension == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || shared_weight == nullptr || output == nullptr ||
      (kApplySiluGate && gate == nullptr)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  headwise_rms_norm_kernel<kCentered, kApplySiluGate>
      <<<row_block_count(head_count), kThreads, 0U, stream>>>(
          input, shared_weight, gate, head_count, head_dimension, epsilon,
          output);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

int launch_embedding_gather_reference_cuda(
    const std::uint16_t* const embedding_table,
    const std::size_t vocabulary_size,
    const std::size_t hidden_size,
    const std::size_t token_id,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (multiply_overflows(vocabulary_size, hidden_size) ||
      token_id >= vocabulary_size) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (hidden_size == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (embedding_table == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  embedding_gather_kernel<<<block_count(hidden_size), kThreads, 0U, stream>>>(
      embedding_table, token_id * hidden_size, hidden_size, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_centered_rms_norm_reference_cuda(
    const std::uint16_t* const input,
    const std::uint16_t* const weight,
    const std::size_t hidden_size,
    const float epsilon,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!valid_epsilon(epsilon)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (hidden_size == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || weight == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  rms_norm_kernel<true><<<1U, kThreads, 0U, stream>>>(
      input, weight, hidden_size, epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_plain_rms_norm_reference_cuda(
    const std::uint16_t* const input,
    const std::uint16_t* const weight,
    const std::size_t hidden_size,
    const float epsilon,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!valid_epsilon(epsilon)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (hidden_size == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || weight == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  rms_norm_kernel<false><<<1U, kThreads, 0U, stream>>>(
      input, weight, hidden_size, epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_headwise_centered_rms_norm_reference_cuda(
    const std::uint16_t* const input,
    const std::uint16_t* const shared_weight,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  return launch_headwise_rms_norm<true, false>(
      input, shared_weight, nullptr, head_count, head_dimension, epsilon,
      output, cuda_stream);
}

int launch_headwise_plain_rms_norm_reference_cuda(
    const std::uint16_t* const input,
    const std::uint16_t* const shared_weight,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  return launch_headwise_rms_norm<false, false>(
      input, shared_weight, nullptr, head_count, head_dimension, epsilon,
      output, cuda_stream);
}

int launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
    const std::uint16_t* const input,
    const std::uint16_t* const shared_weight,
    const std::uint16_t* const gate,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  return launch_headwise_rms_norm<false, true>(
      input, shared_weight, gate, head_count, head_dimension, epsilon, output,
      cuda_stream);
}

int launch_residual_add_reference_cuda(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::size_t element_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (element_count == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (left == nullptr || right == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  residual_add_kernel<<<block_count(element_count), kThreads, 0U, stream>>>(
      left, right, element_count, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_residual_add_centered_rms_norm_5120_cuda(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::uint16_t* const weight,
    const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output,
    void* const cuda_stream) noexcept {
  if (!valid_epsilon(epsilon) || left == nullptr || right == nullptr ||
      weight == nullptr || residual_output == nullptr ||
      normalized_output == nullptr ||
      byte_range_overflows(left, kResidualRmsBytes) ||
      byte_range_overflows(right, kResidualRmsBytes) ||
      byte_range_overflows(weight, kResidualRmsBytes) ||
      byte_range_overflows(residual_output, kResidualRmsBytes) ||
      byte_range_overflows(normalized_output, kResidualRmsBytes) ||
      ranges_overlap(residual_output, kResidualRmsBytes, left,
                     kResidualRmsBytes) ||
      ranges_overlap(residual_output, kResidualRmsBytes, right,
                     kResidualRmsBytes) ||
      ranges_overlap(residual_output, kResidualRmsBytes, weight,
                     kResidualRmsBytes) ||
      ranges_overlap(residual_output, kResidualRmsBytes, normalized_output,
                     kResidualRmsBytes) ||
      ranges_overlap(normalized_output, kResidualRmsBytes, left,
                     kResidualRmsBytes) ||
      partially_overlaps(normalized_output, kResidualRmsBytes, right,
                         kResidualRmsBytes) ||
      ranges_overlap(normalized_output, kResidualRmsBytes, weight,
                     kResidualRmsBytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  const std::uint16_t* left_argument = left;
  const std::uint16_t* right_argument = right;
  const std::uint16_t* weight_argument = weight;
  float epsilon_argument = epsilon;
  std::uint16_t* residual_argument = residual_output;
  std::uint16_t* normalized_argument = normalized_output;
  void* arguments[] = {&left_argument,       &right_argument,
                       &weight_argument,     &epsilon_argument,
                       &residual_argument,   &normalized_argument};
  const cudaError_t launch_status = cudaLaunchCooperativeKernel(
      residual_add_centered_rms_norm_5120_kernel,
      dim3{kResidualRmsBlocks}, dim3{kThreads}, arguments, 0U, stream);
  if (launch_status != cudaSuccess) {
    return static_cast<int>(launch_status);
  }
  return static_cast<int>(cudaGetLastError());
}

int launch_fp32_to_bf16_reference_cuda(
    const float* const input,
    const std::size_t element_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (element_count == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp32_to_bf16_kernel<<<block_count(element_count), kThreads, 0U, stream>>>(
      input, element_count, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_bf16_greedy_argmax_cuda(
    const std::uint16_t* const input,
    const std::size_t element_count,
    Bf16GreedyArgmaxResult* const result_workspace,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kWorkspaceBytes =
      kBf16GreedyArgmaxWorkspaceResults *
      sizeof(Bf16GreedyArgmaxResult);
  if (input == nullptr || result_workspace == nullptr ||
      element_count == 0U ||
      element_count > std::numeric_limits<std::uint32_t>::max() ||
      multiply_overflows(element_count, sizeof(std::uint16_t)) ||
      ranges_overlap(input, element_count * sizeof(std::uint16_t),
                     result_workspace, kWorkspaceBytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  bf16_greedy_argmax_kernel<<<kBf16GreedyArgmaxBlocks, kThreads, 0U,
                              stream>>>(
      input, element_count, result_workspace + 1U);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  bf16_greedy_argmax_finalize_kernel<<<1U, 32U, 0U, stream>>>(
      input, result_workspace + 1U, result_workspace);
  return static_cast<int>(cudaGetLastError());
}

int launch_silu_mul_reference_cuda(
    const std::uint16_t* const gate,
    const std::uint16_t* const up,
    const std::size_t element_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (element_count == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (gate == nullptr || up == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  silu_mul_kernel<<<block_count(element_count), kThreads, 0U, stream>>>(
      gate, up, element_count, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_sigmoid_gate_reference_cuda(
    const std::uint16_t* const value,
    const std::uint16_t* const gate,
    const std::size_t element_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (element_count == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (value == nullptr || gate == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  sigmoid_gate_kernel<<<block_count(element_count), kThreads, 0U, stream>>>(
      value, gate, element_count, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_l2_normalize_heads_reference_cuda(
    const std::uint16_t* const input,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!valid_epsilon(epsilon) ||
      multiply_overflows(head_count, head_dimension)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (head_count == 0U || head_dimension == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  l2_normalize_heads_kernel<<<row_block_count(head_count), kThreads, 0U,
                              stream>>>(input, head_count, head_dimension,
                                        epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_partial_neox_rope_256_64_reference_cuda(
    const std::uint16_t* const input,
    const float* const cosines,
    const float* const sines,
    const std::size_t head_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (multiply_overflows(head_count, kFullAttentionHeadDimension)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (head_count == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || cosines == nullptr || sines == nullptr ||
      output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr std::size_t kTasksPerHead =
      kQwenRotaryDimension / 2U +
      (kFullAttentionHeadDimension - kQwenRotaryDimension);
  const std::size_t work_items = head_count * kTasksPerHead;
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  partial_neox_rope_kernel<<<block_count(work_items), kThreads, 0U, stream>>>(
      input, cosines, sines, head_count, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_qk_partial_neox_rope_tile_24_4_256_64_cuda(
    std::uint16_t* const query,
    std::uint16_t* const key,
    const float* const cosines,
    const float* const sines,
    const std::size_t first_position,
    const std::size_t token_count,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kHalfRotary = kQwenRotaryDimension / 2U;
  if (token_count == 0U || token_count > kQkRopeTileMaximumTokens ||
      first_position >
          std::numeric_limits<std::size_t>::max() - token_count ||
      product3_overflows(first_position + token_count, kHalfRotary,
                         sizeof(float)) ||
      query == nullptr || key == nullptr || cosines == nullptr ||
      sines == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr std::size_t kCombinedHeads =
      kQkRopeQueryHeads + kQkRopeKvHeads;
  constexpr unsigned int kRopeThreads =
      static_cast<unsigned int>(kHalfRotary);
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  qk_partial_neox_rope_tile_24_4_256_64_kernel
      <<<static_cast<unsigned int>(token_count * kCombinedHeads),
         kRopeThreads, 0U, stream>>>(query, key, cosines, sines,
                                     first_position);
  return static_cast<int>(cudaGetLastError());
}

int launch_full_attention_preprocess_24_4_256_64_cuda(
    const std::uint16_t* const interleaved_q_gate,
    std::uint16_t* const key,
    const std::uint16_t* const q_weight,
    const std::uint16_t* const k_weight,
    const float epsilon,
    std::uint16_t* const query_output,
    std::uint16_t* const gate_output,
    const float* const cosines,
    const float* const sines,
    const std::size_t first_position,
    const std::size_t token_count,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kHalfRotary = kQwenRotaryDimension / 2U;
  if (token_count == 0U || token_count > kQkRopeTileMaximumTokens ||
      !valid_epsilon(epsilon) ||
      first_position >
          std::numeric_limits<std::size_t>::max() - token_count ||
      product3_overflows(first_position + token_count, kHalfRotary,
                         sizeof(float)) ||
      interleaved_q_gate == nullptr || key == nullptr ||
      q_weight == nullptr || k_weight == nullptr || query_output == nullptr ||
      gate_output == nullptr || cosines == nullptr || sines == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  constexpr std::size_t kQueryElementsPerToken =
      kFullPreprocessQueryHeads * kFullPreprocessHeadDimension;
  constexpr std::size_t kKeyElementsPerToken =
      kFullPreprocessKvHeads * kFullPreprocessHeadDimension;
  const std::size_t query_bytes =
      token_count * kQueryElementsPerToken * sizeof(std::uint16_t);
  const std::size_t interleaved_bytes = 2U * query_bytes;
  const std::size_t key_bytes =
      token_count * kKeyElementsPerToken * sizeof(std::uint16_t);
  constexpr std::size_t kWeightBytes =
      kFullPreprocessHeadDimension * sizeof(std::uint16_t);
  const std::size_t table_bytes =
      (first_position + token_count) * kHalfRotary * sizeof(float);

  if (ranges_overlap(interleaved_q_gate, interleaved_bytes, key, key_bytes) ||
      ranges_overlap(interleaved_q_gate, interleaved_bytes, query_output,
                     query_bytes) ||
      ranges_overlap(interleaved_q_gate, interleaved_bytes, gate_output,
                     query_bytes) ||
      ranges_overlap(key, key_bytes, q_weight, kWeightBytes) ||
      ranges_overlap(key, key_bytes, k_weight, kWeightBytes) ||
      ranges_overlap(key, key_bytes, query_output, query_bytes) ||
      ranges_overlap(key, key_bytes, gate_output, query_bytes) ||
      ranges_overlap(key, key_bytes, cosines, table_bytes) ||
      ranges_overlap(key, key_bytes, sines, table_bytes) ||
      ranges_overlap(query_output, query_bytes, gate_output, query_bytes) ||
      ranges_overlap(query_output, query_bytes, q_weight, kWeightBytes) ||
      ranges_overlap(query_output, query_bytes, k_weight, kWeightBytes) ||
      ranges_overlap(query_output, query_bytes, cosines, table_bytes) ||
      ranges_overlap(query_output, query_bytes, sines, table_bytes) ||
      ranges_overlap(gate_output, query_bytes, q_weight, kWeightBytes) ||
      ranges_overlap(gate_output, query_bytes, k_weight, kWeightBytes) ||
      ranges_overlap(gate_output, query_bytes, cosines, table_bytes) ||
      ranges_overlap(gate_output, query_bytes, sines, table_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  constexpr std::size_t kCombinedHeads =
      kFullPreprocessQueryHeads + kFullPreprocessKvHeads;
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  full_attention_preprocess_24_4_256_64_kernel
      <<<static_cast<unsigned int>(token_count * kCombinedHeads), kThreads,
         0U, stream>>>(interleaved_q_gate, key, q_weight, k_weight, epsilon,
                       query_output, gate_output, cosines, sines,
                       first_position);
  return static_cast<int>(cudaGetLastError());
}

int launch_softmax_reference_cuda(
    const float* const input,
    const std::size_t rows,
    const std::size_t columns,
    float* const output,
    void* const cuda_stream) noexcept {
  if (multiply_overflows(rows, columns)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (rows == 0U || columns == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  softmax_kernel<<<row_block_count(rows), kThreads, 0U, stream>>>(
      input, rows, columns, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_gqa_attention_reference_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    const float attention_scale,
    float* const probabilities_scratch,
    const std::size_t scratch_elements,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!valid_attention_dimensions(query_head_count, kv_head_count,
                                  sequence_length, head_dimension) ||
      !std::isfinite(attention_scale) || attention_scale < 0.0F ||
      multiply_overflows(query_head_count, head_dimension) ||
      product3_overflows(sequence_length, kv_head_count, head_dimension) ||
      multiply_overflows(query_head_count, sequence_length)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_scratch = query_head_count * sequence_length;
  if (scratch_elements < required_scratch || query == nullptr ||
      key_cache == nullptr || value_cache == nullptr ||
      probabilities_scratch == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  const unsigned int blocks = row_block_count(query_head_count);
  (void)cudaGetLastError();
  attention_scores_kernel<<<blocks, kThreads, 0U, stream>>>(
      query, key_cache, query_head_count, kv_head_count, sequence_length,
      head_dimension, attention_scale, probabilities_scratch);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  softmax_kernel<<<blocks, kThreads, 0U, stream>>>(
      probabilities_scratch, query_head_count, sequence_length,
      probabilities_scratch);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  attention_values_kernel<<<blocks, kThreads, 0U, stream>>>(
      value_cache, probabilities_scratch, query_head_count, kv_head_count,
      sequence_length, head_dimension, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::size_t sequence_length,
    const float attention_scale,
    float* const probabilities_scratch,
    const std::size_t scratch_elements,
    const std::uint16_t* const gate,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (sequence_length == 0U ||
      sequence_length > kFusedGqaMaximumSequenceLength ||
      !std::isfinite(attention_scale) || attention_scale < 0.0F ||
      scratch_elements < kFusedGqaQueryHeads * sequence_length ||
      query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      probabilities_scratch == nullptr || gate == nullptr ||
      output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gqa_sigmoid_gate_24_4_256_kernel
      <<<kFusedGqaQueryHeads, kThreads, 0U, stream>>>(
          query, key_cache, value_cache, sequence_length, attention_scale,
          probabilities_scratch, gate, output);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace q3x::runtime
