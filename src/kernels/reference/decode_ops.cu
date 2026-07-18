#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr std::size_t kMaximumBlocks = 65535U;

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

}  // namespace q3x::runtime
