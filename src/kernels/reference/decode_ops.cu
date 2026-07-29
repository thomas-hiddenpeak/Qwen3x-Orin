#include "q3x/runtime/decode_ops.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {

int launch_bulk_causal_gqa_sigmoid_gate_24_4_256_c512_register_pipeline_cuda(
    const std::uint16_t* query,
    const std::uint16_t* key_cache,
    const std::uint16_t* value_cache,
    const std::uint16_t* gate,
    std::size_t first_position,
    std::size_t token_count,
    std::uint16_t* output,
    void* cuda_stream) noexcept;

namespace {

constexpr unsigned int kThreads = 256U;
constexpr std::size_t kMaximumBlocks = 65535U;
constexpr std::size_t kResidualRmsHiddenSize = 5120U;
constexpr unsigned int kResidualRmsBlocks =
    kResidualRmsHiddenSize / kThreads;
constexpr std::size_t kResidualRmsBytes =
    kResidualRmsHiddenSize * sizeof(std::uint16_t);
constexpr std::size_t kResidualRmsM32TokenCount = 32U;
constexpr std::size_t kResidualRmsPrefillMaximumTokenCount = 512U;
constexpr unsigned int kResidualRmsM32Threads = 512U;
constexpr std::size_t kFusedGqaQueryHeads = 24U;
constexpr std::size_t kFusedGqaKvHeads = 4U;
constexpr std::size_t kFusedGqaHeadDimension = 256U;
constexpr std::size_t kWarpPositionAttentionScoreQueryHeads =
    kFusedGqaQueryHeads;
constexpr std::size_t kWarpPositionAttentionScoreKvHeads =
    kFusedGqaKvHeads;
constexpr std::size_t kWarpPositionAttentionScoreHeadDimension =
    kFusedGqaHeadDimension;
constexpr unsigned int kWarpPositionAttentionScoreWarpsPerBlock = 8U;
constexpr std::size_t kWarpPositionAttentionScoreMaximumSequence =
    kMaximumBlocks * kWarpPositionAttentionScoreWarpsPerBlock;
constexpr std::size_t kExactAttentionValueQueryHeads = kFusedGqaQueryHeads;
constexpr std::size_t kExactAttentionValueKvHeads = kFusedGqaKvHeads;
constexpr std::size_t kExactAttentionValueHeadDimension =
    kFusedGqaHeadDimension;
constexpr unsigned int kExactAttentionValueQueriesPerKv =
    static_cast<unsigned int>(kExactAttentionValueQueryHeads /
                              kExactAttentionValueKvHeads);
constexpr unsigned int kExactAttentionValuePositionStride =
    static_cast<unsigned int>(kExactAttentionValueKvHeads *
                              kExactAttentionValueHeadDimension);
constexpr std::size_t kExactAttentionValueMaximumSequence =
    static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) /
    kExactAttentionValuePositionStride;
constexpr unsigned int kBulkGqaQueryHeads = 24U;
constexpr unsigned int kBulkGqaKvHeads = 4U;
constexpr unsigned int kBulkGqaQueriesPerKv =
    kBulkGqaQueryHeads / kBulkGqaKvHeads;
constexpr unsigned int kBulkGqaHeadDimension = 256U;
constexpr unsigned int kBulkGqaPackedDimension =
    kBulkGqaHeadDimension / 2U;
constexpr unsigned int kBulkGqaQueryTile = 2U;
constexpr unsigned int kBulkGqaKvTile = 16U;
constexpr unsigned int kBulkGqaThreads =
    kBulkGqaQueriesPerKv * 32U;
constexpr std::size_t kBulkGqaMaximumSequence =
    kBulkCausalGqaMaximumSequenceLength;
constexpr float kBulkGqaAttentionScale = 1.0F / 16.0F;
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
static_assert(kResidualRmsHiddenSize % kResidualRmsM32Threads == 0U);
static_assert(kResidualRmsM32TokenCount <= kMaximumBlocks);
static_assert(kResidualRmsPrefillMaximumTokenCount <= kMaximumBlocks);
static_assert(kResidualRmsM32TokenCount <=
              std::numeric_limits<std::size_t>::max() /
                  kResidualRmsHiddenSize);
static_assert(kFusedGqaHeadDimension == kThreads);
static_assert(kWarpPositionAttentionScoreHeadDimension == kThreads);
static_assert(kWarpPositionAttentionScoreWarpsPerBlock * 32U == kThreads);
static_assert(
    kMaximumBlocks <=
    std::numeric_limits<std::size_t>::max() /
        kWarpPositionAttentionScoreWarpsPerBlock);
static_assert(kWarpPositionAttentionScoreMaximumSequence <=
              std::numeric_limits<unsigned int>::max());
static_assert(
    kWarpPositionAttentionScoreMaximumSequence <=
    std::numeric_limits<unsigned int>::max() /
        (kWarpPositionAttentionScoreKvHeads *
         kWarpPositionAttentionScoreHeadDimension));
static_assert(kWarpPositionAttentionScoreMaximumSequence <=
              std::numeric_limits<unsigned int>::max() /
                  kWarpPositionAttentionScoreQueryHeads);
static_assert(kExactAttentionValueQueryHeads % kExactAttentionValueKvHeads ==
              0U);
static_assert(kExactAttentionValueQueriesPerKv *
                  kExactAttentionValueKvHeads ==
              kExactAttentionValueQueryHeads);
static_assert(kExactAttentionValueHeadDimension == kThreads);
static_assert(kExactAttentionValuePositionStride == 1'024U);
static_assert(kExactAttentionValueMaximumSequence != 0U);
static_assert(
    kExactAttentionValueMaximumSequence <=
    std::numeric_limits<unsigned int>::max() /
        kExactAttentionValueQueryHeads);
static_assert(kBulkGqaQueryHeads % kBulkGqaKvHeads == 0U);
static_assert(kBulkGqaQueriesPerKv == 6U);
static_assert(kBulkGqaPackedDimension * 2U == kBulkGqaHeadDimension);
static_assert(kBulkGqaThreads == 192U);

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

__global__ void embedding_gather_prompt_kernel(
    const std::uint16_t* const table,
    const std::size_t vocabulary_size,
    const std::size_t hidden_size,
    const std::uint32_t* const token_ids,
    std::uint16_t* const output) {
  const std::size_t token = blockIdx.x;
  const std::size_t token_id = token_ids[token];
  if (token_id >= vocabulary_size) {
    return;
  }
  const std::size_t table_offset = token_id * hidden_size;
  const std::size_t output_offset = token * hidden_size;
  for (std::size_t hidden = threadIdx.x; hidden < hidden_size;
       hidden += blockDim.x) {
    output[output_offset + hidden] = table[table_offset + hidden];
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

// Test-only fixed-shape candidate. One CTA still owns one Q or K head, but
// warp 0 constructs the 256-value RMS reduction locally. The scalar grouping
// exactly mirrors the production shared-memory tree's 128, 64, and 32 stages;
// the shuffle tree then reproduces its 16, 8, 4, 2, and 1 stages. All threads
// retain production's centered-RMSNorm and BF16 boundary, and the second CTA
// barrier retains production's normalized-input boundary before RoPE.
__global__ void full_attention_preprocess_warp_rms_24_4_256_64_test_kernel(
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
  __shared__ float inverse_rms_shared;

  const std::size_t combined_head = static_cast<std::size_t>(blockIdx.x);
  const std::size_t token = combined_head / kCombinedHeads;
  const std::size_t token_head = combined_head - token * kCombinedHeads;
  const bool is_query = token_head < kFullPreprocessQueryHeads;
  const std::size_t head =
      is_query ? token_head : token_head - kFullPreprocessQueryHeads;
  const std::size_t dimension = threadIdx.x;

  std::size_t packed_head_offset = 0U;
  const std::uint16_t* head_input = nullptr;
  const std::uint16_t* weight = nullptr;
  if (is_query) {
    packed_head_offset =
        (token * kFullPreprocessQueryHeads + head) *
        kFullPreprocessHeadDimension;
    head_input =
        interleaved_q_gate +
        (token * kFullPreprocessQueryHeads + head) *
            (2U * kFullPreprocessHeadDimension);
    gate_output[packed_head_offset + dimension] =
        head_input[kFullPreprocessHeadDimension + dimension];
    weight = q_weight;
  } else {
    packed_head_offset =
        (token * kFullPreprocessKvHeads + head) *
        kFullPreprocessHeadDimension;
    head_input = key + packed_head_offset;
    weight = k_weight;
  }

  const float value = decode_bf16_device(head_input[dimension]);
  if (dimension < 32U) {
    const std::size_t lane = dimension;
    float sum = fmaf(value, value, 0.0F);
    float rhs_value = decode_bf16_device(head_input[lane + 128U]);
    float rhs = fmaf(rhs_value, rhs_value, 0.0F);
    sum += rhs;

    float sibling_value = decode_bf16_device(head_input[lane + 64U]);
    float sibling = fmaf(sibling_value, sibling_value, 0.0F);
    rhs_value = decode_bf16_device(head_input[lane + 192U]);
    rhs = fmaf(rhs_value, rhs_value, 0.0F);
    sibling += rhs;
    sum += sibling;

    sibling_value = decode_bf16_device(head_input[lane + 32U]);
    sibling = fmaf(sibling_value, sibling_value, 0.0F);
    rhs_value = decode_bf16_device(head_input[lane + 160U]);
    rhs = fmaf(rhs_value, rhs_value, 0.0F);
    sibling += rhs;
    float upper_value = decode_bf16_device(head_input[lane + 96U]);
    float upper = fmaf(upper_value, upper_value, 0.0F);
    rhs_value = decode_bf16_device(head_input[lane + 224U]);
    rhs = fmaf(rhs_value, rhs_value, 0.0F);
    upper += rhs;
    sibling += upper;
    sum += sibling;

#pragma unroll
    for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
      rhs = __shfl_down_sync(0xffffffffU, sum, stride);
      if (dimension < stride) {
        sum += rhs;
      }
    }
    if (dimension == 0U) {
      inverse_rms_shared =
          rsqrtf(sum /
                     static_cast<float>(kFullPreprocessHeadDimension) +
                 epsilon);
    }
  }
  __syncthreads();

  const float gamma = decode_bf16_device(weight[dimension]) + 1.0F;
  const std::uint16_t normalized =
      encode_bf16_device(value * inverse_rms_shared * gamma);
  std::uint16_t* const normalized_output = is_query ? query_output : key;
  normalized_output[packed_head_offset + dimension] = normalized;

  constexpr std::size_t kHalfRotary = kQwenRotaryDimension / 2U;
  __syncthreads();
  if (dimension < kHalfRotary) {
    std::uint16_t* const head_output =
        normalized_output + packed_head_offset;
    const std::size_t table_offset =
        (first_position + token) * kHalfRotary + dimension;
    const float cosine = cosines[table_offset];
    const float sine = sines[table_offset];
    const float first = decode_bf16_device(head_output[dimension]);
    const float second =
        decode_bf16_device(head_output[dimension + kHalfRotary]);
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

// Exact prompt-span Prefill path. One 512-thread CTA owns one token and stages
// the rounded residual in shared memory. Grid size is the admitted span; the
// original M32 ABI and the prompt-wide ABI therefore execute the same
// per-token instructions, reduction tree, and BF16 boundaries. The lower 256
// lanes cache residual bits in registers for the output pass. Block-wide
// barriers also make normalized_output == right safe: every right operand is
// consumed before any normalized value is written.
__global__ __launch_bounds__(kResidualRmsM32Threads, 3)
void residual_add_headwise_centered_rms_norm_prefill_5120_kernel(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::uint16_t* const weight,
    const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output) {
  __shared__ std::uint16_t shared_residual[kResidualRmsHiddenSize];
  __shared__ float partial[kThreads];
  const std::size_t token = static_cast<std::size_t>(blockIdx.x);
  const std::size_t token_offset = token * kResidualRmsHiddenSize;
  constexpr unsigned int kValuesPerReductionLane =
      static_cast<unsigned int>(kResidualRmsHiddenSize / kThreads);
  constexpr unsigned int kPackedValuesPerReductionLane =
      kValuesPerReductionLane / 2U;
  static_assert(kValuesPerReductionLane == 20U);
  static_assert(kValuesPerReductionLane % 2U == 0U);
  std::uint32_t packed_residuals[kPackedValuesPerReductionLane];

  for (std::size_t dimension = threadIdx.x;
       dimension < kResidualRmsHiddenSize; dimension += blockDim.x) {
    const std::size_t index = token_offset + dimension;
    // Keep the production operand order: residual-left + projection-right.
    const std::uint16_t residual_bits =
        encode_bf16_device(decode_bf16_device(left[index]) +
                           decode_bf16_device(right[index]));
    residual_output[index] = residual_bits;
    shared_residual[dimension] = residual_bits;
  }

  __syncthreads();
  if (threadIdx.x < kThreads) {
    float sum = 0.0F;
#pragma unroll
    for (unsigned int packed_index = 0U;
         packed_index < kPackedValuesPerReductionLane; ++packed_index) {
      const std::size_t first_dimension =
          static_cast<std::size_t>(threadIdx.x) +
          static_cast<std::size_t>(2U * packed_index) * kThreads;
      const std::size_t second_dimension = first_dimension + kThreads;
      const std::uint16_t first_residual_bits =
          shared_residual[first_dimension];
      const float first_residual =
          decode_bf16_device(first_residual_bits);
      sum = fmaf(first_residual, first_residual, sum);
      const std::uint16_t second_residual_bits =
          shared_residual[second_dimension];
      const float second_residual =
          decode_bf16_device(second_residual_bits);
      sum = fmaf(second_residual, second_residual, sum);
      packed_residuals[packed_index] =
          static_cast<std::uint32_t>(first_residual_bits) |
          (static_cast<std::uint32_t>(second_residual_bits) << 16U);
    }
    partial[threadIdx.x] = sum;
  }
  __syncthreads();
  for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x < kThreads) {
    const float inverse_rms =
        rsqrtf(partial[0] / static_cast<float>(kResidualRmsHiddenSize) +
               epsilon);
#pragma unroll
    for (unsigned int packed_index = 0U;
         packed_index < kPackedValuesPerReductionLane; ++packed_index) {
      const std::size_t first_dimension =
          static_cast<std::size_t>(threadIdx.x) +
          static_cast<std::size_t>(2U * packed_index) * kThreads;
      const std::size_t second_dimension = first_dimension + kThreads;
      const std::size_t first_index = token_offset + first_dimension;
      const std::size_t second_index = token_offset + second_dimension;
      const std::uint32_t packed_residual =
          packed_residuals[packed_index];
      const std::uint16_t first_residual_bits =
          static_cast<std::uint16_t>(packed_residual);
      const std::uint16_t second_residual_bits =
          static_cast<std::uint16_t>(packed_residual >> 16U);
      const float first_gamma =
          decode_bf16_device(weight[first_dimension]) + 1.0F;
      const float second_gamma =
          decode_bf16_device(weight[second_dimension]) + 1.0F;
      normalized_output[first_index] = encode_bf16_device(
          decode_bf16_device(first_residual_bits) * inverse_rms *
          first_gamma);
      normalized_output[second_index] = encode_bf16_device(
          decode_bf16_device(second_residual_bits) * inverse_rms *
          second_gamma);
    }
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

// Exact fixed-shape Q24/KV4/D256 full-attention score kernel. Each warp owns
// one position. The scalar construction below mirrors
// attention_scores_kernel's shared-memory tree exactly:
//   128, 64, 32, 16, 8, 4, 2, 1.
__global__ void attention_scores_warp_positions_24_4_256_kernel(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const unsigned int sequence_length,
    const float attention_scale,
    float* const scores) {
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int position =
      blockIdx.y * kWarpPositionAttentionScoreWarpsPerBlock + warp;
  if (position >= sequence_length) {
    return;
  }

  const unsigned int query_head = blockIdx.x;
  const unsigned int kv_head = query_head / 6U;
  const unsigned int query_offset =
      query_head *
      static_cast<unsigned int>(kWarpPositionAttentionScoreHeadDimension);
  const unsigned int key_offset =
      (position *
           static_cast<unsigned int>(kWarpPositionAttentionScoreKvHeads) +
       kv_head) *
      static_cast<unsigned int>(kWarpPositionAttentionScoreHeadDimension);

  float sum = fmaf(decode_bf16_device(query[query_offset + lane]),
                   decode_bf16_device(key_cache[key_offset + lane]), 0.0F);
  float rhs =
      fmaf(decode_bf16_device(query[query_offset + lane + 128U]),
           decode_bf16_device(key_cache[key_offset + lane + 128U]), 0.0F);
  sum += rhs;

  float sibling =
      fmaf(decode_bf16_device(query[query_offset + lane + 64U]),
           decode_bf16_device(key_cache[key_offset + lane + 64U]), 0.0F);
  rhs = fmaf(decode_bf16_device(query[query_offset + lane + 192U]),
             decode_bf16_device(key_cache[key_offset + lane + 192U]), 0.0F);
  sibling += rhs;
  sum += sibling;

  sibling = fmaf(decode_bf16_device(query[query_offset + lane + 32U]),
                 decode_bf16_device(key_cache[key_offset + lane + 32U]),
                 0.0F);
  rhs = fmaf(decode_bf16_device(query[query_offset + lane + 160U]),
             decode_bf16_device(key_cache[key_offset + lane + 160U]), 0.0F);
  sibling += rhs;
  float upper =
      fmaf(decode_bf16_device(query[query_offset + lane + 96U]),
           decode_bf16_device(key_cache[key_offset + lane + 96U]), 0.0F);
  rhs = fmaf(decode_bf16_device(query[query_offset + lane + 224U]),
             decode_bf16_device(key_cache[key_offset + lane + 224U]), 0.0F);
  upper += rhs;
  sibling += upper;
  sum += sibling;

#pragma unroll
  for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
    rhs = __shfl_down_sync(0xffffffffU, sum, stride);
    if (lane < stride) {
      sum += rhs;
    }
  }
  if (lane == 0U) {
    scores[query_head * sequence_length + position] = sum * attention_scale;
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

// Exact Q24/KV4/D256 attention-value specialization. A two-dimensional 6x4
// grid exposes both the query-within-KV group and KV head directly, so the
// kernel needs no dynamic division or remainder. One thread retains one output
// dimension while traversing positions in exactly the reference FMA order.
__global__ void attention_values_exact_24_4_256_kernel(
    const std::uint16_t* const value_cache,
    const float* const probabilities,
    const unsigned int sequence_length,
    std::uint16_t* const output) {
  const unsigned int query_within_kv = blockIdx.x;
  const unsigned int kv_head = blockIdx.y;
  const unsigned int query_head =
      kv_head * kExactAttentionValueQueriesPerKv + query_within_kv;
  const unsigned int dimension = threadIdx.x;
  unsigned int probability_index = query_head * sequence_length;
  unsigned int value_index =
      kv_head * static_cast<unsigned int>(kExactAttentionValueHeadDimension) +
      dimension;

  float value = 0.0F;
#pragma unroll 4
  for (unsigned int position = 0U; position < sequence_length; ++position) {
    value = fmaf(probabilities[probability_index],
                 decode_bf16_device(value_cache[value_index]), value);
    ++probability_index;
    value_index += kExactAttentionValuePositionStride;
  }
  output[query_head *
             static_cast<unsigned int>(kExactAttentionValueHeadDimension) +
         dimension] = encode_bf16_device(value);
}

// Fixed-shape bulk causal GQA implementation. One CTA owns two adjacent query
// tokens and one KV head. Its six warps own that KV head's six query heads, so
// one shared K/V tile serves twelve query rows. Each lane retains eight head
// dimensions plus FP32 online-softmax state in registers. The final sigmoid
// gate deliberately observes a BF16-rounded attention value, preserving the
// production attention -> BF16 -> gate -> BF16 boundary.
__global__ __launch_bounds__(kBulkGqaThreads)
void bulk_causal_gqa_sigmoid_gate_24_4_256_kernel(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const unsigned int first_position,
    const unsigned int token_count,
    std::uint16_t* const output) {
  __shared__ std::uint32_t
      key_words[kBulkGqaKvTile][kBulkGqaPackedDimension];
  __shared__ std::uint32_t
      value_words[kBulkGqaKvTile][kBulkGqaPackedDimension];

  constexpr unsigned int kWordsPerLane =
      kBulkGqaPackedDimension / 32U;
  constexpr unsigned int kValuesPerLane = 2U * kWordsPerLane;
  constexpr unsigned int kWordsPerKvTile =
      kBulkGqaKvTile * kBulkGqaPackedDimension;
  static_assert(kWordsPerLane == 4U);
  static_assert(kValuesPerLane == 8U);

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread >> 5U;
  const unsigned int lane = thread & 31U;
  const unsigned int kv_head = blockIdx.y;
  const unsigned int query_head =
      kv_head * kBulkGqaQueriesPerKv + warp;
  const unsigned int first_query_token =
      blockIdx.x * kBulkGqaQueryTile;

  float query_values[kBulkGqaQueryTile][kValuesPerLane];
  float accumulators[kBulkGqaQueryTile][kValuesPerLane];
  float maxima[kBulkGqaQueryTile];
  float denominators[kBulkGqaQueryTile];
#pragma unroll
  for (unsigned int local_query = 0U;
       local_query < kBulkGqaQueryTile; ++local_query) {
    const unsigned int token = first_query_token + local_query;
    const bool valid_query = token < token_count;
    maxima[local_query] = -__int_as_float(0x7f800000);
    denominators[local_query] = 0.0F;
#pragma unroll
    for (unsigned int word_slot = 0U; word_slot < kWordsPerLane;
         ++word_slot) {
      const unsigned int value_slot = 2U * word_slot;
      std::uint32_t packed = 0U;
      if (valid_query) {
        const unsigned int word = lane + 32U * word_slot;
        const std::size_t packed_offset =
            (static_cast<std::size_t>(token) * kBulkGqaQueryHeads +
             query_head) *
                kBulkGqaPackedDimension +
            word;
        packed = reinterpret_cast<const std::uint32_t*>(query)[packed_offset];
      }
      query_values[local_query][value_slot] =
          decode_bf16_device(static_cast<std::uint16_t>(packed));
      query_values[local_query][value_slot + 1U] =
          decode_bf16_device(static_cast<std::uint16_t>(packed >> 16U));
      accumulators[local_query][value_slot] = 0.0F;
      accumulators[local_query][value_slot + 1U] = 0.0F;
    }
  }

  const unsigned int last_query_token =
      first_query_token + 1U < token_count
          ? first_query_token + 1U
          : first_query_token;
  const unsigned int causal_kv_length = first_position + last_query_token + 1U;
  for (unsigned int kv_tile_start = 0U;
       kv_tile_start < causal_kv_length;
       kv_tile_start += kBulkGqaKvTile) {
    for (unsigned int packed_index = thread;
         packed_index < 2U * kWordsPerKvTile;
         packed_index += kBulkGqaThreads) {
      const bool is_value = packed_index >= kWordsPerKvTile;
      const unsigned int tile_index =
          is_value ? packed_index - kWordsPerKvTile : packed_index;
      const unsigned int local_position =
          tile_index / kBulkGqaPackedDimension;
      const unsigned int word =
          tile_index - local_position * kBulkGqaPackedDimension;
      const unsigned int position = kv_tile_start + local_position;
      std::uint32_t packed = 0U;
      if (position < causal_kv_length) {
        const std::size_t cache_offset =
            (static_cast<std::size_t>(position) * kBulkGqaKvHeads +
             kv_head) *
                kBulkGqaPackedDimension +
            word;
        packed = is_value
                     ? reinterpret_cast<const std::uint32_t*>(value_cache)
                           [cache_offset]
                     : reinterpret_cast<const std::uint32_t*>(key_cache)
                           [cache_offset];
      }
      if (is_value) {
        value_words[local_position][word] = packed;
      } else {
        key_words[local_position][word] = packed;
      }
    }
    __syncthreads();

    const unsigned int remaining = causal_kv_length - kv_tile_start;
    const unsigned int active_positions =
        remaining < kBulkGqaKvTile ? remaining : kBulkGqaKvTile;
#pragma unroll 1
    for (unsigned int local_position = 0U;
         local_position < active_positions; ++local_position) {
      float key_values[kValuesPerLane];
      float value_values[kValuesPerLane];
#pragma unroll
      for (unsigned int word_slot = 0U; word_slot < kWordsPerLane;
           ++word_slot) {
        const unsigned int word = lane + 32U * word_slot;
        const unsigned int value_slot = 2U * word_slot;
        const std::uint32_t packed_key =
            key_words[local_position][word];
        const std::uint32_t packed_value =
            value_words[local_position][word];
        key_values[value_slot] =
            decode_bf16_device(static_cast<std::uint16_t>(packed_key));
        key_values[value_slot + 1U] = decode_bf16_device(
            static_cast<std::uint16_t>(packed_key >> 16U));
        value_values[value_slot] =
            decode_bf16_device(static_cast<std::uint16_t>(packed_value));
        value_values[value_slot + 1U] = decode_bf16_device(
            static_cast<std::uint16_t>(packed_value >> 16U));
      }
      const unsigned int position = kv_tile_start + local_position;
#pragma unroll
      for (unsigned int local_query = 0U;
           local_query < kBulkGqaQueryTile; ++local_query) {
        const unsigned int token = first_query_token + local_query;
        if (token >= token_count || position > first_position + token) {
          continue;
        }
        float score = 0.0F;
#pragma unroll
        for (unsigned int value_slot = 0U;
             value_slot < kValuesPerLane; ++value_slot) {
          score = fmaf(query_values[local_query][value_slot],
                       key_values[value_slot], score);
        }
#pragma unroll
        for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
          const float other =
              __shfl_down_sync(0xffff'ffffU, score, offset);
          if (lane < offset) {
            score += other;
          }
        }
        score = __shfl_sync(0xffff'ffffU, score, 0U) *
                kBulkGqaAttentionScale;

        if (score > maxima[local_query]) {
          const float correction =
              expf(maxima[local_query] - score);
          denominators[local_query] =
              denominators[local_query] * correction + 1.0F;
#pragma unroll
          for (unsigned int value_slot = 0U;
               value_slot < kValuesPerLane; ++value_slot) {
            accumulators[local_query][value_slot] =
                fmaf(accumulators[local_query][value_slot], correction,
                     value_values[value_slot]);
          }
          maxima[local_query] = score;
        } else {
          const float probability =
              expf(score - maxima[local_query]);
          denominators[local_query] += probability;
#pragma unroll
          for (unsigned int value_slot = 0U;
               value_slot < kValuesPerLane; ++value_slot) {
            accumulators[local_query][value_slot] =
                fmaf(probability, value_values[value_slot],
                     accumulators[local_query][value_slot]);
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (unsigned int local_query = 0U;
       local_query < kBulkGqaQueryTile; ++local_query) {
    const unsigned int token = first_query_token + local_query;
    if (token >= token_count) {
      continue;
    }
#pragma unroll
    for (unsigned int word_slot = 0U; word_slot < kWordsPerLane;
         ++word_slot) {
      const unsigned int word = lane + 32U * word_slot;
      const unsigned int value_slot = 2U * word_slot;
      const std::size_t packed_offset =
          (static_cast<std::size_t>(token) * kBulkGqaQueryHeads +
           query_head) *
              kBulkGqaPackedDimension +
          word;
      const std::uint32_t packed_gate =
          reinterpret_cast<const std::uint32_t*>(gate)[packed_offset];
      std::uint32_t packed_output = 0U;
#pragma unroll
      for (unsigned int pair_element = 0U; pair_element < 2U;
           ++pair_element) {
        const unsigned int slot = value_slot + pair_element;
        const std::uint16_t rounded_attention = encode_bf16_device(
            accumulators[local_query][slot] /
            denominators[local_query]);
        const float gate_value = decode_bf16_device(
            pair_element == 0U
                ? static_cast<std::uint16_t>(packed_gate)
                : static_cast<std::uint16_t>(packed_gate >> 16U));
        const float sigmoid =
            gate_value >= 0.0F
                ? 1.0F / (1.0F + expf(-gate_value))
                : expf(gate_value) / (1.0F + expf(gate_value));
        const std::uint16_t gated = encode_bf16_device(
            decode_bf16_device(rounded_attention) * sigmoid);
        packed_output |= static_cast<std::uint32_t>(gated)
                         << (16U * pair_element);
      }
      reinterpret_cast<std::uint32_t*>(output)[packed_offset] =
          packed_output;
    }
  }
}

// Test-only predecessor retained for direct production comparisons. Each CTA
// owns one query head, and every position uses the original block-wide shared
// reduction tree. The remaining softmax, value, BF16 boundary, probability-
// scratch, and sigmoid-gate phases match the production kernel below.
__global__ void
gqa_sigmoid_gate_shared_tree_predecessor_24_4_256_test_kernel(
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

// Production fixed-shape kernel. Each CTA owns one query head, while each warp
// owns one score position at a time. The scalar construction reproduces the
// predecessor's shared-memory reduction order exactly: 128, 64, 32, 16, 8, 4,
// 2, 1. The remaining softmax, value, BF16 boundary, probability-scratch, and
// sigmoid-gate phases are unchanged.
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

  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int lane = threadIdx.x & 31U;
  for (std::size_t position = warp; position < sequence_length;
       position += kWarpPositionAttentionScoreWarpsPerBlock) {
    const std::size_t key_offset =
        (position * kFusedGqaKvHeads + kv_head) * kFusedGqaHeadDimension;

    float sum = fmaf(decode_bf16_device(query[query_offset + lane]),
                     decode_bf16_device(key_cache[key_offset + lane]), 0.0F);
    float rhs =
        fmaf(decode_bf16_device(query[query_offset + lane + 128U]),
             decode_bf16_device(key_cache[key_offset + lane + 128U]), 0.0F);
    sum += rhs;

    float sibling =
        fmaf(decode_bf16_device(query[query_offset + lane + 64U]),
             decode_bf16_device(key_cache[key_offset + lane + 64U]), 0.0F);
    rhs = fmaf(decode_bf16_device(query[query_offset + lane + 192U]),
               decode_bf16_device(key_cache[key_offset + lane + 192U]),
               0.0F);
    sibling += rhs;
    sum += sibling;

    sibling =
        fmaf(decode_bf16_device(query[query_offset + lane + 32U]),
             decode_bf16_device(key_cache[key_offset + lane + 32U]), 0.0F);
    rhs = fmaf(decode_bf16_device(query[query_offset + lane + 160U]),
               decode_bf16_device(key_cache[key_offset + lane + 160U]),
               0.0F);
    sibling += rhs;
    float upper =
        fmaf(decode_bf16_device(query[query_offset + lane + 96U]),
             decode_bf16_device(key_cache[key_offset + lane + 96U]), 0.0F);
    rhs = fmaf(decode_bf16_device(query[query_offset + lane + 224U]),
               decode_bf16_device(key_cache[key_offset + lane + 224U]),
               0.0F);
    upper += rhs;
    sibling += upper;
    sum += sibling;

#pragma unroll
    for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
      rhs = __shfl_down_sync(0xffffffffU, sum, stride);
      if (lane < stride) {
        sum += rhs;
      }
    }
    if (lane == 0U) {
      probabilities[position] = sum * attention_scale;
    }
  }
  __syncthreads();

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

enum class AttentionScoreImplementation {
  kReference,
  kWarpPositions24_4_256,
};

[[nodiscard]] bool use_attention_scores_warp_positions_24_4_256(
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension) noexcept {
  return query_head_count == kWarpPositionAttentionScoreQueryHeads &&
         kv_head_count == kWarpPositionAttentionScoreKvHeads &&
         sequence_length > kFusedGqaMaximumSequenceLength &&
         sequence_length <= kWarpPositionAttentionScoreMaximumSequence &&
         head_dimension == kWarpPositionAttentionScoreHeadDimension;
}

[[nodiscard]] bool use_attention_values_exact_24_4_256(
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension) noexcept {
  return query_head_count == kExactAttentionValueQueryHeads &&
         kv_head_count == kExactAttentionValueKvHeads &&
         sequence_length != 0U &&
         sequence_length <= kExactAttentionValueMaximumSequence &&
         head_dimension == kExactAttentionValueHeadDimension;
}

void launch_attention_scores_unchecked(
    const AttentionScoreImplementation implementation,
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    const float attention_scale,
    float* const scores,
    const cudaStream_t stream) noexcept {
  if (implementation ==
      AttentionScoreImplementation::kWarpPositions24_4_256) {
    const std::size_t position_blocks =
        sequence_length / kWarpPositionAttentionScoreWarpsPerBlock +
        (sequence_length % kWarpPositionAttentionScoreWarpsPerBlock != 0U
             ? 1U
             : 0U);
    const dim3 blocks(
        static_cast<unsigned int>(kWarpPositionAttentionScoreQueryHeads),
        static_cast<unsigned int>(position_blocks), 1U);
    attention_scores_warp_positions_24_4_256_kernel
        <<<blocks, kThreads, 0U, stream>>>(
            query, key_cache, static_cast<unsigned int>(sequence_length),
            attention_scale, scores);
    return;
  }
  attention_scores_kernel<<<row_block_count(query_head_count), kThreads, 0U,
                            stream>>>(
      query, key_cache, query_head_count, kv_head_count, sequence_length,
      head_dimension, attention_scale, scores);
}

[[nodiscard]] bool valid_residual_rms_prefill_5120_arguments(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::uint16_t* const weight,
    const std::size_t token_count,
    const std::size_t hidden_size,
    const float epsilon,
    const std::uint16_t* const residual_output,
    const std::uint16_t* const normalized_output) noexcept {
  if (token_count == 0U ||
      token_count > kResidualRmsPrefillMaximumTokenCount ||
      hidden_size != kResidualRmsHiddenSize || !valid_epsilon(epsilon) ||
      left == nullptr || right == nullptr || weight == nullptr ||
      residual_output == nullptr || normalized_output == nullptr ||
      product3_overflows(token_count, hidden_size,
                         sizeof(std::uint16_t))) {
    return false;
  }
  const std::size_t bytes =
      token_count * kResidualRmsHiddenSize * sizeof(std::uint16_t);
  if (byte_range_overflows(left, bytes) ||
      byte_range_overflows(right, bytes) ||
      byte_range_overflows(weight, kResidualRmsBytes) ||
      byte_range_overflows(residual_output, bytes) ||
      byte_range_overflows(normalized_output, bytes)) {
    return false;
  }

  // residual_output is always independent. normalized_output may exactly
  // alias projection-right for the production layout, but no partial or
  // other writable overlap is legal.
  return !ranges_overlap(residual_output, bytes, left, bytes) &&
         !ranges_overlap(residual_output, bytes, right, bytes) &&
         !ranges_overlap(residual_output, bytes, weight,
                         kResidualRmsBytes) &&
         !ranges_overlap(residual_output, bytes, normalized_output, bytes) &&
         !ranges_overlap(normalized_output, bytes, left, bytes) &&
         !partially_overlaps(normalized_output, bytes, right, bytes) &&
         !ranges_overlap(normalized_output, bytes, weight,
                         kResidualRmsBytes);
}

[[nodiscard]] bool valid_residual_rms_m32_5120_arguments(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::uint16_t* const weight,
    const std::size_t token_count,
    const std::size_t hidden_size,
    const float epsilon,
    const std::uint16_t* const residual_output,
    const std::uint16_t* const normalized_output) noexcept {
  return token_count == kResidualRmsM32TokenCount &&
         valid_residual_rms_prefill_5120_arguments(
             left, right, weight, token_count, hidden_size, epsilon,
             residual_output, normalized_output);
}

[[nodiscard]] bool valid_attention_score_test_arguments(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    const float attention_scale,
    const float* const scores) noexcept {
  if (query_head_count != kWarpPositionAttentionScoreQueryHeads ||
      kv_head_count != kWarpPositionAttentionScoreKvHeads ||
      sequence_length == 0U ||
      sequence_length > kWarpPositionAttentionScoreMaximumSequence ||
      head_dimension != kWarpPositionAttentionScoreHeadDimension ||
      !std::isfinite(attention_scale) || attention_scale < 0.0F ||
      query == nullptr || key_cache == nullptr || scores == nullptr) {
    return false;
  }

  constexpr std::size_t kQueryBytes =
      kWarpPositionAttentionScoreQueryHeads *
      kWarpPositionAttentionScoreHeadDimension * sizeof(std::uint16_t);
  const std::size_t key_bytes =
      sequence_length * kWarpPositionAttentionScoreKvHeads *
      kWarpPositionAttentionScoreHeadDimension * sizeof(std::uint16_t);
  const std::size_t score_bytes =
      kWarpPositionAttentionScoreQueryHeads * sequence_length * sizeof(float);
  return !byte_range_overflows(query, kQueryBytes) &&
         !byte_range_overflows(key_cache, key_bytes) &&
         !byte_range_overflows(scores, score_bytes) &&
         !ranges_overlap(query, kQueryBytes, scores, score_bytes) &&
         !ranges_overlap(key_cache, key_bytes, scores, score_bytes);
}

[[nodiscard]] bool valid_attention_value_test_arguments(
    const std::uint16_t* const value_cache,
    const float* const probabilities,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    const std::uint16_t* const output) noexcept {
  if (query_head_count != kExactAttentionValueQueryHeads ||
      kv_head_count != kExactAttentionValueKvHeads || sequence_length == 0U ||
      sequence_length > kExactAttentionValueMaximumSequence ||
      head_dimension != kExactAttentionValueHeadDimension ||
      value_cache == nullptr || probabilities == nullptr || output == nullptr) {
    return false;
  }

  const std::size_t value_bytes =
      sequence_length * kExactAttentionValuePositionStride *
      sizeof(std::uint16_t);
  const std::size_t probability_bytes =
      kExactAttentionValueQueryHeads * sequence_length * sizeof(float);
  constexpr std::size_t kOutputBytes =
      kExactAttentionValueQueryHeads * kExactAttentionValueHeadDimension *
      sizeof(std::uint16_t);
  return !byte_range_overflows(value_cache, value_bytes) &&
         !byte_range_overflows(probabilities, probability_bytes) &&
         !byte_range_overflows(output, kOutputBytes) &&
         !ranges_overlap(value_cache, value_bytes, output, kOutputBytes) &&
         !ranges_overlap(probabilities, probability_bytes, output,
                         kOutputBytes);
}

[[nodiscard]] bool valid_bulk_causal_gqa_arguments(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    const std::uint16_t* const output) noexcept {
  if (token_count < kBulkGqaQueryTile || token_count > 512U ||
      first_position > kBulkGqaMaximumSequence - token_count ||
      query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      gate == nullptr || output == nullptr) {
    return false;
  }
  constexpr std::uintptr_t kPackedAlignment = alignof(std::uint32_t);
  if ((reinterpret_cast<std::uintptr_t>(query) % kPackedAlignment) != 0U ||
      (reinterpret_cast<std::uintptr_t>(key_cache) % kPackedAlignment) != 0U ||
      (reinterpret_cast<std::uintptr_t>(value_cache) % kPackedAlignment) !=
          0U ||
      (reinterpret_cast<std::uintptr_t>(gate) % kPackedAlignment) != 0U ||
      (reinterpret_cast<std::uintptr_t>(output) % kPackedAlignment) != 0U) {
    return false;
  }

  const std::size_t query_bytes =
      token_count * kBulkGqaQueryHeads * kBulkGqaHeadDimension *
      sizeof(std::uint16_t);
  const std::size_t cache_bytes =
      (first_position + token_count) * kBulkGqaKvHeads *
      kBulkGqaHeadDimension * sizeof(std::uint16_t);
  if (byte_range_overflows(query, query_bytes) ||
      byte_range_overflows(key_cache, cache_bytes) ||
      byte_range_overflows(value_cache, cache_bytes) ||
      byte_range_overflows(gate, query_bytes) ||
      byte_range_overflows(output, query_bytes)) {
    return false;
  }

  return !ranges_overlap(query, query_bytes, key_cache, cache_bytes) &&
         !ranges_overlap(query, query_bytes, value_cache, cache_bytes) &&
         !ranges_overlap(query, query_bytes, gate, query_bytes) &&
         !ranges_overlap(query, query_bytes, output, query_bytes) &&
         !ranges_overlap(key_cache, cache_bytes, value_cache, cache_bytes) &&
         !ranges_overlap(key_cache, cache_bytes, gate, query_bytes) &&
         !ranges_overlap(key_cache, cache_bytes, output, query_bytes) &&
         !ranges_overlap(value_cache, cache_bytes, gate, query_bytes) &&
         !ranges_overlap(value_cache, cache_bytes, output, query_bytes) &&
         !ranges_overlap(gate, query_bytes, output, query_bytes);
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

int launch_residual_add_headwise_centered_rms_norm_m32_5120_cuda(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::uint16_t* const weight,
    const std::size_t token_count,
    const std::size_t hidden_size,
    const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output,
    void* const cuda_stream) noexcept {
  if (!valid_residual_rms_m32_5120_arguments(
          left, right, weight, token_count, hidden_size, epsilon,
          residual_output, normalized_output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  residual_add_headwise_centered_rms_norm_prefill_5120_kernel
      <<<static_cast<unsigned int>(kResidualRmsM32TokenCount),
         kResidualRmsM32Threads, 0U, stream>>>(
          left, right, weight, epsilon, residual_output, normalized_output);
  return static_cast<int>(cudaGetLastError());
}

int launch_residual_add_headwise_centered_rms_norm_prefill_5120_cuda(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::uint16_t* const weight,
    const std::size_t token_count,
    const std::size_t hidden_size,
    const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output,
    void* const cuda_stream) noexcept {
  if (!valid_residual_rms_prefill_5120_arguments(
          left, right, weight, token_count, hidden_size, epsilon,
          residual_output, normalized_output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  residual_add_headwise_centered_rms_norm_prefill_5120_kernel
      <<<static_cast<unsigned int>(token_count), kResidualRmsM32Threads, 0U,
         stream>>>(left, right, weight, epsilon, residual_output,
                   normalized_output);
  return static_cast<int>(cudaGetLastError());
}

int query_residual_add_headwise_centered_rms_norm_m32_5120_test_cuda_resources(
    int* const registers,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks_per_multiprocessor) noexcept {
  if (registers == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads == nullptr ||
      active_blocks_per_multiprocessor == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      residual_add_headwise_centered_rms_norm_prefill_5120_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      residual_add_headwise_centered_rms_norm_prefill_5120_kernel,
      static_cast<int>(kResidualRmsM32Threads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks_per_multiprocessor = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_attention_scores_baseline_24_4_256_test_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    const float attention_scale,
    float* const scores,
    void* const cuda_stream) noexcept {
  if (!valid_attention_score_test_arguments(
          query, key_cache, query_head_count, kv_head_count, sequence_length,
          head_dimension, attention_scale, scores)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_attention_scores_unchecked(
      AttentionScoreImplementation::kReference, query, key_cache,
      query_head_count, kv_head_count, sequence_length, head_dimension,
      attention_scale, scores, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_attention_scores_warp_positions_24_4_256_test_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    const float attention_scale,
    float* const scores,
    void* const cuda_stream) noexcept {
  if (!valid_attention_score_test_arguments(
          query, key_cache, query_head_count, kv_head_count, sequence_length,
          head_dimension, attention_scale, scores)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_attention_scores_unchecked(
      AttentionScoreImplementation::kWarpPositions24_4_256, query, key_cache,
      query_head_count, kv_head_count, sequence_length, head_dimension,
      attention_scale, scores, stream);
  return static_cast<int>(cudaGetLastError());
}

bool use_attention_scores_warp_positions_24_4_256_test(
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension) noexcept {
  return use_attention_scores_warp_positions_24_4_256(
      query_head_count, kv_head_count, sequence_length, head_dimension);
}

int query_attention_scores_baseline_24_4_256_test_cuda_resources(
    int* const registers,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks_per_multiprocessor) noexcept {
  if (registers == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads == nullptr ||
      active_blocks_per_multiprocessor == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status =
      cudaFuncGetAttributes(&attributes, attention_scores_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, attention_scores_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks_per_multiprocessor = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int query_attention_scores_warp_positions_24_4_256_test_cuda_resources(
    int* const registers,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks_per_multiprocessor) noexcept {
  if (registers == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads == nullptr ||
      active_blocks_per_multiprocessor == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, attention_scores_warp_positions_24_4_256_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, attention_scores_warp_positions_24_4_256_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks_per_multiprocessor = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_gqa_attention_sigmoid_gate_warp_positions_24_4_256_test_cuda(
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

int launch_gqa_attention_sigmoid_gate_shared_tree_predecessor_24_4_256_test_cuda(
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
  gqa_sigmoid_gate_shared_tree_predecessor_24_4_256_test_kernel
      <<<kFusedGqaQueryHeads, kThreads, 0U, stream>>>(
          query, key_cache, value_cache, sequence_length, attention_scale,
          probabilities_scratch, gate, output);
  return static_cast<int>(cudaGetLastError());
}

int query_gqa_attention_sigmoid_gate_24_4_256_resources_test_cuda(
    int* const registers,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks_per_multiprocessor) noexcept {
  if (registers == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads == nullptr ||
      active_blocks_per_multiprocessor == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, gqa_sigmoid_gate_24_4_256_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, gqa_sigmoid_gate_24_4_256_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks_per_multiprocessor = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int query_gqa_attention_sigmoid_gate_warp_positions_24_4_256_resources_test_cuda(
    int* const registers,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks_per_multiprocessor) noexcept {
  if (registers == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads == nullptr ||
      active_blocks_per_multiprocessor == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, gqa_sigmoid_gate_24_4_256_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, gqa_sigmoid_gate_24_4_256_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks_per_multiprocessor = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int query_gqa_attention_sigmoid_gate_shared_tree_predecessor_24_4_256_resources_test_cuda(
    int* const registers,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks_per_multiprocessor) noexcept {
  if (registers == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads == nullptr ||
      active_blocks_per_multiprocessor == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      gqa_sigmoid_gate_shared_tree_predecessor_24_4_256_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      gqa_sigmoid_gate_shared_tree_predecessor_24_4_256_test_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks_per_multiprocessor = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_attention_values_baseline_24_4_256_test_cuda(
    const std::uint16_t* const value_cache,
    const float* const probabilities,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!valid_attention_value_test_arguments(
          value_cache, probabilities, query_head_count, kv_head_count,
          sequence_length, head_dimension, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  attention_values_kernel<<<row_block_count(query_head_count), kThreads, 0U,
                            stream>>>(
      value_cache, probabilities, query_head_count, kv_head_count,
      sequence_length, head_dimension, output);
  return static_cast<int>(cudaGetLastError());
}

int query_attention_values_exact_24_4_256_test_cuda_selection(
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    int* const selected) noexcept {
  if (selected == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *selected = use_attention_values_exact_24_4_256(
                  query_head_count, kv_head_count, sequence_length,
                  head_dimension)
                  ? 1
                  : 0;
  return static_cast<int>(cudaSuccess);
}

int launch_attention_values_exact_24_4_256_test_cuda(
    const std::uint16_t* const value_cache,
    const float* const probabilities,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!valid_attention_value_test_arguments(
          value_cache, probabilities, query_head_count, kv_head_count,
          sequence_length, head_dimension, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  const dim3 blocks(kExactAttentionValueQueriesPerKv,
                    static_cast<unsigned int>(kExactAttentionValueKvHeads),
                    1U);
  (void)cudaGetLastError();
  attention_values_exact_24_4_256_kernel<<<blocks, kThreads, 0U, stream>>>(
      value_cache, probabilities, static_cast<unsigned int>(sequence_length),
      output);
  return static_cast<int>(cudaGetLastError());
}

int query_attention_values_baseline_24_4_256_test_cuda_resources(
    int* const registers,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks_per_multiprocessor) noexcept {
  if (registers == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads == nullptr ||
      active_blocks_per_multiprocessor == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status =
      cudaFuncGetAttributes(&attributes, attention_values_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, attention_values_kernel, static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks_per_multiprocessor = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int query_attention_values_exact_24_4_256_test_cuda_resources(
    int* const registers,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks_per_multiprocessor) noexcept {
  if (registers == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads == nullptr ||
      active_blocks_per_multiprocessor == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, attention_values_exact_24_4_256_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, attention_values_exact_24_4_256_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks_per_multiprocessor = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!valid_bulk_causal_gqa_arguments(query, key_cache, value_cache, gate,
                                       first_position, token_count, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const dim3 blocks(
      static_cast<unsigned int>((token_count + kBulkGqaQueryTile - 1U) /
                                kBulkGqaQueryTile),
      kBulkGqaKvHeads, 1U);
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  if (use_bulk_causal_gqa_group_q64_prefill(first_position, token_count)) {
    return launch_bulk_causal_gqa_sigmoid_gate_24_4_256_c512_register_pipeline_cuda(
        query, key_cache, value_cache, gate, first_position, token_count,
        output, cuda_stream);
  }
  bulk_causal_gqa_sigmoid_gate_24_4_256_kernel
      <<<blocks, kBulkGqaThreads, 0U, stream>>>(
          query, key_cache, value_cache, gate,
          static_cast<unsigned int>(first_position),
          static_cast<unsigned int>(token_count), output);
  return static_cast<int>(cudaGetLastError());
}

int launch_bulk_causal_gqa_sigmoid_24_4_256_qt2_bk16_test_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    const float attention_scale,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (attention_scale != kBulkGqaAttentionScale) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
      query, key_cache, value_cache, gate, first_position, token_count,
      output, cuda_stream);
}

int launch_bulk_causal_gqa_sigmoid_24_4_256_qt2_bk16_generic_test_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    const float attention_scale,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (attention_scale != kBulkGqaAttentionScale ||
      !valid_bulk_causal_gqa_arguments(query, key_cache, value_cache, gate,
                                       first_position, token_count, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const dim3 blocks(
      static_cast<unsigned int>((token_count + kBulkGqaQueryTile - 1U) /
                                kBulkGqaQueryTile),
      kBulkGqaKvHeads, 1U);
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  bulk_causal_gqa_sigmoid_gate_24_4_256_kernel
      <<<blocks, kBulkGqaThreads, 0U, stream>>>(
          query, key_cache, value_cache, gate,
          static_cast<unsigned int>(first_position),
          static_cast<unsigned int>(token_count), output);
  return static_cast<int>(cudaGetLastError());
}

int query_bulk_causal_gqa_sigmoid_24_4_256_qt2_bk16_resources_test_cuda(
    int* const registers,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks_per_multiprocessor) noexcept {
  if (registers == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads == nullptr ||
      active_blocks_per_multiprocessor == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      bulk_causal_gqa_sigmoid_gate_24_4_256_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      bulk_causal_gqa_sigmoid_gate_24_4_256_kernel,
      static_cast<int>(kBulkGqaThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks_per_multiprocessor = active_blocks;
  return static_cast<int>(cudaSuccess);
}

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

int launch_embedding_gather_prompt_reference_cuda(
    const std::uint16_t* const embedding_table,
    const std::size_t vocabulary_size,
    const std::size_t hidden_size,
    const std::uint32_t* const token_ids,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (vocabulary_size == 0U ||
      multiply_overflows(vocabulary_size, hidden_size) ||
      multiply_overflows(token_count, hidden_size) ||
      token_count == 0U || token_count >
                               static_cast<std::size_t>(
                                   std::numeric_limits<unsigned int>::max()) ||
      embedding_table == nullptr || token_ids == nullptr || output == nullptr ||
      hidden_size == 0U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t table_elements = vocabulary_size * hidden_size;
  const std::size_t output_elements = token_count * hidden_size;
  if (multiply_overflows(table_elements, sizeof(std::uint16_t)) ||
      multiply_overflows(token_count, sizeof(std::uint32_t)) ||
      multiply_overflows(output_elements, sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t table_bytes =
      table_elements * sizeof(std::uint16_t);
  const std::size_t token_id_bytes =
      token_count * sizeof(std::uint32_t);
  const std::size_t output_bytes =
      output_elements * sizeof(std::uint16_t);
  if (byte_range_overflows(embedding_table, table_bytes) ||
      byte_range_overflows(token_ids, token_id_bytes) ||
      byte_range_overflows(output, output_bytes) ||
      ranges_overlap(embedding_table, table_bytes, token_ids,
                     token_id_bytes) ||
      ranges_overlap(embedding_table, table_bytes, output, output_bytes) ||
      ranges_overlap(token_ids, token_id_bytes, output, output_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  embedding_gather_prompt_kernel<<<
      static_cast<unsigned int>(token_count), kThreads, 0U, stream>>>(
      embedding_table, vocabulary_size, hidden_size, token_ids, output);
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

int launch_full_attention_preprocess_warp_rms_24_4_256_64_test_cuda(
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
  full_attention_preprocess_warp_rms_24_4_256_64_test_kernel
      <<<static_cast<unsigned int>(token_count * kCombinedHeads), kThreads,
         0U, stream>>>(interleaved_q_gate, key, q_weight, k_weight, epsilon,
                       query_output, gate_output, cosines, sines,
                       first_position);
  return static_cast<int>(cudaGetLastError());
}

int query_full_attention_preprocess_24_4_256_64_resources_test_cuda(
    int* const registers,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks_per_multiprocessor) noexcept {
  if (registers == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads == nullptr ||
      active_blocks_per_multiprocessor == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, full_attention_preprocess_24_4_256_64_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, full_attention_preprocess_24_4_256_64_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks_per_multiprocessor = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int query_full_attention_preprocess_warp_rms_24_4_256_64_resources_test_cuda(
    int* const registers,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks_per_multiprocessor) noexcept {
  if (registers == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads == nullptr ||
      active_blocks_per_multiprocessor == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      full_attention_preprocess_warp_rms_24_4_256_64_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      full_attention_preprocess_warp_rms_24_4_256_64_test_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks_per_multiprocessor = active_blocks;
  return static_cast<int>(cudaSuccess);
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
  const AttentionScoreImplementation score_implementation =
      use_attention_scores_warp_positions_24_4_256(
          query_head_count, kv_head_count, sequence_length, head_dimension)
          ? AttentionScoreImplementation::kWarpPositions24_4_256
          : AttentionScoreImplementation::kReference;
  launch_attention_scores_unchecked(
      score_implementation, query, key_cache, query_head_count, kv_head_count,
      sequence_length, head_dimension, attention_scale,
      probabilities_scratch, stream);
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
  if (use_attention_values_exact_24_4_256(
          query_head_count, kv_head_count, sequence_length,
          head_dimension)) {
    const dim3 value_blocks(
        kExactAttentionValueQueriesPerKv,
        static_cast<unsigned int>(kExactAttentionValueKvHeads), 1U);
    attention_values_exact_24_4_256_kernel
        <<<value_blocks, kThreads, 0U, stream>>>(
            value_cache, probabilities_scratch,
            static_cast<unsigned int>(sequence_length), output);
  } else {
    attention_values_kernel<<<blocks, kThreads, 0U, stream>>>(
        value_cache, probabilities_scratch, query_head_count, kv_head_count,
        sequence_length, head_dimension, output);
  }
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
