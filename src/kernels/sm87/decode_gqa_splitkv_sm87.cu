#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

constexpr unsigned int kQueryHeads = 24U;
constexpr unsigned int kKvHeads = 4U;
constexpr unsigned int kQueriesPerKv = kQueryHeads / kKvHeads;
constexpr unsigned int kHeadDimension = 256U;
constexpr unsigned int kWarpsPerKvCta = kQueriesPerKv;
constexpr unsigned int kSplitThreads = kWarpsPerKvCta * 32U;
constexpr unsigned int kMergeThreads = kHeadDimension;
constexpr unsigned int kFourSplits = 4U;
constexpr unsigned int kEightSplits = 8U;
constexpr unsigned int kEightSplitThreshold = 512U;
constexpr unsigned int kStateMaximumOffset = 0U;
constexpr unsigned int kStateDenominatorOffset = 1U;
constexpr unsigned int kStateValueOffset = 2U;
constexpr unsigned int kStateElements =
    static_cast<unsigned int>(kDecodeGqaSplitKvStateElements);

static_assert(kQueriesPerKv == 6U);
static_assert(kSplitThreads == 192U);
static_assert(kMergeThreads == 256U);
static_assert(kStateElements == kHeadDimension + 2U);
static_assert(kDecodeGqaSplitKvMaximumSplits == kEightSplits);
static_assert(kDecodeGqaSplitKvMaximumWorkspaceElements ==
              kQueryHeads * kEightSplits * kStateElements);

[[nodiscard]] constexpr unsigned int split_count_for_sequence(
    const std::size_t sequence_length) noexcept {
  return sequence_length <= kEightSplitThreshold ? kFourSplits
                                                  : kEightSplits;
}

__device__ __forceinline__ float decode_bf16(const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16(const float value) {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

// A CTA owns one contiguous sequence split and one KV head.  Its six warps
// consume the six Q heads in that GQA group.  K and V are decoded once into
// shared memory per cache row; Q and the FP32 value numerator stay resident in
// registers for the entire split.
__global__ __launch_bounds__(kSplitThreads, 2)
void decode_gqa_splitkv_state_24_4_256_kernel(
    const std::uint16_t* __restrict__ query,
    const std::uint16_t* __restrict__ key_cache,
    const std::uint16_t* __restrict__ value_cache,
    const unsigned int sequence_length,
    const float attention_scale,
    const unsigned int split_count,
    float* __restrict__ states) {
  __shared__ float key_tile[kHeadDimension];
  __shared__ float value_tile[kHeadDimension];

  const unsigned int task = blockIdx.x;
  const unsigned int kv_head = task / split_count;
  const unsigned int split = task - kv_head * split_count;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int query_head = kv_head * kQueriesPerKv + warp;
  const unsigned int query_offset = query_head * kHeadDimension;

  float q[8];
#pragma unroll
  for (unsigned int index = 0U; index < 8U; ++index) {
    q[index] = decode_bf16(query[query_offset + lane + index * 32U]);
  }

  float value_numerator[8] = {0.0F, 0.0F, 0.0F, 0.0F,
                              0.0F, 0.0F, 0.0F, 0.0F};
  float maximum = -__int_as_float(0x7f80'0000U);
  float denominator = 0.0F;

  const unsigned int positions_per_split =
      (sequence_length + split_count - 1U) / split_count;
  const unsigned int begin = split * positions_per_split;
  const unsigned int unclamped_end = begin + positions_per_split;
  const unsigned int end =
      unclamped_end < sequence_length ? unclamped_end : sequence_length;

  for (unsigned int position = begin; position < end; ++position) {
    // 128 threads issue naturally aligned 32-bit cache loads.  Decoding in
    // the loader warps avoids repeating BF16 conversion in all six Q warps.
    if (threadIdx.x < kHeadDimension / 2U) {
      const unsigned int pair = threadIdx.x;
      const unsigned int cache_pair =
          (position * kKvHeads + kv_head) * (kHeadDimension / 2U) + pair;
      const std::uint32_t packed_key =
          reinterpret_cast<const std::uint32_t*>(key_cache)[cache_pair];
      const std::uint32_t packed_value =
          reinterpret_cast<const std::uint32_t*>(value_cache)[cache_pair];
      key_tile[pair * 2U] =
          decode_bf16(static_cast<std::uint16_t>(packed_key));
      key_tile[pair * 2U + 1U] =
          decode_bf16(static_cast<std::uint16_t>(packed_key >> 16U));
      value_tile[pair * 2U] =
          decode_bf16(static_cast<std::uint16_t>(packed_value));
      value_tile[pair * 2U + 1U] =
          decode_bf16(static_cast<std::uint16_t>(packed_value >> 16U));
    }
    __syncthreads();

    // This construction matches the existing fixed-shape score kernel's
    // 256-lane reduction tree: 128,64,32,16,8,4,2,1.
    float score = fmaf(q[0], key_tile[lane], 0.0F);
    float rhs = fmaf(q[4], key_tile[lane + 128U], 0.0F);
    score += rhs;

    float sibling = fmaf(q[2], key_tile[lane + 64U], 0.0F);
    rhs = fmaf(q[6], key_tile[lane + 192U], 0.0F);
    sibling += rhs;
    score += sibling;

    sibling = fmaf(q[1], key_tile[lane + 32U], 0.0F);
    rhs = fmaf(q[5], key_tile[lane + 160U], 0.0F);
    sibling += rhs;
    float upper = fmaf(q[3], key_tile[lane + 96U], 0.0F);
    rhs = fmaf(q[7], key_tile[lane + 224U], 0.0F);
    upper += rhs;
    sibling += upper;
    score += sibling;

#pragma unroll
    for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
      rhs = __shfl_down_sync(0xffff'ffffU, score, stride);
      if (lane < stride) {
        score += rhs;
      }
    }
    score = __shfl_sync(0xffff'ffffU, score, 0U) * attention_scale;

    const float next_maximum = fmaxf(maximum, score);
    const float previous_scale =
        denominator == 0.0F ? 0.0F : expf(maximum - next_maximum);
    const float current_scale = expf(score - next_maximum);
    denominator = denominator * previous_scale + current_scale;
#pragma unroll
    for (unsigned int index = 0U; index < 8U; ++index) {
      value_numerator[index] =
          fmaf(current_scale, value_tile[lane + index * 32U],
               value_numerator[index] * previous_scale);
    }
    maximum = next_maximum;

    // The shared cache row may be overwritten only after all six Q warps have
    // consumed both K and V.
    __syncthreads();
  }

  const std::size_t state_offset =
      (static_cast<std::size_t>(query_head) * split_count + split) *
      kStateElements;
  if (lane == 0U) {
    states[state_offset + kStateMaximumOffset] = maximum;
    states[state_offset + kStateDenominatorOffset] = denominator;
  }
#pragma unroll
  for (unsigned int index = 0U; index < 8U; ++index) {
    states[state_offset + kStateValueOffset + lane + index * 32U] =
        value_numerator[index];
  }
}

// One CTA merges all sequence splits for one Q head.  Split numerators are
// rescaled to the global maximum before summation.  The attention result is
// explicitly rounded to BF16 before the sigmoid gate observes it.
__global__ __launch_bounds__(kMergeThreads, 2)
void decode_gqa_splitkv_merge_gate_24_4_256_kernel(
    const float* __restrict__ states,
    const unsigned int split_count,
    const std::uint16_t* __restrict__ gate,
    std::uint16_t* __restrict__ output) {
  __shared__ float split_scales[kEightSplits];
  __shared__ float inverse_denominator;

  const unsigned int query_head = blockIdx.x;
  const unsigned int dimension = threadIdx.x;
  const std::size_t head_state_offset =
      static_cast<std::size_t>(query_head) * split_count * kStateElements;

  if (dimension == 0U) {
    float maximum = -__int_as_float(0x7f80'0000U);
#pragma unroll
    for (unsigned int split = 0U; split < kEightSplits; ++split) {
      if (split < split_count) {
        maximum = fmaxf(
            maximum,
            states[head_state_offset + split * kStateElements +
                   kStateMaximumOffset]);
      }
    }

    float denominator = 0.0F;
#pragma unroll
    for (unsigned int split = 0U; split < kEightSplits; ++split) {
      if (split < split_count) {
        const std::size_t offset =
            head_state_offset + split * kStateElements;
        const float scale =
            expf(states[offset + kStateMaximumOffset] - maximum);
        split_scales[split] = scale;
        denominator = fmaf(scale,
                           states[offset + kStateDenominatorOffset],
                           denominator);
      }
    }
    inverse_denominator = 1.0F / denominator;
  }
  __syncthreads();

  float numerator = 0.0F;
#pragma unroll
  for (unsigned int split = 0U; split < kEightSplits; ++split) {
    if (split < split_count) {
      const std::size_t offset =
          head_state_offset + split * kStateElements + kStateValueOffset +
          dimension;
      numerator = fmaf(split_scales[split], states[offset], numerator);
    }
  }

  const unsigned int output_index = query_head * kHeadDimension + dimension;
  const std::uint16_t rounded_attention =
      encode_bf16(numerator * inverse_denominator);
  const float gate_value = decode_bf16(gate[output_index]);
  const float sigmoid =
      gate_value >= 0.0F
          ? 1.0F / (1.0F + expf(-gate_value))
          : expf(gate_value) / (1.0F + expf(gate_value));
  output[output_index] =
      encode_bf16(decode_bf16(rounded_attention) * sigmoid);
}

}  // namespace

std::size_t gqa_attention_splitkv_sigmoid_gate_24_4_256_workspace_elements(
    const std::size_t sequence_length) noexcept {
  if (sequence_length <= kFusedGqaMaximumSequenceLength ||
      sequence_length > kDecodeGqaSplitKvMaximumSequenceLength) {
    return 0U;
  }
  return static_cast<std::size_t>(kQueryHeads) *
         split_count_for_sequence(sequence_length) * kStateElements;
}

int launch_gqa_attention_splitkv_sigmoid_gate_24_4_256_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::size_t sequence_length,
    const float attention_scale,
    float* const workspace,
    const std::size_t workspace_elements,
    const std::uint16_t* const gate,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const std::size_t required_workspace =
      gqa_attention_splitkv_sigmoid_gate_24_4_256_workspace_elements(
          sequence_length);
  if (required_workspace == 0U || workspace_elements < required_workspace ||
      !std::isfinite(attention_scale) || attention_scale < 0.0F ||
      query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      workspace == nullptr || gate == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const unsigned int split_count = split_count_for_sequence(sequence_length);
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  decode_gqa_splitkv_state_24_4_256_kernel
      <<<kKvHeads * split_count, kSplitThreads, 0U, stream>>>(
          query, key_cache, value_cache,
          static_cast<unsigned int>(sequence_length), attention_scale,
          split_count, workspace);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  decode_gqa_splitkv_merge_gate_24_4_256_kernel
      <<<kQueryHeads, kMergeThreads, 0U, stream>>>(
          workspace, split_count, gate, output);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace q3x::runtime
