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
constexpr unsigned int kEightSplits = 8U;
constexpr unsigned int kPositionsPerTile = kQueriesPerKv;
constexpr unsigned int kPipelineStages = 2U;
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
static_assert(decode_gqa_splitkv_split_count(
                  kDecodeGqaSplitKvFourSplitMaximumSequenceLength) == 4U);
static_assert(decode_gqa_splitkv_split_count(
                  kDecodeGqaSplitKvFourSplitMaximumSequenceLength + 1U) ==
              kEightSplits);
static_assert(kDecodeGqaSplitKvMaximumWorkspaceElements ==
              kQueryHeads * kEightSplits * kStateElements);

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

__device__ __forceinline__ void cp_async_cg_16_zero_fill(
    void* const shared_destination,
    const void* const global_source,
    const bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));
  const int source_bytes = valid ? 16 : 0;
  asm volatile(
      "cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" :
      : "r"(shared_address), "l"(global_source), "r"(source_bytes)
      : "memory");
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      valid ? *reinterpret_cast<const uint4*>(global_source)
            : make_uint4(0U, 0U, 0U, 0U);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_3() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 3;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

struct __align__(16) DecodeGqaPipelineStorage {
  std::uint16_t key[kPipelineStages][kPositionsPerTile][kHeadDimension];
  std::uint16_t value[kPipelineStages][kPositionsPerTile][kHeadDimension];
};

static_assert(sizeof(DecodeGqaPipelineStorage) == 12U * 1024U);

__device__ __forceinline__ void enqueue_cache_tile(
    std::uint16_t* const shared_tile,
    const std::uint16_t* const cache,
    const unsigned int tile_position,
    const unsigned int chunk_begin,
    const unsigned int chunk_end,
    const unsigned int kv_head,
    const unsigned int warp,
    const unsigned int lane) {
  const unsigned int position = tile_position + warp;
  const bool valid = position < chunk_end;
  const unsigned int safe_position = valid ? position : chunk_begin;
  const std::size_t global_offset =
      (static_cast<std::size_t>(safe_position) * kKvHeads + kv_head) *
          kHeadDimension +
      lane * 8U;
  cp_async_cg_16_zero_fill(
      shared_tile + (warp * kHeadDimension + lane * 8U),
      cache + global_offset, valid);
}

// A CTA owns one contiguous KV chunk and one KV head.  Its six warps are the
// six Q heads in that GQA group.  In each iteration the warps cooperatively
// load six different cache positions, then every Q warp consumes all six rows.
// A two-stage K/V cp.async schedule overlaps the next tile with online-state
// arithmetic and amortizes synchronization across six positions.
template <bool kDynamicGraph>
__global__ __launch_bounds__(kSplitThreads, 2)
void decode_gqa_splitkv_state_24_4_256_kernel(
    const std::uint16_t* __restrict__ query,
    const std::uint16_t* __restrict__ key_cache,
    const std::uint16_t* __restrict__ value_cache,
    const unsigned int captured_sequence_length,
    const float attention_scale,
    const unsigned int captured_split_count,
    float* __restrict__ states,
    const DecodeDynamicGraphParameters* const parameters) {
  __shared__ DecodeGqaPipelineStorage storage;

  const unsigned int sequence_length =
      kDynamicGraph ? parameters->sequence_length : captured_sequence_length;
  const unsigned int split_count =
      kDynamicGraph ? parameters->split_count : captured_split_count;

  const unsigned int task = blockIdx.x;
  if constexpr (kDynamicGraph) {
    if (task >= kKvHeads * split_count) {
      return;
    }
  }
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

  // Prime K0,V0,K1,V1 as four independent async groups.  The steady-state
  // waits below expose K and V independently, allowing their shared slots to
  // be refilled as soon as their corresponding arithmetic phase completes.
  enqueue_cache_tile(storage.key[0][0], key_cache, begin, begin, end,
                     kv_head, warp, lane);
  cp_async_commit();
  enqueue_cache_tile(storage.value[0][0], value_cache, begin, begin, end,
                     kv_head, warp, lane);
  cp_async_commit();
  enqueue_cache_tile(storage.key[1][0], key_cache,
                     begin + kPositionsPerTile, begin, end, kv_head, warp,
                     lane);
  cp_async_commit();
  enqueue_cache_tile(storage.value[1][0], value_cache,
                     begin + kPositionsPerTile, begin, end, kv_head, warp,
                     lane);
  cp_async_commit();

  const unsigned int tile_count =
      (end - begin + kPositionsPerTile - 1U) / kPositionsPerTile;
  for (unsigned int tile = 0U; tile < tile_count; ++tile) {
    const unsigned int stage = tile & 1U;
    const unsigned int tile_position = begin + tile * kPositionsPerTile;
    float scores[kPositionsPerTile];

    cp_async_wait_group_3();
    __syncthreads();
#pragma unroll
    for (unsigned int row = 0U; row < kPositionsPerTile; ++row) {
      const std::uint16_t* const key_row = storage.key[stage][row];

      // Preserve the production 256-lane score tree exactly: scatter each
      // lane over dimensions {lane + i*32}, construct the 128/64/32 partials,
      // then reduce 16,8,4,2,1 with lane-conditional additions.
      float score = fmaf(q[0], decode_bf16(key_row[lane]), 0.0F);
      float rhs =
          fmaf(q[4], decode_bf16(key_row[lane + 128U]), 0.0F);
      score += rhs;

      float sibling =
          fmaf(q[2], decode_bf16(key_row[lane + 64U]), 0.0F);
      rhs = fmaf(q[6], decode_bf16(key_row[lane + 192U]), 0.0F);
      sibling += rhs;
      score += sibling;

      sibling = fmaf(q[1], decode_bf16(key_row[lane + 32U]), 0.0F);
      rhs = fmaf(q[5], decode_bf16(key_row[lane + 160U]), 0.0F);
      sibling += rhs;
      float upper =
          fmaf(q[3], decode_bf16(key_row[lane + 96U]), 0.0F);
      rhs = fmaf(q[7], decode_bf16(key_row[lane + 224U]), 0.0F);
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
      const bool valid = tile_position + row < end;
      scores[row] = valid ? score : -__int_as_float(0x7f80'0000U);
    }

    __syncthreads();
    const unsigned int future_position =
        begin + (tile + kPipelineStages) * kPositionsPerTile;
    enqueue_cache_tile(storage.key[stage][0], key_cache, future_position,
                       begin, end, kv_head, warp, lane);
    cp_async_commit();

    cp_async_wait_group_3();
    __syncthreads();
#pragma unroll
    for (unsigned int row = 0U; row < kPositionsPerTile; ++row) {
      // Keep the original per-position online update order.  The six-row tile
      // changes only cache movement/synchronization, not FP32 arithmetic.
      const float score = scores[row];
      const float next_maximum = fmaxf(maximum, score);
      const float previous_scale =
          denominator == 0.0F ? 0.0F : expf(maximum - next_maximum);
      const float current_scale = expf(score - next_maximum);
      denominator = denominator * previous_scale + current_scale;
#pragma unroll
      for (unsigned int index = 0U; index < 8U; ++index) {
        value_numerator[index] = fmaf(
            current_scale,
            decode_bf16(storage.value[stage][row][lane + index * 32U]),
            value_numerator[index] * previous_scale);
      }
      maximum = next_maximum;
    }

    __syncthreads();
    enqueue_cache_tile(storage.value[stage][0], value_cache, future_position,
                       begin, end, kv_head, warp, lane);
    cp_async_commit();
  }
  cp_async_wait_all();
  __syncthreads();

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
template <bool kDynamicGraph>
__global__ __launch_bounds__(kMergeThreads, 2)
void decode_gqa_splitkv_merge_gate_24_4_256_kernel(
    const float* __restrict__ states,
    const unsigned int captured_split_count,
    const std::uint16_t* __restrict__ gate,
    std::uint16_t* __restrict__ output,
    const DecodeDynamicGraphParameters* const parameters) {
  __shared__ float split_scales[kEightSplits];
  __shared__ float inverse_denominator;

  const unsigned int split_count =
      kDynamicGraph ? parameters->split_count : captured_split_count;

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
         decode_gqa_splitkv_split_count(sequence_length) * kStateElements;
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

  const unsigned int split_count =
      decode_gqa_splitkv_split_count(sequence_length);
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  decode_gqa_splitkv_state_24_4_256_kernel<false>
      <<<kKvHeads * split_count, kSplitThreads, 0U, stream>>>(
          query, key_cache, value_cache,
          static_cast<unsigned int>(sequence_length), attention_scale,
          split_count, workspace, nullptr);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  decode_gqa_splitkv_merge_gate_24_4_256_kernel<false>
      <<<kQueryHeads, kMergeThreads, 0U, stream>>>(
          workspace, split_count, gate, output, nullptr);
  return static_cast<int>(cudaGetLastError());
}

int launch_gqa_attention_splitkv_dynamic_graph_sigmoid_gate_24_4_256_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const float attention_scale,
    float* const workspace,
    const std::size_t workspace_elements,
    const std::uint16_t* const gate,
    std::uint16_t* const output,
    const DecodeDynamicGraphParameters* const parameters,
    void* const cuda_stream) noexcept {
  if (workspace_elements < kDecodeGqaSplitKvMaximumWorkspaceElements ||
      !std::isfinite(attention_scale) || attention_scale < 0.0F ||
      query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      workspace == nullptr || gate == nullptr || output == nullptr ||
      parameters == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  decode_gqa_splitkv_state_24_4_256_kernel<true>
      <<<kKvHeads * kEightSplits, kSplitThreads, 0U, stream>>>(
          query, key_cache, value_cache, 0U, attention_scale, 0U,
          workspace, parameters);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  decode_gqa_splitkv_merge_gate_24_4_256_kernel<true>
      <<<kQueryHeads, kMergeThreads, 0U, stream>>>(
          workspace, 0U, gate, output, parameters);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace q3x::runtime
