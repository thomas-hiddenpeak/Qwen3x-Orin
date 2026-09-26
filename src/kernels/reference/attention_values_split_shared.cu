// Split-chunk GQA-shared attention values path (opt-in via
// Q3X_ENABLE_SPLIT_VALUES). Kept in a dedicated translation unit so the
// production decode_ops.cu compile context (and its ptxas resource
// layout for the warp-RMS contract kernel) is unaffected.
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "cuda_runtime.h"

namespace q3x::runtime {

namespace {

constexpr unsigned int kThreads = 256U;
constexpr std::size_t kExactAttentionValueHeadDimension = 256U;
constexpr unsigned int kExactAttentionValueQueriesPerKv = 6U;
constexpr std::size_t kExactAttentionValuePositionStride = 1024U;
constexpr unsigned int kExactAttentionValueKvHeads = 4U;
constexpr unsigned int kExactAttentionValueQueryHeads = 24U;

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

constexpr unsigned int kSplitValuesTile = 64U;
constexpr unsigned int kSplitValuesThreads = 512U;
// The split path is only selected for long sequences: the short
// (S<=511) decode path is captured into a 3-kernel CUDA graph
// (scores, softmax, values) whose topology is a tested contract, and
// the 6x-redundant V read is small enough there to stay L2-resident.
constexpr unsigned int kSplitValuesMinimumSequence = 512U;

__global__ void __launch_bounds__(kSplitValuesThreads)
attention_values_split_shared_24_4_256_kernel(
    const std::uint16_t* const value_cache,
    const float* const probabilities,
    const unsigned int sequence_length,
    float* const accumulator) {
  extern __shared__ std::uint16_t shared_value[];
  const unsigned int chunk = blockIdx.x;
  const unsigned int kv_head = blockIdx.y;
  const unsigned int thread = threadIdx.x;
  const unsigned int dimension = thread % kExactAttentionValueHeadDimension;
  const unsigned int group = thread / kExactAttentionValueHeadDimension;
  const unsigned int query_head_a =
      kv_head * kExactAttentionValueQueriesPerKv + group;
  const unsigned int query_head_b = query_head_a + 2U;
  const unsigned int query_head_c = query_head_a + 4U;
  const unsigned int chunk_begin = chunk * kSplitValuesTile;
  const unsigned int chunk_end =
      min(chunk_begin + kSplitValuesTile, sequence_length);
  const unsigned int value_base =
      kv_head * static_cast<unsigned int>(kExactAttentionValueHeadDimension) +
      dimension;
  float value_a = 0.0F;
  float value_b = 0.0F;
  float value_c = 0.0F;
  for (unsigned int tile = 0U; tile < kSplitValuesTile; ++tile) {
    const unsigned int position = chunk_begin + tile;
    if (position < sequence_length) {
      shared_value[tile * kExactAttentionValueHeadDimension + dimension] =
          value_cache[
              static_cast<std::size_t>(position) *
                  kExactAttentionValuePositionStride +
              value_base];
    }
  }
  __syncthreads();
  for (unsigned int tile = 0U; tile < kSplitValuesTile; ++tile) {
    const unsigned int position = chunk_begin + tile;
    if (position >= chunk_end) {
      break;
    }
    const float value =
        decode_bf16_device(shared_value[tile *
                                            kExactAttentionValueHeadDimension +
                                        dimension]);
    value_a = fmaf(probabilities[query_head_a * sequence_length + position],
                   value, value_a);
    value_b = fmaf(probabilities[query_head_b * sequence_length + position],
                   value, value_b);
    value_c = fmaf(probabilities[query_head_c * sequence_length + position],
                   value, value_c);
  }
  const unsigned int output_base =
      kv_head * kExactAttentionValueQueriesPerKv *
          static_cast<unsigned int>(kExactAttentionValueHeadDimension) +
      dimension;
  atomicAdd(accumulator + output_base + group *
                                         kExactAttentionValueHeadDimension,
            value_a);
  atomicAdd(accumulator + output_base + (group + 2U) *
                                         kExactAttentionValueHeadDimension,
            value_b);
  atomicAdd(accumulator + output_base + (group + 4U) *
                                         kExactAttentionValueHeadDimension,
            value_c);
}

__global__ void attention_values_split_finalize_24_4_256_kernel(
    const float* const accumulator,
    std::uint16_t* const output) {
  const unsigned int index =
      blockIdx.x * static_cast<unsigned int>(kThreads) + threadIdx.x;
  output[index] = encode_bf16_device(accumulator[index]);
}


// Opt-in split-chunk GQA-shared values path. Reads each V tile from
// DRAM once into smem, 6 query heads share the tile, partial sums are
// combined via atomicAdd. Numerically equivalent to the production
// kernel (a few ULPs) but NOT bit-identical, due to the
// non-deterministic cross-chunk accumulation order.
// Persistent fp32 accumulator for the split path. Lazily allocated once
// (24*256 = 6144 floats = 24 KiB) and reused across every decode step.
// The split path is only taken for eager long-sequence decode (S>=512);
// the graph-captured short decode path (S<=63) always uses the production
// kernel, so this buffer is never touched during graph capture.
float* split_accumulator_buffer() noexcept {
  static float* buffer = nullptr;
  if (buffer == nullptr) {
    if (cudaMalloc(&buffer, kExactAttentionValueQueryHeads *
                                kExactAttentionValueHeadDimension *
                                sizeof(float)) != cudaSuccess) {
      buffer = nullptr;
      (void)cudaGetLastError();
      return nullptr;
    }
  }
  return buffer;
}

int launch_attention_values_split_shared_24_4_256(
    const std::uint16_t* const value_cache,
    float* const probabilities,
    const unsigned int sequence_length,
    const std::size_t query_head_count,
    const std::size_t head_dimension,
    std::uint16_t* const output,
    cudaStream_t stream) noexcept {
  if (sequence_length < kSplitValuesMinimumSequence ||
      getenv("Q3X_ENABLE_SPLIT_VALUES") == nullptr ||
      query_head_count != kExactAttentionValueQueryHeads ||
      head_dimension != kExactAttentionValueHeadDimension) {
    return -1;  // caller falls back to the production kernel
  }
  // The split path is only used in eager (non-captured) execution. During a
  // CUDA graph capture it would (a) perform a cudaMalloc, which is illegal in
  // capture, and (b) change the captured graph topology, which is a tested
  // contract. Production decode graphs only cover S<=63 (never split-eligible),
  // so this guard only affects test-captured long-sequence graphs: they keep
  // the production bit-exact kernel and its 3-node attention topology.
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &capture_status) != cudaSuccess ||
      capture_status == cudaStreamCaptureStatusActive) {
    (void)cudaGetLastError();
    return -1;  // capturing: fall back to the production kernel
  }
  float* const accumulator = split_accumulator_buffer();
  if (accumulator == nullptr) {
    return -1;  // allocation failed: fall back to the production kernel
  }
  const std::size_t accumulator_elements = query_head_count * head_dimension;
  cudaError_t status = cudaMemsetAsync(
      accumulator, 0, accumulator_elements * sizeof(float), stream);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const unsigned int chunks =
      (sequence_length + kSplitValuesTile - 1U) / kSplitValuesTile;
  const dim3 split_blocks(chunks, kExactAttentionValueKvHeads, 1U);
  const std::size_t smem_bytes = kSplitValuesTile *
      kExactAttentionValueHeadDimension * sizeof(std::uint16_t);
  attention_values_split_shared_24_4_256_kernel
      <<<split_blocks, kSplitValuesThreads, smem_bytes, stream>>>(
          value_cache, probabilities, sequence_length, accumulator);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const unsigned int finalize_blocks =
      static_cast<unsigned int>((accumulator_elements + kThreads - 1U) /
                                kThreads);
  attention_values_split_finalize_24_4_256_kernel
      <<<finalize_blocks, kThreads, 0U, stream>>>(accumulator, output);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

int launch_attention_values_split_shared_24_4_256_cuda(
    const std::uint16_t* const value_cache,
    float* const probabilities,
    const unsigned int sequence_length,
    const std::size_t query_head_count,
    const std::size_t head_dimension,
    std::uint16_t* const output,
    cudaStream_t stream) noexcept {
  return launch_attention_values_split_shared_24_4_256(
      value_cache, probabilities, sequence_length, query_head_count,
      head_dimension, output, stream);
}

}  // namespace q3x::runtime
