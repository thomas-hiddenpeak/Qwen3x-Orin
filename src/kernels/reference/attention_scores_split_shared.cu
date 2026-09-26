// Split-chunk GQA-shared attention scores path (opt-in via
// Q3X_ENABLE_SPLIT_SCORES). Kept in a dedicated translation unit so the
// production decode_ops.cu compile context (and its ptxas resource
// layout for the warp-RMS contract kernel) is unaffected.
//
// Unlike the values split path, this kernel is BIT-EXACT: each
// (query_head, position) pair keeps the production FMA order and shuffle
// tree, and writes scores directly (no atomicAdd, no finalize). The only
// change is that the K tile is loaded into smem once per chunk instead of
// being re-read from DRAM by each of the 6 query-head blocks.
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "cuda_runtime.h"

namespace q3x::runtime {

namespace {

constexpr unsigned int kKvHeads = 4U;
constexpr unsigned int kQueriesPerKv = 6U;
constexpr unsigned int kHeadDimension = 256U;
constexpr unsigned int kPositionStride = 1024U;  // 4 kv * 256 dim
constexpr unsigned int kTile = 32U;
constexpr unsigned int kThreads = 512U;
constexpr unsigned int kMinimumSequence = 512U;

__device__ __forceinline__ float decode_bf16_device(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

// grid (S/TILE, 4) x 512.
// Each block loads a TILE-position K tile for one kv_head into smem
// (TILE * 256 * 2 bytes = 16 KiB for TILE=32). 16 warps per block, each
// warp iterates over (pos_in_tile, qh_in_kv) pairs: 6 qh x TILE pos.
// Per (qh, pos) the FMA order and shuffle tree are identical to the
// production warp-positions kernel, so the output is bit-exact.
__global__ void __launch_bounds__(kThreads)
attention_scores_split_shared_24_4_256_kernel(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const unsigned int sequence_length,
    const float attention_scale,
    float* const scores) {
  extern __shared__ std::uint16_t shared_key[];
  const unsigned int chunk = blockIdx.x;
  const unsigned int kv_head = blockIdx.y;
  const unsigned int thread = threadIdx.x;
  const unsigned int chunk_begin = chunk * kTile;
  const unsigned int chunk_end =
      min(chunk_begin + kTile, sequence_length);

  // Load K tile: TILE positions x 256 dim for this kv_head.
  // Each thread loads (TILE * 256) / 512 = TILE/2 elements.
  const unsigned int k_base = kv_head * kHeadDimension;
  for (unsigned int i = thread; i < kTile * kHeadDimension; i += kThreads) {
    const unsigned int pos_in = i / kHeadDimension;
    const unsigned int dim = i % kHeadDimension;
    const unsigned int pos = chunk_begin + pos_in;
    shared_key[i] =
        (pos < sequence_length)
            ? key_cache[static_cast<std::size_t>(pos) * kPositionStride +
                        k_base + dim]
            : 0U;
  }
  __syncthreads();

  // 16 warps; each warp handles (pos_in, qh_in) pairs.
  const unsigned int warp = thread >> 5U;
  const unsigned int lane = thread & 31U;
  const unsigned int qh_base = kv_head * kQueriesPerKv;

  for (unsigned int iter = warp; iter < kQueriesPerKv * kTile; iter += 16U) {
    const unsigned int pos_in = iter / kQueriesPerKv;
    const unsigned int qh_in = iter % kQueriesPerKv;
    const unsigned int pos = chunk_begin + pos_in;
    if (pos >= sequence_length) {
      break;
    }
    const unsigned int qh = qh_base + qh_in;
    const unsigned int q_off = qh * kHeadDimension;
    const unsigned int k_off = pos_in * kHeadDimension;

    // Identical FMA order to the production warp-positions kernel.
    float sum = fmaf(decode_bf16_device(query[q_off + lane]),
                     decode_bf16_device(shared_key[k_off + lane]), 0.0F);
    float rhs = fmaf(decode_bf16_device(query[q_off + lane + 128U]),
                     decode_bf16_device(shared_key[k_off + lane + 128U]), 0.0F);
    sum += rhs;

    float sibling = fmaf(decode_bf16_device(query[q_off + lane + 64U]),
                         decode_bf16_device(shared_key[k_off + lane + 64U]),
                         0.0F);
    rhs = fmaf(decode_bf16_device(query[q_off + lane + 192U]),
               decode_bf16_device(shared_key[k_off + lane + 192U]), 0.0F);
    sibling += rhs;
    sum += sibling;

    sibling = fmaf(decode_bf16_device(query[q_off + lane + 32U]),
                   decode_bf16_device(shared_key[k_off + lane + 32U]), 0.0F);
    rhs = fmaf(decode_bf16_device(query[q_off + lane + 160U]),
               decode_bf16_device(shared_key[k_off + lane + 160U]), 0.0F);
    sibling += rhs;
    float upper = fmaf(decode_bf16_device(query[q_off + lane + 96U]),
                       decode_bf16_device(shared_key[k_off + lane + 96U]),
                       0.0F);
    rhs = fmaf(decode_bf16_device(query[q_off + lane + 224U]),
               decode_bf16_device(shared_key[k_off + lane + 224U]), 0.0F);
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
      scores[qh * sequence_length + pos] = sum * attention_scale;
    }
  }
}

// Returns 0 on success, a positive CUDA error, or -1 when the split path
// is not selected (caller falls back to the production kernel).
int launch_attention_scores_split_shared_24_4_256(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const unsigned int sequence_length,
    const float attention_scale,
    float* const scores,
    cudaStream_t stream) noexcept {
  if (sequence_length < kMinimumSequence ||
      getenv("Q3X_ENABLE_SPLIT_SCORES") == nullptr) {
    return -1;
  }
  // The split path is only used in eager (non-captured) execution, same
  // guard as the values split path.
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &capture_status) != cudaSuccess ||
      capture_status == cudaStreamCaptureStatusActive) {
    (void)cudaGetLastError();
    return -1;
  }
  const unsigned int chunks =
      (sequence_length + kTile - 1U) / kTile;
  const dim3 blocks(chunks, kKvHeads, 1U);
  const std::size_t smem_bytes = kTile * kHeadDimension * sizeof(std::uint16_t);
  attention_scores_split_shared_24_4_256_kernel
      <<<blocks, kThreads, smem_bytes, stream>>>(
          query, key_cache, sequence_length, attention_scale, scores);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

int launch_attention_scores_split_shared_24_4_256_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const unsigned int sequence_length,
    const float attention_scale,
    float* const scores,
    cudaStream_t stream) noexcept {
  return launch_attention_scores_split_shared_24_4_256(
      query, key_cache, sequence_length, attention_scale, scores, stream);
}

}  // namespace q3x::runtime
