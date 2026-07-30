#include "gdn_prefill_whole_span_conv_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_whole_span_conv_detail {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kBlocks =
    static_cast<unsigned int>(kGdnQkvChannels / kThreads);
constexpr unsigned int kTokenTile = 8U;
constexpr unsigned int kQkHeadDimension = 128U;
constexpr unsigned int kQkHeadCount = 16U;
constexpr unsigned int kQChannels = kQkHeadCount * kQkHeadDimension;
constexpr unsigned int kKChannels = kQChannels;
constexpr unsigned int kChunkSize = 64U;

static_assert(kGdnQkvChannels == 10240U);
static_assert(kGdnConvHistoryWidth == 3U);
static_assert(kGdnConvKernelWidth == 4U);
static_assert(kGdnQkvChannels % kThreads == 0U);
static_assert(kBlocks == 40U);
static_assert(kThreads == 2U * kQkHeadDimension);
static_assert(kQChannels + kKChannels == 4096U);

__device__ __forceinline__ float decode_bf16_device(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_device(
    const float value) {
  unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

// One thread owns one complete channel. Keeping its four weights and three
// history samples in registers removes the repeated history traffic and the
// 31 inter-tile launch boundaries of a C512 span while retaining precisely
// the same per-token FP32 FMA/SILU/BF16 sequence as the established C16
// reference kernel.
__global__ __launch_bounds__(kThreads)
void causal_conv1d_silu_update_whole_span_kernel(
    const std::uint16_t* const raw_qkv,
    const unsigned int token_count,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history,
    std::uint16_t* const output) {
  const unsigned int channel = blockIdx.x * kThreads + threadIdx.x;
  const unsigned int history_offset = channel * kGdnConvHistoryWidth;
  const unsigned int weight_offset = channel * kGdnConvKernelWidth;

  const float weight_0 = decode_bf16_device(conv_weight[weight_offset]);
  const float weight_1 =
      decode_bf16_device(conv_weight[weight_offset + 1U]);
  const float weight_2 =
      decode_bf16_device(conv_weight[weight_offset + 2U]);
  const float weight_3 =
      decode_bf16_device(conv_weight[weight_offset + 3U]);
  std::uint16_t history_0 = history[history_offset];
  std::uint16_t history_1 = history[history_offset + 1U];
  std::uint16_t history_2 = history[history_offset + 2U];

  unsigned int token_offset = channel;
#pragma unroll 1
  for (unsigned int token = 0U; token < token_count; ++token) {
    const std::uint16_t current_bits = raw_qkv[token_offset];
    float convolution = 0.0F;
    convolution =
        fmaf(decode_bf16_device(history_0), weight_0, convolution);
    convolution =
        fmaf(decode_bf16_device(history_1), weight_1, convolution);
    convolution =
        fmaf(decode_bf16_device(history_2), weight_2, convolution);
    convolution =
        fmaf(decode_bf16_device(current_bits), weight_3, convolution);
    output[token_offset] = encode_bf16_device(
        convolution / (1.0F + expf(-convolution)));
    history_0 = history_1;
    history_1 = history_2;
    history_2 = current_bits;
    token_offset += static_cast<unsigned int>(kGdnQkvChannels);
  }

  history[history_offset] = history_0;
  history[history_offset + 1U] = history_1;
  history[history_offset + 2U] = history_2;
}

// The width-four convolution depends only on raw projection values, not on
// prior convolution outputs. One CTA therefore owns 256 channels across an
// independent C8 token tile. The first token tile snapshots the incoming
// history before publishing the final raw history; all later tiles source
// their three predecessors directly from the immutable raw tensor. A
// disjoint output is required so token tiles cannot race with those reads.
__global__ __launch_bounds__(kThreads)
void causal_conv1d_silu_update_token_parallel_kernel(
    const std::uint16_t* const raw_qkv,
    const unsigned int token_count,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history,
    std::uint16_t* const output) {
  const unsigned int channel = blockIdx.x * kThreads + threadIdx.x;
  const unsigned int token_base = blockIdx.y * kTokenTile;
  const unsigned int history_offset = channel * kGdnConvHistoryWidth;
  const unsigned int weight_offset = channel * kGdnConvKernelWidth;

  const float weight_0 = decode_bf16_device(conv_weight[weight_offset]);
  const float weight_1 =
      decode_bf16_device(conv_weight[weight_offset + 1U]);
  const float weight_2 =
      decode_bf16_device(conv_weight[weight_offset + 2U]);
  const float weight_3 =
      decode_bf16_device(conv_weight[weight_offset + 3U]);

  std::uint16_t history_0;
  std::uint16_t history_1;
  std::uint16_t history_2;
  if (token_base == 0U) {
    history_0 = history[history_offset];
    history_1 = history[history_offset + 1U];
    history_2 = history[history_offset + 2U];

    // Only this tile consumes the incoming history, so it may safely publish
    // the final raw-input history after retaining the old values in registers.
    if (token_count >= kGdnConvHistoryWidth) {
      history[history_offset] =
          raw_qkv[(static_cast<std::size_t>(token_count) - 3U) *
                      kGdnQkvChannels +
                  channel];
      history[history_offset + 1U] =
          raw_qkv[(static_cast<std::size_t>(token_count) - 2U) *
                      kGdnQkvChannels +
                  channel];
      history[history_offset + 2U] =
          raw_qkv[(static_cast<std::size_t>(token_count) - 1U) *
                      kGdnQkvChannels +
                  channel];
    } else if (token_count == 2U) {
      history[history_offset] = history_2;
      history[history_offset + 1U] = raw_qkv[channel];
      history[history_offset + 2U] =
          raw_qkv[kGdnQkvChannels + channel];
    } else {
      history[history_offset] = history_1;
      history[history_offset + 1U] = history_2;
      history[history_offset + 2U] = raw_qkv[channel];
    }
  } else {
    const std::size_t prior_base =
        (static_cast<std::size_t>(token_base) - 3U) * kGdnQkvChannels +
        channel;
    history_0 = raw_qkv[prior_base];
    history_1 = raw_qkv[prior_base + kGdnQkvChannels];
    history_2 = raw_qkv[prior_base + 2U * kGdnQkvChannels];
  }

#pragma unroll
  for (unsigned int local_token = 0U; local_token < kTokenTile;
       ++local_token) {
    const unsigned int token = token_base + local_token;
    if (token < token_count) {
      const std::size_t element =
          static_cast<std::size_t>(token) * kGdnQkvChannels + channel;
      const std::uint16_t current_bits = raw_qkv[element];
      float convolution = 0.0F;
      convolution =
          fmaf(decode_bf16_device(history_0), weight_0, convolution);
      convolution =
          fmaf(decode_bf16_device(history_1), weight_1, convolution);
      convolution =
          fmaf(decode_bf16_device(history_2), weight_2, convolution);
      convolution =
          fmaf(decode_bf16_device(current_bits), weight_3, convolution);
      output[element] = encode_bf16_device(
          convolution / (1.0F + expf(-convolution)));
      history_0 = history_1;
      history_1 = history_2;
      history_2 = current_bits;
    }
  }
}

// The Q/K channel CTAs already own two complete 128-wide heads over their C8
// token tile.  Preserve the exact convolution BF16 store boundary, then run
// the same 128-thread shared-memory reduction tree as normalize_qk_kernel.
// V CTAs retain the established convolution-only dataflow.  This removes the
// later 8192-CTA normalize launch and its second read of convolved Q/K.
__global__ __launch_bounds__(kThreads)
void causal_conv1d_silu_update_token_parallel_compact_qk_kernel(
    const std::uint16_t* const raw_qkv,
    const unsigned int token_count,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history,
    std::uint16_t* const output,
    const float l2_epsilon,
    std::uint16_t* const compact_q,
    std::uint16_t* const compact_k) {
  // C8 x two-head BF16 stage.  Eight warps subsequently own one token each;
  // two head waves cover the 256 channels without any inter-warp reduction.
  __shared__ std::uint16_t convolved_qk[kTokenTile * kThreads];
  const unsigned int channel_block = blockIdx.x;
  const unsigned int channel = channel_block * kThreads + threadIdx.x;
  const unsigned int token_base = blockIdx.y * kTokenTile;
  const unsigned int history_offset = channel * kGdnConvHistoryWidth;
  const unsigned int weight_offset = channel * kGdnConvKernelWidth;
  const bool owns_qk = channel_block < 16U;
  const bool owns_q = channel_block < 8U;

  const float weight_0 = decode_bf16_device(conv_weight[weight_offset]);
  const float weight_1 =
      decode_bf16_device(conv_weight[weight_offset + 1U]);
  const float weight_2 =
      decode_bf16_device(conv_weight[weight_offset + 2U]);
  const float weight_3 =
      decode_bf16_device(conv_weight[weight_offset + 3U]);

  std::uint16_t history_0;
  std::uint16_t history_1;
  std::uint16_t history_2;
  if (token_base == 0U) {
    history_0 = history[history_offset];
    history_1 = history[history_offset + 1U];
    history_2 = history[history_offset + 2U];
    if (token_count >= kGdnConvHistoryWidth) {
      history[history_offset] =
          raw_qkv[(static_cast<std::size_t>(token_count) - 3U) *
                      kGdnQkvChannels +
                  channel];
      history[history_offset + 1U] =
          raw_qkv[(static_cast<std::size_t>(token_count) - 2U) *
                      kGdnQkvChannels +
                  channel];
      history[history_offset + 2U] =
          raw_qkv[(static_cast<std::size_t>(token_count) - 1U) *
                      kGdnQkvChannels +
                  channel];
    } else if (token_count == 2U) {
      history[history_offset] = history_2;
      history[history_offset + 1U] = raw_qkv[channel];
      history[history_offset + 2U] = raw_qkv[kGdnQkvChannels + channel];
    } else {
      history[history_offset] = history_1;
      history[history_offset + 1U] = history_2;
      history[history_offset + 2U] = raw_qkv[channel];
    }
  } else {
    const std::size_t prior_base =
        (static_cast<std::size_t>(token_base) - 3U) * kGdnQkvChannels +
        channel;
    history_0 = raw_qkv[prior_base];
    history_1 = raw_qkv[prior_base + kGdnQkvChannels];
    history_2 = raw_qkv[prior_base + 2U * kGdnQkvChannels];
  }

#pragma unroll
  for (unsigned int local_token = 0U; local_token < kTokenTile;
       ++local_token) {
    const unsigned int token = token_base + local_token;
    if (token < token_count) {
      const std::size_t element =
          static_cast<std::size_t>(token) * kGdnQkvChannels + channel;
      const std::uint16_t current_bits = raw_qkv[element];
      float convolution = 0.0F;
      convolution =
          fmaf(decode_bf16_device(history_0), weight_0, convolution);
      convolution =
          fmaf(decode_bf16_device(history_1), weight_1, convolution);
      convolution =
          fmaf(decode_bf16_device(history_2), weight_2, convolution);
      convolution =
          fmaf(decode_bf16_device(current_bits), weight_3, convolution);
      const std::uint16_t convolved_bits = encode_bf16_device(
          convolution / (1.0F + expf(-convolution)));
      output[element] = convolved_bits;
      history_0 = history_1;
      history_1 = history_2;
      history_2 = current_bits;

      if (owns_qk) {
        convolved_qk[local_token * kThreads + threadIdx.x] =
            convolved_bits;
      }
    }
  }

  if (!owns_qk) {
    return;
  }
  __syncthreads();
  constexpr unsigned int kFullWarpMask = 0xffffffffU;
  const unsigned int warp = threadIdx.x / 32U;
  const unsigned int lane = threadIdx.x % 32U;
  const unsigned int token = token_base + warp;

  // Baseline tree equivalence for dimensions [lane + 0,32,64,96]:
  // stride64 produces (0+64),(32+96); stride32 adds those pairs; the
  // shuffle sequence reproduces strides16..1.  Thus only execution topology,
  // not floating-point association, changes.
#pragma unroll
  for (unsigned int head_in_block = 0U; head_in_block < 2U;
       ++head_in_block) {
    float values[4];
    float pair_0 = 0.0F;
    float pair_1 = 0.0F;
#pragma unroll
    for (unsigned int item = 0U; item < 4U; ++item) {
      const unsigned int dimension = lane + item * 32U;
      values[item] = decode_bf16_device(
          convolved_qk[warp * kThreads +
                       head_in_block * kQkHeadDimension + dimension]);
    }
    pair_0 = values[0] * values[0] + values[2] * values[2];
    pair_1 = values[1] * values[1] + values[3] * values[3];
    float square_sum = pair_0 + pair_1;
#pragma unroll
    for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
      square_sum +=
          __shfl_down_sync(kFullWarpMask, square_sum, offset);
    }
    float scale = __shfl_sync(
        kFullWarpMask,
        lane == 0U ? rsqrtf(square_sum + l2_epsilon) : 0.0F, 0U);
    if (owns_q) {
      scale *= rsqrtf(static_cast<float>(kQkHeadDimension));
    }
    const unsigned int qk_head =
        (channel_block % 8U) * 2U + head_in_block;
    const std::size_t chunk = token / kChunkSize;
    const std::size_t token_in_chunk = token % kChunkSize;
    const std::size_t destination_base =
        ((chunk * kQkHeadCount + qk_head) * kChunkSize +
         token_in_chunk) *
        kQkHeadDimension;
#pragma unroll
    for (unsigned int item = 0U; item < 4U; ++item) {
      const unsigned int dimension = lane + item * 32U;
      (owns_q ? compact_q : compact_k)[destination_base + dimension] =
          encode_bf16_device(values[item] * scale);
    }
  }
}

[[nodiscard]] bool invalid_alias(
    const std::uint16_t* const raw_qkv,
    const std::uint16_t* const conv_weight,
    const std::uint16_t* const history,
    const std::uint16_t* const output) noexcept {
  return raw_qkv == conv_weight || raw_qkv == history ||
         conv_weight == history || conv_weight == output ||
         history == output;
}

}  // namespace

int launch_causal_conv1d_silu_update_whole_span_exact_cuda(
    const std::uint16_t* const raw_qkv,
    const std::size_t token_count,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history_in_out,
    std::uint16_t* const conv_qkv_output,
    void* const cuda_stream) noexcept {
  if (token_count == 0U || token_count > kMaximumTokenCount ||
      raw_qkv == nullptr || conv_weight == nullptr ||
      history_in_out == nullptr || conv_qkv_output == nullptr ||
      invalid_alias(raw_qkv, conv_weight, history_in_out,
                    conv_qkv_output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  causal_conv1d_silu_update_whole_span_kernel<<<
      kBlocks, kThreads, 0U, stream>>>(
      raw_qkv, static_cast<unsigned int>(token_count), conv_weight,
      history_in_out, conv_qkv_output);
  return static_cast<int>(cudaGetLastError());
}

int launch_causal_conv1d_silu_update_token_parallel_exact_cuda(
    const std::uint16_t* const raw_qkv,
    const std::size_t token_count,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history_in_out,
    std::uint16_t* const conv_qkv_output,
    void* const cuda_stream) noexcept {
  if (token_count == 0U || token_count > kMaximumTokenCount ||
      raw_qkv == nullptr || conv_weight == nullptr ||
      history_in_out == nullptr || conv_qkv_output == nullptr ||
      raw_qkv == conv_qkv_output ||
      invalid_alias(raw_qkv, conv_weight, history_in_out,
                    conv_qkv_output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  const unsigned int token_tiles =
      static_cast<unsigned int>((token_count + kTokenTile - 1U) /
                                kTokenTile);
  (void)cudaGetLastError();
  causal_conv1d_silu_update_token_parallel_kernel<<<
      dim3(kBlocks, token_tiles), kThreads, 0U, stream>>>(
      raw_qkv, static_cast<unsigned int>(token_count), conv_weight,
      history_in_out, conv_qkv_output);
  return static_cast<int>(cudaGetLastError());
}

int launch_causal_conv1d_silu_update_token_parallel_compact_qk_exact_cuda(
    const std::uint16_t* const raw_qkv,
    const std::size_t token_count,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history_in_out,
    std::uint16_t* const conv_qkv_output,
    const float l2_epsilon,
    std::uint16_t* const compact_q,
    std::uint16_t* const compact_k,
    void* const cuda_stream) noexcept {
  if (token_count == 0U || token_count > kMaximumTokenCount ||
      token_count % kChunkSize != 0U || raw_qkv == nullptr ||
      conv_weight == nullptr || history_in_out == nullptr ||
      conv_qkv_output == nullptr || compact_q == nullptr ||
      compact_k == nullptr || !std::isfinite(l2_epsilon) ||
      l2_epsilon <= 0.0F || raw_qkv == conv_qkv_output ||
      compact_q == compact_k || compact_q == raw_qkv ||
      compact_q == conv_weight || compact_q == history_in_out ||
      compact_q == conv_qkv_output || compact_k == raw_qkv ||
      compact_k == conv_weight || compact_k == history_in_out ||
      compact_k == conv_qkv_output ||
      invalid_alias(raw_qkv, conv_weight, history_in_out,
                    conv_qkv_output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  const unsigned int token_tiles =
      static_cast<unsigned int>((token_count + kTokenTile - 1U) /
                                kTokenTile);
  (void)cudaGetLastError();
  causal_conv1d_silu_update_token_parallel_compact_qk_kernel<<<
      dim3(kBlocks, token_tiles), kThreads, 0U, stream>>>(
      raw_qkv, static_cast<unsigned int>(token_count), conv_weight,
      history_in_out, conv_qkv_output, l2_epsilon, compact_q, compact_k);
  return static_cast<int>(cudaGetLastError());
}

int query_causal_conv1d_silu_update_whole_span_resources_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, causal_conv1d_silu_update_whole_span_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, causal_conv1d_silu_update_whole_span_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int query_causal_conv1d_silu_update_token_parallel_compact_qk_resources_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      causal_conv1d_silu_update_token_parallel_compact_qk_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      causal_conv1d_silu_update_token_parallel_compact_qk_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::runtime::gdn_prefill_whole_span_conv_detail
