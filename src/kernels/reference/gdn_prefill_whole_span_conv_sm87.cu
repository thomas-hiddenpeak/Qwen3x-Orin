#include "gdn_prefill_whole_span_conv_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_whole_span_conv_detail {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kBlocks =
    static_cast<unsigned int>(kGdnQkvChannels / kThreads);
constexpr unsigned int kTokenTile = 8U;

static_assert(kGdnQkvChannels == 10240U);
static_assert(kGdnConvHistoryWidth == 3U);
static_assert(kGdnConvKernelWidth == 4U);
static_assert(kGdnQkvChannels % kThreads == 0U);
static_assert(kBlocks == 40U);

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

}  // namespace q3x::runtime::gdn_prefill_whole_span_conv_detail
