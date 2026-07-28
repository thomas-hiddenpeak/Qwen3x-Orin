#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace q3x::runtime {

// Existing test-only production resource query. The candidate remains in this
// isolated translation unit and never enters q3x_kernels or the public ABI.
[[nodiscard]] int
query_gated_delta_net_update_warp_eight_row_register_state_m16_resources_test_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

namespace gdn_prefill_c16_pingpong_test_detail {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kWarpsPerBlock = kThreads / kWarpSize;
constexpr unsigned int kRowsPerWarp = 8U;
constexpr unsigned int kBatchRowOffset = 64U;
constexpr unsigned int kTokenCount = 16U;
constexpr unsigned int kFullWarpMask = 0xffffffffU;
constexpr std::size_t kKeysPerLane = kGdnHeadDimension / kWarpSize;
constexpr std::size_t kKeyPairsPerLane = kKeysPerLane / 2U;
constexpr std::size_t kScratchRowStride = kGdnHeadDimension + 1U;
constexpr std::size_t kExpectedSharedBytes = 35200U;

static_assert(kWarpsPerBlock * kRowsPerWarp * 2U == kGdnHeadDimension);
static_assert(kKeysPerLane == 4U);
static_assert(kKeyPairsPerLane == 2U);
static_assert(kScratchRowStride == 129U);
static_assert((2U * 2U * kGdnHeadDimension + 2U * kTokenCount +
               kWarpsPerBlock * kRowsPerWarp * kScratchRowStride) *
                  sizeof(float) ==
              kExpectedSharedBytes);

[[nodiscard]] bool invalid_gdn_alias(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    const std::uint16_t* const state_output,
    const std::uint16_t* const output) noexcept {
  const bool output_alias =
      output == conv_qkv || output == a || output == b || output == A_log ||
      output == dt_bias || output == state_input || output == state_output;
  const bool state_input_alias =
      state_input == conv_qkv || state_input == a || state_input == b ||
      state_input == A_log || state_input == dt_bias;
  const bool state_output_alias =
      state_output == conv_qkv || state_output == a || state_output == b ||
      state_output == A_log || state_output == dt_bias;
  return output_alias || state_input_alias || state_output_alias;
}

__device__ __forceinline__ float decode_bf16_device(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_device(
    const float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ std::uint32_t encode_bf16_pair_device(
    const float first, const float second) {
  const unsigned int first_bits = __float_as_uint(first);
  const unsigned int second_bits = __float_as_uint(second);
  const __nv_bfloat162_raw converted = static_cast<__nv_bfloat162_raw>(
      __floats2bfloat162_rn(first, second));
  std::uint32_t packed = static_cast<std::uint32_t>(converted.x) |
                         (static_cast<std::uint32_t>(converted.y) << 16U);
  if ((first_bits & 0x7fffffffU) > 0x7f800000U) {
    packed = (packed & 0xffff0000U) |
             static_cast<std::uint32_t>((first_bits >> 16U) | 0x0040U);
  }
  if ((second_bits & 0x7fffffffU) > 0x7f800000U) {
    packed = (packed & 0x0000ffffU) |
             (static_cast<std::uint32_t>((second_bits >> 16U) | 0x0040U)
              << 16U);
  }
  return packed;
}

__device__ __forceinline__ float stable_softplus_device(const float value) {
  return value > 20.0F ? value : log1pf(expf(value));
}

__device__ __forceinline__ float stable_sigmoid_device(const float value) {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + expf(-value));
  }
  const float exponential = expf(value);
  return exponential / (1.0F + exponential);
}

template <bool NormalizeQ>
__device__ __forceinline__ void normalize_q_or_k_direct(
    const std::uint16_t* const conv_qkv, const std::size_t token,
    const std::size_t qk_head, const float l2_epsilon,
    float* const destination, const unsigned int lane) {
  constexpr std::size_t kKOffset = kGdnQElements;
  const std::size_t token_offset = token * kGdnQkvChannels;
  const std::size_t operand_offset =
      token_offset + (NormalizeQ ? 0U : kKOffset) +
      qk_head * kGdnHeadDimension;
  const float first = decode_bf16_device(conv_qkv[operand_offset + lane]);
  const float second =
      decode_bf16_device(conv_qkv[operand_offset + lane + 32U]);
  const float third =
      decode_bf16_device(conv_qkv[operand_offset + lane + 64U]);
  const float fourth =
      decode_bf16_device(conv_qkv[operand_offset + lane + 96U]);

  // This is deliberately the production four-item tree, including its
  // pair order, followed by the same lane-ordered warp reduction.
  const float first_square = first * first;
  const float second_square = second * second;
  const float third_square = third * third;
  const float fourth_square = fourth * fourth;
  const float first_pair = first_square + third_square;
  const float second_pair = second_square + fourth_square;
  float warp_sum = first_pair + second_pair;
#pragma unroll
  for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
    const float shuffled =
        __shfl_down_sync(kFullWarpMask, warp_sum, stride);
    warp_sum += shuffled;
  }
  float scale = 0.0F;
  if (lane == 0U) {
    if constexpr (NormalizeQ) {
      scale = rsqrtf(warp_sum + l2_epsilon) *
              rsqrtf(static_cast<float>(kGdnHeadDimension));
    } else {
      scale = rsqrtf(warp_sum + l2_epsilon);
    }
  }
  scale = __shfl_sync(kFullWarpMask, scale, 0U);
  destination[lane] = first * scale;
  destination[lane + 32U] = second * scale;
  destination[lane + 64U] = third * scale;
  destination[lane + 96U] = fourth * scale;
}

// Exact C16 experiment. Warp 2 lanes 0..15 precompute independent recurrence
// scalars. Warps 0/1 normalize Q/K token 0 before the loop and, after their
// own recurrence work, stage token+1 into the opposite parity slot. The sole
// CTA barrier at the top of each token both publishes the new slot and proves
// that every warp has finished reading the slot that can be reused two tokens
// later: 16 barriers total versus production's 47.
__launch_bounds__(kThreads, 4) __global__ void
gated_delta_net_update_c16_scalar_vector_qk_pingpong_kernel(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    std::uint16_t* const output) {
  __shared__ float normalized_q[2][kGdnHeadDimension];
  __shared__ float normalized_k[2][kGdnHeadDimension];
  __shared__ float alpha[kTokenCount];
  __shared__ float beta[kTokenCount];
  __shared__ float
      row_scratch[kWarpsPerBlock * kRowsPerWarp * kScratchRowStride];

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const std::size_t value_head = blockIdx.x;
  const std::size_t qk_head = value_head / 3U;
  const std::size_t head_state_offset =
      value_head * kGdnHeadDimension * kGdnHeadDimension;
  const std::size_t warp_first_row =
      static_cast<std::size_t>(warp) * kRowsPerWarp;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

  std::uint32_t lower_state_words[kRowsPerWarp][kKeyPairsPerLane];
  std::uint32_t upper_state_words[kRowsPerWarp][kKeyPairsPerLane];
#pragma unroll
  for (unsigned int row = 0U; row < kRowsPerWarp; ++row) {
#pragma unroll
    for (std::size_t pair = 0U; pair < kKeyPairsPerLane; ++pair) {
      const std::size_t first_key =
          lane + pair * 2U * static_cast<std::size_t>(kWarpSize);
      const std::size_t second_key = first_key + kWarpSize;
      const std::size_t lower_row_offset =
          head_state_offset +
          (warp_first_row + static_cast<std::size_t>(row)) *
              kGdnHeadDimension;
      const std::size_t upper_row_offset =
          lower_row_offset +
          static_cast<std::size_t>(kBatchRowOffset) * kGdnHeadDimension;
      lower_state_words[row][pair] =
          static_cast<std::uint32_t>(state_input[lower_row_offset + first_key]) |
          (static_cast<std::uint32_t>(
               state_input[lower_row_offset + second_key])
           << 16U);
      upper_state_words[row][pair] =
          static_cast<std::uint32_t>(state_input[upper_row_offset + first_key]) |
          (static_cast<std::uint32_t>(
               state_input[upper_row_offset + second_key])
           << 16U);
    }
  }

  if (warp == 0U) {
    normalize_q_or_k_direct<true>(conv_qkv, 0U, qk_head, l2_epsilon,
                                  normalized_q[0], lane);
  } else if (warp == 1U) {
    normalize_q_or_k_direct<false>(conv_qkv, 0U, qk_head, l2_epsilon,
                                   normalized_k[0], lane);
  } else if (warp == 2U && lane < kTokenCount) {
    const std::size_t scalar_offset =
        static_cast<std::size_t>(lane) * kGdnValueHeadCount + value_head;
    const float gate_input = decode_bf16_device(a[scalar_offset]) +
                             decode_bf16_device(dt_bias[value_head]);
    const float g = -expf(decode_bf16_device(A_log[value_head])) *
                    stable_softplus_device(gate_input);
    alpha[lane] = expf(g);
    beta[lane] =
        stable_sigmoid_device(decode_bf16_device(b[scalar_offset]));
  }

#pragma unroll 1
  for (unsigned int token = 0U; token < kTokenCount; ++token) {
    __syncthreads();
    const unsigned int parity = token & 1U;
    const float token_alpha = alpha[token];
    const float token_beta = beta[token];
    const std::size_t qkv_token_offset =
        static_cast<std::size_t>(token) * kGdnQkvChannels;
    const std::size_t output_token_offset =
        static_cast<std::size_t>(token) * kGdnVElements;
    float* const warp_scratch =
        row_scratch +
        static_cast<std::size_t>(warp * kRowsPerWarp) * kScratchRowStride;

#pragma unroll 1
    for (unsigned int batch = 0U; batch < 2U; ++batch) {
      const std::size_t first_row =
          warp_first_row +
          static_cast<std::size_t>(batch * kBatchRowOffset);

#pragma unroll
      for (std::size_t pair = 0U; pair < kKeyPairsPerLane; ++pair) {
        const std::size_t first_key =
            lane + pair * 2U * static_cast<std::size_t>(kWarpSize);
        const std::size_t second_key = first_key + kWarpSize;
#pragma unroll
        for (unsigned int row = 0U; row < kRowsPerWarp; ++row) {
          const std::uint32_t word =
              batch == 0U ? lower_state_words[row][pair]
                          : upper_state_words[row][pair];
          const std::size_t scratch_row_offset =
              static_cast<std::size_t>(row) * kScratchRowStride;
          warp_scratch[scratch_row_offset + first_key] =
              token_alpha * decode_bf16_device(
                                static_cast<std::uint16_t>(word));
          warp_scratch[scratch_row_offset + second_key] =
              token_alpha * decode_bf16_device(
                                static_cast<std::uint16_t>(word >> 16U));
        }
      }
      __syncwarp(kFullWarpMask);

      float lane_prediction = 0.0F;
      if (lane < kRowsPerWarp) {
        const float* const lane_scratch =
            warp_scratch +
            static_cast<std::size_t>(lane) * kScratchRowStride;
#pragma unroll
        for (std::size_t key = 0U; key < kGdnHeadDimension; ++key) {
          lane_prediction =
              fmaf(lane_scratch[key], normalized_k[parity][key],
                   lane_prediction);
        }
      }

      float lane_delta = 0.0F;
      if (lane < kRowsPerWarp) {
        const std::size_t value_offset =
            qkv_token_offset + kVOffset + value_head * kGdnHeadDimension;
        const float lane_value =
            decode_bf16_device(conv_qkv[value_offset + first_row + lane]);
        lane_delta = (lane_value - lane_prediction) * token_beta;
      }
      float deltas[kRowsPerWarp];
#pragma unroll
      for (unsigned int row = 0U; row < kRowsPerWarp; ++row) {
        deltas[row] = __shfl_sync(kFullWarpMask, lane_delta, row);
      }

#pragma unroll
      for (std::size_t pair = 0U; pair < kKeyPairsPerLane; ++pair) {
        const std::size_t first_key =
            lane + pair * 2U * static_cast<std::size_t>(kWarpSize);
        const std::size_t second_key = first_key + kWarpSize;
#pragma unroll
        for (unsigned int row = 0U; row < kRowsPerWarp; ++row) {
          const std::size_t scratch_row_offset =
              static_cast<std::size_t>(row) * kScratchRowStride;
          const std::size_t first_scratch_offset =
              scratch_row_offset + first_key;
          const std::size_t second_scratch_offset =
              scratch_row_offset + second_key;
          const float first_updated =
              fmaf(deltas[row], normalized_k[parity][first_key],
                   warp_scratch[first_scratch_offset]);
          const float second_updated =
              fmaf(deltas[row], normalized_k[parity][second_key],
                   warp_scratch[second_scratch_offset]);
          const std::uint32_t rounded =
              encode_bf16_pair_device(first_updated, second_updated);
          if (batch == 0U) {
            lower_state_words[row][pair] = rounded;
          } else {
            upper_state_words[row][pair] = rounded;
          }
          // Output consumes the unrounded FP32 state from this recurrence.
          warp_scratch[first_scratch_offset] = first_updated;
          warp_scratch[second_scratch_offset] = second_updated;
        }
      }
      __syncwarp(kFullWarpMask);

      if (lane < kRowsPerWarp) {
        const float* const lane_scratch =
            warp_scratch +
            static_cast<std::size_t>(lane) * kScratchRowStride;
        float lane_result = 0.0F;
#pragma unroll
        for (std::size_t key = 0U; key < kGdnHeadDimension; ++key) {
          lane_result =
              fmaf(lane_scratch[key], normalized_q[parity][key],
                   lane_result);
        }
        const std::size_t output_offset =
            output_token_offset + value_head * kGdnHeadDimension;
        output[output_offset + first_row + lane] =
            encode_bf16_device(lane_result);
      }
      __syncwarp(kFullWarpMask);
    }

    if (token + 1U < kTokenCount) {
      const unsigned int next_parity = parity ^ 1U;
      if (warp == 0U) {
        normalize_q_or_k_direct<true>(
            conv_qkv, static_cast<std::size_t>(token + 1U), qk_head,
            l2_epsilon, normalized_q[next_parity], lane);
      } else if (warp == 1U) {
        normalize_q_or_k_direct<false>(
            conv_qkv, static_cast<std::size_t>(token + 1U), qk_head,
            l2_epsilon, normalized_k[next_parity], lane);
      }
    }
  }

#pragma unroll
  for (unsigned int row = 0U; row < kRowsPerWarp; ++row) {
#pragma unroll
    for (std::size_t pair = 0U; pair < kKeyPairsPerLane; ++pair) {
      const std::size_t first_key =
          lane + pair * 2U * static_cast<std::size_t>(kWarpSize);
      const std::size_t second_key = first_key + kWarpSize;
      const std::size_t lower_row_offset =
          head_state_offset +
          (warp_first_row + static_cast<std::size_t>(row)) *
              kGdnHeadDimension;
      const std::size_t upper_row_offset =
          lower_row_offset +
          static_cast<std::size_t>(kBatchRowOffset) * kGdnHeadDimension;
      const std::uint32_t lower_word = lower_state_words[row][pair];
      const std::uint32_t upper_word = upper_state_words[row][pair];
      state_output[lower_row_offset + first_key] =
          static_cast<std::uint16_t>(lower_word);
      state_output[lower_row_offset + second_key] =
          static_cast<std::uint16_t>(lower_word >> 16U);
      state_output[upper_row_offset + first_key] =
          static_cast<std::uint16_t>(upper_word);
      state_output[upper_row_offset + second_key] =
          static_cast<std::uint16_t>(upper_word >> 16U);
    }
  }
}

}  // namespace

[[nodiscard]] int launch(
    const std::uint16_t* const conv_qkv, const std::size_t token_count,
    const std::uint16_t* const a, const std::uint16_t* const b,
    const std::uint16_t* const A_log, const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output, const float l2_epsilon,
    std::uint16_t* const output, void* const cuda_stream = nullptr) noexcept {
  if (token_count != kTokenCount || !std::isfinite(l2_epsilon) ||
      l2_epsilon <= 0.0F || conv_qkv == nullptr || a == nullptr ||
      b == nullptr || A_log == nullptr || dt_bias == nullptr ||
      state_input == nullptr || state_output == nullptr || output == nullptr ||
      invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gated_delta_net_update_c16_scalar_vector_qk_pingpong_kernel<<<
      static_cast<unsigned int>(kGdnValueHeadCount), kThreads, 0U, stream>>>(
      conv_qkv, a, b, A_log, dt_bias, state_input, state_output, l2_epsilon,
      output);
  return static_cast<int>(cudaGetLastError());
}

[[nodiscard]] int query_resources(
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
      gated_delta_net_update_c16_scalar_vector_qk_pingpong_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      gated_delta_net_update_c16_scalar_vector_qk_pingpong_kernel,
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

}  // namespace gdn_prefill_c16_pingpong_test_detail
}  // namespace q3x::runtime

namespace {

constexpr std::size_t kTokenCount = 16U;
constexpr std::size_t kOutputElements =
    kTokenCount * q3x::runtime::kGdnVElements;
constexpr float kL2Epsilon = 1.0e-6F;

class TestContext {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] bool cuda_ok(const cudaError_t status,
                             const std::string& operation) {
    expect(status == cudaSuccess,
           operation + ": " + cudaGetErrorString(status));
    return status == cudaSuccess;
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <typename T>
class ManagedBuffer {
 public:
  ManagedBuffer() = default;
  ManagedBuffer(const ManagedBuffer&) = delete;
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  ~ManagedBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t count) {
    count_ = count;
    return cudaMallocManaged(reinterpret_cast<void**>(&data_),
                             count * sizeof(T));
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] T& operator[](const std::size_t index) noexcept {
    return data_[index];
  }
  [[nodiscard]] const T& operator[](const std::size_t index) const noexcept {
    return data_[index];
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

void fill_inputs(ManagedBuffer<std::uint16_t>& conv_qkv,
                 ManagedBuffer<std::uint16_t>& a,
                 ManagedBuffer<std::uint16_t>& b,
                 ManagedBuffer<std::uint16_t>& A_log,
                 ManagedBuffer<std::uint16_t>& dt_bias) {
  constexpr std::size_t kKOffset = q3x::runtime::kGdnQElements;
  constexpr std::size_t kVOffset =
      q3x::runtime::kGdnQElements + q3x::runtime::kGdnKElements;
  for (std::size_t token = 0U; token < kTokenCount; ++token) {
    const std::size_t qkv_offset = token * q3x::runtime::kGdnQkvChannels;
    const std::size_t scalar_offset =
        token * q3x::runtime::kGdnValueHeadCount;
    for (std::size_t head = 0U;
         head < q3x::runtime::kGdnQkHeadCount; ++head) {
      for (std::size_t dimension = 0U;
           dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
        const int q_centered =
            static_cast<int>((head * 11U + dimension * 3U + token * 5U) %
                             29U) -
            14;
        const int k_centered =
            static_cast<int>((head * 7U + dimension * 5U + token * 2U) %
                             31U) -
            15;
        conv_qkv[qkv_offset +
                 head * q3x::runtime::kGdnHeadDimension + dimension] =
            encode_bf16(static_cast<float>(q_centered) / 16.0F);
        conv_qkv[qkv_offset + kKOffset +
                 head * q3x::runtime::kGdnHeadDimension + dimension] =
            encode_bf16(static_cast<float>(k_centered) / 16.0F);
      }
    }
    for (std::size_t index = 0U;
         index < q3x::runtime::kGdnVElements; ++index) {
      const int centered =
          static_cast<int>((index * 13U + token * 7U) % 37U) - 18;
      conv_qkv[qkv_offset + kVOffset + index] =
          encode_bf16(static_cast<float>(centered) / 16.0F);
    }
    for (std::size_t head = 0U;
         head < q3x::runtime::kGdnValueHeadCount; ++head) {
      a[scalar_offset + head] = encode_bf16(
          static_cast<float>(static_cast<int>((head + token) % 9U) - 4) *
          0.25F);
      b[scalar_offset + head] = encode_bf16(
          static_cast<float>(static_cast<int>((head + 2U * token) % 11U) -
                             5) *
          0.5F);
    }
  }
  for (std::size_t head = 0U;
       head < q3x::runtime::kGdnValueHeadCount; ++head) {
    A_log[head] = encode_bf16(
        -1.5F + static_cast<float>(head % 5U) * 0.375F);
    dt_bias[head] = encode_bf16(
        -0.75F + static_cast<float>(head % 7U) * 0.125F);
  }
  b[0U] = encode_bf16(20.0F);
  b[q3x::runtime::kGdnValueHeadCount + 1U] = encode_bf16(-20.0F);
  a[2U * q3x::runtime::kGdnValueHeadCount + 2U] = encode_bf16(25.0F);
  A_log[2U] = encode_bf16(4.0F);
}

void fill_state(std::uint16_t* const state, const bool inject_nans) {
  for (std::size_t index = 0U;
       index < q3x::runtime::kGdnStateElements; ++index) {
    const int centered = static_cast<int>((index * 5U) % 23U) - 11;
    state[index] = encode_bf16(static_cast<float>(centered) / 512.0F);
  }
  if (!inject_nans) {
    return;
  }
  constexpr std::array<std::uint16_t, 4U> kNanWords{
      0x7f81U, 0xff81U, 0x7fc1U, 0xffc1U};
  constexpr std::array<std::size_t, 4U> kRows{0U, 1U, 64U, 65U};
  constexpr std::array<std::size_t, 4U> kKeys{0U, 32U, 64U, 96U};
  for (std::size_t head = 0U;
       head < q3x::runtime::kGdnValueHeadCount; ++head) {
    const std::size_t head_offset =
        head * q3x::runtime::kGdnHeadDimension *
        q3x::runtime::kGdnHeadDimension;
    for (std::size_t item = 0U; item < kNanWords.size(); ++item) {
      state[head_offset + kRows[item] * q3x::runtime::kGdnHeadDimension +
            kKeys[item]] = kNanWords[item];
    }
  }
}

void expect_bitwise(TestContext& test, const std::uint16_t* const actual,
                    const std::uint16_t* const expected,
                    const std::size_t count, const std::string& label) {
  std::size_t mismatches = 0U;
  std::size_t first = count;
  for (std::size_t index = 0U; index < count; ++index) {
    if (actual[index] != expected[index]) {
      if (first == count) {
        first = index;
      }
      ++mismatches;
    }
  }
  test.expect(mismatches == 0U,
              label + " mismatches=" + std::to_string(mismatches) +
                  (first == count ? "" : " first=" + std::to_string(first)));
}

struct KernelTopology {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  std::size_t total_nodes = 0U;
  std::size_t kernel_nodes = 0U;
  cudaKernelNodeParams kernel{};

  KernelTopology() = default;
  KernelTopology(const KernelTopology&) = delete;
  KernelTopology& operator=(const KernelTopology&) = delete;

  ~KernelTopology() {
    if (executable != nullptr) {
      (void)cudaGraphExecDestroy(executable);
    }
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
  }
};

template <typename Launch>
void capture_one_kernel(TestContext& test, cudaStream_t stream,
                        const std::string& label, Launch&& launch,
                        KernelTopology& topology) {
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
      label + " begin capture");
  if (!ready) {
    return;
  }
  test.expect(static_cast<cudaError_t>(launch()) == cudaSuccess,
              label + " captured launch");
  ready = test.cuda_ok(cudaStreamEndCapture(stream, &topology.graph),
                       label + " end capture");
  if (!ready || topology.graph == nullptr) {
    return;
  }
  ready = test.cuda_ok(
      cudaGraphGetNodes(topology.graph, nullptr, &topology.total_nodes),
      label + " count nodes");
  if (!ready) {
    return;
  }
  std::vector<cudaGraphNode_t> nodes(topology.total_nodes);
  std::size_t copied = nodes.size();
  ready = test.cuda_ok(
      cudaGraphGetNodes(topology.graph, nodes.data(), &copied),
      label + " get nodes");
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    ready = test.cuda_ok(cudaGraphNodeGetType(node, &type),
                         label + " node type") && ready;
    if (type == cudaGraphNodeTypeKernel) {
      ++topology.kernel_nodes;
      ready = test.cuda_ok(cudaGraphKernelNodeGetParams(node, &topology.kernel),
                           label + " kernel params") && ready;
    }
  }
  test.expect(topology.total_nodes == 1U && topology.kernel_nodes == 1U,
              label + " captures exactly one kernel");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      cudaGraphInstantiate(&topology.executable, topology.graph, nullptr,
                           nullptr, 0U),
      label + " instantiate");
  if (ready) {
    ready = test.cuda_ok(cudaGraphLaunch(topology.executable, stream),
                         label + " replay");
    (void)(ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                 label + " replay synchronize"));
  }
}

template <typename Launch>
void expect_empty_invalid_capture(TestContext& test, cudaStream_t stream,
                                  const std::string& label,
                                  Launch&& launch) {
  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
      label + " begin capture");
  if (!ready) {
    return;
  }
  test.expect(static_cast<cudaError_t>(launch()) == cudaErrorInvalidValue,
              label + " rejects before enqueue");
  ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                       label + " end capture");
  if (ready && graph != nullptr) {
    std::size_t nodes = 0U;
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &nodes),
                         label + " count nodes");
    test.expect(ready && nodes == 0U, label + " captures zero nodes");
    (void)cudaGraphDestroy(graph);
  }
}

void test_resources(TestContext& test) {
  int candidate_registers = 0;
  std::size_t candidate_shared = 0U;
  std::size_t candidate_local = 0U;
  int candidate_threads = 0;
  int candidate_active = 0;
  bool ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::runtime::gdn_prefill_c16_pingpong_test_detail::
              query_resources(&candidate_registers, &candidate_shared,
                              &candidate_local, &candidate_threads,
                              &candidate_active)),
      "GDN C16 ping-pong query resources");

  int production_registers = 0;
  std::size_t production_shared = 0U;
  std::size_t production_local = 0U;
  int production_threads = 0;
  int production_active = 0;
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           query_gated_delta_net_update_warp_eight_row_register_state_m16_resources_test_cuda(
                               &production_registers, &production_shared,
                               &production_local, &production_threads,
                               &production_active)),
                       "GDN production C16 query resources");
  if (!ready) {
    return;
  }
  const bool candidate_gate =
      candidate_registers <= 64 && candidate_shared == 35200U &&
      candidate_local == 0U && candidate_threads == 256 &&
      candidate_active >= 4;
  const bool production_frozen =
      production_registers == 64 && production_shared == 34056U &&
      production_local == 0U && production_threads == 256 &&
      production_active == 4;
  std::cout << "GDN_C16_PINGPONG_RESOURCES: candidate_registers="
            << candidate_registers
            << " candidate_shared_bytes=" << candidate_shared
            << " candidate_local_bytes=" << candidate_local
            << " candidate_max_threads=" << candidate_threads
            << " candidate_active_blocks_per_sm=" << candidate_active
            << " production_registers=" << production_registers
            << " production_shared_bytes=" << production_shared
            << " production_local_bytes=" << production_local
            << " production_active_blocks_per_sm=" << production_active
            << " candidate_gate=" << (candidate_gate ? "PASS" : "FAIL")
            << " production_frozen="
            << (production_frozen ? "PASS" : "FAIL") << '\n';
  test.expect(candidate_gate, "GDN C16 ping-pong clears resource gate");
  test.expect(production_frozen, "GDN production C16 resources remain frozen");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::gdn_prefill_c16_pingpong_test_detail::
              query_resources(nullptr, &candidate_shared, &candidate_local,
                              &candidate_threads, &candidate_active)) ==
          cudaErrorInvalidValue,
      "GDN C16 ping-pong resource query rejects null output");
}

void test_correctness_and_contract(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kGuard = 17U;
  constexpr std::uint16_t kPrefix = 0xa55aU;
  constexpr std::uint16_t kSuffix = 0x5aa5U;
  constexpr std::uint16_t kPoison = 0x7fc1U;

  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> immutable_state;
  ManagedBuffer<std::uint16_t> production_state;
  ManagedBuffer<std::uint16_t> candidate_inplace_state;
  ManagedBuffer<std::uint16_t> candidate_input_state;
  ManagedBuffer<std::uint16_t> candidate_disjoint_state;
  ManagedBuffer<std::uint16_t> candidate_replay_state;
  ManagedBuffer<std::uint16_t> candidate_graph_state;
  ManagedBuffer<std::uint16_t> production_graph_state;
  ManagedBuffer<std::uint16_t> guarded_state_storage;
  ManagedBuffer<std::uint16_t> production_output;
  ManagedBuffer<std::uint16_t> candidate_inplace_output;
  ManagedBuffer<std::uint16_t> candidate_disjoint_output;
  ManagedBuffer<std::uint16_t> candidate_replay_output;
  ManagedBuffer<std::uint16_t> candidate_graph_output;
  ManagedBuffer<std::uint16_t> production_graph_output;
  ManagedBuffer<std::uint16_t> guarded_output_storage;

  bool ready = test.cuda_ok(
      conv_qkv.allocate(kTokenCount * q3x::runtime::kGdnQkvChannels),
      "GDN C16 ping-pong allocate conv QKV");
  ready = ready && test.cuda_ok(
                       a.allocate(kTokenCount *
                                  q3x::runtime::kGdnValueHeadCount),
                       "GDN C16 ping-pong allocate a");
  ready = ready && test.cuda_ok(
                       b.allocate(kTokenCount *
                                  q3x::runtime::kGdnValueHeadCount),
                       "GDN C16 ping-pong allocate b");
  ready = ready && test.cuda_ok(
                       A_log.allocate(q3x::runtime::kGdnValueHeadCount),
                       "GDN C16 ping-pong allocate A_log");
  ready = ready && test.cuda_ok(
                       dt_bias.allocate(q3x::runtime::kGdnValueHeadCount),
                       "GDN C16 ping-pong allocate dt_bias");
  const auto allocate_state = [&](ManagedBuffer<std::uint16_t>& buffer,
                                  const std::string& label) {
    return test.cuda_ok(buffer.allocate(q3x::runtime::kGdnStateElements),
                        label);
  };
  ready = ready && allocate_state(immutable_state, "allocate immutable state");
  ready = ready && allocate_state(production_state, "allocate production state");
  ready = ready &&
          allocate_state(candidate_inplace_state, "allocate in-place state");
  ready = ready &&
          allocate_state(candidate_input_state, "allocate disjoint input");
  ready = ready &&
          allocate_state(candidate_disjoint_state, "allocate disjoint output");
  ready = ready && allocate_state(candidate_replay_state, "allocate replay state");
  ready = ready && allocate_state(candidate_graph_state, "allocate graph state");
  ready = ready &&
          allocate_state(production_graph_state, "allocate production graph state");
  ready = ready && test.cuda_ok(
                       guarded_state_storage.allocate(
                           q3x::runtime::kGdnStateElements + 2U * kGuard),
                       "allocate guarded state");
  const auto allocate_output = [&](ManagedBuffer<std::uint16_t>& buffer,
                                   const std::string& label) {
    return test.cuda_ok(buffer.allocate(kOutputElements), label);
  };
  ready = ready && allocate_output(production_output, "allocate production output");
  ready = ready &&
          allocate_output(candidate_inplace_output, "allocate in-place output");
  ready = ready &&
          allocate_output(candidate_disjoint_output, "allocate disjoint output");
  ready = ready && allocate_output(candidate_replay_output, "allocate replay output");
  ready = ready && allocate_output(candidate_graph_output, "allocate graph output");
  ready = ready &&
          allocate_output(production_graph_output, "allocate production graph output");
  ready = ready && test.cuda_ok(
                       guarded_output_storage.allocate(kOutputElements +
                                                       2U * kGuard),
                       "allocate guarded output");
  if (!ready) {
    return;
  }

  fill_inputs(conv_qkv, a, b, A_log, dt_bias);
  fill_state(immutable_state.data(), true);
  const std::vector<std::uint16_t> original_conv(
      conv_qkv.data(), conv_qkv.data() + conv_qkv.size());
  const std::vector<std::uint16_t> original_a(a.data(), a.data() + a.size());
  const std::vector<std::uint16_t> original_b(b.data(), b.data() + b.size());
  const std::vector<std::uint16_t> original_A_log(
      A_log.data(), A_log.data() + A_log.size());
  const std::vector<std::uint16_t> original_dt_bias(
      dt_bias.data(), dt_bias.data() + dt_bias.size());
  const auto copy_initial = [&](ManagedBuffer<std::uint16_t>& buffer) {
    std::copy_n(immutable_state.data(), q3x::runtime::kGdnStateElements,
                buffer.data());
  };
  copy_initial(production_state);
  copy_initial(candidate_inplace_state);
  copy_initial(candidate_input_state);
  copy_initial(candidate_replay_state);
  std::fill_n(candidate_disjoint_state.data(), candidate_disjoint_state.size(),
              kPoison);
  std::fill_n(candidate_graph_state.data(), candidate_graph_state.size(),
              kPoison);
  std::fill_n(production_graph_state.data(), production_graph_state.size(),
              kPoison);
  std::fill_n(production_output.data(), production_output.size(), kPoison);
  std::fill_n(candidate_inplace_output.data(), candidate_inplace_output.size(),
              kPoison);
  std::fill_n(candidate_disjoint_output.data(),
              candidate_disjoint_output.size(), kPoison);
  std::fill_n(candidate_replay_output.data(), candidate_replay_output.size(),
              kPoison);
  std::fill_n(candidate_graph_output.data(), candidate_graph_output.size(),
              kPoison);
  std::fill_n(production_graph_output.data(), production_graph_output.size(),
              kPoison);
  std::fill_n(guarded_state_storage.data(), kGuard, kPrefix);
  std::copy_n(immutable_state.data(), q3x::runtime::kGdnStateElements,
              guarded_state_storage.data() + kGuard);
  std::fill_n(guarded_state_storage.data() + kGuard +
                  q3x::runtime::kGdnStateElements,
              kGuard, kSuffix);
  std::fill_n(guarded_output_storage.data(), kGuard, kPrefix);
  std::fill_n(guarded_output_storage.data() + kGuard, kOutputElements,
              kPoison);
  std::fill_n(guarded_output_storage.data() + kGuard + kOutputElements,
              kGuard, kSuffix);

  std::uint16_t* const guarded_state = guarded_state_storage.data() + kGuard;
  std::uint16_t* const guarded_output = guarded_output_storage.data() + kGuard;
  const auto launch_candidate =
      [&](const std::uint16_t* input_state, std::uint16_t* output_state,
          std::uint16_t* output, cudaStream_t target_stream) {
        return q3x::runtime::gdn_prefill_c16_pingpong_test_detail::launch(
            conv_qkv.data(), kTokenCount, a.data(), b.data(), A_log.data(),
            dt_bias.data(), input_state, output_state, kL2Epsilon, output,
            static_cast<void*>(target_stream));
      };
  const auto launch_production =
      [&](const std::uint16_t* input_state, std::uint16_t* output_state,
          std::uint16_t* output, cudaStream_t target_stream) {
        return q3x::runtime::launch_gated_delta_net_update_tile_warp_parallel_cuda(
            conv_qkv.data(), kTokenCount, a.data(), b.data(), A_log.data(),
            dt_bias.data(), input_state, output_state, kL2Epsilon, output, {},
            static_cast<void*>(target_stream));
      };

  // Seed a stale error: both valid launchers must clear it before enqueue.
  const cudaError_t stale =
      cudaMemcpy(nullptr, nullptr, 1U, cudaMemcpyHostToDevice);
  test.expect(stale == cudaErrorInvalidValue,
              "GDN C16 ping-pong seeds stale CUDA error");
  ready = test.cuda_ok(
      static_cast<cudaError_t>(launch_production(
          production_state.data(), production_state.data(),
          production_output.data(), stream)),
      "GDN C16 production launch");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate(
                           candidate_inplace_state.data(),
                           candidate_inplace_state.data(),
                           candidate_inplace_output.data(), stream)),
                       "GDN C16 ping-pong in-place launch");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate(
                           candidate_input_state.data(),
                           candidate_disjoint_state.data(),
                           candidate_disjoint_output.data(), stream)),
                       "GDN C16 ping-pong disjoint launch");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate(
                           immutable_state.data(),
                           candidate_replay_state.data(),
                           candidate_replay_output.data(), stream)),
                       "GDN C16 ping-pong replay launch");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate(
                           guarded_state, guarded_state, guarded_output,
                           stream)),
                       "GDN C16 ping-pong guarded launch");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "GDN C16 correctness synchronize");
  if (!ready) {
    return;
  }

  expect_bitwise(test, candidate_inplace_output.data(),
                 production_output.data(), kOutputElements,
                 "in-place output equals production");
  expect_bitwise(test, candidate_inplace_state.data(), production_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "in-place state equals production");
  expect_bitwise(test, candidate_disjoint_output.data(),
                 production_output.data(), kOutputElements,
                 "disjoint output equals production");
  expect_bitwise(test, candidate_disjoint_state.data(), production_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "disjoint state equals production");
  expect_bitwise(test, candidate_replay_output.data(),
                 production_output.data(), kOutputElements,
                 "replay output equals production");
  expect_bitwise(test, candidate_replay_state.data(), production_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "replay state equals production");
  expect_bitwise(test, guarded_output, production_output.data(),
                 kOutputElements, "guarded output equals production");
  expect_bitwise(test, guarded_state, production_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "guarded state equals production");
  expect_bitwise(test, candidate_input_state.data(), immutable_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "disjoint input remains immutable");

  const bool state_guards =
      std::all_of(guarded_state_storage.data(),
                  guarded_state_storage.data() + kGuard,
                  [](const std::uint16_t value) { return value == kPrefix; }) &&
      std::all_of(guarded_state_storage.data() + kGuard +
                      q3x::runtime::kGdnStateElements,
                  guarded_state_storage.data() +
                      guarded_state_storage.size(),
                  [](const std::uint16_t value) { return value == kSuffix; });
  const bool output_guards =
      std::all_of(guarded_output_storage.data(),
                  guarded_output_storage.data() + kGuard,
                  [](const std::uint16_t value) { return value == kPrefix; }) &&
      std::all_of(guarded_output_storage.data() + kGuard + kOutputElements,
                  guarded_output_storage.data() +
                      guarded_output_storage.size(),
                  [](const std::uint16_t value) { return value == kSuffix; });
  test.expect(state_guards && output_guards,
              "GDN C16 ping-pong preserves prefix/suffix guards");

  std::size_t input_positive_nan = 0U;
  std::size_t input_negative_nan = 0U;
  for (std::size_t index = 0U; index < immutable_state.size(); ++index) {
    const std::uint16_t word = immutable_state[index];
    if ((word & 0x7fffU) > 0x7f80U) {
      (word & 0x8000U) == 0U ? ++input_positive_nan : ++input_negative_nan;
    }
  }
  std::size_t positive_nan = 0U;
  std::size_t negative_nan = 0U;
  for (std::size_t index = 0U; index < production_state.size(); ++index) {
    const std::uint16_t word = production_state[index];
    if ((word & 0x7fffU) > 0x7f80U) {
      (word & 0x8000U) == 0U ? ++positive_nan : ++negative_nan;
    }
  }
  // The injected words cover both signs and all four BF16x2 half-word
  // positions. FP32 FMA is permitted to canonicalize the result's NaN sign;
  // the strict production/candidate word comparison above is the semantic
  // gate after that arithmetic boundary.
  test.expect(input_positive_nan == 96U && input_negative_nan == 96U &&
                  positive_nan + negative_nan > 0U,
              "NaN half-word stress covers both input signs and propagates");
  std::cout << "GDN_C16_PINGPONG_BITWISE: output_elements="
            << kOutputElements
            << " state_elements=" << q3x::runtime::kGdnStateElements
            << " input_positive_nan_words=" << input_positive_nan
            << " input_negative_nan_words=" << input_negative_nan
            << " positive_nan_state_words=" << positive_nan
            << " negative_nan_state_words=" << negative_nan
            << " in_place=true disjoint=true replay=true guards=true\n";

  KernelTopology candidate_topology;
  capture_one_kernel(
      test, stream, "GDN C16 ping-pong graph",
      [&]() {
        return launch_candidate(immutable_state.data(),
                                candidate_graph_state.data(),
                                candidate_graph_output.data(), stream);
      },
      candidate_topology);
  KernelTopology production_topology;
  capture_one_kernel(
      test, stream, "GDN C16 production graph",
      [&]() {
        return launch_production(immutable_state.data(),
                                 production_graph_state.data(),
                                 production_graph_output.data(), stream);
      },
      production_topology);
  if (candidate_topology.kernel_nodes == 1U) {
    test.expect(candidate_topology.kernel.gridDim.x ==
                        q3x::runtime::kGdnValueHeadCount &&
                    candidate_topology.kernel.gridDim.y == 1U &&
                    candidate_topology.kernel.gridDim.z == 1U &&
                    candidate_topology.kernel.blockDim.x == 256U &&
                    candidate_topology.kernel.blockDim.y == 1U &&
                    candidate_topology.kernel.blockDim.z == 1U &&
                    candidate_topology.kernel.sharedMemBytes == 0U,
                "candidate Graph locks 48x256 topology");
  }
  if (candidate_topology.kernel_nodes == 1U &&
      production_topology.kernel_nodes == 1U) {
    test.expect(candidate_topology.kernel.func != nullptr &&
                    production_topology.kernel.func != nullptr &&
                    candidate_topology.kernel.func !=
                        production_topology.kernel.func,
                "candidate Graph kernel remains distinct from production");
  }
  expect_bitwise(test, candidate_graph_output.data(), production_output.data(),
                 kOutputElements, "Graph replay output equals production");
  expect_bitwise(test, candidate_graph_state.data(), production_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "Graph replay state equals production");
  expect_bitwise(test, production_graph_output.data(), production_output.data(),
                 kOutputElements,
                 "production Graph replay output equals direct");
  expect_bitwise(test, production_graph_state.data(), production_state.data(),
                 q3x::runtime::kGdnStateElements,
                 "production Graph replay state equals direct");

  using q3x::runtime::gdn_prefill_c16_pingpong_test_detail::launch;
  const auto expect_invalid = [&](const int status, const std::string& label) {
    test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
                label);
  };
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount - 1U, a.data(), b.data(),
             A_log.data(), dt_bias.data(), immutable_state.data(),
             candidate_graph_state.data(), kL2Epsilon,
             candidate_graph_output.data(), static_cast<void*>(stream)),
      "rejects non-C16 token count");
  expect_invalid(
      launch(nullptr, kTokenCount, a.data(), b.data(), A_log.data(),
             dt_bias.data(), immutable_state.data(), candidate_graph_state.data(),
             kL2Epsilon, candidate_graph_output.data(),
             static_cast<void*>(stream)),
      "rejects null conv QKV");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, nullptr, b.data(), A_log.data(),
             dt_bias.data(), immutable_state.data(), candidate_graph_state.data(),
             kL2Epsilon, candidate_graph_output.data(),
             static_cast<void*>(stream)),
      "rejects null a");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, a.data(), nullptr, A_log.data(),
             dt_bias.data(), immutable_state.data(), candidate_graph_state.data(),
             kL2Epsilon, candidate_graph_output.data(),
             static_cast<void*>(stream)),
      "rejects null b");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, a.data(), b.data(), nullptr,
             dt_bias.data(), immutable_state.data(), candidate_graph_state.data(),
             kL2Epsilon, candidate_graph_output.data(),
             static_cast<void*>(stream)),
      "rejects null A_log");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, a.data(), b.data(), A_log.data(),
             nullptr, immutable_state.data(), candidate_graph_state.data(),
             kL2Epsilon, candidate_graph_output.data(),
             static_cast<void*>(stream)),
      "rejects null dt_bias");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, a.data(), b.data(), A_log.data(),
             dt_bias.data(), nullptr, candidate_graph_state.data(), kL2Epsilon,
             candidate_graph_output.data(), static_cast<void*>(stream)),
      "rejects null state input");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, a.data(), b.data(), A_log.data(),
             dt_bias.data(), immutable_state.data(), nullptr, kL2Epsilon,
             candidate_graph_output.data(), static_cast<void*>(stream)),
      "rejects null state output");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, a.data(), b.data(), A_log.data(),
             dt_bias.data(), immutable_state.data(), candidate_graph_state.data(),
             kL2Epsilon, nullptr, static_cast<void*>(stream)),
      "rejects null output");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, a.data(), b.data(), A_log.data(),
             dt_bias.data(), immutable_state.data(), candidate_graph_state.data(),
             0.0F, candidate_graph_output.data(), static_cast<void*>(stream)),
      "rejects zero epsilon");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, a.data(), b.data(), A_log.data(),
             dt_bias.data(), immutable_state.data(), candidate_graph_state.data(),
             std::numeric_limits<float>::quiet_NaN(),
             candidate_graph_output.data(), static_cast<void*>(stream)),
      "rejects NaN epsilon");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, a.data(), b.data(), A_log.data(),
             dt_bias.data(), immutable_state.data(), candidate_graph_state.data(),
             kL2Epsilon, conv_qkv.data(), static_cast<void*>(stream)),
      "rejects output/input alias");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, a.data(), b.data(), A_log.data(),
             dt_bias.data(), conv_qkv.data(), candidate_graph_state.data(),
             kL2Epsilon, candidate_graph_output.data(),
             static_cast<void*>(stream)),
      "rejects state/input alias");
  expect_invalid(
      launch(conv_qkv.data(), kTokenCount, a.data(), b.data(), A_log.data(),
             dt_bias.data(), immutable_state.data(), conv_qkv.data(),
             kL2Epsilon, candidate_graph_output.data(),
             static_cast<void*>(stream)),
      "rejects state output/input alias");
  expect_empty_invalid_capture(
      test, stream, "GDN C16 ping-pong invalid Graph",
      [&]() {
        return launch(conv_qkv.data(), kTokenCount - 1U, a.data(), b.data(),
                      A_log.data(), dt_bias.data(), immutable_state.data(),
                      candidate_graph_state.data(), kL2Epsilon,
                      candidate_graph_output.data(),
                      static_cast<void*>(stream));
      });

  expect_bitwise(test, conv_qkv.data(), original_conv.data(),
                 original_conv.size(), "conv QKV remains immutable");
  expect_bitwise(test, a.data(), original_a.data(), original_a.size(),
                 "a remains immutable");
  expect_bitwise(test, b.data(), original_b.data(), original_b.size(),
                 "b remains immutable");
  expect_bitwise(test, A_log.data(), original_A_log.data(),
                 original_A_log.size(), "A_log remains immutable");
  expect_bitwise(test, dt_bias.data(), original_dt_bias.data(),
                 original_dt_bias.size(), "dt_bias remains immutable");
  std::cout << "GDN_C16_PINGPONG_INVALID_CONTRACT: cases=14 graph_empty=true\n";
}

template <typename Launch>
[[nodiscard]] float measure_average_cuda_milliseconds(
    TestContext& test, cudaStream_t stream, const std::size_t warmups,
    const std::size_t iterations, const std::string& label,
    Launch&& launch) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ready = test.cuda_ok(cudaEventCreate(&start), label + " create start");
  ready = ready &&
          test.cuda_ok(cudaEventCreate(&stop), label + " create stop");
  if (!ready) {
    if (start != nullptr) {
      (void)cudaEventDestroy(start);
    }
    return std::numeric_limits<float>::quiet_NaN();
  }
  for (std::size_t iteration = 0U; iteration < warmups && ready; ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch()),
                         label + " warmup launch");
  }
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync");
  ready = ready &&
          test.cuda_ok(cudaEventRecord(start, stream), label + " record start");
  for (std::size_t iteration = 0U; iteration < iterations && ready;
       ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch()),
                         label + " measured launch");
  }
  ready = ready &&
          test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop");
  ready = ready &&
          test.cuda_ok(cudaEventSynchronize(stop), label + " sync stop");
  float elapsed = std::numeric_limits<float>::quiet_NaN();
  if (ready) {
    ready = test.cuda_ok(cudaEventElapsedTime(&elapsed, start, stop),
                         label + " elapsed");
  }
  (void)cudaEventDestroy(stop);
  (void)cudaEventDestroy(start);
  return ready ? elapsed / static_cast<float>(iterations)
               : std::numeric_limits<float>::quiet_NaN();
}

void run_optional_performance(TestContext& test, cudaStream_t stream) {
  const char* const value = std::getenv("Q3X_RUN_GDN_C16_PINGPONG_PERF");
  const bool enabled = value != nullptr && value[0] != '\0' &&
                       !(value[0] == '0' && value[1] == '\0');
  if (!enabled) {
    std::cout
        << "SKIP: GDN C16 ping-pong performance; set "
           "Q3X_RUN_GDN_C16_PINGPONG_PERF=1 to enable\n";
    return;
  }

  constexpr std::size_t kBankCount = 24U;
  constexpr std::size_t kWarmups = 48U;
  constexpr std::size_t kIterations = 480U;
  constexpr std::size_t kRounds = 6U;
  constexpr std::size_t kSamples = 2U * kRounds;
  constexpr std::size_t kStatePoolElements =
      kBankCount * q3x::runtime::kGdnStateElements;
  constexpr std::size_t kOutputPoolElements =
      kBankCount * kOutputElements;
  static_assert(kWarmups % kBankCount == 0U);
  static_assert(kIterations % kBankCount == 0U);

  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> immutable_states;
  ManagedBuffer<std::uint16_t> production_states;
  ManagedBuffer<std::uint16_t> candidate_states;
  ManagedBuffer<std::uint16_t> production_outputs;
  ManagedBuffer<std::uint16_t> candidate_outputs;
  bool ready = test.cuda_ok(
      conv_qkv.allocate(kTokenCount * q3x::runtime::kGdnQkvChannels),
      "perf allocate conv QKV");
  ready = ready && test.cuda_ok(
                       a.allocate(kTokenCount *
                                  q3x::runtime::kGdnValueHeadCount),
                       "perf allocate a");
  ready = ready && test.cuda_ok(
                       b.allocate(kTokenCount *
                                  q3x::runtime::kGdnValueHeadCount),
                       "perf allocate b");
  ready = ready && test.cuda_ok(
                       A_log.allocate(q3x::runtime::kGdnValueHeadCount),
                       "perf allocate A_log");
  ready = ready && test.cuda_ok(
                       dt_bias.allocate(q3x::runtime::kGdnValueHeadCount),
                       "perf allocate dt_bias");
  ready = ready && test.cuda_ok(immutable_states.allocate(kStatePoolElements),
                                "perf allocate immutable states");
  ready = ready && test.cuda_ok(production_states.allocate(kStatePoolElements),
                                "perf allocate production states");
  ready = ready && test.cuda_ok(candidate_states.allocate(kStatePoolElements),
                                "perf allocate candidate states");
  ready = ready &&
          test.cuda_ok(production_outputs.allocate(kOutputPoolElements),
                       "perf allocate production outputs");
  ready = ready &&
          test.cuda_ok(candidate_outputs.allocate(kOutputPoolElements),
                       "perf allocate candidate outputs");
  if (!ready) {
    return;
  }
  fill_inputs(conv_qkv, a, b, A_log, dt_bias);
  fill_state(immutable_states.data(), false);
  for (std::size_t bank = 1U; bank < kBankCount; ++bank) {
    std::copy_n(immutable_states.data(), q3x::runtime::kGdnStateElements,
                immutable_states.data() +
                    bank * q3x::runtime::kGdnStateElements);
  }

  const auto reset = [&](ManagedBuffer<std::uint16_t>& states,
                         ManagedBuffer<std::uint16_t>& outputs,
                         const std::string& label) {
    bool reset_ready = test.cuda_ok(
        cudaMemcpyAsync(states.data(), immutable_states.data(),
                        kStatePoolElements * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToDevice, stream),
        label + " reset state banks");
    reset_ready = reset_ready && test.cuda_ok(
                                     cudaMemsetAsync(
                                         outputs.data(), 0xff,
                                         kOutputPoolElements *
                                             sizeof(std::uint16_t),
                                         stream),
                                     label + " poison outputs");
    reset_ready = reset_ready && test.cuda_ok(
                                     cudaStreamSynchronize(stream),
                                     label + " reset synchronize");
    return reset_ready;
  };

  const auto measure_variant = [&](const bool candidate,
                                   const std::string& label) {
    auto& states = candidate ? candidate_states : production_states;
    auto& outputs = candidate ? candidate_outputs : production_outputs;
    if (!reset(states, outputs, label)) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    std::size_t launch_index = 0U;
    const float result = measure_average_cuda_milliseconds(
        test, stream, kWarmups, kIterations, label, [&]() {
          const std::size_t bank = launch_index % kBankCount;
          ++launch_index;
          std::uint16_t* const state =
              states.data() + bank * q3x::runtime::kGdnStateElements;
          std::uint16_t* const output =
              outputs.data() + bank * kOutputElements;
          if (candidate) {
            return q3x::runtime::gdn_prefill_c16_pingpong_test_detail::launch(
                conv_qkv.data(), kTokenCount, a.data(), b.data(),
                A_log.data(), dt_bias.data(), state, state, kL2Epsilon,
                output, static_cast<void*>(stream));
          }
          return q3x::runtime::
              launch_gated_delta_net_update_tile_warp_parallel_cuda(
                  conv_qkv.data(), kTokenCount, a.data(), b.data(),
                  A_log.data(), dt_bias.data(), state, state, kL2Epsilon,
                  output, {}, static_cast<void*>(stream));
        });
    test.expect(launch_index == kWarmups + kIterations,
                label + " rotates every requested launch");
    return result;
  };

  std::array<float, kSamples> production_samples{};
  std::array<float, kSamples> candidate_samples{};
  std::array<float, kRounds> round_speedups{};
  bool all_finite = true;
  bool bitwise = true;
  for (std::size_t round = 0U; round < kRounds; ++round) {
    const std::string label =
        "GDN C16 ping-pong round=" + std::to_string(round + 1U);
    const float b1 = measure_variant(false, label + " B1");
    const float c1 = measure_variant(true, label + " C1");
    const int failures_before_first = test.failures();
    expect_bitwise(test, candidate_states.data(), production_states.data(),
                   kStatePoolElements, label + " B1/C1 states");
    expect_bitwise(test, candidate_outputs.data(), production_outputs.data(),
                   kOutputPoolElements, label + " B1/C1 outputs");
    bitwise = bitwise && test.failures() == failures_before_first;
    const float c2 = measure_variant(true, label + " C2");
    const float b2 = measure_variant(false, label + " B2");
    const int failures_before_second = test.failures();
    expect_bitwise(test, candidate_states.data(), production_states.data(),
                   kStatePoolElements, label + " C2/B2 states");
    expect_bitwise(test, candidate_outputs.data(), production_outputs.data(),
                   kOutputPoolElements, label + " C2/B2 outputs");
    bitwise = bitwise && test.failures() == failures_before_second;

    production_samples[2U * round] = b1;
    production_samples[2U * round + 1U] = b2;
    candidate_samples[2U * round] = c1;
    candidate_samples[2U * round + 1U] = c2;
    const bool finite = std::isfinite(b1) && std::isfinite(b2) &&
                        std::isfinite(c1) && std::isfinite(c2) && b1 > 0.0F &&
                        b2 > 0.0F && c1 > 0.0F && c2 > 0.0F;
    all_finite = all_finite && finite;
    const float baseline_mean = (b1 + b2) * 0.5F;
    const float candidate_mean = (c1 + c2) * 0.5F;
    round_speedups[round] = baseline_mean / candidate_mean;
    std::cout << "PERF_GDN_C16_PINGPONG_ROUND: round=" << round + 1U
              << " baseline_pass1_ms=" << b1
              << " candidate_pass1_ms=" << c1
              << " candidate_pass2_ms=" << c2
              << " baseline_pass2_ms=" << b2
              << " baseline_mean_ms=" << baseline_mean
              << " candidate_mean_ms=" << candidate_mean
              << " speedup=" << round_speedups[round] << '\n';
  }

  const auto median = [](auto values) {
    std::sort(values.begin(), values.end());
    return (values[values.size() / 2U - 1U] +
            values[values.size() / 2U]) *
           0.5F;
  };
  const float baseline_median = median(production_samples);
  const float candidate_median = median(candidate_samples);
  const float separate_median_ratio = baseline_median / candidate_median;
  const float median_speedup = median(round_speedups);
  const float worst_round =
      *std::min_element(round_speedups.begin(), round_speedups.end());
  const bool all_rounds_faster =
      std::all_of(round_speedups.begin(), round_speedups.end(),
                  [](const float speedup) { return speedup > 1.0F; });
  const bool gate = all_finite && bitwise && all_rounds_faster &&
                    median_speedup >= 1.06F && worst_round >= 1.04F;
  std::cout << "PERF_GDN_C16_PINGPONG_AGGREGATE: baseline_median_ms="
            << baseline_median
            << " candidate_median_ms=" << candidate_median
            << " separate_median_ratio=" << separate_median_ratio
            << " median_speedup=" << median_speedup
            << " required_median_speedup=1.06"
            << " worst_round_speedup=" << worst_round
            << " required_worst_round_speedup=1.04"
            << " all_rounds_faster="
            << (all_rounds_faster ? "true" : "false")
            << " bitwise=" << (bitwise ? "true" : "false")
            << " state_banks=" << kBankCount
            << " rounds=" << kRounds
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "GDN C16 ping-pong clears frozen performance gate");
}

}  // namespace

int main() {
  TestContext test;
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: GDN C16 ping-pong requires a CUDA device\n";
    return 77;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                    "query CUDA device")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: GDN C16 ping-pong requires SM87; got sm_"
              << properties.major << properties.minor << '\n';
    return 77;
  }
  std::cout << "GDN_C16_PINGPONG_DEVICE: name=" << properties.name
            << " sm=" << properties.major << properties.minor << '\n';

  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create test stream")) {
    return 1;
  }
  test_resources(test);
  test_correctness_and_contract(test, stream);
  run_optional_performance(test, stream);
  (void)cudaStreamDestroy(stream);

  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " GDN C16 ping-pong assertion(s) failed\n";
    return 1;
  }
  std::cout << "GDN C16 scalar-vector Q/K ping-pong SM87 screen passed\n";
  return 0;
}
