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
#include <vector>

namespace {

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kWarpsPerBlock = kThreads / kWarpSize;
constexpr unsigned int kBlockTokens = 8U;
constexpr unsigned int kRowsPerWarp = 8U;
constexpr unsigned int kSubgroupWidth = 4U;
constexpr unsigned int kFullWarpMask = 0xffffffffU;
constexpr std::size_t kMaximumTokens = 512U;
constexpr float kL2Epsilon = 1.0e-6F;
constexpr std::array<std::size_t, 6U> kCorrectnessShapes{1U, 7U, 8U,
                                                         9U, 15U, 16U};

static_assert(kWarpsPerBlock * kRowsPerWarp * 2U ==
              q3x::runtime::kGdnHeadDimension);
static_assert(kWarpSize / kSubgroupWidth == kRowsPerWarp);

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

[[nodiscard]] std::uint16_t encode_bf16_host(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16_host(const std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
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

// Test-only B8 recurrence. The sequential specialization is the numerical
// oracle: it keeps one state row in FP32 for eight ordered recurrence steps.
// The WY specialization solves the equivalent lower-triangular B8 dependency
// system. It then replays the eight inexpensive elementwise state FMAs in
// token order; this preserves the sequential FP32 state-rounding order without
// repeating any state-vector dot product. Both publish a BF16 state boundary
// after every B8 block, including a final short tail.
template <bool kUseWy>
__launch_bounds__(kThreads, 1) __global__ void
gated_delta_net_update_b8_block_kernel(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    std::uint16_t* const output) {
  __shared__ float normalized_q[kBlockTokens]
                               [q3x::runtime::kGdnHeadDimension];
  __shared__ float normalized_k[kBlockTokens]
                               [q3x::runtime::kGdnHeadDimension];
  __shared__ float alpha[kBlockTokens];
  __shared__ float beta[kBlockTokens];
  __shared__ float prefix_decay[kBlockTokens];
  __shared__ float key_key[kBlockTokens][kBlockTokens];
  __shared__ float key_query[kBlockTokens][kBlockTokens];
  __shared__ float triangular[kBlockTokens][kBlockTokens];
  __shared__ float output_coefficient[kBlockTokens][kBlockTokens];

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const unsigned int row_in_warp = lane / kSubgroupWidth;
  const unsigned int subgroup_lane = lane % kSubgroupWidth;
  const std::size_t value_head = blockIdx.x;
  const std::size_t qk_head = value_head / 3U;
  constexpr std::size_t kKOffset = q3x::runtime::kGdnQElements;
  constexpr std::size_t kVOffset =
      q3x::runtime::kGdnQElements + q3x::runtime::kGdnKElements;
  constexpr unsigned int kItemsPerLane =
      q3x::runtime::kGdnHeadDimension / kSubgroupWidth;
  static_assert(kItemsPerLane == 32U);

#pragma unroll 1
  for (std::size_t block_start = 0U; block_start < token_count;
       block_start += kBlockTokens) {
    const std::size_t remaining_tokens = token_count - block_start;
    const unsigned int block_token_count = static_cast<unsigned int>(
        remaining_tokens < static_cast<std::size_t>(kBlockTokens)
            ? remaining_tokens
            : static_cast<std::size_t>(kBlockTokens));

    if (warp < block_token_count) {
      const std::size_t token = block_start + warp;
      const std::size_t qkv_token_offset =
          token * q3x::runtime::kGdnQkvChannels;
      float q_values[4];
      float k_values[4];
      float q_sum = 0.0F;
      float k_sum = 0.0F;
#pragma unroll
      for (unsigned int item = 0U; item < 4U; ++item) {
        const unsigned int dimension = lane + item * kWarpSize;
        q_values[item] = decode_bf16_device(
            conv_qkv[qkv_token_offset +
                     qk_head * q3x::runtime::kGdnHeadDimension + dimension]);
        k_values[item] = decode_bf16_device(
            conv_qkv[qkv_token_offset + kKOffset +
                     qk_head * q3x::runtime::kGdnHeadDimension + dimension]);
      }
      const float q_first_pair =
          q_values[0] * q_values[0] + q_values[2] * q_values[2];
      const float q_second_pair =
          q_values[1] * q_values[1] + q_values[3] * q_values[3];
      const float k_first_pair =
          k_values[0] * k_values[0] + k_values[2] * k_values[2];
      const float k_second_pair =
          k_values[1] * k_values[1] + k_values[3] * k_values[3];
      q_sum = q_first_pair + q_second_pair;
      k_sum = k_first_pair + k_second_pair;
#pragma unroll
      for (unsigned int stride = kWarpSize / 2U; stride != 0U;
           stride >>= 1U) {
        q_sum += __shfl_down_sync(kFullWarpMask, q_sum, stride);
        k_sum += __shfl_down_sync(kFullWarpMask, k_sum, stride);
      }
      const float q_scale =
          __shfl_sync(kFullWarpMask,
                      lane == 0U
                          ? rsqrtf(q_sum + l2_epsilon) *
                                rsqrtf(static_cast<float>(
                                    q3x::runtime::kGdnHeadDimension))
                          : 0.0F,
                      0U);
      const float k_scale =
          __shfl_sync(kFullWarpMask,
                      lane == 0U ? rsqrtf(k_sum + l2_epsilon) : 0.0F,
                      0U);
#pragma unroll
      for (unsigned int item = 0U; item < 4U; ++item) {
        const unsigned int dimension = lane + item * kWarpSize;
        normalized_q[warp][dimension] = q_values[item] * q_scale;
        normalized_k[warp][dimension] = k_values[item] * k_scale;
      }
      if (lane == 0U) {
        const std::size_t scalar_offset =
            token * q3x::runtime::kGdnValueHeadCount + value_head;
        const float gate_input = decode_bf16_device(a[scalar_offset]) +
                                 decode_bf16_device(dt_bias[value_head]);
        const float g =
            -expf(decode_bf16_device(A_log[value_head])) *
            stable_softplus_device(gate_input);
        alpha[warp] = expf(g);
        beta[warp] = stable_sigmoid_device(
            decode_bf16_device(b[scalar_offset]));
      }
    }
    __syncthreads();

    if constexpr (kUseWy) {
      if (thread < kBlockTokens * kBlockTokens) {
        const unsigned int target = thread / kBlockTokens;
        const unsigned int source = thread % kBlockTokens;
        if (target < block_token_count && source < block_token_count) {
          float kk = 0.0F;
          float kq = 0.0F;
#pragma unroll
          for (unsigned int dimension = 0U;
               dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
            const float source_key = normalized_k[source][dimension];
            kk = fmaf(source_key, normalized_k[target][dimension], kk);
            kq = fmaf(source_key, normalized_q[target][dimension], kq);
          }
          key_key[target][source] = kk;
          key_query[target][source] = kq;
        }
      }
      __syncthreads();
      if (thread == 0U) {
        float cumulative = 1.0F;
#pragma unroll
        for (unsigned int target = 0U; target < kBlockTokens; ++target) {
          if (target < block_token_count) {
            cumulative *= alpha[target];
            prefix_decay[target] = cumulative;
#pragma unroll
            for (unsigned int source = 0U; source < kBlockTokens; ++source) {
              triangular[target][source] = 0.0F;
              output_coefficient[target][source] = 0.0F;
              if (source <= target && source < block_token_count) {
                float decay = 1.0F;
#pragma unroll
                for (unsigned int inner = 0U; inner < kBlockTokens; ++inner) {
                  if (inner > source && inner <= target) {
                    decay *= alpha[inner];
                  }
                }
                if (source < target) {
                  triangular[target][source] =
                      decay * beta[source] * key_key[target][source];
                }
                output_coefficient[target][source] =
                    decay * beta[source] * key_query[target][source];
              }
            }
          }
        }
      }
      __syncthreads();
    }

    const std::uint16_t* const recurrent_state =
        block_start == 0U ? state_input : state_output;
    const std::size_t head_state_offset =
        value_head * q3x::runtime::kGdnHeadDimension *
        q3x::runtime::kGdnHeadDimension;

#pragma unroll
    for (unsigned int batch = 0U; batch < 2U; ++batch) {
      const std::size_t row =
          static_cast<std::size_t>(warp * kRowsPerWarp + row_in_warp) +
          static_cast<std::size_t>(batch) * 64U;
      const std::size_t state_row_offset =
          head_state_offset + row * q3x::runtime::kGdnHeadDimension;
      float state_items[kItemsPerLane];

      if constexpr (kUseWy) {
        float state_key[kBlockTokens];
        float state_query[kBlockTokens];
#pragma unroll
        for (unsigned int token = 0U; token < kBlockTokens; ++token) {
          state_key[token] = 0.0F;
          state_query[token] = 0.0F;
        }
#pragma unroll
        for (unsigned int item = 0U; item < kItemsPerLane; ++item) {
          const unsigned int dimension = subgroup_lane + item * kSubgroupWidth;
          const float state = decode_bf16_device(
              recurrent_state[state_row_offset + dimension]);
          state_items[item] = state;
#pragma unroll
          for (unsigned int token = 0U; token < kBlockTokens; ++token) {
            if (token < block_token_count) {
              state_key[token] =
                  fmaf(state, normalized_k[token][dimension], state_key[token]);
              state_query[token] = fmaf(state, normalized_q[token][dimension],
                                        state_query[token]);
            }
          }
        }
#pragma unroll
        for (unsigned int token = 0U; token < kBlockTokens; ++token) {
          if (token < block_token_count) {
            state_key[token] += __shfl_down_sync(
                kFullWarpMask, state_key[token], 2U, kSubgroupWidth);
            state_key[token] += __shfl_down_sync(
                kFullWarpMask, state_key[token], 1U, kSubgroupWidth);
            state_query[token] += __shfl_down_sync(
                kFullWarpMask, state_query[token], 2U, kSubgroupWidth);
            state_query[token] += __shfl_down_sync(
                kFullWarpMask, state_query[token], 1U, kSubgroupWidth);
          }
        }

        float delta[kBlockTokens];
#pragma unroll
        for (unsigned int token = 0U; token < kBlockTokens; ++token) {
          delta[token] = 0.0F;
        }
        if (subgroup_lane == 0U) {
#pragma unroll
          for (unsigned int target = 0U; target < kBlockTokens; ++target) {
            if (target < block_token_count) {
              const std::size_t value_offset =
                  (block_start + target) * q3x::runtime::kGdnQkvChannels +
                  kVOffset + value_head * q3x::runtime::kGdnHeadDimension +
                  row;
              float value = decode_bf16_device(conv_qkv[value_offset]);
              value = fmaf(-prefix_decay[target], state_key[target], value);
#pragma unroll
              for (unsigned int source = 0U; source < kBlockTokens; ++source) {
                if (source < target) {
                  value = fmaf(-triangular[target][source], delta[source],
                               value);
                }
              }
              delta[target] = value;

              float result = prefix_decay[target] * state_query[target];
#pragma unroll
              for (unsigned int source = 0U; source < kBlockTokens; ++source) {
                if (source <= target) {
                  result = fmaf(output_coefficient[target][source],
                                delta[source], result);
                }
              }
              const std::size_t output_offset =
                  (block_start + target) * q3x::runtime::kGdnVElements +
                  value_head * q3x::runtime::kGdnHeadDimension + row;
              output[output_offset] = encode_bf16_device(result);
            }
          }
        }
#pragma unroll
        for (unsigned int token = 0U; token < kBlockTokens; ++token) {
          if (token < block_token_count) {
            delta[token] = __shfl_sync(kFullWarpMask, delta[token], 0U,
                                       kSubgroupWidth);
          }
        }
#pragma unroll
        for (unsigned int item = 0U; item < kItemsPerLane; ++item) {
          const unsigned int dimension = subgroup_lane + item * kSubgroupWidth;
          float updated = state_items[item];
#pragma unroll
          for (unsigned int token = 0U; token < kBlockTokens; ++token) {
            if (token < block_token_count) {
              updated = fmaf(beta[token] * delta[token],
                             normalized_k[token][dimension],
                             alpha[token] * updated);
            }
          }
          state_output[state_row_offset + dimension] =
              encode_bf16_device(updated);
        }
      } else {
#pragma unroll
        for (unsigned int item = 0U; item < kItemsPerLane; ++item) {
          const unsigned int dimension = subgroup_lane + item * kSubgroupWidth;
          state_items[item] = decode_bf16_device(
              recurrent_state[state_row_offset + dimension]);
        }
#pragma unroll
        for (unsigned int token = 0U; token < kBlockTokens; ++token) {
          if (token < block_token_count) {
            float prediction = 0.0F;
#pragma unroll
            for (unsigned int item = 0U; item < kItemsPerLane; ++item) {
              const unsigned int dimension =
                  subgroup_lane + item * kSubgroupWidth;
              prediction = fmaf(alpha[token] * state_items[item],
                                normalized_k[token][dimension], prediction);
            }
            prediction += __shfl_down_sync(kFullWarpMask, prediction, 2U,
                                           kSubgroupWidth);
            prediction += __shfl_down_sync(kFullWarpMask, prediction, 1U,
                                           kSubgroupWidth);
            float delta = 0.0F;
            if (subgroup_lane == 0U) {
              const std::size_t value_offset =
                  (block_start + token) * q3x::runtime::kGdnQkvChannels +
                  kVOffset + value_head * q3x::runtime::kGdnHeadDimension +
                  row;
              delta =
                  (decode_bf16_device(conv_qkv[value_offset]) - prediction) *
                  beta[token];
            }
            delta = __shfl_sync(kFullWarpMask, delta, 0U, kSubgroupWidth);
            float result = 0.0F;
#pragma unroll
            for (unsigned int item = 0U; item < kItemsPerLane; ++item) {
              const unsigned int dimension =
                  subgroup_lane + item * kSubgroupWidth;
              state_items[item] =
                  fmaf(delta, normalized_k[token][dimension],
                       alpha[token] * state_items[item]);
              result = fmaf(state_items[item], normalized_q[token][dimension],
                            result);
            }
            result += __shfl_down_sync(kFullWarpMask, result, 2U,
                                       kSubgroupWidth);
            result += __shfl_down_sync(kFullWarpMask, result, 1U,
                                       kSubgroupWidth);
            if (subgroup_lane == 0U) {
              const std::size_t output_offset =
                  (block_start + token) * q3x::runtime::kGdnVElements +
                  value_head * q3x::runtime::kGdnHeadDimension + row;
              output[output_offset] = encode_bf16_device(result);
            }
          }
        }
#pragma unroll
        for (unsigned int item = 0U; item < kItemsPerLane; ++item) {
          const unsigned int dimension = subgroup_lane + item * kSubgroupWidth;
          state_output[state_row_offset + dimension] =
              encode_bf16_device(state_items[item]);
        }
      }
    }
    // The next B8 block reads this CTA's just-published BF16 head state.
    __syncthreads();
  }
}

[[nodiscard]] bool invalid_alias(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    const std::uint16_t* const state_output,
    const std::uint16_t* const output) noexcept {
  return output == conv_qkv || output == a || output == b ||
         output == A_log || output == dt_bias || output == state_input ||
         output == state_output || state_input == conv_qkv ||
         state_input == a || state_input == b || state_input == A_log ||
         state_input == dt_bias || state_output == conv_qkv ||
         state_output == a || state_output == b || state_output == A_log ||
         state_output == dt_bias;
}

template <bool kUseWy>
[[nodiscard]] int launch_b8_block(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    std::uint16_t* const output,
    cudaStream_t stream) noexcept {
  if (token_count == 0U || token_count > kMaximumTokens ||
      !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
      conv_qkv == nullptr || a == nullptr || b == nullptr ||
      A_log == nullptr || dt_bias == nullptr || state_input == nullptr ||
      state_output == nullptr || output == nullptr ||
      invalid_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                    state_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  (void)cudaGetLastError();
  gated_delta_net_update_b8_block_kernel<kUseWy><<<
      static_cast<unsigned int>(q3x::runtime::kGdnValueHeadCount), kThreads,
      0U, stream>>>(conv_qkv, token_count, a, b, A_log, dt_bias, state_input,
                    state_output, l2_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

struct KernelResources {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

template <bool kUseWy>
[[nodiscard]] cudaError_t query_resources(KernelResources& resources) {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, gated_delta_net_update_b8_block_kernel<kUseWy>);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, gated_delta_net_update_b8_block_kernel<kUseWy>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return status;
  }
  resources.registers_per_thread = attributes.numRegs;
  resources.static_shared_bytes = attributes.sharedSizeBytes;
  resources.local_bytes = attributes.localSizeBytes;
  resources.maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources.active_blocks_per_sm = active_blocks;
  return cudaSuccess;
}

void fill_inputs(ManagedBuffer<std::uint16_t>& conv_qkv,
                 ManagedBuffer<std::uint16_t>& a,
                 ManagedBuffer<std::uint16_t>& b,
                 ManagedBuffer<std::uint16_t>& A_log,
                 ManagedBuffer<std::uint16_t>& dt_bias,
                 const std::size_t token_count) {
  constexpr std::size_t kKOffset = q3x::runtime::kGdnQElements;
  constexpr std::size_t kVOffset =
      q3x::runtime::kGdnQElements + q3x::runtime::kGdnKElements;
  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t qkv_offset = token * q3x::runtime::kGdnQkvChannels;
    const std::size_t scalar_offset =
        token * q3x::runtime::kGdnValueHeadCount;
    for (std::size_t head = 0U;
         head < q3x::runtime::kGdnQkHeadCount; ++head) {
      for (std::size_t dimension = 0U;
           dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
        const int q_centered = static_cast<int>(
                                   (head * 11U + dimension * 3U +
                                    token * 5U) %
                                   29U) -
                               14;
        const int k_centered = static_cast<int>(
                                   (head * 7U + dimension * 5U +
                                    token * 2U) %
                                   31U) -
                               15;
        conv_qkv[qkv_offset +
                 head * q3x::runtime::kGdnHeadDimension + dimension] =
            encode_bf16_host(static_cast<float>(q_centered) / 16.0F);
        conv_qkv[qkv_offset + kKOffset +
                 head * q3x::runtime::kGdnHeadDimension + dimension] =
            encode_bf16_host(static_cast<float>(k_centered) / 16.0F);
      }
    }
    for (std::size_t index = 0U;
         index < q3x::runtime::kGdnVElements; ++index) {
      const int centered =
          static_cast<int>((index * 13U + token * 7U) % 37U) - 18;
      conv_qkv[qkv_offset + kVOffset + index] =
          encode_bf16_host(static_cast<float>(centered) / 16.0F);
    }
    for (std::size_t head = 0U;
         head < q3x::runtime::kGdnValueHeadCount; ++head) {
      a[scalar_offset + head] = encode_bf16_host(
          static_cast<float>(static_cast<int>((head + token) % 9U) - 4) *
          0.25F);
      b[scalar_offset + head] = encode_bf16_host(
          static_cast<float>(
              static_cast<int>((head + 2U * token) % 11U) - 5) *
          0.5F);
    }
  }
  for (std::size_t head = 0U;
       head < q3x::runtime::kGdnValueHeadCount; ++head) {
    A_log[head] = encode_bf16_host(
        -1.5F + static_cast<float>(head % 5U) * 0.375F);
    dt_bias[head] = encode_bf16_host(
        -0.75F + static_cast<float>(head % 7U) * 0.125F);
  }
  b[0] = encode_bf16_host(20.0F);
  if (token_count > 1U) {
    b[q3x::runtime::kGdnValueHeadCount + 1U] =
        encode_bf16_host(-20.0F);
  }
  if (token_count > 2U) {
    a[2U * q3x::runtime::kGdnValueHeadCount + 2U] =
        encode_bf16_host(25.0F);
  }
  if (token_count > 16U) {
    b[15U * q3x::runtime::kGdnValueHeadCount + 3U] =
        encode_bf16_host(20.0F);
    b[16U * q3x::runtime::kGdnValueHeadCount + 4U] =
        encode_bf16_host(-20.0F);
  }
  if (token_count > 256U) {
    a[255U * q3x::runtime::kGdnValueHeadCount + 5U] =
        encode_bf16_host(25.0F);
    b[256U * q3x::runtime::kGdnValueHeadCount + 6U] =
        encode_bf16_host(-20.0F);
  }
  A_log[2] = encode_bf16_host(4.0F);
}

void fill_state(std::uint16_t* const state,
                const std::size_t state_count,
                const std::size_t salt = 0U) {
  for (std::size_t index = 0U; index < state_count; ++index) {
    const int centered =
        static_cast<int>((index * 5U + salt * 7U) % 23U) - 11;
    state[index] =
        encode_bf16_host(static_cast<float>(centered) / 512.0F);
  }
}

[[nodiscard]] float stable_softplus_host(const float value) {
  return value > 20.0F ? value : std::log1p(std::exp(value));
}

[[nodiscard]] float stable_sigmoid_host(const float value) {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-value));
  }
  const float exponential = std::exp(value);
  return exponential / (1.0F + exponential);
}

struct CpuB8BlockReference {
  std::vector<std::uint16_t> output;
  std::array<std::vector<std::uint16_t>, kCorrectnessShapes.size()>
      state_snapshots;
};

// Independent host oracle for the selected sequential FP32-B8 semantics. It
// uses the public logical layout and scalar recurrence directly, not either
// CUDA specialization above. State remains FP32 inside each B8 block and is
// rounded to BF16 at every complete block or requested final tail.
[[nodiscard]] CpuB8BlockReference build_cpu_b8_block_reference(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const initial_state,
    const std::size_t token_count) {
  constexpr std::size_t kKOffset = q3x::runtime::kGdnQElements;
  constexpr std::size_t kVOffset =
      q3x::runtime::kGdnQElements + q3x::runtime::kGdnKElements;
  const float inverse_sqrt_dimension =
      1.0F /
      std::sqrt(static_cast<float>(q3x::runtime::kGdnHeadDimension));
  std::vector<float> normalized_q(token_count *
                                  q3x::runtime::kGdnQElements);
  std::vector<float> normalized_k(token_count *
                                  q3x::runtime::kGdnKElements);
  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t qkv_token_offset =
        token * q3x::runtime::kGdnQkvChannels;
    const std::size_t normalized_token_offset =
        token * q3x::runtime::kGdnQElements;
    for (std::size_t head = 0U;
         head < q3x::runtime::kGdnQkHeadCount; ++head) {
      const std::size_t head_offset =
          head * q3x::runtime::kGdnHeadDimension;
      float q_sum = 0.0F;
      float k_sum = 0.0F;
      for (std::size_t dimension = 0U;
           dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
        const float q_value = decode_bf16_host(
            conv_qkv[qkv_token_offset + head_offset + dimension]);
        const float k_value = decode_bf16_host(
            conv_qkv[qkv_token_offset + kKOffset + head_offset + dimension]);
        q_sum = std::fma(q_value, q_value, q_sum);
        k_sum = std::fma(k_value, k_value, k_sum);
      }
      const float q_scale =
          1.0F / std::sqrt(q_sum + kL2Epsilon) * inverse_sqrt_dimension;
      const float k_scale = 1.0F / std::sqrt(k_sum + kL2Epsilon);
      for (std::size_t dimension = 0U;
           dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
        normalized_q[normalized_token_offset + head_offset + dimension] =
            decode_bf16_host(
                conv_qkv[qkv_token_offset + head_offset + dimension]) *
            q_scale;
        normalized_k[normalized_token_offset + head_offset + dimension] =
            decode_bf16_host(conv_qkv[qkv_token_offset + kKOffset +
                                      head_offset + dimension]) *
            k_scale;
      }
    }
  }

  CpuB8BlockReference reference{};
  reference.output.resize(token_count * q3x::runtime::kGdnVElements);
  std::vector<float> state(q3x::runtime::kGdnStateElements);
  for (std::size_t index = 0U; index < state.size(); ++index) {
    state[index] = decode_bf16_host(initial_state[index]);
  }
  std::size_t next_snapshot = 0U;
  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t qkv_token_offset =
        token * q3x::runtime::kGdnQkvChannels;
    const std::size_t scalar_token_offset =
        token * q3x::runtime::kGdnValueHeadCount;
    const std::size_t normalized_token_offset =
        token * q3x::runtime::kGdnQElements;
    const std::size_t output_token_offset =
        token * q3x::runtime::kGdnVElements;
    for (std::size_t value_head = 0U;
         value_head < q3x::runtime::kGdnValueHeadCount; ++value_head) {
      const std::size_t qk_head = value_head / 3U;
      const std::size_t normalized_head_offset =
          normalized_token_offset +
          qk_head * q3x::runtime::kGdnHeadDimension;
      const float gate_input =
          decode_bf16_host(a[scalar_token_offset + value_head]) +
          decode_bf16_host(dt_bias[value_head]);
      const float g = -std::exp(decode_bf16_host(A_log[value_head])) *
                      stable_softplus_host(gate_input);
      const float alpha = std::exp(g);
      const float beta = stable_sigmoid_host(
          decode_bf16_host(b[scalar_token_offset + value_head]));
      const std::size_t state_head_offset =
          value_head * q3x::runtime::kGdnHeadDimension *
          q3x::runtime::kGdnHeadDimension;
      const std::size_t value_offset =
          qkv_token_offset + kVOffset +
          value_head * q3x::runtime::kGdnHeadDimension;
      const std::size_t output_head_offset =
          output_token_offset +
          value_head * q3x::runtime::kGdnHeadDimension;
      for (std::size_t row = 0U;
           row < q3x::runtime::kGdnHeadDimension; ++row) {
        const std::size_t state_row_offset =
            state_head_offset + row * q3x::runtime::kGdnHeadDimension;
        float prediction = 0.0F;
        for (std::size_t dimension = 0U;
             dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
          prediction = std::fma(
              alpha * state[state_row_offset + dimension],
              normalized_k[normalized_head_offset + dimension], prediction);
        }
        const float delta =
            (decode_bf16_host(conv_qkv[value_offset + row]) - prediction) *
            beta;
        float result = 0.0F;
        for (std::size_t dimension = 0U;
             dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
          const std::size_t state_index = state_row_offset + dimension;
          const float updated = std::fma(
              delta, normalized_k[normalized_head_offset + dimension],
              alpha * state[state_index]);
          state[state_index] = updated;
          result = std::fma(
              updated, normalized_q[normalized_head_offset + dimension],
              result);
        }
        reference.output[output_head_offset + row] =
            encode_bf16_host(result);
      }
    }

    const std::size_t completed_tokens = token + 1U;
    if (next_snapshot < kCorrectnessShapes.size() &&
        completed_tokens == kCorrectnessShapes[next_snapshot]) {
      std::vector<std::uint16_t>& snapshot =
          reference.state_snapshots[next_snapshot];
      snapshot.resize(state.size());
      for (std::size_t index = 0U; index < state.size(); ++index) {
        snapshot[index] = encode_bf16_host(state[index]);
      }
      ++next_snapshot;
      if (completed_tokens % kBlockTokens == 0U &&
          completed_tokens < token_count) {
        for (std::size_t index = 0U; index < state.size(); ++index) {
          state[index] = decode_bf16_host(snapshot[index]);
        }
      }
    }
  }
  return reference;
}

struct ErrorMetrics {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
  double p99_absolute = 0.0;
  double p99_relative = 0.0;
  double nrmse = 0.0;
  double cosine = 0.0;
  std::size_t non_finite = 0U;
};

[[nodiscard]] ErrorMetrics calculate_error_metrics(
    const std::uint16_t* const actual,
    const std::uint16_t* const reference,
    const std::size_t count) {
  constexpr double kRelativeFloor = 1.0e-2;
  ErrorMetrics metrics{};
  std::vector<double> absolute_errors;
  std::vector<double> relative_errors;
  absolute_errors.reserve(count);
  relative_errors.reserve(count);
  double squared_error = 0.0;
  double squared_reference = 0.0;
  double dot = 0.0;
  double squared_actual = 0.0;
  for (std::size_t index = 0U; index < count; ++index) {
    const double actual_value = decode_bf16_host(actual[index]);
    const double reference_value = decode_bf16_host(reference[index]);
    if (!std::isfinite(actual_value) || !std::isfinite(reference_value)) {
      ++metrics.non_finite;
      continue;
    }
    const double absolute = std::abs(actual_value - reference_value);
    const double relative =
        absolute / std::max(std::abs(reference_value), kRelativeFloor);
    metrics.maximum_absolute = std::max(metrics.maximum_absolute, absolute);
    metrics.maximum_relative = std::max(metrics.maximum_relative, relative);
    absolute_errors.push_back(absolute);
    relative_errors.push_back(relative);
    squared_error += absolute * absolute;
    squared_reference += reference_value * reference_value;
    squared_actual += actual_value * actual_value;
    dot += actual_value * reference_value;
  }
  if (!absolute_errors.empty()) {
    const std::size_t p99_index =
        std::min(absolute_errors.size() - 1U,
                 static_cast<std::size_t>(
                     std::ceil(0.99 * static_cast<double>(
                                          absolute_errors.size()))) -
                     1U);
    std::nth_element(absolute_errors.begin(),
                     absolute_errors.begin() + p99_index,
                     absolute_errors.end());
    std::nth_element(relative_errors.begin(),
                     relative_errors.begin() + p99_index,
                     relative_errors.end());
    metrics.p99_absolute = absolute_errors[p99_index];
    metrics.p99_relative = relative_errors[p99_index];
  }
  metrics.nrmse =
      std::sqrt(squared_error /
                std::max(squared_reference, std::numeric_limits<double>::min()));
  metrics.cosine =
      dot / std::sqrt(std::max(
                squared_actual * squared_reference,
                std::numeric_limits<double>::min()));
  return metrics;
}

void print_metrics(const std::string& reference_name,
                   const std::string& tensor_name,
                   const std::size_t token_count,
                   const ErrorMetrics& metrics) {
  std::cout << "GDN_B8_BLOCK_NUMERICS: reference=" << reference_name
            << " tensor=" << tensor_name
            << " token_count=" << token_count
            << " max_abs=" << metrics.maximum_absolute
            << " max_rel_floor_1e-2=" << metrics.maximum_relative
            << " p99_abs=" << metrics.p99_absolute
            << " p99_rel_floor_1e-2=" << metrics.p99_relative
            << " nrmse=" << metrics.nrmse
            << " cosine=" << metrics.cosine
            << " non_finite=" << metrics.non_finite << '\n';
}

[[nodiscard]] bool clears_output_numerics_gate(const ErrorMetrics& metrics) {
  return metrics.non_finite == 0U && metrics.maximum_absolute <= 1.0e-4 &&
         metrics.maximum_relative <= 1.0e-2 &&
         metrics.p99_relative <= 1.0e-2 && metrics.nrmse <= 1.0e-4 &&
         metrics.cosine >= 0.999999;
}

[[nodiscard]] bool clears_state_numerics_gate(const ErrorMetrics& metrics) {
  // Near-zero BF16 values make a maximum relative error unstable: one ULP at
  // 0.04 already exceeds one percent. Keep reporting that maximum, but freeze
  // the state gate on absolute ULP-scale tails plus global normalized error.
  return metrics.non_finite == 0U &&
         metrics.maximum_absolute <= 0.001953125 &&
         metrics.p99_absolute <= 0.00048828125 &&
         metrics.p99_relative <= 2.5e-2 && metrics.nrmse <= 1.0e-4 &&
         metrics.cosine >= 0.999999;
}

[[nodiscard]] bool clears_cpu_reference_gate(const ErrorMetrics& metrics) {
  return metrics.non_finite == 0U &&
         metrics.maximum_absolute <= 0.00390625 &&
         metrics.p99_absolute <= 0.0009765625 &&
         metrics.p99_relative <= 2.5e-2 && metrics.nrmse <= 1.0e-2 &&
         metrics.cosine >= 0.9999;
}

[[nodiscard]] bool launch_production_chain(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    std::uint16_t* const output,
    cudaStream_t stream) {
  constexpr std::size_t kM16 =
      q3x::runtime::kGdnMaximumTileTokenCount;
  if (token_count == 0U || token_count % kM16 != 0U) {
    return false;
  }
  const std::uint16_t* recurrence_input = state_input;
  for (std::size_t token_offset = 0U; token_offset < token_count;
       token_offset += kM16) {
    const int status =
        q3x::runtime::launch_gated_delta_net_update_tile_warp_parallel_cuda(
            conv_qkv + token_offset * q3x::runtime::kGdnQkvChannels, kM16,
            a + token_offset * q3x::runtime::kGdnValueHeadCount,
            b + token_offset * q3x::runtime::kGdnValueHeadCount, A_log,
            dt_bias, recurrence_input, state_output, kL2Epsilon,
            output + token_offset * q3x::runtime::kGdnVElements, {},
            static_cast<void*>(stream));
    if (status != static_cast<int>(cudaSuccess)) {
      return false;
    }
    recurrence_input = state_output;
  }
  return true;
}

void run_correctness(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kCorrectnessTokens = 16U;
  const std::size_t maximum_output_elements =
      kCorrectnessTokens * q3x::runtime::kGdnVElements;
  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> initial_state;
  ManagedBuffer<std::uint16_t> oracle_state;
  ManagedBuffer<std::uint16_t> candidate_state;
  ManagedBuffer<std::uint16_t> production_state;
  ManagedBuffer<std::uint16_t> split_state;
  ManagedBuffer<std::uint16_t> oracle_output;
  ManagedBuffer<std::uint16_t> candidate_output;
  ManagedBuffer<std::uint16_t> production_output;
  ManagedBuffer<std::uint16_t> split_output;
  bool ready = test.cuda_ok(
      conv_qkv.allocate(kCorrectnessTokens *
                        q3x::runtime::kGdnQkvChannels),
      "B8 block correctness allocate conv QKV");
  ready = ready && test.cuda_ok(
                       a.allocate(kCorrectnessTokens *
                                  q3x::runtime::kGdnValueHeadCount),
                       "B8 block correctness allocate a");
  ready = ready && test.cuda_ok(
                       b.allocate(kCorrectnessTokens *
                                  q3x::runtime::kGdnValueHeadCount),
                       "B8 block correctness allocate b");
  ready = ready && test.cuda_ok(
                       A_log.allocate(q3x::runtime::kGdnValueHeadCount),
                       "B8 block correctness allocate A_log");
  ready = ready && test.cuda_ok(
                       dt_bias.allocate(q3x::runtime::kGdnValueHeadCount),
                       "B8 block correctness allocate dt_bias");
  for (ManagedBuffer<std::uint16_t>* const buffer :
       {&initial_state, &oracle_state, &candidate_state, &production_state,
        &split_state}) {
    ready = ready && test.cuda_ok(
                         buffer->allocate(q3x::runtime::kGdnStateElements),
                         "B8 block correctness allocate state");
  }
  for (ManagedBuffer<std::uint16_t>* const buffer :
       {&oracle_output, &candidate_output, &production_output, &split_output}) {
    ready = ready && test.cuda_ok(buffer->allocate(maximum_output_elements),
                                  "B8 block correctness allocate output");
  }
  if (!ready) {
    return;
  }
  fill_inputs(conv_qkv, a, b, A_log, dt_bias, kCorrectnessTokens);
  fill_state(initial_state.data(), initial_state.size());
  const std::vector<std::uint16_t> frozen_conv_qkv(
      conv_qkv.data(), conv_qkv.data() + conv_qkv.size());
  const std::vector<std::uint16_t> frozen_a(a.data(), a.data() + a.size());
  const std::vector<std::uint16_t> frozen_b(b.data(), b.data() + b.size());
  const std::vector<std::uint16_t> frozen_A_log(
      A_log.data(), A_log.data() + A_log.size());
  const std::vector<std::uint16_t> frozen_dt_bias(
      dt_bias.data(), dt_bias.data() + dt_bias.size());
  const CpuB8BlockReference cpu_reference = build_cpu_b8_block_reference(
      conv_qkv.data(), a.data(), b.data(), A_log.data(), dt_bias.data(),
      initial_state.data(), kCorrectnessTokens);

  for (std::size_t shape_index = 0U;
       shape_index < kCorrectnessShapes.size(); ++shape_index) {
    const std::size_t token_count = kCorrectnessShapes[shape_index];
    std::copy_n(initial_state.data(), initial_state.size(),
                oracle_state.data());
    std::copy_n(initial_state.data(), initial_state.size(),
                candidate_state.data());
    std::fill_n(oracle_output.data(), oracle_output.size(),
                static_cast<std::uint16_t>(0x7fc1U));
    std::fill_n(candidate_output.data(), candidate_output.size(),
                static_cast<std::uint16_t>(0x7fc1U));
    ready = test.cuda_ok(
        static_cast<cudaError_t>(launch_b8_block<false>(
            conv_qkv.data(), token_count, a.data(), b.data(), A_log.data(),
            dt_bias.data(), oracle_state.data(), oracle_state.data(),
            kL2Epsilon, oracle_output.data(), stream)),
        "B8 sequential oracle launch C" + std::to_string(token_count));
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_b8_block<true>(
                             conv_qkv.data(), token_count, a.data(), b.data(),
                             A_log.data(), dt_bias.data(),
                             candidate_state.data(), candidate_state.data(),
                             kL2Epsilon, candidate_output.data(), stream)),
                         "B8 WY candidate launch C" +
                             std::to_string(token_count));
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         "B8 block correctness synchronize C" +
                             std::to_string(token_count));
    if (!ready) {
      return;
    }
    const ErrorMetrics output_metrics = calculate_error_metrics(
        candidate_output.data(), oracle_output.data(),
        token_count * q3x::runtime::kGdnVElements);
    const ErrorMetrics state_metrics = calculate_error_metrics(
        candidate_state.data(), oracle_state.data(),
        q3x::runtime::kGdnStateElements);
    const ErrorMetrics cpu_output_metrics = calculate_error_metrics(
        oracle_output.data(), cpu_reference.output.data(),
        token_count * q3x::runtime::kGdnVElements);
    const ErrorMetrics cpu_state_metrics = calculate_error_metrics(
        oracle_state.data(),
        cpu_reference.state_snapshots[shape_index].data(),
        q3x::runtime::kGdnStateElements);
    print_metrics("fp32-sequential-b8", "output", token_count,
                  output_metrics);
    print_metrics("fp32-sequential-b8", "state", token_count,
                  state_metrics);
    print_metrics("independent-cpu-fp32-b8", "sequential-output",
                  token_count, cpu_output_metrics);
    print_metrics("independent-cpu-fp32-b8", "sequential-state",
                  token_count, cpu_state_metrics);
    test.expect(clears_output_numerics_gate(output_metrics),
                "B8 WY output clears frozen block numerics gate C" +
                    std::to_string(token_count));
    test.expect(clears_state_numerics_gate(state_metrics),
                "B8 WY state clears frozen block numerics gate C" +
                    std::to_string(token_count));
    test.expect(clears_cpu_reference_gate(cpu_output_metrics),
                "sequential B8 output clears independent CPU gate C" +
                    std::to_string(token_count));
    test.expect(clears_cpu_reference_gate(cpu_state_metrics),
                "sequential B8 state clears independent CPU gate C" +
                    std::to_string(token_count));
    const std::size_t used_output =
        token_count * q3x::runtime::kGdnVElements;
    const bool oracle_tail_preserved =
        std::all_of(oracle_output.data() + used_output,
                    oracle_output.data() + oracle_output.size(),
                    [](const std::uint16_t value) { return value == 0x7fc1U; });
    const bool candidate_tail_preserved =
        std::all_of(candidate_output.data() + used_output,
                    candidate_output.data() + candidate_output.size(),
                    [](const std::uint16_t value) { return value == 0x7fc1U; });
    test.expect(oracle_tail_preserved && candidate_tail_preserved,
                "B8 kernels preserve output tail C" +
                    std::to_string(token_count));
  }

  std::copy_n(initial_state.data(), initial_state.size(),
              production_state.data());
  std::fill_n(production_output.data(), production_output.size(),
              static_cast<std::uint16_t>(0x7fc1U));
  const bool production_launched = launch_production_chain(
      conv_qkv.data(), kCorrectnessTokens, a.data(), b.data(), A_log.data(),
      dt_bias.data(), production_state.data(), production_state.data(),
      production_output.data(), stream);
  test.expect(production_launched, "production M16 reference launch C16");
  ready = test.cuda_ok(cudaStreamSynchronize(stream),
                       "production M16 reference synchronize C16");
  if (production_launched && ready) {
    const ErrorMetrics wy_output_metrics = calculate_error_metrics(
        candidate_output.data(), production_output.data(),
        kCorrectnessTokens * q3x::runtime::kGdnVElements);
    const ErrorMetrics wy_state_metrics = calculate_error_metrics(
        candidate_state.data(), production_state.data(),
        q3x::runtime::kGdnStateElements);
    const ErrorMetrics sequential_output_metrics = calculate_error_metrics(
        oracle_output.data(), production_output.data(),
        kCorrectnessTokens * q3x::runtime::kGdnVElements);
    const ErrorMetrics sequential_state_metrics = calculate_error_metrics(
        oracle_state.data(), production_state.data(),
        q3x::runtime::kGdnStateElements);
    print_metrics("production-per-token-bf16", "wy-output",
                  kCorrectnessTokens, wy_output_metrics);
    print_metrics("production-per-token-bf16", "wy-state",
                  kCorrectnessTokens, wy_state_metrics);
    print_metrics("production-per-token-bf16", "sequential-output",
                  kCorrectnessTokens, sequential_output_metrics);
    print_metrics("production-per-token-bf16", "sequential-state",
                  kCorrectnessTokens, sequential_state_metrics);
    test.expect(wy_output_metrics.non_finite == 0U &&
                    wy_output_metrics.nrmse <= 3.0e-2 &&
                    wy_output_metrics.cosine >= 0.999,
                "B8 WY output remains close to production C16");
    test.expect(wy_state_metrics.non_finite == 0U &&
                    wy_state_metrics.nrmse <= 3.0e-2 &&
                    wy_state_metrics.cosine >= 0.999,
                "B8 WY state remains close to production C16");
    test.expect(sequential_output_metrics.non_finite == 0U &&
                    sequential_output_metrics.nrmse <= 3.0e-2 &&
                    sequential_output_metrics.cosine >= 0.999,
                "sequential B8 output remains close to production C16");
    test.expect(sequential_state_metrics.non_finite == 0U &&
                    sequential_state_metrics.nrmse <= 3.0e-2 &&
                    sequential_state_metrics.cosine >= 0.999,
                "sequential B8 state remains close to production C16");
  }

  for (const bool use_wy : {false, true}) {
    const std::string variant = use_wy ? "WY" : "sequential";
    const auto launch_variant =
        [&](const std::uint16_t* const qkv,
            const std::size_t tokens,
            const std::uint16_t* const scalar_a,
            const std::uint16_t* const scalar_b,
            std::uint16_t* const state,
            std::uint16_t* const result) {
      if (use_wy) {
        return launch_b8_block<true>(
            qkv, tokens, scalar_a, scalar_b, A_log.data(), dt_bias.data(),
            state, state, kL2Epsilon, result, stream);
      }
      return launch_b8_block<false>(
          qkv, tokens, scalar_a, scalar_b, A_log.data(), dt_bias.data(),
          state, state, kL2Epsilon, result, stream);
    };
    for (const std::size_t token_count : {9U, 16U}) {
      std::copy_n(initial_state.data(), initial_state.size(),
                  candidate_state.data());
      std::copy_n(initial_state.data(), initial_state.size(),
                  split_state.data());
      std::fill_n(candidate_output.data(), candidate_output.size(),
                  static_cast<std::uint16_t>(0x7fc1U));
      std::fill_n(split_output.data(), split_output.size(),
                  static_cast<std::uint16_t>(0x7fc1U));
      ready = test.cuda_ok(
          static_cast<cudaError_t>(launch_variant(
              conv_qkv.data(), token_count, a.data(), b.data(),
              candidate_state.data(), candidate_output.data())),
          "B8 " + variant + " whole chunk-split oracle C" +
              std::to_string(token_count));
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_variant(
                               conv_qkv.data(), 8U, a.data(), b.data(),
                               split_state.data(), split_output.data())),
                           "B8 " + variant + " split first C8");
      const std::size_t tail_tokens = token_count - 8U;
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_variant(
                               conv_qkv.data() +
                                   8U * q3x::runtime::kGdnQkvChannels,
                               tail_tokens,
                               a.data() +
                                   8U * q3x::runtime::kGdnValueHeadCount,
                               b.data() +
                                   8U * q3x::runtime::kGdnValueHeadCount,
                               split_state.data(),
                               split_output.data() +
                                   8U * q3x::runtime::kGdnVElements)),
                           "B8 " + variant + " split tail C" +
                               std::to_string(tail_tokens));
      ready = ready && test.cuda_ok(
                           cudaStreamSynchronize(stream),
                           "B8 " + variant + " chunk-split synchronize C" +
                               std::to_string(token_count));
      if (!ready) {
        return;
      }
      test.expect(std::equal(candidate_output.data(),
                             candidate_output.data() +
                                 token_count * q3x::runtime::kGdnVElements,
                             split_output.data()),
                  "B8 " + variant +
                      " whole output equals explicit C8+tail C" +
                      std::to_string(token_count));
      test.expect(std::equal(candidate_state.data(),
                             candidate_state.data() +
                                 q3x::runtime::kGdnStateElements,
                             split_state.data()),
                  "B8 " + variant +
                      " whole state equals explicit C8+tail C" +
                      std::to_string(token_count));
    }
  }

  test.expect(std::equal(conv_qkv.data(), conv_qkv.data() + conv_qkv.size(),
                         frozen_conv_qkv.begin()) &&
                  std::equal(a.data(), a.data() + a.size(), frozen_a.begin()) &&
                  std::equal(b.data(), b.data() + b.size(), frozen_b.begin()) &&
                  std::equal(A_log.data(), A_log.data() + A_log.size(),
                             frozen_A_log.begin()) &&
                  std::equal(dt_bias.data(), dt_bias.data() + dt_bias.size(),
                             frozen_dt_bias.begin()),
              "B8 kernels preserve all immutable inputs");

  test.expect(static_cast<cudaError_t>(launch_b8_block<true>(
                  nullptr, 0U, nullptr, nullptr, nullptr, nullptr, nullptr,
                  nullptr, kL2Epsilon, nullptr, stream)) ==
                  cudaErrorInvalidValue,
              "B8 WY rejects zero/null launch");
  test.expect(static_cast<cudaError_t>(launch_b8_block<true>(
                  conv_qkv.data(), kMaximumTokens + 1U, a.data(), b.data(),
                  A_log.data(), dt_bias.data(), candidate_state.data(),
                  candidate_state.data(), kL2Epsilon,
                  candidate_output.data(), stream)) == cudaErrorInvalidValue,
              "B8 WY rejects oversized launch");
  test.expect(static_cast<cudaError_t>(launch_b8_block<true>(
                  conv_qkv.data(), 8U, a.data(), b.data(), A_log.data(),
                  dt_bias.data(), candidate_state.data(),
                  candidate_state.data(), kL2Epsilon, conv_qkv.data(),
                  stream)) == cudaErrorInvalidValue,
              "B8 WY rejects output/conv alias");
  test.expect(static_cast<cudaError_t>(launch_b8_block<false>(
                  nullptr, 0U, nullptr, nullptr, nullptr, nullptr, nullptr,
                  nullptr, kL2Epsilon, nullptr, stream)) ==
                  cudaErrorInvalidValue,
              "selected B8 sequential kernel rejects zero/null launch");
  test.expect(static_cast<cudaError_t>(launch_b8_block<false>(
                  conv_qkv.data(), kMaximumTokens + 1U, a.data(), b.data(),
                  A_log.data(), dt_bias.data(), candidate_state.data(),
                  candidate_state.data(), kL2Epsilon,
                  candidate_output.data(), stream)) == cudaErrorInvalidValue,
              "selected B8 sequential kernel rejects oversized launch");
  test.expect(static_cast<cudaError_t>(launch_b8_block<false>(
                  conv_qkv.data(), 8U, a.data(), b.data(), A_log.data(),
                  dt_bias.data(), candidate_state.data(),
                  candidate_state.data(), kL2Epsilon, conv_qkv.data(),
                  stream)) == cudaErrorInvalidValue,
              "selected B8 sequential kernel rejects output/conv alias");
}

[[nodiscard]] double median(std::vector<float> values) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if (values.size() % 2U != 0U) {
    return values[middle];
  }
  return (static_cast<double>(values[middle - 1U]) + values[middle]) * 0.5;
}

template <typename Launch>
[[nodiscard]] std::vector<float> measure_operation_samples(
    TestContext& test,
    cudaStream_t stream,
    const std::size_t warmup_operations,
    const std::size_t measured_operations,
    const std::string& label,
    Launch&& launch) {
  bool ready = true;
  for (std::size_t operation = 0U;
       operation < warmup_operations && ready; ++operation) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch(operation)),
                         label + " warmup launch");
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " warmup synchronize");
  std::vector<cudaEvent_t> starts(measured_operations, nullptr);
  std::vector<cudaEvent_t> stops(measured_operations, nullptr);
  for (std::size_t operation = 0U;
       operation < measured_operations && ready; ++operation) {
    ready = test.cuda_ok(cudaEventCreate(&starts[operation]),
                         label + " create start event");
    ready = ready && test.cuda_ok(cudaEventCreate(&stops[operation]),
                                  label + " create stop event");
  }
  for (std::size_t operation = 0U;
       operation < measured_operations && ready; ++operation) {
    ready = test.cuda_ok(cudaEventRecord(starts[operation], stream),
                         label + " record start");
    ready = ready &&
            test.cuda_ok(static_cast<cudaError_t>(
                             launch(warmup_operations + operation)),
                         label + " measured launch");
    ready = ready && test.cuda_ok(cudaEventRecord(stops[operation], stream),
                                  label + " record stop");
  }
  if (ready && measured_operations != 0U) {
    ready = test.cuda_ok(cudaEventSynchronize(stops.back()),
                         label + " synchronize last stop");
  }
  std::vector<float> samples;
  samples.reserve(measured_operations);
  for (std::size_t operation = 0U; operation < measured_operations;
       ++operation) {
    if (ready) {
      float milliseconds = 0.0F;
      ready = test.cuda_ok(cudaEventElapsedTime(
                               &milliseconds, starts[operation],
                               stops[operation]),
                           label + " elapsed sample");
      if (ready) {
        samples.push_back(milliseconds);
      }
    }
    if (starts[operation] != nullptr) {
      (void)test.cuda_ok(cudaEventDestroy(starts[operation]),
                         label + " destroy start event");
    }
    if (stops[operation] != nullptr) {
      (void)test.cuda_ok(cudaEventDestroy(stops[operation]),
                         label + " destroy stop event");
    }
  }
  return samples;
}

void run_optional_performance(TestContext& test, cudaStream_t stream) {
  const auto environment_enabled = [](const char* const value) {
    return value != nullptr && value[0] != '\0' &&
           !(value[0] == '0' && value[1] == '\0');
  };
  const bool primary_enabled = environment_enabled(
      std::getenv("Q3X_RUN_GDN_B8_BLOCK_PERF"));
  const bool legacy_alias_enabled = environment_enabled(
      std::getenv("Q3X_RUN_GDN_B8_WY_PERF"));
  const bool enabled = primary_enabled || legacy_alias_enabled;
  if (!enabled) {
    std::cout << "SKIP: GDN B8 block-transition performance screen; set "
                 "Q3X_RUN_GDN_B8_BLOCK_PERF=1 to enable "
                 "(Q3X_RUN_GDN_B8_WY_PERF remains a legacy alias)\n";
    return;
  }
  std::cout << "GDN_B8_BLOCK_PERF_ENVIRONMENT: primary_enabled="
            << (primary_enabled ? "true" : "false")
            << " legacy_wy_alias_enabled="
            << (legacy_alias_enabled ? "true" : "false") << '\n';

  constexpr std::array<std::size_t, 2U> kTokenCounts{256U, 512U};
  constexpr std::size_t kBankCount = 24U;
  constexpr std::size_t kWarmupOperations = 24U;
  constexpr std::size_t kMeasuredOperations = 72U;
  constexpr std::size_t kRounds = 3U;
  constexpr float kRequiredSequentialSpeedup = 1.25F;
  constexpr std::size_t kStatePoolElements =
      kBankCount * q3x::runtime::kGdnStateElements;
  const std::size_t state_pool_bytes =
      kStatePoolElements * sizeof(std::uint16_t);
  const std::size_t maximum_output_elements =
      kMaximumTokens * q3x::runtime::kGdnVElements;

  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> immutable_states;
  ManagedBuffer<std::uint16_t> baseline_states;
  ManagedBuffer<std::uint16_t> candidate_states;
  ManagedBuffer<std::uint16_t> baseline_output;
  ManagedBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(
      conv_qkv.allocate(kMaximumTokens * q3x::runtime::kGdnQkvChannels),
      "B8 block perf allocate conv QKV");
  ready = ready && test.cuda_ok(
                       a.allocate(kMaximumTokens *
                                  q3x::runtime::kGdnValueHeadCount),
                       "B8 block perf allocate a");
  ready = ready && test.cuda_ok(
                       b.allocate(kMaximumTokens *
                                  q3x::runtime::kGdnValueHeadCount),
                       "B8 block perf allocate b");
  ready = ready && test.cuda_ok(
                       A_log.allocate(q3x::runtime::kGdnValueHeadCount),
                       "B8 block perf allocate A_log");
  ready = ready && test.cuda_ok(
                       dt_bias.allocate(q3x::runtime::kGdnValueHeadCount),
                       "B8 block perf allocate dt_bias");
  ready = ready && test.cuda_ok(
                       immutable_states.allocate(kStatePoolElements),
                       "B8 block perf allocate immutable states");
  ready = ready && test.cuda_ok(
                       baseline_states.allocate(kStatePoolElements),
                       "B8 block perf allocate baseline states");
  ready = ready && test.cuda_ok(
                       candidate_states.allocate(kStatePoolElements),
                       "B8 block perf allocate candidate states");
  ready = ready && test.cuda_ok(
                       baseline_output.allocate(maximum_output_elements),
                       "B8 block perf allocate baseline output");
  ready = ready && test.cuda_ok(
                       candidate_output.allocate(maximum_output_elements),
                       "B8 block perf allocate candidate output");
  if (!ready) {
    return;
  }
  fill_inputs(conv_qkv, a, b, A_log, dt_bias, kMaximumTokens);
  for (std::size_t bank = 0U; bank < kBankCount; ++bank) {
    fill_state(immutable_states.data() +
                   bank * q3x::runtime::kGdnStateElements,
               q3x::runtime::kGdnStateElements, bank);
  }

  int device = 0;
  cudaDeviceProp properties{};
  ready = test.cuda_ok(cudaGetDevice(&device), "B8 block perf get device");
  ready = ready && test.cuda_ok(cudaGetDeviceProperties(&properties, device),
                                "B8 block perf get properties");
  if (!ready) {
    return;
  }
  test.expect(state_pool_bytes >
                  static_cast<std::size_t>(properties.l2CacheSize),
              "B8 block 24-bank state working set exceeds L2");

  const auto reset_variant =
      [&](ManagedBuffer<std::uint16_t>& states,
          ManagedBuffer<std::uint16_t>& output,
          const std::size_t output_elements,
          const std::string& label) -> bool {
    bool reset_ready = test.cuda_ok(
        cudaMemcpyAsync(states.data(), immutable_states.data(),
                        state_pool_bytes, cudaMemcpyDeviceToDevice, stream),
        label + " reset 24 state banks");
    reset_ready = reset_ready && test.cuda_ok(
                                     cudaMemsetAsync(
                                         output.data(), 0xff,
                                         output_elements *
                                             sizeof(std::uint16_t),
                                         stream),
                                     label + " poison output");
    reset_ready = reset_ready &&
                  test.cuda_ok(cudaStreamSynchronize(stream),
                               label + " reset synchronize");
    return reset_ready;
  };

  for (const std::size_t token_count : kTokenCounts) {
    const std::size_t output_elements =
        token_count * q3x::runtime::kGdnVElements;
    std::vector<float> all_baseline_samples;
    std::vector<float> all_sequential_samples;
    std::vector<float> all_candidate_samples;
    bool all_wy_rounds_faster = true;
    bool all_sequential_rounds_faster = true;
    bool all_wy_rounds_slower_than_sequential = true;
    for (std::size_t round = 0U; round < kRounds; ++round) {
      enum class Variant : unsigned int {
        kProduction,
        kSequential,
        kWy,
      };
      const auto measure = [&](const Variant variant,
                               const std::string& label) {
        const bool production = variant == Variant::kProduction;
        ManagedBuffer<std::uint16_t>& states =
            production ? baseline_states : candidate_states;
        ManagedBuffer<std::uint16_t>& output =
            production ? baseline_output : candidate_output;
        if (!reset_variant(states, output, output_elements, label)) {
          return std::vector<float>{};
        }
        return measure_operation_samples(
            test, stream, kWarmupOperations, kMeasuredOperations, label,
            [&](const std::size_t operation) {
              const std::size_t bank = operation % kBankCount;
              std::uint16_t* const state =
                  states.data() +
                  bank * q3x::runtime::kGdnStateElements;
              if (variant == Variant::kWy) {
                return launch_b8_block<true>(
                    conv_qkv.data(), token_count, a.data(), b.data(),
                    A_log.data(), dt_bias.data(), state, state, kL2Epsilon,
                    output.data(), stream);
              }
              if (variant == Variant::kSequential) {
                return launch_b8_block<false>(
                    conv_qkv.data(), token_count, a.data(), b.data(),
                    A_log.data(), dt_bias.data(), state, state, kL2Epsilon,
                    output.data(), stream);
              }
              return launch_production_chain(
                         conv_qkv.data(), token_count, a.data(), b.data(),
                         A_log.data(), dt_bias.data(), state, state,
                         output.data(), stream)
                         ? static_cast<int>(cudaSuccess)
                         : static_cast<int>(cudaErrorInvalidValue);
            });
      };

      const std::string prefix =
          "B8 block C" + std::to_string(token_count) +
          " round=" + std::to_string(round + 1U);
      const std::vector<float> baseline_first =
          measure(Variant::kProduction, prefix + " B1");
      const std::vector<float> sequential_first =
          measure(Variant::kSequential, prefix + " S1");
      const std::vector<float> candidate_first =
          measure(Variant::kWy, prefix + " W1");
      const std::vector<float> candidate_second =
          measure(Variant::kWy, prefix + " W2");
      const std::vector<float> sequential_second =
          measure(Variant::kSequential, prefix + " S2");
      const std::vector<float> baseline_second =
          measure(Variant::kProduction, prefix + " B2");
      const double baseline_p50 =
          median([&]() {
            std::vector<float> values = baseline_first;
            values.insert(values.end(), baseline_second.begin(),
                          baseline_second.end());
            return values;
          }());
      const double sequential_p50 =
          median([&]() {
            std::vector<float> values = sequential_first;
            values.insert(values.end(), sequential_second.begin(),
                          sequential_second.end());
            return values;
          }());
      const double candidate_p50 =
          median([&]() {
            std::vector<float> values = candidate_first;
            values.insert(values.end(), candidate_second.begin(),
                          candidate_second.end());
            return values;
          }());
      const bool round_faster = std::isfinite(baseline_p50) &&
                                std::isfinite(candidate_p50) &&
                                candidate_p50 < baseline_p50;
      const bool sequential_round_faster =
          std::isfinite(baseline_p50) && std::isfinite(sequential_p50) &&
          sequential_p50 < baseline_p50;
      const bool wy_round_slower_than_sequential =
          std::isfinite(sequential_p50) && std::isfinite(candidate_p50) &&
          sequential_p50 < candidate_p50;
      all_wy_rounds_faster = all_wy_rounds_faster && round_faster;
      all_sequential_rounds_faster =
          all_sequential_rounds_faster && sequential_round_faster;
      all_wy_rounds_slower_than_sequential =
          all_wy_rounds_slower_than_sequential &&
          wy_round_slower_than_sequential;
      all_baseline_samples.insert(all_baseline_samples.end(),
                                  baseline_first.begin(),
                                  baseline_first.end());
      all_baseline_samples.insert(all_baseline_samples.end(),
                                  baseline_second.begin(),
                                  baseline_second.end());
      all_sequential_samples.insert(all_sequential_samples.end(),
                                    sequential_first.begin(),
                                    sequential_first.end());
      all_sequential_samples.insert(all_sequential_samples.end(),
                                    sequential_second.begin(),
                                    sequential_second.end());
      all_candidate_samples.insert(all_candidate_samples.end(),
                                   candidate_first.begin(),
                                   candidate_first.end());
      all_candidate_samples.insert(all_candidate_samples.end(),
                                   candidate_second.begin(),
                                   candidate_second.end());
      std::cout << "PERF_GDN_B8_BLOCK_ROUND: token_count=" << token_count
                << " round=" << round + 1U
                << " baseline_p50_ms=" << baseline_p50
                << " sequential_p50_ms=" << sequential_p50
                << " candidate_p50_ms=" << candidate_p50
                << " baseline_to_sequential="
                << baseline_p50 / sequential_p50
                << " wy_relative_speed=" << sequential_p50 / candidate_p50
                << " baseline_to_wy=" << baseline_p50 / candidate_p50
                << " sequential_no_reversal="
                << (sequential_round_faster ? "true" : "false")
                << " wy_no_reversal="
                << (round_faster ? "true" : "false")
                << " wy_rejected_round="
                << (wy_round_slower_than_sequential ? "true" : "false")
                << '\n';
    }
    const double baseline_p50 = median(all_baseline_samples);
    const double sequential_p50 = median(all_sequential_samples);
    const double candidate_p50 = median(all_candidate_samples);
    const double speedup = baseline_p50 / candidate_p50;
    const double sequential_speedup = baseline_p50 / sequential_p50;
    const bool wy_baseline_gate =
        all_wy_rounds_faster && std::isfinite(speedup) &&
        speedup >= kRequiredSequentialSpeedup;
    const bool sequential_gate =
        all_sequential_rounds_faster && std::isfinite(sequential_speedup) &&
        sequential_speedup >= kRequiredSequentialSpeedup;
    const double wy_relative_speed = sequential_p50 / candidate_p50;
    const bool wy_rejected_for_current_dataflow =
        all_wy_rounds_slower_than_sequential &&
        std::isfinite(wy_relative_speed) && wy_relative_speed < 1.0;
    std::cout << "PERF_GDN_B8_BLOCK_AGGREGATE: token_count=" << token_count
              << " baseline_p50_ms=" << baseline_p50
              << " sequential_p50_ms=" << sequential_p50
              << " candidate_p50_ms=" << candidate_p50
              << " baseline_to_sequential="
              << baseline_p50 / sequential_p50
              << " wy_relative_speed=" << wy_relative_speed
              << " baseline_to_wy=" << speedup
              << " required_sequential_speedup="
              << kRequiredSequentialSpeedup
              << " state_banks=" << kBankCount
              << " state_working_set_bytes=" << state_pool_bytes
              << " l2_cache_bytes=" << properties.l2CacheSize
              << " samples_per_variant=" << all_baseline_samples.size()
              << " mirrored_order=B-S-W-W-S-B"
              << " sequential_all_rounds_faster="
              << (all_sequential_rounds_faster ? "true" : "false")
              << " wy_all_rounds_faster_than_baseline="
              << (all_wy_rounds_faster ? "true" : "false")
              << " wy_all_rounds_slower_than_sequential="
              << (all_wy_rounds_slower_than_sequential ? "true" : "false")
              << " sequential_gate="
              << (sequential_gate ? "PASS" : "FAIL")
              << " wy_baseline_gate="
              << (wy_baseline_gate ? "PASS" : "FAIL")
              << " wy_rejection_gate="
              << (wy_rejected_for_current_dataflow ? "PASS" : "FAIL")
              << '\n';
    test.expect(sequential_gate,
                "sequential GDN B8 clears 1.25x C" +
                    std::to_string(token_count) + " performance gate");
    const std::size_t expected_samples =
        2U * kRounds * kMeasuredOperations;
    test.expect(all_baseline_samples.size() == expected_samples &&
                    all_sequential_samples.size() == expected_samples &&
                    all_candidate_samples.size() == expected_samples,
                "GDN B8 block screen captures every C" +
                    std::to_string(token_count) + " timing sample");
    test.expect(wy_rejected_for_current_dataflow,
                "WY control is slower than sequential B8 in every C" +
                    std::to_string(token_count) + " mirrored round");
    std::cout << "GDN_B8_SCREEN_DECISION: token_count=" << token_count
              << " selected=sequential-fp32-b8"
              << " sequential_p50_ms=" << sequential_p50
              << " wy_p50_ms=" << candidate_p50
              << " wy_relative_speed=" << wy_relative_speed
              << " wy_rejected_for_current_dataflow="
              << (wy_rejected_for_current_dataflow ? "true" : "false")
              << '\n';
  }
}

}  // namespace

int main() {
  constexpr int kSkipReturnCode = 77;
  TestContext test;
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: GDN B8 block-transition SM87 screen "
                 "(no CUDA device)\n";
    (void)cudaGetLastError();
    return kSkipReturnCode;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                    "B8 block read device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: GDN B8 block-transition screen requires SM87; got sm_"
              << properties.major << properties.minor << '\n';
    return kSkipReturnCode;
  }
  std::cout << "GDN_B8_BLOCK_DEVICE: name=" << properties.name
            << " compute_capability=" << properties.major << '.'
            << properties.minor << " sm_count="
            << properties.multiProcessorCount << " l2_bytes="
            << properties.l2CacheSize << '\n';
  std::cout
      << "GDN_B8_BLOCK_FROZEN_NUMERICS_GATES: relative_denominator_floor=1e-2"
      << " output_max_abs=1e-4 output_max_rel=1e-2"
      << " output_p99_rel=1e-2 output_nrmse=1e-4"
      << " output_min_cosine=0.999999"
      << " state_max_abs=0.001953125 state_p99_abs=0.00048828125"
      << " state_p99_rel=0.025 state_nrmse=1e-4"
      << " state_min_cosine=0.999999"
      << " state_max_rel=reported_not_gated"
      << " cpu_max_abs=0.00390625 cpu_p99_abs=0.0009765625"
      << " cpu_p99_rel=0.025 cpu_nrmse=0.01"
      << " cpu_min_cosine=0.9999 cpu_max_rel=reported_not_gated\n";

  KernelResources oracle_resources{};
  KernelResources candidate_resources{};
  bool ready = test.cuda_ok(query_resources<false>(oracle_resources),
                            "B8 sequential query resources");
  ready = ready && test.cuda_ok(query_resources<true>(candidate_resources),
                                "B8 WY query resources");
  if (ready) {
    std::cout << "GDN_B8_SEQUENTIAL_RESOURCES: registers="
              << oracle_resources.registers_per_thread
              << " static_shared_bytes="
              << oracle_resources.static_shared_bytes
              << " local_bytes=" << oracle_resources.local_bytes
              << " threads=" << kThreads
              << " active_blocks_per_sm="
              << oracle_resources.active_blocks_per_sm << '\n';
    std::cout << "GDN_B8_WY_RESOURCES: registers="
              << candidate_resources.registers_per_thread
              << " static_shared_bytes="
              << candidate_resources.static_shared_bytes
              << " local_bytes=" << candidate_resources.local_bytes
              << " threads=" << kThreads
              << " active_blocks_per_sm="
              << candidate_resources.active_blocks_per_sm << '\n';
    test.expect(oracle_resources.registers_per_thread <= 112,
                "selected B8 sequential kernel uses at most 112 registers");
    test.expect(oracle_resources.static_shared_bytes == 8'256U,
                "selected B8 sequential kernel keeps the frozen shared-memory footprint");
    test.expect(oracle_resources.local_bytes == 0U,
                "selected B8 sequential kernel has no local-memory spill allocation");
    test.expect(oracle_resources.maximum_threads_per_block >=
                    static_cast<int>(kThreads) &&
                    oracle_resources.active_blocks_per_sm >= 2,
                "selected B8 sequential kernel retains two 256-thread CTAs per SM");
    test.expect(candidate_resources.local_bytes == 0U,
                "B8 WY has no local-memory spill allocation");
    test.expect(candidate_resources.static_shared_bytes <= 96U * 1024U,
                "B8 WY static shared memory is at most 96 KiB");
    test.expect(candidate_resources.maximum_threads_per_block >=
                    static_cast<int>(kThreads) &&
                    candidate_resources.active_blocks_per_sm >= 1,
                "B8 WY retains at least one 256-thread CTA per SM");
  }

  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "B8 block create stream")) {
    return 1;
  }
  run_correctness(test, stream);
  run_optional_performance(test, stream);
  (void)test.cuda_ok(cudaStreamDestroy(stream), "B8 block destroy stream");
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " GDN B8 block-transition assertion(s) failed\n";
    return 1;
  }
  std::cout << "GDN B8 block-transition SM87 screen passed\n";
  return 0;
}
