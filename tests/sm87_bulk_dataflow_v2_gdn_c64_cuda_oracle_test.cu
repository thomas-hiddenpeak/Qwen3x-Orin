#include "q3x/kernels/sm87_bulk_dataflow_v2_gdn_c64.h"

#include <cuda_runtime.h>

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace kernels = q3x::kernels;

namespace {

constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kFullWarpMask = 0xffff'ffffU;
constexpr unsigned int kReferenceRecurrenceThreads = 128U;
constexpr unsigned int kCompareThreads = 256U;
constexpr unsigned int kCompareBlocks = 1'024U;
constexpr std::size_t kConvChannels =
    kernels::kSm87TargetAotGdnTotalConvChannels;
constexpr std::size_t kStateElements =
    kernels::kSm87TargetAotGdnTotalStateBytes /
    kernels::kSm87TargetAotGdnBf16Bytes;
constexpr std::size_t kStateElementsPerHead =
    kernels::kSm87TargetAotGdnStateValuesPerHead;
constexpr std::size_t kOutputElements =
    kernels::kSm87BulkV2GdnOutputBytes /
    kernels::kSm87TargetAotGdnBf16Bytes;
constexpr std::size_t kConvHistoryElements =
    kernels::kSm87BulkV2GdnConvHistoryBytes /
    kernels::kSm87TargetAotGdnBf16Bytes;
constexpr std::size_t kTraceElements =
    kernels::kSm87BulkV2GdnStateTraceBytes /
    kernels::kSm87TargetAotGdnBf16Bytes;

static_assert(kReferenceRecurrenceThreads ==
              kernels::kSm87TargetAotGdnStateValueDimension);
static_assert(kStateElements == 786'432U);
static_assert(kTraceElements == 50'331'648U);

[[nodiscard]] __device__ __forceinline__ float reference_decode_bf16(
    const std::uint16_t value) noexcept {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t
reference_encode_bf16_rne(const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ float reference_stable_softplus(
    const float value) noexcept {
  return value > 20.0F ? value : log1pf(expf(value));
}

[[nodiscard]] __device__ __forceinline__ float reference_stable_sigmoid(
    const float value) noexcept {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + expf(-value));
  }
  const float exponential = expf(value);
  return exponential / (1.0F + exponential);
}

// This oracle deliberately does not consume any candidate workspace. One
// thread owns one complete Q/K/V convolution channel and advances the raw
// width-four history in token order. The arithmetic and BF16 publication
// point are the frozen target-AOT contract, while the ownership is independent
// from the candidate's role-by-four-row producer.
__global__ void reference_conv_silu_c64_kernel(
    const std::uint16_t* const raw_qkvz,
    const std::uint16_t* const conv_weight,
    const std::uint16_t* const initial_conv_history,
    std::uint16_t* const convolved_qkv,
    std::uint16_t* const final_conv_history) {
  const unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= kConvChannels) {
    return;
  }

  std::uint16_t history[3U] = {
      initial_conv_history[static_cast<std::size_t>(channel) * 3U],
      initial_conv_history[static_cast<std::size_t>(channel) * 3U + 1U],
      initial_conv_history[static_cast<std::size_t>(channel) * 3U + 2U]};
  float weight[4U];
#pragma unroll
  for (unsigned int tap = 0U; tap < 4U; ++tap) {
    weight[tap] = reference_decode_bf16(
        conv_weight[static_cast<std::size_t>(channel) * 4U + tap]);
  }

#pragma unroll 1
  for (unsigned int token = 0U;
       token < kernels::kSm87BulkV2GdnC64Tokens; ++token) {
    const std::uint16_t current =
        raw_qkvz[static_cast<std::size_t>(token) *
                     kernels::kSm87TargetAotGdnRawQkvZChannels +
                 channel];
    float convolution = 0.0F;
    convolution = fmaf(reference_decode_bf16(history[0U]), weight[0U],
                       convolution);
    convolution = fmaf(reference_decode_bf16(history[1U]), weight[1U],
                       convolution);
    convolution = fmaf(reference_decode_bf16(history[2U]), weight[2U],
                       convolution);
    convolution = fmaf(reference_decode_bf16(current), weight[3U],
                       convolution);
    convolved_qkv[static_cast<std::size_t>(token) * kConvChannels + channel] =
        reference_encode_bf16_rne(
            convolution / (1.0F + expf(-convolution)));
    history[0U] = history[1U];
    history[1U] = history[2U];
    history[2U] = current;
  }

  const std::size_t history_base = static_cast<std::size_t>(channel) * 3U;
  final_conv_history[history_base] = history[0U];
  final_conv_history[history_base + 1U] = history[1U];
  final_conv_history[history_base + 2U] = history[2U];
}

// One independent warp owns one token/QK-group pair. Lane ownership, the
// (0+64)/(32+96) pair tree, shuffle order, rsqrtf sites, and FP32 publication
// match the frozen finite-precision contract exactly.
__global__ void reference_normalize_qk_c64_kernel(
    const std::uint16_t* const convolved_qkv,
    const std::uint32_t l2_epsilon_bits,
    float* const normalized_q,
    float* const normalized_k) {
  const unsigned int lane = threadIdx.x;
  const unsigned int token = blockIdx.x;
  const unsigned int qk_group = blockIdx.y;
  float q_values[4U];
  float k_values[4U];
#pragma unroll
  for (unsigned int item = 0U; item < 4U; ++item) {
    const unsigned int dimension = lane + item * kWarpSize;
    const unsigned int q_channel =
        qk_group * kernels::kSm87TargetAotGdnStateKeyDimension + dimension;
    const unsigned int k_channel =
        kernels::kSm87TargetAotGdnRawKOffset + q_channel;
    const std::size_t token_base =
        static_cast<std::size_t>(token) * kConvChannels;
    q_values[item] =
        reference_decode_bf16(convolved_qkv[token_base + q_channel]);
    k_values[item] =
        reference_decode_bf16(convolved_qkv[token_base + k_channel]);
  }

  const float q_first_square = q_values[0U] * q_values[0U];
  const float q_second_square = q_values[1U] * q_values[1U];
  const float q_third_square = q_values[2U] * q_values[2U];
  const float q_fourth_square = q_values[3U] * q_values[3U];
  const float q_first_pair = q_first_square + q_third_square;
  const float q_second_pair = q_second_square + q_fourth_square;
  float q_sum = q_first_pair + q_second_pair;
  const float k_first_square = k_values[0U] * k_values[0U];
  const float k_second_square = k_values[1U] * k_values[1U];
  const float k_third_square = k_values[2U] * k_values[2U];
  const float k_fourth_square = k_values[3U] * k_values[3U];
  const float k_first_pair = k_first_square + k_third_square;
  const float k_second_pair = k_second_square + k_fourth_square;
  float k_sum = k_first_pair + k_second_pair;
#pragma unroll
  for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
    q_sum += __shfl_down_sync(kFullWarpMask, q_sum, stride);
    k_sum += __shfl_down_sync(kFullWarpMask, k_sum, stride);
  }

  float q_scale = 0.0F;
  float k_scale = 0.0F;
  if (lane == 0U) {
    q_scale = rsqrtf(q_sum + __uint_as_float(l2_epsilon_bits));
    q_scale *= rsqrtf(static_cast<float>(
        kernels::kSm87TargetAotGdnStateKeyDimension));
    k_scale = rsqrtf(k_sum + __uint_as_float(l2_epsilon_bits));
  }
  q_scale = __shfl_sync(kFullWarpMask, q_scale, 0U);
  k_scale = __shfl_sync(kFullWarpMask, k_scale, 0U);
  const std::size_t base =
      (static_cast<std::size_t>(qk_group) *
           kernels::kSm87BulkV2GdnC64Tokens +
       token) *
      kernels::kSm87TargetAotGdnStateKeyDimension;
#pragma unroll
  for (unsigned int item = 0U; item < 4U; ++item) {
    const unsigned int dimension = lane + item * kWarpSize;
    normalized_q[base + dimension] = q_values[item] * q_scale;
    normalized_k[base + dimension] = k_values[item] * k_scale;
  }
}

__global__ void reference_gate_scalars_c64_kernel(
    const std::uint16_t* const interleaved_ab,
    const std::uint16_t* const a_log,
    const std::uint16_t* const dt_bias,
    float* const alpha,
    float* const beta) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned int kScalarCount =
      kernels::kSm87BulkV2GdnC64Tokens *
      kernels::kSm87TargetAotGdnValueHeads;
  if (linear >= kScalarCount) {
    return;
  }
  const unsigned int token =
      linear / kernels::kSm87TargetAotGdnValueHeads;
  const unsigned int value_head =
      linear % kernels::kSm87TargetAotGdnValueHeads;
  const std::size_t ab_row =
      static_cast<std::size_t>(token) * kernels::kSm87TargetAotGdnAbChannels;
  const float gate_input =
      reference_decode_bf16(interleaved_ab[ab_row + value_head]) +
      reference_decode_bf16(dt_bias[value_head]);
  const float g = -expf(reference_decode_bf16(a_log[value_head])) *
                  reference_stable_softplus(gate_input);
  const std::size_t destination =
      static_cast<std::size_t>(value_head) *
          kernels::kSm87BulkV2GdnC64Tokens +
      token;
  alpha[destination] = expf(g);
  beta[destination] = reference_stable_sigmoid(reference_decode_bf16(
      interleaved_ab[ab_row + kernels::kSm87TargetAotGdnValueHeads +
                     value_head]));
}

// One block owns one value head; one thread owns one complete value row. The
// reference intentionally keeps state in global BF16 rather than the
// candidate's packed register ownership. Volatile FP32 row storage preserves
// the candidate contract's explicit scale -> prediction -> update -> output
// boundaries without relying on compiler recomputation or reassociation.
__global__ void reference_recurrence_c64_kernel(
    const std::uint16_t* const convolved_qkv,
    const float* const normalized_q,
    const float* const normalized_k,
    const float* const alpha,
    const float* const beta,
    std::uint16_t* const recurrent_state,
    std::uint16_t* const raw_output,
    std::uint16_t* const state_trace) {
  __shared__ float q[kernels::kSm87TargetAotGdnStateKeyDimension];
  __shared__ float k[kernels::kSm87TargetAotGdnStateKeyDimension];
  __shared__ float scalars[2U];
  const unsigned int value_head = blockIdx.x;
  const unsigned int value_dimension = threadIdx.x;
  const unsigned int qk_group =
      value_head / kernels::kSm87TargetAotGdnValueHeadsPerQkGroup;
  const std::size_t state_row =
      static_cast<std::size_t>(value_head) * kStateElementsPerHead +
      static_cast<std::size_t>(value_dimension) *
          kernels::kSm87TargetAotGdnStateKeyDimension;
  volatile float updated_row[kernels::kSm87TargetAotGdnStateKeyDimension];

#pragma unroll 1
  for (unsigned int token = 0U;
       token < kernels::kSm87BulkV2GdnC64Tokens; ++token) {
    const std::size_t qk_base =
        (static_cast<std::size_t>(qk_group) *
             kernels::kSm87BulkV2GdnC64Tokens +
         token) *
        kernels::kSm87TargetAotGdnStateKeyDimension;
    q[threadIdx.x] = normalized_q[qk_base + threadIdx.x];
    k[threadIdx.x] = normalized_k[qk_base + threadIdx.x];
    if (threadIdx.x == 0U) {
      const std::size_t scalar =
          static_cast<std::size_t>(value_head) *
              kernels::kSm87BulkV2GdnC64Tokens +
          token;
      scalars[0U] = alpha[scalar];
      scalars[1U] = beta[scalar];
    }
    __syncthreads();

    float prediction = 0.0F;
#pragma unroll 1
    for (unsigned int key = 0U;
         key < kernels::kSm87TargetAotGdnStateKeyDimension; ++key) {
      const float scaled =
          scalars[0U] * reference_decode_bf16(recurrent_state[state_row + key]);
      updated_row[key] = scaled;
      prediction = fmaf(scaled, k[key], prediction);
    }
    const unsigned int v_channel =
        kernels::kSm87TargetAotGdnRawVOffset +
        value_head * kernels::kSm87TargetAotGdnStateValueDimension +
        value_dimension;
    const float delta =
        (reference_decode_bf16(
             convolved_qkv[static_cast<std::size_t>(token) * kConvChannels +
                           v_channel]) -
         prediction) *
        scalars[1U];

#pragma unroll 1
    for (unsigned int key = 0U;
         key < kernels::kSm87TargetAotGdnStateKeyDimension; ++key) {
      const float updated = fmaf(delta, k[key], updated_row[key]);
      updated_row[key] = updated;
      const std::uint16_t rounded = reference_encode_bf16_rne(updated);
      recurrent_state[state_row + key] = rounded;
      state_trace[static_cast<std::size_t>(token) * kStateElements +
                  state_row + key] = rounded;
    }

    float result = 0.0F;
#pragma unroll 1
    for (unsigned int key = 0U;
         key < kernels::kSm87TargetAotGdnStateKeyDimension; ++key) {
      result = fmaf(updated_row[key], q[key], result);
    }
    raw_output[(static_cast<std::size_t>(token) *
                    kernels::kSm87TargetAotGdnValueHeads +
                value_head) *
                   kernels::kSm87TargetAotGdnStateValueDimension +
               value_dimension] = reference_encode_bf16_rne(result);
    __syncthreads();
  }
}

__global__ void reference_norm_gate_c64_kernel(
    const std::uint16_t* const raw_output,
    const std::uint16_t* const raw_qkvz,
    const std::uint16_t* const norm_weight,
    const std::uint32_t norm_epsilon_bits,
    std::uint16_t* const output) {
  const unsigned int lane = threadIdx.x;
  const unsigned int row = blockIdx.x;
  const unsigned int token =
      row / kernels::kSm87TargetAotGdnValueHeads;
  const unsigned int value_head =
      row % kernels::kSm87TargetAotGdnValueHeads;
  const std::size_t raw_base =
      static_cast<std::size_t>(row) *
      kernels::kSm87TargetAotGdnStateValueDimension;
  float raw_values[4U];
  float squares[4U];
#pragma unroll
  for (unsigned int item = 0U; item < 4U; ++item) {
    const unsigned int dimension = lane + item * kWarpSize;
    raw_values[item] = reference_decode_bf16(raw_output[raw_base + dimension]);
    squares[item] = fmaf(raw_values[item], raw_values[item], 0.0F);
  }
  const float first_pair = squares[0U] + squares[2U];
  const float second_pair = squares[1U] + squares[3U];
  float warp_sum = first_pair + second_pair;
#pragma unroll
  for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
    warp_sum += __shfl_down_sync(kFullWarpMask, warp_sum, stride);
  }
  float inverse_rms = 0.0F;
  if (lane == 0U) {
    inverse_rms = rsqrtf(
        warp_sum /
                static_cast<float>(
                    kernels::kSm87TargetAotGdnStateValueDimension) +
            __uint_as_float(norm_epsilon_bits));
  }
  inverse_rms = __shfl_sync(kFullWarpMask, inverse_rms, 0U);
#pragma unroll
  for (unsigned int item = 0U; item < 4U; ++item) {
    const unsigned int dimension = lane + item * kWarpSize;
    float value = raw_values[item] * inverse_rms;
    value *= reference_decode_bf16(norm_weight[dimension]);
    const std::size_t z_index =
        static_cast<std::size_t>(token) *
            kernels::kSm87TargetAotGdnRawQkvZChannels +
        kernels::kSm87TargetAotGdnRawZOffset +
        value_head * kernels::kSm87TargetAotGdnStateValueDimension +
        dimension;
    const float z = reference_decode_bf16(raw_qkvz[z_index]);
    value *= z / (1.0F + expf(-z));
    const std::size_t destination =
        static_cast<std::size_t>(token) *
            kernels::kSm87TargetAotGdnOutputChannels +
        value_head * kernels::kSm87TargetAotGdnStateValueDimension +
        dimension;
    output[destination] = reference_encode_bf16_rne(value);
  }
}

struct MismatchRecord final {
  unsigned long long mismatch_count;
  unsigned long long first_index;
};

__global__ void compare_bf16_kernel(const std::uint16_t* const candidate,
                                    const std::uint16_t* const reference,
                                    const std::size_t count,
                                    MismatchRecord* const record) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (candidate[index] != reference[index]) {
      atomicAdd(&record->mismatch_count, 1ULL);
      atomicMin(&record->first_index,
                static_cast<unsigned long long>(index));
    }
  }
}

template <class T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t count) noexcept {
    count_ = count;
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  [[nodiscard]] cudaError_t upload(const std::vector<T>& source,
                                   cudaStream_t stream) noexcept {
    if (source.size() != count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(data_, source.data(), count_ * sizeof(T),
                           cudaMemcpyHostToDevice, stream);
  }

  [[nodiscard]] cudaError_t upload_one(const T& source,
                                       cudaStream_t stream) noexcept {
    if (count_ != 1U) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(data_, &source, sizeof(T), cudaMemcpyHostToDevice,
                           stream);
  }

  [[nodiscard]] cudaError_t download_one(T* const destination,
                                         cudaStream_t stream) const noexcept {
    if (count_ != 1U || destination == nullptr) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(destination, data_, sizeof(T),
                           cudaMemcpyDeviceToHost, stream);
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

[[nodiscard]] std::uint16_t host_encode_bf16_rne(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] std::uint64_t mix_index(std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xbf58'476d'1ce4'e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d0'49bb'1331'11ebULL;
  return value ^ (value >> 31U);
}

template <std::size_t N>
[[nodiscard]] std::uint16_t choose_pattern(
    const std::array<std::uint16_t, N>& patterns,
    const std::size_t index,
    const std::uint64_t salt) noexcept {
  return patterns[mix_index(static_cast<std::uint64_t>(index) + salt) % N];
}

struct HostInputs final {
  std::vector<std::uint16_t> raw_qkvz;
  std::vector<std::uint16_t> interleaved_ab;
  std::vector<std::uint16_t> conv_weight;
  std::vector<std::uint16_t> initial_conv_history;
  std::vector<std::uint16_t> a_log;
  std::vector<std::uint16_t> dt_bias;
  std::vector<std::uint16_t> norm_weight;
  std::vector<std::uint16_t> initial_recurrent_state;
};

[[nodiscard]] HostInputs make_adversarial_inputs() {
  const std::array<std::uint16_t, 18U> signals{{
      0x0000U, 0x8000U, 0x0001U, 0x8001U,
      host_encode_bf16_rne(0.03125F), host_encode_bf16_rne(-0.03125F),
      host_encode_bf16_rne(0.0625F), host_encode_bf16_rne(-0.0625F),
      host_encode_bf16_rne(0.125F), host_encode_bf16_rne(-0.125F),
      host_encode_bf16_rne(0.25F), host_encode_bf16_rne(-0.25F),
      host_encode_bf16_rne(0.5F), host_encode_bf16_rne(-0.5F),
      host_encode_bf16_rne(0.75F), host_encode_bf16_rne(-0.75F),
      host_encode_bf16_rne(1.0F), host_encode_bf16_rne(-1.0F)}};
  const std::array<std::uint16_t, 10U> weights{{
      0x0000U, 0x8000U, host_encode_bf16_rne(0.03125F),
      host_encode_bf16_rne(-0.03125F), host_encode_bf16_rne(0.0625F),
      host_encode_bf16_rne(-0.0625F), host_encode_bf16_rne(0.125F),
      host_encode_bf16_rne(-0.125F), host_encode_bf16_rne(0.25F),
      host_encode_bf16_rne(-0.25F)}};
  const std::array<std::uint16_t, 12U> state_values{{
      0x0000U, 0x8000U, 0x0001U, 0x8001U,
      host_encode_bf16_rne(0.0078125F),
      host_encode_bf16_rne(-0.0078125F),
      host_encode_bf16_rne(0.015625F),
      host_encode_bf16_rne(-0.015625F),
      host_encode_bf16_rne(0.03125F),
      host_encode_bf16_rne(-0.03125F),
      host_encode_bf16_rne(0.0625F),
      host_encode_bf16_rne(-0.0625F)}};
  const std::array<float, 12U> a_values{{
      -24.0F, -20.0F, -1.0F, -0.0F, 0.0F, 1.0F,
      8.0F, 19.0F, 20.0F, 20.125F, 21.0F, 24.0F}};
  const std::array<float, 10U> b_values{{
      -16.0F, -8.0F, -1.0F, -0.0F, 0.0F,
      1.0F, 8.0F, 16.0F, 0.125F, -0.125F}};
  const std::array<float, 6U> a_log_values{{
      -4.0F, -3.0F, -2.0F, -1.0F, -0.75F, -0.5F}};
  const std::array<float, 8U> dt_values{{
      -2.0F, -1.0F, -0.125F, -0.0F,
      0.0F, 0.125F, 1.0F, 2.0F}};
  const std::array<float, 8U> norm_values{{
      -1.5F, -1.0F, -0.5F, -0.125F,
      0.125F, 0.5F, 1.0F, 1.5F}};

  HostInputs inputs;
  inputs.raw_qkvz.resize(
      kernels::kSm87BulkV2GdnC64Tokens *
      kernels::kSm87TargetAotGdnRawQkvZChannels);
  inputs.interleaved_ab.resize(
      kernels::kSm87BulkV2GdnC64Tokens *
      kernels::kSm87TargetAotGdnAbChannels);
  inputs.conv_weight.resize(kernels::kSm87TargetAotGdnConvWeightElements);
  inputs.initial_conv_history.resize(
      kernels::kSm87TargetAotGdnConvHistoryElements);
  inputs.a_log.resize(kernels::kSm87TargetAotGdnScalarHeadElements);
  inputs.dt_bias.resize(kernels::kSm87TargetAotGdnScalarHeadElements);
  inputs.norm_weight.resize(kernels::kSm87TargetAotGdnNormWeightElements);
  inputs.initial_recurrent_state.resize(
      kernels::kSm87TargetAotGdnRecurrentStateElements);

  for (std::size_t index = 0U; index < inputs.raw_qkvz.size(); ++index) {
    inputs.raw_qkvz[index] = choose_pattern(signals, index, 0x101U);
  }
  for (std::size_t index = 0U; index < inputs.conv_weight.size(); ++index) {
    inputs.conv_weight[index] = choose_pattern(weights, index, 0x202U);
  }
  for (std::size_t index = 0U;
       index < inputs.initial_conv_history.size(); ++index) {
    inputs.initial_conv_history[index] =
        choose_pattern(signals, index, 0x303U);
  }
  for (std::size_t index = 0U;
       index < inputs.initial_recurrent_state.size(); ++index) {
    inputs.initial_recurrent_state[index] =
        choose_pattern(state_values, index, 0x404U);
  }

  for (std::size_t token = 0U;
       token < kernels::kSm87BulkV2GdnC64Tokens; ++token) {
    const std::size_t row = token * kernels::kSm87TargetAotGdnAbChannels;
    for (std::size_t head = 0U;
         head < kernels::kSm87TargetAotGdnValueHeads; ++head) {
      inputs.interleaved_ab[row + head] = host_encode_bf16_rne(
          a_values[(token * 7U + head * 5U) % a_values.size()]);
      inputs.interleaved_ab[
          row + kernels::kSm87TargetAotGdnValueHeads + head] =
          host_encode_bf16_rne(
              b_values[(token * 3U + head * 7U) % b_values.size()]);
    }
  }
  for (std::size_t head = 0U;
       head < kernels::kSm87TargetAotGdnValueHeads; ++head) {
    inputs.a_log[head] =
        host_encode_bf16_rne(a_log_values[head % a_log_values.size()]);
    inputs.dt_bias[head] =
        host_encode_bf16_rne(dt_values[(head * 3U) % dt_values.size()]);
  }
  for (std::size_t dimension = 0U;
       dimension < kernels::kSm87TargetAotGdnNormWeightElements;
       ++dimension) {
    inputs.norm_weight[dimension] = host_encode_bf16_rne(
        norm_values[(dimension * 5U) % norm_values.size()]);
  }
  return inputs;
}

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const std::string& operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << "FAIL: " << operation << ": " << cudaGetErrorString(status)
            << '\n';
  return false;
}

template <class T>
[[nodiscard]] bool allocate_buffer(DeviceBuffer<T>& buffer,
                                   const std::size_t count,
                                   const char* const name) {
  return cuda_ok(buffer.allocate(count), std::string("allocate ") + name);
}

struct DeviceInputs final {
  DeviceBuffer<std::uint16_t> raw_qkvz;
  DeviceBuffer<std::uint16_t> interleaved_ab;
  DeviceBuffer<std::uint16_t> conv_weight;
  DeviceBuffer<std::uint16_t> initial_conv_history;
  DeviceBuffer<std::uint16_t> a_log;
  DeviceBuffer<std::uint16_t> dt_bias;
  DeviceBuffer<std::uint16_t> norm_weight;
  DeviceBuffer<std::uint16_t> initial_recurrent_state;
};

[[nodiscard]] bool allocate_and_upload_inputs(DeviceInputs& device,
                                              const HostInputs& host,
                                              cudaStream_t stream) {
  bool ready = true;
  ready = allocate_buffer(device.raw_qkvz, host.raw_qkvz.size(), "raw_qkvz") &&
          ready;
  ready = allocate_buffer(device.interleaved_ab, host.interleaved_ab.size(),
                          "interleaved_ab") &&
          ready;
  ready = allocate_buffer(device.conv_weight, host.conv_weight.size(),
                          "conv_weight") &&
          ready;
  ready = allocate_buffer(device.initial_conv_history,
                          host.initial_conv_history.size(),
                          "initial_conv_history") &&
          ready;
  ready = allocate_buffer(device.a_log, host.a_log.size(), "a_log") && ready;
  ready = allocate_buffer(device.dt_bias, host.dt_bias.size(), "dt_bias") &&
          ready;
  ready = allocate_buffer(device.norm_weight, host.norm_weight.size(),
                          "norm_weight") &&
          ready;
  ready = allocate_buffer(device.initial_recurrent_state,
                          host.initial_recurrent_state.size(),
                          "initial_recurrent_state") &&
          ready;
  if (!ready) {
    return false;
  }
  ready = cuda_ok(device.raw_qkvz.upload(host.raw_qkvz, stream),
                  "upload raw_qkvz") &&
          ready;
  ready = cuda_ok(device.interleaved_ab.upload(host.interleaved_ab, stream),
                  "upload interleaved_ab") &&
          ready;
  ready = cuda_ok(device.conv_weight.upload(host.conv_weight, stream),
                  "upload conv_weight") &&
          ready;
  ready = cuda_ok(device.initial_conv_history.upload(
                      host.initial_conv_history, stream),
                  "upload initial_conv_history") &&
          ready;
  ready = cuda_ok(device.a_log.upload(host.a_log, stream), "upload a_log") &&
          ready;
  ready = cuda_ok(device.dt_bias.upload(host.dt_bias, stream),
                  "upload dt_bias") &&
          ready;
  ready = cuda_ok(device.norm_weight.upload(host.norm_weight, stream),
                  "upload norm_weight") &&
          ready;
  ready = cuda_ok(device.initial_recurrent_state.upload(
                      host.initial_recurrent_state, stream),
                  "upload initial_recurrent_state") &&
          ready;
  return ready;
}

struct CandidateBuffers final {
  DeviceBuffer<float> normalized_q;
  DeviceBuffer<float> normalized_k;
  DeviceBuffer<std::uint16_t> prepared_v;
  DeviceBuffer<float> alpha;
  DeviceBuffer<float> beta;
  DeviceBuffer<std::uint16_t> raw_output;
  DeviceBuffer<std::uint16_t> output;
  DeviceBuffer<std::uint16_t> final_conv_history;
  DeviceBuffer<std::uint16_t> final_recurrent_state;
  DeviceBuffer<std::uint16_t> state_trace;
};

[[nodiscard]] bool allocate_candidate(CandidateBuffers& buffers) {
  bool ready = true;
  ready = allocate_buffer(
              buffers.normalized_q,
              kernels::kSm87BulkV2GdnNormalizedQBytes / sizeof(float),
              "candidate normalized_q") &&
          ready;
  ready = allocate_buffer(
              buffers.normalized_k,
              kernels::kSm87BulkV2GdnNormalizedKBytes / sizeof(float),
              "candidate normalized_k") &&
          ready;
  ready = allocate_buffer(
              buffers.prepared_v,
              kernels::kSm87BulkV2GdnPreparedVBytes / sizeof(std::uint16_t),
              "candidate prepared_v") &&
          ready;
  ready = allocate_buffer(buffers.alpha,
                          kernels::kSm87BulkV2GdnAlphaBytes / sizeof(float),
                          "candidate alpha") &&
          ready;
  ready = allocate_buffer(buffers.beta,
                          kernels::kSm87BulkV2GdnBetaBytes / sizeof(float),
                          "candidate beta") &&
          ready;
  ready = allocate_buffer(
              buffers.raw_output,
              kernels::kSm87BulkV2GdnRawOutputBytes / sizeof(std::uint16_t),
              "candidate raw_output") &&
          ready;
  ready = allocate_buffer(buffers.output, kOutputElements,
                          "candidate output") &&
          ready;
  ready = allocate_buffer(buffers.final_conv_history, kConvHistoryElements,
                          "candidate final_conv_history") &&
          ready;
  ready = allocate_buffer(buffers.final_recurrent_state, kStateElements,
                          "candidate final_recurrent_state") &&
          ready;
  ready = allocate_buffer(buffers.state_trace, kTraceElements,
                          "candidate state_trace") &&
          ready;
  return ready;
}

struct ReferenceBuffers final {
  DeviceBuffer<std::uint16_t> convolved_qkv;
  DeviceBuffer<float> normalized_q;
  DeviceBuffer<float> normalized_k;
  DeviceBuffer<float> alpha;
  DeviceBuffer<float> beta;
  DeviceBuffer<std::uint16_t> raw_output;
  DeviceBuffer<std::uint16_t> output;
  DeviceBuffer<std::uint16_t> final_conv_history;
  DeviceBuffer<std::uint16_t> recurrent_state;
  DeviceBuffer<std::uint16_t> state_trace;
};

[[nodiscard]] bool allocate_reference(ReferenceBuffers& buffers) {
  bool ready = true;
  ready = allocate_buffer(
              buffers.convolved_qkv,
              kernels::kSm87BulkV2GdnC64Tokens * kConvChannels,
              "reference convolved_qkv") &&
          ready;
  ready = allocate_buffer(
              buffers.normalized_q,
              kernels::kSm87BulkV2GdnNormalizedQBytes / sizeof(float),
              "reference normalized_q") &&
          ready;
  ready = allocate_buffer(
              buffers.normalized_k,
              kernels::kSm87BulkV2GdnNormalizedKBytes / sizeof(float),
              "reference normalized_k") &&
          ready;
  ready = allocate_buffer(buffers.alpha,
                          kernels::kSm87BulkV2GdnAlphaBytes / sizeof(float),
                          "reference alpha") &&
          ready;
  ready = allocate_buffer(buffers.beta,
                          kernels::kSm87BulkV2GdnBetaBytes / sizeof(float),
                          "reference beta") &&
          ready;
  ready = allocate_buffer(
              buffers.raw_output,
              kernels::kSm87BulkV2GdnRawOutputBytes / sizeof(std::uint16_t),
              "reference raw_output") &&
          ready;
  ready = allocate_buffer(buffers.output, kOutputElements,
                          "reference output") &&
          ready;
  ready = allocate_buffer(buffers.final_conv_history, kConvHistoryElements,
                          "reference final_conv_history") &&
          ready;
  ready = allocate_buffer(buffers.recurrent_state, kStateElements,
                          "reference recurrent_state") &&
          ready;
  ready = allocate_buffer(buffers.state_trace, kTraceElements,
                          "reference state_trace") &&
          ready;
  return ready;
}

[[nodiscard]] bool poison_outputs(CandidateBuffers& candidate,
                                  ReferenceBuffers& reference,
                                  cudaStream_t stream) {
  bool ready = true;
  ready = cuda_ok(cudaMemsetAsync(candidate.output.data(), 0xa5,
                                  candidate.output.size() *
                                      sizeof(std::uint16_t),
                                  stream),
                  "poison candidate output") &&
          ready;
  ready = cuda_ok(cudaMemsetAsync(candidate.final_conv_history.data(), 0xa5,
                                  candidate.final_conv_history.size() *
                                      sizeof(std::uint16_t),
                                  stream),
                  "poison candidate history") &&
          ready;
  ready = cuda_ok(cudaMemsetAsync(candidate.final_recurrent_state.data(),
                                  0xa5,
                                  candidate.final_recurrent_state.size() *
                                      sizeof(std::uint16_t),
                                  stream),
                  "poison candidate final state") &&
          ready;
  ready = cuda_ok(cudaMemsetAsync(candidate.state_trace.data(), 0xa5,
                                  candidate.state_trace.size() *
                                      sizeof(std::uint16_t),
                                  stream),
                  "poison candidate state trace") &&
          ready;
  ready = cuda_ok(cudaMemsetAsync(reference.output.data(), 0x5a,
                                  reference.output.size() *
                                      sizeof(std::uint16_t),
                                  stream),
                  "poison reference output") &&
          ready;
  ready = cuda_ok(cudaMemsetAsync(reference.final_conv_history.data(), 0x5a,
                                  reference.final_conv_history.size() *
                                      sizeof(std::uint16_t),
                                  stream),
                  "poison reference history") &&
          ready;
  ready = cuda_ok(cudaMemsetAsync(reference.state_trace.data(), 0x5a,
                                  reference.state_trace.size() *
                                      sizeof(std::uint16_t),
                                  stream),
                  "poison reference state trace") &&
          ready;
  return ready;
}

[[nodiscard]] bool run_candidate(const DeviceInputs& inputs,
                                 CandidateBuffers& buffers,
                                 cudaStream_t stream) {
  kernels::Sm87BulkV2GdnC64Arguments arguments;
  arguments.raw_qkvz = inputs.raw_qkvz.data();
  arguments.interleaved_ab = inputs.interleaved_ab.data();
  arguments.conv_weight = inputs.conv_weight.data();
  arguments.initial_conv_history = inputs.initial_conv_history.data();
  arguments.a_log = inputs.a_log.data();
  arguments.dt_bias = inputs.dt_bias.data();
  arguments.norm_weight = inputs.norm_weight.data();
  arguments.initial_recurrent_state = inputs.initial_recurrent_state.data();
  arguments.l2_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.norm_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  // Exercise the continuation contract. Incoming state and convolution
  // history are deliberately nonzero and the candidate receives only the
  // local C64 activation slice, so position changes no numerical parent.
  arguments.first_position = kernels::kSm87BulkV2GdnC64Tokens;
  arguments.token_count = kernels::kSm87BulkV2GdnC64Tokens;
  arguments.normalized_q = buffers.normalized_q.data();
  arguments.normalized_k = buffers.normalized_k.data();
  arguments.prepared_v = buffers.prepared_v.data();
  arguments.alpha = buffers.alpha.data();
  arguments.beta = buffers.beta.data();
  arguments.raw_output = buffers.raw_output.data();
  arguments.output = buffers.output.data();
  arguments.final_conv_history = buffers.final_conv_history.data();
  arguments.final_recurrent_state = buffers.final_recurrent_state.data();
  arguments.cuda_stream = reinterpret_cast<void*>(stream);
  if (!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(arguments) ||
      !kernels::sm87_bulk_v2_gdn_c64_state_trace_valid(
          arguments, buffers.state_trace.data())) {
    std::cerr << "FAIL: candidate arguments or state trace do not validate\n";
    return false;
  }
  return cuda_ok(static_cast<cudaError_t>(
                     kernels::launch_sm87_bulk_dataflow_v2_gdn_c64_state_trace_cuda(
                         arguments, buffers.state_trace.data())),
                 "launch candidate state-trace cell") &&
         cuda_ok(cudaStreamSynchronize(stream), "synchronize candidate cell");
}

[[nodiscard]] bool run_reference(const DeviceInputs& inputs,
                                 ReferenceBuffers& buffers,
                                 cudaStream_t stream) {
  bool ready = cuda_ok(
      cudaMemcpyAsync(buffers.recurrent_state.data(),
                      inputs.initial_recurrent_state.data(),
                      kStateElements * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToDevice, stream),
      "seed reference recurrent state");
  constexpr unsigned int kConvThreads = 256U;
  const unsigned int conv_blocks = static_cast<unsigned int>(
      (kConvChannels + kConvThreads - 1U) / kConvThreads);
  reference_conv_silu_c64_kernel<<<conv_blocks, kConvThreads, 0U, stream>>>(
      inputs.raw_qkvz.data(), inputs.conv_weight.data(),
      inputs.initial_conv_history.data(), buffers.convolved_qkv.data(),
      buffers.final_conv_history.data());
  ready = cuda_ok(cudaPeekAtLastError(), "launch reference convolution") &&
          ready;
  const dim3 qk_grid(
      static_cast<unsigned int>(kernels::kSm87BulkV2GdnC64Tokens),
      static_cast<unsigned int>(kernels::kSm87TargetAotGdnQkGroups));
  reference_normalize_qk_c64_kernel<<<qk_grid, kWarpSize, 0U, stream>>>(
      buffers.convolved_qkv.data(),
      kernels::kSm87TargetAotGdnEpsilonFp32Bits,
      buffers.normalized_q.data(), buffers.normalized_k.data());
  ready = cuda_ok(cudaPeekAtLastError(), "launch reference QK normalization") &&
          ready;
  constexpr unsigned int kScalarThreads = 256U;
  constexpr unsigned int kScalarCount =
      kernels::kSm87BulkV2GdnC64Tokens *
      kernels::kSm87TargetAotGdnValueHeads;
  constexpr unsigned int kScalarBlocks =
      (kScalarCount + kScalarThreads - 1U) / kScalarThreads;
  reference_gate_scalars_c64_kernel<<<kScalarBlocks, kScalarThreads, 0U,
                                      stream>>>(
      inputs.interleaved_ab.data(), inputs.a_log.data(), inputs.dt_bias.data(),
      buffers.alpha.data(), buffers.beta.data());
  ready = cuda_ok(cudaPeekAtLastError(), "launch reference gate scalars") &&
          ready;
  reference_recurrence_c64_kernel<<<
      kernels::kSm87TargetAotGdnValueHeads, kReferenceRecurrenceThreads, 0U,
      stream>>>(buffers.convolved_qkv.data(), buffers.normalized_q.data(),
                buffers.normalized_k.data(), buffers.alpha.data(),
                buffers.beta.data(), buffers.recurrent_state.data(),
                buffers.raw_output.data(), buffers.state_trace.data());
  ready = cuda_ok(cudaPeekAtLastError(), "launch reference recurrence") &&
          ready;
  constexpr unsigned int kNormRows =
      kernels::kSm87BulkV2GdnC64Tokens *
      kernels::kSm87TargetAotGdnValueHeads;
  reference_norm_gate_c64_kernel<<<kNormRows, kWarpSize, 0U, stream>>>(
      buffers.raw_output.data(), inputs.raw_qkvz.data(),
      inputs.norm_weight.data(), kernels::kSm87TargetAotGdnEpsilonFp32Bits,
      buffers.output.data());
  ready = cuda_ok(cudaPeekAtLastError(), "launch reference norm/gate") &&
          ready;
  return cuda_ok(cudaStreamSynchronize(stream), "synchronize reference") &&
         ready;
}

[[nodiscard]] bool compare_complete_surface(
    const char* const label,
    const std::uint16_t* const candidate,
    const std::uint16_t* const reference,
    const std::size_t count,
    cudaStream_t stream) {
  DeviceBuffer<MismatchRecord> device_record;
  if (!allocate_buffer(device_record, 1U, "mismatch record")) {
    return false;
  }
  const MismatchRecord initial{0ULL, ULLONG_MAX};
  if (!cuda_ok(device_record.upload_one(initial, stream),
               std::string("initialize ") + label + " mismatch record")) {
    return false;
  }
  compare_bf16_kernel<<<kCompareBlocks, kCompareThreads, 0U, stream>>>(
      candidate, reference, count, device_record.data());
  if (!cuda_ok(cudaPeekAtLastError(), std::string("launch compare ") + label)) {
    return false;
  }
  MismatchRecord result{};
  if (!cuda_ok(device_record.download_one(&result, stream),
               std::string("download ") + label + " mismatch record") ||
      !cuda_ok(cudaStreamSynchronize(stream),
               std::string("synchronize compare ") + label)) {
    return false;
  }
  if (result.mismatch_count == 0ULL) {
    return true;
  }

  std::uint16_t candidate_value = 0U;
  std::uint16_t reference_value = 0U;
  const auto first = static_cast<std::size_t>(result.first_index);
  bool ready = cuda_ok(cudaMemcpy(&candidate_value, candidate + first,
                                  sizeof(candidate_value),
                                  cudaMemcpyDeviceToHost),
                       std::string("read candidate ") + label) &&
               cuda_ok(cudaMemcpy(&reference_value, reference + first,
                                  sizeof(reference_value),
                                  cudaMemcpyDeviceToHost),
                       std::string("read reference ") + label);
  if (ready) {
    std::cerr << "FAIL: " << label << " has " << result.mismatch_count
              << " BF16 bit mismatches; first index " << first
              << " candidate=0x" << std::hex << candidate_value
              << " reference=0x" << reference_value << std::dec << '\n';
  }
  return false;
}

}  // namespace

int main() {
  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA device unavailable for SM87 GDN C64 oracle\n";
    (void)cudaGetLastError();
    return 77;
  }
  int device = -1;
  if (!cuda_ok(cudaGetDevice(&device), "get CUDA device")) {
    return 1;
  }
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDeviceProperties(&properties, device),
               "query CUDA device")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: oracle requires the fixed 16-SM SM87 target\n";
    return 77;
  }

  cudaStream_t stream = nullptr;
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create oracle stream")) {
    return 1;
  }

  bool passed = true;
  {
    const HostInputs host = make_adversarial_inputs();
    DeviceInputs inputs;
    CandidateBuffers candidate;
    ReferenceBuffers reference;
    passed = allocate_and_upload_inputs(inputs, host, stream);
    passed = allocate_candidate(candidate) && passed;
    passed = allocate_reference(reference) && passed;
    if (passed) {
      passed = poison_outputs(candidate, reference, stream);
    }
    if (passed) {
      passed = run_candidate(inputs, candidate, stream);
    }
    if (passed) {
      passed = run_reference(inputs, reference, stream);
    }
    if (passed) {
      passed = compare_complete_surface(
                   "raw recurrence output", candidate.raw_output.data(),
                   reference.raw_output.data(),
                   kernels::kSm87BulkV2GdnRawOutputBytes /
                       sizeof(std::uint16_t),
                   stream) &&
               passed;
      passed = compare_complete_surface(
                   "final output", candidate.output.data(),
                   reference.output.data(), kOutputElements, stream) &&
               passed;
      passed = compare_complete_surface(
                   "per-token recurrent-state trace",
                   candidate.state_trace.data(), reference.state_trace.data(),
                   kTraceElements, stream) &&
               passed;
      passed = compare_complete_surface(
                   "final recurrent state",
                   candidate.final_recurrent_state.data(),
                   reference.recurrent_state.data(), kStateElements, stream) &&
               passed;
      passed = compare_complete_surface(
                   "final convolution history",
                   candidate.final_conv_history.data(),
                   reference.final_conv_history.data(), kConvHistoryElements,
                   stream) &&
               passed;
    }
  }
  (void)cudaStreamDestroy(stream);
  if (!passed) {
    return 1;
  }
  std::cout << "SM87 bulk-dataflow-v2 GDN C64 synthetic bitwise CUDA oracle "
               "passed (correctness only; no performance or qualification "
               "authority)\n";
  return 0;
}
