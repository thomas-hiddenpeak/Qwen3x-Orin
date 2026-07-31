#include "gdn_prefill_exact_span_sm87.h"
#include "gdn_prefill_c16_norm_gate_sm87.h"
#include "gdn_prefill_whole_span_conv_sm87.h"

#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/gdn_decode.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

namespace {

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

  bool allocate(const std::size_t elements) {
    size_ = elements;
    return cudaMallocManaged(&data_, elements * sizeof(T)) == cudaSuccess;
  }
  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  T& operator[](const std::size_t index) noexcept { return data_[index]; }
  const T& operator[](const std::size_t index) const noexcept {
    return data_[index];
  }
  std::size_t size() const noexcept { return size_; }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0U;
};

std::uint16_t encode_bf16(const float value) {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

bool equal_words(const char* const label,
                 const std::uint16_t* const expected,
                 const std::uint16_t* const actual,
                 const std::size_t elements) {
  for (std::size_t index = 0U; index < elements; ++index) {
    if (expected[index] != actual[index]) {
      std::cerr << label << " mismatch index=" << index
                << " expected=0x" << std::hex << expected[index]
                << " actual=0x" << actual[index] << std::dec << '\n';
      return false;
    }
  }
  return true;
}

void fill_inputs(ManagedBuffer<std::uint16_t>& conv_qkv,
                 ManagedBuffer<std::uint16_t>& a,
                 ManagedBuffer<std::uint16_t>& b,
                 ManagedBuffer<std::uint16_t>& A_log,
                 ManagedBuffer<std::uint16_t>& dt_bias,
                 ManagedBuffer<std::uint16_t>& initial_state,
                 const std::size_t token_count) {
  constexpr std::size_t kKOffset = q3x::runtime::kGdnQElements;
  constexpr std::size_t kVOffset =
      q3x::runtime::kGdnQElements + q3x::runtime::kGdnKElements;
  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t qkv_base = token * q3x::runtime::kGdnQkvChannels;
    const std::size_t scalar_base =
        token * q3x::runtime::kGdnValueHeadCount;
    for (std::size_t head = 0U;
         head < q3x::runtime::kGdnQkHeadCount; ++head) {
      for (std::size_t dimension = 0U;
           dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
        const int q = static_cast<int>(
                          (token * 5U + head * 11U + dimension * 3U) %
                          29U) -
                      14;
        const int k = static_cast<int>(
                          (token * 2U + head * 7U + dimension * 5U) %
                          31U) -
                      15;
        conv_qkv[qkv_base + head * q3x::runtime::kGdnHeadDimension +
                 dimension] = encode_bf16(static_cast<float>(q) / 16.0F);
        conv_qkv[qkv_base + kKOffset +
                 head * q3x::runtime::kGdnHeadDimension + dimension] =
            encode_bf16(static_cast<float>(k) / 16.0F);
      }
    }
    for (std::size_t index = 0U; index < q3x::runtime::kGdnVElements;
         ++index) {
      const int value =
          static_cast<int>((token * 7U + index * 13U) % 37U) - 18;
      conv_qkv[qkv_base + kVOffset + index] =
          encode_bf16(static_cast<float>(value) / 16.0F);
    }
    for (std::size_t head = 0U;
         head < q3x::runtime::kGdnValueHeadCount; ++head) {
      a[scalar_base + head] = encode_bf16(
          static_cast<float>(static_cast<int>((head + token) % 9U) - 4) *
          0.25F);
      b[scalar_base + head] = encode_bf16(
          static_cast<float>(
              static_cast<int>((head + 2U * token) % 11U) - 5) *
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
  A_log[2U] = encode_bf16(4.0F);
  for (std::size_t index = 0U; index < initial_state.size(); ++index) {
    const int value = static_cast<int>((index * 17U + index / 97U) % 41U) -
                      20;
    initial_state[index] =
        encode_bf16(static_cast<float>(value) / 64.0F);
  }
}

bool run_shape(const std::size_t token_count, cudaStream_t stream) {
  const std::size_t qkv_elements =
      token_count * q3x::runtime::kGdnQkvChannels;
  const std::size_t scalar_elements =
      token_count * q3x::runtime::kGdnValueHeadCount;
  const std::size_t output_elements =
      token_count * q3x::runtime::kGdnVElements;
  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> initial_state;
  ManagedBuffer<std::uint16_t> baseline_state;
  ManagedBuffer<std::uint16_t> candidate_state;
  ManagedBuffer<std::uint16_t> baseline_output;
  ManagedBuffer<std::uint16_t> candidate_output;
  if (!conv_qkv.allocate(qkv_elements) || !a.allocate(scalar_elements) ||
      !b.allocate(scalar_elements) ||
      !A_log.allocate(q3x::runtime::kGdnValueHeadCount) ||
      !dt_bias.allocate(q3x::runtime::kGdnValueHeadCount) ||
      !initial_state.allocate(q3x::runtime::kGdnStateElements) ||
      !baseline_state.allocate(q3x::runtime::kGdnStateElements) ||
      !candidate_state.allocate(q3x::runtime::kGdnStateElements) ||
      !baseline_output.allocate(output_elements) ||
      !candidate_output.allocate(output_elements)) {
    std::cerr << "allocation failed for C" << token_count << '\n';
    return false;
  }
  fill_inputs(conv_qkv, a, b, A_log, dt_bias, initial_state, token_count);
  std::memcpy(baseline_state.data(), initial_state.data(),
              initial_state.size() * sizeof(std::uint16_t));
  std::memcpy(candidate_state.data(), initial_state.data(),
              initial_state.size() * sizeof(std::uint16_t));
  std::memset(baseline_output.data(), 0xa5,
              baseline_output.size() * sizeof(std::uint16_t));
  std::memset(candidate_output.data(), 0x5a,
              candidate_output.size() * sizeof(std::uint16_t));

  constexpr float kEpsilon = 1.0e-6F;
  for (std::size_t offset = 0U; offset < token_count; offset += 16U) {
    const int status =
        q3x::runtime::launch_gated_delta_net_update_tile_warp_parallel_cuda(
            conv_qkv.data() + offset * q3x::runtime::kGdnQkvChannels, 16U,
            a.data() + offset * q3x::runtime::kGdnValueHeadCount,
            b.data() + offset * q3x::runtime::kGdnValueHeadCount,
            A_log.data(), dt_bias.data(), baseline_state.data(),
            baseline_state.data(), kEpsilon,
            baseline_output.data() + offset * q3x::runtime::kGdnVElements,
            {}, static_cast<void*>(stream));
    if (status != static_cast<int>(cudaSuccess)) {
      std::cerr << "baseline launch failed C" << token_count
                << " offset=" << offset << " status=" << status << '\n';
      return false;
    }
  }
  const int candidate_status =
      q3x::runtime::gdn_prefill_exact_span_detail::
          launch_row16_register_baton(
              conv_qkv.data(), token_count, a.data(), b.data(), A_log.data(),
              dt_bias.data(), candidate_state.data(), candidate_state.data(),
              kEpsilon, candidate_output.data(),
              static_cast<void*>(stream));
  if (candidate_status != static_cast<int>(cudaSuccess)) {
    std::cerr << "candidate launch failed C" << token_count
              << " status=" << candidate_status << '\n';
    return false;
  }
  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    std::cerr << "synchronize failed C" << token_count << ": "
              << cudaGetErrorString(sync_status) << '\n';
    return false;
  }
  return equal_words("state", baseline_state.data(), candidate_state.data(),
                     baseline_state.size()) &&
         equal_words("output", baseline_output.data(),
                     candidate_output.data(), baseline_output.size());
}

bool run_integrated_c512(cudaStream_t stream) {
  constexpr std::size_t kTokenCount = 512U;
  constexpr std::size_t kQkvElements =
      kTokenCount * q3x::runtime::kGdnQkvChannels;
  constexpr std::size_t kScalarElements =
      kTokenCount * q3x::runtime::kGdnValueHeadCount;
  constexpr std::size_t kOutputElements =
      kTokenCount * q3x::runtime::kGdnVElements;
  constexpr std::size_t kConvWeightElements =
      q3x::runtime::kGdnQkvChannels * q3x::runtime::kGdnConvKernelWidth;
  constexpr std::size_t kHistoryElements =
      q3x::runtime::kGdnQkvChannels * q3x::runtime::kGdnConvHistoryWidth;
  constexpr float kEpsilon = 1.0e-6F;

  ManagedBuffer<std::uint16_t> raw_qkv;
  ManagedBuffer<std::uint16_t> baseline_qkv;
  ManagedBuffer<std::uint16_t> candidate_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> initial_state;
  ManagedBuffer<std::uint16_t> baseline_state;
  ManagedBuffer<std::uint16_t> candidate_state;
  ManagedBuffer<std::uint16_t> conv_weight;
  ManagedBuffer<std::uint16_t> initial_history;
  ManagedBuffer<std::uint16_t> baseline_history;
  ManagedBuffer<std::uint16_t> candidate_history;
  ManagedBuffer<std::uint16_t> norm_weight;
  ManagedBuffer<std::uint16_t> silu_gate;
  ManagedBuffer<std::uint16_t> baseline_output;
  ManagedBuffer<std::uint16_t> candidate_output;
  if (!raw_qkv.allocate(kQkvElements) ||
      !baseline_qkv.allocate(kQkvElements) ||
      !candidate_qkv.allocate(kQkvElements) ||
      !a.allocate(kScalarElements) || !b.allocate(kScalarElements) ||
      !A_log.allocate(q3x::runtime::kGdnValueHeadCount) ||
      !dt_bias.allocate(q3x::runtime::kGdnValueHeadCount) ||
      !initial_state.allocate(q3x::runtime::kGdnStateElements) ||
      !baseline_state.allocate(q3x::runtime::kGdnStateElements) ||
      !candidate_state.allocate(q3x::runtime::kGdnStateElements) ||
      !conv_weight.allocate(kConvWeightElements) ||
      !initial_history.allocate(kHistoryElements) ||
      !baseline_history.allocate(kHistoryElements) ||
      !candidate_history.allocate(kHistoryElements) ||
      !norm_weight.allocate(q3x::runtime::kGdnHeadDimension) ||
      !silu_gate.allocate(kOutputElements) ||
      !baseline_output.allocate(kOutputElements) ||
      !candidate_output.allocate(kOutputElements)) {
    std::cerr << "integrated C512 allocation failed\n";
    return false;
  }
  fill_inputs(raw_qkv, a, b, A_log, dt_bias, initial_state, kTokenCount);
  for (std::size_t index = 0U; index < conv_weight.size(); ++index) {
    const int value = static_cast<int>((index * 5U + index / 31U) % 17U) - 8;
    conv_weight[index] = encode_bf16(static_cast<float>(value) / 32.0F);
  }
  for (std::size_t index = 0U; index < initial_history.size(); ++index) {
    const int value = static_cast<int>((index * 7U + 3U) % 19U) - 9;
    initial_history[index] =
        encode_bf16(static_cast<float>(value) / 16.0F);
  }
  for (std::size_t index = 0U; index < norm_weight.size(); ++index) {
    norm_weight[index] = encode_bf16(
        0.75F + static_cast<float>(index % 13U) / 32.0F);
  }
  for (std::size_t index = 0U; index < silu_gate.size(); ++index) {
    const int value = static_cast<int>((index * 3U + index / 101U) % 23U) -
                      11;
    silu_gate[index] = encode_bf16(static_cast<float>(value) / 8.0F);
  }
  std::memcpy(baseline_qkv.data(), raw_qkv.data(),
              raw_qkv.size() * sizeof(std::uint16_t));
  std::memcpy(candidate_qkv.data(), raw_qkv.data(),
              raw_qkv.size() * sizeof(std::uint16_t));
  std::memcpy(baseline_state.data(), initial_state.data(),
              initial_state.size() * sizeof(std::uint16_t));
  std::memcpy(candidate_state.data(), initial_state.data(),
              initial_state.size() * sizeof(std::uint16_t));
  std::memcpy(baseline_history.data(), initial_history.data(),
              initial_history.size() * sizeof(std::uint16_t));
  std::memcpy(candidate_history.data(), initial_history.data(),
              initial_history.size() * sizeof(std::uint16_t));

  for (std::size_t offset = 0U; offset < kTokenCount; offset += 16U) {
    const std::size_t qkv_offset = offset * q3x::runtime::kGdnQkvChannels;
    const std::size_t scalar_offset =
        offset * q3x::runtime::kGdnValueHeadCount;
    const std::size_t output_offset =
        offset * q3x::runtime::kGdnVElements;
    const int conv_status =
        q3x::runtime::launch_causal_conv1d_silu_update_tile_reference_cuda(
            baseline_qkv.data() + qkv_offset, 16U, conv_weight.data(),
            baseline_history.data(), baseline_qkv.data() + qkv_offset, {},
            static_cast<void*>(stream));
    const int gdn_status =
        q3x::runtime::gdn_prefill_c16_norm_gate_detail::
            launch_shared_boundary(
                baseline_qkv.data() + qkv_offset, 16U,
                a.data() + scalar_offset, b.data() + scalar_offset,
                A_log.data(), dt_bias.data(), baseline_state.data(),
                baseline_state.data(), kEpsilon, norm_weight.data(),
                silu_gate.data() + output_offset, kEpsilon,
                baseline_output.data() + output_offset,
                static_cast<void*>(stream));
    if (conv_status != static_cast<int>(cudaSuccess) ||
        gdn_status != static_cast<int>(cudaSuccess)) {
      std::cerr << "integrated baseline launch failed offset=" << offset
                << " conv=" << conv_status << " gdn=" << gdn_status
                << '\n';
      return false;
    }
  }
  const int conv_status =
      q3x::runtime::gdn_prefill_whole_span_conv_detail::
          launch_causal_conv1d_silu_update_whole_span_exact_cuda(
              candidate_qkv.data(), kTokenCount, conv_weight.data(),
              candidate_history.data(), candidate_qkv.data(),
              static_cast<void*>(stream));
  const int gdn_status =
      q3x::runtime::gdn_prefill_exact_span_detail::
          launch_row16_register_baton(
              candidate_qkv.data(), kTokenCount, a.data(), b.data(),
              A_log.data(), dt_bias.data(), candidate_state.data(),
              candidate_state.data(), kEpsilon, candidate_output.data(),
              static_cast<void*>(stream));
  const int norm_status =
      q3x::runtime::launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
          candidate_output.data(), norm_weight.data(), silu_gate.data(),
          kTokenCount * q3x::runtime::kGdnValueHeadCount,
          q3x::runtime::kGdnHeadDimension, kEpsilon,
          candidate_output.data(), static_cast<void*>(stream));
  if (conv_status != static_cast<int>(cudaSuccess) ||
      gdn_status != static_cast<int>(cudaSuccess) ||
      norm_status != static_cast<int>(cudaSuccess)) {
    std::cerr << "integrated candidate launch failed conv=" << conv_status
              << " gdn=" << gdn_status << " norm=" << norm_status << '\n';
    return false;
  }
  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    std::cerr << "integrated synchronize failed: "
              << cudaGetErrorString(sync_status) << '\n';
    return false;
  }
  return equal_words("integrated convolved QKV", baseline_qkv.data(),
                     candidate_qkv.data(), baseline_qkv.size()) &&
         equal_words("integrated history", baseline_history.data(),
                     candidate_history.data(), baseline_history.size()) &&
         equal_words("integrated state", baseline_state.data(),
                     candidate_state.data(), baseline_state.size()) &&
         equal_words("integrated normalized output", baseline_output.data(),
                     candidate_output.data(), baseline_output.size());
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: exact-span GDN component requires CUDA\n";
    return 77;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
      properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: exact-span GDN component requires SM87\n";
    return 77;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    return 1;
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    return 1;
  }

  int registers = 0;
  std::size_t shared = 0U;
  std::size_t local = 0U;
  int maximum_threads = 0;
  int active_blocks = 0;
  const int resource_status =
      q3x::runtime::gdn_prefill_exact_span_detail::
          query_row16_register_baton_resources(
              &registers, &shared, &local, &maximum_threads, &active_blocks);
  const bool resources_ok =
      resource_status == static_cast<int>(cudaSuccess) && registers <= 85 &&
      shared <= 2048U && local == 0U && maximum_threads >= 256 &&
      active_blocks >= 3;
  std::cout << "GDN_EXACT_SPAN_RESOURCES registers=" << registers
            << " shared=" << shared << " local=" << local
            << " max_threads=" << maximum_threads
            << " active_blocks_per_sm=" << active_blocks << '\n';

  const bool c16 = run_shape(16U, stream);
  const bool c32 = run_shape(32U, stream);
  const bool c512 = run_shape(512U, stream);
  const bool integrated_c512 = run_integrated_c512(stream);
  (void)cudaStreamDestroy(stream);
  if (!resources_ok || !c16 || !c32 || !c512 || !integrated_c512) {
    return 1;
  }
  std::cout << "GDN exact-span row16 component passed\n";
  return 0;
}
