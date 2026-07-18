#include "q3x/runtime/gdn_decode.h"

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

namespace q3x::runtime {

// Test-only entry points linked beside production for exact and mirrored A/B.
[[nodiscard]] int launch_gated_delta_net_update_tile_warp_baseline_test_cuda(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_gated_delta_net_update_tile_warp_row_pair_test_cuda(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::runtime

namespace {

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

[[nodiscard]] float decode_bf16(const std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

template <typename Launch>
[[nodiscard]] bool launch_after_stale(TestContext& test, cudaStream_t stream,
                                      const std::string& label,
                                      Launch&& launch) {
  const cudaError_t stale =
      cudaMemcpy(nullptr, nullptr, 1U, cudaMemcpyHostToDevice);
  test.expect(stale == cudaErrorInvalidValue,
              label + " seeds stale CUDA last-error");
  bool ready = test.cuda_ok(static_cast<cudaError_t>(launch()),
                            label + " launch ignores stale error");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  return ready;
}

template <typename Launch>
[[nodiscard]] float measure_average_cuda_milliseconds(
    TestContext& test, cudaStream_t stream, const std::size_t warmup_count,
    const std::size_t iteration_count, const std::string& label,
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
  for (std::size_t iteration = 0U; iteration < warmup_count; ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch()),
                         label + " warmup launch");
    if (!ready) {
      break;
    }
  }
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync");
  ready = ready &&
          test.cuda_ok(cudaEventRecord(start, stream), label + " record start");
  for (std::size_t iteration = 0U; ready && iteration < iteration_count;
       ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch()),
                         label + " measured launch");
  }
  ready = ready &&
          test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop");
  ready = ready &&
          test.cuda_ok(cudaEventSynchronize(stop), label + " synchronize stop");
  float elapsed = std::numeric_limits<float>::quiet_NaN();
  if (ready) {
    float total = 0.0F;
    ready = test.cuda_ok(cudaEventElapsedTime(&total, start, stop),
                         label + " elapsed time");
    if (ready) {
      elapsed = total / static_cast<float>(iteration_count);
    }
  }
  (void)test.cuda_ok(cudaEventDestroy(start), label + " destroy start");
  (void)test.cuda_ok(cudaEventDestroy(stop), label + " destroy stop");
  return elapsed;
}

void expect_bf16_buffer_near(TestContext& test,
                             const std::uint16_t* const actual,
                             const std::uint16_t* const expected,
                             const std::size_t count,
                             const float absolute_tolerance,
                             const float relative_tolerance,
                             const std::string& label) {
  std::size_t failures = 0U;
  float maximum_error = 0.0F;
  std::size_t first_failure = 0U;
  for (std::size_t index = 0; index < count; ++index) {
    const float actual_value = decode_bf16(actual[index]);
    const float expected_value = decode_bf16(expected[index]);
    bool matches = false;
    float error = 0.0F;
    if (std::isnan(expected_value)) {
      matches = std::isnan(actual_value);
    } else if (std::isinf(expected_value)) {
      matches = std::isinf(actual_value) &&
                std::signbit(actual_value) == std::signbit(expected_value);
    } else {
      error = std::fabs(actual_value - expected_value);
      const float tolerance =
          absolute_tolerance + relative_tolerance * std::fabs(expected_value);
      matches = error <= tolerance;
    }
    maximum_error = std::max(maximum_error, error);
    if (!matches) {
      if (failures == 0U) {
        first_failure = index;
      }
      ++failures;
    }
  }
  test.expect(failures == 0U,
              label + " mismatches=" + std::to_string(failures) +
                  ", first=" + std::to_string(first_failure) +
                  ", max_abs_error=" + std::to_string(maximum_error));
}

void expect_bf16_buffer_bitwise_equal(
    TestContext& test, const std::uint16_t* const actual,
    const std::uint16_t* const expected, const std::size_t count,
    const std::string& label) {
  std::size_t mismatch_count = 0U;
  std::size_t first_mismatch = 0U;
  for (std::size_t index = 0U; index < count; ++index) {
    if (actual[index] != expected[index]) {
      if (mismatch_count == 0U) {
        first_mismatch = index;
      }
      ++mismatch_count;
    }
  }
  std::string detail =
      label + " mismatches=" + std::to_string(mismatch_count);
  if (mismatch_count != 0U) {
    detail += ", first=" + std::to_string(first_mismatch) +
              ", actual_bits=" + std::to_string(actual[first_mismatch]) +
              ", expected_bits=" + std::to_string(expected[first_mismatch]);
  }
  test.expect(mismatch_count == 0U, detail);
}

void test_launch_validation(TestContext& test) {
  const q3x::runtime::GdnDimensions wrong{15U, 48U, 128U};
  const q3x::runtime::GdnDimensions overflow{
      std::numeric_limits<std::size_t>::max(), 48U, 128U};
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::
                      launch_causal_conv1d_silu_update_reference_cuda(
                          nullptr, nullptr, nullptr, nullptr, wrong)) ==
                  cudaErrorInvalidValue,
              "CUDA conv rejects wrong dimensions");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::
                      launch_causal_conv1d_silu_update_reference_cuda(
                          nullptr, nullptr, nullptr, nullptr, overflow)) ==
                  cudaErrorInvalidValue,
              "CUDA conv rejects overflowing dimensions");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_gated_delta_net_update_reference_cuda(
                      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                      nullptr, std::numeric_limits<float>::quiet_NaN(),
                      nullptr)) == cudaErrorInvalidValue,
              "CUDA GDN rejects NaN epsilon");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_gated_delta_net_update_reference_cuda(
                      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                      nullptr, std::numeric_limits<float>::infinity(),
                      nullptr)) == cudaErrorInvalidValue,
              "CUDA GDN rejects infinite epsilon");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::
              launch_causal_conv1d_silu_update_tile_reference_cuda(
                  nullptr, 0U, nullptr, nullptr, nullptr)) ==
          cudaErrorInvalidValue,
      "CUDA tile conv rejects zero tokens");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::
              launch_causal_conv1d_silu_update_tile_reference_cuda(
                  nullptr, 1U, nullptr, nullptr, nullptr)) ==
          cudaErrorInvalidValue,
      "CUDA tile conv M=1 rejects null buffers");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::
              launch_causal_conv1d_silu_update_tile_reference_cuda(
                  nullptr, q3x::runtime::kGdnMaximumTileTokenCount + 1U,
                  nullptr, nullptr, nullptr)) == cudaErrorInvalidValue,
      "CUDA tile conv rejects oversized tile");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::
              launch_causal_conv1d_silu_update_tile_reference_cuda(
                  nullptr, 2U, nullptr, nullptr, nullptr, wrong)) ==
          cudaErrorInvalidValue,
      "CUDA tile conv rejects wrong dimensions");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_gated_delta_net_update_tile_reference_cuda(
              nullptr, 0U, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 1.0e-6F, nullptr)) == cudaErrorInvalidValue,
      "CUDA tile GDN rejects zero tokens");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_gated_delta_net_update_tile_reference_cuda(
              nullptr, 1U, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 1.0e-6F, nullptr)) == cudaErrorInvalidValue,
      "CUDA tile GDN M=1 rejects null buffers");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_gated_delta_net_update_tile_reference_cuda(
              nullptr, q3x::runtime::kGdnMaximumTileTokenCount + 1U,
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1.0e-6F,
              nullptr)) == cudaErrorInvalidValue,
      "CUDA tile GDN rejects oversized tile");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_gated_delta_net_update_tile_reference_cuda(
              nullptr, 2U, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, std::numeric_limits<float>::quiet_NaN(), nullptr)) ==
          cudaErrorInvalidValue,
      "CUDA tile GDN rejects NaN epsilon");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_gated_delta_net_update_warp_parallel_cuda(
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              1.0e-6F, nullptr)) == cudaErrorInvalidValue,
      "CUDA warp-parallel GDN rejects null buffers");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::
              launch_gated_delta_net_update_tile_warp_parallel_cuda(
                  nullptr, q3x::runtime::kGdnMaximumTileTokenCount + 1U,
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                  1.0e-6F, nullptr)) == cudaErrorInvalidValue,
      "CUDA warp-parallel tile GDN rejects oversized tile");
}

void test_conv_tile_bitwise(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kMaximumTokens =
      q3x::runtime::kGdnMaximumTileTokenCount;
  constexpr std::size_t kTileElements =
      kMaximumTokens * q3x::runtime::kGdnQkvChannels;
  constexpr std::size_t kHistoryElements =
      q3x::runtime::kGdnQkvChannels *
      q3x::runtime::kGdnConvHistoryWidth;

  ManagedBuffer<std::uint16_t> sequential_qkv;
  ManagedBuffer<std::uint16_t> tile_qkv;
  ManagedBuffer<std::uint16_t> weight;
  ManagedBuffer<std::uint16_t> sequential_history;
  ManagedBuffer<std::uint16_t> tile_history;
  bool ready = test.cuda_ok(sequential_qkv.allocate(kTileElements),
                            "tile conv allocate sequential QKV");
  ready = ready && test.cuda_ok(tile_qkv.allocate(kTileElements),
                                "tile conv allocate tile QKV");
  ready = ready && test.cuda_ok(
                       weight.allocate(q3x::runtime::kGdnQkvChannels *
                                       q3x::runtime::kGdnConvKernelWidth),
                       "tile conv allocate weight");
  ready = ready && test.cuda_ok(sequential_history.allocate(kHistoryElements),
                                "tile conv allocate sequential history");
  ready = ready && test.cuda_ok(tile_history.allocate(kHistoryElements),
                                "tile conv allocate tile history");
  if (!ready) {
    return;
  }

  std::vector<std::uint16_t> initial_qkv(kTileElements);
  std::vector<std::uint16_t> initial_history(kHistoryElements);
  for (std::size_t index = 0U; index < weight.size(); ++index) {
    const int centered = static_cast<int>((index * 13U) % 29U) - 14;
    weight[index] = encode_bf16(static_cast<float>(centered) / 64.0F);
  }
  for (std::size_t token = 0U; token < kMaximumTokens; ++token) {
    for (std::size_t channel = 0U;
         channel < q3x::runtime::kGdnQkvChannels; ++channel) {
      const int centered = static_cast<int>(
                               (channel * 17U + token * 11U) % 43U) -
                           21;
      initial_qkv[token * q3x::runtime::kGdnQkvChannels + channel] =
          encode_bf16(static_cast<float>(centered) / 32.0F);
    }
  }
  for (std::size_t index = 0U; index < kHistoryElements; ++index) {
    const int centered = static_cast<int>((index * 7U) % 19U) - 9;
    initial_history[index] =
        encode_bf16(static_cast<float>(centered) / 128.0F);
  }

  for (std::size_t token_count = 1U; token_count <= kMaximumTokens;
       ++token_count) {
    std::copy(initial_qkv.begin(), initial_qkv.end(), sequential_qkv.data());
    std::copy(initial_qkv.begin(), initial_qkv.end(), tile_qkv.data());
    std::copy(initial_history.begin(), initial_history.end(),
              sequential_history.data());
    std::copy(initial_history.begin(), initial_history.end(),
              tile_history.data());

    for (std::size_t token = 0U; token < token_count; ++token) {
      std::uint16_t* const qkv =
          sequential_qkv.data() +
          token * q3x::runtime::kGdnQkvChannels;
      if (!test.cuda_ok(
              static_cast<cudaError_t>(
                  q3x::runtime::
                      launch_causal_conv1d_silu_update_reference_cuda(
                          qkv, weight.data(), sequential_history.data(), qkv,
                          {}, static_cast<void*>(stream))),
              "tile conv sequential launch M=" +
                  std::to_string(token_count) +
                  " token=" + std::to_string(token))) {
        return;
      }
    }
    ready = launch_after_stale(
        test, stream, "causal conv tile M=" + std::to_string(token_count),
        [&]() {
          return q3x::runtime::
              launch_causal_conv1d_silu_update_tile_reference_cuda(
                  tile_qkv.data(), token_count, weight.data(),
                  tile_history.data(), tile_qkv.data(), {},
                  static_cast<void*>(stream));
        });
    if (!ready) {
      return;
    }
    expect_bf16_buffer_bitwise_equal(
        test, tile_qkv.data(), sequential_qkv.data(),
        token_count * q3x::runtime::kGdnQkvChannels,
        "CUDA causal conv tile in-place output M=" +
            std::to_string(token_count));
    expect_bf16_buffer_bitwise_equal(
        test, tile_history.data(), sequential_history.data(),
        kHistoryElements,
        "CUDA causal conv tile history M=" + std::to_string(token_count));
  }

  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::
              launch_causal_conv1d_silu_update_tile_reference_cuda(
                  tile_qkv.data(), 2U, weight.data(), tile_history.data(),
                  tile_history.data())) == cudaErrorInvalidValue,
      "CUDA tile conv rejects history/output alias");
}

void test_conv_multistep(TestContext& test, cudaStream_t stream) {
  ManagedBuffer<std::uint16_t> raw;
  ManagedBuffer<std::uint16_t> weight;
  ManagedBuffer<std::uint16_t> history;
  ManagedBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(raw.allocate(q3x::runtime::kGdnQkvChannels),
                            "conv allocate raw");
  ready = ready && test.cuda_ok(
                       weight.allocate(q3x::runtime::kGdnQkvChannels *
                                       q3x::runtime::kGdnConvKernelWidth),
                       "conv allocate weight");
  ready = ready && test.cuda_ok(
                       history.allocate(q3x::runtime::kGdnQkvChannels *
                                        q3x::runtime::kGdnConvHistoryWidth),
                       "conv allocate history");
  ready = ready && test.cuda_ok(output.allocate(q3x::runtime::kGdnQkvChannels),
                                "conv allocate output");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    const int centered = static_cast<int>((index * 5U) % 17U) - 8;
    weight[index] = encode_bf16(static_cast<float>(centered) / 32.0F);
  }
  std::fill_n(history.data(), history.size(), encode_bf16(0.0F));
  std::vector<std::uint16_t> cpu_raw(raw.size());
  std::vector<std::uint16_t> cpu_weight(weight.data(),
                                        weight.data() + weight.size());
  std::vector<std::uint16_t> cpu_history(history.size(), encode_bf16(0.0F));
  std::vector<std::uint16_t> cpu_output(output.size());

  for (std::size_t step = 0; step < 4U; ++step) {
    for (std::size_t channel = 0; channel < raw.size(); ++channel) {
      const int centered =
          static_cast<int>((channel * 7U + step * 3U) % 31U) - 15;
      raw[channel] = encode_bf16(static_cast<float>(centered) / 16.0F);
      cpu_raw[channel] = raw[channel];
    }
    (void)q3x::runtime::causal_conv1d_silu_update_reference_cpu(
        cpu_raw.data(), cpu_weight.data(), cpu_history.data(),
        cpu_output.data());
    ready = launch_after_stale(test, stream,
                               "causal conv step " + std::to_string(step),
                               [&]() {
      return q3x::runtime::launch_causal_conv1d_silu_update_reference_cuda(
          raw.data(), weight.data(), history.data(), output.data(), {},
          static_cast<void*>(stream));
    });
    if (!ready) {
      return;
    }
    expect_bf16_buffer_near(test, output.data(), cpu_output.data(),
                            output.size(), 2.0e-3F, 8.0e-3F,
                            "CUDA causal conv output step " +
                                std::to_string(step));
    test.expect(std::equal(history.data(), history.data() + history.size(),
                           cpu_history.begin()),
                "CUDA causal conv raw BF16 history step " +
                    std::to_string(step));
  }

  auto cpu_alias_history = cpu_history;
  auto cpu_alias_output = cpu_raw;
  (void)q3x::runtime::causal_conv1d_silu_update_reference_cpu(
      cpu_alias_output.data(), cpu_weight.data(), cpu_alias_history.data(),
      cpu_alias_output.data());
  ready = launch_after_stale(test, stream, "causal conv raw/output alias", [&]() {
    return q3x::runtime::launch_causal_conv1d_silu_update_reference_cuda(
        raw.data(), weight.data(), history.data(), raw.data(), {},
        static_cast<void*>(stream));
  });
  if (ready) {
    expect_bf16_buffer_near(test, raw.data(), cpu_alias_output.data(), raw.size(),
                            2.0e-3F, 8.0e-3F,
                            "CUDA aliased causal conv output");
  }
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::
                      launch_causal_conv1d_silu_update_reference_cuda(
                          raw.data(), weight.data(), history.data(),
                          history.data())) == cudaErrorInvalidValue,
              "CUDA conv rejects history/output alias");
}

void fill_gdn_inputs(ManagedBuffer<std::uint16_t>& conv_qkv,
                     ManagedBuffer<std::uint16_t>& a,
                     ManagedBuffer<std::uint16_t>& b,
                     ManagedBuffer<std::uint16_t>& A_log,
                     ManagedBuffer<std::uint16_t>& dt_bias,
                     const std::size_t step) {
  constexpr std::size_t kKOffset = q3x::runtime::kGdnQElements;
  constexpr std::size_t kVOffset =
      q3x::runtime::kGdnQElements + q3x::runtime::kGdnKElements;
  for (std::size_t head = 0; head < q3x::runtime::kGdnQkHeadCount; ++head) {
    for (std::size_t dimension = 0;
         dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
      const int q_centered =
          static_cast<int>((head * 11U + dimension * 3U + step) % 29U) - 14;
      const int k_centered =
          static_cast<int>((head * 7U + dimension * 5U + step * 2U) % 31U) -
          15;
      conv_qkv[head * q3x::runtime::kGdnHeadDimension + dimension] =
          encode_bf16(static_cast<float>(q_centered) / 16.0F);
      conv_qkv[kKOffset + head * q3x::runtime::kGdnHeadDimension + dimension] =
          encode_bf16(static_cast<float>(k_centered) / 16.0F);
    }
  }
  for (std::size_t index = 0; index < q3x::runtime::kGdnVElements; ++index) {
    const int centered =
        static_cast<int>((index * 13U + step * 7U) % 37U) - 18;
    conv_qkv[kVOffset + index] =
        encode_bf16(static_cast<float>(centered) / 16.0F);
  }
  for (std::size_t head = 0; head < q3x::runtime::kGdnValueHeadCount; ++head) {
    a[head] = encode_bf16(
        static_cast<float>(static_cast<int>(head % 9U) - 4) * 0.25F);
    b[head] = encode_bf16(
        static_cast<float>(static_cast<int>(head % 11U) - 5) * 0.5F);
    A_log[head] = encode_bf16(
        -1.5F + static_cast<float>(head % 5U) * 0.375F);
    dt_bias[head] = encode_bf16(-0.75F + static_cast<float>(step) * 0.125F);
  }
  b[0] = encode_bf16(20.0F);
  b[1] = encode_bf16(-20.0F);
  a[2] = encode_bf16(25.0F);
  A_log[2] = encode_bf16(4.0F);
}

void fill_gdn_tile_inputs(ManagedBuffer<std::uint16_t>& conv_qkv,
                          ManagedBuffer<std::uint16_t>& a,
                          ManagedBuffer<std::uint16_t>& b,
                          ManagedBuffer<std::uint16_t>& A_log,
                          ManagedBuffer<std::uint16_t>& dt_bias) {
  constexpr std::size_t kKOffset = q3x::runtime::kGdnQElements;
  constexpr std::size_t kVOffset =
      q3x::runtime::kGdnQElements + q3x::runtime::kGdnKElements;
  for (std::size_t token = 0U;
       token < q3x::runtime::kGdnMaximumTileTokenCount; ++token) {
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
            encode_bf16(static_cast<float>(q_centered) / 16.0F);
        conv_qkv[qkv_offset + kKOffset +
                 head * q3x::runtime::kGdnHeadDimension + dimension] =
            encode_bf16(static_cast<float>(k_centered) / 16.0F);
      }
    }
    for (std::size_t index = 0U; index < q3x::runtime::kGdnVElements;
         ++index) {
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
  for (std::size_t head = 0U; head < q3x::runtime::kGdnValueHeadCount;
       ++head) {
    A_log[head] = encode_bf16(
        -1.5F + static_cast<float>(head % 5U) * 0.375F);
    dt_bias[head] = encode_bf16(
        -0.75F + static_cast<float>(head % 7U) * 0.125F);
  }
  b[0] = encode_bf16(20.0F);
  b[q3x::runtime::kGdnValueHeadCount + 1U] = encode_bf16(-20.0F);
  a[2U * q3x::runtime::kGdnValueHeadCount + 2U] = encode_bf16(25.0F);
  A_log[2] = encode_bf16(4.0F);
}

void test_gdn_tile_bitwise(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kMaximumTokens =
      q3x::runtime::kGdnMaximumTileTokenCount;
  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> sequential_state;
  ManagedBuffer<std::uint16_t> tile_state;
  ManagedBuffer<std::uint16_t> warp_state;
  ManagedBuffer<std::uint16_t> row_pair_inplace_state;
  ManagedBuffer<std::uint16_t> row_pair_disjoint_input_state;
  ManagedBuffer<std::uint16_t> row_pair_disjoint_output_state;
  ManagedBuffer<std::uint16_t> sequential_output;
  ManagedBuffer<std::uint16_t> tile_output;
  ManagedBuffer<std::uint16_t> warp_output;
  ManagedBuffer<std::uint16_t> row_pair_inplace_output;
  ManagedBuffer<std::uint16_t> row_pair_disjoint_output;
  bool ready = test.cuda_ok(
      conv_qkv.allocate(kMaximumTokens * q3x::runtime::kGdnQkvChannels),
      "tile GDN allocate conv QKV");
  ready = ready && test.cuda_ok(
                       a.allocate(kMaximumTokens *
                                  q3x::runtime::kGdnValueHeadCount),
                       "tile GDN allocate a");
  ready = ready && test.cuda_ok(
                       b.allocate(kMaximumTokens *
                                  q3x::runtime::kGdnValueHeadCount),
                       "tile GDN allocate b");
  ready = ready && test.cuda_ok(A_log.allocate(
                                    q3x::runtime::kGdnValueHeadCount),
                                "tile GDN allocate A_log");
  ready = ready && test.cuda_ok(dt_bias.allocate(
                                    q3x::runtime::kGdnValueHeadCount),
                                "tile GDN allocate dt_bias");
  ready = ready && test.cuda_ok(
                       sequential_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "tile GDN allocate sequential state");
  ready = ready && test.cuda_ok(
                       tile_state.allocate(q3x::runtime::kGdnStateElements),
                       "tile GDN allocate tile state");
  ready = ready && test.cuda_ok(
                       warp_state.allocate(q3x::runtime::kGdnStateElements),
                       "tile GDN allocate warp state");
  ready = ready && test.cuda_ok(
                       row_pair_inplace_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "tile GDN allocate row-pair in-place state");
  ready = ready && test.cuda_ok(
                       row_pair_disjoint_input_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "tile GDN allocate row-pair disjoint input state");
  ready = ready && test.cuda_ok(
                       row_pair_disjoint_output_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "tile GDN allocate row-pair disjoint output state");
  ready = ready && test.cuda_ok(
                       sequential_output.allocate(
                           kMaximumTokens * q3x::runtime::kGdnVElements),
                       "tile GDN allocate sequential output");
  ready = ready && test.cuda_ok(
                       tile_output.allocate(
                           kMaximumTokens * q3x::runtime::kGdnVElements),
                       "tile GDN allocate tile output");
  ready = ready && test.cuda_ok(
                       warp_output.allocate(
                           kMaximumTokens * q3x::runtime::kGdnVElements),
                       "tile GDN allocate warp output");
  ready = ready && test.cuda_ok(
                       row_pair_inplace_output.allocate(
                           kMaximumTokens * q3x::runtime::kGdnVElements),
                       "tile GDN allocate row-pair in-place output");
  ready = ready && test.cuda_ok(
                       row_pair_disjoint_output.allocate(
                           kMaximumTokens * q3x::runtime::kGdnVElements),
                       "tile GDN allocate row-pair disjoint output");
  if (!ready) {
    return;
  }

  fill_gdn_tile_inputs(conv_qkv, a, b, A_log, dt_bias);
  std::vector<std::uint16_t> initial_state(
      q3x::runtime::kGdnStateElements);
  for (std::size_t index = 0U; index < initial_state.size(); ++index) {
    const int centered = static_cast<int>((index * 5U) % 23U) - 11;
    initial_state[index] =
        encode_bf16(static_cast<float>(centered) / 512.0F);
  }

  for (std::size_t token_count = 1U; token_count <= kMaximumTokens;
       ++token_count) {
    const bool test_row_pair =
        token_count == 1U || token_count == 2U ||
        token_count == kMaximumTokens;
    std::copy(initial_state.begin(), initial_state.end(),
              sequential_state.data());
    std::copy(initial_state.begin(), initial_state.end(), tile_state.data());
    std::copy(initial_state.begin(), initial_state.end(), warp_state.data());
    if (test_row_pair) {
      std::copy(initial_state.begin(), initial_state.end(),
                row_pair_inplace_state.data());
      std::copy(initial_state.begin(), initial_state.end(),
                row_pair_disjoint_input_state.data());
      std::fill_n(row_pair_disjoint_output_state.data(),
                  row_pair_disjoint_output_state.size(),
                  static_cast<std::uint16_t>(0x5a5aU));
    }
    std::fill_n(sequential_output.data(), sequential_output.size(),
                static_cast<std::uint16_t>(0U));
    std::fill_n(tile_output.data(), tile_output.size(),
                static_cast<std::uint16_t>(0U));
    std::fill_n(warp_output.data(), warp_output.size(),
                static_cast<std::uint16_t>(0U));
    if (test_row_pair) {
      std::fill_n(row_pair_inplace_output.data(),
                  row_pair_inplace_output.size(),
                  static_cast<std::uint16_t>(0U));
      std::fill_n(row_pair_disjoint_output.data(),
                  row_pair_disjoint_output.size(),
                  static_cast<std::uint16_t>(0U));
    }

    for (std::size_t token = 0U; token < token_count; ++token) {
      if (!test.cuda_ok(
              static_cast<cudaError_t>(
                  q3x::runtime::launch_gated_delta_net_update_reference_cuda(
                      conv_qkv.data() +
                          token * q3x::runtime::kGdnQkvChannels,
                      a.data() + token * q3x::runtime::kGdnValueHeadCount,
                      b.data() + token * q3x::runtime::kGdnValueHeadCount,
                      A_log.data(), dt_bias.data(), sequential_state.data(),
                      sequential_state.data(), 1.0e-6F,
                      sequential_output.data() +
                          token * q3x::runtime::kGdnVElements,
                      {}, static_cast<void*>(stream))),
              "tile GDN sequential launch M=" +
                  std::to_string(token_count) +
                  " token=" + std::to_string(token))) {
        return;
      }
    }
    ready = launch_after_stale(
        test, stream, "GDN tile M=" + std::to_string(token_count), [&]() {
          return q3x::runtime::
              launch_gated_delta_net_update_tile_reference_cuda(
                  conv_qkv.data(), token_count, a.data(), b.data(),
                  A_log.data(), dt_bias.data(), tile_state.data(),
                  tile_state.data(), 1.0e-6F, tile_output.data(), {},
                  static_cast<void*>(stream));
        });
    if (!ready) {
      return;
    }
    expect_bf16_buffer_bitwise_equal(
        test, tile_output.data(), sequential_output.data(),
        token_count * q3x::runtime::kGdnVElements,
        "CUDA GDN tile FP32-pre-store output M=" +
            std::to_string(token_count));
    expect_bf16_buffer_bitwise_equal(
        test, tile_state.data(), sequential_state.data(),
        q3x::runtime::kGdnStateElements,
        "CUDA GDN tile BF16 persistent state M=" +
            std::to_string(token_count));

    ready = launch_after_stale(
        test, stream,
        "GDN warp-parallel tile M=" + std::to_string(token_count), [&]() {
          return q3x::runtime::
              launch_gated_delta_net_update_tile_warp_parallel_cuda(
                  conv_qkv.data(), token_count, a.data(), b.data(),
                  A_log.data(), dt_bias.data(), warp_state.data(),
                  warp_state.data(), 1.0e-6F, warp_output.data(), {},
                  static_cast<void*>(stream));
        });
    if (!ready) {
      return;
    }
    expect_bf16_buffer_bitwise_equal(
        test, warp_output.data(), sequential_output.data(),
        token_count * q3x::runtime::kGdnVElements,
        "CUDA GDN warp-parallel FP32-pre-store output M=" +
            std::to_string(token_count));
    expect_bf16_buffer_bitwise_equal(
        test, warp_state.data(), sequential_state.data(),
        q3x::runtime::kGdnStateElements,
        "CUDA GDN warp-parallel BF16 persistent state M=" +
            std::to_string(token_count));

    if (test_row_pair) {
      std::copy(initial_state.begin(), initial_state.end(), warp_state.data());
      std::fill_n(warp_output.data(), warp_output.size(),
                  static_cast<std::uint16_t>(0U));
      ready = launch_after_stale(
          test, stream,
          "GDN warp baseline in-place M=" + std::to_string(token_count),
          [&]() {
            return q3x::runtime::
                launch_gated_delta_net_update_tile_warp_baseline_test_cuda(
                    conv_qkv.data(), token_count, a.data(), b.data(),
                    A_log.data(), dt_bias.data(), warp_state.data(),
                    warp_state.data(), 1.0e-6F, warp_output.data(),
                    static_cast<void*>(stream));
          });
      if (!ready) {
        return;
      }
      expect_bf16_buffer_bitwise_equal(
          test, warp_output.data(), sequential_output.data(),
          token_count * q3x::runtime::kGdnVElements,
          "CUDA GDN warp baseline in-place output M=" +
              std::to_string(token_count));
      expect_bf16_buffer_bitwise_equal(
          test, warp_state.data(), sequential_state.data(),
          q3x::runtime::kGdnStateElements,
          "CUDA GDN warp baseline in-place state M=" +
              std::to_string(token_count));

      ready = launch_after_stale(
          test, stream,
          "GDN warp baseline disjoint M=" + std::to_string(token_count),
          [&]() {
            return q3x::runtime::
                launch_gated_delta_net_update_tile_warp_baseline_test_cuda(
                    conv_qkv.data(), token_count, a.data(), b.data(),
                    A_log.data(), dt_bias.data(),
                    row_pair_disjoint_input_state.data(),
                    row_pair_disjoint_output_state.data(), 1.0e-6F,
                    row_pair_disjoint_output.data(),
                    static_cast<void*>(stream));
          });
      if (!ready) {
        return;
      }
      expect_bf16_buffer_bitwise_equal(
          test, row_pair_disjoint_output.data(), sequential_output.data(),
          token_count * q3x::runtime::kGdnVElements,
          "CUDA GDN warp baseline disjoint output M=" +
              std::to_string(token_count));
      expect_bf16_buffer_bitwise_equal(
          test, row_pair_disjoint_output_state.data(), sequential_state.data(),
          q3x::runtime::kGdnStateElements,
          "CUDA GDN warp baseline disjoint state M=" +
              std::to_string(token_count));
      expect_bf16_buffer_bitwise_equal(
          test, row_pair_disjoint_input_state.data(), initial_state.data(),
          q3x::runtime::kGdnStateElements,
          "CUDA GDN warp baseline preserves disjoint input M=" +
              std::to_string(token_count));
      std::fill_n(row_pair_disjoint_output_state.data(),
                  row_pair_disjoint_output_state.size(),
                  static_cast<std::uint16_t>(0x5a5aU));
      std::fill_n(row_pair_disjoint_output.data(),
                  row_pair_disjoint_output.size(),
                  static_cast<std::uint16_t>(0U));

      ready = launch_after_stale(
          test, stream,
          "GDN warp row-pair in-place M=" + std::to_string(token_count),
          [&]() {
            return q3x::runtime::
                launch_gated_delta_net_update_tile_warp_row_pair_test_cuda(
                    conv_qkv.data(), token_count, a.data(), b.data(),
                    A_log.data(), dt_bias.data(),
                    row_pair_inplace_state.data(),
                    row_pair_inplace_state.data(), 1.0e-6F,
                    row_pair_inplace_output.data(), static_cast<void*>(stream));
          });
      if (!ready) {
        return;
      }
      expect_bf16_buffer_bitwise_equal(
          test, row_pair_inplace_output.data(), sequential_output.data(),
          token_count * q3x::runtime::kGdnVElements,
          "CUDA GDN warp row-pair in-place output M=" +
              std::to_string(token_count));
      expect_bf16_buffer_bitwise_equal(
          test, row_pair_inplace_state.data(), sequential_state.data(),
          q3x::runtime::kGdnStateElements,
          "CUDA GDN warp row-pair in-place state M=" +
              std::to_string(token_count));

      ready = launch_after_stale(
          test, stream,
          "GDN warp row-pair disjoint M=" + std::to_string(token_count),
          [&]() {
            return q3x::runtime::
                launch_gated_delta_net_update_tile_warp_row_pair_test_cuda(
                    conv_qkv.data(), token_count, a.data(), b.data(),
                    A_log.data(), dt_bias.data(),
                    row_pair_disjoint_input_state.data(),
                    row_pair_disjoint_output_state.data(), 1.0e-6F,
                    row_pair_disjoint_output.data(),
                    static_cast<void*>(stream));
          });
      if (!ready) {
        return;
      }
      expect_bf16_buffer_bitwise_equal(
          test, row_pair_disjoint_output.data(), sequential_output.data(),
          token_count * q3x::runtime::kGdnVElements,
          "CUDA GDN warp row-pair disjoint output M=" +
              std::to_string(token_count));
      expect_bf16_buffer_bitwise_equal(
          test, row_pair_disjoint_output_state.data(), sequential_state.data(),
          q3x::runtime::kGdnStateElements,
          "CUDA GDN warp row-pair disjoint state M=" +
              std::to_string(token_count));
      expect_bf16_buffer_bitwise_equal(
          test, row_pair_disjoint_input_state.data(), initial_state.data(),
          q3x::runtime::kGdnStateElements,
          "CUDA GDN warp row-pair preserves disjoint input M=" +
              std::to_string(token_count));
    }
  }

  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_gated_delta_net_update_tile_reference_cuda(
              conv_qkv.data(), 2U, a.data(), b.data(), A_log.data(),
              dt_bias.data(), tile_state.data(), tile_state.data(), 1.0e-6F,
              conv_qkv.data())) == cudaErrorInvalidValue,
      "CUDA tile GDN rejects output/conv alias");

  constexpr std::size_t kWarmupCount = 5U;
  constexpr std::size_t kIterationCount = 50U;
  const float reference_m1_ms = measure_average_cuda_milliseconds(
      test, stream, kWarmupCount, kIterationCount, "GDN reference M1", [&]() {
        return q3x::runtime::launch_gated_delta_net_update_reference_cuda(
            conv_qkv.data(), a.data(), b.data(), A_log.data(), dt_bias.data(),
            sequential_state.data(), sequential_state.data(), 1.0e-6F,
            sequential_output.data(), {}, static_cast<void*>(stream));
      });
  const float warp_m1_ms = measure_average_cuda_milliseconds(
      test, stream, kWarmupCount, kIterationCount, "GDN warp M1", [&]() {
        return q3x::runtime::launch_gated_delta_net_update_warp_parallel_cuda(
            conv_qkv.data(), a.data(), b.data(), A_log.data(), dt_bias.data(),
            warp_state.data(), warp_state.data(), 1.0e-6F,
            warp_output.data(), {}, static_cast<void*>(stream));
      });
  const float reference_m8_ms = measure_average_cuda_milliseconds(
      test, stream, kWarmupCount, kIterationCount, "GDN reference M8", [&]() {
        return q3x::runtime::launch_gated_delta_net_update_tile_reference_cuda(
            conv_qkv.data(), kMaximumTokens, a.data(), b.data(), A_log.data(),
            dt_bias.data(), tile_state.data(), tile_state.data(), 1.0e-6F,
            tile_output.data(), {}, static_cast<void*>(stream));
      });
  const float warp_m8_ms = measure_average_cuda_milliseconds(
      test, stream, kWarmupCount, kIterationCount, "GDN warp M8", [&]() {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_parallel_cuda(
                conv_qkv.data(), kMaximumTokens, a.data(), b.data(),
                A_log.data(), dt_bias.data(), warp_state.data(),
                warp_state.data(), 1.0e-6F, warp_output.data(), {},
                static_cast<void*>(stream));
      });
  if (std::isfinite(reference_m1_ms) && std::isfinite(warp_m1_ms) &&
      std::isfinite(reference_m8_ms) && std::isfinite(warp_m8_ms)) {
    std::cout << "GDN CUDA-event benchmark: M1 reference="
              << reference_m1_ms << " ms, warp=" << warp_m1_ms
              << " ms, speedup=" << reference_m1_ms / warp_m1_ms
              << "x; M8 reference=" << reference_m8_ms
              << " ms, warp=" << warp_m8_ms
              << " ms, speedup=" << reference_m8_ms / warp_m8_ms << "x\n";
  }

  const char* const run_row_pair_perf =
      std::getenv("Q3X_RUN_GDN_ROW_PAIR_PERF");
  if (run_row_pair_perf != nullptr &&
      std::strcmp(run_row_pair_perf, "1") == 0) {
    constexpr std::size_t kPairWarmups = 5U;
    constexpr std::size_t kPairIterations = 50U;

    const auto launch_baseline = [&](const std::size_t token_count) {
      return q3x::runtime::
          launch_gated_delta_net_update_tile_warp_baseline_test_cuda(
              conv_qkv.data(), token_count, a.data(), b.data(), A_log.data(),
              dt_bias.data(), warp_state.data(), warp_state.data(), 1.0e-6F,
              warp_output.data(), static_cast<void*>(stream));
    };
    const auto launch_candidate = [&](const std::size_t token_count) {
      return q3x::runtime::
          launch_gated_delta_net_update_tile_warp_row_pair_test_cuda(
              conv_qkv.data(), token_count, a.data(), b.data(), A_log.data(),
              dt_bias.data(), row_pair_inplace_state.data(),
              row_pair_inplace_state.data(), 1.0e-6F,
              row_pair_inplace_output.data(), static_cast<void*>(stream));
    };
    const auto measure_mirrored =
        [&](const std::size_t token_count,
            const std::string& shape) -> std::array<float, 2> {
      std::copy(initial_state.begin(), initial_state.end(), warp_state.data());
      std::copy(initial_state.begin(), initial_state.end(),
                row_pair_inplace_state.data());
      const float baseline_first = measure_average_cuda_milliseconds(
          test, stream, kPairWarmups, kPairIterations,
          "GDN row-pair " + shape + " baseline first", [&]() {
            return launch_baseline(token_count);
          });
      const float candidate_first = measure_average_cuda_milliseconds(
          test, stream, kPairWarmups, kPairIterations,
          "GDN row-pair " + shape + " candidate first", [&]() {
            return launch_candidate(token_count);
          });
      const float candidate_second = measure_average_cuda_milliseconds(
          test, stream, kPairWarmups, kPairIterations,
          "GDN row-pair " + shape + " candidate second", [&]() {
            return launch_candidate(token_count);
          });
      const float baseline_second = measure_average_cuda_milliseconds(
          test, stream, kPairWarmups, kPairIterations,
          "GDN row-pair " + shape + " baseline second", [&]() {
            return launch_baseline(token_count);
          });
      return {(baseline_first + baseline_second) * 0.5F,
              (candidate_first + candidate_second) * 0.5F};
    };

    const std::array<float, 2> m1 = measure_mirrored(1U, "M1");
    const std::array<float, 2> m2 = measure_mirrored(2U, "M2");
    const std::array<float, 2> m8 =
        measure_mirrored(kMaximumTokens, "M8");
    const bool finite =
        std::isfinite(m1[0]) && std::isfinite(m1[1]) &&
        std::isfinite(m2[0]) && std::isfinite(m2[1]) &&
        std::isfinite(m8[0]) && std::isfinite(m8[1]);
    test.expect(finite, "GDN row-pair mirrored A/B timings are finite");
    if (finite) {
      const float m1_speedup = m1[0] / m1[1];
      const float m2_ratio = m2[1] / m2[0];
      const float m8_speedup = m8[0] / m8[1];
      std::cout << "GDN row-pair mirrored CUDA-event A/B: M1 B=" << m1[0]
                << " ms, C=" << m1[1] << " ms, speedup=" << m1_speedup
                << "x; M2 B=" << m2[0] << " ms, C=" << m2[1]
                << " ms, C/B=" << m2_ratio << "; M8 B=" << m8[0]
                << " ms, C=" << m8[1] << " ms, speedup=" << m8_speedup
                << "x\n";
      test.expect(m1_speedup >= 1.03F,
                  "GDN row-pair M1 reaches the 3% speedup gate");
      test.expect(m2_ratio <= 1.02F,
                  "GDN row-pair M2 does not regress by more than 2%");
      test.expect(m8_speedup >= 1.03F,
                  "GDN row-pair M8 reaches the 3% speedup gate");
    }
  }
}

void test_gdn_multistep(TestContext& test, cudaStream_t stream) {
  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> state;
  ManagedBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(conv_qkv.allocate(q3x::runtime::kGdnQkvChannels),
                            "GDN allocate conv QKV");
  ready = ready && test.cuda_ok(a.allocate(q3x::runtime::kGdnValueHeadCount),
                                "GDN allocate a");
  ready = ready && test.cuda_ok(b.allocate(q3x::runtime::kGdnValueHeadCount),
                                "GDN allocate b");
  ready = ready && test.cuda_ok(A_log.allocate(q3x::runtime::kGdnValueHeadCount),
                                "GDN allocate A_log");
  ready = ready && test.cuda_ok(
                       dt_bias.allocate(q3x::runtime::kGdnValueHeadCount),
                       "GDN allocate dt_bias");
  ready = ready && test.cuda_ok(state.allocate(q3x::runtime::kGdnStateElements),
                                "GDN allocate state");
  ready = ready && test.cuda_ok(output.allocate(q3x::runtime::kGdnVElements),
                                "GDN allocate output");
  if (!ready) {
    return;
  }
  std::vector<std::uint16_t> cpu_conv(conv_qkv.size());
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> cpu_a{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> cpu_b{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> cpu_A_log{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> cpu_dt_bias{};
  std::vector<std::uint16_t> cpu_state(state.size());
  std::vector<std::uint16_t> cpu_output(output.size());
  for (std::size_t index = 0; index < state.size(); ++index) {
    const int centered = static_cast<int>(index % 17U) - 8;
    state[index] = encode_bf16(static_cast<float>(centered) / 512.0F);
    cpu_state[index] = state[index];
  }

  for (std::size_t step = 0; step < 3U; ++step) {
    fill_gdn_inputs(conv_qkv, a, b, A_log, dt_bias, step);
    std::copy_n(conv_qkv.data(), conv_qkv.size(), cpu_conv.begin());
    std::copy_n(a.data(), a.size(), cpu_a.begin());
    std::copy_n(b.data(), b.size(), cpu_b.begin());
    std::copy_n(A_log.data(), A_log.size(), cpu_A_log.begin());
    std::copy_n(dt_bias.data(), dt_bias.size(), cpu_dt_bias.begin());
    (void)q3x::runtime::gated_delta_net_update_reference_cpu(
        cpu_conv.data(), cpu_a.data(), cpu_b.data(), cpu_A_log.data(),
        cpu_dt_bias.data(), cpu_state.data(), cpu_state.data(), 1.0e-6F,
        cpu_output.data());
    ready = launch_after_stale(test, stream,
                               "GDN update step " + std::to_string(step),
                               [&]() {
      return q3x::runtime::launch_gated_delta_net_update_reference_cuda(
          conv_qkv.data(), a.data(), b.data(), A_log.data(), dt_bias.data(),
          state.data(), state.data(), 1.0e-6F, output.data(), {},
          static_cast<void*>(stream));
    });
    if (!ready) {
      return;
    }
    expect_bf16_buffer_near(test, output.data(), cpu_output.data(), output.size(),
                            3.0e-3F, 1.2e-2F,
                            "CUDA GDN FP32-pre-store output step " +
                                std::to_string(step));
    expect_bf16_buffer_near(test, state.data(), cpu_state.data(), state.size(),
                            2.5e-3F, 1.2e-2F,
                            "CUDA GDN BF16 persistent state step " +
                                std::to_string(step));
  }

  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_gated_delta_net_update_reference_cuda(
                      conv_qkv.data(), a.data(), b.data(), A_log.data(),
                      dt_bias.data(), state.data(), state.data(), 1.0e-6F,
                      conv_qkv.data())) == cudaErrorInvalidValue,
              "CUDA GDN rejects output/conv alias");
}

}  // namespace

int main() {
  TestContext test;
  test_launch_validation(test);
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA GDN decode tests (no CUDA device)\n";
    (void)cudaGetLastError();
    return test.failures() == 0 ? 0 : 1;
  }
  cudaDeviceProp properties{};
  if (test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                   "read CUDA device properties")) {
    test.expect(properties.major == 8 && properties.minor == 7,
                "GDN reference runs on required SM87 device");
    std::cout << "CUDA GDN device: " << properties.name << " (sm_"
              << properties.major << properties.minor << ")\n";
  }
  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create GDN stream")) {
    return 1;
  }
  test_conv_multistep(test, stream);
  test_conv_tile_bitwise(test, stream);
  test_gdn_multistep(test, stream);
  test_gdn_tile_bitwise(test, stream);
  (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy GDN stream");
  if (test.failures() != 0) {
    std::cerr << test.failures() << " CUDA GDN assertion(s) failed\n";
    return 1;
  }
  std::cout << "CUDA GDN reference tests passed\n";
  return 0;
}
