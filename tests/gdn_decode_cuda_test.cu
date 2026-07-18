#include "q3x/runtime/gdn_decode.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

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
  test_gdn_multistep(test, stream);
  (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy GDN stream");
  if (test.failures() != 0) {
    std::cerr << test.failures() << " CUDA GDN assertion(s) failed\n";
    return 1;
  }
  std::cout << "CUDA GDN reference tests passed\n";
  return 0;
}
