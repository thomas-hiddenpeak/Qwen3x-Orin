#include "q3x/runtime/model_weights.h"

#include "q3x/kernels/reference_gemv.h"
#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;
namespace runtime = q3x::runtime;

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
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] bool allocate(TestContext& test, const std::size_t count,
                              const std::string& label) {
    return test.cuda_ok(
        cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
        "allocate " + label);
  }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U &&
      (bits & 0x007fffffU) != 0U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

void seed_stale_error(TestContext& test) {
  const cudaError_t status =
      cudaMemcpy(nullptr, nullptr, 1U, cudaMemcpyHostToDevice);
  test.expect(status == cudaErrorInvalidValue,
              "invalid copy seeds a stale CUDA error");
}

void compare(TestContext& test, const std::vector<float>& actual,
             const std::vector<float>& expected, const std::string& label) {
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    const float tolerance = 1.0e-5F + 1.0e-5F * std::fabs(expected[index]);
    test.expect(std::fabs(actual[index] - expected[index]) <= tolerance,
                label + " output " + std::to_string(index));
  }
}

void test_bf16_dispatch(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kRows = 3U;
  constexpr std::size_t kColumns = 16U;
  std::vector<std::uint16_t> activation(kColumns);
  std::vector<std::uint16_t> weight(kRows * kColumns);
  for (std::size_t index = 0U; index < activation.size(); ++index) {
    activation[index] =
        encode_bf16(static_cast<float>(static_cast<int>(index % 7U) - 3) /
                    8.0F);
  }
  for (std::size_t index = 0U; index < weight.size(); ++index) {
    weight[index] =
        encode_bf16(static_cast<float>(static_cast<int>(index % 9U) - 4) /
                    16.0F);
  }
  std::vector<float> expected(kRows);
  (void)kernels::bf16_gemv_reference_cpu(
      weight.data(), activation.data(), kRows, kColumns, expected.data());

  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_weight;
  DeviceBuffer<float> device_output;
  DeviceBuffer<std::uint16_t> device_bf16_output;
  bool ready = device_activation.allocate(test, kColumns, "BF16 activation");
  ready = ready && device_weight.allocate(test, weight.size(), "BF16 weight");
  ready = ready && device_output.allocate(test, kRows, "BF16 FP32 output");
  ready = ready &&
          device_bf16_output.allocate(test, kRows, "BF16 converted output");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      cudaMemcpyAsync(device_activation.get(), activation.data(),
                      activation.size() * sizeof(activation[0]),
                      cudaMemcpyHostToDevice, stream),
      "copy BF16 activation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_weight.get(), weight.data(),
                                       weight.size() * sizeof(weight[0]),
                                       cudaMemcpyHostToDevice, stream),
                       "copy BF16 weight");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight linear = runtime::Bf16LinearWeight{
      device_weight.get(), kRows, kColumns};
  seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(runtime::launch_projection_reference_cuda(
          linear, device_activation.get(), device_output.get(),
          static_cast<void*>(stream))),
      "dispatch BF16 after stale error");
  std::vector<float> actual(kRows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(actual.data(), device_output.get(),
                                       actual.size() * sizeof(actual[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy BF16 FP32 result");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), "sync BF16 dispatch");
  if (ready) {
    compare(test, actual, expected, "BF16 dispatch");
  }

  ready = test.cuda_ok(
      static_cast<cudaError_t>(runtime::launch_projection_to_bf16_reference_cuda(
          linear, device_activation.get(), device_output.get(), kRows,
          device_bf16_output.get(), static_cast<void*>(stream))),
      "dispatch BF16 and convert");
  std::vector<std::uint16_t> converted(kRows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(converted.data(),
                                       device_bf16_output.get(),
                                       converted.size() * sizeof(converted[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy converted BF16 result");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "sync converted BF16 result");
  if (ready) {
    for (std::size_t index = 0U; index < kRows; ++index) {
      test.expect(converted[index] == encode_bf16(expected[index]),
                  "projection-to-BF16 uses RNE conversion " +
                      std::to_string(index));
    }
  }
  test.expect(static_cast<cudaError_t>(
                  runtime::launch_projection_to_bf16_reference_cuda(
                      linear, device_activation.get(), device_output.get(),
                      kRows - 1U, device_bf16_output.get(),
                      static_cast<void*>(stream))) == cudaErrorInvalidValue,
              "projection-to-BF16 rejects insufficient caller scratch");
}

void test_fp8_dispatch(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kRows = 4U;
  constexpr std::size_t kColumns = 16U;
  constexpr float kWeightScale = 0.25F;
  constexpr float kInputScale = 0.5F;
  constexpr std::uint8_t kCodes[] = {0x00U, 0x30U, 0x38U, 0xb8U,
                                     0x40U, 0xc0U, 0x3cU, 0xbcU};
  std::vector<std::uint16_t> activation(kColumns);
  std::vector<std::uint8_t> weight(kRows * kColumns);
  for (std::size_t index = 0U; index < activation.size(); ++index) {
    activation[index] =
        encode_bf16(static_cast<float>(static_cast<int>(index % 5U) - 2) /
                    4.0F);
  }
  for (std::size_t index = 0U; index < weight.size(); ++index) {
    weight[index] = kCodes[index % (sizeof(kCodes) / sizeof(kCodes[0]))];
  }
  std::vector<float> expected(kRows);
  (void)kernels::fp8_gemv_reference_cpu(
      weight.data(), kWeightScale, activation.data(), kRows, kColumns,
      expected.data());

  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint8_t> device_weight;
  DeviceBuffer<float> device_output;
  DeviceBuffer<float> device_scales;
  bool ready = device_activation.allocate(test, kColumns, "FP8 activation");
  ready = ready && device_weight.allocate(test, weight.size(), "FP8 weight");
  ready = ready && device_output.allocate(test, kRows, "FP8 output");
  ready = ready && device_scales.allocate(test, 2U, "FP8 companions");
  if (!ready) {
    return;
  }
  const float scales[2] = {kWeightScale, kInputScale};
  ready = test.cuda_ok(
      cudaMemcpyAsync(device_activation.get(), activation.data(),
                      activation.size() * sizeof(activation[0]),
                      cudaMemcpyHostToDevice, stream),
      "copy FP8 activation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_weight.get(), weight.data(),
                                       weight.size(), cudaMemcpyHostToDevice,
                                       stream),
                       "copy FP8 weight");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_scales.get(), scales,
                                       sizeof(scales), cudaMemcpyHostToDevice,
                                       stream),
                       "copy FP8 companions");
  if (!ready) {
    return;
  }
  const runtime::LinearWeight linear = runtime::Fp8LinearWeight{
      device_weight.get(), device_scales.get(), device_scales.get() + 1U,
      kWeightScale, kInputScale, kRows, kColumns};
  seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(runtime::launch_projection_reference_cuda(
          linear, device_activation.get(), device_output.get(),
          static_cast<void*>(stream))),
      "dispatch FP8 after stale error");
  std::vector<float> actual(kRows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(actual.data(), device_output.get(),
                                       actual.size() * sizeof(actual[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy FP8 result");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), "sync FP8 dispatch");
  if (ready) {
    compare(test, actual, expected, "FP8 dispatch");
  }
  runtime::LinearWeight invalid = std::get<runtime::Fp8LinearWeight>(linear);
  std::get<runtime::Fp8LinearWeight>(invalid).input_scale =
      std::numeric_limits<float>::quiet_NaN();
  test.expect(static_cast<cudaError_t>(
                  runtime::launch_projection_reference_cuda(
                      invalid, device_activation.get(), device_output.get(),
                      static_cast<void*>(stream))) == cudaErrorInvalidValue,
              "dispatch rejects invalid retained FP8 input_scale");
}

void test_nvfp4_dispatch(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kRows = 3U;
  constexpr std::size_t kColumns = 32U;
  constexpr float kWeightScale2 = 0.125F;
  constexpr float kInputScale = 0.5F;
  std::vector<std::uint16_t> activation(kColumns);
  std::vector<std::uint8_t> packed(kRows * kColumns / 2U);
  std::vector<std::uint8_t> block_scale(kRows * kColumns / 16U, 0x38U);
  for (std::size_t index = 0U; index < activation.size(); ++index) {
    activation[index] =
        encode_bf16(static_cast<float>(static_cast<int>(index % 7U) - 3) /
                    8.0F);
  }
  for (std::size_t index = 0U; index < packed.size(); ++index) {
    const std::uint8_t low = static_cast<std::uint8_t>(index % 8U);
    const std::uint8_t high = static_cast<std::uint8_t>((index + 3U) % 8U);
    packed[index] = static_cast<std::uint8_t>(low | (high << 4U));
  }
  std::vector<float> expected(kRows);
  (void)kernels::nvfp4_gemv_reference_cpu(
      packed.data(), block_scale.data(), kWeightScale2, activation.data(),
      kRows, kColumns, expected.data());

  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint8_t> device_block_scale;
  DeviceBuffer<float> device_output;
  DeviceBuffer<float> device_scales;
  bool ready = device_activation.allocate(test, kColumns, "NVFP4 activation");
  ready = ready && device_packed.allocate(test, packed.size(), "NVFP4 weight");
  ready = ready && device_block_scale.allocate(
                       test, block_scale.size(), "NVFP4 block scale");
  ready = ready && device_output.allocate(test, kRows, "NVFP4 output");
  ready = ready && device_scales.allocate(test, 2U, "NVFP4 companions");
  if (!ready) {
    return;
  }
  const float scales[2] = {kWeightScale2, kInputScale};
  ready = test.cuda_ok(
      cudaMemcpyAsync(device_activation.get(), activation.data(),
                      activation.size() * sizeof(activation[0]),
                      cudaMemcpyHostToDevice, stream),
      "copy NVFP4 activation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_packed.get(), packed.data(),
                                       packed.size(), cudaMemcpyHostToDevice,
                                       stream),
                       "copy NVFP4 packed weight");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_block_scale.get(),
                                       block_scale.data(), block_scale.size(),
                                       cudaMemcpyHostToDevice, stream),
                       "copy NVFP4 block scale");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_scales.get(), scales,
                                       sizeof(scales), cudaMemcpyHostToDevice,
                                       stream),
                       "copy NVFP4 companions");
  if (!ready) {
    return;
  }
  const runtime::LinearWeight linear = runtime::NvFp4LinearWeight{
      device_packed.get(), device_block_scale.get(), device_scales.get(),
      device_scales.get() + 1U, kWeightScale2, kInputScale, kRows, kColumns};
  seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(runtime::launch_projection_reference_cuda(
          linear, device_activation.get(), device_output.get(),
          static_cast<void*>(stream))),
      "dispatch NVFP4 after stale error");
  std::vector<float> actual(kRows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(actual.data(), device_output.get(),
                                       actual.size() * sizeof(actual[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy NVFP4 result");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "sync NVFP4 dispatch");
  if (ready) {
    compare(test, actual, expected, "NVFP4 dispatch");
  }
}

}  // namespace

int main() {
  TestContext test;
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: CUDA device unavailable\n";
    return 77;
  }
  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreate(&stream), "create test stream")) {
    return 1;
  }
  test_bf16_dispatch(test, stream);
  test_fp8_dispatch(test, stream);
  test_nvfp4_dispatch(test, stream);
  (void)cudaStreamDestroy(stream);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " projection dispatch test(s) failed\n";
    return 1;
  }
  std::cout << "projection dispatch CUDA tests passed\n";
  return 0;
}
