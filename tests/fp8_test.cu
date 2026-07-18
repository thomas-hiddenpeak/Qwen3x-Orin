#include "q3x/quantization/fp8.h"

#include "q3x/quantization/nvfp4.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

constexpr std::array<std::uint8_t, 12> kWeights = {
    0x00U, 0x80U, 0x01U, 0x08U, 0x30U, 0x38U,
    0x3cU, 0x40U, 0x70U, 0x78U, 0x7eU, 0xfeU,
};

void test_cpu_reference(TestContext& test) {
  constexpr float kScale = 0.25F;
  std::array<float, kWeights.size()> output{};
  const auto status = q3x::quantization::dequantize_fp8_reference(
      kWeights.data(), kScale, 3, 4, output.data());
  test.expect(status == q3x::quantization::Fp8Status::kSuccess,
              "CPU FP8 dequantization succeeds");

  for (std::size_t index = 0; index < output.size(); ++index) {
    const float expected =
        q3x::quantization::decode_e4m3fn(kWeights[index]) * kScale;
    test.expect(output[index] == expected,
                "CPU FP8 value " + std::to_string(index));
    if (expected == 0.0F) {
      test.expect(std::signbit(output[index]) == std::signbit(expected),
                  "CPU FP8 zero sign " + std::to_string(index));
    }
  }
}

void test_validation(TestContext& test) {
  float output = 0.0F;
  const std::uint8_t input = 0x38U;
  using q3x::quantization::Fp8Status;
  using q3x::quantization::dequantize_fp8_reference;

  test.expect(dequantize_fp8_reference(nullptr, 1.0F, 0, 8, nullptr) ==
                  Fp8Status::kSuccess,
              "zero-row matrix accepts null storage");
  test.expect(dequantize_fp8_reference(nullptr, 1.0F, 1, 1, &output) ==
                  Fp8Status::kInvalidArgument,
              "null weight storage is rejected");
  test.expect(dequantize_fp8_reference(&input, 1.0F, 1, 1, nullptr) ==
                  Fp8Status::kInvalidArgument,
              "null output storage is rejected");
  test.expect(dequantize_fp8_reference(&input, -1.0F, 1, 1, &output) ==
                  Fp8Status::kInvalidArgument,
              "negative scale is rejected");
  test.expect(dequantize_fp8_reference(
                  &input, std::numeric_limits<float>::infinity(), 1, 1,
                  &output) == Fp8Status::kInvalidArgument,
              "infinite scale is rejected");
  test.expect(dequantize_fp8_reference(
                  &input, 1.0F, std::numeric_limits<std::size_t>::max(), 2,
                  &output) == Fp8Status::kSizeOverflow,
              "matrix size overflow is rejected");
  test.expect(std::string(q3x::quantization::fp8_status_string(
                  Fp8Status::kSizeOverflow)) ==
                  "matrix element count overflows size_t",
              "status text is stable");
}

void test_cuda_consistency(TestContext& test) {
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA FP8 consistency test (no CUDA device)\n";
    (void)cudaGetLastError();
    return;
  }

  constexpr float kScale = 0.125F;
  std::array<float, kWeights.size()> cpu{};
  std::array<float, kWeights.size()> gpu{};
  (void)q3x::quantization::dequantize_fp8_reference(
      kWeights.data(), kScale, 3, 4, cpu.data());

  std::uint8_t* device_weights = nullptr;
  float* device_output = nullptr;
  auto check_cuda = [&test](const cudaError_t status, const char* operation) {
    test.expect(status == cudaSuccess,
                std::string(operation) + ": " + cudaGetErrorString(status));
    return status == cudaSuccess;
  };

  bool ready = check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&device_weights), kWeights.size()),
      "cudaMalloc weights");
  ready = ready && check_cuda(
                       cudaMalloc(reinterpret_cast<void**>(&device_output),
                                  gpu.size() * sizeof(float)),
                       "cudaMalloc output");
  ready = ready && check_cuda(
                       cudaMemcpy(device_weights, kWeights.data(),
                                  kWeights.size(), cudaMemcpyHostToDevice),
                       "copy weights");
  if (ready) {
    const cudaError_t stale_status =
        cudaMemcpy(nullptr, nullptr, 1, cudaMemcpyHostToDevice);
    test.expect(stale_status == cudaErrorInvalidValue,
                "invalid CUDA call seeds the stale-error regression");
    const int launch_status = q3x::quantization::launch_fp8_reference_cuda(
        device_weights, kScale, 3, 4, device_output);
    ready = check_cuda(static_cast<cudaError_t>(launch_status),
                       "launch FP8 reference ignores an unrelated stale error");
  }
  if (ready) {
    ready = check_cuda(cudaDeviceSynchronize(), "synchronize FP8 reference");
  }
  if (ready) {
    ready = check_cuda(cudaMemcpy(gpu.data(), device_output,
                                  gpu.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "copy FP8 output");
  }
  if (ready) {
    for (std::size_t index = 0; index < gpu.size(); ++index) {
      test.expect(gpu[index] == cpu[index],
                  "CUDA FP8 value " + std::to_string(index));
      if (cpu[index] == 0.0F) {
        test.expect(std::signbit(gpu[index]) == std::signbit(cpu[index]),
                    "CUDA FP8 zero sign " + std::to_string(index));
      }
    }
  }

  if (device_output != nullptr) {
    (void)cudaFree(device_output);
  }
  if (device_weights != nullptr) {
    (void)cudaFree(device_weights);
  }
}

}  // namespace

int main() {
  TestContext test;
  test_cpu_reference(test);
  test_validation(test);
  test_cuda_consistency(test);

  if (test.failures() != 0) {
    std::cerr << test.failures() << " FP8 test assertion(s) failed\n";
    return 1;
  }
  std::cout << "FP8 reference tests passed\n";
  return 0;
}
