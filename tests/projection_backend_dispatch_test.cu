#include "q3x/runtime/model_weights.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

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

template <typename T>
[[nodiscard]] bool upload(TestContext& test, DeviceBuffer<T>& destination,
                          const std::vector<T>& source,
                          const std::string& label) {
  return test.cuda_ok(
      cudaMemcpy(destination.get(), source.data(),
                 source.size() * sizeof(T), cudaMemcpyHostToDevice),
      "upload " + label);
}

[[nodiscard]] bool expect_output(TestContext& test,
                                 const DeviceBuffer<std::uint16_t>& output,
                                 const std::uint16_t expected,
                                 const std::string& label) {
  std::uint16_t actual[2]{};
  bool ready = test.cuda_ok(cudaDeviceSynchronize(), "synchronize " + label);
  ready = ready && test.cuda_ok(
                       cudaMemcpy(actual, output.get(), sizeof(actual),
                                  cudaMemcpyDeviceToHost),
                       "download " + label);
  if (ready) {
    test.expect(actual[0] == expected && actual[1] == expected,
                label + " writes expected BF16 rows");
  }
  return ready;
}

void test_routes(TestContext& test) {
  constexpr std::size_t kRows = 2U;
  constexpr std::size_t kFp8Columns = 32U;
  constexpr std::size_t kNvFp4Columns = 16U;
  constexpr std::size_t kBf16Columns = 4U;
  constexpr std::uint16_t kBf16One = 0x3f80U;

  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> bf16_weight;
  DeviceBuffer<std::uint8_t> fp8_weight;
  DeviceBuffer<std::uint8_t> nvfp4_weight;
  DeviceBuffer<std::uint8_t> nvfp4_scale;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<float> scratch;
  DeviceBuffer<std::uint16_t> output;
  bool ready = activation.allocate(test, kFp8Columns, "activation");
  ready = ready && bf16_weight.allocate(
                       test, kRows * kBf16Columns, "BF16 weight");
  ready = ready && fp8_weight.allocate(
                       test, kRows * kFp8Columns, "FP8 weight");
  ready = ready && nvfp4_weight.allocate(
                       test, kRows * (kNvFp4Columns / 2U), "NVFP4 weight");
  ready = ready && nvfp4_scale.allocate(
                       test, kRows * (kNvFp4Columns / 16U), "NVFP4 scale");
  ready = ready && companion_scales.allocate(test, 4U, "companion scales");
  ready = ready && scratch.allocate(test, kRows, "reference scratch");
  ready = ready && output.allocate(test, kRows, "output");
  if (!ready) {
    return;
  }

  ready = upload(test, activation,
                 std::vector<std::uint16_t>(kFp8Columns, kBf16One),
                 "activation");
  ready = ready && upload(
                       test, bf16_weight,
                       std::vector<std::uint16_t>(
                           kRows * kBf16Columns, kBf16One),
                       "BF16 weight");
  ready = ready && upload(
                       test, fp8_weight,
                       std::vector<std::uint8_t>(
                           kRows * kFp8Columns, 0x38U),
                       "FP8 weight");
  ready = ready && upload(
                       test, nvfp4_weight,
                       std::vector<std::uint8_t>(
                           kRows * (kNvFp4Columns / 2U), 0x22U),
                       "NVFP4 weight");
  ready = ready && upload(
                       test, nvfp4_scale,
                       std::vector<std::uint8_t>(
                           kRows * (kNvFp4Columns / 16U), 0x38U),
                       "NVFP4 scale");
  ready = ready && upload(test, companion_scales,
                          std::vector<float>(4U, 1.0F),
                          "companion scales");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight fp8 = runtime::Fp8LinearWeight{
      fp8_weight.get(), companion_scales.get(), companion_scales.get() + 1,
      1.0F, 1.0F, kRows, kFp8Columns};
  const runtime::LinearWeight nvfp4 = runtime::NvFp4LinearWeight{
      nvfp4_weight.get(), nvfp4_scale.get(), companion_scales.get() + 2,
      companion_scales.get() + 3, 1.0F, 1.0F, kRows, kNvFp4Columns};
  const runtime::LinearWeight bf16 = runtime::Bf16LinearWeight{
      bf16_weight.get(), kRows, kBf16Columns};

  test.expect(static_cast<cudaError_t>(runtime::launch_projection_to_bf16_cuda(
                  runtime::ProjectionBackend::kSm87WeightOnly, fp8,
                  activation.get(), nullptr, 0U, output.get())) == cudaSuccess,
              "SM87 FP8 route does not require reference scratch");
  (void)expect_output(test, output, 0x4200U, "SM87 FP8 route");

  test.expect(static_cast<cudaError_t>(runtime::launch_projection_to_bf16_cuda(
                  runtime::ProjectionBackend::kReference, fp8,
                  activation.get(), nullptr, 0U, output.get())) ==
                  cudaErrorInvalidValue,
              "reference FP8 route still requires FP32 scratch");

  test.expect(static_cast<cudaError_t>(runtime::launch_projection_to_bf16_cuda(
                  runtime::ProjectionBackend::kSm87WeightOnly, nvfp4,
                  activation.get(), nullptr, 0U, output.get())) == cudaSuccess,
              "SM87 NVFP4 route does not require reference scratch");
  (void)expect_output(test, output, 0x4180U, "SM87 NVFP4 route");

  test.expect(static_cast<cudaError_t>(runtime::launch_projection_to_bf16_cuda(
                  runtime::ProjectionBackend::kSm87WeightOnly, bf16,
                  activation.get(), nullptr, 0U, output.get())) ==
                  cudaErrorInvalidValue,
              "SM87 policy routes BF16 through the reference fallback");
  test.expect(static_cast<cudaError_t>(runtime::launch_projection_to_bf16_cuda(
                  runtime::ProjectionBackend::kSm87WeightOnly, bf16,
                  activation.get(), scratch.get(), kRows, output.get())) ==
                  cudaSuccess,
              "SM87 BF16 fallback succeeds with reference scratch");
  (void)expect_output(test, output, 0x4080U, "SM87 BF16 fallback");

  const auto unknown = static_cast<runtime::ProjectionBackend>(0xffU);
  test.expect(!runtime::is_valid_projection_backend(unknown) &&
                  runtime::to_string(unknown) == "unknown",
              "unknown backend is neither valid nor stringified as active");
  test.expect(static_cast<cudaError_t>(runtime::launch_projection_to_bf16_cuda(
                  unknown, fp8, activation.get(), nullptr, 0U,
                  output.get())) == cudaErrorInvalidValue,
              "dispatcher rejects an unknown backend");

  const runtime::LinearWeight malformed_fp8 = runtime::Fp8LinearWeight{
      fp8_weight.get(), nullptr, companion_scales.get() + 1, 1.0F, 1.0F,
      kRows, kFp8Columns};
  const runtime::LinearWeight malformed_nvfp4 = runtime::NvFp4LinearWeight{
      nvfp4_weight.get(), nullptr, companion_scales.get() + 2,
      companion_scales.get() + 3, 1.0F, 1.0F, kRows, kNvFp4Columns};
  test.expect(static_cast<cudaError_t>(runtime::launch_projection_to_bf16_cuda(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  malformed_fp8, activation.get(), nullptr, 0U,
                  output.get())) == cudaErrorInvalidValue,
              "SM87 FP8 route validates its active variant");
  test.expect(static_cast<cudaError_t>(runtime::launch_projection_to_bf16_cuda(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  malformed_nvfp4, activation.get(), nullptr, 0U,
                  output.get())) == cudaErrorInvalidValue,
              "SM87 NVFP4 route validates its active variant");
}

}  // namespace

int main() {
  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA device unavailable\n";
    return 77;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, 0);
  if (status != cudaSuccess || properties.major != 8 ||
      properties.minor != 7) {
    std::cout << "SKIP: projection backend test requires SM87\n";
    return 77;
  }

  TestContext test;
  test_routes(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " projection dispatch assertion(s) failed\n";
    return 1;
  }
  std::cout << "projection backend dispatch tests passed\n";
  return 0;
}
