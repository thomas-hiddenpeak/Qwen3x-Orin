#include "q3x/runtime/model_weights.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
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

[[nodiscard]] bool expect_tile_output(
    TestContext& test, const DeviceBuffer<std::uint16_t>& output,
    const std::size_t token_count, const std::size_t rows,
    const std::vector<std::uint16_t>& expected_by_token,
    const std::string& label) {
  std::vector<std::uint16_t> actual(token_count * rows);
  bool ready = test.cuda_ok(cudaDeviceSynchronize(), "synchronize " + label);
  ready = ready && test.cuda_ok(
                       cudaMemcpy(actual.data(), output.get(),
                                  actual.size() * sizeof(actual.front()),
                                  cudaMemcpyDeviceToHost),
                       "download " + label);
  if (!ready) {
    return false;
  }
  for (std::size_t token = 0U; token < token_count; ++token) {
    for (std::size_t row = 0U; row < rows; ++row) {
      test.expect(actual[token * rows + row] == expected_by_token[token],
                  label + " token " + std::to_string(token) + " row " +
                      std::to_string(row));
    }
  }
  return true;
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

void test_tile_routes(TestContext& test) {
  constexpr std::size_t kRows = 2U;
  constexpr std::size_t kFp8Columns = 1024U;
  constexpr std::size_t kNvFp4Columns = 256U;
  constexpr std::size_t kBf16Columns = 4U;
  constexpr std::size_t kMaximumTokens = 16U;
  constexpr std::uint16_t kBf16One = 0x3f80U;
  constexpr std::array<std::uint16_t, kMaximumTokens> kActivationValues{
      0x3f80U, 0x3f00U, 0xbf80U, 0x4000U,
      0x3e80U, 0xbf00U, 0x4080U, 0xc000U,
      0x3f40U, 0xbf40U, 0x4040U, 0xc040U,
      0x3fc0U, 0xbfc0U, 0x4100U, 0xc100U};
  constexpr std::array<std::uint16_t, kMaximumTokens> kFp8Expected{
      0x4480U, 0x4400U, 0xc480U, 0x4500U,
      0x4380U, 0xc400U, 0x4580U, 0xc500U,
      0x4440U, 0xc440U, 0x4540U, 0xc540U,
      0x44c0U, 0xc4c0U, 0x4600U, 0xc600U};
  constexpr std::array<std::uint16_t, kMaximumTokens> kNvFp4Expected{
      0x4380U, 0x4300U, 0xc380U, 0x4400U,
      0x4280U, 0xc300U, 0x4480U, 0xc400U,
      0x4340U, 0xc340U, 0x4440U, 0xc440U,
      0x43c0U, 0xc3c0U, 0x4500U, 0xc500U};
  constexpr std::array<std::uint16_t, kMaximumTokens> kBf16Expected{
      0x4080U, 0x4000U, 0xc080U, 0x4100U,
      0x3f80U, 0xc000U, 0x4180U, 0xc100U,
      0x4040U, 0xc040U, 0x4140U, 0xc140U,
      0x40c0U, 0xc0c0U, 0x4200U, 0xc200U};

  DeviceBuffer<std::uint16_t> fp8_activation;
  DeviceBuffer<std::uint16_t> nvfp4_activation;
  DeviceBuffer<std::uint16_t> bf16_activation;
  DeviceBuffer<std::uint16_t> bf16_weight;
  DeviceBuffer<std::uint8_t> fp8_weight;
  DeviceBuffer<std::uint8_t> nvfp4_weight;
  DeviceBuffer<std::uint8_t> nvfp4_scale;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<float> scratch;
  DeviceBuffer<std::uint16_t> output;
  bool ready = fp8_activation.allocate(
      test, kMaximumTokens * kFp8Columns, "tile FP8 activation");
  ready = ready && nvfp4_activation.allocate(
                       test, kMaximumTokens * kNvFp4Columns,
                       "tile NVFP4 activation");
  ready = ready && bf16_activation.allocate(
                       test, kMaximumTokens * kBf16Columns,
                       "tile BF16 activation");
  ready = ready && bf16_weight.allocate(
                       test, kRows * kBf16Columns, "tile BF16 weight");
  ready = ready && fp8_weight.allocate(
                       test, kRows * kFp8Columns, "tile FP8 weight");
  ready = ready && nvfp4_weight.allocate(
                       test, kRows * (kNvFp4Columns / 2U),
                       "tile NVFP4 weight");
  ready = ready && nvfp4_scale.allocate(
                       test, kRows * (kNvFp4Columns / 16U),
                       "tile NVFP4 scale");
  ready = ready && companion_scales.allocate(
                       test, 4U, "tile companion scales");
  ready = ready && scratch.allocate(test, kRows, "tile scratch");
  ready = ready && output.allocate(
                       test, kMaximumTokens * kRows, "tile output");
  if (!ready) {
    return;
  }

  std::vector<std::uint16_t> fp8_input(kMaximumTokens * kFp8Columns);
  std::vector<std::uint16_t> nvfp4_input(kMaximumTokens * kNvFp4Columns);
  std::vector<std::uint16_t> bf16_input(kMaximumTokens * kBf16Columns);
  for (std::size_t token = 0U; token < kMaximumTokens; ++token) {
    std::fill_n(fp8_input.begin() + token * kFp8Columns, kFp8Columns,
                kActivationValues[token]);
    std::fill_n(nvfp4_input.begin() + token * kNvFp4Columns,
                kNvFp4Columns, kActivationValues[token]);
    std::fill_n(bf16_input.begin() + token * kBf16Columns, kBf16Columns,
                kActivationValues[token]);
  }
  ready = upload(test, fp8_activation, fp8_input, "tile FP8 activation");
  ready = ready && upload(test, nvfp4_activation, nvfp4_input,
                          "tile NVFP4 activation");
  ready = ready && upload(test, bf16_activation, bf16_input,
                          "tile BF16 activation");
  ready = ready && upload(
                       test, bf16_weight,
                       std::vector<std::uint16_t>(
                           kRows * kBf16Columns, kBf16One),
                       "tile BF16 weight");
  ready = ready && upload(
                       test, fp8_weight,
                       std::vector<std::uint8_t>(
                           kRows * kFp8Columns, 0x38U),
                       "tile FP8 weight");
  ready = ready && upload(
                       test, nvfp4_weight,
                       std::vector<std::uint8_t>(
                           kRows * (kNvFp4Columns / 2U), 0x22U),
                       "tile NVFP4 weight");
  ready = ready && upload(
                       test, nvfp4_scale,
                       std::vector<std::uint8_t>(
                           kRows * (kNvFp4Columns / 16U), 0x38U),
                       "tile NVFP4 scale");
  ready = ready && upload(test, companion_scales,
                          std::vector<float>(4U, 1.0F),
                          "tile companion scales");
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

  const auto run = [&test, &scratch, &output](
                       const runtime::ProjectionBackend backend,
                       const runtime::LinearWeight& weight,
                       const std::uint16_t* const activation,
                       const std::size_t token_count,
                       const std::array<std::uint16_t, kMaximumTokens>&
                           expected,
                       const bool needs_scratch,
                       const std::string& label) {
    const int status = runtime::launch_projection_tile_to_bf16_cuda(
        backend, weight, activation, token_count,
        needs_scratch ? scratch.get() : nullptr,
        needs_scratch ? kRows : 0U, output.get());
    test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
                label + " launch succeeds");
    if (static_cast<cudaError_t>(status) == cudaSuccess) {
      (void)expect_tile_output(
          test, output, token_count, kRows,
          std::vector<std::uint16_t>(expected.begin(),
                                     expected.begin() + token_count),
          label);
    }
  };

  for (std::size_t token_count = 1U; token_count <= kMaximumTokens;
       ++token_count) {
    const std::string suffix = " M=" + std::to_string(token_count);
    run(runtime::ProjectionBackend::kSm87WeightOnly, fp8,
        fp8_activation.get(), token_count, kFp8Expected, false,
        "SM87 FP8 tile" + suffix);
    run(runtime::ProjectionBackend::kReference, fp8, fp8_activation.get(),
        token_count, kFp8Expected, true, "reference FP8 tile" + suffix);
    run(runtime::ProjectionBackend::kSm87WeightOnly, nvfp4,
        nvfp4_activation.get(), token_count, kNvFp4Expected, false,
        "SM87 NVFP4 tile" + suffix);
    run(runtime::ProjectionBackend::kReference, nvfp4,
        nvfp4_activation.get(), token_count, kNvFp4Expected, true,
        "reference NVFP4 tile" + suffix);
    run(runtime::ProjectionBackend::kSm87WeightOnly, bf16,
        bf16_activation.get(), token_count, kBf16Expected, true,
        "SM87 BF16 fallback tile" + suffix);
    run(runtime::ProjectionBackend::kReference, bf16,
        bf16_activation.get(), token_count, kBf16Expected, true,
        "reference BF16 tile" + suffix);
  }

  const runtime::LinearWeight fp8_fallback = runtime::Fp8LinearWeight{
      fp8_weight.get(), companion_scales.get(), companion_scales.get() + 1,
      1.0F, 1.0F, kRows, 32U};
  const runtime::LinearWeight nvfp4_fallback = runtime::NvFp4LinearWeight{
      nvfp4_weight.get(), nvfp4_scale.get(), companion_scales.get() + 2,
      companion_scales.get() + 3, 1.0F, 1.0F, kRows, 16U};
  run(runtime::ProjectionBackend::kSm87WeightOnly, fp8_fallback,
      fp8_activation.get(), 3U,
      std::array<std::uint16_t, kMaximumTokens>{
          0x4200U, 0x4200U, 0x4200U, 0U, 0U, 0U, 0U, 0U},
      false, "SM87 FP8 unsupported-shape fallback");
  run(runtime::ProjectionBackend::kSm87WeightOnly, nvfp4_fallback,
      nvfp4_activation.get(), 3U,
      std::array<std::uint16_t, kMaximumTokens>{
          0x4180U, 0x4180U, 0x4180U, 0U, 0U, 0U, 0U, 0U},
      false, "SM87 NVFP4 unsupported-shape fallback");
}

void test_tile_validation(TestContext& test) {
  constexpr std::size_t kRows = 2U;
  constexpr std::size_t kFp8Columns = 32U;
  constexpr std::size_t kNvFp4Columns = 16U;
  constexpr std::size_t kBf16Columns = 4U;
  constexpr std::size_t kTokens = 16U;

  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> bf16_weight;
  DeviceBuffer<std::uint8_t> fp8_weight;
  DeviceBuffer<std::uint8_t> nvfp4_weight;
  DeviceBuffer<std::uint8_t> nvfp4_scale;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<float> scratch;
  DeviceBuffer<std::uint16_t> output;
  bool ready = activation.allocate(
      test, kTokens * kFp8Columns, "validation activation");
  ready = ready && bf16_weight.allocate(
                       test, kRows * kBf16Columns,
                       "validation BF16 weight");
  ready = ready && fp8_weight.allocate(
                       test, kRows * kFp8Columns,
                       "validation FP8 weight");
  ready = ready && nvfp4_weight.allocate(
                       test, kRows * (kNvFp4Columns / 2U),
                       "validation NVFP4 weight");
  ready = ready && nvfp4_scale.allocate(
                       test, kRows * (kNvFp4Columns / 16U),
                       "validation NVFP4 scale");
  ready = ready && companion_scales.allocate(
                       test, 4U, "validation companion scales");
  ready = ready && scratch.allocate(test, kRows, "validation scratch");
  ready = ready && output.allocate(
                       test, kTokens * kRows, "validation output");
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

  const auto expect_invalid = [&test](
                                  const runtime::ProjectionBackend backend,
                                  const runtime::LinearWeight& weight,
                                  const std::uint16_t* const input,
                                  const std::size_t token_count,
                                  float* const fp32_scratch,
                                  const std::size_t scratch_elements,
                                  std::uint16_t* const tile_output,
                                  const std::string& label) {
    test.expect(
        static_cast<cudaError_t>(
            runtime::launch_projection_tile_to_bf16_cuda(
                backend, weight, input, token_count, fp32_scratch,
                scratch_elements, tile_output)) == cudaErrorInvalidValue,
        label);
  };

  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly, fp8,
                 activation.get(), 0U, nullptr, 0U, output.get(),
                 "tile rejects M=0");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly, fp8,
                 activation.get(), 17U, nullptr, 0U, output.get(),
                 "tile rejects M=17");
  expect_invalid(static_cast<runtime::ProjectionBackend>(0xffU), fp8,
                 activation.get(), kTokens, nullptr, 0U, output.get(),
                 "tile rejects unknown backend");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly, fp8, nullptr,
                 kTokens, nullptr, 0U, output.get(),
                 "tile rejects null input");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly, fp8,
                 activation.get(), kTokens, nullptr, 0U, nullptr,
                 "tile rejects null output");

  const runtime::LinearWeight null_fp8 = runtime::Fp8LinearWeight{
      nullptr, companion_scales.get(), companion_scales.get() + 1,
      1.0F, 1.0F, kRows, kFp8Columns};
  const runtime::LinearWeight null_fp8_scale = runtime::Fp8LinearWeight{
      fp8_weight.get(), nullptr, companion_scales.get() + 1,
      1.0F, 1.0F, kRows, kFp8Columns};
  const runtime::LinearWeight nan_fp8 = runtime::Fp8LinearWeight{
      fp8_weight.get(), companion_scales.get(), companion_scales.get() + 1,
      std::numeric_limits<float>::quiet_NaN(), 1.0F, kRows, kFp8Columns};
  const runtime::LinearWeight negative_input_scale =
      runtime::Fp8LinearWeight{
          fp8_weight.get(), companion_scales.get(),
          companion_scales.get() + 1, 1.0F, -1.0F, kRows, kFp8Columns};
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly, null_fp8,
                 activation.get(), kTokens, nullptr, 0U, output.get(),
                 "tile rejects null FP8 weight");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly,
                 null_fp8_scale, activation.get(), kTokens, nullptr, 0U,
                 output.get(), "tile rejects null FP8 companion scale");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly, nan_fp8,
                 activation.get(), kTokens, nullptr, 0U, output.get(),
                 "tile rejects non-finite weight scale");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly,
                 negative_input_scale, activation.get(), kTokens, nullptr,
                 0U, output.get(), "tile rejects negative input scale");

  const runtime::LinearWeight null_nvfp4 = runtime::NvFp4LinearWeight{
      nvfp4_weight.get(), nullptr, companion_scales.get() + 2,
      companion_scales.get() + 3, 1.0F, 1.0F, kRows, kNvFp4Columns};
  const runtime::LinearWeight ungrouped_nvfp4 =
      runtime::NvFp4LinearWeight{
          nvfp4_weight.get(), nvfp4_scale.get(),
          companion_scales.get() + 2, companion_scales.get() + 3, 1.0F,
          1.0F, kRows, kNvFp4Columns - 1U};
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly, null_nvfp4,
                 activation.get(), kTokens, nullptr, 0U, output.get(),
                 "tile rejects null NVFP4 block scale");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly,
                 ungrouped_nvfp4, activation.get(), kTokens, nullptr, 0U,
                 output.get(), "tile rejects ungrouped NVFP4 K");

  const runtime::LinearWeight empty_rows = runtime::Fp8LinearWeight{
      fp8_weight.get(), companion_scales.get(), companion_scales.get() + 1,
      1.0F, 1.0F, 0U, kFp8Columns};
  const runtime::LinearWeight empty_columns = runtime::Fp8LinearWeight{
      fp8_weight.get(), companion_scales.get(), companion_scales.get() + 1,
      1.0F, 1.0F, kRows, 0U};
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly, empty_rows,
                 activation.get(), kTokens, nullptr, 0U, output.get(),
                 "tile rejects zero rows");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly, empty_columns,
                 activation.get(), kTokens, nullptr, 0U, output.get(),
                 "tile rejects zero columns");

  expect_invalid(runtime::ProjectionBackend::kReference, fp8,
                 activation.get(), kTokens, nullptr, 0U, output.get(),
                 "reference tile requires scratch");
  expect_invalid(runtime::ProjectionBackend::kReference, fp8,
                 activation.get(), kTokens, scratch.get(), kRows - 1U,
                 output.get(), "reference tile rejects short scratch");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly, bf16,
                 activation.get(), kTokens, nullptr, 0U, output.get(),
                 "SM87 BF16 tile requires fallback scratch");

  ready = upload(test, activation,
                 std::vector<std::uint16_t>(kTokens * kFp8Columns,
                                            0x3f80U),
                 "C16 validation activation sentinel");
  ready = ready && test.cuda_ok(
                       cudaMemset(fp8_weight.get(), 0x38,
                                  kRows * kFp8Columns),
                       "initialize C16 validation FP8 weights");
  if (ready) {
    std::uint16_t* const overlapping_output =
        activation.get() + 9U * kFp8Columns;
    const int overlap_status =
        runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, fp8,
            activation.get(), kTokens, nullptr, 0U, overlapping_output);
    test.expect(static_cast<cudaError_t>(overlap_status) ==
                    cudaErrorInvalidValue,
                "C16 tile rejects output overlapping only the second input chunk");
    std::uint16_t preserved = 0U;
    ready = test.cuda_ok(cudaMemcpy(&preserved, overlapping_output,
                                    sizeof(preserved),
                                    cudaMemcpyDeviceToHost),
                         "read C16 validation activation sentinel");
    if (ready) {
      test.expect(
          preserved == 0x3f80U,
          "C16 whole-tile validation rejects before the first SM87 launch");
    }
  }
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, fp8, activation.get(),
      kTokens, nullptr, 0U,
      reinterpret_cast<std::uint16_t*>(fp8_weight.get() +
                                       kRows * kFp8Columns - 2U),
      "tile rejects output overlapping the end of FP8 weights");
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, nvfp4, activation.get(),
      kTokens, nullptr, 0U,
      reinterpret_cast<std::uint16_t*>(nvfp4_scale.get()),
      "tile rejects output overlapping NVFP4 block scales");
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, fp8, activation.get(),
      kTokens, nullptr, 0U,
      reinterpret_cast<std::uint16_t*>(companion_scales.get()),
      "tile rejects output overlapping FP8 companion scales");
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, nvfp4, activation.get(),
      kTokens, nullptr, 0U,
      reinterpret_cast<std::uint16_t*>(companion_scales.get() + 3U),
      "tile rejects output overlapping NVFP4 companion scales");

  expect_invalid(
      runtime::ProjectionBackend::kReference, fp8, activation.get(), kTokens,
      reinterpret_cast<float*>(activation.get() + kFp8Columns), kRows,
      output.get(), "tile rejects scratch overlapping a future input token");
  expect_invalid(runtime::ProjectionBackend::kReference, fp8,
                 activation.get(), kTokens,
                 reinterpret_cast<float*>(output.get()), kRows, output.get(),
                 "tile rejects scratch overlapping output");
  expect_invalid(runtime::ProjectionBackend::kReference, fp8,
                 activation.get(), kTokens,
                 reinterpret_cast<float*>(fp8_weight.get()), kRows,
                 output.get(), "tile rejects scratch overlapping weights");
  expect_invalid(runtime::ProjectionBackend::kReference, nvfp4,
                 activation.get(), kTokens,
                 reinterpret_cast<float*>(nvfp4_scale.get()), kRows,
                 output.get(),
                 "tile rejects scratch overlapping NVFP4 block scales");
  expect_invalid(runtime::ProjectionBackend::kReference, fp8,
                 activation.get(), kTokens, companion_scales.get(), kRows,
                 output.get(),
                 "tile rejects scratch overlapping companion scales");

  constexpr std::size_t kMaximum =
      std::numeric_limits<std::size_t>::max();
  const runtime::LinearWeight product_overflow = runtime::Fp8LinearWeight{
      fp8_weight.get(), companion_scales.get(), companion_scales.get() + 1,
      1.0F, 1.0F, kMaximum, 2U};
  const runtime::LinearWeight input_tile_overflow =
      runtime::Fp8LinearWeight{
          fp8_weight.get(), companion_scales.get(),
          companion_scales.get() + 1, 1.0F, 1.0F, 1U,
          kMaximum / 2U + 1U};
  const runtime::LinearWeight output_byte_overflow =
      runtime::Fp8LinearWeight{
          fp8_weight.get(), companion_scales.get(),
          companion_scales.get() + 1, 1.0F, 1.0F,
          kMaximum / 4U + 1U, 1U};
  const runtime::LinearWeight bf16_byte_overflow =
      runtime::Bf16LinearWeight{bf16_weight.get(), 1U,
                                kMaximum / 2U + 1U};
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly,
                 product_overflow, activation.get(), kTokens, nullptr, 0U,
                 output.get(), "tile rejects N*K overflow");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly,
                 input_tile_overflow, activation.get(), kTokens, nullptr, 0U,
                 output.get(), "tile rejects M*K overflow");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly,
                 output_byte_overflow, activation.get(), kTokens, nullptr,
                 0U, output.get(), "tile rejects output byte overflow");
  expect_invalid(runtime::ProjectionBackend::kReference,
                 bf16_byte_overflow, activation.get(), kTokens,
                 scratch.get(), kRows, output.get(),
                 "tile rejects BF16 weight byte overflow");

  const std::uintptr_t near_end =
      std::numeric_limits<std::uintptr_t>::max() - 3U;
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, fp8,
      reinterpret_cast<const std::uint16_t*>(near_end), kTokens, nullptr, 0U,
      output.get(), "tile rejects wrapping input address range");
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly, fp8,
                 activation.get(), kTokens, nullptr, 0U,
                 reinterpret_cast<std::uint16_t*>(near_end),
                 "tile rejects wrapping output address range");
  const runtime::LinearWeight wrapping_weight = runtime::Fp8LinearWeight{
      reinterpret_cast<const std::uint8_t*>(near_end),
      companion_scales.get(), companion_scales.get() + 1, 1.0F, 1.0F,
      kRows, kFp8Columns};
  expect_invalid(runtime::ProjectionBackend::kSm87WeightOnly,
                 wrapping_weight, activation.get(), kTokens, nullptr, 0U,
                 output.get(), "tile rejects wrapping weight address range");
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
  test_tile_routes(test);
  test_tile_validation(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " projection dispatch assertion(s) failed\n";
    return 1;
  }
  std::cout << "projection backend dispatch tests passed\n";
  return 0;
}
