#include "q3x/runtime/model_weights.h"

#include "q3x/kernels/sm87_weight_only_gemv.h"

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

template <typename Launch>
[[nodiscard]] std::size_t captured_kernel_node_count(
    TestContext& test, Launch&& launch, const std::string& label,
    std::size_t* const total_node_count = nullptr) {
  constexpr std::size_t kInvalidCount =
      std::numeric_limits<std::size_t>::max();
  if (total_node_count != nullptr) {
    *total_node_count = kInvalidCount;
  }
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create capture stream " + label);
  ready = ready && test.cuda_ok(
                       cudaStreamBeginCapture(stream,
                                              cudaStreamCaptureModeGlobal),
                       "begin capture " + label);
  if (ready) {
    test.expect(static_cast<cudaError_t>(launch(stream)) == cudaSuccess,
                label + " captured launch succeeds");
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         "end capture " + label);
  }

  std::size_t kernel_nodes = kInvalidCount;
  if (ready && graph != nullptr) {
    std::size_t node_count = 0U;
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                         "count graph nodes " + label);
    std::vector<cudaGraphNode_t> nodes(node_count);
    if (ready && node_count != 0U) {
      ready = test.cuda_ok(cudaGraphGetNodes(graph, nodes.data(), &node_count),
                           "read graph nodes " + label);
    }
    if (ready) {
      if (total_node_count != nullptr) {
        *total_node_count = node_count;
      }
      kernel_nodes = 0U;
      for (const cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        ready = test.cuda_ok(cudaGraphNodeGetType(node, &type),
                             "read graph node type " + label);
        if (ready && type == cudaGraphNodeTypeKernel) {
          ++kernel_nodes;
        }
      }
    }
  }
  if (graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(graph), "destroy graph " + label);
  }
  if (stream != nullptr) {
    (void)test.cuda_ok(cudaStreamDestroy(stream),
                       "destroy capture stream " + label);
  }
  return ready ? kernel_nodes : kInvalidCount;
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

void test_bf16_direct_production_dispatch(TestContext& test) {
  constexpr std::size_t kRows = 48U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::size_t kTokens = 2U;
  constexpr std::uint16_t kBf16One = 0x3f80U;
  constexpr std::uint16_t kExpected = 0x45a0U;

  DeviceBuffer<std::uint16_t> weights;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<float> scratch;
  DeviceBuffer<std::uint16_t> output;
  bool ready = weights.allocate(test, kRows * kColumns,
                                "direct BF16 dispatch weights");
  ready = ready && activation.allocate(
                       test, kTokens * kColumns,
                       "direct BF16 dispatch activations");
  ready = ready && scratch.allocate(test, kRows,
                                    "direct BF16 dispatch scratch");
  ready = ready && output.allocate(test, kTokens * kRows,
                                   "direct BF16 dispatch output");
  if (!ready) {
    return;
  }
  ready = upload(test, weights,
                 std::vector<std::uint16_t>(kRows * kColumns, kBf16One),
                 "direct BF16 dispatch weights");
  ready = ready && upload(
                       test, activation,
                       std::vector<std::uint16_t>(kTokens * kColumns,
                                                  kBf16One),
                       "direct BF16 dispatch activations");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight production = runtime::Bf16LinearWeight{
      weights.get(), kRows, kColumns};
  test.expect(static_cast<cudaError_t>(runtime::launch_projection_to_bf16_cuda(
                  runtime::ProjectionBackend::kSm87WeightOnly, production,
                  activation.get(), nullptr, 0U, output.get())) == cudaSuccess,
              "exact SM87 BF16 production shape accepts null scratch");
  (void)expect_output(test, output, kExpected,
                      "exact SM87 BF16 direct dispatch");

  std::size_t direct_total_nodes = 0U;
  const std::size_t direct_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production,
            activation.get(), nullptr, 0U, output.get(),
            static_cast<void*>(stream));
      },
      "exact SM87 BF16 direct dispatch graph", &direct_total_nodes);
  test.expect(direct_total_nodes == 1U && direct_kernel_nodes == 1U,
              "exact SM87 BF16 production shape dispatches one kernel");

  test.expect(static_cast<cudaError_t>(runtime::launch_projection_to_bf16_cuda(
                  runtime::ProjectionBackend::kReference, production,
                  activation.get(), nullptr, 0U, output.get())) ==
                  cudaErrorInvalidValue,
              "reference BF16 production shape still requires scratch");
  std::size_t reference_total_nodes = 0U;
  const std::size_t reference_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_to_bf16_cuda(
            runtime::ProjectionBackend::kReference, production,
            activation.get(), scratch.get(), kRows, output.get(),
            static_cast<void*>(stream));
      },
      "reference BF16 production dispatch graph", &reference_total_nodes);
  test.expect(reference_total_nodes == 2U && reference_kernel_nodes == 2U,
              "reference BF16 production shape preserves GEMV plus convert");

  const auto check_near_miss = [&](const std::size_t rows,
                                   const std::size_t columns,
                                   const std::string& label) {
    const runtime::LinearWeight near_miss = runtime::Bf16LinearWeight{
        weights.get(), rows, columns};
    test.expect(static_cast<cudaError_t>(
                    runtime::launch_projection_to_bf16_cuda(
                        runtime::ProjectionBackend::kSm87WeightOnly,
                        near_miss, activation.get(), nullptr, 0U,
                        output.get())) == cudaErrorInvalidValue,
                label + " requires fallback scratch");
    std::size_t total_nodes = 0U;
    const std::size_t kernel_nodes = captured_kernel_node_count(
        test,
        [&](cudaStream_t stream) noexcept {
          return runtime::launch_projection_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly, near_miss,
              activation.get(), scratch.get(), kRows, output.get(),
              static_cast<void*>(stream));
        },
        label + " graph", &total_nodes);
    test.expect(total_nodes == 2U && kernel_nodes == 2U,
                label + " preserves GEMV plus convert fallback");
  };
  check_near_miss(kRows - 1U, kColumns,
                  "SM87 BF16 near-miss 47x5120");
  check_near_miss(kRows, kColumns - 1U,
                  "SM87 BF16 near-miss 48x5119");

  test.expect(static_cast<cudaError_t>(
                  runtime::launch_projection_tile_to_bf16_cuda(
                      runtime::ProjectionBackend::kSm87WeightOnly,
                      production, activation.get(), kTokens, nullptr, 0U,
                      output.get())) == cudaErrorInvalidValue,
              "BF16 production prefill remains scratch-backed");
  std::size_t prefill_total_nodes = 0U;
  const std::size_t prefill_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production,
            activation.get(), kTokens, scratch.get(), kRows, output.get(),
            static_cast<void*>(stream));
      },
      "SM87 BF16 production prefill M2 graph", &prefill_total_nodes);
  test.expect(prefill_total_nodes == 4U && prefill_kernel_nodes == 4U,
              "BF16 production prefill M2 remains two reference pairs");
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
    bool ready = test.cuda_ok(
        cudaMemset(output.get(), 0xa5,
                   kMaximumTokens * kRows * sizeof(std::uint16_t)),
        "initialize output canary " + label);
    if (!ready) {
      return;
    }
    const int status = runtime::launch_projection_tile_to_bf16_cuda(
        backend, weight, activation, token_count,
        needs_scratch ? scratch.get() : nullptr,
        needs_scratch ? kRows : 0U, output.get());
    test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
                label + " launch succeeds");
    if (static_cast<cudaError_t>(status) == cudaSuccess) {
      ready = expect_tile_output(
          test, output, token_count, kRows,
          std::vector<std::uint16_t>(expected.begin(),
                                     expected.begin() + token_count),
          label);
      std::vector<std::uint16_t> tail((kMaximumTokens - token_count) * kRows);
      if (ready && !tail.empty()) {
        ready = test.cuda_ok(
            cudaMemcpy(tail.data(), output.get() + token_count * kRows,
                       tail.size() * sizeof(tail.front()),
                       cudaMemcpyDeviceToHost),
            "download output canary " + label);
      }
      if (ready) {
        test.expect(
            std::all_of(tail.begin(), tail.end(),
                        [](const std::uint16_t value) noexcept {
                          return value == 0xa5a5U;
                        }),
            label + " preserves every token after the requested tile");
      }
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

  const auto capture_fp8 = [&](const std::size_t token_count,
                               const std::string& label) {
    return captured_kernel_node_count(
        test,
        [&](cudaStream_t stream) noexcept {
          return runtime::launch_projection_tile_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly, fp8,
              fp8_activation.get(), token_count, nullptr, 0U, output.get(),
              static_cast<void*>(stream));
        },
        label);
  };
  test.expect(capture_fp8(9U, "SM87 FP8 M9 dispatch graph") == 2U,
              "SM87 FP8 M9 remains M8+M1");
  test.expect(capture_fp8(15U, "SM87 FP8 M15 dispatch graph") == 2U,
              "SM87 FP8 M15 remains M8+M7");
  test.expect(capture_fp8(16U, "SM87 FP8 generic M16 dispatch graph") == 2U,
              "SM87 FP8 generic M16 uses the public two-M8 fallback");
  const auto capture_nvfp4 = [&](const std::size_t token_count,
                                 const std::string& label) {
    return captured_kernel_node_count(
        test,
        [&](cudaStream_t stream) noexcept {
          return runtime::launch_projection_tile_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly, nvfp4,
              nvfp4_activation.get(), token_count, nullptr, 0U, output.get(),
              static_cast<void*>(stream));
        },
        label);
  };
  test.expect(capture_nvfp4(9U, "SM87 NVFP4 M9 dispatch graph") == 2U,
              "SM87 NVFP4 M9 remains M8+M1");
  test.expect(capture_nvfp4(15U, "SM87 NVFP4 M15 dispatch graph") == 2U,
              "SM87 NVFP4 M15 remains M8+M7");
  test.expect(
      capture_nvfp4(16U, "SM87 NVFP4 generic M16 dispatch graph") == 2U,
      "SM87 NVFP4 generic M16 uses the public two-M8 fallback");
}

void test_bf16_projection_pair_dispatch(TestContext& test) {
  constexpr std::size_t kTokens = 16U;
  constexpr std::size_t kRows = 48U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::uint16_t kBf16One = 0x3f80U;
  constexpr std::uint16_t kBf16Half = 0x3f00U;
  constexpr std::uint16_t kFirstExpected = 0x45a0U;
  constexpr std::uint16_t kSecondExpected = 0x4520U;

  DeviceBuffer<std::uint16_t> first_weights;
  DeviceBuffer<std::uint16_t> second_weights;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<float> scratch;
  DeviceBuffer<std::uint16_t> first_output;
  DeviceBuffer<std::uint16_t> second_output;
  bool ready = first_weights.allocate(
      test, kRows * kColumns, "pair first BF16 weights");
  ready = ready && second_weights.allocate(
                       test, kRows * kColumns,
                       "pair second BF16 weights");
  ready = ready && activation.allocate(
                       test, kTokens * kColumns,
                       "pair BF16 activations");
  ready = ready && scratch.allocate(test, kRows, "pair FP32 scratch");
  ready = ready && first_output.allocate(
                       test, kTokens * kRows, "pair first output");
  ready = ready && second_output.allocate(
                       test, kTokens * kRows, "pair second output");
  if (!ready) {
    return;
  }

  ready = upload(test, first_weights,
                 std::vector<std::uint16_t>(kRows * kColumns, kBf16One),
                 "pair first BF16 weights");
  ready = ready && upload(
                       test, second_weights,
                       std::vector<std::uint16_t>(kRows * kColumns,
                                                  kBf16Half),
                       "pair second BF16 weights");
  ready = ready && upload(
                       test, activation,
                       std::vector<std::uint16_t>(kTokens * kColumns,
                                                  kBf16One),
                       "pair BF16 activations");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight first = runtime::Bf16LinearWeight{
      first_weights.get(), kRows, kColumns};
  const runtime::LinearWeight second = runtime::Bf16LinearWeight{
      second_weights.get(), kRows, kColumns};
  test.expect(runtime::supports_bf16_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first,
                  second),
              "production BF16 A/B pair selects the SM87 fast path");

  const auto run_fast = [&](const std::size_t token_count,
                            const std::string& label) {
    const int status = runtime::launch_projection_pair_tile_to_bf16_cuda(
        runtime::ProjectionBackend::kSm87WeightOnly, first, second,
        activation.get(), token_count, nullptr, 0U, first_output.get(),
        second_output.get());
    test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
                label + " accepts null unused scratch");
    if (static_cast<cudaError_t>(status) == cudaSuccess) {
      (void)expect_tile_output(
          test, first_output, token_count, kRows,
          std::vector<std::uint16_t>(token_count, kFirstExpected),
          label + " first output");
      (void)expect_tile_output(
          test, second_output, token_count, kRows,
          std::vector<std::uint16_t>(token_count, kSecondExpected),
          label + " second output");
    }

    std::size_t total_nodes = 0U;
    const std::size_t kernel_nodes = captured_kernel_node_count(
        test,
        [&](cudaStream_t stream) noexcept {
          return runtime::launch_projection_pair_tile_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly, first, second,
              activation.get(), token_count, nullptr, 0U,
              first_output.get(), second_output.get(),
              static_cast<void*>(stream));
        },
        label + " graph", &total_nodes);
    test.expect(total_nodes == 1U && kernel_nodes == 1U,
                label + " graph contains exactly one fused kernel node");
  };
  run_fast(1U, "SM87 production BF16 pair M1");
  run_fast(kTokens, "SM87 production BF16 pair M16");

  std::size_t reference_total_nodes = 0U;
  const std::size_t reference_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kReference, first, second,
            activation.get(), 1U, scratch.get(), kRows,
            first_output.get(), second_output.get(),
            static_cast<void*>(stream));
      },
      "reference BF16 pair M1 graph", &reference_total_nodes);
  test.expect(reference_total_nodes == 4U &&
                  reference_kernel_nodes == 4U,
              "reference BF16 pair preserves two GEMV-plus-convert paths");
  test.expect(static_cast<cudaError_t>(
                  runtime::launch_projection_pair_tile_to_bf16_cuda(
                      runtime::ProjectionBackend::kReference, first, second,
                      activation.get(), 1U, nullptr, 0U,
                      first_output.get(), second_output.get())) ==
                  cudaErrorInvalidValue,
              "reference BF16 pair still requires FP32 scratch");

  const runtime::LinearWeight awkward_first = runtime::Bf16LinearWeight{
      first_weights.get(), kRows - 1U, kColumns};
  const runtime::LinearWeight awkward_second = runtime::Bf16LinearWeight{
      second_weights.get(), kRows - 1U, kColumns};
  test.expect(!runtime::supports_bf16_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  awkward_first, awkward_second),
              "near-miss BF16 shape does not select the fused kernel");
  std::size_t awkward_total_nodes = 0U;
  const std::size_t awkward_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, awkward_first,
            awkward_second, activation.get(), 1U, scratch.get(), kRows,
            first_output.get(), second_output.get(),
            static_cast<void*>(stream));
      },
      "SM87 awkward BF16 pair M1 graph", &awkward_total_nodes);
  test.expect(awkward_total_nodes == 4U && awkward_kernel_nodes == 4U,
              "near-miss BF16 shape preserves two independent fallbacks");

  const auto expect_invalid = [&](const runtime::LinearWeight& first_arg,
                                  const runtime::LinearWeight& second_arg,
                                  const std::uint16_t* const input_arg,
                                  const std::size_t token_count,
                                  float* const scratch_arg,
                                  const std::size_t scratch_elements,
                                  std::uint16_t* const first_output_arg,
                                  std::uint16_t* const second_output_arg,
                                  const std::string& label) {
    const int status = runtime::launch_projection_pair_tile_to_bf16_cuda(
        runtime::ProjectionBackend::kSm87WeightOnly, first_arg, second_arg,
        input_arg, token_count, scratch_arg, scratch_elements,
        first_output_arg, second_output_arg);
    test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
                label);
  };
  const runtime::LinearWeight mismatched_columns =
      runtime::Bf16LinearWeight{second_weights.get(), kRows,
                                kColumns - 1U};
  const runtime::LinearWeight null_second = runtime::Bf16LinearWeight{
      nullptr, kRows, kColumns};
  expect_invalid(first, mismatched_columns, activation.get(), 1U,
                 scratch.get(), kRows, first_output.get(),
                 second_output.get(),
                 "pair rejects projections with different input sizes");
  expect_invalid(first, second, activation.get(), 0U, nullptr, 0U,
                 first_output.get(), second_output.get(),
                 "pair rejects M=0");
  expect_invalid(first, second, activation.get(), kTokens + 1U, nullptr, 0U,
                 first_output.get(), second_output.get(),
                 "pair rejects M=17");
  expect_invalid(first, second, activation.get(), 1U, nullptr, 0U,
                 reinterpret_cast<std::uint16_t*>(second_weights.get()),
                 second_output.get(),
                 "pair rejects first output overlapping second weights");
  expect_invalid(first, second, activation.get(), 1U, nullptr, 0U,
                 first_output.get(),
                 reinterpret_cast<std::uint16_t*>(first_weights.get()),
                 "pair rejects second output overlapping first weights");
  expect_invalid(first, second, activation.get(), 1U, nullptr, 0U,
                 first_output.get(), first_output.get(),
                 "pair rejects overlapping outputs");
  expect_invalid(awkward_first, awkward_second, activation.get(), 1U,
                 reinterpret_cast<float*>(second_output.get()), kRows,
                 first_output.get(), second_output.get(),
                 "pair fallback rejects scratch overlapping second output");

  ready = test.cuda_ok(
      cudaMemset(first_output.get(), 0xa5,
                 kTokens * kRows * sizeof(std::uint16_t)),
      "initialize pair fail-before-enqueue canary");
  if (ready) {
    expect_invalid(first, null_second, activation.get(), kTokens,
                   scratch.get(), kRows, first_output.get(),
                   second_output.get(),
                   "pair validates the second projection before enqueue");
    std::uint16_t preserved = 0U;
    ready = test.cuda_ok(cudaMemcpy(&preserved, first_output.get(),
                                    sizeof(preserved),
                                    cudaMemcpyDeviceToHost),
                         "read pair fail-before-enqueue canary");
    if (ready) {
      test.expect(preserved == 0xa5a5U,
                  "invalid second projection leaves first output untouched");
    }
  }
}

void test_fp8_projection_pair_dispatch(TestContext& test) {
  constexpr std::size_t kRows = 1'024U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr float kFirstWeightScale = 1.0F / 64.0F;
  constexpr float kSecondWeightScale = 1.0F / 128.0F;
  constexpr std::uint16_t kBf16One = 0x3f80U;
  constexpr std::uint16_t kFirstExpected = 0x42a0U;
  constexpr std::uint16_t kSecondExpected = 0x4220U;

  DeviceBuffer<std::uint8_t> first_weights;
  DeviceBuffer<std::uint8_t> second_weights;
  DeviceBuffer<std::uint8_t> misaligned_first_weight_storage;
  DeviceBuffer<std::uint8_t> misaligned_activation_storage;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<float> scratch;
  DeviceBuffer<std::uint16_t> first_output;
  DeviceBuffer<std::uint16_t> second_output;
  bool ready = first_weights.allocate(
      test, kRows * kColumns, "pair first FP8 weights");
  ready = ready && second_weights.allocate(
                       test, kRows * kColumns, "pair second FP8 weights");
  ready = ready && misaligned_first_weight_storage.allocate(
                       test, kRows * kColumns + 1U,
                       "pair misaligned first FP8 weight storage");
  ready = ready && misaligned_activation_storage.allocate(
                       test, 2U * kColumns * sizeof(std::uint16_t) + 2U,
                       "pair misaligned FP8 activation storage");
  ready = ready && activation.allocate(test, 2U * kColumns,
                                       "pair FP8 activations");
  ready = ready && companion_scales.allocate(
                       test, 4U, "pair FP8 companion scales");
  ready = ready && scratch.allocate(test, kRows, "pair FP8 scratch");
  ready = ready && first_output.allocate(
                       test, 2U * kRows, "pair first FP8 output");
  ready = ready && second_output.allocate(
                       test, 2U * kRows, "pair second FP8 output");
  if (!ready) {
    return;
  }

  ready = test.cuda_ok(
      cudaMemset(first_weights.get(), 0x38, kRows * kColumns),
      "initialize pair first FP8 weights");
  ready = ready && test.cuda_ok(
                       cudaMemset(second_weights.get(), 0x38,
                                  kRows * kColumns),
                       "initialize pair second FP8 weights");
  ready = ready && test.cuda_ok(
                       cudaMemset(misaligned_first_weight_storage.get(),
                                  0x38, kRows * kColumns + 1U),
                       "initialize pair misaligned first FP8 weights");
  const std::vector<std::uint16_t> host_activations(
      2U * kColumns, kBf16One);
  ready = ready && upload(
                       test, activation, host_activations,
                       "pair FP8 activations");
  std::uint16_t* const misaligned_activation =
      reinterpret_cast<std::uint16_t*>(
          misaligned_activation_storage.get() + 2U);
  ready = ready && test.cuda_ok(
                       cudaMemcpy(misaligned_activation,
                                  host_activations.data(),
                                  host_activations.size() *
                                      sizeof(host_activations.front()),
                                  cudaMemcpyHostToDevice),
                       "upload pair misaligned FP8 activations");
  ready = ready && upload(
                       test, companion_scales,
                       std::vector<float>{kFirstWeightScale, 1.0F,
                                          kSecondWeightScale, 1.0F},
                       "pair FP8 companion scales");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight first = runtime::Fp8LinearWeight{
      first_weights.get(), companion_scales.get(),
      companion_scales.get() + 1U, kFirstWeightScale, 1.0F, kRows,
      kColumns};
  const runtime::LinearWeight second = runtime::Fp8LinearWeight{
      second_weights.get(), companion_scales.get() + 2U,
      companion_scales.get() + 3U, kSecondWeightScale, 1.0F, kRows,
      kColumns};
  test.expect(runtime::supports_fp8_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first,
                  second),
              "production FP8 K/V pair selects the SM87 fast path");

  int status = runtime::launch_projection_pair_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, first, second,
      activation.get(), 1U, nullptr, 0U, first_output.get(),
      second_output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "SM87 production FP8 pair M1 accepts null unused scratch");
  if (static_cast<cudaError_t>(status) == cudaSuccess) {
    (void)expect_tile_output(
        test, first_output, 1U, kRows,
        std::vector<std::uint16_t>{kFirstExpected},
        "SM87 production FP8 pair M1 first output");
    (void)expect_tile_output(
        test, second_output, 1U, kRows,
        std::vector<std::uint16_t>{kSecondExpected},
        "SM87 production FP8 pair M1 second output");
  }

  std::size_t total_nodes = 0U;
  const std::size_t fused_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, first, second,
            activation.get(), 1U, nullptr, 0U, first_output.get(),
            second_output.get(), static_cast<void*>(stream));
      },
      "SM87 production FP8 pair M1 graph", &total_nodes);
  test.expect(total_nodes == 1U && fused_kernel_nodes == 1U,
              "FP8 K/V M1 graph contains exactly one fused kernel node");

  const runtime::LinearWeight unaligned_first = runtime::Fp8LinearWeight{
      misaligned_first_weight_storage.get() + 1U, companion_scales.get(),
      companion_scales.get() + 1U, kFirstWeightScale, 1.0F, kRows,
      kColumns};
  test.expect(runtime::supports_fp8_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  unaligned_first, second),
              "FP8 K/V model eligibility is independent of launch "
              "alignment");
  status = runtime::launch_projection_pair_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, unaligned_first, second,
      misaligned_activation, 1U, nullptr, 0U, first_output.get(),
      second_output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "unaligned exact FP8 K/V M1 preserves the split fallback");
  if (static_cast<cudaError_t>(status) == cudaSuccess) {
    (void)expect_tile_output(
        test, first_output, 1U, kRows,
        std::vector<std::uint16_t>{kFirstExpected},
        "unaligned exact FP8 pair first fallback output");
    (void)expect_tile_output(
        test, second_output, 1U, kRows,
        std::vector<std::uint16_t>{kSecondExpected},
        "unaligned exact FP8 pair second fallback output");
  }
  std::size_t unaligned_total_nodes = 0U;
  const std::size_t unaligned_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, unaligned_first,
            second, misaligned_activation, 1U, nullptr, 0U,
            first_output.get(), second_output.get(),
            static_cast<void*>(stream));
      },
      "SM87 unaligned exact FP8 pair M1 graph", &unaligned_total_nodes);
  test.expect(unaligned_total_nodes == 2U &&
                  unaligned_kernel_nodes == 2U,
              "unaligned exact FP8 K/V M1 graph preserves two fallback "
              "kernels");

  std::size_t m2_total_nodes = 0U;
  const std::size_t m2_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, first, second,
            activation.get(), 2U, nullptr, 0U, first_output.get(),
            second_output.get(), static_cast<void*>(stream));
      },
      "SM87 production FP8 pair M2 graph", &m2_total_nodes);
  test.expect(m2_total_nodes == 2U && m2_kernel_nodes == 2U,
              "FP8 K/V M2 preserves two independent projection kernels");

  const runtime::LinearWeight near_miss = runtime::Fp8LinearWeight{
      second_weights.get(), companion_scales.get() + 2U,
      companion_scales.get() + 3U, kSecondWeightScale, 1.0F, kRows - 1U,
      kColumns};
  test.expect(!runtime::supports_fp8_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first,
                  near_miss),
              "near-miss FP8 K/V shape does not select the fused kernel");
  std::size_t near_miss_total_nodes = 0U;
  const std::size_t near_miss_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, first, near_miss,
            activation.get(), 1U, nullptr, 0U, first_output.get(),
            second_output.get(), static_cast<void*>(stream));
      },
      "SM87 near-miss FP8 pair M1 graph", &near_miss_total_nodes);
  test.expect(near_miss_total_nodes == 2U && near_miss_kernel_nodes == 2U,
              "near-miss FP8 K/V shape preserves split M1 projections");

  std::size_t reference_total_nodes = 0U;
  const std::size_t reference_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kReference, first, second,
            activation.get(), 1U, scratch.get(), kRows, first_output.get(),
            second_output.get(), static_cast<void*>(stream));
      },
      "reference FP8 pair M1 graph", &reference_total_nodes);
  test.expect(reference_total_nodes == 4U &&
                  reference_kernel_nodes == 4U,
              "reference FP8 K/V pair preserves two GEMV-plus-convert paths");

  status = runtime::launch_projection_pair_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, first, second,
      activation.get(), 1U, nullptr, 0U, first_output.get(),
      first_output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
              "fused FP8 K/V pair rejects overlapping outputs");
  status = runtime::launch_projection_pair_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, first, second,
      activation.get(), 1U, nullptr, 0U,
      reinterpret_cast<std::uint16_t*>(second_weights.get()),
      second_output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
              "fused FP8 K/V pair rejects output overlapping peer weights");

  const runtime::LinearWeight mismatched_columns =
      runtime::Fp8LinearWeight{
          second_weights.get(), companion_scales.get() + 2U,
          companion_scales.get() + 3U, kSecondWeightScale, 1.0F, kRows,
          kColumns - 1U};
  status = runtime::launch_projection_pair_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, first,
      mismatched_columns, activation.get(), 1U, nullptr, 0U,
      first_output.get(), second_output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
              "fused FP8 K/V pair rejects mismatched input sizes");
  status = runtime::launch_projection_pair_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, first, second,
      activation.get(), 0U, nullptr, 0U, first_output.get(),
      second_output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
              "FP8 K/V pair rejects M=0");

  const runtime::LinearWeight malformed_second = runtime::Fp8LinearWeight{
      second_weights.get(), nullptr, companion_scales.get() + 3U,
      kSecondWeightScale, 1.0F, kRows, kColumns};
  ready = test.cuda_ok(
      cudaMemset(first_output.get(), 0xa5,
                 2U * kRows * sizeof(std::uint16_t)),
      "initialize FP8 pair fail-before-enqueue canary");
  if (ready) {
    status = runtime::launch_projection_pair_tile_to_bf16_cuda(
        runtime::ProjectionBackend::kSm87WeightOnly, first,
        malformed_second, activation.get(), 1U, nullptr, 0U,
        first_output.get(), second_output.get());
    test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
                "FP8 pair validates the second projection before enqueue");
    std::uint16_t preserved = 0U;
    ready = test.cuda_ok(cudaMemcpy(&preserved, first_output.get(),
                                    sizeof(preserved),
                                    cudaMemcpyDeviceToHost),
                         "read FP8 pair fail-before-enqueue canary");
    if (ready) {
      test.expect(preserved == 0xa5a5U,
                  "invalid second FP8 projection leaves first output "
                  "untouched");
    }
  }
}

void test_fp8_qkv_z_projection_pair_dispatch(TestContext& test) {
  constexpr std::size_t kQkvRows = 10'240U;
  constexpr std::size_t kZRows = 6'144U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr float kQkvWeightScale = 1.0F / 64.0F;
  constexpr float kZWeightScale = 1.0F / 128.0F;
  constexpr std::uint16_t kBf16One = 0x3f80U;
  constexpr std::uint16_t kQkvExpected = 0x42a0U;
  constexpr std::uint16_t kZExpected = 0x4220U;

  DeviceBuffer<std::uint8_t> qkv_weights;
  DeviceBuffer<std::uint8_t> z_weights;
  DeviceBuffer<std::uint8_t> misaligned_activation_storage;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<float> scratch;
  DeviceBuffer<std::uint16_t> qkv_output;
  DeviceBuffer<std::uint16_t> z_output;
  bool ready = qkv_weights.allocate(
      test, kQkvRows * kColumns, "QKV/Z QKV FP8 weights");
  ready = ready && z_weights.allocate(
                       test, kZRows * kColumns, "QKV/Z Z FP8 weights");
  ready = ready && misaligned_activation_storage.allocate(
                       test, 2U * kColumns * sizeof(std::uint16_t) + 2U,
                       "QKV/Z misaligned activation storage");
  ready = ready && activation.allocate(
                       test, 2U * kColumns, "QKV/Z FP8 activations");
  ready = ready && companion_scales.allocate(
                       test, 4U, "QKV/Z FP8 companion scales");
  ready = ready && scratch.allocate(
                       test, kQkvRows, "QKV/Z FP8 reference scratch");
  ready = ready && qkv_output.allocate(
                       test, 2U * kQkvRows, "QKV/Z QKV output");
  ready = ready && z_output.allocate(
                       test, 2U * kZRows, "QKV/Z Z output");
  if (!ready) {
    return;
  }

  ready = test.cuda_ok(
      cudaMemset(qkv_weights.get(), 0x38, kQkvRows * kColumns),
      "initialize QKV/Z QKV FP8 weights");
  ready = ready && test.cuda_ok(
                       cudaMemset(z_weights.get(), 0x38,
                                  kZRows * kColumns),
                       "initialize QKV/Z Z FP8 weights");
  const std::vector<std::uint16_t> host_activations(
      2U * kColumns, kBf16One);
  ready = ready && upload(test, activation, host_activations,
                           "QKV/Z FP8 activations");
  std::uint16_t* const misaligned_activation =
      reinterpret_cast<std::uint16_t*>(
          misaligned_activation_storage.get() + 2U);
  ready = ready && test.cuda_ok(
                       cudaMemcpy(misaligned_activation,
                                  host_activations.data(),
                                  host_activations.size() *
                                      sizeof(host_activations.front()),
                                  cudaMemcpyHostToDevice),
                       "upload QKV/Z misaligned FP8 activations");
  ready = ready && upload(
                       test, companion_scales,
                       std::vector<float>{kQkvWeightScale, 1.0F,
                                          kZWeightScale, 1.0F},
                       "QKV/Z FP8 companion scales");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight qkv = runtime::Fp8LinearWeight{
      qkv_weights.get(), companion_scales.get(),
      companion_scales.get() + 1U, kQkvWeightScale, 1.0F, kQkvRows,
      kColumns};
  const runtime::LinearWeight z = runtime::Fp8LinearWeight{
      z_weights.get(), companion_scales.get() + 2U,
      companion_scales.get() + 3U, kZWeightScale, 1.0F, kZRows, kColumns};
  test.expect(runtime::supports_fp8_qkv_z_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, qkv, z),
              "production FP8 QKV/Z pair selects the SM87 fast path");

  int status = runtime::launch_projection_pair_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, qkv, z,
      activation.get(), 1U, nullptr, 0U, qkv_output.get(), z_output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "SM87 production FP8 QKV/Z M1 accepts null unused scratch");
  if (static_cast<cudaError_t>(status) == cudaSuccess) {
    (void)expect_tile_output(
        test, qkv_output, 1U, kQkvRows,
        std::vector<std::uint16_t>{kQkvExpected},
        "SM87 production FP8 QKV/Z QKV output");
    (void)expect_tile_output(
        test, z_output, 1U, kZRows,
        std::vector<std::uint16_t>{kZExpected},
        "SM87 production FP8 QKV/Z Z output");
  }

  std::size_t fused_total_nodes = 0U;
  const std::size_t fused_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, z,
            activation.get(), 1U, nullptr, 0U, qkv_output.get(),
            z_output.get(), static_cast<void*>(stream));
      },
      "SM87 production FP8 QKV/Z M1 graph", &fused_total_nodes);
  test.expect(fused_total_nodes == 1U && fused_kernel_nodes == 1U,
              "FP8 QKV/Z M1 graph contains one fused kernel node");

  status = runtime::launch_projection_pair_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, qkv, z,
      misaligned_activation, 1U, nullptr, 0U, qkv_output.get(),
      z_output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "unaligned FP8 QKV/Z M1 preserves split fallback");
  if (static_cast<cudaError_t>(status) == cudaSuccess) {
    (void)expect_tile_output(
        test, qkv_output, 1U, kQkvRows,
        std::vector<std::uint16_t>{kQkvExpected},
        "unaligned FP8 QKV/Z QKV fallback output");
    (void)expect_tile_output(
        test, z_output, 1U, kZRows,
        std::vector<std::uint16_t>{kZExpected},
        "unaligned FP8 QKV/Z Z fallback output");
  }
  std::size_t unaligned_total_nodes = 0U;
  const std::size_t unaligned_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, z,
            misaligned_activation, 1U, nullptr, 0U, qkv_output.get(),
            z_output.get(), static_cast<void*>(stream));
      },
      "SM87 unaligned FP8 QKV/Z M1 graph", &unaligned_total_nodes);
  test.expect(unaligned_total_nodes == 2U &&
                  unaligned_kernel_nodes == 2U,
              "unaligned FP8 QKV/Z M1 graph preserves two kernels");

  std::size_t m2_total_nodes = 0U;
  const std::size_t m2_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, z,
            activation.get(), 2U, nullptr, 0U, qkv_output.get(),
            z_output.get(), static_cast<void*>(stream));
      },
      "SM87 FP8 QKV/Z M2 graph", &m2_total_nodes);
  test.expect(m2_total_nodes == 2U && m2_kernel_nodes == 2U,
              "FP8 QKV/Z M2 preserves two projection kernels");

  const runtime::LinearWeight near_miss_z = runtime::Fp8LinearWeight{
      z_weights.get(), companion_scales.get() + 2U,
      companion_scales.get() + 3U, kZWeightScale, 1.0F, kZRows - 1U,
      kColumns};
  test.expect(!runtime::supports_fp8_qkv_z_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                  near_miss_z),
              "near-miss FP8 QKV/Z shape preserves split fallback");
  std::size_t near_miss_total_nodes = 0U;
  const std::size_t near_miss_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, near_miss_z,
            activation.get(), 1U, nullptr, 0U, qkv_output.get(),
            z_output.get(), static_cast<void*>(stream));
      },
      "SM87 near-miss FP8 QKV/Z M1 graph", &near_miss_total_nodes);
  test.expect(near_miss_total_nodes == 2U &&
                  near_miss_kernel_nodes == 2U,
              "near-miss FP8 QKV/Z graph preserves two kernels");

  std::size_t reference_total_nodes = 0U;
  const std::size_t reference_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kReference, qkv, z,
            activation.get(), 1U, scratch.get(), kQkvRows,
            qkv_output.get(), z_output.get(), static_cast<void*>(stream));
      },
      "reference FP8 QKV/Z M1 graph", &reference_total_nodes);
  test.expect(reference_total_nodes == 4U &&
                  reference_kernel_nodes == 4U,
              "reference FP8 QKV/Z preserves two GEMV-plus-convert paths");

  status = runtime::launch_projection_pair_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, qkv, z,
      activation.get(), 1U, nullptr, 0U, qkv_output.get(),
      qkv_output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
              "fused FP8 QKV/Z rejects overlapping outputs");
  status = runtime::launch_projection_pair_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, qkv, z,
      activation.get(), 1U, nullptr, 0U,
      reinterpret_cast<std::uint16_t*>(z_weights.get()), z_output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
              "fused FP8 QKV/Z rejects QKV output over Z weights");
  status = runtime::launch_projection_pair_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, qkv, z,
      activation.get(), 1U, nullptr, 0U, qkv_output.get(),
      reinterpret_cast<std::uint16_t*>(z_weights.get()));
  test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
              "fused FP8 QKV/Z rejects Z output over its weights");

  const runtime::LinearWeight malformed_z = runtime::Fp8LinearWeight{
      z_weights.get(), nullptr, companion_scales.get() + 3U,
      kZWeightScale, 1.0F, kZRows, kColumns};
  ready = test.cuda_ok(
      cudaMemset(qkv_output.get(), 0xa5,
                 2U * kQkvRows * sizeof(std::uint16_t)),
      "initialize QKV/Z fail-before-enqueue canary");
  if (ready) {
    status = runtime::launch_projection_pair_tile_to_bf16_cuda(
        runtime::ProjectionBackend::kSm87WeightOnly, qkv, malformed_z,
        activation.get(), 1U, nullptr, 0U, qkv_output.get(), z_output.get());
    test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
                "FP8 QKV/Z validates Z before enqueue");
    std::uint16_t preserved = 0U;
    ready = test.cuda_ok(cudaMemcpy(&preserved, qkv_output.get(),
                                    sizeof(preserved),
                                    cudaMemcpyDeviceToHost),
                         "read QKV/Z fail-before-enqueue canary");
    if (ready) {
      test.expect(preserved == 0xa5a5U,
                  "invalid Z leaves QKV output untouched");
    }
  }
}

void test_fp8_m16_production_dispatch(TestContext& test) {
  constexpr std::size_t kTokens = 16U;
  constexpr std::size_t kRows = 5'120U;
  constexpr std::size_t kColumns = 6'144U;
  constexpr float kWeightScale = 1.0F / 64.0F;
  constexpr std::array<std::uint16_t, kTokens> kActivationValues{
      0x3f80U, 0x3f00U, 0xbf80U, 0x4000U,
      0x3e80U, 0xbf00U, 0x4080U, 0xc000U,
      0x3f40U, 0xbf40U, 0x4040U, 0xc040U,
      0x3fc0U, 0xbfc0U, 0x4100U, 0xc100U};
  constexpr std::array<std::uint16_t, kTokens> kExpected{
      0x42c0U, 0x4240U, 0xc2c0U, 0x4340U,
      0x41c0U, 0xc240U, 0x43c0U, 0xc340U,
      0x4290U, 0xc290U, 0x4390U, 0xc390U,
      0x4310U, 0xc310U, 0x4440U, 0xc440U};

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<std::uint16_t> output;
  DeviceBuffer<std::uint16_t> baseline_output;
  bool ready = weights.allocate(test, kRows * kColumns + 4U,
                                "production C16 FP8 weights");
  ready = ready && activations.allocate(
                       test, kTokens * kColumns + 1U,
                       "production C16 FP8 activations");
  ready = ready && companion_scales.allocate(
                       test, 2U, "production C16 companion scales");
  ready = ready && output.allocate(test, kTokens * kRows,
                                   "production C16 output");
  ready = ready && baseline_output.allocate(
                       test, kTokens * kRows,
                       "production C16 fallback baseline output");
  if (!ready) {
    return;
  }

  std::vector<std::uint16_t> host_activations(kTokens * kColumns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    std::fill_n(host_activations.begin() + token * kColumns, kColumns,
                kActivationValues[token]);
  }
  ready = test.cuda_ok(cudaMemset(weights.get(), 0x38, kRows * kColumns),
                       "initialize production C16 FP8 weights");
  ready = ready && upload(test, activations, host_activations,
                          "production C16 FP8 activations");
  ready = ready && upload(
                       test, companion_scales,
                       std::vector<float>{kWeightScale, 1.0F},
                       "production C16 companion scales");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight fp8 = runtime::Fp8LinearWeight{
      weights.get(), companion_scales.get(), companion_scales.get() + 1,
      kWeightScale, 1.0F, kRows, kColumns};
  const int status = runtime::launch_projection_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, fp8, activations.get(),
      kTokens, nullptr, 0U, output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "SM87 FP8 production-shape C16 dispatch succeeds");
  if (static_cast<cudaError_t>(status) == cudaSuccess) {
    (void)expect_tile_output(
        test, output, kTokens, kRows,
        std::vector<std::uint16_t>(kExpected.begin(), kExpected.end()),
        "SM87 FP8 production-shape C16 dispatch");
  }
  const std::size_t production_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, fp8,
            activations.get(), kTokens, nullptr, 0U, output.get(),
            static_cast<void*>(stream));
      },
      "SM87 FP8 production-shape C16 dispatch graph");
  test.expect(production_kernel_nodes == 1U,
              "SM87 FP8 production-shape C16 dispatch uses one WMMA kernel");

  ready = test.cuda_ok(cudaMemset(weights.get() + 4U, 0x38,
                                  kRows * kColumns),
                       "initialize 4-byte-aligned production FP8 weights");
  if (!ready) {
    return;
  }
  const std::size_t weight_fallback_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
            weights.get() + 4U, kWeightScale, activations.get(), kRows,
            kColumns, output.get(), static_cast<void*>(stream));
      },
      "SM87 FP8 production-shape 4-byte weight fallback graph");
  test.expect(weight_fallback_nodes == 2U,
              "4-byte but non-16-byte FP8 weights use two M8 kernels");
  int fallback_status =
      q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
          weights.get() + 4U, kWeightScale, activations.get(), kRows, kColumns,
          output.get());
  test.expect(static_cast<cudaError_t>(fallback_status) == cudaSuccess,
              "4-byte-aligned production-shape fallback succeeds");
  if (static_cast<cudaError_t>(fallback_status) == cudaSuccess) {
    (void)expect_tile_output(
        test, output, kTokens, kRows,
        std::vector<std::uint16_t>(kExpected.begin(), kExpected.end()),
        "4-byte-aligned production-shape fallback");
  }

  ready = test.cuda_ok(
      cudaMemcpy(activations.get() + 1U, host_activations.data(),
                 host_activations.size() * sizeof(host_activations.front()),
                 cudaMemcpyHostToDevice),
      "upload 2-byte-aligned production C16 activations");
  if (!ready) {
    return;
  }
  const std::size_t activation_fallback_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
            weights.get(), kWeightScale, activations.get() + 1U, kRows,
            kColumns, output.get(), static_cast<void*>(stream));
      },
      "SM87 FP8 production-shape 2-byte activation fallback graph");
  test.expect(activation_fallback_nodes == kTokens,
              "2-byte but non-8-byte activations use sixteen safe M1 kernels");
  fallback_status =
      q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
          weights.get(), kWeightScale, activations.get() + 1U, kRows, kColumns,
          output.get());
  test.expect(static_cast<cudaError_t>(fallback_status) == cudaSuccess,
              "2-byte-aligned production-shape fallback succeeds");
  if (static_cast<cudaError_t>(fallback_status) == cudaSuccess) {
    (void)expect_tile_output(
        test, output, kTokens, kRows,
        std::vector<std::uint16_t>(kExpected.begin(), kExpected.end()),
        "2-byte-aligned production-shape fallback");
  }

  constexpr std::size_t kFallbackRows = 1'024U;
  constexpr std::size_t kFallbackColumns = 5'120U;
  std::vector<std::uint16_t> fallback_activations(kTokens * kFallbackColumns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    std::fill_n(fallback_activations.begin() + token * kFallbackColumns,
                kFallbackColumns, kActivationValues[token]);
  }
  ready = test.cuda_ok(
      cudaMemcpy(activations.get(), fallback_activations.data(),
                 fallback_activations.size() *
                     sizeof(fallback_activations.front()),
                 cudaMemcpyHostToDevice),
      "upload 1024x5120 M16 fallback activations");
  if (!ready) {
    return;
  }
  const std::size_t small_shape_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
            weights.get(), kWeightScale, activations.get(), kFallbackRows,
            kFallbackColumns, output.get(), static_cast<void*>(stream));
      },
      "SM87 FP8 1024x5120 public M16 fallback graph");
  test.expect(small_shape_nodes == 2U,
              "1024x5120 public M16 uses exactly two M8 kernels");
  fallback_status =
      q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
          weights.get(), kWeightScale, activations.get(), kFallbackRows,
          kFallbackColumns, output.get());
  int baseline_status =
      q3x::kernels::launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
          weights.get(), kWeightScale, activations.get(), 8U, kFallbackRows,
          kFallbackColumns, baseline_output.get());
  if (baseline_status == static_cast<int>(cudaSuccess)) {
    baseline_status =
        q3x::kernels::launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
            weights.get(), kWeightScale,
            activations.get() + 8U * kFallbackColumns, 8U, kFallbackRows,
            kFallbackColumns, baseline_output.get() + 8U * kFallbackRows);
  }
  test.expect(static_cast<cudaError_t>(fallback_status) == cudaSuccess &&
                  static_cast<cudaError_t>(baseline_status) == cudaSuccess,
              "1024x5120 public and direct two-M8 launches succeed");
  std::vector<std::uint16_t> public_fallback(kTokens * kFallbackRows);
  std::vector<std::uint16_t> direct_two_m8(kTokens * kFallbackRows);
  ready = test.cuda_ok(cudaDeviceSynchronize(),
                       "synchronize 1024x5120 M16 fallback");
  ready = ready && test.cuda_ok(
                       cudaMemcpy(public_fallback.data(), output.get(),
                                  public_fallback.size() *
                                      sizeof(public_fallback.front()),
                                  cudaMemcpyDeviceToHost),
                       "download 1024x5120 public M16 fallback");
  ready = ready && test.cuda_ok(
                       cudaMemcpy(direct_two_m8.data(), baseline_output.get(),
                                  direct_two_m8.size() *
                                      sizeof(direct_two_m8.front()),
                                  cudaMemcpyDeviceToHost),
                       "download 1024x5120 direct two-M8 baseline");
  if (ready) {
    test.expect(public_fallback == direct_two_m8,
                "1024x5120 public M16 is bitwise equal to direct two-M8");
  }
}

void test_nvfp4_m16_production_dispatch(TestContext& test) {
  constexpr std::size_t kTokens = 16U;
  constexpr std::size_t kRows = 17'408U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::size_t kPackedBytes = kRows * (kColumns / 2U);
  constexpr std::size_t kScaleBytes = kRows * (kColumns / 16U);
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr std::array<std::uint16_t, kTokens> kActivationValues{
      0x3f80U, 0x3f00U, 0xbf80U, 0x4000U,
      0x3e80U, 0xbf00U, 0x4080U, 0xc000U,
      0x3f40U, 0xbf40U, 0x4040U, 0xc040U,
      0x3fc0U, 0xbfc0U, 0x4100U, 0xc100U};
  constexpr std::array<std::uint16_t, kTokens> kExpected{
      0x42a0U, 0x4220U, 0xc2a0U, 0x4320U,
      0x41a0U, 0xc220U, 0x43a0U, 0xc320U,
      0x4270U, 0xc270U, 0x4370U, 0xc370U,
      0x42f0U, 0xc2f0U, 0x4420U, 0xc420U};

  DeviceBuffer<std::uint8_t> packed_weights;
  DeviceBuffer<std::uint8_t> block_scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<std::uint16_t> output;
  bool ready = packed_weights.allocate(
      test, kPackedBytes + 4U, "production C16 NVFP4 packed weights");
  ready = ready && block_scales.allocate(
                       test, kScaleBytes + 1U,
                       "production C16 NVFP4 block scales");
  ready = ready && activations.allocate(
                       test, kTokens * kColumns + 1U,
                       "production C16 NVFP4 activations");
  ready = ready && companion_scales.allocate(
                       test, 2U, "production C16 NVFP4 companion scales");
  ready = ready && output.allocate(test, kTokens * kRows,
                                   "production C16 NVFP4 output");
  if (!ready) {
    return;
  }

  std::vector<std::uint16_t> host_activations(kTokens * kColumns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    std::fill_n(host_activations.begin() + token * kColumns, kColumns,
                kActivationValues[token]);
  }
  ready = test.cuda_ok(cudaMemset(packed_weights.get(), 0x22,
                                  kPackedBytes + 4U),
                       "initialize production C16 NVFP4 packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemset(block_scales.get(), 0x38, kScaleBytes + 1U),
                       "initialize production C16 NVFP4 block scales");
  ready = ready && upload(test, activations, host_activations,
                          "production C16 NVFP4 activations");
  ready = ready && upload(
                       test, companion_scales,
                       std::vector<float>{kWeightScale2, 1.0F},
                       "production C16 NVFP4 companion scales");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight nvfp4 = runtime::NvFp4LinearWeight{
      packed_weights.get(), block_scales.get(), companion_scales.get(),
      companion_scales.get() + 1U, kWeightScale2, 1.0F, kRows, kColumns};
  int status = runtime::launch_projection_tile_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, nvfp4, activations.get(),
      kTokens, nullptr, 0U, output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "SM87 NVFP4 production-shape C16 dispatch succeeds");
  if (static_cast<cudaError_t>(status) == cudaSuccess) {
    (void)expect_tile_output(
        test, output, kTokens, kRows,
        std::vector<std::uint16_t>(kExpected.begin(), kExpected.end()),
        "SM87 NVFP4 production-shape C16 dispatch");
  }
  const std::size_t production_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, nvfp4,
            activations.get(), kTokens, nullptr, 0U, output.get(),
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 production-shape C16 dispatch graph");
  test.expect(production_kernel_nodes == 1U,
              "SM87 NVFP4 production C16 dispatch uses one WMMA kernel");

  const auto run_fallback = [&](const std::uint8_t* const packed,
                                const std::uint8_t* const scales,
                                const std::uint16_t* const input,
                                const std::size_t expected_nodes,
                                const std::string& label) {
    const std::size_t nodes = captured_kernel_node_count(
        test,
        [&](cudaStream_t stream) noexcept {
          return q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              packed, scales, kWeightScale2, input, kRows, kColumns,
              output.get(), static_cast<void*>(stream));
        },
        label + " graph");
    test.expect(nodes == expected_nodes,
                label + " uses the expected safe fallback kernel count");
    const int fallback_status =
        q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
            packed, scales, kWeightScale2, input, kRows, kColumns,
            output.get());
    test.expect(static_cast<cudaError_t>(fallback_status) == cudaSuccess,
                label + " launch succeeds");
    if (static_cast<cudaError_t>(fallback_status) == cudaSuccess) {
      (void)expect_tile_output(
          test, output, kTokens, kRows,
          std::vector<std::uint16_t>(kExpected.begin(), kExpected.end()),
          label);
    }
  };

  run_fallback(packed_weights.get() + 4U, block_scales.get(),
               activations.get(), 2U,
               "4-byte but non-16-byte NVFP4 packed-weight fallback");
  run_fallback(packed_weights.get(), block_scales.get() + 1U,
               activations.get(), 2U,
               "byte-aligned NVFP4 block-scale fallback");

  ready = test.cuda_ok(
      cudaMemcpy(activations.get() + 1U, host_activations.data(),
                 host_activations.size() * sizeof(host_activations.front()),
                 cudaMemcpyHostToDevice),
      "upload 2-byte-aligned production C16 NVFP4 activations");
  if (!ready) {
    return;
  }
  run_fallback(packed_weights.get(), block_scales.get(),
               activations.get() + 1U, kTokens,
               "2-byte but non-8-byte NVFP4 activation fallback");

  const auto capture_generic_shape = [&](const std::size_t rows,
                                         const std::size_t columns,
                                         const std::string& label) {
    const auto* const packed =
        reinterpret_cast<const std::uint8_t*>(0x10'0000'0000ULL);
    const auto* const scales =
        reinterpret_cast<const std::uint8_t*>(0x20'0000'0000ULL);
    const auto* const input =
        reinterpret_cast<const std::uint16_t*>(0x30'0000'0000ULL);
    auto* const fake_output =
        reinterpret_cast<std::uint16_t*>(0x40'0000'0000ULL);
    return captured_kernel_node_count(
        test,
        [&](cudaStream_t stream) noexcept {
          return q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              packed, scales, 1.0F, input, rows, columns, fake_output,
              static_cast<void*>(stream));
        },
        label);
  };
  test.expect(capture_generic_shape(
                  1'024U, 5'120U,
                  "SM87 NVFP4 1024x5120 public M16 fallback graph") == 2U,
              "NVFP4 1024x5120 public M16 uses exactly two M8 kernels");
  test.expect(capture_generic_shape(
                  248'320U, 5'120U,
                  "SM87 NVFP4 lm_head public M16 fallback graph") == 2U,
              "NVFP4 lm_head public M16 does not select the WMMA kernel");
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
  test_bf16_direct_production_dispatch(test);
  test_tile_routes(test);
  test_bf16_projection_pair_dispatch(test);
  test_fp8_projection_pair_dispatch(test);
  test_fp8_qkv_z_projection_pair_dispatch(test);
  test_fp8_m16_production_dispatch(test);
  test_nvfp4_m16_production_dispatch(test);
  test_tile_validation(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " projection dispatch assertion(s) failed\n";
    return 1;
  }
  std::cout << "projection backend dispatch tests passed\n";
  return 0;
}
