#include "q3x/runtime/model_weights.h"

#include "q3x/kernels/reference_gemv.h"
#include "q3x/kernels/sm87_weight_only_gemv.h"
#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace q3x::kernels {

// Test-only direct controls used to lock the promoted large-N M128 route and
// the historical M64 K/V route without extending the public kernel header.
[[nodiscard]] int
launch_sm87_fp8_w8a16_whole_chunk_large_n_m128_b_reuse_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_whole_chunk_z_m128_cp_async_canonical_xor_register_feed_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_whole_chunk_m64_historical_control_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels

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

template <std::size_t OutputCount, typename T, std::size_t InputCount>
[[nodiscard]] constexpr std::array<T, OutputCount> repeat_array(
    const std::array<T, InputCount>& input) noexcept {
  static_assert(InputCount != 0U);
  std::array<T, OutputCount> output{};
  for (std::size_t index = 0U; index < OutputCount; ++index) {
    output[index] = input[index % InputCount];
  }
  return output;
}

template <typename Launch>
[[nodiscard]] std::size_t captured_kernel_node_count(
    TestContext& test, Launch&& launch, const std::string& label,
    std::size_t* const total_node_count = nullptr,
    bool* const linear_kernel_chain = nullptr) {
  constexpr std::size_t kInvalidCount =
      std::numeric_limits<std::size_t>::max();
  if (total_node_count != nullptr) {
    *total_node_count = kInvalidCount;
  }
  if (linear_kernel_chain != nullptr) {
    *linear_kernel_chain = false;
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
      if (ready && linear_kernel_chain != nullptr && node_count != 0U &&
          kernel_nodes == node_count) {
        std::size_t edge_count = 0U;
#if CUDART_VERSION >= 12030
        ready = test.cuda_ok(
            cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edge_count),
            "count graph edges " + label);
#else
        ready = test.cuda_ok(
            cudaGraphGetEdges(graph, nullptr, nullptr, &edge_count),
            "count graph edges " + label);
#endif
        std::vector<cudaGraphNode_t> from(edge_count);
        std::vector<cudaGraphNode_t> to(edge_count);
        if (ready && edge_count != 0U) {
#if CUDART_VERSION >= 12030
          ready = test.cuda_ok(
              cudaGraphGetEdges(graph, from.data(), to.data(), nullptr,
                                &edge_count),
              "read graph edges " + label);
#else
          ready = test.cuda_ok(
              cudaGraphGetEdges(graph, from.data(), to.data(), &edge_count),
              "read graph edges " + label);
#endif
        }
        std::vector<std::size_t> indegree(node_count, 0U);
        std::vector<std::size_t> outdegree(node_count, 0U);
        bool known_edges = ready;
        for (std::size_t edge = 0U; edge < edge_count && known_edges;
             ++edge) {
          const auto from_position =
              std::find(nodes.begin(), nodes.end(), from[edge]);
          const auto to_position =
              std::find(nodes.begin(), nodes.end(), to[edge]);
          known_edges = from_position != nodes.end() &&
                        to_position != nodes.end();
          if (known_edges) {
            ++outdegree[static_cast<std::size_t>(from_position -
                                                 nodes.begin())];
            ++indegree[static_cast<std::size_t>(to_position -
                                                nodes.begin())];
          }
        }
        std::size_t sources = 0U;
        std::size_t sinks = 0U;
        bool bounded_degrees = known_edges;
        for (std::size_t node = 0U; node < node_count; ++node) {
          sources += indegree[node] == 0U ? 1U : 0U;
          sinks += outdegree[node] == 0U ? 1U : 0U;
          bounded_degrees = bounded_degrees && indegree[node] <= 1U &&
                            outdegree[node] <= 1U;
        }
        *linear_kernel_chain =
            bounded_degrees && edge_count + 1U == node_count &&
            sources == 1U && sinks == 1U;
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

struct CapturedKernelLaunch {
  void* function = nullptr;
  dim3 grid{};
  dim3 block{};
  unsigned int dynamic_shared_bytes = 0U;
};

struct CapturedKernelChain {
  bool valid = false;
  std::vector<CapturedKernelLaunch> launches;
};

template <typename Launch>
[[nodiscard]] CapturedKernelChain capture_ordered_kernel_chain(
    TestContext& test, Launch&& launch, const std::string& label) {
  CapturedKernelChain result;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create ordered capture stream " + label);
  ready = ready && test.cuda_ok(
                       cudaStreamBeginCapture(stream,
                                              cudaStreamCaptureModeGlobal),
                       "begin ordered capture " + label);
  if (ready) {
    const bool launch_ready =
        static_cast<cudaError_t>(launch(stream)) == cudaSuccess;
    test.expect(launch_ready, label + " ordered capture launch succeeds");
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         "end ordered capture " + label) &&
            launch_ready;
  }

  std::vector<cudaGraphNode_t> nodes;
  std::vector<cudaKernelNodeParams> parameters;
  if (ready && graph != nullptr) {
    std::size_t node_count = 0U;
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                         "count ordered graph nodes " + label);
    nodes.resize(node_count);
    parameters.resize(node_count);
    if (ready && node_count != 0U) {
      ready = test.cuda_ok(cudaGraphGetNodes(graph, nodes.data(), &node_count),
                           "read ordered graph nodes " + label);
    }
    for (std::size_t index = 0U; ready && index < node_count; ++index) {
      cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
      ready = test.cuda_ok(cudaGraphNodeGetType(nodes[index], &type),
                           "read ordered graph node type " + label);
      test.expect(type == cudaGraphNodeTypeKernel,
                  label + " ordered graph contains only kernels");
      ready = ready && type == cudaGraphNodeTypeKernel &&
              test.cuda_ok(cudaGraphKernelNodeGetParams(
                               nodes[index], &parameters[index]),
                           "read ordered kernel parameters " + label);
    }

    std::size_t edge_count = 0U;
    if (ready) {
#if CUDART_VERSION >= 12030
      ready = test.cuda_ok(
          cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edge_count),
          "count ordered graph edges " + label);
#else
      ready = test.cuda_ok(
          cudaGraphGetEdges(graph, nullptr, nullptr, &edge_count),
          "count ordered graph edges " + label);
#endif
    }
    std::vector<cudaGraphNode_t> from(edge_count);
    std::vector<cudaGraphNode_t> to(edge_count);
    if (ready && edge_count != 0U) {
#if CUDART_VERSION >= 12030
      ready = test.cuda_ok(
          cudaGraphGetEdges(graph, from.data(), to.data(), nullptr,
                            &edge_count),
          "read ordered graph edges " + label);
#else
      ready = test.cuda_ok(
          cudaGraphGetEdges(graph, from.data(), to.data(), &edge_count),
          "read ordered graph edges " + label);
#endif
    }

    std::vector<std::size_t> indegree(node_count, 0U);
    std::vector<std::size_t> successor(
        node_count, std::numeric_limits<std::size_t>::max());
    for (std::size_t edge = 0U; ready && edge < edge_count; ++edge) {
      const auto from_position =
          std::find(nodes.begin(), nodes.end(), from[edge]);
      const auto to_position = std::find(nodes.begin(), nodes.end(), to[edge]);
      ready = from_position != nodes.end() && to_position != nodes.end();
      if (ready) {
        const std::size_t from_index = static_cast<std::size_t>(
            from_position - nodes.begin());
        const std::size_t to_index =
            static_cast<std::size_t>(to_position - nodes.begin());
        ready = successor[from_index] ==
                std::numeric_limits<std::size_t>::max();
        successor[from_index] = to_index;
        ++indegree[to_index];
        ready = ready && indegree[to_index] == 1U;
      }
    }
    std::size_t source = std::numeric_limits<std::size_t>::max();
    std::size_t source_count = 0U;
    for (std::size_t index = 0U; index < node_count; ++index) {
      if (indegree[index] == 0U) {
        source = index;
        ++source_count;
      }
    }
    ready = ready && node_count != 0U && source_count == 1U &&
            edge_count + 1U == node_count;
    for (std::size_t visited = 0U; ready && visited < node_count; ++visited) {
      const cudaKernelNodeParams& selected = parameters[source];
      result.launches.push_back(
          {selected.func, selected.gridDim, selected.blockDim,
           selected.sharedMemBytes});
      if (visited + 1U < node_count) {
        source = successor[source];
        ready = source != std::numeric_limits<std::size_t>::max();
      }
    }
  }
  result.valid = ready;
  if (graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(graph),
                       "destroy ordered graph " + label);
  }
  if (stream != nullptr) {
    (void)test.cuda_ok(cudaStreamDestroy(stream),
                       "destroy ordered capture stream " + label);
  }
  return result;
}

template <typename Launch>
void expect_failed_capture_has_no_nodes(TestContext& test, Launch&& launch,
                                        const cudaError_t expected_status,
                                        const std::string& label) {
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create invalid capture stream " + label);
  ready = ready && test.cuda_ok(
                       cudaStreamBeginCapture(stream,
                                              cudaStreamCaptureModeGlobal),
                       "begin invalid capture " + label);
  if (ready) {
    test.expect(static_cast<cudaError_t>(launch(stream)) == expected_status,
                label + " returns " + cudaGetErrorName(expected_status));
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         "end invalid capture " + label);
  }

  if (ready && graph != nullptr) {
    std::size_t node_count = std::numeric_limits<std::size_t>::max();
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                         "count invalid capture nodes " + label);
    if (ready) {
      test.expect(node_count == 0U, label + " enqueues zero graph nodes");
    }
  } else if (ready) {
    test.expect(false, label + " produces an empty CUDA graph");
  }
  if (graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(graph),
                       "destroy invalid capture graph " + label);
  }
  if (stream != nullptr) {
    (void)test.cuda_ok(cudaStreamDestroy(stream),
                       "destroy invalid capture stream " + label);
  }
}

template <typename Launch>
void expect_invalid_capture_has_no_nodes(TestContext& test, Launch&& launch,
                                         const std::string& label) {
  expect_failed_capture_has_no_nodes(test, std::forward<Launch>(launch),
                                     cudaErrorInvalidValue, label);
}

template <typename Launch>
void expect_not_supported_capture_has_no_nodes(TestContext& test,
                                               Launch&& launch,
                                               const std::string& label) {
  expect_failed_capture_has_no_nodes(test, std::forward<Launch>(launch),
                                     cudaErrorNotSupported, label);
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

void test_fp8_m1_output_sidecar_dispatch(TestContext& test) {
  constexpr std::size_t kRows = runtime::kFp8M1OutputProjectionRows;
  constexpr std::size_t kColumns =
      runtime::kFp8M1OutputProjectionColumns;
  constexpr std::size_t kM2 = 2U;
  constexpr std::size_t kM32 = 32U;

  const auto* const canonical_weight =
      reinterpret_cast<const std::uint8_t*>(0x10'0000'0000ULL);
  const auto* const sidecar_weight =
      reinterpret_cast<const std::uint8_t*>(0x20'0000'0000ULL);
  const auto* const companion_scales =
      reinterpret_cast<const float*>(0x40'0000'0000ULL);
  const auto* const activation =
      reinterpret_cast<const std::uint16_t*>(0x50'0000'0000ULL);
  auto* const output =
      reinterpret_cast<std::uint16_t*>(0x60'0000'0000ULL);

  runtime::Fp8LinearWeight canonical_payload{
      canonical_weight, companion_scales, companion_scales + 1U,
      1.0F, 1.0F, kRows, kColumns};
  runtime::Fp8LinearWeight sidecar_payload = canonical_payload;
  sidecar_payload.m1_aosoa4_preswizzled_weight = sidecar_weight;
  const runtime::LinearWeight canonical = canonical_payload;
  const runtime::LinearWeight sidecar = sidecar_payload;

  const CapturedKernelChain canonical_oracle = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
            canonical_weight, 1.0F, activation, kRows, kColumns, output,
            static_cast<void*>(stream));
      },
      "FP8 canonical M1 output identity oracle");
  const CapturedKernelChain sidecar_oracle = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return q3x::kernels::
            launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_bf16_cuda(
                sidecar_weight, 1.0F, activation, kRows, kColumns, output,
                static_cast<void*>(stream));
      },
      "FP8 AoSoA4 M1 output identity oracle");
  const auto capture_dispatch =
      [&](const runtime::LinearWeight& weight,
          const std::size_t token_count, const std::string& label) {
        return capture_ordered_kernel_chain(
            test,
            [&](cudaStream_t stream) noexcept {
              return runtime::launch_projection_tile_to_bf16_cuda(
                  runtime::ProjectionBackend::kSm87WeightOnly, weight,
                  activation, token_count, nullptr, 0U, output,
                  static_cast<void*>(stream));
            },
            label);
      };
  const CapturedKernelChain canonical_m1 = capture_dispatch(
      canonical, 1U, "SM87 FP8 canonical M1 output dispatch graph");
  const CapturedKernelChain sidecar_m1 = capture_dispatch(
      sidecar, 1U, "SM87 FP8 AoSoA4 M1 output dispatch graph");

  const auto has_one_kernel = [](const CapturedKernelChain& chain) noexcept {
    return chain.valid && chain.launches.size() == 1U;
  };
  test.expect(has_one_kernel(canonical_oracle),
              "FP8 canonical M1 output oracle is one kernel");
  test.expect(has_one_kernel(sidecar_oracle),
              "FP8 AoSoA4 M1 output oracle is one kernel");
  test.expect(has_one_kernel(canonical_m1),
              "FP8 null-sidecar M1 dispatch is one kernel");
  test.expect(has_one_kernel(sidecar_m1),
              "FP8 populated-sidecar M1 dispatch is one kernel");
  if (has_one_kernel(canonical_oracle) && has_one_kernel(sidecar_oracle) &&
      has_one_kernel(canonical_m1) && has_one_kernel(sidecar_m1)) {
    const void* const canonical_function =
        canonical_oracle.launches.front().function;
    const void* const sidecar_function = sidecar_oracle.launches.front().function;
    test.expect(canonical_function != nullptr && sidecar_function != nullptr &&
                    canonical_function != sidecar_function,
                "FP8 AoSoA4 M1 output kernel is distinct from canonical M1");
    test.expect(canonical_m1.launches.front().function == canonical_function,
                "FP8 null sidecar preserves canonical M1 dispatch");
    test.expect(sidecar_m1.launches.front().function == sidecar_function,
                "FP8 populated sidecar selects the AoSoA4 M1 dispatch");
  }

  const auto expect_same_dispatch =
      [&](const CapturedKernelChain& first,
          const CapturedKernelChain& second, const std::string& label) {
        test.expect(first.valid && second.valid,
                    label + " captures valid kernel chains");
        test.expect(first.launches.size() == second.launches.size(),
                    label + " preserves kernel count");
        const std::size_t compared =
            std::min(first.launches.size(), second.launches.size());
        for (std::size_t index = 0U; index < compared; ++index) {
          const CapturedKernelLaunch& left = first.launches[index];
          const CapturedKernelLaunch& right = second.launches[index];
          test.expect(left.function == right.function &&
                          left.grid.x == right.grid.x &&
                          left.grid.y == right.grid.y &&
                          left.grid.z == right.grid.z &&
                          left.block.x == right.block.x &&
                          left.block.y == right.block.y &&
                          left.block.z == right.block.z &&
                          left.dynamic_shared_bytes ==
                              right.dynamic_shared_bytes,
                      label + " preserves launch " +
                          std::to_string(index));
        }
      };
  expect_same_dispatch(
      capture_dispatch(canonical, kM2,
                       "SM87 FP8 canonical M2 output dispatch graph"),
      capture_dispatch(sidecar, kM2,
                       "SM87 FP8 sidecar-present M2 output dispatch graph"),
      "FP8 M2 ignores the M1-only sidecar");
  expect_same_dispatch(
      capture_dispatch(canonical, kM32,
                       "SM87 FP8 canonical M32 output dispatch graph"),
      capture_dispatch(sidecar, kM32,
                       "SM87 FP8 sidecar-present M32 output dispatch graph"),
      "FP8 M32 ignores the M1-only sidecar");

  runtime::Fp8LinearWeight near_miss_canonical_payload = canonical_payload;
  near_miss_canonical_payload.output_size = kRows - 1U;
  runtime::Fp8LinearWeight near_miss_sidecar_payload =
      near_miss_canonical_payload;
  near_miss_sidecar_payload.m1_aosoa4_preswizzled_weight = sidecar_weight;
  expect_same_dispatch(
      capture_dispatch(runtime::LinearWeight{near_miss_canonical_payload}, 1U,
                       "SM87 FP8 near-miss canonical M1 dispatch graph"),
      capture_dispatch(runtime::LinearWeight{near_miss_sidecar_payload}, 1U,
                       "SM87 FP8 near-miss sidecar M1 dispatch graph"),
      "FP8 near-miss M1 ignores the exact-shape sidecar");
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
  constexpr std::size_t kMaximumTokens =
      runtime::kMaximumProjectionTileTokenCount;
  constexpr std::uint16_t kBf16One = 0x3f80U;
  constexpr auto kActivationValues = repeat_array<kMaximumTokens>(
      std::array<std::uint16_t, 16U>{
      0x3f80U, 0x3f00U, 0xbf80U, 0x4000U,
      0x3e80U, 0xbf00U, 0x4080U, 0xc000U,
      0x3f40U, 0xbf40U, 0x4040U, 0xc040U,
      0x3fc0U, 0xbfc0U, 0x4100U, 0xc100U});
  constexpr auto kFp8Expected = repeat_array<kMaximumTokens>(
      std::array<std::uint16_t, 16U>{
      0x4480U, 0x4400U, 0xc480U, 0x4500U,
      0x4380U, 0xc400U, 0x4580U, 0xc500U,
      0x4440U, 0xc440U, 0x4540U, 0xc540U,
      0x44c0U, 0xc4c0U, 0x4600U, 0xc600U});
  constexpr auto kNvFp4Expected = repeat_array<kMaximumTokens>(
      std::array<std::uint16_t, 16U>{
      0x4380U, 0x4300U, 0xc380U, 0x4400U,
      0x4280U, 0xc300U, 0x4480U, 0xc400U,
      0x4340U, 0xc340U, 0x4440U, 0xc440U,
      0x43c0U, 0xc3c0U, 0x4500U, 0xc500U});
  constexpr auto kBf16Expected = repeat_array<kMaximumTokens>(
      std::array<std::uint16_t, 16U>{
      0x4080U, 0x4000U, 0xc080U, 0x4100U,
      0x3f80U, 0xc000U, 0x4180U, 0xc100U,
      0x4040U, 0xc040U, 0x4140U, 0xc140U,
      0x40c0U, 0xc0c0U, 0x4200U, 0xc200U});

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
    if (token_count > 16U && token_count != 17U && token_count != 18U &&
        token_count != 24U && token_count != 31U && token_count != 32U &&
        token_count != 33U && token_count != 50U && token_count != 63U &&
        token_count != 64U) {
      continue;
    }
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
                               const std::string& label,
                               bool* const linear_chain = nullptr) {
    std::size_t total_nodes = 0U;
    return captured_kernel_node_count(
        test,
        [&](cudaStream_t stream) noexcept {
          return runtime::launch_projection_tile_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly, fp8,
              fp8_activation.get(), token_count, nullptr, 0U, output.get(),
              static_cast<void*>(stream));
        },
        label, &total_nodes, linear_chain);
  };
  test.expect(capture_fp8(9U, "SM87 FP8 M9 dispatch graph") == 2U,
              "SM87 FP8 M9 remains M8+M1");
  test.expect(capture_fp8(15U, "SM87 FP8 M15 dispatch graph") == 2U,
              "SM87 FP8 M15 remains M8+M7");
  test.expect(capture_fp8(16U, "SM87 FP8 generic M16 dispatch graph") == 2U,
              "SM87 FP8 generic M16 uses the public two-M8 fallback");
  test.expect(capture_fp8(17U, "SM87 FP8 generic M17 dispatch graph") == 3U,
              "SM87 FP8 generic M17 uses M16+M1");
  test.expect(capture_fp8(18U, "SM87 FP8 generic M18 dispatch graph") == 3U,
              "SM87 FP8 generic M18 uses M16+M2");
  test.expect(capture_fp8(24U, "SM87 FP8 generic M24 dispatch graph") == 3U,
              "SM87 FP8 generic M24 uses M16+M8");
  test.expect(capture_fp8(31U, "SM87 FP8 generic M31 dispatch graph") == 4U,
              "SM87 FP8 generic M31 uses M16+M8+M7");
  bool fp8_m32_linear_chain = false;
  test.expect(capture_fp8(32U, "SM87 FP8 generic M32 dispatch graph",
                          &fp8_m32_linear_chain) == 4U,
              "SM87 FP8 generic M32 uses two public M16 fallbacks");
  test.expect(fp8_m32_linear_chain,
              "SM87 FP8 generic M32 preserves one ordered kernel chain");
  bool fp8_m33_linear_chain = false;
  test.expect(capture_fp8(33U, "SM87 FP8 generic M33 dispatch graph",
                          &fp8_m33_linear_chain) == 5U,
              "SM87 FP8 generic M33 preserves M32+M1 scheduling");
  test.expect(fp8_m33_linear_chain,
              "SM87 FP8 generic M33 preserves one ordered kernel chain");
  bool fp8_m64_linear_chain = false;
  test.expect(capture_fp8(64U, "SM87 FP8 generic M64 dispatch graph",
                          &fp8_m64_linear_chain) == 8U,
              "SM87 FP8 generic M64 preserves two M32 schedules");
  test.expect(fp8_m64_linear_chain,
              "SM87 FP8 generic M64 preserves one ordered kernel chain");
  const auto capture_nvfp4 = [&](const std::size_t token_count,
                                 const std::string& label,
                                 bool* const linear_chain = nullptr) {
    std::size_t total_nodes = 0U;
    return captured_kernel_node_count(
        test,
        [&](cudaStream_t stream) noexcept {
          return runtime::launch_projection_tile_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly, nvfp4,
              nvfp4_activation.get(), token_count, nullptr, 0U, output.get(),
              static_cast<void*>(stream));
        },
        label, &total_nodes, linear_chain);
  };
  test.expect(capture_nvfp4(9U, "SM87 NVFP4 M9 dispatch graph") == 2U,
              "SM87 NVFP4 M9 remains M8+M1");
  test.expect(capture_nvfp4(15U, "SM87 NVFP4 M15 dispatch graph") == 2U,
              "SM87 NVFP4 M15 remains M8+M7");
  test.expect(
      capture_nvfp4(16U, "SM87 NVFP4 generic M16 dispatch graph") == 2U,
      "SM87 NVFP4 generic M16 uses the public two-M8 fallback");
  test.expect(
      capture_nvfp4(17U, "SM87 NVFP4 generic M17 dispatch graph") == 3U,
      "SM87 NVFP4 generic M17 uses M16+M1");
  test.expect(
      capture_nvfp4(18U, "SM87 NVFP4 generic M18 dispatch graph") == 3U,
      "SM87 NVFP4 generic M18 uses M16+M2");
  test.expect(
      capture_nvfp4(24U, "SM87 NVFP4 generic M24 dispatch graph") == 3U,
      "SM87 NVFP4 generic M24 uses M16+M8");
  test.expect(
      capture_nvfp4(31U, "SM87 NVFP4 generic M31 dispatch graph") == 4U,
      "SM87 NVFP4 generic M31 uses M16+M8+M7");
  bool nvfp4_m32_linear_chain = false;
  test.expect(capture_nvfp4(32U, "SM87 NVFP4 generic M32 dispatch graph",
                            &nvfp4_m32_linear_chain) == 4U,
              "SM87 NVFP4 generic M32 uses two public M16 fallbacks");
  test.expect(nvfp4_m32_linear_chain,
              "SM87 NVFP4 generic M32 preserves one ordered kernel chain");
  bool nvfp4_m33_linear_chain = false;
  test.expect(capture_nvfp4(33U, "SM87 NVFP4 generic M33 dispatch graph",
                            &nvfp4_m33_linear_chain) == 5U,
              "SM87 NVFP4 generic M33 preserves M32+M1 scheduling");
  test.expect(nvfp4_m33_linear_chain,
              "SM87 NVFP4 generic M33 preserves one ordered kernel chain");
  bool nvfp4_m64_linear_chain = false;
  test.expect(capture_nvfp4(64U, "SM87 NVFP4 generic M64 dispatch graph",
                            &nvfp4_m64_linear_chain) == 8U,
              "SM87 NVFP4 generic M64 preserves two M32 schedules");
  test.expect(nvfp4_m64_linear_chain,
              "SM87 NVFP4 generic M64 preserves one ordered kernel chain");

  const auto* const production_fp8_weight =
      reinterpret_cast<const std::uint8_t*>(0x10'0000'0000ULL);
  const auto* const production_nvfp4_weight =
      reinterpret_cast<const std::uint8_t*>(0x20'0000'0000ULL);
  const auto* const production_nvfp4_second_weight =
      reinterpret_cast<const std::uint8_t*>(0x21'0000'0000ULL);
  const auto* const production_nvfp4_scale =
      reinterpret_cast<const std::uint8_t*>(0x30'0000'0000ULL);
  const auto* const production_nvfp4_second_scale =
      reinterpret_cast<const std::uint8_t*>(0x31'0000'0000ULL);
  const auto* const production_companion_scales =
      reinterpret_cast<const float*>(0x40'0000'0000ULL);
  const auto* const production_input =
      reinterpret_cast<const std::uint16_t*>(0x50'0000'0000ULL);
  auto* const production_output =
      reinterpret_cast<std::uint16_t*>(0x60'0000'0000ULL);
  auto* const production_second_output =
      reinterpret_cast<std::uint16_t*>(0x70'0000'0000ULL);
  const runtime::LinearWeight production_fp8 = runtime::Fp8LinearWeight{
      production_fp8_weight, production_companion_scales,
      production_companion_scales + 1U, 1.0F, 1.0F, 5'120U, 6'144U};
  const runtime::LinearWeight production_nvfp4 = runtime::NvFp4LinearWeight{
      production_nvfp4_weight, production_nvfp4_scale,
      production_companion_scales, production_companion_scales + 1U,
      1.0F, 1.0F, 17'408U, 5'120U};
  const runtime::LinearWeight production_nvfp4_second =
      runtime::NvFp4LinearWeight{
          production_nvfp4_second_weight, production_nvfp4_second_scale,
          production_companion_scales + 2U,
          production_companion_scales + 3U, 1.0F, 1.0F, 17'408U, 5'120U};
  const runtime::LinearWeight production_nvfp4_down =
      runtime::NvFp4LinearWeight{
          production_nvfp4_weight, production_nvfp4_scale,
          production_companion_scales, production_companion_scales + 1U,
          1.0F, 1.0F, 5'120U, 17'408U};

  const auto capture_production_nvfp4_tile =
      [&](const runtime::LinearWeight& weight,
          const std::uint16_t* const input, const std::size_t token_count,
          std::uint16_t* const tile_output, const std::string& label,
          bool* const linear_chain) {
        std::size_t total_nodes = 0U;
        return captured_kernel_node_count(
            test,
            [&](cudaStream_t stream) noexcept {
              return runtime::launch_projection_tile_to_bf16_cuda(
                  runtime::ProjectionBackend::kSm87WeightOnly, weight, input,
                  token_count, nullptr, 0U, tile_output,
                  static_cast<void*>(stream));
            },
            label, &total_nodes, linear_chain);
      };
  const auto capture_production_nvfp4_pair =
      [&](const runtime::LinearWeight& first_weight,
          const runtime::LinearWeight& second_weight,
          const std::uint16_t* const input, const std::size_t token_count,
          std::uint16_t* const first_output,
          std::uint16_t* const second_output, const std::string& label,
          bool* const linear_chain) {
        std::size_t total_nodes = 0U;
        return captured_kernel_node_count(
            test,
            [&](cudaStream_t stream) noexcept {
              return runtime::launch_projection_pair_tile_to_bf16_cuda(
                  runtime::ProjectionBackend::kSm87WeightOnly, first_weight,
                  second_weight, input, token_count, nullptr, 0U,
                  first_output, second_output, static_cast<void*>(stream));
            },
            label, &total_nodes, linear_chain);
      };

  bool production_nvfp4_down_m64_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_down, production_input, 64U, production_output,
          "SM87 NVFP4 production down M64 dispatch graph",
          &production_nvfp4_down_m64_linear_chain) == 1U,
      "SM87 NVFP4 production down M64 uses one exact weight-reuse kernel");
  test.expect(production_nvfp4_down_m64_linear_chain,
              "SM87 NVFP4 production down M64 is a single-node chain");

  bool production_nvfp4_gate_m64_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4, production_input, 64U, production_output,
          "SM87 NVFP4 production gate/up M64 dispatch graph",
          &production_nvfp4_gate_m64_linear_chain) == 2U,
      "SM87 NVFP4 production gate/up M64 preserves two direct M32 kernels");
  test.expect(production_nvfp4_gate_m64_linear_chain,
              "SM87 NVFP4 production gate/up M64 remains ordered");

  const CapturedKernelChain production_fp8_m64 =
      capture_ordered_kernel_chain(
          test,
          [&](cudaStream_t stream) noexcept {
            return runtime::launch_projection_tile_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly,
                production_fp8, production_input, 64U, nullptr, 0U,
                production_output, static_cast<void*>(stream));
          },
          "SM87 FP8 production M64 dispatch graph");
  const CapturedKernelChain production_fp8_m64_direct =
      capture_ordered_kernel_chain(
          test,
          [&](cudaStream_t stream) noexcept {
            return q3x::kernels::
                launch_sm87_fp8_w8a16_m64_attention_output_gemm_bf16_cuda(
                    production_fp8_weight, 1.0F, production_input, 5'120U,
                    6'144U, production_output, static_cast<void*>(stream));
          },
          "SM87 FP8 direct public M64 graph");
  const bool production_fp8_m64_topology =
      production_fp8_m64.valid && production_fp8_m64.launches.size() == 1U &&
      production_fp8_m64_direct.valid &&
      production_fp8_m64_direct.launches.size() == 1U &&
      production_fp8_m64.launches.front().function != nullptr &&
      production_fp8_m64.launches.front().function ==
          production_fp8_m64_direct.launches.front().function &&
      production_fp8_m64.launches.front().grid.x == 40U &&
      production_fp8_m64.launches.front().grid.y == 1U &&
      production_fp8_m64.launches.front().grid.z == 1U &&
      production_fp8_m64.launches.front().block.x == 256U &&
      production_fp8_m64.launches.front().block.y == 1U &&
      production_fp8_m64.launches.front().block.z == 1U &&
      production_fp8_m64.launches.front().dynamic_shared_bytes == 0U &&
      production_fp8_m64_direct.launches.front().grid.x == 40U &&
      production_fp8_m64_direct.launches.front().block.x == 256U &&
      production_fp8_m64_direct.launches.front().dynamic_shared_bytes == 0U;
  test.expect(production_fp8_m64_topology,
              "SM87 FP8 production M64 dispatch matches one direct public "
              "grid40 block256 kernel");

  const runtime::LinearWeight production_fp8_m64_near_miss =
      runtime::Fp8LinearWeight{
          production_fp8_weight, production_companion_scales,
          production_companion_scales + 1U, 1.0F, 1.0F, 5'119U, 6'144U};
  bool production_fp8_m64_near_miss_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_fp8_m64_near_miss, production_input, 64U,
          production_output, "SM87 FP8 near-miss M64 fallback graph",
          &production_fp8_m64_near_miss_linear_chain) == 8U,
      "SM87 FP8 near-miss M64 preserves two public M32 fallbacks");
  test.expect(production_fp8_m64_near_miss_linear_chain,
              "SM87 FP8 near-miss M64 fallback remains ordered");

  const runtime::LinearWeight production_fp8_m64_weight4 =
      runtime::Fp8LinearWeight{
          production_fp8_weight + 4U, production_companion_scales,
          production_companion_scales + 1U, 1.0F, 1.0F, 5'120U, 6'144U};
  bool production_fp8_m64_weight4_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_fp8_m64_weight4, production_input, 64U,
          production_output, "SM87 FP8 weight4 M64 fallback graph",
          &production_fp8_m64_weight4_linear_chain) == 8U,
      "SM87 FP8 non-16-byte weight M64 preserves two public M32 fallbacks");
  test.expect(production_fp8_m64_weight4_linear_chain,
              "SM87 FP8 weight-alignment M64 fallback remains ordered");

  bool production_nvfp4_down_m63_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_down, production_input, 63U, production_output,
          "SM87 NVFP4 production down M63 dispatch graph",
          &production_nvfp4_down_m63_linear_chain) == 2U,
      "SM87 NVFP4 production down M63 preserves direct M32+M31 kernels");
  test.expect(production_nvfp4_down_m63_linear_chain,
              "SM87 NVFP4 production down M63 remains ordered");

  bool production_nvfp4_m18_gate_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4, production_input, 18U, production_output,
          "SM87 NVFP4 production gate/up M18 dispatch graph",
          &production_nvfp4_m18_gate_linear_chain) == 1U,
      "SM87 NVFP4 production gate/up M18 uses one masked-M32 kernel");
  test.expect(production_nvfp4_m18_gate_linear_chain,
              "SM87 NVFP4 production gate/up M18 is a single-node chain");

  bool production_nvfp4_m18_down_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_down, production_input, 18U, production_output,
          "SM87 NVFP4 production down M18 dispatch graph",
          &production_nvfp4_m18_down_linear_chain) == 1U,
      "SM87 NVFP4 production down M18 uses one masked-M32 kernel");
  test.expect(production_nvfp4_m18_down_linear_chain,
              "SM87 NVFP4 production down M18 is a single-node chain");

  bool production_nvfp4_m17_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4, production_input, 17U, production_output,
          "SM87 NVFP4 production M17 dispatch graph",
          &production_nvfp4_m17_linear_chain) == 1U,
      "SM87 NVFP4 production M17 uses one runtime-masked M32 kernel");
  test.expect(production_nvfp4_m17_linear_chain,
              "SM87 NVFP4 production M17 is a single-node chain");

  bool production_nvfp4_m19_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4, production_input, 19U, production_output,
          "SM87 NVFP4 production M19 dispatch graph",
          &production_nvfp4_m19_linear_chain) == 1U,
      "SM87 NVFP4 production M19 uses one runtime-masked M32 kernel");
  test.expect(production_nvfp4_m19_linear_chain,
              "SM87 NVFP4 production M19 is a single-node chain");

  bool production_nvfp4_m25_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4, production_input, 25U, production_output,
          "SM87 NVFP4 production M25 dispatch graph",
          &production_nvfp4_m25_linear_chain) == 1U,
      "SM87 NVFP4 production M25 reduces three kernels to one");
  test.expect(production_nvfp4_m25_linear_chain,
              "SM87 NVFP4 production M25 is a single-node chain");

  bool production_nvfp4_m31_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_down, production_input, 31U, production_output,
          "SM87 NVFP4 production down M31 dispatch graph",
          &production_nvfp4_m31_linear_chain) == 1U,
      "SM87 NVFP4 production down M31 reduces three kernels to one");
  test.expect(production_nvfp4_m31_linear_chain,
              "SM87 NVFP4 production down M31 is a single-node chain");

  const runtime::LinearWeight production_nvfp4_m18_weight4 =
      runtime::NvFp4LinearWeight{
          reinterpret_cast<const std::uint8_t*>(
              reinterpret_cast<std::uintptr_t>(production_nvfp4_weight) +
              4U),
          production_nvfp4_scale, production_companion_scales,
          production_companion_scales + 1U, 1.0F, 1.0F, 17'408U, 5'120U};
  bool production_nvfp4_m18_weight4_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_m18_weight4, production_input, 18U,
          production_output,
          "SM87 NVFP4 4-byte-aligned M18 fallback graph",
          &production_nvfp4_m18_weight4_linear_chain) == 3U,
      "SM87 NVFP4 4-byte-aligned M18 uses two M8 kernels plus M2");
  test.expect(production_nvfp4_m18_weight4_linear_chain,
              "SM87 NVFP4 4-byte-aligned M18 fallback remains ordered");

  const runtime::LinearWeight production_nvfp4_m18_scale1 =
      runtime::NvFp4LinearWeight{
          production_nvfp4_weight, production_nvfp4_scale + 1U,
          production_companion_scales, production_companion_scales + 1U,
          1.0F, 1.0F, 17'408U, 5'120U};
  bool production_nvfp4_m18_scale1_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_m18_scale1, production_input, 18U,
          production_output, "SM87 NVFP4 byte-aligned scale M18 fallback graph",
          &production_nvfp4_m18_scale1_linear_chain) == 3U,
      "SM87 NVFP4 byte-aligned scale M18 uses two M8 kernels plus M2");
  test.expect(production_nvfp4_m18_scale1_linear_chain,
              "SM87 NVFP4 byte-aligned scale M18 fallback remains ordered");

  const auto* const production_input2 =
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<std::uintptr_t>(production_input) + 2U);
  bool production_nvfp4_m18_input2_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4, production_input2, 18U, production_output,
          "SM87 NVFP4 2-byte-aligned input M18 fallback graph",
          &production_nvfp4_m18_input2_linear_chain) == 18U,
      "SM87 NVFP4 2-byte-aligned input M18 preserves scalar token order");
  test.expect(production_nvfp4_m18_input2_linear_chain,
              "SM87 NVFP4 2-byte-aligned input M18 fallback remains ordered");

  const runtime::LinearWeight production_nvfp4_m18_near_miss =
      runtime::NvFp4LinearWeight{
          production_nvfp4_weight, production_nvfp4_scale,
          production_companion_scales, production_companion_scales + 1U,
          1.0F, 1.0F, 17'407U, 5'120U};
  bool production_nvfp4_m18_near_miss_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_m18_near_miss, production_input, 18U,
          production_output, "SM87 NVFP4 near-miss M18 fallback graph",
          &production_nvfp4_m18_near_miss_linear_chain) == 3U,
      "SM87 NVFP4 near-miss M18 uses two M8 kernels plus M2");
  test.expect(production_nvfp4_m18_near_miss_linear_chain,
              "SM87 NVFP4 near-miss M18 fallback remains ordered");

  const runtime::LinearWeight production_nvfp4_down_near_miss =
      runtime::NvFp4LinearWeight{
          production_nvfp4_weight, production_nvfp4_scale,
          production_companion_scales, production_companion_scales + 1U,
          1.0F, 1.0F, 5'119U, 17'408U};
  bool production_nvfp4_down_near_miss_m64_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_down_near_miss, production_input, 64U,
          production_output,
          "SM87 NVFP4 down near-miss M64 fallback graph",
          &production_nvfp4_down_near_miss_m64_linear_chain) == 8U,
      "SM87 NVFP4 down near-miss M64 preserves two public M32 fallbacks");
  test.expect(production_nvfp4_down_near_miss_m64_linear_chain,
              "SM87 NVFP4 down near-miss M64 fallback remains ordered");

  bool production_nvfp4_m25_weight4_linear_chain = false;
  bool production_nvfp4_m24_weight4_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_m18_weight4, production_input, 24U,
          production_output,
          "SM87 NVFP4 4-byte-aligned M24 fallback graph",
          &production_nvfp4_m24_weight4_linear_chain) == 3U,
      "SM87 NVFP4 4-byte-aligned M24 uses M8+M8+M8");
  test.expect(production_nvfp4_m24_weight4_linear_chain,
              "SM87 NVFP4 M24 fallback boundary remains ordered");

  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_m18_weight4, production_input, 25U,
          production_output,
          "SM87 NVFP4 4-byte-aligned M25 fallback graph",
          &production_nvfp4_m25_weight4_linear_chain) == 4U,
      "SM87 NVFP4 4-byte-aligned M25 uses M8+M8+M8+M1");
  test.expect(production_nvfp4_m25_weight4_linear_chain,
              "SM87 NVFP4 4-byte-aligned M25 fallback remains ordered");

  bool production_nvfp4_m25_scale1_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_m18_scale1, production_input, 25U,
          production_output,
          "SM87 NVFP4 byte-aligned scale M25 fallback graph",
          &production_nvfp4_m25_scale1_linear_chain) == 4U,
      "SM87 NVFP4 byte-aligned scale M25 uses M8+M8+M8+M1");
  test.expect(production_nvfp4_m25_scale1_linear_chain,
              "SM87 NVFP4 byte-aligned scale M25 fallback remains ordered");

  bool production_nvfp4_m25_near_miss_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4_m18_near_miss, production_input, 25U,
          production_output, "SM87 NVFP4 near-miss M25 fallback graph",
          &production_nvfp4_m25_near_miss_linear_chain) == 4U,
      "SM87 NVFP4 near-miss M25 uses M8+M8+M8+M1");
  test.expect(production_nvfp4_m25_near_miss_linear_chain,
              "SM87 NVFP4 near-miss M25 fallback remains ordered");

  bool production_nvfp4_m25_input2_linear_chain = false;
  test.expect(
      capture_production_nvfp4_tile(
          production_nvfp4, production_input2, 25U, production_output,
          "SM87 NVFP4 2-byte-aligned input M25 fallback graph",
          &production_nvfp4_m25_input2_linear_chain) == 25U,
      "SM87 NVFP4 2-byte-aligned input M25 preserves scalar token order");
  test.expect(production_nvfp4_m25_input2_linear_chain,
              "SM87 NVFP4 2-byte-aligned input M25 fallback remains ordered");

  bool production_nvfp4_m18_pair_linear_chain = false;
  test.expect(
      capture_production_nvfp4_pair(
          production_nvfp4, production_nvfp4_second, production_input, 18U,
          production_output, production_second_output,
          "SM87 NVFP4 production gate/up pair M18 dispatch graph",
          &production_nvfp4_m18_pair_linear_chain) == 2U,
      "SM87 NVFP4 production gate/up pair M18 uses two masked-M32 kernels");
  test.expect(production_nvfp4_m18_pair_linear_chain,
              "SM87 NVFP4 production gate/up pair M18 remains ordered");

  bool production_nvfp4_m17_pair_linear_chain = false;
  test.expect(
      capture_production_nvfp4_pair(
          production_nvfp4, production_nvfp4_second, production_input, 17U,
          production_output, production_second_output,
          "SM87 NVFP4 production gate/up pair M17 dispatch graph",
          &production_nvfp4_m17_pair_linear_chain) == 2U,
      "SM87 NVFP4 production gate/up pair M17 uses two runtime kernels");
  test.expect(production_nvfp4_m17_pair_linear_chain,
              "SM87 NVFP4 production gate/up pair M17 remains ordered");

  bool production_nvfp4_m19_pair_linear_chain = false;
  test.expect(
      capture_production_nvfp4_pair(
          production_nvfp4, production_nvfp4_second, production_input, 19U,
          production_output, production_second_output,
          "SM87 NVFP4 production gate/up pair M19 dispatch graph",
          &production_nvfp4_m19_pair_linear_chain) == 2U,
      "SM87 NVFP4 production gate/up pair M19 uses two runtime kernels");
  test.expect(production_nvfp4_m19_pair_linear_chain,
              "SM87 NVFP4 production gate/up pair M19 remains ordered");

  bool production_nvfp4_m25_pair_linear_chain = false;
  test.expect(
      capture_production_nvfp4_pair(
          production_nvfp4, production_nvfp4_second, production_input, 25U,
          production_output, production_second_output,
          "SM87 NVFP4 production gate/up pair M25 dispatch graph",
          &production_nvfp4_m25_pair_linear_chain) == 2U,
      "SM87 NVFP4 production gate/up pair M25 reduces six kernels to two");
  test.expect(production_nvfp4_m25_pair_linear_chain,
              "SM87 NVFP4 production gate/up pair M25 remains ordered");

  bool production_nvfp4_m31_pair_linear_chain = false;
  test.expect(
      capture_production_nvfp4_pair(
          production_nvfp4, production_nvfp4_second, production_input, 31U,
          production_output, production_second_output,
          "SM87 NVFP4 production gate/up pair M31 dispatch graph",
          &production_nvfp4_m31_pair_linear_chain) == 2U,
      "SM87 NVFP4 production gate/up pair M31 reduces six kernels to two");
  test.expect(production_nvfp4_m31_pair_linear_chain,
              "SM87 NVFP4 production gate/up pair M31 remains ordered");

  bool production_nvfp4_m25_pair_first_weight4_linear_chain = false;
  test.expect(
      capture_production_nvfp4_pair(
          production_nvfp4_m18_weight4, production_nvfp4_second,
          production_input, 25U, production_output, production_second_output,
          "SM87 NVFP4 first-only ineligible pair M25 fallback graph",
          &production_nvfp4_m25_pair_first_weight4_linear_chain) == 7U,
      "SM87 NVFP4 first-only ineligible pair M25 uses ordered recursive fallback");
  test.expect(
      production_nvfp4_m25_pair_first_weight4_linear_chain,
      "SM87 NVFP4 first-only ineligible pair M25 remains one chain");

  const runtime::LinearWeight production_nvfp4_second_weight4 =
      runtime::NvFp4LinearWeight{
          reinterpret_cast<const std::uint8_t*>(
              reinterpret_cast<std::uintptr_t>(
                  production_nvfp4_second_weight) +
              4U),
          production_nvfp4_second_scale, production_companion_scales + 2U,
          production_companion_scales + 3U, 1.0F, 1.0F, 17'408U, 5'120U};
  bool production_nvfp4_m25_pair_second_weight4_linear_chain = false;
  test.expect(
      capture_production_nvfp4_pair(
          production_nvfp4, production_nvfp4_second_weight4,
          production_input, 25U, production_output, production_second_output,
          "SM87 NVFP4 second-only ineligible pair M25 fallback graph",
          &production_nvfp4_m25_pair_second_weight4_linear_chain) == 7U,
      "SM87 NVFP4 second-only ineligible pair M25 uses ordered recursive fallback");
  test.expect(
      production_nvfp4_m25_pair_second_weight4_linear_chain,
      "SM87 NVFP4 second-only ineligible pair M25 remains one chain");

  bool production_nvfp4_m18_pair_weight4_linear_chain = false;
  test.expect(
      capture_production_nvfp4_pair(
          production_nvfp4_m18_weight4, production_nvfp4_second,
          production_input, 18U, production_output, production_second_output,
          "SM87 NVFP4 one-sided 4-byte-aligned pair M18 fallback graph",
          &production_nvfp4_m18_pair_weight4_linear_chain) == 5U,
      "SM87 NVFP4 one-sided 4-byte-aligned pair M18 uses three prefix plus "
      "two tail kernels");
  test.expect(
      production_nvfp4_m18_pair_weight4_linear_chain,
      "SM87 NVFP4 one-sided 4-byte-aligned pair M18 remains ordered");

  bool production_nvfp4_m18_pair_scale1_linear_chain = false;
  test.expect(
      capture_production_nvfp4_pair(
          production_nvfp4_m18_scale1, production_nvfp4_second,
          production_input, 18U, production_output, production_second_output,
          "SM87 NVFP4 one-sided byte-scale pair M18 fallback graph",
          &production_nvfp4_m18_pair_scale1_linear_chain) == 5U,
      "SM87 NVFP4 one-sided byte-scale pair M18 uses three prefix plus two "
      "tail kernels");
  test.expect(production_nvfp4_m18_pair_scale1_linear_chain,
              "SM87 NVFP4 one-sided byte-scale pair M18 remains ordered");

  bool production_nvfp4_m18_pair_input2_linear_chain = false;
  test.expect(
      capture_production_nvfp4_pair(
          production_nvfp4, production_nvfp4_second, production_input2, 18U,
          production_output, production_second_output,
          "SM87 NVFP4 2-byte-aligned input pair M18 fallback graph",
          &production_nvfp4_m18_pair_input2_linear_chain) == 36U,
      "SM87 NVFP4 2-byte-aligned input pair M18 preserves 36 scalar "
      "kernels");
  test.expect(production_nvfp4_m18_pair_input2_linear_chain,
              "SM87 NVFP4 2-byte-aligned input pair M18 remains ordered");

  auto* const production_last_m18_input_row =
      reinterpret_cast<std::uint16_t*>(
          reinterpret_cast<std::uintptr_t>(production_input) +
          17U * 5'120U * sizeof(std::uint16_t));
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_input, 18U, nullptr, 0U,
            production_last_m18_input_row, static_cast<void*>(stream));
      },
      "SM87 NVFP4 M18 output overlaps final input row");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_nvfp4_second, production_input, 18U, nullptr, 0U,
            production_output, production_output,
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 M18 pair aliases outputs");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_nvfp4_second, production_input, 18U, nullptr, 0U,
            production_output, production_last_m18_input_row,
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 M18 pair second output overlaps final input row");

  auto* const odd_production_output = reinterpret_cast<std::uint16_t*>(
      reinterpret_cast<std::uintptr_t>(production_output) + 1U);
  auto* const odd_production_second_output =
      reinterpret_cast<std::uint16_t*>(
          reinterpret_cast<std::uintptr_t>(production_second_output) + 1U);
  const auto* const odd_production_input =
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<std::uintptr_t>(production_input) + 1U);
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            odd_production_input, 25U, nullptr, 0U, production_output,
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 runtime single rejects odd input before enqueue");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_input, 25U, nullptr, 0U, odd_production_output,
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 runtime single rejects odd output before enqueue");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_nvfp4_second, production_input, 25U, nullptr, 0U,
            odd_production_output, production_second_output,
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 runtime pair rejects odd first output before enqueue");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_nvfp4_second, production_input, 25U, nullptr, 0U,
            production_output, odd_production_second_output,
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 runtime pair rejects odd second output without half enqueue");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_nvfp4_second, production_input, 18U, nullptr, 0U,
            production_output, odd_production_second_output,
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 fixed M18 pair rejects odd second output before enqueue");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_nvfp4_second, production_input, 18U, nullptr, 0U,
            odd_production_output, production_second_output,
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 fixed M18 pair rejects odd first output before enqueue");

  auto* const production_last_m31_input_row =
      reinterpret_cast<std::uint16_t*>(
          reinterpret_cast<std::uintptr_t>(production_input) +
          30U * 5'120U * sizeof(std::uint16_t));
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_input, 31U, nullptr, 0U,
            production_last_m31_input_row, static_cast<void*>(stream));
      },
      "SM87 NVFP4 runtime single rejects final-row input alias");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_nvfp4_second, production_input, 31U, nullptr, 0U,
            production_output, production_output,
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 runtime pair rejects output alias before enqueue");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_nvfp4_second, production_input, 31U, nullptr, 0U,
            production_output, production_last_m31_input_row,
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 runtime pair rejects second final-row alias without half enqueue");

  const runtime::LinearWeight production_nvfp4_overflow =
      runtime::NvFp4LinearWeight{
          production_nvfp4_second_weight, production_nvfp4_second_scale,
          production_companion_scales + 2U,
          production_companion_scales + 3U, 1.0F, 1.0F,
          std::numeric_limits<std::size_t>::max(), 5'120U};
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly,
            production_nvfp4_overflow, production_input, 31U, nullptr, 0U,
            production_output, static_cast<void*>(stream));
      },
      "SM87 NVFP4 runtime single rejects overflow before enqueue");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, production_nvfp4,
            production_nvfp4_overflow, production_input, 31U, nullptr, 0U,
            production_output, production_second_output,
            static_cast<void*>(stream));
      },
      "SM87 NVFP4 runtime pair rejects second overflow without half enqueue");

  const auto capture_production_m32 =
      [&](const runtime::LinearWeight& weight,
          const std::string& label, bool* const linear_chain) {
        std::size_t total_nodes = 0U;
        return captured_kernel_node_count(
            test,
            [&](cudaStream_t stream) noexcept {
              return runtime::launch_projection_tile_to_bf16_cuda(
                  runtime::ProjectionBackend::kSm87WeightOnly, weight,
                  production_input, 32U, nullptr, 0U,
                  production_output, static_cast<void*>(stream));
            },
            label, &total_nodes, linear_chain);
      };
  bool production_fp8_linear_chain = false;
  test.expect(capture_production_m32(
                  production_fp8, "SM87 FP8 production M32 dispatch graph",
                  &production_fp8_linear_chain) == 1U,
              "SM87 FP8 production M32 uses one direct WMMA kernel");
  test.expect(production_fp8_linear_chain,
              "SM87 FP8 production M32 preserves a single-node chain");
  const runtime::LinearWeight production_fp8_weight4 =
      runtime::Fp8LinearWeight{
          reinterpret_cast<const std::uint8_t*>(
              reinterpret_cast<std::uintptr_t>(production_fp8_weight) + 4U),
          production_companion_scales, production_companion_scales + 1U,
          1.0F, 1.0F, 5'120U, 6'144U};
  bool production_fp8_weight4_linear_chain = false;
  test.expect(
      capture_production_m32(production_fp8_weight4,
                             "SM87 FP8 4-byte-aligned M32 fallback graph",
                             &production_fp8_weight4_linear_chain) == 4U,
      "SM87 FP8 4-byte-aligned M32 uses two public M16 fallbacks");
  test.expect(production_fp8_weight4_linear_chain,
              "SM87 FP8 4-byte-aligned M32 fallback remains ordered");
  const runtime::LinearWeight production_fp8_near_miss =
      runtime::Fp8LinearWeight{
          production_fp8_weight, production_companion_scales,
          production_companion_scales + 1U, 1.0F, 1.0F, 5'121U, 6'144U};
  bool production_fp8_near_miss_linear_chain = false;
  test.expect(
      capture_production_m32(production_fp8_near_miss,
                             "SM87 FP8 near-miss M32 fallback graph",
                             &production_fp8_near_miss_linear_chain) == 4U,
      "SM87 FP8 near-miss M32 uses two public M16 fallbacks");
  test.expect(production_fp8_near_miss_linear_chain,
              "SM87 FP8 near-miss M32 fallback remains ordered");
  bool production_nvfp4_linear_chain = false;
  test.expect(
      capture_production_m32(production_nvfp4,
                             "SM87 NVFP4 production M32 dispatch graph",
                             &production_nvfp4_linear_chain) == 1U,
      "SM87 NVFP4 production M32 uses one direct WMMA kernel");
  test.expect(production_nvfp4_linear_chain,
              "SM87 NVFP4 production M32 preserves a single-node chain");

  const runtime::LinearWeight production_nvfp4_weight4 =
      runtime::NvFp4LinearWeight{
          reinterpret_cast<const std::uint8_t*>(
              reinterpret_cast<std::uintptr_t>(production_nvfp4_weight) + 4U),
          production_nvfp4_scale, production_companion_scales,
          production_companion_scales + 1U, 1.0F, 1.0F, 17'408U, 5'120U};
  bool production_nvfp4_weight4_linear_chain = false;
  test.expect(
      capture_production_m32(
          production_nvfp4_weight4,
          "SM87 NVFP4 4-byte-aligned M32 fallback graph",
          &production_nvfp4_weight4_linear_chain) == 4U,
      "SM87 NVFP4 4-byte-aligned M32 fallback expands to four M8 kernels");
  test.expect(production_nvfp4_weight4_linear_chain,
              "SM87 NVFP4 4-byte-aligned M32 fallback remains ordered");

  const runtime::LinearWeight production_nvfp4_scale1 =
      runtime::NvFp4LinearWeight{
          production_nvfp4_weight, production_nvfp4_scale + 1U,
          production_companion_scales, production_companion_scales + 1U,
          1.0F, 1.0F, 17'408U, 5'120U};
  bool production_nvfp4_scale1_linear_chain = false;
  test.expect(
      capture_production_m32(
          production_nvfp4_scale1,
          "SM87 NVFP4 byte-aligned scale M32 fallback graph",
          &production_nvfp4_scale1_linear_chain) == 4U,
      "SM87 NVFP4 byte-aligned scales use two public M16 fallbacks");
  test.expect(production_nvfp4_scale1_linear_chain,
              "SM87 NVFP4 byte-aligned scale fallback remains ordered");

  const runtime::LinearWeight production_nvfp4_near_miss =
      runtime::NvFp4LinearWeight{
          production_nvfp4_weight, production_nvfp4_scale,
          production_companion_scales, production_companion_scales + 1U,
          1.0F, 1.0F, 17'407U, 5'120U};
  bool production_nvfp4_near_miss_linear_chain = false;
  test.expect(
      capture_production_m32(
          production_nvfp4_near_miss,
          "SM87 NVFP4 near-miss M32 fallback graph",
          &production_nvfp4_near_miss_linear_chain) == 4U,
      "SM87 NVFP4 near-miss M32 uses two public M16 fallbacks");
  test.expect(production_nvfp4_near_miss_linear_chain,
              "SM87 NVFP4 near-miss M32 fallback remains ordered");

  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, fp8,
            fp8_activation.get(), kMaximumTokens + 1U, nullptr, 0U,
            output.get(), static_cast<void*>(stream));
      },
      "SM87 FP8 M65 tile guard");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, nvfp4,
            nvfp4_activation.get(), kMaximumTokens + 1U, nullptr, 0U,
            output.get(), static_cast<void*>(stream));
      },
      "SM87 NVFP4 M65 tile guard");
}

void test_exact_fp8_whole_chunk_projection_dispatch(TestContext& test) {
  constexpr std::size_t kM64Tokens = 64U;
  const auto* const encoded_weight =
      reinterpret_cast<const std::uint8_t*>(0x10'0000'0000ULL);
  const auto* const companion_scales =
      reinterpret_cast<const float*>(0x20'0000'0000ULL);
  const auto* const input =
      reinterpret_cast<const std::uint16_t*>(0x30'0000'0000ULL);
  auto* const output =
      reinterpret_cast<std::uint16_t*>(0x40'0000'0000ULL);
  const auto make_fp8 = [&](const std::size_t rows,
                            const std::size_t columns) {
    return runtime::LinearWeight{runtime::Fp8LinearWeight{
        encoded_weight, companion_scales, companion_scales + 1U,
        1.0F, 1.0F, rows, columns}};
  };
  const runtime::LinearWeight qkv = make_fp8(10'240U, 5'120U);
  const runtime::LinearWeight z = make_fp8(6'144U, 5'120U);
  const runtime::LinearWeight full_query = make_fp8(12'288U, 5'120U);
  const runtime::LinearWeight full_kv = make_fp8(1'024U, 5'120U);
  const runtime::LinearWeight attention_output = make_fp8(5'120U, 6'144U);

  test.expect(runtime::kMaximumProjectionTileTokenCount == kM64Tokens,
              "FP8 whole-chunk entry leaves the generic cap at C64");
  const auto expect_exact_grid =
      [&](const runtime::LinearWeight& weight,
          const std::size_t rows, const std::size_t columns,
          const std::size_t token_count, const std::string& label) {
        const CapturedKernelChain dispatch = capture_ordered_kernel_chain(
            test,
            [&](cudaStream_t stream) noexcept {
              return runtime::
                  launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
                      runtime::ProjectionBackend::kSm87WeightOnly, weight,
                      input, token_count, output,
                      static_cast<void*>(stream));
            },
            label + " dispatch graph");
        const CapturedKernelChain direct = capture_ordered_kernel_chain(
            test,
            [&](cudaStream_t stream) noexcept {
              return q3x::kernels::
                  launch_sm87_fp8_w8a16_whole_chunk_gemm_bf16_cuda(
                      encoded_weight, 1.0F, input, token_count, rows,
                      columns, output, static_cast<void*>(stream));
            },
            label + " direct graph");
        const std::size_t production_tile_tokens =
            rows == 1'024U ? kM64Tokens : 128U;
        const unsigned int expected_grid = static_cast<unsigned int>(
            (rows / 128U) * (token_count / production_tile_tokens));
        const unsigned int expected_dynamic_shared =
            rows == 6'144U && columns == 5'120U && token_count == 512U
                ? 79'872U
                : 0U;
        const bool exact =
            dispatch.valid && dispatch.launches.size() == 1U && direct.valid &&
            direct.launches.size() == 1U &&
            dispatch.launches.front().function ==
                direct.launches.front().function &&
            dispatch.launches.front().grid.x == expected_grid &&
            dispatch.launches.front().grid.y == 1U &&
            dispatch.launches.front().grid.z == 1U &&
            dispatch.launches.front().block.x == 256U &&
            dispatch.launches.front().block.y == 1U &&
            dispatch.launches.front().block.z == 1U &&
            dispatch.launches.front().dynamic_shared_bytes ==
                expected_dynamic_shared;
        test.expect(exact,
                    label + " is one exact whole-chunk production grid");
      };
  for (const std::size_t token_count : {256U, 512U}) {
    expect_exact_grid(qkv, 10'240U, 5'120U, token_count,
                      "FP8 QKV C" + std::to_string(token_count));
    expect_exact_grid(z, 6'144U, 5'120U, token_count,
                      "FP8 Z C" + std::to_string(token_count));
    expect_exact_grid(full_query, 12'288U, 5'120U, token_count,
                      "FP8 full Q C" + std::to_string(token_count));
    expect_exact_grid(full_kv, 1'024U, 5'120U, token_count,
                      "FP8 full K/V C" + std::to_string(token_count));
    expect_exact_grid(attention_output, 5'120U, 6'144U, token_count,
                      "FP8 O C" + std::to_string(token_count));
  }

  const auto capture_z_public = [&](const std::uint16_t* const selected_input,
                                    const std::size_t token_count,
                                    const std::string& label) {
    return capture_ordered_kernel_chain(
        test,
        [&](cudaStream_t stream) noexcept {
          return q3x::kernels::
              launch_sm87_fp8_w8a16_whole_chunk_gemm_bf16_cuda(
                  encoded_weight, 1.0F, selected_input, token_count, 6'144U,
                  5'120U, output, static_cast<void*>(stream));
        },
        label);
  };
  const auto capture_z_frozen_m128 =
      [&](const std::uint16_t* const selected_input,
          const std::size_t token_count, const std::string& label) {
        return capture_ordered_kernel_chain(
            test,
            [&](cudaStream_t stream) noexcept {
              return q3x::kernels::
                  launch_sm87_fp8_w8a16_whole_chunk_large_n_m128_b_reuse_test_cuda(
                      encoded_weight, 1.0F, selected_input, token_count,
                      6'144U, 5'120U, output, static_cast<void*>(stream));
            },
            label);
      };
  const CapturedKernelChain z_c512_production =
      capture_z_public(input, 512U, "FP8 Z C512 production graph");
  const CapturedKernelChain z_c512_candidate = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return q3x::kernels::
            launch_sm87_fp8_w8a16_whole_chunk_z_m128_cp_async_canonical_xor_register_feed_test_cuda(
                encoded_weight, 1.0F, input, 512U, 6'144U, 5'120U, output,
                static_cast<void*>(stream));
      },
      "FP8 Z C512 candidate graph");
  const CapturedKernelChain z_c512_frozen = capture_z_frozen_m128(
      input, 512U, "FP8 Z C512 frozen M128 graph");
  test.expect(
      z_c512_production.valid &&
          z_c512_production.launches.size() == 1U &&
          z_c512_candidate.valid && z_c512_candidate.launches.size() == 1U &&
          z_c512_frozen.valid && z_c512_frozen.launches.size() == 1U &&
          z_c512_production.launches.front().function ==
              z_c512_candidate.launches.front().function &&
          z_c512_production.launches.front().function !=
              z_c512_frozen.launches.front().function &&
          z_c512_production.launches.front().grid.x == 192U &&
          z_c512_production.launches.front().block.x == 256U &&
          z_c512_production.launches.front().dynamic_shared_bytes == 79'872U,
      "FP8 Z C512 16-byte activation selects the canonical register-feed "
      "production function");

  const CapturedKernelChain z_c256_production =
      capture_z_public(input, 256U, "FP8 Z C256 production graph");
  const CapturedKernelChain z_c256_frozen =
      capture_z_frozen_m128(input, 256U, "FP8 Z C256 frozen M128 graph");
  test.expect(
      z_c256_production.valid && z_c256_production.launches.size() == 1U &&
          z_c256_frozen.valid && z_c256_frozen.launches.size() == 1U &&
          z_c256_production.launches.front().function ==
              z_c256_frozen.launches.front().function &&
          z_c256_production.launches.front().grid.x == 96U &&
          z_c256_production.launches.front().dynamic_shared_bytes == 0U,
      "FP8 Z C256 remains on the frozen M128 B-reuse function");

  const auto* const z_eight_not_sixteen_aligned_input =
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<std::uintptr_t>(input) + 8U);
  const CapturedKernelChain z_c512_alignment_fallback = capture_z_public(
      z_eight_not_sixteen_aligned_input, 512U,
      "FP8 Z C512 8-byte activation production graph");
  const CapturedKernelChain z_c512_alignment_frozen = capture_z_frozen_m128(
      z_eight_not_sixteen_aligned_input, 512U,
      "FP8 Z C512 8-byte activation frozen M128 graph");
  test.expect(
      z_c512_alignment_fallback.valid &&
          z_c512_alignment_fallback.launches.size() == 1U &&
          z_c512_alignment_frozen.valid &&
          z_c512_alignment_frozen.launches.size() == 1U &&
          z_c512_alignment_fallback.launches.front().function ==
              z_c512_alignment_frozen.launches.front().function &&
          z_c512_alignment_fallback.launches.front().grid.x == 192U &&
          z_c512_alignment_fallback.launches.front().dynamic_shared_bytes ==
              0U,
      "FP8 Z C512 8-byte activation preserves the public ABI via frozen "
      "M128 fallback");

  const auto* const register_feed_sidecar =
      reinterpret_cast<const std::uint8_t*>(0x50'0000'0000ULL);
  runtime::Fp8LinearWeight attached_qkv_payload =
      std::get<runtime::Fp8LinearWeight>(qkv);
  attached_qkv_payload.prefill_qkv_register_feed_sidecar =
      register_feed_sidecar;
  const runtime::LinearWeight attached_qkv = attached_qkv_payload;
  const CapturedKernelChain attached_c512 = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, attached_qkv, input,
            512U, output, static_cast<void*>(stream));
      },
      "FP8 attached QKV C512 production dispatch graph");
  const CapturedKernelChain register_feed_oracle =
      capture_ordered_kernel_chain(
          test,
          [&](cudaStream_t stream) noexcept {
            return q3x::kernels::
                launch_sm87_fp8_w8a16_whole_chunk_qkv_register_feed_gemm_bf16_cuda(
                    register_feed_sidecar, 1.0F, input, 512U, 10'240U,
                    5'120U, output, static_cast<void*>(stream));
          },
          "FP8 QKV C512 register-feed oracle graph");
  const CapturedKernelChain canonical_c512 = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, input, 512U,
            output, static_cast<void*>(stream));
      },
      "FP8 null-sidecar QKV C512 production dispatch graph");
  test.expect(
      attached_c512.valid && attached_c512.launches.size() == 1U &&
          register_feed_oracle.valid &&
          register_feed_oracle.launches.size() == 1U &&
          canonical_c512.valid && canonical_c512.launches.size() == 1U &&
          attached_c512.launches.front().function ==
              register_feed_oracle.launches.front().function &&
          attached_c512.launches.front().function !=
              canonical_c512.launches.front().function &&
          attached_c512.launches.front().grid.x == 320U &&
          attached_c512.launches.front().grid.y == 1U &&
          attached_c512.launches.front().grid.z == 1U &&
          attached_c512.launches.front().block.x == 256U &&
          attached_c512.launches.front().block.y == 1U &&
          attached_c512.launches.front().block.z == 1U &&
          attached_c512.launches.front().dynamic_shared_bytes == 79'872U,
      "FP8 attached exact QKV C512 selects one screened register-feed grid");

  const auto expect_same_single_kernel =
      [&](const CapturedKernelChain& first, const CapturedKernelChain& second,
          const std::string& label) {
        test.expect(first.valid && first.launches.size() == 1U &&
                        second.valid && second.launches.size() == 1U &&
                        first.launches.front().function ==
                            second.launches.front().function &&
                        first.launches.front().grid.x ==
                            second.launches.front().grid.x &&
                        first.launches.front().block.x ==
                            second.launches.front().block.x &&
                        first.launches.front().dynamic_shared_bytes ==
                            second.launches.front().dynamic_shared_bytes,
                    label);
      };
  const CapturedKernelChain attached_c256 = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, attached_qkv, input,
            256U, output, static_cast<void*>(stream));
      },
      "FP8 attached QKV C256 dispatch graph");
  const CapturedKernelChain canonical_c256 = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, input, 256U,
            output, static_cast<void*>(stream));
      },
      "FP8 canonical QKV C256 dispatch graph");
  expect_same_single_kernel(
      attached_c256, canonical_c256,
      "FP8 QKV C256 ignores the exact-C512 register-feed sidecar");

  const auto* const eight_not_sixteen_aligned_input =
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<std::uintptr_t>(input) + 8U);
  const CapturedKernelChain attached_c512_alignment_fallback =
      capture_ordered_kernel_chain(
          test,
          [&](cudaStream_t stream) noexcept {
            return runtime::
                launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
                    runtime::ProjectionBackend::kSm87WeightOnly,
                    attached_qkv, eight_not_sixteen_aligned_input, 512U,
                    output, static_cast<void*>(stream));
          },
          "FP8 attached QKV C512 8-byte activation fallback graph");
  const CapturedKernelChain canonical_c512_alignment_fallback =
      capture_ordered_kernel_chain(
          test,
          [&](cudaStream_t stream) noexcept {
            return runtime::
                launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
                    runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                    eight_not_sixteen_aligned_input, 512U, output,
                    static_cast<void*>(stream));
          },
          "FP8 canonical QKV C512 8-byte activation graph");
  expect_same_single_kernel(
      attached_c512_alignment_fallback,
      canonical_c512_alignment_fallback,
      "FP8 QKV C512 8-byte activation alignment preserves canonical ABI");

  runtime::Fp8LinearWeight attached_z_payload =
      std::get<runtime::Fp8LinearWeight>(z);
  attached_z_payload.prefill_qkv_register_feed_sidecar =
      register_feed_sidecar;
  const CapturedKernelChain attached_z_c512 = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly,
            runtime::LinearWeight{attached_z_payload}, input, 512U, output,
            static_cast<void*>(stream));
      },
      "FP8 attached near-shape Z C512 dispatch graph");
  const CapturedKernelChain canonical_z_c512 = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, z, input, 512U,
            output, static_cast<void*>(stream));
      },
      "FP8 canonical Z C512 dispatch graph");
  expect_same_single_kernel(
      attached_z_c512, canonical_z_c512,
      "FP8 non-QKV C512 ignores the QKV register-feed sidecar");

  const CapturedKernelChain attached_qkv_m1 = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, attached_qkv, input,
            1U, nullptr, 0U, output, static_cast<void*>(stream));
      },
      "FP8 attached QKV M1 dispatch graph");
  const CapturedKernelChain canonical_qkv_m1 = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, input, 1U,
            nullptr, 0U, output, static_cast<void*>(stream));
      },
      "FP8 canonical QKV M1 dispatch graph");
  expect_same_single_kernel(
      attached_qkv_m1, canonical_qkv_m1,
      "FP8 QKV Decode M1 remains on the canonical layout");

  runtime::Fp8LinearWeight misaligned_sidecar_payload = attached_qkv_payload;
  misaligned_sidecar_payload.prefill_qkv_register_feed_sidecar =
      register_feed_sidecar + 1U;
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly,
            runtime::LinearWeight{misaligned_sidecar_payload}, input, 512U,
            output, static_cast<void*>(stream));
      },
      "FP8 QKV C512 rejects a malformed attached register-feed sidecar");

  const auto expect_full_attention_layout =
      [&](const runtime::LinearWeight& weight, const std::size_t rows,
          const std::size_t token_count, const bool is_query,
          const std::string& label) {
        const CapturedKernelChain production = capture_ordered_kernel_chain(
            test,
            [&](cudaStream_t stream) noexcept {
              return runtime::
                  launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
                      runtime::ProjectionBackend::kSm87WeightOnly, weight,
                      input, token_count, output,
                      static_cast<void*>(stream));
            },
            label + " production graph");
        const CapturedKernelChain frozen_layout =
            capture_ordered_kernel_chain(
                test,
                [&](cudaStream_t stream) noexcept {
                  if (is_query) {
                    return q3x::kernels::
                        launch_sm87_fp8_w8a16_whole_chunk_large_n_m128_b_reuse_test_cuda(
                            encoded_weight, 1.0F, input, token_count,
                            rows, 5'120U, output,
                            static_cast<void*>(stream));
                  }
                  return q3x::kernels::
                      launch_sm87_fp8_w8a16_whole_chunk_m64_historical_control_test_cuda(
                          encoded_weight, 1.0F, input, token_count, rows,
                          5'120U, output,
                          static_cast<void*>(stream));
                },
                label + " frozen layout graph");
        test.expect(
            production.valid && production.launches.size() == 1U &&
                frozen_layout.valid && frozen_layout.launches.size() == 1U &&
                production.launches.front().function ==
                    frozen_layout.launches.front().function,
            label + " selects the screened production layout");
      };
  for (const std::size_t token_count : {256U, 512U}) {
    expect_full_attention_layout(
        full_query, 12'288U, token_count, true,
        "FP8 full Q C" + std::to_string(token_count) + " promoted M128");
  }
  expect_full_attention_layout(full_kv, 1'024U, 256U, false,
                               "FP8 full K/V C256 M-major");
  expect_full_attention_layout(full_kv, 1'024U, 512U, false,
                               "FP8 full K/V C512 N-major");

  // Exercise the public runtime route on real storage and a non-default
  // stream. The low-level launcher must clear an unrelated stale last-error
  // before capture, and the resulting one-node graph must instantiate and
  // replay successfully.
  constexpr std::size_t kReplayTokens = 256U;
  constexpr std::size_t kReplayRows = 10'240U;
  constexpr std::size_t kReplayColumns = 5'120U;
  DeviceBuffer<std::uint8_t> replay_weight;
  DeviceBuffer<float> replay_scales;
  DeviceBuffer<std::uint16_t> replay_input;
  DeviceBuffer<std::uint16_t> replay_output;
  bool replay_ready = replay_weight.allocate(
      test, kReplayRows * kReplayColumns,
      "FP8 whole-chunk graph replay weight");
  replay_ready = replay_ready && replay_scales.allocate(
                                     test, 2U,
                                     "FP8 whole-chunk graph replay scales");
  replay_ready = replay_ready && replay_input.allocate(
                                     test, kReplayTokens * kReplayColumns,
                                     "FP8 whole-chunk graph replay input");
  replay_ready = replay_ready && replay_output.allocate(
                                     test, kReplayTokens * kReplayRows,
                                     "FP8 whole-chunk graph replay output");
  replay_ready = replay_ready && test.cuda_ok(
                                     cudaMemset(replay_weight.get(), 0,
                                                kReplayRows * kReplayColumns),
                                     "zero graph replay weights");
  replay_ready = replay_ready && test.cuda_ok(
                                     cudaMemset(
                                         replay_input.get(), 0,
                                         kReplayTokens * kReplayColumns *
                                             sizeof(std::uint16_t)),
                                     "zero graph replay input");
  const std::array<float, 2U> host_scales{{1.0F, 1.0F}};
  replay_ready = replay_ready && test.cuda_ok(
                                     cudaMemcpy(
                                         replay_scales.get(),
                                         host_scales.data(),
                                         sizeof(host_scales),
                                         cudaMemcpyHostToDevice),
                                     "upload graph replay scales");
  cudaStream_t replay_stream = nullptr;
  cudaGraph_t replay_graph = nullptr;
  cudaGraphExec_t replay_exec = nullptr;
  replay_ready = replay_ready && test.cuda_ok(
                                     cudaStreamCreateWithFlags(
                                         &replay_stream,
                                         cudaStreamNonBlocking),
                                     "create graph replay stream");
  if (replay_ready) {
    const runtime::LinearWeight replay_qkv = runtime::Fp8LinearWeight{
        replay_weight.get(), replay_scales.get(), replay_scales.get() + 1U,
        1.0F, 1.0F, kReplayRows, kReplayColumns};
    const cudaError_t stale =
        cudaMemcpy(nullptr, nullptr, 1U, cudaMemcpyHostToDevice);
    test.expect(stale == cudaErrorInvalidValue,
                "FP8 whole chunk seeds a stale CUDA last-error");
    replay_ready = test.cuda_ok(
        cudaStreamBeginCapture(replay_stream, cudaStreamCaptureModeGlobal),
        "begin FP8 whole-chunk replay capture");
    int launch_status = static_cast<int>(cudaErrorUnknown);
    if (replay_ready) {
      launch_status =
          runtime::launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly, replay_qkv,
              replay_input.get(), kReplayTokens, replay_output.get(),
              static_cast<void*>(replay_stream));
      test.expect(static_cast<cudaError_t>(launch_status) == cudaSuccess,
                  "FP8 whole chunk ignores stale error during capture");
      replay_ready = test.cuda_ok(
          cudaStreamEndCapture(replay_stream, &replay_graph),
          "end FP8 whole-chunk replay capture");
    }
    std::size_t replay_nodes = 0U;
    replay_ready = replay_ready && test.cuda_ok(
                                       cudaGraphGetNodes(
                                           replay_graph, nullptr,
                                           &replay_nodes),
                                       "count FP8 replay graph nodes");
    test.expect(replay_nodes == 1U,
                "FP8 whole-chunk replay graph contains one kernel");
    replay_ready = replay_ready && test.cuda_ok(
                                       cudaGraphInstantiate(
                                           &replay_exec, replay_graph,
                                           nullptr, nullptr, 0U),
                                       "instantiate FP8 replay graph");
    replay_ready = replay_ready && test.cuda_ok(
                                       cudaMemsetAsync(
                                           replay_output.get(), 0xff,
                                           kReplayTokens * kReplayRows *
                                               sizeof(std::uint16_t),
                                           replay_stream),
                                       "poison FP8 replay output");
    replay_ready = replay_ready && test.cuda_ok(
                                       cudaGraphLaunch(replay_exec,
                                                       replay_stream),
                                       "launch FP8 replay graph");
    replay_ready = replay_ready && test.cuda_ok(
                                       cudaStreamSynchronize(replay_stream),
                                       "synchronize FP8 replay graph");
    std::array<std::uint16_t, 2U> edge_outputs{{0xffffU, 0xffffU}};
    if (replay_ready) {
      replay_ready = test.cuda_ok(
          cudaMemcpy(&edge_outputs[0], replay_output.get(),
                     sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
          "read first FP8 replay output");
      replay_ready = replay_ready && test.cuda_ok(
          cudaMemcpy(&edge_outputs[1],
                     replay_output.get() +
                         kReplayTokens * kReplayRows - 1U,
                     sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
          "read last FP8 replay output");
    }
    test.expect(replay_ready && edge_outputs[0] == 0U &&
                    edge_outputs[1] == 0U,
                "FP8 whole-chunk graph replay writes the complete output");
  }
  if (replay_exec != nullptr) {
    (void)test.cuda_ok(cudaGraphExecDestroy(replay_exec),
                       "destroy FP8 replay executable");
  }
  if (replay_graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(replay_graph),
                       "destroy FP8 replay graph");
  }
  if (replay_stream != nullptr) {
    (void)test.cuda_ok(cudaStreamDestroy(replay_stream),
                       "destroy FP8 replay stream");
  }

  const auto expect_not_supported =
      [&](const runtime::ProjectionBackend backend,
          const runtime::LinearWeight& weight,
          const std::uint16_t* const selected_input,
          const std::size_t token_count, const std::string& label) {
        expect_not_supported_capture_has_no_nodes(
            test,
            [&](cudaStream_t stream) noexcept {
              return runtime::
                  launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
                      backend, weight, selected_input, token_count, output,
                      static_cast<void*>(stream));
            },
            label);
      };
  expect_not_supported(runtime::ProjectionBackend::kReference, qkv, input,
                       256U, "FP8 whole chunk rejects reference backend");
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                       input, 64U, "FP8 whole chunk rejects C64");
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                       input, 255U, "FP8 whole chunk rejects C255");
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                       input, 513U, "FP8 whole chunk rejects C513");
  const runtime::LinearWeight shape_near_miss =
      make_fp8(10'239U, 5'120U);
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly,
                       shape_near_miss, input, 256U,
                       "FP8 whole chunk rejects QKV N near miss");
  const runtime::LinearWeight full_query_near_miss =
      make_fp8(12'287U, 5'120U);
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly,
                       full_query_near_miss, input, 256U,
                       "FP8 whole chunk rejects full-Q N near miss");
  const runtime::LinearWeight full_kv_near_miss =
      make_fp8(1'023U, 5'120U);
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly,
                       full_kv_near_miss, input, 512U,
                       "FP8 whole chunk rejects full-K/V N near miss");
  const runtime::LinearWeight type_near_miss = runtime::Bf16LinearWeight{
      reinterpret_cast<const std::uint16_t*>(encoded_weight), 10'240U,
      5'120U};
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly,
                       type_near_miss, input, 256U,
                       "FP8 whole chunk rejects BF16 payload");
  const runtime::LinearWeight unaligned_weight = runtime::Fp8LinearWeight{
      encoded_weight + 4U, companion_scales, companion_scales + 1U,
      1.0F, 1.0F, 10'240U, 5'120U};
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly,
                       unaligned_weight, input, 256U,
                       "FP8 whole chunk rejects 4-byte weight alignment");
  const auto* const unaligned_input = reinterpret_cast<const std::uint16_t*>(
      reinterpret_cast<std::uintptr_t>(input) + 2U);
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                       unaligned_input, 256U,
                       "FP8 whole chunk rejects 2-byte input alignment");

  const auto expect_invalid =
      [&](const runtime::LinearWeight& weight,
          const std::uint16_t* const selected_input,
          std::uint16_t* const selected_output, const std::string& label) {
        expect_invalid_capture_has_no_nodes(
            test,
            [&](cudaStream_t stream) noexcept {
              return runtime::
                  launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
                      runtime::ProjectionBackend::kSm87WeightOnly, weight,
                      selected_input, 256U, selected_output,
                      static_cast<void*>(stream));
            },
            label);
      };
  expect_invalid(runtime::Fp8LinearWeight{
                     nullptr, companion_scales, companion_scales + 1U,
                     1.0F, 1.0F, 10'240U, 5'120U},
                 input, output, "FP8 whole chunk rejects null weight");
  expect_invalid(runtime::Fp8LinearWeight{
                     encoded_weight, nullptr, companion_scales + 1U,
                     1.0F, 1.0F, 10'240U, 5'120U},
                 input, output,
                 "FP8 whole chunk rejects null weight-scale device pointer");
  expect_invalid(runtime::Fp8LinearWeight{
                     encoded_weight, companion_scales, nullptr,
                     1.0F, 1.0F, 10'240U, 5'120U},
                 input, output,
                 "FP8 whole chunk rejects null input-scale device pointer");
  expect_invalid(runtime::Fp8LinearWeight{
                     encoded_weight, nullptr, companion_scales + 1U,
                     1.0F, 1.0F, 12'288U, 5'120U},
                 input, output,
                 "FP8 full-Q whole chunk requires weight-scale companion");
  expect_invalid(runtime::Fp8LinearWeight{
                     encoded_weight, companion_scales, nullptr,
                     1.0F, 1.0F, 1'024U, 5'120U},
                 input, output,
                 "FP8 full-K/V whole chunk requires input-scale companion");
  expect_invalid(runtime::Fp8LinearWeight{
                     encoded_weight, companion_scales,
                     companion_scales + 1U,
                     std::numeric_limits<float>::quiet_NaN(), 1.0F,
                     10'240U, 5'120U},
                 input, output, "FP8 whole chunk rejects NaN weight scale");
  expect_invalid(qkv, nullptr, output,
                 "FP8 whole chunk rejects null input");
  expect_invalid(qkv, input, nullptr,
                 "FP8 whole chunk rejects null output");
  expect_invalid(qkv, input, const_cast<std::uint16_t*>(input),
                 "FP8 whole chunk rejects full-chunk input/output alias");
  expect_invalid(
      qkv, input,
      const_cast<std::uint16_t*>(input) + 64U * 5'120U,
      "FP8 whole chunk rejects alias beginning beyond the first C64 input");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, qkv, input,
                512U,
                const_cast<std::uint16_t*>(input) + 64U * 5'120U,
                static_cast<void*>(stream));
      },
      "FP8 C512 rejects alias beginning beyond the first C64 input");
  expect_invalid(
      qkv, input,
      reinterpret_cast<std::uint16_t*>(
          const_cast<std::uint8_t*>(encoded_weight) + 128U),
      "FP8 whole chunk rejects output inside the weight span");
  expect_invalid(qkv, input,
                 reinterpret_cast<std::uint16_t*>(
                     const_cast<float*>(companion_scales)),
                 "FP8 whole chunk rejects output over companion scale");
  auto* const odd_output = reinterpret_cast<std::uint16_t*>(
      reinterpret_cast<std::uintptr_t>(output) + 1U);
  expect_invalid(qkv, input, odd_output,
                 "FP8 whole chunk rejects odd BF16 output");
  const std::uintptr_t near_end =
      std::numeric_limits<std::uintptr_t>::max() - 3U;
  expect_invalid(qkv,
                 reinterpret_cast<const std::uint16_t*>(near_end), output,
                 "FP8 whole chunk rejects wrapping input range");
  const runtime::LinearWeight wrapping_weight = runtime::Fp8LinearWeight{
      reinterpret_cast<const std::uint8_t*>(near_end), companion_scales,
      companion_scales + 1U, 1.0F, 1.0F, 10'240U, 5'120U};
  expect_invalid(wrapping_weight, input, output,
                 "FP8 whole chunk rejects wrapping weight range");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, full_kv,
                input, 512U,
                reinterpret_cast<std::uint16_t*>(near_end),
                static_cast<void*>(stream));
      },
      "FP8 full-K/V C512 rejects wrapping output range");

  std::size_t fallback_total_nodes = 0U;
  bool fallback_linear_chain = false;
  const std::size_t fallback_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        constexpr std::size_t kRunnerSubtileTokens = 32U;
        int status =
            runtime::launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly,
                shape_near_miss, input, 256U, output,
                static_cast<void*>(stream));
        if (static_cast<cudaError_t>(status) != cudaErrorNotSupported) {
          return status;
        }
        for (std::size_t offset = 0U; offset < 256U;
             offset += kRunnerSubtileTokens) {
          status = runtime::launch_projection_tile_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly,
              shape_near_miss, input + offset * 5'120U,
              kRunnerSubtileTokens, nullptr, 0U,
              output + offset * 10'239U, static_cast<void*>(stream));
          if (status != static_cast<int>(cudaSuccess)) {
            return status;
          }
        }
        return static_cast<int>(cudaSuccess);
      },
      "FP8 C256 near-miss whole-chunk fallback graph",
      &fallback_total_nodes, &fallback_linear_chain);
  test.expect(fallback_kernel_nodes == 32U && fallback_total_nodes == 32U &&
                  fallback_linear_chain,
              "FP8 whole-chunk near miss retains eight ordered generic C32 "
              "fallback schedules");
}

void test_exact_nvfp4_whole_chunk_branch_dispatch(TestContext& test) {
  constexpr std::size_t kHiddenSize = 5'120U;
  constexpr std::size_t kIntermediateSize = 17'408U;
  constexpr std::size_t kM64Tokens = 64U;
  const auto* const packed_weight =
      reinterpret_cast<const std::uint8_t*>(0x10'0000'0000ULL);
  const auto* const block_scale =
      reinterpret_cast<const std::uint8_t*>(0x20'0000'0000ULL);
  const auto* const companion_scales =
      reinterpret_cast<const float*>(0x30'0000'0000ULL);
  const auto* const hidden_input =
      reinterpret_cast<const std::uint16_t*>(0x40'0000'0000ULL);
  const auto* const intermediate_input =
      reinterpret_cast<const std::uint16_t*>(0x50'0000'0000ULL);
  auto* const intermediate_output =
      reinterpret_cast<std::uint16_t*>(0x60'0000'0000ULL);
  auto* const hidden_output =
      reinterpret_cast<std::uint16_t*>(0x70'0000'0000ULL);

  const runtime::LinearWeight gate_up = runtime::NvFp4LinearWeight{
      packed_weight, block_scale, companion_scales, companion_scales + 1U,
      1.0F, 1.0F, kIntermediateSize, kHiddenSize};
  const runtime::LinearWeight down = runtime::NvFp4LinearWeight{
      packed_weight, block_scale, companion_scales, companion_scales + 1U,
      1.0F, 1.0F, kHiddenSize, kIntermediateSize};

  test.expect(runtime::kMaximumProjectionTileTokenCount == kM64Tokens,
              "whole-chunk entry leaves the generic projection cap at C64");

  const auto expect_exact_grid =
      [&](const runtime::LinearWeight& weight,
          const std::uint16_t* const input, std::uint16_t* const output,
          const std::size_t token_count, const bool gate_up_shape,
          const std::string& label) {
        const CapturedKernelChain dispatch = capture_ordered_kernel_chain(
            test,
            [&](cudaStream_t stream) noexcept {
              return runtime::
                  launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
                      runtime::ProjectionBackend::kSm87WeightOnly, weight,
                      input, token_count, output,
                      static_cast<void*>(stream));
            },
            label + " dispatch graph");
        const CapturedKernelChain direct = capture_ordered_kernel_chain(
            test,
            [&](cudaStream_t stream) noexcept {
              if (gate_up_shape) {
                return q3x::kernels::
                    launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
                        packed_weight, block_scale, 1.0F, input, token_count,
                        kIntermediateSize, kHiddenSize, output,
                        static_cast<void*>(stream));
              }
              return q3x::kernels::
                  launch_sm87_nvfp4_w4a16_whole_chunk_down_gemm_bf16_cuda(
                      packed_weight, block_scale, 1.0F, input, token_count,
                      kHiddenSize, kIntermediateSize, output,
                      static_cast<void*>(stream));
            },
            label + " direct production graph");
        constexpr std::size_t kWholeChunkTokensPerCta = 128U;
        const bool n256_c512 = token_count == 512U;
        const std::size_t output_columns_per_cta =
            n256_c512 ? 256U : 128U;
        const unsigned int expected_grid = static_cast<unsigned int>(
            (gate_up_shape ? kIntermediateSize : kHiddenSize) /
            output_columns_per_cta *
            (token_count / kWholeChunkTokensPerCta));
        const std::size_t expected_dynamic_shared_bytes =
            n256_c512 ? 118'784U : 0U;
        const bool exact =
            dispatch.valid && dispatch.launches.size() == 1U &&
            direct.valid && direct.launches.size() == 1U &&
            dispatch.launches.front().function != nullptr &&
            dispatch.launches.front().function ==
                direct.launches.front().function &&
            dispatch.launches.front().grid.x == expected_grid &&
            dispatch.launches.front().grid.y == 1U &&
            dispatch.launches.front().grid.z == 1U &&
            dispatch.launches.front().block.x == 256U &&
            dispatch.launches.front().block.y == 1U &&
            dispatch.launches.front().block.z == 1U &&
            dispatch.launches.front().dynamic_shared_bytes ==
                expected_dynamic_shared_bytes &&
            direct.launches.front().grid.x == expected_grid &&
            direct.launches.front().block.x == 256U &&
            direct.launches.front().dynamic_shared_bytes ==
                expected_dynamic_shared_bytes;
        test.expect(exact, label + " is one exact production grid");
      };

  expect_exact_grid(gate_up, hidden_input, intermediate_output, 256U, true,
                    "NVFP4 Gate/Up C256 whole chunk");
  expect_exact_grid(gate_up, hidden_input, intermediate_output, 512U, true,
                    "NVFP4 Gate/Up C512 whole chunk");
  expect_exact_grid(down, intermediate_input, hidden_output, 256U, false,
                    "NVFP4 Down C256 whole chunk");
  expect_exact_grid(down, intermediate_input, hidden_output, 512U, false,
                    "NVFP4 Down C512 whole chunk");

  // Mirror the runner's existing ready/done event fork/join without exposing
  // a pair-specific runtime entry. Gate stays on the main stream, Up runs on
  // the auxiliary stream, and a post-wait marker proves that both independent
  // one-node branches rejoin before the next main-stream operation.
  const auto expect_gate_up_fork_join = [&](const std::size_t token_count) {
    const std::string label = "NVFP4 Gate/Up C" +
                              std::to_string(token_count) +
                              " production fork/join";
    const CapturedKernelChain direct = capture_ordered_kernel_chain(
        test,
        [&](cudaStream_t stream) noexcept {
          return q3x::kernels::
              launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
                  packed_weight, block_scale, 1.0F, hidden_input,
                  token_count, kIntermediateSize, kHiddenSize,
                  intermediate_output, static_cast<void*>(stream));
        },
        label + " direct branch");

    cudaStream_t main_stream = nullptr;
    cudaStream_t auxiliary_stream = nullptr;
    cudaEvent_t branch_ready = nullptr;
    cudaEvent_t branch_done = nullptr;
    cudaGraph_t graph = nullptr;
    DeviceBuffer<std::uint8_t> joined_marker;
    bool ready = direct.valid && direct.launches.size() == 1U;
    ready = ready &&
            joined_marker.allocate(test, 1U, "fork/join marker " + label);
    ready = ready && test.cuda_ok(
                         cudaStreamCreateWithFlags(&main_stream,
                                                   cudaStreamNonBlocking),
                         "create main stream " + label);
    ready = ready && test.cuda_ok(
                         cudaStreamCreateWithFlags(&auxiliary_stream,
                                                   cudaStreamNonBlocking),
                         "create auxiliary stream " + label);
    ready = ready && test.cuda_ok(
                         cudaEventCreateWithFlags(&branch_ready,
                                                  cudaEventDisableTiming),
                         "create ready event " + label);
    ready = ready && test.cuda_ok(
                         cudaEventCreateWithFlags(&branch_done,
                                                  cudaEventDisableTiming),
                         "create done event " + label);
    ready = ready && test.cuda_ok(
                         cudaStreamBeginCapture(main_stream,
                                                cudaStreamCaptureModeGlobal),
                         "begin fork/join capture " + label);
    if (ready) {
      ready = test.cuda_ok(cudaEventRecord(branch_ready, main_stream),
                           "record ready event " + label) &&
              test.cuda_ok(cudaStreamWaitEvent(auxiliary_stream,
                                               branch_ready, 0U),
                           "wait ready event " + label);
    }
    if (ready) {
      const int gate_status =
          runtime::launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly, gate_up,
              hidden_input, token_count, intermediate_output,
              static_cast<void*>(main_stream));
      const int up_status =
          runtime::launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly, gate_up,
              hidden_input, token_count,
              reinterpret_cast<std::uint16_t*>(0x68'0000'0000ULL),
              static_cast<void*>(auxiliary_stream));
      test.expect(static_cast<cudaError_t>(gate_status) == cudaSuccess &&
                      static_cast<cudaError_t>(up_status) == cudaSuccess,
                  label + " enqueues both exact branches");
      ready = gate_status == static_cast<int>(cudaSuccess) &&
              up_status == static_cast<int>(cudaSuccess);
    }
    if (ready) {
      ready = test.cuda_ok(cudaEventRecord(branch_done, auxiliary_stream),
                           "record done event " + label) &&
              test.cuda_ok(cudaStreamWaitEvent(main_stream, branch_done, 0U),
                           "wait done event " + label) &&
              test.cuda_ok(cudaMemsetAsync(joined_marker.get(), 0, 1U,
                                           main_stream),
                           "enqueue joined marker " + label);
    }
    if (main_stream != nullptr) {
      ready = test.cuda_ok(cudaStreamEndCapture(main_stream, &graph),
                           "end fork/join capture " + label) &&
              ready;
    }

    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaGraphNode_t> kernel_nodes;
    cudaGraphNode_t joined_marker_node = nullptr;
    if (ready && graph != nullptr) {
      std::size_t node_count = 0U;
      ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                           "count fork/join nodes " + label);
      nodes.resize(node_count);
      if (ready && node_count != 0U) {
        ready = test.cuda_ok(cudaGraphGetNodes(graph, nodes.data(),
                                               &node_count),
                             "read fork/join nodes " + label);
      }
      for (std::size_t index = 0U; ready && index < node_count; ++index) {
        cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
        ready = test.cuda_ok(cudaGraphNodeGetType(nodes[index], &type),
                             "read fork/join node type " + label);
        if (ready && type == cudaGraphNodeTypeKernel) {
          cudaKernelNodeParams parameters{};
          ready = test.cuda_ok(cudaGraphKernelNodeGetParams(
                                   nodes[index], &parameters),
                               "read fork/join kernel " + label);
          const bool c512 = token_count == 512U;
          const unsigned int expected_grid = static_cast<unsigned int>(
              (kIntermediateSize / (c512 ? 256U : 128U)) *
              (token_count / 128U));
          const unsigned int expected_dynamic_shared_bytes =
              c512 ? 118'784U : 0U;
          test.expect(
              ready && parameters.func == direct.launches.front().function &&
                  parameters.gridDim.x == expected_grid &&
                  parameters.gridDim.y == 1U &&
                  parameters.gridDim.z == 1U &&
                  parameters.blockDim.x == 256U &&
                  parameters.blockDim.y == 1U &&
                  parameters.blockDim.z == 1U &&
                  parameters.sharedMemBytes ==
                      expected_dynamic_shared_bytes,
              label +
                  " branch is the function-identical A-stationary node");
          kernel_nodes.push_back(nodes[index]);
        } else if (ready && type == cudaGraphNodeTypeMemset) {
          test.expect(joined_marker_node == nullptr,
                      label + " contains one post-join marker");
          joined_marker_node = nodes[index];
        }
      }
    }

    std::vector<cudaGraphNode_t> join_dependencies;
    if (ready && joined_marker_node != nullptr) {
      std::size_t dependency_count = 0U;
#if CUDART_VERSION >= 12030
      ready = test.cuda_ok(
          cudaGraphNodeGetDependencies(joined_marker_node, nullptr, nullptr,
                                       &dependency_count),
          "count fork/join marker dependencies " + label);
#else
      ready = test.cuda_ok(
          cudaGraphNodeGetDependencies(joined_marker_node, nullptr,
                                       &dependency_count),
          "count fork/join marker dependencies " + label);
#endif
      join_dependencies.resize(dependency_count);
      if (ready && dependency_count != 0U) {
#if CUDART_VERSION >= 12030
        std::vector<cudaGraphEdgeData> edge_data(dependency_count);
        ready = test.cuda_ok(
            cudaGraphNodeGetDependencies(
                joined_marker_node, join_dependencies.data(), edge_data.data(),
                &dependency_count),
            "read fork/join marker dependencies " + label);
#else
        ready = test.cuda_ok(
            cudaGraphNodeGetDependencies(joined_marker_node,
                                         join_dependencies.data(),
                                         &dependency_count),
            "read fork/join marker dependencies " + label);
#endif
      }
    }
    std::array<std::size_t, 2U> kernel_dependency_counts{
        std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<std::size_t>::max()};
    if (ready && kernel_nodes.size() == 2U) {
      for (std::size_t index = 0U; index < kernel_nodes.size(); ++index) {
#if CUDART_VERSION >= 12030
        ready = test.cuda_ok(
                    cudaGraphNodeGetDependencies(kernel_nodes[index], nullptr,
                                                 nullptr,
                                                 &kernel_dependency_counts[
                                                     index]),
                    "count fork/join kernel dependencies " + label) &&
                ready;
#else
        ready = test.cuda_ok(
                    cudaGraphNodeGetDependencies(
                        kernel_nodes[index], nullptr,
                        &kernel_dependency_counts[index]),
                    "count fork/join kernel dependencies " + label) &&
                ready;
#endif
      }
    }
    const bool independent_and_joined =
        ready && nodes.size() == 3U && kernel_nodes.size() == 2U &&
        kernel_dependency_counts[0U] == 0U &&
        kernel_dependency_counts[1U] == 0U &&
        joined_marker_node != nullptr && join_dependencies.size() == 2U &&
        std::find(join_dependencies.begin(), join_dependencies.end(),
                  kernel_nodes[0U]) != join_dependencies.end() &&
        std::find(join_dependencies.begin(), join_dependencies.end(),
                  kernel_nodes[1U]) != join_dependencies.end();
    test.expect(independent_and_joined,
                label + " contains two independent branches and one join");

    if (graph != nullptr) {
      (void)test.cuda_ok(cudaGraphDestroy(graph), "destroy graph " + label);
    }
    if (branch_done != nullptr) {
      (void)test.cuda_ok(cudaEventDestroy(branch_done),
                         "destroy done event " + label);
    }
    if (branch_ready != nullptr) {
      (void)test.cuda_ok(cudaEventDestroy(branch_ready),
                         "destroy ready event " + label);
    }
    if (auxiliary_stream != nullptr) {
      (void)test.cuda_ok(cudaStreamDestroy(auxiliary_stream),
                         "destroy auxiliary stream " + label);
    }
    if (main_stream != nullptr) {
      (void)test.cuda_ok(cudaStreamDestroy(main_stream),
                         "destroy main stream " + label);
    }
  };
  expect_gate_up_fork_join(256U);
  expect_gate_up_fork_join(512U);

  const auto expect_not_supported =
      [&](const runtime::ProjectionBackend backend,
          const runtime::LinearWeight& weight,
          const std::uint16_t* const input, const std::size_t token_count,
          std::uint16_t* const output, const std::string& label) {
        expect_not_supported_capture_has_no_nodes(
            test,
            [&](cudaStream_t stream) noexcept {
              return runtime::
                  launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
                      backend, weight, input, token_count, output,
                      static_cast<void*>(stream));
            },
            label);
      };
  expect_not_supported(runtime::ProjectionBackend::kReference, gate_up,
                       hidden_input, 256U, intermediate_output,
                       "whole chunk rejects the reference backend");
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly, gate_up,
                       hidden_input, 64U, intermediate_output,
                       "whole chunk rejects C64");
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly, gate_up,
                       hidden_input, 255U, intermediate_output,
                       "whole chunk rejects C255");
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly, gate_up,
                       hidden_input, 513U, intermediate_output,
                       "whole chunk rejects C513");

  const runtime::LinearWeight fp8_near_miss = runtime::Fp8LinearWeight{
      packed_weight, companion_scales, companion_scales + 1U, 1.0F, 1.0F,
      kIntermediateSize, kHiddenSize};
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly,
                       fp8_near_miss, hidden_input, 256U,
                       intermediate_output,
                       "whole chunk rejects a non-NVFP4 payload");
  const runtime::LinearWeight shape_near_miss = runtime::NvFp4LinearWeight{
      packed_weight, block_scale, companion_scales, companion_scales + 1U,
      1.0F, 1.0F, kIntermediateSize - 1U, kHiddenSize};
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly,
                       shape_near_miss, hidden_input, 256U,
                       intermediate_output,
                       "whole chunk rejects a Gate/Up N near miss");
  const runtime::LinearWeight weight_alignment_near_miss =
      runtime::NvFp4LinearWeight{
          packed_weight + 4U, block_scale, companion_scales,
          companion_scales + 1U, 1.0F, 1.0F, kIntermediateSize,
          kHiddenSize};
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly,
                       weight_alignment_near_miss, hidden_input, 256U,
                       intermediate_output,
                       "whole chunk rejects a 4-byte packed-weight alignment");
  const runtime::LinearWeight scale_alignment_near_miss =
      runtime::NvFp4LinearWeight{
          packed_weight, block_scale + 1U, companion_scales,
          companion_scales + 1U, 1.0F, 1.0F, kIntermediateSize,
          kHiddenSize};
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly,
                       scale_alignment_near_miss, hidden_input, 256U,
                       intermediate_output,
                       "whole chunk rejects a byte-aligned block scale");
  const auto* const input_alignment_near_miss =
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<std::uintptr_t>(hidden_input) + 2U);
  expect_not_supported(runtime::ProjectionBackend::kSm87WeightOnly, gate_up,
                       input_alignment_near_miss, 256U,
                       intermediate_output,
                       "whole chunk rejects a 2-byte BF16 input alignment");

  const runtime::LinearWeight missing_payload = runtime::NvFp4LinearWeight{
      nullptr, block_scale, companion_scales, companion_scales + 1U, 1.0F,
      1.0F, kIntermediateSize, kHiddenSize};
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, missing_payload,
            hidden_input, 256U, intermediate_output,
            static_cast<void*>(stream));
      },
      "whole chunk rejects a missing packed-weight payload");
  const runtime::LinearWeight missing_companion =
      runtime::NvFp4LinearWeight{
          packed_weight, block_scale, nullptr, companion_scales + 1U, 1.0F,
          1.0F, kIntermediateSize, kHiddenSize};
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, missing_companion,
            hidden_input, 256U, intermediate_output,
            static_cast<void*>(stream));
      },
      "whole chunk rejects a missing companion scale");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, gate_up,
            hidden_input, 256U,
            const_cast<std::uint16_t*>(hidden_input),
            static_cast<void*>(stream));
      },
      "whole chunk rejects an input/output alias across the full chunk");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, gate_up,
            hidden_input, 256U,
            reinterpret_cast<std::uint16_t*>(
                const_cast<float*>(companion_scales)),
            static_cast<void*>(stream));
      },
      "whole chunk rejects output over a companion scale");

  std::size_t fallback_total_nodes = 0U;
  bool fallback_linear_chain = false;
  const std::size_t fallback_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        int status =
            runtime::launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly,
                shape_near_miss, hidden_input, 256U, intermediate_output,
                static_cast<void*>(stream));
        if (static_cast<cudaError_t>(status) != cudaErrorNotSupported) {
          return status;
        }
        for (std::size_t offset = 0U; offset < 256U;
             offset += runtime::kMaximumProjectionTileTokenCount) {
          status = runtime::launch_projection_tile_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly,
              shape_near_miss, hidden_input + offset * kHiddenSize,
              runtime::kMaximumProjectionTileTokenCount, nullptr, 0U,
              intermediate_output + offset * (kIntermediateSize - 1U),
              static_cast<void*>(stream));
          if (status != static_cast<int>(cudaSuccess)) {
            return status;
          }
        }
        return static_cast<int>(cudaSuccess);
      },
      "NVFP4 C256 near-miss whole-chunk fallback graph",
      &fallback_total_nodes, &fallback_linear_chain);
  test.expect(fallback_kernel_nodes == 32U && fallback_total_nodes == 32U &&
                  fallback_linear_chain,
              "whole-chunk near miss retains four ordered generic C64 "
              "fallback schedules");
}

void test_bf16_projection_pair_dispatch(TestContext& test) {
  constexpr std::size_t kTokens = runtime::kMaximumProjectionTileTokenCount;
  constexpr std::size_t kRows = 48U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::uint16_t kBf16One = 0x3f80U;
  constexpr std::uint16_t kBf16Half = 0x3f00U;
  constexpr std::uint16_t kFirstExpected = 0x45a0U;
  constexpr std::uint16_t kSecondExpected = 0x4520U;
  constexpr std::uint16_t kSiluMulExpected = 0x4b48U;

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

  const auto capture_production_pair = [&](const std::size_t token_count,
                                           const std::string& label) {
    return capture_ordered_kernel_chain(
        test,
        [&](cudaStream_t stream) noexcept {
          return runtime::launch_projection_pair_tile_to_bf16_cuda(
              runtime::ProjectionBackend::kSm87WeightOnly, first, second,
              activation.get(), token_count, nullptr, 0U,
              first_output.get(), second_output.get(),
              static_cast<void*>(stream));
        },
        label);
  };
  const CapturedKernelChain generic_oracle = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
            first_weights.get(), second_weights.get(), activation.get(), 1U,
            kRows, kColumns, first_output.get(), second_output.get(),
            static_cast<void*>(stream));
      },
      "BF16 generic pair identity oracle");
  const CapturedKernelChain exact_oracle = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return q3x::kernels::
            launch_bf16_gemv_pair_m16_projection_fused_cuda(
                first_weights.get(), second_weights.get(), activation.get(),
                first_output.get(), second_output.get(),
                static_cast<void*>(stream));
      },
      "BF16 exact M16 pair identity oracle");
  test.expect(generic_oracle.valid && generic_oracle.launches.size() == 1U,
              "BF16 generic pair identity oracle is one kernel");
  test.expect(exact_oracle.valid && exact_oracle.launches.size() == 1U,
              "BF16 exact M16 pair identity oracle is one kernel");
  if (generic_oracle.valid && generic_oracle.launches.size() == 1U &&
      exact_oracle.valid && exact_oracle.launches.size() == 1U) {
    void* const generic_function =
        generic_oracle.launches.front().function;
    void* const exact_function = exact_oracle.launches.front().function;
    test.expect(generic_function != nullptr && exact_function != nullptr &&
                    generic_function != exact_function,
                "BF16 exact M16 function is distinct from generic pair");
    const auto expect_launch = [&](const CapturedKernelLaunch& launch,
                                   void* const function,
                                   const unsigned int grid_y,
                                   const unsigned int grid_z,
                                   const std::string& label) {
      test.expect(launch.function == function,
                  label + " preserves kernel function identity");
      test.expect(launch.grid.x == kRows && launch.grid.y == grid_y &&
                      launch.grid.z == grid_z,
                  label + " preserves grid identity");
      test.expect(launch.block.x == 256U && launch.block.y == 1U &&
                      launch.block.z == 1U,
                  label + " preserves block identity");
      test.expect(launch.dynamic_shared_bytes == 0U,
                  label + " preserves zero dynamic shared memory");
    };
    const CapturedKernelChain m1 =
        capture_production_pair(1U, "SM87 BF16 pair M1 identity graph");
    const CapturedKernelChain m15 =
        capture_production_pair(15U, "SM87 BF16 pair M15 identity graph");
    const CapturedKernelChain m16 =
        capture_production_pair(16U, "SM87 BF16 pair M16 identity graph");
    const CapturedKernelChain m17 =
        capture_production_pair(17U, "SM87 BF16 pair M17 identity graph");
    const CapturedKernelChain m32 =
        capture_production_pair(32U, "SM87 BF16 pair M32 identity graph");
    const CapturedKernelChain m64 =
        capture_production_pair(64U, "SM87 BF16 pair M64 identity graph");
    test.expect(m1.valid && m1.launches.size() == 1U,
                "SM87 BF16 pair M1 remains one generic kernel");
    test.expect(m15.valid && m15.launches.size() == 1U,
                "SM87 BF16 pair M15 remains one generic kernel");
    test.expect(m16.valid && m16.launches.size() == 1U,
                "SM87 BF16 pair M16 selects one exact kernel");
    test.expect(m17.valid && m17.launches.size() == 2U,
                "SM87 BF16 pair M17 is an ordered exact-plus-generic chain");
    test.expect(m32.valid && m32.launches.size() == 2U,
                "SM87 BF16 pair M32 is an ordered two-exact chain");
    test.expect(m64.valid && m64.launches.size() == 4U,
                "SM87 BF16 pair M64 is an ordered four-exact chain");
    if (m1.valid && m1.launches.size() == 1U) {
      expect_launch(m1.launches[0], generic_oracle.launches[0].function, 1U,
                    2U, "SM87 BF16 pair M1 generic launch");
    }
    if (m15.valid && m15.launches.size() == 1U) {
      expect_launch(m15.launches[0], generic_oracle.launches[0].function, 15U,
                    2U, "SM87 BF16 pair M15 generic launch");
    }
    if (m16.valid && m16.launches.size() == 1U) {
      expect_launch(m16.launches[0], exact_oracle.launches[0].function, 1U,
                    1U, "SM87 BF16 pair M16 exact launch");
    }
    if (m17.valid && m17.launches.size() == 2U) {
      expect_launch(m17.launches[0], exact_oracle.launches[0].function, 1U,
                    1U, "SM87 BF16 pair M17 exact prefix");
      expect_launch(m17.launches[1], generic_oracle.launches[0].function, 1U,
                    2U, "SM87 BF16 pair M17 generic tail");
    }
    if (m32.valid && m32.launches.size() == 2U) {
      expect_launch(m32.launches[0], exact_oracle.launches[0].function, 1U,
                    1U, "SM87 BF16 pair M32 first exact tile");
      expect_launch(m32.launches[1], exact_oracle.launches[0].function, 1U,
                    1U, "SM87 BF16 pair M32 second exact tile");
    }
    if (m64.valid && m64.launches.size() == 4U) {
      for (std::size_t index = 0U; index < m64.launches.size(); ++index) {
        expect_launch(m64.launches[index],
                      exact_oracle.launches[0].function, 1U, 1U,
                      "SM87 BF16 pair M64 exact tile " +
                          std::to_string(index));
      }
    }
    std::cout << "GRAPH_BF16_PAIR_DISPATCH: M1=generic(48x1x2) "
                 "M15=generic(48x15x2) M16=exact(48x1x1) "
                 "M17=exact+generic M32=exact+exact "
                 "M64=4xexact block=256 shared=0\n";
  }

  const auto run_fast = [&](const std::size_t token_count,
                            const std::string& label) {
    bool ready = test.cuda_ok(
        cudaMemset(first_output.get(), 0xa5,
                   kTokens * kRows * sizeof(std::uint16_t)),
        "initialize first pair output canary " + label);
    ready = ready && test.cuda_ok(
                         cudaMemset(second_output.get(), 0xa5,
                                    kTokens * kRows * sizeof(std::uint16_t)),
                         "initialize second pair output canary " + label);
    if (!ready) {
      return;
    }
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
      const std::size_t tail_elements = (kTokens - token_count) * kRows;
      std::vector<std::uint16_t> first_tail(tail_elements);
      std::vector<std::uint16_t> second_tail(tail_elements);
      if (tail_elements != 0U) {
        ready = test.cuda_ok(
            cudaMemcpy(first_tail.data(),
                       first_output.get() + token_count * kRows,
                       tail_elements * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost),
            "download first pair output canary " + label);
        ready = ready && test.cuda_ok(
                             cudaMemcpy(second_tail.data(),
                                        second_output.get() +
                                            token_count * kRows,
                                        tail_elements * sizeof(std::uint16_t),
                                        cudaMemcpyDeviceToHost),
                             "download second pair output canary " + label);
      }
      if (ready) {
        test.expect(std::all_of(first_tail.begin(), first_tail.end(),
                                [](const std::uint16_t value) noexcept {
                                  return value == 0xa5a5U;
                                }) &&
                        std::all_of(second_tail.begin(), second_tail.end(),
                                    [](const std::uint16_t value) noexcept {
                                      return value == 0xa5a5U;
                                    }),
                    label + " preserves both output tails");
      }
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
    const std::size_t expected_nodes = (token_count + 15U) / 16U;
    test.expect(total_nodes == expected_nodes &&
                    kernel_nodes == expected_nodes,
                label + " graph contains one fused kernel per subtile");
  };
  run_fast(1U, "SM87 production BF16 pair M1");
  run_fast(15U, "SM87 production BF16 pair M15");
  run_fast(16U, "SM87 production BF16 pair M16");
  run_fast(17U, "SM87 production BF16 pair M17");
  run_fast(18U, "SM87 production BF16 pair M18");
  run_fast(24U, "SM87 production BF16 pair M24");
  run_fast(31U, "SM87 production BF16 pair M31");
  run_fast(32U, "SM87 production BF16 pair M32");
  run_fast(33U, "SM87 production BF16 pair M33");
  run_fast(63U, "SM87 production BF16 pair M63");
  run_fast(64U, "SM87 production BF16 pair M64");

  const int mlp_status = runtime::launch_mlp_gate_up_silu_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, first, second,
      activation.get(), nullptr, 0U, first_output.get(), second_output.get());
  test.expect(static_cast<cudaError_t>(mlp_status) == cudaSuccess,
              "SM87 BF16 MLP pair accepts null unused scratch");
  if (static_cast<cudaError_t>(mlp_status) == cudaSuccess) {
    (void)expect_tile_output(
        test, first_output, 1U, kRows,
        std::vector<std::uint16_t>{kSiluMulExpected},
        "SM87 BF16 MLP pair SiLU-multiply gate output");
    (void)expect_tile_output(
        test, second_output, 1U, kRows,
        std::vector<std::uint16_t>{kSecondExpected},
        "SM87 BF16 MLP pair retained up output");
  }

  std::size_t mlp_total_nodes = 0U;
  const std::size_t mlp_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_mlp_gate_up_silu_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, first, second,
            activation.get(), nullptr, 0U, first_output.get(),
            second_output.get(), static_cast<void*>(stream));
      },
      "SM87 BF16 MLP pair M1 graph", &mlp_total_nodes);
  test.expect(mlp_total_nodes == 2U && mlp_kernel_nodes == 2U,
              "SM87 BF16 MLP pair graph contains fused projections and SiLU");

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
                 "pair rejects M=65");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, first, second,
            activation.get(), kTokens + 1U, nullptr, 0U,
            first_output.get(), second_output.get(),
            static_cast<void*>(stream));
      },
      "SM87 BF16 pair M65 guard");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, first, second,
            activation.get(), kTokens, nullptr, 0U, first_output.get(),
            first_output.get() + 16U * kRows, static_cast<void*>(stream));
      },
      "SM87 BF16 pair C64 cross-subtile output alias");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_projection_pair_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, first, null_second,
            activation.get(), kTokens, scratch.get(), kRows,
            first_output.get(), second_output.get(),
            static_cast<void*>(stream));
      },
      "SM87 BF16 pair C64 invalid second weight");
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
  constexpr std::size_t kAbRows = 48U;
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
  DeviceBuffer<std::uint16_t> a_weights;
  DeviceBuffer<std::uint16_t> b_weights;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<float> scratch;
  DeviceBuffer<std::uint16_t> qkv_output;
  DeviceBuffer<std::uint16_t> z_output;
  DeviceBuffer<std::uint16_t> a_output;
  DeviceBuffer<std::uint16_t> b_output;
  bool ready = qkv_weights.allocate(
      test, kQkvRows * kColumns, "QKV/Z QKV FP8 weights");
  ready = ready && z_weights.allocate(
                       test, kZRows * kColumns, "QKV/Z Z FP8 weights");
  ready = ready && misaligned_activation_storage.allocate(
                       test, 2U * kColumns * sizeof(std::uint16_t) + 2U,
                       "QKV/Z misaligned activation storage");
  ready = ready && activation.allocate(
                       test, 2U * kColumns, "QKV/Z FP8 activations");
  ready = ready && a_weights.allocate(
                       test, kAbRows * kColumns + 1U,
                       "QKV/Z/A/B A weights");
  ready = ready && b_weights.allocate(
                       test, kAbRows * kColumns, "QKV/Z/A/B B weights");
  ready = ready && companion_scales.allocate(
                       test, kQkvRows / 2U + 4U,
                       "QKV/Z FP8 companion scales");
  ready = ready && scratch.allocate(
                       test, kQkvRows, "QKV/Z FP8 reference scratch");
  ready = ready && qkv_output.allocate(
                       test, 2U * kQkvRows, "QKV/Z QKV output");
  ready = ready && z_output.allocate(
                       test, 2U * kZRows, "QKV/Z Z output");
  ready = ready && a_output.allocate(test, kAbRows, "QKV/Z/A/B A output");
  ready = ready && b_output.allocate(test, kAbRows, "QKV/Z/A/B B output");
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
  const runtime::LinearWeight a = runtime::Bf16LinearWeight{
      a_weights.get(), kAbRows, kColumns};
  const runtime::LinearWeight b = runtime::Bf16LinearWeight{
      b_weights.get(), kAbRows, kColumns};
  test.expect(runtime::supports_fp8_qkv_z_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, qkv, z),
              "production FP8 QKV/Z pair selects the SM87 fast path");

  std::size_t composite_total_nodes = 0U;
  const std::size_t composite_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_linear_attention_qkv_z_ab_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, z, a, b,
            activation.get(), qkv_output.get(), z_output.get(),
            a_output.get(), b_output.get(), static_cast<void*>(stream));
      },
      "SM87 production linear-attention QKV/Z/A/B graph",
      &composite_total_nodes);
  test.expect(composite_total_nodes == 1U &&
                  composite_kernel_nodes == 1U,
              "linear-attention QKV/Z/A/B graph contains one fused kernel");

  int status = runtime::launch_linear_attention_qkv_z_ab_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, qkv, z, a, b,
      misaligned_activation, qkv_output.get(), z_output.get(), a_output.get(),
      b_output.get());
  test.expect(static_cast<cudaError_t>(status) == cudaErrorNotSupported,
              "unaligned linear-attention QKV/Z/A/B returns not supported");

  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_linear_attention_qkv_z_ab_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, z, a, b,
            misaligned_activation, a_weights.get(), z_output.get(),
            a_output.get(), b_output.get(), static_cast<void*>(stream));
      },
      "unaligned linear-attention QKV/Z/A/B with QKV output over A weight");

  const auto* const odd_byte_a_weights =
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<const std::uint8_t*>(a_weights.get()) + 1U);
  const runtime::LinearWeight misaligned_a = runtime::Bf16LinearWeight{
      odd_byte_a_weights, kAbRows, kColumns};
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_linear_attention_qkv_z_ab_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, z,
            misaligned_a, b, activation.get(), qkv_output.get(),
            z_output.get(), a_output.get(), b_output.get(),
            static_cast<void*>(stream));
      },
      "linear-attention QKV/Z/A/B odd-byte A weight");

  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_linear_attention_qkv_z_ab_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, z, a, b,
            activation.get(),
            reinterpret_cast<std::uint16_t*>(companion_scales.get()),
            z_output.get(), a_output.get(), b_output.get(),
            static_cast<void*>(stream));
      },
      "linear-attention QKV output over QKV weight scale");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_linear_attention_qkv_z_ab_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, z, a, b,
            activation.get(), qkv_output.get(),
            reinterpret_cast<std::uint16_t*>(companion_scales.get() + 1U),
            a_output.get(), b_output.get(), static_cast<void*>(stream));
      },
      "linear-attention Z output over QKV input scale");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_linear_attention_qkv_z_ab_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, z, a, b,
            activation.get(), qkv_output.get(), z_output.get(),
            reinterpret_cast<std::uint16_t*>(companion_scales.get() + 2U),
            b_output.get(), static_cast<void*>(stream));
      },
      "linear-attention A output over Z weight scale");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_linear_attention_qkv_z_ab_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, z, a, b,
            activation.get(), qkv_output.get(), z_output.get(),
            a_output.get(), reinterpret_cast<std::uint16_t*>(
                                companion_scales.get() + 3U),
            static_cast<void*>(stream));
      },
      "linear-attention B output over Z input scale");

  const runtime::LinearWeight missing_a_payload =
      runtime::Bf16LinearWeight{nullptr, kAbRows, kColumns};
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_linear_attention_qkv_z_ab_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv, z,
            missing_a_payload, b, activation.get(), qkv_output.get(),
            z_output.get(), a_output.get(), b_output.get(),
            static_cast<void*>(stream));
      },
      "linear-attention QKV/Z/A/B missing A payload");

  const runtime::LinearWeight missing_z_scale = runtime::Fp8LinearWeight{
      z_weights.get(), nullptr, companion_scales.get() + 3U,
      kZWeightScale, 1.0F, kZRows, kColumns};
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_linear_attention_qkv_z_ab_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, qkv,
            missing_z_scale, a, b, activation.get(), qkv_output.get(),
            z_output.get(), a_output.get(), b_output.get(),
            static_cast<void*>(stream));
      },
      "linear-attention QKV/Z/A/B missing Z scale payload");

  status = runtime::launch_projection_pair_tile_to_bf16_cuda(
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

void test_fp8_full_attention_q_kv_dispatch(TestContext& test) {
  constexpr std::size_t kQRows = 12'288U;
  constexpr std::size_t kKvRows = 1'024U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr float kQScale = 1.0F / 64.0F;
  constexpr float kKeyScale = 1.0F / 96.0F;
  constexpr float kValueScale = 1.0F / 128.0F;

  const auto* const fake_q_weights =
      reinterpret_cast<const std::uint8_t*>(0x10'0000'0000ULL);
  const auto* const fake_key_weights =
      reinterpret_cast<const std::uint8_t*>(0x20'0000'0000ULL);
  const auto* const fake_value_weights =
      reinterpret_cast<const std::uint8_t*>(0x30'0000'0000ULL);
  const auto* const fake_unaligned_value_weights = fake_value_weights + 1U;
  const auto* const fake_q_weight_scale =
      reinterpret_cast<const float*>(0x40'0000'0000ULL);
  const auto* const fake_q_input_scale =
      reinterpret_cast<const float*>(0x40'0000'0100ULL);
  const auto* const fake_key_weight_scale =
      reinterpret_cast<const float*>(0x41'0000'0000ULL);
  const auto* const fake_key_input_scale =
      reinterpret_cast<const float*>(0x41'0000'0100ULL);
  const auto* const fake_value_weight_scale =
      reinterpret_cast<const float*>(0x42'0000'0000ULL);
  const auto* const fake_value_input_scale =
      reinterpret_cast<const float*>(0x42'0000'0100ULL);
  const auto* const fake_input =
      reinterpret_cast<const std::uint16_t*>(0x50'0000'0000ULL);
  auto* const fake_q_output =
      reinterpret_cast<std::uint16_t*>(0x60'0000'0000ULL);
  auto* const fake_key_output =
      reinterpret_cast<std::uint16_t*>(0x70'0000'0000ULL);
  auto* const fake_value_output =
      reinterpret_cast<std::uint16_t*>(0x80'0000'0000ULL);
  auto* const fake_scratch =
      reinterpret_cast<float*>(0x90'0000'0000ULL);

  const runtime::LinearWeight q = runtime::Fp8LinearWeight{
      fake_q_weights, fake_q_weight_scale, fake_q_input_scale, kQScale,
      1.0F, kQRows, kColumns};
  const runtime::LinearWeight key = runtime::Fp8LinearWeight{
      fake_key_weights, fake_key_weight_scale, fake_key_input_scale,
      kKeyScale, 1.0F, kKvRows, kColumns};
  const runtime::LinearWeight value = runtime::Fp8LinearWeight{
      fake_value_weights, fake_value_weight_scale, fake_value_input_scale,
      kValueScale, 1.0F, kKvRows, kColumns};
  const runtime::LinearWeight unaligned_value = runtime::Fp8LinearWeight{
      fake_unaligned_value_weights, fake_value_weight_scale,
      fake_value_input_scale, kValueScale, 1.0F, kKvRows, kColumns};
  const runtime::LinearWeight near_q = runtime::Fp8LinearWeight{
      fake_q_weights, fake_q_weight_scale, fake_q_input_scale, kQScale,
      1.0F, kQRows - 1U, kColumns};
  const runtime::LinearWeight near_key = runtime::Fp8LinearWeight{
      fake_key_weights, fake_key_weight_scale, fake_key_input_scale,
      kKeyScale, 1.0F, kKvRows - 1U, kColumns};
  const runtime::LinearWeight near_value_columns =
      runtime::Fp8LinearWeight{
          fake_value_weights, fake_value_weight_scale,
          fake_value_input_scale, kValueScale, 1.0F, kKvRows,
          kColumns - 1U};
  const runtime::LinearWeight bf16_q = runtime::Bf16LinearWeight{
      reinterpret_cast<const std::uint16_t*>(fake_q_weights), kQRows,
      kColumns};
  const runtime::LinearWeight malformed_value = runtime::Fp8LinearWeight{
      fake_value_weights, nullptr, fake_value_input_scale, kValueScale,
      1.0F, kKvRows, kColumns};

  test.expect(runtime::supports_fp8_q_kv_projection_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, q, key,
                  value),
              "production full-attention Q/K/V selects the SM87 fusion");
  test.expect(runtime::supports_fp8_q_kv_projection_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, q, key,
                  unaligned_value),
              "full-attention selector is independent of launch alignment");
  test.expect(!runtime::supports_fp8_q_kv_projection_fusion(
                  runtime::ProjectionBackend::kReference, q, key, value),
              "reference backend never selects full-attention fusion");
  test.expect(!runtime::supports_fp8_q_kv_projection_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, key, q,
                  value),
              "full-attention selector requires Q before K/V");
  test.expect(!runtime::supports_fp8_q_kv_projection_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, near_q, key,
                  value),
              "full-attention selector rejects a near-miss Q row count");
  test.expect(!runtime::supports_fp8_q_kv_projection_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, q, near_key,
                  value),
              "full-attention selector rejects a near-miss K row count");
  test.expect(!runtime::supports_fp8_q_kv_projection_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, q, key,
                  near_value_columns),
              "full-attention selector rejects a near-miss V input size");
  test.expect(!runtime::supports_fp8_q_kv_projection_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, bf16_q, key,
                  value),
              "full-attention selector rejects a non-FP8 Q type");
  test.expect(!runtime::supports_fp8_q_kv_projection_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, q, key,
                  malformed_value),
              "full-attention selector validates the third payload");

  bool exact_linear = false;
  std::size_t exact_total_nodes = 0U;
  const std::size_t exact_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_full_attention_q_kv_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, q, key, value,
            fake_input, nullptr, 0U, fake_q_output, fake_key_output,
            fake_value_output, static_cast<void*>(stream));
      },
      "SM87 exact full-attention Q/K/V graph", &exact_total_nodes,
      &exact_linear);
  test.expect(exact_total_nodes == 1U && exact_kernel_nodes == 1U &&
                  exact_linear,
              "exact aligned full-attention Q/K/V uses one kernel");

  bool unaligned_linear = false;
  std::size_t unaligned_total_nodes = 0U;
  const std::size_t unaligned_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_full_attention_q_kv_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, q, key,
            unaligned_value, fake_input, nullptr, 0U, fake_q_output,
            fake_key_output, fake_value_output,
            static_cast<void*>(stream));
      },
      "SM87 unaligned full-attention Q/K/V graph",
      &unaligned_total_nodes, &unaligned_linear);
  test.expect(unaligned_total_nodes == 3U &&
                  unaligned_kernel_nodes == 3U && unaligned_linear,
              "unaligned V preserves ordered Q then split K/V kernels");

  bool near_key_linear = false;
  std::size_t near_key_total_nodes = 0U;
  const std::size_t near_key_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_full_attention_q_kv_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, q, near_key,
            value, fake_input, nullptr, 0U, fake_q_output, fake_key_output,
            fake_value_output, static_cast<void*>(stream));
      },
      "SM87 near-K full-attention Q/K/V graph", &near_key_total_nodes,
      &near_key_linear);
  test.expect(near_key_total_nodes == 3U && near_key_kernel_nodes == 3U &&
                  near_key_linear,
              "near-miss K preserves ordered Q then split K/V kernels");

  bool near_q_linear = false;
  std::size_t near_q_total_nodes = 0U;
  const std::size_t near_q_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_full_attention_q_kv_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, near_q, key,
            value, fake_input, nullptr, 0U, fake_q_output, fake_key_output,
            fake_value_output, static_cast<void*>(stream));
      },
      "SM87 near-Q full-attention Q/K/V graph", &near_q_total_nodes,
      &near_q_linear);
  test.expect(near_q_total_nodes == 2U && near_q_kernel_nodes == 2U &&
                  near_q_linear,
              "near-miss Q preserves ordered Q then fused K/V kernels");

  bool reference_linear = false;
  std::size_t reference_total_nodes = 0U;
  const std::size_t reference_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_full_attention_q_kv_to_bf16_cuda(
            runtime::ProjectionBackend::kReference, q, key, value,
            fake_input, fake_scratch, kQRows, fake_q_output,
            fake_key_output, fake_value_output,
            static_cast<void*>(stream));
      },
      "reference full-attention Q/K/V graph", &reference_total_nodes,
      &reference_linear);
  test.expect(reference_total_nodes == 6U &&
                  reference_kernel_nodes == 6U && reference_linear,
              "reference full-attention preserves ordered three "
              "GEMV-plus-convert paths");

  const auto expect_invalid =
      [&](const runtime::ProjectionBackend backend,
          const runtime::LinearWeight& q_weight,
          const runtime::LinearWeight& key_weight,
          const runtime::LinearWeight& value_weight,
          const std::uint16_t* const input, float* const scratch,
          const std::size_t scratch_elements,
          std::uint16_t* const q_output,
          std::uint16_t* const key_output,
          std::uint16_t* const value_output,
          const std::string& label) {
        expect_invalid_capture_has_no_nodes(
            test,
            [&](cudaStream_t stream) noexcept {
              return runtime::launch_full_attention_q_kv_to_bf16_cuda(
                  backend, q_weight, key_weight, value_weight, input,
                  scratch, scratch_elements, q_output, key_output,
                  value_output, static_cast<void*>(stream));
            },
            label);
      };
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, q, key, malformed_value,
      fake_input, nullptr, 0U, fake_q_output, fake_key_output,
      fake_value_output,
      "full-attention validates invalid third weight before Q enqueue");
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, q, key, value,
      fake_input, nullptr, 0U, fake_q_output, fake_q_output,
      fake_value_output,
      "full-attention rejects Q/K output alias before enqueue");
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, q, key, value,
      fake_input, nullptr, 0U, fake_q_output, fake_key_output,
      fake_q_output,
      "full-attention rejects Q/V output alias before enqueue");
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, q, key, value,
      fake_input, nullptr, 0U, fake_q_output, fake_key_output,
      fake_key_output,
      "full-attention rejects K/V output alias before enqueue");
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, q, key, value,
      fake_input, nullptr, 0U,
      reinterpret_cast<std::uint16_t*>(
          const_cast<std::uint8_t*>(fake_value_weights)),
      fake_key_output, fake_value_output,
      "full-attention rejects Q output over V weights before enqueue");
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, q, key, value,
      fake_input, nullptr, 0U, fake_q_output,
      reinterpret_cast<std::uint16_t*>(
          const_cast<std::uint8_t*>(fake_q_weights)),
      fake_value_output,
      "full-attention rejects K output over Q weights before enqueue");
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, q, key, value,
      fake_input, nullptr, 0U, fake_q_output, fake_key_output,
      reinterpret_cast<std::uint16_t*>(
          const_cast<std::uint8_t*>(fake_key_weights)),
      "full-attention rejects V output over K weights before enqueue");
  expect_invalid(
      runtime::ProjectionBackend::kSm87WeightOnly, q, key, value,
      fake_input, nullptr, 0U,
      reinterpret_cast<std::uint16_t*>(
          const_cast<float*>(fake_value_weight_scale)),
      fake_key_output, fake_value_output,
      "full-attention rejects Q output over V scalar before enqueue");
  expect_invalid(
      runtime::ProjectionBackend::kReference, q, key, value, fake_input,
      const_cast<float*>(fake_value_weight_scale), kQRows, fake_q_output,
      fake_key_output, fake_value_output,
      "full-attention rejects scratch over V scalar before enqueue");
  expect_invalid(
      runtime::ProjectionBackend::kReference, q, key, value, fake_input,
      reinterpret_cast<float*>(fake_q_output), kQRows, fake_q_output,
      fake_key_output, fake_value_output,
      "full-attention rejects scratch over Q output before enqueue");
  expect_invalid(
      runtime::ProjectionBackend::kReference, q, key, value, fake_input,
      fake_scratch, kQRows - 1U, fake_q_output, fake_key_output,
      fake_value_output,
      "full-attention rejects insufficient scratch before enqueue");

  constexpr std::size_t kSmallQRows = 4U;
  constexpr std::size_t kSmallKvRows = 2U;
  constexpr std::size_t kSmallColumns = 32U;
  DeviceBuffer<std::uint8_t> small_q_weights;
  DeviceBuffer<std::uint8_t> small_key_weights;
  DeviceBuffer<std::uint8_t> small_value_weights;
  DeviceBuffer<float> small_scales;
  DeviceBuffer<std::uint16_t> small_input;
  DeviceBuffer<std::uint16_t> baseline_q;
  DeviceBuffer<std::uint16_t> baseline_key;
  DeviceBuffer<std::uint16_t> baseline_value;
  DeviceBuffer<std::uint16_t> candidate_q;
  DeviceBuffer<std::uint16_t> candidate_key;
  DeviceBuffer<std::uint16_t> candidate_value;
  bool ready = small_q_weights.allocate(
      test, kSmallQRows * kSmallColumns,
      "small full-attention Q weights");
  ready = ready && small_key_weights.allocate(
                       test, kSmallKvRows * kSmallColumns,
                       "small full-attention K weights");
  ready = ready && small_value_weights.allocate(
                       test, kSmallKvRows * kSmallColumns,
                       "small full-attention V weights");
  ready = ready && small_scales.allocate(
                       test, 6U, "small full-attention scales");
  ready = ready && small_input.allocate(
                       test, kSmallColumns, "small full-attention input");
  ready = ready && baseline_q.allocate(
                       test, kSmallQRows,
                       "small full-attention baseline Q output");
  ready = ready && baseline_key.allocate(
                       test, kSmallKvRows,
                       "small full-attention baseline K output");
  ready = ready && baseline_value.allocate(
                       test, kSmallKvRows,
                       "small full-attention baseline V output");
  ready = ready && candidate_q.allocate(
                       test, kSmallQRows,
                       "small full-attention candidate Q output");
  ready = ready && candidate_key.allocate(
                       test, kSmallKvRows,
                       "small full-attention candidate K output");
  ready = ready && candidate_value.allocate(
                       test, kSmallKvRows,
                       "small full-attention candidate V output");
  if (!ready) {
    return;
  }

  constexpr std::array<std::uint8_t, 8U> kQPattern{
      0x30U, 0x34U, 0x38U, 0xb0U, 0xb4U, 0x28U, 0x3cU, 0xa8U};
  constexpr std::array<std::uint8_t, 8U> kKeyPattern{
      0x38U, 0x30U, 0xb4U, 0x28U, 0x34U, 0xb0U, 0x3cU, 0xa8U};
  constexpr std::array<std::uint8_t, 8U> kValuePattern{
      0xb8U, 0x34U, 0x30U, 0xa8U, 0x38U, 0xb4U, 0x28U, 0x3cU};
  std::vector<std::uint8_t> host_small_q(kSmallQRows * kSmallColumns);
  std::vector<std::uint8_t> host_small_key(kSmallKvRows * kSmallColumns);
  std::vector<std::uint8_t> host_small_value(kSmallKvRows * kSmallColumns);
  for (std::size_t index = 0U; index < host_small_q.size(); ++index) {
    host_small_q[index] =
        kQPattern[(index + index / kSmallColumns) % kQPattern.size()];
  }
  for (std::size_t index = 0U; index < host_small_key.size(); ++index) {
    host_small_key[index] =
        kKeyPattern[(3U * index + index / kSmallColumns) %
                    kKeyPattern.size()];
    host_small_value[index] =
        kValuePattern[(5U * index + index / kSmallColumns) %
                      kValuePattern.size()];
  }
  constexpr std::array<std::uint16_t, 8U> kInputPattern{
      0x3f80U, 0xbf00U, 0x3e80U, 0x3f00U,
      0xbf80U, 0x3fc0U, 0xbe80U, 0x4000U};
  std::vector<std::uint16_t> host_small_input(kSmallColumns);
  for (std::size_t column = 0U; column < kSmallColumns; ++column) {
    host_small_input[column] =
        kInputPattern[column % kInputPattern.size()];
  }
  ready = upload(test, small_q_weights, host_small_q,
                 "small full-attention Q weights");
  ready = ready && upload(test, small_key_weights, host_small_key,
                          "small full-attention K weights");
  ready = ready && upload(test, small_value_weights, host_small_value,
                          "small full-attention V weights");
  ready = ready && upload(
                       test, small_scales,
                       std::vector<float>{1.0F / 16.0F, 1.0F,
                                          1.0F / 32.0F, 1.0F,
                                          1.0F / 64.0F, 1.0F},
                       "small full-attention scales");
  ready = ready && upload(test, small_input, host_small_input,
                          "small full-attention input");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight small_q = runtime::Fp8LinearWeight{
      small_q_weights.get(), small_scales.get(), small_scales.get() + 1U,
      1.0F / 16.0F, 1.0F, kSmallQRows, kSmallColumns};
  const runtime::LinearWeight small_key = runtime::Fp8LinearWeight{
      small_key_weights.get(), small_scales.get() + 2U,
      small_scales.get() + 3U, 1.0F / 32.0F, 1.0F, kSmallKvRows,
      kSmallColumns};
  const runtime::LinearWeight small_value = runtime::Fp8LinearWeight{
      small_value_weights.get(), small_scales.get() + 4U,
      small_scales.get() + 5U, 1.0F / 64.0F, 1.0F, kSmallKvRows,
      kSmallColumns};
  test.expect(!runtime::supports_fp8_q_kv_projection_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, small_q,
                  small_key, small_value),
              "small full-attention fixture exercises fallback");

  int status = runtime::launch_projection_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, small_q,
      small_input.get(), nullptr, 0U, baseline_q.get());
  if (static_cast<cudaError_t>(status) == cudaSuccess) {
    status = runtime::launch_projection_pair_tile_to_bf16_cuda(
        runtime::ProjectionBackend::kSm87WeightOnly, small_key,
        small_value, small_input.get(), 1U, nullptr, 0U,
        baseline_key.get(), baseline_value.get());
  }
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "small old full-attention Q then K/V chain succeeds");
  status = runtime::launch_full_attention_q_kv_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, small_q, small_key,
      small_value, small_input.get(), nullptr, 0U, candidate_q.get(),
      candidate_key.get(), candidate_value.get());
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "small full-attention composite fallback succeeds");
  if (static_cast<cudaError_t>(status) != cudaSuccess ||
      !test.cuda_ok(cudaDeviceSynchronize(),
                    "synchronize small full-attention fallback")) {
    return;
  }

  std::vector<std::uint16_t> host_baseline_q(kSmallQRows);
  std::vector<std::uint16_t> host_baseline_key(kSmallKvRows);
  std::vector<std::uint16_t> host_baseline_value(kSmallKvRows);
  std::vector<std::uint16_t> host_candidate_q(kSmallQRows);
  std::vector<std::uint16_t> host_candidate_key(kSmallKvRows);
  std::vector<std::uint16_t> host_candidate_value(kSmallKvRows);
  ready = test.cuda_ok(
      cudaMemcpy(host_baseline_q.data(), baseline_q.get(),
                 host_baseline_q.size() * sizeof(host_baseline_q.front()),
                 cudaMemcpyDeviceToHost),
      "download small baseline Q output");
  ready = ready && test.cuda_ok(
                       cudaMemcpy(
                           host_baseline_key.data(), baseline_key.get(),
                           host_baseline_key.size() *
                               sizeof(host_baseline_key.front()),
                           cudaMemcpyDeviceToHost),
                       "download small baseline K output");
  ready = ready && test.cuda_ok(
                       cudaMemcpy(
                           host_baseline_value.data(), baseline_value.get(),
                           host_baseline_value.size() *
                               sizeof(host_baseline_value.front()),
                           cudaMemcpyDeviceToHost),
                       "download small baseline V output");
  ready = ready && test.cuda_ok(
                       cudaMemcpy(
                           host_candidate_q.data(), candidate_q.get(),
                           host_candidate_q.size() *
                               sizeof(host_candidate_q.front()),
                           cudaMemcpyDeviceToHost),
                       "download small candidate Q output");
  ready = ready && test.cuda_ok(
                       cudaMemcpy(
                           host_candidate_key.data(), candidate_key.get(),
                           host_candidate_key.size() *
                               sizeof(host_candidate_key.front()),
                           cudaMemcpyDeviceToHost),
                       "download small candidate K output");
  ready = ready && test.cuda_ok(
                       cudaMemcpy(
                           host_candidate_value.data(), candidate_value.get(),
                           host_candidate_value.size() *
                               sizeof(host_candidate_value.front()),
                           cudaMemcpyDeviceToHost),
                       "download small candidate V output");
  if (ready) {
    test.expect(host_candidate_q == host_baseline_q,
                "small full-attention fallback preserves Q bits");
    test.expect(host_candidate_key == host_baseline_key,
                "small full-attention fallback preserves K bits");
    test.expect(host_candidate_value == host_baseline_value,
                "small full-attention fallback preserves V bits");
  }

  bool small_linear = false;
  std::size_t small_total_nodes = 0U;
  const std::size_t small_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_full_attention_q_kv_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, small_q, small_key,
            small_value, small_input.get(), nullptr, 0U, candidate_q.get(),
            candidate_key.get(), candidate_value.get(),
            static_cast<void*>(stream));
      },
      "small full-attention fallback graph", &small_total_nodes,
      &small_linear);
  test.expect(small_total_nodes == 3U && small_kernel_nodes == 3U &&
                  small_linear,
              "small full-attention fallback is ordered Q then K then V");
}

void test_nvfp4_mlp_gate_up_silu_dispatch(TestContext& test) {
  constexpr std::size_t kExactRows = 17'408U;
  constexpr std::size_t kExactColumns = 5'120U;
  constexpr float kExactGateScale = 1.0F / 64.0F;
  constexpr float kExactUpScale = 1.0F / 128.0F;

  const auto* const fake_gate_packed =
      reinterpret_cast<const std::uint8_t*>(0x10'0000'0000ULL);
  const auto* const fake_gate_scales =
      reinterpret_cast<const std::uint8_t*>(0x20'0000'0000ULL);
  const auto* const fake_up_packed =
      reinterpret_cast<const std::uint8_t*>(0x30'0000'0000ULL);
  const auto* const fake_up_scales =
      reinterpret_cast<const std::uint8_t*>(0x40'0000'0000ULL);
  const auto* const fake_activation =
      reinterpret_cast<const std::uint16_t*>(0x50'0000'0000ULL);
  auto* const fake_gate_output =
      reinterpret_cast<std::uint16_t*>(0x60'0000'0000ULL);
  auto* const fake_up_output =
      reinterpret_cast<std::uint16_t*>(0x70'0000'0000ULL);
  const auto* const fake_gate_weight_scale =
      reinterpret_cast<const float*>(0x80'0000'0000ULL);
  const auto* const fake_gate_input_scale =
      reinterpret_cast<const float*>(0x81'0000'0000ULL);
  const auto* const fake_up_weight_scale =
      reinterpret_cast<const float*>(0x82'0000'0000ULL);
  const auto* const fake_up_input_scale =
      reinterpret_cast<const float*>(0x83'0000'0000ULL);

  const runtime::LinearWeight exact_gate = runtime::NvFp4LinearWeight{
      fake_gate_packed, fake_gate_scales, fake_gate_weight_scale,
      fake_gate_input_scale, kExactGateScale, 1.0F, kExactRows,
      kExactColumns};
  const runtime::LinearWeight exact_up = runtime::NvFp4LinearWeight{
      fake_up_packed, fake_up_scales, fake_up_weight_scale,
      fake_up_input_scale, kExactUpScale, 1.0F, kExactRows,
      kExactColumns};
  test.expect(runtime::supports_nvfp4_gate_up_silu_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                  exact_up),
              "exact ordered NVFP4 gate/up payload selects MLP fusion");
  test.expect(!runtime::supports_nvfp4_gate_up_silu_fusion(
                  runtime::ProjectionBackend::kReference, exact_gate,
                  exact_up),
              "reference backend does not select NVFP4 MLP fusion");

  std::size_t exact_total_nodes = 0U;
  const std::size_t exact_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_mlp_gate_up_silu_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
            exact_up, fake_activation, nullptr, 0U, fake_gate_output,
            fake_up_output, static_cast<void*>(stream));
      },
      "exact aligned NVFP4 MLP fusion graph", &exact_total_nodes);
  test.expect(exact_total_nodes == 1U && exact_kernel_nodes == 1U,
              "exact aligned NVFP4 MLP graph contains one fused kernel");

  const auto* const unaligned_fake_activation =
      reinterpret_cast<const std::uint16_t*>(0x50'0000'0002ULL);
  std::size_t unaligned_total_nodes = 0U;
  const std::size_t unaligned_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_mlp_gate_up_silu_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
            exact_up, unaligned_fake_activation, nullptr, 0U,
            fake_gate_output, fake_up_output, static_cast<void*>(stream));
      },
      "unaligned exact NVFP4 MLP fallback graph",
      &unaligned_total_nodes);
  test.expect(unaligned_total_nodes == 3U &&
                  unaligned_kernel_nodes == 3U,
              "unaligned exact NVFP4 MLP preserves gate-up-SiLU kernels");

  const runtime::LinearWeight near_miss_gate = runtime::NvFp4LinearWeight{
      fake_gate_packed, fake_gate_scales, fake_gate_weight_scale,
      fake_gate_input_scale, kExactGateScale, 1.0F, kExactRows - 1U,
      kExactColumns};
  const runtime::LinearWeight near_miss_up = runtime::NvFp4LinearWeight{
      fake_up_packed, fake_up_scales, fake_up_weight_scale,
      fake_up_input_scale, kExactUpScale, 1.0F, kExactRows - 1U,
      kExactColumns};
  test.expect(!runtime::supports_nvfp4_gate_up_silu_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  near_miss_gate, near_miss_up),
              "near-miss NVFP4 MLP shape does not select fusion");
  std::size_t near_miss_total_nodes = 0U;
  const std::size_t near_miss_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_mlp_gate_up_silu_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, near_miss_gate,
            near_miss_up, fake_activation, nullptr, 0U, fake_gate_output,
            fake_up_output, static_cast<void*>(stream));
      },
      "near-miss NVFP4 MLP fallback graph", &near_miss_total_nodes);
  test.expect(near_miss_total_nodes == 3U &&
                  near_miss_kernel_nodes == 3U,
              "valid near-miss NVFP4 MLP preserves three-step fallback");

  constexpr std::size_t kRows = 4U;
  constexpr std::size_t kColumns = 16U;
  constexpr std::size_t kPackedBytes = kRows * (kColumns / 2U);
  constexpr std::size_t kScaleBytes = kRows * (kColumns / 16U);
  constexpr float kGateScale = 1.0F / 16.0F;
  constexpr float kUpScale = 1.0F / 32.0F;
  constexpr std::array<std::uint16_t, kColumns> kActivationValues{
      0x3f80U, 0x3f00U, 0xbf80U, 0x4000U,
      0x3e80U, 0xbf00U, 0x4040U, 0xc000U,
      0x3f40U, 0xbf40U, 0x3fc0U, 0xbfc0U,
      0x4080U, 0xc080U, 0x3e00U, 0xbe00U};

  DeviceBuffer<std::uint8_t> gate_packed;
  DeviceBuffer<std::uint8_t> gate_scales;
  DeviceBuffer<std::uint8_t> up_packed;
  DeviceBuffer<std::uint8_t> up_scales;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<float> scratch;
  DeviceBuffer<std::uint16_t> gate_output;
  DeviceBuffer<std::uint16_t> up_output;
  DeviceBuffer<std::uint16_t> baseline_gate_output;
  DeviceBuffer<std::uint16_t> baseline_up_output;
  bool ready = gate_packed.allocate(test, kPackedBytes,
                                    "small MLP gate packed weights");
  ready = ready && gate_scales.allocate(
                       test, kScaleBytes, "small MLP gate block scales");
  ready = ready && up_packed.allocate(test, kPackedBytes,
                                      "small MLP up packed weights");
  ready = ready && up_scales.allocate(
                       test, kScaleBytes, "small MLP up block scales");
  ready = ready && activation.allocate(test, kColumns,
                                        "small MLP activation");
  ready = ready && companion_scales.allocate(
                       test, 4U, "small MLP companion scales");
  ready = ready && scratch.allocate(test, kRows, "small MLP scratch");
  ready = ready && gate_output.allocate(test, kRows,
                                        "small MLP gate output");
  ready = ready && up_output.allocate(test, kRows,
                                      "small MLP up output");
  ready = ready && baseline_gate_output.allocate(
                       test, kRows, "small MLP baseline gate output");
  ready = ready && baseline_up_output.allocate(
                       test, kRows, "small MLP baseline up output");
  if (!ready) {
    return;
  }

  ready = upload(test, gate_packed,
                 std::vector<std::uint8_t>(kPackedBytes, 0x21U),
                 "small MLP gate packed weights");
  ready = ready && upload(
                       test, gate_scales,
                       std::vector<std::uint8_t>(kScaleBytes, 0x38U),
                       "small MLP gate block scales");
  ready = ready && upload(
                       test, up_packed,
                       std::vector<std::uint8_t>(kPackedBytes, 0x32U),
                       "small MLP up packed weights");
  ready = ready && upload(
                       test, up_scales,
                       std::vector<std::uint8_t>(kScaleBytes, 0x38U),
                       "small MLP up block scales");
  ready = ready && upload(
                       test, activation,
                       std::vector<std::uint16_t>(kActivationValues.begin(),
                                                  kActivationValues.end()),
                       "small MLP activation");
  ready = ready && upload(
                       test, companion_scales,
                       std::vector<float>{kGateScale, 1.0F, kUpScale, 1.0F},
                       "small MLP companion scales");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight small_gate = runtime::NvFp4LinearWeight{
      gate_packed.get(), gate_scales.get(), companion_scales.get(),
      companion_scales.get() + 1U, kGateScale, 1.0F, kRows, kColumns};
  const runtime::LinearWeight small_up = runtime::NvFp4LinearWeight{
      up_packed.get(), up_scales.get(), companion_scales.get() + 2U,
      companion_scales.get() + 3U, kUpScale, 1.0F, kRows, kColumns};

  const auto launch_old_three_step = [&]
      (const runtime::ProjectionBackend backend,
       float* const scratch_pointer,
       const std::size_t scratch_elements) {
    int status = runtime::launch_projection_tile_to_bf16_cuda(
        backend, small_gate, activation.get(), 1U, scratch_pointer,
        scratch_elements, baseline_gate_output.get());
    if (static_cast<cudaError_t>(status) != cudaSuccess) {
      return status;
    }
    status = runtime::launch_projection_tile_to_bf16_cuda(
        backend, small_up, activation.get(), 1U, scratch_pointer,
        scratch_elements, baseline_up_output.get());
    if (static_cast<cudaError_t>(status) != cudaSuccess) {
      return status;
    }
    return runtime::launch_silu_mul_reference_cuda(
        baseline_gate_output.get(), baseline_up_output.get(), kRows,
        baseline_gate_output.get());
  };
  const auto expect_fallback_matches = [&]
      (const runtime::ProjectionBackend backend,
       float* const scratch_pointer,
       const std::size_t scratch_elements,
       const std::string& label) {
    int status = launch_old_three_step(backend, scratch_pointer,
                                       scratch_elements);
    test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
                label + " old three-step succeeds");
    if (static_cast<cudaError_t>(status) != cudaSuccess) {
      return;
    }
    status = runtime::launch_mlp_gate_up_silu_to_bf16_cuda(
        backend, small_gate, small_up, activation.get(), scratch_pointer,
        scratch_elements, gate_output.get(), up_output.get());
    test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
                label + " specialized API succeeds");
    if (static_cast<cudaError_t>(status) != cudaSuccess ||
        !test.cuda_ok(cudaDeviceSynchronize(), "synchronize " + label)) {
      return;
    }
    std::array<std::uint16_t, kRows> actual_gate{};
    std::array<std::uint16_t, kRows> actual_up{};
    std::array<std::uint16_t, kRows> expected_gate{};
    std::array<std::uint16_t, kRows> expected_up{};
    bool copied = test.cuda_ok(
        cudaMemcpy(actual_gate.data(), gate_output.get(),
                   sizeof(actual_gate), cudaMemcpyDeviceToHost),
        "download gate output " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(actual_up.data(), up_output.get(),
                                      sizeof(actual_up),
                                      cudaMemcpyDeviceToHost),
                           "download up output " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(expected_gate.data(),
                                      baseline_gate_output.get(),
                                      sizeof(expected_gate),
                                      cudaMemcpyDeviceToHost),
                           "download baseline gate output " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(expected_up.data(),
                                      baseline_up_output.get(),
                                      sizeof(expected_up),
                                      cudaMemcpyDeviceToHost),
                           "download baseline up output " + label);
    if (copied) {
      test.expect(actual_gate == expected_gate,
                  label + " gate/SiLU output is bitwise old-three-step");
      test.expect(actual_up == expected_up,
                  label + " retained up output is bitwise old projection");
    }
  };

  std::size_t small_total_nodes = 0U;
  const std::size_t small_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_mlp_gate_up_silu_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, small_gate,
            small_up, activation.get(), nullptr, 0U, gate_output.get(),
            up_output.get(), static_cast<void*>(stream));
      },
      "small NVFP4 MLP fallback graph", &small_total_nodes);
  test.expect(small_total_nodes == 3U && small_kernel_nodes == 3U,
              "small valid NVFP4 MLP uses gate-up-SiLU fallback");
  expect_fallback_matches(runtime::ProjectionBackend::kSm87WeightOnly,
                          nullptr, 0U, "small SM87 NVFP4 MLP fallback");

  std::size_t reference_total_nodes = 0U;
  const std::size_t reference_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::launch_mlp_gate_up_silu_to_bf16_cuda(
            runtime::ProjectionBackend::kReference, small_gate, small_up,
            activation.get(), scratch.get(), kRows, gate_output.get(),
            up_output.get(), static_cast<void*>(stream));
      },
      "reference NVFP4 MLP fallback graph", &reference_total_nodes);
  test.expect(reference_total_nodes == 5U &&
                  reference_kernel_nodes == 5U,
              "reference NVFP4 MLP preserves two GEMV-convert pairs and SiLU");
  expect_fallback_matches(runtime::ProjectionBackend::kReference,
                          scratch.get(), kRows,
                          "small reference NVFP4 MLP fallback");

  const auto expect_invalid_without_enqueue = [&]
      (const runtime::LinearWeight& gate_arg,
       const runtime::LinearWeight& up_arg,
       const std::uint16_t* const input_arg,
       std::uint16_t* const gate_output_arg,
       std::uint16_t* const up_output_arg,
       const std::string& label) {
    bool initialized = test.cuda_ok(
        cudaMemset(gate_output_arg, 0xa5, kRows * sizeof(std::uint16_t)),
        "initialize gate canary " + label);
    initialized = initialized && test.cuda_ok(
                                     cudaMemset(up_output_arg, 0xa5,
                                                kRows * sizeof(std::uint16_t)),
                                     "initialize up canary " + label);
    if (!initialized) {
      return;
    }
    const int status = runtime::launch_mlp_gate_up_silu_to_bf16_cuda(
        runtime::ProjectionBackend::kSm87WeightOnly, gate_arg, up_arg,
        input_arg, nullptr, 0U, gate_output_arg, up_output_arg);
    test.expect(static_cast<cudaError_t>(status) == cudaErrorInvalidValue,
                label + " returns cudaErrorInvalidValue");
    if (!test.cuda_ok(cudaDeviceSynchronize(), "synchronize " + label)) {
      return;
    }
    std::array<std::uint16_t, kRows> gate_canary{};
    std::array<std::uint16_t, kRows> up_canary{};
    bool copied = test.cuda_ok(
        cudaMemcpy(gate_canary.data(), gate_output_arg, sizeof(gate_canary),
                   cudaMemcpyDeviceToHost),
        "download gate canary " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(up_canary.data(), up_output_arg,
                                      sizeof(up_canary),
                                      cudaMemcpyDeviceToHost),
                           "download up canary " + label);
    if (copied) {
      test.expect(std::all_of(gate_canary.begin(), gate_canary.end(),
                              [](const std::uint16_t value) {
                                return value == 0xa5a5U;
                              }),
                  label + " leaves gate canary unchanged");
      test.expect(std::all_of(up_canary.begin(), up_canary.end(),
                              [](const std::uint16_t value) {
                                return value == 0xa5a5U;
                              }),
                  label + " leaves up canary unchanged");
    }
  };

  const runtime::LinearWeight different_rows_up = runtime::NvFp4LinearWeight{
      up_packed.get(), up_scales.get(), companion_scales.get() + 2U,
      companion_scales.get() + 3U, kUpScale, 1.0F, kRows - 1U, kColumns};
  expect_invalid_without_enqueue(
      small_gate, different_rows_up, activation.get(), gate_output.get(),
      up_output.get(), "different-row MLP projections");

  const runtime::LinearWeight malformed_up = runtime::NvFp4LinearWeight{
      up_packed.get(), nullptr, companion_scales.get() + 2U,
      companion_scales.get() + 3U, kUpScale, 1.0F, kRows, kColumns};
  const runtime::LinearWeight malformed_exact_up =
      runtime::NvFp4LinearWeight{
          fake_up_packed, nullptr, fake_up_weight_scale,
          fake_up_input_scale, kExactUpScale, 1.0F, kExactRows,
          kExactColumns};
  test.expect(!runtime::supports_nvfp4_gate_up_silu_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                  malformed_exact_up),
              "malformed NVFP4 payload does not select MLP fusion");
  expect_invalid_without_enqueue(
      small_gate, malformed_up, activation.get(), gate_output.get(),
      up_output.get(), "malformed up projection");
  expect_invalid_without_enqueue(
      small_gate, small_up, nullptr, gate_output.get(), up_output.get(),
      "null MLP activation");
  expect_invalid_without_enqueue(
      small_gate, small_up, activation.get(), gate_output.get(),
      gate_output.get(), "overlapping MLP outputs");

  expect_invalid_without_enqueue(
      small_gate, small_up, activation.get(),
      reinterpret_cast<std::uint16_t*>(up_packed.get()), up_output.get(),
      "gate output overlapping up packed weights");
}

void test_post_attention_residual_norm_mlp_gate_up_silu_dispatch(
    TestContext& test) {
  constexpr std::size_t kExactRows = 17'408U;
  constexpr std::size_t kBf16PairRows = 48U;
  constexpr std::size_t kFp8PairRows = 1'024U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr float kEpsilon = 1.0e-6F;
  constexpr float kExactGateScale = 1.0F / 64.0F;
  constexpr float kExactUpScale = 1.0F / 128.0F;

  const auto* const fake_gate_packed =
      reinterpret_cast<const std::uint8_t*>(0x10'0000'0000ULL);
  const auto* const fake_unaligned_gate_packed =
      reinterpret_cast<const std::uint8_t*>(0x10'0000'0002ULL);
  const auto* const fake_gate_scales =
      reinterpret_cast<const std::uint8_t*>(0x20'0000'0000ULL);
  const auto* const fake_up_packed =
      reinterpret_cast<const std::uint8_t*>(0x30'0000'0000ULL);
  const auto* const fake_up_scales =
      reinterpret_cast<const std::uint8_t*>(0x40'0000'0000ULL);
  const auto* const fake_gate_weight_scale =
      reinterpret_cast<const float*>(0x50'0000'0000ULL);
  const auto* const fake_gate_input_scale =
      reinterpret_cast<const float*>(0x51'0000'0000ULL);
  const auto* const fake_up_weight_scale =
      reinterpret_cast<const float*>(0x52'0000'0000ULL);
  const auto* const fake_up_input_scale =
      reinterpret_cast<const float*>(0x53'0000'0000ULL);
  const auto* const fake_residual_left =
      reinterpret_cast<const std::uint16_t*>(0x60'0000'0000ULL);
  auto* const fake_residual_right =
      reinterpret_cast<std::uint16_t*>(0x70'0000'0000ULL);
  const auto* const fake_norm_weight =
      reinterpret_cast<const std::uint16_t*>(0x80'0000'0000ULL);
  auto* const fake_residual_output =
      reinterpret_cast<std::uint16_t*>(0x90'0000'0000ULL);
  auto* const fake_gate_output =
      reinterpret_cast<std::uint16_t*>(0xa0'0000'0000ULL);
  auto* const fake_up_output =
      reinterpret_cast<std::uint16_t*>(0xb0'0000'0000ULL);
  const auto* const fake_bf16_gate_weight =
      reinterpret_cast<const std::uint16_t*>(0xc0'0000'0000ULL);
  const auto* const fake_bf16_up_weight =
      reinterpret_cast<const std::uint16_t*>(0xd0'0000'0000ULL);
  const auto* const fake_fp8_gate_weight =
      reinterpret_cast<const std::uint8_t*>(0xe0'0000'0000ULL);
  const auto* const fake_unaligned_fp8_gate_weight =
      reinterpret_cast<const std::uint8_t*>(0xe0'0000'0002ULL);
  const auto* const fake_fp8_up_weight =
      reinterpret_cast<const std::uint8_t*>(0xf0'0000'0000ULL);
  const auto* const fake_fp8_gate_weight_scale =
      reinterpret_cast<const float*>(0x54'0000'0000ULL);
  const auto* const fake_fp8_gate_input_scale =
      reinterpret_cast<const float*>(0x55'0000'0000ULL);
  const auto* const fake_fp8_up_weight_scale =
      reinterpret_cast<const float*>(0x56'0000'0000ULL);
  const auto* const fake_fp8_up_input_scale =
      reinterpret_cast<const float*>(0x57'0000'0000ULL);

  const runtime::LinearWeight exact_gate = runtime::NvFp4LinearWeight{
      fake_gate_packed, fake_gate_scales, fake_gate_weight_scale,
      fake_gate_input_scale, kExactGateScale, 1.0F, kExactRows, kColumns};
  const runtime::LinearWeight exact_up = runtime::NvFp4LinearWeight{
      fake_up_packed, fake_up_scales, fake_up_weight_scale,
      fake_up_input_scale, kExactUpScale, 1.0F, kExactRows, kColumns};
  std::size_t exact_total_nodes = 0U;
  const std::size_t exact_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, kEpsilon, nullptr, 0U,
                fake_residual_output, fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "exact residual-norm NVFP4 MLP fusion graph", &exact_total_nodes);
  test.expect(exact_total_nodes == 1U && exact_kernel_nodes == 1U,
              "exact residual-norm NVFP4 MLP graph has one fused kernel");

  std::size_t exact_runner_total_nodes = 0U;
  const std::size_t exact_runner_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, kEpsilon, nullptr, 0U,
                fake_residual_output, fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "exact Decode-runner dead-up NVFP4 MLP graph",
      &exact_runner_total_nodes);
  test.expect(exact_runner_total_nodes == 1U &&
                  exact_runner_kernel_nodes == 1U,
              "exact Decode-runner dead-up graph has one fused kernel");
  const CapturedKernelChain exact_public_chain =
      capture_ordered_kernel_chain(
          test,
          [&](cudaStream_t stream) noexcept {
            return runtime::
                launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                    runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                    exact_up, fake_residual_left, fake_residual_right,
                    fake_norm_weight, kEpsilon, nullptr, 0U,
                    fake_residual_output, fake_gate_output, fake_up_output,
                    static_cast<void*>(stream));
          },
          "exact public double-output NVFP4 MLP ordered graph");
  const CapturedKernelChain exact_runner_chain =
      capture_ordered_kernel_chain(
          test,
          [&](cudaStream_t stream) noexcept {
            return runtime::
                launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
                    runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                    exact_up, fake_residual_left, fake_residual_right,
                    fake_norm_weight, kEpsilon, nullptr, 0U,
                    fake_residual_output, fake_gate_output, fake_up_output,
                    static_cast<void*>(stream));
          },
          "exact Decode-runner dead-up NVFP4 MLP ordered graph");
  const bool exact_runner_identity =
      exact_public_chain.valid && exact_runner_chain.valid &&
      exact_public_chain.launches.size() == 1U &&
      exact_runner_chain.launches.size() == 1U &&
      exact_public_chain.launches[0].function !=
          exact_runner_chain.launches[0].function &&
      exact_public_chain.launches[0].grid.x == 32U &&
      exact_runner_chain.launches[0].grid.x == 32U &&
      exact_public_chain.launches[0].block.x == 512U &&
      exact_runner_chain.launches[0].block.x == 512U &&
      exact_public_chain.launches[0].dynamic_shared_bytes == 0U &&
      exact_runner_chain.launches[0].dynamic_shared_bytes == 0U;
  test.expect(exact_runner_identity,
              "public and Decode-runner exact routes are distinct 32x512 kernels");

  const runtime::LinearWeight unaligned_exact_gate =
      runtime::NvFp4LinearWeight{
          fake_unaligned_gate_packed, fake_gate_scales,
          fake_gate_weight_scale, fake_gate_input_scale, kExactGateScale,
          1.0F, kExactRows, kColumns};
  std::size_t unaligned_total_nodes = 0U;
  const std::size_t unaligned_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly,
                unaligned_exact_gate, exact_up, fake_residual_left,
                fake_residual_right, fake_norm_weight, kEpsilon, nullptr, 0U,
                fake_residual_output, fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "unaligned residual-norm NVFP4 MLP fallback graph",
      &unaligned_total_nodes);
  test.expect(unaligned_total_nodes == 4U && unaligned_kernel_nodes == 4U,
              "unaligned residual-norm NVFP4 MLP uses norm plus old MLP");

  const runtime::LinearWeight near_miss_gate = runtime::NvFp4LinearWeight{
      fake_gate_packed, fake_gate_scales, fake_gate_weight_scale,
      fake_gate_input_scale, kExactGateScale, 1.0F, kExactRows - 1U,
      kColumns};
  const runtime::LinearWeight near_miss_up = runtime::NvFp4LinearWeight{
      fake_up_packed, fake_up_scales, fake_up_weight_scale,
      fake_up_input_scale, kExactUpScale, 1.0F, kExactRows - 1U, kColumns};
  std::size_t near_miss_total_nodes = 0U;
  const std::size_t near_miss_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly,
                near_miss_gate, near_miss_up, fake_residual_left,
                fake_residual_right, fake_norm_weight, kEpsilon, nullptr, 0U,
                fake_residual_output, fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "near-miss residual-norm NVFP4 MLP fallback graph",
      &near_miss_total_nodes);
  test.expect(near_miss_total_nodes == 4U && near_miss_kernel_nodes == 4U,
              "near-miss residual-norm NVFP4 MLP uses four-step fallback");

  const runtime::LinearWeight bf16_gate = runtime::Bf16LinearWeight{
      fake_bf16_gate_weight, kBf16PairRows, kColumns};
  const runtime::LinearWeight bf16_up = runtime::Bf16LinearWeight{
      fake_bf16_up_weight, kBf16PairRows, kColumns};
  std::size_t bf16_total_nodes = 0U;
  const std::size_t bf16_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, bf16_gate,
                bf16_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, kEpsilon, nullptr, 0U,
                fake_residual_output, fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "SM87 BF16 pair residual-norm MLP graph", &bf16_total_nodes);
  test.expect(bf16_total_nodes == 3U && bf16_kernel_nodes == 3U,
              "SM87 BF16 residual-norm MLP graph has norm, pair, and SiLU");

  const runtime::LinearWeight fp8_gate = runtime::Fp8LinearWeight{
      fake_fp8_gate_weight, fake_fp8_gate_weight_scale,
      fake_fp8_gate_input_scale, 1.0F / 64.0F, 1.0F, kFp8PairRows,
      kColumns};
  const runtime::LinearWeight fp8_up = runtime::Fp8LinearWeight{
      fake_fp8_up_weight, fake_fp8_up_weight_scale,
      fake_fp8_up_input_scale, 1.0F / 128.0F, 1.0F, kFp8PairRows,
      kColumns};
  std::size_t fp8_total_nodes = 0U;
  const std::size_t fp8_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, fp8_gate,
                fp8_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, kEpsilon, nullptr, 0U,
                fake_residual_output, fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "SM87 aligned FP8 pair residual-norm MLP graph", &fp8_total_nodes);
  test.expect(fp8_total_nodes == 3U && fp8_kernel_nodes == 3U,
              "aligned FP8 residual-norm MLP graph has norm, pair, and SiLU");

  const runtime::LinearWeight unaligned_fp8_gate =
      runtime::Fp8LinearWeight{
          fake_unaligned_fp8_gate_weight, fake_fp8_gate_weight_scale,
          fake_fp8_gate_input_scale, 1.0F / 64.0F, 1.0F,
          kFp8PairRows, kColumns};
  std::size_t unaligned_fp8_total_nodes = 0U;
  const std::size_t unaligned_fp8_kernel_nodes =
      captured_kernel_node_count(
          test,
          [&](cudaStream_t stream) noexcept {
            return runtime::
                launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                    runtime::ProjectionBackend::kSm87WeightOnly,
                    unaligned_fp8_gate, fp8_up, fake_residual_left,
                    fake_residual_right, fake_norm_weight, kEpsilon, nullptr,
                    0U, fake_residual_output, fake_gate_output,
                    fake_up_output, static_cast<void*>(stream));
          },
          "SM87 unaligned FP8 pair residual-norm MLP graph",
          &unaligned_fp8_total_nodes);
  test.expect(unaligned_fp8_total_nodes == 4U &&
                  unaligned_fp8_kernel_nodes == 4U,
              "unaligned FP8 residual-norm MLP graph preserves split pair");

  const auto expect_runner_fallback_chain =
      [&](const runtime::LinearWeight& gate,
          const runtime::LinearWeight& up,
          const std::size_t expected_nodes, const std::string& case_label) {
        const CapturedKernelChain public_chain =
            capture_ordered_kernel_chain(
                test,
                [&](cudaStream_t stream) noexcept {
                  return runtime::
                      launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                          runtime::ProjectionBackend::kSm87WeightOnly, gate,
                          up, fake_residual_left, fake_residual_right,
                          fake_norm_weight, kEpsilon, nullptr, 0U,
                          fake_residual_output, fake_gate_output,
                          fake_up_output, static_cast<void*>(stream));
                },
                case_label + " public ordered graph");
        const CapturedKernelChain runner_chain =
            capture_ordered_kernel_chain(
                test,
                [&](cudaStream_t stream) noexcept {
                  return runtime::
                      launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
                          runtime::ProjectionBackend::kSm87WeightOnly, gate,
                          up, fake_residual_left, fake_residual_right,
                          fake_norm_weight, kEpsilon, nullptr, 0U,
                          fake_residual_output, fake_gate_output,
                          fake_up_output, static_cast<void*>(stream));
                },
                case_label + " runner ordered graph");
        bool identical = public_chain.valid && runner_chain.valid &&
                         public_chain.launches.size() == expected_nodes &&
                         runner_chain.launches.size() == expected_nodes;
        for (std::size_t node = 0U;
             identical && node < public_chain.launches.size(); ++node) {
          const CapturedKernelLaunch& public_launch =
              public_chain.launches[node];
          const CapturedKernelLaunch& runner_launch =
              runner_chain.launches[node];
          identical = public_launch.function == runner_launch.function &&
                      public_launch.grid.x == runner_launch.grid.x &&
                      public_launch.grid.y == runner_launch.grid.y &&
                      public_launch.grid.z == runner_launch.grid.z &&
                      public_launch.block.x == runner_launch.block.x &&
                      public_launch.block.y == runner_launch.block.y &&
                      public_launch.block.z == runner_launch.block.z &&
                      public_launch.dynamic_shared_bytes ==
                          runner_launch.dynamic_shared_bytes;
        }
        test.expect(identical,
                    case_label +
                        " Decode-runner fallback is the public kernel chain");
      };
  expect_runner_fallback_chain(unaligned_exact_gate, exact_up, 4U,
                               "unaligned exact NVFP4");
  expect_runner_fallback_chain(near_miss_gate, near_miss_up, 4U,
                               "near-miss NVFP4");
  expect_runner_fallback_chain(bf16_gate, bf16_up, 3U, "BF16 pair");
  expect_runner_fallback_chain(fp8_gate, fp8_up, 3U, "aligned FP8 pair");
  expect_runner_fallback_chain(unaligned_fp8_gate, fp8_up, 4U,
                               "unaligned FP8 pair");

  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, 0.0F, nullptr, 0U, fake_residual_output,
                fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "residual-norm NVFP4 MLP invalid zero epsilon");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight,
                std::numeric_limits<float>::quiet_NaN(), nullptr, 0U,
                fake_residual_output, fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "residual-norm NVFP4 MLP NaN epsilon");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, kEpsilon, nullptr, 0U, fake_gate_output,
                fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "residual output aliases gate output");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, kEpsilon, nullptr, 0U,
                reinterpret_cast<std::uint16_t*>(
                    const_cast<float*>(fake_gate_weight_scale)),
                fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "residual output aliases persistent gate scalar weight");

  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, 0.0F, nullptr, 0U, fake_residual_output,
                fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "Decode-runner dead-up rejects zero epsilon");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight,
                std::numeric_limits<float>::quiet_NaN(), nullptr, 0U,
                fake_residual_output, fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "Decode-runner dead-up rejects NaN epsilon");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, kEpsilon, nullptr, 0U, fake_gate_output,
                fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "Decode-runner dead-up rejects residual/gate alias");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, kEpsilon, nullptr, 0U,
                reinterpret_cast<std::uint16_t*>(
                    const_cast<float*>(fake_gate_weight_scale)),
                fake_gate_output, fake_up_output,
                static_cast<void*>(stream));
      },
      "Decode-runner dead-up rejects persistent scalar alias");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, kEpsilon, nullptr, 0U,
                fake_residual_output, fake_gate_output, nullptr,
                static_cast<void*>(stream));
      },
      "Decode-runner dead-up rejects null up workspace");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, exact_gate,
                exact_up, fake_residual_left, fake_residual_right,
                fake_norm_weight, kEpsilon, nullptr, 0U,
                fake_residual_output, fake_gate_output, fake_gate_output,
                static_cast<void*>(stream));
      },
      "Decode-runner dead-up rejects gate/workspace alias");

  constexpr std::size_t kRows = 4U;
  constexpr std::size_t kPackedBytes = kRows * (kColumns / 2U);
  constexpr std::size_t kScaleBytes = kRows * (kColumns / 16U);
  constexpr float kGateScale = 1.0F / 16.0F;
  constexpr float kUpScale = 1.0F / 32.0F;
  DeviceBuffer<std::uint8_t> gate_packed;
  DeviceBuffer<std::uint8_t> gate_scales;
  DeviceBuffer<std::uint8_t> up_packed;
  DeviceBuffer<std::uint8_t> up_scales;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<float> scratch;
  DeviceBuffer<std::uint16_t> residual_left;
  DeviceBuffer<std::uint16_t> baseline_right;
  DeviceBuffer<std::uint16_t> candidate_right;
  DeviceBuffer<std::uint16_t> norm_weight;
  DeviceBuffer<std::uint16_t> baseline_residual;
  DeviceBuffer<std::uint16_t> candidate_residual;
  DeviceBuffer<std::uint16_t> baseline_gate;
  DeviceBuffer<std::uint16_t> baseline_up;
  DeviceBuffer<std::uint16_t> candidate_gate;
  DeviceBuffer<std::uint16_t> candidate_up;
  bool ready = gate_packed.allocate(
      test, kPackedBytes, "residual-norm small gate packed weights");
  ready = ready && gate_scales.allocate(
                       test, kScaleBytes,
                       "residual-norm small gate block scales");
  ready = ready && up_packed.allocate(
                       test, kPackedBytes,
                       "residual-norm small up packed weights");
  ready = ready && up_scales.allocate(
                       test, kScaleBytes,
                       "residual-norm small up block scales");
  ready = ready && companion_scales.allocate(
                       test, 4U, "residual-norm companion scales");
  ready = ready && scratch.allocate(test, kRows,
                                    "residual-norm reference scratch");
  ready = ready && residual_left.allocate(
                       test, kColumns, "residual-norm left input");
  ready = ready && baseline_right.allocate(
                       test, kColumns, "residual-norm baseline right input");
  ready = ready && candidate_right.allocate(
                       test, kColumns,
                       "residual-norm candidate right input");
  ready = ready && norm_weight.allocate(
                       test, kColumns, "residual-norm weight");
  ready = ready && baseline_residual.allocate(
                       test, kColumns, "residual-norm baseline residual");
  ready = ready && candidate_residual.allocate(
                       test, kColumns, "residual-norm candidate residual");
  ready = ready && baseline_gate.allocate(
                       test, kRows, "residual-norm baseline final");
  ready = ready && baseline_up.allocate(
                       test, kRows, "residual-norm baseline up");
  ready = ready && candidate_gate.allocate(
                       test, kRows, "residual-norm candidate final");
  ready = ready && candidate_up.allocate(
                       test, kRows, "residual-norm candidate up");
  if (!ready) {
    return;
  }

  std::vector<std::uint16_t> host_left(kColumns);
  std::vector<std::uint16_t> host_right(kColumns);
  std::vector<std::uint16_t> host_norm_weight(kColumns);
  constexpr std::array<std::uint16_t, 8U> kLeftPattern{
      0x3f80U, 0xbf00U, 0x3e80U, 0x3f00U,
      0xbf80U, 0x3fc0U, 0xbe80U, 0x4000U};
  constexpr std::array<std::uint16_t, 8U> kRightPattern{
      0x3e00U, 0xbe80U, 0x3f00U, 0xbf40U,
      0x3e80U, 0x3f80U, 0xbf00U, 0x3f40U};
  constexpr std::array<std::uint16_t, 8U> kNormPattern{
      0x3f80U, 0x3f00U, 0x3fc0U, 0x3e80U,
      0x4000U, 0x3f40U, 0x3fa0U, 0x3f20U};
  for (std::size_t column = 0U; column < kColumns; ++column) {
    host_left[column] = kLeftPattern[column % kLeftPattern.size()];
    host_right[column] = kRightPattern[column % kRightPattern.size()];
    host_norm_weight[column] =
        kNormPattern[column % kNormPattern.size()];
  }
  ready = upload(test, gate_packed,
                 std::vector<std::uint8_t>(kPackedBytes, 0x21U),
                 "residual-norm small gate packed weights");
  ready = ready && upload(
                       test, gate_scales,
                       std::vector<std::uint8_t>(kScaleBytes, 0x38U),
                       "residual-norm small gate block scales");
  ready = ready && upload(
                       test, up_packed,
                       std::vector<std::uint8_t>(kPackedBytes, 0x32U),
                       "residual-norm small up packed weights");
  ready = ready && upload(
                       test, up_scales,
                       std::vector<std::uint8_t>(kScaleBytes, 0x38U),
                       "residual-norm small up block scales");
  ready = ready && upload(
                       test, companion_scales,
                       std::vector<float>{kGateScale, 1.0F, kUpScale, 1.0F},
                       "residual-norm companion scales");
  ready = ready && upload(test, residual_left, host_left,
                          "residual-norm left input");
  ready = ready && upload(test, baseline_right, host_right,
                          "residual-norm baseline right input");
  ready = ready && upload(test, candidate_right, host_right,
                          "residual-norm candidate right input");
  ready = ready && upload(test, norm_weight, host_norm_weight,
                          "residual-norm weight");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight small_gate = runtime::NvFp4LinearWeight{
      gate_packed.get(), gate_scales.get(), companion_scales.get(),
      companion_scales.get() + 1U, kGateScale, 1.0F, kRows, kColumns};
  const runtime::LinearWeight small_up = runtime::NvFp4LinearWeight{
      up_packed.get(), up_scales.get(), companion_scales.get() + 2U,
      companion_scales.get() + 3U, kUpScale, 1.0F, kRows, kColumns};

  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly, small_gate,
                small_up, residual_left.get(), candidate_right.get(),
                norm_weight.get(), 0.0F, nullptr, 0U,
                candidate_residual.get(), candidate_gate.get(),
                candidate_up.get(), static_cast<void*>(stream));
      },
      "fallback residual-norm MLP rejects zero epsilon before norm");

  const runtime::LinearWeight near_column_gate =
      runtime::NvFp4LinearWeight{
          gate_packed.get(), gate_scales.get(), companion_scales.get(),
          companion_scales.get() + 1U, kGateScale, 1.0F, kRows,
          kColumns - 16U};
  const runtime::LinearWeight near_column_up = runtime::NvFp4LinearWeight{
      up_packed.get(), up_scales.get(), companion_scales.get() + 2U,
      companion_scales.get() + 3U, kUpScale, 1.0F, kRows,
      kColumns - 16U};
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kSm87WeightOnly,
                near_column_gate, near_column_up, residual_left.get(),
                candidate_right.get(), norm_weight.get(), kEpsilon, nullptr,
                0U, candidate_residual.get(), candidate_gate.get(),
                candidate_up.get(), static_cast<void*>(stream));
      },
      "fallback residual-norm MLP rejects K near miss before norm");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kReference, small_gate, small_up,
                residual_left.get(), candidate_right.get(), norm_weight.get(),
                kEpsilon, scratch.get(), kRows - 1U,
                candidate_residual.get(), candidate_gate.get(),
                candidate_up.get(), static_cast<void*>(stream));
      },
      "reference residual-norm MLP rejects insufficient scratch before norm");
  expect_invalid_capture_has_no_nodes(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kReference, small_gate, small_up,
                residual_left.get(), candidate_right.get(), norm_weight.get(),
                kEpsilon, reinterpret_cast<float*>(norm_weight.get()), kRows,
                candidate_residual.get(), candidate_gate.get(),
                candidate_up.get(), static_cast<void*>(stream));
      },
      "reference residual-norm MLP rejects scratch aliasing norm weight");

  std::size_t reference_total_nodes = 0U;
  const std::size_t reference_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return runtime::
            launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
                runtime::ProjectionBackend::kReference, small_gate, small_up,
                residual_left.get(), candidate_right.get(), norm_weight.get(),
                kEpsilon, scratch.get(), kRows, candidate_residual.get(),
                candidate_gate.get(), candidate_up.get(),
                static_cast<void*>(stream));
      },
      "reference residual-norm NVFP4 MLP fallback graph",
      &reference_total_nodes);
  test.expect(reference_total_nodes == 6U && reference_kernel_nodes == 6U,
              "reference residual-norm MLP preserves one plus five kernels");

  const auto expect_fallback_matches = [&]
      (const runtime::ProjectionBackend backend,
       float* const scratch_pointer, const std::size_t scratch_elements,
       const bool decode_runner_dead_up, const std::string& label) {
    bool reset = upload(test, baseline_right, host_right,
                        label + " reset baseline right");
    reset = reset && upload(test, candidate_right, host_right,
                            label + " reset candidate right");
    if (!reset) {
      return;
    }
    int status = runtime::launch_residual_add_centered_rms_norm_5120_cuda(
        residual_left.get(), baseline_right.get(), norm_weight.get(),
        kEpsilon, baseline_residual.get(), baseline_right.get());
    if (static_cast<cudaError_t>(status) == cudaSuccess) {
      status = runtime::launch_mlp_gate_up_silu_to_bf16_cuda(
          backend, small_gate, small_up, baseline_right.get(),
          scratch_pointer, scratch_elements, baseline_gate.get(),
          baseline_up.get());
    }
    test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
                label + " old two-call chain succeeds");
    if (static_cast<cudaError_t>(status) != cudaSuccess) {
      return;
    }
    if (decode_runner_dead_up) {
      status = runtime::
          launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
              backend, small_gate, small_up, residual_left.get(),
              candidate_right.get(), norm_weight.get(), kEpsilon,
              scratch_pointer, scratch_elements, candidate_residual.get(),
              candidate_gate.get(), candidate_up.get());
    } else {
      status = runtime::
          launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
              backend, small_gate, small_up, residual_left.get(),
              candidate_right.get(), norm_weight.get(), kEpsilon,
              scratch_pointer, scratch_elements, candidate_residual.get(),
              candidate_gate.get(), candidate_up.get());
    }
    test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
                label + " combined API succeeds");
    if (static_cast<cudaError_t>(status) != cudaSuccess ||
        !test.cuda_ok(cudaDeviceSynchronize(), "synchronize " + label)) {
      return;
    }

    std::vector<std::uint16_t> actual_residual(kColumns);
    std::vector<std::uint16_t> expected_residual(kColumns);
    std::vector<std::uint16_t> actual_normalized(kColumns);
    std::vector<std::uint16_t> expected_normalized(kColumns);
    std::array<std::uint16_t, kRows> actual_final{};
    std::array<std::uint16_t, kRows> expected_final{};
    std::array<std::uint16_t, kRows> actual_up{};
    std::array<std::uint16_t, kRows> expected_up{};
    bool copied = test.cuda_ok(
        cudaMemcpy(actual_residual.data(), candidate_residual.get(),
                   actual_residual.size() * sizeof(actual_residual.front()),
                   cudaMemcpyDeviceToHost),
        "download candidate residual " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(expected_residual.data(),
                                      baseline_residual.get(),
                                      expected_residual.size() *
                                          sizeof(expected_residual.front()),
                                      cudaMemcpyDeviceToHost),
                           "download baseline residual " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(actual_normalized.data(),
                                      candidate_right.get(),
                                      actual_normalized.size() *
                                          sizeof(actual_normalized.front()),
                                      cudaMemcpyDeviceToHost),
                           "download candidate normalized input " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(expected_normalized.data(),
                                      baseline_right.get(),
                                      expected_normalized.size() *
                                          sizeof(expected_normalized.front()),
                                      cudaMemcpyDeviceToHost),
                           "download baseline normalized input " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(actual_final.data(),
                                      candidate_gate.get(),
                                      sizeof(actual_final),
                                      cudaMemcpyDeviceToHost),
                           "download candidate final " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(expected_final.data(),
                                      baseline_gate.get(),
                                      sizeof(expected_final),
                                      cudaMemcpyDeviceToHost),
                           "download baseline final " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(actual_up.data(), candidate_up.get(),
                                      sizeof(actual_up),
                                      cudaMemcpyDeviceToHost),
                           "download candidate up " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(expected_up.data(), baseline_up.get(),
                                      sizeof(expected_up),
                                      cudaMemcpyDeviceToHost),
                           "download baseline up " + label);
    if (copied) {
      test.expect(actual_residual == expected_residual,
                  label + " residual is bitwise old chain");
      test.expect(actual_normalized == expected_normalized,
                  label + " normalized right side effect is bitwise old chain");
      test.expect(actual_final == expected_final,
                  label + " final gate/SiLU is bitwise old chain");
      test.expect(actual_up == expected_up,
                  label + " retained up is bitwise old chain");
    }
  };
  expect_fallback_matches(runtime::ProjectionBackend::kSm87WeightOnly,
                          nullptr, 0U, false,
                          "small SM87 residual-norm MLP fallback");
  expect_fallback_matches(
      runtime::ProjectionBackend::kSm87WeightOnly, nullptr, 0U, true,
      "small SM87 Decode-runner residual-norm MLP fallback");
  expect_fallback_matches(runtime::ProjectionBackend::kReference,
                          scratch.get(), kRows, false,
                          "small reference residual-norm MLP fallback");
  expect_fallback_matches(
      runtime::ProjectionBackend::kReference, scratch.get(), kRows, true,
      "small reference Decode-runner residual-norm MLP fallback");
}

void test_nvfp4_mlp_down_residual_norm_dispatch(TestContext& test) {
  constexpr std::size_t kRows = 5'120U;
  constexpr std::size_t kExactColumns = 17'408U;
  constexpr std::size_t kOutputBytes =
      kRows * sizeof(std::uint16_t);
  constexpr float kWeightScale = 1.0F / 64.0F;
  constexpr float kEpsilon = 1.0e-6F;

  const auto* const fake_packed =
      reinterpret_cast<const std::uint8_t*>(0x10'0000'0000ULL);
  const auto* const fake_block_scales =
      reinterpret_cast<const std::uint8_t*>(0x20'0000'0000ULL);
  const auto* const fake_activation =
      reinterpret_cast<const std::uint16_t*>(0x30'0000'0000ULL);
  const auto* const fake_residual_left =
      reinterpret_cast<const std::uint16_t*>(0x40'0000'0000ULL);
  const auto* const fake_norm_weight =
      reinterpret_cast<const std::uint16_t*>(0x50'0000'0000ULL);
  auto* const fake_raw =
      reinterpret_cast<std::uint16_t*>(0x60'0000'0000ULL);
  auto* const fake_residual =
      reinterpret_cast<std::uint16_t*>(0x70'0000'0000ULL);
  auto* const fake_normalized =
      reinterpret_cast<std::uint16_t*>(0x80'0000'0000ULL);
  const auto* const fake_weight_scale =
      reinterpret_cast<const float*>(0x90'0000'0000ULL);
  const auto* const fake_input_scale =
      reinterpret_cast<const float*>(0x91'0000'0000ULL);
  auto* const fake_scratch =
      reinterpret_cast<float*>(0xa0'0000'0000ULL);

  const runtime::LinearWeight exact = runtime::NvFp4LinearWeight{
      fake_packed, fake_block_scales, fake_weight_scale, fake_input_scale,
      kWeightScale, 1.0F, kRows, kExactColumns};
  test.expect(runtime::supports_nvfp4_down_residual_norm_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, exact),
              "exact NVFP4 down payload selects residual/norm fusion");
  test.expect(!runtime::supports_nvfp4_down_residual_norm_fusion(
                  runtime::ProjectionBackend::kReference, exact),
              "reference backend does not select down residual/norm fusion");
  const runtime::LinearWeight near_columns = runtime::NvFp4LinearWeight{
      fake_packed, fake_block_scales, fake_weight_scale, fake_input_scale,
      kWeightScale, 1.0F, kRows, kExactColumns - 16U};
  test.expect(!runtime::supports_nvfp4_down_residual_norm_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  near_columns),
              "near-column NVFP4 down shape does not select fusion");
  const runtime::LinearWeight near_rows = runtime::NvFp4LinearWeight{
      fake_packed, fake_block_scales, fake_weight_scale, fake_input_scale,
      kWeightScale, 1.0F, kRows - 1U, kExactColumns};
  test.expect(!runtime::supports_nvfp4_down_residual_norm_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, near_rows),
              "near-row NVFP4 down shape does not select fusion");
  const runtime::LinearWeight invalid_payload = runtime::NvFp4LinearWeight{
      fake_packed, nullptr, fake_weight_scale, fake_input_scale,
      kWeightScale, 1.0F, kRows, kExactColumns};
  test.expect(!runtime::supports_nvfp4_down_residual_norm_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  invalid_payload),
              "incomplete NVFP4 down payload does not select fusion");
  const runtime::LinearWeight wrong_type = runtime::Bf16LinearWeight{
      reinterpret_cast<const std::uint16_t*>(fake_packed), kRows,
      kExactColumns};
  test.expect(!runtime::supports_nvfp4_down_residual_norm_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  wrong_type),
              "BF16 down payload does not select NVFP4 fusion");

  const auto launch = [&](const runtime::ProjectionBackend backend,
                          const runtime::LinearWeight& weight,
                          const std::uint16_t* activation,
                          const std::uint16_t* residual_left,
                          const std::uint16_t* norm_weight,
                          const float epsilon, float* scratch,
                          const std::size_t scratch_elements,
                          std::uint16_t* raw, std::uint16_t* residual,
                          std::uint16_t* normalized,
                          cudaStream_t stream) noexcept {
    return runtime::launch_mlp_down_residual_norm_to_bf16_cuda(
        backend, weight, activation, residual_left, norm_weight, epsilon,
        scratch, scratch_elements, raw, residual, normalized,
        static_cast<void*>(stream));
  };

  std::size_t exact_total_nodes = 0U;
  const std::size_t exact_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, nullptr, 0U, fake_raw, fake_residual,
                      fake_normalized, stream);
      },
      "exact down residual/norm fusion graph", &exact_total_nodes);
  test.expect(exact_total_nodes == 1U && exact_kernel_nodes == 1U,
              "exact aligned NVFP4 down residual/norm graph has one kernel");

  std::size_t read_overlap_total_nodes = 0U;
  const std::size_t read_overlap_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left,
                      fake_residual_left, kEpsilon, nullptr, 0U, fake_raw,
                      fake_residual, fake_normalized, stream);
      },
      "safe read-only overlap down residual/norm graph",
      &read_overlap_total_nodes);
  test.expect(read_overlap_total_nodes == 1U &&
                  read_overlap_kernel_nodes == 1U,
              "read-only residual/norm overlap remains fusion-eligible");

  const auto* const unaligned_activation =
      reinterpret_cast<const std::uint16_t*>(0x30'0000'0002ULL);
  std::size_t unaligned_total_nodes = 0U;
  bool unaligned_linear = false;
  const std::size_t unaligned_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      unaligned_activation, fake_residual_left,
                      fake_norm_weight, kEpsilon, nullptr, 0U, fake_raw,
                      fake_residual, fake_normalized, stream);
      },
      "unaligned down residual/norm fallback graph", &unaligned_total_nodes,
      &unaligned_linear);
  test.expect(unaligned_total_nodes == 2U && unaligned_kernel_nodes == 2U &&
                  unaligned_linear,
              "unaligned exact down preserves projection then norm order");

  std::size_t near_total_nodes = 0U;
  bool near_linear = false;
  const std::size_t near_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly,
                      near_columns, fake_activation, fake_residual_left,
                      fake_norm_weight, kEpsilon, nullptr, 0U, fake_raw,
                      fake_residual, fake_normalized, stream);
      },
      "near-shape down residual/norm fallback graph", &near_total_nodes,
      &near_linear);
  test.expect(near_total_nodes == 2U && near_kernel_nodes == 2U &&
                  near_linear,
              "near-shape down preserves projection then norm order");

  std::size_t reference_total_nodes = 0U;
  bool reference_linear = false;
  const std::size_t reference_kernel_nodes = captured_kernel_node_count(
      test,
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kReference, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, fake_scratch, kRows, fake_raw,
                      fake_residual, fake_normalized, stream);
      },
      "reference down residual/norm fallback graph", &reference_total_nodes,
      &reference_linear);
  test.expect(reference_total_nodes == 3U && reference_kernel_nodes == 3U &&
                  reference_linear,
              "reference down preserves projection-convert-norm order");

  const auto expect_invalid = [&](auto&& invalid_launch,
                                  const std::string& label) {
    expect_invalid_capture_has_no_nodes(test,
                                        std::forward<decltype(invalid_launch)>(
                                            invalid_launch),
                                        label);
  };
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly,
                      invalid_payload, fake_activation, fake_residual_left,
                      fake_norm_weight, kEpsilon, nullptr, 0U, fake_raw,
                      fake_residual, fake_normalized, stream);
      },
      "down residual/norm rejects invalid weight payload");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, near_rows,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, nullptr, 0U, fake_raw, fake_residual,
                      fake_normalized, stream);
      },
      "down residual/norm rejects non-hidden output before projection");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, nullptr, fake_norm_weight, kEpsilon,
                      nullptr, 0U, fake_raw, fake_residual, fake_normalized,
                      stream);
      },
      "down residual/norm rejects null residual input");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left, nullptr, kEpsilon,
                      nullptr, 0U, fake_raw, fake_residual, fake_normalized,
                      stream);
      },
      "down residual/norm rejects null norm weight");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      0.0F, nullptr, 0U, fake_raw, fake_residual,
                      fake_normalized, stream);
      },
      "down residual/norm rejects zero epsilon");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      std::numeric_limits<float>::quiet_NaN(), nullptr, 0U,
                      fake_raw, fake_residual, fake_normalized, stream);
      },
      "down residual/norm rejects NaN epsilon");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, nullptr, 0U, nullptr, fake_residual,
                      fake_normalized, stream);
      },
      "down residual/norm rejects null raw output");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, nullptr, 0U, fake_raw, nullptr,
                      fake_normalized, stream);
      },
      "down residual/norm rejects null residual output");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, nullptr, 0U, fake_raw, fake_residual, nullptr,
                      stream);
      },
      "down residual/norm rejects null normalized output");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, nullptr, 0U,
                      const_cast<std::uint16_t*>(fake_residual_left),
                      fake_residual, fake_normalized, stream);
      },
      "raw down output rejects residual-input alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, nullptr, 0U, fake_raw,
                      const_cast<std::uint16_t*>(fake_norm_weight),
                      fake_normalized, stream);
      },
      "residual output rejects norm-weight alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(
            runtime::ProjectionBackend::kSm87WeightOnly, exact,
            fake_activation, fake_residual_left, fake_norm_weight, kEpsilon,
            nullptr, 0U, fake_raw, fake_residual,
            const_cast<std::uint16_t*>(fake_residual_left), stream);
      },
      "normalized output rejects residual-input alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(
            runtime::ProjectionBackend::kSm87WeightOnly, exact,
            fake_activation, fake_residual_left, fake_norm_weight, kEpsilon,
            nullptr, 0U, fake_raw,
            const_cast<std::uint16_t*>(fake_activation), fake_normalized,
            stream);
      },
      "residual output rejects activation alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, nullptr, 0U, fake_raw, fake_raw,
                      fake_normalized, stream);
      },
      "residual output rejects raw-down alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kSm87WeightOnly, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, nullptr, 0U, fake_raw, fake_residual,
                      fake_residual, stream);
      },
      "normalized output rejects residual-output alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(
            runtime::ProjectionBackend::kSm87WeightOnly, exact,
            fake_activation, fake_residual_left, fake_norm_weight, kEpsilon,
            nullptr, 0U, fake_raw,
            reinterpret_cast<std::uint16_t*>(
                const_cast<std::uint8_t*>(fake_packed)),
            fake_normalized, stream);
      },
      "residual output rejects packed-weight alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(
            runtime::ProjectionBackend::kSm87WeightOnly, exact,
            fake_activation, fake_residual_left, fake_norm_weight, kEpsilon,
            nullptr, 0U, fake_raw, fake_residual,
            reinterpret_cast<std::uint16_t*>(
                const_cast<std::uint8_t*>(fake_block_scales)),
            stream);
      },
      "normalized output rejects block-scale alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(
            runtime::ProjectionBackend::kSm87WeightOnly, exact,
            fake_activation, fake_residual_left, fake_norm_weight, kEpsilon,
            nullptr, 0U,
            reinterpret_cast<std::uint16_t*>(
                const_cast<float*>(fake_weight_scale)),
            fake_residual, fake_normalized, stream);
      },
      "raw output rejects weight device-scalar alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(
            runtime::ProjectionBackend::kSm87WeightOnly, exact,
            fake_activation, fake_residual_left, fake_norm_weight, kEpsilon,
            nullptr, 0U, fake_raw, fake_residual,
            reinterpret_cast<std::uint16_t*>(
                const_cast<float*>(fake_input_scale)),
            stream);
      },
      "normalized output rejects device-scalar alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kReference, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, fake_scratch, kRows - 1U, fake_raw,
                      fake_residual, fake_normalized, stream);
      },
      "reference down residual/norm rejects short scratch");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(
            runtime::ProjectionBackend::kReference, exact, fake_activation,
            fake_residual_left, fake_norm_weight, kEpsilon,
            reinterpret_cast<float*>(
                const_cast<std::uint16_t*>(fake_norm_weight)),
            kRows, fake_raw, fake_residual, fake_normalized, stream);
      },
      "reference down residual/norm rejects scratch/norm alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        return launch(runtime::ProjectionBackend::kReference, exact,
                      fake_activation, fake_residual_left, fake_norm_weight,
                      kEpsilon, reinterpret_cast<float*>(fake_residual),
                      kRows, fake_raw, fake_residual, fake_normalized,
                      stream);
      },
      "reference down residual/norm rejects scratch/output alias");
  expect_invalid(
      [&](cudaStream_t stream) noexcept {
        const std::uintptr_t near_end =
            std::numeric_limits<std::uintptr_t>::max() -
            (kOutputBytes / 2U);
        return launch(
            runtime::ProjectionBackend::kSm87WeightOnly, exact,
            fake_activation, fake_residual_left, fake_norm_weight, kEpsilon,
            nullptr, 0U, fake_raw, fake_residual,
            reinterpret_cast<std::uint16_t*>(near_end), stream);
      },
      "down residual/norm rejects wrapping normalized range");

  constexpr std::size_t kSmallColumns = 16U;
  constexpr std::size_t kPackedBytes = kRows * (kSmallColumns / 2U);
  constexpr std::size_t kScaleBytes = kRows * (kSmallColumns / 16U);
  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> block_scales;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> residual_left;
  DeviceBuffer<std::uint16_t> norm_weight;
  DeviceBuffer<std::uint16_t> baseline_raw;
  DeviceBuffer<std::uint16_t> baseline_residual;
  DeviceBuffer<std::uint16_t> baseline_normalized;
  DeviceBuffer<std::uint16_t> candidate_raw;
  DeviceBuffer<std::uint16_t> candidate_residual;
  DeviceBuffer<std::uint16_t> candidate_normalized;
  bool ready = packed.allocate(test, kPackedBytes,
                               "small down packed weights");
  ready = ready && block_scales.allocate(test, kScaleBytes,
                                          "small down block scales");
  ready = ready && companion_scales.allocate(test, 2U,
                                              "small down companion scales");
  ready = ready && activation.allocate(test, kSmallColumns,
                                        "small down activation");
  ready = ready && residual_left.allocate(test, kRows,
                                           "small down residual input");
  ready = ready && norm_weight.allocate(test, kRows,
                                         "small down norm weight");
  ready = ready && baseline_raw.allocate(test, kRows,
                                          "small down baseline raw");
  ready = ready && baseline_residual.allocate(test, kRows,
                                               "small down baseline residual");
  ready = ready && baseline_normalized.allocate(
                       test, kRows, "small down baseline normalized");
  ready = ready && candidate_raw.allocate(test, kRows,
                                           "small down candidate raw");
  ready = ready && candidate_residual.allocate(
                       test, kRows, "small down candidate residual");
  ready = ready && candidate_normalized.allocate(
                       test, kRows, "small down candidate normalized");
  if (!ready) {
    return;
  }

  std::vector<std::uint16_t> host_residual(kRows);
  std::vector<std::uint16_t> host_norm(kRows);
  for (std::size_t row = 0U; row < kRows; ++row) {
    host_residual[row] =
        row % 3U == 0U ? 0x3f80U : (row % 3U == 1U ? 0xbf00U : 0x3e80U);
    host_norm[row] = row % 2U == 0U ? 0x3f80U : 0x3f00U;
  }
  ready = upload(test, packed,
                 std::vector<std::uint8_t>(kPackedBytes, 0x21U),
                 "small down packed weights");
  ready = ready && upload(test, block_scales,
                           std::vector<std::uint8_t>(kScaleBytes, 0x38U),
                           "small down block scales");
  ready = ready && upload(test, companion_scales,
                           std::vector<float>{kWeightScale, 1.0F},
                           "small down companion scales");
  ready = ready && upload(
                       test, activation,
                       std::vector<std::uint16_t>{
                           0x3f80U, 0x3f00U, 0xbf80U, 0x4000U,
                           0x3e80U, 0xbf00U, 0x4040U, 0xc000U,
                           0x3f40U, 0xbf40U, 0x3fc0U, 0xbfc0U,
                           0x4080U, 0xc080U, 0x3e00U, 0xbe00U},
                       "small down activation");
  ready = ready && upload(test, residual_left, host_residual,
                           "small down residual input");
  ready = ready && upload(test, norm_weight, host_norm,
                           "small down norm weight");
  if (!ready) {
    return;
  }
  const runtime::LinearWeight small = runtime::NvFp4LinearWeight{
      packed.get(), block_scales.get(), companion_scales.get(),
      companion_scales.get() + 1U, kWeightScale, 1.0F, kRows,
      kSmallColumns};

  int status = runtime::launch_projection_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, small, activation.get(),
      nullptr, 0U, baseline_raw.get());
  if (static_cast<cudaError_t>(status) == cudaSuccess) {
    status = runtime::launch_residual_add_centered_rms_norm_5120_cuda(
        residual_left.get(), baseline_raw.get(), norm_weight.get(), kEpsilon,
        baseline_residual.get(), baseline_normalized.get());
  }
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "small down old projection/norm chain succeeds");
  status = runtime::launch_mlp_down_residual_norm_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, small, activation.get(),
      residual_left.get(), norm_weight.get(), kEpsilon, nullptr, 0U,
      candidate_raw.get(), candidate_residual.get(),
      candidate_normalized.get());
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "small down composite fallback succeeds");
  if (static_cast<cudaError_t>(status) != cudaSuccess ||
      !test.cuda_ok(cudaDeviceSynchronize(),
                    "synchronize small down composite fallback")) {
    return;
  }

  const auto compare = [&](const DeviceBuffer<std::uint16_t>& actual,
                           const DeviceBuffer<std::uint16_t>& expected,
                           const std::string& label) {
    std::vector<std::uint16_t> actual_host(kRows);
    std::vector<std::uint16_t> expected_host(kRows);
    bool copied = test.cuda_ok(
        cudaMemcpy(actual_host.data(), actual.get(), kOutputBytes,
                   cudaMemcpyDeviceToHost),
        "download candidate " + label);
    copied = copied && test.cuda_ok(
                           cudaMemcpy(expected_host.data(), expected.get(),
                                      kOutputBytes, cudaMemcpyDeviceToHost),
                           "download baseline " + label);
    if (copied) {
      test.expect(actual_host == expected_host,
                  "small down " + label + " is bitwise old chain");
    }
  };
  compare(candidate_raw, baseline_raw, "raw output");
  compare(candidate_residual, baseline_residual, "residual output");
  compare(candidate_normalized, baseline_normalized, "normalized output");
}

void test_nvfp4_mlp_down_scale6_dispatch(TestContext& test) {
  constexpr std::size_t kRows = runtime::kNvFp4DownScale6Rows;
  constexpr std::size_t kColumns = runtime::kNvFp4DownScale6Columns;
  constexpr std::size_t kPackedBytes = kRows * kColumns / 2U;
  constexpr std::size_t kCanonicalScaleBytes = kRows * kColumns / 16U;
  constexpr std::size_t kScale6Bytes =
      runtime::kNvFp4DownScale6SidecarBytesPerProjection;
  constexpr std::size_t kOutputBytes = kRows * sizeof(std::uint16_t);
  constexpr unsigned int kScaleBase = 0x38U;
  constexpr float kWeightScale = 1.0F / 64.0F;
  constexpr float kEpsilon = 1.0e-6F;

  const auto* const fake_packed =
      reinterpret_cast<const std::uint8_t*>(0x10'0000'0000ULL);
  const auto* const fake_block_scales =
      reinterpret_cast<const std::uint8_t*>(0x20'0000'0000ULL);
  const auto* const fake_scale6 =
      reinterpret_cast<const std::uint8_t*>(0x21'0000'0000ULL);
  const auto* const fake_activation =
      reinterpret_cast<const std::uint16_t*>(0x30'0000'0000ULL);
  const auto* const fake_residual_left =
      reinterpret_cast<const std::uint16_t*>(0x40'0000'0000ULL);
  const auto* const fake_norm_weight =
      reinterpret_cast<const std::uint16_t*>(0x50'0000'0000ULL);
  auto* const fake_raw =
      reinterpret_cast<std::uint16_t*>(0x60'0000'0000ULL);
  auto* const fake_residual =
      reinterpret_cast<std::uint16_t*>(0x70'0000'0000ULL);
  auto* const fake_normalized =
      reinterpret_cast<std::uint16_t*>(0x80'0000'0000ULL);
  const auto* const fake_weight_scale =
      reinterpret_cast<const float*>(0x90'0000'0000ULL);
  const auto* const fake_input_scale =
      reinterpret_cast<const float*>(0x91'0000'0000ULL);

  const runtime::LinearWeight canonical = runtime::NvFp4LinearWeight{
      fake_packed, fake_block_scales, fake_weight_scale, fake_input_scale,
      kWeightScale, 1.0F, kRows, kColumns};
  runtime::LinearWeight attached = canonical;
  auto& attached_nvfp4 = std::get<runtime::NvFp4LinearWeight>(attached);
  attached_nvfp4.down_scale6_sidecar = fake_scale6;
  attached_nvfp4.down_scale6_base = kScaleBase;

  const auto dispatch = [&](const runtime::LinearWeight& weight,
                            cudaStream_t stream) noexcept {
    return runtime::launch_mlp_down_residual_norm_to_bf16_cuda(
        runtime::ProjectionBackend::kSm87WeightOnly, weight,
        fake_activation, fake_residual_left, fake_norm_weight, kEpsilon,
        nullptr, 0U, fake_raw, fake_residual, fake_normalized,
        static_cast<void*>(stream));
  };
  const CapturedKernelChain canonical_dispatch =
      capture_ordered_kernel_chain(
          test,
          [&](cudaStream_t stream) noexcept {
            return dispatch(canonical, stream);
          },
          "null-sidecar canonical down dispatch graph");
  const CapturedKernelChain canonical_oracle =
      capture_ordered_kernel_chain(
          test,
          [&](cudaStream_t stream) noexcept {
            return q3x::kernels::
                launch_sm87_nvfp4_w4a16_down_residual_norm_bf16_cuda(
                    fake_packed, fake_block_scales, kWeightScale,
                    fake_activation, fake_residual_left, fake_norm_weight,
                    kEpsilon, kRows, kColumns, fake_raw, fake_residual,
                    fake_normalized, static_cast<void*>(stream));
          },
          "canonical down .cs direct oracle graph");
  const CapturedKernelChain scale6_dispatch = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return dispatch(attached, stream);
      },
      "attached scale6 down dispatch graph");
  const CapturedKernelChain scale6_oracle = capture_ordered_kernel_chain(
      test,
      [&](cudaStream_t stream) noexcept {
        return q3x::kernels::
            launch_sm87_nvfp4_w4a16_down_residual_norm_scale6_bf16_cuda(
                fake_packed, fake_scale6, kScaleBase, kWeightScale,
                fake_activation, fake_residual_left, fake_norm_weight,
                kEpsilon, kRows, kColumns, fake_raw, fake_residual,
                fake_normalized, static_cast<void*>(stream));
      },
      "scale6 down direct oracle graph");
  const auto is_exact_one_node = [](const CapturedKernelChain& graph) {
    return graph.valid && graph.launches.size() == 1U &&
           graph.launches.front().function != nullptr &&
           graph.launches.front().grid.x == 32U &&
           graph.launches.front().grid.y == 1U &&
           graph.launches.front().grid.z == 1U &&
           graph.launches.front().block.x == 512U &&
           graph.launches.front().block.y == 1U &&
           graph.launches.front().block.z == 1U &&
           graph.launches.front().dynamic_shared_bytes == 0U;
  };
  const bool graph_gate =
      is_exact_one_node(canonical_dispatch) &&
      is_exact_one_node(canonical_oracle) &&
      is_exact_one_node(scale6_dispatch) &&
      is_exact_one_node(scale6_oracle) &&
      canonical_dispatch.launches.front().function ==
          canonical_oracle.launches.front().function &&
      scale6_dispatch.launches.front().function ==
          scale6_oracle.launches.front().function &&
      canonical_dispatch.launches.front().function !=
          scale6_dispatch.launches.front().function;
  test.expect(graph_gate,
              "attached scale6 selects its distinct 32x512 Function while "
              "null sidecar preserves canonical .cs");
  if (!graph_gate) {
    return;
  }

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> block_scales;
  DeviceBuffer<std::uint8_t> scale6;
  DeviceBuffer<float> companion_scales;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> residual_left;
  DeviceBuffer<std::uint16_t> norm_weight;
  DeviceBuffer<std::uint16_t> canonical_raw;
  DeviceBuffer<std::uint16_t> canonical_residual;
  DeviceBuffer<std::uint16_t> canonical_normalized;
  DeviceBuffer<std::uint16_t> scale6_raw;
  DeviceBuffer<std::uint16_t> scale6_residual;
  DeviceBuffer<std::uint16_t> scale6_normalized;
  bool ready = packed.allocate(test, kPackedBytes,
                               "scale6 dispatch packed weights");
  ready = ready && block_scales.allocate(
                       test, kCanonicalScaleBytes,
                       "scale6 dispatch canonical block scales");
  ready = ready && scale6.allocate(test, kScale6Bytes,
                                   "scale6 dispatch sidecar");
  ready = ready && companion_scales.allocate(
                       test, 2U, "scale6 dispatch companion scales");
  ready = ready && activation.allocate(test, kColumns,
                                        "scale6 dispatch activation");
  ready = ready && residual_left.allocate(
                       test, kRows, "scale6 dispatch residual input");
  ready = ready && norm_weight.allocate(test, kRows,
                                         "scale6 dispatch norm weight");
  ready = ready && canonical_raw.allocate(
                       test, kRows, "canonical down raw output");
  ready = ready && canonical_residual.allocate(
                       test, kRows, "canonical down residual output");
  ready = ready && canonical_normalized.allocate(
                       test, kRows, "canonical down normalized output");
  ready = ready && scale6_raw.allocate(test, kRows,
                                        "scale6 down raw output");
  ready = ready && scale6_residual.allocate(
                       test, kRows, "scale6 down residual output");
  ready = ready && scale6_normalized.allocate(
                       test, kRows, "scale6 down normalized output");
  if (!ready) {
    return;
  }

  ready = test.cuda_ok(cudaMemset(packed.get(), 0x11, kPackedBytes),
                       "initialize scale6 dispatch packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemset(block_scales.get(), kScaleBase,
                                  kCanonicalScaleBytes),
                       "initialize canonical block scales");
  ready = ready && test.cuda_ok(cudaMemset(scale6.get(), 0, kScale6Bytes),
                                "initialize zero-delta scale6 sidecar");
  ready = ready && upload(
                       test, companion_scales,
                       std::vector<float>{kWeightScale, 1.0F},
                       "scale6 dispatch companion scales");
  std::vector<std::uint16_t> host_activation(kColumns, 0x3d80U);
  for (std::size_t column = 1U; column < kColumns; column += 2U) {
    host_activation[column] = 0xbd00U;
  }
  std::vector<std::uint16_t> host_residual(kRows);
  std::vector<std::uint16_t> host_norm(kRows);
  for (std::size_t row = 0U; row < kRows; ++row) {
    host_residual[row] =
        row % 3U == 0U ? 0x3f80U : (row % 3U == 1U ? 0xbf00U : 0x3e80U);
    host_norm[row] = row % 2U == 0U ? 0x0000U : 0x3e80U;
  }
  ready = ready && upload(test, activation, host_activation,
                           "scale6 dispatch activation");
  ready = ready && upload(test, residual_left, host_residual,
                           "scale6 dispatch residual input");
  ready = ready && upload(test, norm_weight, host_norm,
                           "scale6 dispatch norm weight");
  if (!ready) {
    return;
  }

  const runtime::LinearWeight canonical_device =
      runtime::NvFp4LinearWeight{
          packed.get(), block_scales.get(), companion_scales.get(),
          companion_scales.get() + 1U, kWeightScale, 1.0F, kRows, kColumns};
  runtime::LinearWeight scale6_device = canonical_device;
  auto& scale6_nvfp4 =
      std::get<runtime::NvFp4LinearWeight>(scale6_device);
  scale6_nvfp4.down_scale6_sidecar = scale6.get();
  scale6_nvfp4.down_scale6_base = kScaleBase;

  int status = runtime::launch_mlp_down_residual_norm_to_bf16_cuda(
      runtime::ProjectionBackend::kSm87WeightOnly, canonical_device,
      activation.get(), residual_left.get(), norm_weight.get(), kEpsilon,
      nullptr, 0U, canonical_raw.get(), canonical_residual.get(),
      canonical_normalized.get());
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "null-sidecar canonical .cs dispatch executes");
  if (static_cast<cudaError_t>(status) == cudaSuccess) {
    status = runtime::launch_mlp_down_residual_norm_to_bf16_cuda(
        runtime::ProjectionBackend::kSm87WeightOnly, scale6_device,
        activation.get(), residual_left.get(), norm_weight.get(), kEpsilon,
        nullptr, 0U, scale6_raw.get(), scale6_residual.get(),
        scale6_normalized.get());
  }
  test.expect(static_cast<cudaError_t>(status) == cudaSuccess,
              "attached scale6 dispatch executes");
  if (static_cast<cudaError_t>(status) != cudaSuccess ||
      !test.cuda_ok(cudaDeviceSynchronize(),
                    "synchronize scale6 dispatch numerical fixture")) {
    return;
  }

  const auto expect_bitwise_equal =
      [&](const DeviceBuffer<std::uint16_t>& actual,
          const DeviceBuffer<std::uint16_t>& expected,
          const std::string& label) {
        std::vector<std::uint16_t> actual_host(kRows);
        std::vector<std::uint16_t> expected_host(kRows);
        bool copied = test.cuda_ok(
            cudaMemcpy(actual_host.data(), actual.get(), kOutputBytes,
                       cudaMemcpyDeviceToHost),
            "download scale6 " + label);
        copied = copied && test.cuda_ok(
                               cudaMemcpy(expected_host.data(), expected.get(),
                                          kOutputBytes, cudaMemcpyDeviceToHost),
                               "download canonical " + label);
        if (copied) {
          test.expect(actual_host == expected_host,
                      "attached scale6 " + label +
                          " is bitwise canonical .cs");
        }
      };
  expect_bitwise_equal(scale6_raw, canonical_raw, "raw output");
  expect_bitwise_equal(scale6_residual, canonical_residual,
                       "residual output");
  expect_bitwise_equal(scale6_normalized, canonical_normalized,
                       "normalized output");
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
  constexpr std::size_t kTokens = runtime::kMaximumProjectionTileTokenCount;

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
                 activation.get(), kTokens + 1U, nullptr, 0U, output.get(),
                 "tile rejects M=65");
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
                 "C64 validation activation sentinel");
  ready = ready && test.cuda_ok(
                       cudaMemset(fp8_weight.get(), 0x38,
                                  kRows * kFp8Columns),
                       "initialize C64 validation FP8 weights");
  if (ready) {
    std::uint16_t* const overlapping_output =
        activation.get() + 33U * kFp8Columns;
    const int overlap_status =
        runtime::launch_projection_tile_to_bf16_cuda(
            runtime::ProjectionBackend::kSm87WeightOnly, fp8,
            activation.get(), kTokens, nullptr, 0U, overlapping_output);
    test.expect(static_cast<cudaError_t>(overlap_status) ==
                    cudaErrorInvalidValue,
                "C64 tile rejects output overlapping only the second C32 input");
    std::uint16_t preserved = 0U;
    ready = test.cuda_ok(cudaMemcpy(&preserved, overlapping_output,
                                    sizeof(preserved),
                                    cudaMemcpyDeviceToHost),
                         "read C64 validation activation sentinel");
    if (ready) {
      test.expect(
          preserved == 0x3f80U,
          "C64 whole-tile validation rejects before the first SM87 launch");
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
  test_fp8_m1_output_sidecar_dispatch(test);
  test_bf16_direct_production_dispatch(test);
  test_tile_routes(test);
  test_exact_fp8_whole_chunk_projection_dispatch(test);
  test_exact_nvfp4_whole_chunk_branch_dispatch(test);
  test_bf16_projection_pair_dispatch(test);
  test_fp8_projection_pair_dispatch(test);
  test_fp8_qkv_z_projection_pair_dispatch(test);
  test_fp8_full_attention_q_kv_dispatch(test);
  test_nvfp4_mlp_gate_up_silu_dispatch(test);
  test_post_attention_residual_norm_mlp_gate_up_silu_dispatch(test);
  test_nvfp4_mlp_down_residual_norm_dispatch(test);
  test_nvfp4_mlp_down_scale6_dispatch(test);
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
