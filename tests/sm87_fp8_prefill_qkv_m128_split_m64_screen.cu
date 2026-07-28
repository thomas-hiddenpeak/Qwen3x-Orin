#include "q3x/kernels/sm87_weight_only_gemv.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace q3x::kernels {

[[nodiscard]] int
launch_sm87_fp8_w8a16_whole_chunk_qkv_m128_split_m64_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream) noexcept;

[[nodiscard]] int
query_sm87_fp8_w8a16_whole_chunk_qkv_m128_split_m64_resources_test_cuda(
    std::size_t token_count, std::size_t rows, std::size_t columns,
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* maximum_threads_per_block, int* active_blocks_per_sm) noexcept;

}  // namespace q3x::kernels

namespace {

constexpr std::size_t kTokens = 512U;
constexpr std::size_t kRows = 10'240U;
constexpr std::size_t kColumns = 5'120U;
constexpr std::size_t kWeightBytes = kRows * kColumns;
constexpr std::size_t kActivationElements = kTokens * kColumns;
constexpr std::size_t kOutputElements = kTokens * kRows;
constexpr std::size_t kGuardElements = 64U;
constexpr std::size_t kScrubBytes = 32U * 1024U * 1024U;
constexpr float kWeightScale = 0.00100708F;
constexpr int kWarmups = 10;
constexpr int kIterations = 24;
constexpr int kRounds = 6;
constexpr double kRequiredAggregateSpeedup = 1.20;
constexpr double kRequiredRoundSpeedup = 1.15;
constexpr std::array<std::uint8_t, 16U> kCheckpointCodes{{
    0x00U, 0x80U, 0x18U, 0x20U, 0x28U, 0x30U, 0x38U, 0x3cU,
    0xb8U, 0x40U, 0xc0U, 0x48U, 0x50U, 0xd0U, 0x70U, 0xf0U,
}};

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
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      test.expect(false, label + " size is representable");
      return false;
    }
    return test.cuda_ok(
        cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
        label);
  }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
};

class Execution {
 public:
  Execution() = default;
  Execution(const Execution&) = delete;
  Execution& operator=(const Execution&) = delete;
  ~Execution() {
    if (stop_ != nullptr) {
      (void)cudaEventDestroy(stop_);
    }
    if (start_ != nullptr) {
      (void)cudaEventDestroy(start_);
    }
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  [[nodiscard]] bool create(TestContext& test) {
    bool ready = test.cuda_ok(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
        "create nonblocking stream");
    ready = ready && test.cuda_ok(cudaEventCreate(&start_),
                                  "create timing start");
    ready = ready &&
            test.cuda_ok(cudaEventCreate(&stop_), "create timing stop");
    return ready;
  }

  [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
  [[nodiscard]] cudaEvent_t start() const noexcept { return start_; }
  [[nodiscard]] cudaEvent_t stop() const noexcept { return stop_; }

 private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

class CapturedGraph {
 public:
  CapturedGraph() = default;
  CapturedGraph(const CapturedGraph&) = delete;
  CapturedGraph& operator=(const CapturedGraph&) = delete;
  ~CapturedGraph() {
    if (executable_ != nullptr) {
      (void)cudaGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      (void)cudaGraphDestroy(graph_);
    }
  }

  [[nodiscard]] cudaGraph_t* graph_address() noexcept { return &graph_; }
  [[nodiscard]] cudaGraphExec_t* executable_address() noexcept {
    return &executable_;
  }
  [[nodiscard]] cudaGraph_t graph() const noexcept { return graph_; }
  [[nodiscard]] cudaGraphExec_t executable() const noexcept {
    return executable_;
  }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  const std::uint32_t upper = bits >> 16U;
  const std::uint32_t rounding_bias = 0x7fffU + (upper & 1U);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16U);
}

[[nodiscard]] bool is_bf16_nan(const std::uint16_t bits) noexcept {
  return (bits & 0x7f80U) == 0x7f80U && (bits & 0x007fU) != 0U;
}

[[nodiscard]] bool is_bf16_finite(const std::uint16_t bits) noexcept {
  return (bits & 0x7f80U) != 0x7f80U;
}

__global__ void scrub_l2_kernel(std::uint32_t* const words,
                                const std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t current = index; current < count; current += stride) {
    words[current] = words[current] * 1'664'525U + 1'013'904'223U;
  }
}

struct Fixture {
  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> output_store;
  DeviceBuffer<std::uint32_t> scrub;
  std::vector<std::uint8_t> host_weights;
  std::vector<std::uint16_t> host_activations;

  [[nodiscard]] std::uint16_t* output() noexcept {
    return output_store.get() + kGuardElements;
  }
  [[nodiscard]] std::size_t guarded_output_elements() const noexcept {
    return kOutputElements + 2U * kGuardElements;
  }

  [[nodiscard]] bool initialize(TestContext& test,
                                const cudaStream_t stream) {
    host_weights.resize(kWeightBytes);
    host_activations.resize(kActivationElements);
    for (std::size_t index = 0U; index < host_weights.size(); ++index) {
      const std::size_t row = index / kColumns;
      const std::size_t column = index - row * kColumns;
      host_weights[index] = kCheckpointCodes[
          (index * 5U + row * 11U + (column >> 4U)) %
          kCheckpointCodes.size()];
    }
    // The first 1,024 bytes cover all 256 E4M3FN codes in each of the four
    // byte positions of an aligned uint32 word. The remaining payload stays
    // checkpoint-like and finite for the timing cell.
    for (std::size_t code = 0U; code < 256U; ++code) {
      for (std::size_t position = 0U; position < 4U; ++position) {
        host_weights[code * 4U + position] =
            static_cast<std::uint8_t>(code);
      }
    }
    for (std::size_t index = 0U; index < host_activations.size(); ++index) {
      const int centered = static_cast<int>(index % 31U) - 15;
      host_activations[index] =
          encode_bf16(static_cast<float>(centered) / 16.0F);
    }

    bool ready = weights.allocate(test, kWeightBytes,
                                  "allocate FP8 QKV weights");
    ready = ready && activations.allocate(test, kActivationElements,
                                           "allocate BF16 activations");
    ready = ready && output_store.allocate(test, guarded_output_elements(),
                                            "allocate guarded output");
    ready = ready && scrub.allocate(test, kScrubBytes / sizeof(std::uint32_t),
                                    "allocate L2 scrub");
    if (!ready) {
      return false;
    }
    ready = test.cuda_ok(
        cudaMemcpyAsync(weights.get(), host_weights.data(), kWeightBytes,
                        cudaMemcpyHostToDevice, stream),
        "upload FP8 QKV weights");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             activations.get(), host_activations.data(),
                             kActivationElements * sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice, stream),
                         "upload BF16 activations");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(scrub.get(), 0, kScrubBytes, stream),
                         "initialize L2 scrub");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  "fixture upload synchronize");
    return ready;
  }
};

enum class Variant {
  kBaseline,
  kCandidate,
};

[[nodiscard]] const char* variant_name(const Variant variant) noexcept {
  return variant == Variant::kBaseline ? "baseline" : "candidate";
}

[[nodiscard]] cudaError_t launch_variant(Fixture& fixture,
                                         const cudaStream_t stream,
                                         const Variant variant) noexcept {
  const int status =
      variant == Variant::kBaseline
          ? q3x::kernels::launch_sm87_fp8_w8a16_whole_chunk_gemm_bf16_cuda(
                fixture.weights.get(), kWeightScale, fixture.activations.get(),
                kTokens, kRows, kColumns, fixture.output(),
                static_cast<void*>(stream))
          : q3x::kernels::
                launch_sm87_fp8_w8a16_whole_chunk_qkv_m128_split_m64_test_cuda(
                    fixture.weights.get(), kWeightScale,
                    fixture.activations.get(), kTokens, kRows, kColumns,
                    fixture.output(), static_cast<void*>(stream));
  return static_cast<cudaError_t>(status);
}

[[nodiscard]] bool run_resource_gate(TestContext& test) {
  int registers = -1;
  std::size_t static_shared = std::numeric_limits<std::size_t>::max();
  std::size_t dynamic_shared = std::numeric_limits<std::size_t>::max();
  std::size_t local = std::numeric_limits<std::size_t>::max();
  int threads = -1;
  int active = -1;
  const int status = q3x::kernels::
      query_sm87_fp8_w8a16_whole_chunk_qkv_m128_split_m64_resources_test_cuda(
          kTokens, kRows, kColumns, &registers, &static_shared,
          &dynamic_shared, &local, &threads, &active);
  const bool gate =
      status == static_cast<int>(cudaSuccess) && registers == 64 &&
      static_shared == 33'280U && dynamic_shared == 18'432U &&
      static_shared + dynamic_shared == 51'712U && local == 0U &&
      threads == 512 && active == 2;
  std::cout << "FP8_SPLIT_M64_RESOURCES: status=" << status
            << " registers=" << registers
            << " static_shared_bytes=" << static_shared
            << " dynamic_shared_bytes=" << dynamic_shared
            << " total_shared_bytes=" << static_shared + dynamic_shared
            << " local_bytes=" << local << " threads=" << threads
            << " active_blocks_per_sm=" << active
            << " resident_warps_per_sm=" << active * threads / 32
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "split-M64 candidate clears exact resource gate");
  return gate;
}

[[nodiscard]] bool run_invalid_graph_gate(TestContext& test,
                                          const cudaStream_t stream) {
  constexpr std::uintptr_t kWeightAddress = 0x1'0000'0000ULL;
  constexpr std::uintptr_t kActivationAddress = 0x2'0000'0000ULL;
  constexpr std::uintptr_t kOutputAddress = 0x3'0000'0000ULL;
  constexpr std::uintptr_t kMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const auto* const weights =
      reinterpret_cast<const std::uint8_t*>(kWeightAddress);
  const auto* const activations =
      reinterpret_cast<const std::uint16_t*>(kActivationAddress);
  auto* const output = reinterpret_cast<std::uint16_t*>(kOutputAddress);
  const auto launch =
      [&](const std::uint8_t* const w, const float scale,
          const std::uint16_t* const a, const std::size_t tokens,
          const std::size_t rows, const std::size_t columns,
          std::uint16_t* const o) noexcept {
        return q3x::kernels::
            launch_sm87_fp8_w8a16_whole_chunk_qkv_m128_split_m64_test_cuda(
                w, scale, a, tokens, rows, columns, o,
                static_cast<void*>(stream));
      };

  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "invalid begin capture");
  std::array<int, 20U> statuses{};
  statuses.fill(static_cast<int>(cudaErrorUnknown));
  if (ready) {
    statuses[0] = launch(nullptr, kWeightScale, activations, kTokens, kRows,
                         kColumns, output);
    statuses[1] = launch(weights,
                         std::numeric_limits<float>::quiet_NaN(), activations,
                         kTokens, kRows, kColumns, output);
    statuses[2] = launch(weights, -1.0F, activations, kTokens, kRows,
                         kColumns, output);
    statuses[3] = launch(weights, kWeightScale, nullptr, kTokens, kRows,
                         kColumns, output);
    statuses[4] = launch(weights, kWeightScale, activations, kTokens, kRows,
                         kColumns, nullptr);
    statuses[5] = launch(weights, kWeightScale, activations, 256U, kRows,
                         kColumns, output);
    statuses[6] = launch(weights, kWeightScale, activations, 513U, kRows,
                         kColumns, output);
    statuses[7] = launch(weights, kWeightScale, activations, kTokens,
                         kRows - 1U, kColumns, output);
    statuses[8] = launch(weights, kWeightScale, activations, kTokens, kRows,
                         kColumns - 1U, output);
    statuses[9] = launch(weights + 1U, kWeightScale, activations, kTokens,
                         kRows, kColumns, output);
    statuses[10] = launch(
        weights, kWeightScale,
        reinterpret_cast<const std::uint16_t*>(kActivationAddress + 2U),
        kTokens, kRows, kColumns, output);
    statuses[11] = launch(
        weights, kWeightScale, activations, kTokens, kRows, kColumns,
        reinterpret_cast<std::uint16_t*>(kOutputAddress + 1U));
    statuses[12] = launch(
        weights, kWeightScale, activations, kTokens, kRows, kColumns,
        reinterpret_cast<std::uint16_t*>(kActivationAddress + 128U));
    statuses[13] = launch(
        weights, kWeightScale, activations, kTokens, kRows, kColumns,
        reinterpret_cast<std::uint16_t*>(kWeightAddress + 128U));
    statuses[14] = launch(
        reinterpret_cast<const std::uint8_t*>(kMaximum - 15U), kWeightScale,
        activations, kTokens, kRows, kColumns, output);
    statuses[15] = launch(
        weights, kWeightScale,
        reinterpret_cast<const std::uint16_t*>(kMaximum - 7U), kTokens,
        kRows, kColumns, output);
    statuses[16] = launch(
        weights, kWeightScale, activations, kTokens, kRows, kColumns,
        reinterpret_cast<std::uint16_t*>(kMaximum - 1U));
    statuses[17] = launch(weights, kWeightScale, activations, 0U, kRows,
                          kColumns, output);
    statuses[18] = launch(weights, kWeightScale, activations, kTokens, 0U,
                          kColumns, output);
    statuses[19] = launch(
        weights, std::numeric_limits<float>::infinity(), activations,
        kTokens, kRows, kColumns, output);
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         "invalid end capture") &&
            ready;
  }
  std::size_t nodes = 0U;
  if (ready && graph != nullptr) {
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &nodes),
                         "invalid count graph nodes") &&
            ready;
  }
  if (graph != nullptr) {
    ready = test.cuda_ok(cudaGraphDestroy(graph), "invalid destroy graph") &&
            ready;
  }
  const std::size_t invalid_count = static_cast<std::size_t>(std::count(
      statuses.begin(), statuses.end(),
      static_cast<int>(cudaErrorInvalidValue)));
  const bool gate = ready && invalid_count == statuses.size() && nodes == 0U;
  std::cout << "FP8_SPLIT_M64_INVALID_GRAPH: invalid_statuses="
            << invalid_count << '/' << statuses.size()
            << " graph_nodes=" << nodes
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "all invalid candidate calls enqueue zero nodes");
  return gate;
}

[[nodiscard]] bool poison_output(TestContext& test, Fixture& fixture,
                                 const cudaStream_t stream, const int byte,
                                 const std::string& label) {
  bool ready = test.cuda_ok(
      cudaMemsetAsync(fixture.output_store.get(), byte,
                      fixture.guarded_output_elements() *
                          sizeof(std::uint16_t),
                      stream),
      label + " poison output");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream),
                       label + " poison synchronize");
  return ready;
}

[[nodiscard]] bool copy_output(TestContext& test, const Fixture& fixture,
                               const cudaStream_t stream,
                               std::vector<std::uint16_t>& host,
                               const std::string& label) {
  host.resize(fixture.guarded_output_elements());
  bool ready = test.cuda_ok(
      cudaMemcpyAsync(host.data(), fixture.output_store.get(),
                      host.size() * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost, stream),
      label + " copy output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " copy synchronize");
  return ready;
}

struct GraphIdentity {
  cudaKernelNodeParams parameters{};
  std::size_t node_count = 0U;
  bool valid = false;
};

[[nodiscard]] bool capture_variant(TestContext& test, Fixture& fixture,
                                   const cudaStream_t stream,
                                   const Variant variant,
                                   CapturedGraph& captured,
                                   GraphIdentity& identity) {
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      std::string("begin ") + variant_name(variant) + " capture");
  if (ready) {
    ready = test.cuda_ok(launch_variant(fixture, stream, variant),
                         std::string("launch ") + variant_name(variant) +
                             " capture") &&
            ready;
  }
  if (ready) {
    ready = test.cuda_ok(cudaStreamEndCapture(stream,
                                              captured.graph_address()),
                         std::string("end ") + variant_name(variant) +
                             " capture") &&
            ready;
  }
  std::array<cudaGraphNode_t, 2U> nodes{};
  std::size_t capacity = nodes.size();
  cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
  if (ready) {
    ready = test.cuda_ok(
                cudaGraphGetNodes(captured.graph(), nodes.data(), &capacity),
                std::string("get ") + variant_name(variant) +
                    " graph nodes") &&
            ready;
    identity.node_count = capacity;
  }
  if (ready && capacity == 1U) {
    ready = test.cuda_ok(cudaGraphNodeGetType(nodes[0], &type),
                         std::string("get ") + variant_name(variant) +
                             " node type") &&
            ready;
    if (ready && type == cudaGraphNodeTypeKernel) {
      ready = test.cuda_ok(
                  cudaGraphKernelNodeGetParams(nodes[0],
                                               &identity.parameters),
                  std::string("get ") + variant_name(variant) +
                      " kernel params") &&
              ready;
    }
  }
  if (ready) {
    ready = test.cuda_ok(
                cudaGraphInstantiate(captured.executable_address(),
                                     captured.graph(), nullptr, nullptr, 0U),
                std::string("instantiate ") + variant_name(variant) +
                    " graph") &&
            ready;
  }
  const unsigned int expected_block =
      variant == Variant::kBaseline ? 256U : 512U;
  const unsigned int expected_dynamic =
      variant == Variant::kBaseline ? 0U : 18'432U;
  identity.valid =
      ready && capacity == 1U && type == cudaGraphNodeTypeKernel &&
      identity.parameters.gridDim.x == 320U &&
      identity.parameters.blockDim.x == expected_block &&
      identity.parameters.sharedMemBytes == expected_dynamic &&
      identity.parameters.func != nullptr;
  return identity.valid;
}

[[nodiscard]] bool verify_immutable(TestContext& test,
                                    const Fixture& fixture,
                                    const cudaStream_t stream) {
  std::vector<std::uint8_t> weights(kWeightBytes);
  std::vector<std::uint16_t> activations(kActivationElements);
  bool ready = test.cuda_ok(
      cudaMemcpyAsync(weights.data(), fixture.weights.get(), kWeightBytes,
                      cudaMemcpyDeviceToHost, stream),
      "copy immutable weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.data(), fixture.activations.get(),
                           kActivationElements * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       "copy immutable activations");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "immutable copy synchronize");
  const bool gate = ready && weights == fixture.host_weights &&
                    activations == fixture.host_activations;
  test.expect(gate, "weights and activations remain immutable");
  return gate;
}

[[nodiscard]] bool run_correctness(TestContext& test, Fixture& fixture,
                                   const cudaStream_t stream) {
  std::vector<std::uint16_t> baseline;
  std::vector<std::uint16_t> candidate;
  std::vector<std::uint16_t> replay1;
  std::vector<std::uint16_t> replay2;
  bool ready = poison_output(test, fixture, stream, 0x3c, "baseline");
  ready = ready && test.cuda_ok(
                       launch_variant(fixture, stream, Variant::kBaseline),
                       "baseline direct launch");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "baseline synchronize");
  ready = ready && copy_output(test, fixture, stream, baseline, "baseline");

  ready = ready && poison_output(test, fixture, stream, 0xa5, "candidate");
  ready = ready && test.cuda_ok(
                       launch_variant(fixture, stream, Variant::kCandidate),
                       "candidate direct launch");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "candidate synchronize");
  ready = ready && copy_output(test, fixture, stream, candidate,
                               "candidate");

  CapturedGraph baseline_graph;
  CapturedGraph candidate_graph;
  GraphIdentity baseline_identity;
  GraphIdentity candidate_identity;
  ready = ready && capture_variant(test, fixture, stream, Variant::kBaseline,
                                   baseline_graph, baseline_identity);
  ready = ready && capture_variant(test, fixture, stream,
                                   Variant::kCandidate, candidate_graph,
                                   candidate_identity);
  const bool graph_identity =
      ready && baseline_identity.valid && candidate_identity.valid &&
      baseline_identity.parameters.func != candidate_identity.parameters.func;
  std::cout << "FP8_SPLIT_M64_VALID_GRAPH: baseline_nodes="
            << baseline_identity.node_count
            << " candidate_nodes=" << candidate_identity.node_count
            << " baseline_grid_x=" << baseline_identity.parameters.gridDim.x
            << " candidate_grid_x="
            << candidate_identity.parameters.gridDim.x
            << " baseline_block_x="
            << baseline_identity.parameters.blockDim.x
            << " candidate_block_x="
            << candidate_identity.parameters.blockDim.x
            << " candidate_dynamic_shared_bytes="
            << candidate_identity.parameters.sharedMemBytes
            << " functions_distinct="
            << (baseline_identity.parameters.func !=
                        candidate_identity.parameters.func
                    ? "true"
                    : "false")
            << " gate=" << (graph_identity ? "PASS" : "FAIL") << '\n';
  test.expect(graph_identity,
              "baseline and candidate graph identities are exact");

  ready = ready && poison_output(test, fixture, stream, 0x5a, "replay1");
  ready = ready && test.cuda_ok(
                       cudaGraphLaunch(candidate_graph.executable(), stream),
                       "candidate graph replay1");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "candidate replay1 synchronize");
  ready = ready && copy_output(test, fixture, stream, replay1, "replay1");
  ready = ready && poison_output(test, fixture, stream, 0x69, "replay2");
  ready = ready && test.cuda_ok(
                       cudaGraphLaunch(candidate_graph.executable(), stream),
                       "candidate graph replay2");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "candidate replay2 synchronize");
  ready = ready && copy_output(test, fixture, stream, replay2, "replay2");

  std::size_t candidate_mismatches = 0U;
  std::size_t replay1_mismatches = 0U;
  std::size_t replay2_mismatches = 0U;
  std::size_t nan_outputs = 0U;
  std::size_t nan_class_or_sign_mismatches = 0U;
  std::size_t unexpected_nonfinite = 0U;
  bool guards = ready;
  if (ready) {
    for (std::size_t index = 0U; index < kOutputElements; ++index) {
      const std::size_t guarded = kGuardElements + index;
      const std::uint16_t oracle = baseline[guarded];
      candidate_mismatches += candidate[guarded] != oracle ? 1U : 0U;
      replay1_mismatches += replay1[guarded] != oracle ? 1U : 0U;
      replay2_mismatches += replay2[guarded] != oracle ? 1U : 0U;
      if (is_bf16_nan(oracle)) {
        ++nan_outputs;
        const bool class_and_sign =
            is_bf16_nan(candidate[guarded]) &&
            is_bf16_nan(replay1[guarded]) &&
            is_bf16_nan(replay2[guarded]) &&
            ((candidate[guarded] ^ oracle) & 0x8000U) == 0U &&
            ((replay1[guarded] ^ oracle) & 0x8000U) == 0U &&
            ((replay2[guarded] ^ oracle) & 0x8000U) == 0U;
        nan_class_or_sign_mismatches += class_and_sign ? 0U : 1U;
      } else {
        unexpected_nonfinite +=
            !is_bf16_finite(oracle) ||
                    !is_bf16_finite(candidate[guarded]) ||
                    !is_bf16_finite(replay1[guarded]) ||
                    !is_bf16_finite(replay2[guarded])
                ? 1U
                : 0U;
      }
    }
    for (std::size_t index = 0U; index < kGuardElements; ++index) {
      const std::size_t tail = kGuardElements + kOutputElements + index;
      guards = guards && baseline[index] == 0x3c3cU &&
               baseline[tail] == 0x3c3cU &&
               candidate[index] == 0xa5a5U &&
               candidate[tail] == 0xa5a5U &&
               replay1[index] == 0x5a5aU &&
               replay1[tail] == 0x5a5aU &&
               replay2[index] == 0x6969U &&
               replay2[tail] == 0x6969U;
    }
  }
  bool raw_code_coverage = true;
  for (std::size_t code = 0U; code < 256U; ++code) {
    for (std::size_t position = 0U; position < 4U; ++position) {
      raw_code_coverage =
          raw_code_coverage &&
          fixture.host_weights[code * 4U + position] ==
              static_cast<std::uint8_t>(code);
    }
  }
  const bool immutable = ready && verify_immutable(test, fixture, stream);
  const bool gate =
      ready && graph_identity && raw_code_coverage &&
      candidate_mismatches == 0U && replay1_mismatches == 0U &&
      replay2_mismatches == 0U && nan_class_or_sign_mismatches == 0U &&
      unexpected_nonfinite == 0U && guards && immutable;
  std::cout << "FP8_SPLIT_M64_CORRECTNESS: candidate_mismatches="
            << candidate_mismatches << '/' << kOutputElements
            << " replay1_mismatches=" << replay1_mismatches << '/'
            << kOutputElements
            << " replay2_mismatches=" << replay2_mismatches << '/'
            << kOutputElements
            << " raw_codes_per_byte_position=256 byte_positions=4"
            << " raw_code_coverage="
            << (raw_code_coverage ? "true" : "false")
            << " classified_nan_outputs=" << nan_outputs
            << " nan_class_or_sign_mismatches="
            << nan_class_or_sign_mismatches
            << " unexpected_nonfinite=" << unexpected_nonfinite
            << " guards=" << (guards ? "intact" : "BAD")
            << " immutable=" << (immutable ? "true" : "false")
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "candidate is exact across direct and Graph replay");
  return gate;
}

[[nodiscard]] bool scrub_l2(TestContext& test, Fixture& fixture,
                            const cudaStream_t stream,
                            const std::string& label) {
  scrub_l2_kernel<<<256U, 256U, 0U, stream>>>(
      fixture.scrub.get(), kScrubBytes / sizeof(std::uint32_t));
  bool ready = test.cuda_ok(cudaGetLastError(), label + " scrub launch");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " scrub synchronize");
  return ready;
}

[[nodiscard]] float measure_pass(TestContext& test, Fixture& fixture,
                                 const Execution& execution,
                                 const Variant variant,
                                 const std::string& label,
                                 const int warmups = kWarmups,
                                 const int iterations = kIterations,
                                 const bool profiler_range = false) {
  const cudaStream_t stream = execution.stream();
  bool ready = scrub_l2(test, fixture, stream, label);
  for (int warmup = 0; ready && warmup < warmups; ++warmup) {
    ready = test.cuda_ok(launch_variant(fixture, stream, variant),
                         label + " warmup") &&
            ready;
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " warmup synchronize");
  if (ready && profiler_range) {
    ready = test.cuda_ok(cudaProfilerStart(), label + " profiler start");
  }
  const auto wall_start = std::chrono::steady_clock::now();
  ready = ready && test.cuda_ok(cudaEventRecord(execution.start(), stream),
                                label + " record start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = test.cuda_ok(launch_variant(fixture, stream, variant),
                         label + " measured launch") &&
            ready;
  }
  ready = ready && test.cuda_ok(cudaEventRecord(execution.stop(), stream),
                                label + " record stop");
  ready = ready && test.cuda_ok(cudaEventSynchronize(execution.stop()),
                                label + " stop synchronize");
  const auto wall_stop = std::chrono::steady_clock::now();
  if (profiler_range) {
    ready = test.cuda_ok(cudaProfilerStop(), label + " profiler stop") &&
            ready;
  }
  float total_ms = std::numeric_limits<float>::quiet_NaN();
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_ms, execution.start(),
                                            execution.stop()),
                       label + " elapsed");
  const float average =
      ready && iterations > 0
          ? total_ms / static_cast<float>(iterations)
          : std::numeric_limits<float>::quiet_NaN();
  const double wall_ms =
      std::chrono::duration<double, std::milli>(wall_stop - wall_start)
          .count();
  std::cout << "FP8_SPLIT_M64_PASS: label=" << label
            << " variant=" << variant_name(variant)
            << " warmups=" << warmups << " iterations=" << iterations
            << " average_ms=" << average
            << " host_wall_average_ms="
            << (iterations > 0 ? wall_ms / static_cast<double>(iterations)
                               : std::numeric_limits<double>::quiet_NaN())
            << " l2_scrub_bytes_outside_timing=" << kScrubBytes
            << " profiler_range=" << (profiler_range ? "true" : "false")
            << '\n';
  return average;
}

[[nodiscard]] bool run_screen(TestContext& test, Fixture& fixture,
                              const Execution& execution) {
  double baseline_sum = 0.0;
  double candidate_sum = 0.0;
  bool every_round = true;
  for (int round = 0; round < kRounds; ++round) {
    const std::string prefix =
        "round_" + std::to_string(round + 1) + '_';
    const float b1 = measure_pass(test, fixture, execution,
                                  Variant::kBaseline, prefix + "B1");
    const float c1 = measure_pass(test, fixture, execution,
                                  Variant::kCandidate, prefix + "C1");
    const float c2 = measure_pass(test, fixture, execution,
                                  Variant::kCandidate, prefix + "C2");
    const float b2 = measure_pass(test, fixture, execution,
                                  Variant::kBaseline, prefix + "B2");
    const bool finite = std::isfinite(b1) && std::isfinite(c1) &&
                        std::isfinite(c2) && std::isfinite(b2) && b1 > 0.0F &&
                        c1 > 0.0F && c2 > 0.0F && b2 > 0.0F;
    const double speedup =
        finite ? static_cast<double>(b1 + b2) /
                     static_cast<double>(c1 + c2)
               : std::numeric_limits<double>::quiet_NaN();
    every_round =
        every_round && finite && speedup >= kRequiredRoundSpeedup;
    if (finite) {
      baseline_sum += static_cast<double>(b1 + b2);
      candidate_sum += static_cast<double>(c1 + c2);
    }
    std::cout << "PERF_FP8_SPLIT_M64_ROUND: round=" << round + 1
              << " order=B-C-C-B B1_ms=" << b1 << " C1_ms=" << c1
              << " C2_ms=" << c2 << " B2_ms=" << b2
              << " speedup=" << speedup
              << " required_round_speedup=" << kRequiredRoundSpeedup
              << " gate="
              << (finite && speedup >= kRequiredRoundSpeedup ? "PASS"
                                                            : "FAIL")
              << '\n';
  }
  const bool sums_valid = baseline_sum > 0.0 && candidate_sum > 0.0;
  const double baseline_ms =
      sums_valid ? baseline_sum / (2.0 * kRounds)
                 : std::numeric_limits<double>::quiet_NaN();
  const double candidate_ms =
      sums_valid ? candidate_sum / (2.0 * kRounds)
                 : std::numeric_limits<double>::quiet_NaN();
  const double speedup =
      sums_valid ? baseline_sum / candidate_sum
                 : std::numeric_limits<double>::quiet_NaN();
  const bool gate = every_round && std::isfinite(speedup) &&
                    speedup >= kRequiredAggregateSpeedup;
  std::cout << "PERF_FP8_SPLIT_M64_AGGREGATE: baseline_ms=" << baseline_ms
            << " candidate_ms=" << candidate_ms
            << " speedup=" << speedup
            << " required_aggregate_speedup="
            << kRequiredAggregateSpeedup
            << " required_minimum_round_speedup="
            << kRequiredRoundSpeedup
            << " every_round_gate=" << (every_round ? "true" : "false")
            << " rounds=" << kRounds
            << " warmups_per_pass=" << kWarmups
            << " iterations_per_pass=" << kIterations
            << " order=B-C-C-B traffic_and_hmma_unchanged=true"
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "split-M64 clears frozen C512 QKV performance gate");
  return gate;
}

enum class Mode {
  kValidate,
  kScreen,
  kMeasureBaseline,
  kMeasureCandidate,
  kProfileBaseline,
  kProfileCandidate,
};

struct Options {
  Mode mode = Mode::kValidate;
};

[[nodiscard]] bool parse_options(const int argc, char** argv,
                                 Options& options) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--mode=validate") {
      options.mode = Mode::kValidate;
    } else if (argument == "--mode=screen") {
      options.mode = Mode::kScreen;
    } else if (argument == "--mode=measure-baseline") {
      options.mode = Mode::kMeasureBaseline;
    } else if (argument == "--mode=measure-candidate") {
      options.mode = Mode::kMeasureCandidate;
    } else if (argument == "--mode=profile-baseline") {
      options.mode = Mode::kProfileBaseline;
    } else if (argument == "--mode=profile-candidate") {
      options.mode = Mode::kProfileCandidate;
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] const char* mode_name(const Mode mode) noexcept {
  switch (mode) {
    case Mode::kValidate:
      return "validate";
    case Mode::kScreen:
      return "screen";
    case Mode::kMeasureBaseline:
      return "measure_baseline";
    case Mode::kMeasureCandidate:
      return "measure_candidate";
    case Mode::kProfileBaseline:
      return "profile_baseline";
    case Mode::kProfileCandidate:
      return "profile_candidate";
  }
  return "unknown";
}

}  // namespace

int main(const int argc, char** argv) {
  Options options{};
  if (!parse_options(argc, argv, options)) {
    return 2;
  }
  TestContext test;
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: FP8 split-M64 screen requires CUDA\n";
    (void)cudaGetLastError();
    return 77;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                    "read CUDA device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: FP8 split-M64 requires SM87; got sm_"
              << properties.major << properties.minor << '\n';
    return 77;
  }
  std::cout << std::fixed << std::setprecision(6)
            << "FP8_SPLIT_M64_DEVICE: name=" << properties.name
            << " sm=" << properties.major << properties.minor
            << " sm_count=" << properties.multiProcessorCount
            << " shared_per_sm=" << properties.sharedMemPerMultiprocessor
            << " mode=" << mode_name(options.mode)
            << " tokens=" << kTokens << " rows=" << kRows
            << " columns=" << kColumns
            << " production_dispatch_unchanged=true\n";

  Execution execution;
  if (!execution.create(test)) {
    return 1;
  }
  bool ready = run_resource_gate(test);
  ready = run_invalid_graph_gate(test, execution.stream()) && ready;
  if (!ready) {
    return 1;
  }
  Fixture fixture;
  if (!fixture.initialize(test, execution.stream())) {
    return 1;
  }
  const bool correct = run_correctness(test, fixture, execution.stream());
  if (correct) {
    switch (options.mode) {
      case Mode::kValidate:
        break;
      case Mode::kScreen:
        (void)run_screen(test, fixture, execution);
        break;
      case Mode::kMeasureBaseline:
        (void)measure_pass(test, fixture, execution, Variant::kBaseline,
                           "standalone_baseline");
        break;
      case Mode::kMeasureCandidate:
        (void)measure_pass(test, fixture, execution, Variant::kCandidate,
                           "standalone_candidate");
        break;
      case Mode::kProfileBaseline:
      case Mode::kProfileCandidate: {
        const Variant variant = options.mode == Mode::kProfileBaseline
                                    ? Variant::kBaseline
                                    : Variant::kCandidate;
        const float milliseconds = measure_pass(
            test, fixture, execution, variant, "single_profile", 0, 1, true);
        std::cout << "FP8_SPLIT_M64_PROFILE_MARKER: mode="
                  << mode_name(options.mode)
                  << " milliseconds=" << milliseconds
                  << " profiler_range_kernel_launches=1"
                  << " scrub_launches_in_range=0\n";
        break;
      }
    }
  }
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " FP8 split-M64 assertion(s) failed\n";
    return 1;
  }
  std::cout << "FP8 QKV split-M64 SM87 screen passed\n";
  return 0;
}
