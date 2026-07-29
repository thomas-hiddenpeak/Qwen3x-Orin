#include "../src/kernels/reference/gdn_prefill_whole_span_conv_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using q3x::runtime::gdn_prefill_whole_span_conv_detail::
    launch_causal_conv1d_silu_update_token_parallel_exact_cuda;
using q3x::runtime::gdn_prefill_whole_span_conv_detail::
    launch_causal_conv1d_silu_update_whole_span_exact_cuda;

constexpr std::array<std::size_t, 12U> kTokenCounts{
    1U, 2U, 3U, 7U, 8U, 9U, 31U, 32U, 407U, 481U, 511U, 512U};
constexpr std::size_t kGuardElements = 64U;
constexpr std::uint16_t kLeadingGuard = 0x5aa5U;
constexpr std::uint16_t kTrailingGuard = 0xa55aU;
constexpr std::uint16_t kOutputSentinel = 0x3f5aU;

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
class GuardedManagedBuffer {
 public:
  GuardedManagedBuffer() = default;
  GuardedManagedBuffer(const GuardedManagedBuffer&) = delete;
  GuardedManagedBuffer& operator=(const GuardedManagedBuffer&) = delete;

  ~GuardedManagedBuffer() {
    if (storage_ != nullptr) {
      (void)cudaFree(storage_);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t payload_count) {
    payload_count_ = payload_count;
    const std::size_t allocation_count =
        payload_count + 2U * kGuardElements;
    const cudaError_t status = cudaMallocManaged(
        reinterpret_cast<void**>(&storage_), allocation_count * sizeof(T));
    if (status != cudaSuccess) {
      return status;
    }
    data_ = storage_ + kGuardElements;
    reset_guards();
    return cudaSuccess;
  }

  void reset_guards() noexcept {
    std::fill_n(storage_, kGuardElements,
                static_cast<T>(kLeadingGuard));
    std::fill_n(data_ + payload_count_, kGuardElements,
                static_cast<T>(kTrailingGuard));
  }

  [[nodiscard]] bool guards_intact() const noexcept {
    return std::all_of(storage_, storage_ + kGuardElements,
                       [](const T value) {
                         return value == static_cast<T>(kLeadingGuard);
                       }) &&
           std::all_of(data_ + payload_count_,
                       data_ + payload_count_ + kGuardElements,
                       [](const T value) {
                         return value == static_cast<T>(kTrailingGuard);
                       });
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return payload_count_; }

 private:
  T* storage_ = nullptr;
  T* data_ = nullptr;
  std::size_t payload_count_ = 0U;
};

class DeterministicRandom {
 public:
  explicit DeterministicRandom(const std::uint64_t seed) : state_(seed) {}

  [[nodiscard]] std::uint32_t next() noexcept {
    state_ ^= state_ >> 12U;
    state_ ^= state_ << 25U;
    state_ ^= state_ >> 27U;
    return static_cast<std::uint32_t>(
        (state_ * UINT64_C(2685821657736338717)) >> 32U);
  }

  [[nodiscard]] float normalish(const float scale) noexcept {
    // Six bounded uniforms approximate the center-heavy activation and
    // checkpoint-weight distributions without producing NaN/Inf payloads.
    std::int64_t sum = 0;
    for (std::size_t sample = 0U; sample < 6U; ++sample) {
      sum += static_cast<std::int64_t>(next() & 0xffffU) - 32768;
    }
    return scale * static_cast<float>(sum) / (3.0F * 32768.0F);
  }

 private:
  std::uint64_t state_;
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

void fill_random_bf16(std::uint16_t* const destination,
                      const std::size_t count,
                      DeterministicRandom& random,
                      const float scale) {
  for (std::size_t index = 0U; index < count; ++index) {
    destination[index] = encode_bf16(random.normalish(scale));
  }
}

void expect_bitwise_equal(TestContext& test,
                          const std::uint16_t* const actual,
                          const std::uint16_t* const expected,
                          const std::size_t count,
                          const std::string& label) {
  std::size_t mismatches = 0U;
  std::size_t first = 0U;
  for (std::size_t index = 0U; index < count; ++index) {
    if (actual[index] != expected[index]) {
      if (mismatches == 0U) {
        first = index;
      }
      ++mismatches;
    }
  }
  std::string detail = label + " mismatches=" + std::to_string(mismatches);
  if (mismatches != 0U) {
    detail += ", first=" + std::to_string(first) +
              ", actual_bits=" + std::to_string(actual[first]) +
              ", expected_bits=" + std::to_string(expected[first]);
  }
  test.expect(mismatches == 0U, detail);
}

void expect_unchanged(TestContext& test,
                      const GuardedManagedBuffer<std::uint16_t>& buffer,
                      const std::vector<std::uint16_t>& snapshot,
                      const std::string& label) {
  expect_bitwise_equal(test, buffer.data(), snapshot.data(), buffer.size(),
                       label);
  test.expect(buffer.guards_intact(), label + " guards remain intact");
}

struct ConvBuffers {
  GuardedManagedBuffer<std::uint16_t> raw;
  GuardedManagedBuffer<std::uint16_t> weight;
  GuardedManagedBuffer<std::uint16_t> serial_history;
  GuardedManagedBuffer<std::uint16_t> parallel_history;
  GuardedManagedBuffer<std::uint16_t> serial_output;
  GuardedManagedBuffer<std::uint16_t> parallel_output;
  std::vector<std::uint16_t> raw_snapshot;
  std::vector<std::uint16_t> weight_snapshot;
  std::vector<std::uint16_t> initial_history;
};

[[nodiscard]] bool allocate_and_fill(TestContext& test,
                                     const std::size_t token_count,
                                     ConvBuffers& buffers) {
  constexpr std::size_t kWeightElements =
      q3x::runtime::kGdnQkvChannels *
      q3x::runtime::kGdnConvKernelWidth;
  constexpr std::size_t kHistoryElements =
      q3x::runtime::kGdnQkvChannels *
      q3x::runtime::kGdnConvHistoryWidth;
  const std::size_t qkv_elements =
      token_count * q3x::runtime::kGdnQkvChannels;
  bool ready = test.cuda_ok(buffers.raw.allocate(qkv_elements),
                            "allocate raw M=" +
                                std::to_string(token_count));
  ready = ready && test.cuda_ok(buffers.weight.allocate(kWeightElements),
                                "allocate weight M=" +
                                    std::to_string(token_count));
  ready = ready &&
          test.cuda_ok(buffers.serial_history.allocate(kHistoryElements),
                       "allocate serial history M=" +
                           std::to_string(token_count));
  ready = ready &&
          test.cuda_ok(buffers.parallel_history.allocate(kHistoryElements),
                       "allocate parallel history M=" +
                           std::to_string(token_count));
  ready = ready &&
          test.cuda_ok(buffers.serial_output.allocate(qkv_elements),
                       "allocate serial output M=" +
                           std::to_string(token_count));
  ready = ready &&
          test.cuda_ok(buffers.parallel_output.allocate(qkv_elements),
                       "allocate parallel output M=" +
                           std::to_string(token_count));
  if (!ready) {
    return false;
  }

  DeterministicRandom random(UINT64_C(0x517cc1b727220a95) ^ token_count);
  fill_random_bf16(buffers.raw.data(), buffers.raw.size(), random, 0.85F);
  fill_random_bf16(buffers.weight.data(), buffers.weight.size(), random,
                   0.10F);
  fill_random_bf16(buffers.serial_history.data(),
                   buffers.serial_history.size(), random, 0.65F);
  std::copy_n(buffers.serial_history.data(), buffers.serial_history.size(),
              buffers.parallel_history.data());
  std::fill_n(buffers.serial_output.data(), buffers.serial_output.size(),
              kOutputSentinel);
  std::fill_n(buffers.parallel_output.data(), buffers.parallel_output.size(),
              kOutputSentinel);

  // Preserve occasional BF16 outliers resembling real activation tails.
  for (std::size_t index = 4093U; index < buffers.raw.size();
       index += 65537U) {
    buffers.raw.data()[index] =
        encode_bf16((index & 1U) == 0U ? 3.25F : -3.25F);
  }
  buffers.raw_snapshot.assign(buffers.raw.data(),
                              buffers.raw.data() + buffers.raw.size());
  buffers.weight_snapshot.assign(
      buffers.weight.data(), buffers.weight.data() + buffers.weight.size());
  buffers.initial_history.assign(
      buffers.serial_history.data(),
      buffers.serial_history.data() + buffers.serial_history.size());
  return true;
}

void reset_recurrent_buffers(ConvBuffers& buffers) {
  std::copy(buffers.initial_history.begin(), buffers.initial_history.end(),
            buffers.serial_history.data());
  std::copy(buffers.initial_history.begin(), buffers.initial_history.end(),
            buffers.parallel_history.data());
  std::fill_n(buffers.serial_output.data(), buffers.serial_output.size(),
              kOutputSentinel);
  std::fill_n(buffers.parallel_output.data(), buffers.parallel_output.size(),
              kOutputSentinel);
  buffers.serial_history.reset_guards();
  buffers.parallel_history.reset_guards();
  buffers.serial_output.reset_guards();
  buffers.parallel_output.reset_guards();
}

[[nodiscard]] bool launch_serial_and_parallel(
    TestContext& test, cudaStream_t stream, const std::size_t token_count,
    ConvBuffers& buffers, const std::string& phase) {
  bool ready = test.cuda_ok(
      static_cast<cudaError_t>(
          launch_causal_conv1d_silu_update_whole_span_exact_cuda(
              buffers.raw.data(), token_count, buffers.weight.data(),
              buffers.serial_history.data(), buffers.serial_output.data(),
              static_cast<void*>(stream))),
      phase + " serial launch M=" + std::to_string(token_count));
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           launch_causal_conv1d_silu_update_token_parallel_exact_cuda(
                               buffers.raw.data(), token_count,
                               buffers.weight.data(),
                               buffers.parallel_history.data(),
                               buffers.parallel_output.data(),
                               static_cast<void*>(stream))),
                       phase + " parallel launch M=" +
                           std::to_string(token_count));
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream),
                       phase + " synchronize M=" +
                           std::to_string(token_count));
  return ready;
}

void check_results_and_guards(TestContext& test,
                              const std::size_t token_count,
                              const ConvBuffers& buffers,
                              const std::string& phase) {
  expect_bitwise_equal(
      test, buffers.parallel_output.data(), buffers.serial_output.data(),
      buffers.parallel_output.size(),
      phase + " output bitwise M=" + std::to_string(token_count));
  expect_bitwise_equal(
      test, buffers.parallel_history.data(), buffers.serial_history.data(),
      buffers.parallel_history.size(),
      phase + " history bitwise M=" + std::to_string(token_count));
  expect_unchanged(test, buffers.raw, buffers.raw_snapshot,
                   phase + " immutable raw M=" +
                       std::to_string(token_count));
  expect_unchanged(test, buffers.weight, buffers.weight_snapshot,
                   phase + " immutable weight M=" +
                       std::to_string(token_count));
  test.expect(buffers.serial_history.guards_intact(),
              phase + " serial history guards M=" +
                  std::to_string(token_count));
  test.expect(buffers.parallel_history.guards_intact(),
              phase + " parallel history guards M=" +
                  std::to_string(token_count));
  test.expect(buffers.serial_output.guards_intact(),
              phase + " serial output guards M=" +
                  std::to_string(token_count));
  test.expect(buffers.parallel_output.guards_intact(),
              phase + " parallel output guards M=" +
                  std::to_string(token_count));
}

struct CapturedGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
};

void destroy_graph(TestContext& test, CapturedGraph& captured,
                   const std::string& label) {
  if (captured.executable != nullptr) {
    (void)test.cuda_ok(cudaGraphExecDestroy(captured.executable),
                       "destroy executable " + label);
    captured.executable = nullptr;
  }
  if (captured.graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(captured.graph),
                       "destroy graph " + label);
    captured.graph = nullptr;
  }
}

[[nodiscard]] bool capture_parallel_graph(
    TestContext& test, cudaStream_t stream, const std::size_t token_count,
    ConvBuffers& buffers, CapturedGraph& captured) {
  const std::string label = "parallel graph M=" +
                            std::to_string(token_count);
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
      "begin capture " + label);
  if (ready) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(
            launch_causal_conv1d_silu_update_token_parallel_exact_cuda(
                buffers.raw.data(), token_count, buffers.weight.data(),
                buffers.parallel_history.data(),
                buffers.parallel_output.data(), static_cast<void*>(stream))),
        "capture launch " + label);
  }
  if (ready) {
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &captured.graph),
                         "end capture " + label);
  }
  if (!ready || captured.graph == nullptr) {
    return false;
  }

  std::size_t node_count = 0U;
  ready = test.cuda_ok(
      cudaGraphGetNodes(captured.graph, nullptr, &node_count),
      "count nodes " + label);
  test.expect(node_count == 1U, label + " contains exactly one node");
  std::vector<cudaGraphNode_t> nodes(node_count);
  if (ready && node_count != 0U) {
    ready = test.cuda_ok(
        cudaGraphGetNodes(captured.graph, nodes.data(), &node_count),
        "read nodes " + label);
  }
  std::size_t kernel_nodes = 0U;
  for (const cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    ready = ready && test.cuda_ok(cudaGraphNodeGetType(node, &type),
                                  "read node type " + label);
    if (type == cudaGraphNodeTypeKernel) {
      ++kernel_nodes;
      cudaKernelNodeParams parameters{};
      ready = ready && test.cuda_ok(
                           cudaGraphKernelNodeGetParams(node, &parameters),
                           "read kernel parameters " + label);
      const unsigned int expected_token_tiles = static_cast<unsigned int>(
          (token_count + 7U) / 8U);
      test.expect(parameters.gridDim.x == 40U &&
                      parameters.gridDim.y == expected_token_tiles &&
                      parameters.gridDim.z == 1U,
                  label + " grid is (40, ceil(M/8), 1)");
      test.expect(parameters.blockDim.x == 256U &&
                      parameters.blockDim.y == 1U &&
                      parameters.blockDim.z == 1U,
                  label + " block is (256, 1, 1)");
      test.expect(parameters.sharedMemBytes == 0U,
                  label + " uses no dynamic shared memory");
    }
  }
  test.expect(kernel_nodes == 1U,
              label + " contains exactly one kernel node");
  std::size_t edge_count = 0U;
#if CUDART_VERSION >= 12030
  ready = ready && test.cuda_ok(
                       cudaGraphGetEdges(captured.graph, nullptr, nullptr,
                                         nullptr, &edge_count),
                       "count edges " + label);
#else
  ready = ready && test.cuda_ok(
                       cudaGraphGetEdges(captured.graph, nullptr, nullptr,
                                         &edge_count),
                       "count edges " + label);
#endif
  test.expect(edge_count == 0U, label + " contains no dependency edge");
  ready = ready && test.cuda_ok(
                       cudaGraphInstantiate(&captured.executable,
                                            captured.graph, nullptr, nullptr,
                                            0U),
                       "instantiate " + label);
  return ready;
}

void test_shape(TestContext& test, cudaStream_t stream,
                const std::size_t token_count) {
  ConvBuffers buffers;
  if (!allocate_and_fill(test, token_count, buffers)) {
    return;
  }

  if (!launch_serial_and_parallel(test, stream, token_count, buffers,
                                  "direct")) {
    return;
  }
  check_results_and_guards(test, token_count, buffers, "direct");

  reset_recurrent_buffers(buffers);
  CapturedGraph captured;
  if (!capture_parallel_graph(test, stream, token_count, buffers, captured)) {
    destroy_graph(test, captured,
                  "failed M=" + std::to_string(token_count));
    return;
  }
  for (std::size_t replay = 1U; replay <= 2U; ++replay) {
    bool ready = test.cuda_ok(
        cudaGraphLaunch(captured.executable, stream),
        "graph replay " + std::to_string(replay) + " M=" +
            std::to_string(token_count));
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(
                             launch_causal_conv1d_silu_update_whole_span_exact_cuda(
                                 buffers.raw.data(), token_count,
                                 buffers.weight.data(),
                                 buffers.serial_history.data(),
                                 buffers.serial_output.data(),
                                 static_cast<void*>(stream))),
                         "serial graph oracle replay " +
                             std::to_string(replay) + " M=" +
                             std::to_string(token_count));
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         "graph replay synchronize " +
                             std::to_string(replay) + " M=" +
                             std::to_string(token_count));
    if (!ready) {
      break;
    }
    check_results_and_guards(
        test, token_count, buffers,
        "graph replay " + std::to_string(replay));
  }
  destroy_graph(test, captured,
                "parallel M=" + std::to_string(token_count));
  std::cout << "token-parallel exact conv passed M=" << token_count << '\n';
}

void test_invalid_arguments_and_empty_capture(TestContext& test,
                                              cudaStream_t stream) {
  constexpr std::size_t kRawElements =
      2U * q3x::runtime::kGdnQkvChannels;
  constexpr std::size_t kWeightElements =
      q3x::runtime::kGdnQkvChannels *
      q3x::runtime::kGdnConvKernelWidth;
  constexpr std::size_t kHistoryElements =
      q3x::runtime::kGdnQkvChannels *
      q3x::runtime::kGdnConvHistoryWidth;
  GuardedManagedBuffer<std::uint16_t> raw;
  GuardedManagedBuffer<std::uint16_t> weight;
  GuardedManagedBuffer<std::uint16_t> history;
  GuardedManagedBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(raw.allocate(kWeightElements),
                            "invalid test allocate raw");
  ready = ready && test.cuda_ok(weight.allocate(kWeightElements),
                                "invalid test allocate weight");
  ready = ready && test.cuda_ok(history.allocate(kHistoryElements),
                                "invalid test allocate history");
  ready = ready && test.cuda_ok(output.allocate(kRawElements),
                                "invalid test allocate output");
  if (!ready) {
    return;
  }

  const auto launch = [&](const std::uint16_t* const raw_pointer,
                          const std::size_t token_count,
                          const std::uint16_t* const weight_pointer,
                          std::uint16_t* const history_pointer,
                          std::uint16_t* const output_pointer) {
    return static_cast<cudaError_t>(
        launch_causal_conv1d_silu_update_token_parallel_exact_cuda(
            raw_pointer, token_count, weight_pointer, history_pointer,
            output_pointer, static_cast<void*>(stream)));
  };
  const auto expect_invalid = [&](const cudaError_t status,
                                  const std::string& label) {
    test.expect(status == cudaErrorInvalidValue,
                label + " returns cudaErrorInvalidValue");
  };
  expect_invalid(launch(nullptr, 1U, weight.data(), history.data(),
                        output.data()),
                 "null raw");
  expect_invalid(launch(raw.data(), 1U, nullptr, history.data(),
                        output.data()),
                 "null weight");
  expect_invalid(launch(raw.data(), 1U, weight.data(), nullptr,
                        output.data()),
                 "null history");
  expect_invalid(launch(raw.data(), 1U, weight.data(), history.data(),
                        nullptr),
                 "null output");
  expect_invalid(launch(raw.data(), 0U, weight.data(), history.data(),
                        output.data()),
                 "zero tokens");
  expect_invalid(launch(raw.data(), 513U, weight.data(), history.data(),
                        output.data()),
                 "oversized tokens");
  expect_invalid(launch(raw.data(), 1U, weight.data(), history.data(),
                        raw.data()),
                 "raw/output alias");
  expect_invalid(launch(raw.data(), 1U, raw.data(), history.data(),
                        output.data()),
                 "raw/weight alias");
  expect_invalid(launch(raw.data(), 1U, weight.data(), raw.data(),
                        output.data()),
                 "raw/history alias");
  expect_invalid(launch(raw.data(), 1U, weight.data(), weight.data(),
                        output.data()),
                 "weight/history alias");
  expect_invalid(launch(raw.data(), 1U, weight.data(), history.data(),
                        weight.data()),
                 "weight/output alias");
  expect_invalid(launch(raw.data(), 1U, weight.data(), history.data(),
                        history.data()),
                 "history/output alias");

  cudaGraph_t graph = nullptr;
  ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
      "begin invalid-alias capture");
  if (ready) {
    expect_invalid(launch(raw.data(), 1U, weight.data(), history.data(),
                          raw.data()),
                   "captured raw/output alias");
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         "end invalid-alias capture");
  }
  if (ready && graph != nullptr) {
    std::size_t node_count = std::numeric_limits<std::size_t>::max();
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                         "count invalid-alias graph nodes");
    test.expect(node_count == 0U,
                "invalid raw/output alias captures zero graph nodes");
  }
  if (graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(graph),
                       "destroy invalid-alias graph");
  }
  test.expect(raw.guards_intact(), "invalid aliases preserve raw guards");
  test.expect(weight.guards_intact(),
              "invalid aliases preserve weight guards");
  test.expect(history.guards_intact(),
              "invalid aliases preserve history guards");
  test.expect(output.guards_intact(),
              "invalid aliases preserve output guards");
}

}  // namespace

int main() {
  TestContext test;
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: token-parallel conv test requires CUDA\n";
    (void)cudaGetLastError();
    return 77;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                    "read CUDA device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: token-parallel conv test requires SM87\n";
    return 77;
  }

  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreateWithFlags(&stream,
                                               cudaStreamNonBlocking),
                    "create test stream")) {
    return 1;
  }
  test_invalid_arguments_and_empty_capture(test, stream);
  for (const std::size_t token_count : kTokenCounts) {
    test_shape(test, stream, token_count);
  }
  (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy test stream");
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " token-parallel exact-conv assertion(s) failed\n";
    return 1;
  }
  std::cout << "token-parallel exact conv correctness/Graph tests passed"
            << " (model ABI has no conv bias operand)\n";
  return 0;
}
