#include "q3x/runtime/gdn_decode.h"
#include "q3x/runtime/decode_ops.h"

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
#include <utility>
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

[[nodiscard]] int launch_gated_delta_net_update_tile_warp_four_row_test_cuda(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_gated_delta_net_update_tile_warp_four_row_lane_striped_test_cuda(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_gated_delta_net_update_tile_warp_eight_row_lane_striped_test_cuda(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_gated_delta_net_update_tile_warp_eight_row_register_state_m16_test_cuda(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
query_gated_delta_net_update_warp_eight_row_register_state_m16_resources_test_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

[[nodiscard]] int
query_gated_delta_net_update_warp_eight_row_lane_striped_resources_test_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

[[nodiscard]] int
launch_gated_delta_net_update_plain_rms_norm_silu_gate_shared_tree_test_cuda(
    const std::uint16_t* conv_qkv, const std::uint16_t* a,
    const std::uint16_t* b, const std::uint16_t* A_log,
    const std::uint16_t* dt_bias, const std::uint16_t* state_input,
    std::uint16_t* state_output, float l2_epsilon,
    const std::uint16_t* norm_weight, const std::uint16_t* silu_gate,
    float norm_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
query_gated_delta_net_update_warp_eight_row_plain_rms_norm_silu_gate_resources_test_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

[[nodiscard]] int
query_gated_delta_net_update_warp_eight_row_plain_rms_norm_silu_gate_shared_tree_resources_test_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

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

struct CapturedTopology {
  std::size_t total_nodes = std::numeric_limits<std::size_t>::max();
  std::size_t kernel_nodes = std::numeric_limits<std::size_t>::max();
  std::size_t edges = std::numeric_limits<std::size_t>::max();
  struct KernelLaunch {
    void* function = nullptr;
    dim3 grid{};
    dim3 block{};
    unsigned int dynamic_shared_bytes = 0U;
  };
  std::vector<KernelLaunch> kernel_launches;
};

template <typename Launch>
[[nodiscard]] CapturedTopology capture_topology(
    TestContext& test, Launch&& launch, const cudaError_t expected_status,
    const std::string& label) {
  CapturedTopology topology{};
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
    test.expect(static_cast<cudaError_t>(launch(stream)) == expected_status,
                label + " returns expected capture status");
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         "end capture " + label);
  }
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
      topology.total_nodes = node_count;
      topology.kernel_nodes = 0U;
      for (const cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        ready = test.cuda_ok(cudaGraphNodeGetType(node, &type),
                             "read graph node type " + label);
        if (ready && type == cudaGraphNodeTypeKernel) {
          ++topology.kernel_nodes;
          cudaKernelNodeParams parameters{};
          ready = test.cuda_ok(
              cudaGraphKernelNodeGetParams(node, &parameters),
              "read graph kernel parameters " + label);
          if (ready) {
            topology.kernel_launches.push_back(
                {parameters.func, parameters.gridDim, parameters.blockDim,
                 parameters.sharedMemBytes});
          }
        }
      }
    }
    std::size_t edge_count = 0U;
    if (ready) {
#if CUDART_VERSION >= 12030
      ready = test.cuda_ok(
          cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edge_count),
          "count graph edges " + label);
#else
      ready = test.cuda_ok(
          cudaGraphGetEdges(graph, nullptr, nullptr, &edge_count),
          "count graph edges " + label);
#endif
    }
    if (ready) {
      topology.edges = edge_count;
    }
  }
  if (graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(graph), "destroy graph " + label);
  }
  if (stream != nullptr) {
    (void)test.cuda_ok(cudaStreamDestroy(stream),
                       "destroy capture stream " + label);
  }
  return topology;
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
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              1.0e-6F, nullptr, nullptr, q3x::runtime::kGdnValueHeadCount,
              q3x::runtime::kGdnHeadDimension, 1.0e-6F, nullptr)) ==
          cudaErrorInvalidValue,
      "fused GDN norm rejects null buffers");
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              std::numeric_limits<float>::quiet_NaN(), nullptr, nullptr,
              q3x::runtime::kGdnValueHeadCount,
              q3x::runtime::kGdnHeadDimension, 1.0e-6F, nullptr)) ==
          cudaErrorInvalidValue,
      "fused GDN norm rejects NaN GDN epsilon");
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              1.0e-6F, nullptr, nullptr,
              q3x::runtime::kGdnValueHeadCount,
              q3x::runtime::kGdnHeadDimension,
              std::numeric_limits<float>::infinity(), nullptr)) ==
          cudaErrorInvalidValue,
      "fused GDN norm rejects infinite norm epsilon");
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          query_gated_delta_net_update_warp_eight_row_plain_rms_norm_silu_gate_resources_test_cuda(
              nullptr, nullptr, nullptr, nullptr, nullptr)) ==
          cudaErrorInvalidValue,
      "fused GDN norm resource query rejects null outputs");
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

  constexpr std::size_t kSplitTokens = 8U;
  static_assert(kMaximumTokens == 2U * kSplitTokens);
  std::copy(initial_qkv.begin(), initial_qkv.end(), sequential_qkv.data());
  std::copy(initial_qkv.begin(), initial_qkv.end(), tile_qkv.data());
  std::copy(initial_history.begin(), initial_history.end(),
            sequential_history.data());
  std::copy(initial_history.begin(), initial_history.end(),
            tile_history.data());
  ready = launch_after_stale(test, stream, "causal conv C16 oracle", [&]() {
    return q3x::runtime::launch_causal_conv1d_silu_update_tile_reference_cuda(
        tile_qkv.data(), kMaximumTokens, weight.data(), tile_history.data(),
        tile_qkv.data(), {}, static_cast<void*>(stream));
  });
  if (!ready) {
    return;
  }
  ready = launch_after_stale(
      test, stream, "causal conv first C8 split oracle", [&]() {
        return q3x::runtime::
            launch_causal_conv1d_silu_update_tile_reference_cuda(
                sequential_qkv.data(), kSplitTokens, weight.data(),
                sequential_history.data(), sequential_qkv.data(), {},
                static_cast<void*>(stream));
      });
  if (!ready) {
    return;
  }
  ready = launch_after_stale(
      test, stream, "causal conv second C8 split oracle", [&]() {
        constexpr std::size_t kSecondTokenOffset =
            kSplitTokens * q3x::runtime::kGdnQkvChannels;
        return q3x::runtime::
            launch_causal_conv1d_silu_update_tile_reference_cuda(
                sequential_qkv.data() + kSecondTokenOffset, kSplitTokens,
                weight.data(), sequential_history.data(),
                sequential_qkv.data() + kSecondTokenOffset, {},
                static_cast<void*>(stream));
      });
  if (!ready) {
    return;
  }
  expect_bf16_buffer_bitwise_equal(
      test, tile_qkv.data(), sequential_qkv.data(), kTileElements,
      "CUDA causal conv C16 output equals sequential C8+C8");
  expect_bf16_buffer_bitwise_equal(
      test, tile_history.data(), sequential_history.data(), kHistoryElements,
      "CUDA causal conv C16 history equals sequential C8+C8");

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

void test_gdn_plain_norm_silu_gate_production_identity(
    TestContext& test, cudaStream_t stream) {
  const int failures_before = test.failures();
  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> norm_weight;
  ManagedBuffer<std::uint16_t> near_norm_weight;
  ManagedBuffer<std::uint16_t> silu_gate;
  ManagedBuffer<std::uint16_t> reference_state;
  ManagedBuffer<std::uint16_t> candidate_state;
  ManagedBuffer<std::uint16_t> reference_disjoint_state;
  ManagedBuffer<std::uint16_t> candidate_disjoint_state;
  ManagedBuffer<std::uint16_t> reference_output;
  ManagedBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(
      conv_qkv.allocate(q3x::runtime::kGdnQkvChannels),
      "fused GDN norm allocate conv QKV");
  ready = ready && test.cuda_ok(
                       a.allocate(q3x::runtime::kGdnValueHeadCount),
                       "fused GDN norm allocate a");
  ready = ready && test.cuda_ok(
                       b.allocate(q3x::runtime::kGdnValueHeadCount),
                       "fused GDN norm allocate b");
  ready = ready && test.cuda_ok(
                       A_log.allocate(q3x::runtime::kGdnValueHeadCount),
                       "fused GDN norm allocate A_log");
  ready = ready && test.cuda_ok(
                       dt_bias.allocate(q3x::runtime::kGdnValueHeadCount),
                       "fused GDN norm allocate dt_bias");
  ready = ready && test.cuda_ok(
                       norm_weight.allocate(q3x::runtime::kGdnHeadDimension),
                       "fused GDN norm allocate norm weight");
  ready = ready && test.cuda_ok(
                       near_norm_weight.allocate(256U),
                       "fused GDN norm allocate near-miss norm weight");
  ready = ready && test.cuda_ok(
                       silu_gate.allocate(q3x::runtime::kGdnVElements),
                       "fused GDN norm allocate SiLU gate");
  ready = ready && test.cuda_ok(
                       reference_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "fused GDN norm allocate reference state");
  ready = ready && test.cuda_ok(
                       candidate_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "fused GDN norm allocate candidate state");
  ready = ready && test.cuda_ok(
                       reference_disjoint_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "fused GDN norm allocate reference disjoint state");
  ready = ready && test.cuda_ok(
                       candidate_disjoint_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "fused GDN norm allocate candidate disjoint state");
  ready = ready && test.cuda_ok(
                       reference_output.allocate(
                           q3x::runtime::kGdnVElements),
                       "fused GDN norm allocate reference output");
  ready = ready && test.cuda_ok(
                       candidate_output.allocate(
                           q3x::runtime::kGdnVElements),
                       "fused GDN norm allocate candidate output");
  if (!ready) {
    return;
  }

  for (std::size_t dimension = 0U;
       dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
    const int centered = static_cast<int>((dimension * 7U) % 19U) - 9;
    norm_weight[dimension] =
        encode_bf16(1.0F + static_cast<float>(centered) / 64.0F);
  }
  for (std::size_t dimension = 0U; dimension < near_norm_weight.size();
       ++dimension) {
    const int centered = static_cast<int>((dimension * 13U) % 23U) - 11;
    near_norm_weight[dimension] =
        encode_bf16(0.75F + static_cast<float>(centered) / 64.0F);
  }
  for (std::size_t index = 0U; index < q3x::runtime::kGdnVElements;
       ++index) {
    const int centered = static_cast<int>((index * 11U) % 41U) - 20;
    silu_gate[index] =
        encode_bf16(static_cast<float>(centered) / 8.0F);
  }
  silu_gate[0] = encode_bf16(20.0F);
  silu_gate[1] = encode_bf16(-20.0F);

  std::vector<std::uint16_t> initial_state(
      q3x::runtime::kGdnStateElements);
  for (std::size_t index = 0U; index < initial_state.size(); ++index) {
    const int centered = static_cast<int>((index * 5U) % 23U) - 11;
    initial_state[index] =
        encode_bf16(static_cast<float>(centered) / 512.0F);
  }
  std::copy(initial_state.begin(), initial_state.end(),
            reference_state.data());
  std::copy(initial_state.begin(), initial_state.end(),
            candidate_state.data());

  constexpr float kL2Epsilon = 1.0e-6F;
  constexpr float kNormEpsilon = 1.0e-6F;
  for (std::size_t step = 0U; step < 3U; ++step) {
    fill_gdn_inputs(conv_qkv, a, b, A_log, dt_bias, step);
    std::fill_n(reference_output.data(), reference_output.size(),
                static_cast<std::uint16_t>(0x5a5aU));
    std::fill_n(candidate_output.data(), candidate_output.size(),
                static_cast<std::uint16_t>(0xa5a5U));

    if (!test.cuda_ok(
            static_cast<cudaError_t>(
                q3x::runtime::launch_gated_delta_net_update_warp_parallel_cuda(
                    conv_qkv.data(), a.data(), b.data(), A_log.data(),
                    dt_bias.data(), reference_state.data(),
                    reference_state.data(), kL2Epsilon,
                    reference_output.data(), {}, static_cast<void*>(stream))),
            "fused GDN norm ordered GDN launch step " +
                std::to_string(step))) {
      return;
    }
    if (!test.cuda_ok(
            static_cast<cudaError_t>(q3x::runtime::
                launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                    reference_output.data(), norm_weight.data(),
                    silu_gate.data(), q3x::runtime::kGdnValueHeadCount,
                    q3x::runtime::kGdnHeadDimension, kNormEpsilon,
                    reference_output.data(), static_cast<void*>(stream))),
            "fused GDN norm ordered norm/gate launch step " +
                std::to_string(step))) {
      return;
    }
    ready = launch_after_stale(
        test, stream,
        "fused GDN norm candidate step " + std::to_string(step), [&]() {
          return q3x::runtime::
              launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
                  conv_qkv.data(), a.data(), b.data(), A_log.data(),
                  dt_bias.data(), candidate_state.data(),
                  candidate_state.data(), kL2Epsilon, norm_weight.data(),
                  silu_gate.data(), q3x::runtime::kGdnValueHeadCount,
                  q3x::runtime::kGdnHeadDimension, kNormEpsilon,
                  candidate_output.data(), {}, static_cast<void*>(stream));
        });
    if (!ready) {
      return;
    }
    expect_bf16_buffer_bitwise_equal(
        test, candidate_output.data(), reference_output.data(),
        q3x::runtime::kGdnVElements,
        "fused GDN norm BF16 write/read output step " +
            std::to_string(step));
    expect_bf16_buffer_bitwise_equal(
        test, candidate_state.data(), reference_state.data(),
        q3x::runtime::kGdnStateElements,
            "fused GDN norm persistent state step " + std::to_string(step));
  }

  // Isolate the reduction change inside the otherwise-identical fused kernel.
  // The preserved shared-tree baseline and production warp-tail path advance
  // complete independent states through the same three recurrence steps.
  for (std::size_t step = 17U; step < 20U; ++step) {
    fill_gdn_inputs(conv_qkv, a, b, A_log, dt_bias, step);
    std::fill_n(reference_output.data(), reference_output.size(),
                static_cast<std::uint16_t>(0x5a5aU));
    std::fill_n(candidate_output.data(), candidate_output.size(),
                static_cast<std::uint16_t>(0xa5a5U));
    if (!test.cuda_ok(
            static_cast<cudaError_t>(q3x::runtime::
                launch_gated_delta_net_update_plain_rms_norm_silu_gate_shared_tree_test_cuda(
                    conv_qkv.data(), a.data(), b.data(), A_log.data(),
                    dt_bias.data(), reference_state.data(),
                    reference_state.data(), kL2Epsilon, norm_weight.data(),
                    silu_gate.data(), kNormEpsilon, reference_output.data(),
                    static_cast<void*>(stream))),
            "fused GDN norm shared-tree launch step " +
                std::to_string(step))) {
      return;
    }
    ready = launch_after_stale(
        test, stream,
        "fused GDN norm warp-tail launch step " + std::to_string(step), [&]() {
          return q3x::runtime::
              launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
                  conv_qkv.data(), a.data(), b.data(), A_log.data(),
                  dt_bias.data(), candidate_state.data(),
                  candidate_state.data(), kL2Epsilon, norm_weight.data(),
                  silu_gate.data(), q3x::runtime::kGdnValueHeadCount,
                  q3x::runtime::kGdnHeadDimension, kNormEpsilon,
                  candidate_output.data(), {}, static_cast<void*>(stream));
        });
    if (!ready) {
      return;
    }
    expect_bf16_buffer_bitwise_equal(
        test, candidate_output.data(), reference_output.data(),
        q3x::runtime::kGdnVElements,
        "fused GDN norm shared-tree/warp-tail output step " +
            std::to_string(step));
    expect_bf16_buffer_bitwise_equal(
        test, candidate_state.data(), reference_state.data(),
        q3x::runtime::kGdnStateElements,
        "fused GDN norm shared-tree/warp-tail state step " +
            std::to_string(step));
  }

  fill_gdn_inputs(conv_qkv, a, b, A_log, dt_bias, 7U);
  const std::vector<std::uint16_t> disjoint_input(
      reference_state.data(), reference_state.data() + reference_state.size());
  std::fill_n(reference_disjoint_state.data(),
              reference_disjoint_state.size(),
              static_cast<std::uint16_t>(0x5a5aU));
  std::fill_n(candidate_disjoint_state.data(),
              candidate_disjoint_state.size(),
              static_cast<std::uint16_t>(0xa5a5U));
  if (!test.cuda_ok(
          static_cast<cudaError_t>(
              q3x::runtime::launch_gated_delta_net_update_warp_parallel_cuda(
                  conv_qkv.data(), a.data(), b.data(), A_log.data(),
                  dt_bias.data(), reference_state.data(),
                  reference_disjoint_state.data(), kL2Epsilon,
                  reference_output.data(), {}, static_cast<void*>(stream))),
          "fused GDN norm ordered disjoint GDN launch")) {
    return;
  }
  if (!test.cuda_ok(
          static_cast<cudaError_t>(q3x::runtime::
              launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                  reference_output.data(), norm_weight.data(),
                  silu_gate.data(), q3x::runtime::kGdnValueHeadCount,
                  q3x::runtime::kGdnHeadDimension, kNormEpsilon,
                  reference_output.data(), static_cast<void*>(stream))),
          "fused GDN norm ordered disjoint norm/gate launch")) {
    return;
  }
  ready = launch_after_stale(
      test, stream, "fused GDN norm candidate disjoint", [&]() {
        return q3x::runtime::
            launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
                conv_qkv.data(), a.data(), b.data(), A_log.data(),
                dt_bias.data(), candidate_state.data(),
                candidate_disjoint_state.data(), kL2Epsilon,
                norm_weight.data(), silu_gate.data(),
                q3x::runtime::kGdnValueHeadCount,
                q3x::runtime::kGdnHeadDimension, kNormEpsilon,
                candidate_output.data(), {}, static_cast<void*>(stream));
      });
  if (!ready) {
    return;
  }
  expect_bf16_buffer_bitwise_equal(
      test, candidate_output.data(), reference_output.data(),
      q3x::runtime::kGdnVElements,
      "fused GDN norm disjoint BF16 write/read output");
  expect_bf16_buffer_bitwise_equal(
      test, candidate_disjoint_state.data(), reference_disjoint_state.data(),
      q3x::runtime::kGdnStateElements,
      "fused GDN norm disjoint persistent state");
  expect_bf16_buffer_bitwise_equal(
      test, candidate_state.data(), disjoint_input.data(),
      q3x::runtime::kGdnStateElements,
      "fused GDN norm preserves disjoint state input");

  fill_gdn_inputs(conv_qkv, a, b, A_log, dt_bias, 9U);
  std::copy(initial_state.begin(), initial_state.end(),
            reference_state.data());
  std::copy(initial_state.begin(), initial_state.end(),
            candidate_state.data());
  if (!test.cuda_ok(
          static_cast<cudaError_t>(
              q3x::runtime::launch_gated_delta_net_update_warp_parallel_cuda(
                  conv_qkv.data(), a.data(), b.data(), A_log.data(),
                  dt_bias.data(), reference_state.data(),
                  reference_state.data(), kL2Epsilon,
                  reference_output.data(), {}, static_cast<void*>(stream))),
          "production GDN norm near-miss ordered GDN launch")) {
    return;
  }
  if (!test.cuda_ok(
          static_cast<cudaError_t>(q3x::runtime::
              launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                  reference_output.data(), near_norm_weight.data(),
                  silu_gate.data(), 24U, 256U, kNormEpsilon,
                  reference_output.data(), static_cast<void*>(stream))),
          "production GDN norm near-miss ordered norm/gate launch")) {
    return;
  }
  ready = launch_after_stale(
      test, stream, "production GDN norm near-miss fallback", [&]() {
        return q3x::runtime::
            launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
                conv_qkv.data(), a.data(), b.data(), A_log.data(),
                dt_bias.data(), candidate_state.data(),
                candidate_state.data(), kL2Epsilon, near_norm_weight.data(),
                silu_gate.data(), 24U, 256U, kNormEpsilon,
                candidate_output.data(), {}, static_cast<void*>(stream));
      });
  if (!ready) {
    return;
  }
  expect_bf16_buffer_bitwise_equal(
      test, candidate_output.data(), reference_output.data(),
      q3x::runtime::kGdnVElements,
      "production GDN norm near-miss fallback output");
  expect_bf16_buffer_bitwise_equal(
      test, candidate_state.data(), reference_state.data(),
      q3x::runtime::kGdnStateElements,
      "production GDN norm near-miss fallback state");

  const std::vector<std::uint16_t> prevalidation_state(
      candidate_state.data(), candidate_state.data() + candidate_state.size());
  std::fill_n(candidate_output.data(), candidate_output.size(),
              static_cast<std::uint16_t>(0x3c3cU));
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
              conv_qkv.data(), a.data(), b.data(), A_log.data(),
              dt_bias.data(), candidate_state.data(), candidate_state.data(),
              kL2Epsilon, norm_weight.data(), candidate_output.data(),
              q3x::runtime::kGdnValueHeadCount,
              q3x::runtime::kGdnHeadDimension, kNormEpsilon,
              candidate_output.data(), {}, static_cast<void*>(stream))) ==
          cudaErrorInvalidValue,
      "fused GDN norm rejects gate/output overlap");
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
              conv_qkv.data(), a.data(), b.data(), A_log.data(),
              dt_bias.data(), candidate_state.data(),
              candidate_state.data() + 1U, kL2Epsilon, norm_weight.data(),
              silu_gate.data(), q3x::runtime::kGdnValueHeadCount,
              q3x::runtime::kGdnHeadDimension, kNormEpsilon,
              candidate_output.data(), {}, static_cast<void*>(stream))) ==
          cudaErrorInvalidValue,
      "fused GDN norm rejects partial state overlap");
  auto* const misaligned_output = reinterpret_cast<std::uint16_t*>(
      reinterpret_cast<std::uint8_t*>(candidate_output.data()) + 1U);
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
              conv_qkv.data(), a.data(), b.data(), A_log.data(),
              dt_bias.data(), candidate_state.data(), candidate_state.data(),
              kL2Epsilon, norm_weight.data(), silu_gate.data(),
              q3x::runtime::kGdnValueHeadCount,
              q3x::runtime::kGdnHeadDimension, kNormEpsilon,
              misaligned_output, {}, static_cast<void*>(stream))) ==
          cudaErrorInvalidValue,
      "fused GDN norm rejects misaligned output");
  ready = test.cuda_ok(cudaStreamSynchronize(stream),
                       "fused GDN norm invalid launches synchronize");
  if (!ready) {
    return;
  }
  test.expect(std::all_of(candidate_output.data(),
                          candidate_output.data() + candidate_output.size(),
                          [](const std::uint16_t value) {
                            return value == static_cast<std::uint16_t>(
                                                0x3c3cU);
                          }),
              "fused GDN norm validates before first enqueue");
  expect_bf16_buffer_bitwise_equal(
      test, candidate_state.data(), prevalidation_state.data(),
      q3x::runtime::kGdnStateElements,
      "fused GDN norm invalid launches preserve state");

  const CapturedTopology exact_topology = capture_topology(
      test,
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
                conv_qkv.data(), a.data(), b.data(), A_log.data(),
                dt_bias.data(), candidate_state.data(),
                candidate_state.data(), kL2Epsilon, norm_weight.data(),
                silu_gate.data(), q3x::runtime::kGdnValueHeadCount,
                q3x::runtime::kGdnHeadDimension, kNormEpsilon,
                candidate_output.data(), {},
                static_cast<void*>(capture_stream));
      },
      cudaSuccess, "production fused GDN norm exact topology");
  const CapturedTopology detail_exact_topology = capture_topology(
      test,
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::gdn_decode_detail::
            launch_gated_delta_net_update_plain_rms_norm_silu_gate_exact_cuda(
                conv_qkv.data(), a.data(), b.data(), A_log.data(),
                dt_bias.data(), candidate_state.data(),
                candidate_state.data(), kL2Epsilon, norm_weight.data(),
                silu_gate.data(), kNormEpsilon, candidate_output.data(),
                static_cast<void*>(capture_stream));
      },
      cudaSuccess, "detail fused GDN norm exact topology");
  const CapturedTopology shared_tree_topology = capture_topology(
      test,
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_plain_rms_norm_silu_gate_shared_tree_test_cuda(
                conv_qkv.data(), a.data(), b.data(), A_log.data(),
                dt_bias.data(), reference_state.data(), reference_state.data(),
                kL2Epsilon, norm_weight.data(), silu_gate.data(),
                kNormEpsilon, reference_output.data(),
                static_cast<void*>(capture_stream));
      },
      cudaSuccess, "shared-tree fused GDN norm exact topology");
  test.expect(exact_topology.total_nodes == 1U &&
                  exact_topology.kernel_nodes == 1U &&
                  exact_topology.edges == 0U &&
                  exact_topology.kernel_launches.size() == 1U,
              "production fused GDN norm exact path captures one kernel");
  test.expect(detail_exact_topology.total_nodes == 1U &&
                  detail_exact_topology.kernel_nodes == 1U &&
                  detail_exact_topology.edges == 0U &&
                  detail_exact_topology.kernel_launches.size() == 1U,
              "detail fused GDN norm exact path captures one kernel");
  test.expect(shared_tree_topology.total_nodes == 1U &&
                  shared_tree_topology.kernel_nodes == 1U &&
                  shared_tree_topology.edges == 0U &&
                  shared_tree_topology.kernel_launches.size() == 1U,
              "shared-tree fused GDN norm exact path captures one kernel");
  if (exact_topology.kernel_launches.size() == 1U &&
      detail_exact_topology.kernel_launches.size() == 1U &&
      shared_tree_topology.kernel_launches.size() == 1U) {
    const auto& production = exact_topology.kernel_launches.front();
    const auto& detail = detail_exact_topology.kernel_launches.front();
    const auto& shared_tree = shared_tree_topology.kernel_launches.front();
    test.expect(production.function == detail.function,
                "public and detail fused GDN norm capture the same kernel");
    test.expect(production.grid.x == q3x::runtime::kGdnValueHeadCount &&
                    production.grid.y == 1U && production.grid.z == 1U &&
                    production.block.x == 256U &&
                    production.block.y == 1U &&
                    production.block.z == 1U &&
                    production.dynamic_shared_bytes == 0U,
                "public fused GDN norm locks the 48x256 launch topology");
    test.expect(production.grid.x == detail.grid.x &&
                    production.grid.y == detail.grid.y &&
                    production.grid.z == detail.grid.z &&
                    production.block.x == detail.block.x &&
                    production.block.y == detail.block.y &&
                    production.block.z == detail.block.z &&
                    production.dynamic_shared_bytes ==
                        detail.dynamic_shared_bytes,
                "public and detail fused GDN norm launch parameters match");
    test.expect(production.function != shared_tree.function,
                "warp-tail and shared-tree fused GDN norm kernels are distinct");
    test.expect(production.grid.x == shared_tree.grid.x &&
                    production.grid.y == shared_tree.grid.y &&
                    production.grid.z == shared_tree.grid.z &&
                    production.block.x == shared_tree.block.x &&
                    production.block.y == shared_tree.block.y &&
                    production.block.z == shared_tree.block.z &&
                    production.dynamic_shared_bytes ==
                        shared_tree.dynamic_shared_bytes,
                "warp-tail/shared-tree fused GDN norm topology matches");
  }

  const CapturedTopology near_miss_topology = capture_topology(
      test,
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
                conv_qkv.data(), a.data(), b.data(), A_log.data(),
                dt_bias.data(), candidate_state.data(),
                candidate_state.data(), kL2Epsilon, near_norm_weight.data(),
                silu_gate.data(), 24U, 256U, kNormEpsilon,
                candidate_output.data(), {},
                static_cast<void*>(capture_stream));
      },
      cudaSuccess, "production fused GDN norm near-miss topology");
  test.expect(near_miss_topology.total_nodes == 2U &&
                  near_miss_topology.kernel_nodes == 2U &&
                  near_miss_topology.edges == 1U,
              "production fused GDN norm near miss captures ordered pair");

  const CapturedTopology reference_topology = capture_topology(
      test,
      [&](cudaStream_t capture_stream) {
        const int gdn_status =
            q3x::runtime::launch_gated_delta_net_update_warp_parallel_cuda(
                conv_qkv.data(), a.data(), b.data(), A_log.data(),
                dt_bias.data(), reference_state.data(),
                reference_state.data(), kL2Epsilon, reference_output.data(),
                {}, static_cast<void*>(capture_stream));
        if (gdn_status != static_cast<int>(cudaSuccess)) {
          return gdn_status;
        }
        return q3x::runtime::
            launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                reference_output.data(), norm_weight.data(),
                silu_gate.data(), q3x::runtime::kGdnValueHeadCount,
                q3x::runtime::kGdnHeadDimension, kNormEpsilon,
                reference_output.data(), static_cast<void*>(capture_stream));
      },
      cudaSuccess, "production GDN norm reference topology");
  test.expect(reference_topology.total_nodes == 2U &&
                  reference_topology.kernel_nodes == 2U &&
                  reference_topology.edges == 1U,
              "production GDN norm reference captures ordered pair");

  const CapturedTopology invalid_topology = capture_topology(
      test,
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
                conv_qkv.data(), a.data(), b.data(), A_log.data(),
                dt_bias.data(), candidate_state.data(),
                candidate_state.data(), kL2Epsilon, norm_weight.data(),
                candidate_output.data(), q3x::runtime::kGdnValueHeadCount,
                q3x::runtime::kGdnHeadDimension, kNormEpsilon,
                candidate_output.data(), {},
                static_cast<void*>(capture_stream));
      },
      cudaErrorInvalidValue, "production fused GDN norm invalid topology");
  test.expect(invalid_topology.total_nodes == 0U &&
                  invalid_topology.kernel_nodes == 0U &&
                  invalid_topology.edges == 0U &&
                  invalid_topology.kernel_launches.empty(),
              "production fused GDN norm invalid path captures no nodes");

  const bool bitwise_and_identity_gate =
      test.failures() == failures_before;

  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int shared_tree_registers_per_thread = 0;
  std::size_t shared_tree_static_shared_bytes = 0U;
  std::size_t shared_tree_local_bytes = 0U;
  int shared_tree_maximum_threads_per_block = 0;
  int shared_tree_active_blocks_per_sm = 0;
  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_gated_delta_net_update_warp_eight_row_plain_rms_norm_silu_gate_resources_test_cuda(
              &registers_per_thread, &static_shared_bytes, &local_bytes,
              &maximum_threads_per_block, &active_blocks_per_sm)),
      "fused GDN norm query resources");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           query_gated_delta_net_update_warp_eight_row_plain_rms_norm_silu_gate_shared_tree_resources_test_cuda(
                               &shared_tree_registers_per_thread,
                               &shared_tree_static_shared_bytes,
                               &shared_tree_local_bytes,
                               &shared_tree_maximum_threads_per_block,
                               &shared_tree_active_blocks_per_sm)),
                       "shared-tree fused GDN norm query resources");
  if (!ready) {
    return;
  }
  constexpr std::size_t kExpectedSharedBytes =
      (3U * q3x::runtime::kGdnHeadDimension + 2U +
       8U * 8U * (q3x::runtime::kGdnHeadDimension + 1U)) *
      sizeof(float);
  test.expect(registers_per_thread <= 48,
              "fused GDN norm uses at most 48 registers/thread");
  test.expect(static_shared_bytes == kExpectedSharedBytes,
              "fused GDN norm reuses production static shared storage");
  test.expect(local_bytes == 0U,
              "fused GDN norm has no local-memory spill");
  test.expect(maximum_threads_per_block >= 256,
              "fused GDN norm supports its 256-thread block");
  test.expect(active_blocks_per_sm >= 3,
              "fused GDN norm can resident-launch all 48 head CTAs");
  test.expect(registers_per_thread <= shared_tree_registers_per_thread,
              "warp-tail fused GDN norm does not increase registers/thread");
  test.expect(static_shared_bytes == shared_tree_static_shared_bytes,
              "warp-tail/shared-tree fused GDN norm static shared matches");
  test.expect(local_bytes <= shared_tree_local_bytes,
              "warp-tail fused GDN norm does not increase local memory");
  test.expect(maximum_threads_per_block >=
                  shared_tree_maximum_threads_per_block,
              "warp-tail fused GDN norm does not reduce max block size");
  test.expect(active_blocks_per_sm >= shared_tree_active_blocks_per_sm,
              "warp-tail fused GDN norm does not reduce occupancy");
  const bool resource_gate = registers_per_thread <= 48 &&
                             static_shared_bytes == kExpectedSharedBytes &&
                             local_bytes == 0U &&
                             maximum_threads_per_block >= 256 &&
                             active_blocks_per_sm >= 3 &&
                             registers_per_thread <=
                                 shared_tree_registers_per_thread &&
                             static_shared_bytes ==
                                 shared_tree_static_shared_bytes &&
                             local_bytes <= shared_tree_local_bytes &&
                             maximum_threads_per_block >=
                                 shared_tree_maximum_threads_per_block &&
                             active_blocks_per_sm >=
                                 shared_tree_active_blocks_per_sm;
  std::cout << "GDN fused norm/gate resources: registers="
            << registers_per_thread << " static_shared="
            << static_shared_bytes << " local=" << local_bytes
            << " max_threads=" << maximum_threads_per_block
            << " active_blocks_per_sm=" << active_blocks_per_sm << '\n';
  std::cout << "GDN fused norm/gate shared-tree resources: registers="
            << shared_tree_registers_per_thread << " static_shared="
            << shared_tree_static_shared_bytes
            << " local=" << shared_tree_local_bytes
            << " max_threads=" << shared_tree_maximum_threads_per_block
            << " active_blocks_per_sm=" << shared_tree_active_blocks_per_sm
            << '\n';

  const char* const warp_tail_performance_environment =
      std::getenv("Q3X_RUN_GDN_NORM_GATE_WARP_TAIL_PERF");
  const char* const fusion_performance_environment =
      std::getenv("Q3X_RUN_GDN_NORM_GATE_PERF");
  const bool run_warp_tail_performance =
      warp_tail_performance_environment != nullptr &&
      std::string(warp_tail_performance_environment) == "1";
  const bool run_fusion_performance =
      fusion_performance_environment != nullptr &&
      std::string(fusion_performance_environment) == "1";
  if (!run_warp_tail_performance) {
    std::cout << "SKIP: fused GDN norm/gate warp-tail performance segment; "
                 "set Q3X_RUN_GDN_NORM_GATE_WARP_TAIL_PERF=1 to enable\n";
  }
  if (!run_fusion_performance) {
    std::cout << "SKIP: fused GDN norm/gate end-to-end performance segment; "
                 "set Q3X_RUN_GDN_NORM_GATE_PERF=1 to enable\n";
  }
  if (!run_warp_tail_performance && !run_fusion_performance) {
    return;
  }

  fill_gdn_inputs(conv_qkv, a, b, A_log, dt_bias, 11U);
  // Rotate 24 independent persistent states (36 MiB per variant) so a timed
  // pass cannot keep the 1.5 MiB canonical state resident in the 4 MiB Orin
  // L2. Each state advances equally often in every mirrored B-C-C-B pass.
  constexpr std::size_t kPerformanceStateBankCount = 24U;
  constexpr std::size_t kPerformanceWarmupCount =
      2U * kPerformanceStateBankCount;
  constexpr std::size_t kPerformanceIterationCount =
      20U * kPerformanceStateBankCount;
  constexpr std::size_t kPerformanceRounds = 5U;
  constexpr std::size_t kTimedPassCount = 2U * kPerformanceRounds;
  ManagedBuffer<std::uint16_t> reference_state_bank;
  ManagedBuffer<std::uint16_t> candidate_state_bank;
  ready = test.cuda_ok(
      reference_state_bank.allocate(kPerformanceStateBankCount *
                                    q3x::runtime::kGdnStateElements),
      "fused GDN norm allocate reference performance state bank");
  ready = ready && test.cuda_ok(
                       candidate_state_bank.allocate(
                           kPerformanceStateBankCount *
                           q3x::runtime::kGdnStateElements),
                       "fused GDN norm allocate candidate performance state bank");
  if (!ready) {
    return;
  }

  const auto reset_state_bank =
      [&](ManagedBuffer<std::uint16_t>& state_bank,
          ManagedBuffer<std::uint16_t>& output_buffer,
          const std::string& label) -> bool {
    bool reset_ready = true;
    constexpr std::size_t kStateBytes =
        q3x::runtime::kGdnStateElements * sizeof(std::uint16_t);
    for (std::size_t bank = 0U;
         reset_ready && bank < kPerformanceStateBankCount; ++bank) {
      reset_ready = test.cuda_ok(
          cudaMemcpyAsync(
              state_bank.data() + bank * q3x::runtime::kGdnStateElements,
              initial_state.data(), kStateBytes, cudaMemcpyHostToDevice,
              stream),
          label + " copy state " + std::to_string(bank));
    }
    reset_ready =
        reset_ready &&
        test.cuda_ok(cudaMemsetAsync(output_buffer.data(), 0,
                                     output_buffer.size() *
                                         sizeof(std::uint16_t),
                                     stream),
                     label + " clear output");
    reset_ready =
        reset_ready &&
        test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
    return reset_ready;
  };

  if (run_warp_tail_performance) {
    std::size_t reference_cursor = 0U;
    std::size_t candidate_cursor = 0U;
    const auto launch_shared_tree = [&]() {
      const std::size_t bank =
          reference_cursor++ % kPerformanceStateBankCount;
      std::uint16_t* const state =
          reference_state_bank.data() +
          bank * q3x::runtime::kGdnStateElements;
      return q3x::runtime::
          launch_gated_delta_net_update_plain_rms_norm_silu_gate_shared_tree_test_cuda(
              conv_qkv.data(), a.data(), b.data(), A_log.data(),
              dt_bias.data(), state, state, kL2Epsilon, norm_weight.data(),
              silu_gate.data(), kNormEpsilon, reference_output.data(),
              static_cast<void*>(stream));
    };
    const auto launch_candidate = [&]() {
      const std::size_t bank =
          candidate_cursor++ % kPerformanceStateBankCount;
      std::uint16_t* const state =
          candidate_state_bank.data() +
          bank * q3x::runtime::kGdnStateElements;
      return q3x::runtime::gdn_decode_detail::
          launch_gated_delta_net_update_plain_rms_norm_silu_gate_exact_cuda(
              conv_qkv.data(), a.data(), b.data(), A_log.data(),
              dt_bias.data(), state, state, kL2Epsilon, norm_weight.data(),
              silu_gate.data(), kNormEpsilon, candidate_output.data(),
              static_cast<void*>(stream));
    };

    std::array<float, kTimedPassCount> shared_tree_passes{};
    std::array<float, kTimedPassCount> candidate_passes{};
    bool timing_finite = true;
    for (std::size_t round = 0U; round < kPerformanceRounds; ++round) {
      const std::string round_label =
          "GDN norm/gate warp-tail round=" + std::to_string(round + 1U);
      reference_cursor = 0U;
      ready = reset_state_bank(reference_state_bank, reference_output,
                               round_label + " B1 reset");
      const float shared_tree_first =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, kPerformanceWarmupCount,
                      kPerformanceIterationCount, round_label + " B1",
                      launch_shared_tree)
                : std::numeric_limits<float>::quiet_NaN();

      candidate_cursor = 0U;
      ready = ready && reset_state_bank(candidate_state_bank, candidate_output,
                                        round_label + " C1 reset");
      const float candidate_first =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, kPerformanceWarmupCount,
                      kPerformanceIterationCount, round_label + " C1",
                      launch_candidate)
                : std::numeric_limits<float>::quiet_NaN();

      candidate_cursor = 0U;
      ready = ready && reset_state_bank(candidate_state_bank, candidate_output,
                                        round_label + " C2 reset");
      const float candidate_second =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, kPerformanceWarmupCount,
                      kPerformanceIterationCount, round_label + " C2",
                      launch_candidate)
                : std::numeric_limits<float>::quiet_NaN();

      reference_cursor = 0U;
      ready = ready && reset_state_bank(reference_state_bank, reference_output,
                                        round_label + " B2 reset");
      const float shared_tree_second =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, kPerformanceWarmupCount,
                      kPerformanceIterationCount, round_label + " B2",
                      launch_shared_tree)
                : std::numeric_limits<float>::quiet_NaN();

      const std::size_t pass = 2U * round;
      shared_tree_passes[pass] = shared_tree_first;
      shared_tree_passes[pass + 1U] = shared_tree_second;
      candidate_passes[pass] = candidate_first;
      candidate_passes[pass + 1U] = candidate_second;
      timing_finite = timing_finite && ready &&
                      std::isfinite(shared_tree_first) &&
                      std::isfinite(candidate_first) &&
                      std::isfinite(candidate_second) &&
                      std::isfinite(shared_tree_second);
      std::cout << "PERF_GDN_NORM_GATE_WARP_TAIL_ROUND: round=" << round + 1U
                << " order=B-C-C-B state_bank="
                << kPerformanceStateBankCount
                << " warmups=" << kPerformanceWarmupCount
                << " iterations=" << kPerformanceIterationCount
                << " shared_tree_pass1_ms=" << shared_tree_first
                << " candidate_pass1_ms=" << candidate_first
                << " candidate_pass2_ms=" << candidate_second
                << " shared_tree_pass2_ms=" << shared_tree_second << '\n';
    }

    expect_bf16_buffer_bitwise_equal(
        test, candidate_state_bank.data(), reference_state_bank.data(),
        candidate_state_bank.size(),
        "fused GDN norm shared-tree/warp-tail post-timing state bank");
    expect_bf16_buffer_bitwise_equal(
        test, candidate_output.data(), reference_output.data(),
        q3x::runtime::kGdnVElements,
        "fused GDN norm shared-tree/warp-tail post-timing output");
    const bool timing_bitwise_gate =
        timing_finite && test.failures() == failures_before;

    const auto median = [](std::array<float, kTimedPassCount> values) {
      std::sort(values.begin(), values.end());
      return (values[kTimedPassCount / 2U - 1U] +
              values[kTimedPassCount / 2U]) *
             0.5F;
    };
    const float shared_tree_milliseconds = median(shared_tree_passes);
    const float candidate_milliseconds = median(candidate_passes);
    const float speedup = shared_tree_milliseconds / candidate_milliseconds;
    constexpr float kRequiredSpeedup = 1.005F;
    const bool performance_gate =
        bitwise_and_identity_gate && timing_bitwise_gate && resource_gate &&
        std::isfinite(shared_tree_milliseconds) &&
        std::isfinite(candidate_milliseconds) &&
        shared_tree_milliseconds > 0.0F && candidate_milliseconds > 0.0F &&
        speedup >= kRequiredSpeedup;
    std::cout << "PERF_GDN_NORM_GATE_WARP_TAIL: shared_tree_ms="
              << shared_tree_milliseconds
              << " candidate_ms=" << candidate_milliseconds
              << " speedup=" << speedup
              << " required_speedup=" << kRequiredSpeedup
              << " bitwise="
              << (bitwise_and_identity_gate && timing_bitwise_gate ? "true"
                                                                  : "false")
              << " resources=" << (resource_gate ? "true" : "false")
              << " gate="
              << (performance_gate ? "PASS" : "FAIL") << '\n';
    test.expect(
        performance_gate,
        "fused GDN norm/gate warp-tail clears same-binary performance gate");
  }

  if (run_fusion_performance) {
    const int failures_before_fusion_timing = test.failures();
    std::size_t ordered_cursor = 0U;
    std::size_t fused_cursor = 0U;
    const auto launch_ordered = [&]() {
      const std::size_t bank =
          ordered_cursor++ % kPerformanceStateBankCount;
      std::uint16_t* const state =
          reference_state_bank.data() +
          bank * q3x::runtime::kGdnStateElements;
      const int gdn_status =
          q3x::runtime::launch_gated_delta_net_update_warp_parallel_cuda(
              conv_qkv.data(), a.data(), b.data(), A_log.data(),
              dt_bias.data(), state, state, kL2Epsilon,
              reference_output.data(), {}, static_cast<void*>(stream));
      if (gdn_status != static_cast<int>(cudaSuccess)) {
        return gdn_status;
      }
      return q3x::runtime::
          launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
              reference_output.data(), norm_weight.data(), silu_gate.data(),
              q3x::runtime::kGdnValueHeadCount,
              q3x::runtime::kGdnHeadDimension, kNormEpsilon,
              reference_output.data(), static_cast<void*>(stream));
    };
    const auto launch_fused = [&]() {
      const std::size_t bank = fused_cursor++ % kPerformanceStateBankCount;
      std::uint16_t* const state =
          candidate_state_bank.data() +
          bank * q3x::runtime::kGdnStateElements;
      return q3x::runtime::
          launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
              conv_qkv.data(), a.data(), b.data(), A_log.data(),
              dt_bias.data(), state, state, kL2Epsilon, norm_weight.data(),
              silu_gate.data(), q3x::runtime::kGdnValueHeadCount,
              q3x::runtime::kGdnHeadDimension, kNormEpsilon,
              candidate_output.data(), {}, static_cast<void*>(stream));
    };

    std::array<float, kTimedPassCount> ordered_passes{};
    std::array<float, kTimedPassCount> fused_passes{};
    bool timing_finite = true;
    for (std::size_t round = 0U; round < kPerformanceRounds; ++round) {
      const std::string round_label =
          "GDN norm/gate fusion round=" + std::to_string(round + 1U);
      ordered_cursor = 0U;
      ready = reset_state_bank(reference_state_bank, reference_output,
                               round_label + " B1 reset");
      const float ordered_first =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, kPerformanceWarmupCount,
                      kPerformanceIterationCount, round_label + " B1",
                      launch_ordered)
                : std::numeric_limits<float>::quiet_NaN();

      fused_cursor = 0U;
      ready = ready && reset_state_bank(candidate_state_bank, candidate_output,
                                        round_label + " C1 reset");
      const float fused_first =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, kPerformanceWarmupCount,
                      kPerformanceIterationCount, round_label + " C1",
                      launch_fused)
                : std::numeric_limits<float>::quiet_NaN();

      fused_cursor = 0U;
      ready = ready && reset_state_bank(candidate_state_bank, candidate_output,
                                        round_label + " C2 reset");
      const float fused_second =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, kPerformanceWarmupCount,
                      kPerformanceIterationCount, round_label + " C2",
                      launch_fused)
                : std::numeric_limits<float>::quiet_NaN();

      ordered_cursor = 0U;
      ready = ready && reset_state_bank(reference_state_bank, reference_output,
                                        round_label + " B2 reset");
      const float ordered_second =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, kPerformanceWarmupCount,
                      kPerformanceIterationCount, round_label + " B2",
                      launch_ordered)
                : std::numeric_limits<float>::quiet_NaN();

      const std::size_t pass = 2U * round;
      ordered_passes[pass] = ordered_first;
      ordered_passes[pass + 1U] = ordered_second;
      fused_passes[pass] = fused_first;
      fused_passes[pass + 1U] = fused_second;
      timing_finite = timing_finite && ready &&
                      std::isfinite(ordered_first) &&
                      std::isfinite(fused_first) &&
                      std::isfinite(fused_second) &&
                      std::isfinite(ordered_second);
      std::cout << "PERF_GDN_NORM_GATE_ROUND: round=" << round + 1U
                << " order=B-C-C-B state_bank="
                << kPerformanceStateBankCount
                << " warmups=" << kPerformanceWarmupCount
                << " iterations=" << kPerformanceIterationCount
                << " ordered_pass1_ms=" << ordered_first
                << " candidate_pass1_ms=" << fused_first
                << " candidate_pass2_ms=" << fused_second
                << " ordered_pass2_ms=" << ordered_second << '\n';
    }

    expect_bf16_buffer_bitwise_equal(
        test, candidate_state_bank.data(), reference_state_bank.data(),
        candidate_state_bank.size(),
        "fused GDN norm post-timing state bank");
    expect_bf16_buffer_bitwise_equal(
        test, candidate_output.data(), reference_output.data(),
        q3x::runtime::kGdnVElements,
        "fused GDN norm post-timing output");
    const bool timing_bitwise_gate =
        timing_finite && test.failures() == failures_before_fusion_timing;

    const auto median = [](std::array<float, kTimedPassCount> values) {
      std::sort(values.begin(), values.end());
      return (values[kTimedPassCount / 2U - 1U] +
              values[kTimedPassCount / 2U]) *
             0.5F;
    };
    const float ordered_milliseconds = median(ordered_passes);
    const float candidate_milliseconds = median(fused_passes);
    const float speedup = ordered_milliseconds / candidate_milliseconds;
    constexpr float kRequiredSpeedup = 1.005F;
    const bool performance_gate =
        bitwise_and_identity_gate && timing_bitwise_gate && resource_gate &&
        std::isfinite(ordered_milliseconds) &&
        std::isfinite(candidate_milliseconds) &&
        ordered_milliseconds > 0.0F && candidate_milliseconds > 0.0F &&
        speedup >= kRequiredSpeedup;
    std::cout << "PERF_GDN_NORM_GATE: ordered_ms=" << ordered_milliseconds
              << " candidate_ms=" << candidate_milliseconds
              << " speedup=" << speedup
              << " required_speedup=" << kRequiredSpeedup
              << " bitwise="
              << (bitwise_and_identity_gate && timing_bitwise_gate ? "true"
                                                                  : "false")
              << " resources=" << (resource_gate ? "true" : "false")
              << " gate=" << (performance_gate ? "PASS" : "FAIL") << '\n';
    test.expect(performance_gate,
                "fused GDN norm/gate clears same-binary performance gate");
  }
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
      ready = launch_after_stale(
          test, stream,
          "GDN production disjoint M=" + std::to_string(token_count),
          [&]() {
            return q3x::runtime::
                launch_gated_delta_net_update_tile_warp_parallel_cuda(
                    conv_qkv.data(), token_count, a.data(), b.data(),
                    A_log.data(), dt_bias.data(),
                    row_pair_disjoint_input_state.data(),
                    row_pair_disjoint_output_state.data(), 1.0e-6F,
                    row_pair_disjoint_output.data(), {},
                    static_cast<void*>(stream));
          });
      if (!ready) {
        return;
      }
      expect_bf16_buffer_bitwise_equal(
          test, row_pair_disjoint_output.data(), sequential_output.data(),
          token_count * q3x::runtime::kGdnVElements,
          "CUDA GDN production disjoint output M=" +
              std::to_string(token_count));
      expect_bf16_buffer_bitwise_equal(
          test, row_pair_disjoint_output_state.data(),
          sequential_state.data(), q3x::runtime::kGdnStateElements,
          "CUDA GDN production disjoint state M=" +
              std::to_string(token_count));
      expect_bf16_buffer_bitwise_equal(
          test, row_pair_disjoint_input_state.data(), initial_state.data(),
          q3x::runtime::kGdnStateElements,
          "CUDA GDN production preserves disjoint input M=" +
              std::to_string(token_count));
      std::fill_n(row_pair_disjoint_output_state.data(),
                  row_pair_disjoint_output_state.size(),
                  static_cast<std::uint16_t>(0x5a5aU));
      std::fill_n(row_pair_disjoint_output.data(),
                  row_pair_disjoint_output.size(),
                  static_cast<std::uint16_t>(0U));

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

  constexpr std::size_t kSplitTokens = 8U;
  static_assert(kMaximumTokens == 2U * kSplitTokens);
  std::copy(initial_state.begin(), initial_state.end(), tile_state.data());
  std::copy(initial_state.begin(), initial_state.end(), warp_state.data());
  std::fill_n(tile_output.data(), tile_output.size(),
              static_cast<std::uint16_t>(0U));
  std::fill_n(warp_output.data(), warp_output.size(),
              static_cast<std::uint16_t>(0U));
  ready = launch_after_stale(test, stream, "GDN warp C16 oracle", [&]() {
    return q3x::runtime::launch_gated_delta_net_update_tile_warp_parallel_cuda(
        conv_qkv.data(), kMaximumTokens, a.data(), b.data(), A_log.data(),
        dt_bias.data(), tile_state.data(), tile_state.data(), 1.0e-6F,
        tile_output.data(), {}, static_cast<void*>(stream));
  });
  if (!ready) {
    return;
  }
  ready = launch_after_stale(
      test, stream, "GDN warp first C8 split oracle", [&]() {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_parallel_cuda(
                conv_qkv.data(), kSplitTokens, a.data(), b.data(),
                A_log.data(), dt_bias.data(), warp_state.data(),
                warp_state.data(), 1.0e-6F, warp_output.data(), {},
                static_cast<void*>(stream));
      });
  if (!ready) {
    return;
  }
  ready = launch_after_stale(
      test, stream, "GDN warp second C8 split oracle", [&]() {
        constexpr std::size_t kQkvOffset =
            kSplitTokens * q3x::runtime::kGdnQkvChannels;
        constexpr std::size_t kScalarOffset =
            kSplitTokens * q3x::runtime::kGdnValueHeadCount;
        constexpr std::size_t kOutputOffset =
            kSplitTokens * q3x::runtime::kGdnVElements;
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_parallel_cuda(
                conv_qkv.data() + kQkvOffset, kSplitTokens,
                a.data() + kScalarOffset, b.data() + kScalarOffset,
                A_log.data(), dt_bias.data(), warp_state.data(),
                warp_state.data(), 1.0e-6F,
                warp_output.data() + kOutputOffset, {},
                static_cast<void*>(stream));
      });
  if (!ready) {
    return;
  }
  expect_bf16_buffer_bitwise_equal(
      test, tile_output.data(), warp_output.data(),
      kMaximumTokens * q3x::runtime::kGdnVElements,
      "CUDA GDN warp C16 output equals sequential C8+C8");
  expect_bf16_buffer_bitwise_equal(
      test, tile_state.data(), warp_state.data(),
      q3x::runtime::kGdnStateElements,
      "CUDA GDN warp C16 state equals sequential C8+C8");

  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_gated_delta_net_update_tile_reference_cuda(
              conv_qkv.data(), 2U, a.data(), b.data(), A_log.data(),
              dt_bias.data(), tile_state.data(), tile_state.data(), 1.0e-6F,
              conv_qkv.data())) == cudaErrorInvalidValue,
      "CUDA tile GDN rejects output/conv alias");

  constexpr std::size_t kWarmupCount = 5U;
  constexpr std::size_t kIterationCount = 50U;
  constexpr std::size_t kPerformanceTileTokenCount = 8U;
  static_assert(kPerformanceTileTokenCount <= kMaximumTokens);
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
            conv_qkv.data(), kPerformanceTileTokenCount, a.data(), b.data(),
            A_log.data(), dt_bias.data(), tile_state.data(), tile_state.data(),
            1.0e-6F, tile_output.data(), {}, static_cast<void*>(stream));
      });
  const float warp_m8_ms = measure_average_cuda_milliseconds(
      test, stream, kWarmupCount, kIterationCount, "GDN warp M8", [&]() {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_parallel_cuda(
                conv_qkv.data(), kPerformanceTileTokenCount, a.data(),
                b.data(), A_log.data(), dt_bias.data(), warp_state.data(),
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
        measure_mirrored(kPerformanceTileTokenCount, "M8");
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

void test_gdn_register_state_m16_candidate(TestContext& test,
                                           cudaStream_t stream) {
  constexpr std::size_t kTokenCount =
      q3x::runtime::kGdnMaximumTileTokenCount;
  constexpr std::size_t kOutputElements =
      kTokenCount * q3x::runtime::kGdnVElements;
  constexpr std::size_t kGuardElements = 17U;
  constexpr float kL2Epsilon = 1.0e-6F;
  constexpr std::uint16_t kPoison = 0x7fc1U;
  constexpr std::uint16_t kPrefixCanary = 0xa55aU;
  constexpr std::uint16_t kSuffixCanary = 0x5aa5U;

  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> production_state;
  ManagedBuffer<std::uint16_t> candidate_inplace_state;
  ManagedBuffer<std::uint16_t> candidate_disjoint_input_state;
  ManagedBuffer<std::uint16_t> candidate_disjoint_output_state;
  ManagedBuffer<std::uint16_t> candidate_replay_state;
  ManagedBuffer<std::uint16_t> production_output;
  ManagedBuffer<std::uint16_t> candidate_inplace_output;
  ManagedBuffer<std::uint16_t> candidate_disjoint_output;
  ManagedBuffer<std::uint16_t> candidate_replay_output;
  ManagedBuffer<std::uint16_t> guarded_candidate_inplace_state;
  ManagedBuffer<std::uint16_t> guarded_candidate_inplace_output;
  bool ready = test.cuda_ok(
      conv_qkv.allocate(kTokenCount * q3x::runtime::kGdnQkvChannels),
      "GDN register-state M16 allocate conv QKV");
  ready = ready && test.cuda_ok(
                       a.allocate(kTokenCount *
                                  q3x::runtime::kGdnValueHeadCount),
                       "GDN register-state M16 allocate a");
  ready = ready && test.cuda_ok(
                       b.allocate(kTokenCount *
                                  q3x::runtime::kGdnValueHeadCount),
                       "GDN register-state M16 allocate b");
  ready = ready && test.cuda_ok(
                       A_log.allocate(q3x::runtime::kGdnValueHeadCount),
                       "GDN register-state M16 allocate A_log");
  ready = ready && test.cuda_ok(
                       dt_bias.allocate(q3x::runtime::kGdnValueHeadCount),
                       "GDN register-state M16 allocate dt_bias");
  ready = ready && test.cuda_ok(
                       production_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "GDN register-state M16 allocate production state");
  ready = ready && test.cuda_ok(
                       candidate_inplace_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "GDN register-state M16 allocate in-place state");
  ready = ready && test.cuda_ok(
                       candidate_disjoint_input_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "GDN register-state M16 allocate disjoint input");
  ready = ready && test.cuda_ok(
                       candidate_disjoint_output_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "GDN register-state M16 allocate disjoint state");
  ready = ready && test.cuda_ok(
                       candidate_replay_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       "GDN register-state M16 allocate replay state");
  ready = ready && test.cuda_ok(
                       production_output.allocate(kOutputElements),
                       "GDN register-state M16 allocate production output");
  ready = ready && test.cuda_ok(
                       candidate_inplace_output.allocate(kOutputElements),
                       "GDN register-state M16 allocate in-place output");
  ready = ready && test.cuda_ok(
                       candidate_disjoint_output.allocate(kOutputElements),
                       "GDN register-state M16 allocate disjoint output");
  ready = ready && test.cuda_ok(
                       candidate_replay_output.allocate(kOutputElements),
                       "GDN register-state M16 allocate replay output");
  ready = ready && test.cuda_ok(
                       guarded_candidate_inplace_state.allocate(
                           q3x::runtime::kGdnStateElements +
                           2U * kGuardElements),
                       "GDN register-state M16 allocate guarded state");
  ready = ready && test.cuda_ok(
                       guarded_candidate_inplace_output.allocate(
                           kOutputElements + 2U * kGuardElements),
                       "GDN register-state M16 allocate guarded output");
  if (!ready) {
    return;
  }

  fill_gdn_tile_inputs(conv_qkv, a, b, A_log, dt_bias);
  const std::vector<std::uint16_t> initial_conv_qkv(
      conv_qkv.data(), conv_qkv.data() + conv_qkv.size());
  const std::vector<std::uint16_t> initial_a(a.data(), a.data() + a.size());
  const std::vector<std::uint16_t> initial_b(b.data(), b.data() + b.size());
  const std::vector<std::uint16_t> initial_A_log(
      A_log.data(), A_log.data() + A_log.size());
  const std::vector<std::uint16_t> initial_dt_bias(
      dt_bias.data(), dt_bias.data() + dt_bias.size());
  std::vector<std::uint16_t> initial_state(
      q3x::runtime::kGdnStateElements);
  for (std::size_t index = 0U; index < initial_state.size(); ++index) {
    const int centered = static_cast<int>((index * 5U) % 23U) - 11;
    initial_state[index] =
        encode_bf16(static_cast<float>(centered) / 512.0F);
  }
  std::copy(initial_state.begin(), initial_state.end(),
            production_state.data());
  std::copy(initial_state.begin(), initial_state.end(),
            candidate_inplace_state.data());
  std::copy(initial_state.begin(), initial_state.end(),
            candidate_disjoint_input_state.data());
  std::fill_n(candidate_disjoint_output_state.data(),
              candidate_disjoint_output_state.size(), kPoison);
  std::fill_n(candidate_replay_state.data(), candidate_replay_state.size(),
              kPoison);
  std::fill_n(production_output.data(), production_output.size(), kPoison);
  std::fill_n(candidate_inplace_output.data(),
              candidate_inplace_output.size(), kPoison);
  std::fill_n(candidate_disjoint_output.data(),
              candidate_disjoint_output.size(), kPoison);
  std::fill_n(candidate_replay_output.data(), candidate_replay_output.size(),
              kPoison);
  std::uint16_t* const guarded_state =
      guarded_candidate_inplace_state.data() + kGuardElements;
  std::uint16_t* const guarded_output =
      guarded_candidate_inplace_output.data() + kGuardElements;
  std::fill_n(guarded_candidate_inplace_state.data(), kGuardElements,
              kPrefixCanary);
  std::copy(initial_state.begin(), initial_state.end(), guarded_state);
  std::fill_n(guarded_state + q3x::runtime::kGdnStateElements,
              kGuardElements, kSuffixCanary);
  std::fill_n(guarded_candidate_inplace_output.data(), kGuardElements,
              kPrefixCanary);
  std::fill_n(guarded_output, kOutputElements, kPoison);
  std::fill_n(guarded_output + kOutputElements, kGuardElements,
              kSuffixCanary);

  ready = launch_after_stale(
      test, stream, "GDN predecessor row8 M16 oracle", [&]() {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_eight_row_lane_striped_test_cuda(
                conv_qkv.data(), kTokenCount, a.data(), b.data(),
                A_log.data(), dt_bias.data(), production_state.data(),
                production_state.data(), kL2Epsilon,
                production_output.data(), static_cast<void*>(stream));
      });
  ready = ready && launch_after_stale(
                       test, stream,
                       "GDN register-state M16 guarded in-place", [&]() {
                         return q3x::runtime::
                             launch_gated_delta_net_update_tile_warp_parallel_cuda(
                                 conv_qkv.data(), kTokenCount, a.data(),
                                 b.data(), A_log.data(), dt_bias.data(),
                                 guarded_state, guarded_state, kL2Epsilon,
                                 guarded_output, {},
                                 static_cast<void*>(stream));
                       });
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           launch_gated_delta_net_update_tile_warp_parallel_cuda(
                               conv_qkv.data(), kTokenCount, a.data(),
                               b.data(), A_log.data(), dt_bias.data(),
                               candidate_inplace_state.data(),
                               candidate_inplace_state.data(), kL2Epsilon,
                               candidate_inplace_output.data(), {},
                               static_cast<void*>(stream))),
                       "GDN register-state M16 in-place launch");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           launch_gated_delta_net_update_tile_warp_parallel_cuda(
                               conv_qkv.data(), kTokenCount, a.data(),
                               b.data(), A_log.data(), dt_bias.data(),
                               candidate_disjoint_input_state.data(),
                               candidate_disjoint_output_state.data(),
                               kL2Epsilon,
                               candidate_disjoint_output.data(), {},
                               static_cast<void*>(stream))),
                       "GDN register-state M16 disjoint launch");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           launch_gated_delta_net_update_tile_warp_parallel_cuda(
                               conv_qkv.data(), kTokenCount, a.data(),
                               b.data(), A_log.data(), dt_bias.data(),
                               candidate_disjoint_input_state.data(),
                               candidate_replay_state.data(), kL2Epsilon,
                               candidate_replay_output.data(), {},
                               static_cast<void*>(stream))),
                       "GDN register-state M16 replay launch");
  ready = ready && test.cuda_ok(
                       cudaStreamSynchronize(stream),
                       "GDN register-state M16 correctness synchronize");
  if (!ready) {
    return;
  }

  expect_bf16_buffer_bitwise_equal(
      test, candidate_inplace_output.data(), production_output.data(),
      kOutputElements,
      "GDN register-state M16 in-place output equals production");
  expect_bf16_buffer_bitwise_equal(
      test, candidate_inplace_state.data(), production_state.data(),
      q3x::runtime::kGdnStateElements,
      "GDN register-state M16 in-place state equals production");
  expect_bf16_buffer_bitwise_equal(
      test, candidate_disjoint_output.data(), production_output.data(),
      kOutputElements,
      "GDN register-state M16 disjoint output equals production");
  expect_bf16_buffer_bitwise_equal(
      test, candidate_disjoint_output_state.data(), production_state.data(),
      q3x::runtime::kGdnStateElements,
      "GDN register-state M16 disjoint state equals production");
  expect_bf16_buffer_bitwise_equal(
      test, candidate_disjoint_input_state.data(), initial_state.data(),
      initial_state.size(),
      "GDN register-state M16 preserves disjoint input");
  expect_bf16_buffer_bitwise_equal(
      test, candidate_replay_output.data(), candidate_disjoint_output.data(),
      kOutputElements, "GDN register-state M16 replay output is bitwise");
  expect_bf16_buffer_bitwise_equal(
      test, candidate_replay_state.data(),
      candidate_disjoint_output_state.data(),
      q3x::runtime::kGdnStateElements,
      "GDN register-state M16 replay state is bitwise");
  expect_bf16_buffer_bitwise_equal(
      test, guarded_output, production_output.data(), kOutputElements,
      "GDN register-state M16 guarded output equals production");
  expect_bf16_buffer_bitwise_equal(
      test, guarded_state, production_state.data(),
      q3x::runtime::kGdnStateElements,
      "GDN register-state M16 guarded state equals production");

  const auto expect_guards = [&](const ManagedBuffer<std::uint16_t>& storage,
                                 const std::size_t active_elements,
                                 const std::string& label) {
    const bool prefix_intact = std::all_of(
        storage.data(), storage.data() + kGuardElements,
        [](const std::uint16_t value) { return value == kPrefixCanary; });
    const std::uint16_t* const suffix =
        storage.data() + kGuardElements + active_elements;
    const bool suffix_intact =
        std::all_of(suffix, suffix + kGuardElements,
                    [](const std::uint16_t value) {
                      return value == kSuffixCanary;
                    });
    test.expect(prefix_intact && suffix_intact,
                label + " prefix/suffix canaries remain intact");
  };
  expect_guards(guarded_candidate_inplace_state,
                q3x::runtime::kGdnStateElements,
                "GDN register-state M16 guarded state");
  expect_guards(guarded_candidate_inplace_output, kOutputElements,
                "GDN register-state M16 guarded output");

  const auto expect_finite_and_overwritten =
      [&](const std::uint16_t* const values, const std::size_t count,
          const std::string& label) {
        std::size_t poison_count = 0U;
        std::size_t nonfinite_count = 0U;
        for (std::size_t index = 0U; index < count; ++index) {
          poison_count += values[index] == kPoison ? 1U : 0U;
          nonfinite_count += std::isfinite(decode_bf16(values[index]))
                                 ? 0U
                                 : 1U;
        }
        test.expect(poison_count == 0U && nonfinite_count == 0U,
                    label + " poison=" + std::to_string(poison_count) +
                        ", nonfinite=" +
                        std::to_string(nonfinite_count));
      };
  expect_finite_and_overwritten(
      candidate_disjoint_output.data(), kOutputElements,
      "GDN register-state M16 overwrites finite output");
  expect_finite_and_overwritten(
      candidate_disjoint_output_state.data(),
      q3x::runtime::kGdnStateElements,
      "GDN register-state M16 overwrites finite state");

  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int predecessor_registers_per_thread = 0;
  std::size_t predecessor_static_shared_bytes = 0U;
  std::size_t predecessor_local_bytes = 0U;
  int predecessor_maximum_threads_per_block = 0;
  int predecessor_active_blocks_per_sm = 0;
  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_gated_delta_net_update_warp_eight_row_register_state_m16_resources_test_cuda(
              &registers_per_thread, &static_shared_bytes, &local_bytes,
              &maximum_threads_per_block, &active_blocks_per_sm)),
      "GDN register-state M16 query resources");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           query_gated_delta_net_update_warp_eight_row_lane_striped_resources_test_cuda(
                               &predecessor_registers_per_thread,
                               &predecessor_static_shared_bytes,
                               &predecessor_local_bytes,
                               &predecessor_maximum_threads_per_block,
                               &predecessor_active_blocks_per_sm)),
                       "GDN predecessor row8 query resources");
  if (!ready) {
    return;
  }
  constexpr int kMaximumRegistersPerThread = 85;
  constexpr std::size_t kExpectedStaticSharedBytes = 34056U;
  const bool resource_gate =
      registers_per_thread <= kMaximumRegistersPerThread &&
      static_shared_bytes == kExpectedStaticSharedBytes && local_bytes == 0U &&
      maximum_threads_per_block == 256 && active_blocks_per_sm >= 3;
  const bool predecessor_resource_gate =
      predecessor_registers_per_thread == 40 &&
      predecessor_static_shared_bytes == 34568U &&
      predecessor_local_bytes == 0U &&
      predecessor_maximum_threads_per_block >= 256 &&
      predecessor_active_blocks_per_sm == 4;
  std::cout << "GDN_REGISTER_STATE_M16_RESOURCES: registers="
            << registers_per_thread
            << " static_shared_bytes=" << static_shared_bytes
            << " local_bytes=" << local_bytes
            << " maximum_threads_per_block=" << maximum_threads_per_block
            << " active_blocks_per_sm=" << active_blocks_per_sm
            << " gate=" << (resource_gate ? "PASS" : "FAIL") << '\n';
  test.expect(resource_gate,
              "GDN register-state M16 clears resource hard gate");
  std::cout << "GDN_PREDECESSOR_ROW8_RESOURCES: registers="
            << predecessor_registers_per_thread
            << " static_shared_bytes=" << predecessor_static_shared_bytes
            << " local_bytes=" << predecessor_local_bytes
            << " maximum_threads_per_block="
            << predecessor_maximum_threads_per_block
            << " active_blocks_per_sm=" << predecessor_active_blocks_per_sm
            << " gate=" << (predecessor_resource_gate ? "PASS" : "FAIL")
            << '\n';
  test.expect(predecessor_resource_gate,
              "GDN predecessor row8 preserves frozen resources");
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          query_gated_delta_net_update_warp_eight_row_register_state_m16_resources_test_cuda(
              nullptr, &static_shared_bytes, &local_bytes,
              &maximum_threads_per_block, &active_blocks_per_sm)) ==
          cudaErrorInvalidValue,
      "GDN register-state M16 resource query rejects null output");

  const CapturedTopology public_m16_topology = capture_topology(
      test,
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_parallel_cuda(
                conv_qkv.data(), kTokenCount, a.data(), b.data(),
                A_log.data(), dt_bias.data(),
                candidate_disjoint_input_state.data(),
                candidate_disjoint_output_state.data(), kL2Epsilon,
                candidate_disjoint_output.data(), {},
                static_cast<void*>(capture_stream));
      },
      cudaSuccess, "GDN public M16 topology");
  const CapturedTopology exact_m16_topology = capture_topology(
      test,
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_eight_row_register_state_m16_test_cuda(
                conv_qkv.data(), kTokenCount, a.data(), b.data(),
                A_log.data(), dt_bias.data(),
                candidate_disjoint_input_state.data(),
                candidate_disjoint_output_state.data(), kL2Epsilon,
                candidate_disjoint_output.data(),
                static_cast<void*>(capture_stream));
      },
      cudaSuccess, "GDN direct exact M16 topology");
  const CapturedTopology predecessor_m16_topology = capture_topology(
      test,
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_eight_row_lane_striped_test_cuda(
                conv_qkv.data(), kTokenCount, a.data(), b.data(),
                A_log.data(), dt_bias.data(),
                candidate_disjoint_input_state.data(),
                candidate_disjoint_output_state.data(), kL2Epsilon,
                candidate_disjoint_output.data(),
                static_cast<void*>(capture_stream));
      },
      cudaSuccess, "GDN direct predecessor M16 topology");
  const CapturedTopology public_m15_topology = capture_topology(
      test,
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_parallel_cuda(
                conv_qkv.data(), kTokenCount - 1U, a.data(), b.data(),
                A_log.data(), dt_bias.data(),
                candidate_disjoint_input_state.data(),
                candidate_disjoint_output_state.data(), kL2Epsilon,
                candidate_disjoint_output.data(), {},
                static_cast<void*>(capture_stream));
      },
      cudaSuccess, "GDN public M15 fallback topology");
  const CapturedTopology predecessor_m15_topology = capture_topology(
      test,
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_eight_row_lane_striped_test_cuda(
                conv_qkv.data(), kTokenCount - 1U, a.data(), b.data(),
                A_log.data(), dt_bias.data(),
                candidate_disjoint_input_state.data(),
                candidate_disjoint_output_state.data(), kL2Epsilon,
                candidate_disjoint_output.data(),
                static_cast<void*>(capture_stream));
      },
      cudaSuccess, "GDN direct predecessor M15 topology");

  const auto expect_single_48x256 = [&](const CapturedTopology& topology,
                                        const std::string& label) {
    test.expect(topology.total_nodes == 1U && topology.kernel_nodes == 1U &&
                    topology.edges == 0U &&
                    topology.kernel_launches.size() == 1U,
                label + " captures exactly one kernel");
    if (topology.kernel_launches.size() == 1U) {
      const auto& launch = topology.kernel_launches.front();
      test.expect(launch.grid.x == q3x::runtime::kGdnValueHeadCount &&
                      launch.grid.y == 1U && launch.grid.z == 1U &&
                      launch.block.x == 256U && launch.block.y == 1U &&
                      launch.block.z == 1U &&
                      launch.dynamic_shared_bytes == 0U,
                  label + " locks 48x256 topology");
    }
  };
  expect_single_48x256(public_m16_topology, "GDN public M16");
  expect_single_48x256(exact_m16_topology, "GDN direct exact M16");
  expect_single_48x256(predecessor_m16_topology,
                       "GDN direct predecessor M16");
  expect_single_48x256(public_m15_topology, "GDN public M15 fallback");
  expect_single_48x256(predecessor_m15_topology,
                       "GDN direct predecessor M15");

  if (public_m16_topology.kernel_launches.size() == 1U &&
      exact_m16_topology.kernel_launches.size() == 1U &&
      predecessor_m16_topology.kernel_launches.size() == 1U) {
    void* const public_function =
        public_m16_topology.kernel_launches.front().function;
    void* const exact_function =
        exact_m16_topology.kernel_launches.front().function;
    void* const predecessor_function =
        predecessor_m16_topology.kernel_launches.front().function;
    test.expect(public_function != nullptr && exact_function != nullptr &&
                    predecessor_function != nullptr &&
                    public_function == exact_function &&
                    public_function != predecessor_function,
                "GDN public M16 selects exact register-state kernel instead "
                "of predecessor");
  }
  if (public_m15_topology.kernel_launches.size() == 1U &&
      predecessor_m15_topology.kernel_launches.size() == 1U) {
    void* const public_function =
        public_m15_topology.kernel_launches.front().function;
    void* const predecessor_function =
        predecessor_m15_topology.kernel_launches.front().function;
    test.expect(public_function != nullptr && predecessor_function != nullptr &&
                    public_function == predecessor_function,
                "GDN public M15 preserves predecessor fallback kernel");
  }

  const auto expect_invalid_empty_capture =
      [&](auto&& launch, const std::string& label) {
        const CapturedTopology topology = capture_topology(
            test, std::forward<decltype(launch)>(launch),
            cudaErrorInvalidValue, label);
        test.expect(topology.total_nodes == 0U &&
                        topology.kernel_nodes == 0U &&
                        topology.edges == 0U &&
                        topology.kernel_launches.empty(),
                    label + " captures no nodes");
      };
  expect_invalid_empty_capture(
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_eight_row_register_state_m16_test_cuda(
                conv_qkv.data(), kTokenCount - 1U, a.data(), b.data(),
                A_log.data(), dt_bias.data(),
                candidate_disjoint_input_state.data(),
                candidate_disjoint_output_state.data(), kL2Epsilon,
                candidate_disjoint_output.data(),
                static_cast<void*>(capture_stream));
      },
      "GDN register-state rejects non-M16 topology");
  expect_invalid_empty_capture(
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_parallel_cuda(
                nullptr, kTokenCount, a.data(), b.data(), A_log.data(),
                dt_bias.data(), candidate_disjoint_input_state.data(),
                candidate_disjoint_output_state.data(), kL2Epsilon,
                candidate_disjoint_output.data(), {},
                static_cast<void*>(capture_stream));
      },
      "GDN public M16 rejects null topology");
  expect_invalid_empty_capture(
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_parallel_cuda(
                conv_qkv.data(), kTokenCount, a.data(), b.data(),
                A_log.data(), dt_bias.data(),
                candidate_disjoint_input_state.data(),
                candidate_disjoint_output_state.data(), kL2Epsilon,
                conv_qkv.data(), {}, static_cast<void*>(capture_stream));
      },
      "GDN public M16 rejects alias topology");
  expect_invalid_empty_capture(
      [&](cudaStream_t capture_stream) {
        return q3x::runtime::
            launch_gated_delta_net_update_tile_warp_parallel_cuda(
                conv_qkv.data(), kTokenCount, a.data(), b.data(),
                A_log.data(), dt_bias.data(),
                candidate_disjoint_input_state.data(),
                candidate_disjoint_output_state.data(),
                std::numeric_limits<float>::quiet_NaN(),
                candidate_disjoint_output.data(), {},
                static_cast<void*>(capture_stream));
      },
      "GDN public M16 rejects epsilon topology");

  expect_bf16_buffer_bitwise_equal(
      test, conv_qkv.data(), initial_conv_qkv.data(), initial_conv_qkv.size(),
      "GDN register-state M16 preserves conv QKV");
  expect_bf16_buffer_bitwise_equal(test, a.data(), initial_a.data(),
                                   initial_a.size(),
                                   "GDN register-state M16 preserves a");
  expect_bf16_buffer_bitwise_equal(test, b.data(), initial_b.data(),
                                   initial_b.size(),
                                   "GDN register-state M16 preserves b");
  expect_bf16_buffer_bitwise_equal(
      test, A_log.data(), initial_A_log.data(), initial_A_log.size(),
      "GDN register-state M16 preserves A_log");
  expect_bf16_buffer_bitwise_equal(
      test, dt_bias.data(), initial_dt_bias.data(), initial_dt_bias.size(),
      "GDN register-state M16 preserves dt_bias");
}

void run_optional_gdn_register_state_m16_performance(TestContext& test,
                                                     cudaStream_t stream) {
  const char* const value =
      std::getenv("Q3X_RUN_GDN_REGISTER_STATE_M16_PERF");
  const bool enabled = value != nullptr && value[0] != '\0' &&
                       !(value[0] == '0' && value[1] == '\0');
  if (!enabled) {
    std::cout
        << "SKIP: GDN register-state M16 performance segment; set "
           "Q3X_RUN_GDN_REGISTER_STATE_M16_PERF=1 to enable\n";
    return;
  }

  constexpr std::size_t kTokenCount =
      q3x::runtime::kGdnMaximumTileTokenCount;
  constexpr std::size_t kBankCount = 24U;
  constexpr std::size_t kWarmupLaunches = 48U;
  constexpr std::size_t kMeasuredLaunches = 480U;
  constexpr std::size_t kRoundCount = 5U;
  constexpr std::size_t kPassesPerVariant = 2U * kRoundCount;
  constexpr std::size_t kOutputElementsPerBank =
      kTokenCount * q3x::runtime::kGdnVElements;
  constexpr std::size_t kStatePoolElements =
      kBankCount * q3x::runtime::kGdnStateElements;
  constexpr std::size_t kOutputPoolElements =
      kBankCount * kOutputElementsPerBank;
  constexpr float kL2Epsilon = 1.0e-6F;
  constexpr float kRequiredSpeedup = 1.20F;
  static_assert(kWarmupLaunches % kBankCount == 0U);
  static_assert(kMeasuredLaunches % kBankCount == 0U);
  static_assert(kStatePoolElements * sizeof(std::uint16_t) ==
                36U * 1024U * 1024U);

  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> immutable_states;
  ManagedBuffer<std::uint16_t> production_states;
  ManagedBuffer<std::uint16_t> candidate_states;
  ManagedBuffer<std::uint16_t> production_outputs;
  ManagedBuffer<std::uint16_t> candidate_outputs;
  bool ready = test.cuda_ok(
      conv_qkv.allocate(kTokenCount * q3x::runtime::kGdnQkvChannels),
      "GDN register-state perf allocate conv QKV");
  ready = ready && test.cuda_ok(
                       a.allocate(kTokenCount *
                                  q3x::runtime::kGdnValueHeadCount),
                       "GDN register-state perf allocate a");
  ready = ready && test.cuda_ok(
                       b.allocate(kTokenCount *
                                  q3x::runtime::kGdnValueHeadCount),
                       "GDN register-state perf allocate b");
  ready = ready && test.cuda_ok(
                       A_log.allocate(q3x::runtime::kGdnValueHeadCount),
                       "GDN register-state perf allocate A_log");
  ready = ready && test.cuda_ok(
                       dt_bias.allocate(q3x::runtime::kGdnValueHeadCount),
                       "GDN register-state perf allocate dt_bias");
  ready = ready && test.cuda_ok(
                       immutable_states.allocate(kStatePoolElements),
                       "GDN register-state perf allocate immutable banks");
  ready = ready && test.cuda_ok(
                       production_states.allocate(kStatePoolElements),
                       "GDN register-state perf allocate production banks");
  ready = ready && test.cuda_ok(
                       candidate_states.allocate(kStatePoolElements),
                       "GDN register-state perf allocate candidate banks");
  ready = ready && test.cuda_ok(
                       production_outputs.allocate(kOutputPoolElements),
                       "GDN register-state perf allocate production outputs");
  ready = ready && test.cuda_ok(
                       candidate_outputs.allocate(kOutputPoolElements),
                       "GDN register-state perf allocate candidate outputs");
  if (!ready) {
    return;
  }

  fill_gdn_tile_inputs(conv_qkv, a, b, A_log, dt_bias);
  const std::vector<std::uint16_t> initial_conv_qkv(
      conv_qkv.data(), conv_qkv.data() + conv_qkv.size());
  const std::vector<std::uint16_t> initial_a(a.data(), a.data() + a.size());
  const std::vector<std::uint16_t> initial_b(b.data(), b.data() + b.size());
  const std::vector<std::uint16_t> initial_A_log(
      A_log.data(), A_log.data() + A_log.size());
  const std::vector<std::uint16_t> initial_dt_bias(
      dt_bias.data(), dt_bias.data() + dt_bias.size());
  std::vector<std::uint16_t> initial_state(
      q3x::runtime::kGdnStateElements);
  for (std::size_t index = 0U; index < initial_state.size(); ++index) {
    const int centered = static_cast<int>((index * 5U) % 23U) - 11;
    initial_state[index] =
        encode_bf16(static_cast<float>(centered) / 512.0F);
  }
  for (std::size_t bank = 0U; bank < kBankCount; ++bank) {
    std::copy(initial_state.begin(), initial_state.end(),
              immutable_states.data() +
                  bank * q3x::runtime::kGdnStateElements);
  }

  const auto reset_variant = [&](ManagedBuffer<std::uint16_t>& states,
                                 ManagedBuffer<std::uint16_t>& outputs,
                                 const std::string& label) -> bool {
    bool reset_ready = test.cuda_ok(
        cudaMemcpyAsync(states.data(), immutable_states.data(),
                        kStatePoolElements * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToDevice, stream),
        label + " reset 24 state banks");
    reset_ready = reset_ready && test.cuda_ok(
                                     cudaMemsetAsync(
                                         outputs.data(), 0xff,
                                         kOutputPoolElements *
                                             sizeof(std::uint16_t),
                                         stream),
                                     label + " poison output banks");
    reset_ready = reset_ready && test.cuda_ok(
                                     cudaStreamSynchronize(stream),
                                     label + " reset synchronize");
    return reset_ready;
  };

  const auto measure_variant =
      [&](const bool candidate, const std::string& label) -> float {
    ManagedBuffer<std::uint16_t>& states =
        candidate ? candidate_states : production_states;
    ManagedBuffer<std::uint16_t>& outputs =
        candidate ? candidate_outputs : production_outputs;
    if (!reset_variant(states, outputs, label)) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    std::size_t launch_index = 0U;
    const float milliseconds = measure_average_cuda_milliseconds(
        test, stream, kWarmupLaunches, kMeasuredLaunches, label, [&]() {
          const std::size_t bank = launch_index % kBankCount;
          ++launch_index;
          std::uint16_t* const state =
              states.data() + bank * q3x::runtime::kGdnStateElements;
          std::uint16_t* const output =
              outputs.data() + bank * kOutputElementsPerBank;
          if (candidate) {
            return q3x::runtime::
                launch_gated_delta_net_update_tile_warp_parallel_cuda(
                    conv_qkv.data(), kTokenCount, a.data(), b.data(),
                    A_log.data(), dt_bias.data(), state, state, kL2Epsilon,
                    output, {}, static_cast<void*>(stream));
          }
          return q3x::runtime::
              launch_gated_delta_net_update_tile_warp_eight_row_lane_striped_test_cuda(
                  conv_qkv.data(), kTokenCount, a.data(), b.data(),
                  A_log.data(), dt_bias.data(), state, state, kL2Epsilon,
                  output, static_cast<void*>(stream));
        });
    test.expect(launch_index == kWarmupLaunches + kMeasuredLaunches,
                label + " rotates every requested state bank launch");
    return milliseconds;
  };

  bool timing_bitwise = true;
  const auto compare_timing_results = [&](const std::string& label) {
    const int failures_before = test.failures();
    expect_bf16_buffer_bitwise_equal(
        test, candidate_states.data(), production_states.data(),
        kStatePoolElements, label + " final 24-bank states");
    expect_bf16_buffer_bitwise_equal(
        test, candidate_outputs.data(), production_outputs.data(),
        kOutputPoolElements, label + " last outputs");
    timing_bitwise =
        timing_bitwise && test.failures() == failures_before;
  };

  std::array<float, kPassesPerVariant> production_passes{};
  std::array<float, kPassesPerVariant> candidate_passes{};
  bool all_rounds_faster = true;
  bool all_finite = true;
  for (std::size_t round = 0U; round < kRoundCount; ++round) {
    const std::string round_label =
        "GDN register-state M16 round=" + std::to_string(round + 1U);
    const float production_first =
        measure_variant(false, round_label + " B1");
    const float candidate_first =
        measure_variant(true, round_label + " C1");
    compare_timing_results(round_label + " B1/C1");
    const float candidate_second =
        measure_variant(true, round_label + " C2");
    const float production_second =
        measure_variant(false, round_label + " B2");
    compare_timing_results(round_label + " C2/B2");

    production_passes[2U * round] = production_first;
    production_passes[2U * round + 1U] = production_second;
    candidate_passes[2U * round] = candidate_first;
    candidate_passes[2U * round + 1U] = candidate_second;
    const bool round_finite =
        std::isfinite(production_first) &&
        std::isfinite(production_second) &&
        std::isfinite(candidate_first) && std::isfinite(candidate_second) &&
        production_first > 0.0F && production_second > 0.0F &&
        candidate_first > 0.0F && candidate_second > 0.0F;
    const float production_mean =
        (production_first + production_second) * 0.5F;
    const float candidate_mean =
        (candidate_first + candidate_second) * 0.5F;
    const bool round_faster = round_finite && candidate_mean < production_mean;
    all_finite = all_finite && round_finite;
    all_rounds_faster = all_rounds_faster && round_faster;
    std::cout << "PERF_GDN_REGISTER_STATE_M16_ROUND: round="
              << round + 1U << " production_pass1_ms=" << production_first
              << " candidate_pass1_ms=" << candidate_first
              << " candidate_pass2_ms=" << candidate_second
              << " production_pass2_ms=" << production_second
              << " production_mean_ms=" << production_mean
              << " candidate_mean_ms=" << candidate_mean
              << " speedup=" << production_mean / candidate_mean
              << " no_reversal=" << (round_faster ? "true" : "false")
              << '\n';
  }

  const int input_failures_before = test.failures();
  expect_bf16_buffer_bitwise_equal(
      test, conv_qkv.data(), initial_conv_qkv.data(), initial_conv_qkv.size(),
      "GDN register-state M16 perf preserves conv QKV");
  expect_bf16_buffer_bitwise_equal(
      test, a.data(), initial_a.data(), initial_a.size(),
      "GDN register-state M16 perf preserves a");
  expect_bf16_buffer_bitwise_equal(
      test, b.data(), initial_b.data(), initial_b.size(),
      "GDN register-state M16 perf preserves b");
  expect_bf16_buffer_bitwise_equal(
      test, A_log.data(), initial_A_log.data(), initial_A_log.size(),
      "GDN register-state M16 perf preserves A_log");
  expect_bf16_buffer_bitwise_equal(
      test, dt_bias.data(), initial_dt_bias.data(), initial_dt_bias.size(),
      "GDN register-state M16 perf preserves dt_bias");
  timing_bitwise =
      timing_bitwise && test.failures() == input_failures_before;

  const auto median = [](std::array<float, kPassesPerVariant> values) {
    std::sort(values.begin(), values.end());
    return (values[kPassesPerVariant / 2U - 1U] +
            values[kPassesPerVariant / 2U]) *
           0.5F;
  };
  const float production_milliseconds = median(production_passes);
  const float candidate_milliseconds = median(candidate_passes);
  const float speedup = production_milliseconds / candidate_milliseconds;
  const bool performance_gate =
      all_finite && timing_bitwise && all_rounds_faster &&
      std::isfinite(speedup) && speedup >= kRequiredSpeedup;
  std::cout << "PERF_GDN_REGISTER_STATE_M16_AGGREGATE: production_ms="
            << production_milliseconds
            << " candidate_ms=" << candidate_milliseconds
            << " speedup=" << speedup
            << " required_speedup=" << kRequiredSpeedup
            << " state_banks=" << kBankCount
            << " state_working_set_bytes="
            << kStatePoolElements * sizeof(std::uint16_t)
            << " warmup_launches_per_pass=" << kWarmupLaunches
            << " measured_launches_per_pass=" << kMeasuredLaunches
            << " variant_samples=" << kPassesPerVariant
            << " bitwise=" << (timing_bitwise ? "true" : "false")
            << " all_rounds_faster="
            << (all_rounds_faster ? "true" : "false")
            << " gate=" << (performance_gate ? "PASS" : "FAIL") << '\n';
  test.expect(performance_gate,
              "GDN register-state M16 clears 1.20x cold-bank gate");
}

// This mirrored harness covers three successive promotions: row-pair versus
// four-row, four-row versus lane-striped, and four-row lane-striped versus the
// production eight-row lane-striped path.
struct GdnRowPipelineMeasurement {
  float production_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_correct = false;
};

enum class GdnRowPipelineCandidate : unsigned int {
  kFourRow,
  kLaneStriped,
  kEightRowLaneStriped,
};

void run_optional_gdn_row_pipeline_performance(
    TestContext& test, cudaStream_t stream,
    const GdnRowPipelineCandidate candidate_kind) {
  const bool lane_striped =
      candidate_kind != GdnRowPipelineCandidate::kFourRow;
  const bool eight_row =
      candidate_kind == GdnRowPipelineCandidate::kEightRowLaneStriped;
  const char* const environment_name =
      eight_row ? "Q3X_RUN_GDN_EIGHT_ROW_PERF"
                : (lane_striped ? "Q3X_RUN_GDN_LANE_STRIPED_PERF"
                                : "Q3X_RUN_GDN_ROW_QUAD_PERF");
  const char* const candidate_label =
      eight_row ? "GDN eight-row lane-striped"
                : (lane_striped ? "GDN lane-striped" : "GDN four-row");
  const char* const metric_label =
      eight_row ? "GDN_EIGHT_ROW"
                : (lane_striped ? "GDN_LANE_STRIPED" : "GDN_ROW_QUAD");
  const char* const value = std::getenv(environment_name);
  const bool enabled = value != nullptr && value[0] != '\0' &&
                       !(value[0] == '0' && value[1] == '\0');
  if (!enabled) {
    std::cout << "SKIP: " << candidate_label << " performance segment; set "
              << environment_name << "=1 to enable\n";
    return;
  }

  constexpr std::size_t kMaximumTokens =
      q3x::runtime::kGdnMaximumTileTokenCount;
  constexpr std::size_t kWarmupIterations = 10U;
  constexpr std::size_t kMeasuredIterations = 24U;
  constexpr int kMeasurementRounds = 3;
  constexpr float kL2Epsilon = 1.0e-6F;
  struct Shape {
    std::size_t token_count;
    std::size_t calls_per_cluster;
    float minimum_speedup;
    const char* label;
  };
  constexpr std::array<Shape, 4U> kShapes{{
      {1U, 96U, 0.99F, "GDN four-row M1"},
      {2U, 48U, 0.99F, "GDN four-row M2"},
      {8U, 0U, 1.02F, "GDN four-row M8"},
      {16U, 48U, 1.03F, "GDN four-row M16"},
  }};
  constexpr std::array<const char*, 4U> kLaneStripedLabels{{
      "GDN lane-striped M1",
      "GDN lane-striped M2",
      "GDN lane-striped M8",
      "GDN lane-striped M16",
  }};
  constexpr std::array<const char*, 4U> kEightRowLabels{{
      "GDN eight-row M1",
      "GDN eight-row M2",
      "GDN eight-row M8",
      "GDN eight-row M16",
  }};

  ManagedBuffer<std::uint16_t> conv_qkv;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint16_t> A_log;
  ManagedBuffer<std::uint16_t> dt_bias;
  ManagedBuffer<std::uint16_t> timing_initial_state;
  ManagedBuffer<std::uint16_t> production_inplace_state;
  ManagedBuffer<std::uint16_t> candidate_inplace_state;
  ManagedBuffer<std::uint16_t> production_disjoint_input_state;
  ManagedBuffer<std::uint16_t> candidate_disjoint_input_state;
  ManagedBuffer<std::uint16_t> production_disjoint_output_state;
  ManagedBuffer<std::uint16_t> candidate_disjoint_output_state;
  ManagedBuffer<std::uint16_t> production_inplace_output;
  ManagedBuffer<std::uint16_t> candidate_inplace_output;
  ManagedBuffer<std::uint16_t> production_disjoint_output;
  ManagedBuffer<std::uint16_t> candidate_disjoint_output;
  bool ready = test.cuda_ok(
      conv_qkv.allocate(kMaximumTokens * q3x::runtime::kGdnQkvChannels),
      std::string(candidate_label) + " allocate conv QKV");
  ready = ready && test.cuda_ok(
                       a.allocate(kMaximumTokens *
                                  q3x::runtime::kGdnValueHeadCount),
                       std::string(candidate_label) + " allocate a");
  ready = ready && test.cuda_ok(
                       b.allocate(kMaximumTokens *
                                  q3x::runtime::kGdnValueHeadCount),
                       std::string(candidate_label) + " allocate b");
  ready = ready && test.cuda_ok(
                       A_log.allocate(q3x::runtime::kGdnValueHeadCount),
                                std::string(candidate_label) +
                                    " allocate A_log");
  ready = ready && test.cuda_ok(
                       dt_bias.allocate(q3x::runtime::kGdnValueHeadCount),
                                std::string(candidate_label) +
                                    " allocate dt_bias");
  ready = ready && test.cuda_ok(
                       timing_initial_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       std::string(candidate_label) +
                           " allocate immutable timing state");
  ready = ready && test.cuda_ok(
                       production_inplace_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       std::string(candidate_label) +
                           " allocate production in-place state");
  ready = ready && test.cuda_ok(
                       candidate_inplace_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       std::string(candidate_label) +
                           " allocate candidate in-place state");
  ready = ready && test.cuda_ok(
                       production_disjoint_input_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       std::string(candidate_label) +
                           " allocate production disjoint input");
  ready = ready && test.cuda_ok(
                       candidate_disjoint_input_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       std::string(candidate_label) +
                           " allocate candidate disjoint input");
  ready = ready && test.cuda_ok(
                       production_disjoint_output_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       std::string(candidate_label) +
                           " allocate production disjoint state");
  ready = ready && test.cuda_ok(
                       candidate_disjoint_output_state.allocate(
                           q3x::runtime::kGdnStateElements),
                       std::string(candidate_label) +
                           " allocate candidate disjoint state");
  ready = ready && test.cuda_ok(
                       production_inplace_output.allocate(
                           kMaximumTokens * q3x::runtime::kGdnVElements),
                       std::string(candidate_label) +
                           " allocate production in-place output");
  ready = ready && test.cuda_ok(
                       candidate_inplace_output.allocate(
                           kMaximumTokens * q3x::runtime::kGdnVElements),
                       std::string(candidate_label) +
                           " allocate candidate in-place output");
  ready = ready && test.cuda_ok(
                       production_disjoint_output.allocate(
                           kMaximumTokens * q3x::runtime::kGdnVElements),
                       std::string(candidate_label) +
                           " allocate production disjoint output");
  ready = ready && test.cuda_ok(
                       candidate_disjoint_output.allocate(
                           kMaximumTokens * q3x::runtime::kGdnVElements),
                       std::string(candidate_label) +
                           " allocate candidate disjoint output");
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
  std::copy(initial_state.begin(), initial_state.end(),
            timing_initial_state.data());
  const std::vector<std::uint16_t> initial_conv_qkv(
      conv_qkv.data(), conv_qkv.data() + conv_qkv.size());
  const std::vector<std::uint16_t> initial_a(a.data(), a.data() + a.size());
  const std::vector<std::uint16_t> initial_b(b.data(), b.data() + b.size());
  const std::vector<std::uint16_t> initial_A_log(
      A_log.data(), A_log.data() + A_log.size());
  const std::vector<std::uint16_t> initial_dt_bias(
      dt_bias.data(), dt_bias.data() + dt_bias.size());

  const auto expect_equal = [&](const std::uint16_t* const actual,
                                const std::uint16_t* const expected,
                                const std::size_t count,
                                const std::string& label) -> bool {
    const bool equal = std::equal(actual, actual + count, expected);
    expect_bf16_buffer_bitwise_equal(test, actual, expected, count, label);
    return equal;
  };
  const auto inputs_preserved = [&](const std::string& label) -> bool {
    bool preserved = expect_equal(
        conv_qkv.data(), initial_conv_qkv.data(), initial_conv_qkv.size(),
        label + " preserves conv QKV");
    preserved = expect_equal(a.data(), initial_a.data(), initial_a.size(),
                             label + " preserves a") &&
                preserved;
    preserved = expect_equal(b.data(), initial_b.data(), initial_b.size(),
                             label + " preserves b") &&
                preserved;
    preserved = expect_equal(A_log.data(), initial_A_log.data(),
                             initial_A_log.size(),
                             label + " preserves A_log") &&
                preserved;
    preserved = expect_equal(dt_bias.data(), initial_dt_bias.data(),
                             initial_dt_bias.size(),
                             label + " preserves dt_bias") &&
                preserved;
    return preserved;
  };
  const auto launch_production_direct =
      [&](const std::size_t token_count,
          const std::uint16_t* const state_input,
          std::uint16_t* const state_output,
          std::uint16_t* const output) -> int {
    if (eight_row) {
      return q3x::runtime::
          launch_gated_delta_net_update_tile_warp_four_row_lane_striped_test_cuda(
              conv_qkv.data(), token_count, a.data(), b.data(), A_log.data(),
              dt_bias.data(), state_input, state_output, kL2Epsilon, output,
              static_cast<void*>(stream));
    }
    if (lane_striped) {
      return q3x::runtime::
          launch_gated_delta_net_update_tile_warp_four_row_test_cuda(
              conv_qkv.data(), token_count, a.data(), b.data(), A_log.data(),
              dt_bias.data(), state_input, state_output, kL2Epsilon, output,
              static_cast<void*>(stream));
    }
    return q3x::runtime::
        launch_gated_delta_net_update_tile_warp_row_pair_test_cuda(
            conv_qkv.data(), token_count, a.data(), b.data(), A_log.data(),
            dt_bias.data(), state_input, state_output, kL2Epsilon, output,
            static_cast<void*>(stream));
  };
  const auto launch_candidate_direct =
      [&](const std::size_t token_count,
          const std::uint16_t* const state_input,
          std::uint16_t* const state_output,
          std::uint16_t* const output) -> int {
    if (eight_row) {
      return q3x::runtime::
          launch_gated_delta_net_update_tile_warp_eight_row_lane_striped_test_cuda(
              conv_qkv.data(), token_count, a.data(), b.data(), A_log.data(),
              dt_bias.data(), state_input, state_output, kL2Epsilon, output,
              static_cast<void*>(stream));
    }
    if (lane_striped) {
      return q3x::runtime::
          launch_gated_delta_net_update_tile_warp_four_row_lane_striped_test_cuda(
              conv_qkv.data(), token_count, a.data(), b.data(), A_log.data(),
              dt_bias.data(), state_input, state_output, kL2Epsilon, output,
              static_cast<void*>(stream));
    }
    return q3x::runtime::
        launch_gated_delta_net_update_tile_warp_four_row_test_cuda(
            conv_qkv.data(), token_count, a.data(), b.data(), A_log.data(),
            dt_bias.data(), state_input, state_output, kL2Epsilon, output,
            static_cast<void*>(stream));
  };
  const auto launch_production = [&](const std::size_t token_count) -> int {
    return launch_production_direct(
        token_count, production_inplace_state.data(),
        production_inplace_state.data(), production_inplace_output.data());
  };
  const auto launch_candidate = [&](const std::size_t token_count) -> int {
    return launch_candidate_direct(
        token_count, candidate_inplace_state.data(),
        candidate_inplace_state.data(), candidate_inplace_output.data());
  };
  const auto reset_timing_state = [&](ManagedBuffer<std::uint16_t>& state,
                                      ManagedBuffer<std::uint16_t>& output,
                                      const std::string& label) -> bool {
    bool reset_ready = test.cuda_ok(
        cudaMemcpyAsync(state.data(), timing_initial_state.data(),
                        initial_state.size() * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToDevice, stream),
        label + " reset state");
    reset_ready = reset_ready && test.cuda_ok(
                                     cudaMemsetAsync(
                                         output.data(), 0,
                                         output.size() * sizeof(std::uint16_t),
                                         stream),
                                     label + " reset output");
    reset_ready = reset_ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                               label + " reset synchronize");
    return reset_ready;
  };

  std::array<GdnRowPipelineMeasurement, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    Shape shape = kShapes[shape_index];
    if (eight_row) {
      shape.label = kEightRowLabels[shape_index];
    } else if (lane_striped) {
      shape.label = kLaneStripedLabels[shape_index];
    }
    const std::size_t output_count =
        shape.token_count * q3x::runtime::kGdnVElements;
    std::copy(initial_state.begin(), initial_state.end(),
              production_inplace_state.data());
    std::copy(initial_state.begin(), initial_state.end(),
              candidate_inplace_state.data());
    std::fill_n(production_inplace_output.data(),
                production_inplace_output.size(),
                static_cast<std::uint16_t>(0x5a5aU));
    std::fill_n(candidate_inplace_output.data(),
                candidate_inplace_output.size(),
                static_cast<std::uint16_t>(0xa5a5U));
    ready = test.cuda_ok(
        static_cast<cudaError_t>(launch_production(shape.token_count)),
        std::string(shape.label) + " production in-place correctness");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(
                             launch_candidate(shape.token_count)),
                         std::string(shape.label) +
                             " candidate in-place correctness");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         std::string(shape.label) +
                             " in-place correctness synchronize");
    if (!ready) {
      return;
    }
    bool bitwise_correct = expect_equal(
        candidate_inplace_output.data(), production_inplace_output.data(),
        output_count, std::string(shape.label) + " in-place output");
    bitwise_correct =
        expect_equal(candidate_inplace_state.data(),
                     production_inplace_state.data(),
                     q3x::runtime::kGdnStateElements,
                     std::string(shape.label) + " in-place state") &&
        bitwise_correct;

    std::copy(initial_state.begin(), initial_state.end(),
              production_disjoint_input_state.data());
    std::copy(initial_state.begin(), initial_state.end(),
              candidate_disjoint_input_state.data());
    std::fill_n(production_disjoint_output_state.data(),
                production_disjoint_output_state.size(),
                static_cast<std::uint16_t>(0x5a5aU));
    std::fill_n(candidate_disjoint_output_state.data(),
                candidate_disjoint_output_state.size(),
                static_cast<std::uint16_t>(0xa5a5U));
    std::fill_n(production_disjoint_output.data(),
                production_disjoint_output.size(),
                static_cast<std::uint16_t>(0x5a5aU));
    std::fill_n(candidate_disjoint_output.data(),
                candidate_disjoint_output.size(),
                static_cast<std::uint16_t>(0xa5a5U));
    ready = test.cuda_ok(
        static_cast<cudaError_t>(launch_production_direct(
            shape.token_count, production_disjoint_input_state.data(),
            production_disjoint_output_state.data(),
            production_disjoint_output.data())),
        std::string(shape.label) + " production disjoint correctness");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate_direct(
                             shape.token_count,
                             candidate_disjoint_input_state.data(),
                             candidate_disjoint_output_state.data(),
                             candidate_disjoint_output.data())),
                         std::string(shape.label) +
                             " candidate disjoint correctness");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         std::string(shape.label) +
                             " disjoint correctness synchronize");
    if (!ready) {
      return;
    }
    bitwise_correct =
        expect_equal(candidate_disjoint_output.data(),
                     production_disjoint_output.data(), output_count,
                     std::string(shape.label) + " disjoint output") &&
        bitwise_correct;
    bitwise_correct =
        expect_equal(candidate_disjoint_output_state.data(),
                     production_disjoint_output_state.data(),
                     q3x::runtime::kGdnStateElements,
                     std::string(shape.label) + " disjoint state") &&
        bitwise_correct;
    bitwise_correct =
        expect_equal(production_disjoint_input_state.data(),
                     initial_state.data(), initial_state.size(),
                     std::string(shape.label) +
                         " production preserves disjoint input") &&
        bitwise_correct;
    bitwise_correct =
        expect_equal(candidate_disjoint_input_state.data(),
                     initial_state.data(), initial_state.size(),
                     std::string(shape.label) +
                         " candidate preserves disjoint input") &&
        bitwise_correct;
    bitwise_correct =
        expect_equal(production_disjoint_output.data(),
                     production_inplace_output.data(), output_count,
                     std::string(shape.label) +
                         " production alias-mode output") &&
        bitwise_correct;
    bitwise_correct =
        expect_equal(candidate_disjoint_output.data(),
                     candidate_inplace_output.data(), output_count,
                     std::string(shape.label) +
                         " candidate alias-mode output") &&
        bitwise_correct;
    bitwise_correct =
        expect_equal(production_disjoint_output_state.data(),
                     production_inplace_state.data(),
                     q3x::runtime::kGdnStateElements,
                     std::string(shape.label) +
                         " production alias-mode state") &&
        bitwise_correct;
    bitwise_correct =
        expect_equal(candidate_disjoint_output_state.data(),
                     candidate_inplace_state.data(),
                     q3x::runtime::kGdnStateElements,
                     std::string(shape.label) + " candidate alias-mode state") &&
        bitwise_correct;
    bitwise_correct =
        inputs_preserved(std::string(shape.label) + " correctness") &&
        bitwise_correct;
    measurements[shape_index].bitwise_correct = bitwise_correct;
    std::cout << metric_label << "_DIFF: " << shape.label
              << " inplace_disjoint_inputs_preserved="
              << (bitwise_correct ? "true" : "false") << '\n';

    ready = reset_timing_state(production_inplace_state,
                               production_inplace_output,
                               std::string(shape.label) +
                                   " production warmup");
    ready = ready && reset_timing_state(candidate_inplace_state,
                                        candidate_inplace_output,
                                        std::string(shape.label) +
                                            " candidate warmup");
    for (std::size_t iteration = 0U;
         iteration < kWarmupIterations && ready; ++iteration) {
      ready = test.cuda_ok(
          static_cast<cudaError_t>(launch_production(shape.token_count)),
          std::string(shape.label) + " production warmup launch");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(
                               launch_candidate(shape.token_count)),
                           std::string(shape.label) +
                               " candidate warmup launch");
    }
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  std::string(shape.label) +
                                      " warmup synchronize");
    if (!ready) {
      return;
    }

    double production_total = 0.0;
    double candidate_total = 0.0;
    bool finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          std::string(shape.label) + " round=" + std::to_string(round + 1);
      ready = reset_timing_state(production_inplace_state,
                                 production_inplace_output,
                                 round_label + " production pass 1");
      const float production_first =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, 0U, kMeasuredIterations,
                      round_label + " production pass 1", [&]() {
                        return launch_production(shape.token_count);
                      })
                : std::numeric_limits<float>::quiet_NaN();
      ready = ready && reset_timing_state(candidate_inplace_state,
                                          candidate_inplace_output,
                                          round_label + " candidate pass 1");
      const float candidate_first =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, 0U, kMeasuredIterations,
                      round_label + " candidate pass 1", [&]() {
                        return launch_candidate(shape.token_count);
                      })
                : std::numeric_limits<float>::quiet_NaN();
      ready = ready && reset_timing_state(candidate_inplace_state,
                                          candidate_inplace_output,
                                          round_label + " candidate pass 2");
      const float candidate_second =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, 0U, kMeasuredIterations,
                      round_label + " candidate pass 2", [&]() {
                        return launch_candidate(shape.token_count);
                      })
                : std::numeric_limits<float>::quiet_NaN();
      ready = ready && reset_timing_state(production_inplace_state,
                                          production_inplace_output,
                                          round_label + " production pass 2");
      const float production_second =
          ready ? measure_average_cuda_milliseconds(
                      test, stream, 0U, kMeasuredIterations,
                      round_label + " production pass 2", [&]() {
                        return launch_production(shape.token_count);
                      })
                : std::numeric_limits<float>::quiet_NaN();
      const bool round_finite =
          ready && std::isfinite(production_first) &&
          std::isfinite(candidate_first) && std::isfinite(candidate_second) &&
          std::isfinite(production_second);
      finite = finite && round_finite;
      if (round_finite) {
        production_total += production_first + production_second;
        candidate_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_" << metric_label << "_ROUND: " << shape.label
                << " round=" << round + 1
                << " production_pass1_ms=" << production_first
                << " candidate_pass1_ms=" << candidate_first
                << " candidate_pass2_ms=" << candidate_second
                << " production_pass2_ms=" << production_second << '\n';
    }
    constexpr double kTimedPasses =
        2.0 * static_cast<double>(kMeasurementRounds);
    measurements[shape_index].production_milliseconds =
        finite ? static_cast<float>(production_total / kTimedPasses)
               : std::numeric_limits<float>::quiet_NaN();
    measurements[shape_index].candidate_milliseconds =
        finite ? static_cast<float>(candidate_total / kTimedPasses)
               : std::numeric_limits<float>::quiet_NaN();
  }

  const bool final_inputs_preserved =
      inputs_preserved(std::string(candidate_label) + " final");
  double weighted_production = 0.0;
  double weighted_candidate = 0.0;
  bool all_shape_gates = final_inputs_preserved;
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    Shape shape = kShapes[shape_index];
    if (eight_row) {
      shape.label = kEightRowLabels[shape_index];
    } else if (lane_striped) {
      shape.label = kLaneStripedLabels[shape_index];
    }
    const GdnRowPipelineMeasurement& measurement = measurements[shape_index];
    const float speedup = measurement.production_milliseconds /
                          measurement.candidate_milliseconds;
    const bool finite = std::isfinite(measurement.production_milliseconds) &&
                        std::isfinite(measurement.candidate_milliseconds) &&
                        std::isfinite(speedup) &&
                        measurement.production_milliseconds > 0.0F &&
                        measurement.candidate_milliseconds > 0.0F;
    constexpr float kMinimumLaneStripedShapeSpeedup = 1.40F;
    constexpr float kMinimumEightRowShapeSpeedup = 1.03F;
    const float required_speedup =
        eight_row ? kMinimumEightRowShapeSpeedup
                  : (lane_striped ? kMinimumLaneStripedShapeSpeedup
                                  : shape.minimum_speedup);
    const bool gate = measurement.bitwise_correct && finite &&
                      speedup >= required_speedup;
    all_shape_gates = all_shape_gates && gate;
    test.expect(gate, std::string(shape.label) +
                          " clears correctness/performance gate");
    std::cout << "PERF_" << metric_label << "_VALIDATION: " << shape.label
              << " production_ms=" << measurement.production_milliseconds
              << " candidate_ms=" << measurement.candidate_milliseconds
              << " speedup=" << speedup
              << " required_speedup=" << required_speedup
              << " bitwise="
              << (measurement.bitwise_correct ? "true" : "false")
              << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
    constexpr std::array<std::size_t, 4U> kCurrentProfileCalls{{
        1'248U, 48U, 0U, 48U,
    }};
    const std::size_t calls =
        eight_row ? kCurrentProfileCalls[shape_index]
                  : shape.calls_per_cluster;
    weighted_production +=
        static_cast<double>(calls) * measurement.production_milliseconds;
    weighted_candidate +=
        static_cast<double>(calls) * measurement.candidate_milliseconds;
  }
  constexpr double kMinimumAggregateSpeedup = 1.03;
  constexpr double kMinimumLaneStripedAggregateSpeedup = 1.50;
  constexpr double kMinimumEightRowAggregateSpeedup = 1.20;
  const double required_aggregate_speedup =
      eight_row ? kMinimumEightRowAggregateSpeedup
                : (lane_striped ? kMinimumLaneStripedAggregateSpeedup
                                : kMinimumAggregateSpeedup);
  const double weighted_speedup = weighted_production / weighted_candidate;
  const bool aggregate_gate =
      all_shape_gates && std::isfinite(weighted_speedup) &&
      weighted_production > 0.0 && weighted_candidate > 0.0 &&
      weighted_speedup >= required_aggregate_speedup;
  std::cout << "PERF_" << metric_label
            << "_AGGREGATE: weighted_production_ms="
            << weighted_production
            << " weighted_candidate_ms=" << weighted_candidate
            << " speedup=" << weighted_speedup
            << " required_speedup=" << required_aggregate_speedup
            << " call_weights="
            << (eight_row ? "1248:48:0:48" : "96:48:0:48")
            << " all_shapes="
            << (all_shape_gates ? "PASS" : "FAIL")
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate, std::string(candidate_label) +
                                  " clears weighted aggregate gate");
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
  test_gdn_plain_norm_silu_gate_production_identity(test, stream);
  test_gdn_tile_bitwise(test, stream);
  test_gdn_register_state_m16_candidate(test, stream);
  run_optional_gdn_register_state_m16_performance(test, stream);
  run_optional_gdn_row_pipeline_performance(
      test, stream, GdnRowPipelineCandidate::kFourRow);
  run_optional_gdn_row_pipeline_performance(
      test, stream, GdnRowPipelineCandidate::kLaneStriped);
  run_optional_gdn_row_pipeline_performance(
      test, stream, GdnRowPipelineCandidate::kEightRowLaneStriped);
  (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy GDN stream");
  if (test.failures() != 0) {
    std::cerr << test.failures() << " CUDA GDN assertion(s) failed\n";
    return 1;
  }
  std::cout << "CUDA GDN reference tests passed\n";
  return 0;
}
