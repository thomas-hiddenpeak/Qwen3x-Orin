#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kQueryHeads = 24U;
constexpr std::size_t kKvHeads = 4U;
constexpr std::size_t kHeadDimension = 256U;
constexpr std::size_t kQueryElementsPerToken =
    kQueryHeads * kHeadDimension;
constexpr std::size_t kKvElementsPerToken =
    kKvHeads * kHeadDimension;
constexpr std::size_t kGuardElements = 32U;
constexpr std::uint16_t kGuard = 0x7fc1U;

static_assert(
    q3x::runtime::bulk_causal_gqa_group_q128_v4_grid_x(512U) == 24U);
static_assert(
    q3x::runtime::bulk_causal_gqa_group_q128_v4_grid_x(1'024U) == 48U);
static_assert(
    q3x::runtime::bulk_causal_gqa_group_q128_v4_grid_x(7'712U) == 362U);
static_assert(
    q3x::runtime::bulk_causal_gqa_group_q128_v4_grid_x(8'192U) == 384U);
static_assert(
    q3x::runtime::can_launch_bulk_causal_gqa_group_q128_v4_panel(
        32'288U, 7'712U));
static_assert(
    !q3x::runtime::can_launch_bulk_causal_gqa_group_q128_v4_panel(
        262'143U, 2U));

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
  ~ManagedBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }
  ManagedBuffer(const ManagedBuffer&) = delete;
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  [[nodiscard]] cudaError_t allocate(const std::size_t elements) {
    size_ = elements;
    return cudaMallocManaged(reinterpret_cast<void**>(&data_),
                             elements * sizeof(T));
  }
  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0U;
};

class ScopedHostileEnvironment final {
 public:
  ScopedHostileEnvironment() {
    const char* const previous =
        std::getenv("Q3X_FULL_ATTENTION_FLASHINFER_DIRECT");
    if (previous != nullptr) {
      previous_ = previous;
      had_previous_ = true;
    }
    (void)setenv("Q3X_FULL_ATTENTION_FLASHINFER_DIRECT", "1", 1);
  }
  ~ScopedHostileEnvironment() {
    if (had_previous_) {
      (void)setenv("Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
                   previous_.c_str(), 1);
    } else {
      (void)unsetenv("Q3X_FULL_ATTENTION_FLASHINFER_DIRECT");
    }
  }
  ScopedHostileEnvironment(const ScopedHostileEnvironment&) = delete;
  ScopedHostileEnvironment& operator=(
      const ScopedHostileEnvironment&) = delete;

 private:
  std::string previous_;
  bool had_previous_ = false;
};

[[nodiscard]] bool guards_intact(
    const ManagedBuffer<std::uint16_t>& storage,
    const std::size_t payload_elements) {
  return std::all_of(storage.data(),
                     storage.data() + kGuardElements,
                     [](const std::uint16_t bits) {
                       return bits == kGuard;
                     }) &&
         std::all_of(storage.data() + kGuardElements + payload_elements,
                     storage.data() + storage.size(),
                     [](const std::uint16_t bits) {
                       return bits == kGuard;
                     });
}

void run_v4_equivalence(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kFirstPosition = 512U;
  constexpr std::size_t kTokenCount = 512U;
  constexpr std::size_t kQueryElements =
      kTokenCount * kQueryElementsPerToken;
  constexpr std::size_t kCacheElements =
      (kFirstPosition + kTokenCount) * kKvElementsPerToken;

  ManagedBuffer<std::uint16_t> query_storage;
  ManagedBuffer<std::uint16_t> gate_storage;
  ManagedBuffer<std::uint16_t> key_storage;
  ManagedBuffer<std::uint16_t> value_storage;
  ManagedBuffer<std::uint16_t> baseline_storage;
  ManagedBuffer<std::uint16_t> candidate_storage;
  ManagedBuffer<std::uint16_t> replay_storage;
  bool ready = test.cuda_ok(
      query_storage.allocate(kQueryElements + 2U * kGuardElements),
      "allocate query");
  ready = test.cuda_ok(
              gate_storage.allocate(kQueryElements + 2U * kGuardElements),
              "allocate gate") &&
          ready;
  ready = test.cuda_ok(
              key_storage.allocate(kCacheElements + 2U * kGuardElements),
              "allocate key") &&
          ready;
  ready = test.cuda_ok(
              value_storage.allocate(kCacheElements + 2U * kGuardElements),
              "allocate value") &&
          ready;
  ready = test.cuda_ok(
              baseline_storage.allocate(kQueryElements +
                                        2U * kGuardElements),
              "allocate baseline output") &&
          ready;
  ready = test.cuda_ok(
              candidate_storage.allocate(kQueryElements +
                                         2U * kGuardElements),
              "allocate candidate output") &&
          ready;
  ready = test.cuda_ok(
              replay_storage.allocate(kQueryElements +
                                      2U * kGuardElements),
              "allocate replay output") &&
          ready;
  if (!ready) {
    return;
  }

  auto* const query = query_storage.data() + kGuardElements;
  auto* const gate = gate_storage.data() + kGuardElements;
  auto* const key = key_storage.data() + kGuardElements;
  auto* const value = value_storage.data() + kGuardElements;
  auto* const baseline = baseline_storage.data() + kGuardElements;
  auto* const candidate = candidate_storage.data() + kGuardElements;
  auto* const replay = replay_storage.data() + kGuardElements;
  std::fill_n(query_storage.data(), query_storage.size(), kGuard);
  std::fill_n(gate_storage.data(), gate_storage.size(), kGuard);
  std::fill_n(key_storage.data(), key_storage.size(), kGuard);
  std::fill_n(value_storage.data(), value_storage.size(), kGuard);
  std::fill_n(baseline_storage.data(), baseline_storage.size(), kGuard);
  std::fill_n(candidate_storage.data(), candidate_storage.size(), kGuard);
  std::fill_n(replay_storage.data(), replay_storage.size(), kGuard);
  for (std::size_t index = 0U; index < kQueryElements; ++index) {
    query[index] = static_cast<std::uint16_t>(
        0x3f00U + ((index * 17U + 5U) % 127U));
    gate[index] = static_cast<std::uint16_t>(
        0x3e80U + ((index * 23U + 3U) % 97U));
  }
  for (std::size_t index = 0U; index < kCacheElements; ++index) {
    key[index] = static_cast<std::uint16_t>(
        0x3f00U + ((index * 29U + 7U) % 113U));
    value[index] = static_cast<std::uint16_t>(
        0x3e00U + ((index * 31U + 11U) % 109U));
  }
  // P512 makes the Q64 subgroup ending at local token 31 stop before the
  // global K/V32 tile beginning at position 544, while the adjacent subgroup
  // consumes that tile.  A masked-future NaN here detects accidental Q128
  // execution of the first subgroup's extra aggregate-CTA iteration.
  value[(kFirstPosition + 32U) * kKvElementsPerToken] = 0x7fc1U;
  const std::vector<std::uint16_t> query_snapshot(
      query_storage.data(), query_storage.data() + query_storage.size());
  const std::vector<std::uint16_t> gate_snapshot(
      gate_storage.data(), gate_storage.data() + gate_storage.size());
  const std::vector<std::uint16_t> key_snapshot(
      key_storage.data(), key_storage.data() + key_storage.size());
  const std::vector<std::uint16_t> value_snapshot(
      value_storage.data(), value_storage.data() + value_storage.size());

  test.expect(
      q3x::runtime::
          launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q128_v4_panel_fixed_cuda(
              query + 1U, key, value, gate, kFirstPosition, kTokenCount,
              candidate, stream) == static_cast<int>(cudaErrorInvalidValue),
      "v4 entry rejects a misaligned vector operand");
  test.expect(
      q3x::runtime::
          launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q128_v4_panel_fixed_cuda(
              query, key, value, gate, kFirstPosition, kTokenCount, query,
              stream) == static_cast<int>(cudaErrorInvalidValue),
      "v4 entry rejects overlapping input and output ranges");

  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
              query, key, value, gate, kFirstPosition, kTokenCount,
              baseline, stream)),
      "Q64 v3 baseline launch");
  ready = test.cuda_ok(
              static_cast<cudaError_t>(q3x::runtime::
                  launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q128_v4_panel_fixed_cuda(
                      query, key, value, gate, kFirstPosition, kTokenCount,
                      candidate, stream)),
              "Q128 v4 launch") &&
          ready;
  {
    const ScopedHostileEnvironment hostile_environment;
    ready = test.cuda_ok(
                static_cast<cudaError_t>(q3x::runtime::
                    launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q128_v4_panel_fixed_cuda(
                        query, key, value, gate, kFirstPosition,
                        kTokenCount, replay, stream)),
                "hostile-environment v4 replay") &&
            ready;
  }
  ready = test.cuda_ok(cudaStreamSynchronize(stream), "synchronize") &&
          ready;
  if (!ready) {
    return;
  }

  test.expect(std::memcmp(candidate, baseline,
                          kQueryElements * sizeof(std::uint16_t)) == 0,
              "Q128 v4 is bitwise equal to Q64 v3 for C512/P512");
  test.expect(std::memcmp(candidate, replay,
                          kQueryElements * sizeof(std::uint16_t)) == 0,
              "explicit v4 entry is deterministic and ignores hostile "
              "FlashInfer selector state");

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "begin graph capture");
  if (ready) {
    ready = test.cuda_ok(
                static_cast<cudaError_t>(q3x::runtime::
                    launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q128_v4_panel_fixed_cuda(
                        query, key, value, gate, kFirstPosition,
                        kTokenCount, replay, stream)),
                "capture v4 launch") &&
            ready;
  }
  ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                       "end graph capture") &&
          ready;
  if (ready && graph != nullptr) {
    std::size_t node_count = 0U;
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                         "count graph nodes") &&
            ready;
    std::vector<cudaGraphNode_t> nodes(node_count);
    if (ready && node_count != 0U) {
      ready = test.cuda_ok(
                  cudaGraphGetNodes(graph, nodes.data(), &node_count),
                  "read graph nodes") &&
              ready;
    }
    test.expect(node_count == 1U,
                "v4 entry captures exactly one native kernel");
    if (ready && node_count == 1U) {
      cudaKernelNodeParams parameters{};
      ready = test.cuda_ok(cudaGraphKernelNodeGetParams(nodes[0U],
                                                        &parameters),
                           "read v4 kernel parameters") &&
              ready;
      test.expect(
          parameters.gridDim.x == 24U &&
              parameters.gridDim.y == 1U &&
              parameters.gridDim.z == 4U &&
              parameters.blockDim.x ==
                  q3x::runtime::kBulkCausalGqaGroupQ128V4Threads &&
              parameters.blockDim.y == 1U &&
              parameters.blockDim.z == 1U &&
              parameters.sharedMemBytes ==
                  q3x::runtime::
                      kBulkCausalGqaGroupQ128V4DynamicSharedBytes,
          "C512 v4 has packed-Q128/KV-head native topology");
      cudaFuncAttributes attributes{};
      int active_blocks = 0;
      ready = test.cuda_ok(cudaFuncGetAttributes(&attributes,
                                                 parameters.func),
                           "read v4 kernel resources") &&
              ready;
      ready = test.cuda_ok(
                  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                      &active_blocks, parameters.func,
                      static_cast<int>(parameters.blockDim.x),
                      parameters.sharedMemBytes),
                  "read v4 occupancy") &&
              ready;
      test.expect(attributes.localSizeBytes == 0U &&
                      attributes.numRegs <= 255 && active_blocks == 1,
                  "v4 kernel retains zero local bytes and exactly 1 CTA/SM");
      std::cout << "GROUP_Q128_V4_RESOURCES registers="
                << attributes.numRegs
                << " local_bytes=" << attributes.localSizeBytes
                << " dynamic_shared_bytes=" << parameters.sharedMemBytes
                << " active_blocks_per_sm=" << active_blocks << '\n';
    }
  }
  if (ready && graph != nullptr) {
    ready = test.cuda_ok(
                cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr,
                                     0U),
                "instantiate v4 graph") &&
            ready;
  }
  if (ready && graph_exec != nullptr) {
    std::fill_n(replay_storage.data(), replay_storage.size(), kGuard);
    ready = test.cuda_ok(cudaGraphLaunch(graph_exec, stream),
                         "launch v4 graph first replay") &&
            ready;
    ready = test.cuda_ok(cudaGraphLaunch(graph_exec, stream),
                         "launch v4 graph second replay") &&
            ready;
    ready = test.cuda_ok(cudaStreamSynchronize(stream),
                         "synchronize v4 graph replays") &&
            ready;
    if (ready) {
      test.expect(std::memcmp(candidate, replay,
                              kQueryElements * sizeof(std::uint16_t)) == 0,
                  "instantiated v4 graph replays preserve bitwise output");
      test.expect(guards_intact(replay_storage, kQueryElements),
                  "instantiated v4 graph replays preserve guards");
    }
  }
  if (graph_exec != nullptr) {
    (void)test.cuda_ok(cudaGraphExecDestroy(graph_exec),
                       "destroy v4 graph executable");
  }
  if (graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(graph), "destroy graph");
  }

  test.expect(guards_intact(baseline_storage, kQueryElements) &&
                  guards_intact(candidate_storage, kQueryElements) &&
                  guards_intact(replay_storage, kQueryElements),
              "all output guards remain intact");
  test.expect(
      std::memcmp(query_storage.data(), query_snapshot.data(),
                  query_storage.size() * sizeof(std::uint16_t)) == 0 &&
          std::memcmp(gate_storage.data(), gate_snapshot.data(),
                      gate_storage.size() * sizeof(std::uint16_t)) == 0 &&
          std::memcmp(key_storage.data(), key_snapshot.data(),
                      key_storage.size() * sizeof(std::uint16_t)) == 0 &&
          std::memcmp(value_storage.data(), value_snapshot.data(),
                      value_storage.size() * sizeof(std::uint16_t)) == 0,
      "v4 kernel preserves Q/Gate/K/V and their guards");
}

void run_p40k_panel_shape_equivalence(TestContext& test,
                                      cudaStream_t stream) {
  constexpr std::size_t kMaximumTokens = 8'192U;
  constexpr std::size_t kMaximumSequence = 40'000U;
  constexpr std::size_t kQueryElements =
      kMaximumTokens * kQueryElementsPerToken;
  constexpr std::size_t kCacheElements =
      kMaximumSequence * kKvElementsPerToken;
  struct PanelShape {
    std::size_t first_position;
    std::size_t token_count;
  };
  constexpr std::array<PanelShape, 5U> kP40kPanels = {{
      {0U, 8'192U},
      {8'192U, 8'192U},
      {16'384U, 8'192U},
      {24'576U, 7'712U},
      {32'288U, 7'712U},
  }};

  ManagedBuffer<std::uint16_t> query_storage;
  ManagedBuffer<std::uint16_t> gate_storage;
  ManagedBuffer<std::uint16_t> key_storage;
  ManagedBuffer<std::uint16_t> value_storage;
  ManagedBuffer<std::uint16_t> baseline_storage;
  ManagedBuffer<std::uint16_t> candidate_storage;
  bool ready = test.cuda_ok(
      query_storage.allocate(kQueryElements + 2U * kGuardElements),
      "allocate P40K-shape query");
  ready = test.cuda_ok(
              gate_storage.allocate(kQueryElements + 2U * kGuardElements),
              "allocate P40K-shape gate") &&
          ready;
  ready = test.cuda_ok(
              key_storage.allocate(kCacheElements + 2U * kGuardElements),
              "allocate P40K-shape key") &&
          ready;
  ready = test.cuda_ok(
              value_storage.allocate(kCacheElements + 2U * kGuardElements),
              "allocate P40K-shape value") &&
          ready;
  ready = test.cuda_ok(
              baseline_storage.allocate(kQueryElements +
                                        2U * kGuardElements),
              "allocate P40K-shape baseline") &&
          ready;
  ready = test.cuda_ok(
              candidate_storage.allocate(kQueryElements +
                                         2U * kGuardElements),
              "allocate P40K-shape candidate") &&
          ready;
  if (!ready) {
    return;
  }

  auto* const query = query_storage.data() + kGuardElements;
  auto* const gate = gate_storage.data() + kGuardElements;
  auto* const key = key_storage.data() + kGuardElements;
  auto* const value = value_storage.data() + kGuardElements;
  auto* const baseline = baseline_storage.data() + kGuardElements;
  auto* const candidate = candidate_storage.data() + kGuardElements;
  std::fill_n(query_storage.data(), query_storage.size(), kGuard);
  std::fill_n(gate_storage.data(), gate_storage.size(), kGuard);
  std::fill_n(key_storage.data(), key_storage.size(), kGuard);
  std::fill_n(value_storage.data(), value_storage.size(), kGuard);
  for (std::size_t index = 0U; index < kQueryElements; ++index) {
    query[index] = static_cast<std::uint16_t>(
        0x3e80U + ((index * 13U + 19U) % 191U));
    gate[index] = static_cast<std::uint16_t>(
        0x3e00U + ((index * 37U + 7U) % 173U));
  }
  for (std::size_t index = 0U; index < kCacheElements; ++index) {
    key[index] = static_cast<std::uint16_t>(
        0x3e80U + ((index * 43U + 17U) % 181U));
    value[index] = static_cast<std::uint16_t>(
        0x3d80U + ((index * 47U + 29U) % 167U));
  }

  const auto unwritten_output_intact = [](const auto& storage,
                                           const std::size_t active_elements) {
    return std::all_of(storage.data(),
                       storage.data() + kGuardElements,
                       [](const std::uint16_t bits) {
                         return bits == kGuard;
                       }) &&
           std::all_of(storage.data() + kGuardElements + active_elements,
                       storage.data() + storage.size(),
                       [](const std::uint16_t bits) {
                         return bits == kGuard;
                       });
  };

  for (const PanelShape shape : kP40kPanels) {
    const std::size_t active_elements =
        shape.token_count * kQueryElementsPerToken;
    std::fill_n(baseline_storage.data(), baseline_storage.size(), kGuard);
    std::fill_n(candidate_storage.data(), candidate_storage.size(), kGuard);
    ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::runtime::
            launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
                query, key, value, gate, shape.first_position,
                shape.token_count, baseline, stream)),
        "P40K-shape Q64 baseline launch");
    ready = test.cuda_ok(
                static_cast<cudaError_t>(q3x::runtime::
                    launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q128_v4_panel_fixed_cuda(
                        query, key, value, gate, shape.first_position,
                        shape.token_count, candidate, stream)),
                "P40K-shape Q128 v4 launch") &&
            ready;
    ready = test.cuda_ok(cudaStreamSynchronize(stream),
                         "synchronize P40K-shape equivalence") &&
            ready;
    if (!ready) {
      return;
    }
    test.expect(std::memcmp(candidate, baseline,
                            active_elements * sizeof(std::uint16_t)) == 0,
                "Q128 v4 is bitwise equal to Q64 for a P40K panel shape");
    test.expect(unwritten_output_intact(baseline_storage, active_elements) &&
                    unwritten_output_intact(candidate_storage,
                                             active_elements),
                "P40K panel shape writes only its active output extent");
    std::cout << "GROUP_Q128_V4_P40K_SHAPE first_position="
              << shape.first_position
              << " token_count=" << shape.token_count
              << " bitwise=true\n";
  }
}

}  // namespace

int main() {
  TestContext test;
  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreateWithFlags(&stream,
                                               cudaStreamNonBlocking),
                    "create stream")) {
    return 1;
  }
  run_v4_equivalence(test, stream);
  run_p40k_panel_shape_equivalence(test, stream);
  (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy stream");
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " grouped-Q128 v4 CUDA assertion(s) failed\n";
    return 1;
  }
  std::cout << "Grouped-Q128 v4 CUDA tests passed\n";
  return 0;
}
