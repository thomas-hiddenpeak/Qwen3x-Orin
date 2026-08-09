#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
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
constexpr std::size_t kKvElementsPerToken = kKvHeads * kHeadDimension;
constexpr std::size_t kGuardElements = 32U;
constexpr std::uint16_t kGuard = 0x7fc1U;

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

void run_panel_equivalence(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kFirstPosition = 1'024U;
  constexpr std::size_t kTokenCount = 1'024U;
  constexpr std::size_t kSegmentTokens = 512U;
  constexpr std::size_t kQueryElements =
      kTokenCount * kQueryElementsPerToken;
  constexpr std::size_t kCacheElements =
      (kFirstPosition + kTokenCount) * kKvElementsPerToken;

  ManagedBuffer<std::uint16_t> query_storage;
  ManagedBuffer<std::uint16_t> gate_storage;
  ManagedBuffer<std::uint16_t> key_storage;
  ManagedBuffer<std::uint16_t> value_storage;
  ManagedBuffer<std::uint16_t> panel_storage;
  ManagedBuffer<std::uint16_t> segmented_storage;
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
              panel_storage.allocate(kQueryElements + 2U * kGuardElements),
              "allocate panel output") &&
          ready;
  ready = test.cuda_ok(
              segmented_storage.allocate(kQueryElements +
                                         2U * kGuardElements),
              "allocate segmented output") &&
          ready;
  ready = test.cuda_ok(
              replay_storage.allocate(kQueryElements + 2U * kGuardElements),
              "allocate replay output") &&
          ready;
  if (!ready) {
    return;
  }

  auto* const query = query_storage.data() + kGuardElements;
  auto* const gate = gate_storage.data() + kGuardElements;
  auto* const key = key_storage.data() + kGuardElements;
  auto* const value = value_storage.data() + kGuardElements;
  auto* const panel = panel_storage.data() + kGuardElements;
  auto* const segmented = segmented_storage.data() + kGuardElements;
  auto* const replay = replay_storage.data() + kGuardElements;
  std::fill_n(query_storage.data(), query_storage.size(), kGuard);
  std::fill_n(gate_storage.data(), gate_storage.size(), kGuard);
  std::fill_n(key_storage.data(), key_storage.size(), kGuard);
  std::fill_n(value_storage.data(), value_storage.size(), kGuard);
  std::fill_n(panel_storage.data(), panel_storage.size(), kGuard);
  std::fill_n(segmented_storage.data(), segmented_storage.size(), kGuard);
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
          launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
              query + 1U, key, value, gate, kFirstPosition, kTokenCount,
              panel, stream) == static_cast<int>(cudaErrorInvalidValue),
      "panel entry rejects a misaligned vector operand");
  test.expect(
      q3x::runtime::
          launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
              query, key, value, gate, kFirstPosition, kTokenCount, query,
              stream) == static_cast<int>(cudaErrorInvalidValue),
      "panel entry rejects overlapping input and output ranges");

  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
              query, key, value, gate, kFirstPosition, kTokenCount, panel,
              stream)),
      "panel launch");
  for (std::size_t offset = 0U; ready && offset < kTokenCount;
       offset += kSegmentTokens) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::runtime::
            launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
                query + offset * kQueryElementsPerToken, key, value,
                gate + offset * kQueryElementsPerToken,
                kFirstPosition + offset, kSegmentTokens,
                segmented + offset * kQueryElementsPerToken, stream)),
        "segmented group-Q64 launch") &&
            ready;
  }
  {
    const ScopedHostileEnvironment hostile_environment;
    ready = test.cuda_ok(
                static_cast<cudaError_t>(q3x::runtime::
                    launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
                        query, key, value, gate, kFirstPosition,
                        kTokenCount, replay, stream)),
                "hostile-environment replay") &&
            ready;
  }
  ready = test.cuda_ok(cudaStreamSynchronize(stream), "synchronize") &&
          ready;
  if (!ready) {
    return;
  }

  test.expect(std::memcmp(panel, segmented,
                          kQueryElements * sizeof(std::uint16_t)) == 0,
              "one C1024 panel is bitwise equal to two grouped-Q64 C512 "
              "segments");
  test.expect(std::memcmp(panel, replay,
                          kQueryElements * sizeof(std::uint16_t)) == 0,
              "explicit panel entry is deterministic and ignores hostile "
              "FlashInfer selector state");

  cudaGraph_t graph = nullptr;
  ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "begin graph capture");
  if (ready) {
    ready = test.cuda_ok(
                static_cast<cudaError_t>(q3x::runtime::
                    launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
                        query, key, value, gate, kFirstPosition,
                        kTokenCount, replay, stream)),
                "capture panel launch") &&
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
                "panel entry captures exactly one native kernel");
    if (ready && node_count == 1U) {
      cudaKernelNodeParams parameters{};
      ready = test.cuda_ok(cudaGraphKernelNodeGetParams(nodes[0U],
                                                        &parameters),
                           "read panel kernel parameters") &&
              ready;
      test.expect(parameters.gridDim.x == 96U &&
                      parameters.gridDim.y == 1U &&
                      parameters.gridDim.z == 4U &&
                      parameters.blockDim.x == 128U &&
                      parameters.blockDim.y == 1U &&
                      parameters.blockDim.z == 1U &&
                      parameters.sharedMemBytes == 64U * 1024U,
                  "C1024 panel has packed-Q64/KV-head native topology");
      cudaFuncAttributes attributes{};
      int active_blocks = 0;
      ready = test.cuda_ok(cudaFuncGetAttributes(&attributes,
                                                 parameters.func),
                           "read panel kernel resources") &&
              ready;
      ready = test.cuda_ok(
                  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                      &active_blocks, parameters.func, 128,
                      parameters.sharedMemBytes),
                  "read panel occupancy") &&
              ready;
      test.expect(attributes.localSizeBytes == 0U &&
                      attributes.numRegs <= 255 && active_blocks >= 2,
                  "panel kernel retains zero spill and at least 2 CTA/SM");
      std::cout << "GROUP_Q64_PANEL_RESOURCES registers="
                << attributes.numRegs
                << " local_bytes=" << attributes.localSizeBytes
                << " dynamic_shared_bytes=" << parameters.sharedMemBytes
                << " active_blocks_per_sm=" << active_blocks << '\n';
    }
  }
  if (graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(graph), "destroy graph");
  }

  test.expect(guards_intact(panel_storage, kQueryElements) &&
                  guards_intact(segmented_storage, kQueryElements) &&
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
      "panel kernel preserves Q/Gate/K/V and their guards");
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
  run_panel_equivalence(test, stream);
  (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy stream");
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " grouped-Q64 panel CUDA assertion(s) failed\n";
    return 1;
  }
  std::cout << "Grouped-Q64 panel CUDA tests passed\n";
  return 0;
}
