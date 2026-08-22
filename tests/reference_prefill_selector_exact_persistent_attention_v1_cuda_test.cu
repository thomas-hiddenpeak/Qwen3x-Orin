#include "reference_runner_selector_exact_persistent_attention_v1_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace detail = q3x::runtime::reference_runner_detail;

constexpr std::size_t kPromptTokens = 513U;
constexpr std::size_t kQueryHeads = 24U;
constexpr std::size_t kKvHeads = 4U;
constexpr std::size_t kHeadDimension = 256U;
constexpr std::size_t kQueryElementsPerToken =
    kQueryHeads * kHeadDimension;
constexpr std::size_t kKvElementsPerToken = kKvHeads * kHeadDimension;
constexpr std::size_t kGuardElements = 32U;
constexpr std::uint16_t kGuard = 0x7fc1U;

class TestContext final {
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
class ManagedBuffer final {
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

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

[[nodiscard]] bool guards_intact(
    const ManagedBuffer<std::uint16_t>& storage,
    const std::size_t payload_elements) {
  return std::all_of(
             storage.data(), storage.data() + kGuardElements,
             [](const std::uint16_t value) { return value == kGuard; }) &&
         std::all_of(
             storage.data() + kGuardElements + payload_elements,
             storage.data() + storage.size(),
             [](const std::uint16_t value) { return value == kGuard; });
}

void run_p513_bitwise(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kQueryElements =
      kPromptTokens * kQueryElementsPerToken;
  constexpr std::size_t kCacheElements =
      kPromptTokens * kKvElementsPerToken;
  constexpr std::size_t kSecondSpanOffset = 257U;
  constexpr std::size_t kSecondSpanTokens = 256U;

  ManagedBuffer<std::uint16_t> query_storage;
  ManagedBuffer<std::uint16_t> gate_storage;
  ManagedBuffer<std::uint16_t> key_storage;
  ManagedBuffer<std::uint16_t> value_storage;
  ManagedBuffer<std::uint16_t> incumbent_storage;
  ManagedBuffer<std::uint16_t> candidate_storage;
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
              incumbent_storage.allocate(kQueryElements +
                                         2U * kGuardElements),
              "allocate incumbent output") &&
          ready;
  ready = test.cuda_ok(
              candidate_storage.allocate(kQueryElements +
                                         2U * kGuardElements),
              "allocate candidate output") &&
          ready;
  if (!ready) {
    return;
  }

  std::fill_n(query_storage.data(), query_storage.size(), kGuard);
  std::fill_n(gate_storage.data(), gate_storage.size(), kGuard);
  std::fill_n(key_storage.data(), key_storage.size(), kGuard);
  std::fill_n(value_storage.data(), value_storage.size(), kGuard);
  std::fill_n(incumbent_storage.data(), incumbent_storage.size(), kGuard);
  std::fill_n(candidate_storage.data(), candidate_storage.size(), kGuard);
  std::uint16_t* const query = query_storage.data() + kGuardElements;
  std::uint16_t* const gate = gate_storage.data() + kGuardElements;
  std::uint16_t* const key = key_storage.data() + kGuardElements;
  std::uint16_t* const value = value_storage.data() + kGuardElements;
  std::uint16_t* const incumbent =
      incumbent_storage.data() + kGuardElements;
  std::uint16_t* const candidate =
      candidate_storage.data() + kGuardElements;

  for (std::size_t index = 0U; index < kQueryElements; ++index) {
    query[index] = static_cast<std::uint16_t>(
        0x3e00U + ((index * 17U + index / 127U + 5U) % 383U));
    gate[index] = static_cast<std::uint16_t>(
        0x3d80U + ((index * 23U + index / 61U + 3U) % 257U));
  }
  for (std::size_t index = 0U; index < kCacheElements; ++index) {
    key[index] = static_cast<std::uint16_t>(
        0x3d80U + ((index * 29U + index / 47U + 7U) % 509U));
    value[index] = static_cast<std::uint16_t>(
        0x3d00U + ((index * 31U + index / 43U + 11U) % 503U));
  }

  test.expect(
      detail::launch_selector_exact_persistent_attention_v1_q8_generic_suffix_cuda(
          query + kSecondSpanOffset * kQueryElementsPerToken, key, value,
          gate + kSecondSpanOffset * kQueryElementsPerToken,
          kSecondSpanOffset, kSecondSpanTokens - 1U,
          candidate + kSecondSpanOffset * kQueryElementsPerToken,
          stream) == static_cast<int>(cudaErrorInvalidValue),
      "Q8 suffix rejects a non-Q8-aligned token count");

  ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::runtime::launch_bulk_causal_gqa_sigmoid_gate_24_4_256_fixed_cuda(
              query, key, value, gate, 0U, kSecondSpanOffset, incumbent,
              stream)),
      "incumbent P513 GroupQ64 launch");
  ready = test.cuda_ok(
              static_cast<cudaError_t>(
                  q3x::runtime::
                      launch_bulk_causal_gqa_sigmoid_gate_24_4_256_fixed_cuda(
                          query +
                              kSecondSpanOffset * kQueryElementsPerToken,
                          key, value,
                          gate + kSecondSpanOffset *
                                     kQueryElementsPerToken,
                          kSecondSpanOffset, kSecondSpanTokens,
                          incumbent + kSecondSpanOffset *
                                          kQueryElementsPerToken,
                          stream)),
              "incumbent P513 GenericQT2 second-span launch") &&
          ready;

  detail::SelectorExactPersistentAttentionV1LaunchReceipt receipt;
  ready = test.cuda_ok(
              static_cast<cudaError_t>(
                  detail::launch_selector_exact_persistent_attention_v1_cuda(
                      query, key, value, gate, 0U, kPromptTokens, candidate,
                      &receipt, stream)),
              "candidate P513 composite launch") &&
          ready;
  ready = test.cuda_ok(cudaStreamSynchronize(stream), "synchronize") && ready;
  if (!ready) {
    return;
  }

  const std::size_t second_span_elements =
      kSecondSpanTokens * kQueryElementsPerToken;
  test.expect(
      std::memcmp(incumbent +
                      kSecondSpanOffset * kQueryElementsPerToken,
                  candidate +
                      kSecondSpanOffset * kQueryElementsPerToken,
                  second_span_elements * sizeof(std::uint16_t)) == 0,
      "P513 Q8 suffix is bitwise equal to incumbent GenericQT2 second span");
  test.expect(std::memcmp(incumbent, candidate,
                          kQueryElements * sizeof(std::uint16_t)) == 0,
              "P513 complete composite output is bitwise equal to incumbent");
  test.expect(receipt.plan.valid &&
                  receipt.plan.token_count == kPromptTokens &&
                  receipt.plan.physical_submission_count == 2U &&
                  receipt.group_q64_submissions == 1U &&
                  receipt.generic_q8_suffix_submissions == 1U &&
                  receipt.fallback_submissions == 0U &&
                  receipt.persistent_ctas ==
                      detail::
                          kSelectorExactPersistentAttentionV1PersistentBlockCount,
              "P513 candidate receipt proves one Q64 plus one Q8 submission");
  test.expect(guards_intact(incumbent_storage, kQueryElements) &&
                  guards_intact(candidate_storage, kQueryElements),
              "candidate and incumbent output guards are intact");

  cudaGraph_t graph = nullptr;
  ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "begin candidate graph capture");
  detail::SelectorExactPersistentAttentionV1LaunchReceipt graph_receipt;
  if (ready) {
    ready = test.cuda_ok(
                static_cast<cudaError_t>(
                    detail::
                        launch_selector_exact_persistent_attention_v1_cuda(
                            query, key, value, gate, 0U, kPromptTokens,
                            candidate, &graph_receipt, stream)),
                "capture candidate P513 composite") &&
            ready;
  }
  if (ready) {
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         "end candidate graph capture") &&
            ready;
  }
  if (ready) {
    std::size_t graph_nodes = 0U;
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &graph_nodes),
                         "count candidate graph nodes") &&
            ready;
    test.expect(graph_nodes == 2U,
                "P513 composite graph has exactly two physical kernel "
                "submissions");
    test.expect(graph_receipt.group_q64_submissions == 1U &&
                    graph_receipt.generic_q8_suffix_submissions == 1U &&
                    graph_receipt.persistent_ctas == 16U,
                "captured graph receipt retains one Q64 plus one persistent "
                "Q8 suffix");
  }
  if (graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(graph), "destroy candidate graph");
  }
}

void check_resources(TestContext& test) {
  detail::SelectorExactPersistentAttentionV1Resources resources;
  const cudaError_t status = static_cast<cudaError_t>(
      detail::query_selector_exact_persistent_attention_v1_q8_resources_cuda(
          &resources));
  if (!test.cuda_ok(status, "query Q8 resources")) {
    return;
  }
  test.expect(resources.registers_per_thread > 0 &&
                  resources.registers_per_thread <= 255,
              "Q8 register count is within the admitted SM87 bound");
  test.expect(resources.local_bytes == 0U,
              "Q8 has no local-memory spill");
  test.expect(resources.static_shared_bytes == 16'384U,
              "Q8 uses exactly the retained 16 KiB K/V tile");
  test.expect(resources.maximum_threads >= 192 &&
                  resources.threads_per_block == 192,
              "Q8 retains the one-KV-head/six-query-warp CTA");
  test.expect(resources.active_blocks_per_multiprocessor >= 1,
              "Q8 admits at least one active CTA per SM");
  std::cout << "Q8_RESOURCES regs=" << resources.registers_per_thread
            << " local=" << resources.local_bytes
            << " smem=" << resources.static_shared_bytes
            << " threads=" << resources.threads_per_block
            << " active_blocks_per_sm="
            << resources.active_blocks_per_multiprocessor << '\n';
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: CUDA device unavailable\n";
    return 77;
  }

  TestContext test;
  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreate(&stream), "create stream")) {
    return 1;
  }
  check_resources(test);
  run_p513_bitwise(test, stream);
  (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy stream");
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " selector-exact persistent Attention CUDA checks failed\n";
    return 1;
  }
  std::cout << "selector-exact-persistent-attention-v1 CUDA contract PASS\n";
  return 0;
}
