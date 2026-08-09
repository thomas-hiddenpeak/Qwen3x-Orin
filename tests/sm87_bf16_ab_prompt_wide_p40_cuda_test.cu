#include "q3x/kernels/sm87_bf16_ab_prefill.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

struct TestContext final {
  int failures = 0;

  void expect(const bool condition, const char* const message) {
    if (!condition) {
      ++failures;
      std::cerr << message << '\n';
    }
  }
};

[[nodiscard]] bool cuda_ok(TestContext& test, const cudaError_t status,
                           const char* const message) {
  if (status == cudaSuccess) {
    return true;
  }
  ++test.failures;
  std::cerr << message << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

[[nodiscard]] const std::uint16_t* const_pointer(
    const std::uintptr_t address) noexcept {
  return reinterpret_cast<const std::uint16_t*>(address);
}

[[nodiscard]] std::uint16_t* mutable_pointer(
    const std::uintptr_t address) noexcept {
  return reinterpret_cast<std::uint16_t*>(address);
}

void test_invalid_contracts(TestContext& test) {
  constexpr std::uintptr_t kFirstWeights = 0x0000'0010'0000'0000ULL;
  constexpr std::uintptr_t kSecondWeights = 0x0000'0020'0000'0000ULL;
  constexpr std::uintptr_t kInput = 0x0000'0030'0000'0000ULL;
  constexpr std::uintptr_t kFirstOutput = 0x0000'0050'0000'0000ULL;
  constexpr std::uintptr_t kSecondOutput = 0x0000'0060'0000'0000ULL;
  const auto launch = [&](const std::uint16_t* const first_weights,
                          const std::uint16_t* const second_weights,
                          const std::uint16_t* const input,
                          const std::size_t tokens,
                          std::uint16_t* const first_output,
                          std::uint16_t* const second_output) {
    return q3x::kernels::launch_sm87_bf16_ab_prompt_wide_p40_cuda(
        first_weights, second_weights, input, tokens,
        first_output, second_output, nullptr);
  };

  test.expect(
      launch(const_pointer(kFirstWeights), const_pointer(kSecondWeights),
             const_pointer(kInput), 39'999U, mutable_pointer(kFirstOutput),
             mutable_pointer(kSecondOutput)) ==
          static_cast<int>(cudaErrorInvalidValue),
      "P39999 must fail closed before enqueue");
  test.expect(
      launch(const_pointer(kFirstWeights), const_pointer(kSecondWeights),
             const_pointer(kInput), 40'001U, mutable_pointer(kFirstOutput),
             mutable_pointer(kSecondOutput)) ==
          static_cast<int>(cudaErrorInvalidValue),
      "P40001 must fail closed before enqueue");
  test.expect(
      launch(const_pointer(kFirstWeights + 2U),
             const_pointer(kSecondWeights), const_pointer(kInput), 40'000U,
             mutable_pointer(kFirstOutput), mutable_pointer(kSecondOutput)) ==
          static_cast<int>(cudaErrorInvalidValue),
      "misaligned BF16 A weights must fail closed");
  test.expect(
      launch(const_pointer(kFirstWeights), const_pointer(kSecondWeights),
             const_pointer(kInput), 40'000U, mutable_pointer(kInput),
             mutable_pointer(kSecondOutput)) ==
          static_cast<int>(cudaErrorInvalidValue),
      "A output overlapping prompt input must fail closed");
  test.expect(
      launch(const_pointer(kFirstWeights), const_pointer(kSecondWeights),
             const_pointer(kInput), 40'000U,
             mutable_pointer(kFirstOutput),
             mutable_pointer(kFirstOutput + 4U)) ==
          static_cast<int>(cudaErrorInvalidValue),
      "partially overlapping A/B outputs must fail closed");
  test.expect(
      launch(nullptr, const_pointer(kSecondWeights), const_pointer(kInput),
             40'000U, mutable_pointer(kFirstOutput),
             mutable_pointer(kSecondOutput)) ==
          static_cast<int>(cudaErrorInvalidValue),
      "null payload must fail closed");
}

void test_exact_one_grid_capture(TestContext& test, cudaStream_t stream) {
  // Capture only; these non-overlapping canonical addresses are never
  // dereferenced.  This keeps the topology test independent of a 418 MiB
  // P40000 fixture while proving the public entry contributes one kernel node.
  constexpr std::uintptr_t kFirstWeights = 0x0000'0010'0000'0000ULL;
  constexpr std::uintptr_t kSecondWeights = 0x0000'0020'0000'0000ULL;
  constexpr std::uintptr_t kInput = 0x0000'0030'0000'0000ULL;
  constexpr std::uintptr_t kFirstOutput = 0x0000'0050'0000'0000ULL;
  constexpr std::uintptr_t kSecondOutput = 0x0000'0060'0000'0000ULL;

  if (!cuda_ok(test,
               cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "begin exact-P40 graph capture")) {
    return;
  }
  const int launch_status =
      q3x::kernels::launch_sm87_bf16_ab_prompt_wide_p40_cuda(
          const_pointer(kFirstWeights), const_pointer(kSecondWeights),
          const_pointer(kInput), 40'000U, mutable_pointer(kFirstOutput),
          mutable_pointer(kSecondOutput), stream);
  test.expect(launch_status == static_cast<int>(cudaSuccess),
              "exact-P40 prompt-wide entry must enqueue during capture");

  cudaGraph_t graph = nullptr;
  if (!cuda_ok(test, cudaStreamEndCapture(stream, &graph),
               "end exact-P40 graph capture")) {
    return;
  }
  if (graph == nullptr) {
    ++test.failures;
    std::cerr << "exact-P40 graph is null\n";
    return;
  }

  std::size_t node_count = 0U;
  if (!cuda_ok(test, cudaGraphGetNodes(graph, nullptr, &node_count),
               "query exact-P40 graph node count")) {
    (void)cudaGraphDestroy(graph);
    return;
  }
  test.expect(node_count == 1U,
              "exact-P40 BF16 A/B entry must contain exactly one launch");
  std::vector<cudaGraphNode_t> nodes(node_count);
  if (node_count == 1U &&
      cuda_ok(test, cudaGraphGetNodes(graph, nodes.data(), &node_count),
              "read exact-P40 graph node")) {
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    if (cuda_ok(test, cudaGraphNodeGetType(nodes[0], &type),
                "read exact-P40 graph node type")) {
      test.expect(type == cudaGraphNodeTypeKernel,
                  "exact-P40 graph node must be a CUDA kernel");
    }
    cudaKernelNodeParams parameters{};
    if (cuda_ok(test, cudaGraphKernelNodeGetParams(nodes[0], &parameters),
                "read exact-P40 kernel parameters")) {
      test.expect(parameters.gridDim.x == 625U &&
                      parameters.gridDim.y == 1U &&
                      parameters.gridDim.z == 1U,
                  "exact-P40 kernel grid must be dim3(625,1,1)");
      test.expect(parameters.blockDim.x == 256U &&
                      parameters.blockDim.y == 1U &&
                      parameters.blockDim.z == 1U,
                  "exact-P40 kernel block must be dim3(256,1,1)");
      test.expect(parameters.sharedMemBytes == 46'080U,
                  "exact-P40 kernel must retain the M64 arithmetic "
                  "shared-memory boundary");
    }
  }
  (void)cudaGraphDestroy(graph);
}

}  // namespace

int main() {
  TestContext test;
  test_invalid_contracts(test);

  q3x::kernels::Sm87Bf16AbPromptWideP40Resources resources{};
  const int resource_status =
      q3x::kernels::query_sm87_bf16_ab_prompt_wide_p40_resources_cuda(
          &resources);
  if (resource_status == static_cast<int>(cudaErrorNoDevice) ||
      resource_status == static_cast<int>(cudaErrorInsufficientDriver) ||
      resource_status == static_cast<int>(cudaErrorNotSupported)) {
    std::cout << "SKIP: exact SM87 device/cubin capability unavailable\n";
    return test.failures == 0 ? 77 : 1;
  }
  test.expect(resource_status == static_cast<int>(cudaSuccess),
              "SM87 resource query must succeed");
  test.expect(resources.valid(),
              "SM87 resource receipt must satisfy the two-CTA contract");
  if (resource_status != static_cast<int>(cudaSuccess) ||
      !resources.valid()) {
    return 1;
  }

  cudaStream_t stream = nullptr;
  if (!cuda_ok(test, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create capture stream")) {
    return 1;
  }
  test_exact_one_grid_capture(test, stream);
  (void)cudaStreamDestroy(stream);

  if (test.failures != 0) {
    std::cerr << test.failures
              << " BF16 A/B prompt-wide P40 CUDA checks failed\n";
    return 1;
  }
  std::cout << "BF16 A/B prompt-wide P40 CUDA checks passed\n";
  return 0;
}
