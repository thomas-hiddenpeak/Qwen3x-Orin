#include "q3x/kernels/sm87_nvfp4_marlin_p40_parity.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using q3x::kernels::Sm87NvFp4MarlinP40ParityLaunchCounters;
using q3x::kernels::Sm87NvFp4MarlinP40ParityResources;
using q3x::kernels::Sm87NvFp4MarlinP40ParityRole;
using q3x::kernels::kSm87NvFp4MarlinDynamicSharedBytes;
using q3x::kernels::kSm87NvFp4MarlinP40ParityLegacySegmentCount;
using q3x::kernels::kSm87NvFp4MarlinP40ParityLockBytes;
using q3x::kernels::kSm87NvFp4MarlinP40ParityReductionBytes;
using q3x::kernels::kSm87NvFp4MarlinP40ParitySegmentCount;

inline constexpr char kRunEnvironment[] =
    "Q3X_RUN_SM87_NVFP4_MARLIN_P40_PARITY_CUDA";

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

[[nodiscard]] bool explicitly_enabled() noexcept {
  const char* const value = std::getenv(kRunEnvironment);
  return value != nullptr && std::strcmp(value, "1") == 0;
}

template <typename T>
[[nodiscard]] const T* const_pointer(const std::uintptr_t address) noexcept {
  return reinterpret_cast<const T*>(address);
}

template <typename T>
[[nodiscard]] T* mutable_pointer(const std::uintptr_t address) noexcept {
  return reinterpret_cast<T*>(address);
}

void check_resources(TestContext& test,
                     const Sm87NvFp4MarlinP40ParityRole role) {
  Sm87NvFp4MarlinP40ParityResources resources{};
  const int status =
      q3x::kernels::query_sm87_nvfp4_marlin_p40_parity_resources_cuda(
          role, &resources);
  test.expect(status == static_cast<int>(cudaSuccess),
              "P40 parity resource query must succeed");
  if (status != static_cast<int>(cudaSuccess)) {
    return;
  }
  test.expect(resources.supported && resources.role == role,
              "P40 parity resource receipt must preserve its role");
  test.expect(resources.legacy_stripe.dynamic_shared_bytes ==
                  kSm87NvFp4MarlinDynamicSharedBytes &&
                  resources.legacy_stripe.maximum_threads_per_block >= 256 &&
                  resources.legacy_stripe.active_blocks_per_sm >= 1,
              "the shared stock kernel must retain its resource contract");
  test.expect(resources.bulk_and_tail_share_kernel &&
                  resources.reduction_workspace_bytes ==
                      kSm87NvFp4MarlinP40ParityReductionBytes &&
                  resources.lock_bytes ==
                      kSm87NvFp4MarlinP40ParityLockBytes &&
                  resources.requires_zero_initialized_locks &&
                  !resources.atomic_add && resources.fp32_reduce,
              "resource receipt must expose stock tail reduction semantics");
  test.expect(resources.tail_split_k_output_tiles ==
                      (role == Sm87NvFp4MarlinP40ParityRole::kGateUp ? 8U
                                                                    : 12U) &&
                  resources.tail_split_k_partial_slices ==
                      (role == Sm87NvFp4MarlinP40ParityRole::kGateUp ? 16U
                                                                    : 24U),
              "resource receipt must expose role-specific stock split tiles");
}

void check_graph_nodes(TestContext& test, cudaGraph_t graph,
                       const std::size_t expected_nodes,
                       const char* const message) {
  std::size_t node_count = 0U;
  if (!cuda_ok(test, cudaGraphGetNodes(graph, nullptr, &node_count),
               "query P40 parity graph node count")) {
    return;
  }
  test.expect(node_count == expected_nodes, message);
  std::vector<cudaGraphNode_t> nodes(node_count);
  if (node_count == 0U ||
      !cuda_ok(test, cudaGraphGetNodes(graph, nodes.data(), &node_count),
               "read P40 parity graph nodes")) {
    return;
  }
  for (const cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    if (cuda_ok(test, cudaGraphNodeGetType(node, &type),
                "read P40 parity graph node type")) {
      test.expect(type == cudaGraphNodeTypeKernel,
                  "P40 parity graph may only contain kernel nodes");
    }
  }
}

void capture_projection(TestContext& test, cudaStream_t stream,
                        const Sm87NvFp4MarlinP40ParityRole role) {
  // Capture-only canonical addresses.  They are deliberately far enough
  // apart for the public whole-span overlap checks and are never dereferenced.
  constexpr std::uintptr_t kInput = 0x0000'0010'0000'0000ULL;
  constexpr std::uintptr_t kWeight = 0x0000'0030'0000'0000ULL;
  constexpr std::uintptr_t kScales = 0x0000'0040'0000'0000ULL;
  constexpr std::uintptr_t kGlobalScale = 0x0000'0048'0000'0000ULL;
  constexpr std::uintptr_t kReductionWorkspace = 0x0000'0050'0000'0000ULL;
  constexpr std::uintptr_t kLocks = 0x0000'0058'0000'0000ULL;
  constexpr std::uintptr_t kOutput = 0x0000'0060'0000'0000ULL;

  if (!cuda_ok(test,
               cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "begin P40 parity projection capture")) {
    return;
  }
  Sm87NvFp4MarlinP40ParityLaunchCounters counters{};
  const int launch_status =
      role == Sm87NvFp4MarlinP40ParityRole::kGateUp
          ? q3x::kernels::
                launch_sm87_nvfp4_marlin_p40_parity_gate_up_cuda(
                    const_pointer<std::uint16_t>(kInput),
                    const_pointer<std::uint8_t>(kWeight),
                    const_pointer<std::uint8_t>(kScales),
                    const_pointer<float>(kGlobalScale),
                    mutable_pointer<std::uint16_t>(kOutput),
                    mutable_pointer<float>(kReductionWorkspace),
                    kSm87NvFp4MarlinP40ParityReductionBytes,
                    mutable_pointer<std::int32_t>(kLocks),
                    kSm87NvFp4MarlinP40ParityLockBytes, &counters, stream)
          : q3x::kernels::launch_sm87_nvfp4_marlin_p40_parity_down_cuda(
                const_pointer<std::uint16_t>(kInput),
                const_pointer<std::uint8_t>(kWeight),
                const_pointer<std::uint8_t>(kScales),
                const_pointer<float>(kGlobalScale),
                mutable_pointer<std::uint16_t>(kOutput),
                mutable_pointer<float>(kReductionWorkspace),
                kSm87NvFp4MarlinP40ParityReductionBytes,
                mutable_pointer<std::int32_t>(kLocks),
                kSm87NvFp4MarlinP40ParityLockBytes, &counters, stream);
  test.expect(launch_status == static_cast<int>(cudaSuccess),
              "P40 parity projection must enqueue during capture");
  test.expect(counters.legacy_full_k_m1024_launches ==
                      kSm87NvFp4MarlinP40ParityLegacySegmentCount &&
                  counters.legacy_split_k_m64_launches == 1U &&
                  counters.physical_projection_launches ==
                      kSm87NvFp4MarlinP40ParitySegmentCount &&
                  counters.tail_split_k_output_tiles ==
                      (role == Sm87NvFp4MarlinP40ParityRole::kGateUp ? 8U
                                                                    : 12U) &&
                  counters.tail_split_k_partial_slices ==
                      (role == Sm87NvFp4MarlinP40ParityRole::kGateUp ? 16U
                                                                    : 24U) &&
                  counters.complete,
              "projection counters must expose stock 39-full+1-split topology");

  cudaGraph_t graph = nullptr;
  if (!cuda_ok(test, cudaStreamEndCapture(stream, &graph),
               "end P40 parity projection capture")) {
    return;
  }
  if (graph == nullptr) {
    ++test.failures;
    std::cerr << "P40 parity projection graph is null\n";
    return;
  }
  check_graph_nodes(test, graph, kSm87NvFp4MarlinP40ParitySegmentCount,
                    "each projection must contain exactly 39+1 kernel nodes");
  (void)cudaGraphDestroy(graph);
}

void capture_standalone_silu(TestContext& test, cudaStream_t stream) {
  constexpr std::uintptr_t kMergedGateUp = 0x0000'0010'0000'0000ULL;
  constexpr std::uintptr_t kActivated = 0x0000'0040'0000'0000ULL;
  if (!cuda_ok(test,
               cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "begin standalone SiLU capture")) {
    return;
  }
  const int launch_status =
      q3x::kernels::launch_sm87_nvfp4_marlin_p40_parity_silu_cuda(
          const_pointer<std::uint16_t>(kMergedGateUp),
          mutable_pointer<std::uint16_t>(kActivated), stream);
  test.expect(launch_status == static_cast<int>(cudaSuccess),
              "standalone SiLU must enqueue during capture");
  cudaGraph_t graph = nullptr;
  if (!cuda_ok(test, cudaStreamEndCapture(stream, &graph),
               "end standalone SiLU capture")) {
    return;
  }
  if (graph == nullptr) {
    ++test.failures;
    std::cerr << "standalone SiLU graph is null\n";
    return;
  }
  check_graph_nodes(test, graph, 1U,
                    "BF16 Gate/Up publication and SiLU must remain separate");
  (void)cudaGraphDestroy(graph);
}

}  // namespace

int main() {
  if (!explicitly_enabled()) {
    std::cout << "SKIP: set " << kRunEnvironment
              << "=1 only after clean tegrastats/process/GPU-handle preflight\n";
    return 77;
  }

  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      device_count == 0) {
    std::cout << "SKIP: CUDA device unavailable\n";
    return 77;
  }
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(status)
              << '\n';
    return 1;
  }
  int device = 0;
  status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(status)
              << '\n';
    return 1;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties failed: "
              << cudaGetErrorString(status) << '\n';
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: exact 16-SM SM87 target required\n";
    return 77;
  }

  TestContext test;
  check_resources(test, Sm87NvFp4MarlinP40ParityRole::kGateUp);
  check_resources(test, Sm87NvFp4MarlinP40ParityRole::kDown);
  cudaStream_t stream = nullptr;
  if (!cuda_ok(test, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create P40 parity capture stream")) {
    return 1;
  }
  capture_projection(test, stream, Sm87NvFp4MarlinP40ParityRole::kGateUp);
  capture_projection(test, stream, Sm87NvFp4MarlinP40ParityRole::kDown);
  capture_standalone_silu(test, stream);
  (void)cudaStreamDestroy(stream);

  if (test.failures != 0) {
    std::cerr << test.failures << " P40 parity CUDA checks failed\n";
    return 1;
  }
  std::cout << "P40 parity CUDA topology checks passed\n";
  return 0;
}
