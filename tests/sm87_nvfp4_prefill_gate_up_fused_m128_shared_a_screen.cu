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

// Test-only ABI implemented beside the frozen production kernels. It remains
// private to this isolated admission screen.
[[nodiscard]] int
launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_fused_m128_shared_a_test_cuda(
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* gate_block_scales, float gate_weight_scale_2,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* up_block_scales, float up_weight_scale_2,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* gate_output,
    std::uint16_t* up_output, void* cuda_stream) noexcept;

[[nodiscard]] int
query_sm87_nvfp4_w4a16_whole_chunk_gate_up_fused_m128_shared_a_resources_test_cuda(
    std::size_t token_count, std::size_t rows, std::size_t columns,
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* maximum_threads_per_block, int* active_blocks_per_sm) noexcept;

}  // namespace q3x::kernels

namespace {

constexpr std::size_t kRows = 17'408U;
constexpr std::size_t kColumns = 5'120U;
constexpr std::size_t kPackedBytes = kRows * (kColumns / 2U);
constexpr std::size_t kScaleColumns = kColumns / 16U;
constexpr std::size_t kScaleBytes = kRows * kScaleColumns;
constexpr std::size_t kGuardElements = 64U;
constexpr std::size_t kScrubBytes = 32U * 1024U * 1024U;
constexpr int kWarmups = 10;
constexpr int kIterations = 24;
constexpr int kRounds = 6;
constexpr double kRequiredC512Speedup = 1.22;
constexpr std::array<std::uint8_t, 32U> kCheckpointLikeScaleCodes{{
    0x4eU, 0x50U, 0x52U, 0x54U, 0x55U, 0x56U, 0x57U, 0x58U,
    0x58U, 0x58U, 0x59U, 0x59U, 0x59U, 0x5aU, 0x5aU, 0x5bU,
    0x5bU, 0x5cU, 0x5cU, 0x5dU, 0x5dU, 0x5eU, 0x5fU, 0x60U,
    0x60U, 0x61U, 0x62U, 0x63U, 0x64U, 0x65U, 0x66U, 0x67U,
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
      test.expect(false, label + " allocation size is representable");
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
    if (done_ != nullptr) {
      (void)cudaEventDestroy(done_);
    }
    if (ready_ != nullptr) {
      (void)cudaEventDestroy(ready_);
    }
    if (auxiliary_ != nullptr) {
      (void)cudaStreamDestroy(auxiliary_);
    }
    if (main_ != nullptr) {
      (void)cudaStreamDestroy(main_);
    }
  }

  [[nodiscard]] bool create(TestContext& test) {
    bool ready = test.cuda_ok(
        cudaStreamCreateWithFlags(&main_, cudaStreamNonBlocking),
        "create main stream");
    ready = ready && test.cuda_ok(
                         cudaStreamCreateWithFlags(&auxiliary_,
                                                   cudaStreamNonBlocking),
                         "create auxiliary stream");
    ready = ready && test.cuda_ok(
                         cudaEventCreateWithFlags(&ready_,
                                                  cudaEventDisableTiming),
                         "create branch-ready event");
    ready = ready && test.cuda_ok(
                         cudaEventCreateWithFlags(&done_,
                                                  cudaEventDisableTiming),
                         "create branch-done event");
    ready = ready && test.cuda_ok(cudaEventCreate(&start_),
                                  "create timing start");
    ready = ready && test.cuda_ok(cudaEventCreate(&stop_),
                                  "create timing stop");
    return ready;
  }

  [[nodiscard]] cudaStream_t main() const noexcept { return main_; }
  [[nodiscard]] cudaStream_t auxiliary() const noexcept { return auxiliary_; }
  [[nodiscard]] cudaEvent_t ready() const noexcept { return ready_; }
  [[nodiscard]] cudaEvent_t done() const noexcept { return done_; }
  [[nodiscard]] cudaEvent_t start() const noexcept { return start_; }
  [[nodiscard]] cudaEvent_t stop() const noexcept { return stop_; }

 private:
  cudaStream_t main_ = nullptr;
  cudaStream_t auxiliary_ = nullptr;
  cudaEvent_t ready_ = nullptr;
  cudaEvent_t done_ = nullptr;
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
  [[nodiscard]] cudaGraph_t graph() const noexcept { return graph_; }
  [[nodiscard]] cudaGraphExec_t* executable_address() noexcept {
    return &executable_;
  }
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
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] bool is_bf16_finite(const std::uint16_t value) noexcept {
  return (value & 0x7f80U) != 0x7f80U;
}

__global__ void scrub_l2_kernel(std::uint32_t* const words,
                                const std::size_t count) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    words[index] = words[index] + static_cast<std::uint32_t>(index) + 1U;
  }
}

struct Fixture {
  std::size_t token_count = 512U;
  DeviceBuffer<std::uint8_t> gate_packed;
  DeviceBuffer<std::uint8_t> gate_scales;
  DeviceBuffer<std::uint8_t> up_packed;
  DeviceBuffer<std::uint8_t> up_scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> gate_output_store;
  DeviceBuffer<std::uint16_t> up_output_store;
  DeviceBuffer<std::uint32_t> scrub;
  std::vector<std::uint8_t> host_gate_packed;
  std::vector<std::uint8_t> host_gate_scales;
  std::vector<std::uint8_t> host_up_packed;
  std::vector<std::uint8_t> host_up_scales;
  std::vector<std::uint16_t> host_activations;

  [[nodiscard]] std::size_t output_elements() const noexcept {
    return token_count * kRows;
  }

  [[nodiscard]] std::size_t guarded_output_elements() const noexcept {
    return output_elements() + 2U * kGuardElements;
  }

  [[nodiscard]] std::uint16_t* gate_output() noexcept {
    return gate_output_store.get() + kGuardElements;
  }

  [[nodiscard]] std::uint16_t* up_output() noexcept {
    return up_output_store.get() + kGuardElements;
  }

  [[nodiscard]] bool initialize(TestContext& test, const Execution& execution,
                                const std::size_t tokens) {
    token_count = tokens;
    const std::size_t activation_elements = token_count * kColumns;
    host_gate_packed.resize(kPackedBytes);
    host_gate_scales.resize(kScaleBytes);
    host_up_packed.resize(kPackedBytes);
    host_up_scales.resize(kScaleBytes);
    host_activations.resize(activation_elements);

    for (std::size_t index = 0U; index < kPackedBytes; ++index) {
      const std::uint8_t low =
          static_cast<std::uint8_t>((index * 3U + (index >> 7U)) & 0x0fU);
      const std::uint8_t high = static_cast<std::uint8_t>(
          (index * 5U + (index >> 5U) + 1U) & 0x0fU);
      host_gate_packed[index] =
          static_cast<std::uint8_t>(low | (high << 4U));
      host_up_packed[index] = static_cast<std::uint8_t>(
          (low ^ 0x09U) | ((high ^ 0x05U) << 4U));
    }
    for (std::size_t index = 0U; index < kScaleBytes; ++index) {
      const std::size_t row = index / kScaleColumns;
      const std::size_t scale_column = index - row * kScaleColumns;
      host_gate_scales[index] = kCheckpointLikeScaleCodes[
          (scale_column * 5U + row * 11U + (scale_column >> 3U)) %
          kCheckpointLikeScaleCodes.size()];
      host_up_scales[index] = kCheckpointLikeScaleCodes[
          (scale_column * 7U + row * 13U + (scale_column >> 2U) + 3U) %
          kCheckpointLikeScaleCodes.size()];
    }
    for (std::size_t index = 0U; index < activation_elements; ++index) {
      const int centered = static_cast<int>(index % 29U) - 14;
      host_activations[index] =
          encode_bf16(static_cast<float>(centered) / 16.0F);
    }

    bool ready = gate_packed.allocate(test, kPackedBytes,
                                      "allocate Gate packed weights");
    ready = ready && gate_scales.allocate(test, kScaleBytes,
                                          "allocate Gate scales");
    ready = ready && up_packed.allocate(test, kPackedBytes,
                                        "allocate Up packed weights");
    ready = ready && up_scales.allocate(test, kScaleBytes,
                                        "allocate Up scales");
    ready = ready && activations.allocate(test, activation_elements,
                                          "allocate BF16 activations");
    ready = ready && gate_output_store.allocate(
                         test, guarded_output_elements(),
                         "allocate guarded Gate output");
    ready = ready && up_output_store.allocate(
                         test, guarded_output_elements(),
                         "allocate guarded Up output");
    ready = ready && scrub.allocate(test, kScrubBytes / sizeof(std::uint32_t),
                                    "allocate L2 scrub buffer");
    if (!ready) {
      return false;
    }
    const cudaStream_t stream = execution.main();
    ready = test.cuda_ok(
        cudaMemcpyAsync(gate_packed.get(), host_gate_packed.data(),
                        kPackedBytes, cudaMemcpyHostToDevice, stream),
        "upload Gate packed weights");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(gate_scales.get(),
                                         host_gate_scales.data(), kScaleBytes,
                                         cudaMemcpyHostToDevice, stream),
                         "upload Gate scales");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(up_packed.get(),
                                         host_up_packed.data(), kPackedBytes,
                                         cudaMemcpyHostToDevice, stream),
                         "upload Up packed weights");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(up_scales.get(),
                                         host_up_scales.data(), kScaleBytes,
                                         cudaMemcpyHostToDevice, stream),
                         "upload Up scales");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             activations.get(), host_activations.data(),
                             activation_elements * sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice, stream),
                         "upload BF16 activations");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(scrub.get(), 0, kScrubBytes, stream),
                         "initialize L2 scrub buffer");
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

[[nodiscard]] cudaError_t launch_baseline_pair(
    Fixture& fixture, const Execution& execution) noexcept {
  cudaError_t status = cudaEventRecord(execution.ready(), execution.main());
  if (status == cudaSuccess) {
    status = cudaStreamWaitEvent(execution.auxiliary(), execution.ready(), 0U);
  }
  if (status == cudaSuccess) {
    status = static_cast<cudaError_t>(q3x::kernels::
        launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
            fixture.gate_packed.get(), fixture.gate_scales.get(), 1.0F,
            fixture.activations.get(), fixture.token_count, kRows, kColumns,
            fixture.gate_output(), static_cast<void*>(execution.main())));
  }
  if (status == cudaSuccess) {
    status = static_cast<cudaError_t>(q3x::kernels::
        launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
            fixture.up_packed.get(), fixture.up_scales.get(), 1.0F,
            fixture.activations.get(), fixture.token_count, kRows, kColumns,
            fixture.up_output(),
            static_cast<void*>(execution.auxiliary())));
  }
  if (status == cudaSuccess) {
    status = cudaEventRecord(execution.done(), execution.auxiliary());
  }
  if (status == cudaSuccess) {
    status = cudaStreamWaitEvent(execution.main(), execution.done(), 0U);
  }
  return status;
}

[[nodiscard]] cudaError_t launch_candidate_pair(
    Fixture& fixture, const Execution& execution) noexcept {
  return static_cast<cudaError_t>(q3x::kernels::
      launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_fused_m128_shared_a_test_cuda(
          fixture.gate_packed.get(), fixture.gate_scales.get(), 1.0F,
          fixture.up_packed.get(), fixture.up_scales.get(), 1.0F,
          fixture.activations.get(), fixture.token_count, kRows, kColumns,
          fixture.gate_output(), fixture.up_output(),
          static_cast<void*>(execution.main())));
}

[[nodiscard]] cudaError_t launch_variant(
    Fixture& fixture, const Execution& execution,
    const Variant variant) noexcept {
  return variant == Variant::kBaseline
             ? launch_baseline_pair(fixture, execution)
             : launch_candidate_pair(fixture, execution);
}

[[nodiscard]] bool run_resource_gate(TestContext& test) {
  bool complete = true;
  for (const std::size_t token_count : {256U, 512U}) {
    int registers = -1;
    std::size_t static_shared = std::numeric_limits<std::size_t>::max();
    std::size_t dynamic_shared = std::numeric_limits<std::size_t>::max();
    std::size_t local = std::numeric_limits<std::size_t>::max();
    int threads = -1;
    int active = -1;
    const int status = q3x::kernels::
        query_sm87_nvfp4_w4a16_whole_chunk_gate_up_fused_m128_shared_a_resources_test_cuda(
            token_count, kRows, kColumns, &registers, &static_shared,
            &dynamic_shared, &local, &threads, &active);
    const bool gate =
        status == static_cast<int>(cudaSuccess) && registers <= 128 &&
        static_shared == 37'376U && dynamic_shared == 18'432U &&
        static_shared + dynamic_shared == 55'808U && local == 0U &&
        threads == 512 && active >= 1;
    std::cout << "FUSED_SHARED_A_RESOURCES: tokens=" << token_count
              << " status=" << status << " registers=" << registers
              << " static_shared_bytes=" << static_shared
              << " dynamic_shared_bytes=" << dynamic_shared
              << " total_shared_bytes=" << static_shared + dynamic_shared
              << " local_bytes=" << local << " threads=" << threads
              << " active_blocks_per_sm=" << active
              << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
    test.expect(gate, "fused shared-A candidate clears resource gate");
    complete = complete && gate;
  }
  return complete;
}

[[nodiscard]] bool run_invalid_graph_gate(TestContext& test,
                                          const cudaStream_t stream) {
  constexpr std::size_t kTokens = 512U;
  constexpr std::uintptr_t kGateWeightAddress = 0x1'0000'0000ULL;
  constexpr std::uintptr_t kGateScaleAddress = 0x2'0000'0000ULL;
  constexpr std::uintptr_t kUpWeightAddress = 0x3'0000'0000ULL;
  constexpr std::uintptr_t kUpScaleAddress = 0x4'0000'0000ULL;
  constexpr std::uintptr_t kActivationAddress = 0x5'0000'0000ULL;
  constexpr std::uintptr_t kGateOutputAddress = 0x6'0000'0000ULL;
  constexpr std::uintptr_t kUpOutputAddress = 0x7'0000'0000ULL;
  constexpr std::uintptr_t kMaximumAddress =
      std::numeric_limits<std::uintptr_t>::max();
  const auto* const gate_weights =
      reinterpret_cast<const std::uint8_t*>(kGateWeightAddress);
  const auto* const gate_scales =
      reinterpret_cast<const std::uint8_t*>(kGateScaleAddress);
  const auto* const up_weights =
      reinterpret_cast<const std::uint8_t*>(kUpWeightAddress);
  const auto* const up_scales =
      reinterpret_cast<const std::uint8_t*>(kUpScaleAddress);
  const auto* const activations =
      reinterpret_cast<const std::uint16_t*>(kActivationAddress);
  auto* const gate_output =
      reinterpret_cast<std::uint16_t*>(kGateOutputAddress);
  auto* const up_output = reinterpret_cast<std::uint16_t*>(kUpOutputAddress);
  auto* const activation_alias = reinterpret_cast<std::uint16_t*>(
      kActivationAddress + 128U * kColumns * sizeof(std::uint16_t));
  auto* const gate_weight_alias =
      reinterpret_cast<std::uint16_t*>(kGateWeightAddress + 128U);
  auto* const gate_scale_alias =
      reinterpret_cast<std::uint16_t*>(kGateScaleAddress + 128U);
  auto* const up_weight_alias =
      reinterpret_cast<std::uint16_t*>(kUpWeightAddress + 128U);
  auto* const up_scale_alias =
      reinterpret_cast<std::uint16_t*>(kUpScaleAddress + 128U);
  const auto* const wrapping_weight =
      reinterpret_cast<const std::uint8_t*>(kMaximumAddress - 15U);
  const auto* const wrapping_scale =
      reinterpret_cast<const std::uint8_t*>(kMaximumAddress - 1U);
  const auto* const wrapping_activation =
      reinterpret_cast<const std::uint16_t*>(kMaximumAddress - 7U);
  auto* const wrapping_output =
      reinterpret_cast<std::uint16_t*>(kMaximumAddress - 1U);

  const auto launch =
      [&](const std::uint8_t* const gw, const std::uint8_t* const gs,
          const float gscale, const std::uint8_t* const uw,
          const std::uint8_t* const us, const float uscale,
          const std::uint16_t* const a, const std::size_t tokens,
          const std::size_t rows, const std::size_t columns,
          std::uint16_t* const go, std::uint16_t* const uo) noexcept {
        return q3x::kernels::
            launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_fused_m128_shared_a_test_cuda(
                gw, gs, gscale, uw, us, uscale, a, tokens, rows, columns, go,
                uo, static_cast<void*>(stream));
      };

  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "invalid begin capture");
  std::array<int, 34U> statuses{};
  statuses.fill(static_cast<int>(cudaErrorUnknown));
  if (ready) {
    statuses[0] = launch(nullptr, gate_scales, 1.0F, up_weights, up_scales,
                         1.0F, activations, kTokens, kRows, kColumns,
                         gate_output, up_output);
    statuses[1] = launch(gate_weights, nullptr, 1.0F, up_weights, up_scales,
                         1.0F, activations, kTokens, kRows, kColumns,
                         gate_output, up_output);
    statuses[2] = launch(gate_weights, gate_scales, 1.0F, nullptr, up_scales,
                         1.0F, activations, kTokens, kRows, kColumns,
                         gate_output, up_output);
    statuses[3] = launch(gate_weights, gate_scales, 1.0F, up_weights, nullptr,
                         1.0F, activations, kTokens, kRows, kColumns,
                         gate_output, up_output);
    statuses[4] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                         up_scales, 1.0F, nullptr, kTokens, kRows, kColumns,
                         gate_output, up_output);
    statuses[5] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                         up_scales, 1.0F, activations, kTokens, kRows,
                         kColumns, nullptr, up_output);
    statuses[6] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                         up_scales, 1.0F, activations, kTokens, kRows,
                         kColumns, gate_output, nullptr);
    statuses[7] = launch(
        gate_weights, gate_scales,
        std::numeric_limits<float>::quiet_NaN(), up_weights, up_scales, 1.0F,
        activations, kTokens, kRows, kColumns, gate_output, up_output);
    statuses[8] = launch(gate_weights, gate_scales, -1.0F, up_weights,
                         up_scales, 1.0F, activations, kTokens, kRows,
                         kColumns, gate_output, up_output);
    statuses[9] = launch(
        gate_weights, gate_scales, 1.0F, up_weights, up_scales,
        std::numeric_limits<float>::quiet_NaN(), activations, kTokens, kRows,
        kColumns, gate_output, up_output);
    statuses[10] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, -1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, up_output);
    statuses[11] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, 128U, kRows, kColumns,
                          gate_output, up_output);
    statuses[12] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, 513U, kRows, kColumns,
                          gate_output, up_output);
    statuses[13] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows - 1U,
                          kColumns, gate_output, up_output);
    statuses[14] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns - 16U, gate_output, up_output);
    statuses[15] = launch(gate_weights + 1U, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, up_output);
    statuses[16] = launch(gate_weights, gate_scales + 1U, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, up_output);
    statuses[17] = launch(gate_weights, gate_scales, 1.0F, up_weights + 1U,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, up_output);
    statuses[18] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales + 1U, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, up_output);
    statuses[19] = launch(
        gate_weights, gate_scales, 1.0F, up_weights, up_scales, 1.0F,
        reinterpret_cast<const std::uint16_t*>(kActivationAddress + 2U),
        kTokens, kRows, kColumns, gate_output, up_output);
    statuses[20] = launch(
        gate_weights, gate_scales, 1.0F, up_weights, up_scales, 1.0F,
        activations, kTokens, kRows, kColumns,
        reinterpret_cast<std::uint16_t*>(kGateOutputAddress + 1U), up_output);
    statuses[21] = launch(
        gate_weights, gate_scales, 1.0F, up_weights, up_scales, 1.0F,
        activations, kTokens, kRows, kColumns, gate_output,
        reinterpret_cast<std::uint16_t*>(kUpOutputAddress + 1U));
    statuses[22] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, activation_alias, up_output);
    statuses[23] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, activation_alias);
    statuses[24] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, gate_output);
    statuses[25] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_weight_alias, up_output);
    statuses[26] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, up_weight_alias, up_output);
    statuses[27] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, gate_scale_alias);
    statuses[28] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, up_scale_alias);
    statuses[29] = launch(wrapping_weight, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, up_output);
    statuses[30] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          wrapping_scale, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, up_output);
    statuses[31] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, wrapping_activation, kTokens,
                          kRows, kColumns, gate_output, up_output);
    statuses[32] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, wrapping_output, up_output);
    statuses[33] = launch(gate_weights, gate_scales, 1.0F, up_weights,
                          up_scales, 1.0F, activations, kTokens, kRows,
                          kColumns, gate_output, wrapping_output);
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
  std::cout << "FUSED_SHARED_A_INVALID_GRAPH: invalid_statuses="
            << invalid_count << '/' << statuses.size()
            << " graph_nodes=" << nodes
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "all invalid fused calls enqueue zero graph nodes");
  return gate;
}

[[nodiscard]] bool scrub_l2(TestContext& test, Fixture& fixture,
                            const Execution& execution,
                            const std::string& label) {
  scrub_l2_kernel<<<256U, 256U, 0U, execution.main()>>>(
      fixture.scrub.get(), kScrubBytes / sizeof(std::uint32_t));
  bool ready = test.cuda_ok(cudaGetLastError(), label + " launch L2 scrub");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                label + " synchronize L2 scrub");
  return ready;
}

[[nodiscard]] bool poison_outputs(TestContext& test, Fixture& fixture,
                                  const Execution& execution,
                                  const int byte,
                                  const std::string& label) {
  const std::size_t bytes =
      fixture.guarded_output_elements() * sizeof(std::uint16_t);
  bool ready = test.cuda_ok(
      cudaMemsetAsync(fixture.gate_output_store.get(), byte, bytes,
                      execution.main()),
      label + " poison Gate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(fixture.up_output_store.get(), byte,
                                       bytes, execution.main()),
                       label + " poison Up output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                label + " poison synchronize");
  return ready;
}

[[nodiscard]] bool copy_outputs(TestContext& test, const Fixture& fixture,
                                const Execution& execution,
                                std::vector<std::uint16_t>& gate,
                                std::vector<std::uint16_t>& up,
                                const std::string& label) {
  gate.resize(fixture.guarded_output_elements());
  up.resize(fixture.guarded_output_elements());
  const std::size_t bytes = gate.size() * sizeof(std::uint16_t);
  bool ready = test.cuda_ok(
      cudaMemcpyAsync(gate.data(), fixture.gate_output_store.get(), bytes,
                      cudaMemcpyDeviceToHost, execution.main()),
      label + " copy Gate output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(up.data(),
                                       fixture.up_output_store.get(), bytes,
                                       cudaMemcpyDeviceToHost,
                                       execution.main()),
                       label + " copy Up output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                label + " copy outputs synchronize");
  return ready;
}

[[nodiscard]] bool capture_candidate_graph(TestContext& test,
                                           Fixture& fixture,
                                           const Execution& execution,
                                           CapturedGraph& captured,
                                           std::size_t& node_count) {
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(execution.main(),
                             cudaStreamCaptureModeThreadLocal),
      "valid candidate begin capture");
  if (ready) {
    ready = test.cuda_ok(launch_candidate_pair(fixture, execution),
                         "valid candidate capture launch") &&
            ready;
  }
  if (ready) {
    ready = test.cuda_ok(
                cudaStreamEndCapture(execution.main(),
                                     captured.graph_address()),
                "valid candidate end capture") &&
            ready;
  }
  std::array<cudaGraphNode_t, 2U> nodes{};
  std::size_t capacity = nodes.size();
  if (ready) {
    ready = test.cuda_ok(
                cudaGraphGetNodes(captured.graph(), nodes.data(), &capacity),
                "valid candidate get graph nodes") &&
            ready;
    node_count = capacity;
  }
  cudaGraphNodeType node_type = cudaGraphNodeTypeEmpty;
  cudaKernelNodeParams parameters{};
  if (ready && node_count == 1U) {
    ready = test.cuda_ok(cudaGraphNodeGetType(nodes[0], &node_type),
                         "valid candidate read node type") &&
            ready;
    if (ready && node_type == cudaGraphNodeTypeKernel) {
      ready = test.cuda_ok(
                  cudaGraphKernelNodeGetParams(nodes[0], &parameters),
                  "valid candidate read kernel parameters") &&
              ready;
    }
  }
  if (ready) {
    ready = test.cuda_ok(
                cudaGraphInstantiate(captured.executable_address(),
                                     captured.graph(), nullptr, nullptr, 0U),
                "valid candidate instantiate graph") &&
            ready;
  }
  const unsigned int expected_grid =
      fixture.token_count == 256U ? 272U : 544U;
  const bool identity =
      ready && node_count == 1U && node_type == cudaGraphNodeTypeKernel &&
      parameters.gridDim.x == expected_grid && parameters.gridDim.y == 1U &&
      parameters.gridDim.z == 1U && parameters.blockDim.x == 512U &&
      parameters.blockDim.y == 1U && parameters.blockDim.z == 1U &&
      parameters.sharedMemBytes == 18'432U && parameters.func != nullptr;
  std::cout << "FUSED_SHARED_A_VALID_GRAPH: tokens=" << fixture.token_count
            << " graph_nodes=" << node_count
            << " grid_x=" << parameters.gridDim.x
            << " block_x=" << parameters.blockDim.x
            << " dynamic_shared_bytes=" << parameters.sharedMemBytes
            << " instantiate=" << (ready ? "success" : "failure")
            << " gate=" << (identity ? "PASS" : "FAIL") << '\n';
  test.expect(identity,
              "valid candidate graph contains exactly the fused kernel");
  return identity;
}

[[nodiscard]] bool verify_immutable_inputs(TestContext& test,
                                           const Fixture& fixture,
                                           const Execution& execution) {
  std::vector<std::uint8_t> gate_packed(kPackedBytes);
  std::vector<std::uint8_t> gate_scales(kScaleBytes);
  std::vector<std::uint8_t> up_packed(kPackedBytes);
  std::vector<std::uint8_t> up_scales(kScaleBytes);
  std::vector<std::uint16_t> activations(fixture.host_activations.size());
  const cudaStream_t stream = execution.main();
  bool ready = test.cuda_ok(
      cudaMemcpyAsync(gate_packed.data(), fixture.gate_packed.get(),
                      kPackedBytes, cudaMemcpyDeviceToHost, stream),
      "copy immutable Gate weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(gate_scales.data(),
                                       fixture.gate_scales.get(), kScaleBytes,
                                       cudaMemcpyDeviceToHost, stream),
                       "copy immutable Gate scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(up_packed.data(),
                                       fixture.up_packed.get(), kPackedBytes,
                                       cudaMemcpyDeviceToHost, stream),
                       "copy immutable Up weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(up_scales.data(),
                                       fixture.up_scales.get(), kScaleBytes,
                                       cudaMemcpyDeviceToHost, stream),
                       "copy immutable Up scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.data(), fixture.activations.get(),
                           activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       "copy immutable activations");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "immutable inputs synchronize");
  const bool gate = ready && gate_packed == fixture.host_gate_packed &&
                    gate_scales == fixture.host_gate_scales &&
                    up_packed == fixture.host_up_packed &&
                    up_scales == fixture.host_up_scales &&
                    activations == fixture.host_activations;
  std::cout << "FUSED_SHARED_A_IMMUTABLE: tokens=" << fixture.token_count
            << " canonical_and_A_preserved=" << (gate ? "true" : "false")
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "candidate preserves both canonical tensors and A");
  return gate;
}

[[nodiscard]] bool run_correctness(TestContext& test, Fixture& fixture,
                                   const Execution& execution) {
  std::vector<std::uint16_t> baseline_gate;
  std::vector<std::uint16_t> baseline_up;
  std::vector<std::uint16_t> candidate_gate;
  std::vector<std::uint16_t> candidate_up;
  std::vector<std::uint16_t> replay1_gate;
  std::vector<std::uint16_t> replay1_up;
  std::vector<std::uint16_t> replay2_gate;
  std::vector<std::uint16_t> replay2_up;

  bool ready = poison_outputs(test, fixture, execution, 0x3c,
                              "correctness baseline");
  ready = ready && test.cuda_ok(launch_baseline_pair(fixture, execution),
                                "correctness baseline pair");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                "correctness baseline synchronize");
  ready = ready && copy_outputs(test, fixture, execution, baseline_gate,
                                baseline_up, "correctness baseline");

  ready = ready && poison_outputs(test, fixture, execution, 0xa5,
                                  "correctness candidate");
  ready = ready && test.cuda_ok(launch_candidate_pair(fixture, execution),
                                "correctness candidate pair");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                "correctness candidate synchronize");
  ready = ready && copy_outputs(test, fixture, execution, candidate_gate,
                                candidate_up, "correctness candidate");

  CapturedGraph captured;
  std::size_t graph_nodes = 0U;
  ready = ready && capture_candidate_graph(test, fixture, execution, captured,
                                           graph_nodes);
  ready = ready && poison_outputs(test, fixture, execution, 0x5a,
                                  "correctness graph replay 1");
  ready = ready && test.cuda_ok(
                       cudaGraphLaunch(captured.executable(),
                                       execution.main()),
                       "correctness launch candidate graph replay 1");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                "correctness graph replay 1 synchronize");
  ready = ready && copy_outputs(test, fixture, execution, replay1_gate,
                                replay1_up, "correctness graph replay 1");

  ready = ready && poison_outputs(test, fixture, execution, 0x69,
                                  "correctness graph replay 2");
  ready = ready && test.cuda_ok(
                       cudaGraphLaunch(captured.executable(),
                                       execution.main()),
                       "correctness launch candidate graph replay 2");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                "correctness graph replay 2 synchronize");
  ready = ready && copy_outputs(test, fixture, execution, replay2_gate,
                                replay2_up, "correctness graph replay 2");

  std::size_t candidate_mismatches = 0U;
  std::size_t replay1_mismatches = 0U;
  std::size_t replay2_mismatches = 0U;
  std::size_t unexpected_nonfinite = 0U;
  bool guards = ready;
  if (ready) {
    for (std::size_t index = 0U; index < fixture.output_elements(); ++index) {
      const std::size_t guarded = kGuardElements + index;
      candidate_mismatches +=
          baseline_gate[guarded] != candidate_gate[guarded] ? 1U : 0U;
      candidate_mismatches +=
          baseline_up[guarded] != candidate_up[guarded] ? 1U : 0U;
      replay1_mismatches +=
          candidate_gate[guarded] != replay1_gate[guarded] ? 1U : 0U;
      replay1_mismatches +=
          candidate_up[guarded] != replay1_up[guarded] ? 1U : 0U;
      replay2_mismatches +=
          candidate_gate[guarded] != replay2_gate[guarded] ? 1U : 0U;
      replay2_mismatches +=
          candidate_up[guarded] != replay2_up[guarded] ? 1U : 0U;
      unexpected_nonfinite +=
          !is_bf16_finite(baseline_gate[guarded]) ||
                  !is_bf16_finite(baseline_up[guarded]) ||
                  !is_bf16_finite(candidate_gate[guarded]) ||
                  !is_bf16_finite(candidate_up[guarded]) ||
                  !is_bf16_finite(replay1_gate[guarded]) ||
                  !is_bf16_finite(replay1_up[guarded]) ||
                  !is_bf16_finite(replay2_gate[guarded]) ||
                  !is_bf16_finite(replay2_up[guarded])
              ? 1U
              : 0U;
    }
    for (std::size_t index = 0U; index < kGuardElements; ++index) {
      const std::size_t tail =
          kGuardElements + fixture.output_elements() + index;
      guards = guards && baseline_gate[index] == 0x3c3cU &&
               baseline_gate[tail] == 0x3c3cU &&
               baseline_up[index] == 0x3c3cU &&
               baseline_up[tail] == 0x3c3cU &&
               candidate_gate[index] == 0xa5a5U &&
               candidate_gate[tail] == 0xa5a5U &&
               candidate_up[index] == 0xa5a5U &&
               candidate_up[tail] == 0xa5a5U &&
               replay1_gate[index] == 0x5a5aU &&
               replay1_gate[tail] == 0x5a5aU &&
               replay1_up[index] == 0x5a5aU &&
               replay1_up[tail] == 0x5a5aU &&
               replay2_gate[index] == 0x6969U &&
               replay2_gate[tail] == 0x6969U &&
               replay2_up[index] == 0x6969U &&
               replay2_up[tail] == 0x6969U;
    }
  }
  const bool immutable = ready &&
                         verify_immutable_inputs(test, fixture, execution);
  const bool gate = ready && candidate_mismatches == 0U &&
                    replay1_mismatches == 0U && replay2_mismatches == 0U &&
                    unexpected_nonfinite == 0U && guards && immutable &&
                    graph_nodes == 1U;
  std::cout << "FUSED_SHARED_A_CORRECTNESS: tokens=" << fixture.token_count
            << " baseline_candidate_mismatches=" << candidate_mismatches
            << '/' << 2U * fixture.output_elements()
            << " replay1_mismatches=" << replay1_mismatches << '/'
            << 2U * fixture.output_elements()
            << " replay2_mismatches=" << replay2_mismatches << '/'
            << 2U * fixture.output_elements()
            << " graph_replays=2 unexpected_nonfinite="
            << unexpected_nonfinite
            << " output_guards=" << (guards ? "intact" : "BAD")
            << " immutable=" << (immutable ? "true" : "false")
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate,
              "baseline, candidate, and two graph replays are bitwise exact");
  return gate;
}

[[nodiscard]] float measure_pass(TestContext& test, Fixture& fixture,
                                 const Execution& execution,
                                 const Variant variant,
                                 const std::string& label,
                                 const int warmups = kWarmups,
                                 const int iterations = kIterations,
                                 const bool profiler_range = false) {
  bool ready = scrub_l2(test, fixture, execution, label);
  for (int warmup = 0; ready && warmup < warmups; ++warmup) {
    ready = test.cuda_ok(launch_variant(fixture, execution, variant),
                         label + " warmup pair") &&
            ready;
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                label + " warmup synchronize");
  if (ready && profiler_range) {
    ready = test.cuda_ok(cudaProfilerStart(), label + " start profiler range");
  }
  const auto wall_start = std::chrono::steady_clock::now();
  ready = ready && test.cuda_ok(
                       cudaEventRecord(execution.start(), execution.main()),
                       label + " record timing start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = test.cuda_ok(launch_variant(fixture, execution, variant),
                         label + " measured pair") &&
            ready;
  }
  ready = ready && test.cuda_ok(
                       cudaEventRecord(execution.stop(), execution.main()),
                       label + " record timing stop");
  ready = ready && test.cuda_ok(cudaEventSynchronize(execution.stop()),
                                label + " synchronize timing stop");
  const auto wall_stop = std::chrono::steady_clock::now();
  if (profiler_range) {
    ready = test.cuda_ok(cudaProfilerStop(), label + " stop profiler range") &&
            ready;
  }
  float total_ms = std::numeric_limits<float>::quiet_NaN();
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_ms, execution.start(),
                                            execution.stop()),
                       label + " elapsed time");
  const float average_ms =
      ready && iterations > 0
          ? total_ms / static_cast<float>(iterations)
          : std::numeric_limits<float>::quiet_NaN();
  const double wall_ms =
      std::chrono::duration<double, std::milli>(wall_stop - wall_start)
          .count();
  std::cout << "FUSED_SHARED_A_PASS: label=" << label
            << " variant=" << variant_name(variant)
            << " tokens=" << fixture.token_count
            << " warmups=" << warmups << " iterations=" << iterations
            << " scrub_bytes_outside_timing=" << kScrubBytes
            << " profiler_range=" << (profiler_range ? "true" : "false")
            << " average_pair_ms=" << average_ms
            << " host_wall_average_ms="
            << (iterations > 0 ? wall_ms / static_cast<double>(iterations)
                               : std::numeric_limits<double>::quiet_NaN())
            << " baseline_launches_per_pair=2"
            << " candidate_launches_per_pair=1\n";
  return average_ms;
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
    every_round = every_round && finite && speedup > 1.0;
    if (finite) {
      baseline_sum += static_cast<double>(b1 + b2);
      candidate_sum += static_cast<double>(c1 + c2);
    }
    std::cout << "PERF_FUSED_SHARED_A_ROUND: tokens=" << fixture.token_count
              << " round=" << round + 1 << " order=B-C-C-B"
              << " B1_ms=" << b1 << " C1_ms=" << c1
              << " C2_ms=" << c2 << " B2_ms=" << b2
              << " speedup=" << speedup
              << " strict_positive_gate="
              << (finite && speedup > 1.0 ? "PASS" : "FAIL") << '\n';
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
  const bool gate = fixture.token_count == 512U && every_round &&
                    std::isfinite(speedup) &&
                    speedup >= kRequiredC512Speedup;
  std::cout << "PERF_FUSED_SHARED_A_AGGREGATE: tokens=" << fixture.token_count
            << " baseline_pair_ms=" << baseline_ms
            << " candidate_pair_ms=" << candidate_ms
            << " speedup=" << speedup
            << " required_speedup=" << kRequiredC512Speedup
            << " every_round_strict_positive="
            << (every_round ? "true" : "false")
            << " rounds=" << kRounds
            << " warmups_per_pass=" << kWarmups
            << " iterations_per_pass=" << kIterations
            << " order=B-C-C-B canonical_layout=true"
            << " baseline=main_aux_production_pair"
            << " candidate=single_main_stream_fused_kernel"
            << " execution=eager_direct"
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "C512 fused shared-A clears frozen 1.22x pair gate");
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
  std::size_t token_count = 512U;
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
    } else if (argument == "--tokens=256") {
      options.token_count = 256U;
    } else if (argument == "--tokens=512") {
      options.token_count = 512U;
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return false;
    }
  }
  if (options.mode == Mode::kScreen && options.token_count != 512U) {
    std::cerr << "screen mode is frozen to --tokens=512\n";
    return false;
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
    std::cout << "SKIP: fused shared-A Gate/Up screen requires CUDA\n";
    (void)cudaGetLastError();
    return 77;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                    "read CUDA device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: fused shared-A Gate/Up requires SM87; got sm_"
              << properties.major << properties.minor << '\n';
    return 77;
  }
  std::cout << std::fixed << std::setprecision(6)
            << "FUSED_SHARED_A_DEVICE: name=" << properties.name
            << " sm=" << properties.major << properties.minor
            << " sm_count=" << properties.multiProcessorCount
            << " l2_bytes=" << properties.l2CacheSize
            << " mode=" << mode_name(options.mode)
            << " tokens=" << options.token_count
            << " canonical_layout=true shared_A=true"
            << " production_dispatch_unchanged=true\n";

  Execution execution;
  if (!execution.create(test)) {
    return 1;
  }
  bool ready = run_resource_gate(test);
  ready = run_invalid_graph_gate(test, execution.main()) && ready;
  if (!ready) {
    return 1;
  }

  Fixture fixture;
  if (!fixture.initialize(test, execution, options.token_count)) {
    return 1;
  }
  const bool correct = run_correctness(test, fixture, execution);
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
            test, fixture, execution, variant, "single_pair_profile", 0, 1,
            true);
        std::cout << "FUSED_SHARED_A_PROFILE_MARKER: mode="
                  << mode_name(options.mode)
                  << " tokens=" << options.token_count
                  << " pair_ms=" << milliseconds
                  << " scrub_launches_in_range=0"
                  << " profiler_range_kernel_launches="
                  << (variant == Variant::kBaseline ? 2 : 1)
                  << " ncu_replay_mode=app-range"
                  << " ncu_profile_from_start=false"
                  << " kernel_replay_forbidden=true\n";
        break;
      }
    }
  }

  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " fused shared-A Gate/Up assertion(s) failed\n";
    return 1;
  }
  std::cout << "fused shared-A Gate/Up SM87 screen passed\n";
  return 0;
}
