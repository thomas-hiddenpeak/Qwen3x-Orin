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

// These four hooks are test-only ABI.  They deliberately stay out of the
// installed public header until the register-fed candidate clears this
// standalone screen.
[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gate_m128_register_fed_sidecar_build_test_cuda(
    const std::uint8_t* canonical_weights,
    const std::uint8_t* canonical_scales, std::size_t rows,
    std::size_t columns, std::uint32_t* sidecar_weights,
    std::uint16_t* sidecar_scales, void* stream) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_whole_chunk_gate_m128_register_fed_test_cuda(
    const std::uint32_t* sidecar_weights,
    const std::uint16_t* sidecar_scales, float weight_scale_2,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* stream) noexcept;

[[nodiscard]] int
query_sm87_nvfp4_w4a16_whole_chunk_gate_m128_register_fed_resources_test_cuda(
    std::size_t token_count, std::size_t rows, std::size_t columns,
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gate_m128_register_fed_x4_exhaustive_test_cuda(
    std::uint32_t* device_mismatch_count, void* stream) noexcept;

}  // namespace q3x::kernels

namespace {

constexpr std::size_t kRows = 17'408U;
constexpr std::size_t kColumns = 5'120U;
constexpr std::size_t kPackedRowBytes = kColumns / 2U;
constexpr std::size_t kScaleColumns = kColumns / 16U;
constexpr std::size_t kPackedBytes = kRows * kPackedRowBytes;
constexpr std::size_t kScaleBytes = kRows * kScaleColumns;
constexpr std::size_t kN128Blocks = kRows / 128U;
constexpr std::size_t kK16Blocks = kColumns / 16U;
constexpr std::size_t kWarps = 8U;
constexpr std::size_t kLanes = 32U;
constexpr std::size_t kScaleGroupsPerWarp = 8U;
constexpr std::size_t kSidecarWeightCount =
    kN128Blocks * kK16Blocks * kWarps * kLanes;
constexpr std::size_t kSidecarScaleCount =
    kN128Blocks * kK16Blocks * kWarps * kScaleGroupsPerWarp;
constexpr std::size_t kSidecarWeightBytes =
    kSidecarWeightCount * sizeof(std::uint32_t);
constexpr std::size_t kSidecarScaleBytes =
    kSidecarScaleCount * sizeof(std::uint16_t);
constexpr std::size_t kGuardElements = 64U;
constexpr std::size_t kScrubBytes = 32U * 1024U * 1024U;
constexpr int kWarmups = 10;
constexpr int kIterations = 24;
constexpr int kRounds = 6;
constexpr double kRequiredGateSpeedup = 1.22;
constexpr double kRequiredPairSpeedup = 1.20;
constexpr std::uint32_t kSidecarWeightPoison = 0xcdcd'cdcdU;
constexpr std::uint16_t kSidecarScalePoison = 0xcdcdU;
constexpr std::array<std::uint8_t, 32U> kCheckpointLikeScaleCodes{{
    0x4eU, 0x50U, 0x52U, 0x54U, 0x55U, 0x56U, 0x57U, 0x58U,
    0x58U, 0x58U, 0x59U, 0x59U, 0x59U, 0x5aU, 0x5aU, 0x5bU,
    0x5bU, 0x5cU, 0x5cU, 0x5dU, 0x5dU, 0x5eU, 0x5fU, 0x60U,
    0x60U, 0x61U, 0x62U, 0x63U, 0x64U, 0x65U, 0x66U, 0x67U,
}};

static_assert(kN128Blocks == 136U);
static_assert(kK16Blocks == 320U);
static_assert(kSidecarWeightCount == 11'141'120U);
static_assert(kSidecarScaleCount == 2'785'280U);
static_assert(kSidecarWeightBytes == kPackedBytes);
static_assert(kSidecarScaleBytes == kScaleBytes);

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
    ready = ready &&
            test.cuda_ok(cudaEventCreate(&start_), "create timing start");
    ready = ready &&
            test.cuda_ok(cudaEventCreate(&stop_), "create timing stop");
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
  DeviceBuffer<std::uint8_t> up_packed;
  DeviceBuffer<std::uint8_t> gate_scales;
  DeviceBuffer<std::uint8_t> up_scales;
  DeviceBuffer<std::uint32_t> gate_sidecar_weight_store;
  DeviceBuffer<std::uint32_t> up_sidecar_weight_store;
  DeviceBuffer<std::uint16_t> gate_sidecar_scale_store;
  DeviceBuffer<std::uint16_t> up_sidecar_scale_store;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> gate_output_store;
  DeviceBuffer<std::uint16_t> up_output_store;
  DeviceBuffer<std::uint32_t> scrub;
  std::vector<std::uint8_t> host_gate_packed;
  std::vector<std::uint8_t> host_up_packed;
  std::vector<std::uint8_t> host_gate_scales;
  std::vector<std::uint8_t> host_up_scales;
  std::vector<std::uint32_t> host_gate_sidecar_weights;
  std::vector<std::uint32_t> host_up_sidecar_weights;
  std::vector<std::uint16_t> host_gate_sidecar_scales;
  std::vector<std::uint16_t> host_up_sidecar_scales;
  std::vector<std::uint16_t> host_activations;

  [[nodiscard]] std::size_t output_elements() const noexcept {
    return token_count * kRows;
  }

  [[nodiscard]] std::size_t guarded_output_elements() const noexcept {
    return output_elements() + 2U * kGuardElements;
  }

  [[nodiscard]] std::uint32_t* gate_sidecar_weights() noexcept {
    return gate_sidecar_weight_store.get() + kGuardElements;
  }
  [[nodiscard]] std::uint32_t* up_sidecar_weights() noexcept {
    return up_sidecar_weight_store.get() + kGuardElements;
  }
  [[nodiscard]] std::uint16_t* gate_sidecar_scales() noexcept {
    return gate_sidecar_scale_store.get() + kGuardElements;
  }
  [[nodiscard]] std::uint16_t* up_sidecar_scales() noexcept {
    return up_sidecar_scale_store.get() + kGuardElements;
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
    host_up_packed.resize(kPackedBytes);
    host_gate_scales.resize(kScaleBytes);
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
                                      "allocate Gate canonical weights");
    ready = ready && up_packed.allocate(test, kPackedBytes,
                                        "allocate Up canonical weights");
    ready = ready && gate_scales.allocate(test, kScaleBytes,
                                          "allocate Gate canonical scales");
    ready = ready && up_scales.allocate(test, kScaleBytes,
                                        "allocate Up canonical scales");
    ready = ready && gate_sidecar_weight_store.allocate(
                         test, kSidecarWeightCount + 2U * kGuardElements,
                         "allocate guarded Gate weight sidecar");
    ready = ready && up_sidecar_weight_store.allocate(
                         test, kSidecarWeightCount + 2U * kGuardElements,
                         "allocate guarded Up weight sidecar");
    ready = ready && gate_sidecar_scale_store.allocate(
                         test, kSidecarScaleCount + 2U * kGuardElements,
                         "allocate guarded Gate scale sidecar");
    ready = ready && up_sidecar_scale_store.allocate(
                         test, kSidecarScaleCount + 2U * kGuardElements,
                         "allocate guarded Up scale sidecar");
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

    ready = test.cuda_ok(
        cudaMemcpyAsync(gate_packed.get(), host_gate_packed.data(),
                        kPackedBytes, cudaMemcpyHostToDevice,
                        execution.main()),
        "upload Gate canonical weights");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(up_packed.get(),
                                         host_up_packed.data(), kPackedBytes,
                                         cudaMemcpyHostToDevice,
                                         execution.main()),
                         "upload Up canonical weights");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(gate_scales.get(),
                                         host_gate_scales.data(), kScaleBytes,
                                         cudaMemcpyHostToDevice,
                                         execution.main()),
                         "upload Gate canonical scales");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(up_scales.get(),
                                         host_up_scales.data(), kScaleBytes,
                                         cudaMemcpyHostToDevice,
                                         execution.main()),
                         "upload Up canonical scales");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             activations.get(), host_activations.data(),
                             activation_elements * sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice, execution.main()),
                         "upload BF16 activations");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(gate_sidecar_weight_store.get(),
                                         0xcd,
                                         (kSidecarWeightCount +
                                          2U * kGuardElements) *
                                             sizeof(std::uint32_t),
                                         execution.main()),
                         "poison Gate weight sidecar");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(up_sidecar_weight_store.get(), 0xcd,
                                         (kSidecarWeightCount +
                                          2U * kGuardElements) *
                                             sizeof(std::uint32_t),
                                         execution.main()),
                         "poison Up weight sidecar");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(gate_sidecar_scale_store.get(), 0xcd,
                                         (kSidecarScaleCount +
                                          2U * kGuardElements) *
                                             sizeof(std::uint16_t),
                                         execution.main()),
                         "poison Gate scale sidecar");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(up_sidecar_scale_store.get(), 0xcd,
                                         (kSidecarScaleCount +
                                          2U * kGuardElements) *
                                             sizeof(std::uint16_t),
                                         execution.main()),
                         "poison Up scale sidecar");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(scrub.get(), 0, kScrubBytes,
                                         execution.main()),
                         "initialize L2 scrub buffer");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                  "fixture upload synchronize");
    if (!ready) {
      return false;
    }

    // Sidecar construction is a one-time layout transform and is explicitly
    // outside every kernel-performance timing interval below.
    const auto build_start = std::chrono::steady_clock::now();
    ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::kernels::
            launch_sm87_nvfp4_w4a16_gate_m128_register_fed_sidecar_build_test_cuda(
                gate_packed.get(), gate_scales.get(), kRows, kColumns,
                gate_sidecar_weights(), gate_sidecar_scales(),
                static_cast<void*>(execution.main()))),
        "build Gate register-fed sidecars");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(q3x::kernels::
                             launch_sm87_nvfp4_w4a16_gate_m128_register_fed_sidecar_build_test_cuda(
                                 up_packed.get(), up_scales.get(), kRows,
                                 kColumns, up_sidecar_weights(),
                                 up_sidecar_scales(),
                                 static_cast<void*>(execution.main()))),
                         "build Up register-fed sidecars");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                  "sidecar builder synchronize");
    const auto build_stop = std::chrono::steady_clock::now();
    std::cout << "REGISTER_FED_BUILDER: tensors=2 canonical_weight_bytes_each="
              << kPackedBytes << " canonical_scale_bytes_each=" << kScaleBytes
              << " sidecar_weight_bytes_each=" << kSidecarWeightBytes
              << " sidecar_scale_bytes_each=" << kSidecarScaleBytes
              << " host_wall_ms="
              << std::chrono::duration<double, std::milli>(build_stop -
                                                            build_start)
                     .count()
              << " excluded_from_kernel_timing=true\n";
    if (!ready) {
      return false;
    }
    return copy_and_verify_built_sidecars(test, execution.main());
  }

  [[nodiscard]] bool copy_and_verify_built_sidecars(
      TestContext& test, const cudaStream_t stream) {
    std::vector<std::uint32_t> gate_weights_guarded(
        kSidecarWeightCount + 2U * kGuardElements);
    std::vector<std::uint32_t> up_weights_guarded(
        kSidecarWeightCount + 2U * kGuardElements);
    std::vector<std::uint16_t> gate_scales_guarded(
        kSidecarScaleCount + 2U * kGuardElements);
    std::vector<std::uint16_t> up_scales_guarded(
        kSidecarScaleCount + 2U * kGuardElements);
    bool ready = test.cuda_ok(
        cudaMemcpyAsync(gate_weights_guarded.data(),
                        gate_sidecar_weight_store.get(),
                        gate_weights_guarded.size() * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream),
        "copy built Gate weight sidecar");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             up_weights_guarded.data(),
                             up_sidecar_weight_store.get(),
                             up_weights_guarded.size() * sizeof(std::uint32_t),
                             cudaMemcpyDeviceToHost, stream),
                         "copy built Up weight sidecar");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             gate_scales_guarded.data(),
                             gate_sidecar_scale_store.get(),
                             gate_scales_guarded.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         "copy built Gate scale sidecar");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             up_scales_guarded.data(),
                             up_sidecar_scale_store.get(),
                             up_scales_guarded.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         "copy built Up scale sidecar");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  "copy built sidecars synchronize");
    if (!ready) {
      return false;
    }

    bool guards = true;
    for (std::size_t index = 0U; index < kGuardElements; ++index) {
      const std::size_t weight_tail = kGuardElements + kSidecarWeightCount +
                                      index;
      const std::size_t scale_tail = kGuardElements + kSidecarScaleCount +
                                     index;
      guards = guards &&
               gate_weights_guarded[index] == kSidecarWeightPoison &&
               gate_weights_guarded[weight_tail] == kSidecarWeightPoison &&
               up_weights_guarded[index] == kSidecarWeightPoison &&
               up_weights_guarded[weight_tail] == kSidecarWeightPoison &&
               gate_scales_guarded[index] == kSidecarScalePoison &&
               gate_scales_guarded[scale_tail] == kSidecarScalePoison &&
               up_scales_guarded[index] == kSidecarScalePoison &&
               up_scales_guarded[scale_tail] == kSidecarScalePoison;
    }
    host_gate_sidecar_weights.assign(
        gate_weights_guarded.begin() + static_cast<std::ptrdiff_t>(kGuardElements),
        gate_weights_guarded.begin() + static_cast<std::ptrdiff_t>(
                                           kGuardElements +
                                           kSidecarWeightCount));
    host_up_sidecar_weights.assign(
        up_weights_guarded.begin() + static_cast<std::ptrdiff_t>(kGuardElements),
        up_weights_guarded.begin() + static_cast<std::ptrdiff_t>(
                                         kGuardElements +
                                         kSidecarWeightCount));
    host_gate_sidecar_scales.assign(
        gate_scales_guarded.begin() + static_cast<std::ptrdiff_t>(kGuardElements),
        gate_scales_guarded.begin() + static_cast<std::ptrdiff_t>(
                                          kGuardElements +
                                          kSidecarScaleCount));
    host_up_sidecar_scales.assign(
        up_scales_guarded.begin() + static_cast<std::ptrdiff_t>(kGuardElements),
        up_scales_guarded.begin() + static_cast<std::ptrdiff_t>(
                                        kGuardElements +
                                        kSidecarScaleCount));

    const auto verify_tensor = [&](const std::vector<std::uint8_t>& packed,
                                   const std::vector<std::uint8_t>& scales,
                                   const std::vector<std::uint32_t>&
                                       sidecar_weights,
                                   const std::vector<std::uint16_t>&
                                       sidecar_scales,
                                   const char* const name) {
      std::size_t weight_mismatches = 0U;
      std::size_t scale_mismatches = 0U;
      std::size_t first_weight_mismatch = kSidecarWeightCount;
      std::size_t first_scale_mismatch = kSidecarScaleCount;
      for (std::size_t n_block = 0U; n_block < kN128Blocks; ++n_block) {
        for (std::size_t k16 = 0U; k16 < kK16Blocks; ++k16) {
          const std::size_t k_base = k16 * 16U;
          for (std::size_t warp = 0U; warp < kWarps; ++warp) {
            for (std::size_t lane = 0U; lane < kLanes; ++lane) {
              const std::size_t t = lane & 3U;
              const std::size_t group = lane >> 2U;
              const std::size_t n0 =
                  n_block * 128U + warp * 16U + group;
              const std::size_t n1 = n0 + 8U;
              const std::size_t offset0 = k_base / 2U + t;
              const std::size_t offset1 = k_base / 2U + 4U + t;
              const std::uint32_t expected =
                  static_cast<std::uint32_t>(
                      packed[n0 * kPackedRowBytes + offset0]) |
                  (static_cast<std::uint32_t>(
                       packed[n0 * kPackedRowBytes + offset1])
                   << 8U) |
                  (static_cast<std::uint32_t>(
                       packed[n1 * kPackedRowBytes + offset0])
                   << 16U) |
                  (static_cast<std::uint32_t>(
                       packed[n1 * kPackedRowBytes + offset1])
                   << 24U);
              const std::size_t sidecar_index =
                  (((n_block * kK16Blocks + k16) * kWarps + warp) *
                       kLanes +
                   lane);
              if (sidecar_weights[sidecar_index] != expected) {
                if (weight_mismatches == 0U) {
                  first_weight_mismatch = sidecar_index;
                }
                ++weight_mismatches;
              }
            }
            for (std::size_t group = 0U; group < kScaleGroupsPerWarp;
                 ++group) {
              const std::size_t n0 =
                  n_block * 128U + warp * 16U + group;
              const std::size_t n1 = n0 + 8U;
              const std::uint16_t expected =
                  static_cast<std::uint16_t>(
                      scales[n0 * kScaleColumns + k16]) |
                  static_cast<std::uint16_t>(
                      static_cast<std::uint16_t>(
                          scales[n1 * kScaleColumns + k16])
                      << 8U);
              const std::size_t sidecar_index =
                  (((n_block * kK16Blocks + k16) * kWarps + warp) *
                       kScaleGroupsPerWarp +
                   group);
              if (sidecar_scales[sidecar_index] != expected) {
                if (scale_mismatches == 0U) {
                  first_scale_mismatch = sidecar_index;
                }
                ++scale_mismatches;
              }
            }
          }
        }
      }
      const bool exact = weight_mismatches == 0U && scale_mismatches == 0U;
      std::cout << "REGISTER_FED_SIDECAR_LAYOUT: tensor=" << name
                << " layout_weights=[N128][K16][warp8][lane32]_uint32"
                << " layout_scales=[N128][K16][warp8][group8]_uint16"
                << " weight_mismatches=" << weight_mismatches << '/'
                << kSidecarWeightCount
                << " scale_mismatches=" << scale_mismatches << '/'
                << kSidecarScaleCount
                << " first_weight_mismatch=" << first_weight_mismatch
                << " first_scale_mismatch=" << first_scale_mismatch
                << " gate=" << (exact ? "PASS" : "FAIL") << '\n';
      test.expect(exact, std::string(name) +
                             " sidecars exactly match the host layout oracle");
      return exact;
    };

    const bool gate_exact =
        verify_tensor(host_gate_packed, host_gate_scales,
                      host_gate_sidecar_weights, host_gate_sidecar_scales,
                      "Gate");
    const bool up_exact =
        verify_tensor(host_up_packed, host_up_scales,
                      host_up_sidecar_weights, host_up_sidecar_scales, "Up");
    std::cout << "REGISTER_FED_SIDECAR_GUARDS: intact="
              << (guards ? "true" : "false")
              << " gate=" << (guards ? "PASS" : "FAIL") << '\n';
    test.expect(guards, "sidecar builder preserves all allocation guards");
    return gate_exact && up_exact && guards;
  }
};

enum class Variant {
  kBaseline,
  kCandidate,
};

enum class Scope {
  kGate,
  kPair,
};

[[nodiscard]] const char* variant_name(const Variant variant) noexcept {
  return variant == Variant::kBaseline ? "baseline" : "candidate";
}

[[nodiscard]] const char* scope_name(const Scope scope) noexcept {
  return scope == Scope::kGate ? "gate" : "gate_up_pair";
}

[[nodiscard]] cudaError_t launch_branch(
    Fixture& fixture, const Variant variant, const bool up,
    const cudaStream_t stream) noexcept {
  if (variant == Variant::kBaseline) {
    return static_cast<cudaError_t>(q3x::kernels::
        launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
            up ? fixture.up_packed.get() : fixture.gate_packed.get(),
            up ? fixture.up_scales.get() : fixture.gate_scales.get(), 1.0F,
            fixture.activations.get(), fixture.token_count, kRows, kColumns,
            up ? fixture.up_output() : fixture.gate_output(),
            static_cast<void*>(stream)));
  }
  return static_cast<cudaError_t>(q3x::kernels::
      launch_sm87_nvfp4_w4a16_whole_chunk_gate_m128_register_fed_test_cuda(
          up ? fixture.up_sidecar_weights()
             : fixture.gate_sidecar_weights(),
          up ? fixture.up_sidecar_scales() : fixture.gate_sidecar_scales(),
          1.0F, fixture.activations.get(), fixture.token_count, kRows,
          kColumns, up ? fixture.up_output() : fixture.gate_output(),
          static_cast<void*>(stream)));
}

[[nodiscard]] cudaError_t launch_scope(Fixture& fixture,
                                       const Execution& execution,
                                       const Variant variant,
                                       const Scope scope) noexcept {
  if (scope == Scope::kGate) {
    return launch_branch(fixture, variant, false, execution.main());
  }
  cudaError_t status = cudaEventRecord(execution.ready(), execution.main());
  if (status == cudaSuccess) {
    status = cudaStreamWaitEvent(execution.auxiliary(), execution.ready(), 0U);
  }
  if (status == cudaSuccess) {
    status = launch_branch(fixture, variant, false, execution.main());
  }
  if (status == cudaSuccess) {
    status = launch_branch(fixture, variant, true, execution.auxiliary());
  }
  if (status == cudaSuccess) {
    status = cudaEventRecord(execution.done(), execution.auxiliary());
  }
  if (status == cudaSuccess) {
    status = cudaStreamWaitEvent(execution.main(), execution.done(), 0U);
  }
  return status;
}

[[nodiscard]] bool run_resource_gate(TestContext& test) {
  bool complete = true;
  for (const std::size_t token_count : {256U, 512U}) {
    int registers = -1;
    std::size_t shared = std::numeric_limits<std::size_t>::max();
    std::size_t local = std::numeric_limits<std::size_t>::max();
    int threads = -1;
    int active = -1;
    const int status = q3x::kernels::
        query_sm87_nvfp4_w4a16_whole_chunk_gate_m128_register_fed_resources_test_cuda(
            token_count, kRows, kColumns, &registers, &shared, &local,
            &threads, &active);
    const bool gate = status == static_cast<int>(cudaSuccess) &&
                      registers <= 128 && shared == 35'328U && local == 0U &&
                      threads == 256 && active >= 2;
    std::cout << "REGISTER_FED_RESOURCES: tokens=" << token_count
              << " status=" << status << " registers=" << registers
              << " static_shared_bytes=" << shared
              << " local_bytes=" << local << " threads=" << threads
              << " active_blocks_per_sm=" << active
              << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
    test.expect(gate, "register-fed candidate clears frozen resources");
    complete = complete && gate;
  }
  return complete;
}

[[nodiscard]] bool run_x4_exhaustive_gate(TestContext& test,
                                          const cudaStream_t stream) {
  DeviceBuffer<std::uint32_t> mismatch;
  if (!mismatch.allocate(test, 1U, "allocate x4 exhaustive counter")) {
    return false;
  }
  bool ready = test.cuda_ok(
      cudaMemsetAsync(mismatch.get(), 0, sizeof(std::uint32_t), stream),
      "zero x4 exhaustive counter");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_nvfp4_w4a16_gate_m128_register_fed_x4_exhaustive_test_cuda(
                               mismatch.get(), static_cast<void*>(stream))),
                       "launch x4 exhaustive validation");
  std::uint32_t host_mismatch = std::numeric_limits<std::uint32_t>::max();
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(&host_mismatch, mismatch.get(),
                                       sizeof(host_mismatch),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy x4 exhaustive counter");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "x4 exhaustive synchronize");
  const bool gate = ready && host_mismatch == 0U;
  std::cout << "REGISTER_FED_X4_EXHAUSTIVE: mismatch_count="
            << host_mismatch << " gate=" << (gate ? "PASS" : "FAIL")
            << '\n';
  test.expect(gate, "all x4 register-feed values match the reference mapping");
  return gate;
}

[[nodiscard]] bool run_invalid_graph_gate(TestContext& test,
                                          const cudaStream_t stream) {
  constexpr std::size_t kTokens = 512U;
  constexpr std::uintptr_t kWeightAddress = 0x1'0000'0000ULL;
  constexpr std::uintptr_t kScaleAddress = 0x2'0000'0000ULL;
  constexpr std::uintptr_t kActivationAddress = 0x3'0000'0000ULL;
  constexpr std::uintptr_t kOutputAddress = 0x4'0000'0000ULL;
  constexpr std::uintptr_t kMaximumAddress =
      std::numeric_limits<std::uintptr_t>::max();
  const auto* const weights =
      reinterpret_cast<const std::uint32_t*>(kWeightAddress);
  const auto* const scales =
      reinterpret_cast<const std::uint16_t*>(kScaleAddress);
  const auto* const activations =
      reinterpret_cast<const std::uint16_t*>(kActivationAddress);
  auto* const output = reinterpret_cast<std::uint16_t*>(kOutputAddress);
  auto* const weight_alias =
      reinterpret_cast<std::uint16_t*>(kWeightAddress + 256U);
  auto* const scale_alias =
      reinterpret_cast<std::uint16_t*>(kScaleAddress + 256U);
  auto* const activation_alias = reinterpret_cast<std::uint16_t*>(
      kActivationAddress + 128U * kColumns * sizeof(std::uint16_t));
  const auto* const wrapping_weights =
      reinterpret_cast<const std::uint32_t*>(kMaximumAddress - 15U);
  const auto* const wrapping_scales =
      reinterpret_cast<const std::uint16_t*>(kMaximumAddress - 1U);
  const auto* const wrapping_activations =
      reinterpret_cast<const std::uint16_t*>(kMaximumAddress - 7U);
  auto* const wrapping_output =
      reinterpret_cast<std::uint16_t*>(kMaximumAddress - 1U);
  const auto launch = [&](const std::uint32_t* const w,
                          const std::uint16_t* const s, const float scale,
                          const std::uint16_t* const a,
                          const std::size_t tokens, const std::size_t rows,
                          const std::size_t columns,
                          std::uint16_t* const o) noexcept {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_whole_chunk_gate_m128_register_fed_test_cuda(
            w, s, scale, a, tokens, rows, columns, o,
            static_cast<void*>(stream));
  };

  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "invalid begin capture");
  std::array<int, 21U> statuses{};
  statuses.fill(static_cast<int>(cudaErrorUnknown));
  if (ready) {
    statuses[0] = launch(nullptr, scales, 1.0F, activations, kTokens, kRows,
                         kColumns, output);
    statuses[1] = launch(weights, nullptr, 1.0F, activations, kTokens, kRows,
                         kColumns, output);
    statuses[2] = launch(weights, scales, 1.0F, nullptr, kTokens, kRows,
                         kColumns, output);
    statuses[3] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                         kColumns, nullptr);
    statuses[4] = launch(weights, scales,
                         std::numeric_limits<float>::quiet_NaN(), activations,
                         kTokens, kRows, kColumns, output);
    statuses[5] = launch(weights, scales, -1.0F, activations, kTokens, kRows,
                         kColumns, output);
    statuses[6] = launch(weights, scales, 1.0F, activations, 128U, kRows,
                         kColumns, output);
    statuses[7] = launch(weights, scales, 1.0F, activations, 513U, kRows,
                         kColumns, output);
    statuses[8] = launch(weights, scales, 1.0F, activations, kTokens,
                         kRows - 1U, kColumns, output);
    statuses[9] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                         kColumns - 16U, output);
    statuses[10] = launch(
        reinterpret_cast<const std::uint32_t*>(kWeightAddress + 1U), scales,
        1.0F, activations, kTokens, kRows, kColumns, output);
    statuses[11] = launch(
        weights,
        reinterpret_cast<const std::uint16_t*>(kScaleAddress + 1U), 1.0F,
        activations, kTokens, kRows, kColumns, output);
    statuses[12] = launch(
        weights, scales, 1.0F,
        reinterpret_cast<const std::uint16_t*>(kActivationAddress + 2U),
        kTokens, kRows, kColumns, output);
    statuses[13] = launch(
        weights, scales, 1.0F, activations, kTokens, kRows, kColumns,
        reinterpret_cast<std::uint16_t*>(kOutputAddress + 1U));
    statuses[14] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                          kColumns, activation_alias);
    statuses[15] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                          kColumns, weight_alias);
    statuses[16] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                          kColumns, scale_alias);
    statuses[17] = launch(wrapping_weights, scales, 1.0F, activations, kTokens,
                          kRows, kColumns, output);
    statuses[18] = launch(weights, wrapping_scales, 1.0F, activations, kTokens,
                          kRows, kColumns, output);
    statuses[19] = launch(weights, scales, 1.0F, wrapping_activations, kTokens,
                          kRows, kColumns, output);
    statuses[20] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                          kColumns, wrapping_output);
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
  std::cout << "REGISTER_FED_INVALID_GRAPH: invalid_statuses="
            << invalid_count << '/' << statuses.size()
            << " graph_nodes=" << nodes
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "invalid register-fed calls enqueue zero graph nodes");
  return gate;
}

[[nodiscard]] bool scrub_l2(TestContext& test, Fixture& fixture,
                            const Execution& execution,
                            const std::string& label) {
  constexpr unsigned int kThreads = 256U;
  constexpr unsigned int kBlocks = 256U;
  scrub_l2_kernel<<<kBlocks, kThreads, 0U, execution.main()>>>(
      fixture.scrub.get(), kScrubBytes / sizeof(std::uint32_t));
  bool ready = test.cuda_ok(cudaGetLastError(), label + " launch L2 scrub");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                label + " synchronize L2 scrub");
  return ready;
}

[[nodiscard]] bool poison_outputs(TestContext& test, Fixture& fixture,
                                  const Execution& execution, const int byte,
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

[[nodiscard]] bool capture_candidate_pair(TestContext& test,
                                          Fixture& fixture,
                                          const Execution& execution,
                                          CapturedGraph& captured,
                                          std::size_t& node_count) {
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(execution.main(),
                             cudaStreamCaptureModeThreadLocal),
      "valid candidate begin capture");
  if (ready) {
    ready = test.cuda_ok(
                launch_scope(fixture, execution, Variant::kCandidate,
                             Scope::kPair),
                "valid candidate capture pair") &&
            ready;
  }
  if (ready) {
    ready = test.cuda_ok(
                cudaStreamEndCapture(execution.main(),
                                     captured.graph_address()),
                "valid candidate end capture") &&
            ready;
  }
  if (ready) {
    ready = test.cuda_ok(
                cudaGraphGetNodes(captured.graph(), nullptr, &node_count),
                "valid candidate count graph nodes") &&
            ready;
  }
  if (ready) {
    ready = test.cuda_ok(
                cudaGraphInstantiate(captured.executable_address(),
                                     captured.graph(), nullptr, nullptr, 0U),
                "valid candidate instantiate graph") &&
            ready;
  }
  const bool gate = ready && node_count > 0U;
  std::cout << "REGISTER_FED_VALID_GRAPH: graph_nodes=" << node_count
            << " instantiate=" << (ready ? "success" : "failure")
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "valid dual-stream candidate graph captures and instantiates");
  return gate;
}

[[nodiscard]] bool verify_immutable_inputs(TestContext& test,
                                           const Fixture& fixture,
                                           const Execution& execution) {
  std::vector<std::uint8_t> gate_packed(kPackedBytes);
  std::vector<std::uint8_t> up_packed(kPackedBytes);
  std::vector<std::uint8_t> gate_scales(kScaleBytes);
  std::vector<std::uint8_t> up_scales(kScaleBytes);
  std::vector<std::uint16_t> activations(fixture.host_activations.size());
  std::vector<std::uint32_t> gate_weight_store(
      kSidecarWeightCount + 2U * kGuardElements);
  std::vector<std::uint32_t> up_weight_store(
      kSidecarWeightCount + 2U * kGuardElements);
  std::vector<std::uint16_t> gate_scale_store(
      kSidecarScaleCount + 2U * kGuardElements);
  std::vector<std::uint16_t> up_scale_store(
      kSidecarScaleCount + 2U * kGuardElements);
  const cudaStream_t stream = execution.main();
  bool ready = test.cuda_ok(
      cudaMemcpyAsync(gate_packed.data(), fixture.gate_packed.get(),
                      kPackedBytes, cudaMemcpyDeviceToHost, stream),
      "copy immutable Gate canonical weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(up_packed.data(),
                                       fixture.up_packed.get(), kPackedBytes,
                                       cudaMemcpyDeviceToHost, stream),
                       "copy immutable Up canonical weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(gate_scales.data(),
                                       fixture.gate_scales.get(), kScaleBytes,
                                       cudaMemcpyDeviceToHost, stream),
                       "copy immutable Gate canonical scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(up_scales.data(),
                                       fixture.up_scales.get(), kScaleBytes,
                                       cudaMemcpyDeviceToHost, stream),
                       "copy immutable Up canonical scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.data(), fixture.activations.get(),
                           activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       "copy immutable activations");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           gate_weight_store.data(),
                           fixture.gate_sidecar_weight_store.get(),
                           gate_weight_store.size() * sizeof(std::uint32_t),
                           cudaMemcpyDeviceToHost, stream),
                       "copy immutable Gate weight sidecar");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           up_weight_store.data(),
                           fixture.up_sidecar_weight_store.get(),
                           up_weight_store.size() * sizeof(std::uint32_t),
                           cudaMemcpyDeviceToHost, stream),
                       "copy immutable Up weight sidecar");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           gate_scale_store.data(),
                           fixture.gate_sidecar_scale_store.get(),
                           gate_scale_store.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       "copy immutable Gate scale sidecar");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           up_scale_store.data(),
                           fixture.up_sidecar_scale_store.get(),
                           up_scale_store.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       "copy immutable Up scale sidecar");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "immutable inputs synchronize");
  bool sidecar_guards = ready;
  if (ready) {
    for (std::size_t index = 0U; index < kGuardElements; ++index) {
      const std::size_t weight_tail =
          kGuardElements + kSidecarWeightCount + index;
      const std::size_t scale_tail =
          kGuardElements + kSidecarScaleCount + index;
      sidecar_guards = sidecar_guards &&
                       gate_weight_store[index] == kSidecarWeightPoison &&
                       gate_weight_store[weight_tail] == kSidecarWeightPoison &&
                       up_weight_store[index] == kSidecarWeightPoison &&
                       up_weight_store[weight_tail] == kSidecarWeightPoison &&
                       gate_scale_store[index] == kSidecarScalePoison &&
                       gate_scale_store[scale_tail] == kSidecarScalePoison &&
                       up_scale_store[index] == kSidecarScalePoison &&
                       up_scale_store[scale_tail] == kSidecarScalePoison;
    }
  }
  const bool canonical = ready && gate_packed == fixture.host_gate_packed &&
                         up_packed == fixture.host_up_packed &&
                         gate_scales == fixture.host_gate_scales &&
                         up_scales == fixture.host_up_scales;
  const bool activation = ready && activations == fixture.host_activations;
  const bool sidecars =
      ready &&
      std::equal(fixture.host_gate_sidecar_weights.begin(),
                 fixture.host_gate_sidecar_weights.end(),
                 gate_weight_store.begin() +
                     static_cast<std::ptrdiff_t>(kGuardElements)) &&
      std::equal(fixture.host_up_sidecar_weights.begin(),
                 fixture.host_up_sidecar_weights.end(),
                 up_weight_store.begin() +
                     static_cast<std::ptrdiff_t>(kGuardElements)) &&
      std::equal(fixture.host_gate_sidecar_scales.begin(),
                 fixture.host_gate_sidecar_scales.end(),
                 gate_scale_store.begin() +
                     static_cast<std::ptrdiff_t>(kGuardElements)) &&
      std::equal(fixture.host_up_sidecar_scales.begin(),
                 fixture.host_up_sidecar_scales.end(),
                 up_scale_store.begin() +
                     static_cast<std::ptrdiff_t>(kGuardElements));
  const bool gate = canonical && activation && sidecars && sidecar_guards;
  std::cout << "REGISTER_FED_IMMUTABLE: canonical="
            << (canonical ? "true" : "false")
            << " activation=" << (activation ? "true" : "false")
            << " sidecars=" << (sidecars ? "true" : "false")
            << " sidecar_guards=" << (sidecar_guards ? "intact" : "BAD")
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "candidate preserves canonical tensors, sidecars, and A");
  return gate;
}

[[nodiscard]] bool run_correctness(TestContext& test, Fixture& fixture,
                                   const Execution& execution) {
  std::vector<std::uint16_t> baseline_gate;
  std::vector<std::uint16_t> baseline_up;
  std::vector<std::uint16_t> candidate_gate;
  std::vector<std::uint16_t> candidate_up;
  std::vector<std::uint16_t> replay_gate;
  std::vector<std::uint16_t> replay_up;
  bool ready = poison_outputs(test, fixture, execution, 0x3c,
                              "correctness baseline");
  ready = ready && test.cuda_ok(
                       launch_scope(fixture, execution, Variant::kBaseline,
                                    Scope::kPair),
                       "correctness baseline pair");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                "correctness baseline synchronize");
  ready = ready && copy_outputs(test, fixture, execution, baseline_gate,
                                baseline_up, "correctness baseline");

  ready = ready && poison_outputs(test, fixture, execution, 0xa5,
                                  "correctness candidate");
  ready = ready && test.cuda_ok(
                       launch_scope(fixture, execution, Variant::kCandidate,
                                    Scope::kPair),
                       "correctness candidate pair");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                "correctness candidate synchronize");
  ready = ready && copy_outputs(test, fixture, execution, candidate_gate,
                                candidate_up, "correctness candidate");

  CapturedGraph captured;
  std::size_t graph_nodes = 0U;
  ready = ready && capture_candidate_pair(test, fixture, execution, captured,
                                          graph_nodes);
  ready = ready && poison_outputs(test, fixture, execution, 0x5a,
                                  "correctness graph replay");
  ready = ready && test.cuda_ok(
                       cudaGraphLaunch(captured.executable(),
                                       execution.main()),
                       "correctness launch candidate graph");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                "correctness graph synchronize");
  ready = ready && copy_outputs(test, fixture, execution, replay_gate,
                                replay_up, "correctness graph replay");

  std::size_t candidate_mismatches = 0U;
  std::size_t replay_mismatches = 0U;
  std::size_t unexpected_nonfinite = 0U;
  bool guards = ready;
  if (ready) {
    for (std::size_t index = 0U; index < fixture.output_elements(); ++index) {
      const std::size_t guarded_index = kGuardElements + index;
      candidate_mismatches +=
          baseline_gate[guarded_index] != candidate_gate[guarded_index] ? 1U
                                                                        : 0U;
      candidate_mismatches +=
          baseline_up[guarded_index] != candidate_up[guarded_index] ? 1U
                                                                    : 0U;
      replay_mismatches +=
          candidate_gate[guarded_index] != replay_gate[guarded_index] ? 1U
                                                                      : 0U;
      replay_mismatches +=
          candidate_up[guarded_index] != replay_up[guarded_index] ? 1U : 0U;
      unexpected_nonfinite +=
          !is_bf16_finite(baseline_gate[guarded_index]) ||
                  !is_bf16_finite(baseline_up[guarded_index]) ||
                  !is_bf16_finite(candidate_gate[guarded_index]) ||
                  !is_bf16_finite(candidate_up[guarded_index]) ||
                  !is_bf16_finite(replay_gate[guarded_index]) ||
                  !is_bf16_finite(replay_up[guarded_index])
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
               replay_gate[index] == 0x5a5aU &&
               replay_gate[tail] == 0x5a5aU &&
               replay_up[index] == 0x5a5aU && replay_up[tail] == 0x5a5aU;
    }
  }
  const bool immutable = ready && verify_immutable_inputs(test, fixture,
                                                          execution);
  const bool gate = ready && candidate_mismatches == 0U &&
                    replay_mismatches == 0U &&
                    unexpected_nonfinite == 0U && guards && immutable &&
                    graph_nodes > 0U;
  std::cout << "REGISTER_FED_CORRECTNESS: tokens=" << fixture.token_count
            << " baseline_candidate_mismatches=" << candidate_mismatches
            << '/' << 2U * fixture.output_elements()
            << " candidate_graph_replay_mismatches=" << replay_mismatches
            << '/' << 2U * fixture.output_elements()
            << " unexpected_nonfinite=" << unexpected_nonfinite
            << " output_guards=" << (guards ? "intact" : "BAD")
            << " graph_nodes=" << graph_nodes
            << " immutable=" << (immutable ? "true" : "false")
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate,
              "baseline, candidate, and graph replay are bitwise identical");
  return gate;
}

[[nodiscard]] float measure_pass(TestContext& test, Fixture& fixture,
                                 const Execution& execution,
                                 const Variant variant, const Scope scope,
                                 const std::string& label,
                                 const int warmups = kWarmups,
                                 const int iterations = kIterations,
                                 const bool profiler_range = false) {
  // This 32 MiB scrub and all warmups are deliberately outside the timing
  // interval and outside the profiler range.
  bool ready = scrub_l2(test, fixture, execution, label);
  for (int warmup = 0; ready && warmup < warmups; ++warmup) {
    ready = test.cuda_ok(launch_scope(fixture, execution, variant, scope),
                         label + " warmup launch") &&
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
    ready = test.cuda_ok(launch_scope(fixture, execution, variant, scope),
                         label + " measured launch") &&
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
  std::cout << "REGISTER_FED_PASS: label=" << label
            << " variant=" << variant_name(variant)
            << " scope=" << scope_name(scope)
            << " tokens=" << fixture.token_count << " warmups=" << warmups
            << " iterations=" << iterations
            << " scrub_bytes_outside_timing=" << kScrubBytes
            << " builder_outside_timing=true profiler_range="
            << (profiler_range ? "true" : "false")
            << " average_ms=" << average_ms
            << " host_wall_average_ms="
            << (iterations > 0 ? wall_ms / static_cast<double>(iterations)
                               : std::numeric_limits<double>::quiet_NaN())
            << '\n';
  return average_ms;
}

[[nodiscard]] bool run_bccb_screen(TestContext& test, Fixture& fixture,
                                   const Execution& execution,
                                   const Scope scope,
                                   const double required_speedup) {
  double baseline_sum = 0.0;
  double candidate_sum = 0.0;
  bool every_round = true;
  for (int round = 0; round < kRounds; ++round) {
    const std::string prefix = scope_name(scope) +
                               ("_round_" + std::to_string(round + 1) + '_');
    const float b1 = measure_pass(test, fixture, execution,
                                  Variant::kBaseline, scope, prefix + "B1");
    const float c1 = measure_pass(test, fixture, execution,
                                  Variant::kCandidate, scope, prefix + "C1");
    const float c2 = measure_pass(test, fixture, execution,
                                  Variant::kCandidate, scope, prefix + "C2");
    const float b2 = measure_pass(test, fixture, execution,
                                  Variant::kBaseline, scope, prefix + "B2");
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
    std::cout << "PERF_REGISTER_FED_ROUND: scope=" << scope_name(scope)
              << " tokens=" << fixture.token_count << " round=" << round + 1
              << " order=B-C-C-B B1_ms=" << b1 << " C1_ms=" << c1
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
                    std::isfinite(speedup) && speedup >= required_speedup;
  std::cout << "PERF_REGISTER_FED_AGGREGATE: scope=" << scope_name(scope)
            << " tokens=" << fixture.token_count
            << " baseline_ms=" << baseline_ms
            << " candidate_ms=" << candidate_ms << " speedup=" << speedup
            << " required_speedup=" << required_speedup
            << " every_round_strict_positive="
            << (every_round ? "true" : "false") << " rounds=" << kRounds
            << " iterations_per_pass=" << kIterations
            << " warmups_per_pass=" << kWarmups
            << " order=B-C-C-B builder_in_timing=false"
            << " execution=eager_direct"
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, std::string(scope_name(scope)) +
                        " clears the frozen register-fed performance gate");
  return gate;
}

[[nodiscard]] bool run_screen(TestContext& test, Fixture& fixture,
                              const Execution& execution) {
  const bool gate_branch = run_bccb_screen(test, fixture, execution,
                                           Scope::kGate,
                                           kRequiredGateSpeedup);
  if (!gate_branch) {
    std::cout << "PERF_REGISTER_FED_PAIR_SKIPPED: reason="
                 "single_gate_did_not_clear_1.22_stop_loss gate=FAIL\n";
    return false;
  }
  return run_bccb_screen(test, fixture, execution, Scope::kPair,
                         kRequiredPairSpeedup);
}

enum class Mode {
  kValidate,
  kScreen,
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
    std::cout << "SKIP: register-fed Gate screen requires a CUDA device\n";
    (void)cudaGetLastError();
    return 77;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                    "read CUDA device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: register-fed Gate screen requires SM87; got sm_"
              << properties.major << properties.minor << '\n';
    return 77;
  }
  std::cout << std::fixed << std::setprecision(6)
            << "REGISTER_FED_DEVICE: name=" << properties.name
            << " sm=" << properties.major << properties.minor
            << " sm_count=" << properties.multiProcessorCount
            << " l2_bytes=" << properties.l2CacheSize
            << " mode=" << mode_name(options.mode)
            << " tokens=" << options.token_count
            << " canonical_bytes_per_tensor="
            << kPackedBytes + kScaleBytes
            << " sidecar_bytes_per_tensor="
            << kSidecarWeightBytes + kSidecarScaleBytes
            << " same_byte_layout=true\n";

  Execution execution;
  if (!execution.create(test)) {
    return 1;
  }
  bool ready = run_resource_gate(test);
  ready = run_x4_exhaustive_gate(test, execution.main()) && ready;
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
      case Mode::kProfileBaseline:
      case Mode::kProfileCandidate: {
        const Variant variant =
            options.mode == Mode::kProfileBaseline ? Variant::kBaseline
                                                   : Variant::kCandidate;
        const float milliseconds = measure_pass(
            test, fixture, execution, variant, Scope::kGate,
            "single_gate_profile", 0, 1, true);
        std::cout << "REGISTER_FED_PROFILE_MARKER: mode="
                  << mode_name(options.mode)
                  << " tokens=" << options.token_count
                  << " scope=single_gate kernel_ms=" << milliseconds
                  << " builder_launches_in_range=0 scrub_launches_in_range=0"
                  << " profiler_range_kernel_launches=1"
                  << " ncu_replay_mode=app-range"
                  << " ncu_profile_from_start=false"
                  << " kernel_replay_forbidden=true\n";
        break;
      }
    }
  }

  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " register-fed Gate screen assertion(s) failed\n";
    return 1;
  }
  std::cout << "register-fed Gate SM87 screen passed\n";
  return 0;
}
