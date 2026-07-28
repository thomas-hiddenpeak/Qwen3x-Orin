#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kM = 512U;
constexpr std::size_t kK = 5'120U;
constexpr std::size_t kN = 17'408U;
constexpr std::size_t kAElements = kM * kK;
constexpr std::size_t kBElements = kK * kN;
constexpr std::size_t kCElements = kM * kN;
constexpr std::size_t kWorkspaceBytes = 256U * 1024U * 1024U;
constexpr int kMaximumHeuristics = 16;
constexpr int kSelectionWarmups = 2;
constexpr int kSelectionIterations = 4;
constexpr int kFormalWarmups = 10;
constexpr int kFormalIterations = 24;
constexpr int kFormalRounds = 6;
constexpr double kUsefulFlops =
    2.0 * static_cast<double>(kM) * static_cast<double>(kN) *
    static_cast<double>(kK);

static_assert(kAElements == 2'621'440U);
static_assert(kBElements == 89'128'960U);
static_assert(kCElements == 8'912'896U);

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

  [[nodiscard]] bool lt_ok(const cublasStatus_t status,
                           const std::string& operation) {
    expect(status == CUBLAS_STATUS_SUCCESS,
           operation + ": cuBLAS status " +
               std::to_string(static_cast<int>(status)));
    return status == CUBLAS_STATUS_SUCCESS;
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
      test.expect(false, label + " allocation is representable");
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

class LtObjects {
 public:
  LtObjects() = default;
  LtObjects(const LtObjects&) = delete;
  LtObjects& operator=(const LtObjects&) = delete;

  ~LtObjects() {
    if (preference_ != nullptr) {
      (void)cublasLtMatmulPreferenceDestroy(preference_);
    }
    if (output_layout_ != nullptr) {
      (void)cublasLtMatrixLayoutDestroy(output_layout_);
    }
    if (activation_layout_ != nullptr) {
      (void)cublasLtMatrixLayoutDestroy(activation_layout_);
    }
    if (weight_layout_ != nullptr) {
      (void)cublasLtMatrixLayoutDestroy(weight_layout_);
    }
    if (operation_ != nullptr) {
      (void)cublasLtMatmulDescDestroy(operation_);
    }
    if (handle_ != nullptr) {
      (void)cublasLtDestroy(handle_);
    }
  }

  [[nodiscard]] bool create(TestContext& test) {
    bool ready = test.lt_ok(cublasLtCreate(&handle_), "create cuBLASLt");
    ready = ready && test.lt_ok(
                           cublasLtMatmulDescCreate(
                               &operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                           "create BF16 matmul descriptor");

    // A row-major [M,K] allocation is column-major [K,M].  The persistent
    // BF16 weight allocation is row-major [K,N], hence column-major [N,K].
    // Compute C^T = B^T A^T with ordinary column-major descriptors so the
    // visible output allocation remains row-major [M,N].
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &weight_layout_, CUDA_R_16BF, kN, kK, kN),
                           "create persistent BF16 weight layout [N,K]");
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &activation_layout_, CUDA_R_16BF, kK, kM, kK),
                           "create BF16 activation layout [K,M]");
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &output_layout_, CUDA_R_16BF, kN, kM, kN),
                           "create BF16 output layout [N,M]");
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceCreate(&preference_),
                           "create cuBLASLt preference");
    std::size_t workspace_bytes = kWorkspaceBytes;
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceSetAttribute(
                               preference_,
                               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                               &workspace_bytes, sizeof(workspace_bytes)),
                           "set cuBLASLt workspace preference");
    return ready;
  }

  [[nodiscard]] cublasLtHandle_t handle() const noexcept { return handle_; }
  [[nodiscard]] cublasLtMatmulDesc_t operation() const noexcept {
    return operation_;
  }
  [[nodiscard]] cublasLtMatrixLayout_t weight_layout() const noexcept {
    return weight_layout_;
  }
  [[nodiscard]] cublasLtMatrixLayout_t activation_layout() const noexcept {
    return activation_layout_;
  }
  [[nodiscard]] cublasLtMatrixLayout_t output_layout() const noexcept {
    return output_layout_;
  }
  [[nodiscard]] cublasLtMatmulPreference_t preference() const noexcept {
    return preference_;
  }

 private:
  cublasLtHandle_t handle_ = nullptr;
  cublasLtMatmulDesc_t operation_ = nullptr;
  cublasLtMatrixLayout_t weight_layout_ = nullptr;
  cublasLtMatrixLayout_t activation_layout_ = nullptr;
  cublasLtMatrixLayout_t output_layout_ = nullptr;
  cublasLtMatmulPreference_t preference_ = nullptr;
};

__global__ void fill_deterministic_bf16_kernel(__nv_bfloat16* const values,
                                                const std::size_t count,
                                                const std::uint32_t salt,
                                                const float scale) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  std::uint32_t code = static_cast<std::uint32_t>(index) ^ salt;
  code ^= code >> 16U;
  code *= 0x7feb'352dU;
  code ^= code >> 15U;
  code *= 0x846c'a68bU;
  code ^= code >> 16U;
  const int centered = static_cast<int>(code % 17U) - 8;
  values[index] = __float2bfloat16_rn(static_cast<float>(centered) * scale);
}

__global__ void validate_bf16_replay_kernel(
    const __nv_bfloat16* const output,
    const __nv_bfloat16* const replay_reference, const std::size_t count,
    unsigned long long* const mismatch_count,
    unsigned long long* const nonfinite_count,
    unsigned long long* const encoded_sum) {
  unsigned long long local_mismatch = 0U;
  unsigned long long local_nonfinite = 0U;
  unsigned long long local_sum = 0U;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const std::uint16_t encoded =
        reinterpret_cast<const std::uint16_t*>(output)[index];
    const std::uint16_t replay_encoded =
        reinterpret_cast<const std::uint16_t*>(replay_reference)[index];
    local_mismatch += encoded != replay_encoded ? 1U : 0U;
    local_nonfinite +=
        isfinite(__bfloat162float(output[index])) ? 0U : 1U;
    local_sum += encoded;
  }
  if (local_mismatch != 0U) {
    atomicAdd(mismatch_count, local_mismatch);
  }
  if (local_nonfinite != 0U) {
    atomicAdd(nonfinite_count, local_nonfinite);
  }
  atomicAdd(encoded_sum, local_sum);
}

[[nodiscard]] bool launch_lt(
    TestContext& test, const LtObjects& lt,
    const cublasLtMatmulAlgo_t& algorithm,
    const __nv_bfloat16* const persistent_weight,
    const __nv_bfloat16* const activation, __nv_bfloat16* const output,
    void* const workspace, const std::size_t workspace_bytes,
    const cudaStream_t stream, const std::string& label) {
  constexpr float kAlpha = 1.0F;
  constexpr float kBeta = 0.0F;
  return test.lt_ok(
      cublasLtMatmul(
          lt.handle(), lt.operation(), &kAlpha, persistent_weight,
          lt.weight_layout(), activation, lt.activation_layout(), &kBeta,
          output, lt.output_layout(), output, lt.output_layout(), &algorithm,
          workspace, workspace_bytes, stream),
      label);
}

[[nodiscard]] double measure_algorithm(
    TestContext& test, const LtObjects& lt,
    const cublasLtMatmulAlgo_t& algorithm,
    const __nv_bfloat16* const persistent_weight,
    const __nv_bfloat16* const activation, __nv_bfloat16* const output,
    void* const workspace, const std::size_t workspace_bytes,
    const cudaStream_t stream, const int warmups, const int iterations,
    const std::string& label) {
  for (int warmup = 0; warmup < warmups; ++warmup) {
    if (!launch_lt(test, lt, algorithm, persistent_weight, activation, output,
                   workspace, workspace_bytes, stream,
                   label + " warmup " + std::to_string(warmup))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  if (!test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync")) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ready = test.cuda_ok(cudaEventCreate(&start), label + " create start");
  ready = ready &&
          test.cuda_ok(cudaEventCreate(&stop), label + " create stop");
  ready = ready &&
          test.cuda_ok(cudaEventRecord(start, stream), label + " record start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = launch_lt(test, lt, algorithm, persistent_weight, activation,
                      output, workspace, workspace_bytes, stream,
                      label + " measured " + std::to_string(iteration));
  }
  ready = ready &&
          test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop");
  ready = ready &&
          test.cuda_ok(cudaEventSynchronize(stop), label + " stop sync");
  float total_milliseconds = 0.0F;
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_milliseconds, start, stop),
                       label + " elapsed time");
  if (stop != nullptr) {
    (void)cudaEventDestroy(stop);
  }
  if (start != nullptr) {
    (void)cudaEventDestroy(start);
  }
  if (!ready) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total_milliseconds) /
         static_cast<double>(iterations);
}

[[nodiscard]] double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if ((values.size() & 1U) != 0U) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1U] + values[middle]);
}

}  // namespace

int main() {
  TestContext test;
  int device = 0;
  if (!test.cuda_ok(cudaGetDevice(&device), "get active CUDA device")) {
    return 1;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, device),
                    "query active CUDA device")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: persistent-BF16 Gate ceiling requires SM87; found "
              << properties.major << '.' << properties.minor << '\n';
    return 77;
  }

  std::cout << std::fixed << std::setprecision(6)
            << "CUBLASLT_GATE_C512_PROTOCOL: device=" << properties.name
            << " cc=" << properties.major << '.' << properties.minor
            << " M=" << kM << " N=" << kN << " K=" << kK
            << " persistent_B_bytes="
            << kBElements * sizeof(__nv_bfloat16)
            << " dequantization_timed=false workspace_bytes="
            << kWorkspaceBytes << " useful_GFLOP=" << kUsefulFlops / 1.0e9
            << '\n';

  DeviceBuffer<__nv_bfloat16> activation;
  DeviceBuffer<__nv_bfloat16> persistent_weight;
  DeviceBuffer<__nv_bfloat16> output;
  DeviceBuffer<__nv_bfloat16> replay_reference;
  DeviceBuffer<std::uint8_t> workspace;
  DeviceBuffer<unsigned long long> validation;
  bool ready = activation.allocate(test, kAElements, "allocate BF16 A");
  ready = ready && persistent_weight.allocate(test, kBElements,
                                               "allocate persistent BF16 B");
  ready = ready && output.allocate(test, kCElements, "allocate BF16 C");
  ready = ready && replay_reference.allocate(test, kCElements,
                                              "allocate BF16 replay C");
  ready = ready && workspace.allocate(test, kWorkspaceBytes,
                                      "allocate cuBLASLt workspace");
  ready = ready && validation.allocate(test, 3U, "allocate validation counts");

  cudaStream_t stream = nullptr;
  ready = ready && test.cuda_ok(
                       cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "create nonblocking stream");
  LtObjects lt;
  ready = ready && lt.create(test);
  if (!ready) {
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
    return 1;
  }

  constexpr unsigned int kFillThreads = 256U;
  const auto fill = [&](DeviceBuffer<__nv_bfloat16>& buffer,
                        const std::size_t count, const std::uint32_t salt,
                        const float scale) {
    const std::size_t blocks =
        (count + kFillThreads - 1U) / kFillThreads;
    fill_deterministic_bf16_kernel<<<static_cast<unsigned int>(blocks),
                                     kFillThreads, 0, stream>>>(
        buffer.get(), count, salt, scale);
    return test.cuda_ok(cudaGetLastError(), "launch deterministic BF16 fill");
  };
  ready = fill(activation, kAElements, 0x1234'5678U, 1.0F / 64.0F);
  ready = ready && fill(persistent_weight, kBElements, 0x9abc'def0U,
                        1.0F / 128.0F);
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), "finish BF16 fills");

  std::array<cublasLtMatmulHeuristicResult_t, kMaximumHeuristics> heuristics{};
  int returned_algorithms = 0;
  ready = ready && test.lt_ok(
                       cublasLtMatmulAlgoGetHeuristic(
                           lt.handle(), lt.operation(), lt.weight_layout(),
                           lt.activation_layout(), lt.output_layout(),
                           lt.output_layout(), lt.preference(),
                           kMaximumHeuristics, heuristics.data(),
                           &returned_algorithms),
                       "query cuBLASLt algorithms");
  test.expect(returned_algorithms > 0,
              "cuBLASLt returns at least one BF16 algorithm");
  if (!ready || returned_algorithms <= 0) {
    (void)cudaStreamDestroy(stream);
    return 1;
  }

  int selected_index = -1;
  double selected_milliseconds = std::numeric_limits<double>::infinity();
  for (int index = 0; index < returned_algorithms; ++index) {
    if (heuristics[static_cast<std::size_t>(index)].state !=
            CUBLAS_STATUS_SUCCESS ||
        heuristics[static_cast<std::size_t>(index)].workspaceSize >
            kWorkspaceBytes) {
      continue;
    }
    const double milliseconds = measure_algorithm(
        test, lt, heuristics[static_cast<std::size_t>(index)].algo,
        persistent_weight.get(), activation.get(), output.get(),
        workspace.get(), kWorkspaceBytes, stream, kSelectionWarmups,
        kSelectionIterations, "select algorithm " + std::to_string(index));
    const double tflops = kUsefulFlops / (milliseconds * 1.0e9);
    std::cout << "CUBLASLT_GATE_C512_HEURISTIC: index=" << index
              << " workspace_bytes="
              << heuristics[static_cast<std::size_t>(index)].workspaceSize
              << " milliseconds=" << milliseconds
              << " TFLOP_per_s=" << tflops << '\n';
    if (std::isfinite(milliseconds) && milliseconds < selected_milliseconds) {
      selected_milliseconds = milliseconds;
      selected_index = index;
    }
  }
  test.expect(selected_index >= 0,
              "at least one cuBLASLt BF16 algorithm executes");
  if (selected_index < 0) {
    (void)cudaStreamDestroy(stream);
    return 1;
  }

  const auto& selected =
      heuristics[static_cast<std::size_t>(selected_index)].algo;
  ready = launch_lt(test, lt, selected, persistent_weight.get(),
                    activation.get(), output.get(), workspace.get(),
                    kWorkspaceBytes, stream, "validation reference");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(replay_reference.get(), output.get(),
                                       kCElements * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToDevice, stream),
                       "copy replay reference");
  ready = ready && launch_lt(test, lt, selected, persistent_weight.get(),
                             activation.get(), output.get(), workspace.get(),
                             kWorkspaceBytes, stream, "validation replay");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(validation.get(), 0,
                                       3U * sizeof(unsigned long long), stream),
                       "zero validation counts");
  validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
      output.get(), replay_reference.get(), kCElements, validation.get(),
      validation.get() + 1U, validation.get() + 2U);
  ready = ready &&
          test.cuda_ok(cudaGetLastError(), "launch replay validation");
  std::array<unsigned long long, 3U> host_validation{};
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(host_validation.data(), validation.get(),
                                       sizeof(host_validation),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy validation counts");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), "validation sync");
  test.expect(host_validation[0] == 0U, "BF16 output replay is bit exact");
  test.expect(host_validation[1] == 0U, "every BF16 output is finite");
  test.expect(host_validation[2] != 0U, "BF16 encoded checksum is nonzero");
  std::cout << "CUBLASLT_GATE_C512_VALIDATION: replay_mismatches="
            << host_validation[0] << '/' << kCElements
            << " nonfinite=" << host_validation[1]
            << " encoded_sum=" << host_validation[2]
            << " gate="
            << ((host_validation[0] == 0U && host_validation[1] == 0U &&
                 host_validation[2] != 0U)
                    ? "PASS"
                    : "FAIL")
            << '\n';

  std::vector<double> round_milliseconds;
  round_milliseconds.reserve(kFormalRounds);
  for (int round = 0; ready && round < kFormalRounds; ++round) {
    const double milliseconds = measure_algorithm(
        test, lt, selected, persistent_weight.get(), activation.get(),
        output.get(), workspace.get(), kWorkspaceBytes, stream,
        kFormalWarmups, kFormalIterations,
        "formal round " + std::to_string(round + 1));
    ready = ready && std::isfinite(milliseconds) && milliseconds > 0.0;
    test.expect(ready, "formal BF16 timing is finite and positive");
    if (ready) {
      round_milliseconds.push_back(milliseconds);
      std::cout << "CUBLASLT_GATE_C512_ROUND: round=" << round + 1
                << " iterations=" << kFormalIterations
                << " milliseconds=" << milliseconds
                << " TFLOP_per_s="
                << kUsefulFlops / (milliseconds * 1.0e9) << '\n';
    }
  }

  if (round_milliseconds.size() ==
      static_cast<std::size_t>(kFormalRounds)) {
    const double median_milliseconds = median(round_milliseconds);
    const auto [minimum, maximum] =
        std::minmax_element(round_milliseconds.begin(),
                            round_milliseconds.end());
    constexpr double kProductionGateReferenceMilliseconds = 6.561464;
    std::cout << "CUBLASLT_GATE_C512_FINAL: selected_index=" << selected_index
              << " rounds=" << kFormalRounds
              << " median_milliseconds=" << median_milliseconds
              << " minimum_milliseconds=" << *minimum
              << " maximum_milliseconds=" << *maximum
              << " median_TFLOP_per_s="
              << kUsefulFlops / (median_milliseconds * 1.0e9)
              << " directional_speedup_vs_fresh_production_M128="
              << kProductionGateReferenceMilliseconds / median_milliseconds
              << " comparison_scope=absolute_persistent_BF16_control"
              << " dequantization_timed=false gate=PASS\n";
  }

  if (stream != nullptr) {
    (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy stream");
  }
  if (!ready) {
    test.expect(false, "persistent-BF16 ceiling completed");
  }
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " persistent-BF16 Gate ceiling assertion(s) failed\n";
    return 1;
  }
  std::cout << "Persistent-BF16 cuBLASLt Gate C512 ceiling passed\n";
  return 0;
}
