#include "q3x/kernels/sm87_weight_only_gemv.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kM = 512U;
constexpr std::size_t kK = 5'120U;
constexpr std::size_t kN = 17'408U;
constexpr std::size_t kAElements = kM * kK;
constexpr std::size_t kBElements = kN * kK;
constexpr std::size_t kCElements = kM * kN;
constexpr std::size_t kPackedWeightBytes = kN * (kK / 2U);
constexpr std::size_t kBlockScaleBytes = kN * (kK / 16U);
constexpr std::size_t kGuardElements = 256U;
constexpr std::size_t kGuardedOutputElements =
    kGuardElements + kCElements + kGuardElements;
constexpr int kMaximumHeuristics = 16;
constexpr int kSelectionWarmups = 2;
constexpr int kSelectionIterations = 4;
constexpr int kWarmups = 2;
constexpr int kIterations = 8;
constexpr int kRounds = 6;
constexpr double kRequiredSerialVsProduction = 1.22;
constexpr double kRequiredTwoScratchVsSerial = 1.03;
constexpr float kGateWeightScale2 = 1.25F;
constexpr float kUpWeightScale2 = 0.75F;
constexpr std::uint8_t kGuardPoisonByte = 0x3cU;
constexpr std::uint16_t kGuardPoison = 0x3c3cU;

static_assert(kAElements == 2'621'440U);
static_assert(kBElements == 89'128'960U);
static_assert(kCElements == 8'912'896U);
static_assert(kPackedWeightBytes == 44'564'480U);
static_assert(kBlockScaleBytes == 5'570'560U);

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
    if (count == 0U ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      test.expect(false, label + " count is representable and nonzero");
      return false;
    }
    count_ = count;
    const cudaError_t status =
        cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
    if (!test.cuda_ok(status, label)) {
      count_ = 0U;
      return false;
    }
    return true;
  }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0U;
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

  [[nodiscard]] bool create(TestContext& test, const std::string& label) {
    bool ready =
        test.lt_ok(cublasLtCreate(&handle_), label + " create handle");
    ready = ready && test.lt_ok(
                           cublasLtMatmulDescCreate(
                               &operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                           label + " create operation");
    const cublasOperation_t transpose_weight = CUBLAS_OP_T;
    ready = ready && test.lt_ok(
                           cublasLtMatmulDescSetAttribute(
                               operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                               &transpose_weight, sizeof(transpose_weight)),
                           label + " set transpose A");

    // Row-major [N,K] weight storage is column-major [K,N]. Row-major [M,K]
    // activation storage is column-major [K,M]. Compute C^T = W A^T so the
    // visible allocation remains row-major [M,N].
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &weight_layout_, CUDA_R_16BF, kK, kN, kK),
                           label + " create weight layout");
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &activation_layout_, CUDA_R_16BF, kK, kM, kK),
                           label + " create activation layout");
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &output_layout_, CUDA_R_16BF, kN, kM, kN),
                           label + " create output layout");
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceCreate(&preference_),
                           label + " create preference");
    std::uint64_t zero_workspace = 0U;
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceSetAttribute(
                               preference_,
                               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                               &zero_workspace, sizeof(zero_workspace)),
                           label + " require zero workspace");
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
    if (fork_ != nullptr) {
      (void)cudaEventDestroy(fork_);
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
                           cudaEventCreateWithFlags(&fork_,
                                                    cudaEventDisableTiming),
                           "create fork event");
    ready = ready && test.cuda_ok(
                           cudaEventCreateWithFlags(&done_,
                                                    cudaEventDisableTiming),
                           "create done event");
    ready = ready &&
            test.cuda_ok(cudaEventCreate(&start_), "create timing start");
    ready = ready &&
            test.cuda_ok(cudaEventCreate(&stop_), "create timing stop");
    return ready;
  }

  [[nodiscard]] cudaStream_t main() const noexcept { return main_; }
  [[nodiscard]] cudaStream_t auxiliary() const noexcept { return auxiliary_; }
  [[nodiscard]] cudaEvent_t fork() const noexcept { return fork_; }
  [[nodiscard]] cudaEvent_t done() const noexcept { return done_; }
  [[nodiscard]] cudaEvent_t start() const noexcept { return start_; }
  [[nodiscard]] cudaEvent_t stop() const noexcept { return stop_; }

 private:
  cudaStream_t main_ = nullptr;
  cudaStream_t auxiliary_ = nullptr;
  cudaEvent_t fork_ = nullptr;
  cudaEvent_t done_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

struct Fixture {
  DeviceBuffer<__nv_bfloat16> activation;
  DeviceBuffer<std::uint8_t> gate_packed;
  DeviceBuffer<std::uint8_t> gate_scales;
  DeviceBuffer<std::uint8_t> up_packed;
  DeviceBuffer<std::uint8_t> up_scales;
  DeviceBuffer<__nv_bfloat16> scratch0;
  DeviceBuffer<__nv_bfloat16> scratch1;
  DeviceBuffer<std::uint16_t> reference_gate_store;
  DeviceBuffer<std::uint16_t> reference_up_store;
  DeviceBuffer<std::uint16_t> candidate_gate_store;
  DeviceBuffer<std::uint16_t> candidate_up_store;
  DeviceBuffer<unsigned long long> validation;

  [[nodiscard]] std::uint16_t* reference_gate() noexcept {
    return reference_gate_store.get() + kGuardElements;
  }
  [[nodiscard]] std::uint16_t* reference_up() noexcept {
    return reference_up_store.get() + kGuardElements;
  }
  [[nodiscard]] std::uint16_t* candidate_gate() noexcept {
    return candidate_gate_store.get() + kGuardElements;
  }
  [[nodiscard]] std::uint16_t* candidate_up() noexcept {
    return candidate_up_store.get() + kGuardElements;
  }

  [[nodiscard]] std::uint64_t planned_bytes() const noexcept {
    return static_cast<std::uint64_t>(activation.bytes()) +
           gate_packed.bytes() + gate_scales.bytes() + up_packed.bytes() +
           up_scales.bytes() + scratch0.bytes() + scratch1.bytes() +
           reference_gate_store.bytes() + reference_up_store.bytes() +
           candidate_gate_store.bytes() + candidate_up_store.bytes() +
           validation.bytes();
  }
};

struct SelectedAlgorithm {
  cublasLtMatmulAlgo_t value{};
  int heuristic_index = -1;
  double milliseconds = std::numeric_limits<double>::quiet_NaN();
};

enum class Variant : std::uint8_t {
  kProduction,
  kSerialOneScratch,
  kNaiveDual,
  kStaggeredDual,
};

[[nodiscard]] const char* variant_name(const Variant variant) noexcept {
  switch (variant) {
    case Variant::kProduction:
      return "production_m128_fork_join";
    case Variant::kSerialOneScratch:
      return "A_serial_one_handle_one_scratch";
    case Variant::kNaiveDual:
      return "B_naive_two_handle_two_scratch";
    case Variant::kStaggeredDual:
      return "C_staggered_two_handle_two_scratch";
  }
  return "unknown";
}

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

__device__ __forceinline__ std::uint32_t mix_u32(std::uint32_t code) {
  code ^= code >> 16U;
  code *= 0x7feb'352dU;
  code ^= code >> 15U;
  code *= 0x846c'a68bU;
  code ^= code >> 16U;
  return code;
}

__global__ void fill_canonical_nvfp4_kernel(
    std::uint8_t* const values, const std::size_t count,
    const std::uint32_t salt, const bool block_scales) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t code =
      mix_u32(static_cast<std::uint32_t>(index) ^ salt);
  values[index] =
      block_scales ? static_cast<std::uint8_t>(0x50U + code % 24U)
                   : static_cast<std::uint8_t>(code);
}

__device__ __forceinline__ float decode_e4m3fn_device(
    const std::uint8_t bits) {
  const unsigned int sign =
      static_cast<unsigned int>(bits & 0x80U) << 24U;
  const unsigned int magnitude = static_cast<unsigned int>(bits & 0x7fU);
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  if (magnitude == 0x7fU) {
    return __uint_as_float(sign | 0x7fc0'0000U);
  }
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return __uint_as_float(sign);
    }
    const unsigned int leading =
        mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    const unsigned int fp32_exponent = 118U + leading;
    const unsigned int fp32_mantissa =
        (mantissa - (1U << leading)) << (23U - leading);
    return __uint_as_float(sign | (fp32_exponent << 23U) | fp32_mantissa);
  }
  return __uint_as_float(sign | ((120U + exponent) << 23U) |
                         (mantissa << 20U));
}

__device__ __forceinline__ float decode_e2m1_device(
    const std::uint8_t nibble) {
  const unsigned int sign =
      static_cast<unsigned int>(nibble & 0x08U) << 28U;
  const unsigned int magnitude = static_cast<unsigned int>(nibble & 0x07U);
  const unsigned int nonzero_mask =
      0U - static_cast<unsigned int>(magnitude != 0U);
  const unsigned int mantissa =
      ((magnitude & 1U) & static_cast<unsigned int>(magnitude > 1U)) << 22U;
  const unsigned int finite_bits =
      ((126U + (magnitude >> 1U)) << 23U) | mantissa;
  return __uint_as_float(sign | (finite_bits & nonzero_mask));
}

__global__ __launch_bounds__(256, 4)
void dequantize_nvfp4_contiguous_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const canonical_bf16) {
  constexpr unsigned int kPackedPerRow = kK / 2U;
  constexpr unsigned int kScalesPerRow = kK / 16U;
  constexpr unsigned int kThreads = 256U;
  constexpr unsigned int kPasses = kPackedPerRow / kThreads;
  static_assert(kPackedPerRow == kPasses * kThreads);
  const unsigned int n = blockIdx.x;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const std::size_t packed_base =
      static_cast<std::size_t>(n) * kPackedPerRow;
  const std::size_t scale_base =
      static_cast<std::size_t>(n) * kScalesPerRow;
  auto* const output_pairs = reinterpret_cast<std::uint32_t*>(canonical_bf16);

  std::uint8_t packed_values[kPasses];
  std::uint32_t scale_words[kPasses];
#pragma unroll
  for (unsigned int pass = 0U; pass < kPasses; ++pass) {
    const unsigned int packed_k = threadIdx.x + pass * kThreads;
    packed_values[pass] = packed_weights[packed_base + packed_k];
    scale_words[pass] = 0U;
    if (lane == 0U) {
      const std::size_t word_index =
          scale_base + pass * (kThreads / 8U) + warp * 4U;
      scale_words[pass] = *reinterpret_cast<const std::uint32_t*>(
          block_scales + word_index);
    }
  }

#pragma unroll
  for (unsigned int pass = 0U; pass < kPasses; ++pass) {
    const unsigned int packed_k = threadIdx.x + pass * kThreads;
    const std::uint32_t scale_word =
        __shfl_sync(0xffff'ffffU, scale_words[pass], 0);
    const std::uint8_t scale_code = static_cast<std::uint8_t>(
        scale_word >> ((lane >> 3U) * 8U));
    const float scale = decode_e4m3fn_device(scale_code);
    const std::uint8_t packed = packed_values[pass];
    const __nv_bfloat16 low = __float2bfloat16_rn(
        decode_e2m1_device(packed & 0x0fU) * scale);
    const __nv_bfloat16 high = __float2bfloat16_rn(
        decode_e2m1_device(packed >> 4U) * scale);
    output_pairs[packed_base + packed_k] =
        static_cast<std::uint32_t>(__bfloat16_as_ushort(low)) |
        (static_cast<std::uint32_t>(__bfloat16_as_ushort(high)) << 16U);
  }
}

__global__ void validate_pair_kernel(
    const std::uint16_t* const gate,
    const std::uint16_t* const gate_reference,
    const std::uint16_t* const up, const std::uint16_t* const up_reference,
    unsigned long long* const statistics) {
  unsigned long long gate_mismatch = 0U;
  unsigned long long gate_nonfinite = 0U;
  unsigned long long gate_sum = 0U;
  unsigned long long up_mismatch = 0U;
  unsigned long long up_nonfinite = 0U;
  unsigned long long up_sum = 0U;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < kCElements; index += stride) {
    const std::uint16_t gate_value = gate[index];
    const std::uint16_t up_value = up[index];
    gate_mismatch += gate_value != gate_reference[index] ? 1U : 0U;
    up_mismatch += up_value != up_reference[index] ? 1U : 0U;
    gate_nonfinite +=
        isfinite(__bfloat162float(__ushort_as_bfloat16(gate_value))) ? 0U : 1U;
    up_nonfinite +=
        isfinite(__bfloat162float(__ushort_as_bfloat16(up_value))) ? 0U : 1U;
    gate_sum += gate_value;
    up_sum += up_value;
  }
  if (gate_mismatch != 0U) {
    atomicAdd(statistics + 0U, gate_mismatch);
  }
  if (gate_nonfinite != 0U) {
    atomicAdd(statistics + 1U, gate_nonfinite);
  }
  atomicAdd(statistics + 2U, gate_sum);
  if (up_mismatch != 0U) {
    atomicAdd(statistics + 3U, up_mismatch);
  }
  if (up_nonfinite != 0U) {
    atomicAdd(statistics + 4U, up_nonfinite);
  }
  atomicAdd(statistics + 5U, up_sum);
}

[[nodiscard]] bool read_u64(const std::string& path,
                            std::uint64_t& value) {
  std::ifstream input(path);
  return static_cast<bool>(input >> value);
}

struct ClockState {
  std::uint64_t gpu_min = 0U;
  std::uint64_t gpu_current = 0U;
  std::uint64_t gpu_max = 0U;
  std::uint64_t emc_min = 0U;
  std::uint64_t emc_current = 0U;
  std::uint64_t emc_max = 0U;
};

[[nodiscard]] std::optional<ClockState> read_clock_state() {
  constexpr const char* kGpuRoot =
      "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/";
  constexpr const char* kEmcRoot =
      "/sys/devices/platform/bwmgr/devfreq/bwmgr/";
  ClockState state;
  if (!read_u64(std::string(kGpuRoot) + "min_freq", state.gpu_min) ||
      !read_u64(std::string(kGpuRoot) + "cur_freq", state.gpu_current) ||
      !read_u64(std::string(kGpuRoot) + "max_freq", state.gpu_max) ||
      !read_u64(std::string(kEmcRoot) + "min_freq", state.emc_min) ||
      !read_u64(std::string(kEmcRoot) + "cur_freq", state.emc_current) ||
      !read_u64(std::string(kEmcRoot) + "max_freq", state.emc_max)) {
    return std::nullopt;
  }
  return state;
}

[[nodiscard]] bool clocks_are_fixed(const ClockState& state) noexcept {
  return state.gpu_min == state.gpu_max &&
         state.gpu_current == state.gpu_max && state.emc_min == state.emc_max &&
         state.emc_current == state.emc_max;
}

[[nodiscard]] bool launch_dequantize(const std::uint8_t* const packed,
                                     const std::uint8_t* const scales,
                                     __nv_bfloat16* const output,
                                     const cudaStream_t stream) {
  dequantize_nvfp4_contiguous_kernel<<<static_cast<unsigned int>(kN), 256U, 0U,
                                       stream>>>(packed, scales, output);
  return cudaGetLastError() == cudaSuccess;
}

[[nodiscard]] bool launch_lt(const LtObjects& lt,
                             const cublasLtMatmulAlgo_t& algorithm,
                             const float alpha,
                             const __nv_bfloat16* const weight,
                             const __nv_bfloat16* const activation,
                             std::uint16_t* const output,
                             const cudaStream_t stream) {
  constexpr float kBeta = 0.0F;
  return cublasLtMatmul(
             lt.handle(), lt.operation(), &alpha, weight, lt.weight_layout(),
             activation, lt.activation_layout(), &kBeta, output,
             lt.output_layout(), output, lt.output_layout(), &algorithm,
             nullptr, 0U, stream) == CUBLAS_STATUS_SUCCESS;
}

[[nodiscard]] double measure_lt_only(
    TestContext& test, const LtObjects& lt,
    const cublasLtMatmulAlgo_t& algorithm,
    const __nv_bfloat16* const weight,
    const __nv_bfloat16* const activation, std::uint16_t* const output,
    const cudaStream_t stream, const std::string& label) {
  bool ready = true;
  for (int warmup = 0; warmup < kSelectionWarmups; ++warmup) {
    ready = launch_lt(lt, algorithm, 1.0F, weight, activation, output, stream) &&
            ready;
  }
  ready = test.cuda_ok(cudaStreamSynchronize(stream),
                       label + " warmup synchronize") &&
          ready;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  ready = test.cuda_ok(cudaEventCreate(&start), label + " create start") &&
          ready;
  ready = test.cuda_ok(cudaEventCreate(&stop), label + " create stop") && ready;
  ready = test.cuda_ok(cudaEventRecord(start, stream), label + " record start") &&
          ready;
  for (int iteration = 0; iteration < kSelectionIterations; ++iteration) {
    ready = launch_lt(lt, algorithm, 1.0F, weight, activation, output, stream) &&
            ready;
  }
  ready = test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop") &&
          ready;
  ready = test.cuda_ok(cudaEventSynchronize(stop), label + " synchronize stop") &&
          ready;
  float total = 0.0F;
  ready = test.cuda_ok(cudaEventElapsedTime(&total, start, stop),
                       label + " elapsed") &&
          ready;
  if (stop != nullptr) {
    (void)cudaEventDestroy(stop);
  }
  if (start != nullptr) {
    (void)cudaEventDestroy(start);
  }
  if (!ready) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total) / kSelectionIterations;
}

[[nodiscard]] std::optional<SelectedAlgorithm> select_algorithm(
    TestContext& test, const LtObjects& lt,
    const __nv_bfloat16* const weight,
    const __nv_bfloat16* const activation, std::uint16_t* const output,
    const cudaStream_t stream) {
  std::array<cublasLtMatmulHeuristicResult_t, kMaximumHeuristics> heuristics{};
  int returned = 0;
  if (!test.lt_ok(cublasLtMatmulAlgoGetHeuristic(
                      lt.handle(), lt.operation(), lt.weight_layout(),
                      lt.activation_layout(), lt.output_layout(),
                      lt.output_layout(), lt.preference(), kMaximumHeuristics,
                      heuristics.data(), &returned),
                  "query zero-workspace algorithms")) {
    return std::nullopt;
  }
  test.expect(returned > 0, "zero-workspace heuristic list is nonempty");
  SelectedAlgorithm selected;
  selected.milliseconds = std::numeric_limits<double>::infinity();
  for (int index = 0; index < returned; ++index) {
    const auto& result = heuristics[static_cast<std::size_t>(index)];
    if (result.state != CUBLAS_STATUS_SUCCESS || result.workspaceSize != 0U) {
      continue;
    }
    const double milliseconds = measure_lt_only(
        test, lt, result.algo, weight, activation, output, stream,
        "select zero-workspace algorithm " + std::to_string(index));
    std::cout << "NVFP4_PAIR_LT_HEURISTIC: index=" << index
              << " workspace_bytes=" << result.workspaceSize
              << " milliseconds=" << milliseconds << '\n';
    if (std::isfinite(milliseconds) && milliseconds < selected.milliseconds) {
      selected.value = result.algo;
      selected.heuristic_index = index;
      selected.milliseconds = milliseconds;
    }
  }
  if (selected.heuristic_index < 0) {
    test.expect(false, "at least one zero-workspace algorithm executes");
    return std::nullopt;
  }
  return selected;
}

[[nodiscard]] bool launch_production_branch(
    const std::uint8_t* const packed, const std::uint8_t* const scales,
    const float weight_scale_2, const __nv_bfloat16* const activation,
    std::uint16_t* const output, const cudaStream_t stream) {
  return q3x::kernels::
             launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
                 packed, scales, weight_scale_2,
                 reinterpret_cast<const std::uint16_t*>(activation), kM, kN,
                 kK, output, static_cast<void*>(stream)) ==
         static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool launch_variant(
    const Variant variant, Fixture& fixture, const Execution& execution,
    const LtObjects& main_lt, const LtObjects& auxiliary_lt,
    const SelectedAlgorithm& selected) {
  const auto main = execution.main();
  const auto auxiliary = execution.auxiliary();
  if (variant == Variant::kProduction) {
    bool ready = cudaEventRecord(execution.fork(), main) == cudaSuccess;
    ready = ready &&
            cudaStreamWaitEvent(auxiliary, execution.fork(), 0U) == cudaSuccess;
    ready = ready && launch_production_branch(
                         fixture.gate_packed.get(), fixture.gate_scales.get(),
                         kGateWeightScale2, fixture.activation.get(),
                         fixture.candidate_gate(), main);
    ready = ready && launch_production_branch(
                         fixture.up_packed.get(), fixture.up_scales.get(),
                         kUpWeightScale2, fixture.activation.get(),
                         fixture.candidate_up(), auxiliary);
    ready = ready && cudaEventRecord(execution.done(), auxiliary) == cudaSuccess;
    ready = ready &&
            cudaStreamWaitEvent(main, execution.done(), 0U) == cudaSuccess;
    return ready;
  }

  if (variant == Variant::kSerialOneScratch) {
    return launch_dequantize(fixture.gate_packed.get(),
                             fixture.gate_scales.get(), fixture.scratch0.get(),
                             main) &&
           launch_lt(main_lt, selected.value, kGateWeightScale2,
                     fixture.scratch0.get(), fixture.activation.get(),
                     fixture.candidate_gate(), main) &&
           launch_dequantize(fixture.up_packed.get(), fixture.up_scales.get(),
                             fixture.scratch0.get(), main) &&
           launch_lt(main_lt, selected.value, kUpWeightScale2,
                     fixture.scratch0.get(), fixture.activation.get(),
                     fixture.candidate_up(), main);
  }

  if (variant == Variant::kNaiveDual) {
    bool ready = cudaEventRecord(execution.fork(), main) == cudaSuccess;
    ready = ready &&
            cudaStreamWaitEvent(auxiliary, execution.fork(), 0U) == cudaSuccess;
    ready = ready && launch_dequantize(
                         fixture.gate_packed.get(), fixture.gate_scales.get(),
                         fixture.scratch0.get(), main);
    ready = ready && launch_lt(
                         main_lt, selected.value, kGateWeightScale2,
                         fixture.scratch0.get(), fixture.activation.get(),
                         fixture.candidate_gate(), main);
    ready = ready && launch_dequantize(
                         fixture.up_packed.get(), fixture.up_scales.get(),
                         fixture.scratch1.get(), auxiliary);
    ready = ready && launch_lt(
                         auxiliary_lt, selected.value, kUpWeightScale2,
                         fixture.scratch1.get(), fixture.activation.get(),
                         fixture.candidate_up(), auxiliary);
    ready = ready && cudaEventRecord(execution.done(), auxiliary) == cudaSuccess;
    ready = ready &&
            cudaStreamWaitEvent(main, execution.done(), 0U) == cudaSuccess;
    return ready;
  }

  // Gate dequantization is deliberately completed before the auxiliary branch
  // joins. Gate Lt on main can then overlap Up dequantization on auxiliary;
  // Up Lt follows its dequantization in stream order before the final join.
  bool ready = launch_dequantize(
      fixture.gate_packed.get(), fixture.gate_scales.get(),
      fixture.scratch0.get(), main);
  ready = ready && cudaEventRecord(execution.fork(), main) == cudaSuccess;
  ready = ready &&
          cudaStreamWaitEvent(auxiliary, execution.fork(), 0U) == cudaSuccess;
  ready = ready && launch_lt(main_lt, selected.value, kGateWeightScale2,
                             fixture.scratch0.get(), fixture.activation.get(),
                             fixture.candidate_gate(), main);
  ready = ready && launch_dequantize(
                       fixture.up_packed.get(), fixture.up_scales.get(),
                       fixture.scratch1.get(), auxiliary);
  ready = ready && launch_lt(auxiliary_lt, selected.value, kUpWeightScale2,
                             fixture.scratch1.get(), fixture.activation.get(),
                             fixture.candidate_up(), auxiliary);
  ready = ready && cudaEventRecord(execution.done(), auxiliary) == cudaSuccess;
  ready = ready &&
          cudaStreamWaitEvent(main, execution.done(), 0U) == cudaSuccess;
  return ready;
}

[[nodiscard]] bool poison_outputs(TestContext& test, Fixture& fixture,
                                  const Execution& execution,
                                  const std::string& label) {
  bool ready = test.cuda_ok(
      cudaMemsetAsync(fixture.candidate_gate_store.get(), kGuardPoisonByte,
                      fixture.candidate_gate_store.bytes(), execution.main()),
      label + " poison Gate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(fixture.candidate_up_store.get(),
                                       kGuardPoisonByte,
                                       fixture.candidate_up_store.bytes(),
                                       execution.main()),
                       label + " poison Up output");
  return ready;
}

[[nodiscard]] bool check_one_guard(TestContext& test,
                                   const DeviceBuffer<std::uint16_t>& buffer,
                                   const cudaStream_t stream,
                                   const std::string& label) {
  std::array<std::uint16_t, kGuardElements> prefix{};
  std::array<std::uint16_t, kGuardElements> suffix{};
  bool ready = test.cuda_ok(
      cudaMemcpyAsync(prefix.data(), buffer.get(), sizeof(prefix),
                      cudaMemcpyDeviceToHost, stream),
      label + " copy prefix guard");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(suffix.data(),
                                       buffer.get() + kGuardElements +
                                           kCElements,
                                       sizeof(suffix), cudaMemcpyDeviceToHost,
                                       stream),
                       label + " copy suffix guard");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " guard sync");
  const bool intact =
      std::all_of(prefix.begin(), prefix.end(), [](const std::uint16_t value) {
        return value == kGuardPoison;
      }) &&
      std::all_of(suffix.begin(), suffix.end(), [](const std::uint16_t value) {
        return value == kGuardPoison;
      });
  test.expect(intact, label + " guards remain intact");
  return ready && intact;
}

struct PairValidation {
  std::array<unsigned long long, 6U> values{};

  [[nodiscard]] bool exact_finite_nonzero() const noexcept {
    return values[0] == 0U && values[1] == 0U && values[2] != 0U &&
           values[3] == 0U && values[4] == 0U && values[5] != 0U;
  }
};

[[nodiscard]] bool validate_pair(TestContext& test, Fixture& fixture,
                                 const Execution& execution,
                                 const std::string& label,
                                 PairValidation& result) {
  bool ready = test.cuda_ok(
      cudaMemsetAsync(fixture.validation.get(), 0, fixture.validation.bytes(),
                      execution.main()),
      label + " zero validation");
  validate_pair_kernel<<<256U, 256U, 0U, execution.main()>>>(
      fixture.candidate_gate(), fixture.reference_gate(),
      fixture.candidate_up(), fixture.reference_up(), fixture.validation.get());
  ready = test.cuda_ok(cudaGetLastError(), label + " launch validation") &&
          ready;
  ready = test.cuda_ok(
              cudaMemcpyAsync(result.values.data(), fixture.validation.get(),
                              sizeof(result.values), cudaMemcpyDeviceToHost,
                              execution.main()),
              label + " copy validation") &&
          ready;
  ready =
      test.cuda_ok(cudaStreamSynchronize(execution.main()), label + " sync") &&
      ready;
  const bool guards =
      check_one_guard(test, fixture.candidate_gate_store, execution.main(),
                      label + " Gate") &&
      check_one_guard(test, fixture.candidate_up_store, execution.main(),
                      label + " Up");
  const bool exact = result.exact_finite_nonzero();
  test.expect(exact, label + " matches production and is finite");
  std::cout << "NVFP4_PAIR_VALIDATION: label=" << label
            << " gate_mismatches=" << result.values[0] << '/' << kCElements
            << " gate_nonfinite=" << result.values[1]
            << " gate_encoded_sum=" << result.values[2]
            << " up_mismatches=" << result.values[3] << '/' << kCElements
            << " up_nonfinite=" << result.values[4]
            << " up_encoded_sum=" << result.values[5]
            << " guards=" << (guards ? "intact" : "BAD")
            << " gate=" << (ready && guards && exact ? "PASS" : "FAIL")
            << '\n';
  return ready && guards && exact;
}

[[nodiscard]] bool generate_production_reference(
    TestContext& test, Fixture& fixture, const Execution& execution) {
  bool ready = test.cuda_ok(
      cudaMemsetAsync(fixture.reference_gate_store.get(), kGuardPoisonByte,
                      fixture.reference_gate_store.bytes(), execution.main()),
      "poison reference Gate");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(fixture.reference_up_store.get(),
                                       kGuardPoisonByte,
                                       fixture.reference_up_store.bytes(),
                                       execution.main()),
                       "poison reference Up");
  ready = ready &&
          cudaEventRecord(execution.fork(), execution.main()) == cudaSuccess;
  ready = ready && cudaStreamWaitEvent(execution.auxiliary(), execution.fork(),
                                       0U) == cudaSuccess;
  ready = ready && launch_production_branch(
                       fixture.gate_packed.get(), fixture.gate_scales.get(),
                       kGateWeightScale2, fixture.activation.get(),
                       fixture.reference_gate(), execution.main());
  ready = ready && launch_production_branch(
                       fixture.up_packed.get(), fixture.up_scales.get(),
                       kUpWeightScale2, fixture.activation.get(),
                       fixture.reference_up(), execution.auxiliary());
  ready = ready &&
          cudaEventRecord(execution.done(), execution.auxiliary()) == cudaSuccess;
  ready = ready && cudaStreamWaitEvent(execution.main(), execution.done(), 0U) ==
                       cudaSuccess;
  ready = test.cuda_ok(cudaStreamSynchronize(execution.main()),
                       "production reference synchronize") &&
          ready;
  const bool guards =
      check_one_guard(test, fixture.reference_gate_store, execution.main(),
                      "reference Gate") &&
      check_one_guard(test, fixture.reference_up_store, execution.main(),
                      "reference Up");
  return ready && guards;
}

[[nodiscard]] bool run_eager_correctness(
    TestContext& test, Fixture& fixture, const Execution& execution,
    const LtObjects& main_lt, const LtObjects& auxiliary_lt,
    const SelectedAlgorithm& selected, const Variant variant) {
  bool ready = true;
  for (int replay = 0; replay < 2; ++replay) {
    const std::string label = std::string(variant_name(variant)) +
                              "_eager_replay_" + std::to_string(replay + 1);
    ready = poison_outputs(test, fixture, execution, label) && ready;
    ready = launch_variant(variant, fixture, execution, main_lt, auxiliary_lt,
                           selected) &&
            ready;
    PairValidation validation;
    ready = validate_pair(test, fixture, execution, label, validation) && ready;
  }
  return ready;
}

struct GraphCounts {
  std::size_t total = 0U;
  std::size_t kernels = 0U;
  std::size_t event_records = 0U;
  std::size_t event_waits = 0U;
  std::size_t other = 0U;
};

[[nodiscard]] bool query_graph_counts(TestContext& test, cudaGraph_t graph,
                                      GraphCounts& counts,
                                      const std::string& label) {
  bool ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &counts.total),
                            label + " query node count");
  std::vector<cudaGraphNode_t> nodes(counts.total);
  std::size_t returned = counts.total;
  ready = test.cuda_ok(cudaGraphGetNodes(graph, nodes.data(), &returned),
                       label + " query nodes") &&
          ready;
  ready = ready && returned == counts.total;
  for (const cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    ready = test.cuda_ok(cudaGraphNodeGetType(node, &type),
                         label + " query node type") &&
            ready;
    if (type == cudaGraphNodeTypeKernel) {
      ++counts.kernels;
    } else if (type == cudaGraphNodeTypeEventRecord) {
      ++counts.event_records;
    } else if (type == cudaGraphNodeTypeWaitEvent) {
      ++counts.event_waits;
    } else {
      ++counts.other;
    }
  }
  return ready;
}

[[nodiscard]] bool run_graph_correctness(
    TestContext& test, Fixture& fixture, const Execution& execution,
    const LtObjects& main_lt, const LtObjects& auxiliary_lt,
    const SelectedAlgorithm& selected, const Variant variant) {
  const std::string label = variant_name(variant);
  bool ready =
      test.cuda_ok(cudaStreamSynchronize(execution.main()),
                   label + " pre-capture main sync");
  ready = test.cuda_ok(cudaStreamSynchronize(execution.auxiliary()),
                       label + " pre-capture auxiliary sync") &&
          ready;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  ready = test.cuda_ok(
              cudaStreamBeginCapture(execution.main(),
                                     cudaStreamCaptureModeThreadLocal),
              label + " begin capture") &&
          ready;
  const bool launched =
      launch_variant(variant, fixture, execution, main_lt, auxiliary_lt,
                     selected);
  const cudaError_t end_status =
      cudaStreamEndCapture(execution.main(), &graph);
  ready = test.cuda_ok(end_status, label + " end capture") && launched && ready;
  GraphCounts counts;
  if (ready) {
    ready = query_graph_counts(test, graph, counts, label) && ready;
    const std::size_t minimum_kernels =
        variant == Variant::kProduction ? 2U : 4U;
    test.expect(counts.kernels >= minimum_kernels,
                label + " graph contains expected compute nodes");
    ready = ready && counts.kernels >= minimum_kernels;
  }
  if (ready) {
    ready = test.cuda_ok(
                cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0U),
                label + " instantiate") &&
            ready;
  }
  for (int replay = 0; ready && replay < 2; ++replay) {
    const std::string replay_label =
        label + "_graph_replay_" + std::to_string(replay + 1);
    ready = poison_outputs(test, fixture, execution, replay_label) && ready;
    ready = test.cuda_ok(cudaGraphLaunch(executable, execution.main()),
                         replay_label + " launch") &&
            ready;
    PairValidation validation;
    ready = validate_pair(test, fixture, execution, replay_label, validation) &&
            ready;
  }
  if (executable != nullptr) {
    ready = test.cuda_ok(cudaGraphExecDestroy(executable),
                         label + " destroy executable") &&
            ready;
  }
  if (graph != nullptr) {
    ready = test.cuda_ok(cudaGraphDestroy(graph), label + " destroy graph") &&
            ready;
  }
  std::cout << "NVFP4_PAIR_GRAPH: variant=" << label
            << " total_nodes=" << counts.total
            << " kernel_nodes=" << counts.kernels
            << " event_record_nodes=" << counts.event_records
            << " event_wait_nodes=" << counts.event_waits
            << " other_nodes=" << counts.other
            << " cold_and_warm_replays=2 gate="
            << (ready ? "PASS" : "FAIL") << '\n';
  return ready;
}

[[nodiscard]] double measure_variant(
    TestContext& test, Fixture& fixture, const Execution& execution,
    const LtObjects& main_lt, const LtObjects& auxiliary_lt,
    const SelectedAlgorithm& selected, const Variant variant,
    const std::string& label) {
  bool ready = true;
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    ready = launch_variant(variant, fixture, execution, main_lt, auxiliary_lt,
                           selected) &&
            ready;
  }
  ready = test.cuda_ok(cudaStreamSynchronize(execution.main()),
                       label + " warmup synchronize") &&
          ready;
  ready = test.cuda_ok(cudaEventRecord(execution.start(), execution.main()),
                       label + " record start") &&
          ready;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    ready = launch_variant(variant, fixture, execution, main_lt, auxiliary_lt,
                           selected) &&
            ready;
  }
  ready = test.cuda_ok(cudaEventRecord(execution.stop(), execution.main()),
                       label + " record stop") &&
          ready;
  ready = test.cuda_ok(cudaEventSynchronize(execution.stop()),
                       label + " synchronize stop") &&
          ready;
  float total = 0.0F;
  ready = test.cuda_ok(cudaEventElapsedTime(&total, execution.start(),
                                            execution.stop()),
                       label + " elapsed") &&
          ready;
  if (!ready) {
    test.expect(false, label + " measurement completes");
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total) / kIterations;
}

struct ComparisonResult {
  double baseline_milliseconds = std::numeric_limits<double>::quiet_NaN();
  double candidate_milliseconds = std::numeric_limits<double>::quiet_NaN();
  double speedup = std::numeric_limits<double>::quiet_NaN();
  bool every_round_positive = false;
};

[[nodiscard]] ComparisonResult run_bccb(
    TestContext& test, Fixture& fixture, const Execution& execution,
    const LtObjects& main_lt, const LtObjects& auxiliary_lt,
    const SelectedAlgorithm& selected, const Variant baseline,
    const Variant candidate, const std::string& comparison) {
  double baseline_sum = 0.0;
  double candidate_sum = 0.0;
  bool every_round_positive = true;
  for (int round = 0; round < kRounds; ++round) {
    const std::string prefix =
        comparison + "_round_" + std::to_string(round + 1) + '_';
    const double b1 = measure_variant(test, fixture, execution, main_lt,
                                      auxiliary_lt, selected, baseline,
                                      prefix + "B1");
    const double c1 = measure_variant(test, fixture, execution, main_lt,
                                      auxiliary_lt, selected, candidate,
                                      prefix + "C1");
    const double c2 = measure_variant(test, fixture, execution, main_lt,
                                      auxiliary_lt, selected, candidate,
                                      prefix + "C2");
    const double b2 = measure_variant(test, fixture, execution, main_lt,
                                      auxiliary_lt, selected, baseline,
                                      prefix + "B2");
    const bool finite = std::isfinite(b1) && std::isfinite(c1) &&
                        std::isfinite(c2) && std::isfinite(b2) && b1 > 0.0 &&
                        c1 > 0.0 && c2 > 0.0 && b2 > 0.0;
    const double speedup =
        finite ? (b1 + b2) / (c1 + c2)
               : std::numeric_limits<double>::quiet_NaN();
    every_round_positive =
        every_round_positive && finite && speedup > 1.0;
    if (finite) {
      baseline_sum += b1 + b2;
      candidate_sum += c1 + c2;
    }
    std::cout << "PERF_NVFP4_PAIR_ROUND: comparison=" << comparison
              << " round=" << round + 1 << " order=B-C-C-B"
              << " B1_ms=" << b1 << " C1_ms=" << c1
              << " C2_ms=" << c2 << " B2_ms=" << b2
              << " speedup=" << speedup
              << " strict_positive_gate="
              << (finite && speedup > 1.0 ? "PASS" : "FAIL") << '\n';
  }
  ComparisonResult result;
  result.baseline_milliseconds = baseline_sum / (2.0 * kRounds);
  result.candidate_milliseconds = candidate_sum / (2.0 * kRounds);
  result.speedup = baseline_sum / candidate_sum;
  result.every_round_positive = every_round_positive;
  std::cout << "PERF_NVFP4_PAIR_AGGREGATE: comparison=" << comparison
            << " baseline_variant=" << variant_name(baseline)
            << " candidate_variant=" << variant_name(candidate)
            << " baseline_pair_ms=" << result.baseline_milliseconds
            << " candidate_pair_ms=" << result.candidate_milliseconds
            << " speedup=" << result.speedup
            << " every_round_strict_positive="
            << (result.every_round_positive ? "true" : "false")
            << " rounds=" << kRounds << " iterations=" << kIterations
            << " fixed_clock_required=true" << '\n';
  return result;
}

[[nodiscard]] bool allocate_fixture(TestContext& test, Fixture& fixture) {
  bool ready = fixture.activation.allocate(test, kAElements,
                                           "allocate shared activation");
  ready = ready && fixture.gate_packed.allocate(
                       test, kPackedWeightBytes, "allocate Gate packed weight");
  ready = ready && fixture.gate_scales.allocate(
                       test, kBlockScaleBytes, "allocate Gate block scales");
  ready = ready && fixture.up_packed.allocate(
                       test, kPackedWeightBytes, "allocate Up packed weight");
  ready = ready && fixture.up_scales.allocate(
                       test, kBlockScaleBytes, "allocate Up block scales");
  ready = ready && fixture.scratch0.allocate(
                       test, kBElements, "allocate BF16 scratch0");
  ready = ready && fixture.scratch1.allocate(
                       test, kBElements, "allocate BF16 scratch1");
  ready = ready && fixture.reference_gate_store.allocate(
                       test, kGuardedOutputElements,
                       "allocate guarded reference Gate output");
  ready = ready && fixture.reference_up_store.allocate(
                       test, kGuardedOutputElements,
                       "allocate guarded reference Up output");
  ready = ready && fixture.candidate_gate_store.allocate(
                       test, kGuardedOutputElements,
                       "allocate guarded candidate Gate output");
  ready = ready && fixture.candidate_up_store.allocate(
                       test, kGuardedOutputElements,
                       "allocate guarded candidate Up output");
  ready = ready &&
          fixture.validation.allocate(test, 6U, "allocate validation counters");
  return ready;
}

[[nodiscard]] bool initialize_fixture(TestContext& test, Fixture& fixture,
                                      const Execution& execution) {
  constexpr unsigned int kThreads = 256U;
  const auto blocks = [](const std::size_t count) {
    return static_cast<unsigned int>((count + kThreads - 1U) / kThreads);
  };
  fill_deterministic_bf16_kernel<<<blocks(kAElements), kThreads, 0U,
                                   execution.main()>>>(
      fixture.activation.get(), kAElements, 0x1234'5678U, 1.0F / 64.0F);
  bool ready = test.cuda_ok(cudaGetLastError(), "fill shared activation");
  fill_canonical_nvfp4_kernel<<<blocks(kPackedWeightBytes), kThreads, 0U,
                                execution.main()>>>(
      fixture.gate_packed.get(), kPackedWeightBytes, 0x6a09'e667U, false);
  ready = test.cuda_ok(cudaGetLastError(), "fill Gate packed weight") && ready;
  fill_canonical_nvfp4_kernel<<<blocks(kBlockScaleBytes), kThreads, 0U,
                                execution.main()>>>(
      fixture.gate_scales.get(), kBlockScaleBytes, 0xbb67'ae85U, true);
  ready = test.cuda_ok(cudaGetLastError(), "fill Gate block scales") && ready;
  fill_canonical_nvfp4_kernel<<<blocks(kPackedWeightBytes), kThreads, 0U,
                                execution.main()>>>(
      fixture.up_packed.get(), kPackedWeightBytes, 0x3c6e'f372U, false);
  ready = test.cuda_ok(cudaGetLastError(), "fill distinct Up packed weight") &&
          ready;
  fill_canonical_nvfp4_kernel<<<blocks(kBlockScaleBytes), kThreads, 0U,
                                execution.main()>>>(
      fixture.up_scales.get(), kBlockScaleBytes, 0xa54f'f53aU, true);
  ready = test.cuda_ok(cudaGetLastError(), "fill distinct Up block scales") &&
          ready;
  ready = test.cuda_ok(cudaStreamSynchronize(execution.main()),
                       "initialize fixture synchronize") &&
          ready;
  return ready;
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
    std::cout << "SKIP: Gate/Up cuBLASLt pair screen requires SM87; found "
              << properties.major << '.' << properties.minor << '\n';
    return 77;
  }

  const std::optional<ClockState> clocks = read_clock_state();
  if (!clocks.has_value() || !clocks_are_fixed(*clocks)) {
    std::cout << "SKIP: Gate/Up pair performance screen requires fixed GPU and "
                 "EMC clocks"
              << " clock_state_available="
              << (clocks.has_value() ? "true" : "false") << '\n';
    return 77;
  }

  std::size_t free_before = 0U;
  std::size_t total_before = 0U;
  if (!test.cuda_ok(cudaMemGetInfo(&free_before, &total_before),
                    "query memory before fixture")) {
    return 1;
  }

  std::cout << std::fixed << std::setprecision(6)
            << "NVFP4_PAIR_PROTOCOL: device=" << properties.name
            << " cc=" << properties.major << '.' << properties.minor
            << " M=" << kM << " N=" << kN << " K=" << kK
            << " gate_weight_scale_2=" << kGateWeightScale2
            << " up_weight_scale_2=" << kUpWeightScale2
            << " workspace_bytes=0"
            << " gpu_min_hz=" << clocks->gpu_min
            << " gpu_current_hz=" << clocks->gpu_current
            << " gpu_max_hz=" << clocks->gpu_max
            << " emc_min_hz=" << clocks->emc_min
            << " emc_current_hz=" << clocks->emc_current
            << " emc_max_hz=" << clocks->emc_max
            << " fixed_clocks=true" << '\n';

  Fixture fixture;
  Execution execution;
  LtObjects main_lt;
  LtObjects auxiliary_lt;
  bool ready = allocate_fixture(test, fixture);
  ready = ready && execution.create(test);
  ready = ready && main_lt.create(test, "main Lt");
  ready = ready && auxiliary_lt.create(test, "auxiliary Lt");
  if (!ready) {
    return 1;
  }

  std::size_t free_after = 0U;
  std::size_t total_after = 0U;
  ready = test.cuda_ok(cudaMemGetInfo(&free_after, &total_after),
                       "query memory after fixture");
  const std::uint64_t observed_drop =
      free_before >= free_after ? free_before - free_after : 0U;
  constexpr std::uint64_t kPerScratchBytes =
      kBElements * sizeof(__nv_bfloat16);
  constexpr std::uint64_t kTwoScratchBytes = 2U * kPerScratchBytes;
  const std::uint64_t two_scratch_fixture_bytes = fixture.planned_bytes();
  const std::uint64_t one_scratch_fixture_bytes =
      two_scratch_fixture_bytes - kPerScratchBytes;
  std::cout << "NVFP4_PAIR_MEMORY: total_device_bytes=" << total_after
            << " free_before_bytes=" << free_before
            << " free_after_bytes=" << free_after
            << " observed_free_drop_bytes=" << observed_drop
            << " two_scratch_test_fixture_bytes="
            << two_scratch_fixture_bytes
            << " one_scratch_test_fixture_bytes="
            << one_scratch_fixture_bytes
            << " one_scratch_incremental_bytes=" << kPerScratchBytes
            << " two_scratch_incremental_bytes=" << kTwoScratchBytes
            << " per_scratch_bytes=" << kPerScratchBytes
            << " lt_workspace_bytes=0" << '\n';

  ready = initialize_fixture(test, fixture, execution) && ready;
  ready = launch_dequantize(fixture.gate_packed.get(),
                            fixture.gate_scales.get(), fixture.scratch0.get(),
                            execution.main()) &&
          ready;
  ready = test.cuda_ok(cudaStreamSynchronize(execution.main()),
                       "prepare algorithm-selection weight") &&
          ready;
  const std::optional<SelectedAlgorithm> selected = select_algorithm(
      test, main_lt, fixture.scratch0.get(), fixture.activation.get(),
      fixture.candidate_gate(), execution.main());
  if (!ready || !selected.has_value()) {
    return 1;
  }

  cublasLtMatmulHeuristicResult_t auxiliary_check{};
  ready = test.lt_ok(
              cublasLtMatmulAlgoCheck(
                  auxiliary_lt.handle(), auxiliary_lt.operation(),
                  auxiliary_lt.weight_layout(),
                  auxiliary_lt.activation_layout(),
                  auxiliary_lt.output_layout(), auxiliary_lt.output_layout(),
                  &selected->value, &auxiliary_check),
              "validate selected algorithm on auxiliary handle") &&
          ready;
  test.expect(auxiliary_check.workspaceSize == 0U,
              "auxiliary handle keeps selected algorithm zero-workspace");
  ready = ready && auxiliary_check.workspaceSize == 0U;
  std::cout << "NVFP4_PAIR_LT_SELECTED: heuristic_index="
            << selected->heuristic_index
            << " selection_milliseconds=" << selected->milliseconds
            << " main_workspace_bytes=0"
            << " auxiliary_workspace_bytes=" << auxiliary_check.workspaceSize
            << " gate=PASS\n";

  ready = generate_production_reference(test, fixture, execution) && ready;
  for (const Variant variant :
       {Variant::kProduction, Variant::kSerialOneScratch,
        Variant::kNaiveDual, Variant::kStaggeredDual}) {
    ready = run_eager_correctness(test, fixture, execution, main_lt,
                                  auxiliary_lt, *selected, variant) &&
            ready;
    ready = run_graph_correctness(test, fixture, execution, main_lt,
                                  auxiliary_lt, *selected, variant) &&
            ready;
  }

  const ComparisonResult production_vs_a = run_bccb(
      test, fixture, execution, main_lt, auxiliary_lt, *selected,
      Variant::kProduction, Variant::kSerialOneScratch, "production_vs_A");
  const ComparisonResult a_vs_b = run_bccb(
      test, fixture, execution, main_lt, auxiliary_lt, *selected,
      Variant::kSerialOneScratch, Variant::kNaiveDual, "A_vs_B");
  const ComparisonResult a_vs_c = run_bccb(
      test, fixture, execution, main_lt, auxiliary_lt, *selected,
      Variant::kSerialOneScratch, Variant::kStaggeredDual, "A_vs_C");

  const bool serial_gate = production_vs_a.every_round_positive &&
                           std::isfinite(production_vs_a.speedup) &&
                           production_vs_a.speedup >=
                               kRequiredSerialVsProduction;
  const bool naive_recommendation =
      a_vs_b.every_round_positive && std::isfinite(a_vs_b.speedup) &&
      a_vs_b.speedup >= kRequiredTwoScratchVsSerial;
  const bool staggered_recommendation =
      a_vs_c.every_round_positive && std::isfinite(a_vs_c.speedup) &&
      a_vs_c.speedup >= kRequiredTwoScratchVsSerial;
  const Variant recommendation =
      staggered_recommendation &&
              (!naive_recommendation || a_vs_c.speedup >= a_vs_b.speedup)
          ? Variant::kStaggeredDual
          : (naive_recommendation ? Variant::kNaiveDual
                                  : Variant::kSerialOneScratch);
  test.expect(serial_gate,
              "serial one-scratch route clears 1.22x production pair gate");
  std::cout << "NVFP4_PAIR_FINAL: production_vs_A_speedup="
            << production_vs_a.speedup
            << " production_vs_A_required=" << kRequiredSerialVsProduction
            << " production_vs_A_every_round_positive="
            << (production_vs_a.every_round_positive ? "true" : "false")
            << " A_vs_B_speedup=" << a_vs_b.speedup
            << " A_vs_B_required=" << kRequiredTwoScratchVsSerial
            << " A_vs_B_every_round_positive="
            << (a_vs_b.every_round_positive ? "true" : "false")
            << " A_vs_B_recommend="
            << (naive_recommendation ? "true" : "false")
            << " A_vs_C_speedup=" << a_vs_c.speedup
            << " A_vs_C_required=" << kRequiredTwoScratchVsSerial
            << " A_vs_C_every_round_positive="
            << (a_vs_c.every_round_positive ? "true" : "false")
            << " A_vs_C_recommend="
            << (staggered_recommendation ? "true" : "false")
            << " recommended_variant=" << variant_name(recommendation)
            << " recommended_scratch_count="
            << (recommendation == Variant::kSerialOneScratch ? 1 : 2)
            << " hard_gate=" << (serial_gate ? "PASS" : "FAIL") << '\n';

  ready = test.cuda_ok(cudaStreamSynchronize(execution.main()),
                       "final main synchronize") &&
          ready;
  ready = test.cuda_ok(cudaStreamSynchronize(execution.auxiliary()),
                       "final auxiliary synchronize") &&
          ready;
  if (!ready) {
    test.expect(false, "Gate/Up pair screen completed all CUDA work");
  }
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " Gate/Up C512 cuBLASLt pair assertion(s) failed\n";
    return 1;
  }
  std::cout << "Gate/Up C512 cuBLASLt pair screen passed\n";
  return 0;
}
