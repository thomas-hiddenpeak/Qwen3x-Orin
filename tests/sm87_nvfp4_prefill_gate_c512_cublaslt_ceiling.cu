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
constexpr std::size_t kPackedWeightBytes = kN * (kK / 2U);
constexpr std::size_t kBlockScaleBytes = kN * (kK / 16U);
constexpr std::size_t kWorkspaceBytes = 256U * 1024U * 1024U;
constexpr int kMaximumHeuristics = 16;
constexpr int kSelectionWarmups = 2;
constexpr int kSelectionIterations = 4;
constexpr int kFormalWarmups = 10;
constexpr int kFormalIterations = 24;
constexpr int kFormalRounds = 6;
constexpr double kProductionGateReferenceMilliseconds = 6.561464;
constexpr double kRequiredInclusiveSpeedup = 1.22;
constexpr float kWeightScale2 = 1.25F;
constexpr double kUsefulFlops =
    2.0 * static_cast<double>(kM) * static_cast<double>(kN) *
    static_cast<double>(kK);

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

    const cublasOperation_t transpose_weight = CUBLAS_OP_T;
    ready = ready && test.lt_ok(
                           cublasLtMatmulDescSetAttribute(
                               operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                               &transpose_weight,
                               sizeof(transpose_weight)),
                           "transpose canonical BF16 weight operand");

    // A row-major [M,K] allocation is column-major [K,M].  The persistent
    // BF16 weight allocation is canonical row-major [N,K], hence
    // column-major [K,N].  Transpose that first operand to compute
    // C^T = B A^T while the visible output remains row-major [M,N].
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &weight_layout_, CUDA_R_16BF, kK, kN, kK),
                           "create canonical BF16 weight layout [K,N]");
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

// Trusted scalar decoder with canonical contiguous [N,K] output and the exact
// E2M1/E4M3FN-to-BF16 boundary as production.  It is intentionally slow and
// runs only outside timing to independently validate the optimized route.
__global__ void dequantize_nvfp4_contiguous_reference_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const canonical_bf16) {
  const std::size_t count = kN * kK;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const std::size_t n = index / kK;
    const std::size_t k = index - n * kK;
    const std::uint8_t packed =
        packed_weights[n * (kK / 2U) + k / 2U];
    const std::uint8_t nibble =
        (k & 1U) != 0U ? static_cast<std::uint8_t>(packed >> 4U)
                       : static_cast<std::uint8_t>(packed & 0x0fU);
    const std::uint8_t scale_code =
        block_scales[n * (kK / 16U) + k / 16U];
    const float value =
        decode_e2m1_device(nibble) * decode_e4m3fn_device(scale_code);
    canonical_bf16[index] = __float2bfloat16_rn(value);
  }
}

// Production-shaped large-M staging route: one CTA owns one canonical N row.
// Every warp prefetches ten 32-byte packed spans plus aligned four-scale words
// before decoding, then writes adjacent uint32 BF16 pairs.  The cuBLASLt weight
// operand is transposed by its descriptor, so no physical [N,K] -> [K,N]
// transpose is required.
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
    const cublasLtMatmulAlgo_t& algorithm, const float alpha,
    const __nv_bfloat16* const persistent_weight,
    const __nv_bfloat16* const activation, __nv_bfloat16* const output,
    void* const workspace, const std::size_t workspace_bytes,
    const cudaStream_t stream, const std::string& label) {
  constexpr float kBeta = 0.0F;
  return test.lt_ok(
      cublasLtMatmul(
          lt.handle(), lt.operation(), &alpha, persistent_weight,
          lt.weight_layout(), activation, lt.activation_layout(), &kBeta,
          output, lt.output_layout(), output, lt.output_layout(), &algorithm,
          workspace, workspace_bytes, stream),
      label);
}

[[nodiscard]] bool launch_dequantize_contiguous(
    TestContext& test, const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const transient_weight, const cudaStream_t stream,
    const std::string& label) {
  constexpr unsigned int kThreads = 256U;
  dequantize_nvfp4_contiguous_kernel<<<static_cast<unsigned int>(kN),
                                       kThreads, 0, stream>>>(
      packed_weights, block_scales, transient_weight);
  return test.cuda_ok(cudaGetLastError(), label);
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
    if (!launch_lt(test, lt, algorithm, 1.0F, persistent_weight, activation,
                   output, workspace, workspace_bytes, stream,
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
    ready = launch_lt(test, lt, algorithm, 1.0F, persistent_weight,
                      activation, output, workspace, workspace_bytes, stream,
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

[[nodiscard]] double measure_dequantize(
    TestContext& test, const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const transient_weight, const cudaStream_t stream,
    const int warmups, const int iterations, const std::string& label) {
  for (int warmup = 0; warmup < warmups; ++warmup) {
    if (!launch_dequantize_contiguous(
            test, packed_weights, block_scales, transient_weight, stream,
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
    ready = launch_dequantize_contiguous(
        test, packed_weights, block_scales, transient_weight, stream,
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

[[nodiscard]] double measure_inclusive(
    TestContext& test, const LtObjects& lt,
    const cublasLtMatmulAlgo_t& algorithm,
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const transient_weight,
    const __nv_bfloat16* const activation, __nv_bfloat16* const output,
    void* const workspace, const std::size_t workspace_bytes,
    const cudaStream_t stream, const float weight_scale_2,
    const int warmups, const int iterations, const std::string& label) {
  const auto launch_chain = [&](const std::string& iteration_label) {
    bool launched = launch_dequantize_contiguous(
        test, packed_weights, block_scales, transient_weight, stream,
        iteration_label + " dequantize");
    launched = launched && launch_lt(
                               test, lt, algorithm, weight_scale_2,
                               transient_weight,
                               activation, output, workspace, workspace_bytes,
                               stream, iteration_label + " cuBLASLt");
    return launched;
  };
  for (int warmup = 0; warmup < warmups; ++warmup) {
    if (!launch_chain(label + " warmup " + std::to_string(warmup))) {
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
    ready = launch_chain(label + " measured " + std::to_string(iteration));
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
  DeviceBuffer<__nv_bfloat16> dequant_reference;
  DeviceBuffer<std::uint8_t> canonical_packed_weight;
  DeviceBuffer<std::uint8_t> canonical_block_scale;
  DeviceBuffer<std::uint8_t> workspace;
  DeviceBuffer<unsigned long long> validation;
  bool ready = activation.allocate(test, kAElements, "allocate BF16 A");
  ready = ready && persistent_weight.allocate(test, kBElements,
                                               "allocate persistent BF16 B");
  ready = ready && output.allocate(test, kCElements, "allocate BF16 C");
  ready = ready && replay_reference.allocate(test, kCElements,
                                              "allocate BF16 replay C");
  ready = ready && dequant_reference.allocate(
                       test, kBElements, "allocate BF16 dequant reference");
  ready = ready && canonical_packed_weight.allocate(
                       test, kPackedWeightBytes,
                       "allocate canonical NVFP4 packed weight");
  ready = ready && canonical_block_scale.allocate(
                       test, kBlockScaleBytes,
                       "allocate canonical NVFP4 block scales");
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
  ready = launch_lt(test, lt, selected, 1.0F, persistent_weight.get(),
                    activation.get(), output.get(), workspace.get(),
                    kWorkspaceBytes, stream, "validation reference");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(replay_reference.get(), output.get(),
                                       kCElements * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToDevice, stream),
                       "copy replay reference");
  ready = ready && launch_lt(test, lt, selected, 1.0F,
                             persistent_weight.get(), activation.get(),
                             output.get(), workspace.get(), kWorkspaceBytes,
                             stream, "validation replay");
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
  double persistent_median_milliseconds =
      std::numeric_limits<double>::quiet_NaN();
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
    persistent_median_milliseconds = median(round_milliseconds);
    const auto [minimum, maximum] =
        std::minmax_element(round_milliseconds.begin(),
                            round_milliseconds.end());
    std::cout << "CUBLASLT_GATE_C512_FINAL: selected_index=" << selected_index
              << " rounds=" << kFormalRounds
              << " median_milliseconds=" << persistent_median_milliseconds
              << " minimum_milliseconds=" << *minimum
              << " maximum_milliseconds=" << *maximum
              << " median_TFLOP_per_s="
              << kUsefulFlops /
                     (persistent_median_milliseconds * 1.0e9)
              << " directional_speedup_vs_fresh_production_M128="
              << kProductionGateReferenceMilliseconds /
                     persistent_median_milliseconds
              << " comparison_scope=absolute_persistent_BF16_control"
              << " dequantization_timed=false gate=PASS\n";
  }

  // Build a canonical production-shaped NVFP4 Gate matrix in [N,K] order,
  // decode it through both the trusted scalar route and the optimized direct
  // route, then time the entire transient-dequantization + cuBLASLt chain.
  const auto fill_canonical = [&](DeviceBuffer<std::uint8_t>& buffer,
                                  const std::size_t count,
                                  const std::uint32_t salt,
                                  const bool block_scales,
                                  const std::string& label) {
    const std::size_t blocks =
        (count + kFillThreads - 1U) / kFillThreads;
    fill_canonical_nvfp4_kernel<<<static_cast<unsigned int>(blocks),
                                  kFillThreads, 0, stream>>>(
        buffer.get(), count, salt, block_scales);
    return test.cuda_ok(cudaGetLastError(), label);
  };
  ready = ready && fill_canonical(
                       canonical_packed_weight, kPackedWeightBytes,
                       0x6a09'e667U, false,
                       "launch canonical packed NVFP4 fill");
  ready = ready && fill_canonical(
                       canonical_block_scale, kBlockScaleBytes,
                       0xbb67'ae85U, true,
                       "launch canonical E4M3FN scale fill");
  constexpr unsigned int kReferenceBlocks = 65'535U;
  dequantize_nvfp4_contiguous_reference_kernel<<<kReferenceBlocks,
                                                 kFillThreads, 0, stream>>>(
      canonical_packed_weight.get(), canonical_block_scale.get(),
      dequant_reference.get());
  ready = ready && test.cuda_ok(cudaGetLastError(),
                                "launch trusted NVFP4 decoder");
  ready = ready && launch_dequantize_contiguous(
                       test, canonical_packed_weight.get(),
                       canonical_block_scale.get(), persistent_weight.get(),
                       stream, "launch contiguous NVFP4 decoder validation");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(validation.get(), 0,
                                       3U * sizeof(unsigned long long), stream),
                       "zero NVFP4 decode validation counts");
  validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
      persistent_weight.get(), dequant_reference.get(), kBElements,
      validation.get(), validation.get() + 1U, validation.get() + 2U);
  ready = ready && test.cuda_ok(cudaGetLastError(),
                                "launch NVFP4 decode validation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(host_validation.data(), validation.get(),
                                       sizeof(host_validation),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy NVFP4 decode validation counts");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "NVFP4 decode validation sync");
  test.expect(host_validation[0] == 0U,
              "contiguous NVFP4 decoder is bit exact versus trusted decoder");
  test.expect(host_validation[1] == 0U,
              "every contiguous NVFP4 decoded BF16 value is finite");
  test.expect(host_validation[2] != 0U,
              "contiguous NVFP4 decoded checksum is nonzero");
  std::cout << "NVFP4_DEQUANT_C512_VALIDATION: mismatches="
            << host_validation[0] << '/' << kBElements
            << " nonfinite=" << host_validation[1]
            << " encoded_sum=" << host_validation[2]
            << " comparison=trusted_scalar_exact_bf16"
            << " gate="
            << ((host_validation[0] == 0U && host_validation[1] == 0U &&
                 host_validation[2] != 0U)
                    ? "PASS"
                    : "FAIL")
            << '\n';

  // A second full decode must replay the same trusted BF16 matrix.  This also
  // rules out stale shared-memory or incomplete-grid behavior before timing.
  ready = ready && launch_dequantize_contiguous(
                       test, canonical_packed_weight.get(),
                       canonical_block_scale.get(), persistent_weight.get(),
                       stream, "launch contiguous NVFP4 decoder replay");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(validation.get(), 0,
                                       3U * sizeof(unsigned long long), stream),
                       "zero NVFP4 decode replay counts");
  validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
      persistent_weight.get(), dequant_reference.get(), kBElements,
      validation.get(), validation.get() + 1U, validation.get() + 2U);
  ready = ready && test.cuda_ok(cudaGetLastError(),
                                "launch NVFP4 decode replay validation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(host_validation.data(), validation.get(),
                                       sizeof(host_validation),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy NVFP4 decode replay counts");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "NVFP4 decode replay sync");
  test.expect(host_validation[0] == 0U,
              "contiguous NVFP4 decoder replay remains bit exact");
  test.expect(host_validation[1] == 0U,
              "every replayed NVFP4 decoded BF16 value is finite");
  std::cout << "NVFP4_DEQUANT_C512_REPLAY: mismatches="
            << host_validation[0] << '/' << kBElements
            << " nonfinite=" << host_validation[1]
            << " encoded_sum=" << host_validation[2]
            << " gate="
            << ((host_validation[0] == 0U && host_validation[1] == 0U)
                    ? "PASS"
                    : "FAIL")
            << '\n';

  // Validate the exact timed chain against the trusted scalar-dequantized
  // matrix.  Production applies the non-unit global weight scale as Lt alpha;
  // it is deliberately not rounded into the transient BF16 weights.
  ready = ready && launch_dequantize_contiguous(
                       test, canonical_packed_weight.get(),
                       canonical_block_scale.get(), persistent_weight.get(),
                       stream,
                       "launch inclusive candidate dequant");
  ready = ready && launch_lt(
                       test, lt, selected, kWeightScale2,
                       persistent_weight.get(), activation.get(), output.get(),
                       workspace.get(), kWorkspaceBytes, stream,
                       "launch inclusive candidate GEMM");
  ready = ready && launch_lt(
                       test, lt, selected, kWeightScale2,
                       dequant_reference.get(), activation.get(),
                       replay_reference.get(), workspace.get(),
                       kWorkspaceBytes, stream,
                       "launch trusted-dequant inclusive reference GEMM");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(validation.get(), 0,
                                       3U * sizeof(unsigned long long), stream),
                       "zero inclusive validation counts");
  validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
      output.get(), replay_reference.get(), kCElements, validation.get(),
      validation.get() + 1U, validation.get() + 2U);
  ready = ready && test.cuda_ok(cudaGetLastError(),
                                "launch inclusive replay validation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(host_validation.data(), validation.get(),
                                       sizeof(host_validation),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy inclusive validation counts");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "inclusive validation sync");
  test.expect(host_validation[0] == 0U,
              "inclusive output matches trusted scalar dequantization");
  test.expect(host_validation[1] == 0U,
              "every inclusive NVFP4 + cuBLASLt output is finite");
  test.expect(host_validation[2] != 0U,
              "inclusive NVFP4 + cuBLASLt checksum is nonzero");
  std::cout << "NVFP4_CUBLASLT_GATE_C512_VALIDATION: reference_mismatches="
            << host_validation[0] << '/' << kCElements
            << " nonfinite=" << host_validation[1]
            << " encoded_sum=" << host_validation[2]
            << " weight_scale_2=" << kWeightScale2
            << " scale_application=cuBLASLt_alpha_after_BF16_dequant"
            << " production_M128_bitwise_compared=false"
            << " gate="
            << ((host_validation[0] == 0U && host_validation[1] == 0U &&
                 host_validation[2] != 0U)
                    ? "PASS"
                    : "FAIL")
            << '\n';

  // The production prefill path is graph-driven.  Capture the exact
  // dequantize + selected-Lt chain, instantiate it, then require both the
  // first (cold) and second (warm) graph replays to match the trusted eager
  // output bit for bit.
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  bool graph_instantiated = false;
  bool graph_ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "begin NVFP4 inclusive graph capture");
  if (graph_ready) {
    const bool captured_dequant = launch_dequantize_contiguous(
        test, canonical_packed_weight.get(), canonical_block_scale.get(),
        persistent_weight.get(), stream, "capture contiguous NVFP4 dequant");
    const bool captured_lt =
        captured_dequant &&
        launch_lt(test, lt, selected, kWeightScale2,
                  persistent_weight.get(), activation.get(), output.get(),
                  workspace.get(), kWorkspaceBytes, stream,
                  "capture selected cuBLASLt GEMM");
    graph_ready = captured_dequant && captured_lt;
    const cudaError_t end_capture_status =
        cudaStreamEndCapture(stream, &graph);
    graph_ready = test.cuda_ok(end_capture_status,
                               "end NVFP4 inclusive graph capture") &&
                  graph_ready;
  }

  std::size_t graph_node_count = 0U;
  if (graph_ready) {
    graph_ready = test.cuda_ok(
        cudaGraphGetNodes(graph, nullptr, &graph_node_count),
        "query NVFP4 inclusive graph nodes");
    test.expect(graph_node_count >= 2U,
                "inclusive graph contains dequant and GEMM nodes");
    graph_ready = graph_ready && graph_node_count >= 2U;
  }
  if (graph_ready) {
    graph_ready = test.cuda_ok(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0U),
        "instantiate NVFP4 inclusive graph");
    graph_instantiated = graph_ready && graph_exec != nullptr;
  }

  const auto validate_graph_replay =
      [&](const std::string& label,
          std::array<unsigned long long, 3U>& graph_validation) {
        bool replay_ready = test.cuda_ok(
            cudaGraphLaunch(graph_exec, stream), label + " launch");
        replay_ready =
            replay_ready &&
            test.cuda_ok(cudaMemsetAsync(
                             validation.get(), 0,
                             3U * sizeof(unsigned long long), stream),
                         label + " zero validation counts");
        validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
            output.get(), replay_reference.get(), kCElements,
            validation.get(), validation.get() + 1U, validation.get() + 2U);
        replay_ready =
            replay_ready &&
            test.cuda_ok(cudaGetLastError(), label + " validate output");
        replay_ready =
            replay_ready &&
            test.cuda_ok(cudaMemcpyAsync(
                             graph_validation.data(), validation.get(),
                             sizeof(graph_validation), cudaMemcpyDeviceToHost,
                             stream),
                         label + " copy validation counts");
        replay_ready = replay_ready && test.cuda_ok(
                                           cudaStreamSynchronize(stream),
                                           label + " synchronize");
        return replay_ready;
      };

  std::array<unsigned long long, 3U> cold_graph_validation{};
  std::array<unsigned long long, 3U> warm_graph_validation{};
  if (graph_ready) {
    graph_ready = validate_graph_replay("cold graph replay",
                                        cold_graph_validation);
  }
  if (graph_ready) {
    graph_ready = validate_graph_replay("warm graph replay",
                                        warm_graph_validation);
  }
  if (graph_exec != nullptr) {
    graph_ready = test.cuda_ok(cudaGraphExecDestroy(graph_exec),
                               "destroy NVFP4 inclusive graph executable") &&
                  graph_ready;
  }
  if (graph != nullptr) {
    graph_ready = test.cuda_ok(cudaGraphDestroy(graph),
                               "destroy NVFP4 inclusive graph") &&
                  graph_ready;
  }
  ready = ready && graph_ready;
  test.expect(cold_graph_validation[0] == 0U,
              "cold graph output matches trusted eager output");
  test.expect(cold_graph_validation[1] == 0U,
              "cold graph output is finite");
  test.expect(cold_graph_validation[2] != 0U,
              "cold graph output checksum is nonzero");
  test.expect(warm_graph_validation[0] == 0U,
              "warm graph output matches trusted eager output");
  test.expect(warm_graph_validation[1] == 0U,
              "warm graph output is finite");
  test.expect(warm_graph_validation[2] == cold_graph_validation[2],
              "cold and warm graph checksums are identical");
  std::cout << "NVFP4_CUBLASLT_GATE_C512_GRAPH: nodes="
            << graph_node_count
            << " instantiated=" << (graph_instantiated ? "true" : "false")
            << " cold_reference_mismatches=" << cold_graph_validation[0]
            << '/' << kCElements
            << " cold_nonfinite=" << cold_graph_validation[1]
            << " cold_encoded_sum=" << cold_graph_validation[2]
            << " warm_reference_mismatches=" << warm_graph_validation[0]
            << '/' << kCElements
            << " warm_nonfinite=" << warm_graph_validation[1]
            << " warm_encoded_sum=" << warm_graph_validation[2]
            << " weight_scale_2=" << kWeightScale2
            << " production_M128_bitwise_compared=false"
            << " gate="
            << ((graph_ready && cold_graph_validation[0] == 0U &&
                 cold_graph_validation[1] == 0U &&
                 cold_graph_validation[2] != 0U &&
                 warm_graph_validation[0] == 0U &&
                 warm_graph_validation[1] == 0U &&
                 warm_graph_validation[2] == cold_graph_validation[2])
                    ? "PASS"
                    : "FAIL")
            << '\n';

  std::vector<double> dequant_round_milliseconds;
  std::vector<double> inclusive_round_milliseconds;
  dequant_round_milliseconds.reserve(kFormalRounds);
  inclusive_round_milliseconds.reserve(kFormalRounds);
  for (int round = 0; ready && round < kFormalRounds; ++round) {
    double dequant_milliseconds = 0.0;
    double inclusive_milliseconds = 0.0;
    // Alternate order across rounds to keep either measurement from always
    // receiving the first thermal/frequency position.
    if ((round & 1) == 0) {
      dequant_milliseconds = measure_dequantize(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          persistent_weight.get(), stream, kFormalWarmups, kFormalIterations,
          "NVFP4 dequant round " + std::to_string(round + 1));
      inclusive_milliseconds = measure_inclusive(
          test, lt, selected, canonical_packed_weight.get(),
          canonical_block_scale.get(), persistent_weight.get(),
          activation.get(), output.get(), workspace.get(), kWorkspaceBytes,
          stream, kWeightScale2, kFormalWarmups, kFormalIterations,
          "NVFP4 inclusive round " + std::to_string(round + 1));
    } else {
      inclusive_milliseconds = measure_inclusive(
          test, lt, selected, canonical_packed_weight.get(),
          canonical_block_scale.get(), persistent_weight.get(),
          activation.get(), output.get(), workspace.get(), kWorkspaceBytes,
          stream, kWeightScale2, kFormalWarmups, kFormalIterations,
          "NVFP4 inclusive round " + std::to_string(round + 1));
      dequant_milliseconds = measure_dequantize(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          persistent_weight.get(), stream, kFormalWarmups, kFormalIterations,
          "NVFP4 dequant round " + std::to_string(round + 1));
    }
    const bool timing_ok = std::isfinite(dequant_milliseconds) &&
                           dequant_milliseconds > 0.0 &&
                           std::isfinite(inclusive_milliseconds) &&
                           inclusive_milliseconds > 0.0;
    ready = ready && timing_ok;
    test.expect(timing_ok,
                "inclusive NVFP4 timing is finite and positive");
    if (ready) {
      dequant_round_milliseconds.push_back(dequant_milliseconds);
      inclusive_round_milliseconds.push_back(inclusive_milliseconds);
      std::cout << "NVFP4_CUBLASLT_GATE_C512_ROUND: round=" << round + 1
                << " iterations=" << kFormalIterations
                << " dequant_milliseconds=" << dequant_milliseconds
                << " persistent_cuBLASLt_median_milliseconds="
                << persistent_median_milliseconds
                << " inclusive_milliseconds=" << inclusive_milliseconds
                << " inclusive_speedup_vs_fresh_production_M128="
                << kProductionGateReferenceMilliseconds /
                       inclusive_milliseconds
                << '\n';
    }
  }

  if (inclusive_round_milliseconds.size() ==
          static_cast<std::size_t>(kFormalRounds) &&
      dequant_round_milliseconds.size() ==
          static_cast<std::size_t>(kFormalRounds)) {
    const double dequant_median = median(dequant_round_milliseconds);
    const double inclusive_median = median(inclusive_round_milliseconds);
    const auto [dequant_minimum, dequant_maximum] =
        std::minmax_element(dequant_round_milliseconds.begin(),
                            dequant_round_milliseconds.end());
    const auto [inclusive_minimum, inclusive_maximum] =
        std::minmax_element(inclusive_round_milliseconds.begin(),
                            inclusive_round_milliseconds.end());
    const double inclusive_speedup =
        kProductionGateReferenceMilliseconds / inclusive_median;
    constexpr double kMinimumDequantBytes =
        static_cast<double>(kPackedWeightBytes + kBlockScaleBytes +
                            kBElements * sizeof(__nv_bfloat16));
    const double dequant_effective_gigabytes_per_second =
        kMinimumDequantBytes / (dequant_median * 1.0e6);
    const bool speed_gate = inclusive_speedup >= kRequiredInclusiveSpeedup;
    test.expect(speed_gate,
                "inclusive NVFP4 + cuBLASLt reaches the 1.22x Gate target");
    std::cout << "NVFP4_CUBLASLT_GATE_C512_FINAL: selected_index="
              << selected_index << " rounds=" << kFormalRounds
              << " dequant_median_milliseconds=" << dequant_median
              << " dequant_minimum_milliseconds=" << *dequant_minimum
              << " dequant_maximum_milliseconds=" << *dequant_maximum
              << " dequant_minimum_traffic_GB_per_s="
              << dequant_effective_gigabytes_per_second
              << " persistent_cuBLASLt_median_milliseconds="
              << persistent_median_milliseconds
              << " inclusive_median_milliseconds=" << inclusive_median
              << " inclusive_minimum_milliseconds=" << *inclusive_minimum
              << " inclusive_maximum_milliseconds=" << *inclusive_maximum
              << " inclusive_TFLOP_per_s="
              << kUsefulFlops / (inclusive_median * 1.0e9)
              << " inclusive_speedup_vs_fresh_production_M128="
              << inclusive_speedup
              << " required_speedup=" << kRequiredInclusiveSpeedup
              << " comparison_scope=canonical_NVFP4_dequant_plus_cuBLASLt"
              << " weight_scale_2=" << kWeightScale2
              << " scale_application=cuBLASLt_alpha_after_BF16_dequant"
              << " production_M128_bitwise_compared=false"
              << " dequantization_timed=true gate="
              << (speed_gate ? "PASS" : "FAIL") << '\n';
  }

  if (stream != nullptr) {
    (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy stream");
  }
  if (!ready) {
    test.expect(false, "NVFP4 inclusive ceiling completed");
  }
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " NVFP4 Gate ceiling assertion(s) failed\n";
    return 1;
  }
  std::cout << "Canonical-NVFP4 inclusive Gate C512 ceiling passed\n";
  return 0;
}
