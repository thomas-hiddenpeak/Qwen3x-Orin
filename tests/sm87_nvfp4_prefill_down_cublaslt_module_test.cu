#include "q3x/kernels/sm87_nvfp4_prefill_cublaslt.h"
#include "q3x/kernels/sm87_weight_only_gemv.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace q3x::kernels {
[[nodiscard]] int
query_sm87_nvfp4_prefill_down_cublaslt_dequant_resources_test_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes_per_thread,
    int* active_blocks_per_sm) noexcept;
}  // namespace q3x::kernels

namespace {

constexpr std::size_t kTokens = 512U;
constexpr std::size_t kRows = 5'120U;
constexpr std::size_t kColumns = 17'408U;
constexpr std::size_t kPackedBytes = kRows * (kColumns / 2U);
constexpr std::size_t kScaleBytes = kRows * (kColumns / 16U);
constexpr std::size_t kActivationBytes =
    kTokens * kColumns * sizeof(std::uint16_t);
constexpr std::size_t kOutputBytes =
    kTokens * kRows * sizeof(std::uint16_t);
constexpr std::size_t kScratchBytes =
    kRows * kColumns * sizeof(std::uint16_t);
// cudaMalloc is at least 256-byte aligned. Offset guarded payloads by 528
// bytes so every valid module operand is 16-byte aligned but deliberately not
// 256-byte aligned, locking the alignment advertised to cuBLASLt heuristics.
constexpr std::size_t kGuardBytes = 528U;
constexpr std::uint8_t kGuardValue = 0xa5U;
constexpr float kWeightScale2 = 1.25F;

static_assert(kPackedBytes == 44'564'480U);
static_assert(kScaleBytes == 5'570'560U);
static_assert(kActivationBytes == 17'825'792U);
static_assert(kOutputBytes == 5'242'880U);
static_assert(kScratchBytes == 178'257'920U);

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

class DeviceBytes {
 public:
  DeviceBytes() = default;
  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;

  ~DeviceBytes() {
    if (base_ != nullptr) {
      (void)cudaFree(base_);
    }
  }

  [[nodiscard]] bool allocate_guarded(TestContext& test,
                                      const std::size_t payload_bytes,
                                      const std::string& label) {
    payload_bytes_ = payload_bytes;
    const std::size_t allocation_bytes =
        payload_bytes + 2U * kGuardBytes;
    if (!test.cuda_ok(cudaMalloc(reinterpret_cast<void**>(&base_),
                                 allocation_bytes),
                      "allocate " + label)) {
      return false;
    }
    payload_ = base_ + kGuardBytes;
    return test.cuda_ok(cudaMemset(base_, kGuardValue, allocation_bytes),
                        "initialize " + label + " guards");
  }

  [[nodiscard]] bool allocate(TestContext& test,
                              const std::size_t bytes,
                              const std::string& label) {
    payload_bytes_ = bytes;
    if (!test.cuda_ok(
            cudaMalloc(reinterpret_cast<void**>(&base_), bytes),
            "allocate " + label)) {
      return false;
    }
    payload_ = base_;
    return true;
  }

  [[nodiscard]] std::uint8_t* payload() noexcept { return payload_; }
  [[nodiscard]] const std::uint8_t* payload() const noexcept {
    return payload_;
  }
  [[nodiscard]] const std::uint8_t* base() const noexcept { return base_; }
  [[nodiscard]] std::size_t payload_bytes() const noexcept {
    return payload_bytes_;
  }

 private:
  std::uint8_t* base_ = nullptr;
  std::uint8_t* payload_ = nullptr;
  std::size_t payload_bytes_ = 0U;
};

[[nodiscard]] std::uint32_t host_hash(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7feb'352dU;
  value ^= value >> 15U;
  value *= 0x846c'a68bU;
  value ^= value >> 16U;
  return value;
}

__device__ __forceinline__ std::uint32_t device_hash(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7feb'352dU;
  value ^= value >> 15U;
  value *= 0x846c'a68bU;
  value ^= value >> 16U;
  return value;
}

__global__ void fill_bytes_kernel(std::uint8_t* const values,
                                  const std::size_t count,
                                  const std::uint32_t seed) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    values[index] = static_cast<std::uint8_t>(
        device_hash(static_cast<std::uint32_t>(index) ^ seed));
  }
}

__global__ void fill_scales_kernel(std::uint8_t* const scales,
                                   const std::size_t count) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    // Positive, finite, modest E4M3FN values keep the GEMM output finite.
    scales[index] = static_cast<std::uint8_t>(
        0x18U + (device_hash(static_cast<std::uint32_t>(index)) & 0x0fU));
  }
}

__global__ void fill_activations_kernel(std::uint16_t* const activations,
                                        const std::size_t count) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int centered = static_cast<int>(
        device_hash(static_cast<std::uint32_t>(index) ^ 0x91e1'0da5U) %
        33U) - 16;
    const __nv_bfloat16 value =
        __float2bfloat16_rn(static_cast<float>(centered) * 0.03125F);
    activations[index] = __bfloat16_as_ushort(value);
  }
}

__device__ __forceinline__ float decode_e4m3fn(const std::uint8_t bits) {
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

__device__ __forceinline__ float decode_e2m1(const std::uint8_t nibble) {
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

__global__ void scalar_dequantize_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    std::uint16_t* const canonical_bf16) {
  constexpr std::size_t kElements = kRows * kColumns;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < kElements;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t row = index / kColumns;
    const std::size_t column = index - row * kColumns;
    const std::uint8_t packed =
        packed_weights[row * (kColumns / 2U) + column / 2U];
    const std::uint8_t nibble =
        (column & 1U) == 0U ? (packed & 0x0fU) : (packed >> 4U);
    const std::uint8_t scale =
        block_scales[row * (kColumns / 16U) + column / 16U];
    canonical_bf16[index] = __bfloat16_as_ushort(
        __float2bfloat16_rn(decode_e2m1(nibble) * decode_e4m3fn(scale)));
  }
}

struct CompareStats {
  unsigned long long mismatches;
  unsigned long long left_nonfinite;
  unsigned long long right_nonfinite;
};

__device__ __forceinline__ bool bf16_nonfinite(const std::uint16_t bits) {
  return (bits & 0x7f80U) == 0x7f80U;
}

__global__ void compare_bf16_kernel(const std::uint16_t* const left,
                                    const std::uint16_t* const right,
                                    const std::size_t count,
                                    CompareStats* const stats) {
  unsigned long long mismatches = 0U;
  unsigned long long left_nonfinite = 0U;
  unsigned long long right_nonfinite = 0U;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::uint16_t left_bits = left[index];
    const std::uint16_t right_bits = right[index];
    mismatches += left_bits != right_bits ? 1U : 0U;
    left_nonfinite += bf16_nonfinite(left_bits) ? 1U : 0U;
    right_nonfinite += bf16_nonfinite(right_bits) ? 1U : 0U;
  }
  if (mismatches != 0U) {
    atomicAdd(&stats->mismatches, mismatches);
  }
  if (left_nonfinite != 0U) {
    atomicAdd(&stats->left_nonfinite, left_nonfinite);
  }
  if (right_nonfinite != 0U) {
    atomicAdd(&stats->right_nonfinite, right_nonfinite);
  }
}

__global__ void compare_bytes_kernel(const std::uint8_t* const left,
                                     const std::uint8_t* const right,
                                     const std::size_t count,
                                     unsigned long long* const mismatches) {
  unsigned long long local = 0U;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    local += left[index] != right[index] ? 1U : 0U;
  }
  if (local != 0U) {
    atomicAdd(mismatches, local);
  }
}

[[nodiscard]] bool validate_guards(TestContext& test,
                                   const DeviceBytes& buffer,
                                   const std::string& label) {
  std::array<std::uint8_t, kGuardBytes> prefix{};
  std::array<std::uint8_t, kGuardBytes> suffix{};
  bool ready = test.cuda_ok(
      cudaMemcpy(prefix.data(), buffer.base(), kGuardBytes,
                 cudaMemcpyDeviceToHost),
      "copy " + label + " prefix guard");
  ready = test.cuda_ok(
              cudaMemcpy(suffix.data(),
                         buffer.payload() + buffer.payload_bytes(),
                         kGuardBytes, cudaMemcpyDeviceToHost),
              "copy " + label + " suffix guard") &&
          ready;
  const auto intact = [](const auto& values) {
    return std::all_of(values.begin(), values.end(), [](const auto value) {
      return value == kGuardValue;
    });
  };
  test.expect(intact(prefix), label + " prefix guard remains intact");
  test.expect(intact(suffix), label + " suffix guard remains intact");
  return ready && intact(prefix) && intact(suffix);
}

[[nodiscard]] CompareStats compare_bf16(
    TestContext& test, const std::uint16_t* const left,
    const std::uint16_t* const right, const std::size_t count,
    CompareStats* const device_stats, cudaStream_t stream,
    const std::string& label) {
  CompareStats host{};
  if (!test.cuda_ok(cudaMemsetAsync(device_stats, 0, sizeof(CompareStats),
                                    stream),
                    "clear " + label + " comparison stats")) {
    return host;
  }
  compare_bf16_kernel<<<1024U, 256U, 0U, stream>>>(
      left, right, count, device_stats);
  if (!test.cuda_ok(cudaGetLastError(), "launch " + label + " comparison")) {
    return host;
  }
  if (!test.cuda_ok(cudaMemcpyAsync(&host, device_stats,
                                    sizeof(CompareStats),
                                    cudaMemcpyDeviceToHost, stream),
                    "copy " + label + " comparison stats")) {
    return host;
  }
  (void)test.cuda_ok(cudaStreamSynchronize(stream),
                     "synchronize " + label + " comparison");
  return host;
}

[[nodiscard]] unsigned long long compare_bytes(
    TestContext& test, const std::uint8_t* const left,
    const std::uint8_t* const right, const std::size_t count,
    unsigned long long* const device_mismatches, cudaStream_t stream,
    const std::string& label) {
  unsigned long long host = 0U;
  if (!test.cuda_ok(cudaMemsetAsync(device_mismatches, 0,
                                    sizeof(unsigned long long), stream),
                    "clear " + label + " mismatch count")) {
    return host;
  }
  compare_bytes_kernel<<<1024U, 256U, 0U, stream>>>(
      left, right, count, device_mismatches);
  if (!test.cuda_ok(cudaGetLastError(), "launch " + label + " comparison")) {
    return host;
  }
  if (!test.cuda_ok(cudaMemcpyAsync(&host, device_mismatches,
                                    sizeof(unsigned long long),
                                    cudaMemcpyDeviceToHost, stream),
                    "copy " + label + " mismatch count")) {
    return host;
  }
  (void)test.cuda_ok(cudaStreamSynchronize(stream),
                     "synchronize " + label + " comparison");
  return host;
}

[[nodiscard]] bool expect_invalid_zero_node_capture(
    TestContext& test,
    q3x::kernels::Sm87Nvfp4PrefillDownCublasLtContext* const context,
    const std::uint8_t* const packed, const std::uint8_t* const scales,
    const std::uint16_t* const activations, std::uint16_t* const scratch,
    std::uint16_t* const output, cudaStream_t stream) {
  if (!test.cuda_ok(cudaStreamBeginCapture(stream,
                                           cudaStreamCaptureModeGlobal),
                    "begin invalid-call capture")) {
    return false;
  }
  const int invalid = static_cast<int>(cudaErrorInvalidValue);
  int invalid_case_count = 0;
  bool invalid_status_gate = true;
  const auto expect_invalid = [&](const int status, const char* const label) {
    ++invalid_case_count;
    test.expect(status == invalid, std::string(label) + " fails closed");
    invalid_status_gate = invalid_status_gate && status == invalid;
  };
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     nullptr, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "null context");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, nullptr, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "null packed weights");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, nullptr, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "null scales");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, 0.0F, activations, kTokens,
                     kRows, kColumns, scratch, kScratchBytes, output, stream),
                 "positive-zero scale");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, -0.0F, activations, kTokens,
                     kRows, kColumns, scratch, kScratchBytes, output, stream),
                 "negative-zero scale");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, -1.0F, activations, kTokens,
                     kRows, kColumns, scratch, kScratchBytes, output, stream),
                 "negative scale");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales,
                     std::numeric_limits<float>::quiet_NaN(), activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "NaN scale");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales,
                     std::numeric_limits<float>::infinity(), activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "infinite scale");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales,
                     -std::numeric_limits<float>::infinity(), activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "negative-infinite scale");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, nullptr, kTokens,
                     kRows, kColumns, scratch, kScratchBytes, output, stream),
                 "null activations");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens - 1U, kRows, kColumns, scratch, kScratchBytes,
                     output, stream),
                 "wrong token count");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows - 1U, kColumns, scratch, kScratchBytes,
                     output, stream),
                 "wrong rows");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns - 16U, scratch, kScratchBytes,
                     output, stream),
                 "wrong columns");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, nullptr, kScratchBytes, output,
                     stream),
                 "null scratch");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes - 2U,
                     output, stream),
                 "undersized scratch");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, nullptr,
                     stream),
                 "null output");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed + 1U, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "misaligned packed weights");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales + 1U, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "misaligned scales");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations + 1U,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "misaligned activations");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch + 1U, kScratchBytes,
                     output, stream),
                 "misaligned scratch");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes,
                     output + 1U, stream),
                 "misaligned output");
  // Every pair among packed/scales/activation/scratch/output is exercised as
  // a distinct 16-byte-offset partial overlap, locking full-span validation
  // rather than relying only on pointer-equality rejection.
  auto* const mutable_packed = const_cast<std::uint8_t*>(packed);
  auto* const mutable_scales = const_cast<std::uint8_t*>(scales);
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, packed + 16U, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "packed/scales partial alias");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2,
                     reinterpret_cast<const std::uint16_t*>(packed + 16U),
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "packed/activation partial alias");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns,
                     reinterpret_cast<std::uint16_t*>(mutable_packed + 16U),
                     kScratchBytes, output, stream),
                 "packed/scratch partial alias");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes,
                     reinterpret_cast<std::uint16_t*>(mutable_packed + 16U),
                     stream),
                 "packed/output partial alias");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2,
                     reinterpret_cast<const std::uint16_t*>(scales + 16U),
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "scales/activation partial alias");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns,
                     reinterpret_cast<std::uint16_t*>(mutable_scales + 16U),
                     kScratchBytes, output, stream),
                 "scales/scratch partial alias");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes,
                     reinterpret_cast<std::uint16_t*>(mutable_scales + 16U),
                     stream),
                 "scales/output partial alias");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns,
                     const_cast<std::uint16_t*>(activations + 8U),
                     kScratchBytes, output, stream),
                 "activation/scratch partial alias");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes,
                     const_cast<std::uint16_t*>(activations + 8U), stream),
                 "activation/output partial alias");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes,
                     scratch + 8U, stream),
                 "scratch/output partial alias");

  const std::uintptr_t aligned_near_max =
      std::numeric_limits<std::uintptr_t>::max() &
      ~static_cast<std::uintptr_t>(0x0fU);
  const auto* const wrapped_bytes =
      reinterpret_cast<const std::uint8_t*>(aligned_near_max);
  const auto* const wrapped_bf16 =
      reinterpret_cast<const std::uint16_t*>(aligned_near_max);
  auto* const mutable_wrapped_bf16 =
      reinterpret_cast<std::uint16_t*>(aligned_near_max);
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, wrapped_bytes, scales, kWeightScale2,
                     activations, kTokens, kRows, kColumns, scratch,
                     kScratchBytes, output, stream),
                 "wrapped packed range");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, wrapped_bytes, kWeightScale2,
                     activations, kTokens, kRows, kColumns, scratch,
                     kScratchBytes, output, stream),
                 "wrapped scale range");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, wrapped_bf16,
                     kTokens, kRows, kColumns, scratch, kScratchBytes, output,
                     stream),
                 "wrapped activation range");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, mutable_wrapped_bf16,
                     kScratchBytes, output, stream),
                 "wrapped scratch range");
  expect_invalid(q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
                     context, packed, scales, kWeightScale2, activations,
                     kTokens, kRows, kColumns, scratch, kScratchBytes,
                     mutable_wrapped_bf16, stream),
                 "wrapped output range");

  cudaGraph_t graph = nullptr;
  if (!test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                    "end invalid-call capture")) {
    return false;
  }
  std::size_t node_count = 0U;
  bool ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                            "count invalid-call graph nodes");
  test.expect(node_count == 0U,
              "all invalid calls enqueue zero graph nodes");
  ready = test.cuda_ok(cudaGraphDestroy(graph),
                       "destroy invalid-call graph") &&
          ready;
  std::cout << "NVFP4_PREFILL_DOWN_CUBLASLT_INVALID: cases="
            << invalid_case_count << " graph_nodes=" << node_count
            << " result="
            << ((ready && invalid_status_gate && node_count == 0U) ? "PASS"
                                                                    : "FAIL")
            << '\n';
  return ready && invalid_status_gate && node_count == 0U;
}

[[nodiscard]] bool expect_wrong_device_zero_node_capture(
    TestContext& test,
    q3x::kernels::Sm87Nvfp4PrefillDownCublasLtContext* const context,
    const std::uint8_t* const packed, const std::uint8_t* const scales,
    const std::uint16_t* const activations, std::uint16_t* const scratch,
    std::uint16_t* const output, const int owning_device,
    const int device_count) {
  if (device_count < 2) {
    std::cout << "NVFP4_PREFILL_DOWN_CUBLASLT_WRONG_DEVICE: "
                 "result=SKIP reason=single_device\n";
    return true;
  }
  const int other_device = owning_device == 0 ? 1 : 0;
  bool ready = test.cuda_ok(cudaSetDevice(other_device),
                            "select non-owning CUDA device");
  cudaStream_t other_stream = nullptr;
  ready = test.cuda_ok(
              cudaStreamCreateWithFlags(&other_stream, cudaStreamNonBlocking),
              "create non-owning-device stream") &&
          ready;
  cudaGraph_t graph = nullptr;
  if (ready) {
    ready = test.cuda_ok(
                cudaStreamBeginCapture(other_stream,
                                       cudaStreamCaptureModeGlobal),
                "begin wrong-device capture") &&
            ready;
  }
  int status = static_cast<int>(cudaErrorUnknown);
  if (ready) {
    status = q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
        context, packed, scales, kWeightScale2, activations, kTokens, kRows,
        kColumns, scratch, kScratchBytes, output, other_stream);
    test.expect(status == static_cast<int>(cudaErrorInvalidDevice),
                "wrong-device launch fails before enqueue");
    ready = test.cuda_ok(cudaStreamEndCapture(other_stream, &graph),
                         "end wrong-device capture") &&
            ready;
  }
  std::size_t node_count = 0U;
  if (ready && graph != nullptr) {
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                         "count wrong-device graph nodes") &&
            ready;
    test.expect(node_count == 0U,
                "wrong-device launch enqueues zero graph nodes");
  }
  if (graph != nullptr) {
    ready = test.cuda_ok(cudaGraphDestroy(graph),
                         "destroy wrong-device graph") &&
            ready;
  }
  if (other_stream != nullptr) {
    ready = test.cuda_ok(cudaStreamDestroy(other_stream),
                         "destroy non-owning-device stream") &&
            ready;
  }
  ready = test.cuda_ok(cudaSetDevice(owning_device),
                       "restore owning CUDA device") &&
          ready;
  std::cout << "NVFP4_PREFILL_DOWN_CUBLASLT_WRONG_DEVICE: status=" << status
            << " graph_nodes=" << node_count
            << " result="
            << ((ready && status == static_cast<int>(cudaErrorInvalidDevice) &&
                 node_count == 0U)
                    ? "PASS"
                    : "FAIL")
            << '\n';
  return ready && status == static_cast<int>(cudaErrorInvalidDevice) &&
         node_count == 0U;
}

}  // namespace

int main() {
  TestContext test;

  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA device unavailable\n";
    return 77;
  }
  int device = -1;
  if (!test.cuda_ok(cudaGetDevice(&device), "query current device")) {
    return 1;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, device),
                    "query current device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: exact Down module requires SM87\n";
    return 77;
  }

  int dequant_registers = 0;
  std::size_t dequant_static_shared = 0U;
  std::size_t dequant_local_bytes = 0U;
  int dequant_active_blocks = 0;
  const int resource_status = q3x::kernels::
      query_sm87_nvfp4_prefill_down_cublaslt_dequant_resources_test_cuda(
          &dequant_registers, &dequant_static_shared, &dequant_local_bytes,
          &dequant_active_blocks);
  test.expect(resource_status == static_cast<int>(cudaSuccess),
              "query Window8 dequant resources");
  test.expect(dequant_registers <= 40,
              "Window8 dequant stays within the six-CTA register budget");
  test.expect(dequant_static_shared == 0U,
              "Window8 dequant uses zero static shared memory");
  test.expect(dequant_local_bytes == 0U,
              "Window8 dequant uses zero local memory");
  test.expect(dequant_active_blocks >= 6,
              "Window8 dequant retains at least six active CTAs per SM");

  q3x::kernels::Sm87Nvfp4PrefillDownCublasLtContext* context = nullptr;
  test.expect(
      q3x::kernels::create_sm87_nvfp4_prefill_down_cublaslt_context(nullptr) ==
          static_cast<int>(cudaErrorInvalidValue),
      "factory rejects null output pointer");

  // Context construction may overlap across runners during process startup.
  // Every factory must independently find the real-checkpoint-pinned
  // zero-workspace configuration without timing or synthetic operands. The
  // runtime heuristic-list rank may differ and is deliberately not fixed.
  constexpr std::size_t kConcurrentContextCount = 2U;
  std::array<q3x::kernels::Sm87Nvfp4PrefillDownCublasLtContext*,
             kConcurrentContextCount>
      concurrent_contexts{};
  std::array<int, kConcurrentContextCount> concurrent_statuses{};
  std::array<std::thread, kConcurrentContextCount> context_threads;
  for (std::size_t index = 0U; index < kConcurrentContextCount; ++index) {
    context_threads[index] = std::thread([&, index]() {
      const cudaError_t set_device_status = cudaSetDevice(device);
      if (set_device_status != cudaSuccess) {
        concurrent_statuses[index] = static_cast<int>(set_device_status);
        return;
      }
      concurrent_statuses[index] = q3x::kernels::
          create_sm87_nvfp4_prefill_down_cublaslt_context(
              &concurrent_contexts[index]);
    });
  }
  for (auto& context_thread : context_threads) {
    context_thread.join();
  }
  for (std::size_t index = 0U; index < kConcurrentContextCount; ++index) {
    test.expect(concurrent_statuses[index] == static_cast<int>(cudaSuccess),
                "concurrent factory finds the pinned algorithm");
    test.expect(concurrent_contexts[index] != nullptr,
                "concurrent factory returns an opaque context");
    if (concurrent_contexts[index] != nullptr) {
      std::size_t concurrent_scratch_bytes = 0U;
      std::size_t concurrent_workspace_bytes =
          std::numeric_limits<std::size_t>::max();
      int concurrent_rank = -1;
      test.expect(
          q3x::kernels::query_sm87_nvfp4_prefill_down_cublaslt_context(
              concurrent_contexts[index], &concurrent_scratch_bytes,
              &concurrent_workspace_bytes, &concurrent_rank) ==
              static_cast<int>(cudaSuccess),
          "query concurrent exact-C512 module contract");
      test.expect(concurrent_scratch_bytes == kScratchBytes,
                  "concurrent context retains exact scratch contract");
      test.expect(concurrent_workspace_bytes == 0U,
                  "concurrent context retains zero-workspace algorithm");
      test.expect(concurrent_rank >= 0,
                  "concurrent context records the pinned algorithm rank");
    }
    q3x::kernels::destroy_sm87_nvfp4_prefill_down_cublaslt_context(
        concurrent_contexts[index]);
  }

  const int create_status =
      q3x::kernels::create_sm87_nvfp4_prefill_down_cublaslt_context(&context);
  test.expect(create_status == static_cast<int>(cudaSuccess),
              "create exact-C512 cuBLASLt context");
  test.expect(context != nullptr, "factory returns an opaque context");
  if (context == nullptr || create_status != static_cast<int>(cudaSuccess)) {
    return 1;
  }

  std::size_t scratch_bytes = 0U;
  std::size_t workspace_bytes = std::numeric_limits<std::size_t>::max();
  int heuristic_rank = -1;
  const int query_status =
      q3x::kernels::query_sm87_nvfp4_prefill_down_cublaslt_context(
          context, &scratch_bytes, &workspace_bytes, &heuristic_rank);
  test.expect(query_status == static_cast<int>(cudaSuccess),
              "query exact-C512 module contract");
  test.expect(scratch_bytes == kScratchBytes,
              "query reports exact 170 MiB transient scratch");
  test.expect(workspace_bytes == 0U,
              "selected cuBLASLt algorithm has zero workspace");
  test.expect(heuristic_rank >= 0,
              "factory records the pinned algorithm runtime rank");
  std::size_t ignored_scratch = 0U;
  std::size_t ignored_workspace = 0U;
  int ignored_rank = -1;
  test.expect(q3x::kernels::query_sm87_nvfp4_prefill_down_cublaslt_context(
                  nullptr, &ignored_scratch, &ignored_workspace,
                  &ignored_rank) == static_cast<int>(cudaErrorInvalidValue),
              "query rejects null context");
  test.expect(q3x::kernels::query_sm87_nvfp4_prefill_down_cublaslt_context(
                  context, nullptr, &ignored_workspace, &ignored_rank) ==
                  static_cast<int>(cudaErrorInvalidValue),
              "query rejects null scratch output pointer");
  test.expect(q3x::kernels::query_sm87_nvfp4_prefill_down_cublaslt_context(
                  context, &ignored_scratch, nullptr, &ignored_rank) ==
                  static_cast<int>(cudaErrorInvalidValue),
              "query rejects null workspace output pointer");
  test.expect(q3x::kernels::query_sm87_nvfp4_prefill_down_cublaslt_context(
                  context, &ignored_scratch, &ignored_workspace, nullptr) ==
                  static_cast<int>(cudaErrorInvalidValue),
              "query rejects null heuristic-rank output pointer");

  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create nonblocking stream")) {
    q3x::kernels::destroy_sm87_nvfp4_prefill_down_cublaslt_context(context);
    return 1;
  }

  DeviceBytes packed;
  DeviceBytes scales;
  DeviceBytes activations;
  DeviceBytes scratch;
  DeviceBytes scalar_reference;
  DeviceBytes output;
  DeviceBytes production_output;
  DeviceBytes scale_4_alignment_storage;
  DeviceBytes packed_snapshot;
  DeviceBytes scale_snapshot;
  DeviceBytes activation_snapshot;
  DeviceBytes compare_storage;
  DeviceBytes byte_compare_storage;
  bool allocated = packed.allocate_guarded(test, kPackedBytes, "packed") &&
                   scales.allocate_guarded(test, kScaleBytes, "scales") &&
                   activations.allocate_guarded(test, kActivationBytes,
                                                "activations") &&
                   scratch.allocate_guarded(test, kScratchBytes, "scratch") &&
                   scalar_reference.allocate_guarded(
                       test, kScratchBytes, "scalar reference") &&
                   output.allocate_guarded(test, kOutputBytes, "module output") &&
                   production_output.allocate_guarded(
                       test, kOutputBytes, "production output") &&
                   scale_4_alignment_storage.allocate(
                       test, kScaleBytes + 4U,
                       "4-byte-aligned scale contract storage") &&
                   packed_snapshot.allocate(test, kPackedBytes,
                                            "packed snapshot") &&
                   scale_snapshot.allocate(test, kScaleBytes,
                                           "scale snapshot") &&
                   activation_snapshot.allocate(test, kActivationBytes,
                                                "activation snapshot") &&
                   compare_storage.allocate(test, sizeof(CompareStats),
                                           "BF16 comparison stats") &&
                   byte_compare_storage.allocate(
                       test, sizeof(unsigned long long),
                       "byte comparison count");
  if (!allocated) {
    (void)cudaStreamDestroy(stream);
    q3x::kernels::destroy_sm87_nvfp4_prefill_down_cublaslt_context(context);
    return 1;
  }

  const auto is_16_not_256_aligned = [](const void* const pointer) {
    const std::uintptr_t address =
        reinterpret_cast<std::uintptr_t>(pointer);
    return address % 16U == 0U && address % 256U != 0U;
  };
  test.expect(is_16_not_256_aligned(scratch.payload()),
              "Lt A scratch is 16-byte but not 256-byte aligned");
  test.expect(is_16_not_256_aligned(activations.payload()),
              "Lt B activations are 16-byte but not 256-byte aligned");
  test.expect(is_16_not_256_aligned(output.payload()),
              "Lt C/D output is 16-byte but not 256-byte aligned");
  test.expect(is_16_not_256_aligned(packed.payload()),
              "packed weights are 16-byte but not 256-byte aligned");
  test.expect(is_16_not_256_aligned(scales.payload()),
              "block scales are 16-byte but not 256-byte aligned");

  fill_bytes_kernel<<<1024U, 256U, 0U, stream>>>(
      packed.payload(), kPackedBytes, host_hash(0x1234'5678U));
  fill_scales_kernel<<<1024U, 256U, 0U, stream>>>(scales.payload(),
                                                  kScaleBytes);
  auto* const scales_4_aligned =
      scale_4_alignment_storage.payload() + 4U;
  fill_scales_kernel<<<1024U, 256U, 0U, stream>>>(scales_4_aligned,
                                                  kScaleBytes);
  fill_activations_kernel<<<1024U, 256U, 0U, stream>>>(
      reinterpret_cast<std::uint16_t*>(activations.payload()),
      kActivationBytes / sizeof(std::uint16_t));
  bool ready = test.cuda_ok(cudaGetLastError(), "launch deterministic fills");
  ready = test.cuda_ok(cudaMemcpyAsync(packed_snapshot.payload(),
                                       packed.payload(), kPackedBytes,
                                       cudaMemcpyDeviceToDevice, stream),
                       "snapshot packed weights") &&
          ready;
  ready = test.cuda_ok(cudaMemcpyAsync(scale_snapshot.payload(),
                                       scales.payload(), kScaleBytes,
                                       cudaMemcpyDeviceToDevice, stream),
                       "snapshot scales") &&
          ready;
  ready = test.cuda_ok(cudaMemcpyAsync(activation_snapshot.payload(),
                                       activations.payload(),
                                       kActivationBytes,
                                       cudaMemcpyDeviceToDevice, stream),
                       "snapshot activations") &&
          ready;
  scalar_dequantize_kernel<<<65'535U, 256U, 0U, stream>>>(
      packed.payload(), scales.payload(),
      reinterpret_cast<std::uint16_t*>(scalar_reference.payload()));
  ready = test.cuda_ok(cudaGetLastError(), "launch scalar NVFP4 decode") &&
          ready;

  const int module_status =
      q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
          context, packed.payload(), scales.payload(), kWeightScale2,
          reinterpret_cast<const std::uint16_t*>(activations.payload()),
          kTokens, kRows, kColumns,
          reinterpret_cast<std::uint16_t*>(scratch.payload()), scratch_bytes,
          reinterpret_cast<std::uint16_t*>(output.payload()), stream);
  test.expect(module_status == static_cast<int>(cudaSuccess),
              "launch module eager path");

  const int production_status = q3x::kernels::
      launch_sm87_nvfp4_w4a16_whole_chunk_down_gemm_bf16_cuda(
          packed.payload(), scales.payload(), kWeightScale2,
          reinterpret_cast<const std::uint16_t*>(activations.payload()),
          kTokens, kRows, kColumns,
          reinterpret_cast<std::uint16_t*>(production_output.payload()),
          stream);
  test.expect(production_status == static_cast<int>(cudaSuccess),
              "launch production exact-C512 M128 Down path");
  ready = test.cuda_ok(cudaStreamSynchronize(stream),
                       "synchronize eager work") &&
          ready;

  auto* const device_stats =
      reinterpret_cast<CompareStats*>(compare_storage.payload());
  const CompareStats scalar_stats = compare_bf16(
      test, reinterpret_cast<const std::uint16_t*>(scratch.payload()),
      reinterpret_cast<const std::uint16_t*>(scalar_reference.payload()),
      kScratchBytes / sizeof(std::uint16_t), device_stats, stream,
      "scalar decode");
  test.expect(scalar_stats.mismatches == 0U,
              "direct dequant is bitwise-equal to scalar decode");
  test.expect(scalar_stats.left_nonfinite == 0U &&
                  scalar_stats.right_nonfinite == 0U,
              "direct and scalar dequant remain finite");

  const CompareStats production_stats = compare_bf16(
      test, reinterpret_cast<const std::uint16_t*>(output.payload()),
      reinterpret_cast<const std::uint16_t*>(production_output.payload()),
      kOutputBytes / sizeof(std::uint16_t), device_stats, stream,
      "production M128");
  test.expect(production_stats.mismatches == 0U,
              "module output is bitwise-equal to production M128");
  test.expect(production_stats.left_nonfinite == 0U &&
                  production_stats.right_nonfinite == 0U,
              "module and production outputs remain finite");

  const std::uintptr_t scales_4_aligned_address =
      reinterpret_cast<std::uintptr_t>(scales_4_aligned);
  test.expect(scales_4_aligned_address % 4U == 0U &&
                  scales_4_aligned_address % 16U != 0U,
              "positive scale operand exercises 4-byte but not 16-byte "
              "alignment");
  const int scale_4_alignment_status =
      q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
          context, packed.payload(), scales_4_aligned, kWeightScale2,
          reinterpret_cast<const std::uint16_t*>(activations.payload()),
          kTokens, kRows, kColumns,
          reinterpret_cast<std::uint16_t*>(scratch.payload()), scratch_bytes,
          reinterpret_cast<std::uint16_t*>(output.payload()), stream);
  test.expect(scale_4_alignment_status == static_cast<int>(cudaSuccess),
              "module accepts its documented 4-byte scale alignment");
  ready = test.cuda_ok(cudaStreamSynchronize(stream),
                       "synchronize 4-byte scale alignment launch") &&
          ready;
  const CompareStats scale_4_alignment_stats = compare_bf16(
      test, reinterpret_cast<const std::uint16_t*>(output.payload()),
      reinterpret_cast<const std::uint16_t*>(production_output.payload()),
      kOutputBytes / sizeof(std::uint16_t), device_stats, stream,
      "4-byte scale alignment");
  test.expect(scale_4_alignment_stats.mismatches == 0U,
              "4-byte-aligned scales preserve production-bitwise output");
  test.expect(scale_4_alignment_stats.left_nonfinite == 0U &&
                  scale_4_alignment_stats.right_nonfinite == 0U,
              "4-byte-aligned scale output remains finite");

  cudaGraph_t graph = nullptr;
  ready = test.cuda_ok(cudaStreamBeginCapture(stream,
                                              cudaStreamCaptureModeGlobal),
                       "begin valid module capture") &&
          ready;
  const int captured_status =
      q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
          context, packed.payload(), scales.payload(), kWeightScale2,
          reinterpret_cast<const std::uint16_t*>(activations.payload()),
          kTokens, kRows, kColumns,
          reinterpret_cast<std::uint16_t*>(scratch.payload()), scratch_bytes,
          reinterpret_cast<std::uint16_t*>(output.payload()), stream);
  test.expect(captured_status == static_cast<int>(cudaSuccess),
              "capture valid module launch");
  ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                       "end valid module capture") &&
          ready;

  std::size_t graph_node_count = 0U;
  ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &graph_node_count),
                       "count valid module graph nodes") &&
          ready;
  test.expect(graph_node_count == 2U,
              "valid module capture contains exactly two nodes");
  std::vector<cudaGraphNode_t> graph_nodes(graph_node_count);
  std::size_t copied_node_count = graph_node_count;
  ready = test.cuda_ok(cudaGraphGetNodes(graph, graph_nodes.data(),
                                         &copied_node_count),
                       "read valid module graph nodes") &&
          ready;
  std::size_t kernel_node_count = 0U;
  for (const cudaGraphNode_t node : graph_nodes) {
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    if (test.cuda_ok(cudaGraphNodeGetType(node, &type),
                     "query valid module graph node type")) {
      kernel_node_count += type == cudaGraphNodeTypeKernel ? 1U : 0U;
    }
  }
  test.expect(copied_node_count == 2U && kernel_node_count == 2U,
              "both captured module operations are kernel nodes");

  cudaGraphExec_t graph_exec = nullptr;
  ready = test.cuda_ok(cudaGraphInstantiate(&graph_exec, graph, nullptr,
                                            nullptr, 0U),
                       "instantiate valid module graph") &&
          ready;
  for (int iteration = 0; iteration < 2; ++iteration) {
    ready = test.cuda_ok(cudaMemsetAsync(output.payload(), 0x7f,
                                         kOutputBytes, stream),
                         "poison graph output") &&
            ready;
    ready = test.cuda_ok(cudaGraphLaunch(graph_exec, stream),
                         "launch valid module graph") &&
            ready;
    const CompareStats graph_stats = compare_bf16(
        test, reinterpret_cast<const std::uint16_t*>(output.payload()),
        reinterpret_cast<const std::uint16_t*>(production_output.payload()),
        kOutputBytes / sizeof(std::uint16_t), device_stats, stream,
        "graph replay");
    test.expect(graph_stats.mismatches == 0U,
                "cold and warm graph replay remain production-bitwise");
    test.expect(graph_stats.left_nonfinite == 0U &&
                    graph_stats.right_nonfinite == 0U,
                "cold and warm graph replay remain finite");
  }
  ready = test.cuda_ok(cudaGraphExecDestroy(graph_exec),
                       "destroy valid module graph executable") &&
          ready;
  ready = test.cuda_ok(cudaGraphDestroy(graph),
                       "destroy valid module graph") &&
          ready;

  const bool invalid_gate = expect_invalid_zero_node_capture(
      test, context, packed.payload(), scales.payload(),
      reinterpret_cast<const std::uint16_t*>(activations.payload()),
      reinterpret_cast<std::uint16_t*>(scratch.payload()),
      reinterpret_cast<std::uint16_t*>(output.payload()), stream);
  test.expect(invalid_gate, "invalid-call zero-node capture completes");
  const bool wrong_device_gate = expect_wrong_device_zero_node_capture(
      test, context, packed.payload(), scales.payload(),
      reinterpret_cast<const std::uint16_t*>(activations.payload()),
      reinterpret_cast<std::uint16_t*>(scratch.payload()),
      reinterpret_cast<std::uint16_t*>(output.payload()), device,
      device_count);
  test.expect(wrong_device_gate,
              "wrong-device zero-node condition completes or skips safely");

  auto* const byte_mismatches = reinterpret_cast<unsigned long long*>(
      byte_compare_storage.payload());
  const unsigned long long packed_mismatches = compare_bytes(
      test, packed.payload(), packed_snapshot.payload(), kPackedBytes,
      byte_mismatches, stream, "packed immutability");
  const unsigned long long scale_mismatches = compare_bytes(
      test, scales.payload(), scale_snapshot.payload(), kScaleBytes,
      byte_mismatches, stream, "scale immutability");
  const unsigned long long activation_mismatches = compare_bytes(
      test, activations.payload(), activation_snapshot.payload(),
      kActivationBytes, byte_mismatches, stream, "activation immutability");
  test.expect(packed_mismatches == 0U, "packed weights remain immutable");
  test.expect(scale_mismatches == 0U, "block scales remain immutable");
  test.expect(activation_mismatches == 0U,
              "activations remain immutable");

  bool guards_ok = validate_guards(test, packed, "packed") &&
                   validate_guards(test, scales, "scales") &&
                   validate_guards(test, activations, "activations") &&
                   validate_guards(test, scratch, "scratch") &&
                   validate_guards(test, scalar_reference,
                                   "scalar reference") &&
                   validate_guards(test, output, "module output") &&
                   validate_guards(test, production_output,
                                   "production output");
  test.expect(guards_ok, "all guarded allocations remain bounded");

  ready = test.cuda_ok(cudaStreamSynchronize(stream),
                       "final stream synchronization") &&
          ready;
  q3x::kernels::destroy_sm87_nvfp4_prefill_down_cublaslt_context(context);
  q3x::kernels::destroy_sm87_nvfp4_prefill_down_cublaslt_context(nullptr);
  ready = test.cuda_ok(cudaStreamDestroy(stream), "destroy stream") && ready;

  std::cout << "NVFP4_PREFILL_DOWN_CUBLASLT_MODULE: heuristic_rank="
            << heuristic_rank << " scratch_bytes=" << scratch_bytes
            << " workspace_bytes=" << workspace_bytes
            << " scalar_mismatches=" << scalar_stats.mismatches << '/'
            << (kScratchBytes / sizeof(std::uint16_t))
            << " production_M128_mismatches="
            << production_stats.mismatches << '/'
            << (kOutputBytes / sizeof(std::uint16_t))
            << " graph_nodes=" << graph_node_count
            << " graph_kernel_nodes=" << kernel_node_count
            << " invalid_graph_nodes=0"
            << " wrong_device="
            << (wrong_device_gate ? "PASS_OR_SINGLE_DEVICE_SKIP" : "FAIL")
            << " dequant_registers=" << dequant_registers
            << " dequant_static_shared_bytes=" << dequant_static_shared
            << " dequant_local_bytes=" << dequant_local_bytes
            << " dequant_active_blocks_per_sm=" << dequant_active_blocks
            << " guards=" << (guards_ok ? "PASS" : "FAIL")
            << " inputs_immutable="
            << ((packed_mismatches == 0U && scale_mismatches == 0U &&
                 activation_mismatches == 0U)
                    ? "PASS"
                    : "FAIL")
            << '\n';

  return ready && test.failures() == 0 ? 0 : 1;
}
