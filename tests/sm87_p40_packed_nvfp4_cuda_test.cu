#include "q3x/kernels/sm87_p40_packed_projection.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace q3x::kernels {

// These narrow launchers are intentionally test-only. They instantiate the
// production mainloop with one K64 cell and carry no performance authority.
int launch_sm87_p40_packed_nvfp4_gate_up_cell_test_cuda(
    const std::uint16_t* input, const std::uint8_t* packed_cell,
    float gate_scale, float up_scale, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

int launch_sm87_p40_packed_nvfp4_down_cell_test_cuda(
    const std::uint16_t* input, const std::uint8_t* packed_cell,
    float global_scale, std::uint16_t* residual_in_out,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels

namespace {

using q3x::kernels::Sm87P40PackedProjectionResources;
using q3x::kernels::Sm87P40PackedProjectionRole;

inline constexpr unsigned int kTokens = 64U;
inline constexpr unsigned int kInputFeatures = 64U;
inline constexpr unsigned int kOutputFeatures = 128U;
inline constexpr unsigned int kFragmentWeightBytes = 256U;
inline constexpr unsigned int kFragmentBytes = 288U;
inline constexpr unsigned int kGatePhysicalWarps = 8U;
inline constexpr unsigned int kDownPhysicalWarps = 4U;
inline constexpr std::size_t kGateCellBytes =
    4U * kGatePhysicalWarps * kFragmentBytes;
inline constexpr std::size_t kDownCellBytes =
    4U * kDownPhysicalWarps * kFragmentBytes;
inline constexpr std::size_t kOutputElements =
    kTokens * kOutputFeatures;
inline constexpr std::size_t kCanaryElements = 256U;
inline constexpr std::uint16_t kOutputCanary = 0xa55aU;
inline constexpr const char* kRunEnvironment =
    "Q3X_RUN_SM87_P40_PACKED_NVFP4_NUMERICAL";

struct TestContext final {
  int failures = 0;

  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  bool cuda_ok(const cudaError_t status, const std::string& operation) {
    expect(status == cudaSuccess,
           operation + ": " + cudaGetErrorString(status));
    return status == cudaSuccess;
  }
};

template <typename T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t count) noexcept {
    if (data_ != nullptr || count == 0U ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return cudaErrorInvalidValue;
    }
    count_ = count;
    return cudaMalloc(reinterpret_cast<void**>(&data_), bytes());
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

class Stream final {
 public:
  Stream() = default;
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  ~Stream() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  [[nodiscard]] cudaError_t create() noexcept {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

[[nodiscard]] constexpr std::uint8_t weight_code(
    const unsigned int branch, const unsigned int output,
    const unsigned int k) noexcept {
  return static_cast<std::uint8_t>(
      (branch * 11U + output * 5U + k * 7U) & 0x0fU);
}

[[nodiscard]] constexpr std::uint8_t scale_code(
    const unsigned int branch, const unsigned int output,
    const unsigned int k16) noexcept {
  // Across each pair of 128-column branches this covers all 256 E4M3FN raw
  // encodings, including both signed zero and both NaN encodings.
  return static_cast<std::uint8_t>(
      branch * 128U + output * 4U + k16);
}

void fill_cell(std::vector<std::uint8_t>* const cell,
               const unsigned int physical_warps,
               const bool gate_up) {
  for (unsigned int k16 = 0U; k16 < 4U; ++k16) {
    for (unsigned int physical_warp = 0U;
         physical_warp < physical_warps; ++physical_warp) {
      const unsigned int branch = gate_up ? physical_warp / 4U : 0U;
      const unsigned int local_warp = gate_up ? physical_warp % 4U
                                              : physical_warp;
      const std::size_t fragment_offset =
          (static_cast<std::size_t>(k16) * physical_warps + physical_warp) *
          kFragmentBytes;
      for (unsigned int row = 0U; row < 32U; ++row) {
        const unsigned int output = local_warp * 32U + row;
        for (unsigned int packed_k = 0U; packed_k < 8U; ++packed_k) {
          const unsigned int even_k = k16 * 16U + packed_k * 2U;
          const std::uint8_t low = weight_code(branch, output, even_k);
          const std::uint8_t high =
              weight_code(branch, output, even_k + 1U);
          (*cell)[fragment_offset + row * 8U + packed_k] =
              static_cast<std::uint8_t>(low | (high << 4U));
        }
        (*cell)[fragment_offset + kFragmentWeightBytes + row] =
            scale_code(branch, output, k16);
      }
    }
  }
}

[[nodiscard]] __device__ __forceinline__ float decode_e2m1(
    const std::uint8_t code) noexcept {
  const unsigned int sign =
      static_cast<unsigned int>(code & 0x08U) << 28U;
  const unsigned int magnitude = code & 0x07U;
  const unsigned int nonzero_mask =
      0U - static_cast<unsigned int>(magnitude != 0U);
  const unsigned int mantissa =
      ((magnitude & 1U) & static_cast<unsigned int>(magnitude > 1U))
      << 22U;
  const unsigned int finite =
      ((126U + (magnitude >> 1U)) << 23U) | mantissa;
  return __uint_as_float(sign | (finite & nonzero_mask));
}

[[nodiscard]] __device__ __forceinline__ float decode_e4m3fn(
    const std::uint8_t code) noexcept {
  const unsigned int sign =
      static_cast<unsigned int>(code & 0x80U) << 24U;
  const unsigned int magnitude = code & 0x7fU;
  if (magnitude == 0x7fU) {
    return __uint_as_float(sign | 0x7fc0'0000U);
  }
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return __uint_as_float(sign);
    }
    const unsigned int leading =
        mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    return __uint_as_float(
        sign | ((118U + leading) << 23U) |
        ((mantissa - (1U << leading)) << (23U - leading)));
  }
  return __uint_as_float(sign | ((120U + exponent) << 23U) |
                         (mantissa << 20U));
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16(
    const float value) noexcept {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) noexcept {
  return __bfloat162float(__ushort_as_bfloat16(bits));
}

[[nodiscard]] __device__ __forceinline__ float decoded_cell_weight(
    const std::uint8_t* const cell, const unsigned int physical_warps,
    const unsigned int branch, const unsigned int output,
    const unsigned int k) noexcept {
  const unsigned int local_warp = output / 32U;
  const unsigned int physical_warp =
      physical_warps == kGatePhysicalWarps ? branch * 4U + local_warp
                                           : local_warp;
  const unsigned int row = output % 32U;
  const unsigned int k16 = k / 16U;
  const unsigned int local_k = k % 16U;
  const std::size_t fragment_offset =
      (static_cast<std::size_t>(k16) * physical_warps + physical_warp) *
      kFragmentBytes;
  const std::uint8_t packed =
      cell[fragment_offset + row * 8U + local_k / 2U];
  const std::uint8_t e2m1 =
      (local_k & 1U) == 0U ? packed & 0x0fU : packed >> 4U;
  const std::uint8_t e4m3 =
      cell[fragment_offset + kFragmentWeightBytes + row];
  return decode_bf16(encode_bf16(decode_e2m1(e2m1) *
                                  decode_e4m3fn(e4m3)));
}

[[nodiscard]] __device__ __forceinline__ float reference_dot(
    const std::uint16_t* const input, const std::uint8_t* const cell,
    const unsigned int physical_warps, const unsigned int branch,
    const unsigned int token, const unsigned int output) noexcept {
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned int k = 0U; k < kInputFeatures; ++k) {
    accumulator = fmaf(
        decode_bf16(input[token * kInputFeatures + k]),
        decoded_cell_weight(cell, physical_warps, branch, output, k),
        accumulator);
  }
  return accumulator;
}

__global__ void gate_reference_kernel(
    const std::uint16_t* const input, const std::uint8_t* const cell,
    const float gate_scale, const float up_scale,
    std::uint16_t* const output) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= kOutputElements) {
    return;
  }
  const unsigned int token = linear / kOutputFeatures;
  const unsigned int column = linear % kOutputFeatures;
  const std::uint16_t gate_bits = encode_bf16(
      reference_dot(input, cell, kGatePhysicalWarps, 0U, token, column) *
      gate_scale);
  const std::uint16_t up_bits = encode_bf16(
      reference_dot(input, cell, kGatePhysicalWarps, 1U, token, column) *
      up_scale);
  const float gate = decode_bf16(gate_bits);
  const float up = decode_bf16(up_bits);
  output[linear] =
      encode_bf16(gate / (1.0F + expf(-gate)) * up);
}

__global__ void down_reference_kernel(
    const std::uint16_t* const input, const std::uint8_t* const cell,
    const float global_scale, const std::uint16_t* const residual,
    std::uint16_t* const output) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= kOutputElements) {
    return;
  }
  const unsigned int token = linear / kOutputFeatures;
  const unsigned int column = linear % kOutputFeatures;
  const std::uint16_t branch = encode_bf16(
      reference_dot(input, cell, kDownPhysicalWarps, 0U, token, column) *
      global_scale);
  output[linear] = encode_bf16(
      decode_bf16(branch) + decode_bf16(residual[linear]));
}

void initialize_input(std::vector<std::uint16_t>* const input) {
  std::fill(input->begin(), input->end(), 0U);
  const std::uint16_t one = 0x3f80U;
  for (unsigned int token = 0U; token < kTokens; ++token) {
    (*input)[token * kInputFeatures + token] = one;
  }
}

void initialize_residual(std::vector<std::uint16_t>* const residual) {
  constexpr std::array<std::uint16_t, 8U> kValues{{
      0x0000U, 0x3e80U, 0xbe80U, 0x3f00U,
      0xbf00U, 0x3f80U, 0xbf80U, 0x4000U,
  }};
  for (std::size_t index = 0U; index < residual->size(); ++index) {
    (*residual)[index] = kValues[index % kValues.size()];
  }
}

[[nodiscard]] bool is_nan_bf16(const std::uint16_t bits) noexcept {
  return (bits & 0x7f80U) == 0x7f80U && (bits & 0x007fU) != 0U;
}

void compare_output(TestContext& test, const std::vector<std::uint16_t>& got,
                    const std::vector<std::uint16_t>& expected,
                    const std::string& label) {
  std::size_t mismatches = 0U;
  std::size_t first = 0U;
  for (std::size_t index = 0U; index < kOutputElements; ++index) {
    const bool equal =
        got[index] == expected[index] ||
        (is_nan_bf16(got[index]) && is_nan_bf16(expected[index])) ||
        ((got[index] & 0x7fffU) == 0U &&
         (expected[index] & 0x7fffU) == 0U);
    if (!equal) {
      if (mismatches == 0U) {
        first = index;
      }
      ++mismatches;
    }
  }
  test.expect(mismatches == 0U,
              label + " mismatch count=" + std::to_string(mismatches) +
                  " first=" + std::to_string(first));
  bool canary_ok = true;
  for (std::size_t index = kOutputElements; index < got.size(); ++index) {
    canary_ok &= got[index] == kOutputCanary;
  }
  test.expect(canary_ok, label + " output canary changed");
}

void check_resources(TestContext& test,
                     const Sm87P40PackedProjectionRole role,
                     const std::size_t expected_shared,
                     const std::string& label) {
  Sm87P40PackedProjectionResources resources{};
  const int status =
      q3x::kernels::query_sm87_p40_packed_projection_resources_cuda(
          role, &resources);
  test.expect(status == static_cast<int>(cudaSuccess),
              label + " resource query: " +
                  cudaGetErrorString(static_cast<cudaError_t>(status)));
  if (status != static_cast<int>(cudaSuccess)) {
    return;
  }
  test.expect(resources.registers_per_thread <= 128,
              label + " exceeds 128 registers/thread");
  test.expect(resources.dynamic_shared_bytes == expected_shared,
              label + " dynamic shared-memory drift");
  test.expect(resources.local_bytes == 0U,
              label + " uses local memory");
  test.expect(resources.active_blocks_per_sm >= 2,
              label + " is below two CTA/SM");
  std::cout << label << " resources: regs/thread="
            << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active_cta/sm=" << resources.active_blocks_per_sm << '\n';
}

void run_gate(TestContext& test, const cudaStream_t stream) {
  constexpr float kGateScale = 0.75F;
  constexpr float kUpScale = 1.25F;
  std::vector<std::uint16_t> input(kTokens * kInputFeatures);
  std::vector<std::uint8_t> cell(kGateCellBytes, 0U);
  std::vector<std::uint16_t> initialized_output(
      kOutputElements + kCanaryElements, kOutputCanary);
  initialize_input(&input);
  fill_cell(&cell, kGatePhysicalWarps, true);

  DeviceBuffer<std::uint16_t> device_input;
  DeviceBuffer<std::uint8_t> device_cell;
  DeviceBuffer<std::uint16_t> candidate;
  DeviceBuffer<std::uint16_t> reference;
  if (!test.cuda_ok(device_input.allocate(input.size()),
                    "allocate Gate input") ||
      !test.cuda_ok(device_cell.allocate(cell.size()),
                    "allocate Gate cell") ||
      !test.cuda_ok(candidate.allocate(initialized_output.size()),
                    "allocate Gate candidate") ||
      !test.cuda_ok(reference.allocate(kOutputElements),
                    "allocate Gate reference")) {
    return;
  }
  test.cuda_ok(cudaMemcpyAsync(device_input.data(), input.data(),
                               device_input.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "copy Gate input");
  test.cuda_ok(cudaMemcpyAsync(device_cell.data(), cell.data(),
                               device_cell.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "copy Gate cell");
  test.cuda_ok(cudaMemcpyAsync(candidate.data(), initialized_output.data(),
                               candidate.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "seed Gate canary");

  const int invalid =
      q3x::kernels::launch_sm87_p40_packed_nvfp4_gate_up_cell_test_cuda(
          nullptr, device_cell.data(), kGateScale, kUpScale,
          candidate.data(), stream);
  test.expect(invalid == static_cast<int>(cudaErrorInvalidValue),
              "Gate cell launcher rejects null input");
  const int launch =
      q3x::kernels::launch_sm87_p40_packed_nvfp4_gate_up_cell_test_cuda(
          device_input.data(), device_cell.data(), kGateScale, kUpScale,
          candidate.data(), stream);
  test.expect(launch == static_cast<int>(cudaSuccess),
              "launch Gate cell: " +
                  std::string(cudaGetErrorString(
                      static_cast<cudaError_t>(launch))));
  gate_reference_kernel<<<32U, 256U, 0U, stream>>>(
      device_input.data(), device_cell.data(), kGateScale, kUpScale,
      reference.data());
  test.cuda_ok(cudaPeekAtLastError(), "launch Gate reference");

  std::vector<std::uint16_t> got(initialized_output.size());
  std::vector<std::uint16_t> expected(kOutputElements);
  test.cuda_ok(cudaMemcpyAsync(got.data(), candidate.data(), candidate.bytes(),
                               cudaMemcpyDeviceToHost, stream),
               "copy Gate candidate");
  test.cuda_ok(cudaMemcpyAsync(expected.data(), reference.data(),
                               reference.bytes(), cudaMemcpyDeviceToHost,
                               stream),
               "copy Gate reference");
  if (!test.cuda_ok(cudaStreamSynchronize(stream), "synchronize Gate")) {
    return;
  }
  compare_output(test, got, expected, "Gate+Up");
}

void run_down(TestContext& test, const cudaStream_t stream) {
  constexpr float kGlobalScale = 1.5F;
  std::vector<std::uint16_t> input(kTokens * kInputFeatures);
  std::vector<std::uint8_t> cell(kDownCellBytes, 0U);
  std::vector<std::uint16_t> initial(kOutputElements + kCanaryElements,
                                     kOutputCanary);
  std::vector<std::uint16_t> residual(kOutputElements);
  initialize_input(&input);
  initialize_residual(&residual);
  std::copy(residual.begin(), residual.end(), initial.begin());
  fill_cell(&cell, kDownPhysicalWarps, false);

  DeviceBuffer<std::uint16_t> device_input;
  DeviceBuffer<std::uint8_t> device_cell;
  DeviceBuffer<std::uint16_t> device_residual;
  DeviceBuffer<std::uint16_t> candidate;
  DeviceBuffer<std::uint16_t> reference;
  if (!test.cuda_ok(device_input.allocate(input.size()),
                    "allocate Down input") ||
      !test.cuda_ok(device_cell.allocate(cell.size()),
                    "allocate Down cell") ||
      !test.cuda_ok(device_residual.allocate(residual.size()),
                    "allocate Down residual") ||
      !test.cuda_ok(candidate.allocate(initial.size()),
                    "allocate Down candidate") ||
      !test.cuda_ok(reference.allocate(kOutputElements),
                    "allocate Down reference")) {
    return;
  }
  test.cuda_ok(cudaMemcpyAsync(device_input.data(), input.data(),
                               device_input.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "copy Down input");
  test.cuda_ok(cudaMemcpyAsync(device_cell.data(), cell.data(),
                               device_cell.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "copy Down cell");
  test.cuda_ok(cudaMemcpyAsync(device_residual.data(), residual.data(),
                               device_residual.bytes(),
                               cudaMemcpyHostToDevice, stream),
               "copy Down residual");
  test.cuda_ok(cudaMemcpyAsync(candidate.data(), initial.data(),
                               candidate.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "seed Down candidate/canary");

  const int invalid =
      q3x::kernels::launch_sm87_p40_packed_nvfp4_down_cell_test_cuda(
          nullptr, device_cell.data(), kGlobalScale, candidate.data(),
          stream);
  test.expect(invalid == static_cast<int>(cudaErrorInvalidValue),
              "Down cell launcher rejects null input");
  const int launch =
      q3x::kernels::launch_sm87_p40_packed_nvfp4_down_cell_test_cuda(
          device_input.data(), device_cell.data(), kGlobalScale,
          candidate.data(), stream);
  test.expect(launch == static_cast<int>(cudaSuccess),
              "launch Down cell: " +
                  std::string(cudaGetErrorString(
                      static_cast<cudaError_t>(launch))));
  down_reference_kernel<<<32U, 256U, 0U, stream>>>(
      device_input.data(), device_cell.data(), kGlobalScale,
      device_residual.data(), reference.data());
  test.cuda_ok(cudaPeekAtLastError(), "launch Down reference");

  std::vector<std::uint16_t> got(initial.size());
  std::vector<std::uint16_t> expected(kOutputElements);
  test.cuda_ok(cudaMemcpyAsync(got.data(), candidate.data(), candidate.bytes(),
                               cudaMemcpyDeviceToHost, stream),
               "copy Down candidate");
  test.cuda_ok(cudaMemcpyAsync(expected.data(), reference.data(),
                               reference.bytes(), cudaMemcpyDeviceToHost,
                               stream),
               "copy Down reference");
  if (!test.cuda_ok(cudaStreamSynchronize(stream), "synchronize Down")) {
    return;
  }
  compare_output(test, got, expected, "Down+residual");
}

[[nodiscard]] bool explicitly_enabled() noexcept {
  const char* const value = std::getenv(kRunEnvironment);
  return value != nullptr && std::strcmp(value, "1") == 0;
}

}  // namespace

int main() {
  if (!explicitly_enabled()) {
    std::cout << "SKIP: set " << kRunEnvironment
              << "=1 only after clean tegrastats/process/GPU-handle "
                 "preflight\n";
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
    std::cout << "SKIP: exact SM87/16-SM device required\n";
    return 77;
  }

  TestContext test;
  check_resources(test, Sm87P40PackedProjectionRole::kNvFp4GateUp,
                  69'632U, "Gate+Up");
  check_resources(test, Sm87P40PackedProjectionRole::kNvFp4Down,
                  51'200U, "Down");
  Stream stream;
  if (!test.cuda_ok(stream.create(), "create packed NVFP4 test stream")) {
    return 1;
  }
  run_gate(test, stream.get());
  run_down(test, stream.get());

  if (test.failures != 0) {
    std::cerr << test.failures << " packed-P40 NVFP4 CUDA checks failed\n";
    return 1;
  }
  std::cout << "packed-P40 NVFP4 CUDA correctness/canary/resource checks "
               "passed\n";
  return 0;
}
