#include "q3x/kernels/sm87_p40_packed_nvfp4_v2.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
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

namespace {

using q3x::kernels::Sm87P40PackedProjectionResources;
using q3x::kernels::Sm87P40PackedProjectionRole;

inline constexpr unsigned int kTokens = 128U;
inline constexpr unsigned int kInputFeatures = 128U;
inline constexpr unsigned int kGateOutputFeatures = 128U;
inline constexpr unsigned int kDownOutputFeatures = 256U;
inline constexpr unsigned int kFragmentWeightBytes = 256U;
inline constexpr unsigned int kFragmentBytes = 288U;
inline constexpr unsigned int kGatePhysicalWarps = 8U;
inline constexpr unsigned int kDownPhysicalWarps = 4U;
inline constexpr std::size_t kGateCellBytes =
    4U * kGatePhysicalWarps * kFragmentBytes;
inline constexpr std::size_t kDownCellBytes =
    4U * kDownPhysicalWarps * kFragmentBytes;
inline constexpr std::size_t kGatePayloadBytes = 2U * kGateCellBytes;
inline constexpr std::size_t kDownPayloadBytes = 4U * kDownCellBytes;
inline constexpr unsigned int kFullGateInputFeatures =
    q3x::kernels::kSm87P40PackedProjectionHidden;
inline constexpr unsigned int kFullDownInputFeatures =
    q3x::kernels::kSm87P40PackedProjectionIntermediate;
inline constexpr unsigned int kFullGateK64Tiles =
    kFullGateInputFeatures / 64U;
inline constexpr unsigned int kFullDownK64Tiles =
    kFullDownInputFeatures / 64U;
inline constexpr std::size_t kFullGatePayloadBytes =
    static_cast<std::size_t>(kFullGateK64Tiles) * kGateCellBytes;
inline constexpr std::size_t kFullDownPayloadBytes =
    2U * static_cast<std::size_t>(kFullDownK64Tiles) * kDownCellBytes;
inline constexpr std::size_t kCanaryElements = 256U;
inline constexpr std::uint16_t kCanary = 0xa55aU;
inline constexpr const char* kRunEnvironment =
    "Q3X_RUN_SM87_P40_PACKED_NVFP4_V2_NUMERICAL";

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
  return static_cast<std::uint8_t>(
      branch * 128U + output * 4U + k16);
}

void fill_cell(std::uint8_t* const cell, const unsigned int physical_warps,
               const bool gate_up, const unsigned int global_output_base,
               const unsigned int global_k_base) {
  for (unsigned int k16 = 0U; k16 < 4U; ++k16) {
    for (unsigned int physical_warp = 0U;
         physical_warp < physical_warps; ++physical_warp) {
      const unsigned int branch = gate_up ? physical_warp / 4U : 0U;
      const unsigned int local_warp =
          gate_up ? physical_warp % 4U : physical_warp;
      const std::size_t fragment_offset =
          (static_cast<std::size_t>(k16) * physical_warps +
           physical_warp) *
          kFragmentBytes;
      for (unsigned int row = 0U; row < 32U; ++row) {
        const unsigned int output =
            global_output_base + local_warp * 32U + row;
        for (unsigned int packed_k = 0U; packed_k < 8U; ++packed_k) {
          const unsigned int even_k =
              global_k_base + k16 * 16U + packed_k * 2U;
          const std::uint8_t low =
              weight_code(branch, output, even_k);
          const std::uint8_t high =
              weight_code(branch, output, even_k + 1U);
          cell[fragment_offset + row * 8U + packed_k] =
              static_cast<std::uint8_t>(low | (high << 4U));
        }
        cell[fragment_offset + kFragmentWeightBytes + row] =
            scale_code(branch, output,
                       (global_k_base / 16U) + k16);
      }
    }
  }
}

void fill_gate_payload(std::vector<std::uint8_t>* const payload) {
  for (unsigned int k64 = 0U; k64 < 2U; ++k64) {
    fill_cell(payload->data() + k64 * kGateCellBytes,
              kGatePhysicalWarps, true, 0U, k64 * 64U);
  }
}

void fill_down_payload(std::vector<std::uint8_t>* const payload) {
  for (unsigned int n_half = 0U; n_half < 2U; ++n_half) {
    for (unsigned int k64 = 0U; k64 < 2U; ++k64) {
      fill_cell(payload->data() +
                    (n_half * 2U + k64) * kDownCellBytes,
                kDownPhysicalWarps, false, n_half * 128U,
                k64 * 64U);
    }
  }
}

[[nodiscard]] constexpr std::uint8_t exact_fixture_weight_code(
    const unsigned int branch, const unsigned int output,
    const unsigned int k) noexcept {
  const unsigned int within_k64 = k % 64U;
  if (within_k64 > 1U) {
    return 0U;
  }
  const unsigned int k128 = k / 128U;
  const unsigned int k64_half = (k / 64U) % 2U;
  // E2M1 magnitudes 1..3 are exactly 0.5, 1.0, and 1.5.  Combined with the
  // exact power-of-two activations below, every partial sum is exactly
  // representable in FP32 regardless of MMA grouping.
  return static_cast<std::uint8_t>(
      1U + ((k128 + 2U * k64_half + within_k64 + output % 5U + branch) %
            3U));
}

void fill_exact_full_k_cell(std::uint8_t* const cell,
                            const unsigned int physical_warps,
                            const bool gate_up,
                            const unsigned int global_output_base,
                            const unsigned int global_k_base) {
  constexpr std::uint8_t kExactOneE4M3 = 0x38U;
  for (unsigned int k16 = 0U; k16 < 4U; ++k16) {
    for (unsigned int physical_warp = 0U;
         physical_warp < physical_warps; ++physical_warp) {
      const unsigned int branch = gate_up ? physical_warp / 4U : 0U;
      const unsigned int local_warp =
          gate_up ? physical_warp % 4U : physical_warp;
      const std::size_t fragment_offset =
          (static_cast<std::size_t>(k16) * physical_warps +
           physical_warp) *
          kFragmentBytes;
      for (unsigned int row = 0U; row < 32U; ++row) {
        const unsigned int output =
            global_output_base + local_warp * 32U + row;
        for (unsigned int packed_k = 0U; packed_k < 8U; ++packed_k) {
          const unsigned int even_k =
              global_k_base + k16 * 16U + packed_k * 2U;
          const std::uint8_t low =
              exact_fixture_weight_code(branch, output, even_k);
          const std::uint8_t high =
              exact_fixture_weight_code(branch, output, even_k + 1U);
          cell[fragment_offset + row * 8U + packed_k] =
              static_cast<std::uint8_t>(low | (high << 4U));
        }
        cell[fragment_offset + kFragmentWeightBytes + row] =
            kExactOneE4M3;
      }
    }
  }
}

void fill_full_gate_payload(std::vector<std::uint8_t>* const payload) {
  for (unsigned int k64 = 0U; k64 < kFullGateK64Tiles; ++k64) {
    fill_exact_full_k_cell(payload->data() +
                               static_cast<std::size_t>(k64) *
                                   kGateCellBytes,
                           kGatePhysicalWarps, true, 0U, k64 * 64U);
  }
}

void fill_full_down_payload(std::vector<std::uint8_t>* const payload) {
  for (unsigned int n_half = 0U; n_half < 2U; ++n_half) {
    for (unsigned int k64 = 0U; k64 < kFullDownK64Tiles; ++k64) {
      fill_exact_full_k_cell(
          payload->data() +
              (static_cast<std::size_t>(n_half) * kFullDownK64Tiles + k64) *
                  kDownCellBytes,
          kDownPhysicalWarps, false, n_half * 128U, k64 * 64U);
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

[[nodiscard]] __device__ __forceinline__ float decoded_weight(
    const std::uint8_t* const payload, const bool gate_up,
    const unsigned int branch, const unsigned int output,
    const unsigned int k) noexcept {
  const unsigned int k64 = k / 64U;
  const unsigned int k16 = (k % 64U) / 16U;
  const unsigned int local_k = k % 16U;
  const unsigned int n_half = gate_up ? 0U : output / 128U;
  const unsigned int local_output = gate_up ? output : output % 128U;
  const unsigned int local_warp = local_output / 32U;
  const unsigned int physical_warps =
      gate_up ? kGatePhysicalWarps : kDownPhysicalWarps;
  const unsigned int physical_warp =
      gate_up ? branch * 4U + local_warp : local_warp;
  const std::size_t cell_bytes = gate_up ? kGateCellBytes
                                         : kDownCellBytes;
  const std::size_t cell_index =
      gate_up ? k64 : n_half * 2U + k64;
  const auto* const cell = payload + cell_index * cell_bytes;
  const std::size_t fragment_offset =
      (static_cast<std::size_t>(k16) * physical_warps +
       physical_warp) *
      kFragmentBytes;
  const unsigned int row = local_output % 32U;
  const std::uint8_t packed =
      cell[fragment_offset + row * 8U + local_k / 2U];
  const std::uint8_t e2m1 =
      (local_k & 1U) == 0U ? packed & 0x0fU : packed >> 4U;
  const std::uint8_t e4m3 =
      cell[fragment_offset + kFragmentWeightBytes + row];
  return decode_bf16(
      encode_bf16(decode_e2m1(e2m1) * decode_e4m3fn(e4m3)));
}

[[nodiscard]] __device__ __forceinline__ float reference_dot(
    const std::uint16_t* const input,
    const std::uint8_t* const payload, const bool gate_up,
    const unsigned int branch, const unsigned int token,
    const unsigned int output) noexcept {
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned int k = 0U; k < kInputFeatures; ++k) {
    accumulator = fmaf(
        decode_bf16(input[token * kInputFeatures + k]),
        decoded_weight(payload, gate_up, branch, output, k),
        accumulator);
  }
  return accumulator;
}

__global__ void gate_reference_kernel(
    const std::uint16_t* input, const std::uint8_t* payload,
    const float gate_scale, const float up_scale,
    std::uint16_t* output) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned int kElements = kTokens * kGateOutputFeatures;
  if (linear >= kElements) {
    return;
  }
  const unsigned int token = linear / kGateOutputFeatures;
  const unsigned int column = linear % kGateOutputFeatures;
  const std::uint16_t gate_bits = encode_bf16(
      reference_dot(input, payload, true, 0U, token, column) *
      gate_scale);
  const std::uint16_t up_bits = encode_bf16(
      reference_dot(input, payload, true, 1U, token, column) *
      up_scale);
  const float gate = decode_bf16(gate_bits);
  const float up = decode_bf16(up_bits);
  output[linear] =
      encode_bf16(gate / (1.0F + expf(-gate)) * up);
}

__global__ void down_reference_kernel(
    const std::uint16_t* input, const std::uint8_t* payload,
    const float global_scale, const std::uint16_t* residual,
    std::uint16_t* output) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned int kElements = kTokens * kDownOutputFeatures;
  if (linear >= kElements) {
    return;
  }
  const unsigned int token = linear / kDownOutputFeatures;
  const unsigned int column = linear % kDownOutputFeatures;
  const std::uint16_t branch = encode_bf16(
      reference_dot(input, payload, false, 0U, token, column) *
      global_scale);
  output[linear] = encode_bf16(
      decode_bf16(branch) + decode_bf16(residual[linear]));
}

template <bool kGateUp, unsigned int kK64Tiles>
[[nodiscard]] __device__ __forceinline__ float decoded_full_k_weight(
    const std::uint8_t* const payload, const unsigned int branch,
    const unsigned int output, const unsigned int k) noexcept {
  const unsigned int k64 = k / 64U;
  const unsigned int k16 = (k % 64U) / 16U;
  const unsigned int local_k = k % 16U;
  const unsigned int n_half = kGateUp ? 0U : output / 128U;
  const unsigned int local_output = kGateUp ? output : output % 128U;
  const unsigned int local_warp = local_output / 32U;
  constexpr unsigned int kPhysicalWarps =
      kGateUp ? kGatePhysicalWarps : kDownPhysicalWarps;
  const unsigned int physical_warp =
      kGateUp ? branch * 4U + local_warp : local_warp;
  constexpr std::size_t kCellBytes =
      kGateUp ? kGateCellBytes : kDownCellBytes;
  const std::size_t cell_index =
      kGateUp ? k64
              : static_cast<std::size_t>(n_half) * kK64Tiles + k64;
  const auto* const cell = payload + cell_index * kCellBytes;
  const std::size_t fragment_offset =
      (static_cast<std::size_t>(k16) * kPhysicalWarps + physical_warp) *
      kFragmentBytes;
  const unsigned int row = local_output % 32U;
  const std::uint8_t packed =
      cell[fragment_offset + row * 8U + local_k / 2U];
  const std::uint8_t e2m1 =
      (local_k & 1U) == 0U ? packed & 0x0fU : packed >> 4U;
  const std::uint8_t e4m3 =
      cell[fragment_offset + kFragmentWeightBytes + row];
  return decode_bf16(
      encode_bf16(decode_e2m1(e2m1) * decode_e4m3fn(e4m3)));
}

template <bool kGateUp, unsigned int kInputFeatureCount,
          unsigned int kK64Tiles>
[[nodiscard]] __device__ __forceinline__ float full_k_reference_dot(
    const std::uint16_t* const input,
    const std::uint8_t* const payload, const unsigned int branch,
    const unsigned int token, const unsigned int output) noexcept {
  float accumulator = 0.0F;
#pragma unroll 1
  for (unsigned int k = 0U; k < kInputFeatureCount; ++k) {
    accumulator = fmaf(
        decode_bf16(input[token * kInputFeatureCount + k]),
        decoded_full_k_weight<kGateUp, kK64Tiles>(payload, branch, output, k),
        accumulator);
  }
  return accumulator;
}

__global__ void gate_full_k_reference_kernel(
    const std::uint16_t* input, const std::uint8_t* payload,
    const unsigned int token_limit, const float gate_scale,
    const float up_scale, std::uint16_t* output) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned int kElements = kTokens * kGateOutputFeatures;
  if (linear >= kElements) {
    return;
  }
  const unsigned int token = linear / kGateOutputFeatures;
  if (token >= token_limit) {
    return;
  }
  const unsigned int column = linear % kGateOutputFeatures;
  const std::uint16_t gate_bits = encode_bf16(
      full_k_reference_dot<true, kFullGateInputFeatures,
                           kFullGateK64Tiles>(
          input, payload, 0U, token, column) *
      gate_scale);
  const std::uint16_t up_bits = encode_bf16(
      full_k_reference_dot<true, kFullGateInputFeatures,
                           kFullGateK64Tiles>(
          input, payload, 1U, token, column) *
      up_scale);
  const float gate = decode_bf16(gate_bits);
  const float up = decode_bf16(up_bits);
  output[linear] = encode_bf16(gate / (1.0F + expf(-gate)) * up);
}

__global__ void down_full_k_reference_kernel(
    const std::uint16_t* input, const std::uint8_t* payload,
    const unsigned int token_limit, const float global_scale,
    const std::uint16_t* residual, std::uint16_t* output) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned int kElements = kTokens * kDownOutputFeatures;
  if (linear >= kElements) {
    return;
  }
  const unsigned int token = linear / kDownOutputFeatures;
  if (token >= token_limit) {
    return;
  }
  const unsigned int column = linear % kDownOutputFeatures;
  const std::uint16_t branch = encode_bf16(
      full_k_reference_dot<false, kFullDownInputFeatures,
                           kFullDownK64Tiles>(
          input, payload, 0U, token, column) *
      global_scale);
  output[linear] = encode_bf16(
      decode_bf16(branch) + decode_bf16(residual[linear]));
}

void initialize_input(std::vector<std::uint16_t>* const input) {
  std::fill(input->begin(), input->end(), 0U);
  for (unsigned int token = 0U; token < kTokens; ++token) {
    (*input)[token * kInputFeatures + token] = 0x3f80U;
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

void initialize_full_k_input(std::vector<std::uint16_t>* const input,
                             const unsigned int input_features) {
  constexpr std::array<std::uint16_t, 4U> kExactValues{{
      0x3d00U,  // 1/32
      0x3d80U,  // 1/16
      0x3e00U,  // 1/8
      0x3e80U,  // 1/4
  }};
  std::fill(input->begin(), input->end(), 0U);
  for (unsigned int token = 0U; token < kTokens; ++token) {
    const std::uint16_t value = kExactValues[token % kExactValues.size()];
    for (unsigned int first_k = 0U; first_k < input_features;
         first_k += 128U) {
      for (const unsigned int offset :
           std::array<unsigned int, 4U>{{0U, 1U, 64U, 65U}}) {
        (*input)[static_cast<std::size_t>(token) * input_features + first_k +
                 offset] = value;
      }
    }
  }
}

void compare_output(TestContext& test,
                    const std::vector<std::uint16_t>& got,
                    const std::vector<std::uint16_t>& expected,
                    const std::size_t elements,
                    const std::string& label) {
  std::size_t mismatches = 0U;
  std::size_t first = 0U;
  for (std::size_t index = 0U; index < elements; ++index) {
    if (got[index] != expected[index]) {
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
  for (std::size_t index = elements; index < got.size(); ++index) {
    canary_ok &= got[index] == kCanary;
  }
  test.expect(canary_ok, label + " output canary changed");
}

void check_resources(TestContext& test,
                     const Sm87P40PackedProjectionRole role,
                     const std::string& label) {
  Sm87P40PackedProjectionResources resources{};
  const int status =
      q3x::kernels::query_sm87_p40_packed_nvfp4_v2_resources_cuda(
          role, &resources);
  test.expect(status == static_cast<int>(cudaSuccess),
              label + " resource query: " +
                  cudaGetErrorString(static_cast<cudaError_t>(status)));
  if (status != static_cast<int>(cudaSuccess)) {
    return;
  }
  test.expect(resources.registers_per_thread <= 224,
              label + " exceeds 224 registers/thread");
  test.expect(resources.dynamic_shared_bytes ==
                  q3x::kernels::kSm87P40PackedNvFp4V2DynamicSharedBytes,
              label + " dynamic shared-memory drift");
  test.expect(resources.local_bytes == 0U,
              label + " uses local memory");
  test.expect(resources.active_blocks_per_sm >= 1,
              label + " has no resident CTA");
  std::cout << label << " resources: regs/thread="
            << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active_cta/sm=" << resources.active_blocks_per_sm << '\n';
}

template <bool kGate>
void run_tile(TestContext& test, const cudaStream_t stream) {
  constexpr unsigned int kOutputFeatures =
      kGate ? kGateOutputFeatures : kDownOutputFeatures;
  constexpr std::size_t kElements = kTokens * kOutputFeatures;
  constexpr std::size_t kPayloadBytes =
      kGate ? kGatePayloadBytes : kDownPayloadBytes;
  constexpr float kScale0 = kGate ? 0.75F : 1.5F;
  constexpr float kScale1 = 1.25F;
  const std::string label = kGate ? "Gate+Up v2" : "Down+residual v2";

  std::vector<std::uint16_t> input(kTokens * kInputFeatures);
  std::vector<std::uint8_t> payload(kPayloadBytes, 0U);
  std::vector<std::uint16_t> initial(kElements + kCanaryElements,
                                     kCanary);
  std::vector<std::uint16_t> residual(kElements);
  initialize_input(&input);
  if constexpr (kGate) {
    fill_gate_payload(&payload);
  } else {
    fill_down_payload(&payload);
    initialize_residual(&residual);
    std::copy(residual.begin(), residual.end(), initial.begin());
  }

  DeviceBuffer<std::uint16_t> device_input;
  DeviceBuffer<std::uint8_t> device_payload;
  DeviceBuffer<std::uint16_t> candidate;
  DeviceBuffer<std::uint16_t> reference;
  DeviceBuffer<std::uint16_t> device_residual;
  if (!test.cuda_ok(device_input.allocate(input.size()),
                    "allocate " + label + " input") ||
      !test.cuda_ok(device_payload.allocate(payload.size()),
                    "allocate " + label + " payload") ||
      !test.cuda_ok(candidate.allocate(initial.size()),
                    "allocate " + label + " candidate") ||
      !test.cuda_ok(reference.allocate(kElements),
                    "allocate " + label + " reference") ||
      (!kGate && !test.cuda_ok(device_residual.allocate(residual.size()),
                               "allocate Down residual"))) {
    return;
  }
  test.cuda_ok(cudaMemcpyAsync(device_input.data(), input.data(),
                               device_input.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "copy " + label + " input");
  test.cuda_ok(cudaMemcpyAsync(device_payload.data(), payload.data(),
                               device_payload.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "copy " + label + " payload");
  test.cuda_ok(cudaMemcpyAsync(candidate.data(), initial.data(),
                               candidate.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "seed " + label + " candidate/canary");
  if constexpr (!kGate) {
    test.cuda_ok(cudaMemcpyAsync(device_residual.data(), residual.data(),
                                 device_residual.bytes(),
                                 cudaMemcpyHostToDevice, stream),
                 "copy Down residual");
  }

  int invalid = 0;
  int launch = 0;
  if constexpr (kGate) {
    invalid = q3x::kernels::
        launch_sm87_p40_packed_nvfp4_v2_gate_up_tile_test_cuda(
            nullptr, device_payload.data(), kScale0, kScale1,
            candidate.data(), stream);
    launch = q3x::kernels::
        launch_sm87_p40_packed_nvfp4_v2_gate_up_tile_test_cuda(
            device_input.data(), device_payload.data(), kScale0, kScale1,
            candidate.data(), stream);
    gate_reference_kernel<<<64U, 256U, 0U, stream>>>(
        device_input.data(), device_payload.data(), kScale0, kScale1,
        reference.data());
  } else {
    invalid = q3x::kernels::
        launch_sm87_p40_packed_nvfp4_v2_down_tile_test_cuda(
            nullptr, device_payload.data(), kScale0, candidate.data(),
            stream);
    launch = q3x::kernels::
        launch_sm87_p40_packed_nvfp4_v2_down_tile_test_cuda(
            device_input.data(), device_payload.data(), kScale0,
            candidate.data(), stream);
    down_reference_kernel<<<128U, 256U, 0U, stream>>>(
        device_input.data(), device_payload.data(), kScale0,
        device_residual.data(), reference.data());
  }
  test.expect(invalid == static_cast<int>(cudaErrorInvalidValue),
              label + " rejects null input");
  test.expect(launch == static_cast<int>(cudaSuccess),
              label + " launch: " +
                  cudaGetErrorString(static_cast<cudaError_t>(launch)));
  test.cuda_ok(cudaPeekAtLastError(), "launch " + label + " reference");

  std::vector<std::uint16_t> got(initial.size());
  std::vector<std::uint16_t> expected(kElements);
  test.cuda_ok(cudaMemcpyAsync(got.data(), candidate.data(),
                               candidate.bytes(), cudaMemcpyDeviceToHost,
                               stream),
               "copy " + label + " candidate");
  test.cuda_ok(cudaMemcpyAsync(expected.data(), reference.data(),
                               reference.bytes(), cudaMemcpyDeviceToHost,
                               stream),
               "copy " + label + " reference");
  if (!test.cuda_ok(cudaStreamSynchronize(stream),
                    "synchronize " + label)) {
    return;
  }
  compare_output(test, got, expected, kElements, label);
}

template <bool kGate>
void run_full_k(TestContext& test, const cudaStream_t stream) {
  constexpr unsigned int kInputFeatureCount =
      kGate ? kFullGateInputFeatures : kFullDownInputFeatures;
  constexpr unsigned int kOutputFeatureCount =
      kGate ? kGateOutputFeatures : kDownOutputFeatures;
  constexpr std::size_t kElements = kTokens * kOutputFeatureCount;
  constexpr std::size_t kPayloadBytes =
      kGate ? kFullGatePayloadBytes : kFullDownPayloadBytes;
  constexpr float kScale0 = 1.0F;
  constexpr float kScale1 = 1.0F;
  constexpr std::size_t kTailTokens = 64U;
  const std::string label =
      kGate ? "Gate+Up v2 full-K" : "Down+residual v2 full-K";

  std::vector<std::uint16_t> input(
      static_cast<std::size_t>(kTokens) * kInputFeatureCount);
  std::vector<std::uint8_t> payload(kPayloadBytes, 0U);
  std::vector<std::uint16_t> initial(kElements + kCanaryElements,
                                     kCanary);
  std::vector<std::uint16_t> residual(kElements);
  initialize_full_k_input(&input, kInputFeatureCount);
  if constexpr (kGate) {
    fill_full_gate_payload(&payload);
  } else {
    fill_full_down_payload(&payload);
    initialize_residual(&residual);
    std::copy(residual.begin(), residual.end(), initial.begin());
  }

  DeviceBuffer<std::uint16_t> device_input;
  DeviceBuffer<std::uint8_t> device_payload;
  DeviceBuffer<std::uint16_t> candidate;
  DeviceBuffer<std::uint16_t> reference;
  DeviceBuffer<std::uint16_t> device_residual;
  if (!test.cuda_ok(device_input.allocate(input.size()),
                    "allocate " + label + " input") ||
      !test.cuda_ok(device_payload.allocate(payload.size()),
                    "allocate " + label + " payload") ||
      !test.cuda_ok(candidate.allocate(initial.size()),
                    "allocate " + label + " candidate") ||
      !test.cuda_ok(reference.allocate(initial.size()),
                    "allocate " + label + " reference") ||
      (!kGate && !test.cuda_ok(device_residual.allocate(residual.size()),
                               "allocate full-K Down residual"))) {
    return;
  }
  test.cuda_ok(cudaMemcpyAsync(device_input.data(), input.data(),
                               device_input.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "copy " + label + " input");
  test.cuda_ok(cudaMemcpyAsync(device_payload.data(), payload.data(),
                               device_payload.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "copy " + label + " payload");
  test.cuda_ok(cudaMemcpyAsync(candidate.data(), initial.data(),
                               candidate.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "seed " + label + " candidate");
  test.cuda_ok(cudaMemcpyAsync(reference.data(), initial.data(),
                               reference.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "seed " + label + " reference");
  if constexpr (!kGate) {
    test.cuda_ok(cudaMemcpyAsync(device_residual.data(), residual.data(),
                                 device_residual.bytes(),
                                 cudaMemcpyHostToDevice, stream),
                 "copy full-K Down residual");
  }

  int invalid_limit = 0;
  int launch = 0;
  if constexpr (kGate) {
    invalid_limit = q3x::kernels::
        launch_sm87_p40_packed_nvfp4_v2_gate_up_full_k_test_cuda(
            device_input.data(), device_payload.data(), 127U, kScale0,
            kScale1, candidate.data(), stream);
    launch = q3x::kernels::
        launch_sm87_p40_packed_nvfp4_v2_gate_up_full_k_test_cuda(
            device_input.data(), device_payload.data(), kTokens, kScale0,
            kScale1, candidate.data(), stream);
    gate_full_k_reference_kernel<<<64U, 256U, 0U, stream>>>(
        device_input.data(), device_payload.data(), kTokens, kScale0, kScale1,
        reference.data());
  } else {
    invalid_limit = q3x::kernels::
        launch_sm87_p40_packed_nvfp4_v2_down_full_k_test_cuda(
            device_input.data(), device_payload.data(), 127U, kScale0,
            candidate.data(), stream);
    launch = q3x::kernels::
        launch_sm87_p40_packed_nvfp4_v2_down_full_k_test_cuda(
            device_input.data(), device_payload.data(), kTokens, kScale0,
            candidate.data(), stream);
    down_full_k_reference_kernel<<<128U, 256U, 0U, stream>>>(
        device_input.data(), device_payload.data(), kTokens, kScale0,
        device_residual.data(), reference.data());
  }
  test.expect(invalid_limit == static_cast<int>(cudaErrorInvalidValue),
              label + " rejects a non-production token limit");
  test.expect(launch == static_cast<int>(cudaSuccess),
              label + " launch: " +
                  cudaGetErrorString(static_cast<cudaError_t>(launch)));
  test.cuda_ok(cudaPeekAtLastError(), "launch " + label + " reference");

  std::vector<std::uint16_t> got(initial.size());
  std::vector<std::uint16_t> expected(initial.size());
  test.cuda_ok(cudaMemcpyAsync(got.data(), candidate.data(), candidate.bytes(),
                               cudaMemcpyDeviceToHost, stream),
               "copy " + label + " candidate");
  test.cuda_ok(cudaMemcpyAsync(expected.data(), reference.data(),
                               reference.bytes(), cudaMemcpyDeviceToHost,
                               stream),
               "copy " + label + " reference");
  if (!test.cuda_ok(cudaStreamSynchronize(stream),
                    "synchronize " + label)) {
    return;
  }
  compare_output(test, got, expected, kElements, label);

  // Reuse the full-tile exact oracle for the valid half of the P40000 tail.
  // The invalid half must remain byte-for-byte at its original contents.
  std::vector<std::uint16_t> tail_expected = initial;
  std::copy(expected.begin(),
            expected.begin() + kTailTokens * kOutputFeatureCount,
            tail_expected.begin());
  test.cuda_ok(cudaMemcpyAsync(candidate.data(), initial.data(),
                               candidate.bytes(), cudaMemcpyHostToDevice,
                               stream),
               "reset " + label + " tail candidate");
  int tail_launch = 0;
  if constexpr (kGate) {
    tail_launch = q3x::kernels::
        launch_sm87_p40_packed_nvfp4_v2_gate_up_full_k_test_cuda(
            device_input.data(), device_payload.data(), kTailTokens, kScale0,
            kScale1, candidate.data(), stream);
  } else {
    tail_launch = q3x::kernels::
        launch_sm87_p40_packed_nvfp4_v2_down_full_k_test_cuda(
            device_input.data(), device_payload.data(), kTailTokens, kScale0,
            candidate.data(), stream);
  }
  test.expect(tail_launch == static_cast<int>(cudaSuccess),
              label + " tail launch: " +
                  cudaGetErrorString(static_cast<cudaError_t>(tail_launch)));
  test.cuda_ok(cudaMemcpyAsync(got.data(), candidate.data(), candidate.bytes(),
                               cudaMemcpyDeviceToHost, stream),
               "copy " + label + " tail candidate");
  if (!test.cuda_ok(cudaStreamSynchronize(stream),
                    "synchronize " + label + " tail")) {
    return;
  }
  compare_output(test, got, tail_expected, kElements, label + " M64 tail");
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
                  "Gate+Up v2");
  check_resources(test, Sm87P40PackedProjectionRole::kNvFp4Down,
                  "Down v2");
  Stream stream;
  if (!test.cuda_ok(stream.create(), "create packed NVFP4 v2 test stream")) {
    return 1;
  }
  run_tile<true>(test, stream.get());
  run_tile<false>(test, stream.get());
  run_full_k<true>(test, stream.get());
  run_full_k<false>(test, stream.get());

  if (test.failures != 0) {
    std::cerr << test.failures
              << " packed-P40 NVFP4 v2 CUDA checks failed\n";
    return 1;
  }
  std::cout << "packed-P40 NVFP4 v2 M128/K128 correctness/canary/resource "
               "checks passed\n";
  return 0;
}
