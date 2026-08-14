#include "../src/kernels/sm87/sm87_macrofeed_v3_nvfp4_decode.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

struct DecodeInput final {
  std::uint16_t packed = 0U;
  std::uint8_t scale = 0U;
  std::uint8_t padding = 0U;
};

struct DecodeOutput final {
  std::uint32_t first = 0U;
  std::uint32_t second = 0U;
};

__global__ void decode_kernel(const DecodeInput* const input,
                              DecodeOutput* const output,
                              const std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const uint2 decoded =
      q3x::kernels::macrofeed_v3_nvfp4_detail::decode_nvfp4x4_to_bf16x4(
          input[index].packed, input[index].scale);
  output[index] = {decoded.x, decoded.y};
}

[[nodiscard]] std::uint16_t encode_bf16_rne(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_e2m1(const std::uint8_t code) noexcept {
  constexpr std::array<float, 8U> kMagnitude{
      {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F}};
  const float value = kMagnitude[code & 0x07U];
  return (code & 0x08U) != 0U ? -value : value;
}

[[nodiscard]] float decode_e4m3fn_finite_terminal(
    const std::uint8_t code) noexcept {
  const unsigned int magnitude = code & 0x7fU;
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  const float value =
      exponent == 0U
          ? std::ldexp(static_cast<float>(mantissa), -9)
          : std::ldexp(static_cast<float>(8U + mantissa),
                       static_cast<int>(exponent) - 10);
  return (code & 0x80U) != 0U ? -value : value;
}

[[nodiscard]] std::uint16_t expected_component(
    const std::uint16_t packed, const std::uint8_t scale,
    const unsigned int persisted_component) noexcept {
  const auto code = static_cast<std::uint8_t>(
      (packed >> (4U * persisted_component)) & 0x0fU);
  return encode_bf16_rne(decode_e2m1(code) *
                         decode_e4m3fn_finite_terminal(scale));
}

[[nodiscard]] DecodeOutput expected_output(const DecodeInput input) noexcept {
  // Canonical payload [K0,K8,K1,K9] must become PTX MMA pairs
  // x=[K0,K1] and y=[K8,K9].
  const std::uint16_t k0 = expected_component(input.packed, input.scale, 0U);
  const std::uint16_t k8 = expected_component(input.packed, input.scale, 1U);
  const std::uint16_t k1 = expected_component(input.packed, input.scale, 2U);
  const std::uint16_t k9 = expected_component(input.packed, input.scale, 3U);
  return {static_cast<std::uint32_t>(k0) |
              (static_cast<std::uint32_t>(k1) << 16U),
          static_cast<std::uint32_t>(k8) |
              (static_cast<std::uint32_t>(k9) << 16U)};
}

bool cuda_ok(const cudaError_t status, const char* const operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA device unavailable\n";
    return 77;
  }
  int device = -1;
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice") ||
      !cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: test requires the SM87 target\n";
    return 77;
  }

  std::vector<DecodeInput> input;
  input.reserve(65'536U + 16U * 256U);
  // Exhaust every four-nibble permutation at scale 1.0.  This is the direct
  // guard against a PRMT order bug hidden by uniform-nibble GEMM fixtures.
  for (std::uint32_t packed = 0U; packed <= 0xffffU; ++packed) {
    input.push_back(
        {static_cast<std::uint16_t>(packed), 0x38U, 0U});
  }
  constexpr std::array<std::uint16_t, 16U> kAsymmetricPatterns{{
      0x3210U, 0x7654U, 0xba98U, 0xfedcU,
      0xf3a1U, 0x18e6U, 0x90c7U, 0x5b2dU,
      0x8421U, 0x7d30U, 0xc69aU, 0xe105U,
      0x2468U, 0x1357U, 0x8aceU, 0x9bdfU,
  }};
  // Sweep every finite-terminal E4M3 code with non-uniform components.
  for (std::uint32_t scale = 0U; scale <= 0xffU; ++scale) {
    for (const std::uint16_t packed : kAsymmetricPatterns) {
      input.push_back({packed, static_cast<std::uint8_t>(scale), 0U});
    }
  }

  DecodeInput* device_input = nullptr;
  DecodeOutput* device_output = nullptr;
  bool ok =
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_input),
                         input.size() * sizeof(DecodeInput)),
              "cudaMalloc input") &&
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_output),
                         input.size() * sizeof(DecodeOutput)),
              "cudaMalloc output");
  if (ok) {
    ok = cuda_ok(cudaMemcpy(device_input, input.data(),
                            input.size() * sizeof(DecodeInput),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy input H2D");
  }
  if (ok) {
    constexpr unsigned int kThreads = 256U;
    const unsigned int blocks = static_cast<unsigned int>(
        (input.size() + kThreads - 1U) / kThreads);
    decode_kernel<<<blocks, kThreads>>>(device_input, device_output,
                                        input.size());
    ok = cuda_ok(cudaPeekAtLastError(), "decode launch") &&
         cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }

  std::vector<DecodeOutput> output(input.size());
  if (ok) {
    ok = cuda_ok(cudaMemcpy(output.data(), device_output,
                            output.size() * sizeof(DecodeOutput),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy output D2H");
  }
  if (ok) {
    for (std::size_t index = 0U; index < input.size(); ++index) {
      const DecodeOutput expected = expected_output(input[index]);
      if (output[index].first != expected.first ||
          output[index].second != expected.second) {
        std::cerr << "mismatch index=" << index << " packed=0x" << std::hex
                  << input[index].packed << " scale=0x"
                  << static_cast<unsigned int>(input[index].scale)
                  << " expected={0x" << expected.first << ",0x"
                  << expected.second << "} actual={0x"
                  << output[index].first << ",0x" << output[index].second
                  << "}" << std::dec << '\n';
        ok = false;
        break;
      }
    }
  }

  if (device_output != nullptr) {
    (void)cudaFree(device_output);
  }
  if (device_input != nullptr) {
    (void)cudaFree(device_input);
  }
  if (ok) {
    std::cout << "PASS: exhaustive asymmetric NVFP4 PRMT order and E4M3 "
                 "scale sweep\n";
    return 0;
  }
  return 1;
}
