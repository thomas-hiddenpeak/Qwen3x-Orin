#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

inline constexpr std::size_t kM = 65U;
inline constexpr std::size_t kN = 64U;
inline constexpr std::size_t kK = 128U;
inline constexpr std::size_t kGroups = kK / 64U;
inline constexpr std::size_t kPackedRowBytes = kK / 2U;

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

  [[nodiscard]] bool allocate(const std::size_t elements) noexcept {
    return cudaMalloc(reinterpret_cast<void**>(&data_),
                      elements * sizeof(T)) == cudaSuccess;
  }
  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const std::string& operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorName(status) << " ("
            << cudaGetErrorString(status) << ")\n";
  return false;
}

[[nodiscard]] bool launch_ok(const int status,
                             const std::string& operation) {
  return cuda_ok(static_cast<cudaError_t>(status), operation);
}

[[nodiscard]] int signed_code(const std::uint8_t packed,
                              const std::size_t k) noexcept {
  return kernels::sm87_a4w4_unpack_signed(packed, k);
}

[[nodiscard]] bool device_is_target() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      count == 0) {
    (void)cudaGetLastError();
    std::cout << "SKIP: CUDA device unavailable\n";
    return false;
  }
  if (!cuda_ok(status, "cudaGetDeviceCount")) {
    return false;
  }
  int device = 0;
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice")) {
    return false;
  }
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return false;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: requires the 16-SM SM87 target\n";
    return false;
  }
  return true;
}

[[nodiscard]] bool run_projection() {
  std::vector<std::uint16_t> input(kM * kK);
  for (std::size_t m = 0U; m < kM; ++m) {
    for (std::size_t k = 0U; k < kK; ++k) {
      const int centered =
          static_cast<int>((17U * m + 11U * k + 3U) % 37U) - 18;
      input[m * kK + k] = encode_bf16(static_cast<float>(centered) / 8.0F);
    }
  }

  std::vector<std::uint8_t> packed_b(kN * kPackedRowBytes);
  for (std::size_t n = 0U; n < kN; ++n) {
    for (std::size_t byte = 0U; byte < kPackedRowBytes; ++byte) {
      const int even =
          static_cast<int>((13U * n + 5U * (2U * byte) + 1U) % 15U) - 7;
      const int odd = static_cast<int>(
                          (7U * n + 3U * (2U * byte + 1U) + 2U) % 15U) -
                      7;
      packed_b[n * kPackedRowBytes + byte] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
  }
  std::vector<std::uint16_t> b_scales(kN * kGroups);
  for (std::size_t n = 0U; n < kN; ++n) {
    for (std::size_t group = 0U; group < kGroups; ++group) {
      b_scales[n * kGroups + group] = encode_bf16(
          static_cast<float>(1U + ((n + group) % 4U)) / 32.0F);
    }
  }

  DeviceBuffer<std::uint16_t> device_input;
  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint8_t> device_b;
  DeviceBuffer<std::uint16_t> device_b_scales;
  DeviceBuffer<std::uint16_t> device_output;
  if (!device_input.allocate(input.size()) ||
      !device_a.allocate(kM * kPackedRowBytes) ||
      !device_a_scales.allocate(kM * kGroups) ||
      !device_b.allocate(packed_b.size()) ||
      !device_b_scales.allocate(b_scales.size()) ||
      !device_output.allocate(kM * kN)) {
    std::cerr << "device allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_input.get(), input.data(),
                          input.size() * sizeof(input[0]),
                          cudaMemcpyHostToDevice),
               "copy input") ||
      !cuda_ok(cudaMemcpy(device_b.get(), packed_b.data(), packed_b.size(),
                          cudaMemcpyHostToDevice),
               "copy B") ||
      !cuda_ok(cudaMemcpy(device_b_scales.get(), b_scales.data(),
                          b_scales.size() * sizeof(b_scales[0]),
                          cudaMemcpyHostToDevice),
               "copy B scales")) {
    return false;
  }

  const int invalid_quant = kernels::launch_sm87_a4_quantize_bf16_cuda(
      device_input.get(), kK, kM, kK, 0.0F, device_a.get(),
      kPackedRowBytes, device_a_scales.get(), kGroups);
  if (invalid_quant != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "zero clip ratio was not rejected\n";
    return false;
  }

  kernels::Sm87A4W4PrefillGemmResources resources{};
  if (!launch_ok(kernels::query_sm87_a4w4_prefill_gemm_resources_cuda(
                     &resources),
                 "query GEMM resources")) {
    return false;
  }
  if (resources.active_blocks_per_sm < 2 || resources.local_bytes != 0U ||
      resources.static_shared_bytes != 12'288U) {
    std::cerr << "resource contract failed: registers="
              << resources.registers_per_thread
              << " shared=" << resources.static_shared_bytes
              << " local=" << resources.local_bytes
              << " active_blocks=" << resources.active_blocks_per_sm
              << '\n';
    return false;
  }

  if (!launch_ok(kernels::launch_sm87_a4_quantize_bf16_cuda(
                     device_input.get(), kK, kM, kK, 1.0F,
                     device_a.get(), kPackedRowBytes,
                     device_a_scales.get(), kGroups),
                 "quantize A4") ||
      !launch_ok(kernels::launch_sm87_a4w4_prefill_gemm_bf16_cuda(
                     device_a.get(), kPackedRowBytes,
                     device_a_scales.get(), kGroups, device_b.get(),
                     kPackedRowBytes, device_b_scales.get(), kM, kN, kK,
                     device_output.get(), kN),
                 "A4W4 GEMM") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize A4W4 GEMM")) {
    return false;
  }

  std::vector<std::uint8_t> packed_a(kM * kPackedRowBytes);
  std::vector<std::uint16_t> a_scales(kM * kGroups);
  std::vector<std::uint16_t> output(kM * kN);
  if (!cuda_ok(cudaMemcpy(packed_a.data(), device_a.get(), packed_a.size(),
                          cudaMemcpyDeviceToHost),
               "copy packed A") ||
      !cuda_ok(cudaMemcpy(a_scales.data(), device_a_scales.get(),
                          a_scales.size() * sizeof(a_scales[0]),
                          cudaMemcpyDeviceToHost),
               "copy A scales") ||
      !cuda_ok(cudaMemcpy(output.data(), device_output.get(),
                          output.size() * sizeof(output[0]),
                          cudaMemcpyDeviceToHost),
               "copy output")) {
    return false;
  }

  std::size_t mismatches = 0U;
  double maximum_absolute_error = 0.0;
  for (std::size_t m = 0U; m < kM; ++m) {
    for (std::size_t n = 0U; n < kN; ++n) {
      float expected = 0.0F;
      for (std::size_t group = 0U; group < kGroups; ++group) {
        std::int32_t integer_partial = 0;
        for (std::size_t inner = 0U; inner < 64U; ++inner) {
          const std::size_t k = group * 64U + inner;
          const int a = signed_code(
              packed_a[m * kPackedRowBytes + k / 2U], k);
          const int b = signed_code(
              packed_b[n * kPackedRowBytes + k / 2U], k);
          integer_partial += a * b;
        }
        expected += static_cast<float>(integer_partial) *
                    decode_bf16(a_scales[m * kGroups + group]) *
                    decode_bf16(b_scales[n * kGroups + group]);
      }
      const std::uint16_t expected_bits = encode_bf16(expected);
      const std::uint16_t actual_bits = output[m * kN + n];
      if (actual_bits != expected_bits) {
        ++mismatches;
        maximum_absolute_error = std::fmax(
            maximum_absolute_error,
            std::fabs(static_cast<double>(decode_bf16(actual_bits)) -
                      static_cast<double>(decode_bf16(expected_bits))));
        if (mismatches <= 8U) {
          std::cerr << "mismatch m=" << m << " n=" << n
                    << " expected=" << decode_bf16(expected_bits)
                    << " actual=" << decode_bf16(actual_bits) << '\n';
        }
      }
    }
  }
  if (mismatches != 0U) {
    std::cerr << "A4W4 GEMM mismatches=" << mismatches
              << " max_abs=" << maximum_absolute_error << '\n';
    return false;
  }
  std::cout << "SM87 A4W4 persistent GEMM passed: M=" << kM
            << " N=" << kN << " K=" << kK
            << " registers=" << resources.registers_per_thread
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return true;
}

}  // namespace

int main() {
  if (!device_is_target()) {
    return 77;
  }
  return run_projection() ? 0 : 1;
}
