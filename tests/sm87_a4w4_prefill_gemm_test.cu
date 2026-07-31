#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

inline constexpr std::size_t kM = 65U;
inline constexpr std::size_t kN = 128U;
inline constexpr std::size_t kK = 128U;
inline constexpr std::size_t kGroups = kK / 64U;
inline constexpr std::size_t kPackedRowBytes = kK / 2U;
inline constexpr std::size_t kAPackedBytes =
    kernels::sm87_a4w4_consumer_packed_capacity_bytes(kM, kK);
inline constexpr std::size_t kAScaleElements =
    kernels::sm87_a4w4_consumer_scale_capacity_elements(kM, kK);
inline constexpr std::size_t kBPackedBytes =
    kernels::sm87_a4w4_consumer_packed_capacity_bytes(kN, kK);
inline constexpr std::size_t kBScaleElements =
    kernels::sm87_a4w4_consumer_scale_capacity_elements(kN, kK);

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

  std::vector<std::uint8_t> packed_b(kBPackedBytes);
  for (std::size_t n = 0U; n < kN; ++n) {
    for (std::size_t byte = 0U; byte < kPackedRowBytes; ++byte) {
      const int even =
          static_cast<int>((13U * n + 5U * (2U * byte) + 1U) % 15U) - 7;
      const int odd = static_cast<int>(
                          (7U * n + 3U * (2U * byte + 1U) + 2U) % 15U) -
                      7;
      const std::size_t group = byte / 32U;
      const std::size_t byte_in_group = byte % 32U;
      packed_b[kernels::sm87_a4w4_consumer_packed_offset(
          n, group, byte_in_group, kGroups)] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
  }
  std::vector<std::uint16_t> b_scales(kBScaleElements);
  for (std::size_t n = 0U; n < kN; ++n) {
    for (std::size_t group = 0U; group < kGroups; ++group) {
      b_scales[kernels::sm87_a4w4_consumer_scale_offset(
          n, group, kGroups)] = encode_bf16(
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
      !device_a.allocate(kAPackedBytes) ||
      !device_a_scales.allocate(kAScaleElements) ||
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
      kAPackedBytes, device_a_scales.get(), kAScaleElements);
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
      resources.static_shared_bytes != 16'320U) {
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
                     device_a.get(), kAPackedBytes,
                     device_a_scales.get(), kAScaleElements),
                 "quantize A4") ||
      !launch_ok(kernels::launch_sm87_a4w4_prefill_gemm_bf16_cuda(
                     device_a.get(), kAPackedBytes,
                     device_a_scales.get(), kAScaleElements, device_b.get(),
                     kBPackedBytes, device_b_scales.get(), kBScaleElements,
                     kM, kN, kK,
                     device_output.get(), kN),
                 "A4W4 GEMM") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize A4W4 GEMM")) {
    return false;
  }

  std::vector<std::uint8_t> packed_a(kAPackedBytes);
  std::vector<std::uint16_t> a_scales(kAScaleElements);
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

  for (std::size_t m = 0U; m < kM; ++m) {
    for (std::size_t group = 0U; group < kGroups; ++group) {
      float maximum = 0.0F;
      for (std::size_t inner = 0U; inner < 64U; ++inner) {
        maximum = std::fmax(
            maximum,
            std::fabs(decode_bf16(input[m * kK + group * 64U + inner])));
      }
      std::uint16_t expected_scale_bits =
          encode_bf16(maximum == 0.0F ? 1.0F : maximum / 7.0F);
      float stored_scale = decode_bf16(expected_scale_bits);
      if (maximum != 0.0F && stored_scale == 0.0F) {
        expected_scale_bits = 1U;
        stored_scale = decode_bf16(expected_scale_bits);
      }
      const std::size_t scale_offset =
          kernels::sm87_a4w4_consumer_scale_offset(m, group, kGroups);
      if (a_scales[scale_offset] != expected_scale_bits) {
        std::cerr << "dynamic A stored-scale mismatch at m=" << m
                  << " group=" << group << '\n';
        return false;
      }
      for (std::size_t inner = 0U; inner < 64U; ++inner) {
        const float value = decode_bf16(
            input[m * kK + group * 64U + inner]);
        const float clipped = std::max(-maximum, std::min(maximum, value));
        const int expected_code = stored_scale == 0.0F
                                      ? 0
                                      : std::max(
                                            -7, std::min(
                                                    7, static_cast<int>(
                                                           std::nearbyint(
                                                               clipped /
                                                               stored_scale))));
        const int actual_code = signed_code(
            packed_a[kernels::sm87_a4w4_consumer_packed_offset(
                m, group, inner / 2U, kGroups)],
            inner);
        if (actual_code != expected_code) {
          std::cerr << "dynamic A stored-scale code mismatch at m=" << m
                    << " group=" << group << " inner=" << inner << '\n';
          return false;
        }
      }
    }
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
              packed_a[kernels::sm87_a4w4_consumer_packed_offset(
                  m, group, (k % 64U) / 2U, kGroups)], k);
          const int b = signed_code(
              packed_b[kernels::sm87_a4w4_consumer_packed_offset(
                  n, group, (k % 64U) / 2U, kGroups)], k);
          integer_partial += a * b;
        }
        expected += static_cast<float>(integer_partial) *
                    decode_bf16(a_scales[
                        kernels::sm87_a4w4_consumer_scale_offset(
                            m, group, kGroups)]) *
                    decode_bf16(b_scales[
                        kernels::sm87_a4w4_consumer_scale_offset(
                            n, group, kGroups)]);
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

[[nodiscard]] bool run_large_m_projection() {
  constexpr std::size_t kLargeM = 1'024U;
  constexpr std::size_t kLargeN = 128U;
  constexpr std::size_t kLargeK = 64U;
  constexpr std::size_t kLargeGroups = 1U;
  constexpr std::size_t kLargeAPackedBytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kLargeM, kLargeK);
  constexpr std::size_t kLargeAScaleElements =
      kernels::sm87_a4w4_consumer_scale_capacity_elements(kLargeM, kLargeK);
  constexpr std::size_t kLargeBPackedBytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kLargeN, kLargeK);
  constexpr std::size_t kLargeBScaleElements =
      kernels::sm87_a4w4_consumer_scale_capacity_elements(kLargeN, kLargeK);
  static_assert(kernels::sm87_a4w4_prefill_uses_large_m_candidate(kLargeM));

  std::vector<std::uint8_t> packed_a(kLargeAPackedBytes);
  std::vector<std::uint16_t> a_scales(kLargeAScaleElements);
  for (std::size_t m = 0U; m < kLargeM; ++m) {
    for (std::size_t byte = 0U; byte < kLargeK / 2U; ++byte) {
      const int even =
          static_cast<int>((7U * m + 5U * (2U * byte) + 1U) % 15U) - 7;
      const int odd = static_cast<int>(
                          (11U * m + 3U * (2U * byte + 1U) + 2U) % 15U) -
                      7;
      packed_a[kernels::sm87_a4w4_consumer_packed_offset(
          m, 0U, byte, kLargeGroups)] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
    a_scales[kernels::sm87_a4w4_consumer_scale_offset(
        m, 0U, kLargeGroups)] =
        encode_bf16(static_cast<float>(1U + m % 4U) / 16.0F);
  }

  std::vector<std::uint8_t> packed_b(kLargeBPackedBytes);
  std::vector<std::uint16_t> b_scales(kLargeBScaleElements);
  for (std::size_t n = 0U; n < kLargeN; ++n) {
    for (std::size_t byte = 0U; byte < kLargeK / 2U; ++byte) {
      const int even =
          static_cast<int>((13U * n + 2U * byte + 3U) % 15U) - 7;
      const int odd =
          static_cast<int>((3U * n + 7U * byte + 4U) % 15U) - 7;
      packed_b[kernels::sm87_a4w4_consumer_packed_offset(
          n, 0U, byte, kLargeGroups)] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
    b_scales[kernels::sm87_a4w4_consumer_scale_offset(
        n, 0U, kLargeGroups)] =
        encode_bf16(static_cast<float>(1U + n % 3U) / 32.0F);
  }

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint8_t> device_b;
  DeviceBuffer<std::uint16_t> device_b_scales;
  DeviceBuffer<std::uint16_t> device_output;
  if (!device_a.allocate(packed_a.size()) ||
      !device_a_scales.allocate(a_scales.size()) ||
      !device_b.allocate(packed_b.size()) ||
      !device_b_scales.allocate(b_scales.size()) ||
      !device_output.allocate(kLargeM * kLargeN)) {
    std::cerr << "large-M device allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_a.get(), packed_a.data(), packed_a.size(),
                          cudaMemcpyHostToDevice),
               "copy large-M A") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), a_scales.data(),
                          a_scales.size() * sizeof(a_scales[0]),
                          cudaMemcpyHostToDevice),
               "copy large-M A scales") ||
      !cuda_ok(cudaMemcpy(device_b.get(), packed_b.data(), packed_b.size(),
                          cudaMemcpyHostToDevice),
               "copy large-M B") ||
      !cuda_ok(cudaMemcpy(device_b_scales.get(), b_scales.data(),
                          b_scales.size() * sizeof(b_scales[0]),
                          cudaMemcpyHostToDevice),
               "copy large-M B scales")) {
    return false;
  }

  kernels::Sm87A4W4PrefillGemmResources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_prefill_gemm_m64n64_resources_cuda(
              &resources),
          "query M64N64 resources")) {
    return false;
  }
  if (resources.active_blocks_per_sm < 2 || resources.local_bytes != 0U ||
      resources.static_shared_bytes != 13'056U) {
    std::cerr << "M64N64 resource contract failed: registers="
              << resources.registers_per_thread
              << " shared=" << resources.static_shared_bytes
              << " local=" << resources.local_bytes
              << " active_blocks=" << resources.active_blocks_per_sm << '\n';
    return false;
  }

  if (!launch_ok(kernels::launch_sm87_a4w4_prefill_gemm_bf16_cuda(
                     device_a.get(), packed_a.size(), device_a_scales.get(),
                     a_scales.size(), device_b.get(), packed_b.size(),
                     device_b_scales.get(), b_scales.size(), kLargeM,
                     kLargeN, kLargeK, device_output.get(), kLargeN),
                 "M64N64 A4W4 GEMM") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize M64N64 GEMM")) {
    return false;
  }

  std::vector<std::uint16_t> output(kLargeM * kLargeN);
  if (!cuda_ok(cudaMemcpy(output.data(), device_output.get(),
                          output.size() * sizeof(output[0]),
                          cudaMemcpyDeviceToHost),
               "copy M64N64 output")) {
    return false;
  }

  std::size_t mismatches = 0U;
  for (std::size_t m = 0U; m < kLargeM; ++m) {
    for (std::size_t n = 0U; n < kLargeN; ++n) {
      std::int32_t integer_partial = 0;
      for (std::size_t k = 0U; k < kLargeK; ++k) {
        const int a = signed_code(
            packed_a[kernels::sm87_a4w4_consumer_packed_offset(
                m, 0U, k / 2U, kLargeGroups)],
            k);
        const int b = signed_code(
            packed_b[kernels::sm87_a4w4_consumer_packed_offset(
                n, 0U, k / 2U, kLargeGroups)],
            k);
        integer_partial += a * b;
      }
      const float expected =
          static_cast<float>(integer_partial) *
          decode_bf16(a_scales[kernels::sm87_a4w4_consumer_scale_offset(
              m, 0U, kLargeGroups)]) *
          decode_bf16(b_scales[kernels::sm87_a4w4_consumer_scale_offset(
              n, 0U, kLargeGroups)]);
      const std::uint16_t expected_bits = encode_bf16(expected);
      const std::uint16_t actual_bits = output[m * kLargeN + n];
      if (actual_bits != expected_bits) {
        ++mismatches;
        if (mismatches <= 8U) {
          std::cerr << "M64N64 mismatch m=" << m << " n=" << n
                    << " expected=" << decode_bf16(expected_bits)
                    << " actual=" << decode_bf16(actual_bits) << '\n';
        }
      }
    }
  }
  if (mismatches != 0U) {
    std::cerr << "M64N64 GEMM mismatches=" << mismatches << '\n';
    return false;
  }
  std::cout << "SM87 A4W4 M64N64 GEMM passed: M=" << kLargeM
            << " N=" << kLargeN << " K=" << kLargeK
            << " registers=" << resources.registers_per_thread
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return true;
}

[[nodiscard]] bool run_wide_m_projection() {
  constexpr std::size_t kWideM = 2'048U;
  constexpr std::size_t kWideN = 256U;
  constexpr std::size_t kWideK = 192U;
  constexpr std::size_t kWideGroups = kWideK / 64U;
  constexpr std::size_t kWideOutputStride = kWideN + 16U;
  constexpr std::size_t kWideAPackedBytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kWideM, kWideK);
  constexpr std::size_t kWideAScaleElements =
      kernels::sm87_a4w4_consumer_scale_capacity_elements(kWideM, kWideK);
  constexpr std::size_t kWideBPackedBytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kWideN, kWideK);
  constexpr std::size_t kWideBScaleElements =
      kernels::sm87_a4w4_consumer_scale_capacity_elements(kWideN, kWideK);
  static_assert(kernels::sm87_a4w4_prefill_uses_m64n256_candidate(
      kWideM, kWideN));
  static_assert(kernels::sm87_a4w4_prefill_gemm_m64n256_plan(
                    kWideM, kWideN, kWideK)
                    .work_tiles == 32U);

  std::vector<std::uint8_t> packed_a(kWideAPackedBytes);
  std::vector<std::uint16_t> a_scales(kWideAScaleElements);
  for (std::size_t m = 0U; m < kWideM; ++m) {
    for (std::size_t group = 0U; group < kWideGroups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const int even = static_cast<int>(
                             (7U * m + 11U * group + 5U * byte + 1U) %
                             15U) -
                         7;
        const int odd = static_cast<int>(
                            (3U * m + 13U * group + 9U * byte + 2U) %
                            15U) -
                        7;
        packed_a[kernels::sm87_a4w4_consumer_packed_offset(
            m, group, byte, kWideGroups)] =
            kernels::sm87_a4w4_pack_signed_pair(even, odd);
      }
      a_scales[kernels::sm87_a4w4_consumer_scale_offset(
          m, group, kWideGroups)] =
          encode_bf16(static_cast<float>(1U + (m + 3U * group) % 7U) /
                      32.0F);
    }
  }

  std::vector<std::uint8_t> packed_b(kWideBPackedBytes);
  std::vector<std::uint16_t> b_scales(kWideBScaleElements);
  for (std::size_t n = 0U; n < kWideN; ++n) {
    for (std::size_t group = 0U; group < kWideGroups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const int even = static_cast<int>(
                             (13U * n + 7U * group + 2U * byte + 3U) %
                             15U) -
                         7;
        const int odd = static_cast<int>(
                            (5U * n + 11U * group + 7U * byte + 4U) %
                            15U) -
                        7;
        packed_b[kernels::sm87_a4w4_consumer_packed_offset(
            n, group, byte, kWideGroups)] =
            kernels::sm87_a4w4_pack_signed_pair(even, odd);
      }
      b_scales[kernels::sm87_a4w4_consumer_scale_offset(
          n, group, kWideGroups)] =
          encode_bf16(static_cast<float>(1U + (n + 5U * group) % 5U) /
                      64.0F);
    }
  }

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint8_t> device_b;
  DeviceBuffer<std::uint16_t> device_b_scales;
  DeviceBuffer<std::uint16_t> device_output;
  if (!device_a.allocate(packed_a.size()) ||
      !device_a_scales.allocate(a_scales.size()) ||
      !device_b.allocate(packed_b.size()) ||
      !device_b_scales.allocate(b_scales.size()) ||
      !device_output.allocate(kWideM * kWideOutputStride)) {
    std::cerr << "wide-M device allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_a.get(), packed_a.data(), packed_a.size(),
                          cudaMemcpyHostToDevice),
               "copy wide-M A") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), a_scales.data(),
                          a_scales.size() * sizeof(a_scales[0]),
                          cudaMemcpyHostToDevice),
               "copy wide-M A scales") ||
      !cuda_ok(cudaMemcpy(device_b.get(), packed_b.data(), packed_b.size(),
                          cudaMemcpyHostToDevice),
               "copy wide-M B") ||
      !cuda_ok(cudaMemcpy(device_b_scales.get(), b_scales.data(),
                          b_scales.size() * sizeof(b_scales[0]),
                          cudaMemcpyHostToDevice),
               "copy wide-M B scales") ||
      !cuda_ok(cudaMemset(device_output.get(), 0x5a,
                          kWideM * kWideOutputStride *
                              sizeof(std::uint16_t)),
               "initialize wide-M output")) {
    return false;
  }

  kernels::Sm87A4W4PrefillGemmResources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_prefill_gemm_m64n256_resources_cuda(
              &resources),
          "query M64N256 resources")) {
    return false;
  }
  if (resources.active_blocks_per_sm < 2 || resources.local_bytes != 0U ||
      resources.static_shared_bytes != 43'520U) {
    std::cerr << "M64N256 resource contract failed: registers="
              << resources.registers_per_thread
              << " shared=" << resources.static_shared_bytes
              << " local=" << resources.local_bytes
              << " active_blocks=" << resources.active_blocks_per_sm << '\n';
    return false;
  }

  if (!launch_ok(kernels::launch_sm87_a4w4_prefill_gemm_bf16_cuda(
                     device_a.get(), packed_a.size(), device_a_scales.get(),
                     a_scales.size(), device_b.get(), packed_b.size(),
                     device_b_scales.get(), b_scales.size(), kWideM, kWideN,
                     kWideK, device_output.get(), kWideOutputStride),
                 "M64N256 A4W4 GEMM") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize M64N256 GEMM")) {
    return false;
  }

  std::vector<std::uint16_t> output(kWideM * kWideOutputStride);
  if (!cuda_ok(cudaMemcpy(output.data(), device_output.get(),
                          output.size() * sizeof(output[0]),
                          cudaMemcpyDeviceToHost),
               "copy M64N256 output")) {
    return false;
  }

  std::size_t mismatches = 0U;
  for (std::size_t m = 0U; m < kWideM; ++m) {
    for (std::size_t n = 0U; n < kWideN; ++n) {
      float expected = 0.0F;
      for (std::size_t group = 0U; group < kWideGroups; ++group) {
        std::int32_t integer_partial = 0;
        for (std::size_t k = 0U; k < 64U; ++k) {
          const int a = signed_code(
              packed_a[kernels::sm87_a4w4_consumer_packed_offset(
                  m, group, k / 2U, kWideGroups)],
              k);
          const int b = signed_code(
              packed_b[kernels::sm87_a4w4_consumer_packed_offset(
                  n, group, k / 2U, kWideGroups)],
              k);
          integer_partial += a * b;
        }
        expected +=
            static_cast<float>(integer_partial) *
            decode_bf16(a_scales[kernels::sm87_a4w4_consumer_scale_offset(
                m, group, kWideGroups)]) *
            decode_bf16(b_scales[kernels::sm87_a4w4_consumer_scale_offset(
                n, group, kWideGroups)]);
      }
      const std::uint16_t expected_bits = encode_bf16(expected);
      const std::uint16_t actual_bits =
          output[m * kWideOutputStride + n];
      if (actual_bits != expected_bits) {
        ++mismatches;
        if (mismatches <= 8U) {
          std::cerr << "M64N256 mismatch m=" << m << " n=" << n
                    << " expected=" << decode_bf16(expected_bits)
                    << " actual=" << decode_bf16(actual_bits) << '\n';
        }
      }
    }
    for (std::size_t n = kWideN; n < kWideOutputStride; ++n) {
      if (output[m * kWideOutputStride + n] != 0x5a5aU) {
        ++mismatches;
        if (mismatches <= 8U) {
          std::cerr << "M64N256 padding overwrite m=" << m
                    << " n=" << n << '\n';
        }
      }
    }
  }
  if (mismatches != 0U) {
    std::cerr << "M64N256 GEMM mismatches=" << mismatches << '\n';
    return false;
  }
  std::cout << "SM87 A4W4 M64N256 GEMM passed: M=" << kWideM
            << " N=" << kWideN << " K=" << kWideK
            << " registers=" << resources.registers_per_thread
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return true;
}

[[nodiscard]] bool run_arbitrary_m_composition_case(
    const std::size_t m_count) {
  constexpr std::size_t n_count = 256U;
  constexpr std::size_t k_count = 64U;
  constexpr std::size_t k64_groups = 1U;
  constexpr std::size_t output_stride = n_count + 16U;
  constexpr std::size_t guard_elements = 64U;
  const std::size_t a_packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          m_count, k_count);
  const std::size_t a_scale_elements =
      kernels::sm87_a4w4_consumer_scale_capacity_elements(
          m_count, k_count);
  constexpr std::size_t b_packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          n_count, k_count);
  constexpr std::size_t b_scale_elements =
      kernels::sm87_a4w4_consumer_scale_capacity_elements(
          n_count, k_count);
  const std::size_t output_elements =
      m_count * output_stride + guard_elements;

  std::vector<std::uint8_t> packed_a(a_packed_bytes, 0U);
  std::vector<std::uint16_t> a_scales(a_scale_elements, 0U);
  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t byte = 0U; byte < 32U; ++byte) {
      const int even =
          static_cast<int>((7U * m + 3U * byte + 1U) % 15U) - 7;
      const int odd =
          static_cast<int>((11U * m + 5U * byte + 2U) % 15U) - 7;
      packed_a[kernels::sm87_a4w4_consumer_packed_offset(
          m, 0U, byte, k64_groups)] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
    a_scales[kernels::sm87_a4w4_consumer_scale_offset(
        m, 0U, k64_groups)] =
        encode_bf16(static_cast<float>(1U + m % 5U) / 32.0F);
  }
  std::vector<std::uint8_t> packed_b(b_packed_bytes, 0U);
  std::vector<std::uint16_t> b_scales(b_scale_elements, 0U);
  for (std::size_t n = 0U; n < n_count; ++n) {
    for (std::size_t byte = 0U; byte < 32U; ++byte) {
      const int even =
          static_cast<int>((13U * n + 2U * byte + 3U) % 15U) - 7;
      const int odd =
          static_cast<int>((3U * n + 7U * byte + 4U) % 15U) - 7;
      packed_b[kernels::sm87_a4w4_consumer_packed_offset(
          n, 0U, byte, k64_groups)] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
    b_scales[kernels::sm87_a4w4_consumer_scale_offset(
        n, 0U, k64_groups)] =
        encode_bf16(static_cast<float>(1U + n % 3U) / 64.0F);
  }

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint8_t> device_b;
  DeviceBuffer<std::uint16_t> device_b_scales;
  DeviceBuffer<std::uint16_t> candidate_output;
  DeviceBuffer<std::uint16_t> baseline_output;
  if (!device_a.allocate(a_packed_bytes) ||
      !device_a_scales.allocate(a_scale_elements) ||
      !device_b.allocate(b_packed_bytes) ||
      !device_b_scales.allocate(b_scale_elements) ||
      !candidate_output.allocate(output_elements) ||
      !baseline_output.allocate(output_elements)) {
    std::cerr << "arbitrary-M generic allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_a.get(), packed_a.data(),
                          a_packed_bytes, cudaMemcpyHostToDevice),
               "copy arbitrary-M generic A") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), a_scales.data(),
                          a_scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy arbitrary-M generic A scales") ||
      !cuda_ok(cudaMemcpy(device_b.get(), packed_b.data(),
                          b_packed_bytes, cudaMemcpyHostToDevice),
               "copy arbitrary-M generic B") ||
      !cuda_ok(cudaMemcpy(device_b_scales.get(), b_scales.data(),
                          b_scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy arbitrary-M generic B scales") ||
      !cuda_ok(cudaMemset(candidate_output.get(), 0x5a,
                          output_elements * sizeof(std::uint16_t)),
               "initialize arbitrary-M candidate") ||
      !cuda_ok(cudaMemset(baseline_output.get(), 0x5a,
                          output_elements * sizeof(std::uint16_t)),
               "initialize arbitrary-M baseline")) {
    return false;
  }

  if (m_count == 3'987U &&
      kernels::launch_sm87_a4w4_prefill_gemm_bf16_cuda(
          device_a.get(), a_packed_bytes - 1U, device_a_scales.get(),
          a_scale_elements, device_b.get(), b_packed_bytes,
          device_b_scales.get(), b_scale_elements, m_count, n_count,
          k_count, candidate_output.get(), output_stride) !=
          static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "arbitrary-M generic short A capacity was not rejected\n";
    return false;
  }
  if (!launch_ok(kernels::launch_sm87_a4w4_prefill_gemm_bf16_cuda(
                     device_a.get(), a_packed_bytes,
                     device_a_scales.get(), a_scale_elements,
                     device_b.get(), b_packed_bytes,
                     device_b_scales.get(), b_scale_elements,
                     m_count, n_count, k_count, candidate_output.get(),
                     output_stride),
                 "launch arbitrary-M generic candidate")) {
    return false;
  }

  std::size_t first_m = 0U;
  while (first_m < m_count) {
    std::size_t chunk_m = std::min<std::size_t>(512U, m_count - first_m);
    if (m_count == 65U && first_m == 0U) {
      chunk_m = 64U;
    }
    const std::size_t a_byte_offset = first_m * (k_count / 2U);
    const std::size_t a_scale_offset = first_m * k64_groups;
    const std::size_t chunk_a_bytes =
        kernels::sm87_a4w4_consumer_packed_capacity_bytes(
            chunk_m, k_count);
    const std::size_t chunk_a_scales =
        kernels::sm87_a4w4_consumer_scale_capacity_elements(
            chunk_m, k_count);
    if (!launch_ok(kernels::launch_sm87_a4w4_prefill_gemm_bf16_cuda(
                       device_a.get() + a_byte_offset, chunk_a_bytes,
                       device_a_scales.get() + a_scale_offset,
                       chunk_a_scales, device_b.get(), b_packed_bytes,
                       device_b_scales.get(), b_scale_elements,
                       chunk_m, n_count, k_count,
                       baseline_output.get() + first_m * output_stride,
                       output_stride),
                   "launch arbitrary-M generic retained baseline")) {
      return false;
    }
    first_m += chunk_m;
  }
  if (!cuda_ok(cudaDeviceSynchronize(),
               "synchronize arbitrary-M generic comparison")) {
    return false;
  }

  std::vector<std::uint16_t> candidate(output_elements);
  std::vector<std::uint16_t> baseline(output_elements);
  if (!cuda_ok(cudaMemcpy(candidate.data(), candidate_output.get(),
                          output_elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy arbitrary-M generic candidate") ||
      !cuda_ok(cudaMemcpy(baseline.data(), baseline_output.get(),
                          output_elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy arbitrary-M generic baseline")) {
    return false;
  }
  for (std::size_t index = 0U; index < output_elements; ++index) {
    if (candidate[index] != baseline[index]) {
      std::cerr << "arbitrary-M generic bit mismatch: M=" << m_count
                << " element=" << index
                << " candidate=" << candidate[index]
                << " baseline=" << baseline[index] << '\n';
      return false;
    }
  }
  std::cout << "SM87 A4W4 generic arbitrary-M composition bit-exact: M="
            << m_count << '\n';
  return true;
}

[[nodiscard]] bool run_arbitrary_m_composition() {
  for (const std::size_t m_count : {65U, 1'025U, 1'804U, 3'987U}) {
    if (!run_arbitrary_m_composition_case(m_count)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool run_shared_k128_projection() {
  constexpr std::size_t m_count = 64U;
  constexpr std::size_t n_count = 256U;
  constexpr std::size_t k_count = 256U;
  constexpr std::size_t output_stride = n_count + 8U;
  constexpr std::size_t k128_groups = k_count / 128U;
  constexpr std::size_t physical_k64_groups = k_count / 64U;
  constexpr std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count, k_count);
  constexpr std::size_t a_scale_elements =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(m_count,
                                                               k_count);
  constexpr std::size_t b_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(n_count, k_count);
  constexpr std::size_t b_scale_elements =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(n_count,
                                                               k_count);
  static_assert(kernels::sm87_a4w4_prefill_gemm_k128_plan(
                    m_count, n_count, k_count)
                    .launch_ctas == 1U);
  static_assert(k128_groups == 2U && physical_k64_groups == 4U);

  std::vector<std::uint16_t> input(m_count * k_count);
  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t k = 0U; k < k_count; ++k) {
      const int centered =
          static_cast<int>((23U * m + 17U * k + 5U) % 61U) - 30;
      input[m * k_count + k] =
          encode_bf16(static_cast<float>(centered) / 16.0F);
    }
  }

  std::vector<std::uint8_t> packed_b(b_bytes);
  for (std::size_t n = 0U; n < n_count; ++n) {
    for (std::size_t k = 0U; k < k_count; k += 2U) {
      const int even =
          static_cast<int>((11U * n + 7U * k + 3U) % 15U) - 7;
      const int odd =
          static_cast<int>((19U * n + 5U * (k + 1U) + 1U) % 15U) - 7;
      const std::size_t physical_group = k / 64U;
      packed_b[kernels::sm87_a4w4_consumer_packed_offset(
          n, physical_group, (k % 64U) / 2U,
          physical_k64_groups)] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
  }
  std::vector<std::uint16_t> b_scales(b_scale_elements);
  for (std::size_t n = 0U; n < n_count; ++n) {
    for (std::size_t group = 0U; group < k128_groups; ++group) {
      b_scales[kernels::sm87_a4w4_consumer_k128_scale_offset(
          n, group, k128_groups)] = encode_bf16(
          static_cast<float>(1U + ((3U * n + group) % 7U)) / 64.0F);
    }
  }

  DeviceBuffer<std::uint16_t> device_input;
  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint8_t> device_b;
  DeviceBuffer<std::uint16_t> device_b_scales;
  DeviceBuffer<std::uint16_t> device_output;
  if (!device_input.allocate(input.size()) ||
      !device_a.allocate(a_bytes) ||
      !device_a_scales.allocate(a_scale_elements) ||
      !device_b.allocate(b_bytes) ||
      !device_b_scales.allocate(b_scale_elements) ||
      !device_output.allocate(m_count * output_stride)) {
    std::cerr << "K128 device allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_input.get(), input.data(),
                          input.size() * sizeof(input[0]),
                          cudaMemcpyHostToDevice),
               "copy K128 input") ||
      !cuda_ok(cudaMemcpy(device_b.get(), packed_b.data(), packed_b.size(),
                          cudaMemcpyHostToDevice),
               "copy K128 B") ||
      !cuda_ok(cudaMemcpy(device_b_scales.get(), b_scales.data(),
                          b_scales.size() * sizeof(b_scales[0]),
                          cudaMemcpyHostToDevice),
               "copy K128 B scales") ||
      !cuda_ok(cudaMemset(device_output.get(), 0x5a,
                          m_count * output_stride *
                              sizeof(std::uint16_t)),
               "initialize K128 output guard")) {
    return false;
  }

  const int overflow_status =
      kernels::launch_sm87_a4_quantize_bf16_k128_cuda(
          device_input.get(), k_count,
          std::numeric_limits<std::size_t>::max(), k_count, 1.0F,
          device_a.get(), a_bytes, device_a_scales.get(), a_scale_elements);
  if (overflow_status != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "K128 quantizer overflow shape was not rejected\n";
    return false;
  }
  const int short_quant_capacity_status =
      kernels::launch_sm87_a4_quantize_bf16_k128_cuda(
          device_input.get(), k_count, m_count, k_count, 1.0F,
          device_a.get(), a_bytes - 1U, device_a_scales.get(),
          a_scale_elements);
  if (short_quant_capacity_status !=
      static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "K128 quantizer short packed capacity was not rejected\n";
    return false;
  }
  const int short_scale_status =
      kernels::launch_sm87_a4w4_prefill_gemm_k128_bf16_cuda(
          device_a.get(), a_bytes, device_a_scales.get(),
          a_scale_elements - 1U, device_b.get(), b_bytes,
          device_b_scales.get(), b_scale_elements, m_count, n_count,
          k_count, device_output.get(), output_stride);
  if (short_scale_status != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "K128 projection short scale capacity was not rejected\n";
    return false;
  }

  kernels::Sm87A4W4PrefillGemmResources resources{};
  if (!launch_ok(kernels::query_sm87_a4w4_prefill_gemm_k128_resources_cuda(
                     &resources),
                 "query K128 projection resources")) {
    return false;
  }
  if (resources.registers_per_thread > 128 ||
      resources.active_blocks_per_sm < 2 || resources.local_bytes != 0U ||
      resources.static_shared_bytes != 42'240U) {
    std::cerr << "K128 resource contract failed: registers="
              << resources.registers_per_thread
              << " shared=" << resources.static_shared_bytes
              << " local=" << resources.local_bytes
              << " active_blocks=" << resources.active_blocks_per_sm
              << '\n';
    return false;
  }

  if (!launch_ok(kernels::launch_sm87_a4_quantize_bf16_k128_cuda(
                     device_input.get(), k_count, m_count, k_count, 1.0F,
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_elements),
                 "quantize shared-K128 A4") ||
      !launch_ok(kernels::launch_sm87_a4w4_prefill_gemm_k128_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_elements, device_b.get(), b_bytes,
                     device_b_scales.get(), b_scale_elements, m_count,
                     n_count, k_count, device_output.get(), output_stride),
                 "launch shared-K128 projection") ||
      !cuda_ok(cudaDeviceSynchronize(),
               "synchronize shared-K128 projection")) {
    return false;
  }

  std::vector<std::uint8_t> packed_a(a_bytes);
  std::vector<std::uint16_t> a_scales(a_scale_elements);
  std::vector<std::uint16_t> output(m_count * output_stride);
  if (!cuda_ok(cudaMemcpy(packed_a.data(), device_a.get(), packed_a.size(),
                          cudaMemcpyDeviceToHost),
               "copy K128 packed A") ||
      !cuda_ok(cudaMemcpy(a_scales.data(), device_a_scales.get(),
                          a_scales.size() * sizeof(a_scales[0]),
                          cudaMemcpyDeviceToHost),
               "copy K128 A scales") ||
      !cuda_ok(cudaMemcpy(output.data(), device_output.get(),
                          output.size() * sizeof(output[0]),
                          cudaMemcpyDeviceToHost),
               "copy K128 output")) {
    return false;
  }

  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t group = 0U; group < k128_groups; ++group) {
      float maximum = 0.0F;
      for (std::size_t inner = 0U; inner < 128U; ++inner) {
        maximum = std::fmax(
            maximum,
            std::fabs(decode_bf16(
                input[m * k_count + group * 128U + inner])));
      }
      std::uint16_t expected_scale =
          encode_bf16(maximum == 0.0F ? 1.0F : maximum / 7.0F);
      float stored_scale = decode_bf16(expected_scale);
      if (maximum != 0.0F && stored_scale == 0.0F) {
        expected_scale = 1U;
        stored_scale = decode_bf16(expected_scale);
      }
      const std::size_t scale_offset =
          kernels::sm87_a4w4_consumer_k128_scale_offset(
              m, group, k128_groups);
      if (a_scales[scale_offset] != expected_scale) {
        std::cerr << "K128 dynamic scale mismatch m=" << m
                  << " group=" << group << '\n';
        return false;
      }
      for (std::size_t inner = 0U; inner < 128U; ++inner) {
        const std::size_t k = group * 128U + inner;
        const float value = decode_bf16(input[m * k_count + k]);
        const int expected_code = stored_scale == 0.0F
                                      ? 0
                                      : std::max(
                                            -7,
                                            std::min(
                                                7,
                                                static_cast<int>(
                                                    std::nearbyint(
                                                        value /
                                                        stored_scale))));
        const std::size_t physical_group = k / 64U;
        const int actual_code = kernels::sm87_a4w4_unpack_signed(
            packed_a[kernels::sm87_a4w4_consumer_packed_offset(
                m, physical_group, (k % 64U) / 2U,
                physical_k64_groups)],
            k);
        if (actual_code != expected_code) {
          std::cerr << "K128 dynamic code mismatch m=" << m
                    << " k=" << k << '\n';
          return false;
        }
      }
    }
  }

  std::size_t mismatches = 0U;
  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t n = 0U; n < n_count; ++n) {
      float expected = 0.0F;
      for (std::size_t group = 0U; group < k128_groups; ++group) {
        std::int32_t integer_partial = 0;
        for (std::size_t inner = 0U; inner < 128U; ++inner) {
          const std::size_t k = group * 128U + inner;
          const std::size_t physical_group = k / 64U;
          const int a = kernels::sm87_a4w4_unpack_signed(
              packed_a[kernels::sm87_a4w4_consumer_packed_offset(
                  m, physical_group, (k % 64U) / 2U,
                  physical_k64_groups)],
              k);
          const int b = kernels::sm87_a4w4_unpack_signed(
              packed_b[kernels::sm87_a4w4_consumer_packed_offset(
                  n, physical_group, (k % 64U) / 2U,
                  physical_k64_groups)],
              k);
          integer_partial += a * b;
        }
        const float scale_product =
            decode_bf16(a_scales[
                kernels::sm87_a4w4_consumer_k128_scale_offset(
                    m, group, k128_groups)]) *
            decode_bf16(b_scales[
                kernels::sm87_a4w4_consumer_k128_scale_offset(
                    n, group, k128_groups)]);
        expected += static_cast<float>(integer_partial) * scale_product;
      }
      const std::uint16_t expected_bits = encode_bf16(expected);
      if (output[m * output_stride + n] != expected_bits) {
        ++mismatches;
        if (mismatches <= 8U) {
          std::cerr << "shared-K128 mismatch m=" << m << " n=" << n
                    << " expected=" << decode_bf16(expected_bits)
                    << " actual="
                    << decode_bf16(output[m * output_stride + n]) << '\n';
        }
      }
    }
    for (std::size_t n = n_count; n < output_stride; ++n) {
      if (output[m * output_stride + n] != 0x5a5aU) {
        ++mismatches;
      }
    }
  }
  if (mismatches != 0U) {
    std::cerr << "shared-K128 projection mismatches=" << mismatches << '\n';
    return false;
  }
  std::cout << "SM87 A4W4 shared-K128 projection passed: M=" << m_count
            << " N=" << n_count << " K=" << k_count
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
  return run_projection() && run_large_m_projection() &&
                 run_wide_m_projection() && run_arbitrary_m_composition() &&
                 run_shared_k128_projection()
             ? 0
             : 1;
}
