#include "q3x/kernels/sm87_a4w4_gateup_paired.h"
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
inline constexpr std::size_t kK = 192U;
inline constexpr std::size_t kGroups = kK / 64U;
inline constexpr std::size_t kOutputGroups = kN / 64U;
inline constexpr std::size_t kPackedInputRowBytes = kK / 2U;
inline constexpr std::size_t kPackedOutputRowBytes = kN / 2U;
inline constexpr std::size_t kInputScaleStride = kGroups + 1U;
inline constexpr std::size_t kWeightScaleStride = kGroups + 1U;
inline constexpr std::size_t kOutputPackedStride =
    kPackedOutputRowBytes + 16U;
inline constexpr std::size_t kOutputScaleStride = kOutputGroups + 2U;
inline constexpr float kOutputClipRatio = 0.9375F;

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

[[nodiscard]] float silu_product(const float gate, const float up) noexcept {
  if (gate >= 0.0F) {
    return (gate / (1.0F + std::exp(-gate))) * up;
  }
  const float exponential = std::exp(gate);
  return (gate * exponential / (1.0F + exponential)) * up;
}

[[nodiscard]] int round_and_clamp(const float value) noexcept {
  const int rounded = static_cast<int>(std::nearbyint(value));
  return std::max(-7, std::min(7, rounded));
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

[[nodiscard]] int code_at(const std::vector<std::uint8_t>& packed,
                          const std::size_t row_stride,
                          const std::size_t row,
                          const std::size_t inner) noexcept {
  return kernels::sm87_a4w4_unpack_signed(
      packed[row * row_stride + inner / 2U], inner);
}

struct HostPayload final {
  std::vector<std::uint8_t> a;
  std::vector<std::uint16_t> a_scales;
  std::vector<std::uint8_t> gate_b;
  std::vector<std::uint16_t> gate_scales;
  std::vector<std::uint8_t> up_b;
  std::vector<std::uint16_t> up_scales;
};

[[nodiscard]] HostPayload make_payload() {
  HostPayload payload{
      std::vector<std::uint8_t>(kM * kPackedInputRowBytes),
      std::vector<std::uint16_t>(kM * kInputScaleStride,
                                 static_cast<std::uint16_t>(0x7fc1U)),
      std::vector<std::uint8_t>(kN * kPackedInputRowBytes),
      std::vector<std::uint16_t>(kN * kWeightScaleStride,
                                 static_cast<std::uint16_t>(0x7fc1U)),
      std::vector<std::uint8_t>(kN * kPackedInputRowBytes),
      std::vector<std::uint16_t>(kN * kWeightScaleStride,
                                 static_cast<std::uint16_t>(0x7fc1U))};

  for (std::size_t m = 0U; m < kM; ++m) {
    for (std::size_t byte = 0U; byte < kPackedInputRowBytes; ++byte) {
      const int even =
          static_cast<int>((19U * m + 5U * (2U * byte) + 3U) % 15U) - 7;
      const int odd = static_cast<int>(
                          (11U * m + 7U * (2U * byte + 1U) + 1U) % 15U) -
                      7;
      payload.a[m * kPackedInputRowBytes + byte] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
    for (std::size_t group = 0U; group < kGroups; ++group) {
      payload.a_scales[m * kInputScaleStride + group] = encode_bf16(
          static_cast<float>(2U + ((m + 3U * group) % 5U)) / 64.0F);
    }
  }

  for (std::size_t n = 0U; n < kN; ++n) {
    for (std::size_t byte = 0U; byte < kPackedInputRowBytes; ++byte) {
      const int gate_even =
          static_cast<int>((13U * n + 3U * (2U * byte) + 2U) % 15U) - 7;
      const int gate_odd = static_cast<int>(
                               (5U * n + 11U * (2U * byte + 1U) + 4U) %
                               15U) -
                           7;
      const int up_even =
          static_cast<int>((7U * n + 9U * (2U * byte) + 6U) % 15U) - 7;
      const int up_odd = static_cast<int>(
                             (17U * n + 5U * (2U * byte + 1U) + 8U) %
                             15U) -
                         7;
      payload.gate_b[n * kPackedInputRowBytes + byte] =
          kernels::sm87_a4w4_pack_signed_pair(gate_even, gate_odd);
      payload.up_b[n * kPackedInputRowBytes + byte] =
          kernels::sm87_a4w4_pack_signed_pair(up_even, up_odd);
    }
    for (std::size_t group = 0U; group < kGroups; ++group) {
      payload.gate_scales[n * kWeightScaleStride + group] = encode_bf16(
          static_cast<float>(1U + ((n + group) % 4U)) / 128.0F);
      payload.up_scales[n * kWeightScaleStride + group] = encode_bf16(
          static_cast<float>(2U + ((3U * n + group) % 5U)) / 128.0F);
    }
  }
  return payload;
}

[[nodiscard]] std::vector<float> reference_products(
    const HostPayload& payload) {
  std::vector<float> products(kM * kN);
  for (std::size_t m = 0U; m < kM; ++m) {
    for (std::size_t n = 0U; n < kN; ++n) {
      float gate = 0.0F;
      float up = 0.0F;
      for (std::size_t group = 0U; group < kGroups; ++group) {
        std::int32_t gate_integer = 0;
        std::int32_t up_integer = 0;
        for (std::size_t inner = 0U; inner < 64U; ++inner) {
          const std::size_t k = group * 64U + inner;
          const int a = code_at(payload.a, kPackedInputRowBytes, m, k);
          const int gate_b =
              code_at(payload.gate_b, kPackedInputRowBytes, n, k);
          const int up_b =
              code_at(payload.up_b, kPackedInputRowBytes, n, k);
          gate_integer += a * gate_b;
          up_integer += a * up_b;
        }
        const float a_scale =
            decode_bf16(payload.a_scales[m * kInputScaleStride + group]);
        gate += static_cast<float>(gate_integer) * a_scale *
                decode_bf16(payload.gate_scales[
                    n * kWeightScaleStride + group]);
        up += static_cast<float>(up_integer) * a_scale *
              decode_bf16(
                  payload.up_scales[n * kWeightScaleStride + group]);
      }
      products[m * kN + n] = silu_product(gate, up);
    }
  }
  return products;
}

struct QuantizedReference final {
  std::vector<std::uint8_t> packed;
  std::vector<std::uint16_t> scales;
};

[[nodiscard]] QuantizedReference quantize_reference(
    const std::vector<float>& products) {
  QuantizedReference reference{
      std::vector<std::uint8_t>(kM * kPackedOutputRowBytes),
      std::vector<std::uint16_t>(kM * kOutputGroups)};
  for (std::size_t m = 0U; m < kM; ++m) {
    for (std::size_t group = 0U; group < kOutputGroups; ++group) {
      float maximum = 0.0F;
      for (std::size_t inner = 0U; inner < 64U; ++inner) {
        maximum = std::fmax(
            maximum, std::fabs(products[m * kN + group * 64U + inner]));
      }
      const float clipped_maximum = maximum * kOutputClipRatio;
      const float scale =
          clipped_maximum > 0.0F ? clipped_maximum / 7.0F : 1.0F;
      reference.scales[m * kOutputGroups + group] = encode_bf16(scale);
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const float even_value = std::max(
            -clipped_maximum,
            std::min(clipped_maximum,
                     products[m * kN + group * 64U + 2U * byte]));
        const float odd_value = std::max(
            -clipped_maximum,
            std::min(clipped_maximum,
                     products[m * kN + group * 64U + 2U * byte + 1U]));
        reference.packed[m * kPackedOutputRowBytes + group * 32U + byte] =
            kernels::sm87_a4w4_pack_signed_pair(
                round_and_clamp(even_value / scale),
                round_and_clamp(odd_value / scale));
      }
    }
  }
  return reference;
}

[[nodiscard]] bool run_correctness() {
  const HostPayload payload = make_payload();
  const std::vector<float> products = reference_products(payload);
  const QuantizedReference reference = quantize_reference(products);

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint8_t> device_gate_b;
  DeviceBuffer<std::uint16_t> device_gate_scales;
  DeviceBuffer<std::uint8_t> device_up_b;
  DeviceBuffer<std::uint16_t> device_up_scales;
  DeviceBuffer<std::uint8_t> device_output;
  DeviceBuffer<std::uint16_t> device_output_scales;
  const std::size_t output_rows_with_guard = kM + 1U;
  if (!device_a.allocate(payload.a.size()) ||
      !device_a_scales.allocate(payload.a_scales.size()) ||
      !device_gate_b.allocate(payload.gate_b.size()) ||
      !device_gate_scales.allocate(payload.gate_scales.size()) ||
      !device_up_b.allocate(payload.up_b.size()) ||
      !device_up_scales.allocate(payload.up_scales.size()) ||
      !device_output.allocate(output_rows_with_guard * kOutputPackedStride) ||
      !device_output_scales.allocate(output_rows_with_guard *
                                     kOutputScaleStride)) {
    std::cerr << "device allocation failed\n";
    return false;
  }

  if (!cuda_ok(cudaMemcpy(device_a.get(), payload.a.data(), payload.a.size(),
                          cudaMemcpyHostToDevice),
               "copy A") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), payload.a_scales.data(),
                          payload.a_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy A scales") ||
      !cuda_ok(cudaMemcpy(device_gate_b.get(), payload.gate_b.data(),
                          payload.gate_b.size(), cudaMemcpyHostToDevice),
               "copy Gate B") ||
      !cuda_ok(cudaMemcpy(device_gate_scales.get(),
                          payload.gate_scales.data(),
                          payload.gate_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy Gate scales") ||
      !cuda_ok(cudaMemcpy(device_up_b.get(), payload.up_b.data(),
                          payload.up_b.size(), cudaMemcpyHostToDevice),
               "copy Up B") ||
      !cuda_ok(cudaMemcpy(device_up_scales.get(), payload.up_scales.data(),
                          payload.up_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy Up scales") ||
      !cuda_ok(cudaMemset(device_output.get(), 0xa5,
                          output_rows_with_guard * kOutputPackedStride),
               "initialize packed output guard") ||
      !cuda_ok(cudaMemset(device_output_scales.get(), 0xad,
                          output_rows_with_guard * kOutputScaleStride *
                              sizeof(std::uint16_t)),
               "initialize scale output guard")) {
    return false;
  }

  kernels::Sm87A4W4GateUpPairedResources resources{};
  if (!launch_ok(kernels::query_sm87_a4w4_gateup_paired_resources_cuda(
                     &resources),
                 "query paired Gate+Up resources")) {
    return false;
  }
  if (resources.local_bytes != 0U || resources.active_blocks_per_sm < 2 ||
      resources.static_shared_bytes != 35'968U ||
      resources.maximum_threads_per_block < 256) {
    std::cerr << "resource contract failed: registers="
              << resources.registers_per_thread
              << " shared=" << resources.static_shared_bytes
              << " local=" << resources.local_bytes
              << " active_blocks=" << resources.active_blocks_per_sm
              << '\n';
    return false;
  }

  if (!launch_ok(kernels::launch_sm87_a4w4_gateup_paired_cuda(
                     device_a.get(), kPackedInputRowBytes,
                     device_a_scales.get(), kInputScaleStride,
                     device_gate_b.get(), kPackedInputRowBytes,
                     device_gate_scales.get(), kWeightScaleStride,
                     device_up_b.get(), kPackedInputRowBytes,
                     device_up_scales.get(), kWeightScaleStride, kM, kN, kK,
                     kOutputClipRatio, device_output.get(),
                     kOutputPackedStride, device_output_scales.get(),
                     kOutputScaleStride),
                 "launch paired Gate+Up") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize paired Gate+Up")) {
    return false;
  }

  std::vector<std::uint8_t> actual_packed(
      output_rows_with_guard * kOutputPackedStride);
  std::vector<std::uint16_t> actual_scales(
      output_rows_with_guard * kOutputScaleStride);
  if (!cuda_ok(cudaMemcpy(actual_packed.data(), device_output.get(),
                          actual_packed.size(), cudaMemcpyDeviceToHost),
               "copy paired output") ||
      !cuda_ok(cudaMemcpy(actual_scales.data(), device_output_scales.get(),
                          actual_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy paired scales")) {
    return false;
  }

  std::size_t code_mismatches = 0U;
  std::size_t scale_mismatches = 0U;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  double maximum_scale_relative_error = 0.0;
  for (std::size_t m = 0U; m < kM; ++m) {
    for (std::size_t group = 0U; group < kOutputGroups; ++group) {
      const float expected_scale =
          decode_bf16(reference.scales[m * kOutputGroups + group]);
      const float actual_scale = decode_bf16(
          actual_scales[m * kOutputScaleStride + group]);
      if (actual_scales[m * kOutputScaleStride + group] !=
          reference.scales[m * kOutputGroups + group]) {
        ++scale_mismatches;
      }
      maximum_scale_relative_error = std::fmax(
          maximum_scale_relative_error,
          std::fabs(static_cast<double>(actual_scale - expected_scale)) /
              std::max(1.0e-12, std::fabs(static_cast<double>(expected_scale))));

      for (std::size_t inner = 0U; inner < 64U; ++inner) {
        const std::size_t n = group * 64U + inner;
        const int expected_code = kernels::sm87_a4w4_unpack_signed(
            reference.packed[m * kPackedOutputRowBytes + n / 2U], n);
        const int actual_code = kernels::sm87_a4w4_unpack_signed(
            actual_packed[m * kOutputPackedStride + n / 2U], n);
        if (actual_code != expected_code) {
          ++code_mismatches;
        }
        const double expected_value =
            static_cast<double>(expected_code) * expected_scale;
        const double actual_value =
            static_cast<double>(actual_code) * actual_scale;
        const double error = actual_value - expected_value;
        squared_error += error * error;
        squared_reference += expected_value * expected_value;
      }
    }
    for (std::size_t byte = kPackedOutputRowBytes;
         byte < kOutputPackedStride; ++byte) {
      if (actual_packed[m * kOutputPackedStride + byte] != 0xa5U) {
        std::cerr << "packed row padding was overwritten at m=" << m
                  << " byte=" << byte << '\n';
        return false;
      }
    }
    for (std::size_t group = kOutputGroups;
         group < kOutputScaleStride; ++group) {
      if (actual_scales[m * kOutputScaleStride + group] != 0xadadU) {
        std::cerr << "scale row padding was overwritten at m=" << m
                  << " group=" << group << '\n';
        return false;
      }
    }
  }
  for (std::size_t byte = 0U; byte < kOutputPackedStride; ++byte) {
    if (actual_packed[kM * kOutputPackedStride + byte] != 0xa5U) {
      std::cerr << "M-tail packed guard row was overwritten\n";
      return false;
    }
  }
  for (std::size_t group = 0U; group < kOutputScaleStride; ++group) {
    if (actual_scales[kM * kOutputScaleStride + group] != 0xadadU) {
      std::cerr << "M-tail scale guard row was overwritten\n";
      return false;
    }
  }

  const double nrmse =
      std::sqrt(squared_error / std::max(1.0e-30, squared_reference));
  // CPU and device exp implementations may place a value on opposite sides
  // of an A4 rounding boundary.  Preserve a tight reconstructed-value gate
  // while reporting the bitwise counts for the real-device admission log.
  if (maximum_scale_relative_error > 0.01 || nrmse > 0.02 ||
      code_mismatches > kM * kN / 100U ||
      scale_mismatches > kM * kOutputGroups / 100U) {
    std::cerr << "paired Gate+Up mismatch: code=" << code_mismatches
              << " scale=" << scale_mismatches
              << " scale_rel=" << maximum_scale_relative_error
              << " nrmse=" << nrmse << '\n';
    return false;
  }

  std::cout << "SM87 A4W4 paired Gate+Up correctness passed: M=" << kM
            << " N=" << kN << " K=" << kK
            << " code_mismatches=" << code_mismatches
            << " scale_mismatches=" << scale_mismatches
            << " nrmse=" << nrmse
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
  return run_correctness() ? 0 : 1;
}
