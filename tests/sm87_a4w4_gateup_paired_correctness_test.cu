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

inline constexpr std::size_t kM = 2'048U;
inline constexpr std::size_t kN = 128U;
inline constexpr std::size_t kK = 256U;
inline constexpr std::size_t kGroups = kK / 64U;
inline constexpr std::size_t kOutputGroups = kN / 64U;
inline constexpr std::size_t kPackedInputRowBytes = kK / 2U;
inline constexpr std::size_t kAPackedBytes =
    kernels::sm87_a4w4_consumer_packed_capacity_bytes(kM, kK);
inline constexpr std::size_t kAScaleElements =
    kernels::sm87_a4w4_consumer_scale_capacity_elements(kM, kK);
inline constexpr std::size_t kBPackedBytes =
    kernels::sm87_a4w4_consumer_packed_capacity_bytes(kN, kK);
inline constexpr std::size_t kBScaleElements =
    kernels::sm87_a4w4_consumer_scale_capacity_elements(kN, kK);
inline constexpr std::size_t kOutputPackedBytes =
    kernels::sm87_a4w4_consumer_packed_capacity_bytes(kM, kN);
inline constexpr std::size_t kOutputScaleElements =
    kernels::sm87_a4w4_consumer_scale_capacity_elements(kM, kN);
inline constexpr std::size_t kPackedGuardBytes = 64U;
inline constexpr std::size_t kScaleGuardElements = 64U;
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
                          const std::size_t outer,
                          const std::size_t inner) noexcept {
  const std::size_t group = inner / 64U;
  const std::size_t byte_in_group = (inner % 64U) / 2U;
  return kernels::sm87_a4w4_unpack_signed(
      packed[kernels::sm87_a4w4_consumer_packed_offset(
          outer, group, byte_in_group, kGroups)],
      inner);
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
      std::vector<std::uint8_t>(kAPackedBytes),
      std::vector<std::uint16_t>(kAScaleElements,
                                 static_cast<std::uint16_t>(0x7fc1U)),
      std::vector<std::uint8_t>(kBPackedBytes),
      std::vector<std::uint16_t>(kBScaleElements,
                                 static_cast<std::uint16_t>(0x7fc1U)),
      std::vector<std::uint8_t>(kBPackedBytes),
      std::vector<std::uint16_t>(kBScaleElements,
                                 static_cast<std::uint16_t>(0x7fc1U))};

  for (std::size_t m = 0U; m < kM; ++m) {
    for (std::size_t byte = 0U; byte < kPackedInputRowBytes; ++byte) {
      const int even =
          static_cast<int>((19U * m + 5U * (2U * byte) + 3U) % 15U) - 7;
      const int odd = static_cast<int>(
                          (11U * m + 7U * (2U * byte + 1U) + 1U) % 15U) -
                      7;
      const std::size_t group = byte / 32U;
      const std::size_t byte_in_group = byte % 32U;
      payload.a[kernels::sm87_a4w4_consumer_packed_offset(
          m, group, byte_in_group, kGroups)] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
    for (std::size_t group = 0U; group < kGroups; ++group) {
      payload.a_scales[kernels::sm87_a4w4_consumer_scale_offset(
          m, group, kGroups)] = encode_bf16(
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
      const std::size_t group = byte / 32U;
      const std::size_t byte_in_group = byte % 32U;
      payload.gate_b[kernels::sm87_a4w4_consumer_packed_offset(
          n, group, byte_in_group, kGroups)] =
          kernels::sm87_a4w4_pack_signed_pair(gate_even, gate_odd);
      payload.up_b[kernels::sm87_a4w4_consumer_packed_offset(
          n, group, byte_in_group, kGroups)] =
          kernels::sm87_a4w4_pack_signed_pair(up_even, up_odd);
    }
    for (std::size_t group = 0U; group < kGroups; ++group) {
      payload.gate_scales[kernels::sm87_a4w4_consumer_scale_offset(
          n, group, kGroups)] = encode_bf16(
          static_cast<float>(1U + ((n + group) % 4U)) / 128.0F);
      payload.up_scales[kernels::sm87_a4w4_consumer_scale_offset(
          n, group, kGroups)] = encode_bf16(
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
          const int a = code_at(payload.a, m, k);
          const int gate_b = code_at(payload.gate_b, n, k);
          const int up_b = code_at(payload.up_b, n, k);
          gate_integer += a * gate_b;
          up_integer += a * up_b;
        }
        const float a_scale =
            decode_bf16(payload.a_scales[
                kernels::sm87_a4w4_consumer_scale_offset(
                    m, group, kGroups)]);
        gate += static_cast<float>(gate_integer) * a_scale *
                decode_bf16(payload.gate_scales[
                    kernels::sm87_a4w4_consumer_scale_offset(
                        n, group, kGroups)]);
        up += static_cast<float>(up_integer) * a_scale *
              decode_bf16(payload.up_scales[
                  kernels::sm87_a4w4_consumer_scale_offset(
                      n, group, kGroups)]);
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
      std::vector<std::uint8_t>(kOutputPackedBytes),
      std::vector<std::uint16_t>(kOutputScaleElements)};
  for (std::size_t m = 0U; m < kM; ++m) {
    for (std::size_t group = 0U; group < kOutputGroups; ++group) {
      float maximum = 0.0F;
      for (std::size_t inner = 0U; inner < 64U; ++inner) {
        maximum = std::fmax(
            maximum, std::fabs(products[m * kN + group * 64U + inner]));
      }
      const float clipped_maximum = maximum * kOutputClipRatio;
      const std::uint16_t scale_bits = encode_bf16(
          maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
      const float stored_scale = decode_bf16(scale_bits);
      reference.scales[kernels::sm87_a4w4_consumer_scale_offset(
          m, group, kOutputGroups)] = scale_bits;
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const float even_value = std::max(
            -clipped_maximum,
            std::min(clipped_maximum,
                     products[m * kN + group * 64U + 2U * byte]));
        const float odd_value = std::max(
            -clipped_maximum,
            std::min(clipped_maximum,
                     products[m * kN + group * 64U + 2U * byte + 1U]));
        reference.packed[kernels::sm87_a4w4_consumer_packed_offset(
            m, group, byte, kOutputGroups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                stored_scale == 0.0F
                    ? 0
                    : round_and_clamp(even_value / stored_scale),
                stored_scale == 0.0F
                    ? 0
                    : round_and_clamp(odd_value / stored_scale));
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
  if (!device_a.allocate(payload.a.size()) ||
      !device_a_scales.allocate(payload.a_scales.size()) ||
      !device_gate_b.allocate(payload.gate_b.size()) ||
      !device_gate_scales.allocate(payload.gate_scales.size()) ||
      !device_up_b.allocate(payload.up_b.size()) ||
      !device_up_scales.allocate(payload.up_scales.size()) ||
      !device_output.allocate(kOutputPackedBytes + kPackedGuardBytes) ||
      !device_output_scales.allocate(kOutputScaleElements +
                                     kScaleGuardElements)) {
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
                          kOutputPackedBytes + kPackedGuardBytes),
               "initialize packed output guard") ||
      !cuda_ok(cudaMemset(device_output_scales.get(), 0xad,
                          (kOutputScaleElements + kScaleGuardElements) *
                              sizeof(std::uint16_t)),
               "initialize scale output guard")) {
    return false;
  }

  kernels::Sm87A4W4GateUpPairedResources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_gateup_paired_wide_large_m_resources_cuda(
              &resources),
          "query paired Gate+Up wide-large-M resources")) {
    return false;
  }
  if (resources.local_bytes != 0U || resources.active_blocks_per_sm < 2 ||
      resources.static_shared_bytes != 43'520U ||
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
                     device_a.get(), kAPackedBytes,
                     device_a_scales.get(), kAScaleElements,
                     device_gate_b.get(), kBPackedBytes,
                     device_gate_scales.get(), kBScaleElements,
                     device_up_b.get(), kBPackedBytes,
                     device_up_scales.get(), kBScaleElements, kM, kN, kK,
                     kOutputClipRatio, device_output.get(),
                     kOutputPackedBytes, device_output_scales.get(),
                     kOutputScaleElements),
                 "launch paired Gate+Up") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize paired Gate+Up")) {
    return false;
  }

  std::vector<std::uint8_t> actual_packed(
      kOutputPackedBytes + kPackedGuardBytes);
  std::vector<std::uint16_t> actual_scales(
      kOutputScaleElements + kScaleGuardElements);
  if (!cuda_ok(cudaMemcpy(actual_packed.data(), device_output.get(),
                          actual_packed.size(), cudaMemcpyDeviceToHost),
               "copy paired output") ||
      !cuda_ok(cudaMemcpy(actual_scales.data(), device_output_scales.get(),
                          actual_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy paired scales")) {
    return false;
  }

  // Establish bit-level equivalence against the retained M64N64 kernel, not
  // merely against a tolerant CPU reconstruction.  Four P1024/N64 launches
  // cover the same P2048/N128 tensor, then the host stitches their unchanged
  // consumer-order blocks into the candidate layout.
  constexpr std::size_t kBaselineM = 1'024U;
  constexpr std::size_t kBaselineN = 64U;
  constexpr std::size_t kBaselineAPackedBytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kBaselineM, kK);
  constexpr std::size_t kBaselineAScaleElements =
      kernels::sm87_a4w4_consumer_scale_capacity_elements(kBaselineM, kK);
  constexpr std::size_t kBaselineBPackedBytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kBaselineN, kK);
  constexpr std::size_t kBaselineBScaleElements =
      kernels::sm87_a4w4_consumer_scale_capacity_elements(kBaselineN, kK);
  constexpr std::size_t kBaselineOutputPackedBytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          kBaselineM, kBaselineN);
  constexpr std::size_t kBaselineOutputScaleElements =
      kernels::sm87_a4w4_consumer_scale_capacity_elements(
          kBaselineM, kBaselineN);
  DeviceBuffer<std::uint8_t> baseline_device_output;
  DeviceBuffer<std::uint16_t> baseline_device_scales;
  if (!baseline_device_output.allocate(kBaselineOutputPackedBytes) ||
      !baseline_device_scales.allocate(kBaselineOutputScaleElements)) {
    std::cerr << "baseline device allocation failed\n";
    return false;
  }
  std::vector<std::uint8_t> baseline_packed(kOutputPackedBytes);
  std::vector<std::uint16_t> baseline_scales(kOutputScaleElements);
  std::vector<std::uint8_t> baseline_chunk_packed(
      kBaselineOutputPackedBytes);
  std::vector<std::uint16_t> baseline_chunk_scales(
      kBaselineOutputScaleElements);
  for (std::size_t m_chunk = 0U; m_chunk < kM / kBaselineM; ++m_chunk) {
    for (std::size_t n_chunk = 0U; n_chunk < kN / kBaselineN; ++n_chunk) {
      if (!launch_ok(kernels::launch_sm87_a4w4_gateup_paired_cuda(
                         device_a.get() + m_chunk * kBaselineAPackedBytes,
                         kBaselineAPackedBytes,
                         device_a_scales.get() +
                             m_chunk * kBaselineAScaleElements,
                         kBaselineAScaleElements,
                         device_gate_b.get() +
                             n_chunk * kBaselineBPackedBytes,
                         kBaselineBPackedBytes,
                         device_gate_scales.get() +
                             n_chunk * kBaselineBScaleElements,
                         kBaselineBScaleElements,
                         device_up_b.get() +
                             n_chunk * kBaselineBPackedBytes,
                         kBaselineBPackedBytes,
                         device_up_scales.get() +
                             n_chunk * kBaselineBScaleElements,
                         kBaselineBScaleElements, kBaselineM, kBaselineN,
                         kK, kOutputClipRatio, baseline_device_output.get(),
                         kBaselineOutputPackedBytes,
                         baseline_device_scales.get(),
                         kBaselineOutputScaleElements),
                     "launch retained M64N64 baseline") ||
          !cuda_ok(cudaDeviceSynchronize(),
                   "synchronize retained M64N64 baseline") ||
          !cuda_ok(cudaMemcpy(baseline_chunk_packed.data(),
                              baseline_device_output.get(),
                              kBaselineOutputPackedBytes,
                              cudaMemcpyDeviceToHost),
                   "copy retained M64N64 packed output") ||
          !cuda_ok(cudaMemcpy(baseline_chunk_scales.data(),
                              baseline_device_scales.get(),
                              kBaselineOutputScaleElements *
                                  sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "copy retained M64N64 scale output")) {
        return false;
      }
      for (std::size_t local_m = 0U; local_m < kBaselineM; ++local_m) {
        const std::size_t global_m = m_chunk * kBaselineM + local_m;
        for (std::size_t byte = 0U; byte < 32U; ++byte) {
          baseline_packed[kernels::sm87_a4w4_consumer_packed_offset(
              global_m, n_chunk, byte, kOutputGroups)] =
              baseline_chunk_packed[
                  kernels::sm87_a4w4_consumer_packed_offset(
                      local_m, 0U, byte, 1U)];
        }
        baseline_scales[kernels::sm87_a4w4_consumer_scale_offset(
            global_m, n_chunk, kOutputGroups)] =
            baseline_chunk_scales[
                kernels::sm87_a4w4_consumer_scale_offset(
                    local_m, 0U, 1U)];
      }
    }
  }
  std::size_t baseline_code_byte_mismatches = 0U;
  for (std::size_t index = 0U; index < baseline_packed.size(); ++index) {
    baseline_code_byte_mismatches +=
        baseline_packed[index] != actual_packed[index] ? 1U : 0U;
  }
  std::size_t baseline_scale_mismatches = 0U;
  for (std::size_t index = 0U; index < baseline_scales.size(); ++index) {
    baseline_scale_mismatches +=
        baseline_scales[index] != actual_scales[index] ? 1U : 0U;
  }
  if (baseline_code_byte_mismatches != 0U ||
      baseline_scale_mismatches != 0U) {
    std::cerr << "wide-large-M is not bit-exact with M64N64: packed="
              << baseline_code_byte_mismatches
              << " scales=" << baseline_scale_mismatches << '\n';
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
          decode_bf16(reference.scales[
              kernels::sm87_a4w4_consumer_scale_offset(
                  m, group, kOutputGroups)]);
      const float actual_scale = decode_bf16(
          actual_scales[kernels::sm87_a4w4_consumer_scale_offset(
              m, group, kOutputGroups)]);
      if (actual_scales[kernels::sm87_a4w4_consumer_scale_offset(
              m, group, kOutputGroups)] !=
          reference.scales[kernels::sm87_a4w4_consumer_scale_offset(
              m, group, kOutputGroups)]) {
        ++scale_mismatches;
      }
      maximum_scale_relative_error = std::fmax(
          maximum_scale_relative_error,
          std::fabs(static_cast<double>(actual_scale - expected_scale)) /
              std::max(1.0e-12, std::fabs(static_cast<double>(expected_scale))));

      for (std::size_t inner = 0U; inner < 64U; ++inner) {
        const std::size_t n = group * 64U + inner;
        const int expected_code = kernels::sm87_a4w4_unpack_signed(
            reference.packed[kernels::sm87_a4w4_consumer_packed_offset(
                m, group, inner / 2U, kOutputGroups)],
            n);
        const int actual_code = kernels::sm87_a4w4_unpack_signed(
            actual_packed[kernels::sm87_a4w4_consumer_packed_offset(
                m, group, inner / 2U, kOutputGroups)],
            n);
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
  }
  const std::size_t padded_m_end =
      kernels::sm87_a4w4_consumer_outer_block_count(kM) * 64U;
  for (std::size_t m = kM; m < padded_m_end; ++m) {
    for (std::size_t group = 0U; group < kOutputGroups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        if (actual_packed[kernels::sm87_a4w4_consumer_packed_offset(
                m, group, byte, kOutputGroups)] != 0xa5U) {
          std::cerr << "M-tail packed padding was overwritten at m=" << m
                    << " group=" << group << " byte=" << byte << '\n';
          return false;
        }
      }
      if (actual_scales[kernels::sm87_a4w4_consumer_scale_offset(
              m, group, kOutputGroups)] != 0xadadU) {
        std::cerr << "M-tail scale padding was overwritten at m=" << m
                  << " group=" << group << '\n';
        return false;
      }
    }
  }
  for (std::size_t byte = kOutputPackedBytes;
       byte < actual_packed.size(); ++byte) {
    if (actual_packed[byte] != 0xa5U) {
      std::cerr << "packed allocation guard was overwritten\n";
      return false;
    }
  }
  for (std::size_t element = kOutputScaleElements;
       element < actual_scales.size(); ++element) {
    if (actual_scales[element] != 0xadadU) {
      std::cerr << "scale allocation guard was overwritten\n";
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
            << " retained_kernel_bit_mismatches=0"
            << " nrmse=" << nrmse
            << " registers=" << resources.registers_per_thread
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return true;
}

[[nodiscard]] bool run_arbitrary_m_composition_case(
    const std::size_t m_count) {
  constexpr std::size_t n_count = 128U;
  constexpr std::size_t k_count = 64U;
  constexpr std::size_t k64_groups = 1U;
  constexpr std::size_t output_k64_groups = n_count / 64U;
  constexpr std::size_t packed_guard_bytes = 64U;
  constexpr std::size_t scale_guard_elements = 64U;
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
  const std::size_t output_packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          m_count, n_count);
  const std::size_t output_scale_elements =
      kernels::sm87_a4w4_consumer_scale_capacity_elements(
          m_count, n_count);

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
  std::vector<std::uint8_t> gate_b(b_packed_bytes, 0U);
  std::vector<std::uint16_t> gate_scales(b_scale_elements, 0U);
  std::vector<std::uint8_t> up_b(b_packed_bytes, 0U);
  std::vector<std::uint16_t> up_scales(b_scale_elements, 0U);
  for (std::size_t n = 0U; n < n_count; ++n) {
    for (std::size_t byte = 0U; byte < 32U; ++byte) {
      const int gate_even =
          static_cast<int>((13U * n + 2U * byte + 3U) % 15U) - 7;
      const int gate_odd =
          static_cast<int>((3U * n + 7U * byte + 4U) % 15U) - 7;
      const int up_even =
          static_cast<int>((5U * n + 11U * byte + 2U) % 15U) - 7;
      const int up_odd =
          static_cast<int>((9U * n + 3U * byte + 1U) % 15U) - 7;
      gate_b[kernels::sm87_a4w4_consumer_packed_offset(
          n, 0U, byte, k64_groups)] =
          kernels::sm87_a4w4_pack_signed_pair(gate_even, gate_odd);
      up_b[kernels::sm87_a4w4_consumer_packed_offset(
          n, 0U, byte, k64_groups)] =
          kernels::sm87_a4w4_pack_signed_pair(up_even, up_odd);
    }
    gate_scales[kernels::sm87_a4w4_consumer_scale_offset(
        n, 0U, k64_groups)] =
        encode_bf16(static_cast<float>(1U + n % 3U) / 64.0F);
    up_scales[kernels::sm87_a4w4_consumer_scale_offset(
        n, 0U, k64_groups)] =
        encode_bf16(static_cast<float>(1U + n % 7U) / 128.0F);
  }

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint8_t> device_gate_b;
  DeviceBuffer<std::uint16_t> device_gate_scales;
  DeviceBuffer<std::uint8_t> device_up_b;
  DeviceBuffer<std::uint16_t> device_up_scales;
  DeviceBuffer<std::uint8_t> candidate_output;
  DeviceBuffer<std::uint16_t> candidate_output_scales;
  DeviceBuffer<std::uint8_t> baseline_output;
  DeviceBuffer<std::uint16_t> baseline_output_scales;
  if (!device_a.allocate(a_packed_bytes) ||
      !device_a_scales.allocate(a_scale_elements) ||
      !device_gate_b.allocate(b_packed_bytes) ||
      !device_gate_scales.allocate(b_scale_elements) ||
      !device_up_b.allocate(b_packed_bytes) ||
      !device_up_scales.allocate(b_scale_elements) ||
      !candidate_output.allocate(output_packed_bytes + packed_guard_bytes) ||
      !candidate_output_scales.allocate(output_scale_elements +
                                        scale_guard_elements) ||
      !baseline_output.allocate(output_packed_bytes + packed_guard_bytes) ||
      !baseline_output_scales.allocate(output_scale_elements +
                                       scale_guard_elements)) {
    std::cerr << "arbitrary-M paired allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_a.get(), packed_a.data(),
                          a_packed_bytes, cudaMemcpyHostToDevice),
               "copy arbitrary-M paired A") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), a_scales.data(),
                          a_scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy arbitrary-M paired A scales") ||
      !cuda_ok(cudaMemcpy(device_gate_b.get(), gate_b.data(),
                          b_packed_bytes, cudaMemcpyHostToDevice),
               "copy arbitrary-M paired Gate B") ||
      !cuda_ok(cudaMemcpy(device_gate_scales.get(), gate_scales.data(),
                          b_scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy arbitrary-M paired Gate scales") ||
      !cuda_ok(cudaMemcpy(device_up_b.get(), up_b.data(),
                          b_packed_bytes, cudaMemcpyHostToDevice),
               "copy arbitrary-M paired Up B") ||
      !cuda_ok(cudaMemcpy(device_up_scales.get(), up_scales.data(),
                          b_scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy arbitrary-M paired Up scales") ||
      !cuda_ok(cudaMemset(candidate_output.get(), 0xa5,
                          output_packed_bytes + packed_guard_bytes),
               "initialize arbitrary-M paired candidate output") ||
      !cuda_ok(cudaMemset(candidate_output_scales.get(), 0xad,
                          (output_scale_elements + scale_guard_elements) *
                              sizeof(std::uint16_t)),
               "initialize arbitrary-M paired candidate scales") ||
      !cuda_ok(cudaMemset(baseline_output.get(), 0xa5,
                          output_packed_bytes + packed_guard_bytes),
               "initialize arbitrary-M paired baseline output") ||
      !cuda_ok(cudaMemset(baseline_output_scales.get(), 0xad,
                          (output_scale_elements + scale_guard_elements) *
                              sizeof(std::uint16_t)),
               "initialize arbitrary-M paired baseline scales")) {
    return false;
  }

  if (m_count == 3'987U &&
      kernels::launch_sm87_a4w4_gateup_paired_cuda(
          device_a.get(), a_packed_bytes, device_a_scales.get(),
          a_scale_elements, device_gate_b.get(), b_packed_bytes,
          device_gate_scales.get(), b_scale_elements, device_up_b.get(),
          b_packed_bytes, device_up_scales.get(), b_scale_elements,
          m_count, n_count, k_count, kOutputClipRatio,
          candidate_output.get(), output_packed_bytes - 1U,
          candidate_output_scales.get(), output_scale_elements) !=
          static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "arbitrary-M paired short output capacity was not rejected\n";
    return false;
  }
  if (!launch_ok(kernels::launch_sm87_a4w4_gateup_paired_cuda(
                     device_a.get(), a_packed_bytes,
                     device_a_scales.get(), a_scale_elements,
                     device_gate_b.get(), b_packed_bytes,
                     device_gate_scales.get(), b_scale_elements,
                     device_up_b.get(), b_packed_bytes,
                     device_up_scales.get(), b_scale_elements,
                     m_count, n_count, k_count, kOutputClipRatio,
                     candidate_output.get(), output_packed_bytes,
                     candidate_output_scales.get(), output_scale_elements),
                 "launch arbitrary-M paired candidate")) {
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
    const std::size_t output_byte_offset = first_m * (n_count / 2U);
    const std::size_t output_scale_offset =
        first_m * output_k64_groups;
    const std::size_t chunk_a_bytes =
        kernels::sm87_a4w4_consumer_packed_capacity_bytes(
            chunk_m, k_count);
    const std::size_t chunk_a_scales =
        kernels::sm87_a4w4_consumer_scale_capacity_elements(
            chunk_m, k_count);
    const std::size_t chunk_output_bytes =
        kernels::sm87_a4w4_consumer_packed_capacity_bytes(
            chunk_m, n_count);
    const std::size_t chunk_output_scales =
        kernels::sm87_a4w4_consumer_scale_capacity_elements(
            chunk_m, n_count);
    if (!launch_ok(kernels::launch_sm87_a4w4_gateup_paired_cuda(
                       device_a.get() + a_byte_offset, chunk_a_bytes,
                       device_a_scales.get() + a_scale_offset,
                       chunk_a_scales, device_gate_b.get(), b_packed_bytes,
                       device_gate_scales.get(), b_scale_elements,
                       device_up_b.get(), b_packed_bytes,
                       device_up_scales.get(), b_scale_elements,
                       chunk_m, n_count, k_count, kOutputClipRatio,
                       baseline_output.get() + output_byte_offset,
                       chunk_output_bytes,
                       baseline_output_scales.get() + output_scale_offset,
                       chunk_output_scales),
                   "launch arbitrary-M paired retained baseline")) {
      return false;
    }
    first_m += chunk_m;
  }
  if (!cuda_ok(cudaDeviceSynchronize(),
               "synchronize arbitrary-M paired comparison")) {
    return false;
  }

  std::vector<std::uint8_t> candidate_packed(
      output_packed_bytes + packed_guard_bytes);
  std::vector<std::uint8_t> baseline_packed(
      output_packed_bytes + packed_guard_bytes);
  std::vector<std::uint16_t> candidate_scales(
      output_scale_elements + scale_guard_elements);
  std::vector<std::uint16_t> baseline_scales(
      output_scale_elements + scale_guard_elements);
  if (!cuda_ok(cudaMemcpy(candidate_packed.data(), candidate_output.get(),
                          candidate_packed.size(), cudaMemcpyDeviceToHost),
               "copy arbitrary-M paired candidate output") ||
      !cuda_ok(cudaMemcpy(baseline_packed.data(), baseline_output.get(),
                          baseline_packed.size(), cudaMemcpyDeviceToHost),
               "copy arbitrary-M paired baseline output") ||
      !cuda_ok(cudaMemcpy(candidate_scales.data(),
                          candidate_output_scales.get(),
                          candidate_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy arbitrary-M paired candidate scales") ||
      !cuda_ok(cudaMemcpy(baseline_scales.data(),
                          baseline_output_scales.get(),
                          baseline_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy arbitrary-M paired baseline scales")) {
    return false;
  }
  for (std::size_t index = 0U; index < candidate_packed.size(); ++index) {
    if (candidate_packed[index] != baseline_packed[index]) {
      std::cerr << "arbitrary-M paired packed bit mismatch: M=" << m_count
                << " byte=" << index << '\n';
      return false;
    }
  }
  for (std::size_t index = 0U; index < candidate_scales.size(); ++index) {
    if (candidate_scales[index] != baseline_scales[index]) {
      std::cerr << "arbitrary-M paired scale bit mismatch: M=" << m_count
                << " element=" << index << '\n';
      return false;
    }
  }
  std::cout << "SM87 A4W4 paired arbitrary-M composition bit-exact: M="
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

[[nodiscard]] bool run_shared_k128_correctness() {
  constexpr std::size_t m_count = 64U;
  constexpr std::size_t n_count = 128U;
  constexpr std::size_t k_count = 128U;
  constexpr std::size_t physical_k64_groups = k_count / 64U;
  constexpr std::size_t k128_groups = k_count / 128U;
  constexpr std::size_t output_physical_groups = n_count / 64U;
  constexpr std::size_t output_k128_groups = n_count / 128U;
  constexpr std::size_t a_packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count, k_count);
  constexpr std::size_t a_scale_elements =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(
          m_count, k_count);
  constexpr std::size_t b_packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(n_count, k_count);
  constexpr std::size_t b_scale_elements =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(
          n_count, k_count);
  constexpr std::size_t output_packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count, n_count);
  constexpr std::size_t output_scale_elements =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(
          m_count, n_count);
  static_assert(kernels::sm87_a4w4_gateup_paired_k128_plan(
                    m_count, n_count, k_count)
                    .work_tiles == 1U);
  static_assert(physical_k64_groups == 2U && k128_groups == 1U &&
                output_physical_groups == 2U &&
                output_k128_groups == 1U);

  std::vector<std::uint8_t> a(a_packed_bytes);
  std::vector<std::uint16_t> a_scales(
      a_scale_elements, encode_bf16(0.25F));
  std::vector<std::uint8_t> gate_b(b_packed_bytes);
  std::vector<std::uint16_t> gate_scales(
      b_scale_elements, encode_bf16(0.25F));
  std::vector<std::uint8_t> up_b(b_packed_bytes);
  std::vector<std::uint16_t> up_scales(
      b_scale_elements, encode_bf16(1.0F / 64.0F));

  // Gate is deliberately large and positive, making SiLU's denominator
  // exactly one in both host and device FP32.  This removes libm variance
  // from a bit-exact CPU oracle while the nontrivial Up pattern still covers
  // both physical K64 halves, every N row, and signed accumulation.
  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t group = 0U; group < physical_k64_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        a[kernels::sm87_a4w4_consumer_packed_offset(
            m, group, byte, physical_k64_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(7, 7);
      }
    }
  }
  for (std::size_t n = 0U; n < n_count; ++n) {
    for (std::size_t group = 0U; group < physical_k64_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        gate_b[kernels::sm87_a4w4_consumer_packed_offset(
            n, group, byte, physical_k64_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(7, 7);
        const int even =
            static_cast<int>((n + 3U * group + 2U * byte) % 15U) - 7;
        const int odd =
            static_cast<int>((5U * n + group + 2U * byte + 1U) % 15U) -
            7;
        up_b[kernels::sm87_a4w4_consumer_packed_offset(
            n, group, byte, physical_k64_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(even, odd);
      }
    }
  }

  std::vector<float> products(m_count * n_count);
  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t n = 0U; n < n_count; ++n) {
      std::int32_t gate_integer = 0;
      std::int32_t up_integer = 0;
      for (std::size_t k = 0U; k < k_count; ++k) {
        const std::size_t group = k / 64U;
        const std::size_t byte = (k % 64U) / 2U;
        const int a_code = kernels::sm87_a4w4_unpack_signed(
            a[kernels::sm87_a4w4_consumer_packed_offset(
                m, group, byte, physical_k64_groups)],
            k);
        const int gate_code = kernels::sm87_a4w4_unpack_signed(
            gate_b[kernels::sm87_a4w4_consumer_packed_offset(
                n, group, byte, physical_k64_groups)],
            k);
        const int up_code = kernels::sm87_a4w4_unpack_signed(
            up_b[kernels::sm87_a4w4_consumer_packed_offset(
                n, group, byte, physical_k64_groups)],
            k);
        gate_integer += a_code * gate_code;
        up_integer += a_code * up_code;
      }
      const float a_scale = decode_bf16(
          a_scales[kernels::sm87_a4w4_consumer_k128_scale_offset(
              m, 0U, k128_groups)]);
      const float gate = static_cast<float>(gate_integer) *
                         (a_scale * decode_bf16(
                              gate_scales[
                                  kernels::sm87_a4w4_consumer_k128_scale_offset(
                                      n, 0U, k128_groups)]));
      const float up = static_cast<float>(up_integer) *
                       (a_scale * decode_bf16(
                            up_scales[
                                kernels::sm87_a4w4_consumer_k128_scale_offset(
                                    n, 0U, k128_groups)]));
      products[m * n_count + n] = silu_product(gate, up);
    }
  }

  std::vector<std::uint8_t> expected_packed(output_packed_bytes);
  std::vector<std::uint16_t> expected_scales(output_scale_elements);
  for (std::size_t m = 0U; m < m_count; ++m) {
    float maximum = 0.0F;
    for (std::size_t n = 0U; n < n_count; ++n) {
      maximum = std::fmax(
          maximum, std::fabs(products[m * n_count + n]));
    }
    const float clipped_maximum = maximum * kOutputClipRatio;
    const std::uint16_t scale_bits = encode_bf16(
        maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
    const float stored_scale = decode_bf16(scale_bits);
    expected_scales[kernels::sm87_a4w4_consumer_k128_scale_offset(
        m, 0U, output_k128_groups)] = scale_bits;
    for (std::size_t n = 0U; n < n_count; n += 2U) {
      const float even = std::max(
          -clipped_maximum,
          std::min(clipped_maximum, products[m * n_count + n]));
      const float odd = std::max(
          -clipped_maximum,
          std::min(clipped_maximum, products[m * n_count + n + 1U]));
      const std::size_t group = n / 64U;
      const std::size_t byte = (n % 64U) / 2U;
      expected_packed[kernels::sm87_a4w4_consumer_packed_offset(
          m, group, byte, output_physical_groups)] =
          kernels::sm87_a4w4_pack_signed_pair(
              round_and_clamp(even / stored_scale),
              round_and_clamp(odd / stored_scale));
    }
  }

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint8_t> device_gate_b;
  DeviceBuffer<std::uint16_t> device_gate_scales;
  DeviceBuffer<std::uint8_t> device_up_b;
  DeviceBuffer<std::uint16_t> device_up_scales;
  DeviceBuffer<std::uint8_t> device_output;
  DeviceBuffer<std::uint16_t> device_output_scales;
  if (!device_a.allocate(a.size()) ||
      !device_a_scales.allocate(a_scales.size()) ||
      !device_gate_b.allocate(gate_b.size()) ||
      !device_gate_scales.allocate(gate_scales.size()) ||
      !device_up_b.allocate(up_b.size()) ||
      !device_up_scales.allocate(up_scales.size()) ||
      !device_output.allocate(output_packed_bytes + kPackedGuardBytes) ||
      !device_output_scales.allocate(output_scale_elements +
                                     kScaleGuardElements)) {
    std::cerr << "shared-K128 device allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_a.get(), a.data(), a.size(),
                          cudaMemcpyHostToDevice),
               "copy shared-K128 A") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), a_scales.data(),
                          a_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy shared-K128 A scales") ||
      !cuda_ok(cudaMemcpy(device_gate_b.get(), gate_b.data(), gate_b.size(),
                          cudaMemcpyHostToDevice),
               "copy shared-K128 Gate B") ||
      !cuda_ok(cudaMemcpy(device_gate_scales.get(), gate_scales.data(),
                          gate_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy shared-K128 Gate scales") ||
      !cuda_ok(cudaMemcpy(device_up_b.get(), up_b.data(), up_b.size(),
                          cudaMemcpyHostToDevice),
               "copy shared-K128 Up B") ||
      !cuda_ok(cudaMemcpy(device_up_scales.get(), up_scales.data(),
                          up_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy shared-K128 Up scales") ||
      !cuda_ok(cudaMemset(device_output.get(), 0xa5,
                          output_packed_bytes + kPackedGuardBytes),
               "initialize shared-K128 packed guard") ||
      !cuda_ok(cudaMemset(device_output_scales.get(), 0xad,
                          (output_scale_elements + kScaleGuardElements) *
                              sizeof(std::uint16_t)),
               "initialize shared-K128 scale guard")) {
    return false;
  }

  kernels::Sm87A4W4GateUpPairedResources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_gateup_paired_k128_resources_cuda(
              &resources),
          "query shared-K128 paired Gate+Up resources")) {
    return false;
  }
  if (resources.registers_per_thread > 128 ||
      resources.static_shared_bytes != 42'240U ||
      resources.local_bytes != 0U || resources.active_blocks_per_sm < 2 ||
      resources.maximum_threads_per_block < 256) {
    std::cerr << "shared-K128 resource contract failed: registers="
              << resources.registers_per_thread
              << " shared=" << resources.static_shared_bytes
              << " local=" << resources.local_bytes
              << " active_blocks=" << resources.active_blocks_per_sm
              << '\n';
    return false;
  }

  if (kernels::launch_sm87_a4w4_gateup_paired_k128_cuda(
          device_a.get(), a_packed_bytes, device_a_scales.get(),
          a_scale_elements, device_gate_b.get(), b_packed_bytes,
          device_gate_scales.get(), b_scale_elements, device_up_b.get(),
          b_packed_bytes, device_up_scales.get(), b_scale_elements,
          m_count, n_count, k_count, kOutputClipRatio, device_output.get(),
          output_packed_bytes, device_output_scales.get(),
          output_scale_elements - 1U) !=
      static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "shared-K128 short output scale capacity was not rejected\n";
    return false;
  }
  if (!launch_ok(kernels::launch_sm87_a4w4_gateup_paired_k128_cuda(
                     device_a.get(), a_packed_bytes,
                     device_a_scales.get(), a_scale_elements,
                     device_gate_b.get(), b_packed_bytes,
                     device_gate_scales.get(), b_scale_elements,
                     device_up_b.get(), b_packed_bytes,
                     device_up_scales.get(), b_scale_elements, m_count,
                     n_count, k_count, kOutputClipRatio, device_output.get(),
                     output_packed_bytes, device_output_scales.get(),
                     output_scale_elements),
                 "launch shared-K128 paired Gate+Up") ||
      !cuda_ok(cudaDeviceSynchronize(),
               "synchronize shared-K128 paired Gate+Up")) {
    return false;
  }

  std::vector<std::uint8_t> actual_packed(
      output_packed_bytes + kPackedGuardBytes);
  std::vector<std::uint16_t> actual_scales(
      output_scale_elements + kScaleGuardElements);
  if (!cuda_ok(cudaMemcpy(actual_packed.data(), device_output.get(),
                          actual_packed.size(), cudaMemcpyDeviceToHost),
               "copy shared-K128 packed output") ||
      !cuda_ok(cudaMemcpy(actual_scales.data(), device_output_scales.get(),
                          actual_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy shared-K128 output scales")) {
    return false;
  }
  for (std::size_t index = 0U; index < expected_packed.size(); ++index) {
    if (actual_packed[index] != expected_packed[index]) {
      std::cerr << "shared-K128 CPU bit mismatch in packed output at "
                << index << " expected="
                << static_cast<unsigned int>(expected_packed[index])
                << " actual="
                << static_cast<unsigned int>(actual_packed[index]) << '\n';
      return false;
    }
  }
  for (std::size_t index = 0U; index < expected_scales.size(); ++index) {
    if (actual_scales[index] != expected_scales[index]) {
      std::cerr << "shared-K128 CPU bit mismatch in output scale at "
                << index << " expected=" << expected_scales[index]
                << " actual=" << actual_scales[index] << '\n';
      return false;
    }
  }
  for (std::size_t index = output_packed_bytes;
       index < actual_packed.size(); ++index) {
    if (actual_packed[index] != 0xa5U) {
      std::cerr << "shared-K128 packed guard was overwritten\n";
      return false;
    }
  }
  for (std::size_t index = output_scale_elements;
       index < actual_scales.size(); ++index) {
    if (actual_scales[index] != 0xadadU) {
      std::cerr << "shared-K128 scale guard was overwritten\n";
      return false;
    }
  }

  std::cout << "SM87 A4W4 paired Gate+Up shared-K128 CPU bit-exact passed: "
            << "M=" << m_count << " N=" << n_count << " K=" << k_count
            << " registers=" << resources.registers_per_thread
            << " shared=" << resources.static_shared_bytes
            << " local=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return true;
}

}  // namespace

int main() {
  if (!device_is_target()) {
    return 77;
  }
  return run_correctness() && run_arbitrary_m_composition() &&
                 run_shared_k128_correctness()
             ? 0
             : 1;
}
