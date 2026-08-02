#include "q3x/kernels/sm87_a4w4_down_k512_m16n64_v2.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] float bits_float(const std::uint32_t bits) noexcept {
  float value{};
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = float_bits(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t bits) noexcept {
  return bits_float(static_cast<std::uint32_t>(bits) << 16U);
}

[[nodiscard]] float multiply_rn(const float first,
                                const float second) noexcept {
  volatile float product = first * second;
  return product;
}

[[nodiscard]] std::int8_t a_code(const std::size_t row,
                                 const std::size_t k) noexcept {
  std::uint32_t mixed =
      static_cast<std::uint32_t>(row * 0x9e3779b9U) ^
      static_cast<std::uint32_t>(k * 0x85ebca6bU + 0x27d4eb2dU);
  mixed ^= mixed >> 16U;
  mixed ^= mixed >> 7U;
  return static_cast<std::int8_t>(static_cast<int>(mixed & 15U) - 8);
}

[[nodiscard]] std::int8_t b_code(const std::size_t row,
                                 const std::size_t k) noexcept {
  std::uint32_t mixed =
      static_cast<std::uint32_t>(row * 0xc2b2ae35U + 0x165667b1U) ^
      static_cast<std::uint32_t>(k * 0x27d4eb2fU);
  mixed ^= mixed >> 15U;
  mixed ^= mixed >> 9U;
  return static_cast<std::int8_t>(static_cast<int>(mixed & 15U) - 8);
}

[[nodiscard]] std::uint8_t pack_pair(const std::int8_t even,
                                     const std::int8_t odd) noexcept {
  return static_cast<std::uint8_t>(
      (static_cast<unsigned int>(even) & 15U) |
      ((static_cast<unsigned int>(odd) & 15U) << 4U));
}

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const std::string& operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << " failed: " << cudaGetErrorName(status)
            << " (" << cudaGetErrorString(status) << ")\n";
  return false;
}

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

  [[nodiscard]] bool allocate(const std::size_t count) noexcept {
    return count != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&data_),
                      count * sizeof(T)) == cudaSuccess;
  }

  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  T* data_{};
};

[[nodiscard]] int target_state() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      count == 0) {
    (void)cudaGetLastError();
    std::cout << "SKIP: CUDA device unavailable\n";
    return 0;
  }
  if (!cuda_ok(status, "cudaGetDeviceCount")) {
    return -1;
  }
  int device = -1;
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice")) {
    return -1;
  }
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return -1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16 ||
      properties.sharedMemPerBlockOptin <
          kernels::kSm87A4W4DownK512M16N64V2DynamicSharedBytes) {
    std::cout << "SKIP: requires 16-SM SM87 with >=128 KiB opt-in shared\n";
    return 0;
  }
  return 1;
}

[[nodiscard]] bool resource_gate() {
  kernels::Sm87A4W4DownK512M16N64V2Resources resources{};
  const int status =
      kernels::query_sm87_a4w4_down_k512_m16n64_v2_resources_cuda(
          &resources);
  if (status != static_cast<int>(cudaSuccess)) {
    std::cerr << "resource query failed: "
              << cudaGetErrorName(static_cast<cudaError_t>(status)) << '\n';
    return false;
  }
  const bool pass =
      resources.registers_per_thread > 0 &&
      resources.registers_per_thread <=
          static_cast<int>(
              kernels::kSm87A4W4DownK512M16N64V2MaximumRegisters) &&
      resources.static_shared_bytes == 0U &&
      resources.dynamic_shared_bytes == 128U * 1'024U &&
      resources.configured_dynamic_shared_limit_bytes >= 128U * 1'024U &&
      resources.device_optin_shared_limit_bytes >= 128U * 1'024U &&
      resources.local_bytes == 0U &&
      resources.maximum_threads_per_block >= 512 &&
      resources.active_blocks_per_sm == 1 &&
      resources.compute_major == 8 && resources.compute_minor == 7;
  std::cout << "Down K512 M16N64 v2 resources: registers="
            << resources.registers_per_thread
            << " static_shared=" << resources.static_shared_bytes
            << " requested_dynamic=" << resources.dynamic_shared_bytes
            << " configured_dynamic="
            << resources.configured_dynamic_shared_limit_bytes
            << " device_optin=" << resources.device_optin_shared_limit_bytes
            << " local=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << " gate=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass;
}

[[nodiscard]] bool run_case(const std::size_t k_count) {
  constexpr std::size_t m_count = 128U;
  constexpr std::size_t n_count = 256U;
  constexpr std::size_t output_stride = n_count + 8U;
  const auto plan =
      kernels::sm87_a4w4_down_k512_m16n64_v2_test_plan(
          m_count, n_count, k_count);
  if (plan.work_tiles != 2U || plan.launch_ctas != 1U) {
    std::cerr << "K" << k_count << ": invalid synthetic plan\n";
    return false;
  }

  const std::size_t a_bytes =
      kernels::sm87_a4w4_down_k512_packed_capacity_bytes(
          m_count, k_count);
  const std::size_t b_bytes =
      kernels::sm87_a4w4_down_k512_packed_capacity_bytes(
          n_count, k_count);
  const std::size_t a_scale_count =
      kernels::sm87_a4w4_down_k512_scale_capacity_elements(
          m_count, k_count);
  const std::size_t b_scale_count =
      kernels::sm87_a4w4_down_k512_scale_capacity_elements(
          n_count, k_count);
  const std::size_t output_count = m_count * output_stride;
  std::vector<std::uint8_t> packed_a(a_bytes, 0U);
  std::vector<std::uint8_t> packed_b(b_bytes, 0U);
  std::vector<std::uint16_t> a_scales(a_scale_count);
  std::vector<std::uint16_t> b_scales(b_scale_count);
  std::vector<std::uint16_t> expected(output_count, 0x7fc1U);
  std::vector<std::uint16_t> actual(output_count, 0x7fc1U);

  for (std::size_t row = 0U; row < m_count; ++row) {
    for (std::size_t group = 0U; group < plan.physical_k64_groups;
         ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t k = group * 64U + 2U * byte;
        packed_a[kernels::sm87_a4w4_down_k512_packed_offset(
            row, group, byte, plan.physical_k64_groups)] =
            pack_pair(a_code(row, k), a_code(row, k + 1U));
      }
    }
    for (std::size_t group = 0U; group < plan.k512_groups; ++group) {
      const float scale =
          0.0037F *
          static_cast<float>(9U + ((3U * row + 5U * group) % 19U));
      a_scales[kernels::sm87_a4w4_down_k512_scale_offset(
          row, group, plan.k512_groups)] = encode_bf16(scale);
    }
  }
  for (std::size_t row = 0U; row < n_count; ++row) {
    for (std::size_t group = 0U; group < plan.physical_k64_groups;
         ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t k = group * 64U + 2U * byte;
        packed_b[kernels::sm87_a4w4_down_k512_packed_offset(
            row, group, byte, plan.physical_k64_groups)] =
            pack_pair(b_code(row, k), b_code(row, k + 1U));
      }
    }
    for (std::size_t group = 0U; group < plan.k512_groups; ++group) {
      const float scale =
          0.0023F *
          static_cast<float>(7U + ((7U * row + 11U * group) % 23U));
      b_scales[kernels::sm87_a4w4_down_k512_scale_offset(
          row, group, plan.k512_groups)] = encode_bf16(scale);
    }
  }

  bool observed_split_boundary = false;
  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t n = 0U; n < n_count; ++n) {
      float combined = 0.0F;
      float split = 0.0F;
      for (std::size_t group = 0U; group < plan.k512_groups; ++group) {
        std::int32_t partial = 0;
        std::int32_t half[2U]{};
        for (std::size_t inner = 0U; inner < 512U; ++inner) {
          const std::int32_t product =
              static_cast<std::int32_t>(
                  a_code(m, group * 512U + inner)) *
              static_cast<std::int32_t>(
                  b_code(n, group * 512U + inner));
          partial += product;
          half[inner / 256U] += product;
        }
        const float a_scale = decode_bf16(
            a_scales[kernels::sm87_a4w4_down_k512_scale_offset(
                m, group, plan.k512_groups)]);
        const float b_scale = decode_bf16(
            b_scales[kernels::sm87_a4w4_down_k512_scale_offset(
                n, group, plan.k512_groups)]);
        const float scale_product = multiply_rn(a_scale, b_scale);
        combined = std::fma(static_cast<float>(partial), scale_product,
                            combined);
        split = std::fma(static_cast<float>(half[0U]), scale_product, split);
        split = std::fma(static_cast<float>(half[1U]), scale_product, split);
      }
      observed_split_boundary =
          observed_split_boundary || float_bits(combined) != float_bits(split);
      expected[m * output_stride + n] = encode_bf16(combined);
    }
  }
  if (plan.k512_groups > 1U && !observed_split_boundary) {
    std::cerr << "K" << k_count
              << ": fixture did not expose combined-vs-split boundary\n";
    return false;
  }

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint8_t> device_b;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint16_t> device_b_scales;
  DeviceBuffer<std::uint16_t> device_output;
  if (!device_a.allocate(a_bytes) || !device_b.allocate(b_bytes) ||
      !device_a_scales.allocate(a_scale_count) ||
      !device_b_scales.allocate(b_scale_count) ||
      !device_output.allocate(output_count)) {
    std::cerr << "K" << k_count << ": device allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_a.get(), packed_a.data(), a_bytes,
                          cudaMemcpyHostToDevice),
               "copy A") ||
      !cuda_ok(cudaMemcpy(device_b.get(), packed_b.data(), b_bytes,
                          cudaMemcpyHostToDevice),
               "copy B") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), a_scales.data(),
                          a_scale_count * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy A scales") ||
      !cuda_ok(cudaMemcpy(device_b_scales.get(), b_scales.data(),
                          b_scale_count * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy B scales") ||
      !cuda_ok(cudaMemcpy(device_output.get(), actual.data(),
                          output_count * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy output guards")) {
    return false;
  }

  if (k_count == 512U) {
    const auto invalid = [&](const std::uint8_t* a,
                             const std::size_t a_capacity,
                             const std::size_t m,
                             const std::size_t output_capacity,
                             std::uint16_t* output) {
      return kernels::launch_sm87_a4w4_down_k512_m16n64_v2_test_bf16_cuda(
          a, a_capacity, device_a_scales.get(), a_scale_count,
          device_b.get(), b_bytes, device_b_scales.get(), b_scale_count,
          m, n_count, k_count, output, output_stride, output_capacity, 1U);
    };
    const int short_a = invalid(device_a.get(), a_bytes - 1U, m_count,
                                output_count, device_output.get());
    const int unaligned = invalid(device_a.get() + 1U, a_bytes - 1U,
                                  m_count, output_count,
                                  device_output.get());
    const int m_tail = invalid(device_a.get(), a_bytes, m_count - 1U,
                               output_count, device_output.get());
    const int short_output = invalid(
        device_a.get(), a_bytes, m_count, output_count - 1U,
        device_output.get());
    const int alias_output = invalid(
        device_a.get(), a_bytes, m_count, output_count,
        reinterpret_cast<std::uint16_t*>(device_a.get()));
    const int production_wrong_shape =
        kernels::launch_sm87_a4w4_down_k512_m16n64_v2_bf16_cuda(
            device_a.get(), a_bytes, device_a_scales.get(), a_scale_count,
            device_b.get(), b_bytes, device_b_scales.get(), b_scale_count,
            m_count, n_count, k_count, device_output.get(), output_stride,
            output_count);
    if (short_a != static_cast<int>(cudaErrorInvalidValue) ||
        unaligned != static_cast<int>(cudaErrorInvalidValue) ||
        m_tail != static_cast<int>(cudaErrorInvalidValue) ||
        short_output != static_cast<int>(cudaErrorInvalidValue) ||
        alias_output != static_cast<int>(cudaErrorInvalidValue) ||
        production_wrong_shape != static_cast<int>(cudaErrorInvalidValue)) {
      std::cerr << "K512 capacity/alignment/shape/alias gate failed\n";
      return false;
    }
  }

  const int launch_status =
      kernels::launch_sm87_a4w4_down_k512_m16n64_v2_test_bf16_cuda(
          device_a.get(), a_bytes, device_a_scales.get(), a_scale_count,
          device_b.get(), b_bytes, device_b_scales.get(), b_scale_count,
          m_count, n_count, k_count, device_output.get(), output_stride,
          output_count, 1U);
  if (launch_status != static_cast<int>(cudaSuccess) ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize kernel") ||
      !cuda_ok(cudaMemcpy(actual.data(), device_output.get(),
                          output_count * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy output")) {
    if (launch_status != static_cast<int>(cudaSuccess)) {
      std::cerr << "launch failed: "
                << cudaGetErrorName(
                       static_cast<cudaError_t>(launch_status))
                << '\n';
    }
    return false;
  }

  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t n = 0U; n < n_count; ++n) {
      const std::size_t index = m * output_stride + n;
      if (actual[index] != expected[index]) {
        std::cerr << "K" << k_count << " bitwise mismatch at (" << m
                  << ',' << n << "): expected 0x" << std::hex
                  << expected[index] << ", got 0x" << actual[index]
                  << std::dec << '\n';
        return false;
      }
    }
    for (std::size_t n = n_count; n < output_stride; ++n) {
      if (actual[m * output_stride + n] != 0x7fc1U) {
        std::cerr << "K" << k_count << ": output padding overwritten\n";
        return false;
      }
    }
  }
  std::cout << "Down K512 M16N64 v2 correctness passed: M=" << m_count
            << " N=" << n_count << " K=" << k_count << '\n';
  return true;
}

}  // namespace

int main() {
  const int target = target_state();
  if (target == 0) {
    return 77;
  }
  if (target < 0 || !resource_gate()) {
    return 1;
  }
  constexpr std::array<std::size_t, 2U> kCases{{512U, 1'024U}};
  for (const std::size_t k : kCases) {
    if (!run_case(k)) {
      return 1;
    }
  }
  return 0;
}
