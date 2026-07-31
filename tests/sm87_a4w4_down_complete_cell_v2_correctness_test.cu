#include "q3x/kernels/sm87_a4w4_down_complete_cell_v2.h"
#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

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
    return elements != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&data_),
                      elements * sizeof(T)) == cudaSuccess;
  }
  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  T* data_{};
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

[[nodiscard]] bool run_resource_gate() {
  kernels::Sm87A4W4DownCellV2Resources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_down_complete_cell_v2_resources_cuda(
              &resources),
          "query Down complete-cell v2 resources")) {
    return false;
  }
  const bool gate =
      resources.registers_per_thread <= 128 &&
      resources.static_shared_bytes == 42'240U &&
      resources.local_bytes == 0U && resources.active_blocks_per_sm >= 2 &&
      resources.maximum_threads_per_block >= 256 &&
      resources.compute_major == 8 && resources.compute_minor == 7;
  std::cout << "Down complete-cell v2 resources: registers="
            << resources.registers_per_thread
            << " static_shared=" << resources.static_shared_bytes
            << " dynamic_shared_limit=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  return gate;
}

[[nodiscard]] bool run_shape(const std::size_t k_count) {
  constexpr std::size_t m_count = 128U;
  constexpr std::size_t n_count = 256U;
  constexpr std::size_t output_stride = n_count + 8U;
  const std::size_t k128_groups = k_count / 128U;
  const std::size_t physical_groups = k_count / 64U;
  const std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count, k_count);
  const std::size_t b_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(n_count, k_count);
  const std::size_t a_scales_count =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(m_count,
                                                               k_count);
  const std::size_t b_scales_count =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(n_count,
                                                               k_count);
  const std::size_t output_elements = m_count * output_stride;

  std::vector<std::uint8_t> packed_a(a_bytes);
  std::vector<std::uint8_t> packed_b(b_bytes);
  std::vector<std::uint16_t> a_scales(a_scales_count);
  std::vector<std::uint16_t> b_scales(b_scales_count);
  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t k = 0U; k < k_count; k += 2U) {
      const int even =
          static_cast<int>((13U * m + 7U * k + 3U) % 15U) - 7;
      const int odd =
          static_cast<int>((5U * m + 11U * (k + 1U) + 1U) % 15U) - 7;
      packed_a[kernels::sm87_a4w4_consumer_packed_offset(
          m, k / 64U, (k % 64U) / 2U, physical_groups)] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
    for (std::size_t group = 0U; group < k128_groups; ++group) {
      a_scales[kernels::sm87_a4w4_consumer_k128_scale_offset(
          m, group, k128_groups)] =
          encode_bf16(static_cast<float>(1U + ((m + 2U * group) % 4U)) /
                      32.0F);
    }
  }
  for (std::size_t n = 0U; n < n_count; ++n) {
    for (std::size_t k = 0U; k < k_count; k += 2U) {
      const int even =
          static_cast<int>((17U * n + 3U * k + 2U) % 15U) - 7;
      const int odd =
          static_cast<int>((7U * n + 5U * (k + 1U) + 4U) % 15U) - 7;
      packed_b[kernels::sm87_a4w4_consumer_packed_offset(
          n, k / 64U, (k % 64U) / 2U, physical_groups)] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
    for (std::size_t group = 0U; group < k128_groups; ++group) {
      b_scales[kernels::sm87_a4w4_consumer_k128_scale_offset(
          n, group, k128_groups)] =
          encode_bf16(static_cast<float>(1U + ((3U * n + group) % 4U)) /
                      64.0F);
    }
  }

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint8_t> device_b;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint16_t> device_b_scales;
  DeviceBuffer<std::uint16_t> candidate_output;
  DeviceBuffer<std::uint16_t> baseline_output;
  if (!device_a.allocate(a_bytes) || !device_b.allocate(b_bytes) ||
      !device_a_scales.allocate(a_scales_count) ||
      !device_b_scales.allocate(b_scales_count) ||
      !candidate_output.allocate(output_elements) ||
      !baseline_output.allocate(output_elements)) {
    std::cerr << "K" << k_count << " device allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_a.get(), packed_a.data(), a_bytes,
                          cudaMemcpyHostToDevice),
               "copy A") ||
      !cuda_ok(cudaMemcpy(device_b.get(), packed_b.data(), b_bytes,
                          cudaMemcpyHostToDevice),
               "copy B") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), a_scales.data(),
                          a_scales.size() * sizeof(a_scales[0]),
                          cudaMemcpyHostToDevice),
               "copy A scales") ||
      !cuda_ok(cudaMemcpy(device_b_scales.get(), b_scales.data(),
                          b_scales.size() * sizeof(b_scales[0]),
                          cudaMemcpyHostToDevice),
               "copy B scales") ||
      !cuda_ok(cudaMemset(candidate_output.get(), 0x5a,
                          output_elements * sizeof(std::uint16_t)),
               "initialize candidate guard") ||
      !cuda_ok(cudaMemset(baseline_output.get(), 0x5a,
                          output_elements * sizeof(std::uint16_t)),
               "initialize baseline guard")) {
    return false;
  }

  if (k_count == 128U) {
    const int short_capacity =
        kernels::launch_sm87_a4w4_down_complete_cell_v2_bf16_cuda(
            device_a.get(), a_bytes - 1U, device_a_scales.get(),
            a_scales_count, device_b.get(), b_bytes,
            device_b_scales.get(), b_scales_count, m_count, n_count,
            k_count, candidate_output.get(), output_stride);
    const int unaligned =
        kernels::launch_sm87_a4w4_down_complete_cell_v2_bf16_cuda(
            device_a.get() + 1U, a_bytes - 1U, device_a_scales.get(),
            a_scales_count, device_b.get(), b_bytes,
            device_b_scales.get(), b_scales_count, m_count, n_count,
            k_count, candidate_output.get(), output_stride);
    const int m_tail =
        kernels::launch_sm87_a4w4_down_complete_cell_v2_bf16_cuda(
            device_a.get(), a_bytes, device_a_scales.get(), a_scales_count,
            device_b.get(), b_bytes, device_b_scales.get(), b_scales_count,
            64U, n_count, k_count, candidate_output.get(), output_stride);
    if (short_capacity != static_cast<int>(cudaErrorInvalidValue) ||
        unaligned != static_cast<int>(cudaErrorInvalidValue) ||
        m_tail != static_cast<int>(cudaErrorInvalidValue)) {
      std::cerr << "Down complete-cell v2 invalid calls did not fail closed\n";
      return false;
    }
  }

  if (!launch_ok(
          kernels::launch_sm87_a4w4_down_complete_cell_v2_bf16_cuda(
              device_a.get(), a_bytes, device_a_scales.get(),
              a_scales_count, device_b.get(), b_bytes,
              device_b_scales.get(), b_scales_count, m_count, n_count,
              k_count, candidate_output.get(), output_stride),
          "launch Down complete-cell v2") ||
      !launch_ok(kernels::launch_sm87_a4w4_prefill_gemm_k128_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scales_count, device_b.get(), b_bytes,
                     device_b_scales.get(), b_scales_count, m_count,
                     n_count, k_count, baseline_output.get(), output_stride),
                 "launch established K128 baseline") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize equivalence")) {
    return false;
  }

  std::vector<std::uint16_t> candidate(output_elements);
  std::vector<std::uint16_t> baseline(output_elements);
  if (!cuda_ok(cudaMemcpy(candidate.data(), candidate_output.get(),
                          candidate.size() * sizeof(candidate[0]),
                          cudaMemcpyDeviceToHost),
               "copy candidate") ||
      !cuda_ok(cudaMemcpy(baseline.data(), baseline_output.get(),
                          baseline.size() * sizeof(baseline[0]),
                          cudaMemcpyDeviceToHost),
               "copy baseline")) {
    return false;
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
                  m, physical_group, (k % 64U) / 2U, physical_groups)],
              k);
          const int b = kernels::sm87_a4w4_unpack_signed(
              packed_b[kernels::sm87_a4w4_consumer_packed_offset(
                  n, physical_group, (k % 64U) / 2U, physical_groups)],
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
      const std::size_t offset = m * output_stride + n;
      const std::uint16_t expected_bits = encode_bf16(expected);
      if (candidate[offset] != baseline[offset] ||
          candidate[offset] != expected_bits) {
        ++mismatches;
        if (mismatches <= 8U) {
          std::cerr << "K" << k_count << " mismatch m=" << m
                    << " n=" << n
                    << " expected=" << decode_bf16(expected_bits)
                    << " candidate=" << decode_bf16(candidate[offset])
                    << " baseline=" << decode_bf16(baseline[offset])
                    << '\n';
        }
      }
    }
    for (std::size_t n = n_count; n < output_stride; ++n) {
      const std::size_t offset = m * output_stride + n;
      if (candidate[offset] != 0x5a5aU || baseline[offset] != 0x5a5aU) {
        ++mismatches;
      }
    }
  }
  if (mismatches != 0U) {
    std::cerr << "K" << k_count << " total mismatches=" << mismatches
              << '\n';
    return false;
  }
  std::cout << "Down complete-cell v2 bitwise ring case passed: M="
            << m_count << " N=" << n_count << " K=" << k_count << '\n';
  return true;
}

}  // namespace

int main() {
  if (!device_is_target()) {
    return 77;
  }
  constexpr std::array<std::size_t, 4U> kCases{{128U, 256U, 384U, 512U}};
  if (!run_resource_gate()) {
    return 1;
  }
  for (const std::size_t k_count : kCases) {
    if (!run_shape(k_count)) {
      return 1;
    }
  }
  return 0;
}
