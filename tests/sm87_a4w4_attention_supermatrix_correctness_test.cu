#include "q3x/kernels/sm87_a4w4_attention_supermatrix_cell.h"
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
  const auto query = [](const char* const name, const auto query_function) {
    kernels::Sm87A4W4AttentionSupermatrixResources resources{};
    if (!launch_ok(query_function(&resources),
                   std::string("query ") + name + " resources")) {
      return false;
    }
    const bool gate =
        resources.registers_per_thread <= 128 &&
        resources.static_shared_bytes == 42'240U &&
        resources.local_bytes == 0U &&
        resources.active_blocks_per_sm >= 2 &&
        resources.maximum_threads_per_block >= 256 &&
        resources.compute_major == 8 && resources.compute_minor == 7;
    std::cout << name << " resources: registers="
              << resources.registers_per_thread
              << " static_shared=" << resources.static_shared_bytes
              << " dynamic_shared_limit="
              << resources.dynamic_shared_bytes
              << " local=" << resources.local_bytes
              << " active_blocks_per_sm="
              << resources.active_blocks_per_sm
              << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
    return gate;
  };
  return query(
             "Linear QKV+Z",
             kernels::
                 query_sm87_a4w4_linear_qkv_z_supermatrix_resources_cuda) &&
         query(
             "Full Q/K/V",
             kernels::
                 query_sm87_a4w4_full_q_k_v_supermatrix_resources_cuda) &&
         query("Attention O",
               kernels::query_sm87_a4w4_attention_o_supermatrix_resources_cuda);
}

void fill_codes(std::vector<std::uint8_t>& packed,
                const std::size_t outer_count, const std::size_t k_count,
                const unsigned int seed) {
  const std::size_t physical_groups = k_count / 64U;
  for (std::size_t outer = 0U; outer < outer_count; ++outer) {
    for (std::size_t k = 0U; k < k_count; k += 2U) {
      const int even = static_cast<int>(
                           (13U * outer + 7U * k + 3U * seed) % 15U) -
                       7;
      const int odd = static_cast<int>(
                          (5U * outer + 11U * (k + 1U) + seed) % 15U) -
                      7;
      packed[kernels::sm87_a4w4_consumer_packed_offset(
          outer, k / 64U, (k % 64U) / 2U, physical_groups)] =
          kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
  }
}

void fill_scales(std::vector<std::uint16_t>& scales,
                 const std::size_t outer_count, const std::size_t k_count,
                 const unsigned int seed) {
  const std::size_t groups = k_count / 128U;
  for (std::size_t outer = 0U; outer < outer_count; ++outer) {
    for (std::size_t group = 0U; group < groups; ++group) {
      scales[kernels::sm87_a4w4_consumer_k128_scale_offset(
          outer, group, groups)] =
          encode_bf16(static_cast<float>(
                          1U + ((seed * outer + group + seed) % 4U)) /
                      static_cast<float>(16U << seed));
    }
  }
}

[[nodiscard]] std::uint16_t expected_value(
    const std::vector<std::uint8_t>& packed_a,
    const std::vector<std::uint16_t>& a_scales,
    const std::vector<std::uint8_t>& packed_b,
    const std::vector<std::uint16_t>& b_scales, const std::size_t m,
    const std::size_t n, const std::size_t k_count) {
  const std::size_t groups = k_count / 128U;
  const std::size_t physical_groups = k_count / 64U;
  float expected = 0.0F;
  for (std::size_t group = 0U; group < groups; ++group) {
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
            kernels::sm87_a4w4_consumer_k128_scale_offset(m, group,
                                                           groups)]) *
        decode_bf16(b_scales[
            kernels::sm87_a4w4_consumer_k128_scale_offset(n, group,
                                                           groups)]);
    expected += static_cast<float>(integer_partial) * scale_product;
  }
  return encode_bf16(expected);
}

[[nodiscard]] bool run_shape(const std::size_t k_count) {
  constexpr std::size_t m_count = 128U;
  constexpr std::size_t first_n_count = 256U;
  constexpr std::size_t second_n_count = 128U;
  constexpr std::size_t first_output_stride = first_n_count + 8U;
  constexpr std::size_t second_output_stride = second_n_count + 8U;
  const std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count, k_count);
  const std::size_t first_b_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(first_n_count,
                                                        k_count);
  const std::size_t second_b_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(second_n_count,
                                                        k_count);
  const std::size_t a_scale_count =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(m_count,
                                                               k_count);
  const std::size_t first_b_scale_count =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(
          first_n_count, k_count);
  const std::size_t second_b_scale_count =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(
          second_n_count, k_count);
  const std::size_t first_output_elements =
      m_count * first_output_stride;
  const std::size_t second_output_elements =
      m_count * second_output_stride;

  std::vector<std::uint8_t> packed_a(a_bytes);
  std::vector<std::uint8_t> packed_first_b(first_b_bytes);
  std::vector<std::uint8_t> packed_second_b(second_b_bytes);
  std::vector<std::uint16_t> a_scales(a_scale_count);
  std::vector<std::uint16_t> first_b_scales(first_b_scale_count);
  std::vector<std::uint16_t> second_b_scales(second_b_scale_count);
  fill_codes(packed_a, m_count, k_count, 1U);
  fill_codes(packed_first_b, first_n_count, k_count, 2U);
  fill_codes(packed_second_b, second_n_count, k_count, 3U);
  fill_scales(a_scales, m_count, k_count, 1U);
  fill_scales(first_b_scales, first_n_count, k_count, 2U);
  fill_scales(second_b_scales, second_n_count, k_count, 3U);

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint8_t> device_first_b;
  DeviceBuffer<std::uint8_t> device_second_b;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint16_t> device_first_b_scales;
  DeviceBuffer<std::uint16_t> device_second_b_scales;
  DeviceBuffer<std::uint16_t> first_output;
  DeviceBuffer<std::uint16_t> second_output;
  if (!device_a.allocate(a_bytes) ||
      !device_first_b.allocate(first_b_bytes) ||
      !device_second_b.allocate(second_b_bytes) ||
      !device_a_scales.allocate(a_scale_count) ||
      !device_first_b_scales.allocate(first_b_scale_count) ||
      !device_second_b_scales.allocate(second_b_scale_count) ||
      !first_output.allocate(first_output_elements) ||
      !second_output.allocate(second_output_elements)) {
    std::cerr << "K" << k_count << " device allocation failed\n";
    return false;
  }

  const auto copy_to_device = [](auto* destination, const auto& source,
                                 const char* operation) {
    return cuda_ok(cudaMemcpy(destination, source.data(),
                              source.size() * sizeof(source[0]),
                              cudaMemcpyHostToDevice),
                   operation);
  };
  if (!copy_to_device(device_a.get(), packed_a, "copy A") ||
      !copy_to_device(device_first_b.get(), packed_first_b,
                      "copy first B") ||
      !copy_to_device(device_second_b.get(), packed_second_b,
                      "copy second B") ||
      !copy_to_device(device_a_scales.get(), a_scales, "copy A scales") ||
      !copy_to_device(device_first_b_scales.get(), first_b_scales,
                      "copy first B scales") ||
      !copy_to_device(device_second_b_scales.get(), second_b_scales,
                      "copy second B scales") ||
      !cuda_ok(cudaMemset(first_output.get(), 0x5a,
                          first_output_elements * sizeof(std::uint16_t)),
               "initialize first output guard") ||
      !cuda_ok(cudaMemset(second_output.get(), 0x5a,
                          second_output_elements * sizeof(std::uint16_t)),
               "initialize second output guard")) {
    return false;
  }

  if (k_count == 128U) {
    const auto launch = [&](const std::uint8_t* a, const std::size_t a_cap,
                            const std::size_t tokens,
                            std::uint16_t* first,
                            const std::size_t first_stride,
                            const std::size_t first_capacity,
                            std::uint16_t* second) {
      return kernels::
          launch_sm87_a4w4_attention_pair_supermatrix_test_bf16_cuda(
              a, a_cap, device_a_scales.get(), a_scale_count,
              device_first_b.get(), first_b_bytes,
              device_first_b_scales.get(), first_b_scale_count,
              device_second_b.get(), second_b_bytes,
              device_second_b_scales.get(), second_b_scale_count, tokens,
              k_count, first, first_stride, first_capacity, second,
              second_output_stride, second_output_elements);
    };
    const int short_capacity =
        launch(device_a.get(), a_bytes - 1U, m_count, first_output.get(),
               first_output_stride, first_output_elements,
               second_output.get());
    const int unaligned =
        launch(device_a.get() + 1U, a_bytes - 1U, m_count,
               first_output.get(), first_output_stride,
               first_output_elements,
               second_output.get());
    const int m_tail =
        launch(device_a.get(), a_bytes, 64U, first_output.get(),
               first_output_stride, first_output_elements,
               second_output.get());
    const int short_stride =
        launch(device_a.get(), a_bytes, m_count, first_output.get(),
               first_n_count - 1U, first_output_elements,
               second_output.get());
    const int short_output =
        launch(device_a.get(), a_bytes, m_count, first_output.get(),
               first_output_stride, first_output_elements - 1U,
               second_output.get());
    const int output_alias =
        launch(device_a.get(), a_bytes, m_count, first_output.get(),
               first_output_stride, first_output_elements,
               first_output.get());
    const int output_overlap =
        launch(device_a.get(), a_bytes, m_count, first_output.get(),
               first_output_stride, first_output_elements,
               first_output.get() + 2U);
    const int input_output_overlap =
        launch(device_a.get(), a_bytes, m_count,
               reinterpret_cast<std::uint16_t*>(device_a.get()),
               first_output_stride, first_output_elements,
               second_output.get());
    if (short_capacity != static_cast<int>(cudaErrorInvalidValue) ||
        unaligned != static_cast<int>(cudaErrorInvalidValue) ||
        m_tail != static_cast<int>(cudaErrorInvalidValue) ||
        short_stride != static_cast<int>(cudaErrorInvalidValue) ||
        short_output != static_cast<int>(cudaErrorInvalidValue) ||
        output_alias != static_cast<int>(cudaErrorInvalidValue) ||
        output_overlap != static_cast<int>(cudaErrorInvalidValue) ||
        input_output_overlap != static_cast<int>(cudaErrorInvalidValue)) {
      std::cerr << "Attention supermatrix invalid calls did not fail closed\n";
      return false;
    }
  }

  if (!launch_ok(
          kernels::
              launch_sm87_a4w4_attention_pair_supermatrix_test_bf16_cuda(
                  device_a.get(), a_bytes, device_a_scales.get(),
                  a_scale_count, device_first_b.get(), first_b_bytes,
                  device_first_b_scales.get(), first_b_scale_count,
                  device_second_b.get(), second_b_bytes,
                  device_second_b_scales.get(), second_b_scale_count,
                  m_count, k_count, first_output.get(),
                  first_output_stride, first_output_elements,
                  second_output.get(), second_output_stride,
                  second_output_elements),
          "launch attention pair supermatrix") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize supermatrix")) {
    return false;
  }

  std::vector<std::uint16_t> first(first_output_elements);
  std::vector<std::uint16_t> second(second_output_elements);
  if (!cuda_ok(cudaMemcpy(first.data(), first_output.get(),
                          first.size() * sizeof(first[0]),
                          cudaMemcpyDeviceToHost),
               "copy first output") ||
      !cuda_ok(cudaMemcpy(second.data(), second_output.get(),
                          second.size() * sizeof(second[0]),
                          cudaMemcpyDeviceToHost),
               "copy second output")) {
    return false;
  }

  std::size_t mismatches = 0U;
  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t n = 0U; n < first_n_count; ++n) {
      const std::uint16_t expected_first =
          expected_value(packed_a, a_scales, packed_first_b,
                         first_b_scales, m, n, k_count);
      const std::size_t offset = m * first_output_stride + n;
      if (first[offset] != expected_first) {
        ++mismatches;
        if (mismatches <= 8U) {
          std::cerr << "K" << k_count << " first mismatch m=" << m
                    << " n=" << n
                    << " expected_first=" << decode_bf16(expected_first)
                    << " first=" << decode_bf16(first[offset]) << '\n';
        }
      }
    }
    for (std::size_t n = first_n_count; n < first_output_stride; ++n) {
      if (first[m * first_output_stride + n] != 0x5a5aU) {
        ++mismatches;
      }
    }
    for (std::size_t n = 0U; n < second_n_count; ++n) {
      const std::uint16_t expected_second =
          expected_value(packed_a, a_scales, packed_second_b,
                         second_b_scales, m, n, k_count);
      const std::size_t offset = m * second_output_stride + n;
      if (second[offset] != expected_second) {
        ++mismatches;
        if (mismatches <= 8U) {
          std::cerr << "K" << k_count << " second mismatch m=" << m
                    << " n=" << n << " expected_second="
                    << decode_bf16(expected_second)
                    << " second=" << decode_bf16(second[offset]) << '\n';
        }
      }
    }
    for (std::size_t n = second_n_count; n < second_output_stride; ++n) {
      if (second[m * second_output_stride + n] != 0x5a5aU) {
        ++mismatches;
      }
    }
  }
  if (mismatches != 0U) {
    std::cerr << "K" << k_count << " total mismatches=" << mismatches
              << '\n';
    return false;
  }
  std::cout << "Attention pair supermatrix bitwise case passed: M="
            << m_count << " N0=" << first_n_count
            << " N1=" << second_n_count << " K=" << k_count << '\n';
  return true;
}

[[nodiscard]] bool verify_matrix(
    const char* const name, const std::size_t k_count,
    const std::size_t m_count, const std::size_t n_count,
    const std::size_t output_stride,
    const std::vector<std::uint16_t>& output,
    const std::vector<std::uint8_t>& packed_a,
    const std::vector<std::uint16_t>& a_scales,
    const std::vector<std::uint8_t>& packed_b,
    const std::vector<std::uint16_t>& b_scales) {
  std::size_t mismatches = 0U;
  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t n = 0U; n < n_count; ++n) {
      const std::uint16_t expected =
          expected_value(packed_a, a_scales, packed_b, b_scales, m, n,
                         k_count);
      const std::size_t offset = m * output_stride + n;
      if (output[offset] != expected) {
        ++mismatches;
        if (mismatches <= 8U) {
          std::cerr << name << " K" << k_count << " mismatch m=" << m
                    << " n=" << n
                    << " expected=" << decode_bf16(expected)
                    << " actual=" << decode_bf16(output[offset]) << '\n';
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
    std::cerr << name << " K" << k_count
              << " total mismatches=" << mismatches << '\n';
  }
  return mismatches == 0U;
}

[[nodiscard]] bool run_full_shape(const std::size_t k_count) {
  constexpr std::size_t m_count = 128U;
  constexpr std::size_t q_count = 256U;
  constexpr std::size_t k_output_count = 64U;
  constexpr std::size_t v_count = 64U;
  constexpr std::size_t q_stride = q_count + 8U;
  constexpr std::size_t k_stride = k_output_count + 8U;
  constexpr std::size_t v_stride = v_count + 8U;
  const std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count, k_count);
  const std::size_t q_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(q_count, k_count);
  const std::size_t k_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(k_output_count,
                                                        k_count);
  const std::size_t v_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(v_count, k_count);
  const std::size_t a_scale_count =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(m_count,
                                                               k_count);
  const std::size_t q_scale_count =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(q_count,
                                                               k_count);
  const std::size_t k_scale_count =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(
          k_output_count, k_count);
  const std::size_t v_scale_count =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(v_count,
                                                               k_count);
  const std::size_t q_elements = m_count * q_stride;
  const std::size_t k_elements = m_count * k_stride;
  const std::size_t v_elements = m_count * v_stride;

  std::vector<std::uint8_t> packed_a(a_bytes);
  std::vector<std::uint8_t> packed_q(q_bytes);
  std::vector<std::uint8_t> packed_k(k_bytes);
  std::vector<std::uint8_t> packed_v(v_bytes);
  std::vector<std::uint16_t> a_scales(a_scale_count);
  std::vector<std::uint16_t> q_scales(q_scale_count);
  std::vector<std::uint16_t> k_scales(k_scale_count);
  std::vector<std::uint16_t> v_scales(v_scale_count);
  fill_codes(packed_a, m_count, k_count, 1U);
  fill_codes(packed_q, q_count, k_count, 2U);
  fill_codes(packed_k, k_output_count, k_count, 3U);
  fill_codes(packed_v, v_count, k_count, 4U);
  fill_scales(a_scales, m_count, k_count, 1U);
  fill_scales(q_scales, q_count, k_count, 2U);
  fill_scales(k_scales, k_output_count, k_count, 3U);
  fill_scales(v_scales, v_count, k_count, 4U);

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint8_t> device_q;
  DeviceBuffer<std::uint8_t> device_k;
  DeviceBuffer<std::uint8_t> device_v;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint16_t> device_q_scales;
  DeviceBuffer<std::uint16_t> device_k_scales;
  DeviceBuffer<std::uint16_t> device_v_scales;
  DeviceBuffer<std::uint16_t> q_output;
  DeviceBuffer<std::uint16_t> k_output;
  DeviceBuffer<std::uint16_t> v_output;
  if (!device_a.allocate(a_bytes) || !device_q.allocate(q_bytes) ||
      !device_k.allocate(k_bytes) || !device_v.allocate(v_bytes) ||
      !device_a_scales.allocate(a_scale_count) ||
      !device_q_scales.allocate(q_scale_count) ||
      !device_k_scales.allocate(k_scale_count) ||
      !device_v_scales.allocate(v_scale_count) ||
      !q_output.allocate(q_elements) || !k_output.allocate(k_elements) ||
      !v_output.allocate(v_elements)) {
    std::cerr << "Full K" << k_count << " device allocation failed\n";
    return false;
  }
  const auto copy = [](auto* destination, const auto& source,
                       const char* operation) {
    return cuda_ok(cudaMemcpy(destination, source.data(),
                              source.size() * sizeof(source[0]),
                              cudaMemcpyHostToDevice),
                   operation);
  };
  if (!copy(device_a.get(), packed_a, "copy Full A") ||
      !copy(device_q.get(), packed_q, "copy Full Q") ||
      !copy(device_k.get(), packed_k, "copy Full K") ||
      !copy(device_v.get(), packed_v, "copy Full V") ||
      !copy(device_a_scales.get(), a_scales, "copy Full A scales") ||
      !copy(device_q_scales.get(), q_scales, "copy Full Q scales") ||
      !copy(device_k_scales.get(), k_scales, "copy Full K scales") ||
      !copy(device_v_scales.get(), v_scales, "copy Full V scales") ||
      !cuda_ok(cudaMemset(q_output.get(), 0x5a,
                          q_elements * sizeof(std::uint16_t)),
               "initialize Full Q guard") ||
      !cuda_ok(cudaMemset(k_output.get(), 0x5a,
                          k_elements * sizeof(std::uint16_t)),
               "initialize Full K guard") ||
      !cuda_ok(cudaMemset(v_output.get(), 0x5a,
                          v_elements * sizeof(std::uint16_t)),
               "initialize Full V guard")) {
    return false;
  }

  const auto launch = [&](const std::size_t tokens,
                          const std::size_t q_capacity,
                          const std::size_t k_scale_capacity,
                          std::uint16_t* q_out,
                          const std::size_t q_output_stride,
                          const std::size_t q_output_capacity,
                          std::uint16_t* k_out) {
    return kernels::launch_sm87_a4w4_full_q_k_v_supermatrix_test_bf16_cuda(
        device_a.get(), a_bytes, device_a_scales.get(), a_scale_count,
        device_q.get(), q_capacity, device_q_scales.get(), q_scale_count,
        device_k.get(), k_bytes, device_k_scales.get(), k_scale_capacity,
        device_v.get(), v_bytes, device_v_scales.get(), v_scale_count,
        tokens, k_count, q_out, q_output_stride, q_output_capacity, k_out,
        k_stride, k_elements, v_output.get(), v_stride, v_elements);
  };
  if (k_count == 128U) {
    const int short_q =
        launch(m_count, q_bytes - 1U, k_scale_count, q_output.get(),
               q_stride, q_elements, k_output.get());
    const int short_k_scales =
        launch(m_count, q_bytes, k_scale_count - 1U, q_output.get(),
               q_stride, q_elements, k_output.get());
    const int m_tail =
        launch(64U, q_bytes, k_scale_count, q_output.get(), q_stride,
               q_elements, k_output.get());
    const int short_stride =
        launch(m_count, q_bytes, k_scale_count, q_output.get(),
               q_count - 1U, q_elements, k_output.get());
    const int short_output =
        launch(m_count, q_bytes, k_scale_count, q_output.get(), q_stride,
               q_elements - 1U, k_output.get());
    const int output_overlap =
        launch(m_count, q_bytes, k_scale_count, q_output.get(), q_stride,
               q_elements, q_output.get());
    if (short_q != static_cast<int>(cudaErrorInvalidValue) ||
        short_k_scales != static_cast<int>(cudaErrorInvalidValue) ||
        m_tail != static_cast<int>(cudaErrorInvalidValue) ||
        short_stride != static_cast<int>(cudaErrorInvalidValue) ||
        short_output != static_cast<int>(cudaErrorInvalidValue) ||
        output_overlap != static_cast<int>(cudaErrorInvalidValue)) {
      std::cerr << "Full supermatrix invalid calls did not fail closed\n";
      return false;
    }
  }

  if (!launch_ok(
          launch(m_count, q_bytes, k_scale_count, q_output.get(), q_stride,
                 q_elements, k_output.get()),
          "launch Full Q/K/V supermatrix") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize Full supermatrix")) {
    return false;
  }
  std::vector<std::uint16_t> q_host(q_elements);
  std::vector<std::uint16_t> k_host(k_elements);
  std::vector<std::uint16_t> v_host(v_elements);
  if (!cuda_ok(cudaMemcpy(q_host.data(), q_output.get(),
                          q_host.size() * sizeof(q_host[0]),
                          cudaMemcpyDeviceToHost),
               "copy Full Q output") ||
      !cuda_ok(cudaMemcpy(k_host.data(), k_output.get(),
                          k_host.size() * sizeof(k_host[0]),
                          cudaMemcpyDeviceToHost),
               "copy Full K output") ||
      !cuda_ok(cudaMemcpy(v_host.data(), v_output.get(),
                          v_host.size() * sizeof(v_host[0]),
                          cudaMemcpyDeviceToHost),
               "copy Full V output")) {
    return false;
  }
  if (!verify_matrix("Full Q", k_count, m_count, q_count, q_stride,
                     q_host, packed_a, a_scales, packed_q, q_scales) ||
      !verify_matrix("Full K", k_count, m_count, k_output_count, k_stride,
                     k_host, packed_a, a_scales, packed_k, k_scales) ||
      !verify_matrix("Full V", k_count, m_count, v_count, v_stride,
                     v_host, packed_a, a_scales, packed_v, v_scales)) {
    return false;
  }
  std::cout << "Full Q/K/V supermatrix bitwise case passed: M=" << m_count
            << " Q=" << q_count << " K/V=" << k_output_count
            << " inner=" << k_count << '\n';
  return true;
}

[[nodiscard]] bool run_o_shape(const std::size_t k_count) {
  constexpr std::size_t m_count = 128U;
  constexpr std::size_t n_count = 256U;
  constexpr std::size_t output_stride = n_count + 8U;
  const std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count, k_count);
  const std::size_t b_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(n_count, k_count);
  const std::size_t a_scale_count =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(m_count,
                                                               k_count);
  const std::size_t b_scale_count =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(n_count,
                                                               k_count);
  const std::size_t output_elements = m_count * output_stride;
  std::vector<std::uint8_t> packed_a(a_bytes);
  std::vector<std::uint8_t> packed_b(b_bytes);
  std::vector<std::uint16_t> a_scales(a_scale_count);
  std::vector<std::uint16_t> b_scales(b_scale_count);
  fill_codes(packed_a, m_count, k_count, 1U);
  fill_codes(packed_b, n_count, k_count, 4U);
  fill_scales(a_scales, m_count, k_count, 1U);
  fill_scales(b_scales, n_count, k_count, 4U);

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint8_t> device_b;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint16_t> device_b_scales;
  DeviceBuffer<std::uint16_t> output;
  if (!device_a.allocate(a_bytes) || !device_b.allocate(b_bytes) ||
      !device_a_scales.allocate(a_scale_count) ||
      !device_b_scales.allocate(b_scale_count) ||
      !output.allocate(output_elements)) {
    std::cerr << "O K" << k_count << " device allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_a.get(), packed_a.data(), a_bytes,
                          cudaMemcpyHostToDevice),
               "copy O A") ||
      !cuda_ok(cudaMemcpy(device_b.get(), packed_b.data(), b_bytes,
                          cudaMemcpyHostToDevice),
               "copy O B") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), a_scales.data(),
                          a_scales.size() * sizeof(a_scales[0]),
                          cudaMemcpyHostToDevice),
               "copy O A scales") ||
      !cuda_ok(cudaMemcpy(device_b_scales.get(), b_scales.data(),
                          b_scales.size() * sizeof(b_scales[0]),
                          cudaMemcpyHostToDevice),
               "copy O B scales") ||
      !cuda_ok(cudaMemset(output.get(), 0x5a,
                          output_elements * sizeof(std::uint16_t)),
               "initialize O guard")) {
    return false;
  }
  const auto launch = [&](const std::size_t tokens,
                          const std::size_t b_capacity,
                          std::uint16_t* output_pointer,
                          const std::size_t stride,
                          const std::size_t output_capacity) {
    return kernels::launch_sm87_a4w4_attention_o_supermatrix_test_bf16_cuda(
        device_a.get(), a_bytes, device_a_scales.get(), a_scale_count,
        device_b.get(), b_capacity, device_b_scales.get(), b_scale_count,
        tokens, k_count, output_pointer, stride, output_capacity);
  };
  if (k_count == 128U) {
    const int short_b = launch(m_count, b_bytes - 1U, output.get(),
                               output_stride, output_elements);
    const int m_tail = launch(64U, b_bytes, output.get(), output_stride,
                              output_elements);
    const int short_stride = launch(m_count, b_bytes, output.get(),
                                    n_count - 1U, output_elements);
    const int short_output = launch(m_count, b_bytes, output.get(),
                                    output_stride, output_elements - 1U);
    const int input_overlap = launch(
        m_count, b_bytes,
        reinterpret_cast<std::uint16_t*>(device_a.get()), output_stride,
        output_elements);
    if (short_b != static_cast<int>(cudaErrorInvalidValue) ||
        m_tail != static_cast<int>(cudaErrorInvalidValue) ||
        short_stride != static_cast<int>(cudaErrorInvalidValue) ||
        short_output != static_cast<int>(cudaErrorInvalidValue) ||
        input_overlap != static_cast<int>(cudaErrorInvalidValue)) {
      std::cerr << "O supermatrix invalid calls did not fail closed\n";
      return false;
    }
  }
  if (!launch_ok(launch(m_count, b_bytes, output.get(), output_stride,
                        output_elements),
                 "launch O supermatrix") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize O supermatrix")) {
    return false;
  }
  std::vector<std::uint16_t> host(output_elements);
  if (!cuda_ok(cudaMemcpy(host.data(), output.get(),
                          host.size() * sizeof(host[0]),
                          cudaMemcpyDeviceToHost),
               "copy O output")) {
    return false;
  }
  if (!verify_matrix("Attention O", k_count, m_count, n_count,
                     output_stride, host, packed_a, a_scales, packed_b,
                     b_scales)) {
    return false;
  }
  std::cout << "Attention O supermatrix bitwise case passed: M=" << m_count
            << " N=" << n_count << " K=" << k_count << '\n';
  return true;
}

}  // namespace

int main() {
  if (!device_is_target()) {
    return 77;
  }
  if (!run_resource_gate()) {
    return 1;
  }
  constexpr std::array<std::size_t, 4U> kCases{{128U, 256U, 384U, 512U}};
  for (const std::size_t k_count : kCases) {
    if (!run_shape(k_count) || !run_full_shape(k_count) ||
        !run_o_shape(k_count)) {
      return 1;
    }
  }
  return 0;
}
