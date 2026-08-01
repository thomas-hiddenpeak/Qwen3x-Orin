#include "q3x/kernels/sm87_a4w4_attention_o_k512_cell.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

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

// Prevent the host compiler from contracting the scale multiply into the
// following std::fma.  This matches the device's explicit __fmul_rn +
// __fmaf_rn oracle.
[[nodiscard]] float multiply_rn(const float first, const float second) noexcept {
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
                           const char* const operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
  return false;
}

struct DeviceBuffers final {
  std::uint8_t* a{};
  std::uint16_t* a_scales{};
  std::uint8_t* b{};
  std::uint16_t* b_scales{};
  std::uint16_t* output{};

  ~DeviceBuffers() {
    cudaFree(output);
    cudaFree(b_scales);
    cudaFree(b);
    cudaFree(a_scales);
    cudaFree(a);
  }
};

[[nodiscard]] bool run_case(const std::string& label,
                            const std::size_t m_count,
                            const std::size_t n_count,
                            const std::size_t k_count,
                            const unsigned int maximum_launch_ctas) {
  using namespace q3x::kernels;
  const Sm87A4W4AttentionOK512Plan plan =
      sm87_a4w4_attention_o_k512_plan(m_count, n_count, k_count);
  if (plan.launch_ctas == 0U) {
    std::cerr << label << ": invalid test plan\n";
    return false;
  }

  const std::size_t a_bytes =
      sm87_a4w4_attention_o_k512_packed_capacity_bytes(m_count, k_count);
  const std::size_t b_bytes =
      sm87_a4w4_attention_o_k512_packed_capacity_bytes(n_count, k_count);
  const std::size_t a_scale_count =
      sm87_a4w4_attention_o_k512_scale_capacity_elements(m_count, k_count);
  const std::size_t b_scale_count =
      sm87_a4w4_attention_o_k512_scale_capacity_elements(n_count, k_count);
  const std::size_t output_stride = n_count + 2U;
  const std::size_t output_count = m_count * output_stride;
  std::vector<std::uint8_t> packed_a(a_bytes, 0U);
  std::vector<std::uint8_t> packed_b(b_bytes, 0U);
  std::vector<std::uint16_t> a_scales(a_scale_count, encode_bf16(1.0F));
  std::vector<std::uint16_t> b_scales(b_scale_count, encode_bf16(1.0F));
  std::vector<std::uint16_t> expected(output_count, 0x7fc1U);
  std::vector<std::uint16_t> actual(output_count, 0x7fc1U);

  for (std::size_t row = 0U; row < m_count; ++row) {
    for (std::size_t group = 0U; group < plan.physical_k64_groups;
         ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t k = group * 64U + 2U * byte;
        packed_a[sm87_a4w4_attention_o_k512_packed_offset(
            row, group, byte, plan.physical_k64_groups)] =
            pack_pair(a_code(row, k), a_code(row, k + 1U));
      }
    }
    for (std::size_t group = 0U; group < plan.k512_groups; ++group) {
      // Values are rounded to BF16 before either packing or oracle use.
      const float scale =
          0.0037F *
          static_cast<float>(9U + ((row * 3U + group * 5U) % 19U));
      a_scales[sm87_a4w4_attention_o_k512_scale_offset(
          row, group, plan.k512_groups)] = encode_bf16(scale);
    }
  }
  for (std::size_t row = 0U; row < n_count; ++row) {
    for (std::size_t group = 0U; group < plan.physical_k64_groups;
         ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t k = group * 64U + 2U * byte;
        packed_b[sm87_a4w4_attention_o_k512_packed_offset(
            row, group, byte, plan.physical_k64_groups)] =
            pack_pair(b_code(row, k), b_code(row, k + 1U));
      }
    }
    for (std::size_t group = 0U; group < plan.k512_groups; ++group) {
      const float scale =
          0.0023F *
          static_cast<float>(7U + ((row * 7U + group * 11U) % 23U));
      b_scales[sm87_a4w4_attention_o_k512_scale_offset(
          row, group, plan.k512_groups)] = encode_bf16(scale);
    }
  }

  bool observed_k512_boundary = false;
  bool observed_bf16_k512_boundary = false;
  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t n = 0U; n < n_count; ++n) {
      float k512_accumulator = 0.0F;
      float split_k256_accumulator = 0.0F;
      for (std::size_t group = 0U; group < plan.k512_groups; ++group) {
        std::int32_t partial = 0;
        std::int32_t half_partial[2U]{};
        for (std::size_t inner = 0U; inner < 512U; ++inner) {
          const std::int32_t product =
              static_cast<std::int32_t>(a_code(m, group * 512U + inner)) *
              static_cast<std::int32_t>(b_code(n, group * 512U + inner));
          partial += product;
          half_partial[inner / 256U] += product;
        }
        const float a_scale = decode_bf16(
            a_scales[sm87_a4w4_attention_o_k512_scale_offset(
                m, group, plan.k512_groups)]);
        const float b_scale = decode_bf16(
            b_scales[sm87_a4w4_attention_o_k512_scale_offset(
                n, group, plan.k512_groups)]);
        const float scale_product = multiply_rn(a_scale, b_scale);
        k512_accumulator =
            std::fma(static_cast<float>(partial), scale_product,
                     k512_accumulator);
        split_k256_accumulator =
            std::fma(static_cast<float>(half_partial[0U]), scale_product,
                     split_k256_accumulator);
        split_k256_accumulator =
            std::fma(static_cast<float>(half_partial[1U]), scale_product,
                     split_k256_accumulator);
      }
      observed_k512_boundary =
          observed_k512_boundary ||
          float_bits(k512_accumulator) != float_bits(split_k256_accumulator);
      observed_bf16_k512_boundary =
          observed_bf16_k512_boundary ||
          encode_bf16(k512_accumulator) !=
              encode_bf16(split_k256_accumulator);
      expected[m * output_stride + n] = encode_bf16(k512_accumulator);
    }
  }
  // With a zero accumulator and a single macro-group, the selected integer
  // fixture can legitimately round the combined and split expressions to the
  // same FP32 value.  Two or more macro-groups must still expose the defining
  // one-FMA-per-K512 boundary after a nonzero running accumulator exists.
  if (plan.k512_groups > 1U && !observed_k512_boundary) {
    std::cerr << label
              << ": fixture did not distinguish K512 from split-K256 FMA"
              << '\n';
    return false;
  }
  if (plan.k512_groups >= 12U && !observed_bf16_k512_boundary) {
    std::cerr << label
              << ": fixture distinguished FP32 but not observable BF16 K512 "
                 "output from split-K256"
              << '\n';
    return false;
  }

  DeviceBuffers device;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device.a), a_bytes),
               "cudaMalloc(A)") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device.a_scales),
                          a_scale_count * sizeof(std::uint16_t)),
               "cudaMalloc(A scales)") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device.b), b_bytes),
               "cudaMalloc(B)") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device.b_scales),
                          b_scale_count * sizeof(std::uint16_t)),
               "cudaMalloc(B scales)") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device.output),
                          output_count * sizeof(std::uint16_t)),
               "cudaMalloc(output)")) {
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device.a, packed_a.data(), a_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(A)") ||
      !cuda_ok(cudaMemcpy(device.a_scales, a_scales.data(),
                          a_scale_count * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(A scales)") ||
      !cuda_ok(cudaMemcpy(device.b, packed_b.data(), b_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(B)") ||
      !cuda_ok(cudaMemcpy(device.b_scales, b_scales.data(),
                          b_scale_count * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(B scales)") ||
      !cuda_ok(cudaMemcpy(device.output, actual.data(),
                          output_count * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(output sentinel)")) {
    return false;
  }

  const int launch_status =
      launch_sm87_a4w4_attention_o_k512_test_bf16_cuda(
          device.a, a_bytes, device.a_scales, a_scale_count, device.b,
          b_bytes, device.b_scales, b_scale_count, m_count, n_count,
          k_count, device.output, output_stride, output_count,
          maximum_launch_ctas);
  if (launch_status != static_cast<int>(cudaSuccess)) {
    std::cerr << label << ": launch failed: "
              << cudaGetErrorString(static_cast<cudaError_t>(launch_status))
              << '\n';
    return false;
  }
  if (!cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
      !cuda_ok(cudaMemcpy(actual.data(), device.output,
                          output_count * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(output)")) {
    return false;
  }

  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t n = 0U; n < n_count; ++n) {
      const std::size_t index = m * output_stride + n;
      if (actual[index] != expected[index]) {
        std::cerr << label << ": bitwise mismatch at (" << m << ',' << n
                  << "): expected 0x" << std::hex << expected[index]
                  << ", got 0x" << actual[index] << std::dec << '\n';
        return false;
      }
    }
    for (std::size_t n = n_count; n < output_stride; ++n) {
      if (actual[m * output_stride + n] != 0x7fc1U) {
        std::cerr << label << ": output row padding was overwritten\n";
        return false;
      }
    }
  }

  std::cout << label << " passed: M=" << m_count << ", N=" << n_count
            << ", K=" << k_count << ", work_tiles=" << plan.work_tiles
            << '\n';
  return true;
}

}  // namespace

int main() {
  q3x::kernels::Sm87A4W4AttentionOK512Resources resources{};
  const int resource_status =
      q3x::kernels::query_sm87_a4w4_attention_o_k512_resources_cuda(
          &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    std::cerr << "K512 resource gate failed: "
              << cudaGetErrorString(
                     static_cast<cudaError_t>(resource_status))
              << '\n';
    return 1;
  }
  std::cout << "resources: registers=" << resources.registers_per_thread
            << ", shared=" << resources.static_shared_bytes
            << ", local=" << resources.local_bytes
            << ", active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return run_case("single K512 macro", 128U, 64U, 512U, 1U) &&
                 run_case("persistent two-macro grid", 256U, 128U,
                          1'024U, 2U) &&
                 run_case("real Attention-O K depth", 128U, 64U,
                          6'144U, 1U)
             ? 0
             : 1;
}
