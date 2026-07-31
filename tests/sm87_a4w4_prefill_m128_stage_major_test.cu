#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"
#include "q3x/kernels/sm87_a4w4_gateup_paired.h"
#include "q3x/kernels/sm87_a4w4_prefill_m128_stage_major.h"
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

template <typename T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  [[nodiscard]] bool allocate(const std::size_t elements) noexcept {
    return cudaMalloc(reinterpret_cast<void**>(&pointer_),
                      elements * sizeof(T)) == cudaSuccess;
  }

  [[nodiscard]] T* get() const noexcept { return pointer_; }

 private:
  T* pointer_ = nullptr;
};

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

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t bits) noexcept {
  const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

[[nodiscard]] int round_and_clamp(const float value) noexcept {
  const int rounded = static_cast<int>(std::nearbyint(value));
  return rounded < -7 ? -7 : (rounded > 7 ? 7 : rounded);
}

[[nodiscard]] bool target_available() {
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
  int device = -1;
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice") ||
      !cuda_ok(cudaGetDeviceProperties(&properties, device),
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

struct ProjectionCase final {
  std::size_t m{};
  std::size_t n{};
  std::size_t k{};
  std::size_t stride{};
};

[[nodiscard]] bool run_projection_case(const ProjectionCase shape,
                                       const bool cpu_reference) {
  const std::size_t physical_groups = shape.k / 64U;
  const std::size_t k128_groups = shape.k / 128U;
  const std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(shape.m, shape.k);
  const std::size_t b_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(shape.n, shape.k);
  const std::size_t a_scale_elements =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(shape.m,
                                                                shape.k);
  const std::size_t b_scale_elements =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(shape.n,
                                                                shape.k);
  const std::uint16_t guard = 0x5a5aU;

  std::vector<std::uint8_t> packed_a(a_bytes);
  std::vector<std::uint8_t> packed_b(b_bytes);
  std::vector<std::uint16_t> a_scales(a_scale_elements, encode_bf16(1.0F));
  std::vector<std::uint16_t> b_scales(b_scale_elements, encode_bf16(1.0F));

  for (std::size_t m = 0U; m < shape.m; ++m) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const int even =
            static_cast<int>((7U * m + 11U * group + 3U * byte + 1U) %
                             15U) -
            7;
        const int odd =
            static_cast<int>((5U * m + 13U * group + 9U * byte + 2U) %
                             15U) -
            7;
        packed_a[kernels::sm87_a4w4_consumer_packed_offset(
            m, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(even, odd);
      }
    }
    for (std::size_t group = 0U; group < k128_groups; ++group) {
      // Powers of two make the host reference exact while still exercising
      // distinct shared K128 scale products and ordered group accumulation.
      a_scales[kernels::sm87_a4w4_consumer_k128_scale_offset(
          m, group, k128_groups)] =
          encode_bf16(std::ldexp(1.0F,
                                 -3 - static_cast<int>((m + group) % 3U)));
    }
  }
  for (std::size_t n = 0U; n < shape.n; ++n) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const int even =
            static_cast<int>((13U * n + 7U * group + 5U * byte + 3U) %
                             15U) -
            7;
        const int odd =
            static_cast<int>((3U * n + 5U * group + 11U * byte + 4U) %
                             15U) -
            7;
        packed_b[kernels::sm87_a4w4_consumer_packed_offset(
            n, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(even, odd);
      }
    }
    for (std::size_t group = 0U; group < k128_groups; ++group) {
      b_scales[kernels::sm87_a4w4_consumer_k128_scale_offset(
          n, group, k128_groups)] =
          encode_bf16(std::ldexp(1.0F,
                                 -4 - static_cast<int>((n + group) % 3U)));
    }
  }

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint8_t> device_b;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint16_t> device_b_scales;
  DeviceBuffer<std::uint16_t> device_candidate;
  DeviceBuffer<std::uint16_t> device_baseline;
  const std::size_t output_elements = shape.m * shape.stride;
  if (!device_a.allocate(a_bytes + 16U) ||
      !device_b.allocate(b_bytes) ||
      !device_a_scales.allocate(a_scale_elements) ||
      !device_b_scales.allocate(b_scale_elements) ||
      !device_candidate.allocate(output_elements) ||
      !device_baseline.allocate(output_elements)) {
    std::cerr << "device allocation failed for M=" << shape.m << '\n';
    return false;
  }
  std::vector<std::uint16_t> initialized(output_elements, guard);
  if (!cuda_ok(cudaMemcpy(device_a.get(), packed_a.data(), a_bytes,
                          cudaMemcpyHostToDevice),
               "copy packed A") ||
      !cuda_ok(cudaMemcpy(device_b.get(), packed_b.data(), b_bytes,
                          cudaMemcpyHostToDevice),
               "copy packed B") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), a_scales.data(),
                          a_scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy A scales") ||
      !cuda_ok(cudaMemcpy(device_b_scales.get(), b_scales.data(),
                          b_scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy B scales") ||
      !cuda_ok(cudaMemcpy(device_candidate.get(), initialized.data(),
                          output_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "initialize candidate output") ||
      !cuda_ok(cudaMemcpy(device_baseline.get(), initialized.data(),
                          output_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "initialize baseline output")) {
    return false;
  }

  if (!launch_ok(kernels::launch_sm87_a4w4_prefill_gemm_k128_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_elements, device_b.get(), b_bytes,
                     device_b_scales.get(), b_scale_elements, shape.m,
                     shape.n, shape.k, device_baseline.get(), shape.stride),
                 "launch established K128 baseline") ||
      !launch_ok(kernels::launch_sm87_a4w4_m128_stage_major_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_elements, device_b.get(), b_bytes,
                     device_b_scales.get(), b_scale_elements, shape.m,
                     shape.n, shape.k, device_candidate.get(), shape.stride),
                 "launch M128 stage-major candidate") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize projections")) {
    return false;
  }

  std::vector<std::uint16_t> candidate(output_elements);
  std::vector<std::uint16_t> baseline(output_elements);
  if (!cuda_ok(cudaMemcpy(candidate.data(), device_candidate.get(),
                          output_elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy candidate output") ||
      !cuda_ok(cudaMemcpy(baseline.data(), device_baseline.get(),
                          output_elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy baseline output")) {
    return false;
  }

  std::size_t mismatches = 0U;
  std::size_t guard_mismatches = 0U;
  std::size_t cpu_mismatches = 0U;
  for (std::size_t m = 0U; m < shape.m; ++m) {
    for (std::size_t n = 0U; n < shape.stride; ++n) {
      const std::size_t offset = m * shape.stride + n;
      if (candidate[offset] != baseline[offset]) {
        ++mismatches;
        if (mismatches <= 8U) {
          std::cerr << "baseline mismatch M=" << shape.m << " m=" << m
                    << " n=" << n << " baseline=0x" << std::hex
                    << baseline[offset] << " candidate=0x"
                    << candidate[offset] << std::dec << '\n';
        }
      }
      if (n >= shape.n && candidate[offset] != guard) {
        ++guard_mismatches;
      }
      if (cpu_reference && n < shape.n) {
        float expected = 0.0F;
        for (std::size_t group = 0U; group < k128_groups; ++group) {
          std::int32_t integer_partial = 0;
          for (std::size_t inner = 0U; inner < 128U; ++inner) {
            const std::size_t physical_group = 2U * group + inner / 64U;
            const std::size_t k_in_physical = inner % 64U;
            const std::uint8_t a_byte =
                packed_a[kernels::sm87_a4w4_consumer_packed_offset(
                    m, physical_group, k_in_physical / 2U,
                    physical_groups)];
            const std::uint8_t b_byte =
                packed_b[kernels::sm87_a4w4_consumer_packed_offset(
                    n, physical_group, k_in_physical / 2U,
                    physical_groups)];
            integer_partial +=
                static_cast<std::int32_t>(kernels::sm87_a4w4_unpack_signed(
                    a_byte, k_in_physical)) *
                static_cast<std::int32_t>(kernels::sm87_a4w4_unpack_signed(
                    b_byte, k_in_physical));
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
        if (candidate[offset] != encode_bf16(expected)) {
          ++cpu_mismatches;
        }
      }
    }
  }
  if (mismatches != 0U || guard_mismatches != 0U ||
      cpu_mismatches != 0U) {
    std::cerr << "M128 stage-major correctness failed M=" << shape.m
              << " baseline_mismatches=" << mismatches
              << " cpu_mismatches=" << cpu_mismatches
              << " guard_mismatches=" << guard_mismatches << '\n';
    return false;
  }

  // Exercise fail-closed shape, stride, alignment, and capacity contracts on
  // an otherwise valid allocation.  These calls return before launch.
  if (shape.m == 128U) {
    const auto reject = [&](const int status, const char* const label) {
      if (status == static_cast<int>(cudaErrorInvalidValue)) {
        return true;
      }
      std::cerr << label << " was not rejected: "
                << cudaGetErrorName(static_cast<cudaError_t>(status)) << '\n';
      return false;
    };
    if (!reject(kernels::launch_sm87_a4w4_m128_stage_major_bf16_cuda(
                    device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_elements, device_b.get(), b_bytes,
                    device_b_scales.get(), b_scale_elements, shape.m - 1U,
                    shape.n, shape.k, device_candidate.get(), shape.stride),
                "M127") ||
        !reject(kernels::launch_sm87_a4w4_m128_stage_major_bf16_cuda(
                    device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_elements, device_b.get(), b_bytes,
                    device_b_scales.get(), b_scale_elements, shape.m, 128U,
                    shape.k, device_candidate.get(), shape.stride),
                "N128") ||
        !reject(kernels::launch_sm87_a4w4_m128_stage_major_bf16_cuda(
                    device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_elements, device_b.get(), b_bytes,
                    device_b_scales.get(), b_scale_elements, shape.m,
                    shape.n, 64U, device_candidate.get(), shape.stride),
                "K64") ||
        !reject(kernels::launch_sm87_a4w4_m128_stage_major_bf16_cuda(
                    device_a.get(), a_bytes - 1U, device_a_scales.get(),
                    a_scale_elements, device_b.get(), b_bytes,
                    device_b_scales.get(), b_scale_elements, shape.m,
                    shape.n, shape.k, device_candidate.get(), shape.stride),
                "short A capacity") ||
        !reject(kernels::launch_sm87_a4w4_m128_stage_major_bf16_cuda(
                    device_a.get() + 1U, a_bytes, device_a_scales.get(),
                    a_scale_elements, device_b.get(), b_bytes,
                    device_b_scales.get(), b_scale_elements, shape.m,
                    shape.n, shape.k, device_candidate.get(), shape.stride),
                "misaligned A") ||
        !reject(kernels::launch_sm87_a4w4_m128_stage_major_bf16_cuda(
                    device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_elements, device_b.get(), b_bytes,
                    device_b_scales.get(), b_scale_elements, shape.m,
                    shape.n, shape.k, device_candidate.get(), shape.n - 1U),
                "short output stride")) {
      return false;
    }
  }

  std::cout << "bit-exact M128 stage-major projection passed: M=" << shape.m
            << " N=" << shape.n << " K=" << shape.k
            << " stride=" << shape.stride << '\n';
  return true;
}

[[nodiscard]] bool run_paired_case(const std::size_t m_count,
                                   const std::size_t n_count,
                                   const std::size_t k_count,
                                   const bool cpu_reference) {
  const std::size_t physical_groups = k_count / 64U;
  const std::size_t k128_groups = k_count / 128U;
  const std::size_t output_physical_groups = n_count / 64U;
  const std::size_t output_k128_groups = n_count / 128U;
  constexpr std::size_t packed_guard_bytes = 64U;
  constexpr std::size_t scale_guard_elements = 32U;
  constexpr float output_clip_ratio = 0.9375F;
  const std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count, k_count);
  const std::size_t a_scale_elements =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(m_count,
                                                                k_count);
  const std::size_t b_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(n_count, k_count);
  const std::size_t b_scale_elements =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(n_count,
                                                                k_count);
  const std::size_t output_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count, n_count);
  const std::size_t output_scale_elements =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(m_count,
                                                                n_count);

  std::vector<std::uint8_t> packed_a(a_bytes);
  std::vector<std::uint16_t> a_scales(
      a_scale_elements, encode_bf16(0.25F));
  std::vector<std::uint8_t> gate_b(b_bytes);
  std::vector<std::uint16_t> gate_scales(
      b_scale_elements, encode_bf16(0.25F));
  std::vector<std::uint8_t> up_b(b_bytes);
  std::vector<std::uint16_t> up_scales(
      b_scale_elements, encode_bf16(1.0F / 64.0F));

  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        packed_a[kernels::sm87_a4w4_consumer_packed_offset(
            m, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(7, 7);
      }
    }
  }
  for (std::size_t n = 0U; n < n_count; ++n) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        gate_b[kernels::sm87_a4w4_consumer_packed_offset(
            n, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(7, 7);
        const int even =
            static_cast<int>((n + 3U * group + 2U * byte) % 15U) - 7;
        const int odd =
            static_cast<int>((5U * n + group + 2U * byte + 1U) % 15U) -
            7;
        up_b[kernels::sm87_a4w4_consumer_packed_offset(
            n, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(even, odd);
      }
    }
  }

  std::vector<std::uint8_t> expected_output(output_bytes);
  std::vector<std::uint16_t> expected_scales(output_scale_elements);
  if (cpu_reference) {
    std::vector<float> products(m_count * n_count);
    for (std::size_t m = 0U; m < m_count; ++m) {
      for (std::size_t n = 0U; n < n_count; ++n) {
        std::int32_t gate_integer = 0;
        std::int32_t up_integer = 0;
        for (std::size_t k = 0U; k < k_count; ++k) {
          const std::size_t group = k / 64U;
          const std::size_t byte = (k % 64U) / 2U;
          const int a_code = kernels::sm87_a4w4_unpack_signed(
              packed_a[kernels::sm87_a4w4_consumer_packed_offset(
                  m, group, byte, physical_groups)],
              k);
          const int gate_code = kernels::sm87_a4w4_unpack_signed(
              gate_b[kernels::sm87_a4w4_consumer_packed_offset(
                  n, group, byte, physical_groups)],
              k);
          const int up_code = kernels::sm87_a4w4_unpack_signed(
              up_b[kernels::sm87_a4w4_consumer_packed_offset(
                  n, group, byte, physical_groups)],
              k);
          gate_integer += a_code * gate_code;
          up_integer += a_code * up_code;
        }
        const float a_scale = decode_bf16(
            a_scales[kernels::sm87_a4w4_consumer_k128_scale_offset(
                m, 0U, k128_groups)]);
        const float gate =
            static_cast<float>(gate_integer) *
            (a_scale * decode_bf16(
                           gate_scales[
                               kernels::sm87_a4w4_consumer_k128_scale_offset(
                                   n, 0U, k128_groups)]));
        const float up =
            static_cast<float>(up_integer) *
            (a_scale * decode_bf16(
                           up_scales[
                               kernels::sm87_a4w4_consumer_k128_scale_offset(
                                   n, 0U, k128_groups)]));
        // gate=392, so exp(-gate) underflows to zero in host and device FP32.
        products[m * n_count + n] = gate * up;
      }
    }
    for (std::size_t m = 0U; m < m_count; ++m) {
      float maximum = 0.0F;
      for (std::size_t n = 0U; n < n_count; ++n) {
        maximum = std::fmax(maximum,
                            std::fabs(products[m * n_count + n]));
      }
      const float clipped_maximum = maximum * output_clip_ratio;
      const std::uint16_t scale_bits = encode_bf16(
          maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
      const float stored_scale = decode_bf16(scale_bits);
      expected_scales[kernels::sm87_a4w4_consumer_k128_scale_offset(
          m, 0U, output_k128_groups)] = scale_bits;
      for (std::size_t n = 0U; n < n_count; n += 2U) {
        const float even = std::fmin(
            std::fmax(products[m * n_count + n], -clipped_maximum),
            clipped_maximum);
        const float odd = std::fmin(
            std::fmax(products[m * n_count + n + 1U], -clipped_maximum),
            clipped_maximum);
        expected_output[kernels::sm87_a4w4_consumer_packed_offset(
            m, n / 64U, (n % 64U) / 2U, output_physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                round_and_clamp(even / stored_scale),
                round_and_clamp(odd / stored_scale));
      }
    }
  }

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint8_t> device_gate_b;
  DeviceBuffer<std::uint16_t> device_gate_scales;
  DeviceBuffer<std::uint8_t> device_up_b;
  DeviceBuffer<std::uint16_t> device_up_scales;
  DeviceBuffer<std::uint8_t> candidate_output;
  DeviceBuffer<std::uint16_t> candidate_scales;
  DeviceBuffer<std::uint8_t> baseline_output;
  DeviceBuffer<std::uint16_t> baseline_scales;
  if (!device_a.allocate(a_bytes + 16U) ||
      !device_a_scales.allocate(a_scale_elements) ||
      !device_gate_b.allocate(b_bytes) ||
      !device_gate_scales.allocate(b_scale_elements) ||
      !device_up_b.allocate(b_bytes) ||
      !device_up_scales.allocate(b_scale_elements) ||
      !candidate_output.allocate(output_bytes + packed_guard_bytes) ||
      !candidate_scales.allocate(output_scale_elements +
                                 scale_guard_elements) ||
      !baseline_output.allocate(output_bytes + packed_guard_bytes) ||
      !baseline_scales.allocate(output_scale_elements +
                                scale_guard_elements)) {
    std::cerr << "paired device allocation failed M=" << m_count << '\n';
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_a.get(), packed_a.data(), a_bytes,
                          cudaMemcpyHostToDevice),
               "copy paired A") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), a_scales.data(),
                          a_scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy paired A scales") ||
      !cuda_ok(cudaMemcpy(device_gate_b.get(), gate_b.data(), b_bytes,
                          cudaMemcpyHostToDevice),
               "copy paired Gate B") ||
      !cuda_ok(cudaMemcpy(device_gate_scales.get(), gate_scales.data(),
                          b_scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy paired Gate scales") ||
      !cuda_ok(cudaMemcpy(device_up_b.get(), up_b.data(), b_bytes,
                          cudaMemcpyHostToDevice),
               "copy paired Up B") ||
      !cuda_ok(cudaMemcpy(device_up_scales.get(), up_scales.data(),
                          b_scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy paired Up scales") ||
      !cuda_ok(cudaMemset(candidate_output.get(), 0xa5,
                          output_bytes + packed_guard_bytes),
               "initialize paired candidate output") ||
      !cuda_ok(cudaMemset(candidate_scales.get(), 0xad,
                          (output_scale_elements + scale_guard_elements) *
                              sizeof(std::uint16_t)),
               "initialize paired candidate scales") ||
      !cuda_ok(cudaMemset(baseline_output.get(), 0xa5,
                          output_bytes + packed_guard_bytes),
               "initialize paired baseline output") ||
      !cuda_ok(cudaMemset(baseline_scales.get(), 0xad,
                          (output_scale_elements + scale_guard_elements) *
                              sizeof(std::uint16_t)),
               "initialize paired baseline scales")) {
    return false;
  }

  if (!launch_ok(kernels::launch_sm87_a4w4_gateup_paired_k128_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_elements, device_gate_b.get(), b_bytes,
                     device_gate_scales.get(), b_scale_elements,
                     device_up_b.get(), b_bytes, device_up_scales.get(),
                     b_scale_elements, m_count, n_count, k_count,
                     output_clip_ratio, baseline_output.get(), output_bytes,
                     baseline_scales.get(), output_scale_elements),
                 "launch established paired K128 baseline") ||
      !launch_ok(kernels::launch_sm87_a4w4_m128_stage_major_paired_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_elements, device_gate_b.get(), b_bytes,
                     device_gate_scales.get(), b_scale_elements,
                     device_up_b.get(), b_bytes, device_up_scales.get(),
                     b_scale_elements, m_count, n_count, k_count,
                     output_clip_ratio, candidate_output.get(), output_bytes,
                     candidate_scales.get(), output_scale_elements),
                 "launch M128 stage-major paired candidate") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize paired projections")) {
    return false;
  }

  std::vector<std::uint8_t> candidate_packed(output_bytes +
                                              packed_guard_bytes);
  std::vector<std::uint8_t> baseline_packed(output_bytes +
                                             packed_guard_bytes);
  std::vector<std::uint16_t> candidate_scale_bits(
      output_scale_elements + scale_guard_elements);
  std::vector<std::uint16_t> baseline_scale_bits(
      output_scale_elements + scale_guard_elements);
  if (!cuda_ok(cudaMemcpy(candidate_packed.data(), candidate_output.get(),
                          candidate_packed.size(), cudaMemcpyDeviceToHost),
               "copy paired candidate output") ||
      !cuda_ok(cudaMemcpy(baseline_packed.data(), baseline_output.get(),
                          baseline_packed.size(), cudaMemcpyDeviceToHost),
               "copy paired baseline output") ||
      !cuda_ok(cudaMemcpy(candidate_scale_bits.data(), candidate_scales.get(),
                          candidate_scale_bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy paired candidate scales") ||
      !cuda_ok(cudaMemcpy(baseline_scale_bits.data(), baseline_scales.get(),
                          baseline_scale_bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy paired baseline scales")) {
    return false;
  }
  if (candidate_packed != baseline_packed ||
      candidate_scale_bits != baseline_scale_bits) {
    std::cerr << "paired candidate differs from established K128 ABI M="
              << m_count << '\n';
    return false;
  }
  if (cpu_reference) {
    for (std::size_t index = 0U; index < output_bytes; ++index) {
      if (candidate_packed[index] != expected_output[index]) {
        std::cerr << "paired CPU packed mismatch index=" << index << '\n';
        return false;
      }
    }
    for (std::size_t index = 0U; index < output_scale_elements; ++index) {
      if (candidate_scale_bits[index] != expected_scales[index]) {
        std::cerr << "paired CPU scale mismatch index=" << index << '\n';
        return false;
      }
    }
  }

  for (std::size_t index = output_bytes; index < candidate_packed.size();
       ++index) {
    if (candidate_packed[index] != 0xa5U) {
      std::cerr << "paired packed guard overwritten index=" << index << '\n';
      return false;
    }
  }
  for (std::size_t index = output_scale_elements;
       index < candidate_scale_bits.size(); ++index) {
    if (candidate_scale_bits[index] != 0xadadU) {
      std::cerr << "paired scale guard overwritten index=" << index << '\n';
      return false;
    }
  }

  if (m_count == 128U && n_count == 128U && k_count == 128U) {
    const int short_scale =
        kernels::launch_sm87_a4w4_m128_stage_major_paired_cuda(
            device_a.get(), a_bytes, device_a_scales.get(),
            a_scale_elements, device_gate_b.get(), b_bytes,
            device_gate_scales.get(), b_scale_elements, device_up_b.get(),
            b_bytes, device_up_scales.get(), b_scale_elements, m_count,
            n_count, k_count, output_clip_ratio, candidate_output.get(),
            output_bytes, candidate_scales.get(), output_scale_elements - 1U);
    const int bad_clip =
        kernels::launch_sm87_a4w4_m128_stage_major_paired_cuda(
            device_a.get(), a_bytes, device_a_scales.get(),
            a_scale_elements, device_gate_b.get(), b_bytes,
            device_gate_scales.get(), b_scale_elements, device_up_b.get(),
            b_bytes, device_up_scales.get(), b_scale_elements, m_count,
            n_count, k_count, 0.0F, candidate_output.get(), output_bytes,
            candidate_scales.get(), output_scale_elements);
    const int bad_shape =
        kernels::launch_sm87_a4w4_m128_stage_major_paired_cuda(
            device_a.get(), a_bytes, device_a_scales.get(),
            a_scale_elements, device_gate_b.get(), b_bytes,
            device_gate_scales.get(), b_scale_elements, device_up_b.get(),
            b_bytes, device_up_scales.get(), b_scale_elements, 64U, n_count,
            k_count, output_clip_ratio, candidate_output.get(), output_bytes,
            candidate_scales.get(), output_scale_elements);
    const int misaligned =
        kernels::launch_sm87_a4w4_m128_stage_major_paired_cuda(
            device_a.get() + 1U, a_bytes, device_a_scales.get(),
            a_scale_elements, device_gate_b.get(), b_bytes,
            device_gate_scales.get(), b_scale_elements, device_up_b.get(),
            b_bytes, device_up_scales.get(), b_scale_elements, m_count,
            n_count, k_count, output_clip_ratio, candidate_output.get(),
            output_bytes, candidate_scales.get(), output_scale_elements);
    if (short_scale != static_cast<int>(cudaErrorInvalidValue) ||
        bad_clip != static_cast<int>(cudaErrorInvalidValue) ||
        bad_shape != static_cast<int>(cudaErrorInvalidValue) ||
        misaligned != static_cast<int>(cudaErrorInvalidValue)) {
      std::cerr << "paired invalid contract was not rejected\n";
      return false;
    }
  }

  std::cout << "bit-exact M128 stage-major paired Gate+Up passed: M="
            << m_count << " N=" << n_count << " K=" << k_count << '\n';
  return true;
}

}  // namespace

int main() {
  if (!target_available()) {
    return 77;
  }

  if (kernels::query_sm87_a4w4_m128_stage_major_resources_cuda(nullptr) !=
      static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "null resource query was not rejected\n";
    return 1;
  }
  kernels::Sm87A4W4M128StageMajorResources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_m128_stage_major_resources_cuda(
              &resources),
          "query M128 stage-major resources")) {
    return 1;
  }
  if (resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 255 || resources.local_bytes != 0U ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 50'688U ||
      resources.active_blocks_per_sm < 1 ||
      resources.maximum_threads_per_block < 256) {
    std::cerr << "resource contract failed: registers="
              << resources.registers_per_thread
              << " static_shared=" << resources.static_shared_bytes
              << " dynamic_shared=" << resources.dynamic_shared_bytes
              << " local=" << resources.local_bytes
              << " active_blocks=" << resources.active_blocks_per_sm
              << " max_threads=" << resources.maximum_threads_per_block
              << '\n';
    return 1;
  }

  kernels::Sm87A4W4M128StageMajorResources paired_resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_m128_stage_major_paired_resources_cuda(
              &paired_resources),
          "query M128 stage-major paired resources")) {
    return 1;
  }
  if (paired_resources.registers_per_thread <= 0 ||
      paired_resources.registers_per_thread > 255 ||
      paired_resources.local_bytes != 0U ||
      paired_resources.static_shared_bytes != 0U ||
      paired_resources.dynamic_shared_bytes != 65'536U ||
      paired_resources.active_blocks_per_sm < 1 ||
      paired_resources.maximum_threads_per_block < 256) {
    std::cerr << "paired resource contract failed: registers="
              << paired_resources.registers_per_thread
              << " static_shared=" << paired_resources.static_shared_bytes
              << " dynamic_shared=" << paired_resources.dynamic_shared_bytes
              << " local=" << paired_resources.local_bytes
              << " active_blocks=" << paired_resources.active_blocks_per_sm
              << '\n';
    return 1;
  }

  if (!run_projection_case({128U, 256U, 256U, 272U}, true) ||
      !run_projection_case({2'048U, 256U, 128U, 272U}, false) ||
      !run_projection_case({128U, 256U, 5'120U, 272U}, false) ||
      !run_projection_case({128U, 1'024U, 128U, 1'040U}, false) ||
      !run_paired_case(128U, 128U, 128U, true) ||
      !run_paired_case(2'048U, 128U, 128U, false) ||
      !run_paired_case(128U, 128U, 5'120U, false) ||
      !run_paired_case(128U, 512U, 512U, false)) {
    return 1;
  }

  std::cout << "M128N256K128 stage-major structural candidate passed: "
            << "registers=" << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  std::cout << "M128N128K128 paired structural candidate passed: registers="
            << paired_resources.registers_per_thread
            << " dynamic_shared=" << paired_resources.dynamic_shared_bytes
            << " local=" << paired_resources.local_bytes
            << " active_blocks_per_sm="
            << paired_resources.active_blocks_per_sm << '\n';
  return 0;
}
