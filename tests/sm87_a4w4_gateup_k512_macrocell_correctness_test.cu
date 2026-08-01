#include "q3x/kernels/sm87_a4w4_gateup_k512_macrocell.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

inline constexpr std::size_t kGuardElements = 64U;
inline constexpr std::uint16_t kSentinel = 0x7fc1U;

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
    return cudaMalloc(reinterpret_cast<void**>(&data_),
                      count * sizeof(T)) == cudaSuccess;
  }

  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  T* data_{};
};

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = float_bits(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
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

[[nodiscard]] int target_status() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      count == 0) {
    (void)cudaGetLastError();
    return 77;
  }
  if (!cuda_ok(status, "cudaGetDeviceCount")) {
    return 1;
  }
  int device = 0;
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice") ||
      !cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: requires the 16-SM SM87 target\n";
    return 77;
  }
  return 0;
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

[[nodiscard]] std::int8_t gate_code(const std::size_t row,
                                    const std::size_t k) noexcept {
  std::uint32_t mixed =
      static_cast<std::uint32_t>(row * 0xc2b2ae35U + 0x165667b1U) ^
      static_cast<std::uint32_t>(k * 0x27d4eb2fU);
  mixed ^= mixed >> 15U;
  mixed ^= mixed >> 9U;
  return static_cast<std::int8_t>(static_cast<int>(mixed & 15U) - 8);
}

[[nodiscard]] std::int8_t up_code(const std::size_t row,
                                  const std::size_t k) noexcept {
  std::uint32_t mixed =
      static_cast<std::uint32_t>(row * 0x165667b1U + 0x9e3779b9U) ^
      static_cast<std::uint32_t>(k * 0x7feb352dU + 0x846ca68bU);
  mixed ^= mixed >> 13U;
  mixed ^= mixed >> 11U;
  return static_cast<std::int8_t>(static_cast<int>(mixed & 15U) - 8);
}

struct Payload final {
  std::vector<std::uint8_t> a;
  std::vector<std::uint16_t> a_scales;
  std::vector<std::uint8_t> gate;
  std::vector<std::uint16_t> gate_scales;
  std::vector<std::uint8_t> up;
  std::vector<std::uint16_t> up_scales;
};

[[nodiscard]] Payload make_payload(const std::size_t m_count,
                                   const std::size_t full_n_count,
                                   const std::size_t k_count) {
  const std::size_t physical_groups = k_count / 64U;
  const std::size_t k512_groups = k_count / 512U;
  Payload result{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count,
                                                             k_count)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_k512_macro_scale_capacity_elements(
              m_count, k_count)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(full_n_count,
                                                             k_count)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_k512_macro_scale_capacity_elements(
              full_n_count, k_count)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(full_n_count,
                                                             k_count)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_k512_macro_scale_capacity_elements(
              full_n_count, k_count))};

  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t k = group * 64U + 2U * byte;
        result.a[kernels::sm87_a4w4_consumer_packed_offset(
            m, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                a_code(m, k), a_code(m, k + 1U));
      }
    }
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      result.a_scales[
          kernels::sm87_a4w4_gateup_k512_macro_scale_offset(
              m, group, k512_groups)] =
          encode_bf16(
              0.0021F *
              static_cast<float>(5U + (3U * m + group) % 17U));
    }
  }

  for (std::size_t n = 0U; n < full_n_count; ++n) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t k = group * 64U + 2U * byte;
        const std::size_t offset =
            kernels::sm87_a4w4_consumer_packed_offset(
                n, group, byte, physical_groups);
        result.gate[offset] = kernels::sm87_a4w4_pack_signed_pair(
            gate_code(n, k), gate_code(n, k + 1U));
        result.up[offset] = kernels::sm87_a4w4_pack_signed_pair(
            up_code(n, k), up_code(n, k + 1U));
      }
    }
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      const std::size_t offset =
          kernels::sm87_a4w4_gateup_k512_macro_scale_offset(
              n, group, k512_groups);
      result.gate_scales[offset] = encode_bf16(
          0.0017F *
          static_cast<float>(7U + (5U * n + 3U * group) % 19U));
      result.up_scales[offset] = encode_bf16(
          0.0013F *
          static_cast<float>(9U + (7U * n + group) % 23U));
    }
  }
  return result;
}

[[nodiscard]] __device__ __forceinline__ float decode_bf16_device(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_device(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ float silu_product_device(
    const float gate, const float up) noexcept {
  if (gate >= 0.0F) {
    return (gate / (1.0F + expf(-gate))) * up;
  }
  const float exponential = expf(gate);
  return (gate * exponential / (1.0F + exponential)) * up;
}

// Scalar CUDA oracle: it shares only the public physical layout and K512
// arithmetic contract with the candidate.  It does not use shared memory,
// cp.async, warp fragments, MMA, crew mapping, or the macrocell epilogue.
__global__ void gateup_k512_scalar_oracle(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_scales,
    const std::uint8_t* const packed_gate,
    const std::uint16_t* const gate_scales,
    const std::uint8_t* const packed_up,
    const std::uint16_t* const up_scales,
    const unsigned int m_count,
    const unsigned int n_start,
    const unsigned int n_count,
    const unsigned int k512_groups,
    const unsigned int physical_k64_groups,
    std::uint16_t* const output,
    const unsigned int output_stride) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int output_count = m_count * n_count;
  if (linear >= output_count) {
    return;
  }
  const unsigned int m = linear / n_count;
  const unsigned int output_n = linear - m * n_count;
  const unsigned int absolute_n = n_start + output_n;
  float gate_accumulator = 0.0F;
  float up_accumulator = 0.0F;

  for (unsigned int group = 0U; group < k512_groups; ++group) {
    std::int32_t gate_partial = 0;
    std::int32_t up_partial = 0;
    for (unsigned int inner = 0U; inner < 512U; ++inner) {
      const unsigned int logical_k = group * 512U + inner;
      const unsigned int physical_group = logical_k / 64U;
      const unsigned int byte = (logical_k % 64U) / 2U;
      const unsigned int nibble = logical_k % 2U;
      const std::int32_t a =
          kernels::sm87_a4w4_unpack_signed(
              packed_a[kernels::sm87_a4w4_consumer_packed_offset(
                  m, physical_group, byte, physical_k64_groups)],
              nibble);
      const std::size_t b_offset =
          kernels::sm87_a4w4_consumer_packed_offset(
              absolute_n, physical_group, byte,
              physical_k64_groups);
      gate_partial +=
          a * kernels::sm87_a4w4_unpack_signed(packed_gate[b_offset],
                                               nibble);
      up_partial +=
          a * kernels::sm87_a4w4_unpack_signed(packed_up[b_offset],
                                               nibble);
    }
    const float a_scale = decode_bf16_device(
        a_scales[kernels::sm87_a4w4_gateup_k512_macro_scale_offset(
            m, group, k512_groups)]);
    const float gate_scale = decode_bf16_device(
        gate_scales[
            kernels::sm87_a4w4_gateup_k512_macro_scale_offset(
                absolute_n, group, k512_groups)]);
    const float up_scale = decode_bf16_device(
        up_scales[kernels::sm87_a4w4_gateup_k512_macro_scale_offset(
            absolute_n, group, k512_groups)]);
    gate_accumulator = __fmaf_rn(
        static_cast<float>(gate_partial),
        __fmul_rn(a_scale, gate_scale), gate_accumulator);
    up_accumulator = __fmaf_rn(
        static_cast<float>(up_partial), __fmul_rn(a_scale, up_scale),
        up_accumulator);
  }
  output[static_cast<std::size_t>(m) * output_stride + output_n] =
      encode_bf16_device(
          silu_product_device(gate_accumulator, up_accumulator));
}

[[nodiscard]] bool run_case(const std::string& label,
                            const std::size_t m_count,
                            const std::size_t full_n_count,
                            const std::size_t n_start,
                            const std::size_t n_count,
                            const std::size_t k_count,
                            const unsigned int maximum_launch_ctas) {
  const kernels::Sm87A4W4GateUpK512MacroPlan plan =
      kernels::sm87_a4w4_gateup_k512_macro_plan(
          m_count, full_n_count, k_count, n_start, n_count);
  if (plan.launch_ctas == 0U) {
    std::cerr << label << ": invalid test plan\n";
    return false;
  }
  const Payload payload = make_payload(m_count, full_n_count, k_count);
  const std::size_t output_stride = n_count + 8U;
  const std::size_t output_elements = m_count * output_stride;

  DeviceBuffer<std::uint8_t> a;
  DeviceBuffer<std::uint16_t> a_scales;
  DeviceBuffer<std::uint8_t> gate;
  DeviceBuffer<std::uint16_t> gate_scales;
  DeviceBuffer<std::uint8_t> up;
  DeviceBuffer<std::uint16_t> up_scales;
  DeviceBuffer<std::uint16_t> candidate;
  DeviceBuffer<std::uint16_t> oracle;
  if (!a.allocate(payload.a.size()) ||
      !a_scales.allocate(payload.a_scales.size()) ||
      !gate.allocate(payload.gate.size()) ||
      !gate_scales.allocate(payload.gate_scales.size()) ||
      !up.allocate(payload.up.size()) ||
      !up_scales.allocate(payload.up_scales.size()) ||
      !candidate.allocate(output_elements + kGuardElements) ||
      !oracle.allocate(output_elements + kGuardElements)) {
    std::cerr << label << ": device allocation failed\n";
    return false;
  }

  std::vector<std::uint16_t> initialized(
      output_elements + kGuardElements, kSentinel);
  if (!cuda_ok(cudaMemcpy(a.get(), payload.a.data(), payload.a.size(),
                          cudaMemcpyHostToDevice), "copy A") ||
      !cuda_ok(cudaMemcpy(a_scales.get(), payload.a_scales.data(),
                          payload.a_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice), "copy A scales") ||
      !cuda_ok(cudaMemcpy(gate.get(), payload.gate.data(),
                          payload.gate.size(), cudaMemcpyHostToDevice),
               "copy Gate") ||
      !cuda_ok(cudaMemcpy(gate_scales.get(), payload.gate_scales.data(),
                          payload.gate_scales.size() *
                              sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice), "copy Gate scales") ||
      !cuda_ok(cudaMemcpy(up.get(), payload.up.data(), payload.up.size(),
                          cudaMemcpyHostToDevice), "copy Up") ||
      !cuda_ok(cudaMemcpy(up_scales.get(), payload.up_scales.data(),
                          payload.up_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice), "copy Up scales") ||
      !cuda_ok(cudaMemcpy(candidate.get(), initialized.data(),
                          initialized.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "initialize candidate output") ||
      !cuda_ok(cudaMemcpy(oracle.get(), initialized.data(),
                          initialized.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "initialize oracle output")) {
    return false;
  }

  const int short_capacity_status =
      kernels::launch_sm87_a4w4_gateup_k512_macrocell_test_bf16_cuda(
          a.get(), payload.a.size(), a_scales.get(),
          payload.a_scales.size(), gate.get(), payload.gate.size() - 1U,
          gate_scales.get(), payload.gate_scales.size(), up.get(),
          payload.up.size(), up_scales.get(), payload.up_scales.size(),
          m_count, full_n_count, k_count, n_start, n_count,
          candidate.get(), output_stride, output_elements,
          maximum_launch_ctas);
  if (short_capacity_status != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << label << ": short full-weight capacity was accepted\n";
    return false;
  }
  const int alias_status =
      kernels::launch_sm87_a4w4_gateup_k512_macrocell_test_bf16_cuda(
          a.get(), payload.a.size(), a_scales.get(),
          payload.a_scales.size(), gate.get(), payload.gate.size(),
          gate_scales.get(), payload.gate_scales.size(), up.get(),
          payload.up.size(), up_scales.get(), payload.up_scales.size(),
          m_count, full_n_count, k_count, n_start, n_count,
          reinterpret_cast<std::uint16_t*>(a.get()), output_stride,
          output_elements, maximum_launch_ctas);
  if (alias_status != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << label << ": output/input alias was accepted\n";
    return false;
  }

  if (!launch_ok(
          kernels::launch_sm87_a4w4_gateup_k512_macrocell_test_bf16_cuda(
              a.get(), payload.a.size(), a_scales.get(),
              payload.a_scales.size(), gate.get(), payload.gate.size(),
              gate_scales.get(), payload.gate_scales.size(), up.get(),
              payload.up.size(), up_scales.get(), payload.up_scales.size(),
              m_count, full_n_count, k_count, n_start, n_count,
              candidate.get(), output_stride, output_elements,
              maximum_launch_ctas),
          label + " candidate launch")) {
    return false;
  }

  constexpr unsigned int threads = 128U;
  const unsigned int logical_outputs =
      static_cast<unsigned int>(m_count * n_count);
  const unsigned int blocks = (logical_outputs + threads - 1U) / threads;
  gateup_k512_scalar_oracle<<<blocks, threads>>>(
      a.get(), a_scales.get(), gate.get(), gate_scales.get(), up.get(),
      up_scales.get(), static_cast<unsigned int>(m_count),
      static_cast<unsigned int>(n_start),
      static_cast<unsigned int>(n_count),
      static_cast<unsigned int>(plan.k512_groups),
      static_cast<unsigned int>(plan.physical_k64_groups), oracle.get(),
      static_cast<unsigned int>(output_stride));
  if (!cuda_ok(cudaPeekAtLastError(), label + " oracle launch") ||
      !cuda_ok(cudaDeviceSynchronize(), label + " synchronize")) {
    return false;
  }

  std::vector<std::uint16_t> candidate_host(initialized.size());
  std::vector<std::uint16_t> oracle_host(initialized.size());
  if (!cuda_ok(cudaMemcpy(candidate_host.data(), candidate.get(),
                          candidate_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy candidate output") ||
      !cuda_ok(cudaMemcpy(oracle_host.data(), oracle.get(),
                          oracle_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy oracle output")) {
    return false;
  }

  if (candidate_host != oracle_host) {
    std::size_t mismatches = 0U;
    for (std::size_t index = 0U; index < candidate_host.size(); ++index) {
      if (candidate_host[index] != oracle_host[index]) {
        if (mismatches == 0U) {
          std::cerr << label << ": first mismatch at " << index
                    << ", expected 0x" << std::hex << oracle_host[index]
                    << ", got 0x" << candidate_host[index] << std::dec
                    << '\n';
        }
        ++mismatches;
      }
    }
    std::cerr << label << ": total BF16 mismatches=" << mismatches
              << '\n';
    return false;
  }

  std::cout << label << " bit-exact: M=" << m_count
            << " fullN=" << full_n_count << " window=[" << n_start
            << ',' << n_start + n_count << ") K=" << k_count
            << " work_cells=" << plan.work_cells << '\n';
  return true;
}

}  // namespace

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }

  kernels::Sm87A4W4GateUpK512MacroResources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_gateup_k512_macrocell_resources_cuda(
              &resources),
          "query K512 macrocell resources") ||
      resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 128 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 83'200U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 512 ||
      resources.active_blocks_per_sm < 1) {
    std::cerr << "resource gate failed: regs="
              << resources.registers_per_thread
              << " static_shared=" << resources.static_shared_bytes
              << " dynamic_shared=" << resources.dynamic_shared_bytes
              << " local=" << resources.local_bytes
              << " active=" << resources.active_blocks_per_sm << '\n';
    return 1;
  }

  // The first case proves absolute weight-row addressing at nonzero n_start.
  // The second rotates both raw stages and both scale slots, spans two N128
  // output tiles, and exercises two M-owner CTAs.
  if (!run_case("nonzero-N window", 64U, 256U, 128U, 128U, 512U,
                1U) ||
      !run_case("two-stage/two-scale rotation", 128U, 384U, 128U,
                256U, 1'024U, 2U)) {
    return 1;
  }

  std::cout << "resource gate: registers="
            << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return 0;
}
