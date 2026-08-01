#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"
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

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const std::string& operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorName(status) << " ("
            << cudaGetErrorString(status) << ")\n";
  return false;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __host__ __device__ constexpr std::size_t a_scale_offset(
    const std::size_t row, const std::size_t group,
    const std::size_t group_count) noexcept {
  return ((row / 64U * group_count + group) * 64U + row % 64U);
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
  std::vector<std::uint8_t> paired_codes;
  std::vector<std::uint16_t> paired_scales;
};

[[nodiscard]] Payload make_payload(const std::size_t m_count,
                                   const std::size_t n_count,
                                   const std::size_t k_count) {
  const std::size_t k64_groups = k_count / 64U;
  const std::size_t k512_groups = k_count / 512U;
  Payload result{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(m_count,
                                                             k_count)),
      std::vector<std::uint16_t>(m_count * k512_groups),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(n_count,
                                                             k_count)),
      std::vector<std::uint16_t>(n_count * k512_groups),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(n_count,
                                                             k_count)),
      std::vector<std::uint16_t>(n_count * k512_groups),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
              n_count, k_count)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_k512_fragment_native_scale_capacity_elements(
              n_count, k_count))};

  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t group = 0U; group < k64_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t k = group * 64U + 2U * byte;
        result.a[kernels::sm87_a4w4_consumer_packed_offset(
            m, group, byte, k64_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                a_code(m, k), a_code(m, k + 1U));
      }
    }
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      result.a_scales[a_scale_offset(m, group, k512_groups)] =
          encode_bf16(
              0.0021F *
              static_cast<float>(5U + (3U * m + group) % 17U));
    }
  }
  for (std::size_t n = 0U; n < n_count; ++n) {
    for (std::size_t group = 0U; group < k64_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t k = group * 64U + 2U * byte;
        const std::size_t offset =
            kernels::sm87_a4w4_consumer_packed_offset(
                n, group, byte, k64_groups);
        result.gate[offset] = kernels::sm87_a4w4_pack_signed_pair(
            gate_code(n, k), gate_code(n, k + 1U));
        result.up[offset] = kernels::sm87_a4w4_pack_signed_pair(
            up_code(n, k), up_code(n, k + 1U));
      }
    }
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      const std::uint16_t gate_scale = encode_bf16(
          0.0017F *
          static_cast<float>(7U + (5U * n + 3U * group) % 19U));
      const std::uint16_t up_scale = encode_bf16(
          0.0013F *
          static_cast<float>(9U + (7U * n + group) % 23U));
      result.gate_scales[a_scale_offset(n, group, k512_groups)] =
          gate_scale;
      result.up_scales[a_scale_offset(n, group, k512_groups)] =
          up_scale;
      const std::size_t paired_offset =
          kernels::sm87_a4w4_gateup_k512_fragment_native_scale_pair_offset(
              n, group, k512_groups);
      result.paired_scales[paired_offset] = gate_scale;
      result.paired_scales[paired_offset + 1U] = up_scale;
    }
  }

  // Offline canonical -> native publication.  The test constructs it from
  // canonical bytes solely to prove the ABI permutation; the candidate
  // kernel receives only the immutable fragment-native payload.
  for (std::size_t block = 0U; block < n_count / 64U; ++block) {
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      for (std::size_t phase = 0U; phase < 8U; ++phase) {
        for (std::size_t fragment = 0U; fragment < 8U; ++fragment) {
          const std::size_t fragment_n = block * 64U + fragment * 8U;
          for (std::size_t lane = 0U; lane < 32U; ++lane) {
            const std::size_t row = fragment_n + lane / 4U;
            const std::size_t canonical0 =
                kernels::sm87_a4w4_consumer_packed_offset(
                    row, group * 8U + phase, 4U * (lane % 4U),
                    k64_groups);
            const std::size_t canonical1 =
                kernels::sm87_a4w4_consumer_packed_offset(
                    row, group * 8U + phase,
                    16U + 4U * (lane % 4U), k64_groups);
            const std::size_t native =
                kernels::sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                    fragment_n, group, phase, lane, k512_groups);
            std::memcpy(result.paired_codes.data() + native,
                        result.gate.data() + canonical0, 4U);
            std::memcpy(result.paired_codes.data() + native + 4U,
                        result.gate.data() + canonical1, 4U);
            std::memcpy(result.paired_codes.data() + native + 8U,
                        result.up.data() + canonical0, 4U);
            std::memcpy(result.paired_codes.data() + native + 12U,
                        result.up.data() + canonical1, 4U);
          }
        }
      }
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

__global__ void scalar_oracle(
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
    const unsigned int k64_groups,
    std::uint16_t* const output,
    const unsigned int output_stride) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= m_count * n_count) {
    return;
  }
  const unsigned int m = linear / n_count;
  const unsigned int output_n = linear - m * n_count;
  const unsigned int n = n_start + output_n;
  float gate_accumulator = 0.0F;
  float up_accumulator = 0.0F;
  for (unsigned int group = 0U; group < k512_groups; ++group) {
    std::int32_t gate_partial = 0;
    std::int32_t up_partial = 0;
    for (unsigned int inner = 0U; inner < 512U; ++inner) {
      const unsigned int logical_k = group * 512U + inner;
      const unsigned int k64 = logical_k / 64U;
      const unsigned int byte = (logical_k % 64U) / 2U;
      const unsigned int nibble = logical_k & 1U;
      const std::int32_t a = kernels::sm87_a4w4_unpack_signed(
          packed_a[kernels::sm87_a4w4_consumer_packed_offset(
              m, k64, byte, k64_groups)],
          nibble);
      const std::size_t b_offset =
          kernels::sm87_a4w4_consumer_packed_offset(
              n, k64, byte, k64_groups);
      gate_partial += a * kernels::sm87_a4w4_unpack_signed(
                              packed_gate[b_offset], nibble);
      up_partial += a * kernels::sm87_a4w4_unpack_signed(
                            packed_up[b_offset], nibble);
    }
    const float a_scale = decode_bf16_device(
        a_scales[a_scale_offset(m, group, k512_groups)]);
    const float gate_scale = decode_bf16_device(
        gate_scales[a_scale_offset(n, group, k512_groups)]);
    const float up_scale = decode_bf16_device(
        up_scales[a_scale_offset(n, group, k512_groups)]);
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
                            const std::size_t k_count) {
  const Payload payload = make_payload(m_count, full_n_count, k_count);
  const std::size_t output_stride = n_count + 8U;
  const std::size_t output_elements = m_count * output_stride;
  DeviceBuffer<std::uint8_t> a;
  DeviceBuffer<std::uint16_t> a_scales;
  DeviceBuffer<std::uint8_t> gate;
  DeviceBuffer<std::uint16_t> gate_scales;
  DeviceBuffer<std::uint8_t> up;
  DeviceBuffer<std::uint16_t> up_scales;
  DeviceBuffer<std::uint8_t> paired_codes;
  DeviceBuffer<std::uint16_t> paired_scales;
  DeviceBuffer<std::uint16_t> candidate;
  DeviceBuffer<std::uint16_t> oracle;
  if (!a.allocate(payload.a.size()) ||
      !a_scales.allocate(payload.a_scales.size()) ||
      !gate.allocate(payload.gate.size()) ||
      !gate_scales.allocate(payload.gate_scales.size()) ||
      !up.allocate(payload.up.size()) ||
      !up_scales.allocate(payload.up_scales.size()) ||
      !paired_codes.allocate(payload.paired_codes.size()) ||
      !paired_scales.allocate(payload.paired_scales.size()) ||
      !candidate.allocate(output_elements + kGuardElements) ||
      !oracle.allocate(output_elements + kGuardElements)) {
    std::cerr << label << ": allocation failed\n";
    return false;
  }
  const std::vector<std::uint16_t> initialized(
      output_elements + kGuardElements, kSentinel);
#define COPY_TO_DEVICE(device, host, description)                         \
  if (!cuda_ok(cudaMemcpy((device).get(), (host).data(),                 \
                          (host).size() * sizeof((host)[0]),             \
                          cudaMemcpyHostToDevice),                       \
               (description))) {                                        \
    return false;                                                        \
  }
  COPY_TO_DEVICE(a, payload.a, "copy A");
  COPY_TO_DEVICE(a_scales, payload.a_scales, "copy A scales");
  COPY_TO_DEVICE(gate, payload.gate, "copy Gate");
  COPY_TO_DEVICE(gate_scales, payload.gate_scales, "copy Gate scales");
  COPY_TO_DEVICE(up, payload.up, "copy Up");
  COPY_TO_DEVICE(up_scales, payload.up_scales, "copy Up scales");
  COPY_TO_DEVICE(paired_codes, payload.paired_codes, "copy paired codes");
  COPY_TO_DEVICE(paired_scales, payload.paired_scales,
                 "copy paired scales");
  COPY_TO_DEVICE(candidate, initialized, "initialize candidate");
  COPY_TO_DEVICE(oracle, initialized, "initialize oracle");
#undef COPY_TO_DEVICE

  const int launch_status =
      kernels::launch_sm87_a4w4_gateup_k512_fragment_native_test_bf16_cuda(
          a.get(), payload.a.size(), a_scales.get(),
          payload.a_scales.size(), paired_codes.get(),
          payload.paired_codes.size(), paired_scales.get(),
          payload.paired_scales.size(), m_count, full_n_count, k_count,
          n_start, n_count, candidate.get(), output_stride,
          output_elements, 32U);
  if (!cuda_ok(static_cast<cudaError_t>(launch_status),
               label + " candidate launch")) {
    return false;
  }
  constexpr unsigned int threads = 256U;
  const unsigned int blocks = static_cast<unsigned int>(
      (m_count * n_count + threads - 1U) / threads);
  scalar_oracle<<<blocks, threads>>>(
      a.get(), a_scales.get(), gate.get(), gate_scales.get(), up.get(),
      up_scales.get(), static_cast<unsigned int>(m_count),
      static_cast<unsigned int>(n_start),
      static_cast<unsigned int>(n_count),
      static_cast<unsigned int>(k_count / 512U),
      static_cast<unsigned int>(k_count / 64U), oracle.get(),
      static_cast<unsigned int>(output_stride));
  if (!cuda_ok(cudaGetLastError(), label + " oracle launch") ||
      !cuda_ok(cudaDeviceSynchronize(), label + " synchronize")) {
    return false;
  }
  std::vector<std::uint16_t> candidate_host(
      output_elements + kGuardElements);
  std::vector<std::uint16_t> oracle_host(
      output_elements + kGuardElements);
  if (!cuda_ok(cudaMemcpy(candidate_host.data(), candidate.get(),
                          candidate_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               label + " read candidate") ||
      !cuda_ok(cudaMemcpy(oracle_host.data(), oracle.get(),
                          oracle_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               label + " read oracle")) {
    return false;
  }
  for (std::size_t m = 0U; m < m_count; ++m) {
    for (std::size_t n = 0U; n < n_count; ++n) {
      const std::size_t offset = m * output_stride + n;
      if (candidate_host[offset] != oracle_host[offset]) {
        std::cerr << label << ": mismatch m=" << m << " n=" << n
                  << " candidate=0x" << std::hex
                  << candidate_host[offset] << " oracle=0x"
                  << oracle_host[offset] << std::dec << '\n';
        return false;
      }
    }
    for (std::size_t n = n_count; n < output_stride; ++n) {
      if (candidate_host[m * output_stride + n] != kSentinel) {
        std::cerr << label << ": row padding overwritten\n";
        return false;
      }
    }
  }
  for (std::size_t index = output_elements;
       index < candidate_host.size(); ++index) {
    if (candidate_host[index] != kSentinel) {
      std::cerr << label << ": tail guard overwritten\n";
      return false;
    }
  }
  std::cout << "PASS: " << label << " is BF16 bit-exact\n";
  return true;
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

}  // namespace

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }
  kernels::Sm87A4W4GateUpK512FragmentNativeResources resources{};
  const int resource_status =
      kernels::query_sm87_a4w4_gateup_k512_fragment_native_resources_cuda(
          &resources);
  if (!cuda_ok(static_cast<cudaError_t>(resource_status),
               "resource query")) {
    return 1;
  }
  std::cout << "resources: regs=" << resources.registers_per_thread
            << " local=" << resources.local_bytes
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  if (resources.registers_per_thread > 128 ||
      resources.local_bytes != 0U ||
      resources.active_blocks_per_sm < 2) {
    std::cerr << "resource hard gate failed\n";
    return 1;
  }
  return run_case("M64_N64_K512", 64U, 64U, 0U, 64U, 512U) &&
                 run_case("M128_N64_window_K1024", 128U, 128U, 64U,
                          64U, 1'024U) &&
                 run_case("M64_N64_model_K5120", 64U, 64U, 0U, 64U,
                          5'120U)
             ? 0
             : 1;
}
