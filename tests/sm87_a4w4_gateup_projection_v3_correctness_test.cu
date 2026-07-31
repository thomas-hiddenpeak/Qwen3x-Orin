#include "q3x/kernels/sm87_a4w4_gateup_paired.h"
#include "q3x/kernels/sm87_a4w4_gateup_projection_v3.h"
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

inline constexpr std::size_t kM = 64U;
inline constexpr std::size_t kN = 128U;
inline constexpr std::size_t kGuardBytes = 64U;
inline constexpr std::size_t kGuardScales = 64U;
inline constexpr float kClipRatio = 0.9375F;

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

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
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

struct Payload final {
  std::vector<std::uint8_t> a;
  std::vector<std::uint16_t> a_scales;
  std::vector<std::uint8_t> gate;
  std::vector<std::uint16_t> gate_scales;
  std::vector<std::uint8_t> up;
  std::vector<std::uint16_t> up_scales;
};

[[nodiscard]] Payload make_payload(const std::size_t k) {
  const std::size_t physical_groups = k / 64U;
  const std::size_t k128_groups = k / 128U;
  Payload result{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(kM, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(kM, k)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(kN, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(kN, k)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(kN, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(kN, k))};

  for (std::size_t m = 0U; m < kM; ++m) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const int even =
            static_cast<int>((13U * m + 5U * group + 3U * byte) % 15U) - 7;
        const int odd = static_cast<int>(
                            (7U * m + 11U * group + 5U * byte + 1U) % 15U) -
                        7;
        result.a[kernels::sm87_a4w4_consumer_packed_offset(
            m, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(even, odd);
      }
    }
    for (std::size_t group = 0U; group < k128_groups; ++group) {
      result.a_scales[kernels::sm87_a4w4_consumer_k128_scale_offset(
          m, group, k128_groups)] = encode_bf16(
          static_cast<float>(2U + (m + group) % 7U) / 64.0F);
    }
  }
  for (std::size_t n = 0U; n < kN; ++n) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const int gate_even = static_cast<int>(
                                  (5U * n + 3U * group + 7U * byte) % 15U) -
                              7;
        const int gate_odd = static_cast<int>(
                                 (11U * n + group + 2U * byte + 1U) % 15U) -
                             7;
        const int up_even = static_cast<int>(
                                (7U * n + 13U * group + 5U * byte) % 15U) -
                            7;
        const int up_odd = static_cast<int>(
                               (3U * n + 9U * group + 11U * byte + 2U) %
                               15U) -
                           7;
        const std::size_t offset =
            kernels::sm87_a4w4_consumer_packed_offset(
                n, group, byte, physical_groups);
        result.gate[offset] =
            kernels::sm87_a4w4_pack_signed_pair(gate_even, gate_odd);
        result.up[offset] =
            kernels::sm87_a4w4_pack_signed_pair(up_even, up_odd);
      }
    }
    for (std::size_t group = 0U; group < k128_groups; ++group) {
      const std::size_t offset =
          kernels::sm87_a4w4_consumer_k128_scale_offset(
              n, group, k128_groups);
      result.gate_scales[offset] = encode_bf16(
          static_cast<float>(1U + (n + 2U * group) % 5U) / 128.0F);
      result.up_scales[offset] = encode_bf16(
          static_cast<float>(2U + (3U * n + group) % 7U) / 128.0F);
    }
  }
  return result;
}

[[nodiscard]] bool run_case(const std::size_t k) {
  const Payload payload = make_payload(k);
  const std::size_t output_packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kM, kN);
  const std::size_t output_scale_elements =
      kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(kM, kN);
  DeviceBuffer<std::uint8_t> a;
  DeviceBuffer<std::uint16_t> a_scales;
  DeviceBuffer<std::uint8_t> gate;
  DeviceBuffer<std::uint16_t> gate_scales;
  DeviceBuffer<std::uint8_t> up;
  DeviceBuffer<std::uint16_t> up_scales;
  DeviceBuffer<std::uint8_t> baseline;
  DeviceBuffer<std::uint16_t> baseline_scales;
  DeviceBuffer<std::uint8_t> candidate;
  DeviceBuffer<std::uint16_t> candidate_scales;
  if (!a.allocate(payload.a.size()) ||
      !a_scales.allocate(payload.a_scales.size()) ||
      !gate.allocate(payload.gate.size()) ||
      !gate_scales.allocate(payload.gate_scales.size()) ||
      !up.allocate(payload.up.size()) ||
      !up_scales.allocate(payload.up_scales.size()) ||
      !baseline.allocate(output_packed_bytes + kGuardBytes) ||
      !baseline_scales.allocate(output_scale_elements + kGuardScales) ||
      !candidate.allocate(output_packed_bytes + kGuardBytes) ||
      !candidate_scales.allocate(output_scale_elements + kGuardScales)) {
    std::cerr << "device allocation failed for K=" << k << '\n';
    return false;
  }
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
      !cuda_ok(cudaMemset(baseline.get(), 0xa5,
                          output_packed_bytes + kGuardBytes),
               "initialize baseline") ||
      !cuda_ok(cudaMemset(candidate.get(), 0xa5,
                          output_packed_bytes + kGuardBytes),
               "initialize candidate") ||
      !cuda_ok(cudaMemset(baseline_scales.get(), 0xad,
                          (output_scale_elements + kGuardScales) *
                              sizeof(std::uint16_t)),
               "initialize baseline scales") ||
      !cuda_ok(cudaMemset(candidate_scales.get(), 0xad,
                          (output_scale_elements + kGuardScales) *
                              sizeof(std::uint16_t)),
               "initialize candidate scales")) {
    return false;
  }

  const int short_capacity_status =
      kernels::launch_sm87_a4w4_gateup_projection_v3_cuda(
          a.get(), payload.a.size() - 1U, a_scales.get(),
          payload.a_scales.size(), gate.get(), payload.gate.size(),
          gate_scales.get(), payload.gate_scales.size(), up.get(),
          payload.up.size(), up_scales.get(), payload.up_scales.size(), kM,
          kN, k, kClipRatio, candidate.get(), output_packed_bytes,
          candidate_scales.get(), output_scale_elements);
  if (short_capacity_status != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "short-capacity launch did not fail closed for K=" << k
              << '\n';
    return false;
  }

  if (!launch_ok(kernels::launch_sm87_a4w4_gateup_paired_k128_cuda(
                     a.get(), payload.a.size(), a_scales.get(),
                     payload.a_scales.size(), gate.get(), payload.gate.size(),
                     gate_scales.get(), payload.gate_scales.size(), up.get(),
                     payload.up.size(), up_scales.get(),
                     payload.up_scales.size(), kM, kN, k, kClipRatio,
                     baseline.get(), output_packed_bytes,
                     baseline_scales.get(), output_scale_elements),
                 "launch established K128 baseline") ||
      !launch_ok(kernels::launch_sm87_a4w4_gateup_projection_v3_cuda(
                     a.get(), payload.a.size(), a_scales.get(),
                     payload.a_scales.size(), gate.get(), payload.gate.size(),
                     gate_scales.get(), payload.gate_scales.size(), up.get(),
                     payload.up.size(), up_scales.get(),
                     payload.up_scales.size(), kM, kN, k, kClipRatio,
                     candidate.get(), output_packed_bytes,
                     candidate_scales.get(), output_scale_elements),
                 "launch Gate+Up projection v3") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize kernels")) {
    return false;
  }

  std::vector<std::uint8_t> baseline_host(
      output_packed_bytes + kGuardBytes);
  std::vector<std::uint8_t> candidate_host(
      output_packed_bytes + kGuardBytes);
  std::vector<std::uint16_t> baseline_scale_host(
      output_scale_elements + kGuardScales);
  std::vector<std::uint16_t> candidate_scale_host(
      output_scale_elements + kGuardScales);
  if (!cuda_ok(cudaMemcpy(baseline_host.data(), baseline.get(),
                          baseline_host.size(), cudaMemcpyDeviceToHost),
               "copy baseline output") ||
      !cuda_ok(cudaMemcpy(candidate_host.data(), candidate.get(),
                          candidate_host.size(), cudaMemcpyDeviceToHost),
               "copy candidate output") ||
      !cuda_ok(cudaMemcpy(baseline_scale_host.data(), baseline_scales.get(),
                          baseline_scale_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "copy baseline scales") ||
      !cuda_ok(cudaMemcpy(candidate_scale_host.data(), candidate_scales.get(),
                          candidate_scale_host.size() *
                              sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "copy candidate scales")) {
    return false;
  }
  for (std::size_t i = output_packed_bytes; i < baseline_host.size(); ++i) {
    if (baseline_host[i] != 0xa5U || candidate_host[i] != 0xa5U) {
      std::cerr << "packed guard modified for K=" << k << " at " << i
                << '\n';
      return false;
    }
  }
  for (std::size_t i = output_scale_elements;
       i < baseline_scale_host.size(); ++i) {
    if (baseline_scale_host[i] != 0xadadU ||
        candidate_scale_host[i] != 0xadadU) {
      std::cerr << "scale guard modified for K=" << k << " at " << i
                << '\n';
      return false;
    }
  }
  if (baseline_host != candidate_host ||
      baseline_scale_host != candidate_scale_host) {
    std::size_t code_mismatches = 0U;
    std::size_t scale_mismatches = 0U;
    for (std::size_t i = 0U; i < baseline_host.size(); ++i) {
      code_mismatches += baseline_host[i] != candidate_host[i] ? 1U : 0U;
    }
    for (std::size_t i = 0U; i < baseline_scale_host.size(); ++i) {
      scale_mismatches +=
          baseline_scale_host[i] != candidate_scale_host[i] ? 1U : 0U;
    }
    std::cerr << "bit mismatch for K=" << k << ": packed="
              << code_mismatches << " scales=" << scale_mismatches << '\n';
    return false;
  }
  std::cout << "Gate+Up projection v3 bit-exact for K=" << k << '\n';
  return true;
}

}  // namespace

int main() {
  const int status = target_status();
  if (status != 0) {
    return status;
  }

  kernels::Sm87A4W4GateUpProjectionV3Resources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_gateup_projection_v3_resources_cuda(
              &resources),
          "query v3 resources") ||
      resources.registers_per_thread > 128 ||
      resources.static_shared_bytes != 42'240U ||
      resources.dynamic_shared_bytes != 0U || resources.local_bytes != 0U ||
      resources.active_blocks_per_sm < 2 ||
      resources.maximum_threads_per_block < 256) {
    std::cerr << "v3 resource gate failed: regs="
              << resources.registers_per_thread
              << " shared=" << resources.static_shared_bytes
              << " local=" << resources.local_bytes
              << " active=" << resources.active_blocks_per_sm << '\n';
    return 1;
  }

  constexpr std::array<std::size_t, 4U> cases = {
      128U, 256U, 384U, 512U};
  for (const std::size_t k : cases) {
    if (!run_case(k)) {
      return 1;
    }
  }
  std::cout << "Gate+Up projection v3 resource gate: registers="
            << resources.registers_per_thread
            << " shared=" << resources.static_shared_bytes
            << " local=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return 0;
}
