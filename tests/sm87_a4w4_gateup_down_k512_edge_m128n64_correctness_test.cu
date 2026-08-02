#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m128n64.h"
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

inline constexpr std::size_t kLogicalM = 129U;
inline constexpr std::size_t kLaunchM = 256U;
// Three K512 groups force the two shared scale slots to wrap back to slot 0.
// This covers the first overwrite/consume lifetime that a two-group case
// cannot exercise while keeping the synthetic admission footprint small.
inline constexpr std::size_t kInputK = 1'536U;
inline constexpr std::size_t kIntermediateN = 1'024U;
inline constexpr std::size_t kDownN = 128U;
inline constexpr std::size_t kDownStride = kDownN + 8U;
inline constexpr unsigned int kCandidateCtas = 3U;
inline constexpr float kClipRatio = 0.9375F;
inline constexpr std::size_t kGuardBytes = 64U;
inline constexpr std::size_t kGuardElements = 64U;

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
    std::cout << "SKIP: CUDA device unavailable\n";
    return 77;
  }
  if (!cuda_ok(status, "cudaGetDeviceCount")) {
    return 1;
  }
  int device = -1;
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice") ||
      !cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16 ||
      properties.sharedMemPerBlockOptin <
          kernels::kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes) {
    std::cout << "SKIP: requires 16-SM SM87 with >=164864 B opt-in shared\n";
    return 77;
  }
  return 0;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] int code(const std::size_t index,
                       const std::uint32_t salt) noexcept {
  std::uint32_t value = static_cast<std::uint32_t>(index) ^ salt;
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  return static_cast<int>(value % 15U) - 7;
}

void fill_packed(std::vector<std::uint8_t>& values,
                 const std::uint32_t salt) {
  for (std::size_t index = 0U; index < values.size(); ++index) {
    values[index] = kernels::sm87_a4w4_pack_signed_pair(
        code(2U * index, salt), code(2U * index + 1U, salt));
  }
}

void fill_scales(std::vector<std::uint16_t>& values,
                 const std::uint32_t salt) {
  for (std::size_t index = 0U; index < values.size(); ++index) {
    values[index] = encode_bf16(
        0.001F * static_cast<float>(1U + ((index + salt) % 53U)));
  }
}

template <typename T>
[[nodiscard]] bool copy_to_device(DeviceBuffer<T>& destination,
                                  const std::vector<T>& source,
                                  const std::string& label) {
  return destination.allocate(source.size()) &&
         cuda_ok(cudaMemcpy(destination.get(), source.data(),
                            source.size() * sizeof(T),
                            cudaMemcpyHostToDevice),
                 label);
}

template <typename T>
[[nodiscard]] bool compare(const std::vector<T>& expected,
                           const std::vector<T>& actual,
                           const std::string& label) {
  if (expected == actual) {
    return true;
  }
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    if (expected[index] != actual[index]) {
      std::cerr << label << ": first mismatch at " << index
                << ", expected="
                << static_cast<unsigned long long>(expected[index])
                << ", actual="
                << static_cast<unsigned long long>(actual[index]) << '\n';
      break;
    }
  }
  return false;
}

template <typename T>
[[nodiscard]] bool guard_is(const std::vector<T>& values,
                            const std::size_t payload_elements,
                            const T sentinel, const std::string& label) {
  for (std::size_t index = payload_elements; index < values.size();
       ++index) {
    if (values[index] != sentinel) {
      std::cerr << label << ": guard changed at "
                << index - payload_elements << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool run_case() {
  const std::size_t a_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          kLaunchM, kInputK);
  const std::size_t a_scale_elements =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          kLaunchM, kInputK);
  const std::size_t b_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          kIntermediateN, kInputK);
  const std::size_t b_scale_elements =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          kIntermediateN, kInputK);
  const std::size_t output_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          kLaunchM, kIntermediateN);
  const std::size_t output_scale_elements =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          kLaunchM, kIntermediateN);
  const std::size_t down_weight_bytes =
      kernels::sm87_a4w4_down_k512_packed_capacity_bytes(
          kDownN, kIntermediateN);
  const std::size_t down_scale_elements =
      kernels::sm87_a4w4_down_k512_scale_capacity_elements(
          kDownN, kIntermediateN);
  const std::size_t down_elements = kLaunchM * kDownStride;

  std::vector<std::uint8_t> host_a(a_bytes);
  std::vector<std::uint16_t> host_a_scales(a_scale_elements);
  std::vector<std::uint8_t> host_gate(b_bytes);
  std::vector<std::uint16_t> host_gate_scales(b_scale_elements);
  std::vector<std::uint8_t> host_up(b_bytes);
  std::vector<std::uint16_t> host_up_scales(b_scale_elements);
  std::vector<std::uint8_t> host_down_weight(down_weight_bytes);
  std::vector<std::uint16_t> host_down_scales(down_scale_elements);
  fill_packed(host_a, 0x1234U);
  fill_scales(host_a_scales, 3U);
  fill_packed(host_gate, 0x4567U);
  fill_scales(host_gate_scales, 7U);
  fill_packed(host_up, 0x89abU);
  fill_scales(host_up_scales, 11U);
  fill_packed(host_down_weight, 0xcdefU);
  fill_scales(host_down_scales, 17U);

  DeviceBuffer<std::uint8_t> a;
  DeviceBuffer<std::uint16_t> a_scales;
  DeviceBuffer<std::uint8_t> gate;
  DeviceBuffer<std::uint16_t> gate_scales;
  DeviceBuffer<std::uint8_t> up;
  DeviceBuffer<std::uint16_t> up_scales;
  DeviceBuffer<std::uint8_t> baseline_packed;
  DeviceBuffer<std::uint16_t> baseline_scales;
  DeviceBuffer<std::uint8_t> candidate_packed;
  DeviceBuffer<std::uint16_t> candidate_scales;
  DeviceBuffer<std::uint8_t> down_weight;
  DeviceBuffer<std::uint16_t> down_scales;
  DeviceBuffer<std::uint16_t> baseline_down;
  DeviceBuffer<std::uint16_t> candidate_down;
  if (!copy_to_device(a, host_a, "copy A") ||
      !copy_to_device(a_scales, host_a_scales, "copy A scales") ||
      !copy_to_device(gate, host_gate, "copy Gate") ||
      !copy_to_device(gate_scales, host_gate_scales,
                      "copy Gate scales") ||
      !copy_to_device(up, host_up, "copy Up") ||
      !copy_to_device(up_scales, host_up_scales, "copy Up scales") ||
      !copy_to_device(down_weight, host_down_weight, "copy Down weight") ||
      !copy_to_device(down_scales, host_down_scales, "copy Down scales") ||
      !baseline_packed.allocate(output_bytes + kGuardBytes) ||
      !candidate_packed.allocate(output_bytes + kGuardBytes) ||
      !baseline_scales.allocate(output_scale_elements + kGuardElements) ||
      !candidate_scales.allocate(output_scale_elements + kGuardElements) ||
      !baseline_down.allocate(down_elements + kGuardElements) ||
      !candidate_down.allocate(down_elements + kGuardElements)) {
    std::cerr << "device allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemset(baseline_packed.get(), 0xa5,
                          output_bytes + kGuardBytes),
               "initialize baseline codes") ||
      !cuda_ok(cudaMemset(candidate_packed.get(), 0xa5,
                          output_bytes + kGuardBytes),
               "initialize candidate codes") ||
      !cuda_ok(cudaMemset(baseline_scales.get(), 0xad,
                          (output_scale_elements + kGuardElements) *
                              sizeof(std::uint16_t)),
               "initialize baseline scales") ||
      !cuda_ok(cudaMemset(candidate_scales.get(), 0xad,
                          (output_scale_elements + kGuardElements) *
                              sizeof(std::uint16_t)),
               "initialize candidate scales") ||
      !cuda_ok(cudaMemset(baseline_down.get(), 0x5a,
                          (down_elements + kGuardElements) *
                              sizeof(std::uint16_t)),
               "initialize baseline Down") ||
      !cuda_ok(cudaMemset(candidate_down.get(), 0x5a,
                          (down_elements + kGuardElements) *
                              sizeof(std::uint16_t)),
               "initialize candidate Down")) {
    return false;
  }

  kernels::Sm87A4W4GateUpDownEdgeM128N64Resources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_gateup_down_k512_edge_m128n64_resources_cuda(
              &resources),
          "query M128N64 resources") ||
      resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 128 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 164'864U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 512 ||
      resources.active_blocks_per_sm != 1) {
    std::cerr << "resource gate failed: regs="
              << resources.registers_per_thread
              << " shared=" << resources.dynamic_shared_bytes
              << " local=" << resources.local_bytes
              << " active=" << resources.active_blocks_per_sm << '\n';
    return false;
  }

  const int short_output =
      kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n64_test_cuda(
          a.get(), a_bytes, a_scales.get(), a_scale_elements,
          gate.get(), b_bytes, gate_scales.get(), b_scale_elements,
          up.get(), b_bytes, up_scales.get(), b_scale_elements,
          kLogicalM, kLaunchM, kIntermediateN, kInputK, kClipRatio,
          candidate_packed.get(), output_bytes - 1U,
          candidate_scales.get(), output_scale_elements, kCandidateCtas);
  if (short_output != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "short output capacity was accepted\n";
    return false;
  }

  if (!launch_ok(
          kernels::launch_sm87_a4w4_gateup_down_k512_edge_test_cuda(
              a.get(), a_bytes, a_scales.get(), a_scale_elements,
              gate.get(), b_bytes, gate_scales.get(), b_scale_elements,
              up.get(), b_bytes, up_scales.get(), b_scale_elements,
              kLogicalM, kLaunchM, kIntermediateN, kInputK, kClipRatio,
              baseline_packed.get(), output_bytes, baseline_scales.get(),
              output_scale_elements, kCandidateCtas),
          "launch retained edge") ||
      !launch_ok(
          kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n64_test_cuda(
              a.get(), a_bytes, a_scales.get(), a_scale_elements,
              gate.get(), b_bytes, gate_scales.get(), b_scale_elements,
              up.get(), b_bytes, up_scales.get(), b_scale_elements,
              kLogicalM, kLaunchM, kIntermediateN, kInputK, kClipRatio,
              candidate_packed.get(), output_bytes,
              candidate_scales.get(), output_scale_elements,
              kCandidateCtas),
          "launch M128N64 edge") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize edge producers")) {
    return false;
  }

  std::vector<std::uint8_t> baseline_codes(output_bytes + kGuardBytes);
  std::vector<std::uint8_t> candidate_codes(output_bytes + kGuardBytes);
  std::vector<std::uint16_t> baseline_scale_words(
      output_scale_elements + kGuardElements);
  std::vector<std::uint16_t> candidate_scale_words(
      output_scale_elements + kGuardElements);
  if (!cuda_ok(cudaMemcpy(baseline_codes.data(), baseline_packed.get(),
                          baseline_codes.size(), cudaMemcpyDeviceToHost),
               "copy baseline codes") ||
      !cuda_ok(cudaMemcpy(candidate_codes.data(), candidate_packed.get(),
                          candidate_codes.size(), cudaMemcpyDeviceToHost),
               "copy candidate codes") ||
      !cuda_ok(cudaMemcpy(baseline_scale_words.data(), baseline_scales.get(),
                          baseline_scale_words.size() *
                              sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy baseline scales") ||
      !cuda_ok(cudaMemcpy(candidate_scale_words.data(), candidate_scales.get(),
                          candidate_scale_words.size() *
                              sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy candidate scales") ||
      !compare(baseline_codes, candidate_codes, "packed K512 edge") ||
      !compare(baseline_scale_words, candidate_scale_words,
               "BF16 K512 scales") ||
      !guard_is(baseline_codes, output_bytes,
                static_cast<std::uint8_t>(0xa5U),
                "retained edge codes") ||
      !guard_is(candidate_codes, output_bytes,
                static_cast<std::uint8_t>(0xa5U),
                "M128N64 edge codes") ||
      !guard_is(baseline_scale_words, output_scale_elements,
                static_cast<std::uint16_t>(0xadadU),
                "retained edge scales") ||
      !guard_is(candidate_scale_words, output_scale_elements,
                static_cast<std::uint16_t>(0xadadU),
                "M128N64 edge scales")) {
    return false;
  }

  if (!launch_ok(
          kernels::launch_sm87_a4w4_down_k512_macrocell_test_bf16_cuda(
              baseline_packed.get(), output_bytes, baseline_scales.get(),
              output_scale_elements, down_weight.get(), down_weight_bytes,
              down_scales.get(), down_scale_elements, kLaunchM, kDownN,
              kIntermediateN, baseline_down.get(), kDownStride,
              down_elements, 2U),
          "launch baseline Down") ||
      !launch_ok(
          kernels::launch_sm87_a4w4_down_k512_macrocell_test_bf16_cuda(
              candidate_packed.get(), output_bytes, candidate_scales.get(),
              output_scale_elements, down_weight.get(), down_weight_bytes,
              down_scales.get(), down_scale_elements, kLaunchM, kDownN,
              kIntermediateN, candidate_down.get(), kDownStride,
              down_elements, 2U),
          "launch candidate Down") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize Down")) {
    return false;
  }
  std::vector<std::uint16_t> baseline_down_host(
      down_elements + kGuardElements);
  std::vector<std::uint16_t> candidate_down_host(
      down_elements + kGuardElements);
  if (!cuda_ok(cudaMemcpy(baseline_down_host.data(), baseline_down.get(),
                          baseline_down_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy baseline Down") ||
      !cuda_ok(cudaMemcpy(candidate_down_host.data(), candidate_down.get(),
                          candidate_down_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy candidate Down") ||
      !compare(baseline_down_host, candidate_down_host,
               "Down BF16 output") ||
      !guard_is(baseline_down_host, down_elements,
                static_cast<std::uint16_t>(0x5a5aU),
                "retained Down") ||
      !guard_is(candidate_down_host, down_elements,
                static_cast<std::uint16_t>(0x5a5aU),
                "candidate Down")) {
    return false;
  }

  std::cout << "M128N64 edge bit-exact: logicalM=" << kLogicalM
            << " launchM=" << kLaunchM << " N=" << kIntermediateN
            << " K=" << kInputK
            << " regs=" << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  std::cout << "Down K512 consumer output bit-exact\n";
  return true;
}

}  // namespace

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }
  return run_case() ? 0 : 1;
}
