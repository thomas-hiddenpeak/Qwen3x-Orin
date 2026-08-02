#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"
#include "q3x/kernels/sm87_a4w4_gateup_k512_macrocell.h"
#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"
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
inline constexpr std::size_t kInputK = 1'024U;
inline constexpr std::size_t kIntermediateN = 1'024U;
inline constexpr std::size_t kPrimaryN = 512U;
inline constexpr std::size_t kSecondaryN = 512U;
inline constexpr std::size_t kDownN = 128U;
inline constexpr std::size_t kPrimaryStride = kPrimaryN + 8U;
inline constexpr std::size_t kSecondaryStride = kSecondaryN + 8U;
inline constexpr std::size_t kDownStride = kDownN + 8U;
inline constexpr unsigned int kCandidateCtas = 3U;
inline constexpr float kClipRatio = 0.9375F;
inline constexpr std::size_t kGuardBytes = 64U;
inline constexpr std::size_t kGuardElements = 64U;
inline constexpr std::uint16_t kBf16Sentinel = 0x7fc1U;

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

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits{};
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
          kernels::kSm87A4W4GateUpDownEdgeDynamicSharedBytes) {
    std::cout << "SKIP: requires 16-SM SM87 with >=148736 B opt-in shared\n";
    return 77;
  }
  return 0;
}

[[nodiscard]] std::int8_t code(const std::size_t outer,
                               const std::size_t inner,
                               const std::uint32_t salt) noexcept {
  std::uint32_t mixed =
      static_cast<std::uint32_t>(outer * (0x9e3779b9U ^ salt)) ^
      static_cast<std::uint32_t>(inner * (0x85ebca6bU + salt));
  mixed ^= mixed >> 16U;
  mixed *= 0x7feb352dU;
  mixed ^= mixed >> 15U;
  return static_cast<std::int8_t>(static_cast<int>(mixed % 15U) - 7);
}

struct GateUpPayload final {
  std::vector<std::uint8_t> a;
  std::vector<std::uint16_t> a_scales;
  std::vector<std::uint8_t> gate;
  std::vector<std::uint16_t> gate_scales;
  std::vector<std::uint8_t> up;
  std::vector<std::uint16_t> up_scales;
};

[[nodiscard]] GateUpPayload make_gateup_payload() {
  constexpr std::size_t physical_groups = kInputK / 64U;
  constexpr std::size_t k512_groups = kInputK / 512U;
  GateUpPayload result{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(
              kLaunchM, kInputK)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
              kLaunchM, kInputK)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(
              kIntermediateN, kInputK)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
              kIntermediateN, kInputK)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(
              kIntermediateN, kInputK)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
              kIntermediateN, kInputK))};

  // Tail A rows are deliberately nonzero.  The candidate must still publish
  // canonical zero codes / BF16-one scales for rows >= logical M, exactly as
  // the established split quantizer does without reading those BF16 rows.
  for (std::size_t row = 0U; row < kLaunchM; ++row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = group * 64U + 2U * byte;
        result.a[kernels::sm87_a4w4_consumer_packed_offset(
            row, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                code(row, inner, 0x1234U),
                code(row, inner + 1U, 0x1234U));
      }
    }
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      result.a_scales[
          kernels::sm87_a4w4_gateup_down_edge_scale_offset(
              row, group, k512_groups)] =
          encode_bf16(0.0021F *
                      static_cast<float>(5U + (3U * row + group) % 17U));
    }
  }

  for (std::size_t row = 0U; row < kIntermediateN; ++row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = group * 64U + 2U * byte;
        const std::size_t offset =
            kernels::sm87_a4w4_consumer_packed_offset(
                row, group, byte, physical_groups);
        result.gate[offset] = kernels::sm87_a4w4_pack_signed_pair(
            code(row, inner, 0x4567U),
            code(row, inner + 1U, 0x4567U));
        result.up[offset] = kernels::sm87_a4w4_pack_signed_pair(
            code(row, inner, 0x89abU),
            code(row, inner + 1U, 0x89abU));
      }
    }
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      const std::size_t offset =
          kernels::sm87_a4w4_gateup_down_edge_scale_offset(
              row, group, k512_groups);
      result.gate_scales[offset] = encode_bf16(
          0.0017F *
          static_cast<float>(7U + (5U * row + 3U * group) % 19U));
      result.up_scales[offset] = encode_bf16(
          0.0013F *
          static_cast<float>(9U + (7U * row + group) % 23U));
    }
  }
  return result;
}

struct DownPayload final {
  std::vector<std::uint8_t> weight;
  std::vector<std::uint16_t> scales;
};

[[nodiscard]] DownPayload make_down_payload() {
  constexpr std::size_t physical_groups = kIntermediateN / 64U;
  constexpr std::size_t k512_groups = kIntermediateN / 512U;
  DownPayload result{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_down_k512_packed_capacity_bytes(
              kDownN, kIntermediateN)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_down_k512_scale_capacity_elements(
              kDownN, kIntermediateN))};
  for (std::size_t row = 0U; row < kDownN; ++row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = group * 64U + 2U * byte;
        result.weight[kernels::sm87_a4w4_down_k512_packed_offset(
            row, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                code(row, inner, 0xcdefU),
                code(row, inner + 1U, 0xcdefU));
      }
    }
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      result.scales[kernels::sm87_a4w4_down_k512_scale_offset(
          row, group, k512_groups)] =
          encode_bf16(0.0023F *
                      static_cast<float>(7U + (7U * row + group) % 23U));
    }
  }
  return result;
}

template <typename T>
[[nodiscard]] bool copy_to_device(DeviceBuffer<T>& destination,
                                  const std::vector<T>& source,
                                  const std::string& label) {
  return destination.allocate(source.size()) &&
         cuda_ok(cudaMemcpy(destination.get(), source.data(),
                            source.size() * sizeof(T),
                            cudaMemcpyHostToDevice), label);
}

[[nodiscard]] bool compare_bytes(const std::vector<std::uint8_t>& expected,
                                 const std::vector<std::uint8_t>& actual,
                                 const std::string& label) {
  if (expected == actual) {
    return true;
  }
  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    if (expected[index] != actual[index]) {
      if (mismatches == 0U) {
        std::cerr << label << ": first mismatch at byte " << index
                  << ", expected 0x" << std::hex
                  << static_cast<unsigned int>(expected[index])
                  << ", got 0x"
                  << static_cast<unsigned int>(actual[index])
                  << std::dec << '\n';
      }
      ++mismatches;
    }
  }
  std::cerr << label << ": mismatches=" << mismatches << '\n';
  return false;
}

[[nodiscard]] bool compare_words(const std::vector<std::uint16_t>& expected,
                                 const std::vector<std::uint16_t>& actual,
                                 const std::string& label) {
  if (expected == actual) {
    return true;
  }
  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    if (expected[index] != actual[index]) {
      if (mismatches == 0U) {
        std::cerr << label << ": first mismatch at element " << index
                  << ", expected 0x" << std::hex << expected[index]
                  << ", got 0x" << actual[index] << std::dec << '\n';
      }
      ++mismatches;
    }
  }
  std::cerr << label << ": mismatches=" << mismatches << '\n';
  return false;
}

[[nodiscard]] bool run_case() {
  const GateUpPayload payload = make_gateup_payload();
  const DownPayload down_payload = make_down_payload();
  constexpr std::size_t output_packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          kLaunchM, kIntermediateN);
  constexpr std::size_t output_scale_elements =
      kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(
          kLaunchM, kIntermediateN);
  constexpr std::size_t primary_elements = kLaunchM * kPrimaryStride;
  constexpr std::size_t secondary_elements = kLaunchM * kSecondaryStride;
  constexpr std::size_t down_elements = kLaunchM * kDownStride;

  DeviceBuffer<std::uint8_t> a;
  DeviceBuffer<std::uint16_t> a_scales;
  DeviceBuffer<std::uint8_t> gate;
  DeviceBuffer<std::uint16_t> gate_scales;
  DeviceBuffer<std::uint8_t> up;
  DeviceBuffer<std::uint16_t> up_scales;
  DeviceBuffer<std::uint16_t> primary_bf16;
  DeviceBuffer<std::uint16_t> secondary_bf16;
  DeviceBuffer<std::uint8_t> baseline_packed;
  DeviceBuffer<std::uint16_t> baseline_scales;
  DeviceBuffer<std::uint8_t> candidate_packed;
  DeviceBuffer<std::uint16_t> candidate_scales;
  DeviceBuffer<std::uint8_t> down_weight;
  DeviceBuffer<std::uint16_t> down_scales;
  DeviceBuffer<std::uint16_t> baseline_down;
  DeviceBuffer<std::uint16_t> candidate_down;
  if (!copy_to_device(a, payload.a, "copy A") ||
      !copy_to_device(a_scales, payload.a_scales, "copy A scales") ||
      !copy_to_device(gate, payload.gate, "copy Gate") ||
      !copy_to_device(gate_scales, payload.gate_scales,
                      "copy Gate scales") ||
      !copy_to_device(up, payload.up, "copy Up") ||
      !copy_to_device(up_scales, payload.up_scales, "copy Up scales") ||
      !copy_to_device(down_weight, down_payload.weight,
                      "copy Down weight") ||
      !copy_to_device(down_scales, down_payload.scales,
                      "copy Down scales") ||
      !primary_bf16.allocate(primary_elements) ||
      !secondary_bf16.allocate(secondary_elements) ||
      !baseline_packed.allocate(output_packed_bytes + kGuardBytes) ||
      !candidate_packed.allocate(output_packed_bytes + kGuardBytes) ||
      !baseline_scales.allocate(output_scale_elements + kGuardElements) ||
      !candidate_scales.allocate(output_scale_elements + kGuardElements) ||
      !baseline_down.allocate(down_elements + kGuardElements) ||
      !candidate_down.allocate(down_elements + kGuardElements)) {
    std::cerr << "device allocation failed\n";
    return false;
  }

  std::vector<std::uint16_t> bf16_guard(
      down_elements + kGuardElements, kBf16Sentinel);
  if (!cuda_ok(cudaMemset(baseline_packed.get(), 0xa5,
                          output_packed_bytes + kGuardBytes),
               "initialize baseline packed") ||
      !cuda_ok(cudaMemset(candidate_packed.get(), 0xa5,
                          output_packed_bytes + kGuardBytes),
               "initialize candidate packed") ||
      !cuda_ok(cudaMemset(baseline_scales.get(), 0xad,
                          (output_scale_elements + kGuardElements) *
                              sizeof(std::uint16_t)),
               "initialize baseline scales") ||
      !cuda_ok(cudaMemset(candidate_scales.get(), 0xad,
                          (output_scale_elements + kGuardElements) *
                              sizeof(std::uint16_t)),
               "initialize candidate scales") ||
      !cuda_ok(cudaMemcpy(baseline_down.get(), bf16_guard.data(),
                          bf16_guard.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "initialize baseline Down") ||
      !cuda_ok(cudaMemcpy(candidate_down.get(), bf16_guard.data(),
                          bf16_guard.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "initialize candidate Down")) {
    return false;
  }

  kernels::Sm87A4W4GateUpDownEdgeResources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_gateup_down_k512_edge_resources_cuda(
              &resources),
          "query edge resources") ||
      resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 128 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 148'736U ||
      resources.configured_dynamic_shared_limit_bytes < 148'736U ||
      resources.device_optin_shared_limit_bytes < 148'736U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 512 ||
      resources.active_blocks_per_sm != 1) {
    std::cerr << "edge resource gate failed: regs="
              << resources.registers_per_thread
              << " dynamic=" << resources.dynamic_shared_bytes
              << " local=" << resources.local_bytes
              << " active=" << resources.active_blocks_per_sm << '\n';
    return false;
  }

  const int short_output =
      kernels::launch_sm87_a4w4_gateup_down_k512_edge_test_cuda(
          a.get(), payload.a.size(), a_scales.get(),
          payload.a_scales.size(), gate.get(), payload.gate.size(),
          gate_scales.get(), payload.gate_scales.size(), up.get(),
          payload.up.size(), up_scales.get(), payload.up_scales.size(),
          kLogicalM, kLaunchM, kIntermediateN, kInputK, kClipRatio,
          candidate_packed.get(), output_packed_bytes - 1U,
          candidate_scales.get(), output_scale_elements, kCandidateCtas);
  if (short_output != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "short edge output capacity was accepted\n";
    return false;
  }

  if (!launch_ok(
          kernels::launch_sm87_a4w4_gateup_k512_macrocell_test_bf16_cuda(
              a.get(), payload.a.size(), a_scales.get(),
              payload.a_scales.size(), gate.get(), payload.gate.size(),
              gate_scales.get(), payload.gate_scales.size(), up.get(),
              payload.up.size(), up_scales.get(), payload.up_scales.size(),
              kLaunchM, kIntermediateN, kInputK, 0U, kPrimaryN,
              primary_bf16.get(), kPrimaryStride, primary_elements,
              kCandidateCtas),
          "launch baseline primary GateUp") ||
      !launch_ok(
          kernels::launch_sm87_a4w4_gateup_k512_macrocell_test_bf16_cuda(
              a.get(), payload.a.size(), a_scales.get(),
              payload.a_scales.size(), gate.get(), payload.gate.size(),
              gate_scales.get(), payload.gate_scales.size(), up.get(),
              payload.up.size(), up_scales.get(), payload.up_scales.size(),
              kLaunchM, kIntermediateN, kInputK, kPrimaryN, kSecondaryN,
              secondary_bf16.get(), kSecondaryStride, secondary_elements,
              kCandidateCtas),
          "launch baseline secondary GateUp") ||
      !launch_ok(
          kernels::launch_sm87_a4_quantize_bf16_k512_split_cuda(
              primary_bf16.get(), kPrimaryStride, kPrimaryN,
              secondary_bf16.get(), kSecondaryStride, kSecondaryN,
              kLogicalM, kLaunchM, kClipRatio, baseline_packed.get(),
              output_packed_bytes, baseline_scales.get(),
              output_scale_elements),
          "launch baseline split quantizer") ||
      !launch_ok(
          kernels::launch_sm87_a4w4_gateup_down_k512_edge_test_cuda(
              a.get(), payload.a.size(), a_scales.get(),
              payload.a_scales.size(), gate.get(), payload.gate.size(),
              gate_scales.get(), payload.gate_scales.size(), up.get(),
              payload.up.size(), up_scales.get(), payload.up_scales.size(),
              kLogicalM, kLaunchM, kIntermediateN, kInputK, kClipRatio,
              candidate_packed.get(), output_packed_bytes,
              candidate_scales.get(), output_scale_elements,
              kCandidateCtas),
          "launch GateUp->Down edge") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize GateUp producers")) {
    return false;
  }

  std::vector<std::uint8_t> baseline_packed_host(
      output_packed_bytes + kGuardBytes);
  std::vector<std::uint8_t> candidate_packed_host(
      output_packed_bytes + kGuardBytes);
  std::vector<std::uint16_t> baseline_scale_host(
      output_scale_elements + kGuardElements);
  std::vector<std::uint16_t> candidate_scale_host(
      output_scale_elements + kGuardElements);
  if (!cuda_ok(cudaMemcpy(baseline_packed_host.data(),
                          baseline_packed.get(),
                          baseline_packed_host.size(),
                          cudaMemcpyDeviceToHost),
               "copy baseline packed") ||
      !cuda_ok(cudaMemcpy(candidate_packed_host.data(),
                          candidate_packed.get(),
                          candidate_packed_host.size(),
                          cudaMemcpyDeviceToHost),
               "copy candidate packed") ||
      !cuda_ok(cudaMemcpy(baseline_scale_host.data(),
                          baseline_scales.get(),
                          baseline_scale_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy baseline scales") ||
      !cuda_ok(cudaMemcpy(candidate_scale_host.data(),
                          candidate_scales.get(),
                          candidate_scale_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy candidate scales")) {
    return false;
  }
  if (!compare_bytes(baseline_packed_host, candidate_packed_host,
                     "packed K512 edge") ||
      !compare_words(baseline_scale_host, candidate_scale_host,
                     "BF16 K512 edge scales")) {
    return false;
  }

  if (!launch_ok(
          kernels::launch_sm87_a4w4_down_k512_macrocell_test_bf16_cuda(
              baseline_packed.get(), output_packed_bytes,
              baseline_scales.get(), output_scale_elements,
              down_weight.get(), down_payload.weight.size(),
              down_scales.get(), down_payload.scales.size(), kLaunchM,
              kDownN, kIntermediateN, baseline_down.get(), kDownStride,
              down_elements, 2U),
          "launch baseline Down") ||
      !launch_ok(
          kernels::launch_sm87_a4w4_down_k512_macrocell_test_bf16_cuda(
              candidate_packed.get(), output_packed_bytes,
              candidate_scales.get(), output_scale_elements,
              down_weight.get(), down_payload.weight.size(),
              down_scales.get(), down_payload.scales.size(), kLaunchM,
              kDownN, kIntermediateN, candidate_down.get(), kDownStride,
              down_elements, 2U),
          "launch candidate Down") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize Down")) {
    return false;
  }

  std::vector<std::uint16_t> baseline_down_host(bf16_guard.size());
  std::vector<std::uint16_t> candidate_down_host(bf16_guard.size());
  if (!cuda_ok(cudaMemcpy(baseline_down_host.data(), baseline_down.get(),
                          baseline_down_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy baseline Down") ||
      !cuda_ok(cudaMemcpy(candidate_down_host.data(), candidate_down.get(),
                          candidate_down_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy candidate Down") ||
      !compare_words(baseline_down_host, candidate_down_host,
                     "Down BF16 output")) {
    return false;
  }

  std::cout << "GateUp->K512 edge bit-exact against macrocell+split: "
            << "logicalM=" << kLogicalM << " launchM=" << kLaunchM
            << " N=" << kIntermediateN << " K=" << kInputK
            << " regs=" << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
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
