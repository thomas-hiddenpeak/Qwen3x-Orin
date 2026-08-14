#include "q3x/kernels/sm87_macrofeed_v3_fp8.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

namespace kernels = q3x::kernels;
using Role = kernels::Sm87TargetAotProjectionRole;

constexpr std::size_t kRows = kernels::kSm87MacroFeedV3Fp8BlockM;
constexpr std::size_t kColumns = kernels::kSm87MacroFeedV3Fp8BlockN;
constexpr std::size_t kInputFeatures =
    kernels::kSm87MacroFeedV3Fp8TestInputFeatures;
constexpr std::size_t kGuardElements = 16U;
constexpr std::uint16_t kGuardBits = 0x5a5aU;
constexpr std::uint16_t kCompensatedScaleOneBits = 0x7b80U;

bool cuda_ok(const cudaError_t status, const char* const operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

[[nodiscard]] std::uint16_t encode_bf16_rne(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_fp8_marlin(const std::uint8_t code) {
  const unsigned int magnitude = code & 0x7fU;
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  float value = exponent == 0U
                    ? std::ldexp(static_cast<float>(mantissa), -9)
                    : std::ldexp(static_cast<float>(8U + mantissa),
                                 static_cast<int>(exponent) - 10);
  return (code & 0x80U) != 0U ? -value : value;
}

[[nodiscard]] float input_value(const std::size_t row,
                                const std::size_t k) {
  constexpr std::array<float, 4U> kRowsPattern{{1.0F, -1.0F, 2.0F, 0.5F}};
  constexpr std::array<float, 4U> kKPattern{{1.0F, 2.0F, 4.0F, 8.0F}};
  return kRowsPattern[row % kRowsPattern.size()] *
         kKPattern[k % kKPattern.size()];
}

[[nodiscard]] std::uint8_t logical_weight_code(
    const Role role, const std::size_t partition,
    const std::size_t n, const std::size_t k) {
  // Includes both terminal bytes in the arithmetic oracle.  All values are
  // binary rationals, keeping the 256-term FP32 sum exact at this magnitude.
  constexpr std::array<std::uint8_t, 8U> kCodes{{
      0x38U, 0x40U, 0xb8U, 0xc0U,
      0x7fU, 0xffU, 0x30U, 0xb0U,
  }};
  const std::size_t role_seed = static_cast<std::size_t>(role);
  return kCodes[(17U * n + 29U * k + 3U * partition + role_seed) %
                kCodes.size()];
}

void build_input(std::vector<std::uint16_t>& input) {
  for (std::size_t row = 0U; row < kRows; ++row) {
    for (std::size_t k = 0U; k < kInputFeatures; ++k) {
      input[row * kInputFeatures + k] =
          encode_bf16_rne(input_value(row, k));
    }
  }
}

bool build_payload(const Role role, const std::size_t partition,
                   std::vector<std::uint8_t>& payload) {
  const auto layout = kernels::sm87_target_aot_projection_packed_layout(role);
  if (!layout.valid() || partition >= layout.partition_count) {
    return false;
  }
  const auto& part = layout.partitions[partition];
  std::vector<std::uint8_t> assigned(payload.size(), 0U);
  for (std::size_t n = 0U; n < kColumns; ++n) {
    for (std::size_t k = 0U; k < kInputFeatures; ++k) {
      const auto address =
          kernels::sm87_target_aot_projection_packed_weight_address(
              layout, partition, n, k);
      if (!address.valid || address.byte_offset < part.payload_offset) {
        return false;
      }
      const std::uint64_t relative =
          address.byte_offset - part.payload_offset;
      if (relative >= payload.size() || assigned[relative] != 0U) {
        return false;
      }
      payload[relative] = logical_weight_code(role, partition, n, k);
      assigned[relative] = 1U;
    }
  }
  for (const std::uint8_t witness : assigned) {
    if (witness != 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::uint16_t expected_value(
    const Role role, const std::size_t partition,
    const std::size_t row, const std::size_t column) {
  float accumulator = 0.0F;
  for (std::size_t k = 0U; k < kInputFeatures; ++k) {
    accumulator += input_value(row, k) *
                   decode_fp8_marlin(
                       logical_weight_code(role, partition, column, k));
  }
  return encode_bf16_rne(accumulator);
}

bool run_tile_case(
    const Role role, const std::size_t partition,
    const std::size_t valid_rows,
    const std::vector<std::uint16_t>& host_input,
    const std::vector<std::uint8_t>& host_payload,
    std::uint16_t* const device_input,
    std::uint8_t* const device_payload,
    std::uint16_t* const device_storage) {
  const std::size_t body_elements = kRows * kColumns;
  const std::size_t total_elements = body_elements + 2U * kGuardElements;
  std::vector<std::uint16_t> initial(total_elements, kGuardBits);
  std::vector<std::uint16_t> first(total_elements, 0U);
  std::vector<std::uint16_t> second(total_elements, 0U);
  std::vector<std::uint16_t> input_after(host_input.size(), 0U);
  std::vector<std::uint8_t> payload_after(host_payload.size(), 0U);

  const auto launch_once = [&](std::vector<std::uint16_t>& actual) {
    if (!cuda_ok(cudaMemcpy(device_storage, initial.data(),
                            total_elements * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy output sentinel H2D")) {
      return false;
    }
    const int launch_status =
        kernels::launch_sm87_macrofeed_v3_fp8_tile_test_cuda(
            role, partition, device_input, device_payload,
            kCompensatedScaleOneBits, valid_rows,
            device_storage + kGuardElements, nullptr);
    if (launch_status != static_cast<int>(cudaSuccess)) {
      std::cerr << "tile launch: "
                << cudaGetErrorString(
                       static_cast<cudaError_t>(launch_status))
                << '\n';
      return false;
    }
    return cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize tile") &&
           cuda_ok(cudaMemcpy(actual.data(), device_storage,
                              total_elements * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy output D2H");
  };

  if (!launch_once(first) || !launch_once(second)) {
    return false;
  }
  if (first != second) {
    std::cerr << "FAIL: nondeterministic tile output role="
              << static_cast<unsigned int>(role)
              << " partition=" << partition
              << " rows=" << valid_rows << '\n';
    return false;
  }
  for (std::size_t index = 0U; index < kGuardElements; ++index) {
    if (first[index] != kGuardBits ||
        first[kGuardElements + body_elements + index] != kGuardBits) {
      std::cerr << "FAIL: output guard overwritten\n";
      return false;
    }
  }
  for (std::size_t row = 0U; row < kRows; ++row) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const std::size_t index =
          kGuardElements + row * kColumns + column;
      const std::uint16_t expected =
          row < valid_rows
              ? expected_value(role, partition, row, column)
              : kGuardBits;
      if (first[index] != expected) {
        std::cerr << "FAIL: bit mismatch role="
                  << static_cast<unsigned int>(role)
                  << " partition=" << partition
                  << " rows=" << valid_rows << " row=" << row
                  << " column=" << column << " expected=0x" << std::hex
                  << expected << " actual=0x" << first[index] << std::dec
                  << '\n';
        return false;
      }
    }
  }
  if (!cuda_ok(cudaMemcpy(input_after.data(), device_input,
                          host_input.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy immutable input D2H") ||
      !cuda_ok(cudaMemcpy(payload_after.data(), device_payload,
                          host_payload.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy immutable payload D2H")) {
    return false;
  }
  if (input_after != host_input || payload_after != host_payload) {
    std::cerr << "FAIL: input or authenticated payload mutated\n";
    return false;
  }
  return true;
}

bool run_all_codes() {
  std::array<std::uint8_t, 256U> codes{};
  for (std::size_t index = 0U; index < codes.size(); ++index) {
    codes[index] = static_cast<std::uint8_t>(index);
  }
  std::uint8_t* device_codes = nullptr;
  std::uint16_t* device_bits = nullptr;
  bool ok =
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_codes),
                         codes.size()),
              "cudaMalloc codes") &&
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_bits),
                         codes.size() * sizeof(std::uint16_t)),
              "cudaMalloc decoded bits") &&
      cuda_ok(cudaMemcpy(device_codes, codes.data(), codes.size(),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy codes H2D");
  if (ok) {
    const int status = kernels::launch_sm87_macrofeed_v3_fp8_code_test_cuda(
        device_codes, device_bits, nullptr);
    ok = status == static_cast<int>(cudaSuccess) &&
         cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize codes");
  }
  std::array<std::uint16_t, 256U> actual{};
  if (ok) {
    ok = cuda_ok(cudaMemcpy(actual.data(), device_bits,
                            actual.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy decoded bits D2H");
  }
  if (ok) {
    for (std::size_t index = 0U; index < actual.size(); ++index) {
      const auto code = static_cast<std::uint8_t>(index);
      if (!kernels::sm87_macrofeed_v3_fp8_weight_code_admitted(code) ||
          actual[index] !=
              kernels::sm87_macrofeed_v3_fp8_bias_shift_bf16_bits(code)) {
        std::cerr << "FAIL: raw code mapping index=" << index << '\n';
        ok = false;
        break;
      }
    }
    ok = ok && actual[0x7fU] == 0x07f0U &&
         actual[0xffU] == 0x87f0U;
  }
  if (device_bits != nullptr) {
    (void)cudaFree(device_bits);
  }
  if (device_codes != nullptr) {
    (void)cudaFree(device_codes);
  }
  return ok;
}

}  // namespace

int main() {
  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA device unavailable\n";
    return 77;
  }
  int device = -1;
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice") ||
      !cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: test requires the 16-SM SM87 target\n";
    return 77;
  }

  constexpr std::array<Role, 3U> kRoles{{
      Role::kFp8GdnQkvZ,
      Role::kFp8FullQkv,
      Role::kFp8AttentionOutput,
  }};
  bool ok = true;
  for (const Role role : kRoles) {
    kernels::Sm87MacroFeedV3Fp8CudaResources resources{};
    const int query_status =
        kernels::query_sm87_macrofeed_v3_fp8_cuda_resources(role, &resources);
    if (query_status != static_cast<int>(cudaSuccess)) {
      std::cerr << "resource query: "
                << cudaGetErrorString(
                       static_cast<cudaError_t>(query_status))
                << '\n';
      return 1;
    }
    std::cout << "resources role=" << static_cast<unsigned int>(role)
              << " regs=" << resources.registers_per_thread
              << " dynamic_shared=" << resources.dynamic_shared_bytes
              << " local=" << resources.local_bytes
              << " active_blocks_per_sm=" << resources.active_blocks_per_sm
              << '\n';
    ok &= resources.static_resource_gate_passed &&
          kernels::sm87_macrofeed_v3_fp8_resource_gate(resources);
    kernels::Sm87MacroFeedV3Fp8StartupSeal seal{};
    ok &= kernels::seal_sm87_macrofeed_v3_fp8_startup_cuda(role, &seal) ==
              static_cast<int>(cudaSuccess) &&
          kernels::sm87_macrofeed_v3_fp8_startup_seal_valid(seal);
  }
  if (!ok) {
    std::cerr << "FAIL: role resource/startup seal rejected\n";
    return 1;
  }
  if (!run_all_codes()) {
    std::cerr << "FAIL: 256-code bit-domain oracle\n";
    return 1;
  }

  const std::size_t input_elements = kRows * kInputFeatures;
  const std::size_t body_elements = kRows * kColumns;
  const std::size_t output_elements =
      body_elements + 2U * kGuardElements;
  std::vector<std::uint16_t> input(input_elements, 0U);
  build_input(input);
  std::uint16_t* device_input = nullptr;
  std::uint8_t* device_payload = nullptr;
  std::uint16_t* device_output = nullptr;
  ok = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_input),
                          input.size() * sizeof(std::uint16_t)),
               "cudaMalloc input") &&
       cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_payload),
                          kernels::kSm87MacroFeedV3Fp8TestPayloadBytes),
               "cudaMalloc payload") &&
       cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_output),
                          output_elements * sizeof(std::uint16_t)),
               "cudaMalloc output") &&
       cuda_ok(cudaMemcpy(device_input, input.data(),
                          input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy input H2D");

  if (ok) {
    const int invalid_rows =
        kernels::launch_sm87_macrofeed_v3_fp8_tile_test_cuda(
            Role::kFp8GdnQkvZ, 0U, device_input, device_payload,
            kCompensatedScaleOneBits, 63U,
            device_output + kGuardElements, nullptr);
    const int invalid_partition =
        kernels::launch_sm87_macrofeed_v3_fp8_tile_test_cuda(
            Role::kFp8AttentionOutput, 1U, device_input, device_payload,
            kCompensatedScaleOneBits, 64U,
            device_output + kGuardElements, nullptr);
    ok = invalid_rows == static_cast<int>(cudaErrorInvalidValue) &&
         invalid_partition == static_cast<int>(cudaErrorInvalidValue);
    if (!ok) {
      std::cerr << "FAIL: invalid tile shape/partition did not fail closed\n";
    }
  }

  // These role/partition fixtures validate the common canonical-cell
  // arithmetic and payload permutation.  Production offset/scatter and the
  // persistent 16-CTA scheduler remain explicitly outside this T1 seam.
  for (const Role role : kRoles) {
    if (!ok) {
      break;
    }
    const auto plan = kernels::sm87_macrofeed_v3_fp8_plan(role, 40'000U);
    for (std::size_t partition = 0U; partition < plan.partition_count;
         ++partition) {
      std::vector<std::uint8_t> payload(
          kernels::kSm87MacroFeedV3Fp8TestPayloadBytes, 0U);
      ok = build_payload(role, partition, payload) &&
           cuda_ok(cudaMemcpy(device_payload, payload.data(), payload.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy payload H2D") &&
           run_tile_case(role, partition, 128U, input, payload, device_input,
                         device_payload, device_output) &&
           run_tile_case(role, partition, 64U, input, payload, device_input,
                         device_payload, device_output);
      if (!ok) {
        break;
      }
    }
  }

  if (device_output != nullptr) {
    (void)cudaFree(device_output);
  }
  if (device_payload != nullptr) {
    (void)cudaFree(device_payload);
  }
  if (device_input != nullptr) {
    (void)cudaFree(device_input);
  }
  if (ok) {
    std::cout << "sm87_macrofeed_v3_fp8_cuda_test: PASS\n";
  }
  return ok ? 0 : 1;
}
