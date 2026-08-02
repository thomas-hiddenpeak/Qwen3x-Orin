#include "q3x/kernels/sm87_a4w4_ldmatrix_operand_probe.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

inline constexpr std::size_t kGuardWords = 4U;
inline constexpr std::int32_t kGuard =
    static_cast<std::int32_t>(0x5a3c'17e9U);

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const char* const action) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << action << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

[[nodiscard]] std::size_t packed_offset(
    const std::size_t outer, const std::size_t k64_group,
    const std::size_t byte_in_k64) {
  return kernels::sm87_a4w4_consumer_packed_offset(
      outer, k64_group, byte_in_k64,
      kernels::kSm87A4W4LdmatrixProbeK64Groups);
}

enum class Pattern : unsigned int {
  kExtrema,
  kK512ConsumerCodeBlock,
};

[[nodiscard]] int code(const Pattern pattern, const unsigned int operand,
                       const std::size_t outer,
                       const std::size_t absolute_k) {
  constexpr std::array<int, 4U> extrema{-8, -7, 0, 7};
  if (pattern == Pattern::kExtrema) {
    return extrema[(5U * outer + 3U * absolute_k + operand) %
                   extrema.size()];
  }

  // A deterministic, nonuniform code distribution in the exact production
  // K512 consumer block ordering. This is intentionally not a row-major K512
  // stand-in: all eight physical K64 groups and their 64-row padded blocks
  // are materialized. Extremal codes are retained among the common near-zero
  // codes so signed-nibble ordering errors cannot hide behind benign inputs.
  constexpr std::array<int, 32U> realistic{
      0,  0,  1,  -1, 0,  2,  -2, 1,
      0,  -1, 3,  -3, 0,  1,  -1, 4,
      -4, 2,  -2, 5,  -5, 6,  -6, 7,
      -7, -8, 0,  1,  -1, 2,  0,  -2,
  };
  std::uint32_t mixed =
      static_cast<std::uint32_t>((outer + 1U) * 0x9e37'79b9U);
  mixed ^= static_cast<std::uint32_t>((absolute_k + 17U) * 0x85eb'ca6bU);
  mixed ^= operand * 0xc2b2'ae35U;
  mixed ^= mixed >> 16U;
  return realistic[mixed & 31U];
}

[[nodiscard]] std::vector<std::uint8_t> make_payload(
    const Pattern pattern, const unsigned int operand,
    const std::size_t logical_outer) {
  std::vector<std::uint8_t> payload(
      kernels::kSm87A4W4LdmatrixProbePayloadBytes, 0xa5U);
  std::array<bool, 16U> seen{};
  for (std::size_t outer = 0U; outer < logical_outer; ++outer) {
    for (std::size_t group = 0U;
         group < kernels::kSm87A4W4LdmatrixProbeK64Groups; ++group) {
      for (std::size_t byte = 0U;
           byte < kernels::kSm87A4W4LdmatrixProbePackedK64Bytes; ++byte) {
        const std::size_t absolute_k = group * 64U + byte * 2U;
        const int even = code(pattern, operand, outer, absolute_k);
        const int odd = code(pattern, operand, outer, absolute_k + 1U);
        seen[static_cast<std::size_t>(even + 8)] = true;
        seen[static_cast<std::size_t>(odd + 8)] = true;
        payload[packed_offset(outer, group, byte)] =
            kernels::sm87_a4w4_pack_signed_pair(even, odd);
      }
    }
  }
  for (const int required : {-8, -7, 0, 7}) {
    if (!seen[static_cast<std::size_t>(required + 8)]) {
      std::cerr << "pattern omitted required signed code " << required
                << '\n';
      return {};
    }
  }
  return payload;
}

[[nodiscard]] int unpack(const std::vector<std::uint8_t>& payload,
                         const std::size_t outer,
                         const std::size_t k64_group,
                         const std::size_t k_in_group) {
  const std::uint8_t packed =
      payload[packed_offset(outer, k64_group, k_in_group / 2U)];
  return kernels::sm87_a4w4_unpack_signed(packed, k_in_group);
}

[[nodiscard]] std::int32_t oracle(
    const std::vector<std::uint8_t>& a,
    const std::vector<std::uint8_t>& b,
    const std::size_t m, const std::size_t n,
    const std::size_t k64_group) {
  std::int32_t sum = 0;
  for (std::size_t k = 0U; k < 64U; ++k) {
    sum += unpack(a, m, k64_group, k) *
           unpack(b, n, k64_group, k);
  }
  return sum;
}

[[nodiscard]] bool guards_intact(
    const std::vector<std::int32_t>& values,
    const std::string& label) {
  for (std::size_t index = 0U; index < kGuardWords; ++index) {
    if (values[index] != kGuard ||
        values[values.size() - kGuardWords + index] != kGuard) {
      std::cerr << label << " guard mismatch at " << index << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool run_pattern(const Pattern pattern,
                               const std::string& label) {
  const std::vector<std::uint8_t> host_a = make_payload(pattern, 0U, 16U);
  const std::vector<std::uint8_t> host_b = make_payload(pattern, 1U, 8U);
  if (host_a.empty() || host_b.empty()) {
    return false;
  }

  std::uint8_t* device_a = nullptr;
  std::uint8_t* device_b = nullptr;
  std::int32_t* device_scalar_storage = nullptr;
  std::int32_t* device_ldmatrix_storage = nullptr;
  std::int32_t* device_scalar_a_ldmatrix_b_storage = nullptr;
  std::int32_t* device_ldmatrix_a_scalar_b_storage = nullptr;
  constexpr std::size_t output_storage_words =
      kernels::kSm87A4W4LdmatrixProbeAccumulatorWords + 2U * kGuardWords;
  std::vector<std::int32_t> initialized(output_storage_words, kGuard);
  std::vector<std::int32_t> scalar(initialized.size());
  std::vector<std::int32_t> ldmatrix(initialized.size());
  std::vector<std::int32_t> scalar_a_ldmatrix_b(initialized.size());
  std::vector<std::int32_t> ldmatrix_a_scalar_b(initialized.size());

  bool ok =
      cuda_ok(cudaMalloc(&device_a, host_a.size()), "allocate A") &&
      cuda_ok(cudaMalloc(&device_b, host_b.size()), "allocate B") &&
      cuda_ok(cudaMalloc(&device_scalar_storage,
                         output_storage_words * sizeof(std::int32_t)),
              "allocate scalar output") &&
      cuda_ok(cudaMalloc(&device_ldmatrix_storage,
                         output_storage_words * sizeof(std::int32_t)),
              "allocate ldmatrix output") &&
      cuda_ok(cudaMalloc(&device_scalar_a_ldmatrix_b_storage,
                         output_storage_words * sizeof(std::int32_t)),
              "allocate scalar-A/ldmatrix-B output") &&
      cuda_ok(cudaMalloc(&device_ldmatrix_a_scalar_b_storage,
                         output_storage_words * sizeof(std::int32_t)),
              "allocate ldmatrix-A/scalar-B output") &&
      cuda_ok(cudaMemcpy(device_a, host_a.data(), host_a.size(),
                         cudaMemcpyHostToDevice),
              "copy A") &&
      cuda_ok(cudaMemcpy(device_b, host_b.data(), host_b.size(),
                         cudaMemcpyHostToDevice),
              "copy B");

  for (std::size_t group = 0U;
       ok && group < kernels::kSm87A4W4LdmatrixProbeK64Groups; ++group) {
    ok = cuda_ok(cudaMemcpy(device_scalar_storage, initialized.data(),
                            output_storage_words * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice),
                 "reset scalar output") &&
         cuda_ok(cudaMemcpy(device_ldmatrix_storage, initialized.data(),
                            output_storage_words * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice),
                 "reset ldmatrix output") &&
         cuda_ok(cudaMemcpy(device_scalar_a_ldmatrix_b_storage,
                            initialized.data(),
                            output_storage_words * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice),
                 "reset scalar-A/ldmatrix-B output") &&
         cuda_ok(cudaMemcpy(device_ldmatrix_a_scalar_b_storage,
                            initialized.data(),
                            output_storage_words * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice),
                 "reset ldmatrix-A/scalar-B output");
    if (!ok) {
      break;
    }
    const int launch_status =
        kernels::launch_sm87_a4w4_ldmatrix_operand_probe_cuda(
            device_a, host_a.size(), device_b, host_b.size(), group,
            device_scalar_storage + kGuardWords,
            kernels::kSm87A4W4LdmatrixProbeAccumulatorWords,
            device_ldmatrix_storage + kGuardWords,
            kernels::kSm87A4W4LdmatrixProbeAccumulatorWords,
            device_scalar_a_ldmatrix_b_storage + kGuardWords,
            kernels::kSm87A4W4LdmatrixProbeAccumulatorWords,
            device_ldmatrix_a_scalar_b_storage + kGuardWords,
            kernels::kSm87A4W4LdmatrixProbeAccumulatorWords, nullptr);
    if (launch_status != static_cast<int>(cudaSuccess)) {
      std::cerr << label << " group " << group << " launch failed: "
                << cudaGetErrorString(static_cast<cudaError_t>(launch_status))
                << '\n';
      ok = false;
      break;
    }
    ok = cuda_ok(cudaDeviceSynchronize(), "synchronize operand probe") &&
         cuda_ok(cudaMemcpy(scalar.data(), device_scalar_storage,
                            output_storage_words * sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost),
                 "copy scalar output") &&
         cuda_ok(cudaMemcpy(ldmatrix.data(), device_ldmatrix_storage,
                            output_storage_words * sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost),
                 "copy ldmatrix output") &&
         cuda_ok(cudaMemcpy(scalar_a_ldmatrix_b.data(),
                            device_scalar_a_ldmatrix_b_storage,
                            output_storage_words * sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost),
                 "copy scalar-A/ldmatrix-B output") &&
         cuda_ok(cudaMemcpy(ldmatrix_a_scalar_b.data(),
                            device_ldmatrix_a_scalar_b_storage,
                            output_storage_words * sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost),
                 "copy ldmatrix-A/scalar-B output") &&
         guards_intact(scalar, label + " scalar") &&
         guards_intact(ldmatrix, label + " ldmatrix") &&
         guards_intact(scalar_a_ldmatrix_b,
                       label + " scalar-A/ldmatrix-B") &&
         guards_intact(ldmatrix_a_scalar_b,
                       label + " ldmatrix-A/scalar-B");
    for (std::size_t lane = 0U; ok && lane < 32U; ++lane) {
      for (std::size_t reg = 0U; reg < 4U; ++reg) {
        const std::size_t output_index =
            kGuardWords + lane * 4U + reg;
        const auto coordinate =
            kernels::sm87_a4w4_accumulator_coordinate(lane, reg);
        const std::int32_t expected =
            oracle(host_a, host_b, coordinate.m, coordinate.n, group);
        if (scalar[output_index] != ldmatrix[output_index] ||
            scalar[output_index] !=
                scalar_a_ldmatrix_b[output_index] ||
            scalar[output_index] !=
                ldmatrix_a_scalar_b[output_index] ||
            scalar[output_index] != expected) {
          std::cerr << label << " group " << group << " lane " << lane
                    << " reg " << reg << " (M" << coordinate.m << ",N"
                    << coordinate.n << ") mismatch: scalar="
                    << scalar[output_index] << " ldmatrix="
                    << ldmatrix[output_index]
                    << " scalar-A/ldmatrix-B="
                    << scalar_a_ldmatrix_b[output_index]
                    << " ldmatrix-A/scalar-B="
                    << ldmatrix_a_scalar_b[output_index]
                    << " oracle=" << expected
                    << '\n';
          ok = false;
          break;
        }
      }
    }
  }

  if (device_ldmatrix_a_scalar_b_storage != nullptr) {
    (void)cudaFree(device_ldmatrix_a_scalar_b_storage);
  }
  if (device_scalar_a_ldmatrix_b_storage != nullptr) {
    (void)cudaFree(device_scalar_a_ldmatrix_b_storage);
  }
  if (device_ldmatrix_storage != nullptr) {
    (void)cudaFree(device_ldmatrix_storage);
  }
  if (device_scalar_storage != nullptr) {
    (void)cudaFree(device_scalar_storage);
  }
  if (device_b != nullptr) {
    (void)cudaFree(device_b);
  }
  if (device_a != nullptr) {
    (void)cudaFree(device_a);
  }
  return ok;
}

}  // namespace

int main() {
  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA device unavailable\n";
    (void)cudaGetLastError();
    return 77;
  }
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDeviceProperties(&properties, 0),
               "query device properties")) {
    return 1;
  }
  if (properties.major != kernels::kSm87A4W4RequiredComputeMajor ||
      properties.minor != kernels::kSm87A4W4RequiredComputeMinor) {
    std::cout << "SKIP: exact SM87 device required\n";
    return 77;
  }

  kernels::Sm87A4W4LdmatrixOperandProbeResources resources{};
  const int resource_status =
      kernels::query_sm87_a4w4_ldmatrix_operand_probe_resources_cuda(
          &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    std::cerr << "resource query failed: "
              << cudaGetErrorString(static_cast<cudaError_t>(resource_status))
              << '\n';
    return 1;
  }
  if (resources.scalar_static_shared_bytes != 768U ||
      resources.ldmatrix_static_shared_bytes != 768U ||
      resources.scalar_a_ldmatrix_b_static_shared_bytes != 768U ||
      resources.ldmatrix_a_scalar_b_static_shared_bytes != 768U ||
      resources.scalar_local_bytes != 0U ||
      resources.ldmatrix_local_bytes != 0U ||
      resources.scalar_a_ldmatrix_b_local_bytes != 0U ||
      resources.ldmatrix_a_scalar_b_local_bytes != 0U ||
      resources.scalar_active_blocks_per_sm <= 0 ||
      resources.ldmatrix_active_blocks_per_sm <= 0 ||
      resources.scalar_a_ldmatrix_b_active_blocks_per_sm <= 0 ||
      resources.ldmatrix_a_scalar_b_active_blocks_per_sm <= 0) {
    std::cerr << "unexpected resource contract: scalar regs="
              << resources.scalar_registers_per_thread << " shared="
              << resources.scalar_static_shared_bytes << " local="
              << resources.scalar_local_bytes << " active="
              << resources.scalar_active_blocks_per_sm << "; ldmatrix regs="
              << resources.ldmatrix_registers_per_thread << " shared="
              << resources.ldmatrix_static_shared_bytes << " local="
              << resources.ldmatrix_local_bytes << " active="
              << resources.ldmatrix_active_blocks_per_sm
              << "; scalar-A/ldmatrix-B regs="
              << resources.scalar_a_ldmatrix_b_registers_per_thread
              << " shared="
              << resources.scalar_a_ldmatrix_b_static_shared_bytes
              << " local=" << resources.scalar_a_ldmatrix_b_local_bytes
              << " active="
              << resources.scalar_a_ldmatrix_b_active_blocks_per_sm
              << "; ldmatrix-A/scalar-B regs="
              << resources.ldmatrix_a_scalar_b_registers_per_thread
              << " shared="
              << resources.ldmatrix_a_scalar_b_static_shared_bytes
              << " local=" << resources.ldmatrix_a_scalar_b_local_bytes
              << " active="
              << resources.ldmatrix_a_scalar_b_active_blocks_per_sm << '\n';
    return 1;
  }

  const bool ok = run_pattern(Pattern::kExtrema, "extrema") &&
                  run_pattern(Pattern::kK512ConsumerCodeBlock,
                              "K512 consumer code block");
  if (!ok) {
    return 1;
  }
  std::cout << "PASS: scalar/scalar, LDSM/LDSM, scalar-A/LDSM-B, and "
               "LDSM-A/scalar-B emit bitwise identical S32 fragments "
               "across two K512 code blocks\n";
  return 0;
}
