#include "q3x/kernels/sm87_macrofeed_v4_nvfp4_gate_up.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using q3x::kernels::kSm87MacroFeedV3NvFp4GateUpCellBytes;
using q3x::kernels::kSm87MacroFeedV3NvFp4GateUpWeightBytesPerCell;
using q3x::kernels::kSm87MacroFeedV4NvFp4GateUpBlockM;
using q3x::kernels::kSm87MacroFeedV4NvFp4GateUpBlockN;
using q3x::kernels::kSm87MacroFeedV4NvFp4GateUpBranches;
using q3x::kernels::kSm87MacroFeedV4NvFp4GateUpTestInputFeatures;
using q3x::kernels::kSm87MacroFeedV4NvFp4GateUpTestKTiles;
using q3x::kernels::kSm87MacroFeedV4NvFp4GateUpTestPartitionBytes;
using q3x::kernels::kSm87MacroFeedV4NvFp4GateUpTestPayloadBytes;

constexpr std::size_t kGuardElements = 8U;
constexpr std::uint16_t kGuardBits = 0x5a5aU;
constexpr std::uint16_t kPoisonBits = 0x7fc1U;

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

[[nodiscard]] std::uint8_t weight_code(const std::size_t branch,
                                       const std::size_t stage,
                                       const std::size_t k16,
                                       const std::size_t n_warp,
                                       const std::size_t n8,
                                       const std::size_t lane,
                                       const std::size_t component) {
  return static_cast<std::uint8_t>(
      (branch * 11U + stage * 7U + k16 * 5U + n_warp * 3U + n8 +
       lane * 9U + component * 5U) &
      0x0fU);
}

[[nodiscard]] std::uint8_t scale_code(const std::size_t branch,
                                      const std::size_t stage,
                                      const std::size_t k16,
                                      const std::size_t n_warp,
                                      const std::size_t n8,
                                      const std::size_t lane_group) {
  constexpr std::array<std::uint8_t, 16U> kCodes{{
      0x00U, 0x01U, 0x07U, 0x08U, 0x18U, 0x30U, 0x38U, 0x48U,
      0x7eU, 0x7fU, 0x80U, 0x81U, 0x87U, 0xb8U, 0xfeU, 0xffU}};
  return kCodes[(branch * 13U + stage * 7U + k16 * 5U +
                 n_warp * 3U + n8 + lane_group * 11U) %
                kCodes.size()];
}

void build_input(std::vector<std::uint16_t>& input) {
  for (std::size_t row = 0U; row < kSm87MacroFeedV4NvFp4GateUpBlockM;
       ++row) {
    for (std::size_t stage = 0U;
         stage < kSm87MacroFeedV4NvFp4GateUpTestKTiles; ++stage) {
      for (std::size_t k16 = 0U; k16 < 4U; ++k16) {
        for (std::size_t element = 0U; element < 16U; ++element) {
          const bool live = element == ((row + stage + k16) & 15U);
          const float sign = ((row + 2U * stage + k16) & 1U) == 0U
                                 ? 1.0F
                                 : -1.0F;
          input[row * kSm87MacroFeedV4NvFp4GateUpTestInputFeatures +
                stage * 64U + k16 * 16U + element] =
              live ? encode_bf16_rne(sign * 0.000244140625F) : 0U;
        }
      }
    }
  }
}

void build_payload(std::vector<std::uint8_t>& payload) {
  for (std::size_t branch = 0U;
       branch < kSm87MacroFeedV4NvFp4GateUpBranches; ++branch) {
    auto* const partition =
        payload.data() +
        branch * kSm87MacroFeedV4NvFp4GateUpTestPartitionBytes;
    for (std::size_t stage = 0U;
         stage < kSm87MacroFeedV4NvFp4GateUpTestKTiles; ++stage) {
      auto* const cell =
          partition + stage * kSm87MacroFeedV3NvFp4GateUpCellBytes;
      for (std::size_t k16 = 0U; k16 < 4U; ++k16) {
        for (std::size_t n_warp = 0U; n_warp < 4U; ++n_warp) {
          for (std::size_t n8 = 0U; n8 < 8U; ++n8) {
            const std::size_t fragment = (k16 * 4U + n_warp) * 8U + n8;
            for (std::size_t lane = 0U; lane < 32U; ++lane) {
              std::uint16_t packed = 0U;
              for (std::size_t component = 0U; component < 4U;
                   ++component) {
                packed |= static_cast<std::uint16_t>(
                              weight_code(branch, stage, k16, n_warp, n8,
                                          lane, component))
                          << (4U * component);
              }
              std::memcpy(cell + fragment * 64U + lane * 2U, &packed,
                          sizeof(packed));
            }
            for (std::size_t lane_group = 0U; lane_group < 8U;
                 ++lane_group) {
              cell[kSm87MacroFeedV3NvFp4GateUpWeightBytesPerCell +
                   fragment * 8U + lane_group] =
                  scale_code(branch, stage, k16, n_warp, n8, lane_group);
            }
          }
        }
      }
    }
  }
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t device_encode_bf16(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ float device_decode_bf16(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ float device_decode_e2m1(
    const std::uint8_t code) noexcept {
  constexpr float kMagnitude[8U]{0.0F, 0.5F, 1.0F, 1.5F,
                                 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = kMagnitude[code & 0x07U];
  return (code & 0x08U) != 0U ? -value : value;
}

[[nodiscard]] __device__ __forceinline__ float device_decode_e4m3_finite(
    const std::uint8_t code) noexcept {
  const unsigned int magnitude = code & 0x7fU;
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  const float value =
      exponent == 0U
          ? ldexpf(static_cast<float>(mantissa), -9)
          : ldexpf(static_cast<float>(8U + mantissa),
                   static_cast<int>(exponent) - 10);
  return (code & 0x80U) != 0U ? -value : value;
}

[[nodiscard]] __device__ __forceinline__ float reference_weight(
    const std::uint8_t* const payload, const unsigned int branch,
    const unsigned int stage, const unsigned int k16,
    const unsigned int element, const unsigned int column,
    const unsigned int canonical_n_half) noexcept {
  const unsigned int canonical_column =
      canonical_n_half *
          static_cast<unsigned int>(kSm87MacroFeedV4NvFp4GateUpBlockN) +
      column;
  const unsigned int n_warp = canonical_column / 64U;
  const unsigned int n8 = (canonical_column % 64U) / 8U;
  const auto* const cell =
      payload +
      static_cast<std::size_t>(branch) *
          kSm87MacroFeedV4NvFp4GateUpTestPartitionBytes +
      static_cast<std::size_t>(stage) *
          kSm87MacroFeedV3NvFp4GateUpCellBytes;
  const std::size_t fragment =
      (static_cast<std::size_t>(k16) * 4U + n_warp) * 8U + n8;
  const unsigned int lane =
      (canonical_column % 8U) * 4U + (element % 8U) / 2U;
  const unsigned int persisted_component =
      element < 8U ? ((element & 1U) == 0U ? 0U : 2U)
                   : ((element & 1U) == 0U ? 1U : 3U);
  const std::uint16_t packed = *reinterpret_cast<const std::uint16_t*>(
      cell + fragment * 64U + lane * 2U);
  const std::uint8_t scale =
      cell[kSm87MacroFeedV3NvFp4GateUpWeightBytesPerCell +
           fragment * 8U + canonical_column % 8U];
  const auto code = static_cast<std::uint8_t>(
      (packed >> (4U * persisted_component)) & 0x0fU);
  return device_decode_bf16(device_encode_bf16(
      device_decode_e2m1(code) *
      device_decode_e4m3_finite(scale)));
}

[[nodiscard]] __device__ __forceinline__ float reference_dot(
    const std::uint16_t* const input, const std::uint8_t* const payload,
    const unsigned int branch, const unsigned int row,
    const unsigned int column,
    const unsigned int canonical_n_half) noexcept {
  float accumulator = 0.0F;
#pragma unroll 1
  for (unsigned int stage = 0U;
       stage < kSm87MacroFeedV4NvFp4GateUpTestKTiles; ++stage) {
#pragma unroll
    for (unsigned int k16 = 0U; k16 < 4U; ++k16) {
#pragma unroll
      for (unsigned int element = 0U; element < 16U; ++element) {
        const float weight =
            reference_weight(payload, branch, stage, k16, element, column,
                             canonical_n_half);
        accumulator = fmaf(
            device_decode_bf16(
                input[static_cast<std::size_t>(row) *
                          kSm87MacroFeedV4NvFp4GateUpTestInputFeatures +
                      stage * 64U + k16 * 16U + element]),
            weight, accumulator);
      }
    }
  }
  return accumulator;
}

__global__ void gate_up_reference_kernel(
    const std::uint16_t* const input, const std::uint8_t* const payload,
    const unsigned int valid_rows, const unsigned int canonical_n_half,
    const float gate_tensor_scale, const float up_tensor_scale,
    std::uint16_t* const output) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned int kElements =
      static_cast<unsigned int>(kSm87MacroFeedV4NvFp4GateUpBlockM *
                                kSm87MacroFeedV4NvFp4GateUpBlockN);
  if (linear >= kElements) {
    return;
  }
  const unsigned int row =
      linear / kSm87MacroFeedV4NvFp4GateUpBlockN;
  if (row >= valid_rows) {
    return;
  }
  const unsigned int column =
      linear % kSm87MacroFeedV4NvFp4GateUpBlockN;
  const std::uint16_t gate_bits = device_encode_bf16(
      reference_dot(input, payload, 0U, row, column, canonical_n_half) *
      gate_tensor_scale);
  const std::uint16_t up_bits = device_encode_bf16(
      reference_dot(input, payload, 1U, row, column, canonical_n_half) *
      up_tensor_scale);
  const float gate = device_decode_bf16(gate_bits);
  const float up = device_decode_bf16(up_bits);
  output[linear] =
      device_encode_bf16(gate / (1.0F + expf(-gate)) * up);
}

bool cuda_ok(const cudaError_t status, const char* const operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool run_case(const std::size_t valid_rows,
              const std::size_t canonical_n_half,
              const std::uint16_t* const device_input,
              const std::uint8_t* const device_payload,
              std::uint16_t* const device_candidate_storage,
              std::uint16_t* const device_reference_storage,
              const float gate_tensor_scale,
              const float up_tensor_scale) {
  constexpr std::size_t kBodyElements =
      kSm87MacroFeedV4NvFp4GateUpBlockM *
      kSm87MacroFeedV4NvFp4GateUpBlockN;
  constexpr std::size_t kTotalElements =
      kBodyElements + 2U * kGuardElements;
  std::vector<std::uint16_t> initial(kTotalElements, kPoisonBits);
  for (std::size_t index = 0U; index < kGuardElements; ++index) {
    initial[index] = kGuardBits;
    initial[kGuardElements + kBodyElements + index] = kGuardBits;
  }
  if (!cuda_ok(cudaMemcpy(device_candidate_storage, initial.data(),
                          kTotalElements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy candidate init") ||
      !cuda_ok(cudaMemcpy(device_reference_storage, initial.data(),
                          kTotalElements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy reference init")) {
    return false;
  }

  auto* const device_candidate =
      device_candidate_storage + kGuardElements;
  auto* const device_reference =
      device_reference_storage + kGuardElements;
  const int launch_status =
      q3x::kernels::launch_sm87_macrofeed_v4_nvfp4_gate_up_tile_test_cuda(
          device_input, device_payload, gate_tensor_scale, up_tensor_scale,
          valid_rows, canonical_n_half, device_candidate, nullptr);
  if (launch_status != static_cast<int>(cudaSuccess)) {
    std::cerr << "tile launch: "
              << cudaGetErrorString(static_cast<cudaError_t>(launch_status))
              << '\n';
    return false;
  }
  constexpr unsigned int kReferenceThreads = 256U;
  constexpr unsigned int kReferenceBlocks =
      static_cast<unsigned int>((kBodyElements + kReferenceThreads - 1U) /
                                kReferenceThreads);
  gate_up_reference_kernel<<<kReferenceBlocks, kReferenceThreads>>>(
      device_input, device_payload, static_cast<unsigned int>(valid_rows),
      static_cast<unsigned int>(canonical_n_half), gate_tensor_scale,
      up_tensor_scale, device_reference);
  if (!cuda_ok(cudaPeekAtLastError(), "reference launch") ||
      !cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    return false;
  }

  std::vector<std::uint16_t> candidate(kTotalElements, 0U);
  std::vector<std::uint16_t> reference(kTotalElements, 0U);
  if (!cuda_ok(cudaMemcpy(candidate.data(), device_candidate_storage,
                          kTotalElements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy candidate D2H") ||
      !cuda_ok(cudaMemcpy(reference.data(), device_reference_storage,
                          kTotalElements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy reference D2H")) {
    return false;
  }
  for (std::size_t index = 0U; index < kTotalElements; ++index) {
    if (candidate[index] != reference[index]) {
      std::cerr << "bit mismatch valid_rows=" << valid_rows
                << " n_half=" << canonical_n_half << " index=" << index
                << " expected=0x" << std::hex
                << reference[index] << " actual=0x" << candidate[index]
                << std::dec << '\n';
      return false;
    }
  }
  return true;
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

  q3x::kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources resources{};
  const int query_status =
      q3x::kernels::query_sm87_macrofeed_v4_nvfp4_gate_up_cuda_resources(
          &resources);
  if (query_status != static_cast<int>(cudaSuccess)) {
    std::cerr << "resource query: "
              << cudaGetErrorString(static_cast<cudaError_t>(query_status))
              << '\n';
    return 1;
  }
  std::cout << "resources: regs=" << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  if (!resources.static_resource_gate_passed ||
      resources.active_blocks_per_sm < 2 ||
      !q3x::kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(
          resources)) {
    std::cerr << "FAIL: V4 Gate+Up hard two-CTA/SM resource gate rejected "
                 "kernel\n";
    return 1;
  }

  const std::size_t input_elements =
      kSm87MacroFeedV4NvFp4GateUpBlockM *
      kSm87MacroFeedV4NvFp4GateUpTestInputFeatures;
  std::vector<std::uint16_t> input(input_elements, 0U);
  std::vector<std::uint8_t> payload(
      kSm87MacroFeedV4NvFp4GateUpTestPayloadBytes, 0U);
  build_input(input);
  build_payload(payload);

  std::uint16_t* device_input = nullptr;
  std::uint8_t* device_payload = nullptr;
  std::uint16_t* device_candidate_storage = nullptr;
  std::uint16_t* device_reference_storage = nullptr;
  constexpr std::size_t kBodyElements =
      kSm87MacroFeedV4NvFp4GateUpBlockM *
      kSm87MacroFeedV4NvFp4GateUpBlockN;
  constexpr std::size_t kStorageElements =
      kBodyElements + 2U * kGuardElements;
  bool ok =
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_input),
                         input.size() * sizeof(std::uint16_t)),
              "cudaMalloc input") &&
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_payload),
                         payload.size()),
              "cudaMalloc payload") &&
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_candidate_storage),
                         kStorageElements * sizeof(std::uint16_t)),
              "cudaMalloc candidate") &&
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_reference_storage),
                         kStorageElements * sizeof(std::uint16_t)),
              "cudaMalloc reference");
  if (ok) {
    ok = cuda_ok(cudaMemcpy(device_input, input.data(),
                            input.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy input H2D") &&
         cuda_ok(cudaMemcpy(device_payload, payload.data(), payload.size(),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy payload H2D");
  }

  constexpr float kGateTensorScale = 0.03125F;
  constexpr float kUpTensorScale = 0.015625F;
  if (ok) {
    const int invalid_status =
        q3x::kernels::
            launch_sm87_macrofeed_v4_nvfp4_gate_up_tile_test_cuda(
                device_input, device_payload, kGateTensorScale,
                kUpTensorScale, 65U, 0U,
                device_candidate_storage + kGuardElements, nullptr);
    ok = invalid_status == static_cast<int>(cudaErrorInvalidValue);
    if (!ok) {
      std::cerr << "FAIL: invalid M65 test shape did not fail closed\n";
    }
  }
  if (ok) {
    const int invalid_half_status =
        q3x::kernels::
            launch_sm87_macrofeed_v4_nvfp4_gate_up_tile_test_cuda(
                device_input, device_payload, kGateTensorScale,
                kUpTensorScale, 64U, 2U,
                device_candidate_storage + kGuardElements, nullptr);
    ok = invalid_half_status == static_cast<int>(cudaErrorInvalidValue);
    if (!ok) {
      std::cerr << "FAIL: invalid canonical N half did not fail closed\n";
    }
  }
  if (ok) {
    ok = run_case(64U, 0U, device_input, device_payload,
                  device_candidate_storage, device_reference_storage,
                  kGateTensorScale, kUpTensorScale);
  }
  if (ok) {
    ok = run_case(37U, 1U, device_input, device_payload,
                  device_candidate_storage, device_reference_storage,
                  kGateTensorScale, kUpTensorScale);
  }

  if (ok) {
    std::vector<std::uint16_t> input_after(input.size(), 0U);
    std::vector<std::uint8_t> payload_after(payload.size(), 0U);
    ok = cuda_ok(cudaMemcpy(input_after.data(), device_input,
                            input_after.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy input immutability") &&
         cuda_ok(cudaMemcpy(payload_after.data(), device_payload,
                            payload_after.size(), cudaMemcpyDeviceToHost),
                 "cudaMemcpy payload immutability") &&
         input_after == input && payload_after == payload;
    if (!ok) {
      std::cerr << "FAIL: input or payload mutated\n";
    }
  }

  if (device_reference_storage != nullptr) {
    (void)cudaFree(device_reference_storage);
  }
  if (device_candidate_storage != nullptr) {
    (void)cudaFree(device_candidate_storage);
  }
  if (device_payload != nullptr) {
    (void)cudaFree(device_payload);
  }
  if (device_input != nullptr) {
    (void)cudaFree(device_input);
  }
  if (ok) {
    std::cout << "sm87_macrofeed_v4_nvfp4_gate_up_cuda_test: PASS\n";
  }
  return ok ? 0 : 1;
}
