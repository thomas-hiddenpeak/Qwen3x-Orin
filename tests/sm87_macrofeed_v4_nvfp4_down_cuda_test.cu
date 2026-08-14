#include "q3x/kernels/sm87_macrofeed_v4_nvfp4_down.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using q3x::kernels::kSm87MacroFeedV4NvFp4DownBlockM;
using q3x::kernels::kSm87MacroFeedV4NvFp4DownBlockN;
using q3x::kernels::kSm87MacroFeedV4NvFp4DownCanonicalCellBytes;
using q3x::kernels::
    kSm87MacroFeedV4NvFp4DownCanonicalScaleBytesPerCell;
using q3x::kernels::
    kSm87MacroFeedV4NvFp4DownCanonicalWeightBytesPerCell;
using q3x::kernels::kSm87MacroFeedV4NvFp4DownTestInputFeatures;
using q3x::kernels::kSm87MacroFeedV4NvFp4DownTestKTiles;
using q3x::kernels::kSm87MacroFeedV4NvFp4DownTestPayloadBytes;

constexpr std::size_t kGuardElements = 8U;
constexpr std::uint16_t kGuardBits = 0x5a5aU;

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

[[nodiscard]] float decode_bf16(const std::uint16_t bits) {
  const std::uint32_t value = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

[[nodiscard]] float decode_e2m1(const std::uint8_t code) {
  constexpr std::array<float, 8U> kMagnitude{
      0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = kMagnitude[code & 0x07U];
  return (code & 0x08U) != 0U ? -value : value;
}

[[nodiscard]] float decode_e4m3fn(const std::uint8_t code) {
  const unsigned int magnitude = code & 0x7fU;
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  const float value =
      exponent == 0U
          ? std::ldexp(static_cast<float>(mantissa), -9)
          : std::ldexp(static_cast<float>(8U + mantissa),
                       static_cast<int>(exponent) - 10);
  return (code & 0x80U) != 0U ? -value : value;
}

[[nodiscard]] std::uint8_t weight_code(const std::size_t stage,
                                       const std::size_t k16,
                                       const std::size_t n_warp,
                                       const std::size_t n8,
                                       const std::size_t lane,
                                       const std::size_t component) {
  return static_cast<std::uint8_t>(
      (stage * 7U + k16 * 5U + n_warp * 3U + n8 + lane * 9U +
       component * 5U) &
      0x0fU);
}

[[nodiscard]] std::uint8_t scale_code(const std::size_t stage,
                                      const std::size_t k16,
                                      const std::size_t n_warp,
                                      const std::size_t n8,
                                      const std::size_t lane_group) {
  constexpr std::array<std::uint8_t, 16U> kCodes{{
      0x00U, 0x01U, 0x07U, 0x08U, 0x18U, 0x30U, 0x38U, 0x48U,
      0x7eU, 0x7fU, 0x80U, 0x81U, 0x87U, 0xb8U, 0xfeU, 0xffU}};
  return kCodes[(2U * stage + k16 + 3U * n_warp + n8 +
                 lane_group * 11U) %
                kCodes.size()];
}

[[nodiscard]] float activation_value(const std::size_t row,
                                     const std::size_t stage,
                                     const std::size_t k16,
                                     const std::size_t element) {
  const bool live = element == ((row + 3U * stage + 5U * k16) & 15U);
  if (!live) {
    return 0.0F;
  }
  return ((row + stage + 2U * k16) & 1U) == 0U ? 0.015625F
                                                : -0.015625F;
}

[[nodiscard]] std::uint16_t residual_value(const std::size_t row,
                                           const std::size_t column) {
  constexpr std::array<float, 4U> kValues{{0.25F, -0.5F, 1.0F, 2.0F}};
  return encode_bf16_rne(kValues[(row + column / 8U) % kValues.size()]);
}

void build_input(std::vector<std::uint16_t>& input) {
  for (std::size_t row = 0U; row < kSm87MacroFeedV4NvFp4DownBlockM;
       ++row) {
    for (std::size_t stage = 0U;
      stage < kSm87MacroFeedV4NvFp4DownTestKTiles; ++stage) {
      for (std::size_t k16 = 0U; k16 < 4U; ++k16) {
        for (std::size_t element = 0U; element < 16U; ++element) {
          input[row * kSm87MacroFeedV4NvFp4DownTestInputFeatures +
                stage * 64U + k16 * 16U + element] =
              encode_bf16_rne(activation_value(row, stage, k16, element));
        }
      }
    }
  }
}

void build_payload(std::vector<std::uint8_t>& payload) {
  static_assert(
      kSm87MacroFeedV4NvFp4DownCanonicalScaleBytesPerCell == 1'024U);
  for (std::size_t stage = 0U;
       stage < kSm87MacroFeedV4NvFp4DownTestKTiles; ++stage) {
    std::uint8_t* const cell =
        payload.data() +
        stage * kSm87MacroFeedV4NvFp4DownCanonicalCellBytes;
    for (std::size_t k16 = 0U; k16 < 4U; ++k16) {
      for (std::size_t n_warp = 0U; n_warp < 4U; ++n_warp) {
        for (std::size_t n8 = 0U; n8 < 8U; ++n8) {
          const std::size_t fragment = (k16 * 4U + n_warp) * 8U + n8;
          for (std::size_t lane = 0U; lane < 32U; ++lane) {
            std::uint16_t packed = 0U;
            for (std::size_t component = 0U; component < 4U;
                 ++component) {
              packed |= static_cast<std::uint16_t>(
                            weight_code(stage, k16, n_warp, n8, lane,
                                        component))
                        << (4U * component);
            }
            std::memcpy(cell + fragment * 64U + lane * 2U, &packed,
                        sizeof(packed));
          }
          for (std::size_t lane_group = 0U; lane_group < 8U;
               ++lane_group) {
            cell[kSm87MacroFeedV4NvFp4DownCanonicalWeightBytesPerCell +
                 fragment * 8U + lane_group] =
                scale_code(stage, k16, n_warp, n8, lane_group);
          }
        }
      }
    }
  }
}

[[nodiscard]] std::uint16_t expected_value(
    const std::vector<std::uint16_t>& input, const std::size_t row,
    const std::size_t local_column, const std::size_t canonical_n_half,
    const std::vector<std::uint8_t>& payload,
    const std::uint16_t residual, const float tensor_scale) {
  const std::size_t canonical_column =
      canonical_n_half * kSm87MacroFeedV4NvFp4DownBlockN + local_column;
  const std::size_t n_warp = canonical_column / 64U;
  const std::size_t n8 = (canonical_column % 64U) / 8U;
  float accumulator = 0.0F;
  for (std::size_t stage = 0U;
       stage < kSm87MacroFeedV4NvFp4DownTestKTiles; ++stage) {
    const std::uint8_t* const cell =
        payload.data() +
        stage * kSm87MacroFeedV4NvFp4DownCanonicalCellBytes;
    for (std::size_t k16 = 0U; k16 < 4U; ++k16) {
      const std::size_t fragment = (k16 * 4U + n_warp) * 8U + n8;
      for (std::size_t element = 0U; element < 16U; ++element) {
        const std::size_t lane =
            (canonical_column % 8U) * 4U + (element % 8U) / 2U;
        const std::size_t persisted_component =
            element < 8U ? ((element & 1U) == 0U ? 0U : 2U)
                         : ((element & 1U) == 0U ? 1U : 3U);
        std::uint16_t packed = 0U;
        std::memcpy(&packed, cell + fragment * 64U + lane * 2U,
                    sizeof(packed));
        const auto code = static_cast<std::uint8_t>(
            (packed >> (4U * persisted_component)) & 0x0fU);
        const std::uint8_t scale =
            cell[kSm87MacroFeedV4NvFp4DownCanonicalWeightBytesPerCell +
                 fragment * 8U + canonical_column % 8U];
        const float weight = decode_bf16(encode_bf16_rne(
            decode_e2m1(code) * decode_e4m3fn(scale)));
        const float activation = decode_bf16(
            input[row * kSm87MacroFeedV4NvFp4DownTestInputFeatures +
                  stage * 64U + k16 * 16U + element]);
        accumulator = std::fma(activation, weight, accumulator);
      }
    }
  }
  const std::uint16_t branch =
      encode_bf16_rne(accumulator * tensor_scale);
  return encode_bf16_rne(decode_bf16(branch) + decode_bf16(residual));
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
              const std::vector<std::uint16_t>& input,
              const std::vector<std::uint8_t>& payload,
              const std::uint16_t* const device_input,
              const std::uint8_t* const device_payload,
              std::uint16_t* const device_storage,
              const float tensor_scale) {
  const std::size_t body_elements =
      kSm87MacroFeedV4NvFp4DownBlockM *
      kSm87MacroFeedV4NvFp4DownBlockN;
  const std::size_t total_elements = body_elements + 2U * kGuardElements;
  std::vector<std::uint16_t> initial(total_elements, kGuardBits);
  for (std::size_t row = 0U; row < kSm87MacroFeedV4NvFp4DownBlockM;
       ++row) {
    for (std::size_t column = 0U;
         column < kSm87MacroFeedV4NvFp4DownBlockN; ++column) {
      initial[kGuardElements +
              row * kSm87MacroFeedV4NvFp4DownBlockN + column] =
          residual_value(row, column);
    }
  }
  if (!cuda_ok(cudaMemcpy(device_storage, initial.data(),
                          total_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy residual H2D")) {
    return false;
  }

  std::uint16_t* const device_residual = device_storage + kGuardElements;
  const int launch_status =
      q3x::kernels::launch_sm87_macrofeed_v4_nvfp4_down_tile_test_cuda(
          device_input, device_payload, tensor_scale, valid_rows,
          canonical_n_half, device_residual, nullptr);
  if (launch_status != static_cast<int>(cudaSuccess)) {
    std::cerr << "tile launch: "
              << cudaGetErrorString(static_cast<cudaError_t>(launch_status))
              << '\n';
    return false;
  }
  if (!cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    return false;
  }

  std::vector<std::uint16_t> actual(total_elements, 0U);
  if (!cuda_ok(cudaMemcpy(actual.data(), device_storage,
                          total_elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy residual D2H")) {
    return false;
  }
  for (std::size_t index = 0U; index < kGuardElements; ++index) {
    if (actual[index] != kGuardBits ||
        actual[kGuardElements + body_elements + index] != kGuardBits) {
      std::cerr << "guard overwritten for valid_rows=" << valid_rows
                << " n_half=" << canonical_n_half << '\n';
      return false;
    }
  }
  for (std::size_t row = 0U; row < kSm87MacroFeedV4NvFp4DownBlockM;
       ++row) {
    for (std::size_t column = 0U;
         column < kSm87MacroFeedV4NvFp4DownBlockN; ++column) {
      const std::size_t index =
          kGuardElements +
          row * kSm87MacroFeedV4NvFp4DownBlockN + column;
      const std::uint16_t expected =
          row < valid_rows
              ? expected_value(input, row, column, canonical_n_half,
                               payload, initial[index], tensor_scale)
              : initial[index];
      if (actual[index] != expected) {
        std::cerr << "bit mismatch valid_rows=" << valid_rows
                  << " n_half=" << canonical_n_half << " row=" << row
                  << " column=" << column << " expected=0x" << std::hex
                  << expected << " actual=0x" << actual[index] << std::dec
                  << '\n';
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t device_count_status = cudaGetDeviceCount(&device_count);
  if (device_count_status != cudaSuccess || device_count == 0) {
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

  constexpr auto kPlan =
      q3x::kernels::sm87_macrofeed_v4_nvfp4_down_plan(8'000U);
  static_assert(kPlan.valid());
  if (!kPlan.valid() || kPlan.stream_k || !kPlan.m_major_n_adjacent ||
      !kPlan.ordinary_full_grid) {
    std::cerr << "FAIL: V4 plan contract is not the sealed full-grid raster\n";
    return 1;
  }

  q3x::kernels::Sm87MacroFeedV4NvFp4DownCudaResources resources{};
  const int query_status =
      q3x::kernels::query_sm87_macrofeed_v4_nvfp4_down_cuda_resources(
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
      !q3x::kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(resources)) {
    std::cerr << "FAIL: V4 two-CTA/SM resource gate rejected kernel\n";
    return 1;
  }

  const std::size_t input_elements =
      kSm87MacroFeedV4NvFp4DownBlockM *
      kSm87MacroFeedV4NvFp4DownTestInputFeatures;
  std::vector<std::uint16_t> input(input_elements, 0U);
  std::vector<std::uint8_t> payload(
      kSm87MacroFeedV4NvFp4DownTestPayloadBytes, 0U);
  build_input(input);
  build_payload(payload);

  std::uint16_t* device_input = nullptr;
  std::uint8_t* device_payload = nullptr;
  std::uint16_t* device_storage = nullptr;
  const std::size_t body_elements =
      kSm87MacroFeedV4NvFp4DownBlockM *
      kSm87MacroFeedV4NvFp4DownBlockN;
  bool ok =
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_input),
                         input.size() * sizeof(std::uint16_t)),
              "cudaMalloc input") &&
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_payload),
                         payload.size()),
              "cudaMalloc payload") &&
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_storage),
                         (body_elements + 2U * kGuardElements) *
                             sizeof(std::uint16_t)),
              "cudaMalloc residual");
  if (ok) {
    ok = cuda_ok(cudaMemcpy(device_input, input.data(),
                            input.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy input H2D") &&
         cuda_ok(cudaMemcpy(device_payload, payload.data(), payload.size(),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy payload H2D");
  }

  constexpr float kTensorScale = 0.5F;
  if (ok) {
    const int invalid_rows =
        q3x::kernels::launch_sm87_macrofeed_v4_nvfp4_down_tile_test_cuda(
            device_input, device_payload, kTensorScale, 65U, 0U,
            device_storage + kGuardElements, nullptr);
    const int invalid_half =
        q3x::kernels::launch_sm87_macrofeed_v4_nvfp4_down_tile_test_cuda(
            device_input, device_payload, kTensorScale, 64U, 2U,
            device_storage + kGuardElements, nullptr);
    ok = invalid_rows == static_cast<int>(cudaErrorInvalidValue) &&
         invalid_half == static_cast<int>(cudaErrorInvalidValue);
    if (!ok) {
      std::cerr << "FAIL: invalid tile geometry did not fail closed\n";
    }
  }
  if (ok) {
    ok = run_case(64U, 0U, input, payload, device_input, device_payload,
                  device_storage, kTensorScale);
  }
  if (ok) {
    ok = run_case(37U, 1U, input, payload, device_input, device_payload,
                  device_storage, kTensorScale);
  }

  if (device_storage != nullptr) {
    (void)cudaFree(device_storage);
  }
  if (device_payload != nullptr) {
    (void)cudaFree(device_payload);
  }
  if (device_input != nullptr) {
    (void)cudaFree(device_input);
  }
  if (ok) {
    std::cout << "sm87_macrofeed_v4_nvfp4_down_cuda_test: PASS\n";
  }
  return ok ? 0 : 1;
}
