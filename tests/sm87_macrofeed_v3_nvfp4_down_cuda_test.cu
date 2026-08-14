#include "q3x/kernels/sm87_macrofeed_v3_nvfp4_down.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using q3x::kernels::kSm87MacroFeedV3NvFp4DownBlockM;
using q3x::kernels::kSm87MacroFeedV3NvFp4DownBlockN;
using q3x::kernels::kSm87MacroFeedV3NvFp4DownCellBytes;
using q3x::kernels::kSm87MacroFeedV3NvFp4DownScaleBytesPerCell;
using q3x::kernels::kSm87MacroFeedV3NvFp4DownTestInputFeatures;
using q3x::kernels::kSm87MacroFeedV3NvFp4DownTestKTiles;
using q3x::kernels::kSm87MacroFeedV3NvFp4DownTestPayloadBytes;
using q3x::kernels::kSm87MacroFeedV3NvFp4DownWeightBytesPerCell;

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
  float value = exponent == 0U
                    ? std::ldexp(static_cast<float>(mantissa), -9)
                    : std::ldexp(static_cast<float>(8U + mantissa),
                                 static_cast<int>(exponent) - 10);
  return (code & 0x80U) != 0U ? -value : value;
}

[[nodiscard]] std::uint8_t weight_code(const std::size_t stage,
                                       const std::size_t k16,
                                       const std::size_t n_warp,
                                       const std::size_t n8) {
  constexpr std::array<std::uint8_t, 4U> kCodes{{1U, 2U, 4U, 9U}};
  return kCodes[(stage + 2U * k16 + n_warp + 3U * n8) % kCodes.size()];
}

[[nodiscard]] std::uint8_t scale_code(const std::size_t stage,
                                      const std::size_t k16,
                                      const std::size_t n_warp,
                                      const std::size_t n8) {
  constexpr std::array<std::uint8_t, 4U> kCodes{{0x30U, 0x38U, 0x40U,
                                                 0x48U}};
  return kCodes[(2U * stage + k16 + 3U * n_warp + n8) % kCodes.size()];
}

[[nodiscard]] float activation_value(const std::size_t row,
                                     const std::size_t stage,
                                     const std::size_t k16) {
  constexpr std::array<float, 4U> kRowScale{{1.0F, 0.5F, -1.0F, 2.0F}};
  constexpr std::array<float, 4U> kKScale{{1.0F, -0.5F, 2.0F, 0.25F}};
  return kRowScale[row % kRowScale.size()] *
         kKScale[(stage + k16) % kKScale.size()];
}

[[nodiscard]] std::uint16_t residual_value(const std::size_t row,
                                           const std::size_t column) {
  constexpr std::array<float, 4U> kValues{{0.25F, -0.5F, 1.0F, 2.0F}};
  return encode_bf16_rne(kValues[(row + column / 8U) % kValues.size()]);
}

void build_input(std::vector<std::uint16_t>& input) {
  for (std::size_t row = 0U; row < kSm87MacroFeedV3NvFp4DownBlockM;
       ++row) {
    for (std::size_t stage = 0U;
         stage < kSm87MacroFeedV3NvFp4DownTestKTiles; ++stage) {
      for (std::size_t k16 = 0U; k16 < 4U; ++k16) {
        const std::uint16_t bits =
            encode_bf16_rne(activation_value(row, stage, k16));
        for (std::size_t element = 0U; element < 16U; ++element) {
          input[row * kSm87MacroFeedV3NvFp4DownTestInputFeatures +
                stage * 64U + k16 * 16U + element] = bits;
        }
      }
    }
  }
}

void build_payload(std::vector<std::uint8_t>& payload) {
  static_assert(kSm87MacroFeedV3NvFp4DownScaleBytesPerCell == 1'024U);
  for (std::size_t stage = 0U;
       stage < kSm87MacroFeedV3NvFp4DownTestKTiles; ++stage) {
    std::uint8_t* const cell =
        payload.data() + stage * kSm87MacroFeedV3NvFp4DownCellBytes;
    for (std::size_t k16 = 0U; k16 < 4U; ++k16) {
      for (std::size_t n_warp = 0U; n_warp < 4U; ++n_warp) {
        for (std::size_t n8 = 0U; n8 < 8U; ++n8) {
          const std::size_t fragment = (k16 * 4U + n_warp) * 8U + n8;
          const std::uint8_t code = weight_code(stage, k16, n_warp, n8);
          const std::uint16_t packed =
              static_cast<std::uint16_t>(code) * 0x1111U;
          for (std::size_t lane = 0U; lane < 32U; ++lane) {
            std::memcpy(cell + fragment * 64U + lane * 2U, &packed,
                        sizeof(packed));
          }
          const std::uint8_t scale =
              scale_code(stage, k16, n_warp, n8);
          for (std::size_t lane_group = 0U; lane_group < 8U;
               ++lane_group) {
            cell[kSm87MacroFeedV3NvFp4DownWeightBytesPerCell +
                 fragment * 8U + lane_group] = scale;
          }
        }
      }
    }
  }
}

[[nodiscard]] std::uint16_t expected_value(
    const std::vector<std::uint16_t>& input, const std::size_t row,
    const std::size_t column, const std::uint16_t residual,
    const float tensor_scale) {
  const std::size_t n_warp = column / 64U;
  const std::size_t n8 = (column % 64U) / 8U;
  float accumulator = 0.0F;
  for (std::size_t stage = 0U;
       stage < kSm87MacroFeedV3NvFp4DownTestKTiles; ++stage) {
    for (std::size_t k16 = 0U; k16 < 4U; ++k16) {
      const float weight = decode_bf16(encode_bf16_rne(
          decode_e2m1(weight_code(stage, k16, n_warp, n8)) *
          decode_e4m3fn(scale_code(stage, k16, n_warp, n8))));
      for (std::size_t element = 0U; element < 16U; ++element) {
        const float activation = decode_bf16(
            input[row * kSm87MacroFeedV3NvFp4DownTestInputFeatures +
                  stage * 64U + k16 * 16U + element]);
        accumulator += activation * weight;
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
              const std::vector<std::uint16_t>& input,
              const std::vector<std::uint8_t>& payload,
              const std::uint16_t* const device_input,
              const std::uint8_t* const device_payload,
              std::uint16_t* const device_storage,
              const float tensor_scale) {
  const std::size_t body_elements =
      kSm87MacroFeedV3NvFp4DownBlockM *
      kSm87MacroFeedV3NvFp4DownBlockN;
  const std::size_t total_elements = body_elements + 2U * kGuardElements;
  std::vector<std::uint16_t> initial(total_elements, kGuardBits);
  for (std::size_t row = 0U; row < kSm87MacroFeedV3NvFp4DownBlockM;
       ++row) {
    for (std::size_t column = 0U;
         column < kSm87MacroFeedV3NvFp4DownBlockN; ++column) {
      initial[kGuardElements +
              row * kSm87MacroFeedV3NvFp4DownBlockN + column] =
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
      q3x::kernels::launch_sm87_macrofeed_v3_nvfp4_down_tile_test_cuda(
          device_input, device_payload, tensor_scale, valid_rows,
          device_residual, nullptr);
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
      std::cerr << "guard overwritten for valid_rows=" << valid_rows << '\n';
      return false;
    }
  }
  for (std::size_t row = 0U; row < kSm87MacroFeedV3NvFp4DownBlockM;
       ++row) {
    for (std::size_t column = 0U;
         column < kSm87MacroFeedV3NvFp4DownBlockN; ++column) {
      const std::size_t index =
          kGuardElements + row * kSm87MacroFeedV3NvFp4DownBlockN + column;
      const std::uint16_t expected =
          row < valid_rows
              ? expected_value(input, row, column, initial[index],
                               tensor_scale)
              : initial[index];
      if (actual[index] != expected) {
        std::cerr << "bit mismatch valid_rows=" << valid_rows
                  << " row=" << row << " column=" << column
                  << " expected=0x" << std::hex << expected
                  << " actual=0x" << actual[index] << std::dec << '\n';
        return false;
      }
    }
  }
  (void)payload;
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

  q3x::kernels::Sm87MacroFeedV3NvFp4DownCudaResources resources{};
  const int query_status =
      q3x::kernels::query_sm87_macrofeed_v3_nvfp4_down_cuda_resources(
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
      !q3x::kernels::sm87_macrofeed_v3_nvfp4_down_resource_gate(resources)) {
    std::cerr << "FAIL: independent V3 resource gate rejected kernel\n";
    return 1;
  }

  const std::size_t input_elements =
      kSm87MacroFeedV3NvFp4DownBlockM *
      kSm87MacroFeedV3NvFp4DownTestInputFeatures;
  std::vector<std::uint16_t> input(input_elements, 0U);
  std::vector<std::uint8_t> payload(
      kSm87MacroFeedV3NvFp4DownTestPayloadBytes, 0U);
  build_input(input);
  build_payload(payload);

  std::uint16_t* device_input = nullptr;
  std::uint8_t* device_payload = nullptr;
  std::uint16_t* device_storage = nullptr;
  const std::size_t body_elements =
      kSm87MacroFeedV3NvFp4DownBlockM *
      kSm87MacroFeedV3NvFp4DownBlockN;
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
    const int invalid_status =
        q3x::kernels::launch_sm87_macrofeed_v3_nvfp4_down_tile_test_cuda(
            device_input, device_payload, kTensorScale, 63U,
            device_storage + kGuardElements, nullptr);
    ok = invalid_status == static_cast<int>(cudaErrorInvalidValue);
    if (!ok) {
      std::cerr << "FAIL: non-M64/M128 test shape did not fail closed\n";
    }
  }
  if (ok) {
    ok = run_case(128U, input, payload, device_input, device_payload,
                  device_storage, kTensorScale);
  }
  if (ok) {
    ok = run_case(64U, input, payload, device_input, device_payload,
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
    std::cout << "sm87_macrofeed_v3_nvfp4_down_cuda_test: PASS\n";
  }
  return ok ? 0 : 1;
}
