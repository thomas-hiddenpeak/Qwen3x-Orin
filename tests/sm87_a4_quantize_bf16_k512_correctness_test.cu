#include "q3x/kernels/sm87_a4w4_attention_o_k512_cell.h"
#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

constexpr std::size_t kLogicalM = 129U;
constexpr std::size_t kLaunchM = 256U;
constexpr std::size_t kInputK = 6'144U;
constexpr std::size_t kInputStride = kInputK + 13U;
constexpr float kClipRatio = 0.91F;
constexpr std::size_t kByteGuard = 64U;
constexpr std::size_t kElementGuard = 32U;

static_assert(
    kernels::sm87_a4w4_prefill_k512_launch_token_count(kLogicalM) ==
    kLaunchM);
static_assert(kernels::sm87_a4w4_prefill_k512_group_count(kInputK) == 12U);
static_assert(
    kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(kLaunchM,
                                                            kInputK) ==
    kernels::sm87_a4w4_attention_o_k512_scale_capacity_elements(kLaunchM,
                                                                kInputK));
static_assert(
    kernels::sm87_a4w4_consumer_packed_capacity_bytes(kLaunchM, kInputK) ==
    kernels::sm87_a4w4_attention_o_k512_packed_capacity_bytes(kLaunchM,
                                                             kInputK));
static_assert(kernels::sm87_a4w4_prefill_k512_scale_offset(128U, 7U, 12U) ==
              kernels::sm87_a4w4_attention_o_k512_scale_offset(128U, 7U,
                                                               12U));
static_assert(kernels::sm87_a4w4_consumer_packed_offset(128U, 17U, 23U,
                                                       96U) ==
              kernels::sm87_a4w4_attention_o_k512_packed_offset(
                  128U, 17U, 23U, 96U));

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

  [[nodiscard]] bool allocate(const std::size_t elements) noexcept {
    return elements != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&data_),
                      elements * sizeof(T)) == cudaSuccess;
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

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
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

[[nodiscard]] bool device_is_target() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      count == 0) {
    (void)cudaGetLastError();
    std::cout << "SKIP: CUDA device unavailable\n";
    return false;
  }
  if (!cuda_ok(status, "cudaGetDeviceCount")) {
    return false;
  }
  int device = 0;
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice")) {
    return false;
  }
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return false;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: requires the 16-SM SM87 target\n";
    return false;
  }
  return true;
}

[[nodiscard]] int nearest_even_code(const float value,
                                    const float clipped_maximum,
                                    const float stored_scale) noexcept {
  const float clipped =
      std::fmin(std::fmax(value, -clipped_maximum), clipped_maximum);
  const int rounded = stored_scale == 0.0F
                          ? 0
                          : static_cast<int>(
                                std::nearbyint(clipped / stored_scale));
  return std::max(-7, std::min(7, rounded));
}

struct Oracle final {
  std::vector<std::uint8_t> packed;
  std::vector<std::uint16_t> scales;
  std::size_t non_power_of_two_input_count{};
  std::size_t non_power_of_two_scale_count{};
};

[[nodiscard]] Oracle make_oracle(const std::uint16_t* const input) {
  constexpr std::size_t k512Groups = kInputK / 512U;
  constexpr std::size_t k64Groups = kInputK / 64U;
  constexpr std::size_t packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kLaunchM, kInputK);
  constexpr std::size_t scale_elements =
      kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(kLaunchM,
                                                              kInputK);
  Oracle result{std::vector<std::uint8_t>(packed_bytes, 0xcdU),
                std::vector<std::uint16_t>(scale_elements, 0xdeadU)};
  for (std::size_t row = 0U; row < kLaunchM; ++row) {
    const bool valid_row = row < kLogicalM;
    for (std::size_t group = 0U; group < k512Groups; ++group) {
      float maximum = 0.0F;
      if (valid_row) {
        for (std::size_t inner = 0U; inner < 512U; ++inner) {
          const std::uint16_t bits =
              input[row * kInputStride + group * 512U + inner];
          if ((bits & 0x7fU) != 0U) {
            ++result.non_power_of_two_input_count;
          }
          maximum = std::fmax(maximum, std::fabs(decode_bf16(bits)));
        }
      }
      const float clipped_maximum = maximum * kClipRatio;
      std::uint16_t scale_bits =
          encode_bf16(maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
      float stored_scale = decode_bf16(scale_bits);
      if (maximum != 0.0F && stored_scale == 0.0F) {
        scale_bits = 1U;
        stored_scale = decode_bf16(scale_bits);
      }
      if (valid_row && maximum != 0.0F && (scale_bits & 0x7fU) != 0U) {
        ++result.non_power_of_two_scale_count;
      }
      result.scales[kernels::sm87_a4w4_prefill_k512_scale_offset(
          row, group, k512Groups)] = scale_bits;

      for (std::size_t inner = 0U; inner < 512U; inner += 2U) {
        const float even =
            valid_row
                ? decode_bf16(input[row * kInputStride + group * 512U +
                                    inner])
                : 0.0F;
        const float odd =
            valid_row
                ? decode_bf16(input[row * kInputStride + group * 512U +
                                    inner + 1U])
                : 0.0F;
        const int even_code =
            nearest_even_code(even, clipped_maximum, stored_scale);
        const int odd_code =
            nearest_even_code(odd, clipped_maximum, stored_scale);
        const std::size_t global_k = group * 512U + inner;
        result.packed[kernels::sm87_a4w4_consumer_packed_offset(
            row, global_k / 64U, (global_k % 64U) / 2U, k64Groups)] =
            kernels::sm87_a4w4_pack_signed_pair(even_code, odd_code);
      }
    }
  }
  return result;
}

[[nodiscard]] bool expect_invalid(const int status,
                                  const std::string& operation) {
  if (status == static_cast<int>(cudaErrorInvalidValue)) {
    return true;
  }
  std::cerr << operation << " did not fail closed, status=" << status
            << '\n';
  return false;
}

[[nodiscard]] bool run_correctness() {
  constexpr std::uint16_t input_guard = 0x7fc1U;
  constexpr std::uint8_t packed_guard = 0xa5U;
  constexpr std::uint16_t scale_guard = 0xbeefU;
  constexpr std::size_t input_elements = kLogicalM * kInputStride;
  constexpr std::size_t packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kLaunchM, kInputK);
  constexpr std::size_t scale_elements =
      kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(kLaunchM,
                                                              kInputK);

  std::vector<std::uint16_t> host_input_storage(
      2U * kElementGuard + input_elements, input_guard);
  std::uint16_t* const host_input =
      host_input_storage.data() + kElementGuard;
  for (std::size_t row = 0U; row < kLogicalM; ++row) {
    for (std::size_t inner = 0U; inner < kInputK; ++inner) {
      const std::size_t group = inner / 512U;
      float value = 0.0F;
      if (!(row == 0U && group == 0U)) {
        const int centered = static_cast<int>(
                                 (row * 1'009U + inner * 313U +
                                  (row + 3U) * (inner % 29U)) %
                                 2'003U) -
                             1'001;
        const float row_gain =
            0.77F + 0.013F * static_cast<float>((row + group) % 17U);
        value = static_cast<float>(centered) * 0.00317F * row_gain;
      }
      host_input[row * kInputStride + inner] = encode_bf16(value);
    }
  }
  host_input[1U * kInputStride] = encode_bf16(3.1415927F);
  host_input[1U * kInputStride + 1U] = encode_bf16(-2.7182817F);
  for (std::size_t group = 0U; group < kInputK / 512U; ++group) {
    host_input[(kLogicalM - 1U) * kInputStride + group * 512U + 37U] =
        encode_bf16(5.371F + 0.137F * static_cast<float>(group));
  }
  const std::vector<std::uint16_t> original_input_storage =
      host_input_storage;
  const Oracle oracle = make_oracle(host_input);
  if (oracle.non_power_of_two_input_count == 0U ||
      oracle.non_power_of_two_scale_count == 0U) {
    std::cerr << "oracle failed to exercise non-power-of-two BF16 data"
              << '\n';
    return false;
  }

  std::vector<std::uint8_t> packed_initial(
      2U * kByteGuard + packed_bytes, packed_guard);
  std::vector<std::uint16_t> scale_initial(
      2U * kElementGuard + scale_elements, scale_guard);
  DeviceBuffer<std::uint16_t> device_input_storage;
  DeviceBuffer<std::uint8_t> device_packed_storage;
  DeviceBuffer<std::uint16_t> device_scale_storage;
  if (!device_input_storage.allocate(host_input_storage.size()) ||
      !device_packed_storage.allocate(packed_initial.size()) ||
      !device_scale_storage.allocate(scale_initial.size())) {
    std::cerr << "K512 quantizer device allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_input_storage.get(),
                          host_input_storage.data(),
                          host_input_storage.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy guarded K512 input") ||
      !cuda_ok(cudaMemcpy(device_packed_storage.get(), packed_initial.data(),
                          packed_initial.size(), cudaMemcpyHostToDevice),
               "initialize guarded K512 packed output") ||
      !cuda_ok(cudaMemcpy(device_scale_storage.get(), scale_initial.data(),
                          scale_initial.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "initialize guarded K512 scale output")) {
    return false;
  }

  std::uint16_t* const device_input =
      device_input_storage.get() + kElementGuard;
  std::uint8_t* const device_packed =
      device_packed_storage.get() + kByteGuard;
  std::uint16_t* const device_scales =
      device_scale_storage.get() + kElementGuard;
  if (!expect_invalid(kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
                          device_input, kInputStride, kLogicalM,
                          kLaunchM - 128U, kInputK, kClipRatio, device_packed,
                          packed_bytes, device_scales, scale_elements),
                      "short launch_M") ||
      !expect_invalid(kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
                          device_input, kInputStride, kLogicalM,
                          kLaunchM + 128U, kInputK, kClipRatio, device_packed,
                          packed_bytes, device_scales, scale_elements),
                      "oversized launch_M") ||
      !expect_invalid(kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
                          device_input, kInputStride, kLogicalM, kLaunchM,
                          kInputK - 1U, kClipRatio, device_packed,
                          packed_bytes, device_scales, scale_elements),
                      "non-K512 input") ||
      !expect_invalid(kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
                          device_input, kInputStride, kLogicalM, kLaunchM,
                          kInputK, kClipRatio, device_packed,
                          packed_bytes - 1U, device_scales, scale_elements),
                      "short packed capacity") ||
      !expect_invalid(kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
                          device_input, kInputStride, kLogicalM, kLaunchM,
                          kInputK, kClipRatio, device_packed, packed_bytes,
                          device_scales, scale_elements - 1U),
                      "short scale capacity") ||
      !expect_invalid(kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
                          device_input, kInputStride, 0U, 0U, kInputK,
                          kClipRatio, device_packed, packed_bytes,
                          device_scales, scale_elements),
                      "zero logical_M")) {
    return false;
  }

  const int launch_status =
      kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
          device_input, kInputStride, kLogicalM, kLaunchM, kInputK,
          kClipRatio, device_packed, packed_bytes, device_scales,
          scale_elements);
  if (!cuda_ok(static_cast<cudaError_t>(launch_status),
               "launch K512 activation quantizer") ||
      !cuda_ok(cudaDeviceSynchronize(),
               "synchronize K512 activation quantizer")) {
    return false;
  }

  std::vector<std::uint16_t> actual_input_storage(host_input_storage.size());
  std::vector<std::uint8_t> actual_packed_storage(packed_initial.size());
  std::vector<std::uint16_t> actual_scale_storage(scale_initial.size());
  if (!cuda_ok(cudaMemcpy(actual_input_storage.data(),
                          device_input_storage.get(),
                          actual_input_storage.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy guarded K512 input back") ||
      !cuda_ok(cudaMemcpy(actual_packed_storage.data(),
                          device_packed_storage.get(),
                          actual_packed_storage.size(),
                          cudaMemcpyDeviceToHost),
               "copy guarded K512 packed output") ||
      !cuda_ok(cudaMemcpy(actual_scale_storage.data(),
                          device_scale_storage.get(),
                          actual_scale_storage.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy guarded K512 scales")) {
    return false;
  }

  if (actual_input_storage != original_input_storage) {
    std::cerr << "K512 quantizer modified its guarded input\n";
    return false;
  }
  if (!std::all_of(actual_packed_storage.begin(),
                   actual_packed_storage.begin() + kByteGuard,
                   [](const std::uint8_t value) {
                     return value == packed_guard;
                   }) ||
      !std::all_of(actual_packed_storage.end() - kByteGuard,
                   actual_packed_storage.end(),
                   [](const std::uint8_t value) {
                     return value == packed_guard;
                   }) ||
      !std::all_of(actual_scale_storage.begin(),
                   actual_scale_storage.begin() + kElementGuard,
                   [](const std::uint16_t value) {
                     return value == scale_guard;
                   }) ||
      !std::all_of(actual_scale_storage.end() - kElementGuard,
                   actual_scale_storage.end(),
                   [](const std::uint16_t value) {
                     return value == scale_guard;
                   })) {
    std::cerr << "K512 quantizer wrote outside an output capacity\n";
    return false;
  }

  const std::uint8_t* const actual_packed =
      actual_packed_storage.data() + kByteGuard;
  const std::uint16_t* const actual_scales =
      actual_scale_storage.data() + kElementGuard;
  std::size_t packed_mismatches = 0U;
  for (std::size_t index = 0U; index < packed_bytes; ++index) {
    if (actual_packed[index] != oracle.packed[index]) {
      ++packed_mismatches;
      if (packed_mismatches <= 8U) {
        std::cerr << "packed mismatch index=" << index << " expected="
                  << static_cast<unsigned int>(oracle.packed[index])
                  << " actual=" << static_cast<unsigned int>(
                         actual_packed[index])
                  << '\n';
      }
    }
  }
  std::size_t scale_mismatches = 0U;
  for (std::size_t index = 0U; index < scale_elements; ++index) {
    if (actual_scales[index] != oracle.scales[index]) {
      ++scale_mismatches;
      if (scale_mismatches <= 8U) {
        std::cerr << "scale mismatch index=" << index << " expected=0x"
                  << std::hex << oracle.scales[index] << " actual=0x"
                  << actual_scales[index] << std::dec << '\n';
      }
    }
  }

  constexpr std::uint16_t bf16_one = 0x3f80U;
  std::size_t padded_code_violations = 0U;
  std::size_t padded_scale_violations = 0U;
  constexpr std::size_t k64Groups = kInputK / 64U;
  constexpr std::size_t k512Groups = kInputK / 512U;
  for (std::size_t row = kLogicalM; row < kLaunchM; ++row) {
    for (std::size_t group = 0U; group < k64Groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        if (actual_packed[kernels::sm87_a4w4_consumer_packed_offset(
                row, group, byte, k64Groups)] != 0U) {
          ++padded_code_violations;
        }
      }
    }
    for (std::size_t group = 0U; group < k512Groups; ++group) {
      if (actual_scales[kernels::sm87_a4w4_prefill_k512_scale_offset(
              row, group, k512Groups)] != bf16_one) {
        ++padded_scale_violations;
      }
    }
  }
  if (packed_mismatches != 0U || scale_mismatches != 0U ||
      padded_code_violations != 0U || padded_scale_violations != 0U) {
    std::cerr << "K512 quantizer FAIL: packed_mismatches="
              << packed_mismatches
              << " scale_mismatches=" << scale_mismatches
              << " padded_code_violations=" << padded_code_violations
              << " padded_scale_violations=" << padded_scale_violations
              << '\n';
    return false;
  }

  std::cout << "K512 activation quantizer PASS: logical_M=" << kLogicalM
            << " launch_M=" << kLaunchM << " K=" << kInputK
            << " non_power_input=" << oracle.non_power_of_two_input_count
            << " non_power_scale=" << oracle.non_power_of_two_scale_count
            << " padded_rows=" << (kLaunchM - kLogicalM) << '\n';
  return true;
}

}  // namespace

int main() {
  if (std::fesetround(FE_TONEAREST) != 0) {
    std::cerr << "failed to select host nearest-even rounding\n";
    return 1;
  }
  if (!device_is_target()) {
    return 0;
  }
  return run_correctness() ? 0 : 1;
}
