#include "q3x/kernels/sm87_a4w4_down_factorized_r4_m192n128.h"
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

constexpr std::size_t kLogicalM = 1'853U;
constexpr std::size_t kLaunchM = 1'920U;
constexpr std::size_t kN = 5'120U;
constexpr std::size_t kK = 17'408U;
constexpr std::size_t kGroups = kK / 64U;
constexpr std::size_t kGroupsPerLane = kGroups / 4U;
constexpr std::uint16_t kGuard = 0x5555U;

template <typename T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  [[nodiscard]] bool allocate(const std::size_t elements) noexcept {
    return elements != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&pointer_),
                      elements * sizeof(T)) == cudaSuccess;
  }
  [[nodiscard]] T* get() const noexcept { return pointer_; }

 private:
  T* pointer_{};
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

// 1: target, 0: clean unsupported-device skip, -1: CUDA failure.
[[nodiscard]] int target_status() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      (status == cudaSuccess && count == 0)) {
    (void)cudaGetLastError();
    std::cout << "SKIP: CUDA target unavailable\n";
    return 0;
  }
  if (!cuda_ok(status, "cudaGetDeviceCount")) {
    return -1;
  }
  int device = -1;
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice")) {
    return -1;
  }
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return -1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: requires the 16-SM SM87 target\n";
    return 0;
  }
  return 1;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] int a_even(const std::size_t group) noexcept {
  const int magnitude = static_cast<int>(1U + (5U * group + 1U) % 7U);
  return group % 3U == 0U ? -magnitude : magnitude;
}

[[nodiscard]] int a_odd(const std::size_t group) noexcept {
  const int magnitude = static_cast<int>(1U + (2U * group + 3U) % 7U);
  return (group & 1U) == 0U ? magnitude : -magnitude;
}

[[nodiscard]] int b_even(const std::size_t n,
                         const std::size_t group) noexcept {
  return (n & 1U) == 0U ? a_even(group) : -a_even(group);
}

[[nodiscard]] int b_odd(const std::size_t n,
                        const std::size_t group) noexcept {
  return (n & 2U) == 0U ? a_odd(group) : -a_odd(group);
}

[[nodiscard]] float row_factor(const std::size_t m) noexcept {
  return static_cast<float>(1U << ((m / 64U + m % 5U) % 3U));
}

[[nodiscard]] float column_factor(const std::size_t n) noexcept {
  return static_cast<float>(1U << ((n / 64U + n % 3U) % 3U));
}

[[nodiscard]] std::uint16_t expected_value(const std::size_t m,
                                           const std::size_t n) noexcept {
  const std::size_t row_class = (m / 64U + m % 5U) % 3U;
  const std::size_t column_class = (n / 64U + n % 3U) % 3U;
  const std::size_t code_class = n % 4U;
  static std::uint16_t cache[3][3][4]{};
  static bool initialized[3][3][4]{};
  if (initialized[row_class][column_class][code_class]) {
    return cache[row_class][column_class][code_class];
  }

  constexpr std::array<float, 4> kAScale = {0.125F, 0.25F, 0.5F, 1.0F};
  constexpr std::array<float, 4> kBScale = {1.0F, 2.0F, 4.0F, 8.0F};
  float result = 0.0F;
  for (std::size_t lane = 0U; lane < 4U; ++lane) {
    std::int32_t partial = 0;
    for (std::size_t in_lane = 0U; in_lane < kGroupsPerLane;
         ++in_lane) {
      const std::size_t group = lane * kGroupsPerLane + in_lane;
      partial += 32 *
                 (a_even(group) * b_even(n, group) +
                  a_odd(group) * b_odd(n, group));
    }
    const volatile float scale =
        (kAScale[lane] * row_factor(m)) *
        (kBScale[lane] * column_factor(n));
    const volatile float term = static_cast<float>(partial) * scale;
    result = result + term;
  }
  const std::uint16_t encoded = encode_bf16(result);
  cache[row_class][column_class][code_class] = encoded;
  initialized[row_class][column_class][code_class] = true;
  return encoded;
}

void fill_codes(std::vector<std::uint8_t>& packed_a,
                std::vector<std::uint8_t>& packed_b) {
  for (std::size_t m = 0U; m < kLaunchM; ++m) {
    for (std::size_t group = 0U; group < kGroups; ++group) {
      const std::uint8_t code = kernels::sm87_a4w4_pack_signed_pair(
          a_even(group), a_odd(group));
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        packed_a[kernels::sm87_a4w4_consumer_packed_offset(
            m, group, byte, kGroups)] = code;
      }
    }
  }
  for (std::size_t n = 0U; n < kN; ++n) {
    for (std::size_t group = 0U; group < kGroups; ++group) {
      const std::uint8_t code = kernels::sm87_a4w4_pack_signed_pair(
          b_even(n, group), b_odd(n, group));
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        packed_b[kernels::sm87_a4w4_consumer_packed_offset(
            n, group, byte, kGroups)] = code;
      }
    }
  }
}

void fill_scales(std::vector<std::uint16_t>& a_scales,
                 std::vector<std::uint16_t>& b_scales) {
  constexpr std::array<float, 4> kAScale = {0.125F, 0.25F, 0.5F, 1.0F};
  constexpr std::array<float, 4> kBScale = {1.0F, 2.0F, 4.0F, 8.0F};
  for (std::size_t m = 0U; m < kLaunchM; ++m) {
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      a_scales[kernels::
          sm87_a4w4_down_factorized_r4_m192n128_scale_offset(m, lane)] =
          encode_bf16(kAScale[lane] * row_factor(m));
    }
  }
  for (std::size_t n = 0U; n < kN; ++n) {
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      b_scales[kernels::
          sm87_a4w4_down_factorized_r4_m192n128_scale_offset(n, lane)] =
          encode_bf16(kBScale[lane] * column_factor(n));
    }
  }
}

[[nodiscard]] bool reject(const int status, const char* const label) {
  if (status == static_cast<int>(cudaErrorInvalidValue)) {
    return true;
  }
  std::cerr << label << " did not fail closed: " << status << '\n';
  return false;
}

[[nodiscard]] bool run() {
  constexpr std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kLaunchM, kK);
  constexpr std::size_t b_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kN, kK);
  constexpr std::size_t a_scale_count = kernels::
      sm87_a4w4_down_factorized_r4_m192n128_scale_capacity_elements(
          kLaunchM);
  constexpr std::size_t b_scale_count = kernels::
      sm87_a4w4_down_factorized_r4_m192n128_scale_capacity_elements(kN);
  constexpr std::size_t output_count = kLaunchM * kN;

  kernels::Sm87A4W4DownFactorizedR4M192N128Resources resources{};
  if (!cuda_ok(static_cast<cudaError_t>(kernels::
                   query_sm87_a4w4_down_factorized_r4_m192n128_resources_cuda(
                       &resources)),
               "Down R4 resource query") ||
      resources.registers_per_thread > 168 || resources.local_bytes != 0U ||
      resources.active_blocks_per_sm < 1 || resources.resident_blocks < 16) {
    std::cerr << "Down R4 resource hard gate failed\n";
    return false;
  }

  std::vector<std::uint8_t> packed_a(a_bytes);
  std::vector<std::uint8_t> packed_b(b_bytes);
  std::vector<std::uint16_t> a_scales(a_scale_count);
  std::vector<std::uint16_t> b_scales(b_scale_count);
  fill_codes(packed_a, packed_b);
  fill_scales(a_scales, b_scales);

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint8_t> device_b;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint16_t> device_b_scales;
  DeviceBuffer<std::uint16_t> device_output;
  if (!device_a.allocate(a_bytes) || !device_b.allocate(b_bytes) ||
      !device_a_scales.allocate(a_scale_count) ||
      !device_b_scales.allocate(b_scale_count) ||
      !device_output.allocate(output_count)) {
    std::cerr << "Down R4 release-shape allocation failed\n";
    return false;
  }

  cudaStream_t stream{};
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create nondefault stream")) {
    return false;
  }
  const auto destroy_stream = [&]() { (void)cudaStreamDestroy(stream); };
  const auto copy = [&](void* destination, const void* source,
                        const std::size_t bytes, const char* label) {
    return cuda_ok(cudaMemcpyAsync(destination, source, bytes,
                                   cudaMemcpyHostToDevice, stream),
                   label);
  };
  if (!copy(device_a.get(), packed_a.data(), a_bytes, "copy A") ||
      !copy(device_b.get(), packed_b.data(), b_bytes, "copy B") ||
      !copy(device_a_scales.get(), a_scales.data(),
            a_scale_count * sizeof(std::uint16_t), "copy A scales") ||
      !copy(device_b_scales.get(), b_scales.data(),
            b_scale_count * sizeof(std::uint16_t), "copy B scales") ||
      !cuda_ok(cudaMemsetAsync(device_output.get(), 0x55,
                              output_count * sizeof(std::uint16_t), stream),
               "initialize output") ||
      !cuda_ok(cudaStreamSynchronize(stream), "prepare release-shape data")) {
    destroy_stream();
    return false;
  }

  const auto launch = [&](const std::uint8_t* a, std::size_t a_capacity,
                          const std::uint16_t* a_scale,
                          std::size_t a_scale_capacity,
                          const std::uint8_t* b, std::size_t b_capacity,
                          std::uint16_t* output, std::size_t stride,
                          std::size_t output_capacity,
                          std::size_t logical = kLogicalM,
                          std::size_t launch_m = kLaunchM,
                          std::size_t n = kN,
                          std::size_t k = kK) {
    return kernels::
        launch_sm87_a4w4_down_factorized_r4_m192n128_bf16_cuda(
            a, a_capacity, a_scale, a_scale_capacity, b, b_capacity,
            device_b_scales.get(), b_scale_count, logical, launch_m, n, k,
            output, stride, output_capacity, stream);
  };

  const bool negatives =
      reject(launch(nullptr, a_bytes, device_a_scales.get(), a_scale_count,
                    device_b.get(), b_bytes, device_output.get(), kN,
                    output_count),
             "null A") &&
      reject(launch(device_a.get(), a_bytes - 1U, device_a_scales.get(),
                    a_scale_count, device_b.get(), b_bytes,
                    device_output.get(), kN, output_count),
             "short A") &&
      reject(launch(device_a.get(), a_bytes,
                    reinterpret_cast<const std::uint16_t*>(
                        reinterpret_cast<const std::uint8_t*>(
                            device_a_scales.get()) + 2U),
                    a_scale_count, device_b.get(), b_bytes,
                    device_output.get(), kN, output_count),
             "misaligned A scale") &&
      reject(launch(device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_count, device_b.get(), b_bytes,
                    device_output.get(), kN + 1U, output_count),
             "wrong output stride") &&
      reject(launch(reinterpret_cast<const std::uint8_t*>(
                        device_output.get()),
                    a_bytes, device_a_scales.get(), a_scale_count,
                    device_b.get(), b_bytes, device_output.get(), kN,
                    output_count),
             "input/output alias") &&
      reject(launch(device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_count, device_b.get(), b_bytes,
                    device_output.get(), kN, output_count, 1'792U),
             "logical token lower bound") &&
      reject(launch(device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_count, device_b.get(), b_bytes,
                    device_output.get(), kN, output_count, kLogicalM,
                    2'048U),
             "wrong launch token count") &&
      reject(launch(device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_count, device_b.get(), b_bytes,
                    device_output.get(), kN, output_count, kLogicalM,
                    kLaunchM, kN + 128U),
             "wrong output shape") &&
      reject(launch(device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_count, device_b.get(), b_bytes,
                    device_output.get(), kN, output_count, kLogicalM,
                    kLaunchM, kN, kK + 64U),
             "wrong input shape");
  const bool capacity_negatives =
      reject(kernels::
                 launch_sm87_a4w4_down_factorized_r4_m192n128_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count - 1U, device_b.get(), b_bytes,
                     device_b_scales.get(), b_scale_count, kLogicalM,
                     kLaunchM, kN, kK, device_output.get(), kN,
                     output_count, stream),
             "short A scale") &&
      reject(kernels::
                 launch_sm87_a4w4_down_factorized_r4_m192n128_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count, device_b.get(), b_bytes - 1U,
                     device_b_scales.get(), b_scale_count, kLogicalM,
                     kLaunchM, kN, kK, device_output.get(), kN,
                     output_count, stream),
             "short B code") &&
      reject(kernels::
                 launch_sm87_a4w4_down_factorized_r4_m192n128_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count, device_b.get(), b_bytes,
                     device_b_scales.get(), b_scale_count - 1U, kLogicalM,
                     kLaunchM, kN, kK, device_output.get(), kN,
                     output_count, stream),
             "short B scale") &&
      reject(kernels::
                 launch_sm87_a4w4_down_factorized_r4_m192n128_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count, device_b.get(), b_bytes,
                     device_b_scales.get(), b_scale_count, kLogicalM,
                     kLaunchM, kN, kK, device_output.get(), kN,
                     output_count - 1U, stream),
             "short output");
  if (!negatives || !capacity_negatives) {
    destroy_stream();
    return false;
  }

  const int launch_status = launch(
      device_a.get(), a_bytes, device_a_scales.get(), a_scale_count,
      device_b.get(), b_bytes, device_output.get(), kN, output_count);
  if (!cuda_ok(static_cast<cudaError_t>(launch_status),
               "launch Down R4 release shape") ||
      !cuda_ok(cudaStreamSynchronize(stream),
               "synchronize Down R4 nondefault stream")) {
    destroy_stream();
    return false;
  }

  std::vector<std::uint16_t> output(output_count);
  if (!cuda_ok(cudaMemcpy(output.data(), device_output.get(),
                          output_count * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy Down result")) {
    destroy_stream();
    return false;
  }
  destroy_stream();

  for (std::size_t m = 0U; m < kLaunchM; ++m) {
    for (std::size_t n = 0U; n < kN; ++n) {
      const std::uint16_t expected =
          m < kLogicalM ? expected_value(m, n) : kGuard;
      const std::uint16_t actual = output[m * kN + n];
      if (actual != expected) {
        std::cerr << "Down mismatch m=" << m << " n=" << n
                  << " actual=0x" << std::hex << actual << " expected=0x"
                  << expected << std::dec << '\n';
        return false;
      }
    }
  }

  std::cout << "PASS: Down R4 M192N128 release-shape correctness; "
               "P1853->P1920 tail, K-lane/stage boundaries, M/N owners, "
               "guards, nondefault stream, launch negatives, and resource "
               "gates\n";
  return true;
}

}  // namespace

int main() {
  const int target = target_status();
  if (target <= 0) {
    return target == 0 ? 0 : 1;
  }
  return run() ? 0 : 1;
}
