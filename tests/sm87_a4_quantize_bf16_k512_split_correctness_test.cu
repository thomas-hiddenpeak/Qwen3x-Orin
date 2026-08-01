#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

constexpr std::size_t kLogicalM = 3U;
constexpr std::size_t kLaunchM = 128U;
constexpr std::size_t kPrimaryK = 1'024U;
constexpr std::size_t kSecondaryK = 512U;
constexpr std::size_t kTotalK = kPrimaryK + kSecondaryK;
constexpr std::size_t kPrimaryStride = kPrimaryK + 7U;
constexpr std::size_t kSecondaryStride = kSecondaryK + 11U;
constexpr std::size_t kContiguousStride = kTotalK + 13U;
constexpr float kClipRatio = 0.93F;

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

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const char* const operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorName(status) << " ("
            << cudaGetErrorString(status) << ")\n";
  return false;
}

[[nodiscard]] int run() {
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

  std::vector<std::uint16_t> primary(kLogicalM * kPrimaryStride, 0U);
  std::vector<std::uint16_t> secondary(kLogicalM * kSecondaryStride, 0U);
  std::vector<std::uint16_t> contiguous(kLogicalM * kContiguousStride, 0U);
  for (std::size_t row = 0U; row < kLogicalM; ++row) {
    for (std::size_t column = 0U; column < kTotalK; ++column) {
      const int centered = static_cast<int>(
                               (row * 911U + column * 313U +
                                (row + 5U) * (column % 37U)) %
                               2'003U) -
                           1'001;
      const std::uint16_t bits =
          encode_bf16(static_cast<float>(centered) * 0.00371F);
      contiguous[row * kContiguousStride + column] = bits;
      if (column < kPrimaryK) {
        primary[row * kPrimaryStride + column] = bits;
      } else {
        secondary[row * kSecondaryStride + column - kPrimaryK] = bits;
      }
    }
  }

  constexpr std::size_t packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kLaunchM, kTotalK);
  constexpr std::size_t scale_elements =
      kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(kLaunchM,
                                                              kTotalK);
  DeviceBuffer<std::uint16_t> device_primary;
  DeviceBuffer<std::uint16_t> device_secondary;
  DeviceBuffer<std::uint16_t> device_contiguous;
  DeviceBuffer<std::uint8_t> reference_packed;
  DeviceBuffer<std::uint16_t> reference_scales;
  DeviceBuffer<std::uint8_t> split_packed;
  DeviceBuffer<std::uint16_t> split_scales;
  if (!device_primary.allocate(primary.size()) ||
      !device_secondary.allocate(secondary.size()) ||
      !device_contiguous.allocate(contiguous.size()) ||
      !reference_packed.allocate(packed_bytes) ||
      !reference_scales.allocate(scale_elements) ||
      !split_packed.allocate(packed_bytes) ||
      !split_scales.allocate(scale_elements)) {
    std::cerr << "device allocation failed\n";
    return 1;
  }
  if (!cuda_ok(cudaMemcpy(device_primary.get(), primary.data(),
                          primary.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy primary") ||
      !cuda_ok(cudaMemcpy(device_secondary.get(), secondary.data(),
                          secondary.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy secondary") ||
      !cuda_ok(cudaMemcpy(device_contiguous.get(), contiguous.data(),
                          contiguous.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy contiguous")) {
    return 1;
  }

  const int invalid = kernels::launch_sm87_a4_quantize_bf16_k512_split_cuda(
      device_primary.get(), kPrimaryStride, kPrimaryK - 1U,
      device_secondary.get(), kSecondaryStride, kSecondaryK, kLogicalM,
      kLaunchM, kClipRatio, split_packed.get(), packed_bytes,
      split_scales.get(), scale_elements);
  if (invalid != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "non-K512 split did not fail closed\n";
    return 1;
  }
  const int reference = kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
      device_contiguous.get(), kContiguousStride, kLogicalM, kLaunchM,
      kTotalK, kClipRatio, reference_packed.get(), packed_bytes,
      reference_scales.get(), scale_elements);
  const int split = kernels::launch_sm87_a4_quantize_bf16_k512_split_cuda(
      device_primary.get(), kPrimaryStride, kPrimaryK,
      device_secondary.get(), kSecondaryStride, kSecondaryK, kLogicalM,
      kLaunchM, kClipRatio, split_packed.get(), packed_bytes,
      split_scales.get(), scale_elements);
  if (!cuda_ok(static_cast<cudaError_t>(reference),
               "launch contiguous quantizer") ||
      !cuda_ok(static_cast<cudaError_t>(split), "launch split quantizer") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize quantizers")) {
    return 1;
  }

  std::vector<std::uint8_t> host_reference_packed(packed_bytes);
  std::vector<std::uint8_t> host_split_packed(packed_bytes);
  std::vector<std::uint16_t> host_reference_scales(scale_elements);
  std::vector<std::uint16_t> host_split_scales(scale_elements);
  if (!cuda_ok(cudaMemcpy(host_reference_packed.data(),
                          reference_packed.get(), packed_bytes,
                          cudaMemcpyDeviceToHost),
               "copy reference packed") ||
      !cuda_ok(cudaMemcpy(host_split_packed.data(), split_packed.get(),
                          packed_bytes, cudaMemcpyDeviceToHost),
               "copy split packed") ||
      !cuda_ok(cudaMemcpy(host_reference_scales.data(),
                          reference_scales.get(),
                          scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy reference scales") ||
      !cuda_ok(cudaMemcpy(host_split_scales.data(), split_scales.get(),
                          scale_elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy split scales")) {
    return 1;
  }
  if (host_reference_packed != host_split_packed ||
      host_reference_scales != host_split_scales) {
    std::cerr << "split K512 producer differs from row-wise concatenation\n";
    return 1;
  }
  std::cout << "SM87 split-plane K512 quantizer correctness passed\n";
  return 0;
}

}  // namespace

int main() { return run(); }
