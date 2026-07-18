#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/layout_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

__global__ void no_op() {}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    if (cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)) !=
        cudaSuccess) {
      data_ = nullptr;
    }
  }
  ~DeviceBuffer() { (void)cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return data_ != nullptr;
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA device unavailable\n";
    return 77;
  }

  constexpr std::size_t kHeads = 24U;
  constexpr std::size_t kDimension =
      q3x::runtime::kFullAttentionHeadDimension;
  constexpr std::size_t kOutputElements = kHeads * kDimension;
  std::vector<std::uint16_t> input(kOutputElements * 2U);
  std::vector<std::uint16_t> query(kOutputElements, 0U);
  std::vector<std::uint16_t> gate(kOutputElements, 0U);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<std::uint16_t>((index * 37U + 11U) & 0xffffU);
  }

  DeviceBuffer<std::uint16_t> device_input(input.size());
  DeviceBuffer<std::uint16_t> device_query(query.size());
  DeviceBuffer<std::uint16_t> device_gate(gate.size());
  if (!device_input || !device_query || !device_gate) {
    std::cerr << "FAIL: CUDA allocation\n";
    return 1;
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaMemcpyAsync(device_input.data(), input.data(),
                      input.size() * sizeof(std::uint16_t),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    std::cerr << "FAIL: CUDA setup\n";
    (void)cudaStreamDestroy(stream);
    return 1;
  }

  // The public launch must discard an unrelated prior launch error.
  no_op<<<0U, 1U, 0U, stream>>>();
  const int launch_status =
      q3x::runtime::launch_split_interleaved_q_gate_reference_cuda(
          device_input.data(), kHeads, kDimension, device_query.data(),
          device_gate.data(), static_cast<void*>(stream));
  int failures = 0;
  if (launch_status != static_cast<int>(cudaSuccess)) {
    ++failures;
    std::cerr << "FAIL: split launch: "
              << cudaGetErrorString(static_cast<cudaError_t>(launch_status))
              << '\n';
  }
  if (cudaMemcpyAsync(query.data(), device_query.data(),
                      query.size() * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(gate.data(), device_gate.data(),
                      gate.size() * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    ++failures;
    std::cerr << "FAIL: split synchronization/copy\n";
  }

  for (std::size_t head = 0; head < kHeads; ++head) {
    for (std::size_t dimension = 0; dimension < kDimension; ++dimension) {
      const std::size_t output_index = head * kDimension + dimension;
      const std::size_t input_index = head * 2U * kDimension + dimension;
      if (query[output_index] != input[input_index] ||
          gate[output_index] != input[input_index + kDimension]) {
        ++failures;
        std::cerr << "FAIL: exact Q/Gate layout at " << output_index << '\n';
        head = kHeads;
        break;
      }
    }
  }

  const int overlap_status =
      q3x::runtime::launch_split_interleaved_q_gate_reference_cuda(
          device_input.data(), kHeads, kDimension, device_input.data(),
          device_gate.data(), static_cast<void*>(stream));
  if (overlap_status != static_cast<int>(cudaErrorInvalidValue)) {
    ++failures;
    std::cerr << "FAIL: overlap is rejected before launch\n";
  }
  if (q3x::runtime::launch_split_interleaved_q_gate_reference_cuda(
          nullptr, 0U, kDimension, nullptr, nullptr,
          static_cast<void*>(stream)) != static_cast<int>(cudaSuccess)) {
    ++failures;
    std::cerr << "FAIL: empty launch is a no-op\n";
  }

  (void)cudaStreamDestroy(stream);
  if (failures != 0) {
    return 1;
  }
  std::cout << "layout ops CUDA tests passed\n";
  return 0;
}
