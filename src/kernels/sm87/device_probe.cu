#include "q3x/kernels/device_probe.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace q3x::kernels {
namespace {

constexpr std::uint32_t kProbeInput = 0x51335887U;
constexpr std::uint32_t kProbeMask = 0x0A17C0DEU;

__global__ void DeviceSmokeKernel(std::uint32_t* output) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    output[0] = kProbeInput ^ kProbeMask;
  }
}

void SetError(std::string* destination, const char* operation,
              cudaError_t error) {
  if (destination != nullptr) {
    *destination = std::string(operation) + ": " + cudaGetErrorString(error);
  }
}

}  // namespace

bool RunDeviceSmokeTest(std::string* error_message) {
  std::uint32_t* device_output = nullptr;
  cudaError_t error = cudaMalloc(&device_output, sizeof(std::uint32_t));
  if (error != cudaSuccess) {
    SetError(error_message, "cudaMalloc", error);
    return false;
  }

  DeviceSmokeKernel<<<1, 1>>>(device_output);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    SetError(error_message, "DeviceSmokeKernel launch", error);
    cudaFree(device_output);
    return false;
  }

  std::uint32_t host_output = 0;
  error = cudaMemcpy(&host_output, device_output, sizeof(std::uint32_t),
                     cudaMemcpyDeviceToHost);
  const cudaError_t free_error = cudaFree(device_output);
  if (error != cudaSuccess) {
    SetError(error_message, "cudaMemcpy", error);
    return false;
  }
  if (free_error != cudaSuccess) {
    SetError(error_message, "cudaFree", free_error);
    return false;
  }
  if (host_output != (kProbeInput ^ kProbeMask)) {
    if (error_message != nullptr) {
      *error_message = "device smoke result mismatch";
    }
    return false;
  }
  return true;
}

}  // namespace q3x::kernels
