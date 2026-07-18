#include "q3x/core/device_info.h"
#include "q3x/kernels/device_probe.h"

#include <exception>
#include <iostream>
#include <string>

int main() {
  if (q3x::core::ClassifyComputeCapability(8, 7) !=
          q3x::core::DeviceSupport::kOptimizedSm87 ||
      q3x::core::ClassifyComputeCapability(8, 0) !=
          q3x::core::DeviceSupport::kGenericAmpere ||
      q3x::core::ClassifyComputeCapability(8, 6) !=
          q3x::core::DeviceSupport::kGenericAmpere ||
      q3x::core::ClassifyComputeCapability(8, 9) !=
          q3x::core::DeviceSupport::kUnsupported ||
      q3x::core::ClassifyComputeCapability(9, 0) !=
          q3x::core::DeviceSupport::kUnsupported) {
    std::cerr << "invalid CUDA compute-capability classification\n";
    return 1;
  }

  try {
    if (q3x::core::DeviceCount() == 0) {
      std::cout << "SKIP: no CUDA device\n";
      return 77;
    }
    const auto info = q3x::core::QueryDeviceInfo();
    if (info.name.empty() || info.compute_major <= 0 ||
        info.total_memory_bytes == 0) {
      std::cerr << "invalid CUDA device properties\n";
      return 1;
    }
    std::string error;
    if (!q3x::kernels::RunDeviceSmokeTest(&error)) {
      std::cerr << error << '\n';
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
