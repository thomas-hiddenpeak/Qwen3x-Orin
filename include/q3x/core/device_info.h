#pragma once

#include <cstddef>
#include <string>

namespace q3x::core {

enum class DeviceSupport {
  kUnsupported,
  // The generic kernels are source-compatible with these Ampere devices, but
  // the installed binary must still contain a matching cubin or PTX image.
  kGenericAmpere,
  kOptimizedSm87,
};

struct DeviceInfo {
  int device_id = 0;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  int multiprocessor_count = 0;
  std::size_t total_memory_bytes = 0;
  std::size_t l2_cache_bytes = 0;
  std::size_t shared_memory_per_sm_bytes = 0;
  int memory_clock_khz = 0;
  int memory_bus_width_bits = 0;
  bool unified_addressing = false;
  bool managed_memory = false;
  int cuda_driver_version = 0;
  int cuda_runtime_version = 0;
  DeviceSupport support = DeviceSupport::kUnsupported;

  [[nodiscard]] double theoretical_memory_bandwidth_gbps() const;
};

[[nodiscard]] int DeviceCount();
[[nodiscard]] DeviceSupport ClassifyComputeCapability(int major,
                                                      int minor) noexcept;
[[nodiscard]] DeviceInfo QueryDeviceInfo(int device_id = 0);
[[nodiscard]] const char* DeviceSupportName(DeviceSupport support);
[[nodiscard]] std::string FormatDeviceInfo(const DeviceInfo& info);

}  // namespace q3x::core
