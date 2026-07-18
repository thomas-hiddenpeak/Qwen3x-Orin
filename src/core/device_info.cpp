#include "q3x/core/device_info.h"

#include <cuda_runtime_api.h>

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace q3x::core {
namespace {

void CheckCuda(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(operation) + ": " +
                           cudaGetErrorString(error));
}

int QueryOptionalDeviceAttribute(const int device_id,
                                 const cudaDeviceAttr attribute) {
  int value = 0;
  const cudaError_t status =
      cudaDeviceGetAttribute(&value, attribute, device_id);
  if (status == cudaSuccess) {
    return value;
  }

  // Some Jetson driver/runtime combinations do not expose every desktop CUDA
  // device attribute. Treat those attributes as unavailable and keep probing.
  static_cast<void>(cudaGetLastError());
  return 0;
}

std::string FormatCudaVersion(int version) {
  if (version <= 0) {
    return "unknown";
  }
  std::ostringstream out;
  out << version / 1000 << '.' << (version % 1000) / 10;
  return out.str();
}

}  // namespace

double DeviceInfo::theoretical_memory_bandwidth_gbps() const {
  if (memory_clock_khz <= 0 || memory_bus_width_bits <= 0) {
    return 0.0;
  }
  // CUDA reports the data clock in kHz. DDR transfers twice per clock.
  return 2.0 * static_cast<double>(memory_clock_khz) *
         static_cast<double>(memory_bus_width_bits) / 8.0 / 1'000'000.0;
}

int DeviceCount() {
  int count = 0;
  CheckCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  return count;
}

DeviceSupport ClassifyComputeCapability(const int major,
                                        const int minor) noexcept {
  if (major != 8) {
    return DeviceSupport::kUnsupported;
  }
  switch (minor) {
    case 7:
      return DeviceSupport::kOptimizedSm87;
    case 0:
    case 6:
      return DeviceSupport::kGenericAmpere;
    default:
      // In particular, SM89 is Ada rather than Ampere.
      return DeviceSupport::kUnsupported;
  }
}

DeviceInfo QueryDeviceInfo(int device_id) {
  cudaDeviceProp properties{};
  CheckCuda(cudaGetDeviceProperties(&properties, device_id),
            "cudaGetDeviceProperties");

  DeviceInfo info;
  info.device_id = device_id;
  info.name = properties.name;
  info.compute_major = properties.major;
  info.compute_minor = properties.minor;
  info.multiprocessor_count = properties.multiProcessorCount;
  info.total_memory_bytes = properties.totalGlobalMem;
  info.l2_cache_bytes = static_cast<std::size_t>(properties.l2CacheSize);
  info.shared_memory_per_sm_bytes = properties.sharedMemPerMultiprocessor;
  info.memory_clock_khz = QueryOptionalDeviceAttribute(
      device_id, cudaDevAttrMemoryClockRate);
  info.memory_bus_width_bits = QueryOptionalDeviceAttribute(
      device_id, cudaDevAttrGlobalMemoryBusWidth);
  info.unified_addressing = properties.unifiedAddressing != 0;
  info.managed_memory = properties.managedMemory != 0;
  CheckCuda(cudaDriverGetVersion(&info.cuda_driver_version),
            "cudaDriverGetVersion");
  CheckCuda(cudaRuntimeGetVersion(&info.cuda_runtime_version),
            "cudaRuntimeGetVersion");
  info.support =
      ClassifyComputeCapability(properties.major, properties.minor);
  return info;
}

const char* DeviceSupportName(DeviceSupport support) {
  switch (support) {
    case DeviceSupport::kOptimizedSm87:
      return "optimized-sm87";
    case DeviceSupport::kGenericAmpere:
      return "generic-ampere";
    case DeviceSupport::kUnsupported:
      return "unsupported";
  }
  return "unknown";
}

std::string FormatDeviceInfo(const DeviceInfo& info) {
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  constexpr double kMiB = 1024.0 * 1024.0;

  std::ostringstream out;
  out << "Device " << info.device_id << ": " << info.name << '\n'
      << "  Compute capability: " << info.compute_major << '.'
      << info.compute_minor << " (" << DeviceSupportName(info.support) << ")\n"
      << "  SM count: " << info.multiprocessor_count << '\n'
      << "  Unified memory: " << std::fixed << std::setprecision(2)
      << static_cast<double>(info.total_memory_bytes) / kGiB << " GiB\n"
      << "  L2 cache: " << static_cast<double>(info.l2_cache_bytes) / kMiB
      << " MiB\n"
      << "  Shared memory / SM: "
      << static_cast<double>(info.shared_memory_per_sm_bytes) / 1024.0
      << " KiB\n"
      << "  Unified addressing: " << (info.unified_addressing ? "yes" : "no")
      << '\n'
      << "  Managed memory: " << (info.managed_memory ? "yes" : "no") << '\n'
      << "  CUDA driver/runtime: " << FormatCudaVersion(info.cuda_driver_version)
      << '/' << FormatCudaVersion(info.cuda_runtime_version) << '\n';

  const double bandwidth = info.theoretical_memory_bandwidth_gbps();
  if (bandwidth > 0.0) {
    out << "  Reported DDR bandwidth: " << std::setprecision(1) << bandwidth
        << " GB/s\n";
  }
  return out.str();
}

}  // namespace q3x::core
