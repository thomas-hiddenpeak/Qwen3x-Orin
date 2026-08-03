#include "q3x/kernels/sm87_a4w4_down_k512_m128n128_16warp_pairring.h"
#include "q3x/kernels/sm87_a4w4_down_k512_m256n128_k512resident.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

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

  [[nodiscard]] bool allocate(const std::size_t count) noexcept {
    count_ = count;
    return count != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&pointer_),
                      count * sizeof(T)) == cudaSuccess;
  }
  [[nodiscard]] T* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }

 private:
  T* pointer_{};
  std::size_t count_{};
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

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] std::int8_t code(const std::size_t outer,
                               const std::size_t inner,
                               const std::uint32_t salt) noexcept {
  std::uint32_t mixed =
      static_cast<std::uint32_t>(outer * (0x9e3779b9U ^ salt)) ^
      static_cast<std::uint32_t>(inner * (0x85ebca6bU + salt));
  mixed ^= mixed >> 16U;
  mixed *= 0x7feb352dU;
  mixed ^= mixed >> 15U;
  return static_cast<std::int8_t>(static_cast<int>(mixed & 15U) - 8);
}

struct Payload final {
  std::vector<std::uint8_t> a;
  std::vector<std::uint16_t> a_scales;
  std::vector<std::uint8_t> b;
  std::vector<std::uint16_t> b_scales;
};

[[nodiscard]] Payload make_payload(const std::size_t logical_m,
                                   const std::size_t launch_m,
                                   const std::size_t n,
                                   const std::size_t k) {
  const std::size_t physical_groups = k / 64U;
  const std::size_t scale_groups = k / 512U;
  Payload result{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_down_k512_packed_capacity_bytes(launch_m, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_down_k512_scale_capacity_elements(launch_m, k),
          encode_bf16(1.0F)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_down_k512_packed_capacity_bytes(n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_down_k512_scale_capacity_elements(n, k))};

  for (std::size_t row = 0U; row < logical_m; ++row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = group * 64U + 2U * byte;
        result.a[kernels::sm87_a4w4_down_k512_packed_offset(
            row, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                code(row, inner, 0x1234U),
                code(row, inner + 1U, 0x1234U));
      }
    }
    for (std::size_t group = 0U; group < scale_groups; ++group) {
      result.a_scales[kernels::sm87_a4w4_down_k512_scale_offset(
          row, group, scale_groups)] =
          encode_bf16(0.0021F * static_cast<float>(
              5U + (3U * row + 7U * group) % 29U));
    }
  }
  for (std::size_t row = 0U; row < n; ++row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = group * 64U + 2U * byte;
        result.b[kernels::sm87_a4w4_down_k512_packed_offset(
            row, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                code(row, inner, 0x89abU),
                code(row, inner + 1U, 0x89abU));
      }
    }
    for (std::size_t group = 0U; group < scale_groups; ++group) {
      result.b_scales[kernels::sm87_a4w4_down_k512_scale_offset(
          row, group, scale_groups)] =
          encode_bf16(0.0017F * static_cast<float>(
              7U + (5U * row + 11U * group) % 31U));
    }
  }
  return result;
}

template <typename T>
[[nodiscard]] bool upload(DeviceBuffer<T>& device,
                          const std::vector<T>& host,
                          const std::string& label) {
  return device.allocate(host.size()) &&
         cuda_ok(cudaMemcpy(device.get(), host.data(),
                            host.size() * sizeof(T),
                            cudaMemcpyHostToDevice),
                 "upload " + label);
}

template <typename T>
[[nodiscard]] bool unchanged(const DeviceBuffer<T>& device,
                             const std::vector<T>& expected,
                             const std::string& label) {
  std::vector<T> actual(expected.size());
  if (!cuda_ok(cudaMemcpy(actual.data(), device.get(),
                          actual.size() * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "download " + label)) {
    return false;
  }
  if (actual == expected) {
    return true;
  }
  const auto mismatch =
      std::mismatch(expected.begin(), expected.end(), actual.begin());
  std::cerr << label << " modified at "
            << std::distance(expected.begin(), mismatch.first) << '\n';
  return false;
}

[[nodiscard]] bool run_case(const std::size_t logical_m,
                            const std::size_t launch_m,
                            const std::size_t n,
                            const std::size_t k) {
  constexpr std::uint16_t kSentinel = 0x7fc1U;
  const Payload payload = make_payload(logical_m, launch_m, n, k);
  DeviceBuffer<std::uint8_t> a;
  DeviceBuffer<std::uint16_t> a_scales;
  DeviceBuffer<std::uint8_t> b;
  DeviceBuffer<std::uint16_t> b_scales;
  DeviceBuffer<std::uint16_t> candidate;
  DeviceBuffer<std::uint16_t> reference;
  const std::size_t output_elements = launch_m * n;
  const std::vector<std::uint16_t> initialized(output_elements, kSentinel);
  if (!upload(a, payload.a, "A") ||
      !upload(a_scales, payload.a_scales, "A scales") ||
      !upload(b, payload.b, "B") ||
      !upload(b_scales, payload.b_scales, "B scales") ||
      !upload(candidate, initialized, "candidate output") ||
      !upload(reference, initialized, "reference output")) {
    return false;
  }

  const int candidate_status =
      kernels::launch_sm87_a4w4_down_k512_m256n128_k512resident_test_bf16_cuda(
          a.get(), a.count(), a_scales.get(), a_scales.count(), b.get(),
          b.count(), b_scales.get(), b_scales.count(), logical_m, launch_m,
          n, k, candidate.get(), n, candidate.count(), 16U);
  const int reference_status =
      kernels::launch_sm87_a4w4_down_k512_m128n128_16warp_pairring_test_bf16_cuda(
          a.get(), a.count(), a_scales.get(), a_scales.count(), b.get(),
          b.count(), b_scales.get(), b_scales.count(), launch_m, n, k,
          reference.get(), n, reference.count(), 16U);
  if (!cuda_ok(static_cast<cudaError_t>(candidate_status),
               "launch candidate") ||
      !cuda_ok(static_cast<cudaError_t>(reference_status),
               "launch reference") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize kernels")) {
    return false;
  }

  std::vector<std::uint16_t> actual(output_elements);
  std::vector<std::uint16_t> expected(output_elements);
  if (!cuda_ok(cudaMemcpy(actual.data(), candidate.get(),
                          output_elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "download candidate") ||
      !cuda_ok(cudaMemcpy(expected.data(), reference.get(),
                          output_elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "download reference")) {
    return false;
  }
  for (std::size_t row = 0U; row < logical_m; ++row) {
    for (std::size_t column = 0U; column < n; ++column) {
      const std::size_t index = row * n + column;
      if (actual[index] != expected[index]) {
        std::cerr << "bit mismatch M=" << logical_m << " N=" << n
                  << " K=" << k << " at (" << row << ',' << column
                  << "): candidate=0x" << std::hex << actual[index]
                  << " reference=0x" << expected[index] << std::dec << '\n';
        return false;
      }
    }
  }
  const bool masked = std::all_of(
      actual.begin() + static_cast<std::ptrdiff_t>(logical_m * n),
      actual.end(),
      [](const std::uint16_t value) { return value == kSentinel; });
  if (!masked) {
    std::cerr << "candidate wrote a logical-M tail row\n";
    return false;
  }
  return unchanged(a, payload.a, "A") &&
         unchanged(a_scales, payload.a_scales, "A scales") &&
         unchanged(b, payload.b, "B") &&
         unchanged(b_scales, payload.b_scales, "B scales");
}

}  // namespace

int main() {
  int device = -1;
  if (!cuda_ok(cudaGetDevice(&device), "get CUDA device")) {
    return 1;
  }
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDeviceProperties(&properties, device),
               "get device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: requires the 16-SM SM87 target\n";
    return 77;
  }

  const auto tail_plan =
      kernels::sm87_a4w4_down_k512_m256n128_k512resident_test_plan(
          257U, 384U, 128U, 512U);
  const auto model_plan =
      kernels::sm87_a4w4_down_k512_m256n128_k512resident_plan(
          1'853U, 1'920U, 5'120U, 17'408U);
  const auto invalid_plan =
      kernels::sm87_a4w4_down_k512_m256n128_k512resident_test_plan(
          257U, 256U, 128U, 512U);
  if (tail_plan.m_tiles != 2U || tail_plan.work_tiles != 2U ||
      model_plan.m_tiles != 8U || model_plan.n_tiles != 40U ||
      model_plan.work_tiles != 320U || model_plan.launch_ctas != 16U ||
      invalid_plan.launch_ctas != 0U) {
    std::cerr << "host plan contract failed\n";
    return 1;
  }

  kernels::Sm87A4W4DownK512M256N128ResidentResources resources{};
  const auto resource_status = static_cast<cudaError_t>(
      kernels::query_sm87_a4w4_down_k512_m256n128_k512resident_resources_cuda(
          &resources));
  std::cout << "resources regs=" << resources.registers_per_thread
            << " static_shared=" << resources.static_shared_bytes
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " max_threads=" << resources.maximum_threads_per_block
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  if (resource_status == cudaErrorLaunchOutOfResources &&
      resources.registers_per_thread == 128 &&
      resources.static_shared_bytes == 0U &&
      resources.dynamic_shared_bytes == 99'072U &&
      resources.configured_dynamic_shared_limit_bytes >= 99'072U &&
      resources.device_optin_shared_limit_bytes >= 99'072U &&
      resources.local_bytes != 0U &&
      resources.maximum_threads_per_block >= 512 &&
      resources.active_blocks_per_sm == 1 &&
      resources.compute_major == 8 && resources.compute_minor == 7) {
    std::cout << "PASS: expected zero-local resource rejection before "
                 "correctness\n";
    return 0;
  }
  if (!cuda_ok(resource_status, "query resources")) {
    return 1;
  }

  if (!run_case(257U, 384U, 128U, 512U) ||
      !run_case(384U, 384U, 256U, 1'024U)) {
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
