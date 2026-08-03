#include "q3x/kernels/sm87_a4w4_down_k512_m128n128_ldmatrix_pairring.h"
#include "q3x/kernels/sm87_a4w4_down_k512_m64n256_16warp_pairring.h"
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

inline constexpr std::size_t kLogicalM = 117U;
inline constexpr std::size_t kLaunchM = 128U;
inline constexpr std::size_t kN = 5'120U;
inline constexpr std::size_t kK = 17'408U;
inline constexpr std::size_t kStride = kN + 8U;
inline constexpr std::size_t kOutputGuards = 32U;
inline constexpr std::uint16_t kOutputSentinel = 0x7fc1U;

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const std::string& operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorName(status) << " ("
            << cudaGetErrorString(status) << ")\n";
  return false;
}

[[nodiscard]] bool launch_ok(const int status,
                             const std::string& operation) {
  return cuda_ok(static_cast<cudaError_t>(status), operation);
}

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

  [[nodiscard]] bool allocate(const std::size_t count) noexcept {
    count_ = count;
    return count != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&data_),
                      count * sizeof(T)) == cudaSuccess;
  }
  [[nodiscard]] T* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }

 private:
  T* data_{};
  std::size_t count_{};
};

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

[[nodiscard]] Payload make_production_payload() {
  const std::size_t physical_groups = kK / 64U;
  const std::size_t k512_groups = kK / 512U;
  Payload result{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_down_k512_packed_capacity_bytes(kLaunchM,
                                                              kK)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_down_k512_scale_capacity_elements(kLaunchM,
                                                                kK),
          encode_bf16(1.0F)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_down_k512_packed_capacity_bytes(kN, kK)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_down_k512_scale_capacity_elements(kN, kK))};

  // Only logical rows receive A codes.  The remaining eleven rows reproduce
  // the authenticated quantizer's zero-code/BF16-one padding contract.
  for (std::size_t row = 0U; row < kLogicalM; ++row) {
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
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      result.a_scales[kernels::sm87_a4w4_down_k512_scale_offset(
          row, group, k512_groups)] =
          encode_bf16(0.0021F * static_cast<float>(
              5U + (3U * row + 7U * group) % 29U));
    }
  }

  for (std::size_t row = 0U; row < kN; ++row) {
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
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      result.b_scales[kernels::sm87_a4w4_down_k512_scale_offset(
          row, group, k512_groups)] =
          encode_bf16(0.0017F * static_cast<float>(
              7U + (5U * row + 11U * group) % 31U));
    }
  }
  return result;
}

template <typename T>
[[nodiscard]] bool copy_to_device(DeviceBuffer<T>& destination,
                                  const std::vector<T>& source,
                                  const std::string& label) {
  return destination.allocate(source.size()) &&
         cuda_ok(cudaMemcpy(destination.get(), source.data(),
                            source.size() * sizeof(T),
                            cudaMemcpyHostToDevice),
                 "copy " + label);
}

template <typename T>
[[nodiscard]] bool unchanged(const DeviceBuffer<T>& device,
                             const std::vector<T>& expected,
                             const std::string& label) {
  std::vector<T> actual(expected.size());
  if (!cuda_ok(cudaMemcpy(actual.data(), device.get(),
                          actual.size() * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "copy back " + label)) {
    return false;
  }
  if (actual == expected) {
    return true;
  }
  const auto mismatch =
      std::mismatch(expected.begin(), expected.end(), actual.begin());
  std::cerr << label << " modified at element "
            << std::distance(expected.begin(), mismatch.first) << '\n';
  return false;
}

[[nodiscard]] bool mapping_contract() {
  const auto p1920 =
      kernels::sm87_a4w4_down_k512_m64n256_16warp_pairring_plan(
          1'853U, 1'920U, kN, kK);
  const auto p2176 =
      kernels::sm87_a4w4_down_k512_m64n256_16warp_pairring_plan(
          2'049U, 2'176U, kN, kK);
  if (p1920.work_cells != 600U || p1920.launch_ctas != 16U ||
      p1920.minimum_cells_per_cta != 37U ||
      p1920.maximum_cells_per_cta != 38U ||
      p2176.work_cells != 680U || p2176.launch_ctas != 16U ||
      p2176.minimum_cells_per_cta != 42U ||
      p2176.maximum_cells_per_cta != 43U) {
    std::cerr << "flat production plan mismatch\n";
    return false;
  }

  for (const auto& plan : {p1920, p2176}) {
    std::vector<unsigned int> visits(plan.work_cells, 0U);
    for (std::size_t cta = 0U; cta < plan.launch_ctas; ++cta) {
      for (std::size_t iteration = 0U;; ++iteration) {
        const auto cell =
            kernels::sm87_a4w4_down_k512_m64n256_16warp_pairring_cell(
                plan, cta, iteration);
        if (!cell.valid) {
          break;
        }
        ++visits[cell.n_tile * plan.m_tiles + cell.m_tile];
      }
    }
    if (!std::all_of(visits.begin(), visits.end(),
                     [](const unsigned int count) {
                       return count == 1U;
                     })) {
      std::cerr << "flat production plan does not cover each cell once\n";
      return false;
    }
  }
  return true;
}

[[nodiscard]] int target_status() {
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
  int device = -1;
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice") ||
      !cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16 ||
      properties.sharedMemPerBlockOptin <
          kernels::
              kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes) {
    std::cout << "SKIP: requires 16-SM SM87 with >=165120 B opt-in shared\n";
    return 77;
  }
  return 0;
}

[[nodiscard]] bool resource_gate() {
  kernels::Sm87A4W4DownK512M64N256Pairring16Resources resources{};
  return kernels::
                 query_sm87_a4w4_down_k512_m64n256_16warp_pairring_resources_cuda(
                     nullptr) == static_cast<int>(cudaErrorInvalidValue) &&
         launch_ok(
             kernels::
                 query_sm87_a4w4_down_k512_m64n256_16warp_pairring_resources_cuda(
                     &resources),
             "query M64N256 resources") &&
         resources.registers_per_thread == 128 &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes == 165'120U &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >= 512 &&
         resources.active_blocks_per_sm == 1;
}

[[nodiscard]] int launch_candidate(
    const std::uint8_t* const a,
    const std::size_t a_capacity,
    const std::uint16_t* const a_scales,
    const std::size_t a_scale_capacity,
    const std::uint8_t* const b,
    const std::size_t b_capacity,
    const std::uint16_t* const b_scales,
    const std::size_t b_scale_capacity,
    std::uint16_t* const output,
    const std::size_t output_capacity,
    cudaStream_t stream) {
  return kernels::
      launch_sm87_a4w4_down_k512_m64n256_16warp_pairring_bf16_cuda(
          a, a_capacity, a_scales, a_scale_capacity, b, b_capacity,
          b_scales, b_scale_capacity, kLogicalM, kLaunchM, kN, kK,
          output, kStride, output_capacity, stream);
}

[[nodiscard]] bool admission_rejections(
    DeviceBuffer<std::uint8_t>& a,
    DeviceBuffer<std::uint16_t>& a_scales,
    DeviceBuffer<std::uint8_t>& b,
    DeviceBuffer<std::uint16_t>& b_scales,
    std::uint16_t* const output,
    const std::size_t output_count) {
  const auto rejected = [](const int status) {
    return status == static_cast<int>(cudaErrorInvalidValue);
  };
  const bool capacities =
      rejected(launch_candidate(
          a.get(), a.count() - 1U, a_scales.get(), a_scales.count(),
          b.get(), b.count(), b_scales.get(), b_scales.count(), output,
          output_count, nullptr)) &&
      rejected(launch_candidate(
          a.get(), a.count(), a_scales.get(), a_scales.count() - 1U,
          b.get(), b.count(), b_scales.get(), b_scales.count(), output,
          output_count, nullptr)) &&
      rejected(launch_candidate(
          a.get(), a.count(), a_scales.get(), a_scales.count(), b.get(),
          b.count() - 1U, b_scales.get(), b_scales.count(), output,
          output_count, nullptr)) &&
      rejected(launch_candidate(
          a.get(), a.count(), a_scales.get(), a_scales.count(), b.get(),
          b.count(), b_scales.get(), b_scales.count() - 1U, output,
          output_count, nullptr)) &&
      rejected(launch_candidate(
          a.get(), a.count(), a_scales.get(), a_scales.count(), b.get(),
          b.count(), b_scales.get(), b_scales.count(), output,
          output_count - 1U, nullptr));
  const bool aliases =
      rejected(launch_candidate(
          a.get(), a.count(), a_scales.get(), a_scales.count(), b.get(),
          b.count(), b_scales.get(), b_scales.count(),
          reinterpret_cast<std::uint16_t*>(a.get()), output_count,
          nullptr)) &&
      rejected(launch_candidate(
          a.get(), a.count(), a_scales.get(), a_scales.count(), b.get(),
          b.count(), b_scales.get(), b_scales.count(),
          a_scales.get(), output_count, nullptr)) &&
      rejected(launch_candidate(
          a.get(), a.count(), a_scales.get(), a_scales.count(), b.get(),
          b.count(), b_scales.get(), b_scales.count(),
          reinterpret_cast<std::uint16_t*>(b.get()), output_count,
          nullptr)) &&
      rejected(launch_candidate(
          a.get(), a.count(), a_scales.get(), a_scales.count(), b.get(),
          b.count(), b_scales.get(), b_scales.count(), b_scales.get(),
          output_count, nullptr));
  if (!capacities || !aliases) {
    std::cerr << "capacity or alias admission did not fail closed\n";
  }
  return capacities && aliases;
}

[[nodiscard]] bool production_bit_exact_case() {
  const Payload host = make_production_payload();
  DeviceBuffer<std::uint8_t> a;
  DeviceBuffer<std::uint16_t> a_scales;
  DeviceBuffer<std::uint8_t> b;
  DeviceBuffer<std::uint16_t> b_scales;
  if (!copy_to_device(a, host.a, "A") ||
      !copy_to_device(a_scales, host.a_scales, "A scales") ||
      !copy_to_device(b, host.b, "B") ||
      !copy_to_device(b_scales, host.b_scales, "B scales")) {
    return false;
  }

  const std::size_t output_count = kLaunchM * kStride;
  const std::size_t guarded_count = output_count + 2U * kOutputGuards;
  const std::vector<std::uint16_t> initial(guarded_count, kOutputSentinel);
  DeviceBuffer<std::uint16_t> oracle;
  DeviceBuffer<std::uint16_t> candidate;
  if (!copy_to_device(oracle, initial, "oracle output") ||
      !copy_to_device(candidate, initial, "candidate output")) {
    return false;
  }
  std::uint16_t* const oracle_output = oracle.get() + kOutputGuards;
  std::uint16_t* const candidate_output = candidate.get() + kOutputGuards;

  if (!admission_rejections(a, a_scales, b, b_scales,
                            candidate_output, output_count)) {
    return false;
  }

  cudaStream_t stream{};
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create non-default stream")) {
    return false;
  }
  bool ok = launch_ok(
                kernels::
                    launch_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_test_bf16_cuda(
                        a.get(), a.count(), a_scales.get(),
                        a_scales.count(), b.get(), b.count(),
                        b_scales.get(), b_scales.count(), kLaunchM, kN,
                        kK, oracle_output, kStride, output_count, 16U,
                        stream),
                "launch bounded authenticated oracle") &&
            launch_ok(
                launch_candidate(
                    a.get(), a.count(), a_scales.get(),
                    a_scales.count(), b.get(), b.count(), b_scales.get(),
                    b_scales.count(), candidate_output, output_count,
                    stream),
                "launch production candidate on non-default stream") &&
            cuda_ok(cudaStreamSynchronize(stream),
                    "synchronize non-default stream");
  (void)cudaStreamDestroy(stream);
  if (!ok) {
    return false;
  }

  std::vector<std::uint16_t> oracle_host(guarded_count);
  std::vector<std::uint16_t> candidate_host(guarded_count);
  ok = cuda_ok(cudaMemcpy(oracle_host.data(), oracle.get(),
                          guarded_count * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy oracle output") &&
       cuda_ok(cudaMemcpy(candidate_host.data(), candidate.get(),
                          guarded_count * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy candidate output");
  if (!ok) {
    return false;
  }
  if (oracle_host != candidate_host) {
    const auto mismatch =
        std::mismatch(oracle_host.begin(), oracle_host.end(),
                      candidate_host.begin());
    std::cerr << "production bitwise mismatch at guarded word "
              << std::distance(oracle_host.begin(), mismatch.first)
              << ": oracle=0x" << std::hex << *mismatch.first
              << " candidate=0x" << *mismatch.second << std::dec << '\n';
    return false;
  }

  const auto guards_are = [&](const std::vector<std::uint16_t>& values) {
    return std::all_of(values.begin(),
                       values.begin() +
                           static_cast<std::ptrdiff_t>(kOutputGuards),
                       [](const std::uint16_t value) {
                         return value == kOutputSentinel;
                       }) &&
           std::all_of(values.end() -
                           static_cast<std::ptrdiff_t>(kOutputGuards),
                       values.end(), [](const std::uint16_t value) {
                         return value == kOutputSentinel;
                       });
  };
  if (!guards_are(candidate_host)) {
    std::cerr << "candidate output guard modified\n";
    return false;
  }
  for (std::size_t row = 0U; row < kLaunchM; ++row) {
    for (std::size_t column = kN; column < kStride; ++column) {
      if (candidate_host[kOutputGuards + row * kStride + column] !=
          kOutputSentinel) {
        std::cerr << "candidate stride guard modified\n";
        return false;
      }
    }
    if (row >= kLogicalM) {
      for (std::size_t column = 0U; column < kN; ++column) {
        if (candidate_host[kOutputGuards + row * kStride + column] !=
            encode_bf16(0.0F)) {
          std::cerr << "padded output row is not bitwise zero\n";
          return false;
        }
      }
    }
  }

  return unchanged(a, host.a, "A") &&
         unchanged(a_scales, host.a_scales, "A scales") &&
         unchanged(b, host.b, "B") &&
         unchanged(b_scales, host.b_scales, "B scales");
}

}  // namespace

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }
  if (!mapping_contract() || !resource_gate() ||
      !production_bit_exact_case()) {
    return 1;
  }
  std::cout << "Down M64N256 16-warp pairring production-shape "
               "correctness passed\n";
  return 0;
}
