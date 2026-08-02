#include "q3x/kernels/sm87_a4w4_down_k512_m128n128_ldmatrix_pairring.h"
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

inline constexpr std::size_t kByteGuards = 64U;
inline constexpr std::size_t kWordGuards = 32U;
inline constexpr std::uint8_t kByteSentinel = 0xa5U;
inline constexpr std::uint16_t kWordSentinel = 0xadadU;
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
    return count != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&data_),
                      count * sizeof(T)) == cudaSuccess;
  }
  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  T* data_{};
};

template <typename T>
class GuardedDevice final {
 public:
  [[nodiscard]] bool initialize(const std::vector<T>& payload,
                                const std::size_t guards,
                                const T sentinel,
                                const std::string& label) {
    guards_ = guards;
    payload_count_ = payload.size();
    sentinel_ = sentinel;
    initial_.assign(payload_count_ + 2U * guards_, sentinel_);
    std::copy(payload.begin(), payload.end(),
              initial_.begin() + static_cast<std::ptrdiff_t>(guards_));
    return storage_.allocate(initial_.size()) &&
           cuda_ok(cudaMemcpy(storage_.get(), initial_.data(),
                              initial_.size() * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "copy " + label);
  }

  [[nodiscard]] T* payload() const noexcept {
    return storage_.get() + guards_;
  }
  [[nodiscard]] std::size_t payload_count() const noexcept {
    return payload_count_;
  }
  [[nodiscard]] const std::vector<T>& initial() const noexcept {
    return initial_;
  }

  [[nodiscard]] bool copy(std::vector<T>& output,
                          const std::string& label) const {
    output.resize(initial_.size());
    return cuda_ok(cudaMemcpy(output.data(), storage_.get(),
                              output.size() * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "copy " + label);
  }

  [[nodiscard]] bool unchanged(const std::string& label) const {
    std::vector<T> actual;
    if (!copy(actual, label)) {
      return false;
    }
    if (actual == initial_) {
      return true;
    }
    std::cerr << label << " was modified\n";
    return false;
  }

  [[nodiscard]] bool guards_intact(const std::vector<T>& actual,
                                   const std::string& label) const {
    const auto prefix_end =
        actual.begin() + static_cast<std::ptrdiff_t>(guards_);
    const auto suffix_begin =
        prefix_end + static_cast<std::ptrdiff_t>(payload_count_);
    const bool prefix =
        std::all_of(actual.begin(), prefix_end,
                    [&](const T value) { return value == sentinel_; });
    const bool suffix =
        std::all_of(suffix_begin, actual.end(),
                    [&](const T value) { return value == sentinel_; });
    if (!prefix || !suffix) {
      std::cerr << label << " guard was modified\n";
    }
    return prefix && suffix;
  }

 private:
  DeviceBuffer<T> storage_;
  std::vector<T> initial_;
  std::size_t guards_{};
  std::size_t payload_count_{};
  T sentinel_{};
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

[[nodiscard]] Payload make_payload(const std::size_t logical_m,
                                   const std::size_t launch_m,
                                   const std::size_t n,
                                   const std::size_t k) {
  const std::size_t physical_groups = k / 64U;
  const std::size_t k512_groups = k / 512U;
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
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      result.a_scales[kernels::sm87_a4w4_down_k512_scale_offset(
          row, group, k512_groups)] =
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
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      result.b_scales[kernels::sm87_a4w4_down_k512_scale_offset(
          row, group, k512_groups)] =
          encode_bf16(0.0017F * static_cast<float>(
              7U + (5U * row + 11U * group) % 31U));
    }
  }
  return result;
}

[[nodiscard]] bool compare_words(const std::vector<std::uint16_t>& expected,
                                 const std::vector<std::uint16_t>& actual,
                                 const std::string& label) {
  if (expected == actual) {
    return true;
  }
  const auto mismatch = std::mismatch(expected.begin(), expected.end(),
                                      actual.begin(), actual.end());
  std::cerr << label << " first mismatch at word "
            << std::distance(expected.begin(), mismatch.first)
            << ": expected=0x" << std::hex << *mismatch.first
            << " got=0x" << *mismatch.second << std::dec << '\n';
  return false;
}

[[nodiscard]] bool capture_and_replay(
    GuardedDevice<std::uint8_t>& a,
    GuardedDevice<std::uint16_t>& a_scales,
    GuardedDevice<std::uint8_t>& b,
    GuardedDevice<std::uint16_t>& b_scales,
    const std::size_t launch_m, const std::size_t n,
    const std::size_t k, const std::size_t stride,
    GuardedDevice<std::uint16_t>& output,
    const unsigned int maximum_ctas) {
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  bool ok = cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create pair-ring graph stream") &&
            cuda_ok(cudaStreamBeginCapture(
                        stream, cudaStreamCaptureModeThreadLocal),
                    "begin pair-ring graph capture");
  if (ok) {
    ok = launch_ok(
             kernels::launch_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_test_bf16_cuda(
                 a.payload(), a.payload_count(), a_scales.payload(),
                 a_scales.payload_count(), b.payload(), b.payload_count(),
                 b_scales.payload(), b_scales.payload_count(), launch_m, n,
                 k, output.payload(), stride, output.payload_count(),
                 maximum_ctas, stream),
             "capture pair-ring launch") &&
         cuda_ok(cudaStreamEndCapture(stream, &graph),
                 "end pair-ring graph capture") &&
         cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr,
                                      0U),
                 "instantiate pair-ring graph") &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 "first pair-ring graph replay") &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 "second pair-ring graph replay") &&
         cuda_ok(cudaStreamSynchronize(stream),
                 "synchronize pair-ring graph replay");
  }
  if (executable != nullptr) {
    (void)cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }
  if (stream != nullptr) {
    (void)cudaStreamDestroy(stream);
  }
  return ok;
}

[[nodiscard]] bool run_case(const std::size_t logical_m,
                            const std::size_t launch_m,
                            const std::size_t n,
                            const std::size_t k,
                            const unsigned int maximum_ctas,
                            const bool graph_replay) {
  const std::string shape = "M" + std::to_string(logical_m) + "/P" +
                            std::to_string(launch_m) + " N" +
                            std::to_string(n) + " K" + std::to_string(k);
  const auto plan = kernels::sm87_a4w4_down_k512_test_plan(
      launch_m, n, k);
  if (plan.launch_ctas == 0U) {
    std::cerr << shape << " has no complete-cell plan\n";
    return false;
  }
  const Payload host = make_payload(logical_m, launch_m, n, k);
  const std::size_t stride = n + 8U;
  const std::size_t output_elements = launch_m * stride;

  GuardedDevice<std::uint8_t> a;
  GuardedDevice<std::uint16_t> a_scales;
  GuardedDevice<std::uint8_t> b;
  GuardedDevice<std::uint16_t> b_scales;
  GuardedDevice<std::uint16_t> incumbent;
  GuardedDevice<std::uint16_t> candidate;
  if (!a.initialize(host.a, kByteGuards, kByteSentinel, "A") ||
      !a_scales.initialize(host.a_scales, kWordGuards, kWordSentinel,
                           "A scales") ||
      !b.initialize(host.b, kByteGuards, kByteSentinel, "B") ||
      !b_scales.initialize(host.b_scales, kWordGuards, kWordSentinel,
                           "B scales") ||
      !incumbent.initialize(
          std::vector<std::uint16_t>(output_elements, kOutputSentinel),
          kWordGuards, kOutputSentinel, "incumbent output") ||
      !candidate.initialize(
          std::vector<std::uint16_t>(output_elements, kOutputSentinel),
          kWordGuards, kOutputSentinel, "candidate output")) {
    return false;
  }

  const int incumbent_status =
      kernels::launch_sm87_a4w4_down_k512_macrocell_test_bf16_cuda(
          a.payload(), a.payload_count(), a_scales.payload(),
          a_scales.payload_count(), b.payload(), b.payload_count(),
          b_scales.payload(), b_scales.payload_count(), launch_m, n, k,
          incumbent.payload(), stride, incumbent.payload_count(),
          maximum_ctas);
  const int candidate_status =
      kernels::launch_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_test_bf16_cuda(
          a.payload(), a.payload_count(), a_scales.payload(),
          a_scales.payload_count(), b.payload(), b.payload_count(),
          b_scales.payload(), b_scales.payload_count(), launch_m, n, k,
          candidate.payload(), stride, candidate.payload_count(),
          maximum_ctas);
  if (!launch_ok(incumbent_status, "launch incumbent " + shape) ||
      !launch_ok(candidate_status, "launch candidate " + shape) ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize " + shape)) {
    return false;
  }

  std::vector<std::uint16_t> incumbent_host;
  std::vector<std::uint16_t> candidate_host;
  if (!incumbent.copy(incumbent_host, "incumbent " + shape) ||
      !candidate.copy(candidate_host, "candidate " + shape) ||
      !incumbent.guards_intact(incumbent_host, "incumbent " + shape) ||
      !candidate.guards_intact(candidate_host, "candidate " + shape) ||
      !compare_words(incumbent_host, candidate_host,
                     shape + " bitwise output")) {
    return false;
  }

  // The M129 case models the real ceil128 launch: padded activation rows are
  // zero-coded and must stay numerically zero; output stride padding remains
  // guarded for every row.
  for (std::size_t row = 0U; row < launch_m; ++row) {
    for (std::size_t column = n; column < stride; ++column) {
      const std::size_t index = kWordGuards + row * stride + column;
      if (candidate_host[index] != kOutputSentinel) {
        std::cerr << shape << " output-stride guard overwritten at row "
                  << row << " column " << column << '\n';
        return false;
      }
    }
    if (row >= logical_m) {
      for (std::size_t column = 0U; column < n; ++column) {
        const std::size_t index = kWordGuards + row * stride + column;
        if (candidate_host[index] != encode_bf16(0.0F)) {
          std::cerr << shape << " padded row is nonzero at (" << row
                    << ',' << column << ")\n";
          return false;
        }
      }
    }
  }

  if (graph_replay) {
    if (!capture_and_replay(a, a_scales, b, b_scales, launch_m, n, k,
                            stride, candidate, maximum_ctas) ||
        !candidate.copy(candidate_host, "graph candidate " + shape) ||
        !candidate.guards_intact(candidate_host,
                                 "graph candidate " + shape) ||
        !compare_words(incumbent_host, candidate_host,
                       shape + " graphx2 bitwise output")) {
      return false;
    }
  }

  if (!a.unchanged("A " + shape) ||
      !a_scales.unchanged("A scales " + shape) ||
      !b.unchanged("B " + shape) ||
      !b_scales.unchanged("B scales " + shape)) {
    return false;
  }
  std::cout << "PASS: Down LDSM pair-ring bit-exact " << shape
            << (graph_replay ? " graphx2" : "") << '\n';
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
      properties.sharedMemPerBlockOptin < 128U * 1'024U) {
    std::cout << "SKIP: requires 16-SM SM87 with >=128 KiB opt-in shared\n";
    return 77;
  }
  return 0;
}

}  // namespace

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }

  kernels::Sm87A4W4DownK512M128N128LdmatrixPairringResources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_resources_cuda(
              &resources),
          "query pair-ring resources") ||
      resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 255 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 128U * 1'024U ||
      resources.configured_dynamic_shared_limit_bytes < 128U * 1'024U ||
      resources.device_optin_shared_limit_bytes < 128U * 1'024U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 256 ||
      resources.active_blocks_per_sm != 1) {
    std::cerr << "pair-ring resource gate failed: regs="
              << resources.registers_per_thread
              << " local=" << resources.local_bytes
              << " active=" << resources.active_blocks_per_sm << '\n';
    return 1;
  }

  const bool ok =
      run_case(128U, 128U, 128U, 512U, 1U, false) &&
      run_case(129U, 256U, 256U, 1'024U, 4U, false) &&
      run_case(129U, 256U, 256U, 17'408U, 3U, true);
  if (!ok) {
    return 1;
  }
  std::cout << "Down M128N128 LDSM pair-ring correctness passed: regs="
            << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return 0;
}
