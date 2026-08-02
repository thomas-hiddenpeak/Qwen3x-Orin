#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix.h"
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

inline constexpr float kClipRatio = 0.9375F;
inline constexpr std::size_t kByteGuard = 64U;
inline constexpr std::size_t kWordGuard = 32U;
inline constexpr std::uint8_t kByteSentinel = 0xa5U;
inline constexpr std::uint16_t kWordSentinel = 0xadadU;
inline constexpr std::uint16_t kBf16Sentinel = 0x7fc1U;

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
      properties.sharedMemPerBlockOptin < 148'736U) {
    std::cout << "SKIP: requires 16-SM SM87 with >=148736 B opt-in shared\n";
    return 77;
  }
  return 0;
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
  return static_cast<std::int8_t>(static_cast<int>(mixed % 15U) - 7);
}

template <typename T>
class GuardedDevice final {
 public:
  [[nodiscard]] bool initialize(const std::vector<T>& payload,
                                const std::size_t guard,
                                const T sentinel,
                                const std::string& label) {
    guard_ = guard;
    payload_count_ = payload.size();
    sentinel_ = sentinel;
    initial_.assign(payload_count_ + 2U * guard_, sentinel_);
    std::copy(payload.begin(), payload.end(),
              initial_.begin() + static_cast<std::ptrdiff_t>(guard_));
    return device_.allocate(initial_.size()) &&
           cuda_ok(cudaMemcpy(device_.get(), initial_.data(),
                              initial_.size() * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "copy " + label);
  }

  [[nodiscard]] T* payload() const noexcept {
    return device_.get() + guard_;
  }
  [[nodiscard]] std::size_t payload_count() const noexcept {
    return payload_count_;
  }

  [[nodiscard]] bool copy(std::vector<T>& output,
                          const std::string& label) const {
    output.resize(initial_.size());
    return cuda_ok(cudaMemcpy(output.data(), device_.get(),
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
        actual.begin() + static_cast<std::ptrdiff_t>(guard_);
    const auto suffix_begin =
        prefix_end + static_cast<std::ptrdiff_t>(payload_count_);
    const bool prefix = std::all_of(
        actual.begin(), prefix_end,
        [&](const T value) { return value == sentinel_; });
    const bool suffix = std::all_of(
        suffix_begin, actual.end(),
        [&](const T value) { return value == sentinel_; });
    if (!prefix || !suffix) {
      std::cerr << label << " guard was modified\n";
    }
    return prefix && suffix;
  }

 private:
  DeviceBuffer<T> device_;
  std::vector<T> initial_;
  std::size_t guard_{};
  std::size_t payload_count_{};
  T sentinel_{};
};

struct GateUpPayload final {
  std::vector<std::uint8_t> a;
  std::vector<std::uint16_t> a_scales;
  std::vector<std::uint8_t> gate;
  std::vector<std::uint16_t> gate_scales;
  std::vector<std::uint8_t> up;
  std::vector<std::uint16_t> up_scales;
};

[[nodiscard]] GateUpPayload make_gateup_payload(
    const std::size_t launch_m, const std::size_t n,
    const std::size_t k) {
  const std::size_t physical_groups = k / 64U;
  const std::size_t k512_groups = k / 512U;
  GateUpPayload result{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
              launch_m, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
              launch_m, k)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(n, k)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(n, k))};

  for (std::size_t row = 0U; row < launch_m; ++row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = group * 64U + 2U * byte;
        result.a[kernels::sm87_a4w4_gateup_down_edge_packed_offset(
            row, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                code(row, inner, 0x1234U),
                code(row, inner + 1U, 0x1234U));
      }
    }
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      result.a_scales[
          kernels::sm87_a4w4_gateup_down_edge_scale_offset(
              row, group, k512_groups)] =
          encode_bf16(0.0021F *
                      static_cast<float>(5U + (3U * row + group) % 17U));
    }
  }
  for (std::size_t row = 0U; row < n; ++row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = group * 64U + 2U * byte;
        const std::size_t offset =
            kernels::sm87_a4w4_gateup_down_edge_packed_offset(
                row, group, byte, physical_groups);
        result.gate[offset] = kernels::sm87_a4w4_pack_signed_pair(
            code(row, inner, 0x4567U),
            code(row, inner + 1U, 0x4567U));
        result.up[offset] = kernels::sm87_a4w4_pack_signed_pair(
            code(row, inner, 0x89abU),
            code(row, inner + 1U, 0x89abU));
      }
    }
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      const std::size_t offset =
          kernels::sm87_a4w4_gateup_down_edge_scale_offset(
              row, group, k512_groups);
      result.gate_scales[offset] = encode_bf16(
          0.0017F *
          static_cast<float>(7U + (5U * row + 3U * group) % 19U));
      result.up_scales[offset] = encode_bf16(
          0.0013F *
          static_cast<float>(9U + (7U * row + group) % 23U));
    }
  }
  return result;
}

struct DownPayload final {
  std::vector<std::uint8_t> weight;
  std::vector<std::uint16_t> scales;
};

[[nodiscard]] DownPayload make_down_payload(const std::size_t n,
                                            const std::size_t k) {
  const std::size_t physical_groups = k / 64U;
  const std::size_t k512_groups = k / 512U;
  DownPayload result{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_down_k512_packed_capacity_bytes(n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_down_k512_scale_capacity_elements(n, k))};
  for (std::size_t row = 0U; row < n; ++row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = group * 64U + 2U * byte;
        result.weight[kernels::sm87_a4w4_down_k512_packed_offset(
            row, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                code(row, inner, 0xcdefU),
                code(row, inner + 1U, 0xcdefU));
      }
    }
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      result.scales[kernels::sm87_a4w4_down_k512_scale_offset(
          row, group, k512_groups)] =
          encode_bf16(0.0023F *
                      static_cast<float>(7U + (7U * row + group) % 23U));
    }
  }
  return result;
}

template <typename T>
[[nodiscard]] bool compare(const std::vector<T>& expected,
                           const std::vector<T>& actual,
                           const std::string& label) {
  if (expected == actual) {
    return true;
  }
  const auto mismatch = std::mismatch(expected.begin(), expected.end(),
                                      actual.begin(), actual.end());
  std::cerr << label << " first mismatch at "
            << std::distance(expected.begin(), mismatch.first) << '\n';
  return false;
}

[[nodiscard]] bool capture_and_replay(
    GuardedDevice<std::uint8_t>& a,
    GuardedDevice<std::uint16_t>& a_scales,
    GuardedDevice<std::uint8_t>& gate,
    GuardedDevice<std::uint16_t>& gate_scales,
    GuardedDevice<std::uint8_t>& up,
    GuardedDevice<std::uint16_t>& up_scales,
    const std::size_t logical_m, const std::size_t launch_m,
    const std::size_t n, const std::size_t k,
    GuardedDevice<std::uint8_t>& scratch,
    GuardedDevice<std::uint8_t>& packed,
    GuardedDevice<std::uint16_t>& scales,
    const unsigned int maximum_ctas) {
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  bool ok = cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create graph stream") &&
            cuda_ok(cudaStreamBeginCapture(
                        stream, cudaStreamCaptureModeThreadLocal),
                    "begin graph capture");
  if (ok) {
    ok = launch_ok(
             kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_test_cuda(
                 a.payload(), a.payload_count(), a_scales.payload(),
                 a_scales.payload_count(), gate.payload(),
                 gate.payload_count(), gate_scales.payload(),
                 gate_scales.payload_count(), up.payload(),
                 up.payload_count(), up_scales.payload(),
                 up_scales.payload_count(), logical_m, launch_m, n, k,
                 kClipRatio, scratch.payload(), scratch.payload_count(),
                 packed.payload(), packed.payload_count(), scales.payload(),
                 scales.payload_count(), maximum_ctas, stream),
             "capture candidate launch") &&
         cuda_ok(cudaStreamEndCapture(stream, &graph),
                 "end graph capture") &&
         cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr,
                                      0U),
                 "instantiate graph") &&
         cuda_ok(cudaGraphLaunch(executable, stream), "graph replay one") &&
         cuda_ok(cudaGraphLaunch(executable, stream), "graph replay two") &&
         cuda_ok(cudaStreamSynchronize(stream), "synchronize graph");
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
                            const bool graph_replay,
                            const bool downstream) {
  const std::string shape = "M" + std::to_string(logical_m) + "/P" +
                            std::to_string(launch_m) + " N" +
                            std::to_string(n) + " K" + std::to_string(k);
  const GateUpPayload host = make_gateup_payload(launch_m, n, k);
  const auto plan =
      kernels::sm87_a4w4_gateup_down_edge_m128n512_ldmatrix_test_plan(
          logical_m, launch_m, n, k, maximum_ctas);
  const std::size_t output_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          launch_m, n);
  const std::size_t output_scales =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          launch_m, n);

  GuardedDevice<std::uint8_t> a;
  GuardedDevice<std::uint16_t> a_scales;
  GuardedDevice<std::uint8_t> gate;
  GuardedDevice<std::uint16_t> gate_scales;
  GuardedDevice<std::uint8_t> up;
  GuardedDevice<std::uint16_t> up_scales;
  GuardedDevice<std::uint8_t> scratch;
  GuardedDevice<std::uint8_t> baseline_packed;
  GuardedDevice<std::uint16_t> baseline_scales;
  GuardedDevice<std::uint8_t> candidate_packed;
  GuardedDevice<std::uint16_t> candidate_scales;
  if (!a.initialize(host.a, kByteGuard, kByteSentinel, "A") ||
      !a_scales.initialize(host.a_scales, kWordGuard, kWordSentinel,
                           "A scales") ||
      !gate.initialize(host.gate, kByteGuard, kByteSentinel, "Gate") ||
      !gate_scales.initialize(host.gate_scales, kWordGuard, kWordSentinel,
                              "Gate scales") ||
      !up.initialize(host.up, kByteGuard, kByteSentinel, "Up") ||
      !up_scales.initialize(host.up_scales, kWordGuard, kWordSentinel,
                            "Up scales") ||
      !scratch.initialize(std::vector<std::uint8_t>(
                              plan.required_scratch_bytes, 0x3cU),
                          kByteGuard, kByteSentinel, "scratch") ||
      !baseline_packed.initialize(
          std::vector<std::uint8_t>(output_bytes, kByteSentinel),
          kByteGuard, kByteSentinel, "baseline packed") ||
      !baseline_scales.initialize(
          std::vector<std::uint16_t>(output_scales, kWordSentinel),
          kWordGuard, kWordSentinel, "baseline scales") ||
      !candidate_packed.initialize(
          std::vector<std::uint8_t>(output_bytes, kByteSentinel),
          kByteGuard, kByteSentinel, "candidate packed") ||
      !candidate_scales.initialize(
          std::vector<std::uint16_t>(output_scales, kWordSentinel),
          kWordGuard, kWordSentinel, "candidate scales")) {
    std::cerr << shape << " allocation/copy failed\n";
    return false;
  }

  const int short_scratch =
      kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_test_cuda(
          a.payload(), a.payload_count(), a_scales.payload(),
          a_scales.payload_count(), gate.payload(), gate.payload_count(),
          gate_scales.payload(), gate_scales.payload_count(), up.payload(),
          up.payload_count(), up_scales.payload(), up_scales.payload_count(),
          logical_m, launch_m, n, k, kClipRatio, scratch.payload(),
          scratch.payload_count() - 1U, candidate_packed.payload(),
          candidate_packed.payload_count(), candidate_scales.payload(),
          candidate_scales.payload_count(), maximum_ctas);
  const int aliased_scratch =
      kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_test_cuda(
          a.payload(), a.payload_count(), a_scales.payload(),
          a_scales.payload_count(), gate.payload(), gate.payload_count(),
          gate_scales.payload(), gate_scales.payload_count(), up.payload(),
          up.payload_count(), up_scales.payload(), up_scales.payload_count(),
          logical_m, launch_m, n, k, kClipRatio,
          candidate_packed.payload(), plan.required_scratch_bytes,
          candidate_packed.payload(), candidate_packed.payload_count(),
          candidate_scales.payload(), candidate_scales.payload_count(),
          maximum_ctas);
  if (short_scratch != static_cast<int>(cudaErrorInvalidValue) ||
      aliased_scratch != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << shape << " scratch capacity/alias rejection failed\n";
    return false;
  }

  if (!launch_ok(
          kernels::launch_sm87_a4w4_gateup_down_k512_edge_test_cuda(
              a.payload(), a.payload_count(), a_scales.payload(),
              a_scales.payload_count(), gate.payload(), gate.payload_count(),
              gate_scales.payload(), gate_scales.payload_count(), up.payload(),
              up.payload_count(), up_scales.payload(), up_scales.payload_count(),
              logical_m, launch_m, n, k, kClipRatio,
              baseline_packed.payload(), baseline_packed.payload_count(),
              baseline_scales.payload(), baseline_scales.payload_count(),
              maximum_ctas),
          "launch incumbent " + shape) ||
      !launch_ok(
          kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_test_cuda(
              a.payload(), a.payload_count(), a_scales.payload(),
              a_scales.payload_count(), gate.payload(), gate.payload_count(),
              gate_scales.payload(), gate_scales.payload_count(), up.payload(),
              up.payload_count(), up_scales.payload(), up_scales.payload_count(),
              logical_m, launch_m, n, k, kClipRatio, scratch.payload(),
              scratch.payload_count(), candidate_packed.payload(),
              candidate_packed.payload_count(), candidate_scales.payload(),
              candidate_scales.payload_count(), maximum_ctas),
          "launch candidate " + shape) ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize " + shape)) {
    return false;
  }

  std::vector<std::uint8_t> baseline_packed_host;
  std::vector<std::uint8_t> candidate_packed_host;
  std::vector<std::uint16_t> baseline_scales_host;
  std::vector<std::uint16_t> candidate_scales_host;
  std::vector<std::uint8_t> scratch_host;
  if (!baseline_packed.copy(baseline_packed_host, "baseline packed") ||
      !candidate_packed.copy(candidate_packed_host, "candidate packed") ||
      !baseline_scales.copy(baseline_scales_host, "baseline scales") ||
      !candidate_scales.copy(candidate_scales_host, "candidate scales") ||
      !scratch.copy(scratch_host, "scratch") ||
      !baseline_packed.guards_intact(baseline_packed_host,
                                     "baseline packed") ||
      !candidate_packed.guards_intact(candidate_packed_host,
                                      "candidate packed") ||
      !baseline_scales.guards_intact(baseline_scales_host,
                                     "baseline scales") ||
      !candidate_scales.guards_intact(candidate_scales_host,
                                      "candidate scales") ||
      !scratch.guards_intact(scratch_host, "scratch") ||
      !compare(baseline_packed_host, candidate_packed_host,
               shape + " packed") ||
      !compare(baseline_scales_host, candidate_scales_host,
               shape + " scales")) {
    return false;
  }

  if (graph_replay &&
      (!capture_and_replay(a, a_scales, gate, gate_scales, up, up_scales,
                           logical_m, launch_m, n, k, scratch,
                           candidate_packed, candidate_scales,
                           maximum_ctas) ||
       !candidate_packed.copy(candidate_packed_host,
                              "graph candidate packed") ||
       !candidate_scales.copy(candidate_scales_host,
                              "graph candidate scales") ||
       !compare(baseline_packed_host, candidate_packed_host,
                shape + " graph packed") ||
       !compare(baseline_scales_host, candidate_scales_host,
                shape + " graph scales"))) {
    return false;
  }

  if (downstream) {
    constexpr std::size_t down_n = 128U;
    constexpr std::size_t down_stride = down_n + 8U;
    const DownPayload down_host = make_down_payload(down_n, n);
    GuardedDevice<std::uint8_t> down_weight;
    GuardedDevice<std::uint16_t> down_scales;
    GuardedDevice<std::uint16_t> baseline_down;
    GuardedDevice<std::uint16_t> candidate_down;
    const std::size_t down_elements = launch_m * down_stride;
    if (!down_weight.initialize(down_host.weight, kByteGuard,
                                kByteSentinel, "Down weight") ||
        !down_scales.initialize(down_host.scales, kWordGuard,
                                kWordSentinel, "Down scales") ||
        !baseline_down.initialize(
            std::vector<std::uint16_t>(down_elements, kBf16Sentinel),
            kWordGuard, kBf16Sentinel, "baseline Down") ||
        !candidate_down.initialize(
            std::vector<std::uint16_t>(down_elements, kBf16Sentinel),
            kWordGuard, kBf16Sentinel, "candidate Down") ||
        !launch_ok(
            kernels::launch_sm87_a4w4_down_k512_macrocell_test_bf16_cuda(
                baseline_packed.payload(), baseline_packed.payload_count(),
                baseline_scales.payload(), baseline_scales.payload_count(),
                down_weight.payload(), down_weight.payload_count(),
                down_scales.payload(), down_scales.payload_count(),
                launch_m, down_n, n, baseline_down.payload(), down_stride,
                down_elements, 2U),
            "launch baseline Down") ||
        !launch_ok(
            kernels::launch_sm87_a4w4_down_k512_macrocell_test_bf16_cuda(
                candidate_packed.payload(), candidate_packed.payload_count(),
                candidate_scales.payload(), candidate_scales.payload_count(),
                down_weight.payload(), down_weight.payload_count(),
                down_scales.payload(), down_scales.payload_count(),
                launch_m, down_n, n, candidate_down.payload(), down_stride,
                down_elements, 2U),
            "launch candidate Down") ||
        !cuda_ok(cudaDeviceSynchronize(), "synchronize Down")) {
      return false;
    }
    std::vector<std::uint16_t> baseline_down_host;
    std::vector<std::uint16_t> candidate_down_host;
    if (!baseline_down.copy(baseline_down_host, "baseline Down") ||
        !candidate_down.copy(candidate_down_host, "candidate Down") ||
        !baseline_down.guards_intact(baseline_down_host, "baseline Down") ||
        !candidate_down.guards_intact(candidate_down_host,
                                      "candidate Down") ||
        !compare(baseline_down_host, candidate_down_host,
                 shape + " Down BF16") ||
        !down_weight.unchanged("Down weight") ||
        !down_scales.unchanged("Down scales")) {
      return false;
    }
  }

  if (!a.unchanged("A") || !a_scales.unchanged("A scales") ||
      !gate.unchanged("Gate") ||
      !gate_scales.unchanged("Gate scales") || !up.unchanged("Up") ||
      !up_scales.unchanged("Up scales")) {
    return false;
  }
  std::cout << "PASS: M128N512 LDSM bit-exact " << shape
            << (graph_replay ? " graphx2" : "")
            << (downstream ? " Down-BF16" : "") << '\n';
  return true;
}

}  // namespace

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }
  kernels::Sm87A4W4GateUpDownEdgeM128N512LdmatrixResources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_resources_cuda(
              &resources),
          "query M128N512 LDSM resources") ||
      resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 255 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 132'096U ||
      resources.configured_dynamic_shared_limit_bytes < 132'096U ||
      resources.device_optin_shared_limit_bytes < 132'096U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 256 ||
      resources.active_blocks_per_sm != 1) {
    std::cerr << "M128N512 LDSM resource gate failed\n";
    return 1;
  }

  const bool ok =
      run_case(128U, 128U, 512U, 512U, 1U, false, false) &&
      // Seventeen M128 tiles over the production 16-CTA grid exercise one
      // complete base wave followed by a residual M tile in the same launch.
      run_case(2'049U, 2'176U, 512U, 512U, 16U, false, false) &&
      // Exercise the exact production K=5120 depth: ten K512 groups cover
      // repeated odd/even stage reuse instead of only the first hand-off.
      run_case(129U, 256U, 1'024U, 5'120U, 3U, true, true);
  if (!ok) {
    return 1;
  }
  std::cout << "M128N512 LDSM GateUp correctness passed: regs="
            << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return 0;
}
