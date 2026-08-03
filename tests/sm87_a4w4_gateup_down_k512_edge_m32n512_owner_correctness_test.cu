#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m32n512_owner.h"
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating.h"
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
inline constexpr std::size_t kGuardBytes = 64U;
inline constexpr std::size_t kGuardWords = 32U;
inline constexpr std::uint8_t kByteSentinel = 0xa5U;
inline constexpr std::uint16_t kWordSentinel = 0xadadU;
inline constexpr unsigned int kBaselineMaximumCtas = 16U;

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
      properties.sharedMemPerBlockOptin <
          kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes ||
      properties.sharedMemPerMultiprocessor <
          2U *
              kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes) {
    std::cout << "SKIP: requires 16-SM SM87 with two resident owner CTAs\n";
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

  [[nodiscard]] bool reset_payload_async(const unsigned char byte_value,
                                         cudaStream_t stream,
                                         const std::string& label) const {
    return cuda_ok(cudaMemsetAsync(payload(), byte_value,
                                   payload_count_ * sizeof(T), stream),
                   "reset " + label + " payload");
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

[[nodiscard]] GateUpPayload make_payload(const std::size_t launch_m,
                                         const std::size_t n,
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
      result.a_scales[kernels::sm87_a4w4_gateup_down_edge_scale_offset(
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

struct GateUpDevice final {
  GuardedDevice<std::uint8_t> a;
  GuardedDevice<std::uint16_t> a_scales;
  GuardedDevice<std::uint8_t> gate;
  GuardedDevice<std::uint16_t> gate_scales;
  GuardedDevice<std::uint8_t> up;
  GuardedDevice<std::uint16_t> up_scales;
};

struct OutputDevice final {
  GuardedDevice<std::uint8_t> packed;
  GuardedDevice<std::uint16_t> scales;
};

[[nodiscard]] int launch_baseline(
    const GateUpDevice& input, const std::size_t logical_m,
    const std::size_t launch_m, const std::size_t n, const std::size_t k,
    OutputDevice& output, const std::size_t packed_capacity,
    const std::size_t scale_capacity, cudaStream_t stream) {
  return kernels::
      launch_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_test_cuda(
          input.a.payload(), input.a.payload_count(), input.a_scales.payload(),
          input.a_scales.payload_count(), input.gate.payload(),
          input.gate.payload_count(), input.gate_scales.payload(),
          input.gate_scales.payload_count(), input.up.payload(),
          input.up.payload_count(), input.up_scales.payload(),
          input.up_scales.payload_count(), logical_m, launch_m, n, k,
          kClipRatio, output.packed.payload(), packed_capacity,
          output.scales.payload(), scale_capacity, kBaselineMaximumCtas,
          stream);
}

[[nodiscard]] int launch_candidate(
    const GateUpDevice& input, const std::size_t logical_m,
    const std::size_t launch_m, const std::size_t n, const std::size_t k,
    OutputDevice& output, const std::size_t packed_capacity,
    const std::size_t scale_capacity, cudaStream_t stream) {
  return kernels::
      launch_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_test_cuda(
          input.a.payload(), input.a.payload_count(), input.a_scales.payload(),
          input.a_scales.payload_count(), input.gate.payload(),
          input.gate.payload_count(), input.gate_scales.payload(),
          input.gate_scales.payload_count(), input.up.payload(),
          input.up.payload_count(), input.up_scales.payload(),
          input.up_scales.payload_count(), logical_m, launch_m, n, k,
          kClipRatio, output.packed.payload(), packed_capacity,
          output.scales.payload(), scale_capacity, stream);
}

[[nodiscard]] bool resources_valid(
    const kernels::Sm87A4W4GateUpDownEdgeM32N512OwnerResources& resources) {
  return resources.registers_per_thread > 0 &&
         resources.registers_per_thread <=
             static_cast<int>(
                 kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerMaximumRegisters) &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes &&
         resources.configured_dynamic_shared_limit_bytes >=
             kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes &&
         resources.device_optin_shared_limit_bytes >=
             kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes &&
         resources.device_shared_per_sm_bytes >=
             2U *
                 kernels::
                     kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<int>(
                 kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerThreads) &&
         resources.active_blocks_per_sm ==
             static_cast<int>(
                 kernels::kSm87A4W4GateUpDownEdgeM32N512OwnerCtasPerSm);
}

template <typename T>
[[nodiscard]] bool compare_exact(const std::vector<T>& expected,
                                 const std::vector<T>& actual,
                                 const std::string& unit,
                                 const std::string& label) {
  if (expected == actual) {
    return true;
  }
  const auto mismatch = std::mismatch(expected.begin(), expected.end(),
                                      actual.begin(), actual.end());
  std::cerr << label << " first mismatch at " << unit << ' '
            << std::distance(expected.begin(), mismatch.first) << '\n';
  return false;
}

[[nodiscard]] bool capture_and_replay(
    const GateUpDevice& input, const std::size_t logical_m,
    const std::size_t launch_m, const std::size_t n, const std::size_t k,
    OutputDevice& candidate) {
  kernels::Sm87A4W4GateUpDownEdgeM32N512OwnerResources resources{};
  if (!launch_ok(
          kernels::
              query_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_resources_cuda(
                  &resources),
          "query owner resources before graph") ||
      !resources_valid(resources)) {
    std::cerr << "owner resource gate failed before graph capture\n";
    return false;
  }

  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  bool ok = cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create owner graph stream");
  if (ok) {
    ok = candidate.packed.reset_payload_async(kByteSentinel, stream,
                                               "owner packed") &&
         candidate.scales.reset_payload_async(
             static_cast<unsigned char>(kWordSentinel & 0xffU), stream,
             "owner scales") &&
         cuda_ok(cudaStreamSynchronize(stream),
                 "synchronize owner graph payload reset");
  }
  if (ok) {
    ok = cuda_ok(cudaStreamBeginCapture(
                     stream, cudaStreamCaptureModeThreadLocal),
                 "begin owner graph capture");
  }
  if (ok) {
    ok = launch_ok(
             launch_candidate(input, logical_m, launch_m, n, k, candidate,
                              candidate.packed.payload_count(),
                              candidate.scales.payload_count(), stream),
             "capture owner launch") &&
         cuda_ok(cudaStreamEndCapture(stream, &graph),
                 "end owner graph capture") &&
         cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr,
                                      0U),
                 "instantiate owner graph") &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 "first owner graph replay") &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 "second owner graph replay") &&
         cuda_ok(cudaStreamSynchronize(stream),
                 "synchronize owner graph replay");
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
                            const bool graph_replay) {
  const std::string shape = "M" + std::to_string(logical_m) +
                            "/P" + std::to_string(launch_m) +
                            " N" + std::to_string(n) +
                            " K" + std::to_string(k);
  const GateUpPayload host = make_payload(launch_m, n, k);
  const std::size_t output_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(launch_m, n);
  const std::size_t output_scales =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(launch_m,
                                                                  n);

  GateUpDevice input;
  OutputDevice baseline;
  OutputDevice candidate;
  if (!input.a.initialize(host.a, kGuardBytes, kByteSentinel, "A") ||
      !input.a_scales.initialize(host.a_scales, kGuardWords, kWordSentinel,
                                 "A scales") ||
      !input.gate.initialize(host.gate, kGuardBytes, kByteSentinel,
                             "Gate") ||
      !input.gate_scales.initialize(host.gate_scales, kGuardWords,
                                    kWordSentinel, "Gate scales") ||
      !input.up.initialize(host.up, kGuardBytes, kByteSentinel, "Up") ||
      !input.up_scales.initialize(host.up_scales, kGuardWords, kWordSentinel,
                                  "Up scales") ||
      !baseline.packed.initialize(
          std::vector<std::uint8_t>(output_bytes, kByteSentinel),
          kGuardBytes, kByteSentinel, "baseline packed") ||
      !baseline.scales.initialize(
          std::vector<std::uint16_t>(output_scales, kWordSentinel),
          kGuardWords, kWordSentinel, "baseline scales") ||
      !candidate.packed.initialize(
          std::vector<std::uint8_t>(output_bytes, kByteSentinel),
          kGuardBytes, kByteSentinel, "candidate packed") ||
      !candidate.scales.initialize(
          std::vector<std::uint16_t>(output_scales, kWordSentinel),
          kGuardWords, kWordSentinel, "candidate scales")) {
    std::cerr << shape << " allocation/copy failed\n";
    return false;
  }

  cudaStream_t stream{};
  bool ok = cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create owner direct stream");
  if (ok) {
    const int short_output =
        launch_candidate(input, logical_m, launch_m, n, k, candidate,
                         output_bytes - 1U, output_scales, stream);
    if (short_output != static_cast<int>(cudaErrorInvalidValue)) {
      std::cerr << shape << " accepted short output capacity\n";
      ok = false;
    }
  }
  if (ok) {
    const int misaligned_input = kernels::
        launch_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_test_cuda(
            input.a.payload() + 1U, input.a.payload_count() - 1U,
            input.a_scales.payload(), input.a_scales.payload_count(),
            input.gate.payload(), input.gate.payload_count(),
            input.gate_scales.payload(), input.gate_scales.payload_count(),
            input.up.payload(), input.up.payload_count(),
            input.up_scales.payload(), input.up_scales.payload_count(),
            logical_m, launch_m, n, k, kClipRatio,
            candidate.packed.payload(), output_bytes,
            candidate.scales.payload(), output_scales, stream);
    if (misaligned_input != static_cast<int>(cudaErrorInvalidValue)) {
      std::cerr << shape << " accepted misaligned A input\n";
      ok = false;
    }
  }
  if (ok) {
    const int aliased_output = kernels::
        launch_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_test_cuda(
            input.a.payload(), input.a.payload_count(),
            input.a_scales.payload(), input.a_scales.payload_count(),
            input.gate.payload(), input.gate.payload_count(),
            input.gate_scales.payload(), input.gate_scales.payload_count(),
            input.up.payload(), input.up.payload_count(),
            input.up_scales.payload(), input.up_scales.payload_count(),
            logical_m, launch_m, n, k, kClipRatio, input.a.payload(),
            output_bytes, candidate.scales.payload(), output_scales,
            stream);
    if (aliased_output != static_cast<int>(cudaErrorInvalidValue)) {
      std::cerr << shape << " accepted aliased packed output\n";
      ok = false;
    }
  }
  if (ok) {
    ok = launch_ok(launch_baseline(input, logical_m, launch_m, n, k,
                                   baseline, output_bytes, output_scales,
                                   stream),
                   "launch pairfeed baseline " + shape) &&
         launch_ok(launch_candidate(input, logical_m, launch_m, n, k,
                                    candidate, output_bytes, output_scales,
                                    stream),
                   "launch owner candidate " + shape) &&
         cuda_ok(cudaStreamSynchronize(stream),
                 "synchronize nondefault stream " + shape);
  }
  if (stream != nullptr) {
    (void)cudaStreamDestroy(stream);
  }
  if (!ok) {
    return false;
  }

  std::vector<std::uint8_t> baseline_packed;
  std::vector<std::uint8_t> candidate_packed;
  std::vector<std::uint16_t> baseline_scales;
  std::vector<std::uint16_t> candidate_scales;
  if (!baseline.packed.copy(baseline_packed, "baseline packed") ||
      !candidate.packed.copy(candidate_packed, "candidate packed") ||
      !baseline.scales.copy(baseline_scales, "baseline scales") ||
      !candidate.scales.copy(candidate_scales, "candidate scales") ||
      !baseline.packed.guards_intact(baseline_packed,
                                     "baseline packed") ||
      !candidate.packed.guards_intact(candidate_packed,
                                      "candidate packed") ||
      !baseline.scales.guards_intact(baseline_scales,
                                     "baseline scales") ||
      !candidate.scales.guards_intact(candidate_scales,
                                      "candidate scales") ||
      !compare_exact(baseline_packed, candidate_packed, "byte",
                     shape + " packed") ||
      !compare_exact(baseline_scales, candidate_scales, "element",
                     shape + " scales")) {
    return false;
  }

  if (graph_replay) {
    if (!capture_and_replay(input, logical_m, launch_m, n, k, candidate) ||
        !candidate.packed.copy(candidate_packed,
                               "graph candidate packed") ||
        !candidate.scales.copy(candidate_scales,
                               "graph candidate scales") ||
        !candidate.packed.guards_intact(candidate_packed,
                                        "graph candidate packed") ||
        !candidate.scales.guards_intact(candidate_scales,
                                        "graph candidate scales") ||
        !compare_exact(baseline_packed, candidate_packed, "byte",
                       shape + " graph packed") ||
        !compare_exact(baseline_scales, candidate_scales, "element",
                       shape + " graph scales")) {
      return false;
    }
  }

  if (!input.a.unchanged("A") ||
      !input.a_scales.unchanged("A scales") ||
      !input.gate.unchanged("Gate") ||
      !input.gate_scales.unchanged("Gate scales") ||
      !input.up.unchanged("Up") ||
      !input.up_scales.unchanged("Up scales")) {
    return false;
  }

  std::cout << "PASS: owner bit-exact " << shape
            << " nondefault-stream"
            << (graph_replay ? " graphx2" : "") << '\n';
  return true;
}

}  // namespace

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }

  kernels::Sm87A4W4GateUpDownEdgeM32N512OwnerResources resources{};
  if (!launch_ok(
          kernels::
              query_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_resources_cuda(
                  &resources),
          "query owner resources") ||
      !resources_valid(resources)) {
    std::cerr << "owner resource gate failed\n";
    return 1;
  }

  const bool ok = run_case(117U, 128U, 512U, 512U, false) &&
                  run_case(128U, 128U, 512U, 512U, true) &&
                  run_case(117U, 128U, 1'024U, 1'536U, true);
  if (!ok) {
    return 1;
  }

  std::cout << "M32N512 owner correctness passed: regs="
            << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return 0;
}
