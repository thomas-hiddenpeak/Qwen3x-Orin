#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m64n256_marlin.h"
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
          kernels::kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes ||
      properties.sharedMemPerMultiprocessor <
          kernels::kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes) {
    std::cout << "SKIP: requires 16-SM SM87 with one resident Marlin CTA\n";
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
      launch_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_test_cuda(
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
    const kernels::Sm87A4W4GateUpDownEdgeM64N256MarlinResources& resources) {
  return resources.registers_per_thread > 0 &&
         resources.registers_per_thread <=
             static_cast<int>(
                 kernels::kSm87A4W4GateUpDownEdgeM64N256MarlinMaximumRegisters) &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kernels::kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes &&
         resources.configured_dynamic_shared_limit_bytes >=
             kernels::kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes &&
         resources.device_optin_shared_limit_bytes >=
             kernels::kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes &&
         resources.device_shared_per_sm_bytes >=
             kernels::kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<int>(
                 kernels::kSm87A4W4GateUpDownEdgeM64N256MarlinThreads) &&
         resources.active_blocks_per_sm ==
             static_cast<int>(
                 kernels::kSm87A4W4GateUpDownEdgeM64N256MarlinCtasPerSm);
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

[[nodiscard]] bool run_case(const std::size_t logical_m,
                            const std::size_t launch_m,
                            const std::size_t n,
                            const std::size_t k) {
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
                    "create marlin direct stream");
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
        launch_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_test_cuda(
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
        launch_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_test_cuda(
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
                   "launch marlin candidate " + shape) &&
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

  if (!input.a.unchanged("A") ||
      !input.a_scales.unchanged("A scales") ||
      !input.gate.unchanged("Gate") ||
      !input.gate_scales.unchanged("Gate scales") ||
      !input.up.unchanged("Up") ||
      !input.up_scales.unchanged("Up scales")) {
    return false;
  }

  std::cout << "PASS: marlin bit-exact " << shape
            << " nondefault-stream\n";
  return true;
}

// The numerical oracle remains bounded.  This separate zero-data route
// proves that the production launcher admits the real P1853 model plan,
// visits its padded tail, and publishes the complete canonical edge without
// allocating a host-side full-model oracle.
[[nodiscard]] bool run_p1853_production_route() {
  constexpr std::size_t kLogicalM = 1'853U;
  constexpr std::size_t kLaunchM = 1'920U;
  constexpr std::size_t kN = 17'408U;
  constexpr std::size_t kK = 5'120U;
  const auto plan =
      kernels::sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_plan(
          kLogicalM, kLaunchM, kN, kK);
  if (plan.logical_token_count != kLogicalM ||
      plan.launch_token_count != kLaunchM || plan.m64_tiles != 30U ||
      plan.edge_groups != 34U || plan.input_k512_groups != 10U ||
      plan.input_physical_k64_groups != 80U ||
      plan.output_physical_k64_groups != 272U ||
      plan.work_edges != 1'020U || plan.launch_ctas != 16U ||
      plan.minimum_edges_per_cta != 63U ||
      plan.maximum_edges_per_cta != 64U) {
    std::cerr << "P1853 production plan mismatch\n";
    return false;
  }

  const std::size_t a_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          kLaunchM, kK);
  const std::size_t a_scales =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          kLaunchM, kK);
  const std::size_t b_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(kN, kK);
  const std::size_t b_scales =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(kN, kK);
  const std::size_t output_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          kLaunchM, kN);
  const std::size_t output_scales =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          kLaunchM, kN);

  DeviceBuffer<std::uint8_t> a;
  DeviceBuffer<std::uint16_t> a_scale;
  DeviceBuffer<std::uint8_t> gate;
  DeviceBuffer<std::uint16_t> gate_scale;
  DeviceBuffer<std::uint8_t> up;
  DeviceBuffer<std::uint16_t> up_scale;
  DeviceBuffer<std::uint8_t> output;
  DeviceBuffer<std::uint16_t> output_scale;
  if (!a.allocate(a_bytes) || !a_scale.allocate(a_scales) ||
      !gate.allocate(b_bytes) || !gate_scale.allocate(b_scales) ||
      !up.allocate(b_bytes) || !up_scale.allocate(b_scales) ||
      !output.allocate(output_bytes) ||
      !output_scale.allocate(output_scales)) {
    std::cerr << "P1853 production route allocation failed\n";
    return false;
  }

  cudaStream_t stream{};
  bool ok = cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create P1853 route stream");
  if (ok) {
    ok = cuda_ok(cudaMemsetAsync(a.get(), 0, a_bytes, stream),
                 "zero P1853 A") &&
         cuda_ok(cudaMemsetAsync(a_scale.get(), 0,
                                 a_scales * sizeof(std::uint16_t), stream),
                 "zero P1853 A scales") &&
         cuda_ok(cudaMemsetAsync(gate.get(), 0, b_bytes, stream),
                 "zero P1853 Gate") &&
         cuda_ok(cudaMemsetAsync(gate_scale.get(), 0,
                                 b_scales * sizeof(std::uint16_t), stream),
                 "zero P1853 Gate scales") &&
         cuda_ok(cudaMemsetAsync(up.get(), 0, b_bytes, stream),
                 "zero P1853 Up") &&
         cuda_ok(cudaMemsetAsync(up_scale.get(), 0,
                                 b_scales * sizeof(std::uint16_t), stream),
                 "zero P1853 Up scales") &&
         cuda_ok(cudaMemsetAsync(output.get(), kByteSentinel, output_bytes,
                                 stream),
                 "poison P1853 output") &&
         cuda_ok(cudaMemsetAsync(output_scale.get(), 0xad,
                                 output_scales * sizeof(std::uint16_t),
                                 stream),
                 "poison P1853 output scales");
  }
  if (ok) {
    ok = launch_ok(
             kernels::launch_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_cuda(
                 a.get(), a_bytes, a_scale.get(), a_scales, gate.get(),
                 b_bytes, gate_scale.get(), b_scales, up.get(), b_bytes,
                 up_scale.get(), b_scales, kLogicalM, kLaunchM, kN, kK,
                 kClipRatio, output.get(), output_bytes, output_scale.get(),
                 output_scales, stream),
             "launch P1853 production route") &&
         cuda_ok(cudaStreamSynchronize(stream),
                 "synchronize P1853 production route");
  }

  std::vector<std::uint8_t> host_output(output_bytes);
  std::vector<std::uint16_t> host_scales(output_scales);
  if (ok) {
    ok = cuda_ok(cudaMemcpy(host_output.data(), output.get(), output_bytes,
                            cudaMemcpyDeviceToHost),
                 "copy P1853 output") &&
         cuda_ok(cudaMemcpy(host_scales.data(), output_scale.get(),
                            output_scales * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost),
                 "copy P1853 output scales");
  }
  if (stream != nullptr) {
    (void)cudaStreamDestroy(stream);
  }
  if (!ok) {
    return false;
  }

  const std::uint16_t unit_scale = encode_bf16(1.0F);
  if (!std::all_of(host_output.begin(), host_output.end(),
                   [](const std::uint8_t value) { return value == 0U; }) ||
      !std::all_of(host_scales.begin(), host_scales.end(),
                   [&](const std::uint16_t value) {
                     return value == unit_scale;
                   })) {
    std::cerr << "P1853 zero route did not publish canonical zero A4/K512\n";
    return false;
  }
  std::cout << "PASS: P1853/P1920 production plan and zero-data route\n";
  return true;
}

}  // namespace

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }

  kernels::Sm87A4W4GateUpDownEdgeM64N256MarlinResources resources{};
  if (!launch_ok(
          kernels::
              query_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_resources_cuda(
                  &resources),
          "query marlin resources") ||
      !resources_valid(resources)) {
    std::cerr << "marlin resource gate failed\n";
    return 1;
  }

  const bool ok = run_case(117U, 128U, 512U, 1'536U) &&
                  run_p1853_production_route();
  if (!ok) {
    return 1;
  }

  const bool production_eligible = resources.local_bytes == 0U;
  std::cout << "M64N256 Marlin correctness passed: regs="
            << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local_bytes=" << resources.local_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << " production_eligible="
            << (production_eligible ? "yes" : "no")
            << '\n';
  return 0;
}
