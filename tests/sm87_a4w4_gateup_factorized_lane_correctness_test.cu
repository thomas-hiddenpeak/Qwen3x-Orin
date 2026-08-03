#include "q3x/kernels/sm87_a4w4_gateup_factorized_lane.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
    return count != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&pointer_),
                      count * sizeof(T)) == cudaSuccess;
  }
  [[nodiscard]] T* get() const noexcept { return pointer_; }

 private:
  T* pointer_{};
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
                   "upload " + label);
  }

  [[nodiscard]] T* payload() const noexcept {
    return storage_.get() + guards_;
  }
  [[nodiscard]] std::size_t payload_count() const noexcept {
    return payload_count_;
  }
  [[nodiscard]] std::size_t guards() const noexcept { return guards_; }

  [[nodiscard]] bool copy(std::vector<T>& output,
                          const std::string& label) const {
    output.resize(initial_.size());
    return cuda_ok(cudaMemcpy(output.data(), storage_.get(),
                              output.size() * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "download " + label);
  }

  [[nodiscard]] bool unchanged(const std::string& label) const {
    std::vector<T> actual;
    if (!copy(actual, label)) {
      return false;
    }
    if (actual == initial_) {
      return true;
    }
    const auto mismatch =
        std::mismatch(initial_.begin(), initial_.end(), actual.begin());
    std::cerr << label << " modified at element "
              << std::distance(initial_.begin(), mismatch.first) << '\n';
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
      std::cerr << label << " guard modified\n";
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

[[nodiscard]] float decode_bf16(const std::uint16_t bits) noexcept {
  const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
  float value{};
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
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

struct Payload final {
  std::vector<std::uint8_t> a;
  std::vector<std::uint16_t> a_scales;
  std::vector<std::uint8_t> gate;
  std::vector<std::uint16_t> gate_scales;
  std::vector<std::uint8_t> up;
  std::vector<std::uint16_t> up_scales;
  std::vector<std::uint8_t> paired_b_codes;
  std::vector<std::uint16_t> paired_b_scales;
};

[[nodiscard]] Payload make_payload(const std::size_t launch_m,
                                   const std::size_t n,
                                   const std::size_t k) {
  const std::size_t physical_groups = k / 64U;
  Payload payload{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_gateup_factorized_packed_capacity_bytes(
              launch_m, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_factorized_scale_capacity_elements(
              launch_m)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_gateup_factorized_packed_capacity_bytes(n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_factorized_scale_capacity_elements(n)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_gateup_factorized_packed_capacity_bytes(n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_factorized_scale_capacity_elements(n)),
      std::vector<std::uint8_t>(
          kernels::
              sm87_a4w4_gateup_factorized_v2_paired_code_capacity_bytes(
                  n, k)),
      std::vector<std::uint16_t>(
          kernels::
              sm87_a4w4_gateup_factorized_v2_paired_scale_capacity_elements(
                  n))};

  for (std::size_t row = 0U; row < launch_m; ++row) {
    payload.a_scales[kernels::sm87_a4w4_gateup_factorized_scale_offset(row)] =
        encode_bf16((row & 1U) == 0U ? 0.03125F : 0.015625F);
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = 64U * group + 2U * byte;
        payload.a[kernels::sm87_a4w4_gateup_factorized_packed_offset(
            row, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                code(row, inner, 0x1234U),
                code(row, inner + 1U, 0x1234U));
      }
    }
  }
  for (std::size_t row = 0U; row < n; ++row) {
    payload.gate_scales[
        kernels::sm87_a4w4_gateup_factorized_scale_offset(row)] =
        encode_bf16((row & 2U) == 0U ? 0.015625F : 0.0078125F);
    payload.up_scales[
        kernels::sm87_a4w4_gateup_factorized_scale_offset(row)] =
        encode_bf16((row & 4U) == 0U ? 0.03125F : 0.015625F);
    payload.paired_b_scales[
        kernels::sm87_a4w4_gateup_factorized_v2_paired_scale_offset(
            row, 0U)] =
        payload.gate_scales[
            kernels::sm87_a4w4_gateup_factorized_scale_offset(row)];
    payload.paired_b_scales[
        kernels::sm87_a4w4_gateup_factorized_v2_paired_scale_offset(
            row, 1U)] =
        payload.up_scales[
            kernels::sm87_a4w4_gateup_factorized_scale_offset(row)];
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = 64U * group + 2U * byte;
        payload.gate[kernels::sm87_a4w4_gateup_factorized_packed_offset(
            row, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                code(row, inner, 0x89abU),
                code(row, inner + 1U, 0x89abU));
        payload.up[kernels::sm87_a4w4_gateup_factorized_packed_offset(
            row, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                code(row, inner, 0xcdefU),
                code(row, inner + 1U, 0xcdefU));
      }
    }
  }

  for (std::size_t fragment_n = 0U; fragment_n < n;
       fragment_n += kernels::kSm87A4W4GateUpFactorizedV2N8) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t lane = 0U;
           lane < kernels::kSm87A4W4GateUpFactorizedV2FragmentLanes;
           ++lane) {
        const std::size_t row = fragment_n + lane / 4U;
        const std::size_t source0 =
            kernels::sm87_a4w4_gateup_factorized_packed_offset(
                row, group, 4U * (lane % 4U), physical_groups);
        const std::size_t source1 =
            kernels::sm87_a4w4_gateup_factorized_packed_offset(
                row, group, 16U + 4U * (lane % 4U), physical_groups);
        const std::size_t destination =
            kernels::sm87_a4w4_gateup_factorized_v2_paired_code_offset(
                fragment_n, group, lane, physical_groups);
        std::memcpy(payload.paired_b_codes.data() + destination,
                    payload.gate.data() + source0, 4U);
        std::memcpy(payload.paired_b_codes.data() + destination + 4U,
                    payload.gate.data() + source1, 4U);
        std::memcpy(payload.paired_b_codes.data() + destination + 8U,
                    payload.up.data() + source0, 4U);
        std::memcpy(payload.paired_b_codes.data() + destination + 12U,
                    payload.up.data() + source1, 4U);
      }
    }
  }
  return payload;
}

[[nodiscard]] std::int8_t payload_code(
    const std::vector<std::uint8_t>& packed,
    const std::size_t outer, const std::size_t inner,
    const std::size_t physical_groups) noexcept {
  const std::size_t group = inner / 64U;
  const std::size_t byte = (inner % 64U) / 2U;
  return kernels::sm87_a4w4_unpack_signed(
      packed[kernels::sm87_a4w4_gateup_factorized_packed_offset(
          outer, group, byte, physical_groups)],
      inner);
}

[[nodiscard]] std::uint16_t cpu_oracle(
    const Payload& payload, const std::size_t row,
    const std::size_t column, const std::size_t k) noexcept {
  const std::size_t physical_groups = k / 64U;
  std::int32_t gate_sum = 0;
  std::int32_t up_sum = 0;
  for (std::size_t inner = 0U; inner < k; ++inner) {
    const std::int32_t a = payload_code(
        payload.a, row, inner, physical_groups);
    gate_sum += a * static_cast<std::int32_t>(payload_code(
                        payload.gate, column, inner, physical_groups));
    up_sum += a * static_cast<std::int32_t>(payload_code(
                      payload.up, column, inner, physical_groups));
  }
  const float a_scale = decode_bf16(
      payload.a_scales[
          kernels::sm87_a4w4_gateup_factorized_scale_offset(row)]);
  const float gate_scale = decode_bf16(
      payload.gate_scales[
          kernels::sm87_a4w4_gateup_factorized_scale_offset(column)]);
  const float up_scale = decode_bf16(
      payload.up_scales[
          kernels::sm87_a4w4_gateup_factorized_scale_offset(column)]);
  const float gate = static_cast<float>(gate_sum) *
                     (a_scale * gate_scale);
  const float up = static_cast<float>(up_sum) * (a_scale * up_scale);
  float product{};
  if (gate >= 0.0F) {
    product = (gate / (1.0F + std::exp(-gate))) * up;
  } else {
    const float exponential = std::exp(gate);
    product = (gate * exponential / (1.0F + exponential)) * up;
  }
  return encode_bf16(product);
}

[[nodiscard]] unsigned int ordered_bf16(
    const std::uint16_t bits) noexcept {
  return (bits & 0x8000U) != 0U
             ? static_cast<unsigned int>(0xffffU - bits)
             : static_cast<unsigned int>(bits) + 0x8000U;
}

[[nodiscard]] unsigned int bf16_ulp_distance(
    const std::uint16_t first, const std::uint16_t second) noexcept {
  const unsigned int a = ordered_bf16(first);
  const unsigned int b = ordered_bf16(second);
  return a > b ? a - b : b - a;
}

struct LaunchArguments final {
  const std::uint8_t* a{};
  std::size_t a_capacity{};
  const std::uint16_t* a_scales{};
  std::size_t a_scale_capacity{};
  const std::uint8_t* gate{};
  std::size_t gate_capacity{};
  const std::uint16_t* gate_scales{};
  std::size_t gate_scale_capacity{};
  const std::uint8_t* up{};
  std::size_t up_capacity{};
  const std::uint16_t* up_scales{};
  std::size_t up_scale_capacity{};
  const std::uint8_t* paired_b_codes{};
  std::size_t paired_b_code_capacity{};
  const std::uint16_t* paired_b_scales{};
  std::size_t paired_b_scale_capacity{};
  std::size_t logical_m{};
  std::size_t launch_m{};
  std::size_t n{};
  std::size_t k{};
  std::size_t primary_width{};
  std::uint16_t* primary{};
  std::size_t primary_stride{};
  std::size_t primary_capacity{};
  std::uint16_t* secondary{};
  std::size_t secondary_stride{};
  std::size_t secondary_capacity{};
  unsigned int maximum_ctas{16U};
  cudaStream_t stream{};
};

[[nodiscard]] int launch(const LaunchArguments& args) noexcept {
  return kernels::launch_sm87_a4w4_gateup_factorized_lane_test_bf16_cuda(
      args.a, args.a_capacity, args.a_scales, args.a_scale_capacity,
      args.gate, args.gate_capacity, args.gate_scales,
      args.gate_scale_capacity, args.up, args.up_capacity, args.up_scales,
      args.up_scale_capacity, args.logical_m, args.launch_m, args.n, args.k,
      args.primary_width, args.primary, args.primary_stride,
      args.primary_capacity, args.secondary, args.secondary_stride,
      args.secondary_capacity, args.maximum_ctas, args.stream);
}

[[nodiscard]] int launch_v2(const LaunchArguments& args) noexcept {
  return kernels::
      launch_sm87_a4w4_gateup_factorized_lane_r1_v2_test_bf16_cuda(
          args.a, args.a_capacity, args.a_scales, args.a_scale_capacity,
          args.paired_b_codes, args.paired_b_code_capacity,
          args.paired_b_scales, args.paired_b_scale_capacity,
          args.logical_m, args.launch_m, args.n, args.k,
          args.primary_width, args.primary, args.primary_stride,
          args.primary_capacity, args.secondary, args.secondary_stride,
          args.secondary_capacity, args.maximum_ctas, args.stream);
}

[[nodiscard]] bool capture_and_replay(const LaunchArguments& arguments,
                                      const std::string& shape,
                                      const bool v2) {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  bool ok = cuda_ok(
      cudaStreamBeginCapture(arguments.stream,
                             cudaStreamCaptureModeThreadLocal),
      "begin graph " + shape);
  if (ok) {
    ok = cuda_ok(static_cast<cudaError_t>(
                     v2 ? launch_v2(arguments) : launch(arguments)),
                 "capture launch " + shape) &&
         cuda_ok(cudaStreamEndCapture(arguments.stream, &graph),
                 "end graph " + shape) &&
         cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr,
                                      0U),
                 "instantiate graph " + shape) &&
         cuda_ok(cudaGraphLaunch(executable, arguments.stream),
                 "graph replay 1 " + shape) &&
         cuda_ok(cudaGraphLaunch(executable, arguments.stream),
                 "graph replay 2 " + shape) &&
         cuda_ok(cudaStreamSynchronize(arguments.stream),
                 "sync graph " + shape);
  }
  if (executable != nullptr) {
    (void)cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }
  return ok;
}

[[nodiscard]] bool compare_plane(
    const std::vector<std::uint16_t>& actual,
    const GuardedDevice<std::uint16_t>& device,
    const Payload& payload, const std::size_t logical_m,
    const std::size_t launch_m, const std::size_t absolute_n_start,
    const std::size_t width, const std::size_t stride,
    const std::size_t k, const std::string& label,
    std::size_t& one_ulp_count) {
  for (std::size_t row = 0U; row < launch_m; ++row) {
    for (std::size_t column = 0U; column < stride; ++column) {
      const std::size_t index =
          device.guards() + row * stride + column;
      if (row >= logical_m || column >= width) {
        if (actual[index] != kOutputSentinel) {
          std::cerr << label << " canary overwritten at (" << row << ','
                    << column << ")\n";
          return false;
        }
        continue;
      }
      const std::uint16_t expected = cpu_oracle(
          payload, row, absolute_n_start + column, k);
      const unsigned int distance =
          bf16_ulp_distance(expected, actual[index]);
      if (distance > 1U) {
        std::cerr << label << " exceeds the 1-BF16-ULP expf/libm bound at ("
                  << row << ',' << column << "): expected=0x" << std::hex
                  << expected << " actual=0x" << actual[index] << std::dec
                  << " distance=" << distance << '\n';
        return false;
      }
      one_ulp_count += distance == 1U ? 1U : 0U;
    }
  }
  return true;
}

[[nodiscard]] bool run_case(const std::size_t logical_m,
                            const std::size_t launch_m,
                            const std::size_t n,
                            const std::size_t k,
                            const std::size_t primary_width,
                            const bool graph_replay,
                            const bool check_cpu_oracle = true) {
  const std::string shape = "M" + std::to_string(logical_m) + "/P" +
                            std::to_string(launch_m) + " N" +
                            std::to_string(n) + " K" + std::to_string(k) +
                            " split" + std::to_string(primary_width);
  const Payload payload = make_payload(launch_m, n, k);
  const std::size_t secondary_width = n - primary_width;
  const std::size_t primary_stride = primary_width + 8U;
  const std::size_t secondary_stride = secondary_width + 8U;
  GuardedDevice<std::uint8_t> a;
  GuardedDevice<std::uint16_t> a_scales;
  GuardedDevice<std::uint8_t> gate;
  GuardedDevice<std::uint16_t> gate_scales;
  GuardedDevice<std::uint8_t> up;
  GuardedDevice<std::uint16_t> up_scales;
  GuardedDevice<std::uint8_t> paired_b_codes;
  GuardedDevice<std::uint16_t> paired_b_scales;
  GuardedDevice<std::uint16_t> primary;
  GuardedDevice<std::uint16_t> secondary;
  GuardedDevice<std::uint16_t> primary_v2;
  GuardedDevice<std::uint16_t> secondary_v2;
  if (!a.initialize(payload.a, kByteGuards, kByteSentinel, "A " + shape) ||
      !a_scales.initialize(payload.a_scales, kWordGuards, kWordSentinel,
                           "A scales " + shape) ||
      !gate.initialize(payload.gate, kByteGuards, kByteSentinel,
                       "Gate " + shape) ||
      !gate_scales.initialize(payload.gate_scales, kWordGuards,
                              kWordSentinel, "Gate scales " + shape) ||
      !up.initialize(payload.up, kByteGuards, kByteSentinel,
                     "Up " + shape) ||
      !up_scales.initialize(payload.up_scales, kWordGuards, kWordSentinel,
                            "Up scales " + shape) ||
      !paired_b_codes.initialize(payload.paired_b_codes, kByteGuards,
                                 kByteSentinel,
                                 "paired B codes " + shape) ||
      !paired_b_scales.initialize(payload.paired_b_scales, kWordGuards,
                                  kWordSentinel,
                                  "paired B scales " + shape) ||
      !primary.initialize(
          std::vector<std::uint16_t>(launch_m * primary_stride,
                                     kOutputSentinel),
          kWordGuards, kOutputSentinel, "primary " + shape) ||
      !secondary.initialize(
          std::vector<std::uint16_t>(launch_m * secondary_stride,
                                     kOutputSentinel),
          kWordGuards, kOutputSentinel, "secondary " + shape) ||
      !primary_v2.initialize(
          std::vector<std::uint16_t>(launch_m * primary_stride,
                                     kOutputSentinel),
          kWordGuards, kOutputSentinel, "primary v2 " + shape) ||
      !secondary_v2.initialize(
          std::vector<std::uint16_t>(launch_m * secondary_stride,
                                     kOutputSentinel),
          kWordGuards, kOutputSentinel, "secondary v2 " + shape)) {
    return false;
  }

  cudaStream_t stream{};
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create non-default stream " + shape)) {
    return false;
  }
  const LaunchArguments args{
      a.payload(),
      a.payload_count(),
      a_scales.payload(),
      a_scales.payload_count(),
      gate.payload(),
      gate.payload_count(),
      gate_scales.payload(),
      gate_scales.payload_count(),
      up.payload(),
      up.payload_count(),
      up_scales.payload(),
      up_scales.payload_count(),
      paired_b_codes.payload(),
      paired_b_codes.payload_count(),
      paired_b_scales.payload(),
      paired_b_scales.payload_count(),
      logical_m,
      launch_m,
      n,
      k,
      primary_width,
      primary.payload(),
      primary_stride,
      primary.payload_count(),
      secondary.payload(),
      secondary_stride,
      secondary.payload_count(),
      16U,
      stream};
  LaunchArguments v2_args = args;
  v2_args.primary = primary_v2.payload();
  v2_args.primary_capacity = primary_v2.payload_count();
  v2_args.secondary = secondary_v2.payload();
  v2_args.secondary_capacity = secondary_v2.payload_count();
  bool ok = cuda_ok(static_cast<cudaError_t>(launch(args)),
                    "launch " + shape) &&
            cuda_ok(static_cast<cudaError_t>(launch_v2(v2_args)),
                    "launch v2 " + shape) &&
            cuda_ok(cudaStreamSynchronize(stream), "sync " + shape);
  if (ok && graph_replay) {
    ok = capture_and_replay(args, shape + " v1", false) &&
         capture_and_replay(v2_args, shape + " v2", true);
  }

  std::vector<std::uint16_t> primary_host;
  std::vector<std::uint16_t> secondary_host;
  std::vector<std::uint16_t> primary_v2_host;
  std::vector<std::uint16_t> secondary_v2_host;
  if (ok) {
    ok = primary.copy(primary_host, "primary " + shape) &&
         secondary.copy(secondary_host, "secondary " + shape) &&
         primary_v2.copy(primary_v2_host, "primary v2 " + shape) &&
         secondary_v2.copy(secondary_v2_host, "secondary v2 " + shape) &&
         primary.guards_intact(primary_host, "primary " + shape) &&
         secondary.guards_intact(secondary_host,
                                 "secondary " + shape) &&
         primary_v2.guards_intact(primary_v2_host,
                                  "primary v2 " + shape) &&
         secondary_v2.guards_intact(secondary_v2_host,
                                    "secondary v2 " + shape);
  }
  if (ok && (primary_host != primary_v2_host ||
             secondary_host != secondary_v2_host)) {
    std::cerr << "v1/v2 output mismatch " << shape << '\n';
    ok = false;
  }
  std::size_t one_ulp_count = 0U;
  if (ok && check_cpu_oracle) {
    ok = compare_plane(primary_host, primary, payload, logical_m, launch_m,
                       0U, primary_width, primary_stride, k,
                       "primary " + shape, one_ulp_count) &&
         compare_plane(secondary_host, secondary, payload, logical_m,
                       launch_m, primary_width, secondary_width,
                       secondary_stride, k, "secondary " + shape,
                       one_ulp_count);
  }
  if (ok) {
    ok = a.unchanged("A " + shape) &&
         a_scales.unchanged("A scales " + shape) &&
         gate.unchanged("Gate " + shape) &&
         gate_scales.unchanged("Gate scales " + shape) &&
         up.unchanged("Up " + shape) &&
         up_scales.unchanged("Up scales " + shape) &&
         paired_b_codes.unchanged("paired B codes " + shape) &&
         paired_b_scales.unchanged("paired B scales " + shape);
  }
  (void)cudaStreamDestroy(stream);
  if (ok) {
    std::cout << "PASS: v1/v2 bit-exact whole-K S32 " << shape
              << (check_cpu_oracle
                      ? " CPU-oracle BF16_ulp<=1 one_ulp=" +
                            std::to_string(one_ulp_count)
                      : "")
              << (graph_replay ? " graphx2" : "") << '\n';
  }
  return ok;
}

__global__ void stale_error_probe() {}

[[nodiscard]] bool invalid_contracts() {
  constexpr std::size_t kLogicalM = 1U;
  constexpr std::size_t kLaunchM = 128U;
  constexpr std::size_t kN = 256U;
  constexpr std::size_t kK = 256U;
  constexpr std::size_t kPrimaryWidth = 128U;
  constexpr std::size_t kPrimaryStride = 136U;
  constexpr std::size_t kSecondaryStride = 136U;
  const Payload payload = make_payload(kLaunchM, kN, kK);
  GuardedDevice<std::uint8_t> a;
  GuardedDevice<std::uint16_t> a_scales;
  GuardedDevice<std::uint8_t> gate;
  GuardedDevice<std::uint16_t> gate_scales;
  GuardedDevice<std::uint8_t> up;
  GuardedDevice<std::uint16_t> up_scales;
  GuardedDevice<std::uint8_t> paired_b_codes;
  GuardedDevice<std::uint16_t> paired_b_scales;
  GuardedDevice<std::uint16_t> primary;
  GuardedDevice<std::uint16_t> secondary;
  if (!a.initialize(payload.a, kByteGuards, kByteSentinel, "invalid A") ||
      !a_scales.initialize(payload.a_scales, kWordGuards, kWordSentinel,
                           "invalid A scales") ||
      !gate.initialize(payload.gate, kByteGuards, kByteSentinel,
                       "invalid Gate") ||
      !gate_scales.initialize(payload.gate_scales, kWordGuards,
                              kWordSentinel, "invalid Gate scales") ||
      !up.initialize(payload.up, kByteGuards, kByteSentinel,
                     "invalid Up") ||
      !up_scales.initialize(payload.up_scales, kWordGuards, kWordSentinel,
                            "invalid Up scales") ||
      !paired_b_codes.initialize(payload.paired_b_codes, kByteGuards,
                                 kByteSentinel, "invalid paired B codes") ||
      !paired_b_scales.initialize(payload.paired_b_scales, kWordGuards,
                                  kWordSentinel,
                                  "invalid paired B scales") ||
      !primary.initialize(
          std::vector<std::uint16_t>(kLaunchM * kPrimaryStride,
                                     kOutputSentinel),
          kWordGuards, kOutputSentinel, "invalid primary") ||
      !secondary.initialize(
          std::vector<std::uint16_t>(kLaunchM * kSecondaryStride,
                                     kOutputSentinel),
          kWordGuards, kOutputSentinel, "invalid secondary")) {
    return false;
  }

  const LaunchArguments base{
      a.payload(),
      a.payload_count(),
      a_scales.payload(),
      a_scales.payload_count(),
      gate.payload(),
      gate.payload_count(),
      gate_scales.payload(),
      gate_scales.payload_count(),
      up.payload(),
      up.payload_count(),
      up_scales.payload(),
      up_scales.payload_count(),
      paired_b_codes.payload(),
      paired_b_codes.payload_count(),
      paired_b_scales.payload(),
      paired_b_scales.payload_count(),
      kLogicalM,
      kLaunchM,
      kN,
      kK,
      kPrimaryWidth,
      primary.payload(),
      kPrimaryStride,
      primary.payload_count(),
      secondary.payload(),
      kSecondaryStride,
      secondary.payload_count(),
      16U,
      nullptr};
  const auto expect_invalid =
      [&](const char* const label, const auto& mutate) {
        LaunchArguments candidate = base;
        mutate(candidate);
        const int status = launch(candidate);
        if (status == static_cast<int>(cudaErrorInvalidValue)) {
          return true;
        }
        std::cerr << label << " did not fail with cudaErrorInvalidValue: "
                  << status << '\n';
        return false;
      };
  const auto expect_invalid_v2 =
      [&](const char* const label, const auto& mutate) {
        LaunchArguments candidate = base;
        mutate(candidate);
        const int status = launch_v2(candidate);
        if (status == static_cast<int>(cudaErrorInvalidValue)) {
          return true;
        }
        std::cerr << label << " did not fail with cudaErrorInvalidValue: "
                  << status << '\n';
        return false;
      };

  bool ok = expect_invalid("short A", [](LaunchArguments& x) {
              --x.a_capacity;
            }) &&
            expect_invalid("short Gate", [](LaunchArguments& x) {
              --x.gate_capacity;
            }) &&
            expect_invalid("short Up", [](LaunchArguments& x) {
              --x.up_capacity;
            }) &&
            expect_invalid("short A scales", [](LaunchArguments& x) {
              --x.a_scale_capacity;
            }) &&
            expect_invalid("short Gate scales", [](LaunchArguments& x) {
              --x.gate_scale_capacity;
            }) &&
            expect_invalid("short Up scales", [](LaunchArguments& x) {
              --x.up_scale_capacity;
            }) &&
            expect_invalid("short primary", [](LaunchArguments& x) {
              x.primary_capacity = x.logical_m * x.primary_stride - 1U;
            }) &&
            expect_invalid("short secondary", [](LaunchArguments& x) {
              x.secondary_capacity =
                  x.logical_m * x.secondary_stride - 1U;
            }) &&
            expect_invalid("misaligned A", [](LaunchArguments& x) {
              ++x.a;
            }) &&
            expect_invalid("misaligned Gate scales",
                           [](LaunchArguments& x) { ++x.gate_scales; }) &&
            expect_invalid("misaligned primary", [](LaunchArguments& x) {
              ++x.primary;
            }) &&
            expect_invalid("primary alias A", [&](LaunchArguments& x) {
              x.primary = reinterpret_cast<std::uint16_t*>(a.payload());
              x.primary_capacity = a.payload_count() / 2U;
            }) &&
            expect_invalid("plane alias", [&](LaunchArguments& x) {
              x.secondary = primary.payload();
            }) &&
            expect_invalid("logical exceeds launch",
                           [](LaunchArguments& x) { x.logical_m = 129U; }) &&
            expect_invalid("noncanonical launch",
                           [](LaunchArguments& x) { x.launch_m = 256U; }) &&
            expect_invalid("bad N", [](LaunchArguments& x) {
              x.n = 128U;
              x.primary_width = 64U;
            }) &&
            expect_invalid("bad K", [](LaunchArguments& x) {
              x.k = 128U;
            }) &&
            expect_invalid("bad split", [](LaunchArguments& x) {
              x.primary_width = 64U;
            }) &&
            expect_invalid("short primary stride", [](LaunchArguments& x) {
              x.primary_stride = 126U;
            }) &&
            expect_invalid("short secondary stride",
                           [](LaunchArguments& x) {
                             x.secondary_stride = 126U;
                           }) &&
            expect_invalid("zero grid", [](LaunchArguments& x) {
              x.maximum_ctas = 0U;
            });
  if (ok) {
    ok = cuda_ok(static_cast<cudaError_t>(launch_v2(base)),
                 "valid v2 contract control") &&
         cuda_ok(cudaDeviceSynchronize(), "sync valid v2 contract") &&
         expect_invalid_v2("short v2 paired codes", [](LaunchArguments& x) {
           --x.paired_b_code_capacity;
         }) &&
         expect_invalid_v2("short v2 paired scales", [](LaunchArguments& x) {
           --x.paired_b_scale_capacity;
         }) &&
         expect_invalid_v2("misaligned v2 paired codes",
                           [](LaunchArguments& x) { ++x.paired_b_codes; });
  }
  if (!ok) {
    return false;
  }

  stale_error_probe<<<1U, 0U>>>();
  if (cudaPeekAtLastError() == cudaSuccess) {
    std::cerr << "failed to manufacture a stale CUDA launch error\n";
    return false;
  }
  ok = cuda_ok(static_cast<cudaError_t>(launch(base)),
               "valid launch clears stale CUDA error") &&
       cuda_ok(cudaDeviceSynchronize(), "sync stale-error control");
  if (ok) {
    std::cout << "PASS: invalid contracts and stale-error clearing\n";
  }
  return ok;
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

  const auto model_plan = kernels::sm87_a4w4_gateup_factorized_lane_plan(
      1'853U, 1'920U, 17'408U, 5'120U);
  const auto tail_plan =
      kernels::sm87_a4w4_gateup_factorized_lane_test_plan(
          129U, 256U, 256U, 256U, 128U);
  if (model_plan.m_tiles != 15U || model_plan.n_tiles != 136U ||
      model_plan.k256_stages != 20U || model_plan.work_tiles != 2'040U ||
      model_plan.launch_ctas != 16U ||
      model_plan.primary_width != 12'288U ||
      model_plan.secondary_width != 5'120U || tail_plan.m_tiles != 2U ||
      tail_plan.work_tiles != 4U ||
      kernels::sm87_a4w4_gateup_factorized_lane_plan(
          1'853U, 2'048U, 17'408U, 5'120U)
              .launch_ctas != 0U) {
    std::cerr << "host plan contract failed\n";
    return 1;
  }

  kernels::Sm87A4W4GateUpFactorizedLaneResources resources{};
  const auto resource_status = static_cast<cudaError_t>(
      kernels::query_sm87_a4w4_gateup_factorized_lane_resources_cuda(
          &resources));
  std::cout << "resources regs=" << resources.registers_per_thread
            << " static_shared=" << resources.static_shared_bytes
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " max_threads=" << resources.maximum_threads_per_block
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  if (!cuda_ok(resource_status, "paired Gate+Up resource hard gate")) {
    std::cerr << "STOP: resource gate failed before bit correctness or "
                 "performance\n";
    return 1;
  }
  if (resources.registers_per_thread > 128 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 99'072U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 512 ||
      resources.active_blocks_per_sm < 1) {
    std::cerr << "resource query returned success outside the hard gate\n";
    return 1;
  }

  kernels::Sm87A4W4GateUpFactorizedLaneResources v2_resources{};
  const auto v2_resource_status = static_cast<cudaError_t>(
      kernels::
          query_sm87_a4w4_gateup_factorized_lane_r1_v2_resources_cuda(
              &v2_resources));
  std::cout << "v2 resources regs=" << v2_resources.registers_per_thread
            << " static_shared=" << v2_resources.static_shared_bytes
            << " dynamic_shared=" << v2_resources.dynamic_shared_bytes
            << " local=" << v2_resources.local_bytes
            << " max_threads=" << v2_resources.maximum_threads_per_block
            << " active_blocks_per_sm="
            << v2_resources.active_blocks_per_sm << '\n';
  if (!cuda_ok(v2_resource_status, "Gate+Up R1-v2 resource hard gate") ||
      v2_resources.registers_per_thread > 128 ||
      v2_resources.static_shared_bytes != 0U ||
      v2_resources.dynamic_shared_bytes != 99'072U ||
      v2_resources.local_bytes != 0U ||
      v2_resources.maximum_threads_per_block < 512 ||
      v2_resources.active_blocks_per_sm < 1) {
    std::cerr << "v2 resource query failed the <=128-reg/zero-local gate\n";
    return 1;
  }

  if (kernels::
          sm87_a4w4_gateup_factorized_v2_paired_code_capacity_bytes(
              17'408U, 5'120U) !=
          17'408U * 5'120U ||
      kernels::
          sm87_a4w4_gateup_factorized_v2_paired_scale_capacity_elements(
              17'408U) !=
          2U * 17'408U) {
    std::cerr << "v2 equal-byte publication contract failed\n";
    return 1;
  }

  if (!invalid_contracts() ||
      !run_case(1U, 128U, 256U, 256U, 128U, true) ||
      !run_case(129U, 256U, 256U, 256U, 128U, false) ||
      !run_case(5U, 128U, 512U, 512U, 256U, false) ||
      !run_case(3U, 128U, 256U, 5'120U, 128U, false) ||
      !run_case(1'853U, 1'920U, 17'408U, 5'120U, 12'288U, false,
                false)) {
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
