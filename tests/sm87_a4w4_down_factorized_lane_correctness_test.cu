#include "q3x/kernels/sm87_a4w4_down_factorized_lane.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
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

  [[nodiscard]] std::size_t guards() const noexcept { return guards_; }

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
  std::vector<std::uint8_t> b;
  std::vector<std::uint16_t> b_scales;
  std::vector<std::uint8_t> paired_b_codes;
};

[[nodiscard]] Payload make_payload(const std::size_t launch_m,
                                   const std::size_t n,
                                   const std::size_t k) {
  const std::size_t physical_groups = k / 64U;
  Payload payload{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_down_factorized_packed_capacity_bytes(
              launch_m, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_down_factorized_scale_capacity_elements(
              launch_m)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_down_factorized_packed_capacity_bytes(n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_down_factorized_scale_capacity_elements(n)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_down_factorized_v2_paired_code_capacity_bytes(
              n, k))};

  for (std::size_t row = 0U; row < launch_m; ++row) {
    payload.a_scales[kernels::sm87_a4w4_down_factorized_scale_offset(row)] =
        encode_bf16((row & 1U) == 0U ? 0.03125F : 0.015625F);
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = 64U * group + 2U * byte;
        payload.a[kernels::sm87_a4w4_down_factorized_packed_offset(
            row, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                code(row, inner, 0x1234U),
                code(row, inner + 1U, 0x1234U));
      }
    }
  }
  for (std::size_t row = 0U; row < n; ++row) {
    payload.b_scales[kernels::sm87_a4w4_down_factorized_scale_offset(row)] =
        encode_bf16((row & 2U) == 0U ? 0.015625F : 0.0078125F);
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = 64U * group + 2U * byte;
        payload.b[kernels::sm87_a4w4_down_factorized_packed_offset(
            row, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                code(row, inner, 0x89abU),
                code(row, inner + 1U, 0x89abU));
      }
    }
  }

  for (std::size_t panel_n = 0U; panel_n < n;
       panel_n += kernels::kSm87A4W4DownFactorizedV2PanelN) {
    for (std::size_t n16 = 0U;
         n16 < kernels::kSm87A4W4DownFactorizedV2N16PerPanel; ++n16) {
      const std::size_t even_fragment_n =
          panel_n + n16 * kernels::kSm87A4W4DownFactorizedV2N16;
      const std::size_t odd_fragment_n = even_fragment_n + 8U;
      for (std::size_t group = 0U; group < physical_groups; ++group) {
        for (std::size_t lane = 0U;
             lane < kernels::kSm87A4W4DownFactorizedV2FragmentLanes;
             ++lane) {
          const std::size_t even_row = even_fragment_n + lane / 4U;
          const std::size_t odd_row = odd_fragment_n + lane / 4U;
          const std::size_t byte0 = 4U * (lane % 4U);
          const std::size_t byte1 = 16U + byte0;
          const std::size_t even_source0 =
              kernels::sm87_a4w4_down_factorized_packed_offset(
                  even_row, group, byte0, physical_groups);
          const std::size_t even_source1 =
              kernels::sm87_a4w4_down_factorized_packed_offset(
                  even_row, group, byte1, physical_groups);
          const std::size_t odd_source0 =
              kernels::sm87_a4w4_down_factorized_packed_offset(
                  odd_row, group, byte0, physical_groups);
          const std::size_t odd_source1 =
              kernels::sm87_a4w4_down_factorized_packed_offset(
                  odd_row, group, byte1, physical_groups);
          const std::size_t destination =
              kernels::sm87_a4w4_down_factorized_v2_paired_code_offset(
                  even_fragment_n, group, lane, physical_groups);
          std::memcpy(payload.paired_b_codes.data() + destination,
                      payload.b.data() + even_source0, 4U);
          std::memcpy(payload.paired_b_codes.data() + destination + 4U,
                      payload.b.data() + even_source1, 4U);
          std::memcpy(payload.paired_b_codes.data() + destination + 8U,
                      payload.b.data() + odd_source0, 4U);
          std::memcpy(payload.paired_b_codes.data() + destination + 12U,
                      payload.b.data() + odd_source1, 4U);
        }
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
      packed[kernels::sm87_a4w4_down_factorized_packed_offset(
          outer, group, byte, physical_groups)],
      inner);
}

[[nodiscard]] std::uint16_t cpu_oracle(
    const Payload& payload, const std::size_t row,
    const std::size_t column, const std::size_t k) noexcept {
  const std::size_t physical_groups = k / 64U;
  std::int32_t sum = 0;
  for (std::size_t inner = 0U; inner < k; ++inner) {
    sum += static_cast<std::int32_t>(
               payload_code(payload.a, row, inner, physical_groups)) *
           static_cast<std::int32_t>(
               payload_code(payload.b, column, inner, physical_groups));
  }
  const float a_scale = decode_bf16(
      payload.a_scales[
          kernels::sm87_a4w4_down_factorized_scale_offset(row)]);
  const float b_scale = decode_bf16(
      payload.b_scales[
          kernels::sm87_a4w4_down_factorized_scale_offset(column)]);
  const float scale = a_scale * b_scale;
  return encode_bf16(static_cast<float>(sum) * scale);
}

[[nodiscard]] int launch_test(
    const GuardedDevice<std::uint8_t>& a,
    const GuardedDevice<std::uint16_t>& a_scales,
    const GuardedDevice<std::uint8_t>& b,
    const GuardedDevice<std::uint16_t>& b_scales,
    const std::size_t logical_m, const std::size_t launch_m,
    const std::size_t n, const std::size_t k,
    GuardedDevice<std::uint16_t>& output,
    const std::size_t stride, cudaStream_t stream) noexcept {
  return kernels::launch_sm87_a4w4_down_factorized_lane_test_bf16_cuda(
      a.payload(), a.payload_count(), a_scales.payload(),
      a_scales.payload_count(), b.payload(), b.payload_count(),
      b_scales.payload(), b_scales.payload_count(), logical_m, launch_m, n,
      k, output.payload(), stride, output.payload_count(), 16U, stream);
}

[[nodiscard]] int launch_test_v2(
    const GuardedDevice<std::uint8_t>& a,
    const GuardedDevice<std::uint16_t>& a_scales,
    const GuardedDevice<std::uint8_t>& paired_b_codes,
    const GuardedDevice<std::uint16_t>& b_scales,
    const std::size_t logical_m, const std::size_t launch_m,
    const std::size_t n, const std::size_t k,
    GuardedDevice<std::uint16_t>& output,
    const std::size_t stride, cudaStream_t stream) noexcept {
  return kernels::
      launch_sm87_a4w4_down_factorized_lane_r1_v2_test_bf16_cuda(
          a.payload(), a.payload_count(), a_scales.payload(),
          a_scales.payload_count(), paired_b_codes.payload(),
          paired_b_codes.payload_count(), b_scales.payload(),
          b_scales.payload_count(), logical_m, launch_m, n, k,
          output.payload(), stride, output.payload_count(), 16U, stream);
}

[[nodiscard]] bool capture_and_replay(
    const GuardedDevice<std::uint8_t>& a,
    const GuardedDevice<std::uint16_t>& a_scales,
    const GuardedDevice<std::uint8_t>& b,
    const GuardedDevice<std::uint16_t>& b_scales,
    const std::size_t logical_m, const std::size_t launch_m,
    const std::size_t n, const std::size_t k,
    GuardedDevice<std::uint16_t>& output,
    const std::size_t stride, cudaStream_t stream,
    const std::string& shape, const bool v2) {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  bool ok = cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "begin graph " + shape);
  if (ok) {
    ok = cuda_ok(static_cast<cudaError_t>(
                     v2 ? launch_test_v2(
                              a, a_scales, b, b_scales, logical_m,
                              launch_m, n, k, output, stride, stream)
                        : launch_test(a, a_scales, b, b_scales, logical_m,
                                      launch_m, n, k, output, stride,
                                      stream)),
                 "capture launch " + shape) &&
         cuda_ok(cudaStreamEndCapture(stream, &graph),
                 "end graph " + shape) &&
         cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr,
                                      0U),
                 "instantiate graph " + shape) &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 "graph replay 1 " + shape) &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 "graph replay 2 " + shape) &&
         cuda_ok(cudaStreamSynchronize(stream), "sync graph " + shape);
  }
  if (executable != nullptr) {
    (void)cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }
  return ok;
}

[[nodiscard]] bool run_case(const std::size_t logical_m,
                            const std::size_t launch_m,
                            const std::size_t n,
                            const std::size_t k,
                            const bool graph_replay,
                            const bool check_cpu_oracle = true) {
  const std::string shape = "M" + std::to_string(logical_m) + "/P" +
                            std::to_string(launch_m) + " N" +
                            std::to_string(n) + " K" + std::to_string(k);
  const Payload payload = make_payload(launch_m, n, k);
  const std::size_t stride = n + 8U;
  const std::size_t output_elements = launch_m * stride;
  GuardedDevice<std::uint8_t> a;
  GuardedDevice<std::uint16_t> a_scales;
  GuardedDevice<std::uint8_t> b;
  GuardedDevice<std::uint16_t> b_scales;
  GuardedDevice<std::uint8_t> paired_b_codes;
  GuardedDevice<std::uint16_t> output;
  GuardedDevice<std::uint16_t> output_v2;
  if (!a.initialize(payload.a, kByteGuards, kByteSentinel, "A " + shape) ||
      !a_scales.initialize(payload.a_scales, kWordGuards, kWordSentinel,
                           "A scales " + shape) ||
      !b.initialize(payload.b, kByteGuards, kByteSentinel, "B " + shape) ||
      !b_scales.initialize(payload.b_scales, kWordGuards, kWordSentinel,
                           "B scales " + shape) ||
      !paired_b_codes.initialize(payload.paired_b_codes, kByteGuards,
                                 kByteSentinel,
                                 "paired B codes " + shape) ||
      !output.initialize(
          std::vector<std::uint16_t>(output_elements, kOutputSentinel),
          kWordGuards, kOutputSentinel, "output " + shape) ||
      !output_v2.initialize(
          std::vector<std::uint16_t>(output_elements, kOutputSentinel),
          kWordGuards, kOutputSentinel, "output v2 " + shape)) {
    return false;
  }

  cudaStream_t stream{};
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create non-default stream " + shape)) {
    return false;
  }
  bool ok = cuda_ok(static_cast<cudaError_t>(launch_test(
                        a, a_scales, b, b_scales, logical_m, launch_m, n,
                        k, output, stride, stream)),
                    "launch " + shape) &&
            cuda_ok(static_cast<cudaError_t>(launch_test_v2(
                        a, a_scales, paired_b_codes, b_scales, logical_m,
                        launch_m, n, k, output_v2, stride, stream)),
                    "launch v2 " + shape) &&
            cuda_ok(cudaStreamSynchronize(stream), "sync " + shape);
  if (ok && graph_replay) {
    ok = capture_and_replay(a, a_scales, b, b_scales, logical_m, launch_m,
                            n, k, output, stride, stream, shape + " v1",
                            false) &&
         capture_and_replay(a, a_scales, paired_b_codes, b_scales,
                            logical_m, launch_m, n, k, output_v2, stride,
                            stream, shape + " v2", true);
  }

  std::vector<std::uint16_t> actual;
  std::vector<std::uint16_t> actual_v2;
  if (ok) {
    ok = output.copy(actual, "output " + shape) &&
         output_v2.copy(actual_v2, "output v2 " + shape) &&
         output.guards_intact(actual, "output " + shape) &&
         output_v2.guards_intact(actual_v2, "output v2 " + shape);
  }
  if (ok && actual != actual_v2) {
    std::cerr << "v1/v2 output mismatch " << shape << '\n';
    ok = false;
  }
  for (std::size_t row = 0U;
       row < launch_m && ok && check_cpu_oracle; ++row) {
    for (std::size_t column = 0U; column < stride; ++column) {
      const std::size_t index = output.guards() + row * stride + column;
      const std::uint16_t expected =
          row < logical_m && column < n
              ? cpu_oracle(payload, row, column, k)
              : kOutputSentinel;
      if (actual[index] != expected) {
        std::cerr << shape << " mismatch at (" << row << ',' << column
                  << "): expected=0x" << std::hex << expected
                  << " actual=0x" << actual[index] << std::dec << '\n';
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    ok = a.unchanged("A " + shape) &&
         a_scales.unchanged("A scales " + shape) &&
         b.unchanged("B " + shape) &&
         b_scales.unchanged("B scales " + shape) &&
         paired_b_codes.unchanged("paired B codes " + shape);
  }
  (void)cudaStreamDestroy(stream);
  if (ok) {
    std::cout << "PASS: v1/v2 bit-exact whole-K S32 " << shape
              << (check_cpu_oracle ? " CPU-oracle" : "")
              << (graph_replay ? " graphx2" : "") << '\n';
  }
  return ok;
}

[[nodiscard]] bool invalid_contracts() {
  constexpr std::size_t kLogicalM = 1U;
  constexpr std::size_t kLaunchM = 256U;
  constexpr std::size_t kN = 128U;
  constexpr std::size_t kK = 256U;
  constexpr std::size_t kStride = kN + 8U;
  const Payload payload = make_payload(kLaunchM, kN, kK);
  GuardedDevice<std::uint8_t> a;
  GuardedDevice<std::uint16_t> a_scales;
  GuardedDevice<std::uint8_t> b;
  GuardedDevice<std::uint16_t> b_scales;
  GuardedDevice<std::uint8_t> paired_b_codes;
  GuardedDevice<std::uint16_t> output;
  if (!a.initialize(payload.a, kByteGuards, kByteSentinel, "invalid A") ||
      !a_scales.initialize(payload.a_scales, kWordGuards, kWordSentinel,
                           "invalid A scales") ||
      !b.initialize(payload.b, kByteGuards, kByteSentinel, "invalid B") ||
      !b_scales.initialize(payload.b_scales, kWordGuards, kWordSentinel,
                           "invalid B scales") ||
      !paired_b_codes.initialize(payload.paired_b_codes, kByteGuards,
                                 kByteSentinel,
                                 "invalid paired B codes") ||
      !output.initialize(
          std::vector<std::uint16_t>(kLaunchM * kStride, kOutputSentinel),
          kWordGuards, kOutputSentinel, "invalid output")) {
    return false;
  }

  const auto launch = [&](const std::uint8_t* const a_pointer,
                          const std::size_t a_capacity,
                          const std::uint16_t* const a_scale_pointer,
                          const std::size_t a_scale_capacity,
                          const std::uint8_t* const b_pointer,
                          const std::size_t b_capacity,
                          const std::uint16_t* const b_scale_pointer,
                          const std::size_t b_scale_capacity,
                          const std::size_t logical_m,
                          const std::size_t launch_m,
                          const std::size_t n, const std::size_t k,
                          std::uint16_t* const output_pointer,
                          const std::size_t stride,
                          const std::size_t output_capacity,
                          const unsigned int grid) {
    return kernels::launch_sm87_a4w4_down_factorized_lane_test_bf16_cuda(
        a_pointer, a_capacity, a_scale_pointer, a_scale_capacity,
        b_pointer, b_capacity, b_scale_pointer, b_scale_capacity,
        logical_m, launch_m, n, k, output_pointer, stride,
        output_capacity, grid);
  };
  const auto invalid = [&](const int status, const char* const label) {
    if (status == static_cast<int>(cudaErrorInvalidValue)) {
      return true;
    }
    std::cerr << label << " did not fail with cudaErrorInvalidValue: "
              << status << '\n';
    return false;
  };
  const auto launch_v2 =
      [&](const std::uint8_t* const b_pointer,
          const std::size_t b_capacity) {
        return kernels::
            launch_sm87_a4w4_down_factorized_lane_r1_v2_test_bf16_cuda(
                a.payload(), a.payload_count(), a_scales.payload(),
                a_scales.payload_count(), b_pointer, b_capacity,
                b_scales.payload(), b_scales.payload_count(), kLogicalM,
                kLaunchM, kN, kK, output.payload(), kStride,
                output.payload_count(), 16U);
      };

  const auto valid_args = [&]() {
    return launch(a.payload(), a.payload_count(), a_scales.payload(),
                  a_scales.payload_count(), b.payload(), b.payload_count(),
                  b_scales.payload(), b_scales.payload_count(), kLogicalM,
                  kLaunchM, kN, kK, output.payload(), kStride,
                  output.payload_count(), 16U);
  };
  bool ok = valid_args() == static_cast<int>(cudaSuccess) &&
            cuda_ok(cudaDeviceSynchronize(), "valid contract control");
  ok = ok && invalid(launch(a.payload(), a.payload_count() - 1U,
                            a_scales.payload(), a_scales.payload_count(),
                            b.payload(), b.payload_count(),
                            b_scales.payload(), b_scales.payload_count(),
                            kLogicalM, kLaunchM, kN, kK, output.payload(),
                            kStride, output.payload_count(), 16U),
                     "short A") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count(), b.payload(),
                      b.payload_count() - 1U, b_scales.payload(),
                      b_scales.payload_count(), kLogicalM, kLaunchM, kN, kK,
                      output.payload(), kStride, output.payload_count(), 16U),
               "short B") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count() - 1U, b.payload(),
                      b.payload_count(), b_scales.payload(),
                      b_scales.payload_count(), kLogicalM, kLaunchM, kN, kK,
                      output.payload(), kStride, output.payload_count(), 16U),
               "short A scales") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count(), b.payload(), b.payload_count(),
                      b_scales.payload(), b_scales.payload_count() - 1U,
                      kLogicalM, kLaunchM, kN, kK, output.payload(), kStride,
                      output.payload_count(), 16U),
               "short B scales") &&
       invalid(launch(a.payload() + 1U, a.payload_count() - 1U,
                      a_scales.payload(), a_scales.payload_count(),
                      b.payload(), b.payload_count(), b_scales.payload(),
                      b_scales.payload_count(), kLogicalM, kLaunchM, kN, kK,
                      output.payload(), kStride, output.payload_count(), 16U),
               "misaligned A") &&
       invalid(launch(a.payload(), a.payload_count(),
                      a_scales.payload() + 1U,
                      a_scales.payload_count() - 1U, b.payload(),
                      b.payload_count(), b_scales.payload(),
                      b_scales.payload_count(), kLogicalM, kLaunchM, kN, kK,
                      output.payload(), kStride, output.payload_count(), 16U),
               "misaligned A scales") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count(), b.payload(), b.payload_count(),
                      b_scales.payload(), b_scales.payload_count(), kLogicalM,
                      kLaunchM, kN, kK, output.payload() + 1U, kStride,
                      output.payload_count() - 1U, 16U),
               "misaligned output") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count(), b.payload(), b.payload_count(),
                      b_scales.payload(), b_scales.payload_count(), kLogicalM,
                      kLaunchM, kN, kK,
                      reinterpret_cast<std::uint16_t*>(a.payload()), kN,
                      a.payload_count() / 2U, 16U),
               "output alias A") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count(), b.payload(), b.payload_count(),
                      b_scales.payload(), b_scales.payload_count(), kLogicalM,
                      kLaunchM, kN, kK, output.payload(), kN - 2U,
                      output.payload_count(), 16U),
               "short stride") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count(), b.payload(), b.payload_count(),
                      b_scales.payload(), b_scales.payload_count(), kLogicalM,
                      kLaunchM, kN, kK, output.payload(), kStride,
                      kLogicalM * kStride - 1U, 16U),
               "short output") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count(), b.payload(), b.payload_count(),
                      b_scales.payload(), b_scales.payload_count(), 257U,
                      kLaunchM, kN, kK, output.payload(), kStride,
                      output.payload_count(), 16U),
               "logical exceeds launch") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count(), b.payload(), b.payload_count(),
                      b_scales.payload(), b_scales.payload_count(), kLogicalM,
                      255U, kN, kK, output.payload(), kStride,
                      output.payload_count(), 16U),
               "bad launch padding") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count(), b.payload(), b.payload_count(),
                      b_scales.payload(), b_scales.payload_count(), kLogicalM,
                      kLaunchM, 64U, kK, output.payload(), kStride,
                      output.payload_count(), 16U),
               "bad N") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count(), b.payload(), b.payload_count(),
                      b_scales.payload(), b_scales.payload_count(), kLogicalM,
                      kLaunchM, kN, 128U, output.payload(), kStride,
                      output.payload_count(), 16U),
               "bad K") &&
       invalid(launch(a.payload(), a.payload_count(), a_scales.payload(),
                      a_scales.payload_count(), b.payload(), b.payload_count(),
                      b_scales.payload(), b_scales.payload_count(), kLogicalM,
                      kLaunchM, kN, kK, output.payload(), kStride,
                      output.payload_count(), 0U),
               "zero grid");
  ok = ok &&
       launch_v2(paired_b_codes.payload(),
                 paired_b_codes.payload_count()) ==
           static_cast<int>(cudaSuccess) &&
       cuda_ok(cudaDeviceSynchronize(), "valid v2 contract control") &&
       invalid(launch_v2(paired_b_codes.payload(),
                         paired_b_codes.payload_count() - 1U),
               "short v2 paired B") &&
       invalid(launch_v2(paired_b_codes.payload() + 1U,
                         paired_b_codes.payload_count() - 1U),
               "misaligned v2 paired B");
  if (ok) {
    std::cout << "PASS: invalid contracts fail closed\n";
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

  const auto model_plan = kernels::sm87_a4w4_down_factorized_lane_plan(
      1'853U, 2'048U, 5'120U, 17'408U);
  const auto tail_plan =
      kernels::sm87_a4w4_down_factorized_lane_test_plan(
          257U, 512U, 128U, 256U);
  if (model_plan.m_tiles != 8U || model_plan.n_tiles != 40U ||
      model_plan.k256_stages != 68U || model_plan.work_tiles != 320U ||
      model_plan.launch_ctas != 16U || tail_plan.m_tiles != 2U ||
      tail_plan.work_tiles != 2U ||
      kernels::sm87_a4w4_down_factorized_lane_plan(
          1'853U, 1'920U, 5'120U, 17'408U)
              .launch_ctas != 0U) {
    std::cerr << "host plan contract failed\n";
    return 1;
  }

  kernels::Sm87A4W4DownFactorizedLaneResources resources{};
  const auto resource_status = static_cast<cudaError_t>(
      kernels::query_sm87_a4w4_down_factorized_lane_resources_cuda(
          &resources));
  std::cout << "resources regs=" << resources.registers_per_thread
            << " static_shared=" << resources.static_shared_bytes
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " max_threads=" << resources.maximum_threads_per_block
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  if (!cuda_ok(resource_status, "factorized-lane resource hard gate")) {
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

  kernels::Sm87A4W4DownFactorizedLaneResources v2_resources{};
  const auto v2_resource_status = static_cast<cudaError_t>(
      kernels::query_sm87_a4w4_down_factorized_lane_r1_v2_resources_cuda(
          &v2_resources));
  std::cout << "v2 resources regs=" << v2_resources.registers_per_thread
            << " static_shared=" << v2_resources.static_shared_bytes
            << " dynamic_shared=" << v2_resources.dynamic_shared_bytes
            << " local=" << v2_resources.local_bytes
            << " max_threads=" << v2_resources.maximum_threads_per_block
            << " active_blocks_per_sm="
            << v2_resources.active_blocks_per_sm << '\n';
  if (!cuda_ok(v2_resource_status, "Down R1-v2 resource hard gate") ||
      v2_resources.registers_per_thread > 128 ||
      v2_resources.static_shared_bytes != 0U ||
      v2_resources.dynamic_shared_bytes != 99'072U ||
      v2_resources.local_bytes != 0U ||
      v2_resources.maximum_threads_per_block < 512 ||
      v2_resources.active_blocks_per_sm < 1) {
    std::cerr << "v2 resource query failed the <=128-reg/zero-local gate\n";
    return 1;
  }
  if (kernels::sm87_a4w4_down_factorized_v2_paired_code_capacity_bytes(
          5'120U, 17'408U) !=
      kernels::sm87_a4w4_down_factorized_packed_capacity_bytes(
          5'120U, 17'408U)) {
    std::cerr << "v2 adjacent-N8 publication is not equal-byte\n";
    return 1;
  }

  if (!invalid_contracts() ||
      !run_case(1U, 256U, 128U, 256U, true) ||
      !run_case(257U, 512U, 128U, 256U, false) ||
      !run_case(5U, 256U, 256U, 512U, false) ||
      !run_case(3U, 256U, 128U, 17'408U, false) ||
      !run_case(1'853U, 2'048U, 5'120U, 17'408U, false, false)) {
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
