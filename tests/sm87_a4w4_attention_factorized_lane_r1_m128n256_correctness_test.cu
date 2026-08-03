#include "q3x/kernels/sm87_a4w4_attention_factorized_lane_r1_m128n256.h"
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

inline constexpr std::size_t kM = 128U;
inline constexpr std::size_t kN = 256U;
inline constexpr std::size_t kK = 512U;
inline constexpr std::size_t kStride = kN + 8U;
inline constexpr std::size_t kByteGuard = 64U;
inline constexpr std::size_t kWordGuard = 8U;
inline constexpr std::uint8_t kByteSentinel = 0xa5U;
inline constexpr std::uint16_t kWordSentinel = 0xadadU;
inline constexpr std::uint16_t kOutputSentinel = 0x7fc1U;

static_assert(kernels::sm87_a4w4_attention_k256_test_plan(kM, kK, 1U)
                  .work_cells == 1U);
static_assert(
    kernels::sm87_a4w4_attention_factorized_lane_r1_scale_capacity_elements(
        kN) == kN);
static_assert(kernels::sm87_a4w4_attention_k256_fixed_panel(
                  kernels::Sm87A4W4AttentionK256Topology::kLinearQkvZ,
                  0U, 2U)
                      .projection == 1U);
static_assert(kernels::sm87_a4w4_attention_k256_fixed_panel(
                  kernels::Sm87A4W4AttentionK256Topology::kFullQkv,
                  8U, 2U)
                      .projection == 2U);

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

  [[nodiscard]] bool allocate(const std::size_t elements) noexcept {
    return elements != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&pointer_),
                      elements * sizeof(T)) == cudaSuccess;
  }

  [[nodiscard]] T* get() const noexcept { return pointer_; }

 private:
  T* pointer_{};
};

template <typename T>
class GuardedDevice final {
 public:
  [[nodiscard]] bool initialize(const std::vector<T>& payload,
                                const std::size_t guard,
                                const T sentinel,
                                const std::string& label) {
    guard_ = guard;
    payload_size_ = payload.size();
    sentinel_ = sentinel;
    initial_.assign(payload_size_ + 2U * guard_, sentinel_);
    std::copy(payload.begin(), payload.end(),
              initial_.begin() + static_cast<std::ptrdiff_t>(guard_));
    return storage_.allocate(initial_.size()) &&
           cuda_ok(cudaMemcpy(storage_.get(), initial_.data(),
                              initial_.size() * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "upload " + label);
  }

  [[nodiscard]] T* payload() const noexcept {
    return storage_.get() + guard_;
  }
  [[nodiscard]] std::size_t payload_size() const noexcept {
    return payload_size_;
  }

  [[nodiscard]] bool download(std::vector<T>* const output,
                              const std::string& label) const {
    if (output == nullptr) {
      return false;
    }
    output->resize(initial_.size());
    return cuda_ok(cudaMemcpy(output->data(), storage_.get(),
                              output->size() * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "download " + label);
  }

  [[nodiscard]] bool unchanged(const std::string& label) const {
    std::vector<T> actual;
    if (!download(&actual, label)) {
      return false;
    }
    if (actual == initial_) {
      return true;
    }
    const auto mismatch =
        std::mismatch(initial_.begin(), initial_.end(), actual.begin());
    std::cerr << label << " changed at "
              << std::distance(initial_.begin(), mismatch.first) << '\n';
    return false;
  }

  [[nodiscard]] bool guards_intact(const std::vector<T>& actual,
                                   const std::string& label) const {
    const auto payload_begin =
        actual.begin() + static_cast<std::ptrdiff_t>(guard_);
    const auto payload_end =
        payload_begin + static_cast<std::ptrdiff_t>(payload_size_);
    const bool prefix = std::all_of(
        actual.begin(), payload_begin,
        [&](const T value) { return value == sentinel_; });
    const bool suffix = std::all_of(
        payload_end, actual.end(),
        [&](const T value) { return value == sentinel_; });
    if (!prefix || !suffix) {
      std::cerr << label << " guard changed\n";
    }
    return prefix && suffix;
  }

  [[nodiscard]] std::size_t guard() const noexcept { return guard_; }

 private:
  DeviceBuffer<T> storage_;
  std::vector<T> initial_;
  std::size_t guard_{};
  std::size_t payload_size_{};
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

[[nodiscard]] std::int8_t signed_code(const std::size_t outer,
                                      const std::size_t inner,
                                      const std::uint32_t salt) noexcept {
  std::uint32_t value =
      static_cast<std::uint32_t>(outer * 0x9e3779b9U) ^
      static_cast<std::uint32_t>(inner * 0x85ebca6bU) ^ salt;
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  return static_cast<std::int8_t>(static_cast<int>(value % 15U) - 7);
}

[[nodiscard]] std::vector<std::uint8_t> make_codes(
    const std::size_t outer_count, const std::uint32_t salt) {
  const std::size_t physical_groups = kK / 64U;
  std::vector<std::uint8_t> result(
      kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(
          outer_count, kK));
  for (std::size_t outer = 0U; outer < outer_count; ++outer) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = 64U * group + 2U * byte;
        result[kernels::sm87_a4w4_attention_k256_packed_offset(
            outer, group, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                signed_code(outer, inner, salt),
                signed_code(outer, inner + 1U, salt));
      }
    }
  }
  return result;
}

[[nodiscard]] std::int8_t unpack(
    const std::vector<std::uint8_t>& packed,
    const std::size_t outer,
    const std::size_t inner) noexcept {
  const std::size_t physical_groups = kK / 64U;
  const std::size_t group = inner / 64U;
  const std::size_t byte = (inner % 64U) / 2U;
  return kernels::sm87_a4w4_unpack_signed(
      packed[kernels::sm87_a4w4_attention_k256_packed_offset(
          outer, group, byte, physical_groups)],
      inner);
}

[[nodiscard]] std::uint16_t oracle(
    const std::vector<std::uint8_t>& a,
    const std::vector<std::uint8_t>& b,
    const std::vector<std::uint16_t>& a_scales,
    const std::vector<std::uint16_t>& b_scales,
    const std::size_t m,
    const std::size_t n) noexcept {
  std::int32_t sum = 0;
  for (std::size_t k = 0U; k < kK; ++k) {
    sum += static_cast<std::int32_t>(unpack(a, m, k)) *
           static_cast<std::int32_t>(unpack(b, n, k));
  }
  const float a_scale = decode_bf16(
      a_scales[kernels::
                   sm87_a4w4_attention_factorized_lane_r1_scale_offset(m)]);
  const float b_scale = decode_bf16(
      b_scales[kernels::
                   sm87_a4w4_attention_factorized_lane_r1_scale_offset(n)]);
  // Power-of-two BF16 scales make both round-to-nearest multiplications
  // exact, so this is a bit-for-bit oracle for the CUDA epilogue.
  const float scale = a_scale * b_scale;
  return encode_bf16(static_cast<float>(sum) * scale);
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
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: requires the 16-SM SM87 target\n";
    return 77;
  }
  return 0;
}

}  // namespace

int main() {
  const int device = target_status();
  if (device != 0) {
    return device;
  }

  kernels::Sm87A4W4AttentionFactorizedLaneR1Resources resources{};
  if (!cuda_ok(
          static_cast<cudaError_t>(
              kernels::
                  query_sm87_a4w4_attention_factorized_lane_r1_m128n256_resources_cuda(
                      &resources)),
          "resource admission")) {
    return 1;
  }
  std::cout << "resources regs=" << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active_cta=" << resources.active_blocks_per_sm << '\n';
  if (resources.registers_per_thread > 255 ||
      resources.dynamic_shared_bytes != 148'224U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 256 ||
      resources.active_blocks_per_sm != 1) {
    std::cerr << "resource query escaped hard gate\n";
    return 1;
  }

  const std::vector<std::uint8_t> host_a = make_codes(kM, 0x1234U);
  const std::vector<std::uint8_t> host_b = make_codes(kN, 0x89abU);
  std::vector<std::uint16_t> host_a_scales(
      kernels::
          sm87_a4w4_attention_factorized_lane_r1_scale_capacity_elements(
              kM));
  std::vector<std::uint16_t> host_b_scales(
      kernels::
          sm87_a4w4_attention_factorized_lane_r1_scale_capacity_elements(
              kN));
  for (std::size_t m = 0U; m < kM; ++m) {
    host_a_scales[
        kernels::sm87_a4w4_attention_factorized_lane_r1_scale_offset(m)] =
        encode_bf16((m & 1U) == 0U ? 0.03125F : 0.015625F);
  }
  for (std::size_t n = 0U; n < kN; ++n) {
    host_b_scales[
        kernels::sm87_a4w4_attention_factorized_lane_r1_scale_offset(n)] =
        encode_bf16((n & 2U) == 0U ? 0.015625F : 0.0078125F);
  }

  GuardedDevice<std::uint8_t> a;
  GuardedDevice<std::uint16_t> a_scales;
  GuardedDevice<std::uint8_t> b;
  GuardedDevice<std::uint16_t> b_scales;
  GuardedDevice<std::uint16_t> output;
  if (!a.initialize(host_a, kByteGuard, kByteSentinel, "A") ||
      !a_scales.initialize(host_a_scales, kWordGuard, kWordSentinel,
                           "A scales") ||
      !b.initialize(host_b, kByteGuard, kByteSentinel, "B") ||
      !b_scales.initialize(host_b_scales, kWordGuard, kWordSentinel,
                           "B scales") ||
      !output.initialize(
          std::vector<std::uint16_t>(kM * kStride, kOutputSentinel),
          kWordGuard, kOutputSentinel, "output")) {
    return 1;
  }

  cudaStream_t stream{};
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create stream")) {
    return 1;
  }
  const kernels::Sm87A4W4AttentionFactorizedLaneR1ProjectionView view{
      b.payload(), b.payload_size(), b_scales.payload(),
      b_scales.payload_size(), kN, output.payload(), kStride,
      output.payload_size()};
  const int launch_status =
      kernels::
          launch_sm87_a4w4_attention_factorized_lane_r1_m128n256_test_bf16_cuda(
              a.payload(), a.payload_size(), a_scales.payload(),
              a_scales.payload_size(), kM, kK, &view, 1U,
              kernels::Sm87A4W4AttentionFactorizedLaneR1EpilogueMode::
                  kTopologyBf16,
              16U, stream);
  bool ok = cuda_ok(static_cast<cudaError_t>(launch_status), "launch") &&
            cuda_ok(cudaStreamSynchronize(stream), "stream sync");

  cudaGraph_t graph{};
  cudaGraphExec_t graph_exec{};
  if (ok) {
    ok = cuda_ok(cudaStreamBeginCapture(
                     stream, cudaStreamCaptureModeThreadLocal),
                 "begin graph capture");
  }
  if (ok) {
    const int capture_status =
        kernels::
            launch_sm87_a4w4_attention_factorized_lane_r1_m128n256_test_bf16_cuda(
                a.payload(), a.payload_size(), a_scales.payload(),
                a_scales.payload_size(), kM, kK, &view, 1U,
                kernels::Sm87A4W4AttentionFactorizedLaneR1EpilogueMode::
                    kTopologyBf16,
                16U, stream);
    ok = cuda_ok(static_cast<cudaError_t>(capture_status),
                 "capture launch") &&
         cuda_ok(cudaStreamEndCapture(stream, &graph),
                 "end graph capture") &&
         cuda_ok(cudaGraphInstantiate(&graph_exec, graph, nullptr,
                                      nullptr, 0U),
                 "instantiate graph") &&
         cuda_ok(cudaGraphLaunch(graph_exec, stream), "graph replay") &&
         cuda_ok(cudaStreamSynchronize(stream), "graph replay sync");
  }

  std::vector<std::uint16_t> actual;
  if (ok) {
    ok = output.download(&actual, "output") &&
         output.guards_intact(actual, "output");
  }
  if (ok) {
    for (std::size_t m = 0U; m < kM && ok; ++m) {
      for (std::size_t n = 0U; n < kStride; ++n) {
        const std::uint16_t observed =
            actual[output.guard() + m * kStride + n];
        const std::uint16_t expected =
            n < kN ? oracle(host_a, host_b, host_a_scales,
                            host_b_scales, m, n)
                   : kOutputSentinel;
        if (observed != expected) {
          std::cerr << "bit mismatch at (" << m << ',' << n
                    << "): expected=0x" << std::hex << expected
                    << " actual=0x" << observed << std::dec << '\n';
          ok = false;
          break;
        }
      }
    }
  }
  if (ok) {
    ok = a.unchanged("A") && a_scales.unchanged("A scales") &&
         b.unchanged("B") && b_scales.unchanged("B scales");
  }

  const int invalid_mode_status =
      kernels::
          launch_sm87_a4w4_attention_factorized_lane_r1_m128n256_test_bf16_cuda(
              a.payload(), a.payload_size(), a_scales.payload(),
              a_scales.payload_size(), kM, kK, &view, 1U,
              static_cast<
                  kernels::Sm87A4W4AttentionFactorizedLaneR1EpilogueMode>(
                  0xffU),
              16U, stream);
  if (invalid_mode_status != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "unknown epilogue mode was not rejected\n";
    ok = false;
  }
  if (graph_exec != nullptr) {
    (void)cudaGraphExecDestroy(graph_exec);
  }
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }
  (void)cudaStreamDestroy(stream);
  if (!ok) {
    return 1;
  }
  std::cout << "PASS: M128N256 K512 full-K S32 is bit-exact; topology "
               "padding, guards, immutable inputs, graph replay, explicit "
               "epilogue, and resource admission hold\n";
  return 0;
}
