#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"
#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256_a_exchange_b4.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

inline constexpr std::size_t kLogicalTokens = 1'853U;
inline constexpr std::size_t kLaunchTokens = 1'920U;
inline constexpr std::size_t kByteGuard = 16U;
inline constexpr std::size_t kWordGuard = 8U;
inline constexpr std::uint8_t kByteSentinel = 0xa5U;
inline constexpr std::uint16_t kWordSentinel = 0x6b6bU;
inline constexpr std::uint16_t kOutputSentinel = 0x7fc1U;

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const char* const operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << " failed: " << cudaGetErrorString(status)
            << '\n';
  return false;
}

template <class T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { cudaFree(pointer_); }

  [[nodiscard]] bool allocate(const std::size_t count) {
    count_ = count;
    return cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                              count * sizeof(T)),
                   "cudaMalloc");
  }

  [[nodiscard]] T* get() noexcept { return pointer_; }
  [[nodiscard]] const T* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  T* pointer_{};
  std::size_t count_{};
};

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = float_bits(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] int signed_code(const std::uint64_t seed) noexcept {
  return static_cast<int>(seed % 15U) - 7;
}

[[nodiscard]] std::vector<std::uint8_t> make_a_codes(
    const std::size_t k) {
  const std::size_t physical_groups = k / 64U;
  std::vector<std::uint8_t> result(
      kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(
          kLaunchTokens, k),
      0U);
  for (std::size_t row = 0U; row < kLaunchTokens; ++row) {
    for (std::size_t physical = 0U; physical < physical_groups;
         ++physical) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        int even = 0;
        int odd = 0;
        if (row < kLogicalTokens) {
          const std::size_t inner = physical * 64U + 2U * byte;
          even = signed_code(row * 131U + inner * 17U + physical * 7U);
          odd = signed_code(row * 193U + (inner + 1U) * 29U +
                            physical * 11U + 3U);
        }
        result[kernels::sm87_a4w4_attention_k256_packed_offset(
            row, physical, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(even, odd);
      }
    }
  }
  return result;
}

[[nodiscard]] std::vector<std::uint16_t> make_a_scales(
    const std::size_t k) {
  const std::size_t groups = k / 256U;
  std::vector<std::uint16_t> result(
      kernels::sm87_a4w4_attention_k256_scale_capacity_elements(
          kLaunchTokens, k),
      encode_bf16(1.0F));
  for (std::size_t row = 0U; row < kLogicalTokens; ++row) {
    for (std::size_t group = 0U; group < groups; ++group) {
      const float scale =
          0.0023F * static_cast<float>(1U + ((row * 3U + group * 5U) % 9U));
      result[kernels::sm87_a4w4_attention_k256_scale_offset(
          row, group, groups)] = encode_bf16(scale);
    }
  }
  return result;
}

[[nodiscard]] std::vector<std::uint8_t> make_b_codes(
    const std::size_t projection,
    const std::size_t n,
    const std::size_t k) {
  const std::size_t physical_groups = k / 64U;
  std::vector<std::uint8_t> result(
      kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(n, k), 0U);
  for (std::size_t row = 0U; row < n; ++row) {
    for (std::size_t physical = 0U; physical < physical_groups;
         ++physical) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = physical * 64U + 2U * byte;
        const int even = signed_code(
            projection * 101U + row * 37U + inner * 13U + physical * 3U);
        const int odd = signed_code(
            projection * 211U + row * 43U + (inner + 1U) * 19U +
            physical * 5U + 1U);
        result[kernels::sm87_a4w4_attention_k256_packed_offset(
            row, physical, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(even, odd);
      }
    }
  }
  return result;
}

[[nodiscard]] std::vector<std::uint16_t> make_b_scales(
    const std::size_t projection,
    const std::size_t n,
    const std::size_t k) {
  const std::size_t groups = k / 256U;
  std::vector<std::uint16_t> result(
      kernels::sm87_a4w4_attention_k256_scale_capacity_elements(n, k),
      0U);
  for (std::size_t row = 0U; row < n; ++row) {
    for (std::size_t group = 0U; group < groups; ++group) {
      const float scale = 0.0017F * static_cast<float>(
          1U + ((projection * 7U + row * 11U + group * 13U) % 11U));
      result[kernels::sm87_a4w4_attention_k256_scale_offset(
          row, group, groups)] = encode_bf16(scale);
    }
  }
  return result;
}

template <class T>
[[nodiscard]] std::vector<T> guarded(const std::vector<T>& payload,
                                     const std::size_t guard,
                                     const T sentinel) {
  std::vector<T> result(2U * guard + payload.size(), sentinel);
  std::copy(payload.begin(), payload.end(),
            result.begin() + static_cast<std::ptrdiff_t>(guard));
  return result;
}

template <class T>
[[nodiscard]] bool upload(DeviceBuffer<T>& destination,
                          const std::vector<T>& source,
                          const char* const label) {
  return destination.allocate(source.size()) &&
         cuda_ok(cudaMemcpy(destination.get(), source.data(),
                            source.size() * sizeof(T),
                            cudaMemcpyHostToDevice),
                 label);
}

template <class T>
[[nodiscard]] bool unchanged(const DeviceBuffer<T>& device,
                             const std::vector<T>& expected,
                             const char* const label) {
  std::vector<T> actual(expected.size());
  if (!cuda_ok(cudaMemcpy(actual.data(), device.get(),
                          actual.size() * sizeof(T),
                          cudaMemcpyDeviceToHost),
               label)) {
    return false;
  }
  return actual == expected;
}

struct ProjectionStorage final {
  std::size_t n{};
  std::size_t stride{};
  std::vector<std::uint8_t> b_codes;
  std::vector<std::uint16_t> b_scales;
  std::vector<std::uint16_t> output_initial;
  DeviceBuffer<std::uint8_t> device_b_codes;
  DeviceBuffer<std::uint16_t> device_b_scales;
  DeviceBuffer<std::uint16_t> candidate_output;
  DeviceBuffer<std::uint16_t> incumbent_output;
};

[[nodiscard]] const char* topology_name(
    const kernels::Sm87A4W4AttentionK256Topology topology) noexcept {
  switch (topology) {
    case kernels::Sm87A4W4AttentionK256Topology::kLinearQkvZ:
      return "LinearQkvZ";
    case kernels::Sm87A4W4AttentionK256Topology::kFullQkv:
      return "FullQkv";
    case kernels::Sm87A4W4AttentionK256Topology::kAttentionO:
      return "AttentionO";
  }
  return "invalid";
}

[[nodiscard]] bool graph_replay_candidate(
    const kernels::Sm87A4W4AttentionK256Topology topology,
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_bytes,
    const std::uint16_t* const a_scales,
    const std::size_t a_scale_elements,
    const kernels::Sm87A4W4AttentionK256ProjectionView* const views,
    const std::size_t projection_count) {
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  if (!cuda_ok(cudaStreamCreate(&stream), "create candidate graph stream") ||
      !cuda_ok(cudaStreamBeginCapture(stream,
                                      cudaStreamCaptureModeThreadLocal),
               "begin candidate capture")) {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    return false;
  }
  const int launch_status =
      kernels::launch_sm87_a4w4_attention_k256_m128n256_a_exchange_b4_bf16_cuda(
          topology, packed_a, packed_a_bytes, a_scales,
          a_scale_elements, kLaunchTokens, views, projection_count,
          stream);
  bool ok = cuda_ok(static_cast<cudaError_t>(launch_status),
                    "capture candidate launch") &&
            cuda_ok(cudaStreamEndCapture(stream, &graph),
                    "end candidate capture") &&
            cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr,
                                         nullptr, 0U),
                    "instantiate candidate graph") &&
            cuda_ok(cudaGraphLaunch(executable, stream),
                    "candidate graph replay 1") &&
            cuda_ok(cudaGraphLaunch(executable, stream),
                    "candidate graph replay 2") &&
            cuda_ok(cudaStreamSynchronize(stream),
                    "sync candidate graph replay");
  if (executable != nullptr) {
    cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) {
    cudaGraphDestroy(graph);
  }
  cudaStreamDestroy(stream);
  return ok;
}

[[nodiscard]] bool run_topology(
    const kernels::Sm87A4W4AttentionK256Topology topology) {
  const char* const label = topology_name(topology);
  const std::size_t k =
      kernels::sm87_a4w4_attention_k256_fixed_input_size(topology);
  const std::size_t projection_count =
      kernels::sm87_a4w4_attention_k256_fixed_projection_count(topology);
  std::vector<std::uint8_t> a_code_payload = make_a_codes(k);
  std::vector<std::uint16_t> a_scale_payload = make_a_scales(k);
  const std::vector<std::uint8_t> a_codes =
      guarded(a_code_payload, kByteGuard, kByteSentinel);
  const std::vector<std::uint16_t> a_scales =
      guarded(a_scale_payload, kWordGuard, kWordSentinel);
  DeviceBuffer<std::uint8_t> device_a_codes;
  DeviceBuffer<std::uint16_t> device_a_scales;
  if (!upload(device_a_codes, a_codes, "upload guarded A codes") ||
      !upload(device_a_scales, a_scales, "upload guarded A scales")) {
    return false;
  }
  const std::uint8_t* const packed_a =
      device_a_codes.get() + kByteGuard;
  const std::uint16_t* const scales_a =
      device_a_scales.get() + kWordGuard;

  std::array<ProjectionStorage, 3U> storage{};
  std::array<kernels::Sm87A4W4AttentionK256ProjectionView, 3U>
      candidate_views{};
  std::array<kernels::Sm87A4W4AttentionK256ProjectionView, 3U>
      incumbent_views{};
  for (std::size_t projection = 0U; projection < projection_count;
       ++projection) {
    auto& item = storage[projection];
    item.n = kernels::sm87_a4w4_attention_k256_fixed_projection_panels(
                 topology, projection) *
             kernels::kSm87A4W4AttentionK256PanelN;
    item.stride = item.n + 2U;
    const auto b_code_payload = make_b_codes(projection, item.n, k);
    const auto b_scale_payload = make_b_scales(projection, item.n, k);
    item.b_codes = guarded(b_code_payload, kByteGuard, kByteSentinel);
    item.b_scales = guarded(b_scale_payload, kWordGuard, kWordSentinel);
    const std::vector<std::uint16_t> output_payload(
        kLaunchTokens * item.stride, kOutputSentinel);
    item.output_initial =
        guarded(output_payload, kWordGuard, kWordSentinel);
    if (!upload(item.device_b_codes, item.b_codes,
                "upload guarded B codes") ||
        !upload(item.device_b_scales, item.b_scales,
                "upload guarded B scales") ||
        !upload(item.candidate_output, item.output_initial,
                "initialize candidate output") ||
        !upload(item.incumbent_output, item.output_initial,
                "initialize incumbent output")) {
      return false;
    }
    const kernels::Sm87A4W4AttentionK256ProjectionView common{
        item.device_b_codes.get() + kByteGuard,
        b_code_payload.size(),
        item.device_b_scales.get() + kWordGuard,
        b_scale_payload.size(),
        item.n,
        nullptr,
        item.stride,
        output_payload.size()};
    candidate_views[projection] = common;
    candidate_views[projection].output_bf16 =
        item.candidate_output.get() + kWordGuard;
    incumbent_views[projection] = common;
    incumbent_views[projection].output_bf16 =
        item.incumbent_output.get() + kWordGuard;
  }

  const int invalid_token =
      kernels::launch_sm87_a4w4_attention_k256_m128n256_a_exchange_b4_bf16_cuda(
          topology, packed_a, a_code_payload.size(), scales_a,
          a_scale_payload.size(), kLogicalTokens, candidate_views.data(),
          projection_count);
  const int invalid_capacity =
      kernels::launch_sm87_a4w4_attention_k256_m128n256_a_exchange_b4_bf16_cuda(
          topology, packed_a, a_code_payload.size() - 1U, scales_a,
          a_scale_payload.size(), kLaunchTokens, candidate_views.data(),
          projection_count);
  if (invalid_token != static_cast<int>(cudaErrorInvalidValue) ||
      invalid_capacity != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << label << ": invalid padding/capacity was accepted\n";
    return false;
  }

  const int incumbent_status =
      kernels::launch_sm87_a4w4_attention_k256_m128n256_bf16_cuda(
          topology, packed_a, a_code_payload.size(), scales_a,
          a_scale_payload.size(), kLaunchTokens, incumbent_views.data(),
          projection_count);
  if (!cuda_ok(static_cast<cudaError_t>(incumbent_status),
               "launch incumbent") ||
      !cuda_ok(cudaDeviceSynchronize(), "sync incumbent") ||
      !graph_replay_candidate(
          topology, packed_a, a_code_payload.size(), scales_a,
          a_scale_payload.size(), candidate_views.data(),
          projection_count)) {
    return false;
  }

  if (!unchanged(device_a_codes, a_codes, "copy immutable A codes") ||
      !unchanged(device_a_scales, a_scales, "copy immutable A scales")) {
    std::cerr << label << ": A payload/guard was modified\n";
    return false;
  }

  for (std::size_t projection = 0U; projection < projection_count;
       ++projection) {
    auto& item = storage[projection];
    if (!unchanged(item.device_b_codes, item.b_codes,
                   "copy immutable B codes") ||
        !unchanged(item.device_b_scales, item.b_scales,
                   "copy immutable B scales")) {
      std::cerr << label << ": B payload/guard was modified at projection "
                << projection << '\n';
      return false;
    }
    std::vector<std::uint16_t> candidate(item.output_initial.size());
    std::vector<std::uint16_t> incumbent(item.output_initial.size());
    if (!cuda_ok(cudaMemcpy(candidate.data(), item.candidate_output.get(),
                            candidate.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost),
                 "copy candidate output") ||
        !cuda_ok(cudaMemcpy(incumbent.data(), item.incumbent_output.get(),
                            incumbent.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost),
                 "copy incumbent output")) {
      return false;
    }
    const std::size_t payload_begin = kWordGuard;
    const std::size_t payload_end =
        payload_begin + kLaunchTokens * item.stride;
    const bool outer_guards =
        std::all_of(candidate.begin(),
                    candidate.begin() +
                        static_cast<std::ptrdiff_t>(payload_begin),
                    [](const std::uint16_t value) {
                      return value == kWordSentinel;
                    }) &&
        std::all_of(candidate.begin() +
                        static_cast<std::ptrdiff_t>(payload_end),
                    candidate.end(), [](const std::uint16_t value) {
                      return value == kWordSentinel;
                    });
    if (!outer_guards) {
      std::cerr << label << ": output outer guard overwritten\n";
      return false;
    }
    for (std::size_t row = 0U; row < kLaunchTokens; ++row) {
      const std::size_t base = payload_begin + row * item.stride;
      for (std::size_t column = 0U; column < item.n; ++column) {
        if (candidate[base + column] != incumbent[base + column]) {
          std::cerr << label << ": BF16 mismatch projection="
                    << projection << " row=" << row
                    << " column=" << column << " candidate=0x"
                    << std::hex << candidate[base + column]
                    << " incumbent=0x" << incumbent[base + column]
                    << std::dec << '\n';
          return false;
        }
        if (row >= kLogicalTokens && candidate[base + column] != 0U) {
          std::cerr << label << ": padded row was not zero\n";
          return false;
        }
      }
      for (std::size_t column = item.n; column < item.stride; ++column) {
        if (candidate[base + column] != kOutputSentinel ||
            incumbent[base + column] != kOutputSentinel) {
          std::cerr << label << ": row guard overwritten\n";
          return false;
        }
      }
    }
  }

  std::cout << label << " P1920/K" << k
            << " bit-exact incumbent/graph/guard check passed\n";
  return true;
}

}  // namespace

int main() {
  kernels::Sm87A4W4AttentionK256Resources resources{};
  const int resource_status =
      kernels::query_sm87_a4w4_attention_k256_m128n256_a_exchange_b4_resources_cuda(
          &resources);
  if (resource_status == static_cast<int>(cudaErrorNotSupported)) {
    std::cerr << "SM87 16-SM target is unavailable\n";
    return 77;
  }
  if (!cuda_ok(static_cast<cudaError_t>(resource_status),
               "candidate resource gate")) {
    return 1;
  }
  if (resources.registers_per_thread > 255 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 149'760U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 256 ||
      resources.active_blocks_per_sm != 1) {
    std::cerr << "candidate resource values violated the hard contract\n";
    return 1;
  }
  std::cout << "resources: regs=" << resources.registers_per_thread
            << " dynamic=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active=" << resources.active_blocks_per_sm << '\n';

  if (!run_topology(
          kernels::Sm87A4W4AttentionK256Topology::kLinearQkvZ) ||
      !run_topology(kernels::Sm87A4W4AttentionK256Topology::kFullQkv) ||
      !run_topology(kernels::Sm87A4W4AttentionK256Topology::kAttentionO)) {
    return 1;
  }
  std::cout << "Attention A-exchange/B4 structural correctness passed\n";
  return 0;
}
