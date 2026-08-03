#include "q3x/kernels/sm87_a4w4_gateup_k512_m64n8_paired_warp_register_pipeline.h"
#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"
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
inline constexpr std::uint16_t kWordSentinel = 0xbeefU;

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
                                const std::size_t guard,
                                const T sentinel,
                                const std::string& label) {
    (void)label;
    guard_ = guard;
    payload_count_ = payload.size();
    initial_.assign(payload_count_ + 2U * guard_, sentinel);
    std::copy(payload.begin(), payload.end(),
              initial_.begin() + static_cast<std::ptrdiff_t>(guard_));
    return device_.allocate(initial_.size()) &&
           cudaMemcpy(device_.get(), initial_.data(),
                      initial_.size() * sizeof(T),
                      cudaMemcpyHostToDevice) == cudaSuccess;
  }

  [[nodiscard]] T* payload() const noexcept {
    return device_.get() + guard_;
  }
  [[nodiscard]] std::size_t payload_count() const noexcept {
    return payload_count_;
  }
  [[nodiscard]] T* storage() const noexcept { return device_.get(); }
  [[nodiscard]] const std::vector<T>& initial() const noexcept {
    return initial_;
  }

  [[nodiscard]] bool reset(const cudaStream_t stream) const noexcept {
    return cudaMemcpyAsync(device_.get(), initial_.data(),
                           initial_.size() * sizeof(T),
                           cudaMemcpyHostToDevice, stream) == cudaSuccess;
  }

  [[nodiscard]] bool copy(std::vector<T>& output,
                          const std::string& label) const {
    output.resize(initial_.size());
    const cudaError_t status =
        cudaMemcpy(output.data(), device_.get(),
                   output.size() * sizeof(T), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) {
      return true;
    }
    std::cerr << label << ": " << cudaGetErrorName(status) << '\n';
    return false;
  }

  [[nodiscard]] bool unchanged(const std::string& label) const {
    std::vector<T> actual;
    if (!copy(actual, label)) {
      return false;
    }
    if (actual == initial_) {
      return true;
    }
    std::cerr << label << " changed\n";
    return false;
  }

 private:
  DeviceBuffer<T> device_;
  std::vector<T> initial_;
  std::size_t guard_{};
  std::size_t payload_count_{};
};

class StreamHandle final {
 public:
  StreamHandle() = default;
  StreamHandle(const StreamHandle&) = delete;
  StreamHandle& operator=(const StreamHandle&) = delete;
  ~StreamHandle() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }
  [[nodiscard]] bool create() noexcept {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) ==
           cudaSuccess;
  }
  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_{};
};

class GraphHandle final {
 public:
  GraphHandle() = default;
  GraphHandle(const GraphHandle&) = delete;
  GraphHandle& operator=(const GraphHandle&) = delete;
  ~GraphHandle() {
    if (graph_ != nullptr) {
      (void)cudaGraphDestroy(graph_);
    }
  }
  [[nodiscard]] cudaGraph_t* output() noexcept { return &graph_; }
  [[nodiscard]] cudaGraph_t get() const noexcept { return graph_; }

 private:
  cudaGraph_t graph_{};
};

class GraphExecHandle final {
 public:
  GraphExecHandle() = default;
  GraphExecHandle(const GraphExecHandle&) = delete;
  GraphExecHandle& operator=(const GraphExecHandle&) = delete;
  ~GraphExecHandle() {
    if (executable_ != nullptr) {
      (void)cudaGraphExecDestroy(executable_);
    }
  }
  [[nodiscard]] cudaGraphExec_t* output() noexcept { return &executable_; }
  [[nodiscard]] cudaGraphExec_t get() const noexcept { return executable_; }

 private:
  cudaGraphExec_t executable_{};
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

[[nodiscard]] bool expect_invalid(const int status,
                                  const std::string& operation) {
  if (status == static_cast<int>(cudaErrorInvalidValue)) {
    return true;
  }
  std::cerr << operation << " did not fail closed, status=" << status
            << '\n';
  return false;
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
      properties.sharedMemPerBlockOptin < 99'584U) {
    std::cout << "SKIP: requires 16-SM SM87 with >=99584 B opt-in shared\n";
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
  return static_cast<std::int8_t>(static_cast<int>(mixed & 15U) - 8);
}

struct Payload final {
  std::vector<std::uint8_t> a;
  std::vector<std::uint16_t> a_scales;
  std::vector<std::uint8_t> gate;
  std::vector<std::uint16_t> gate_scales;
  std::vector<std::uint8_t> up;
  std::vector<std::uint16_t> up_scales;
  std::vector<std::uint8_t> paired_codes;
  std::vector<std::uint16_t> paired_scales;
};

[[nodiscard]] Payload make_payload(const std::size_t launch_m,
                                   const std::size_t n,
                                   const std::size_t k) {
  const std::size_t physical_groups = k / 64U;
  const std::size_t k512_groups = k / 512U;
  Payload result{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(launch_m, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
              launch_m, k)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(n, k)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(n, k)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
              n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_k512_fragment_native_scale_capacity_elements(
              n, k))};

  for (std::size_t row = 0U; row < launch_m; ++row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = group * 64U + 2U * byte;
        result.a[kernels::sm87_a4w4_consumer_packed_offset(
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
            kernels::sm87_a4w4_consumer_packed_offset(
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
      const std::size_t canonical =
          kernels::sm87_a4w4_gateup_down_edge_scale_offset(
              row, group, k512_groups);
      result.gate_scales[canonical] = encode_bf16(
          0.0017F *
          static_cast<float>(7U + (5U * row + 3U * group) % 19U));
      result.up_scales[canonical] = encode_bf16(
          0.0013F *
          static_cast<float>(9U + (7U * row + group) % 23U));
      const std::size_t paired =
          kernels::sm87_a4w4_gateup_k512_fragment_native_scale_pair_offset(
              row, group, k512_groups);
      result.paired_scales[paired] = result.gate_scales[canonical];
      result.paired_scales[paired + 1U] = result.up_scales[canonical];
    }
  }

  for (std::size_t block = 0U; block < n / 64U; ++block) {
    for (std::size_t fragment = 0U; fragment < 8U; ++fragment) {
      const std::size_t fragment_n = block * 64U + fragment * 8U;
      for (std::size_t group = 0U; group < k512_groups; ++group) {
        for (std::size_t phase = 0U; phase < 8U; ++phase) {
          for (std::size_t lane = 0U; lane < 32U; ++lane) {
            const std::size_t row = fragment_n + lane / 4U;
            const std::size_t source0 =
                kernels::sm87_a4w4_consumer_packed_offset(
                    row, group * 8U + phase, 4U * (lane % 4U),
                    physical_groups);
            const std::size_t source1 =
                kernels::sm87_a4w4_consumer_packed_offset(
                    row, group * 8U + phase,
                    16U + 4U * (lane % 4U), physical_groups);
            const std::size_t destination =
                kernels::sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                    fragment_n, group, phase, lane, k512_groups);
            std::memcpy(result.paired_codes.data() + destination,
                        result.gate.data() + source0, 4U);
            std::memcpy(result.paired_codes.data() + destination + 4U,
                        result.gate.data() + source1, 4U);
            std::memcpy(result.paired_codes.data() + destination + 8U,
                        result.up.data() + source0, 4U);
            std::memcpy(result.paired_codes.data() + destination + 12U,
                        result.up.data() + source1, 4U);
          }
        }
      }
    }
  }
  return result;
}

[[nodiscard]] __device__ __forceinline__ float decode_bf16_device(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_device(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ float silu_product_device(
    const float gate, const float up) noexcept {
  if (gate >= 0.0F) {
    return (gate / (1.0F + expf(-gate))) * up;
  }
  const float exponential = expf(gate);
  return (gate * exponential / (1.0F + exponential)) * up;
}

// Independent scalar oracle: canonical-v1 Gate/Up codes and scales only.
// It shares no paired-v2 layout, MMA ownership, fused edge, or
// quantizer implementation with the candidate.
__global__ void gateup_k512_scalar_reference(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_scales,
    const std::uint8_t* const packed_gate,
    const std::uint16_t* const gate_scales,
    const std::uint8_t* const packed_up,
    const std::uint16_t* const up_scales,
    const unsigned int logical_m,
    const unsigned int n,
    const unsigned int k512_groups,
    const unsigned int physical_k64_groups,
    std::uint16_t* const output) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int output_count = logical_m * n;
  if (linear >= output_count) {
    return;
  }
  const unsigned int m = linear / n;
  const unsigned int output_n = linear - m * n;
  float gate_accumulator = 0.0F;
  float up_accumulator = 0.0F;
  for (unsigned int group = 0U; group < k512_groups; ++group) {
    std::int32_t gate_partial = 0;
    std::int32_t up_partial = 0;
    for (unsigned int inner = 0U; inner < 512U; ++inner) {
      const unsigned int logical_k = group * 512U + inner;
      const unsigned int physical_group = logical_k / 64U;
      const unsigned int byte = (logical_k % 64U) / 2U;
      const unsigned int nibble = logical_k & 1U;
      const std::int32_t a = kernels::sm87_a4w4_unpack_signed(
          packed_a[kernels::sm87_a4w4_consumer_packed_offset(
              m, physical_group, byte, physical_k64_groups)],
          nibble);
      const std::size_t b_offset =
          kernels::sm87_a4w4_consumer_packed_offset(
              output_n, physical_group, byte, physical_k64_groups);
      gate_partial +=
          a * kernels::sm87_a4w4_unpack_signed(packed_gate[b_offset],
                                               nibble);
      up_partial +=
          a * kernels::sm87_a4w4_unpack_signed(packed_up[b_offset],
                                               nibble);
    }
    const float a_scale = decode_bf16_device(
        a_scales[kernels::sm87_a4w4_gateup_down_edge_scale_offset(
            m, group, k512_groups)]);
    const float gate_scale = decode_bf16_device(
        gate_scales[kernels::sm87_a4w4_gateup_down_edge_scale_offset(
            output_n, group, k512_groups)]);
    const float up_scale = decode_bf16_device(
        up_scales[kernels::sm87_a4w4_gateup_down_edge_scale_offset(
            output_n, group, k512_groups)]);
    gate_accumulator = __fmaf_rn(
        static_cast<float>(gate_partial),
        __fmul_rn(a_scale, gate_scale), gate_accumulator);
    up_accumulator = __fmaf_rn(
        static_cast<float>(up_partial), __fmul_rn(a_scale, up_scale),
        up_accumulator);
  }
  output[static_cast<std::size_t>(m) * n + output_n] =
      encode_bf16_device(
          silu_product_device(gate_accumulator, up_accumulator));
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
            << std::distance(expected.begin(), mismatch.first)
            << ", expected=0x" << std::hex
            << static_cast<unsigned long long>(*mismatch.first)
            << ", actual=0x"
            << static_cast<unsigned long long>(*mismatch.second)
            << std::dec << '\n';
  return false;
}

[[nodiscard]] bool canonical_tail_holds(
    const std::vector<std::uint8_t>& packed,
    const std::vector<std::uint16_t>& scales,
    const std::size_t logical_m,
    const std::size_t launch_m,
    const std::size_t n) {
  const std::size_t physical_groups = n / 64U;
  const std::size_t scale_groups = n / 512U;
  for (std::size_t row = logical_m; row < launch_m; ++row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t offset =
            kernels::sm87_a4w4_gateup_down_edge_packed_offset(
                row, group, byte, physical_groups);
        if (packed[kByteGuard + offset] != 0U) {
          std::cerr << "tail row emitted nonzero A4 code\n";
          return false;
        }
      }
    }
    for (std::size_t group = 0U; group < scale_groups; ++group) {
      const std::size_t offset =
          kernels::sm87_a4w4_gateup_down_edge_scale_offset(
              row, group, scale_groups);
      if (scales[kWordGuard + offset] != encode_bf16(1.0F)) {
        std::cerr << "tail row emitted non-unit BF16 scale\n";
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool capture_and_replay(
    GuardedDevice<std::uint8_t>& a,
    GuardedDevice<std::uint16_t>& a_scales,
    GuardedDevice<std::uint8_t>& paired_codes,
    GuardedDevice<std::uint16_t>& paired_scales,
    const std::size_t logical_m,
    const std::size_t launch_m,
    const std::size_t n,
    const std::size_t k,
    GuardedDevice<std::uint8_t>& packed,
    GuardedDevice<std::uint16_t>& scales,
    const unsigned int maximum_ctas,
    const cudaStream_t stream) {
  GraphHandle graph;
  GraphExecHandle executable;
  if (!cuda_ok(cudaStreamBeginCapture(
                   stream, cudaStreamCaptureModeThreadLocal),
               "begin register-pipeline graph capture")) {
    return false;
  }
  const int captured =
      kernels::
          launch_sm87_a4w4_gateup_k512_m64n8_paired_warp_register_pipeline_test_cuda(
              a.payload(), a.payload_count(), a_scales.payload(),
              a_scales.payload_count(), paired_codes.payload(),
              paired_codes.payload_count(),
              paired_scales.payload(), paired_scales.payload_count(),
              logical_m, launch_m, n, k, kClipRatio, packed.payload(),
              packed.payload_count(), scales.payload(),
              scales.payload_count(), maximum_ctas, stream);
  const cudaError_t end_status =
      cudaStreamEndCapture(stream, graph.output());
  if (!launch_ok(captured, "capture register-pipeline launch") ||
      !cuda_ok(end_status, "end register-pipeline graph capture")) {
    return false;
  }
  std::size_t nodes = 0U;
  if (!cuda_ok(cudaGraphGetNodes(graph.get(), nullptr, &nodes),
               "count register-pipeline graph nodes") ||
      nodes != 1U ||
      !cuda_ok(cudaGraphInstantiate(executable.output(), graph.get(),
                                    nullptr, nullptr, 0U),
               "instantiate register-pipeline graph") ||
      !cuda_ok(cudaGraphLaunch(executable.get(), stream),
               "register-pipeline graph replay one") ||
      !cuda_ok(cudaGraphLaunch(executable.get(), stream),
               "register-pipeline graph replay two") ||
      !cuda_ok(cudaStreamSynchronize(stream),
               "synchronize register-pipeline graph")) {
    std::cerr << "register-pipeline graph nodes=" << nodes << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool run_case(const std::string& label,
                            const std::size_t logical_m,
                            const std::size_t launch_m,
                            const std::size_t n,
                            const std::size_t k,
                            const unsigned int maximum_ctas,
                            const bool graph_replay) {
  const auto plan = kernels::sm87_a4w4_gateup_down_edge_test_plan(
      logical_m, launch_m, n, k, maximum_ctas);
  if (plan.launch_ctas == 0U) {
    std::cerr << label << ": invalid plan\n";
    return false;
  }
  const Payload payload = make_payload(launch_m, n, k);
  const std::size_t packed_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          launch_m, n);
  const std::size_t scale_elements =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          launch_m, n);

  GuardedDevice<std::uint8_t> a;
  GuardedDevice<std::uint16_t> a_scales;
  GuardedDevice<std::uint8_t> gate;
  GuardedDevice<std::uint16_t> gate_scales;
  GuardedDevice<std::uint8_t> up;
  GuardedDevice<std::uint16_t> up_scales;
  GuardedDevice<std::uint8_t> paired_codes;
  GuardedDevice<std::uint16_t> paired_scales;
  GuardedDevice<std::uint8_t> candidate_packed;
  GuardedDevice<std::uint16_t> candidate_scales;
  GuardedDevice<std::uint8_t> reference_packed;
  GuardedDevice<std::uint16_t> reference_scales;
  DeviceBuffer<std::uint16_t> reference_bf16;
  if (!a.initialize(payload.a, kByteGuard, kByteSentinel, label + " A") ||
      !a_scales.initialize(payload.a_scales, kWordGuard, kWordSentinel,
                           label + " A scales") ||
      !gate.initialize(payload.gate, kByteGuard, kByteSentinel,
                       label + " canonical Gate") ||
      !gate_scales.initialize(payload.gate_scales, kWordGuard,
                              kWordSentinel, label + " Gate scales") ||
      !up.initialize(payload.up, kByteGuard, kByteSentinel,
                     label + " canonical Up") ||
      !up_scales.initialize(payload.up_scales, kWordGuard,
                            kWordSentinel, label + " Up scales") ||
      !paired_codes.initialize(
          payload.paired_codes, kByteGuard, kByteSentinel,
          label + " paired-v2 codes") ||
      !paired_scales.initialize(payload.paired_scales, kWordGuard,
                                kWordSentinel, label + " paired scales") ||
      !candidate_packed.initialize(
          std::vector<std::uint8_t>(packed_bytes, kByteSentinel),
          kByteGuard, kByteSentinel, label + " candidate codes") ||
      !candidate_scales.initialize(
          std::vector<std::uint16_t>(scale_elements, kWordSentinel),
          kWordGuard, kWordSentinel, label + " candidate scales") ||
      !reference_packed.initialize(
          std::vector<std::uint8_t>(packed_bytes, kByteSentinel),
          kByteGuard, kByteSentinel, label + " reference codes") ||
      !reference_scales.initialize(
          std::vector<std::uint16_t>(scale_elements, kWordSentinel),
          kWordGuard, kWordSentinel, label + " reference scales") ||
      !reference_bf16.allocate(logical_m * n)) {
    std::cerr << label << ": allocation/copy failed\n";
    return false;
  }

  StreamHandle stream;
  if (!stream.create()) {
    std::cerr << label << ": nondefault stream creation failed\n";
    return false;
  }
  const auto launch_candidate =
      [&](const std::size_t b_capacity,
          std::uint8_t* const output,
          const std::size_t output_capacity,
          std::uint16_t* const output_scales,
          const std::size_t output_scale_capacity) noexcept {
        return kernels::
            launch_sm87_a4w4_gateup_k512_m64n8_paired_warp_register_pipeline_test_cuda(
                a.payload(), a.payload_count(), a_scales.payload(),
                a_scales.payload_count(), paired_codes.payload(),
                b_capacity, paired_scales.payload(),
                paired_scales.payload_count(), logical_m, launch_m, n, k,
                kClipRatio, output, output_capacity, output_scales,
                output_scale_capacity, maximum_ctas, stream.get());
      };
  const auto launch_input_capacities =
      [&](const std::size_t a_capacity,
          const std::size_t a_scale_capacity,
          const std::size_t b_capacity,
          const std::size_t b_scale_capacity) noexcept {
        return kernels::
            launch_sm87_a4w4_gateup_k512_m64n8_paired_warp_register_pipeline_test_cuda(
                a.payload(), a_capacity, a_scales.payload(),
                a_scale_capacity, paired_codes.payload(),
                b_capacity, paired_scales.payload(), b_scale_capacity,
                logical_m, launch_m, n, k, kClipRatio,
                candidate_packed.payload(), packed_bytes,
                candidate_scales.payload(), scale_elements,
                maximum_ctas, stream.get());
      };

  if (!expect_invalid(
          launch_input_capacities(
              a.payload_count() - 1U, a_scales.payload_count(),
              paired_codes.payload_count(),
              paired_scales.payload_count()),
          label + " short A code capacity") ||
      !expect_invalid(
          launch_input_capacities(
              a.payload_count(), a_scales.payload_count() - 1U,
              paired_codes.payload_count(),
              paired_scales.payload_count()),
          label + " short A scale capacity") ||
      !expect_invalid(
          launch_input_capacities(
              a.payload_count(), a_scales.payload_count(),
              paired_codes.payload_count() - 1U,
              paired_scales.payload_count()),
          label + " short paired-code capacity") ||
      !expect_invalid(
          launch_input_capacities(
              a.payload_count(), a_scales.payload_count(),
              paired_codes.payload_count(),
              paired_scales.payload_count() - 1U),
          label + " short paired-scale capacity") ||
      !expect_invalid(
          launch_candidate(paired_codes.payload_count(),
                           candidate_packed.payload(), packed_bytes - 1U,
                           candidate_scales.payload(), scale_elements),
          label + " short output capacity") ||
      !expect_invalid(
          launch_candidate(paired_codes.payload_count(),
                           candidate_packed.payload(), packed_bytes,
                           candidate_scales.payload(), scale_elements - 1U),
          label + " short output scale capacity") ||
      !expect_invalid(
          launch_candidate(paired_codes.payload_count(),
                           a.payload(), packed_bytes,
                           candidate_scales.payload(), scale_elements),
          label + " output/A alias") ||
      !expect_invalid(
          launch_candidate(
              paired_codes.payload_count(),
              candidate_packed.payload(), packed_bytes,
              reinterpret_cast<std::uint16_t*>(candidate_packed.payload()),
              scale_elements),
          label + " output code/scale alias")) {
    return false;
  }
  if (!candidate_packed.unchanged(label + " fail-closed output codes") ||
      !candidate_scales.unchanged(label + " fail-closed output scales")) {
    return false;
  }

  if (!launch_ok(
          launch_candidate(paired_codes.payload_count(),
                           candidate_packed.payload(), packed_bytes,
                           candidate_scales.payload(), scale_elements),
          label + " candidate nondefault-stream launch")) {
    return false;
  }
  constexpr unsigned int threads = 128U;
  const unsigned int output_count =
      static_cast<unsigned int>(logical_m * n);
  const unsigned int blocks = (output_count + threads - 1U) / threads;
  gateup_k512_scalar_reference<<<blocks, threads, 0U, stream.get()>>>(
      a.payload(), a_scales.payload(), gate.payload(),
      gate_scales.payload(), up.payload(), up_scales.payload(),
      static_cast<unsigned int>(logical_m), static_cast<unsigned int>(n),
      static_cast<unsigned int>(plan.input_k512_groups),
      static_cast<unsigned int>(plan.input_physical_k64_groups),
      reference_bf16.get());
  if (!cuda_ok(cudaPeekAtLastError(), label + " scalar oracle launch") ||
      !launch_ok(
          kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
              reference_bf16.get(), n, logical_m, launch_m, n,
              kClipRatio, reference_packed.payload(), packed_bytes,
              reference_scales.payload(), scale_elements, stream.get()),
          label + " canonical standalone quantizer") ||
      !cuda_ok(cudaStreamSynchronize(stream.get()),
               label + " synchronize")) {
    return false;
  }

  std::vector<std::uint8_t> candidate_packed_host;
  std::vector<std::uint16_t> candidate_scales_host;
  std::vector<std::uint8_t> reference_packed_host;
  std::vector<std::uint16_t> reference_scales_host;
  const auto verify = [&]() {
    return candidate_packed.copy(candidate_packed_host,
                                 label + " candidate codes") &&
           candidate_scales.copy(candidate_scales_host,
                                 label + " candidate scales") &&
           reference_packed.copy(reference_packed_host,
                                 label + " reference codes") &&
           reference_scales.copy(reference_scales_host,
                                 label + " reference scales") &&
           compare(reference_packed_host, candidate_packed_host,
                   label + " bit-exact A4 codes") &&
           compare(reference_scales_host, candidate_scales_host,
                   label + " bit-exact BF16 scales") &&
           canonical_tail_holds(candidate_packed_host,
                                candidate_scales_host, logical_m,
                                launch_m, n);
  };
  if (!verify()) {
    return false;
  }

  if (graph_replay) {
    if (!candidate_packed.reset(stream.get()) ||
        !candidate_scales.reset(stream.get()) ||
        !cuda_ok(cudaStreamSynchronize(stream.get()),
                 label + " synchronize graph reset") ||
        !capture_and_replay(
            a, a_scales, paired_codes, paired_scales,
            logical_m, launch_m, n, k, candidate_packed,
            candidate_scales, maximum_ctas, stream.get()) ||
        !verify()) {
      std::cerr << label << ": graph replay verification failed\n";
      return false;
    }
  }

  if (!a.unchanged(label + " A") ||
      !a_scales.unchanged(label + " A scales") ||
      !gate.unchanged(label + " canonical Gate") ||
      !gate_scales.unchanged(label + " Gate scales") ||
      !up.unchanged(label + " canonical Up") ||
      !up_scales.unchanged(label + " Up scales") ||
      !paired_codes.unchanged(label + " paired codes") ||
      !paired_scales.unchanged(label + " paired scales")) {
    return false;
  }
  std::cout << "PASS: " << label << " logicalM=" << logical_m
            << " launchM=" << launch_m << " N=" << n << " K=" << k
            << (graph_replay ? " graphx2" : "") << '\n';
  return true;
}

}  // namespace

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }
  kernels::Sm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineResources resources{};
  if (!launch_ok(
          kernels::
              query_sm87_a4w4_gateup_k512_m64n8_paired_warp_register_pipeline_resources_cuda(
                  &resources),
          "query register-pipeline resources") ||
      resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 128 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 99'584U ||
      resources.configured_dynamic_shared_limit_bytes < 99'584U ||
      resources.device_optin_shared_limit_bytes < 99'584U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 512 ||
      resources.active_blocks_per_sm != 1) {
    std::cerr << "register-pipeline resource gate failed: regs="
              << resources.registers_per_thread
              << " local=" << resources.local_bytes
              << " shared=" << resources.dynamic_shared_bytes
              << " active=" << resources.active_blocks_per_sm << '\n';
    return 1;
  }

  if (!run_case("M64/P128 single-edge K512", 64U, 128U, 512U,
                512U, 1U, true) ||
      !run_case("P513/P640 tail", 513U, 640U, 512U, 512U, 16U,
                false) ||
      !run_case("multi-edge offsets K1536", 1U, 128U, 1'536U,
                1'536U, 3U, false) ||
      !run_case("full-input K5120", 1U, 128U, 512U, 5'120U, 1U,
                false)) {
    return 1;
  }
  std::cout << "register-pipeline scalar/canonical correctness passed"
            << " (regs=" << resources.registers_per_thread
            << ", shared=" << resources.dynamic_shared_bytes
            << ", local=" << resources.local_bytes
            << ", active=" << resources.active_blocks_per_sm << ")\n";
  return 0;
}
