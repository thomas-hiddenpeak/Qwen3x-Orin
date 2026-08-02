#include "q3x/kernels/sm87_a4w4_gateup_k512_m128n512_fused_quantize.h"
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
inline constexpr std::size_t kElementGuard = 64U;
inline constexpr std::uint8_t kPackedSentinel = 0xa5U;
inline constexpr std::uint16_t kScaleSentinel = 0xbeefU;
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
  [[nodiscard]] cudaGraphExec_t get() const noexcept {
    return executable_;
  }

 private:
  cudaGraphExec_t executable_{};
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

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
      properties.sharedMemPerBlockOptin <
          kernels::
              kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes) {
    std::cout << "SKIP: requires 16-SM SM87 with 165376 B opt-in shared\n";
    return 77;
  }
  return 0;
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
          kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(n,
                                                                      k)),
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(n,
                                                                      k))};

  // Padded activation rows are intentionally nonzero.  Both the fused
  // producer and the canonical quantizer must nevertheless publish zero
  // codes and BF16-one scales for rows beyond logical M.
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
          encode_bf16(
              0.0021F *
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

// Independent scalar Gate+Up reference.  It consumes only the canonical-v1
// physical payload and exact K512 arithmetic contract; it shares no MMA,
// shared-memory pipeline, warp ownership, or fused quantization code.
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
[[nodiscard]] bool copy_to_device(DeviceBuffer<T>& destination,
                                  const std::vector<T>& source,
                                  const std::string& label) {
  return destination.allocate(source.size()) &&
         cuda_ok(cudaMemcpy(destination.get(), source.data(),
                            source.size() * sizeof(T),
                            cudaMemcpyHostToDevice),
                 label);
}

template <typename T>
[[nodiscard]] bool guards_hold(const std::vector<T>& storage,
                               const std::size_t guard,
                               const std::size_t payload,
                               const T sentinel) {
  return std::all_of(storage.begin(),
                     storage.begin() + static_cast<std::ptrdiff_t>(guard),
                     [sentinel](const T value) {
                       return value == sentinel;
                     }) &&
         std::all_of(
             storage.begin() +
                 static_cast<std::ptrdiff_t>(guard + payload),
             storage.end(), [sentinel](const T value) {
               return value == sentinel;
             });
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
  std::cerr << label << ": first mismatch at "
            << std::distance(expected.begin(), mismatch.first)
            << ", expected=0x" << std::hex
            << static_cast<unsigned long long>(*mismatch.first)
            << ", actual=0x"
            << static_cast<unsigned long long>(*mismatch.second)
            << std::dec << '\n';
  return false;
}

[[nodiscard]] bool canonical_tail_holds(
    const std::vector<std::uint8_t>& packed_storage,
    const std::vector<std::uint16_t>& scale_storage,
    const std::size_t logical_m,
    const std::size_t launch_m,
    const std::size_t n) {
  const std::size_t physical_groups = n / 64U;
  const std::size_t k512_groups = n / 512U;
  for (std::size_t row = logical_m; row < launch_m; ++row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t offset =
            kernels::sm87_a4w4_consumer_packed_offset(
                row, group, byte, physical_groups);
        if (packed_storage[kByteGuard + offset] != 0U) {
          std::cerr << "tail row published a nonzero A4 code\n";
          return false;
        }
      }
    }
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      const std::size_t offset =
          kernels::sm87_a4w4_prefill_k512_scale_offset(
              row, group, k512_groups);
      if (scale_storage[kElementGuard + offset] != encode_bf16(1.0F)) {
        std::cerr << "tail row published a non-unit BF16 scale\n";
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool capture_and_replay(
    const Payload& payload,
    DeviceBuffer<std::uint8_t>& a,
    DeviceBuffer<std::uint16_t>& a_scales,
    DeviceBuffer<std::uint8_t>& gate,
    DeviceBuffer<std::uint16_t>& gate_scales,
    DeviceBuffer<std::uint8_t>& up,
    DeviceBuffer<std::uint16_t>& up_scales,
    const std::size_t logical_m,
    const std::size_t launch_m,
    const std::size_t n,
    const std::size_t k,
    std::uint8_t* const packed,
    const std::size_t packed_bytes,
    std::uint16_t* const scales,
    const std::size_t scale_elements,
    const unsigned int maximum_ctas,
    const cudaStream_t stream) {
  GraphHandle graph;
  GraphExecHandle executable;
  if (!cuda_ok(cudaStreamBeginCapture(
                   stream, cudaStreamCaptureModeThreadLocal),
               "begin fused-quantize graph capture")) {
    return false;
  }
  const int captured =
      kernels::
          launch_sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_cuda(
              a.get(), payload.a.size(), a_scales.get(),
              payload.a_scales.size(), gate.get(), payload.gate.size(),
              gate_scales.get(), payload.gate_scales.size(), up.get(),
              payload.up.size(), up_scales.get(), payload.up_scales.size(),
              logical_m, launch_m, n, k, kClipRatio, packed, packed_bytes,
              scales, scale_elements, maximum_ctas, stream);
  const cudaError_t end_status = cudaStreamEndCapture(stream, graph.output());
  if (!launch_ok(captured, "capture fused-quantize launch") ||
      !cuda_ok(end_status, "end fused-quantize graph capture")) {
    return false;
  }
  std::size_t node_count = 0U;
  if (!cuda_ok(cudaGraphGetNodes(graph.get(), nullptr, &node_count),
               "count fused-quantize graph nodes") ||
      node_count != 1U ||
      !cuda_ok(cudaGraphInstantiate(executable.output(), graph.get(),
                                    nullptr, nullptr, 0U),
               "instantiate fused-quantize graph") ||
      !cuda_ok(cudaGraphLaunch(executable.get(), stream),
               "fused-quantize graph replay one") ||
      !cuda_ok(cudaGraphLaunch(executable.get(), stream),
               "fused-quantize graph replay two") ||
      !cuda_ok(cudaStreamSynchronize(stream),
               "synchronize fused-quantize graph")) {
    if (node_count != 1U) {
      std::cerr << "fused-quantize graph nodes=" << node_count
                << ", expected one\n";
    }
    return false;
  }
  return true;
}

[[nodiscard]] bool run_case(const std::string& label,
                            const std::size_t logical_m,
                            const std::size_t launch_m,
                            const std::size_t n,
                            const std::size_t k,
                            const bool graph_replay) {
  const auto plan =
      kernels::sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_plan(
          logical_m, launch_m, n, k, 1U);
  if (plan.launch_ctas != 1U) {
    std::cerr << label << ": invalid test plan\n";
    return false;
  }
  const Payload payload = make_payload(launch_m, n, k);
  const std::size_t packed_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(launch_m, n);
  const std::size_t scale_elements =
      kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(launch_m, n);
  const std::size_t bf16_elements = logical_m * n;
  std::vector<std::uint8_t> packed_initial(
      2U * kByteGuard + packed_bytes, kPackedSentinel);
  std::vector<std::uint16_t> scale_initial(
      2U * kElementGuard + scale_elements, kScaleSentinel);
  std::vector<std::uint16_t> bf16_initial(
      2U * kElementGuard + bf16_elements, kBf16Sentinel);

  DeviceBuffer<std::uint8_t> a;
  DeviceBuffer<std::uint16_t> a_scales;
  DeviceBuffer<std::uint8_t> gate;
  DeviceBuffer<std::uint16_t> gate_scales;
  DeviceBuffer<std::uint8_t> up;
  DeviceBuffer<std::uint16_t> up_scales;
  DeviceBuffer<std::uint8_t> candidate_packed_storage;
  DeviceBuffer<std::uint16_t> candidate_scale_storage;
  DeviceBuffer<std::uint8_t> reference_packed_storage;
  DeviceBuffer<std::uint16_t> reference_scale_storage;
  DeviceBuffer<std::uint16_t> reference_bf16_storage;
  if (!copy_to_device(a, payload.a, label + " copy A") ||
      !copy_to_device(a_scales, payload.a_scales,
                      label + " copy A scales") ||
      !copy_to_device(gate, payload.gate, label + " copy Gate") ||
      !copy_to_device(gate_scales, payload.gate_scales,
                      label + " copy Gate scales") ||
      !copy_to_device(up, payload.up, label + " copy Up") ||
      !copy_to_device(up_scales, payload.up_scales,
                      label + " copy Up scales") ||
      !copy_to_device(candidate_packed_storage, packed_initial,
                      label + " initialize candidate codes") ||
      !copy_to_device(candidate_scale_storage, scale_initial,
                      label + " initialize candidate scales") ||
      !copy_to_device(reference_packed_storage, packed_initial,
                      label + " initialize reference codes") ||
      !copy_to_device(reference_scale_storage, scale_initial,
                      label + " initialize reference scales") ||
      !copy_to_device(reference_bf16_storage, bf16_initial,
                      label + " initialize scalar BF16")) {
    std::cerr << label << ": device allocation/copy failed\n";
    return false;
  }

  std::uint8_t* const candidate_packed =
      candidate_packed_storage.get() + kByteGuard;
  std::uint16_t* const candidate_scales =
      candidate_scale_storage.get() + kElementGuard;
  std::uint8_t* const reference_packed =
      reference_packed_storage.get() + kByteGuard;
  std::uint16_t* const reference_scales =
      reference_scale_storage.get() + kElementGuard;
  std::uint16_t* const reference_bf16 =
      reference_bf16_storage.get() + kElementGuard;

  StreamHandle stream;
  if (!stream.create()) {
    std::cerr << label << ": nondefault stream creation failed\n";
    return false;
  }
  const auto launch_candidate =
      [&](const std::size_t gate_capacity,
          std::uint8_t* const output,
          const std::size_t output_capacity,
          std::uint16_t* const output_scales,
          const std::size_t output_scale_capacity) noexcept {
        return kernels::
            launch_sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_cuda(
                a.get(), payload.a.size(), a_scales.get(),
                payload.a_scales.size(), gate.get(), gate_capacity,
                gate_scales.get(), payload.gate_scales.size(), up.get(),
                payload.up.size(), up_scales.get(),
                payload.up_scales.size(), logical_m, launch_m, n, k,
                kClipRatio, output, output_capacity, output_scales,
                output_scale_capacity, 1U, stream.get());
      };

  if (!expect_invalid(
          launch_candidate(payload.gate.size() - 1U, candidate_packed,
                           packed_bytes, candidate_scales, scale_elements),
          label + " short Gate capacity") ||
      !expect_invalid(
          launch_candidate(payload.gate.size(), candidate_packed,
                           packed_bytes - 1U, candidate_scales,
                           scale_elements),
          label + " short packed capacity") ||
      !expect_invalid(
          launch_candidate(payload.gate.size(), candidate_packed,
                           packed_bytes, candidate_scales,
                           scale_elements - 1U),
          label + " short scale capacity") ||
      !expect_invalid(
          launch_candidate(payload.gate.size(), a.get(), packed_bytes,
                           candidate_scales, scale_elements),
          label + " packed-output/A alias") ||
      !expect_invalid(
          launch_candidate(
              payload.gate.size(), candidate_packed, packed_bytes,
              reinterpret_cast<std::uint16_t*>(candidate_packed),
              scale_elements),
          label + " code/scale alias")) {
    return false;
  }

  if (!launch_ok(
          launch_candidate(payload.gate.size(), candidate_packed,
                           packed_bytes, candidate_scales, scale_elements),
          label + " fused candidate on nondefault stream")) {
    return false;
  }
  constexpr unsigned int threads = 128U;
  const unsigned int output_count =
      static_cast<unsigned int>(logical_m * n);
  const unsigned int blocks = (output_count + threads - 1U) / threads;
  gateup_k512_scalar_reference<<<blocks, threads, 0U, stream.get()>>>(
      a.get(), a_scales.get(), gate.get(), gate_scales.get(), up.get(),
      up_scales.get(), static_cast<unsigned int>(logical_m),
      static_cast<unsigned int>(n),
      static_cast<unsigned int>(plan.input_k512_groups),
      static_cast<unsigned int>(plan.input_physical_k64_groups),
      reference_bf16);
  if (!cuda_ok(cudaPeekAtLastError(), label + " scalar reference launch") ||
      !launch_ok(
          kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
              reference_bf16, n, logical_m, launch_m, n, kClipRatio,
              reference_packed, packed_bytes, reference_scales,
              scale_elements, stream.get()),
          label + " canonical standalone K512 quantizer") ||
      !cuda_ok(cudaStreamSynchronize(stream.get()),
               label + " synchronize nondefault stream")) {
    return false;
  }

  std::vector<std::uint8_t> candidate_packed_host(packed_initial.size());
  std::vector<std::uint16_t> candidate_scale_host(scale_initial.size());
  std::vector<std::uint8_t> reference_packed_host(packed_initial.size());
  std::vector<std::uint16_t> reference_scale_host(scale_initial.size());
  std::vector<std::uint16_t> reference_bf16_host(bf16_initial.size());
  const auto copy_results = [&]() {
    return cuda_ok(
               cudaMemcpy(candidate_packed_host.data(),
                          candidate_packed_storage.get(),
                          candidate_packed_host.size(),
                          cudaMemcpyDeviceToHost),
               label + " copy candidate codes") &&
           cuda_ok(
               cudaMemcpy(candidate_scale_host.data(),
                          candidate_scale_storage.get(),
                          candidate_scale_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               label + " copy candidate scales") &&
           cuda_ok(
               cudaMemcpy(reference_packed_host.data(),
                          reference_packed_storage.get(),
                          reference_packed_host.size(),
                          cudaMemcpyDeviceToHost),
               label + " copy reference codes") &&
           cuda_ok(
               cudaMemcpy(reference_scale_host.data(),
                          reference_scale_storage.get(),
                          reference_scale_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               label + " copy reference scales") &&
           cuda_ok(
               cudaMemcpy(reference_bf16_host.data(),
                          reference_bf16_storage.get(),
                          reference_bf16_host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               label + " copy scalar BF16 storage");
  };
  const auto verify_results = [&]() {
    return guards_hold(candidate_packed_host, kByteGuard, packed_bytes,
                       kPackedSentinel) &&
           guards_hold(candidate_scale_host, kElementGuard, scale_elements,
                       kScaleSentinel) &&
           guards_hold(reference_packed_host, kByteGuard, packed_bytes,
                       kPackedSentinel) &&
           guards_hold(reference_scale_host, kElementGuard, scale_elements,
                       kScaleSentinel) &&
           guards_hold(reference_bf16_host, kElementGuard, bf16_elements,
                       kBf16Sentinel) &&
           compare(reference_packed_host, candidate_packed_host,
                   label + " bit-exact A4 codes") &&
           compare(reference_scale_host, candidate_scale_host,
                   label + " bit-exact BF16 scales") &&
           canonical_tail_holds(reference_packed_host,
                                reference_scale_host, logical_m, launch_m,
                                n);
  };
  if (!copy_results() || !verify_results()) {
    std::cerr << label << ": direct result/canary verification failed\n";
    return false;
  }

  if (graph_replay) {
    if (!cuda_ok(
            cudaMemcpyAsync(candidate_packed_storage.get(),
                            packed_initial.data(), packed_initial.size(),
                            cudaMemcpyHostToDevice, stream.get()),
            label + " reset graph codes") ||
        !cuda_ok(
            cudaMemcpyAsync(candidate_scale_storage.get(),
                            scale_initial.data(),
                            scale_initial.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice, stream.get()),
            label + " reset graph scales") ||
        !cuda_ok(cudaStreamSynchronize(stream.get()),
                 label + " synchronize graph reset") ||
        !capture_and_replay(
            payload, a, a_scales, gate, gate_scales, up, up_scales,
            logical_m, launch_m, n, k, candidate_packed, packed_bytes,
            candidate_scales, scale_elements, 1U, stream.get()) ||
        !copy_results() || !verify_results()) {
      std::cerr << label << ": graph replay verification failed\n";
      return false;
    }
  }

  std::cout << label << " bit-exact: logicalM=" << logical_m
            << " launchM=" << launch_m << " N=" << n << " K=" << k
            << " graph_replay=" << (graph_replay ? "yes" : "no")
            << '\n';
  return true;
}

}  // namespace

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }
  kernels::Sm87A4W4GateUpK512M128N512FusedQuantizeResources resources{};
  if (!launch_ok(
          kernels::
              query_sm87_a4w4_gateup_k512_m128n512_fused_quantize_resources_cuda(
                  &resources),
          "query M128N512 fused-quantize resources") ||
      resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 128 ||
      resources.dynamic_shared_bytes != 165'376U ||
      resources.local_bytes != 0U || resources.active_blocks_per_sm < 1) {
    std::cerr << "M128N512 fused-quantize resource gate failed\n";
    return 1;
  }
  if (!run_case("M128/N512/K512", 128U, 128U, 512U, 512U, true) ||
      !run_case("M117/P128/N512/K1536 tail", 117U, 128U, 512U,
                1'536U, false) ||
      !run_case("M129/P256/N1024/K512 persistent offsets", 129U,
                256U, 1'024U, 512U, false)) {
    return 1;
  }
  std::cout << "M128N512 fused-quantize scalar+canonical correctness passed"
            << " (regs=" << resources.registers_per_thread
            << ", shared=" << resources.dynamic_shared_bytes
            << ", local=" << resources.local_bytes
            << ", active=" << resources.active_blocks_per_sm << ")\n";
  return 0;
}
