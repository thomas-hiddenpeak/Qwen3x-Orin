#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"
#include "q3x/kernels/sm87_a4w4_factorized_lane_quantize.h"
#include "q3x/kernels/sm87_a4w4_prefill_handoff.h"
#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;
namespace runtime = q3x::runtime;

inline constexpr std::size_t kHidden =
    kernels::kSm87A4W4PrefillHandoffHiddenSize;
inline constexpr std::size_t kWordGuard = 32U;
inline constexpr std::size_t kByteGuard = 64U;
inline constexpr unsigned char kSentinelByte = 0xa5U;
inline constexpr std::uint16_t kWordSentinel = 0xa5a5U;
inline constexpr float kEpsilon = 1.0e-6F;
inline constexpr float kClipRatio = 0.91F;

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const std::string& operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorName(status) << " ("
            << cudaGetErrorString(status) << ")\n";
  return false;
}

[[nodiscard]] int target_device_status() {
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

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
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

  [[nodiscard]] bool allocate(const std::size_t elements) noexcept {
    elements_ = elements;
    return elements != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&data_),
                      elements * sizeof(T)) == cudaSuccess;
  }

  [[nodiscard]] T* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return elements_; }

 private:
  T* data_{};
  std::size_t elements_{};
};

class Stream final {
 public:
  Stream() = default;
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() {
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

template <typename T>
class GuardedDeviceBuffer final {
 public:
  GuardedDeviceBuffer() = default;
  GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
  GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;

  [[nodiscard]] bool allocate(const std::size_t payload_elements,
                              const std::size_t guard_elements) {
    payload_elements_ = payload_elements;
    guard_elements_ = guard_elements;
    return storage_.allocate(payload_elements + 2U * guard_elements) &&
           cuda_ok(cudaMemset(storage_.get(), kSentinelByte,
                              storage_.size() * sizeof(T)),
                   "initialize guarded device buffer");
  }

  [[nodiscard]] T* data() const noexcept {
    return storage_.get() + guard_elements_;
  }
  [[nodiscard]] std::size_t payload_size() const noexcept {
    return payload_elements_;
  }

  [[nodiscard]] bool copy_to_host(std::vector<T>* const output,
                                  const std::string& operation) const {
    output->resize(storage_.size());
    return cuda_ok(cudaMemcpy(output->data(), storage_.get(),
                              storage_.size() * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   operation);
  }

  [[nodiscard]] bool guards_are_untouched(
      const std::vector<T>& host) const noexcept {
    const T sentinel = []() {
      T value{};
      std::memset(&value, kSentinelByte, sizeof(value));
      return value;
    }();
    return std::all_of(host.begin(),
                       host.begin() +
                           static_cast<std::ptrdiff_t>(guard_elements_),
                       [sentinel](const T value) { return value == sentinel; }) &&
           std::all_of(host.begin() + static_cast<std::ptrdiff_t>(
                                          guard_elements_ + payload_elements_),
                       host.end(),
                       [sentinel](const T value) { return value == sentinel; });
  }

  [[nodiscard]] const T* host_payload(
      const std::vector<T>& host) const noexcept {
    return host.data() + guard_elements_;
  }

 private:
  DeviceBuffer<T> storage_;
  std::size_t payload_elements_{};
  std::size_t guard_elements_{};
};

struct CaseSpec final {
  std::string name;
  std::size_t logical_tokens{};
  std::size_t launch_tokens{};
  bool graph_replay{};
};

struct HostInputs final {
  std::vector<std::uint16_t> left;
  std::vector<std::uint16_t> right;
  std::vector<std::uint16_t> centered_weight;
  std::vector<float> inverse_alpha;
};

[[nodiscard]] HostInputs make_inputs(const CaseSpec& spec) {
  HostInputs result{
      std::vector<std::uint16_t>(spec.logical_tokens * kHidden),
      std::vector<std::uint16_t>(spec.logical_tokens * kHidden),
      std::vector<std::uint16_t>(kHidden), std::vector<float>(kHidden)};

  for (std::size_t dimension = 0U; dimension < kHidden; ++dimension) {
    const int centered_weight =
        static_cast<int>((dimension * 29U + 17U) % 257U) - 128;
    result.centered_weight[dimension] =
        encode_bf16(static_cast<float>(centered_weight) * 0.00131F);
    result.inverse_alpha[dimension] =
        0.625F + 0.00625F *
                     static_cast<float>((dimension * 37U + 11U) % 101U);
  }

  // Row zero deliberately stays all-zero. It covers the scale-one contract
  // for a logical row independently of padded-row handling.
  for (std::size_t row = 1U; row < spec.logical_tokens; ++row) {
    for (std::size_t dimension = 0U; dimension < kHidden; ++dimension) {
      const int left_code =
          static_cast<int>((row * 1'009U + dimension * 313U +
                            (row + 3U) * (dimension % 29U)) %
                           2'003U) -
          1'001;
      const int right_code =
          static_cast<int>((row * 619U + dimension * 997U +
                            (row + 5U) * (dimension % 43U)) %
                           1'723U) -
          861;
      result.left[row * kHidden + dimension] = encode_bf16(
          static_cast<float>(left_code) * 0.00211F);
      result.right[row * kHidden + dimension] = encode_bf16(
          static_cast<float>(right_code) * 0.00173F);
    }
  }
  return result;
}

template <typename Launch>
[[nodiscard]] bool launch_warm_and_optional_graph(
    const CaseSpec& spec, const Launch& launch, const cudaStream_t stream,
    const std::string& label) {
  if (!cuda_ok(static_cast<cudaError_t>(launch()), label + " warm launch") ||
      !cuda_ok(cudaStreamSynchronize(stream), label + " warm sync")) {
    return false;
  }
  if (!spec.graph_replay) {
    return true;
  }

  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  bool ok = cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      label + " begin graph capture");
  if (ok) {
    ok = cuda_ok(static_cast<cudaError_t>(launch()),
                 label + " captured launch") &&
         cuda_ok(cudaStreamEndCapture(stream, &graph),
                 label + " end graph capture") &&
         cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr,
                                      0U),
                 label + " instantiate graph") &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 label + " graph replay 1") &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 label + " graph replay 2") &&
         cuda_ok(cudaStreamSynchronize(stream), label + " graph sync");
  }
  if (executable != nullptr) {
    (void)cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }
  return ok;
}

[[nodiscard]] bool copy_inputs(
    const HostInputs& host, DeviceBuffer<std::uint16_t>* const left,
    DeviceBuffer<std::uint16_t>* const right,
    DeviceBuffer<std::uint16_t>* const weight,
    DeviceBuffer<float>* const inverse_alpha) {
  return left->allocate(host.left.size()) && right->allocate(host.right.size()) &&
         weight->allocate(host.centered_weight.size()) &&
         inverse_alpha->allocate(host.inverse_alpha.size()) &&
         cuda_ok(cudaMemcpy(left->get(), host.left.data(),
                            host.left.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "copy left input") &&
         cuda_ok(cudaMemcpy(right->get(), host.right.data(),
                            host.right.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "copy right input") &&
         cuda_ok(cudaMemcpy(weight->get(), host.centered_weight.data(),
                            host.centered_weight.size() *
                                sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "copy centered RMS weight") &&
         cuda_ok(cudaMemcpy(inverse_alpha->get(), host.inverse_alpha.data(),
                            host.inverse_alpha.size() * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "copy authenticated inverse alpha");
}

[[nodiscard]] bool logical_payload_equal(
    const GuardedDeviceBuffer<std::uint16_t>& first_buffer,
    const std::vector<std::uint16_t>& first,
    const GuardedDeviceBuffer<std::uint16_t>& second_buffer,
    const std::vector<std::uint16_t>& second,
    const std::size_t logical_elements) {
  return std::equal(first_buffer.host_payload(first),
                    first_buffer.host_payload(first) + logical_elements,
                    second_buffer.host_payload(second));
}

[[nodiscard]] bool word_suffix_is_sentinel(
    const GuardedDeviceBuffer<std::uint16_t>& buffer,
    const std::vector<std::uint16_t>& host,
    const std::size_t logical_elements) {
  const std::uint16_t* const payload = buffer.host_payload(host);
  return std::all_of(payload + logical_elements,
                     payload + buffer.payload_size(),
                     [](const std::uint16_t value) {
                       return value == kWordSentinel;
                     });
}

[[nodiscard]] bool padded_r1_is_zero_one(
    const std::uint8_t* const packed, const std::uint16_t* const scales,
    const std::size_t logical_tokens, const std::size_t launch_tokens) {
  constexpr std::size_t kPhysicalGroups = kHidden / 64U;
  const std::uint16_t one = encode_bf16(1.0F);
  for (std::size_t row = logical_tokens; row < launch_tokens; ++row) {
    if (scales[kernels::sm87_a4w4_factorized_lane_scale_offset(
            row, 0U, 1U)] != one) {
      return false;
    }
    for (std::size_t group = 0U; group < kPhysicalGroups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        if (packed[kernels::sm87_a4w4_consumer_packed_offset(
                row, group, byte, kPhysicalGroups)] != 0U) {
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool padded_k256_is_zero_one(
    const std::uint8_t* const packed, const std::uint16_t* const scales,
    const std::size_t logical_tokens, const std::size_t launch_tokens) {
  constexpr std::size_t kGroups = kHidden / 256U;
  constexpr std::size_t kPhysicalGroups = kHidden / 64U;
  const std::uint16_t one = encode_bf16(1.0F);
  for (std::size_t row = logical_tokens; row < launch_tokens; ++row) {
    for (std::size_t group = 0U; group < kGroups; ++group) {
      if (scales[kernels::sm87_a4w4_attention_k256_scale_offset(
              row, group, kGroups)] != one) {
        return false;
      }
    }
    for (std::size_t group = 0U; group < kPhysicalGroups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        if (packed[kernels::sm87_a4w4_attention_k256_packed_offset(
                row, group, byte, kPhysicalGroups)] != 0U) {
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool run_attention_to_gate_case(const CaseSpec& spec,
                                               const HostInputs& host) {
  const std::size_t logical_elements = spec.logical_tokens * kHidden;
  const std::size_t launch_elements = spec.launch_tokens * kHidden;
  const std::size_t packed_elements =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(spec.launch_tokens,
                                                        kHidden);
  const std::size_t scale_elements =
      kernels::sm87_a4w4_factorized_lane_scale_capacity_elements(
          spec.launch_tokens, 1U);
  if (packed_elements == 0U || scale_elements == 0U) {
    std::cerr << spec.name << ": invalid R1 capacities\n";
    return false;
  }

  DeviceBuffer<std::uint16_t> left;
  DeviceBuffer<std::uint16_t> right;
  DeviceBuffer<std::uint16_t> weight;
  DeviceBuffer<float> inverse_alpha;
  GuardedDeviceBuffer<std::uint16_t> oracle_residual;
  GuardedDeviceBuffer<std::uint16_t> oracle_normalized;
  GuardedDeviceBuffer<std::uint16_t> candidate_residual;
  GuardedDeviceBuffer<std::uint8_t> oracle_packed;
  GuardedDeviceBuffer<std::uint8_t> candidate_packed;
  GuardedDeviceBuffer<std::uint16_t> oracle_scales;
  GuardedDeviceBuffer<std::uint16_t> candidate_scales;
  Stream stream;
  if (!copy_inputs(host, &left, &right, &weight, &inverse_alpha) ||
      !oracle_residual.allocate(launch_elements, kWordGuard) ||
      !oracle_normalized.allocate(launch_elements, kWordGuard) ||
      !candidate_residual.allocate(launch_elements, kWordGuard) ||
      !oracle_packed.allocate(packed_elements, kByteGuard) ||
      !candidate_packed.allocate(packed_elements, kByteGuard) ||
      !oracle_scales.allocate(scale_elements, kWordGuard) ||
      !candidate_scales.allocate(scale_elements, kWordGuard) ||
      !stream.create()) {
    std::cerr << spec.name << ": attention-to-gate allocation failed\n";
    return false;
  }

  // Independent incumbent chain: residual add, per-token centered RMS, then
  // authenticated lane-count-one factorized quantization.
  if (!cuda_ok(static_cast<cudaError_t>(runtime::launch_residual_add_reference_cuda(
                   left.get(), right.get(), logical_elements,
                   oracle_residual.data(), stream.get())),
               spec.name + " oracle residual add") ||
      !cuda_ok(static_cast<cudaError_t>(
                   runtime::launch_headwise_centered_rms_norm_reference_cuda(
                       oracle_residual.data(), weight.get(),
                       spec.logical_tokens, kHidden, kEpsilon,
                       oracle_normalized.data(), stream.get())),
               spec.name + " oracle centered RMS") ||
      !cuda_ok(static_cast<cudaError_t>(
                   kernels::launch_sm87_a4w4_factorized_lane_quantize_bf16_cuda(
                       oracle_normalized.data(), kHidden, logical_elements,
                       inverse_alpha.get(), kHidden, spec.logical_tokens,
                       spec.launch_tokens, kHidden, 1U, kClipRatio,
                       oracle_packed.data(), packed_elements,
                       oracle_scales.data(), scale_elements, stream.get())),
               spec.name + " oracle R1 quantize")) {
    return false;
  }

  const auto launch_candidate = [&]() noexcept {
    return kernels::
        launch_sm87_a4w4_attention_residual_post_norm_r1_quantize_cuda(
            left.get(), right.get(), weight.get(), inverse_alpha.get(),
            kHidden, spec.logical_tokens, spec.launch_tokens, kEpsilon,
            kClipRatio, candidate_residual.data(), logical_elements,
            candidate_packed.data(), packed_elements, candidate_scales.data(),
            scale_elements, stream.get());
  };
  if (!launch_warm_and_optional_graph(spec, launch_candidate, stream.get(),
                                      spec.name + " attention-to-gate")) {
    return false;
  }

  std::vector<std::uint16_t> host_oracle_residual;
  std::vector<std::uint16_t> host_oracle_normalized;
  std::vector<std::uint16_t> host_candidate_residual;
  std::vector<std::uint8_t> host_oracle_packed;
  std::vector<std::uint8_t> host_candidate_packed;
  std::vector<std::uint16_t> host_oracle_scales;
  std::vector<std::uint16_t> host_candidate_scales;
  if (!oracle_residual.copy_to_host(&host_oracle_residual,
                                    spec.name + " copy oracle residual") ||
      !oracle_normalized.copy_to_host(
          &host_oracle_normalized,
          spec.name + " copy oracle normalized scratch") ||
      !candidate_residual.copy_to_host(
          &host_candidate_residual,
          spec.name + " copy candidate residual") ||
      !oracle_packed.copy_to_host(&host_oracle_packed,
                                  spec.name + " copy oracle R1 codes") ||
      !candidate_packed.copy_to_host(
          &host_candidate_packed,
          spec.name + " copy candidate R1 codes") ||
      !oracle_scales.copy_to_host(&host_oracle_scales,
                                  spec.name + " copy oracle R1 scales") ||
      !candidate_scales.copy_to_host(
          &host_candidate_scales,
          spec.name + " copy candidate R1 scales")) {
    return false;
  }

  const bool guards =
      oracle_residual.guards_are_untouched(host_oracle_residual) &&
      oracle_normalized.guards_are_untouched(host_oracle_normalized) &&
      candidate_residual.guards_are_untouched(host_candidate_residual) &&
      oracle_packed.guards_are_untouched(host_oracle_packed) &&
      candidate_packed.guards_are_untouched(host_candidate_packed) &&
      oracle_scales.guards_are_untouched(host_oracle_scales) &&
      candidate_scales.guards_are_untouched(host_candidate_scales);
  const bool residual_exact = logical_payload_equal(
      oracle_residual, host_oracle_residual, candidate_residual,
      host_candidate_residual, logical_elements);
  const bool packed_exact = std::equal(
      oracle_packed.host_payload(host_oracle_packed),
      oracle_packed.host_payload(host_oracle_packed) + packed_elements,
      candidate_packed.host_payload(host_candidate_packed));
  const bool scales_exact = std::equal(
      oracle_scales.host_payload(host_oracle_scales),
      oracle_scales.host_payload(host_oracle_scales) + scale_elements,
      candidate_scales.host_payload(host_candidate_scales));
  const bool suffix =
      word_suffix_is_sentinel(oracle_residual, host_oracle_residual,
                              logical_elements) &&
      word_suffix_is_sentinel(oracle_normalized, host_oracle_normalized,
                              logical_elements) &&
      word_suffix_is_sentinel(candidate_residual, host_candidate_residual,
                              logical_elements);
  const bool padded =
      padded_r1_is_zero_one(
          candidate_packed.host_payload(host_candidate_packed),
          candidate_scales.host_payload(host_candidate_scales),
          spec.logical_tokens, spec.launch_tokens) &&
      padded_r1_is_zero_one(
          candidate_packed.host_payload(host_candidate_packed),
          candidate_scales.host_payload(host_candidate_scales), 0U, 1U);
  if (!guards || !residual_exact || !packed_exact || !scales_exact ||
      !suffix || !padded) {
    std::cerr << spec.name
              << ": attention-to-gate bit/guard/padding mismatch"
              << " guards=" << guards << " residual=" << residual_exact
              << " packed=" << packed_exact << " scales=" << scales_exact
              << " suffix=" << suffix << " padded=" << padded << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool launch_exact_residual_norm_oracle(
    const std::uint16_t* const left, const std::uint16_t* const right,
    const std::uint16_t* const weight, const std::size_t logical_tokens,
    std::uint16_t* const residual, std::uint16_t* const normalized,
    const cudaStream_t stream, const std::string& label) {
  constexpr std::size_t kMaximumOracleTile = 512U;
  for (std::size_t offset = 0U; offset < logical_tokens;
       offset += kMaximumOracleTile) {
    const std::size_t tile =
        std::min(kMaximumOracleTile, logical_tokens - offset);
    const std::size_t element_offset = offset * kHidden;
    const int status =
        runtime::launch_residual_add_headwise_centered_rms_norm_prefill_5120_cuda(
            left + element_offset, right + element_offset, weight, tile,
            kHidden, kEpsilon, residual + element_offset,
            normalized + element_offset, stream);
    if (!cuda_ok(static_cast<cudaError_t>(status),
                 label + " residual-norm tile")) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool run_mlp_to_attention_case(const CaseSpec& spec,
                                              const HostInputs& host) {
  const std::size_t logical_elements = spec.logical_tokens * kHidden;
  const std::size_t launch_elements = spec.launch_tokens * kHidden;
  const std::size_t packed_elements =
      kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(
          spec.launch_tokens, kHidden);
  const std::size_t scale_elements =
      kernels::sm87_a4w4_attention_k256_scale_capacity_elements(
          spec.launch_tokens, kHidden);
  if (packed_elements == 0U || scale_elements == 0U) {
    std::cerr << spec.name << ": invalid K256 capacities\n";
    return false;
  }

  DeviceBuffer<std::uint16_t> left;
  DeviceBuffer<std::uint16_t> right;
  DeviceBuffer<std::uint16_t> weight;
  DeviceBuffer<float> inverse_alpha;
  GuardedDeviceBuffer<std::uint16_t> oracle_residual;
  GuardedDeviceBuffer<std::uint16_t> oracle_normalized;
  GuardedDeviceBuffer<std::uint16_t> candidate_residual;
  GuardedDeviceBuffer<std::uint16_t> candidate_normalized;
  GuardedDeviceBuffer<std::uint8_t> oracle_packed;
  GuardedDeviceBuffer<std::uint8_t> candidate_packed;
  GuardedDeviceBuffer<std::uint16_t> oracle_scales;
  GuardedDeviceBuffer<std::uint16_t> candidate_scales;
  Stream stream;
  if (!copy_inputs(host, &left, &right, &weight, &inverse_alpha) ||
      !oracle_residual.allocate(launch_elements, kWordGuard) ||
      !oracle_normalized.allocate(launch_elements, kWordGuard) ||
      !candidate_residual.allocate(launch_elements, kWordGuard) ||
      !candidate_normalized.allocate(launch_elements, kWordGuard) ||
      !oracle_packed.allocate(packed_elements, kByteGuard) ||
      !candidate_packed.allocate(packed_elements, kByteGuard) ||
      !oracle_scales.allocate(scale_elements, kWordGuard) ||
      !candidate_scales.allocate(scale_elements, kWordGuard) ||
      !stream.create()) {
    std::cerr << spec.name << ": MLP-to-attention allocation failed\n";
    return false;
  }

  if (!launch_exact_residual_norm_oracle(
          left.get(), right.get(), weight.get(), spec.logical_tokens,
          oracle_residual.data(), oracle_normalized.data(), stream.get(),
          spec.name + " oracle") ||
      !cuda_ok(static_cast<cudaError_t>(
                   kernels::launch_sm87_a4_quantize_bf16_k256_cuda(
                       oracle_normalized.data(), kHidden,
                       spec.logical_tokens, spec.launch_tokens, kHidden,
                       kClipRatio, oracle_packed.data(), packed_elements,
                       oracle_scales.data(), scale_elements, stream.get())),
               spec.name + " oracle K256 quantize")) {
    return false;
  }

  const auto launch_candidate = [&]() noexcept {
    return kernels::
        launch_sm87_a4w4_mlp_residual_next_norm_k256_quantize_cuda(
            left.get(), right.get(), weight.get(), spec.logical_tokens,
            spec.launch_tokens, kEpsilon, kClipRatio,
            candidate_residual.data(), logical_elements,
            candidate_normalized.data(), logical_elements,
            candidate_packed.data(), packed_elements, candidate_scales.data(),
            scale_elements, stream.get());
  };
  if (!launch_warm_and_optional_graph(spec, launch_candidate, stream.get(),
                                      spec.name + " MLP-to-attention")) {
    return false;
  }

  std::vector<std::uint16_t> host_oracle_residual;
  std::vector<std::uint16_t> host_oracle_normalized;
  std::vector<std::uint16_t> host_candidate_residual;
  std::vector<std::uint16_t> host_candidate_normalized;
  std::vector<std::uint8_t> host_oracle_packed;
  std::vector<std::uint8_t> host_candidate_packed;
  std::vector<std::uint16_t> host_oracle_scales;
  std::vector<std::uint16_t> host_candidate_scales;
  if (!oracle_residual.copy_to_host(&host_oracle_residual,
                                    spec.name + " copy oracle residual") ||
      !oracle_normalized.copy_to_host(
          &host_oracle_normalized, spec.name + " copy oracle normalized") ||
      !candidate_residual.copy_to_host(
          &host_candidate_residual,
          spec.name + " copy candidate residual") ||
      !candidate_normalized.copy_to_host(
          &host_candidate_normalized,
          spec.name + " copy candidate normalized") ||
      !oracle_packed.copy_to_host(&host_oracle_packed,
                                  spec.name + " copy oracle K256 codes") ||
      !candidate_packed.copy_to_host(
          &host_candidate_packed,
          spec.name + " copy candidate K256 codes") ||
      !oracle_scales.copy_to_host(&host_oracle_scales,
                                  spec.name + " copy oracle K256 scales") ||
      !candidate_scales.copy_to_host(
          &host_candidate_scales,
          spec.name + " copy candidate K256 scales")) {
    return false;
  }

  const bool guards =
      oracle_residual.guards_are_untouched(host_oracle_residual) &&
      oracle_normalized.guards_are_untouched(host_oracle_normalized) &&
      candidate_residual.guards_are_untouched(host_candidate_residual) &&
      candidate_normalized.guards_are_untouched(host_candidate_normalized) &&
      oracle_packed.guards_are_untouched(host_oracle_packed) &&
      candidate_packed.guards_are_untouched(host_candidate_packed) &&
      oracle_scales.guards_are_untouched(host_oracle_scales) &&
      candidate_scales.guards_are_untouched(host_candidate_scales);
  const bool residual_exact = logical_payload_equal(
      oracle_residual, host_oracle_residual, candidate_residual,
      host_candidate_residual, logical_elements);
  const bool normalized_exact = logical_payload_equal(
      oracle_normalized, host_oracle_normalized, candidate_normalized,
      host_candidate_normalized, logical_elements);
  const bool packed_exact = std::equal(
      oracle_packed.host_payload(host_oracle_packed),
      oracle_packed.host_payload(host_oracle_packed) + packed_elements,
      candidate_packed.host_payload(host_candidate_packed));
  const bool scales_exact = std::equal(
      oracle_scales.host_payload(host_oracle_scales),
      oracle_scales.host_payload(host_oracle_scales) + scale_elements,
      candidate_scales.host_payload(host_candidate_scales));
  const bool suffix =
      word_suffix_is_sentinel(oracle_residual, host_oracle_residual,
                              logical_elements) &&
      word_suffix_is_sentinel(oracle_normalized, host_oracle_normalized,
                              logical_elements) &&
      word_suffix_is_sentinel(candidate_residual, host_candidate_residual,
                              logical_elements) &&
      word_suffix_is_sentinel(candidate_normalized,
                              host_candidate_normalized, logical_elements);
  const bool padded =
      padded_k256_is_zero_one(
          candidate_packed.host_payload(host_candidate_packed),
          candidate_scales.host_payload(host_candidate_scales),
          spec.logical_tokens, spec.launch_tokens) &&
      padded_k256_is_zero_one(
          candidate_packed.host_payload(host_candidate_packed),
          candidate_scales.host_payload(host_candidate_scales), 0U, 1U);
  if (!guards || !residual_exact || !normalized_exact || !packed_exact ||
      !scales_exact || !suffix || !padded) {
    std::cerr << spec.name
              << ": MLP-to-attention bit/guard/padding mismatch"
              << " guards=" << guards << " residual=" << residual_exact
              << " normalized=" << normalized_exact
              << " packed=" << packed_exact << " scales=" << scales_exact
              << " suffix=" << suffix << " padded=" << padded << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool expect_invalid(const int status,
                                  const std::string& operation) {
  if (status == static_cast<int>(cudaErrorInvalidValue)) {
    return true;
  }
  std::cerr << operation << " returned " << status
            << ", expected cudaErrorInvalidValue\n";
  return false;
}

[[nodiscard]] bool test_invalid_arguments_and_capacities() {
  constexpr std::size_t kLogical = 1U;
  constexpr std::size_t kLaunch = 128U;
  constexpr std::size_t kLogicalElements = kLogical * kHidden;
  constexpr std::size_t kLaunchElements = kLaunch * kHidden;
  const std::size_t packed_capacity =
      kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(kLaunch,
                                                               kHidden);
  const std::size_t r1_scale_capacity =
      kernels::sm87_a4w4_factorized_lane_scale_capacity_elements(kLaunch,
                                                                  1U);
  const std::size_t k256_scale_capacity =
      kernels::sm87_a4w4_attention_k256_scale_capacity_elements(kLaunch,
                                                                 kHidden);
  DeviceBuffer<std::uint16_t> left;
  DeviceBuffer<std::uint16_t> right;
  DeviceBuffer<std::uint16_t> weight;
  DeviceBuffer<float> inverse_alpha;
  DeviceBuffer<std::uint16_t> residual;
  DeviceBuffer<std::uint16_t> normalized;
  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> unaligned_packed_storage;
  DeviceBuffer<std::uint16_t> r1_scales;
  DeviceBuffer<std::uint16_t> k256_scales;
  if (!left.allocate(kLogicalElements) || !right.allocate(kLogicalElements) ||
      !weight.allocate(kHidden) || !inverse_alpha.allocate(kHidden) ||
      !residual.allocate(kLaunchElements) ||
      !normalized.allocate(kLaunchElements) ||
      !packed.allocate(packed_capacity) ||
      !unaligned_packed_storage.allocate(packed_capacity + 1U) ||
      !r1_scales.allocate(r1_scale_capacity) ||
      !k256_scales.allocate(k256_scale_capacity)) {
    std::cerr << "invalid-argument fixture allocation failed\n";
    return false;
  }

  const auto attention =
      [&](const std::uint16_t* left_argument,
          const std::size_t inverse_capacity,
          const std::size_t logical_tokens,
          const std::size_t launch_tokens,
          const float epsilon,
          const float clip_ratio,
          const std::size_t residual_capacity,
          std::uint8_t* packed_argument,
          const std::size_t packed_bytes,
          const std::size_t scale_capacity) {
        return kernels::
            launch_sm87_a4w4_attention_residual_post_norm_r1_quantize_cuda(
                left_argument, right.get(), weight.get(), inverse_alpha.get(),
                inverse_capacity, logical_tokens, launch_tokens, epsilon,
                clip_ratio, residual.get(), residual_capacity,
                packed_argument, packed_bytes, r1_scales.get(),
                scale_capacity);
      };
  const auto mlp =
      [&](const std::uint16_t* left_argument,
          const std::size_t logical_tokens,
          const std::size_t launch_tokens,
          const float epsilon,
          const float clip_ratio,
          const std::size_t residual_capacity,
          std::uint16_t* normalized_argument,
          const std::size_t normalized_capacity,
          std::uint8_t* packed_argument,
          const std::size_t packed_bytes,
          const std::size_t scale_capacity) {
        return kernels::
            launch_sm87_a4w4_mlp_residual_next_norm_k256_quantize_cuda(
                left_argument, right.get(), weight.get(), logical_tokens,
                launch_tokens, epsilon, clip_ratio, residual.get(),
                residual_capacity, normalized_argument, normalized_capacity,
                packed_argument, packed_bytes, k256_scales.get(),
                scale_capacity);
      };

  bool ok = true;
  ok = expect_invalid(attention(nullptr, kHidden, kLogical, kLaunch,
                                kEpsilon, kClipRatio, kLogicalElements,
                                packed.get(), packed_capacity,
                                r1_scale_capacity),
                      "attention null input") &&
       ok;
  ok = expect_invalid(attention(left.get(), kHidden - 1U, kLogical, kLaunch,
                                kEpsilon, kClipRatio, kLogicalElements,
                                packed.get(), packed_capacity,
                                r1_scale_capacity),
                      "attention short inverse alpha") &&
       ok;
  ok = expect_invalid(attention(left.get(), kHidden, 0U, kLaunch, kEpsilon,
                                kClipRatio, kLogicalElements, packed.get(),
                                packed_capacity, r1_scale_capacity),
                      "attention zero logical tokens") &&
       ok;
  ok = expect_invalid(attention(left.get(), kHidden, kLogical, 127U,
                                kEpsilon, kClipRatio, kLogicalElements,
                                packed.get(), packed_capacity,
                                r1_scale_capacity),
                      "attention invalid launch multiple") &&
       ok;
  ok = expect_invalid(attention(
                          left.get(), kHidden, kLogical,
                          kernels::kSm87A4W4PrefillHandoffMaximumTokens + 64U,
                          kEpsilon, kClipRatio, kLogicalElements, packed.get(),
                          packed_capacity, r1_scale_capacity),
                      "attention excessive launch tokens") &&
       ok;
  ok = expect_invalid(attention(left.get(), kHidden, kLogical, kLaunch, 0.0F,
                                kClipRatio, kLogicalElements, packed.get(),
                                packed_capacity, r1_scale_capacity),
                      "attention zero epsilon") &&
       ok;
  ok = expect_invalid(attention(
                          left.get(), kHidden, kLogical, kLaunch,
                          std::numeric_limits<float>::quiet_NaN(), kClipRatio,
                          kLogicalElements, packed.get(), packed_capacity,
                          r1_scale_capacity),
                      "attention NaN epsilon") &&
       ok;
  ok = expect_invalid(attention(left.get(), kHidden, kLogical, kLaunch,
                                kEpsilon, 0.0F, kLogicalElements, packed.get(),
                                packed_capacity, r1_scale_capacity),
                      "attention zero clip") &&
       ok;
  ok = expect_invalid(attention(left.get(), kHidden, kLogical, kLaunch,
                                kEpsilon, kClipRatio,
                                kLogicalElements - 1U, packed.get(),
                                packed_capacity, r1_scale_capacity),
                      "attention short residual") &&
       ok;
  ok = expect_invalid(attention(left.get(), kHidden, kLogical, kLaunch,
                                kEpsilon, kClipRatio, kLogicalElements,
                                packed.get(), packed_capacity - 1U,
                                r1_scale_capacity),
                      "attention short packed") &&
       ok;
  ok = expect_invalid(attention(left.get(), kHidden, kLogical, kLaunch,
                                kEpsilon, kClipRatio, kLogicalElements,
                                unaligned_packed_storage.get() + 1U,
                                packed_capacity, r1_scale_capacity),
                      "attention unaligned packed") &&
       ok;
  ok = expect_invalid(attention(left.get(), kHidden, kLogical, kLaunch,
                                kEpsilon, kClipRatio, kLogicalElements,
                                packed.get(), packed_capacity,
                                r1_scale_capacity - 1U),
                      "attention short scales") &&
       ok;

  ok = expect_invalid(mlp(nullptr, kLogical, kLaunch, kEpsilon, kClipRatio,
                          kLogicalElements, normalized.get(),
                          kLogicalElements, packed.get(), packed_capacity,
                          k256_scale_capacity),
                      "MLP null input") &&
       ok;
  ok = expect_invalid(mlp(left.get(), kLogical, 192U, kEpsilon, kClipRatio,
                          kLogicalElements, normalized.get(),
                          kLogicalElements, packed.get(), packed_capacity,
                          k256_scale_capacity),
                      "MLP noncanonical launch tokens") &&
       ok;
  ok = expect_invalid(mlp(left.get(), kLogical, kLaunch, kEpsilon,
                          kClipRatio, kLogicalElements - 1U,
                          normalized.get(), kLogicalElements, packed.get(),
                          packed_capacity, k256_scale_capacity),
                      "MLP short residual") &&
       ok;
  ok = expect_invalid(mlp(left.get(), kLogical, kLaunch, kEpsilon,
                          kClipRatio, kLogicalElements, nullptr,
                          kLogicalElements, packed.get(), packed_capacity,
                          k256_scale_capacity),
                      "MLP null normalized output") &&
       ok;
  ok = expect_invalid(mlp(left.get(), kLogical, kLaunch, kEpsilon,
                          kClipRatio, kLogicalElements, normalized.get(),
                          kLogicalElements - 1U, packed.get(), packed_capacity,
                          k256_scale_capacity),
                      "MLP short normalized output") &&
       ok;
  ok = expect_invalid(mlp(left.get(), kLogical, kLaunch, kEpsilon,
                          1.01F, kLogicalElements, normalized.get(),
                          kLogicalElements, packed.get(), packed_capacity,
                          k256_scale_capacity),
                      "MLP excessive clip") &&
       ok;
  ok = expect_invalid(mlp(left.get(), kLogical, kLaunch, kEpsilon,
                          kClipRatio, kLogicalElements, normalized.get(),
                          kLogicalElements, packed.get(), packed_capacity - 1U,
                          k256_scale_capacity),
                      "MLP short packed") &&
       ok;
  ok = expect_invalid(mlp(left.get(), kLogical, kLaunch, kEpsilon,
                          kClipRatio, kLogicalElements, normalized.get(),
                          kLogicalElements, packed.get(), packed_capacity,
                          k256_scale_capacity - 1U),
                      "MLP short scales") &&
       ok;
  return ok;
}

[[nodiscard]] bool test_resource_contract() {
  kernels::Sm87A4W4PrefillHandoffResources resources{};
  const int status =
      kernels::query_sm87_a4w4_prefill_handoff_resources_cuda(&resources);
  if (!cuda_ok(static_cast<cudaError_t>(status),
               "query prefill handoff resources")) {
    return false;
  }
  const bool valid =
      resources.attention_to_gate_registers_per_thread > 0 &&
      resources.attention_to_gate_registers_per_thread <=
          static_cast<int>(
              kernels::kSm87A4W4PrefillHandoffMaximumRegisters) &&
      resources.mlp_to_attention_registers_per_thread > 0 &&
      resources.mlp_to_attention_registers_per_thread <=
          static_cast<int>(
              kernels::kSm87A4W4PrefillHandoffMaximumRegisters) &&
      resources.attention_to_gate_local_bytes == 0U &&
      resources.mlp_to_attention_local_bytes == 0U &&
      resources.attention_to_gate_active_blocks_per_sm >=
          static_cast<int>(
              kernels::kSm87A4W4PrefillHandoffMinimumBlocksPerSm) &&
      resources.mlp_to_attention_active_blocks_per_sm >=
          static_cast<int>(
              kernels::kSm87A4W4PrefillHandoffMinimumBlocksPerSm) &&
      resources.maximum_threads_per_block >=
          static_cast<int>(kernels::kSm87A4W4PrefillHandoffThreads) &&
      resources.compute_major == 8 && resources.compute_minor == 7 &&
      resources.multiprocessor_count == 16;
  if (!valid) {
    std::cerr << "prefill handoff resource contract mismatch\n";
  }
  return valid;
}

}  // namespace

int main() {
  const int target_status = target_device_status();
  if (target_status != 0) {
    return target_status;
  }

  bool ok = test_resource_contract() &&
            test_invalid_arguments_and_capacities();
  const std::vector<CaseSpec> cases{
      {"P1_to_P128_graph", 1U, 128U, true},
      {"P127_to_P128", 127U, 128U, false},
      {"P1853_to_P1920", 1'853U, 1'920U, false},
  };
  for (const CaseSpec& spec : cases) {
    const HostInputs inputs = make_inputs(spec);
    const bool attention_ok = run_attention_to_gate_case(spec, inputs);
    const bool mlp_ok = run_mlp_to_attention_case(spec, inputs);
    std::cout << spec.name << ": attention_to_gate=" << attention_ok
              << " mlp_to_attention=" << mlp_ok << '\n';
    ok = attention_ok && mlp_ok && ok;
  }

  if (!ok) {
    return 1;
  }
  std::cout << "PASS: prefill handoff kernels are bit-exact against the "
               "incumbent residual/RMS/quantize chains across direct, "
               "non-default-stream, Graph replay, and P1853 padding\n";
  return 0;
}
