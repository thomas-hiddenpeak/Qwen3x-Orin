#include "q3x/kernels/sm87_macrofeed_v3_gdn_p40.h"
#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/gdn_decode.h"

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
namespace runtime = q3x::runtime;

constexpr std::size_t kTokens = 128U;
constexpr std::size_t kQkvElements =
    kTokens * kernels::kSm87TargetAotGdnTotalConvChannels;
constexpr std::size_t kScalarElements =
    kTokens * kernels::kSm87TargetAotGdnValueHeads;
constexpr std::size_t kZElements =
    kTokens * kernels::kSm87TargetAotGdnOutputChannels;
constexpr std::size_t kConvWeightElements =
    kernels::kSm87TargetAotGdnConvWeightElements;
constexpr std::size_t kConvHistoryElements =
    kernels::kSm87TargetAotGdnConvHistoryElements;
constexpr std::size_t kStateElements =
    kernels::kSm87TargetAotGdnRecurrentStateElements;

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const std::string& operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << "FAIL: " << operation << ": "
            << cudaGetErrorString(status) << '\n';
  return false;
}

template <class T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0U)) {}

  [[nodiscard]] bool allocate(const std::size_t count) {
    size_ = count;
    return cuda_ok(cudaMalloc(reinterpret_cast<void**>(&data_),
                              count * sizeof(T)),
                   "allocate device buffer");
  }
  [[nodiscard]] bool upload(const std::vector<T>& source,
                            cudaStream_t stream) {
    return source.size() == size_ &&
           cuda_ok(cudaMemcpyAsync(data_, source.data(),
                                   size_ * sizeof(T),
                                   cudaMemcpyHostToDevice, stream),
                   "upload device buffer");
  }
  [[nodiscard]] bool copy_from(const DeviceBuffer& source,
                               cudaStream_t stream) {
    return source.size_ == size_ &&
           cuda_ok(cudaMemcpyAsync(data_, source.data_,
                                   size_ * sizeof(T),
                                   cudaMemcpyDeviceToDevice, stream),
                   "copy device buffer");
  }
  [[nodiscard]] bool download(std::vector<T>& destination,
                              cudaStream_t stream) const {
    destination.resize(size_);
    return cuda_ok(cudaMemcpyAsync(destination.data(), data_,
                                   size_ * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream),
                   "download device buffer");
  }
  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0U;
};

struct Inputs final {
  DeviceBuffer<std::uint16_t> raw_qkv;
  DeviceBuffer<std::uint16_t> a;
  DeviceBuffer<std::uint16_t> b;
  DeviceBuffer<std::uint16_t> conv_weight;
  DeviceBuffer<std::uint16_t> a_log;
  DeviceBuffer<std::uint16_t> dt_bias;
  DeviceBuffer<std::uint16_t> norm_weight;
  DeviceBuffer<std::uint16_t> z;
  DeviceBuffer<std::uint16_t> initial_history;
  DeviceBuffer<std::uint16_t> initial_state;
};

struct Outputs final {
  DeviceBuffer<std::uint16_t> history;
  DeviceBuffer<std::uint16_t> state;
  DeviceBuffer<std::uint16_t> conv_qkv;
  DeviceBuffer<std::uint16_t> raw_output;
  DeviceBuffer<std::uint16_t> output;
};

[[nodiscard]] std::uint16_t patterned_bf16(const std::size_t index) {
  constexpr std::uint16_t values[] = {
      0x0000U, 0x3d00U, 0xbd00U, 0x3e00U,
      0xbe00U, 0x3e80U, 0xbe80U, 0x3f00U,
      0xbf00U, 0x3f80U, 0xbf80U,
  };
  return values[(index * 17U + index / 13U) %
                (sizeof(values) / sizeof(values[0]))];
}

[[nodiscard]] bool prepare_inputs(Inputs& inputs, cudaStream_t stream) {
  const bool allocated =
      inputs.raw_qkv.allocate(kQkvElements) &&
      inputs.a.allocate(kScalarElements) &&
      inputs.b.allocate(kScalarElements) &&
      inputs.conv_weight.allocate(kConvWeightElements) &&
      inputs.a_log.allocate(kernels::kSm87TargetAotGdnValueHeads) &&
      inputs.dt_bias.allocate(kernels::kSm87TargetAotGdnValueHeads) &&
      inputs.norm_weight.allocate(
          kernels::kSm87TargetAotGdnNormWeightElements) &&
      inputs.z.allocate(kZElements) &&
      inputs.initial_history.allocate(kConvHistoryElements) &&
      inputs.initial_state.allocate(kStateElements);
  if (!allocated) {
    return false;
  }

  std::vector<std::uint16_t> raw(kQkvElements);
  std::vector<std::uint16_t> a(kScalarElements);
  std::vector<std::uint16_t> b(kScalarElements);
  std::vector<std::uint16_t> weights(kConvWeightElements);
  std::vector<std::uint16_t> a_log(
      kernels::kSm87TargetAotGdnValueHeads, 0x0000U);
  std::vector<std::uint16_t> dt_bias(
      kernels::kSm87TargetAotGdnValueHeads, 0x0000U);
  std::vector<std::uint16_t> norm(
      kernels::kSm87TargetAotGdnNormWeightElements, 0x3f80U);
  std::vector<std::uint16_t> z(kZElements);
  std::vector<std::uint16_t> history(kConvHistoryElements);
  std::vector<std::uint16_t> state(kStateElements);
  for (std::size_t index = 0U; index < raw.size(); ++index) {
    raw[index] = patterned_bf16(index);
  }
  for (std::size_t index = 0U; index < a.size(); ++index) {
    a[index] = patterned_bf16(index + 31U);
    b[index] = patterned_bf16(index + 67U);
  }
  constexpr std::uint16_t taps[] = {
      0x3e00U, 0xbd80U, 0x3e80U, 0x3d00U};
  for (std::size_t index = 0U; index < weights.size(); ++index) {
    weights[index] = taps[index % 4U];
  }
  for (std::size_t index = 0U; index < z.size(); ++index) {
    z[index] = patterned_bf16(index + 101U);
  }
  for (std::size_t index = 0U; index < history.size(); ++index) {
    history[index] = patterned_bf16(index + 149U);
  }
  for (std::size_t index = 0U; index < state.size(); ++index) {
    state[index] = patterned_bf16(index + 211U);
  }
  return inputs.raw_qkv.upload(raw, stream) &&
         inputs.a.upload(a, stream) && inputs.b.upload(b, stream) &&
         inputs.conv_weight.upload(weights, stream) &&
         inputs.a_log.upload(a_log, stream) &&
         inputs.dt_bias.upload(dt_bias, stream) &&
         inputs.norm_weight.upload(norm, stream) &&
         inputs.z.upload(z, stream) &&
         inputs.initial_history.upload(history, stream) &&
         inputs.initial_state.upload(state, stream);
}

[[nodiscard]] bool prepare_outputs(Outputs& outputs,
                                   const Inputs& inputs,
                                   cudaStream_t stream) {
  return outputs.history.allocate(kConvHistoryElements) &&
         outputs.state.allocate(kStateElements) &&
         outputs.conv_qkv.allocate(kQkvElements) &&
         outputs.raw_output.allocate(kZElements) &&
         outputs.output.allocate(kZElements) &&
         outputs.history.copy_from(inputs.initial_history, stream) &&
         outputs.state.copy_from(inputs.initial_state, stream);
}

[[nodiscard]] float epsilon() {
  const std::uint32_t bits = kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] bool run_candidate(const Inputs& inputs,
                                 Outputs& outputs,
                                 cudaStream_t stream) {
  kernels::Sm87MacrofeedV3GdnC64OracleArguments arguments;
  arguments.raw_qkv = inputs.raw_qkv.data();
  arguments.a = inputs.a.data();
  arguments.b = inputs.b.data();
  arguments.conv_weight = inputs.conv_weight.data();
  arguments.a_log = inputs.a_log.data();
  arguments.dt_bias = inputs.dt_bias.data();
  arguments.norm_weight = inputs.norm_weight.data();
  arguments.z = inputs.z.data();
  arguments.conv_history = outputs.history.data();
  arguments.recurrent_state = outputs.state.data();
  arguments.conv_qkv = outputs.conv_qkv.data();
  arguments.output = outputs.output.data();
  arguments.l2_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.norm_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.cuda_stream = stream;
  return cuda_ok(static_cast<cudaError_t>(
                     kernels::
                         launch_sm87_macrofeed_v3_gdn_c128_two_epoch_oracle_cuda(
                             arguments)),
                 "launch MacroFeed-v3 GDN C128 two-epoch oracle");
}

[[nodiscard]] bool run_reference(const Inputs& inputs,
                                 Outputs& outputs,
                                 cudaStream_t stream) {
  constexpr std::size_t kTile = 16U;
  for (std::size_t offset = 0U; offset < kTokens; offset += kTile) {
    if (!cuda_ok(static_cast<cudaError_t>(
                     runtime::launch_causal_conv1d_silu_update_tile_reference_cuda(
                         inputs.raw_qkv.data() +
                             offset * runtime::kGdnQkvChannels,
                         kTile, inputs.conv_weight.data(),
                         outputs.history.data(),
                         outputs.conv_qkv.data() +
                             offset * runtime::kGdnQkvChannels,
                         {}, stream)),
                 "launch exact C16 convolution reference")) {
      return false;
    }
  }
  for (std::size_t offset = 0U; offset < kTokens; offset += kTile) {
    if (!cuda_ok(static_cast<cudaError_t>(
                     runtime::launch_gated_delta_net_update_tile_warp_parallel_cuda(
                         outputs.conv_qkv.data() +
                             offset * runtime::kGdnQkvChannels,
                         kTile,
                         inputs.a.data() +
                             offset * runtime::kGdnValueHeadCount,
                         inputs.b.data() +
                             offset * runtime::kGdnValueHeadCount,
                         inputs.a_log.data(), inputs.dt_bias.data(),
                         outputs.state.data(), outputs.state.data(), epsilon(),
                         outputs.raw_output.data() +
                             offset * runtime::kGdnVElements,
                         {}, stream)),
                 "launch exact C16 recurrence reference")) {
      return false;
    }
  }
  return cuda_ok(
      static_cast<cudaError_t>(
          runtime::launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
              outputs.raw_output.data(), inputs.norm_weight.data(),
              inputs.z.data(),
              kTokens * runtime::kGdnValueHeadCount,
              runtime::kGdnHeadDimension, epsilon(), outputs.output.data(),
              stream)),
      "launch exact C128 norm/gate reference");
}

[[nodiscard]] bool compare(const char* const label,
                           const DeviceBuffer<std::uint16_t>& candidate,
                           const DeviceBuffer<std::uint16_t>& reference,
                           cudaStream_t stream) {
  std::vector<std::uint16_t> left;
  std::vector<std::uint16_t> right;
  if (!candidate.download(left, stream) ||
      !reference.download(right, stream) ||
      !cuda_ok(cudaStreamSynchronize(stream),
               std::string("synchronize ") + label)) {
    return false;
  }
  const auto mismatch = std::mismatch(left.begin(), left.end(), right.begin());
  if (mismatch.first == left.end()) {
    return true;
  }
  const std::size_t index =
      static_cast<std::size_t>(mismatch.first - left.begin());
  std::cerr << "FAIL: " << label << " first BF16 mismatch at " << index
            << " candidate=0x" << std::hex << *mismatch.first
            << " reference=0x" << *mismatch.second << std::dec << '\n';
  return false;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA device unavailable\n";
    (void)cudaGetLastError();
    return 77;
  }
  int device = -1;
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDevice(&device), "get CUDA device") ||
      !cuda_ok(cudaGetDeviceProperties(&properties, device),
               "query CUDA device")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: test requires the fixed 16-SM SM87 target\n";
    return 77;
  }

  kernels::Sm87MacrofeedV3GdnResources resources{};
  if (!cuda_ok(static_cast<cudaError_t>(
                   kernels::query_sm87_macrofeed_v3_gdn_p40_resources_cuda(
                       &resources)),
               "query MacroFeed-v3 GDN resources") ||
      !kernels::sm87_macrofeed_v3_gdn_resources_valid(resources)) {
    std::cerr << "FAIL: MacroFeed-v3 GDN resource envelope rejected\n";
    return 1;
  }
  std::cout << "resources: conv_regs="
            << resources.convolution.registers_per_thread
            << " conv_active=" << resources.convolution.active_blocks_per_sm
            << " recurrence_regs="
            << resources.recurrence_epilogue.registers_per_thread
            << " recurrence_shared="
            << resources.recurrence_epilogue.static_shared_bytes
            << " recurrence_active="
            << resources.recurrence_epilogue.active_blocks_per_sm << '\n';

  cudaStream_t stream = nullptr;
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create oracle stream")) {
    return 1;
  }
  Inputs inputs;
  Outputs candidate;
  Outputs reference;
  bool passed = prepare_inputs(inputs, stream) &&
                prepare_outputs(candidate, inputs, stream) &&
                prepare_outputs(reference, inputs, stream) &&
                run_candidate(inputs, candidate, stream) &&
                run_reference(inputs, reference, stream) &&
                cuda_ok(cudaStreamSynchronize(stream),
                        "synchronize both GDN paths");
  if (passed) {
    passed = compare("convolved QKV", candidate.conv_qkv,
                     reference.conv_qkv, stream) &&
             compare("convolution history", candidate.history,
                     reference.history, stream) &&
             compare("recurrent state", candidate.state, reference.state,
                     stream) &&
             compare("normalized gated output", candidate.output,
                     reference.output, stream);
  }
  (void)cudaStreamDestroy(stream);
  if (!passed) {
    return 1;
  }
  std::cout << "SM87 MacroFeed-v3 GDN C128 two-epoch bitwise CUDA oracle passed "
               "(correctness only; no performance or production authority)\n";
  return 0;
}
