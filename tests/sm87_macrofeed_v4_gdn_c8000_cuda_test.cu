#include "q3x/kernels/sm87_macrofeed_v4_gdn_c8000.h"
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

constexpr std::size_t kGuardElements = 8U;
constexpr std::uint16_t kGuardBits = 0x5a5aU;
constexpr std::uint16_t kPoisonBits = 0x7fc1U;

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
class GuardedDeviceBuffer final {
 public:
  GuardedDeviceBuffer() = default;
  ~GuardedDeviceBuffer() {
    if (base_ != nullptr) {
      (void)cudaFree(base_);
    }
  }
  GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
  GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;

  [[nodiscard]] bool allocate_and_upload(const std::vector<T>& logical,
                                         const T guard,
                                         cudaStream_t stream) {
    logical_size_ = logical.size();
    const std::size_t total = logical_size_ + 2U * kGuardElements;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&base_),
                            total * sizeof(T)),
                 "allocate guarded device buffer")) {
      return false;
    }
    std::vector<T> host(total, guard);
    std::copy(logical.begin(), logical.end(),
              host.begin() + static_cast<std::ptrdiff_t>(kGuardElements));
    return cuda_ok(cudaMemcpyAsync(base_, host.data(), total * sizeof(T),
                                   cudaMemcpyHostToDevice, stream),
                   "upload guarded device buffer");
  }

  [[nodiscard]] bool download(std::vector<T>& logical,
                              cudaStream_t stream) const {
    logical.resize(logical_size_);
    return cuda_ok(cudaMemcpyAsync(logical.data(), data(),
                                   logical_size_ * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream),
                   "download guarded logical buffer");
  }

  [[nodiscard]] bool guards_intact(const T guard,
                                   cudaStream_t stream,
                                   const char* label) const {
    const std::size_t total = logical_size_ + 2U * kGuardElements;
    std::vector<T> host(total);
    if (!cuda_ok(cudaMemcpyAsync(host.data(), base_, total * sizeof(T),
                                 cudaMemcpyDeviceToHost, stream),
                 std::string("download guards for ") + label) ||
        !cuda_ok(cudaStreamSynchronize(stream),
                 std::string("synchronize guards for ") + label)) {
      return false;
    }
    for (std::size_t index = 0U; index < kGuardElements; ++index) {
      if (host[index] != guard ||
          host[kGuardElements + logical_size_ + index] != guard) {
        std::cerr << "FAIL: guard corruption in " << label << " at "
                  << index << '\n';
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] T* data() noexcept {
    return base_ + static_cast<std::ptrdiff_t>(kGuardElements);
  }
  [[nodiscard]] const T* data() const noexcept {
    return base_ + static_cast<std::ptrdiff_t>(kGuardElements);
  }

 private:
  T* base_ = nullptr;
  std::size_t logical_size_ = 0U;
};

[[nodiscard]] std::uint16_t patterned_bf16(const std::size_t index) {
  constexpr std::uint16_t values[] = {
      0x0000U, 0x3c80U, 0xbc80U, 0x3d00U, 0xbd00U, 0x3d80U,
      0xbd80U, 0x3e00U, 0xbe00U, 0x3e80U, 0xbe80U, 0x3f00U,
      0xbf00U, 0x3f80U, 0xbf80U, 0x4000U, 0xc000U,
  };
  return values[(index * 19U + index / 11U + 3U) %
                (sizeof(values) / sizeof(values[0]))];
}

struct HostFixture final {
  std::vector<std::uint16_t> scratch;
  std::vector<std::uint16_t> raw_qkv;
  std::vector<std::uint16_t> a;
  std::vector<std::uint16_t> b;
  std::vector<std::uint16_t> z;
  std::vector<std::uint16_t> conv_weight;
  std::vector<std::uint16_t> a_log;
  std::vector<std::uint16_t> dt_bias;
  std::vector<std::uint16_t> norm_weight;
  std::vector<std::uint16_t> active_history;
  std::vector<std::uint16_t> active_state;
};

[[nodiscard]] HostFixture make_fixture(const std::size_t token_count) {
  HostFixture fixture;
  fixture.scratch.assign(
      token_count * kernels::kSm87MacroFeedV4GdnScratchRowStride,
      kPoisonBits);
  fixture.raw_qkv.resize(
      token_count * kernels::kSm87TargetAotGdnTotalConvChannels);
  fixture.a.resize(token_count * kernels::kSm87TargetAotGdnValueHeads);
  fixture.b.resize(token_count * kernels::kSm87TargetAotGdnValueHeads);
  fixture.z.resize(token_count * kernels::kSm87TargetAotGdnOutputChannels);
  fixture.conv_weight.resize(kernels::kSm87TargetAotGdnConvWeightElements);
  fixture.a_log.resize(kernels::kSm87TargetAotGdnValueHeads);
  fixture.dt_bias.resize(kernels::kSm87TargetAotGdnValueHeads);
  fixture.norm_weight.resize(kernels::kSm87TargetAotGdnNormWeightElements);
  fixture.active_history.resize(
      kernels::kSm87TargetAotGdnConvHistoryElements);
  fixture.active_state.resize(kernels::kSm87TargetAotGdnRecurrentStateElements);

  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t scratch_row =
        token * kernels::kSm87MacroFeedV4GdnScratchRowStride;
    const std::size_t qkv_row =
        token * kernels::kSm87TargetAotGdnTotalConvChannels;
    for (std::size_t channel = 0U;
         channel < kernels::kSm87TargetAotGdnTotalConvChannels; ++channel) {
      const std::uint16_t value =
          patterned_bf16(token * 131U + channel * 7U);
      fixture.raw_qkv[qkv_row + channel] = value;
      fixture.scratch[scratch_row + kernels::kSm87MacroFeedV4GdnQkvOffset +
                      channel] = value;
    }
    for (std::size_t channel = 0U;
         channel < kernels::kSm87TargetAotGdnOutputChannels; ++channel) {
      const std::uint16_t value =
          patterned_bf16(token * 173U + channel * 13U + 97U);
      fixture.z[token * kernels::kSm87TargetAotGdnOutputChannels + channel] =
          value;
      fixture.scratch[scratch_row + kernels::kSm87MacroFeedV4GdnZOffset +
                      channel] = value;
    }
    for (std::size_t head = 0U;
         head < kernels::kSm87TargetAotGdnValueHeads; ++head) {
      const std::uint16_t a_value =
          patterned_bf16(token * 23U + head * 29U + 31U);
      const std::uint16_t b_value =
          patterned_bf16(token * 37U + head * 17U + 61U);
      fixture.a[token * kernels::kSm87TargetAotGdnValueHeads + head] =
          a_value;
      fixture.b[token * kernels::kSm87TargetAotGdnValueHeads + head] =
          b_value;
      fixture.scratch[scratch_row + kernels::kSm87MacroFeedV4GdnAOffset +
                      head] = a_value;
      fixture.scratch[scratch_row + kernels::kSm87MacroFeedV4GdnBOffset +
                      head] = b_value;
    }
  }

  constexpr std::uint16_t taps[] = {
      0x3e00U, 0xbd80U, 0x3e80U, 0x3d00U};
  for (std::size_t index = 0U; index < fixture.conv_weight.size(); ++index) {
    fixture.conv_weight[index] = taps[index % 4U];
  }
  for (std::size_t head = 0U;
       head < kernels::kSm87TargetAotGdnValueHeads; ++head) {
    fixture.a_log[head] = patterned_bf16(head * 5U + 1U);
    fixture.dt_bias[head] = patterned_bf16(head * 7U + 2U);
  }
  for (std::size_t dimension = 0U;
       dimension < kernels::kSm87TargetAotGdnNormWeightElements;
       ++dimension) {
    fixture.norm_weight[dimension] =
        patterned_bf16(dimension * 11U + 5U);
  }
  for (std::size_t index = 0U; index < fixture.active_history.size();
       ++index) {
    fixture.active_history[index] = patterned_bf16(index * 3U + 149U);
  }
  for (std::size_t index = 0U; index < fixture.active_state.size(); ++index) {
    fixture.active_state[index] = patterned_bf16(index * 5U + 211U);
  }
  return fixture;
}

struct CandidateBuffers final {
  GuardedDeviceBuffer<std::uint16_t> scratch;
  GuardedDeviceBuffer<std::uint16_t> conv_weight;
  GuardedDeviceBuffer<std::uint16_t> a_log;
  GuardedDeviceBuffer<std::uint16_t> dt_bias;
  GuardedDeviceBuffer<std::uint16_t> norm_weight;
  GuardedDeviceBuffer<std::uint16_t> active_history;
  GuardedDeviceBuffer<std::uint16_t> candidate_history;
  GuardedDeviceBuffer<std::uint16_t> active_state;
  GuardedDeviceBuffer<std::uint16_t> candidate_state;
};

struct ReferenceBuffers final {
  GuardedDeviceBuffer<std::uint16_t> raw_qkv;
  GuardedDeviceBuffer<std::uint16_t> a;
  GuardedDeviceBuffer<std::uint16_t> b;
  GuardedDeviceBuffer<std::uint16_t> z;
  GuardedDeviceBuffer<std::uint16_t> history;
  GuardedDeviceBuffer<std::uint16_t> state;
  GuardedDeviceBuffer<std::uint16_t> conv_qkv;
  GuardedDeviceBuffer<std::uint16_t> raw_output;
  GuardedDeviceBuffer<std::uint16_t> output;
};

[[nodiscard]] bool prepare_candidate(const HostFixture& fixture,
                                     CandidateBuffers& buffers,
                                     cudaStream_t stream) {
  const std::vector<std::uint16_t> poison_history(
      kernels::kSm87TargetAotGdnConvHistoryElements, kPoisonBits);
  const std::vector<std::uint16_t> poison_state(
      kernels::kSm87TargetAotGdnRecurrentStateElements, kPoisonBits);
  return buffers.scratch.allocate_and_upload(fixture.scratch, kGuardBits,
                                             stream) &&
         buffers.conv_weight.allocate_and_upload(fixture.conv_weight,
                                                 kGuardBits, stream) &&
         buffers.a_log.allocate_and_upload(fixture.a_log, kGuardBits,
                                           stream) &&
         buffers.dt_bias.allocate_and_upload(fixture.dt_bias, kGuardBits,
                                             stream) &&
         buffers.norm_weight.allocate_and_upload(fixture.norm_weight,
                                                 kGuardBits, stream) &&
         buffers.active_history.allocate_and_upload(fixture.active_history,
                                                    kGuardBits, stream) &&
         buffers.candidate_history.allocate_and_upload(poison_history,
                                                       kGuardBits, stream) &&
         buffers.active_state.allocate_and_upload(fixture.active_state,
                                                  kGuardBits, stream) &&
         buffers.candidate_state.allocate_and_upload(poison_state,
                                                     kGuardBits, stream);
}

[[nodiscard]] bool prepare_reference(const HostFixture& fixture,
                                     ReferenceBuffers& buffers,
                                     cudaStream_t stream) {
  const std::size_t token_count =
      fixture.raw_qkv.size() / kernels::kSm87TargetAotGdnTotalConvChannels;
  const std::vector<std::uint16_t> poison_qkv(
      fixture.raw_qkv.size(), kPoisonBits);
  const std::vector<std::uint16_t> poison_output(
      token_count * kernels::kSm87TargetAotGdnOutputChannels, kPoisonBits);
  return buffers.raw_qkv.allocate_and_upload(fixture.raw_qkv, kGuardBits,
                                             stream) &&
         buffers.a.allocate_and_upload(fixture.a, kGuardBits, stream) &&
         buffers.b.allocate_and_upload(fixture.b, kGuardBits, stream) &&
         buffers.z.allocate_and_upload(fixture.z, kGuardBits, stream) &&
         buffers.history.allocate_and_upload(fixture.active_history,
                                             kGuardBits, stream) &&
         buffers.state.allocate_and_upload(fixture.active_state, kGuardBits,
                                           stream) &&
         buffers.conv_qkv.allocate_and_upload(poison_qkv, kGuardBits,
                                              stream) &&
         buffers.raw_output.allocate_and_upload(poison_output, kGuardBits,
                                                stream) &&
         buffers.output.allocate_and_upload(poison_output, kGuardBits,
                                            stream);
}

[[nodiscard]] float epsilon() {
  const std::uint32_t bits = kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] bool run_candidate(const std::size_t token_count,
                                 CandidateBuffers& buffers,
                                 cudaStream_t stream) {
  kernels::Sm87MacroFeedV4GdnOracleArguments arguments{};
  arguments.scratch = buffers.scratch.data();
  arguments.token_count = token_count;
  arguments.scratch_row_stride =
      kernels::kSm87MacroFeedV4GdnScratchRowStride;
  arguments.conv_weight = buffers.conv_weight.data();
  arguments.a_log = buffers.a_log.data();
  arguments.dt_bias = buffers.dt_bias.data();
  arguments.norm_weight = buffers.norm_weight.data();
  arguments.active_conv_history = buffers.active_history.data();
  arguments.candidate_conv_history = buffers.candidate_history.data();
  arguments.active_recurrent_state = buffers.active_state.data();
  arguments.candidate_recurrent_state = buffers.candidate_state.data();
  arguments.l2_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.norm_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.cuda_stream = stream;
  return cuda_ok(
      static_cast<cudaError_t>(
          kernels::launch_sm87_macrofeed_v4_gdn_oracle_cuda(arguments)),
      "launch V4 GDN boundary oracle");
}

[[nodiscard]] bool run_reference(const std::size_t token_count,
                                 const CandidateBuffers& inputs,
                                 ReferenceBuffers& outputs,
                                 cudaStream_t stream) {
  constexpr std::size_t kReferenceTile = 16U;
  for (std::size_t first = 0U; first < token_count;
       first += kReferenceTile) {
    const std::size_t count =
        std::min(kReferenceTile, token_count - first);
    if (!cuda_ok(
            static_cast<cudaError_t>(
                runtime::launch_causal_conv1d_silu_update_tile_reference_cuda(
                    outputs.raw_qkv.data() +
                        first * runtime::kGdnQkvChannels,
                    count, inputs.conv_weight.data(), outputs.history.data(),
                    outputs.conv_qkv.data() +
                        first * runtime::kGdnQkvChannels,
                    {}, stream)),
            "launch independent convolution reference")) {
      return false;
    }
  }
  for (std::size_t first = 0U; first < token_count;
       first += kReferenceTile) {
    const std::size_t count =
        std::min(kReferenceTile, token_count - first);
    if (!cuda_ok(
            static_cast<cudaError_t>(
                runtime::launch_gated_delta_net_update_tile_warp_parallel_cuda(
                    outputs.conv_qkv.data() +
                        first * runtime::kGdnQkvChannels,
                    count,
                    outputs.a.data() + first * runtime::kGdnValueHeadCount,
                    outputs.b.data() + first * runtime::kGdnValueHeadCount,
                    inputs.a_log.data(), inputs.dt_bias.data(),
                    outputs.state.data(), outputs.state.data(), epsilon(),
                    outputs.raw_output.data() +
                        first * runtime::kGdnVElements,
                    {}, stream)),
            "launch independent recurrence reference")) {
      return false;
    }
  }
  return cuda_ok(
      static_cast<cudaError_t>(
          runtime::launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
              outputs.raw_output.data(), inputs.norm_weight.data(),
              outputs.z.data(),
              token_count * runtime::kGdnValueHeadCount,
              runtime::kGdnHeadDimension, epsilon(), outputs.output.data(),
              stream)),
      "launch independent norm/gate reference");
}

[[nodiscard]] bool exact_equal(const std::vector<std::uint16_t>& left,
                               const std::vector<std::uint16_t>& right,
                               const char* label) {
  if (left.size() != right.size()) {
    std::cerr << "FAIL: " << label << " size mismatch\n";
    return false;
  }
  const auto mismatch = std::mismatch(left.begin(), left.end(), right.begin());
  if (mismatch.first == left.end()) {
    return true;
  }
  const std::size_t index =
      static_cast<std::size_t>(mismatch.first - left.begin());
  std::cerr << "FAIL: " << label << " mismatch at " << index
            << " candidate=0x" << std::hex << *mismatch.first
            << " reference=0x" << *mismatch.second << std::dec << '\n';
  return false;
}

[[nodiscard]] bool verify_case(const std::size_t token_count,
                               cudaStream_t stream) {
  const HostFixture fixture = make_fixture(token_count);
  CandidateBuffers candidate;
  ReferenceBuffers reference;
  if (!prepare_candidate(fixture, candidate, stream) ||
      !prepare_reference(fixture, reference, stream) ||
      !cuda_ok(cudaStreamSynchronize(stream), "finish fixture uploads") ||
      !run_candidate(token_count, candidate, stream) ||
      !run_reference(token_count, candidate, reference, stream) ||
      !cuda_ok(cudaStreamSynchronize(stream), "finish GDN oracle case")) {
    return false;
  }

  std::vector<std::uint16_t> candidate_scratch;
  std::vector<std::uint16_t> candidate_history;
  std::vector<std::uint16_t> candidate_state;
  std::vector<std::uint16_t> active_history;
  std::vector<std::uint16_t> active_state;
  std::vector<std::uint16_t> reference_conv;
  std::vector<std::uint16_t> reference_output;
  std::vector<std::uint16_t> reference_history;
  std::vector<std::uint16_t> reference_state;
  if (!candidate.scratch.download(candidate_scratch, stream) ||
      !candidate.candidate_history.download(candidate_history, stream) ||
      !candidate.candidate_state.download(candidate_state, stream) ||
      !candidate.active_history.download(active_history, stream) ||
      !candidate.active_state.download(active_state, stream) ||
      !reference.conv_qkv.download(reference_conv, stream) ||
      !reference.output.download(reference_output, stream) ||
      !reference.history.download(reference_history, stream) ||
      !reference.state.download(reference_state, stream) ||
      !cuda_ok(cudaStreamSynchronize(stream), "download GDN oracle outputs")) {
    return false;
  }

  bool ok = exact_equal(candidate_history, reference_history,
                        "candidate convolution history") &&
            exact_equal(candidate_state, reference_state,
                        "candidate recurrent state") &&
            exact_equal(active_history, fixture.active_history,
                        "const active convolution history") &&
            exact_equal(active_state, fixture.active_state,
                        "const active recurrent state");

  for (std::size_t token = 0U; token < token_count && ok; ++token) {
    const std::size_t scratch_row =
        token * kernels::kSm87MacroFeedV4GdnScratchRowStride;
    const std::size_t conv_row =
        token * kernels::kSm87TargetAotGdnTotalConvChannels;
    const std::size_t output_row =
        token * kernels::kSm87TargetAotGdnOutputChannels;
    for (std::size_t channel = 0U;
         channel < kernels::kSm87TargetAotGdnRawVOffset; ++channel) {
      if (candidate_scratch[scratch_row + channel] !=
          reference_conv[conv_row + channel]) {
        std::cerr << "FAIL: convolved Q/K mismatch token=" << token
                  << " channel=" << channel << '\n';
        ok = false;
        break;
      }
    }
    for (std::size_t channel = 0U;
         channel < kernels::kSm87TargetAotGdnOutputChannels && ok;
         ++channel) {
      if (candidate_scratch[
              scratch_row + kernels::kSm87MacroFeedV4GdnOutputOffset +
              channel] != reference_output[output_row + channel]) {
        std::cerr << "FAIL: in-place V output mismatch token=" << token
                  << " channel=" << channel << '\n';
        ok = false;
        break;
      }
    }
    for (std::size_t channel = kernels::kSm87MacroFeedV4GdnZOffset;
         channel < kernels::kSm87MacroFeedV4GdnScratchRowStride && ok;
         ++channel) {
      if (candidate_scratch[scratch_row + channel] !=
          fixture.scratch[scratch_row + channel]) {
        std::cerr << "FAIL: aliased scratch write escaped V token=" << token
                  << " channel=" << channel << '\n';
        ok = false;
        break;
      }
    }
  }

  const struct {
    const GuardedDeviceBuffer<std::uint16_t>* buffer;
    const char* label;
  } guarded[] = {
      {&candidate.scratch, "candidate scratch"},
      {&candidate.conv_weight, "convolution weight"},
      {&candidate.a_log, "A-log"},
      {&candidate.dt_bias, "dt-bias"},
      {&candidate.norm_weight, "norm weight"},
      {&candidate.active_history, "active history"},
      {&candidate.candidate_history, "candidate history"},
      {&candidate.active_state, "active state"},
      {&candidate.candidate_state, "candidate state"},
      {&reference.raw_qkv, "reference raw QKV"},
      {&reference.a, "reference A"},
      {&reference.b, "reference B"},
      {&reference.z, "reference Z"},
      {&reference.history, "reference history"},
      {&reference.state, "reference state"},
      {&reference.conv_qkv, "reference convolved QKV"},
      {&reference.raw_output, "reference raw output"},
      {&reference.output, "reference output"},
  };
  for (const auto& item : guarded) {
    ok = item.buffer->guards_intact(kGuardBits, stream, item.label) && ok;
  }
  if (ok) {
    std::cout << "PASS: C" << token_count
              << " exact active-to-candidate/in-place oracle\n";
  }
  return ok;
}

[[nodiscard]] bool verify_cancelled_case(cudaStream_t stream) {
  constexpr std::size_t kTokens = 65U;
  constexpr std::uint32_t kCancellationGuard = 0xa5a5'a5a5U;
  const HostFixture fixture = make_fixture(kTokens);
  CandidateBuffers buffers;
  GuardedDeviceBuffer<std::uint32_t> cancellation;
  if (!prepare_candidate(fixture, buffers, stream) ||
      !cancellation.allocate_and_upload(std::vector<std::uint32_t>{1U},
                                       kCancellationGuard, stream) ||
      !cuda_ok(cudaStreamSynchronize(stream),
               "prepare cancellation fixture")) {
    return false;
  }

  kernels::Sm87MacroFeedV4GdnOracleArguments arguments{};
  arguments.scratch = buffers.scratch.data();
  arguments.token_count = kTokens;
  arguments.scratch_row_stride =
      kernels::kSm87MacroFeedV4GdnScratchRowStride;
  arguments.conv_weight = buffers.conv_weight.data();
  arguments.a_log = buffers.a_log.data();
  arguments.dt_bias = buffers.dt_bias.data();
  arguments.norm_weight = buffers.norm_weight.data();
  arguments.active_conv_history = buffers.active_history.data();
  arguments.candidate_conv_history = buffers.candidate_history.data();
  arguments.active_recurrent_state = buffers.active_state.data();
  arguments.candidate_recurrent_state = buffers.candidate_state.data();
  arguments.cancellation_signal = cancellation.data();
  arguments.l2_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.norm_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.cuda_stream = stream;
  if (!cuda_ok(static_cast<cudaError_t>(
                   kernels::launch_sm87_macrofeed_v4_gdn_oracle_cuda(
                       arguments)),
               "launch cancelled V4 GDN oracle") ||
      !cuda_ok(cudaStreamSynchronize(stream),
               "finish cancelled V4 GDN oracle")) {
    return false;
  }

  std::vector<std::uint16_t> scratch;
  std::vector<std::uint16_t> active_history;
  std::vector<std::uint16_t> candidate_history;
  std::vector<std::uint16_t> active_state;
  std::vector<std::uint16_t> candidate_state;
  if (!buffers.scratch.download(scratch, stream) ||
      !buffers.active_history.download(active_history, stream) ||
      !buffers.candidate_history.download(candidate_history, stream) ||
      !buffers.active_state.download(active_state, stream) ||
      !buffers.candidate_state.download(candidate_state, stream) ||
      !cuda_ok(cudaStreamSynchronize(stream),
               "download cancellation outputs")) {
    return false;
  }
  const std::vector<std::uint16_t> poison_state(
      kernels::kSm87TargetAotGdnRecurrentStateElements, kPoisonBits);
  bool ok = exact_equal(scratch, fixture.scratch,
                        "cancelled scratch remains unmodified") &&
            exact_equal(active_history, fixture.active_history,
                        "cancelled active history remains const") &&
            exact_equal(candidate_history, fixture.active_history,
                        "cancelled candidate receives history copy only") &&
            exact_equal(active_state, fixture.active_state,
                        "cancelled active state remains const") &&
            exact_equal(candidate_state, poison_state,
                        "cancelled candidate state remains unpublished") &&
            buffers.scratch.guards_intact(kGuardBits, stream,
                                          "cancelled scratch") &&
            buffers.candidate_history.guards_intact(
                kGuardBits, stream, "cancelled candidate history") &&
            buffers.candidate_state.guards_intact(
                kGuardBits, stream, "cancelled candidate state") &&
            cancellation.guards_intact(kCancellationGuard, stream,
                                       "cancellation signal");
  if (!ok) {
    return false;
  }
  std::cout << "PASS: cancellation leaves active epoch authoritative and "
               "candidate discardable\n";
  return true;
}

[[nodiscard]] bool validate_failure_surface(cudaStream_t stream) {
  constexpr std::size_t kTokens = 1U;
  const HostFixture fixture = make_fixture(kTokens);
  CandidateBuffers buffers;
  if (!prepare_candidate(fixture, buffers, stream) ||
      !cuda_ok(cudaStreamSynchronize(stream),
               "prepare failure-validation fixture")) {
    return false;
  }

  kernels::Sm87MacroFeedV4GdnOracleArguments good{};
  good.scratch = buffers.scratch.data();
  good.token_count = kTokens;
  good.scratch_row_stride = kernels::kSm87MacroFeedV4GdnScratchRowStride;
  good.conv_weight = buffers.conv_weight.data();
  good.a_log = buffers.a_log.data();
  good.dt_bias = buffers.dt_bias.data();
  good.norm_weight = buffers.norm_weight.data();
  good.active_conv_history = buffers.active_history.data();
  good.candidate_conv_history = buffers.candidate_history.data();
  good.active_recurrent_state = buffers.active_state.data();
  good.candidate_recurrent_state = buffers.candidate_state.data();
  good.l2_epsilon_fp32_bits = kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  good.norm_epsilon_fp32_bits = kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  good.cuda_stream = stream;

  const auto rejected = [](const int status) {
    return status == static_cast<int>(cudaErrorInvalidValue);
  };
  bool ok = true;
  const auto admission_plan = kernels::sm87_macrofeed_v4_gdn_c8000_plan(
      kernels::kSm87MacroFeedV4GdnC8000Tokens,
      kernels::kSm87MacroFeedV4GdnScratchRowStride);
  ok = admission_plan.valid() && admission_plan.default_off &&
       admission_plan.startup_package_unbound &&
       !admission_plan.execution_capability &&
       !admission_plan.caller_snapshot_grants_production_authority && ok;
  auto bad = good;
  bad.token_count = 0U;
  ok = rejected(kernels::launch_sm87_macrofeed_v4_gdn_oracle_cuda(bad)) && ok;
  bad = good;
  bad.token_count = kernels::kSm87MacroFeedV4GdnOracleMaximumTokens + 1U;
  ok = rejected(kernels::launch_sm87_macrofeed_v4_gdn_oracle_cuda(bad)) && ok;
  bad = good;
  bad.scratch_row_stride--;
  ok = rejected(kernels::launch_sm87_macrofeed_v4_gdn_oracle_cuda(bad)) && ok;
  bad = good;
  bad.candidate_conv_history =
      const_cast<std::uint16_t*>(bad.active_conv_history);
  ok = rejected(kernels::launch_sm87_macrofeed_v4_gdn_oracle_cuda(bad)) && ok;
  bad = good;
  bad.candidate_recurrent_state =
      const_cast<std::uint16_t*>(bad.active_recurrent_state);
  ok = rejected(kernels::launch_sm87_macrofeed_v4_gdn_oracle_cuda(bad)) && ok;
  bad = good;
  bad.l2_epsilon_fp32_bits ^= 1U;
  ok = rejected(kernels::launch_sm87_macrofeed_v4_gdn_oracle_cuda(bad)) && ok;
  bad = good;
  bad.cuda_stream = nullptr;
  ok = rejected(kernels::launch_sm87_macrofeed_v4_gdn_oracle_cuda(bad)) && ok;

  kernels::Sm87MacroFeedV4GdnC8000Arguments production{};
  production.scratch = good.scratch;
  production.token_count = good.token_count;
  production.scratch_row_stride = good.scratch_row_stride;
  production.conv_weight = good.conv_weight;
  production.a_log = good.a_log;
  production.dt_bias = good.dt_bias;
  production.norm_weight = good.norm_weight;
  production.active_conv_history = good.active_conv_history;
  production.candidate_conv_history = good.candidate_conv_history;
  production.active_recurrent_state = good.active_recurrent_state;
  production.candidate_recurrent_state = good.candidate_recurrent_state;
  production.l2_epsilon_fp32_bits = good.l2_epsilon_fp32_bits;
  production.norm_epsilon_fp32_bits = good.norm_epsilon_fp32_bits;
  production.cuda_stream = good.cuda_stream;
  ok = !kernels::sm87_macrofeed_v4_gdn_c8000_arguments_valid(production) &&
       ok;

  // Host-only admission uses non-dereferenced, disjoint aligned identities so
  // the exact C8000 range contract is tested without allocating a synthetic
  // 278,528,000-byte production scratch plane.
  kernels::Sm87MacroFeedV4GdnC8000Arguments contract{};
  contract.scratch = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x1000'0000'0000ULL));
  contract.token_count = kernels::kSm87MacroFeedV4GdnC8000Tokens;
  contract.scratch_row_stride =
      kernels::kSm87MacroFeedV4GdnScratchRowStride;
  contract.conv_weight = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x2000'0000'0000ULL));
  contract.a_log = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x3000'0000'0000ULL));
  contract.dt_bias = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x4000'0000'0000ULL));
  contract.norm_weight = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x5000'0000'0000ULL));
  contract.active_conv_history = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x6000'0000'0000ULL));
  contract.candidate_conv_history = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x7000'0000'0000ULL));
  contract.active_recurrent_state = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x8000'0000'0000ULL));
  contract.candidate_recurrent_state = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x9000'0000'0000ULL));
  contract.cancellation_signal = reinterpret_cast<const std::uint32_t*>(
      static_cast<std::uintptr_t>(0xa000'0000'0000ULL));
  contract.l2_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  contract.norm_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  contract.cuda_stream = stream;
  ok = kernels::sm87_macrofeed_v4_gdn_c8000_arguments_valid(contract) && ok;
  auto aliased_contract = contract;
  aliased_contract.candidate_recurrent_state =
      const_cast<std::uint16_t*>(aliased_contract.active_recurrent_state);
  ok = !kernels::sm87_macrofeed_v4_gdn_c8000_arguments_valid(
           aliased_contract) &&
       ok;

  kernels::Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot resources{};
  const int resource_query =
      kernels::query_sm87_macrofeed_v4_gdn_c8000_admission_resource_snapshot_cuda(
          &resources);
  if (!cuda_ok(static_cast<cudaError_t>(resource_query),
               "query V4 GDN resources")) {
    return false;
  }
  ok = kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
           resources) &&
       resources.startup_package_unbound &&
       !resources.execution_capability &&
       !resources.caller_snapshot_grants_production_authority &&
       resources.convolution.static_shared_bytes ==
           kernels::kSm87MacroFeedV4GdnConvStaticSharedBytes &&
       resources.recurrence_epilogue.static_shared_bytes ==
           kernels::kSm87MacroFeedV4GdnRecurrenceStaticSharedBytes &&
       ok;
  auto malformed_resources = resources;
  malformed_resources.recurrence_epilogue.local_bytes = 4U;
  malformed_resources.static_resource_gate_passed = true;
  ok = !kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
           malformed_resources) &&
       ok;
  malformed_resources = resources;
  malformed_resources.recurrence_epilogue.active_blocks_per_sm = 2;
  malformed_resources.static_resource_gate_passed = true;
  ok = !kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
           malformed_resources) &&
       ok;
  malformed_resources = resources;
  malformed_resources.convolution.static_shared_bytes += 4U;
  ok = !kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
           malformed_resources) &&
       ok;
  malformed_resources = resources;
  malformed_resources.recurrence_epilogue.static_shared_bytes -= 4U;
  ok = !kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
           malformed_resources) &&
       ok;
  malformed_resources = resources;
  malformed_resources.startup_package_unbound = false;
  ok = !kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
           malformed_resources) &&
       ok;
  malformed_resources = resources;
  malformed_resources.execution_capability = true;
  ok = !kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
           malformed_resources) &&
       ok;
  malformed_resources = resources;
  malformed_resources.caller_snapshot_grants_production_authority = true;
  ok = !kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
           malformed_resources) &&
       ok;

  const auto admission_rejected_as = [](
      const kernels::Sm87MacroFeedV4GdnC8000Arguments& arguments,
      const kernels::Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot&
          snapshot,
      const cudaError_t expected) {
    kernels::Sm87MacroFeedV4GdnC8000AdmissionLaunchReceipt receipt{};
    const int status =
        kernels::launch_sm87_macrofeed_v4_gdn_c8000_admission_cuda(
            arguments, snapshot, &receipt);
    return status == static_cast<int>(expected) &&
           !receipt.valid_enqueue_receipt() &&
           receipt.startup_package_unbound &&
           !receipt.execution_capability &&
           !receipt.caller_snapshot_grants_production_authority;
  };

  auto ordinal_mismatch = resources;
  ++ordinal_mismatch.device_ordinal;
  ok = kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
           ordinal_mismatch) &&
       admission_rejected_as(contract, ordinal_mismatch,
                             cudaErrorInvalidDevice) &&
       ok;
  auto recomputed_snapshot = resources;
  ++recomputed_snapshot.convolution.registers_per_thread;
  ok = kernels::sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
           recomputed_snapshot) &&
       admission_rejected_as(contract, recomputed_snapshot,
                             cudaErrorLaunchOutOfResources) &&
       ok;
  ok = admission_rejected_as(contract, resources,
                             cudaErrorInvalidDevicePointer) &&
       ok;
  alignas(16) std::uint16_t host_scratch[8U]{};
  auto host_pointer_contract = contract;
  host_pointer_contract.scratch = host_scratch;
  ok = kernels::sm87_macrofeed_v4_gdn_c8000_arguments_valid(
           host_pointer_contract) &&
       admission_rejected_as(host_pointer_contract, resources,
                             cudaErrorInvalidDevicePointer) &&
       ok;

  // The first byte belongs to the correct CUDA device, but the one-token test
  // allocation cannot cover the declared 278,528,000-byte C8000 scratch span.
  auto undersized_device_contract = contract;
  undersized_device_contract.scratch = buffers.scratch.data();
  ok = kernels::sm87_macrofeed_v4_gdn_c8000_arguments_valid(
           undersized_device_contract) &&
       admission_rejected_as(undersized_device_contract, resources,
                             cudaErrorInvalidDevicePointer) &&
       ok;

  // The D2D history pair is valid device memory while scratch and the other
  // fake ranges are not.  Pointer ownership must reject before the legal copy
  // can mask an asynchronously invalid kernel enqueue.
  auto mixed_device_contract = host_pointer_contract;
  mixed_device_contract.active_conv_history = buffers.active_history.data();
  mixed_device_contract.candidate_conv_history =
      buffers.candidate_history.data();
  mixed_device_contract.cancellation_signal = nullptr;
  ok = kernels::sm87_macrofeed_v4_gdn_c8000_arguments_valid(
           mixed_device_contract) &&
       admission_rejected_as(mixed_device_contract, resources,
                             cudaErrorInvalidDevicePointer) &&
       ok;

  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count),
               "query wrong-device validation topology")) {
    return false;
  }
  if (device_count > 1) {
    const int current_device = resources.device_ordinal;
    const int other_device = (current_device + 1) % device_count;
    void* other_device_pointer = nullptr;
    cudaStream_t other_device_stream = nullptr;
    bool wrong_device_ready =
        cuda_ok(cudaSetDevice(other_device),
                "select wrong-device validation device") &&
        cuda_ok(cudaMalloc(&other_device_pointer, 16U),
                "allocate wrong-device validation pointer") &&
        cuda_ok(cudaStreamCreateWithFlags(&other_device_stream,
                                         cudaStreamNonBlocking),
                "create wrong-device validation stream") &&
        cuda_ok(cudaSetDevice(current_device),
                "restore fixed validation device");
    if (wrong_device_ready) {
      auto wrong_device_contract = mixed_device_contract;
      wrong_device_contract.scratch =
          static_cast<std::uint16_t*>(other_device_pointer);
      ok = kernels::sm87_macrofeed_v4_gdn_c8000_arguments_valid(
               wrong_device_contract) &&
           admission_rejected_as(wrong_device_contract, resources,
                                 cudaErrorInvalidDevicePointer) &&
           ok;
      auto wrong_stream_contract = contract;
      wrong_stream_contract.cuda_stream = other_device_stream;
      ok = admission_rejected_as(wrong_stream_contract, resources,
                                 cudaErrorInvalidDevice) &&
           ok;
    } else {
      ok = false;
    }
    if (!cuda_ok(cudaSetDevice(other_device),
                 "select wrong-device cleanup device") ||
        (other_device_stream != nullptr &&
         !cuda_ok(cudaStreamDestroy(other_device_stream),
                  "destroy wrong-device validation stream")) ||
        !cuda_ok(cudaFree(other_device_pointer),
                 "free wrong-device validation pointer") ||
        !cuda_ok(cudaSetDevice(current_device),
                 "restore device after wrong-device cleanup")) {
      return false;
    }
  }

  kernels::Sm87MacroFeedV4GdnC8000AdmissionLaunchReceipt receipt_contract{};
  receipt_contract.identity = kernels::kSm87MacroFeedV4GdnC8000Identity;
  receipt_contract.token_count = kernels::kSm87MacroFeedV4GdnC8000Tokens;
  receipt_contract.scratch_row_stride =
      kernels::kSm87MacroFeedV4GdnScratchRowStride;
  receipt_contract.conv_history_copy_bytes =
      kernels::kSm87MacroFeedV4GdnConvHistoryBytes;
  receipt_contract.whole_recurrent_epoch_copy_bytes = 0U;
  receipt_contract.physical_kernel_launches = static_cast<std::uint32_t>(
      kernels::kSm87MacroFeedV4GdnPhysicalKernelLaunches);
  receipt_contract.asynchronous_d2d_copies = static_cast<std::uint32_t>(
      kernels::kSm87MacroFeedV4GdnHistoryCopies);
  receipt_contract.active_state_const = true;
  receipt_contract.candidate_state_full_assignment_required = true;
  receipt_contract.in_place_qkv_convolution = true;
  receipt_contract.in_place_v_output = true;
  receipt_contract.current_device_revalidated = true;
  receipt_contract.caller_snapshot_exact_observed_match = true;
  receipt_contract.caller_supplied_live_stream_required = true;
  receipt_contract.live_stream_device_observed = true;
  receipt_contract.device_allocation_ranges_owned = true;
  receipt_contract.launch_enqueued = true;
  receipt_contract.numerical_contract_qualified = false;
  receipt_contract.production_dispatch_eligible = false;
  receipt_contract.startup_package_unbound = true;
  receipt_contract.execution_capability = false;
  receipt_contract.caller_snapshot_grants_production_authority = false;
  ok = receipt_contract.valid_enqueue_receipt() &&
       receipt_contract.startup_package_unbound &&
       !receipt_contract.execution_capability &&
       !receipt_contract.caller_snapshot_grants_production_authority && ok;
  auto authority_claiming_receipt = receipt_contract;
  authority_claiming_receipt.execution_capability = true;
  ok = !authority_claiming_receipt.valid_enqueue_receipt() && ok;
  auto unvalidated_receipt = receipt_contract;
  unvalidated_receipt.current_device_revalidated = false;
  ok = !unvalidated_receipt.valid_enqueue_receipt() && ok;
  unvalidated_receipt = receipt_contract;
  unvalidated_receipt.caller_snapshot_exact_observed_match = false;
  ok = !unvalidated_receipt.valid_enqueue_receipt() && ok;
  unvalidated_receipt = receipt_contract;
  unvalidated_receipt.caller_supplied_live_stream_required = false;
  ok = !unvalidated_receipt.valid_enqueue_receipt() && ok;
  unvalidated_receipt = receipt_contract;
  unvalidated_receipt.live_stream_device_observed = false;
  ok = !unvalidated_receipt.valid_enqueue_receipt() && ok;
  unvalidated_receipt = receipt_contract;
  unvalidated_receipt.device_allocation_ranges_owned = false;
  ok = !unvalidated_receipt.valid_enqueue_receipt() && ok;

  ok = admission_rejected_as(production, resources, cudaErrorInvalidValue) &&
       ok;
  (void)cudaGetLastError();

  if (!ok) {
    std::cerr << "FAIL: V4 GDN failure/resource validation\n";
    return false;
  }
  std::cout << "PASS: failure/resource validation; conv_regs="
            << resources.convolution.registers_per_thread
            << " conv_shared=" << resources.convolution.static_shared_bytes
            << " recurrence_regs="
            << resources.recurrence_epilogue.registers_per_thread
            << " recurrence_shared="
            << resources.recurrence_epilogue.static_shared_bytes << '\n';
  return true;
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
               "get CUDA device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(kernels::kSm87MacroFeedV4GdnSmCount)) {
    std::cout << "SKIP: exact SM87/16-SM target unavailable\n";
    return 77;
  }

  cudaStream_t stream = nullptr;
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create CUDA stream")) {
    return 1;
  }
  const bool ok = validate_failure_surface(stream) &&
                  verify_case(1U, stream) && verify_case(65U, stream) &&
                  verify_cancelled_case(stream);
  const bool destroyed = cuda_ok(cudaStreamDestroy(stream),
                                 "destroy CUDA stream");
  if (!ok || !destroyed) {
    return 1;
  }
  std::cout << "PASS: MacroFeed-v4 C8000 exact GDN constituent\n";
  return 0;
}
