#include "q3x/kernels/sm87_macrofeed_v4_full_attention_preprocess.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

constexpr std::size_t kGuardElements = 32U;
constexpr std::uint16_t kGuardBits = 0x5a5aU;
constexpr std::uint16_t kKeySentinelBits = 0x3d80U;
constexpr std::uint16_t kValueSentinelBits = 0x4242U;
constexpr std::uint16_t kGapSentinelBits = 0x7fc1U;

[[nodiscard]] bool expect(const bool condition,
                          const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const std::string& operation) {
  return expect(status == cudaSuccess,
                operation + ": " + cudaGetErrorString(status));
}

template <class T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] bool allocate(const std::size_t elements) noexcept {
    if (elements == 0U || pointer_ != nullptr ||
        cudaMalloc(reinterpret_cast<void**>(&pointer_),
                   elements * sizeof(T)) != cudaSuccess) {
      return false;
    }
    elements_ = elements;
    return true;
  }

  [[nodiscard]] bool zero(const cudaStream_t stream) noexcept {
    return pointer_ != nullptr &&
           cudaMemsetAsync(pointer_, 0, elements_ * sizeof(T), stream) ==
               cudaSuccess;
  }

  [[nodiscard]] bool upload(const std::vector<T>& host,
                            const cudaStream_t stream) noexcept {
    return pointer_ != nullptr && host.size() == elements_ &&
           cudaMemcpyAsync(pointer_, host.data(), elements_ * sizeof(T),
                           cudaMemcpyHostToDevice, stream) == cudaSuccess;
  }

  [[nodiscard]] bool download(std::vector<T>& host,
                              const cudaStream_t stream) const {
    host.resize(elements_);
    return pointer_ != nullptr &&
           cudaMemcpyAsync(host.data(), pointer_, elements_ * sizeof(T),
                           cudaMemcpyDeviceToHost, stream) == cudaSuccess;
  }

  [[nodiscard]] T* data() noexcept { return pointer_; }
  [[nodiscard]] const T* data() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t size() const noexcept { return elements_; }

 private:
  T* pointer_ = nullptr;
  std::size_t elements_ = 0U;
};

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
                                         const cudaStream_t stream) {
    if (logical.empty() || base_ != nullptr) {
      return false;
    }
    logical_elements_ = logical.size();
    const std::size_t total = logical_elements_ + 2U * kGuardElements;
    if (cudaMalloc(reinterpret_cast<void**>(&base_),
                   total * sizeof(T)) != cudaSuccess) {
      return false;
    }
    std::vector<T> host(total, guard);
    std::copy(logical.begin(), logical.end(),
              host.begin() + static_cast<std::ptrdiff_t>(kGuardElements));
    return cudaMemcpyAsync(base_, host.data(), total * sizeof(T),
                           cudaMemcpyHostToDevice, stream) == cudaSuccess;
  }

  [[nodiscard]] bool download(std::vector<T>& logical,
                              const cudaStream_t stream) const {
    logical.resize(logical_elements_);
    return cudaMemcpyAsync(logical.data(), data(),
                           logical_elements_ * sizeof(T),
                           cudaMemcpyDeviceToHost, stream) == cudaSuccess;
  }

  [[nodiscard]] bool guards_intact(const T guard) const {
    std::array<T, kGuardElements> prefix{};
    std::array<T, kGuardElements> suffix{};
    if (cudaMemcpy(prefix.data(), base_, sizeof(prefix),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(suffix.data(), data() + logical_elements_,
                   sizeof(suffix), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    return std::all_of(prefix.begin(), prefix.end(),
                       [guard](const T value) { return value == guard; }) &&
           std::all_of(suffix.begin(), suffix.end(),
                       [guard](const T value) { return value == guard; });
  }

  [[nodiscard]] T* data() noexcept { return base_ + kGuardElements; }
  [[nodiscard]] const T* data() const noexcept {
    return base_ + kGuardElements;
  }

 private:
  T* base_ = nullptr;
  std::size_t logical_elements_ = 0U;
};

class ManagedBuffer final {
 public:
  ManagedBuffer() = default;
  ~ManagedBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }
  ManagedBuffer(const ManagedBuffer&) = delete;
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  [[nodiscard]] bool allocate(const std::size_t bytes) noexcept {
    return bytes != 0U && pointer_ == nullptr &&
           cudaMallocManaged(&pointer_, bytes) == cudaSuccess;
  }
  [[nodiscard]] void* data() noexcept { return pointer_; }

 private:
  void* pointer_ = nullptr;
};

[[nodiscard]] std::uint16_t patterned_bf16(const std::size_t index) {
  constexpr std::array<std::uint16_t, 20U> values{{
      0x0000U, 0x3c00U, 0xbc00U, 0x3c80U, 0xbc80U,
      0x3d00U, 0xbd00U, 0x3d80U, 0xbd80U, 0x3e00U,
      0xbe00U, 0x3e80U, 0xbe80U, 0x3f00U, 0xbf00U,
      0x3f40U, 0xbf40U, 0x3f80U, 0xbf80U, 0x4000U,
  }};
  return values[(index * 29U + index / 7U + 3U) % values.size()];
}

[[nodiscard]] bool host_contract_test() {
  bool ok = true;
  constexpr std::array<std::size_t, 5U> first_positions{{
      0U, 8'000U, 16'000U, 24'000U, 32'000U,
  }};
  for (const std::size_t first_position : first_positions) {
    const auto plan =
        kernels::sm87_macrofeed_v4_full_attention_preprocess_plan(
            first_position,
            kernels::kSm87MacroFeedV4FullAttentionPreprocessTokens);
    ok &= expect(plan.valid() && plan.q_in_place && plan.k_in_place &&
                     plan.gate_bitwise_preserved &&
                     plan.scratch_gap_bitwise_preserved &&
                     plan.private_nhd_key_cache &&
                     !plan.value_cache_addressable && plan.default_off &&
                     !plan.selector_present && !plan.fallback_permitted &&
                     !plan.numerical_contract_qualified &&
                     !plan.production_dispatch_eligible &&
                     plan.startup_package_unbound &&
                     !plan.execution_capability &&
                     !plan.caller_snapshot_grants_production_authority,
                 "C8000 plan contract failed for first_position=" +
                     std::to_string(first_position));
  }
  ok &= expect(
      !kernels::sm87_macrofeed_v4_full_attention_preprocess_plan(1U, 8'000U)
           .valid() &&
          !kernels::sm87_macrofeed_v4_full_attention_preprocess_plan(
               40'000U, 8'000U)
               .valid() &&
          !kernels::sm87_macrofeed_v4_full_attention_preprocess_plan(
               0U, 7'999U)
               .valid(),
      "invalid panel shapes must fail closed");

  auto* const scratch = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'1000'0000'0000ULL));
  auto* const key = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'2000'0000'0000ULL));
  const auto* const q_weight = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'3000'0000'0000ULL));
  const auto* const k_weight = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'4000'0000'0000ULL));
  const auto* const cosines = reinterpret_cast<const float*>(
      static_cast<std::uintptr_t>(0x0000'5000'0000'0000ULL));
  const auto* const sines = reinterpret_cast<const float*>(
      static_cast<std::uintptr_t>(0x0000'6000'0000'0000ULL));
  kernels::Sm87MacroFeedV4FullAttentionPreprocessArguments arguments{
      scratch,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessTokens,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride,
      key,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride,
      q_weight,
      k_weight,
      cosines,
      sines,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf,
      32'000U,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessEpsilonFp32Bits,
      reinterpret_cast<void*>(0x1000U)};
  ok &= expect(
      kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          arguments),
      "structural seam must accept coherent host-only fake ranges");
  auto changed = arguments;
  changed.first_position = 1U;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          changed),
      "unaligned first_position must fail closed");
  changed = arguments;
  changed.token_count = 7'999U;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          changed),
      "non-C8000 admission extent must fail closed");
  changed = arguments;
  changed.q_norm_weight = changed.k_norm_weight;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          changed),
      "aliased norm weights must fail closed");
  changed = arguments;
  changed.epsilon_fp32_bits ^= 1U;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          changed),
      "noncanonical epsilon must fail closed");
  changed = arguments;
  changed.cuda_stream = nullptr;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          changed),
      "null caller stream must fail closed");
  return ok;
}

struct OracleHostFixture final {
  std::vector<std::uint16_t> scratch;
  std::vector<std::uint16_t> key;
  std::vector<std::uint16_t> value;
  std::vector<std::uint16_t> q_weight;
  std::vector<std::uint16_t> k_weight;
  std::vector<float> cosines;
  std::vector<float> sines;
};

[[nodiscard]] OracleHostFixture make_oracle_fixture(
    const std::size_t first_position,
    const std::size_t token_count) {
  OracleHostFixture fixture;
  fixture.scratch.assign(
      token_count *
          kernels::kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride,
      kGapSentinelBits);
  fixture.key.assign(
      kernels::kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions *
          kernels::kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride,
      kKeySentinelBits);
  fixture.value.assign(fixture.key.size(), kValueSentinelBits);
  fixture.q_weight.resize(
      kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension);
  fixture.k_weight.resize(fixture.q_weight.size());
  fixture.cosines.assign(
      kernels::kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions *
          kernels::kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf,
      0.0F);
  fixture.sines.assign(fixture.cosines.size(), 0.0F);

  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t row =
        token *
        kernels::kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride;
    for (std::size_t head = 0U;
         head <
         kernels::kSm87MacroFeedV4FullAttentionPreprocessQueryHeads;
         ++head) {
      const std::size_t head_base =
          row + head *
                    kernels::
                        kSm87MacroFeedV4FullAttentionPreprocessQGateHeadStride;
      for (std::size_t dimension = 0U;
           dimension <
           kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
           ++dimension) {
        fixture.scratch[head_base + dimension] = patterned_bf16(
            token * 131U + head * 37U + dimension * 11U);
        fixture.scratch[
            head_base +
            kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension +
            dimension] = static_cast<std::uint16_t>(
            0x4100U + ((token * 17U + head * 13U + dimension) & 0x007fU));
      }
    }
    for (std::size_t gap =
             kernels::kSm87MacroFeedV4FullAttentionPreprocessQGateSpan;
         gap <
         kernels::kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride;
         ++gap) {
      fixture.scratch[row + gap] = static_cast<std::uint16_t>(
          kGapSentinelBits ^ ((token + gap) & 0x003fU));
    }
    const std::size_t key_row =
        (first_position + token) *
        kernels::kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride;
    for (std::size_t head = 0U;
         head <
         kernels::kSm87MacroFeedV4FullAttentionPreprocessLogicalKvHeads;
         ++head) {
      for (std::size_t dimension = 0U;
           dimension <
           kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
           ++dimension) {
        fixture.key[key_row +
                    head * kernels::
                               kSm87MacroFeedV4FullAttentionPreprocessHeadDimension +
                    dimension] = patterned_bf16(
            token * 193U + head * 43U + dimension * 7U + 5U);
      }
    }
  }
  for (std::size_t dimension = 0U;
       dimension <
       kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
       ++dimension) {
    fixture.q_weight[dimension] = patterned_bf16(dimension * 3U + 17U);
    fixture.k_weight[dimension] = patterned_bf16(dimension * 5U + 29U);
  }
  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t table_row =
        (first_position + token) *
        kernels::kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf;
    for (std::size_t dimension = 0U;
         dimension <
         kernels::kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf;
         ++dimension) {
      const float angle = static_cast<float>(
                              (first_position + token + 1U) *
                                  (dimension + 3U)) *
                          0.000'013F;
      fixture.cosines[table_row + dimension] = std::cos(angle);
      fixture.sines[table_row + dimension] = std::sin(angle);
    }
  }
  return fixture;
}

[[nodiscard]] bool run_oracle_case(const std::size_t first_position,
                                   const std::size_t token_count,
                                   const cudaStream_t stream) {
  const std::string label = "C" + std::to_string(token_count) +
                            " first=" + std::to_string(first_position);
  OracleHostFixture fixture = make_oracle_fixture(first_position, token_count);
  const auto original_scratch = fixture.scratch;
  const auto original_key = fixture.key;
  const auto original_value = fixture.value;

  GuardedDeviceBuffer<std::uint16_t> candidate_scratch;
  GuardedDeviceBuffer<std::uint16_t> reference_scratch;
  GuardedDeviceBuffer<std::uint16_t> candidate_key;
  GuardedDeviceBuffer<std::uint16_t> reference_key;
  GuardedDeviceBuffer<std::uint16_t> value;
  DeviceBuffer<std::uint16_t> q_weight;
  DeviceBuffer<std::uint16_t> k_weight;
  DeviceBuffer<float> cosines;
  DeviceBuffer<float> sines;
  bool ready = candidate_scratch.allocate_and_upload(
      fixture.scratch, kGuardBits, stream);
  ready = reference_scratch.allocate_and_upload(fixture.scratch, kGuardBits,
                                                stream) &&
          ready;
  ready = candidate_key.allocate_and_upload(fixture.key, kGuardBits, stream) &&
          ready;
  ready = reference_key.allocate_and_upload(fixture.key, kGuardBits, stream) &&
          ready;
  ready = value.allocate_and_upload(fixture.value, kGuardBits, stream) && ready;
  ready = q_weight.allocate(fixture.q_weight.size()) &&
          q_weight.upload(fixture.q_weight, stream) && ready;
  ready = k_weight.allocate(fixture.k_weight.size()) &&
          k_weight.upload(fixture.k_weight, stream) && ready;
  ready = cosines.allocate(fixture.cosines.size()) &&
          cosines.upload(fixture.cosines, stream) && ready;
  ready = sines.allocate(fixture.sines.size()) &&
          sines.upload(fixture.sines, stream) && ready;
  if (!expect(ready, label + " device fixture setup failed")) {
    return false;
  }

  kernels::Sm87MacroFeedV4FullAttentionPreprocessOracleArguments candidate{
      candidate_scratch.data(),
      token_count,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride,
      candidate_key.data(),
      kernels::kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride,
      q_weight.data(),
      k_weight.data(),
      cosines.data(),
      sines.data(),
      kernels::kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf,
      first_position,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessEpsilonFp32Bits,
      stream};
  auto reference = candidate;
  reference.q_gate_scratch = reference_scratch.data();
  reference.key_cache = reference_key.data();
  ready = cuda_ok(static_cast<cudaError_t>(
                      kernels::
                          launch_sm87_macrofeed_v4_full_attention_preprocess_candidate_128_oracle_cuda(
                              candidate)),
                  label + " candidate launch") &&
          ready;
  ready = cuda_ok(static_cast<cudaError_t>(
                      kernels::
                          launch_sm87_macrofeed_v4_full_attention_preprocess_reference_256_oracle_cuda(
                              reference)),
                  label + " independent reference launch") &&
          ready;
  ready = cuda_ok(cudaStreamSynchronize(stream), label + " synchronize") &&
          ready;
  if (!ready) {
    return false;
  }

  std::vector<std::uint16_t> candidate_scratch_host;
  std::vector<std::uint16_t> reference_scratch_host;
  std::vector<std::uint16_t> candidate_key_host;
  std::vector<std::uint16_t> reference_key_host;
  std::vector<std::uint16_t> value_host;
  ready = candidate_scratch.download(candidate_scratch_host, stream);
  ready = reference_scratch.download(reference_scratch_host, stream) && ready;
  ready = candidate_key.download(candidate_key_host, stream) && ready;
  ready = reference_key.download(reference_key_host, stream) && ready;
  ready = value.download(value_host, stream) && ready;
  ready = cuda_ok(cudaStreamSynchronize(stream), label + " download sync") &&
          ready;
  if (!expect(ready, label + " download failed")) {
    return false;
  }

  bool ok = true;
  ok &= expect(candidate_scratch_host == reference_scratch_host,
               label + " Q scratch differs from 256-thread exact tree");
  ok &= expect(candidate_key_host == reference_key_host,
               label + " K cache differs from 256-thread exact tree");
  ok &= expect(value_host == original_value,
               label + " unaddressable V cache changed");

  std::array<bool,
             kernels::kSm87MacroFeedV4FullAttentionPreprocessQueryHeads>
      q_head_changed{};
  std::array<bool,
             kernels::kSm87MacroFeedV4FullAttentionPreprocessLogicalKvHeads>
      k_head_changed{};
  bool gates_preserved = true;
  bool scratch_gaps_preserved = true;
  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t scratch_row =
        token *
        kernels::kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride;
    for (std::size_t head = 0U;
         head <
         kernels::kSm87MacroFeedV4FullAttentionPreprocessQueryHeads;
         ++head) {
      const std::size_t head_base =
          scratch_row +
          head * kernels::
                     kSm87MacroFeedV4FullAttentionPreprocessQGateHeadStride;
      for (std::size_t dimension = 0U;
           dimension <
           kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
           ++dimension) {
        q_head_changed[head] =
            q_head_changed[head] ||
            candidate_scratch_host[head_base + dimension] !=
                original_scratch[head_base + dimension];
        const std::size_t gate_index =
            head_base +
            kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension +
            dimension;
        gates_preserved =
            gates_preserved &&
            candidate_scratch_host[gate_index] ==
                original_scratch[gate_index];
      }
    }
    for (std::size_t gap =
             kernels::kSm87MacroFeedV4FullAttentionPreprocessQGateSpan;
         gap <
         kernels::kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride;
         ++gap) {
      scratch_gaps_preserved =
          scratch_gaps_preserved &&
          candidate_scratch_host[scratch_row + gap] ==
              original_scratch[scratch_row + gap];
    }
    const std::size_t key_row =
        (first_position + token) *
        kernels::kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride;
    for (std::size_t head = 0U;
         head <
         kernels::kSm87MacroFeedV4FullAttentionPreprocessLogicalKvHeads;
         ++head) {
      for (std::size_t dimension = 0U;
           dimension <
           kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
           ++dimension) {
        const std::size_t index =
            key_row +
            head * kernels::
                       kSm87MacroFeedV4FullAttentionPreprocessHeadDimension +
            dimension;
        k_head_changed[head] =
            k_head_changed[head] ||
            candidate_key_host[index] != original_key[index];
      }
    }
  }
  for (std::size_t head = 0U; head < q_head_changed.size(); ++head) {
    ok &= expect(q_head_changed[head],
                 label + " Q head " + std::to_string(head) +
                     " was not exercised");
  }
  for (std::size_t head = 0U; head < k_head_changed.size(); ++head) {
    ok &= expect(k_head_changed[head],
                 label + " K logical head " + std::to_string(head) +
                     " was not exercised");
  }
  ok &= expect(gates_preserved,
               label + " Gate sentinel changed");
  ok &= expect(scratch_gaps_preserved,
               label + " scratch row-gap sentinel changed");

  const std::size_t first_key_element =
      first_position *
      kernels::kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride;
  const std::size_t end_key_element =
      (first_position + token_count) *
      kernels::kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride;
  ok &= expect(std::equal(candidate_key_host.begin(),
                          candidate_key_host.begin() +
                              static_cast<std::ptrdiff_t>(first_key_element),
                          original_key.begin()),
               label + " K prefix outside current panel changed");
  ok &= expect(std::equal(
                   candidate_key_host.begin() +
                       static_cast<std::ptrdiff_t>(end_key_element),
                   candidate_key_host.end(),
                   original_key.begin() +
                       static_cast<std::ptrdiff_t>(end_key_element)),
               label + " K suffix outside current panel changed");
  ok &= expect(candidate_scratch.guards_intact(kGuardBits) &&
                   reference_scratch.guards_intact(kGuardBits) &&
                   candidate_key.guards_intact(kGuardBits) &&
                   reference_key.guards_intact(kGuardBits) &&
                   value.guards_intact(kGuardBits),
               label + " allocation guard changed");
  return ok;
}

[[nodiscard]] bool admission_test(const cudaStream_t stream) {
  bool ok = true;
  kernels::Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot
      resources{};
  const int resource_status = kernels::
      query_sm87_macrofeed_v4_full_attention_preprocess_admission_resources_cuda(
          &resources);
  ok &= expect(
      resource_status == static_cast<int>(cudaSuccess) &&
          kernels::
              sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
                  resources) &&
          !resources.numerical_contract_qualified &&
          !resources.production_dispatch_eligible &&
          resources.startup_package_unbound &&
          !resources.execution_capability &&
          !resources.caller_snapshot_grants_production_authority,
      "admission resource query failed");
  if (!ok) {
    return false;
  }

  DeviceBuffer<std::uint16_t> scratch;
  DeviceBuffer<std::uint16_t> key;
  DeviceBuffer<std::uint16_t> q_weight;
  DeviceBuffer<std::uint16_t> k_weight;
  DeviceBuffer<float> cosines;
  DeviceBuffer<float> sines;
  DeviceBuffer<std::uint16_t> undersized_scratch;
  ManagedBuffer managed_weight;
  bool ready = scratch.allocate(
      kernels::kSm87MacroFeedV4FullAttentionPreprocessScratchBytes /
      sizeof(std::uint16_t));
  ready = key.allocate(
              kernels::kSm87MacroFeedV4FullAttentionPreprocessKeyCacheBytes /
              sizeof(std::uint16_t)) &&
          ready;
  ready = q_weight.allocate(
              kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension) &&
          ready;
  ready = k_weight.allocate(
              kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension) &&
          ready;
  ready = cosines.allocate(
              kernels::kSm87MacroFeedV4FullAttentionPreprocessRopeTableBytes /
              sizeof(float)) &&
          ready;
  ready = sines.allocate(
              kernels::kSm87MacroFeedV4FullAttentionPreprocessRopeTableBytes /
              sizeof(float)) &&
          ready;
  ready = undersized_scratch.allocate(128U) && ready;
  ready = managed_weight.allocate(
              kernels::
                  kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes) &&
          ready;
  ready = scratch.zero(stream) && key.zero(stream) && q_weight.zero(stream) &&
          k_weight.zero(stream) && cosines.zero(stream) &&
          sines.zero(stream) && ready;
  ready = cuda_ok(cudaStreamSynchronize(stream),
                  "admission fixture initialization") &&
          ready;
  if (!expect(ready, "admission device fixture setup failed")) {
    return false;
  }

  kernels::Sm87MacroFeedV4FullAttentionPreprocessArguments base{
      scratch.data(),
      kernels::kSm87MacroFeedV4FullAttentionPreprocessTokens,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride,
      key.data(),
      kernels::kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride,
      q_weight.data(),
      k_weight.data(),
      cosines.data(),
      sines.data(),
      kernels::kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf,
      32'000U,
      kernels::kSm87MacroFeedV4FullAttentionPreprocessEpsilonFp32Bits,
      stream};
  ok &= expect(
      kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          base),
      "production admission arguments must be structurally valid");

  kernels::Sm87MacroFeedV4FullAttentionPreprocessAdmissionLaunchReceipt
      receipt{};
  const int launch_status = kernels::
      launch_sm87_macrofeed_v4_full_attention_preprocess_admission_cuda(
          base, resources, &receipt);
  ok &= expect(launch_status == static_cast<int>(cudaSuccess) &&
                   receipt.valid_enqueue_receipt() &&
                   receipt.private_nhd_key_cache &&
                   receipt.value_cache_unaddressable &&
                   !receipt.stream_owner_verified,
               "positive C8000 admission enqueue failed");
  ok &= cuda_ok(cudaStreamSynchronize(stream),
                "positive C8000 admission synchronize");
  auto elevated_receipt = receipt;
  elevated_receipt.execution_capability = true;
  ok &= expect(!elevated_receipt.valid_enqueue_receipt(),
               "execution-capability receipt claim must fail closed");
  elevated_receipt = receipt;
  elevated_receipt.startup_package_unbound = false;
  ok &= expect(!elevated_receipt.valid_enqueue_receipt(),
               "startup-bound receipt claim must fail closed");
  ok &= expect(
      kernels::
          launch_sm87_macrofeed_v4_full_attention_preprocess_admission_cuda(
              base, resources, nullptr) ==
          static_cast<int>(cudaErrorInvalidValue),
      "null receipt must fail closed");
  if (!ok) {
    return false;
  }

  const auto expect_rejected = [&ok](
                                   const char* const label,
                                   const kernels::
                                       Sm87MacroFeedV4FullAttentionPreprocessArguments&
                                           arguments,
                                   const kernels::
                                       Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&
                                           snapshot) {
    kernels::Sm87MacroFeedV4FullAttentionPreprocessAdmissionLaunchReceipt
        rejected_receipt{};
    const int status = kernels::
        launch_sm87_macrofeed_v4_full_attention_preprocess_admission_cuda(
            arguments, snapshot, &rejected_receipt);
    ok &= expect(status == static_cast<int>(cudaErrorInvalidValue) &&
                     !rejected_receipt.launch_enqueued &&
                     rejected_receipt.physical_kernel_launches == 0U &&
                     !rejected_receipt.valid_enqueue_receipt(),
                 std::string(label) + " must return an empty receipt");
    (void)cudaGetLastError();
  };

  auto forged = resources;
  ++forged.kernel.registers_per_thread;
  ok &= expect(kernels::
                   sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
                       forged),
               "forged register snapshot must remain structurally admitted");
  expect_rejected("forged register resource snapshot", base, forged);
  forged = resources;
  ++forged.kernel.active_blocks_per_sm;
  ok &= expect(kernels::
                   sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
                       forged),
               "forged occupancy snapshot must remain structurally admitted");
  expect_rejected("forged occupancy resource snapshot", base, forged);
  forged = resources;
  ++forged.device_ordinal;
  ok &= expect(kernels::
                   sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
                       forged),
               "ordinal-mismatched snapshot must remain structurally admitted");
  expect_rejected("snapshot/current-device ordinal mismatch", base, forged);
  forged = resources;
  forged.startup_package_unbound = false;
  expect_rejected("forged startup-bound authority", base, forged);
  forged = resources;
  forged.execution_capability = true;
  expect_rejected("forged execution capability", base, forged);
  forged = resources;
  forged.caller_snapshot_grants_production_authority = true;
  expect_rejected("forged caller production authority", base, forged);
  forged = resources;
  forged.production_dispatch_eligible = true;
  expect_rejected("forged production eligibility", base, forged);

  alignas(256) std::array<std::uint16_t, 128U> host_scratch{};
  auto invalid = base;
  invalid.q_gate_scratch = host_scratch.data();
  ok &= expect(
      kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          invalid),
      "host scratch negative must reach live device validation");
  expect_rejected("host scratch", invalid, resources);

  invalid = base;
  invalid.key_cache = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'2000'0000'0000ULL));
  ok &= expect(
      kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          invalid),
      "fake key negative must reach live device validation");
  expect_rejected("fake key cache", invalid, resources);

  invalid = base;
  invalid.q_gate_scratch = undersized_scratch.data();
  invalid.key_cache = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'2000'0000'0000ULL));
  invalid.q_norm_weight = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'3000'0000'0000ULL));
  invalid.k_norm_weight = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'4000'0000'0000ULL));
  invalid.cosines = reinterpret_cast<const float*>(
      static_cast<std::uintptr_t>(0x0000'5000'0000'0000ULL));
  invalid.sines = reinterpret_cast<const float*>(
      static_cast<std::uintptr_t>(0x0000'6000'0000'0000ULL));
  ok &= expect(
      kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          invalid),
      "undersized scratch negative must reach full-range validation");
  expect_rejected("undersized current-device scratch", invalid, resources);

  invalid = base;
  invalid.q_norm_weight =
      reinterpret_cast<const std::uint16_t*>(managed_weight.data());
  ok &= expect(
      kernels::sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          invalid),
      "managed-weight negative must reach allocation-kind validation");
  expect_rejected("managed norm weight", invalid, resources);

  invalid = base;
  invalid.cuda_stream = nullptr;
  expect_rejected("null caller stream", invalid, resources);
  invalid = base;
  invalid.first_position = 1U;
  expect_rejected("invalid first_position", invalid, resources);
  return ok;
}

}  // namespace

int main() {
  bool ok = host_contract_test();
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "sm87_macrofeed_v4_full_attention_preprocess_cuda_test: "
                 "SKIP (CUDA unavailable)\n";
    return ok ? 0 : 1;
  }
  int device = -1;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
      properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(
              kernels::kSm87MacroFeedV4FullAttentionPreprocessSmCount)) {
    std::cout << "sm87_macrofeed_v4_full_attention_preprocess_cuda_test: "
                 "SKIP (requires SM87/16SM)\n";
    return ok ? 0 : 1;
  }

  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    return 1;
  }
  kernels::Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot
      resources{};
  const int query_status = kernels::
      query_sm87_macrofeed_v4_full_attention_preprocess_admission_resources_cuda(
          &resources);
  ok &= expect(query_status == static_cast<int>(cudaSuccess),
               "resource query before oracle tests failed");
  if (query_status == static_cast<int>(cudaSuccess)) {
    std::cout << "Full-Attention preprocess resources: regs="
              << resources.kernel.registers_per_thread
              << " shared=" << resources.kernel.static_shared_bytes
              << " local=" << resources.kernel.local_bytes
              << " active_cta_per_sm="
              << resources.kernel.active_blocks_per_sm << '\n';
  }
  ok &= run_oracle_case(0U, 1U, stream);
  ok &= run_oracle_case(32'000U, 65U, stream);
  ok &= admission_test(stream);
  ok &= expect(cudaStreamDestroy(stream) == cudaSuccess,
               "stream destruction failed");
  if (ok) {
    std::cout
        << "sm87_macrofeed_v4_full_attention_preprocess_cuda_test: PASS\n";
  }
  return ok ? 0 : 1;
}
