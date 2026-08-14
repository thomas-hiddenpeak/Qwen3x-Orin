#include "q3x/kernels/sm87_macrofeed_v4_attention_c8000.h"
#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using q3x::kernels::Sm87MacroFeedV4AttentionC8000AdmissionLaunchReceipt;
using q3x::kernels::Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot;
using q3x::kernels::Sm87MacroFeedV4AttentionC8000Arguments;
using q3x::kernels::Sm87MacroFeedV4AttentionC8000OracleArguments;

bool expect(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

bool expect_cuda(const cudaError_t status, const std::string_view message) {
  if (status != cudaSuccess) {
    std::cerr << "FAIL: " << message << ": "
              << cudaGetErrorString(status) << '\n';
    return false;
  }
  return true;
}

std::uint16_t encode_bf16(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

template <class T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(const std::size_t count) { allocate(count); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)),
        count_(std::exchange(other.count_, 0U)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      pointer_ = std::exchange(other.pointer_, nullptr);
      count_ = std::exchange(other.count_, 0U);
    }
    return *this;
  }
  ~DeviceBuffer() { release(); }

  bool allocate(const std::size_t count) {
    release();
    if (count == 0U) {
      return false;
    }
    count_ = count;
    if (cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)) !=
        cudaSuccess) {
      pointer_ = nullptr;
      count_ = 0U;
      return false;
    }
    return true;
  }

  void release() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
      pointer_ = nullptr;
      count_ = 0U;
    }
  }

  T* get() const noexcept { return pointer_; }
  std::size_t count() const noexcept { return count_; }

 private:
  T* pointer_ = nullptr;
  std::size_t count_ = 0U;
};

bool copy_to_device(DeviceBuffer<std::uint16_t>& destination,
                    const std::vector<std::uint16_t>& source) {
  return destination.count() == source.size() &&
         expect_cuda(cudaMemcpy(destination.get(), source.data(),
                                source.size() * sizeof(source[0]),
                                cudaMemcpyHostToDevice),
                     "copy host BF16 buffer to device");
}

bool copy_from_device(std::vector<std::uint16_t>* const destination,
                      const DeviceBuffer<std::uint16_t>& source) {
  return destination != nullptr && destination->size() == source.count() &&
         expect_cuda(cudaMemcpy(destination->data(), source.get(),
                                destination->size() * sizeof((*destination)[0]),
                                cudaMemcpyDeviceToHost),
                     "copy device BF16 buffer to host");
}

Sm87MacroFeedV4AttentionC8000OracleArguments oracle_arguments(
    const DeviceBuffer<std::uint16_t>& input,
    const DeviceBuffer<std::uint16_t>& output,
    const DeviceBuffer<std::uint16_t>& key,
    const DeviceBuffer<std::uint16_t>& value, const std::size_t token_count,
    const std::size_t first_position, const std::size_t capacity,
    cudaStream_t const stream) {
  return {input.get(),
          output.get(),
          token_count,
          q3x::kernels::kSm87MacroFeedV4AttentionC8000ScratchRowStride,
          key.get(),
          value.get(),
          capacity,
          q3x::kernels::kSm87MacroFeedV4AttentionC8000KvRowStride,
          first_position,
          reinterpret_cast<void*>(stream)};
}

bool untouched_gate_and_gap(
    const std::vector<std::uint16_t>& before,
    const std::vector<std::uint16_t>& after, const std::size_t token_count) {
  using namespace q3x::kernels;
  if (before.size() != after.size()) {
    return false;
  }
  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::size_t row =
        token * kSm87MacroFeedV4AttentionC8000ScratchRowStride;
    for (std::size_t head = 0U;
         head < kSm87MacroFeedV4AttentionC8000QueryHeads; ++head) {
      const std::size_t gate =
          row + head * kSm87MacroFeedV4AttentionC8000QGateHeadStride +
          kSm87MacroFeedV4AttentionC8000GateSlotOffset;
      for (std::size_t dimension = 0U;
           dimension < kSm87MacroFeedV4AttentionC8000HeadDimension;
           ++dimension) {
        if (before[gate + dimension] != after[gate + dimension]) {
          return false;
        }
      }
    }
    for (std::size_t index =
             kSm87MacroFeedV4AttentionC8000QGateSpan;
         index < kSm87MacroFeedV4AttentionC8000ScratchRowStride; ++index) {
      if (before[row + index] != after[row + index]) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::uint16_t> make_scratch(const std::size_t token_count) {
  using namespace q3x::kernels;
  std::vector<std::uint16_t> scratch(
      token_count * kSm87MacroFeedV4AttentionC8000ScratchRowStride,
      0x55aaU);
  for (std::size_t token = 0U; token < token_count; ++token) {
    for (std::size_t head = 0U;
         head < kSm87MacroFeedV4AttentionC8000QueryHeads; ++head) {
      for (std::size_t dimension = 0U;
           dimension < kSm87MacroFeedV4AttentionC8000HeadDimension;
           ++dimension) {
        scratch[sm87_macrofeed_v4_attention_c8000_q_physical_offset(
            token, head, dimension)] = encode_bf16(2.0F);
        scratch[sm87_macrofeed_v4_attention_c8000_gate_physical_offset(
            token, head, dimension)] = encode_bf16(0.0F);
      }
    }
  }
  return scratch;
}

std::vector<std::uint16_t> make_value_cache(const std::size_t capacity) {
  using namespace q3x::kernels;
  std::vector<std::uint16_t> value(
      capacity * kSm87MacroFeedV4AttentionC8000KvRowStride);
  for (std::size_t position = 0U; position < capacity; ++position) {
    for (std::size_t head = 0U;
         head < kSm87MacroFeedV4AttentionC8000KvHeads; ++head) {
      const std::uint16_t bits =
          encode_bf16(2.0F * static_cast<float>(head + 1U));
      const std::size_t head_begin =
          position * kSm87MacroFeedV4AttentionC8000KvRowStride +
          head * kSm87MacroFeedV4AttentionC8000HeadDimension;
      std::fill_n(value.begin() + static_cast<std::ptrdiff_t>(head_begin),
                  kSm87MacroFeedV4AttentionC8000HeadDimension, bits);
    }
  }
  return value;
}

std::vector<std::uint16_t> make_patterned_scratch(
    const std::size_t token_count) {
  using namespace q3x::kernels;
  std::vector<std::uint16_t> scratch(
      token_count * kSm87MacroFeedV4AttentionC8000ScratchRowStride,
      0x55aaU);
  for (std::size_t token = 0U; token < token_count; ++token) {
    for (std::size_t head = 0U;
         head < kSm87MacroFeedV4AttentionC8000QueryHeads; ++head) {
      for (std::size_t dimension = 0U;
           dimension < kSm87MacroFeedV4AttentionC8000HeadDimension;
           ++dimension) {
        const int query_pattern = static_cast<int>(
                                      (token * 17U + head * 7U +
                                       dimension * 3U) %
                                      19U) -
                                  9;
        const int gate_pattern = static_cast<int>(
                                     (token * 5U + head * 11U +
                                      dimension * 13U) %
                                     17U) -
                                 8;
        scratch[sm87_macrofeed_v4_attention_c8000_q_physical_offset(
            token, head, dimension)] =
            encode_bf16(static_cast<float>(query_pattern) / 64.0F);
        scratch[sm87_macrofeed_v4_attention_c8000_gate_physical_offset(
            token, head, dimension)] =
            encode_bf16(static_cast<float>(gate_pattern) / 8.0F);
      }
    }
  }
  return scratch;
}

std::vector<std::uint16_t> make_patterned_cache(
    const std::size_t capacity, const bool value) {
  using namespace q3x::kernels;
  std::vector<std::uint16_t> cache(
      capacity * kSm87MacroFeedV4AttentionC8000KvRowStride);
  for (std::size_t position = 0U; position < capacity; ++position) {
    for (std::size_t head = 0U;
         head < kSm87MacroFeedV4AttentionC8000KvHeads; ++head) {
      for (std::size_t dimension = 0U;
           dimension < kSm87MacroFeedV4AttentionC8000HeadDimension;
           ++dimension) {
        const std::size_t index =
            position * kSm87MacroFeedV4AttentionC8000KvRowStride +
            head * kSm87MacroFeedV4AttentionC8000HeadDimension + dimension;
        const int pattern = value
                                ? static_cast<int>((position * 13U +
                                                    head * 7U +
                                                    dimension * 5U) %
                                                   23U) -
                                      11
                                : static_cast<int>((position * 11U +
                                                    head * 3U +
                                                    dimension * 7U) %
                                                   17U) -
                                      8;
        cache[index] = encode_bf16(
            static_cast<float>(pattern) / (value ? 16.0F : 64.0F));
      }
    }
  }
  return cache;
}

bool run_alias_pair(
    const std::vector<std::uint16_t>& initial_scratch,
    const std::vector<std::uint16_t>& key,
    const std::vector<std::uint16_t>& value, const std::size_t token_count,
    const std::size_t first_position, const std::size_t capacity,
    cudaStream_t const stream, std::vector<std::uint16_t>* const result) {
  DeviceBuffer<std::uint16_t> input(initial_scratch.size());
  DeviceBuffer<std::uint16_t> separate_output(initial_scratch.size());
  DeviceBuffer<std::uint16_t> in_place(initial_scratch.size());
  DeviceBuffer<std::uint16_t> device_key(key.size());
  DeviceBuffer<std::uint16_t> device_value(value.size());
  if (!expect(input.get() != nullptr && separate_output.get() != nullptr &&
                  in_place.get() != nullptr && device_key.get() != nullptr &&
                  device_value.get() != nullptr,
              "allocate bounded Attention oracle buffers") ||
      !copy_to_device(input, initial_scratch) ||
      !copy_to_device(separate_output, initial_scratch) ||
      !copy_to_device(in_place, initial_scratch) ||
      !copy_to_device(device_key, key) || !copy_to_device(device_value, value)) {
    return false;
  }

  auto separate = oracle_arguments(input, separate_output, device_key,
                                   device_value, token_count, first_position,
                                   capacity, stream);
  auto aliased = oracle_arguments(in_place, in_place, device_key, device_value,
                                  token_count, first_position, capacity,
                                  stream);
  if (!expect(q3x::kernels::
                      launch_sm87_macrofeed_v4_attention_c8000_oracle_cuda(
                          separate) == static_cast<int>(cudaSuccess),
              "launch disjoint-output Attention oracle") ||
      !expect(q3x::kernels::
                      launch_sm87_macrofeed_v4_attention_c8000_oracle_cuda(
                          aliased) == static_cast<int>(cudaSuccess),
              "launch q==o Attention oracle") ||
      !expect_cuda(cudaStreamSynchronize(stream),
                   "synchronize Attention alias pair")) {
    return false;
  }

  std::vector<std::uint16_t> separate_host(initial_scratch.size());
  std::vector<std::uint16_t> aliased_host(initial_scratch.size());
  if (!copy_from_device(&separate_host, separate_output) ||
      !copy_from_device(&aliased_host, in_place)) {
    return false;
  }
  if (!expect(separate_host == aliased_host,
              "q==o is bitwise equal to disjoint output") ||
      !expect(untouched_gate_and_gap(initial_scratch, aliased_host,
                                     token_count),
              "Gate slots and scratch gap remain bitwise unchanged")) {
    return false;
  }
  if (result != nullptr) {
    *result = std::move(aliased_host);
  }
  return true;
}

bool test_plan_and_host_validation() {
  using namespace q3x::kernels;
  bool ok = true;
  for (std::size_t panel = 0U;
       panel < kSm87MacroFeedV4AttentionC8000PanelCount; ++panel) {
    const std::size_t first =
        panel * kSm87MacroFeedV4AttentionC8000Tokens;
    const auto plan = sm87_macrofeed_v4_attention_c8000_plan(
        first, kSm87MacroFeedV4AttentionC8000Tokens);
    ok &= expect(plan.valid(), "all five C8000 plans are valid");
    ok &= expect(plan.ready_end == first + 8'000U,
                 "plan ready_end follows the panel");
    ok &= expect(plan.grid_x == 375U && plan.grid_y == 1U &&
                     plan.grid_z == 4U && plan.threads_per_cta == 256U,
                 "plan freezes grid(375,1,4)/block256");
    ok &= expect(plan.dynamic_shared_bytes == 128U * 1'024U &&
                     !plan.partition_kv &&
                     plan.split_kv_workspace_bytes == 0U &&
                     !plan.merge_kernel_present,
                 "plan freezes unsplit two-stage online Attention");
  }
  ok &= expect(!sm87_macrofeed_v4_attention_c8000_plan(8'001U, 8'000U)
                    .valid(),
               "non-panel first_position is rejected");
  ok &= expect(
      sm87_macrofeed_v4_attention_c8000_q_physical_offset(1U, 1U, 0U) ==
          17'920U,
      "Q physical stride is row17408/head512");
  ok &= expect(
      sm87_macrofeed_v4_attention_c8000_gate_physical_offset(0U, 23U,
                                                              255U) ==
          12'287U,
      "last Gate element ends at the 12288 live span");

  constexpr std::uintptr_t kScratch = 0x10'0000'0000ULL;
  constexpr std::uintptr_t kKey = 0x20'0000'0000ULL;
  constexpr std::uintptr_t kValue = 0x30'0000'0000ULL;
  Sm87MacroFeedV4AttentionC8000Arguments arguments{
      reinterpret_cast<std::uint16_t*>(kScratch),
      kSm87MacroFeedV4AttentionC8000Tokens,
      kSm87MacroFeedV4AttentionC8000ScratchRowStride,
      reinterpret_cast<const std::uint16_t*>(kKey),
      reinterpret_cast<const std::uint16_t*>(kValue),
      kSm87MacroFeedV4AttentionC8000MaximumPositions,
      kSm87MacroFeedV4AttentionC8000KvRowStride,
      16'000U,
      reinterpret_cast<void*>(0x10U)};
  ok &= expect(sm87_macrofeed_v4_attention_c8000_arguments_valid(arguments),
               "structural production arguments admit full origins");
  auto bad = arguments;
  bad.first_position = 1U;
  ok &= expect(!sm87_macrofeed_v4_attention_c8000_arguments_valid(bad),
               "structural arguments reject an unsupported first_position");
  bad = arguments;
  bad.value_cache = bad.key_cache;
  ok &= expect(!sm87_macrofeed_v4_attention_c8000_arguments_valid(bad),
               "structural arguments reject aliased K/V");
  bad = arguments;
  bad.q_gate_scratch =
      reinterpret_cast<std::uint16_t*>(kKey + 16U);
  ok &= expect(!sm87_macrofeed_v4_attention_c8000_arguments_valid(bad),
               "structural arguments reject scratch/K overlap");

  Sm87MacroFeedV4AttentionC8000AdmissionLaunchReceipt receipt{};
  ok &= expect(!receipt.valid_enqueue_receipt(),
               "default receipt grants no enqueue authority");
  return ok;
}

bool test_resources(
    Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot* const snapshot) {
  using namespace q3x::kernels;
  if (!expect(snapshot != nullptr, "resource snapshot output exists")) {
    return false;
  }
  const int status =
      query_sm87_macrofeed_v4_attention_c8000_admission_resources_cuda(
          snapshot);
  if (!expect(status == static_cast<int>(cudaSuccess),
              "query fixed C8000 Attention resources")) {
    return false;
  }
  bool ok = true;
  ok &= expect(sm87_macrofeed_v4_attention_c8000_admission_resource_gate(
                   *snapshot),
               "observed resource snapshot passes the T1 gate");
  ok &= expect(snapshot->kernel.dynamic_shared_bytes == 128U * 1'024U &&
                   snapshot->kernel.active_blocks_per_sm == 1 &&
                   snapshot->kernel.local_bytes == 0U,
               "resource snapshot freezes 128KiB and one CTA/SM without local memory");
  ok &= expect(snapshot->kernel.grid_x == 375 &&
                   snapshot->kernel.grid_y == 1 &&
                   snapshot->kernel.grid_z == 4 &&
                   snapshot->kernel.physical_grid_ctas == 1'500,
               "resource snapshot freezes the physical grid");
  auto forged = *snapshot;
  forged.kernel.grid_x = 374;
  ok &= expect(!sm87_macrofeed_v4_attention_c8000_admission_resource_gate(
                    forged),
               "forged resource topology is rejected");
  if (ok) {
    std::cout << "V4_C8000_ATTENTION_RESOURCES registers="
              << snapshot->kernel.registers_per_thread
              << " local_bytes=" << snapshot->kernel.local_bytes
              << " dynamic_shared_bytes="
              << snapshot->kernel.dynamic_shared_bytes
              << " active_blocks_per_sm="
              << snapshot->kernel.active_blocks_per_sm << '\n';
  }
  return ok;
}

bool test_c1(cudaStream_t const stream) {
  using namespace q3x::kernels;
  constexpr std::size_t kTokens = 1U;
  constexpr std::size_t kFirst = 0U;
  constexpr std::size_t kCapacity = 1U;
  const auto scratch = make_scratch(kTokens);
  std::vector<std::uint16_t> key(
      kCapacity * kSm87MacroFeedV4AttentionC8000KvRowStride,
      encode_bf16(0.0F));
  const auto value = make_value_cache(kCapacity);
  std::vector<std::uint16_t> output;
  if (!run_alias_pair(scratch, key, value, kTokens, kFirst, kCapacity,
                      stream, &output)) {
    return false;
  }
  bool ok = true;
  for (std::size_t query_head = 0U;
       query_head < kSm87MacroFeedV4AttentionC8000QueryHeads; ++query_head) {
    const std::uint16_t expected =
        encode_bf16(static_cast<float>(query_head / 6U + 1U));
    for (std::size_t dimension = 0U;
         dimension < kSm87MacroFeedV4AttentionC8000HeadDimension;
         ++dimension) {
      const std::size_t offset =
          sm87_macrofeed_v4_attention_c8000_q_physical_offset(
              0U, query_head, dimension);
      if (output[offset] != expected) {
        ok = false;
        break;
      }
    }
  }
  return expect(ok,
                "C1 maps six Q heads to each NHD KV head and applies Gate after BF16 Attention");
}

bool launch_in_place_once(
    const std::vector<std::uint16_t>& scratch,
    const std::vector<std::uint16_t>& key,
    const std::vector<std::uint16_t>& value, const std::size_t token_count,
    const std::size_t first_position, const std::size_t capacity,
    cudaStream_t const stream, std::vector<std::uint16_t>* const output) {
  DeviceBuffer<std::uint16_t> device_scratch(scratch.size());
  DeviceBuffer<std::uint16_t> device_key(key.size());
  DeviceBuffer<std::uint16_t> device_value(value.size());
  if (!expect(device_scratch.get() != nullptr && device_key.get() != nullptr &&
                  device_value.get() != nullptr,
              "allocate one in-place Attention oracle") ||
      !copy_to_device(device_scratch, scratch) ||
      !copy_to_device(device_key, key) || !copy_to_device(device_value, value)) {
    return false;
  }
  const auto arguments = oracle_arguments(
      device_scratch, device_scratch, device_key, device_value, token_count,
      first_position, capacity, stream);
  if (!expect(q3x::kernels::
                      launch_sm87_macrofeed_v4_attention_c8000_oracle_cuda(
                          arguments) == static_cast<int>(cudaSuccess),
              "launch one in-place Attention oracle") ||
      !expect_cuda(cudaStreamSynchronize(stream),
                   "synchronize one in-place Attention oracle")) {
    return false;
  }
  output->resize(scratch.size());
  return copy_from_device(output, device_scratch);
}

bool test_c65_continuation_and_causality(cudaStream_t const stream) {
  using namespace q3x::kernels;
  constexpr std::size_t kTokens = 65U;
  constexpr std::size_t kFirst = 32U;
  constexpr std::size_t kCapacity = kFirst + kTokens;
  const auto scratch = make_scratch(kTokens);
  std::vector<std::uint16_t> key(
      kCapacity * kSm87MacroFeedV4AttentionC8000KvRowStride,
      encode_bf16(0.0F));
  const auto base_value = make_value_cache(kCapacity);
  std::vector<std::uint16_t> base_output;
  if (!run_alias_pair(scratch, key, base_value, kTokens, kFirst, kCapacity,
                      stream, &base_output)) {
    return false;
  }

  bool expected_constants = true;
  for (std::size_t token : {std::size_t{0U}, std::size_t{64U}}) {
    for (std::size_t query_head = 0U;
         query_head < kSm87MacroFeedV4AttentionC8000QueryHeads;
         ++query_head) {
      const auto expected =
          encode_bf16(static_cast<float>(query_head / 6U + 1U));
      const auto offset =
          sm87_macrofeed_v4_attention_c8000_q_physical_offset(
              token, query_head, 0U);
      expected_constants &= base_output[offset] == expected;
    }
  }
  if (!expect(expected_constants,
              "C65 nonzero-first scan preserves constant NHD values")) {
    return false;
  }

  auto future_poison = base_value;
  for (std::size_t position = kFirst + 1U; position < kCapacity; ++position) {
    const std::size_t begin =
        position * kSm87MacroFeedV4AttentionC8000KvRowStride;
    std::fill_n(future_poison.begin() + static_cast<std::ptrdiff_t>(begin),
                kSm87MacroFeedV4AttentionC8000HeadDimension,
                encode_bf16(100.0F));
  }
  std::vector<std::uint16_t> future_output;
  if (!launch_in_place_once(scratch, key, future_poison, kTokens, kFirst,
                            kCapacity, stream, &future_output)) {
    return false;
  }
  bool first_query_unchanged = true;
  for (std::size_t query_head = 0U; query_head < 6U; ++query_head) {
    for (std::size_t dimension : {std::size_t{0U}, std::size_t{127U},
                                  std::size_t{255U}}) {
      const auto offset =
          sm87_macrofeed_v4_attention_c8000_q_physical_offset(
              0U, query_head, dimension);
      first_query_unchanged &=
          base_output[offset] == future_output[offset];
    }
  }
  const auto last_offset =
      sm87_macrofeed_v4_attention_c8000_q_physical_offset(64U, 0U, 0U);
  if (!expect(first_query_unchanged,
              "future V poison cannot affect the first continuation query") ||
      !expect(base_output[last_offset] != future_output[last_offset],
              "later causal queries observe newly visible V positions")) {
    return false;
  }

  auto prefix_mutation = base_value;
  for (std::size_t position = 0U; position < kFirst; ++position) {
    const std::size_t begin =
        position * kSm87MacroFeedV4AttentionC8000KvRowStride;
    std::fill_n(prefix_mutation.begin() + static_cast<std::ptrdiff_t>(begin),
                kSm87MacroFeedV4AttentionC8000HeadDimension,
                encode_bf16(4.0F));
  }
  std::vector<std::uint16_t> prefix_output;
  if (!launch_in_place_once(scratch, key, prefix_mutation, kTokens, kFirst,
                            kCapacity, stream, &prefix_output)) {
    return false;
  }
  const auto first_offset =
      sm87_macrofeed_v4_attention_c8000_q_physical_offset(0U, 0U, 0U);
  return expect(base_output[first_offset] != prefix_output[first_offset],
                "nonzero first_position reads the complete KV allocation origin prefix");
}

bool test_nonzero_qk_against_public_q128_v4(cudaStream_t const stream) {
  using namespace q3x::kernels;
  constexpr std::size_t kTokens = 65U;
  constexpr std::size_t kFirst = 32U;
  constexpr std::size_t kCapacity = kFirst + kTokens;
  constexpr std::size_t kCompactElements =
      kTokens * kSm87MacroFeedV4AttentionC8000QueryHeads *
      kSm87MacroFeedV4AttentionC8000HeadDimension;

  const auto scratch = make_patterned_scratch(kTokens);
  const auto key = make_patterned_cache(kCapacity, false);
  const auto value = make_patterned_cache(kCapacity, true);
  std::vector<std::uint16_t> compact_query(kCompactElements);
  std::vector<std::uint16_t> compact_gate(kCompactElements);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t head = 0U;
         head < kSm87MacroFeedV4AttentionC8000QueryHeads; ++head) {
      for (std::size_t dimension = 0U;
           dimension < kSm87MacroFeedV4AttentionC8000HeadDimension;
           ++dimension) {
        const std::size_t compact =
            (token * kSm87MacroFeedV4AttentionC8000QueryHeads + head) *
                kSm87MacroFeedV4AttentionC8000HeadDimension +
            dimension;
        compact_query[compact] =
            scratch[sm87_macrofeed_v4_attention_c8000_q_physical_offset(
                token, head, dimension)];
        compact_gate[compact] =
            scratch[sm87_macrofeed_v4_attention_c8000_gate_physical_offset(
                token, head, dimension)];
      }
    }
  }

  DeviceBuffer<std::uint16_t> device_scratch(scratch.size());
  DeviceBuffer<std::uint16_t> device_output(scratch.size());
  DeviceBuffer<std::uint16_t> device_query(compact_query.size());
  DeviceBuffer<std::uint16_t> device_gate(compact_gate.size());
  DeviceBuffer<std::uint16_t> device_reference(kCompactElements);
  DeviceBuffer<std::uint16_t> device_key(key.size());
  DeviceBuffer<std::uint16_t> device_value(value.size());
  if (!expect(device_scratch.get() != nullptr &&
                  device_output.get() != nullptr &&
                  device_query.get() != nullptr && device_gate.get() != nullptr &&
                  device_reference.get() != nullptr &&
                  device_key.get() != nullptr && device_value.get() != nullptr,
              "allocate nonzero-QK comparison buffers") ||
      !copy_to_device(device_scratch, scratch) ||
      !copy_to_device(device_output, scratch) ||
      !copy_to_device(device_query, compact_query) ||
      !copy_to_device(device_gate, compact_gate) ||
      !copy_to_device(device_key, key) || !copy_to_device(device_value, value)) {
    return false;
  }

  const auto candidate_arguments = oracle_arguments(
      device_scratch, device_output, device_key, device_value, kTokens,
      kFirst, kCapacity, stream);
  if (!expect(
          launch_sm87_macrofeed_v4_attention_c8000_oracle_cuda(
              candidate_arguments) == static_cast<int>(cudaSuccess),
          "launch strided two-stage nonzero-QK candidate") ||
      !expect(
          q3x::runtime::
                  launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q128_v4_panel_fixed_cuda(
                      device_query.get(), device_key.get(),
                      device_value.get(), device_gate.get(), kFirst, kTokens,
                      device_reference.get(), reinterpret_cast<void*>(stream)) ==
              static_cast<int>(cudaSuccess),
          "launch compact public Q128-v4 nonzero-QK reference") ||
      !expect_cuda(cudaStreamSynchronize(stream),
                   "synchronize nonzero-QK comparison")) {
    return false;
  }

  std::vector<std::uint16_t> candidate(scratch.size());
  std::vector<std::uint16_t> reference(kCompactElements);
  if (!copy_from_device(&candidate, device_output) ||
      !copy_from_device(&reference, device_reference)) {
    return false;
  }
  bool bitwise_equal = true;
  bool qk_path_observable = false;
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t head = 0U;
         head < kSm87MacroFeedV4AttentionC8000QueryHeads; ++head) {
      for (std::size_t dimension = 0U;
           dimension < kSm87MacroFeedV4AttentionC8000HeadDimension;
           ++dimension) {
        const std::size_t compact =
            (token * kSm87MacroFeedV4AttentionC8000QueryHeads + head) *
                kSm87MacroFeedV4AttentionC8000HeadDimension +
            dimension;
        const std::size_t physical =
            sm87_macrofeed_v4_attention_c8000_q_physical_offset(
                token, head, dimension);
        bitwise_equal &= candidate[physical] == reference[compact];
        qk_path_observable |= candidate[physical] != compact_query[compact];
      }
    }
  }
  return expect(bitwise_equal,
                "nonzero patterned Q/K/V/Gate is bitwise equal to public Q128-v4") &&
         expect(qk_path_observable,
                "nonzero-QK comparison observes a non-identity Attention result") &&
         expect(untouched_gate_and_gap(scratch, candidate, kTokens),
                "nonzero-QK candidate preserves Gate and scratch gap");
}

bool test_oracle_negative_ranges(cudaStream_t const stream) {
  using namespace q3x::kernels;
  constexpr std::size_t kCapacity = 66U;
  const auto scratch = make_scratch(65U);
  std::vector<std::uint16_t> key(
      kCapacity * kSm87MacroFeedV4AttentionC8000KvRowStride,
      encode_bf16(0.0F));
  const auto value = make_value_cache(kCapacity);
  DeviceBuffer<std::uint16_t> device_scratch(scratch.size());
  DeviceBuffer<std::uint16_t> device_output(scratch.size());
  DeviceBuffer<std::uint16_t> device_key(key.size());
  DeviceBuffer<std::uint16_t> device_value(value.size());
  DeviceBuffer<std::uint16_t> isolated_key(key.size());
  if (!copy_to_device(device_scratch, scratch) ||
      !copy_to_device(device_output, scratch) ||
      !copy_to_device(device_key, key) ||
      !copy_to_device(device_value, value) ||
      !copy_to_device(isolated_key, key)) {
    return false;
  }
  auto arguments = oracle_arguments(
      device_scratch, device_output, device_key, device_value, 65U, 1U,
      kCapacity, stream);
  arguments.output_q_gate_scratch = device_scratch.get() + 8U;
  bool ok = expect(
      launch_sm87_macrofeed_v4_attention_c8000_oracle_cuda(arguments) ==
          static_cast<int>(cudaErrorInvalidValue),
      "oracle rejects partial Q/output overlap");
  arguments = oracle_arguments(device_scratch, device_output, device_key,
                               device_value, 65U, 1U, kCapacity, stream);
  arguments.token_count = 2U;
  ok &= expect(
      launch_sm87_macrofeed_v4_attention_c8000_oracle_cuda(arguments) ==
          static_cast<int>(cudaErrorInvalidValue),
      "oracle admits only the C1/C65 correctness geometries");
  arguments = oracle_arguments(device_scratch, device_output, isolated_key,
                               device_value, 65U, 1U, kCapacity, stream);
  arguments.key_cache += kSm87MacroFeedV4AttentionC8000KvRowStride;
  const int append_slice_status =
      launch_sm87_macrofeed_v4_attention_c8000_oracle_cuda(arguments);
  ok &= expect(
      append_slice_status == static_cast<int>(cudaErrorInvalidDevicePointer) ||
          append_slice_status == static_cast<int>(cudaErrorInvalidValue),
      "structural/full-allocation guards reject an append-slice KV pointer");
  (void)cudaGetLastError();
  return ok;
}

bool test_production_launch_negatives(
    const Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot& resources,
    cudaStream_t const stream) {
  using namespace q3x::kernels;
  constexpr std::uintptr_t kScratch = 0x10'0000'0000ULL;
  constexpr std::uintptr_t kKey = 0x20'0000'0000ULL;
  constexpr std::uintptr_t kValue = 0x30'0000'0000ULL;
  Sm87MacroFeedV4AttentionC8000Arguments arguments{
      reinterpret_cast<std::uint16_t*>(kScratch),
      kSm87MacroFeedV4AttentionC8000Tokens,
      kSm87MacroFeedV4AttentionC8000ScratchRowStride,
      reinterpret_cast<const std::uint16_t*>(kKey),
      reinterpret_cast<const std::uint16_t*>(kValue),
      kSm87MacroFeedV4AttentionC8000MaximumPositions,
      kSm87MacroFeedV4AttentionC8000KvRowStride,
      0U,
      reinterpret_cast<void*>(stream)};

  bool ok = expect(
      launch_sm87_macrofeed_v4_attention_c8000_admission_cuda(
          arguments, resources, nullptr) == static_cast<int>(cudaErrorInvalidValue),
      "production launch rejects a null receipt");

  Sm87MacroFeedV4AttentionC8000AdmissionLaunchReceipt receipt{};
  auto invalid_host = arguments;
  invalid_host.q_gate_scratch = nullptr;
  ok &= expect(
      launch_sm87_macrofeed_v4_attention_c8000_admission_cuda(
          invalid_host, resources, &receipt) ==
          static_cast<int>(cudaErrorInvalidValue),
      "production launch rejects invalid host geometry");
  ok &= expect(!receipt.valid_enqueue_receipt(),
               "invalid host launch leaves a non-authorizing receipt");

  auto forged = resources;
  forged.kernel.grid_x -= 1;
  ok &= expect(
      launch_sm87_macrofeed_v4_attention_c8000_admission_cuda(
          arguments, forged, &receipt) ==
          static_cast<int>(cudaErrorLaunchOutOfResources),
      "production launch rejects forged resource evidence");
  ok &= expect(!receipt.valid_enqueue_receipt(),
               "forged resource launch leaves a non-authorizing receipt");

  auto wrong_device = resources;
  wrong_device.device_ordinal += 1;
  ok &= expect(
      launch_sm87_macrofeed_v4_attention_c8000_admission_cuda(
          arguments, wrong_device, &receipt) ==
          static_cast<int>(cudaErrorInvalidDevice),
      "production launch revalidates the current device");
  ok &= expect(!receipt.valid_enqueue_receipt(),
               "wrong-device launch leaves a non-authorizing receipt");

  ok &= expect(
      launch_sm87_macrofeed_v4_attention_c8000_admission_cuda(
          arguments, resources, &receipt) ==
          static_cast<int>(cudaErrorInvalidDevicePointer),
      "production launch rejects aligned fake device addresses");
  ok &= expect(!receipt.valid_enqueue_receipt(),
               "fake-pointer launch leaves a non-authorizing receipt");
  (void)cudaGetLastError();

  DeviceBuffer<std::uint16_t> undersized_scratch(16U);
  if (!expect(undersized_scratch.get() != nullptr,
              "allocate undersized production scratch sentinel")) {
    return false;
  }
  const std::uintptr_t base =
      reinterpret_cast<std::uintptr_t>(undersized_scratch.get());
  auto undersized = arguments;
  undersized.q_gate_scratch = undersized_scratch.get();
  undersized.key_cache =
      reinterpret_cast<const std::uint16_t*>(base + 0x10'0000'0000ULL);
  undersized.value_cache =
      reinterpret_cast<const std::uint16_t*>(base + 0x20'0000'0000ULL);
  ok &= expect(
      launch_sm87_macrofeed_v4_attention_c8000_admission_cuda(
          undersized, resources, &receipt) ==
          static_cast<int>(cudaErrorInvalidDevicePointer),
      "production launch rejects an undersized declared scratch allocation");
  ok &= expect(!receipt.valid_enqueue_receipt(),
               "undersized launch leaves a non-authorizing receipt");
  (void)cudaGetLastError();
  return ok;
}

}  // namespace

int main() {
  bool ok = test_plan_and_host_validation();
  int device_count = 0;
  if (!expect_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices") ||
      device_count == 0) {
    std::cout << "SKIP: no CUDA device\n";
    return ok ? 0 : 1;
  }
  cudaDeviceProp properties{};
  if (!expect_cuda(cudaGetDeviceProperties(&properties, 0),
                   "query CUDA device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: fixed Attention constituent requires 16-SM SM87\n";
    return ok ? 0 : 1;
  }
  ok &= expect_cuda(cudaSetDevice(0), "select SM87 device");
  Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot resources{};
  ok &= test_resources(&resources);

  cudaStream_t stream = nullptr;
  ok &= expect_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create nonblocking Attention test stream");
  if (stream != nullptr) {
    ok &= test_c1(stream);
    ok &= test_c65_continuation_and_causality(stream);
    ok &= test_nonzero_qk_against_public_q128_v4(stream);
    ok &= test_oracle_negative_ranges(stream);
    ok &= test_production_launch_negatives(resources, stream);
    ok &= expect_cuda(cudaStreamDestroy(stream),
                      "destroy Attention test stream");
  }
  if (ok) {
    std::cout << "PASS: SM87 MacroFeed V4 C8000 Attention\n";
  }
  return ok ? 0 : 1;
}
