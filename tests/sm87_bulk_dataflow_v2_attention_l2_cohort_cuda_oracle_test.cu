#include "q3x/kernels/sm87_bulk_dataflow_v2_attention_l2_cohort.h"

#include "sm87_bulk_dataflow_v2_attention_l2_cohort_oracle_internal.h"
#include "sm87_target_aot_attention_launch_internal.h"

#include <cuda_runtime.h>

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace kernels = q3x::kernels;
namespace control =
    q3x::runtime::sm87_target_aot_attention_execution_detail;
namespace oracle =
    q3x::kernels::sm87_bulk_v2_attention_oracle_detail;

namespace {

constexpr std::size_t kGuardBytes = 4U * 1024U;
constexpr unsigned char kPrefixGuard = 0xd3U;
constexpr unsigned char kSuffixGuard = 0x3dU;
constexpr unsigned int kGridBlocks = 4'096U;
constexpr unsigned int kThreads = 256U;
constexpr std::size_t kQueryElements =
    kernels::kSm87BulkV2AttentionQueryBytes / sizeof(std::uint16_t);
constexpr std::size_t kKvElements =
    kernels::kSm87BulkV2AttentionKvBytes / sizeof(std::uint16_t);
constexpr std::size_t kOutputElements = kQueryElements;
constexpr std::size_t kElementsPerOutputTile =
    kernels::kSm87TargetAotAttentionQueryRows *
    kernels::kSm87TargetAotAttentionHeadDimension;

static_assert(kQueryElements == 245'760'000U);
static_assert(kKvElements == 40'960'000U);
static_assert(kOutputElements ==
              kernels::kSm87BulkV2AttentionRealOutputs *
                  kElementsPerOutputTile);
static_assert(kernels::kSm87BulkV2AttentionRealOutputs == 7'500U);
static_assert(kernels::kSm87BulkV2AttentionStoreDisabledBodies == 52U);

struct MismatchRecord final {
  unsigned long long count = 0ULL;
  unsigned long long first_index = ULLONG_MAX;
};

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_rne(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint64_t mix_index(
    std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xbf58'476d'1ce4'e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d0'49bb'1331'11ebULL;
  value ^= value >> 31U;
  return value;
}

__global__ void initialize_bf16_fixture_kernel(
    std::uint16_t* const destination, const std::size_t count,
    const std::uint64_t seed, const float quantum) {
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index = first; index < count; index += stride) {
    const std::uint64_t mixed = mix_index(index ^ seed);
    const int centered = static_cast<int>(mixed % 513ULL) - 256;
    destination[index] =
        encode_bf16_rne(static_cast<float>(centered) * quantum);
  }
}

__global__ void compare_u16_kernel(
    const std::uint16_t* const left, const std::uint16_t* const right,
    const std::size_t count, MismatchRecord* const record) {
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index = first; index < count; index += stride) {
    if (left[index] != right[index]) {
      atomicAdd(&record->count, 1ULL);
      atomicMin(&record->first_index,
                static_cast<unsigned long long>(index));
    }
  }
}

__global__ void compare_u16_constant_kernel(
    const std::uint16_t* const values, const std::size_t count,
    const std::uint16_t expected, MismatchRecord* const record) {
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index = first; index < count; index += stride) {
    if (values[index] != expected) {
      atomicAdd(&record->count, 1ULL);
      atomicMin(&record->first_index,
                static_cast<unsigned long long>(index));
    }
  }
}

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const std::string& operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << "FAIL: " << operation << ": "
            << cudaGetErrorString(status) << '\n';
  return false;
}

class Stream final {
 public:
  Stream() = default;
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() {
    if (value_ != nullptr) {
      (void)cudaStreamDestroy(value_);
    }
  }

  [[nodiscard]] bool create() {
    return cuda_ok(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking),
                   "create oracle stream");
  }
  [[nodiscard]] cudaStream_t get() const noexcept { return value_; }

 private:
  cudaStream_t value_ = nullptr;
};

template <class T>
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

  [[nodiscard]] bool allocate(const std::size_t count,
                              const char* const label) {
    count_ = count;
    return cuda_ok(cudaMalloc(reinterpret_cast<void**>(&data_),
                              count * sizeof(T)),
                   std::string("allocate ") + label);
  }
  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

template <class T>
class GuardedBuffer final {
 public:
  GuardedBuffer() = default;
  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;
  ~GuardedBuffer() {
    if (raw_ != nullptr) {
      (void)cudaFree(raw_);
    }
  }

  [[nodiscard]] bool allocate(const std::size_t count,
                              const char* const label) {
    label_ = label;
    count_ = count;
    payload_bytes_ = count * sizeof(T);
    const std::size_t allocation_bytes =
        kGuardBytes + payload_bytes_ + kGuardBytes;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&raw_),
                            allocation_bytes),
                 std::string("allocate guarded ") + label_)) {
      return false;
    }
    data_ = reinterpret_cast<T*>(raw_ + kGuardBytes);
    return reinterpret_cast<std::uintptr_t>(data_) %
               kernels::kSm87BulkV2AttentionPointerAlignment ==
           0U;
  }

  [[nodiscard]] bool initialize_guards(cudaStream_t stream) {
    return cuda_ok(cudaMemsetAsync(raw_, kPrefixGuard, kGuardBytes, stream),
                   std::string("initialize prefix guard ") + label_) &&
           cuda_ok(cudaMemsetAsync(raw_ + kGuardBytes + payload_bytes_,
                                   kSuffixGuard, kGuardBytes, stream),
                   std::string("initialize suffix guard ") + label_);
  }

  [[nodiscard]] bool poison_payload(const unsigned char value,
                                    cudaStream_t stream) {
    return cuda_ok(cudaMemsetAsync(data_, value, payload_bytes_, stream),
                   std::string("poison payload ") + label_);
  }

  [[nodiscard]] bool guards_intact() const {
    std::array<unsigned char, kGuardBytes> host{};
    if (!cuda_ok(cudaMemcpy(host.data(), raw_, kGuardBytes,
                            cudaMemcpyDeviceToHost),
                 std::string("download prefix guard ") + label_)) {
      return false;
    }
    for (const unsigned char value : host) {
      if (value != kPrefixGuard) {
        std::cerr << "FAIL: prefix redzone changed for " << label_ << '\n';
        return false;
      }
    }
    if (!cuda_ok(cudaMemcpy(host.data(),
                            raw_ + kGuardBytes + payload_bytes_,
                            kGuardBytes, cudaMemcpyDeviceToHost),
                 std::string("download suffix guard ") + label_)) {
      return false;
    }
    for (const unsigned char value : host) {
      if (value != kSuffixGuard) {
        std::cerr << "FAIL: suffix redzone changed for " << label_ << '\n';
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return payload_bytes_; }

 private:
  unsigned char* raw_ = nullptr;
  T* data_ = nullptr;
  std::size_t count_ = 0U;
  std::size_t payload_bytes_ = 0U;
  std::string label_;
};

[[nodiscard]] bool initialize_mismatch_record(
    DeviceBuffer<MismatchRecord>& device_record, cudaStream_t stream,
    const char* const label) {
  const MismatchRecord initial{};
  return device_record.allocate(1U, label) &&
         cuda_ok(cudaMemcpyAsync(device_record.data(), &initial,
                                 sizeof(initial), cudaMemcpyHostToDevice,
                                 stream),
                 std::string("initialize mismatch record ") + label);
}

[[nodiscard]] bool download_mismatch_record(
    const DeviceBuffer<MismatchRecord>& device_record,
    MismatchRecord& result, cudaStream_t stream,
    const char* const label) {
  return cuda_ok(cudaMemcpyAsync(&result, device_record.data(), sizeof(result),
                                 cudaMemcpyDeviceToHost, stream),
                 std::string("download mismatch record ") + label) &&
         cuda_ok(cudaStreamSynchronize(stream),
                 std::string("synchronize comparison ") + label);
}

[[nodiscard]] bool compare_surfaces(
    const char* const label, const std::uint16_t* const left,
    const std::uint16_t* const right, const std::size_t count,
    cudaStream_t stream) {
  DeviceBuffer<MismatchRecord> device_record;
  if (!initialize_mismatch_record(device_record, stream, label)) {
    return false;
  }
  compare_u16_kernel<<<kGridBlocks, kThreads, 0U, stream>>>(
      left, right, count, device_record.data());
  if (!cuda_ok(cudaPeekAtLastError(),
               std::string("launch comparison ") + label)) {
    return false;
  }
  MismatchRecord result{};
  if (!download_mismatch_record(device_record, result, stream, label)) {
    return false;
  }
  if (result.count != 0ULL) {
    std::cerr << "FAIL: " << label << " has " << result.count
              << " bit mismatches; first index " << result.first_index
              << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool compare_constant(
    const char* const label, const std::uint16_t* const values,
    const std::size_t count, const std::uint16_t expected,
    cudaStream_t stream) {
  DeviceBuffer<MismatchRecord> device_record;
  if (!initialize_mismatch_record(device_record, stream, label)) {
    return false;
  }
  compare_u16_constant_kernel<<<kGridBlocks, kThreads, 0U, stream>>>(
      values, count, expected, device_record.data());
  if (!cuda_ok(cudaPeekAtLastError(),
               std::string("launch constant comparison ") + label)) {
    return false;
  }
  MismatchRecord result{};
  if (!download_mismatch_record(device_record, result, stream, label)) {
    return false;
  }
  if (result.count != 0ULL) {
    std::cerr << "FAIL: " << label << " changed " << result.count
              << " canary elements; first index " << result.first_index
              << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool verify_single_writer_mapping() {
  std::array<std::uint8_t, kernels::kSm87BulkV2AttentionRealOutputs>
      write_counts{};
  std::size_t suppressed = 0U;
  for (std::size_t kv_head = 0U;
       kv_head < kernels::kSm87BulkV2AttentionKvHeads; ++kv_head) {
    for (std::size_t lane = 0U;
         lane < kernels::kSm87BulkV2AttentionPersistentLanes; ++lane) {
      for (std::size_t epoch = 0U;
           epoch < kernels::kSm87BulkV2AttentionSnakeEpochs; ++epoch) {
        const auto work = kernels::sm87_bulk_v2_attention_work_item(
            kv_head, lane, epoch);
        if (!work.valid) {
          return false;
        }
        if (work.store_enabled) {
          const std::size_t output_tile =
              kv_head * kernels::kSm87BulkV2AttentionQueryTilesPerKvHead +
              work.query_tile;
          ++write_counts[output_tile];
        } else {
          ++suppressed;
        }
      }
    }
  }
  if (suppressed != kernels::kSm87BulkV2AttentionStoreDisabledBodies) {
    return false;
  }
  for (const std::uint8_t count : write_counts) {
    if (count != 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool initialize_fixture(
    GuardedBuffer<std::uint16_t>& query,
    GuardedBuffer<std::uint16_t>& key,
    GuardedBuffer<std::uint16_t>& value,
    GuardedBuffer<std::uint16_t>& gate, cudaStream_t stream) {
  initialize_bf16_fixture_kernel<<<kGridBlocks, kThreads, 0U, stream>>>(
      query.data(), query.size(), 0x1010'2020'3030'4040ULL,
      1.0F / 512.0F);
  initialize_bf16_fixture_kernel<<<kGridBlocks, kThreads, 0U, stream>>>(
      key.data(), key.size(), 0x5151'6262'7373'8484ULL,
      1.0F / 512.0F);
  initialize_bf16_fixture_kernel<<<kGridBlocks, kThreads, 0U, stream>>>(
      value.data(), value.size(), 0x9191'a2a2'b3b3'c4c4ULL,
      1.0F / 256.0F);
  initialize_bf16_fixture_kernel<<<kGridBlocks, kThreads, 0U, stream>>>(
      gate.data(), gate.size(), 0xd1d1'e2e2'f3f3'0404ULL,
      1.0F / 64.0F);
  return cuda_ok(cudaPeekAtLastError(), "launch deterministic input fixture");
}

}  // namespace

int main() {
  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    (void)cudaGetLastError();
    std::cout << "SKIP: CUDA device unavailable\n";
    return 77;
  }
  int device = -1;
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDevice(&device), "query current device") ||
      !cuda_ok(cudaGetDeviceProperties(&properties, device),
               "query device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          kernels::kSm87BulkV2AttentionRequiredSmCount) {
    std::cout << "SKIP: exact SM87/16-SM target unavailable\n";
    return 77;
  }

  Stream stream;
  if (!stream.create()) {
    return 1;
  }

  GuardedBuffer<std::uint16_t> query;
  GuardedBuffer<std::uint16_t> key;
  GuardedBuffer<std::uint16_t> value;
  GuardedBuffer<std::uint16_t> gate;
  GuardedBuffer<std::uint16_t> control_output;
  GuardedBuffer<std::uint16_t> candidate_output;
  GuardedBuffer<std::uint16_t> replay_output;
  if (!query.allocate(kQueryElements, "query") ||
      !key.allocate(kKvElements, "key") ||
      !value.allocate(kKvElements, "value") ||
      !gate.allocate(kQueryElements, "gate") ||
      !control_output.allocate(kOutputElements, "control output") ||
      !candidate_output.allocate(kOutputElements, "candidate output") ||
      !replay_output.allocate(kOutputElements, "replay output")) {
    return 1;
  }
  if (!query.initialize_guards(stream.get()) ||
      !key.initialize_guards(stream.get()) ||
      !value.initialize_guards(stream.get()) ||
      !gate.initialize_guards(stream.get()) ||
      !control_output.initialize_guards(stream.get()) ||
      !candidate_output.initialize_guards(stream.get()) ||
      !replay_output.initialize_guards(stream.get()) ||
      !initialize_fixture(query, key, value, gate, stream.get())) {
    return 1;
  }

  DeviceBuffer<std::uint16_t> query_snapshot;
  DeviceBuffer<std::uint16_t> key_snapshot;
  DeviceBuffer<std::uint16_t> value_snapshot;
  DeviceBuffer<std::uint16_t> gate_snapshot;
  if (!query_snapshot.allocate(kQueryElements, "query snapshot") ||
      !key_snapshot.allocate(kKvElements, "key snapshot") ||
      !value_snapshot.allocate(kKvElements, "value snapshot") ||
      !gate_snapshot.allocate(kQueryElements, "gate snapshot") ||
      !cuda_ok(cudaMemcpyAsync(query_snapshot.data(), query.data(),
                               query.bytes(), cudaMemcpyDeviceToDevice,
                               stream.get()),
               "snapshot query") ||
      !cuda_ok(cudaMemcpyAsync(key_snapshot.data(), key.data(), key.bytes(),
                               cudaMemcpyDeviceToDevice, stream.get()),
               "snapshot key") ||
      !cuda_ok(cudaMemcpyAsync(value_snapshot.data(), value.data(),
                               value.bytes(), cudaMemcpyDeviceToDevice,
                               stream.get()),
               "snapshot value") ||
      !cuda_ok(cudaMemcpyAsync(gate_snapshot.data(), gate.data(), gate.bytes(),
                               cudaMemcpyDeviceToDevice, stream.get()),
               "snapshot gate") ||
      !control_output.poison_payload(0xa5U, stream.get()) ||
      !candidate_output.poison_payload(0x5aU, stream.get()) ||
      !replay_output.poison_payload(0x3cU, stream.get()) ||
      !cuda_ok(cudaStreamSynchronize(stream.get()),
               "synchronize oracle setup")) {
    return 1;
  }

  control::TargetP40Resources control_resources{};
  kernels::Sm87BulkV2AttentionResources candidate_resources{};
  if (!cuda_ok(static_cast<cudaError_t>(
                   control::query_q128_kv32_p40_two_stage_resources(
                       device, &control_resources)),
               "query target-AOT control resources") ||
      !cuda_ok(static_cast<cudaError_t>(
                   kernels::query_sm87_bulk_dataflow_v2_attention_l2_cohort_resources_cuda(
                       device, &candidate_resources)),
               "query Attention-v2 candidate resources") ||
      !control::target_p40_resources_structurally_valid(control_resources) ||
      !kernels::sm87_bulk_v2_attention_resources_valid(candidate_resources) ||
      control_resources.numerical_contract_qualified ||
      control_resources.production_dispatch_eligible ||
      candidate_resources.numerical_contract_qualified ||
      candidate_resources.production_dispatch_eligible) {
    std::cerr << "FAIL: control/candidate resource or qualification boundary\n";
    return 1;
  }

  control::TargetP40Arguments control_arguments;
  control_arguments.processed_query = query.data();
  control_arguments.processed_key = key.data();
  control_arguments.processed_value = value.data();
  control_arguments.processed_gate = gate.data();
  control_arguments.gated_output = control_output.data();
  control_arguments.token_count = kernels::kSm87BulkV2AttentionTokens;
  control_arguments.device_ordinal = device;
  control_arguments.cuda_stream = reinterpret_cast<void*>(stream.get());
  if (!control::target_p40_arguments_structurally_valid(control_arguments) ||
      !cuda_ok(static_cast<cudaError_t>(
                   control::launch_q128_kv32_p40_two_stage(control_arguments)),
               "launch target-AOT Q128/KV32 control")) {
    return 1;
  }

  kernels::Sm87BulkV2AttentionArguments candidate_arguments;
  candidate_arguments.processed_query = query.data();
  candidate_arguments.processed_key = key.data();
  candidate_arguments.processed_value = value.data();
  candidate_arguments.processed_gate = gate.data();
  candidate_arguments.gated_output = candidate_output.data();
  candidate_arguments.token_count = kernels::kSm87BulkV2AttentionTokens;
  candidate_arguments.device_ordinal = device;
  candidate_arguments.cuda_stream = reinterpret_cast<void*>(stream.get());
  if (!kernels::sm87_bulk_v2_attention_arguments_valid(candidate_arguments) ||
      !cuda_ok(static_cast<cudaError_t>(
                   kernels::launch_sm87_bulk_dataflow_v2_attention_l2_cohort_cuda(
                       candidate_arguments)),
               "launch Attention-v2 L2 cohort candidate")) {
    return 1;
  }

  candidate_arguments.gated_output = replay_output.data();
  if (!kernels::sm87_bulk_v2_attention_arguments_valid(candidate_arguments) ||
      !cuda_ok(static_cast<cudaError_t>(
                   kernels::launch_sm87_bulk_dataflow_v2_attention_l2_cohort_cuda(
                       candidate_arguments)),
               "launch deterministic candidate replay") ||
      !cuda_ok(cudaStreamSynchronize(stream.get()),
               "synchronize control/candidate/replay")) {
    return 1;
  }

  if (!verify_single_writer_mapping() ||
      !compare_surfaces("7,500 output tiles control vs candidate",
                        control_output.data(), candidate_output.data(),
                        kOutputElements, stream.get()) ||
      !compare_surfaces("7,500 output tiles candidate vs replay",
                        candidate_output.data(), replay_output.data(),
                        kOutputElements, stream.get())) {
    std::cerr << "FAIL: output ownership, differential, or replay contract\n";
    return 1;
  }

  // Poison the complete output, then execute exactly the 52 repeated
  // final-epoch bodies through the shared arithmetic body with stores
  // suppressed. Any payload or redzone write is a correctness failure.
  if (!replay_output.poison_payload(0xc7U, stream.get())) {
    return 1;
  }
  candidate_arguments.gated_output = replay_output.data();
  if (!cuda_ok(static_cast<cudaError_t>(
                   oracle::launch_suppressed_repeat_bodies_cuda(
                       candidate_arguments)),
               "launch 52 suppressed-repeat bodies") ||
      !cuda_ok(cudaStreamSynchronize(stream.get()),
               "synchronize suppressed-repeat oracle") ||
      !compare_constant("52 suppressed-repeat output canary",
                        replay_output.data(), kOutputElements, 0xc7c7U,
                        stream.get())) {
    return 1;
  }

  if (!compare_surfaces("query immutability", query.data(),
                        query_snapshot.data(), kQueryElements, stream.get()) ||
      !compare_surfaces("key immutability", key.data(), key_snapshot.data(),
                        kKvElements, stream.get()) ||
      !compare_surfaces("value immutability", value.data(),
                        value_snapshot.data(), kKvElements, stream.get()) ||
      !compare_surfaces("gate immutability", gate.data(), gate_snapshot.data(),
                        kQueryElements, stream.get()) ||
      !query.guards_intact() || !key.guards_intact() ||
      !value.guards_intact() || !gate.guards_intact() ||
      !control_output.guards_intact() ||
      !candidate_output.guards_intact() || !replay_output.guards_intact()) {
    return 1;
  }

  std::cout << "ATTENTION_V2_CUDA_ORACLE output_tiles="
            << kernels::kSm87BulkV2AttentionRealOutputs
            << " output_elements=" << kOutputElements
            << " suppressed_repeat_bodies="
            << kernels::kSm87BulkV2AttentionStoreDisabledBodies
            << " candidate_registers="
            << candidate_resources.registers_per_thread
            << " control_registers="
            << control_resources.registers_per_thread
            << " numerical_contract_qualified=false"
            << " production_dispatch_eligible=false\n";
  std::cout << "Attention-v2 CUDA differential oracle passed\n";
  return 0;
}
