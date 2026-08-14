#include "q3x/kernels/sm87_bulk_dataflow_v2_gdn_p40_plan.h"

#include <cuda_runtime.h>

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

namespace kernels = q3x::kernels;

namespace {

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kBlocks = 1'024U;
constexpr std::size_t kGuardBytes = 64U;
constexpr unsigned char kGuardPattern = 0xd3U;
constexpr std::size_t kRawElements =
    kernels::kSm87BulkV2GdnP40RawQkvzBytes / sizeof(std::uint16_t);
constexpr std::size_t kAbElements =
    kernels::kSm87BulkV2GdnP40InterleavedAbBytes / sizeof(std::uint16_t);
constexpr std::size_t kOutputElements =
    kernels::kSm87BulkV2GdnP40OutputBytes / sizeof(std::uint16_t);
constexpr std::size_t kStateElements =
    kernels::kSm87BulkV2GdnStateBytes / sizeof(std::uint16_t);
constexpr std::size_t kHistoryElements =
    kernels::kSm87BulkV2GdnConvHistoryBytes / sizeof(std::uint16_t);

enum class PatternKind : unsigned int {
  kSignal = 0U,
  kGate,
  kConvWeight,
  kALog,
  kDtBias,
  kNormWeight,
  kState,
};

struct MismatchRecord final {
  unsigned long long mismatch_count;
  unsigned long long first_index;
};

[[nodiscard]] __device__ __forceinline__ std::uint64_t mix_index(
    std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xbf58'476d'1ce4'e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d0'49bb'1331'11ebULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t pattern_value(
    const PatternKind kind, const std::size_t index,
    const std::uint64_t salt) noexcept {
  constexpr std::uint16_t kSignal[] = {
      0x0000U, 0x8000U, 0x0001U, 0x8001U, 0x3c00U, 0xbc00U,
      0x3c80U, 0xbc80U, 0x3d00U, 0xbd00U, 0x3d80U, 0xbd80U,
      0x3e00U, 0xbe00U, 0x3e80U, 0xbe80U, 0x3f00U, 0xbf00U,
      0x3f40U, 0xbf40U, 0x3f80U, 0xbf80U};
  constexpr std::uint16_t kGate[] = {
      0xc1c0U, 0xc1a8U, 0xc1a1U, 0xc1a0U, 0xc180U, 0xc100U,
      0xbf80U, 0x8000U, 0x0000U, 0x3e00U, 0x3f80U, 0x4100U,
      0x4180U, 0x41a0U, 0x41a1U, 0x41a8U, 0x41c0U};
  constexpr std::uint16_t kWeight[] = {
      0x0000U, 0x8000U, 0x3b80U, 0xbb80U, 0x3c00U, 0xbc00U,
      0x3c80U, 0xbc80U, 0x3d00U, 0xbd00U, 0x3d80U, 0xbd80U};
  constexpr std::uint16_t kALog[] = {
      0xc080U, 0xc040U, 0xc000U, 0xbf80U, 0xbf40U, 0xbf00U};
  constexpr std::uint16_t kDt[] = {
      0xc000U, 0xbf80U, 0xbe00U, 0x8000U, 0x0000U,
      0x3e00U, 0x3f80U, 0x4000U};
  constexpr std::uint16_t kNorm[] = {
      0xbfc0U, 0xbf80U, 0xbf00U, 0xbe00U,
      0x3e00U, 0x3f00U, 0x3f80U, 0x3fc0U};
  constexpr std::uint16_t kState[] = {
      0x0000U, 0x8000U, 0x0001U, 0x8001U, 0x3b00U, 0xbb00U,
      0x3b80U, 0xbb80U, 0x3c00U, 0xbc00U, 0x3c80U, 0xbc80U,
      0x3d00U, 0xbd00U, 0x3d80U, 0xbd80U};
  const std::uint64_t mixed = mix_index(static_cast<std::uint64_t>(index) +
                                        salt);
  switch (kind) {
    case PatternKind::kSignal:
      return kSignal[mixed % (sizeof(kSignal) / sizeof(kSignal[0U]))];
    case PatternKind::kGate:
      return kGate[mixed % (sizeof(kGate) / sizeof(kGate[0U]))];
    case PatternKind::kConvWeight:
      return kWeight[mixed % (sizeof(kWeight) / sizeof(kWeight[0U]))];
    case PatternKind::kALog:
      return kALog[mixed % (sizeof(kALog) / sizeof(kALog[0U]))];
    case PatternKind::kDtBias:
      return kDt[mixed % (sizeof(kDt) / sizeof(kDt[0U]))];
    case PatternKind::kNormWeight:
      return kNorm[mixed % (sizeof(kNorm) / sizeof(kNorm[0U]))];
    case PatternKind::kState:
      return kState[mixed % (sizeof(kState) / sizeof(kState[0U]))];
  }
  return 0U;
}

__global__ void fill_pattern_kernel(std::uint16_t* const data,
                                    const std::size_t count,
                                    const PatternKind kind,
                                    const std::uint64_t salt) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    data[index] = pattern_value(kind, index, salt);
  }
}

__global__ void compare_bytes_kernel(const unsigned char* const left,
                                     const unsigned char* const right,
                                     const std::size_t count,
                                     MismatchRecord* const record) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (left[index] != right[index]) {
      atomicAdd(&record->mismatch_count, 1ULL);
      atomicMin(&record->first_index,
                static_cast<unsigned long long>(index));
    }
  }
}

__global__ void verify_pattern_kernel(const std::uint16_t* const data,
                                      const std::size_t count,
                                      const PatternKind kind,
                                      const std::uint64_t salt,
                                      MismatchRecord* const record) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (data[index] != pattern_value(kind, index, salt)) {
      atomicAdd(&record->mismatch_count, 1ULL);
      atomicMin(&record->first_index,
                static_cast<unsigned long long>(index));
    }
  }
}

__global__ void verify_guards_kernel(const unsigned char* const base,
                                     const std::size_t payload_bytes,
                                     MismatchRecord* const record) {
  for (std::size_t index = threadIdx.x; index < 2U * kGuardBytes;
       index += blockDim.x) {
    const std::size_t address =
        index < kGuardBytes ? index : kGuardBytes + payload_bytes +
                                         index - kGuardBytes;
    if (base[address] != kGuardPattern) {
      atomicAdd(&record->mismatch_count, 1ULL);
      atomicMin(&record->first_index,
                static_cast<unsigned long long>(index));
    }
  }
}

__global__ void verify_uniform_byte_kernel(
    const unsigned char* const data, const std::size_t count,
    const unsigned char expected, MismatchRecord* const record) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (data[index] != expected) {
      atomicAdd(&record->mismatch_count, 1ULL);
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
  std::cerr << "FAIL: " << operation << ": " << cudaGetErrorString(status)
            << '\n';
  return false;
}

template <class T>
class GuardedBuffer final {
 public:
  GuardedBuffer() = default;
  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;

  ~GuardedBuffer() {
    if (base_ != nullptr) {
      (void)cudaFree(base_);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t count) noexcept {
    count_ = count;
    const std::size_t bytes = count * sizeof(T) + 2U * kGuardBytes;
    cudaError_t status =
        cudaMalloc(reinterpret_cast<void**>(&base_), bytes);
    if (status == cudaSuccess) {
      data_ = reinterpret_cast<T*>(
          reinterpret_cast<unsigned char*>(base_) + kGuardBytes);
    }
    return status;
  }

  [[nodiscard]] cudaError_t initialize(const unsigned char poison,
                                       cudaStream_t stream) noexcept {
    cudaError_t status = cudaMemsetAsync(
        base_, kGuardPattern, count_ * sizeof(T) + 2U * kGuardBytes, stream);
    if (status == cudaSuccess) {
      status = cudaMemsetAsync(data_, poison, count_ * sizeof(T), stream);
    }
    return status;
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] unsigned char* bytes() noexcept {
    return reinterpret_cast<unsigned char*>(data_);
  }
  [[nodiscard]] const unsigned char* bytes() const noexcept {
    return reinterpret_cast<const unsigned char*>(data_);
  }
  [[nodiscard]] const unsigned char* base_bytes() const noexcept {
    return reinterpret_cast<const unsigned char*>(base_);
  }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] std::size_t payload_bytes() const noexcept {
    return count_ * sizeof(T);
  }

 private:
  void* base_ = nullptr;
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

template <class T>
[[nodiscard]] bool allocate(GuardedBuffer<T>& buffer,
                            const std::size_t count,
                            const char* const label) {
  return cuda_ok(buffer.allocate(count), std::string("allocate ") + label);
}

struct Inputs final {
  GuardedBuffer<std::uint16_t> raw_qkvz;
  GuardedBuffer<std::uint16_t> interleaved_ab;
  GuardedBuffer<std::uint16_t> conv_weight;
  GuardedBuffer<std::uint16_t> initial_history;
  GuardedBuffer<std::uint16_t> a_log;
  GuardedBuffer<std::uint16_t> dt_bias;
  GuardedBuffer<std::uint16_t> norm_weight;
  GuardedBuffer<std::uint16_t> initial_state;
};

struct PipelineBuffers final {
  std::array<GuardedBuffer<float>, 2U> normalized_q;
  std::array<GuardedBuffer<float>, 2U> normalized_k;
  std::array<GuardedBuffer<std::uint16_t>, 2U> prepared_v;
  std::array<GuardedBuffer<float>, 2U> alpha;
  std::array<GuardedBuffer<float>, 2U> beta;
  std::array<GuardedBuffer<std::uint16_t>, 2U> raw_output;
  GuardedBuffer<std::uint16_t> output;
  std::array<GuardedBuffer<std::uint16_t>, 2U> history;
  GuardedBuffer<std::uint16_t> state;
  std::array<GuardedBuffer<std::uint32_t>, 2U> cancellation_snapshot;
};

class ExecutionOwner final {
 public:
  ExecutionOwner() = default;
  ExecutionOwner(const ExecutionOwner&) = delete;
  ExecutionOwner& operator=(const ExecutionOwner&) = delete;

  ~ExecutionOwner() {
    for (auto event : events_) {
      if (event != nullptr) {
        (void)cudaEventDestroy(event);
      }
    }
    for (auto stream : streams_) {
      if (stream != nullptr) {
        (void)cudaStreamDestroy(stream);
      }
    }
  }

  [[nodiscard]] bool create() {
    for (std::size_t index = 0U; index < streams_.size(); ++index) {
      if (!cuda_ok(cudaStreamCreateWithFlags(&streams_[index],
                                             cudaStreamNonBlocking),
                   "create nonblocking stream")) {
        return false;
      }
    }
    for (std::size_t index = 0U; index < events_.size(); ++index) {
      if (!cuda_ok(cudaEventCreateWithFlags(&events_[index],
                                            cudaEventDisableTiming),
                   "create disable-timing event")) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] cudaStream_t stream(const std::size_t index) const noexcept {
    return streams_[index];
  }
  [[nodiscard]] cudaEvent_t event(const std::size_t index) const noexcept {
    return events_[index];
  }
  [[nodiscard]] kernels::Sm87BulkV2GdnP40SubmissionReceipt* receipt() noexcept {
    return &receipt_;
  }

 private:
  std::array<cudaStream_t, 3U> streams_{};
  std::array<cudaEvent_t, 6U> events_{};
  kernels::Sm87BulkV2GdnP40SubmissionReceipt receipt_ =
      kernels::sm87_bulk_v2_gdn_p40_submission_receipt();
};

class CancellationOwner final {
 public:
  CancellationOwner() = default;
  CancellationOwner(const CancellationOwner&) = delete;
  CancellationOwner& operator=(const CancellationOwner&) = delete;

  ~CancellationOwner() {
    if (host_ != nullptr) {
      (void)cudaFreeHost(host_);
    }
  }

  [[nodiscard]] bool create() {
    if (!cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&host_),
                               sizeof(std::uint32_t), cudaHostAllocMapped),
                 "allocate mapped cancellation word")) {
      return false;
    }
    void* device = nullptr;
    if (!cuda_ok(cudaHostGetDevicePointer(&device, host_, 0U),
                 "map cancellation word") ||
        device == nullptr) {
      return false;
    }
    device_ = static_cast<const std::uint32_t*>(device);
    store(0U);
    return true;
  }

  void store(const std::uint32_t value) noexcept {
    __atomic_store_n(host_, value, __ATOMIC_RELEASE);
  }
  [[nodiscard]] std::uint32_t load() const noexcept {
    return __atomic_load_n(host_, __ATOMIC_ACQUIRE);
  }

  [[nodiscard]] const std::uint32_t* device() const noexcept {
    return device_;
  }
  [[nodiscard]] std::uint32_t* host() const noexcept { return host_; }

 private:
  std::uint32_t* host_ = nullptr;
  const std::uint32_t* device_ = nullptr;
};

[[nodiscard]] bool allocate_inputs(Inputs& inputs) {
  bool ready = true;
  ready = allocate(inputs.raw_qkvz, kRawElements, "P40 raw QKVZ") && ready;
  ready = allocate(inputs.interleaved_ab, kAbElements, "P40 AB") && ready;
  ready = allocate(inputs.conv_weight,
                   kernels::kSm87TargetAotGdnConvWeightElements,
                   "conv weight") && ready;
  ready = allocate(inputs.initial_history, kHistoryElements,
                   "initial history") && ready;
  ready = allocate(inputs.a_log,
                   kernels::kSm87TargetAotGdnScalarHeadElements, "A log") &&
          ready;
  ready = allocate(inputs.dt_bias,
                   kernels::kSm87TargetAotGdnScalarHeadElements, "dt bias") &&
          ready;
  ready = allocate(inputs.norm_weight,
                   kernels::kSm87TargetAotGdnNormWeightElements,
                   "norm weight") && ready;
  ready = allocate(inputs.initial_state, kStateElements, "initial state") &&
          ready;
  return ready;
}

[[nodiscard]] bool allocate_pipeline(PipelineBuffers& buffers,
                                     const char* const owner) {
  bool ready = true;
  for (std::size_t slot = 0U; slot < 2U; ++slot) {
    ready = allocate(buffers.normalized_q[slot],
                     kernels::kSm87BulkV2GdnNormalizedQBytes / sizeof(float),
                     owner) && ready;
    ready = allocate(buffers.normalized_k[slot],
                     kernels::kSm87BulkV2GdnNormalizedKBytes / sizeof(float),
                     owner) && ready;
    ready = allocate(buffers.prepared_v[slot],
                     kernels::kSm87BulkV2GdnPreparedVBytes /
                         sizeof(std::uint16_t),
                     owner) && ready;
    ready = allocate(buffers.alpha[slot],
                     kernels::kSm87BulkV2GdnAlphaBytes / sizeof(float),
                     owner) && ready;
    ready = allocate(buffers.beta[slot],
                     kernels::kSm87BulkV2GdnBetaBytes / sizeof(float),
                     owner) && ready;
    ready = allocate(buffers.raw_output[slot],
                     kernels::kSm87BulkV2GdnRawOutputBytes /
                         sizeof(std::uint16_t),
                     owner) && ready;
    ready = allocate(buffers.history[slot], kHistoryElements, owner) && ready;
    ready = allocate(buffers.cancellation_snapshot[slot], 1U, owner) && ready;
  }
  ready = allocate(buffers.output, kOutputElements, owner) && ready;
  ready = allocate(buffers.state, kStateElements, owner) && ready;
  return ready;
}

template <class T>
[[nodiscard]] bool initialize_one(GuardedBuffer<T>& buffer,
                                  const unsigned char poison,
                                  cudaStream_t stream,
                                  const char* const label) {
  return cuda_ok(buffer.initialize(poison, stream),
                 std::string("initialize ") + label);
}

[[nodiscard]] bool initialize_inputs(Inputs& inputs, cudaStream_t stream) {
  bool ready = true;
#define Q3X_FILL_INPUT(field, kind, salt)                                      \
  ready = initialize_one(inputs.field, 0U, stream, #field) && ready;           \
  fill_pattern_kernel<<<kBlocks, kThreads, 0U, stream>>>(                      \
      inputs.field.data(), inputs.field.size(), PatternKind::kind, salt);      \
  ready = cuda_ok(cudaPeekAtLastError(), "fill " #field) && ready
  Q3X_FILL_INPUT(raw_qkvz, kSignal, 0x101ULL);
  Q3X_FILL_INPUT(interleaved_ab, kGate, 0x202ULL);
  Q3X_FILL_INPUT(conv_weight, kConvWeight, 0x303ULL);
  Q3X_FILL_INPUT(initial_history, kSignal, 0x404ULL);
  Q3X_FILL_INPUT(a_log, kALog, 0x505ULL);
  Q3X_FILL_INPUT(dt_bias, kDtBias, 0x606ULL);
  Q3X_FILL_INPUT(norm_weight, kNormWeight, 0x707ULL);
  Q3X_FILL_INPUT(initial_state, kState, 0x808ULL);
#undef Q3X_FILL_INPUT
  return cuda_ok(cudaStreamSynchronize(stream), "finish input initialization") &&
         ready;
}

[[nodiscard]] bool initialize_pipeline(PipelineBuffers& buffers,
                                       const unsigned char poison,
                                       cudaStream_t stream,
                                       const char* const label) {
  bool ready = true;
  for (std::size_t slot = 0U; slot < 2U; ++slot) {
    ready = initialize_one(buffers.normalized_q[slot], poison, stream, label) &&
            ready;
    ready = initialize_one(buffers.normalized_k[slot], poison, stream, label) &&
            ready;
    ready = initialize_one(buffers.prepared_v[slot], poison, stream, label) &&
            ready;
    ready = initialize_one(buffers.alpha[slot], poison, stream, label) && ready;
    ready = initialize_one(buffers.beta[slot], poison, stream, label) && ready;
    ready = initialize_one(buffers.raw_output[slot], poison, stream, label) &&
            ready;
    ready = initialize_one(buffers.history[slot], poison, stream, label) &&
            ready;
    ready = initialize_one(buffers.cancellation_snapshot[slot], poison, stream,
                           label) && ready;
  }
  ready = initialize_one(buffers.output, poison, stream, label) && ready;
  ready = initialize_one(buffers.state, poison, stream, label) && ready;
  return cuda_ok(cudaStreamSynchronize(stream),
                 std::string("finish initialization ") + label) && ready;
}

[[nodiscard]] kernels::Sm87BulkV2GdnP40Arguments make_arguments(
    const Inputs& inputs, PipelineBuffers& buffers,
    const CancellationOwner& cancellation) noexcept {
  kernels::Sm87BulkV2GdnP40Arguments arguments;
  arguments.raw_qkvz = inputs.raw_qkvz.data();
  arguments.interleaved_ab = inputs.interleaved_ab.data();
  arguments.conv_weight = inputs.conv_weight.data();
  arguments.initial_conv_history = inputs.initial_history.data();
  arguments.a_log = inputs.a_log.data();
  arguments.dt_bias = inputs.dt_bias.data();
  arguments.norm_weight = inputs.norm_weight.data();
  arguments.initial_recurrent_state = inputs.initial_state.data();
  arguments.l2_epsilon_fp32_bits = kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.norm_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  for (std::size_t slot = 0U; slot < 2U; ++slot) {
    arguments.normalized_q[slot] = buffers.normalized_q[slot].data();
    arguments.normalized_k[slot] = buffers.normalized_k[slot].data();
    arguments.prepared_v[slot] = buffers.prepared_v[slot].data();
    arguments.alpha[slot] = buffers.alpha[slot].data();
    arguments.beta[slot] = buffers.beta[slot].data();
    arguments.raw_output[slot] = buffers.raw_output[slot].data();
    arguments.conv_history[slot] = buffers.history[slot].data();
    arguments.cancellation_snapshot[slot] =
        buffers.cancellation_snapshot[slot].data();
  }
  arguments.output = buffers.output.data();
  arguments.transactional_recurrent_state = buffers.state.data();
  arguments.cancellation_host_word = cancellation.host();
  arguments.cancellation_device_alias = cancellation.device();
  return arguments;
}

void bind_owner(kernels::Sm87BulkV2GdnP40Arguments& arguments,
                ExecutionOwner& owner) noexcept {
  for (std::size_t index = 0U; index < 3U; ++index) {
    arguments.streams[index] = reinterpret_cast<void*>(owner.stream(index));
  }
  for (std::size_t slot = 0U; slot < 2U; ++slot) {
    arguments.prepared_ready_events[slot] =
        reinterpret_cast<void*>(owner.event(slot));
    arguments.recurrence_done_events[slot] =
        reinterpret_cast<void*>(owner.event(2U + slot));
    arguments.epilogue_done_events[slot] =
        reinterpret_cast<void*>(owner.event(4U + slot));
  }
  arguments.submission_receipt = owner.receipt();
}

[[nodiscard]] bool run_candidate(
    kernels::Sm87BulkV2GdnP40Arguments& arguments,
    ExecutionOwner& owner) {
  if (!kernels::sm87_bulk_v2_gdn_p40_arguments_valid(arguments)) {
    std::cerr << "FAIL: candidate P40 arguments do not validate\n";
    return false;
  }
  if (!cuda_ok(static_cast<cudaError_t>(
                   kernels::launch_sm87_bulk_dataflow_v2_gdn_p40_cuda(
                       arguments)),
               "enqueue candidate P40 double-slot pipeline")) {
    return false;
  }
  return cuda_ok(static_cast<cudaError_t>(
                     kernels::drain_sm87_bulk_dataflow_v2_gdn_p40_cuda(
                         arguments)),
                 "drain all three candidate streams") &&
         owner.receipt()->drain_attempted &&
         owner.receipt()->drain_completed && owner.receipt()->reusable &&
         owner.receipt()->lifecycle ==
             kernels::Sm87BulkV2GdnP40OwnerLifecycle::kReady;
}

[[nodiscard]] bool run_serial(
    kernels::Sm87BulkV2GdnP40Arguments& arguments,
    cudaStream_t stream) {
  constexpr auto plan = kernels::sm87_bulk_v2_gdn_p40_plan();
  for (std::size_t index = 0U; index < plan.chunk_count; ++index) {
    const auto chunk = kernels::sm87_bulk_v2_gdn_p40_chunk(plan, index);
    const auto cell = kernels::sm87_bulk_v2_gdn_p40_chunk_arguments(
        arguments, chunk, reinterpret_cast<void*>(stream));
    if (!kernels::sm87_bulk_v2_gdn_c64_arguments_valid(cell) ||
        !cuda_ok(static_cast<cudaError_t>(
                     kernels::launch_sm87_bulk_dataflow_v2_gdn_c64_cuda(
                         cell)),
                 "enqueue serial C64 control")) {
      return false;
    }
  }
  return cuda_ok(cudaStreamSynchronize(stream),
                 "synchronize serial P40 control");
}

[[nodiscard]] bool reset_record(GuardedBuffer<MismatchRecord>& record,
                                cudaStream_t stream) {
  const MismatchRecord initial{0ULL, ULLONG_MAX};
  return cuda_ok(cudaMemcpyAsync(record.data(), &initial, sizeof(initial),
                                 cudaMemcpyHostToDevice, stream),
                 "reset mismatch record");
}

[[nodiscard]] bool read_record(const char* const label,
                               GuardedBuffer<MismatchRecord>& record,
                               cudaStream_t stream) {
  MismatchRecord result{};
  if (!cuda_ok(cudaMemcpyAsync(&result, record.data(), sizeof(result),
                               cudaMemcpyDeviceToHost, stream),
               std::string("read ") + label) ||
      !cuda_ok(cudaStreamSynchronize(stream),
               std::string("synchronize ") + label)) {
    return false;
  }
  if (result.mismatch_count == 0ULL) {
    return true;
  }
  std::cerr << "FAIL: " << label << " has " << result.mismatch_count
            << " mismatching bytes; first index " << result.first_index
            << '\n';
  return false;
}

[[nodiscard]] bool compare_surface(const char* const label,
                                   const void* const candidate,
                                   const void* const control,
                                   const std::size_t bytes,
                                   GuardedBuffer<MismatchRecord>& record,
                                   cudaStream_t stream) {
  if (!reset_record(record, stream)) {
    return false;
  }
  compare_bytes_kernel<<<kBlocks, kThreads, 0U, stream>>>(
      static_cast<const unsigned char*>(candidate),
      static_cast<const unsigned char*>(control), bytes, record.data());
  return cuda_ok(cudaPeekAtLastError(), std::string("compare ") + label) &&
         read_record(label, record, stream);
}

[[nodiscard]] bool verify_uniform_surface(
    const char* const label, const void* const data, const std::size_t bytes,
    const unsigned char expected, GuardedBuffer<MismatchRecord>& record,
    cudaStream_t stream) {
  if (!reset_record(record, stream)) {
    return false;
  }
  verify_uniform_byte_kernel<<<kBlocks, kThreads, 0U, stream>>>(
      static_cast<const unsigned char*>(data), bytes, expected, record.data());
  return cuda_ok(cudaPeekAtLastError(),
                 std::string("verify uniform ") + label) &&
         read_record(label, record, stream);
}

template <class T>
[[nodiscard]] bool verify_guard(const char* const label,
                                const GuardedBuffer<T>& buffer,
                                GuardedBuffer<MismatchRecord>& record,
                                cudaStream_t stream) {
  if (!reset_record(record, stream)) {
    return false;
  }
  verify_guards_kernel<<<1U, 128U, 0U, stream>>>(
      buffer.base_bytes(), buffer.payload_bytes(), record.data());
  return cuda_ok(cudaPeekAtLastError(), std::string("verify guard ") + label) &&
         read_record(label, record, stream);
}

[[nodiscard]] bool compare_pipelines(PipelineBuffers& candidate,
                                     PipelineBuffers& control,
                                     GuardedBuffer<MismatchRecord>& record,
                                     cudaStream_t stream) {
  bool passed = compare_surface("P40 final output", candidate.output.data(),
                                control.output.data(),
                                candidate.output.payload_bytes(), record,
                                stream);
  passed = compare_surface("P40 final recurrent state", candidate.state.data(),
                           control.state.data(), candidate.state.payload_bytes(),
                           record, stream) && passed;
  for (std::size_t slot = 0U; slot < 2U; ++slot) {
    const std::string suffix = " slot " + std::to_string(slot);
    passed = compare_surface(("P40 final history" + suffix).c_str(),
                             candidate.history[slot].data(),
                             control.history[slot].data(),
                             candidate.history[slot].payload_bytes(), record,
                             stream) && passed;
    passed = compare_surface(("P40 raw output" + suffix).c_str(),
                             candidate.raw_output[slot].data(),
                             control.raw_output[slot].data(),
                             candidate.raw_output[slot].payload_bytes(), record,
                             stream) && passed;
    passed = compare_surface(("P40 normalized Q" + suffix).c_str(),
                             candidate.normalized_q[slot].data(),
                             control.normalized_q[slot].data(),
                             candidate.normalized_q[slot].payload_bytes(),
                             record, stream) && passed;
    passed = compare_surface(("P40 normalized K" + suffix).c_str(),
                             candidate.normalized_k[slot].data(),
                             control.normalized_k[slot].data(),
                             candidate.normalized_k[slot].payload_bytes(),
                             record, stream) && passed;
    passed = compare_surface(("P40 prepared V" + suffix).c_str(),
                             candidate.prepared_v[slot].data(),
                             control.prepared_v[slot].data(),
                             candidate.prepared_v[slot].payload_bytes(), record,
                             stream) && passed;
    passed = compare_surface(("P40 alpha" + suffix).c_str(),
                             candidate.alpha[slot].data(),
                             control.alpha[slot].data(),
                             candidate.alpha[slot].payload_bytes(), record,
                             stream) && passed;
    passed = compare_surface(("P40 beta" + suffix).c_str(),
                             candidate.beta[slot].data(),
                             control.beta[slot].data(),
                             candidate.beta[slot].payload_bytes(), record,
                             stream) && passed;
  }
  passed = verify_guard("candidate output canaries", candidate.output, record,
                        stream) && passed;
  passed = verify_guard("candidate state canaries", candidate.state, record,
                        stream) && passed;
  for (std::size_t slot = 0U; slot < 2U; ++slot) {
    passed = verify_guard("candidate history canaries", candidate.history[slot],
                          record, stream) && passed;
    passed = verify_guard("candidate raw-output canaries",
                          candidate.raw_output[slot], record, stream) && passed;
    passed = verify_guard("candidate normalized-Q canaries",
                          candidate.normalized_q[slot], record, stream) &&
             passed;
    passed = verify_guard("candidate normalized-K canaries",
                          candidate.normalized_k[slot], record, stream) &&
             passed;
    passed = verify_guard("candidate prepared-V canaries",
                          candidate.prepared_v[slot], record, stream) && passed;
    passed = verify_guard("candidate alpha canaries", candidate.alpha[slot],
                          record, stream) && passed;
    passed = verify_guard("candidate beta canaries", candidate.beta[slot],
                          record, stream) && passed;
    passed = verify_guard("candidate cancellation-snapshot canaries",
                          candidate.cancellation_snapshot[slot], record,
                          stream) && passed;
  }
  return passed;
}

[[nodiscard]] bool verify_pipeline_poison(
    PipelineBuffers& buffers, const unsigned char poison,
    GuardedBuffer<MismatchRecord>& record, cudaStream_t stream,
    const char* const phase) {
  bool passed = true;
  const auto verify = [&](const char* const label, const void* const data,
                          const std::size_t bytes) {
    return verify_uniform_surface(
        (std::string(phase) + " " + label).c_str(), data, bytes, poison, record,
        stream);
  };
  passed = verify("output", buffers.output.data(),
                  buffers.output.payload_bytes()) &&
           passed;
  passed = verify("state", buffers.state.data(), buffers.state.payload_bytes()) &&
           passed;
  passed = verify_guard("poison output canaries", buffers.output, record,
                        stream) && passed;
  passed = verify_guard("poison state canaries", buffers.state, record, stream) &&
           passed;
  for (std::size_t slot = 0U; slot < 2U; ++slot) {
    passed = verify("history", buffers.history[slot].data(),
                    buffers.history[slot].payload_bytes()) &&
             passed;
    passed = verify("raw output", buffers.raw_output[slot].data(),
                    buffers.raw_output[slot].payload_bytes()) &&
             passed;
    passed = verify("normalized Q", buffers.normalized_q[slot].data(),
                    buffers.normalized_q[slot].payload_bytes()) &&
             passed;
    passed = verify("normalized K", buffers.normalized_k[slot].data(),
                    buffers.normalized_k[slot].payload_bytes()) &&
             passed;
    passed = verify("prepared V", buffers.prepared_v[slot].data(),
                    buffers.prepared_v[slot].payload_bytes()) &&
             passed;
    passed = verify("alpha", buffers.alpha[slot].data(),
                    buffers.alpha[slot].payload_bytes()) &&
             passed;
    passed = verify("beta", buffers.beta[slot].data(),
                    buffers.beta[slot].payload_bytes()) &&
             passed;
    passed = verify_guard("poison history canaries", buffers.history[slot],
                          record, stream) && passed;
    passed = verify_guard("poison raw-output canaries",
                          buffers.raw_output[slot], record, stream) && passed;
    passed = verify_guard("poison normalized-Q canaries",
                          buffers.normalized_q[slot], record, stream) && passed;
    passed = verify_guard("poison normalized-K canaries",
                          buffers.normalized_k[slot], record, stream) && passed;
    passed = verify_guard("poison prepared-V canaries",
                          buffers.prepared_v[slot], record, stream) && passed;
    passed = verify_guard("poison alpha canaries", buffers.alpha[slot], record,
                          stream) && passed;
    passed = verify_guard("poison beta canaries", buffers.beta[slot], record,
                          stream) && passed;
    passed = verify_guard("poison cancellation-snapshot canaries",
                          buffers.cancellation_snapshot[slot], record,
                          stream) && passed;
  }
  return passed;
}

[[nodiscard]] bool verify_u32_value(
    const char* const label, const std::uint32_t* const device_value,
    const std::uint32_t expected, cudaStream_t stream) {
  std::uint32_t observed = 0U;
  return cuda_ok(cudaMemcpyAsync(&observed, device_value, sizeof(observed),
                                 cudaMemcpyDeviceToHost, stream),
                 std::string("read ") + label) &&
         cuda_ok(cudaStreamSynchronize(stream),
                 std::string("synchronize ") + label) &&
         (observed == expected ||
          (std::cerr << "FAIL: " << label << " expected " << expected
                     << " observed " << observed << '\n',
           false));
}

[[nodiscard]] bool reject_undersized_allocation_without_enqueue(
    kernels::Sm87BulkV2GdnP40Arguments& arguments,
    PipelineBuffers& candidate, GuardedBuffer<MismatchRecord>& record,
    cudaStream_t verification_stream) {
  void* undersized = nullptr;
  constexpr std::size_t kUndersizedBytes =
      kernels::kSm87BulkV2GdnNormalizedQBytes - 16U;
  if (!cuda_ok(cudaMalloc(&undersized, kUndersizedBytes),
               "allocate intentionally undersized Q slot")) {
    return false;
  }
  auto rejected = arguments;
  rejected.normalized_q[0U] = static_cast<float*>(undersized);
  const auto before = *arguments.submission_receipt;
  const int status =
      kernels::launch_sm87_bulk_dataflow_v2_gdn_p40_cuda(rejected);
  const auto after = *arguments.submission_receipt;
  const bool receipt_unchanged =
      before.generation == after.generation &&
      before.successful_submission_calls == after.successful_submission_calls &&
      !after.submission_started && !after.drain_attempted && after.reusable &&
      after.lifecycle == kernels::Sm87BulkV2GdnP40OwnerLifecycle::kReady;
  const bool freed = cuda_ok(cudaFree(undersized),
                             "free intentionally undersized Q slot");
  return (status == static_cast<int>(cudaErrorInvalidDevicePointer) ||
          (std::cerr << "FAIL: undersized allocation returned " << status
                     << " instead of cudaErrorInvalidDevicePointer\n",
           false)) &&
         (receipt_unchanged ||
          (std::cerr << "FAIL: undersized allocation enqueued or mutated the "
                        "owner receipt\n",
           false)) &&
         verify_pipeline_poison(candidate, 0xa5U, record, verification_stream,
                                "undersized-zero-enqueue") &&
         freed;
}

template <class T>
[[nodiscard]] bool verify_input(const char* const label,
                                const GuardedBuffer<T>& input,
                                const PatternKind kind,
                                const std::uint64_t salt,
                                GuardedBuffer<MismatchRecord>& record,
                                cudaStream_t stream) {
  static_assert(sizeof(T) == sizeof(std::uint16_t));
  if (!reset_record(record, stream)) {
    return false;
  }
  verify_pattern_kernel<<<kBlocks, kThreads, 0U, stream>>>(
      reinterpret_cast<const std::uint16_t*>(input.data()), input.size(), kind,
      salt, record.data());
  return cuda_ok(cudaPeekAtLastError(), std::string("verify input ") + label) &&
         read_record(label, record, stream) &&
         verify_guard(label, input, record, stream);
}

[[nodiscard]] bool verify_inputs(const Inputs& inputs,
                                 GuardedBuffer<MismatchRecord>& record,
                                 cudaStream_t stream) {
  bool passed = true;
#define Q3X_VERIFY_INPUT(field, kind, salt)                                    \
  passed = verify_input(#field, inputs.field, PatternKind::kind, salt, record, \
                        stream) && passed
  Q3X_VERIFY_INPUT(raw_qkvz, kSignal, 0x101ULL);
  Q3X_VERIFY_INPUT(interleaved_ab, kGate, 0x202ULL);
  Q3X_VERIFY_INPUT(conv_weight, kConvWeight, 0x303ULL);
  Q3X_VERIFY_INPUT(initial_history, kSignal, 0x404ULL);
  Q3X_VERIFY_INPUT(a_log, kALog, 0x505ULL);
  Q3X_VERIFY_INPUT(dt_bias, kDtBias, 0x606ULL);
  Q3X_VERIFY_INPUT(norm_weight, kNormWeight, 0x707ULL);
  Q3X_VERIFY_INPUT(initial_state, kState, 0x808ULL);
#undef Q3X_VERIFY_INPUT
  return passed;
}

}  // namespace

int main() {
  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA device unavailable for SM87 GDN P40 oracle\n";
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
    std::cout << "SKIP: oracle requires the fixed 16-SM SM87 target\n";
    return 77;
  }

  ExecutionOwner candidate_owner;
  ExecutionOwner control_owner;
  CancellationOwner cancellation;
  if (!candidate_owner.create() || !control_owner.create() ||
      !cancellation.create()) {
    return 1;
  }

  bool passed = true;
  {
    Inputs inputs;
    PipelineBuffers candidate;
    PipelineBuffers control;
    GuardedBuffer<MismatchRecord> mismatch;
    passed = allocate_inputs(inputs);
    passed = allocate_pipeline(candidate, "candidate P40 pipeline") && passed;
    passed = allocate_pipeline(control, "serial P40 control") && passed;
    passed = allocate(mismatch, 1U, "mismatch record") && passed;
    if (passed) {
      passed = initialize_inputs(inputs, control_owner.stream(0U));
      passed = initialize_pipeline(candidate, 0xa5U,
                                   control_owner.stream(0U), "candidate") &&
               passed;
      passed = initialize_pipeline(control, 0x5aU, control_owner.stream(0U),
                                   "control") && passed;
    }

    auto candidate_arguments =
        make_arguments(inputs, candidate, cancellation);
    auto control_arguments =
        make_arguments(inputs, control, cancellation);
    bind_owner(candidate_arguments, candidate_owner);
    if (passed) {
      passed = reject_undersized_allocation_without_enqueue(
          candidate_arguments, candidate, mismatch, control_owner.stream(0U));
    }

    // Only the exact cudaHostAllocMapped host/device pair is an admissible
    // cancellation owner. Ordinary device and managed words must fail before
    // the first submission and leave the owner receipt reusable.
    if (passed) {
      void* device_word = nullptr;
      void* managed_word = nullptr;
      passed = cuda_ok(cudaMalloc(&device_word, sizeof(std::uint32_t)),
                       "allocate rejected device cancellation word");
      passed = cuda_ok(cudaMallocManaged(&managed_word, sizeof(std::uint32_t)),
                       "allocate rejected managed cancellation word") &&
               passed;
      if (passed) {
        auto rejected = candidate_arguments;
        rejected.cancellation_device_alias =
            static_cast<const std::uint32_t*>(device_word);
        const auto before = *candidate_owner.receipt();
        const int device_status =
            kernels::launch_sm87_bulk_dataflow_v2_gdn_p40_cuda(rejected);
        rejected.cancellation_host_word =
            static_cast<std::uint32_t*>(managed_word);
        rejected.cancellation_device_alias =
            static_cast<const std::uint32_t*>(managed_word);
        const int managed_status =
            kernels::launch_sm87_bulk_dataflow_v2_gdn_p40_cuda(rejected);
        const auto after = *candidate_owner.receipt();
        passed = device_status == static_cast<int>(cudaErrorInvalidDevicePointer) &&
                 managed_status == static_cast<int>(cudaErrorInvalidDevicePointer) &&
                 before.generation == after.generation && after.reusable &&
                 !after.submission_started;
        if (!passed) {
          std::cerr << "FAIL: non-mapped cancellation owner was not rejected "
                       "without submission\n";
        }
      }
      passed = cuda_ok(cudaFree(device_word),
                       "free rejected device cancellation word") && passed;
      passed = cuda_ok(cudaFree(managed_word),
                       "free rejected managed cancellation word") && passed;
    }
    if (passed) {
      passed = run_candidate(candidate_arguments, candidate_owner);
      passed = run_serial(control_arguments, control_owner.stream(0U)) &&
               passed;
    }
    if (passed) {
      passed = compare_pipelines(candidate, control, mismatch,
                                 control_owner.stream(0U));
      passed = verify_inputs(inputs, mismatch, control_owner.stream(0U)) &&
               passed;
    }

    // Reuse the same owner streams/events and bounded slots.  A second exact
    // pass catches stale-event and incomplete slot-reinitialization hazards.
    if (passed) {
      passed = initialize_pipeline(candidate, 0x3cU,
                                   control_owner.stream(0U), "candidate replay");
      candidate_arguments =
          make_arguments(inputs, candidate, cancellation);
      bind_owner(candidate_arguments, candidate_owner);
      passed = run_candidate(candidate_arguments, candidate_owner) && passed;
      passed = compare_pipelines(candidate, control, mismatch,
                                 control_owner.stream(0U)) && passed;
      passed = verify_inputs(inputs, mismatch, control_owner.stream(0U)) &&
               passed;
    }


    // A preasserted owner cancellation word must suppress every producer,
    // recurrence, epilogue and final-state publication. All numerical private
    // surfaces remain poison; only the two device control snapshots become 1.
    if (passed) {
      cancellation.store(1U);
      passed = initialize_pipeline(candidate, 0xc7U,
                                   control_owner.stream(0U),
                                   "pre-cancelled candidate");
      candidate_arguments =
          make_arguments(inputs, candidate, cancellation);
      bind_owner(candidate_arguments, candidate_owner);
      passed = run_candidate(candidate_arguments, candidate_owner) && passed;
      passed = verify_pipeline_poison(candidate, 0xc7U, mismatch,
                                      control_owner.stream(0U),
                                      "pre-cancel") &&
               passed;
      for (std::size_t slot = 0U; slot < 2U; ++slot) {
        passed = verify_u32_value(
                     "pre-cancel device snapshot",
                     candidate.cancellation_snapshot[slot].data(), 1U,
                     control_owner.stream(0U)) &&
                 passed;
      }
      passed = verify_inputs(inputs, mismatch, control_owner.stream(0U)) &&
               passed;
      cancellation.store(0U);
    }

    // Deterministic mid-flight cancellation executes a producer-stream host
    // callback immediately before C64 chunk four's sampler. Epilogue output
    // from that boundary onward must remain poison; explicit retirement drains
    // producer, recurrence and epilogue and leaves the owner reusable.
    if (passed) {
      constexpr std::size_t kCancelBeforeChunk = 4U;
      constexpr std::size_t kCancelledOutputByteOffset =
          kCancelBeforeChunk * kernels::kSm87BulkV2GdnC64Tokens *
          kernels::kSm87TargetAotGdnOutputChannels * sizeof(std::uint16_t);
      passed = initialize_pipeline(candidate, 0xb4U,
                                   control_owner.stream(0U),
                                   "mid-flight cancellation") &&
               passed;
      candidate_arguments = make_arguments(inputs, candidate, cancellation);
      candidate_arguments.test_only_cancel_before_chunk = kCancelBeforeChunk;
      bind_owner(candidate_arguments, candidate_owner);
      passed = run_candidate(candidate_arguments, candidate_owner) && passed;
      passed = verify_uniform_surface(
                   "mid-flight cancelled output tail",
                   candidate.output.bytes() + kCancelledOutputByteOffset,
                   candidate.output.payload_bytes() -
                       kCancelledOutputByteOffset,
                   0xb4U, mismatch, control_owner.stream(0U)) &&
               passed;
      passed = verify_guard("mid-flight output canaries", candidate.output,
                            mismatch, control_owner.stream(0U)) && passed;
      const std::uint32_t cancellation_value = cancellation.load();
      passed = (cancellation_value == 1U) && passed;
      if (cancellation_value != 1U) {
        std::cerr << "FAIL: deterministic mid-flight cancellation callback "
                     "did not update the owner word\n";
      }
      passed = verify_inputs(inputs, mismatch, control_owner.stream(0U)) &&
               passed;

      // Clearing the owner word after a complete three-stream drain must make
      // the exact same stream/event owner replayable without stale snapshots.
      cancellation.store(0U);
      passed = initialize_pipeline(candidate, 0x91U,
                                   control_owner.stream(0U),
                                   "post-cancel replay") &&
               passed;
      candidate_arguments = make_arguments(inputs, candidate, cancellation);
      bind_owner(candidate_arguments, candidate_owner);
      passed = run_candidate(candidate_arguments, candidate_owner) && passed;
      passed = compare_pipelines(candidate, control, mismatch,
                                 control_owner.stream(0U)) && passed;
      passed = verify_inputs(inputs, mismatch, control_owner.stream(0U)) &&
               passed;
    }

    // Inject a host-side submission failure after work has entered the DAG.
    // The launcher must preserve the queued count, drain all streams, poison
    // the owner permanently, and reject a second launch using the same receipt.
    if (passed) {
      passed = initialize_pipeline(candidate, 0xe1U,
                                   control_owner.stream(0U),
                                   "submission failure") &&
               passed;
      candidate_arguments = make_arguments(inputs, candidate, cancellation);
      candidate_arguments.test_only_fail_after_successful_submissions = 8U;
      bind_owner(candidate_arguments, candidate_owner);
      const std::uint64_t expected_generation =
          candidate_owner.receipt()->generation + 1U;
      const int injected =
          kernels::launch_sm87_bulk_dataflow_v2_gdn_p40_cuda(
              candidate_arguments);
      const auto failed_receipt = *candidate_owner.receipt();
      const int rejected_reuse =
          kernels::launch_sm87_bulk_dataflow_v2_gdn_p40_cuda(
              candidate_arguments);
      passed = injected == static_cast<int>(cudaErrorUnknown) &&
               rejected_reuse ==
                   static_cast<int>(cudaErrorInvalidResourceHandle) &&
               failed_receipt.generation == expected_generation &&
               failed_receipt.successful_submission_calls == 8U &&
               failed_receipt.first_error ==
                   static_cast<std::int32_t>(cudaErrorUnknown) &&
               failed_receipt.submission_started &&
               failed_receipt.drain_attempted && failed_receipt.drain_completed &&
               !failed_receipt.reusable &&
               failed_receipt.lifecycle ==
                   kernels::Sm87BulkV2GdnP40OwnerLifecycle::kPoisoned;
      if (!passed) {
        std::cerr << "FAIL: injected host submission failure did not retain "
                     "queued/drained/poisoned receipt semantics\n";
      }
      passed = verify_inputs(inputs, mismatch, control_owner.stream(0U)) &&
               passed;
    }
  }

  if (!passed) {
    return 1;
  }
  std::cout << "SM87 bulk-dataflow-v2 GDN P40000 double-slot/three-stream "
               "bitwise scheduling oracle passed (correctness only; no "
               "performance or production authority)\n";
  return 0;
}
