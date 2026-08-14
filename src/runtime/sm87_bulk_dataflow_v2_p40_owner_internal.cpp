#include "sm87_bulk_dataflow_v2_p40_owner_internal.h"

#include "sm87_bulk_dataflow_v2_p40_request_state_internal.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail {
namespace {

std::atomic<std::uint64_t> g_next_v2_owner_identity{1U};

[[nodiscard]] std::uint64_t next_identity() noexcept {
  for (;;) {
    const std::uint64_t value =
        g_next_v2_owner_identity.fetch_add(1U, std::memory_order_relaxed);
    if (value != 0U) {
      return value;
    }
  }
}

[[nodiscard]] constexpr Sm87BulkV2P40OwnerStatus ok() noexcept {
  return {};
}

[[nodiscard]] constexpr Sm87BulkV2P40OwnerStatus error(
    const Sm87BulkV2P40OwnerError code, const char* const context,
    const int cuda_error = 0,
    const std::size_t resource_index =
        std::numeric_limits<std::size_t>::max()) noexcept {
  return {code, context, cuda_error, resource_index};
}

[[nodiscard]] bool identity_equal(
    const Sm87BulkV2P40OwnerIdentity& left,
    const Sm87BulkV2P40OwnerIdentity& right) noexcept {
  return sm87_bulk_v2_p40_magic_equal(left.plan_magic, right.plan_magic) &&
         left.abi_major == right.abi_major &&
         left.abi_minor == right.abi_minor &&
         left.owner_identity == right.owner_identity &&
         left.seal_nonce == right.seal_nonce &&
         left.deployment_identity == right.deployment_identity &&
         left.model_identity == right.model_identity &&
         left.request_allocation_identity ==
             right.request_allocation_identity &&
         left.stream_event_owner_identity ==
             right.stream_event_owner_identity &&
         left.asset_catalog_identity == right.asset_catalog_identity &&
         left.binary_evidence_identity == right.binary_evidence_identity &&
         left.fp8_oracle_evidence_identity ==
             right.fp8_oracle_evidence_identity &&
         left.attention_oracle_evidence_identity ==
             right.attention_oracle_evidence_identity &&
         left.gdn_oracle_evidence_identity ==
             right.gdn_oracle_evidence_identity &&
         left.nvfp4_oracle_evidence_identity ==
             right.nvfp4_oracle_evidence_identity &&
         left.device_ordinal == right.device_ordinal &&
         left.execution_class == right.execution_class &&
         left.authenticated_real_constituents ==
             right.authenticated_real_constituents &&
         left.exact_numerical_contract_qualified ==
             right.exact_numerical_contract_qualified &&
         left.development_execution_eligible ==
             right.development_execution_eligible &&
         left.production_dispatch_eligible ==
             right.production_dispatch_eligible;
}

[[nodiscard]] constexpr std::size_t stream_index(
    const Sm87BulkV2P40Stream stream) noexcept {
  return static_cast<std::size_t>(stream);
}

[[nodiscard]] constexpr std::size_t event_index(
    const Sm87BulkV2P40ReusableEvent event) noexcept {
  return static_cast<std::size_t>(event);
}

[[nodiscard]] constexpr bool valid_stream(
    const Sm87BulkV2P40Stream stream) noexcept {
  return stream_index(stream) < kSm87BulkV2P40StreamCount;
}

[[nodiscard]] constexpr bool valid_event(
    const Sm87BulkV2P40ReusableEvent event) noexcept {
  return event_index(event) < kSm87BulkV2P40ReusableEventCount;
}

[[nodiscard]] constexpr std::uint32_t auxiliary_stream_mask() noexcept {
  return ((1U << static_cast<std::uint32_t>(kSm87BulkV2P40StreamCount)) -
          1U) &
         ~1U;
}

[[nodiscard]] bool checked_increment(std::size_t* const destination,
                                     const std::size_t count) noexcept {
  if (destination == nullptr || count == 0U ||
      *destination > std::numeric_limits<std::size_t>::max() - count) {
    return false;
  }
  *destination += count;
  return true;
}

inline constexpr std::size_t kGdnChunksPerLayer =
    q3x::kernels::kSm87BulkV2GdnP40Chunks;
inline constexpr std::size_t kAttentionPreprocessPanelsPerFullLayer = 5U;
inline constexpr std::size_t kGdnPersistentCopiesPerLayer = 2U;
inline constexpr std::size_t kGdnLogicalProjectionRolesPerLayer = 8U;
inline constexpr std::size_t kFullLogicalProjectionRolesPerLayer = 7U;
inline constexpr std::size_t kGdnFusedOuterOperationsPerLayer = 5U;
inline constexpr std::size_t kFullFusedOuterOperationsPerLayer = 4U;
inline constexpr std::uint64_t kNvFp4ConventionalOperationsPerLayer =
    6ULL * kSm87BulkV2P40Tokens * kSm87BulkV2P40Hidden *
    kSm87BulkV2P40Intermediate;
inline constexpr std::uint64_t kFp8OutputConventionalOperationsPerLayer =
    2ULL * kSm87BulkV2P40Tokens * kSm87BulkV2P40AttentionWidth *
    kSm87BulkV2P40Hidden;
inline constexpr std::uint64_t kFp8GdnInputConventionalOperationsPerLayer =
    2ULL * kSm87BulkV2P40Tokens * kSm87BulkV2P40Hidden *
    kSm87BulkV2P40GdnRawWidth;
inline constexpr std::uint64_t kFp8FullInputConventionalOperationsPerLayer =
    2ULL * kSm87BulkV2P40Tokens * kSm87BulkV2P40Hidden *
    (kSm87BulkV2P40AttentionQGateWidth +
     2ULL * kSm87TargetAotP40KvWidth);
inline constexpr std::uint64_t kBf16AbConventionalOperationsPerLayer =
    2ULL * kSm87BulkV2P40Tokens * kSm87BulkV2P40Hidden *
    kSm87BulkV2P40GdnAbWidth;
inline constexpr std::uint64_t kGdnConventionalOperationsPerLayer =
    kNvFp4ConventionalOperationsPerLayer +
    kFp8OutputConventionalOperationsPerLayer +
    kFp8GdnInputConventionalOperationsPerLayer +
    kBf16AbConventionalOperationsPerLayer;
inline constexpr std::uint64_t kFullConventionalOperationsPerLayer =
    kNvFp4ConventionalOperationsPerLayer +
    kFp8OutputConventionalOperationsPerLayer +
    kFp8FullInputConventionalOperationsPerLayer;

static_assert(kGdnChunksPerLayer == 625U);
static_assert(kSm87BulkV2P40GdnLayers *
                      kGdnLogicalProjectionRolesPerLayer +
                  kSm87BulkV2P40FullLayers *
                      kFullLogicalProjectionRolesPerLayer ==
              kSm87BulkV2P40LogicalProjectionRoles);
static_assert(kSm87BulkV2P40GdnLayers *
                      kGdnFusedOuterOperationsPerLayer +
                  kSm87BulkV2P40FullLayers *
                      kFullFusedOuterOperationsPerLayer ==
              kSm87BulkV2P40FusedOuterOperations);
static_assert(kSm87BulkV2P40GdnLayers *
                          kGdnConventionalOperationsPerLayer +
                      kSm87BulkV2P40FullLayers *
                          kFullConventionalOperationsPerLayer ==
                  kSm87BulkV2P40ProjectionConventionalOperations);

struct Sm87BulkV2P40LayerPrefix final {
  std::size_t layers = 0U;
  std::size_t gdn_layers = 0U;
  std::size_t full_layers = 0U;
};

[[nodiscard]] constexpr Sm87BulkV2P40LayerPrefix layer_prefix(
    const std::size_t model_layer) noexcept {
  const std::size_t layers = model_layer + 1U;
  const std::size_t full_layers = layers / 4U;
  return {layers, layers - full_layers, full_layers};
}

[[nodiscard]] bool layer_work_exact(
    const Sm87BulkV2P40RequestReceipt& receipt,
    const Sm87BulkV2P40LayerPrefix& prefix) noexcept {
  const auto& successor = receipt.projection_successor;
  return successor.fp8_gdn_input_whole_launches == prefix.gdn_layers &&
         successor.fp8_full_input_whole_launches == prefix.full_layers &&
         successor.fp8_output_whole_launches == prefix.layers &&
         successor.fp8_whole_role_launches ==
             prefix.gdn_layers + prefix.full_layers + prefix.layers &&
         successor.nvfp4_gate_up_whole_launches == prefix.layers &&
         successor.nvfp4_down_whole_launches == prefix.layers &&
         successor.nvfp4_whole_role_launches == 2U * prefix.layers &&
         successor.bf16_ab_physical_launches == prefix.gdn_layers &&
         receipt.enqueued_attention_launches ==
             prefix.full_layers *
                 q3x::kernels::kSm87BulkV2AttentionKernelLaunches &&
         receipt.enqueued_attention_preprocess_panels ==
             prefix.full_layers * kAttentionPreprocessPanelsPerFullLayer &&
         receipt.enqueued_bf16_ab_launches == prefix.gdn_layers &&
         receipt.enqueued_gdn_producer_chunks ==
             prefix.gdn_layers * kGdnChunksPerLayer &&
         receipt.enqueued_gdn_recurrence_chunks ==
             prefix.gdn_layers * kGdnChunksPerLayer &&
         receipt.enqueued_gdn_epilogue_chunks ==
             prefix.gdn_layers * kGdnChunksPerLayer &&
         receipt.enqueued_gdn_persistent_copies ==
             prefix.gdn_layers * kGdnPersistentCopiesPerLayer &&
         receipt.enqueued_final_norm == 0U && receipt.enqueued_lm_head == 0U &&
         receipt.enqueued_argmax == 0U && receipt.enqueued_handoff_d2h == 0U;
}

[[nodiscard]] constexpr bool final_submission_counter(
    const Sm87BulkV2P40SubmissionCounter counter) noexcept {
  return counter == Sm87BulkV2P40SubmissionCounter::kFinalNorm ||
         counter == Sm87BulkV2P40SubmissionCounter::kLmHead ||
         counter == Sm87BulkV2P40SubmissionCounter::kArgmax ||
         counter == Sm87BulkV2P40SubmissionCounter::kHandoffD2h;
}

[[nodiscard]] constexpr bool gdn_only_submission_counter(
    const Sm87BulkV2P40SubmissionCounter counter) noexcept {
  return counter ==
             Sm87BulkV2P40SubmissionCounter::kFp8GdnInputWholeRoleLaunch ||
         counter == Sm87BulkV2P40SubmissionCounter::kBf16AbLaunch ||
         counter == Sm87BulkV2P40SubmissionCounter::kGdnProducerChunk ||
         counter == Sm87BulkV2P40SubmissionCounter::kGdnRecurrenceChunk ||
         counter == Sm87BulkV2P40SubmissionCounter::kGdnEpilogueChunk ||
         counter == Sm87BulkV2P40SubmissionCounter::kGdnPersistentCopy;
}

[[nodiscard]] constexpr bool full_only_submission_counter(
    const Sm87BulkV2P40SubmissionCounter counter) noexcept {
  return counter ==
             Sm87BulkV2P40SubmissionCounter::kFp8FullInputWholeRoleLaunch ||
         counter == Sm87BulkV2P40SubmissionCounter::kAttentionLaunch ||
         counter ==
             Sm87BulkV2P40SubmissionCounter::kAttentionPreprocessPanel;
}

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_OWNER_ADMISSION)
class CudartSm87BulkV2P40Runtime final
    : public Sm87BulkV2P40CudaRuntime {
 public:
  [[nodiscard]] int get_current_device(
      std::int32_t* const device_ordinal) noexcept override {
    return static_cast<int>(cudaGetDevice(device_ordinal));
  }

  [[nodiscard]] int get_device_properties(
      const std::int32_t device_ordinal,
      Sm87BulkV2P40DeviceProperties* const properties) noexcept override {
    if (properties == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaDeviceProp raw{};
    const cudaError_t status = cudaGetDeviceProperties(&raw, device_ordinal);
    if (status == cudaSuccess) {
      properties->major = raw.major;
      properties->minor = raw.minor;
      properties->multiprocessor_count = raw.multiProcessorCount;
    }
    return static_cast<int>(status);
  }

  [[nodiscard]] int create_nonblocking_stream(
      void** const stream) noexcept override {
    if (stream == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t raw = nullptr;
    const cudaError_t status =
        cudaStreamCreateWithFlags(&raw, cudaStreamNonBlocking);
    if (status == cudaSuccess) {
      *stream = reinterpret_cast<void*>(raw);
    }
    return static_cast<int>(status);
  }

  [[nodiscard]] int get_stream_flags(
      void* const stream, unsigned int* const flags) noexcept override {
    return static_cast<int>(cudaStreamGetFlags(
        reinterpret_cast<cudaStream_t>(stream), flags));
  }

  [[nodiscard]] int destroy_stream(void* const stream) noexcept override {
    return static_cast<int>(
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
  }

  [[nodiscard]] int create_disable_timing_event(
      void** const event) noexcept override {
    if (event == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaEvent_t raw = nullptr;
    const cudaError_t status =
        cudaEventCreateWithFlags(&raw, cudaEventDisableTiming);
    if (status == cudaSuccess) {
      *event = reinterpret_cast<void*>(raw);
    }
    return static_cast<int>(status);
  }

  [[nodiscard]] int destroy_event(void* const event) noexcept override {
    return static_cast<int>(
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event)));
  }

  [[nodiscard]] int allocate_device(void** const pointer,
                                    const std::size_t bytes) noexcept override {
    return static_cast<int>(cudaMalloc(pointer, bytes));
  }

  [[nodiscard]] int free_device(void* const pointer) noexcept override {
    return static_cast<int>(cudaFree(pointer));
  }

  [[nodiscard]] int allocate_mapped_host(
      void** const pointer, const std::size_t bytes) noexcept override {
    return static_cast<int>(cudaHostAlloc(pointer, bytes, cudaHostAllocMapped));
  }

  [[nodiscard]] int mapped_device_alias(
      void** const device_alias, void* const host_pointer) noexcept override {
    return static_cast<int>(
        cudaHostGetDevicePointer(device_alias, host_pointer, 0U));
  }

  [[nodiscard]] int free_mapped_host(void* const pointer) noexcept override {
    return static_cast<int>(cudaFreeHost(pointer));
  }

  [[nodiscard]] int query_pointer(
      const void* const pointer,
      Sm87BulkV2P40PointerAttributes* const attributes) noexcept override {
    if (attributes == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaPointerAttributes raw{};
    const cudaError_t status = cudaPointerGetAttributes(&raw, pointer);
    if (status == cudaSuccess) {
      attributes->kind =
          raw.type == cudaMemoryTypeHost
              ? Sm87BulkV2P40PointerKind::kHost
              : (raw.type == cudaMemoryTypeDevice
                     ? Sm87BulkV2P40PointerKind::kDevice
                     : Sm87BulkV2P40PointerKind::kUnknown);
      attributes->host_pointer = raw.hostPointer;
      attributes->device_pointer = raw.devicePointer;
      attributes->device_ordinal = raw.device;
    }
    return static_cast<int>(status);
  }

  [[nodiscard]] int memset_async(void* const pointer, const int value,
                                 const std::size_t bytes,
                                 void* const stream) noexcept override {
    return static_cast<int>(cudaMemsetAsync(
        pointer, value, bytes, reinterpret_cast<cudaStream_t>(stream)));
  }

  [[nodiscard]] int record_event(void* const event,
                                 void* const stream) noexcept override {
    return static_cast<int>(cudaEventRecord(
        reinterpret_cast<cudaEvent_t>(event),
        reinterpret_cast<cudaStream_t>(stream)));
  }

  [[nodiscard]] int stream_wait_event(void* const stream,
                                      void* const event) noexcept override {
    return static_cast<int>(cudaStreamWaitEvent(
        reinterpret_cast<cudaStream_t>(stream),
        reinterpret_cast<cudaEvent_t>(event), 0U));
  }

  [[nodiscard]] int synchronize_stream(
      void* const stream) noexcept override {
    return static_cast<int>(
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
  }
};
#endif

[[nodiscard]] Sm87BulkV2P40OwnerReceipt make_active_receipt(
    const Sm87BulkV2P40OwnerIdentity& identity,
    const std::uint64_t request_epoch) noexcept {
  Sm87BulkV2P40OwnerReceipt receipt;
  receipt.identity = identity;
  receipt.aggregate.plan_magic = identity.plan_magic;
  receipt.aggregate.abi_major = identity.abi_major;
  receipt.aggregate.abi_minor = identity.abi_minor;
  receipt.aggregate.lifecycle = Sm87BulkV2P40OwnerLifecycle::kActive;
  receipt.aggregate.seal_nonce = identity.seal_nonce;
  receipt.aggregate.request_epoch = request_epoch;
  receipt.aggregate.deployment_identity = identity.deployment_identity;
  receipt.aggregate.model_identity = identity.model_identity;
  receipt.aggregate.allocation_identity =
      identity.request_allocation_identity;
  receipt.aggregate.stream_event_owner_identity =
      identity.stream_event_owner_identity;
  receipt.aggregate.asset_catalog_identity = identity.asset_catalog_identity;
  receipt.aggregate.projection_successor =
      sm87_bulk_v2_p40_projection_successor_receipt();
  receipt.owner_bound_capability_used = true;
  receipt.public_aggregate_used_as_authority = false;
  return receipt;
}

}  // namespace

bool Sm87BulkV2P40OwnerIdentity::valid() const noexcept {
  const bool common =
      sm87_bulk_v2_p40_magic_equal(plan_magic,
                                   kSm87BulkV2P40PlanMagic) &&
      abi_major == kSm87BulkV2P40PlanAbiMajor &&
      abi_minor == kSm87BulkV2P40PlanAbiMinor && owner_identity != 0U &&
      seal_nonce != 0U && deployment_identity != 0U &&
      model_identity != 0U && request_allocation_identity != 0U &&
      stream_event_owner_identity != 0U && asset_catalog_identity != 0U &&
      binary_evidence_identity != 0U &&
      fp8_oracle_evidence_identity != 0U &&
      attention_oracle_evidence_identity != 0U &&
      gdn_oracle_evidence_identity != 0U &&
      nvfp4_oracle_evidence_identity != 0U && device_ordinal >= 0;
  return common &&
         (direction_witness_valid() || development_candidate_valid() ||
          synthetic_host_contract_valid());
}

bool Sm87BulkV2P40OwnerIdentity::direction_witness_valid()
    const noexcept {
  return execution_class ==
             Sm87BulkV2P40ExecutionClass::kDefaultOffDirectionWitness &&
         authenticated_real_constituents &&
         !exact_numerical_contract_qualified &&
         development_execution_eligible &&
         !production_dispatch_eligible;
}

bool Sm87BulkV2P40OwnerIdentity::development_candidate_valid()
    const noexcept {
  return execution_class ==
             Sm87BulkV2P40ExecutionClass::kDefaultOffDevelopmentCandidate &&
         authenticated_real_constituents &&
         exact_numerical_contract_qualified &&
         development_execution_eligible &&
         !production_dispatch_eligible;
}

bool Sm87BulkV2P40OwnerIdentity::synthetic_host_contract_valid()
    const noexcept {
  return execution_class ==
             Sm87BulkV2P40ExecutionClass::kSyntheticHostContract &&
         !authenticated_real_constituents &&
         !exact_numerical_contract_qualified &&
         !development_execution_eligible &&
         !production_dispatch_eligible;
}

bool Sm87BulkV2P40OwnerReceipt::identity_valid() const noexcept {
  return identity.valid() && owner_bound_capability_used &&
         !public_aggregate_used_as_authority &&
         aggregate.request_epoch != 0U &&
         sm87_bulk_v2_p40_magic_equal(aggregate.plan_magic,
                                      identity.plan_magic) &&
         aggregate.abi_major == identity.abi_major &&
         aggregate.abi_minor == identity.abi_minor &&
         aggregate.seal_nonce == identity.seal_nonce &&
         aggregate.deployment_identity == identity.deployment_identity &&
         aggregate.model_identity == identity.model_identity &&
         aggregate.allocation_identity ==
             identity.request_allocation_identity &&
         aggregate.stream_event_owner_identity ==
             identity.stream_event_owner_identity &&
         aggregate.asset_catalog_identity == identity.asset_catalog_identity;
}

Sm87BulkV2P40ExecutionAccess::Sm87BulkV2P40ExecutionAccess(
    const Sm87BulkV2P40Owner* const owner,
    const Sm87BulkV2P40OwnerIdentity& identity,
    const std::array<void*, kSm87BulkV2P40StreamCount>& streams,
    const std::array<void*, kSm87BulkV2P40ReusableEventCount>& events,
    void* const device_control_arena,
    const std::uint32_t* const cancellation_device_alias) noexcept
    : owner_(owner),
      identity_(identity),
      streams_(streams),
      events_(events),
      device_control_arena_(device_control_arena),
      cancellation_device_alias_(cancellation_device_alias) {}

void* Sm87BulkV2P40ExecutionAccess::cuda_stream(
    const Sm87BulkV2P40Stream stream) const noexcept {
  return valid_stream(stream) ? streams_[stream_index(stream)] : nullptr;
}

void* Sm87BulkV2P40ExecutionAccess::cuda_event(
    const Sm87BulkV2P40ReusableEvent event) const noexcept {
  return valid_event(event) ? events_[event_index(event)] : nullptr;
}

Sm87BulkV2P40Owner::Sm87BulkV2P40Owner(
    Sm87BulkV2P40CudaRuntime* const cuda) noexcept
    : cuda_(cuda), owner_identity_(next_identity()) {
  event_producers_.fill(Sm87BulkV2P40Stream::kCount);
}

Sm87BulkV2P40Owner::~Sm87BulkV2P40Owner() { release_resources(); }

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::initialize_resources() noexcept {
  if (cuda_ == nullptr || state_ != Sm87BulkV2P40OwnerState::kEmpty ||
      !sm87_bulk_v2_p40_execution_plan_valid(
          kSm87BulkV2P40FrozenExecutionPlan)) {
    return error(Sm87BulkV2P40OwnerError::kInvalidPlan,
                 "v2_p40_frozen_execution_plan");
  }

  int cuda_status = cuda_->get_current_device(&device_ordinal_);
  if (cuda_status != 0) {
    return error(Sm87BulkV2P40OwnerError::kDeviceQuery,
                 "get_current_device", cuda_status);
  }
  Sm87BulkV2P40DeviceProperties properties{};
  cuda_status = cuda_->get_device_properties(device_ordinal_, &properties);
  if (cuda_status != 0) {
    return error(Sm87BulkV2P40OwnerError::kDeviceQuery,
                 "get_device_properties", cuda_status);
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiprocessor_count != 16) {
    return error(Sm87BulkV2P40OwnerError::kWrongDevice,
                 "exact_sm87_16sm_required");
  }

  for (std::size_t index = 0U; index < streams_.size(); ++index) {
    cuda_status = cuda_->create_nonblocking_stream(&streams_[index]);
    if (cuda_status != 0 || streams_[index] == nullptr) {
      return error(Sm87BulkV2P40OwnerError::kStreamCreate,
                   "create_nonblocking_stream", cuda_status, index);
    }
    unsigned int flags = 0U;
    cuda_status = cuda_->get_stream_flags(streams_[index], &flags);
    if (cuda_status != 0 ||
        (flags & kSm87BulkV2P40NonBlockingStreamFlag) == 0U) {
      return error(Sm87BulkV2P40OwnerError::kStreamValidation,
                   "validate_nonblocking_stream", cuda_status, index);
    }
  }

  for (std::size_t index = 0U; index < events_.size(); ++index) {
    cuda_status = cuda_->create_disable_timing_event(&events_[index]);
    if (cuda_status != 0 || events_[index] == nullptr) {
      return error(Sm87BulkV2P40OwnerError::kEventCreate,
                   "create_disable_timing_event", cuda_status, index);
    }
  }

  cuda_status = cuda_->allocate_device(
      &device_control_arena_,
      static_cast<std::size_t>(kSm87BulkV2P40ControlArenaBytes));
  if (cuda_status != 0 || device_control_arena_ == nullptr) {
    return error(Sm87BulkV2P40OwnerError::kDeviceControlAllocation,
                 "allocate_device_control_arena", cuda_status);
  }
  Sm87BulkV2P40PointerAttributes device_attributes{};
  cuda_status =
      cuda_->query_pointer(device_control_arena_, &device_attributes);
  if (cuda_status != 0 ||
      device_attributes.kind != Sm87BulkV2P40PointerKind::kDevice ||
      device_attributes.device_pointer != device_control_arena_ ||
      device_attributes.device_ordinal != device_ordinal_) {
    return error(Sm87BulkV2P40OwnerError::kDeviceControlValidation,
                 "validate_device_control_arena", cuda_status);
  }

  void* cancellation_host = nullptr;
  cuda_status = cuda_->allocate_mapped_host(
      &cancellation_host,
      static_cast<std::size_t>(kSm87BulkV2P40MappedCancellationBytes));
  if (cuda_status != 0 || cancellation_host == nullptr) {
    return error(Sm87BulkV2P40OwnerError::kCancellationAllocation,
                 "allocate_exact_mapped_cancellation_word", cuda_status);
  }
  cancellation_host_word_ = static_cast<std::uint32_t*>(cancellation_host);
  void* cancellation_alias = nullptr;
  cuda_status =
      cuda_->mapped_device_alias(&cancellation_alias, cancellation_host_word_);
  if (cuda_status != 0 || cancellation_alias == nullptr) {
    return error(Sm87BulkV2P40OwnerError::kCancellationMapping,
                 "map_exact_cancellation_alias", cuda_status);
  }
  cancellation_device_alias_ =
      static_cast<const std::uint32_t*>(cancellation_alias);

  Sm87BulkV2P40PointerAttributes host_attributes{};
  Sm87BulkV2P40PointerAttributes alias_attributes{};
  const int host_query =
      cuda_->query_pointer(cancellation_host_word_, &host_attributes);
  const int alias_query =
      cuda_->query_pointer(cancellation_device_alias_, &alias_attributes);
  if (host_query != 0 || alias_query != 0 ||
      host_attributes.kind != Sm87BulkV2P40PointerKind::kHost ||
      alias_attributes.kind != Sm87BulkV2P40PointerKind::kHost ||
      host_attributes.host_pointer != cancellation_host_word_ ||
      host_attributes.device_pointer != cancellation_device_alias_ ||
      alias_attributes.host_pointer != cancellation_host_word_ ||
      alias_attributes.device_pointer != cancellation_device_alias_) {
    return error(Sm87BulkV2P40OwnerError::kCancellationValidation,
                 "validate_exact_mapped_cancellation_pair",
                 host_query != 0 ? host_query : alias_query);
  }

  publish_cancellation(0U);
  cuda_status = cuda_->memset_async(
      device_control_arena_, 0,
      static_cast<std::size_t>(kSm87BulkV2P40ControlArenaBytes), streams_[0U]);
  if (cuda_status == 0) {
    cuda_status = cuda_->synchronize_stream(streams_[0U]);
  }
  if (cuda_status != 0) {
    return error(Sm87BulkV2P40OwnerError::kCudaSubmission,
                 "initialize_device_control_arena", cuda_status);
  }

  state_ = Sm87BulkV2P40OwnerState::kResourcesReady;
  return ok();
}

Sm87BulkV2P40OwnerStatus
Sm87BulkV2P40Owner::seal_for_default_off_development_execution(
    const Sm87BulkV2P40ConstituentSealAccess& seal) noexcept {
  return install_execution_access(
      seal,
      Sm87BulkV2P40ExecutionClass::kDefaultOffDevelopmentCandidate);
}

Sm87BulkV2P40OwnerStatus
Sm87BulkV2P40Owner::seal_for_default_off_direction_witness(
    const Sm87BulkV2P40ConstituentSealAccess& seal) noexcept {
  return install_execution_access(
      seal, Sm87BulkV2P40ExecutionClass::kDefaultOffDirectionWitness);
}

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_OWNER_HOST_FIXTURE)
Sm87BulkV2P40OwnerStatus
Sm87BulkV2P40Owner::seal_synthetic_for_host_contract(
    const Sm87BulkV2P40ConstituentSealAccess& seal) noexcept {
  return install_execution_access(
      seal, Sm87BulkV2P40ExecutionClass::kSyntheticHostContract);
}
#endif

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::install_execution_access(
    const Sm87BulkV2P40ConstituentSealAccess& seal,
    const Sm87BulkV2P40ExecutionClass required_execution_class) noexcept {
  if (state_ != Sm87BulkV2P40OwnerState::kResourcesReady ||
      execution_access_ != nullptr) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "seal_requires_resources_ready");
  }
  if (seal.bound_owner_identity_ == 0U) {
    return error(Sm87BulkV2P40OwnerError::kMissingConstituentSeal,
                 "all_real_constituent_seals_required");
  }
  if (seal.bound_owner_identity_ != owner_identity_ ||
      seal.identity_.owner_identity != owner_identity_) {
    return error(Sm87BulkV2P40OwnerError::kForeignConstituentSeal,
                 "constituent_seal_owner_mismatch");
  }
  const bool real_development_seal =
      seal.real_fp8_binding_seal && seal.real_attention_binding_seal &&
      seal.real_bf16_ab_binding_seal && seal.real_gdn_session_seal &&
      seal.real_nvfp4_binding_seal && seal.real_request_arena_seal &&
      seal.real_pinned_handoff_seal &&
      seal.all_static_resource_checks_complete &&
      seal.authenticated_real_constituents &&
      seal.exact_numerical_contract_qualified &&
      !seal.default_off_direction_witness_eligible &&
      seal.default_off_candidate_eligible &&
      !seal.production_dispatch_eligible &&
      !seal.synthetic_host_contract_only &&
      seal.identity_.development_candidate_valid();
  const bool direction_witness_seal =
      seal.real_fp8_binding_seal && seal.real_attention_binding_seal &&
      seal.real_bf16_ab_binding_seal && seal.real_gdn_session_seal &&
      seal.real_nvfp4_binding_seal && seal.real_request_arena_seal &&
      seal.real_pinned_handoff_seal &&
      seal.all_static_resource_checks_complete &&
      seal.authenticated_real_constituents &&
      !seal.exact_numerical_contract_qualified &&
      seal.default_off_direction_witness_eligible &&
      !seal.default_off_candidate_eligible &&
      !seal.production_dispatch_eligible &&
      !seal.synthetic_host_contract_only &&
      seal.identity_.direction_witness_valid();
  const bool synthetic_host_seal =
      !seal.real_fp8_binding_seal && !seal.real_attention_binding_seal &&
      !seal.real_bf16_ab_binding_seal && !seal.real_gdn_session_seal &&
      !seal.real_nvfp4_binding_seal && !seal.real_request_arena_seal &&
      !seal.real_pinned_handoff_seal &&
      seal.all_static_resource_checks_complete &&
      !seal.authenticated_real_constituents &&
      !seal.exact_numerical_contract_qualified &&
      !seal.default_off_direction_witness_eligible &&
      !seal.default_off_candidate_eligible &&
      !seal.production_dispatch_eligible &&
      seal.synthetic_host_contract_only &&
      seal.identity_.synthetic_host_contract_valid();
  bool required_seal_valid = false;
  switch (required_execution_class) {
    case Sm87BulkV2P40ExecutionClass::kDefaultOffDirectionWitness:
      required_seal_valid = direction_witness_seal;
      break;
    case Sm87BulkV2P40ExecutionClass::kDefaultOffDevelopmentCandidate:
      required_seal_valid = real_development_seal;
      break;
    case Sm87BulkV2P40ExecutionClass::kSyntheticHostContract:
      required_seal_valid = synthetic_host_seal;
      break;
    case Sm87BulkV2P40ExecutionClass::kInvalid:
      break;
  }
  if (!seal.identity_.valid() ||
      seal.identity_.device_ordinal != device_ordinal_ ||
      seal.streams_ != streams_ || seal.events_ != events_ ||
      seal.device_control_arena_ != device_control_arena_ ||
      seal.cancellation_host_word_ != cancellation_host_word_ ||
      seal.cancellation_device_alias_ != cancellation_device_alias_ ||
      !required_seal_valid) {
    return error(Sm87BulkV2P40OwnerError::kInvalidConstituentSeal,
                 "constituent_seal_incomplete_or_unbound");
  }

  std::unique_ptr<Sm87BulkV2P40ExecutionAccess> access(
      new (std::nothrow) Sm87BulkV2P40ExecutionAccess(
          this, seal.identity_, streams_, events_, device_control_arena_,
          cancellation_device_alias_));
  if (access == nullptr) {
    return error(Sm87BulkV2P40OwnerError::kInvalidConstituentSeal,
                 "execution_access_allocation");
  }
  execution_access_ = std::move(access);
  state_ = Sm87BulkV2P40OwnerState::kSealed;
  return ok();
}

bool Sm87BulkV2P40Owner::access_matches(
    const Sm87BulkV2P40ExecutionAccess& access) const noexcept {
  return execution_access_ != nullptr && execution_access_.get() == &access &&
         access.owner_ == this && identity_equal(access.identity_,
                                                  execution_access_->identity_) &&
         access.streams_ == streams_ && access.events_ == events_ &&
         access.device_control_arena_ == device_control_arena_ &&
         access.cancellation_device_alias_ == cancellation_device_alias_;
}

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::begin_request(
    const Sm87BulkV2P40ExecutionAccess& access,
    Sm87BulkV2P40RequestState& request_state,
    const Sm87BulkV2P40RequestStateSealedAccess& request_access,
    const std::uint64_t request_epoch) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "begin_request_owner_bound_access");
  }
  if ((state_ != Sm87BulkV2P40OwnerState::kSealed &&
       state_ != Sm87BulkV2P40OwnerState::kCompleted &&
       state_ != Sm87BulkV2P40OwnerState::kCancelled) ||
      active_request_state_ != nullptr || active_request_access_ != nullptr) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "begin_request_requires_quiesced_unbound_owner");
  }
  if (request_epoch == 0U || request_epoch <= last_request_epoch_) {
    return error(Sm87BulkV2P40OwnerError::kInvalidRequestEpoch,
                 "begin_request_requires_fresh_nonzero_epoch");
  }

  bool request_streams_match = true;
  for (std::size_t index = 0U; index < kSm87BulkV2P40StreamCount; ++index) {
    if (request_access.cuda_stream(
            static_cast<Sm87BulkV2P40Stream>(index)) != streams_[index]) {
      request_streams_match = false;
      break;
    }
  }
  if (!request_streams_match ||
      !request_state.owner_begin_binding_valid(
          request_access, owner_identity_,
          access.identity_.request_allocation_identity,
          access.identity_.stream_event_owner_identity, device_ordinal_)) {
    return error(Sm87BulkV2P40OwnerError::kForeignRequestState,
                 "begin_requires_exact_owner_allocation_and_stream_binding");
  }
  const Sm87BulkV2P40RequestStateStatus state_begin =
      request_state.begin_request(request_access, request_epoch);
  if (!state_begin) {
    return error(Sm87BulkV2P40OwnerError::kInvalidRequestEpoch,
                 "request_state_rejected_owner_epoch",
                 state_begin.cuda_error);
  }
  active_request_state_ = &request_state;
  active_request_access_ = &request_access;

  receipt_ = make_active_receipt(access.identity_, request_epoch);
  event_generations_.fill(0U);
  event_producers_.fill(Sm87BulkV2P40Stream::kCount);
  stream_submission_generations_.fill(0U);
  event_stream_generations_.fill(0U);
  main_joined_stream_generations_.fill(0U);
  publish_cancellation(0U);
  const int cuda_status = cuda_->memset_async(
      device_control_arena_, 0,
      static_cast<std::size_t>(kSm87BulkV2P40ControlArenaBytes), streams_[0U]);
  receipt_.aggregate.submission_started = true;
  if (cuda_status != 0) {
    state_ = Sm87BulkV2P40OwnerState::kActive;
    return cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned, cuda_status);
  }
  ++receipt_.device_ordering_operations;
  last_request_epoch_ = request_epoch;
  state_ = Sm87BulkV2P40OwnerState::kActive;
  return ok();
}

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_OWNER_HOST_FIXTURE)
Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::begin_request(
    const Sm87BulkV2P40ExecutionAccess& access,
    const std::uint64_t request_epoch) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "synthetic_begin_request_owner_bound_access");
  }
  if ((state_ != Sm87BulkV2P40OwnerState::kSealed &&
       state_ != Sm87BulkV2P40OwnerState::kCompleted &&
       state_ != Sm87BulkV2P40OwnerState::kCancelled) ||
      active_request_state_ != nullptr || active_request_access_ != nullptr) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "synthetic_begin_requires_quiesced_unbound_owner");
  }
  if (request_epoch == 0U || request_epoch <= last_request_epoch_) {
    return error(Sm87BulkV2P40OwnerError::kInvalidRequestEpoch,
                 "synthetic_begin_requires_fresh_nonzero_epoch");
  }

  receipt_ = make_active_receipt(access.identity_, request_epoch);
  event_generations_.fill(0U);
  event_producers_.fill(Sm87BulkV2P40Stream::kCount);
  stream_submission_generations_.fill(0U);
  event_stream_generations_.fill(0U);
  main_joined_stream_generations_.fill(0U);
  publish_cancellation(0U);
  const int cuda_status = cuda_->memset_async(
      device_control_arena_, 0,
      static_cast<std::size_t>(kSm87BulkV2P40ControlArenaBytes), streams_[0U]);
  receipt_.aggregate.submission_started = true;
  if (cuda_status != 0) {
    state_ = Sm87BulkV2P40OwnerState::kActive;
    return cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned, cuda_status);
  }
  ++receipt_.device_ordering_operations;
  last_request_epoch_ = request_epoch;
  state_ = Sm87BulkV2P40OwnerState::kActive;
  return ok();
}
#endif

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::record_event(
    const Sm87BulkV2P40ExecutionAccess& access,
    const Sm87BulkV2P40Stream producer,
    const Sm87BulkV2P40ReusableEvent event) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "record_event_owner_bound_access");
  }
  if (state_ != Sm87BulkV2P40OwnerState::kActive || !valid_stream(producer) ||
      !valid_event(event) ||
      event == Sm87BulkV2P40ReusableEvent::kRequestTerminal) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "record_event_invalid_active_ordering_edge");
  }
  const int cuda_status = cuda_->record_event(
      events_[event_index(event)], streams_[stream_index(producer)]);
  if (cuda_status != 0) {
    return cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned, cuda_status);
  }
  ++event_generations_[event_index(event)];
  event_producers_[event_index(event)] = producer;
  event_stream_generations_[event_index(event)] =
      stream_submission_generations_[stream_index(producer)];
  ++receipt_.device_ordering_operations;
  return ok();
}

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::wait_event(
    const Sm87BulkV2P40ExecutionAccess& access,
    const Sm87BulkV2P40Stream consumer,
    const Sm87BulkV2P40ReusableEvent event) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "wait_event_owner_bound_access");
  }
  if (state_ != Sm87BulkV2P40OwnerState::kActive || !valid_stream(consumer) ||
      !valid_event(event)) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "wait_event_invalid_active_ordering_edge");
  }
  const std::size_t index = event_index(event);
  if (event_generations_[index] == 0U ||
      !valid_stream(event_producers_[index])) {
    return error(Sm87BulkV2P40OwnerError::kEventNotRecorded,
                 "wait_requires_recorded_event", 0, index);
  }
  const int cuda_status = cuda_->stream_wait_event(
      streams_[stream_index(consumer)], events_[index]);
  if (cuda_status != 0) {
    return cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned, cuda_status);
  }
  if (consumer == Sm87BulkV2P40Stream::kMain &&
      event_producers_[index] != Sm87BulkV2P40Stream::kMain) {
    const std::size_t producer_index =
        stream_index(event_producers_[index]);
    if (event_stream_generations_[index] >
        main_joined_stream_generations_[producer_index]) {
      main_joined_stream_generations_[producer_index] =
          event_stream_generations_[index];
    }
    if (main_joined_stream_generations_[producer_index] ==
            stream_submission_generations_[producer_index] &&
        stream_submission_generations_[producer_index] != 0U) {
      receipt_.joined_auxiliary_stream_mask |=
          1U << static_cast<std::uint32_t>(event_producers_[index]);
    }
  }
  ++receipt_.device_ordering_operations;
  return ok();
}

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::note_submission(
    const Sm87BulkV2P40ExecutionAccess& access,
    const Sm87BulkV2P40Stream producer,
    const Sm87BulkV2P40SubmissionCounter counter, const std::size_t count,
    const std::size_t layer, const Sm87BulkV2P40FamilyPhase family,
    const std::size_t segment, const std::size_t constituent) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "note_submission_owner_bound_access");
  }
  if (state_ != Sm87BulkV2P40OwnerState::kActive || !valid_stream(producer) ||
      count == 0U ||
      layer >= kSm87BulkV2P40Layers ||
      family == Sm87BulkV2P40FamilyPhase::kCount) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "note_submission_invalid_active_work");
  }

  const bool final_counter = final_submission_counter(counter);
  if (final_counter) {
    if (receipt_.aggregate.completed_layers != kSm87BulkV2P40Layers ||
        layer != kSm87BulkV2P40Layers - 1U ||
        family != Sm87BulkV2P40FamilyPhase::kFinalHandoff || count != 1U ||
        segment != 0U) {
      return error(Sm87BulkV2P40OwnerError::kInvalidLayerOrder,
                   "final_work_requires_closed_layers_and_unit_submission");
    }
    const auto& final_receipt = receipt_.aggregate;
    bool final_order_valid = false;
    switch (counter) {
      case Sm87BulkV2P40SubmissionCounter::kFinalNorm:
        final_order_valid =
            constituent == 0U && final_receipt.enqueued_final_norm == 0U &&
            final_receipt.enqueued_lm_head == 0U &&
            final_receipt.enqueued_argmax == 0U &&
            final_receipt.enqueued_handoff_d2h == 0U;
        break;
      case Sm87BulkV2P40SubmissionCounter::kLmHead:
        final_order_valid =
            constituent == 1U && final_receipt.enqueued_final_norm == 1U &&
            final_receipt.enqueued_lm_head == 0U &&
            final_receipt.enqueued_argmax == 0U &&
            final_receipt.enqueued_handoff_d2h == 0U;
        break;
      case Sm87BulkV2P40SubmissionCounter::kArgmax:
        final_order_valid =
            constituent == 2U && final_receipt.enqueued_final_norm == 1U &&
            final_receipt.enqueued_lm_head == 1U &&
            final_receipt.enqueued_argmax == 0U &&
            final_receipt.enqueued_handoff_d2h == 0U;
        break;
      case Sm87BulkV2P40SubmissionCounter::kHandoffD2h:
        return error(
            Sm87BulkV2P40OwnerError::kMissingOwnerBoundHandoff,
            "handoff_counter_requires_owner_bound_fixed_d2h_transaction");
      default:
        break;
    }
    if (!final_order_valid) {
      return error(Sm87BulkV2P40OwnerError::kInvalidFinalOrder,
                   "final_submissions_require_norm_lm_head_argmax_order");
    }
  } else {
    if (receipt_.aggregate.completed_layers >= kSm87BulkV2P40Layers ||
        layer != receipt_.aggregate.completed_layers) {
      return error(Sm87BulkV2P40OwnerError::kInvalidLayerOrder,
                   "constituent_work_requires_current_natural_layer");
    }
    const bool full_layer = sm87_bulk_v2_p40_is_full_layer(layer);
    if ((full_layer && gdn_only_submission_counter(counter)) ||
        (!full_layer && full_only_submission_counter(counter))) {
      return error(Sm87BulkV2P40OwnerError::kWrongLayerKind,
                   "constituent_counter_disagrees_with_frozen_layer_kind");
    }
  }

  std::size_t* destination = nullptr;
  std::size_t* aggregate_destination = nullptr;
  switch (counter) {
    case Sm87BulkV2P40SubmissionCounter::kFp8GdnInputWholeRoleLaunch:
      destination = &receipt_.aggregate.projection_successor
                         .fp8_gdn_input_whole_launches;
      aggregate_destination = &receipt_.aggregate.projection_successor
                                   .fp8_whole_role_launches;
      break;
    case Sm87BulkV2P40SubmissionCounter::kFp8FullInputWholeRoleLaunch:
      destination = &receipt_.aggregate.projection_successor
                         .fp8_full_input_whole_launches;
      aggregate_destination = &receipt_.aggregate.projection_successor
                                   .fp8_whole_role_launches;
      break;
    case Sm87BulkV2P40SubmissionCounter::kFp8OutputWholeRoleLaunch:
      destination = &receipt_.aggregate.projection_successor
                         .fp8_output_whole_launches;
      aggregate_destination = &receipt_.aggregate.projection_successor
                                   .fp8_whole_role_launches;
      break;
    case Sm87BulkV2P40SubmissionCounter::kNvFp4GateUpWholeRoleLaunch:
      destination = &receipt_.aggregate.projection_successor
                         .nvfp4_gate_up_whole_launches;
      aggregate_destination = &receipt_.aggregate.projection_successor
                                   .nvfp4_whole_role_launches;
      break;
    case Sm87BulkV2P40SubmissionCounter::kNvFp4DownWholeRoleLaunch:
      destination = &receipt_.aggregate.projection_successor
                         .nvfp4_down_whole_launches;
      aggregate_destination = &receipt_.aggregate.projection_successor
                                   .nvfp4_whole_role_launches;
      break;
    case Sm87BulkV2P40SubmissionCounter::kAttentionLaunch:
      destination = &receipt_.aggregate.enqueued_attention_launches;
      break;
    case Sm87BulkV2P40SubmissionCounter::kAttentionPreprocessPanel:
      destination =
          &receipt_.aggregate.enqueued_attention_preprocess_panels;
      break;
    case Sm87BulkV2P40SubmissionCounter::kBf16AbLaunch:
      destination = &receipt_.aggregate.enqueued_bf16_ab_launches;
      aggregate_destination = &receipt_.aggregate.projection_successor
                                   .bf16_ab_physical_launches;
      break;
    case Sm87BulkV2P40SubmissionCounter::kGdnProducerChunk:
      destination = &receipt_.aggregate.enqueued_gdn_producer_chunks;
      break;
    case Sm87BulkV2P40SubmissionCounter::kGdnRecurrenceChunk:
      destination = &receipt_.aggregate.enqueued_gdn_recurrence_chunks;
      break;
    case Sm87BulkV2P40SubmissionCounter::kGdnEpilogueChunk:
      destination = &receipt_.aggregate.enqueued_gdn_epilogue_chunks;
      break;
    case Sm87BulkV2P40SubmissionCounter::kGdnPersistentCopy:
      destination = &receipt_.aggregate.enqueued_gdn_persistent_copies;
      break;
    case Sm87BulkV2P40SubmissionCounter::kFinalNorm:
      destination = &receipt_.aggregate.enqueued_final_norm;
      break;
    case Sm87BulkV2P40SubmissionCounter::kLmHead:
      destination = &receipt_.aggregate.enqueued_lm_head;
      break;
    case Sm87BulkV2P40SubmissionCounter::kArgmax:
      destination = &receipt_.aggregate.enqueued_argmax;
      break;
    case Sm87BulkV2P40SubmissionCounter::kHandoffD2h:
      destination = &receipt_.aggregate.enqueued_handoff_d2h;
      break;
  }
  if (!checked_increment(destination, count) ||
      (aggregate_destination != nullptr &&
       !checked_increment(aggregate_destination, count))) {
    return cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned,
        static_cast<int>(cudaErrorInvalidValue));
  }
  receipt_.aggregate.submission_started = true;
  receipt_.aggregate.last_submitted_layer = layer;
  receipt_.aggregate.last_submitted_family = family;
  receipt_.aggregate.last_submitted_segment = segment;
  receipt_.aggregate.last_submitted_constituent = constituent;
  if (!checked_increment(
          &stream_submission_generations_[stream_index(producer)], count)) {
    return cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned,
        static_cast<int>(cudaErrorInvalidValue));
  }
  if (producer != Sm87BulkV2P40Stream::kMain) {
    receipt_.joined_auxiliary_stream_mask &=
        ~(1U << static_cast<std::uint32_t>(producer));
  }
  return ok();
}

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::close_layer(
    const Sm87BulkV2P40ExecutionAccess& access,
    const std::size_t model_layer,
    const Sm87BulkV2P40LayerKind expected_kind) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "close_layer_owner_bound_access");
  }
  if (state_ != Sm87BulkV2P40OwnerState::kActive ||
      model_layer >= kSm87BulkV2P40Layers ||
      model_layer != receipt_.aggregate.completed_layers) {
    return error(Sm87BulkV2P40OwnerError::kInvalidLayerOrder,
                 "close_requires_exact_next_natural_model_layer");
  }

  const bool is_full = sm87_bulk_v2_p40_is_full_layer(model_layer);
  const Sm87BulkV2P40LayerKind authoritative_kind =
      is_full ? Sm87BulkV2P40LayerKind::kFull
              : Sm87BulkV2P40LayerKind::kGdn;
  if (expected_kind != authoritative_kind) {
    return error(Sm87BulkV2P40OwnerError::kWrongLayerKind,
                 "caller_kind_disagrees_with_frozen_natural_layer_map");
  }

  auto& receipt = receipt_.aggregate;
  const Sm87BulkV2P40LayerPrefix prefix = layer_prefix(model_layer);
  const std::size_t previous_gdn = prefix.gdn_layers - (is_full ? 0U : 1U);
  const std::size_t previous_full = prefix.full_layers - (is_full ? 1U : 0U);
  const std::size_t expected_previous_logical =
      previous_gdn * kGdnLogicalProjectionRolesPerLayer +
      previous_full * kFullLogicalProjectionRolesPerLayer;
  const std::size_t expected_previous_fused =
      previous_gdn * kGdnFusedOuterOperationsPerLayer +
      previous_full * kFullFusedOuterOperationsPerLayer;
  const std::uint64_t expected_previous_conventional =
      previous_gdn * kGdnConventionalOperationsPerLayer +
      previous_full * kFullConventionalOperationsPerLayer;
  if (receipt.completed_gdn_layers != previous_gdn ||
      receipt.completed_full_layers != previous_full ||
      receipt.closed_layer_residuals != model_layer ||
      receipt.closed_gdn_state_publications != previous_gdn ||
      receipt.logical_projection_roles != expected_previous_logical ||
      receipt.fused_outer_operations != expected_previous_fused ||
      receipt.projection_conventional_operations !=
          expected_previous_conventional ||
      !layer_work_exact(receipt, prefix)) {
    return error(Sm87BulkV2P40OwnerError::kIncompleteLayerWork,
                 "layer_close_requires_exact_constituent_prefix_receipt");
  }

  // Publish the whole layer receipt atomically from owner-derived prefix
  // values.  No increment depends on caller data and no partial mutation can
  // occur after the exact-prefix proof above.
  receipt.completed_layers = prefix.layers;
  receipt.completed_gdn_layers = prefix.gdn_layers;
  receipt.completed_full_layers = prefix.full_layers;
  receipt.closed_layer_residuals = prefix.layers;
  receipt.closed_gdn_state_publications = prefix.gdn_layers;
  receipt.logical_projection_roles =
      prefix.gdn_layers * kGdnLogicalProjectionRolesPerLayer +
      prefix.full_layers * kFullLogicalProjectionRolesPerLayer;
  receipt.fused_outer_operations =
      prefix.gdn_layers * kGdnFusedOuterOperationsPerLayer +
      prefix.full_layers * kFullFusedOuterOperationsPerLayer;
  receipt.projection_conventional_operations =
      prefix.gdn_layers * kGdnConventionalOperationsPerLayer +
      prefix.full_layers * kFullConventionalOperationsPerLayer;
  return ok();
}

Sm87BulkV2P40OwnerStatus
Sm87BulkV2P40Owner::enqueue_owner_bound_handoff_d2h(
    const Sm87BulkV2P40ExecutionAccess& access) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "handoff_d2h_owner_bound_execution_access");
  }
  if (state_ != Sm87BulkV2P40OwnerState::kActive ||
      active_request_state_ == nullptr || active_request_access_ == nullptr) {
    return error(Sm87BulkV2P40OwnerError::kMissingOwnerBoundRequestState,
                 "handoff_d2h_requires_active_owner_bound_request_state");
  }
  const auto& receipt = receipt_.aggregate;
  if (receipt.completed_layers != kSm87BulkV2P40Layers ||
      receipt.completed_gdn_layers != kSm87BulkV2P40GdnLayers ||
      receipt.completed_full_layers != kSm87BulkV2P40FullLayers ||
      receipt.enqueued_final_norm != 1U ||
      receipt.enqueued_lm_head != 1U || receipt.enqueued_argmax != 1U ||
      receipt.enqueued_handoff_d2h != 0U ||
      receipt.last_submitted_layer != kSm87BulkV2P40Layers - 1U ||
      receipt.last_submitted_family !=
          Sm87BulkV2P40FamilyPhase::kFinalHandoff ||
      receipt.last_submitted_segment != 0U ||
      receipt.last_submitted_constituent != 2U) {
    return error(
        Sm87BulkV2P40OwnerError::kInvalidFinalOrder,
        "fixed_d2h_requires_closed_layers_and_norm_lm_head_argmax_prefix");
  }
  if (!active_request_state_->owner_completion_binding_valid(
          *active_request_access_, owner_identity_, receipt.request_epoch,
          access.identity_.request_allocation_identity,
          access.identity_.stream_event_owner_identity, device_ordinal_)) {
    return error(Sm87BulkV2P40OwnerError::kForeignRequestState,
                 "handoff_d2h_active_request_binding_drift");
  }

  const Sm87BulkV2P40RequestStateStatus handoff =
      active_request_state_->enqueue_handoff_d2h(*active_request_access_);
  if (!handoff) {
    const int first_error =
        handoff.cuda_error != 0 ? handoff.cuda_error
                                : static_cast<int>(cudaErrorInvalidValue);
    (void)cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned, first_error);
    return error(Sm87BulkV2P40OwnerError::kCudaSubmission,
                 "owner_bound_fixed_8_byte_handoff_submission",
                 first_error);
  }

  auto& mutable_receipt = receipt_.aggregate;
  mutable_receipt.enqueued_handoff_d2h = 1U;
  mutable_receipt.submission_started = true;
  mutable_receipt.last_submitted_layer = kSm87BulkV2P40Layers - 1U;
  mutable_receipt.last_submitted_family =
      Sm87BulkV2P40FamilyPhase::kFinalHandoff;
  mutable_receipt.last_submitted_segment = 0U;
  mutable_receipt.last_submitted_constituent = 3U;
  if (!checked_increment(
          &stream_submission_generations_[stream_index(
              Sm87BulkV2P40Stream::kMain)],
          1U)) {
    return cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned,
        static_cast<int>(cudaErrorInvalidValue));
  }
  return ok();
}

Sm87BulkV2P40OwnerStatus
Sm87BulkV2P40Owner::poison_after_submission_failure(
    const Sm87BulkV2P40ExecutionAccess& access, const int cuda_error,
    const std::size_t layer, const Sm87BulkV2P40FamilyPhase family,
    const std::size_t segment, const std::size_t constituent) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "poison_owner_bound_access");
  }
  if (state_ != Sm87BulkV2P40OwnerState::kActive || cuda_error == 0 ||
      layer >= kSm87BulkV2P40Layers ||
      family == Sm87BulkV2P40FamilyPhase::kCount) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "poison_requires_partial_active_submission");
  }
  receipt_.aggregate.submission_started = true;
  receipt_.aggregate.last_submitted_layer = layer;
  receipt_.aggregate.last_submitted_family = family;
  receipt_.aggregate.last_submitted_segment = segment;
  receipt_.aggregate.last_submitted_constituent = constituent;
  return cancel_drain_and_transition(
      Sm87BulkV2P40OwnerLifecycle::kPoisoned, cuda_error);
}

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::cancel_request(
    const Sm87BulkV2P40ExecutionAccess& access) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "cancel_owner_bound_access");
  }
  if (state_ != Sm87BulkV2P40OwnerState::kActive) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "cancel_requires_active_request");
  }
  return cancel_drain_and_transition(
      Sm87BulkV2P40OwnerLifecycle::kCancelled, 0);
}

Sm87BulkV2P40OwnerStatus
Sm87BulkV2P40Owner::cancel_drain_and_transition(
    const Sm87BulkV2P40OwnerLifecycle terminal_lifecycle,
    const int first_error) noexcept {
  state_ = Sm87BulkV2P40OwnerState::kDraining;
  publish_cancellation(1U);
  receipt_.aggregate.cancellation_published = true;
  receipt_.aggregate.state_committed = false;
  receipt_.aggregate.handoff_observed = false;
  receipt_.aggregate.first_error = first_error;
  const Sm87BulkV2P40OwnerStatus drain = drain_all_streams();
  Sm87BulkV2P40OwnerLifecycle actual_lifecycle = terminal_lifecycle;
  int request_transition_error = 0;
  if (active_request_state_ != nullptr && active_request_access_ != nullptr) {
    if (active_request_state_->lifecycle() ==
        Sm87BulkV2P40RequestStateLifecycle::kActive) {
      Sm87BulkV2P40RequestStateStatus request_transition;
      if (terminal_lifecycle == Sm87BulkV2P40OwnerLifecycle::kCancelled &&
          static_cast<bool>(drain)) {
        request_transition =
            active_request_state_->mark_cancelled_after_owner_drain(
                *active_request_access_);
      } else {
        const int poison_error =
            first_error != 0
                ? first_error
                : (drain.cuda_error != 0
                       ? drain.cuda_error
                       : static_cast<int>(cudaErrorInvalidResourceHandle));
        request_transition = active_request_state_->poison_after_owner_drain(
            *active_request_access_, poison_error);
      }
      if (!request_transition) {
        request_transition_error =
            request_transition.cuda_error != 0
                ? request_transition.cuda_error
                : static_cast<int>(cudaErrorInvalidResourceHandle);
        actual_lifecycle = Sm87BulkV2P40OwnerLifecycle::kPoisoned;
      }
    } else if (active_request_state_->lifecycle() !=
               Sm87BulkV2P40RequestStateLifecycle::kPoisoned) {
      request_transition_error =
          static_cast<int>(cudaErrorInvalidResourceHandle);
      actual_lifecycle = Sm87BulkV2P40OwnerLifecycle::kPoisoned;
    }
    active_request_state_ = nullptr;
    active_request_access_ = nullptr;
  }
  if (!drain) {
    actual_lifecycle = Sm87BulkV2P40OwnerLifecycle::kPoisoned;
    if (receipt_.aggregate.first_error == 0) {
      receipt_.aggregate.first_error =
          drain.cuda_error != 0
              ? drain.cuda_error
              : static_cast<int>(cudaErrorInvalidResourceHandle);
    }
  }
  receipt_.aggregate.lifecycle = actual_lifecycle;
  if (actual_lifecycle == Sm87BulkV2P40OwnerLifecycle::kPoisoned) {
    state_ = Sm87BulkV2P40OwnerState::kPoisoned;
  } else {
    state_ = Sm87BulkV2P40OwnerState::kCancelled;
  }
  if (request_transition_error != 0) {
    receipt_.aggregate.first_error = request_transition_error;
    return error(Sm87BulkV2P40OwnerError::kForeignRequestState,
                 "owner_drain_request_state_transition_failed",
                 request_transition_error);
  }
  if (first_error != 0) {
    return error(Sm87BulkV2P40OwnerError::kCudaSubmission,
                 "partial_submission_cancelled_drained_poisoned",
                 first_error);
  }
  return drain;
}

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::drain_all_streams() noexcept {
  int first_cuda_error = 0;
  std::size_t first_error_index = std::numeric_limits<std::size_t>::max();
  for (std::size_t index = 0U; index < streams_.size(); ++index) {
    if (streams_[index] == nullptr) {
      continue;
    }
    const int cuda_status = cuda_->synchronize_stream(streams_[index]);
    if (first_cuda_error == 0 && cuda_status != 0) {
      first_cuda_error = cuda_status;
      first_error_index = index;
    }
  }
  receipt_.aggregate.terminal_host_drains = 1U;
  receipt_.aggregate.all_streams_drained = first_cuda_error == 0;
  if (first_cuda_error != 0) {
    return error(Sm87BulkV2P40OwnerError::kDrainFailure,
                 "drain_all_five_streams", first_cuda_error,
                 first_error_index);
  }
  return ok();
}

bool Sm87BulkV2P40Owner::work_receipt_complete() const noexcept {
  const auto& receipt = receipt_.aggregate;
  return receipt.completed_layers == kSm87BulkV2P40Layers &&
         receipt.completed_gdn_layers == kSm87BulkV2P40GdnLayers &&
         receipt.completed_full_layers == kSm87BulkV2P40FullLayers &&
         receipt.closed_layer_residuals == kSm87BulkV2P40Layers &&
         receipt.closed_gdn_state_publications == kSm87BulkV2P40GdnLayers &&
         receipt.logical_projection_roles ==
             kSm87BulkV2P40LogicalProjectionRoles &&
         receipt.fused_outer_operations ==
             kSm87BulkV2P40FusedOuterOperations &&
         receipt.projection_conventional_operations ==
             kSm87BulkV2P40ProjectionConventionalOperations &&
         sm87_bulk_v2_p40_projection_successor_receipt_complete(
             receipt.projection_successor) &&
         receipt.enqueued_attention_launches ==
             kSm87BulkV2P40FullLayers *
                 q3x::kernels::kSm87BulkV2AttentionKernelLaunches &&
         receipt.enqueued_attention_preprocess_panels == 80U &&
         receipt.enqueued_bf16_ab_launches == 48U &&
         receipt.enqueued_gdn_producer_chunks == 30'000U &&
         receipt.enqueued_gdn_recurrence_chunks == 30'000U &&
         receipt.enqueued_gdn_epilogue_chunks == 30'000U &&
         receipt.enqueued_gdn_persistent_copies == 96U &&
         receipt.enqueued_final_norm == 1U && receipt.enqueued_lm_head == 1U &&
         receipt.enqueued_argmax == 1U &&
         receipt.enqueued_handoff_d2h == 1U &&
         receipt.last_submitted_layer == kSm87BulkV2P40Layers - 1U &&
         receipt.last_submitted_family ==
             Sm87BulkV2P40FamilyPhase::kFinalHandoff &&
         receipt.last_submitted_segment == 0U &&
         receipt.last_submitted_constituent == 3U;
}

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::complete_request(
    const Sm87BulkV2P40ExecutionAccess& access) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "complete_owner_bound_access");
  }
  if (state_ != Sm87BulkV2P40OwnerState::kActive) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "complete_requires_active_request");
  }
  // Real completion may read only owner-bound pinned handoff storage after a
  // terminal GPU wait.  This skeleton owns no such storage and therefore has
  // no successful real completion path.  In particular, it accepts no
  // caller-supplied token/nonfinite values as completion authority.
  (void)cancel_drain_and_transition(
      Sm87BulkV2P40OwnerLifecycle::kPoisoned,
      static_cast<int>(cudaErrorInvalidResourceHandle));
  return error(Sm87BulkV2P40OwnerError::kMissingOwnerBoundHandoff,
               "owner_bound_pinned_handoff_capability_required");
}

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40Owner::complete_request(
    const Sm87BulkV2P40ExecutionAccess& access,
    Sm87BulkV2P40RequestState& request_state,
    const Sm87BulkV2P40RequestStateSealedAccess& request_access) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "complete_owner_bound_access_and_request_state");
  }
  if (state_ != Sm87BulkV2P40OwnerState::kActive) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "owner_bound_completion_requires_active_request");
  }

  const auto poison_bound_request =
      [&](const Sm87BulkV2P40OwnerError code, const char* const context,
          const int first_error,
          const std::size_t resource_index =
              std::numeric_limits<std::size_t>::max()) noexcept {
        (void)cancel_drain_and_transition(
            Sm87BulkV2P40OwnerLifecycle::kPoisoned, first_error);
        return error(code, context, first_error, resource_index);
      };

  const auto& identity = access.identity_;
  bool request_streams_match = true;
  for (std::size_t index = 0U; index < kSm87BulkV2P40StreamCount; ++index) {
    if (request_access.cuda_stream(
            static_cast<Sm87BulkV2P40Stream>(index)) != streams_[index]) {
      request_streams_match = false;
      break;
    }
  }
  if (active_request_state_ != &request_state ||
      active_request_access_ != &request_access || !request_streams_match ||
      !request_state.owner_completion_binding_valid(
          request_access, owner_identity_, receipt_.aggregate.request_epoch,
          identity.request_allocation_identity,
          identity.stream_event_owner_identity, device_ordinal_)) {
    return poison_bound_request(
        Sm87BulkV2P40OwnerError::kForeignRequestState,
        "completion_requires_exact_owner_allocation_epoch_access_binding",
        static_cast<int>(cudaErrorInvalidResourceHandle));
  }
  if (receipt_.joined_auxiliary_stream_mask != auxiliary_stream_mask()) {
    return poison_bound_request(
        Sm87BulkV2P40OwnerError::kIncompleteDeviceJoin,
        "all_auxiliary_streams_must_join_main_before_real_terminal",
        static_cast<int>(cudaErrorInvalidResourceHandle));
  }
  for (std::size_t index = 1U; index < kSm87BulkV2P40StreamCount; ++index) {
    if (stream_submission_generations_[index] == 0U ||
        main_joined_stream_generations_[index] !=
            stream_submission_generations_[index]) {
      return poison_bound_request(
          Sm87BulkV2P40OwnerError::kIncompleteDeviceJoin,
          "main_must_join_latest_auxiliary_generation_before_real_terminal",
          static_cast<int>(cudaErrorInvalidResourceHandle), index);
    }
  }
  if (!work_receipt_complete()) {
    return poison_bound_request(
        Sm87BulkV2P40OwnerError::kIncompleteWorkReceipt,
        "real_terminal_requires_complete_exact_work_receipt",
        static_cast<int>(cudaErrorInvalidValue));
  }

  const std::size_t terminal =
      event_index(Sm87BulkV2P40ReusableEvent::kRequestTerminal);
  const int record_status =
      cuda_->record_event(events_[terminal], streams_[0U]);
  if (record_status != 0) {
    return poison_bound_request(
        Sm87BulkV2P40OwnerError::kCudaSubmission,
        "record_owner_terminal_event_before_request_state_observation",
        record_status);
  }
  ++receipt_.device_ordering_operations;
  ++event_generations_[terminal];
  event_producers_[terminal] = Sm87BulkV2P40Stream::kMain;
  event_stream_generations_[terminal] =
      stream_submission_generations_[0U];

  // This is the sole normal-completion host wait.  RequestState owns both the
  // fixed 8-byte destination and its observation; this owner receives only the
  // value after the terminal Main synchronization has completed.
  const Sm87BulkV2P40RequestStateHandoffResult handoff =
      request_state.synchronize_terminal_main_and_observe_handoff(
          request_access);
  if (!handoff) {
    if (handoff.terminal_main_synchronized) {
      receipt_.aggregate.terminal_host_waits = 1U;
    }
    const bool missing =
        handoff.status.error ==
        Sm87BulkV2P40RequestStateError::kHandoffNotEnqueued;
    const bool invalid = handoff.status.error ==
                         Sm87BulkV2P40RequestStateError::kInvalidHandoff;
    const int first_error =
        handoff.status.cuda_error != 0
            ? handoff.status.cuda_error
            : static_cast<int>(missing ? cudaErrorInvalidResourceHandle
                                       : cudaErrorInvalidValue);
    return poison_bound_request(
        missing ? Sm87BulkV2P40OwnerError::kMissingOwnerBoundHandoff
                : (invalid ? Sm87BulkV2P40OwnerError::kInvalidHandoff
                           : Sm87BulkV2P40OwnerError::kCudaSubmission),
        missing ? "owner_bound_handoff_d2h_must_precede_terminal"
                : (invalid
                       ? "owner_bound_terminal_handoff_is_invalid"
                       : "request_state_terminal_main_observation_failed"),
        first_error);
  }

  receipt_.aggregate.lifecycle = Sm87BulkV2P40OwnerLifecycle::kCompleted;
  receipt_.aggregate.terminal_host_waits = 1U;
  // One transitive Main drain covers every latest joined auxiliary
  // generation.  This is a receipt semantic, not five additional host waits.
  receipt_.aggregate.terminal_host_drains = 1U;
  receipt_.aggregate.all_streams_drained = true;
  receipt_.aggregate.state_committed = true;
  receipt_.aggregate.handoff_observed = true;
  receipt_.aggregate.handoff_token_id = handoff.token_id;
  receipt_.aggregate.handoff_nonfinite = handoff.nonfinite;
  receipt_.handoff_value_bits = handoff.value_bits;
  receipt_.aggregate.first_error = 0;
  active_request_state_ = nullptr;
  active_request_access_ = nullptr;
  state_ = Sm87BulkV2P40OwnerState::kCompleted;
  return ok();
}

Sm87BulkV2P40OwnerStatus
Sm87BulkV2P40Owner::hot_rearm_gdn_session_after_completed_request(
    const Sm87BulkV2P40ExecutionAccess& access,
    q3x::kernels::Sm87BulkV2GdnP40Session& session) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "gdn_hot_rearm_owner_bound_access");
  }

  // kCompleted is reached only after RequestState synchronizes this Owner's
  // Main stream.  Completion itself proves that every latest auxiliary
  // generation joined Main and that the fixed D2H handoff was observed.  Do
  // not accept the public aggregate, a caller boolean, cancellation, or the
  // five-stream error drain as a substitute for that exact transition.
  const auto& terminal = receipt_.aggregate;
  if (state_ != Sm87BulkV2P40OwnerState::kCompleted ||
      terminal.lifecycle != Sm87BulkV2P40OwnerLifecycle::kCompleted ||
      terminal.terminal_host_waits != 1U ||
      terminal.terminal_host_drains != 1U ||
      !terminal.all_streams_drained || !terminal.state_committed ||
      !terminal.handoff_observed || terminal.first_error != 0 ||
      terminal.cancellation_published || !receipt_.identity_valid() ||
      !work_receipt_complete()) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "gdn_hot_rearm_requires_exact_completed_terminal_sync");
  }

  if (!q3x::kernels::
          sm87_bulk_v2_gdn_p40_session_hot_rearm_candidate(session)) {
    return error(Sm87BulkV2P40OwnerError::kGdnSessionNotRearmable,
                 "gdn_hot_rearm_requires_48_bridged_healthy_epochs");
  }

  const auto& plan = session.sealed_plan;
  const auto& gdn_owner = plan.layers[0U];
  const std::array<void*, q3x::kernels::kSm87BulkV2GdnP40StreamCount>
      expected_streams{{
          streams_[stream_index(
              Sm87BulkV2P40Stream::kProjectionAndGdnProducer)],
          streams_[stream_index(Sm87BulkV2P40Stream::kGdnRecurrence)],
          streams_[stream_index(Sm87BulkV2P40Stream::kGdnEpilogue)],
      }};
  const std::array<void*, q3x::kernels::kSm87BulkV2GdnP40SlotCount>
      expected_prepared{{
          events_[event_index(Sm87BulkV2P40ReusableEvent::kGdnPrepared0)],
          events_[event_index(Sm87BulkV2P40ReusableEvent::kGdnPrepared1)],
      }};
  const std::array<void*, q3x::kernels::kSm87BulkV2GdnP40SlotCount>
      expected_recurrence{{
          events_[event_index(Sm87BulkV2P40ReusableEvent::kGdnRecurrence0)],
          events_[event_index(Sm87BulkV2P40ReusableEvent::kGdnRecurrence1)],
      }};
  const std::array<void*, q3x::kernels::kSm87BulkV2GdnP40SlotCount>
      expected_epilogue{{
          events_[event_index(Sm87BulkV2P40ReusableEvent::kGdnEpilogue0)],
          events_[event_index(Sm87BulkV2P40ReusableEvent::kGdnEpilogue1)],
      }};
  if (plan.main_stream != streams_[stream_index(Sm87BulkV2P40Stream::kMain)] ||
      plan.ingress_ready_event !=
          events_[event_index(
              Sm87BulkV2P40ReusableEvent::kProjectionInputReady)] ||
      !q3x::kernels::sm87_bulk_v2_gdn_p40_same_identities(
          gdn_owner.streams, expected_streams) ||
      !q3x::kernels::sm87_bulk_v2_gdn_p40_same_identities(
          gdn_owner.prepared_ready_events, expected_prepared) ||
      !q3x::kernels::sm87_bulk_v2_gdn_p40_same_identities(
          gdn_owner.recurrence_done_events, expected_recurrence) ||
      !q3x::kernels::sm87_bulk_v2_gdn_p40_same_identities(
          gdn_owner.epilogue_done_events, expected_epilogue) ||
      gdn_owner.cancellation_host_word != cancellation_host_word_ ||
      gdn_owner.cancellation_device_alias != cancellation_device_alias_) {
    return error(Sm87BulkV2P40OwnerError::kForeignGdnSession,
                 "gdn_hot_rearm_requires_exact_owner_stream_event_binding");
  }

  auto* const gdn_receipt = gdn_owner.submission_receipt;
  // The sole terminal Main wait is a transitive completion proof for these
  // exact streams and events.  Retire the GDN generation in host state only;
  // no cudaEventQuery, cudaStreamQuery, cudaStreamSynchronize, or other CUDA
  // call is made here.  The next epoch-zero enqueue increments generation and
  // clears the retained audit fields through begin_p40_submission().
  gdn_receipt->drain_attempted = true;
  gdn_receipt->drain_completed = true;
  gdn_receipt->lifecycle =
      q3x::kernels::Sm87BulkV2GdnP40OwnerLifecycle::kReady;
  gdn_receipt->reusable = true;
  session.next_epoch = 0U;
  session.bridged_epochs = 0U;
  session.bridge_pending = false;
  session.lifecycle =
      q3x::kernels::Sm87BulkV2GdnP40SessionLifecycle::kReady;

  if (!q3x::kernels::sm87_bulk_v2_gdn_p40_submission_receipt_valid(
          *gdn_receipt) ||
      !q3x::kernels::sm87_bulk_v2_gdn_p40_session_state_valid(session)) {
    // This branch is unreachable for a valid admitted session and does not
    // attempt to make an inconsistent owner reusable.
    gdn_receipt->lifecycle =
        q3x::kernels::Sm87BulkV2GdnP40OwnerLifecycle::kPoisoned;
    gdn_receipt->reusable = false;
    session.lifecycle =
        q3x::kernels::Sm87BulkV2GdnP40SessionLifecycle::kPoisoned;
    return error(Sm87BulkV2P40OwnerError::kGdnSessionNotRearmable,
                 "gdn_hot_rearm_postcondition_failed");
  }
  return ok();
}

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_OWNER_HOST_FIXTURE)
Sm87BulkV2P40OwnerStatus
Sm87BulkV2P40Owner::complete_synthetic_for_host_contract(
    const Sm87BulkV2P40ExecutionAccess& access,
    const std::uint32_t handoff_token_id,
    const std::uint32_t handoff_nonfinite) noexcept {
  if (!access_matches(access)) {
    return error(Sm87BulkV2P40OwnerError::kForeignExecutionAccess,
                 "synthetic_complete_owner_bound_access");
  }
  if (state_ != Sm87BulkV2P40OwnerState::kActive) {
    return error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                 "synthetic_complete_requires_active_request");
  }
  if (receipt_.joined_auxiliary_stream_mask != auxiliary_stream_mask()) {
    (void)cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned,
        static_cast<int>(cudaErrorInvalidResourceHandle));
    return error(Sm87BulkV2P40OwnerError::kIncompleteDeviceJoin,
                 "all_auxiliary_streams_must_join_main_before_terminal");
  }
  for (std::size_t index = 1U; index < kSm87BulkV2P40StreamCount; ++index) {
    if (stream_submission_generations_[index] == 0U ||
        main_joined_stream_generations_[index] !=
            stream_submission_generations_[index]) {
      (void)cancel_drain_and_transition(
          Sm87BulkV2P40OwnerLifecycle::kPoisoned,
          static_cast<int>(cudaErrorInvalidResourceHandle));
      return error(Sm87BulkV2P40OwnerError::kIncompleteDeviceJoin,
                   "main_must_join_latest_auxiliary_generation", 0, index);
    }
  }
  if (!work_receipt_complete()) {
    (void)cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned,
        static_cast<int>(cudaErrorInvalidValue));
    return error(Sm87BulkV2P40OwnerError::kIncompleteWorkReceipt,
                 "terminal_requires_complete_exact_work_receipt");
  }

  const std::size_t terminal = event_index(
      Sm87BulkV2P40ReusableEvent::kRequestTerminal);
  int cuda_status = cuda_->record_event(events_[terminal], streams_[0U]);
  if (cuda_status != 0) {
    return cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned, cuda_status);
  }
  ++receipt_.device_ordering_operations;
  ++event_generations_[terminal];
  event_producers_[terminal] = Sm87BulkV2P40Stream::kMain;

  cuda_status = cuda_->synchronize_stream(streams_[0U]);
  receipt_.aggregate.terminal_host_waits = 1U;
  receipt_.aggregate.terminal_host_drains = 1U;
  if (cuda_status != 0) {
    return cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned, cuda_status);
  }

  // This test-only value is observed strictly after terminal GPU completion.
  // A production compilation has no equivalent caller-value entry point.
  if (handoff_token_id >= kSm87BulkV2P40Vocabulary ||
      handoff_nonfinite != 0U) {
    (void)cancel_drain_and_transition(
        Sm87BulkV2P40OwnerLifecycle::kPoisoned,
        static_cast<int>(cudaErrorInvalidValue));
    return error(Sm87BulkV2P40OwnerError::kInvalidHandoff,
                 "terminal_handoff_requires_finite_in_vocabulary_token");
  }

  receipt_.aggregate.lifecycle = Sm87BulkV2P40OwnerLifecycle::kCompleted;
  receipt_.aggregate.all_streams_drained = true;
  receipt_.aggregate.state_committed = true;
  receipt_.aggregate.handoff_observed = true;
  receipt_.aggregate.handoff_token_id = handoff_token_id;
  receipt_.aggregate.handoff_nonfinite = handoff_nonfinite;
  receipt_.aggregate.first_error = 0;
  state_ = Sm87BulkV2P40OwnerState::kCompleted;
  return ok();
}
#endif

void Sm87BulkV2P40Owner::publish_cancellation(
    const std::uint32_t value) noexcept {
  if (cancellation_host_word_ != nullptr) {
    __atomic_store_n(cancellation_host_word_, value, __ATOMIC_RELEASE);
  }
}

void Sm87BulkV2P40Owner::release_resources() noexcept {
  if (state_ == Sm87BulkV2P40OwnerState::kDestroyed) {
    return;
  }
  if (cuda_ != nullptr &&
      (state_ == Sm87BulkV2P40OwnerState::kActive ||
       state_ == Sm87BulkV2P40OwnerState::kDraining)) {
    publish_cancellation(1U);
    for (void* const stream : streams_) {
      if (stream != nullptr) {
        (void)cuda_->synchronize_stream(stream);
      }
    }
  }
  // Non-owning active-request links are never dereferenced during teardown;
  // RequestState may have been destroyed first after performing its own
  // borrowed-stream drain.
  active_request_state_ = nullptr;
  active_request_access_ = nullptr;
  execution_access_.reset();
  if (cuda_ != nullptr) {
    for (std::size_t index = events_.size(); index > 0U; --index) {
      void*& event = events_[index - 1U];
      if (event != nullptr) {
        (void)cuda_->destroy_event(event);
        event = nullptr;
      }
    }
    for (std::size_t index = streams_.size(); index > 0U; --index) {
      void*& stream = streams_[index - 1U];
      if (stream != nullptr) {
        (void)cuda_->destroy_stream(stream);
        stream = nullptr;
      }
    }
    if (cancellation_host_word_ != nullptr) {
      (void)cuda_->free_mapped_host(cancellation_host_word_);
      cancellation_host_word_ = nullptr;
      cancellation_device_alias_ = nullptr;
    }
    if (device_control_arena_ != nullptr) {
      (void)cuda_->free_device(device_control_arena_);
      device_control_arena_ = nullptr;
    }
  }
  state_ = Sm87BulkV2P40OwnerState::kDestroyed;
}

Sm87BulkV2P40OwnerCreateResult
create_sm87_bulk_dataflow_v2_p40_owner_resources() noexcept {
  Sm87BulkV2P40OwnerCreateResult result;
#if !defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_OWNER_ADMISSION)
  result.status = error(Sm87BulkV2P40OwnerError::kAdmissionDisabled,
                        "Q3X_BUILD_SM87_BULK_DATAFLOW_V2_P40_OWNER_ADMISSION");
  return result;
#else
  static CudartSm87BulkV2P40Runtime cuda_runtime;
  std::unique_ptr<Sm87BulkV2P40Owner> owner(
      new (std::nothrow) Sm87BulkV2P40Owner(&cuda_runtime));
  if (owner == nullptr) {
    result.status = error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                          "allocate_v2_p40_owner");
    return result;
  }
  result.status = owner->initialize_resources();
  if (!result.status) {
    return result;
  }
  result.owner = std::move(owner);
  return result;
#endif
}

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_OWNER_HOST_FIXTURE)
Sm87BulkV2P40OwnerCreateResult Sm87BulkV2P40OwnerHostFixture::create(
    Sm87BulkV2P40CudaRuntime* const cuda) noexcept {
  Sm87BulkV2P40OwnerCreateResult result;
  std::unique_ptr<Sm87BulkV2P40Owner> owner(
      new (std::nothrow) Sm87BulkV2P40Owner(cuda));
  if (owner == nullptr) {
    result.status = error(Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                          "allocate_host_fixture_owner");
    return result;
  }
  result.status = owner->initialize_resources();
  if (!result.status) {
    return result;
  }
  result.owner = std::move(owner);
  return result;
}

std::unique_ptr<Sm87BulkV2P40ConstituentSealAccess>
Sm87BulkV2P40OwnerHostFixture::mint_synthetic_constituent_seal(
    const Sm87BulkV2P40Owner& owner,
    const Sm87BulkV2P40OwnerIdentity& evidence) noexcept {
  if (owner.state_ != Sm87BulkV2P40OwnerState::kResourcesReady) {
    return nullptr;
  }
  std::unique_ptr<Sm87BulkV2P40ConstituentSealAccess> seal(
      new (std::nothrow) Sm87BulkV2P40ConstituentSealAccess());
  if (seal == nullptr) {
    return nullptr;
  }
  seal->identity_ = evidence;
  seal->identity_.plan_magic = kSm87BulkV2P40PlanMagic;
  seal->identity_.abi_major = kSm87BulkV2P40PlanAbiMajor;
  seal->identity_.abi_minor = kSm87BulkV2P40PlanAbiMinor;
  seal->identity_.owner_identity = owner.owner_identity_;
  seal->identity_.seal_nonce = next_identity();
  seal->identity_.device_ordinal = owner.device_ordinal_;
  seal->identity_.execution_class =
      Sm87BulkV2P40ExecutionClass::kSyntheticHostContract;
  seal->identity_.authenticated_real_constituents = false;
  seal->identity_.exact_numerical_contract_qualified = false;
  seal->identity_.development_execution_eligible = false;
  seal->identity_.production_dispatch_eligible = false;
  seal->bound_owner_identity_ = owner.owner_identity_;
  seal->streams_ = owner.streams_;
  seal->events_ = owner.events_;
  seal->device_control_arena_ = owner.device_control_arena_;
  seal->cancellation_host_word_ = owner.cancellation_host_word_;
  seal->cancellation_device_alias_ = owner.cancellation_device_alias_;
  seal->real_fp8_binding_seal = false;
  seal->real_attention_binding_seal = false;
  seal->real_bf16_ab_binding_seal = false;
  seal->real_gdn_session_seal = false;
  seal->real_nvfp4_binding_seal = false;
  seal->real_request_arena_seal = false;
  seal->real_pinned_handoff_seal = false;
  seal->all_static_resource_checks_complete = true;
  seal->authenticated_real_constituents = false;
  seal->exact_numerical_contract_qualified = false;
  seal->default_off_direction_witness_eligible = false;
  seal->default_off_candidate_eligible = false;
  seal->production_dispatch_eligible = false;
  seal->synthetic_host_contract_only = true;
  return seal;
}

std::unique_ptr<Sm87BulkV2P40ConstituentSealAccess>
Sm87BulkV2P40OwnerHostFixture::mint_direction_witness_constituent_seal(
    const Sm87BulkV2P40Owner& owner,
    const Sm87BulkV2P40OwnerIdentity& evidence) noexcept {
  if (owner.state_ != Sm87BulkV2P40OwnerState::kResourcesReady) {
    return nullptr;
  }
  std::unique_ptr<Sm87BulkV2P40ConstituentSealAccess> seal(
      new (std::nothrow) Sm87BulkV2P40ConstituentSealAccess());
  if (seal == nullptr) {
    return nullptr;
  }
  seal->identity_ = evidence;
  seal->identity_.plan_magic = kSm87BulkV2P40PlanMagic;
  seal->identity_.abi_major = kSm87BulkV2P40PlanAbiMajor;
  seal->identity_.abi_minor = kSm87BulkV2P40PlanAbiMinor;
  seal->identity_.owner_identity = owner.owner_identity_;
  seal->identity_.seal_nonce = next_identity();
  seal->identity_.device_ordinal = owner.device_ordinal_;
  seal->identity_.execution_class =
      Sm87BulkV2P40ExecutionClass::kDefaultOffDirectionWitness;
  seal->identity_.authenticated_real_constituents = true;
  seal->identity_.exact_numerical_contract_qualified = false;
  seal->identity_.development_execution_eligible = true;
  seal->identity_.production_dispatch_eligible = false;
  seal->bound_owner_identity_ = owner.owner_identity_;
  seal->streams_ = owner.streams_;
  seal->events_ = owner.events_;
  seal->device_control_arena_ = owner.device_control_arena_;
  seal->cancellation_host_word_ = owner.cancellation_host_word_;
  seal->cancellation_device_alias_ = owner.cancellation_device_alias_;
  seal->real_fp8_binding_seal = true;
  seal->real_attention_binding_seal = true;
  seal->real_bf16_ab_binding_seal = true;
  seal->real_gdn_session_seal = true;
  seal->real_nvfp4_binding_seal = true;
  seal->real_request_arena_seal = true;
  seal->real_pinned_handoff_seal = true;
  seal->all_static_resource_checks_complete = true;
  seal->authenticated_real_constituents = true;
  seal->exact_numerical_contract_qualified = false;
  seal->default_off_direction_witness_eligible = true;
  seal->default_off_candidate_eligible = false;
  seal->production_dispatch_eligible = false;
  seal->synthetic_host_contract_only = false;
  return seal;
}

void Sm87BulkV2P40OwnerHostFixture::populate_complete_work_receipt(
    Sm87BulkV2P40Owner* const owner) noexcept {
  if (owner == nullptr || owner->state_ != Sm87BulkV2P40OwnerState::kActive) {
    return;
  }
  auto& receipt = owner->receipt_.aggregate;
  receipt.completed_layers = kSm87BulkV2P40Layers;
  receipt.completed_gdn_layers = kSm87BulkV2P40GdnLayers;
  receipt.completed_full_layers = kSm87BulkV2P40FullLayers;
  receipt.closed_layer_residuals = kSm87BulkV2P40Layers;
  receipt.closed_gdn_state_publications = kSm87BulkV2P40GdnLayers;
  receipt.logical_projection_roles = kSm87BulkV2P40LogicalProjectionRoles;
  receipt.fused_outer_operations = kSm87BulkV2P40FusedOuterOperations;
  receipt.projection_conventional_operations =
      kSm87BulkV2P40ProjectionConventionalOperations;
  receipt.projection_successor =
      sm87_bulk_v2_p40_projection_successor_receipt();
  receipt.projection_successor.fp8_gdn_input_whole_launches =
      kSm87BulkV2P40Fp8GdnInputOuterOperations;
  receipt.projection_successor.fp8_full_input_whole_launches =
      kSm87BulkV2P40Fp8FullInputOuterOperations;
  receipt.projection_successor.fp8_output_whole_launches =
      kSm87BulkV2P40Fp8OutputOuterOperations;
  receipt.projection_successor.fp8_whole_role_launches =
      kSm87BulkV2P40Fp8WholeRoleLaunches;
  receipt.projection_successor.nvfp4_gate_up_whole_launches =
      kSm87BulkV2P40NvFp4GateUpOuterOperations;
  receipt.projection_successor.nvfp4_down_whole_launches =
      kSm87BulkV2P40NvFp4DownOuterOperations;
  receipt.projection_successor.nvfp4_whole_role_launches =
      kSm87BulkV2P40NvFp4WholeRoleLaunches;
  receipt.projection_successor.bf16_ab_physical_launches =
      kSm87BulkV2P40Bf16AbPhysicalLaunches;
  receipt.enqueued_attention_launches =
      kSm87BulkV2P40FullLayers *
      q3x::kernels::kSm87BulkV2AttentionKernelLaunches;
  receipt.enqueued_attention_preprocess_panels = 80U;
  receipt.enqueued_bf16_ab_launches = 48U;
  receipt.enqueued_gdn_producer_chunks = 30'000U;
  receipt.enqueued_gdn_recurrence_chunks = 30'000U;
  receipt.enqueued_gdn_epilogue_chunks = 30'000U;
  receipt.enqueued_gdn_persistent_copies = 96U;
  receipt.enqueued_final_norm = 1U;
  receipt.enqueued_lm_head = 1U;
  receipt.enqueued_argmax = 1U;
  receipt.enqueued_handoff_d2h = 1U;
  receipt.last_submitted_layer = kSm87BulkV2P40Layers - 1U;
  receipt.last_submitted_family = Sm87BulkV2P40FamilyPhase::kFinalHandoff;
  receipt.last_submitted_segment = 0U;
  receipt.last_submitted_constituent = 3U;
}

std::array<void*, kSm87BulkV2P40StreamCount>
Sm87BulkV2P40OwnerHostFixture::borrow_streams_for_request_state(
    const Sm87BulkV2P40Owner& owner) noexcept {
  return owner.state_ == Sm87BulkV2P40OwnerState::kResourcesReady
             ? owner.streams_
             : std::array<void*, kSm87BulkV2P40StreamCount>{};
}

void Sm87BulkV2P40OwnerHostFixture::
    bind_gdn_owner_handles_for_host_contract(
        const Sm87BulkV2P40Owner& owner,
        q3x::kernels::Sm87BulkV2GdnP40SessionPlan* const plan) noexcept {
  if (plan == nullptr || owner.state_ == Sm87BulkV2P40OwnerState::kEmpty ||
      owner.state_ == Sm87BulkV2P40OwnerState::kDestroyed) {
    return;
  }
  const std::array<void*, q3x::kernels::kSm87BulkV2GdnP40StreamCount>
      streams{{
          owner.streams_[stream_index(
              Sm87BulkV2P40Stream::kProjectionAndGdnProducer)],
          owner.streams_[stream_index(Sm87BulkV2P40Stream::kGdnRecurrence)],
          owner.streams_[stream_index(Sm87BulkV2P40Stream::kGdnEpilogue)],
      }};
  const std::array<void*, q3x::kernels::kSm87BulkV2GdnP40SlotCount>
      prepared{{
          owner.events_[event_index(
              Sm87BulkV2P40ReusableEvent::kGdnPrepared0)],
          owner.events_[event_index(
              Sm87BulkV2P40ReusableEvent::kGdnPrepared1)],
      }};
  const std::array<void*, q3x::kernels::kSm87BulkV2GdnP40SlotCount>
      recurrence{{
          owner.events_[event_index(
              Sm87BulkV2P40ReusableEvent::kGdnRecurrence0)],
          owner.events_[event_index(
              Sm87BulkV2P40ReusableEvent::kGdnRecurrence1)],
      }};
  const std::array<void*, q3x::kernels::kSm87BulkV2GdnP40SlotCount>
      epilogue{{
          owner.events_[event_index(
              Sm87BulkV2P40ReusableEvent::kGdnEpilogue0)],
          owner.events_[event_index(
              Sm87BulkV2P40ReusableEvent::kGdnEpilogue1)],
      }};
  for (auto& layer : plan->layers) {
    layer.streams = streams;
    layer.prepared_ready_events = prepared;
    layer.recurrence_done_events = recurrence;
    layer.epilogue_done_events = epilogue;
    layer.cancellation_host_word = owner.cancellation_host_word_;
    layer.cancellation_device_alias = owner.cancellation_device_alias_;
  }
  plan->main_stream =
      owner.streams_[stream_index(Sm87BulkV2P40Stream::kMain)];
  plan->ingress_ready_event = owner.events_[event_index(
      Sm87BulkV2P40ReusableEvent::kProjectionInputReady)];
}

Sm87BulkV2P40OwnerStatus Sm87BulkV2P40OwnerHostFixture::
    hot_rearm_gdn_session_after_completed_request(
        Sm87BulkV2P40Owner& owner,
        const Sm87BulkV2P40ExecutionAccess& access,
        q3x::kernels::Sm87BulkV2GdnP40Session& session) noexcept {
  return owner.hot_rearm_gdn_session_after_completed_request(access, session);
}
#endif

}  // namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail
