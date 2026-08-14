#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_attention_l2_cohort.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace q3x::kernels::sm87_bulk_v2_attention_execution_detail {

inline constexpr std::size_t kSm87BulkV2AttentionFullLayerBindings = 16U;
inline constexpr std::uint64_t kSm87BulkV2AttentionSealReceiptMagic =
    0x5133'5832'4154'534cULL;
inline constexpr std::uint32_t kSm87BulkV2AttentionSealReceiptVersion = 1U;
inline constexpr std::uint64_t kSm87BulkV2AttentionSubmissionReceiptMagic =
    0x5133'5832'4154'5355ULL;
inline constexpr std::uint32_t
    kSm87BulkV2AttentionSubmissionReceiptVersion = 1U;

[[nodiscard]] constexpr std::size_t sm87_bulk_v2_attention_full_ordinal(
    const std::size_t model_layer) noexcept {
  return model_layer < 64U && (model_layer + 1U) % 4U == 0U
             ? model_layer / 4U
             : kSm87BulkV2AttentionFullLayerBindings;
}

struct Sm87BulkV2AttentionBinding final {
  std::uint32_t model_layer = 0U;
  Sm87BulkV2AttentionArguments arguments{};
};

// The startup owner supplies all 16 full-Attention layer bindings.  Their
// buffers and the exact nonblocking stream must outlive the sealed access.
struct Sm87BulkV2AttentionSealRequest final {
  const Sm87BulkV2AttentionBinding* layer_bindings = nullptr;
  std::size_t layer_binding_count = 0U;
  std::int32_t device_ordinal = -1;
  void* cuda_stream = nullptr;
  std::uint64_t deployment_identity = 0U;
  std::uint64_t binding_catalog_identity = 0U;
  std::uint64_t binding_lifetime_owner_identity = 0U;
  std::uint64_t cuda_stream_owner_identity = 0U;
};

struct Sm87BulkV2AttentionSealReceipt final {
  std::uint64_t magic = 0U;
  std::uint32_t version = 0U;
  std::uint64_t seal_nonce = 0U;
  std::uint64_t deployment_identity = 0U;
  std::uint64_t binding_catalog_identity = 0U;
  std::uint64_t binding_lifetime_owner_identity = 0U;
  std::uint64_t cuda_stream_owner_identity = 0U;
  std::int32_t device_ordinal = -1;
  void* cuda_stream = nullptr;
  std::size_t sealed_layer_bindings = 0U;
  std::size_t sealed_resource_kernels = 0U;
  std::size_t hot_path_static_cuda_queries =
      std::numeric_limits<std::size_t>::max();
  bool exact_sm87_device_validated = false;
  bool nonblocking_stream_validated = false;
  bool complete_device_ranges_validated = false;
  bool static_resources_validated_at_startup = false;
  bool request_hot_path_prevalidated = false;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] inline bool sm87_bulk_v2_attention_seal_receipt_valid(
    const Sm87BulkV2AttentionSealReceipt& receipt) noexcept {
  return receipt.magic == kSm87BulkV2AttentionSealReceiptMagic &&
         receipt.version == kSm87BulkV2AttentionSealReceiptVersion &&
         receipt.seal_nonce != 0U && receipt.deployment_identity != 0U &&
         receipt.binding_catalog_identity != 0U &&
         receipt.binding_lifetime_owner_identity != 0U &&
         receipt.cuda_stream_owner_identity != 0U &&
         receipt.device_ordinal >= 0 && receipt.cuda_stream != nullptr &&
         receipt.sealed_layer_bindings ==
             kSm87BulkV2AttentionFullLayerBindings &&
         receipt.sealed_resource_kernels == 1U &&
         receipt.hot_path_static_cuda_queries == 0U &&
         receipt.exact_sm87_device_validated &&
         receipt.nonblocking_stream_validated &&
         receipt.complete_device_ranges_validated &&
         receipt.static_resources_validated_at_startup &&
         receipt.request_hot_path_prevalidated &&
         !receipt.numerical_contract_qualified &&
         !receipt.production_dispatch_eligible;
}

enum class Sm87BulkV2AttentionSealFailure : std::uint8_t {
  kNone = 0U,
  kInvalidRequest,
  kDevice,
  kStream,
  kResources,
  kBindingIdentity,
  kBindingRange,
  kAllocation,
};

enum class Sm87BulkV2AttentionSubmissionState : std::uint8_t {
  kInvalid = 0U,
  kFailedBeforeSubmission,
  kFailedAfterPartialSubmission,
  kSubmitted,
};

struct Sm87BulkV2AttentionSubmissionReceipt final {
  std::uint64_t magic = 0U;
  std::uint32_t version = 0U;
  std::uint64_t seal_nonce = 0U;
  std::uint64_t request_epoch = 0U;
  std::uint32_t model_layer = 0U;
  std::size_t expected_launches = 0U;
  std::size_t attempted_launches = 0U;
  std::size_t submitted_launches = 0U;
  std::size_t failed_launch_ordinal =
      std::numeric_limits<std::size_t>::max();
  int cuda_error = 0;
  Sm87BulkV2AttentionSubmissionState state =
      Sm87BulkV2AttentionSubmissionState::kInvalid;
  bool prevalidated_hot_path_used = false;
  bool static_cuda_query_issued = true;
  bool owner_drain_required = false;
};

[[nodiscard]] inline bool sm87_bulk_v2_attention_submission_receipt_valid(
    const Sm87BulkV2AttentionSubmissionReceipt& receipt) noexcept {
  if (receipt.magic != kSm87BulkV2AttentionSubmissionReceiptMagic ||
      receipt.version != kSm87BulkV2AttentionSubmissionReceiptVersion ||
      receipt.seal_nonce == 0U || receipt.request_epoch == 0U ||
      sm87_bulk_v2_attention_full_ordinal(receipt.model_layer) >=
          kSm87BulkV2AttentionFullLayerBindings ||
      receipt.expected_launches != kSm87BulkV2AttentionKernelLaunches ||
      !receipt.prevalidated_hot_path_used ||
      receipt.static_cuda_query_issued) {
    return false;
  }
  if (receipt.state == Sm87BulkV2AttentionSubmissionState::kSubmitted) {
    return receipt.attempted_launches == receipt.expected_launches &&
           receipt.submitted_launches == receipt.expected_launches &&
           receipt.failed_launch_ordinal ==
               std::numeric_limits<std::size_t>::max() &&
           receipt.cuda_error == 0 && !receipt.owner_drain_required;
  }
  if (receipt.state ==
      Sm87BulkV2AttentionSubmissionState::kFailedBeforeSubmission) {
    const bool preexisting_error =
        receipt.attempted_launches == 0U &&
        receipt.failed_launch_ordinal ==
            std::numeric_limits<std::size_t>::max();
    const bool first_launch_rejected =
        receipt.attempted_launches == 1U &&
        receipt.failed_launch_ordinal == 0U;
    return receipt.submitted_launches == 0U && receipt.cuda_error != 0 &&
           !receipt.owner_drain_required &&
           (preexisting_error || first_launch_rejected);
  }
  if (receipt.state ==
      Sm87BulkV2AttentionSubmissionState::kFailedAfterPartialSubmission) {
    return receipt.submitted_launches > 0U &&
           receipt.submitted_launches < receipt.expected_launches &&
           receipt.attempted_launches == receipt.submitted_launches + 1U &&
           receipt.failed_launch_ordinal == receipt.submitted_launches &&
           receipt.cuda_error != 0 && receipt.owner_drain_required;
  }
  return false;
}

struct Sm87BulkV2AttentionSealResult;

class Sm87BulkV2AttentionSealedAccess final {
 public:
  Sm87BulkV2AttentionSealedAccess(
      const Sm87BulkV2AttentionSealedAccess&) = delete;
  Sm87BulkV2AttentionSealedAccess& operator=(
      const Sm87BulkV2AttentionSealedAccess&) = delete;
  Sm87BulkV2AttentionSealedAccess(
      Sm87BulkV2AttentionSealedAccess&&) = delete;
  Sm87BulkV2AttentionSealedAccess& operator=(
      Sm87BulkV2AttentionSealedAccess&&) = delete;

  [[nodiscard]] bool valid() const noexcept {
    return sm87_bulk_v2_attention_seal_receipt_valid(receipt_) &&
           receipt_.cuda_stream == cuda_stream_;
  }
  [[nodiscard]] const Sm87BulkV2AttentionSealReceipt& receipt()
      const noexcept {
    return receipt_;
  }
  [[nodiscard]] void* cuda_stream() const noexcept { return cuda_stream_; }

 private:
  Sm87BulkV2AttentionSealedAccess(
      const std::array<Sm87BulkV2AttentionBinding,
                       kSm87BulkV2AttentionFullLayerBindings>& bindings,
      void* const cuda_stream,
      const Sm87BulkV2AttentionSealReceipt& receipt) noexcept
      : bindings_(bindings), cuda_stream_(cuda_stream), receipt_(receipt) {}

  std::array<Sm87BulkV2AttentionBinding,
             kSm87BulkV2AttentionFullLayerBindings>
      bindings_{};
  void* cuda_stream_ = nullptr;
  Sm87BulkV2AttentionSealReceipt receipt_{};

  friend Sm87BulkV2AttentionSealResult
  seal_sm87_bulk_v2_attention_p40_cuda(
      const Sm87BulkV2AttentionSealRequest&) noexcept;
  friend int enqueue_sm87_bulk_v2_attention_p40_prevalidated_cuda(
      const Sm87BulkV2AttentionSealedAccess&, std::uint64_t, std::size_t,
      Sm87BulkV2AttentionSubmissionReceipt*) noexcept;
};

struct Sm87BulkV2AttentionSealResult final {
  std::unique_ptr<Sm87BulkV2AttentionSealedAccess> access;
  Sm87BulkV2AttentionSealReceipt receipt{};
  int cuda_error = 0;
  Sm87BulkV2AttentionSealFailure failure =
      Sm87BulkV2AttentionSealFailure::kNone;

  [[nodiscard]] explicit operator bool() const noexcept {
    return access != nullptr && access->valid() &&
           sm87_bulk_v2_attention_seal_receipt_valid(receipt) &&
           access->receipt().seal_nonce == receipt.seal_nonce &&
           cuda_error == 0 &&
           failure == Sm87BulkV2AttentionSealFailure::kNone;
  }
};

// Performs the complete 16-layer device/stream/resource/range check before
// request admission and returns the only authority accepted by hot enqueue.
[[nodiscard]] Sm87BulkV2AttentionSealResult
seal_sm87_bulk_v2_attention_p40_cuda(
    const Sm87BulkV2AttentionSealRequest& request) noexcept;

// Enqueues the four same-KV-head launches for one canonical full-Attention
// layer.  No device, pointer, allocation, stream, function, shared-attribute,
// or occupancy query is reachable from this function.  Partial submission is
// explicit and leaves request-wide drain/poison policy with the caller.
[[nodiscard]] int enqueue_sm87_bulk_v2_attention_p40_prevalidated_cuda(
    const Sm87BulkV2AttentionSealedAccess& access,
    std::uint64_t request_epoch, std::size_t model_layer,
    Sm87BulkV2AttentionSubmissionReceipt* receipt) noexcept;

static_assert(sm87_bulk_v2_attention_full_ordinal(3U) == 0U);
static_assert(sm87_bulk_v2_attention_full_ordinal(63U) == 15U);
static_assert(sm87_bulk_v2_attention_full_ordinal(0U) ==
              kSm87BulkV2AttentionFullLayerBindings);

}  // namespace q3x::kernels::sm87_bulk_v2_attention_execution_detail
