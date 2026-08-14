#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_projection.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace q3x::kernels::sm87_bulk_v2_fp8_execution_detail {

inline constexpr std::uint64_t kSm87BulkV2Fp8SealReceiptMagic =
    0x5133'5832'5038'534cULL;
inline constexpr std::uint32_t kSm87BulkV2Fp8SealReceiptVersion = 1U;
inline constexpr std::uint64_t kSm87BulkV2Fp8SubmissionReceiptMagic =
    0x5133'5832'5038'5355ULL;
inline constexpr std::uint32_t kSm87BulkV2Fp8SubmissionReceiptVersion = 1U;

// The startup owner supplies the complete, canonical 128-role catalog.  The
// catalog and every pointed-to allocation must outlive the returned access.
// The access copies the bindings, but deliberately does not take ownership of
// the stream, activations, outputs, or authenticated weight arena.
struct Sm87BulkV2Fp8SealRequest final {
  const Sm87BulkV2Fp8RoleArguments* role_bindings = nullptr;
  std::size_t role_binding_count = 0U;
  std::int32_t device_ordinal = -1;
  void* cuda_stream = nullptr;
  std::uint64_t deployment_identity = 0U;
  std::uint64_t binding_catalog_identity = 0U;
  std::uint64_t binding_lifetime_owner_identity = 0U;
  std::uint64_t cuda_stream_owner_identity = 0U;
  std::uint64_t authenticated_weight_owner_identity = 0U;
};

struct Sm87BulkV2Fp8SealReceipt final {
  std::uint64_t magic = 0U;
  std::uint32_t version = 0U;
  std::uint64_t seal_nonce = 0U;
  std::uint64_t deployment_identity = 0U;
  std::uint64_t binding_catalog_identity = 0U;
  std::uint64_t binding_lifetime_owner_identity = 0U;
  std::uint64_t cuda_stream_owner_identity = 0U;
  std::uint64_t authenticated_weight_owner_identity = 0U;
  std::int32_t device_ordinal = -1;
  void* cuda_stream = nullptr;
  std::size_t sealed_role_bindings = 0U;
  std::size_t sealed_resource_roles = 0U;
  std::size_t hot_path_static_cuda_queries =
      std::numeric_limits<std::size_t>::max();
  bool exact_sm87_device_validated = false;
  bool nonblocking_stream_validated = false;
  bool complete_device_ranges_validated = false;
  bool all_authenticated_weight_owners_match = false;
  bool static_resources_validated_at_startup = false;
  bool request_hot_path_prevalidated = false;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] inline bool sm87_bulk_v2_fp8_seal_receipt_valid(
    const Sm87BulkV2Fp8SealReceipt& receipt) noexcept {
  return receipt.magic == kSm87BulkV2Fp8SealReceiptMagic &&
         receipt.version == kSm87BulkV2Fp8SealReceiptVersion &&
         receipt.seal_nonce != 0U && receipt.deployment_identity != 0U &&
         receipt.binding_catalog_identity != 0U &&
         receipt.binding_lifetime_owner_identity != 0U &&
         receipt.cuda_stream_owner_identity != 0U &&
         receipt.authenticated_weight_owner_identity != 0U &&
         receipt.device_ordinal >= 0 && receipt.cuda_stream != nullptr &&
         receipt.sealed_role_bindings == kSm87BulkV2Fp8LogicalRoleCount &&
         receipt.sealed_resource_roles == 3U &&
         receipt.hot_path_static_cuda_queries == 0U &&
         receipt.exact_sm87_device_validated &&
         receipt.nonblocking_stream_validated &&
         receipt.complete_device_ranges_validated &&
         receipt.all_authenticated_weight_owners_match &&
         receipt.static_resources_validated_at_startup &&
         receipt.request_hot_path_prevalidated &&
         !receipt.numerical_contract_qualified &&
         !receipt.production_dispatch_eligible;
}

enum class Sm87BulkV2Fp8SealFailure : std::uint8_t {
  kNone = 0U,
  kInvalidRequest,
  kDevice,
  kStream,
  kResources,
  kBindingIdentity,
  kBindingRange,
  kAllocation,
};

enum class Sm87BulkV2Fp8SubmissionState : std::uint8_t {
  kInvalid = 0U,
  kFailedBeforeSubmission,
  kFailedAfterPartialSubmission,
  kSubmitted,
};

struct Sm87BulkV2Fp8SubmissionReceipt final {
  std::uint64_t magic = 0U;
  std::uint32_t version = 0U;
  std::uint64_t seal_nonce = 0U;
  std::uint64_t request_epoch = 0U;
  std::uint32_t layer = 0U;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::size_t expected_launches = 0U;
  std::size_t attempted_launches = 0U;
  std::size_t submitted_launches = 0U;
  std::size_t failed_launch_ordinal =
      std::numeric_limits<std::size_t>::max();
  int cuda_error = 0;
  Sm87BulkV2Fp8SubmissionState state =
      Sm87BulkV2Fp8SubmissionState::kInvalid;
  bool prevalidated_hot_path_used = false;
  bool static_cuda_query_issued = true;
  bool owner_drain_required = false;
};

[[nodiscard]] constexpr std::size_t sm87_bulk_v2_fp8_role_ordinal(
    const std::size_t layer,
    const Sm87TargetAotProjectionRole role) noexcept {
  if (layer >= kSm87BulkV2Fp8LayerCount) {
    return kSm87BulkV2Fp8LogicalRoleCount;
  }
  if (role == Sm87TargetAotProjectionRole::kFp8AttentionOutput) {
    return layer * 2U + 1U;
  }
  const auto expected_input =
      (layer + 1U) % 4U == 0U
          ? Sm87TargetAotProjectionRole::kFp8FullQkv
          : Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
  return role == expected_input ? layer * 2U
                                : kSm87BulkV2Fp8LogicalRoleCount;
}

[[nodiscard]] inline bool sm87_bulk_v2_fp8_submission_receipt_valid(
    const Sm87BulkV2Fp8SubmissionReceipt& receipt) noexcept {
  if (receipt.magic != kSm87BulkV2Fp8SubmissionReceiptMagic ||
      receipt.version != kSm87BulkV2Fp8SubmissionReceiptVersion ||
      receipt.seal_nonce == 0U || receipt.request_epoch == 0U ||
      sm87_bulk_v2_fp8_role_ordinal(receipt.layer, receipt.role) >=
          kSm87BulkV2Fp8LogicalRoleCount ||
      receipt.expected_launches != kSm87BulkV2Fp8SegmentsPerRole ||
      !receipt.prevalidated_hot_path_used ||
      receipt.static_cuda_query_issued) {
    return false;
  }
  if (receipt.state == Sm87BulkV2Fp8SubmissionState::kSubmitted) {
    return receipt.attempted_launches == receipt.expected_launches &&
           receipt.submitted_launches == receipt.expected_launches &&
           receipt.failed_launch_ordinal ==
               std::numeric_limits<std::size_t>::max() &&
           receipt.cuda_error == 0 && !receipt.owner_drain_required;
  }
  if (receipt.state ==
      Sm87BulkV2Fp8SubmissionState::kFailedBeforeSubmission) {
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
      Sm87BulkV2Fp8SubmissionState::kFailedAfterPartialSubmission) {
    return receipt.submitted_launches > 0U &&
           receipt.submitted_launches < receipt.expected_launches &&
           receipt.attempted_launches == receipt.submitted_launches + 1U &&
           receipt.failed_launch_ordinal == receipt.submitted_launches &&
           receipt.cuda_error != 0 && receipt.owner_drain_required;
  }
  return false;
}

struct Sm87BulkV2Fp8SealResult;

// This is the only launch authority accepted by the hot entry point.  The
// private constructor prevents a caller from presenting unchecked pointers as
// a startup-sealed binding catalog.
class Sm87BulkV2Fp8SealedAccess final {
 public:
  Sm87BulkV2Fp8SealedAccess(const Sm87BulkV2Fp8SealedAccess&) = delete;
  Sm87BulkV2Fp8SealedAccess& operator=(
      const Sm87BulkV2Fp8SealedAccess&) = delete;
  Sm87BulkV2Fp8SealedAccess(Sm87BulkV2Fp8SealedAccess&&) = delete;
  Sm87BulkV2Fp8SealedAccess& operator=(
      Sm87BulkV2Fp8SealedAccess&&) = delete;

  [[nodiscard]] bool valid() const noexcept {
    return sm87_bulk_v2_fp8_seal_receipt_valid(receipt_) &&
           receipt_.cuda_stream == cuda_stream_;
  }
  [[nodiscard]] const Sm87BulkV2Fp8SealReceipt& receipt() const noexcept {
    return receipt_;
  }
  [[nodiscard]] void* cuda_stream() const noexcept { return cuda_stream_; }

 private:
  Sm87BulkV2Fp8SealedAccess(
      const std::array<Sm87BulkV2Fp8RoleArguments,
                       kSm87BulkV2Fp8LogicalRoleCount>& bindings,
      void* const cuda_stream,
      const Sm87BulkV2Fp8SealReceipt& receipt) noexcept
      : bindings_(bindings), cuda_stream_(cuda_stream), receipt_(receipt) {}

  std::array<Sm87BulkV2Fp8RoleArguments,
             kSm87BulkV2Fp8LogicalRoleCount>
      bindings_{};
  void* cuda_stream_ = nullptr;
  Sm87BulkV2Fp8SealReceipt receipt_{};

  friend Sm87BulkV2Fp8SealResult seal_sm87_bulk_v2_fp8_p40_cuda(
      const Sm87BulkV2Fp8SealRequest&) noexcept;
  friend int enqueue_sm87_bulk_v2_fp8_role_p40_prevalidated_cuda(
      const Sm87BulkV2Fp8SealedAccess&, std::uint64_t, std::size_t,
      Sm87TargetAotProjectionRole,
      Sm87BulkV2Fp8SubmissionReceipt*) noexcept;
};

struct Sm87BulkV2Fp8SealResult final {
  std::unique_ptr<Sm87BulkV2Fp8SealedAccess> access;
  Sm87BulkV2Fp8SealReceipt receipt{};
  int cuda_error = 0;
  Sm87BulkV2Fp8SealFailure failure = Sm87BulkV2Fp8SealFailure::kNone;

  [[nodiscard]] explicit operator bool() const noexcept {
    return access != nullptr && access->valid() &&
           sm87_bulk_v2_fp8_seal_receipt_valid(receipt) &&
           access->receipt().seal_nonce == receipt.seal_nonce &&
           cuda_error == 0 && failure == Sm87BulkV2Fp8SealFailure::kNone;
  }
};

// Performs every device, stream, resource, authenticated-owner, structural,
// and complete allocation-range check once, before request admission.
[[nodiscard]] Sm87BulkV2Fp8SealResult seal_sm87_bulk_v2_fp8_p40_cuda(
    const Sm87BulkV2Fp8SealRequest& request) noexcept;

// Enqueues exactly the 40 P40 segments for one canonical layer/role.  No
// cudaGetDevice, cudaGetDeviceProperties, cudaPointerGetAttributes,
// cudaMemGetAddressRange, cudaStreamGetFlags, cudaFuncGetAttributes,
// cudaFuncSetAttribute, or occupancy query is reachable from this function.
// A failure after at least one successful enqueue is reported as owner-drain
// required; the caller owns request-wide drain/poison policy.
[[nodiscard]] int enqueue_sm87_bulk_v2_fp8_role_p40_prevalidated_cuda(
    const Sm87BulkV2Fp8SealedAccess& access, std::uint64_t request_epoch,
    std::size_t layer, Sm87TargetAotProjectionRole role,
    Sm87BulkV2Fp8SubmissionReceipt* receipt) noexcept;

static_assert(sm87_bulk_v2_fp8_role_ordinal(
                  0U, Sm87TargetAotProjectionRole::kFp8GdnQkvZ) == 0U);
static_assert(sm87_bulk_v2_fp8_role_ordinal(
                  3U, Sm87TargetAotProjectionRole::kFp8FullQkv) == 6U);
static_assert(sm87_bulk_v2_fp8_role_ordinal(
                  63U,
                  Sm87TargetAotProjectionRole::kFp8AttentionOutput) == 127U);

}  // namespace q3x::kernels::sm87_bulk_v2_fp8_execution_detail
