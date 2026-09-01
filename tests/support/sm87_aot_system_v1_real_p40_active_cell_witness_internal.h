#pragma once

#if !defined(Q3X_ENABLE_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS)
#error "The real-P40 active-cell witness is a private, default-off admission"
#endif
#if !defined(Q3X_ENABLE_SM87_AOT_SYSTEM_V1_BMMA_LOWER_BOUND)
#error "The real-P40 active-cell witness requires the private BMMA lower bound"
#endif

#include "sm87_aot_system_v1_bmma_lower_bound_internal.h"
#include "src/runtime/reference_runner_aot_arithmetic_witness_internal.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace q3x::test::sm87_aot_real_p40_active_cell {

namespace lower_bound = q3x::test::sm87_aot_bmma_lower_bound;
namespace witness_hook = q3x::runtime::reference_runner_detail;

inline constexpr std::size_t kRealP40SegmentCount = 79U;
inline constexpr std::size_t kRealP40CallsPerLayer = 4U;
inline constexpr std::size_t kRealP40CallsPerSegment =
    q3x::runtime::kQwen36DenseLayerCount * kRealP40CallsPerLayer;
inline constexpr std::size_t kPinnedActivationSlotBytes = 10U * 1024U * 1024U;
inline constexpr const char* kActivationCaptureChainDomain =
    "q3x.sm87.aot.real-p40.activation-support-mask-chain.v2";
inline constexpr const char* kActivationCallDomain =
    "q3x.sm87.aot.real-p40.activation-support-mask-call.v2";
inline constexpr const char* kActivationSourceDtype = "BF16";
inline constexpr const char* kActivationPayloadDtype = "UINT16_LE";

// cudaErrorNotReady is a stable, non-sticky CUDA status: it cannot poison the
// stream or device, and the enclosing runner already treats every nonzero hook
// result as a fail-closed abort.  Callers distinguish this expected stop from
// an actual CUDA failure through early_reject_requested().
inline constexpr cudaError_t kEarlyRejectCudaStatus = cudaErrorNotReady;

struct CollectorIdentity final {
  lower_bound::Sha256Digest real_p40_route_sha256{};
  lower_bound::Sha256Digest authenticated_shard_manifest_sha256{};
  bool real_p40_route_authenticated = false;
  bool shard_manifest_authenticated = false;
};

enum class CollectorFailureCode : std::uint8_t {
  kNone = 0U,
  kHostAllocation,
  kCudaAllocation,
  kInvalidIdentity,
  kProtocolOrder,
  kInvalidOperandView,
  kWeightInventoryMismatch,
  kCudaMaskLaunch,
  kCudaCopy,
  kCudaSynchronize,
  kArithmeticOverflow,
  kDigestFailure,
  kLowerBoundIssuerFailure,
  kIncompleteSegment,
  kAlreadyFinalized,
  kExceptionalOperand = 15U,
};

struct RoleSummary final {
  lower_bound::ProjectionRole role = lower_bound::ProjectionRole::kInvalid;
  std::uint64_t observed_geometric_joint_k16_cells = 0U;
  std::uint64_t proven_active_joint_k16_cells = 0U;
  lower_bound::Sha256Digest activation_inventory_sha256{};
  lower_bound::Sha256Digest joint_enumeration_sha256{};
  std::uint64_t activation_call_count = 0U;
  std::uint64_t activation_mask_element_count = 0U;
  std::uint64_t activation_payload_bytes = 0U;
  std::uint32_t weight_partition_count = 0U;
  bool weight_inventory_complete = false;
  bool activation_inventory_authenticated = false;
  bool joint_enumeration_authenticated = false;
};

struct CollectorResult final {
  // Empty only if no lower-bound receipt could be issued (for example, host
  // allocation failed before an implementation object existed).
  std::optional<lower_bound::DecisionReceipt> lower_bound_receipt;
  std::uint32_t covered_prompt_rows = 0U;
  std::uint32_t completed_segments = 0U;
  CollectorFailureCode failure_code = CollectorFailureCode::kNone;
  int cuda_status = static_cast<int>(cudaSuccess);
  std::array<char, 256U> failure_text{};
  std::array<RoleSummary, lower_bound::kProjectionRoleCount> roles{};
  lower_bound::Sha256Digest checkpoint_manifest_sha256{};
  bool checkpoint_inventory_complete = false;
  bool checkpoint_manifest_authenticated = false;
  bool early_reject_requested = false;
  // The exact device value copied at the last completed segment boundary.
  // A nonzero value is fail-closed and excludes that segment from all counts
  // and receipt issuance.
  std::uint32_t exceptional_flag_value = 0U;
  bool exceptional_operand_detected = false;
};

// Collects a strict, monotone lower bound from the ordinary Legacy-C512
// Prefix hook.  It observes only positions [0,39999); the production scalar
// at position 39999 is deliberately charged as free.  All host/device buffers
// are allocated by construction.  The callback performs no host or device
// allocation and never mutates the observed activation or checkpoint weights.
class RealP40ActiveCellWitnessCollector final {
 public:
  explicit RealP40ActiveCellWitnessCollector(
      CollectorIdentity identity) noexcept;
  ~RealP40ActiveCellWitnessCollector();

  RealP40ActiveCellWitnessCollector(
      const RealP40ActiveCellWitnessCollector&) = delete;
  RealP40ActiveCellWitnessCollector& operator=(
      const RealP40ActiveCellWitnessCollector&) = delete;
  RealP40ActiveCellWitnessCollector(
      RealP40ActiveCellWitnessCollector&&) = delete;
  RealP40ActiveCellWitnessCollector& operator=(
      RealP40ActiveCellWitnessCollector&&) = delete;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool early_reject_requested() const noexcept;
  [[nodiscard]] static constexpr cudaError_t early_reject_sentinel() noexcept {
    return kEarlyRejectCudaStatus;
  }

  [[nodiscard]] witness_hook::AotArithmeticWitnessAOperandHook hook() noexcept;

  // Finalize never returns PASS: the private issuer has only INCONCLUSIVE and
  // REJECT.  A partial in-flight segment is reported and excluded; every count
  // and digest in the result covers complete segments only.
  [[nodiscard]] CollectorResult finalize() noexcept;

  [[nodiscard]] static int a_operand_callback(
      const witness_hook::AotArithmeticWitnessAOperandView& view,
      void* context) noexcept;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

// Pure host protocol/geometry checks.  This function performs no CUDA call and
// does not construct a collector; numerical mask tests belong in a CUDA driver.
[[nodiscard]] bool run_real_p40_active_cell_protocol_self_test() noexcept;

}  // namespace q3x::test::sm87_aot_real_p40_active_cell
