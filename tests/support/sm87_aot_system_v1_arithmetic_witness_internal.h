#pragma once

#if !defined(Q3X_ENABLE_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS)
#error "The SM87 AOT arithmetic witness is a private, default-off test admission"
#endif

#include "q3x/core/sha256.h"
#include "q3x/runtime/sm87_aot_prefill_system_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace q3x::test::sm87_aot_arithmetic_witness {

inline constexpr std::size_t kP40Tokens = 40'000U;
inline constexpr std::size_t kPanelRows = 128U;
inline constexpr std::size_t kMmaRows = 16U;
inline constexpr std::size_t kK16 = 16U;
inline constexpr std::size_t kK64 = 64U;
inline constexpr double kProjectionBudgetSeconds = 5.0;

inline constexpr std::size_t kExponentSpanHistogramSize = 256U;
inline constexpr std::size_t kAlignedBitSpanHistogramSize = 262U;
inline constexpr std::size_t kInt8LimbHistogramSize = 39U;
inline constexpr std::size_t kInt4LimbHistogramSize = 88U;
inline constexpr std::size_t kTrailingZeroHistogramSize = 8U;

enum class OperandRole : std::uint8_t {
  kInvalid = 0U,
  kNvFp4GateUp,
  kNvFp4Down,
  kFp8GdnQkvZ,
  kFp8FullQkv,
  kFp8AttentionOutput,
  kCount,
};

inline constexpr std::size_t kOperandRoleCount =
    static_cast<std::size_t>(OperandRole::kCount) - 1U;

enum class Decision : std::uint8_t {
  kInconclusive = 0U,
  kReject,
  kPass,
};

enum class ArithmeticClass : std::uint8_t {
  kInvalid = 0U,
  kK16SignedInt8Limbs,
  kK16SignedInt4Limbs,
  kK16ExactBitPlanes,
  kK64SignedInt8Limbs,
  kK64SignedInt4Limbs,
  kK64ExactBitPlanes,
};

enum class OperandCapturePoint : std::uint8_t {
  kInvalid = 0U,
  kGateUpInputAfterPostAttentionNorm,
  kDownInputAfterGatedActivation,
  kGdnQkvZInputAfterInputNorm,
  kFullQkvInputAfterInputNorm,
  kAttentionOutputInputAfterPinnedCore,
};

enum class CostScope : std::uint8_t {
  kInvalid = 0U,
  kOrderedBf16Ab,
  kProjectionRole,
};

enum WitnessIssue : std::uint32_t {
  kWitnessIssueNone = 0U,
  kWitnessIssueInvalidInput = 1U << 0U,
  kWitnessIssueMissingP40Coverage = 1U << 1U,
  kWitnessIssueUnauthenticatedCapture = 1U << 2U,
  kWitnessIssueMissingExactnessOracle = 1U << 3U,
  kWitnessIssueExactnessViolation = 1U << 4U,
  kWitnessIssueMissingCostEnvelope = 1U << 5U,
  kWitnessIssueUnchargedFallback = 1U << 6U,
  kWitnessIssueUnchargedRepresentation = 1U << 7U,
  kWitnessIssueArithmeticOverflow = 1U << 8U,
  kWitnessIssueOptimisticBudgetExceeded = 1U << 9U,
  kWitnessIssueConservativeBudgetNotClosed = 1U << 10U,
  kWitnessIssueMissingJointPassReceipt = 1U << 11U,
  kWitnessIssueDuplicateOrWrongLayer = 1U << 12U,
  kWitnessIssueMissingAuthoritativeEvidenceReader = 1U << 13U,
};

struct OperandInstanceIdentity {
  OperandRole role = OperandRole::kInvalid;
  OperandCapturePoint capture_point = OperandCapturePoint::kInvalid;
  std::size_t model_layer_index =
      runtime::kSm87AotPrefillSystemLayerCount;
  bool checkpoint_identity_authenticated = false;
  bool route_identity_authenticated = false;
  bool capture_boundary_authenticated = false;
};

struct CellDomainSummary {
  // K16 summaries use one incumbent MMA parent A cell (M16xK16). K64
  // summaries use one proposed physical load cell (up to M128xK64 at the
  // final M tail). Every cell aligns finite BF16 significands to its least
  // represented power of two. Limb histograms record the exact minimum
  // number of radix-2^7 signed INT8 or radix-2^3 signed INT4 digits needed by
  // the worst value in that aligned cell. Special-value cells are counted in
  // special_cell_count and intentionally excluded from both limb histograms.
  std::uint64_t cell_count = 0U;
  std::uint64_t value_count = 0U;
  std::uint64_t finite_nonzero_count = 0U;
  std::uint64_t zero_count = 0U;
  std::uint64_t negative_zero_count = 0U;
  std::uint64_t subnormal_count = 0U;
  std::uint64_t infinity_count = 0U;
  std::uint64_t nan_count = 0U;
  std::uint64_t special_cell_count = 0U;
  std::uint32_t maximum_exponent_span = 0U;
  std::uint32_t maximum_aligned_bit_span = 0U;
  std::uint32_t maximum_int8_limb_count = 0U;
  std::uint32_t maximum_int4_limb_count = 0U;
  std::uint64_t residual_plane_set_bits = 0U;
  std::uint64_t residual_plane_slots = 0U;
  // For every row, each consecutive K4 group is examined independently on
  // every occupied aligned magnitude plane. The numerator counts planes with
  // at most two nonzero lanes; empty planes are excluded from both counts.
  // This is only an exact bit-plane encoding statistic. It is not valid as a
  // signed-limb 2:4 statistic because carries can occupy different K4 lanes.
  std::uint64_t exact_bit_plane_two_of_four_groups = 0U;
  std::uint64_t occupied_bit_plane_two_of_four_groups = 0U;
  std::array<std::uint64_t, kExponentSpanHistogramSize>
      exponent_span_histogram{};
  std::array<std::uint64_t, kAlignedBitSpanHistogramSize>
      aligned_bit_span_histogram{};
  std::array<std::uint64_t, kInt8LimbHistogramSize>
      int8_limb_histogram{};
  std::array<std::uint64_t, kInt4LimbHistogramSize>
      int4_limb_histogram{};
  std::array<std::uint64_t, kTrailingZeroHistogramSize>
      significand_trailing_zero_histogram{};
};

struct OperandDomainSummary {
  OperandRole role = OperandRole::kInvalid;
  OperandCapturePoint capture_point = OperandCapturePoint::kInvalid;
  std::size_t expected_rows = 0U;
  std::size_t input_features = 0U;
  std::size_t next_row = 0U;
  std::size_t panel_count = 0U;
  std::size_t instance_count = 0U;
  std::uint64_t model_layer_mask = 0U;
  std::uint64_t layer_payload_digest_mask = 0U;
  std::array<std::array<std::uint8_t, 32U>,
             runtime::kSm87AotPrefillSystemLayerCount>
      layer_payload_sha256{};
  std::array<std::uint8_t, 32U> payload_inventory_sha256{};
  std::array<std::uint8_t, 32U> analysis_inventory_sha256{};
  CellDomainSummary k16{};
  CellDomainSummary k64{};
  bool capture_identity_authenticated = false;
  bool payload_digest_present = false;
  bool payload_inventory_digest_present = false;
  bool analysis_inventory_digest_present = false;
  bool instance_complete = false;
  // Means every expected model layer for this role is present for the
  // self-described rows/columns.  It is not a P40000 qualification claim;
  // the evaluator separately requires the exact P40 geometry.
  bool layer_inventory_complete = false;
  bool valid = false;
};

class Bf16OperandAccumulator {
 public:
  Bf16OperandAccumulator(OperandInstanceIdentity identity,
                         std::size_t expected_rows,
                         std::size_t input_features) noexcept;

  // `bits` must remain stable and exclusively readable until this call
  // returns.  A future online joint enumerator must be invoked from this same
  // call path before the source buffer can be reused; a second, independently
  // scheduled consumer cannot establish same-byte-stream evidence.
  [[nodiscard]] bool consume_panel(std::size_t first_row,
                                   const std::uint16_t* bits,
                                   std::size_t row_count,
                                   std::size_t row_stride_elements) noexcept;
  [[nodiscard]] OperandDomainSummary finalize() noexcept;
  [[nodiscard]] bool failed() const noexcept { return failed_; }

 private:
  OperandDomainSummary summary_{};
  core::Sha256 payload_hasher_{};
  std::uint64_t expected_logical_value_count_ = 0U;
  std::uint64_t hashed_logical_value_count_ = 0U;
  bool source_identity_authenticated_ = false;
  bool payload_hash_initialized_ = false;
  bool failed_ = false;
  bool finalized_ = false;
};

class OperandDomainMerger {
 public:
  [[nodiscard]] bool add(const OperandDomainSummary& instance) noexcept;
  [[nodiscard]] OperandDomainSummary finalize() noexcept;
  [[nodiscard]] bool failed() const noexcept { return failed_; }

 private:
  OperandDomainSummary summary_{};
  bool started_ = false;
  bool failed_ = false;
  bool finalized_ = false;
};

struct JointPassReceipt {
  // This is a joint activation x checkpoint-weight enumeration.  It must be
  // accumulated at the real (role, partition, selected K16/K64 alignment
  // cell, N-tile) granularity; a product of independent activation and weight
  // averages is inadmissible. The mapping digest also binds limb versus
  // bit-plane encoding, sign/carry metadata, fallback, and rejoin order.
  OperandRole role = OperandRole::kInvalid;
  ArithmeticClass arithmetic_class = ArithmeticClass::kInvalid;
  std::uint64_t source_alignment_cells = 0U;
  std::uint64_t direct_alignment_cells = 0U;
  std::uint64_t fallback_alignment_cells = 0U;
  std::uint64_t exact_direct_physical_ops = 0U;
  std::size_t partition_count = 0U;
  std::array<std::uint8_t, 32U> arithmetic_mapping_sha256{};
  std::array<std::uint8_t, 32U> checkpoint_manifest_sha256{};
  std::array<std::uint8_t, 32U> pass_receipt_sha256{};
  bool activation_weight_joint_enumeration_complete = false;
  bool checkpoint_weights_authenticated = false;
  bool partition_inventory_complete = false;
  bool encoding_metadata_complete = false;
  bool pass_accounting_authenticated = false;
};

struct RoleMapping {
  OperandRole role = OperandRole::kInvalid;
  ArithmeticClass arithmetic_class = ArithmeticClass::kInvalid;
  OperandDomainSummary operand{};
  JointPassReceipt joint_pass{};
};

struct ExactnessReceipt {
  std::array<std::uint8_t, 32U> oracle_receipt_sha256{};
  std::size_t compared_projection_instances = 0U;
  std::uint64_t special_value_test_cases = 0U;
  std::uint64_t special_value_mismatches = 0U;
  std::uint64_t signed_zero_test_cases = 0U;
  std::uint64_t signed_zero_mismatches = 0U;
  std::uint64_t ordered_fp32_partial_checks = 0U;
  std::uint64_t ordered_fp32_partial_mismatches = 0U;
  std::uint64_t scale_rejoin_checks = 0U;
  std::uint64_t scale_rejoin_mismatches = 0U;
  std::uint64_t bf16_publication_checks = 0U;
  std::uint64_t bf16_publication_mismatches = 0U;
  bool oracle_identity_authenticated = false;
};

struct CostEnvelope {
  CostScope scope = CostScope::kInvalid;
  OperandRole role = OperandRole::kInvalid;
  ArithmeticClass arithmetic_class = ArithmeticClass::kInvalid;
  std::array<std::uint8_t, 32U> arithmetic_mapping_sha256{};
  std::array<std::uint8_t, 32U> measurement_receipt_sha256{};
  bool measurement_identity_authenticated = false;
  bool calibrated_same_sm87 = false;
  std::size_t measured_sample_count = 0U;
  std::uint64_t charged_fallback_cells = 0U;
  std::size_t representation_component_count = 0U;
  std::size_t charged_representation_component_count = 0U;
  double optimistic_throughput_ops_per_second = 0.0;
  double conservative_throughput_ops_per_second = 0.0;
  // The optimistic throughput ceiling yields a lower time bound and can only
  // reject. The conservative measured floor yields an upper time bound and is
  // required to pass. Both must describe the actual proposed instruction
  // class on the same SM87 device; nominal peak rates are inadmissible.
  double fallback_seconds_lower = 0.0;
  double fallback_seconds_upper = 0.0;
  double representation_seconds_lower = 0.0;
  double representation_seconds_upper = 0.0;
};

struct Assessment {
  std::array<RoleMapping, kOperandRoleCount> roles{};
  ExactnessReceipt exactness{};
  CostEnvelope ordered_bf16_cost{};
  std::array<CostEnvelope, kOperandRoleCount> role_costs{};
};

struct DecisionReceipt {
  Decision decision = Decision::kInconclusive;
  std::uint32_t issues = kWitnessIssueNone;
  std::uint64_t logical_projection_ops = 0U;
  std::uint64_t direct_physical_ops = 0U;
  std::uint64_t ordered_bf16_physical_ops = 0U;
  std::array<std::uint64_t, kOperandRoleCount> role_source_alignment_cells{};
  std::array<std::uint64_t, kOperandRoleCount> role_direct_alignment_cells{};
  std::array<std::uint64_t, kOperandRoleCount> role_fallback_alignment_cells{};
  std::array<std::uint64_t, kOperandRoleCount> role_physical_ops{};
  double ordered_bf16_optimistic_seconds_lower = 0.0;
  double ordered_bf16_conservative_seconds_upper = 0.0;
  std::array<double, kOperandRoleCount> role_optimistic_seconds_lower{};
  std::array<double, kOperandRoleCount> role_conservative_seconds_upper{};
  // Diagnostic only. Fallback and representation work are charged as
  // measured seconds below and are deliberately not disguised as the same
  // instruction-class pass count.
  double average_direct_mma_pass_count_excluding_fallback = 0.0;
  double optimistic_seconds_lower = 0.0;
  double conservative_seconds_upper = 0.0;
};

[[nodiscard]] constexpr std::size_t role_index(OperandRole role) noexcept {
  return static_cast<std::size_t>(role) - 1U;
}

[[nodiscard]] std::string_view to_string(OperandRole role) noexcept;
[[nodiscard]] std::string_view to_string(Decision decision) noexcept;
[[nodiscard]] std::string_view to_string(ArithmeticClass value) noexcept;
[[nodiscard]] std::string_view to_string(OperandCapturePoint value) noexcept;
[[nodiscard]] std::string_view to_string(CostScope value) noexcept;

[[nodiscard]] std::size_t expected_role_instances(OperandRole role) noexcept;
[[nodiscard]] std::size_t expected_input_features(OperandRole role) noexcept;
[[nodiscard]] OperandCapturePoint expected_capture_point(
    OperandRole role) noexcept;
[[nodiscard]] std::uint64_t expected_model_layer_mask(
    OperandRole role) noexcept;
[[nodiscard]] std::uint64_t role_logical_projection_ops(
    OperandRole role) noexcept;
[[nodiscard]] std::uint64_t total_logical_projection_ops() noexcept;

// The first witness slice deliberately has no real operand reader or receipt
// issuer.  This public entry point is therefore structurally unable to PASS
// or REJECT: it always reports INCONCLUSIVE with
// kWitnessIssueMissingAuthoritativeEvidenceReader.  A later slice must add a
// private issuer that derives every identity and accounting field from the
// authenticated P40 capture before exposing an authoritative decision path.
[[nodiscard]] DecisionReceipt evaluate(const Assessment& assessment) noexcept;

[[nodiscard]] bool run_self_test() noexcept;

}  // namespace q3x::test::sm87_aot_arithmetic_witness
