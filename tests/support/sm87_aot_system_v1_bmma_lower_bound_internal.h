#pragma once

#if !defined(Q3X_ENABLE_SM87_AOT_SYSTEM_V1_BMMA_LOWER_BOUND)
#error "The SM87 AOT BMMA lower bound is a private, default-off test admission"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace q3x::test::sm87_aot_bmma_lower_bound {

inline constexpr std::size_t kProjectionRoleCount = 5U;
inline constexpr std::size_t kSha256Bytes = 32U;

using Sha256Digest = std::array<std::uint8_t, kSha256Bytes>;

enum class ProjectionRole : std::uint8_t {
  kInvalid = 0U,
  kNvFp4GateUp,
  kNvFp4Down,
  kFp8GdnQkvZ,
  kFp8FullQkv,
  kFp8AttentionOutput,
  kCount,
};

enum class Decision : std::uint8_t {
  kInconclusive = 0U,
  kReject,
};

enum LowerBoundIssue : std::uint32_t {
  kLowerBoundIssueNone = 0U,
  kLowerBoundIssueInvalidInput = 1U << 0U,
  kLowerBoundIssueDuplicateRole = 1U << 1U,
  kLowerBoundIssueMissingRole = 1U << 2U,
  kLowerBoundIssueUnexpectedP40CellCount = 1U << 3U,
  kLowerBoundIssueIncompleteP40CellCoverage = 1U << 4U,
  kLowerBoundIssueUnauthenticatedEvidence = 1U << 5U,
  kLowerBoundIssueArithmeticOverflow = 1U << 6U,
  kLowerBoundIssueFiveSecondBudgetNotExceeded = 1U << 7U,
  kLowerBoundIssueAlreadyFinalized = 1U << 8U,
  kLowerBoundIssuePartialP40Coverage = 1U << 9U,
};

// Frozen optimistic *static-schedule* mapping. One support-active
// M16xN8xK16 parent is executed by one K256 binary MMA warp instruction with
// K[16, 256) set to zero. The active predicate and the forbidden elisions are
// part of the mapping identity: this is not a lower bound for a different
// exact implementation that proves an ordered parent result is zero and skips
// it, shares it, or packs it with another parent. Every sign/exponent/limb,
// residual-plane, and fallback pass after the mandatory first binary pass is
// optimistically charged as free. Therefore one admitted active joint K16
// cell costs exactly one instruction in this narrow class; the 16x zero-fill
// loss must not be multiplied into an already-K16-granular count again. Any
// incomplete M16 tail is excluded from the count rather than forcing a tail
// implementation into this class.
struct MappingSpec final {
  std::string_view identity;
  std::string_view instruction;
  std::string_view support_active_parent_predicate;
  std::string_view exact_arithmetic_pass_floor;
  std::uint32_t mma_m;
  std::uint32_t mma_n;
  std::uint32_t bmma_k;
  std::uint32_t parent_k;
  std::uint32_t zero_fill_factor;
  std::uint32_t physical_instructions_per_active_joint_k16_cell;
  std::uint32_t maximum_warp_instructions_per_sm_cycle;
  std::uint32_t sm_count;
  std::uint64_t clock_hz;
  std::uint32_t projection_budget_seconds;
  std::uint32_t p40_prompt_rows;
  std::uint32_t production_prefix_rows;
  std::uint32_t terminal_scalar_rows;
  bool partial_m16_tail_rows_charged_free;
  bool one_parent_per_instruction;
  bool cross_parent_packing_forbidden;
  bool support_active_parent_must_issue;
  bool result_aware_parent_elision_forbidden;
  bool cross_parent_common_subexpression_elimination_forbidden;
  bool additional_exactness_passes_charged_free;
};

// These are claims supplied by the real-P40 joint activation x checkpoint-
// weight enumerator. The lower-bound issuer validates and copies them; it does
// not let a caller supply any derived receipt, issue, timing, or decision
// field.
struct RoleEvidence final {
  ProjectionRole role = ProjectionRole::kInvalid;
  std::uint64_t expected_joint_k16_cells = 0U;
  std::uint64_t observed_joint_k16_cells = 0U;
  std::uint64_t active_joint_k16_cells = 0U;
  std::uint32_t covered_prompt_rows = 0U;

  Sha256Digest real_p40_route_sha256{};
  Sha256Digest activation_inventory_sha256{};
  Sha256Digest checkpoint_manifest_sha256{};
  Sha256Digest joint_enumeration_sha256{};
  bool real_p40_route_authenticated = false;
  bool activation_inventory_authenticated = false;
  bool checkpoint_manifest_authenticated = false;
  bool joint_enumeration_authenticated = false;
  bool covered_rows_are_contiguous_prefix = false;
  bool observed_subset_complete = false;
};

struct RoleLowerBound final {
  ProjectionRole role = ProjectionRole::kInvalid;
  std::uint64_t expected_joint_k16_cells = 0U;
  std::uint64_t reported_expected_joint_k16_cells = 0U;
  std::uint64_t observed_geometric_joint_k16_cells = 0U;
  std::uint64_t proven_active_joint_k16_cells = 0U;
  std::uint32_t covered_prompt_rows = 0U;
  std::uint64_t proven_warp_instruction_lower_bound = 0U;
  double optimistic_seconds_lower_bound = 0.0;
  bool processed_subset_complete = false;
  bool complete_real_p40_coverage = false;
  bool evidence_authenticated = false;
};

class DecisionReceipt final {
 public:
  DecisionReceipt(const DecisionReceipt&) = default;
  DecisionReceipt& operator=(const DecisionReceipt&) = default;
  DecisionReceipt(DecisionReceipt&&) = default;
  DecisionReceipt& operator=(DecisionReceipt&&) = default;

  [[nodiscard]] Decision decision() const noexcept { return decision_; }
  [[nodiscard]] std::uint32_t issues() const noexcept { return issues_; }
  [[nodiscard]] bool privately_issued() const noexcept {
    return privately_issued_;
  }
  [[nodiscard]] bool all_expected_real_p40_cells_complete() const noexcept {
    return all_expected_real_p40_cells_complete_;
  }
  [[nodiscard]] bool all_processed_subsets_complete() const noexcept {
    return all_processed_subsets_complete_;
  }
  [[nodiscard]] bool all_processed_evidence_authenticated() const noexcept {
    return all_processed_evidence_authenticated_;
  }
  [[nodiscard]] const std::array<RoleLowerBound, kProjectionRoleCount>& roles()
      const noexcept {
    return roles_;
  }
  [[nodiscard]] std::uint64_t expected_joint_k16_cells() const noexcept {
    return expected_joint_k16_cells_;
  }
  [[nodiscard]] std::uint64_t reported_expected_joint_k16_cells()
      const noexcept {
    return reported_expected_joint_k16_cells_;
  }
  [[nodiscard]] std::uint64_t observed_geometric_joint_k16_cells()
      const noexcept {
    return observed_geometric_joint_k16_cells_;
  }
  [[nodiscard]] std::uint64_t proven_active_joint_k16_cells() const noexcept {
    return proven_active_joint_k16_cells_;
  }
  [[nodiscard]] const std::array<std::uint32_t, kProjectionRoleCount>&
  covered_prompt_rows_by_role() const noexcept {
    return covered_prompt_rows_by_role_;
  }
  [[nodiscard]] std::uint64_t proven_warp_instruction_lower_bound()
      const noexcept {
    return proven_warp_instruction_lower_bound_;
  }
  [[nodiscard]] std::uint64_t absolute_warp_instructions_per_second()
      const noexcept {
    return absolute_warp_instructions_per_second_;
  }
  [[nodiscard]] std::uint64_t five_second_absolute_instruction_capacity()
      const noexcept {
    return five_second_absolute_instruction_capacity_;
  }
  [[nodiscard]] double optimistic_seconds_lower_bound() const noexcept {
    return optimistic_seconds_lower_bound_;
  }

 private:
  friend class LowerBoundIssuer;
  DecisionReceipt() = default;

  Decision decision_ = Decision::kInconclusive;
  std::uint32_t issues_ = kLowerBoundIssueNone;
  bool privately_issued_ = false;
  bool all_expected_real_p40_cells_complete_ = false;
  bool all_processed_subsets_complete_ = false;
  bool all_processed_evidence_authenticated_ = false;
  std::array<RoleLowerBound, kProjectionRoleCount> roles_{};
  std::uint64_t expected_joint_k16_cells_ = 0U;
  std::uint64_t reported_expected_joint_k16_cells_ = 0U;
  std::uint64_t observed_geometric_joint_k16_cells_ = 0U;
  std::uint64_t proven_active_joint_k16_cells_ = 0U;
  std::array<std::uint32_t, kProjectionRoleCount>
      covered_prompt_rows_by_role_{};
  std::uint64_t proven_warp_instruction_lower_bound_ = 0U;
  std::uint64_t absolute_warp_instructions_per_second_ = 0U;
  std::uint64_t five_second_absolute_instruction_capacity_ = 0U;
  double optimistic_seconds_lower_bound_ = 0.0;
};

// This is the only receipt issuer. It owns the five-role inventory privately,
// derives every result field during finalize(), and can be finalized once.
// Missing roles, uncovered rows, and unobserved cells contribute exactly zero
// to the monotone lower bound; they do not prevent REJECT when the complete,
// authenticated processed subset alone is already over budget. RoleEvidence
// contains no caller-writable issuance, instruction, timing, or decision
// field.
class LowerBoundIssuer final {
 public:
  LowerBoundIssuer() = default;
  LowerBoundIssuer(const LowerBoundIssuer&) = delete;
  LowerBoundIssuer& operator=(const LowerBoundIssuer&) = delete;
  LowerBoundIssuer(LowerBoundIssuer&&) = delete;
  LowerBoundIssuer& operator=(LowerBoundIssuer&&) = delete;

  [[nodiscard]] bool add(RoleEvidence evidence) noexcept;
  [[nodiscard]] DecisionReceipt finalize() noexcept;
  [[nodiscard]] bool failed() const noexcept { return failed_; }

 private:
  std::array<RoleEvidence, kProjectionRoleCount> evidence_{};
  std::array<bool, kProjectionRoleCount> seen_{};
  std::uint32_t issues_ = kLowerBoundIssueNone;
  bool failed_ = false;
  bool finalized_ = false;
};

[[nodiscard]] const MappingSpec& frozen_mapping_spec() noexcept;
[[nodiscard]] std::uint64_t frozen_expected_joint_k16_cells(
    ProjectionRole role) noexcept;
[[nodiscard]] std::uint64_t frozen_geometric_joint_k16_cells_for_prefix_rows(
    ProjectionRole role, std::uint32_t covered_prompt_rows) noexcept;
[[nodiscard]] std::string_view to_string(ProjectionRole role) noexcept;
[[nodiscard]] std::string_view to_string(Decision decision) noexcept;

// Host-only adversarial contract test. It covers a one-cell coverage hole,
// a zero authentication digest, checked-integer overflow, and both sides of
// the strict five-second rejection threshold.
[[nodiscard]] bool run_bmma_lower_bound_self_test() noexcept;

}  // namespace q3x::test::sm87_aot_bmma_lower_bound
