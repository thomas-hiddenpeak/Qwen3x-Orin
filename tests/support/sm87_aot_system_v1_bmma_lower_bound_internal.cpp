#include "sm87_aot_system_v1_bmma_lower_bound_internal.h"

#include <algorithm>
#include <limits>

namespace q3x::test::sm87_aot_bmma_lower_bound {
namespace {

constexpr MappingSpec kFrozenMappingSpec{
    "q3x.sm87-aot-system-v1/bmma-static-support-k16-parent-zero-fill/v2",
    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc",
    "a_m16_k16_support_mask_eq_0xffff_and_b_n8_k16_support_mask_ne_0",
    "static-support-exact-signed-limb-bitplane/v1;"
    "one-mandatory-binary-pass-floor;"
    "all-additional-sign-exponent-limb-bitplane-residual-fallback-passes-free",
    16U,
    8U,
    256U,
    16U,
    16U,
    1U,
    4U,
    16U,
    1'300'500'000ULL,
    5U,
    40'000U,
    39'999U,
    1U,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
};

// M16 x N8 x K16 joint cells over the five frozen real-P40 projection roles.
// The formulas are evaluated here only to make their dimensional origin
// reviewable; the resulting values are also frozen with static assertions.
constexpr std::uint64_t kP40M16Tiles = 40'000ULL / 16ULL;
constexpr std::array<std::uint64_t, kProjectionRoleCount>
    kFrozenExpectedJointK16Cells{{
        64ULL * kP40M16Tiles * (5'120ULL / 16ULL) * (34'816ULL / 8ULL),
        64ULL * kP40M16Tiles * (17'408ULL / 16ULL) * (5'120ULL / 8ULL),
        48ULL * kP40M16Tiles * (5'120ULL / 16ULL) * (16'384ULL / 8ULL),
        16ULL * kP40M16Tiles * (5'120ULL / 16ULL) * (14'336ULL / 8ULL),
        64ULL * kP40M16Tiles * (6'144ULL / 16ULL) * (5'120ULL / 8ULL),
    }};

static_assert(kFrozenMappingSpec.bmma_k /
                  kFrozenMappingSpec.parent_k ==
              kFrozenMappingSpec.zero_fill_factor);
static_assert(
    kFrozenMappingSpec.physical_instructions_per_active_joint_k16_cell == 1U);
static_assert(kFrozenMappingSpec.one_parent_per_instruction &&
              kFrozenMappingSpec.partial_m16_tail_rows_charged_free &&
              kFrozenMappingSpec.cross_parent_packing_forbidden &&
              kFrozenMappingSpec.support_active_parent_must_issue &&
              kFrozenMappingSpec.result_aware_parent_elision_forbidden &&
              kFrozenMappingSpec
                  .cross_parent_common_subexpression_elimination_forbidden &&
              kFrozenMappingSpec.additional_exactness_passes_charged_free);
static_assert(kFrozenExpectedJointK16Cells[0U] == 222'822'400'000ULL);
static_assert(kFrozenExpectedJointK16Cells[1U] == 111'411'200'000ULL);
static_assert(kFrozenExpectedJointK16Cells[2U] == 78'643'200'000ULL);
static_assert(kFrozenExpectedJointK16Cells[3U] == 22'937'600'000ULL);
static_assert(kFrozenExpectedJointK16Cells[4U] == 39'321'600'000ULL);
static_assert(kFrozenExpectedJointK16Cells[0U] +
                      kFrozenExpectedJointK16Cells[1U] +
                      kFrozenExpectedJointK16Cells[2U] +
                      kFrozenExpectedJointK16Cells[3U] +
                      kFrozenExpectedJointK16Cells[4U] ==
                  475'136'000'000ULL,
              "The frozen five-role real-P40 K16 inventory changed");
static_assert(kFrozenMappingSpec.production_prefix_rows +
                  kFrozenMappingSpec.terminal_scalar_rows ==
              kFrozenMappingSpec.p40_prompt_rows);
static_assert(
    static_cast<std::uint64_t>(
        kFrozenMappingSpec.maximum_warp_instructions_per_sm_cycle) *
            kFrozenMappingSpec.sm_count * kFrozenMappingSpec.clock_hz ==
        83'232'000'000ULL,
    "The frozen SM87 1.3005 GHz absolute issue ceiling changed");
static_assert(
    static_cast<std::uint64_t>(
        kFrozenMappingSpec.maximum_warp_instructions_per_sm_cycle) *
            kFrozenMappingSpec.sm_count * kFrozenMappingSpec.clock_hz *
            kFrozenMappingSpec.projection_budget_seconds ==
        416'160'000'000ULL,
    "The strict five-second absolute BMMA capacity changed");

constexpr std::array<std::uint64_t, kProjectionRoleCount>
    kJointK16CellsPerM16Tile{{
        64ULL * (5'120ULL / 16ULL) * (34'816ULL / 8ULL),
        64ULL * (17'408ULL / 16ULL) * (5'120ULL / 8ULL),
        48ULL * (5'120ULL / 16ULL) * (16'384ULL / 8ULL),
        16ULL * (5'120ULL / 16ULL) * (14'336ULL / 8ULL),
        64ULL * (6'144ULL / 16ULL) * (5'120ULL / 8ULL),
    }};

[[nodiscard]] constexpr bool valid_role(const ProjectionRole role) noexcept {
  return role > ProjectionRole::kInvalid && role < ProjectionRole::kCount;
}

[[nodiscard]] constexpr std::size_t role_index(
    const ProjectionRole role) noexcept {
  return static_cast<std::size_t>(role) - 1U;
}

[[nodiscard]] bool checked_add(std::uint64_t& target,
                               const std::uint64_t value) noexcept {
  if (value > std::numeric_limits<std::uint64_t>::max() - target) {
    return false;
  }
  target += value;
  return true;
}

[[nodiscard]] bool checked_multiply(const std::uint64_t left,
                                    const std::uint64_t right,
                                    std::uint64_t& result) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] bool digest_is_nonzero(const Sha256Digest& digest) noexcept {
  return std::any_of(digest.begin(), digest.end(),
                     [](const std::uint8_t value) { return value != 0U; });
}

[[nodiscard]] bool evidence_is_authenticated(
    const RoleEvidence& evidence) noexcept {
  return evidence.real_p40_route_authenticated &&
         evidence.activation_inventory_authenticated &&
         evidence.checkpoint_manifest_authenticated &&
         evidence.joint_enumeration_authenticated &&
         digest_is_nonzero(evidence.real_p40_route_sha256) &&
         digest_is_nonzero(evidence.activation_inventory_sha256) &&
         digest_is_nonzero(evidence.checkpoint_manifest_sha256) &&
         digest_is_nonzero(evidence.joint_enumeration_sha256);
}

[[nodiscard]] constexpr ProjectionRole role_at(
    const std::size_t index) noexcept {
  return static_cast<ProjectionRole>(index + 1U);
}

[[nodiscard]] RoleEvidence complete_evidence(
    const ProjectionRole role, const std::uint64_t active_cells,
    const std::uint32_t covered_prompt_rows = 40'000U) noexcept {
  RoleEvidence result;
  result.role = role;
  result.expected_joint_k16_cells =
      frozen_expected_joint_k16_cells(role);
  result.observed_joint_k16_cells =
      frozen_geometric_joint_k16_cells_for_prefix_rows(
          role, covered_prompt_rows);
  result.active_joint_k16_cells = active_cells;
  result.covered_prompt_rows = covered_prompt_rows;
  const std::uint8_t digest_byte =
      static_cast<std::uint8_t>(role_index(role) + 1U);
  result.real_p40_route_sha256.fill(0xa5U);
  result.activation_inventory_sha256.fill(digest_byte);
  result.checkpoint_manifest_sha256.fill(0x3cU);
  result.joint_enumeration_sha256.fill(
      static_cast<std::uint8_t>(digest_byte + 13U));
  result.real_p40_route_authenticated = true;
  result.activation_inventory_authenticated = true;
  result.checkpoint_manifest_authenticated = true;
  result.joint_enumeration_authenticated = true;
  result.covered_rows_are_contiguous_prefix = true;
  result.observed_subset_complete = true;
  return result;
}

[[nodiscard]] bool add_complete_inventory_with_active_total(
    LowerBoundIssuer& issuer, std::uint64_t remaining_active,
    const std::uint32_t covered_prompt_rows = 40'000U) noexcept {
  for (std::size_t index = 0U; index < kProjectionRoleCount; ++index) {
    const ProjectionRole role = role_at(index);
    const std::uint64_t observed =
        frozen_geometric_joint_k16_cells_for_prefix_rows(
            role, covered_prompt_rows);
    const std::uint64_t active = std::min(observed, remaining_active);
    if (!issuer.add(complete_evidence(role, active, covered_prompt_rows))) {
      return false;
    }
    remaining_active -= active;
  }
  return remaining_active == 0U;
}

}  // namespace

const MappingSpec& frozen_mapping_spec() noexcept {
  return kFrozenMappingSpec;
}

std::uint64_t frozen_expected_joint_k16_cells(
    const ProjectionRole role) noexcept {
  return valid_role(role) ? kFrozenExpectedJointK16Cells[role_index(role)]
                          : 0U;
}

std::uint64_t frozen_geometric_joint_k16_cells_for_prefix_rows(
    const ProjectionRole role,
    const std::uint32_t covered_prompt_rows) noexcept {
  if (!valid_role(role) ||
      covered_prompt_rows > kFrozenMappingSpec.p40_prompt_rows) {
    return 0U;
  }
  // Any incomplete M16 tail is deliberately outside the strict count. This
  // keeps the lower bound valid for an exact implementation with a different
  // tail path and charges the final scalar's containing tile entirely free.
  const std::uint64_t m16_tiles =
      static_cast<std::uint64_t>(covered_prompt_rows) / 16ULL;
  std::uint64_t result = 0U;
  return checked_multiply(kJointK16CellsPerM16Tile[role_index(role)],
                          m16_tiles, result)
             ? result
             : 0U;
}

std::string_view to_string(const ProjectionRole role) noexcept {
  switch (role) {
    case ProjectionRole::kNvFp4GateUp:
      return "nvfp4_gate_up";
    case ProjectionRole::kNvFp4Down:
      return "nvfp4_down";
    case ProjectionRole::kFp8GdnQkvZ:
      return "fp8_gdn_qkvz";
    case ProjectionRole::kFp8FullQkv:
      return "fp8_full_qkv";
    case ProjectionRole::kFp8AttentionOutput:
      return "fp8_attention_output";
    case ProjectionRole::kInvalid:
    case ProjectionRole::kCount:
      return "invalid";
  }
  return "invalid";
}

std::string_view to_string(const Decision decision) noexcept {
  switch (decision) {
    case Decision::kInconclusive:
      return "INCONCLUSIVE";
    case Decision::kReject:
      return "REJECT";
  }
  return "INCONCLUSIVE";
}

bool LowerBoundIssuer::add(RoleEvidence evidence) noexcept {
  if (finalized_) {
    issues_ |= kLowerBoundIssueAlreadyFinalized;
    failed_ = true;
    return false;
  }
  if (!valid_role(evidence.role)) {
    issues_ |= kLowerBoundIssueInvalidInput;
    failed_ = true;
    return false;
  }
  const std::size_t index = role_index(evidence.role);
  if (seen_[index]) {
    issues_ |= kLowerBoundIssueDuplicateRole;
    failed_ = true;
    return false;
  }
  evidence_[index] = evidence;
  seen_[index] = true;
  return true;
}

DecisionReceipt LowerBoundIssuer::finalize() noexcept {
  DecisionReceipt receipt;
  receipt.privately_issued_ = true;
  receipt.issues_ = issues_;

  if (finalized_) {
    receipt.issues_ |= kLowerBoundIssueAlreadyFinalized;
    receipt.decision_ = Decision::kInconclusive;
    return receipt;
  }
  finalized_ = true;

  bool arithmetic_ok = true;
  arithmetic_ok =
      checked_multiply(
          kFrozenMappingSpec.maximum_warp_instructions_per_sm_cycle,
          kFrozenMappingSpec.sm_count,
          receipt.absolute_warp_instructions_per_second_) &&
      checked_multiply(receipt.absolute_warp_instructions_per_second_,
                       kFrozenMappingSpec.clock_hz,
                       receipt.absolute_warp_instructions_per_second_);
  arithmetic_ok =
      arithmetic_ok &&
      checked_multiply(receipt.absolute_warp_instructions_per_second_,
                       kFrozenMappingSpec.projection_budget_seconds,
                       receipt.five_second_absolute_instruction_capacity_);

  bool all_complete = true;
  bool all_processed_subsets_complete = true;
  bool all_authenticated = true;
  bool identity_started = false;
  Sha256Digest route_identity{};
  Sha256Digest checkpoint_identity{};
  for (std::size_t index = 0U; index < kProjectionRoleCount; ++index) {
    const ProjectionRole role = role_at(index);
    const std::uint64_t frozen_expected =
        kFrozenExpectedJointK16Cells[index];
    auto& output = receipt.roles_[index];
    output.role = role;
    output.expected_joint_k16_cells = frozen_expected;

    if (!checked_add(receipt.expected_joint_k16_cells_,
                     frozen_expected)) {
      arithmetic_ok = false;
    }
    if (!seen_[index]) {
      receipt.issues_ |= kLowerBoundIssueMissingRole;
      receipt.issues_ |= kLowerBoundIssuePartialP40Coverage;
      all_complete = false;
      continue;
    }

    const RoleEvidence& input = evidence_[index];
    output.reported_expected_joint_k16_cells =
        input.expected_joint_k16_cells;
    output.observed_geometric_joint_k16_cells =
        input.observed_joint_k16_cells;
    output.proven_active_joint_k16_cells = input.active_joint_k16_cells;
    output.covered_prompt_rows = input.covered_prompt_rows;
    receipt.covered_prompt_rows_by_role_[index] =
        input.covered_prompt_rows;

    const bool expected_matches =
        input.expected_joint_k16_cells == frozen_expected;
    const std::uint64_t expected_observed_subset =
        frozen_geometric_joint_k16_cells_for_prefix_rows(
            role, input.covered_prompt_rows);
    const bool ordered_counts =
        input.active_joint_k16_cells <= input.observed_joint_k16_cells &&
        input.observed_joint_k16_cells <= input.expected_joint_k16_cells;
    output.processed_subset_complete =
        expected_matches && ordered_counts &&
        input.covered_prompt_rows <= kFrozenMappingSpec.p40_prompt_rows &&
        input.covered_rows_are_contiguous_prefix &&
        input.observed_subset_complete &&
        input.observed_joint_k16_cells == expected_observed_subset;
    output.complete_real_p40_coverage =
        output.processed_subset_complete &&
        input.covered_prompt_rows == kFrozenMappingSpec.p40_prompt_rows &&
        input.observed_joint_k16_cells == frozen_expected;
    bool identity_coherent = true;
    if (!identity_started) {
      route_identity = input.real_p40_route_sha256;
      checkpoint_identity = input.checkpoint_manifest_sha256;
      identity_started = true;
    } else {
      identity_coherent =
          input.real_p40_route_sha256 == route_identity &&
          input.checkpoint_manifest_sha256 == checkpoint_identity;
    }
    output.evidence_authenticated =
        identity_coherent && evidence_is_authenticated(input);

    if (!expected_matches) {
      receipt.issues_ |= kLowerBoundIssueUnexpectedP40CellCount;
    }
    if (!ordered_counts) {
      receipt.issues_ |= kLowerBoundIssueInvalidInput;
    }
    if (!output.processed_subset_complete) {
      receipt.issues_ |= kLowerBoundIssueIncompleteP40CellCoverage;
      all_processed_subsets_complete = false;
    }
    if (!output.complete_real_p40_coverage) {
      receipt.issues_ |= kLowerBoundIssuePartialP40Coverage;
      all_complete = false;
    }
    if (!output.evidence_authenticated) {
      receipt.issues_ |= kLowerBoundIssueUnauthenticatedEvidence;
      all_authenticated = false;
    }

    if (!checked_multiply(
            input.active_joint_k16_cells,
            kFrozenMappingSpec
                .physical_instructions_per_active_joint_k16_cell,
            output.proven_warp_instruction_lower_bound) ||
        !checked_add(receipt.reported_expected_joint_k16_cells_,
                     input.expected_joint_k16_cells) ||
        !checked_add(receipt.observed_geometric_joint_k16_cells_,
                     input.observed_joint_k16_cells) ||
        !checked_add(receipt.proven_active_joint_k16_cells_,
                     input.active_joint_k16_cells) ||
        !checked_add(receipt.proven_warp_instruction_lower_bound_,
                     output.proven_warp_instruction_lower_bound)) {
      arithmetic_ok = false;
    }
    if (receipt.absolute_warp_instructions_per_second_ != 0U) {
      output.optimistic_seconds_lower_bound =
          static_cast<double>(output.proven_warp_instruction_lower_bound) /
          static_cast<double>(
              receipt.absolute_warp_instructions_per_second_);
    }
  }

  receipt.all_expected_real_p40_cells_complete_ = all_complete;
  receipt.all_processed_subsets_complete_ = all_processed_subsets_complete;
  receipt.all_processed_evidence_authenticated_ = all_authenticated;
  if (!arithmetic_ok) {
    receipt.issues_ |= kLowerBoundIssueArithmeticOverflow;
  }
  if (receipt.absolute_warp_instructions_per_second_ != 0U) {
    receipt.optimistic_seconds_lower_bound_ =
        static_cast<double>(receipt.proven_warp_instruction_lower_bound_) /
        static_cast<double>(receipt.absolute_warp_instructions_per_second_);
  }

  // The comparison is intentionally integral and strict. Floating-point
  // formatting of exactly 5.0 seconds cannot turn an inconclusive boundary
  // into REJECT.
  const bool reject =
      arithmetic_ok && !failed_ && all_processed_subsets_complete &&
      all_authenticated && receipt.proven_warp_instruction_lower_bound_ >
          receipt.five_second_absolute_instruction_capacity_;
  receipt.decision_ = reject ? Decision::kReject : Decision::kInconclusive;
  if (arithmetic_ok && all_processed_subsets_complete && all_authenticated &&
      receipt.proven_warp_instruction_lower_bound_ <=
          receipt.five_second_absolute_instruction_capacity_) {
    receipt.issues_ |= kLowerBoundIssueFiveSecondBudgetNotExceeded;
  }
  return receipt;
}

bool run_bmma_lower_bound_self_test() noexcept {
  bool ok = true;
  const auto expect = [&ok](const bool condition) { ok = ok && condition; };

  std::uint64_t frozen_total = 0U;
  for (const std::uint64_t cells : kFrozenExpectedJointK16Cells) {
    expect(checked_add(frozen_total, cells));
  }
  expect(kFrozenMappingSpec.bmma_k == 256U &&
         kFrozenMappingSpec.parent_k == 16U &&
         kFrozenMappingSpec.zero_fill_factor == 16U &&
         !kFrozenMappingSpec.support_active_parent_predicate.empty() &&
         !kFrozenMappingSpec.exact_arithmetic_pass_floor.empty() &&
         kFrozenMappingSpec.partial_m16_tail_rows_charged_free &&
         kFrozenMappingSpec.one_parent_per_instruction &&
         kFrozenMappingSpec.cross_parent_packing_forbidden &&
         kFrozenMappingSpec.support_active_parent_must_issue &&
         kFrozenMappingSpec.result_aware_parent_elision_forbidden &&
         kFrozenMappingSpec
             .cross_parent_common_subexpression_elimination_forbidden &&
         kFrozenMappingSpec.additional_exactness_passes_charged_free &&
         frozen_total == 475'136'000'000ULL);

  std::uint64_t absolute_rate = 0U;
  std::uint64_t threshold = 0U;
  expect(checked_multiply(
             kFrozenMappingSpec.maximum_warp_instructions_per_sm_cycle,
             kFrozenMappingSpec.sm_count, absolute_rate) &&
         checked_multiply(absolute_rate, kFrozenMappingSpec.clock_hz,
                          absolute_rate) &&
         checked_multiply(absolute_rate,
                          kFrozenMappingSpec.projection_budget_seconds,
                          threshold) &&
         absolute_rate == 83'232'000'000ULL &&
         threshold == 416'160'000'000ULL);

  {
    LowerBoundIssuer issuer;
    expect(add_complete_inventory_with_active_total(issuer, threshold));
    const DecisionReceipt receipt = issuer.finalize();
    expect(receipt.privately_issued() &&
           receipt.decision() == Decision::kInconclusive &&
           receipt.proven_warp_instruction_lower_bound() == threshold &&
           receipt.optimistic_seconds_lower_bound() == 5.0 &&
           (receipt.issues() &
            kLowerBoundIssueFiveSecondBudgetNotExceeded) != 0U);
  }

  {
    LowerBoundIssuer issuer;
    expect(add_complete_inventory_with_active_total(issuer, threshold + 1U));
    const DecisionReceipt receipt = issuer.finalize();
    expect(receipt.decision() == Decision::kReject &&
           receipt.all_expected_real_p40_cells_complete() &&
           receipt.all_processed_evidence_authenticated() &&
           receipt.proven_warp_instruction_lower_bound() == threshold + 1U &&
           receipt.optimistic_seconds_lower_bound() > 5.0);
  }

  // A single unobserved joint cell must prevent rejection even though the
  // remaining fully-active inventory still exceeds five seconds.
  {
    LowerBoundIssuer issuer;
    for (std::size_t index = 0U; index < kProjectionRoleCount; ++index) {
      const ProjectionRole role = role_at(index);
      RoleEvidence evidence = complete_evidence(
          role, frozen_expected_joint_k16_cells(role));
      if (index == kProjectionRoleCount - 1U) {
        --evidence.observed_joint_k16_cells;
        --evidence.active_joint_k16_cells;
      }
      expect(issuer.add(evidence));
    }
    const DecisionReceipt receipt = issuer.finalize();
    expect(receipt.decision() == Decision::kInconclusive &&
           !receipt.all_expected_real_p40_cells_complete() &&
           (receipt.issues() &
            kLowerBoundIssueIncompleteP40CellCoverage) != 0U);
  }

  // The production hook can authenticate only the 39,999-row prefix. Its
  // active cells may still form a strict monotone lower bound that rejects;
  // the terminal scalar remains uncovered and is charged as zero/free.
  {
    LowerBoundIssuer issuer;
    expect(add_complete_inventory_with_active_total(
        issuer, threshold + 1U, kFrozenMappingSpec.production_prefix_rows));
    const DecisionReceipt receipt = issuer.finalize();
    expect(receipt.decision() == Decision::kReject &&
           receipt.all_processed_subsets_complete() &&
           !receipt.all_expected_real_p40_cells_complete() &&
           (receipt.issues() & kLowerBoundIssuePartialP40Coverage) != 0U);
    for (const RoleLowerBound& role : receipt.roles()) {
      expect(role.covered_prompt_rows ==
                 kFrozenMappingSpec.production_prefix_rows &&
             role.processed_subset_complete &&
             !role.complete_real_p40_coverage);
    }
  }

  // The fifth role is not required to issue a monotone rejection. Its entire
  // inventory is deliberately free here; four authenticated role prefixes
  // already prove more physical instructions than the absolute capacity.
  {
    LowerBoundIssuer issuer;
    std::uint64_t remaining_active = threshold + 1U;
    for (std::size_t index = 0U; index < 4U; ++index) {
      const ProjectionRole role = role_at(index);
      const std::uint64_t observed =
          frozen_geometric_joint_k16_cells_for_prefix_rows(
              role, kFrozenMappingSpec.production_prefix_rows);
      const std::uint64_t active = std::min(observed, remaining_active);
      expect(issuer.add(complete_evidence(
          role, active, kFrozenMappingSpec.production_prefix_rows)));
      remaining_active -= active;
    }
    const DecisionReceipt receipt = issuer.finalize();
    expect(remaining_active == 0U && receipt.decision() == Decision::kReject &&
           receipt.all_processed_subsets_complete() &&
           (receipt.issues() & kLowerBoundIssueMissingRole) != 0U);
  }

  {
    LowerBoundIssuer issuer;
    for (std::size_t index = 0U; index < kProjectionRoleCount; ++index) {
      const ProjectionRole role = role_at(index);
      RoleEvidence evidence = complete_evidence(
          role, frozen_expected_joint_k16_cells(role));
      if (index == 2U) {
        evidence.joint_enumeration_sha256.fill(0U);
      }
      expect(issuer.add(evidence));
    }
    const DecisionReceipt receipt = issuer.finalize();
    expect(receipt.decision() == Decision::kInconclusive &&
           !receipt.all_processed_evidence_authenticated() &&
           (receipt.issues() &
            kLowerBoundIssueUnauthenticatedEvidence) != 0U);
  }

  {
    LowerBoundIssuer issuer;
    const RoleEvidence evidence = complete_evidence(
        ProjectionRole::kNvFp4GateUp,
        frozen_expected_joint_k16_cells(ProjectionRole::kNvFp4GateUp));
    expect(issuer.add(evidence));
    expect(!issuer.add(evidence));
    const DecisionReceipt receipt = issuer.finalize();
    expect(receipt.decision() == Decision::kInconclusive && issuer.failed() &&
           (receipt.issues() & kLowerBoundIssueDuplicateRole) != 0U);
  }

  // Drive the public accumulator through a real overflow, not merely a
  // maximum-value boundary. The cell geometry is intentionally invalid too,
  // but overflow must be independently and observably classified.
  {
    LowerBoundIssuer issuer;
    for (std::size_t index = 0U; index < kProjectionRoleCount; ++index) {
      RoleEvidence evidence = complete_evidence(role_at(index), 0U);
      evidence.expected_joint_k16_cells =
          std::numeric_limits<std::uint64_t>::max();
      evidence.observed_joint_k16_cells =
          std::numeric_limits<std::uint64_t>::max();
      evidence.active_joint_k16_cells =
          std::numeric_limits<std::uint64_t>::max();
      expect(issuer.add(evidence));
    }
    const DecisionReceipt receipt = issuer.finalize();
    expect(receipt.decision() == Decision::kInconclusive &&
           (receipt.issues() & kLowerBoundIssueArithmeticOverflow) != 0U);
  }

  {
    std::uint64_t add_target = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t product = 0U;
    expect(!checked_add(add_target, 1U) &&
           !checked_multiply(std::numeric_limits<std::uint64_t>::max(), 2U,
                             product));
  }

  return ok;
}

}  // namespace q3x::test::sm87_aot_bmma_lower_bound
