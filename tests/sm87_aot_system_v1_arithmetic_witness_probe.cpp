#include "sm87_aot_system_v1_arithmetic_witness_internal.h"
#include "sm87_aot_system_v1_bmma_lower_bound_internal.h"
#include "sm87_aot_system_v1_checkpoint_reader_internal.h"

#include <iostream>
#include <string_view>

namespace witness = q3x::test::sm87_aot_arithmetic_witness;
namespace bmma_lower_bound = q3x::test::sm87_aot_bmma_lower_bound;
namespace checkpoint_reader = q3x::test::sm87_aot_checkpoint_reader;

namespace {

void write_description() {
  std::uint64_t joint_cells = 0U;
  for (std::size_t index = 0U;
       index < bmma_lower_bound::kProjectionRoleCount; ++index) {
    joint_cells += bmma_lower_bound::frozen_expected_joint_k16_cells(
        static_cast<bmma_lower_bound::ProjectionRole>(index + 1U));
  }
  const auto& mapping = bmma_lower_bound::frozen_mapping_spec();
  const std::uint64_t absolute_issue_rate =
      static_cast<std::uint64_t>(
          mapping.maximum_warp_instructions_per_sm_cycle) *
      mapping.sm_count * mapping.clock_hz;
  std::cout
      << "{\"schema\":\"q3x.sm87.aot-system-v1.arithmetic-witness.v1\","
      << "\"candidate\":\""
      << q3x::runtime::kSm87AotPrefillSystemCandidateId << "\","
      << "\"prompt_tokens\":" << witness::kP40Tokens << ','
      << "\"projection_budget_seconds\":"
      << witness::kProjectionBudgetSeconds << ','
      << "\"logical_projection_ops\":"
      << witness::total_logical_projection_ops() << ','
      << "\"k16_joint_mapping_cells\":" << joint_cells << ','
      << "\"bmma_absolute_warp_instructions_per_second\":"
      << absolute_issue_rate << ','
      << "\"bmma_five_second_absolute_instruction_capacity\":"
      << absolute_issue_rate * mapping.projection_budget_seconds << ','
      << "\"required_roles\":[";
  for (std::size_t index = 0U; index < witness::kOperandRoleCount; ++index) {
    const auto role = static_cast<witness::OperandRole>(index + 1U);
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << "{\"role\":\"" << witness::to_string(role) << "\","
              << "\"instances\":"
              << witness::expected_role_instances(role) << ','
              << "\"input_features\":"
              << witness::expected_input_features(role) << '}';
  }
  std::cout
      << "],\"decision\":\"INCONCLUSIVE\","
      << "\"reasons\":[\"real_p40000_operand_capture_not_supplied\","
      << "\"joint_activation_weight_partition_pass_receipts_not_supplied\","
      << "\"exact_ordered_partial_scale_rejoin_and_bf16_oracle_not_supplied\","
      << "\"same_sm87_actual_instruction_cost_envelope_not_supplied\"],"
      << "\"real_operand_authority\":false,"
      << "\"cuda_or_model_run\":false}\n";
}

}  // namespace

int main(const int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
    const bool witness_passed = witness::run_self_test();
    const bool reader_passed =
        checkpoint_reader::run_checkpoint_reader_self_test();
    const bool bmma_lower_bound_passed =
        bmma_lower_bound::run_bmma_lower_bound_self_test();
    const bool passed =
        witness_passed && reader_passed && bmma_lower_bound_passed;
    std::cout
        << "{\"schema\":\"q3x.sm87.aot-system-v1.arithmetic-witness.self-test.v1\","
        << "\"candidate\":\""
        << q3x::runtime::kSm87AotPrefillSystemCandidateId << "\","
        << "\"self_test\":\"" << (passed ? "pass" : "fail") << "\","
        << "\"witness_self_test\":\""
        << (witness_passed ? "pass" : "fail") << "\","
        << "\"checkpoint_reader_self_test\":\""
        << (reader_passed ? "pass" : "fail") << "\","
        << "\"bmma_lower_bound_self_test\":\""
        << (bmma_lower_bound_passed ? "pass" : "fail") << "\","
        << "\"real_operand_authority\":false,"
        << "\"cuda_or_model_run\":false}\n";
    return passed ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--describe") {
    write_description();
    return 0;
  }
  std::cerr << "usage: " << argv[0] << " --self-test|--describe\n";
  return 2;
}
