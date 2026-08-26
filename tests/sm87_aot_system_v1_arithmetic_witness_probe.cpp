#include "sm87_aot_system_v1_arithmetic_witness_internal.h"

#include <iostream>
#include <string_view>

namespace witness = q3x::test::sm87_aot_arithmetic_witness;

namespace {

void write_description() {
  std::cout
      << "{\"schema\":\"q3x.sm87.aot-system-v1.arithmetic-witness.v1\","
      << "\"candidate\":\""
      << q3x::runtime::kSm87AotPrefillSystemCandidateId << "\","
      << "\"prompt_tokens\":" << witness::kP40Tokens << ','
      << "\"projection_budget_seconds\":"
      << witness::kProjectionBudgetSeconds << ','
      << "\"logical_projection_ops\":"
      << witness::total_logical_projection_ops() << ','
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
    const bool passed = witness::run_self_test();
    std::cout
        << "{\"schema\":\"q3x.sm87.aot-system-v1.arithmetic-witness.self-test.v1\","
        << "\"candidate\":\""
        << q3x::runtime::kSm87AotPrefillSystemCandidateId << "\","
        << "\"self_test\":\"" << (passed ? "pass" : "fail") << "\","
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
