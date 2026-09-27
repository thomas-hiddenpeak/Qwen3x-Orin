#include "q3x/io/json.h"
#include "q3x/server/openai_protocol.h"
#include <iostream>
int main() {
  q3x::server::TargetPrefillWitnessRecord record;
  record.prompt_tokens = 40000;
  record.consumed_prompt_tokens = 40000;
  record.completion_tokens = 256;
  record.prefill_route_evidence.valid = true;
  record.prefill_route_evidence.complete = true;
  record.prefill_route_evidence.operators[0].exact_fallback_hits = 16;
  const auto text = q3x::server::serialize_target_prefill_witness(record);
  if (!q3x::io::json::parse(text) ||
#if defined(Q3X_ENABLE_WHOLE_CORE_EXACT_DECODE_ADMISSION)
      text.find("target-prefill-witness-whole-core-exact-decode-admission-v9") == std::string::npos ||
      text.find("\"schema_version\":2") == std::string::npos ||
#else
      text.find("target-prefill-witness-fused-decode-admission-v7") == std::string::npos ||
      text.find("\"schema_version\":7") == std::string::npos ||
#endif
      text.find("ordered-pipeline-unqualified") == std::string::npos ||
      text.find("\"completed_fallback_hits_unqualified\":16") == std::string::npos ||
      text.find("completed_exact_fallback_hits") != std::string::npos ||
      text.find("\"scope\":\"architecture_candidate_unqualified\"") == std::string::npos ||
      text.find("\"approximate_numerics\":false") == std::string::npos) {
    std::cerr << "Admission witness falsely qualified a numerical fallback\n";
    return 1;
  }
#if defined(Q3X_ENABLE_WHOLE_CORE_EXACT_DECODE_ADMISSION)
  namespace rt = q3x::runtime;
  record = {};
  record.prompt_tokens = record.consumed_prompt_tokens = 8192;
  record.full_prompt_consumed = true;
  record.completion_tokens = 16;
  record.deployment_plan_id = rt::kLayerMajorNativePromptWideP40WholeCoreDeploymentPlanId;
  record.prefill_execution_mode = rt::ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
  record.prefill_logical_panel_count = 2;
  record.request_memory_profile = rt::RequestMemoryProfile::kLayerMajorP40WholeCore;
  record.mlp_schedule_tactic = rt::LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore;
  record.bounded_submission_window = true;
  record.submission_window_retirements = 384;
  record.route_layer_pass_count = 1;
  auto& route = record.prefill_route_evidence;
  route.valid = route.complete = true;
  route.completed_layer_passes = route.expected_layer_passes = 1;
  for (std::size_t i = 0; i < route.operators.size(); ++i)
    route.operators[i].production_hits = rt::kExpectedPrefillLogicalOperatorsPerTile[i];
  record.prompt_wide_p40_whole_core_layer_hits = 64;
  record.prompt_wide_p40_fill_panel_hits = record.prompt_wide_p40_drain_panel_hits = 128;
  record.prompt_wide_p40_prompt_core_hits = 64;
  record.prompt_wide_p40_fp8_projection_hits = record.prompt_wide_p40_fp8_projection_physical_launches = 416;
  record.prompt_wide_p40_bf16_ab_hits = record.prompt_wide_p40_gdn_hits = 48;
  record.native_flashinfer_exact_whole_prompt_hits = 16;
  record.layer_wide_p40_mlp_layer_hits = record.persistent_p40_nvfp4_gate_up_hits = record.persistent_p40_nvfp4_down_residual_hits = 64;
  record.persistent_p40_nvfp4_physical_launches = 128;
  auto variable = q3x::server::serialize_target_prefill_witness(record);
  if (variable.find("\"package_complete\":true") == std::string::npos ||
      variable.find("\"expected_fp8_projection_hits\":416") == std::string::npos) return 1;
  record.prompt_tokens = record.consumed_prompt_tokens = 8193;
  record.persistent_p40_nvfp4_physical_launches = 192;
  variable = q3x::server::serialize_target_prefill_witness(record);
  if (variable.find("\"package_complete\":true") == std::string::npos) return 1;
  ++record.prompt_wide_p40_fill_panel_hits;
  variable = q3x::server::serialize_target_prefill_witness(record);
  if (variable.find("\"package_complete\":false") == std::string::npos) return 1;
#endif
  std::cout << "PASS unqualified fused witness preserves counts without exactness claims\n";
}
