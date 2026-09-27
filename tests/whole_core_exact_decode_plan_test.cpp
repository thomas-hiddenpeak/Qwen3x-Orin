#include "q3x/runtime/prefill_workspace_plan.h"
#include "q3x/runtime/request_state.h"
#include "q3x/server/evaluation_server.h"
#include <iostream>

int main() {
  namespace rt = q3x::runtime;
  namespace server = q3x::server;
  const auto workspace = rt::build_unbound_layer_major_p40_whole_core_workspace_plan();
  rt::LayerMajorRequestMemoryOptions options;
  options.max_sequence_length = 44'095U;
  options.max_arena_bytes = 8'952'211'200ULL;
  options.layout = rt::LayerMajorRequestLayout::kP40WholeCorePromptWide;
  options.mlp_layout = rt::LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan;
  const auto request = rt::build_layer_major_request_memory_plan(options);
  if (!workspace || !request) {
    std::cerr << "composition plan rejected: "
              << (request ? "workspace" : request.diagnostic.message) << '\n';
    return 1;
  }
  const auto& plan = *request.value;
  if (workspace.value->required_bytes != plan.common.arena_bytes ||
      plan.common.arena_bytes != options.max_arena_bytes ||
      plan.common.fp32_scratch.element_capacity < 24ULL * 44'095U ||
      plan.p40_whole_core.request_capacity_tokens != 44'095U ||
      plan.prompt_residual_bf16.row_capacity != 44'095U ||
      plan.p40_whole_core.prompt_token_count != 40'000U) return 1;
  --options.max_arena_bytes;
  if (rt::build_layer_major_request_memory_plan(options)) return 1;

  const auto& admission = server::selected_p40_whole_core_plan();
  if (admission.request_arena_bytes != plan.common.arena_bytes ||
      admission.maximum_output_tokens != 4096U ||
      admission.max_sequence_length != 44095U ||
      admission.min_free_bytes_after_create != 8ULL * 1024U * 1024U * 1024U ||
      admission.decode_gate_up_layers != 64U ||
      admission.decode_down_consumer_order_layers != 53U ||
      admission.prefill_supermatrix_projections != 0U ||
      server::kP40WholeCoreV1ProductionPlan.max_sequence_length != 40016U)
    return 1;
  server::OpenAIRequest input;
  input.endpoint = server::OpenAIEndpoint::kCompletions;
  input.prompt_kind = server::OpenAIPromptKind::kTokenIds;
  input.prompt_token_ids.resize(40000U, 1U);
  input.stream = true;
  input.include_usage = true;
  for (const auto output : {1U, 16U, 256U, 4096U}) {
    input.max_tokens = output;
    if (!server::is_p40_whole_core_v10_request(input)) return 1;
  }
  input.max_tokens = 4097U;
  if (server::is_p40_whole_core_v10_request(input)) return 1;
  input.max_tokens = 0U;
  if (server::is_p40_whole_core_v10_request(input)) return 1;
  input.max_tokens = 1U;
  input.prompt_token_ids.pop_back();
  if (server::is_p40_whole_core_v10_request(input)) return 1;
  input.prompt_token_ids.resize(40001U, 1U);
  if (server::is_p40_whole_core_v10_request(input)) return 1;
  input.prompt_token_ids.resize(40000U);
  input.stream = false;
  if (server::is_p40_whole_core_v10_request(input)) return 1;
  input.stream = true;
  input.include_usage = false;
  if (server::is_p40_whole_core_v10_request(input)) return 1;
  std::cout << "PASS composition arena, full Decode scratch, output boundary and fixed-profile isolation\n";
}
