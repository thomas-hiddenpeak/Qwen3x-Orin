#include "q3x/io/json.h"
#include "q3x/server/evaluation_server.h"
#include "q3x/server/openai_protocol.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace server = q3x::server;

class TestContext {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }
  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

[[nodiscard]] bool valid_json(const std::string_view text) {
  return static_cast<bool>(q3x::io::json::parse(text));
}

void test_evalscope_chat_contract(TestContext& test) {
  const auto parsed = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"system","content":"brief"},{"role":"user","content":"你好"}],"max_tokens":2048,"temperature":0.0,"top_p":1,"n":1,"seed":42,"stream":true,"stream_options":{"include_usage":true},"stop":null})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 4096U, 4096U);
  test.expect(parsed && parsed.value->model == "qwen3.6-27b-nvfp4" &&
                  parsed.value->messages.size() == 2U &&
                  parsed.value->messages[0].role == "system" &&
                  parsed.value->messages[1].content == "你好" &&
                  parsed.value->max_tokens == 2048U &&
                  parsed.value->seed == 42 &&
                  parsed.value->stream && parsed.value->include_usage,
              "EvalScope chat request maps to the strict greedy contract");
}

void test_raw_completion_contract(TestContext& test) {
  auto parsed = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","prompt":"raw prompt","max_tokens":3,"temperature":0,"stream":false})",
      server::OpenAIEndpoint::kCompletions, "qwen3.6-27b-nvfp4",
      32U, 4096U);
  test.expect(parsed &&
                  parsed.value->prompt_kind ==
                      server::OpenAIPromptKind::kRawText &&
                  parsed.value->prompt == "raw prompt" &&
                  !parsed.value->stream,
              "plain completions retain raw text without a chat template");

  parsed = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","prompt":[1,248046,42],"max_completion_tokens":2,"temperature":0,"stream":true,"stream_options":{"include_usage":false}})",
      server::OpenAIEndpoint::kCompletions, "qwen3.6-27b-nvfp4",
      32U, 4096U);
  test.expect(parsed &&
                  parsed.value->prompt_kind ==
                      server::OpenAIPromptKind::kTokenIds &&
                  parsed.value->prompt_token_ids.size() == 3U &&
                  parsed.value->prompt_token_ids[1] == 248046U &&
                  parsed.value->max_tokens == 2U &&
                  parsed.value->stream && !parsed.value->include_usage,
              "EvalScope tokenized prompt remains one flat exact id array");
}

void test_target_length_token_id_contract(TestContext& test) {
  constexpr std::size_t kPromptTokens = 130'000U;
  constexpr std::uint32_t kOutputTokens = 32U;
  constexpr std::uint32_t kAdmittedSequenceLength = 130'032U;

  std::string body =
      R"({"model":"qwen3.6-27b-nvfp4","prompt":[)";
  body.reserve(kPromptTokens * 2U + 128U);
  for (std::size_t index = 0U; index < kPromptTokens; ++index) {
    if (index != 0U) {
      body.push_back(',');
    }
    body.push_back('1');
  }
  body +=
      R"(],"max_tokens":32,"temperature":0,"stream":true})";

  const auto admitted = server::parse_openai_request(
      body, server::OpenAIEndpoint::kCompletions,
      "qwen3.6-27b-nvfp4", kOutputTokens,
      kAdmittedSequenceLength);
  test.expect(admitted &&
                  admitted.value->prompt_kind ==
                      server::OpenAIPromptKind::kTokenIds &&
                  admitted.value->prompt_token_ids.size() == kPromptTokens,
              "the bounded JSON protocol admits an exact 130K token-id "
              "witness");

  const auto over_capacity = server::parse_openai_request(
      body, server::OpenAIEndpoint::kCompletions,
      "qwen3.6-27b-nvfp4", kOutputTokens, 130'030U);
  test.expect(!over_capacity &&
                  over_capacity.error.code ==
                      "context_length_exceeded" &&
                  over_capacity.error.param == "prompt",
              "token-id capacity fails closed before inference enqueue");
}

void test_fail_closed_parameters(TestContext& test) {
  const auto unsupported = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0.7})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U, 4096U);
  test.expect(!unsupported &&
                  unsupported.error.code == "unsupported_sampling" &&
                  unsupported.error.param == "temperature",
              "non-greedy sampling fails closed");

  const auto tool = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"x"}],"tools":[]})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U, 4096U);
  test.expect(!tool && tool.error.code == "unsupported_parameter" &&
                  tool.error.param == "tools",
              "unimplemented tools are rejected rather than ignored");

  const auto wrong_model = server::parse_openai_request(
      R"({"model":"other","messages":[{"role":"user","content":"x"}]})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U, 4096U);
  test.expect(!wrong_model && wrong_model.error.http_status == 404 &&
                  wrong_model.error.code == "model_not_found",
              "unknown served model has a stable 404 error");

  const auto content_array = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":[{"type":"text","text":"x"}]}]})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U, 4096U);
  test.expect(!content_array &&
                  content_array.error.code == "unsupported_message",
              "multimodal content arrays fail closed");

  const auto fractional_seed = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0,"seed":1.5})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U, 4096U);
  test.expect(!fractional_seed &&
                  fractional_seed.error.code == "invalid_seed" &&
                  fractional_seed.error.param == "seed",
              "EvalScope seed compatibility remains strict about integer type");

  const auto implicit_sampling = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"x"}],"max_tokens":1})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U, 4096U);
  test.expect(!implicit_sampling &&
                  implicit_sampling.error.code == "missing_parameter" &&
                  implicit_sampling.error.param == "temperature",
              "the OpenAI sampling default is never silently treated as "
              "greedy");

  const auto implicit_output_cap = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"x"}],"temperature":0})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U, 4096U);
  test.expect(!implicit_output_cap &&
                  implicit_output_cap.error.code == "missing_parameter" &&
                  implicit_output_cap.error.param == "max_tokens",
              "the bounded output cap must be explicit");
}

void test_serialization(TestContext& test) {
  const server::OpenAIUsage usage{513U, 17U, 530U};
  const std::string completion = server::serialize_chat_completion(
      "chatcmpl-1", 1234, "qwen3.6-27b-nvfp4", "a\n\"中",
      server::OpenAIFinishReason::kStop, usage);
  test.expect(valid_json(completion) &&
                  completion.find("\\n\\\"中") != std::string::npos &&
                  completion.find("\"prompt_tokens\":513") !=
                      std::string::npos &&
                  completion.find("\"finish_reason\":\"stop\"") !=
                      std::string::npos,
              "non-stream completion is valid escaped JSON with exact usage");

  const std::string first = server::serialize_chat_chunk(
      "chatcmpl-1", 1234, "qwen3.6-27b-nvfp4", "a", true,
      std::nullopt);
  const std::string last = server::serialize_chat_chunk(
      "chatcmpl-1", 1234, "qwen3.6-27b-nvfp4", "中", false,
      server::OpenAIFinishReason::kStop);
  const std::string usage_chunk = server::serialize_usage_chunk(
      server::OpenAIEndpoint::kChatCompletions, "chatcmpl-1", 1234,
      "qwen3.6-27b-nvfp4", usage);
  test.expect(valid_json(first) && valid_json(last) &&
                  valid_json(usage_chunk) &&
                  first.find("\"role\":\"assistant\"") !=
                      std::string::npos &&
                  last.find("\"finish_reason\":\"stop\"") !=
                      std::string::npos &&
                  last.find("\"content\":\"中\"") !=
                      std::string::npos &&
                  usage_chunk.find("\"choices\":[]") !=
                      std::string::npos,
              "stream token and usage chunks preserve EvalScope timing shape");

  server::OpenAIProtocolError error;
  error.code = "unsupported_sampling";
  error.message = "only greedy";
  error.param = "temperature";
  test.expect(valid_json(server::serialize_openai_error(error)),
              "structured OpenAI errors are valid JSON");

  const std::string health =
      server::serialize_health_response("qwen\"model");
  test.expect(valid_json(health) &&
                  health.find("qwen\\\"model") != std::string::npos,
              "health response escapes the configured model alias");
}

void test_target_prefill_witness_evidence(TestContext& test) {
  const std::vector<std::uint32_t> token_ids{0U, 1U, 0x01020304U,
                                             248'046U};
  constexpr std::string_view kExpectedTokenHash =
      "6d4ae539080ab0cc26e32f3b9899fea8801c06d3797227da854a66f0b2271aa5";
  const std::string token_hash =
      server::sha256_token_ids_u32le(token_ids);
  test.expect(token_hash == kExpectedTokenHash,
              "token-id evidence hashes canonical u32 little-endian bytes");

  server::TargetPrefillWitnessRecord record;
  record.request_id = "cmpl-\"1";
  record.request_body_sha256 = "body-hash";
  record.model = "model";
  record.endpoint = server::OpenAIEndpoint::kCompletions;
  record.prompt_kind = server::OpenAIPromptKind::kTokenIds;
  record.prompt_tokens = token_ids.size();
  record.prompt_token_ids_u32le_sha256 = token_hash;
  record.consumed_prompt_tokens = token_ids.size();
  record.full_prompt_consumed = true;
  record.completion_tokens = 1U;
  record.queue = {"queue", 1.25, {}};
  record.admission = {"admission", std::nullopt, "not_instrumented_here"};
  record.generation = {"generation", 2.0, {}};
  record.pure_prefill = {"prefill", 3.0, {}};
  record.finalize = {"finalize", 0.5, {}};
  record.ttft = {"ttft", 4.0, {}};
  record.first_byte = {"first_byte", std::nullopt,
                       "socket_write_not_instrumented"};
  record.decode = {"decode", 0.0, {}};
  record.total = {"total", 5.0, {}};
  record.requested_prefill_chunk_size = 512U;
  record.effective_prefill_chunk_size = 256U;
  record.prefix_execution_count = 2U;
  q3x::runtime::PrefillRouteEvidence route_tile;
  for (std::size_t index = 0U;
       index < q3x::runtime::kPrefillOperatorRoleCount; ++index) {
    test.expect(q3x::runtime::record_prefill_operator_route(
                    route_tile,
                    static_cast<q3x::runtime::PrefillOperatorRole>(index),
                    q3x::runtime::PrefillRouteDisposition::kProduction,
                    q3x::runtime::
                        kExpectedPrefillLogicalOperatorsPerTile[index]),
                "route fixture records complete logical coverage");
  }
  q3x::runtime::reset_prefill_route_request(record.prefill_route_evidence);
  test.expect(q3x::runtime::commit_prefill_route_layer_pass(
                  record.prefill_route_evidence, route_tile) &&
                  q3x::runtime::commit_prefill_route_layer_pass(
                      record.prefill_route_evidence, route_tile),
              "route fixture commits two completed tiles");
  record.prefill_route_evidence =
      q3x::runtime::finalize_prefill_route_request(
          record.prefill_route_evidence, record.prefix_execution_count);
  record.projection_backend =
      q3x::runtime::ProjectionBackend::kReference;

  const std::string serialized =
      server::serialize_target_prefill_witness(record);
  const std::string expected =
      R"({"record":"target-prefill-witness-v1","schema_version":1,"request":{"id":"cmpl-\"1","body_sha256":"body-hash"},"model":"model","endpoint":"/v1/completions","prompt":{"kind":"token_ids","tokens":4,"token_ids_u32le_sha256":"6d4ae539080ab0cc26e32f3b9899fea8801c06d3797227da854a66f0b2271aa5","consumed_tokens":4,"fully_consumed":true},"completion":{"tokens":1},"timing":{"queue":{"available":true,"scope":"queue","milliseconds":1.250000,"unavailable_reason":null},"admission":{"available":false,"scope":"admission","milliseconds":null,"unavailable_reason":"not_instrumented_here"},"generation":{"available":true,"scope":"generation","milliseconds":2.000000,"unavailable_reason":null},"pure_prefill":{"available":true,"scope":"prefill","milliseconds":3.000000,"unavailable_reason":null},"finalize":{"available":true,"scope":"finalize","milliseconds":0.500000,"unavailable_reason":null},"ttft":{"available":true,"scope":"ttft","milliseconds":4.000000,"unavailable_reason":null},"first_byte":{"available":false,"scope":"first_byte","milliseconds":null,"unavailable_reason":"socket_write_not_instrumented"},"decode":{"available":true,"scope":"decode","milliseconds":0.000000,"unavailable_reason":null},"total":{"available":true,"scope":"total","milliseconds":5.000000,"unavailable_reason":null}},"prefill":{"requested_chunk":512,"effective_chunk":256,"prefix_execution_count":2},"route":{"scope":"request_witness","projection_backend":{"available":true,"scope":"configured_engine_fact","value":"reference"},"deployment_plan":{"available":false,"reason":"not_implemented"},"per_operator_route_hits":{"available":true,"scope":"request_completed_prefill_logical_operators","complete":true,"reason":null,"coverage":{"completed_layer_passes":2,"expected_layer_passes":2,"layers_per_pass":64,"gdn_layers_per_pass":48,"attention_layers_per_pass":16},"operators":{"nvfp4_gate_up":{"completed_production_hits":128,"completed_exact_fallback_hits":0,"completed_forbidden_hits":0},"nvfp4_down":{"completed_production_hits":128,"completed_exact_fallback_hits":0,"completed_forbidden_hits":0},"fp8_qkv":{"completed_production_hits":192,"completed_exact_fallback_hits":0,"completed_forbidden_hits":0},"fp8_z":{"completed_production_hits":96,"completed_exact_fallback_hits":0,"completed_forbidden_hits":0},"fp8_o":{"completed_production_hits":128,"completed_exact_fallback_hits":0,"completed_forbidden_hits":0},"attention":{"completed_production_hits":32,"completed_exact_fallback_hits":0,"completed_forbidden_hits":0},"gdn":{"completed_production_hits":96,"completed_exact_fallback_hits":0,"completed_forbidden_hits":0}},"forbidden_route_hits":{"prefix_cache":0,"mtp":0,"cublaslt":0,"external_reference":0,"approximate_numerics":0}},"cache_hits":{"available":false,"reason":"not_instrumented"},"disabled_boundaries":{"scope":"production_contract","prefix_cache":true,"mtp":true,"cublaslt_production":true,"approximate_numerics":true}}})";
  test.expect(serialized == expected,
              "request evidence serialization has a stable field contract");
  test.expect(valid_json(serialized),
              "request evidence serialization remains valid JSON");

  server::TargetPrefillWitnessRecord sealed_record = record;
  sealed_record.prefill_execution_mode = q3x::runtime::
      ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
  sealed_record.prefill_logical_panel_count = 2U;
  sealed_record.request_memory_profile =
      q3x::runtime::RequestMemoryProfile::kLayerMajorC8192;
  sealed_record.bounded_submission_window = true;
  sealed_record.submission_window_retirements = 128U;
  sealed_record.deployment_plan_id =
      q3x::runtime::kLayerMajorOperatorPanelDeploymentPlanId;
  const std::string sealed_serialized =
      server::serialize_target_prefill_witness(sealed_record);
  test.expect(
      valid_json(sealed_serialized) &&
          sealed_serialized.find(
              R"("record":"target-prefill-witness-v2","schema_version":2)") !=
              std::string::npos &&
          sealed_serialized.find(
              R"("execution_mode":"layer-major","logical_panel_count":2,"request_memory_profile":"layer-major-c8192","bounded_submission_window":true,"submission_window_retirements":128)") !=
              std::string::npos &&
          sealed_serialized.find(
              R"("deployment_plan":{"available":true,"scope":"engine_lifetime_sealed_native_plan","id":"q3x.sm87.exact.layer-major-c8192.operator-panel.v3"})") !=
              std::string::npos,
      "sealed whole-request evidence keeps the stable v2 contract");

  server::TargetPrefillWitnessRecord candidate_record = sealed_record;
  candidate_record.operator_panel_executor_hits = 128U;
  candidate_record.native_group_q64_panel_hits = 32U;
  candidate_record.generic_qt2_hits = 0U;
  candidate_record.deployment_plan_id =
      q3x::runtime::kLayerMajorNativeGroupQ64PanelDeploymentPlanId;
  const std::string candidate_serialized =
      server::serialize_target_prefill_witness(candidate_record);
  test.expect(
      valid_json(candidate_serialized) &&
          candidate_serialized.find(
              R"("record":"target-prefill-witness-v3","schema_version":3)") !=
              std::string::npos &&
          candidate_serialized.find(
              R"("attention_tactic":"native-group-q64-panel","operator_panel_executor_hits":128,"native_group_q64_panel_hits":32,"native_group_q128_v4_panel_hits":0,"generic_qt2_hits":0)") !=
              std::string::npos &&
          candidate_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-layermajor-8k.native-group-q64-panel.v1","qualification":"accuracy-unqualified-architecture-candidate","numerical_contract":{"qualified":false,"reason":"full-state-accuracy-qualification-not-run"})") !=
              std::string::npos &&
          candidate_serialized.find(
              R"("native_flashinfer_exact_panel_hits")") ==
              std::string::npos &&
          candidate_serialized.find(
              R"("disabled_boundaries":{"scope":"architecture_candidate_unqualified","prefix_cache":true,"mtp":true,"cublaslt_production":true,"approximate_numerics":false})") !=
              std::string::npos,
      "native grouped-Q64 candidate has an explicit unqualified v3 "
      "execution witness");

  server::TargetPrefillWitnessRecord q128_candidate_record =
      candidate_record;
  q128_candidate_record.native_group_q64_panel_hits = 0U;
  q128_candidate_record.native_group_q128_v4_panel_hits = 32U;
  q128_candidate_record.deployment_plan_id =
      q3x::runtime::kLayerMajorNativeGroupQ128V4PanelDeploymentPlanId;
  const std::string q128_candidate_serialized =
      server::serialize_target_prefill_witness(q128_candidate_record);
  test.expect(
      valid_json(q128_candidate_serialized) &&
          q128_candidate_serialized.find(
              R"("record":"target-prefill-witness-v3","schema_version":3)") !=
              std::string::npos &&
          q128_candidate_serialized.find(
              R"("attention_tactic":"native-group-q128-v4-panel","operator_panel_executor_hits":128,"native_group_q64_panel_hits":0,"native_group_q128_v4_panel_hits":32,"generic_qt2_hits":0)") !=
              std::string::npos &&
          q128_candidate_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-layermajor-8k.native-group-q128-v4-panel.v1","qualification":"accuracy-unqualified-architecture-candidate","numerical_contract":{"qualified":false,"reason":"full-state-accuracy-qualification-not-run"})") !=
              std::string::npos &&
          q128_candidate_serialized.find(
              R"("disabled_boundaries":{"scope":"architecture_candidate_unqualified","prefix_cache":true,"mtp":true,"cublaslt_production":true,"approximate_numerics":false})") !=
              std::string::npos,
      "native grouped-Q128-v4 candidate has a distinct unqualified v3 "
      "execution witness");

  server::TargetPrefillWitnessRecord flashinfer_candidate_record =
      sealed_record;
  flashinfer_candidate_record.operator_panel_executor_hits = 128U;
  flashinfer_candidate_record.native_flashinfer_exact_panel_hits = 32U;
  flashinfer_candidate_record.generic_qt2_hits = 0U;
  flashinfer_candidate_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeFlashInferExactPanelDeploymentPlanId;
  const std::string flashinfer_candidate_serialized =
      server::serialize_target_prefill_witness(
          flashinfer_candidate_record);
  test.expect(
      valid_json(flashinfer_candidate_serialized) &&
          flashinfer_candidate_serialized.find(
              R"("record":"target-prefill-witness-v6","schema_version":6)") !=
              std::string::npos &&
          flashinfer_candidate_serialized.find(
              R"("projection_tactic":"exact-segmented")") !=
              std::string::npos &&
          flashinfer_candidate_serialized.find(
              R"("attention_tactic":"native-flashinfer-exact-panel")") !=
              std::string::npos &&
          flashinfer_candidate_serialized.find(
              R"("native_flashinfer_exact_panel_hits":32)") !=
              std::string::npos &&
          flashinfer_candidate_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-prompt-wide-v2.exact-segmented-projection.native-flashinfer-exact-panel-attention.v1","qualification":"accuracy-unqualified-architecture-candidate","numerical_contract":{"qualified":false,"reason":"full-state-accuracy-qualification-not-run"})") !=
              std::string::npos,
      "exact FlashInfer panel Attention has an explicit unqualified v6 "
      "execution witness");

  server::TargetPrefillWitnessRecord projection_candidate_record =
      sealed_record;
  projection_candidate_record.operator_panel_executor_hits = 128U;
  projection_candidate_record.native_group_q64_panel_hits = 0U;
  projection_candidate_record.generic_qt2_hits = 4U;
  projection_candidate_record.segmented_panel_projection_hits = 672U;
  projection_candidate_record.segmented_panel_projection_physical_launches =
      4'928U;
  projection_candidate_record.deployment_plan_id =
      q3x::runtime::kLayerMajorSegmentedMarlinProjectionDeploymentPlanId;
  const std::string projection_candidate_serialized =
      server::serialize_target_prefill_witness(
          projection_candidate_record);
  test.expect(
      valid_json(projection_candidate_serialized) &&
          projection_candidate_serialized.find(
              R"("record":"target-prefill-witness-v4","schema_version":4)") !=
              std::string::npos &&
          projection_candidate_serialized.find(
              R"("projection_tactic":"segmented-marlin-operator-panel","attention_tactic":"exact-segmented","operator_panel_executor_hits":128,"native_group_q64_panel_hits":0,"native_group_q128_v4_panel_hits":0,"generic_qt2_hits":4,"segmented_panel_projection_hits":672,"segmented_panel_projection_physical_launches":4928)") !=
              std::string::npos &&
          projection_candidate_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-layermajor-8k.segmented-marlin-operator-panel.exact-segmented-attention.v1","qualification":"accuracy-unqualified-architecture-candidate","numerical_contract":{"qualified":false,"reason":"full-state-accuracy-qualification-not-run"})") !=
              std::string::npos &&
          projection_candidate_serialized.find(
              R"("native_flashinfer_exact_panel_hits")") ==
              std::string::npos,
      "native Marlin panel projections have explicit unqualified v4 route "
      "and physical-launch evidence");

  projection_candidate_record.deployment_plan_id = q3x::runtime::
      kLayerMajorSegmentedMarlinProjectionGroupQ64DeploymentPlanId;
  projection_candidate_record.native_group_q64_panel_hits = 32U;
  projection_candidate_record.native_group_q128_v4_panel_hits = 0U;
  projection_candidate_record.generic_qt2_hits = 0U;
  const std::string combined_candidate_serialized =
      server::serialize_target_prefill_witness(
          projection_candidate_record);
  test.expect(
      valid_json(combined_candidate_serialized) &&
          combined_candidate_serialized.find(
              R"("attention_tactic":"native-group-q64-panel")") !=
              std::string::npos &&
          combined_candidate_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-layermajor-8k.segmented-marlin-operator-panel.native-group-q64-attention.v1")") !=
              std::string::npos,
      "combined native projection and Attention tactics have a distinct "
      "deployment identity");

  projection_candidate_record.deployment_plan_id = q3x::runtime::
      kLayerMajorSegmentedMarlinProjectionGroupQ128V4DeploymentPlanId;
  projection_candidate_record.native_group_q64_panel_hits = 0U;
  projection_candidate_record.native_group_q128_v4_panel_hits = 32U;
  const std::string combined_q128_candidate_serialized =
      server::serialize_target_prefill_witness(
          projection_candidate_record);
  test.expect(
      valid_json(combined_q128_candidate_serialized) &&
          combined_q128_candidate_serialized.find(
              R"("attention_tactic":"native-group-q128-v4-panel")") !=
              std::string::npos &&
          combined_q128_candidate_serialized.find(
              R"("native_group_q64_panel_hits":0,"native_group_q128_v4_panel_hits":32)") !=
              std::string::npos &&
          combined_q128_candidate_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-layermajor-8k.segmented-marlin-operator-panel.native-group-q128-v4-attention.v1")") !=
              std::string::npos,
      "segmented projections plus Q128-v4 Attention have a distinct "
      "unqualified v4 witness identity");

  projection_candidate_record.deployment_plan_id = q3x::runtime::
      kLayerMajorSegmentedMarlinProjectionFlashInferExactDeploymentPlanId;
  projection_candidate_record.native_group_q128_v4_panel_hits = 0U;
  projection_candidate_record.native_flashinfer_exact_panel_hits = 32U;
  projection_candidate_record.generic_qt2_hits = 0U;
  const std::string segmented_flashinfer_candidate_serialized =
      server::serialize_target_prefill_witness(
          projection_candidate_record);
  test.expect(
      valid_json(segmented_flashinfer_candidate_serialized) &&
          segmented_flashinfer_candidate_serialized.find(
              R"("record":"target-prefill-witness-v6","schema_version":6)") !=
              std::string::npos &&
          segmented_flashinfer_candidate_serialized.find(
              R"("projection_tactic":"segmented-marlin-operator-panel")") !=
              std::string::npos &&
          segmented_flashinfer_candidate_serialized.find(
              R"("attention_tactic":"native-flashinfer-exact-panel")") !=
              std::string::npos &&
          segmented_flashinfer_candidate_serialized.find(
              R"("native_flashinfer_exact_panel_hits":32)") !=
              std::string::npos &&
          segmented_flashinfer_candidate_serialized.find(
              R"("segmented_panel_projection_hits":672,"segmented_panel_projection_physical_launches":4928)") !=
              std::string::npos &&
          segmented_flashinfer_candidate_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-prompt-wide-v2.segmented-marlin-operator-panel.native-flashinfer-exact-panel-attention.v1")") !=
              std::string::npos,
      "segmented projections plus exact FlashInfer Attention have a distinct "
      "unqualified v6 witness identity");

  server::TargetPrefillWitnessRecord native_large_m_record = sealed_record;
  native_large_m_record.operator_panel_executor_hits = 128U;
  native_large_m_record.generic_qt2_hits = 4U;
  native_large_m_record.native_large_m_projection_hits = 672U;
  native_large_m_record.native_large_m_projection_bulk_hits = 672U;
  native_large_m_record.native_large_m_projection_oracle_partial_hits = 0U;
  native_large_m_record.native_large_m_projection_physical_launches = 672U;
  native_large_m_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeQuantizedLargeMProjectionDeploymentPlanId;
  const std::string native_large_m_serialized =
      server::serialize_target_prefill_witness(native_large_m_record);
  test.expect(
      valid_json(native_large_m_serialized) &&
          native_large_m_serialized.find(
              R"("record":"target-prefill-witness-v5","schema_version":5)") !=
              std::string::npos &&
          native_large_m_serialized.find(
              R"("projection_tactic":"native-quantized-large-m-operator-panel","attention_tactic":"exact-segmented","operator_panel_executor_hits":128,"native_group_q64_panel_hits":0,"native_group_q128_v4_panel_hits":0,"generic_qt2_hits":4,"native_large_m_projection_hits":672,"native_large_m_projection_bulk_hits":672,"native_large_m_projection_oracle_partial_hits":0,"native_large_m_projection_physical_launches":672)") !=
              std::string::npos &&
          native_large_m_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-layermajor-8k.native-quantized-large-m-operator-panel.exact-segmented-attention.v1","qualification":"accuracy-unqualified-architecture-candidate","numerical_contract":{"qualified":false,"reason":"full-state-accuracy-qualification-not-run"})") !=
              std::string::npos &&
          native_large_m_serialized.find(
              R"("segmented_panel_projection_hits")") ==
              std::string::npos &&
          native_large_m_serialized.find(
              R"("native_flashinfer_exact_panel_hits")") ==
              std::string::npos,
      "native large-M projections have explicit unqualified v5 route and "
      "physical-launch evidence");

  native_large_m_record.native_group_q64_panel_hits = 32U;
  native_large_m_record.generic_qt2_hits = 0U;
  native_large_m_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeQuantizedLargeMProjectionGroupQ64DeploymentPlanId;
  const std::string native_large_m_q64_serialized =
      server::serialize_target_prefill_witness(native_large_m_record);
  test.expect(
      valid_json(native_large_m_q64_serialized) &&
          native_large_m_q64_serialized.find(
              R"("attention_tactic":"native-group-q64-panel")") !=
              std::string::npos &&
          native_large_m_q64_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-layermajor-8k.native-quantized-large-m-operator-panel.native-group-q64-attention.v1")") !=
              std::string::npos,
      "native large-M projections plus Q64 Attention have a distinct v5 "
      "deployment identity");

  native_large_m_record.native_group_q64_panel_hits = 0U;
  native_large_m_record.native_group_q128_v4_panel_hits = 32U;
  native_large_m_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeQuantizedLargeMProjectionGroupQ128V4DeploymentPlanId;
  const std::string native_large_m_q128_serialized =
      server::serialize_target_prefill_witness(native_large_m_record);
  test.expect(
      valid_json(native_large_m_q128_serialized) &&
          native_large_m_q128_serialized.find(
              R"("attention_tactic":"native-group-q128-v4-panel")") !=
              std::string::npos &&
          native_large_m_q128_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-layermajor-8k.native-quantized-large-m-operator-panel.native-group-q128-v4-attention.v1")") !=
              std::string::npos,
      "native large-M projections plus Q128-v4 Attention have a distinct "
      "unqualified v5 witness identity");

  native_large_m_record.native_group_q128_v4_panel_hits = 0U;
  native_large_m_record.native_flashinfer_exact_panel_hits = 32U;
  native_large_m_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeQuantizedLargeMProjectionFlashInferExactDeploymentPlanId;
  const std::string native_large_m_flashinfer_serialized =
      server::serialize_target_prefill_witness(native_large_m_record);
  test.expect(
      valid_json(native_large_m_flashinfer_serialized) &&
          native_large_m_flashinfer_serialized.find(
              R"("record":"target-prefill-witness-v6","schema_version":6)") !=
              std::string::npos &&
          native_large_m_flashinfer_serialized.find(
              R"("projection_tactic":"native-quantized-large-m-operator-panel")") !=
              std::string::npos &&
          native_large_m_flashinfer_serialized.find(
              R"("attention_tactic":"native-flashinfer-exact-panel")") !=
              std::string::npos &&
          native_large_m_flashinfer_serialized.find(
              R"("native_flashinfer_exact_panel_hits":32)") !=
              std::string::npos &&
          native_large_m_flashinfer_serialized.find(
              R"("native_large_m_projection_hits":672,"native_large_m_projection_bulk_hits":672,"native_large_m_projection_oracle_partial_hits":0,"native_large_m_projection_physical_launches":672)") !=
              std::string::npos &&
          native_large_m_flashinfer_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-prompt-wide-v2.native-quantized-large-m-operator-panel.native-flashinfer-exact-panel-attention.v1")") !=
              std::string::npos,
      "native large-M projections plus exact FlashInfer Attention have a "
      "distinct unqualified v6 witness identity");

  server::TargetPrefillWitnessRecord true_large_m_nvfp4_record =
      sealed_record;
  true_large_m_nvfp4_record.operator_panel_executor_hits = 128U;
  true_large_m_nvfp4_record.generic_qt2_hits = 4U;
  true_large_m_nvfp4_record
      .nvfp4_true_large_m_route_fp8_projection_hits = 416U;
  true_large_m_nvfp4_record
      .nvfp4_true_large_m_route_fp8_projection_bulk_hits = 208U;
  true_large_m_nvfp4_record
      .nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits = 208U;
  true_large_m_nvfp4_record
      .nvfp4_true_large_m_route_fp8_projection_physical_launches = 1'872U;
  true_large_m_nvfp4_record
      .native_nvfp4_true_large_m_projection_hits = 192U;
  true_large_m_nvfp4_record.native_nvfp4_true_large_m_gate_up_hits = 96U;
  true_large_m_nvfp4_record.native_nvfp4_true_large_m_down_hits = 96U;
  true_large_m_nvfp4_record
      .native_nvfp4_true_large_m_physical_launches = 192U;
  true_large_m_nvfp4_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeNvfp4TrueLargeMProjectionDeploymentPlanId;
  const std::string true_large_m_nvfp4_serialized =
      server::serialize_target_prefill_witness(true_large_m_nvfp4_record);
  test.expect(
      valid_json(true_large_m_nvfp4_serialized) &&
          true_large_m_nvfp4_serialized.find(
              R"("record":"target-prefill-witness-v7","schema_version":7)") !=
              std::string::npos &&
          true_large_m_nvfp4_serialized.find(
              R"("projection_tactic":"native-nvfp4-true-large-m-operator-panel","attention_tactic":"exact-segmented")") !=
              std::string::npos &&
          true_large_m_nvfp4_serialized.find(
              R"("nvfp4_true_large_m_route_fp8_projection_hits":416,"nvfp4_true_large_m_route_fp8_projection_bulk_hits":208,"nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits":208,"nvfp4_true_large_m_route_fp8_projection_physical_launches":1872,"native_nvfp4_true_large_m_projection_hits":192,"native_nvfp4_true_large_m_gate_up_hits":96,"native_nvfp4_true_large_m_down_hits":96,"native_nvfp4_true_large_m_physical_launches":192)") !=
              std::string::npos &&
          true_large_m_nvfp4_serialized.find(
              R"("native_large_m_projection_hits")") ==
              std::string::npos &&
          true_large_m_nvfp4_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-prompt-wide-v2.native-nvfp4-true-large-m-operator-panel.exact-segmented-attention.v1","qualification":"accuracy-unqualified-architecture-candidate","numerical_contract":{"qualified":false,"reason":"full-state-accuracy-qualification-not-run"})") !=
              std::string::npos,
      "true-large-M NVFP4 Gate+Up/Down uses independent unqualified v7 "
      "execution evidence");

  true_large_m_nvfp4_record.native_group_q64_panel_hits = 32U;
  true_large_m_nvfp4_record.generic_qt2_hits = 0U;
  true_large_m_nvfp4_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeNvfp4TrueLargeMProjectionGroupQ64DeploymentPlanId;
  const std::string true_large_m_nvfp4_q64_serialized =
      server::serialize_target_prefill_witness(true_large_m_nvfp4_record);
  test.expect(
      valid_json(true_large_m_nvfp4_q64_serialized) &&
          true_large_m_nvfp4_q64_serialized.find(
              R"("attention_tactic":"native-group-q64-panel")") !=
              std::string::npos,
      "true-large-M NVFP4 composes with grouped-Q64 Attention in v7");

  true_large_m_nvfp4_record.native_group_q64_panel_hits = 0U;
  true_large_m_nvfp4_record.native_group_q128_v4_panel_hits = 32U;
  true_large_m_nvfp4_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeNvfp4TrueLargeMProjectionGroupQ128V4DeploymentPlanId;
  const std::string true_large_m_nvfp4_q128_serialized =
      server::serialize_target_prefill_witness(true_large_m_nvfp4_record);
  test.expect(
      valid_json(true_large_m_nvfp4_q128_serialized) &&
          true_large_m_nvfp4_q128_serialized.find(
              R"("attention_tactic":"native-group-q128-v4-panel")") !=
              std::string::npos,
      "true-large-M NVFP4 composes with grouped-Q128-v4 Attention in v7");

  true_large_m_nvfp4_record.native_group_q128_v4_panel_hits = 0U;
  true_large_m_nvfp4_record.native_flashinfer_exact_panel_hits = 32U;
  true_large_m_nvfp4_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeNvfp4TrueLargeMProjectionFlashInferExactDeploymentPlanId;
  const std::string true_large_m_nvfp4_flashinfer_serialized =
      server::serialize_target_prefill_witness(true_large_m_nvfp4_record);
  test.expect(
      valid_json(true_large_m_nvfp4_flashinfer_serialized) &&
          true_large_m_nvfp4_flashinfer_serialized.find(
              R"("record":"target-prefill-witness-v7","schema_version":7)") !=
              std::string::npos &&
          true_large_m_nvfp4_flashinfer_serialized.find(
              R"("attention_tactic":"native-flashinfer-exact-panel")") !=
              std::string::npos,
      "true-large-M NVFP4 composes with exact FlashInfer Attention in v7");

  server::TargetPrefillWitnessRecord g2_d2_nvfp4_record =
      true_large_m_nvfp4_record;
  g2_d2_nvfp4_record.native_flashinfer_exact_panel_hits = 0U;
  g2_d2_nvfp4_record.generic_qt2_hits = 4U;
  g2_d2_nvfp4_record.native_nvfp4_true_large_m_projection_hits = 256U;
  g2_d2_nvfp4_record.native_nvfp4_true_large_m_gate_up_hits = 128U;
  g2_d2_nvfp4_record.native_nvfp4_true_large_m_down_hits = 128U;
  g2_d2_nvfp4_record.native_nvfp4_true_large_m_physical_launches = 256U;
  g2_d2_nvfp4_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeNvfp4G2D2LargeMProjectionDeploymentPlanId;
  const std::string g2_d2_nvfp4_serialized =
      server::serialize_target_prefill_witness(g2_d2_nvfp4_record);
  test.expect(
      valid_json(g2_d2_nvfp4_serialized) &&
          g2_d2_nvfp4_serialized.find(
              R"("record":"target-prefill-witness-v8","schema_version":8)") !=
              std::string::npos &&
          g2_d2_nvfp4_serialized.find(
              R"("projection_tactic":"native-nvfp4-g2-d2-large-m-operator-panel")") !=
              std::string::npos &&
          g2_d2_nvfp4_serialized.find(
              R"("nvfp4_package":"gate-g2+down-d2","nvfp4_gate_role_tactic":"GateUpG2","nvfp4_down_role_tactic":"DownD2","package_complete":true,"attention_tactic":"exact-segmented")") !=
              std::string::npos &&
          g2_d2_nvfp4_serialized.find(
              R"("native_nvfp4_g2_d2_projection_hits":256,"native_nvfp4_gate_up_g2_hits":128,"native_nvfp4_down_d2_hits":128,"native_nvfp4_g2_d2_physical_launches":256)") !=
              std::string::npos &&
          g2_d2_nvfp4_serialized.find(
              R"("native_nvfp4_true_large_m_projection_hits")") ==
              std::string::npos &&
          g2_d2_nvfp4_serialized.find(
              R"("nvfp4_g2_d2_package":{"identity":"native-nvfp4-g2-d2-large-m","selection":"coupled-fail-closed","complete":true,"roles":{"gate_up":{"identity":"GateUpG2","publishes":"activated_bf16","fused_silu_gate":true,"logical_hits":128},"down":{"identity":"DownD2","publishes":"prompt_residual_bf16","fused_in_place_residual":true,"logical_hits":128}},"count_validation":{"complete":true,"expected_layer_panel_hits":128,"logical_panel_count":2,"expected_nvfp4_logical_projection_hits":256,"observed_nvfp4_logical_projection_hits":256,"expected_nvfp4_physical_launches":256,"observed_nvfp4_physical_launches":256,"expected_fp8_companion_projection_hits":416,"observed_fp8_companion_projection_hits":416,"attention_route_complete":true,"expected_native_attention_hits":0}})") !=
              std::string::npos &&
          g2_d2_nvfp4_serialized.find(
              R"("id":"q3x.sm87.ac-prefill-prompt-wide-v2.native-nvfp4-g2-d2-large-m-operator-panel.exact-segmented-attention.v2")") !=
              std::string::npos,
      "G2/D2 emits a distinct v8 coupled-package witness with complete role "
      "and launch counts");

  server::TargetPrefillWitnessRecord g2_d2_missing_exact_attention =
      g2_d2_nvfp4_record;
  g2_d2_missing_exact_attention.generic_qt2_hits = 0U;
  const std::string g2_d2_missing_exact_attention_serialized =
      server::serialize_target_prefill_witness(
          g2_d2_missing_exact_attention);
  test.expect(
      valid_json(g2_d2_missing_exact_attention_serialized) &&
          g2_d2_missing_exact_attention_serialized.find(
              R"("package_complete":false,"attention_tactic":"exact-segmented")") !=
              std::string::npos &&
          g2_d2_missing_exact_attention_serialized.find(
              R"("attention_route_complete":false)") != std::string::npos,
      "G2/D2 exact-segmented v8 requires nonzero generic QT2 evidence");

  g2_d2_nvfp4_record.native_group_q64_panel_hits = 32U;
  g2_d2_nvfp4_record.generic_qt2_hits = 0U;
  g2_d2_nvfp4_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeNvfp4G2D2LargeMProjectionGroupQ64DeploymentPlanId;
  const std::string g2_d2_nvfp4_q64_serialized =
      server::serialize_target_prefill_witness(g2_d2_nvfp4_record);
  test.expect(
      valid_json(g2_d2_nvfp4_q64_serialized) &&
          g2_d2_nvfp4_q64_serialized.find(
              R"("package_complete":true,"attention_tactic":"native-group-q64-panel")") !=
              std::string::npos,
      "G2/D2 composes with grouped-Q64 Attention under a distinct v8 id");

  g2_d2_nvfp4_record.native_group_q64_panel_hits = 0U;
  g2_d2_nvfp4_record.native_group_q128_v4_panel_hits = 32U;
  g2_d2_nvfp4_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeNvfp4G2D2LargeMProjectionGroupQ128V4DeploymentPlanId;
  const std::string g2_d2_nvfp4_q128_serialized =
      server::serialize_target_prefill_witness(g2_d2_nvfp4_record);
  test.expect(
      valid_json(g2_d2_nvfp4_q128_serialized) &&
          g2_d2_nvfp4_q128_serialized.find(
              R"("package_complete":true,"attention_tactic":"native-group-q128-v4-panel")") !=
              std::string::npos,
      "G2/D2 composes with grouped-Q128-v4 Attention under a distinct v8 id");

  g2_d2_nvfp4_record.native_group_q128_v4_panel_hits = 0U;
  g2_d2_nvfp4_record.native_flashinfer_exact_panel_hits = 32U;
  g2_d2_nvfp4_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeNvfp4G2D2LargeMProjectionFlashInferExactDeploymentPlanId;
  const std::string g2_d2_nvfp4_flashinfer_serialized =
      server::serialize_target_prefill_witness(g2_d2_nvfp4_record);
  test.expect(
      valid_json(g2_d2_nvfp4_flashinfer_serialized) &&
          g2_d2_nvfp4_flashinfer_serialized.find(
              R"("package_complete":true,"attention_tactic":"native-flashinfer-exact-panel")") !=
              std::string::npos,
      "G2/D2 composes with exact FlashInfer Attention under a distinct v8 id");

  server::TargetPrefillWitnessRecord g2_d2_bad_panel_count =
      g2_d2_nvfp4_record;
  g2_d2_bad_panel_count.prefill_logical_panel_count = 3U;
  const std::string g2_d2_bad_panel_count_serialized =
      server::serialize_target_prefill_witness(g2_d2_bad_panel_count);
  test.expect(
      valid_json(g2_d2_bad_panel_count_serialized) &&
          g2_d2_bad_panel_count_serialized.find(
              R"("selection":"coupled-fail-closed","complete":false)") !=
              std::string::npos,
      "G2/D2 v8 count validation binds executor hits to logical panels");

  server::TargetPrefillWitnessRecord g2_d2_bad_attention =
      g2_d2_nvfp4_record;
  --g2_d2_bad_attention.native_flashinfer_exact_panel_hits;
  const std::string g2_d2_bad_attention_serialized =
      server::serialize_target_prefill_witness(g2_d2_bad_attention);
  test.expect(
      valid_json(g2_d2_bad_attention_serialized) &&
          g2_d2_bad_attention_serialized.find(
              R"("selection":"coupled-fail-closed","complete":false)") !=
              std::string::npos &&
          g2_d2_bad_attention_serialized.find(
              R"("attention_route_complete":false)") != std::string::npos,
      "G2/D2 v8 count validation rejects an Attention deployment/count "
      "mismatch");

  server::TargetPrefillWitnessRecord g2_d2_bad_fp8 = g2_d2_nvfp4_record;
  g2_d2_bad_fp8.nvfp4_true_large_m_route_fp8_projection_physical_launches =
      g2_d2_bad_fp8.nvfp4_true_large_m_route_fp8_projection_hits - 1U;
  const std::string g2_d2_bad_fp8_serialized =
      server::serialize_target_prefill_witness(g2_d2_bad_fp8);
  test.expect(
      valid_json(g2_d2_bad_fp8_serialized) &&
          g2_d2_bad_fp8_serialized.find(
              R"("selection":"coupled-fail-closed","complete":false)") !=
              std::string::npos,
      "G2/D2 v8 count validation rejects incomplete FP8 physical coverage");

  --g2_d2_nvfp4_record.native_nvfp4_true_large_m_gate_up_hits;
  const std::string incomplete_g2_d2_serialized =
      server::serialize_target_prefill_witness(g2_d2_nvfp4_record);
  test.expect(
      valid_json(incomplete_g2_d2_serialized) &&
          incomplete_g2_d2_serialized.find(
              R"("nvfp4_g2_d2_package":{"identity":"native-nvfp4-g2-d2-large-m","selection":"coupled-fail-closed","complete":false)") !=
              std::string::npos &&
          incomplete_g2_d2_serialized.find(
              R"("count_validation":{"complete":false)") !=
              std::string::npos,
      "G2/D2 v8 witness never reports an incomplete role count as complete");

  server::TargetPrefillWitnessRecord persistent_p40_record = sealed_record;
  persistent_p40_record.prompt_tokens =
      q3x::runtime::kLayerMajorPrefillLayerWideMlpP40Tokens;
  persistent_p40_record.consumed_prompt_tokens =
      persistent_p40_record.prompt_tokens;
  persistent_p40_record.prefix_execution_count = 5U;
  persistent_p40_record.prefill_logical_panel_count = 5U;
  persistent_p40_record.submission_window_retirements = 384U;
  persistent_p40_record.operator_panel_executor_hits = 320U;
  persistent_p40_record.native_flashinfer_exact_panel_hits = 80U;
  persistent_p40_record.mlp_schedule_tactic = q3x::runtime::
      LayerMajorPrefillMlpScheduleTactic::kLayerWideP40ExactFullM;
  persistent_p40_record.route_layer_pass_count = 1U;
  persistent_p40_record.layer_wide_p40_mlp_layer_hits = 64U;
  persistent_p40_record.persistent_p40_nvfp4_gate_up_hits = 64U;
  persistent_p40_record.persistent_p40_nvfp4_down_residual_hits = 64U;
  persistent_p40_record.persistent_p40_nvfp4_physical_launches = 128U;
  persistent_p40_record.persistent_p40_fp8_projection_hits = 1'040U;
  persistent_p40_record.persistent_p40_fp8_projection_bulk_hits = 624U;
  persistent_p40_record
      .persistent_p40_fp8_projection_oracle_partial_hits = 416U;
  persistent_p40_record
      .persistent_p40_fp8_projection_physical_launches = 1'040U;
  q3x::runtime::reset_prefill_route_request(
      persistent_p40_record.prefill_route_evidence);
  test.expect(q3x::runtime::commit_prefill_route_layer_pass(
                  persistent_p40_record.prefill_route_evidence,
                  route_tile),
              "P40000 witness fixture commits one request-wide route pass");
  persistent_p40_record.prefill_route_evidence =
      q3x::runtime::finalize_prefill_route_request(
          persistent_p40_record.prefill_route_evidence, 1U);
  persistent_p40_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativeNvfp4PersistentP40MlpFlashInferExactDeploymentPlanId;
  const std::string persistent_p40_serialized =
      server::serialize_target_prefill_witness(persistent_p40_record);
  test.expect(
      valid_json(persistent_p40_serialized) &&
          persistent_p40_serialized.find(
              R"("record":"target-prefill-witness-v9","schema_version":9)") !=
              std::string::npos &&
          persistent_p40_serialized.find(
              R"("projection_tactic":"native-nvfp4-persistent-p40-layer-wide-mlp","mlp_schedule":"layer-wide-p40-exact-full-m","attention_tactic":"native-flashinfer-exact-panel","package_complete":true)") !=
              std::string::npos &&
          persistent_p40_serialized.find(
              R"("persistent_p40_package":{"identity":"exact-p40000-layer-wide-nvfp4-mlp","selection":"sealed-fail-closed","complete":true)") !=
              std::string::npos &&
          persistent_p40_serialized.find(
              R"("expected_fp8_hits":1040,"expected_bulk_fp8_hits":624,"expected_partial_fp8_hits":416,"attention_route_complete":true)") !=
              std::string::npos,
      "exact-P40000 persistent MLP emits a complete v9 package witness");

  --persistent_p40_record.persistent_p40_nvfp4_down_residual_hits;
  const std::string incomplete_persistent_p40_serialized =
      server::serialize_target_prefill_witness(persistent_p40_record);
  test.expect(
      valid_json(incomplete_persistent_p40_serialized) &&
          incomplete_persistent_p40_serialized.find(
              R"("package_complete":false)") != std::string::npos &&
          incomplete_persistent_p40_serialized.find(
              R"("selection":"sealed-fail-closed","complete":false)") !=
              std::string::npos,
      "P40000 v9 witness fails closed on an incomplete Down role count");

  server::TargetPrefillWitnessRecord whole_core_p40_record = sealed_record;
  whole_core_p40_record.prompt_tokens =
      q3x::runtime::kLayerMajorPrefillPromptWideP40Tokens;
  whole_core_p40_record.consumed_prompt_tokens =
      whole_core_p40_record.prompt_tokens;
  whole_core_p40_record.prefix_execution_count = 1U;
  whole_core_p40_record.prefill_logical_panel_count = 5U;
  whole_core_p40_record.request_memory_profile =
      q3x::runtime::RequestMemoryProfile::kLayerMajorP40WholeCore;
  whole_core_p40_record.submission_window_retirements = 768U;
  whole_core_p40_record.mlp_schedule_tactic = q3x::runtime::
      LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore;
  whole_core_p40_record.route_layer_pass_count = 1U;
  whole_core_p40_record.prompt_wide_p40_whole_core_layer_hits = 64U;
  whole_core_p40_record.prompt_wide_p40_fill_panel_hits = 320U;
  whole_core_p40_record.prompt_wide_p40_prompt_core_hits = 64U;
  whole_core_p40_record.prompt_wide_p40_drain_panel_hits = 320U;
  whole_core_p40_record.prompt_wide_p40_fp8_projection_hits = 1'040U;
  whole_core_p40_record
      .prompt_wide_p40_fp8_projection_physical_launches = 1'040U;
  whole_core_p40_record.prompt_wide_p40_bf16_ab_hits = 48U;
  whole_core_p40_record.prompt_wide_p40_gdn_hits = 48U;
  whole_core_p40_record.native_flashinfer_exact_whole_prompt_hits = 16U;
  whole_core_p40_record.layer_wide_p40_mlp_layer_hits = 64U;
  whole_core_p40_record.persistent_p40_nvfp4_gate_up_hits = 64U;
  whole_core_p40_record.persistent_p40_nvfp4_down_residual_hits = 64U;
  whole_core_p40_record.persistent_p40_nvfp4_physical_launches = 128U;
  q3x::runtime::reset_prefill_route_request(
      whole_core_p40_record.prefill_route_evidence);
  test.expect(q3x::runtime::commit_prefill_route_layer_pass(
                  whole_core_p40_record.prefill_route_evidence,
                  route_tile),
              "P40000 whole-core witness fixture commits one route pass");
  whole_core_p40_record.prefill_route_evidence =
      q3x::runtime::finalize_prefill_route_request(
          whole_core_p40_record.prefill_route_evidence, 1U);
  whole_core_p40_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativePromptWideP40WholeCoreDeploymentPlanId;
  const std::string whole_core_p40_serialized =
      server::serialize_target_prefill_witness(whole_core_p40_record);
  test.expect(
      valid_json(whole_core_p40_serialized) &&
          whole_core_p40_serialized.find(
              R"("record":"target-prefill-witness-v10","schema_version":10)") !=
              std::string::npos &&
          whole_core_p40_serialized.find(
              R"("request_memory_profile":"layer-major-p40-whole-core")") !=
              std::string::npos &&
          whole_core_p40_serialized.find(
              R"("projection_tactic":"native-prompt-wide-p40-whole-core","mlp_schedule":"prompt-wide-p40-whole-core","attention_tactic":"native-flashinfer-exact-whole-prompt","package_complete":true)") !=
              std::string::npos &&
          whole_core_p40_serialized.find(
              R"("prompt_wide_p40_fp8_projection_hits":1040,"prompt_wide_p40_fp8_projection_physical_launches":1040,"prompt_wide_p40_bf16_ab_hits":48,"prompt_wide_p40_gdn_hits":48,"native_flashinfer_exact_whole_prompt_hits":16)") !=
              std::string::npos &&
          whole_core_p40_serialized.find(
              R"("prompt_wide_p40_whole_core_package":{"identity":"exact-p40000-five-p8000-whole-core-v1","selection":"sealed-fail-closed","complete":true)") !=
              std::string::npos,
      "exact-P40000 whole-core route emits a complete v10 package witness");

  server::TargetPrefillWitnessRecord projection_reset_p40_record =
      whole_core_p40_record;
  projection_reset_p40_record.mlp_schedule_tactic = q3x::runtime::
      LayerMajorPrefillMlpScheduleTactic::kPromptWideP40ProjectionReset;
  projection_reset_p40_record.prompt_wide_p40_fp8_projection_hits = 208U;
  projection_reset_p40_record
      .prompt_wide_p40_fp8_projection_physical_launches = 128U;
  projection_reset_p40_record.deployment_plan_id = q3x::runtime::
      kLayerMajorNativePromptWideP40ProjectionResetDeploymentPlanId;
  const std::string projection_reset_p40_serialized =
      server::serialize_target_prefill_witness(projection_reset_p40_record);
  test.expect(
      valid_json(projection_reset_p40_serialized) &&
          projection_reset_p40_serialized.find(
              R"("record":"target-prefill-witness-v11","schema_version":11)") !=
              std::string::npos &&
          projection_reset_p40_serialized.find(
              R"("projection_tactic":"native-prompt-wide-p40-projection-reset","mlp_schedule":"prompt-wide-p40-projection-reset","attention_tactic":"native-flashinfer-exact-whole-prompt","package_complete":true)") !=
              std::string::npos &&
          projection_reset_p40_serialized.find(
              R"("prompt_wide_p40_fp8_projection_hits":208,"prompt_wide_p40_fp8_projection_physical_launches":128)") !=
              std::string::npos &&
          projection_reset_p40_serialized.find(
              R"("p40_projection_reset_package":{"identity":"exact-p40000-grouped-projection-reset-v1","selection":"sealed-fail-closed","complete":true)") !=
              std::string::npos &&
          projection_reset_p40_serialized.find(
              R"("expected_fp8_tensor_role_hits":208,"expected_fp8_physical_launches":128,"expected_nvfp4_physical_launches":128)") !=
              std::string::npos &&
          projection_reset_p40_serialized.find(
              R"("qualification":"accuracy-unqualified-architecture-candidate")") !=
              std::string::npos,
      "exact-P40000 grouped projection reset emits a distinct complete v11 "
      "witness with separate logical and physical FP8 counts");

  server::TargetPrefillWitnessRecord reset_not_fully_consumed =
      projection_reset_p40_record;
  reset_not_fully_consumed.full_prompt_consumed = false;
  server::TargetPrefillWitnessRecord reset_short_consumption =
      projection_reset_p40_record;
  --reset_short_consumption.consumed_prompt_tokens;
  server::TargetPrefillWitnessRecord reset_without_completion =
      projection_reset_p40_record;
  reset_without_completion.completion_tokens = 0U;
  const std::string reset_not_fully_consumed_serialized =
      server::serialize_target_prefill_witness(reset_not_fully_consumed);
  const std::string reset_short_consumption_serialized =
      server::serialize_target_prefill_witness(reset_short_consumption);
  const std::string reset_without_completion_serialized =
      server::serialize_target_prefill_witness(reset_without_completion);
  test.expect(
      valid_json(reset_not_fully_consumed_serialized) &&
          valid_json(reset_short_consumption_serialized) &&
          valid_json(reset_without_completion_serialized) &&
          reset_not_fully_consumed_serialized.find(
              R"("package_complete":false)") != std::string::npos &&
          reset_short_consumption_serialized.find(
              R"("package_complete":false)") != std::string::npos &&
          reset_without_completion_serialized.find(
              R"("package_complete":false)") != std::string::npos,
      "P40000 v11 package completeness requires full exact prompt "
      "consumption and a committed completion token");

  --projection_reset_p40_record
        .prompt_wide_p40_fp8_projection_physical_launches;
  const std::string incomplete_projection_reset_p40_serialized =
      server::serialize_target_prefill_witness(projection_reset_p40_record);
  test.expect(
      valid_json(incomplete_projection_reset_p40_serialized) &&
          incomplete_projection_reset_p40_serialized.find(
              R"("package_complete":false)") != std::string::npos &&
          incomplete_projection_reset_p40_serialized.find(
              R"("selection":"sealed-fail-closed","complete":false)") !=
              std::string::npos,
      "P40000 v11 witness fails closed on a missing grouped FP8 launch");

  --whole_core_p40_record.prompt_wide_p40_fp8_projection_physical_launches;
  const std::string incomplete_whole_core_p40_serialized =
      server::serialize_target_prefill_witness(whole_core_p40_record);
  test.expect(
      valid_json(incomplete_whole_core_p40_serialized) &&
          incomplete_whole_core_p40_serialized.find(
              R"("package_complete":false)") != std::string::npos &&
          incomplete_whole_core_p40_serialized.find(
              R"("selection":"sealed-fail-closed","complete":false)") !=
              std::string::npos,
      "P40000 v10 witness fails closed on a missing physical FP8 launch");

  ++whole_core_p40_record
        .prompt_wide_p40_fp8_projection_physical_launches;
  whole_core_p40_record.submission_window_retirements = 767U;
  const std::string short_retirement_whole_core_p40_serialized =
      server::serialize_target_prefill_witness(whole_core_p40_record);
  whole_core_p40_record.submission_window_retirements = 704U;
  const std::string legacy_retirement_whole_core_p40_serialized =
      server::serialize_target_prefill_witness(whole_core_p40_record);
  whole_core_p40_record.submission_window_retirements = 768U;
  whole_core_p40_record.bounded_submission_window = false;
  const std::string unbounded_whole_core_p40_serialized =
      server::serialize_target_prefill_witness(whole_core_p40_record);
  test.expect(
      valid_json(short_retirement_whole_core_p40_serialized) &&
          valid_json(legacy_retirement_whole_core_p40_serialized) &&
          valid_json(unbounded_whole_core_p40_serialized) &&
          short_retirement_whole_core_p40_serialized.find(
              R"("package_complete":false)") != std::string::npos &&
          legacy_retirement_whole_core_p40_serialized.find(
              R"("package_complete":false)") != std::string::npos &&
          unbounded_whole_core_p40_serialized.find(
              R"("package_complete":false)") != std::string::npos,
      "P40000 v10 witness requires a bounded 12-phase-per-layer submission "
      "window with exactly 768 retirements");

  server::TargetPrefillWitnessRecord unbound_layer_major_record =
      sealed_record;
  unbound_layer_major_record.deployment_plan_id.clear();
  const std::string unbound_layer_major_serialized =
      server::serialize_target_prefill_witness(unbound_layer_major_record);
  test.expect(
      valid_json(unbound_layer_major_serialized) &&
          unbound_layer_major_serialized.find(
              R"("record":"target-prefill-witness-v1","schema_version":1)") !=
              std::string::npos &&
          unbound_layer_major_serialized.find(
              R"("deployment_plan":{"available":false,"reason":"not_implemented"})") !=
              std::string::npos &&
          unbound_layer_major_serialized.find(
              R"("record":"target-prefill-witness-v2")") ==
              std::string::npos,
      "layer-major mode alone cannot synthesize a sealed deployment plan");

  server::TargetPrefillWitnessRecord invalid_record = record;
  invalid_record.prefill_route_evidence.expected_layer_passes = 3U;
  const std::string invalid_serialized =
      server::serialize_target_prefill_witness(invalid_record);
  test.expect(
      invalid_serialized.find(
          R"("per_operator_route_hits":{"available":false,"scope":"request_completed_prefill_logical_operators","complete":true,"reason":"unexpected_layer_pass_count")") !=
          std::string::npos &&
          invalid_serialized.find(
              R"("completed_layer_passes":2,"expected_layer_passes":3)") !=
              std::string::npos,
      "route mismatch is explicit and never serialized as executed evidence");
  test.expect(valid_json(invalid_serialized),
              "invalid route evidence remains parseable JSON");
}

void test_prefill_tactic_defaults_remain_exact(TestContext& test) {
  const server::EvaluationServerOptions server_options;
  const q3x::runtime::ReferenceEngineOptions engine_options;
  test.expect(
      server_options.prefill_execution_mode == q3x::runtime::
              ReferencePrefillExecutionMode::kLegacyC512Tiled &&
          server_options.prefill_full_attention_tactic == q3x::runtime::
              LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512 &&
          engine_options.prefill_execution_mode == q3x::runtime::
              ReferencePrefillExecutionMode::kLegacyC512Tiled &&
          engine_options.prefill_full_attention_tactic == q3x::runtime::
              LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512,
      "Q128-v4 exposure does not change server or engine defaults");
}

}  // namespace

int main() {
  TestContext test;
  test_evalscope_chat_contract(test);
  test_raw_completion_contract(test);
  test_target_length_token_id_contract(test);
  test_fail_closed_parameters(test);
  test_serialization(test);
  test_target_prefill_witness_evidence(test);
  test_prefill_tactic_defaults_remain_exact(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " OpenAI protocol test(s) failed\n";
    return 1;
  }
  std::cout << "All OpenAI protocol tests passed\n";
  return 0;
}
