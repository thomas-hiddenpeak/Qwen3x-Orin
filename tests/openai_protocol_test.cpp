#include "q3x/io/json.h"
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
      "sealed whole-request evidence upgrades to the stable v3 contract");

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

}  // namespace

int main() {
  TestContext test;
  test_evalscope_chat_contract(test);
  test_raw_completion_contract(test);
  test_target_length_token_id_contract(test);
  test_fail_closed_parameters(test);
  test_serialization(test);
  test_target_prefill_witness_evidence(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " OpenAI protocol test(s) failed\n";
    return 1;
  }
  std::cout << "All OpenAI protocol tests passed\n";
  return 0;
}
