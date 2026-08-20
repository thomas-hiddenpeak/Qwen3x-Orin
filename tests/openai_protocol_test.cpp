#include "q3x/io/json.h"
#include "q3x/server/openai_protocol.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

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

[[nodiscard]] const q3x::io::json::Value* member(
    const q3x::io::json::Value* const value,
    const std::string_view name) {
  return value == nullptr ? nullptr : value->find(name);
}

[[nodiscard]] bool is_json_string(
    const q3x::io::json::Value* const value,
    const std::string_view expected) {
  const std::string* const text =
      value == nullptr ? nullptr : value->as_string();
  return text != nullptr && *text == expected;
}

[[nodiscard]] bool is_json_bool(const q3x::io::json::Value* const value,
                                const bool expected) {
  const bool* const parsed = value == nullptr ? nullptr : value->as_bool();
  return parsed != nullptr && *parsed == expected;
}

[[nodiscard]] bool is_json_uint64(
    const q3x::io::json::Value* const value,
    const std::uint64_t expected) {
  const q3x::io::json::Number* const number =
      value == nullptr ? nullptr : value->as_number();
  std::uint64_t parsed = 0U;
  return number != nullptr && number->to_uint64(parsed) &&
         parsed == expected;
}

void test_evalscope_chat_contract(TestContext& test) {
  const auto parsed = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"system","content":"brief"},{"role":"user","content":"你好"}],"max_tokens":2048,"temperature":0.0,"top_p":1,"n":1,"seed":42,"stream":true,"stream_options":{"include_usage":true},"stop":null})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 4096U);
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
      32U);
  test.expect(parsed &&
                  parsed.value->prompt_kind ==
                      server::OpenAIPromptKind::kRawText &&
                  parsed.value->prompt == "raw prompt" &&
                  !parsed.value->stream,
              "plain completions retain raw text without a chat template");

  parsed = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","prompt":[1,248046,42],"max_completion_tokens":2,"temperature":0,"stream":true,"stream_options":{"include_usage":false}})",
      server::OpenAIEndpoint::kCompletions, "qwen3.6-27b-nvfp4",
      32U);
  test.expect(parsed &&
                  parsed.value->prompt_kind ==
                      server::OpenAIPromptKind::kTokenIds &&
                  parsed.value->prompt_token_ids.size() == 3U &&
                  parsed.value->prompt_token_ids[1] == 248046U &&
                  parsed.value->max_tokens == 2U &&
                  parsed.value->stream && !parsed.value->include_usage,
              "EvalScope tokenized prompt remains one flat exact id array");
}

void test_fail_closed_parameters(TestContext& test) {
  const auto unsupported = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0.7})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U);
  test.expect(!unsupported &&
                  unsupported.error.code == "unsupported_sampling" &&
                  unsupported.error.param == "temperature",
              "non-greedy sampling fails closed");

  const auto tool = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"x"}],"tools":[]})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U);
  test.expect(!tool && tool.error.code == "unsupported_parameter" &&
                  tool.error.param == "tools",
              "unimplemented tools are rejected rather than ignored");

  const auto wrong_model = server::parse_openai_request(
      R"({"model":"other","messages":[{"role":"user","content":"x"}]})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U);
  test.expect(!wrong_model && wrong_model.error.http_status == 404 &&
                  wrong_model.error.code == "model_not_found",
              "unknown served model has a stable 404 error");

  const auto content_array = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":[{"type":"text","text":"x"}]}]})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U);
  test.expect(!content_array &&
                  content_array.error.code == "unsupported_message",
              "multimodal content arrays fail closed");

  const auto fractional_seed = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0,"seed":1.5})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U);
  test.expect(!fractional_seed &&
                  fractional_seed.error.code == "invalid_seed" &&
                  fractional_seed.error.param == "seed",
              "EvalScope seed compatibility remains strict about integer type");

  const auto implicit_sampling = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"x"}],"max_tokens":1})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U);
  test.expect(!implicit_sampling &&
                  implicit_sampling.error.code == "missing_parameter" &&
                  implicit_sampling.error.param == "temperature",
              "the OpenAI sampling default is never silently treated as "
              "greedy");

  const auto implicit_output_cap = server::parse_openai_request(
      R"({"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"x"}],"temperature":0})",
      server::OpenAIEndpoint::kChatCompletions,
      "qwen3.6-27b-nvfp4", 32U);
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

  q3x::runtime::ReferenceEngineLoadStats load;
  load.projection_backend =
      q3x::runtime::ProjectionBackend::kSm87WeightOnly;
  load.request_max_sequence_length = 131'072U;
  load.request_prefill_chunk_size = 512U;
  load.request_arena_bytes = 9'876'543'210ULL;
  load.fp8_prefill_supermatrix_sidecars_enabled = true;
  load.fp8_prefill_supermatrix_sidecar_projections = 208U;
  load.fp8_prefill_supermatrix_sidecar_bytes = 7'516'192'768ULL;
  load.nvfp4_down_scale6_sidecars_enabled = true;
  load.nvfp4_down_scale6_sidecar_eligible_layers = 63U;
  load.nvfp4_down_scale6_sidecar_fallback_layers = 1U;
  load.nvfp4_down_scale6_sidecar_bytes = 4'177'920U;
  load.nvfp4_down_scale6_sidecar_fallback_reason = "layer \"63\"";
  load.decode_graph_cache_requested_policy =
      q3x::runtime::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions;
  load.decode_graph_cache_effective_policy =
      q3x::runtime::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions;
  load.decode_graph_cache_first_position = 1U;
  load.decode_graph_cache_last_position = 16U;
  load.decode_graph_cache_slot_count = 16U;

  const server::ExecutionRouteAttestation attestation =
      server::make_execution_route_attestation(load, 7U);
  q3x::runtime::ReferenceEngineLoadStats disabled_scale6 = load;
  disabled_scale6.nvfp4_down_scale6_sidecars_enabled = false;
  const server::ExecutionRouteAttestation disabled_attestation =
      server::make_execution_route_attestation(disabled_scale6);
  test.expect(
      disabled_attestation.nvfp4_down_scale6.attached_artifacts == 0U &&
          disabled_attestation.nvfp4_down_scale6.fallback_artifacts == 64U,
      "an unadmitted scale6 inventory reports every layer on fallback, not "
      "eligible layers as attached sidecars");
  const std::string health = server::serialize_health_response(
      "qwen\"model", attestation);
  const q3x::io::json::ParseResult parsed_health =
      q3x::io::json::parse(health);
  const q3x::io::json::Value* const root =
      parsed_health ? &*parsed_health.value : nullptr;
  const q3x::io::json::Value* const route =
      member(root, "route_attestation");
  const q3x::io::json::Value* const projection =
      member(route, "projection");
  const q3x::io::json::Value* const prefill = member(route, "prefill");
  const q3x::io::json::Value* const sidecars = member(route, "sidecars");
  const q3x::io::json::Value* const supermatrix =
      member(sidecars, "fp8_prefill_supermatrix");
  const q3x::io::json::Value* const scale6 =
      member(sidecars, "nvfp4_down_scale6");
  const q3x::io::json::Value* const graph =
      member(route, "decode_graph");
  test.expect(
      parsed_health && is_json_string(member(root, "model"), "qwen\"model") &&
          is_json_string(member(route, "kind"), "engine_build_facts") &&
          is_json_uint64(member(route, "schema_version"), 1U) &&
          is_json_bool(member(route, "deployment_plan_available"), false) &&
          is_json_bool(member(route, "per_operator_route_hits_available"),
                       false) &&
          is_json_uint64(member(route, "generation_attempts_observed"), 7U) &&
          is_json_string(member(projection, "engine_backend"),
                         "sm87_weight_only") &&
          is_json_uint64(member(prefill, "chunk_size"), 512U) &&
          is_json_uint64(member(prefill, "max_sequence_length"), 131'072U) &&
          is_json_uint64(member(prefill, "request_arena_bytes"),
                         9'876'543'210ULL) &&
          is_json_bool(member(supermatrix, "enabled"), true) &&
          is_json_uint64(member(supermatrix, "attached_artifacts"), 208U) &&
          is_json_uint64(member(supermatrix, "bytes"), 7'516'192'768ULL) &&
          is_json_uint64(member(scale6, "fallback_artifacts"), 1U) &&
          is_json_string(member(scale6, "fallback_reason"),
                         "layer \"63\"") &&
          is_json_string(member(graph, "requested_policy"),
                         "sm87_short_positions") &&
          is_json_string(member(graph, "effective_policy"),
                         "sm87_short_positions") &&
          is_json_uint64(member(graph, "slots"), 16U),
      "health response truthfully exposes mapped engine-build route facts "
      "without claiming a DeploymentPlan or per-operator hits");
}

}  // namespace

int main() {
  TestContext test;
  test_evalscope_chat_contract(test);
  test_raw_completion_contract(test);
  test_fail_closed_parameters(test);
  test_serialization(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " OpenAI protocol test(s) failed\n";
    return 1;
  }
  std::cout << "All OpenAI protocol tests passed\n";
  return 0;
}
