#include "q3x/io/json.h"
#include "q3x/server/openai_protocol.h"

#include <cstddef>
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

}  // namespace

int main() {
  TestContext test;
  test_evalscope_chat_contract(test);
  test_raw_completion_contract(test);
  test_target_length_token_id_contract(test);
  test_fail_closed_parameters(test);
  test_serialization(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " OpenAI protocol test(s) failed\n";
    return 1;
  }
  std::cout << "All OpenAI protocol tests passed\n";
  return 0;
}
