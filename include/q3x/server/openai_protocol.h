#pragma once

#include "q3x/runtime/reference_engine.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::server {

enum class OpenAIEndpoint : std::uint8_t {
  kChatCompletions,
  kCompletions,
};

enum class OpenAIPromptKind : std::uint8_t {
  kChatMessages,
  kRawText,
  kTokenIds,
};

struct OpenAIRequest {
  OpenAIEndpoint endpoint = OpenAIEndpoint::kChatCompletions;
  OpenAIPromptKind prompt_kind = OpenAIPromptKind::kChatMessages;
  std::string model;
  std::vector<runtime::ReferenceChatMessage> messages;
  std::string prompt;
  std::vector<std::uint32_t> prompt_token_ids;
  std::uint32_t max_tokens = 16U;
  // EvalScope attaches its run seed to OpenAI requests. Greedy generation is
  // deterministic, so this is retained for audit identity but is a deliberate
  // inference no-op.
  std::optional<std::int64_t> seed;
  bool stream = false;
  bool include_usage = false;
};

struct OpenAIProtocolError {
  int http_status = 400;
  std::string code;
  std::string message;
  std::string param;
};

struct OpenAIParseResult {
  std::optional<OpenAIRequest> value;
  OpenAIProtocolError error;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value();
  }
};

struct OpenAIUsage {
  std::uint64_t prompt_tokens = 0U;
  std::uint64_t completion_tokens = 0U;
  std::uint64_t total_tokens = 0U;
};

enum class OpenAIFinishReason : std::uint8_t {
  kStop,
  kLength,
};

[[nodiscard]] OpenAIParseResult parse_openai_request(
    std::string_view body, OpenAIEndpoint endpoint,
    std::string_view served_model, std::uint32_t maximum_output_tokens,
    std::uint32_t maximum_sequence_length);

[[nodiscard]] std::string serialize_openai_error(
    const OpenAIProtocolError& error);
[[nodiscard]] std::string serialize_models_response(
    std::string_view served_model, std::int64_t created);
[[nodiscard]] std::string serialize_health_response(
    std::string_view served_model);
[[nodiscard]] std::string serialize_chat_completion(
    std::string_view id, std::int64_t created, std::string_view model,
    std::string_view text, OpenAIFinishReason finish_reason,
    const OpenAIUsage& usage);
[[nodiscard]] std::string serialize_text_completion(
    std::string_view id, std::int64_t created, std::string_view model,
    std::string_view text, OpenAIFinishReason finish_reason,
    const OpenAIUsage& usage);
[[nodiscard]] std::string serialize_chat_chunk(
    std::string_view id, std::int64_t created, std::string_view model,
    std::string_view text_delta, bool include_role,
    std::optional<OpenAIFinishReason> finish_reason);
[[nodiscard]] std::string serialize_text_chunk(
    std::string_view id, std::int64_t created, std::string_view model,
    std::string_view text_delta,
    std::optional<OpenAIFinishReason> finish_reason);
[[nodiscard]] std::string serialize_usage_chunk(
    OpenAIEndpoint endpoint, std::string_view id, std::int64_t created,
    std::string_view model, const OpenAIUsage& usage);

[[nodiscard]] std::string_view to_string(
    OpenAIFinishReason reason) noexcept;

}  // namespace q3x::server
