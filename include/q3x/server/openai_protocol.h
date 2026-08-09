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

// One measured or explicitly unavailable request phase. Scope is serialized
// with every value so engine/device intervals cannot be mistaken for the
// externally observed EvalScope interval.
struct RequestPhaseEvidence {
  std::string scope;
  std::optional<double> milliseconds;
  std::string unavailable_reason;
};

// Minimal request-level evidence emitted by the evaluation adapter for a
// successful production-path generation. Projection backend remains a
// configured engine fact; per-operator counts below are completed logical
// Prefix operations merged only after their tile synchronized and committed.
struct TargetPrefillWitnessRecord {
  std::string request_id;
  std::string request_body_sha256;
  std::string model;
  OpenAIEndpoint endpoint = OpenAIEndpoint::kCompletions;
  OpenAIPromptKind prompt_kind = OpenAIPromptKind::kTokenIds;
  std::uint64_t prompt_tokens = 0U;
  std::string prompt_token_ids_u32le_sha256;
  std::uint64_t consumed_prompt_tokens = 0U;
  bool full_prompt_consumed = false;
  std::uint64_t completion_tokens = 0U;
  RequestPhaseEvidence queue;
  RequestPhaseEvidence admission;
  RequestPhaseEvidence generation;
  RequestPhaseEvidence pure_prefill;
  RequestPhaseEvidence finalize;
  RequestPhaseEvidence ttft;
  RequestPhaseEvidence first_byte;
  RequestPhaseEvidence decode;
  RequestPhaseEvidence total;
  std::uint32_t requested_prefill_chunk_size = 0U;
  std::uint32_t effective_prefill_chunk_size = 0U;
  std::uint64_t prefix_execution_count = 0U;
  runtime::PrefillRouteEvidence prefill_route_evidence;
  runtime::ProjectionBackend projection_backend =
      runtime::ProjectionBackend::kReference;
  runtime::ReferencePrefillExecutionMode prefill_execution_mode =
      runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled;
  std::uint64_t prefill_logical_panel_count = 0U;
  runtime::RequestMemoryProfile request_memory_profile =
      runtime::RequestMemoryProfile::kLegacyC512;
  bool bounded_submission_window = false;
  std::uint64_t submission_window_retirements = 0U;
  // Empty for legacy/unsealed paths. A non-empty identifier is emitted only
  // after a sealed whole-request generation has completed successfully.
  std::string deployment_plan_id;
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
// Hashes the canonical concatenation of every token id encoded as four
// little-endian bytes. The definition is host-endian independent and streams
// without allocating a prompt-sized copy.
[[nodiscard]] std::string sha256_token_ids_u32le(
    const std::vector<std::uint32_t>& token_ids);
[[nodiscard]] std::string serialize_target_prefill_witness(
    const TargetPrefillWitnessRecord& record);
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
[[nodiscard]] std::string_view to_string(OpenAIEndpoint endpoint) noexcept;
[[nodiscard]] std::string_view to_string(OpenAIPromptKind kind) noexcept;

}  // namespace q3x::server
