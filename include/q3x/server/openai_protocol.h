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

// A bounded, read-only account of facts already established while building
// the resident engine. This deliberately stops short of a DeploymentPlan:
// the current runner does not yet retain per-shape tactic identities or
// per-operator route-hit counters. Consumers must therefore interpret the
// projection backend as an engine-wide dispatcher policy, not proof that
// every projection launched a specialized kernel.
struct ExecutionSidecarFact {
  bool enabled = false;
  std::uint64_t attached_artifacts = 0U;
  std::uint64_t fallback_artifacts = 0U;
  std::uint64_t bytes = 0U;
  // Empty means that the engine published no fallback reason. It does not
  // distinguish an unrequested inventory from an inventory with no failure.
  std::string fallback_reason;
};

struct ExecutionRouteAttestation {
  std::uint32_t schema_version = 1U;
  bool deployment_plan_available = false;
  bool per_operator_route_hits_available = false;
  // Number of generation attempts after which this snapshot was refreshed.
  // Zero is the engine-ready snapshot before the first request.
  std::uint64_t generation_attempts_observed = 0U;
  runtime::ProjectionBackend projection_backend =
      runtime::ProjectionBackend::kReference;
  std::uint32_t max_sequence_length = 0U;
  std::uint32_t prefill_chunk_size = 0U;
  std::uint64_t request_arena_bytes = 0U;
  ExecutionSidecarFact fp8_output;
  ExecutionSidecarFact nvfp4_down_scale6;
  ExecutionSidecarFact fp8_prefill_qkv;
  ExecutionSidecarFact fp8_prefill_supermatrix;
  ExecutionSidecarFact nvfp4_marlin_prefill;
  runtime::ReferenceDecodeGraphCachePolicy decode_graph_requested_policy =
      runtime::ReferenceDecodeGraphCachePolicy::kDisabled;
  runtime::ReferenceDecodeGraphCachePolicy decode_graph_effective_policy =
      runtime::ReferenceDecodeGraphCachePolicy::kDisabled;
  std::uint32_t decode_graph_first_position = 0U;
  std::uint32_t decode_graph_last_position = 0U;
  std::uint64_t decode_graph_slots = 0U;
  std::string decode_graph_fallback_reason;
};

enum class OpenAIFinishReason : std::uint8_t {
  kStop,
  kLength,
};

[[nodiscard]] OpenAIParseResult parse_openai_request(
    std::string_view body, OpenAIEndpoint endpoint,
    std::string_view served_model, std::uint32_t maximum_output_tokens);

[[nodiscard]] std::string serialize_openai_error(
    const OpenAIProtocolError& error);
[[nodiscard]] std::string serialize_models_response(
    std::string_view served_model, std::int64_t created);
[[nodiscard]] ExecutionRouteAttestation make_execution_route_attestation(
    const runtime::ReferenceEngineLoadStats& load,
    std::uint64_t generation_attempts_observed = 0U);
[[nodiscard]] std::string serialize_health_response(
    std::string_view served_model,
    const ExecutionRouteAttestation& attestation);
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
