#include "q3x/server/openai_protocol.h"

#include "q3x/core/sha256.h"
#include "q3x/io/json.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace q3x::server {
namespace {

namespace json = q3x::io::json;

OpenAIParseResult failure(std::string code, std::string message,
                          std::string param = {},
                          const int http_status = 400) {
  OpenAIParseResult result;
  result.error.http_status = http_status;
  result.error.code = std::move(code);
  result.error.message = std::move(message);
  result.error.param = std::move(param);
  return result;
}

[[nodiscard]] bool has_only_members(
    const json::Value::Object& object,
    const std::set<std::string_view>& allowed,
    std::string& unexpected) {
  for (const auto& [key, value] : object) {
    (void)value;
    if (allowed.find(key) == allowed.end()) {
      unexpected = key;
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool parse_uint32(const json::Value& value,
                                std::uint32_t& output) noexcept {
  const json::Number* const number = value.as_number();
  if (number == nullptr) {
    return false;
  }
  std::uint64_t parsed = 0U;
  if (!number->to_uint64(parsed) ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  output = static_cast<std::uint32_t>(parsed);
  return true;
}

[[nodiscard]] bool is_number(const json::Value& value,
                             const double expected) noexcept {
  const json::Number* const number = value.as_number();
  double parsed = 0.0;
  return number != nullptr && number->to_double(parsed) &&
         std::isfinite(parsed) && parsed == expected;
}

void append_json_string(std::string& output, const std::string_view value) {
  constexpr char kHex[] = "0123456789abcdef";
  output.push_back('"');
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output += "\\u00";
          output.push_back(kHex[(byte >> 4U) & 0x0fU]);
          output.push_back(kHex[byte & 0x0fU]);
        } else {
          output.push_back(static_cast<char>(byte));
        }
        break;
    }
  }
  output.push_back('"');
}

void append_common_prefix(std::string& output, const std::string_view id,
                          const std::string_view object,
                          const std::int64_t created,
                          const std::string_view model) {
  output += "{\"id\":";
  append_json_string(output, id);
  output += ",\"object\":";
  append_json_string(output, object);
  output += ",\"created\":" + std::to_string(created) + ",\"model\":";
  append_json_string(output, model);
}

void append_usage(std::string& output, const OpenAIUsage& usage) {
  output += "{\"prompt_tokens\":" +
            std::to_string(usage.prompt_tokens) +
            ",\"completion_tokens\":" +
            std::to_string(usage.completion_tokens) +
            ",\"total_tokens\":" +
            std::to_string(usage.total_tokens) + "}";
}

void append_phase_evidence(std::string& output,
                           const std::string_view name,
                           const RequestPhaseEvidence& phase) {
  append_json_string(output, name);
  output += ":{\"available\":";
  const bool available = phase.milliseconds.has_value() &&
                         std::isfinite(*phase.milliseconds) &&
                         *phase.milliseconds >= 0.0;
  output += available ? "true" : "false";
  output += ",\"scope\":";
  append_json_string(output, phase.scope);
  output += ",\"milliseconds\":";
  if (available) {
    output += std::to_string(*phase.milliseconds);
  } else {
    output += "null";
  }
  output += ",\"unavailable_reason\":";
  if (available) {
    output += "null";
  } else if (!phase.unavailable_reason.empty()) {
    append_json_string(output, phase.unavailable_reason);
  } else if (phase.milliseconds.has_value()) {
    append_json_string(output, "invalid_measurement");
  } else {
    append_json_string(output, "not_instrumented");
  }
  output += "}";
}

void append_prefill_route_evidence(
    std::string& output, const runtime::PrefillRouteEvidence& evidence) {
  const bool count_matches =
      evidence.completed_layer_passes == evidence.expected_layer_passes;
  const bool available = evidence.valid && evidence.complete &&
                         !evidence.request_active && count_matches;
  output += "{\"available\":";
  output += available ? "true" : "false";
  output +=
      ",\"scope\":\"request_completed_prefill_logical_operators\",";
  output += "\"complete\":";
  output += evidence.complete ? "true" : "false";
  output += ",\"reason\":";
  if (available) {
    output += "null";
  } else if (!count_matches) {
    append_json_string(output, "unexpected_layer_pass_count");
  } else {
    append_json_string(output, runtime::to_string(evidence.error));
  }
  output += ",\"coverage\":{\"completed_layer_passes\":" +
            std::to_string(evidence.completed_layer_passes) +
            ",\"expected_layer_passes\":" +
            std::to_string(evidence.expected_layer_passes) +
            ",\"layers_per_pass\":64,\"gdn_layers_per_pass\":48,"
            "\"attention_layers_per_pass\":16},\"operators\":{";
  for (std::size_t index = 0U;
       index < runtime::kPrefillOperatorRoleCount; ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    const auto role = static_cast<runtime::PrefillOperatorRole>(index);
    const runtime::PrefillOperatorRouteCounts& counts =
        evidence.operators[index];
    append_json_string(output, runtime::to_string(role));
    output += ":{\"completed_production_hits\":" +
              std::to_string(counts.production_hits) +
              ",\"completed_exact_fallback_hits\":" +
              std::to_string(counts.exact_fallback_hits) +
              ",\"completed_forbidden_hits\":" +
              std::to_string(counts.forbidden_hits) + "}";
  }
  const auto boundary = [&evidence](
                            const runtime::PrefillForbiddenBoundary value) {
    return evidence.forbidden_boundary_hits[static_cast<std::size_t>(value)];
  };
  output +=
      "},\"forbidden_route_hits\":{\"prefix_cache\":" +
      std::to_string(boundary(runtime::PrefillForbiddenBoundary::kPrefixCache)) +
      ",\"mtp\":" +
      std::to_string(boundary(runtime::PrefillForbiddenBoundary::kMtp)) +
      ",\"cublaslt\":" +
      std::to_string(boundary(runtime::PrefillForbiddenBoundary::kCublasLt)) +
      ",\"external_reference\":" +
      std::to_string(
          boundary(runtime::PrefillForbiddenBoundary::kExternalReference)) +
      ",\"approximate_numerics\":" +
      std::to_string(
          boundary(runtime::PrefillForbiddenBoundary::kApproximateNumerics)) +
      "}}";
}

OpenAIParseResult parse_messages(const json::Value& value,
                                 OpenAIRequest& request) {
  const json::Value::Array* const messages = value.as_array();
  if (messages == nullptr || messages->empty()) {
    return failure("invalid_messages",
                   "messages must be a non-empty array", "messages");
  }
  request.messages.reserve(messages->size());
  for (std::size_t index = 0U; index < messages->size(); ++index) {
    const json::Value::Object* const message = (*messages)[index].as_object();
    if (message == nullptr) {
      return failure("invalid_messages",
                     "each message must be an object",
                     "messages[" + std::to_string(index) + "]");
    }
    std::string unexpected;
    if (!has_only_members(*message, {"role", "content"}, unexpected)) {
      return failure("unsupported_message_field",
                     "unsupported message field: " + unexpected,
                     "messages[" + std::to_string(index) + "]." +
                         unexpected);
    }
    const json::Value* const role_value = (*messages)[index].find("role");
    const json::Value* const content_value =
        (*messages)[index].find("content");
    const std::string* const role =
        role_value == nullptr ? nullptr : role_value->as_string();
    const std::string* const content =
        content_value == nullptr ? nullptr : content_value->as_string();
    if (role == nullptr || content == nullptr) {
      return failure("unsupported_message",
                     "message role and content must be strings",
                     "messages[" + std::to_string(index) + "]");
    }
    if (*role != "system" && *role != "user" && *role != "assistant") {
      return failure("unsupported_message_role",
                     "only system, user, and assistant roles are supported",
                     "messages[" + std::to_string(index) + "].role");
    }
    request.messages.push_back({*role, *content});
  }
  std::size_t index = 0U;
  if (request.messages.front().role == "system") {
    index = 1U;
  }
  if (index == request.messages.size()) {
    return failure("invalid_messages",
                   "a system message must be followed by a user message",
                   "messages");
  }
  for (; index < request.messages.size(); ++index) {
    const std::string_view expected =
        ((index - (request.messages.front().role == "system" ? 1U : 0U)) %
         2U) == 0U
            ? "user"
            : "assistant";
    if (request.messages[index].role != expected) {
      return failure("invalid_messages",
                     "messages must alternate user and assistant after an "
                     "optional leading system message",
                     "messages[" + std::to_string(index) + "].role");
    }
  }
  if (request.messages.back().role != "user") {
    return failure("invalid_messages",
                   "the final chat message must have role user",
                   "messages[" +
                       std::to_string(request.messages.size() - 1U) +
                       "].role");
  }
  request.prompt_kind = OpenAIPromptKind::kChatMessages;
  OpenAIParseResult result;
  result.value.emplace();
  return result;
}

OpenAIParseResult parse_prompt(const json::Value& value,
                               OpenAIRequest& request) {
  if (const std::string* const prompt = value.as_string()) {
    if (prompt->empty()) {
      return failure("invalid_prompt", "prompt must not be empty", "prompt");
    }
    request.prompt = *prompt;
    request.prompt_kind = OpenAIPromptKind::kRawText;
  } else if (const json::Value::Array* const ids = value.as_array()) {
    if (ids->empty()) {
      return failure("invalid_prompt",
                     "token-id prompt must not be empty", "prompt");
    }
    request.prompt_token_ids.reserve(ids->size());
    for (std::size_t index = 0U; index < ids->size(); ++index) {
      std::uint32_t id = 0U;
      if (!parse_uint32((*ids)[index], id)) {
        return failure("invalid_prompt",
                       "token-id prompt must contain unsigned integers",
                       "prompt[" + std::to_string(index) + "]");
      }
      if (id >= runtime::kReferenceVocabularySize) {
        return failure("invalid_prompt",
                       "token id is outside the pinned vocabulary",
                       "prompt[" + std::to_string(index) + "]");
      }
      request.prompt_token_ids.push_back(id);
    }
    request.prompt_kind = OpenAIPromptKind::kTokenIds;
  } else {
    return failure("unsupported_prompt",
                   "prompt must be one string or one flat token-id array",
                   "prompt");
  }
  OpenAIParseResult result;
  result.value.emplace();
  return result;
}

}  // namespace

OpenAIParseResult parse_openai_request(
    const std::string_view body, const OpenAIEndpoint endpoint,
    const std::string_view served_model,
    const std::uint32_t maximum_output_tokens,
    const std::uint32_t maximum_sequence_length) {
  if (served_model.empty() || maximum_output_tokens == 0U ||
      maximum_sequence_length == 0U ||
      maximum_sequence_length > runtime::kAbsoluteRequestMaxSequenceLength) {
    return failure("server_configuration_error",
                   "served model, output limit, and sequence capacity are "
                   "not configured",
                   {}, 500);
  }
  json::ParseOptions parse_options;
  parse_options.max_input_bytes = 4U * 1024U * 1024U;
  parse_options.max_nesting_depth = 32U;
  // One exact token-id prompt contributes one JSON value and one container
  // item per admitted token. Keep a small bounded allowance for the root
  // request, protocol fields, and chat-message structure. This makes the JSON
  // budget follow the configured request capacity instead of silently making
  // 100K the protocol ceiling for a runner that admits up to 262,144 steps.
  constexpr std::size_t kProtocolValueAllowance = 256U;
  parse_options.max_values =
      static_cast<std::size_t>(maximum_sequence_length) +
      kProtocolValueAllowance;
  parse_options.max_container_items = parse_options.max_values;
  json::ParseResult parsed = json::parse(body, parse_options);
  if (!parsed) {
    return failure("invalid_json", "request body is not valid bounded JSON");
  }
  const json::Value::Object* const root = parsed.value->as_object();
  if (root == nullptr) {
    return failure("invalid_request", "request body must be a JSON object");
  }

  const std::set<std::string_view> common = {
      "model", "max_tokens", "max_completion_tokens", "temperature",
      "top_p", "n", "seed", "stream", "stream_options", "stop"};
  std::set<std::string_view> allowed = common;
  allowed.insert(endpoint == OpenAIEndpoint::kChatCompletions
                     ? "messages"
                     : "prompt");
  std::string unexpected;
  if (!has_only_members(*root, allowed, unexpected)) {
    return failure("unsupported_parameter",
                   "unsupported request parameter: " + unexpected,
                   unexpected);
  }

  OpenAIRequest request;
  request.endpoint = endpoint;
  request.max_tokens = std::min<std::uint32_t>(16U, maximum_output_tokens);

  const json::Value* const model_value = parsed.value->find("model");
  const std::string* const model =
      model_value == nullptr ? nullptr : model_value->as_string();
  if (model == nullptr || model->empty()) {
    return failure("invalid_model", "model must be a non-empty string",
                   "model");
  }
  if (*model != served_model) {
    return failure("model_not_found", "the requested model is not served",
                   "model", 404);
  }
  request.model = *model;

  const json::Value* const payload = parsed.value->find(
      endpoint == OpenAIEndpoint::kChatCompletions ? "messages" : "prompt");
  if (payload == nullptr) {
    return failure("missing_parameter",
                   endpoint == OpenAIEndpoint::kChatCompletions
                       ? "messages is required"
                       : "prompt is required",
                   endpoint == OpenAIEndpoint::kChatCompletions
                       ? "messages"
                       : "prompt");
  }
  OpenAIParseResult prompt_result =
      endpoint == OpenAIEndpoint::kChatCompletions
          ? parse_messages(*payload, request)
          : parse_prompt(*payload, request);
  if (!prompt_result) {
    return prompt_result;
  }

  const json::Value* const max_tokens = parsed.value->find("max_tokens");
  const json::Value* const max_completion_tokens =
      parsed.value->find("max_completion_tokens");
  if (max_tokens != nullptr && max_completion_tokens != nullptr) {
    return failure("conflicting_parameters",
                   "specify only one of max_tokens and max_completion_tokens",
                   "max_tokens");
  }
  const json::Value* const selected_max =
      max_tokens != nullptr ? max_tokens : max_completion_tokens;
  if (selected_max == nullptr) {
    return failure("missing_parameter",
                   "max_tokens or max_completion_tokens is required by "
                   "this bounded evaluation gateway",
                   "max_tokens");
  }
  if (!parse_uint32(*selected_max, request.max_tokens) ||
       request.max_tokens == 0U ||
       request.max_tokens > maximum_output_tokens) {
    return failure("invalid_max_tokens",
                   "max_tokens must be within the configured positive limit",
                   max_tokens != nullptr ? "max_tokens"
                                         : "max_completion_tokens");
  }

  if (request.prompt_kind == OpenAIPromptKind::kTokenIds) {
    const std::uint64_t required_steps =
        static_cast<std::uint64_t>(request.prompt_token_ids.size()) +
        static_cast<std::uint64_t>(request.max_tokens) - 1U;
    if (required_steps > maximum_sequence_length) {
      return failure(
          "context_length_exceeded",
          "prompt tokens plus requested output exceed the configured "
          "sequence capacity",
          "prompt");
    }
  }

  const json::Value* const temperature = parsed.value->find("temperature");
  if (temperature == nullptr) {
    return failure("missing_parameter",
                   "temperature=0 must be explicit because this gateway "
                   "does not implement the OpenAI sampling default",
                   "temperature");
  }
  if (!is_number(*temperature, 0.0)) {
    return failure("unsupported_sampling",
                   "only greedy temperature=0 is supported",
                   "temperature");
  }
  if (const json::Value* const top_p = parsed.value->find("top_p")) {
    if (!is_number(*top_p, 1.0)) {
      return failure("unsupported_sampling", "only top_p=1 is supported",
                     "top_p");
    }
  }
  if (const json::Value* const n = parsed.value->find("n")) {
    std::uint32_t value = 0U;
    if (!parse_uint32(*n, value) || value != 1U) {
      return failure("unsupported_batch", "only n=1 is supported", "n");
    }
  }
  if (const json::Value* const seed = parsed.value->find("seed")) {
    const json::Number* const number = seed->as_number();
    std::int64_t value = 0;
    if (number == nullptr || !number->to_int64(value)) {
      return failure("invalid_seed", "seed must be a signed 64-bit integer",
                     "seed");
    }
    request.seed = value;
  }
  if (const json::Value* const stop = parsed.value->find("stop")) {
    if (!stop->is_null()) {
      return failure("unsupported_stop",
                     "custom stop sequences are not supported", "stop");
    }
  }
  if (const json::Value* const stream = parsed.value->find("stream")) {
    const bool* const enabled = stream->as_bool();
    if (enabled == nullptr) {
      return failure("invalid_stream", "stream must be a boolean", "stream");
    }
    request.stream = *enabled;
  }
  if (const json::Value* const stream_options =
          parsed.value->find("stream_options")) {
    if (!request.stream) {
      return failure("invalid_stream_options",
                     "stream_options requires stream=true",
                     "stream_options");
    }
    const json::Value::Object* const object = stream_options->as_object();
    if (object == nullptr) {
      return failure("invalid_stream_options",
                     "stream_options must be an object", "stream_options");
    }
    if (!has_only_members(*object, {"include_usage"}, unexpected)) {
      return failure("unsupported_parameter",
                     "unsupported stream option: " + unexpected,
                     "stream_options." + unexpected);
    }
    if (const json::Value* const include_usage =
            stream_options->find("include_usage")) {
      const bool* const enabled = include_usage->as_bool();
      if (enabled == nullptr) {
        return failure("invalid_stream_options",
                       "include_usage must be a boolean",
                       "stream_options.include_usage");
      }
      request.include_usage = *enabled;
    }
  }

  OpenAIParseResult result;
  result.value.emplace(std::move(request));
  return result;
}

std::string serialize_openai_error(const OpenAIProtocolError& error) {
  std::string output = "{\"error\":{\"message\":";
  append_json_string(output, error.message);
  output += ",\"type\":\"invalid_request_error\",\"param\":";
  if (error.param.empty()) {
    output += "null";
  } else {
    append_json_string(output, error.param);
  }
  output += ",\"code\":";
  append_json_string(output, error.code);
  output += "}}";
  return output;
}

std::string serialize_models_response(const std::string_view served_model,
                                      const std::int64_t created) {
  std::string output = "{\"object\":\"list\",\"data\":[{\"id\":";
  append_json_string(output, served_model);
  output += ",\"object\":\"model\",\"created\":" +
            std::to_string(created) + ",\"owned_by\":\"qwen3x-orin\"}]}";
  return output;
}

std::string serialize_health_response(const std::string_view served_model) {
  std::string output = "{\"status\":\"ok\",\"ready\":true,\"model\":";
  append_json_string(output, served_model);
  output += "}";
  return output;
}

std::string sha256_token_ids_u32le(
    const std::vector<std::uint32_t>& token_ids) {
  q3x::core::Sha256 hash;
  std::array<std::uint8_t, 4096U> bytes{};
  std::size_t used = 0U;
  for (const std::uint32_t token_id : token_ids) {
    bytes[used++] = static_cast<std::uint8_t>(token_id & 0xffU);
    bytes[used++] = static_cast<std::uint8_t>((token_id >> 8U) & 0xffU);
    bytes[used++] = static_cast<std::uint8_t>((token_id >> 16U) & 0xffU);
    bytes[used++] = static_cast<std::uint8_t>((token_id >> 24U) & 0xffU);
    if (used == bytes.size()) {
      (void)hash.update(bytes.data(), used);
      used = 0U;
    }
  }
  if (used != 0U) {
    (void)hash.update(bytes.data(), used);
  }
  return hash.finalize().hex();
}

std::string serialize_target_prefill_witness(
    const TargetPrefillWitnessRecord& record) {
  std::string output =
      "{\"record\":\"target-prefill-witness-v1\",\"schema_version\":1,"
      "\"request\":{\"id\":";
  append_json_string(output, record.request_id);
  output += ",\"body_sha256\":";
  append_json_string(output, record.request_body_sha256);
  output += "},\"model\":";
  append_json_string(output, record.model);
  output += ",\"endpoint\":";
  append_json_string(output, to_string(record.endpoint));
  output += ",\"prompt\":{\"kind\":";
  append_json_string(output, to_string(record.prompt_kind));
  output += ",\"tokens\":" + std::to_string(record.prompt_tokens) +
            ",\"token_ids_u32le_sha256\":";
  append_json_string(output, record.prompt_token_ids_u32le_sha256);
  output += ",\"consumed_tokens\":" +
            std::to_string(record.consumed_prompt_tokens) +
            ",\"fully_consumed\":";
  output += record.full_prompt_consumed ? "true" : "false";
  output += "},\"completion\":{\"tokens\":" +
            std::to_string(record.completion_tokens) + "},\"timing\":{";
  append_phase_evidence(output, "queue", record.queue);
  output += ',';
  append_phase_evidence(output, "admission", record.admission);
  output += ',';
  append_phase_evidence(output, "generation", record.generation);
  output += ',';
  append_phase_evidence(output, "pure_prefill", record.pure_prefill);
  output += ',';
  append_phase_evidence(output, "finalize", record.finalize);
  output += ',';
  append_phase_evidence(output, "ttft", record.ttft);
  output += ',';
  append_phase_evidence(output, "first_byte", record.first_byte);
  output += ',';
  append_phase_evidence(output, "decode", record.decode);
  output += ',';
  append_phase_evidence(output, "total", record.total);
  output += "},\"prefill\":{\"requested_chunk\":" +
            std::to_string(record.requested_prefill_chunk_size) +
            ",\"effective_chunk\":" +
            std::to_string(record.effective_prefill_chunk_size) +
            ",\"prefix_execution_count\":" +
            std::to_string(record.prefix_execution_count) +
            "},\"route\":{\"scope\":\"request_witness\","
            "\"projection_backend\":{\"available\":true,"
            "\"scope\":\"configured_engine_fact\",\"value\":";
  append_json_string(output, runtime::to_string(record.projection_backend));
  output += "},\"deployment_plan\":{\"available\":false,\"reason\":"
            "\"not_implemented\"},\"per_operator_route_hits\":";
  append_prefill_route_evidence(output, record.prefill_route_evidence);
  output +=
      ","
      "\"cache_hits\":{\"available\":false,\"reason\":"
      "\"not_instrumented\"},"
      "\"disabled_boundaries\":{\"scope\":\"production_contract\","
      "\"prefix_cache\":true,\"mtp\":true,"
      "\"cublaslt_production\":true,\"approximate_numerics\":true}}}";
  return output;
}

std::string serialize_chat_completion(
    const std::string_view id, const std::int64_t created,
    const std::string_view model, const std::string_view text,
    const OpenAIFinishReason finish_reason, const OpenAIUsage& usage) {
  std::string output;
  append_common_prefix(output, id, "chat.completion", created, model);
  output += ",\"choices\":[{\"index\":0,\"message\":{\"role\":"
            "\"assistant\",\"content\":";
  append_json_string(output, text);
  output += "},\"logprobs\":null,\"finish_reason\":";
  append_json_string(output, to_string(finish_reason));
  output += "}],\"usage\":";
  append_usage(output, usage);
  output += "}";
  return output;
}

std::string serialize_text_completion(
    const std::string_view id, const std::int64_t created,
    const std::string_view model, const std::string_view text,
    const OpenAIFinishReason finish_reason, const OpenAIUsage& usage) {
  std::string output;
  append_common_prefix(output, id, "text_completion", created, model);
  output += ",\"choices\":[{\"index\":0,\"text\":";
  append_json_string(output, text);
  output += ",\"logprobs\":null,\"finish_reason\":";
  append_json_string(output, to_string(finish_reason));
  output += "}],\"usage\":";
  append_usage(output, usage);
  output += "}";
  return output;
}

std::string serialize_chat_chunk(
    const std::string_view id, const std::int64_t created,
    const std::string_view model, const std::string_view text_delta,
    const bool include_role,
    const std::optional<OpenAIFinishReason> finish_reason) {
  std::string output;
  append_common_prefix(output, id, "chat.completion.chunk", created, model);
  output += ",\"choices\":[{\"index\":0,\"delta\":{";
  if (include_role) {
    output += "\"role\":\"assistant\",";
  }
  output += "\"content\":";
  append_json_string(output, text_delta);
  output += "},\"logprobs\":null,\"finish_reason\":";
  if (finish_reason.has_value()) {
    append_json_string(output, to_string(*finish_reason));
  } else {
    output += "null";
  }
  output += "}]}";
  return output;
}

std::string serialize_text_chunk(
    const std::string_view id, const std::int64_t created,
    const std::string_view model, const std::string_view text_delta,
    const std::optional<OpenAIFinishReason> finish_reason) {
  std::string output;
  append_common_prefix(output, id, "text_completion", created, model);
  output += ",\"choices\":[{\"index\":0,\"text\":";
  append_json_string(output, text_delta);
  output += ",\"logprobs\":null,\"finish_reason\":";
  if (finish_reason.has_value()) {
    append_json_string(output, to_string(*finish_reason));
  } else {
    output += "null";
  }
  output += "}]}";
  return output;
}

std::string serialize_usage_chunk(
    const OpenAIEndpoint endpoint, const std::string_view id,
    const std::int64_t created, const std::string_view model,
    const OpenAIUsage& usage) {
  std::string output;
  append_common_prefix(output, id,
                       endpoint == OpenAIEndpoint::kChatCompletions
                           ? "chat.completion.chunk"
                           : "text_completion",
                       created, model);
  output += ",\"choices\":[],\"usage\":";
  append_usage(output, usage);
  output += "}";
  return output;
}

std::string_view to_string(const OpenAIFinishReason reason) noexcept {
  switch (reason) {
    case OpenAIFinishReason::kStop:
      return "stop";
    case OpenAIFinishReason::kLength:
      return "length";
  }
  return "unknown";
}

std::string_view to_string(const OpenAIEndpoint endpoint) noexcept {
  switch (endpoint) {
    case OpenAIEndpoint::kChatCompletions:
      return "/v1/chat/completions";
    case OpenAIEndpoint::kCompletions:
      return "/v1/completions";
  }
  return "unknown";
}

std::string_view to_string(const OpenAIPromptKind kind) noexcept {
  switch (kind) {
    case OpenAIPromptKind::kChatMessages:
      return "chat_messages";
    case OpenAIPromptKind::kRawText:
      return "raw_text";
    case OpenAIPromptKind::kTokenIds:
      return "token_ids";
  }
  return "unknown";
}

}  // namespace q3x::server
