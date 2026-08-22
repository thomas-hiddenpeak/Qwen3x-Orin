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

void append_production_identity(
    std::string& output, const OpenAIProductionIdentity& production) {
  output += "{\"profile\":";
  append_json_string(output, production.profile_id);
  output += ",\"capacity\":{\"target_prompt_tokens\":";
  output += std::to_string(production.target_prompt_tokens);
  output += ",\"maximum_output_tokens\":";
  output += std::to_string(production.maximum_output_tokens);
  output += ",\"max_sequence_length\":";
  output += std::to_string(production.max_sequence_length);
  output += ",\"request_arena_bytes\":";
  output += std::to_string(production.request_arena_bytes);
  output += "},\"prefill\":{\"supermatrix_projections\":";
  output += std::to_string(production.prefill_supermatrix_projections);
  output += ",\"supermatrix_sidecar_bytes\":";
  output += std::to_string(production.prefill_supermatrix_sidecar_bytes);
  output += "},\"decode\":{\"route\":";
  append_json_string(output, production.decode_route_id);
  output += ",\"fp8_output_layers\":";
  output += std::to_string(production.decode_fp8_output_layers);
  output += ",\"fp8_output_sidecar_bytes\":";
  output += std::to_string(production.decode_fp8_output_sidecar_bytes);
  output += ",\"gate_up_layers\":";
  output += std::to_string(production.decode_gate_up_layers);
  output += ",\"gate_up_sidecar_bytes\":";
  output += std::to_string(production.decode_gate_up_sidecar_bytes);
  output += ",\"down_scale6_layers\":";
  output += std::to_string(production.decode_down_scale6_layers);
  output += ",\"down_scale6_sidecar_bytes\":";
  output += std::to_string(production.decode_down_scale6_sidecar_bytes);
  output += ",\"down_consumer_order_layers\":";
  output += std::to_string(production.decode_down_consumer_order_layers);
  output += ",\"down_consumer_order_sidecar_bytes\":";
  output +=
      std::to_string(production.decode_down_consumer_order_sidecar_bytes);
  output += ",\"retained_sidecar_bytes\":";
  output += std::to_string(production.decode_retained_sidecar_bytes);
  output += ",\"graph_cache\":{\"first_position\":";
  output += std::to_string(production.decode_graph_first_position);
  output += ",\"last_position\":";
  output += std::to_string(production.decode_graph_last_position);
  output += ",\"slots\":";
  output += std::to_string(production.decode_graph_slots);
  output += "}";
  output += "},\"retained_acceleration_sidecar_bytes\":";
  output += std::to_string(production.retained_acceleration_sidecar_bytes);
  output += ",\"BUILD_TESTING\":";
  output += production.build_testing ? "true" : "false";
  output += ",\"production_eligible\":";
  output += production.production_eligible ? "true" : "false";
  output += ",\"release_qualified\":";
  output += production.release_qualified ? "true" : "false";
  output += "}";
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

[[nodiscard]] bool valid_request_state_reset_receipt(
    const runtime::RequestStateResetReceipt& receipt) noexcept {
  if (!std::isfinite(receipt.milliseconds) || receipt.milliseconds < 0.0) {
    return false;
  }
  if (receipt.mode == runtime::RequestStateResetMode::kAlreadyClean) {
    return receipt.cleared_positions == 0U && receipt.zeroed_bytes == 0U;
  }
  if ((receipt.mode !=
           runtime::RequestStateResetMode::kCommittedDirtyPrefix &&
       receipt.mode != runtime::RequestStateResetMode::kConservativeFull) ||
      receipt.cleared_positions == 0U) {
    return false;
  }
  constexpr std::uint64_t kFixedBytes =
      runtime::kRequestConvStateBytes + runtime::kRequestGdnStateBytes;
  if (receipt.cleared_positions >
      (std::numeric_limits<std::uint64_t>::max() - kFixedBytes) /
          runtime::kRequestKvBytesPerToken) {
    return false;
  }
  return receipt.zeroed_bytes ==
         kFixedBytes +
             static_cast<std::uint64_t>(receipt.cleared_positions) *
                 runtime::kRequestKvBytesPerToken;
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

[[nodiscard]] bool complete_vllm_marlin_parity_layer_receipts(
    const TargetPrefillWitnessRecord& record) noexcept {
  if (record.vllm_marlin_parity_layer_completion_receipt_count !=
      runtime::kReferenceDecoderLayerCount) {
    return false;
  }
  for (std::size_t layer = 0U; layer < runtime::kReferenceDecoderLayerCount;
       ++layer) {
    const runtime::ReferenceP40VllmMarlinParityLayerCompletionReceipt& receipt =
        record.vllm_marlin_parity_layer_completion_receipts[layer];
    if ((receipt.packed & 0x80000000U) != 0U ||
        receipt.layer_index() != layer ||
        receipt.request_lock_clear_operations() != (layer == 0U ? 1U : 0U) ||
        receipt.gate_up_full_m1024_launches() != runtime::
            kLayerMajorPrefillVllmMarlinParityFullSegmentsPerProjection ||
        receipt.gate_up_split_m64_launches() != runtime::
            kLayerMajorPrefillVllmMarlinParityTailSegmentsPerProjection ||
        receipt.standalone_silu_launches() != runtime::
            kLayerMajorPrefillVllmMarlinParityStandaloneSiluLaunchesPerLayer ||
        receipt.down_full_m1024_launches() != runtime::
            kLayerMajorPrefillVllmMarlinParityFullSegmentsPerProjection ||
        receipt.down_split_m64_launches() != runtime::
            kLayerMajorPrefillVllmMarlinParityTailSegmentsPerProjection ||
        receipt.standalone_residual_launches() != runtime::
            kLayerMajorPrefillVllmMarlinParityStandaloneResidualLaunchesPerLayer ||
        !receipt.retained_prompt_core_complete() ||
        !receipt.canonical_gate_then_up_bf16_published() ||
        !receipt.activated_bf16_published() ||
        !receipt.down_bf16_published() ||
        !receipt.stable_lock_owner_bound() ||
        !receipt.lock_owner_alias_exclusion_proved() ||
        !receipt.ordered_lock_protocol_completed() ||
        !receipt.request_stream_completion_observed()) {
      return false;
    }
  }
  return true;
}

void append_vllm_marlin_parity_layer_receipts(
    std::string& output, const TargetPrefillWitnessRecord& record) {
  output.push_back('[');
  const std::size_t serialized_count = static_cast<std::size_t>(std::min<
      std::uint64_t>(record.vllm_marlin_parity_layer_completion_receipt_count,
                     runtime::kReferenceDecoderLayerCount));
  for (std::size_t layer = 0U; layer < serialized_count; ++layer) {
    if (layer != 0U) {
      output.push_back(',');
    }
    const runtime::ReferenceP40VllmMarlinParityLayerCompletionReceipt& receipt =
        record.vllm_marlin_parity_layer_completion_receipts[layer];
    output +=
        "{\"layer\":" + std::to_string(receipt.layer_index()) +
        ",\"request_lock_clear_operations\":" +
        std::to_string(receipt.request_lock_clear_operations()) +
        ",\"gate_up\":{\"full_m1024_launches\":" +
        std::to_string(receipt.gate_up_full_m1024_launches()) +
        ",\"split_m64_launches\":" +
        std::to_string(receipt.gate_up_split_m64_launches()) +
        "},\"standalone_silu_launches\":" +
        std::to_string(receipt.standalone_silu_launches()) +
        ",\"down\":{\"full_m1024_launches\":" +
        std::to_string(receipt.down_full_m1024_launches()) +
        ",\"split_m64_launches\":" +
        std::to_string(receipt.down_split_m64_launches()) +
        "},\"standalone_residual_launches\":" +
        std::to_string(receipt.standalone_residual_launches()) +
        ",\"publication\":{\"retained_prompt_core_complete\":";
    output += receipt.retained_prompt_core_complete() ? "true" : "false";
    output += ",\"canonical_gate_then_up_bf16_published\":";
    output += receipt.canonical_gate_then_up_bf16_published() ? "true"
                                                              : "false";
    output += ",\"activated_bf16_published\":";
    output += receipt.activated_bf16_published() ? "true" : "false";
    output += ",\"down_bf16_published\":";
    output += receipt.down_bf16_published() ? "true" : "false";
    output += "},\"lock_lifetime\":{\"stable_owner_bound\":";
    output += receipt.stable_lock_owner_bound() ? "true" : "false";
    output += ",\"alias_exclusion_proved\":";
    output +=
        receipt.lock_owner_alias_exclusion_proved() ? "true" : "false";
    output += ",\"ordered_protocol_completed\":";
    output += receipt.ordered_lock_protocol_completed() ? "true" : "false";
    output += "},\"request_stream_completion_observed\":";
    output +=
        receipt.request_stream_completion_observed() ? "true" : "false";
    output.push_back('}');
  }
  output.push_back(']');
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

bool constant_time_bearer_authorization_matches(
    const std::string_view authorization,
    const std::string_view api_key) noexcept {
  constexpr std::string_view prefix = "Bearer ";
  const std::size_t expected_size = prefix.size() + api_key.size();
  std::size_t difference = authorization.size() ^ expected_size;
  difference |= static_cast<std::size_t>(api_key.empty());
  for (std::size_t index = 0U; index < expected_size; ++index) {
    const unsigned char expected = static_cast<unsigned char>(
        index < prefix.size() ? prefix[index]
                              : api_key[index - prefix.size()]);
    const unsigned char actual =
        index < authorization.size()
            ? static_cast<unsigned char>(authorization[index])
            : 0U;
    difference |= static_cast<std::size_t>(actual ^ expected);
  }
  return difference == 0U;
}

std::string serialize_models_response(const std::string_view served_model,
                                      const std::int64_t created,
                                      const OpenAIProductionIdentity&
                                          production) {
  std::string output = "{\"object\":\"list\",\"data\":[{\"id\":";
  append_json_string(output, served_model);
  output += ",\"object\":\"model\",\"created\":" +
            std::to_string(created) +
            ",\"owned_by\":\"qwen3x-orin\",\"q3x_production\":";
  append_production_identity(output, production);
  output += "}]}";
  return output;
}

std::string serialize_health_response(
    const std::string_view served_model,
    const OpenAIProductionIdentity& production) {
  std::string output = "{\"status\":\"ok\",\"ready\":true,\"model\":";
  append_json_string(output, served_model);
  output += ",\"q3x_production\":";
  append_production_identity(output, production);
  output += "}";
  return output;
}

std::string serialize_unhealthy_health_response(
    const std::string_view served_model,
    const OpenAIProductionIdentity& production,
    const std::string_view reason) {
  std::string output =
      "{\"status\":\"unhealthy\",\"ready\":false,\"model\":";
  append_json_string(output, served_model);
  output += ",\"reason\":";
  append_json_string(output, reason);
  output += ",\"q3x_production\":";
  append_production_identity(output, production);
  output += "}";
  return output;
}

std::string serialize_p40_whole_core_v10_models_response(
    const std::string_view served_model, const std::int64_t created) {
  std::string output = "{\"object\":\"list\",\"data\":[{\"id\":";
  append_json_string(output, served_model);
  output += ",\"object\":\"model\",\"created\":" +
            std::to_string(created) +
            ",\"owned_by\":\"qwen3x-orin\",";
  output +=
      "\"q3x_development_route\":{\"id\":\"p40-whole-core-v10\",";
  output +=
      "\"numerical_contract\":{\"qualified\":false,\"reason\":";
  append_json_string(
      output,
      "known-p513-full-state-mismatch-in-inherited-flashinfer-arithmetic");
  output += ",\"p40000_full_state\":\"not_measured\"},"
            "\"evidence\":{\"artifact\":";
  append_json_string(
      output,
      "qwen36-27b-prefill-p40k-whole-core-direction-2026-08-10.json");
  output += ",\"sha256\":";
  append_json_string(
      output,
      "b0847b1f1965570d7311f6b73c137f11e34bc65c893ae76783583bb4fcd7a9fa");
  output += "},"
            "\"release_qualified\":false,\"production_eligible\":false}}]}";
  return output;
}

std::string serialize_p40_whole_core_v10_health_response(
    const std::string_view served_model) {
  std::string output = "{\"status\":\"ok\",\"ready\":true,\"model\":";
  append_json_string(output, served_model);
  output +=
      ",\"development_route\":\"p40-whole-core-v10\",";
  output +=
      "\"numerical_contract\":{\"qualified\":false,\"reason\":";
  append_json_string(
      output,
      "known-p513-full-state-mismatch-in-inherited-flashinfer-arithmetic");
  output += ",\"p40000_full_state\":\"not_measured\",\"evidence\":{"
            "\"artifact\":";
  append_json_string(
      output,
      "qwen36-27b-prefill-p40k-whole-core-direction-2026-08-10.json");
  output += ",\"sha256\":";
  append_json_string(
      output,
      "b0847b1f1965570d7311f6b73c137f11e34bc65c893ae76783583bb4fcd7a9fa");
  output += "}},\"release_qualified\":false,"
            "\"production_eligible\":false}";
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
  const bool sealed = !record.deployment_plan_id.empty();
  const bool production_reset_v16 =
      !sealed && record.request_state_reset.has_value() &&
      record.prefill_execution_mode ==
          runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled &&
      record.request_memory_profile ==
          runtime::RequestMemoryProfile::kLegacyC512 &&
      valid_request_state_reset_receipt(*record.request_state_reset);
  const bool candidate_q64_v3 =
      record.deployment_plan_id ==
      runtime::kLayerMajorNativeGroupQ64PanelDeploymentPlanId;
  const bool candidate_q128_v3 =
      record.deployment_plan_id ==
      runtime::kLayerMajorNativeGroupQ128V4PanelDeploymentPlanId;
  const bool candidate_v3 = candidate_q64_v3 || candidate_q128_v3;
  const bool projection_candidate_v4 =
      record.deployment_plan_id ==
          runtime::kLayerMajorSegmentedMarlinProjectionDeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorSegmentedMarlinProjectionGroupQ64DeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorSegmentedMarlinProjectionGroupQ128V4DeploymentPlanId;
  const bool projection_candidate_uses_q64_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorSegmentedMarlinProjectionGroupQ64DeploymentPlanId;
  const bool projection_candidate_uses_q128_v4_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorSegmentedMarlinProjectionGroupQ128V4DeploymentPlanId;
  const bool native_large_m_candidate_v5 =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeQuantizedLargeMProjectionDeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeQuantizedLargeMProjectionGroupQ64DeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeQuantizedLargeMProjectionGroupQ128V4DeploymentPlanId;
  const bool native_large_m_candidate_uses_q64_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeQuantizedLargeMProjectionGroupQ64DeploymentPlanId;
  const bool native_large_m_candidate_uses_q128_v4_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeQuantizedLargeMProjectionGroupQ128V4DeploymentPlanId;
  const bool flashinfer_exact_candidate_v6 =
      record.deployment_plan_id ==
          runtime::kLayerMajorNativeFlashInferExactPanelDeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorSegmentedMarlinProjectionFlashInferExactDeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeQuantizedLargeMProjectionFlashInferExactDeploymentPlanId;
  const bool flashinfer_exact_candidate_uses_segmented_projection =
      record.deployment_plan_id == runtime::
          kLayerMajorSegmentedMarlinProjectionFlashInferExactDeploymentPlanId;
  const bool flashinfer_exact_candidate_uses_native_large_m_projection =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeQuantizedLargeMProjectionFlashInferExactDeploymentPlanId;
  const bool true_large_m_nvfp4_candidate_v7 =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4TrueLargeMProjectionDeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4TrueLargeMProjectionGroupQ64DeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4TrueLargeMProjectionGroupQ128V4DeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4TrueLargeMProjectionFlashInferExactDeploymentPlanId;
  const bool true_large_m_nvfp4_candidate_uses_q64_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4TrueLargeMProjectionGroupQ64DeploymentPlanId;
  const bool true_large_m_nvfp4_candidate_uses_q128_v4_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4TrueLargeMProjectionGroupQ128V4DeploymentPlanId;
  const bool true_large_m_nvfp4_candidate_uses_flashinfer_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4TrueLargeMProjectionFlashInferExactDeploymentPlanId;
  const bool g2_d2_nvfp4_candidate_v8 =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4G2D2LargeMProjectionDeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4G2D2LargeMProjectionGroupQ64DeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4G2D2LargeMProjectionGroupQ128V4DeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4G2D2LargeMProjectionFlashInferExactDeploymentPlanId;
  const bool g2_d2_nvfp4_candidate_uses_q64_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4G2D2LargeMProjectionGroupQ64DeploymentPlanId;
  const bool g2_d2_nvfp4_candidate_uses_q128_v4_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4G2D2LargeMProjectionGroupQ128V4DeploymentPlanId;
  const bool g2_d2_nvfp4_candidate_uses_flashinfer_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4G2D2LargeMProjectionFlashInferExactDeploymentPlanId;
  const bool persistent_p40_candidate_v9 =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4PersistentP40MlpDeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4PersistentP40MlpGroupQ64DeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4PersistentP40MlpGroupQ128V4DeploymentPlanId ||
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4PersistentP40MlpFlashInferExactDeploymentPlanId;
  const bool persistent_p40_candidate_uses_q64_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4PersistentP40MlpGroupQ64DeploymentPlanId;
  const bool persistent_p40_candidate_uses_q128_v4_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4PersistentP40MlpGroupQ128V4DeploymentPlanId;
  const bool persistent_p40_candidate_uses_flashinfer_attention =
      record.deployment_plan_id == runtime::
          kLayerMajorNativeNvfp4PersistentP40MlpFlashInferExactDeploymentPlanId;
  const bool prompt_wide_p40_whole_core_candidate_v10 =
      record.deployment_plan_id ==
      runtime::kLayerMajorNativePromptWideP40WholeCoreDeploymentPlanId;
  const bool p40_projection_reset_candidate_v11 =
      record.deployment_plan_id == runtime::
          kLayerMajorNativePromptWideP40ProjectionResetDeploymentPlanId;
  const bool p40_packed_projection_candidate_v13 =
      record.deployment_plan_id == runtime::
          kLayerMajorNativePromptWideP40PackedProjectionDeploymentPlanId;
  const bool p40_packed_nvfp4_v2_candidate_v14 =
      record.deployment_plan_id == runtime::
          kLayerMajorNativePromptWideP40PackedNvfp4V2DeploymentPlanId;
  const bool p40_vllm_marlin_parity_candidate_v15 =
      record.deployment_plan_id == runtime::
          kLayerMajorNativePromptWideP40VllmMarlinParityDeploymentPlanId;
  constexpr std::uint64_t kLayerCount = 64U;
  constexpr std::uint64_t kLinearAttentionLayerCount = 48U;
  constexpr std::uint64_t kAttentionLayerCount = 16U;
  constexpr std::uint64_t kFp8ProjectionCountPerPanel = 208U;
  const bool g2_d2_panel_count_valid =
      record.operator_panel_executor_hits != 0U &&
      record.operator_panel_executor_hits % kLayerCount == 0U;
  const std::uint64_t g2_d2_panel_count =
      g2_d2_panel_count_valid
          ? record.operator_panel_executor_hits / kLayerCount
          : 0U;
  const bool g2_d2_panel_coverage_complete =
      g2_d2_panel_count_valid &&
      record.prefill_logical_panel_count == g2_d2_panel_count &&
      record.prefill_route_evidence.valid &&
      record.prefill_route_evidence.complete &&
      !record.prefill_route_evidence.request_active &&
      record.prefill_route_evidence.completed_layer_passes ==
          g2_d2_panel_count &&
      record.prefill_route_evidence.expected_layer_passes ==
          g2_d2_panel_count;
  const bool g2_d2_expected_fp8_count_representable =
      g2_d2_panel_count <= std::numeric_limits<std::uint64_t>::max() /
                               kFp8ProjectionCountPerPanel;
  const std::uint64_t g2_d2_expected_fp8_projection_hits =
      g2_d2_expected_fp8_count_representable
          ? g2_d2_panel_count * kFp8ProjectionCountPerPanel
          : 0U;
  const bool g2_d2_expected_nvfp4_count_representable =
      record.operator_panel_executor_hits <=
      std::numeric_limits<std::uint64_t>::max() / 2U;
  const std::uint64_t g2_d2_expected_nvfp4_projection_hits =
      g2_d2_expected_nvfp4_count_representable
          ? 2U * record.operator_panel_executor_hits
          : 0U;
  const bool g2_d2_expected_attention_count_representable =
      g2_d2_panel_count <= std::numeric_limits<std::uint64_t>::max() /
                               kAttentionLayerCount;
  const std::uint64_t g2_d2_expected_native_attention_hits =
      g2_d2_expected_attention_count_representable
          ? g2_d2_panel_count * kAttentionLayerCount
          : 0U;
  const bool g2_d2_attention_counts_complete =
      g2_d2_expected_attention_count_representable &&
      (g2_d2_nvfp4_candidate_uses_q64_attention
           ? record.native_group_q64_panel_hits ==
                     g2_d2_expected_native_attention_hits &&
                 record.native_group_q128_v4_panel_hits == 0U &&
                 record.native_flashinfer_exact_panel_hits == 0U &&
                 record.generic_qt2_hits == 0U
       : g2_d2_nvfp4_candidate_uses_q128_v4_attention
           ? record.native_group_q64_panel_hits == 0U &&
                 record.native_group_q128_v4_panel_hits ==
                     g2_d2_expected_native_attention_hits &&
                 record.native_flashinfer_exact_panel_hits == 0U &&
                 record.generic_qt2_hits == 0U
       : g2_d2_nvfp4_candidate_uses_flashinfer_attention
           ? record.native_group_q64_panel_hits == 0U &&
                 record.native_group_q128_v4_panel_hits == 0U &&
                 record.native_flashinfer_exact_panel_hits ==
                     g2_d2_expected_native_attention_hits &&
                 record.generic_qt2_hits == 0U
           : record.native_group_q64_panel_hits == 0U &&
                 record.native_group_q128_v4_panel_hits == 0U &&
                 record.native_flashinfer_exact_panel_hits == 0U &&
                 record.generic_qt2_hits != 0U);
  const bool g2_d2_fp8_partition_complete =
      record.nvfp4_true_large_m_route_fp8_projection_bulk_hits <=
          record.nvfp4_true_large_m_route_fp8_projection_hits &&
      record.nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits ==
          record.nvfp4_true_large_m_route_fp8_projection_hits -
              record.nvfp4_true_large_m_route_fp8_projection_bulk_hits;
  const bool g2_d2_package_counts_complete =
      g2_d2_panel_coverage_complete &&
      g2_d2_expected_fp8_count_representable &&
      g2_d2_expected_nvfp4_count_representable &&
      g2_d2_attention_counts_complete &&
      record.native_nvfp4_true_large_m_gate_up_hits ==
          record.operator_panel_executor_hits &&
      record.native_nvfp4_true_large_m_down_hits ==
          record.operator_panel_executor_hits &&
      record.native_nvfp4_true_large_m_projection_hits ==
          g2_d2_expected_nvfp4_projection_hits &&
      record.native_nvfp4_true_large_m_physical_launches ==
          record.native_nvfp4_true_large_m_projection_hits &&
      record.nvfp4_true_large_m_route_fp8_projection_hits ==
          g2_d2_expected_fp8_projection_hits &&
      g2_d2_fp8_partition_complete &&
      record.nvfp4_true_large_m_route_fp8_projection_physical_launches >=
          record.nvfp4_true_large_m_route_fp8_projection_hits;
  constexpr std::uint64_t kP40LogicalPanelCount = 5U;
  constexpr std::uint64_t kP40BulkPanelCount = 3U;
  constexpr std::uint64_t kP40PartialPanelCount = 2U;
  constexpr std::uint64_t kP40ExpectedFp8ProjectionHits =
      kP40LogicalPanelCount * kFp8ProjectionCountPerPanel;
  constexpr std::uint64_t kP40ExpectedBulkFp8ProjectionHits =
      kP40BulkPanelCount * kFp8ProjectionCountPerPanel;
  constexpr std::uint64_t kP40ExpectedPartialFp8ProjectionHits =
      kP40PartialPanelCount * kFp8ProjectionCountPerPanel;
  constexpr std::uint64_t kP40ExpectedAttentionHits =
      kP40LogicalPanelCount * kAttentionLayerCount;
  const bool p40_attention_counts_complete =
      persistent_p40_candidate_uses_q64_attention
          ? record.native_group_q64_panel_hits ==
                    kP40ExpectedAttentionHits &&
                record.native_group_q128_v4_panel_hits == 0U &&
                record.native_flashinfer_exact_panel_hits == 0U &&
                record.generic_qt2_hits == 0U
      : persistent_p40_candidate_uses_q128_v4_attention
          ? record.native_group_q64_panel_hits == 0U &&
                record.native_group_q128_v4_panel_hits ==
                    kP40ExpectedAttentionHits &&
                record.native_flashinfer_exact_panel_hits == 0U &&
                record.generic_qt2_hits == 0U
      : persistent_p40_candidate_uses_flashinfer_attention
          ? record.native_group_q64_panel_hits == 0U &&
                record.native_group_q128_v4_panel_hits == 0U &&
                record.native_flashinfer_exact_panel_hits ==
                    kP40ExpectedAttentionHits &&
                record.generic_qt2_hits == 0U
          : record.native_group_q64_panel_hits == 0U &&
                record.native_group_q128_v4_panel_hits == 0U &&
                record.native_flashinfer_exact_panel_hits == 0U &&
                record.generic_qt2_hits != 0U;
  const bool persistent_p40_package_counts_complete =
      record.prompt_tokens ==
          runtime::kLayerMajorPrefillLayerWideMlpP40Tokens &&
      record.prefill_logical_panel_count == kP40LogicalPanelCount &&
      record.operator_panel_executor_hits ==
          kLayerCount * kP40LogicalPanelCount &&
      record.mlp_schedule_tactic == runtime::
          LayerMajorPrefillMlpScheduleTactic::kLayerWideP40ExactFullM &&
      record.route_layer_pass_count == 1U &&
      record.prefill_route_evidence.valid &&
      record.prefill_route_evidence.complete &&
      !record.prefill_route_evidence.request_active &&
      record.prefill_route_evidence.completed_layer_passes == 1U &&
      record.prefill_route_evidence.expected_layer_passes == 1U &&
      record.layer_wide_p40_mlp_layer_hits == kLayerCount &&
      record.persistent_p40_nvfp4_gate_up_hits == kLayerCount &&
      record.persistent_p40_nvfp4_down_residual_hits == kLayerCount &&
      record.persistent_p40_nvfp4_physical_launches == 2U * kLayerCount &&
      record.persistent_p40_fp8_projection_hits ==
          kP40ExpectedFp8ProjectionHits &&
      record.persistent_p40_fp8_projection_bulk_hits ==
          kP40ExpectedBulkFp8ProjectionHits &&
      record.persistent_p40_fp8_projection_oracle_partial_hits ==
          kP40ExpectedPartialFp8ProjectionHits &&
      record.persistent_p40_fp8_projection_physical_launches >=
          record.persistent_p40_fp8_projection_hits &&
      p40_attention_counts_complete;
  constexpr std::uint64_t kP40WholeCoreExpectedLayerHits = kLayerCount;
  constexpr std::uint64_t kP40WholeCoreExpectedPanelPhaseHits =
      kLayerCount * kP40LogicalPanelCount;
  constexpr std::uint64_t kP40WholeCoreExpectedSubmissionRetirements =
      kLayerCount * (kP40LogicalPanelCount + 1U +
                     kP40LogicalPanelCount + 1U);
  static_assert(kP40WholeCoreExpectedSubmissionRetirements == 768U);
  constexpr std::uint64_t kP40WholeCoreExpectedFp8ProjectionHits =
      kP40LogicalPanelCount *
      (3U * kLinearAttentionLayerCount + 4U * kAttentionLayerCount);
  static_assert(kP40WholeCoreExpectedFp8ProjectionHits == 1'040U);
  constexpr std::uint64_t kP40ProjectionResetExpectedFp8TensorRoleHits =
      3U * kLinearAttentionLayerCount + 4U * kAttentionLayerCount;
  constexpr std::uint64_t kP40ProjectionResetExpectedFp8PhysicalLaunches =
      2U * kLayerCount;
  constexpr std::uint64_t kP40PackedExpectedFp8TensorRoleHits =
      kP40ProjectionResetExpectedFp8TensorRoleHits;
  constexpr std::uint64_t kP40PackedExpectedFp8PhysicalLaunches =
      kP40ProjectionResetExpectedFp8PhysicalLaunches;
  constexpr std::uint64_t kP40PackedExpectedNvFp4PhysicalLaunches =
      2U * kLayerCount;
  constexpr std::uint64_t kP40PackedNvfp4V2ExpectedFp8TensorRoleHits =
      kP40WholeCoreExpectedFp8ProjectionHits;
  constexpr std::uint64_t kP40PackedNvfp4V2ExpectedFp8PhysicalLaunches =
      kP40WholeCoreExpectedFp8ProjectionHits;
  static_assert(kP40ProjectionResetExpectedFp8TensorRoleHits == 208U);
  static_assert(kP40ProjectionResetExpectedFp8PhysicalLaunches == 128U);
  static_assert(kP40PackedExpectedFp8TensorRoleHits == 208U);
  static_assert(kP40PackedExpectedFp8PhysicalLaunches == 128U);
  static_assert(kP40PackedExpectedNvFp4PhysicalLaunches == 128U);
  static_assert(kP40PackedNvfp4V2ExpectedFp8TensorRoleHits == 1'040U);
  static_assert(kP40PackedNvfp4V2ExpectedFp8PhysicalLaunches == 1'040U);
  bool prompt_wide_p40_whole_core_route_is_production_only = true;
  for (std::size_t index = 0U;
       index < runtime::kPrefillOperatorRoleCount; ++index) {
    const runtime::PrefillOperatorRouteCounts& counts =
        record.prefill_route_evidence.operators[index];
    prompt_wide_p40_whole_core_route_is_production_only =
        prompt_wide_p40_whole_core_route_is_production_only &&
        counts.production_hits ==
            runtime::kExpectedPrefillLogicalOperatorsPerTile[index] &&
        counts.exact_fallback_hits == 0U && counts.forbidden_hits == 0U;
  }
  for (const std::uint64_t boundary_hits :
       record.prefill_route_evidence.forbidden_boundary_hits) {
    prompt_wide_p40_whole_core_route_is_production_only =
        prompt_wide_p40_whole_core_route_is_production_only &&
        boundary_hits == 0U;
  }
  const bool prompt_wide_p40_common_non_mlp_counts_complete =
      record.prompt_tokens ==
          runtime::kLayerMajorPrefillPromptWideP40Tokens &&
      record.prefill_logical_panel_count == kP40LogicalPanelCount &&
      record.request_memory_profile ==
          runtime::RequestMemoryProfile::kLayerMajorP40WholeCore &&
      record.bounded_submission_window &&
      record.submission_window_retirements ==
          kP40WholeCoreExpectedSubmissionRetirements &&
      record.route_layer_pass_count == 1U &&
      record.prefill_route_evidence.valid &&
      record.prefill_route_evidence.complete &&
      !record.prefill_route_evidence.request_active &&
      record.prefill_route_evidence.completed_layer_passes == 1U &&
      record.prefill_route_evidence.expected_layer_passes == 1U &&
      prompt_wide_p40_whole_core_route_is_production_only &&
      record.prompt_wide_p40_whole_core_layer_hits ==
          kP40WholeCoreExpectedLayerHits &&
      record.prompt_wide_p40_fill_panel_hits ==
          kP40WholeCoreExpectedPanelPhaseHits &&
      record.prompt_wide_p40_prompt_core_hits == kLayerCount &&
      record.prompt_wide_p40_drain_panel_hits ==
          kP40WholeCoreExpectedPanelPhaseHits &&
      record.prompt_wide_p40_bf16_ab_hits ==
          kLinearAttentionLayerCount &&
      record.prompt_wide_p40_gdn_hits == kLinearAttentionLayerCount &&
      record.native_flashinfer_exact_whole_prompt_hits ==
          kAttentionLayerCount &&
      record.layer_wide_p40_mlp_layer_hits == kLayerCount &&
      record.operator_panel_executor_hits == 0U &&
      record.native_group_q64_panel_hits == 0U &&
      record.native_group_q128_v4_panel_hits == 0U &&
      record.native_flashinfer_exact_panel_hits == 0U &&
      record.generic_qt2_hits == 0U &&
      record.segmented_panel_projection_hits == 0U &&
      record.segmented_panel_projection_physical_launches == 0U &&
      record.native_large_m_projection_hits == 0U &&
      record.native_large_m_projection_bulk_hits == 0U &&
      record.native_large_m_projection_oracle_partial_hits == 0U &&
      record.native_large_m_projection_physical_launches == 0U &&
      record.nvfp4_true_large_m_route_fp8_projection_hits == 0U &&
      record.nvfp4_true_large_m_route_fp8_projection_bulk_hits == 0U &&
      record
              .nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits ==
          0U &&
      record
              .nvfp4_true_large_m_route_fp8_projection_physical_launches ==
          0U &&
      record.native_nvfp4_true_large_m_projection_hits == 0U &&
      record.native_nvfp4_true_large_m_gate_up_hits == 0U &&
      record.native_nvfp4_true_large_m_down_hits == 0U &&
      record.native_nvfp4_true_large_m_physical_launches == 0U &&
      record.persistent_p40_fp8_projection_hits == 0U &&
      record.persistent_p40_fp8_projection_bulk_hits == 0U &&
      record.persistent_p40_fp8_projection_oracle_partial_hits == 0U &&
      record.persistent_p40_fp8_projection_physical_launches == 0U;
  const bool prompt_wide_p40_common_package_counts_complete =
      prompt_wide_p40_common_non_mlp_counts_complete &&
      record.persistent_p40_nvfp4_gate_up_hits == kLayerCount &&
      record.persistent_p40_nvfp4_down_residual_hits == kLayerCount &&
      record.persistent_p40_nvfp4_physical_launches == 2U * kLayerCount;
  const bool prompt_wide_p40_whole_core_package_counts_complete =
      prompt_wide_p40_common_package_counts_complete &&
      record.mlp_schedule_tactic == runtime::
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore &&
      record.prompt_wide_p40_fp8_projection_hits ==
          kP40WholeCoreExpectedFp8ProjectionHits &&
      record.prompt_wide_p40_fp8_projection_physical_launches ==
          record.prompt_wide_p40_fp8_projection_hits;
  const bool p40_projection_reset_package_counts_complete =
      prompt_wide_p40_common_package_counts_complete &&
      record.full_prompt_consumed &&
      record.consumed_prompt_tokens == record.prompt_tokens &&
      record.completion_tokens > 0U &&
      record.mlp_schedule_tactic == runtime::
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40ProjectionReset &&
      record.prompt_wide_p40_fp8_projection_hits ==
          kP40ProjectionResetExpectedFp8TensorRoleHits &&
      record.prompt_wide_p40_fp8_projection_physical_launches ==
          kP40ProjectionResetExpectedFp8PhysicalLaunches;
  const bool p40_packed_projection_package_counts_complete =
      prompt_wide_p40_common_package_counts_complete &&
      record.full_prompt_consumed &&
      record.consumed_prompt_tokens == record.prompt_tokens &&
      record.completion_tokens > 0U &&
      record.mlp_schedule_tactic == runtime::
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40PackedProjection &&
      record.prompt_wide_p40_fp8_projection_hits ==
          kP40PackedExpectedFp8TensorRoleHits &&
      record.prompt_wide_p40_fp8_projection_physical_launches ==
          kP40PackedExpectedFp8PhysicalLaunches &&
      record.persistent_p40_nvfp4_physical_launches ==
          kP40PackedExpectedNvFp4PhysicalLaunches;
  const bool p40_packed_nvfp4_v2_package_counts_complete =
      prompt_wide_p40_common_package_counts_complete &&
      record.full_prompt_consumed &&
      record.consumed_prompt_tokens == record.prompt_tokens &&
      record.completion_tokens > 0U &&
      record.mlp_schedule_tactic == runtime::
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40PackedNvfp4V2 &&
      record.prompt_wide_p40_fp8_projection_hits ==
          kP40PackedNvfp4V2ExpectedFp8TensorRoleHits &&
      record.prompt_wide_p40_fp8_projection_physical_launches ==
          kP40PackedNvfp4V2ExpectedFp8PhysicalLaunches &&
      record.persistent_p40_nvfp4_physical_launches ==
          kP40PackedExpectedNvFp4PhysicalLaunches &&
      record.packed_nvfp4_v2_gate_up_hits == kLayerCount &&
      record.packed_nvfp4_v2_down_hits == kLayerCount &&
      record.packed_nvfp4_v2_physical_launches ==
          kP40PackedExpectedNvFp4PhysicalLaunches;
  const bool p40_vllm_marlin_parity_package_counts_complete =
      prompt_wide_p40_common_non_mlp_counts_complete &&
      record.full_prompt_consumed &&
      record.consumed_prompt_tokens == record.prompt_tokens &&
      record.completion_tokens > 0U &&
      record.prefix_execution_count == 1U &&
      record.prefill_execution_mode == runtime::
          ReferencePrefillExecutionMode::kWholeRequestLayerMajor &&
      record.mlp_schedule_tactic == runtime::
          LayerMajorPrefillMlpScheduleTactic::
              kPromptWideP40VllmMarlinParity &&
      record.prompt_wide_p40_fp8_projection_hits == runtime::
          kLayerMajorPrefillVllmMarlinParityFp8TensorRoleHitsPerRequest &&
      record.prompt_wide_p40_fp8_projection_physical_launches == runtime::
          kLayerMajorPrefillVllmMarlinParityFp8PhysicalLaunchesPerRequest &&
      record.vllm_marlin_parity_gate_up_hits == runtime::
          kLayerMajorPrefillVllmMarlinParityGateUpLogicalRoleHitsPerRequest &&
      record.vllm_marlin_parity_down_hits == runtime::
          kLayerMajorPrefillVllmMarlinParityDownLogicalRoleHitsPerRequest &&
      record.vllm_marlin_parity_physical_launches == runtime::
          kLayerMajorPrefillVllmMarlinParityNvFp4PhysicalLaunchesPerRequest &&
      record.vllm_marlin_parity_standalone_silu_launches ==
          kLayerCount * runtime::
              kLayerMajorPrefillVllmMarlinParityStandaloneSiluLaunchesPerLayer &&
      record.vllm_marlin_parity_standalone_residual_launches ==
          kLayerCount * runtime::
              kLayerMajorPrefillVllmMarlinParityStandaloneResidualLaunchesPerLayer &&
      record.vllm_marlin_parity_lock_clear_operations == runtime::
          kLayerMajorPrefillVllmMarlinParityLockClearOperationsPerRequest &&
      complete_vllm_marlin_parity_layer_receipts(record) &&
      record.persistent_p40_nvfp4_gate_up_hits == 0U &&
      record.persistent_p40_nvfp4_down_residual_hits == 0U &&
      record.persistent_p40_nvfp4_physical_launches == 0U &&
      record.packed_nvfp4_v2_gate_up_hits == 0U &&
      record.packed_nvfp4_v2_down_hits == 0U &&
      record.packed_nvfp4_v2_physical_launches == 0U;
  const bool accuracy_unqualified_candidate =
      candidate_v3 || projection_candidate_v4 ||
      native_large_m_candidate_v5 || flashinfer_exact_candidate_v6 ||
      true_large_m_nvfp4_candidate_v7 || g2_d2_nvfp4_candidate_v8 ||
      persistent_p40_candidate_v9 ||
      prompt_wide_p40_whole_core_candidate_v10 ||
      p40_projection_reset_candidate_v11 ||
      p40_packed_projection_candidate_v13 ||
      p40_packed_nvfp4_v2_candidate_v14 ||
      p40_vllm_marlin_parity_candidate_v15;
  std::string output =
      p40_vllm_marlin_parity_candidate_v15
          ? "{\"record\":\"target-prefill-witness-v15\","
            "\"schema_version\":15,\"request\":{\"id\":"
      : p40_packed_nvfp4_v2_candidate_v14
          ? "{\"record\":\"target-prefill-witness-v14\","
            "\"schema_version\":14,\"request\":{\"id\":"
      : p40_packed_projection_candidate_v13
          ? "{\"record\":\"target-prefill-witness-v13\","
            "\"schema_version\":13,\"request\":{\"id\":"
      : p40_projection_reset_candidate_v11
          ? "{\"record\":\"target-prefill-witness-v11\","
            "\"schema_version\":11,\"request\":{\"id\":"
      : prompt_wide_p40_whole_core_candidate_v10
          ? "{\"record\":\"target-prefill-witness-v10\","
            "\"schema_version\":10,\"request\":{\"id\":"
      : persistent_p40_candidate_v9
          ? "{\"record\":\"target-prefill-witness-v9\","
            "\"schema_version\":9,\"request\":{\"id\":"
      : g2_d2_nvfp4_candidate_v8
          ? "{\"record\":\"target-prefill-witness-v8\","
            "\"schema_version\":8,\"request\":{\"id\":"
      : true_large_m_nvfp4_candidate_v7
          ? "{\"record\":\"target-prefill-witness-v7\","
            "\"schema_version\":7,\"request\":{\"id\":"
      : flashinfer_exact_candidate_v6
          ? "{\"record\":\"target-prefill-witness-v6\","
            "\"schema_version\":6,\"request\":{\"id\":"
      : native_large_m_candidate_v5
          ? "{\"record\":\"target-prefill-witness-v5\","
            "\"schema_version\":5,\"request\":{\"id\":"
      : projection_candidate_v4
          ? "{\"record\":\"target-prefill-witness-v4\","
            "\"schema_version\":4,\"request\":{\"id\":"
          : candidate_v3
          ? "{\"record\":\"target-prefill-witness-v3\","
            "\"schema_version\":3,\"request\":{\"id\":"
          : production_reset_v16
                ? "{\"record\":\"target-prefill-witness-v16\","
                  "\"schema_version\":16,\"request\":{\"id\":"
          : sealed
                ? "{\"record\":\"target-prefill-witness-v2\","
                  "\"schema_version\":2,\"request\":{\"id\":"
                : "{\"record\":\"target-prefill-witness-v1\","
                  "\"schema_version\":1,\"request\":{\"id\":";
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
  output += "}";
  if (production_reset_v16) {
    output += ",\"request_state_reset\":{\"mode\":";
    append_json_string(
        output, runtime::to_string(record.request_state_reset->mode));
    output += ",\"positions\":" +
              std::to_string(
                  record.request_state_reset->cleared_positions) +
              ",\"bytes\":" +
              std::to_string(record.request_state_reset->zeroed_bytes) +
              ",\"milliseconds\":" +
              std::to_string(record.request_state_reset->milliseconds) + "}";
  }
  output += ",\"prefill\":{\"requested_chunk\":" +
            std::to_string(record.requested_prefill_chunk_size) +
            ",\"effective_chunk\":" +
            std::to_string(record.effective_prefill_chunk_size) +
            ",\"prefix_execution_count\":" +
            std::to_string(record.prefix_execution_count);
  if (sealed) {
    output += ",\"execution_mode\":";
    append_json_string(
        output,
        record.prefill_execution_mode == runtime::
                ReferencePrefillExecutionMode::kWholeRequestLayerMajor
            ? "layer-major"
            : "legacy");
    output += ",\"logical_panel_count\":" +
              std::to_string(record.prefill_logical_panel_count) +
              ",\"request_memory_profile\":";
    switch (record.request_memory_profile) {
      case runtime::RequestMemoryProfile::kLayerMajorP40WholeCore:
        append_json_string(output, "layer-major-p40-whole-core");
        break;
      case runtime::RequestMemoryProfile::kLayerMajorC8192:
        append_json_string(output, "layer-major-c8192");
        break;
      case runtime::RequestMemoryProfile::kLegacyC512:
      default:
        append_json_string(output, "legacy-c512");
        break;
    }
    output += ",\"bounded_submission_window\":";
    output += record.bounded_submission_window ? "true" : "false";
    output += ",\"submission_window_retirements\":" +
              std::to_string(record.submission_window_retirements);
    if (p40_vllm_marlin_parity_candidate_v15) {
      const bool parity_receipts_complete =
          complete_vllm_marlin_parity_layer_receipts(record);
      output += ",\"projection_tactic\":";
      append_json_string(output,
                         "native-prompt-wide-p40-vllm-marlin-parity");
      output += ",\"mlp_schedule\":";
      append_json_string(output, "prompt-wide-p40-vllm-marlin-parity");
      output += ",\"attention_tactic\":";
      append_json_string(output, "native-flashinfer-exact-whole-prompt");
      output += ",\"package_complete\":";
      output += p40_vllm_marlin_parity_package_counts_complete ? "true"
                                                               : "false";
      output +=
          ",\"route_layer_pass_count\":" +
          std::to_string(record.route_layer_pass_count) +
          ",\"operator_panel_executor_hits\":" +
          std::to_string(record.operator_panel_executor_hits) +
          ",\"prompt_wide_p40_whole_core_layer_hits\":" +
          std::to_string(record.prompt_wide_p40_whole_core_layer_hits) +
          ",\"prompt_wide_p40_fill_panel_hits\":" +
          std::to_string(record.prompt_wide_p40_fill_panel_hits) +
          ",\"prompt_wide_p40_prompt_core_hits\":" +
          std::to_string(record.prompt_wide_p40_prompt_core_hits) +
          ",\"prompt_wide_p40_drain_panel_hits\":" +
          std::to_string(record.prompt_wide_p40_drain_panel_hits) +
          ",\"prompt_wide_p40_fp8_projection_hits\":" +
          std::to_string(record.prompt_wide_p40_fp8_projection_hits) +
          ",\"prompt_wide_p40_fp8_projection_physical_launches\":" +
          std::to_string(
              record.prompt_wide_p40_fp8_projection_physical_launches) +
          ",\"prompt_wide_p40_bf16_ab_hits\":" +
          std::to_string(record.prompt_wide_p40_bf16_ab_hits) +
          ",\"prompt_wide_p40_gdn_hits\":" +
          std::to_string(record.prompt_wide_p40_gdn_hits) +
          ",\"native_flashinfer_exact_whole_prompt_hits\":" +
          std::to_string(record.native_flashinfer_exact_whole_prompt_hits) +
          ",\"layer_wide_p40_mlp_layer_hits\":" +
          std::to_string(record.layer_wide_p40_mlp_layer_hits) +
          ",\"vllm_marlin_parity_gate_up_hits\":" +
          std::to_string(record.vllm_marlin_parity_gate_up_hits) +
          ",\"vllm_marlin_parity_down_hits\":" +
          std::to_string(record.vllm_marlin_parity_down_hits) +
          ",\"vllm_marlin_parity_physical_launches\":" +
          std::to_string(record.vllm_marlin_parity_physical_launches) +
          ",\"vllm_marlin_parity_standalone_silu_launches\":" +
          std::to_string(
              record.vllm_marlin_parity_standalone_silu_launches) +
          ",\"vllm_marlin_parity_standalone_residual_launches\":" +
          std::to_string(
              record.vllm_marlin_parity_standalone_residual_launches) +
          ",\"vllm_marlin_parity_lock_clear_operations\":" +
          std::to_string(record.vllm_marlin_parity_lock_clear_operations) +
          ",\"persistent_p40_nvfp4_gate_up_hits\":" +
          std::to_string(record.persistent_p40_nvfp4_gate_up_hits) +
          ",\"persistent_p40_nvfp4_down_residual_hits\":" +
          std::to_string(record.persistent_p40_nvfp4_down_residual_hits) +
          ",\"persistent_p40_nvfp4_physical_launches\":" +
          std::to_string(record.persistent_p40_nvfp4_physical_launches) +
          ",\"packed_nvfp4_v2_gate_up_hits\":" +
          std::to_string(record.packed_nvfp4_v2_gate_up_hits) +
          ",\"packed_nvfp4_v2_down_hits\":" +
          std::to_string(record.packed_nvfp4_v2_down_hits) +
          ",\"packed_nvfp4_v2_physical_launches\":" +
          std::to_string(record.packed_nvfp4_v2_physical_launches) +
          ",\"vllm_marlin_parity_layer_completion_receipt_count\":" +
          std::to_string(
              record.vllm_marlin_parity_layer_completion_receipt_count) +
          ",\"vllm_marlin_parity_layer_completion_receipts\":";
      append_vllm_marlin_parity_layer_receipts(output, record);
      output += ",\"p40_vllm_marlin_parity_package\":{\"identity\":";
      append_json_string(
          output,
          "stock-vllm-marlin-p40000-projection-host-dispatch-parity-v1");
      output +=
          ",\"scope\":\"projection-host-dispatch-reference\","
          "\"selection\":\"sealed-fail-closed\",\"complete\":";
      output += p40_vllm_marlin_parity_package_counts_complete ? "true"
                                                               : "false";
      output +=
          ",\"publication_order\":"
          "\"retained-prompt-core->canonical-gate-then-up-bf16->"
          "standalone-silu-activated-bf16->down-bf16->"
          "standalone-residual\",\"count_validation\":{"
          "\"expected_fp8_tensor_role_hits\":" +
          std::to_string(runtime::
                             kLayerMajorPrefillVllmMarlinParityFp8TensorRoleHitsPerRequest) +
          ",\"expected_fp8_physical_launches\":" +
          std::to_string(runtime::
                             kLayerMajorPrefillVllmMarlinParityFp8PhysicalLaunchesPerRequest) +
          ",\"expected_gate_up_hits\":" +
          std::to_string(runtime::
                             kLayerMajorPrefillVllmMarlinParityGateUpLogicalRoleHitsPerRequest) +
          ",\"expected_down_hits\":" +
          std::to_string(runtime::
                             kLayerMajorPrefillVllmMarlinParityDownLogicalRoleHitsPerRequest) +
          ",\"expected_nvfp4_physical_launches\":" +
          std::to_string(runtime::
                             kLayerMajorPrefillVllmMarlinParityNvFp4PhysicalLaunchesPerRequest) +
          ",\"expected_full_m1024_launches_per_role_per_layer\":" +
          std::to_string(runtime::
                             kLayerMajorPrefillVllmMarlinParityFullSegmentsPerProjection) +
          ",\"expected_split_m64_launches_per_role_per_layer\":" +
          std::to_string(runtime::
                             kLayerMajorPrefillVllmMarlinParityTailSegmentsPerProjection) +
          ",\"expected_standalone_silu_launches\":" +
          std::to_string(kLayerCount) +
          ",\"expected_standalone_residual_launches\":" +
          std::to_string(kLayerCount) +
          ",\"expected_lock_clear_operations\":" +
          std::to_string(runtime::
                             kLayerMajorPrefillVllmMarlinParityLockClearOperationsPerRequest) +
          ",\"expected_layer_completion_receipts\":" +
          std::to_string(kLayerCount) +
          ",\"all_layer_receipts_complete\":";
      output += parity_receipts_complete ? "true" : "false";
      output += "}}";
    } else if (p40_packed_nvfp4_v2_candidate_v14) {
      output += ",\"projection_tactic\":";
      append_json_string(output,
                         "native-prompt-wide-p40-packed-nvfp4-v2");
      output += ",\"mlp_schedule\":";
      append_json_string(output, "prompt-wide-p40-packed-nvfp4-v2");
      output += ",\"attention_tactic\":";
      append_json_string(output, "native-flashinfer-exact-whole-prompt");
      output += ",\"package_complete\":";
      output += p40_packed_nvfp4_v2_package_counts_complete ? "true"
                                                            : "false";
      output +=
          ",\"route_layer_pass_count\":" +
          std::to_string(record.route_layer_pass_count) +
          ",\"operator_panel_executor_hits\":" +
          std::to_string(record.operator_panel_executor_hits) +
          ",\"prompt_wide_p40_whole_core_layer_hits\":" +
          std::to_string(record.prompt_wide_p40_whole_core_layer_hits) +
          ",\"prompt_wide_p40_fill_panel_hits\":" +
          std::to_string(record.prompt_wide_p40_fill_panel_hits) +
          ",\"prompt_wide_p40_prompt_core_hits\":" +
          std::to_string(record.prompt_wide_p40_prompt_core_hits) +
          ",\"prompt_wide_p40_drain_panel_hits\":" +
          std::to_string(record.prompt_wide_p40_drain_panel_hits) +
          ",\"prompt_wide_p40_fp8_projection_hits\":" +
          std::to_string(record.prompt_wide_p40_fp8_projection_hits) +
          ",\"prompt_wide_p40_fp8_projection_physical_launches\":" +
          std::to_string(
              record.prompt_wide_p40_fp8_projection_physical_launches) +
          ",\"prompt_wide_p40_bf16_ab_hits\":" +
          std::to_string(record.prompt_wide_p40_bf16_ab_hits) +
          ",\"prompt_wide_p40_gdn_hits\":" +
          std::to_string(record.prompt_wide_p40_gdn_hits) +
          ",\"native_flashinfer_exact_whole_prompt_hits\":" +
          std::to_string(record.native_flashinfer_exact_whole_prompt_hits) +
          ",\"layer_wide_p40_mlp_layer_hits\":" +
          std::to_string(record.layer_wide_p40_mlp_layer_hits) +
          ",\"persistent_p40_nvfp4_gate_up_hits\":" +
          std::to_string(record.persistent_p40_nvfp4_gate_up_hits) +
          ",\"persistent_p40_nvfp4_down_residual_hits\":" +
          std::to_string(record.persistent_p40_nvfp4_down_residual_hits) +
          ",\"persistent_p40_nvfp4_physical_launches\":" +
          std::to_string(record.persistent_p40_nvfp4_physical_launches) +
          ",\"packed_nvfp4_v2_gate_up_hits\":" +
          std::to_string(record.packed_nvfp4_v2_gate_up_hits) +
          ",\"packed_nvfp4_v2_down_hits\":" +
          std::to_string(record.packed_nvfp4_v2_down_hits) +
          ",\"packed_nvfp4_v2_physical_launches\":" +
          std::to_string(record.packed_nvfp4_v2_physical_launches) +
          ",\"p40_packed_nvfp4_v2_package\":{\"identity\":";
      append_json_string(output,
                         "exact-p40000-packed-nvfp4-v2-dataflow-v1");
      output += ",\"selection\":\"sealed-fail-closed\",\"complete\":";
      output += p40_packed_nvfp4_v2_package_counts_complete ? "true"
                                                            : "false";
      output +=
          ",\"logical_panels\":" +
          std::to_string(kP40LogicalPanelCount) +
          ",\"panel_tokens\":" +
          std::to_string(
              runtime::kLayerMajorPrefillPromptWideP40PanelTokens) +
          ",\"projection_m_tokens\":" +
          std::to_string(runtime::kLayerMajorPrefillPromptWideP40Tokens) +
          ",\"count_validation\":{\"expected_layer_hits\":" +
          std::to_string(kP40WholeCoreExpectedLayerHits) +
          ",\"expected_panel_phase_hits\":" +
          std::to_string(kP40WholeCoreExpectedPanelPhaseHits) +
          ",\"expected_fp8_tensor_role_hits\":" +
          std::to_string(kP40PackedNvfp4V2ExpectedFp8TensorRoleHits) +
          ",\"expected_fp8_physical_launches\":" +
          std::to_string(kP40PackedNvfp4V2ExpectedFp8PhysicalLaunches) +
          ",\"expected_nvfp4_gate_up_hits\":" +
          std::to_string(kLayerCount) +
          ",\"expected_nvfp4_down_hits\":" +
          std::to_string(kLayerCount) +
          ",\"expected_nvfp4_physical_launches\":" +
          std::to_string(kP40PackedExpectedNvFp4PhysicalLaunches) +
          ",\"expected_bf16_ab_hits\":" +
          std::to_string(kLinearAttentionLayerCount) +
          ",\"expected_gdn_hits\":" +
          std::to_string(kLinearAttentionLayerCount) +
          ",\"expected_whole_prompt_attention_hits\":" +
          std::to_string(kAttentionLayerCount) + "}}";
    } else if (p40_packed_projection_candidate_v13) {
      output += ",\"projection_tactic\":";
      append_json_string(output,
                         "native-prompt-wide-p40-packed-projection");
      output += ",\"mlp_schedule\":";
      append_json_string(output, "prompt-wide-p40-packed-projection");
      output += ",\"attention_tactic\":";
      append_json_string(output, "native-flashinfer-exact-whole-prompt");
      output += ",\"package_complete\":";
      output += p40_packed_projection_package_counts_complete ? "true"
                                                              : "false";
      output +=
          ",\"route_layer_pass_count\":" +
          std::to_string(record.route_layer_pass_count) +
          ",\"operator_panel_executor_hits\":" +
          std::to_string(record.operator_panel_executor_hits) +
          ",\"prompt_wide_p40_whole_core_layer_hits\":" +
          std::to_string(record.prompt_wide_p40_whole_core_layer_hits) +
          ",\"prompt_wide_p40_fill_panel_hits\":" +
          std::to_string(record.prompt_wide_p40_fill_panel_hits) +
          ",\"prompt_wide_p40_prompt_core_hits\":" +
          std::to_string(record.prompt_wide_p40_prompt_core_hits) +
          ",\"prompt_wide_p40_drain_panel_hits\":" +
          std::to_string(record.prompt_wide_p40_drain_panel_hits) +
          ",\"prompt_wide_p40_fp8_projection_hits\":" +
          std::to_string(record.prompt_wide_p40_fp8_projection_hits) +
          ",\"prompt_wide_p40_fp8_projection_physical_launches\":" +
          std::to_string(
              record.prompt_wide_p40_fp8_projection_physical_launches) +
          ",\"prompt_wide_p40_bf16_ab_hits\":" +
          std::to_string(record.prompt_wide_p40_bf16_ab_hits) +
          ",\"prompt_wide_p40_gdn_hits\":" +
          std::to_string(record.prompt_wide_p40_gdn_hits) +
          ",\"native_flashinfer_exact_whole_prompt_hits\":" +
          std::to_string(record.native_flashinfer_exact_whole_prompt_hits) +
          ",\"layer_wide_p40_mlp_layer_hits\":" +
          std::to_string(record.layer_wide_p40_mlp_layer_hits) +
          ",\"persistent_p40_nvfp4_gate_up_hits\":" +
          std::to_string(record.persistent_p40_nvfp4_gate_up_hits) +
          ",\"persistent_p40_nvfp4_down_residual_hits\":" +
          std::to_string(record.persistent_p40_nvfp4_down_residual_hits) +
          ",\"persistent_p40_nvfp4_physical_launches\":" +
          std::to_string(record.persistent_p40_nvfp4_physical_launches) +
          ",\"p40_packed_projection_package\":{\"identity\":";
      append_json_string(output,
                         "exact-p40000-packed-projection-dataflow-v1");
      output += ",\"selection\":\"sealed-fail-closed\",\"complete\":";
      output += p40_packed_projection_package_counts_complete ? "true"
                                                              : "false";
      output +=
          ",\"logical_panels\":" +
          std::to_string(kP40LogicalPanelCount) +
          ",\"panel_tokens\":" +
          std::to_string(
              runtime::kLayerMajorPrefillPromptWideP40PanelTokens) +
          ",\"projection_m_tokens\":" +
          std::to_string(runtime::kLayerMajorPrefillPromptWideP40Tokens) +
          ",\"count_validation\":{\"expected_layer_hits\":" +
          std::to_string(kP40WholeCoreExpectedLayerHits) +
          ",\"expected_panel_phase_hits\":" +
          std::to_string(kP40WholeCoreExpectedPanelPhaseHits) +
          ",\"expected_fp8_tensor_role_hits\":" +
          std::to_string(kP40PackedExpectedFp8TensorRoleHits) +
          ",\"expected_fp8_physical_launches\":" +
          std::to_string(kP40PackedExpectedFp8PhysicalLaunches) +
          ",\"expected_nvfp4_physical_launches\":" +
          std::to_string(kP40PackedExpectedNvFp4PhysicalLaunches) +
          ",\"expected_bf16_ab_hits\":" +
          std::to_string(kLinearAttentionLayerCount) +
          ",\"expected_gdn_hits\":" +
          std::to_string(kLinearAttentionLayerCount) +
          ",\"expected_whole_prompt_attention_hits\":" +
          std::to_string(kAttentionLayerCount) + "}}";
    } else if (p40_projection_reset_candidate_v11) {
      output += ",\"projection_tactic\":";
      append_json_string(output,
                         "native-prompt-wide-p40-projection-reset");
      output += ",\"mlp_schedule\":";
      append_json_string(output, "prompt-wide-p40-projection-reset");
      output += ",\"attention_tactic\":";
      append_json_string(output, "native-flashinfer-exact-whole-prompt");
      output += ",\"package_complete\":";
      output += p40_projection_reset_package_counts_complete ? "true"
                                                             : "false";
      output +=
          ",\"route_layer_pass_count\":" +
          std::to_string(record.route_layer_pass_count) +
          ",\"operator_panel_executor_hits\":" +
          std::to_string(record.operator_panel_executor_hits) +
          ",\"prompt_wide_p40_whole_core_layer_hits\":" +
          std::to_string(record.prompt_wide_p40_whole_core_layer_hits) +
          ",\"prompt_wide_p40_fill_panel_hits\":" +
          std::to_string(record.prompt_wide_p40_fill_panel_hits) +
          ",\"prompt_wide_p40_prompt_core_hits\":" +
          std::to_string(record.prompt_wide_p40_prompt_core_hits) +
          ",\"prompt_wide_p40_drain_panel_hits\":" +
          std::to_string(record.prompt_wide_p40_drain_panel_hits) +
          ",\"prompt_wide_p40_fp8_projection_hits\":" +
          std::to_string(record.prompt_wide_p40_fp8_projection_hits) +
          ",\"prompt_wide_p40_fp8_projection_physical_launches\":" +
          std::to_string(
              record.prompt_wide_p40_fp8_projection_physical_launches) +
          ",\"prompt_wide_p40_bf16_ab_hits\":" +
          std::to_string(record.prompt_wide_p40_bf16_ab_hits) +
          ",\"prompt_wide_p40_gdn_hits\":" +
          std::to_string(record.prompt_wide_p40_gdn_hits) +
          ",\"native_flashinfer_exact_whole_prompt_hits\":" +
          std::to_string(record.native_flashinfer_exact_whole_prompt_hits) +
          ",\"layer_wide_p40_mlp_layer_hits\":" +
          std::to_string(record.layer_wide_p40_mlp_layer_hits) +
          ",\"persistent_p40_nvfp4_gate_up_hits\":" +
          std::to_string(record.persistent_p40_nvfp4_gate_up_hits) +
          ",\"persistent_p40_nvfp4_down_residual_hits\":" +
          std::to_string(record.persistent_p40_nvfp4_down_residual_hits) +
          ",\"persistent_p40_nvfp4_physical_launches\":" +
          std::to_string(record.persistent_p40_nvfp4_physical_launches) +
          ",\"p40_projection_reset_package\":{\"identity\":";
      append_json_string(output,
                         "exact-p40000-grouped-projection-reset-v1");
      output += ",\"selection\":\"sealed-fail-closed\",\"complete\":";
      output += p40_projection_reset_package_counts_complete ? "true"
                                                             : "false";
      output +=
          ",\"logical_panels\":" +
          std::to_string(kP40LogicalPanelCount) +
          ",\"panel_tokens\":" +
          std::to_string(
              runtime::kLayerMajorPrefillPromptWideP40PanelTokens) +
          ",\"projection_m_tokens\":" +
          std::to_string(runtime::kLayerMajorPrefillPromptWideP40Tokens) +
          ",\"count_validation\":{\"expected_layer_hits\":" +
          std::to_string(kP40WholeCoreExpectedLayerHits) +
          ",\"expected_panel_phase_hits\":" +
          std::to_string(kP40WholeCoreExpectedPanelPhaseHits) +
          ",\"expected_fp8_tensor_role_hits\":" +
          std::to_string(kP40ProjectionResetExpectedFp8TensorRoleHits) +
          ",\"expected_fp8_physical_launches\":" +
          std::to_string(kP40ProjectionResetExpectedFp8PhysicalLaunches) +
          ",\"expected_nvfp4_physical_launches\":" +
          std::to_string(2U * kLayerCount) +
          ",\"expected_bf16_ab_hits\":" +
          std::to_string(kLinearAttentionLayerCount) +
          ",\"expected_gdn_hits\":" +
          std::to_string(kLinearAttentionLayerCount) +
          ",\"expected_whole_prompt_attention_hits\":" +
          std::to_string(kAttentionLayerCount) + "}}";
    } else if (prompt_wide_p40_whole_core_candidate_v10) {
      output += ",\"projection_tactic\":";
      append_json_string(output, "native-prompt-wide-p40-whole-core");
      output += ",\"mlp_schedule\":";
      append_json_string(output, "prompt-wide-p40-whole-core");
      output += ",\"attention_tactic\":";
      append_json_string(output, "native-flashinfer-exact-whole-prompt");
      output += ",\"package_complete\":";
      output += prompt_wide_p40_whole_core_package_counts_complete
                    ? "true"
                    : "false";
      output +=
          ",\"route_layer_pass_count\":" +
          std::to_string(record.route_layer_pass_count) +
          ",\"operator_panel_executor_hits\":" +
          std::to_string(record.operator_panel_executor_hits) +
          ",\"prompt_wide_p40_whole_core_layer_hits\":" +
          std::to_string(record.prompt_wide_p40_whole_core_layer_hits) +
          ",\"prompt_wide_p40_fill_panel_hits\":" +
          std::to_string(record.prompt_wide_p40_fill_panel_hits) +
          ",\"prompt_wide_p40_prompt_core_hits\":" +
          std::to_string(record.prompt_wide_p40_prompt_core_hits) +
          ",\"prompt_wide_p40_drain_panel_hits\":" +
          std::to_string(record.prompt_wide_p40_drain_panel_hits) +
          ",\"prompt_wide_p40_fp8_projection_hits\":" +
          std::to_string(record.prompt_wide_p40_fp8_projection_hits) +
          ",\"prompt_wide_p40_fp8_projection_physical_launches\":" +
          std::to_string(
              record.prompt_wide_p40_fp8_projection_physical_launches) +
          ",\"prompt_wide_p40_bf16_ab_hits\":" +
          std::to_string(record.prompt_wide_p40_bf16_ab_hits) +
          ",\"prompt_wide_p40_gdn_hits\":" +
          std::to_string(record.prompt_wide_p40_gdn_hits) +
          ",\"native_flashinfer_exact_whole_prompt_hits\":" +
          std::to_string(record.native_flashinfer_exact_whole_prompt_hits) +
          ",\"layer_wide_p40_mlp_layer_hits\":" +
          std::to_string(record.layer_wide_p40_mlp_layer_hits) +
          ",\"persistent_p40_nvfp4_gate_up_hits\":" +
          std::to_string(record.persistent_p40_nvfp4_gate_up_hits) +
          ",\"persistent_p40_nvfp4_down_residual_hits\":" +
          std::to_string(record.persistent_p40_nvfp4_down_residual_hits) +
          ",\"persistent_p40_nvfp4_physical_launches\":" +
          std::to_string(record.persistent_p40_nvfp4_physical_launches) +
          ",\"prompt_wide_p40_whole_core_package\":{\"identity\":";
      append_json_string(output,
                         "exact-p40000-five-p8000-whole-core-v1");
      output += ",\"selection\":\"sealed-fail-closed\",\"complete\":";
      output += prompt_wide_p40_whole_core_package_counts_complete
                    ? "true"
                    : "false";
      output +=
          ",\"logical_panels\":" +
          std::to_string(kP40LogicalPanelCount) +
          ",\"panel_tokens\":" +
          std::to_string(
              runtime::kLayerMajorPrefillPromptWideP40PanelTokens) +
          ",\"count_validation\":{\"expected_layer_hits\":" +
          std::to_string(kP40WholeCoreExpectedLayerHits) +
          ",\"expected_panel_phase_hits\":" +
          std::to_string(kP40WholeCoreExpectedPanelPhaseHits) +
          ",\"expected_fp8_projection_hits\":" +
          std::to_string(kP40WholeCoreExpectedFp8ProjectionHits) +
          ",\"expected_bf16_ab_hits\":" +
          std::to_string(kLinearAttentionLayerCount) +
          ",\"expected_gdn_hits\":" +
          std::to_string(kLinearAttentionLayerCount) +
          ",\"expected_whole_prompt_attention_hits\":" +
          std::to_string(kAttentionLayerCount) + "}}";
    } else if (persistent_p40_candidate_v9) {
      output += ",\"projection_tactic\":";
      append_json_string(
          output, "native-nvfp4-persistent-p40-layer-wide-mlp");
      output += ",\"mlp_schedule\":";
      append_json_string(output, "layer-wide-p40-exact-full-m");
      output += ",\"attention_tactic\":";
      append_json_string(
          output, persistent_p40_candidate_uses_flashinfer_attention
                      ? "native-flashinfer-exact-panel"
                  : persistent_p40_candidate_uses_q128_v4_attention
                      ? "native-group-q128-v4-panel"
                  : persistent_p40_candidate_uses_q64_attention
                      ? "native-group-q64-panel"
                      : "exact-segmented");
      output += ",\"package_complete\":";
      output += persistent_p40_package_counts_complete ? "true" : "false";
      output += ",\"route_layer_pass_count\":" +
                std::to_string(record.route_layer_pass_count) +
                ",\"operator_panel_executor_hits\":" +
                std::to_string(record.operator_panel_executor_hits) +
                ",\"layer_wide_p40_mlp_layer_hits\":" +
                std::to_string(record.layer_wide_p40_mlp_layer_hits) +
                ",\"persistent_p40_nvfp4_gate_up_hits\":" +
                std::to_string(record.persistent_p40_nvfp4_gate_up_hits) +
                ",\"persistent_p40_nvfp4_down_residual_hits\":" +
                std::to_string(
                    record.persistent_p40_nvfp4_down_residual_hits) +
                ",\"persistent_p40_nvfp4_physical_launches\":" +
                std::to_string(
                    record.persistent_p40_nvfp4_physical_launches) +
                ",\"persistent_p40_fp8_projection_hits\":" +
                std::to_string(record.persistent_p40_fp8_projection_hits) +
                ",\"persistent_p40_fp8_projection_bulk_hits\":" +
                std::to_string(
                    record.persistent_p40_fp8_projection_bulk_hits) +
                ",\"persistent_p40_fp8_projection_oracle_partial_hits\":" +
                std::to_string(record
                                   .persistent_p40_fp8_projection_oracle_partial_hits) +
                ",\"persistent_p40_fp8_projection_physical_launches\":" +
                std::to_string(
                    record.persistent_p40_fp8_projection_physical_launches) +
                ",\"native_flashinfer_exact_panel_hits\":" +
                std::to_string(record.native_flashinfer_exact_panel_hits) +
                ",\"native_group_q64_panel_hits\":" +
                std::to_string(record.native_group_q64_panel_hits) +
                ",\"native_group_q128_v4_panel_hits\":" +
                std::to_string(record.native_group_q128_v4_panel_hits) +
                ",\"generic_qt2_hits\":" +
                std::to_string(record.generic_qt2_hits) +
                ",\"persistent_p40_package\":{\"identity\":";
      append_json_string(output, "exact-p40000-layer-wide-nvfp4-mlp");
      output += ",\"selection\":\"sealed-fail-closed\",\"complete\":";
      output += persistent_p40_package_counts_complete ? "true" : "false";
      output += ",\"logical_panels\":" +
                std::to_string(kP40LogicalPanelCount) +
                ",\"route_passes\":1,\"roles\":{\"gate_up\":{"
                "\"logical_hits\":" +
                std::to_string(record.persistent_p40_nvfp4_gate_up_hits) +
                "},\"down_residual\":{\"logical_hits\":" +
                std::to_string(
                    record.persistent_p40_nvfp4_down_residual_hits) +
                "}},\"count_validation\":{\"expected_fp8_hits\":" +
                std::to_string(kP40ExpectedFp8ProjectionHits) +
                ",\"expected_bulk_fp8_hits\":" +
                std::to_string(kP40ExpectedBulkFp8ProjectionHits) +
                ",\"expected_partial_fp8_hits\":" +
                std::to_string(kP40ExpectedPartialFp8ProjectionHits) +
                ",\"attention_route_complete\":";
      output += p40_attention_counts_complete ? "true" : "false";
      output += "}}";
    } else if (g2_d2_nvfp4_candidate_v8) {
      output += ",\"projection_tactic\":";
      append_json_string(output,
                         "native-nvfp4-g2-d2-large-m-operator-panel");
      output += ",\"nvfp4_package\":\"gate-g2+down-d2\","
                "\"nvfp4_gate_role_tactic\":\"GateUpG2\","
                "\"nvfp4_down_role_tactic\":\"DownD2\","
                "\"package_complete\":";
      output += g2_d2_package_counts_complete ? "true" : "false";
      output += ",\"attention_tactic\":";
      append_json_string(
          output, g2_d2_nvfp4_candidate_uses_flashinfer_attention
                      ? "native-flashinfer-exact-panel"
                  : g2_d2_nvfp4_candidate_uses_q128_v4_attention
                      ? "native-group-q128-v4-panel"
                  : g2_d2_nvfp4_candidate_uses_q64_attention
                      ? "native-group-q64-panel"
                      : "exact-segmented");
      output += ",\"operator_panel_executor_hits\":" +
                std::to_string(record.operator_panel_executor_hits) +
                ",\"native_flashinfer_exact_panel_hits\":" +
                std::to_string(record.native_flashinfer_exact_panel_hits) +
                ",\"native_group_q64_panel_hits\":" +
                std::to_string(record.native_group_q64_panel_hits) +
                ",\"native_group_q128_v4_panel_hits\":" +
                std::to_string(record.native_group_q128_v4_panel_hits) +
                ",\"generic_qt2_hits\":" +
                std::to_string(record.generic_qt2_hits) +
                ",\"nvfp4_g2_d2_route_fp8_projection_hits\":" +
                std::to_string(
                    record.nvfp4_true_large_m_route_fp8_projection_hits) +
                ",\"nvfp4_g2_d2_route_fp8_projection_bulk_hits\":" +
                std::to_string(record
                                   .nvfp4_true_large_m_route_fp8_projection_bulk_hits) +
                ",\"nvfp4_g2_d2_route_fp8_projection_oracle_partial_hits\":" +
                std::to_string(
                    record
                        .nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits) +
                ",\"nvfp4_g2_d2_route_fp8_projection_physical_launches\":" +
                std::to_string(
                    record
                        .nvfp4_true_large_m_route_fp8_projection_physical_launches) +
                ",\"native_nvfp4_g2_d2_projection_hits\":" +
                std::to_string(
                    record.native_nvfp4_true_large_m_projection_hits) +
                ",\"native_nvfp4_gate_up_g2_hits\":" +
                std::to_string(
                    record.native_nvfp4_true_large_m_gate_up_hits) +
                ",\"native_nvfp4_down_d2_hits\":" +
                std::to_string(record.native_nvfp4_true_large_m_down_hits) +
                ",\"native_nvfp4_g2_d2_physical_launches\":" +
                std::to_string(
                    record.native_nvfp4_true_large_m_physical_launches) +
                ",\"nvfp4_g2_d2_package\":{\"identity\":";
      append_json_string(output, "native-nvfp4-g2-d2-large-m");
      output += ",\"selection\":\"coupled-fail-closed\",\"complete\":";
      output += g2_d2_package_counts_complete ? "true" : "false";
      output += ",\"roles\":{\"gate_up\":{\"identity\":\"GateUpG2\","
                "\"publishes\":\"activated_bf16\","
                "\"fused_silu_gate\":true,\"logical_hits\":" +
                std::to_string(
                    record.native_nvfp4_true_large_m_gate_up_hits) +
                "},\"down\":{\"identity\":\"DownD2\","
                "\"publishes\":\"prompt_residual_bf16\","
                "\"fused_in_place_residual\":true,\"logical_hits\":" +
                std::to_string(record.native_nvfp4_true_large_m_down_hits) +
                "}},\"count_validation\":{\"complete\":";
      output += g2_d2_package_counts_complete ? "true" : "false";
      output += ",\"expected_layer_panel_hits\":" +
                std::to_string(record.operator_panel_executor_hits) +
                ",\"logical_panel_count\":" +
                std::to_string(g2_d2_panel_count) +
                ",\"expected_nvfp4_logical_projection_hits\":" +
                std::to_string(g2_d2_expected_nvfp4_projection_hits) +
                ",\"observed_nvfp4_logical_projection_hits\":" +
                std::to_string(
                    record.native_nvfp4_true_large_m_projection_hits) +
                ",\"expected_nvfp4_physical_launches\":" +
                std::to_string(g2_d2_expected_nvfp4_projection_hits) +
                ",\"observed_nvfp4_physical_launches\":" +
                std::to_string(
                    record.native_nvfp4_true_large_m_physical_launches) +
                ",\"expected_fp8_companion_projection_hits\":" +
                std::to_string(g2_d2_expected_fp8_projection_hits) +
                ",\"observed_fp8_companion_projection_hits\":" +
                std::to_string(
                    record.nvfp4_true_large_m_route_fp8_projection_hits) +
                ",\"attention_route_complete\":";
      output += g2_d2_attention_counts_complete ? "true" : "false";
      output += ",\"expected_native_attention_hits\":" +
                std::to_string(
                    g2_d2_nvfp4_candidate_uses_q64_attention ||
                            g2_d2_nvfp4_candidate_uses_q128_v4_attention ||
                            g2_d2_nvfp4_candidate_uses_flashinfer_attention
                        ? g2_d2_expected_native_attention_hits
                        : 0U) +
                "}}";
    } else if (true_large_m_nvfp4_candidate_v7) {
      output += ",\"projection_tactic\":";
      append_json_string(output,
                         "native-nvfp4-true-large-m-operator-panel");
      output += ",\"attention_tactic\":";
      append_json_string(
          output, true_large_m_nvfp4_candidate_uses_flashinfer_attention
                      ? "native-flashinfer-exact-panel"
                  : true_large_m_nvfp4_candidate_uses_q128_v4_attention
                      ? "native-group-q128-v4-panel"
                  : true_large_m_nvfp4_candidate_uses_q64_attention
                      ? "native-group-q64-panel"
                      : "exact-segmented");
      output += ",\"operator_panel_executor_hits\":" +
                std::to_string(record.operator_panel_executor_hits) +
                ",\"native_flashinfer_exact_panel_hits\":" +
                std::to_string(record.native_flashinfer_exact_panel_hits) +
                ",\"native_group_q64_panel_hits\":" +
                std::to_string(record.native_group_q64_panel_hits) +
                ",\"native_group_q128_v4_panel_hits\":" +
                std::to_string(record.native_group_q128_v4_panel_hits) +
                ",\"generic_qt2_hits\":" +
                std::to_string(record.generic_qt2_hits) +
                ",\"nvfp4_true_large_m_route_fp8_projection_hits\":" +
                std::to_string(
                    record.nvfp4_true_large_m_route_fp8_projection_hits) +
                ",\"nvfp4_true_large_m_route_fp8_projection_bulk_hits\":" +
                std::to_string(record
                                   .nvfp4_true_large_m_route_fp8_projection_bulk_hits) +
                ",\"nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits\":" +
                std::to_string(
                    record
                        .nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits) +
                ",\"nvfp4_true_large_m_route_fp8_projection_physical_launches\":" +
                std::to_string(
                    record
                        .nvfp4_true_large_m_route_fp8_projection_physical_launches) +
                ",\"native_nvfp4_true_large_m_projection_hits\":" +
                std::to_string(
                    record.native_nvfp4_true_large_m_projection_hits) +
                ",\"native_nvfp4_true_large_m_gate_up_hits\":" +
                std::to_string(
                    record.native_nvfp4_true_large_m_gate_up_hits) +
                ",\"native_nvfp4_true_large_m_down_hits\":" +
                std::to_string(record.native_nvfp4_true_large_m_down_hits) +
                ",\"native_nvfp4_true_large_m_physical_launches\":" +
                std::to_string(
                    record.native_nvfp4_true_large_m_physical_launches);
    } else if (flashinfer_exact_candidate_v6) {
      output += ",\"projection_tactic\":";
      append_json_string(
          output,
          flashinfer_exact_candidate_uses_native_large_m_projection
              ? "native-quantized-large-m-operator-panel"
          : flashinfer_exact_candidate_uses_segmented_projection
              ? "segmented-marlin-operator-panel"
              : "exact-segmented");
      output += ",\"attention_tactic\":";
      append_json_string(output, "native-flashinfer-exact-panel");
      output += ",\"operator_panel_executor_hits\":" +
                std::to_string(record.operator_panel_executor_hits) +
                ",\"native_flashinfer_exact_panel_hits\":" +
                std::to_string(record.native_flashinfer_exact_panel_hits) +
                ",\"native_group_q64_panel_hits\":" +
                std::to_string(record.native_group_q64_panel_hits) +
                ",\"native_group_q128_v4_panel_hits\":" +
                std::to_string(record.native_group_q128_v4_panel_hits) +
                ",\"generic_qt2_hits\":" +
                std::to_string(record.generic_qt2_hits) +
                ",\"segmented_panel_projection_hits\":" +
                std::to_string(record.segmented_panel_projection_hits) +
                ",\"segmented_panel_projection_physical_launches\":" +
                std::to_string(
                    record.segmented_panel_projection_physical_launches) +
                ",\"native_large_m_projection_hits\":" +
                std::to_string(record.native_large_m_projection_hits) +
                ",\"native_large_m_projection_bulk_hits\":" +
                std::to_string(record.native_large_m_projection_bulk_hits) +
                ",\"native_large_m_projection_oracle_partial_hits\":" +
                std::to_string(
                    record.native_large_m_projection_oracle_partial_hits) +
                ",\"native_large_m_projection_physical_launches\":" +
                std::to_string(
                    record.native_large_m_projection_physical_launches);
    } else if (candidate_v3) {
      output += ",\"attention_tactic\":";
      append_json_string(output, candidate_q128_v3
                                     ? "native-group-q128-v4-panel"
                                     : "native-group-q64-panel");
      output += ",\"operator_panel_executor_hits\":" +
                std::to_string(record.operator_panel_executor_hits) +
                ",\"native_group_q64_panel_hits\":" +
                std::to_string(record.native_group_q64_panel_hits) +
                ",\"native_group_q128_v4_panel_hits\":" +
                std::to_string(record.native_group_q128_v4_panel_hits) +
                ",\"generic_qt2_hits\":" +
                std::to_string(record.generic_qt2_hits);
    } else if (projection_candidate_v4) {
      output += ",\"projection_tactic\":";
      append_json_string(output, "segmented-marlin-operator-panel");
      output += ",\"attention_tactic\":";
      append_json_string(
          output, projection_candidate_uses_q128_v4_attention
                      ? "native-group-q128-v4-panel"
                  : projection_candidate_uses_q64_attention
                      ? "native-group-q64-panel"
                      : "exact-segmented");
      output += ",\"operator_panel_executor_hits\":" +
                std::to_string(record.operator_panel_executor_hits) +
                ",\"native_group_q64_panel_hits\":" +
                std::to_string(record.native_group_q64_panel_hits) +
                ",\"native_group_q128_v4_panel_hits\":" +
                std::to_string(record.native_group_q128_v4_panel_hits) +
                ",\"generic_qt2_hits\":" +
                std::to_string(record.generic_qt2_hits) +
                ",\"segmented_panel_projection_hits\":" +
                std::to_string(record.segmented_panel_projection_hits) +
                ",\"segmented_panel_projection_physical_launches\":" +
                std::to_string(
                    record.segmented_panel_projection_physical_launches);
    } else if (native_large_m_candidate_v5) {
      output += ",\"projection_tactic\":";
      append_json_string(output,
                         "native-quantized-large-m-operator-panel");
      output += ",\"attention_tactic\":";
      append_json_string(
          output, native_large_m_candidate_uses_q128_v4_attention
                      ? "native-group-q128-v4-panel"
                  : native_large_m_candidate_uses_q64_attention
                      ? "native-group-q64-panel"
                      : "exact-segmented");
      output += ",\"operator_panel_executor_hits\":" +
                std::to_string(record.operator_panel_executor_hits) +
                ",\"native_group_q64_panel_hits\":" +
                std::to_string(record.native_group_q64_panel_hits) +
                ",\"native_group_q128_v4_panel_hits\":" +
                std::to_string(record.native_group_q128_v4_panel_hits) +
                ",\"generic_qt2_hits\":" +
                std::to_string(record.generic_qt2_hits) +
                ",\"native_large_m_projection_hits\":" +
                std::to_string(record.native_large_m_projection_hits) +
                ",\"native_large_m_projection_bulk_hits\":" +
                std::to_string(record.native_large_m_projection_bulk_hits) +
                ",\"native_large_m_projection_oracle_partial_hits\":" +
                std::to_string(
                    record.native_large_m_projection_oracle_partial_hits) +
                ",\"native_large_m_projection_physical_launches\":" +
                std::to_string(
                    record.native_large_m_projection_physical_launches);
    }
  }
  output += "},\"route\":{\"scope\":\"request_witness\","
            "\"projection_backend\":{\"available\":true,"
            "\"scope\":\"configured_engine_fact\",\"value\":";
  append_json_string(output, runtime::to_string(record.projection_backend));
  output += "},\"deployment_plan\":{";
  if (!sealed) {
    output += "\"available\":false,\"reason\":\"not_implemented\"";
  } else {
    output += "\"available\":true,\"scope\":"
              "\"engine_lifetime_sealed_native_plan\",\"id\":";
    append_json_string(output, record.deployment_plan_id);
    if (accuracy_unqualified_candidate) {
      output += ",\"qualification\":";
      append_json_string(output,
                         "accuracy-unqualified-architecture-candidate");
      output += ",\"numerical_contract\":{\"qualified\":false,\"reason\":";
      append_json_string(output,
                         "full-state-accuracy-qualification-not-run");
      output += "}";
    }
  }
  output += "},\"per_operator_route_hits\":";
  append_prefill_route_evidence(output, record.prefill_route_evidence);
  output +=
      ","
      "\"cache_hits\":{\"available\":false,\"reason\":"
      "\"not_instrumented\"},\"disabled_boundaries\":{\"scope\":";
  append_json_string(output, accuracy_unqualified_candidate
                                 ? "architecture_candidate_unqualified"
                                 : "production_contract");
  output +=
      ",\"prefix_cache\":true,\"mtp\":true,"
      "\"cublaslt_production\":true,\"approximate_numerics\":";
  output += accuracy_unqualified_candidate ? "false" : "true";
  output += "}}}";
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
