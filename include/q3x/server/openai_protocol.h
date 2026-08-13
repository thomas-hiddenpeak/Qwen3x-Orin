#pragma once

#include "q3x/runtime/reference_engine.h"

#include <array>
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
  runtime::ReferenceGenerationRoute generation_route =
      runtime::ReferenceGenerationRoute::kReference;
  runtime::ReferencePrefillExecutionMode prefill_execution_mode =
      runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled;
  std::uint64_t prefill_logical_panel_count = 0U;
  runtime::RequestMemoryProfile request_memory_profile =
      runtime::RequestMemoryProfile::kLegacyC512;
  bool bounded_submission_window = false;
  std::uint64_t submission_window_retirements = 0U;
  std::uint64_t operator_panel_executor_hits = 0U;
  std::uint64_t native_group_q64_panel_hits = 0U;
  std::uint64_t native_group_q128_v4_panel_hits = 0U;
  std::uint64_t native_flashinfer_exact_panel_hits = 0U;
  std::uint64_t generic_qt2_hits = 0U;
  std::uint64_t segmented_panel_projection_hits = 0U;
  std::uint64_t segmented_panel_projection_physical_launches = 0U;
  std::uint64_t native_large_m_projection_hits = 0U;
  std::uint64_t native_large_m_projection_bulk_hits = 0U;
  std::uint64_t native_large_m_projection_oracle_partial_hits = 0U;
  std::uint64_t native_large_m_projection_physical_launches = 0U;
  std::uint64_t nvfp4_true_large_m_route_fp8_projection_hits = 0U;
  std::uint64_t nvfp4_true_large_m_route_fp8_projection_bulk_hits = 0U;
  std::uint64_t
      nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits = 0U;
  std::uint64_t nvfp4_true_large_m_route_fp8_projection_physical_launches =
      0U;
  std::uint64_t native_nvfp4_true_large_m_projection_hits = 0U;
  std::uint64_t native_nvfp4_true_large_m_gate_up_hits = 0U;
  std::uint64_t native_nvfp4_true_large_m_down_hits = 0U;
  std::uint64_t native_nvfp4_true_large_m_physical_launches = 0U;
  runtime::LayerMajorPrefillMlpScheduleTactic mlp_schedule_tactic =
      runtime::LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel;
  std::uint64_t route_layer_pass_count = 0U;
  std::uint64_t layer_wide_p40_mlp_layer_hits = 0U;
  std::uint64_t persistent_p40_nvfp4_gate_up_hits = 0U;
  std::uint64_t persistent_p40_nvfp4_down_residual_hits = 0U;
  std::uint64_t persistent_p40_nvfp4_physical_launches = 0U;
  std::uint64_t persistent_p40_fp8_projection_hits = 0U;
  std::uint64_t persistent_p40_fp8_projection_bulk_hits = 0U;
  std::uint64_t persistent_p40_fp8_projection_oracle_partial_hits = 0U;
  std::uint64_t persistent_p40_fp8_projection_physical_launches = 0U;
  // Exact-P40000 whole-core completed-launch witnesses. These counters are
  // deliberately separate from the incumbent operator-panel route so the API
  // record cannot relabel five panel Attention launches as one whole-prompt
  // launch, or report a configured tactic that never reached completion.
  std::uint64_t prompt_wide_p40_whole_core_layer_hits = 0U;
  std::uint64_t prompt_wide_p40_fill_panel_hits = 0U;
  std::uint64_t prompt_wide_p40_prompt_core_hits = 0U;
  std::uint64_t prompt_wide_p40_drain_panel_hits = 0U;
  std::uint64_t prompt_wide_p40_fp8_projection_hits = 0U;
  std::uint64_t prompt_wide_p40_fp8_projection_physical_launches = 0U;
  std::uint64_t prompt_wide_p40_bf16_ab_hits = 0U;
  std::uint64_t prompt_wide_p40_gdn_hits = 0U;
  std::uint64_t native_flashinfer_exact_whole_prompt_hits = 0U;
  std::uint64_t packed_nvfp4_v2_gate_up_hits = 0U;
  std::uint64_t packed_nvfp4_v2_down_hits = 0U;
  std::uint64_t packed_nvfp4_v2_physical_launches = 0U;
  // Independent stock-vLLM-Marlin P40000 projection host-dispatch evidence.
  // These fields are copied only from the successfully committed generation
  // result and never alias the older persistent or packed-NVFP4-v2 ledgers.
  std::uint64_t vllm_marlin_parity_gate_up_hits = 0U;
  std::uint64_t vllm_marlin_parity_down_hits = 0U;
  std::uint64_t vllm_marlin_parity_physical_launches = 0U;
  std::uint64_t vllm_marlin_parity_standalone_silu_launches = 0U;
  std::uint64_t vllm_marlin_parity_standalone_residual_launches = 0U;
  std::uint64_t vllm_marlin_parity_lock_clear_operations = 0U;
  std::array<runtime::ReferenceP40VllmMarlinParityLayerCompletionReceipt,
             runtime::kReferenceDecoderLayerCount>
      vllm_marlin_parity_layer_completion_receipts{};
  std::uint64_t vllm_marlin_parity_layer_completion_receipt_count = 0U;
  // Complete target-AOT engine evidence. These fields are populated only
  // from a committed executor receipt and its engine-lifetime authenticated
  // 256-artifact owner; configured selectors cannot synthesize them.
  std::size_t target_aot_complete_projection_artifacts = 0U;
  std::size_t target_aot_complete_projection_sources = 0U;
  std::string target_aot_complete_projection_catalog_sha256;
  std::uint64_t target_aot_admission_epoch = 0U;
  std::uint64_t target_aot_transaction_epoch = 0U;
  std::uint64_t target_aot_completed_layers = 0U;
  std::uint64_t target_aot_completed_gdn_layers = 0U;
  std::uint64_t target_aot_completed_full_attention_layers = 0U;
  std::uint64_t target_aot_completed_attention_panels = 0U;
  std::uint64_t target_aot_recorded_layer_events = 0U;
  std::uint64_t target_aot_recorded_global_events = 0U;
  bool target_aot_transaction_committed = false;
  bool target_aot_handoff_result_observed = false;
  bool target_aot_handoff_complete = false;
  bool target_aot_used_fallback = false;
  bool target_aot_used_mtp = false;
  bool target_aot_used_cublaslt = false;
  bool target_aot_used_jit = false;
  // Append-only phase authority.  A false value means a retained diagnostic
  // duration must not be promoted as pure Prefill; external API TTFT/total
  // evidence remains independently observable.
  bool pure_prefill_phase_qualified = true;
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
