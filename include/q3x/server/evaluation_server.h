#pragma once

#include "q3x/runtime/reference_engine.h"
#include "q3x/server/openai_protocol.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace q3x::server {

// Explicit acknowledgement for evaluation-only routes that are intentionally
// outside the numerical and production contract. A non-default value never
// changes the installed/default runner; the matching build inventory and the
// complete fixed request profile must also be present.
enum class EvaluationDevelopmentRoute : std::uint8_t {
  kNone = 0,
  kP40WholeCoreV10,
};

[[nodiscard]] constexpr bool is_valid_evaluation_development_route(
    const EvaluationDevelopmentRoute route) noexcept {
  return route == EvaluationDevelopmentRoute::kNone ||
         route == EvaluationDevelopmentRoute::kP40WholeCoreV10;
}

[[nodiscard]] constexpr std::string_view to_string(
    const EvaluationDevelopmentRoute route) noexcept {
  switch (route) {
    case EvaluationDevelopmentRoute::kNone:
      return "none";
    case EvaluationDevelopmentRoute::kP40WholeCoreV10:
      return "p40-whole-core-v10";
  }
  return "unknown";
}

struct EvaluationServerOptions {
  std::filesystem::path model_directory;
  std::string bind_address = "127.0.0.1";
  std::uint16_t port = 8000U;
  std::string served_model = "qwen3.6-27b-nvfp4";
  std::uint32_t max_sequence_length = 8'192U;
  std::uint32_t maximum_output_tokens = 4'096U;
  std::uint32_t prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  // One thread may wait on each admitted batch-one request. Keep enough
  // threads for the active request, every bounded queued request, and one
  // control-plane/overload response.
  std::size_t ingress_threads = 10U;
  std::size_t accepted_connection_capacity = 16U;
  std::size_t inference_queue_capacity = 8U;
  std::size_t stream_event_capacity = 16U;
  std::uint32_t read_timeout_milliseconds = 10'000U;
  std::uint32_t write_timeout_milliseconds = 5'000U;
  std::uint64_t request_max_arena_bytes =
      2ULL * 1024ULL * 1024ULL * 1024ULL;
  std::uint64_t request_min_free_bytes_after_create =
      8ULL * 1024ULL * 1024ULL * 1024ULL;
  runtime::ProjectionBackend projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  runtime::ReferencePrefillExecutionMode prefill_execution_mode =
      runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled;
  runtime::LayerMajorPrefillFullAttentionTactic
      prefill_full_attention_tactic =
          runtime::LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512;
  runtime::LayerMajorPrefillProjectionTactic prefill_projection_tactic =
      runtime::LayerMajorPrefillProjectionTactic::kExactSegmentedC512;
  // Single typed acknowledgement for the accuracy-unqualified v10 baseline.
  // The server rejects both a route without this acknowledgement and an
  // acknowledgement whose fixed P40000 configuration has been altered.
  EvaluationDevelopmentRoute development_route =
      EvaluationDevelopmentRoute::kNone;
  // Profiling-only observability. Ordinary API requests keep the runtime's
  // zero-range default unless the server is launched with the explicit flag.
  bool emit_nvtx_phase_ranges = false;
};

[[nodiscard]] inline bool is_p40_whole_core_v10_fixed_profile(
    const EvaluationServerOptions& options) noexcept {
  return options.development_route ==
             EvaluationDevelopmentRoute::kP40WholeCoreV10 &&
         options.max_sequence_length == 40'001U &&
         options.maximum_output_tokens == 1U &&
         options.prefill_chunk_size ==
             runtime::kMaximumRequestPrefillChunkSize &&
         options.prefill_execution_mode == runtime::
             ReferencePrefillExecutionMode::kWholeRequestLayerMajor &&
         options.prefill_full_attention_tactic == runtime::
             LayerMajorPrefillFullAttentionTactic::
                 kNativeFlashInferExactWholePrompt &&
         options.prefill_projection_tactic == runtime::
             LayerMajorPrefillProjectionTactic::
                 kNativePromptWideP40WholeCore &&
         options.projection_backend ==
             runtime::ProjectionBackend::kSm87WeightOnly &&
         options.request_max_arena_bytes == 8'640'542'976ULL &&
         options.request_min_free_bytes_after_create ==
             4ULL * 1024ULL * 1024ULL * 1024ULL &&
         options.inference_queue_capacity == 1U &&
         options.ingress_threads == 3U;
}

[[nodiscard]] inline bool is_p40_whole_core_v10_request(
    const OpenAIRequest& request) noexcept {
  return request.endpoint == OpenAIEndpoint::kCompletions &&
         request.prompt_kind == OpenAIPromptKind::kTokenIds &&
         request.prompt_token_ids.size() == 40'000U &&
         request.max_tokens == 1U && request.stream && request.include_usage;
}

// Loads one resident model, starts a bounded HTTP ingress and exactly one
// inference worker, and blocks until stop_requested becomes true or a fatal
// server error occurs. The listener is not exposed until model loading has
// succeeded.
[[nodiscard]] int run_evaluation_server(
    const EvaluationServerOptions& options,
    std::atomic<bool>& stop_requested,
    std::string& error_message);

}  // namespace q3x::server
