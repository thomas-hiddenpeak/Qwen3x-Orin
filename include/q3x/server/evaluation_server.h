#pragma once

#include "q3x/runtime/reference_engine.h"
#include "q3x/runtime/sm87_target_aot_request_state.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace q3x::server {

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
  // Profiling-only observability. Ordinary API requests keep the runtime's
  // zero-range default unless the server is launched with the explicit flag.
  bool emit_nvtx_phase_ranges = false;
  // Append-only complete-engine selector. The default preserves the existing
  // evaluation adapter. The target route is an exact P40000 -> 1 contract and
  // fails closed unless its complete executor is compiled and admitted.
  runtime::ReferenceGenerationRoute engine_route =
      runtime::ReferenceGenerationRoute::kReference;
};

enum class EvaluationServerEngineRouteContractError : std::uint8_t {
  kNone = 0U,
  kInvalidRoute,
  kTargetRequiresExactP40001Capacity,
  kTargetRequiresOneOutputToken,
  kTargetRequiresSm87ProjectionBackend,
  kTargetRequiresExactRequestArenaCapacity,
  kTargetRequiresFixedC512HostChunk,
  kTargetRequiresNeutralLegacyPrefillSelectors,
  kPrefillPlanUnavailable,
};

[[nodiscard]] inline EvaluationServerEngineRouteContractError
validate_evaluation_server_engine_route_contract(
    const EvaluationServerOptions& options) noexcept {
  if (!runtime::is_valid_reference_generation_route(options.engine_route)) {
    return EvaluationServerEngineRouteContractError::kInvalidRoute;
  }
  if (options.engine_route ==
      runtime::ReferenceGenerationRoute::kReference) {
    return EvaluationServerEngineRouteContractError::kNone;
  }
  // Both complete P40 routes own their execution geometry.  Neither may
  // inherit a legacy layer-major tactic as an implicit implementation choice.
  if (!runtime::is_exact_p40_generation_route(options.engine_route)) {
    return EvaluationServerEngineRouteContractError::kInvalidRoute;
  }
  if (options.max_sequence_length !=
      runtime::kSm87TargetAotP40RequestCapacityTokens) {
    return EvaluationServerEngineRouteContractError::
        kTargetRequiresExactP40001Capacity;
  }
  if (options.maximum_output_tokens != 1U) {
    return EvaluationServerEngineRouteContractError::
        kTargetRequiresOneOutputToken;
  }
  if (options.projection_backend !=
      runtime::ProjectionBackend::kSm87WeightOnly) {
    return EvaluationServerEngineRouteContractError::
        kTargetRequiresSm87ProjectionBackend;
  }
  if (options.request_max_arena_bytes <
      runtime::kSm87TargetAotP40RequestArenaBytes) {
    return EvaluationServerEngineRouteContractError::
        kTargetRequiresExactRequestArenaCapacity;
  }
  if (options.prefill_chunk_size !=
      runtime::kMaximumRequestPrefillChunkSize) {
    return EvaluationServerEngineRouteContractError::
        kTargetRequiresFixedC512HostChunk;
  }
  if (options.prefill_execution_mode !=
          runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled ||
      options.prefill_full_attention_tactic !=
          runtime::LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512 ||
      options.prefill_projection_tactic !=
          runtime::LayerMajorPrefillProjectionTactic::kExactSegmentedC512) {
    return EvaluationServerEngineRouteContractError::
        kTargetRequiresNeutralLegacyPrefillSelectors;
  }
  if (options.engine_route ==
          runtime::ReferenceGenerationRoute::kSm87BulkV2P40 &&
      !runtime::is_reference_generation_route_compiled(
          options.engine_route)) {
    return EvaluationServerEngineRouteContractError::
        kPrefillPlanUnavailable;
  }
  return EvaluationServerEngineRouteContractError::kNone;
}

[[nodiscard]] constexpr std::string_view to_string(
    const EvaluationServerEngineRouteContractError error) noexcept {
  switch (error) {
    case EvaluationServerEngineRouteContractError::kNone:
      return "none";
    case EvaluationServerEngineRouteContractError::kInvalidRoute:
      return "unknown complete-engine route";
    case EvaluationServerEngineRouteContractError::
        kTargetRequiresExactP40001Capacity:
      return "complete P40 routes require --max-sequence-length 40001";
    case EvaluationServerEngineRouteContractError::
        kTargetRequiresOneOutputToken:
      return "complete P40 routes require --max-output-tokens 1";
    case EvaluationServerEngineRouteContractError::
        kTargetRequiresSm87ProjectionBackend:
      return "complete P40 routes require --projection-backend sm87";
    case EvaluationServerEngineRouteContractError::
        kTargetRequiresExactRequestArenaCapacity:
      return "complete P40 routes require --request-max-arena-bytes of at "
             "least 5075652608";
    case EvaluationServerEngineRouteContractError::
        kTargetRequiresFixedC512HostChunk:
      return "complete P40 routes require --prefill-chunk-size 512";
    case EvaluationServerEngineRouteContractError::
        kTargetRequiresNeutralLegacyPrefillSelectors:
      return "complete P40 routes cannot be combined with layer-major or "
             "non-default legacy Prefill tactic selectors";
    case EvaluationServerEngineRouteContractError::kPrefillPlanUnavailable:
      return "selected complete-engine Prefill plan is unavailable in this "
             "binary";
  }
  return "unknown complete-engine route contract error";
}

// A bounded Prefill cancellation publishes no partial response.  The worker
// closes the event stream instead of relabelling the expected cancellation as
// an internal server failure, even if the caller flag changes between the
// probe and result handling.
[[nodiscard]] constexpr bool
close_cancelled_generation_without_error_response(
    const bool request_cancelled, const bool server_stopping,
    const runtime::ReferenceEngineError diagnostic) noexcept {
  return request_cancelled || server_stopping ||
         diagnostic == runtime::ReferenceEngineError::kCancelled;
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
