#pragma once

#include "q3x/runtime/reference_engine.h"
#include "q3x/server/openai_protocol.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace q3x::server {

inline constexpr std::size_t kMaximumEvaluationApiKeyBytes = 4'096U;

struct EvaluationApiKeyLoadResult {
  std::optional<std::string> value;
  std::string error;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value();
  }
};

// Opens one owner-only regular file without following symlinks and loads one
// bounded ASCII Bearer token. A single trailing LF or CRLF is ignored.
[[nodiscard]] EvaluationApiKeyLoadResult load_evaluation_api_key_file(
    const std::filesystem::path& path);

// Installed/default evaluation-server authority.  A production profile is a
// complete execution and capacity selection, not a bag of independently
// mutable command-line tactics.  The first profile deliberately retains the
// accuracy-qualified Legacy-C512 arithmetic while admitting the P40000
// product prompt together with the ordinary 4096-token output ceiling.
enum class EvaluationProductionProfile : std::uint8_t {
  kNone = 0,
  kP40ExactLegacyC512,
};

struct EvaluationProductionDeploymentPlan {
  EvaluationProductionProfile profile =
      EvaluationProductionProfile::kNone;
  std::string_view id;
  std::string_view decode_route_id;
  std::uint32_t target_prompt_tokens = 0U;
  std::uint32_t maximum_output_tokens = 0U;
  std::uint32_t max_sequence_length = 0U;
  std::uint32_t prefill_chunk_size = 0U;
  std::uint64_t request_arena_bytes = 0U;
  std::uint64_t min_free_bytes_after_create = 0U;
  std::size_t prefill_supermatrix_projections = 0U;
  std::uint64_t prefill_supermatrix_sidecar_bytes = 0U;
  std::size_t decode_fp8_output_layers = 0U;
  std::uint64_t decode_fp8_output_sidecar_bytes = 0U;
  std::size_t decode_gate_up_layers = 0U;
  std::uint64_t decode_gate_up_sidecar_bytes = 0U;
  std::size_t decode_down_scale6_layers = 0U;
  std::uint64_t decode_down_scale6_sidecar_bytes = 0U;
  std::size_t decode_down_consumer_order_layers = 0U;
  std::uint64_t decode_down_consumer_order_sidecar_bytes = 0U;
  std::uint64_t decode_retained_sidecar_bytes = 0U;
  std::uint64_t retained_acceleration_sidecar_bytes = 0U;
  runtime::ReferenceDecodeGraphCachePolicy decode_graph_cache_policy =
      runtime::ReferenceDecodeGraphCachePolicy::kDisabled;
  std::uint32_t decode_graph_first_position = 0U;
  std::uint32_t decode_graph_last_position = 0U;
  std::size_t decode_graph_slots = 0U;
  runtime::ProjectionBackend projection_backend =
      runtime::ProjectionBackend::kReference;
  runtime::ReferencePrefillExecutionMode prefill_execution_mode =
      runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled;
  runtime::LayerMajorPrefillFullAttentionTactic
      prefill_full_attention_tactic =
          runtime::LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512;
  runtime::LayerMajorPrefillProjectionTactic prefill_projection_tactic =
      runtime::LayerMajorPrefillProjectionTactic::kExactSegmentedC512;
};

inline constexpr EvaluationProductionDeploymentPlan
    kP40ExactLegacyC512ProductionPlan{
        EvaluationProductionProfile::kP40ExactLegacyC512,
        "q3x.sm87.production.p40.legacy-c512-exact.v2",
        "q3x.sm87.decode.coupled-feed-down-consumer-order.v1",
        40'000U,
        4'096U,
        44'095U,
        runtime::kMaximumRequestPrefillChunkSize,
        3'070'908'416ULL,
        8ULL * 1024ULL * 1024ULL * 1024ULL,
        runtime::kFp8PrefillSupermatrixProjectionCount,
        runtime::kQwen36Fp8PrefillSupermatrixSidecarBytes,
        runtime::kQwen36DenseLayerCount,
        runtime::kQwen36Fp8M1OutputProjectionAosoa4PreswizzledBytes,
        runtime::kQwen36DenseLayerCount,
        runtime::kQwen36NvFp4GateUpCoupledFeedBytes,
        runtime::kQwen36NvFp4DownScale6LayerCount,
        runtime::kQwen36NvFp4DownScale6SidecarBytes,
        runtime::kQwen36NvFp4DownScale6LayerCount,
        runtime::kQwen36NvFp4DownConsumerOrderBytes,
        11'013'898'240ULL,
        18'228'101'120ULL,
        runtime::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions,
        19U,
        43U,
        25U,
        runtime::ProjectionBackend::kSm87WeightOnly,
        runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled,
        runtime::LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512,
        runtime::LayerMajorPrefillProjectionTactic::kExactSegmentedC512,
    };

static_assert(kP40ExactLegacyC512ProductionPlan.max_sequence_length ==
              kP40ExactLegacyC512ProductionPlan.target_prompt_tokens +
                  kP40ExactLegacyC512ProductionPlan.maximum_output_tokens -
                  1U);
static_assert(
    kP40ExactLegacyC512ProductionPlan.decode_down_consumer_order_sidecar_bytes ==
    kP40ExactLegacyC512ProductionPlan.decode_down_consumer_order_layers *
        runtime::kNvFp4DownConsumerOrderWeightBytesPerProjection);
static_assert(
    kP40ExactLegacyC512ProductionPlan.decode_down_scale6_sidecar_bytes ==
    kP40ExactLegacyC512ProductionPlan.decode_down_scale6_layers *
        runtime::kNvFp4DownScale6SidecarBytesPerProjection);
static_assert(
    kP40ExactLegacyC512ProductionPlan.decode_retained_sidecar_bytes ==
    kP40ExactLegacyC512ProductionPlan.decode_fp8_output_sidecar_bytes +
        kP40ExactLegacyC512ProductionPlan.decode_gate_up_sidecar_bytes +
        kP40ExactLegacyC512ProductionPlan.decode_down_scale6_sidecar_bytes +
        kP40ExactLegacyC512ProductionPlan
            .decode_down_consumer_order_sidecar_bytes);
static_assert(
    kP40ExactLegacyC512ProductionPlan.retained_acceleration_sidecar_bytes ==
    kP40ExactLegacyC512ProductionPlan.prefill_supermatrix_sidecar_bytes +
        kP40ExactLegacyC512ProductionPlan.decode_retained_sidecar_bytes);

[[nodiscard]] constexpr bool is_valid_evaluation_production_profile(
    const EvaluationProductionProfile profile) noexcept {
  return profile == EvaluationProductionProfile::kNone ||
         profile == EvaluationProductionProfile::kP40ExactLegacyC512;
}

[[nodiscard]] constexpr std::string_view to_string(
    const EvaluationProductionProfile profile) noexcept {
  switch (profile) {
    case EvaluationProductionProfile::kNone:
      return "none";
    case EvaluationProductionProfile::kP40ExactLegacyC512:
      return kP40ExactLegacyC512ProductionPlan.id;
  }
  return "unknown";
}

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
  std::filesystem::path api_key_file;
  std::string bind_address = "127.0.0.1";
  std::uint16_t port = 8000U;
  std::string served_model = "qwen3.6-27b-nvfp4";
  EvaluationProductionProfile production_profile =
      EvaluationProductionProfile::kP40ExactLegacyC512;
  std::uint32_t max_sequence_length =
      kP40ExactLegacyC512ProductionPlan.max_sequence_length;
  std::uint32_t maximum_output_tokens =
      kP40ExactLegacyC512ProductionPlan.maximum_output_tokens;
  std::uint32_t prefill_chunk_size =
      kP40ExactLegacyC512ProductionPlan.prefill_chunk_size;
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
      kP40ExactLegacyC512ProductionPlan.request_arena_bytes;
  std::uint64_t request_min_free_bytes_after_create =
      kP40ExactLegacyC512ProductionPlan.min_free_bytes_after_create;
  runtime::ProjectionBackend projection_backend =
      kP40ExactLegacyC512ProductionPlan.projection_backend;
  runtime::ReferencePrefillExecutionMode prefill_execution_mode =
      kP40ExactLegacyC512ProductionPlan.prefill_execution_mode;
  runtime::LayerMajorPrefillFullAttentionTactic
      prefill_full_attention_tactic =
          kP40ExactLegacyC512ProductionPlan.prefill_full_attention_tactic;
  runtime::LayerMajorPrefillProjectionTactic prefill_projection_tactic =
      kP40ExactLegacyC512ProductionPlan.prefill_projection_tactic;
  // Single typed acknowledgement for the accuracy-unqualified v10 baseline.
  // The server rejects both a route without this acknowledgement and an
  // acknowledgement whose fixed P40000 configuration has been altered.
  EvaluationDevelopmentRoute development_route =
      EvaluationDevelopmentRoute::kNone;
  // Profiling-only observability. Ordinary API requests keep the runtime's
  // zero-range default unless the server is launched with the explicit flag.
  bool emit_nvtx_phase_ranges = false;
};

[[nodiscard]] inline bool is_p40_exact_legacy_c512_production_profile(
    const EvaluationServerOptions& options) noexcept {
  const EvaluationProductionDeploymentPlan& plan =
      kP40ExactLegacyC512ProductionPlan;
  return options.production_profile == plan.profile &&
         options.development_route == EvaluationDevelopmentRoute::kNone &&
         options.max_sequence_length == plan.max_sequence_length &&
         options.maximum_output_tokens == plan.maximum_output_tokens &&
         options.prefill_chunk_size == plan.prefill_chunk_size &&
         options.request_max_arena_bytes == plan.request_arena_bytes &&
         options.request_min_free_bytes_after_create ==
             plan.min_free_bytes_after_create &&
         options.projection_backend == plan.projection_backend &&
         options.prefill_execution_mode == plan.prefill_execution_mode &&
         options.prefill_full_attention_tactic ==
             plan.prefill_full_attention_tactic &&
         options.prefill_projection_tactic ==
             plan.prefill_projection_tactic;
}

[[nodiscard]] inline bool is_p40_whole_core_v10_fixed_profile(
    const EvaluationServerOptions& options) noexcept {
  return options.development_route ==
             EvaluationDevelopmentRoute::kP40WholeCoreV10 &&
         options.production_profile == EvaluationProductionProfile::kNone &&
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
