#pragma once

#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/prefill_execution_plan.h"
#include "q3x/runtime/reference_runner.h"
#include "q3x/runtime/request_state.h"
#include "q3x/runtime/resident_weights.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime {

struct Sm87TargetAotProjectionDevicePreparationStats;
struct Sm87TargetAotCompleteDevicePreparationStats;
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
namespace reference_engine_test_detail {
class Sm87TargetAotLayer0M192OracleAccess;
}
#endif

inline constexpr std::uint32_t kQwen36ImEndTokenId = 248'046U;

enum class ReferenceEngineError : std::uint8_t {
  kNone = 0,
  kInvalidArgument,
  kCapacityExceeded,
  kArithmeticOverflow,
  kTokenizerFailure,
  kResidentLoadFailure,
  kWeightBindFailure,
  kRequestStateFailure,
  kRunnerFactoryFailure,
  kRunnerStepFailure,
  kRunnerResetFailure,
  kMissingLogits,
  kMissingTiming,
  kDecodeFailure,
  kTraceFailure,
  kAllocationFailure,
  kMissingPrediction,
  // The caller selected a Prefill route that this engine instance did not
  // bind and provision at creation time.
  kPrefillPlanUnavailable,
  kCancelled,
};

struct ReferenceEngineDiagnostic {
  ReferenceEngineError code = ReferenceEngineError::kNone;
  std::string stage;
  std::string message;
  std::string context;
  int dependency_error = 0;
  int cuda_error = 0;
  std::size_t layer = kReferenceNoLayer;
  std::string operation;
  std::uint64_t retired_prefill_quanta = 0U;
};

enum class ReferenceDecodeGraphCachePolicy : std::uint8_t {
  kDisabled = 0,
  kSm87ShortPositions,
};

[[nodiscard]] constexpr bool is_valid_reference_decode_graph_cache_policy(
    const ReferenceDecodeGraphCachePolicy policy) noexcept {
  return policy == ReferenceDecodeGraphCachePolicy::kDisabled ||
         policy == ReferenceDecodeGraphCachePolicy::kSm87ShortPositions;
}

// Stable host/API identity for the complete generation implementation. This
// is deliberately above ProjectionBackend and ReferencePrefillExecutionMode:
// selecting one of those lower-level mechanisms must never imply that the
// complete target-AOT runner was selected. The target route remains
// fail-closed until its Engine and server boundaries are explicitly wired.
enum class ReferenceGenerationRoute : std::uint8_t {
  kReference = 0,
  kSm87TargetAotP40,
};

[[nodiscard]] constexpr bool is_valid_reference_generation_route(
    const ReferenceGenerationRoute route) noexcept {
  return route == ReferenceGenerationRoute::kReference ||
         route == ReferenceGenerationRoute::kSm87TargetAotP40;
}

[[nodiscard]] constexpr std::string_view to_string(
    const ReferenceGenerationRoute route) noexcept {
  switch (route) {
    case ReferenceGenerationRoute::kReference:
      return "reference";
    case ReferenceGenerationRoute::kSm87TargetAotP40:
      return "sm87-target-aot-p40";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::optional<ReferenceGenerationRoute>
parse_reference_generation_route(const std::string_view text) noexcept {
  if (text == "reference") {
    return ReferenceGenerationRoute::kReference;
  }
  if (text == "sm87-target-aot-p40") {
    return ReferenceGenerationRoute::kSm87TargetAotP40;
  }
  return std::nullopt;
}

// Host-observable Prefill scheduling identity. The legacy value covers the
// existing public C512-bounded controller (including smaller canonical
// tiles); the layer-major value is reserved for the explicit default-off
// whole-request seam whose internal logical panels are bounded by C8192.
enum class ReferencePrefillExecutionMode : std::uint8_t {
  kLegacyC512Tiled = 0,
  kWholeRequestLayerMajor,
};

inline constexpr std::string_view kLayerMajorOperatorPanelDeploymentPlanId =
    "q3x.sm87.exact.layer-major-c8192.operator-panel.v3";
inline constexpr std::string_view
    kLayerMajorNativeGroupQ64PanelDeploymentPlanId =
        "q3x.sm87.ac-prefill-layermajor-8k.native-group-q64-panel.v1";
inline constexpr std::string_view
    kLayerMajorNativeGroupQ128V4PanelDeploymentPlanId =
        "q3x.sm87.ac-prefill-layermajor-8k.native-group-q128-v4-panel.v1";
inline constexpr std::string_view
    kLayerMajorNativeFlashInferExactPanelDeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2.exact-segmented-projection."
        "native-flashinfer-exact-panel-attention.v1";
inline constexpr std::string_view
    kLayerMajorSegmentedMarlinProjectionDeploymentPlanId =
        "q3x.sm87.ac-prefill-layermajor-8k.segmented-marlin-operator-panel."
        "exact-segmented-attention.v1";
inline constexpr std::string_view
    kLayerMajorSegmentedMarlinProjectionGroupQ64DeploymentPlanId =
        "q3x.sm87.ac-prefill-layermajor-8k.segmented-marlin-operator-panel."
        "native-group-q64-attention.v1";
inline constexpr std::string_view
    kLayerMajorSegmentedMarlinProjectionGroupQ128V4DeploymentPlanId =
        "q3x.sm87.ac-prefill-layermajor-8k.segmented-marlin-operator-panel."
        "native-group-q128-v4-attention.v1";
inline constexpr std::string_view
    kLayerMajorSegmentedMarlinProjectionFlashInferExactDeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "segmented-marlin-operator-panel."
        "native-flashinfer-exact-panel-attention.v1";
inline constexpr std::string_view
    kLayerMajorNativeQuantizedLargeMProjectionDeploymentPlanId =
        "q3x.sm87.ac-prefill-layermajor-8k."
        "native-quantized-large-m-operator-panel."
        "exact-segmented-attention.v1";
inline constexpr std::string_view
    kLayerMajorNativeQuantizedLargeMProjectionGroupQ64DeploymentPlanId =
        "q3x.sm87.ac-prefill-layermajor-8k."
        "native-quantized-large-m-operator-panel."
        "native-group-q64-attention.v1";
inline constexpr std::string_view
    kLayerMajorNativeQuantizedLargeMProjectionGroupQ128V4DeploymentPlanId =
        "q3x.sm87.ac-prefill-layermajor-8k."
        "native-quantized-large-m-operator-panel."
        "native-group-q128-v4-attention.v1";
inline constexpr std::string_view
    kLayerMajorNativeQuantizedLargeMProjectionFlashInferExactDeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-quantized-large-m-operator-panel."
        "native-flashinfer-exact-panel-attention.v1";
// WP-V2-C1 is deliberately a separate, default-off deployment identity.
// Its NVFP4 Gate+Up and Down launches form one fail-closed true-large-M
// package; none of these ids may describe the older Marlin fallback route.
inline constexpr std::string_view
    kLayerMajorNativeNvfp4TrueLargeMProjectionDeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-true-large-m-operator-panel."
        "exact-segmented-attention.v1";
inline constexpr std::string_view
    kLayerMajorNativeNvfp4TrueLargeMProjectionGroupQ64DeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-true-large-m-operator-panel."
        "native-group-q64-attention.v1";
inline constexpr std::string_view
    kLayerMajorNativeNvfp4TrueLargeMProjectionGroupQ128V4DeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-true-large-m-operator-panel."
        "native-group-q128-v4-attention.v1";
inline constexpr std::string_view
    kLayerMajorNativeNvfp4TrueLargeMProjectionFlashInferExactDeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-true-large-m-operator-panel."
        "native-flashinfer-exact-panel-attention.v1";
// G2/D2 is a new coupled package identity. These v2 deployment
// identifiers cannot be confused with the rejected C1 Gate+Up/Down v1 route.
inline constexpr std::string_view
    kLayerMajorNativeNvfp4G2D2LargeMProjectionDeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-g2-d2-large-m-operator-panel."
        "exact-segmented-attention.v2";
inline constexpr std::string_view
    kLayerMajorNativeNvfp4G2D2LargeMProjectionGroupQ64DeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-g2-d2-large-m-operator-panel."
        "native-group-q64-attention.v2";
inline constexpr std::string_view
    kLayerMajorNativeNvfp4G2D2LargeMProjectionGroupQ128V4DeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-g2-d2-large-m-operator-panel."
        "native-group-q128-v4-attention.v2";
inline constexpr std::string_view
    kLayerMajorNativeNvfp4G2D2LargeMProjectionFlashInferExactDeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-g2-d2-large-m-operator-panel."
        "native-flashinfer-exact-panel-attention.v2";
// Exact-P40000 persistent MLP is a distinct request-wide schedule and must
// never inherit an operator-panel deployment identity.
inline constexpr std::string_view
    kLayerMajorNativeNvfp4PersistentP40MlpDeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-persistent-p40-layer-wide-mlp."
        "exact-segmented-attention.v1";
inline constexpr std::string_view
    kLayerMajorNativeNvfp4PersistentP40MlpGroupQ64DeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-persistent-p40-layer-wide-mlp."
        "native-group-q64-attention.v1";
inline constexpr std::string_view
    kLayerMajorNativeNvfp4PersistentP40MlpGroupQ128V4DeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-persistent-p40-layer-wide-mlp."
        "native-group-q128-v4-attention.v1";
inline constexpr std::string_view
    kLayerMajorNativeNvfp4PersistentP40MlpFlashInferExactDeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-nvfp4-persistent-p40-layer-wide-mlp."
        "native-flashinfer-exact-panel-attention.v1";
// Exact-P40000 whole-core composition. This identity requires the independent
// whole-core projection selector and complete-prompt FlashInfer selector;
// neither the panel Attention route nor the layer-wide-MLP-only route may
// emit it.
inline constexpr std::string_view
    kLayerMajorNativePromptWideP40WholeCoreDeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2.native-p40-whole-core.v1";
inline constexpr std::string_view
    kLayerMajorNativePromptWideP40ProjectionResetDeploymentPlanId =
        "q3x.sm87.ac-prefill-prompt-wide-v2."
        "native-p40-projection-reset.v1";
inline constexpr std::string_view
    kLayerMajorNativePromptWideP40PackedProjectionDeploymentPlanId =
        "q3x.sm87.ac-prefill-p40-packed-dataflow."
        "native-p40-packed-projection.v1";
inline constexpr std::string_view
    kLayerMajorNativePromptWideP40PackedNvfp4V2DeploymentPlanId =
        "q3x.sm87.ac-prefill-p40-packed-dataflow-v2."
        "native-p40-packed-nvfp4-shape-specific.v1";
inline constexpr std::string_view
    kLayerMajorNativePromptWideP40VllmMarlinParityDeploymentPlanId =
        "q3x.sm87.ac-prefill-p40-vllm-marlin-parity."
        "native-p40-canonical-nvfp4-legacy-stripe.v1";
// Independent complete-engine identity for the exact-P40000 target-AOT
// execution chain. It is never inferred from a projection/Attention tactic.
inline constexpr std::string_view kSm87TargetAotP40DeploymentPlanId =
    "q3x.sm87.target-aot.exact-p40000-one-token.v1";

[[nodiscard]] constexpr bool is_valid_reference_prefill_execution_mode(
    const ReferencePrefillExecutionMode mode) noexcept {
  return mode == ReferencePrefillExecutionMode::kLegacyC512Tiled ||
         mode == ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
}

struct ReferenceEngineOptions {
  ResidentLoadOptions resident_options;
  RequestMemoryOptions request_options;
  bool enable_trace = false;
  ProjectionBackend projection_backend = ProjectionBackend::kReference;
  // Engine-lifetime resource policy. The selected short-position cache is
  // prepared during engine creation and never lazily inside generate().
  ReferenceDecodeGraphCachePolicy decode_graph_cache_policy =
      ReferenceDecodeGraphCachePolicy::kDisabled;
  // Engine-lifetime request-arena provisioning. Whole-request execution is
  // still selected per generate() call, but it can only be requested from an
  // engine whose arena was explicitly created with the isolated layer-major
  // profile. Keeping this default legacy preserves every existing caller.
  ReferencePrefillExecutionMode prefill_execution_mode =
      ReferencePrefillExecutionMode::kLegacyC512Tiled;
  // Layer-major-only, engine-lifetime Attention tactic. The default preserves
  // the exact segmented incumbent. The native grouped-Q64 and Q128-v4 panel
  // values are explicit accuracy-unqualified architecture tactics and each
  // receives a distinct deployment ID.
  LayerMajorPrefillFullAttentionTactic prefill_full_attention_tactic =
      LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512;
  // Layer-major-only, engine-lifetime projection tactic. The default retains
  // the exact C512 arithmetic sequence. The segmented-wrapper value is an
  // accuracy-unqualified dependency screen with a distinct plan ID. The
  // native quantized large-M value is a separate accuracy-unqualified
  // architecture tactic whose receipts bind the exact Marlin sidecars,
  // reduction arenas, and locks. Only M8192 uses one physical bulk launch;
  // partial panels retain the authenticated exact span ledger.
  LayerMajorPrefillProjectionTactic prefill_projection_tactic =
      LayerMajorPrefillProjectionTactic::kExactSegmentedC512;
  // Default-off startup-only target-AOT NVFP4 asset preparation. This does
  // not select a runner tactic or open an executable route; it exists solely
  // to close real-checkpoint ownership and attachment before layer-0 oracle
  // admission. Unsupported binaries and incompatible engine modes fail
  // closed instead of ignoring a requested preparation.
  bool prepare_sm87_target_aot_projection_device_assets = false;
  // Default-off direct-load successor to the online preparation-only path.
  // A non-empty bundle path selects persisted loading and is mutually
  // exclusive with `prepare_sm87_target_aot_projection_device_assets`.
  // The expected catalog SHA-256 is caller-supplied test-admission data; a
  // digest stored only inside the bundle is never sufficient to authorize
  // its payloads. Production promotion must bind this value to an
  // authenticated DeploymentPlan/release attestation rather than a free
  // option.
  std::filesystem::path load_sm87_target_aot_projection_bundle;
  // Optional create-only offline output for the online preparation path.
  // This is a one-time deployment asset build, never a request-time cache.
  // The final file is atomically published without replacing an existing
  // path and must later be consumed through the direct-load option above.
  std::filesystem::path create_sm87_target_aot_projection_bundle;
  std::string
      expected_sm87_target_aot_projection_payload_catalog_sha256;
  // Append-only complete generation-route identity. The default preserves
  // the existing ReferenceEngine path. The target value is a host contract
  // only until the Engine explicitly binds its executor; callers must never
  // interpret a lower-level SM87 backend or Prefill tactic as this route.
  ReferenceGenerationRoute generation_route =
      ReferenceGenerationRoute::kReference;
};

// Text-only messages accepted by the pinned Qwen 3.6 chat formatter. The
// tokenizer deliberately fails closed on unsupported roles, non-alternating
// histories, tool payloads, and multimodal content.
struct ReferenceChatMessage {
  std::string role;
  std::string content;
};

// A token is reported only after its Prefill/Decode step has completed and
// the id has been committed to the generation transcript. String views are
// valid only for the duration of the callback. text_delta may be empty when a
// token contributes no visible text (including the configured stop token).
struct ReferenceTokenEvent {
  std::size_t index = 0U;
  std::uint32_t token_id = 0U;
  std::string_view text_delta;
  std::string_view generated_text;
  double elapsed_milliseconds = 0.0;
  bool is_stop_token = false;
};

// Returning false requests cancellation at the current committed-token
// boundary. The observer must not throw and is invoked synchronously on the
// engine's serialized generation thread.
using ReferenceTokenObserver = bool (*)(
    void* context, const ReferenceTokenEvent& event) noexcept;

enum class ReferencePrefillProgressLayerKind : std::uint8_t {
  kGdn = 0U,
  kFullAttention,
};

// Target-AOT progress is reported only after the owner-issued LayerComplete
// event has finished on the device. It is observational and carries no state
// publication or cancellation authority.
struct ReferencePrefillProgressEvent {
  ReferenceGenerationRoute generation_route =
      ReferenceGenerationRoute::kReference;
  std::uint64_t transaction_epoch = 0U;
  std::size_t completed_layer_index = kReferenceNoLayer;
  std::size_t completed_layers = 0U;
  std::size_t total_layers = 0U;
  ReferencePrefillProgressLayerKind layer_kind =
      ReferencePrefillProgressLayerKind::kGdn;
};

using ReferencePrefillProgressObserver = void (*)(
    void* context, const ReferencePrefillProgressEvent& event) noexcept;

struct ReferenceGenerateOptions {
  std::uint32_t max_new_tokens = 16U;
  bool capture_trace = false;
  std::uint32_t stop_token_id = kQwen36ImEndTokenId;
  std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
  // Full statistics is the compatibility default. Prediction-only compute
  // steps populate ReferenceStepResult::prediction instead of logits.
  ReferenceLogitsMode logits_mode = ReferenceLogitsMode::kFullStatistics;
  // Emit static NVTX ranges around the host-side prefill/decode runner
  // boundaries. Disabled by default so ordinary generation pays one
  // predictable dispatch branch.
  bool emit_nvtx_phase_ranges = false;
  // Test-only production-alignment hook. When enabled, predicted-only Decode
  // replays an already-prepared position-specialized graph and falls back to
  // the ordinary serial step on every incompatible option or cache miss.
  // Graph preparation is explicit and never occurs in a timed request.
  bool use_prepared_decode_graph_cache = false;
  // Optional synchronous per-token observer used by the streaming gateway.
  // Keep the callback and context in their historical aggregate-initializer
  // positions. New fields are append-only below them.
  ReferenceTokenObserver token_observer = nullptr;
  void* token_observer_context = nullptr;
  // The whole-request mode remains a default-off host integration surface.
  // It is accepted only by an engine created with the matching isolated
  // layer-major profile and sealed native operator plan.
  ReferencePrefillExecutionMode prefill_execution_mode =
      ReferencePrefillExecutionMode::kLegacyC512Tiled;
  // Whole-request-only cancellation probe. Supplying this callback also
  // enables the runner's two-quantum bounded submission window.
  ReferenceWholeRequestPrefillOptions::CancellationProbe
      prefill_cancellation_probe = nullptr;
  void* prefill_cancellation_context = nullptr;
  // Optional target-AOT-only host progress observer. Fields remain append-only
  // so the historical token-observer aggregate positions stay unchanged.
  ReferencePrefillProgressObserver prefill_progress_observer = nullptr;
  void* prefill_progress_context = nullptr;
};

enum class ReferenceStopReason : std::uint8_t {
  kImEnd,
  kMaxNewTokens,
  kCancelled,
};

struct ReferenceGenerationTiming {
  // One aggregate duration for each prefix_step or prefix_tile execution, or
  // exactly one duration for the opt-in whole-request callback. Under either
  // test-only all-prompt admission, the final prompt token is included here;
  // its logits-only finalize duration remains in
  // finish_prefill_milliseconds below.
  std::vector<double> prefix_execution_milliseconds;
  double finish_prefill_milliseconds = 0.0;
  double prompt_prefill_milliseconds = 0.0;
  double time_to_first_token_milliseconds = 0.0;
  std::vector<double> subsequent_token_milliseconds;
  double decode_after_first_milliseconds = 0.0;
  double total_generation_milliseconds = 0.0;
  // Whole-request-only final publication interval. Legacy routes retain the
  // exact zero default. prompt_prefill/TTFT/total include this value.
  double commit_prefill_milliseconds = 0.0;
  // False when prompt_prefill_milliseconds is only a diagnostic whole-engine
  // interval and cannot qualify or promote a pure-Prefill result.  Existing
  // routes retain their current true default and byte-stable witness schema.
  bool prompt_prefill_phase_qualified = true;
};

struct ReferenceTraceDigest {
  std::uint32_t position = 0U;
  std::uint32_t input_token_id = 0U;
  std::size_t element_count = 0U;
  std::string full_sha256;
  std::string embedding_sha256;
  std::array<std::string, kReferenceDecoderLayerCount> layer_hidden_sha256;
  std::array<std::string, kReferenceDecoderLayerCount> layer_residual_sha256;
  std::string final_norm_sha256;
};

struct ReferenceGeneration {
  std::string rendered_prompt;
  std::vector<std::uint32_t> prompt_token_ids;
  std::vector<std::uint32_t> generated_token_ids;
  std::string generated_text;
  ReferenceStopReason stop_reason = ReferenceStopReason::kMaxNewTokens;
  std::uint32_t requested_prefill_chunk_size =
      kDefaultRequestPrefillChunkSize;
  std::uint32_t effective_prefill_chunk_size =
      kDefaultRequestPrefillChunkSize;
  // True only when every prompt token was committed by prefill tiles and the
  // final prediction was produced from the retained last hidden row.
  bool all_prompt_tokens_prefilled_by_tiles = false;
  // True only for the test admission that dispatches each request-sized
  // prefix chunk as one arbitrary 1..512-token layer-major tile instead of
  // decomposing it into canonical C512/C256/C64/C32/tail executions.
  bool single_arbitrary_prefill_tiles = false;
  ReferenceGenerationTiming timing;
  // Completed-request Prefill route evidence. Configured/admitted selectors
  // are deliberately excluded; counters merge only after each Prefix
  // execution has synchronized and committed its state.
  PrefillRouteEvidence prefill_route_evidence;
  std::vector<ReferenceStepResult> steps;
  std::vector<ReferenceTraceDigest> traces;
  std::size_t decode_graph_replays = 0U;
  std::size_t decode_graph_serial_fallbacks = 0U;
  // Complete-engine route actually used by this committed result.  This is
  // execution evidence, not a copy of a requested selector.  The target
  // route publishes the exact executor receipt below and never synthesizes
  // legacy Prefix steps.
  ReferenceGenerationRoute generation_route =
      ReferenceGenerationRoute::kReference;
  RequestMemoryProfile request_memory_profile =
      RequestMemoryProfile::kLegacyC512;
  std::uint64_t consumed_prompt_tokens = 0U;
  bool full_prompt_consumed = false;
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
  ReferencePrefillExecutionMode prefill_execution_mode =
      ReferencePrefillExecutionMode::kLegacyC512Tiled;
  // Empty for legacy execution. Whole-request execution copies the identity
  // from the engine-lifetime BoundPrefillExecutionPlan; gateways must not
  // infer it from the public mode enum.
  std::string prefill_deployment_plan_id;
  // Number of complete logical 64-layer route records expected for this
  // request. In legacy mode this follows controller Prefix/final executions;
  // in whole-request mode it is the immutable topology's C8192 panel count,
  // and is intentionally independent of timing-vector cardinality.
  std::uint64_t prefill_logical_panel_count = 0U;
  bool prefill_bounded_submission_window = false;
  std::uint64_t prefill_submission_window_retirements = 0U;
  // Completed-launch witnesses from the bound layer-major executor. They are
  // published only after the whole Prefill transaction has synchronized and
  // committed; configured tactics alone never synthesize hits.
  std::uint64_t prefill_operator_panel_executor_hits = 0U;
  std::uint64_t prefill_native_group_q64_panel_hits = 0U;
  std::uint64_t prefill_native_group_q128_v4_panel_hits = 0U;
  std::uint64_t prefill_native_flashinfer_exact_panel_hits = 0U;
  std::uint64_t prefill_generic_qt2_hits = 0U;
  // Segmented-wrapper evidence. A hit is one completed logical FP8 or NVFP4
  // operator-panel wrapper invocation. Physical launches count the
  // shape-aware Marlin kernel segments submitted by those wrappers.
  std::uint64_t prefill_segmented_panel_projection_hits = 0U;
  std::uint64_t prefill_segmented_panel_projection_physical_launches = 0U;
  // Native quantized large-M evidence. A hit is one completed logical FP8 or
  // NVFP4 projection. Each projection is exclusively either one complete
  // M8192 bulk launch or the full partial-panel exact-oracle launch ledger;
  // hybrid bulk-plus-tail execution is not part of this plan identity.
  std::uint64_t prefill_native_large_m_projection_hits = 0U;
  std::uint64_t prefill_native_large_m_projection_bulk_hits = 0U;
  std::uint64_t prefill_native_large_m_projection_oracle_partial_hits = 0U;
  std::uint64_t prefill_native_large_m_projection_physical_launches = 0U;
  // WP-V2-C1 evidence is independent from the older native-large-M Marlin
  // route. Gate+Up and Down must both complete for a coupled panel package.
  std::uint64_t prefill_nvfp4_true_large_m_route_fp8_projection_hits = 0U;
  std::uint64_t prefill_nvfp4_true_large_m_route_fp8_projection_bulk_hits = 0U;
  std::uint64_t
      prefill_nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits = 0U;
  std::uint64_t
      prefill_nvfp4_true_large_m_route_fp8_projection_physical_launches = 0U;
  std::uint64_t prefill_native_nvfp4_true_large_m_projection_hits = 0U;
  std::uint64_t prefill_native_nvfp4_true_large_m_gate_up_hits = 0U;
  std::uint64_t prefill_native_nvfp4_true_large_m_down_hits = 0U;
  std::uint64_t prefill_native_nvfp4_true_large_m_physical_launches = 0U;
  LayerMajorPrefillMlpScheduleTactic prefill_mlp_schedule_tactic =
      LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel;
  std::uint64_t prefill_route_layer_pass_count = 0U;
  // Exact-P40000 layer-wide MLP witnesses. The GateUp/Down counts are one per
  // decoder layer, never multiplied by the five Attention/GDN panels.
  std::uint64_t prefill_layer_wide_p40_mlp_layer_hits = 0U;
  std::uint64_t prefill_persistent_p40_nvfp4_gate_up_hits = 0U;
  std::uint64_t prefill_persistent_p40_nvfp4_down_residual_hits = 0U;
  std::uint64_t prefill_persistent_p40_nvfp4_physical_launches = 0U;
  std::uint64_t prefill_persistent_p40_fp8_projection_hits = 0U;
  std::uint64_t prefill_persistent_p40_fp8_projection_bulk_hits = 0U;
  std::uint64_t
      prefill_persistent_p40_fp8_projection_oracle_partial_hits = 0U;
  std::uint64_t prefill_persistent_p40_fp8_projection_physical_launches = 0U;
  // Exact-P40000 whole-core completed-launch witnesses. These are copied
  // from the committed runner result; selector admission never creates hits.
  std::uint64_t prefill_prompt_wide_p40_whole_core_layer_hits = 0U;
  std::uint64_t prefill_prompt_wide_p40_fill_panel_hits = 0U;
  std::uint64_t prefill_prompt_wide_p40_prompt_core_hits = 0U;
  std::uint64_t prefill_prompt_wide_p40_drain_panel_hits = 0U;
  std::uint64_t prefill_prompt_wide_p40_fp8_projection_hits = 0U;
  std::uint64_t
      prefill_prompt_wide_p40_fp8_projection_physical_launches = 0U;
  std::uint64_t prefill_prompt_wide_p40_bf16_ab_hits = 0U;
  std::uint64_t prefill_prompt_wide_p40_gdn_hits = 0U;
  std::uint64_t prefill_native_flashinfer_exact_whole_prompt_hits = 0U;
  // Packed-NVFP4-v2 executor evidence is additive to the common P40 packed
  // FP8 and whole-core counters above. Selector admission never creates hits.
  std::uint64_t prefill_packed_nvfp4_v2_gate_up_hits = 0U;
  std::uint64_t prefill_packed_nvfp4_v2_down_hits = 0U;
  std::uint64_t prefill_packed_nvfp4_v2_physical_launches = 0U;
  // Independent stock-vLLM Marlin host-dispatch reference. These counters
  // and receipts never alias the historical persistent/packed ledgers.
  std::uint64_t prefill_vllm_marlin_parity_gate_up_hits = 0U;
  std::uint64_t prefill_vllm_marlin_parity_down_hits = 0U;
  std::uint64_t prefill_vllm_marlin_parity_physical_launches = 0U;
  std::uint64_t prefill_vllm_marlin_parity_standalone_silu_launches = 0U;
  std::uint64_t
      prefill_vllm_marlin_parity_standalone_residual_launches = 0U;
  std::uint64_t prefill_vllm_marlin_parity_lock_clear_operations = 0U;
  std::array<ReferenceP40VllmMarlinParityLayerCompletionReceipt,
             kReferenceDecoderLayerCount>
      prefill_vllm_marlin_parity_layer_completion_receipts{};
  std::size_t
      prefill_vllm_marlin_parity_layer_completion_receipt_count = 0U;
};

struct ReferenceEngineLoadStats {
  // Engine-wide dispatcher policy accepted by the factory. This is a
  // configured route fact, not per-operator launch attestation.
  ProjectionBackend projection_backend = ProjectionBackend::kReference;
  ReferenceGenerationRoute generation_route =
      ReferenceGenerationRoute::kReference;
  double tokenizer_milliseconds = 0.0;
  double resident_load_milliseconds = 0.0;
  double weight_bind_milliseconds = 0.0;
  double request_state_milliseconds = 0.0;
  double fp8_output_sidecar_milliseconds = 0.0;
  double nvfp4_down_scale6_sidecar_milliseconds = 0.0;
  double nvfp4_down_consumer_order_sidecar_milliseconds = 0.0;
  double nvfp4_gate_up_coupled_feed_milliseconds = 0.0;
  double fp8_prefill_qkv_sidecar_milliseconds = 0.0;
  double fp8_prefill_supermatrix_sidecar_milliseconds = 0.0;
  double fp8_marlin_prefill_sidecar_milliseconds = 0.0;
  double nvfp4_marlin_prefill_sidecar_milliseconds = 0.0;
  double p40_packed_projection_asset_milliseconds = 0.0;
  double nvfp4_marlin_p40_parity_sidecar_milliseconds = 0.0;
  double target_aot_projection_device_asset_milliseconds = 0.0;
  double target_aot_complete_projection_device_asset_milliseconds = 0.0;
  double target_aot_engine_rope_milliseconds = 0.0;
  double target_aot_request_owner_milliseconds = 0.0;
  double target_aot_executor_bind_milliseconds = 0.0;
  double runner_factory_milliseconds = 0.0;
  ReferenceDecodeGraphCachePolicy decode_graph_cache_requested_policy =
      ReferenceDecodeGraphCachePolicy::kDisabled;
  ReferenceDecodeGraphCachePolicy decode_graph_cache_effective_policy =
      ReferenceDecodeGraphCachePolicy::kDisabled;
  std::uint32_t decode_graph_cache_first_position = 0U;
  std::uint32_t decode_graph_cache_last_position = 0U;
  std::size_t decode_graph_cache_slot_count = 0U;
  double decode_graph_cache_capture_enqueue_milliseconds = 0.0;
  double decode_graph_cache_topology_inspection_milliseconds = 0.0;
  double decode_graph_cache_instantiate_milliseconds = 0.0;
  double decode_graph_cache_upload_ready_milliseconds = 0.0;
  double decode_graph_cache_prepare_milliseconds = 0.0;
  std::uint64_t decode_graph_cache_free_bytes_before = 0U;
  std::uint64_t decode_graph_cache_free_bytes_after = 0U;
  std::uint64_t decode_graph_cache_free_drop_bytes = 0U;
  // Empty when the requested cache was completely prepared or when the
  // policy was disabled. Any preparation/admission failure is rolled back
  // before this stable serial-fallback reason is published.
  std::string decode_graph_cache_fallback_reason;
  double total_milliseconds = 0.0;
  ResidentLoadStats resident;
  WeightBindingStats binding;
  std::uint64_t request_arena_bytes = 0U;
  std::uint32_t request_max_sequence_length = 0U;
  std::uint32_t request_prefill_chunk_size =
      kDefaultRequestPrefillChunkSize;
  RequestMemoryProfile request_memory_profile =
      RequestMemoryProfile::kLegacyC512;
  bool fp8_output_sidecars_enabled = false;
  std::size_t fp8_output_sidecar_layers = 0U;
  std::uint64_t fp8_output_sidecar_bytes = 0U;
  // Empty when the SM87 sidecar path was enabled or was not requested.
  // Optional allocation/memory-gate failures retain the canonical M1 route
  // and record a stable reason here instead of failing engine creation.
  std::string fp8_output_sidecar_fallback_reason;
  // The down-only scale6 inventory is derived from all 64 canonical NVFP4
  // block-scale tensors. Eligible layers receive compact sidecars; fallback
  // layers retain their canonical scales. bytes is nonzero only when the
  // complete eligible inventory was attached successfully.
  bool nvfp4_down_scale6_sidecars_enabled = false;
  std::size_t nvfp4_down_scale6_sidecar_eligible_layers = 0U;
  std::size_t nvfp4_down_scale6_sidecar_fallback_layers = 0U;
  std::uint64_t nvfp4_down_scale6_sidecar_bytes = 0U;
  // Empty when the optional SM87 sidecar inventory was attached or was not
  // requested. Admission/allocation failures preserve canonical execution.
  std::string nvfp4_down_scale6_sidecar_fallback_reason;
  // Explicit same-ELF Decode Down K512 consumer-order admission. The
  // equal-byte weight arena is attached only to scale6-eligible layers and
  // is prepared after all production/Prefill sidecars so it cannot displace
  // them. A requested admission is all-or-nothing and fails engine creation
  // instead of silently benchmarking the production kernel.
  bool nvfp4_down_consumer_order_sidecars_requested = false;
  bool nvfp4_down_consumer_order_sidecars_enabled = false;
  std::size_t nvfp4_down_consumer_order_sidecar_layers = 0U;
  std::uint64_t nvfp4_down_consumer_order_sidecar_bytes = 0U;

  // Explicit Decode admission only.  The arena is an equal-byte Gate+Up
  // weight+scale permutation and therefore coexists with canonical tensors.
  bool nvfp4_gate_up_coupled_feed_requested = false;
  bool nvfp4_gate_up_coupled_feed_enabled = false;
  std::size_t nvfp4_gate_up_coupled_feed_layers = 0U;
  std::uint64_t nvfp4_gate_up_coupled_feed_bytes = 0U;
  // The exact C512 register-feed layout is optional and intentionally
  // co-resident with the canonical 48 linear-attention QKV tensors. A
  // capacity miss preserves the canonical Prefill route; all non-capacity
  // preparation failures remain engine-creation failures.
  bool fp8_prefill_qkv_sidecars_enabled = false;
  std::size_t fp8_prefill_qkv_sidecar_layers = 0U;
  std::uint64_t fp8_prefill_qkv_sidecar_bytes = 0U;
  // Empty when the complete 48-layer inventory was attached or when the
  // exact SM87 C512 sidecar route was not requested.
  std::string fp8_prefill_qkv_sidecar_fallback_reason;
  // The fixed 208-projection exact-C512 inventory replaces the legacy
  // QKV-only sidecars when its complete arena is admitted.
  bool fp8_prefill_supermatrix_sidecars_enabled = false;
  std::size_t fp8_prefill_supermatrix_sidecar_projections = 0U;
  std::uint64_t fp8_prefill_supermatrix_sidecar_bytes = 0U;
  // Populated only by the direct W8A16 test-admission build. The retained
  // compact weight and BF16 channel-scale arenas replace the equal-byte
  // production supermatrix arena; disposable transpose scratch is excluded.
  bool fp8_marlin_prefill_sidecars_enabled = false;
  std::size_t fp8_marlin_prefill_sidecar_projections = 0U;
  std::uint64_t fp8_marlin_prefill_sidecar_bytes = 0U;
  // Populated only by the dedicated test-admission build. This is the sum of
  // all retained Gate+Up and Down Marlin weights, processed scales, and
  // global scales; the disposable load-time transpose scratch is excluded.
  bool nvfp4_marlin_prefill_sidecars_enabled = false;
  std::size_t nvfp4_marlin_prefill_sidecar_layers = 0U;
  std::uint64_t nvfp4_marlin_prefill_sidecar_bytes = 0U;
  // Exact-P40000 AOT packed projection inventory. One engine-lifetime arena
  // owns 256 authenticated physical artifacts sourced from all 400 logical
  // FP8/NVFP4 checkpoint tensors; no request may repack or select tactics.
  bool p40_packed_projection_assets_enabled = false;
  std::size_t p40_packed_projection_artifacts = 0U;
  std::size_t p40_packed_projection_sources = 0U;
  std::size_t p40_packed_projection_fp8_logical_roles = 0U;
  std::size_t p40_packed_projection_fp8_physical_launches = 0U;
  std::size_t p40_packed_projection_nvfp4_physical_launches = 0U;
  std::uint64_t p40_packed_projection_asset_bytes = 0U;
  // Independent P40000 vLLM-Marlin reference owner. Its 128 artifacts are
  // derived from 192 authenticated canonical checkpoint sources and never
  // populate either prefill_marlin_* or packed projection views.
  bool nvfp4_marlin_p40_parity_sidecars_enabled = false;
  std::size_t nvfp4_marlin_p40_parity_layers = 0U;
  std::size_t nvfp4_marlin_p40_parity_artifacts = 0U;
  std::size_t nvfp4_marlin_p40_parity_sources = 0U;
  std::uint64_t nvfp4_marlin_p40_parity_sidecar_bytes = 0U;
  std::string nvfp4_marlin_p40_parity_manifest_sha256;
  // Startup-only target-AOT NVFP4 ownership evidence. `enabled` means all
  // 128 artifacts were uploaded/read back and owner-backed attachment to
  // ModelWeights completed; it does not mean a launcher or route exists.
  bool target_aot_projection_device_assets_requested = false;
  bool target_aot_projection_device_assets_enabled = false;
  bool target_aot_projection_device_assets_attached = false;
  bool target_aot_projection_device_assets_loaded_from_persisted_bundle =
      false;
  std::size_t target_aot_projection_device_asset_artifacts = 0U;
  std::size_t target_aot_projection_device_asset_sources = 0U;
  std::uint64_t target_aot_projection_device_asset_bytes = 0U;
  std::uint64_t target_aot_projection_host_staging_peak_bytes = 0U;
  std::uint64_t target_aot_projection_source_d2h_bytes = 0U;
  std::uint64_t target_aot_projection_payload_h2d_bytes = 0U;
  std::uint64_t target_aot_projection_verification_d2h_bytes = 0U;
  std::uint64_t target_aot_projection_persistent_bundle_file_bytes_read =
      0U;
  std::uint32_t
      target_aot_projection_persistent_bundle_host_authentication_passes = 0U;
  bool target_aot_projection_persistent_bundle_created = false;
  std::uint64_t target_aot_projection_persistent_bundle_file_bytes_written =
      0U;
  std::string
      target_aot_projection_persistent_record_header_catalog_sha256;
  // Stable identity of the complete ordered manifest/source/device-readback
  // digest catalog. Runtime owner/allocation identities remain lifetime
  // witnesses only.
  std::string target_aot_projection_verified_payload_catalog_sha256;
  std::uint64_t target_aot_projection_owner_identity = 0U;
  std::uint64_t target_aot_projection_allocation_identity = 0U;
  std::int32_t target_aot_projection_device_ordinal = -1;
  bool target_aot_complete_projection_device_assets_enabled = false;
  bool target_aot_complete_projection_device_assets_attached = false;
  std::size_t target_aot_complete_projection_artifacts = 0U;
  std::size_t target_aot_complete_projection_sources = 0U;
  std::uint64_t target_aot_complete_projection_bytes = 0U;
  std::uint64_t target_aot_complete_projection_source_d2h_bytes = 0U;
  std::uint64_t target_aot_complete_projection_payload_h2d_bytes = 0U;
  std::uint64_t target_aot_complete_projection_verification_d2h_bytes = 0U;
  std::string target_aot_complete_projection_catalog_sha256;
  std::uint64_t target_aot_complete_projection_owner_identity = 0U;
  std::uint64_t target_aot_complete_projection_allocation_identity = 0U;
  std::int32_t target_aot_complete_projection_device_ordinal = -1;
  bool target_aot_engine_rope_ready = false;
  std::uint64_t target_aot_engine_rope_bytes = 0U;
  bool target_aot_request_owner_ready = false;
  bool target_aot_executor_ready = false;
  // True only when tokenizer parsing and resident loading actually executed
  // concurrently. When true, total_milliseconds is wall time and phase
  // timings intentionally overlap.
  bool tokenizer_resident_overlap = false;
};

struct ReferenceGenerateResult {
  std::optional<ReferenceGeneration> value;
  ReferenceEngineDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.code == ReferenceEngineError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

inline constexpr std::size_t kReferenceDecodeGraphP1ScreenRounds = 5U;

struct ReferenceDecodeGraphP1ScreenOptions {
  std::uint32_t expected_position = 19U;
  std::uint32_t expected_input_token_id = 77'517U;
  std::uint32_t expected_prediction = 220U;
  std::uint32_t alternate_input_token_id = 220U;
  std::uint32_t expected_alternate_prediction = 52'965U;
  std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
};

struct ReferenceDecodeGraphP1RoundTiming {
  double serial_first_milliseconds = 0.0;
  double graph_first_milliseconds = 0.0;
  double graph_second_milliseconds = 0.0;
  double serial_second_milliseconds = 0.0;
};

struct ReferenceDecodeGraphP1ScreenResult {
  ReferenceDecodeGraphP1Stats graph;
  std::uint32_t serial_prediction = 0U;
  std::uint32_t graph_prediction = 0U;
  std::uint32_t alternate_serial_prediction = 0U;
  std::uint32_t alternate_graph_prediction = 0U;
  bool primary_arena_exact = false;
  bool alternate_arena_exact = false;
  std::uint64_t compared_arena_bytes = 0U;
  std::array<ReferenceDecodeGraphP1RoundTiming,
             kReferenceDecodeGraphP1ScreenRounds>
      rounds{};
};

struct ReferenceDecodeGraphP1ScreenOutcome {
  std::optional<ReferenceDecodeGraphP1ScreenResult> value;
  ReferenceEngineDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.code == ReferenceEngineError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

inline constexpr std::size_t kReferenceDecodeGraphP2ScreenRounds = 5U;
inline constexpr std::size_t kReferenceDecodeGraphP2ContinuousSteps = 25U;
// Decode Graph admission remains pinned to the frozen P19/C32 baseline even
// when the independent request Prefill capacity grows.
inline constexpr std::uint32_t kReferenceDecodeGraphScreenPrefillChunkSize =
    32U;

struct ReferenceDecodeGraphP2ScreenOptions {
  std::uint32_t first_decode_position = 19U;
  std::uint32_t last_decode_position = 43U;
  std::uint32_t boundary_graph_position = 63U;
  std::uint32_t capture_input_token_id = 0U;
  std::uint32_t boundary_input_token_id = 9'419U;
  std::uint32_t max_new_tokens = 26U;
  std::uint32_t prefill_chunk_size =
      kReferenceDecodeGraphScreenPrefillChunkSize;
};

struct ReferenceDecodeGraphP2RoundTiming {
  double serial_first_milliseconds_per_token = 0.0;
  double graph_first_milliseconds_per_token = 0.0;
  double graph_second_milliseconds_per_token = 0.0;
  double serial_second_milliseconds_per_token = 0.0;
};

struct ReferenceDecodeGraphP2ScreenResult {
  std::array<ReferenceDecodeGraphP1Stats,
             kReferenceDecodeGraphP2ContinuousSteps>
      continuous_graphs{};
  ReferenceDecodeGraphP1Stats boundary_graph;
  double cache_prepare_milliseconds = 0.0;
  std::uint64_t cache_free_bytes_before = 0U;
  std::uint64_t cache_free_bytes_after = 0U;
  std::uint64_t cache_cuda_free_drop_bytes = 0U;
  std::uint64_t boundary_free_bytes_after = 0U;
  std::uint64_t boundary_cuda_free_drop_bytes = 0U;
  std::uint64_t cache_plus_boundary_cuda_free_drop_bytes = 0U;
  std::uint64_t cache_host_private_bytes_before = 0U;
  std::uint64_t cache_host_private_bytes_after = 0U;
  std::uint64_t cache_host_private_increase_bytes = 0U;
  bool cache_host_private_observed = false;
  bool cache_prepare_arena_exact = false;
  bool boundary_prepare_arena_exact = false;
  ReferenceGeneration serial_generation;
  ReferenceGeneration graph_generation;
  bool continuous_generation_exact = false;
  bool continuous_arena_exact = false;
  bool cache_miss_fallback_exact = false;
  bool full_statistics_fallback_exact = false;
  bool trace_fallback_exact = false;
  bool boundary_cache_hit = false;
  bool boundary_cache_miss_fallback = false;
  std::uint32_t boundary_serial_first_prediction = 0U;
  std::uint32_t boundary_serial_second_prediction = 0U;
  std::uint32_t boundary_graph_first_prediction = 0U;
  std::uint32_t boundary_fallback_second_prediction = 0U;
  bool boundary_arena_exact = false;
  std::uint64_t compared_arena_bytes = 0U;
  std::array<ReferenceDecodeGraphP2RoundTiming,
             kReferenceDecodeGraphP2ScreenRounds>
      rounds{};
};

struct ReferenceDecodeGraphP2ScreenOutcome {
  std::optional<ReferenceDecodeGraphP2ScreenResult> value;
  ReferenceEngineDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.code == ReferenceEngineError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct ReferenceEngineCreateResult;
struct ReferenceOneShotOptions;
struct ReferenceOneShotResult;

// High-level owner for the exact lifetime chain:
// ResidentWeights -> ModelWeights -> RequestState -> ReferenceRunner.
// The implementation is heap-stable so moving ReferenceEngine never changes
// the addresses retained by the non-owning runner.
class ReferenceEngine {
 public:
  ~ReferenceEngine();
  ReferenceEngine(const ReferenceEngine&) = delete;
  ReferenceEngine& operator=(const ReferenceEngine&) = delete;
  ReferenceEngine(ReferenceEngine&&) noexcept;
  ReferenceEngine& operator=(ReferenceEngine&&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] const ReferenceEngineLoadStats& load_stats() const noexcept;
  [[nodiscard]] std::uint32_t max_sequence_length() const noexcept;

  // Formats exactly one user message with thinking=false, encodes it with the
  // pinned tokenizer, resets request state, then performs sequential batch-one
  // prefill and greedy decode.
  [[nodiscard]] ReferenceGenerateResult generate(
      std::string_view user_prompt,
      const ReferenceGenerateOptions& options = {});

  // Formats a supported text-only chat history with the pinned template,
  // resets request state, then performs the same batch-one greedy path as
  // generate(). The final message must be a user message.
  [[nodiscard]] ReferenceGenerateResult generate_chat(
      const std::vector<ReferenceChatMessage>& messages,
      const ReferenceGenerateOptions& options = {});

  // Raw completion surfaces. generate_prompt() encodes bytes without a chat
  // template; generate_prompt_token_ids() executes caller-supplied pinned
  // vocabulary ids exactly. They are used by /v1/completions and must never
  // be emulated through generate_chat().
  [[nodiscard]] ReferenceGenerateResult generate_prompt(
      std::string_view prompt,
      const ReferenceGenerateOptions& options = {});
  [[nodiscard]] ReferenceGenerateResult generate_prompt_token_ids(
      const std::vector<std::uint32_t>& prompt_token_ids,
      const ReferenceGenerateOptions& options = {});

  // Test-only, fixed-position CUDA Graph P1 screen. It first performs an exact
  // one-token predicted-only generation so the owned runner is left at the
  // prompt boundary, then snapshots the complete request arena. Serial and
  // graph steps are restored from that snapshot before every correctness or
  // timing sample. The graph remains a single-position experiment; no
  // production graph cache or generation dispatch is installed.
  [[nodiscard]] ReferenceDecodeGraphP1ScreenOutcome
  screen_fixed_position_decode_graph_p1(
      std::string_view user_prompt,
      const ReferenceDecodeGraphP1ScreenOptions& options = {});

  // Test-only P2 screen for a pre-uploaded short-position GraphExec cache.
  // It validates continuous generation and serial fallbacks from identical
  // complete arena snapshots before measuring hot subsequent-token latency.
  [[nodiscard]] ReferenceDecodeGraphP2ScreenOutcome
  screen_short_decode_graph_cache_p2(
      std::string_view user_prompt,
      const ReferenceDecodeGraphP2ScreenOptions& options = {});

 private:
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  // The test-only implementation and all of its evidence types live in a
  // source-private header. This friendship exposes no installed method,
  // result type, device view, or production selector.
  friend class reference_engine_test_detail::
      Sm87TargetAotLayer0M192OracleAccess;
#endif
  friend struct ReferenceEngineCreateResult;
  friend ReferenceEngineCreateResult create_reference_engine(
      const std::filesystem::path&, const ReferenceEngineOptions&);
  friend ReferenceOneShotResult generate_reference(
      const std::filesystem::path&, std::string_view,
      const ReferenceOneShotOptions&);
  [[nodiscard]] static Sm87TargetAotProjectionDevicePreparationStats
  prepare_target_aot_projection_device_assets(
      const ResidentWeights& resident, const ModelWeights& model_weights,
      const std::filesystem::path& create_bundle_path,
      std::string_view expected_verified_payload_catalog_sha256,
      std::uint64_t minimum_free_bytes_after_prepare,
      Sm87TargetAotProjectionDeviceAssets& owner);
  [[nodiscard]] static Sm87TargetAotProjectionDevicePreparationStats
  load_target_aot_projection_device_assets(
      const ResidentWeights& resident, const ModelWeights& model_weights,
      const std::filesystem::path& bundle_path,
      std::string_view expected_verified_payload_catalog_sha256,
      std::uint64_t minimum_free_bytes_after_load,
      Sm87TargetAotProjectionDeviceAssets& owner);
  [[nodiscard]] static bool attach_target_aot_projection_device_assets(
      ModelWeights& model_weights,
      Sm87TargetAotProjectionDeviceAssets& owner) noexcept;
  [[nodiscard]] static Sm87TargetAotCompleteDevicePreparationStats
  prepare_target_aot_complete_projection_device_assets(
      const ResidentWeights& resident, const ModelWeights& model_weights,
      std::uint64_t minimum_free_bytes_after_prepare,
      Sm87TargetAotCompleteProjectionDeviceAssets& owner);
  [[nodiscard]] static bool
  attach_target_aot_complete_projection_device_assets(
      ModelWeights& model_weights,
      Sm87TargetAotCompleteProjectionDeviceAssets& owner) noexcept;
  struct Impl;
  explicit ReferenceEngine(std::unique_ptr<Impl> impl) noexcept;
  [[nodiscard]] ReferenceGenerateResult generate_tokenized(
      std::string rendered_prompt,
      std::vector<std::uint32_t> prompt_token_ids,
      const ReferenceGenerateOptions& options);

  std::unique_ptr<Impl> impl_;
};

struct ReferenceEngineCreateResult {
  std::optional<ReferenceEngine> value;
  ReferenceEngineDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.code == ReferenceEngineError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

[[nodiscard]] ReferenceEngineCreateResult create_reference_engine(
    const std::filesystem::path& model_directory,
    const ReferenceEngineOptions& options = {});

struct ReferenceOneShotOptions {
  ResidentLoadOptions resident_options;
  std::uint64_t request_max_arena_bytes =
      2ULL * 1024ULL * 1024ULL * 1024ULL;
  std::uint64_t request_min_free_bytes_after_create =
      8ULL * 1024ULL * 1024ULL * 1024ULL;
  ProjectionBackend projection_backend = ProjectionBackend::kReference;
  ReferenceGenerateOptions generation;
  // Tokenizer parsing is CPU-only and independent of checkpoint
  // authentication/copy, so the production one-shot path overlaps them by
  // default. Disable this only for diagnostics or controlled benchmarks.
  bool overlap_tokenizer_and_resident_load = true;
  // Keep new one-shot policy fields at the end so positional aggregate
  // initialization of the pre-existing surface remains source-compatible.
  ReferenceDecodeGraphCachePolicy decode_graph_cache_policy =
      ReferenceDecodeGraphCachePolicy::kDisabled;
};

struct ReferenceOneShotGeneration {
  ReferenceEngineLoadStats load;
  ReferenceGeneration generation;
};

struct ReferenceOneShotResult {
  std::optional<ReferenceOneShotGeneration> value;
  ReferenceEngineDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.code == ReferenceEngineError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// One-shot production path used by the CLI. It loads and validates the pinned
// tokenizer while the resident checkpoint loads, derives the smallest request
// capacity that can execute the prompt plus max_new_tokens, then creates the
// four-stage native lifetime chain without loading either asset a second time.
[[nodiscard]] ReferenceOneShotResult generate_reference(
    const std::filesystem::path& model_directory,
    std::string_view user_prompt,
    const ReferenceOneShotOptions& options = {});

[[nodiscard]] std::string_view to_string(ReferenceEngineError error) noexcept;
[[nodiscard]] std::string_view to_string(ReferenceStopReason reason) noexcept;

namespace reference_engine_detail {

inline constexpr std::size_t kOptimizedPrefillSubtileTokens = 32U;
inline constexpr std::size_t kPrefillM64TileTokens = 64U;
inline constexpr std::size_t kPrefillM256TileTokens = 256U;
static_assert(kMaximumRequestPrefillChunkSize == 512U);

// The public C512 request boundary is independent from individual kernel
// limits. Scheduler calls use only exact C512/C256/C64/C32 tiles plus an
// ordered <=31 tail. Non-canonical request caps such as C128, C192, and C320
// therefore decompose into the same explicit production sizes.
[[nodiscard]] constexpr std::size_t next_prefix_tile_token_count(
    const std::size_t remaining_tokens,
    const std::size_t requested_chunk_size) noexcept {
  if (remaining_tokens == 0U || requested_chunk_size == 0U) {
    return 0U;
  }
  const std::size_t candidate = remaining_tokens < requested_chunk_size
                                    ? remaining_tokens
                                    : requested_chunk_size;
  return next_prefill_physical_segment_token_count(candidate);
}

[[nodiscard]] constexpr std::size_t prefix_execution_count(
    const std::size_t prefix_token_count,
    const std::size_t effective_prefill_chunk_size) noexcept {
  if (effective_prefill_chunk_size == 0U) {
    return 0U;
  }
  if (effective_prefill_chunk_size == 1U) {
    return prefix_token_count;
  }
  std::size_t count = 0U;
  std::size_t remaining = prefix_token_count;
  while (remaining != 0U) {
    remaining -= next_prefix_tile_token_count(
        remaining, effective_prefill_chunk_size);
    ++count;
  }
  return count;
}

// Test-only scheduler used to measure whether eliminating repeated
// layer/weight traversal is the missing system-level prefill architecture.
// Individual subkernels remain free to use their established ordered
// fallbacks inside this one public layer-major tile.
[[nodiscard]] constexpr std::size_t
next_single_arbitrary_prefix_tile_token_count(
    const std::size_t remaining_tokens,
    const std::size_t requested_chunk_size) noexcept {
  if (remaining_tokens == 0U || requested_chunk_size == 0U) {
    return 0U;
  }
  return remaining_tokens < requested_chunk_size ? remaining_tokens
                                                 : requested_chunk_size;
}

[[nodiscard]] constexpr std::size_t
single_arbitrary_prefix_execution_count(
    const std::size_t prefix_token_count,
    const std::size_t effective_prefill_chunk_size) noexcept {
  if (effective_prefill_chunk_size == 0U) {
    return 0U;
  }
  return prefix_token_count / effective_prefill_chunk_size +
         (prefix_token_count % effective_prefill_chunk_size != 0U ? 1U : 0U);
}

enum class GenerationControlError : std::uint8_t {
  kNone = 0,
  kInvalidArgument,
  kCapacityExceeded,
  kArithmeticOverflow,
  kRunnerFailure,
  kUnexpectedStep,
  kMissingLogits,
  kMissingTiming,
  kAllocationFailure,
  kMissingPrediction,
};

using StepFunction = ReferenceStepOutcome (*)(
    void* context, std::uint32_t input_token_id,
    const ReferenceStepOptions& options);
using PrefillTileFunction = ReferencePrefillTileOutcome (*)(
    void* context, const std::uint32_t* input_token_ids,
    std::size_t token_count, const ReferencePrefillTileOptions& options);

// Host-only transcript returned for one logical C8192 panel. The panel
// metadata is deliberately repeated from the immutable unbound topology so
// the generation controller can fail closed on omissions, reordering, or
// discontinuities before accepting the transcript. It does not attest bound
// operators or make PrefillExecutionPlan executable.
struct PrefillPromptPanelResult {
  std::size_t logical_panel_ordinal = 0U;
  std::size_t prompt_token_offset = 0U;
  std::uint32_t first_position = 0U;
  std::uint32_t end_position = 0U;
  std::vector<ReferenceStepResult> steps;
};

// A whole-request callback reports exactly one aggregate timing. Individual
// panel and placeholder steps carry no timing or logits; the existing
// retained-hidden finalizer replaces the last placeholder afterward.
struct PrefillPromptResult {
  std::vector<PrefillPromptPanelResult> panels;
  std::size_t logical_panel_count = 0U;
  std::size_t prompt_token_count = 0U;
  // Host-side proof that every logical panel completed in every layer, the
  // correct KV/GDN domain reached the prompt end, and final hidden became
  // ready. The callback must leave prefill_state_committed false; the
  // controller owns the later single commit. This does not substitute for
  // future bound stream/event evidence.
  PrefillExecutionProgress progress;
  std::optional<ReferenceStepTiming> timing;
};

struct PrefillPromptOutcome {
  std::optional<PrefillPromptResult> value;
  ReferenceRunnerStatus status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct PrefillPromptOptions {
  bool measure_timing = false;
  bool retain_last_hidden_for_logits = false;
};

// input_token_ids and unbound_immutable_topology are synchronous-call
// references: the callback must not retain either address after returning.
// Success means every reported device operation is complete and visible, not
// merely enqueued. The callback must not throw; the controller nevertheless
// maps a contract-violating exception to a runner failure.
using PrefillPromptFunction = PrefillPromptOutcome (*)(
    void* context, const std::uint32_t* input_token_ids,
    std::size_t token_count,
    const PrefillExecutionPlan& unbound_immutable_topology,
    const PrefillPromptOptions& options);

// Atomically publishes the completed, uncommitted whole-request state. The
// topology/progress references are valid only during the call. This callback
// is no-throw. It must validate every fallible precondition before one no-fail
// initial->final publication; a non-OK status guarantees that state did not
// change. The controller supplies the immutable transition and owns the sole
// invocation, so the callback cannot invent a receipt or commit count.
using PrefillCommitFunction = ReferenceRunnerStatus (*)(
    void* context, const PrefillExecutionPlan& unbound_immutable_topology,
    const PrefillExecutionProgress& completed_uncommitted_progress) noexcept;
using CommittedTokenFunction = bool (*)(
    void* context, std::uint32_t token_id, std::size_t token_index,
    double elapsed_milliseconds) noexcept;

// Host-side execution seams for the two generation phases. The first
// implementation may route every callback to the same runner and stream; the
// separate plans make the final-prompt boundary explicit without changing the
// runner, request-state, workspace, or public engine contracts. Both plans
// must provide every scalar callback even when a particular request would not
// exercise it; prefix_tile is required only for a legacy chunked non-empty
// prefix, while whole_request is default-null and separately opted in.
struct PrefillPlan {
  void* context = nullptr;
  StepFunction prefix_step = nullptr;
  StepFunction finish_prefill = nullptr;
  PrefillTileFunction prefix_tile = nullptr;
  // Required only by the legacy test-only whole-prompt tile admission. This
  // callback finalizes logits from the last prompt step already committed by
  // a marked prefix tile; it must not append another model-state step.
  StepFunction finish_prefill_from_tile = nullptr;
  // Default-null host seam for AC-PREFILL-LAYERMAJOR-8K-v1. The callback is
  // reachable only through the explicit whole-request control option. Its
  // PrefillExecutionPlan argument is immutable unbound topology
  // (executable()==false), not a production operator binding or route.
  PrefillPromptFunction whole_request = nullptr;
  // Whole-request-only finalizer for retained hidden whose state is complete
  // but still uncommitted. It must produce logits without changing logical
  // state; commit_whole_request is the sole publication transition.
  StepFunction finish_whole_request_from_uncommitted_retained = nullptr;
  PrefillCommitFunction commit_whole_request = nullptr;
};

struct DecodePlan {
  void* context = nullptr;
  StepFunction decode_step = nullptr;
};

struct GenerationControlOptions {
  std::uint32_t max_new_tokens = 0U;
  std::uint32_t stop_token_id = kQwen36ImEndTokenId;
  std::uint32_t max_sequence_length = 0U;
  std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
  bool capture_trace = false;
  ReferenceLogitsMode logits_mode = ReferenceLogitsMode::kFullStatistics;
  bool emit_nvtx_phase_ranges = false;
  // Test-only admission. Ordinarily, all prompt tokens are submitted to
  // prefix tiles and the final tile retains its last normalized hidden row.
  // The separate whole-request opt-in below instead submits one immutable
  // logical-panel topology, its dedicated uncommitted finalizer, and its
  // single commit callback. Default-off keeps the production prompt-(P-1)
  // plus scalar-final-step schedule.
  bool prefill_all_prompt_tokens = false;
  // Test-only admission layered on top of prefill_all_prompt_tokens. Each
  // request-sized chunk is one arbitrary 1..512-token prefix_tile call. It is
  // invalid unless whole-prompt admission is also enabled.
  bool prefill_single_arbitrary_tile = false;
  // Host-control opt-in only. This submits the complete prompt and its
  // immutable C8192 logical-panel topology exactly once, never prefix_tile.
  // It requires all-prompt admission, dedicated retained-hidden final/commit
  // callbacks, no trace, and no single-arbitrary-tile mode. No
  // ReferenceEngine sets it only when its engine-lifetime sealed plan exists.
  bool prefill_whole_request_layer_major = false;
  // Engine-sealed MLP topology. The default preserves per-panel execution;
  // the P40 value is valid only with the whole-request callback and exact
  // cold P40000 geometry.
  LayerMajorPrefillMlpScheduleTactic prefill_mlp_schedule_tactic =
      LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel;
  void* committed_token_context = nullptr;
  CommittedTokenFunction committed_token = nullptr;
};

struct GenerationControl {
  std::vector<std::uint32_t> generated_token_ids;
  std::vector<ReferenceStepResult> steps;
  ReferenceStopReason stop_reason = ReferenceStopReason::kMaxNewTokens;
  ReferenceGenerationTiming timing;
  ReferencePrefillExecutionMode prefill_execution_mode =
      ReferencePrefillExecutionMode::kLegacyC512Tiled;
  std::uint64_t prefill_logical_panel_count = 0U;
  // Route evidence passes are intentionally independent of logical panel
  // count. The P40 layer-wide MLP route publishes one complete request-wide
  // 64-layer pass after all five panels and all 64 full-M phases complete.
  std::uint64_t prefill_route_layer_pass_count = 0U;
};

struct GenerationControlResult {
  std::optional<GenerationControl> value;
  GenerationControlError error = GenerationControlError::kNone;
  ReferenceRunnerStatus runner_status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && error == GenerationControlError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Pure host generation state machine. The callback is the only runner boundary
// and makes token ordering/prefill policy testable without CUDA or model files.
[[nodiscard]] GenerationControlResult run_generation_control(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const GenerationControlOptions& options,
    const PrefillPlan& prefill_plan,
    const DecodePlan& decode_plan);

// Compatibility entry point for callers that intentionally use one callback
// context for both phases. New engine code should pass explicit plans above.
[[nodiscard]] GenerationControlResult run_generation_control(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const GenerationControlOptions& options,
    void* step_context,
    StepFunction step_function,
    PrefillTileFunction prefill_tile_function = nullptr);

// Returns the prefix that should be decoded for user-visible text. The exact
// generated id sequence retains a terminal stop id for oracle comparisons;
// only a stop that was actually observed is hidden from the text view.
[[nodiscard]] std::size_t generated_text_token_count(
    const std::vector<std::uint32_t>& generated_token_ids,
    ReferenceStopReason stop_reason,
    std::uint32_t stop_token_id) noexcept;

[[nodiscard]] std::string_view to_string(
    GenerationControlError error) noexcept;

}  // namespace reference_engine_detail

}  // namespace q3x::runtime
