#pragma once

#include "q3x/model/model_config.h"
#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/prefill_execution_plan.h"
#include "q3x/runtime/prefill_route_evidence.h"
#include "q3x/runtime/request_state.h"
#include "q3x/runtime/sm87_macrofeed_v3_receipt.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace q3x::runtime {

namespace reference_engine_detail {
class ReferenceEnginePrefillPlanFactory;
class ReferenceEnginePrefillExecutor;
}  // namespace reference_engine_detail

namespace sm87_macrofeed_v3_p40_execution_package_detail {
class Sm87MacroFeedV3P40ExecutionPackage;
}  // namespace sm87_macrofeed_v3_p40_execution_package_detail

inline constexpr std::size_t kReferenceVocabularySize = 248'320U;
inline constexpr std::size_t kReferenceHiddenSize = 5'120U;
inline constexpr std::size_t kReferenceIntermediateSize = 17'408U;
inline constexpr std::size_t kReferenceDecoderLayerCount = 64U;
inline constexpr std::size_t kReferenceTraceElements =
    (2U + 2U * kReferenceDecoderLayerCount) * kReferenceHiddenSize;
inline constexpr std::size_t kReferenceNoLayer =
    static_cast<std::size_t>(-1);

enum class ReferenceRunnerError : std::uint8_t {
  kNone = 0,
  kInvalidDependency,
  kInvalidModelWeights,
  kInvalidRequestState,
  kInvalidLayerSchedule,
  kCudaFailure,
  kAllocationFailure,
  kInvalidRunner,
  kPoisoned,
  kTokenOutOfRange,
  kCapacityExceeded,
  kTraceUnavailable,
  kNonFiniteLogits,
  kStateCommitFailure,
  kInvalidStepOptions,
  kRouteEvidenceFailure,
  kCancelled,
};

struct ReferenceRunnerStatus {
  ReferenceRunnerError error = ReferenceRunnerError::kNone;
  int cuda_error = 0;
  std::size_t layer = kReferenceNoLayer;
  const char* operation = nullptr;
  // Number of layer/phase completion events retired before cancellation was
  // observed. A younger in-flight quantum may be drained during rollback but
  // is intentionally not presented as a cancellation observation point.
  std::uint64_t retired_prefill_quanta = 0U;

  [[nodiscard]] bool ok() const noexcept {
    return error == ReferenceRunnerError::kNone && cuda_error == 0;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

[[nodiscard]] const char* reference_runner_error_string(
    ReferenceRunnerError error) noexcept;

struct ReferenceRunnerOptions {
  // Reserve one pinned BF16 trace buffer at factory time. A step only copies
  // activations when ReferenceStepOptions::capture_trace is true.
  bool enable_trace = false;
  // Explicitly opt into the SM87 weight-only projection kernels. Correctness
  // reference dispatch remains the stable default.
  ProjectionBackend projection_backend = ProjectionBackend::kReference;
};

enum class ReferenceLogitsMode : std::uint8_t {
  // Preserve the exact public reference result: chosen logit, stable
  // logsumexp, maximum log-probability, and greedy token id.
  kFullStatistics = 0,
  // Validate finiteness and return only the greedy token id. This is intended
  // for callers such as the CLI that do not expose probability statistics.
  kPredictedTokenOnly,
};

[[nodiscard]] constexpr bool is_valid_reference_logits_mode(
    const ReferenceLogitsMode mode) noexcept {
  return mode == ReferenceLogitsMode::kFullStatistics ||
         mode == ReferenceLogitsMode::kPredictedTokenOnly;
}

struct ReferenceStepOptions {
  // False is the prompt-prefix path: all 64 layers and persistent-state
  // updates still execute, while lm_head and the logits D2H copy are skipped.
  bool compute_logits = true;
  bool capture_trace = false;
  bool measure_timing = false;
  ReferenceLogitsMode logits_mode = ReferenceLogitsMode::kFullStatistics;
};

struct ReferenceStepLogits {
  std::uint32_t predicted_token_id = 0U;
  float chosen_logit = 0.0F;
  double max_log_probability = 0.0;
  double logsumexp = 0.0;
};

struct ReferenceStepPrediction {
  std::uint32_t predicted_token_id = 0U;
};

struct ReferenceStepTiming {
  // End-to-end host elapsed time, including the required stream synchronize
  // and (when requested) BF16-logits analysis.
  double elapsed_milliseconds = 0.0;
};

struct ReferenceStepResult {
  std::uint32_t position = 0U;
  std::uint32_t input_token_id = 0U;
  std::optional<ReferenceStepLogits> logits;
  std::optional<ReferenceStepTiming> timing;
  std::optional<ReferenceStepPrediction> prediction;
};

struct ReferenceStepOutcome {
  std::optional<ReferenceStepResult> value;
  ReferenceRunnerStatus status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

inline constexpr std::size_t kReferenceDecodeGraphP2MaximumSlots = 64U;

// Test-only screening surface for a short fixed-position full Decode CUDA
// Graph cache. It is deliberately bounded and is not the production decode
// scheduler. prepare captures, instantiates, and uploads the current-position
// slot without executing or committing; replay selects that position's slot
// and uses the same synchronize, host prediction validation, and commit
// boundary as step.
struct ReferenceDecodeGraphP1Stats {
  std::uint32_t position = 0U;
  std::uint32_t input_token_id = 0U;
  std::size_t node_count = 0U;
  std::size_t kernel_node_count = 0U;
  std::size_t memcpy_node_count = 0U;
  std::size_t other_node_count = 0U;
  double capture_enqueue_milliseconds = 0.0;
  double topology_inspection_milliseconds = 0.0;
  double instantiate_milliseconds = 0.0;
  double upload_ready_milliseconds = 0.0;
  double total_prepare_milliseconds = 0.0;
};

struct ReferenceDecodeGraphP1PrepareOutcome {
  std::optional<ReferenceDecodeGraphP1Stats> value;
  ReferenceRunnerStatus status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Engine-lifetime preparation result for a contiguous fixed-position cache.
// graphs[0..graph_count) are packed in ascending position order. The complete
// bank is published only after every requested slot is uploaded and ready.
struct ReferenceDecodeGraphCachePrepareResult {
  std::array<ReferenceDecodeGraphP1Stats,
             kReferenceDecodeGraphP2MaximumSlots>
      graphs{};
  std::size_t graph_count = 0U;
  std::uint64_t prepared_mask = 0U;
};

struct ReferenceDecodeGraphCachePrepareOutcome {
  std::optional<ReferenceDecodeGraphCachePrepareResult> value;
  ReferenceRunnerStatus status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct ReferencePrefillTileOptions {
  bool measure_timing = false;
  // Test-only whole-prompt admission contract. A successful tile retains the
  // final token's already-normalized hidden row for exactly one subsequent
  // finish_prefill_from_retained_tile call. The tile still commits every
  // persistent KV/GDN/conv update and the complete logical sequence length.
  bool retain_last_hidden_for_logits = false;
};

// A prefix tile never produces logits or trace data. The C512 fixed-capacity
// result keeps the runner boundary allocation-free while retaining
// one position/input record per committed token for the high-level generation
// transcript. When timing is requested, timing contains the aggregate tile
// latency. Individual step timings are absent for M>1; M=1 preserves the
// delegated step timing.
struct ReferencePrefillTileResult {
  std::array<ReferenceStepResult, kMaximumRequestPrefillChunkSize> steps{};
  std::size_t step_count = 0U;
  std::optional<ReferenceStepTiming> timing;
};

struct ReferencePrefillTileOutcome {
  std::optional<ReferencePrefillTileResult> value;
  ReferenceRunnerStatus status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Fixed-size runner result for the layer-major whole-request seam. The
// Engine adapter, not the CUDA runner, owns the potentially 40K--130K host
// transcript. Device execution remains uncommitted until the dedicated
// logits finalizer and single publication callback both succeed.
struct ReferenceWholeRequestPrefillOptions {
  bool measure_timing = false;
  // A non-null probe enables the bounded two-quantum submission window. The
  // runner polls only after the oldest layer-panel completion is visible.
  // At most one younger quantum remains in flight; failure handling drains it
  // before rollback and never publishes partially advanced request state.
  using CancellationProbe = bool (*)(void* context) noexcept;
  CancellationProbe cancellation_probe = nullptr;
  void* cancellation_context = nullptr;
};

// Compact host-visible form of the planner's device-completion receipt. The
// fixed parity geometry fits losslessly in 31 bits: layer/full-launch counts
// use six bits, the remaining counts are one-bit, and completion owns eight
// flags. Retaining the planner's size_t-heavy form for 64 layers would violate
// the runner result's fixed-size, transcript-free boundary. The runner first
// validates the full receipt through the progress transition and only then
// encodes this witness.
struct ReferenceP40VllmMarlinParityLayerCompletionReceipt {
  std::uint32_t packed = 0U;

  [[nodiscard]] static constexpr
      ReferenceP40VllmMarlinParityLayerCompletionReceipt
      from_validated(
          const PrefillP40VllmMarlinParityLayerCompletionReceipt& source)
          noexcept {
    ReferenceP40VllmMarlinParityLayerCompletionReceipt encoded{};
    encoded.packed =
        (static_cast<std::uint32_t>(source.layer_index) & 0x3fU) |
        ((static_cast<std::uint32_t>(
              source.request_lock_clear_operations) &
          0x1U)
         << 6U) |
        ((static_cast<std::uint32_t>(
              source.gate_up_full_m1024_launches) &
          0x3fU)
         << 7U) |
        ((static_cast<std::uint32_t>(
              source.gate_up_split_m64_launches) &
          0x1U)
         << 13U) |
        ((static_cast<std::uint32_t>(source.standalone_silu_launches) &
          0x1U)
         << 14U) |
        ((static_cast<std::uint32_t>(source.down_full_m1024_launches) &
          0x3fU)
         << 15U) |
        ((static_cast<std::uint32_t>(source.down_split_m64_launches) &
          0x1U)
         << 21U) |
        ((static_cast<std::uint32_t>(source.standalone_residual_launches) &
          0x1U)
         << 22U) |
        (static_cast<std::uint32_t>(source.retained_prompt_core_complete)
         << 23U) |
        (static_cast<std::uint32_t>(
             source.canonical_gate_then_up_bf16_published)
         << 24U) |
        (static_cast<std::uint32_t>(source.activated_bf16_published)
         << 25U) |
        (static_cast<std::uint32_t>(source.down_bf16_published) << 26U) |
        (static_cast<std::uint32_t>(source.stable_lock_owner_bound)
         << 27U) |
        (static_cast<std::uint32_t>(
             source.lock_owner_alias_exclusion_proved)
         << 28U) |
        (static_cast<std::uint32_t>(source.ordered_lock_protocol_completed)
         << 29U) |
        (static_cast<std::uint32_t>(
             source.request_stream_completion_observed)
         << 30U);
    return encoded;
  }

  [[nodiscard]] constexpr std::size_t layer_index() const noexcept {
    return packed & 0x3fU;
  }
  [[nodiscard]] constexpr std::size_t
  request_lock_clear_operations() const noexcept {
    return (packed >> 6U) & 0x1U;
  }
  [[nodiscard]] constexpr std::size_t
  gate_up_full_m1024_launches() const noexcept {
    return (packed >> 7U) & 0x3fU;
  }
  [[nodiscard]] constexpr std::size_t
  gate_up_split_m64_launches() const noexcept {
    return (packed >> 13U) & 0x1U;
  }
  [[nodiscard]] constexpr std::size_t standalone_silu_launches()
      const noexcept {
    return (packed >> 14U) & 0x1U;
  }
  [[nodiscard]] constexpr std::size_t
  down_full_m1024_launches() const noexcept {
    return (packed >> 15U) & 0x3fU;
  }
  [[nodiscard]] constexpr std::size_t
  down_split_m64_launches() const noexcept {
    return (packed >> 21U) & 0x1U;
  }
  [[nodiscard]] constexpr std::size_t standalone_residual_launches()
      const noexcept {
    return (packed >> 22U) & 0x1U;
  }
  [[nodiscard]] constexpr bool retained_prompt_core_complete()
      const noexcept {
    return ((packed >> 23U) & 0x1U) != 0U;
  }
  [[nodiscard]] constexpr bool canonical_gate_then_up_bf16_published()
      const noexcept {
    return ((packed >> 24U) & 0x1U) != 0U;
  }
  [[nodiscard]] constexpr bool activated_bf16_published() const noexcept {
    return ((packed >> 25U) & 0x1U) != 0U;
  }
  [[nodiscard]] constexpr bool down_bf16_published() const noexcept {
    return ((packed >> 26U) & 0x1U) != 0U;
  }
  [[nodiscard]] constexpr bool stable_lock_owner_bound() const noexcept {
    return ((packed >> 27U) & 0x1U) != 0U;
  }
  [[nodiscard]] constexpr bool lock_owner_alias_exclusion_proved()
      const noexcept {
    return ((packed >> 28U) & 0x1U) != 0U;
  }
  [[nodiscard]] constexpr bool ordered_lock_protocol_completed()
      const noexcept {
    return ((packed >> 29U) & 0x1U) != 0U;
  }
  [[nodiscard]] constexpr bool request_stream_completion_observed()
      const noexcept {
    return ((packed >> 30U) & 0x1U) != 0U;
  }
};

static_assert(
    sizeof(ReferenceP40VllmMarlinParityLayerCompletionReceipt) == 4U);

struct ReferenceWholeRequestPrefillResult {
  std::uint32_t first_position = 0U;
  std::uint32_t final_position = 0U;
  std::size_t prompt_token_count = 0U;
  std::size_t logical_panel_count = 0U;
  bool bounded_submission_window = false;
  std::size_t submission_window_retirements = 0U;
  std::size_t operator_panel_executor_hits = 0U;
  std::size_t native_group_q64_panel_hits = 0U;
  std::size_t native_group_q128_v4_panel_hits = 0U;
  std::size_t native_flashinfer_exact_panel_hits = 0U;
  std::size_t generic_qt2_hits = 0U;
  std::size_t segmented_panel_projection_hits = 0U;
  std::size_t segmented_panel_projection_physical_launches = 0U;
  std::size_t native_large_m_projection_hits = 0U;
  std::size_t native_large_m_projection_bulk_hits = 0U;
  std::size_t native_large_m_projection_oracle_partial_hits = 0U;
  std::size_t native_large_m_projection_physical_launches = 0U;
  std::size_t nvfp4_true_large_m_route_fp8_projection_hits = 0U;
  std::size_t nvfp4_true_large_m_route_fp8_projection_bulk_hits = 0U;
  std::size_t nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits = 0U;
  std::size_t nvfp4_true_large_m_route_fp8_projection_physical_launches = 0U;
  std::size_t native_nvfp4_true_large_m_projection_hits = 0U;
  std::size_t native_nvfp4_true_large_m_gate_up_hits = 0U;
  std::size_t native_nvfp4_true_large_m_down_hits = 0U;
  std::size_t native_nvfp4_true_large_m_physical_launches = 0U;
  LayerMajorPrefillMlpScheduleTactic mlp_schedule_tactic =
      LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel;
  // Physical, committed witnesses for the exact-P40000 layer-wide phase.
  // These count one layer phase / GateUp / Down pair, never one per panel.
  std::size_t layer_wide_p40_mlp_layer_hits = 0U;
  std::size_t persistent_p40_nvfp4_gate_up_hits = 0U;
  std::size_t persistent_p40_nvfp4_down_residual_hits = 0U;
  std::size_t persistent_p40_nvfp4_physical_launches = 0U;
  std::size_t persistent_p40_fp8_projection_hits = 0U;
  std::size_t persistent_p40_fp8_projection_bulk_hits = 0U;
  std::size_t persistent_p40_fp8_projection_oracle_partial_hits = 0U;
  std::size_t persistent_p40_fp8_projection_physical_launches = 0U;
  // Exact-P40000 prompt-wide whole-core witnesses. These counters are
  // intentionally orthogonal to the older logical-panel and layer-wide-MLP
  // routes: a whole-prompt Attention launch is never reported as five panel
  // launches, and a fill/drain phase is never reported as an operator-panel
  // executor hit.
  std::size_t prompt_wide_p40_whole_core_layer_hits = 0U;
  std::size_t prompt_wide_p40_fill_panel_hits = 0U;
  std::size_t prompt_wide_p40_prompt_core_hits = 0U;
  std::size_t prompt_wide_p40_drain_panel_hits = 0U;
  std::size_t prompt_wide_p40_fp8_projection_hits = 0U;
  std::size_t prompt_wide_p40_fp8_projection_physical_launches = 0U;
  std::size_t prompt_wide_p40_bf16_ab_hits = 0U;
  std::size_t prompt_wide_p40_gdn_hits = 0U;
  std::size_t native_flashinfer_exact_whole_prompt_hits = 0U;
  std::size_t packed_nvfp4_v2_gate_up_hits = 0U;
  std::size_t packed_nvfp4_v2_down_hits = 0U;
  std::size_t packed_nvfp4_v2_physical_launches = 0U;
  // Default-off stock-vLLM-Marlin P40000 projection-host-dispatch
  // witnesses. These remain disjoint from both the historical persistent
  // P40 counters and packed-v2 counters: one logical Gate+Up/Down role owns
  // forty physical projection launches, with standalone SiLU and residual
  // publication boundaries. Per-layer receipts are populated only after the
  // request stream has completed and the documented tail-lock protocol is
  // therefore observable.
  std::size_t vllm_marlin_parity_gate_up_hits = 0U;
  std::size_t vllm_marlin_parity_down_hits = 0U;
  std::size_t vllm_marlin_parity_physical_launches = 0U;
  std::size_t vllm_marlin_parity_standalone_silu_launches = 0U;
  std::size_t vllm_marlin_parity_standalone_residual_launches = 0U;
  std::size_t vllm_marlin_parity_lock_clear_operations = 0U;
  std::array<ReferenceP40VllmMarlinParityLayerCompletionReceipt,
             kReferenceDecoderLayerCount>
      vllm_marlin_parity_layer_completion_receipts{};
  std::size_t vllm_marlin_parity_layer_completion_receipt_count = 0U;
  // Present only for the default-off MacroFeed-v3 route and only after the
  // final main-stream completion barrier has authenticated all 64 layer-local
  // constituent enqueue receipts. This record is capability-free.
  std::optional<Sm87MacroFeedV3TransactionReceipt>
      macrofeed_v3_transaction_receipt;
  PrefillExecutionProgress progress;
  std::optional<ReferenceStepTiming> timing;
};

struct ReferenceWholeRequestPrefillOutcome {
  std::optional<ReferenceWholeRequestPrefillResult> value;
  ReferenceRunnerStatus status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct ConstBf16Span {
  const std::uint16_t* data = nullptr;
  std::size_t size = 0U;

  [[nodiscard]] bool empty() const noexcept { return size == 0U; }
  [[nodiscard]] const std::uint16_t& operator[](
      const std::size_t index) const noexcept {
    return data[index];
  }
};

// Non-owning view of the most recently captured, successfully committed step.
// It is invalidated by reset, runner destruction, or the next captured step.
// Raw layout is embedding, then [hidden_i, residual_i] for i=0..63, then
// final_norm. Each logical vector contains exactly 5120 BF16 elements.
struct ReferenceTraceView {
  std::uint32_t position = 0U;
  std::uint32_t input_token_id = 0U;
  const std::uint16_t* data = nullptr;
  std::size_t element_count = 0U;

  [[nodiscard]] ConstBf16Span raw() const noexcept;
  [[nodiscard]] ConstBf16Span embedding() const noexcept;
  [[nodiscard]] ConstBf16Span layer_hidden(
      std::size_t layer) const noexcept;
  [[nodiscard]] ConstBf16Span layer_residual(
      std::size_t layer) const noexcept;
  [[nodiscard]] ConstBf16Span final_norm() const noexcept;
};

// Candidate-only identity for the typed RequestState seam prepared for
// AC-PREFILL-LAYERMAJOR-8K-v1.  Building or collecting this descriptor does
// not bind a launcher, stream, event, operator, selector, or production
// route.  In particular, it is not an executable-plan attestation.
enum class ReferenceLayerMajorBindingDisposition : std::uint8_t {
  kUnboundCandidateOnly = 0,
};

// Pure-host description of every RequestState region that a future
// layer-major runner must bind.  The owning C8192 arena is deliberately not
// present: operator storage is exposed only through the typed phase regions.
struct ReferenceLayerMajorRequestBindingDescriptor {
  ReferenceLayerMajorBindingDisposition disposition =
      ReferenceLayerMajorBindingDisposition::kUnboundCandidateOnly;
  RequestMemoryProfile profile = RequestMemoryProfile::kLegacyC512;
  std::uint32_t max_sequence_length = 0U;
  std::uint32_t operator_panel_capacity_tokens = 0U;
  std::uint32_t legacy_prefill_chunk_size = 0U;
  std::uint32_t mlp_capacity_tokens = 0U;
  LayerMajorRequestLayout layout =
      LayerMajorRequestLayout::kC8192FamilyOverlay;
  LayerMajorRequestMlpLayout mlp_layout =
      LayerMajorRequestMlpLayout::kPanelLocalThreeSpan;
  std::uint64_t arena_bytes = 0U;

  std::array<RequestLayerSlot, kRequestLayerCount> layers{};
  RequestRegion conv_state_bf16;
  RequestRegion gdn_state_bf16;
  std::array<RequestRegion, kRequestFullLayerCount> key_cache_bf16;
  std::array<RequestRegion, kRequestFullLayerCount> value_cache_bf16;
  RequestRegion rope_cos_fp32;
  RequestRegion rope_sin_fp32;

  RequestMatrixRegion prompt_residual_bf16;
  RequestMatrixRegion panel_token_ids_u32;
  LayerMajorGdnPhaseRegions gdn;
  LayerMajorAttentionPhaseRegions attention;
  LayerMajorMlpPhaseRegions mlp;
  LayerMajorLegacyC512Regions legacy_c512;
  LayerMajorP40WholeCoreRegions p40_whole_core;
  RequestMatrixRegion final_hidden_bf16;

  PrefillHiddenStrategy hidden_strategy{};
  PrefillOperatorScratchStrategy scratch_strategy{};
  PrefillGdnPhysicalTactic gdn_tactic{};
  PrefillLegacyGdnPhysicalTactic legacy_gdn_tactic{};
  PrefillMlpPhysicalTactic mlp_tactic{};
};

// Pure host validator for the request-long P40 parity lock lifetime. It
// proves that the one stable lock prefix is outside every active persistent,
// family-phase, residual, handoff, RoPE, and other legacy workspace writer.
[[nodiscard]] bool
valid_reference_p40_vllm_marlin_parity_lock_lifetime_descriptor(
    const ReferenceLayerMajorRequestBindingDescriptor& descriptor) noexcept;

struct ReferenceLayerMajorRequestDescriptorOutcome {
  std::optional<ReferenceLayerMajorRequestBindingDescriptor> value;
  ReferenceRunnerStatus status;
  RequestAccessError access_error = RequestAccessError::kNone;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok() &&
           access_error == RequestAccessError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Compact persistent-state views are indexed by the RequestLayerSlot::slot
// carried in descriptor.layers.  This keeps wrong-family access impossible
// without exposing sparse untyped per-layer pointers.
struct ReferenceLayerMajorPersistentViews {
  std::array<DeviceBufferView, kRequestLinearLayerCount> conv_state_bf16;
  std::array<DeviceBufferView, kRequestLinearLayerCount> gdn_state_bf16;
  std::array<DeviceBufferView, kRequestFullLayerCount> key_cache_bf16;
  std::array<DeviceBufferView, kRequestFullLayerCount> value_cache_bf16;
  ConstDeviceBufferView rope_cos_fp32;
  ConstDeviceBufferView rope_sin_fp32;
};

// One non-owning snapshot of every typed layer-major RequestState view.  The
// exact RequestState object and its allocation must outlive every consumer.
// No raw view of c8192_family_phase_arena is exposed here.
struct ReferenceLayerMajorRequestViews {
  ReferenceLayerMajorRequestBindingDescriptor descriptor;
  DeviceMatrixView prompt_residual_bf16;
  DeviceMatrixView panel_token_ids_u32;
  LayerMajorGdnPhaseViews gdn;
  LayerMajorAttentionPhaseViews attention;
  LayerMajorMlpPhaseViews mlp;
  LayerMajorLegacyC512Views legacy_c512;
  LayerMajorP40WholeCoreViews p40_whole_core;
  DeviceMatrixView final_hidden_bf16;
  ReferenceLayerMajorPersistentViews persistent;
};

struct ReferenceLayerMajorRequestViewsOutcome {
  std::optional<ReferenceLayerMajorRequestViews> value;
  ReferenceRunnerStatus status;
  RequestAccessError access_error = RequestAccessError::kNone;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok() &&
           access_error == RequestAccessError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Pure-host, allocation-free validation and descriptor construction.  It
// accepts only the fixed layer-major RequestState profile and fails closed on
// every malformed typed family, persistent/KV/RoPE identity, tactic identity,
// schedule, capacity, or alias topology.
[[nodiscard]] ReferenceLayerMajorRequestDescriptorOutcome
build_reference_layer_major_candidate_binding_descriptor(
    const LayerMajorRequestMemoryPlan& plan) noexcept;

// Explicit typed seam. Profile validation deliberately precedes empty state
// validation so passing a legacy RequestState reports kMemoryProfileMismatch
// without touching CUDA. create_reference_runner() uses it only to bind a
// layer-major RequestState's disjoint legacy C512 compatibility views; no
// whole-request selector or executable operator plan is implied.
[[nodiscard]] ReferenceLayerMajorRequestViewsOutcome
collect_reference_layer_major_candidate_views(RequestState* state) noexcept;

[[nodiscard]] inline ReferenceLayerMajorRequestViewsOutcome
collect_reference_layer_major_candidate_views(RequestState& state) noexcept {
  return collect_reference_layer_major_candidate_views(&state);
}

namespace reference_runner_detail {

// Public, allocation-free test/oracle helpers. The logits analyzer first
// rounds every value in-place to BF16 RNE and expands it back to FP32, matching
// the vLLM BF16 logits boundary. Any NaN or infinity is rejected. Argmax ties
// select the smallest index.
[[nodiscard]] std::uint16_t float_to_bf16_rne(float value) noexcept;
[[nodiscard]] float bf16_to_float(std::uint16_t bits) noexcept;
[[nodiscard]] float round_float_to_bf16(float value) noexcept;

enum class LogitsAnalysisStatus : std::uint8_t {
  kSuccess = 0,
  kInvalidArgument,
  kNonFinite,
};

struct LogitsAnalysis {
  LogitsAnalysisStatus status = LogitsAnalysisStatus::kInvalidArgument;
  std::size_t predicted_index = 0U;
  float maximum = 0.0F;
  double logsumexp = 0.0;
  double max_log_probability = 0.0;

  [[nodiscard]] bool ok() const noexcept {
    return status == LogitsAnalysisStatus::kSuccess;
  }
};

[[nodiscard]] LogitsAnalysis analyze_bf16_logits_in_place(
    float* logits, std::size_t element_count) noexcept;
[[nodiscard]] LogitsAnalysis analyze_bf16_logits_bits(
    const std::uint16_t* logits, std::size_t element_count) noexcept;
[[nodiscard]] LogitsAnalysis analyze_bf16_argmax_in_place(
    float* logits, std::size_t element_count) noexcept;
[[nodiscard]] LogitsAnalysis analyze_bf16_argmax_bits(
    const std::uint16_t* logits, std::size_t element_count) noexcept;

// Exact payload/dimension preflight used by the runner factory. Exposed here
// so small fake weights can test FP8/NVFP4 scalar constraints without a model
// arena or a CUDA context.
[[nodiscard]] bool valid_reference_linear_weight_contract(
    const LinearWeight& weight, std::size_t output_size,
    std::size_t input_size) noexcept;

[[nodiscard]] model::LayerType expected_reference_layer_type(
    std::size_t layer) noexcept;

// Pure-host selector for the exact-shape fused GQA/gate path. first_position
// is zero-based; a complete tile is selected only when every token's causal
// sequence length is within the fused kernel limit.
[[nodiscard]] bool use_fused_gqa_sigmoid_gate_tile(
    std::size_t first_position, std::size_t token_count) noexcept;

// Admission-only decode selector.  The legacy score/softmax/value/gate path
// remains in the same ELF and is selected unless the split-KV admission flag
// is enabled and the dynamic sequence length is in [65, 4096].
[[nodiscard]] bool use_decode_gqa_splitkv(
    std::size_t sequence_length) noexcept;

// Returns the leading token count whose causal positions remain within the
// fused GQA/Gate kernel limit. A C256/C512 tile beginning before position 64
// keeps this prefix fused while its suffix follows the reference fallback.
[[nodiscard]] std::size_t fused_gqa_sigmoid_gate_prefix_token_count(
    std::size_t first_position, std::size_t token_count) noexcept;

// Pure-host selector for the production bulk causal full-attention Prefill
// route. Only an explicitly selected SM87 backend, a full-attention layer,
// and a 2..512-token tile whose complete global causal range fits the kernel
// ABI may bypass the established per-token GQA/Gate schedule. The kernel's
// QT2 grid masks the second row of an odd final query pair.
[[nodiscard]] bool use_bulk_causal_gqa_sigmoid_gate_prefill(
    ProjectionBackend backend, model::LayerType layer_type,
    std::size_t first_position, std::size_t token_count) noexcept;

// Pure-host selector for the fixed Q=24, KV=4, D=256, rotary=64 fused
// full-attention preprocessing tile. It also rejects position-table
// arithmetic overflow; callers retain the split/norm/RoPE fallback.
[[nodiscard]] bool use_qk_rope_tile(
    std::size_t first_position, std::size_t token_count) noexcept;

// Pure-host selector for the same fused preprocessing dataflow when the
// independent-token grid spans one complete request Prefill tile. Unlike the
// standalone Q/K RoPE helper, the production fused kernel has no cross-token
// state and can expose all 512 prompt rows in one launch.
[[nodiscard]] bool use_full_attention_preprocess_tile(
    std::size_t first_position, std::size_t token_count) noexcept;

inline constexpr std::size_t kPrefillResidualRmsM32Tokens = 32U;

// Pure-host decomposition of an arbitrary Prefill span into exact-M32 fused
// tiles and a final 1..31-token reference tail. A zero fused prefix means the
// existing all-reference schedule must remain in force.
struct PrefillResidualRmsM32Schedule {
  std::size_t fused_prefix_tokens = 0U;
  std::size_t fallback_tail_tokens = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return fused_prefix_tokens != 0U;
  }
};

[[nodiscard]] constexpr PrefillResidualRmsM32Schedule
prefill_residual_rms_m32_schedule(
    const std::size_t token_count,
    const std::size_t hidden_size) noexcept {
  if (token_count < kPrefillResidualRmsM32Tokens ||
      token_count > kMaximumRequestPrefillChunkSize ||
      hidden_size != kReferenceHiddenSize) {
    return {};
  }
  const std::size_t fused_prefix_tokens =
      token_count - token_count % kPrefillResidualRmsM32Tokens;
  return {fused_prefix_tokens, token_count - fused_prefix_tokens};
}

// Selects the exact-M32 residual-add plus centered-RMSNorm prefix schedule for
// every M=32..512 span. A final 1..31-token suffix uses the established
// residual-add and centered-RMSNorm launches. When selected, layer 0 retains
// its standalone input norm; each MLP residual produces the normalized input
// for the next layer (or the final norm after the final layer), so no
// subsequent whole-span input/final norm is scheduled.
[[nodiscard]] bool use_m32_prefill_residual_rms_fusion(
    std::size_t token_count, std::size_t hidden_size) noexcept;

// Pure-host selector for exact FP8 C256/C512 whole-chunk Prefill projection.
// Only explicitly selected SM87, linear-attention QKV [10240,5120] and Z
// [6144,5120], full-attention Q [12288,5120] and K/V [1024,5120], or
// attention output [5120,6144], and the production weight/input/output
// alignments may bypass the runner's established tiled schedule. Device
// companion-scale pointers are intentionally irrelevant to this kernel-only
// eligibility decision.
[[nodiscard]] bool use_fp8_whole_chunk_prefill_projection(
    ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* input, std::uint16_t* output,
    std::size_t token_count) noexcept;

// Pure-host selector for the exact FP8 C64 attention-output projection.
// Aligned [5120,6144] is handed to the tile dispatcher once; C32, decode,
// near-miss alignment/shape, and all other C64 weights preserve the
// established runner schedule.
[[nodiscard]] bool use_fp8_m64_prefill_attention_output_projection(
    ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* input, std::uint16_t* output,
    std::size_t token_count) noexcept;

// Pure-host selector for the exact NVFP4 Down whole-chunk projection. Only
// explicitly selected SM87, [5120,17408], C256/C512, and the narrow kernel's
// production alignments may bypass the runner's established C32 schedule.
// Gate/Up is intentionally ineligible for this selector.
[[nodiscard]] bool use_nvfp4_whole_chunk_prefill_down_projection(
    ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* input, std::uint16_t* output,
    std::size_t token_count) noexcept;

// Pure-host selector for the exact NVFP4 Gate/Up whole-chunk pair. Only
// explicitly selected SM87, two aligned [17408,5120] branches, C256/C512,
// and non-overlapping complete aligned output spans may use the runner's
// existing auxiliary-stream fork/join. Device companion-scale pointers remain
// launcher-validation state: a malformed selected payload must fail instead of
// becoming a serial fallback.
[[nodiscard]] bool use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
    ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* input,
    std::uint16_t* gate_output, std::uint16_t* up_output,
    std::size_t token_count) noexcept;

// Pure-host selector for the narrow C32/C64 NVFP4 MLP scheduling optimization.
// C64 retains two ordered C32 launches per branch. It accepts only the two
// exact aligned direct-output projections with non-overlapping complete output
// spans, so every route that could touch the shared FP32 fallback scratch
// remains serial.
[[nodiscard]] bool use_nvfp4_m32_prefill_gate_up_dual_stream(
    ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* input,
    std::uint16_t* gate_output, std::uint16_t* up_output,
    std::size_t token_count) noexcept;

// Test-admission controls for the scheduler-wide, all-64-layer vLLM-Marlin
// MLP route. It covers <=32/C64/C256/C512 with vLLM's corresponding M tile;
// the build-time admission remains absent from ordinary production binaries.
// These calls provide deterministic route-hit accounting to real generation.
bool exchange_nvfp4_marlin_prefill_admission_test_enabled(
    bool enabled) noexcept;
std::size_t exchange_nvfp4_marlin_prefill_admission_test_hits(
    std::size_t hits) noexcept;

// Test-admission controls for the complete 208-projection FP8 W8A16 Marlin
// route. A/B remain their canonical BF16 pair because the checkpoint stores
// them as BF16. Ordinary builds return false/zero and contain no W8 kernels.
bool exchange_fp8_marlin_prefill_admission_test_enabled(
    bool enabled) noexcept;
std::size_t exchange_fp8_marlin_prefill_admission_test_hits(
    std::size_t hits) noexcept;

[[nodiscard]] bool use_fp8_marlin_prefill_projection(
    ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* input, std::uint16_t* output,
    std::size_t token_count) noexcept;

// Pure-host validation entry used by tests and factory preflight. It checks
// exact batch-one workspace, cache, RoPE, and 48/16 schedule capacities.
[[nodiscard]] ReferenceRunnerError validate_reference_workspace_plan(
    const RequestMemoryPlan& plan) noexcept;

}  // namespace reference_runner_detail

struct ReferenceRunnerFactoryResult;

// Correctness-first, batch-one CUDA runner. ModelWeights and RequestState are
// non-owning dependencies: both exact objects, their backing CUDA allocations,
// and all bound ResidentWeights storage must outlive this runner. They must not
// be moved, reset externally, or used from another stream while it is alive.
class ReferenceRunner {
 public:
  ~ReferenceRunner();

  ReferenceRunner(const ReferenceRunner&) = delete;
  ReferenceRunner& operator=(const ReferenceRunner&) = delete;
  ReferenceRunner(ReferenceRunner&& other) noexcept;
  ReferenceRunner& operator=(ReferenceRunner&& other) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::uint32_t current_position() const noexcept;

  [[nodiscard]] ReferenceStepOutcome step(
      std::uint32_t input_token_id,
      const ReferenceStepOptions& options = {}) noexcept;

  // Strict SM87 predicted-token-only experiment. Each prepared slot fixes one
  // current_position(); replay updates its root embedding node even when the
  // token is unchanged. Slots are indexed by positions [0, 64). reset may
  // restore a prepared logical position before another replay; graph pointers
  // remain valid because RequestState storage is stable for the runner
  // lifetime.
  [[nodiscard]] ReferenceDecodeGraphP1PrepareOutcome
  prepare_fixed_position_decode_graph_p1(
      std::uint32_t input_token_id) noexcept;
  // Transactionally prepares every position in [first_position,
  // last_position]. The range must currently be empty. Failure leaves the
  // live graph bank unchanged and restores the entry sequence length.
  [[nodiscard]] ReferenceDecodeGraphCachePrepareOutcome
  prepare_fixed_position_decode_graph_cache(
      std::uint32_t first_position, std::uint32_t last_position,
      std::uint32_t input_token_id) noexcept;
  [[nodiscard]] std::uint64_t
  fixed_position_decode_graph_cache_mask() const noexcept;
  // Synchronizes both owned streams before detaching the complete bank. This
  // does not reset request state, trace state, or an existing poison marker.
  [[nodiscard]] ReferenceRunnerStatus
  clear_fixed_position_decode_graph_cache() noexcept;
  [[nodiscard]] bool has_fixed_position_decode_graph_p1(
      std::uint32_t position) const noexcept;
  [[nodiscard]] std::optional<ReferenceDecodeGraphP1Stats>
  fixed_position_decode_graph_p1_stats(
      std::uint32_t position) const noexcept;
  // Callers select their serial fallback with has_* before replay. Directly
  // replaying a missing slot is an invalid experimental operation.
  [[nodiscard]] ReferenceStepOutcome replay_fixed_position_decode_graph_p1(
      std::uint32_t input_token_id, bool measure_timing = false) noexcept;

  // Executes 1..512 non-logit prompt-prefix tokens in layer-major order. The
  // request plan must reserve at least token_count workspace rows. Operations
  // with a narrower kernel contract are enqueued as ordered subtiles.
  // Persistent conv/GDN/KV state is updated in token order. The exact aligned
  // SM87 FP8 C64 attention-output projection uses one exact kernel. Exact
  // aligned C256/C512 FP8 QKV/Z/O and NVFP4 Down projections each use one
  // whole-chunk self-hosted grid. Exact aligned NVFP4 C512 Gate/Up use the
  // native fork/join path when the auxiliary stream is available and the
  // native serial path otherwise; Down remains on the native main-stream
  // route. C32/C64 retains the M32 dual-stream schedule and C64 preserves two
  // ordered C32 launches on each branch. External-library comparators are not
  // linked into the runner and cannot be selected or used as fallbacks.
  // Exact SM87 C256/C512 full-attention tiles use one bulk causal GQA/Gate
  // launch with tile-local Q/Gate/output and global NHD K/V caches. Every
  // fallback remains on the main stream, and the logical request length is
  // committed only after the complete tile synchronizes.
  [[nodiscard]] ReferencePrefillTileOutcome prefill_prefix_tile(
      const std::uint32_t* input_token_ids, std::size_t token_count,
      const ReferencePrefillTileOptions& options = {}) noexcept;

  // Test-only final-prompt admission boundary. Consumes the retained final
  // normalized hidden row from the immediately preceding marked prefill tile
  // and runs only lm_head/logits analysis. It never gathers an embedding,
  // executes a decoder layer, updates persistent model state, or advances the
  // logical request position.
  [[nodiscard]] ReferenceStepOutcome finish_prefill_from_retained_tile(
      std::uint32_t input_token_id,
      const ReferenceStepOptions& options = {}) noexcept;

  // A successful reset synchronizes the owned streams, clears all persistent
  // request state through RequestState::reset_async, clears poison, and
  // invalidates the prior trace. Reset is the only poison recovery operation.
  [[nodiscard]] ReferenceRunnerStatus reset() noexcept;

  // Request-level route evidence is finalized by the engine only after the
  // generation controller has accepted every Prefix execution. Scalar
  // prefix steps use the exact M1 path and are therefore recorded as explicit
  // fallbacks rather than disappearing from the request witness.
  [[nodiscard]] ReferenceRunnerStatus
  record_scalar_prefill_route_fallback() noexcept;
  [[nodiscard]] PrefillRouteEvidence finalize_prefill_route_evidence(
      std::uint64_t expected_layer_passes) noexcept;

  [[nodiscard]] std::optional<ReferenceTraceView> last_trace() const noexcept;

 private:
  friend struct ReferenceRunnerFactoryResult;
  // Pure-host control tests use a friend peer so the candidate execution
  // control remains private and cannot become a production selector surface.
  friend struct ReferenceRunnerPrefillControlTestPeer;
  friend class reference_engine_detail::ReferenceEnginePrefillPlanFactory;
  friend class reference_engine_detail::ReferenceEnginePrefillExecutor;
  friend ReferenceRunnerFactoryResult create_reference_runner(
      const ModelWeights*, RequestState*, const ReferenceRunnerOptions&) noexcept;

  struct Views;

  enum class LayerMajorLayerExecutor : std::uint8_t {
    kCompatibilitySegments = 0,
    kOperatorPanel,
  };

  enum class WholeRequestPrefillStagePhase : std::uint8_t {
    kIdle = 0,
    kExecuting,
    kAwaitingLogits,
    kAwaitingCommit,
  };

  struct WholeRequestPrefillStage {
    WholeRequestPrefillStagePhase phase =
        WholeRequestPrefillStagePhase::kIdle;
    std::uint32_t expected_initial_sequence_length = 0U;
    std::uint32_t committed_sequence_length = 0U;
    std::uint32_t final_position = 0U;
    std::uint32_t final_input_token_id = 0U;
    std::size_t prompt_token_count = 0U;
    std::size_t logical_panel_count = 0U;
    std::size_t mlp_phase_count_per_layer = 0U;
    const std::uint16_t* final_normalized_hidden = nullptr;
    PrefillExecutionProgress completed_uncommitted_progress{};
    PrefillRouteEvidence route_evidence_after_commit{};
    std::optional<Sm87MacroFeedV3TransactionReceipt>
        macrofeed_v3_transaction_receipt;
  };

  // Internal exact compatibility core reached only through the Engine's
  // sealed BoundPrefillExecutionPlan and one-shot request receipt. The
  // unbound topology describes ordering only; the Engine owns cancellation,
  // bounded completion, rollback, finalization, and publication policy.
  [[nodiscard]] ReferenceWholeRequestPrefillOutcome
  prefill_whole_request_layer_major_compatibility_core(
      const std::uint32_t* input_token_ids, std::size_t token_count,
      const PrefillExecutionPlan& immutable_topology,
      const ReferenceWholeRequestPrefillOptions& options = {}) noexcept;
  [[nodiscard]] ReferenceWholeRequestPrefillOutcome
  prefill_whole_request_layer_major_panel_core(
      const std::uint32_t* input_token_ids, std::size_t token_count,
      const PrefillExecutionPlan& immutable_topology,
      LayerMajorPrefillProjectionTactic projection_tactic,
      LayerMajorPrefillFullAttentionTactic full_attention_tactic,
      const ReferenceWholeRequestPrefillOptions& options = {},
      const sm87_macrofeed_v3_p40_execution_package_detail::
          Sm87MacroFeedV3P40ExecutionPackage* macrofeed_v3_package = nullptr,
      std::uint64_t macrofeed_v3_request_identity = 0U) noexcept;
  [[nodiscard]] ReferenceWholeRequestPrefillOutcome
  prefill_whole_request_layer_major_core(
      const std::uint32_t* input_token_ids, std::size_t token_count,
      const PrefillExecutionPlan& immutable_topology,
      LayerMajorLayerExecutor executor,
      LayerMajorPrefillProjectionTactic projection_tactic,
      LayerMajorPrefillFullAttentionTactic full_attention_tactic,
      const ReferenceWholeRequestPrefillOptions& options,
      const sm87_macrofeed_v3_p40_execution_package_detail::
          Sm87MacroFeedV3P40ExecutionPackage* macrofeed_v3_package,
      std::uint64_t macrofeed_v3_request_identity) noexcept;
  [[nodiscard]] ReferenceStepOutcome
  finish_whole_request_compatibility_core(
      std::uint32_t input_token_id,
      const ReferenceStepOptions& options = {}) noexcept;
  [[nodiscard]] ReferenceRunnerStatus
  commit_whole_request_layer_major_compatibility_core(
      const PrefillExecutionPlan& immutable_topology,
      const PrefillExecutionProgress& completed_uncommitted_progress) noexcept;
  [[nodiscard]] static RequestMemoryProfile
  whole_request_commit_memory_profile(
      LayerMajorPrefillMlpScheduleTactic mlp_schedule_tactic) noexcept;

  struct PrefillTileExecutionControl {
    // The public legacy path reads RequestState::current_position(). A future
    // whole-request executor must provide the logical panel position because
    // it deliberately leaves host sequence length uncommitted until the end.
    std::optional<std::uint32_t> first_position_override;
    std::size_t layer_begin = 0U;
    std::size_t layer_end = kReferenceDecoderLayerCount;
    bool gather_embedding = true;
    bool apply_final_norm = true;
    bool synchronize = true;
    bool commit_state = true;
    bool commit_route = true;
    bool allow_scalar_m1_delegate = true;
    bool allow_cross_layer_m32_fusion = true;
    bool emit_commit_hooks = true;
    // These process-selected routes default closed so a hand-built
    // single-layer control cannot inherit an admission switch.
    // legacy_prefill_tile_execution_control() explicitly restores the
    // historical public behavior. The Engine-sealed whole-request plan may
    // separately force the exact native C64 GDN route below.
    bool allow_experimental_gdn_b8_admission = false;
    bool allow_experimental_gdn_chunk64_native_admission = false;
    bool allow_experimental_gdn_chunk64_reference_admission = false;
    // Engine-sealed layer-major evaluation plans bind the installed Marlin
    // sidecars and exact native GDN workspace once at engine creation. These
    // flags bypass request-time environment selection only for that private
    // compatibility core. Bound GDN must execute natively throughout its
    // certified M32..M512 envelope; smaller segments retain the certified
    // self-owned exact fallback and remain visible as such in route evidence.
    bool force_bound_nvfp4_marlin_prefill = false;
    bool force_bound_fp8_marlin_prefill = false;
    bool force_bound_gdn_chunk64_native_prefill = false;
  };

  struct PrefillTileExecutionSelection {
    std::uint32_t first_position = 0U;
    std::uint32_t completed_position = 0U;
    bool delegate_scalar_m1 = false;
  };

  // Route slots retain the ordinal identity that the request-level evidence
  // intentionally erases. In particular, full-Attention Q, K, and V must be
  // reduced independently across physical segments before their three hits
  // are collapsed into the public kFp8Qkv role.
  enum class PrefillLayerRouteSlot : std::uint8_t {
    kNvFp4GateUp = 0,
    kNvFp4Down,
    kQOrLinearQkv,
    kFullK,
    kFullV,
    kLinearZ,
    kO,
    kAttention,
    kGdn,
    kCount,
  };

  inline static constexpr std::size_t kPrefillLayerRouteSlotCount =
      static_cast<std::size_t>(PrefillLayerRouteSlot::kCount);
  static_assert(kPrefillLayerRouteSlotCount <= 16U);
  static_assert(kPrefillForbiddenBoundaryCount <= 8U);

  struct PrefillLayerSegmentRouteFragment {
    std::array<PrefillRouteDisposition, kPrefillLayerRouteSlotCount>
        dispositions{};
    std::uint16_t recorded_slots = 0U;
    std::uint8_t forbidden_boundaries = 0U;
    std::size_t layer = kReferenceNoLayer;
    std::uint32_t first_position = 0U;
    std::uint32_t token_count = 0U;
  };

  struct PrefillEnqueueRouteFragment {
    // Public 0..64 execution collapses each completed layer here and commits
    // this exact legacy evidence only after its normal synchronization point.
    PrefillRouteEvidence legacy_layer_pass;
    // Candidate selection admits exactly one layer, for which ordinal route
    // identity remains available to the same-layer physical-segment reducer.
    PrefillLayerSegmentRouteFragment layer_segment;
    bool has_single_layer_segment = false;
  };

  // Device-enqueue result for one already-admitted physical segment. It is
  // deliberately independent of ReferencePrefillTileOutcome: candidate
  // layer-major traversal must not materialize the legacy C512 per-token
  // transcript after every model layer.
  struct PrefillLayerSegmentEnqueueResult {
    ReferenceRunnerStatus status;
    PrefillEnqueueRouteFragment route_fragment;
    std::size_t segmented_panel_projection_hits = 0U;
    std::size_t segmented_panel_projection_physical_launches = 0U;
    std::size_t native_large_m_projection_hits = 0U;
    std::size_t native_large_m_projection_bulk_hits = 0U;
    std::size_t native_large_m_projection_oracle_partial_hits = 0U;
    std::size_t native_large_m_projection_physical_launches = 0U;
    std::size_t nvfp4_true_large_m_route_fp8_projection_hits = 0U;
    std::size_t nvfp4_true_large_m_route_fp8_projection_bulk_hits = 0U;
    std::size_t nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits =
        0U;
    std::size_t nvfp4_true_large_m_route_fp8_projection_physical_launches =
        0U;
    std::size_t native_nvfp4_true_large_m_projection_hits = 0U;
    std::size_t native_nvfp4_true_large_m_gate_up_hits = 0U;
    std::size_t native_nvfp4_true_large_m_down_hits = 0U;
    std::size_t native_nvfp4_true_large_m_physical_launches = 0U;
    std::size_t persistent_p40_fp8_projection_hits = 0U;
    std::size_t persistent_p40_fp8_projection_bulk_hits = 0U;
    std::size_t persistent_p40_fp8_projection_oracle_partial_hits = 0U;
    std::size_t persistent_p40_fp8_projection_physical_launches = 0U;
    bool mlp_deferred = false;

    [[nodiscard]] bool ok() const noexcept { return status.ok(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  };

  // Same-layer reduction state for the physical C512 segments that implement
  // one logical operator panel. Ordinal slots take the weakest observed
  // disposition without multiplying logical hits by physical segment count.
  struct PrefillLayerRouteReducer {
    PrefillLayerSegmentRouteFragment route_fragment;
    bool initialized = false;
  };

  [[nodiscard]] static PrefillTileExecutionControl
  legacy_prefill_tile_execution_control() noexcept;
  [[nodiscard]] static bool is_legacy_prefill_tile_execution_control(
      const PrefillTileExecutionControl& control) noexcept;
  [[nodiscard]] static ReferenceRunnerStatus select_prefill_tile_execution(
      const PrefillTileExecutionControl& control,
      std::uint32_t current_position, std::uint32_t max_sequence_length,
      std::uint32_t workspace_token_capacity, std::size_t token_count,
      const ReferencePrefillTileOptions& options,
      PrefillTileExecutionSelection& selection) noexcept;
  [[nodiscard]] static ReferenceRunnerStatus
  reduce_prefill_layer_route_fragment(
      const PrefillLayerSegmentRouteFragment& segment_fragment,
      PrefillLayerRouteReducer& reducer) noexcept;
  [[nodiscard]] static std::uint16_t expected_prefill_layer_route_slots(
      std::size_t layer) noexcept;
  [[nodiscard]] static ReferenceRunnerStatus
  validate_prefill_layer_route_fragment(
      const PrefillLayerSegmentRouteFragment& fragment) noexcept;
  [[nodiscard]] static ReferenceRunnerStatus
  validate_layer_wide_p40_prefill_layer_route_fragment(
      const PrefillLayerSegmentRouteFragment& fragment,
      LayerMajorPrefillProjectionTactic projection_tactic) noexcept;
  [[nodiscard]] static ReferenceRunnerStatus
  collapse_prefill_layer_route_fragment(
      const PrefillLayerSegmentRouteFragment& layer_fragment,
      PrefillRouteEvidence& layer_pass) noexcept;
  [[nodiscard]] static ReferenceRunnerStatus
  collapse_layer_wide_p40_prefill_layer_route_fragment(
      const PrefillLayerSegmentRouteFragment& layer_fragment,
      PrefillRouteEvidence& request_pass,
      LayerMajorPrefillProjectionTactic projection_tactic) noexcept;
  [[nodiscard]] static ReferenceRunnerStatus
  collapse_validated_prefill_layer_route_fragment(
      const PrefillLayerSegmentRouteFragment& layer_fragment,
      PrefillRouteEvidence& layer_pass) noexcept;
  [[nodiscard]] PrefillLayerSegmentEnqueueResult
  enqueue_prefill_layer_segment(
      const std::uint32_t* input_token_ids, std::size_t token_count,
      std::uint32_t first_position,
      const PrefillTileExecutionControl& control,
      const Views& execution_views) noexcept;
  [[nodiscard]] PrefillLayerSegmentEnqueueResult
  enqueue_prefill_layer_panel(
      const std::uint32_t* input_token_ids, std::size_t token_count,
      std::uint32_t first_position, std::size_t layer,
      LayerMajorPrefillProjectionTactic projection_tactic,
      LayerMajorPrefillFullAttentionTactic full_attention_tactic,
      bool defer_mlp,
      const ReferenceLayerMajorRequestViews& request_views) noexcept;
  enum class PromptWideP40ProjectionPackage : std::uint8_t {
    kWholeCoreV10 = 0,
    kProjectionResetV11,
    kPackedV1,
    kPackedNvfp4V2,
    kVllmMarlinParityV1,
    kMacroFeedV3,
  };

  struct MacroFeedV3LayerEnqueueEvidence;
  [[nodiscard]] ReferenceRunnerStatus enqueue_layer_wide_p40_mlp(
      std::size_t layer,
      const ReferenceLayerMajorRequestViews& request_views,
      PromptWideP40ProjectionPackage projection_package =
          PromptWideP40ProjectionPackage::kWholeCoreV10) noexcept;
  [[nodiscard]] ReferenceRunnerStatus
  enqueue_prompt_wide_p40_vllm_marlin_parity_mlp(
      std::size_t layer,
      const ReferenceLayerMajorRequestViews& request_views,
      PrefillP40VllmMarlinParityLayerCompletionReceipt& receipt) noexcept;
  [[nodiscard]] static bool
  valid_prompt_wide_p40_whole_core_runner_contract(
      const PrefillExecutionPlan& immutable_topology,
      RequestMemoryProfile memory_profile,
      std::uint32_t max_sequence_length,
      LayerMajorLayerExecutor executor,
      LayerMajorPrefillProjectionTactic projection_tactic,
      LayerMajorPrefillFullAttentionTactic full_attention_tactic) noexcept;
  [[nodiscard]] ReferenceRunnerStatus
  enqueue_prompt_wide_p40_whole_core_fill_panel(
      std::size_t layer, const PrefillOperatorPanel& panel,
      const ReferenceLayerMajorRequestViews& request_views,
      PromptWideP40ProjectionPackage projection_package,
      std::size_t& fp8_projection_hits,
      std::size_t& fp8_physical_launches) noexcept;
  [[nodiscard]] ReferenceRunnerStatus
  enqueue_prompt_wide_p40_whole_core_prompt_core(
      std::size_t layer,
      const ReferenceLayerMajorRequestViews& request_views,
      PromptWideP40ProjectionPackage projection_package,
      std::size_t& fp8_projection_hits,
      std::size_t& fp8_physical_launches) noexcept;
  [[nodiscard]] ReferenceRunnerStatus
  enqueue_prompt_wide_p40_whole_core_drain_panel(
      std::size_t layer, const PrefillOperatorPanel& panel,
      const ReferenceLayerMajorRequestViews& request_views,
      PromptWideP40ProjectionPackage projection_package,
      std::size_t& fp8_projection_hits,
      std::size_t& fp8_physical_launches) noexcept;
#if defined(Q3X_ENABLE_SM87_MACROFEED_V3_P40_EXECUTOR_ADMISSION)
  [[nodiscard]] ReferenceRunnerStatus
  enqueue_prompt_wide_p40_macrofeed_v3_fill_panel(
      std::size_t layer, const PrefillOperatorPanel& panel,
      const ReferenceLayerMajorRequestViews& request_views,
      MacroFeedV3LayerEnqueueEvidence& evidence) noexcept;
  [[nodiscard]] ReferenceRunnerStatus
  enqueue_prompt_wide_p40_macrofeed_v3_prompt_core(
      std::size_t layer,
      const ReferenceLayerMajorRequestViews& request_views,
      const sm87_macrofeed_v3_p40_execution_package_detail::
          Sm87MacroFeedV3P40ExecutionPackage& package,
      MacroFeedV3LayerEnqueueEvidence& evidence,
      std::size_t& fp8_projection_hits,
      std::size_t& fp8_physical_launches) noexcept;
  [[nodiscard]] ReferenceRunnerStatus
  enqueue_prompt_wide_p40_macrofeed_v3_drain_panel(
      std::size_t layer, const PrefillOperatorPanel& panel,
      const ReferenceLayerMajorRequestViews& request_views,
      MacroFeedV3LayerEnqueueEvidence& evidence) noexcept;
  [[nodiscard]] ReferenceRunnerStatus
  enqueue_prompt_wide_p40_macrofeed_v3_mlp(
      std::size_t layer,
      const ReferenceLayerMajorRequestViews& request_views,
      const sm87_macrofeed_v3_p40_execution_package_detail::
          Sm87MacroFeedV3P40ExecutionPackage& package,
      MacroFeedV3LayerEnqueueEvidence& evidence) noexcept;
#endif
  [[nodiscard]] static bool same_prefill_execution_progress(
      const PrefillExecutionProgress& left,
      const PrefillExecutionProgress& right) noexcept;
  [[nodiscard]] bool whole_request_prefill_active() const noexcept;
  [[nodiscard]] ReferenceWholeRequestPrefillOutcome
  fail_whole_request_prefill(ReferenceRunnerStatus status) noexcept;
  [[nodiscard]] ReferenceRunnerStatus fail_whole_request_status(
      ReferenceRunnerStatus status) noexcept;

  struct Views {
    std::uint16_t* hidden[3]{};
    std::uint16_t* projection[4]{};
    std::uint16_t* linear_a = nullptr;
    std::uint16_t* linear_b = nullptr;
    float* fp32_scratch = nullptr;
    std::size_t fp32_scratch_elements = 0U;
    std::uint16_t* conv_state[kReferenceDecoderLayerCount]{};
    std::uint16_t* gdn_state[kReferenceDecoderLayerCount]{};
    std::uint16_t* key_cache[kReferenceDecoderLayerCount]{};
    std::uint16_t* value_cache[kReferenceDecoderLayerCount]{};
    const float* rope_cos = nullptr;
    const float* rope_sin = nullptr;
  };

  [[nodiscard]] static ReferenceRunnerStatus collect_request_views(
      RequestState* state, Views& views) noexcept;
  // Converts the already authenticated candidate snapshot into the compact
  // pointer table shared by Decode and the extracted C512 layer body.  The
  // complete typed snapshot remains cached separately for the future
  // whole-request executor; this mapper never reinterprets the untyped C8192
  // family arena.
  [[nodiscard]] static ReferenceRunnerStatus
  map_layer_major_candidate_views(
      const ReferenceLayerMajorRequestViews& candidate,
      Views& views) noexcept;
  [[nodiscard]] ReferenceRunnerStatus bind_layer_major_candidate_views(
      ReferenceLayerMajorRequestViews&& candidate) noexcept;
  ReferenceRunner() noexcept = default;
  void release() noexcept;
  [[nodiscard]] ReferenceStepOutcome fail_step(
      ReferenceRunnerStatus status) noexcept;
  [[nodiscard]] ReferencePrefillTileOutcome fail_prefill_tile(
      ReferenceRunnerStatus status) noexcept;

  enum class DecodeGraphP1Action : std::uint8_t {
    kDisabled = 0,
    kCaptureOnly,
    kReplay,
  };
  struct DecodeGraphP1Slot;
  [[nodiscard]] ReferenceStepOutcome step_impl(
      std::uint32_t input_token_id, const ReferenceStepOptions& options,
      DecodeGraphP1Action graph_action,
      DecodeGraphP1Slot* capture_destination = nullptr) noexcept;

  struct DecodeGraphP1KernelLaunch {
    void* function = nullptr;
    std::array<unsigned int, 3U> grid{};
    std::array<unsigned int, 3U> block{};
    unsigned int shared_memory_bytes = 0U;
  };

  struct DecodeGraphP1Slot {
    void* graph = nullptr;
    void* exec = nullptr;
    void* embedding_node = nullptr;
    ReferenceDecodeGraphP1Stats stats{};
    DecodeGraphP1KernelLaunch embedding_launch{};
  };

  [[nodiscard]] static int destroy_decode_graph_p1_slot(
      DecodeGraphP1Slot& slot) noexcept;
  void destroy_decode_graph_p1_slot(std::size_t position) noexcept;
  void destroy_decode_graph_p1() noexcept;

  const ModelWeights* weights_ = nullptr;
  RequestState* state_ = nullptr;
  void* stream_ = nullptr;
  void* prefill_auxiliary_stream_ = nullptr;
  void* prefill_branch_ready_event_ = nullptr;
  void* prefill_branch_done_event_ = nullptr;
  inline static constexpr std::size_t
      kWholeRequestSubmissionWindowSlots = 2U;
  std::array<void*, kWholeRequestSubmissionWindowSlots>
      whole_request_submission_events_{};
  void* prefill_gdn_chunk64_reference_context_ = nullptr;
  void* prefill_gdn_chunk64_reference_workspace_ = nullptr;
  std::size_t prefill_gdn_chunk64_reference_workspace_bytes_ = 0U;
  void* prefill_gdn_chunk64_native_workspace_ = nullptr;
  std::size_t prefill_gdn_chunk64_native_workspace_bytes_ = 0U;
  void* pinned_logits_ = nullptr;
  std::uint16_t* pinned_trace_ = nullptr;
  std::array<DecodeGraphP1Slot, kReferenceDecodeGraphP2MaximumSlots>
      decode_graph_p1_slots_{};
  bool decode_graph_capture_active_ = false;
  Views views_{};
  std::optional<ReferenceLayerMajorRequestViews>
      layer_major_request_views_;
  WholeRequestPrefillStage whole_request_prefill_stage_{};
  ProjectionBackend projection_backend_ = ProjectionBackend::kReference;
  bool trace_enabled_ = false;
  bool trace_valid_ = false;
  bool poisoned_ = false;
  // V3 exact GDN may have advanced persistent state before a cancellation or
  // later failure becomes observable. While set, ordinary persistent-only
  // reset is insufficient; reset() must clear the complete mutable arena.
  bool macrofeed_v3_cold_reset_required_ = false;
  bool retained_prefill_hidden_valid_ = false;
  std::uint32_t retained_prefill_position_ = 0U;
  std::uint32_t retained_prefill_input_token_ = 0U;
  std::size_t retained_prefill_hidden_row_ = 0U;
  std::uint32_t trace_position_ = 0U;
  std::uint32_t trace_input_token_ = 0U;
  PrefillRouteEvidence prefill_route_evidence_{};
};

struct ReferenceRunnerFactoryResult {
  std::optional<ReferenceRunner> value;
  ReferenceRunnerStatus diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Pointer form deliberately permits deterministic null-dependency factory
// error tests without creating CUDA state. Valid runners retain non-owning
// references exactly as documented on ReferenceRunner.
[[nodiscard]] ReferenceRunnerFactoryResult create_reference_runner(
    const ModelWeights* weights, RequestState* state,
    const ReferenceRunnerOptions& options = {}) noexcept;

[[nodiscard]] inline ReferenceRunnerFactoryResult create_reference_runner(
    const ModelWeights& weights, RequestState& state,
    const ReferenceRunnerOptions& options = {}) noexcept {
  return create_reference_runner(&weights, &state, options);
}

}  // namespace q3x::runtime
