#pragma once

#include "q3x/model/model_config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace q3x::runtime {

inline constexpr std::uint64_t kRequestArenaAlignment = 256U;
inline constexpr std::uint64_t kDefaultRequestMaxSequenceLength = 128U;
inline constexpr std::uint64_t kAbsoluteRequestMaxSequenceLength = 262'144U;
inline constexpr std::uint32_t kDefaultRequestPrefillChunkSize = 1U;
inline constexpr std::uint32_t kMaximumRequestPrefillChunkSize = 512U;
inline constexpr std::size_t kRequestLayerCount = 64U;
inline constexpr std::size_t kRequestLinearLayerCount = 48U;
inline constexpr std::size_t kRequestFullLayerCount = 16U;
inline constexpr std::size_t kRequestHiddenBufferCount = 3U;
inline constexpr std::size_t kRequestProjectionBufferCount = 4U;
inline constexpr std::uint64_t kRequestConvStateBytes = 2'949'120U;
inline constexpr std::uint64_t kRequestGdnStateBytes = 75'497'472U;
inline constexpr std::uint64_t kRequestKvBytesPerToken = 65'536U;
inline constexpr std::uint64_t kDefaultRequestArenaBytes = 88'087'040U;
inline constexpr std::uint64_t kMaximumRequestArenaBytes = 17'437'720'576ULL;
inline constexpr std::uint32_t kLayerMajorRequestOperatorPanelCapacity = 8'192U;

// Defined by prefill_workspace_plan.h. Fixed underlying types let this header
// retain the selected planner identities without including the planner back
// through its RequestState dependency.
enum class PrefillHiddenStrategy : std::uint8_t;
enum class PrefillOperatorScratchStrategy : std::uint8_t;
enum class PrefillGdnPhysicalTactic : std::uint8_t;
enum class PrefillLegacyGdnPhysicalTactic : std::uint8_t;
enum class PrefillMlpPhysicalTactic : std::uint8_t;

enum class RequestMemoryProfile : std::uint8_t {
    kLegacyC512 = 0,
    kLayerMajorC8192,
    // Default-off exact-P40000 whole-core arena.  This profile is distinct
    // from the C8192 compatibility layout and admits only P40000 prompt work
    // inside a P40001 request capacity.
    kLayerMajorP40WholeCore,
};

enum class RequestErrorCode : std::uint8_t {
    kNone,
    kInvalidOption,
    kArithmeticOverflow,
    kInvalidLayerSchedule,
    kArenaLimitExceeded,
    kInsufficientDeviceMemory,
    kCudaFailure,
    kAllocationFailure,
};

struct RequestDiagnostic {
    RequestErrorCode code = RequestErrorCode::kNone;
    std::string message;
    std::string context;
    std::string expected;
    std::string actual;
    int cuda_error = 0;
};

struct RequestMemoryOptions {
    // The runtime is intentionally batch-one. Keeping this explicit makes an
    // accidental future batch>1 caller fail closed instead of underallocating.
    std::uint32_t batch_size = 1U;
    std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
    std::uint64_t max_sequence_length = kDefaultRequestMaxSequenceLength;
    std::uint64_t max_arena_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    std::uint64_t min_free_bytes_after_create =
        8ULL * 1024ULL * 1024ULL * 1024ULL;
};

// The layer-major architecture has no implicit sequence capacity. Callers
// must name the admitted request bucket and use the separate plan/create
// entry points below; it can never be selected by the legacy defaults.
enum class LayerMajorRequestMlpLayout : std::uint8_t {
    kPanelLocalThreeSpan = 0,
    // Test-only exact-P40000 persistent route. One full-M activated span and
    // one disjoint normalized-input span are sufficient because GateUp fuses
    // SiLU and Down publishes directly into the request residual.
    kLayerWideP40PersistentTwoSpan,
    // Default-off stock-vLLM-Marlin parity layout for exact P40000. One
    // canonical token-major [M,34816] GateThenUp output and one independent
    // [M,17408] activated output are materialized. Both projection temporary
    // regions alias one sequentially reused Ctmp span; request-long locks use
    // a stable physically disjoint legacy owner. Only the legacy Gate/Up
    // matrix views are empty.
    kLayerWideP40MarlinParityMergedGateUp,
};

// Orthogonal to the MLP lifetime choice.  The old profile remains a C8192
// family overlay; the new profile is one prompt-wide whole-core allocation
// scheduled as exactly five P8000 panels.
enum class LayerMajorRequestLayout : std::uint8_t {
    kC8192FamilyOverlay = 0,
    kP40WholeCorePromptWide,
};

struct LayerMajorRequestMemoryOptions {
    std::uint32_t batch_size = 1U;
    std::uint64_t max_sequence_length = 0U;
    std::uint64_t max_arena_bytes = kMaximumRequestArenaBytes;
    std::uint64_t min_free_bytes_after_create =
        8ULL * 1024ULL * 1024ULL * 1024ULL;
    LayerMajorRequestMlpLayout mlp_layout =
        LayerMajorRequestMlpLayout::kPanelLocalThreeSpan;
    LayerMajorRequestLayout layout =
        LayerMajorRequestLayout::kC8192FamilyOverlay;
};

struct RequestRegion {
    std::uint64_t arena_offset = 0U;
    std::uint64_t byte_size = 0U;
    std::uint64_t element_capacity = 0U;
    std::uint32_t element_size_bytes = 0U;
};

struct RequestMatrixRegion {
    RequestRegion storage;
    std::uint32_t row_capacity = 0U;
    std::uint32_t columns = 0U;
    std::uint64_t row_stride_elements = 0U;
};

struct RequestLayerSlot {
    model::LayerType type = model::LayerType::kInvalid;
    std::size_t slot = 0U;
};

enum class RequestAccessError : std::uint8_t {
    kNone,
    kLayerOutOfRange,
    kLayerTypeMismatch,
    kSlotOutOfRange,
    kPositionOutOfRange,
    kCapacityExceeded,
    kInvalidBufferIndex,
    kEmptyState,
    kMemoryProfileMismatch,
    kSequenceLengthMismatch,
    kSequenceLengthRegression,
    kInvalidResetPlan,
};

// Request reuse never exposes a caller-selected production tactic. The
// ordinary engine derives one of these reset modes from its sealed request
// lifecycle: a newly created/already-reset arena can be reused as-is, an
// exactly committed request can clear only its visible KV prefix, and every
// uncertain boundary falls back to the complete persistent span.
enum class RequestStateResetMode : std::uint8_t {
    kAlreadyClean = 0,
    kCommittedDirtyPrefix,
    kConservativeFull,
};

struct RequestStateResetPlan {
    bool valid = false;
    RequestStateResetMode mode = RequestStateResetMode::kConservativeFull;
    std::uint32_t cleared_positions = 0U;
    std::uint64_t zeroed_bytes = 0U;
    std::uint32_t memset_operations = 0U;
};

// Appended to the successful ordinary request witness. The elapsed interval
// includes the completion synchronization, so it is a request-visible reset
// cost rather than enqueue-only host overhead.
struct RequestStateResetReceipt {
    RequestStateResetMode mode = RequestStateResetMode::kConservativeFull;
    std::uint32_t cleared_positions = 0U;
    std::uint64_t zeroed_bytes = 0U;
    double milliseconds = 0.0;
};

struct RequestLayerSlotResult {
    std::optional<RequestLayerSlot> value;
    RequestAccessError error = RequestAccessError::kNone;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && error == RequestAccessError::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Maps the fixed Qwen3.6-27B schedule (layers 3,7,...,63 are full
// attention). required_type must be linear or full; a mismatched or out-of-
// range request is rejected rather than silently returning another slot kind.
[[nodiscard]] RequestLayerSlotResult map_request_layer(
    std::size_t layer_index,
    model::LayerType required_type) noexcept;

// Pure profile gate shared by the legacy flat accessors and layer-major typed
// accessors. A caller must never reinterpret one workspace profile as the
// other even when physical offsets happen to be representable.
[[nodiscard]] RequestAccessError validate_request_memory_profile(
    RequestMemoryProfile actual,
    RequestMemoryProfile required) noexcept;

// Pure validation for the conditional whole-request host-length publication.
// The check order is part of the contract: a stale expected value wins over
// desired-value errors, then capacity is checked before monotonicity.
[[nodiscard]] RequestAccessError validate_sequence_length_publication(
    std::uint32_t current,
    std::uint32_t expected_current,
    std::uint32_t desired,
    std::uint32_t max_sequence_length) noexcept;

struct RequestMemoryPlan {
    std::uint32_t batch_size = 1U;
    std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
    std::uint32_t max_sequence_length = 0U;
    std::uint64_t arena_bytes = 0U;
    std::uint64_t persistent_offset = 0U;
    std::uint64_t persistent_bytes = 0U;
    std::uint64_t workspace_offset = 0U;
    std::uint64_t workspace_bytes = 0U;
    std::uint64_t rope_offset = 0U;
    std::uint64_t rope_bytes = 0U;

    // Aggregate canonical BF16 persistent states. Per-layer views are derived
    // with the checked schedule mapping.
    RequestRegion conv_state;  // [48, 10240, 3]
    RequestRegion gdn_state;   // [48, 48, 128, 128]
    std::array<RequestRegion, kRequestFullLayerCount> key_cache;
    std::array<RequestRegion, kRequestFullLayerCount> value_cache;

    std::array<RequestRegion, kRequestHiddenBufferCount> hidden_bf16;
    std::array<RequestRegion, kRequestProjectionBufferCount> projection_bf16;
    RequestRegion linear_a_bf16;  // [48], independent from projection buffers
    RequestRegion linear_b_bf16;  // [48], independent from projection buffers
    RequestRegion fp32_scratch;
    // Aliases fp32_scratch. element_capacity is the logical minimum required
    // for [24, max_sequence_length] attention probabilities.
    RequestRegion gqa_probability_scratch;
    // [max_sequence_length, 32]. Values are generated in FP32, rounded to
    // BF16 RNE like the vLLM cache, then expanded back to FP32 storage for the
    // current decode-op float-pointer ABI.
    RequestRegion rope_cos_fp32;
    RequestRegion rope_sin_fp32;

    std::array<RequestLayerSlot, kRequestLayerCount> layers;
    // Appended in ABI 0.5.0 so every legacy field retains its prior offset.
    RequestMemoryProfile profile = RequestMemoryProfile::kLegacyC512;
};

// Pure host plan used by the automatic runner lifecycle and its host tests.
// committed_positions is consumed only by kCommittedDirtyPrefix. A valid
// dirty-prefix plan always covers the complete Conv/GDN span plus the exact
// prefix of every one of the 16 K/V pairs. A conservative plan is the legacy
// one-span reset; the already-clean plan enqueues no work.
[[nodiscard]] RequestStateResetPlan build_request_state_reset_plan(
    const RequestMemoryPlan& plan,
    RequestStateResetMode mode,
    std::uint32_t committed_positions = 0U) noexcept;

struct RequestPlanResult {
    std::optional<RequestMemoryPlan> value;
    RequestDiagnostic diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && diagnostic.code == RequestErrorCode::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Typed aliases into one C8192 family arena. Each field is a contractual
// phase view; callers never receive an untyped view of the owning arena.
struct LayerMajorGdnPhaseRegions {
    RequestMatrixRegion qkv_bf16;                 // [8192, 10240]
    RequestMatrixRegion z_bf16;                   // [8192, 6144]
    RequestMatrixRegion a_bf16;                   // [8192, 48]
    RequestMatrixRegion b_bf16;                   // [8192, 48]
    RequestMatrixRegion recurrent_core_bf16;      // [8192, 6144]
    RequestRegion native_c64_workspace;
    // Dense normalized input initially occupies the future recurrent-core
    // span and must die after the input projections, before core output is
    // written. This is not a row-prefix alias.
    RequestMatrixRegion normalized_input_bf16;
    RequestRegion input_projection_temporary;
    // Dense temporal reinterpretation after the complete QKV consumer phase.
    RequestMatrixRegion branch_output_bf16;
    RequestRegion output_projection_temporary;
};

struct LayerMajorAttentionPhaseRegions {
    RequestMatrixRegion raw_q_gate_bf16;          // [8192, 12288]
    RequestMatrixRegion processed_q_bf16;         // [8192, 6144]
    RequestMatrixRegion packed_gate_bf16;         // [8192, 6144]
    // Dense normalized input initially occupies the future processed-Q span
    // and must die after Q/K/V projection, before preprocess writes processed
    // Q. This is not a row-prefix alias.
    RequestMatrixRegion normalized_input_bf16;
    RequestRegion input_projection_temporary;
    RequestMatrixRegion core_output_bf16;         // dead raw first span
    RequestMatrixRegion branch_output_bf16;       // dead raw second span
    RequestRegion output_projection_temporary;
};

struct LayerMajorMlpPhaseRegions {
    // Present only for the exact-P40000 Marlin-parity layout. It is one
    // token-major [40000,34816] row-major tensor with per-token columns
    // [Gate(17408), Up(17408)]. Every other layout leaves it empty.
    RequestMatrixRegion merged_gate_up_bf16;
    // Panel-local layouts expose independent Gate/Up matrices. Persistent
    // fused layouts retain their historical aliases. Marlin parity leaves
    // both legacy halves empty so tensor-major storage cannot masquerade as
    // its canonical per-token GateThenUp publication.
    RequestMatrixRegion gate_bf16;                // [8192, 17408]
    RequestMatrixRegion up_bf16;                  // [8192, 17408]
    RequestMatrixRegion activated_bf16;           // [8192, 17408]
    // Dense normalized input initially occupies the future activated span and
    // must die after Gate/Up projection, before SiLU writes activation. This
    // is not a row-prefix alias.
    RequestMatrixRegion normalized_input_bf16;
    // Marlin projection parity aliases GateUp and Down to one sequentially
    // reused Ctmp region. Request-long locks use a stable physically disjoint
    // legacy owner. Other layouts retain their established semantics.
    RequestRegion gate_up_projection_temporary;
    RequestMatrixRegion branch_output_bf16;       // dead Up contiguous span
    RequestRegion down_projection_temporary;      // dead Gate contiguous span
};

struct LayerMajorLegacyC512Regions {
    std::array<RequestMatrixRegion, kRequestHiddenBufferCount> hidden_bf16;
    std::array<RequestMatrixRegion, kRequestProjectionBufferCount>
        projection_bf16;
    RequestMatrixRegion linear_a_bf16;
    RequestMatrixRegion linear_b_bf16;
    RequestRegion fp32_scratch;
    RequestRegion gqa_probability_scratch;  // aliases fp32_scratch prefix
};

struct LayerMajorP40WholeCoreLinearPhaseRegions {
    RequestMatrixRegion raw_qkv_bf16;             // [40000, 10240]
    RequestMatrixRegion conv_qkv_bf16;            // [40000, 10240]
    RequestMatrixRegion z_bf16;                   // [40000, 6144]
    RequestMatrixRegion a_bf16;                   // [40000, 48]
    RequestMatrixRegion b_bf16;                   // [40000, 48]
    RequestRegion prompt_wide_workspace;          // exact P40 GDN graph
    RequestMatrixRegion output_bf16;              // [40000, 6144]

    // These aliases have ordered lifetimes. Normalized input occupies the
    // future independent output span until every input projection consumes
    // it. Branch output reuses dead raw-QKV only after the GDN graph ends.
    RequestMatrixRegion normalized_input_bf16;    // [40000, 5120]
    RequestMatrixRegion branch_output_bf16;       // [40000, 5120]
};

struct LayerMajorP40WholeCoreFullAttentionPhaseRegions {
    RequestMatrixRegion raw_q_gate_bf16;          // [40000, 12288]
    RequestMatrixRegion processed_q_bf16;         // [40000, 6144]
    RequestMatrixRegion packed_gate_bf16;         // [40000, 6144]
    // K/V are the per-layer persistent key/value cache regions in `common`;
    // they are not duplicated in the transient whole-core family arena.
    RequestMatrixRegion normalized_input_bf16;    // [40000, 5120]
    RequestMatrixRegion core_output_bf16;         // [40000, 6144]
    RequestMatrixRegion branch_output_bf16;       // [40000, 5120]
};

struct LayerMajorP40WholeCoreRegions {
    std::uint32_t prompt_token_count = 0U;
    std::uint32_t request_capacity_tokens = 0U;
    std::uint32_t logical_panel_capacity_tokens = 0U;
    std::uint32_t logical_panel_count = 0U;
    // Owning region. No raw RequestState accessor is provided.
    RequestRegion family_phase_arena;
    // Full P40000 token staging inside prompt_wide_workspace. This region is
    // never at family offset zero and dies at embedding completion.
    RequestMatrixRegion prompt_token_ids_u32;
    LayerMajorP40WholeCoreLinearPhaseRegions linear;
    LayerMajorP40WholeCoreFullAttentionPhaseRegions full_attention;
};

// A separate profile is intentionally not reachable through
// build_request_memory_plan(). `common` preserves the shared state/KV/RoPE
// query contract without slicing this complete candidate plan into the
// legacy RequestMemoryPlan stored by RequestState.
struct LayerMajorRequestMemoryPlan {
    RequestMemoryPlan common;
    LayerMajorRequestLayout layout =
        LayerMajorRequestLayout::kC8192FamilyOverlay;
    std::uint32_t operator_panel_capacity_tokens =
        kLayerMajorRequestOperatorPanelCapacity;
    std::uint32_t legacy_prefill_chunk_size =
        kMaximumRequestPrefillChunkSize;
    std::uint32_t mlp_capacity_tokens =
        kLayerMajorRequestOperatorPanelCapacity;
    LayerMajorRequestMlpLayout mlp_layout =
        LayerMajorRequestMlpLayout::kPanelLocalThreeSpan;

    RequestMatrixRegion prompt_residual_bf16;
    // Owning capacity only. RequestState intentionally exposes no raw view.
    RequestRegion c8192_family_phase_arena;
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
    // Conservative request-arena reservation identity.  The sealed
    // DeploymentPlan/witness, rather than this field, proves whether a
    // candidate actually executes a separate or fused MLP kernel path.
    PrefillMlpPhysicalTactic mlp_tactic{};

    bool prompt_residual_in_place_contract_bound = false;
    bool family_completion_events_bound = false;
    bool intra_family_phase_contract_bound = false;
    bool prompt_token_ids_consumed_event_bound = false;
    // Also attests serialized or fused-pair Gate/Up ownership; the legacy
    // dual-stream projection schedule cannot consume the single subrange.
    bool projection_workspace_subrange_binding_bound = false;
    bool operator_bindings_complete = false;

    [[nodiscard]] constexpr bool executable() const noexcept {
        return prompt_residual_in_place_contract_bound &&
               family_completion_events_bound &&
               intra_family_phase_contract_bound &&
               prompt_token_ids_consumed_event_bound &&
               projection_workspace_subrange_binding_bound &&
               operator_bindings_complete;
    }
};

struct LayerMajorRequestPlanResult {
    std::optional<LayerMajorRequestMemoryPlan> value;
    RequestDiagnostic diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && diagnostic.code == RequestErrorCode::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Pure CPU, deterministic and allocation-free except for the optional result
// object itself. Every region begins at a checked 256-byte-aligned offset.
[[nodiscard]] RequestPlanResult build_request_memory_plan(
    const RequestMemoryOptions& options = {});

// Explicit architecture-candidate planner. It binds only the fixed
// single-residual / C8192-family-overlay + disjoint-legacy / exact C64
// in-place / Release C16 / separate-SiLU physical strategy. Runtime launch,
// event, alias, and operator bindings remain false, so the result is not an
// executable production route.
[[nodiscard]] LayerMajorRequestPlanResult
build_layer_major_request_memory_plan(
    const LayerMajorRequestMemoryOptions& options);

struct DeviceBufferView {
    void* device_data = nullptr;
    std::uint64_t arena_offset = 0U;
    std::uint64_t byte_size = 0U;
    std::uint64_t element_capacity = 0U;
    std::uint32_t element_size_bytes = 0U;
};

struct ConstDeviceBufferView {
    const void* device_data = nullptr;
    std::uint64_t arena_offset = 0U;
    std::uint64_t byte_size = 0U;
    std::uint64_t element_capacity = 0U;
    std::uint32_t element_size_bytes = 0U;
};

struct DeviceMatrixView {
    DeviceBufferView storage;
    std::uint32_t row_capacity = 0U;
    std::uint32_t columns = 0U;
    std::uint64_t row_stride_elements = 0U;
};

struct RequestViewResult {
    std::optional<DeviceBufferView> value;
    RequestAccessError error = RequestAccessError::kNone;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && error == RequestAccessError::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct RequestConstViewResult {
    std::optional<ConstDeviceBufferView> value;
    RequestAccessError error = RequestAccessError::kNone;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && error == RequestAccessError::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct RequestMatrixViewResult {
    std::optional<DeviceMatrixView> value;
    RequestAccessError error = RequestAccessError::kNone;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && error == RequestAccessError::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct LayerMajorGdnPhaseViews {
    DeviceMatrixView qkv_bf16;
    DeviceMatrixView z_bf16;
    DeviceMatrixView a_bf16;
    DeviceMatrixView b_bf16;
    DeviceMatrixView recurrent_core_bf16;
    DeviceBufferView native_c64_workspace;
    DeviceMatrixView normalized_input_bf16;
    DeviceBufferView input_projection_temporary;
    DeviceMatrixView branch_output_bf16;
    DeviceBufferView output_projection_temporary;
};

struct LayerMajorAttentionPhaseViews {
    DeviceMatrixView raw_q_gate_bf16;
    DeviceMatrixView processed_q_bf16;
    DeviceMatrixView packed_gate_bf16;
    DeviceMatrixView normalized_input_bf16;
    DeviceBufferView input_projection_temporary;
    DeviceMatrixView core_output_bf16;
    DeviceMatrixView branch_output_bf16;
    DeviceBufferView output_projection_temporary;
};

struct LayerMajorMlpPhaseViews {
    DeviceMatrixView merged_gate_up_bf16;
    DeviceMatrixView gate_bf16;
    DeviceMatrixView up_bf16;
    DeviceMatrixView activated_bf16;
    DeviceMatrixView normalized_input_bf16;
    DeviceBufferView gate_up_projection_temporary;
    DeviceMatrixView branch_output_bf16;
    DeviceBufferView down_projection_temporary;
};

struct LayerMajorLegacyC512Views {
    std::array<DeviceMatrixView, kRequestHiddenBufferCount> hidden_bf16;
    std::array<DeviceMatrixView, kRequestProjectionBufferCount>
        projection_bf16;
    DeviceMatrixView linear_a_bf16;
    DeviceMatrixView linear_b_bf16;
    DeviceBufferView fp32_scratch;
    DeviceBufferView gqa_probability_scratch;
};

struct LayerMajorP40WholeCoreLinearPhaseViews {
    DeviceMatrixView raw_qkv_bf16;
    DeviceMatrixView conv_qkv_bf16;
    DeviceMatrixView z_bf16;
    DeviceMatrixView a_bf16;
    DeviceMatrixView b_bf16;
    DeviceBufferView prompt_wide_workspace;
    DeviceMatrixView output_bf16;
    DeviceMatrixView normalized_input_bf16;
    DeviceMatrixView branch_output_bf16;
};

struct LayerMajorP40WholeCoreFullAttentionPhaseViews {
    DeviceMatrixView raw_q_gate_bf16;
    DeviceMatrixView processed_q_bf16;
    DeviceMatrixView packed_gate_bf16;
    DeviceMatrixView normalized_input_bf16;
    DeviceMatrixView core_output_bf16;
    DeviceMatrixView branch_output_bf16;
};

struct LayerMajorP40WholeCoreViews {
    DeviceMatrixView prompt_token_ids_u32;
    LayerMajorP40WholeCoreLinearPhaseViews linear;
    LayerMajorP40WholeCoreFullAttentionPhaseViews full_attention;
};

template <typename Views>
struct LayerMajorTypedViewResult {
    std::optional<Views> value;
    RequestAccessError error = RequestAccessError::kNone;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && error == RequestAccessError::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

using LayerMajorGdnPhaseViewResult =
    LayerMajorTypedViewResult<LayerMajorGdnPhaseViews>;
using LayerMajorAttentionPhaseViewResult =
    LayerMajorTypedViewResult<LayerMajorAttentionPhaseViews>;
using LayerMajorMlpPhaseViewResult =
    LayerMajorTypedViewResult<LayerMajorMlpPhaseViews>;
using LayerMajorLegacyC512ViewResult =
    LayerMajorTypedViewResult<LayerMajorLegacyC512Views>;
using LayerMajorP40WholeCoreViewResult =
    LayerMajorTypedViewResult<LayerMajorP40WholeCoreViews>;

struct RequestOperationStatus {
    RequestAccessError error = RequestAccessError::kNone;
    int cuda_error = 0;

    [[nodiscard]] bool ok() const noexcept {
        return error == RequestAccessError::kNone && cuda_error == 0;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct RequestStateResult;
class ReferenceRunner;
struct RequestStateResetCudaTestPeer;

class RequestState {
  public:
    RequestState() noexcept = default;
    ~RequestState();

    RequestState(const RequestState&) = delete;
    RequestState& operator=(const RequestState&) = delete;
    RequestState(RequestState&& other) noexcept;
    RequestState& operator=(RequestState&& other) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept {
        return arena_ != nullptr;
    }
    [[nodiscard]] void* arena_data() noexcept { return arena_; }
    [[nodiscard]] const void* arena_data() const noexcept { return arena_; }
    [[nodiscard]] std::uint64_t arena_bytes() const noexcept {
        return common_plan().arena_bytes;
    }
    // Common persistent/KV/RoPE and capacity metadata for either profile.
    // The complete candidate-only layout remains available through
    // layer_major_plan(); it is never sliced into this reference.
    [[nodiscard]] const RequestMemoryPlan& plan() const noexcept {
        return common_plan();
    }
    [[nodiscard]] RequestMemoryProfile memory_profile() const noexcept {
        return common_plan().profile;
    }
    [[nodiscard]] const LayerMajorRequestMemoryPlan* layer_major_plan()
        const noexcept {
        return layer_major_plan_ ? &*layer_major_plan_ : nullptr;
    }

    [[nodiscard]] std::uint32_t sequence_length() const noexcept {
        return sequence_length_;
    }
    [[nodiscard]] std::uint32_t current_position() const noexcept {
        return sequence_length_;
    }
    [[nodiscard]] std::uint32_t max_sequence_length() const noexcept {
        return common_plan().max_sequence_length;
    }
    [[nodiscard]] std::uint32_t remaining_capacity() const noexcept {
        return common_plan().max_sequence_length >= sequence_length_
                   ? common_plan().max_sequence_length - sequence_length_
                   : 0U;
    }
    [[nodiscard]] RequestOperationStatus commit_token() noexcept;
    [[nodiscard]] RequestOperationStatus set_sequence_length(
        std::uint32_t length) noexcept;
    [[nodiscard]] RequestOperationStatus publish_sequence_length(
        std::uint32_t expected_current,
        std::uint32_t desired) noexcept;

    [[nodiscard]] RequestViewResult conv_state(std::size_t layer_index) noexcept;
    [[nodiscard]] RequestViewResult gdn_state(std::size_t layer_index) noexcept;
    [[nodiscard]] RequestViewResult key_cache(std::size_t layer_index) noexcept;
    [[nodiscard]] RequestViewResult value_cache(std::size_t layer_index) noexcept;
    [[nodiscard]] RequestViewResult key_position(std::size_t layer_index,
                                                 std::size_t position) noexcept;
    [[nodiscard]] RequestViewResult value_position(std::size_t layer_index,
                                                   std::size_t position) noexcept;

    [[nodiscard]] RequestViewResult hidden_buffer(std::size_t index) noexcept;
    [[nodiscard]] RequestViewResult projection_buffer(std::size_t index) noexcept;
    [[nodiscard]] RequestViewResult linear_a_buffer() noexcept;
    [[nodiscard]] RequestViewResult linear_b_buffer() noexcept;
    [[nodiscard]] RequestViewResult fp32_scratch() noexcept;
    [[nodiscard]] RequestViewResult gqa_probability_scratch() noexcept;
    [[nodiscard]] RequestMatrixViewResult
    layer_major_prompt_residual() noexcept;
    [[nodiscard]] RequestMatrixViewResult
    layer_major_panel_token_ids() noexcept;
    [[nodiscard]] LayerMajorGdnPhaseViewResult
    layer_major_gdn_phase_views() noexcept;
    [[nodiscard]] LayerMajorAttentionPhaseViewResult
    layer_major_attention_phase_views() noexcept;
    [[nodiscard]] LayerMajorMlpPhaseViewResult
    layer_major_mlp_phase_views() noexcept;
    [[nodiscard]] LayerMajorLegacyC512ViewResult
    layer_major_legacy_c512_views() noexcept;
    [[nodiscard]] LayerMajorP40WholeCoreViewResult
    layer_major_p40_whole_core_views() noexcept;
    [[nodiscard]] RequestMatrixViewResult
    layer_major_final_hidden() noexcept;
    [[nodiscard]] RequestConstViewResult rope_cos(std::size_t position) const noexcept;
    [[nodiscard]] RequestConstViewResult rope_sin(std::size_t position) const noexcept;
    [[nodiscard]] RequestConstViewResult current_rope_cos() const noexcept;
    [[nodiscard]] RequestConstViewResult current_rope_sin() const noexcept;

    // Enqueues zeroing of all persistent conv/GDN/KV storage and immediately
    // resets host logical length. The caller must order subsequent use on the
    // same stream (or synchronize explicitly). Workspace and RoPE are untouched.
    [[nodiscard]] RequestOperationStatus reset_async(
        void* cuda_stream = nullptr) noexcept;

  private:
    friend RequestStateResult create_request_state(
        const RequestMemoryOptions&);
    friend RequestStateResult create_layer_major_request_state(
        const LayerMajorRequestMemoryOptions&);
    friend class ReferenceRunner;
    // Narrow correctness-test peer for exercising the lifecycle-selected
    // reset primitive against real CUDA storage. It is not a production
    // selector and cannot be reached through the public engine API.
    friend struct RequestStateResetCudaTestPeer;

    void release() noexcept;
    [[nodiscard]] DeviceBufferView mutable_view(
        const RequestRegion& region) noexcept;
    [[nodiscard]] ConstDeviceBufferView const_view(
        const RequestRegion& region) const noexcept;
    [[nodiscard]] DeviceMatrixView mutable_matrix_view(
        const RequestMatrixRegion& region) noexcept;
    [[nodiscard]] const RequestMemoryPlan& common_plan() const noexcept {
        return layer_major_plan_ ? layer_major_plan_->common : plan_;
    }
    [[nodiscard]] RequestOperationStatus reset_for_request_async(
        RequestStateResetMode mode,
        std::uint32_t committed_positions,
        void* cuda_stream,
        RequestStateResetReceipt& receipt) noexcept;

    void* arena_ = nullptr;
    RequestMemoryPlan plan_;
    std::uint32_t sequence_length_ = 0U;
    // Appended in ABI 0.5.0; legacy members retain their prior offsets.
    std::optional<LayerMajorRequestMemoryPlan> layer_major_plan_;
};

struct RequestStateResult {
    std::optional<RequestState> value;
    RequestDiagnostic diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && diagnostic.code == RequestErrorCode::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Creates one device arena, zeros the complete persistent span, generates the
// fixed theta=1e7 / rotary_dim=64 BF16-rounded-in-FP32 cos/sin cache, copies it
// to the arena, and synchronizes initialization before returning.
[[nodiscard]] RequestStateResult create_request_state(
    const RequestMemoryOptions& options = {});

// Creates the isolated layer-major allocation profile. This reserves memory
// and initializes common persistent/RoPE state, but does not bind any of the
// alias/event/operator contracts required by LayerMajorRequestMemoryPlan::
// executable().
[[nodiscard]] RequestStateResult create_layer_major_request_state(
    const LayerMajorRequestMemoryOptions& options);

[[nodiscard]] std::string_view to_string(RequestErrorCode code) noexcept;
[[nodiscard]] std::string_view to_string(RequestAccessError error) noexcept;
[[nodiscard]] std::string_view to_string(RequestStateResetMode mode) noexcept;

}  // namespace q3x::runtime
