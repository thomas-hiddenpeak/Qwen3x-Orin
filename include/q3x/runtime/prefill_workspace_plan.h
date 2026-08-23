#pragma once

#include "q3x/kernels/gdn_prefill_chunk64_workspace_abi.h"
#include "q3x/kernels/sm87_nvfp4_marlin_p40_parity.h"
#include "q3x/runtime/prefill_execution_plan.h"
#include "q3x/runtime/request_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace q3x::runtime {

// AC-PREFILL-LAYERMAJOR-8K-v1 memory requirements. These constants describe
// an unbound host plan only. They do not reserve device memory, authenticate
// a sidecar, bind a launcher, or make the architecture executable.
inline constexpr std::size_t kLayerMajorPrefillScratchFamilyCount = 3U;
inline constexpr std::size_t kLegacyC512HiddenScratchBufferCount = 3U;
inline constexpr std::size_t kLegacyC512ProjectionScratchBufferCount = 4U;
inline constexpr std::size_t kLegacyC512LinearScalarScratchBufferCount = 2U;
inline constexpr std::uint64_t kLayerMajorPrefillFp32MinimumScratchElements =
    262'144U;
// Exact capacity of the current test-only C64-native GDN workspace. A plan
// counts this only when its explicit physical tactic selects that mechanism;
// the Release C16 route does not own this allocation.
inline constexpr std::uint64_t
    kLayerMajorPrefillGdnC64NativeWorkspaceBytes =
        kernels::kGdnPrefillChunk64NativeWorkspaceBytes;

static_assert(kLayerMajorPrefillOperatorPanelTokens == 8'192U);
static_assert(kRequestArenaAlignment == 256U);

enum class PrefillMemoryOwner : std::uint8_t {
  kRequestStateArena = 0,
  kEngineResidentModel,
  kDeploymentPlanSidecar,
};

enum class PrefillMemoryLifetime : std::uint8_t {
  kRequestThroughDecode = 0,
  kRequestPrefill,
  kOperatorPanel,
  kEngine,
};

enum class PrefillMemoryAliasCondition : std::uint8_t {
  kDisjoint = 0,
  // One prompt-wide hidden allocation is legal only after every operator in a
  // panel proves that its input can be overwritten after the last input and
  // residual consumer has completed.
  kPanelwiseInputConsumedBeforeOutputOverwrite,
  // Natural layer order allows two operator-family live sets to share a raw
  // span only after the first family's named last-consumer event completes.
  kSequentialFamilyLiveSetOverlay,
  // C512 and C8192 views may share storage only when route selection and
  // completion events prove that the two routes cannot use it concurrently.
  kMutuallyExclusiveRouteOverlay,
  // Panel token IDs occupy the operator arena prefix only until embedding
  // gather completes; family or legacy scratch may cover it afterward.
  kPromptTokenIdsConsumedBeforeOperatorScratchReuse,
};

enum class PrefillMemoryCapacityVerdict : std::uint8_t {
  kFitsDeclaredLimit = 0,
  kExceedsDeclaredLimit,
  kIndeterminate,
};

enum class PrefillHiddenStrategy : std::uint8_t {
  kUnselected = 0,
  kSinglePromptWideConditional,
  kDoublePromptWideConservative,
};

enum class PrefillOperatorScratchStrategy : std::uint8_t {
  kUnselected = 0,
  // Natural layer order overlays family live sets after their named last
  // consumer. Selecting this also excludes a resident legacy C512 workspace.
  kC8192FamilyOverlayConditional,
  // The family-overlay C8192 span and C512 workspace additionally share raw
  // storage under a route-exclusive event.
  kOverlayLegacyC512MutuallyExclusive,
  // The three C8192 families share one sequential span, while the complete
  // legacy C512 route owns physically disjoint storage. This is the initial
  // RequestState allocation shape; it needs family events but no route alias.
  kC8192FamilyOverlayWithDisjointLegacyC512,
  // The C8192 GDN/Attention/MLP families and legacy C512 workspace are
  // disjoint. Phase-local reuse inside each explicitly selected physical
  // tactic remains a separately named, unbound contract.
  kDisjointAllFamiliesAndLegacyC512,
  // Exact-P40000 prompt-wide whole-core family arena plus one physically
  // disjoint C512 compatibility workspace. This is not a C8192 overlay.
  kP40WholeCorePromptWideWithDisjointLegacyC512,
};

enum class PrefillGdnPhysicalTactic : std::uint8_t {
  kUnselected = 0,
  // Exact C64-native recurrence reuses the projected QKV span as the causal
  // convolution output. The fixed C512 native workspace is reused serially
  // by every physical segment of one logical C8192 panel.
  kC64NativeInPlaceConv,
  // Exact C64-native recurrence keeps one independent C512 convolution output
  // while the full-panel projected QKV remains live.
  kC64NativeTokenParallelConv,
  // One exact P40000 prompt-wide chunk graph with independent raw,
  // convolution, gate/scalar, workspace, and output ranges.
  kP40PromptWideChunkGraph,
};

enum class PrefillLegacyGdnPhysicalTactic : std::uint8_t {
  kUnselected = 0,
  // Current Release/default exact C16 composite; no runner-lifetime native
  // C64 workspace exists in this configuration.
  kC16Composite,
  // Experimental/test-only C64-native legacy route. A future production use
  // must explicitly promote and bind its fixed workspace.
  kC64Native,
};

enum class PrefillMlpPhysicalTactic : std::uint8_t {
  kUnselected = 0,
  // Conservative three-span workspace identity.  It reserves independently
  // addressable Gate, Up, and activated matrices for the separate exact
  // SiLU/gate path.  A sealed experimental DeploymentPlan may prove a
  // narrower fused lifetime and reuse one dead span; this workspace value is
  // not, by itself, evidence of the kernels executed by that DeploymentPlan.
  kSeparateGateUpAndSilu,
  // The fused epilogue keeps N, merged Gate+Up, activated Down input, Marlin
  // reduction storage, and locks live together.
  kFusedGateUpEpilogue,
  // Test-only exact-P40000 layer-wide schedule. Attention/GDN and the
  // post-attention residual finish panelwise, then three prompt-wide BF16
  // spans hold merged Gate+Up, Up/Down input, and activated output for one
  // exact full-M MLP phase. This is a byte/shape requirement only; no
  // allocation, launcher, or production route is bound here.
  kLayerWideP40AlignedMarlin,
  // Target full-M ownership: normalized [M,5120] is consumed by one exact
  // persistent GateUp kernel whose fused SiLU epilogue publishes only
  // activated [M,17408]. Persistent Down then accumulates and publishes
  // directly into prompt residual. No merged Gate/Up or branch-output
  // materialization is reserved.
  kLayerWideP40PersistentFusedGateUp,
  // Default-off stock-vLLM-Marlin parity ownership for exact P40000. One
  // canonical token-major [M,34816] GateThenUp matrix and one independent
  // [M,17408] activated matrix are simultaneously live. Gate/Up and Down use
  // stock LegacyStripe ownership. Its M64 tail reuses one Ctmp region for
  // GateUp and Down; request-long locks use a physically disjoint stable
  // legacy owner because the family arena is overwritten by intervening
  // phases. The 39 M1024 segments do not consume Ctmp.
  // This identity selects no execution route.
  kLayerWideP40MarlinParityMergedGateUp,
};

enum class PrefillWorkspacePlanError : std::uint8_t {
  kNone = 0,
  kInvalidArgument,
  kArithmeticOverflow,
  kCapacityExceeded,
  kModelContractMismatch,
  kInvalidLayout,
};

struct PrefillMemoryRequirement {
  PrefillMemoryOwner owner = PrefillMemoryOwner::kRequestStateArena;
  PrefillMemoryLifetime lifetime = PrefillMemoryLifetime::kRequestPrefill;
  PrefillMemoryAliasCondition alias_condition =
      PrefillMemoryAliasCondition::kDisjoint;
  std::uint64_t required_bytes = 0U;
  std::uint64_t element_capacity = 0U;
  std::uint32_t element_size_bytes = 0U;

  // False for every plan produced by this host-only planner. The byte count
  // is a requirement, not proof that an allocation or alias contract exists.
  bool allocation_bound = false;
  bool alias_contract_bound = false;
};

// Exact request-workspace contract for the first P40 whole-core composition.
// This is deliberately separate from the C8192 family-overlay planner above:
// the new route owns one P40000 operator graph, five exact P8000 scheduling
// panels, and a P40001 request capacity (prompt plus one generated token).
// Keeping a distinct type prevents a caller from presenting the old C8192
// layout or its 8192/7712 panel geometry as this architecture.
inline constexpr std::uint32_t kLayerMajorP40WholeCorePromptTokens = 40'000U;
inline constexpr std::uint32_t
    kLayerMajorP40WholeCoreRequestCapacityTokens = 40'001U;
inline constexpr std::uint32_t kLayerMajorP40WholeCorePanelTokens = 8'000U;
inline constexpr std::uint32_t kLayerMajorP40WholeCorePanelCount = 5U;
inline constexpr std::uint64_t kLayerMajorP40WholeCoreFamilyArenaBytes =
    5'429'760'000U;
#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
inline constexpr std::uint64_t kLayerMajorP40WholeCoreArenaBytes =
    8'640'542'976U;
inline constexpr std::uint32_t
    kSelectorExactPersistentAttentionV1P40WholeCoreRequestCapacityTokens =
        40'016U;
inline constexpr std::uint64_t
    kSelectorExactPersistentAttentionV1P40WholeCoreArenaBytes =
        8'641'683'456U;

[[nodiscard]] constexpr bool is_layer_major_p40_whole_core_request_capacity(
    const std::uint64_t capacity_tokens) noexcept {
  if (capacity_tokens == kLayerMajorP40WholeCoreRequestCapacityTokens) {
    return true;
  }
  return capacity_tokens ==
         kSelectorExactPersistentAttentionV1P40WholeCoreRequestCapacityTokens;
}

[[nodiscard]] constexpr std::uint64_t
layer_major_p40_whole_core_arena_bytes(
    const std::uint64_t capacity_tokens) noexcept {
  if (capacity_tokens == kLayerMajorP40WholeCoreRequestCapacityTokens) {
    return kLayerMajorP40WholeCoreArenaBytes;
  }
  if (capacity_tokens ==
      kSelectorExactPersistentAttentionV1P40WholeCoreRequestCapacityTokens) {
    return kSelectorExactPersistentAttentionV1P40WholeCoreArenaBytes;
  }
  return 0U;
}
#endif

// Exact family-relative typed-view ledger for the default-off P40000
// stock-vLLM-Marlin parity MLP. The whole-core family arena is owned by the
// linear/GDN high-water; these ranges are sequential-family lifetime aliases,
// not additional capacity. GateThenUp is one canonical row-major
// [40000,34816] tensor: within physical token row r, Gate occupies columns
// [0,17408) and Up occupies [17408,34816). It must never be exposed as two
// tensor-major matrices. Activated is one independent [40000,17408] tensor.
// GateUp and Down execute sequentially and alias one stock Ctmp region. The
// request-long lock bytes use an external stable owner and are not part of
// this family span's live payload.
inline constexpr std::uint32_t
    kLayerMajorP40MarlinParityGateUpHalfFeatures = 17'408U;
inline constexpr std::uint32_t
    kLayerMajorP40MarlinParityMergedGateUpFeatures = 34'816U;
inline constexpr std::uint64_t
    kLayerMajorP40MarlinParityMergedGateUpRowStrideElements = 34'816U;
inline constexpr std::uint64_t
    kLayerMajorP40MarlinParityGateColumnOffsetElements = 0U;
inline constexpr std::uint64_t
    kLayerMajorP40MarlinParityUpColumnOffsetElements = 17'408U;
inline constexpr std::uint64_t
    kLayerMajorP40MarlinParityMergedGateUpRowStrideBytes =
        kLayerMajorP40MarlinParityMergedGateUpRowStrideElements *
        sizeof(std::uint16_t);
inline constexpr std::uint64_t
    kLayerMajorP40MarlinParityUpColumnOffsetBytes =
        kLayerMajorP40MarlinParityUpColumnOffsetElements *
        sizeof(std::uint16_t);
inline constexpr std::uint64_t
    kLayerMajorP40MarlinParityMergedGateUpOffset = 0U;
inline constexpr std::uint64_t
    kLayerMajorP40MarlinParityMergedGateUpBytes =
        static_cast<std::uint64_t>(kLayerMajorP40WholeCorePromptTokens) *
        kLayerMajorP40MarlinParityMergedGateUpRowStrideElements *
        sizeof(std::uint16_t);
inline constexpr std::uint64_t kLayerMajorP40MarlinParityActivatedOffset =
    kLayerMajorP40MarlinParityMergedGateUpBytes;
inline constexpr std::uint64_t kLayerMajorP40MarlinParityActivatedBytes =
    static_cast<std::uint64_t>(kLayerMajorP40WholeCorePromptTokens) *
    kLayerMajorP40MarlinParityGateUpHalfFeatures * sizeof(std::uint16_t);
inline constexpr std::uint64_t kLayerMajorP40MarlinParityTemporaryOffset =
    kLayerMajorP40MarlinParityActivatedOffset +
    kLayerMajorP40MarlinParityActivatedBytes;
inline constexpr std::uint64_t
    kLayerMajorP40MarlinParityReductionWorkspaceOffset =
        kLayerMajorP40MarlinParityTemporaryOffset;
inline constexpr std::uint64_t
    kLayerMajorP40MarlinParityReductionWorkspaceBytes =
        kernels::kSm87NvFp4MarlinP40ParityReductionBytes;
inline constexpr std::uint64_t kLayerMajorP40MarlinParityLockBytes =
    kernels::kSm87NvFp4MarlinP40ParityLockBytes;
inline constexpr std::uint64_t
    kLayerMajorP40MarlinParityTemporaryPayloadBytes =
        kLayerMajorP40MarlinParityReductionWorkspaceBytes;
// Preserve the established projection-temporary view extent. The aligned
// padding above the 1-MiB Ctmp is dead and carries no lock semantics.
inline constexpr std::uint64_t kLayerMajorP40MarlinParityTemporaryBytes =
    1'048'832U;
inline constexpr std::uint64_t kLayerMajorP40MarlinParityNormalizedOffset =
    4'938'240'000U;
inline constexpr std::uint64_t kLayerMajorP40MarlinParityNormalizedBytes =
    static_cast<std::uint64_t>(kLayerMajorP40WholeCorePromptTokens) * 5'120U *
    sizeof(std::uint16_t);

static_assert(kLayerMajorP40MarlinParityMergedGateUpFeatures ==
              2U * kLayerMajorP40MarlinParityGateUpHalfFeatures);
static_assert(kLayerMajorP40MarlinParityMergedGateUpRowStrideBytes ==
              69'632U);
static_assert(kLayerMajorP40MarlinParityUpColumnOffsetBytes == 34'816U);
static_assert(kLayerMajorP40MarlinParityMergedGateUpBytes ==
              2'785'280'000U);
static_assert(kLayerMajorP40MarlinParityActivatedOffset == 2'785'280'000U);
static_assert(kLayerMajorP40MarlinParityActivatedBytes == 1'392'640'000U);
static_assert(kLayerMajorP40MarlinParityTemporaryOffset == 4'177'920'000U);
static_assert(kLayerMajorP40MarlinParityReductionWorkspaceOffset ==
              kLayerMajorP40MarlinParityTemporaryOffset);
static_assert(kLayerMajorP40MarlinParityReductionWorkspaceBytes ==
              1'048'576U);
static_assert(kLayerMajorP40MarlinParityLockBytes == 64U);
static_assert(kLayerMajorP40MarlinParityTemporaryPayloadBytes ==
              1'048'576U);
static_assert(kLayerMajorP40MarlinParityTemporaryBytes == 1'048'832U);
static_assert(kLayerMajorP40MarlinParityNormalizedBytes == 409'600'000U);
static_assert(kLayerMajorP40MarlinParityTemporaryOffset +
                  kLayerMajorP40MarlinParityTemporaryBytes <=
              kLayerMajorP40MarlinParityNormalizedOffset);
static_assert(kLayerMajorP40MarlinParityNormalizedOffset +
                  kLayerMajorP40MarlinParityNormalizedBytes <=
              kLayerMajorP40WholeCoreFamilyArenaBytes);

struct LayerMajorP40WholeCoreFamilyRegionRequirement {
  PrefillMemoryRequirement memory;
  std::uint64_t family_relative_offset = 0U;
};

struct LayerMajorP40WholeCoreWorkspaceOptions {
  std::uint64_t prompt_token_count = kLayerMajorP40WholeCorePromptTokens;
  std::uint64_t request_sequence_capacity_tokens =
      kLayerMajorP40WholeCoreRequestCapacityTokens;
  std::uint64_t logical_panel_capacity_tokens =
      kLayerMajorP40WholeCorePanelTokens;
  std::uint64_t request_arena_limit_bytes = kMaximumRequestArenaBytes;
};

struct LayerMajorP40WholeCoreWorkspacePlan {
  std::uint32_t prompt_token_count = 0U;
  std::uint32_t request_sequence_capacity_tokens = 0U;
  std::uint32_t logical_panel_capacity_tokens = 0U;
  std::uint32_t logical_panel_count = 0U;

  PrefillMemoryRequirement persistent_and_kv;
  PrefillMemoryRequirement prompt_residual_bf16;

  // One sequential whole-core family arena.  The linear/GDN layout below is
  // the owning high-water. Full-Attention and MLP use only explicitly typed
  // lifetime aliases within this same allocation.
  PrefillMemoryRequirement whole_core_family_arena;
  LayerMajorP40WholeCoreFamilyRegionRequirement linear_raw_qkv_bf16;
  LayerMajorP40WholeCoreFamilyRegionRequirement linear_conv_qkv_bf16;
  LayerMajorP40WholeCoreFamilyRegionRequirement linear_z_bf16;
  LayerMajorP40WholeCoreFamilyRegionRequirement linear_a_bf16;
  LayerMajorP40WholeCoreFamilyRegionRequirement linear_b_bf16;
  LayerMajorP40WholeCoreFamilyRegionRequirement
      linear_prompt_wide_workspace;
  LayerMajorP40WholeCoreFamilyRegionRequirement linear_output_bf16;

  // Prompt IDs live in a pre-GDN lifetime at the beginning of the prompt-wide
  // GDN workspace. They never cover raw-QKV offset zero; embedding completion
  // is the required reuse boundary.
  LayerMajorP40WholeCoreFamilyRegionRequirement prompt_token_ids_u32;

  PrefillMemoryRequirement legacy_c512_workspace;
  PrefillMemoryRequirement final_hidden_handoff_bf16;
  PrefillMemoryRequirement rope_cos_sin_fp32;
  std::uint64_t required_bytes = 0U;
  std::uint64_t request_arena_limit_bytes = 0U;
  PrefillMemoryCapacityVerdict capacity =
      PrefillMemoryCapacityVerdict::kIndeterminate;

  // This pure host calculation never proves allocation, event/lifetime
  // aliases, operator binding, or whole-process residency.
  bool request_arena_reservation_bound = false;
  bool lifetime_alias_contracts_bound = false;
  bool operator_bindings_complete = false;

  [[nodiscard]] constexpr bool executable() const noexcept {
    return request_arena_reservation_bound &&
           lifetime_alias_contracts_bound && operator_bindings_complete &&
           capacity == PrefillMemoryCapacityVerdict::kFitsDeclaredLimit;
  }
};

struct LayerMajorP40WholeCoreWorkspacePlanResult {
  std::optional<LayerMajorP40WholeCoreWorkspacePlan> value;
  PrefillWorkspacePlanError error = PrefillWorkspacePlanError::kNone;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && error == PrefillWorkspacePlanError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct PrefillPersistentStateMemoryPlan {
  PrefillMemoryRequirement convolution_state;
  PrefillMemoryRequirement gdn_recurrent_state;
  PrefillMemoryRequirement full_attention_key_cache;
  PrefillMemoryRequirement full_attention_value_cache;
  std::uint64_t total_required_bytes = 0U;
};

struct PrefillPositionMemoryPlan {
  PrefillMemoryRequirement rope_cos_fp32;
  PrefillMemoryRequirement rope_sin_fp32;
  std::uint64_t total_required_bytes = 0U;
};

struct PrefillPromptWideHiddenVariant {
  PrefillHiddenStrategy strategy = PrefillHiddenStrategy::kUnselected;
  PrefillMemoryRequirement aggregate_bf16;
  std::uint32_t buffer_count = 0U;
  std::uint32_t hidden_width = 0U;
  std::uint32_t sequence_capacity_tokens = 0U;
  bool requires_panelwise_in_place_contract = false;
};

struct PrefillPromptWideHiddenMemoryPlan {
  // A conditional lower requirement. It is not semantically selectable until
  // every layer/operator binds the panelwise in-place alias contract.
  PrefillPromptWideHiddenVariant minimum_conditional;
  // A dependency-independent memory requirement with distinct layer input and
  // output buffers. Events are still unbound in this host plan.
  PrefillPromptWideHiddenVariant conservative;
  PrefillPromptWideHiddenVariant selected;
};

enum class PrefillProjectionRole : std::uint8_t {
  kGateUp = 0,
  kDown,
};

// N is the output width of one logical matrix. Gate/Up has two independently
// authenticated matrices with the same M/N/K shape; Down is a different
// asymmetric role and cannot inherit its tactic from Gate/Up.
struct PrefillProjectionShapeRequirement {
  PrefillProjectionRole role = PrefillProjectionRole::kGateUp;
  std::uint32_t maximum_m = 0U;
  std::uint32_t n = 0U;
  std::uint32_t k = 0U;
  std::uint32_t logical_matrix_count = 0U;
  bool tactic_bound = false;
};

enum class PrefillScratchFamily : std::uint8_t {
  kGdnProjectionAndRecurrentCore = 0,
  kFullAttentionProjectionAndCore,
  kMlpGateUpAndDown,
};

enum class PrefillScratchProducer : std::uint8_t {
  kGdnInProjQkvzAndInProjBa = 0,
  kFullAttentionQkvGateProjection,
  kMlpMergedGateUpProjection,
};

enum class PrefillScratchLastConsumer : std::uint8_t {
  kGdnRecurrentUpdateNormAndOutputProjection = 0,
  kFullAttentionCoreGateAndOutputProjection,
  kMlpSiluGateAndDownProjection,
};

// A live set is the storage simultaneously needed from one real producer to
// its last consumer. Different family live sets may overlay only after a
// future event contract proves that producer-consumer interval complete.
struct PrefillOperatorFamilyLiveSet {
  PrefillScratchFamily family =
      PrefillScratchFamily::kGdnProjectionAndRecurrentCore;
  PrefillScratchProducer producer =
      PrefillScratchProducer::kGdnInProjQkvzAndInProjBa;
  PrefillScratchLastConsumer last_consumer =
      PrefillScratchLastConsumer::kGdnRecurrentUpdateNormAndOutputProjection;
  PrefillMemoryRequirement aggregate;
};

enum class PrefillWorkspaceBackingIdentity : std::uint8_t {
  kNone = 0,
  // One physical projection workspace is reused by the sequential C8192
  // family arena after the named family-completion events.
  kC8192SequentialFamilyPhaseArena,
  // Each disjoint C8192 family arena owns its own embedded projection view.
  kC8192DisjointFamilyPhaseArenas,
  kOperatorArenaPromptTokenPrefix,
  kC8192GdnNativeArena,
  kLegacyC512GdnNativeArena,
  // C8192 and legacy C512 routes reuse one native workspace only after the
  // route mutual-exclusion event is bound.
  kMutuallyExclusiveC8192LegacyGdnNativeArena,
  // C8192 and legacy C512 routes retain two physically distinct workspaces.
  kDisjointC8192LegacyGdnNativeArenas,
};

enum class PrefillWorkspacePhaseOwnership : std::uint8_t {
  kNone = 0,
  kC8192ProjectionFamilies,
  kPromptTokenIdsBeforeOperatorFamilies,
  kC8192GdnRecurrentPhase,
  kLegacyC512GdnRoute,
  kMutuallyExclusiveC8192LegacyGdnRoutes,
  kDisjointC8192LegacyGdnRoutes,
};

// Machine-readable physical backing requirement. `minimum_instance_count`
// describes distinct backing instances required by this variant; it is not a
// launch count. The bytes are already included in the variant high-water, but
// the concrete subrange remains unbound in every host-only plan.
struct PrefillWorkspaceBackingContract {
  PrefillWorkspaceBackingIdentity identity =
      PrefillWorkspaceBackingIdentity::kNone;
  PrefillWorkspacePhaseOwnership phase_ownership =
      PrefillWorkspacePhaseOwnership::kNone;
  std::uint64_t bytes_per_instance = 0U;
  std::uint32_t minimum_instance_count = 0U;
  bool capacity_included_in_variant_total = false;
  bool subrange_binding_required = false;
};

struct PrefillOperatorScratchVariant {
  PrefillOperatorScratchStrategy strategy =
      PrefillOperatorScratchStrategy::kUnselected;
  PrefillMemoryRequirement aggregate;
  std::uint64_t total_required_bytes = 0U;
  bool requires_legacy_route_exclusion = false;
  bool requires_family_completion_events = false;
  bool requires_route_mutual_exclusion_event = false;
  bool requires_intra_family_phase_contract = false;
  bool requires_prompt_token_ids_consumed_event = false;
  PrefillWorkspaceBackingContract prompt_token_ids_backing;
  PrefillWorkspaceBackingContract projection_workspace_backing;
  PrefillWorkspaceBackingContract gdn_native_workspace_backing;
};

struct PrefillOperatorScratchMemoryPlan {
  std::uint32_t c8192_panel_capacity_tokens = 0U;
  std::uint32_t legacy_c512_panel_capacity_tokens = 0U;
  // MLP scratch is normally panel-capacity. The isolated P40 candidate owns
  // the entire prompt in one layer-wide phase and therefore records M40000.
  std::uint32_t mlp_capacity_tokens = 0U;
  PrefillGdnPhysicalTactic gdn_tactic =
      PrefillGdnPhysicalTactic::kUnselected;
  PrefillLegacyGdnPhysicalTactic legacy_gdn_tactic =
      PrefillLegacyGdnPhysicalTactic::kUnselected;
  PrefillMlpPhysicalTactic mlp_tactic =
      PrefillMlpPhysicalTactic::kUnselected;

  // Phase details are non-additive named evidence for each family high-water.
  // The aggregate family requirement is the maximum of its phases, not their
  // sum. All aliases remain unbound until a DeploymentPlan authenticates the
  // selected tactic, workspace partition, launcher, stream, and event.
  PrefillMemoryRequirement gdn_projection_phase;
  PrefillMemoryRequirement prompt_token_ids_u32;
  PrefillMemoryRequirement gdn_recurrent_phase;
  PrefillMemoryRequirement gdn_c64_native_workspace;
  std::optional<PrefillMemoryRequirement>
      gdn_token_parallel_c512_conv_output;
  PrefillMemoryRequirement full_attention_preprocess_phase;
  PrefillMemoryRequirement mlp_gate_up_down_phase;
  PrefillMemoryRequirement shared_projection_reduction_and_locks;
  std::array<PrefillOperatorFamilyLiveSet,
             kLayerMajorPrefillScratchFamilyCount>
      c8192_family_live_sets{};
  PrefillOperatorScratchVariant c8192_family_overlay_conditional;
  PrefillOperatorScratchVariant c8192_disjoint_families;
  PrefillMemoryRequirement legacy_c512_hidden_bf16;
  PrefillMemoryRequirement legacy_c512_projection_bf16;
  PrefillMemoryRequirement legacy_c512_linear_scalar_bf16;
  PrefillMemoryRequirement legacy_c512_fp32_scratch;
  std::optional<PrefillMemoryRequirement> legacy_c512_gdn_native_workspace;
  PrefillOperatorScratchVariant legacy_c512_only;
  PrefillOperatorScratchVariant mutually_exclusive_overlay;
  PrefillOperatorScratchVariant
      c8192_family_overlay_with_disjoint_legacy_c512;
  PrefillOperatorScratchVariant disjoint_conservative;
  PrefillOperatorScratchVariant selected;
  PrefillProjectionShapeRequirement gate_up;
  PrefillProjectionShapeRequirement down;
};

struct PrefillArenaCapacityProfile {
  PrefillHiddenStrategy hidden_strategy = PrefillHiddenStrategy::kUnselected;
  PrefillOperatorScratchStrategy scratch_strategy =
      PrefillOperatorScratchStrategy::kUnselected;
  std::uint64_t required_bytes = 0U;
  PrefillMemoryCapacityVerdict capacity =
      PrefillMemoryCapacityVerdict::kIndeterminate;
  bool requires_unbound_alias_or_route_contract = false;
};

struct PrefillEngineAssetMemoryRequirement {
  PrefillMemoryOwner owner = PrefillMemoryOwner::kEngineResidentModel;
  PrefillMemoryLifetime lifetime = PrefillMemoryLifetime::kEngine;
  std::optional<std::uint64_t> required_bytes;
  bool authenticated = false;
  bool residency_bound = false;
};

struct LayerMajorPrefillWorkspaceOptions {
  std::uint64_t sequence_capacity_tokens = 0U;
  std::uint64_t request_arena_limit_bytes = kMaximumRequestArenaBytes;
  PrefillHiddenStrategy hidden_strategy = PrefillHiddenStrategy::kUnselected;
  PrefillOperatorScratchStrategy scratch_strategy =
      PrefillOperatorScratchStrategy::kUnselected;
  PrefillGdnPhysicalTactic gdn_tactic =
      PrefillGdnPhysicalTactic::kUnselected;
  PrefillLegacyGdnPhysicalTactic legacy_gdn_tactic =
      PrefillLegacyGdnPhysicalTactic::kUnselected;
  PrefillMlpPhysicalTactic mlp_tactic =
      PrefillMlpPhysicalTactic::kUnselected;
};

struct LayerMajorPrefillWorkspacePlan {
  std::uint32_t sequence_capacity_tokens = 0U;
  std::uint32_t operator_panel_capacity_tokens =
      kLayerMajorPrefillOperatorPanelTokens;
  std::uint64_t request_arena_limit_bytes = 0U;

  PrefillPersistentStateMemoryPlan persistent_state;
  PrefillPositionMemoryPlan position_state;
  // Stable one-token BF16 handoff from the final layer into sampling/decode.
  // It is physically independent of prompt-wide and legacy hidden buffers.
  PrefillMemoryRequirement final_hidden_handoff_bf16;
  PrefillPromptWideHiddenMemoryPlan prompt_wide_hidden;
  PrefillOperatorScratchMemoryPlan operator_scratch;

  PrefillMemoryOwner request_arena_owner =
      PrefillMemoryOwner::kRequestStateArena;
  // Every profile is conditional on the explicitly selected physical tactics.
  // minimum_conditional overlays operator families; conservative separates
  // the three C8192 families and legacy route but still names phase-local
  // tactic reuse. selected is exactly the caller's explicit strategy pair.
  PrefillArenaCapacityProfile minimum_conditional;
  PrefillArenaCapacityProfile selected;
  PrefillArenaCapacityProfile conservative;

  // Model and sidecar sizes are deliberately absent until a later
  // authenticated DeploymentPlan binds exact artifacts. Consequently this
  // host plan cannot make a whole-process/device-fit statement.
  PrefillEngineAssetMemoryRequirement resident_model;
  PrefillEngineAssetMemoryRequirement derived_sidecars;
  std::optional<std::uint64_t> whole_process_required_bytes;
  PrefillMemoryCapacityVerdict whole_process_capacity =
      PrefillMemoryCapacityVerdict::kIndeterminate;

  bool request_arena_reservation_bound = false;
  bool operator_bindings_complete = false;

  [[nodiscard]] constexpr bool executable() const noexcept {
    return request_arena_reservation_bound && operator_bindings_complete &&
           !selected.requires_unbound_alias_or_route_contract &&
           whole_process_required_bytes.has_value() &&
           whole_process_capacity ==
               PrefillMemoryCapacityVerdict::kFitsDeclaredLimit;
  }
};

struct LayerMajorPrefillWorkspacePlanResult {
  std::optional<LayerMajorPrefillWorkspacePlan> value;
  PrefillWorkspacePlanError error = PrefillWorkspacePlanError::kNone;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && error == PrefillWorkspacePlanError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Computes byte requirements for an explicit hidden/scratch strategy pair.
// It does not allocate, inspect free device memory, load or size checkpoint
// assets, authenticate alias conditions, or connect to the engine/runner.
// Unselected strategies fail closed. Exceeding request_arena_limit_bytes is a
// valid plan with an explicit capacity verdict so later admission design can
// distinguish arithmetic from policy.
[[nodiscard]] LayerMajorPrefillWorkspacePlanResult
build_unbound_layer_major_prefill_workspace_plan(
    const LayerMajorPrefillWorkspaceOptions& options) noexcept;

// Computes the exact P40000/P40001 whole-core request arena described above.
// Any other prompt, request capacity, or panel geometry is rejected.  Like
// the generic planner, this function has byte authority only: it performs no
// allocation and binds no alias/event/operator contract.
[[nodiscard]] LayerMajorP40WholeCoreWorkspacePlanResult
build_unbound_layer_major_p40_whole_core_workspace_plan(
    const LayerMajorP40WholeCoreWorkspaceOptions& options = {}) noexcept;

}  // namespace q3x::runtime
