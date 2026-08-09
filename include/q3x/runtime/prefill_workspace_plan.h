#pragma once

#include "q3x/kernels/gdn_prefill_chunk64_workspace_abi.h"
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

}  // namespace q3x::runtime
