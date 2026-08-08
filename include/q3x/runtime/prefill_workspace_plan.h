#pragma once

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
inline constexpr std::size_t kLayerMajorPrefillScratchFamilyCount = 4U;
inline constexpr std::size_t kLegacyC512HiddenScratchBufferCount = 3U;
inline constexpr std::size_t kLegacyC512ProjectionScratchBufferCount = 4U;
inline constexpr std::size_t kLegacyC512LinearScalarScratchBufferCount = 2U;
inline constexpr std::uint64_t kLayerMajorPrefillFp32MinimumScratchElements =
    262'144U;

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
  // Every C8192 family and the legacy C512 workspace are disjoint. This is the
  // conservative coexistence requirement and makes no alias assumption.
  kDisjointAllFamiliesAndLegacyC512,
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
  kGdnMergedInputProjection = 0,
  kGdnFusedPostConvPrep,
  kFullAttentionProjectionAndCore,
  kMlpMergedGateUp,
};

enum class PrefillScratchProducer : std::uint8_t {
  kGdnInProjQkvzAndInProjBa = 0,
  kGdnFusedPostConvPrepQkvGBeta,
  kFullAttentionQkvGateProjection,
  kMlpMergedGateUpProjection,
};

enum class PrefillScratchLastConsumer : std::uint8_t {
  kGdnFusedPostConvPrep = 0,
  kGdnRecurrentUpdateNormAndOutputProjection,
  kFullAttentionCoreGateAndOutputProjection,
  kMlpSiluGateAndDownProjection,
};

// A live set is the storage simultaneously needed from one real producer to
// its last consumer. Different family live sets may overlay only after a
// future event contract proves that producer-consumer interval complete.
struct PrefillOperatorFamilyLiveSet {
  PrefillScratchFamily family = PrefillScratchFamily::kGdnMergedInputProjection;
  PrefillScratchProducer producer =
      PrefillScratchProducer::kGdnInProjQkvzAndInProjBa;
  PrefillScratchLastConsumer last_consumer =
      PrefillScratchLastConsumer::kGdnFusedPostConvPrep;
  PrefillMemoryRequirement aggregate;
};

struct PrefillOperatorScratchVariant {
  PrefillOperatorScratchStrategy strategy =
      PrefillOperatorScratchStrategy::kUnselected;
  PrefillMemoryRequirement aggregate;
  std::uint64_t total_required_bytes = 0U;
  bool requires_legacy_route_exclusion = false;
  bool requires_family_completion_events = false;
  bool requires_route_mutual_exclusion_event = false;
};

struct PrefillOperatorScratchMemoryPlan {
  std::uint32_t c8192_panel_capacity_tokens = 0U;
  std::uint32_t legacy_c512_panel_capacity_tokens = 0U;
  std::array<PrefillOperatorFamilyLiveSet,
             kLayerMajorPrefillScratchFamilyCount>
      c8192_family_live_sets{};
  PrefillOperatorScratchVariant c8192_family_overlay_conditional;
  PrefillOperatorScratchVariant c8192_disjoint_families;
  PrefillMemoryRequirement legacy_c512_hidden_bf16;
  PrefillMemoryRequirement legacy_c512_projection_bf16;
  PrefillMemoryRequirement legacy_c512_linear_scalar_bf16;
  PrefillMemoryRequirement legacy_c512_fp32_scratch;
  PrefillOperatorScratchVariant legacy_c512_only;
  PrefillOperatorScratchVariant mutually_exclusive_overlay;
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
};

struct LayerMajorPrefillWorkspacePlan {
  std::uint32_t sequence_capacity_tokens = 0U;
  std::uint32_t operator_panel_capacity_tokens =
      kLayerMajorPrefillOperatorPanelTokens;
  std::uint64_t request_arena_limit_bytes = 0U;

  PrefillPersistentStateMemoryPlan persistent_state;
  PrefillPositionMemoryPlan position_state;
  PrefillPromptWideHiddenMemoryPlan prompt_wide_hidden;
  PrefillOperatorScratchMemoryPlan operator_scratch;

  PrefillMemoryOwner request_arena_owner =
      PrefillMemoryOwner::kRequestStateArena;
  // minimum_conditional is not a safe default; conservative is an upper
  // coexistence requirement; selected is exactly the caller's explicit pair.
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
