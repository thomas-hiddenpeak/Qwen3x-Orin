#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace q3x::runtime {

// Host-only contract for the default-off MacroFeed-v4 P40000 panel-major
// wavefront.  It carries no CUDA handle, device pointer, launcher, selector,
// or production-dispatch authority.  The future executor must bind this exact
// contract through an authenticated AOT DeploymentPlan before it may issue an
// API witness.
inline constexpr std::array<std::uint8_t, 8U>
    kSm87MacroFeedV4PanelWavefrontMagic{{'Q', '3', 'X', 'M', 'F', '4', 'P',
                                         '1'}};
inline constexpr std::uint16_t kSm87MacroFeedV4PanelWavefrontAbiMajor = 1U;
inline constexpr std::uint16_t kSm87MacroFeedV4PanelWavefrontAbiMinor = 0U;

inline constexpr std::string_view kSm87MacroFeedV4CandidateId =
    "AC-PREFILL-SM87-MACROFEED-v4";
inline constexpr std::string_view kSm87MacroFeedV4P40DeploymentPlanId =
    "q3x.sm87.ac-prefill-sm87-macrofeed-v4."
    "native-p40-c8000x5-panel-wavefront.v1";
inline constexpr std::string_view kSm87MacroFeedV4P40ApiRouteId =
    "q3x.sm87.ac-prefill-sm87-macrofeed-v4."
    "openai-p40000-cold-no-cache-one-token.v1";
inline constexpr std::string_view kSm87MacroFeedV4P40Endpoint =
    "/v1/completions";
inline constexpr std::string_view kSm87MacroFeedV4P40Model =
    "qwen3.6-27b-nvfp4";

inline constexpr std::size_t kSm87MacroFeedV4P40Tokens = 40'000U;
inline constexpr std::size_t kSm87MacroFeedV4PanelTokens = 8'000U;
inline constexpr std::size_t kSm87MacroFeedV4PanelCount = 5U;
inline constexpr std::size_t kSm87MacroFeedV4LayerCount = 64U;
inline constexpr std::size_t kSm87MacroFeedV4GdnLayerCount = 48U;
inline constexpr std::size_t kSm87MacroFeedV4FullAttentionLayerCount = 16U;
inline constexpr std::size_t kSm87MacroFeedV4Hidden = 5'120U;
inline constexpr std::size_t kSm87MacroFeedV4Intermediate = 17'408U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionQuery = 6'144U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionRawQGate = 12'288U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionLegacyLiveColumns =
    kSm87MacroFeedV4AttentionRawQGate +
    2U * kSm87MacroFeedV4AttentionQuery;
inline constexpr std::size_t kSm87MacroFeedV4GdnQkv = 10'240U;
inline constexpr std::size_t kSm87MacroFeedV4GdnZ = 6'144U;
inline constexpr std::size_t kSm87MacroFeedV4GdnScalarPair = 96U;
inline constexpr std::size_t kSm87MacroFeedV4GdnAliasedLiveColumns =
    kSm87MacroFeedV4GdnQkv + kSm87MacroFeedV4GdnZ +
    kSm87MacroFeedV4GdnScalarPair;
inline constexpr std::size_t kSm87MacroFeedV4Bf16Bytes = 2U;
inline constexpr std::size_t kSm87MacroFeedV4LayerStepCount =
    kSm87MacroFeedV4PanelCount * kSm87MacroFeedV4LayerCount;

inline constexpr std::uint64_t kSm87MacroFeedV4HiddenPanelBytes =
    static_cast<std::uint64_t>(kSm87MacroFeedV4PanelTokens) *
    kSm87MacroFeedV4Hidden * kSm87MacroFeedV4Bf16Bytes;
inline constexpr std::uint64_t kSm87MacroFeedV4PanelScratchBytes =
    static_cast<std::uint64_t>(kSm87MacroFeedV4PanelTokens) *
    kSm87MacroFeedV4Intermediate * kSm87MacroFeedV4Bf16Bytes;
inline constexpr std::uint64_t kSm87MacroFeedV4TransientArenaBytes =
    2U * kSm87MacroFeedV4HiddenPanelBytes +
    kSm87MacroFeedV4PanelScratchBytes;
inline constexpr std::uint64_t kSm87MacroFeedV4ConvEpochBytes =
    2'949'120U;
inline constexpr std::uint64_t kSm87MacroFeedV4GdnEpochBytes =
    75'497'472U;
inline constexpr std::uint64_t kSm87MacroFeedV4RecurrentEpochBytes =
    kSm87MacroFeedV4ConvEpochBytes + kSm87MacroFeedV4GdnEpochBytes;
inline constexpr std::uint64_t kSm87MacroFeedV4RecurrentStorageBytes =
    2U * kSm87MacroFeedV4RecurrentEpochBytes;

static_assert(kSm87MacroFeedV4P40Tokens ==
              kSm87MacroFeedV4PanelCount * kSm87MacroFeedV4PanelTokens);
static_assert(kSm87MacroFeedV4GdnLayerCount +
                  kSm87MacroFeedV4FullAttentionLayerCount ==
              kSm87MacroFeedV4LayerCount);
static_assert(kSm87MacroFeedV4Intermediate <
              kSm87MacroFeedV4AttentionLegacyLiveColumns);
static_assert(kSm87MacroFeedV4Intermediate >=
              kSm87MacroFeedV4AttentionRawQGate);
static_assert(kSm87MacroFeedV4Intermediate >=
              kSm87MacroFeedV4GdnAliasedLiveColumns);
static_assert(kSm87MacroFeedV4RecurrentEpochBytes == 78'446'592U);
static_assert(kSm87MacroFeedV4RecurrentStorageBytes == 156'893'184U);

enum class Sm87MacroFeedV4LayerKind : std::uint8_t {
  kInvalid = 0U,
  kGdn,
  kFullAttention,
};

[[nodiscard]] constexpr Sm87MacroFeedV4LayerKind
sm87_macrofeed_v4_expected_layer_kind(const std::size_t layer) noexcept {
  if (layer >= kSm87MacroFeedV4LayerCount) {
    return Sm87MacroFeedV4LayerKind::kInvalid;
  }
  return ((layer + 1U) % 4U) == 0U
             ? Sm87MacroFeedV4LayerKind::kFullAttention
             : Sm87MacroFeedV4LayerKind::kGdn;
}

enum class Sm87MacroFeedV4WorkspaceRole : std::uint8_t {
  kInvalid = 0U,
  kPingHidden,
  kPongHidden,
  kPanelScratch,
};

enum class Sm87MacroFeedV4Traversal : std::uint8_t {
  kInvalid = 0U,
  kPanelMajorLayerWavefront,
};

enum class Sm87MacroFeedV4StateWriteMode : std::uint8_t {
  kInvalid = 0U,
  kPrivatePanelStage,
};

struct Sm87MacroFeedV4P40ApiIdentity final {
  std::string_view route_id{};
  std::string_view endpoint{};
  std::string_view served_model{};
  std::size_t prompt_tokens = 0U;
  std::size_t maximum_output_tokens = 0U;
  std::size_t batch_size = 0U;
  bool openai_compatible = false;
  bool exact_token_ids = false;
  bool cold_request = false;
  bool prefix_cache_disabled = false;
  bool kv_reuse_disabled = false;
  bool streaming_first_committed_token = false;
  bool full_prompt_consumption_required = false;
};

struct Sm87MacroFeedV4TransientBuffer final {
  Sm87MacroFeedV4WorkspaceRole role =
      Sm87MacroFeedV4WorkspaceRole::kInvalid;
  std::uint64_t storage_identity = 0U;
  std::uint64_t offset = 0U;
  std::uint64_t bytes = 0U;
  std::size_t token_capacity = 0U;
  std::size_t row_width = 0U;
  bool panel_local = false;
  bool reuse_waits_for_completion = false;

  [[nodiscard]] constexpr std::uint64_t end() const noexcept {
    return offset + bytes;
  }
};

struct Sm87MacroFeedV4WorkspacePlan final {
  std::array<Sm87MacroFeedV4TransientBuffer, 3U> buffers{};
  std::uint64_t transient_arena_bytes = 0U;
  std::size_t maximum_temporary_tokens = 0U;
  bool ping_pong_hidden = false;
  bool scratch_reused_by_phase = false;
  bool full_p40_temporary_plane_allowed = true;
  bool persistent_kv_is_outside_transient_arena = false;
  bool persistent_conv_gdn_state_is_outside_transient_arena = false;
};

// The single C8000 scratch plane is intentionally smaller than the incumbent
// Full-Attention live set.  These aliases are therefore required execution
// semantics, not optional memory optimizations.  An executor that materializes
// raw Q+gate, processed Q, and packed gate concurrently is not V4.
struct Sm87MacroFeedV4PhaseAliasingPlan final {
  bool attention_q_preprocess_overwrites_raw_q_gate = false;
  bool attention_online_core_reuses_processed_q = false;
  bool gdn_recurrent_reuses_consumed_qkv = false;
  bool gate_up_activation_owns_panel_scratch = false;
  bool every_phase_fits_one_panel_scratch = false;
};

// Recurrent state cannot be updated in the canonical Decode-visible bank while
// a panel is fallible.  V4 owns an active/candidate epoch pair and a private KV
// valid-end fence; canonical publication is the final request operation.
struct Sm87MacroFeedV4StateOwnershipPlan final {
  std::size_t recurrent_epoch_bank_count = 0U;
  std::uint64_t recurrent_epoch_bytes = 0U;
  std::uint64_t recurrent_storage_bytes = 0U;
  std::uint64_t active_recurrent_storage_identity = 0U;
  std::uint64_t candidate_recurrent_storage_identity = 0U;
  std::uint64_t private_kv_valid_end_storage_identity = 0U;
  std::uint64_t panel_commit_event_identity = 0U;
  std::uint64_t final_publish_event_identity = 0U;
  bool private_kv_valid_end = false;
  bool candidate_epoch_copies_active_before_panel = false;
  bool active_candidate_swap_after_layer_63 = false;
  bool panel_failure_discards_candidate_epoch = false;
  bool canonical_recurrent_publish_after_final_panel = false;
  bool sequence_length_is_final_visibility_fence = false;
  bool no_fallible_work_after_sequence_publication = false;
};

struct Sm87MacroFeedV4LayerStep final {
  std::size_t panel_index = kSm87MacroFeedV4PanelCount;
  std::size_t layer_index = kSm87MacroFeedV4LayerCount;
  std::size_t sequence_ordinal = kSm87MacroFeedV4LayerStepCount;
  std::size_t token_begin = kSm87MacroFeedV4P40Tokens;
  std::size_t token_count = 0U;
  Sm87MacroFeedV4LayerKind layer_kind =
      Sm87MacroFeedV4LayerKind::kInvalid;
  Sm87MacroFeedV4WorkspaceRole input_workspace =
      Sm87MacroFeedV4WorkspaceRole::kInvalid;
  Sm87MacroFeedV4WorkspaceRole output_workspace =
      Sm87MacroFeedV4WorkspaceRole::kInvalid;
  Sm87MacroFeedV4StateWriteMode state_write_mode =
      Sm87MacroFeedV4StateWriteMode::kInvalid;
  bool input_consumed_before_output_publication = false;
  bool output_reuse_waits_for_completion = false;
  bool stages_kv = false;
  bool stages_conv_state = false;
  bool stages_gdn_state = false;
  bool publishes_state_to_decode = true;
};

struct Sm87MacroFeedV4PanelStateTransaction final {
  std::size_t panel_index = kSm87MacroFeedV4PanelCount;
  std::size_t token_begin = kSm87MacroFeedV4P40Tokens;
  std::size_t token_end = kSm87MacroFeedV4P40Tokens;
  std::size_t incoming_state_epoch = kSm87MacroFeedV4PanelCount + 1U;
  std::size_t outgoing_state_epoch = kSm87MacroFeedV4PanelCount + 1U;
  std::size_t commit_dependency_sequence_ordinal =
      kSm87MacroFeedV4LayerStepCount;
  std::size_t kv_layer_count = 0U;
  std::size_t conv_layer_count = 0U;
  std::size_t gdn_layer_count = 0U;
  bool kv_uses_disjoint_final_token_slice = false;
  bool conv_and_gdn_use_private_next_epoch = false;
  bool atomic_kv_conv_gdn_commit = false;
  bool commit_after_layer_63 = false;
  bool next_panel_waits_for_commit = false;
  bool rollback_discards_uncommitted_panel_state = false;
  bool state_private_to_prefill_until_request_commit = false;
  bool state_visible_to_decode = true;
};

struct Sm87MacroFeedV4PanelPlan final {
  std::size_t panel_index = kSm87MacroFeedV4PanelCount;
  std::size_t token_begin = kSm87MacroFeedV4P40Tokens;
  std::size_t token_count = 0U;
  std::size_t sequence_begin = kSm87MacroFeedV4LayerStepCount;
  std::size_t sequence_end = kSm87MacroFeedV4LayerStepCount;
  Sm87MacroFeedV4WorkspaceRole initial_workspace =
      Sm87MacroFeedV4WorkspaceRole::kInvalid;
  Sm87MacroFeedV4WorkspaceRole final_workspace =
      Sm87MacroFeedV4WorkspaceRole::kInvalid;
  std::array<Sm87MacroFeedV4LayerStep, kSm87MacroFeedV4LayerCount> layers{};
  Sm87MacroFeedV4PanelStateTransaction state_transaction{};
  bool embedding_publishes_initial_workspace = false;
  bool workspace_reuse_waits_for_panel_commit = false;
};

struct Sm87MacroFeedV4RoutePolicy final {
  bool sm87_only = false;
  bool real_checkpoint_required = false;
  bool authenticated_aot_deployment_plan = false;
  bool startup_bound_tactics = false;
  bool request_time_jit_allowed = true;
  bool request_time_repack_allowed = true;
  bool request_time_autotune_allowed = true;
  bool fallback_allowed = true;
  bool cublaslt_allowed = true;
  bool mtp_allowed = true;
  bool approximate_numerics_allowed = true;
  bool default_off = false;
  bool test_only_contract = false;
  bool selector_bound = true;
  bool launcher_present = true;
  bool production_dispatch_eligible = true;
  bool numerical_qualification_complete = true;
};

struct Sm87MacroFeedV4PanelWavefrontPlan final {
  std::array<std::uint8_t, 8U> magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  std::string_view candidate_id{};
  std::string_view deployment_plan_id{};
  Sm87MacroFeedV4P40ApiIdentity api{};
  Sm87MacroFeedV4Traversal traversal = Sm87MacroFeedV4Traversal::kInvalid;
  std::size_t prompt_tokens = 0U;
  std::size_t panel_tokens = 0U;
  std::size_t panel_count = 0U;
  std::size_t layer_count = 0U;
  Sm87MacroFeedV4WorkspacePlan workspace{};
  Sm87MacroFeedV4PhaseAliasingPlan phase_aliasing{};
  Sm87MacroFeedV4StateOwnershipPlan state_ownership{};
  std::array<Sm87MacroFeedV4PanelPlan, kSm87MacroFeedV4PanelCount> panels{};
  Sm87MacroFeedV4RoutePolicy route{};
  bool panel_loop_is_outermost = false;
  bool layer_loop_is_natural_order_innermost = false;
  bool final_request_commit_after_all_panels = false;
  bool partial_panel_commit_visible_to_decode = true;
};

enum class Sm87MacroFeedV4PlanIssue : std::uint32_t {
  kNone = 0U,
  kIdentity = 1U << 0U,
  kApiIdentity = 1U << 1U,
  kGeometry = 1U << 2U,
  kTraversal = 1U << 3U,
  kWorkspace = 1U << 4U,
  kWholeP40Temporary = 1U << 5U,
  kLayerSchedule = 1U << 6U,
  kStateTransaction = 1U << 7U,
  kForbiddenRoute = 1U << 8U,
  kDispatchBoundary = 1U << 9U,
  kPhaseAliasing = 1U << 10U,
  kStateOwnership = 1U << 11U,
};

struct Sm87MacroFeedV4PlanValidation final {
  std::uint32_t issue_mask = 0U;
  std::size_t first_bad_panel = kSm87MacroFeedV4PanelCount;
  std::size_t first_bad_layer = kSm87MacroFeedV4LayerCount;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return issue_mask == 0U;
  }
};

[[nodiscard]] constexpr bool has_sm87_macrofeed_v4_plan_issue(
    const Sm87MacroFeedV4PlanValidation& validation,
    const Sm87MacroFeedV4PlanIssue issue) noexcept {
  return (validation.issue_mask & static_cast<std::uint32_t>(issue)) != 0U;
}

[[nodiscard]] Sm87MacroFeedV4PanelWavefrontPlan
make_sm87_macrofeed_v4_p40_panel_wavefront_plan() noexcept;

[[nodiscard]] Sm87MacroFeedV4PlanValidation
validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(
    const Sm87MacroFeedV4PanelWavefrontPlan& plan) noexcept;

[[nodiscard]] bool sm87_macrofeed_v4_p40_panel_wavefront_plan_valid(
    const Sm87MacroFeedV4PanelWavefrontPlan& plan) noexcept;

}  // namespace q3x::runtime
