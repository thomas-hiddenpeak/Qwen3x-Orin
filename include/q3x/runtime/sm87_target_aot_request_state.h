#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace q3x::runtime {

// Exact cold-request owner for the P40000 target-AOT system candidate.  This
// surface is intentionally unrelated to RequestState: it has no legacy
// profile, raw-arena accessor, caller-supplied address, or caller-supplied
// identity.  The only executable view is declared in an engine-internal
// header and is derived by this owner from its one allocation and one
// nonblocking execution stream.
inline constexpr std::size_t kSm87TargetAotP40PromptTokens = 40'000U;
inline constexpr std::size_t kSm87TargetAotP40RequestCapacityTokens = 40'001U;
inline constexpr std::size_t kSm87TargetAotP40LayerCount = 64U;
inline constexpr std::size_t kSm87TargetAotP40GdnLayerCount = 48U;
inline constexpr std::size_t kSm87TargetAotP40FullLayerCount = 16U;
inline constexpr std::size_t kSm87TargetAotP40Hidden = 5'120U;
inline constexpr std::size_t kSm87TargetAotP40Intermediate = 17'408U;
inline constexpr std::size_t kSm87TargetAotP40GdnRawWidth = 16'384U;
inline constexpr std::size_t kSm87TargetAotP40GdnAbWidth = 96U;
inline constexpr std::size_t kSm87TargetAotP40AttentionQGateWidth = 12'288U;
inline constexpr std::size_t kSm87TargetAotP40AttentionWidth = 6'144U;
inline constexpr std::size_t kSm87TargetAotP40KvWidth = 1'024U;
inline constexpr std::uint64_t kSm87TargetAotP40ArenaAlignment = 256U;
inline constexpr std::uint64_t kSm87TargetAotP40PersistentBytes =
    2'699'952'128ULL;
inline constexpr std::uint64_t kSm87TargetAotP40ResidualBytes =
    409'610'240ULL;
inline constexpr std::uint64_t kSm87TargetAotP40FamilyArenaBytes =
    1'966'080'000ULL;
inline constexpr std::uint64_t kSm87TargetAotP40FinalHiddenBytes = 10'240ULL;
inline constexpr std::uint64_t kSm87TargetAotP40RequestArenaBytes =
    5'075'652'608ULL;
inline constexpr std::uint64_t kSm87TargetAotP40RequestLocalRopeBytes = 0ULL;
inline constexpr std::size_t kSm87TargetAotP40LayerEventCount = 8U;
inline constexpr std::size_t kSm87TargetAotP40GlobalEventCount = 7U;
inline constexpr std::size_t kSm87TargetAotP40OwnedEventCount =
    kSm87TargetAotP40LayerCount * kSm87TargetAotP40LayerEventCount +
    kSm87TargetAotP40GlobalEventCount;
inline constexpr std::size_t kSm87TargetAotCancellationSignalBytes =
    sizeof(std::uint32_t);

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_REQUEST_STATE_V1_ADMISSION)
inline constexpr bool kSm87TargetAotRequestStateV1AdmissionCompiled = true;
#else
inline constexpr bool kSm87TargetAotRequestStateV1AdmissionCompiled = false;
#endif

enum class Sm87TargetAotRequestScalarType : std::uint8_t {
  kInvalid = 0U,
  kU32,
  kBf16,
};

enum class Sm87TargetAotRequestLifetime : std::uint8_t {
  kInvalid = 0U,
  kRequestOwner,
  kRequestTransactionUnpublishedUntilCommit,
  kEmbeddingOnly,
  kLayerResidual,
  kProjectionInput,
  kGdnProducerUntilCoreCompletion,
  kGdnOutputUntilOProjectionCompletion,
  kAttentionPreparation,
  kAttentionCore,
  kAttentionOutputUntilOProjectionCompletion,
  kMlpActivatedUntilDownCompletion,
  kProjectionBranchUntilResidualCompletion,
  kFinalHiddenUntilRequestCommit,
  kEngineExternal,
};

enum class Sm87TargetAotRequestLayerKind : std::uint8_t {
  kInvalid = 0U,
  kGdn,
  kFullAttention,
};

enum class Sm87TargetAotRequestAliasTransition : std::uint8_t {
  kInvalid = 0U,
  kEmbeddingComplete,
  kInputProjectionsComplete,
  kGdnCoreComplete,
  kAttentionPreparationComplete,
  kAttentionCoreComplete,
  kAttentionOProjectionComplete,
  kGateUpComplete,
  kExactInPlaceKvPreprocess,
};

enum class Sm87TargetAotRequestAliasRole : std::uint8_t {
  kInvalid = 0U,
  kTokenIds,
  kFamilyArena,
  kGdnNormalizedInput,
  kGdnOutput,
  kGdnRawQkvZ,
  kGdnOBranch,
  kAttentionNormalizedInput,
  kAttentionProcessedQGate,
  kAttentionRawQGate,
  kAttentionPreGateAndGated,
  kAttentionOBranch,
  kMlpNormalizedInput,
  kMlpDownBranch,
  kRawKeyTransactionSpan,
  kProcessedKeyTransactionSpan,
  kRawValueTransactionSpan,
  kProcessedValueTransactionSpan,
  kLayerResidualInput,
  kLayerResidualPublication,
};

struct Sm87TargetAotRequestRegion {
  std::uint64_t arena_offset = 0U;
  std::uint64_t byte_size = 0U;
  Sm87TargetAotRequestScalarType scalar_type =
      Sm87TargetAotRequestScalarType::kInvalid;
  Sm87TargetAotRequestLifetime lifetime =
      Sm87TargetAotRequestLifetime::kInvalid;
};

struct Sm87TargetAotRequestMatrixRegion {
  Sm87TargetAotRequestRegion storage{};
  std::uint32_t rows = 0U;
  std::uint32_t columns = 0U;
  std::uint64_t row_stride_elements = 0U;
};

struct Sm87TargetAotRequestAliasContract {
  Sm87TargetAotRequestAliasRole producer =
      Sm87TargetAotRequestAliasRole::kInvalid;
  Sm87TargetAotRequestAliasRole successor =
      Sm87TargetAotRequestAliasRole::kInvalid;
  Sm87TargetAotRequestAliasTransition transition =
      Sm87TargetAotRequestAliasTransition::kInvalid;
  std::size_t family_cardinality = 0U;
  bool same_allocation_required = false;
  bool same_begin_required = false;
  bool same_extent_required = false;
  bool successor_forbidden_before_transition = false;
};

struct Sm87TargetAotP40PersistentRegions {
  std::array<Sm87TargetAotRequestRegion, kSm87TargetAotP40GdnLayerCount>
      conv_history{};
  std::array<Sm87TargetAotRequestRegion, kSm87TargetAotP40GdnLayerCount>
      recurrent_state{};
  std::array<Sm87TargetAotRequestMatrixRegion,
             kSm87TargetAotP40FullLayerCount>
      key{};
  std::array<Sm87TargetAotRequestMatrixRegion,
             kSm87TargetAotP40FullLayerCount>
      value{};
};

struct Sm87TargetAotP40GdnFamilyRegions {
  Sm87TargetAotRequestMatrixRegion raw_qkvz{};
  Sm87TargetAotRequestMatrixRegion bf16_ab{};
  Sm87TargetAotRequestMatrixRegion normalized_input{};
  Sm87TargetAotRequestMatrixRegion output{};
  Sm87TargetAotRequestMatrixRegion o_branch{};
};

struct Sm87TargetAotP40AttentionFamilyRegions {
  Sm87TargetAotRequestMatrixRegion raw_q_gate{};
  Sm87TargetAotRequestMatrixRegion normalized_input{};
  Sm87TargetAotRequestMatrixRegion processed_q{};
  Sm87TargetAotRequestMatrixRegion processed_gate{};
  Sm87TargetAotRequestMatrixRegion pre_gate_output{};
  Sm87TargetAotRequestMatrixRegion gated_output{};
  Sm87TargetAotRequestMatrixRegion o_branch{};
};

struct Sm87TargetAotP40MlpFamilyRegions {
  Sm87TargetAotRequestMatrixRegion normalized_input{};
  Sm87TargetAotRequestMatrixRegion activated{};
  Sm87TargetAotRequestMatrixRegion down_branch{};
};

struct Sm87TargetAotP40LayerMemoryBinding {
  Sm87TargetAotRequestLayerKind kind =
      Sm87TargetAotRequestLayerKind::kInvalid;
  std::size_t family_ordinal = 0U;
  Sm87TargetAotRequestRegion conv_history{};
  Sm87TargetAotRequestRegion recurrent_state{};
  Sm87TargetAotRequestMatrixRegion key{};
  Sm87TargetAotRequestMatrixRegion value{};
};

struct Sm87TargetAotP40EngineRopeRequirement {
  std::size_t maximum_position_embeddings = 262'144U;
  std::size_t rotary_pairs = 32U;
  std::size_t scalar_bytes = sizeof(float);
  std::uint64_t complete_table_bytes = 67'108'864ULL;
  Sm87TargetAotRequestLifetime lifetime =
      Sm87TargetAotRequestLifetime::kEngineExternal;
  bool included_in_request_arena = false;
};

inline constexpr std::size_t kSm87TargetAotP40AliasContractCount = 10U;

struct Sm87TargetAotP40RequestMemoryPlan {
  std::uint32_t prompt_tokens = 0U;
  std::uint32_t request_capacity_tokens = 0U;
  std::uint64_t arena_bytes = 0U;
  Sm87TargetAotRequestRegion persistent_arena{};
  Sm87TargetAotP40PersistentRegions persistent{};
  Sm87TargetAotRequestMatrixRegion residual{};
  Sm87TargetAotRequestRegion family_arena{};
  Sm87TargetAotRequestMatrixRegion token_ids{};
  Sm87TargetAotP40GdnFamilyRegions gdn{};
  Sm87TargetAotP40AttentionFamilyRegions attention{};
  Sm87TargetAotP40MlpFamilyRegions mlp{};
  Sm87TargetAotRequestMatrixRegion final_hidden{};
  std::array<Sm87TargetAotP40LayerMemoryBinding,
             kSm87TargetAotP40LayerCount>
      layers{};
  std::array<Sm87TargetAotRequestAliasContract,
             kSm87TargetAotP40AliasContractCount>
      alias_contracts{};
  Sm87TargetAotP40EngineRopeRequirement engine_rope{};
  std::size_t owned_event_count = 0U;
  bool single_allocation = false;
  bool cold_request_only = false;
  bool one_request_wide_commit = false;
  bool cancellation_discards_unpublished = false;
  bool kv_preprocess_in_place_alias_required = false;
  bool exposes_raw_arena = true;
  bool permits_legacy_fallback = true;
};

[[nodiscard]] Sm87TargetAotP40RequestMemoryPlan
build_sm87_target_aot_p40_request_memory_plan() noexcept;

[[nodiscard]] bool validate_sm87_target_aot_p40_request_memory_plan(
    const Sm87TargetAotP40RequestMemoryPlan& plan) noexcept;

enum class Sm87TargetAotRequestTransactionPhase : std::uint8_t {
  kInvalid = 0U,
  kAdmittedUnpublished,
  kPrefillActiveUnpublished,
  kCommitted,
  kCancelled,
};

enum class Sm87TargetAotRequestStateError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kPlanInvalid,
  kInsufficientDeviceMemory,
  kAllocationFailure,
  kStreamCreationFailure,
  kCancellationControlFailure,
  kEventCreationFailure,
  kInitializationFailure,
  kOwnerSnapshotInvalid,
};

struct Sm87TargetAotRequestStateDiagnostic {
  Sm87TargetAotRequestStateError code = Sm87TargetAotRequestStateError::kNone;
  std::string context;
  int cuda_error = 0;
};

namespace sm87_target_aot_request_detail {
class Sm87TargetAotRequestStateAccess;
}

class Sm87TargetAotP40RequestState;
struct Sm87TargetAotRequestStateResult;
[[nodiscard]] Sm87TargetAotRequestStateResult
create_sm87_target_aot_p40_request_state() noexcept;

class Sm87TargetAotP40RequestState final {
 public:
  Sm87TargetAotP40RequestState() = delete;
  ~Sm87TargetAotP40RequestState();
  Sm87TargetAotP40RequestState(const Sm87TargetAotP40RequestState&) = delete;
  Sm87TargetAotP40RequestState& operator=(
      const Sm87TargetAotP40RequestState&) = delete;
  Sm87TargetAotP40RequestState(Sm87TargetAotP40RequestState&&) = delete;
  Sm87TargetAotP40RequestState& operator=(
      Sm87TargetAotP40RequestState&&) = delete;

  [[nodiscard]] const Sm87TargetAotP40RequestMemoryPlan& plan() const noexcept;
  [[nodiscard]] Sm87TargetAotRequestTransactionPhase transaction_phase()
      const noexcept;
  [[nodiscard]] bool committed() const noexcept;

 private:
  struct Impl;
  explicit Sm87TargetAotP40RequestState(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;

  friend class sm87_target_aot_request_detail::
      Sm87TargetAotRequestStateAccess;
  friend Sm87TargetAotRequestStateResult
  create_sm87_target_aot_p40_request_state() noexcept;
};

struct Sm87TargetAotRequestStateResult {
  std::unique_ptr<Sm87TargetAotP40RequestState> value;
  Sm87TargetAotRequestStateDiagnostic diagnostic{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return value != nullptr &&
           diagnostic.code == Sm87TargetAotRequestStateError::kNone;
  }
};

// Default-off and fixed-shape. The caller supplies neither memory nor an
// identity namespace. When admission is compiled this function owns the one
// 5,075,652,608-byte allocation, one nonblocking CUDA stream, one mapped
// owner-controlled cancellation word, and all CUDA events; otherwise it fails
// before querying or mutating the GPU. The cancellation word is control state,
// not a second device arena allocation.
[[nodiscard]] const char* to_string(
    Sm87TargetAotRequestStateError error) noexcept;

}  // namespace q3x::runtime
