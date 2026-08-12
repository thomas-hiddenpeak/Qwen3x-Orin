#pragma once

#include "q3x/kernels/sm87_target_aot_attention_plan.h"
#include "q3x/kernels/sm87_target_aot_gdn_plan.h"
#include "q3x/kernels/sm87_target_aot_projection_plan.h"
#include "q3x/runtime/prefill_operator_roles.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace q3x::runtime {

// Host-only composition schema for AC-PREFILL-SM87-AOT-SYSTEM-v1. It owns
// canonical constituent designs, logical publications, physical execution
// groups, and opaque DeploymentPlan keys. It contains no launcher, device
// pointer, CUDA handle, callable function, selector, or old-plan asset.
inline constexpr std::string_view kSm87AotPrefillSystemCandidateId =
    "AC-PREFILL-SM87-AOT-SYSTEM-v1";
inline constexpr std::string_view kSm87AotPrefillSystemP40PlanId =
    "q3x.sm87.ac-prefill-aot-system.p40.v1";
inline constexpr std::string_view kSm87AotPrefillSystemP60PlanId =
    "q3x.sm87.ac-prefill-aot-system.p60.v1";
inline constexpr std::string_view kSm87AotPrefillSystemP130PlanId =
    "q3x.sm87.ac-prefill-aot-system.p130.v1";

inline constexpr std::size_t kSm87AotPrefillSystemMaximumStagingSpans = 16U;
inline constexpr std::size_t kSm87AotPrefillSystemRequiredRoleCount =
    kPrefillRequiredOperatorRoleCount;
inline constexpr std::size_t kSm87AotPrefillSystemProjectionPlanCount = 5U;
inline constexpr std::size_t kSm87AotPrefillSystemProjectionDataflowCount =
    6U;
inline constexpr std::size_t kSm87AotPrefillSystemLayerCount = 64U;
inline constexpr std::size_t kSm87AotPrefillSystemGdnLayerCount = 48U;
inline constexpr std::size_t kSm87AotPrefillSystemFullAttentionLayerCount =
    16U;
inline constexpr std::size_t kSm87AotPrefillLayerStageCount = 8U;
inline constexpr std::size_t kSm87AotPrefillMaximumStagePredecessors = 2U;
inline constexpr std::size_t kSm87AotPrefillProjectionPartitionBindingCount =
    9U;
inline constexpr std::size_t kSm87AotPrefillTypedResourceEdgeCount = 39U;
inline constexpr std::size_t kSm87AotPrefillTypedEventEdgeCount = 13U;
static_assert(kSm87AotPrefillSystemRequiredRoleCount == 17U);

#if defined(Q3X_ENABLE_SM87_AOT_SYSTEM_V1_ADMISSION)
inline constexpr bool kSm87AotPrefillSystemSchemaAdmissionCompiled = true;
#else
inline constexpr bool kSm87AotPrefillSystemSchemaAdmissionCompiled = false;
#endif

enum class Sm87AotPrefillSystemQualification : std::uint8_t {
  kInvalid = 0U,
  kAccuracyUnqualified,
};

enum class Sm87AotPrefillExecutionGroup : std::uint8_t {
  kEmbedding = 0U,
  kInputAndLayerNorm,
  kLinearQkvZ,
  kLinearAb,
  kGdn,
  kLinearOResidual,
  kFullQkv,
  kFullQkNormRopePublish,
  kAttention,
  kFullOResidual,
  kPostAttentionLayerNorm,
  kGateUp,
  kDownResidual,
  kFinalHandoff,
  kCount,
  kInvalid = 0xffU,
};

inline constexpr std::size_t kSm87AotPrefillExecutionGroupCount =
    static_cast<std::size_t>(Sm87AotPrefillExecutionGroup::kCount);
static_assert(kSm87AotPrefillExecutionGroupCount == 14U);
static_assert(kSm87AotPrefillExecutionGroupCount <= 64U);

inline constexpr std::array<kernels::Sm87TargetAotProjectionRole,
                            kSm87AotPrefillSystemProjectionPlanCount>
    kSm87AotPrefillProjectionRoles{{
        kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp,
        kernels::Sm87TargetAotProjectionRole::kNvFp4Down,
        kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
        kernels::Sm87TargetAotProjectionRole::kFp8FullQkv,
        kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
    }};

struct Sm87AotPrefillSystemCapacityPlan {
  kernels::Sm87TargetAotCapacityContract prompt;
  std::array<std::size_t, kSm87AotPrefillSystemMaximumStagingSpans>
      staging_span_tokens{};
  std::size_t staging_span_count = 0U;

  // Preliminary host planning values only. They are not simultaneous-fit or
  // production-capacity evidence until a bound runtime plan qualifies them.
  std::uint64_t selected_request_memory_bytes = 0U;
  std::uint64_t conservative_request_memory_bytes = 0U;
};

// New-system BF16 A/B constituent. This describes one paired native AOT
// topology for two independent [48,5120] BF16 weights. It is deliberately
// not an admission of the historical P40 implementation: all three exact
// witnesses use a predicated M64 tail and remain CUDA/resource/numerical
// unqualified until a new-system body is measured and validated.
enum class Sm87AotPrefillBf16AbRole : std::uint8_t {
  kInvalid = 0U,
  kInProjA,
  kInProjB,
};

struct Sm87AotPrefillBf16AbPartition {
  Sm87AotPrefillBf16AbRole role = Sm87AotPrefillBf16AbRole::kInvalid;
  std::size_t paired_row_offset = 0U;
  std::size_t output_rows = 0U;
  std::size_t checkpoint_tensor_ordinal = 0U;
  std::size_t independent_publication_ordinal = 0U;
};

struct Sm87AotPrefillBf16AbPlan {
  kernels::Sm87TargetAotCapacityBucket capacity_bucket =
      kernels::Sm87TargetAotCapacityBucket::kInvalid;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t rows_per_projection = 0U;
  std::size_t projection_count = 0U;
  std::size_t paired_output_rows = 0U;
  std::array<Sm87AotPrefillBf16AbPartition, 2U> partitions{};
  std::size_t tile_m = 0U;
  std::size_t tile_n = 0U;
  std::size_t tile_k = 0U;
  std::size_t grid_m = 0U;
  std::size_t k_tiles = 0U;
  std::size_t tail_rows = 0U;
  std::size_t logical_pair_tasks = 0U;
  std::size_t physical_ctas = 0U;
  std::size_t threads_per_cta = 0U;
  std::size_t pipeline_stages = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t split_k_workspace_bytes = 0U;
  std::size_t minimum_active_ctas_per_sm = 0U;
  bool predicated_exact_tail = false;
  bool independent_fp32_accumulators = false;
  bool independent_bf16_rne_publications = false;
  bool cuda_implementation_present = false;
  bool static_resources_qualified = false;
  bool numerical_reduction_qualified = false;
  bool production_dispatch_eligible = false;

  [[nodiscard]] constexpr bool valid() const noexcept;
};

struct Sm87AotPrefillConstituentDesign {
  std::array<kernels::Sm87TargetAotProjectionPlan,
             kSm87AotPrefillSystemProjectionPlanCount>
      projections{};
  Sm87AotPrefillBf16AbPlan bf16_ab;
  kernels::Sm87TargetAotAttentionPlan attention;
  kernels::Sm87TargetAotGdnPlan gdn;
};

enum class Sm87AotPrefillLayerKind : std::uint8_t {
  kInvalid = 0U,
  kGdn,
  kFullAttention,
};

enum class Sm87AotPrefillLayerInputSource : std::uint8_t {
  kInvalid = 0U,
  kEmbedding,
  kPriorLayerDownResidual,
};

enum class Sm87AotPrefillInstanceSource : std::uint8_t {
  kInvalid = 0U,
  kEmbedding,
  kLayerStage,
};

enum class Sm87AotPrefillInstanceEdgeKind : std::uint8_t {
  kInvalid = 0U,
  kData,
  kResidualBypass,
};

struct Sm87AotPrefillInstancePredecessor {
  Sm87AotPrefillInstanceSource source =
      Sm87AotPrefillInstanceSource::kInvalid;
  Sm87AotPrefillInstanceEdgeKind edge_kind =
      Sm87AotPrefillInstanceEdgeKind::kInvalid;
  std::size_t layer_index = kSm87AotPrefillSystemLayerCount;
  std::size_t stage_index = kSm87AotPrefillLayerStageCount;
  Sm87AotPrefillExecutionGroup group =
      Sm87AotPrefillExecutionGroup::kInvalid;
};

// This instance is the synchronization authority. Group masks below are only
// summaries over this 64x8 DAG and must never be used to launch or wait.
struct Sm87AotPrefillLayerStageInstance {
  std::size_t layer_index = kSm87AotPrefillSystemLayerCount;
  std::size_t stage_index = kSm87AotPrefillLayerStageCount;
  Sm87AotPrefillExecutionGroup group =
      Sm87AotPrefillExecutionGroup::kInvalid;
  std::array<Sm87AotPrefillInstancePredecessor,
             kSm87AotPrefillMaximumStagePredecessors>
      predecessors{};
  std::size_t predecessor_count = 0U;
};

// This is the non-aggregate 64-layer DAG witness. Full-Attention layers are
// exactly 3,7,...,63 (zero based); every other layer is GDN. The fixed stage
// array is a canonical topological audit order, not a forced stream order.
// Actual synchronization is defined only by each stage instance's typed
// predecessor edges: QKVZ and A/B may proceed independently after input norm,
// and GDN waits for both. Stage 0 also publishes the original layer-input span
// as a bit-exact zero-copy residual; O-residual consumes that publication, not
// the normalized values. Group masks are summaries only. No unmeasured
// multi-stream performance claim is implied.
struct Sm87AotPrefillLayerSchedule {
  std::size_t layer_index = kSm87AotPrefillSystemLayerCount;
  Sm87AotPrefillLayerKind kind = Sm87AotPrefillLayerKind::kInvalid;
  Sm87AotPrefillLayerInputSource input_source =
      Sm87AotPrefillLayerInputSource::kInvalid;
  std::size_t family_ordinal = 0U;
  std::array<Sm87AotPrefillExecutionGroup,
             kSm87AotPrefillLayerStageCount>
      ordered_groups{};
  std::array<Sm87AotPrefillLayerStageInstance,
             kSm87AotPrefillLayerStageCount>
      stage_instances{};
};

struct Sm87AotPrefillFinalHandoffInstance {
  Sm87AotPrefillExecutionGroup group =
      Sm87AotPrefillExecutionGroup::kInvalid;
  Sm87AotPrefillInstancePredecessor final_layer_predecessor{};
};

// Opaque typed keys bind the raw Full-QKV producer, Q/K RMSNorm + position
// RoPE transform, processed Q/gate/K/V publications and Attention consumer.
// Each key denotes a future per-layer binding table, never a pointer.
struct Sm87AotPrefillAttentionDataflowBinding {
  Sm87AotPrefillExecutionGroup raw_producer_group =
      Sm87AotPrefillExecutionGroup::kInvalid;
  Sm87AotPrefillExecutionGroup preprocess_group =
      Sm87AotPrefillExecutionGroup::kInvalid;
  Sm87AotPrefillExecutionGroup consumer_group =
      Sm87AotPrefillExecutionGroup::kInvalid;
  std::uint64_t raw_q_gate_input_span_table_identity = 0U;
  std::uint64_t raw_k_input_span_table_identity = 0U;
  std::uint64_t raw_v_input_span_table_identity = 0U;
  std::uint64_t raw_q_gate_ready_event_table_identity = 0U;
  std::uint64_t raw_k_ready_event_table_identity = 0U;
  std::uint64_t raw_v_ready_event_table_identity = 0U;
  std::uint64_t q_norm_weight_table_identity = 0U;
  std::uint64_t k_norm_weight_table_identity = 0U;
  std::uint64_t rope_position_contract_identity = 0U;
  std::uint64_t processed_q_gate_publication_table_identity = 0U;
  std::uint64_t processed_k_publication_table_identity = 0U;
  std::uint64_t processed_v_publication_table_identity = 0U;
  std::uint64_t preparation_ready_event_table_identity = 0U;
  std::uint64_t attention_q_gate_input_span_table_identity = 0U;
  std::uint64_t attention_k_input_span_table_identity = 0U;
  std::uint64_t attention_v_input_span_table_identity = 0U;
  std::uint64_t attention_core_input_ready_event_table_identity = 0U;
  std::uint64_t kv_cache_arena_table_identity = 0U;
  std::uint64_t kv_cache_lifetime_contract_identity = 0U;
  std::uint64_t staged_kv_transaction_publication_table_identity = 0U;
  std::uint64_t staged_kv_transaction_ready_event_table_identity = 0U;
  std::uint64_t position_rope_epoch_publication_table_identity = 0U;
  std::uint64_t position_rope_epoch_ready_event_table_identity = 0U;
  std::uint64_t pre_gate_bf16_output_span_table_identity = 0U;
  std::uint64_t pre_gate_bf16_publication_table_identity = 0U;
  std::uint64_t pre_gate_bf16_lifetime_contract_identity = 0U;
  std::uint64_t pre_gate_bf16_completion_event_table_identity = 0U;
  std::uint64_t gated_output_span_table_identity = 0U;
  std::uint64_t gated_output_publication_table_identity = 0U;
  std::uint64_t gated_output_lifetime_contract_identity = 0U;
  std::uint64_t gated_output_completion_event_table_identity = 0U;
  std::size_t first_position = 0U;
  std::size_t initial_kv_length = 0U;
  std::size_t kv_capacity_tokens = 0U;
  std::size_t max_position_embeddings = 0U;
  std::size_t rotary_elements = 0U;
  std::size_t rotary_pairs = 0U;
  std::uint32_t rms_epsilon_fp32_bits = 0U;
  std::uint32_t attention_scale_fp32_bits = 0U;
  std::uint64_t rope_theta = 0U;
  std::size_t output_tokens = 0U;
  std::size_t output_features = 0U;
  bool rope_cache_is_fp32 = false;
  bool raw_q_gate_is_per_head_interleaved = false;
  bool gate_is_bit_exact_split_copy = false;
  bool processed_kv_is_nhd_transaction_staged_unpublished = false;
  bool pre_gate_output_is_bf16 = false;
  bool gated_output_is_bf16 = false;
};

struct Sm87AotPrefillProjectionPartitionBinding {
  kernels::Sm87TargetAotLogicalRole logical_role =
      kernels::Sm87TargetAotLogicalRole::kInvalid;
  std::size_t output_offset = 0U;
  std::size_t output_features = 0U;
  std::uint64_t source_packed_asset_identity = 0U;
  std::uint64_t independent_scale_identity = 0U;
  std::uint64_t raw_bits_contract_identity = 0U;
  std::uint64_t output_span_identity = 0U;
  std::uint64_t publication_identity = 0U;
  std::uint64_t completion_event_identity = 0U;
};

enum class Sm87AotPrefillProjectionFamily : std::uint8_t {
  kInvalid = 0U,
  kAllLayers,
  kGdnLayers,
  kFullAttentionLayers,
};

struct Sm87AotPrefillProjectionDataflowBinding {
  kernels::Sm87TargetAotProjectionRole role =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  Sm87AotPrefillProjectionFamily family =
      Sm87AotPrefillProjectionFamily::kInvalid;
  std::uint64_t execution_group_mask = 0U;
  std::uint64_t input_span_table_identity = 0U;
  std::uint64_t input_ready_event_table_identity = 0U;
  std::uint64_t aggregate_completion_event_table_identity = 0U;
  std::array<Sm87AotPrefillProjectionPartitionBinding,
             kernels::kSm87TargetAotProjectionMaximumPartitions>
      partitions{};
  std::size_t partition_count = 0U;
};

// Remaining route resources are named explicitly so a generic group key can
// never stand in for a missing model operand, nonlinear publication or
// residual lifetime. Each identity denotes a 64-layer (or family-specific)
// authenticated binding table in the future DeploymentPlan.
struct Sm87AotPrefillModelDataflowBinding {
  Sm87AotPrefillBf16AbRole bf16_a_role =
      Sm87AotPrefillBf16AbRole::kInvalid;
  Sm87AotPrefillBf16AbRole bf16_b_role =
      Sm87AotPrefillBf16AbRole::kInvalid;
  std::uint64_t token_id_span_identity = 0U;
  std::uint64_t embedding_token_input_span_identity = 0U;
  std::uint64_t embedding_weight_table_identity = 0U;
  std::uint64_t embedding_publication_table_identity = 0U;
  std::uint64_t layer_input_span_table_identity = 0U;
  std::uint64_t input_norm_input_span_table_identity = 0U;
  std::uint64_t input_norm_weight_table_identity = 0U;
  std::uint64_t input_norm_publication_table_identity = 0U;
  std::uint64_t input_norm_epsilon_contract_identity = 0U;
  std::uint64_t gdn_o_residual_bypass_input_span_identity = 0U;
  std::uint64_t full_o_residual_bypass_input_span_identity = 0U;
  std::uint64_t gdn_o_projection_residual_input_span_identity = 0U;
  std::uint64_t full_o_projection_residual_input_span_identity = 0U;
  std::uint64_t gdn_attention_residual_publication_identity = 0U;
  std::uint64_t full_attention_residual_publication_identity = 0U;
  std::uint64_t gdn_post_norm_input_span_identity = 0U;
  std::uint64_t full_post_norm_input_span_identity = 0U;
  std::uint64_t gdn_down_residual_bypass_input_span_identity = 0U;
  std::uint64_t full_down_residual_bypass_input_span_identity = 0U;
  std::uint64_t post_attention_norm_weight_table_identity = 0U;
  std::uint64_t post_attention_norm_publication_table_identity = 0U;
  std::uint64_t post_attention_norm_epsilon_contract_identity = 0U;
  std::uint64_t silu_gate_input_span_table_identity = 0U;
  std::uint64_t silu_up_input_span_table_identity = 0U;
  std::uint64_t silu_times_up_publication_table_identity = 0U;
  std::uint64_t down_branch_residual_input_span_identity = 0U;
  std::uint64_t next_layer_input_publication_table_identity = 0U;
  std::uint64_t residual_lifetime_contract_identity = 0U;
  std::uint64_t bf16_ab_input_span_table_identity = 0U;
  std::uint64_t bf16_a_weight_table_identity = 0U;
  std::uint64_t bf16_b_weight_table_identity = 0U;
  std::uint64_t bf16_a_publication_table_identity = 0U;
  std::uint64_t bf16_b_publication_table_identity = 0U;
  // Kernel completion is diagnostic/lifetime evidence. GDN consumes only
  // after the distinct BF16-RNE publication event below.
  std::uint64_t bf16_a_completion_event_table_identity = 0U;
  std::uint64_t bf16_b_completion_event_table_identity = 0U;
  std::uint64_t bf16_a_rne_publication_event_identity = 0U;
  std::uint64_t bf16_b_rne_publication_event_identity = 0U;
  std::uint64_t final_norm_weight_identity = 0U;
  std::uint64_t final_norm_input_span_identity = 0U;
  std::uint64_t final_norm_publication_identity = 0U;
  std::uint64_t final_norm_completion_event_identity = 0U;
  std::uint64_t final_norm_epsilon_contract_identity = 0U;
  std::uint64_t final_handoff_identity = 0U;
  float centered_rms_epsilon = 0.0F;
  bool layer_residual_is_bit_exact_zero_copy_publication = false;
};

// The GDN graph binds every non-ephemeral operand that changes the recurrence
// or its publication boundary. These are table identities for all 48 layers.
struct Sm87AotPrefillGdnDataflowBinding {
  Sm87AotPrefillExecutionGroup qkvz_producer_group =
      Sm87AotPrefillExecutionGroup::kInvalid;
  Sm87AotPrefillExecutionGroup ab_producer_group =
      Sm87AotPrefillExecutionGroup::kInvalid;
  Sm87AotPrefillExecutionGroup consumer_group =
      Sm87AotPrefillExecutionGroup::kInvalid;
  std::uint64_t raw_qkv_input_span_table_identity = 0U;
  std::uint64_t raw_z_input_span_table_identity = 0U;
  std::uint64_t raw_qkvz_ready_event_table_identity = 0U;
  std::uint64_t a_input_span_table_identity = 0U;
  std::uint64_t a_ready_event_table_identity = 0U;
  std::uint64_t b_input_span_table_identity = 0U;
  std::uint64_t b_ready_event_table_identity = 0U;
  std::uint64_t conv_weight_table_identity = 0U;
  std::uint64_t a_log_table_identity = 0U;
  std::uint64_t dt_bias_table_identity = 0U;
  std::uint64_t output_norm_weight_table_identity = 0U;
  std::uint64_t raw_bf16_publication_table_identity = 0U;
  std::uint64_t norm_silu_z_publication_table_identity = 0U;
  std::uint64_t output_span_table_identity = 0U;
  std::uint64_t output_publication_table_identity = 0U;
  std::uint64_t final_conv_history_publication_table_identity = 0U;
  std::uint64_t final_recurrent_state_publication_table_identity = 0U;
  std::uint64_t ordinary_kernel_completion_event_table_identity = 0U;
  std::uint64_t post_kernel_stream_ordered_state_ready_receipt_identity = 0U;
  std::uint64_t request_transaction_lifetime_contract_identity = 0U;
  std::uint64_t reset_zero_epoch_table_identity = 0U;
  float l2_norm_epsilon = 0.0F;
  float output_norm_epsilon = 0.0F;
  std::size_t first_position = 0U;
  bool initial_state_is_zero = false;
  bool initial_conv_history_is_zero = false;
  bool reads_initial_state_from_dram = false;
  bool reads_initial_conv_history_from_dram = false;
  bool uses_ordinary_kernel_completion = false;
  bool permits_per_owner_commit = false;
  bool state_transaction_spans_prebound_at_request_admission = false;
  bool requires_cpu_callback_or_host_sync = false;
};

enum class Sm87AotPrefillTypedResource : std::uint8_t {
  kInvalid = 0U,
  kTokenIdsToEmbedding,
  kEmbeddingToLayerZeroInput,
  kPriorDownToLayerInput,
  kLayerInputToInputNorm,
  kInputNormToGdnQkvProjection,
  kInputNormToGdnAbProjection,
  kInputNormToFullQkvProjection,
  kLayerResidualToGdnOResidual,
  kLayerResidualToFullOResidual,
  kGateToSilu,
  kUpToMultiply,
  kSiluTimesUpToDownProjection,
  kGdnQkvToCore,
  kGdnZToCore,
  kFullQGateToPreprocess,
  kFullKToPreprocess,
  kFullVToPreprocess,
  kBf16AToGdn,
  kBf16BToGdn,
  kProcessedQGateToAttention,
  kProcessedKToAttention,
  kProcessedVToAttention,
  kAttentionGatedToOutputProjection,
  kGdnOutputToOutputProjection,
  kGdnOProjectionToResidual,
  kFullOProjectionToResidual,
  kGdnResidualToPostNorm,
  kFullResidualToPostNorm,
  kPostNormToGateUp,
  kDownProjectionToResidual,
  kGdnResidualToDownResidual,
  kFullResidualToDownResidual,
  kLastDownResidualToFinalNorm,
  kKvStateToRequestTransaction,
  kConvHistoryToRequestTransaction,
  kRecurrentStateToRequestTransaction,
  kPositionRopeEpochToRequestTransaction,
  kFinalHiddenToRequestTransaction,
  kCommittedReceiptToDecodeHandoff,
  kCount,
};
static_assert(static_cast<std::size_t>(
                  Sm87AotPrefillTypedResource::kCount) -
                  1U ==
              kSm87AotPrefillTypedResourceEdgeCount);

struct Sm87AotPrefillTypedResourceEdge {
  Sm87AotPrefillTypedResource resource =
      Sm87AotPrefillTypedResource::kInvalid;
  Sm87AotPrefillExecutionGroup producer_group =
      Sm87AotPrefillExecutionGroup::kInvalid;
  Sm87AotPrefillExecutionGroup consumer_group =
      Sm87AotPrefillExecutionGroup::kInvalid;
  std::uint64_t producer_publication_identity = 0U;
  std::uint64_t consumer_input_identity = 0U;
};

struct Sm87AotPrefillTypedEventEdge {
  Sm87AotPrefillTypedResource resource =
      Sm87AotPrefillTypedResource::kInvalid;
  std::uint64_t producer_completion_event_identity = 0U;
  std::uint64_t consumer_ready_event_identity = 0U;
};

struct Sm87AotPrefillRequestTransactionBinding {
  std::size_t kv_layer_publication_count = 0U;
  std::size_t conv_history_publication_count = 0U;
  std::size_t recurrent_state_publication_count = 0U;
  std::size_t position_rope_epoch_publication_count = 0U;
  std::size_t final_hidden_publication_count = 0U;
  std::size_t transaction_commit_count = 0U;
  std::size_t prefill_state_committed_receipt_count = 0U;
  std::uint64_t staged_kv_input_table_identity = 0U;
  std::uint64_t conv_history_input_table_identity = 0U;
  std::uint64_t recurrent_state_input_table_identity = 0U;
  std::uint64_t position_rope_epoch_input_identity = 0U;
  std::uint64_t final_hidden_input_identity = 0U;
  std::uint64_t transaction_lifetime_contract_identity = 0U;
  std::uint64_t prefill_state_committed_receipt_schema_identity = 0U;
  std::uint64_t prefill_state_committed_receipt_identity = 0U;
  std::uint64_t decode_handoff_receipt_input_identity = 0U;
  std::uint64_t staged_kv_ready_input_event_identity = 0U;
  std::uint64_t gdn_state_ready_input_event_identity = 0U;
  std::uint64_t position_rope_epoch_ready_input_event_identity = 0U;
  std::uint64_t final_hidden_ready_input_event_identity = 0U;
  bool one_request_wide_commit = false;
  bool commit_waits_for_all_component_ready_events = false;
  bool commit_is_owned_by_final_handoff = false;
  bool decode_visible_only_after_prefill_state_committed = false;
  bool cancellation_discards_all_unpublished = false;
};

// A logical role records publication work and the physical groups that own
// it. It never duplicates a physical launch/task count. kExact is declared
// intent only; the enclosing route remains accuracy-unqualified.
struct Sm87AotPrefillLogicalPublication {
  PrefillBindingRole role = PrefillBindingRole::kInvalid;
  PrefillNumericalMode declared_numerical_intent =
      PrefillNumericalMode::kUnbound;
  std::uint64_t execution_group_mask = 0U;
  std::uint64_t expected_logical_publications = 0U;
};

// Different task domains stay in different fields. No comparison between
// logical publication count, launches, projection tiles, Attention CTAs, or
// ordered GDN state tasks is implied.
struct Sm87AotPrefillPhysicalWork {
  std::uint64_t expected_launches = 0U;
  std::uint64_t projection_tile_tasks = 0U;
  std::uint64_t bf16_ab_pair_tile_tasks = 0U;
  std::uint64_t attention_cta_tasks = 0U;
  std::uint64_t attention_q_rmsnorm_head_rows = 0U;
  std::uint64_t attention_k_rmsnorm_head_rows = 0U;
  std::uint64_t attention_q_rope_head_rows = 0U;
  std::uint64_t attention_k_rope_head_rows = 0U;
  std::uint64_t attention_processed_q_gate_head_rows = 0U;
  std::uint64_t attention_processed_k_head_rows = 0U;
  std::uint64_t attention_published_v_head_rows = 0U;
  std::uint64_t attention_position_rows = 0U;
  std::uint64_t gdn_preparation_owner_tasks = 0U;
  std::uint64_t gdn_exact_c16_owner_tasks = 0U;
  bool canonical_counts_valid = false;
};

struct Sm87AotPrefillPhysicalExecutionGroup {
  Sm87AotPrefillExecutionGroup group =
      Sm87AotPrefillExecutionGroup::kInvalid;
  std::uint64_t logical_role_mask = 0U;
  // Audit summaries derived from the canonical instance DAG. They are never
  // synchronization or lifetime authority because a group recurs by layer.
  std::uint64_t predecessor_group_mask = 0U;
  // A producer's publication stays live through all summarized consumers.
  std::uint64_t publication_consumer_group_mask = 0U;
  PrefillNumericalMode declared_numerical_intent =
      PrefillNumericalMode::kUnbound;
  PrefillOperatorProvider provider = PrefillOperatorProvider::kUnbound;
  PrefillTacticMode tactic_mode = PrefillTacticMode::kUnbound;
  std::uint64_t tactic_identity = 0U;
  std::uint64_t binding_identity = 0U;
  Sm87AotPrefillPhysicalWork work;
  bool uses_mtp = false;
  bool allows_fallback = false;
};

struct Sm87AotPrefillSystemPlan {
  std::string_view candidate_id;
  std::string_view deployment_plan_id;
  Sm87AotPrefillSystemCapacityPlan capacity;
  Sm87AotPrefillConstituentDesign constituents;
  std::array<Sm87AotPrefillLayerSchedule,
             kSm87AotPrefillSystemLayerCount>
      layer_schedule{};
  std::size_t layer_schedule_count = 0U;
  Sm87AotPrefillFinalHandoffInstance final_handoff_instance;
  Sm87AotPrefillAttentionDataflowBinding attention_dataflow;
  Sm87AotPrefillGdnDataflowBinding gdn_dataflow;
  std::array<Sm87AotPrefillProjectionDataflowBinding,
             kSm87AotPrefillSystemProjectionDataflowCount>
      projection_dataflows{};
  Sm87AotPrefillModelDataflowBinding model_dataflow;
  Sm87AotPrefillRequestTransactionBinding request_transaction;
  std::array<Sm87AotPrefillTypedResourceEdge,
             kSm87AotPrefillTypedResourceEdgeCount>
      typed_resource_edges{};
  std::size_t typed_resource_edge_count = 0U;
  std::array<Sm87AotPrefillTypedEventEdge,
             kSm87AotPrefillTypedEventEdgeCount>
      typed_event_edges{};
  std::size_t typed_event_edge_count = 0U;

  // Opaque nonzero keys into a future authenticated DeploymentPlan. They are
  // never addresses or proof that real resources are bound.
  std::uint64_t binary_identity = 0U;
  std::uint64_t checkpoint_identity = 0U;
  std::uint64_t platform_identity = 0U;
  std::uint64_t numerical_contract_identity = 0U;
  std::uint64_t request_memory_plan_identity = 0U;
  std::uint64_t stream_plan_identity = 0U;
  std::uint64_t handoff_plan_identity = 0U;

  std::array<Sm87AotPrefillLogicalPublication,
             kSm87AotPrefillSystemRequiredRoleCount>
      logical_publications{};
  std::size_t logical_publication_count = 0U;
  std::array<Sm87AotPrefillPhysicalExecutionGroup,
             kSm87AotPrefillExecutionGroupCount>
      execution_groups{};
  std::size_t execution_group_count = 0U;

  Sm87AotPrefillSystemQualification qualification =
      Sm87AotPrefillSystemQualification::kInvalid;
  bool accuracy_qualified = false;
  bool production_dispatch_eligible = false;
  bool uses_prefix_cache = false;
  bool uses_mtp = false;
  bool uses_cublaslt = false;
  bool uses_external_runtime = false;
  bool uses_approximate_numerics = false;
  bool uses_request_time_jit = false;
  bool uses_request_time_autotune = false;
  bool uses_request_time_repack = false;
  bool uses_request_time_tactic_discovery = false;
  bool allows_fallback = false;
  bool allows_silent_truncation = false;
};

enum class Sm87AotPrefillSystemPlanIssue : std::uint64_t {
  kNone = 0U,
  kInvalidCapacity = 1ULL << 0U,
  kCandidateIdentityMismatch = 1ULL << 1U,
  kDeploymentPlanIdentityMismatch = 1ULL << 2U,
  kLegacyOrForeignIdentityForbidden = 1ULL << 3U,
  kStagingGeometryMismatch = 1ULL << 4U,
  kPreliminaryMemoryContractMismatch = 1ULL << 5U,
  kSystemIdentityUnbound = 1ULL << 6U,
  kConstituentDesignMismatch = 1ULL << 7U,
  kInvalidLogicalPublicationCount = 1ULL << 8U,
  kInvalidLogicalRole = 1ULL << 9U,
  kMissingLogicalRole = 1ULL << 10U,
  kDuplicateLogicalRole = 1ULL << 11U,
  kLogicalPublicationDesignMismatch = 1ULL << 12U,
  kDeclaredExactIntentMissing = 1ULL << 13U,
  kInvalidExecutionGroupCount = 1ULL << 14U,
  kInvalidExecutionGroup = 1ULL << 15U,
  kMissingExecutionGroup = 1ULL << 16U,
  kDuplicateExecutionGroup = 1ULL << 17U,
  kExecutionGroupDesignMismatch = 1ULL << 18U,
  kNonNativeExecutionGroup = 1ULL << 19U,
  kExecutionGroupNotAot = 1ULL << 20U,
  kExecutionGroupIdentityUnbound = 1ULL << 21U,
  kExpectedPhysicalWorkMismatch = 1ULL << 22U,
  kLogicalPhysicalOwnershipMismatch = 1ULL << 23U,
  kGateDownTacticAlias = 1ULL << 24U,
  kUnexpectedExecutionGroupIdentityAlias = 1ULL << 25U,
  kForbiddenBoundaryEnabled = 1ULL << 26U,
  kQualificationBoundaryInvalid = 1ULL << 27U,
  kMalformedEnumEncoding = 1ULL << 28U,
  kLayerScheduleMismatch = 1ULL << 29U,
  kDependencyGraphMismatch = 1ULL << 30U,
  kTypedDataflowBindingIncomplete = 1ULL << 31U,
  kUnexpectedIdentityAlias = 1ULL << 32U,
};

struct Sm87AotPrefillSystemPlanValidation {
  std::uint64_t issue_mask = 0U;
  std::array<std::uint8_t, kSm87AotPrefillSystemRequiredRoleCount>
      role_counts{};
  std::array<std::uint8_t, kSm87AotPrefillExecutionGroupCount>
      group_counts{};
  bool identity_schema_complete = false;
  bool capacity_contract_complete = false;
  bool staging_design_complete = false;
  bool preliminary_memory_design_complete = false;
  bool constituent_design_complete = false;
  bool layer_schedule_design_complete = false;
  bool dependency_graph_complete = false;
  bool typed_dataflow_bindings_complete = false;
  bool logical_publication_design_complete = false;
  bool physical_execution_design_complete = false;
  bool declared_exact_intent_complete = false;
  bool native_provider_design_complete = false;
  bool aot_tactic_design_complete = false;
  bool physical_group_identities_complete = false;
  bool unified_identity_namespace_complete = false;
  bool expected_physical_work_complete = false;
  bool logical_physical_ownership_complete = false;
  bool forbidden_boundaries_satisfied = false;
  bool qualification_boundary_satisfied = false;
  bool operator_bindings_complete = false;
  bool canonical_design_complete = false;
  bool descriptor_schema_complete = false;
  bool schema_admission_compiled =
      kSm87AotPrefillSystemSchemaAdmissionCompiled;

  [[nodiscard]] constexpr bool descriptor_schema_available() const noexcept {
    return schema_admission_compiled && descriptor_schema_complete;
  }

  // A separate bound execution authority must authenticate artifacts, bind
  // real handles, connect Engine/runner state, and qualify accuracy.
  [[nodiscard]] constexpr bool executable() const noexcept { return false; }
};

[[nodiscard]] constexpr bool has_sm87_aot_prefill_system_plan_issue(
    const Sm87AotPrefillSystemPlanValidation& validation,
    const Sm87AotPrefillSystemPlanIssue issue) noexcept {
  return (validation.issue_mask & static_cast<std::uint64_t>(issue)) != 0U;
}

[[nodiscard]] constexpr std::uint64_t sm87_aot_prefill_role_bit(
    const PrefillBindingRole role) noexcept {
  const std::size_t index = static_cast<std::size_t>(role);
  return index < kSm87AotPrefillSystemRequiredRoleCount
             ? (1ULL << index)
             : 0U;
}

[[nodiscard]] constexpr std::uint64_t sm87_aot_prefill_group_bit(
    const Sm87AotPrefillExecutionGroup group) noexcept {
  const std::size_t index = static_cast<std::size_t>(group);
  return index < kSm87AotPrefillExecutionGroupCount ? (1ULL << index) : 0U;
}

[[nodiscard]] constexpr std::string_view sm87_aot_prefill_system_plan_id(
    const kernels::Sm87TargetAotCapacityBucket bucket) noexcept {
  switch (bucket) {
    case kernels::Sm87TargetAotCapacityBucket::kP40:
      return kSm87AotPrefillSystemP40PlanId;
    case kernels::Sm87TargetAotCapacityBucket::kP60:
      return kSm87AotPrefillSystemP60PlanId;
    case kernels::Sm87TargetAotCapacityBucket::kP130:
      return kSm87AotPrefillSystemP130PlanId;
    case kernels::Sm87TargetAotCapacityBucket::kInvalid:
      return {};
  }
  return {};
}

[[nodiscard]] constexpr Sm87AotPrefillSystemCapacityPlan
sm87_aot_prefill_system_capacity_plan(
    const kernels::Sm87TargetAotCapacityBucket bucket) noexcept {
  Sm87AotPrefillSystemCapacityPlan result;
  result.prompt = kernels::sm87_target_aot_capacity_contract(bucket);
  switch (bucket) {
    case kernels::Sm87TargetAotCapacityBucket::kP40:
      result.staging_span_tokens =
          {{8'192U, 8'192U, 8'192U, 7'712U, 7'712U}};
      result.staging_span_count = 5U;
      result.selected_request_memory_bytes = 3'975'374'848ULL;
      result.conservative_request_memory_bytes = 5'324'963'840ULL;
      return result;
    case kernels::Sm87TargetAotCapacityBucket::kP60:
      result.staging_span_tokens =
          {{8'192U, 8'192U, 8'192U, 8'192U, 8'192U, 8'192U,
            5'424U, 5'424U}};
      result.staging_span_count = 8U;
      result.selected_request_memory_bytes = 5'496'014'848ULL;
      result.conservative_request_memory_bytes = 7'052'323'840ULL;
      return result;
    case kernels::Sm87TargetAotCapacityBucket::kP130:
      result.staging_span_tokens =
          {{8'192U, 8'192U, 8'192U, 8'192U, 8'192U, 8'192U,
            8'192U, 8'192U, 8'192U, 8'192U, 8'192U, 8'192U,
            8'192U, 8'192U, 7'656U, 7'656U}};
      result.staging_span_count = 16U;
      result.selected_request_memory_bytes = 10'818'254'848ULL;
      result.conservative_request_memory_bytes = 13'098'083'840ULL;
      return result;
    case kernels::Sm87TargetAotCapacityBucket::kInvalid:
      return {};
  }
  return {};
}

[[nodiscard]] constexpr bool sm87_aot_prefill_same_staging_design(
    const Sm87AotPrefillSystemCapacityPlan& left,
    const Sm87AotPrefillSystemCapacityPlan& right) noexcept {
  if (left.staging_span_count != right.staging_span_count) {
    return false;
  }
  for (std::size_t index = 0U;
       index < kSm87AotPrefillSystemMaximumStagingSpans; ++index) {
    if (left.staging_span_tokens[index] != right.staging_span_tokens[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr Sm87AotPrefillBf16AbPlan
sm87_aot_prefill_bf16_ab_plan(const std::size_t token_count) noexcept {
  const auto capacity =
      kernels::sm87_target_aot_capacity_for_witness(token_count);
  if (!capacity.valid()) {
    return {};
  }
  Sm87AotPrefillBf16AbPlan result;
  result.capacity_bucket = capacity.bucket;
  result.token_count = token_count;
  result.input_features = 5'120U;
  result.rows_per_projection = 48U;
  result.projection_count = 2U;
  result.paired_output_rows = 96U;
  result.partitions[0U] = {Sm87AotPrefillBf16AbRole::kInProjA, 0U, 48U,
                           0U, 0U};
  result.partitions[1U] = {Sm87AotPrefillBf16AbRole::kInProjB, 48U, 48U,
                           1U, 1U};
  result.tile_m = 64U;
  result.tile_n = 96U;
  result.tile_k = 64U;
  result.grid_m = (token_count + result.tile_m - 1U) / result.tile_m;
  result.k_tiles = result.input_features / result.tile_k;
  result.tail_rows = token_count % result.tile_m;
  if (result.tail_rows == 0U) {
    result.tail_rows = result.tile_m;
  }
  result.logical_pair_tasks = result.grid_m;
  result.physical_ctas = result.logical_pair_tasks;
  result.threads_per_cta = 256U;
  result.pipeline_stages = 2U;
  result.dynamic_shared_bytes = 46'080U;
  result.split_k_workspace_bytes = 0U;
  result.minimum_active_ctas_per_sm = 2U;
  result.predicated_exact_tail = true;
  result.independent_fp32_accumulators = true;
  result.independent_bf16_rne_publications = true;
  return result;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_same_bf16_ab_plan(
    const Sm87AotPrefillBf16AbPlan& left,
    const Sm87AotPrefillBf16AbPlan& right) noexcept {
  if (left.capacity_bucket != right.capacity_bucket ||
      left.token_count != right.token_count ||
      left.input_features != right.input_features ||
      left.rows_per_projection != right.rows_per_projection ||
      left.projection_count != right.projection_count ||
      left.paired_output_rows != right.paired_output_rows) {
    return false;
  }
  for (std::size_t index = 0U; index < left.partitions.size(); ++index) {
    if (left.partitions[index].role != right.partitions[index].role ||
        left.partitions[index].paired_row_offset !=
            right.partitions[index].paired_row_offset ||
        left.partitions[index].output_rows !=
            right.partitions[index].output_rows ||
        left.partitions[index].checkpoint_tensor_ordinal !=
            right.partitions[index].checkpoint_tensor_ordinal ||
        left.partitions[index].independent_publication_ordinal !=
            right.partitions[index].independent_publication_ordinal) {
      return false;
    }
  }
  return
         left.tile_m == right.tile_m && left.tile_n == right.tile_n &&
         left.tile_k == right.tile_k && left.grid_m == right.grid_m &&
         left.k_tiles == right.k_tiles &&
         left.tail_rows == right.tail_rows &&
         left.logical_pair_tasks == right.logical_pair_tasks &&
         left.physical_ctas == right.physical_ctas &&
         left.threads_per_cta == right.threads_per_cta &&
         left.pipeline_stages == right.pipeline_stages &&
         left.dynamic_shared_bytes == right.dynamic_shared_bytes &&
         left.split_k_workspace_bytes == right.split_k_workspace_bytes &&
         left.minimum_active_ctas_per_sm ==
             right.minimum_active_ctas_per_sm &&
         left.predicated_exact_tail == right.predicated_exact_tail &&
         left.independent_fp32_accumulators ==
             right.independent_fp32_accumulators &&
         left.independent_bf16_rne_publications ==
             right.independent_bf16_rne_publications &&
         left.cuda_implementation_present ==
             right.cuda_implementation_present &&
         left.static_resources_qualified == right.static_resources_qualified &&
         left.numerical_reduction_qualified ==
             right.numerical_reduction_qualified &&
         left.production_dispatch_eligible ==
             right.production_dispatch_eligible;
}

constexpr bool Sm87AotPrefillBf16AbPlan::valid() const noexcept {
  return capacity_bucket != kernels::Sm87TargetAotCapacityBucket::kInvalid &&
         kernels::sm87_target_aot_exact_witness_tokens(token_count) &&
         sm87_aot_prefill_same_bf16_ab_plan(
      *this, sm87_aot_prefill_bf16_ab_plan(token_count));
}

[[nodiscard]] constexpr Sm87AotPrefillConstituentDesign
sm87_aot_prefill_constituent_design(const std::size_t token_count) noexcept {
  Sm87AotPrefillConstituentDesign result;
  for (std::size_t index = 0U;
       index < kSm87AotPrefillSystemProjectionPlanCount; ++index) {
    result.projections[index] = kernels::sm87_target_aot_projection_plan(
        kSm87AotPrefillProjectionRoles[index], token_count);
  }
  result.bf16_ab = sm87_aot_prefill_bf16_ab_plan(token_count);
  result.attention = kernels::sm87_target_aot_attention_plan(token_count);
  result.gdn = kernels::sm87_target_aot_gdn_plan(token_count);
  return result;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_same_constituent_design(
    const Sm87AotPrefillConstituentDesign& left,
    const Sm87AotPrefillConstituentDesign& right) noexcept {
  for (std::size_t index = 0U;
       index < kSm87AotPrefillSystemProjectionPlanCount; ++index) {
    if (!left.projections[index].valid() ||
        !kernels::sm87_target_aot_same_plan(left.projections[index],
                                            right.projections[index])) {
      return false;
    }
  }
  return left.bf16_ab.valid() && right.bf16_ab.valid() &&
         sm87_aot_prefill_same_bf16_ab_plan(left.bf16_ab, right.bf16_ab) &&
         left.attention.valid() && right.attention.valid() &&
         kernels::sm87_target_aot_same_attention_plan(left.attention,
                                                      right.attention) &&
         left.gdn.valid() && right.gdn.valid() &&
         kernels::sm87_target_aot_same_gdn_plan(left.gdn, right.gdn);
}

namespace detail {

[[nodiscard]] constexpr bool checked_multiply(const std::uint64_t left,
                                              const std::uint64_t right,
                                              std::uint64_t& result) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    result = 0U;
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] constexpr bool checked_multiply3(
    const std::uint64_t first, const std::uint64_t second,
    const std::uint64_t third, std::uint64_t& result) noexcept {
  std::uint64_t intermediate = 0U;
  return checked_multiply(first, second, intermediate) &&
         checked_multiply(intermediate, third, result);
}

[[nodiscard]] constexpr bool valid_numerical_intent(
    const PrefillNumericalMode mode) noexcept {
  return mode == PrefillNumericalMode::kUnbound ||
         mode == PrefillNumericalMode::kExact ||
         mode == PrefillNumericalMode::kApproximate;
}

[[nodiscard]] constexpr bool valid_provider(
    const PrefillOperatorProvider provider) noexcept {
  return provider == PrefillOperatorProvider::kUnbound ||
         provider == PrefillOperatorProvider::kNative ||
         provider == PrefillOperatorProvider::kCuBlasLtReference ||
         provider == PrefillOperatorProvider::kExternalRuntime;
}

[[nodiscard]] constexpr bool valid_tactic_mode(
    const PrefillTacticMode mode) noexcept {
  return mode == PrefillTacticMode::kUnbound ||
         mode == PrefillTacticMode::kAheadOfTime ||
         mode == PrefillTacticMode::kJit ||
         mode == PrefillTacticMode::kRequestTimeSelected;
}

}  // namespace detail

[[nodiscard]] constexpr Sm87AotPrefillInstancePredecessor
sm87_aot_prefill_embedding_predecessor(
    const Sm87AotPrefillInstanceEdgeKind edge_kind) noexcept {
  return {Sm87AotPrefillInstanceSource::kEmbedding, edge_kind,
          kSm87AotPrefillSystemLayerCount, kSm87AotPrefillLayerStageCount,
          Sm87AotPrefillExecutionGroup::kEmbedding};
}

[[nodiscard]] constexpr Sm87AotPrefillInstancePredecessor
sm87_aot_prefill_stage_predecessor(
    const std::size_t layer_index, const std::size_t stage_index,
    const Sm87AotPrefillExecutionGroup group,
    const Sm87AotPrefillInstanceEdgeKind edge_kind) noexcept {
  return {Sm87AotPrefillInstanceSource::kLayerStage, edge_kind, layer_index,
          stage_index, group};
}

[[nodiscard]] constexpr bool sm87_aot_prefill_same_instance_predecessor(
    const Sm87AotPrefillInstancePredecessor& left,
    const Sm87AotPrefillInstancePredecessor& right) noexcept {
  return left.source == right.source && left.edge_kind == right.edge_kind &&
         left.layer_index == right.layer_index &&
         left.stage_index == right.stage_index && left.group == right.group;
}

[[nodiscard]] constexpr Sm87AotPrefillInstancePredecessor
sm87_aot_prefill_layer_input_predecessor(
    const std::size_t layer_index,
    const Sm87AotPrefillInstanceEdgeKind edge_kind) noexcept {
  return layer_index == 0U
             ? sm87_aot_prefill_embedding_predecessor(edge_kind)
             : sm87_aot_prefill_stage_predecessor(
                   layer_index - 1U, 7U,
                   Sm87AotPrefillExecutionGroup::kDownResidual, edge_kind);
}

[[nodiscard]] constexpr Sm87AotPrefillLayerSchedule
sm87_aot_prefill_layer_schedule(const std::size_t layer_index) noexcept {
  if (layer_index >= kSm87AotPrefillSystemLayerCount) {
    return {};
  }
  Sm87AotPrefillLayerSchedule result;
  result.layer_index = layer_index;
  result.input_source =
      layer_index == 0U
          ? Sm87AotPrefillLayerInputSource::kEmbedding
          : Sm87AotPrefillLayerInputSource::kPriorLayerDownResidual;
  const bool full_attention = (layer_index + 1U) % 4U == 0U;
  if (full_attention) {
    result.kind = Sm87AotPrefillLayerKind::kFullAttention;
    result.family_ordinal = layer_index / 4U;
    result.ordered_groups = {{
        Sm87AotPrefillExecutionGroup::kInputAndLayerNorm,
        Sm87AotPrefillExecutionGroup::kFullQkv,
        Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish,
        Sm87AotPrefillExecutionGroup::kAttention,
        Sm87AotPrefillExecutionGroup::kFullOResidual,
        Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm,
        Sm87AotPrefillExecutionGroup::kGateUp,
        Sm87AotPrefillExecutionGroup::kDownResidual,
    }};
  } else {
    result.kind = Sm87AotPrefillLayerKind::kGdn;
    result.family_ordinal = layer_index - layer_index / 4U;
    result.ordered_groups = {{
        Sm87AotPrefillExecutionGroup::kInputAndLayerNorm,
        Sm87AotPrefillExecutionGroup::kLinearQkvZ,
        Sm87AotPrefillExecutionGroup::kLinearAb,
        Sm87AotPrefillExecutionGroup::kGdn,
        Sm87AotPrefillExecutionGroup::kLinearOResidual,
        Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm,
        Sm87AotPrefillExecutionGroup::kGateUp,
        Sm87AotPrefillExecutionGroup::kDownResidual,
    }};
  }
  for (std::size_t stage = 0U; stage < kSm87AotPrefillLayerStageCount;
       ++stage) {
    auto& instance = result.stage_instances[stage];
    instance.layer_index = layer_index;
    instance.stage_index = stage;
    instance.group = result.ordered_groups[stage];
  }
  result.stage_instances[0U].predecessors[0U] =
      sm87_aot_prefill_layer_input_predecessor(
          layer_index, Sm87AotPrefillInstanceEdgeKind::kData);
  result.stage_instances[0U].predecessor_count = 1U;
  result.stage_instances[1U].predecessors[0U] =
      sm87_aot_prefill_stage_predecessor(
          layer_index, 0U, result.ordered_groups[0U],
          Sm87AotPrefillInstanceEdgeKind::kData);
  result.stage_instances[1U].predecessor_count = 1U;
  if (result.kind == Sm87AotPrefillLayerKind::kGdn) {
    result.stage_instances[2U].predecessors[0U] =
        sm87_aot_prefill_stage_predecessor(
            layer_index, 0U, result.ordered_groups[0U],
            Sm87AotPrefillInstanceEdgeKind::kData);
    result.stage_instances[2U].predecessor_count = 1U;
    result.stage_instances[3U].predecessors[0U] =
        sm87_aot_prefill_stage_predecessor(
            layer_index, 1U, result.ordered_groups[1U],
            Sm87AotPrefillInstanceEdgeKind::kData);
    result.stage_instances[3U].predecessors[1U] =
        sm87_aot_prefill_stage_predecessor(
            layer_index, 2U, result.ordered_groups[2U],
            Sm87AotPrefillInstanceEdgeKind::kData);
    result.stage_instances[3U].predecessor_count = 2U;
  } else {
    result.stage_instances[2U].predecessors[0U] =
        sm87_aot_prefill_stage_predecessor(
            layer_index, 1U, result.ordered_groups[1U],
            Sm87AotPrefillInstanceEdgeKind::kData);
    result.stage_instances[2U].predecessor_count = 1U;
    result.stage_instances[3U].predecessors[0U] =
        sm87_aot_prefill_stage_predecessor(
            layer_index, 2U, result.ordered_groups[2U],
            Sm87AotPrefillInstanceEdgeKind::kData);
    result.stage_instances[3U].predecessor_count = 1U;
  }
  result.stage_instances[4U].predecessors[0U] =
      sm87_aot_prefill_stage_predecessor(
          layer_index, 3U, result.ordered_groups[3U],
          Sm87AotPrefillInstanceEdgeKind::kData);
  result.stage_instances[4U].predecessors[1U] =
      sm87_aot_prefill_stage_predecessor(
          layer_index, 0U, result.ordered_groups[0U],
          Sm87AotPrefillInstanceEdgeKind::kResidualBypass);
  result.stage_instances[4U].predecessor_count = 2U;
  result.stage_instances[5U].predecessors[0U] =
      sm87_aot_prefill_stage_predecessor(
          layer_index, 4U, result.ordered_groups[4U],
          Sm87AotPrefillInstanceEdgeKind::kData);
  result.stage_instances[5U].predecessor_count = 1U;
  result.stage_instances[6U].predecessors[0U] =
      sm87_aot_prefill_stage_predecessor(
          layer_index, 5U, result.ordered_groups[5U],
          Sm87AotPrefillInstanceEdgeKind::kData);
  result.stage_instances[6U].predecessor_count = 1U;
  result.stage_instances[7U].predecessors[0U] =
      sm87_aot_prefill_stage_predecessor(
          layer_index, 6U, result.ordered_groups[6U],
          Sm87AotPrefillInstanceEdgeKind::kData);
  result.stage_instances[7U].predecessors[1U] =
      sm87_aot_prefill_stage_predecessor(
          layer_index, 4U, result.ordered_groups[4U],
          Sm87AotPrefillInstanceEdgeKind::kResidualBypass);
  result.stage_instances[7U].predecessor_count = 2U;
  return result;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_same_layer_schedule(
    const Sm87AotPrefillLayerSchedule& left,
    const Sm87AotPrefillLayerSchedule& right) noexcept {
  if (left.layer_index != right.layer_index || left.kind != right.kind ||
      left.input_source != right.input_source ||
      left.family_ordinal != right.family_ordinal) {
    return false;
  }
  for (std::size_t index = 0U; index < left.ordered_groups.size(); ++index) {
    if (left.ordered_groups[index] != right.ordered_groups[index]) {
      return false;
    }
    const auto& left_instance = left.stage_instances[index];
    const auto& right_instance = right.stage_instances[index];
    if (left_instance.layer_index != right_instance.layer_index ||
        left_instance.stage_index != right_instance.stage_index ||
        left_instance.group != right_instance.group ||
        left_instance.predecessor_count !=
            right_instance.predecessor_count) {
      return false;
    }
    for (std::size_t predecessor = 0U;
         predecessor < kSm87AotPrefillMaximumStagePredecessors;
         ++predecessor) {
      if (!sm87_aot_prefill_same_instance_predecessor(
              left_instance.predecessors[predecessor],
              right_instance.predecessors[predecessor])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] constexpr std::uint64_t
sm87_aot_prefill_group_layer_occurrences(
    const Sm87AotPrefillExecutionGroup group) noexcept {
  std::uint64_t occurrences = 0U;
  for (std::size_t layer = 0U; layer < kSm87AotPrefillSystemLayerCount;
       ++layer) {
    const auto schedule = sm87_aot_prefill_layer_schedule(layer);
    bool observed_in_layer = false;
    for (const auto scheduled_group : schedule.ordered_groups) {
      if (scheduled_group == group) {
        if (observed_in_layer) {
          return 0U;
        }
        observed_in_layer = true;
      }
    }
    if (observed_in_layer) {
      ++occurrences;
    }
  }
  return occurrences;
}

[[nodiscard]] constexpr std::uint64_t sm87_aot_prefill_predecessor_mask(
    const Sm87AotPrefillExecutionGroup group) noexcept {
  switch (group) {
    case Sm87AotPrefillExecutionGroup::kEmbedding:
      return 0U;
    case Sm87AotPrefillExecutionGroup::kInputAndLayerNorm:
      return sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kEmbedding) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kDownResidual);
    case Sm87AotPrefillExecutionGroup::kLinearQkvZ:
    case Sm87AotPrefillExecutionGroup::kLinearAb:
    case Sm87AotPrefillExecutionGroup::kFullQkv:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kInputAndLayerNorm);
    case Sm87AotPrefillExecutionGroup::kGdn:
      return sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kLinearQkvZ) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kLinearAb);
    case Sm87AotPrefillExecutionGroup::kLinearOResidual:
      return sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kGdn) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kInputAndLayerNorm);
    case Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kFullQkv);
    case Sm87AotPrefillExecutionGroup::kAttention:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish);
    case Sm87AotPrefillExecutionGroup::kFullOResidual:
      return sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kAttention) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kInputAndLayerNorm);
    case Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm:
      return sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kLinearOResidual) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kFullOResidual);
    case Sm87AotPrefillExecutionGroup::kGateUp:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm);
    case Sm87AotPrefillExecutionGroup::kDownResidual:
      return sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kGateUp) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kLinearOResidual) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kFullOResidual);
    case Sm87AotPrefillExecutionGroup::kFinalHandoff:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kDownResidual);
    case Sm87AotPrefillExecutionGroup::kCount:
    case Sm87AotPrefillExecutionGroup::kInvalid:
      return 0U;
  }
  return 0U;
}

[[nodiscard]] constexpr std::uint64_t sm87_aot_prefill_consumer_mask(
    const Sm87AotPrefillExecutionGroup group) noexcept {
  switch (group) {
    case Sm87AotPrefillExecutionGroup::kEmbedding:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kInputAndLayerNorm);
    case Sm87AotPrefillExecutionGroup::kInputAndLayerNorm:
      return sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kLinearQkvZ) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kLinearAb) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kFullQkv) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kLinearOResidual) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kFullOResidual);
    case Sm87AotPrefillExecutionGroup::kLinearQkvZ:
    case Sm87AotPrefillExecutionGroup::kLinearAb:
      return sm87_aot_prefill_group_bit(Sm87AotPrefillExecutionGroup::kGdn);
    case Sm87AotPrefillExecutionGroup::kGdn:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kLinearOResidual);
    case Sm87AotPrefillExecutionGroup::kLinearOResidual:
    case Sm87AotPrefillExecutionGroup::kFullOResidual:
      return sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kDownResidual);
    case Sm87AotPrefillExecutionGroup::kFullQkv:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish);
    case Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kAttention);
    case Sm87AotPrefillExecutionGroup::kAttention:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kFullOResidual);
    case Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kGateUp);
    case Sm87AotPrefillExecutionGroup::kGateUp:
      return sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kDownResidual);
    case Sm87AotPrefillExecutionGroup::kDownResidual:
      return sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kInputAndLayerNorm) |
             sm87_aot_prefill_group_bit(
                 Sm87AotPrefillExecutionGroup::kFinalHandoff);
    case Sm87AotPrefillExecutionGroup::kFinalHandoff:
    case Sm87AotPrefillExecutionGroup::kCount:
    case Sm87AotPrefillExecutionGroup::kInvalid:
      return 0U;
  }
  return 0U;
}

[[nodiscard]] constexpr Sm87AotPrefillLogicalPublication
sm87_aot_prefill_logical_publication(
    const PrefillBindingRole role) noexcept {
  Sm87AotPrefillLogicalPublication result;
  result.role = role;
  result.declared_numerical_intent = PrefillNumericalMode::kExact;
  switch (role) {
    case PrefillBindingRole::kNvfp4GateUp:
      result.execution_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kGateUp);
      result.expected_logical_publications =
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kGateUp);
      return result;
    case PrefillBindingRole::kNvfp4Down:
      result.execution_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kDownResidual);
      result.expected_logical_publications =
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kDownResidual);
      return result;
    case PrefillBindingRole::kLinearFp8Qkv:
    case PrefillBindingRole::kLinearFp8Z:
      result.execution_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kLinearQkvZ);
      result.expected_logical_publications =
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kLinearQkvZ);
      return result;
    case PrefillBindingRole::kLinearFp8O:
      result.execution_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kLinearOResidual);
      result.expected_logical_publications =
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kLinearOResidual);
      return result;
    case PrefillBindingRole::kFullFp8Q:
    case PrefillBindingRole::kFullFp8K:
    case PrefillBindingRole::kFullFp8V:
      result.execution_group_mask =
          sm87_aot_prefill_group_bit(
              Sm87AotPrefillExecutionGroup::kFullQkv) |
          sm87_aot_prefill_group_bit(
              Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish);
      result.expected_logical_publications =
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kFullQkv);
      return result;
    case PrefillBindingRole::kFullFp8O:
      result.execution_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kFullOResidual);
      result.expected_logical_publications =
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kFullOResidual);
      return result;
    case PrefillBindingRole::kLinearBf16A:
    case PrefillBindingRole::kLinearBf16B:
      result.execution_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kLinearAb);
      result.expected_logical_publications =
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kLinearAb);
      return result;
    case PrefillBindingRole::kExactGdn:
      result.execution_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kGdn);
      result.expected_logical_publications =
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kGdn);
      return result;
    case PrefillBindingRole::kExactCausalAttention:
      result.execution_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kAttention);
      result.expected_logical_publications =
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kAttention);
      return result;
    case PrefillBindingRole::kResidual:
      result.execution_group_mask =
          sm87_aot_prefill_group_bit(
              Sm87AotPrefillExecutionGroup::kLinearOResidual) |
          sm87_aot_prefill_group_bit(
              Sm87AotPrefillExecutionGroup::kFullOResidual) |
          sm87_aot_prefill_group_bit(
              Sm87AotPrefillExecutionGroup::kDownResidual);
      result.expected_logical_publications =
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kLinearOResidual) +
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kFullOResidual) +
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kDownResidual);
      return result;
    case PrefillBindingRole::kNormalization:
      result.execution_group_mask =
          sm87_aot_prefill_group_bit(
              Sm87AotPrefillExecutionGroup::kInputAndLayerNorm) |
          sm87_aot_prefill_group_bit(Sm87AotPrefillExecutionGroup::kGdn) |
          sm87_aot_prefill_group_bit(
              Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish) |
          sm87_aot_prefill_group_bit(
              Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm) |
          sm87_aot_prefill_group_bit(
              Sm87AotPrefillExecutionGroup::kFinalHandoff);
      result.expected_logical_publications =
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kInputAndLayerNorm) +
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kGdn) +
          2U * sm87_aot_prefill_group_layer_occurrences(
                   Sm87AotPrefillExecutionGroup::
                       kFullQkNormRopePublish) +
          sm87_aot_prefill_group_layer_occurrences(
              Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm) +
          1U;
      return result;
    case PrefillBindingRole::kEmbedding:
      result.execution_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kEmbedding);
      result.expected_logical_publications = 1U;
      return result;
    case PrefillBindingRole::kFinalHandoff:
      result.execution_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kFinalHandoff);
      result.expected_logical_publications = 1U;
      return result;
    case PrefillBindingRole::kCount:
    case PrefillBindingRole::kInvalid:
      return {};
  }
  return {};
}

[[nodiscard]] constexpr bool sm87_aot_prefill_same_logical_publication(
    const Sm87AotPrefillLogicalPublication& left,
    const Sm87AotPrefillLogicalPublication& right) noexcept {
  return left.role == right.role &&
         left.declared_numerical_intent ==
             right.declared_numerical_intent &&
         left.execution_group_mask == right.execution_group_mask &&
         left.expected_logical_publications ==
             right.expected_logical_publications;
}

[[nodiscard]] constexpr std::uint64_t sm87_aot_prefill_group_role_mask(
    const Sm87AotPrefillExecutionGroup group) noexcept {
  switch (group) {
    case Sm87AotPrefillExecutionGroup::kEmbedding:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kEmbedding);
    case Sm87AotPrefillExecutionGroup::kInputAndLayerNorm:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kNormalization);
    case Sm87AotPrefillExecutionGroup::kLinearQkvZ:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kLinearFp8Qkv) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kLinearFp8Z);
    case Sm87AotPrefillExecutionGroup::kLinearAb:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kLinearBf16A) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kLinearBf16B);
    case Sm87AotPrefillExecutionGroup::kGdn:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kExactGdn) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kNormalization);
    case Sm87AotPrefillExecutionGroup::kLinearOResidual:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kLinearFp8O) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kResidual);
    case Sm87AotPrefillExecutionGroup::kFullQkv:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kFullFp8Q) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kFullFp8K) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kFullFp8V);
    case Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kFullFp8Q) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kFullFp8K) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kFullFp8V) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kNormalization);
    case Sm87AotPrefillExecutionGroup::kAttention:
      return sm87_aot_prefill_role_bit(
          PrefillBindingRole::kExactCausalAttention);
    case Sm87AotPrefillExecutionGroup::kFullOResidual:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kFullFp8O) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kResidual);
    case Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kNormalization);
    case Sm87AotPrefillExecutionGroup::kGateUp:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kNvfp4GateUp);
    case Sm87AotPrefillExecutionGroup::kDownResidual:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kNvfp4Down) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kResidual);
    case Sm87AotPrefillExecutionGroup::kFinalHandoff:
      return sm87_aot_prefill_role_bit(PrefillBindingRole::kFinalHandoff) |
             sm87_aot_prefill_role_bit(PrefillBindingRole::kNormalization);
    case Sm87AotPrefillExecutionGroup::kCount:
    case Sm87AotPrefillExecutionGroup::kInvalid:
      return 0U;
  }
  return 0U;
}

constexpr void sm87_aot_prefill_projection_work(
    const kernels::Sm87TargetAotProjectionPlan& plan,
    const std::uint64_t layer_count,
    Sm87AotPrefillPhysicalWork& work) noexcept {
  work.expected_launches = layer_count;
  work.canonical_counts_valid =
      plan.valid() &&
      detail::checked_multiply(static_cast<std::uint64_t>(plan.mma_tile_tasks),
                               layer_count, work.projection_tile_tasks);
}

[[nodiscard]] constexpr Sm87AotPrefillPhysicalExecutionGroup
sm87_aot_prefill_execution_group(
    const Sm87AotPrefillExecutionGroup group,
    const Sm87AotPrefillConstituentDesign& constituents) noexcept {
  Sm87AotPrefillPhysicalExecutionGroup result;
  result.group = group;
  result.logical_role_mask = sm87_aot_prefill_group_role_mask(group);
  result.predecessor_group_mask = sm87_aot_prefill_predecessor_mask(group);
  result.publication_consumer_group_mask =
      sm87_aot_prefill_consumer_mask(group);
  const std::uint64_t layer_occurrences =
      sm87_aot_prefill_group_layer_occurrences(group);
  result.declared_numerical_intent = PrefillNumericalMode::kExact;
  result.provider = PrefillOperatorProvider::kNative;
  result.tactic_mode = PrefillTacticMode::kAheadOfTime;

  switch (group) {
    case Sm87AotPrefillExecutionGroup::kEmbedding:
      result.work.expected_launches = 1U;
      result.work.canonical_counts_valid = true;
      return result;
    case Sm87AotPrefillExecutionGroup::kInputAndLayerNorm:
      result.work.expected_launches = layer_occurrences;
      result.work.canonical_counts_valid = layer_occurrences != 0U;
      return result;
    case Sm87AotPrefillExecutionGroup::kLinearQkvZ:
      sm87_aot_prefill_projection_work(constituents.projections[2U],
                                       layer_occurrences, result.work);
      return result;
    case Sm87AotPrefillExecutionGroup::kLinearAb:
      result.work.expected_launches = layer_occurrences;
      result.work.canonical_counts_valid =
          constituents.bf16_ab.valid() && detail::checked_multiply(
              static_cast<std::uint64_t>(
                  constituents.bf16_ab.logical_pair_tasks),
              layer_occurrences, result.work.bf16_ab_pair_tile_tasks);
      return result;
    case Sm87AotPrefillExecutionGroup::kGdn:
      result.work.expected_launches = layer_occurrences;
      result.work.canonical_counts_valid =
          constituents.gdn.valid() &&
          detail::checked_multiply3(
              static_cast<std::uint64_t>(constituents.gdn.owner_ctas),
              static_cast<std::uint64_t>(
                  constituents.gdn.preparation_c64_macros),
              layer_occurrences,
              result.work.gdn_preparation_owner_tasks) &&
          detail::checked_multiply3(
              static_cast<std::uint64_t>(constituents.gdn.owner_ctas),
              static_cast<std::uint64_t>(constituents.gdn.exact_c16_blocks),
              layer_occurrences, result.work.gdn_exact_c16_owner_tasks);
      return result;
    case Sm87AotPrefillExecutionGroup::kLinearOResidual:
      sm87_aot_prefill_projection_work(constituents.projections[4U],
                                       layer_occurrences, result.work);
      return result;
    case Sm87AotPrefillExecutionGroup::kFullQkv:
      sm87_aot_prefill_projection_work(constituents.projections[3U],
                                       layer_occurrences, result.work);
      return result;
    case Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish:
      result.work.expected_launches = layer_occurrences;
      result.work.canonical_counts_valid =
          constituents.attention.valid() &&
          detail::checked_multiply(
              static_cast<std::uint64_t>(
                  constituents.attention.q_rmsnorm_head_rows),
              layer_occurrences,
              result.work.attention_q_rmsnorm_head_rows) &&
          detail::checked_multiply(
              static_cast<std::uint64_t>(
                  constituents.attention.k_rmsnorm_head_rows),
              layer_occurrences,
              result.work.attention_k_rmsnorm_head_rows) &&
          detail::checked_multiply(
              static_cast<std::uint64_t>(
                  constituents.attention.q_rope_head_rows),
              layer_occurrences, result.work.attention_q_rope_head_rows) &&
          detail::checked_multiply(
              static_cast<std::uint64_t>(
                  constituents.attention.k_rope_head_rows),
              layer_occurrences, result.work.attention_k_rope_head_rows) &&
          detail::checked_multiply(
              static_cast<std::uint64_t>(
                  constituents.attention.processed_q_gate_head_rows),
              layer_occurrences,
              result.work.attention_processed_q_gate_head_rows) &&
          detail::checked_multiply(
              static_cast<std::uint64_t>(
                  constituents.attention.processed_k_head_rows),
              layer_occurrences,
              result.work.attention_processed_k_head_rows) &&
          detail::checked_multiply(
              static_cast<std::uint64_t>(
                  constituents.attention.published_v_head_rows),
              layer_occurrences,
              result.work.attention_published_v_head_rows) &&
          detail::checked_multiply(
              static_cast<std::uint64_t>(
                  constituents.attention.position_rows),
              layer_occurrences, result.work.attention_position_rows);
      return result;
    case Sm87AotPrefillExecutionGroup::kAttention:
      result.work.expected_launches = layer_occurrences;
      result.work.canonical_counts_valid =
          constituents.attention.valid() && detail::checked_multiply(
              static_cast<std::uint64_t>(constituents.attention.total_ctas),
              layer_occurrences, result.work.attention_cta_tasks);
      return result;
    case Sm87AotPrefillExecutionGroup::kFullOResidual:
      sm87_aot_prefill_projection_work(constituents.projections[4U],
                                       layer_occurrences, result.work);
      return result;
    case Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm:
      result.work.expected_launches = layer_occurrences;
      result.work.canonical_counts_valid = layer_occurrences != 0U;
      return result;
    case Sm87AotPrefillExecutionGroup::kGateUp:
      sm87_aot_prefill_projection_work(constituents.projections[0U],
                                       layer_occurrences, result.work);
      return result;
    case Sm87AotPrefillExecutionGroup::kDownResidual:
      sm87_aot_prefill_projection_work(constituents.projections[1U],
                                       layer_occurrences, result.work);
      return result;
    case Sm87AotPrefillExecutionGroup::kFinalHandoff:
      result.work.expected_launches = 1U;
      result.work.canonical_counts_valid = true;
      return result;
    case Sm87AotPrefillExecutionGroup::kCount:
    case Sm87AotPrefillExecutionGroup::kInvalid:
      return {};
  }
  return {};
}

[[nodiscard]] constexpr bool sm87_aot_prefill_same_physical_work(
    const Sm87AotPrefillPhysicalWork& left,
    const Sm87AotPrefillPhysicalWork& right) noexcept {
  return left.expected_launches == right.expected_launches &&
         left.projection_tile_tasks == right.projection_tile_tasks &&
         left.bf16_ab_pair_tile_tasks == right.bf16_ab_pair_tile_tasks &&
         left.attention_cta_tasks == right.attention_cta_tasks &&
         left.attention_q_rmsnorm_head_rows ==
             right.attention_q_rmsnorm_head_rows &&
         left.attention_k_rmsnorm_head_rows ==
             right.attention_k_rmsnorm_head_rows &&
         left.attention_q_rope_head_rows ==
             right.attention_q_rope_head_rows &&
         left.attention_k_rope_head_rows ==
             right.attention_k_rope_head_rows &&
         left.attention_processed_q_gate_head_rows ==
             right.attention_processed_q_gate_head_rows &&
         left.attention_processed_k_head_rows ==
             right.attention_processed_k_head_rows &&
         left.attention_published_v_head_rows ==
             right.attention_published_v_head_rows &&
         left.attention_position_rows == right.attention_position_rows &&
         left.gdn_preparation_owner_tasks ==
             right.gdn_preparation_owner_tasks &&
         left.gdn_exact_c16_owner_tasks ==
             right.gdn_exact_c16_owner_tasks &&
         left.canonical_counts_valid == right.canonical_counts_valid;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_same_execution_group_design(
    const Sm87AotPrefillPhysicalExecutionGroup& left,
    const Sm87AotPrefillPhysicalExecutionGroup& right) noexcept {
  return left.group == right.group &&
         left.logical_role_mask == right.logical_role_mask &&
         left.predecessor_group_mask == right.predecessor_group_mask &&
         left.publication_consumer_group_mask ==
             right.publication_consumer_group_mask &&
         left.declared_numerical_intent ==
             right.declared_numerical_intent &&
         left.provider == right.provider &&
         left.tactic_mode == right.tactic_mode &&
         sm87_aot_prefill_same_physical_work(left.work, right.work) &&
         left.uses_mtp == right.uses_mtp &&
         left.allows_fallback == right.allows_fallback;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_tactic_alias_allowed(
    const Sm87AotPrefillExecutionGroup left,
    const Sm87AotPrefillExecutionGroup right) noexcept {
  return (left == Sm87AotPrefillExecutionGroup::kLinearOResidual &&
          right == Sm87AotPrefillExecutionGroup::kFullOResidual) ||
         (left == Sm87AotPrefillExecutionGroup::kFullOResidual &&
          right == Sm87AotPrefillExecutionGroup::kLinearOResidual);
}

[[nodiscard]] constexpr Sm87AotPrefillSystemPlan
make_unbound_sm87_aot_prefill_system_plan(
    const kernels::Sm87TargetAotCapacityBucket bucket) noexcept {
  const kernels::Sm87TargetAotCapacityContract prompt =
      kernels::sm87_target_aot_capacity_contract(bucket);
  if (!prompt.valid()) {
    return {};
  }

  Sm87AotPrefillSystemPlan plan;
  plan.candidate_id = kSm87AotPrefillSystemCandidateId;
  plan.deployment_plan_id = sm87_aot_prefill_system_plan_id(bucket);
  plan.capacity = sm87_aot_prefill_system_capacity_plan(bucket);
  plan.constituents =
      sm87_aot_prefill_constituent_design(prompt.witness_prompt_tokens);
  plan.layer_schedule_count = kSm87AotPrefillSystemLayerCount;
  for (std::size_t index = 0U; index < kSm87AotPrefillSystemLayerCount;
       ++index) {
    plan.layer_schedule[index] = sm87_aot_prefill_layer_schedule(index);
  }
  plan.final_handoff_instance.group =
      Sm87AotPrefillExecutionGroup::kFinalHandoff;
  plan.final_handoff_instance.final_layer_predecessor =
      sm87_aot_prefill_stage_predecessor(
          kSm87AotPrefillSystemLayerCount - 1U,
          kSm87AotPrefillLayerStageCount - 1U,
          Sm87AotPrefillExecutionGroup::kDownResidual,
          Sm87AotPrefillInstanceEdgeKind::kData);
  plan.attention_dataflow.raw_producer_group =
      Sm87AotPrefillExecutionGroup::kFullQkv;
  plan.attention_dataflow.preprocess_group =
      Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish;
  plan.attention_dataflow.consumer_group =
      Sm87AotPrefillExecutionGroup::kAttention;
  plan.attention_dataflow.first_position = 0U;
  plan.attention_dataflow.initial_kv_length = 0U;
  plan.attention_dataflow.kv_capacity_tokens =
      prompt.request_capacity_tokens;
  plan.attention_dataflow.max_position_embeddings = 262'144U;
  plan.attention_dataflow.rotary_elements =
      kernels::kSm87TargetAotAttentionRotaryElements;
  plan.attention_dataflow.rotary_pairs =
      kernels::kSm87TargetAotAttentionRotaryPairs;
  plan.attention_dataflow.rms_epsilon_fp32_bits =
      kernels::kSm87TargetAotAttentionRmsEpsilonFp32Bits;
  plan.attention_dataflow.attention_scale_fp32_bits =
      kernels::kSm87TargetAotAttentionScaleFp32Bits;
  plan.attention_dataflow.rope_theta =
      kernels::kSm87TargetAotAttentionRopeTheta;
  plan.attention_dataflow.output_tokens = prompt.witness_prompt_tokens;
  plan.attention_dataflow.output_features = 6'144U;
  plan.attention_dataflow.rope_cache_is_fp32 = true;
  plan.attention_dataflow.raw_q_gate_is_per_head_interleaved = true;
  plan.attention_dataflow.gate_is_bit_exact_split_copy = true;
  plan.attention_dataflow.processed_kv_is_nhd_transaction_staged_unpublished =
      true;
  plan.attention_dataflow.pre_gate_output_is_bf16 = true;
  plan.attention_dataflow.gated_output_is_bf16 = true;
  plan.gdn_dataflow.qkvz_producer_group =
      Sm87AotPrefillExecutionGroup::kLinearQkvZ;
  plan.gdn_dataflow.ab_producer_group =
      Sm87AotPrefillExecutionGroup::kLinearAb;
  plan.gdn_dataflow.consumer_group = Sm87AotPrefillExecutionGroup::kGdn;
  plan.gdn_dataflow.l2_norm_epsilon = 1.0e-6F;
  plan.gdn_dataflow.output_norm_epsilon = 1.0e-6F;
  plan.gdn_dataflow.first_position = 0U;
  plan.gdn_dataflow.initial_state_is_zero = true;
  plan.gdn_dataflow.initial_conv_history_is_zero = true;
  plan.gdn_dataflow.reads_initial_state_from_dram = false;
  plan.gdn_dataflow.reads_initial_conv_history_from_dram = false;
  plan.gdn_dataflow.uses_ordinary_kernel_completion = true;
  plan.gdn_dataflow.permits_per_owner_commit = false;
  plan.gdn_dataflow.state_transaction_spans_prebound_at_request_admission =
      true;
  plan.gdn_dataflow.requires_cpu_callback_or_host_sync = false;
  plan.model_dataflow.centered_rms_epsilon = 1.0e-6F;
  plan.model_dataflow.layer_residual_is_bit_exact_zero_copy_publication =
      true;
  plan.model_dataflow.bf16_a_role = Sm87AotPrefillBf16AbRole::kInProjA;
  plan.model_dataflow.bf16_b_role = Sm87AotPrefillBf16AbRole::kInProjB;
  for (std::size_t index = 0U;
       index < kSm87AotPrefillSystemProjectionDataflowCount; ++index) {
    auto& binding = plan.projection_dataflows[index];
    const std::size_t constituent_index =
        index < kSm87AotPrefillSystemProjectionPlanCount ? index : 4U;
    binding.role = kSm87AotPrefillProjectionRoles[constituent_index];
    binding.family = Sm87AotPrefillProjectionFamily::kAllLayers;
    const auto& projection = plan.constituents.projections[constituent_index];
    binding.partition_count = projection.partition_count;
    for (std::size_t partition_index = 0U;
         partition_index < projection.partition_count; ++partition_index) {
      binding.partitions[partition_index].logical_role =
          projection.partitions[partition_index].role;
      binding.partitions[partition_index].output_offset =
          projection.partitions[partition_index].output_offset;
      binding.partitions[partition_index].output_features =
          projection.partitions[partition_index].output_features;
    }
    switch (binding.role) {
      case kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp:
        binding.execution_group_mask = sm87_aot_prefill_group_bit(
            Sm87AotPrefillExecutionGroup::kGateUp);
        break;
      case kernels::Sm87TargetAotProjectionRole::kNvFp4Down:
        binding.execution_group_mask = sm87_aot_prefill_group_bit(
            Sm87AotPrefillExecutionGroup::kDownResidual);
        break;
      case kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ:
        binding.execution_group_mask = sm87_aot_prefill_group_bit(
            Sm87AotPrefillExecutionGroup::kLinearQkvZ);
        break;
      case kernels::Sm87TargetAotProjectionRole::kFp8FullQkv:
        binding.execution_group_mask = sm87_aot_prefill_group_bit(
            Sm87AotPrefillExecutionGroup::kFullQkv);
        break;
      case kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput:
        if (index == 4U) {
          binding.family = Sm87AotPrefillProjectionFamily::kGdnLayers;
          binding.execution_group_mask = sm87_aot_prefill_group_bit(
              Sm87AotPrefillExecutionGroup::kLinearOResidual);
        } else {
          binding.family =
              Sm87AotPrefillProjectionFamily::kFullAttentionLayers;
          binding.execution_group_mask = sm87_aot_prefill_group_bit(
              Sm87AotPrefillExecutionGroup::kFullOResidual);
        }
        break;
      case kernels::Sm87TargetAotProjectionRole::kInvalid:
      case kernels::Sm87TargetAotProjectionRole::kCount:
        break;
    }
  }
  plan.request_transaction.kv_layer_publication_count =
      kSm87AotPrefillSystemFullAttentionLayerCount;
  plan.request_transaction.conv_history_publication_count =
      kSm87AotPrefillSystemGdnLayerCount;
  plan.request_transaction.recurrent_state_publication_count =
      kSm87AotPrefillSystemGdnLayerCount;
  plan.request_transaction.position_rope_epoch_publication_count = 1U;
  plan.request_transaction.final_hidden_publication_count = 1U;
  plan.request_transaction.transaction_commit_count = 1U;
  plan.request_transaction.prefill_state_committed_receipt_count = 1U;
  plan.request_transaction.one_request_wide_commit = true;
  plan.request_transaction.commit_waits_for_all_component_ready_events = true;
  plan.request_transaction.commit_is_owned_by_final_handoff = true;
  plan.request_transaction.decode_visible_only_after_prefill_state_committed =
      true;
  plan.request_transaction.cancellation_discards_all_unpublished = true;
  plan.typed_resource_edge_count = kSm87AotPrefillTypedResourceEdgeCount;
  plan.typed_resource_edges = {{
      {Sm87AotPrefillTypedResource::kTokenIdsToEmbedding,
       Sm87AotPrefillExecutionGroup::kEmbedding,
       Sm87AotPrefillExecutionGroup::kEmbedding},
      {Sm87AotPrefillTypedResource::kEmbeddingToLayerZeroInput,
       Sm87AotPrefillExecutionGroup::kEmbedding,
       Sm87AotPrefillExecutionGroup::kInputAndLayerNorm},
      {Sm87AotPrefillTypedResource::kPriorDownToLayerInput,
       Sm87AotPrefillExecutionGroup::kDownResidual,
       Sm87AotPrefillExecutionGroup::kInputAndLayerNorm},
      {Sm87AotPrefillTypedResource::kLayerInputToInputNorm,
       Sm87AotPrefillExecutionGroup::kInputAndLayerNorm,
       Sm87AotPrefillExecutionGroup::kInputAndLayerNorm},
      {Sm87AotPrefillTypedResource::kInputNormToGdnQkvProjection,
       Sm87AotPrefillExecutionGroup::kInputAndLayerNorm,
       Sm87AotPrefillExecutionGroup::kLinearQkvZ},
      {Sm87AotPrefillTypedResource::kInputNormToGdnAbProjection,
       Sm87AotPrefillExecutionGroup::kInputAndLayerNorm,
       Sm87AotPrefillExecutionGroup::kLinearAb},
      {Sm87AotPrefillTypedResource::kInputNormToFullQkvProjection,
       Sm87AotPrefillExecutionGroup::kInputAndLayerNorm,
       Sm87AotPrefillExecutionGroup::kFullQkv},
      {Sm87AotPrefillTypedResource::kLayerResidualToGdnOResidual,
       Sm87AotPrefillExecutionGroup::kInputAndLayerNorm,
       Sm87AotPrefillExecutionGroup::kLinearOResidual},
      {Sm87AotPrefillTypedResource::kLayerResidualToFullOResidual,
       Sm87AotPrefillExecutionGroup::kInputAndLayerNorm,
       Sm87AotPrefillExecutionGroup::kFullOResidual},
      {Sm87AotPrefillTypedResource::kGateToSilu,
       Sm87AotPrefillExecutionGroup::kGateUp,
       Sm87AotPrefillExecutionGroup::kGateUp},
      {Sm87AotPrefillTypedResource::kUpToMultiply,
       Sm87AotPrefillExecutionGroup::kGateUp,
       Sm87AotPrefillExecutionGroup::kGateUp},
      {Sm87AotPrefillTypedResource::kSiluTimesUpToDownProjection,
       Sm87AotPrefillExecutionGroup::kGateUp,
       Sm87AotPrefillExecutionGroup::kDownResidual},
      {Sm87AotPrefillTypedResource::kGdnQkvToCore,
       Sm87AotPrefillExecutionGroup::kLinearQkvZ,
       Sm87AotPrefillExecutionGroup::kGdn},
      {Sm87AotPrefillTypedResource::kGdnZToCore,
       Sm87AotPrefillExecutionGroup::kLinearQkvZ,
       Sm87AotPrefillExecutionGroup::kGdn},
      {Sm87AotPrefillTypedResource::kFullQGateToPreprocess,
       Sm87AotPrefillExecutionGroup::kFullQkv,
       Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish},
      {Sm87AotPrefillTypedResource::kFullKToPreprocess,
       Sm87AotPrefillExecutionGroup::kFullQkv,
       Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish},
      {Sm87AotPrefillTypedResource::kFullVToPreprocess,
       Sm87AotPrefillExecutionGroup::kFullQkv,
       Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish},
      {Sm87AotPrefillTypedResource::kBf16AToGdn,
       Sm87AotPrefillExecutionGroup::kLinearAb,
       Sm87AotPrefillExecutionGroup::kGdn},
      {Sm87AotPrefillTypedResource::kBf16BToGdn,
       Sm87AotPrefillExecutionGroup::kLinearAb,
       Sm87AotPrefillExecutionGroup::kGdn},
      {Sm87AotPrefillTypedResource::kProcessedQGateToAttention,
       Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish,
       Sm87AotPrefillExecutionGroup::kAttention},
      {Sm87AotPrefillTypedResource::kProcessedKToAttention,
       Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish,
       Sm87AotPrefillExecutionGroup::kAttention},
      {Sm87AotPrefillTypedResource::kProcessedVToAttention,
       Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish,
       Sm87AotPrefillExecutionGroup::kAttention},
      {Sm87AotPrefillTypedResource::kAttentionGatedToOutputProjection,
       Sm87AotPrefillExecutionGroup::kAttention,
       Sm87AotPrefillExecutionGroup::kFullOResidual},
      {Sm87AotPrefillTypedResource::kGdnOutputToOutputProjection,
       Sm87AotPrefillExecutionGroup::kGdn,
       Sm87AotPrefillExecutionGroup::kLinearOResidual},
      {Sm87AotPrefillTypedResource::kGdnOProjectionToResidual,
       Sm87AotPrefillExecutionGroup::kLinearOResidual,
       Sm87AotPrefillExecutionGroup::kLinearOResidual},
      {Sm87AotPrefillTypedResource::kFullOProjectionToResidual,
       Sm87AotPrefillExecutionGroup::kFullOResidual,
       Sm87AotPrefillExecutionGroup::kFullOResidual},
      {Sm87AotPrefillTypedResource::kGdnResidualToPostNorm,
       Sm87AotPrefillExecutionGroup::kLinearOResidual,
       Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm},
      {Sm87AotPrefillTypedResource::kFullResidualToPostNorm,
       Sm87AotPrefillExecutionGroup::kFullOResidual,
       Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm},
      {Sm87AotPrefillTypedResource::kPostNormToGateUp,
       Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm,
       Sm87AotPrefillExecutionGroup::kGateUp},
      {Sm87AotPrefillTypedResource::kDownProjectionToResidual,
       Sm87AotPrefillExecutionGroup::kDownResidual,
       Sm87AotPrefillExecutionGroup::kDownResidual},
      {Sm87AotPrefillTypedResource::kGdnResidualToDownResidual,
       Sm87AotPrefillExecutionGroup::kLinearOResidual,
       Sm87AotPrefillExecutionGroup::kDownResidual},
      {Sm87AotPrefillTypedResource::kFullResidualToDownResidual,
       Sm87AotPrefillExecutionGroup::kFullOResidual,
       Sm87AotPrefillExecutionGroup::kDownResidual},
      {Sm87AotPrefillTypedResource::kLastDownResidualToFinalNorm,
       Sm87AotPrefillExecutionGroup::kDownResidual,
       Sm87AotPrefillExecutionGroup::kFinalHandoff},
      {Sm87AotPrefillTypedResource::kKvStateToRequestTransaction,
       Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish,
       Sm87AotPrefillExecutionGroup::kFinalHandoff},
      {Sm87AotPrefillTypedResource::kConvHistoryToRequestTransaction,
       Sm87AotPrefillExecutionGroup::kGdn,
       Sm87AotPrefillExecutionGroup::kFinalHandoff},
      {Sm87AotPrefillTypedResource::kRecurrentStateToRequestTransaction,
       Sm87AotPrefillExecutionGroup::kGdn,
       Sm87AotPrefillExecutionGroup::kFinalHandoff},
      {Sm87AotPrefillTypedResource::kPositionRopeEpochToRequestTransaction,
       Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish,
       Sm87AotPrefillExecutionGroup::kFinalHandoff},
      {Sm87AotPrefillTypedResource::kFinalHiddenToRequestTransaction,
       Sm87AotPrefillExecutionGroup::kFinalHandoff,
       Sm87AotPrefillExecutionGroup::kFinalHandoff},
      {Sm87AotPrefillTypedResource::kCommittedReceiptToDecodeHandoff,
       Sm87AotPrefillExecutionGroup::kFinalHandoff,
       Sm87AotPrefillExecutionGroup::kFinalHandoff},
  }};
  plan.typed_event_edge_count = kSm87AotPrefillTypedEventEdgeCount;
  plan.typed_event_edges = {{
      {Sm87AotPrefillTypedResource::kGdnQkvToCore},
      {Sm87AotPrefillTypedResource::kFullQGateToPreprocess},
      {Sm87AotPrefillTypedResource::kFullKToPreprocess},
      {Sm87AotPrefillTypedResource::kFullVToPreprocess},
      {Sm87AotPrefillTypedResource::kBf16AToGdn},
      {Sm87AotPrefillTypedResource::kBf16BToGdn},
      {Sm87AotPrefillTypedResource::kProcessedQGateToAttention},
      {Sm87AotPrefillTypedResource::kAttentionGatedToOutputProjection},
      {Sm87AotPrefillTypedResource::kGdnOutputToOutputProjection},
      {Sm87AotPrefillTypedResource::kKvStateToRequestTransaction},
      {Sm87AotPrefillTypedResource::kConvHistoryToRequestTransaction},
      {Sm87AotPrefillTypedResource::kPositionRopeEpochToRequestTransaction},
      {Sm87AotPrefillTypedResource::kFinalHiddenToRequestTransaction},
  }};
  plan.logical_publication_count = kSm87AotPrefillSystemRequiredRoleCount;
  for (std::size_t index = 0U;
       index < kSm87AotPrefillSystemRequiredRoleCount; ++index) {
    plan.logical_publications[index] = sm87_aot_prefill_logical_publication(
        static_cast<PrefillBindingRole>(index));
  }
  plan.execution_group_count = kSm87AotPrefillExecutionGroupCount;
  for (std::size_t index = 0U;
       index < kSm87AotPrefillExecutionGroupCount; ++index) {
    plan.execution_groups[index] = sm87_aot_prefill_execution_group(
        static_cast<Sm87AotPrefillExecutionGroup>(index), plan.constituents);
  }
  plan.qualification =
      Sm87AotPrefillSystemQualification::kAccuracyUnqualified;
  return plan;
}

[[nodiscard]] constexpr const Sm87AotPrefillProjectionPartitionBinding*
sm87_aot_prefill_projection_partition(
    const Sm87AotPrefillSystemPlan& plan,
    const kernels::Sm87TargetAotLogicalRole role) noexcept {
  for (const auto& projection : plan.projection_dataflows) {
    for (std::size_t index = 0U; index < projection.partition_count &&
                                  index < projection.partitions.size();
         ++index) {
      if (projection.partitions[index].logical_role == role) {
        return &projection.partitions[index];
      }
    }
  }
  return nullptr;
}

[[nodiscard]] constexpr const Sm87AotPrefillProjectionDataflowBinding*
sm87_aot_prefill_projection_binding(
    const Sm87AotPrefillSystemPlan& plan,
    const kernels::Sm87TargetAotProjectionRole role,
    const Sm87AotPrefillProjectionFamily family =
        Sm87AotPrefillProjectionFamily::kAllLayers) noexcept {
  for (const auto& projection : plan.projection_dataflows) {
    if (projection.role == role && projection.family == family) {
      return &projection;
    }
  }
  return nullptr;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_attention_dataflow_complete(
    const Sm87AotPrefillAttentionDataflowBinding& binding,
    const std::size_t token_count,
    const std::size_t request_capacity_tokens) noexcept {
  const std::array<std::uint64_t, 31U> identities{{
      binding.raw_q_gate_input_span_table_identity,
      binding.raw_k_input_span_table_identity,
      binding.raw_v_input_span_table_identity,
      binding.raw_q_gate_ready_event_table_identity,
      binding.raw_k_ready_event_table_identity,
      binding.raw_v_ready_event_table_identity,
      binding.q_norm_weight_table_identity,
      binding.k_norm_weight_table_identity,
      binding.rope_position_contract_identity,
      binding.processed_q_gate_publication_table_identity,
      binding.processed_k_publication_table_identity,
      binding.processed_v_publication_table_identity,
      binding.preparation_ready_event_table_identity,
      binding.attention_q_gate_input_span_table_identity,
      binding.attention_k_input_span_table_identity,
      binding.attention_v_input_span_table_identity,
      binding.attention_core_input_ready_event_table_identity,
      binding.kv_cache_arena_table_identity,
      binding.kv_cache_lifetime_contract_identity,
      binding.staged_kv_transaction_publication_table_identity,
      binding.staged_kv_transaction_ready_event_table_identity,
      binding.position_rope_epoch_publication_table_identity,
      binding.position_rope_epoch_ready_event_table_identity,
      binding.pre_gate_bf16_output_span_table_identity,
      binding.pre_gate_bf16_publication_table_identity,
      binding.pre_gate_bf16_lifetime_contract_identity,
      binding.pre_gate_bf16_completion_event_table_identity,
      binding.gated_output_span_table_identity,
      binding.gated_output_publication_table_identity,
      binding.gated_output_lifetime_contract_identity,
      binding.gated_output_completion_event_table_identity,
  }};
  bool all_identities_bound = true;
  for (const auto identity : identities) {
    all_identities_bound = all_identities_bound && identity != 0U;
  }
  return binding.raw_producer_group ==
             Sm87AotPrefillExecutionGroup::kFullQkv &&
         binding.preprocess_group ==
             Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish &&
         binding.consumer_group ==
             Sm87AotPrefillExecutionGroup::kAttention &&
         binding.first_position == 0U && binding.initial_kv_length == 0U &&
         binding.kv_capacity_tokens == request_capacity_tokens &&
         binding.max_position_embeddings == 262'144U &&
         binding.rotary_elements ==
             kernels::kSm87TargetAotAttentionRotaryElements &&
         binding.rotary_pairs ==
             kernels::kSm87TargetAotAttentionRotaryPairs &&
         binding.rms_epsilon_fp32_bits ==
             kernels::kSm87TargetAotAttentionRmsEpsilonFp32Bits &&
         binding.attention_scale_fp32_bits ==
             kernels::kSm87TargetAotAttentionScaleFp32Bits &&
         binding.rope_theta == kernels::kSm87TargetAotAttentionRopeTheta &&
         binding.output_tokens == token_count &&
         binding.output_features == 6'144U &&
         binding.rope_cache_is_fp32 &&
         binding.raw_q_gate_is_per_head_interleaved &&
         binding.gate_is_bit_exact_split_copy &&
         binding.processed_kv_is_nhd_transaction_staged_unpublished &&
         binding.pre_gate_output_is_bf16 && binding.gated_output_is_bf16 &&
         all_identities_bound;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_gdn_dataflow_complete(
    const Sm87AotPrefillGdnDataflowBinding& binding) noexcept {
  const std::array<std::uint64_t, 21U> identities{{
      binding.raw_qkv_input_span_table_identity,
      binding.raw_z_input_span_table_identity,
      binding.raw_qkvz_ready_event_table_identity,
      binding.a_input_span_table_identity,
      binding.a_ready_event_table_identity,
      binding.b_input_span_table_identity,
      binding.b_ready_event_table_identity,
      binding.conv_weight_table_identity,
      binding.a_log_table_identity,
      binding.dt_bias_table_identity,
      binding.output_norm_weight_table_identity,
      binding.raw_bf16_publication_table_identity,
      binding.norm_silu_z_publication_table_identity,
      binding.output_span_table_identity,
      binding.output_publication_table_identity,
      binding.final_conv_history_publication_table_identity,
      binding.final_recurrent_state_publication_table_identity,
      binding.ordinary_kernel_completion_event_table_identity,
      binding.post_kernel_stream_ordered_state_ready_receipt_identity,
      binding.request_transaction_lifetime_contract_identity,
      binding.reset_zero_epoch_table_identity,
  }};
  return binding.qkvz_producer_group ==
             Sm87AotPrefillExecutionGroup::kLinearQkvZ &&
         binding.ab_producer_group ==
             Sm87AotPrefillExecutionGroup::kLinearAb &&
         binding.consumer_group == Sm87AotPrefillExecutionGroup::kGdn &&
         binding.l2_norm_epsilon == 1.0e-6F &&
         binding.output_norm_epsilon == 1.0e-6F &&
         binding.first_position == 0U && binding.initial_state_is_zero &&
         binding.initial_conv_history_is_zero &&
         !binding.reads_initial_state_from_dram &&
         !binding.reads_initial_conv_history_from_dram &&
         binding.uses_ordinary_kernel_completion &&
         !binding.permits_per_owner_commit &&
         binding.state_transaction_spans_prebound_at_request_admission &&
         !binding.requires_cpu_callback_or_host_sync &&
         kernels::sm87_target_aot_nonzero_unique_identities(identities);
}

[[nodiscard]] constexpr bool sm87_aot_prefill_projection_dataflow_complete(
    const Sm87AotPrefillProjectionDataflowBinding& binding,
    const kernels::Sm87TargetAotProjectionPlan& expected_plan) noexcept {
  const auto expected_role = expected_plan.role;
  std::uint64_t expected_group_mask = 0U;
  switch (expected_role) {
    case kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp:
      expected_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kGateUp);
      break;
    case kernels::Sm87TargetAotProjectionRole::kNvFp4Down:
      expected_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kDownResidual);
      break;
    case kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ:
      expected_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kLinearQkvZ);
      break;
    case kernels::Sm87TargetAotProjectionRole::kFp8FullQkv:
      expected_group_mask = sm87_aot_prefill_group_bit(
          Sm87AotPrefillExecutionGroup::kFullQkv);
      break;
    case kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput:
      if (binding.family == Sm87AotPrefillProjectionFamily::kGdnLayers) {
        expected_group_mask = sm87_aot_prefill_group_bit(
            Sm87AotPrefillExecutionGroup::kLinearOResidual);
      } else if (binding.family ==
                 Sm87AotPrefillProjectionFamily::kFullAttentionLayers) {
        expected_group_mask = sm87_aot_prefill_group_bit(
            Sm87AotPrefillExecutionGroup::kFullOResidual);
      } else {
        return false;
      }
      break;
    case kernels::Sm87TargetAotProjectionRole::kInvalid:
    case kernels::Sm87TargetAotProjectionRole::kCount:
      return false;
  }
  if (binding.role != expected_role ||
      (expected_role !=
           kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput &&
       binding.family != Sm87AotPrefillProjectionFamily::kAllLayers) ||
      binding.execution_group_mask != expected_group_mask ||
      binding.partition_count != expected_plan.partition_count ||
      binding.input_span_table_identity == 0U) {
    return false;
  }
  const bool needs_input_ready_event =
      expected_role ==
      kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput;
  const bool needs_aggregate_completion =
      expected_role == kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
  if ((binding.input_ready_event_table_identity != 0U) !=
          needs_input_ready_event ||
      (binding.aggregate_completion_event_table_identity != 0U) !=
          needs_aggregate_completion) {
    return false;
  }
  for (std::size_t index = 0U; index < binding.partitions.size(); ++index) {
    const auto& partition = binding.partitions[index];
    const auto& expected = expected_plan.partitions[index];
    if (partition.logical_role != expected.role ||
        partition.output_offset != expected.output_offset ||
        partition.output_features != expected.output_features) {
      return false;
    }
    if (index < binding.partition_count) {
      const std::array<std::uint64_t, 6U> identities{{
          partition.source_packed_asset_identity,
          partition.independent_scale_identity,
          partition.raw_bits_contract_identity,
          partition.output_span_identity,
          partition.publication_identity,
          partition.completion_event_identity,
      }};
      if (!kernels::sm87_target_aot_nonzero_unique_identities(identities)) {
        return false;
      }
    } else if (partition.source_packed_asset_identity != 0U ||
               partition.independent_scale_identity != 0U ||
               partition.raw_bits_contract_identity != 0U ||
               partition.output_span_identity != 0U ||
               partition.publication_identity != 0U ||
               partition.completion_event_identity != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_model_dataflow_complete(
    const Sm87AotPrefillModelDataflowBinding& binding) noexcept {
  const std::array identities{
      binding.token_id_span_identity,
      binding.embedding_token_input_span_identity,
      binding.embedding_weight_table_identity,
      binding.embedding_publication_table_identity,
      binding.layer_input_span_table_identity,
      binding.input_norm_input_span_table_identity,
      binding.input_norm_weight_table_identity,
      binding.input_norm_publication_table_identity,
      binding.input_norm_epsilon_contract_identity,
      binding.gdn_o_residual_bypass_input_span_identity,
      binding.full_o_residual_bypass_input_span_identity,
      binding.gdn_o_projection_residual_input_span_identity,
      binding.full_o_projection_residual_input_span_identity,
      binding.gdn_attention_residual_publication_identity,
      binding.full_attention_residual_publication_identity,
      binding.gdn_post_norm_input_span_identity,
      binding.full_post_norm_input_span_identity,
      binding.gdn_down_residual_bypass_input_span_identity,
      binding.full_down_residual_bypass_input_span_identity,
      binding.post_attention_norm_weight_table_identity,
      binding.post_attention_norm_publication_table_identity,
      binding.post_attention_norm_epsilon_contract_identity,
      binding.silu_gate_input_span_table_identity,
      binding.silu_up_input_span_table_identity,
      binding.silu_times_up_publication_table_identity,
      binding.down_branch_residual_input_span_identity,
      binding.next_layer_input_publication_table_identity,
      binding.residual_lifetime_contract_identity,
      binding.bf16_ab_input_span_table_identity,
      binding.bf16_a_weight_table_identity,
      binding.bf16_b_weight_table_identity,
      binding.bf16_a_publication_table_identity,
      binding.bf16_b_publication_table_identity,
      binding.bf16_a_completion_event_table_identity,
      binding.bf16_b_completion_event_table_identity,
      binding.bf16_a_rne_publication_event_identity,
      binding.bf16_b_rne_publication_event_identity,
      binding.final_norm_weight_identity,
      binding.final_norm_input_span_identity,
      binding.final_norm_publication_identity,
      binding.final_norm_completion_event_identity,
      binding.final_norm_epsilon_contract_identity,
      binding.final_handoff_identity,
  };
  bool all_identities_bound = true;
  for (const auto identity : identities) {
    all_identities_bound = all_identities_bound && identity != 0U;
  }
  return binding.bf16_a_role == Sm87AotPrefillBf16AbRole::kInProjA &&
         binding.bf16_b_role == Sm87AotPrefillBf16AbRole::kInProjB &&
         binding.centered_rms_epsilon == 1.0e-6F &&
         binding.layer_residual_is_bit_exact_zero_copy_publication &&
         all_identities_bound;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_request_transaction_complete(
    const Sm87AotPrefillRequestTransactionBinding& binding) noexcept {
  return binding.kv_layer_publication_count ==
             kSm87AotPrefillSystemFullAttentionLayerCount &&
         binding.conv_history_publication_count ==
             kSm87AotPrefillSystemGdnLayerCount &&
         binding.recurrent_state_publication_count ==
             kSm87AotPrefillSystemGdnLayerCount &&
         binding.position_rope_epoch_publication_count == 1U &&
         binding.final_hidden_publication_count == 1U &&
         binding.transaction_commit_count == 1U &&
         binding.prefill_state_committed_receipt_count == 1U &&
         binding.staged_kv_input_table_identity != 0U &&
         binding.conv_history_input_table_identity != 0U &&
         binding.recurrent_state_input_table_identity != 0U &&
         binding.position_rope_epoch_input_identity != 0U &&
         binding.final_hidden_input_identity != 0U &&
         binding.transaction_lifetime_contract_identity != 0U &&
         binding.prefill_state_committed_receipt_schema_identity != 0U &&
         binding.prefill_state_committed_receipt_identity != 0U &&
         binding.decode_handoff_receipt_input_identity != 0U &&
         binding.staged_kv_ready_input_event_identity != 0U &&
         binding.gdn_state_ready_input_event_identity != 0U &&
         binding.position_rope_epoch_ready_input_event_identity != 0U &&
         binding.final_hidden_ready_input_event_identity != 0U &&
         binding.one_request_wide_commit &&
         binding.commit_waits_for_all_component_ready_events &&
         binding.commit_is_owned_by_final_handoff &&
         binding.decode_visible_only_after_prefill_state_committed &&
         binding.cancellation_discards_all_unpublished;
}

[[nodiscard]] constexpr Sm87AotPrefillTypedResourceEdge
sm87_aot_prefill_expected_typed_resource_edge(
    const Sm87AotPrefillSystemPlan& plan,
    const Sm87AotPrefillTypedResource resource) noexcept {
  Sm87AotPrefillTypedResourceEdge result;
  result.resource = resource;
  const Sm87AotPrefillProjectionPartitionBinding* producer = nullptr;
  switch (resource) {
    case Sm87AotPrefillTypedResource::kTokenIdsToEmbedding:
      result.producer_group = Sm87AotPrefillExecutionGroup::kEmbedding;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kEmbedding;
      result.producer_publication_identity =
          plan.model_dataflow.token_id_span_identity;
      result.consumer_input_identity =
          plan.model_dataflow.embedding_token_input_span_identity;
      break;
    case Sm87AotPrefillTypedResource::kEmbeddingToLayerZeroInput:
      result.producer_group = Sm87AotPrefillExecutionGroup::kEmbedding;
      result.consumer_group =
          Sm87AotPrefillExecutionGroup::kInputAndLayerNorm;
      result.producer_publication_identity =
          plan.model_dataflow.embedding_publication_table_identity;
      result.consumer_input_identity =
          plan.model_dataflow.layer_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kPriorDownToLayerInput:
      result.producer_group = Sm87AotPrefillExecutionGroup::kDownResidual;
      result.consumer_group =
          Sm87AotPrefillExecutionGroup::kInputAndLayerNorm;
      result.producer_publication_identity =
          plan.model_dataflow.next_layer_input_publication_table_identity;
      result.consumer_input_identity =
          plan.model_dataflow.layer_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kLayerInputToInputNorm:
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kInputAndLayerNorm;
      result.consumer_group =
          Sm87AotPrefillExecutionGroup::kInputAndLayerNorm;
      result.producer_publication_identity =
          plan.model_dataflow.layer_input_span_table_identity;
      result.consumer_input_identity =
          plan.model_dataflow.input_norm_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kInputNormToGdnQkvProjection: {
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kInputAndLayerNorm;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kLinearQkvZ;
      result.producer_publication_identity =
          plan.model_dataflow.input_norm_publication_table_identity;
      const auto* projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ);
      result.consumer_input_identity =
          projection == nullptr ? 0U : projection->input_span_table_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kInputNormToGdnAbProjection:
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kInputAndLayerNorm;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kLinearAb;
      result.producer_publication_identity =
          plan.model_dataflow.input_norm_publication_table_identity;
      result.consumer_input_identity =
          plan.model_dataflow.bf16_ab_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kInputNormToFullQkvProjection: {
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kInputAndLayerNorm;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kFullQkv;
      result.producer_publication_identity =
          plan.model_dataflow.input_norm_publication_table_identity;
      const auto* projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kFp8FullQkv);
      result.consumer_input_identity =
          projection == nullptr ? 0U : projection->input_span_table_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kLayerResidualToGdnOResidual:
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kInputAndLayerNorm;
      result.consumer_group =
          Sm87AotPrefillExecutionGroup::kLinearOResidual;
      result.producer_publication_identity =
          plan.model_dataflow.layer_input_span_table_identity;
      result.consumer_input_identity =
          plan.model_dataflow.gdn_o_residual_bypass_input_span_identity;
      break;
    case Sm87AotPrefillTypedResource::kLayerResidualToFullOResidual:
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kInputAndLayerNorm;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kFullOResidual;
      result.producer_publication_identity =
          plan.model_dataflow.layer_input_span_table_identity;
      result.consumer_input_identity =
          plan.model_dataflow.full_o_residual_bypass_input_span_identity;
      break;
    case Sm87AotPrefillTypedResource::kGateToSilu:
      result.producer_group = Sm87AotPrefillExecutionGroup::kGateUp;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kGateUp;
      producer = sm87_aot_prefill_projection_partition(
          plan, kernels::Sm87TargetAotLogicalRole::kNvFp4Gate);
      result.consumer_input_identity =
          plan.model_dataflow.silu_gate_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kUpToMultiply:
      result.producer_group = Sm87AotPrefillExecutionGroup::kGateUp;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kGateUp;
      producer = sm87_aot_prefill_projection_partition(
          plan, kernels::Sm87TargetAotLogicalRole::kNvFp4Up);
      result.consumer_input_identity =
          plan.model_dataflow.silu_up_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kSiluTimesUpToDownProjection: {
      result.producer_group = Sm87AotPrefillExecutionGroup::kGateUp;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kDownResidual;
      result.producer_publication_identity =
          plan.model_dataflow.silu_times_up_publication_table_identity;
      const auto* projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kNvFp4Down);
      result.consumer_input_identity =
          projection == nullptr ? 0U : projection->input_span_table_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kGdnQkvToCore:
      result.producer_group = Sm87AotPrefillExecutionGroup::kLinearQkvZ;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kGdn;
      producer = sm87_aot_prefill_projection_partition(
          plan, kernels::Sm87TargetAotLogicalRole::kFp8GdnQkv);
      result.consumer_input_identity =
          plan.gdn_dataflow.raw_qkv_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kGdnZToCore:
      result.producer_group = Sm87AotPrefillExecutionGroup::kLinearQkvZ;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kGdn;
      producer = sm87_aot_prefill_projection_partition(
          plan, kernels::Sm87TargetAotLogicalRole::kFp8GdnZ);
      result.consumer_input_identity =
          plan.gdn_dataflow.raw_z_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kFullQGateToPreprocess:
      result.producer_group = Sm87AotPrefillExecutionGroup::kFullQkv;
      result.consumer_group =
          Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish;
      producer = sm87_aot_prefill_projection_partition(
          plan, kernels::Sm87TargetAotLogicalRole::kFp8FullQGate);
      result.consumer_input_identity =
          plan.attention_dataflow.raw_q_gate_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kFullKToPreprocess:
      result.producer_group = Sm87AotPrefillExecutionGroup::kFullQkv;
      result.consumer_group =
          Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish;
      producer = sm87_aot_prefill_projection_partition(
          plan, kernels::Sm87TargetAotLogicalRole::kFp8FullK);
      result.consumer_input_identity =
          plan.attention_dataflow.raw_k_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kFullVToPreprocess:
      result.producer_group = Sm87AotPrefillExecutionGroup::kFullQkv;
      result.consumer_group =
          Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish;
      producer = sm87_aot_prefill_projection_partition(
          plan, kernels::Sm87TargetAotLogicalRole::kFp8FullV);
      result.consumer_input_identity =
          plan.attention_dataflow.raw_v_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kBf16AToGdn:
      result.producer_group = Sm87AotPrefillExecutionGroup::kLinearAb;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kGdn;
      result.producer_publication_identity =
          plan.model_dataflow.bf16_a_publication_table_identity;
      result.consumer_input_identity =
          plan.gdn_dataflow.a_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kBf16BToGdn:
      result.producer_group = Sm87AotPrefillExecutionGroup::kLinearAb;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kGdn;
      result.producer_publication_identity =
          plan.model_dataflow.bf16_b_publication_table_identity;
      result.consumer_input_identity =
          plan.gdn_dataflow.b_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kProcessedQGateToAttention:
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kAttention;
      result.producer_publication_identity = plan.attention_dataflow
                                                 .processed_q_gate_publication_table_identity;
      result.consumer_input_identity = plan.attention_dataflow
                                           .attention_q_gate_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kProcessedKToAttention:
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kAttention;
      result.producer_publication_identity =
          plan.attention_dataflow.processed_k_publication_table_identity;
      result.consumer_input_identity =
          plan.attention_dataflow.attention_k_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kProcessedVToAttention:
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kAttention;
      result.producer_publication_identity =
          plan.attention_dataflow.processed_v_publication_table_identity;
      result.consumer_input_identity =
          plan.attention_dataflow.attention_v_input_span_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kAttentionGatedToOutputProjection: {
      result.producer_group = Sm87AotPrefillExecutionGroup::kAttention;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kFullOResidual;
      result.producer_publication_identity =
          plan.attention_dataflow.gated_output_publication_table_identity;
      const auto* output_projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
          Sm87AotPrefillProjectionFamily::kFullAttentionLayers);
      result.consumer_input_identity =
          output_projection == nullptr
              ? 0U
              : output_projection->input_span_table_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kGdnOutputToOutputProjection: {
      result.producer_group = Sm87AotPrefillExecutionGroup::kGdn;
      result.consumer_group =
          Sm87AotPrefillExecutionGroup::kLinearOResidual;
      result.producer_publication_identity =
          plan.gdn_dataflow.output_publication_table_identity;
      const auto* output_projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
          Sm87AotPrefillProjectionFamily::kGdnLayers);
      result.consumer_input_identity =
          output_projection == nullptr
              ? 0U
              : output_projection->input_span_table_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kGdnOProjectionToResidual: {
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kLinearOResidual;
      result.consumer_group =
          Sm87AotPrefillExecutionGroup::kLinearOResidual;
      const auto* output_projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
          Sm87AotPrefillProjectionFamily::kGdnLayers);
      result.producer_publication_identity =
          output_projection == nullptr
              ? 0U
              : output_projection->partitions[0U].publication_identity;
      result.consumer_input_identity = plan.model_dataflow
                                           .gdn_o_projection_residual_input_span_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kFullOProjectionToResidual: {
      result.producer_group = Sm87AotPrefillExecutionGroup::kFullOResidual;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kFullOResidual;
      const auto* output_projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
          Sm87AotPrefillProjectionFamily::kFullAttentionLayers);
      result.producer_publication_identity =
          output_projection == nullptr
              ? 0U
              : output_projection->partitions[0U].publication_identity;
      result.consumer_input_identity = plan.model_dataflow
                                           .full_o_projection_residual_input_span_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kGdnResidualToPostNorm:
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kLinearOResidual;
      result.consumer_group =
          Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm;
      result.producer_publication_identity =
          plan.model_dataflow.gdn_attention_residual_publication_identity;
      result.consumer_input_identity =
          plan.model_dataflow.gdn_post_norm_input_span_identity;
      break;
    case Sm87AotPrefillTypedResource::kFullResidualToPostNorm:
      result.producer_group = Sm87AotPrefillExecutionGroup::kFullOResidual;
      result.consumer_group =
          Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm;
      result.producer_publication_identity =
          plan.model_dataflow.full_attention_residual_publication_identity;
      result.consumer_input_identity =
          plan.model_dataflow.full_post_norm_input_span_identity;
      break;
    case Sm87AotPrefillTypedResource::kPostNormToGateUp: {
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kPostAttentionLayerNorm;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kGateUp;
      result.producer_publication_identity =
          plan.model_dataflow.post_attention_norm_publication_table_identity;
      const auto* projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp);
      result.consumer_input_identity =
          projection == nullptr ? 0U : projection->input_span_table_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kDownProjectionToResidual: {
      result.producer_group = Sm87AotPrefillExecutionGroup::kDownResidual;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kDownResidual;
      const auto* projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kNvFp4Down);
      result.producer_publication_identity =
          projection == nullptr
              ? 0U
              : projection->partitions[0U].publication_identity;
      result.consumer_input_identity =
          plan.model_dataflow.down_branch_residual_input_span_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kGdnResidualToDownResidual:
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kLinearOResidual;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kDownResidual;
      result.producer_publication_identity =
          plan.model_dataflow.gdn_attention_residual_publication_identity;
      result.consumer_input_identity = plan.model_dataflow
                                           .gdn_down_residual_bypass_input_span_identity;
      break;
    case Sm87AotPrefillTypedResource::kFullResidualToDownResidual:
      result.producer_group = Sm87AotPrefillExecutionGroup::kFullOResidual;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kDownResidual;
      result.producer_publication_identity =
          plan.model_dataflow.full_attention_residual_publication_identity;
      result.consumer_input_identity = plan.model_dataflow
                                           .full_down_residual_bypass_input_span_identity;
      break;
    case Sm87AotPrefillTypedResource::kLastDownResidualToFinalNorm:
      result.producer_group = Sm87AotPrefillExecutionGroup::kDownResidual;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kFinalHandoff;
      result.producer_publication_identity =
          plan.model_dataflow.next_layer_input_publication_table_identity;
      result.consumer_input_identity =
          plan.model_dataflow.final_norm_input_span_identity;
      break;
    case Sm87AotPrefillTypedResource::kKvStateToRequestTransaction:
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kFinalHandoff;
      result.producer_publication_identity =
          plan.attention_dataflow
              .staged_kv_transaction_publication_table_identity;
      result.consumer_input_identity =
          plan.request_transaction.staged_kv_input_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kConvHistoryToRequestTransaction:
      result.producer_group = Sm87AotPrefillExecutionGroup::kGdn;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kFinalHandoff;
      result.producer_publication_identity =
          plan.gdn_dataflow.final_conv_history_publication_table_identity;
      result.consumer_input_identity =
          plan.request_transaction.conv_history_input_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kRecurrentStateToRequestTransaction:
      result.producer_group = Sm87AotPrefillExecutionGroup::kGdn;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kFinalHandoff;
      result.producer_publication_identity =
          plan.gdn_dataflow.final_recurrent_state_publication_table_identity;
      result.consumer_input_identity =
          plan.request_transaction.recurrent_state_input_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kPositionRopeEpochToRequestTransaction:
      result.producer_group =
          Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kFinalHandoff;
      result.producer_publication_identity = plan.attention_dataflow
                                                 .position_rope_epoch_publication_table_identity;
      result.consumer_input_identity =
          plan.request_transaction.position_rope_epoch_input_identity;
      break;
    case Sm87AotPrefillTypedResource::kFinalHiddenToRequestTransaction:
      result.producer_group = Sm87AotPrefillExecutionGroup::kFinalHandoff;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kFinalHandoff;
      result.producer_publication_identity =
          plan.model_dataflow.final_norm_publication_identity;
      result.consumer_input_identity =
          plan.request_transaction.final_hidden_input_identity;
      break;
    case Sm87AotPrefillTypedResource::kCommittedReceiptToDecodeHandoff:
      result.producer_group = Sm87AotPrefillExecutionGroup::kFinalHandoff;
      result.consumer_group = Sm87AotPrefillExecutionGroup::kFinalHandoff;
      result.producer_publication_identity =
          plan.request_transaction.prefill_state_committed_receipt_identity;
      result.consumer_input_identity =
          plan.request_transaction.decode_handoff_receipt_input_identity;
      break;
    case Sm87AotPrefillTypedResource::kInvalid:
    case Sm87AotPrefillTypedResource::kCount:
      return {};
  }
  if (producer != nullptr) {
    result.producer_publication_identity = producer->publication_identity;
  }
  return result;
}

[[nodiscard]] constexpr Sm87AotPrefillTypedEventEdge
sm87_aot_prefill_expected_typed_event_edge(
    const Sm87AotPrefillSystemPlan& plan,
    const Sm87AotPrefillTypedResource resource) noexcept {
  Sm87AotPrefillTypedEventEdge result;
  result.resource = resource;
  const Sm87AotPrefillProjectionPartitionBinding* partition = nullptr;
  switch (resource) {
    case Sm87AotPrefillTypedResource::kGdnQkvToCore: {
      const auto* projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ);
      result.producer_completion_event_identity =
          projection == nullptr
              ? 0U
              : projection->aggregate_completion_event_table_identity;
      result.consumer_ready_event_identity =
          plan.gdn_dataflow.raw_qkvz_ready_event_table_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kFullQGateToPreprocess:
      partition = sm87_aot_prefill_projection_partition(
          plan, kernels::Sm87TargetAotLogicalRole::kFp8FullQGate);
      result.consumer_ready_event_identity =
          plan.attention_dataflow.raw_q_gate_ready_event_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kFullKToPreprocess:
      partition = sm87_aot_prefill_projection_partition(
          plan, kernels::Sm87TargetAotLogicalRole::kFp8FullK);
      result.consumer_ready_event_identity =
          plan.attention_dataflow.raw_k_ready_event_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kFullVToPreprocess:
      partition = sm87_aot_prefill_projection_partition(
          plan, kernels::Sm87TargetAotLogicalRole::kFp8FullV);
      result.consumer_ready_event_identity =
          plan.attention_dataflow.raw_v_ready_event_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kBf16AToGdn:
      result.producer_completion_event_identity =
          plan.model_dataflow.bf16_a_rne_publication_event_identity;
      result.consumer_ready_event_identity =
          plan.gdn_dataflow.a_ready_event_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kBf16BToGdn:
      result.producer_completion_event_identity =
          plan.model_dataflow.bf16_b_rne_publication_event_identity;
      result.consumer_ready_event_identity =
          plan.gdn_dataflow.b_ready_event_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kProcessedQGateToAttention:
      result.producer_completion_event_identity =
          plan.attention_dataflow.preparation_ready_event_table_identity;
      result.consumer_ready_event_identity = plan.attention_dataflow
                                                 .attention_core_input_ready_event_table_identity;
      break;
    case Sm87AotPrefillTypedResource::kAttentionGatedToOutputProjection: {
      result.producer_completion_event_identity = plan.attention_dataflow
                                                       .gated_output_completion_event_table_identity;
      const auto* projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
          Sm87AotPrefillProjectionFamily::kFullAttentionLayers);
      result.consumer_ready_event_identity =
          projection == nullptr
              ? 0U
              : projection->input_ready_event_table_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kGdnOutputToOutputProjection: {
      result.producer_completion_event_identity =
          plan.gdn_dataflow.ordinary_kernel_completion_event_table_identity;
      const auto* projection = sm87_aot_prefill_projection_binding(
          plan, kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
          Sm87AotPrefillProjectionFamily::kGdnLayers);
      result.consumer_ready_event_identity =
          projection == nullptr
              ? 0U
              : projection->input_ready_event_table_identity;
      break;
    }
    case Sm87AotPrefillTypedResource::kKvStateToRequestTransaction:
      result.producer_completion_event_identity = plan.attention_dataflow
                                                       .staged_kv_transaction_ready_event_table_identity;
      result.consumer_ready_event_identity =
          plan.request_transaction.staged_kv_ready_input_event_identity;
      break;
    case Sm87AotPrefillTypedResource::kConvHistoryToRequestTransaction:
      // One stream-ordered GDN receipt covers both history and recurrent
      // state spans, which were pre-bound at request admission.
      result.producer_completion_event_identity = plan.gdn_dataflow
                                                       .post_kernel_stream_ordered_state_ready_receipt_identity;
      result.consumer_ready_event_identity =
          plan.request_transaction.gdn_state_ready_input_event_identity;
      break;
    case Sm87AotPrefillTypedResource::kPositionRopeEpochToRequestTransaction:
      result.producer_completion_event_identity = plan.attention_dataflow
                                                       .position_rope_epoch_ready_event_table_identity;
      result.consumer_ready_event_identity = plan.request_transaction
                                                 .position_rope_epoch_ready_input_event_identity;
      break;
    case Sm87AotPrefillTypedResource::kFinalHiddenToRequestTransaction:
      result.producer_completion_event_identity =
          plan.model_dataflow.final_norm_completion_event_identity;
      result.consumer_ready_event_identity =
          plan.request_transaction.final_hidden_ready_input_event_identity;
      break;
    default:
      return {};
  }
  if (partition != nullptr) {
    result.producer_completion_event_identity =
        partition->completion_event_identity;
  }
  return result;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_typed_event_edges_complete(
    const Sm87AotPrefillSystemPlan& plan) noexcept {
  if (plan.typed_event_edge_count != kSm87AotPrefillTypedEventEdgeCount) {
    return false;
  }
  for (std::size_t index = 0U; index < plan.typed_event_edge_count; ++index) {
    const auto& edge = plan.typed_event_edges[index];
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (plan.typed_event_edges[prior].resource == edge.resource) {
        return false;
      }
    }
    const auto expected =
        sm87_aot_prefill_expected_typed_event_edge(plan, edge.resource);
    if (expected.resource != edge.resource ||
        edge.producer_completion_event_identity !=
            expected.producer_completion_event_identity ||
        edge.consumer_ready_event_identity !=
            expected.consumer_ready_event_identity ||
        edge.producer_completion_event_identity == 0U ||
        edge.producer_completion_event_identity !=
            edge.consumer_ready_event_identity) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_typed_edges_complete(
    const Sm87AotPrefillSystemPlan& plan) noexcept {
  if (plan.typed_resource_edge_count !=
      kSm87AotPrefillTypedResourceEdgeCount) {
    return false;
  }
  std::array<std::uint8_t, kSm87AotPrefillTypedResourceEdgeCount> counts{};
  for (std::size_t index = 0U; index < plan.typed_resource_edge_count;
       ++index) {
    const auto& edge = plan.typed_resource_edges[index];
    const std::size_t resource_index =
        static_cast<std::size_t>(edge.resource);
    if (resource_index == 0U ||
        resource_index > kSm87AotPrefillTypedResourceEdgeCount ||
        ++counts[resource_index - 1U] != 1U) {
      return false;
    }
    const auto expected =
        sm87_aot_prefill_expected_typed_resource_edge(plan, edge.resource);
    if (edge.producer_group != expected.producer_group ||
        edge.consumer_group != expected.consumer_group ||
        edge.producer_publication_identity !=
            expected.producer_publication_identity ||
        edge.consumer_input_identity != expected.consumer_input_identity ||
        edge.producer_publication_identity == 0U ||
        edge.producer_publication_identity != edge.consumer_input_identity) {
      return false;
    }
  }
  for (const auto count : counts) {
    if (count != 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool sm87_aot_prefill_typed_dataflow_complete(
    const Sm87AotPrefillSystemPlan& plan) noexcept {
  if (!sm87_aot_prefill_attention_dataflow_complete(
          plan.attention_dataflow,
          plan.capacity.prompt.witness_prompt_tokens,
          plan.capacity.prompt.request_capacity_tokens) ||
      !sm87_aot_prefill_gdn_dataflow_complete(plan.gdn_dataflow) ||
      !sm87_aot_prefill_model_dataflow_complete(plan.model_dataflow) ||
      !sm87_aot_prefill_request_transaction_complete(
          plan.request_transaction) ||
      !sm87_aot_prefill_typed_edges_complete(plan) ||
      !sm87_aot_prefill_typed_event_edges_complete(plan) ||
      plan.gdn_dataflow.request_transaction_lifetime_contract_identity !=
          plan.request_transaction.transaction_lifetime_contract_identity) {
    return false;
  }
  for (std::size_t index = 0U;
       index < kSm87AotPrefillSystemProjectionDataflowCount; ++index) {
    const std::size_t constituent_index =
        index < kSm87AotPrefillSystemProjectionPlanCount ? index : 4U;
    if (!sm87_aot_prefill_projection_dataflow_complete(
            plan.projection_dataflows[index],
            plan.constituents.projections[constituent_index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr Sm87AotPrefillSystemPlanValidation
validate_sm87_aot_prefill_system_plan(
    const Sm87AotPrefillSystemPlan& plan) noexcept {
  Sm87AotPrefillSystemPlanValidation validation;
  const auto add_issue = [&validation](
                             const Sm87AotPrefillSystemPlanIssue issue) {
    validation.issue_mask |= static_cast<std::uint64_t>(issue);
  };
  bool malformed_enum = false;

  const auto canonical_prompt =
      kernels::sm87_target_aot_capacity_contract(plan.capacity.prompt.bucket);
  const auto canonical_capacity =
      sm87_aot_prefill_system_capacity_plan(plan.capacity.prompt.bucket);
  validation.capacity_contract_complete =
      canonical_prompt.valid() && plan.capacity.prompt.valid() &&
      plan.capacity.prompt.minimum_prompt_tokens ==
          canonical_prompt.minimum_prompt_tokens &&
      plan.capacity.prompt.maximum_prompt_tokens ==
          canonical_prompt.maximum_prompt_tokens &&
      plan.capacity.prompt.witness_prompt_tokens ==
          canonical_prompt.witness_prompt_tokens &&
      plan.capacity.prompt.request_capacity_tokens ==
          canonical_prompt.request_capacity_tokens;
  if (!validation.capacity_contract_complete) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kInvalidCapacity);
  }

  validation.staging_design_complete =
      canonical_prompt.valid() && plan.capacity.staging_span_count != 0U &&
      sm87_aot_prefill_same_staging_design(plan.capacity, canonical_capacity);
  if (!validation.staging_design_complete) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kStagingGeometryMismatch);
  }
  validation.preliminary_memory_design_complete =
      canonical_prompt.valid() &&
      plan.capacity.selected_request_memory_bytes ==
          canonical_capacity.selected_request_memory_bytes &&
      plan.capacity.conservative_request_memory_bytes ==
          canonical_capacity.conservative_request_memory_bytes &&
      plan.capacity.selected_request_memory_bytes != 0U &&
      plan.capacity.conservative_request_memory_bytes >=
          plan.capacity.selected_request_memory_bytes;
  if (!validation.preliminary_memory_design_complete) {
    add_issue(
        Sm87AotPrefillSystemPlanIssue::kPreliminaryMemoryContractMismatch);
  }

  const std::string_view expected_plan_id =
      sm87_aot_prefill_system_plan_id(plan.capacity.prompt.bucket);
  const bool candidate_identity_matches =
      plan.candidate_id == kSm87AotPrefillSystemCandidateId;
  const bool deployment_identity_matches =
      !expected_plan_id.empty() && plan.deployment_plan_id == expected_plan_id;
  if (!candidate_identity_matches) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kCandidateIdentityMismatch);
    if (!plan.candidate_id.empty()) {
      add_issue(
          Sm87AotPrefillSystemPlanIssue::kLegacyOrForeignIdentityForbidden);
    }
  }
  if (!deployment_identity_matches) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kDeploymentPlanIdentityMismatch);
    if (!plan.deployment_plan_id.empty()) {
      add_issue(
          Sm87AotPrefillSystemPlanIssue::kLegacyOrForeignIdentityForbidden);
    }
  }
  const std::array<std::uint64_t, 7U> system_identities{{
      plan.binary_identity,
      plan.checkpoint_identity,
      plan.platform_identity,
      plan.numerical_contract_identity,
      plan.request_memory_plan_identity,
      plan.stream_plan_identity,
      plan.handoff_plan_identity,
  }};
  const bool opaque_system_identities =
      kernels::sm87_target_aot_nonzero_unique_identities(system_identities);
  if (!opaque_system_identities) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kSystemIdentityUnbound);
  }
  validation.identity_schema_complete =
      candidate_identity_matches && deployment_identity_matches &&
      opaque_system_identities;

  const Sm87AotPrefillConstituentDesign canonical_constituents =
      sm87_aot_prefill_constituent_design(
          canonical_prompt.witness_prompt_tokens);
  validation.constituent_design_complete =
      canonical_prompt.valid() &&
      sm87_aot_prefill_same_constituent_design(plan.constituents,
                                               canonical_constituents);
  if (!validation.constituent_design_complete) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kConstituentDesignMismatch);
  }

  bool layer_schedule_matches =
      plan.layer_schedule_count == kSm87AotPrefillSystemLayerCount;
  bool instance_dependency_graph_matches = layer_schedule_matches;
  std::size_t observed_gdn_layers = 0U;
  std::size_t observed_full_attention_layers = 0U;
  const std::size_t schedule_count =
      plan.layer_schedule_count < kSm87AotPrefillSystemLayerCount
          ? plan.layer_schedule_count
          : kSm87AotPrefillSystemLayerCount;
  for (std::size_t index = 0U; index < schedule_count; ++index) {
    const auto& layer = plan.layer_schedule[index];
    const auto canonical_layer = sm87_aot_prefill_layer_schedule(index);
    if (!sm87_aot_prefill_same_layer_schedule(
            layer, canonical_layer)) {
      layer_schedule_matches = false;
    }
    for (std::size_t stage = 0U;
         stage < kSm87AotPrefillLayerStageCount; ++stage) {
      const auto& observed = layer.stage_instances[stage];
      const auto& expected = canonical_layer.stage_instances[stage];
      if (observed.layer_index != expected.layer_index ||
          observed.stage_index != expected.stage_index ||
          observed.group != expected.group ||
          observed.predecessor_count != expected.predecessor_count) {
        instance_dependency_graph_matches = false;
      }
      for (std::size_t predecessor = 0U;
           predecessor < kSm87AotPrefillMaximumStagePredecessors;
           ++predecessor) {
        if (!sm87_aot_prefill_same_instance_predecessor(
                observed.predecessors[predecessor],
                expected.predecessors[predecessor])) {
          instance_dependency_graph_matches = false;
        }
      }
    }
    if (layer.kind == Sm87AotPrefillLayerKind::kGdn) {
      ++observed_gdn_layers;
    } else if (layer.kind == Sm87AotPrefillLayerKind::kFullAttention) {
      ++observed_full_attention_layers;
    } else {
      malformed_enum = true;
      layer_schedule_matches = false;
    }
  }
  validation.layer_schedule_design_complete =
      layer_schedule_matches &&
      observed_gdn_layers == kSm87AotPrefillSystemGdnLayerCount &&
      observed_full_attention_layers ==
          kSm87AotPrefillSystemFullAttentionLayerCount &&
      plan.final_handoff_instance.group ==
          Sm87AotPrefillExecutionGroup::kFinalHandoff &&
      sm87_aot_prefill_same_instance_predecessor(
          plan.final_handoff_instance.final_layer_predecessor,
          sm87_aot_prefill_stage_predecessor(
              kSm87AotPrefillSystemLayerCount - 1U,
              kSm87AotPrefillLayerStageCount - 1U,
              Sm87AotPrefillExecutionGroup::kDownResidual,
              Sm87AotPrefillInstanceEdgeKind::kData));
  instance_dependency_graph_matches =
      instance_dependency_graph_matches &&
      plan.final_handoff_instance.group ==
          Sm87AotPrefillExecutionGroup::kFinalHandoff &&
      sm87_aot_prefill_same_instance_predecessor(
          plan.final_handoff_instance.final_layer_predecessor,
          sm87_aot_prefill_stage_predecessor(
              kSm87AotPrefillSystemLayerCount - 1U,
              kSm87AotPrefillLayerStageCount - 1U,
              Sm87AotPrefillExecutionGroup::kDownResidual,
              Sm87AotPrefillInstanceEdgeKind::kData));
  if (!validation.layer_schedule_design_complete) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kLayerScheduleMismatch);
  }
  if (!instance_dependency_graph_matches) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kDependencyGraphMismatch);
  }

  validation.typed_dataflow_bindings_complete =
      sm87_aot_prefill_typed_dataflow_complete(plan);
  if (!validation.typed_dataflow_bindings_complete) {
    add_issue(
        Sm87AotPrefillSystemPlanIssue::kTypedDataflowBindingIncomplete);
  }

  if (plan.logical_publication_count !=
      kSm87AotPrefillSystemRequiredRoleCount) {
    add_issue(Sm87AotPrefillSystemPlanIssue::
                  kInvalidLogicalPublicationCount);
  }
  bool logical_design = true;
  bool logical_exact_intent = true;
  const std::size_t logical_count =
      plan.logical_publication_count < kSm87AotPrefillSystemRequiredRoleCount
          ? plan.logical_publication_count
          : kSm87AotPrefillSystemRequiredRoleCount;
  for (std::size_t index = 0U; index < logical_count; ++index) {
    const auto& publication = plan.logical_publications[index];
    const std::size_t role_index = static_cast<std::size_t>(publication.role);
    if (role_index >= kSm87AotPrefillSystemRequiredRoleCount) {
      add_issue(Sm87AotPrefillSystemPlanIssue::kInvalidLogicalRole);
      logical_design = false;
      malformed_enum = true;
      continue;
    }
    if (++validation.role_counts[role_index] > 1U) {
      add_issue(Sm87AotPrefillSystemPlanIssue::kDuplicateLogicalRole);
      logical_design = false;
    }
    if (!detail::valid_numerical_intent(
            publication.declared_numerical_intent)) {
      malformed_enum = true;
    }
    if (publication.declared_numerical_intent !=
        PrefillNumericalMode::kExact) {
      logical_exact_intent = false;
      add_issue(Sm87AotPrefillSystemPlanIssue::kDeclaredExactIntentMissing);
    }
    if (!sm87_aot_prefill_same_logical_publication(
            publication,
            sm87_aot_prefill_logical_publication(publication.role))) {
      logical_design = false;
      add_issue(
          Sm87AotPrefillSystemPlanIssue::kLogicalPublicationDesignMismatch);
    }
  }
  bool every_role_once =
      plan.logical_publication_count == kSm87AotPrefillSystemRequiredRoleCount;
  for (const std::uint8_t count : validation.role_counts) {
    if (count == 0U) {
      add_issue(Sm87AotPrefillSystemPlanIssue::kMissingLogicalRole);
      every_role_once = false;
    } else if (count != 1U) {
      every_role_once = false;
    }
  }
  validation.logical_publication_design_complete =
      every_role_once && logical_design && !malformed_enum;

  if (plan.execution_group_count != kSm87AotPrefillExecutionGroupCount) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kInvalidExecutionGroupCount);
  }
  bool group_design = true;
  bool group_exact_intent = true;
  bool native_groups = true;
  bool aot_groups = true;
  bool group_identities = true;
  bool expected_work = true;
  bool dependency_graph = true;
  bool group_forbidden = false;
  const std::size_t group_count =
      plan.execution_group_count < kSm87AotPrefillExecutionGroupCount
          ? plan.execution_group_count
          : kSm87AotPrefillExecutionGroupCount;
  for (std::size_t index = 0U; index < group_count; ++index) {
    const auto& group = plan.execution_groups[index];
    const std::size_t group_index = static_cast<std::size_t>(group.group);
    if (group_index >= kSm87AotPrefillExecutionGroupCount) {
      add_issue(Sm87AotPrefillSystemPlanIssue::kInvalidExecutionGroup);
      group_design = false;
      malformed_enum = true;
      continue;
    }
    if (++validation.group_counts[group_index] > 1U) {
      add_issue(Sm87AotPrefillSystemPlanIssue::kDuplicateExecutionGroup);
      group_design = false;
    }
    if (!detail::valid_numerical_intent(group.declared_numerical_intent) ||
        !detail::valid_provider(group.provider) ||
        !detail::valid_tactic_mode(group.tactic_mode)) {
      malformed_enum = true;
    }
    if (group.declared_numerical_intent != PrefillNumericalMode::kExact) {
      group_exact_intent = false;
      add_issue(Sm87AotPrefillSystemPlanIssue::kDeclaredExactIntentMissing);
    }
    if (group.provider != PrefillOperatorProvider::kNative) {
      native_groups = false;
      add_issue(Sm87AotPrefillSystemPlanIssue::kNonNativeExecutionGroup);
    }
    if (group.tactic_mode != PrefillTacticMode::kAheadOfTime) {
      aot_groups = false;
      add_issue(Sm87AotPrefillSystemPlanIssue::kExecutionGroupNotAot);
    }
    if (group.tactic_identity == 0U || group.binding_identity == 0U) {
      group_identities = false;
      add_issue(
          Sm87AotPrefillSystemPlanIssue::kExecutionGroupIdentityUnbound);
    }
    const auto canonical_group = sm87_aot_prefill_execution_group(
        group.group, canonical_constituents);
    if (group.predecessor_group_mask !=
            canonical_group.predecessor_group_mask ||
        group.publication_consumer_group_mask !=
            canonical_group.publication_consumer_group_mask) {
      dependency_graph = false;
      add_issue(Sm87AotPrefillSystemPlanIssue::kDependencyGraphMismatch);
    }
    if (!sm87_aot_prefill_same_execution_group_design(group,
                                                       canonical_group)) {
      group_design = false;
      add_issue(
          Sm87AotPrefillSystemPlanIssue::kExecutionGroupDesignMismatch);
    }
    if (!group.work.canonical_counts_valid ||
        !sm87_aot_prefill_same_physical_work(group.work,
                                             canonical_group.work)) {
      expected_work = false;
      add_issue(Sm87AotPrefillSystemPlanIssue::kExpectedPhysicalWorkMismatch);
    }
    if (group.uses_mtp || group.allows_fallback) {
      group_forbidden = true;
    }
  }
  bool every_group_once =
      plan.execution_group_count == kSm87AotPrefillExecutionGroupCount;
  for (const std::uint8_t count : validation.group_counts) {
    if (count == 0U) {
      add_issue(Sm87AotPrefillSystemPlanIssue::kMissingExecutionGroup);
      every_group_once = false;
    } else if (count != 1U) {
      every_group_once = false;
    }
  }

  for (std::size_t left_index = 0U; left_index < group_count; ++left_index) {
    const auto& left = plan.execution_groups[left_index];
    if (static_cast<std::size_t>(left.group) >=
        kSm87AotPrefillExecutionGroupCount) {
      continue;
    }
    for (std::size_t right_index = left_index + 1U;
         right_index < group_count; ++right_index) {
      const auto& right = plan.execution_groups[right_index];
      if (static_cast<std::size_t>(right.group) >=
          kSm87AotPrefillExecutionGroupCount) {
        continue;
      }
      if (left.binding_identity != 0U &&
          left.binding_identity == right.binding_identity) {
        add_issue(Sm87AotPrefillSystemPlanIssue::
                      kUnexpectedExecutionGroupIdentityAlias);
        group_identities = false;
      }
      if (left.tactic_identity != 0U &&
          left.tactic_identity == right.tactic_identity &&
          !sm87_aot_prefill_tactic_alias_allowed(left.group, right.group)) {
        add_issue(Sm87AotPrefillSystemPlanIssue::
                      kUnexpectedExecutionGroupIdentityAlias);
        group_identities = false;
      }
    }
  }

  enum class IdentityEntryKind : std::uint8_t {
    kSystem,
    kGroupTactic,
    kGroupBinding,
    kDataflow,
  };
  struct IdentityEntry {
    std::uint64_t identity = 0U;
    IdentityEntryKind kind = IdentityEntryKind::kSystem;
    Sm87AotPrefillExecutionGroup group =
        Sm87AotPrefillExecutionGroup::kInvalid;
    // Nonzero only for an explicitly typed producer->consumer alias class.
    std::uint8_t typed_alias_tag = 0U;
  };
  constexpr std::size_t kUnifiedIdentityEntryCount = 256U;
  std::array<IdentityEntry, kUnifiedIdentityEntryCount> identity_entries{};
  std::size_t identity_entry_count = 0U;
  for (const std::uint64_t identity : system_identities) {
    identity_entries[identity_entry_count++] =
        {identity, IdentityEntryKind::kSystem,
         Sm87AotPrefillExecutionGroup::kInvalid, 0U};
  }
  for (const auto& group : plan.execution_groups) {
    identity_entries[identity_entry_count++] =
        {group.tactic_identity, IdentityEntryKind::kGroupTactic,
         group.group, 0U};
    identity_entries[identity_entry_count++] =
        {group.binding_identity, IdentityEntryKind::kGroupBinding,
         group.group, 0U};
  }
  const auto append_dataflow_identity =
      [&identity_entries, &identity_entry_count](
          const std::uint64_t identity,
          const std::uint8_t typed_alias_tag = 0U) {
        if (identity_entry_count >= identity_entries.size()) {
          return;
        }
        identity_entries[identity_entry_count++] =
            {identity, IdentityEntryKind::kDataflow,
             Sm87AotPrefillExecutionGroup::kInvalid, typed_alias_tag};
      };
  // Tags denote resource-storage equivalence classes, not edge ordinals.
  // Fan-out edges therefore share a tag while unrelated resources cannot.
  const auto resource_alias_tag = [](const Sm87AotPrefillTypedResource resource) {
    switch (resource) {
      case Sm87AotPrefillTypedResource::kTokenIdsToEmbedding:
        return std::uint8_t{1U};
      case Sm87AotPrefillTypedResource::kEmbeddingToLayerZeroInput:
      case Sm87AotPrefillTypedResource::kPriorDownToLayerInput:
      case Sm87AotPrefillTypedResource::kLayerInputToInputNorm:
      case Sm87AotPrefillTypedResource::kLayerResidualToGdnOResidual:
      case Sm87AotPrefillTypedResource::kLayerResidualToFullOResidual:
      case Sm87AotPrefillTypedResource::kLastDownResidualToFinalNorm:
        return std::uint8_t{2U};
      case Sm87AotPrefillTypedResource::kInputNormToGdnQkvProjection:
      case Sm87AotPrefillTypedResource::kInputNormToGdnAbProjection:
      case Sm87AotPrefillTypedResource::kInputNormToFullQkvProjection:
        return std::uint8_t{3U};
      case Sm87AotPrefillTypedResource::kGateToSilu:
        return std::uint8_t{4U};
      case Sm87AotPrefillTypedResource::kUpToMultiply:
        return std::uint8_t{5U};
      case Sm87AotPrefillTypedResource::kSiluTimesUpToDownProjection:
        return std::uint8_t{6U};
      case Sm87AotPrefillTypedResource::kGdnQkvToCore:
        return std::uint8_t{7U};
      case Sm87AotPrefillTypedResource::kGdnZToCore:
        return std::uint8_t{8U};
      case Sm87AotPrefillTypedResource::kFullQGateToPreprocess:
        return std::uint8_t{9U};
      case Sm87AotPrefillTypedResource::kFullKToPreprocess:
        return std::uint8_t{10U};
      case Sm87AotPrefillTypedResource::kFullVToPreprocess:
        return std::uint8_t{11U};
      case Sm87AotPrefillTypedResource::kBf16AToGdn:
        return std::uint8_t{12U};
      case Sm87AotPrefillTypedResource::kBf16BToGdn:
        return std::uint8_t{13U};
      case Sm87AotPrefillTypedResource::kProcessedQGateToAttention:
        return std::uint8_t{14U};
      case Sm87AotPrefillTypedResource::kProcessedKToAttention:
        return std::uint8_t{15U};
      case Sm87AotPrefillTypedResource::kProcessedVToAttention:
        return std::uint8_t{16U};
      case Sm87AotPrefillTypedResource::kAttentionGatedToOutputProjection:
        return std::uint8_t{17U};
      case Sm87AotPrefillTypedResource::kGdnOutputToOutputProjection:
        return std::uint8_t{18U};
      case Sm87AotPrefillTypedResource::kGdnOProjectionToResidual:
        return std::uint8_t{19U};
      case Sm87AotPrefillTypedResource::kFullOProjectionToResidual:
        return std::uint8_t{20U};
      case Sm87AotPrefillTypedResource::kGdnResidualToPostNorm:
      case Sm87AotPrefillTypedResource::kGdnResidualToDownResidual:
        return std::uint8_t{21U};
      case Sm87AotPrefillTypedResource::kFullResidualToPostNorm:
      case Sm87AotPrefillTypedResource::kFullResidualToDownResidual:
        return std::uint8_t{22U};
      case Sm87AotPrefillTypedResource::kPostNormToGateUp:
        return std::uint8_t{23U};
      case Sm87AotPrefillTypedResource::kDownProjectionToResidual:
        return std::uint8_t{24U};
      case Sm87AotPrefillTypedResource::kKvStateToRequestTransaction:
        return std::uint8_t{25U};
      case Sm87AotPrefillTypedResource::kConvHistoryToRequestTransaction:
        return std::uint8_t{26U};
      case Sm87AotPrefillTypedResource::kRecurrentStateToRequestTransaction:
        return std::uint8_t{27U};
      case Sm87AotPrefillTypedResource::kPositionRopeEpochToRequestTransaction:
        return std::uint8_t{28U};
      case Sm87AotPrefillTypedResource::kFinalHiddenToRequestTransaction:
        return std::uint8_t{29U};
      case Sm87AotPrefillTypedResource::kCommittedReceiptToDecodeHandoff:
        return std::uint8_t{30U};
      case Sm87AotPrefillTypedResource::kInvalid:
      case Sm87AotPrefillTypedResource::kCount:
        return std::uint8_t{0U};
    }
    return std::uint8_t{0U};
  };
  for (const auto& edge : plan.typed_resource_edges) {
    const auto tag = resource_alias_tag(edge.resource);
    append_dataflow_identity(edge.producer_publication_identity, tag);
    append_dataflow_identity(edge.consumer_input_identity, tag);
  }
  for (std::size_t index = 0U; index < plan.typed_event_edge_count;
       ++index) {
    const auto tag = static_cast<std::uint8_t>(128U + index);
    append_dataflow_identity(
        plan.typed_event_edges[index].producer_completion_event_identity,
        tag);
    append_dataflow_identity(
        plan.typed_event_edges[index].consumer_ready_event_identity, tag);
  }

  const std::array attention_unaliased{
      plan.attention_dataflow.q_norm_weight_table_identity,
      plan.attention_dataflow.k_norm_weight_table_identity,
      plan.attention_dataflow.rope_position_contract_identity,
      plan.attention_dataflow.kv_cache_arena_table_identity,
      plan.attention_dataflow.kv_cache_lifetime_contract_identity,
      plan.attention_dataflow.pre_gate_bf16_output_span_table_identity,
      plan.attention_dataflow.pre_gate_bf16_publication_table_identity,
      plan.attention_dataflow.pre_gate_bf16_lifetime_contract_identity,
      plan.attention_dataflow.pre_gate_bf16_completion_event_table_identity,
      plan.attention_dataflow.gated_output_span_table_identity,
      plan.attention_dataflow.gated_output_lifetime_contract_identity,
  };
  for (const auto identity : attention_unaliased) {
    append_dataflow_identity(identity);
  }
  const std::array gdn_unaliased{
      plan.gdn_dataflow.conv_weight_table_identity,
      plan.gdn_dataflow.a_log_table_identity,
      plan.gdn_dataflow.dt_bias_table_identity,
      plan.gdn_dataflow.output_norm_weight_table_identity,
      plan.gdn_dataflow.raw_bf16_publication_table_identity,
      plan.gdn_dataflow.norm_silu_z_publication_table_identity,
      plan.gdn_dataflow.output_span_table_identity,
      plan.gdn_dataflow.reset_zero_epoch_table_identity,
  };
  for (const auto identity : gdn_unaliased) {
    append_dataflow_identity(identity);
  }
  constexpr std::uint8_t kRequestTransactionLifetimeAlias = 0xfeU;
  append_dataflow_identity(
      plan.gdn_dataflow.request_transaction_lifetime_contract_identity,
      kRequestTransactionLifetimeAlias);

  for (const auto& projection : plan.projection_dataflows) {
    for (std::size_t index = 0U; index < projection.partition_count &&
                                  index < projection.partitions.size();
         ++index) {
      const auto& partition = projection.partitions[index];
      append_dataflow_identity(partition.source_packed_asset_identity);
      append_dataflow_identity(partition.independent_scale_identity);
      append_dataflow_identity(partition.raw_bits_contract_identity);
      append_dataflow_identity(partition.output_span_identity);
      const bool event_is_typed =
          partition.logical_role ==
              kernels::Sm87TargetAotLogicalRole::kFp8FullQGate ||
          partition.logical_role ==
              kernels::Sm87TargetAotLogicalRole::kFp8FullK ||
          partition.logical_role ==
              kernels::Sm87TargetAotLogicalRole::kFp8FullV;
      if (!event_is_typed) {
        append_dataflow_identity(partition.completion_event_identity);
      }
    }
  }

  const std::array model_unaliased{
      plan.model_dataflow.embedding_weight_table_identity,
      plan.model_dataflow.input_norm_weight_table_identity,
      plan.model_dataflow.input_norm_epsilon_contract_identity,
      plan.model_dataflow.post_attention_norm_weight_table_identity,
      plan.model_dataflow.post_attention_norm_epsilon_contract_identity,
      plan.model_dataflow.residual_lifetime_contract_identity,
      plan.model_dataflow.bf16_a_weight_table_identity,
      plan.model_dataflow.bf16_b_weight_table_identity,
      plan.model_dataflow.bf16_a_completion_event_table_identity,
      plan.model_dataflow.bf16_b_completion_event_table_identity,
      plan.model_dataflow.final_norm_weight_identity,
      plan.model_dataflow.final_norm_epsilon_contract_identity,
      plan.model_dataflow.final_handoff_identity,
  };
  for (const auto identity : model_unaliased) {
    append_dataflow_identity(identity);
  }
  append_dataflow_identity(
      plan.request_transaction.transaction_lifetime_contract_identity,
      kRequestTransactionLifetimeAlias);
  append_dataflow_identity(
      plan.request_transaction.prefill_state_committed_receipt_schema_identity);

  bool unified_identity_namespace =
      identity_entry_count < identity_entries.size();
  for (std::size_t left = 0U; left < identity_entry_count; ++left) {
    if (identity_entries[left].identity == 0U) {
      unified_identity_namespace = false;
      continue;
    }
    for (std::size_t right = left + 1U; right < identity_entry_count;
         ++right) {
      if (identity_entries[left].identity !=
          identity_entries[right].identity) {
        continue;
      }
      const bool admitted_tactic_alias =
          identity_entries[left].kind == IdentityEntryKind::kGroupTactic &&
          identity_entries[right].kind == IdentityEntryKind::kGroupTactic &&
          sm87_aot_prefill_tactic_alias_allowed(
              identity_entries[left].group, identity_entries[right].group);
      const bool admitted_typed_edge_alias =
          identity_entries[left].kind == IdentityEntryKind::kDataflow &&
          identity_entries[right].kind == IdentityEntryKind::kDataflow &&
          identity_entries[left].typed_alias_tag != 0U &&
          identity_entries[left].typed_alias_tag ==
              identity_entries[right].typed_alias_tag;
      if (!admitted_tactic_alias && !admitted_typed_edge_alias) {
        unified_identity_namespace = false;
        add_issue(Sm87AotPrefillSystemPlanIssue::kUnexpectedIdentityAlias);
      }
    }
  }
  validation.unified_identity_namespace_complete =
      unified_identity_namespace;
  validation.identity_schema_complete =
      validation.identity_schema_complete && unified_identity_namespace;

  const Sm87AotPrefillPhysicalExecutionGroup* gate_up = nullptr;
  const Sm87AotPrefillPhysicalExecutionGroup* down = nullptr;
  for (std::size_t index = 0U; index < group_count; ++index) {
    if (plan.execution_groups[index].group ==
        Sm87AotPrefillExecutionGroup::kGateUp) {
      gate_up = &plan.execution_groups[index];
    } else if (plan.execution_groups[index].group ==
               Sm87AotPrefillExecutionGroup::kDownResidual) {
      down = &plan.execution_groups[index];
    }
  }
  if (gate_up != nullptr && down != nullptr &&
      gate_up->tactic_identity != 0U &&
      gate_up->tactic_identity == down->tactic_identity) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kGateDownTacticAlias);
    group_identities = false;
  }

  validation.physical_execution_design_complete =
      every_group_once && group_design && !malformed_enum;
  validation.dependency_graph_complete =
      instance_dependency_graph_matches && every_group_once &&
      dependency_graph && !malformed_enum;
  validation.declared_exact_intent_complete =
      every_role_once && every_group_once && logical_exact_intent &&
      group_exact_intent && !malformed_enum;
  validation.native_provider_design_complete =
      every_group_once && native_groups && !malformed_enum;
  validation.aot_tactic_design_complete =
      every_group_once && aot_groups && !malformed_enum;
  validation.physical_group_identities_complete =
      every_group_once && group_identities;
  validation.expected_physical_work_complete =
      every_group_once && expected_work;

  bool ownership = every_role_once && every_group_once;
  for (std::size_t role_index = 0U;
       role_index < kSm87AotPrefillSystemRequiredRoleCount; ++role_index) {
    const auto role = static_cast<PrefillBindingRole>(role_index);
    const std::uint64_t expected_group_mask =
        sm87_aot_prefill_logical_publication(role).execution_group_mask;
    std::uint64_t reverse_group_mask = 0U;
    for (std::size_t group_index = 0U; group_index < group_count;
         ++group_index) {
      if ((plan.execution_groups[group_index].logical_role_mask &
           sm87_aot_prefill_role_bit(role)) != 0U) {
        reverse_group_mask |=
            sm87_aot_prefill_group_bit(plan.execution_groups[group_index].group);
      }
    }
    const Sm87AotPrefillLogicalPublication* bound_publication = nullptr;
    for (std::size_t publication_index = 0U;
         publication_index < logical_count; ++publication_index) {
      if (plan.logical_publications[publication_index].role == role) {
        bound_publication = &plan.logical_publications[publication_index];
        break;
      }
    }
    if (bound_publication == nullptr ||
        bound_publication->execution_group_mask != expected_group_mask ||
        reverse_group_mask != expected_group_mask) {
      ownership = false;
    }
  }
  validation.logical_physical_ownership_complete = ownership;
  if (!ownership) {
    add_issue(
        Sm87AotPrefillSystemPlanIssue::kLogicalPhysicalOwnershipMismatch);
  }

  const bool system_forbidden =
      plan.uses_prefix_cache || plan.uses_mtp || plan.uses_cublaslt ||
      plan.uses_external_runtime || plan.uses_approximate_numerics ||
      plan.uses_request_time_jit || plan.uses_request_time_autotune ||
      plan.uses_request_time_repack ||
      plan.uses_request_time_tactic_discovery || plan.allows_fallback ||
      plan.allows_silent_truncation;
  validation.forbidden_boundaries_satisfied =
      !system_forbidden && !group_forbidden;
  if (!validation.forbidden_boundaries_satisfied) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kForbiddenBoundaryEnabled);
  }

  validation.qualification_boundary_satisfied =
      plan.qualification ==
          Sm87AotPrefillSystemQualification::kAccuracyUnqualified &&
      !plan.accuracy_qualified && !plan.production_dispatch_eligible;
  if (plan.qualification != Sm87AotPrefillSystemQualification::kInvalid &&
      plan.qualification !=
          Sm87AotPrefillSystemQualification::kAccuracyUnqualified) {
    malformed_enum = true;
  }
  if (!validation.qualification_boundary_satisfied) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kQualificationBoundaryInvalid);
  }
  if (malformed_enum) {
    add_issue(Sm87AotPrefillSystemPlanIssue::kMalformedEnumEncoding);
  }

  validation.operator_bindings_complete =
      validation.logical_publication_design_complete &&
      validation.physical_execution_design_complete &&
      validation.layer_schedule_design_complete &&
      validation.dependency_graph_complete &&
      validation.typed_dataflow_bindings_complete &&
      validation.declared_exact_intent_complete &&
      validation.native_provider_design_complete &&
      validation.aot_tactic_design_complete &&
      validation.physical_group_identities_complete &&
      validation.unified_identity_namespace_complete &&
      validation.expected_physical_work_complete &&
      validation.logical_physical_ownership_complete &&
      validation.forbidden_boundaries_satisfied;
  validation.canonical_design_complete =
      candidate_identity_matches && deployment_identity_matches &&
      validation.identity_schema_complete &&
      validation.capacity_contract_complete &&
      validation.staging_design_complete &&
      validation.preliminary_memory_design_complete &&
      validation.constituent_design_complete &&
      validation.layer_schedule_design_complete &&
      validation.dependency_graph_complete &&
      validation.typed_dataflow_bindings_complete &&
      validation.logical_publication_design_complete &&
      validation.physical_execution_design_complete &&
      validation.declared_exact_intent_complete &&
      validation.native_provider_design_complete &&
      validation.aot_tactic_design_complete &&
      validation.physical_group_identities_complete &&
      validation.unified_identity_namespace_complete &&
      validation.expected_physical_work_complete &&
      validation.logical_physical_ownership_complete &&
      validation.forbidden_boundaries_satisfied &&
      validation.qualification_boundary_satisfied;
  validation.descriptor_schema_complete =
      validation.issue_mask == 0U && validation.identity_schema_complete &&
      validation.constituent_design_complete &&
      validation.operator_bindings_complete &&
      validation.canonical_design_complete;
  return validation;
}

// This is a design-identity check, not runtime admission. It first requires
// the complete canonical constituent/ownership design and then verifies the
// one exact target witness plus prompt+one request capacity.
[[nodiscard]] constexpr bool sm87_aot_prefill_matches_exact_witness_design(
    const Sm87AotPrefillSystemPlan& plan,
    const std::size_t prompt_tokens,
    const std::size_t request_capacity_tokens) noexcept {
  const auto validation = validate_sm87_aot_prefill_system_plan(plan);
  return validation.canonical_design_complete &&
         prompt_tokens == plan.capacity.prompt.witness_prompt_tokens &&
         request_capacity_tokens ==
             plan.capacity.prompt.request_capacity_tokens &&
         kernels::sm87_target_aot_exact_witness_tokens(prompt_tokens);
}

static_assert(sm87_aot_prefill_system_capacity_plan(
                  kernels::Sm87TargetAotCapacityBucket::kP40)
                  .staging_span_count == 5U);
static_assert(sm87_aot_prefill_system_capacity_plan(
                  kernels::Sm87TargetAotCapacityBucket::kP60)
                  .staging_span_count == 8U);
static_assert(sm87_aot_prefill_system_capacity_plan(
                  kernels::Sm87TargetAotCapacityBucket::kP130)
                  .staging_span_count == 16U);
static_assert(sm87_aot_prefill_group_layer_occurrences(
                  Sm87AotPrefillExecutionGroup::kInputAndLayerNorm) == 64U);
static_assert(sm87_aot_prefill_group_layer_occurrences(
                  Sm87AotPrefillExecutionGroup::kLinearQkvZ) == 48U);
static_assert(sm87_aot_prefill_group_layer_occurrences(
                  Sm87AotPrefillExecutionGroup::kLinearAb) == 48U);
static_assert(sm87_aot_prefill_group_layer_occurrences(
                  Sm87AotPrefillExecutionGroup::kFullQkv) == 16U);
static_assert(sm87_aot_prefill_group_layer_occurrences(
                  Sm87AotPrefillExecutionGroup::
                      kFullQkNormRopePublish) == 16U);
static_assert(sm87_aot_prefill_group_layer_occurrences(
                  Sm87AotPrefillExecutionGroup::
                      kPostAttentionLayerNorm) == 64U);

}  // namespace q3x::runtime
