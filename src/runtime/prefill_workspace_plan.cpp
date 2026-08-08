#include "q3x/runtime/prefill_workspace_plan.h"

#include "q3x/model/model_config.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

constexpr std::uint32_t kBf16Bytes = 2U;
constexpr std::uint32_t kFp32Bytes = 4U;

[[nodiscard]] LayerMajorPrefillWorkspacePlanResult failure(
    const PrefillWorkspacePlanError error) noexcept {
  LayerMajorPrefillWorkspacePlanResult result;
  result.error = error;
  return result;
}

[[nodiscard]] bool checked_add(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

[[nodiscard]] bool checked_mul(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}

[[nodiscard]] bool checked_align(const std::uint64_t value,
                                 std::uint64_t& output) noexcept {
  constexpr std::uint64_t mask = kRequestArenaAlignment - 1U;
  static_assert((kRequestArenaAlignment & mask) == 0U);
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return false;
  }
  output = (value + mask) & ~mask;
  return true;
}

template <std::size_t Count>
[[nodiscard]] bool checked_sum(
    const std::array<std::uint64_t, Count>& values,
    std::uint64_t& output) noexcept {
  output = 0U;
  for (const std::uint64_t value : values) {
    if (!checked_add(output, value, output)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool make_typed_requirement(
    const std::uint64_t elements, const std::uint32_t element_size,
    const PrefillMemoryLifetime lifetime,
    const PrefillMemoryAliasCondition alias,
    PrefillMemoryRequirement& output) noexcept {
  std::uint64_t payload_bytes = 0U;
  std::uint64_t aligned_bytes = 0U;
  if (elements == 0U || element_size == 0U ||
      !checked_mul(elements, element_size, payload_bytes) ||
      !checked_align(payload_bytes, aligned_bytes)) {
    return false;
  }
  output.owner = PrefillMemoryOwner::kRequestStateArena;
  output.lifetime = lifetime;
  output.alias_condition = alias;
  output.required_bytes = aligned_bytes;
  output.element_capacity = elements;
  output.element_size_bytes = element_size;
  output.allocation_bound = false;
  output.alias_contract_bound = false;
  return true;
}

[[nodiscard]] bool make_raw_requirement(
    const std::uint64_t bytes, const PrefillMemoryLifetime lifetime,
    const PrefillMemoryAliasCondition alias,
    PrefillMemoryRequirement& output) noexcept {
  return make_typed_requirement(bytes, 1U, lifetime, alias, output);
}

[[nodiscard]] PrefillMemoryCapacityVerdict capacity(
    const std::uint64_t required, const std::uint64_t limit) noexcept {
  return required <= limit
             ? PrefillMemoryCapacityVerdict::kFitsDeclaredLimit
             : PrefillMemoryCapacityVerdict::kExceedsDeclaredLimit;
}

[[nodiscard]] bool valid_hidden_strategy(
    const PrefillHiddenStrategy strategy) noexcept {
  return strategy == PrefillHiddenStrategy::kSinglePromptWideConditional ||
         strategy == PrefillHiddenStrategy::kDoublePromptWideConservative;
}

[[nodiscard]] bool valid_scratch_strategy(
    const PrefillOperatorScratchStrategy strategy) noexcept {
  return strategy == PrefillOperatorScratchStrategy::
                         kC8192FamilyOverlayConditional ||
         strategy == PrefillOperatorScratchStrategy::
                         kOverlayLegacyC512MutuallyExclusive ||
         strategy == PrefillOperatorScratchStrategy::
                         kDisjointAllFamiliesAndLegacyC512;
}

[[nodiscard]] bool valid_requirement(
    const PrefillMemoryRequirement& requirement,
    const PrefillMemoryLifetime expected_lifetime) noexcept {
  std::uint64_t payload_bytes = 0U;
  return requirement.owner == PrefillMemoryOwner::kRequestStateArena &&
         requirement.lifetime == expected_lifetime &&
         requirement.required_bytes != 0U &&
         requirement.required_bytes % kRequestArenaAlignment == 0U &&
         requirement.element_capacity != 0U &&
         requirement.element_size_bytes != 0U &&
         checked_mul(requirement.element_capacity,
                     requirement.element_size_bytes, payload_bytes) &&
         payload_bytes <= requirement.required_bytes &&
         !requirement.allocation_bound && !requirement.alias_contract_bound;
}

[[nodiscard]] bool valid_plan(
    const LayerMajorPrefillWorkspacePlan& plan) noexcept {
  if (plan.sequence_capacity_tokens == 0U ||
      plan.operator_panel_capacity_tokens !=
          kLayerMajorPrefillOperatorPanelTokens ||
      plan.request_arena_limit_bytes == 0U ||
      plan.request_arena_owner != PrefillMemoryOwner::kRequestStateArena ||
      !valid_hidden_strategy(plan.selected.hidden_strategy) ||
      !valid_scratch_strategy(plan.selected.scratch_strategy) ||
      plan.minimum_conditional.required_bytes > plan.selected.required_bytes ||
      plan.selected.required_bytes > plan.conservative.required_bytes ||
      plan.minimum_conditional.capacity !=
          capacity(plan.minimum_conditional.required_bytes,
                   plan.request_arena_limit_bytes) ||
      plan.selected.capacity !=
          capacity(plan.selected.required_bytes,
                   plan.request_arena_limit_bytes) ||
      plan.conservative.capacity !=
          capacity(plan.conservative.required_bytes,
                   plan.request_arena_limit_bytes) ||
      !plan.minimum_conditional.requires_unbound_alias_or_route_contract ||
      plan.conservative.requires_unbound_alias_or_route_contract ||
      plan.resident_model.owner !=
          PrefillMemoryOwner::kEngineResidentModel ||
      plan.derived_sidecars.owner !=
          PrefillMemoryOwner::kDeploymentPlanSidecar ||
      plan.resident_model.required_bytes.has_value() ||
      plan.derived_sidecars.required_bytes.has_value() ||
      plan.resident_model.authenticated ||
      plan.derived_sidecars.authenticated ||
      plan.resident_model.residency_bound ||
      plan.derived_sidecars.residency_bound ||
      plan.whole_process_required_bytes.has_value() ||
      plan.whole_process_capacity !=
          PrefillMemoryCapacityVerdict::kIndeterminate ||
      plan.request_arena_reservation_bound ||
      plan.operator_bindings_complete || plan.executable()) {
    return false;
  }

  const auto& persistent = plan.persistent_state;
  const auto& position = plan.position_state;
  if (!valid_requirement(persistent.convolution_state,
                         PrefillMemoryLifetime::kRequestThroughDecode) ||
      !valid_requirement(persistent.gdn_recurrent_state,
                         PrefillMemoryLifetime::kRequestThroughDecode) ||
      !valid_requirement(persistent.full_attention_key_cache,
                         PrefillMemoryLifetime::kRequestThroughDecode) ||
      !valid_requirement(persistent.full_attention_value_cache,
                         PrefillMemoryLifetime::kRequestThroughDecode) ||
      !valid_requirement(position.rope_cos_fp32,
                         PrefillMemoryLifetime::kRequestThroughDecode) ||
      !valid_requirement(position.rope_sin_fp32,
                         PrefillMemoryLifetime::kRequestThroughDecode)) {
    return false;
  }

  std::uint64_t persistent_total = 0U;
  std::uint64_t position_total = 0U;
  if (!checked_sum(std::array<std::uint64_t, 4U>{
                       persistent.convolution_state.required_bytes,
                       persistent.gdn_recurrent_state.required_bytes,
                       persistent.full_attention_key_cache.required_bytes,
                       persistent.full_attention_value_cache.required_bytes},
                   persistent_total) ||
      !checked_sum(std::array<std::uint64_t, 2U>{
                       position.rope_cos_fp32.required_bytes,
                       position.rope_sin_fp32.required_bytes},
                   position_total) ||
      persistent_total != persistent.total_required_bytes ||
      position_total != position.total_required_bytes) {
    return false;
  }

  const auto& hidden = plan.prompt_wide_hidden;
  if (hidden.minimum_conditional.strategy !=
          PrefillHiddenStrategy::kSinglePromptWideConditional ||
      hidden.minimum_conditional.buffer_count != 1U ||
      !hidden.minimum_conditional.requires_panelwise_in_place_contract ||
      hidden.minimum_conditional.aggregate_bf16.alias_condition !=
          PrefillMemoryAliasCondition::
              kPanelwiseInputConsumedBeforeOutputOverwrite ||
      hidden.conservative.strategy !=
          PrefillHiddenStrategy::kDoublePromptWideConservative ||
      hidden.conservative.buffer_count != 2U ||
      hidden.conservative.requires_panelwise_in_place_contract ||
      hidden.conservative.aggregate_bf16.alias_condition !=
          PrefillMemoryAliasCondition::kDisjoint ||
      !valid_requirement(hidden.minimum_conditional.aggregate_bf16,
                         PrefillMemoryLifetime::kRequestPrefill) ||
      !valid_requirement(hidden.conservative.aggregate_bf16,
                         PrefillMemoryLifetime::kRequestPrefill) ||
      !valid_requirement(hidden.selected.aggregate_bf16,
                         PrefillMemoryLifetime::kRequestPrefill)) {
    return false;
  }

  const auto& scratch = plan.operator_scratch;
  if (scratch.c8192_panel_capacity_tokens !=
          kLayerMajorPrefillOperatorPanelTokens ||
      scratch.legacy_c512_panel_capacity_tokens !=
          kLayerMajorPrefillLegacyPublicTileTokens ||
      scratch.gate_up.role != PrefillProjectionRole::kGateUp ||
      scratch.gate_up.maximum_m !=
          kLayerMajorPrefillOperatorPanelTokens ||
      scratch.gate_up.n != 17'408U || scratch.gate_up.k != 5'120U ||
      scratch.gate_up.logical_matrix_count != 2U ||
      scratch.gate_up.tactic_bound ||
      scratch.down.role != PrefillProjectionRole::kDown ||
      scratch.down.maximum_m != kLayerMajorPrefillOperatorPanelTokens ||
      scratch.down.n != 5'120U || scratch.down.k != 17'408U ||
      scratch.down.logical_matrix_count != 1U ||
      scratch.down.tactic_bound) {
    return false;
  }

  for (const PrefillOperatorFamilyLiveSet& live_set :
       scratch.c8192_family_live_sets) {
    if (!valid_requirement(live_set.aggregate,
                           PrefillMemoryLifetime::kOperatorPanel)) {
      return false;
    }
  }
  const PrefillOperatorScratchVariant* const variants[] = {
      &scratch.c8192_family_overlay_conditional,
      &scratch.c8192_disjoint_families,
      &scratch.legacy_c512_only,
      &scratch.mutually_exclusive_overlay,
      &scratch.disjoint_conservative,
      &scratch.selected};
  for (const PrefillOperatorScratchVariant* const variant : variants) {
    if (!valid_requirement(variant->aggregate,
                           PrefillMemoryLifetime::kOperatorPanel) ||
        variant->total_required_bytes != variant->aggregate.required_bytes) {
      return false;
    }
  }
  return true;
}

}  // namespace

LayerMajorPrefillWorkspacePlanResult
build_unbound_layer_major_prefill_workspace_plan(
    const LayerMajorPrefillWorkspaceOptions& options) noexcept {
  if (options.sequence_capacity_tokens == 0U ||
      options.request_arena_limit_bytes == 0U ||
      !valid_hidden_strategy(options.hidden_strategy) ||
      !valid_scratch_strategy(options.scratch_strategy)) {
    return failure(PrefillWorkspacePlanError::kInvalidArgument);
  }

  const model::ModelConfig* const config =
      model::find_known_model(model::KnownModel::kQwen36_27B);
  if (config == nullptr || !model::validate(*config) || config->is_moe() ||
      config->num_hidden_layers != kLayerMajorPrefillLayerCount ||
      config->num_linear_attention_layers() !=
          kLayerMajorPrefillLinearLayerCount ||
      config->num_full_attention_layers() !=
          kLayerMajorPrefillFullLayerCount ||
      config->hidden_size != 5'120U ||
      config->intermediate_size != 17'408U ||
      config->linear_conv_kernel_dim < 2U ||
      config->rotary_dim() == 0U || config->rotary_dim() % 2U != 0U) {
    return failure(PrefillWorkspacePlanError::kModelContractMismatch);
  }

  const std::uint64_t sequence = options.sequence_capacity_tokens;
  const std::uint64_t linear_layers =
      config->num_linear_attention_layers();
  const std::uint64_t full_layers = config->num_full_attention_layers();
  const std::uint64_t qkv_width = config->linear_qkv_projection_dim();
  const std::uint64_t value_width = config->linear_value_dim();

  std::uint64_t conv_elements = 0U;
  std::uint64_t gdn_elements = 0U;
  std::uint64_t key_elements = 0U;
  std::uint64_t value_elements = 0U;
  std::uint64_t rope_elements = 0U;
  std::uint64_t prompt_hidden_elements = 0U;
  if (!checked_mul(linear_layers, qkv_width, conv_elements) ||
      !checked_mul(conv_elements, config->linear_conv_kernel_dim - 1U,
                   conv_elements) ||
      !checked_mul(linear_layers,
                   config->linear_state_elements_per_layer(), gdn_elements) ||
      !checked_mul(full_layers, sequence, key_elements) ||
      !checked_mul(key_elements, config->kv_dim(), key_elements) ||
      !checked_mul(full_layers, sequence, value_elements) ||
      !checked_mul(value_elements, config->kv_dim(), value_elements) ||
      !checked_mul(sequence, config->rotary_dim() / 2U, rope_elements) ||
      !checked_mul(sequence, config->hidden_size,
                   prompt_hidden_elements)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  if (sequence > kLayerMajorPrefillMaximumSequenceTokens ||
      sequence > std::numeric_limits<std::uint32_t>::max()) {
    return failure(PrefillWorkspacePlanError::kCapacityExceeded);
  }

  LayerMajorPrefillWorkspacePlan plan;
  plan.sequence_capacity_tokens = static_cast<std::uint32_t>(sequence);
  plan.request_arena_limit_bytes = options.request_arena_limit_bytes;

  auto& persistent = plan.persistent_state;
  if (!make_typed_requirement(
          conv_elements, kBf16Bytes,
          PrefillMemoryLifetime::kRequestThroughDecode,
          PrefillMemoryAliasCondition::kDisjoint,
          persistent.convolution_state) ||
      !make_typed_requirement(
          gdn_elements, kBf16Bytes,
          PrefillMemoryLifetime::kRequestThroughDecode,
          PrefillMemoryAliasCondition::kDisjoint,
          persistent.gdn_recurrent_state) ||
      !make_typed_requirement(
          key_elements, kBf16Bytes,
          PrefillMemoryLifetime::kRequestThroughDecode,
          PrefillMemoryAliasCondition::kDisjoint,
          persistent.full_attention_key_cache) ||
      !make_typed_requirement(
          value_elements, kBf16Bytes,
          PrefillMemoryLifetime::kRequestThroughDecode,
          PrefillMemoryAliasCondition::kDisjoint,
          persistent.full_attention_value_cache) ||
      !checked_sum(std::array<std::uint64_t, 4U>{
                       persistent.convolution_state.required_bytes,
                       persistent.gdn_recurrent_state.required_bytes,
                       persistent.full_attention_key_cache.required_bytes,
                       persistent.full_attention_value_cache.required_bytes},
                   persistent.total_required_bytes)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }

  auto& position = plan.position_state;
  if (!make_typed_requirement(
          rope_elements, kFp32Bytes,
          PrefillMemoryLifetime::kRequestThroughDecode,
          PrefillMemoryAliasCondition::kDisjoint,
          position.rope_cos_fp32) ||
      !make_typed_requirement(
          rope_elements, kFp32Bytes,
          PrefillMemoryLifetime::kRequestThroughDecode,
          PrefillMemoryAliasCondition::kDisjoint,
          position.rope_sin_fp32) ||
      !checked_sum(std::array<std::uint64_t, 2U>{
                       position.rope_cos_fp32.required_bytes,
                       position.rope_sin_fp32.required_bytes},
                   position.total_required_bytes)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }

  auto& hidden = plan.prompt_wide_hidden;
  hidden.minimum_conditional.strategy =
      PrefillHiddenStrategy::kSinglePromptWideConditional;
  hidden.minimum_conditional.buffer_count = 1U;
  hidden.minimum_conditional.hidden_width = config->hidden_size;
  hidden.minimum_conditional.sequence_capacity_tokens =
      plan.sequence_capacity_tokens;
  hidden.minimum_conditional.requires_panelwise_in_place_contract = true;
  if (!make_typed_requirement(
          prompt_hidden_elements, kBf16Bytes,
          PrefillMemoryLifetime::kRequestPrefill,
          PrefillMemoryAliasCondition::
              kPanelwiseInputConsumedBeforeOutputOverwrite,
          hidden.minimum_conditional.aggregate_bf16)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  std::uint64_t double_hidden_elements = 0U;
  if (!checked_mul(prompt_hidden_elements, 2U, double_hidden_elements)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  hidden.conservative.strategy =
      PrefillHiddenStrategy::kDoublePromptWideConservative;
  hidden.conservative.buffer_count = 2U;
  hidden.conservative.hidden_width = config->hidden_size;
  hidden.conservative.sequence_capacity_tokens = plan.sequence_capacity_tokens;
  if (!make_typed_requirement(
          double_hidden_elements, kBf16Bytes,
          PrefillMemoryLifetime::kRequestPrefill,
          PrefillMemoryAliasCondition::kDisjoint,
          hidden.conservative.aggregate_bf16)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  hidden.selected =
      options.hidden_strategy ==
              PrefillHiddenStrategy::kSinglePromptWideConditional
          ? hidden.minimum_conditional
          : hidden.conservative;

  auto& scratch = plan.operator_scratch;
  scratch.c8192_panel_capacity_tokens =
      kLayerMajorPrefillOperatorPanelTokens;
  scratch.legacy_c512_panel_capacity_tokens =
      kLayerMajorPrefillLegacyPublicTileTokens;

  const std::uint64_t panel = kLayerMajorPrefillOperatorPanelTokens;
  const std::uint64_t gdn_qkvz_width = qkv_width + value_width;
  const std::uint64_t gdn_ba_width =
      2U * config->linear_num_value_heads;
  const std::uint64_t gdn_fp32_gate_width = gdn_ba_width;
  const std::uint64_t attention_projection_width =
      config->q_projection_dim() + 2U * config->kv_dim();
  // Without a bound in-place/consumer-epilogue contract, the merged Gate+Up
  // projection [M, 2I] and its activated Down input [M, I] coexist until the
  // SiLU/gate consumer finishes producing the complete activated panel.
  const std::uint64_t gate_up_live_width =
      3U * config->intermediate_size;

  std::uint64_t gdn_projection_elements = 0U;
  std::uint64_t gdn_projection_bytes = 0U;
  std::uint64_t gdn_post_bf16_elements = 0U;
  std::uint64_t gdn_post_bf16_bytes = 0U;
  std::uint64_t gdn_post_fp32_elements = 0U;
  std::uint64_t gdn_post_fp32_bytes = 0U;
  std::uint64_t gdn_post_bytes = 0U;
  std::uint64_t attention_elements = 0U;
  std::uint64_t attention_bytes = 0U;
  std::uint64_t gate_up_elements = 0U;
  std::uint64_t gate_up_bytes = 0U;
  if (!checked_mul(panel, gdn_qkvz_width + gdn_ba_width,
                   gdn_projection_elements) ||
      !checked_mul(gdn_projection_elements, kBf16Bytes,
                   gdn_projection_bytes) ||
      // q/k/v plus retained z and the recurrent core output coexist until
      // GDN normalization/output projection consumes them.
      !checked_mul(panel, qkv_width + value_width + value_width,
                   gdn_post_bf16_elements) ||
      !checked_mul(gdn_post_bf16_elements, kBf16Bytes,
                   gdn_post_bf16_bytes) ||
      !checked_mul(panel, gdn_fp32_gate_width,
                   gdn_post_fp32_elements) ||
      !checked_mul(gdn_post_fp32_elements, kFp32Bytes,
                   gdn_post_fp32_bytes) ||
      !checked_add(gdn_post_bf16_bytes, gdn_post_fp32_bytes,
                   gdn_post_bytes) ||
      !checked_mul(panel,
                   attention_projection_width + config->q_dim(),
                   attention_elements) ||
      !checked_mul(attention_elements, kBf16Bytes, attention_bytes) ||
      !checked_mul(panel, gate_up_live_width, gate_up_elements) ||
      !checked_mul(gate_up_elements, kBf16Bytes, gate_up_bytes)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }

  const std::array<std::uint64_t, kLayerMajorPrefillScratchFamilyCount>
      family_bytes{gdn_projection_bytes, gdn_post_bytes, attention_bytes,
                   gate_up_bytes};
  const std::array<PrefillScratchFamily,
                   kLayerMajorPrefillScratchFamilyCount>
      families{PrefillScratchFamily::kGdnMergedInputProjection,
               PrefillScratchFamily::kGdnFusedPostConvPrep,
               PrefillScratchFamily::kFullAttentionProjectionAndCore,
               PrefillScratchFamily::kMlpMergedGateUp};
  const std::array<PrefillScratchProducer,
                   kLayerMajorPrefillScratchFamilyCount>
      producers{PrefillScratchProducer::kGdnInProjQkvzAndInProjBa,
                PrefillScratchProducer::kGdnFusedPostConvPrepQkvGBeta,
                PrefillScratchProducer::kFullAttentionQkvGateProjection,
                PrefillScratchProducer::kMlpMergedGateUpProjection};
  const std::array<PrefillScratchLastConsumer,
                   kLayerMajorPrefillScratchFamilyCount>
      consumers{
          PrefillScratchLastConsumer::kGdnFusedPostConvPrep,
          PrefillScratchLastConsumer::
              kGdnRecurrentUpdateNormAndOutputProjection,
          PrefillScratchLastConsumer::
              kFullAttentionCoreGateAndOutputProjection,
          PrefillScratchLastConsumer::kMlpSiluGateAndDownProjection};
  for (std::size_t index = 0U; index < family_bytes.size(); ++index) {
    auto& live_set = scratch.c8192_family_live_sets[index];
    live_set.family = families[index];
    live_set.producer = producers[index];
    live_set.last_consumer = consumers[index];
    if (!make_raw_requirement(family_bytes[index],
                              PrefillMemoryLifetime::kOperatorPanel,
                              PrefillMemoryAliasCondition::kDisjoint,
                              live_set.aggregate)) {
      return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
    }
  }

  std::uint64_t family_disjoint_bytes = 0U;
  if (!checked_sum(family_bytes, family_disjoint_bytes)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  const std::uint64_t family_overlay_bytes =
      *std::max_element(family_bytes.begin(), family_bytes.end());

  auto& family_overlay = scratch.c8192_family_overlay_conditional;
  family_overlay.strategy = PrefillOperatorScratchStrategy::
      kC8192FamilyOverlayConditional;
  family_overlay.requires_legacy_route_exclusion = true;
  family_overlay.requires_family_completion_events = true;
  if (!make_raw_requirement(
          family_overlay_bytes, PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kSequentialFamilyLiveSetOverlay,
          family_overlay.aggregate)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  family_overlay.total_required_bytes =
      family_overlay.aggregate.required_bytes;

  auto& family_disjoint = scratch.c8192_disjoint_families;
  family_disjoint.strategy = PrefillOperatorScratchStrategy::
      kDisjointAllFamiliesAndLegacyC512;
  family_disjoint.requires_legacy_route_exclusion = true;
  if (!make_raw_requirement(family_disjoint_bytes,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            family_disjoint.aggregate)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  family_disjoint.total_required_bytes =
      family_disjoint.aggregate.required_bytes;

  std::uint64_t legacy_hidden_elements = 0U;
  std::uint64_t legacy_projection_elements = 0U;
  std::uint64_t legacy_linear_elements = 0U;
  std::uint64_t legacy_probability_elements = 0U;
  if (!checked_mul(kLegacyC512HiddenScratchBufferCount,
                   kLayerMajorPrefillLegacyPublicTileTokens,
                   legacy_hidden_elements) ||
      !checked_mul(legacy_hidden_elements, config->hidden_size,
                   legacy_hidden_elements) ||
      !checked_mul(kLegacyC512ProjectionScratchBufferCount,
                   kLayerMajorPrefillLegacyPublicTileTokens,
                   legacy_projection_elements) ||
      !checked_mul(legacy_projection_elements, config->intermediate_size,
                   legacy_projection_elements) ||
      !checked_mul(kLegacyC512LinearScalarScratchBufferCount,
                   kLayerMajorPrefillLegacyPublicTileTokens,
                   legacy_linear_elements) ||
      !checked_mul(legacy_linear_elements, config->linear_num_value_heads,
                   legacy_linear_elements) ||
      !checked_mul(sequence, config->num_attention_heads,
                   legacy_probability_elements)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  const std::uint64_t legacy_fp32_elements =
      std::max(kLayerMajorPrefillFp32MinimumScratchElements,
               legacy_probability_elements);
  if (!make_typed_requirement(
          legacy_hidden_elements, kBf16Bytes,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kDisjoint,
          scratch.legacy_c512_hidden_bf16) ||
      !make_typed_requirement(
          legacy_projection_elements, kBf16Bytes,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kDisjoint,
          scratch.legacy_c512_projection_bf16) ||
      !make_typed_requirement(
          legacy_linear_elements, kBf16Bytes,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kDisjoint,
          scratch.legacy_c512_linear_scalar_bf16) ||
      !make_typed_requirement(
          legacy_fp32_elements, kFp32Bytes,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kDisjoint,
          scratch.legacy_c512_fp32_scratch)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  std::uint64_t legacy_bytes = 0U;
  if (!checked_sum(std::array<std::uint64_t, 4U>{
                       scratch.legacy_c512_hidden_bf16.required_bytes,
                       scratch.legacy_c512_projection_bf16.required_bytes,
                       scratch.legacy_c512_linear_scalar_bf16.required_bytes,
                       scratch.legacy_c512_fp32_scratch.required_bytes},
                   legacy_bytes)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  auto& legacy = scratch.legacy_c512_only;
  if (!make_raw_requirement(legacy_bytes,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            legacy.aggregate)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  legacy.total_required_bytes = legacy.aggregate.required_bytes;

  auto& route_overlay = scratch.mutually_exclusive_overlay;
  route_overlay.strategy = PrefillOperatorScratchStrategy::
      kOverlayLegacyC512MutuallyExclusive;
  route_overlay.requires_family_completion_events = true;
  route_overlay.requires_route_mutual_exclusion_event = true;
  if (!make_raw_requirement(
          std::max(family_overlay.total_required_bytes,
                   legacy.total_required_bytes),
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kMutuallyExclusiveRouteOverlay,
          route_overlay.aggregate)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  route_overlay.total_required_bytes =
      route_overlay.aggregate.required_bytes;

  std::uint64_t all_disjoint_bytes = 0U;
  if (!checked_add(family_disjoint.total_required_bytes,
                   legacy.total_required_bytes, all_disjoint_bytes)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  auto& all_disjoint = scratch.disjoint_conservative;
  all_disjoint.strategy = PrefillOperatorScratchStrategy::
      kDisjointAllFamiliesAndLegacyC512;
  if (!make_raw_requirement(all_disjoint_bytes,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            all_disjoint.aggregate)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  all_disjoint.total_required_bytes =
      all_disjoint.aggregate.required_bytes;

  switch (options.scratch_strategy) {
    case PrefillOperatorScratchStrategy::kC8192FamilyOverlayConditional:
      scratch.selected = family_overlay;
      break;
    case PrefillOperatorScratchStrategy::
        kOverlayLegacyC512MutuallyExclusive:
      scratch.selected = route_overlay;
      break;
    case PrefillOperatorScratchStrategy::
        kDisjointAllFamiliesAndLegacyC512:
      scratch.selected = all_disjoint;
      break;
    case PrefillOperatorScratchStrategy::kUnselected:
      return failure(PrefillWorkspacePlanError::kInvalidArgument);
  }

  scratch.gate_up = PrefillProjectionShapeRequirement{
      PrefillProjectionRole::kGateUp,
      kLayerMajorPrefillOperatorPanelTokens,
      config->intermediate_size,
      config->hidden_size,
      2U,
      false};
  scratch.down = PrefillProjectionShapeRequirement{
      PrefillProjectionRole::kDown,
      kLayerMajorPrefillOperatorPanelTokens,
      config->hidden_size,
      config->intermediate_size,
      1U,
      false};

  std::uint64_t common_bytes = 0U;
  if (!checked_add(persistent.total_required_bytes,
                   position.total_required_bytes, common_bytes) ||
      !checked_sum(std::array<std::uint64_t, 3U>{
                       common_bytes,
                       hidden.minimum_conditional.aggregate_bf16.required_bytes,
                       family_overlay.total_required_bytes},
                   plan.minimum_conditional.required_bytes) ||
      !checked_sum(std::array<std::uint64_t, 3U>{
                       common_bytes,
                       hidden.selected.aggregate_bf16.required_bytes,
                       scratch.selected.total_required_bytes},
                   plan.selected.required_bytes) ||
      !checked_sum(std::array<std::uint64_t, 3U>{
                       common_bytes,
                       hidden.conservative.aggregate_bf16.required_bytes,
                       all_disjoint.total_required_bytes},
                   plan.conservative.required_bytes)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }

  plan.minimum_conditional.hidden_strategy =
      PrefillHiddenStrategy::kSinglePromptWideConditional;
  plan.minimum_conditional.scratch_strategy =
      PrefillOperatorScratchStrategy::kC8192FamilyOverlayConditional;
  plan.minimum_conditional.capacity =
      capacity(plan.minimum_conditional.required_bytes,
               plan.request_arena_limit_bytes);
  plan.minimum_conditional.requires_unbound_alias_or_route_contract = true;

  plan.selected.hidden_strategy = options.hidden_strategy;
  plan.selected.scratch_strategy = options.scratch_strategy;
  plan.selected.capacity =
      capacity(plan.selected.required_bytes, plan.request_arena_limit_bytes);
  plan.selected.requires_unbound_alias_or_route_contract =
      hidden.selected.requires_panelwise_in_place_contract ||
      scratch.selected.requires_legacy_route_exclusion ||
      scratch.selected.requires_family_completion_events ||
      scratch.selected.requires_route_mutual_exclusion_event;

  plan.conservative.hidden_strategy =
      PrefillHiddenStrategy::kDoublePromptWideConservative;
  plan.conservative.scratch_strategy = PrefillOperatorScratchStrategy::
      kDisjointAllFamiliesAndLegacyC512;
  plan.conservative.capacity =
      capacity(plan.conservative.required_bytes,
               plan.request_arena_limit_bytes);

  plan.resident_model.owner = PrefillMemoryOwner::kEngineResidentModel;
  plan.resident_model.lifetime = PrefillMemoryLifetime::kEngine;
  plan.derived_sidecars.owner =
      PrefillMemoryOwner::kDeploymentPlanSidecar;
  plan.derived_sidecars.lifetime = PrefillMemoryLifetime::kEngine;
  plan.whole_process_capacity =
      PrefillMemoryCapacityVerdict::kIndeterminate;

  if (!valid_plan(plan)) {
    return failure(PrefillWorkspacePlanError::kInvalidLayout);
  }

  LayerMajorPrefillWorkspacePlanResult result;
  result.value.emplace(plan);
  return result;
}

}  // namespace q3x::runtime
