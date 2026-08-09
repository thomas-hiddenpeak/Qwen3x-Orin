#include "q3x/runtime/prefill_workspace_plan.h"

#include "q3x/kernels/sm87_fp8_marlin_w8a16.h"
#include "q3x/kernels/sm87_nvfp4_marlin.h"
#include "q3x/model/model_config.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

constexpr std::uint32_t kBf16Bytes = 2U;
constexpr std::uint32_t kFp32Bytes = 4U;
constexpr std::uint32_t kU32Bytes = sizeof(std::uint32_t);

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
                         kC8192FamilyOverlayWithDisjointLegacyC512 ||
         strategy == PrefillOperatorScratchStrategy::
                         kDisjointAllFamiliesAndLegacyC512;
}

[[nodiscard]] bool valid_gdn_tactic(
    const PrefillGdnPhysicalTactic tactic) noexcept {
  return tactic == PrefillGdnPhysicalTactic::kC64NativeInPlaceConv ||
         tactic ==
             PrefillGdnPhysicalTactic::kC64NativeTokenParallelConv;
}

[[nodiscard]] bool valid_legacy_gdn_tactic(
    const PrefillLegacyGdnPhysicalTactic tactic) noexcept {
  return tactic == PrefillLegacyGdnPhysicalTactic::kC16Composite ||
         tactic == PrefillLegacyGdnPhysicalTactic::kC64Native;
}

[[nodiscard]] bool valid_mlp_tactic(
    const PrefillMlpPhysicalTactic tactic) noexcept {
  return tactic == PrefillMlpPhysicalTactic::kSeparateGateUpAndSilu ||
         tactic == PrefillMlpPhysicalTactic::kFusedGateUpEpilogue;
}

[[nodiscard]] bool valid_layer_major_model_contract(
    const model::ModelConfig* const config) noexcept {
  return config != nullptr && model::validate(*config) && !config->is_moe() &&
         config->num_hidden_layers == kLayerMajorPrefillLayerCount &&
         config->num_linear_attention_layers() ==
             kLayerMajorPrefillLinearLayerCount &&
         config->num_full_attention_layers() ==
             kLayerMajorPrefillFullLayerCount &&
         config->hidden_size == 5'120U &&
         config->intermediate_size == 17'408U &&
         config->linear_conv_kernel_dim >= 2U && config->rotary_dim() != 0U &&
         config->rotary_dim() % 2U == 0U;
}

struct PrefillTacticByteRequirements {
  std::uint64_t gdn_projection_phase = 0U;
  std::uint64_t gdn_recurrent_phase = 0U;
  std::uint64_t gdn_native_workspace = 0U;
  std::uint64_t gdn_token_parallel_conv_output = 0U;
  std::uint64_t attention_preprocess_phase = 0U;
  std::uint64_t mlp_gate_up_down_phase = 0U;
  std::uint64_t projection_reduction_and_locks = 0U;
  std::uint64_t legacy_gdn_native_workspace = 0U;
};

[[nodiscard]] bool checked_tactic_byte_requirements(
    const model::ModelConfig& config,
    const PrefillGdnPhysicalTactic gdn_tactic,
    const PrefillLegacyGdnPhysicalTactic legacy_gdn_tactic,
    const PrefillMlpPhysicalTactic mlp_tactic,
    PrefillTacticByteRequirements& output) noexcept {
  if (!valid_gdn_tactic(gdn_tactic) ||
      !valid_legacy_gdn_tactic(legacy_gdn_tactic) ||
      !valid_mlp_tactic(mlp_tactic)) {
    return false;
  }

  PrefillTacticByteRequirements result;
  const std::uint64_t panel = kLayerMajorPrefillOperatorPanelTokens;
  const std::uint64_t qkv_width = config.linear_qkv_projection_dim();
  const std::uint64_t value_width = config.linear_value_dim();
  const std::uint64_t gdn_qkvz_width = qkv_width + value_width;
  const std::uint64_t gdn_ba_width = 2U * config.linear_num_value_heads;
  const std::uint64_t attention_live_width =
      config.q_projection_dim() + 2U * config.q_dim();
  const std::uint64_t mlp_separate_live_width =
      3U * config.intermediate_size;

  std::uint64_t elements = 0U;
  std::uint64_t recurrent_base_bytes = 0U;
  std::uint64_t token_parallel_bytes = 0U;
  if (!checked_mul(panel,
                   config.hidden_size + gdn_qkvz_width + gdn_ba_width,
                   elements) ||
      !checked_mul(elements, kBf16Bytes, result.gdn_projection_phase) ||
      !checked_mul(panel,
                   gdn_qkvz_width + gdn_ba_width + value_width,
                   elements) ||
      !checked_mul(elements, kBf16Bytes, recurrent_base_bytes) ||
      !checked_add(recurrent_base_bytes,
                   kLayerMajorPrefillGdnC64NativeWorkspaceBytes,
                   result.gdn_recurrent_phase) ||
      !checked_mul(kLayerMajorPrefillLegacyPublicTileTokens, qkv_width,
                   elements) ||
      !checked_mul(elements, kBf16Bytes, token_parallel_bytes) ||
      !checked_mul(panel, attention_live_width, elements) ||
      !checked_mul(elements, kBf16Bytes,
                   result.attention_preprocess_phase) ||
      !checked_mul(panel, mlp_separate_live_width, elements) ||
      !checked_mul(elements, kBf16Bytes,
                   result.mlp_gate_up_down_phase)) {
    return false;
  }

  result.gdn_native_workspace =
      kLayerMajorPrefillGdnC64NativeWorkspaceBytes;
  if (gdn_tactic ==
      PrefillGdnPhysicalTactic::kC64NativeTokenParallelConv) {
    result.gdn_token_parallel_conv_output = token_parallel_bytes;
    if (!checked_add(result.gdn_recurrent_phase, token_parallel_bytes,
                     result.gdn_recurrent_phase)) {
      return false;
    }
  }

  constexpr std::uint64_t maximum_projection_reduction_bytes =
      kernels::kSm87NvFp4MarlinReductionBytes >
              kernels::kSm87Fp8MarlinReductionBytes
          ? kernels::kSm87NvFp4MarlinReductionBytes
          : kernels::kSm87Fp8MarlinReductionBytes;
  constexpr std::uint64_t maximum_projection_lock_bytes =
      kernels::kSm87NvFp4MarlinLockBytes >
              kernels::kSm87Fp8MarlinLockBytes
          ? kernels::kSm87NvFp4MarlinLockBytes
          : kernels::kSm87Fp8MarlinLockBytes;
  std::uint64_t projection_payload = 0U;
  if (!checked_add(maximum_projection_reduction_bytes,
                   maximum_projection_lock_bytes, projection_payload) ||
      !checked_align(projection_payload,
                     result.projection_reduction_and_locks)) {
    return false;
  }

  if (mlp_tactic == PrefillMlpPhysicalTactic::kFusedGateUpEpilogue) {
    std::uint64_t panel_hidden_bytes = 0U;
    if (!checked_mul(panel, config.hidden_size, elements) ||
        !checked_mul(elements, kBf16Bytes, panel_hidden_bytes) ||
        !checked_add(result.mlp_gate_up_down_phase, panel_hidden_bytes,
                     result.mlp_gate_up_down_phase) ||
        !checked_add(result.mlp_gate_up_down_phase,
                     result.projection_reduction_and_locks,
                     result.mlp_gate_up_down_phase)) {
      return false;
    }
  }

  if (legacy_gdn_tactic == PrefillLegacyGdnPhysicalTactic::kC64Native) {
    result.legacy_gdn_native_workspace =
        kLayerMajorPrefillGdnC64NativeWorkspaceBytes;
  }
  output = result;
  return true;
}

[[nodiscard]] PrefillWorkspaceBackingContract backing_contract(
    const PrefillWorkspaceBackingIdentity identity,
    const PrefillWorkspacePhaseOwnership phase_ownership,
    const std::uint64_t bytes_per_instance,
    const std::uint32_t minimum_instance_count) noexcept {
  PrefillWorkspaceBackingContract result;
  result.identity = identity;
  result.phase_ownership = phase_ownership;
  result.bytes_per_instance = bytes_per_instance;
  result.minimum_instance_count = minimum_instance_count;
  result.capacity_included_in_variant_total = minimum_instance_count != 0U;
  result.subrange_binding_required = minimum_instance_count != 0U;
  return result;
}

enum class ScratchVariantTopology : std::uint8_t {
  kC8192FamilyOverlayOnly = 0,
  kC8192DisjointFamiliesOnly,
  kLegacyC512Only,
  kMutuallyExclusiveRouteOverlay,
  kC8192FamilyOverlayWithDisjointLegacyC512,
  kDisjointAllFamiliesAndLegacyC512,
};

struct VariantBackingRequirements {
  PrefillWorkspaceBackingContract prompt_token_ids;
  PrefillWorkspaceBackingContract projection;
  PrefillWorkspaceBackingContract gdn_native;
};

[[nodiscard]] VariantBackingRequirements variant_backing_requirements(
    const ScratchVariantTopology topology,
    const PrefillLegacyGdnPhysicalTactic legacy_gdn_tactic,
    const PrefillTacticByteRequirements& bytes) noexcept {
  VariantBackingRequirements result;
  result.prompt_token_ids = backing_contract(
      PrefillWorkspaceBackingIdentity::kOperatorArenaPromptTokenPrefix,
      PrefillWorkspacePhaseOwnership::kPromptTokenIdsBeforeOperatorFamilies,
      kLayerMajorPrefillOperatorPanelTokens * kU32Bytes, 1U);
  switch (topology) {
    case ScratchVariantTopology::kC8192FamilyOverlayOnly:
    case ScratchVariantTopology::kMutuallyExclusiveRouteOverlay:
    case ScratchVariantTopology::
        kC8192FamilyOverlayWithDisjointLegacyC512:
      result.projection = backing_contract(
          PrefillWorkspaceBackingIdentity::kC8192SequentialFamilyPhaseArena,
          PrefillWorkspacePhaseOwnership::kC8192ProjectionFamilies,
          bytes.projection_reduction_and_locks, 1U);
      break;
    case ScratchVariantTopology::kC8192DisjointFamiliesOnly:
    case ScratchVariantTopology::kDisjointAllFamiliesAndLegacyC512:
      result.projection = backing_contract(
          PrefillWorkspaceBackingIdentity::kC8192DisjointFamilyPhaseArenas,
          PrefillWorkspacePhaseOwnership::kC8192ProjectionFamilies,
          bytes.projection_reduction_and_locks,
          static_cast<std::uint32_t>(
              kLayerMajorPrefillScratchFamilyCount));
      break;
    case ScratchVariantTopology::kLegacyC512Only:
      break;
  }

  const bool native_legacy =
      legacy_gdn_tactic == PrefillLegacyGdnPhysicalTactic::kC64Native;
  switch (topology) {
    case ScratchVariantTopology::kC8192FamilyOverlayOnly:
    case ScratchVariantTopology::kC8192DisjointFamiliesOnly:
      result.gdn_native = backing_contract(
          PrefillWorkspaceBackingIdentity::kC8192GdnNativeArena,
          PrefillWorkspacePhaseOwnership::kC8192GdnRecurrentPhase,
          bytes.gdn_native_workspace, 1U);
      break;
    case ScratchVariantTopology::kLegacyC512Only:
      if (native_legacy) {
        result.gdn_native = backing_contract(
            PrefillWorkspaceBackingIdentity::kLegacyC512GdnNativeArena,
            PrefillWorkspacePhaseOwnership::kLegacyC512GdnRoute,
            bytes.legacy_gdn_native_workspace, 1U);
      }
      break;
    case ScratchVariantTopology::kMutuallyExclusiveRouteOverlay:
      result.gdn_native = native_legacy
                              ? backing_contract(
                                    PrefillWorkspaceBackingIdentity::
                                        kMutuallyExclusiveC8192LegacyGdnNativeArena,
                                    PrefillWorkspacePhaseOwnership::
                                        kMutuallyExclusiveC8192LegacyGdnRoutes,
                                    bytes.gdn_native_workspace, 1U)
                              : backing_contract(
                                    PrefillWorkspaceBackingIdentity::
                                        kC8192GdnNativeArena,
                                    PrefillWorkspacePhaseOwnership::
                                        kC8192GdnRecurrentPhase,
                                    bytes.gdn_native_workspace, 1U);
      break;
    case ScratchVariantTopology::
        kC8192FamilyOverlayWithDisjointLegacyC512:
    case ScratchVariantTopology::kDisjointAllFamiliesAndLegacyC512:
      result.gdn_native = native_legacy
                              ? backing_contract(
                                    PrefillWorkspaceBackingIdentity::
                                        kDisjointC8192LegacyGdnNativeArenas,
                                    PrefillWorkspacePhaseOwnership::
                                        kDisjointC8192LegacyGdnRoutes,
                                    bytes.gdn_native_workspace, 2U)
                              : backing_contract(
                                    PrefillWorkspaceBackingIdentity::
                                        kC8192GdnNativeArena,
                                    PrefillWorkspacePhaseOwnership::
                                        kC8192GdnRecurrentPhase,
                                    bytes.gdn_native_workspace, 1U);
      break;
  }
  return result;
}

[[nodiscard]] bool same_requirement(
    const PrefillMemoryRequirement& left,
    const PrefillMemoryRequirement& right) noexcept {
  return left.owner == right.owner && left.lifetime == right.lifetime &&
         left.alias_condition == right.alias_condition &&
         left.required_bytes == right.required_bytes &&
         left.element_capacity == right.element_capacity &&
         left.element_size_bytes == right.element_size_bytes &&
         left.allocation_bound == right.allocation_bound &&
         left.alias_contract_bound == right.alias_contract_bound;
}

[[nodiscard]] bool same_backing_contract(
    const PrefillWorkspaceBackingContract& left,
    const PrefillWorkspaceBackingContract& right) noexcept {
  return left.identity == right.identity &&
         left.phase_ownership == right.phase_ownership &&
         left.bytes_per_instance == right.bytes_per_instance &&
         left.minimum_instance_count == right.minimum_instance_count &&
         left.capacity_included_in_variant_total ==
             right.capacity_included_in_variant_total &&
         left.subrange_binding_required == right.subrange_binding_required;
}

[[nodiscard]] bool same_hidden_variant(
    const PrefillPromptWideHiddenVariant& left,
    const PrefillPromptWideHiddenVariant& right) noexcept {
  return left.strategy == right.strategy &&
         same_requirement(left.aggregate_bf16, right.aggregate_bf16) &&
         left.buffer_count == right.buffer_count &&
         left.hidden_width == right.hidden_width &&
         left.sequence_capacity_tokens == right.sequence_capacity_tokens &&
         left.requires_panelwise_in_place_contract ==
             right.requires_panelwise_in_place_contract;
}

[[nodiscard]] bool same_scratch_variant(
    const PrefillOperatorScratchVariant& left,
    const PrefillOperatorScratchVariant& right) noexcept {
  return left.strategy == right.strategy &&
         same_requirement(left.aggregate, right.aggregate) &&
         left.total_required_bytes == right.total_required_bytes &&
         left.requires_legacy_route_exclusion ==
             right.requires_legacy_route_exclusion &&
         left.requires_family_completion_events ==
             right.requires_family_completion_events &&
         left.requires_route_mutual_exclusion_event ==
             right.requires_route_mutual_exclusion_event &&
         left.requires_intra_family_phase_contract ==
             right.requires_intra_family_phase_contract &&
         left.requires_prompt_token_ids_consumed_event ==
             right.requires_prompt_token_ids_consumed_event &&
         same_backing_contract(left.prompt_token_ids_backing,
                               right.prompt_token_ids_backing) &&
         same_backing_contract(left.projection_workspace_backing,
                               right.projection_workspace_backing) &&
         same_backing_contract(left.gdn_native_workspace_backing,
                               right.gdn_native_workspace_backing);
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

[[nodiscard]] bool valid_raw_requirement_exact(
    const PrefillMemoryRequirement& requirement,
    const PrefillMemoryLifetime lifetime,
    const PrefillMemoryAliasCondition alias,
    const std::uint64_t exact_bytes) noexcept {
  return valid_requirement(requirement, lifetime) &&
         requirement.alias_condition == alias &&
         requirement.required_bytes == exact_bytes &&
         requirement.element_capacity == exact_bytes &&
         requirement.element_size_bytes == 1U;
}

[[nodiscard]] bool valid_scratch_variant_exact(
    const PrefillOperatorScratchVariant& variant,
    const PrefillOperatorScratchStrategy strategy,
    const PrefillMemoryAliasCondition alias,
    const std::uint64_t exact_bytes,
    const bool requires_legacy_route_exclusion,
    const bool requires_family_completion_events,
    const bool requires_route_mutual_exclusion_event,
    const bool requires_intra_family_phase_contract,
    const bool requires_prompt_token_ids_consumed_event,
    const VariantBackingRequirements& expected_backings) noexcept {
  std::uint64_t prompt_token_capacity = 0U;
  std::uint64_t projection_capacity = 0U;
  std::uint64_t native_capacity = 0U;
  if (!checked_mul(expected_backings.prompt_token_ids.bytes_per_instance,
                   expected_backings.prompt_token_ids.minimum_instance_count,
                   prompt_token_capacity) ||
      !checked_mul(expected_backings.projection.bytes_per_instance,
                   expected_backings.projection.minimum_instance_count,
                   projection_capacity) ||
      !checked_mul(expected_backings.gdn_native.bytes_per_instance,
                   expected_backings.gdn_native.minimum_instance_count,
                   native_capacity) ||
      prompt_token_capacity > exact_bytes ||
      projection_capacity > exact_bytes || native_capacity > exact_bytes) {
    return false;
  }
  return variant.strategy == strategy &&
         valid_raw_requirement_exact(
             variant.aggregate, PrefillMemoryLifetime::kOperatorPanel, alias,
             exact_bytes) &&
         variant.total_required_bytes == exact_bytes &&
         variant.requires_legacy_route_exclusion ==
             requires_legacy_route_exclusion &&
         variant.requires_family_completion_events ==
             requires_family_completion_events &&
         variant.requires_route_mutual_exclusion_event ==
             requires_route_mutual_exclusion_event &&
         variant.requires_intra_family_phase_contract ==
             requires_intra_family_phase_contract &&
         variant.requires_prompt_token_ids_consumed_event ==
             requires_prompt_token_ids_consumed_event &&
         same_backing_contract(variant.prompt_token_ids_backing,
                               expected_backings.prompt_token_ids) &&
         same_backing_contract(variant.projection_workspace_backing,
                               expected_backings.projection) &&
         same_backing_contract(variant.gdn_native_workspace_backing,
                               expected_backings.gdn_native);
}

[[nodiscard]] bool valid_plan(
    const LayerMajorPrefillWorkspacePlan& plan) noexcept {
  if (plan.sequence_capacity_tokens == 0U ||
      plan.operator_panel_capacity_tokens !=
          kLayerMajorPrefillOperatorPanelTokens ||
      plan.request_arena_limit_bytes == 0U ||
      plan.request_arena_owner != PrefillMemoryOwner::kRequestStateArena ||
      plan.minimum_conditional.hidden_strategy !=
          PrefillHiddenStrategy::kSinglePromptWideConditional ||
      plan.minimum_conditional.scratch_strategy !=
          PrefillOperatorScratchStrategy::kC8192FamilyOverlayConditional ||
      !valid_hidden_strategy(plan.selected.hidden_strategy) ||
      !valid_scratch_strategy(plan.selected.scratch_strategy) ||
      plan.conservative.hidden_strategy !=
          PrefillHiddenStrategy::kDoublePromptWideConservative ||
      plan.conservative.scratch_strategy !=
          PrefillOperatorScratchStrategy::
              kDisjointAllFamiliesAndLegacyC512 ||
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
      !plan.selected.requires_unbound_alias_or_route_contract ||
      !plan.conservative.requires_unbound_alias_or_route_contract ||
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

  const model::ModelConfig* const config =
      model::find_known_model(model::KnownModel::kQwen36_27B);
  if (!valid_layer_major_model_contract(config)) {
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

  PrefillMemoryRequirement expected_final_hidden;
  if (!make_typed_requirement(
          config->hidden_size, kBf16Bytes,
          PrefillMemoryLifetime::kRequestPrefill,
          PrefillMemoryAliasCondition::kDisjoint,
          expected_final_hidden) ||
      !same_requirement(plan.final_hidden_handoff_bf16,
                        expected_final_hidden)) {
    return false;
  }

  const auto& hidden = plan.prompt_wide_hidden;
  std::uint64_t prompt_hidden_elements = 0U;
  std::uint64_t double_hidden_elements = 0U;
  PrefillPromptWideHiddenVariant expected_minimum_hidden;
  expected_minimum_hidden.strategy =
      PrefillHiddenStrategy::kSinglePromptWideConditional;
  expected_minimum_hidden.buffer_count = 1U;
  expected_minimum_hidden.hidden_width = config->hidden_size;
  expected_minimum_hidden.sequence_capacity_tokens =
      plan.sequence_capacity_tokens;
  expected_minimum_hidden.requires_panelwise_in_place_contract = true;
  PrefillPromptWideHiddenVariant expected_conservative_hidden;
  expected_conservative_hidden.strategy =
      PrefillHiddenStrategy::kDoublePromptWideConservative;
  expected_conservative_hidden.buffer_count = 2U;
  expected_conservative_hidden.hidden_width = config->hidden_size;
  expected_conservative_hidden.sequence_capacity_tokens =
      plan.sequence_capacity_tokens;
  if (!checked_mul(plan.sequence_capacity_tokens, config->hidden_size,
                   prompt_hidden_elements) ||
      !checked_mul(prompt_hidden_elements, 2U, double_hidden_elements) ||
      !make_typed_requirement(
          prompt_hidden_elements, kBf16Bytes,
          PrefillMemoryLifetime::kRequestPrefill,
          PrefillMemoryAliasCondition::
              kPanelwiseInputConsumedBeforeOutputOverwrite,
          expected_minimum_hidden.aggregate_bf16) ||
      !make_typed_requirement(double_hidden_elements, kBf16Bytes,
                              PrefillMemoryLifetime::kRequestPrefill,
                              PrefillMemoryAliasCondition::kDisjoint,
                              expected_conservative_hidden.aggregate_bf16) ||
      !same_hidden_variant(hidden.minimum_conditional,
                           expected_minimum_hidden) ||
      !same_hidden_variant(hidden.conservative,
                           expected_conservative_hidden)) {
    return false;
  }
  const PrefillPromptWideHiddenVariant& expected_selected_hidden =
      plan.selected.hidden_strategy ==
              PrefillHiddenStrategy::kSinglePromptWideConditional
          ? expected_minimum_hidden
          : expected_conservative_hidden;
  if (!same_hidden_variant(hidden.selected, expected_selected_hidden)) {
    return false;
  }

  const auto& scratch = plan.operator_scratch;
  if (scratch.c8192_panel_capacity_tokens !=
          kLayerMajorPrefillOperatorPanelTokens ||
      scratch.legacy_c512_panel_capacity_tokens !=
          kLayerMajorPrefillLegacyPublicTileTokens ||
      !valid_gdn_tactic(scratch.gdn_tactic) ||
      !valid_legacy_gdn_tactic(scratch.legacy_gdn_tactic) ||
      !valid_mlp_tactic(scratch.mlp_tactic) ||
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

  PrefillMemoryRequirement expected_prompt_token_ids;
  if (!make_typed_requirement(
          kLayerMajorPrefillOperatorPanelTokens, kU32Bytes,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::
              kPromptTokenIdsConsumedBeforeOperatorScratchReuse,
          expected_prompt_token_ids) ||
      !same_requirement(scratch.prompt_token_ids_u32,
                        expected_prompt_token_ids)) {
    return false;
  }

  PrefillTacticByteRequirements expected_tactic_bytes;
  if (!checked_tactic_byte_requirements(
          *config, scratch.gdn_tactic, scratch.legacy_gdn_tactic,
          scratch.mlp_tactic, expected_tactic_bytes) ||
      !valid_raw_requirement_exact(
          scratch.gdn_projection_phase,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kDisjoint,
          expected_tactic_bytes.gdn_projection_phase) ||
      !valid_raw_requirement_exact(
          scratch.gdn_recurrent_phase,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kDisjoint,
          expected_tactic_bytes.gdn_recurrent_phase) ||
      !valid_raw_requirement_exact(
          scratch.gdn_c64_native_workspace,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kDisjoint,
          expected_tactic_bytes.gdn_native_workspace) ||
      !valid_raw_requirement_exact(
          scratch.full_attention_preprocess_phase,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kDisjoint,
          expected_tactic_bytes.attention_preprocess_phase) ||
      !valid_raw_requirement_exact(
          scratch.mlp_gate_up_down_phase,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kDisjoint,
          expected_tactic_bytes.mlp_gate_up_down_phase) ||
      !valid_raw_requirement_exact(
          scratch.shared_projection_reduction_and_locks,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kDisjoint,
          expected_tactic_bytes.projection_reduction_and_locks) ||
      (expected_tactic_bytes.gdn_token_parallel_conv_output != 0U) !=
          scratch.gdn_token_parallel_c512_conv_output.has_value() ||
      (scratch.gdn_token_parallel_c512_conv_output.has_value() &&
       !valid_raw_requirement_exact(
           *scratch.gdn_token_parallel_c512_conv_output,
           PrefillMemoryLifetime::kOperatorPanel,
           PrefillMemoryAliasCondition::kDisjoint,
           expected_tactic_bytes.gdn_token_parallel_conv_output)) ||
      (expected_tactic_bytes.legacy_gdn_native_workspace != 0U) !=
          scratch.legacy_c512_gdn_native_workspace.has_value() ||
      (scratch.legacy_c512_gdn_native_workspace.has_value() &&
       !valid_raw_requirement_exact(
           *scratch.legacy_c512_gdn_native_workspace,
           PrefillMemoryLifetime::kOperatorPanel,
           PrefillMemoryAliasCondition::kDisjoint,
           expected_tactic_bytes.legacy_gdn_native_workspace))) {
    return false;
  }

  const std::array<PrefillScratchFamily,
                   kLayerMajorPrefillScratchFamilyCount>
      expected_families{
          PrefillScratchFamily::kGdnProjectionAndRecurrentCore,
          PrefillScratchFamily::kFullAttentionProjectionAndCore,
          PrefillScratchFamily::kMlpGateUpAndDown};
  const std::array<PrefillScratchProducer,
                   kLayerMajorPrefillScratchFamilyCount>
      expected_producers{
          PrefillScratchProducer::kGdnInProjQkvzAndInProjBa,
          PrefillScratchProducer::kFullAttentionQkvGateProjection,
          PrefillScratchProducer::kMlpMergedGateUpProjection};
  const std::array<PrefillScratchLastConsumer,
                   kLayerMajorPrefillScratchFamilyCount>
      expected_consumers{
          PrefillScratchLastConsumer::
              kGdnRecurrentUpdateNormAndOutputProjection,
          PrefillScratchLastConsumer::
              kFullAttentionCoreGateAndOutputProjection,
          PrefillScratchLastConsumer::kMlpSiluGateAndDownProjection};

  for (std::size_t index = 0U;
       index < scratch.c8192_family_live_sets.size(); ++index) {
    const PrefillOperatorFamilyLiveSet& live_set =
        scratch.c8192_family_live_sets[index];
    if (!valid_requirement(live_set.aggregate,
                           PrefillMemoryLifetime::kOperatorPanel) ||
        live_set.family != expected_families[index] ||
        live_set.producer != expected_producers[index] ||
        live_set.last_consumer != expected_consumers[index]) {
      return false;
    }
  }
  const std::uint64_t gdn_family_bytes = std::max(
      scratch.gdn_projection_phase.required_bytes,
      scratch.gdn_recurrent_phase.required_bytes);
  std::uint64_t disjoint_family_bytes = 0U;
  std::uint64_t legacy_bytes = 0U;
  std::uint64_t family_overlay_legacy_disjoint_bytes = 0U;
  std::uint64_t disjoint_all_bytes = 0U;
  const std::uint64_t family_overlay_bytes =
      std::max({scratch.c8192_family_live_sets[0].aggregate.required_bytes,
                scratch.c8192_family_live_sets[1].aggregate.required_bytes,
                scratch.c8192_family_live_sets[2].aggregate.required_bytes});

  std::uint64_t legacy_hidden_elements = 0U;
  std::uint64_t legacy_projection_elements = 0U;
  std::uint64_t legacy_linear_elements = 0U;
  std::uint64_t legacy_probability_elements = 0U;
  PrefillMemoryRequirement expected_legacy_hidden;
  PrefillMemoryRequirement expected_legacy_projection;
  PrefillMemoryRequirement expected_legacy_linear;
  PrefillMemoryRequirement expected_legacy_fp32;
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
      !checked_mul(plan.sequence_capacity_tokens,
                   config->num_attention_heads,
                   legacy_probability_elements) ||
      !make_typed_requirement(legacy_hidden_elements, kBf16Bytes,
                              PrefillMemoryLifetime::kOperatorPanel,
                              PrefillMemoryAliasCondition::kDisjoint,
                              expected_legacy_hidden) ||
      !make_typed_requirement(legacy_projection_elements, kBf16Bytes,
                              PrefillMemoryLifetime::kOperatorPanel,
                              PrefillMemoryAliasCondition::kDisjoint,
                              expected_legacy_projection) ||
      !make_typed_requirement(legacy_linear_elements, kBf16Bytes,
                              PrefillMemoryLifetime::kOperatorPanel,
                              PrefillMemoryAliasCondition::kDisjoint,
                              expected_legacy_linear) ||
      !make_typed_requirement(
          std::max(kLayerMajorPrefillFp32MinimumScratchElements,
                   legacy_probability_elements),
          kFp32Bytes, PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kDisjoint, expected_legacy_fp32) ||
      !same_requirement(scratch.legacy_c512_hidden_bf16,
                        expected_legacy_hidden) ||
      !same_requirement(scratch.legacy_c512_projection_bf16,
                        expected_legacy_projection) ||
      !same_requirement(scratch.legacy_c512_linear_scalar_bf16,
                        expected_legacy_linear) ||
      !same_requirement(scratch.legacy_c512_fp32_scratch,
                        expected_legacy_fp32) ||
      scratch.c8192_family_live_sets[0].aggregate.required_bytes !=
          gdn_family_bytes ||
      scratch.c8192_family_live_sets[1].aggregate.required_bytes !=
          scratch.full_attention_preprocess_phase.required_bytes ||
      scratch.c8192_family_live_sets[2].aggregate.required_bytes !=
          scratch.mlp_gate_up_down_phase.required_bytes ||
      !checked_sum(std::array<std::uint64_t,
                             kLayerMajorPrefillScratchFamilyCount>{
                       scratch.c8192_family_live_sets[0]
                           .aggregate.required_bytes,
                       scratch.c8192_family_live_sets[1]
                           .aggregate.required_bytes,
                       scratch.c8192_family_live_sets[2]
                           .aggregate.required_bytes},
                   disjoint_family_bytes) ||
      !checked_sum(std::array<std::uint64_t, 4U>{
                       scratch.legacy_c512_hidden_bf16.required_bytes,
                       scratch.legacy_c512_projection_bf16.required_bytes,
                       scratch.legacy_c512_linear_scalar_bf16.required_bytes,
                       scratch.legacy_c512_fp32_scratch.required_bytes},
                   legacy_bytes) ||
      (scratch.legacy_c512_gdn_native_workspace.has_value() &&
       !checked_add(
           legacy_bytes,
           scratch.legacy_c512_gdn_native_workspace->required_bytes,
           legacy_bytes)) ||
      !checked_add(family_overlay_bytes, legacy_bytes,
                   family_overlay_legacy_disjoint_bytes) ||
      !checked_add(disjoint_family_bytes, legacy_bytes,
                   disjoint_all_bytes)) {
    return false;
  }

  const VariantBackingRequirements family_overlay_backings =
      variant_backing_requirements(
          ScratchVariantTopology::kC8192FamilyOverlayOnly,
          scratch.legacy_gdn_tactic, expected_tactic_bytes);
  const VariantBackingRequirements family_disjoint_backings =
      variant_backing_requirements(
          ScratchVariantTopology::kC8192DisjointFamiliesOnly,
          scratch.legacy_gdn_tactic, expected_tactic_bytes);
  const VariantBackingRequirements legacy_backings =
      variant_backing_requirements(
          ScratchVariantTopology::kLegacyC512Only,
          scratch.legacy_gdn_tactic, expected_tactic_bytes);
  const VariantBackingRequirements route_overlay_backings =
      variant_backing_requirements(
          ScratchVariantTopology::kMutuallyExclusiveRouteOverlay,
          scratch.legacy_gdn_tactic, expected_tactic_bytes);
  const VariantBackingRequirements family_overlay_legacy_backings =
      variant_backing_requirements(
          ScratchVariantTopology::
              kC8192FamilyOverlayWithDisjointLegacyC512,
          scratch.legacy_gdn_tactic, expected_tactic_bytes);
  const VariantBackingRequirements all_disjoint_backings =
      variant_backing_requirements(
          ScratchVariantTopology::kDisjointAllFamiliesAndLegacyC512,
          scratch.legacy_gdn_tactic, expected_tactic_bytes);
  if (!valid_scratch_variant_exact(
          scratch.c8192_family_overlay_conditional,
          PrefillOperatorScratchStrategy::kC8192FamilyOverlayConditional,
          PrefillMemoryAliasCondition::kSequentialFamilyLiveSetOverlay,
          family_overlay_bytes, true, true, false, true, true,
          family_overlay_backings) ||
      !valid_scratch_variant_exact(
          scratch.c8192_disjoint_families,
          PrefillOperatorScratchStrategy::
              kDisjointAllFamiliesAndLegacyC512,
          PrefillMemoryAliasCondition::kDisjoint, disjoint_family_bytes, true,
          false, false, true, true, family_disjoint_backings) ||
      !valid_scratch_variant_exact(
          scratch.legacy_c512_only,
          PrefillOperatorScratchStrategy::kUnselected,
          PrefillMemoryAliasCondition::kDisjoint, legacy_bytes, false, false,
          false, false, true, legacy_backings) ||
      !valid_scratch_variant_exact(
          scratch.mutually_exclusive_overlay,
          PrefillOperatorScratchStrategy::
              kOverlayLegacyC512MutuallyExclusive,
          PrefillMemoryAliasCondition::kMutuallyExclusiveRouteOverlay,
          std::max(family_overlay_bytes, legacy_bytes), false, true, true,
          true, true, route_overlay_backings) ||
      !valid_scratch_variant_exact(
          scratch.c8192_family_overlay_with_disjoint_legacy_c512,
          PrefillOperatorScratchStrategy::
              kC8192FamilyOverlayWithDisjointLegacyC512,
          PrefillMemoryAliasCondition::kSequentialFamilyLiveSetOverlay,
          family_overlay_legacy_disjoint_bytes, false, true, false, true,
          true, family_overlay_legacy_backings) ||
      !valid_scratch_variant_exact(
          scratch.disjoint_conservative,
          PrefillOperatorScratchStrategy::
              kDisjointAllFamiliesAndLegacyC512,
          PrefillMemoryAliasCondition::kDisjoint, disjoint_all_bytes, false,
          false, false, true, true, all_disjoint_backings)) {
    return false;
  }

  const PrefillOperatorScratchVariant* expected_selected_scratch = nullptr;
  switch (plan.selected.scratch_strategy) {
    case PrefillOperatorScratchStrategy::kC8192FamilyOverlayConditional:
      expected_selected_scratch =
          &scratch.c8192_family_overlay_conditional;
      break;
    case PrefillOperatorScratchStrategy::
        kOverlayLegacyC512MutuallyExclusive:
      expected_selected_scratch = &scratch.mutually_exclusive_overlay;
      break;
    case PrefillOperatorScratchStrategy::
        kC8192FamilyOverlayWithDisjointLegacyC512:
      expected_selected_scratch =
          &scratch.c8192_family_overlay_with_disjoint_legacy_c512;
      break;
    case PrefillOperatorScratchStrategy::
        kDisjointAllFamiliesAndLegacyC512:
      expected_selected_scratch = &scratch.disjoint_conservative;
      break;
    case PrefillOperatorScratchStrategy::kUnselected:
      return false;
  }
  if (expected_selected_scratch == nullptr ||
      !same_scratch_variant(scratch.selected,
                            *expected_selected_scratch)) {
    return false;
  }

  std::uint64_t common_bytes = 0U;
  std::uint64_t expected_minimum_bytes = 0U;
  std::uint64_t expected_selected_bytes = 0U;
  std::uint64_t expected_conservative_bytes = 0U;
  const bool expected_selected_unbound =
      hidden.selected.requires_panelwise_in_place_contract ||
      scratch.selected.requires_legacy_route_exclusion ||
      scratch.selected.requires_family_completion_events ||
      scratch.selected.requires_route_mutual_exclusion_event ||
      scratch.selected.requires_intra_family_phase_contract ||
      scratch.selected.requires_prompt_token_ids_consumed_event;
  if (!checked_sum(std::array<std::uint64_t, 3U>{
                       persistent.total_required_bytes,
                       position.total_required_bytes,
                       plan.final_hidden_handoff_bf16.required_bytes},
                   common_bytes) ||
      !checked_sum(std::array<std::uint64_t, 3U>{
                       common_bytes,
                       hidden.minimum_conditional.aggregate_bf16.required_bytes,
                       family_overlay_bytes},
                   expected_minimum_bytes) ||
      !checked_sum(std::array<std::uint64_t, 3U>{
                       common_bytes,
                       hidden.selected.aggregate_bf16.required_bytes,
                       scratch.selected.total_required_bytes},
                   expected_selected_bytes) ||
      !checked_sum(std::array<std::uint64_t, 3U>{
                       common_bytes,
                       hidden.conservative.aggregate_bf16.required_bytes,
                       disjoint_all_bytes},
                   expected_conservative_bytes) ||
      plan.minimum_conditional.required_bytes != expected_minimum_bytes ||
      plan.selected.required_bytes != expected_selected_bytes ||
      plan.conservative.required_bytes != expected_conservative_bytes ||
      plan.selected.requires_unbound_alias_or_route_contract !=
          expected_selected_unbound) {
    return false;
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
      !valid_scratch_strategy(options.scratch_strategy) ||
      !valid_gdn_tactic(options.gdn_tactic) ||
      !valid_legacy_gdn_tactic(options.legacy_gdn_tactic) ||
      !valid_mlp_tactic(options.mlp_tactic)) {
    return failure(PrefillWorkspacePlanError::kInvalidArgument);
  }

  const model::ModelConfig* const config =
      model::find_known_model(model::KnownModel::kQwen36_27B);
  if (!valid_layer_major_model_contract(config)) {
    return failure(PrefillWorkspacePlanError::kModelContractMismatch);
  }

  const std::uint64_t sequence = options.sequence_capacity_tokens;
  const std::uint64_t linear_layers =
      config->num_linear_attention_layers();
  const std::uint64_t full_layers = config->num_full_attention_layers();
  const std::uint64_t qkv_width = config->linear_qkv_projection_dim();

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

  if (!make_typed_requirement(
          config->hidden_size, kBf16Bytes,
          PrefillMemoryLifetime::kRequestPrefill,
          PrefillMemoryAliasCondition::kDisjoint,
          plan.final_hidden_handoff_bf16)) {
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
  scratch.gdn_tactic = options.gdn_tactic;
  scratch.legacy_gdn_tactic = options.legacy_gdn_tactic;
  scratch.mlp_tactic = options.mlp_tactic;

  PrefillTacticByteRequirements tactic_bytes;
  if (!checked_tactic_byte_requirements(
          *config, options.gdn_tactic, options.legacy_gdn_tactic,
          options.mlp_tactic, tactic_bytes)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }

  if (!make_typed_requirement(
          kLayerMajorPrefillOperatorPanelTokens, kU32Bytes,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::
              kPromptTokenIdsConsumedBeforeOperatorScratchReuse,
          scratch.prompt_token_ids_u32) ||
      !make_raw_requirement(tactic_bytes.gdn_projection_phase,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            scratch.gdn_projection_phase) ||
      !make_raw_requirement(tactic_bytes.gdn_recurrent_phase,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            scratch.gdn_recurrent_phase) ||
      !make_raw_requirement(tactic_bytes.gdn_native_workspace,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            scratch.gdn_c64_native_workspace) ||
      !make_raw_requirement(tactic_bytes.attention_preprocess_phase,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            scratch.full_attention_preprocess_phase) ||
      !make_raw_requirement(tactic_bytes.mlp_gate_up_down_phase,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            scratch.mlp_gate_up_down_phase) ||
      !make_raw_requirement(tactic_bytes.projection_reduction_and_locks,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            scratch.shared_projection_reduction_and_locks)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  if (options.gdn_tactic ==
      PrefillGdnPhysicalTactic::kC64NativeTokenParallelConv) {
    PrefillMemoryRequirement conv_output;
    if (!make_raw_requirement(tactic_bytes.gdn_token_parallel_conv_output,
                              PrefillMemoryLifetime::kOperatorPanel,
                              PrefillMemoryAliasCondition::kDisjoint,
                              conv_output)) {
      return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
    }
    scratch.gdn_token_parallel_c512_conv_output.emplace(conv_output);
  }

  const std::array<std::uint64_t, kLayerMajorPrefillScratchFamilyCount>
      family_bytes{
          std::max(tactic_bytes.gdn_projection_phase,
                   tactic_bytes.gdn_recurrent_phase),
          tactic_bytes.attention_preprocess_phase,
          tactic_bytes.mlp_gate_up_down_phase};
  const std::array<PrefillScratchFamily,
                   kLayerMajorPrefillScratchFamilyCount>
      families{PrefillScratchFamily::kGdnProjectionAndRecurrentCore,
               PrefillScratchFamily::kFullAttentionProjectionAndCore,
               PrefillScratchFamily::kMlpGateUpAndDown};
  const std::array<PrefillScratchProducer,
                   kLayerMajorPrefillScratchFamilyCount>
      producers{PrefillScratchProducer::kGdnInProjQkvzAndInProjBa,
                PrefillScratchProducer::kFullAttentionQkvGateProjection,
                PrefillScratchProducer::kMlpMergedGateUpProjection};
  const std::array<PrefillScratchLastConsumer,
                   kLayerMajorPrefillScratchFamilyCount>
      consumers{
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
  family_overlay.requires_intra_family_phase_contract = true;
  family_overlay.requires_prompt_token_ids_consumed_event = true;
  if (!make_raw_requirement(
          family_overlay_bytes, PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kSequentialFamilyLiveSetOverlay,
          family_overlay.aggregate)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  family_overlay.total_required_bytes =
      family_overlay.aggregate.required_bytes;
  {
    const VariantBackingRequirements backings =
        variant_backing_requirements(
            ScratchVariantTopology::kC8192FamilyOverlayOnly,
            options.legacy_gdn_tactic, tactic_bytes);
    family_overlay.prompt_token_ids_backing = backings.prompt_token_ids;
    family_overlay.projection_workspace_backing = backings.projection;
    family_overlay.gdn_native_workspace_backing = backings.gdn_native;
  }

  auto& family_disjoint = scratch.c8192_disjoint_families;
  family_disjoint.strategy = PrefillOperatorScratchStrategy::
      kDisjointAllFamiliesAndLegacyC512;
  family_disjoint.requires_legacy_route_exclusion = true;
  family_disjoint.requires_intra_family_phase_contract = true;
  family_disjoint.requires_prompt_token_ids_consumed_event = true;
  if (!make_raw_requirement(family_disjoint_bytes,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            family_disjoint.aggregate)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  family_disjoint.total_required_bytes =
      family_disjoint.aggregate.required_bytes;
  {
    const VariantBackingRequirements backings =
        variant_backing_requirements(
            ScratchVariantTopology::kC8192DisjointFamiliesOnly,
            options.legacy_gdn_tactic, tactic_bytes);
    family_disjoint.prompt_token_ids_backing = backings.prompt_token_ids;
    family_disjoint.projection_workspace_backing = backings.projection;
    family_disjoint.gdn_native_workspace_backing = backings.gdn_native;
  }

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
  if (options.legacy_gdn_tactic ==
      PrefillLegacyGdnPhysicalTactic::kC64Native) {
    PrefillMemoryRequirement native_workspace;
    if (!make_raw_requirement(
            tactic_bytes.legacy_gdn_native_workspace,
            PrefillMemoryLifetime::kOperatorPanel,
            PrefillMemoryAliasCondition::kDisjoint,
            native_workspace) ||
        !checked_add(legacy_bytes, native_workspace.required_bytes,
                     legacy_bytes)) {
      return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
    }
    scratch.legacy_c512_gdn_native_workspace.emplace(native_workspace);
  }
  auto& legacy = scratch.legacy_c512_only;
  legacy.requires_prompt_token_ids_consumed_event = true;
  if (!make_raw_requirement(legacy_bytes,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            legacy.aggregate)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  legacy.total_required_bytes = legacy.aggregate.required_bytes;
  {
    const VariantBackingRequirements backings =
        variant_backing_requirements(ScratchVariantTopology::kLegacyC512Only,
                                     options.legacy_gdn_tactic,
                                     tactic_bytes);
    legacy.prompt_token_ids_backing = backings.prompt_token_ids;
    legacy.projection_workspace_backing = backings.projection;
    legacy.gdn_native_workspace_backing = backings.gdn_native;
  }

  auto& route_overlay = scratch.mutually_exclusive_overlay;
  route_overlay.strategy = PrefillOperatorScratchStrategy::
      kOverlayLegacyC512MutuallyExclusive;
  route_overlay.requires_family_completion_events = true;
  route_overlay.requires_route_mutual_exclusion_event = true;
  route_overlay.requires_intra_family_phase_contract = true;
  route_overlay.requires_prompt_token_ids_consumed_event = true;
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
  {
    const VariantBackingRequirements backings =
        variant_backing_requirements(
            ScratchVariantTopology::kMutuallyExclusiveRouteOverlay,
            options.legacy_gdn_tactic, tactic_bytes);
    route_overlay.prompt_token_ids_backing = backings.prompt_token_ids;
    route_overlay.projection_workspace_backing = backings.projection;
    route_overlay.gdn_native_workspace_backing = backings.gdn_native;
  }

  std::uint64_t family_overlay_legacy_disjoint_bytes = 0U;
  std::uint64_t all_disjoint_bytes = 0U;
  if (!checked_add(family_overlay.total_required_bytes,
                   legacy.total_required_bytes,
                   family_overlay_legacy_disjoint_bytes) ||
      !checked_add(family_disjoint.total_required_bytes,
                   legacy.total_required_bytes, all_disjoint_bytes)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  auto& family_overlay_legacy_disjoint =
      scratch.c8192_family_overlay_with_disjoint_legacy_c512;
  family_overlay_legacy_disjoint.strategy = PrefillOperatorScratchStrategy::
      kC8192FamilyOverlayWithDisjointLegacyC512;
  family_overlay_legacy_disjoint.requires_family_completion_events = true;
  family_overlay_legacy_disjoint.requires_intra_family_phase_contract = true;
  family_overlay_legacy_disjoint.requires_prompt_token_ids_consumed_event =
      true;
  if (!make_raw_requirement(
          family_overlay_legacy_disjoint_bytes,
          PrefillMemoryLifetime::kOperatorPanel,
          PrefillMemoryAliasCondition::kSequentialFamilyLiveSetOverlay,
          family_overlay_legacy_disjoint.aggregate)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  family_overlay_legacy_disjoint.total_required_bytes =
      family_overlay_legacy_disjoint.aggregate.required_bytes;
  {
    const VariantBackingRequirements backings =
        variant_backing_requirements(
            ScratchVariantTopology::
                kC8192FamilyOverlayWithDisjointLegacyC512,
            options.legacy_gdn_tactic, tactic_bytes);
    family_overlay_legacy_disjoint.prompt_token_ids_backing =
        backings.prompt_token_ids;
    family_overlay_legacy_disjoint.projection_workspace_backing =
        backings.projection;
    family_overlay_legacy_disjoint.gdn_native_workspace_backing =
        backings.gdn_native;
  }

  auto& all_disjoint = scratch.disjoint_conservative;
  all_disjoint.strategy = PrefillOperatorScratchStrategy::
      kDisjointAllFamiliesAndLegacyC512;
  all_disjoint.requires_intra_family_phase_contract = true;
  all_disjoint.requires_prompt_token_ids_consumed_event = true;
  if (!make_raw_requirement(all_disjoint_bytes,
                            PrefillMemoryLifetime::kOperatorPanel,
                            PrefillMemoryAliasCondition::kDisjoint,
                            all_disjoint.aggregate)) {
    return failure(PrefillWorkspacePlanError::kArithmeticOverflow);
  }
  all_disjoint.total_required_bytes =
      all_disjoint.aggregate.required_bytes;
  {
    const VariantBackingRequirements backings =
        variant_backing_requirements(
            ScratchVariantTopology::kDisjointAllFamiliesAndLegacyC512,
            options.legacy_gdn_tactic, tactic_bytes);
    all_disjoint.prompt_token_ids_backing = backings.prompt_token_ids;
    all_disjoint.projection_workspace_backing = backings.projection;
    all_disjoint.gdn_native_workspace_backing = backings.gdn_native;
  }

  switch (options.scratch_strategy) {
    case PrefillOperatorScratchStrategy::kC8192FamilyOverlayConditional:
      scratch.selected = family_overlay;
      break;
    case PrefillOperatorScratchStrategy::
        kOverlayLegacyC512MutuallyExclusive:
      scratch.selected = route_overlay;
      break;
    case PrefillOperatorScratchStrategy::
        kC8192FamilyOverlayWithDisjointLegacyC512:
      scratch.selected = family_overlay_legacy_disjoint;
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
  if (!checked_sum(std::array<std::uint64_t, 3U>{
                       persistent.total_required_bytes,
                       position.total_required_bytes,
                       plan.final_hidden_handoff_bf16.required_bytes},
                   common_bytes) ||
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
      scratch.selected.requires_route_mutual_exclusion_event ||
      scratch.selected.requires_intra_family_phase_contract ||
      scratch.selected.requires_prompt_token_ids_consumed_event;

  plan.conservative.hidden_strategy =
      PrefillHiddenStrategy::kDoublePromptWideConservative;
  plan.conservative.scratch_strategy = PrefillOperatorScratchStrategy::
      kDisjointAllFamiliesAndLegacyC512;
  plan.conservative.capacity =
      capacity(plan.conservative.required_bytes,
               plan.request_arena_limit_bytes);
  plan.conservative.requires_unbound_alias_or_route_contract = true;

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
