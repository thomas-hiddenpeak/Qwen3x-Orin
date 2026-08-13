#include "q3x/runtime/sm87_target_aot_request_state.h"

#include "sm87_target_aot_request_state_access_internal.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

using sm87_target_aot_request_detail::BoundEvent;
using sm87_target_aot_request_detail::BoundSpan;
using sm87_target_aot_request_detail::OwnerSnapshot;

constexpr std::uint64_t kBf16Bytes = 2U;
constexpr std::uint64_t kU32Bytes = 4U;
constexpr std::uint64_t kConvBytesPerLayer = 61'440U;
constexpr std::uint64_t kGdnStateBytesPerLayer = 1'572'864U;
constexpr std::uint64_t kKvBytesPerLayer = 81'922'048U;
constexpr std::uint64_t kPromptHiddenBytes = 409'600'000U;
constexpr std::uint64_t kGdnRawBytes = 1'310'720'000U;
constexpr std::uint64_t kGdnAbBytes = 7'680'000U;
constexpr std::uint64_t kAttentionQGateBytes = 983'040'000U;
constexpr std::uint64_t kAttentionSpanBytes = 491'520'000U;
constexpr std::uint64_t kMlpActivatedBytes = 1'392'640'000U;

constexpr std::uint64_t kFamilyBankB = 983'040'000U;
constexpr std::uint64_t kGdnAbOffset = 1'310'720'000U;
constexpr std::uint64_t kGdnTailOffset = 1'318'400'000U;
constexpr std::uint64_t kAttentionProcessedGateOffset = 1'474'560'000U;
constexpr std::uint64_t kAttentionGatedOffset = 491'520'000U;
constexpr std::uint64_t kMlpNormalizedOffset = 1'392'640'000U;

std::atomic<std::uint64_t> g_next_owner_identity{1U};

[[nodiscard]] std::uint64_t next_identity() noexcept {
  const std::uint64_t result =
      g_next_owner_identity.fetch_add(1U, std::memory_order_relaxed);
  return result == 0U ?
             g_next_owner_identity.fetch_add(1U, std::memory_order_relaxed) :
             result;
}

[[nodiscard]] constexpr Sm87TargetAotRequestRegion region(
    const std::uint64_t offset, const std::uint64_t bytes,
    const Sm87TargetAotRequestScalarType scalar,
    const Sm87TargetAotRequestLifetime lifetime) noexcept {
  return {offset, bytes, scalar, lifetime};
}

[[nodiscard]] constexpr Sm87TargetAotRequestMatrixRegion matrix(
    const std::uint64_t offset, const std::uint32_t rows,
    const std::uint32_t columns, const std::uint64_t stride,
    const Sm87TargetAotRequestScalarType scalar,
    const Sm87TargetAotRequestLifetime lifetime) noexcept {
  const std::uint64_t scalar_bytes =
      scalar == Sm87TargetAotRequestScalarType::kU32 ? kU32Bytes :
                                                       kBf16Bytes;
  return {region(offset,
                 static_cast<std::uint64_t>(rows) * stride * scalar_bytes,
                 scalar, lifetime),
          rows, columns, stride};
}

[[nodiscard]] constexpr bool same_region(
    const Sm87TargetAotRequestRegion& left,
    const Sm87TargetAotRequestRegion& right) noexcept {
  return left.arena_offset == right.arena_offset &&
         left.byte_size == right.byte_size &&
         left.scalar_type == right.scalar_type &&
         left.lifetime == right.lifetime;
}

[[nodiscard]] constexpr bool same_matrix(
    const Sm87TargetAotRequestMatrixRegion& left,
    const Sm87TargetAotRequestMatrixRegion& right) noexcept {
  return same_region(left.storage, right.storage) && left.rows == right.rows &&
         left.columns == right.columns &&
         left.row_stride_elements == right.row_stride_elements;
}

[[nodiscard]] constexpr bool same_alias(
    const Sm87TargetAotRequestAliasContract& left,
    const Sm87TargetAotRequestAliasContract& right) noexcept {
  return left.producer == right.producer &&
         left.successor == right.successor &&
         left.transition == right.transition &&
         left.family_cardinality == right.family_cardinality &&
         left.same_allocation_required == right.same_allocation_required &&
         left.same_begin_required == right.same_begin_required &&
         left.same_extent_required == right.same_extent_required &&
         left.successor_forbidden_before_transition ==
             right.successor_forbidden_before_transition;
}

[[nodiscard]] constexpr Sm87TargetAotRequestAliasContract alias(
    const Sm87TargetAotRequestAliasRole producer,
    const Sm87TargetAotRequestAliasRole successor,
    const Sm87TargetAotRequestAliasTransition transition,
    const std::size_t cardinality, const bool same_begin,
    const bool same_extent) noexcept {
  return {producer, successor, transition, cardinality, true, same_begin,
          same_extent, true};
}

[[nodiscard]] bool region_inside_arena(
    const Sm87TargetAotRequestRegion& value,
    const std::uint64_t arena_bytes) noexcept {
  return value.byte_size != 0U &&
         value.arena_offset <= arena_bytes &&
         value.byte_size <= arena_bytes - value.arena_offset;
}

[[nodiscard]] bool valid_matrix_shape(
    const Sm87TargetAotRequestMatrixRegion& value) noexcept {
  const std::uint64_t scalar_bytes =
      value.storage.scalar_type == Sm87TargetAotRequestScalarType::kU32
          ? kU32Bytes
          : value.storage.scalar_type ==
                    Sm87TargetAotRequestScalarType::kBf16
                ? kBf16Bytes
                : 0U;
  return scalar_bytes != 0U && value.rows != 0U && value.columns != 0U &&
         value.row_stride_elements >= value.columns &&
         value.storage.byte_size ==
             static_cast<std::uint64_t>(value.rows) *
                 value.row_stride_elements * scalar_bytes;
}

struct PhysicalRangeIdentity {
  std::uint64_t offset = 0U;
  std::uint64_t bytes = 0U;
  std::uint64_t identity = 0U;
};

}  // namespace

Sm87TargetAotP40RequestMemoryPlan
build_sm87_target_aot_p40_request_memory_plan() noexcept {
  Sm87TargetAotP40RequestMemoryPlan plan;
  plan.prompt_tokens = kSm87TargetAotP40PromptTokens;
  plan.request_capacity_tokens = kSm87TargetAotP40RequestCapacityTokens;
  plan.arena_bytes = kSm87TargetAotP40RequestArenaBytes;

  plan.persistent_arena =
      region(0U, kSm87TargetAotP40PersistentBytes,
             Sm87TargetAotRequestScalarType::kBf16,
             Sm87TargetAotRequestLifetime::kRequestOwner);
  std::uint64_t cursor = 0U;
  for (auto& current : plan.persistent.conv_history) {
    current = region(
        cursor, kConvBytesPerLayer, Sm87TargetAotRequestScalarType::kBf16,
        Sm87TargetAotRequestLifetime::
            kRequestTransactionUnpublishedUntilCommit);
    cursor += kConvBytesPerLayer;
  }
  for (auto& current : plan.persistent.recurrent_state) {
    current = region(
        cursor, kGdnStateBytesPerLayer,
        Sm87TargetAotRequestScalarType::kBf16,
        Sm87TargetAotRequestLifetime::
            kRequestTransactionUnpublishedUntilCommit);
    cursor += kGdnStateBytesPerLayer;
  }
  for (std::size_t slot = 0U; slot < kSm87TargetAotP40FullLayerCount;
       ++slot) {
    plan.persistent.key[slot] = matrix(
        cursor, kSm87TargetAotP40RequestCapacityTokens,
        kSm87TargetAotP40KvWidth, kSm87TargetAotP40KvWidth,
        Sm87TargetAotRequestScalarType::kBf16,
        Sm87TargetAotRequestLifetime::
            kRequestTransactionUnpublishedUntilCommit);
    cursor += kKvBytesPerLayer;
    plan.persistent.value[slot] = matrix(
        cursor, kSm87TargetAotP40RequestCapacityTokens,
        kSm87TargetAotP40KvWidth, kSm87TargetAotP40KvWidth,
        Sm87TargetAotRequestScalarType::kBf16,
        Sm87TargetAotRequestLifetime::
            kRequestTransactionUnpublishedUntilCommit);
    cursor += kKvBytesPerLayer;
  }

  const std::uint64_t residual_offset = kSm87TargetAotP40PersistentBytes;
  plan.residual = matrix(
      residual_offset, kSm87TargetAotP40RequestCapacityTokens,
      kSm87TargetAotP40Hidden, kSm87TargetAotP40Hidden,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kLayerResidual);
  const std::uint64_t family_offset =
      residual_offset + kSm87TargetAotP40ResidualBytes;
  plan.family_arena =
      region(family_offset, kSm87TargetAotP40FamilyArenaBytes,
             Sm87TargetAotRequestScalarType::kBf16,
             Sm87TargetAotRequestLifetime::kRequestOwner);

  plan.token_ids = matrix(
      family_offset, kSm87TargetAotP40PromptTokens, 1U, 1U,
      Sm87TargetAotRequestScalarType::kU32,
      Sm87TargetAotRequestLifetime::kEmbeddingOnly);

  plan.gdn.raw_qkvz = matrix(
      family_offset, kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40GdnRawWidth, kSm87TargetAotP40GdnRawWidth,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kGdnProducerUntilCoreCompletion);
  plan.gdn.bf16_ab = matrix(
      family_offset + kGdnAbOffset, kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40GdnAbWidth, kSm87TargetAotP40GdnAbWidth,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kGdnProducerUntilCoreCompletion);
  plan.gdn.normalized_input = matrix(
      family_offset + kGdnTailOffset, kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40Hidden, kSm87TargetAotP40Hidden,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kProjectionInput);
  plan.gdn.output = matrix(
      family_offset + kGdnTailOffset, kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40AttentionWidth, kSm87TargetAotP40AttentionWidth,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kGdnOutputUntilOProjectionCompletion);
  plan.gdn.o_branch = matrix(
      family_offset, kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40Hidden, kSm87TargetAotP40Hidden,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kProjectionBranchUntilResidualCompletion);

  plan.attention.raw_q_gate = matrix(
      family_offset, kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40AttentionQGateWidth,
      kSm87TargetAotP40AttentionQGateWidth,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kAttentionPreparation);
  plan.attention.normalized_input = matrix(
      family_offset + kFamilyBankB, kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40Hidden, kSm87TargetAotP40Hidden,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kProjectionInput);
  plan.attention.processed_q = matrix(
      family_offset + kFamilyBankB, kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40AttentionWidth, kSm87TargetAotP40AttentionWidth,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kAttentionCore);
  plan.attention.processed_gate = matrix(
      family_offset + kAttentionProcessedGateOffset,
      kSm87TargetAotP40PromptTokens, kSm87TargetAotP40AttentionWidth,
      kSm87TargetAotP40AttentionWidth,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kAttentionCore);
  plan.attention.pre_gate_output = matrix(
      family_offset, kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40AttentionWidth, kSm87TargetAotP40AttentionWidth,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kAttentionCore);
  plan.attention.gated_output = matrix(
      family_offset + kAttentionGatedOffset,
      kSm87TargetAotP40PromptTokens, kSm87TargetAotP40AttentionWidth,
      kSm87TargetAotP40AttentionWidth,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::
          kAttentionOutputUntilOProjectionCompletion);
  plan.attention.o_branch = matrix(
      family_offset + kFamilyBankB, kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40Hidden, kSm87TargetAotP40Hidden,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kProjectionBranchUntilResidualCompletion);

  plan.mlp.activated = matrix(
      family_offset, kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40Intermediate, kSm87TargetAotP40Intermediate,
      Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kMlpActivatedUntilDownCompletion);
  plan.mlp.normalized_input = matrix(
      family_offset + kMlpNormalizedOffset,
      kSm87TargetAotP40PromptTokens, kSm87TargetAotP40Hidden,
      kSm87TargetAotP40Hidden, Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kProjectionInput);
  plan.mlp.down_branch = matrix(
      family_offset + kMlpNormalizedOffset,
      kSm87TargetAotP40PromptTokens, kSm87TargetAotP40Hidden,
      kSm87TargetAotP40Hidden, Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kProjectionBranchUntilResidualCompletion);

  const std::uint64_t final_offset =
      family_offset + kSm87TargetAotP40FamilyArenaBytes;
  plan.final_hidden = matrix(
      final_offset, 1U, kSm87TargetAotP40Hidden,
      kSm87TargetAotP40Hidden, Sm87TargetAotRequestScalarType::kBf16,
      Sm87TargetAotRequestLifetime::kFinalHiddenUntilRequestCommit);

  std::size_t gdn_slot = 0U;
  std::size_t full_slot = 0U;
  for (std::size_t layer = 0U; layer < plan.layers.size(); ++layer) {
    auto& binding = plan.layers[layer];
    if ((layer + 1U) % 4U == 0U) {
      binding.kind = Sm87TargetAotRequestLayerKind::kFullAttention;
      binding.family_ordinal = full_slot;
      binding.key = plan.persistent.key[full_slot];
      binding.value = plan.persistent.value[full_slot];
      ++full_slot;
    } else {
      binding.kind = Sm87TargetAotRequestLayerKind::kGdn;
      binding.family_ordinal = gdn_slot;
      binding.conv_history = plan.persistent.conv_history[gdn_slot];
      binding.recurrent_state = plan.persistent.recurrent_state[gdn_slot];
      ++gdn_slot;
    }
  }

  plan.alias_contracts = {{
      alias(Sm87TargetAotRequestAliasRole::kTokenIds,
            Sm87TargetAotRequestAliasRole::kFamilyArena,
            Sm87TargetAotRequestAliasTransition::kEmbeddingComplete, 1U,
            true, false),
      alias(Sm87TargetAotRequestAliasRole::kGdnNormalizedInput,
            Sm87TargetAotRequestAliasRole::kGdnOutput,
            Sm87TargetAotRequestAliasTransition::kInputProjectionsComplete,
            kSm87TargetAotP40GdnLayerCount, true, false),
      alias(Sm87TargetAotRequestAliasRole::kGdnRawQkvZ,
            Sm87TargetAotRequestAliasRole::kGdnOBranch,
            Sm87TargetAotRequestAliasTransition::kGdnCoreComplete,
            kSm87TargetAotP40GdnLayerCount, true, false),
      alias(Sm87TargetAotRequestAliasRole::kAttentionNormalizedInput,
            Sm87TargetAotRequestAliasRole::kAttentionProcessedQGate,
            Sm87TargetAotRequestAliasTransition::
                kAttentionPreparationComplete,
            kSm87TargetAotP40FullLayerCount, true, false),
      alias(Sm87TargetAotRequestAliasRole::kAttentionRawQGate,
            Sm87TargetAotRequestAliasRole::kAttentionPreGateAndGated,
            Sm87TargetAotRequestAliasTransition::kAttentionCoreComplete,
            kSm87TargetAotP40FullLayerCount, true, true),
      alias(Sm87TargetAotRequestAliasRole::kAttentionProcessedQGate,
            Sm87TargetAotRequestAliasRole::kAttentionOBranch,
            Sm87TargetAotRequestAliasTransition::
                kAttentionOProjectionComplete,
            kSm87TargetAotP40FullLayerCount, true, false),
      alias(Sm87TargetAotRequestAliasRole::kMlpNormalizedInput,
            Sm87TargetAotRequestAliasRole::kMlpDownBranch,
            Sm87TargetAotRequestAliasTransition::kGateUpComplete,
            kSm87TargetAotP40LayerCount, true, true),
      alias(Sm87TargetAotRequestAliasRole::kRawKeyTransactionSpan,
            Sm87TargetAotRequestAliasRole::kProcessedKeyTransactionSpan,
            Sm87TargetAotRequestAliasTransition::kExactInPlaceKvPreprocess,
            kSm87TargetAotP40FullLayerCount, true, true),
      alias(Sm87TargetAotRequestAliasRole::kRawValueTransactionSpan,
            Sm87TargetAotRequestAliasRole::kProcessedValueTransactionSpan,
            Sm87TargetAotRequestAliasTransition::kExactInPlaceKvPreprocess,
            kSm87TargetAotP40FullLayerCount, true, true),
      alias(Sm87TargetAotRequestAliasRole::kLayerResidualInput,
            Sm87TargetAotRequestAliasRole::kLayerResidualPublication,
            Sm87TargetAotRequestAliasTransition::kGateUpComplete,
            kSm87TargetAotP40LayerCount, true, true),
  }};

  plan.owned_event_count = kSm87TargetAotP40OwnedEventCount;
  plan.single_allocation = true;
  plan.cold_request_only = true;
  plan.one_request_wide_commit = true;
  plan.cancellation_discards_unpublished = true;
  plan.kv_preprocess_in_place_alias_required = true;
  plan.exposes_raw_arena = false;
  plan.permits_legacy_fallback = false;
  return plan;
}

bool validate_sm87_target_aot_p40_request_memory_plan(
    const Sm87TargetAotP40RequestMemoryPlan& plan) noexcept {
  const auto expected = build_sm87_target_aot_p40_request_memory_plan();
  if (plan.prompt_tokens != expected.prompt_tokens ||
      plan.request_capacity_tokens != expected.request_capacity_tokens ||
      plan.arena_bytes != expected.arena_bytes ||
      !same_region(plan.persistent_arena, expected.persistent_arena) ||
      !same_matrix(plan.residual, expected.residual) ||
      !same_region(plan.family_arena, expected.family_arena) ||
      !same_matrix(plan.token_ids, expected.token_ids) ||
      !same_matrix(plan.gdn.raw_qkvz, expected.gdn.raw_qkvz) ||
      !same_matrix(plan.gdn.bf16_ab, expected.gdn.bf16_ab) ||
      !same_matrix(plan.gdn.normalized_input,
                   expected.gdn.normalized_input) ||
      !same_matrix(plan.gdn.output, expected.gdn.output) ||
      !same_matrix(plan.gdn.o_branch, expected.gdn.o_branch) ||
      !same_matrix(plan.attention.raw_q_gate,
                   expected.attention.raw_q_gate) ||
      !same_matrix(plan.attention.normalized_input,
                   expected.attention.normalized_input) ||
      !same_matrix(plan.attention.processed_q,
                   expected.attention.processed_q) ||
      !same_matrix(plan.attention.processed_gate,
                   expected.attention.processed_gate) ||
      !same_matrix(plan.attention.pre_gate_output,
                   expected.attention.pre_gate_output) ||
      !same_matrix(plan.attention.gated_output,
                   expected.attention.gated_output) ||
      !same_matrix(plan.attention.o_branch,
                   expected.attention.o_branch) ||
      !same_matrix(plan.mlp.normalized_input,
                   expected.mlp.normalized_input) ||
      !same_matrix(plan.mlp.activated, expected.mlp.activated) ||
      !same_matrix(plan.mlp.down_branch, expected.mlp.down_branch) ||
      !same_matrix(plan.final_hidden, expected.final_hidden) ||
      plan.owned_event_count != expected.owned_event_count ||
      plan.single_allocation != expected.single_allocation ||
      plan.cold_request_only != expected.cold_request_only ||
      plan.one_request_wide_commit != expected.one_request_wide_commit ||
      plan.cancellation_discards_unpublished !=
          expected.cancellation_discards_unpublished ||
      plan.kv_preprocess_in_place_alias_required !=
          expected.kv_preprocess_in_place_alias_required ||
      plan.exposes_raw_arena || plan.permits_legacy_fallback ||
      plan.engine_rope.maximum_position_embeddings !=
          expected.engine_rope.maximum_position_embeddings ||
      plan.engine_rope.rotary_pairs != expected.engine_rope.rotary_pairs ||
      plan.engine_rope.scalar_bytes != expected.engine_rope.scalar_bytes ||
      plan.engine_rope.complete_table_bytes !=
          expected.engine_rope.complete_table_bytes ||
      plan.engine_rope.lifetime != expected.engine_rope.lifetime ||
      plan.engine_rope.included_in_request_arena ||
      !region_inside_arena(plan.persistent_arena, plan.arena_bytes) ||
      !region_inside_arena(plan.residual.storage, plan.arena_bytes) ||
      !region_inside_arena(plan.family_arena, plan.arena_bytes) ||
      !region_inside_arena(plan.final_hidden.storage, plan.arena_bytes) ||
      !valid_matrix_shape(plan.residual) ||
      !valid_matrix_shape(plan.final_hidden)) {
    return false;
  }
  for (std::size_t index = 0U; index < plan.persistent.conv_history.size();
       ++index) {
    if (!same_region(plan.persistent.conv_history[index],
                     expected.persistent.conv_history[index]) ||
        !same_region(plan.persistent.recurrent_state[index],
                     expected.persistent.recurrent_state[index])) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < plan.persistent.key.size(); ++index) {
    if (!same_matrix(plan.persistent.key[index],
                     expected.persistent.key[index]) ||
        !same_matrix(plan.persistent.value[index],
                     expected.persistent.value[index])) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < plan.layers.size(); ++index) {
    const auto& left = plan.layers[index];
    const auto& right = expected.layers[index];
    if (left.kind != right.kind || left.family_ordinal != right.family_ordinal ||
        !same_region(left.conv_history, right.conv_history) ||
        !same_region(left.recurrent_state, right.recurrent_state) ||
        !same_matrix(left.key, right.key) ||
        !same_matrix(left.value, right.value)) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < plan.alias_contracts.size(); ++index) {
    if (!same_alias(plan.alias_contracts[index],
                    expected.alias_contracts[index])) {
      return false;
    }
  }
  return plan.persistent_arena.arena_offset == 0U &&
         plan.persistent_arena.byte_size == kSm87TargetAotP40PersistentBytes &&
         plan.residual.storage.arena_offset ==
             kSm87TargetAotP40PersistentBytes &&
         plan.family_arena.arena_offset ==
             kSm87TargetAotP40PersistentBytes +
                 kSm87TargetAotP40ResidualBytes &&
         plan.final_hidden.storage.arena_offset +
                 plan.final_hidden.storage.byte_size ==
             plan.arena_bytes &&
         plan.gdn.normalized_input.storage.arena_offset ==
             plan.gdn.output.storage.arena_offset &&
         plan.attention.raw_q_gate.storage.arena_offset ==
             plan.attention.pre_gate_output.storage.arena_offset &&
         plan.attention.raw_q_gate.storage.byte_size ==
             plan.attention.pre_gate_output.storage.byte_size +
                 plan.attention.gated_output.storage.byte_size &&
         plan.mlp.normalized_input.storage.arena_offset ==
             plan.mlp.down_branch.storage.arena_offset &&
         plan.persistent.value.back().storage.arena_offset +
                 plan.persistent.value.back().storage.byte_size ==
             kSm87TargetAotP40PersistentBytes;
}

struct Sm87TargetAotP40RequestState::Impl final {
  Sm87TargetAotP40RequestMemoryPlan plan{};
  void* arena = nullptr;
  std::vector<cudaEvent_t> events;
  std::vector<PhysicalRangeIdentity> physical_ranges;
  OwnerSnapshot snapshot{};

  ~Impl() {
    for (const cudaEvent_t event : events) {
      if (event != nullptr) {
        (void)cudaEventDestroy(event);
      }
    }
    if (arena != nullptr) {
      (void)cudaFree(arena);
    }
    (void)cudaGetLastError();
  }

  [[nodiscard]] std::uint64_t physical_identity(
      const Sm87TargetAotRequestRegion& value) {
    for (const auto& current : physical_ranges) {
      if (current.offset == value.arena_offset &&
          current.bytes == value.byte_size) {
        return current.identity;
      }
    }
    const std::uint64_t identity = next_identity();
    physical_ranges.push_back(
        {value.arena_offset, value.byte_size, identity});
    return identity;
  }

  [[nodiscard]] BoundSpan bind(
      const Sm87TargetAotRequestRegion& value) {
    auto* const base = static_cast<std::uint8_t*>(arena);
    return {base + value.arena_offset, value.byte_size,
            physical_identity(value), next_identity(), next_identity()};
  }
};

Sm87TargetAotP40RequestState::Sm87TargetAotP40RequestState(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

Sm87TargetAotP40RequestState::~Sm87TargetAotP40RequestState() = default;

const Sm87TargetAotP40RequestMemoryPlan&
Sm87TargetAotP40RequestState::plan() const noexcept {
  return impl_->plan;
}

Sm87TargetAotRequestTransactionPhase
Sm87TargetAotP40RequestState::transaction_phase() const noexcept {
  return impl_->snapshot.transaction_phase;
}

bool Sm87TargetAotP40RequestState::committed() const noexcept {
  return transaction_phase() == Sm87TargetAotRequestTransactionPhase::kCommitted;
}

Sm87TargetAotRequestStateResult
create_sm87_target_aot_p40_request_state() noexcept {
  Sm87TargetAotRequestStateResult result;
#if !defined(Q3X_ENABLE_SM87_TARGET_AOT_REQUEST_STATE_V1_ADMISSION)
  result.diagnostic = {Sm87TargetAotRequestStateError::kAdmissionDisabled,
                       "Q3X_BUILD_SM87_TARGET_AOT_REQUEST_STATE_V1_ADMISSION",
                       0};
  return result;
#else
  try {
    auto impl = std::make_unique<Sm87TargetAotP40RequestState::Impl>();
    impl->plan = build_sm87_target_aot_p40_request_memory_plan();
    if (!validate_sm87_target_aot_p40_request_memory_plan(impl->plan)) {
      result.diagnostic = {Sm87TargetAotRequestStateError::kPlanInvalid,
                           "target_aot_p40_request_memory_plan", 0};
      return result;
    }
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    (void)total_bytes;
    if (status != cudaSuccess ||
        impl->plan.arena_bytes > static_cast<std::uint64_t>(free_bytes)) {
      result.diagnostic = {
          Sm87TargetAotRequestStateError::kInsufficientDeviceMemory,
          "cudaMemGetInfo(target_aot_p40_request_state)",
          static_cast<int>(status)};
      return result;
    }
    status = cudaMalloc(&impl->arena,
                        static_cast<std::size_t>(impl->plan.arena_bytes));
    if (status != cudaSuccess) {
      result.diagnostic = {Sm87TargetAotRequestStateError::kAllocationFailure,
                           "cudaMalloc(target_aot_p40_request_state)",
                           static_cast<int>(status)};
      return result;
    }
    impl->events.reserve(kSm87TargetAotP40OwnedEventCount);
    for (std::size_t index = 0U; index < kSm87TargetAotP40OwnedEventCount;
         ++index) {
      cudaEvent_t event = nullptr;
      status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      if (status != cudaSuccess) {
        result.diagnostic = {
            Sm87TargetAotRequestStateError::kEventCreationFailure,
            "cudaEventCreateWithFlags(target_aot_p40_request_state)",
            static_cast<int>(status)};
        return result;
      }
      impl->events.push_back(event);
    }
    status = cudaMemset(impl->arena, 0,
                        static_cast<std::size_t>(
                            impl->plan.persistent_arena.byte_size));
    if (status != cudaSuccess) {
      result.diagnostic = {
          Sm87TargetAotRequestStateError::kInitializationFailure,
          "cudaMemset(target_aot_p40_persistent_transaction)",
          static_cast<int>(status)};
      return result;
    }

    OwnerSnapshot& snapshot = impl->snapshot;
    snapshot.plan = &impl->plan;
    snapshot.allocation_base = impl->arena;
    snapshot.allocation_bytes = impl->plan.arena_bytes;
    snapshot.allocation_identity = next_identity();
    snapshot.epochs.admission_epoch = next_identity();
    snapshot.epochs.request_transaction_epoch = next_identity();
    snapshot.epochs.cold_reset_epoch = next_identity();
    snapshot.epochs.staged_kv_epoch = snapshot.epochs.cold_reset_epoch;
    snapshot.epochs.position_rope_epoch = next_identity();
    const std::array<std::uint64_t*, 17U> table_entries{{
        &snapshot.tables.request_memory_plan,
        &snapshot.tables.layer_residual,
        &snapshot.tables.input_norm,
        &snapshot.tables.gdn_raw_qkvz,
        &snapshot.tables.gdn_bf16_ab,
        &snapshot.tables.gdn_output,
        &snapshot.tables.attention_raw_qkv,
        &snapshot.tables.attention_processed_qkv,
        &snapshot.tables.attention_output,
        &snapshot.tables.mlp_activated,
        &snapshot.tables.down_branch,
        &snapshot.tables.staged_kv_transaction,
        &snapshot.tables.conv_history_transaction,
        &snapshot.tables.recurrent_state_transaction,
        &snapshot.tables.layer_completion_events,
        &snapshot.tables.request_transaction_events,
        &snapshot.tables.final_hidden,
    }};
    for (std::uint64_t* const entry : table_entries) {
      *entry = next_identity();
    }

    std::size_t event_index = 0U;
    for (std::size_t layer = 0U; layer < snapshot.layers.size(); ++layer) {
      auto& target = snapshot.layers[layer];
      const auto& binding = impl->plan.layers[layer];
      target.kind = binding.kind;
      target.family_ordinal = binding.family_ordinal;
      target.residual = impl->bind(impl->plan.residual.storage);
      target.mlp.normalized_input =
          impl->bind(impl->plan.mlp.normalized_input.storage);
      target.mlp.activated = impl->bind(impl->plan.mlp.activated.storage);
      target.mlp.down_branch = impl->bind(impl->plan.mlp.down_branch.storage);
      if (binding.kind == Sm87TargetAotRequestLayerKind::kGdn) {
        target.input_normalized =
            impl->bind(impl->plan.gdn.normalized_input.storage);
        target.gdn.raw_qkvz = impl->bind(impl->plan.gdn.raw_qkvz.storage);
        target.gdn.bf16_ab = impl->bind(impl->plan.gdn.bf16_ab.storage);
        target.gdn.output = impl->bind(impl->plan.gdn.output.storage);
        target.gdn.o_branch = impl->bind(impl->plan.gdn.o_branch.storage);
        target.gdn.final_conv_history = impl->bind(binding.conv_history);
        target.gdn.final_recurrent_state =
            impl->bind(binding.recurrent_state);
      } else {
        target.input_normalized =
            impl->bind(impl->plan.attention.normalized_input.storage);
        target.attention.raw_q_gate =
            impl->bind(impl->plan.attention.raw_q_gate.storage);
        target.attention.raw_k = impl->bind(binding.key.storage);
        target.attention.raw_v = impl->bind(binding.value.storage);
        target.attention.processed_q =
            impl->bind(impl->plan.attention.processed_q.storage);
        target.attention.processed_gate =
            impl->bind(impl->plan.attention.processed_gate.storage);
        target.attention.processed_k = target.attention.raw_k;
        target.attention.processed_k.lifetime_identity = next_identity();
        target.attention.processed_k.publication_identity = next_identity();
        target.attention.processed_v = target.attention.raw_v;
        target.attention.processed_v.lifetime_identity = next_identity();
        target.attention.processed_v.publication_identity = next_identity();
        target.attention.pre_gate_output =
            impl->bind(impl->plan.attention.pre_gate_output.storage);
        target.attention.gated_output =
            impl->bind(impl->plan.attention.gated_output.storage);
        target.attention.o_branch =
            impl->bind(impl->plan.attention.o_branch.storage);
        target.attention.k_in_place_transaction_alias_verified =
            target.attention.raw_k.device_data ==
                target.attention.processed_k.device_data &&
            target.attention.raw_k.byte_size ==
                target.attention.processed_k.byte_size &&
            target.attention.raw_k.physical_span_identity ==
                target.attention.processed_k.physical_span_identity;
        target.attention.v_in_place_transaction_alias_verified =
            target.attention.raw_v.device_data ==
                target.attention.processed_v.device_data &&
            target.attention.raw_v.byte_size ==
                target.attention.processed_v.byte_size &&
            target.attention.raw_v.physical_span_identity ==
                target.attention.processed_v.physical_span_identity;
      }
      for (auto& event : target.completion_events) {
        event = {reinterpret_cast<const void*>(impl->events[event_index]),
                 next_identity()};
        ++event_index;
      }
    }
    snapshot.final_hidden = impl->bind(impl->plan.final_hidden.storage);
    for (auto& event : snapshot.global_events) {
      event = {reinterpret_cast<const void*>(impl->events[event_index]),
               next_identity()};
      ++event_index;
    }
    snapshot.transaction_phase =
        Sm87TargetAotRequestTransactionPhase::kAdmittedUnpublished;
    snapshot.engine_rope_is_external =
        !impl->plan.engine_rope.included_in_request_arena;
    snapshot.all_kv_in_place_aliases_verified = true;
    for (const auto& layer : snapshot.layers) {
      if (layer.kind == Sm87TargetAotRequestLayerKind::kFullAttention) {
        snapshot.all_kv_in_place_aliases_verified =
            snapshot.all_kv_in_place_aliases_verified &&
            layer.attention.k_in_place_transaction_alias_verified &&
            layer.attention.v_in_place_transaction_alias_verified;
      }
    }
    if (event_index != impl->events.size() ||
        !sm87_target_aot_request_detail::validate_owner_snapshot(snapshot)) {
      result.diagnostic = {
          Sm87TargetAotRequestStateError::kOwnerSnapshotInvalid,
          "target_aot_p40_owner_snapshot", 0};
      return result;
    }
    result.value.reset(
        new Sm87TargetAotP40RequestState(std::move(impl)));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = {Sm87TargetAotRequestStateError::kAllocationFailure,
                         "host_owner_allocation", 0};
    return result;
  } catch (...) {
    result.diagnostic = {Sm87TargetAotRequestStateError::kAllocationFailure,
                         "unexpected_owner_creation_failure", 0};
    return result;
  }
#endif
}

const char* to_string(const Sm87TargetAotRequestStateError error) noexcept {
  switch (error) {
    case Sm87TargetAotRequestStateError::kNone:
      return "none";
    case Sm87TargetAotRequestStateError::kAdmissionDisabled:
      return "admission_disabled";
    case Sm87TargetAotRequestStateError::kPlanInvalid:
      return "plan_invalid";
    case Sm87TargetAotRequestStateError::kInsufficientDeviceMemory:
      return "insufficient_device_memory";
    case Sm87TargetAotRequestStateError::kAllocationFailure:
      return "allocation_failure";
    case Sm87TargetAotRequestStateError::kEventCreationFailure:
      return "event_creation_failure";
    case Sm87TargetAotRequestStateError::kInitializationFailure:
      return "initialization_failure";
    case Sm87TargetAotRequestStateError::kOwnerSnapshotInvalid:
      return "owner_snapshot_invalid";
  }
  return "unknown";
}

}  // namespace q3x::runtime

namespace q3x::runtime::sm87_target_aot_request_detail {
namespace {

[[nodiscard]] bool bound_span_valid(const OwnerSnapshot& snapshot,
                                    const BoundSpan& span) noexcept {
  if (snapshot.allocation_base == nullptr || span.device_data == nullptr ||
      span.byte_size == 0U || span.physical_span_identity == 0U ||
      span.lifetime_identity == 0U || span.publication_identity == 0U) {
    return false;
  }
  const auto base = reinterpret_cast<std::uintptr_t>(snapshot.allocation_base);
  const auto pointer = reinterpret_cast<std::uintptr_t>(span.device_data);
  return pointer >= base && pointer - base <= snapshot.allocation_bytes &&
         span.byte_size <= snapshot.allocation_bytes - (pointer - base);
}

[[nodiscard]] bool bound_event_valid(const BoundEvent& event) noexcept {
  return event.cuda_event != nullptr && event.event_identity != 0U;
}

[[nodiscard]] bool tables_valid(
    const OwnerTableIdentities& tables) noexcept {
  const std::array<std::uint64_t, 17U> values{{
      tables.request_memory_plan,
      tables.layer_residual,
      tables.input_norm,
      tables.gdn_raw_qkvz,
      tables.gdn_bf16_ab,
      tables.gdn_output,
      tables.attention_raw_qkv,
      tables.attention_processed_qkv,
      tables.attention_output,
      tables.mlp_activated,
      tables.down_branch,
      tables.staged_kv_transaction,
      tables.conv_history_transaction,
      tables.recurrent_state_transaction,
      tables.layer_completion_events,
      tables.request_transaction_events,
      tables.final_hidden,
  }};
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (values[index] == 0U) {
      return false;
    }
    for (std::size_t other = index + 1U; other < values.size(); ++other) {
      if (values[index] == values[other]) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool validate_owner_snapshot(const OwnerSnapshot& snapshot) noexcept {
  if (snapshot.plan == nullptr ||
      !validate_sm87_target_aot_p40_request_memory_plan(*snapshot.plan) ||
      snapshot.allocation_base == nullptr ||
      snapshot.allocation_bytes != snapshot.plan->arena_bytes ||
      snapshot.allocation_identity == 0U ||
      snapshot.epochs.admission_epoch == 0U ||
      snapshot.epochs.request_transaction_epoch == 0U ||
      snapshot.epochs.cold_reset_epoch == 0U ||
      snapshot.epochs.staged_kv_epoch != snapshot.epochs.cold_reset_epoch ||
      snapshot.epochs.position_rope_epoch == 0U ||
      !tables_valid(snapshot.tables) || !snapshot.engine_rope_is_external ||
      !snapshot.all_kv_in_place_aliases_verified ||
      snapshot.transaction_phase ==
          Sm87TargetAotRequestTransactionPhase::kInvalid ||
      !bound_span_valid(snapshot, snapshot.final_hidden)) {
    return false;
  }
  std::size_t gdn_count = 0U;
  std::size_t full_count = 0U;
  const void* residual_pointer = nullptr;
  std::uint64_t residual_physical_identity = 0U;
  for (const auto& layer : snapshot.layers) {
    if (!bound_span_valid(snapshot, layer.residual) ||
        !bound_span_valid(snapshot, layer.input_normalized) ||
        !bound_span_valid(snapshot, layer.mlp.normalized_input) ||
        !bound_span_valid(snapshot, layer.mlp.activated) ||
        !bound_span_valid(snapshot, layer.mlp.down_branch)) {
      return false;
    }
    if (residual_pointer == nullptr) {
      residual_pointer = layer.residual.device_data;
      residual_physical_identity = layer.residual.physical_span_identity;
    } else if (layer.residual.device_data != residual_pointer ||
               layer.residual.physical_span_identity !=
                   residual_physical_identity) {
      return false;
    }
    if (layer.mlp.normalized_input.device_data !=
            layer.mlp.down_branch.device_data ||
        layer.mlp.normalized_input.byte_size !=
            layer.mlp.down_branch.byte_size ||
        layer.mlp.normalized_input.physical_span_identity !=
            layer.mlp.down_branch.physical_span_identity) {
      return false;
    }
    for (const auto& event : layer.completion_events) {
      if (!bound_event_valid(event)) {
        return false;
      }
    }
    if (layer.kind == Sm87TargetAotRequestLayerKind::kGdn) {
      ++gdn_count;
      if (!bound_span_valid(snapshot, layer.gdn.raw_qkvz) ||
          !bound_span_valid(snapshot, layer.gdn.bf16_ab) ||
          !bound_span_valid(snapshot, layer.gdn.output) ||
          !bound_span_valid(snapshot, layer.gdn.o_branch) ||
          !bound_span_valid(snapshot, layer.gdn.final_conv_history) ||
          !bound_span_valid(snapshot, layer.gdn.final_recurrent_state) ||
          layer.input_normalized.device_data !=
              layer.gdn.output.device_data ||
          layer.gdn.raw_qkvz.device_data !=
              layer.gdn.o_branch.device_data) {
        return false;
      }
    } else if (layer.kind ==
               Sm87TargetAotRequestLayerKind::kFullAttention) {
      ++full_count;
      const auto& attention = layer.attention;
      if (!bound_span_valid(snapshot, attention.raw_q_gate) ||
          !bound_span_valid(snapshot, attention.raw_k) ||
          !bound_span_valid(snapshot, attention.raw_v) ||
          !bound_span_valid(snapshot, attention.processed_q) ||
          !bound_span_valid(snapshot, attention.processed_gate) ||
          !bound_span_valid(snapshot, attention.processed_k) ||
          !bound_span_valid(snapshot, attention.processed_v) ||
          !bound_span_valid(snapshot, attention.pre_gate_output) ||
          !bound_span_valid(snapshot, attention.gated_output) ||
          !bound_span_valid(snapshot, attention.o_branch) ||
          !attention.k_in_place_transaction_alias_verified ||
          !attention.v_in_place_transaction_alias_verified ||
          layer.input_normalized.device_data !=
              attention.processed_q.device_data ||
          attention.raw_q_gate.device_data !=
              attention.pre_gate_output.device_data ||
          static_cast<std::uint8_t*>(
              attention.pre_gate_output.device_data) +
                  attention.pre_gate_output.byte_size !=
              attention.gated_output.device_data ||
          attention.processed_q.device_data !=
              attention.o_branch.device_data ||
          attention.raw_k.device_data != attention.processed_k.device_data ||
          attention.raw_k.byte_size != attention.processed_k.byte_size ||
          attention.raw_k.physical_span_identity !=
              attention.processed_k.physical_span_identity ||
          attention.raw_v.device_data != attention.processed_v.device_data ||
          attention.raw_v.byte_size != attention.processed_v.byte_size ||
          attention.raw_v.physical_span_identity !=
              attention.processed_v.physical_span_identity) {
        return false;
      }
    } else {
      return false;
    }
  }
  for (const auto& event : snapshot.global_events) {
    if (!bound_event_valid(event)) {
      return false;
    }
  }
  return gdn_count == kSm87TargetAotP40GdnLayerCount &&
         full_count == kSm87TargetAotP40FullLayerCount;
}

const OwnerSnapshot* Sm87TargetAotRequestStateAccess::snapshot(
    const Sm87TargetAotP40RequestState& owner) noexcept {
  return owner.impl_ != nullptr && validate_owner_snapshot(owner.impl_->snapshot)
             ? &owner.impl_->snapshot
             : nullptr;
}

}  // namespace q3x::runtime::sm87_target_aot_request_detail
