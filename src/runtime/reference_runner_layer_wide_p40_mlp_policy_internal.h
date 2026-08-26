#pragma once

#include "q3x/runtime/prefill_execution_plan.h"
#include "q3x/runtime/request_state.h"
#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
#include "q3x/runtime/prefill_workspace_plan.h"
#endif

#include <cstdint>

namespace q3x::runtime::reference_runner_detail {

// Pure-host policy for the shared layer-wide P40 MLP launcher. The retained
// P40001 path deliberately depends only on its original MLP layout/capacity
// contract. P40016 is an additional selector-test identity and therefore
// must prove the complete whole-core descriptor before it can reach the same
// fixed-M40000 kernels.
struct LayerWideP40MlpDescriptorPolicyInput {
  LayerMajorRequestMlpLayout mlp_layout =
      LayerMajorRequestMlpLayout::kPanelLocalThreeSpan;
  std::uint32_t max_sequence_length = 0U;
  std::uint32_t mlp_capacity_tokens = 0U;
#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
  bool whole_core_projection_package = false;
  RequestMemoryProfile profile = RequestMemoryProfile::kLegacyC512;
  LayerMajorRequestLayout layout =
      LayerMajorRequestLayout::kC8192FamilyOverlay;
  std::uint32_t whole_core_request_capacity_tokens = 0U;
  std::uint64_t arena_bytes = 0U;
#endif
};

[[nodiscard]] constexpr bool valid_layer_wide_p40_mlp_descriptor_policy(
    const LayerWideP40MlpDescriptorPolicyInput& input) noexcept {
  if (input.mlp_layout !=
          LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan ||
      input.mlp_capacity_tokens !=
          kLayerMajorPrefillLayerWideMlpP40Tokens) {
    return false;
  }
  if (input.max_sequence_length ==
      kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens) {
    return true;
  }
#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
  return input.whole_core_projection_package &&
         input.profile == RequestMemoryProfile::kLayerMajorP40WholeCore &&
         input.layout ==
             LayerMajorRequestLayout::kP40WholeCorePromptWide &&
         is_selector_exact_persistent_attention_v1_p40_request_capacity_tokens(
             input.max_sequence_length) &&
         input.whole_core_request_capacity_tokens ==
             input.max_sequence_length &&
         input.arena_bytes == layer_major_p40_whole_core_arena_bytes(
                                  input.max_sequence_length);
#else
  return false;
#endif
}

[[nodiscard]] constexpr bool valid_layer_wide_p40_mlp_residual_capacity(
    const LayerWideP40MlpDescriptorPolicyInput& input,
    const std::uint32_t residual_row_capacity) noexcept {
  return valid_layer_wide_p40_mlp_descriptor_policy(input) &&
         residual_row_capacity == input.max_sequence_length;
}

#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
[[nodiscard]] constexpr LayerWideP40MlpDescriptorPolicyInput
selector_exact_p40016_mlp_descriptor_policy_input() noexcept {
  LayerWideP40MlpDescriptorPolicyInput input;
  input.mlp_layout =
      LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan;
  input.max_sequence_length =
      kSelectorExactPersistentAttentionV1P40RequestCapacityTokens;
  input.mlp_capacity_tokens = kLayerMajorPrefillLayerWideMlpP40Tokens;
  input.whole_core_projection_package = true;
  input.profile = RequestMemoryProfile::kLayerMajorP40WholeCore;
  input.layout = LayerMajorRequestLayout::kP40WholeCorePromptWide;
  input.whole_core_request_capacity_tokens = input.max_sequence_length;
  input.arena_bytes =
      layer_major_p40_whole_core_arena_bytes(input.max_sequence_length);
  return input;
}

[[nodiscard]] constexpr bool
selector_exact_p40016_mlp_descriptor_policy_rejects_mutations() noexcept {
  LayerWideP40MlpDescriptorPolicyInput malformed =
      selector_exact_p40016_mlp_descriptor_policy_input();
  malformed.whole_core_projection_package = false;
  if (valid_layer_wide_p40_mlp_descriptor_policy(malformed)) {
    return false;
  }
  malformed = selector_exact_p40016_mlp_descriptor_policy_input();
  malformed.profile = RequestMemoryProfile::kLayerMajorC8192;
  if (valid_layer_wide_p40_mlp_descriptor_policy(malformed)) {
    return false;
  }
  malformed = selector_exact_p40016_mlp_descriptor_policy_input();
  malformed.layout = LayerMajorRequestLayout::kC8192FamilyOverlay;
  if (valid_layer_wide_p40_mlp_descriptor_policy(malformed)) {
    return false;
  }
  malformed = selector_exact_p40016_mlp_descriptor_policy_input();
  malformed.max_sequence_length = 40'015U;
  malformed.whole_core_request_capacity_tokens = 40'015U;
  if (valid_layer_wide_p40_mlp_descriptor_policy(malformed)) {
    return false;
  }
  malformed = selector_exact_p40016_mlp_descriptor_policy_input();
  malformed.max_sequence_length = 40'017U;
  malformed.whole_core_request_capacity_tokens = 40'017U;
  if (valid_layer_wide_p40_mlp_descriptor_policy(malformed)) {
    return false;
  }
  malformed = selector_exact_p40016_mlp_descriptor_policy_input();
  malformed.whole_core_request_capacity_tokens = 40'001U;
  if (valid_layer_wide_p40_mlp_descriptor_policy(malformed)) {
    return false;
  }
  malformed = selector_exact_p40016_mlp_descriptor_policy_input();
  ++malformed.arena_bytes;
  if (valid_layer_wide_p40_mlp_descriptor_policy(malformed)) {
    return false;
  }
  malformed = selector_exact_p40016_mlp_descriptor_policy_input();
  malformed.mlp_capacity_tokens = 40'016U;
  return !valid_layer_wide_p40_mlp_descriptor_policy(malformed);
}

static_assert(valid_layer_wide_p40_mlp_descriptor_policy(
    selector_exact_p40016_mlp_descriptor_policy_input()));
static_assert(valid_layer_wide_p40_mlp_residual_capacity(
    selector_exact_p40016_mlp_descriptor_policy_input(), 40'016U));
static_assert(!valid_layer_wide_p40_mlp_residual_capacity(
    selector_exact_p40016_mlp_descriptor_policy_input(), 40'001U));
static_assert(
    selector_exact_p40016_mlp_descriptor_policy_rejects_mutations());
#endif

}  // namespace q3x::runtime::reference_runner_detail
