#include "reference_engine_prefill_authority.h"

#include "q3x/kernels/gdn_prefill_prompt_wide_chunk_graph_abi.h"
#include "q3x/kernels/sm87_fp8_marlin_w8a16.h"
#include "q3x/kernels/sm87_nvfp4_marlin.h"
#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_bf16_ab_prefill.h"
#endif
#if defined(Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_nvfp4_prefill_large_m.h"
#endif
#if defined(Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_nvfp4_prefill_g2_d2.h"
#endif
#if defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_nvfp4_prefill_persistent.h"
#endif
#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/reference_engine.h"

#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
#include "../kernels/sm87/gdn_prefill_chunk64_native_sm87.h"
#endif

#include <new>
#include <utility>

namespace q3x::runtime::reference_engine_detail {
namespace {

thread_local bool
    g_reference_engine_prefill_compatibility_oracle_for_test = false;

[[nodiscard]] BoundPrefillPlanResult plan_failure(
    const BoundPrefillPlanError error,
    const ReferenceRunnerError runner_error,
    const char* const operation) noexcept {
  BoundPrefillPlanResult result;
  result.error = error;
  result.status.error = runner_error;
  result.status.operation = operation;
  return result;
}

[[nodiscard]] ReferenceWholeRequestPrefillOutcome execution_failure(
    const char* const operation) noexcept {
  ReferenceWholeRequestPrefillOutcome result;
  result.status.error = ReferenceRunnerError::kInvalidStepOptions;
  result.status.operation = operation;
  return result;
}

[[nodiscard]] ReferenceRunnerStatus status_failure(
    const char* const operation) noexcept {
  ReferenceRunnerStatus result;
  result.error = ReferenceRunnerError::kInvalidStepOptions;
  result.operation = operation;
  return result;
}

[[maybe_unused, nodiscard]] bool valid_fp8_marlin_binding(
    const LinearWeight& weight, const std::size_t expected_output_size,
    const std::size_t expected_input_size) noexcept {
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
  const auto aligned_16 = [](const void* const pointer) noexcept {
    return pointer != nullptr &&
           reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
  };
  return fp8 != nullptr && fp8->weight != nullptr &&
         fp8->weight_scale_device != nullptr &&
         fp8->input_scale_device != nullptr &&
         aligned_16(fp8->prefill_marlin_weight) &&
         aligned_16(fp8->prefill_marlin_scales) &&
         fp8->output_size == expected_output_size &&
         fp8->input_size == expected_input_size;
#else
  (void)weight;
  (void)expected_output_size;
  (void)expected_input_size;
  return false;
#endif
}

[[maybe_unused, nodiscard]] bool valid_nvfp4_marlin_binding(
    const LinearWeight& weight, const std::size_t expected_output_size,
    const std::size_t expected_input_size) noexcept {
#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
  const auto* const nvfp4 = std::get_if<NvFp4LinearWeight>(&weight);
  const auto aligned_16 = [](const void* const pointer) noexcept {
    return pointer != nullptr &&
           reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
  };
  return nvfp4 != nullptr && nvfp4->packed_weight != nullptr &&
         nvfp4->block_scale != nullptr &&
         nvfp4->weight_scale_2_device != nullptr &&
         nvfp4->input_scale_device != nullptr &&
         aligned_16(nvfp4->prefill_marlin_weight) &&
         aligned_16(nvfp4->prefill_marlin_scales) &&
         nvfp4->prefill_marlin_global_scale != nullptr &&
         nvfp4->output_size == expected_output_size &&
         nvfp4->input_size == expected_input_size;
#else
  (void)weight;
  (void)expected_output_size;
  (void)expected_input_size;
  return false;
#endif
}

[[maybe_unused, nodiscard]] bool valid_bf16_binding(
    const LinearWeight& weight, const std::size_t expected_output_size,
    const std::size_t expected_input_size) noexcept {
  const auto* const bf16 = std::get_if<Bf16LinearWeight>(&weight);
  return bf16 != nullptr && bf16->weight != nullptr &&
         bf16->output_size == expected_output_size &&
         bf16->input_size == expected_input_size;
}

[[nodiscard]] bool complete_installed_projection_inventory(
    const ModelWeights& weights,
    const NvFp4MarlinGateUpLayout expected_gate_up_layout) noexcept {
#if !defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION) || \
    !defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION) || \
    !defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  (void)weights;
  (void)expected_gate_up_layout;
  return false;
#else
  for (std::size_t layer = 0U;
       layer < kReferenceDecoderLayerCount; ++layer) {
    const DecoderLayerWeights& layer_weights = weights.layer(layer);
    if (!valid_nvfp4_marlin_binding(
            layer_weights.mlp.gate_proj, 17'408U, kReferenceHiddenSize) ||
        !valid_nvfp4_marlin_binding(
            layer_weights.mlp.up_proj, 17'408U, kReferenceHiddenSize) ||
        !valid_nvfp4_marlin_binding(
            layer_weights.mlp.down_proj, kReferenceHiddenSize, 17'408U)) {
      return false;
    }
    const auto* const gate =
        std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.gate_proj);
    const auto* const up =
        std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.up_proj);
    if (gate == nullptr || up == nullptr ||
        gate->prefill_marlin_gate_up_layout != expected_gate_up_layout ||
        up->prefill_marlin_gate_up_layout !=
            gate->prefill_marlin_gate_up_layout ||
        gate->prefill_marlin_weight != up->prefill_marlin_weight ||
        gate->prefill_marlin_scales != up->prefill_marlin_scales ||
        gate->prefill_marlin_global_scale !=
            up->prefill_marlin_global_scale) {
      return false;
    }

    const model::LayerType layer_type =
        reference_runner_detail::expected_reference_layer_type(layer);
    if (layer_type == model::LayerType::kLinearAttention) {
      const auto* const linear =
          std::get_if<LinearAttentionWeights>(&layer_weights.attention);
      if (linear == nullptr ||
          !valid_fp8_marlin_binding(
              linear->in_proj_qkv, 10'240U, kReferenceHiddenSize) ||
          !valid_fp8_marlin_binding(
              linear->in_proj_z, 6'144U, kReferenceHiddenSize) ||
          !valid_bf16_binding(
              linear->in_proj_a, 48U, kReferenceHiddenSize) ||
          !valid_bf16_binding(
              linear->in_proj_b, 48U, kReferenceHiddenSize) ||
          !valid_fp8_marlin_binding(
              linear->out_proj, kReferenceHiddenSize, 6'144U) ||
          linear->conv1d.data == nullptr || linear->a_log.data == nullptr ||
          linear->dt_bias.data == nullptr || linear->norm.data == nullptr) {
        return false;
      }
    } else if (layer_type == model::LayerType::kFullAttention) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr ||
          !valid_fp8_marlin_binding(
              attention->q_proj, 12'288U, kReferenceHiddenSize) ||
          !valid_fp8_marlin_binding(
              attention->k_proj, 1'024U, kReferenceHiddenSize) ||
          !valid_fp8_marlin_binding(
              attention->v_proj, 1'024U, kReferenceHiddenSize) ||
          !valid_fp8_marlin_binding(
              attention->o_proj, kReferenceHiddenSize, 6'144U) ||
          attention->q_norm.data == nullptr ||
          attention->q_norm.element_count != 256U ||
          attention->k_norm.data == nullptr ||
          attention->k_norm.element_count != 256U) {
        return false;
      }
    } else {
      return false;
    }
    if (layer_weights.input_layernorm.data == nullptr ||
        layer_weights.input_layernorm.element_count != kReferenceHiddenSize ||
        layer_weights.post_attention_layernorm.data == nullptr ||
        layer_weights.post_attention_layernorm.element_count !=
            kReferenceHiddenSize) {
      return false;
    }
  }
  return weights.embed_tokens().weight != nullptr &&
         weights.embed_tokens().input_size == kReferenceHiddenSize &&
         weights.final_norm().data != nullptr &&
         weights.final_norm().element_count == kReferenceHiddenSize;
#endif
}

[[maybe_unused, nodiscard]] bool
complete_nvfp4_true_large_m_capability() noexcept {
#if !defined(Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
  return false;
#else
  static_assert(kLayerMajorPrefillOperatorPanelTokens ==
                kernels::kSm87NvFp4PrefillLargeMMaximumTokens);
  static_assert(kLayerMajorPrefillTrueLargeMPartialPanelTokens ==
                kernels::kSm87NvFp4PrefillLargeMShortPanelTokens);
  constexpr std::array<kernels::Sm87NvFp4PrefillLargeMRole, 2U> kRoles{
      kernels::Sm87NvFp4PrefillLargeMRole::kGateUp,
      kernels::Sm87NvFp4PrefillLargeMRole::kDown};
  constexpr std::array<std::size_t, 2U> kPanelSizes{
      kLayerMajorPrefillOperatorPanelTokens,
      kLayerMajorPrefillTrueLargeMPartialPanelTokens};
  for (const kernels::Sm87NvFp4PrefillLargeMRole role : kRoles) {
    for (const std::size_t panel_m : kPanelSizes) {
      kernels::Sm87NvFp4PrefillLargeMCapability capability;
      kernels::Sm87NvFp4PrefillLargeMResources resources;
      const int status =
          kernels::query_sm87_nvfp4_prefill_large_m_capability_cuda(
              role, panel_m, &capability);
      if (status != 0 || !capability.supported ||
          !capability.plan.valid() || capability.plan.role != role ||
          capability.plan.token_count != panel_m) {
        return false;
      }
      const int resource_status =
          kernels::query_sm87_nvfp4_prefill_large_m_resources_cuda(
              role, panel_m, &resources);
      if (resource_status != 0 || resources.active_blocks_per_sm < 1 ||
          resources.maximum_threads_per_block <
              static_cast<int>(capability.plan.threads) ||
          resources.dynamic_shared_bytes !=
              capability.plan.dynamic_shared_bytes ||
          resources.dynamic_shared_bytes != 82'944U ||
          resources.local_bytes != 0U) {
        return false;
      }
    }
  }
  return true;
#endif
}

[[maybe_unused, nodiscard]] bool
complete_nvfp4_g2_d2_capability() noexcept {
#if !defined(Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION)
  return false;
#else
  constexpr std::array<kernels::Sm87NvFp4PrefillG2D2Role, 2U> kRoles{
      kernels::Sm87NvFp4PrefillG2D2Role::kGateUpG2,
      kernels::Sm87NvFp4PrefillG2D2Role::kDownD2};
  constexpr std::array<std::size_t, 2U> kPanelSizes{
      kLayerMajorPrefillOperatorPanelTokens,
      kLayerMajorPrefillTrueLargeMPartialPanelTokens};
  const auto same_plan = [](
                             const kernels::Sm87NvFp4PrefillG2D2Plan& left,
                             const kernels::Sm87NvFp4PrefillG2D2Plan& right)
      noexcept {
        return left.role == right.role && left.dataflow == right.dataflow &&
               left.token_count == right.token_count &&
               left.input_features == right.input_features &&
               left.weight_output_features == right.weight_output_features &&
               left.published_output_features ==
                   right.published_output_features &&
               left.tile_m == right.tile_m &&
               left.branch_tile_n == right.branch_tile_n &&
               left.tile_k == right.tile_k && left.threads == right.threads &&
               left.pipeline_stages == right.pipeline_stages &&
               left.grid_m == right.grid_m && left.grid_n == right.grid_n &&
               left.tail_rows == right.tail_rows &&
               left.dynamic_shared_bytes == right.dynamic_shared_bytes;
      };
  constexpr std::size_t kMaximumTotalSharedBytes = 50U * 1024U;
  for (const kernels::Sm87NvFp4PrefillG2D2Role role : kRoles) {
    for (const std::size_t panel_m : kPanelSizes) {
      const kernels::Sm87NvFp4PrefillG2D2Plan expected =
          kernels::sm87_nvfp4_prefill_g2_d2_plan(role, panel_m);
      kernels::Sm87NvFp4PrefillG2D2Capability capability;
      kernels::Sm87NvFp4PrefillG2D2Resources resources;
      if (!kernels::sm87_nvfp4_prefill_g2_d2_supports(role, panel_m) ||
          !expected.valid() ||
          kernels::query_sm87_nvfp4_prefill_g2_d2_capability_cuda(
              role, panel_m, &capability) != 0 ||
          !capability.supported || !capability.plan.valid() ||
          !same_plan(capability.plan, expected) ||
          kernels::query_sm87_nvfp4_prefill_g2_d2_resources_cuda(
              role, panel_m, &resources) != 0 ||
          resources.active_blocks_per_sm < 2 ||
          resources.registers_per_thread <= 0 ||
          resources.registers_per_thread > 128 ||
          resources.maximum_threads_per_block <
              static_cast<int>(expected.threads) ||
          resources.dynamic_shared_bytes != expected.dynamic_shared_bytes ||
          resources.static_shared_bytes > kMaximumTotalSharedBytes ||
          resources.dynamic_shared_bytes >
              kMaximumTotalSharedBytes - resources.static_shared_bytes ||
          resources.local_bytes != 0U) {
        return false;
      }
    }
  }
  return true;
#endif
}

[[maybe_unused, nodiscard]] bool
complete_nvfp4_persistent_p40_capability() noexcept {
#if !defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION)
  return false;
#else
  static_assert(kLayerMajorPrefillLayerWideMlpP40Tokens ==
                kernels::kSm87NvFp4PersistentPrefillP40Tokens);
  constexpr std::array<kernels::Sm87NvFp4PersistentPrefillRole, 2U> kRoles{
      kernels::Sm87NvFp4PersistentPrefillRole::kGateUpPaired,
      kernels::Sm87NvFp4PersistentPrefillRole::kDown};
  for (const kernels::Sm87NvFp4PersistentPrefillRole role : kRoles) {
    const kernels::Sm87NvFp4PersistentPrefillPlan expected =
        kernels::sm87_nvfp4_persistent_prefill_plan(
            role, kLayerMajorPrefillLayerWideMlpP40Tokens);
    kernels::Sm87NvFp4PersistentPrefillCapability capability;
    kernels::Sm87NvFp4PersistentPrefillResources resources;
    const std::size_t maximum_local_bytes =
        role == kernels::Sm87NvFp4PersistentPrefillRole::kGateUpPaired
            ? kernels::kSm87NvFp4PersistentPrefillGateMaximumLocalBytes
            : kernels::kSm87NvFp4PersistentPrefillDownMaximumLocalBytes;
    if (!expected.valid() ||
        kernels::query_sm87_nvfp4_persistent_prefill_capability_cuda(
            role, kLayerMajorPrefillLayerWideMlpP40Tokens,
            &capability) != 0 ||
        !capability.supported || !capability.plan.valid() ||
        capability.plan.role != expected.role ||
        capability.plan.raster != expected.raster ||
        capability.plan.token_count != expected.token_count ||
        capability.plan.task_count != expected.task_count ||
        capability.plan.persistent_ctas != expected.persistent_ctas ||
        capability.plan.dynamic_shared_bytes !=
            expected.dynamic_shared_bytes ||
        kernels::query_sm87_nvfp4_persistent_prefill_resources_cuda(
            role, kLayerMajorPrefillLayerWideMlpP40Tokens,
            &resources) != 0 ||
        resources.active_blocks_per_sm < 1 ||
        resources.maximum_threads_per_block <
            static_cast<int>(expected.threads) ||
        resources.dynamic_shared_bytes != expected.dynamic_shared_bytes ||
        resources.local_bytes > maximum_local_bytes) {
      return false;
    }
  }
  return true;
#endif
}

[[maybe_unused, nodiscard]] bool
complete_bf16_ab_prompt_wide_p40_capability() noexcept {
#if !defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
  return false;
#else
  static_assert(kLayerMajorPrefillPromptWideP40Tokens ==
                kernels::kSm87Bf16AbPromptWideP40Tokens);
  const kernels::Sm87Bf16AbPromptWideP40Plan plan =
      kernels::make_sm87_bf16_ab_prompt_wide_p40_plan(
          kLayerMajorPrefillPromptWideP40Tokens);
  kernels::Sm87Bf16AbPromptWideP40Resources resources{};
  return plan.valid() &&
         kernels::query_sm87_bf16_ab_prompt_wide_p40_resources_cuda(
             &resources) == 0 &&
         resources.valid() && resources.local_bytes == 0U &&
         resources.dynamic_shared_bytes == plan.dynamic_shared_bytes;
#endif
}

[[maybe_unused, nodiscard]] bool
complete_gdn_prompt_wide_p40_capability() noexcept {
#if !defined(Q3X_ENABLE_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION) || \
    !defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  return false;
#else
  static_assert(kLayerMajorPrefillPromptWideP40Tokens ==
                kernels::kGdnPromptWideChunkGraphP40Tokens);
  gdn_prefill_prompt_wide_chunk_graph_detail::ResourcePreflightReceipt
      resources{};
  return kernels::kGdnPromptWideChunkGraphP40WorkspacePlan.ok() &&
         kernels::kGdnPromptWideChunkGraphP40WorkspacePlan.chunk_count ==
             625U &&
         gdn_prefill_prompt_wide_chunk_graph_detail::supports(
             kLayerMajorPrefillPromptWideP40Tokens) &&
         gdn_prefill_prompt_wide_chunk_graph_detail::workspace_bytes() ==
             kernels::kGdnPromptWideChunkGraphP40WorkspaceBytes &&
         gdn_prefill_prompt_wide_chunk_graph_detail::preflight_resources(
             &resources) == 0 &&
         resources.registers_per_thread > 0 &&
         resources.maximum_threads_per_block > 0 &&
         resources.active_blocks_per_sm > 0;
#endif
}

[[maybe_unused, nodiscard]] bool
complete_flashinfer_whole_prompt_p40_capability() noexcept {
#if !defined(Q3X_ENABLE_FLASHINFER_PREFILL_ATTENTION_ADMISSION)
  return false;
#else
  return has_bulk_causal_gqa_flashinfer_exact_panel_cuda() &&
         can_launch_bulk_causal_gqa_flashinfer_exact_whole_prompt(
             0U, kLayerMajorPrefillPromptWideP40Tokens);
#endif
}

[[nodiscard]] constexpr bool
whole_core_compile_inventory_enabled() noexcept {
#if defined(Q3X_ENABLE_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION) && \
    defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION) && \
    defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION) && \
    defined(Q3X_ENABLE_LAYER_WIDE_P40_MLP_ADMISSION) && \
    defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION) && \
    defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION) && \
    defined(Q3X_ENABLE_FLASHINFER_PREFILL_ATTENTION_ADMISSION) && \
    defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION) && \
    defined(Q3X_ENABLE_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION)
  return true;
#else
  return false;
#endif
}

[[maybe_unused, nodiscard]] bool
complete_exact_gdn_chunk64_native_inventory(
    const ModelWeights& weights) noexcept {
#if !defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  (void)weights;
  return false;
#else
  std::size_t linear_attention_layer_count = 0U;
  for (std::size_t layer = 0U;
       layer < kReferenceDecoderLayerCount; ++layer) {
    if (reference_runner_detail::expected_reference_layer_type(layer) !=
        model::LayerType::kLinearAttention) {
      continue;
    }
    ++linear_attention_layer_count;
    const auto* const linear = std::get_if<LinearAttentionWeights>(
        &weights.layer(layer).attention);
    if (linear == nullptr ||
        linear_output_size(linear->in_proj_qkv) != 10'240U ||
        linear_input_size(linear->in_proj_qkv) != kReferenceHiddenSize ||
        linear_output_size(linear->in_proj_a) != 48U ||
        linear_input_size(linear->in_proj_a) != kReferenceHiddenSize ||
        linear_output_size(linear->in_proj_b) != 48U ||
        linear_input_size(linear->in_proj_b) != kReferenceHiddenSize ||
        linear->conv1d.data == nullptr ||
        linear->conv1d.shape !=
            std::array<std::size_t, 3U>{10'240U, 1U, 4U} ||
        linear->a_log.data == nullptr ||
        linear->a_log.element_count != 48U ||
        linear->dt_bias.data == nullptr ||
        linear->dt_bias.element_count != 48U ||
        linear->norm.data == nullptr ||
        linear->norm.element_count != 128U) {
      return false;
    }
  }
  return linear_attention_layer_count == kQwen36LinearAttentionLayerCount;
#endif
}

[[nodiscard]] bool same_geometry(
    const PrefillExecutionPlan& left,
    const PrefillExecutionPlan& right) noexcept {
  if (!is_valid_unbound_layer_major_prefill_execution_plan(left) ||
      !is_valid_unbound_layer_major_prefill_execution_plan(right) ||
      left.traversal != right.traversal ||
      left.legacy_public_tile_limit != right.legacy_public_tile_limit ||
      left.operator_panel_capacity != right.operator_panel_capacity ||
      left.first_position != right.first_position ||
      left.prompt_token_count != right.prompt_token_count ||
      left.final_position != right.final_position ||
      left.panel_count != right.panel_count ||
      left.final_commit.expected_initial_sequence_length !=
          right.final_commit.expected_initial_sequence_length ||
      left.final_commit.committed_sequence_length !=
          right.final_commit.committed_sequence_length ||
      left.final_commit.commit_count != right.final_commit.commit_count ||
      left.mlp_schedule.tactic != right.mlp_schedule.tactic ||
      left.mlp_schedule.operator_panel_phase_count_per_layer !=
          right.mlp_schedule.operator_panel_phase_count_per_layer ||
      left.mlp_schedule.mlp_phase_submission_count_per_layer !=
          right.mlp_schedule.mlp_phase_submission_count_per_layer ||
      left.mlp_schedule.maximum_m_per_mlp_submission !=
          right.mlp_schedule.maximum_m_per_mlp_submission ||
      left.mlp_schedule.waits_for_all_operator_panels !=
          right.mlp_schedule.waits_for_all_operator_panels ||
      left.mlp_schedule.post_attention_norm_is_prompt_wide !=
          right.mlp_schedule.post_attention_norm_is_prompt_wide ||
      left.mlp_schedule.exact_full_m_binding_required !=
          right.mlp_schedule.exact_full_m_binding_required ||
      left.mlp_schedule.internal_m_segmentation_forbidden !=
          right.mlp_schedule.internal_m_segmentation_forbidden ||
      left.whole_core_schedule.enabled != right.whole_core_schedule.enabled ||
      left.whole_core_schedule.fill_panel_phase_count_per_layer !=
          right.whole_core_schedule.fill_panel_phase_count_per_layer ||
      left.whole_core_schedule.prompt_core_phase_count_per_layer !=
          right.whole_core_schedule.prompt_core_phase_count_per_layer ||
      left.whole_core_schedule.drain_panel_phase_count_per_layer !=
          right.whole_core_schedule.drain_panel_phase_count_per_layer ||
      left.whole_core_schedule.persistent_mlp_phase_count_per_layer !=
          right.whole_core_schedule.persistent_mlp_phase_count_per_layer ||
      left.whole_core_schedule.panel_token_count !=
          right.whole_core_schedule.panel_token_count ||
      left.whole_core_schedule.prompt_core_token_count !=
          right.whole_core_schedule.prompt_core_token_count ||
      left.whole_core_schedule.request_capacity_tokens !=
          right.whole_core_schedule.request_capacity_tokens ||
      left.whole_core_schedule.route_pass_count !=
          right.whole_core_schedule.route_pass_count ||
      left.whole_core_schedule.fp8_single_launch_per_projection_required !=
          right.whole_core_schedule.fp8_single_launch_per_projection_required ||
      left.whole_core_schedule.bf16_ab_prompt_wide_required !=
          right.whole_core_schedule.bf16_ab_prompt_wide_required ||
      left.whole_core_schedule.gdn_prompt_wide_required !=
          right.whole_core_schedule.gdn_prompt_wide_required ||
      left.whole_core_schedule.flashinfer_whole_prompt_required !=
          right.whole_core_schedule.flashinfer_whole_prompt_required) {
    return false;
  }
  for (std::size_t panel = 0U; panel < left.panel_count; ++panel) {
    const PrefillOperatorPanel& a = left.panels[panel];
    const PrefillOperatorPanel& b = right.panels[panel];
    if (a.ordinal != b.ordinal || a.first_position != b.first_position ||
        a.token_count != b.token_count ||
        a.end_position != b.end_position) {
      return false;
    }
  }
  for (std::size_t layer = 0U; layer < left.layers.size(); ++layer) {
    const PrefillLayerExecution& a = left.layers[layer];
    const PrefillLayerExecution& b = right.layers[layer];
    if (a.layer_index != b.layer_index || a.layer_type != b.layer_type ||
        a.progress_domain != b.progress_domain ||
        a.panel_count != b.panel_count) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr NativePrefillTactic native_attention_tactic(
    const LayerMajorPrefillFullAttentionTactic tactic) noexcept {
  switch (tactic) {
    case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel:
      return NativePrefillTactic::
          kNativeCausalAttentionGroupQ64OperatorPanel;
    case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ128V4Panel:
      return NativePrefillTactic::
          kNativeCausalAttentionGroupQ128V4OperatorPanel;
    case LayerMajorPrefillFullAttentionTactic::kNativeFlashInferExactPanel:
      return NativePrefillTactic::
          kNativeCausalAttentionFlashInferExactOperatorPanel;
    case LayerMajorPrefillFullAttentionTactic::
        kNativeFlashInferExactWholePrompt:
      return NativePrefillTactic::
          kNativeCausalAttentionFlashInferExactWholePrompt;
    case LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512:
    default:
      return NativePrefillTactic::
          kExactCausalAttentionOracleSpanC512C16Reference256;
  }
}

[[nodiscard]] constexpr std::uint32_t native_attention_maximum_physical_m(
    const LayerMajorPrefillFullAttentionTactic tactic) noexcept {
  return tactic == LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512
             ? kPrefillPhysicalSegmentMaximumTokens
         : tactic == LayerMajorPrefillFullAttentionTactic::
                       kNativeFlashInferExactWholePrompt
             ? kLayerMajorPrefillPromptWideP40Tokens
             : kLayerMajorPrefillOperatorPanelTokens;
}

[[nodiscard]] constexpr const LayerMajorPrefillArithmeticContract*
projection_arithmetic_contract(
    const LayerMajorPrefillProjectionTactic tactic) noexcept {
  return tactic == LayerMajorPrefillProjectionTactic::
                       kNativePromptWideP40WholeCore
             ? &kLayerMajorPrefillPromptWideP40WholeCoreArithmeticContract
         : tactic == LayerMajorPrefillProjectionTactic::
                       kNativeNvfp4PersistentP40LayerWideMlp
             ? &kLayerMajorPrefillPersistentP40NvFp4ArithmeticContract
         : tactic == LayerMajorPrefillProjectionTactic::
                       kNativeNvfp4G2D2LargeMOperatorPanel
             ? &kLayerMajorPrefillTrueLargeMNvFp4ArithmeticContract
         : tactic == LayerMajorPrefillProjectionTactic::
                       kNativeNvfp4TrueLargeMOperatorPanel
             ? &kLayerMajorPrefillTrueLargeMNvFp4ArithmeticContract
         : tactic == LayerMajorPrefillProjectionTactic::
                       kNativeQuantizedLargeMOperatorPanel
             ? &kLayerMajorPrefillExactMarlinM8192ArithmeticContract
         : tactic == LayerMajorPrefillProjectionTactic::
                       kSegmentedMarlinOperatorPanel
             ? &kLayerMajorPrefillSegmentedMarlinArithmeticContract
         : tactic == LayerMajorPrefillProjectionTactic::kExactSegmentedC512
             ? &kLayerMajorPrefillExactArithmeticContract
             : nullptr;
}

inline constexpr std::uint64_t kPromptWideP40WholeCoreArenaBytes =
    8'640'542'976U;

[[nodiscard]] bool exact_matrix_view(
    const DeviceMatrixView& view, const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t element_bytes = sizeof(std::uint16_t)) noexcept {
  const std::uint64_t elements =
      static_cast<std::uint64_t>(rows) * columns;
  return view.storage.device_data != nullptr &&
         view.storage.element_size_bytes == element_bytes &&
         view.storage.element_capacity == elements &&
         view.storage.byte_size == elements * element_bytes &&
         view.row_capacity == rows && view.columns == columns &&
         view.row_stride_elements == columns;
}

[[nodiscard]] bool complete_prompt_wide_p40_whole_core_views(
    const ReferenceLayerMajorRequestViews& views) noexcept {
  const LayerMajorP40WholeCoreViews& whole = views.p40_whole_core;
  const LayerMajorP40WholeCoreLinearPhaseViews& linear = whole.linear;
  const LayerMajorP40WholeCoreFullAttentionPhaseViews& full =
      whole.full_attention;
  return views.descriptor.profile ==
             RequestMemoryProfile::kLayerMajorP40WholeCore &&
         views.descriptor.layout ==
             LayerMajorRequestLayout::kP40WholeCorePromptWide &&
         views.descriptor.max_sequence_length ==
             kLayerMajorPrefillPromptWideP40RequestCapacityTokens &&
         views.descriptor.operator_panel_capacity_tokens ==
             kLayerMajorPrefillPromptWideP40PanelTokens &&
         views.descriptor.mlp_capacity_tokens ==
             kLayerMajorPrefillPromptWideP40Tokens &&
         views.descriptor.mlp_layout ==
             LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan &&
         views.descriptor.arena_bytes == kPromptWideP40WholeCoreArenaBytes &&
         exact_matrix_view(
             views.prompt_residual_bf16,
             kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
             kReferenceHiddenSize) &&
         exact_matrix_view(whole.prompt_token_ids_u32,
                           kLayerMajorPrefillPromptWideP40Tokens, 1U,
                           sizeof(std::uint32_t)) &&
         exact_matrix_view(linear.raw_qkv_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens, 10'240U) &&
         exact_matrix_view(linear.conv_qkv_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens, 10'240U) &&
         exact_matrix_view(linear.z_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens, 6'144U) &&
         exact_matrix_view(linear.a_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens, 48U) &&
         exact_matrix_view(linear.b_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens, 48U) &&
         linear.prompt_wide_workspace.device_data != nullptr &&
         linear.prompt_wide_workspace.byte_size ==
             kernels::kGdnPromptWideChunkGraphP40WorkspaceBytes &&
         exact_matrix_view(linear.output_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens, 6'144U) &&
         exact_matrix_view(linear.normalized_input_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens,
                           kReferenceHiddenSize) &&
         exact_matrix_view(linear.branch_output_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens,
                           kReferenceHiddenSize) &&
         exact_matrix_view(full.raw_q_gate_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens, 12'288U) &&
         exact_matrix_view(full.processed_q_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens, 6'144U) &&
         exact_matrix_view(full.packed_gate_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens, 6'144U) &&
         exact_matrix_view(full.normalized_input_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens,
                           kReferenceHiddenSize) &&
         exact_matrix_view(full.core_output_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens, 6'144U) &&
         exact_matrix_view(full.branch_output_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens,
                           kReferenceHiddenSize) &&
         exact_matrix_view(views.mlp.gate_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens,
                           kReferenceIntermediateSize) &&
         exact_matrix_view(views.mlp.normalized_input_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens,
                           kReferenceHiddenSize) &&
         exact_matrix_view(views.mlp.branch_output_bf16,
                           kLayerMajorPrefillPromptWideP40Tokens,
                           kReferenceHiddenSize) &&
         exact_matrix_view(views.final_hidden_bf16, 1U,
                           kReferenceHiddenSize) &&
         views.legacy_c512.fp32_scratch.device_data != nullptr &&
         views.legacy_c512.fp32_scratch.byte_size >=
             kernels::kSm87Fp8MarlinReductionBytes &&
         views.legacy_c512.projection_bf16[3U].storage.device_data != nullptr &&
         views.legacy_c512.projection_bf16[3U].storage.byte_size >=
             kernels::kSm87Fp8MarlinLockBytes;
}

[[nodiscard]] constexpr bool prompt_wide_p40_fp8_role(
    const PrefillBindingRole role) noexcept {
  return role == PrefillBindingRole::kLinearFp8Qkv ||
         role == PrefillBindingRole::kLinearFp8Z ||
         role == PrefillBindingRole::kLinearFp8O ||
         role == PrefillBindingRole::kFullFp8Q ||
         role == PrefillBindingRole::kFullFp8K ||
         role == PrefillBindingRole::kFullFp8V ||
         role == PrefillBindingRole::kFullFp8O;
}

[[nodiscard]] constexpr std::uint32_t
prompt_wide_p40_role_maximum_logical_m(
    const PrefillBindingRole role) noexcept {
  if (role == PrefillBindingRole::kFinalHandoff) {
    return 1U;
  }
  return prompt_wide_p40_fp8_role(role)
             ? kLayerMajorPrefillPromptWideP40PanelTokens
             : kLayerMajorPrefillPromptWideP40Tokens;
}

}  // namespace

bool prompt_wide_p40_whole_core_prefill_authority_enabled() noexcept {
  return whole_core_compile_inventory_enabled();
}

bool exchange_reference_engine_prefill_compatibility_oracle_for_test(
    const bool enabled) noexcept {
  return std::exchange(
      g_reference_engine_prefill_compatibility_oracle_for_test, enabled);
}

BoundPrefillExecutionPlan::BoundPrefillExecutionPlan(
    const ModelWeights* const weights, RequestState* const state,
    ReferenceRunner* const runner, const void* const arena_base,
    const std::uint64_t arena_bytes,
    const LayerMajorRequestMemoryPlan* const memory_plan,
    const LayerMajorPrefillArithmeticContract* const arithmetic_contract,
    const bool exact_c512_arithmetic_workspace_bound,
    const LayerMajorPrefillProjectionTactic projection_tactic,
    const LayerMajorPrefillFullAttentionTactic full_attention_tactic,
    const LayerMajorPrefillMlpScheduleTactic mlp_schedule_tactic,
    const void* const main_stream, const void* const auxiliary_stream,
    std::array<const void*, kBoundPrefillSubmissionEventCount>
        submission_events,
    std::array<NativePrefillRoleReceipt,
               kLayerMajorPrefillRequiredOperatorRoleCount>
        roles) noexcept
    : weights_(weights),
      state_(state),
      runner_(runner),
      arena_base_(arena_base),
      arena_bytes_(arena_bytes),
      memory_plan_(memory_plan),
      arithmetic_contract_(arithmetic_contract),
      exact_c512_arithmetic_workspace_bound_(
          exact_c512_arithmetic_workspace_bound),
      projection_tactic_(projection_tactic),
      full_attention_tactic_(full_attention_tactic),
      mlp_schedule_tactic_(mlp_schedule_tactic),
      main_stream_(main_stream),
      auxiliary_stream_(auxiliary_stream),
      submission_events_(std::move(submission_events)),
      roles_(std::move(roles)) {}

BoundPrefillRequestReceipt::BoundPrefillRequestReceipt(
    const BoundPrefillExecutionPlan* const plan,
    ReferenceRunner* const runner,
    const PrefillExecutionPlan& geometry) noexcept
    : plan_(plan), runner_(runner), geometry_(geometry) {}

BoundPrefillRequestReceipt::BoundPrefillRequestReceipt(
    BoundPrefillRequestReceipt&& other) noexcept
    : plan_(other.plan_),
      runner_(other.runner_),
      geometry_(other.geometry_),
      phase_(other.phase_) {
  other.plan_ = nullptr;
  other.runner_ = nullptr;
  other.geometry_ = {};
  other.phase_ = Phase::kPoisoned;
}

BoundPrefillPlanResult ReferenceEnginePrefillPlanFactory::bind(
    const ModelWeights* const weights, RequestState* const state,
    ReferenceRunner* const runner,
    const LayerMajorPrefillProjectionTactic projection_tactic,
    const LayerMajorPrefillFullAttentionTactic
        full_attention_tactic) noexcept {
  static_assert(kBoundPrefillSubmissionEventCount ==
                ReferenceRunner::kWholeRequestSubmissionWindowSlots);
  const LayerMajorPrefillArithmeticContract* const arithmetic_contract =
      projection_arithmetic_contract(projection_tactic);
  const bool layer_wide_p40_projection =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativeNvfp4PersistentP40LayerWideMlp;
  const bool whole_core_projection =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40WholeCore;
  const bool persistent_p40_projection =
      layer_wide_p40_projection || whole_core_projection;
  const bool whole_prompt_attention =
      full_attention_tactic == LayerMajorPrefillFullAttentionTactic::
                                   kNativeFlashInferExactWholePrompt;
  const LayerMajorPrefillMlpScheduleTactic mlp_schedule_tactic =
      whole_core_projection
          ? LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore
      : layer_wide_p40_projection
          ? LayerMajorPrefillMlpScheduleTactic::kLayerWideP40ExactFullM
          : LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel;
  const NvFp4MarlinGateUpLayout expected_gate_up_layout =
      persistent_p40_projection
          ? NvFp4MarlinGateUpLayout::kInterleavedGateUp
          : NvFp4MarlinGateUpLayout::kCanonicalGateThenUp;
  if (weights == nullptr || state == nullptr || runner == nullptr ||
      !static_cast<bool>(*state) || !static_cast<bool>(*runner) ||
      !is_valid_layer_major_prefill_projection_tactic(projection_tactic) ||
      !is_valid_layer_major_prefill_full_attention_tactic(
          full_attention_tactic) ||
      arithmetic_contract == nullptr ||
      !is_valid_layer_major_prefill_arithmetic_contract(
          *arithmetic_contract) ||
      whole_core_projection != whole_prompt_attention) {
    return plan_failure(BoundPrefillPlanError::kInvalidDependency,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_dependencies");
  }
  const RequestMemoryProfile expected_memory_profile =
      whole_core_projection ? RequestMemoryProfile::kLayerMajorP40WholeCore
                            : RequestMemoryProfile::kLayerMajorC8192;
  const LayerMajorRequestLayout expected_request_layout =
      whole_core_projection
          ? LayerMajorRequestLayout::kP40WholeCorePromptWide
          : LayerMajorRequestLayout::kC8192FamilyOverlay;
  if (state->memory_profile() != expected_memory_profile ||
      state->layer_major_plan() == nullptr ||
      state->layer_major_plan()->layout != expected_request_layout) {
    return plan_failure(BoundPrefillPlanError::kWrongMemoryProfile,
                        ReferenceRunnerError::kInvalidRequestState,
                        "bound_prefill_memory_profile");
  }
  if (runner->weights_ != weights || runner->state_ != state ||
      runner->projection_backend_ != ProjectionBackend::kSm87WeightOnly ||
      runner->poisoned_ || runner->whole_request_prefill_active() ||
      runner->stream_ == nullptr ||
      runner->whole_request_submission_events_[0U] == nullptr ||
      runner->whole_request_submission_events_[1U] == nullptr) {
    return plan_failure(BoundPrefillPlanError::kRunnerIdentityMismatch,
                        ReferenceRunnerError::kInvalidRunner,
                        "bound_prefill_runner_identity");
  }
  if (!runner->layer_major_request_views_.has_value() ||
      runner->layer_major_request_views_->descriptor.profile !=
          expected_memory_profile ||
      runner->layer_major_request_views_->descriptor.layout !=
          expected_request_layout ||
      runner->layer_major_request_views_->descriptor
              .operator_panel_capacity_tokens !=
          (whole_core_projection
               ? kLayerMajorPrefillPromptWideP40PanelTokens
               : kLayerMajorPrefillOperatorPanelTokens) ||
      runner->layer_major_request_views_->descriptor.mlp_capacity_tokens !=
          (persistent_p40_projection
               ? kLayerMajorPrefillLayerWideMlpP40Tokens
               : kLayerMajorPrefillOperatorPanelTokens) ||
      runner->layer_major_request_views_->descriptor.mlp_layout !=
          (persistent_p40_projection
               ? LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan
               : LayerMajorRequestMlpLayout::kPanelLocalThreeSpan) ||
      (persistent_p40_projection &&
       runner->layer_major_request_views_->descriptor.max_sequence_length !=
           kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens) ||
      (whole_core_projection &&
       runner->layer_major_request_views_->descriptor.arena_bytes !=
           kPromptWideP40WholeCoreArenaBytes) ||
      runner->layer_major_request_views_->prompt_residual_bf16.storage
              .device_data == nullptr ||
      runner->layer_major_request_views_->final_hidden_bf16.storage
              .device_data == nullptr) {
    return plan_failure(BoundPrefillPlanError::kIncompleteTypedViews,
                        ReferenceRunnerError::kInvalidRequestState,
                        "bound_prefill_typed_views");
  }
  const bool native_large_m_projection =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativeQuantizedLargeMOperatorPanel;
  [[maybe_unused]] const bool true_large_m_nvfp4_projection =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativeNvfp4TrueLargeMOperatorPanel;
  [[maybe_unused]] const bool g2_d2_nvfp4_projection =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativeNvfp4G2D2LargeMOperatorPanel;
  if (whole_core_projection && !whole_core_compile_inventory_enabled()) {
    return plan_failure(BoundPrefillPlanError::kUnsupportedBinary,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_prompt_wide_p40_whole_core_binary");
  }
#if !defined(Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
  if (true_large_m_nvfp4_projection) {
    return plan_failure(BoundPrefillPlanError::kUnsupportedBinary,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_nvfp4_true_large_m_binary");
  }
#else
  if (true_large_m_nvfp4_projection &&
      !complete_nvfp4_true_large_m_capability()) {
    return plan_failure(BoundPrefillPlanError::kUnsupportedBinary,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_nvfp4_true_large_m_coupled_capability");
  }
#endif
#if !defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION)
  if (persistent_p40_projection) {
    return plan_failure(BoundPrefillPlanError::kUnsupportedBinary,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_nvfp4_persistent_p40_binary");
  }
#else
  if (persistent_p40_projection &&
      !complete_nvfp4_persistent_p40_capability()) {
    return plan_failure(
        BoundPrefillPlanError::kUnsupportedBinary,
        ReferenceRunnerError::kInvalidDependency,
        "bound_prefill_nvfp4_persistent_p40_complete_package_capability");
  }
#endif
#if !defined(Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION)
  if (g2_d2_nvfp4_projection) {
    return plan_failure(BoundPrefillPlanError::kUnsupportedBinary,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_nvfp4_g2_d2_binary");
  }
#else
  if (g2_d2_nvfp4_projection && !complete_nvfp4_g2_d2_capability()) {
    return plan_failure(BoundPrefillPlanError::kUnsupportedBinary,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_nvfp4_g2_d2_complete_package_capability");
  }
#endif
  if (full_attention_tactic == LayerMajorPrefillFullAttentionTactic::
                                   kNativeFlashInferExactPanel &&
      !has_bulk_causal_gqa_flashinfer_exact_panel_cuda()) {
    return plan_failure(BoundPrefillPlanError::kUnsupportedBinary,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_flashinfer_exact_panel_binary");
  }
  if (whole_core_projection &&
      !complete_flashinfer_whole_prompt_p40_capability()) {
    return plan_failure(
        BoundPrefillPlanError::kUnsupportedBinary,
        ReferenceRunnerError::kInvalidDependency,
        "bound_prefill_flashinfer_exact_whole_prompt_capability");
  }
  if (whole_core_projection &&
      !complete_bf16_ab_prompt_wide_p40_capability()) {
    return plan_failure(
        BoundPrefillPlanError::kUnsupportedBinary,
        ReferenceRunnerError::kInvalidDependency,
        "bound_prefill_bf16_ab_prompt_wide_p40_capability");
  }
  if (whole_core_projection &&
      !complete_gdn_prompt_wide_p40_capability()) {
    return plan_failure(
        BoundPrefillPlanError::kUnsupportedBinary,
        ReferenceRunnerError::kInvalidDependency,
        "bound_prefill_gdn_prompt_wide_p40_capability");
  }
  if (!complete_installed_projection_inventory(*weights,
                                                expected_gate_up_layout)) {
    return plan_failure(BoundPrefillPlanError::kUnsupportedBinary,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_installed_binary");
  }

  const ReferenceLayerMajorRequestViews& views =
      *runner->layer_major_request_views_;
  bool exact_c512_arithmetic_workspace_bound =
      views.legacy_c512.linear_a_bf16.storage.device_data != nullptr &&
      views.legacy_c512.linear_b_bf16.storage.device_data != nullptr &&
      views.legacy_c512.fp32_scratch.device_data != nullptr;
  for (const DeviceMatrixView& hidden : views.legacy_c512.hidden_bf16) {
    exact_c512_arithmetic_workspace_bound =
        exact_c512_arithmetic_workspace_bound &&
        hidden.storage.device_data != nullptr;
  }
  for (const DeviceMatrixView& projection :
       views.legacy_c512.projection_bf16) {
    exact_c512_arithmetic_workspace_bound =
        exact_c512_arithmetic_workspace_bound &&
        projection.storage.device_data != nullptr;
  }
  const DeviceMatrixView& oracle_gate =
      views.legacy_c512.projection_bf16[0];
  const DeviceMatrixView& oracle_up =
      views.legacy_c512.projection_bf16[1];
  const DeviceMatrixView& oracle_locks =
      views.legacy_c512.projection_bf16[3];
  exact_c512_arithmetic_workspace_bound =
      exact_c512_arithmetic_workspace_bound &&
      views.legacy_c512.fp32_scratch.byte_size >=
          kernels::kSm87NvFp4MarlinReductionBytes &&
      oracle_locks.storage.byte_size >=
          kernels::kSm87NvFp4MarlinLockBytes &&
      reinterpret_cast<std::uintptr_t>(oracle_up.storage.device_data) ==
          reinterpret_cast<std::uintptr_t>(oracle_gate.storage.device_data) +
              oracle_gate.storage.byte_size;
  if (!exact_c512_arithmetic_workspace_bound) {
    return plan_failure(BoundPrefillPlanError::kIncompleteTypedViews,
                        ReferenceRunnerError::kInvalidRequestState,
                        "bound_prefill_exact_c512_arithmetic_views");
  }
  if (whole_core_projection) {
    if (!complete_prompt_wide_p40_whole_core_views(views)) {
      return plan_failure(BoundPrefillPlanError::kIncompleteTypedViews,
                          ReferenceRunnerError::kInvalidRequestState,
                          "bound_prefill_prompt_wide_p40_whole_core_views");
    }

    const DecoderLayerWeights& linear_layer = weights->layer(0U);
    const DecoderLayerWeights& full_layer = weights->layer(3U);
    const auto* const linear =
        std::get_if<LinearAttentionWeights>(&linear_layer.attention);
    const auto* const attention =
        std::get_if<FullAttentionWeights>(&full_layer.attention);
    const auto* const gate =
        std::get_if<NvFp4LinearWeight>(&linear_layer.mlp.gate_proj);
    const auto* const up =
        std::get_if<NvFp4LinearWeight>(&linear_layer.mlp.up_proj);
    const auto* const down =
        std::get_if<NvFp4LinearWeight>(&linear_layer.mlp.down_proj);
    const auto* const linear_qkv =
        linear == nullptr
            ? nullptr
            : std::get_if<Fp8LinearWeight>(&linear->in_proj_qkv);
    const auto* const linear_z =
        linear == nullptr ? nullptr
                          : std::get_if<Fp8LinearWeight>(&linear->in_proj_z);
    const auto* const linear_o =
        linear == nullptr ? nullptr
                          : std::get_if<Fp8LinearWeight>(&linear->out_proj);
    const auto* const linear_a =
        linear == nullptr ? nullptr
                          : std::get_if<Bf16LinearWeight>(&linear->in_proj_a);
    const auto* const linear_b =
        linear == nullptr ? nullptr
                          : std::get_if<Bf16LinearWeight>(&linear->in_proj_b);
    const auto* const full_q =
        attention == nullptr ? nullptr
                             : std::get_if<Fp8LinearWeight>(&attention->q_proj);
    const auto* const full_k =
        attention == nullptr ? nullptr
                             : std::get_if<Fp8LinearWeight>(&attention->k_proj);
    const auto* const full_v =
        attention == nullptr ? nullptr
                             : std::get_if<Fp8LinearWeight>(&attention->v_proj);
    const auto* const full_o =
        attention == nullptr ? nullptr
                             : std::get_if<Fp8LinearWeight>(&attention->o_proj);
    if (linear == nullptr || attention == nullptr || gate == nullptr ||
        up == nullptr || down == nullptr ||
        gate->prefill_marlin_gate_up_layout != expected_gate_up_layout ||
        up->prefill_marlin_gate_up_layout !=
            gate->prefill_marlin_gate_up_layout ||
        linear_qkv == nullptr || linear_z == nullptr || linear_o == nullptr ||
        linear_a == nullptr || linear_b == nullptr || full_q == nullptr ||
        full_k == nullptr || full_v == nullptr || full_o == nullptr ||
        !complete_exact_gdn_chunk64_native_inventory(*weights)) {
      return plan_failure(BoundPrefillPlanError::kIncompleteNativeRole,
                          ReferenceRunnerError::kInvalidModelWeights,
                          "bound_prefill_prompt_wide_p40_role_inventory");
    }

    const auto receipt = [](
        const PrefillBindingRole role, const NativePrefillTactic tactic,
        const void* const artifact, const void* const workspace,
        const std::uint64_t workspace_bytes,
        const std::uint32_t minimum_physical_m,
        const std::uint32_t maximum_physical_m,
        const void* const auxiliary_workspace = nullptr,
        const std::uint64_t auxiliary_workspace_bytes = 0U,
        const std::uint32_t maximum_logical_m =
            kLayerMajorPrefillPromptWideP40Tokens) noexcept {
      return NativePrefillRoleReceipt{
          role,
          tactic,
          NativePrefillCompletionDomain::kMainStreamBarrier,
          artifact,
          workspace,
          workspace_bytes,
          auxiliary_workspace,
          auxiliary_workspace_bytes,
          maximum_logical_m,
          minimum_physical_m,
          maximum_physical_m};
    };
    std::array<NativePrefillRoleReceipt,
               kLayerMajorPrefillRequiredOperatorRoleCount>
        roles{};
    constexpr std::uint64_t kGateUpScaleBytes =
        static_cast<std::uint64_t>(kReferenceHiddenSize) *
        (2U * kReferenceIntermediateSize) / 16U;
    constexpr std::uint64_t kDownScaleBytes =
        static_cast<std::uint64_t>(kReferenceIntermediateSize) *
        kReferenceHiddenSize / 16U;
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4GateUp)] =
        receipt(
            PrefillBindingRole::kNvfp4GateUp,
            NativePrefillTactic::kNvfp4GateUpPersistentP40LayerWide,
            gate->prefill_marlin_weight, gate->prefill_marlin_scales,
            kGateUpScaleBytes, kLayerMajorPrefillPromptWideP40Tokens,
            kLayerMajorPrefillPromptWideP40Tokens,
            gate->prefill_marlin_global_scale, sizeof(float));
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4Down)] =
        receipt(
            PrefillBindingRole::kNvfp4Down,
            NativePrefillTactic::kNvfp4DownResidualPersistentP40LayerWide,
            down->prefill_marlin_weight, down->prefill_marlin_scales,
            kDownScaleBytes, kLayerMajorPrefillPromptWideP40Tokens,
            kLayerMajorPrefillPromptWideP40Tokens,
            down->prefill_marlin_global_scale, sizeof(float));

    const DeviceBufferView& fp8_reduction =
        views.legacy_c512.fp32_scratch;
    const DeviceBufferView& fp8_locks =
        views.legacy_c512.projection_bf16[3U].storage;
    const auto fp8_receipt = [&receipt, &fp8_reduction, &fp8_locks](
                                const PrefillBindingRole role,
                                const Fp8LinearWeight& weight) noexcept {
      return receipt(
          role, NativePrefillTactic::kFp8PromptWideP40FillDrain,
          weight.prefill_marlin_weight, fp8_reduction.device_data,
          kernels::kSm87Fp8MarlinReductionBytes,
          kLayerMajorPrefillPromptWideP40PanelTokens,
          kLayerMajorPrefillPromptWideP40PanelTokens,
          fp8_locks.device_data, kernels::kSm87Fp8MarlinLockBytes,
          kLayerMajorPrefillPromptWideP40PanelTokens);
    };
    roles[static_cast<std::size_t>(PrefillBindingRole::kLinearFp8Qkv)] =
        fp8_receipt(PrefillBindingRole::kLinearFp8Qkv, *linear_qkv);
    roles[static_cast<std::size_t>(PrefillBindingRole::kLinearFp8Z)] =
        fp8_receipt(PrefillBindingRole::kLinearFp8Z, *linear_z);
    roles[static_cast<std::size_t>(PrefillBindingRole::kLinearFp8O)] =
        fp8_receipt(PrefillBindingRole::kLinearFp8O, *linear_o);
    roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8Q)] =
        fp8_receipt(PrefillBindingRole::kFullFp8Q, *full_q);
    roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8K)] =
        fp8_receipt(PrefillBindingRole::kFullFp8K, *full_k);
    roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8V)] =
        fp8_receipt(PrefillBindingRole::kFullFp8V, *full_v);
    roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8O)] =
        fp8_receipt(PrefillBindingRole::kFullFp8O, *full_o);

    const LayerMajorP40WholeCoreViews& whole = views.p40_whole_core;
    roles[static_cast<std::size_t>(PrefillBindingRole::kLinearBf16A)] =
        receipt(PrefillBindingRole::kLinearBf16A,
                NativePrefillTactic::kBf16AbPromptWideP40,
                linear_a->weight,
                whole.linear.a_bf16.storage.device_data,
                whole.linear.a_bf16.storage.byte_size,
                kLayerMajorPrefillPromptWideP40Tokens,
                kLayerMajorPrefillPromptWideP40Tokens);
    roles[static_cast<std::size_t>(PrefillBindingRole::kLinearBf16B)] =
        receipt(PrefillBindingRole::kLinearBf16B,
                NativePrefillTactic::kBf16AbPromptWideP40,
                linear_b->weight,
                whole.linear.b_bf16.storage.device_data,
                whole.linear.b_bf16.storage.byte_size,
                kLayerMajorPrefillPromptWideP40Tokens,
                kLayerMajorPrefillPromptWideP40Tokens);
    roles[static_cast<std::size_t>(PrefillBindingRole::kExactGdn)] =
        receipt(PrefillBindingRole::kExactGdn,
                NativePrefillTactic::kExactGdnPromptWideP40ChunkGraph,
                linear->conv1d.data,
                whole.linear.prompt_wide_workspace.device_data,
                kernels::kGdnPromptWideChunkGraphP40WorkspaceBytes,
                kLayerMajorPrefillPromptWideP40Tokens,
                kLayerMajorPrefillPromptWideP40Tokens);
    roles[static_cast<std::size_t>(
        PrefillBindingRole::kExactCausalAttention)] =
        receipt(
            PrefillBindingRole::kExactCausalAttention,
            NativePrefillTactic::kNativeCausalAttentionFlashInferExactWholePrompt,
            attention->q_norm.data,
            whole.full_attention.core_output_bf16.storage.device_data,
            whole.full_attention.core_output_bf16.storage.byte_size,
            kLayerMajorPrefillPromptWideP40Tokens,
            kLayerMajorPrefillPromptWideP40Tokens);
    roles[static_cast<std::size_t>(PrefillBindingRole::kResidual)] =
        receipt(PrefillBindingRole::kResidual,
                NativePrefillTactic::kResidualOperatorPanel, weights,
                views.prompt_residual_bf16.storage.device_data,
                views.prompt_residual_bf16.storage.byte_size,
                kLayerMajorPrefillPromptWideP40PanelTokens,
                kLayerMajorPrefillPromptWideP40PanelTokens);
    roles[static_cast<std::size_t>(PrefillBindingRole::kNormalization)] =
        receipt(PrefillBindingRole::kNormalization,
                NativePrefillTactic::kNormalizationOperatorPanel,
                linear_layer.post_attention_layernorm.data,
                whole.linear.normalized_input_bf16.storage.device_data,
                whole.linear.normalized_input_bf16.storage.byte_size,
                kLayerMajorPrefillPromptWideP40PanelTokens,
                kLayerMajorPrefillPromptWideP40Tokens);
    roles[static_cast<std::size_t>(PrefillBindingRole::kEmbedding)] =
        receipt(PrefillBindingRole::kEmbedding,
                NativePrefillTactic::kEmbeddingOperatorPanel,
                weights->embed_tokens().weight,
                whole.prompt_token_ids_u32.storage.device_data,
                whole.prompt_token_ids_u32.storage.byte_size,
                kLayerMajorPrefillPromptWideP40PanelTokens,
                kLayerMajorPrefillPromptWideP40PanelTokens);
    roles[static_cast<std::size_t>(PrefillBindingRole::kFinalHandoff)] =
        receipt(PrefillBindingRole::kFinalHandoff,
                NativePrefillTactic::kFinalHandoff,
                weights->final_norm().data,
                views.final_hidden_bf16.storage.device_data,
                views.final_hidden_bf16.storage.byte_size, 1U, 1U, nullptr,
                0U, 1U);

    for (std::size_t index = 0U; index < roles.size(); ++index) {
      const PrefillBindingRole role_identity =
          static_cast<PrefillBindingRole>(index);
      const NativePrefillRoleReceipt& role = roles[index];
      const bool workspace_valid =
          role.workspace_owner != nullptr && role.workspace_bytes != 0U &&
          ((role.auxiliary_workspace_owner == nullptr) ==
           (role.auxiliary_workspace_bytes == 0U));
      if (role.role != role_identity || role.artifact_owner == nullptr ||
          !workspace_valid ||
          role.maximum_logical_panel_m !=
              prompt_wide_p40_role_maximum_logical_m(role_identity) ||
          role.minimum_physical_m == 0U ||
          role.minimum_physical_m > role.maximum_physical_m ||
          role.maximum_physical_m > role.maximum_logical_panel_m ||
          role.completion !=
              NativePrefillCompletionDomain::kMainStreamBarrier) {
        return plan_failure(BoundPrefillPlanError::kIncompleteNativeRole,
                            ReferenceRunnerError::kInvalidDependency,
                            "bound_prefill_prompt_wide_p40_native_role");
      }
    }

    const std::array<const void*, kBoundPrefillSubmissionEventCount>
        submission_events{
            runner->whole_request_submission_events_[0U],
            runner->whole_request_submission_events_[1U]};
    auto* const allocation = new (std::nothrow) BoundPrefillExecutionPlan(
        weights, state, runner, state->arena_data(), state->arena_bytes(),
        state->layer_major_plan(), arithmetic_contract,
        exact_c512_arithmetic_workspace_bound, projection_tactic,
        full_attention_tactic, mlp_schedule_tactic, runner->stream_,
        runner->prefill_auxiliary_stream_, submission_events,
        std::move(roles));
    if (allocation == nullptr) {
      return plan_failure(BoundPrefillPlanError::kInvalidDependency,
                          ReferenceRunnerError::kAllocationFailure,
                          "bound_prefill_prompt_wide_p40_plan_allocation");
    }
    BoundPrefillPlanResult result;
    result.value.reset(allocation);
    return result;
  }
  const bool segmented_projection =
      projection_tactic ==
      LayerMajorPrefillProjectionTactic::kSegmentedMarlinOperatorPanel;
  constexpr std::size_t kFp8MarlinTemporaryBytes =
      kernels::kSm87Fp8MarlinReductionBytes +
      kernels::kSm87Fp8MarlinLockBytes;
  constexpr std::size_t kNvFp4MarlinTemporaryBytes =
      kernels::kSm87NvFp4MarlinReductionBytes +
      kernels::kSm87NvFp4MarlinLockBytes;
  const auto valid_fp8_temporary = [](const DeviceBufferView& view) noexcept {
    return view.device_data != nullptr &&
           view.byte_size >= kFp8MarlinTemporaryBytes;
  };
  const auto valid_nvfp4_temporary = [](const DeviceBufferView& view) noexcept {
    return view.device_data != nullptr &&
           view.byte_size >= kNvFp4MarlinTemporaryBytes;
  };
  const LayerMajorMlpPhaseViews& mlp_views = views.mlp;
  const bool operator_panel_shared_outputs_complete =
      mlp_views.normalized_input_bf16.storage.device_data != nullptr &&
      views.gdn.qkv_bf16.storage.device_data != nullptr &&
      views.gdn.z_bf16.storage.device_data != nullptr &&
      views.gdn.branch_output_bf16.storage.device_data != nullptr &&
      views.attention.raw_q_gate_bf16.storage.device_data != nullptr &&
      views.attention.branch_output_bf16.storage.device_data != nullptr;
  const bool legacy_operator_panel_mlp_outputs_complete =
      operator_panel_shared_outputs_complete &&
      mlp_views.gate_bf16.storage.device_data != nullptr &&
      mlp_views.up_bf16.storage.device_data != nullptr &&
      mlp_views.activated_bf16.storage.device_data != nullptr &&
      mlp_views.branch_output_bf16.storage.device_data != nullptr &&
      reinterpret_cast<std::uintptr_t>(
          mlp_views.up_bf16.storage.device_data) ==
          reinterpret_cast<std::uintptr_t>(
              mlp_views.gate_bf16.storage.device_data) +
              mlp_views.gate_bf16.storage.byte_size;
  const bool segmented_projection_views_complete =
      legacy_operator_panel_mlp_outputs_complete &&
      valid_nvfp4_temporary(mlp_views.gate_up_projection_temporary) &&
      valid_nvfp4_temporary(mlp_views.down_projection_temporary) &&
      valid_fp8_temporary(views.gdn.input_projection_temporary) &&
      valid_fp8_temporary(views.gdn.output_projection_temporary) &&
      valid_fp8_temporary(views.attention.input_projection_temporary) &&
      valid_fp8_temporary(views.attention.output_projection_temporary);
  const bool true_large_m_projection_views_complete =
      legacy_operator_panel_mlp_outputs_complete &&
      valid_fp8_temporary(views.gdn.input_projection_temporary) &&
      valid_fp8_temporary(views.gdn.output_projection_temporary) &&
      valid_fp8_temporary(views.attention.input_projection_temporary) &&
      valid_fp8_temporary(views.attention.output_projection_temporary);
  const bool g2_d2_projection_views_complete =
      operator_panel_shared_outputs_complete &&
      mlp_views.gate_bf16.storage.device_data != nullptr &&
      mlp_views.gate_bf16.storage.device_data !=
          mlp_views.normalized_input_bf16.storage.device_data &&
      valid_fp8_temporary(views.gdn.input_projection_temporary) &&
      valid_fp8_temporary(views.gdn.output_projection_temporary) &&
      valid_fp8_temporary(views.attention.input_projection_temporary) &&
      valid_fp8_temporary(views.attention.output_projection_temporary);
  const bool persistent_p40_projection_views_complete =
      operator_panel_shared_outputs_complete &&
      views.descriptor.mlp_layout ==
          LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan &&
      views.descriptor.mlp_capacity_tokens ==
          kLayerMajorPrefillLayerWideMlpP40Tokens &&
      mlp_views.normalized_input_bf16.row_capacity ==
          kLayerMajorPrefillLayerWideMlpP40Tokens &&
      mlp_views.normalized_input_bf16.columns == kReferenceHiddenSize &&
      mlp_views.normalized_input_bf16.row_stride_elements ==
          kReferenceHiddenSize &&
      mlp_views.gate_bf16.row_capacity ==
          kLayerMajorPrefillLayerWideMlpP40Tokens &&
      mlp_views.gate_bf16.columns == kReferenceIntermediateSize &&
      mlp_views.gate_bf16.row_stride_elements ==
          kReferenceIntermediateSize &&
      mlp_views.gate_bf16.storage.device_data != nullptr &&
      mlp_views.normalized_input_bf16.storage.device_data != nullptr &&
      mlp_views.gate_bf16.storage.device_data !=
          mlp_views.normalized_input_bf16.storage.device_data;
  if ((segmented_projection || native_large_m_projection) &&
      !segmented_projection_views_complete) {
    return plan_failure(BoundPrefillPlanError::kIncompleteTypedViews,
                        ReferenceRunnerError::kInvalidRequestState,
                        "bound_prefill_marlin_operator_panel_views");
  }
  if (true_large_m_nvfp4_projection &&
      !true_large_m_projection_views_complete) {
    return plan_failure(BoundPrefillPlanError::kIncompleteTypedViews,
                        ReferenceRunnerError::kInvalidRequestState,
                        "bound_prefill_nvfp4_true_large_m_panel_views");
  }
  if (g2_d2_nvfp4_projection && !g2_d2_projection_views_complete) {
    return plan_failure(BoundPrefillPlanError::kIncompleteTypedViews,
                        ReferenceRunnerError::kInvalidRequestState,
                        "bound_prefill_nvfp4_g2_d2_fused_panel_views");
  }
  if (persistent_p40_projection &&
      !persistent_p40_projection_views_complete) {
    return plan_failure(BoundPrefillPlanError::kIncompleteTypedViews,
                        ReferenceRunnerError::kInvalidRequestState,
                        "bound_prefill_nvfp4_persistent_p40_full_m_views");
  }
  const DecoderLayerWeights& linear_layer = weights->layer(0U);
  const DecoderLayerWeights& full_layer = weights->layer(3U);
  const auto* const linear =
      std::get_if<LinearAttentionWeights>(&linear_layer.attention);
  const auto* const attention =
      std::get_if<FullAttentionWeights>(&full_layer.attention);
  const auto* const gate =
      std::get_if<NvFp4LinearWeight>(&linear_layer.mlp.gate_proj);
  const auto* const up =
      std::get_if<NvFp4LinearWeight>(&linear_layer.mlp.up_proj);
  const auto* const down =
      std::get_if<NvFp4LinearWeight>(&linear_layer.mlp.down_proj);
  const auto* const linear_qkv =
      linear == nullptr
          ? nullptr
          : std::get_if<Fp8LinearWeight>(&linear->in_proj_qkv);
  const auto* const linear_z =
      linear == nullptr ? nullptr
                        : std::get_if<Fp8LinearWeight>(&linear->in_proj_z);
  const auto* const linear_o =
      linear == nullptr ? nullptr
                        : std::get_if<Fp8LinearWeight>(&linear->out_proj);
  const auto* const linear_a =
      linear == nullptr ? nullptr
                        : std::get_if<Bf16LinearWeight>(&linear->in_proj_a);
  const auto* const linear_b =
      linear == nullptr ? nullptr
                        : std::get_if<Bf16LinearWeight>(&linear->in_proj_b);
  const auto* const full_q =
      attention == nullptr ? nullptr
                           : std::get_if<Fp8LinearWeight>(&attention->q_proj);
  const auto* const full_k =
      attention == nullptr ? nullptr
                           : std::get_if<Fp8LinearWeight>(&attention->k_proj);
  const auto* const full_v =
      attention == nullptr ? nullptr
                           : std::get_if<Fp8LinearWeight>(&attention->v_proj);
  const auto* const full_o =
      attention == nullptr ? nullptr
                           : std::get_if<Fp8LinearWeight>(&attention->o_proj);
  if (linear == nullptr || attention == nullptr || gate == nullptr ||
      up == nullptr || down == nullptr ||
      gate->prefill_marlin_gate_up_layout != expected_gate_up_layout ||
      up->prefill_marlin_gate_up_layout !=
          gate->prefill_marlin_gate_up_layout ||
      linear_qkv == nullptr || linear_z == nullptr ||
      linear_o == nullptr || linear_a == nullptr || linear_b == nullptr ||
      full_q == nullptr || full_k == nullptr || full_v == nullptr ||
      full_o == nullptr) {
    return plan_failure(BoundPrefillPlanError::kIncompleteNativeRole,
                        ReferenceRunnerError::kInvalidModelWeights,
                        "bound_prefill_role_inventory");
  }
  std::size_t exact_gdn_workspace_bytes = 0U;
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  exact_gdn_workspace_bytes =
      gdn_prefill_chunk64_native_detail::workspace_bytes();
  if (!complete_exact_gdn_chunk64_native_inventory(*weights)) {
    return plan_failure(BoundPrefillPlanError::kIncompleteNativeRole,
                        ReferenceRunnerError::kInvalidModelWeights,
                        "bound_prefill_exact_gdn_inventory");
  }
  if (views.gdn.native_c64_workspace.device_data == nullptr ||
      exact_gdn_workspace_bytes == 0U ||
      views.gdn.native_c64_workspace.byte_size !=
          exact_gdn_workspace_bytes) {
    return plan_failure(BoundPrefillPlanError::kIncompleteNativeRole,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_exact_gdn_workspace");
  }
#else
  return plan_failure(BoundPrefillPlanError::kUnsupportedBinary,
                      ReferenceRunnerError::kInvalidDependency,
                      "bound_prefill_exact_gdn_binary");
#endif

  const auto receipt = [](
      const PrefillBindingRole role, const NativePrefillTactic tactic,
      const void* const artifact, const void* const workspace,
      const std::uint64_t workspace_bytes,
      const std::uint32_t minimum_physical_m,
      const std::uint32_t maximum_physical_m,
      const void* const auxiliary_workspace = nullptr,
      const std::uint64_t auxiliary_workspace_bytes = 0U,
      const std::uint32_t maximum_logical_m =
          kLayerMajorPrefillOperatorPanelTokens) noexcept {
    return NativePrefillRoleReceipt{
        role, tactic, NativePrefillCompletionDomain::kMainStreamBarrier,
        artifact, workspace, workspace_bytes, auxiliary_workspace,
        auxiliary_workspace_bytes,
        maximum_logical_m, minimum_physical_m,
        maximum_physical_m};
  };
  std::array<NativePrefillRoleReceipt,
             kLayerMajorPrefillRequiredOperatorRoleCount>
      roles{};
  const auto fp8_locks = [](const DeviceBufferView& view) noexcept {
    return static_cast<void*>(static_cast<std::uint8_t*>(view.device_data) +
                              kernels::kSm87Fp8MarlinReductionBytes);
  };
  const auto nvfp4_locks = [](const DeviceBufferView& view) noexcept {
    return static_cast<void*>(static_cast<std::uint8_t*>(view.device_data) +
                              kernels::kSm87NvFp4MarlinReductionBytes);
  };
  if (persistent_p40_projection) {
#if defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION)
    constexpr std::uint64_t kGateUpScaleBytes =
        static_cast<std::uint64_t>(kReferenceHiddenSize) *
        (2U * kReferenceIntermediateSize) / 16U;
    constexpr std::uint64_t kDownScaleBytes =
        static_cast<std::uint64_t>(kReferenceIntermediateSize) *
        kReferenceHiddenSize / 16U;
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4GateUp)] =
        receipt(
            PrefillBindingRole::kNvfp4GateUp,
            NativePrefillTactic::kNvfp4GateUpPersistentP40LayerWide,
            gate->prefill_marlin_weight, gate->prefill_marlin_scales,
            kGateUpScaleBytes, kLayerMajorPrefillLayerWideMlpP40Tokens,
            kLayerMajorPrefillLayerWideMlpP40Tokens,
            gate->prefill_marlin_global_scale, sizeof(float),
            kLayerMajorPrefillLayerWideMlpP40Tokens);
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4Down)] =
        receipt(
            PrefillBindingRole::kNvfp4Down,
            NativePrefillTactic::
                kNvfp4DownResidualPersistentP40LayerWide,
            down->prefill_marlin_weight, down->prefill_marlin_scales,
            kDownScaleBytes, kLayerMajorPrefillLayerWideMlpP40Tokens,
            kLayerMajorPrefillLayerWideMlpP40Tokens,
            down->prefill_marlin_global_scale, sizeof(float),
            kLayerMajorPrefillLayerWideMlpP40Tokens);
#endif
  } else if (g2_d2_nvfp4_projection) {
#if defined(Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION)
    constexpr std::uint64_t kGateUpScaleBytes =
        static_cast<std::uint64_t>(kReferenceHiddenSize) *
        (2U * kReferenceIntermediateSize) / 16U;
    constexpr std::uint64_t kDownScaleBytes =
        static_cast<std::uint64_t>(kReferenceIntermediateSize) *
        kReferenceHiddenSize / 16U;
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4GateUp)] =
        receipt(PrefillBindingRole::kNvfp4GateUp,
                NativePrefillTactic::kNvfp4GateUpG2LargeMOperatorPanel,
                gate->prefill_marlin_weight, gate->prefill_marlin_scales,
                kGateUpScaleBytes,
                kLayerMajorPrefillTrueLargeMPartialPanelTokens,
                kLayerMajorPrefillOperatorPanelTokens,
                gate->prefill_marlin_global_scale, sizeof(float));
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4Down)] =
        receipt(PrefillBindingRole::kNvfp4Down,
                NativePrefillTactic::kNvfp4DownD2LargeMOperatorPanel,
                down->prefill_marlin_weight, down->prefill_marlin_scales,
                kDownScaleBytes,
                kLayerMajorPrefillTrueLargeMPartialPanelTokens,
                kLayerMajorPrefillOperatorPanelTokens,
                down->prefill_marlin_global_scale, sizeof(float));
#endif
  } else if (true_large_m_nvfp4_projection) {
#if defined(Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
    constexpr std::uint64_t kGateUpScaleBytes =
        kernels::kSm87NvFp4PrefillLargeMHidden *
        kernels::kSm87NvFp4PrefillLargeMGateUpOutput / 16U;
    constexpr std::uint64_t kDownScaleBytes =
        kernels::kSm87NvFp4PrefillLargeMIntermediate *
        kernels::kSm87NvFp4PrefillLargeMHidden / 16U;
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4GateUp)] =
        receipt(PrefillBindingRole::kNvfp4GateUp,
                NativePrefillTactic::kNvfp4GateUpTrueLargeMOperatorPanel,
                gate->prefill_marlin_weight, gate->prefill_marlin_scales,
                kGateUpScaleBytes,
                kLayerMajorPrefillTrueLargeMPartialPanelTokens,
                kLayerMajorPrefillOperatorPanelTokens,
                gate->prefill_marlin_global_scale, sizeof(float));
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4Down)] =
        receipt(PrefillBindingRole::kNvfp4Down,
                NativePrefillTactic::kNvfp4DownTrueLargeMOperatorPanel,
                down->prefill_marlin_weight, down->prefill_marlin_scales,
                kDownScaleBytes,
                kLayerMajorPrefillTrueLargeMPartialPanelTokens,
                kLayerMajorPrefillOperatorPanelTokens,
                down->prefill_marlin_global_scale, sizeof(float));
#endif
  } else if (native_large_m_projection) {
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4GateUp)] =
        receipt(
            PrefillBindingRole::kNvfp4GateUp,
            NativePrefillTactic::
                kNvfp4GateUpNativeQuantizedLargeMOperatorPanel,
            gate->prefill_marlin_weight,
            mlp_views.gate_up_projection_temporary.device_data,
            kernels::kSm87NvFp4MarlinReductionBytes, 1U,
            kLayerMajorPrefillOperatorPanelTokens,
            nvfp4_locks(mlp_views.gate_up_projection_temporary),
            kernels::kSm87NvFp4MarlinLockBytes);
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4Down)] =
        receipt(PrefillBindingRole::kNvfp4Down,
                NativePrefillTactic::
                    kNvfp4DownNativeQuantizedLargeMOperatorPanel,
                down->prefill_marlin_weight,
                mlp_views.down_projection_temporary.device_data,
                kernels::kSm87NvFp4MarlinReductionBytes, 1U,
                kLayerMajorPrefillOperatorPanelTokens,
                nvfp4_locks(mlp_views.down_projection_temporary),
                kernels::kSm87NvFp4MarlinLockBytes);
  } else if (segmented_projection) {
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4GateUp)] =
        receipt(PrefillBindingRole::kNvfp4GateUp,
                NativePrefillTactic::kNvfp4GateUpSegmentedMarlinOperatorPanel,
                gate->prefill_marlin_weight,
                mlp_views.gate_up_projection_temporary.device_data,
                kernels::kSm87NvFp4MarlinReductionBytes, 1U,
                kernels::kSm87NvFp4MarlinMaximumKernelSegmentTokens,
                nvfp4_locks(mlp_views.gate_up_projection_temporary),
                kernels::kSm87NvFp4MarlinLockBytes);
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4Down)] =
        receipt(PrefillBindingRole::kNvfp4Down,
                NativePrefillTactic::kNvfp4DownSegmentedMarlinOperatorPanel,
                down->prefill_marlin_weight,
                mlp_views.down_projection_temporary.device_data,
                kernels::kSm87NvFp4MarlinReductionBytes, 1U,
                kernels::kSm87NvFp4MarlinMaximumKernelSegmentTokens,
                nvfp4_locks(mlp_views.down_projection_temporary),
                kernels::kSm87NvFp4MarlinLockBytes);
  } else {
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4GateUp)] =
        receipt(PrefillBindingRole::kNvfp4GateUp,
                NativePrefillTactic::kNvfp4GateUpOracleSpanC512,
                gate->prefill_marlin_weight,
                views.legacy_c512.fp32_scratch.device_data,
                kernels::kSm87NvFp4MarlinReductionBytes, 1U,
                kPrefillPhysicalSegmentMaximumTokens,
                views.legacy_c512.projection_bf16[3].storage.device_data,
                kernels::kSm87NvFp4MarlinLockBytes);
    roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4Down)] =
        receipt(PrefillBindingRole::kNvfp4Down,
                NativePrefillTactic::kNvfp4DownOracleSpanC512,
                down->prefill_marlin_weight,
                views.legacy_c512.fp32_scratch.device_data,
                kernels::kSm87NvFp4MarlinReductionBytes, 1U,
                kPrefillPhysicalSegmentMaximumTokens,
                views.legacy_c512.projection_bf16[3].storage.device_data,
                kernels::kSm87NvFp4MarlinLockBytes);
  }
  const auto fp8_receipt = [&receipt, &fp8_locks, segmented_projection,
                            native_large_m_projection,
                            true_large_m_nvfp4_projection,
                            g2_d2_nvfp4_projection,
                            persistent_p40_projection](
                               const PrefillBindingRole role,
                               const Fp8LinearWeight& weight,
                               const DeviceBufferView& temporary) noexcept {
    return (true_large_m_nvfp4_projection || g2_d2_nvfp4_projection ||
            persistent_p40_projection)
               ? receipt(role,
                         NativePrefillTactic::
                             kFp8Nvfp4TrueLargeMRouteCompanion,
                         weight.prefill_marlin_weight,
                         temporary.device_data,
                         kernels::kSm87Fp8MarlinReductionBytes, 2U,
                         kLayerMajorPrefillOperatorPanelTokens,
                         fp8_locks(temporary),
                         kernels::kSm87Fp8MarlinLockBytes)
           : native_large_m_projection
               ? receipt(role,
                         NativePrefillTactic::
                             kFp8NativeQuantizedLargeMOperatorPanel,
                         weight.prefill_marlin_weight,
                         temporary.device_data,
                         kernels::kSm87Fp8MarlinReductionBytes, 2U,
                         kLayerMajorPrefillOperatorPanelTokens,
                         fp8_locks(temporary),
                         kernels::kSm87Fp8MarlinLockBytes)
           : segmented_projection
               ? receipt(
                     role, NativePrefillTactic::kFp8SegmentedMarlinOperatorPanel,
                     weight.prefill_marlin_weight, temporary.device_data,
                     kernels::kSm87Fp8MarlinReductionBytes, 2U,
                     static_cast<std::uint32_t>(
                         kernels::sm87_fp8_marlin_maximum_kernel_segment_tokens(
                             weight.output_size)),
                     fp8_locks(temporary),
                     kernels::kSm87Fp8MarlinLockBytes)
               : receipt(role, NativePrefillTactic::kFp8OracleSpanC512,
                         weight.prefill_marlin_weight, temporary.device_data,
                         temporary.byte_size, 2U,
                         kPrefillPhysicalSegmentMaximumTokens);
  };
  roles[static_cast<std::size_t>(PrefillBindingRole::kLinearFp8Qkv)] =
      fp8_receipt(PrefillBindingRole::kLinearFp8Qkv, *linear_qkv,
                  views.gdn.input_projection_temporary);
  roles[static_cast<std::size_t>(PrefillBindingRole::kLinearFp8Z)] =
      fp8_receipt(PrefillBindingRole::kLinearFp8Z, *linear_z,
                  views.gdn.input_projection_temporary);
  roles[static_cast<std::size_t>(PrefillBindingRole::kLinearFp8O)] =
      fp8_receipt(PrefillBindingRole::kLinearFp8O, *linear_o,
                  views.gdn.output_projection_temporary);
  roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8Q)] =
      fp8_receipt(PrefillBindingRole::kFullFp8Q, *full_q,
                  views.attention.input_projection_temporary);
  roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8K)] =
      fp8_receipt(PrefillBindingRole::kFullFp8K, *full_k,
                  views.attention.input_projection_temporary);
  roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8V)] =
      fp8_receipt(PrefillBindingRole::kFullFp8V, *full_v,
                  views.attention.input_projection_temporary);
  roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8O)] =
      fp8_receipt(PrefillBindingRole::kFullFp8O, *full_o,
                  views.attention.output_projection_temporary);
  roles[static_cast<std::size_t>(PrefillBindingRole::kLinearBf16A)] =
      receipt(PrefillBindingRole::kLinearBf16A,
              NativePrefillTactic::kBf16AbOracleSpanEstablishedM32,
              linear_a->weight,
              views.gdn.a_bf16.storage.device_data,
              views.gdn.a_bf16.storage.byte_size, 2U,
              kPrefillPhysicalSegmentMaximumTokens);
  roles[static_cast<std::size_t>(PrefillBindingRole::kLinearBf16B)] =
      receipt(PrefillBindingRole::kLinearBf16B,
              NativePrefillTactic::kBf16AbOracleSpanEstablishedM32,
              linear_b->weight,
              views.gdn.b_bf16.storage.device_data,
              views.gdn.b_bf16.storage.byte_size, 2U,
              kPrefillPhysicalSegmentMaximumTokens);
  roles[static_cast<std::size_t>(PrefillBindingRole::kExactGdn)] =
      receipt(PrefillBindingRole::kExactGdn,
              NativePrefillTactic::kExactGdnOracleSpanWholeRawQkvC512,
              linear->conv1d.data,
              views.gdn.native_c64_workspace.device_data,
              exact_gdn_workspace_bytes, 32U,
              kPrefillPhysicalSegmentMaximumTokens);
  roles[static_cast<std::size_t>(
      PrefillBindingRole::kExactCausalAttention)] =
      receipt(PrefillBindingRole::kExactCausalAttention,
              native_attention_tactic(full_attention_tactic),
              attention->q_norm.data,
              views.attention.core_output_bf16.storage.device_data,
              views.attention.core_output_bf16.storage.byte_size, 2U,
              native_attention_maximum_physical_m(full_attention_tactic));
  roles[static_cast<std::size_t>(PrefillBindingRole::kResidual)] =
      receipt(PrefillBindingRole::kResidual,
              NativePrefillTactic::kResidualOperatorPanel, weights,
              views.prompt_residual_bf16.storage.device_data,
              views.prompt_residual_bf16.storage.byte_size, 1U,
              persistent_p40_projection
                  ? kLayerMajorPrefillLayerWideMlpP40Tokens
                  : kLayerMajorPrefillOperatorPanelTokens,
              nullptr, 0U,
              persistent_p40_projection
                  ? kLayerMajorPrefillLayerWideMlpP40Tokens
                  : kLayerMajorPrefillOperatorPanelTokens);
  roles[static_cast<std::size_t>(PrefillBindingRole::kNormalization)] =
      receipt(PrefillBindingRole::kNormalization,
              NativePrefillTactic::kNormalizationOperatorPanel,
              persistent_p40_projection
                  ? linear_layer.post_attention_layernorm.data
                  : linear_layer.input_layernorm.data,
              persistent_p40_projection
                  ? views.mlp.normalized_input_bf16.storage.device_data
                  : views.gdn.normalized_input_bf16.storage.device_data,
              persistent_p40_projection
                  ? views.mlp.normalized_input_bf16.storage.byte_size
                  : views.gdn.normalized_input_bf16.storage.byte_size,
              1U,
              persistent_p40_projection
                  ? kLayerMajorPrefillLayerWideMlpP40Tokens
                  : kLayerMajorPrefillOperatorPanelTokens,
              nullptr, 0U,
              persistent_p40_projection
                  ? kLayerMajorPrefillLayerWideMlpP40Tokens
                  : kLayerMajorPrefillOperatorPanelTokens);
  roles[static_cast<std::size_t>(PrefillBindingRole::kEmbedding)] =
      receipt(PrefillBindingRole::kEmbedding,
              NativePrefillTactic::kEmbeddingOperatorPanel,
              weights->embed_tokens().weight,
              views.panel_token_ids_u32.storage.device_data,
              views.panel_token_ids_u32.storage.byte_size, 1U,
              kLayerMajorPrefillOperatorPanelTokens);
  roles[static_cast<std::size_t>(PrefillBindingRole::kFinalHandoff)] =
      receipt(PrefillBindingRole::kFinalHandoff,
              NativePrefillTactic::kFinalHandoff,
              weights->final_norm().data,
              views.final_hidden_bf16.storage.device_data,
              views.final_hidden_bf16.storage.byte_size, 1U, 1U);

  for (std::size_t index = 0U; index < roles.size(); ++index) {
    const NativePrefillRoleReceipt& role = roles[index];
    const PrefillBindingRole role_identity =
        static_cast<PrefillBindingRole>(index);
    const bool p40_full_m_role =
        persistent_p40_projection &&
        (role_identity == PrefillBindingRole::kNvfp4GateUp ||
         role_identity == PrefillBindingRole::kNvfp4Down ||
         role_identity == PrefillBindingRole::kResidual ||
         role_identity == PrefillBindingRole::kNormalization);
    const std::uint32_t expected_maximum_logical_m =
        p40_full_m_role ? kLayerMajorPrefillLayerWideMlpP40Tokens
                        : kLayerMajorPrefillOperatorPanelTokens;
    const bool workspace_valid =
        role.workspace_owner != nullptr && role.workspace_bytes != 0U &&
        ((role.auxiliary_workspace_owner == nullptr) ==
         (role.auxiliary_workspace_bytes == 0U));
    if (role.role != role_identity ||
        role.maximum_logical_panel_m != expected_maximum_logical_m ||
        role.artifact_owner == nullptr ||
        !workspace_valid || role.minimum_physical_m == 0U ||
        role.minimum_physical_m > role.maximum_physical_m ||
        role.maximum_physical_m > role.maximum_logical_panel_m ||
        role.completion !=
            NativePrefillCompletionDomain::kMainStreamBarrier) {
      return plan_failure(BoundPrefillPlanError::kIncompleteNativeRole,
                          ReferenceRunnerError::kInvalidDependency,
                          "bound_prefill_native_role");
    }
  }
  if (roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4GateUp)]
          .tactic ==
      roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4Down)]
          .tactic) {
    return plan_failure(BoundPrefillPlanError::kIncompleteNativeRole,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_gate_down_tactic_alias");
  }
  const NativePrefillRoleReceipt& exact_gdn =
      roles[static_cast<std::size_t>(PrefillBindingRole::kExactGdn)];
  if (exact_gdn.tactic !=
          NativePrefillTactic::kExactGdnOracleSpanWholeRawQkvC512 ||
      exact_gdn.workspace_owner !=
          views.gdn.native_c64_workspace.device_data ||
      exact_gdn.workspace_bytes != exact_gdn_workspace_bytes ||
      exact_gdn.minimum_physical_m != 32U ||
      exact_gdn.maximum_physical_m != 512U) {
    return plan_failure(BoundPrefillPlanError::kIncompleteNativeRole,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_exact_gdn_receipt");
  }

  const std::array<const void*, kBoundPrefillSubmissionEventCount>
      submission_events{
          runner->whole_request_submission_events_[0U],
          runner->whole_request_submission_events_[1U]};
  auto* const allocation = new (std::nothrow) BoundPrefillExecutionPlan(
      weights, state, runner, state->arena_data(), state->arena_bytes(),
      state->layer_major_plan(), arithmetic_contract,
      exact_c512_arithmetic_workspace_bound,
      projection_tactic,
      full_attention_tactic,
      mlp_schedule_tactic,
      runner->stream_,
      runner->prefill_auxiliary_stream_, submission_events,
      std::move(roles));
  if (allocation == nullptr) {
    return plan_failure(BoundPrefillPlanError::kInvalidDependency,
                        ReferenceRunnerError::kAllocationFailure,
                        "bound_prefill_plan_allocation");
  }
  BoundPrefillPlanResult result;
  result.value.reset(allocation);
  return result;
}

bool ReferenceEnginePrefillExecutor::plan_matches_runner(
    const BoundPrefillExecutionPlan& plan,
    const ReferenceRunner& runner) noexcept {
#if !defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION) || \
    !defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION) || \
    !defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  (void)plan;
  (void)runner;
  return false;
#else
  const LayerMajorPrefillArithmeticContract* const expected_contract =
      projection_arithmetic_contract(plan.projection_tactic_);
  const bool layer_wide_p40_projection =
      plan.projection_tactic_ == LayerMajorPrefillProjectionTactic::
                                     kNativeNvfp4PersistentP40LayerWideMlp;
  const bool whole_core_projection =
      plan.projection_tactic_ == LayerMajorPrefillProjectionTactic::
                                     kNativePromptWideP40WholeCore;
  const bool persistent_p40_projection =
      layer_wide_p40_projection || whole_core_projection;
  const LayerMajorPrefillMlpScheduleTactic expected_mlp_schedule =
      whole_core_projection
          ? LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore
      : layer_wide_p40_projection
          ? LayerMajorPrefillMlpScheduleTactic::kLayerWideP40ExactFullM
          : LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel;
  const NvFp4MarlinGateUpLayout expected_gate_up_layout =
      persistent_p40_projection
          ? NvFp4MarlinGateUpLayout::kInterleavedGateUp
          : NvFp4MarlinGateUpLayout::kCanonicalGateThenUp;
  if (plan.runner_ != &runner || plan.weights_ != runner.weights_ ||
      plan.state_ != runner.state_ || plan.state_ == nullptr ||
      plan.arena_base_ != plan.state_->arena_data() ||
      plan.arena_bytes_ != plan.state_->arena_bytes() ||
      plan.memory_plan_ != plan.state_->layer_major_plan() ||
      expected_contract == nullptr ||
      plan.arithmetic_contract_ != expected_contract ||
      !is_valid_layer_major_prefill_arithmetic_contract(
          *plan.arithmetic_contract_) ||
      !plan.exact_c512_arithmetic_workspace_bound_ ||
      !is_valid_layer_major_prefill_projection_tactic(
          plan.projection_tactic_) ||
      !is_valid_layer_major_prefill_full_attention_tactic(
          plan.full_attention_tactic_) ||
      (plan.full_attention_tactic_ ==
           LayerMajorPrefillFullAttentionTactic::
               kNativeFlashInferExactWholePrompt) != whole_core_projection ||
      plan.mlp_schedule_tactic_ != expected_mlp_schedule ||
      plan.main_stream_ != runner.stream_ ||
      plan.auxiliary_stream_ != runner.prefill_auxiliary_stream_ ||
      plan.submission_events_[0U] !=
          runner.whole_request_submission_events_[0U] ||
      plan.submission_events_[1U] !=
          runner.whole_request_submission_events_[1U] ||
      !runner.layer_major_request_views_.has_value() ||
      runner.projection_backend_ != ProjectionBackend::kSm87WeightOnly) {
    return false;
  }
  const ReferenceLayerMajorRequestViews& views =
      *runner.layer_major_request_views_;
  const bool native_large_m_projection =
      plan.projection_tactic_ == LayerMajorPrefillProjectionTactic::
                                      kNativeQuantizedLargeMOperatorPanel;
  [[maybe_unused]] const bool true_large_m_nvfp4_projection =
      plan.projection_tactic_ == LayerMajorPrefillProjectionTactic::
                                      kNativeNvfp4TrueLargeMOperatorPanel;
  [[maybe_unused]] const bool g2_d2_nvfp4_projection =
      plan.projection_tactic_ == LayerMajorPrefillProjectionTactic::
                                      kNativeNvfp4G2D2LargeMOperatorPanel;
  if (whole_core_projection &&
      (!whole_core_compile_inventory_enabled() ||
       !complete_flashinfer_whole_prompt_p40_capability() ||
       !complete_bf16_ab_prompt_wide_p40_capability() ||
       !complete_gdn_prompt_wide_p40_capability())) {
    return false;
  }
#if !defined(Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
  if (true_large_m_nvfp4_projection) {
    return false;
  }
#else
  if (true_large_m_nvfp4_projection &&
      !complete_nvfp4_true_large_m_capability()) {
    return false;
  }
#endif
#if !defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION)
  if (persistent_p40_projection) {
    return false;
  }
#else
  if (persistent_p40_projection &&
      !complete_nvfp4_persistent_p40_capability()) {
    return false;
  }
#endif
#if !defined(Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION)
  if (g2_d2_nvfp4_projection) {
    return false;
  }
#else
  if (g2_d2_nvfp4_projection && !complete_nvfp4_g2_d2_capability()) {
    return false;
  }
#endif
  if (plan.full_attention_tactic_ == LayerMajorPrefillFullAttentionTactic::
                                         kNativeFlashInferExactPanel &&
      !has_bulk_causal_gqa_flashinfer_exact_panel_cuda()) {
    return false;
  }
  if (!complete_installed_projection_inventory(*runner.weights_,
                                                expected_gate_up_layout)) {
    return false;
  }
  bool fallback_views_match =
      views.legacy_c512.linear_a_bf16.storage.device_data != nullptr &&
      views.legacy_c512.linear_b_bf16.storage.device_data != nullptr &&
      views.legacy_c512.fp32_scratch.device_data != nullptr;
  for (const DeviceMatrixView& hidden : views.legacy_c512.hidden_bf16) {
    fallback_views_match = fallback_views_match &&
                           hidden.storage.device_data != nullptr;
  }
  for (const DeviceMatrixView& projection :
       views.legacy_c512.projection_bf16) {
    fallback_views_match = fallback_views_match &&
                           projection.storage.device_data != nullptr;
  }
  fallback_views_match =
      fallback_views_match &&
      views.legacy_c512.fp32_scratch.byte_size >=
          kernels::kSm87NvFp4MarlinReductionBytes &&
      views.legacy_c512.projection_bf16[3].storage.byte_size >=
          kernels::kSm87NvFp4MarlinLockBytes &&
      reinterpret_cast<std::uintptr_t>(
          views.legacy_c512.projection_bf16[1].storage.device_data) ==
          reinterpret_cast<std::uintptr_t>(
              views.legacy_c512.projection_bf16[0].storage.device_data) +
              views.legacy_c512.projection_bf16[0].storage.byte_size;
  if (!fallback_views_match) {
    return false;
  }
  const DecoderLayerWeights& linear_layer = runner.weights_->layer(0U);
  const DecoderLayerWeights& full_layer = runner.weights_->layer(3U);
  const auto* const linear =
      std::get_if<LinearAttentionWeights>(&linear_layer.attention);
  const auto* const attention =
      std::get_if<FullAttentionWeights>(&full_layer.attention);
  const auto* const gate =
      std::get_if<NvFp4LinearWeight>(&linear_layer.mlp.gate_proj);
  const auto* const up =
      std::get_if<NvFp4LinearWeight>(&linear_layer.mlp.up_proj);
  const auto* const down =
      std::get_if<NvFp4LinearWeight>(&linear_layer.mlp.down_proj);
  if (linear == nullptr || attention == nullptr || gate == nullptr ||
      up == nullptr || down == nullptr ||
      gate->prefill_marlin_gate_up_layout != expected_gate_up_layout ||
      up->prefill_marlin_gate_up_layout !=
          gate->prefill_marlin_gate_up_layout ||
      (!whole_core_projection &&
       views.attention.normalized_input_bf16.storage.device_data == nullptr) ||
      views.mlp.normalized_input_bf16.storage.device_data == nullptr) {
    return false;
  }
  if (whole_core_projection) {
    if (plan.state_->memory_profile() !=
            RequestMemoryProfile::kLayerMajorP40WholeCore ||
        plan.memory_plan_ == nullptr ||
        plan.memory_plan_->layout !=
            LayerMajorRequestLayout::kP40WholeCorePromptWide ||
        plan.arena_bytes_ != kPromptWideP40WholeCoreArenaBytes ||
        !complete_prompt_wide_p40_whole_core_views(views) ||
        !complete_exact_gdn_chunk64_native_inventory(*runner.weights_)) {
      return false;
    }
    const auto* const linear_qkv =
        std::get_if<Fp8LinearWeight>(&linear->in_proj_qkv);
    const auto* const linear_z =
        std::get_if<Fp8LinearWeight>(&linear->in_proj_z);
    const auto* const linear_o =
        std::get_if<Fp8LinearWeight>(&linear->out_proj);
    const auto* const linear_a =
        std::get_if<Bf16LinearWeight>(&linear->in_proj_a);
    const auto* const linear_b =
        std::get_if<Bf16LinearWeight>(&linear->in_proj_b);
    const auto* const full_q =
        std::get_if<Fp8LinearWeight>(&attention->q_proj);
    const auto* const full_k =
        std::get_if<Fp8LinearWeight>(&attention->k_proj);
    const auto* const full_v =
        std::get_if<Fp8LinearWeight>(&attention->v_proj);
    const auto* const full_o =
        std::get_if<Fp8LinearWeight>(&attention->o_proj);
    if (linear_qkv == nullptr || linear_z == nullptr || linear_o == nullptr ||
        linear_a == nullptr || linear_b == nullptr || full_q == nullptr ||
        full_k == nullptr || full_v == nullptr || full_o == nullptr) {
      return false;
    }
    const auto matches = [&plan](
                             const PrefillBindingRole role,
                             const NativePrefillTactic tactic,
                             const void* const artifact,
                             const void* const workspace,
                             const std::uint64_t workspace_bytes,
                             const std::uint32_t minimum_physical_m,
                             const std::uint32_t maximum_physical_m,
                             const void* const auxiliary_workspace = nullptr,
                             const std::uint64_t auxiliary_workspace_bytes = 0U,
                             const std::uint32_t maximum_logical_m =
                                 kLayerMajorPrefillPromptWideP40Tokens) noexcept {
      const NativePrefillRoleReceipt& receipt =
          plan.roles_[static_cast<std::size_t>(role)];
      return receipt.role == role && receipt.tactic == tactic &&
             receipt.completion ==
                 NativePrefillCompletionDomain::kMainStreamBarrier &&
             receipt.artifact_owner == artifact &&
             receipt.workspace_owner == workspace &&
             receipt.workspace_bytes == workspace_bytes &&
             receipt.auxiliary_workspace_owner == auxiliary_workspace &&
             receipt.auxiliary_workspace_bytes ==
                 auxiliary_workspace_bytes &&
             receipt.maximum_logical_panel_m == maximum_logical_m &&
             receipt.minimum_physical_m == minimum_physical_m &&
             receipt.maximum_physical_m == maximum_physical_m;
    };
    constexpr std::uint64_t kGateUpScaleBytes =
        static_cast<std::uint64_t>(kReferenceHiddenSize) *
        (2U * kReferenceIntermediateSize) / 16U;
    constexpr std::uint64_t kDownScaleBytes =
        static_cast<std::uint64_t>(kReferenceIntermediateSize) *
        kReferenceHiddenSize / 16U;
    const DeviceBufferView& reduction = views.legacy_c512.fp32_scratch;
    const DeviceBufferView& locks =
        views.legacy_c512.projection_bf16[3U].storage;
    const auto fp8_matches = [&matches, &reduction, &locks](
                                 const PrefillBindingRole role,
                                 const Fp8LinearWeight& weight) noexcept {
      return matches(
          role, NativePrefillTactic::kFp8PromptWideP40FillDrain,
          weight.prefill_marlin_weight, reduction.device_data,
          kernels::kSm87Fp8MarlinReductionBytes,
          kLayerMajorPrefillPromptWideP40PanelTokens,
          kLayerMajorPrefillPromptWideP40PanelTokens, locks.device_data,
          kernels::kSm87Fp8MarlinLockBytes,
          kLayerMajorPrefillPromptWideP40PanelTokens);
    };
    const LayerMajorP40WholeCoreViews& whole = views.p40_whole_core;
    return matches(
               PrefillBindingRole::kNvfp4GateUp,
               NativePrefillTactic::kNvfp4GateUpPersistentP40LayerWide,
               gate->prefill_marlin_weight, gate->prefill_marlin_scales,
               kGateUpScaleBytes, kLayerMajorPrefillPromptWideP40Tokens,
               kLayerMajorPrefillPromptWideP40Tokens,
               gate->prefill_marlin_global_scale, sizeof(float)) &&
           matches(
               PrefillBindingRole::kNvfp4Down,
               NativePrefillTactic::kNvfp4DownResidualPersistentP40LayerWide,
               down->prefill_marlin_weight, down->prefill_marlin_scales,
               kDownScaleBytes, kLayerMajorPrefillPromptWideP40Tokens,
               kLayerMajorPrefillPromptWideP40Tokens,
               down->prefill_marlin_global_scale, sizeof(float)) &&
           fp8_matches(PrefillBindingRole::kLinearFp8Qkv, *linear_qkv) &&
           fp8_matches(PrefillBindingRole::kLinearFp8Z, *linear_z) &&
           fp8_matches(PrefillBindingRole::kLinearFp8O, *linear_o) &&
           fp8_matches(PrefillBindingRole::kFullFp8Q, *full_q) &&
           fp8_matches(PrefillBindingRole::kFullFp8K, *full_k) &&
           fp8_matches(PrefillBindingRole::kFullFp8V, *full_v) &&
           fp8_matches(PrefillBindingRole::kFullFp8O, *full_o) &&
           matches(PrefillBindingRole::kLinearBf16A,
                   NativePrefillTactic::kBf16AbPromptWideP40,
                   linear_a->weight,
                   whole.linear.a_bf16.storage.device_data,
                   whole.linear.a_bf16.storage.byte_size,
                   kLayerMajorPrefillPromptWideP40Tokens,
                   kLayerMajorPrefillPromptWideP40Tokens) &&
           matches(PrefillBindingRole::kLinearBf16B,
                   NativePrefillTactic::kBf16AbPromptWideP40,
                   linear_b->weight,
                   whole.linear.b_bf16.storage.device_data,
                   whole.linear.b_bf16.storage.byte_size,
                   kLayerMajorPrefillPromptWideP40Tokens,
                   kLayerMajorPrefillPromptWideP40Tokens) &&
           matches(
               PrefillBindingRole::kExactGdn,
               NativePrefillTactic::kExactGdnPromptWideP40ChunkGraph,
               linear->conv1d.data,
               whole.linear.prompt_wide_workspace.device_data,
               kernels::kGdnPromptWideChunkGraphP40WorkspaceBytes,
               kLayerMajorPrefillPromptWideP40Tokens,
               kLayerMajorPrefillPromptWideP40Tokens) &&
           matches(
               PrefillBindingRole::kExactCausalAttention,
               NativePrefillTactic::
                   kNativeCausalAttentionFlashInferExactWholePrompt,
               attention->q_norm.data,
               whole.full_attention.core_output_bf16.storage.device_data,
               whole.full_attention.core_output_bf16.storage.byte_size,
               kLayerMajorPrefillPromptWideP40Tokens,
               kLayerMajorPrefillPromptWideP40Tokens) &&
           matches(PrefillBindingRole::kResidual,
                   NativePrefillTactic::kResidualOperatorPanel,
                   runner.weights_,
                   views.prompt_residual_bf16.storage.device_data,
                   views.prompt_residual_bf16.storage.byte_size,
                   kLayerMajorPrefillPromptWideP40PanelTokens,
                   kLayerMajorPrefillPromptWideP40PanelTokens) &&
           matches(
               PrefillBindingRole::kNormalization,
               NativePrefillTactic::kNormalizationOperatorPanel,
               linear_layer.post_attention_layernorm.data,
               whole.linear.normalized_input_bf16.storage.device_data,
               whole.linear.normalized_input_bf16.storage.byte_size,
               kLayerMajorPrefillPromptWideP40PanelTokens,
               kLayerMajorPrefillPromptWideP40Tokens) &&
           matches(PrefillBindingRole::kEmbedding,
                   NativePrefillTactic::kEmbeddingOperatorPanel,
                   runner.weights_->embed_tokens().weight,
                   whole.prompt_token_ids_u32.storage.device_data,
                   whole.prompt_token_ids_u32.storage.byte_size,
                   kLayerMajorPrefillPromptWideP40PanelTokens,
                   kLayerMajorPrefillPromptWideP40PanelTokens) &&
           matches(PrefillBindingRole::kFinalHandoff,
                   NativePrefillTactic::kFinalHandoff,
                   runner.weights_->final_norm().data,
                   views.final_hidden_bf16.storage.device_data,
                   views.final_hidden_bf16.storage.byte_size, 1U, 1U,
                   nullptr, 0U, 1U);
  }
  constexpr std::size_t kFp8MarlinTemporaryBytes =
      kernels::kSm87Fp8MarlinReductionBytes +
      kernels::kSm87Fp8MarlinLockBytes;
  const auto valid_fp8_temporary = [](const DeviceBufferView& view) noexcept {
    return view.device_data != nullptr &&
           view.byte_size >= kFp8MarlinTemporaryBytes;
  };
  const bool true_large_m_fp8_temporaries_complete =
      valid_fp8_temporary(views.gdn.input_projection_temporary) &&
      valid_fp8_temporary(views.gdn.output_projection_temporary) &&
      valid_fp8_temporary(views.attention.input_projection_temporary) &&
      valid_fp8_temporary(views.attention.output_projection_temporary);
  if (true_large_m_nvfp4_projection &&
      (views.mlp.gate_bf16.storage.device_data == nullptr ||
       views.mlp.up_bf16.storage.device_data == nullptr ||
       views.mlp.activated_bf16.storage.device_data == nullptr ||
       views.mlp.branch_output_bf16.storage.device_data == nullptr ||
       !true_large_m_fp8_temporaries_complete ||
       reinterpret_cast<std::uintptr_t>(
           views.mlp.up_bf16.storage.device_data) !=
           reinterpret_cast<std::uintptr_t>(
               views.mlp.gate_bf16.storage.device_data) +
               views.mlp.gate_bf16.storage.byte_size)) {
    return false;
  }
  if (g2_d2_nvfp4_projection &&
      (views.mlp.gate_bf16.storage.device_data == nullptr ||
       views.mlp.gate_bf16.storage.device_data ==
           views.mlp.normalized_input_bf16.storage.device_data ||
       !true_large_m_fp8_temporaries_complete)) {
    return false;
  }
  if (persistent_p40_projection &&
      (views.descriptor.max_sequence_length !=
           kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens ||
       views.descriptor.mlp_capacity_tokens !=
           kLayerMajorPrefillLayerWideMlpP40Tokens ||
       views.descriptor.mlp_layout !=
           LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan ||
       views.mlp.normalized_input_bf16.row_capacity !=
           kLayerMajorPrefillLayerWideMlpP40Tokens ||
       views.mlp.gate_bf16.row_capacity !=
           kLayerMajorPrefillLayerWideMlpP40Tokens ||
       views.mlp.gate_bf16.storage.device_data == nullptr ||
       views.mlp.gate_bf16.storage.device_data ==
           views.mlp.normalized_input_bf16.storage.device_data ||
       !true_large_m_fp8_temporaries_complete)) {
    return false;
  }
  const auto* const linear_qkv =
      std::get_if<Fp8LinearWeight>(&linear->in_proj_qkv);
  const auto* const linear_z =
      std::get_if<Fp8LinearWeight>(&linear->in_proj_z);
  const auto* const linear_o =
      std::get_if<Fp8LinearWeight>(&linear->out_proj);
  const auto* const linear_a =
      std::get_if<Bf16LinearWeight>(&linear->in_proj_a);
  const auto* const linear_b =
      std::get_if<Bf16LinearWeight>(&linear->in_proj_b);
  const auto* const full_q =
      std::get_if<Fp8LinearWeight>(&attention->q_proj);
  const auto* const full_k =
      std::get_if<Fp8LinearWeight>(&attention->k_proj);
  const auto* const full_v =
      std::get_if<Fp8LinearWeight>(&attention->v_proj);
  const auto* const full_o =
      std::get_if<Fp8LinearWeight>(&attention->o_proj);
  if (linear_qkv == nullptr || linear_z == nullptr || linear_o == nullptr ||
      linear_a == nullptr || linear_b == nullptr || full_q == nullptr ||
      full_k == nullptr || full_v == nullptr || full_o == nullptr) {
    return false;
  }
  const auto matches = [&plan](
                           const PrefillBindingRole role,
                           const NativePrefillTactic tactic,
                           const void* const artifact,
                           const void* const workspace,
                           const std::uint64_t workspace_bytes,
                           const std::uint32_t minimum_physical_m,
                           const std::uint32_t maximum_physical_m,
                           const void* const auxiliary_workspace = nullptr,
                           const std::uint64_t auxiliary_workspace_bytes =
                               0U,
                           const std::uint32_t maximum_logical_m =
                               kLayerMajorPrefillOperatorPanelTokens) noexcept {
    const NativePrefillRoleReceipt& receipt =
        plan.roles_[static_cast<std::size_t>(role)];
    return receipt.role == role && receipt.tactic == tactic &&
           receipt.completion ==
               NativePrefillCompletionDomain::kMainStreamBarrier &&
           receipt.artifact_owner == artifact &&
           receipt.workspace_owner == workspace &&
           receipt.workspace_bytes == workspace_bytes &&
           receipt.auxiliary_workspace_owner == auxiliary_workspace &&
           receipt.auxiliary_workspace_bytes ==
               auxiliary_workspace_bytes &&
           receipt.maximum_logical_panel_m == maximum_logical_m &&
           receipt.minimum_physical_m == minimum_physical_m &&
           receipt.maximum_physical_m == maximum_physical_m;
  };
  const bool segmented_projection =
      plan.projection_tactic_ ==
      LayerMajorPrefillProjectionTactic::kSegmentedMarlinOperatorPanel;
  const auto fp8_locks = [](const DeviceBufferView& temporary) noexcept {
    return static_cast<void*>(
        static_cast<std::uint8_t*>(temporary.device_data) +
        kernels::kSm87Fp8MarlinReductionBytes);
  };
  const auto nvfp4_locks = [](const DeviceBufferView& temporary) noexcept {
    return static_cast<void*>(
        static_cast<std::uint8_t*>(temporary.device_data) +
        kernels::kSm87NvFp4MarlinReductionBytes);
  };
  const auto fp8_matches = [&matches, &fp8_locks, segmented_projection,
                            native_large_m_projection,
                            true_large_m_nvfp4_projection,
                            g2_d2_nvfp4_projection,
                            persistent_p40_projection](
                               const PrefillBindingRole role,
                               const Fp8LinearWeight& weight,
                               const DeviceBufferView& temporary) noexcept {
    return (true_large_m_nvfp4_projection || g2_d2_nvfp4_projection ||
            persistent_p40_projection)
               ? matches(role,
                         NativePrefillTactic::
                             kFp8Nvfp4TrueLargeMRouteCompanion,
                         weight.prefill_marlin_weight,
                         temporary.device_data,
                         kernels::kSm87Fp8MarlinReductionBytes, 2U,
                         kLayerMajorPrefillOperatorPanelTokens,
                         fp8_locks(temporary),
                         kernels::kSm87Fp8MarlinLockBytes)
           : native_large_m_projection
               ? matches(role,
                         NativePrefillTactic::
                             kFp8NativeQuantizedLargeMOperatorPanel,
                         weight.prefill_marlin_weight,
                         temporary.device_data,
                         kernels::kSm87Fp8MarlinReductionBytes, 2U,
                         kLayerMajorPrefillOperatorPanelTokens,
                         fp8_locks(temporary),
                         kernels::kSm87Fp8MarlinLockBytes)
           : segmented_projection
               ? matches(
                     role, NativePrefillTactic::kFp8SegmentedMarlinOperatorPanel,
                     weight.prefill_marlin_weight, temporary.device_data,
                     kernels::kSm87Fp8MarlinReductionBytes, 2U,
                     static_cast<std::uint32_t>(
                         kernels::sm87_fp8_marlin_maximum_kernel_segment_tokens(
                             weight.output_size)),
                     fp8_locks(temporary),
                     kernels::kSm87Fp8MarlinLockBytes)
               : matches(role, NativePrefillTactic::kFp8OracleSpanC512,
                         weight.prefill_marlin_weight, temporary.device_data,
                         temporary.byte_size, 2U,
                         kPrefillPhysicalSegmentMaximumTokens);
  };
  bool g2_d2_nvfp4_receipts_match = false;
#if defined(Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION)
  constexpr std::uint64_t kG2GateUpScaleBytes =
      static_cast<std::uint64_t>(kReferenceHiddenSize) *
      (2U * kReferenceIntermediateSize) / 16U;
  constexpr std::uint64_t kD2DownScaleBytes =
      static_cast<std::uint64_t>(kReferenceIntermediateSize) *
      kReferenceHiddenSize / 16U;
  g2_d2_nvfp4_receipts_match =
      matches(PrefillBindingRole::kNvfp4GateUp,
              NativePrefillTactic::kNvfp4GateUpG2LargeMOperatorPanel,
              gate->prefill_marlin_weight, gate->prefill_marlin_scales,
              kG2GateUpScaleBytes,
              kLayerMajorPrefillTrueLargeMPartialPanelTokens,
              kLayerMajorPrefillOperatorPanelTokens,
              gate->prefill_marlin_global_scale, sizeof(float)) &&
      matches(PrefillBindingRole::kNvfp4Down,
              NativePrefillTactic::kNvfp4DownD2LargeMOperatorPanel,
              down->prefill_marlin_weight, down->prefill_marlin_scales,
              kD2DownScaleBytes,
              kLayerMajorPrefillTrueLargeMPartialPanelTokens,
              kLayerMajorPrefillOperatorPanelTokens,
              down->prefill_marlin_global_scale, sizeof(float));
#endif
  bool true_large_m_nvfp4_receipts_match = false;
#if defined(Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
  constexpr std::uint64_t kGateUpScaleBytes =
      kernels::kSm87NvFp4PrefillLargeMHidden *
      kernels::kSm87NvFp4PrefillLargeMGateUpOutput / 16U;
  constexpr std::uint64_t kDownScaleBytes =
      kernels::kSm87NvFp4PrefillLargeMIntermediate *
      kernels::kSm87NvFp4PrefillLargeMHidden / 16U;
  true_large_m_nvfp4_receipts_match =
      matches(PrefillBindingRole::kNvfp4GateUp,
              NativePrefillTactic::kNvfp4GateUpTrueLargeMOperatorPanel,
              gate->prefill_marlin_weight, gate->prefill_marlin_scales,
              kGateUpScaleBytes,
              kLayerMajorPrefillTrueLargeMPartialPanelTokens,
              kLayerMajorPrefillOperatorPanelTokens,
              gate->prefill_marlin_global_scale, sizeof(float)) &&
      matches(PrefillBindingRole::kNvfp4Down,
              NativePrefillTactic::kNvfp4DownTrueLargeMOperatorPanel,
              down->prefill_marlin_weight, down->prefill_marlin_scales,
              kDownScaleBytes,
              kLayerMajorPrefillTrueLargeMPartialPanelTokens,
              kLayerMajorPrefillOperatorPanelTokens,
              down->prefill_marlin_global_scale, sizeof(float));
#endif
  bool persistent_p40_nvfp4_receipts_match = false;
#if defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION)
  constexpr std::uint64_t kPersistentGateUpScaleBytes =
      static_cast<std::uint64_t>(kReferenceHiddenSize) *
      (2U * kReferenceIntermediateSize) / 16U;
  constexpr std::uint64_t kPersistentDownScaleBytes =
      static_cast<std::uint64_t>(kReferenceIntermediateSize) *
      kReferenceHiddenSize / 16U;
  persistent_p40_nvfp4_receipts_match =
      matches(
          PrefillBindingRole::kNvfp4GateUp,
          NativePrefillTactic::kNvfp4GateUpPersistentP40LayerWide,
          gate->prefill_marlin_weight, gate->prefill_marlin_scales,
          kPersistentGateUpScaleBytes,
          kLayerMajorPrefillLayerWideMlpP40Tokens,
          kLayerMajorPrefillLayerWideMlpP40Tokens,
          gate->prefill_marlin_global_scale, sizeof(float),
          kLayerMajorPrefillLayerWideMlpP40Tokens) &&
      matches(
          PrefillBindingRole::kNvfp4Down,
          NativePrefillTactic::
              kNvfp4DownResidualPersistentP40LayerWide,
          down->prefill_marlin_weight, down->prefill_marlin_scales,
          kPersistentDownScaleBytes,
          kLayerMajorPrefillLayerWideMlpP40Tokens,
          kLayerMajorPrefillLayerWideMlpP40Tokens,
          down->prefill_marlin_global_scale, sizeof(float),
          kLayerMajorPrefillLayerWideMlpP40Tokens);
#endif
  const bool nvfp4_matches =
      persistent_p40_projection
          ? persistent_p40_nvfp4_receipts_match
      : g2_d2_nvfp4_projection
          ? g2_d2_nvfp4_receipts_match
      : true_large_m_nvfp4_projection
          ? true_large_m_nvfp4_receipts_match
      : native_large_m_projection
          ? matches(
                PrefillBindingRole::kNvfp4GateUp,
                NativePrefillTactic::
                    kNvfp4GateUpNativeQuantizedLargeMOperatorPanel,
                gate->prefill_marlin_weight,
                views.mlp.gate_up_projection_temporary.device_data,
                kernels::kSm87NvFp4MarlinReductionBytes, 1U,
                kLayerMajorPrefillOperatorPanelTokens,
                nvfp4_locks(views.mlp.gate_up_projection_temporary),
                kernels::kSm87NvFp4MarlinLockBytes) &&
                matches(
                    PrefillBindingRole::kNvfp4Down,
                    NativePrefillTactic::
                        kNvfp4DownNativeQuantizedLargeMOperatorPanel,
                    down->prefill_marlin_weight,
                    views.mlp.down_projection_temporary.device_data,
                    kernels::kSm87NvFp4MarlinReductionBytes, 1U,
                    kLayerMajorPrefillOperatorPanelTokens,
                    nvfp4_locks(views.mlp.down_projection_temporary),
                    kernels::kSm87NvFp4MarlinLockBytes)
      : segmented_projection
          ? matches(
                PrefillBindingRole::kNvfp4GateUp,
                NativePrefillTactic::kNvfp4GateUpSegmentedMarlinOperatorPanel,
                gate->prefill_marlin_weight,
                views.mlp.gate_up_projection_temporary.device_data,
                kernels::kSm87NvFp4MarlinReductionBytes, 1U,
                kernels::kSm87NvFp4MarlinMaximumKernelSegmentTokens,
                nvfp4_locks(views.mlp.gate_up_projection_temporary),
                kernels::kSm87NvFp4MarlinLockBytes) &&
                matches(
                    PrefillBindingRole::kNvfp4Down,
                    NativePrefillTactic::kNvfp4DownSegmentedMarlinOperatorPanel,
                    down->prefill_marlin_weight,
                    views.mlp.down_projection_temporary.device_data,
                    kernels::kSm87NvFp4MarlinReductionBytes, 1U,
                    kernels::kSm87NvFp4MarlinMaximumKernelSegmentTokens,
                    nvfp4_locks(views.mlp.down_projection_temporary),
                    kernels::kSm87NvFp4MarlinLockBytes)
          : matches(PrefillBindingRole::kNvfp4GateUp,
                    NativePrefillTactic::kNvfp4GateUpOracleSpanC512,
                    gate->prefill_marlin_weight,
                    views.legacy_c512.fp32_scratch.device_data,
                    kernels::kSm87NvFp4MarlinReductionBytes, 1U,
                    kPrefillPhysicalSegmentMaximumTokens,
                    views.legacy_c512.projection_bf16[3].storage.device_data,
                    kernels::kSm87NvFp4MarlinLockBytes) &&
                matches(PrefillBindingRole::kNvfp4Down,
                        NativePrefillTactic::kNvfp4DownOracleSpanC512,
                        down->prefill_marlin_weight,
                        views.legacy_c512.fp32_scratch.device_data,
                        kernels::kSm87NvFp4MarlinReductionBytes, 1U,
                        kPrefillPhysicalSegmentMaximumTokens,
                        views.legacy_c512.projection_bf16[3].storage.device_data,
                        kernels::kSm87NvFp4MarlinLockBytes);
  const bool projection_receipts_match =
      nvfp4_matches &&
      fp8_matches(PrefillBindingRole::kLinearFp8Qkv, *linear_qkv,
                  views.gdn.input_projection_temporary) &&
      fp8_matches(PrefillBindingRole::kLinearFp8Z, *linear_z,
                  views.gdn.input_projection_temporary) &&
      fp8_matches(PrefillBindingRole::kLinearFp8O, *linear_o,
                  views.gdn.output_projection_temporary) &&
      fp8_matches(PrefillBindingRole::kFullFp8Q, *full_q,
                  views.attention.input_projection_temporary) &&
      fp8_matches(PrefillBindingRole::kFullFp8K, *full_k,
                  views.attention.input_projection_temporary) &&
      fp8_matches(PrefillBindingRole::kFullFp8V, *full_v,
                  views.attention.input_projection_temporary) &&
      fp8_matches(PrefillBindingRole::kFullFp8O, *full_o,
                  views.attention.output_projection_temporary);
  return projection_receipts_match &&
         matches(PrefillBindingRole::kLinearBf16A,
                 NativePrefillTactic::kBf16AbOracleSpanEstablishedM32,
                 linear_a->weight, views.gdn.a_bf16.storage.device_data,
                 views.gdn.a_bf16.storage.byte_size, 2U,
                 kPrefillPhysicalSegmentMaximumTokens) &&
         matches(PrefillBindingRole::kLinearBf16B,
                 NativePrefillTactic::kBf16AbOracleSpanEstablishedM32,
                 linear_b->weight, views.gdn.b_bf16.storage.device_data,
                 views.gdn.b_bf16.storage.byte_size, 2U,
                 kPrefillPhysicalSegmentMaximumTokens) &&
         matches(PrefillBindingRole::kExactGdn,
                 NativePrefillTactic::kExactGdnOracleSpanWholeRawQkvC512,
                 linear->conv1d.data,
                 views.gdn.native_c64_workspace.device_data,
                 gdn_prefill_chunk64_native_detail::workspace_bytes(), 32U,
                 kPrefillPhysicalSegmentMaximumTokens) &&
         matches(PrefillBindingRole::kExactCausalAttention,
                 native_attention_tactic(plan.full_attention_tactic_),
                 attention->q_norm.data,
                 views.attention.core_output_bf16.storage.device_data,
                 views.attention.core_output_bf16.storage.byte_size, 2U,
                 native_attention_maximum_physical_m(
                     plan.full_attention_tactic_)) &&
         matches(PrefillBindingRole::kResidual,
                 NativePrefillTactic::kResidualOperatorPanel,
                 runner.weights_,
                 views.prompt_residual_bf16.storage.device_data,
                 views.prompt_residual_bf16.storage.byte_size, 1U,
                 persistent_p40_projection
                     ? kLayerMajorPrefillLayerWideMlpP40Tokens
                     : kLayerMajorPrefillOperatorPanelTokens,
                 nullptr, 0U,
                 persistent_p40_projection
                     ? kLayerMajorPrefillLayerWideMlpP40Tokens
                     : kLayerMajorPrefillOperatorPanelTokens) &&
         matches(PrefillBindingRole::kNormalization,
                 NativePrefillTactic::kNormalizationOperatorPanel,
                 persistent_p40_projection
                     ? linear_layer.post_attention_layernorm.data
                     : linear_layer.input_layernorm.data,
                 persistent_p40_projection
                     ? views.mlp.normalized_input_bf16.storage.device_data
                     : views.gdn.normalized_input_bf16.storage.device_data,
                 persistent_p40_projection
                     ? views.mlp.normalized_input_bf16.storage.byte_size
                     : views.gdn.normalized_input_bf16.storage.byte_size,
                 1U,
                 persistent_p40_projection
                     ? kLayerMajorPrefillLayerWideMlpP40Tokens
                     : kLayerMajorPrefillOperatorPanelTokens,
                 nullptr, 0U,
                 persistent_p40_projection
                     ? kLayerMajorPrefillLayerWideMlpP40Tokens
                     : kLayerMajorPrefillOperatorPanelTokens) &&
         matches(PrefillBindingRole::kEmbedding,
                 NativePrefillTactic::kEmbeddingOperatorPanel,
                 runner.weights_->embed_tokens().weight,
                 views.panel_token_ids_u32.storage.device_data,
                 views.panel_token_ids_u32.storage.byte_size, 1U,
                 kLayerMajorPrefillOperatorPanelTokens) &&
         matches(PrefillBindingRole::kFinalHandoff,
                 NativePrefillTactic::kFinalHandoff,
                 runner.weights_->final_norm().data,
                 views.final_hidden_bf16.storage.device_data,
                 views.final_hidden_bf16.storage.byte_size, 1U, 1U);
#endif
}

std::string_view ReferenceEnginePrefillExecutor::deployment_plan_id(
    const BoundPrefillExecutionPlan& plan) noexcept {
  if (!plan.exact_c512_arithmetic_workspace_bound_) {
    return {};
  }
  if (plan.projection_tactic_ ==
      LayerMajorPrefillProjectionTactic::kNativePromptWideP40WholeCore) {
    return plan.full_attention_tactic_ ==
                   LayerMajorPrefillFullAttentionTactic::
                       kNativeFlashInferExactWholePrompt
               ? kLayerMajorNativePromptWideP40WholeCoreDeploymentPlanId
               : std::string_view{};
  }
  if (plan.projection_tactic_ == LayerMajorPrefillProjectionTactic::
                                     kNativeNvfp4PersistentP40LayerWideMlp) {
    switch (plan.full_attention_tactic_) {
      case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel:
        return kLayerMajorNativeNvfp4PersistentP40MlpGroupQ64DeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ128V4Panel:
        return kLayerMajorNativeNvfp4PersistentP40MlpGroupQ128V4DeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kNativeFlashInferExactPanel:
        return kLayerMajorNativeNvfp4PersistentP40MlpFlashInferExactDeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512:
      default:
        return kLayerMajorNativeNvfp4PersistentP40MlpDeploymentPlanId;
    }
  }
  if (plan.projection_tactic_ == LayerMajorPrefillProjectionTactic::
                                     kNativeNvfp4G2D2LargeMOperatorPanel) {
    switch (plan.full_attention_tactic_) {
      case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel:
        return kLayerMajorNativeNvfp4G2D2LargeMProjectionGroupQ64DeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ128V4Panel:
        return kLayerMajorNativeNvfp4G2D2LargeMProjectionGroupQ128V4DeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kNativeFlashInferExactPanel:
        return kLayerMajorNativeNvfp4G2D2LargeMProjectionFlashInferExactDeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512:
      default:
        return kLayerMajorNativeNvfp4G2D2LargeMProjectionDeploymentPlanId;
    }
  }
  if (plan.projection_tactic_ == LayerMajorPrefillProjectionTactic::
                                     kNativeNvfp4TrueLargeMOperatorPanel) {
    switch (plan.full_attention_tactic_) {
      case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel:
        return kLayerMajorNativeNvfp4TrueLargeMProjectionGroupQ64DeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ128V4Panel:
        return kLayerMajorNativeNvfp4TrueLargeMProjectionGroupQ128V4DeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kNativeFlashInferExactPanel:
        return kLayerMajorNativeNvfp4TrueLargeMProjectionFlashInferExactDeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512:
      default:
        return kLayerMajorNativeNvfp4TrueLargeMProjectionDeploymentPlanId;
    }
  }
  if (plan.projection_tactic_ == LayerMajorPrefillProjectionTactic::
                                     kNativeQuantizedLargeMOperatorPanel) {
    switch (plan.full_attention_tactic_) {
      case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel:
        return kLayerMajorNativeQuantizedLargeMProjectionGroupQ64DeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ128V4Panel:
        return kLayerMajorNativeQuantizedLargeMProjectionGroupQ128V4DeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kNativeFlashInferExactPanel:
        return kLayerMajorNativeQuantizedLargeMProjectionFlashInferExactDeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512:
      default:
        return kLayerMajorNativeQuantizedLargeMProjectionDeploymentPlanId;
    }
  }
  if (plan.projection_tactic_ ==
      LayerMajorPrefillProjectionTactic::kSegmentedMarlinOperatorPanel) {
    switch (plan.full_attention_tactic_) {
      case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel:
        return kLayerMajorSegmentedMarlinProjectionGroupQ64DeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ128V4Panel:
        return kLayerMajorSegmentedMarlinProjectionGroupQ128V4DeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kNativeFlashInferExactPanel:
        return kLayerMajorSegmentedMarlinProjectionFlashInferExactDeploymentPlanId;
      case LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512:
      default:
        return kLayerMajorSegmentedMarlinProjectionDeploymentPlanId;
    }
  }
  switch (plan.full_attention_tactic_) {
    case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel:
      return kLayerMajorNativeGroupQ64PanelDeploymentPlanId;
    case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ128V4Panel:
      return kLayerMajorNativeGroupQ128V4PanelDeploymentPlanId;
    case LayerMajorPrefillFullAttentionTactic::kNativeFlashInferExactPanel:
      return kLayerMajorNativeFlashInferExactPanelDeploymentPlanId;
    case LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512:
    default:
      return kLayerMajorOperatorPanelDeploymentPlanId;
  }
}

ReferenceWholeRequestPrefillOutcome ReferenceEnginePrefillExecutor::execute(
    const BoundPrefillExecutionPlan& plan, ReferenceRunner& runner,
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const PrefillExecutionPlan& geometry,
    const ReferenceWholeRequestPrefillOptions& options,
    std::optional<BoundPrefillRequestReceipt>& receipt) noexcept {
  if (receipt.has_value() || !plan_matches_runner(plan, runner) ||
      !is_valid_unbound_layer_major_prefill_execution_plan(geometry) ||
      geometry.mlp_schedule.tactic != plan.mlp_schedule_tactic_ ||
      (plan.mlp_schedule_tactic_ !=
           LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel &&
       (geometry.first_position != 0U ||
        geometry.prompt_token_count !=
            kLayerMajorPrefillLayerWideMlpP40Tokens ||
        geometry.mlp_schedule.mlp_phase_submission_count_per_layer != 1U)) ||
      (plan.mlp_schedule_tactic_ ==
           LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore &&
       g_reference_engine_prefill_compatibility_oracle_for_test)) {
    return execution_failure("bound_prefill_execute_authority");
  }
  receipt.emplace(BoundPrefillRequestReceipt(&plan, &runner, geometry));
  ReferenceWholeRequestPrefillOutcome outcome =
      g_reference_engine_prefill_compatibility_oracle_for_test
          ? runner.prefill_whole_request_layer_major_compatibility_core(
                input_token_ids, token_count, geometry, options)
          : runner.prefill_whole_request_layer_major_panel_core(
                input_token_ids, token_count, geometry,
                plan.projection_tactic_,
                plan.full_attention_tactic_, options);
  receipt->phase_ = outcome
                        ? BoundPrefillRequestReceipt::Phase::kAwaitingLogits
                        : BoundPrefillRequestReceipt::Phase::kPoisoned;
  return outcome;
}

ReferenceStepOutcome ReferenceEnginePrefillExecutor::finish(
    const BoundPrefillExecutionPlan& plan, ReferenceRunner& runner,
    BoundPrefillRequestReceipt& receipt,
    const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) noexcept {
  if (!plan_matches_runner(plan, runner) || receipt.plan_ != &plan ||
      receipt.runner_ != &runner ||
      receipt.phase_ !=
          BoundPrefillRequestReceipt::Phase::kAwaitingLogits) {
    receipt.phase_ = BoundPrefillRequestReceipt::Phase::kPoisoned;
    return runner.fail_step(
        status_failure("bound_prefill_finish_authority"));
  }
  ReferenceStepOutcome outcome =
      runner.finish_whole_request_compatibility_core(input_token_id,
                                                     options);
  receipt.phase_ = outcome
                        ? BoundPrefillRequestReceipt::Phase::kAwaitingCommit
                        : BoundPrefillRequestReceipt::Phase::kPoisoned;
  return outcome;
}

ReferenceRunnerStatus ReferenceEnginePrefillExecutor::commit(
    const BoundPrefillExecutionPlan& plan, ReferenceRunner& runner,
    BoundPrefillRequestReceipt& receipt,
    const PrefillExecutionPlan& geometry,
    const PrefillExecutionProgress& progress) noexcept {
  if (!plan_matches_runner(plan, runner) || receipt.plan_ != &plan ||
      receipt.runner_ != &runner ||
      receipt.phase_ !=
          BoundPrefillRequestReceipt::Phase::kAwaitingCommit ||
      !same_geometry(receipt.geometry_, geometry)) {
    receipt.phase_ = BoundPrefillRequestReceipt::Phase::kPoisoned;
    return runner.fail_whole_request_status(
        status_failure("bound_prefill_commit_authority"));
  }
  ReferenceRunnerStatus status =
      runner.commit_whole_request_layer_major_compatibility_core(
          geometry, progress);
  receipt.phase_ = status
                       ? BoundPrefillRequestReceipt::Phase::kConsumed
                       : BoundPrefillRequestReceipt::Phase::kPoisoned;
  return status;
}

}  // namespace q3x::runtime::reference_engine_detail
