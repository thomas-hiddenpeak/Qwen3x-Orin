#include "reference_engine_prefill_authority.h"

#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
#include "../kernels/sm87/gdn_prefill_chunk64_native_sm87.h"
#endif

#include <new>
#include <utility>

namespace q3x::runtime::reference_engine_detail {
namespace {

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
    const LinearWeight& weight) noexcept {
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
  return fp8 != nullptr && fp8->weight != nullptr &&
         fp8->weight_scale_device != nullptr &&
         fp8->input_scale_device != nullptr &&
         fp8->prefill_marlin_weight != nullptr &&
         fp8->prefill_marlin_scales != nullptr;
#else
  (void)weight;
  return false;
#endif
}

[[maybe_unused, nodiscard]] bool valid_nvfp4_marlin_binding(
    const LinearWeight& weight) noexcept {
#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
  const auto* const nvfp4 = std::get_if<NvFp4LinearWeight>(&weight);
  return nvfp4 != nullptr && nvfp4->packed_weight != nullptr &&
         nvfp4->block_scale != nullptr &&
         nvfp4->weight_scale_2_device != nullptr &&
         nvfp4->input_scale_device != nullptr &&
         nvfp4->prefill_marlin_weight != nullptr &&
         nvfp4->prefill_marlin_scales != nullptr &&
         nvfp4->prefill_marlin_global_scale != nullptr;
#else
  (void)weight;
  return false;
#endif
}

[[maybe_unused, nodiscard]] bool valid_bf16_binding(
    const LinearWeight& weight) noexcept {
  const auto* const bf16 = std::get_if<Bf16LinearWeight>(&weight);
  return bf16 != nullptr && bf16->weight != nullptr;
}

[[nodiscard]] bool complete_installed_projection_inventory(
    const ModelWeights& weights) noexcept {
#if !defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION) || \
    !defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
  (void)weights;
  return false;
#else
  for (std::size_t layer = 0U;
       layer < kReferenceDecoderLayerCount; ++layer) {
    const DecoderLayerWeights& layer_weights = weights.layer(layer);
    if (!valid_nvfp4_marlin_binding(layer_weights.mlp.gate_proj) ||
        !valid_nvfp4_marlin_binding(layer_weights.mlp.up_proj) ||
        !valid_nvfp4_marlin_binding(layer_weights.mlp.down_proj)) {
      return false;
    }
    const auto* const gate =
        std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.gate_proj);
    const auto* const up =
        std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.up_proj);
    if (gate == nullptr || up == nullptr ||
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
          !valid_fp8_marlin_binding(linear->in_proj_qkv) ||
          !valid_fp8_marlin_binding(linear->in_proj_z) ||
          !valid_bf16_binding(linear->in_proj_a) ||
          !valid_bf16_binding(linear->in_proj_b) ||
          !valid_fp8_marlin_binding(linear->out_proj) ||
          linear->conv1d.data == nullptr || linear->a_log.data == nullptr ||
          linear->dt_bias.data == nullptr || linear->norm.data == nullptr) {
        return false;
      }
    } else if (layer_type == model::LayerType::kFullAttention) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr ||
          !valid_fp8_marlin_binding(attention->q_proj) ||
          !valid_fp8_marlin_binding(attention->k_proj) ||
          !valid_fp8_marlin_binding(attention->v_proj) ||
          !valid_fp8_marlin_binding(attention->o_proj) ||
          attention->q_norm.data == nullptr ||
          attention->k_norm.data == nullptr) {
        return false;
      }
    } else {
      return false;
    }
    if (layer_weights.input_layernorm.data == nullptr ||
        layer_weights.post_attention_layernorm.data == nullptr) {
      return false;
    }
  }
  return weights.embed_tokens().weight != nullptr &&
         weights.final_norm().data != nullptr;
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
      left.final_commit.commit_count != right.final_commit.commit_count) {
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

}  // namespace

BoundPrefillExecutionPlan::BoundPrefillExecutionPlan(
    const ModelWeights* const weights, RequestState* const state,
    ReferenceRunner* const runner, const void* const arena_base,
    const std::uint64_t arena_bytes,
    const LayerMajorRequestMemoryPlan* const memory_plan,
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
    ReferenceRunner* const runner) noexcept {
  static_assert(kBoundPrefillSubmissionEventCount ==
                ReferenceRunner::kWholeRequestSubmissionWindowSlots);
  if (weights == nullptr || state == nullptr || runner == nullptr ||
      !static_cast<bool>(*state) || !static_cast<bool>(*runner)) {
    return plan_failure(BoundPrefillPlanError::kInvalidDependency,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_dependencies");
  }
  if (state->memory_profile() != RequestMemoryProfile::kLayerMajorC8192 ||
      state->layer_major_plan() == nullptr) {
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
          RequestMemoryProfile::kLayerMajorC8192 ||
      runner->layer_major_request_views_->descriptor
              .operator_panel_capacity_tokens !=
          kLayerMajorPrefillOperatorPanelTokens ||
      runner->layer_major_request_views_->prompt_residual_bf16.storage
              .device_data == nullptr ||
      runner->layer_major_request_views_->final_hidden_bf16.storage
              .device_data == nullptr) {
    return plan_failure(BoundPrefillPlanError::kIncompleteTypedViews,
                        ReferenceRunnerError::kInvalidRequestState,
                        "bound_prefill_typed_views");
  }
  if (!complete_installed_projection_inventory(*weights)) {
    return plan_failure(BoundPrefillPlanError::kUnsupportedBinary,
                        ReferenceRunnerError::kInvalidDependency,
                        "bound_prefill_installed_binary");
  }

  const ReferenceLayerMajorRequestViews& views =
      *runner->layer_major_request_views_;
  const DecoderLayerWeights& linear_layer = weights->layer(0U);
  const DecoderLayerWeights& full_layer = weights->layer(3U);
  const auto* const linear =
      std::get_if<LinearAttentionWeights>(&linear_layer.attention);
  const auto* const attention =
      std::get_if<FullAttentionWeights>(&full_layer.attention);
  const auto* const gate =
      std::get_if<NvFp4LinearWeight>(&linear_layer.mlp.gate_proj);
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
      down == nullptr || linear_qkv == nullptr || linear_z == nullptr ||
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
  if (runner->prefill_gdn_chunk64_native_workspace_ == nullptr ||
      exact_gdn_workspace_bytes == 0U ||
      runner->prefill_gdn_chunk64_native_workspace_bytes_ !=
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
      const void* const artifact, const void* const workspace) noexcept {
    return NativePrefillRoleReceipt{
        role, tactic, NativePrefillCompletionDomain::kMainStreamBarrier,
        artifact, workspace, 0U, kLayerMajorPrefillOperatorPanelTokens,
        0U, 0U};
  };
  std::array<NativePrefillRoleReceipt,
             kLayerMajorPrefillRequiredOperatorRoleCount>
      roles{};
  roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4GateUp)] =
      receipt(PrefillBindingRole::kNvfp4GateUp,
              NativePrefillTactic::kNvfp4GateUpCanonicalPanel,
              gate->prefill_marlin_weight,
              views.legacy_c512.projection_bf16[0U].storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kNvfp4Down)] =
      receipt(PrefillBindingRole::kNvfp4Down,
              NativePrefillTactic::kNvfp4DownCanonicalPanel,
              down->prefill_marlin_weight,
              views.legacy_c512.projection_bf16[2U].storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kLinearFp8Qkv)] =
      receipt(PrefillBindingRole::kLinearFp8Qkv,
              NativePrefillTactic::kFp8CanonicalPanel,
              linear_qkv->prefill_marlin_weight,
              views.legacy_c512.projection_bf16[0U].storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kLinearFp8Z)] =
      receipt(PrefillBindingRole::kLinearFp8Z,
              NativePrefillTactic::kFp8CanonicalPanel,
              linear_z->prefill_marlin_weight,
              views.legacy_c512.projection_bf16[1U].storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kLinearFp8O)] =
      receipt(PrefillBindingRole::kLinearFp8O,
              NativePrefillTactic::kFp8CanonicalPanel,
              linear_o->prefill_marlin_weight,
              views.legacy_c512.hidden_bf16[1U].storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8Q)] =
      receipt(PrefillBindingRole::kFullFp8Q,
              NativePrefillTactic::kFp8CanonicalPanel,
              full_q->prefill_marlin_weight,
              views.legacy_c512.projection_bf16[0U].storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8K)] =
      receipt(PrefillBindingRole::kFullFp8K,
              NativePrefillTactic::kFp8CanonicalPanel,
              full_k->prefill_marlin_weight,
              views.legacy_c512.projection_bf16[1U].storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8V)] =
      receipt(PrefillBindingRole::kFullFp8V,
              NativePrefillTactic::kFp8CanonicalPanel,
              full_v->prefill_marlin_weight,
              views.legacy_c512.projection_bf16[2U].storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kFullFp8O)] =
      receipt(PrefillBindingRole::kFullFp8O,
              NativePrefillTactic::kFp8CanonicalPanel,
              full_o->prefill_marlin_weight,
              views.legacy_c512.hidden_bf16[1U].storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kLinearBf16A)] =
      receipt(PrefillBindingRole::kLinearBf16A,
              NativePrefillTactic::kBf16CanonicalPanel, linear_a->weight,
              views.legacy_c512.linear_a_bf16.storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kLinearBf16B)] =
      receipt(PrefillBindingRole::kLinearBf16B,
              NativePrefillTactic::kBf16CanonicalPanel, linear_b->weight,
              views.legacy_c512.linear_b_bf16.storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kExactGdn)] =
      receipt(PrefillBindingRole::kExactGdn,
              NativePrefillTactic::kExactGdnChunk64Native,
              linear->conv1d.data,
              runner->prefill_gdn_chunk64_native_workspace_);
  roles[static_cast<std::size_t>(PrefillBindingRole::kExactGdn)]
      .workspace_bytes = exact_gdn_workspace_bytes;
  roles[static_cast<std::size_t>(PrefillBindingRole::kExactGdn)]
      .minimum_physical_m = 32U;
  roles[static_cast<std::size_t>(PrefillBindingRole::kExactGdn)]
      .maximum_physical_m = 512U;
  roles[static_cast<std::size_t>(
      PrefillBindingRole::kExactCausalAttention)] =
      receipt(PrefillBindingRole::kExactCausalAttention,
              NativePrefillTactic::kExactCausalAttentionCanonicalPanel,
              attention->q_norm.data,
              views.legacy_c512.fp32_scratch.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kResidual)] =
      receipt(PrefillBindingRole::kResidual,
              NativePrefillTactic::kResidualCanonicalPanel,
              weights, views.prompt_residual_bf16.storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kNormalization)] =
      receipt(PrefillBindingRole::kNormalization,
              NativePrefillTactic::kNormalizationCanonicalPanel,
              linear_layer.input_layernorm.data,
              views.legacy_c512.hidden_bf16[1U].storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kEmbedding)] =
      receipt(PrefillBindingRole::kEmbedding,
              NativePrefillTactic::kEmbeddingCanonicalPanel,
              weights->embed_tokens().weight,
              views.prompt_residual_bf16.storage.device_data);
  roles[static_cast<std::size_t>(PrefillBindingRole::kFinalHandoff)] =
      receipt(PrefillBindingRole::kFinalHandoff,
              NativePrefillTactic::kFinalHandoff,
              weights->final_norm().data,
              views.final_hidden_bf16.storage.device_data);

  for (std::size_t index = 0U; index < roles.size(); ++index) {
    const NativePrefillRoleReceipt& role = roles[index];
    if (role.role != static_cast<PrefillBindingRole>(index) ||
        role.maximum_logical_panel_m !=
            kLayerMajorPrefillOperatorPanelTokens ||
        role.artifact_owner == nullptr || role.workspace_owner == nullptr ||
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
  if (exact_gdn.tactic != NativePrefillTactic::kExactGdnChunk64Native ||
      exact_gdn.workspace_owner !=
          runner->prefill_gdn_chunk64_native_workspace_ ||
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
      state->layer_major_plan(), runner->stream_,
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
#if !defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  (void)plan;
  (void)runner;
  return false;
#else
  const NativePrefillRoleReceipt& exact_gdn =
      plan.roles_[static_cast<std::size_t>(PrefillBindingRole::kExactGdn)];
  return plan.runner_ == &runner && plan.weights_ == runner.weights_ &&
         plan.state_ == runner.state_ && plan.state_ != nullptr &&
         plan.arena_base_ == plan.state_->arena_data() &&
         plan.arena_bytes_ == plan.state_->arena_bytes() &&
         plan.memory_plan_ == plan.state_->layer_major_plan() &&
         plan.main_stream_ == runner.stream_ &&
         plan.auxiliary_stream_ == runner.prefill_auxiliary_stream_ &&
         plan.submission_events_[0U] ==
             runner.whole_request_submission_events_[0U] &&
         plan.submission_events_[1U] ==
             runner.whole_request_submission_events_[1U] &&
         runner.layer_major_request_views_.has_value() &&
         runner.projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
         exact_gdn.role == PrefillBindingRole::kExactGdn &&
         exact_gdn.tactic == NativePrefillTactic::kExactGdnChunk64Native &&
         exact_gdn.workspace_owner ==
             runner.prefill_gdn_chunk64_native_workspace_ &&
         exact_gdn.workspace_bytes ==
             runner.prefill_gdn_chunk64_native_workspace_bytes_ &&
         exact_gdn.workspace_bytes ==
             gdn_prefill_chunk64_native_detail::workspace_bytes() &&
         exact_gdn.minimum_physical_m == 32U &&
         exact_gdn.maximum_physical_m == 512U;
#endif
}

ReferenceWholeRequestPrefillOutcome ReferenceEnginePrefillExecutor::execute(
    const BoundPrefillExecutionPlan& plan, ReferenceRunner& runner,
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const PrefillExecutionPlan& geometry,
    const ReferenceWholeRequestPrefillOptions& options,
    std::optional<BoundPrefillRequestReceipt>& receipt) noexcept {
  if (receipt.has_value() || !plan_matches_runner(plan, runner) ||
      !is_valid_unbound_layer_major_prefill_execution_plan(geometry)) {
    return execution_failure("bound_prefill_execute_authority");
  }
  receipt.emplace(BoundPrefillRequestReceipt(&plan, &runner, geometry));
  ReferenceWholeRequestPrefillOutcome outcome =
      runner.prefill_whole_request_layer_major_compatibility_core(
          input_token_ids, token_count, geometry, options);
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
