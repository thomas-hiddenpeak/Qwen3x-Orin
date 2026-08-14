#include "sm87_macrofeed_v4_p40_startup_package_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail {
namespace {

using Role = kernels::Sm87TargetAotProjectionRole;
using Encoding = kernels::Sm87TargetAotProjectionEncoding;
using Error = Sm87MacroFeedV4P40StartupPackageError;
using Status = Sm87MacroFeedV4P40StartupPackageStatus;
using Package = Sm87MacroFeedV4P40StartupPackage;
using CreateResult = Sm87MacroFeedV4P40StartupPackageCreateResult;

inline constexpr std::uint64_t kGateUpSealIssuerNonce =
    0x5133'4d46'5634'474eULL;
inline constexpr std::uint64_t kDownSealIssuerNonce =
    0x5133'4d46'5634'444eULL;

[[nodiscard]] constexpr std::array<Role, 4U> layer_roles(
    const std::size_t layer_index) noexcept {
  return {{Role::kNvFp4GateUp, Role::kNvFp4Down,
           sm87_target_aot_complete_is_full_layer(layer_index)
               ? Role::kFp8FullQkv
               : Role::kFp8GdnQkvZ,
           Role::kFp8AttentionOutput}};
}

[[nodiscard]] Status failure(
    const Error error, const char* const context, const int cuda_error = 0,
    const std::size_t layer = kSm87MacroFeedV4P40StartupPackageLayers,
    const Role role = Role::kInvalid) noexcept {
  return {error, context, layer, role, cuda_error};
}

[[nodiscard]] constexpr std::uint64_t mix(
    std::uint64_t hash, const std::uint64_t value) noexcept {
  hash ^= value + 0x9e37'79b9'7f4a'7c15ULL + (hash << 6U) + (hash >> 2U);
  return hash;
}

[[nodiscard]] std::uint64_t mix_string(
    std::uint64_t hash, const std::string_view value) noexcept {
  hash = mix(hash, value.size());
  for (const char byte : value) {
    hash = mix(hash, static_cast<std::uint8_t>(byte));
  }
  return hash;
}

[[nodiscard]] std::uint64_t mix_digest(
    std::uint64_t hash,
    const kernels::Sm87TargetAotProjectionSha256Digest& digest) noexcept {
  for (const std::uint8_t byte : digest.bytes) {
    hash = mix(hash, byte);
  }
  return hash;
}

template <typename UploadReceipt>
[[nodiscard]] bool authenticated_upload_complete(
    const UploadReceipt& upload, const std::uint64_t owner_identity,
    const std::uint64_t allocation_identity,
    const std::int32_t device_ordinal, const std::uintptr_t payload_begin,
    const std::uintptr_t payload_end,
    const std::uint64_t payload_bytes) noexcept {
  return upload.receipt_identity != 0U && owner_identity != 0U &&
         allocation_identity != 0U && device_ordinal >= 0 &&
         upload.device_allocation_owner_identity == owner_identity &&
         upload.device_allocation_identity == allocation_identity &&
         upload.device_ordinal == device_ordinal &&
         upload.device_payload_begin == payload_begin &&
         upload.device_payload_end == payload_end &&
         upload.device_payload_bytes == payload_bytes &&
         upload.host_payload_digest_verified_before_copy &&
         upload.host_payload_immutable_until_completion &&
         upload.copy_enqueued_to_exact_payload_range &&
         upload.completion_event_recorded_after_copy &&
         upload.completion_event_observed && upload.upload_completed &&
         upload.verification_copy_enqueued_from_exact_payload_range &&
         upload.verification_event_recorded_after_copy &&
         upload.verification_event_observed &&
         upload.verification_completed &&
         upload.device_payload_matches_host_payload &&
         upload.allocation_retained_for_asset_lifetime;
}

[[nodiscard]] constexpr std::uint64_t expected_consumer_tactic_identity(
    const Role role) noexcept {
  return role == Role::kNvFp4GateUp
             ? static_cast<std::uint64_t>(
                   kernels::kSm87MacroFeedV4NvFp4GateUpIdentity)
             : (role == Role::kNvFp4Down
                    ? static_cast<std::uint64_t>(
                          kernels::kSm87MacroFeedV4NvFp4DownIdentity)
                    : 0U);
}

[[nodiscard]] std::uint64_t gate_up_seal_identity(
    const Sm87MacroFeedV4GateUpStartupSeal& seal) noexcept {
  const auto& resources = seal.resources;
  if (seal.package_identity == 0U ||
      seal.deployment_plan_identity == 0U ||
      !seal.canonical_c8000_plan || !seal.issued_by_v4_package ||
      seal.caller_receipt_accepted || seal.launcher_authority ||
      seal.production_dispatch_eligible ||
      !resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(resources)) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5634'4753ULL;
  identity = mix(identity, seal.package_identity);
  identity = mix(identity, seal.deployment_plan_identity);
  identity = mix(identity, static_cast<std::uint64_t>(resources.identity));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(resources.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(resources.binary_version));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.registers_per_thread));
  identity = mix(identity, resources.static_shared_bytes);
  identity = mix(identity, resources.dynamic_shared_bytes);
  identity = mix(identity, resources.local_bytes);
  identity = mix(
      identity, static_cast<std::uint64_t>(resources.maximum_threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.active_blocks_per_sm));
  identity = mix(identity, resources.kernel_compiled);
  identity = mix(identity, resources.static_resource_gate_passed);
  identity = mix(identity, resources.numerical_contract_qualified);
  identity = mix(identity, resources.production_dispatch_eligible);
  identity = mix(identity, seal.canonical_c8000_plan);
  identity = mix(identity, seal.issued_by_v4_package);
  identity = mix(identity, seal.caller_receipt_accepted);
  identity = mix(identity, seal.launcher_authority);
  identity = mix(identity, seal.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5634'4753ULL : identity;
}

[[nodiscard]] std::uint64_t down_seal_identity(
    const Sm87MacroFeedV4DownStartupSeal& seal) noexcept {
  const auto& resources = seal.resources;
  if (seal.package_identity == 0U ||
      seal.deployment_plan_identity == 0U ||
      !seal.canonical_c8000_plan || !seal.issued_by_v4_package ||
      seal.caller_receipt_accepted || seal.launcher_authority ||
      seal.production_dispatch_eligible ||
      !resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(resources)) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5634'4453ULL;
  identity = mix(identity, seal.package_identity);
  identity = mix(identity, seal.deployment_plan_identity);
  identity = mix(identity, static_cast<std::uint64_t>(resources.identity));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(resources.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(resources.binary_version));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.registers_per_thread));
  identity = mix(identity, resources.static_shared_bytes);
  identity = mix(identity, resources.dynamic_shared_bytes);
  identity = mix(identity, resources.local_bytes);
  identity = mix(
      identity, static_cast<std::uint64_t>(resources.maximum_threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.active_blocks_per_sm));
  identity = mix(identity, resources.shared_bytes_per_sm);
  identity = mix(identity, resources.optin_shared_bytes_per_block);
  identity = mix(identity, resources.kernel_compiled);
  identity = mix(identity, resources.static_resource_gate_passed);
  identity = mix(identity, resources.numerical_contract_qualified);
  identity = mix(identity, resources.production_dispatch_eligible);
  identity = mix(identity, seal.canonical_c8000_plan);
  identity = mix(identity, seal.issued_by_v4_package);
  identity = mix(identity, seal.caller_receipt_accepted);
  identity = mix(identity, seal.launcher_authority);
  identity = mix(identity, seal.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5634'4453ULL : identity;
}

}  // namespace

bool Sm87MacroFeedV4GateUpStartupSeal::valid() const noexcept {
  return issuer_nonce_ == kGateUpSealIssuerNonce && seal_identity != 0U &&
         seal_identity == gate_up_seal_identity(*this);
}

bool Sm87MacroFeedV4DownStartupSeal::valid() const noexcept {
  return issuer_nonce_ == kDownSealIssuerNonce && seal_identity != 0U &&
         seal_identity == down_seal_identity(*this);
}

Sm87MacroFeedV4P40StartupPackage::Sm87MacroFeedV4P40StartupPackage(
    ProjectionAccess access,
    std::array<AssetCapability, kSm87MacroFeedV4P40StartupPackageArtifacts>
        capabilities,
    Sm87MacroFeedV4PanelWavefrontPlan plan, StartupSeals seals,
    Sm87MacroFeedV4P40StartupPackageAudit audit) noexcept
    : projection_access_(std::move(access)),
      capabilities_(std::move(capabilities)),
      plan_(std::move(plan)),
      seals_(std::move(seals)),
      audit_(audit) {}

Sm87MacroFeedV4ProjectionStartupBinding::
    Sm87MacroFeedV4ProjectionStartupBinding(
        ProjectionAccess access, ProjectionAsset asset,
        Snapshot snapshot) noexcept
    : projection_access_(std::move(access)),
      asset_(std::move(asset)),
      snapshot_(std::move(snapshot)) {}

Sm87MacroFeedV4P40StartupPackageCreateResult::operator bool()
    const noexcept {
  return package != nullptr && static_cast<bool>(status) && audit.valid() &&
         package->valid() &&
         package->audit().package_identity == audit.package_identity;
}

Sm87MacroFeedV4P40StartupPackageCreateResult
Sm87MacroFeedV4P40StartupPackage::create(
    const ModelWeights& model_weights) noexcept {
#if !defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  (void)model_weights;
  CreateResult result;
  result.status = failure(Error::kAdmissionDisabled, "admission_disabled");
  return result;
#else
  auto plan = make_sm87_macrofeed_v4_p40_panel_wavefront_plan();
  if (!sm87_macrofeed_v4_p40_panel_wavefront_plan_valid(plan) ||
      compute_deployment_plan_identity(plan) == 0U) {
    CreateResult result;
    result.status = failure(Error::kCanonicalPlan, "canonical_v4_plan");
    return result;
  }

  auto access = ProjectionAccess::bind(model_weights);
  if (!access) {
    CreateResult result;
    result.status =
        failure(Error::kProjectionAccessBind, "projection_access_bind");
    return result;
  }

  kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources gate_up;
  int status =
      kernels::query_sm87_macrofeed_v4_nvfp4_gate_up_cuda_resources(&gate_up);
  if (status != 0 || !gate_up.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(gate_up)) {
    CreateResult result;
    result.status = failure(Error::kGateUpStartupSeal,
                            "gate_up_startup_resource_seal", status);
    return result;
  }

  kernels::Sm87MacroFeedV4NvFp4DownCudaResources down;
  status = kernels::query_sm87_macrofeed_v4_nvfp4_down_cuda_resources(&down);
  if (status != 0 || !down.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(down)) {
    CreateResult result;
    result.status = failure(Error::kDownStartupSeal,
                            "down_startup_resource_seal", status);
    return result;
  }
  return build_from_private_authority(std::move(*access), std::move(plan),
                                      gate_up, down);
#endif
}

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_deployment_plan_identity(
        const Sm87MacroFeedV4PanelWavefrontPlan& plan) noexcept {
  if (!sm87_macrofeed_v4_p40_panel_wavefront_plan_valid(plan)) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5634'504cULL;
  for (const std::uint8_t byte : plan.magic) {
    identity = mix(identity, byte);
  }
  identity = mix(identity, plan.abi_major);
  identity = mix(identity, plan.abi_minor);
  identity = mix_string(identity, plan.candidate_id);
  identity = mix_string(identity, plan.deployment_plan_id);
  identity = mix_string(identity, plan.api.route_id);
  identity = mix_string(identity, plan.api.endpoint);
  identity = mix_string(identity, plan.api.served_model);
  identity = mix(identity, plan.api.prompt_tokens);
  identity = mix(identity, plan.api.maximum_output_tokens);
  identity = mix(identity, plan.api.batch_size);
  identity = mix(identity, plan.api.openai_compatible);
  identity = mix(identity, plan.api.exact_token_ids);
  identity = mix(identity, plan.api.cold_request);
  identity = mix(identity, plan.api.prefix_cache_disabled);
  identity = mix(identity, plan.api.kv_reuse_disabled);
  identity = mix(identity, plan.api.streaming_first_committed_token);
  identity = mix(identity, plan.api.full_prompt_consumption_required);
  identity = mix(identity, static_cast<std::uint64_t>(plan.traversal));
  identity = mix(identity, plan.prompt_tokens);
  identity = mix(identity, plan.panel_tokens);
  identity = mix(identity, plan.panel_count);
  identity = mix(identity, plan.layer_count);
  for (const auto& buffer : plan.workspace.buffers) {
    identity = mix(identity, static_cast<std::uint64_t>(buffer.role));
    identity = mix(identity, buffer.storage_identity);
    identity = mix(identity, buffer.offset);
    identity = mix(identity, buffer.bytes);
    identity = mix(identity, buffer.token_capacity);
    identity = mix(identity, buffer.row_width);
    identity = mix(identity, buffer.panel_local);
    identity = mix(identity, buffer.reuse_waits_for_completion);
  }
  identity = mix(identity, plan.workspace.transient_arena_bytes);
  identity = mix(identity, plan.workspace.maximum_temporary_tokens);
  identity = mix(identity, plan.workspace.ping_pong_hidden);
  identity = mix(identity, plan.workspace.scratch_reused_by_phase);
  identity = mix(identity, plan.workspace.full_p40_temporary_plane_allowed);
  identity = mix(
      identity, plan.workspace.persistent_kv_is_outside_transient_arena);
  identity = mix(
      identity,
      plan.workspace.persistent_conv_gdn_state_is_outside_transient_arena);
  identity = mix(
      identity,
      plan.phase_aliasing.attention_q_preprocess_overwrites_raw_q_slots);
  identity = mix(
      identity,
      plan.phase_aliasing.attention_online_core_reuses_processed_q);
  identity = mix(
      identity,
      plan.phase_aliasing.attention_interleaved_q_gate_layout_retained);
  identity = mix(
      identity,
      plan.phase_aliasing.attention_result_overwrites_q_slots_in_place);
  identity = mix(
      identity, plan.phase_aliasing.attention_gate_slots_remain_in_place);
  identity = mix(
      identity,
      plan.phase_aliasing
          .attention_output_projection_gathers_interleaved_q_slots);
  identity = mix(identity,
                 plan.phase_aliasing.gdn_recurrent_reuses_consumed_qkv);
  identity = mix(identity,
                 plan.phase_aliasing.gate_up_activation_owns_panel_scratch);
  identity = mix(identity,
                 plan.phase_aliasing.every_phase_fits_one_panel_scratch);
  identity = mix(identity,
                 plan.state_ownership.recurrent_epoch_bank_count);
  identity = mix(identity, plan.state_ownership.recurrent_epoch_bytes);
  identity = mix(identity, plan.state_ownership.recurrent_storage_bytes);
  identity = mix(
      identity, plan.state_ownership.active_recurrent_storage_identity);
  identity = mix(
      identity, plan.state_ownership.candidate_recurrent_storage_identity);
  identity = mix(
      identity, plan.state_ownership.private_kv_valid_end_storage_identity);
  identity = mix(identity,
                 plan.state_ownership.panel_commit_event_identity);
  identity = mix(identity,
                 plan.state_ownership.final_publish_event_identity);
  identity = mix(identity, plan.state_ownership.private_kv_valid_end);
  identity = mix(
      identity,
      plan.state_ownership
          .conv_history_copies_active_to_candidate_per_layer);
  identity = mix(
      identity,
      plan.state_ownership.gdn_state_writes_active_to_candidate_per_layer);
  identity = mix(
      identity,
      plan.state_ownership.candidate_epoch_fully_assigned_before_swap);
  identity = mix(
      identity,
      plan.state_ownership.whole_recurrent_epoch_copy_before_panel_allowed);
  identity = mix(
      identity, plan.state_ownership.active_candidate_swap_after_layer_63);
  identity = mix(
      identity, plan.state_ownership.panel_failure_discards_candidate_epoch);
  identity = mix(
      identity,
      plan.state_ownership.canonical_recurrent_publish_after_final_panel);
  identity = mix(
      identity,
      plan.state_ownership.sequence_length_is_final_visibility_fence);
  identity = mix(
      identity,
      plan.state_ownership.no_fallible_work_after_sequence_publication);
  for (const auto& panel : plan.panels) {
    identity = mix(identity, panel.panel_index + 1U);
    identity = mix(identity, panel.token_begin);
    identity = mix(identity, panel.token_count);
    identity = mix(identity, panel.sequence_begin);
    identity = mix(identity, panel.sequence_end);
    identity = mix(
        identity,
        static_cast<std::uint64_t>(panel.initial_workspace));
    identity = mix(identity,
                   static_cast<std::uint64_t>(panel.final_workspace));
    identity = mix(identity, panel.embedding_publishes_initial_workspace);
    identity = mix(identity, panel.workspace_reuse_waits_for_panel_commit);
    identity = mix(identity, panel.state_transaction.panel_index + 1U);
    identity = mix(identity, panel.state_transaction.token_begin);
    identity = mix(identity, panel.state_transaction.token_end);
    identity = mix(identity,
                   panel.state_transaction.incoming_state_epoch + 1U);
    identity = mix(identity,
                   panel.state_transaction.outgoing_state_epoch + 1U);
    identity = mix(
        identity,
        panel.state_transaction.commit_dependency_sequence_ordinal + 1U);
    identity = mix(identity, panel.state_transaction.kv_layer_count);
    identity = mix(identity, panel.state_transaction.conv_layer_count);
    identity = mix(identity, panel.state_transaction.gdn_layer_count);
    identity = mix(
        identity,
        panel.state_transaction.kv_uses_disjoint_final_token_slice);
    identity = mix(
        identity,
        panel.state_transaction.conv_and_gdn_use_private_next_epoch);
    identity = mix(identity,
                   panel.state_transaction.atomic_kv_conv_gdn_commit);
    identity = mix(identity, panel.state_transaction.commit_after_layer_63);
    identity = mix(identity,
                   panel.state_transaction.next_panel_waits_for_commit);
    identity = mix(
        identity,
        panel.state_transaction.rollback_discards_uncommitted_panel_state);
    identity = mix(
        identity,
        panel.state_transaction.state_private_to_prefill_until_request_commit);
    identity = mix(identity, panel.state_transaction.state_visible_to_decode);
    for (const auto& layer : panel.layers) {
      identity = mix(identity, layer.panel_index + 1U);
      identity = mix(identity, layer.sequence_ordinal + 1U);
      identity = mix(identity, layer.layer_index + 1U);
      identity = mix(identity, layer.token_begin);
      identity = mix(identity, layer.token_count);
      identity = mix(identity,
                     static_cast<std::uint64_t>(layer.layer_kind));
      identity = mix(identity,
                     static_cast<std::uint64_t>(layer.input_workspace));
      identity = mix(identity,
                     static_cast<std::uint64_t>(layer.output_workspace));
      identity = mix(identity,
                     static_cast<std::uint64_t>(layer.state_write_mode));
      identity = mix(
          identity, layer.input_consumed_before_output_publication);
      identity = mix(identity, layer.output_reuse_waits_for_completion);
      identity = mix(identity, layer.stages_kv);
      identity = mix(identity, layer.stages_conv_state);
      identity = mix(identity, layer.stages_gdn_state);
      identity = mix(identity, layer.publishes_state_to_decode);
    }
  }
  identity = mix(identity, plan.panel_loop_is_outermost);
  identity = mix(identity, plan.layer_loop_is_natural_order_innermost);
  identity = mix(identity, plan.final_request_commit_after_all_panels);
  identity = mix(identity, plan.partial_panel_commit_visible_to_decode);
  identity = mix(identity, plan.route.sm87_only);
  identity = mix(identity, plan.route.real_checkpoint_required);
  identity = mix(identity, plan.route.authenticated_aot_deployment_plan);
  identity = mix(identity, plan.route.startup_bound_tactics);
  identity = mix(identity, plan.route.request_time_jit_allowed);
  identity = mix(identity, plan.route.request_time_repack_allowed);
  identity = mix(identity, plan.route.request_time_autotune_allowed);
  identity = mix(identity, plan.route.fallback_allowed);
  identity = mix(identity, plan.route.cublaslt_allowed);
  identity = mix(identity, plan.route.mtp_allowed);
  identity = mix(identity, plan.route.approximate_numerics_allowed);
  identity = mix(identity, plan.route.default_off);
  identity = mix(identity, plan.route.test_only_contract);
  identity = mix(identity, plan.route.selector_bound);
  identity = mix(identity, plan.route.launcher_present);
  identity = mix(identity, plan.route.production_dispatch_eligible);
  identity = mix(identity, plan.route.numerical_qualification_complete);
  return identity == 0U ? 0x5133'4d46'5634'504cULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::compute_package_identity(
    const ProjectionAccess& access,
    const std::array<AssetCapability,
                     kSm87MacroFeedV4P40StartupPackageArtifacts>&
        capabilities,
    const Sm87MacroFeedV4PanelWavefrontPlan& plan,
    const kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources& gate_up,
    const kernels::Sm87MacroFeedV4NvFp4DownCudaResources& down,
    const std::size_t sources) noexcept {
  const std::uint64_t plan_identity =
      compute_deployment_plan_identity(plan);
  const std::uint64_t catalog_identity = access.catalog_identity();
  if (!access.attached() || access.owner_identity() == 0U ||
      access.allocation_identity() == 0U || access.device_identity() == 0U ||
      access.device_ordinal() < 0 || catalog_identity == 0U ||
      plan_identity == 0U ||
      sources != kSm87MacroFeedV4P40StartupPackageSources ||
      !gate_up.static_resource_gate_passed ||
      !down.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(gate_up) ||
      !kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(down) ||
      gate_up.device_ordinal != access.device_ordinal() ||
      down.device_ordinal != access.device_ordinal()) {
    return 0U;
  }

  std::uint64_t identity = 0x5133'4d46'5634'504bULL;
  for (const std::uint8_t byte : kSm87MacroFeedV4P40StartupPackageMagic) {
    identity = mix(identity, byte);
  }
  identity = mix(identity, kSm87MacroFeedV4P40StartupPackageAbiMajor);
  identity = mix(identity, kSm87MacroFeedV4P40StartupPackageAbiMinor);
  identity = mix(identity, plan_identity);
  identity = mix(identity, access.owner_identity());
  identity = mix(identity, access.allocation_identity());
  identity = mix(identity, catalog_identity);
  identity = mix(identity, access.device_identity());
  identity = mix(identity,
                 static_cast<std::uint64_t>(access.device_ordinal() + 1));
  identity = mix(identity, capabilities.size());
  identity = mix(identity, sources);
  for (std::size_t index = 0U; index < capabilities.size(); ++index) {
    const auto& capability = capabilities[index];
    if (!capability.asset || capability.artifact_identity == 0U ||
        capability.source_inventory_identity == 0U ||
        capability.manifest_seal == 0U ||
        capability.upload_receipt_identity == 0U ||
        kernels::sm87_target_aot_projection_digest_is_zero(
            capability.payload_digest) ||
        capability.payload_begin == 0U || capability.payload_bytes == 0U ||
        capability.payload_end <= capability.payload_begin ||
        capability.source_count == 0U) {
      return 0U;
    }
    identity = mix(identity, index + 1U);
    identity = mix(identity, capability.layer_index + 1U);
    identity = mix(identity, static_cast<std::uint64_t>(capability.role));
    identity = mix(identity,
                   static_cast<std::uint64_t>(capability.encoding));
    identity = mix(identity, capability.artifact_identity);
    identity = mix(identity, capability.source_inventory_identity);
    identity = mix(identity, capability.manifest_seal);
    identity = mix(identity, capability.upload_receipt_identity);
    identity = mix_digest(identity, capability.payload_digest);
    identity = mix(identity, capability.payload_begin);
    identity = mix(identity, capability.payload_end);
    identity = mix(identity, capability.payload_bytes);
    identity = mix(identity, capability.source_count);
    for (const std::uint32_t bits : capability.tensor_scale_bits) {
      identity = mix(identity, bits);
    }
  }
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.identity));
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.binary_version));
  identity = mix(identity,
                 static_cast<std::uint64_t>(gate_up.registers_per_thread));
  identity = mix(identity, gate_up.static_shared_bytes);
  identity = mix(identity, gate_up.dynamic_shared_bytes);
  identity = mix(identity, gate_up.local_bytes);
  identity = mix(
      identity,
      static_cast<std::uint64_t>(gate_up.maximum_threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(gate_up.active_blocks_per_sm));
  identity = mix(identity, gate_up.kernel_compiled);
  identity = mix(identity, gate_up.static_resource_gate_passed);
  identity = mix(identity, gate_up.numerical_contract_qualified);
  identity = mix(identity, gate_up.production_dispatch_eligible);
  identity = mix(identity, static_cast<std::uint64_t>(down.identity));
  identity = mix(identity, static_cast<std::uint64_t>(down.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(down.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(down.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(down.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(down.binary_version));
  identity = mix(identity,
                 static_cast<std::uint64_t>(down.registers_per_thread));
  identity = mix(identity, down.static_shared_bytes);
  identity = mix(identity, down.dynamic_shared_bytes);
  identity = mix(identity, down.local_bytes);
  identity = mix(
      identity,
      static_cast<std::uint64_t>(down.maximum_threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(down.active_blocks_per_sm));
  identity = mix(identity, down.shared_bytes_per_sm);
  identity = mix(identity, down.optin_shared_bytes_per_block);
  identity = mix(identity, down.kernel_compiled);
  identity = mix(identity, down.static_resource_gate_passed);
  identity = mix(identity, down.numerical_contract_qualified);
  identity = mix(identity, down.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5634'504bULL : identity;
}

Sm87MacroFeedV4P40StartupPackage::StartupSeals
Sm87MacroFeedV4P40StartupPackage::mint_startup_seals(
    const std::uint64_t package_identity,
    const std::uint64_t plan_identity,
    const kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources& gate_up,
    const kernels::Sm87MacroFeedV4NvFp4DownCudaResources& down) noexcept {
  StartupSeals seals;
  seals.gate_up.package_identity = package_identity;
  seals.gate_up.deployment_plan_identity = plan_identity;
  seals.gate_up.resources = gate_up;
  seals.gate_up.canonical_c8000_plan = true;
  seals.gate_up.issued_by_v4_package = true;
  seals.gate_up.caller_receipt_accepted = false;
  seals.gate_up.launcher_authority = false;
  seals.gate_up.production_dispatch_eligible = false;
  seals.gate_up.issuer_nonce_ = kGateUpSealIssuerNonce;
  seals.gate_up.seal_identity = gate_up_seal_identity(seals.gate_up);

  seals.down.package_identity = package_identity;
  seals.down.deployment_plan_identity = plan_identity;
  seals.down.resources = down;
  seals.down.canonical_c8000_plan = true;
  seals.down.issued_by_v4_package = true;
  seals.down.caller_receipt_accepted = false;
  seals.down.launcher_authority = false;
  seals.down.production_dispatch_eligible = false;
  seals.down.issuer_nonce_ = kDownSealIssuerNonce;
  seals.down.seal_identity = down_seal_identity(seals.down);
  return seals;
}

bool Sm87MacroFeedV4P40StartupPackage::startup_seals_valid(
    const StartupSeals& seals, const std::uint64_t package_identity,
    const std::uint64_t plan_identity,
    const std::int32_t device_ordinal) noexcept {
  return package_identity != 0U && plan_identity != 0U &&
         device_ordinal >= 0 && seals.gate_up.valid() &&
         seals.down.valid() &&
         seals.gate_up.package_identity == package_identity &&
         seals.down.package_identity == package_identity &&
         seals.gate_up.deployment_plan_identity == plan_identity &&
         seals.down.deployment_plan_identity == plan_identity &&
         seals.gate_up.resources.device_ordinal == device_ordinal &&
         seals.down.resources.device_ordinal == device_ordinal;
}

Sm87MacroFeedV4P40StartupPackageCreateResult
Sm87MacroFeedV4P40StartupPackage::build_from_private_authority(
    ProjectionAccess access, Sm87MacroFeedV4PanelWavefrontPlan plan,
    kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources gate_up,
    kernels::Sm87MacroFeedV4NvFp4DownCudaResources down) noexcept {
  CreateResult result;
  if (!access.attached() ||
      access.artifact_count() != kSm87MacroFeedV4P40StartupPackageArtifacts) {
    result.status =
        failure(Error::kProjectionAttachment, "projection_attachment");
    return result;
  }
  const std::uint64_t owner_identity = access.owner_identity();
  const std::uint64_t allocation_identity = access.allocation_identity();
  const std::uint64_t catalog_identity = access.catalog_identity();
  const std::uint64_t device_identity = access.device_identity();
  const std::int32_t device_ordinal = access.device_ordinal();
  const std::uint64_t plan_identity =
      compute_deployment_plan_identity(plan);
  if (owner_identity == 0U || allocation_identity == 0U ||
      catalog_identity == 0U || device_identity == 0U ||
      device_ordinal < 0 || plan_identity == 0U) {
    result.status = failure(Error::kProjectionCatalog, "projection_catalog");
    return result;
  }
  if (gate_up.device_ordinal != device_ordinal ||
      down.device_ordinal != device_ordinal) {
    result.status = failure(Error::kDeviceMismatch, "startup_seal_device");
    return result;
  }

  std::array<AssetCapability, kSm87MacroFeedV4P40StartupPackageArtifacts>
      capabilities{};
  std::array<std::uint64_t, kSm87MacroFeedV4P40StartupPackageArtifacts>
      artifact_identities{};
  std::array<std::uint64_t, kSm87MacroFeedV4P40StartupPackageArtifacts>
      inventory_identities{};
  std::size_t artifacts = 0U;
  std::size_t sources = 0U;
  std::size_t gate_up_assets = 0U;
  std::size_t down_assets = 0U;
  std::size_t gdn_assets = 0U;
  std::size_t full_assets = 0U;
  std::size_t output_assets = 0U;

  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV4P40StartupPackageLayers;
       ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != artifacts || index >= capabilities.size()) {
        result.status = failure(Error::kProjectionInventory,
                                "projection_descriptor_order", 0,
                                layer_index, role);
        return result;
      }
      auto asset = access.resolve(layer_index, role);
      const auto layout =
          kernels::sm87_target_aot_projection_packed_layout(role);
      if (!asset || !layout.valid() ||
          asset->layer_index() != layer_index || asset->role() != role ||
          asset->artifact_identity() == 0U ||
          asset->source_inventory_identity() == 0U ||
          asset->encoding() != layout.encoding ||
          asset->payload_bytes() != layout.payload_bytes ||
          std::find(artifact_identities.begin(),
                    artifact_identities.begin() + artifacts,
                    asset->artifact_identity()) !=
              artifact_identities.begin() + artifacts ||
          std::find(inventory_identities.begin(),
                    inventory_identities.begin() + artifacts,
                    asset->source_inventory_identity()) !=
              inventory_identities.begin() + artifacts) {
        result.status = failure(Error::kProjectionInventory,
                                "projection_asset_identity", 0,
                                layer_index, role);
        return result;
      }

      AssetCapability capability;
      capability.layer_index = layer_index;
      capability.role = role;
      capability.encoding = asset->encoding();
      capability.artifact_identity = asset->artifact_identity();
      capability.source_inventory_identity =
          asset->source_inventory_identity();
      capability.payload_bytes = asset->payload_bytes();

      bool typed_borrow_valid = false;
      if (sm87_target_aot_complete_role_is_nvfp4(role)) {
        const auto* const view = asset->borrow_nvfp4_cuda_asset();
        if (view != nullptr && asset->borrow_fp8_cuda_asset() == nullptr &&
            kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*view) &&
            view->artifact_identity == capability.artifact_identity &&
            view->source_inventory_identity ==
                capability.source_inventory_identity &&
            view->tensor_scale_count == layout.partition_count &&
            view->tensor_scale_count <= capability.tensor_scale_bits.size() &&
            authenticated_upload_complete(
                view->device_upload_receipt, owner_identity,
                allocation_identity, device_ordinal, view->payload.begin,
                view->payload.end, view->payload.bytes)) {
          capability.manifest_seal = view->host_manifest_seal.value;
          capability.upload_receipt_identity =
              view->device_upload_receipt.receipt_identity;
          capability.payload_digest = view->host_payload_digest;
          capability.payload_begin = view->payload.begin;
          capability.payload_end = view->payload.end;
          capability.source_count = view->tensor_scale_count;
          for (std::size_t source = 0U;
               source < capability.source_count; ++source) {
            capability.tensor_scale_bits[source] =
                view->tensor_scale_bits[source];
          }
          typed_borrow_valid = true;
        }
      } else if (sm87_target_aot_complete_role_is_fp8(role)) {
        const auto* const view = asset->borrow_fp8_cuda_asset();
        if (view != nullptr && asset->borrow_nvfp4_cuda_asset() == nullptr &&
            kernels::sm87_target_aot_fp8_cuda_asset_valid(*view) &&
            view->artifact_identity == capability.artifact_identity &&
            view->source_inventory_identity ==
                capability.source_inventory_identity &&
            view->tensor_scale_count == layout.partition_count &&
            view->tensor_scale_count <= capability.tensor_scale_bits.size() &&
            authenticated_upload_complete(
                view->device_upload_receipt, owner_identity,
                allocation_identity, device_ordinal, view->payload.begin,
                view->payload.end, view->payload.bytes)) {
          capability.manifest_seal = view->host_manifest_seal.value;
          capability.upload_receipt_identity =
              view->device_upload_receipt.receipt_identity;
          capability.payload_digest = view->host_payload_digest;
          capability.payload_begin = view->payload.begin;
          capability.payload_end = view->payload.end;
          capability.source_count = view->tensor_scale_count;
          for (std::size_t source = 0U;
               source < capability.source_count; ++source) {
            capability.tensor_scale_bits[source] =
                view->tensor_scale_bits[source];
          }
          typed_borrow_valid = true;
        }
      }
      if (!typed_borrow_valid || capability.manifest_seal == 0U ||
          capability.upload_receipt_identity == 0U ||
          kernels::sm87_target_aot_projection_digest_is_zero(
              capability.payload_digest) ||
          capability.payload_begin == 0U ||
          capability.payload_end <= capability.payload_begin ||
          capability.payload_end - capability.payload_begin !=
              capability.payload_bytes ||
          capability.source_count != layout.partition_count) {
        result.status = failure(Error::kProjectionAssetBorrow,
                                "projection_asset_borrow", 0,
                                layer_index, role);
        return result;
      }

      capability.asset = std::move(*asset);
      capabilities[index] = std::move(capability);
      artifact_identities[artifacts] =
          capabilities[index].artifact_identity;
      inventory_identities[artifacts] =
          capabilities[index].source_inventory_identity;
      sources += capabilities[index].source_count;
      ++artifacts;
      if (role == Role::kNvFp4GateUp) {
        ++gate_up_assets;
      } else if (role == Role::kNvFp4Down) {
        ++down_assets;
      } else if (role == Role::kFp8GdnQkvZ) {
        ++gdn_assets;
      } else if (role == Role::kFp8FullQkv) {
        ++full_assets;
      } else if (role == Role::kFp8AttentionOutput) {
        ++output_assets;
      }
    }
  }
  if (artifacts != kSm87MacroFeedV4P40StartupPackageArtifacts ||
      sources != kSm87MacroFeedV4P40StartupPackageSources ||
      gate_up_assets != kSm87MacroFeedV4P40StartupPackageLayers ||
      down_assets != kSm87MacroFeedV4P40StartupPackageLayers ||
      gdn_assets != kSm87MacroFeedV4P40StartupPackageGdnLayers ||
      full_assets != kSm87MacroFeedV4P40StartupPackageFullLayers ||
      output_assets != kSm87MacroFeedV4P40StartupPackageLayers ||
      access.catalog_identity() != catalog_identity) {
    result.status =
        failure(Error::kProjectionInventory, "projection_inventory");
    return result;
  }

  const std::uint64_t package_identity = compute_package_identity(
      access, capabilities, plan, gate_up, down, sources);
  StartupSeals seals = mint_startup_seals(
      package_identity, plan_identity, gate_up, down);
  if (package_identity == 0U ||
      !startup_seals_valid(seals, package_identity, plan_identity,
                           device_ordinal)) {
    result.status = failure(Error::kPackageIdentity, "startup_seals");
    return result;
  }

  Sm87MacroFeedV4P40StartupPackageAudit audit;
  audit.magic = kSm87MacroFeedV4P40StartupPackageMagic;
  audit.abi_major = kSm87MacroFeedV4P40StartupPackageAbiMajor;
  audit.abi_minor = kSm87MacroFeedV4P40StartupPackageAbiMinor;
  audit.candidate_id = kSm87MacroFeedV4CandidateId;
  audit.deployment_plan_id = kSm87MacroFeedV4P40DeploymentPlanId;
  audit.deployment_plan_identity = plan_identity;
  audit.package_identity = package_identity;
  audit.owner_identity = owner_identity;
  audit.allocation_identity = allocation_identity;
  audit.catalog_identity = catalog_identity;
  audit.device_identity = device_identity;
  audit.device_ordinal = device_ordinal;
  audit.layers = kSm87MacroFeedV4P40StartupPackageLayers;
  audit.artifacts = artifacts;
  audit.sources = sources;
  audit.gate_up_assets = gate_up_assets;
  audit.down_assets = down_assets;
  audit.gdn_projection_assets = gdn_assets;
  audit.full_projection_assets = full_assets;
  audit.attention_output_assets = output_assets;
  audit.canonical_plan_generated_internally = true;
  audit.caller_plan_accepted = false;
  audit.complete_projection_access_retained = true;
  audit.catalog_revalidated = true;
  audit.typed_capabilities_retained = true;
  audit.authenticated_source_manifests_retained = true;
  audit.authenticated_upload_readback_retained = true;
  audit.projection_bindings_complete = true;
  audit.nvfp4_startup_seals_complete = true;
  audit.caller_raw_receipts_accepted = false;
  audit.v3_execution_identity_reused = false;
  audit.request_time_repack_jit_autotune_or_fallback_permitted = false;
  audit.fp8_executor_bound = false;
  audit.gdn_executor_bound = false;
  audit.attention_executor_bound = false;
  audit.request_state_bound = false;
  audit.finalizer_bound = false;
  audit.physical_receipt_bound = false;
  audit.host_only = true;
  audit.default_off = true;
  audit.test_only = true;
  audit.selector_bound = false;
  audit.launcher_present = false;
  audit.execution_ready = false;
  audit.numerical_qualification_complete = false;
  audit.production_dispatch_eligible = false;
  if (!audit.valid()) {
    result.status = failure(Error::kPackageIdentity, "package_audit");
    return result;
  }

  auto package = std::unique_ptr<Package>(new (std::nothrow) Package(
      std::move(access), std::move(capabilities), std::move(plan),
      std::move(seals), audit));
  if (!package) {
    result.status = failure(Error::kAllocationFailure, "package_allocation");
    return result;
  }
  if (!package->populate_projection_bindings() || !package->valid()) {
    result.status =
        failure(Error::kBindingConstruction, "binding_revalidation");
    return result;
  }
  result.audit = package->audit();
  result.package = std::move(package);
  result.status = {};
  return result;
}

std::uint64_t Sm87MacroFeedV4ProjectionStartupBinding::
    compute_binding_identity(const Snapshot& snapshot) noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(snapshot.role);
  const std::uint64_t expected_tactic =
      expected_consumer_tactic_identity(snapshot.role);
  if (snapshot.package_identity == 0U ||
      snapshot.deployment_plan_identity == 0U ||
      snapshot.owner_identity == 0U ||
      snapshot.allocation_identity == 0U ||
      snapshot.catalog_identity == 0U || snapshot.device_identity == 0U ||
      snapshot.artifact_identity == 0U ||
      snapshot.source_inventory_identity == 0U ||
      snapshot.manifest_seal == 0U ||
      snapshot.upload_receipt_identity == 0U ||
      kernels::sm87_target_aot_projection_digest_is_zero(
          snapshot.payload_digest) ||
      snapshot.device_ordinal < 0 ||
      snapshot.layer_index >= kSm87MacroFeedV4P40StartupPackageLayers ||
      !layout.valid() || snapshot.encoding != layout.encoding ||
      snapshot.payload_begin == 0U ||
      snapshot.payload_bytes != layout.payload_bytes ||
      snapshot.payload_begin >
          std::numeric_limits<std::uintptr_t>::max() -
              snapshot.payload_bytes ||
      snapshot.payload_end != snapshot.payload_begin + snapshot.payload_bytes ||
      snapshot.source_count != layout.partition_count ||
      snapshot.source_count > snapshot.tensor_scale_bits.size() ||
      snapshot.consumer_tactic_identity != expected_tactic ||
      !snapshot.issued_from_live_complete_asset ||
      !snapshot.canonical_payload_layout_retained ||
      snapshot.caller_raw_receipt_accepted ||
      snapshot.v3_execution_identity_reused || !snapshot.t0_only ||
      snapshot.launcher_authority ||
      snapshot.production_dispatch_eligible) {
    return 0U;
  }
  for (std::size_t index = 0U;
       index < snapshot.tensor_scale_bits.size(); ++index) {
    if (index < snapshot.source_count) {
      if (!kernels::sm87_target_aot_projection_scale_bits_valid(
              snapshot.tensor_scale_bits[index])) {
        return 0U;
      }
    } else if (snapshot.tensor_scale_bits[index] != 0U) {
      return 0U;
    }
  }

  std::uint64_t identity = 0x5133'4d46'5634'424eULL;
  identity = mix(identity, snapshot.package_identity);
  identity = mix(identity, snapshot.deployment_plan_identity);
  identity = mix(identity, snapshot.owner_identity);
  identity = mix(identity, snapshot.allocation_identity);
  identity = mix(identity, snapshot.catalog_identity);
  identity = mix(identity, snapshot.device_identity);
  identity = mix(identity, snapshot.consumer_tactic_identity);
  identity = mix(identity, snapshot.artifact_identity);
  identity = mix(identity, snapshot.source_inventory_identity);
  identity = mix(identity, snapshot.manifest_seal);
  identity = mix(identity, snapshot.upload_receipt_identity);
  identity = mix_digest(identity, snapshot.payload_digest);
  identity = mix(identity, snapshot.payload_begin);
  identity = mix(identity, snapshot.payload_end);
  identity = mix(identity, snapshot.payload_bytes);
  identity = mix(identity,
                 static_cast<std::uint64_t>(snapshot.device_ordinal + 1));
  identity = mix(identity, snapshot.layer_index + 1U);
  identity = mix(identity, static_cast<std::uint64_t>(snapshot.role));
  identity = mix(identity, static_cast<std::uint64_t>(snapshot.encoding));
  identity = mix(identity, snapshot.source_count);
  for (const std::uint32_t bits : snapshot.tensor_scale_bits) {
    identity = mix(identity, bits);
  }
  identity = mix(identity, snapshot.issued_from_live_complete_asset);
  identity = mix(identity, snapshot.canonical_payload_layout_retained);
  identity = mix(identity, snapshot.caller_raw_receipt_accepted);
  identity = mix(identity, snapshot.v3_execution_identity_reused);
  identity = mix(identity, snapshot.t0_only);
  identity = mix(identity, snapshot.launcher_authority);
  identity = mix(identity, snapshot.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5634'424eULL : identity;
}

std::uint32_t Sm87MacroFeedV4ProjectionStartupBinding::tensor_scale_bits(
    const std::size_t source_index) const noexcept {
  return source_index < snapshot_.source_count
             ? snapshot_.tensor_scale_bits[source_index]
             : 0U;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid_with_catalog(
    const std::uint64_t catalog_identity) const noexcept {
  return catalog_identity != 0U && projection_access_.attached() &&
         projection_access_.catalog_identity() == catalog_identity &&
         valid_with_prevalidated_catalog(catalog_identity);
}

bool Sm87MacroFeedV4ProjectionStartupBinding::
    valid_with_prevalidated_catalog(
        const std::uint64_t catalog_identity) const noexcept {
  if (snapshot_.binding_identity == 0U ||
      snapshot_.binding_identity != compute_binding_identity(snapshot_) ||
      !projection_access_.attached() ||
      projection_access_.owner_identity() != snapshot_.owner_identity ||
      projection_access_.allocation_identity() !=
          snapshot_.allocation_identity ||
      catalog_identity != snapshot_.catalog_identity ||
      projection_access_.device_identity() != snapshot_.device_identity ||
      projection_access_.device_ordinal() != snapshot_.device_ordinal ||
      asset_.layer_index() != snapshot_.layer_index ||
      asset_.role() != snapshot_.role ||
      asset_.encoding() != snapshot_.encoding ||
      asset_.artifact_identity() != snapshot_.artifact_identity ||
      asset_.source_inventory_identity() !=
          snapshot_.source_inventory_identity ||
      asset_.payload_bytes() != snapshot_.payload_bytes) {
    return false;
  }
  auto fresh = projection_access_.resolve(snapshot_.layer_index,
                                          snapshot_.role);
  if (!fresh || fresh->artifact_identity() != snapshot_.artifact_identity ||
      fresh->source_inventory_identity() !=
          snapshot_.source_inventory_identity ||
      fresh->encoding() != snapshot_.encoding ||
      fresh->payload_bytes() != snapshot_.payload_bytes) {
    return false;
  }

  const auto common_view_matches = [&](const auto& view) noexcept {
    if (view.artifact_identity != snapshot_.artifact_identity ||
        view.source_inventory_identity !=
            snapshot_.source_inventory_identity ||
        view.host_manifest_seal.value != snapshot_.manifest_seal ||
        view.device_upload_receipt.receipt_identity !=
            snapshot_.upload_receipt_identity ||
        view.host_payload_digest != snapshot_.payload_digest ||
        view.payload.begin != snapshot_.payload_begin ||
        view.payload.end != snapshot_.payload_end ||
        view.payload.bytes != snapshot_.payload_bytes ||
        view.tensor_scale_count != snapshot_.source_count ||
        !authenticated_upload_complete(
            view.device_upload_receipt, snapshot_.owner_identity,
            snapshot_.allocation_identity, snapshot_.device_ordinal,
            snapshot_.payload_begin, snapshot_.payload_end,
            snapshot_.payload_bytes)) {
      return false;
    }
    for (std::size_t index = 0U; index < snapshot_.source_count; ++index) {
      if (view.tensor_scale_bits[index] !=
          snapshot_.tensor_scale_bits[index]) {
        return false;
      }
    }
    return true;
  };

  if (sm87_target_aot_complete_role_is_nvfp4(snapshot_.role)) {
    const auto* const view = asset_.borrow_nvfp4_cuda_asset();
    const auto* const fresh_view = fresh->borrow_nvfp4_cuda_asset();
    return view != nullptr && fresh_view != nullptr &&
           asset_.borrow_fp8_cuda_asset() == nullptr &&
           kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*view) &&
           kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*fresh_view) &&
           common_view_matches(*view) && common_view_matches(*fresh_view);
  }
  if (sm87_target_aot_complete_role_is_fp8(snapshot_.role)) {
    const auto* const view = asset_.borrow_fp8_cuda_asset();
    const auto* const fresh_view = fresh->borrow_fp8_cuda_asset();
    return view != nullptr && fresh_view != nullptr &&
           asset_.borrow_nvfp4_cuda_asset() == nullptr &&
           kernels::sm87_target_aot_fp8_cuda_asset_valid(*view) &&
           kernels::sm87_target_aot_fp8_cuda_asset_valid(*fresh_view) &&
           common_view_matches(*view) && common_view_matches(*fresh_view);
  }
  return false;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid() const noexcept {
  if (!projection_access_.attached()) {
    return false;
  }
  const std::uint64_t catalog_identity =
      projection_access_.catalog_identity();
  return catalog_identity != 0U &&
         valid_with_prevalidated_catalog(catalog_identity);
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid_for(
    const std::size_t layer_index, const Role role,
    const std::uint64_t package_identity) const noexcept {
  if (!projection_access_.attached()) {
    return false;
  }
  const std::uint64_t catalog_identity =
      projection_access_.catalog_identity();
  return catalog_identity != 0U && valid_for_prevalidated_catalog(
                                       layer_index, role, package_identity,
                                       catalog_identity);
}

bool Sm87MacroFeedV4ProjectionStartupBinding::
    valid_for_prevalidated_catalog(
        const std::size_t layer_index, const Role role,
        const std::uint64_t package_identity,
        const std::uint64_t catalog_identity) const noexcept {
  return layer_index == snapshot_.layer_index && role == snapshot_.role &&
         package_identity != 0U &&
         package_identity == snapshot_.package_identity &&
         valid_with_prevalidated_catalog(catalog_identity);
}

const Sm87MacroFeedV4P40StartupPackage::AssetCapability*
Sm87MacroFeedV4P40StartupPackage::capability(
    const std::size_t layer_index, const Role role) const noexcept {
  const std::size_t index =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  if (index >= capabilities_.size()) {
    return nullptr;
  }
  const auto& capability = capabilities_[index];
  return capability.asset && capability.layer_index == layer_index &&
                 capability.role == role &&
                 capability.artifact_identity != 0U &&
                 capability.source_inventory_identity != 0U &&
                 capability.manifest_seal != 0U &&
                 capability.upload_receipt_identity != 0U &&
                 capability.payload_begin != 0U &&
                 capability.payload_end > capability.payload_begin &&
                 capability.payload_bytes != 0U &&
                 capability.source_count != 0U
             ? &capability
             : nullptr;
}

std::optional<Sm87MacroFeedV4ProjectionStartupBinding>
Sm87MacroFeedV4P40StartupPackage::make_projection_binding(
    const std::size_t layer_index, const Role role) const noexcept {
  if (!audit_.valid() || !projection_access_.attached()) {
    return std::nullopt;
  }
  const AssetCapability* const retained = capability(layer_index, role);
  auto asset = projection_access_.resolve(layer_index, role);
  if (retained == nullptr || !asset || !retained->asset ||
      asset->artifact_identity() != retained->artifact_identity ||
      asset->source_inventory_identity() !=
          retained->source_inventory_identity ||
      asset->encoding() != retained->encoding ||
      asset->payload_bytes() != retained->payload_bytes) {
    return std::nullopt;
  }

  Sm87MacroFeedV4ProjectionStartupBinding::Snapshot snapshot;
  snapshot.package_identity = audit_.package_identity;
  snapshot.deployment_plan_identity = audit_.deployment_plan_identity;
  snapshot.owner_identity = audit_.owner_identity;
  snapshot.allocation_identity = audit_.allocation_identity;
  snapshot.catalog_identity = audit_.catalog_identity;
  snapshot.device_identity = audit_.device_identity;
  snapshot.consumer_tactic_identity = expected_consumer_tactic_identity(role);
  snapshot.artifact_identity = retained->artifact_identity;
  snapshot.source_inventory_identity = retained->source_inventory_identity;
  snapshot.manifest_seal = retained->manifest_seal;
  snapshot.upload_receipt_identity = retained->upload_receipt_identity;
  snapshot.payload_digest = retained->payload_digest;
  snapshot.payload_begin = retained->payload_begin;
  snapshot.payload_end = retained->payload_end;
  snapshot.payload_bytes = retained->payload_bytes;
  snapshot.device_ordinal = audit_.device_ordinal;
  snapshot.layer_index = layer_index;
  snapshot.role = role;
  snapshot.encoding = retained->encoding;
  snapshot.tensor_scale_bits = retained->tensor_scale_bits;
  snapshot.source_count = retained->source_count;
  snapshot.issued_from_live_complete_asset = true;
  snapshot.canonical_payload_layout_retained = true;
  snapshot.caller_raw_receipt_accepted = false;
  snapshot.v3_execution_identity_reused = false;
  snapshot.t0_only = true;
  snapshot.launcher_authority = false;
  snapshot.production_dispatch_eligible = false;
  snapshot.binding_identity =
      Sm87MacroFeedV4ProjectionStartupBinding::compute_binding_identity(
          snapshot);
  if (snapshot.binding_identity == 0U) {
    return std::nullopt;
  }
  Sm87MacroFeedV4ProjectionStartupBinding binding(
      projection_access_, std::move(*asset), std::move(snapshot));
  if (!binding.valid_with_prevalidated_catalog(audit_.catalog_identity)) {
    return std::nullopt;
  }
  return std::optional<Sm87MacroFeedV4ProjectionStartupBinding>(
      std::move(binding));
}

bool Sm87MacroFeedV4P40StartupPackage::base_valid() const noexcept {
  const std::uint64_t plan_identity =
      compute_deployment_plan_identity(plan_);
  if (!audit_.valid() || !projection_access_.attached() ||
      projection_access_.owner_identity() != audit_.owner_identity ||
      projection_access_.allocation_identity() !=
          audit_.allocation_identity ||
      projection_access_.catalog_identity() != audit_.catalog_identity ||
      projection_access_.device_identity() != audit_.device_identity ||
      projection_access_.device_ordinal() != audit_.device_ordinal ||
      projection_access_.artifact_count() != audit_.artifacts ||
      plan_identity == 0U ||
      plan_identity != audit_.deployment_plan_identity ||
      !startup_seals_valid(seals_, audit_.package_identity, plan_identity,
                           audit_.device_ordinal)) {
    return false;
  }

  std::size_t artifacts = 0U;
  std::size_t sources = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV4P40StartupPackageLayers;
       ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      const AssetCapability* const retained =
          capability(layer_index, role);
      if (index != artifacts || retained == nullptr ||
          index >= capabilities_.size()) {
        return false;
      }
      auto fresh = projection_access_.resolve(layer_index, role);
      if (!fresh || fresh->artifact_identity() != retained->artifact_identity ||
          fresh->source_inventory_identity() !=
              retained->source_inventory_identity ||
          fresh->encoding() != retained->encoding ||
          fresh->payload_bytes() != retained->payload_bytes) {
        return false;
      }
      const auto live_view_matches = [&](const auto& view) noexcept {
        if (view.artifact_identity != retained->artifact_identity ||
            view.source_inventory_identity !=
                retained->source_inventory_identity ||
            view.host_manifest_seal.value != retained->manifest_seal ||
            view.device_upload_receipt.receipt_identity !=
                retained->upload_receipt_identity ||
            view.host_payload_digest != retained->payload_digest ||
            view.payload.begin != retained->payload_begin ||
            view.payload.end != retained->payload_end ||
            view.payload.bytes != retained->payload_bytes ||
            view.tensor_scale_count != retained->source_count ||
            !authenticated_upload_complete(
                view.device_upload_receipt, audit_.owner_identity,
                audit_.allocation_identity, audit_.device_ordinal,
                retained->payload_begin, retained->payload_end,
                retained->payload_bytes)) {
          return false;
        }
        for (std::size_t source = 0U; source < retained->source_count;
             ++source) {
          if (view.tensor_scale_bits[source] !=
              retained->tensor_scale_bits[source]) {
            return false;
          }
        }
        return true;
      };
      bool view_valid = false;
      if (sm87_target_aot_complete_role_is_nvfp4(role)) {
        const auto* const view = fresh->borrow_nvfp4_cuda_asset();
        view_valid =
            view != nullptr && fresh->borrow_fp8_cuda_asset() == nullptr &&
            kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*view) &&
            live_view_matches(*view);
      } else {
        const auto* const view = fresh->borrow_fp8_cuda_asset();
        view_valid =
            view != nullptr && fresh->borrow_nvfp4_cuda_asset() == nullptr &&
            kernels::sm87_target_aot_fp8_cuda_asset_valid(*view) &&
            live_view_matches(*view);
      }
      if (!view_valid) {
        return false;
      }
      sources += retained->source_count;
      ++artifacts;
    }
  }
  return artifacts == audit_.artifacts && sources == audit_.sources &&
         audit_.package_identity == compute_package_identity(
                                        projection_access_, capabilities_,
                                        plan_, seals_.gate_up.resources,
                                        seals_.down.resources, sources);
}

bool Sm87MacroFeedV4P40StartupPackage::populate_projection_bindings()
    noexcept {
  if (!base_valid()) {
    return false;
  }
  std::size_t bindings = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV4P40StartupPackageLayers;
       ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != bindings || index >= projection_bindings_.size() ||
          projection_bindings_[index].has_value()) {
        return false;
      }
      auto binding = make_projection_binding(layer_index, role);
      if (!binding) {
        return false;
      }
      projection_bindings_[index].emplace(std::move(*binding));
      ++bindings;
    }
  }
  return bindings == projection_bindings_.size();
}

bool Sm87MacroFeedV4P40StartupPackage::valid() const noexcept {
  if (!base_valid()) {
    return false;
  }
  std::size_t bindings = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV4P40StartupPackageLayers;
       ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != bindings || index >= projection_bindings_.size() ||
          !projection_bindings_[index] ||
          !projection_bindings_[index]->valid_for_prevalidated_catalog(
              layer_index, role, audit_.package_identity,
              audit_.catalog_identity) ||
          projection_bindings_[index]->deployment_plan_identity() !=
              audit_.deployment_plan_identity) {
        return false;
      }
      ++bindings;
    }
  }
  return bindings == projection_bindings_.size();
}

const Sm87MacroFeedV4ProjectionStartupBinding*
Sm87MacroFeedV4P40StartupPackage::borrow_projection_startup_binding(
    const std::size_t layer_index, const Role role) const noexcept {
  const std::size_t index =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  if (index >= projection_bindings_.size() ||
      !projection_bindings_[index] || !base_valid()) {
    return nullptr;
  }
  const auto& binding = *projection_bindings_[index];
  return binding.valid_for_prevalidated_catalog(
             layer_index, role, audit_.package_identity,
             audit_.catalog_identity) &&
                 binding.deployment_plan_identity() ==
                     audit_.deployment_plan_identity
             ? &binding
             : nullptr;
}

bool Sm87MacroFeedV4P40StartupPackage::borrow_projection_startup_catalog(
    ProjectionStartupBindingCatalog* const catalog) const noexcept {
  if (catalog == nullptr) {
    return false;
  }
  catalog->fill(nullptr);
  if (!base_valid()) {
    return false;
  }

  std::size_t bindings = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV4P40StartupPackageLayers;
       ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != bindings || index >= projection_bindings_.size() ||
          !projection_bindings_[index] ||
          !projection_bindings_[index]->valid_for_prevalidated_catalog(
              layer_index, role, audit_.package_identity,
              audit_.catalog_identity) ||
          projection_bindings_[index]->deployment_plan_identity() !=
              audit_.deployment_plan_identity) {
        catalog->fill(nullptr);
        return false;
      }
      (*catalog)[index] = &*projection_bindings_[index];
      ++bindings;
    }
  }
  if (bindings != catalog->size()) {
    catalog->fill(nullptr);
    return false;
  }
  return true;
}

#else

std::uint32_t Sm87MacroFeedV4ProjectionStartupBinding::tensor_scale_bits(
    const std::size_t source_index) const noexcept {
  (void)source_index;
  return 0U;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid() const noexcept {
  return false;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid_with_catalog(
    const std::uint64_t catalog_identity) const noexcept {
  (void)catalog_identity;
  return false;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::
    valid_with_prevalidated_catalog(
        const std::uint64_t catalog_identity) const noexcept {
  (void)catalog_identity;
  return false;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::
    valid_for_prevalidated_catalog(
        const std::size_t layer_index, const Role role,
        const std::uint64_t package_identity,
        const std::uint64_t catalog_identity) const noexcept {
  (void)layer_index;
  (void)role;
  (void)package_identity;
  (void)catalog_identity;
  return false;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid_for(
    const std::size_t layer_index, const Role role,
    const std::uint64_t package_identity) const noexcept {
  (void)layer_index;
  (void)role;
  (void)package_identity;
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::valid() const noexcept {
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::base_valid() const noexcept {
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::populate_projection_bindings()
    noexcept {
  return false;
}

const Sm87MacroFeedV4ProjectionStartupBinding*
Sm87MacroFeedV4P40StartupPackage::borrow_projection_startup_binding(
    const std::size_t layer_index, const Role role) const noexcept {
  (void)layer_index;
  (void)role;
  return nullptr;
}

bool Sm87MacroFeedV4P40StartupPackage::borrow_projection_startup_catalog(
    ProjectionStartupBindingCatalog* const catalog) const noexcept {
  if (catalog != nullptr) {
    catalog->fill(nullptr);
  }
  return false;
}

#endif

}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail
