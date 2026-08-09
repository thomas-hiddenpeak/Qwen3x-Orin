#include "q3x/core/sha256.h"
#include "q3x/kernels/sm87_fp8_marlin_w8a16.h"
#include "q3x/kernels/sm87_nvfp4_marlin.h"
#include "q3x/runtime/reference_engine.h"

#include "reference_engine_prefill_authority.h"
#include "reference_runner_gdn_chunk64_native_admission.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace core = q3x::core;
namespace runtime = q3x::runtime;
namespace engine_detail = q3x::runtime::reference_engine_detail;
namespace runner_detail = q3x::runtime::reference_runner_detail;

constexpr std::array<std::size_t, 4U> kPromptTokenCounts{
    {513U, 1'025U, 8'192U, 8'193U}};
constexpr std::size_t kPromptTemplateTokens = 12U;
constexpr std::size_t kHashCopyChunkBytes = 4U * 1024U * 1024U;
constexpr std::size_t kKvHeadCount = 4U;
constexpr std::size_t kKvHeadDimension = 256U;
constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);

enum class SnapshotError : std::uint8_t {
  kNone = 0,
  kDuplicateHook,
  kInvalidState,
  kUnexpectedSequenceLength,
  kInvalidPersistentRegion,
  kInvalidKvRegion,
  kInvalidPromptResidual,
  kInvalidFinalHidden,
  kHashFailure,
};

[[nodiscard]] const char* to_string(const SnapshotError error) noexcept {
  switch (error) {
    case SnapshotError::kNone:
      return "none";
    case SnapshotError::kDuplicateHook:
      return "duplicate_hook";
    case SnapshotError::kInvalidState:
      return "invalid_state";
    case SnapshotError::kUnexpectedSequenceLength:
      return "unexpected_sequence_length";
    case SnapshotError::kInvalidPersistentRegion:
      return "invalid_persistent_region";
    case SnapshotError::kInvalidKvRegion:
      return "invalid_kv_region";
    case SnapshotError::kInvalidPromptResidual:
      return "invalid_prompt_residual";
    case SnapshotError::kInvalidFinalHidden:
      return "invalid_final_hidden";
    case SnapshotError::kHashFailure:
      return "hash_failure";
  }
  return "unknown";
}

struct StateSnapshot {
  explicit StateSnapshot(const std::size_t prompt_tokens)
      : expected_prompt_tokens(prompt_tokens),
        copy_scratch(kHashCopyChunkBytes) {}

  std::size_t expected_prompt_tokens = 0U;
  std::size_t hook_calls = 0U;
  std::uint32_t sequence_length = 0U;
  SnapshotError error = SnapshotError::kNone;
  int cuda_error = static_cast<int>(cudaSuccess);
  core::Sha256Digest conv_state;
  core::Sha256Digest gdn_state;
  std::array<core::Sha256Digest, runtime::kRequestFullLayerCount> key_cache;
  std::array<core::Sha256Digest, runtime::kRequestFullLayerCount> value_cache;
  core::Sha256Digest prompt_residual;
  core::Sha256Digest final_hidden;
  core::Sha256Digest aggregate;
  std::vector<std::uint8_t> copy_scratch;
};

[[nodiscard]] bool region_fits(const runtime::RequestRegion& region,
                               const std::uint64_t arena_bytes) noexcept {
  return region.byte_size != 0U && region.element_size_bytes != 0U &&
         region.arena_offset <= arena_bytes &&
         region.byte_size <= arena_bytes - region.arena_offset;
}

[[nodiscard]] bool hash_device_bytes(
    const std::uint8_t* const source, const std::size_t bytes,
    std::vector<std::uint8_t>& scratch, core::Sha256Digest& digest,
    int& cuda_error) noexcept {
  if (source == nullptr || bytes == 0U || scratch.empty()) {
    return false;
  }
  core::Sha256 hash;
  std::size_t offset = 0U;
  while (offset < bytes) {
    const std::size_t chunk =
        std::min(scratch.size(), bytes - offset);
    const cudaError_t status = cudaMemcpy(
        scratch.data(), source + offset, chunk, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      return false;
    }
    if (!hash.update(scratch.data(), chunk)) {
      return false;
    }
    offset += chunk;
  }
  digest = hash.finalize();
  return true;
}

[[nodiscard]] bool hash_arena_region(
    const runtime::RequestState& state,
    const runtime::RequestRegion& region, const std::size_t bytes,
    StateSnapshot& snapshot, core::Sha256Digest& digest) noexcept {
  if (!region_fits(region, state.arena_bytes()) || bytes == 0U ||
      bytes > region.byte_size ||
      region.arena_offset >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      bytes > std::numeric_limits<std::size_t>::max() -
                  static_cast<std::size_t>(region.arena_offset)) {
    return false;
  }
  const auto* const arena =
      static_cast<const std::uint8_t*>(state.arena_data());
  return hash_device_bytes(
      arena + static_cast<std::size_t>(region.arena_offset), bytes,
      snapshot.copy_scratch, digest, snapshot.cuda_error);
}

void collect_generate_return_snapshot(
    const runtime::RequestState& state, void* const context) noexcept {
  auto* const snapshot = static_cast<StateSnapshot*>(context);
  if (snapshot == nullptr) {
    return;
  }
  ++snapshot->hook_calls;
  if (snapshot->hook_calls != 1U) {
    snapshot->error = SnapshotError::kDuplicateHook;
    return;
  }
  if (!state || state.memory_profile() !=
                    runtime::RequestMemoryProfile::kLayerMajorC8192 ||
      state.arena_data() == nullptr || state.layer_major_plan() == nullptr ||
      snapshot->copy_scratch.empty()) {
    snapshot->error = SnapshotError::kInvalidState;
    return;
  }
  const cudaError_t synchronize_status = cudaDeviceSynchronize();
  if (synchronize_status != cudaSuccess) {
    snapshot->cuda_error = static_cast<int>(synchronize_status);
    snapshot->error = SnapshotError::kHashFailure;
    return;
  }

  snapshot->sequence_length = state.sequence_length();
  if (snapshot->sequence_length != snapshot->expected_prompt_tokens ||
      snapshot->expected_prompt_tokens > state.max_sequence_length()) {
    snapshot->error = SnapshotError::kUnexpectedSequenceLength;
    return;
  }

  const runtime::RequestMemoryPlan& common = state.plan();
  if (common.conv_state.byte_size != runtime::kRequestConvStateBytes ||
      common.gdn_state.byte_size != runtime::kRequestGdnStateBytes ||
      !hash_arena_region(state, common.conv_state,
                         static_cast<std::size_t>(common.conv_state.byte_size),
                         *snapshot, snapshot->conv_state) ||
      !hash_arena_region(state, common.gdn_state,
                         static_cast<std::size_t>(common.gdn_state.byte_size),
                         *snapshot, snapshot->gdn_state)) {
    snapshot->error = snapshot->cuda_error == static_cast<int>(cudaSuccess)
                          ? SnapshotError::kInvalidPersistentRegion
                          : SnapshotError::kHashFailure;
    return;
  }

  constexpr std::size_t kKvBytesPerLayerPosition =
      kKvHeadCount * kKvHeadDimension * kBf16Bytes;
  if (snapshot->expected_prompt_tokens >
      std::numeric_limits<std::size_t>::max() /
          kKvBytesPerLayerPosition) {
    snapshot->error = SnapshotError::kInvalidKvRegion;
    return;
  }
  const std::size_t used_kv_bytes =
      snapshot->expected_prompt_tokens * kKvBytesPerLayerPosition;
  for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
       ++slot) {
    const runtime::RequestRegion& key = common.key_cache[slot];
    const runtime::RequestRegion& value = common.value_cache[slot];
    const bool valid_shape =
        key.element_size_bytes == kBf16Bytes &&
        value.element_size_bytes == kBf16Bytes &&
        key.byte_size == value.byte_size && used_kv_bytes <= key.byte_size;
    if (!valid_shape ||
        !hash_arena_region(state, key, used_kv_bytes, *snapshot,
                           snapshot->key_cache[slot]) ||
        !hash_arena_region(state, value, used_kv_bytes, *snapshot,
                           snapshot->value_cache[slot])) {
      snapshot->error = snapshot->cuda_error == static_cast<int>(cudaSuccess)
                            ? SnapshotError::kInvalidKvRegion
                            : SnapshotError::kHashFailure;
      return;
    }
  }

  const runtime::LayerMajorRequestMemoryPlan& layer_major =
      *state.layer_major_plan();
  const runtime::RequestMatrixRegion& prompt =
      layer_major.prompt_residual_bf16;
  if (prompt.storage.element_size_bytes != kBf16Bytes ||
      prompt.columns != runtime::kReferenceHiddenSize ||
      prompt.row_stride_elements != runtime::kReferenceHiddenSize ||
      prompt.row_capacity < snapshot->expected_prompt_tokens ||
      snapshot->expected_prompt_tokens >
          std::numeric_limits<std::size_t>::max() /
              (runtime::kReferenceHiddenSize * kBf16Bytes)) {
    snapshot->error = SnapshotError::kInvalidPromptResidual;
    return;
  }
  const std::size_t prompt_bytes =
      snapshot->expected_prompt_tokens * runtime::kReferenceHiddenSize *
      kBf16Bytes;
  if (!hash_arena_region(state, prompt.storage, prompt_bytes, *snapshot,
                         snapshot->prompt_residual)) {
    snapshot->error = snapshot->cuda_error == static_cast<int>(cudaSuccess)
                          ? SnapshotError::kInvalidPromptResidual
                          : SnapshotError::kHashFailure;
    return;
  }

  const runtime::RequestMatrixRegion& final_hidden =
      layer_major.final_hidden_bf16;
  if (final_hidden.storage.element_size_bytes != kBf16Bytes ||
      final_hidden.row_capacity != 1U ||
      final_hidden.columns != runtime::kReferenceHiddenSize ||
      final_hidden.row_stride_elements != runtime::kReferenceHiddenSize ||
      !hash_arena_region(
          state, final_hidden.storage,
          static_cast<std::size_t>(final_hidden.storage.byte_size),
          *snapshot, snapshot->final_hidden)) {
    snapshot->error = snapshot->cuda_error == static_cast<int>(cudaSuccess)
                          ? SnapshotError::kInvalidFinalHidden
                          : SnapshotError::kHashFailure;
    return;
  }

  core::Sha256 aggregate;
  const auto add_digest = [&aggregate](
                              const core::Sha256Digest& digest) noexcept {
    return aggregate.update(digest.bytes.data(), digest.bytes.size());
  };
  bool complete = add_digest(snapshot->conv_state) &&
                  add_digest(snapshot->gdn_state);
  for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
       ++slot) {
    complete = complete && add_digest(snapshot->key_cache[slot]) &&
               add_digest(snapshot->value_cache[slot]);
  }
  complete = complete && add_digest(snapshot->prompt_residual) &&
             add_digest(snapshot->final_hidden);
  if (!complete) {
    snapshot->error = SnapshotError::kHashFailure;
    return;
  }
  snapshot->aggregate = aggregate.finalize();
}

class ScopedGenerateReturnSnapshot final {
 public:
  explicit ScopedGenerateReturnSnapshot(StateSnapshot& snapshot) noexcept
      : previous_(
            runner_detail::exchange_reference_engine_generate_return_snapshot_hook(
                {collect_generate_return_snapshot, &snapshot})) {}

  ~ScopedGenerateReturnSnapshot() {
    (void)runner_detail::
        exchange_reference_engine_generate_return_snapshot_hook(previous_);
  }

  ScopedGenerateReturnSnapshot(const ScopedGenerateReturnSnapshot&) = delete;
  ScopedGenerateReturnSnapshot& operator=(
      const ScopedGenerateReturnSnapshot&) = delete;

 private:
  runner_detail::ReferenceEngineGenerateReturnSnapshotHook previous_{};
};

class ScopedCompatibilityOracle final {
 public:
  explicit ScopedCompatibilityOracle(const bool enabled) noexcept
      : previous_(engine_detail::
                      exchange_reference_engine_prefill_compatibility_oracle_for_test(
                          enabled)) {}

  ~ScopedCompatibilityOracle() {
    (void)engine_detail::
        exchange_reference_engine_prefill_compatibility_oracle_for_test(
            previous_);
  }

  ScopedCompatibilityOracle(const ScopedCompatibilityOracle&) = delete;
  ScopedCompatibilityOracle& operator=(const ScopedCompatibilityOracle&) =
      delete;

 private:
  bool previous_ = false;
};

[[nodiscard]] std::string model_directory_from(
    const int argc, char** const argv) {
  if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0' &&
      std::string_view(argv[1]) != "-") {
    return argv[1];
  }
  const char* const environment = std::getenv("Q3X_E2E_MODEL_DIR");
  return environment == nullptr ? std::string{} : std::string(environment);
}

[[nodiscard]] std::string repeated_hello_prompt(
    const std::size_t prompt_tokens) {
  if (prompt_tokens < kPromptTemplateTokens) {
    return {};
  }
  const std::size_t words = prompt_tokens - kPromptTemplateTokens;
  std::string prompt;
  prompt.reserve(words * 6U);
  for (std::size_t index = 0U; index < words; ++index) {
    if (index != 0U) {
      prompt.push_back(' ');
    }
    prompt.append("hello");
  }
  return prompt;
}

void print_diagnostic(
    const runtime::ReferenceEngineDiagnostic& diagnostic) {
  std::cerr << "code=" << runtime::to_string(diagnostic.code)
            << " stage=" << diagnostic.stage
            << " message=" << diagnostic.message
            << " context=" << diagnostic.context
            << " cuda_error=" << diagnostic.cuda_error
            << " layer=" << diagnostic.layer
            << " operation=" << diagnostic.operation << '\n';
}

[[nodiscard]] bool same_route(
    const runtime::PrefillRouteEvidence& left,
    const runtime::PrefillRouteEvidence& right) noexcept {
  if (left.completed_layer_passes != right.completed_layer_passes ||
      left.expected_layer_passes != right.expected_layer_passes ||
      left.request_active != right.request_active ||
      left.complete != right.complete || left.valid != right.valid ||
      left.error != right.error ||
      left.forbidden_boundary_hits != right.forbidden_boundary_hits) {
    return false;
  }
  for (std::size_t role = 0U; role < runtime::kPrefillOperatorRoleCount;
       ++role) {
    const runtime::PrefillOperatorRouteCounts& a = left.operators[role];
    const runtime::PrefillOperatorRouteCounts& b = right.operators[role];
    if (a.production_hits != b.production_hits ||
        a.exact_fallback_hits != b.exact_fallback_hits ||
        a.forbidden_hits != b.forbidden_hits) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool valid_completed_native_route(
    const runtime::PrefillRouteEvidence& evidence,
    const std::uint64_t logical_panels) noexcept {
  if (!evidence.complete || !evidence.valid || evidence.request_active ||
      evidence.error != runtime::PrefillRouteEvidenceError::kNone ||
      evidence.completed_layer_passes != logical_panels ||
      evidence.expected_layer_passes != logical_panels) {
    return false;
  }
  for (const std::uint64_t count : evidence.forbidden_boundary_hits) {
    if (count != 0U) {
      return false;
    }
  }
  for (std::size_t role = 0U; role < runtime::kPrefillOperatorRoleCount;
       ++role) {
    const runtime::PrefillOperatorRouteCounts& counts =
        evidence.operators[role];
    if (counts.production_hits !=
            runtime::kExpectedPrefillLogicalOperatorsPerTile[role] *
                logical_panels ||
        counts.exact_fallback_hits != 0U || counts.forbidden_hits != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool valid_snapshot(const StateSnapshot& snapshot) noexcept {
  return snapshot.hook_calls == 1U &&
         snapshot.sequence_length == snapshot.expected_prompt_tokens &&
         snapshot.error == SnapshotError::kNone &&
         snapshot.cuda_error == static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool same_snapshots(const StateSnapshot& oracle,
                                  const StateSnapshot& panel) {
  bool exact = oracle.conv_state == panel.conv_state &&
               oracle.gdn_state == panel.gdn_state &&
               oracle.prompt_residual == panel.prompt_residual &&
               oracle.final_hidden == panel.final_hidden &&
               oracle.aggregate == panel.aggregate;
  for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
       ++slot) {
    exact = exact && oracle.key_cache[slot] == panel.key_cache[slot] &&
            oracle.value_cache[slot] == panel.value_cache[slot];
  }
  if (!exact) {
    const auto report = [](const std::string_view region,
                           const core::Sha256Digest& expected,
                           const core::Sha256Digest& actual) {
      if (!(expected == actual)) {
        std::cerr << "mismatch region=" << region
                  << " oracle_sha256=" << expected.hex()
                  << " panel_sha256=" << actual.hex() << '\n';
      }
    };
    report("conv_state", oracle.conv_state, panel.conv_state);
    report("gdn_state", oracle.gdn_state, panel.gdn_state);
    report("prompt_residual", oracle.prompt_residual,
           panel.prompt_residual);
    report("final_hidden", oracle.final_hidden, panel.final_hidden);
    for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
         ++slot) {
      const std::size_t layer = 3U + 4U * slot;
      report("key_cache_layer_" + std::to_string(layer),
             oracle.key_cache[slot], panel.key_cache[slot]);
      report("value_cache_layer_" + std::to_string(layer),
             oracle.value_cache[slot], panel.value_cache[slot]);
    }
  }
  return exact;
}

[[nodiscard]] bool run_generation(
    runtime::ReferenceEngine& engine, const std::string& prompt,
    const std::size_t prompt_tokens, const bool compatibility_oracle,
    StateSnapshot& snapshot, runtime::ReferenceGenerateResult& result) {
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens = 1U;
  options.prefill_chunk_size = runtime::kMaximumRequestPrefillChunkSize;
  options.logits_mode = runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  options.prefill_execution_mode =
      runtime::ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
  {
    const ScopedGenerateReturnSnapshot snapshot_hook(snapshot);
    const ScopedCompatibilityOracle oracle_route(compatibility_oracle);
    result = engine.generate(prompt, options);
  }
  if (!result) {
    std::cerr << (compatibility_oracle ? "compatibility oracle" : "panel")
              << " generation failed for P" << prompt_tokens << ": ";
    print_diagnostic(result.diagnostic);
    return false;
  }
  if (!valid_snapshot(snapshot)) {
    std::cerr << (compatibility_oracle ? "compatibility oracle" : "panel")
              << " snapshot failed for P" << prompt_tokens
              << " calls=" << snapshot.hook_calls
              << " sequence_length=" << snapshot.sequence_length
              << " snapshot_error=" << to_string(snapshot.error)
              << " cuda_error=" << snapshot.cuda_error << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool run_case(runtime::ReferenceEngine& engine,
                            const std::size_t prompt_tokens,
                            const bool native_group_q64_panel,
                            const bool segmented_marlin_operator_panel) {
  const std::size_t expected_logical_panels =
      (prompt_tokens + runtime::kLayerMajorPrefillOperatorPanelTokens - 1U) /
      runtime::kLayerMajorPrefillOperatorPanelTokens;
  runtime::PrefillExecutionPlanOptions topology_options;
  topology_options.prompt_token_count = prompt_tokens;
  const runtime::PrefillExecutionPlanResult topology =
      runtime::build_unbound_layer_major_prefill_execution_plan(
          topology_options);
  bool topology_exact =
      topology && topology.value->panel_count == expected_logical_panels;
  if (topology_exact) {
    std::uint32_t expected_first_position = 0U;
    for (std::size_t panel_index = 0U;
         panel_index < topology.value->panel_count; ++panel_index) {
      const runtime::PrefillOperatorPanel& panel =
          topology.value->panels[panel_index];
      topology_exact =
          topology_exact && panel.first_position == expected_first_position &&
          panel.token_count >= runtime::kPrefillPhysicalSegmentM32Tokens &&
          panel.end_position == panel.first_position + panel.token_count;
      expected_first_position = panel.end_position;
    }
    topology_exact =
        topology_exact && expected_first_position == prompt_tokens;
  }
  if (topology_exact && prompt_tokens == 8'192U) {
    topology_exact = topology_exact && topology.value->panel_count == 1U &&
                     topology.value->panels[0].token_count == 8'192U;
  } else if (topology_exact && prompt_tokens == 8'193U) {
    topology_exact = topology_exact && topology.value->panel_count == 2U &&
                     topology.value->panels[0].token_count == 4'097U &&
                     topology.value->panels[1].token_count == 4'096U;
  }
  if (!topology_exact) {
    std::cerr << "operator-panel topology gate failed for P"
              << prompt_tokens << '\n';
    return false;
  }

  const std::string prompt = repeated_hello_prompt(prompt_tokens);
  StateSnapshot oracle_snapshot(prompt_tokens);
  StateSnapshot panel_snapshot(prompt_tokens);
  runtime::ReferenceGenerateResult oracle_result;
  runtime::ReferenceGenerateResult panel_result;
  if (!run_generation(engine, prompt, prompt_tokens, true, oracle_snapshot,
                      oracle_result) ||
      !run_generation(engine, prompt, prompt_tokens, false, panel_snapshot,
                      panel_result)) {
    return false;
  }

  const runtime::ReferenceGeneration& oracle = *oracle_result.value;
  const runtime::ReferenceGeneration& panel = *panel_result.value;
  const bool architecture_candidate =
      native_group_q64_panel || segmented_marlin_operator_panel;
  std::size_t expected_segmented_projection_physical_launches = 0U;
  if (segmented_marlin_operator_panel) {
    for (std::size_t panel_index = 0U;
         panel_index < topology.value->panel_count; ++panel_index) {
      const std::size_t panel_tokens =
          topology.value->panels[panel_index].token_count;
      const std::size_t nvfp4_launches =
          q3x::kernels::sm87_nvfp4_marlin_execution_plan(panel_tokens)
              .launch_count;
      const std::size_t fp8_large_n_launches =
          q3x::kernels::sm87_fp8_marlin_execution_plan(panel_tokens, 5'120U)
              .launch_count;
      const std::size_t fp8_small_n_launches =
          q3x::kernels::sm87_fp8_marlin_execution_plan(panel_tokens, 1'024U)
              .launch_count;
      expected_segmented_projection_physical_launches +=
          128U * nvfp4_launches + 176U * fp8_large_n_launches +
          32U * fp8_small_n_launches;
    }
  }
  const std::string_view expected_deployment_plan_id =
      segmented_marlin_operator_panel
          ? (native_group_q64_panel
                 ? runtime::
                       kLayerMajorSegmentedMarlinProjectionGroupQ64DeploymentPlanId
                 : runtime::kLayerMajorSegmentedMarlinProjectionDeploymentPlanId)
          : (native_group_q64_panel
                 ? runtime::kLayerMajorNativeGroupQ64PanelDeploymentPlanId
                 : runtime::kLayerMajorOperatorPanelDeploymentPlanId);
  const bool output_exact =
      oracle.prompt_token_ids.size() == prompt_tokens &&
      panel.prompt_token_ids == oracle.prompt_token_ids &&
      panel.generated_token_ids == oracle.generated_token_ids &&
      panel.generated_text == oracle.generated_text &&
      panel.stop_reason == oracle.stop_reason;
  const bool route_exact = same_route(oracle.prefill_route_evidence,
                                      panel.prefill_route_evidence);
  const bool route_contract =
      oracle.prefill_execution_mode ==
          runtime::ReferencePrefillExecutionMode::kWholeRequestLayerMajor &&
      panel.prefill_execution_mode == oracle.prefill_execution_mode &&
      oracle.prefill_deployment_plan_id == expected_deployment_plan_id &&
      panel.prefill_deployment_plan_id ==
          oracle.prefill_deployment_plan_id &&
      oracle.prefill_logical_panel_count == expected_logical_panels &&
      panel.prefill_logical_panel_count ==
          oracle.prefill_logical_panel_count &&
      valid_completed_native_route(oracle.prefill_route_evidence,
                                   expected_logical_panels) &&
      valid_completed_native_route(panel.prefill_route_evidence,
                                   expected_logical_panels) &&
      panel.prefill_operator_panel_executor_hits ==
          runtime::kReferenceDecoderLayerCount * expected_logical_panels &&
      panel.prefill_native_group_q64_panel_hits ==
          (native_group_q64_panel
               ? runtime::kRequestFullLayerCount * expected_logical_panels
               : 0U) &&
      (!native_group_q64_panel || panel.prefill_generic_qt2_hits == 0U) &&
      panel.prefill_segmented_panel_projection_hits ==
          (segmented_marlin_operator_panel
               ? 336U * expected_logical_panels
               : 0U) &&
      panel.prefill_segmented_panel_projection_physical_launches ==
          expected_segmented_projection_physical_launches &&
      oracle.prefill_operator_panel_executor_hits == 0U &&
      oracle.prefill_native_group_q64_panel_hits == 0U &&
      oracle.prefill_segmented_panel_projection_hits == 0U &&
      oracle.prefill_segmented_panel_projection_physical_launches == 0U;
  const bool state_exact =
      same_snapshots(oracle_snapshot, panel_snapshot);
  const bool direction_passed =
      output_exact && route_exact && route_contract;
  const bool exact_equivalence_passed = direction_passed && state_exact;
  // A projection-only candidate changes wrapper segmentation but not the
  // admitted arithmetic, recurrent state, or Attention tactic, so it must
  // retain the complete real-model state oracle.  Native grouped Attention
  // remains explicitly accuracy-unqualified and is screened only for output
  // and sealed-route integrity here.
  const bool candidate_passed =
      direction_passed && (native_group_q64_panel || state_exact);
  const bool passed = architecture_candidate ? candidate_passed
                                              : exact_equivalence_passed;
  std::cout << (architecture_candidate
                    ? "PREFILL_OPERATOR_PANEL_DIRECTION_SCREEN"
                    : "PREFILL_OPERATOR_PANEL_EQUIVALENCE")
            << " prompt_tokens=" << prompt_tokens
            << " logical_panels=" << expected_logical_panels
            << " oracle_state_sha256=" << oracle_snapshot.aggregate.hex()
            << " panel_state_sha256=" << panel_snapshot.aggregate.hex()
            << " generated_token="
            << (panel.generated_token_ids.empty()
                    ? runtime::kReferenceVocabularySize
                    : panel.generated_token_ids.front())
            << " generated_text=" << panel.generated_text
            << " output_exact=" << (output_exact ? "true" : "false")
            << " route_exact=" << (route_exact ? "true" : "false")
            << " route_contract=" << (route_contract ? "true" : "false")
            << " state_exact=" << (state_exact ? "true" : "false")
            << " attention_tactic="
            << (native_group_q64_panel ? "native-group-q64-panel"
                                       : "exact-segmented")
            << " projection_tactic="
            << (segmented_marlin_operator_panel
                    ? "segmented-marlin-operator-panel"
                    : "exact-segmented")
            << " operator_panel_executor_hits="
            << panel.prefill_operator_panel_executor_hits
            << " native_group_q64_panel_hits="
            << panel.prefill_native_group_q64_panel_hits
            << " generic_qt2_hits=" << panel.prefill_generic_qt2_hits
            << " segmented_panel_projection_hits="
            << panel.prefill_segmented_panel_projection_hits
            << " segmented_panel_projection_physical_launches="
            << panel.prefill_segmented_panel_projection_physical_launches
            << (architecture_candidate
                    ? (candidate_passed
                           ? " direction_gate=PASS accuracy_gate=NOT_RUN"
                           : " direction_gate=FAIL accuracy_gate=NOT_RUN")
                    : (exact_equivalence_passed
                           ? " equivalence_gate=PASS"
                           : " equivalence_gate=FAIL"))
            << (architecture_candidate
                    ? " qualification=ACCURACY_UNQUALIFIED"
                      " authority=REAL_MODEL_DIRECTION_SCREEN_ONLY\n"
                    : " qualification=BITWISE_EXACT"
                      " authority=REAL_MODEL_CORRECTNESS_ONLY\n");
  return passed;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr << "usage: q3x_reference_prefill_operator_panel_"
                 "equivalence_e2e_test [MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }
  const char* const enabled =
      std::getenv("Q3X_RUN_PREFILL_OPERATOR_PANEL_EQUIVALENCE");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "SKIP: set Q3X_RUN_PREFILL_OPERATOR_PANEL_EQUIVALENCE=1 "
                 "after a clean-host GPU preflight\n";
    return 77;
  }
  const char* const attention_tactic =
      std::getenv("Q3X_TEST_PREFILL_ATTENTION_TACTIC");
  const bool native_group_q64_panel =
      attention_tactic != nullptr &&
      std::string_view(attention_tactic) == "native-group-q64-panel";
  if (attention_tactic != nullptr && !native_group_q64_panel &&
      std::string_view(attention_tactic) != "exact-segmented") {
    std::cerr << "Q3X_TEST_PREFILL_ATTENTION_TACTIC must be "
                 "exact-segmented or native-group-q64-panel\n";
    return 2;
  }
  const char* const projection_tactic =
      std::getenv("Q3X_TEST_PREFILL_PROJECTION_TACTIC");
  const bool segmented_marlin_operator_panel =
      projection_tactic != nullptr &&
      std::string_view(projection_tactic) ==
          "segmented-marlin-operator-panel";
  if (projection_tactic != nullptr && !segmented_marlin_operator_panel &&
      std::string_view(projection_tactic) != "exact-segmented") {
    std::cerr << "Q3X_TEST_PREFILL_PROJECTION_TACTIC must be "
                 "exact-segmented or segmented-marlin-operator-panel\n";
    return 2;
  }

  try {
    runtime::ReferenceEngineOptions options;
    options.request_options.prefill_chunk_size =
        runtime::kMaximumRequestPrefillChunkSize;
    options.request_options.max_sequence_length =
        kPromptTokenCounts.back() + 1U;
    options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
    options.prefill_execution_mode =
        runtime::ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
    options.prefill_full_attention_tactic =
        native_group_q64_panel
            ? runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeGroupQ64Panel
            : runtime::LayerMajorPrefillFullAttentionTactic::
                  kExactSegmentedC512;
    options.prefill_projection_tactic =
        segmented_marlin_operator_panel
            ? runtime::LayerMajorPrefillProjectionTactic::
                  kSegmentedMarlinOperatorPanel
            : runtime::LayerMajorPrefillProjectionTactic::
                  kExactSegmentedC512;
    runtime::ReferenceEngineCreateResult created =
        runtime::create_reference_engine(
            std::filesystem::path(model_directory), options);
    if (!created) {
      std::cerr << "engine creation failed: ";
      print_diagnostic(created.diagnostic);
      return 1;
    }

    for (const std::size_t prompt_tokens : kPromptTokenCounts) {
      if (!run_case(*created.value, prompt_tokens,
                    native_group_q64_panel,
                    segmented_marlin_operator_panel)) {
        return 1;
      }
    }
    return 0;
  } catch (const std::bad_alloc&) {
    std::cerr << "host allocation failed\n";
    return 1;
  }
}
