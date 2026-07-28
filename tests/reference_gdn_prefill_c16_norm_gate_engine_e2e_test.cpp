#include "q3x/runtime/reference_engine.h"
#include "q3x/runtime/gdn_decode.h"

#include "reference_runner_gdn_c16_norm_gate_admission.h"

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
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace runtime = q3x::runtime;
namespace detail = q3x::runtime::reference_runner_detail;

constexpr std::size_t kPromptTokens = 513U;
constexpr std::size_t kPrefixTokens = 512U;
constexpr std::size_t kC16Slices = kPrefixTokens / 16U;
constexpr std::size_t kExpectedRouteHits =
    runtime::kRequestLinearLayerCount * kC16Slices;
constexpr std::uint32_t kExpectedGeneratedToken = 9'419U;
constexpr std::string_view kExpectedGeneratedText = "Hello";
constexpr std::size_t kSnapshotStageCount = 2U;
constexpr std::array<std::uint32_t, kSnapshotStageCount>
    kExpectedCommittedPositions{kPrefixTokens, kPromptTokens};

using SnapshotStage =
    detail::PrefillGdnC16NormGateAdmissionSnapshotStage;
using SnapshotHook = detail::PrefillGdnC16NormGateAdmissionSnapshotHook;

enum class SnapshotMode : std::uint8_t {
  kStoreReference = 0,
  kCompareReference,
};

struct SnapshotCollector {
  SnapshotMode mode = SnapshotMode::kStoreReference;
  const SnapshotCollector* reference = nullptr;
  std::array<std::vector<std::uint16_t>, kSnapshotStageCount> snapshots;
  std::vector<std::uint16_t> scratch;
  std::array<std::size_t, kSnapshotStageCount> calls{};
  std::array<std::uint32_t, kSnapshotStageCount> committed_positions{};
  std::array<std::uint64_t, kSnapshotStageCount> region_bytes{};
  std::array<std::size_t, kSnapshotStageCount> unequal_elements{};
  std::array<std::size_t, kSnapshotStageCount> first_mismatch{
      std::numeric_limits<std::size_t>::max(),
      std::numeric_limits<std::size_t>::max()};
  int cuda_error = static_cast<int>(cudaSuccess);
  bool contract_error = false;
};

[[nodiscard]] constexpr std::size_t snapshot_stage_index(
    const SnapshotStage stage) noexcept {
  return stage == SnapshotStage::kPrefixTile ? 0U : 1U;
}

[[nodiscard]] constexpr std::string_view snapshot_stage_name(
    const std::size_t index) noexcept {
  return index == 0U ? "prefix_p_minus_1" : "finish_prefill_step";
}

void collect_gdn_snapshot(const runtime::RequestState& state,
                          const SnapshotStage stage,
                          void* const context) noexcept {
  auto* const collector = static_cast<SnapshotCollector*>(context);
  if (collector == nullptr ||
      (stage != SnapshotStage::kPrefixTile &&
       stage != SnapshotStage::kStep)) {
    return;
  }
  const std::size_t stage_index = snapshot_stage_index(stage);
  ++collector->calls[stage_index];
  collector->committed_positions[stage_index] = state.current_position();
  if (collector->calls[stage_index] != 1U) {
    collector->contract_error = true;
    return;
  }

  const runtime::RequestRegion& region = state.plan().gdn_state;
  collector->region_bytes[stage_index] = region.byte_size;
  const bool valid_region =
      state.arena_data() != nullptr &&
      region.byte_size == runtime::kRequestGdnStateBytes &&
      region.element_size_bytes == sizeof(std::uint16_t) &&
      region.element_capacity ==
          runtime::kRequestLinearLayerCount * runtime::kGdnStateElements &&
      region.arena_offset <= state.arena_bytes() &&
      region.byte_size <= state.arena_bytes() - region.arena_offset;
  if (!valid_region) {
    collector->contract_error = true;
    return;
  }
  const std::size_t element_count =
      static_cast<std::size_t>(region.element_capacity);
  std::uint16_t* destination = nullptr;
  if (collector->mode == SnapshotMode::kStoreReference) {
    if (collector->snapshots[stage_index].size() != element_count) {
      collector->contract_error = true;
      return;
    }
    destination = collector->snapshots[stage_index].data();
  } else {
    if (collector->scratch.size() != element_count ||
        collector->reference == nullptr ||
        collector->reference->calls[stage_index] != 1U ||
        collector->reference->snapshots[stage_index].size() !=
            element_count) {
      collector->contract_error = true;
      return;
    }
    destination = collector->scratch.data();
  }

  const auto* const source =
      static_cast<const std::uint8_t*>(state.arena_data()) +
      static_cast<std::size_t>(region.arena_offset);
  const cudaError_t copy_status = cudaMemcpy(
      destination, source, static_cast<std::size_t>(region.byte_size),
      cudaMemcpyDeviceToHost);
  if (copy_status != cudaSuccess) {
    collector->cuda_error = static_cast<int>(copy_status);
    return;
  }
  if (collector->mode == SnapshotMode::kCompareReference) {
    const std::vector<std::uint16_t>& expected =
        collector->reference->snapshots[stage_index];
    std::size_t unequal = 0U;
    std::size_t first = std::numeric_limits<std::size_t>::max();
    for (std::size_t index = 0U; index < element_count; ++index) {
      if (destination[index] != expected[index]) {
        if (first == std::numeric_limits<std::size_t>::max()) {
          first = index;
        }
        ++unequal;
      }
    }
    collector->unequal_elements[stage_index] = unequal;
    collector->first_mismatch[stage_index] = first;
  }
}

class ScopedAdmission {
 public:
  explicit ScopedAdmission(const bool enabled) noexcept
      : previous_(detail::
                      exchange_prefill_gdn_c16_norm_gate_admission_test_enabled(
                          enabled)) {}
  ~ScopedAdmission() {
    (void)detail::
        exchange_prefill_gdn_c16_norm_gate_admission_test_enabled(previous_);
  }

  ScopedAdmission(const ScopedAdmission&) = delete;
  ScopedAdmission& operator=(const ScopedAdmission&) = delete;

 private:
  bool previous_ = false;
};

class ScopedSnapshotHook {
 public:
  explicit ScopedSnapshotHook(SnapshotCollector& collector) noexcept
      : previous_(
            detail::exchange_prefill_gdn_c16_norm_gate_admission_snapshot_hook(
                SnapshotHook{collect_gdn_snapshot, &collector})) {}
  ~ScopedSnapshotHook() {
    (void)detail::
        exchange_prefill_gdn_c16_norm_gate_admission_snapshot_hook(previous_);
  }

  ScopedSnapshotHook(const ScopedSnapshotHook&) = delete;
  ScopedSnapshotHook& operator=(const ScopedSnapshotHook&) = delete;

 private:
  SnapshotHook previous_{};
};

[[nodiscard]] bool prepare_collector(SnapshotCollector& collector) {
  const std::size_t elements =
      static_cast<std::size_t>(runtime::kRequestGdnStateBytes /
                               sizeof(std::uint16_t));
  try {
    if (collector.mode == SnapshotMode::kStoreReference) {
      for (std::vector<std::uint16_t>& snapshot : collector.snapshots) {
        snapshot.resize(elements);
      }
    } else {
      collector.scratch.resize(elements);
    }
  } catch (const std::bad_alloc&) {
    std::cerr << "host allocation failed while preparing GDN snapshots\n";
    return false;
  } catch (const std::length_error&) {
    std::cerr << "GDN snapshot size exceeds host vector capacity\n";
    return false;
  }
  return true;
}

[[nodiscard]] bool validate_collector(const SnapshotCollector& collector,
                                      const bool require_bitwise_exact) {
  bool passed = !collector.contract_error &&
                collector.cuda_error == static_cast<int>(cudaSuccess);
  for (std::size_t index = 0U; index < kSnapshotStageCount; ++index) {
    const bool exact = !require_bitwise_exact ||
                       collector.unequal_elements[index] == 0U;
    const bool stage_passed =
        collector.calls[index] == 1U &&
        collector.committed_positions[index] ==
            kExpectedCommittedPositions[index] &&
        collector.region_bytes[index] == runtime::kRequestGdnStateBytes &&
        exact;
    passed = passed && stage_passed;
    std::cout << "GDN_C16_ENGINE_STATE stage="
              << snapshot_stage_name(index)
              << " calls=" << collector.calls[index]
              << " committed_position="
              << collector.committed_positions[index]
              << " region_bytes=" << collector.region_bytes[index]
              << " elements="
              << runtime::kRequestGdnStateBytes / sizeof(std::uint16_t);
    if (require_bitwise_exact) {
      std::cout << " unequal_bf16=" << collector.unequal_elements[index]
                << " first_mismatch=";
      if (collector.first_mismatch[index] ==
          std::numeric_limits<std::size_t>::max()) {
        std::cout << "none";
      } else {
        std::cout << collector.first_mismatch[index];
      }
      std::cout << " bitwise_exact=" << (exact ? "true" : "false");
    }
    std::cout << " gate=" << (stage_passed ? "PASS" : "FAIL") << '\n';
  }
  if (collector.cuda_error != static_cast<int>(cudaSuccess)) {
    std::cerr << "GDN snapshot CUDA failure: "
              << cudaGetErrorString(
                     static_cast<cudaError_t>(collector.cuda_error))
              << '\n';
  }
  return passed;
}

[[nodiscard]] bool validate_reference_activity(
    const SnapshotCollector& collector) {
  std::array<std::size_t, kSnapshotStageCount> nonzero{};
  for (std::size_t stage = 0U; stage < kSnapshotStageCount; ++stage) {
    nonzero[stage] = static_cast<std::size_t>(std::count_if(
        collector.snapshots[stage].begin(),
        collector.snapshots[stage].end(),
        [](const std::uint16_t value) { return (value & 0x7fffU) != 0U; }));
  }
  std::size_t changed = 0U;
  for (std::size_t index = 0U;
       index < collector.snapshots[0].size(); ++index) {
    changed += collector.snapshots[0][index] !=
                       collector.snapshots[1][index]
                   ? 1U
                   : 0U;
  }
  const bool passed = nonzero[0] != 0U && nonzero[1] != 0U && changed != 0U;
  std::cout << "GDN_C16_ENGINE_STATE_ACTIVITY prefix_nonzero_bf16="
            << nonzero[0] << " finish_prefill_nonzero_bf16=" << nonzero[1]
            << " changed_between_stages_bf16=" << changed
            << " gate=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}

[[nodiscard]] std::string model_directory_from(
    const int argc, char** const argv) {
  if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0' &&
      std::string_view(argv[1]) != "-") {
    return argv[1];
  }
  const char* const environment = std::getenv("Q3X_E2E_MODEL_DIR");
  return environment == nullptr ? std::string{} : std::string(environment);
}

[[nodiscard]] std::string repeated_hello_prompt() {
  constexpr std::size_t kWords = kPromptTokens - 12U;
  std::string prompt;
  prompt.reserve(kWords * 6U);
  for (std::size_t index = 0U; index < kWords; ++index) {
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

[[nodiscard]] bool expected_generation(
    const runtime::ReferenceGeneration& generation) {
  bool steps_exact = generation.steps.size() == kPromptTokens &&
                     generation.prompt_token_ids.size() == kPromptTokens;
  if (steps_exact) {
    for (std::size_t index = 0U; index < generation.steps.size(); ++index) {
      const runtime::ReferenceStepResult& step = generation.steps[index];
      const bool prefix_step = index < kPrefixTokens;
      const bool prediction_exact =
          prefix_step
              ? !step.prediction.has_value()
              : (step.prediction.has_value() &&
                 step.prediction->predicted_token_id ==
                     kExpectedGeneratedToken);
      if (step.position != static_cast<std::uint32_t>(index) ||
          step.input_token_id != generation.prompt_token_ids[index] ||
          step.logits.has_value() || !prediction_exact) {
        steps_exact = false;
        break;
      }
    }
  }
  return generation.prompt_token_ids.size() == kPromptTokens &&
         generation.generated_token_ids ==
             std::vector<std::uint32_t>{kExpectedGeneratedToken} &&
         std::string_view(generation.generated_text) ==
             kExpectedGeneratedText &&
         generation.stop_reason == runtime::ReferenceStopReason::kMaxNewTokens &&
         generation.requested_prefill_chunk_size == kPrefixTokens &&
         generation.effective_prefill_chunk_size == kPrefixTokens &&
         generation.decode_graph_replays == 0U &&
         generation.decode_graph_serial_fallbacks == 0U &&
         generation.traces.empty() && steps_exact;
}

[[nodiscard]] bool same_generation_semantics(
    const runtime::ReferenceGeneration& expected,
    const runtime::ReferenceGeneration& actual) noexcept {
  if (expected.rendered_prompt != actual.rendered_prompt ||
      expected.prompt_token_ids != actual.prompt_token_ids ||
      expected.generated_token_ids != actual.generated_token_ids ||
      expected.generated_text != actual.generated_text ||
      expected.stop_reason != actual.stop_reason ||
      expected.requested_prefill_chunk_size !=
          actual.requested_prefill_chunk_size ||
      expected.effective_prefill_chunk_size !=
          actual.effective_prefill_chunk_size ||
      expected.steps.size() != actual.steps.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < expected.steps.size(); ++index) {
    const runtime::ReferenceStepResult& left = expected.steps[index];
    const runtime::ReferenceStepResult& right = actual.steps[index];
    if (left.position != right.position ||
        left.input_token_id != right.input_token_id ||
        left.logits.has_value() != right.logits.has_value() ||
        left.prediction.has_value() != right.prediction.has_value()) {
      return false;
    }
    if (left.logits.has_value() &&
        (left.logits->predicted_token_id !=
             right.logits->predicted_token_id ||
         left.logits->chosen_logit != right.logits->chosen_logit ||
         left.logits->max_log_probability !=
             right.logits->max_log_probability ||
         left.logits->logsumexp != right.logits->logsumexp)) {
      return false;
    }
    if (left.prediction.has_value() &&
        left.prediction->predicted_token_id !=
            right.prediction->predicted_token_id) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] runtime::ReferenceGenerateResult run_generation(
    runtime::ReferenceEngine& engine,
    const std::string& prompt,
    const bool admission_enabled,
    SnapshotCollector& collector,
    std::size_t& route_hits) {
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens = 1U;
  options.prefill_chunk_size = kPrefixTokens;
  options.logits_mode = runtime::ReferenceLogitsMode::kPredictedTokenOnly;

  (void)detail::exchange_prefill_gdn_c16_norm_gate_admission_test_hits(0U);
  runtime::ReferenceGenerateResult result;
  {
    const ScopedSnapshotHook snapshot_hook(collector);
    const ScopedAdmission admission(admission_enabled);
    result = engine.generate(prompt, options);
  }
  route_hits =
      detail::exchange_prefill_gdn_c16_norm_gate_admission_test_hits(0U);
  return result;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr << "usage: "
                 "q3x_reference_gdn_prefill_c16_norm_gate_engine_e2e_test "
                 "[MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }
  const char* const enabled =
      std::getenv("Q3X_RUN_GDN_C16_NORM_GATE_ADMISSION");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "SKIP: set Q3X_RUN_GDN_C16_NORM_GATE_ADMISSION=1 to run "
                 "the exact-C16 real-model Engine correctness gate\n";
    return 77;
  }

  (void)detail::
      exchange_prefill_gdn_c16_norm_gate_admission_test_enabled(false);
  (void)detail::exchange_prefill_gdn_c16_norm_gate_admission_test_hits(0U);
  (void)detail::exchange_prefill_gdn_c16_norm_gate_admission_snapshot_hook({});

  SnapshotCollector baseline_collector;
  SnapshotCollector candidate_collector;
  candidate_collector.mode = SnapshotMode::kCompareReference;
  candidate_collector.reference = &baseline_collector;
  if (!prepare_collector(baseline_collector) ||
      !prepare_collector(candidate_collector)) {
    return 1;
  }

  runtime::ReferenceEngineOptions options;
  options.request_options.prefill_chunk_size = kPrefixTokens;
  options.request_options.max_sequence_length = kPromptTokens + 1U;
  options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
  runtime::ReferenceEngineCreateResult created =
      runtime::create_reference_engine(std::filesystem::path(model_directory),
                                       options);
  if (!created) {
    std::cerr << "engine creation failed: ";
    print_diagnostic(created.diagnostic);
    return 1;
  }
  const runtime::ReferenceEngineLoadStats& load = created.value->load_stats();
  std::cout << "GDN_C16_ENGINE_LOAD fp8_output_sidecars="
            << (load.fp8_output_sidecars_enabled ? "true" : "false")
            << " fp8_output_sidecar_layers="
            << load.fp8_output_sidecar_layers
            << " fp8_prefill_qkv_sidecars="
            << (load.fp8_prefill_qkv_sidecars_enabled ? "true" : "false")
            << " fp8_prefill_qkv_sidecar_layers="
            << load.fp8_prefill_qkv_sidecar_layers
            << " request_arena_bytes=" << load.request_arena_bytes << '\n';

  const std::string prompt = repeated_hello_prompt();
  std::size_t baseline_hits = 0U;
  runtime::ReferenceGenerateResult baseline = run_generation(
      *created.value, prompt, false, baseline_collector, baseline_hits);
  if (!baseline) {
    std::cerr << "baseline generation failed: ";
    print_diagnostic(baseline.diagnostic);
    return 1;
  }
  std::size_t candidate_hits = 0U;
  runtime::ReferenceGenerateResult candidate = run_generation(
      *created.value, prompt, true, candidate_collector, candidate_hits);
  if (!candidate) {
    std::cerr << "candidate generation failed: ";
    print_diagnostic(candidate.diagnostic);
    return 1;
  }

  const bool baseline_expected = expected_generation(*baseline.value);
  const bool candidate_expected = expected_generation(*candidate.value);
  const bool candidate_exact =
      same_generation_semantics(*baseline.value, *candidate.value);
  const bool route_exact =
      baseline_hits == 0U && candidate_hits == kExpectedRouteHits;
  const bool baseline_snapshots =
      validate_collector(baseline_collector, false);
  const bool baseline_activity =
      validate_reference_activity(baseline_collector);
  const bool candidate_snapshots =
      validate_collector(candidate_collector, true);
  const runtime::ReferenceGeneration& value = *candidate.value;
  std::cout << "GDN_C16_ENGINE_E2E prompt_tokens="
            << value.prompt_token_ids.size()
            << " chunk=" << value.effective_prefill_chunk_size
            << " generated_token="
            << (value.generated_token_ids.empty()
                    ? runtime::kReferenceVocabularySize
                    : value.generated_token_ids.front())
            << " generated_text=" << value.generated_text
            << " steps=" << value.steps.size()
            << " baseline_hits=" << baseline_hits
            << " candidate_hits=" << candidate_hits
            << " expected_hits=" << kExpectedRouteHits
            << " baseline_expected="
            << (baseline_expected ? "true" : "false")
            << " candidate_expected="
            << (candidate_expected ? "true" : "false")
            << " candidate_semantics_exact="
            << (candidate_exact ? "true" : "false")
            << " route_exact=" << (route_exact ? "true" : "false")
            << '\n';

  (void)detail::
      exchange_prefill_gdn_c16_norm_gate_admission_test_enabled(false);
  (void)detail::exchange_prefill_gdn_c16_norm_gate_admission_test_hits(0U);
  (void)detail::exchange_prefill_gdn_c16_norm_gate_admission_snapshot_hook({});
  const bool passed = baseline_expected && candidate_expected &&
                      candidate_exact && route_exact && baseline_snapshots &&
                      baseline_activity && candidate_snapshots;
  std::cout << "GDN_C16_ENGINE_CORRECTNESS "
            << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
