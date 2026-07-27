#include "q3x/runtime/gdn_decode.h"
#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/reference_runner.h"
#include "q3x/runtime/request_state.h"
#include "q3x/runtime/resident_weights.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace q3x::runtime::reference_runner_detail {

// Admission-only symbol intentionally omitted from the installed header.
[[nodiscard]] bool exchange_prefill_gdn_b8_admission_test_enabled(
    bool enabled) noexcept;
[[nodiscard]] std::size_t exchange_prefill_gdn_b8_admission_test_hits(
    std::size_t hits) noexcept;

}  // namespace q3x::runtime::reference_runner_detail

namespace {

namespace runtime = q3x::runtime;

constexpr std::array<std::uint32_t, 4U> kPromptOpening{
    248'045U, 846U, 198U, 14'556U};
constexpr std::uint32_t kPromptBodyToken = 23'066U;
constexpr std::array<std::uint32_t, 9U> kPromptSuffix{
    248'046U, 198U, 248'045U, 74'455U, 198U,
    248'068U, 271U, 248'069U, 271U};
constexpr std::uint32_t kExpectedFirstGeneratedToken = 9'419U;
constexpr double kMaximumLayerNrmse = 3.0e-2;
constexpr double kMinimumLayerCosine = 0.999;
constexpr double kMaximumAggregateNrmse = 1.0e-2;
constexpr double kMinimumAggregateCosine = 0.9999;
constexpr double kRelativeFloor = 1.0e-2;

struct Profile {
  std::size_t prompt_tokens;
  std::size_t maximum_decode_steps;
};

constexpr std::array<Profile, 4U> kProfiles{{
    {257U, 8U},
    {513U, 8U},
    {769U, 1U},
    {1'025U, 8U},
}};

class ScopedB8Admission {
 public:
  explicit ScopedB8Admission(const bool enabled) noexcept
      : previous_(runtime::reference_runner_detail::
                      exchange_prefill_gdn_b8_admission_test_enabled(
                          enabled)) {}
  ~ScopedB8Admission() {
    (void)runtime::reference_runner_detail::
        exchange_prefill_gdn_b8_admission_test_enabled(previous_);
  }

  ScopedB8Admission(const ScopedB8Admission&) = delete;
  ScopedB8Admission& operator=(const ScopedB8Admission&) = delete;

 private:
  bool previous_ = false;
};

struct ErrorMetrics {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
  double p99_absolute = 0.0;
  double p99_relative = 0.0;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  double squared_actual = 0.0;
  double dot = 0.0;
  double nrmse = 0.0;
  double cosine = 0.0;
  std::size_t non_finite = 0U;
  std::size_t unequal_bf16 = 0U;
};

struct StageSummary {
  ErrorMetrics aggregate;
  double worst_layer_nrmse = 0.0;
  double lowest_layer_cosine = 1.0;
  double worst_layer_p99_absolute = 0.0;
  double worst_layer_p99_relative = 0.0;
  std::size_t worst_nrmse_layer = runtime::kReferenceNoLayer;
  std::size_t lowest_cosine_layer = runtime::kReferenceNoLayer;
  bool passed = true;
};

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float decoded = 0.0F;
  std::memcpy(&decoded, &bits, sizeof(decoded));
  return decoded;
}

[[nodiscard]] std::vector<std::uint32_t> build_prompt_tokens(
    const std::size_t prompt_token_count) {
  std::vector<std::uint32_t> result;
  result.reserve(prompt_token_count);
  result.insert(result.end(), kPromptOpening.begin(), kPromptOpening.end());
  result.insert(result.end(), prompt_token_count - 13U, kPromptBodyToken);
  result.insert(result.end(), kPromptSuffix.begin(), kPromptSuffix.end());
  return result;
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

[[nodiscard]] ErrorMetrics calculate_metrics(
    const std::uint16_t* const actual,
    const std::uint16_t* const reference,
    const std::size_t count,
    const bool calculate_percentiles) {
  ErrorMetrics metrics;
  std::vector<float> absolute_errors;
  std::vector<float> relative_errors;
  if (calculate_percentiles) {
    absolute_errors.reserve(count);
    relative_errors.reserve(count);
  }
  for (std::size_t index = 0U; index < count; ++index) {
    metrics.unequal_bf16 += actual[index] != reference[index] ? 1U : 0U;
    const double actual_value = decode_bf16(actual[index]);
    const double reference_value = decode_bf16(reference[index]);
    if (!std::isfinite(actual_value) || !std::isfinite(reference_value)) {
      ++metrics.non_finite;
      continue;
    }
    const double absolute = std::abs(actual_value - reference_value);
    const double relative =
        absolute / std::max(std::abs(reference_value), kRelativeFloor);
    metrics.maximum_absolute = std::max(metrics.maximum_absolute, absolute);
    metrics.maximum_relative = std::max(metrics.maximum_relative, relative);
    metrics.squared_error += absolute * absolute;
    metrics.squared_reference += reference_value * reference_value;
    metrics.squared_actual += actual_value * actual_value;
    metrics.dot += actual_value * reference_value;
    if (calculate_percentiles) {
      absolute_errors.push_back(static_cast<float>(absolute));
      relative_errors.push_back(static_cast<float>(relative));
    }
  }
  metrics.nrmse = std::sqrt(
      metrics.squared_error /
      std::max(metrics.squared_reference,
               std::numeric_limits<double>::min()));
  metrics.cosine =
      metrics.dot /
      std::sqrt(std::max(metrics.squared_actual * metrics.squared_reference,
                         std::numeric_limits<double>::min()));
  if (!absolute_errors.empty()) {
    const std::size_t p99_index =
        std::min(absolute_errors.size() - 1U,
                 static_cast<std::size_t>(std::ceil(
                     0.99 * static_cast<double>(absolute_errors.size()))) -
                     1U);
    std::nth_element(absolute_errors.begin(),
                     absolute_errors.begin() + p99_index,
                     absolute_errors.end());
    std::nth_element(relative_errors.begin(),
                     relative_errors.begin() + p99_index,
                     relative_errors.end());
    metrics.p99_absolute = absolute_errors[p99_index];
    metrics.p99_relative = relative_errors[p99_index];
  }
  return metrics;
}

void accumulate_metrics(ErrorMetrics& aggregate,
                        const ErrorMetrics& layer) noexcept {
  aggregate.maximum_absolute =
      std::max(aggregate.maximum_absolute, layer.maximum_absolute);
  aggregate.maximum_relative =
      std::max(aggregate.maximum_relative, layer.maximum_relative);
  aggregate.p99_absolute =
      std::max(aggregate.p99_absolute, layer.p99_absolute);
  aggregate.p99_relative =
      std::max(aggregate.p99_relative, layer.p99_relative);
  aggregate.squared_error += layer.squared_error;
  aggregate.squared_reference += layer.squared_reference;
  aggregate.squared_actual += layer.squared_actual;
  aggregate.dot += layer.dot;
  aggregate.non_finite += layer.non_finite;
  aggregate.unequal_bf16 += layer.unequal_bf16;
}

void finalize_metrics(ErrorMetrics& metrics) noexcept {
  metrics.nrmse = std::sqrt(
      metrics.squared_error /
      std::max(metrics.squared_reference,
               std::numeric_limits<double>::min()));
  metrics.cosine =
      metrics.dot /
      std::sqrt(std::max(metrics.squared_actual * metrics.squared_reference,
                         std::numeric_limits<double>::min()));
}

[[nodiscard]] StageSummary compare_gdn_state(
    const std::vector<std::uint16_t>& actual,
    const std::vector<std::uint16_t>& reference,
    const std::size_t prompt_tokens,
    const std::string_view stage) {
  StageSummary summary;
  for (std::size_t layer = 0U;
       layer < runtime::kReferenceDecoderLayerCount; ++layer) {
    const runtime::RequestLayerSlotResult mapped = runtime::map_request_layer(
        layer, q3x::model::LayerType::kLinearAttention);
    if (!mapped) {
      continue;
    }
    const std::size_t offset =
        mapped.value->slot * runtime::kGdnStateElements;
    const ErrorMetrics metrics = calculate_metrics(
        actual.data() + offset, reference.data() + offset,
        runtime::kGdnStateElements, true);
    accumulate_metrics(summary.aggregate, metrics);
    if (metrics.nrmse > summary.worst_layer_nrmse) {
      summary.worst_layer_nrmse = metrics.nrmse;
      summary.worst_nrmse_layer = layer;
    }
    if (metrics.cosine < summary.lowest_layer_cosine) {
      summary.lowest_layer_cosine = metrics.cosine;
      summary.lowest_cosine_layer = layer;
    }
    summary.worst_layer_p99_absolute =
        std::max(summary.worst_layer_p99_absolute, metrics.p99_absolute);
    summary.worst_layer_p99_relative =
        std::max(summary.worst_layer_p99_relative, metrics.p99_relative);
    const bool layer_passed =
        metrics.non_finite == 0U && metrics.nrmse <= kMaximumLayerNrmse &&
        metrics.cosine >= kMinimumLayerCosine;
    summary.passed = summary.passed && layer_passed;
    std::cout << "GDN_B8_ADMISSION_LAYER prompt_tokens=" << prompt_tokens
              << " stage=" << stage << " layer=" << layer
              << " nrmse=" << metrics.nrmse
              << " cosine=" << metrics.cosine
              << " max_abs=" << metrics.maximum_absolute
              << " max_rel_floor_1e-2=" << metrics.maximum_relative
              << " p99_abs=" << metrics.p99_absolute
              << " p99_rel_floor_1e-2=" << metrics.p99_relative
              << " non_finite=" << metrics.non_finite
              << " unequal_bf16=" << metrics.unequal_bf16
              << " gate=" << (layer_passed ? "PASS" : "FAIL") << '\n';
  }
  finalize_metrics(summary.aggregate);
  const bool aggregate_passed =
      summary.aggregate.non_finite == 0U &&
      summary.aggregate.nrmse <= kMaximumAggregateNrmse &&
      summary.aggregate.cosine >= kMinimumAggregateCosine;
  summary.passed = summary.passed && aggregate_passed;
  std::cout << "GDN_B8_ADMISSION_AGGREGATE prompt_tokens=" << prompt_tokens
            << " stage=" << stage
            << " nrmse=" << summary.aggregate.nrmse
            << " cosine=" << summary.aggregate.cosine
            << " max_abs=" << summary.aggregate.maximum_absolute
            << " max_rel_floor_1e-2="
            << summary.aggregate.maximum_relative
            << " worst_layer_p99_abs="
            << summary.worst_layer_p99_absolute
            << " worst_layer_p99_rel_floor_1e-2="
            << summary.worst_layer_p99_relative
            << " non_finite=" << summary.aggregate.non_finite
            << " unequal_bf16=" << summary.aggregate.unequal_bf16
            << " worst_nrmse_layer=" << summary.worst_nrmse_layer
            << " worst_layer_nrmse=" << summary.worst_layer_nrmse
            << " lowest_cosine_layer=" << summary.lowest_cosine_layer
            << " lowest_layer_cosine=" << summary.lowest_layer_cosine
            << " gate=" << (summary.passed ? "PASS" : "FAIL") << '\n';
  return summary;
}

[[nodiscard]] bool snapshot_gdn_state(
    const runtime::RequestState& state,
    std::vector<std::uint16_t>& destination,
    const std::string_view label) {
  const runtime::RequestRegion& region = state.plan().gdn_state;
  if (region.byte_size != runtime::kRequestGdnStateBytes ||
      region.element_size_bytes != sizeof(std::uint16_t) ||
      region.element_capacity !=
          runtime::kRequestLinearLayerCount * runtime::kGdnStateElements) {
    std::cerr << "invalid GDN aggregate region at " << label << '\n';
    return false;
  }
  destination.resize(static_cast<std::size_t>(region.element_capacity));
  const auto* const source =
      static_cast<const std::uint8_t*>(state.arena_data()) +
      static_cast<std::size_t>(region.arena_offset);
  const cudaError_t status =
      cudaMemcpy(destination.data(), source,
                 static_cast<std::size_t>(region.byte_size),
                 cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    std::cerr << "GDN snapshot failed at " << label
              << ": " << cudaGetErrorString(status) << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool run_prefix(runtime::ReferenceRunner& runner,
                              const std::vector<std::uint32_t>& tokens,
                              const std::size_t prefix_tokens) {
  std::size_t offset = 0U;
  while (offset < prefix_tokens) {
    const std::size_t tile =
        std::min<std::size_t>(runtime::kMaximumRequestPrefillChunkSize,
                              prefix_tokens - offset);
    const runtime::ReferencePrefillTileOutcome outcome =
        runner.prefill_prefix_tile(tokens.data() + offset, tile);
    if (!outcome) {
      std::cerr << "Prefill failed offset=" << offset << " tokens=" << tile
                << " error="
                << runtime::reference_runner_error_string(
                       outcome.status.error)
                << " cuda_error=" << outcome.status.cuda_error
                << " layer=" << outcome.status.layer
                << " operation="
                << (outcome.status.operation == nullptr
                        ? ""
                        : outcome.status.operation)
                << '\n';
      return false;
    }
    if (outcome.value->step_count != tile) {
      std::cerr << "Prefill transcript count mismatch offset=" << offset
                << " expected=" << tile
                << " actual=" << outcome.value->step_count << '\n';
      return false;
    }
    for (std::size_t index = 0U; index < tile; ++index) {
      const runtime::ReferenceStepResult& step = outcome.value->steps[index];
      if (step.position != offset + index ||
          step.input_token_id != tokens[offset + index]) {
        std::cerr << "Prefill transcript mismatch offset=" << offset
                  << " index=" << index << '\n';
        return false;
      }
    }
    offset += tile;
  }
  return true;
}

[[nodiscard]] bool reset_runner(runtime::ReferenceRunner& runner,
                                const std::string_view label) {
  const runtime::ReferenceRunnerStatus status = runner.reset();
  if (!status) {
    std::cerr << "runner reset failed at " << label
              << " error=" << runtime::reference_runner_error_string(
                                     status.error)
              << " cuda_error=" << status.cuda_error << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool run_profile(runtime::ReferenceRunner& runner,
                               runtime::RequestState& state,
                               const Profile& profile) {
  const std::vector<std::uint32_t> tokens =
      build_prompt_tokens(profile.prompt_tokens);
  if (tokens.size() != profile.prompt_tokens) {
    return false;
  }
  const std::size_t prefix_tokens = profile.prompt_tokens - 1U;
  std::vector<std::vector<std::uint16_t>> baseline_states(
      profile.maximum_decode_steps + 1U);
  std::vector<std::uint32_t> baseline_generated;
  baseline_generated.reserve(profile.maximum_decode_steps);

  if (!reset_runner(runner, "baseline")) {
    return false;
  }
  (void)runtime::reference_runner_detail::
      exchange_prefill_gdn_b8_admission_test_hits(0U);
  {
    const ScopedB8Admission admission(false);
    if (!run_prefix(runner, tokens, prefix_tokens)) {
      return false;
    }
  }
  if (!snapshot_gdn_state(state, baseline_states[0],
                          "baseline-prefix-p-minus-1")) {
    return false;
  }
  if (runtime::reference_runner_detail::
          exchange_prefill_gdn_b8_admission_test_hits(0U) != 0U) {
    std::cerr << "baseline unexpectedly hit B8 route prompt_tokens="
              << profile.prompt_tokens << '\n';
    return false;
  }

  runtime::ReferenceStepOptions step_options;
  step_options.compute_logits = true;
  step_options.logits_mode = runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  std::uint32_t input_token = tokens.back();
  for (std::size_t step = 0U; step < profile.maximum_decode_steps; ++step) {
    const runtime::ReferenceStepOutcome outcome =
        runner.step(input_token, step_options);
    if (!outcome || !outcome.value->prediction.has_value()) {
      std::cerr << "baseline Decode failed prompt_tokens="
                << profile.prompt_tokens << " step=" << step << '\n';
      return false;
    }
    const std::uint32_t predicted =
        outcome.value->prediction->predicted_token_id;
    baseline_generated.push_back(predicted);
    input_token = predicted;
    if (!snapshot_gdn_state(
            state, baseline_states[step + 1U],
            "baseline-decode-" + std::to_string(step + 1U))) {
      return false;
    }
  }

  if (!reset_runner(runner, "candidate")) {
    return false;
  }
  (void)runtime::reference_runner_detail::
      exchange_prefill_gdn_b8_admission_test_hits(0U);
  {
    const ScopedB8Admission admission(true);
    if (!run_prefix(runner, tokens, prefix_tokens)) {
      return false;
    }
  }
  const std::size_t expected_hits =
      ((prefix_tokens + runtime::kMaximumRequestPrefillChunkSize - 1U) /
       runtime::kMaximumRequestPrefillChunkSize) *
      runtime::kRequestLinearLayerCount;
  const std::size_t candidate_hits = runtime::reference_runner_detail::
      exchange_prefill_gdn_b8_admission_test_hits(0U);
  if (candidate_hits != expected_hits) {
    std::cerr << "candidate B8 route-hit mismatch prompt_tokens="
              << profile.prompt_tokens << " expected=" << expected_hits
              << " actual=" << candidate_hits << '\n';
    return false;
  }
  std::cout << "GDN_B8_ADMISSION_ROUTE prompt_tokens="
            << profile.prompt_tokens << " hits=" << candidate_hits
            << " gate=PASS\n";
  std::vector<std::uint16_t> candidate_state;
  if (!snapshot_gdn_state(state, candidate_state,
                          "candidate-prefix-p-minus-1")) {
    return false;
  }
  bool passed = compare_gdn_state(candidate_state, baseline_states[0],
                                  profile.prompt_tokens,
                                  "prefix_p_minus_1")
                    .passed;

  input_token = tokens.back();
  for (std::size_t step = 0U; step < profile.maximum_decode_steps; ++step) {
    const runtime::ReferenceStepOutcome outcome =
        runner.step(input_token, step_options);
    if (!outcome || !outcome.value->prediction.has_value()) {
      std::cerr << "candidate Decode failed prompt_tokens="
                << profile.prompt_tokens << " step=" << step << '\n';
      return false;
    }
    const std::uint32_t predicted =
        outcome.value->prediction->predicted_token_id;
    const bool token_exact = predicted == baseline_generated[step];
    passed = passed && token_exact;
    if (!snapshot_gdn_state(
            state, candidate_state,
            "candidate-decode-" + std::to_string(step + 1U))) {
      return false;
    }
    const std::string stage =
        step == 0U ? "prefill_final"
                   : "decode" + std::to_string(step);
    const StageSummary summary = compare_gdn_state(
        candidate_state, baseline_states[step + 1U], profile.prompt_tokens,
        stage);
    passed = passed && summary.passed;
    std::cout << "GDN_B8_ADMISSION_TOKEN prompt_tokens="
              << profile.prompt_tokens << " step=" << step + 1U
              << " baseline=" << baseline_generated[step]
              << " candidate=" << predicted
              << " exact=" << (token_exact ? "true" : "false") << '\n';
    // Teacher-force the baseline token so subsequent state comparisons use
    // an identical continuation even if this gate has already failed.
    input_token = baseline_generated[step];
  }

  if ((profile.prompt_tokens == 257U || profile.prompt_tokens == 513U) &&
      (baseline_generated.empty() ||
       baseline_generated.front() != kExpectedFirstGeneratedToken)) {
    std::cerr << "pinned first-token gate failed prompt_tokens="
              << profile.prompt_tokens << '\n';
    passed = false;
  }
  std::cout << "GDN_B8_ADMISSION_PROFILE prompt_tokens="
            << profile.prompt_tokens << " prefix_tokens=" << prefix_tokens
            << " decode_steps=" << profile.maximum_decode_steps
            << " first_generated="
            << (baseline_generated.empty()
                    ? runtime::kReferenceVocabularySize
                    : baseline_generated.front())
            << " result=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr << "usage: q3x_reference_gdn_prefill_b8_admission_e2e_test "
                 "[MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }
  const char* const enabled = std::getenv("Q3X_RUN_GDN_B8_ADMISSION");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "SKIP: set Q3X_RUN_GDN_B8_ADMISSION=1 to run the "
                 "real-checkpoint admission gate\n";
    return 77;
  }

  (void)runtime::reference_runner_detail::
      exchange_prefill_gdn_b8_admission_test_enabled(false);
  runtime::ResidentLoadResult resident = runtime::load_pinned_qwen36_27b(
      std::filesystem::path(model_directory));
  if (!resident) {
    std::cerr << "resident load failed: "
              << runtime::to_string(resident.diagnostic.code)
              << " message=" << resident.diagnostic.message
              << " context=" << resident.diagnostic.context << '\n';
    return 1;
  }
  runtime::WeightBindResult weights =
      runtime::bind_qwen36_27b_weights(*resident.value);
  if (!weights) {
    std::cerr << "weight bind failed: "
              << runtime::to_string(weights.diagnostic.code)
              << " tensor=" << weights.diagnostic.tensor
              << " message=" << weights.diagnostic.message << '\n';
    return 1;
  }

  runtime::RequestMemoryOptions request_options;
  request_options.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  request_options.max_sequence_length =
      kProfiles.back().prompt_tokens + kProfiles.back().maximum_decode_steps;
  runtime::RequestStateResult request =
      runtime::create_request_state(request_options);
  if (!request) {
    std::cerr << "request-state creation failed: "
              << runtime::to_string(request.diagnostic.code)
              << " message=" << request.diagnostic.message << '\n';
    return 1;
  }

  runtime::ReferenceRunnerOptions runner_options;
  runner_options.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  runtime::ReferenceRunnerFactoryResult runner =
      runtime::create_reference_runner(*weights.value, *request.value,
                                       runner_options);
  if (!runner) {
    std::cerr << "runner creation failed: "
              << runtime::reference_runner_error_string(
                     runner.diagnostic.error)
              << " cuda_error=" << runner.diagnostic.cuda_error
              << " operation="
              << (runner.diagnostic.operation == nullptr
                      ? ""
                      : runner.diagnostic.operation)
              << '\n';
    return 1;
  }

  bool passed = true;
  for (const Profile& profile : kProfiles) {
    passed = run_profile(*runner.value, *request.value, profile) && passed;
  }
  (void)runtime::reference_runner_detail::
      exchange_prefill_gdn_b8_admission_test_enabled(false);
  std::cout << "GDN_B8_REAL_CHECKPOINT_ADMISSION "
            << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
