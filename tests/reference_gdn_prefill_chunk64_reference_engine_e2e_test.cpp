#include "q3x/runtime/reference_engine.h"
#include "q3x/runtime/gdn_decode.h"

#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
#include "gdn_prefill_chunk64_native_sm87.h"
#include "reference_runner_gdn_chunk64_native_admission.h"
#else
#include "reference_runner_gdn_chunk64_reference_admission.h"
#endif

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace runtime = q3x::runtime;
namespace detail = q3x::runtime::reference_runner_detail;

constexpr std::size_t kPromptTokens = 513U;
constexpr std::size_t kPrefixTokens = 512U;
constexpr std::size_t kExpectedRouteHits =
    runtime::kRequestLinearLayerCount;
constexpr std::uint32_t kExpectedGeneratedToken = 9'419U;
constexpr std::string_view kExpectedGeneratedText = "Hello";
#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
constexpr std::string_view kRouteMarker = "GDN_CHUNK64_NATIVE";
constexpr std::string_view kRunEnvironment =
    "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION";
using SelectedSnapshotHook = detail::PrefillGdnChunk64NativeSnapshotHook;

[[nodiscard]] bool exchange_admission(const bool enabled) noexcept {
  return detail::exchange_prefill_gdn_chunk64_native_admission_test_enabled(
      enabled);
}

[[nodiscard]] std::size_t exchange_hits(const std::size_t hits) noexcept {
  return detail::exchange_prefill_gdn_chunk64_native_admission_test_hits(hits);
}

[[nodiscard]] SelectedSnapshotHook exchange_snapshot_hook(
    const SelectedSnapshotHook hook) noexcept {
  return detail::exchange_prefill_gdn_chunk64_native_snapshot_hook(hook);
}
#else
constexpr std::string_view kRouteMarker = "GDN_CHUNK64_REFERENCE";
constexpr std::string_view kRunEnvironment =
    "Q3X_RUN_GDN_CHUNK64_REFERENCE_ADMISSION";
using SelectedSnapshotHook = detail::PrefillGdnChunk64ReferenceSnapshotHook;

[[nodiscard]] bool exchange_admission(const bool enabled) noexcept {
  return detail::exchange_prefill_gdn_chunk64_reference_admission_test_enabled(
      enabled);
}

[[nodiscard]] std::size_t exchange_hits(const std::size_t hits) noexcept {
  return detail::exchange_prefill_gdn_chunk64_reference_admission_test_hits(
      hits);
}

[[nodiscard]] SelectedSnapshotHook exchange_snapshot_hook(
    const SelectedSnapshotHook hook) noexcept {
  return detail::exchange_prefill_gdn_chunk64_reference_snapshot_hook(hook);
}
#endif

class ScopedAdmission {
 public:
  explicit ScopedAdmission(const bool enabled) noexcept
      : previous_(exchange_admission(enabled)) {}

  ~ScopedAdmission() {
    (void)exchange_admission(previous_);
  }

  ScopedAdmission(const ScopedAdmission&) = delete;
  ScopedAdmission& operator=(const ScopedAdmission&) = delete;

 private:
  bool previous_ = false;
};

struct Sample {
  double prefix_milliseconds = 0.0;
  double ttft_milliseconds = 0.0;
  std::size_t route_hits = 0U;
  bool semantic_oracle = false;
};

struct StateSnapshot {
  std::vector<std::uint16_t> values;
  std::size_t calls = 0U;
  std::uint32_t committed_position = 0U;
  std::uint64_t region_bytes = 0U;
  int cuda_error = static_cast<int>(cudaSuccess);
  bool contract_error = false;
};

void collect_state_snapshot(const runtime::RequestState& state,
                            void* const context) noexcept {
  auto* const snapshot = static_cast<StateSnapshot*>(context);
  if (snapshot == nullptr || ++snapshot->calls != 1U) {
    if (snapshot != nullptr) {
      snapshot->contract_error = true;
    }
    return;
  }
  snapshot->committed_position = state.current_position();
  const runtime::RequestRegion& region = state.plan().gdn_state;
  snapshot->region_bytes = region.byte_size;
  const bool valid =
      state.arena_data() != nullptr &&
      region.byte_size == runtime::kRequestGdnStateBytes &&
      region.element_size_bytes == sizeof(std::uint16_t) &&
      region.element_capacity ==
          runtime::kRequestLinearLayerCount * runtime::kGdnStateElements &&
      region.arena_offset <= state.arena_bytes() &&
      region.byte_size <= state.arena_bytes() - region.arena_offset &&
      snapshot->values.size() == region.element_capacity;
  if (!valid) {
    snapshot->contract_error = true;
    return;
  }
  const auto* const source =
      static_cast<const std::uint8_t*>(state.arena_data()) +
      static_cast<std::size_t>(region.arena_offset);
  snapshot->cuda_error = static_cast<int>(cudaMemcpy(
      snapshot->values.data(), source,
      static_cast<std::size_t>(region.byte_size), cudaMemcpyDeviceToHost));
}

class ScopedSnapshotHook {
 public:
  explicit ScopedSnapshotHook(StateSnapshot& snapshot) noexcept
      : previous_(exchange_snapshot_hook(
            SelectedSnapshotHook{collect_state_snapshot, &snapshot})) {}

  ~ScopedSnapshotHook() {
    (void)exchange_snapshot_hook(previous_);
  }

  ScopedSnapshotHook(const ScopedSnapshotHook&) = delete;
  ScopedSnapshotHook& operator=(const ScopedSnapshotHook&) = delete;

 private:
  SelectedSnapshotHook previous_{};
};

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float decoded = 0.0F;
  std::memcpy(&decoded, &bits, sizeof(decoded));
  return decoded;
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
  return generation.prompt_token_ids.size() == kPromptTokens &&
         generation.generated_token_ids.size() == 1U &&
         generation.generated_token_ids.front() <
             runtime::kReferenceVocabularySize &&
         generation.stop_reason ==
             runtime::ReferenceStopReason::kMaxNewTokens &&
         generation.requested_prefill_chunk_size == kPrefixTokens &&
         generation.effective_prefill_chunk_size == kPrefixTokens &&
         generation.steps.size() == kPromptTokens &&
         generation.generated_token_ids.front() == kExpectedGeneratedToken &&
         std::string_view(generation.generated_text) == kExpectedGeneratedText;
}

[[nodiscard]] bool validate_native_resources() {
#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  const int status =
      q3x::runtime::gdn_prefill_chunk64_native_detail::query_resources(
          &registers_per_thread, &static_shared_bytes, &local_bytes,
          &maximum_threads_per_block, &active_blocks_per_sm);
  const bool passed = status == static_cast<int>(cudaSuccess) &&
                      registers_per_thread > 0 &&
                      registers_per_thread <= 255 && local_bytes == 0U &&
                      maximum_threads_per_block >= 256 &&
                      active_blocks_per_sm >= 1 &&
                      q3x::runtime::gdn_prefill_chunk64_native_detail::
                              workspace_bytes() > 0U;
  std::cout << "GDN_CHUNK64_NATIVE_RESOURCES"
            << " cuda_status=" << status
            << " registers_per_thread=" << registers_per_thread
            << " static_shared_bytes=" << static_shared_bytes
            << " local_bytes=" << local_bytes
            << " maximum_threads_per_block=" << maximum_threads_per_block
            << " active_blocks_per_sm=" << active_blocks_per_sm
            << " workspace_bytes="
            << q3x::runtime::gdn_prefill_chunk64_native_detail::
                   workspace_bytes()
            << " gate=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
#else
  return true;
#endif
}

[[nodiscard]] bool run_sample(runtime::ReferenceEngine& engine,
                              const std::string& prompt,
                              const bool candidate,
                              const std::string_view phase,
                              Sample& sample) {
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens = 1U;
  options.prefill_chunk_size = kPrefixTokens;
  options.logits_mode = runtime::ReferenceLogitsMode::kPredictedTokenOnly;

  (void)exchange_hits(0U);
  runtime::ReferenceGenerateResult result;
  {
    const ScopedAdmission admission(candidate);
    result = engine.generate(prompt, options);
  }
  sample.route_hits = exchange_hits(0U);
  if (!result) {
    std::cerr << kRouteMarker << "_SAMPLE phase=" << phase
              << " route=" << (candidate ? "candidate" : "baseline")
              << " generation failed: ";
    print_diagnostic(result.diagnostic);
    return false;
  }

  for (const double elapsed :
       result.value->timing.prefix_execution_milliseconds) {
    sample.prefix_milliseconds += elapsed;
  }
  sample.ttft_milliseconds =
      result.value->timing.time_to_first_token_milliseconds;
  sample.semantic_oracle = expected_generation(*result.value);
  const std::size_t expected_hits = candidate ? kExpectedRouteHits : 0U;
  const bool structural_oracle =
      result.value->timing.prefix_execution_milliseconds.size() == 1U &&
      std::isfinite(sample.prefix_milliseconds) &&
      sample.prefix_milliseconds > 0.0 &&
      std::isfinite(sample.ttft_milliseconds) &&
      sample.ttft_milliseconds > 0.0 && sample.route_hits == expected_hits;

  std::cout << kRouteMarker << "_SAMPLE phase=" << phase
            << " route=" << (candidate ? "candidate" : "baseline")
            << " prefix_ms=" << sample.prefix_milliseconds
            << " ttft_ms=" << sample.ttft_milliseconds
            << " generated_token="
            << (result.value->generated_token_ids.empty()
                    ? runtime::kReferenceVocabularySize
                    : result.value->generated_token_ids.front())
            << " generated_text=" << result.value->generated_text
            << " route_hits=" << sample.route_hits
            << " expected_hits=" << expected_hits
            << " structural_oracle="
            << (structural_oracle ? "PASS" : "FAIL")
            << " semantic_oracle="
            << (sample.semantic_oracle ? "PASS" : "FAIL") << '\n';
  return structural_oracle;
}

[[nodiscard]] runtime::ReferenceGenerateResult run_snapshot_generation(
    runtime::ReferenceEngine& engine,
    const std::string& prompt,
    const bool candidate,
    StateSnapshot& snapshot,
    std::size_t& route_hits) {
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens = 1U;
  options.prefill_chunk_size = kPrefixTokens;
  options.logits_mode = runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  (void)exchange_hits(0U);
  runtime::ReferenceGenerateResult result;
  {
    const ScopedSnapshotHook hook(snapshot);
    const ScopedAdmission admission(candidate);
    result = engine.generate(prompt, options);
  }
  route_hits = exchange_hits(0U);
  return result;
}

[[nodiscard]] bool valid_snapshot(const StateSnapshot& snapshot) {
  return !snapshot.contract_error &&
         snapshot.cuda_error == static_cast<int>(cudaSuccess) &&
         snapshot.calls == 1U &&
         snapshot.committed_position == kPrefixTokens &&
         snapshot.region_bytes == runtime::kRequestGdnStateBytes;
}

[[nodiscard]] bool run_state_characterization(runtime::ReferenceEngine& engine,
                                              const std::string& prompt) {
  const std::size_t element_count =
      static_cast<std::size_t>(runtime::kRequestGdnStateBytes /
                               sizeof(std::uint16_t));
  StateSnapshot baseline;
  StateSnapshot candidate;
  try {
    baseline.values.resize(element_count);
    candidate.values.resize(element_count);
  } catch (const std::bad_alloc&) {
    std::cerr << "state characterization host allocation failed\n";
    return false;
  }

  std::size_t baseline_hits = 0U;
  runtime::ReferenceGenerateResult baseline_result =
      run_snapshot_generation(engine, prompt, false, baseline,
                              baseline_hits);
  std::size_t candidate_hits = 0U;
  runtime::ReferenceGenerateResult candidate_result =
      run_snapshot_generation(engine, prompt, true, candidate,
                              candidate_hits);
  if (!baseline_result || !candidate_result) {
    std::cerr << "state characterization generation failed\n";
    if (!baseline_result) {
      print_diagnostic(baseline_result.diagnostic);
    }
    if (!candidate_result) {
      print_diagnostic(candidate_result.diagnostic);
    }
    return false;
  }

  double reference_square_sum = 0.0;
  double error_square_sum = 0.0;
  double maximum_absolute_error = 0.0;
  std::size_t unequal = 0U;
  bool finite = true;
  constexpr std::size_t kLayerElements = runtime::kGdnStateElements;
  double maximum_layer_nrmse = 0.0;
  std::size_t maximum_layer = 0U;
  for (std::size_t layer = 0U;
       layer < runtime::kRequestLinearLayerCount; ++layer) {
    double layer_reference_square_sum = 0.0;
    double layer_error_square_sum = 0.0;
    const std::size_t begin = layer * kLayerElements;
    const std::size_t end = begin + kLayerElements;
    for (std::size_t index = begin; index < end; ++index) {
      const double expected =
          static_cast<double>(decode_bf16(baseline.values[index]));
      const double actual =
          static_cast<double>(decode_bf16(candidate.values[index]));
      const double error = actual - expected;
      finite = finite && std::isfinite(expected) && std::isfinite(actual);
      reference_square_sum += expected * expected;
      error_square_sum += error * error;
      layer_reference_square_sum += expected * expected;
      layer_error_square_sum += error * error;
      maximum_absolute_error =
          std::fmax(maximum_absolute_error, std::fabs(error));
      unequal += baseline.values[index] != candidate.values[index] ? 1U : 0U;
    }
    const double layer_nrmse =
        layer_reference_square_sum > 0.0
            ? std::sqrt(layer_error_square_sum /
                        layer_reference_square_sum)
            : std::numeric_limits<double>::infinity();
    if (layer_nrmse > maximum_layer_nrmse) {
      maximum_layer_nrmse = layer_nrmse;
      maximum_layer = layer;
    }
  }
  const double nrmse =
      reference_square_sum > 0.0
          ? std::sqrt(error_square_sum / reference_square_sum)
          : std::numeric_limits<double>::infinity();
  const double unequal_fraction =
      static_cast<double>(unequal) / static_cast<double>(element_count);
  const bool semantics = expected_generation(*baseline_result.value) &&
                         expected_generation(*candidate_result.value);
  const bool passed = valid_snapshot(baseline) && valid_snapshot(candidate) &&
                      baseline_hits == 0U &&
                      candidate_hits == kExpectedRouteHits && finite &&
                      semantics && std::isfinite(nrmse);
  std::cout << kRouteMarker << "_STATE_CHARACTERIZATION"
            << " elements=" << element_count
            << " unequal_bf16=" << unequal
            << " unequal_fraction=" << unequal_fraction
            << " nrmse=" << nrmse
            << " max_abs_error=" << maximum_absolute_error
            << " max_layer_nrmse=" << maximum_layer_nrmse
            << " max_layer=" << maximum_layer
            << " baseline_hits=" << baseline_hits
            << " candidate_hits=" << candidate_hits
            << " finite=" << (finite ? "true" : "false")
            << " generation_semantics=" << (semantics ? "PASS" : "FAIL")
            << " gate=" << (passed ? "PASS" : "FAIL")
            << " authority=CHARACTERIZATION_ONLY\n";
  return passed;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr << "usage: q3x_reference_gdn_prefill_chunk64_"
#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
                 "native"
#else
                 "reference"
#endif
                 "_engine_e2e_test [MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }
  const char* const enabled = std::getenv(kRunEnvironment.data());
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "SKIP: set " << kRunEnvironment
              << "=1 to run the isolated architecture screen\n";
    return 77;
  }

  (void)exchange_admission(false);
  if (!validate_native_resources()) {
    return 6;
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

  std::cout << std::fixed << std::setprecision(9);
  const std::string prompt = repeated_hello_prompt();
  Sample baseline_warmup;
  Sample candidate_warmup;
  Sample baseline;
  Sample candidate;
  const bool profile_candidate =
      std::getenv("Q3X_GDN_CHUNK64_PROFILE_CANDIDATE") != nullptr;
  bool structural_oracle =
      run_sample(*created.value, prompt, false, "warmup", baseline_warmup) &&
      run_sample(*created.value, prompt, true, "warmup", candidate_warmup) &&
      run_sample(*created.value, prompt, false, "measured", baseline);
  if (structural_oracle && profile_candidate &&
      cudaProfilerStart() != cudaSuccess) {
    std::cerr << "cudaProfilerStart failed\n";
    structural_oracle = false;
  }
  if (structural_oracle) {
    structural_oracle =
        run_sample(*created.value, prompt, true, "measured", candidate);
  }
  if (profile_candidate && cudaProfilerStop() != cudaSuccess) {
    std::cerr << "cudaProfilerStop failed\n";
    structural_oracle = false;
  }
  if (!structural_oracle) {
    std::cout << kRouteMarker << "_DIRECTION result=INVALID\n";
    return 1;
  }

  const double prefix_saved =
      baseline.prefix_milliseconds - candidate.prefix_milliseconds;
  const double ttft_saved =
      baseline.ttft_milliseconds - candidate.ttft_milliseconds;
  const bool positive = prefix_saved > 0.0 && ttft_saved > 0.0;
  const bool semantic_oracle =
      baseline.semantic_oracle && candidate.semantic_oracle;
  std::cout << kRouteMarker << "_DIRECTION"
            << " baseline_prefix_ms=" << baseline.prefix_milliseconds
            << " candidate_prefix_ms=" << candidate.prefix_milliseconds
            << " prefix_saved_ms=" << prefix_saved
            << " prefix_speedup="
            << baseline.prefix_milliseconds / candidate.prefix_milliseconds
            << " baseline_ttft_ms=" << baseline.ttft_milliseconds
            << " candidate_ttft_ms=" << candidate.ttft_milliseconds
            << " ttft_saved_ms=" << ttft_saved
            << " ttft_speedup="
            << baseline.ttft_milliseconds / candidate.ttft_milliseconds
            << " direction=" << (positive ? "POSITIVE" : "NEGATIVE")
            << " semantic_oracle="
            << (semantic_oracle ? "PASS" : "FAIL")
            << " authority=ARCHITECTURE_SCREEN_ONLY"
            << " production_unchanged=true\n";
  if (!semantic_oracle) {
    return 4;
  }
#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
  if (!run_state_characterization(*created.value, prompt)) {
    return 5;
  }
#else
  if (std::getenv("Q3X_GDN_CHUNK64_CHARACTERIZE_STATE") != nullptr &&
      !run_state_characterization(*created.value, prompt)) {
    return 5;
  }
#endif
  return positive ? 0 : 3;
}
