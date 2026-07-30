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

constexpr std::size_t kDefaultPromptTokens = 513U;
constexpr std::size_t kPrefillChunkTokens = 512U;
constexpr std::uint32_t kExpectedGeneratedToken = 9'419U;
constexpr std::string_view kExpectedGeneratedText = "Hello";
std::size_t g_prompt_tokens = kDefaultPromptTokens;
#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
constexpr std::string_view kRouteMarker = "GDN_CHUNK64_NATIVE";
constexpr std::string_view kRunEnvironment =
    "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION";
using SelectedSnapshotHook = detail::PrefillGdnChunk64NativeSnapshotHook;

static_assert(detail::prefill_gdn_chunk64_native_prefix_token_count(63U) ==
              0U);
static_assert(detail::prefill_gdn_chunk64_native_prefix_token_count(64U) ==
              64U);
static_assert(detail::prefill_gdn_chunk64_native_prefix_token_count(127U) ==
              64U);
static_assert(detail::prefill_gdn_chunk64_native_prefix_token_count(481U) ==
              448U);
static_assert(detail::prefill_gdn_chunk64_legacy_tail_token_count(64U) == 0U);
static_assert(detail::prefill_gdn_chunk64_legacy_tail_token_count(127U) ==
              63U);
static_assert(detail::prefill_gdn_chunk64_legacy_tail_token_count(481U) ==
              33U);

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
  std::uint32_t generated_token = runtime::kReferenceVocabularySize;
  std::string generated_text;
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

#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
struct NativeBoundarySnapshot {
  std::vector<std::uint16_t> transform;
  std::vector<std::uint16_t> w;
  std::vector<std::uint16_t> u;
  std::vector<std::uint16_t> v_new;
  std::vector<std::uint16_t> boundary_state;
  std::vector<std::uint16_t> compact_k;
  std::vector<float> gamma;
  std::vector<std::uint16_t> diagnostic_k_decay;
  std::vector<std::uint16_t> diagnostic_post_update_state;
  std::vector<std::uint16_t> state;
  std::vector<std::uint16_t> output;
  std::size_t capture_call = runtime::kRequestLinearLayerCount;
  std::size_t calls = 0U;
  int cuda_error = static_cast<int>(cudaSuccess);
  bool contract_error = false;
  bool captured = false;
};

class ScopedFusedKktBaseline {
 public:
  explicit ScopedFusedKktBaseline(const bool enabled) noexcept
      : previous_(q3x::runtime::gdn_prefill_chunk64_native_detail::
                      exchange_force_fused_kkt_baseline_for_test(enabled)) {}

  ~ScopedFusedKktBaseline() {
    (void)q3x::runtime::gdn_prefill_chunk64_native_detail::
        exchange_force_fused_kkt_baseline_for_test(previous_);
  }

  [[nodiscard]] bool valid() const noexcept { return true; }

  ScopedFusedKktBaseline(const ScopedFusedKktBaseline&) = delete;
  ScopedFusedKktBaseline& operator=(const ScopedFusedKktBaseline&) = delete;

 private:
  bool previous_ = false;
};

class ScopedSplitWyBaseline {
 public:
  explicit ScopedSplitWyBaseline(const bool enabled) noexcept
      : previous_(q3x::runtime::gdn_prefill_chunk64_native_detail::
                      exchange_force_split_wy_baseline_for_test(enabled)) {}

  ~ScopedSplitWyBaseline() {
    (void)q3x::runtime::gdn_prefill_chunk64_native_detail::
        exchange_force_split_wy_baseline_for_test(previous_);
  }

  [[nodiscard]] bool valid() const noexcept { return true; }

  ScopedSplitWyBaseline(const ScopedSplitWyBaseline&) = delete;
  ScopedSplitWyBaseline& operator=(const ScopedSplitWyBaseline&) = delete;

 private:
  bool previous_ = false;
};

class ScopedPackedQkvBaseline {
 public:
  explicit ScopedPackedQkvBaseline(const bool enabled) noexcept
      : previous_(q3x::runtime::gdn_prefill_chunk64_native_detail::
                      exchange_force_packed_qkv_baseline_for_test(enabled)) {}

  ~ScopedPackedQkvBaseline() {
    (void)q3x::runtime::gdn_prefill_chunk64_native_detail::
        exchange_force_packed_qkv_baseline_for_test(previous_);
  }

  ScopedPackedQkvBaseline(const ScopedPackedQkvBaseline&) = delete;
  ScopedPackedQkvBaseline& operator=(const ScopedPackedQkvBaseline&) = delete;

 private:
  bool previous_ = false;
};

class ScopedWyTimingHook {
 public:
  ScopedWyTimingHook(cudaEvent_t begin, cudaEvent_t after_initial,
                     cudaEvent_t after_qk, cudaEvent_t after_final) noexcept
      : previous_(q3x::runtime::gdn_prefill_chunk64_native_detail::
                      exchange_wy_timing_hook(
                          {begin, after_initial, after_qk, after_final})) {}

  ~ScopedWyTimingHook() {
    (void)q3x::runtime::gdn_prefill_chunk64_native_detail::
        exchange_wy_timing_hook(previous_);
  }

  ScopedWyTimingHook(const ScopedWyTimingHook&) = delete;
  ScopedWyTimingHook& operator=(const ScopedWyTimingHook&) = delete;

 private:
  q3x::runtime::gdn_prefill_chunk64_native_detail::WyTimingHook previous_{};
};

class ScopedResidentStateBaseline {
 public:
  explicit ScopedResidentStateBaseline(const bool enabled) noexcept
      : previous_(q3x::runtime::gdn_prefill_chunk64_native_detail::
                      exchange_force_resident_state_baseline_for_test(
                          enabled)) {}

  ~ScopedResidentStateBaseline() {
    (void)q3x::runtime::gdn_prefill_chunk64_native_detail::
        exchange_force_resident_state_baseline_for_test(previous_);
  }

  [[nodiscard]] bool valid() const noexcept { return true; }

  ScopedResidentStateBaseline(const ScopedResidentStateBaseline&) = delete;
  ScopedResidentStateBaseline& operator=(
      const ScopedResidentStateBaseline&) = delete;

 private:
  bool previous_ = false;
};

class ScopedPacklessResidentStateFallback {
 public:
  explicit ScopedPacklessResidentStateFallback(const bool enabled) noexcept
      : previous_(q3x::runtime::gdn_prefill_chunk64_native_detail::
                      exchange_force_packless_resident_state_fallback_for_test(
                          enabled)) {}

  ~ScopedPacklessResidentStateFallback() {
    (void)q3x::runtime::gdn_prefill_chunk64_native_detail::
        exchange_force_packless_resident_state_fallback_for_test(previous_);
  }

  ScopedPacklessResidentStateFallback(
      const ScopedPacklessResidentStateFallback&) = delete;
  ScopedPacklessResidentStateFallback& operator=(
      const ScopedPacklessResidentStateFallback&) = delete;

 private:
  bool previous_ = false;
};

void collect_native_boundaries(
    const std::uint16_t* const transform,
    const std::size_t transform_elements,
    const std::uint16_t* const w,
    const std::size_t w_elements,
    const std::uint16_t* const u,
    const std::size_t u_elements,
    const std::uint16_t* const v_new,
    const std::size_t v_new_elements,
    const std::uint16_t* const boundary_state,
    const std::size_t boundary_state_elements,
    const std::uint16_t* const compact_k,
    const std::size_t compact_k_elements,
    const float* const gamma,
    const std::size_t gamma_elements,
    const std::uint16_t* const diagnostic_k_decay,
    const std::size_t diagnostic_k_decay_elements,
    const std::uint16_t* const diagnostic_post_update_state,
    const std::size_t diagnostic_post_update_state_elements,
    const std::uint16_t* const state_output,
    const std::size_t state_elements,
    const std::uint16_t* const output,
    const std::size_t output_elements,
    void* const cuda_stream,
    void* const context) noexcept {
  auto* const snapshot = static_cast<NativeBoundarySnapshot*>(context);
  if (snapshot == nullptr) {
    return;
  }
  ++snapshot->calls;
  if (snapshot->calls != snapshot->capture_call) {
    return;
  }
  const bool valid =
      snapshot->capture_call != 0U &&
      snapshot->capture_call <= runtime::kRequestLinearLayerCount &&
      (snapshot->transform.empty() ||
       (transform != nullptr &&
        snapshot->transform.size() == transform_elements)) &&
      (snapshot->w.empty() ||
       (w != nullptr && snapshot->w.size() == w_elements)) &&
      (snapshot->u.empty() ||
       (u != nullptr && snapshot->u.size() == u_elements)) &&
      (snapshot->v_new.empty() ||
       (v_new != nullptr && snapshot->v_new.size() == v_new_elements)) &&
      (snapshot->boundary_state.empty() ||
       (boundary_state != nullptr &&
        snapshot->boundary_state.size() == boundary_state_elements)) &&
      (snapshot->compact_k.empty() ||
       (compact_k != nullptr &&
        snapshot->compact_k.size() == compact_k_elements)) &&
      (snapshot->gamma.empty() ||
       (gamma != nullptr && snapshot->gamma.size() == gamma_elements)) &&
      (snapshot->diagnostic_k_decay.empty() ||
       (diagnostic_k_decay != nullptr &&
        snapshot->diagnostic_k_decay.size() ==
            diagnostic_k_decay_elements)) &&
      (snapshot->diagnostic_post_update_state.empty() ||
       (diagnostic_post_update_state != nullptr &&
        snapshot->diagnostic_post_update_state.size() ==
            diagnostic_post_update_state_elements)) &&
      (snapshot->state.empty() ||
       (state_output != nullptr && snapshot->state.size() == state_elements)) &&
      (snapshot->output.empty() ||
       (output != nullptr && snapshot->output.size() == output_elements));
  if (!valid) {
    snapshot->contract_error = true;
    return;
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = cudaSuccess;
  const auto copy_if_requested =
      [&](std::vector<std::uint16_t>& destination,
          const std::uint16_t* const source,
          const std::size_t elements) noexcept {
        if (status == cudaSuccess && !destination.empty()) {
          status = cudaMemcpyAsync(
              destination.data(), source,
              elements * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
              stream);
        }
      };
  copy_if_requested(snapshot->transform, transform, transform_elements);
  copy_if_requested(snapshot->w, w, w_elements);
  copy_if_requested(snapshot->u, u, u_elements);
  copy_if_requested(snapshot->v_new, v_new, v_new_elements);
  copy_if_requested(snapshot->boundary_state, boundary_state,
                    boundary_state_elements);
  copy_if_requested(snapshot->compact_k, compact_k, compact_k_elements);
  if (status == cudaSuccess && !snapshot->gamma.empty()) {
    status = cudaMemcpyAsync(
        snapshot->gamma.data(), gamma,
        gamma_elements * sizeof(float), cudaMemcpyDeviceToHost, stream);
  }
  copy_if_requested(snapshot->diagnostic_k_decay, diagnostic_k_decay,
                    diagnostic_k_decay_elements);
  copy_if_requested(snapshot->diagnostic_post_update_state,
                    diagnostic_post_update_state,
                    diagnostic_post_update_state_elements);
  copy_if_requested(snapshot->state, state_output, state_elements);
  copy_if_requested(snapshot->output, output, output_elements);
  if (status == cudaSuccess) {
    status = cudaStreamSynchronize(stream);
  }
  snapshot->cuda_error = static_cast<int>(status);
  snapshot->captured = status == cudaSuccess;
}

class ScopedNativeInspectionHook {
 public:
  explicit ScopedNativeInspectionHook(
      NativeBoundarySnapshot& snapshot) noexcept
      : previous_(q3x::runtime::gdn_prefill_chunk64_native_detail::
                      exchange_inspection_hook(
                          {collect_native_boundaries, &snapshot})) {}

  ~ScopedNativeInspectionHook() {
    (void)q3x::runtime::gdn_prefill_chunk64_native_detail::
        exchange_inspection_hook(previous_);
  }

  ScopedNativeInspectionHook(const ScopedNativeInspectionHook&) = delete;
  ScopedNativeInspectionHook& operator=(
      const ScopedNativeInspectionHook&) = delete;

 private:
  q3x::runtime::gdn_prefill_chunk64_native_detail::InspectionHook previous_{};
};
#endif

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

#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
class ScopedGenerateReturnSnapshotHook {
 public:
  explicit ScopedGenerateReturnSnapshotHook(
      StateSnapshot& snapshot) noexcept
      : previous_(detail::
            exchange_reference_engine_generate_return_snapshot_hook(
                {collect_state_snapshot, &snapshot})) {}

  ~ScopedGenerateReturnSnapshotHook() {
    (void)detail::exchange_reference_engine_generate_return_snapshot_hook(
        previous_);
  }

  ScopedGenerateReturnSnapshotHook(
      const ScopedGenerateReturnSnapshotHook&) = delete;
  ScopedGenerateReturnSnapshotHook& operator=(
      const ScopedGenerateReturnSnapshotHook&) = delete;

 private:
  detail::ReferenceEngineGenerateReturnSnapshotHook previous_{};
};
#endif

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float decoded = 0.0F;
  std::memcpy(&decoded, &bits, sizeof(decoded));
  return decoded;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
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
  const std::size_t words = g_prompt_tokens - 12U;
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

[[nodiscard]] bool expected_generation(
    const runtime::ReferenceGeneration& generation) {
  const bool generic =
      generation.prompt_token_ids.size() == g_prompt_tokens &&
      generation.generated_token_ids.size() == 1U &&
      generation.generated_token_ids.front() <
          runtime::kReferenceVocabularySize &&
      generation.stop_reason ==
          runtime::ReferenceStopReason::kMaxNewTokens &&
      generation.requested_prefill_chunk_size == kPrefillChunkTokens &&
      generation.effective_prefill_chunk_size == kPrefillChunkTokens &&
      generation.steps.size() == g_prompt_tokens;
  return generic &&
         (g_prompt_tokens != kDefaultPromptTokens ||
          (generation.generated_token_ids.front() == kExpectedGeneratedToken &&
           std::string_view(generation.generated_text) ==
               kExpectedGeneratedText));
}

[[nodiscard]] std::size_t expected_prefix_executions() noexcept {
  return runtime::reference_engine_detail::prefix_execution_count(
      g_prompt_tokens - 1U, kPrefillChunkTokens);
}

[[nodiscard]] std::size_t expected_native_route_hits() noexcept {
  std::size_t remaining = g_prompt_tokens - 1U;
  std::size_t admitted_tiles = 0U;
  while (remaining != 0U) {
    const std::size_t tile =
        runtime::reference_engine_detail::next_prefix_tile_token_count(
            remaining, kPrefillChunkTokens);
#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
    const bool admitted = tile >= 64U;
#else
    const bool admitted = tile >= 64U && tile % 64U == 0U;
#endif
    if (admitted) {
      ++admitted_tiles;
    }
    remaining -= tile;
  }
  return admitted_tiles * runtime::kRequestLinearLayerCount;
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
                      maximum_threads_per_block >= 128 &&
                      active_blocks_per_sm >= 2 &&
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
  options.prefill_chunk_size = kPrefillChunkTokens;
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
  if (!result.value->generated_token_ids.empty()) {
    sample.generated_token = result.value->generated_token_ids.front();
  }
  sample.generated_text = result.value->generated_text;
  sample.semantic_oracle = expected_generation(*result.value);
  const std::size_t expected_hits =
      candidate ? expected_native_route_hits() : 0U;
  const bool structural_oracle =
      result.value->timing.prefix_execution_milliseconds.size() ==
          expected_prefix_executions() &&
      std::isfinite(sample.prefix_milliseconds) &&
      sample.prefix_milliseconds > 0.0 &&
      std::isfinite(sample.ttft_milliseconds) &&
      sample.ttft_milliseconds > 0.0 && sample.route_hits == expected_hits;

  std::cout << kRouteMarker << "_SAMPLE phase=" << phase
            << " prompt_tokens=" << g_prompt_tokens
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
  options.prefill_chunk_size = kPrefillChunkTokens;
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
         snapshot.committed_position == g_prompt_tokens - 1U &&
         snapshot.region_bytes == runtime::kRequestGdnStateBytes;
}

[[nodiscard]] bool valid_snapshot_contract(const StateSnapshot& snapshot) {
  return !snapshot.contract_error &&
         snapshot.cuda_error == static_cast<int>(cudaSuccess) &&
         snapshot.calls == 1U &&
         snapshot.region_bytes == runtime::kRequestGdnStateBytes;
}

#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
struct DifferenceMetrics {
  std::size_t elements = 0U;
  std::size_t unequal_bf16 = 0U;
  std::size_t nonfinite = 0U;
  double maximum_absolute_error = 0.0;
  double nrmse = 0.0;
  double cosine = 0.0;
};

[[nodiscard]] DifferenceMetrics compare_bf16_range(
    const std::vector<std::uint16_t>& baseline,
    const std::vector<std::uint16_t>& candidate,
    const std::size_t begin, const std::size_t end) {
  DifferenceMetrics metrics;
  if (baseline.size() != candidate.size() || begin > end ||
      end > baseline.size()) {
    metrics.nonfinite = 1U;
    return metrics;
  }
  metrics.elements = end - begin;
  double baseline_square_sum = 0.0;
  double candidate_square_sum = 0.0;
  double error_square_sum = 0.0;
  double dot = 0.0;
  for (std::size_t index = begin; index < end; ++index) {
    const double expected =
        static_cast<double>(decode_bf16(baseline[index]));
    const double actual =
        static_cast<double>(decode_bf16(candidate[index]));
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
      ++metrics.nonfinite;
      continue;
    }
    const double error = actual - expected;
    baseline_square_sum += expected * expected;
    candidate_square_sum += actual * actual;
    error_square_sum += error * error;
    dot += expected * actual;
    metrics.maximum_absolute_error =
        std::fmax(metrics.maximum_absolute_error, std::fabs(error));
    metrics.unequal_bf16 += baseline[index] != candidate[index] ? 1U : 0U;
  }
  metrics.nrmse =
      baseline_square_sum > 0.0
          ? std::sqrt(error_square_sum / baseline_square_sum)
          : (error_square_sum == 0.0
                 ? 0.0
                 : std::numeric_limits<double>::infinity());
  metrics.cosine =
      baseline_square_sum > 0.0 && candidate_square_sum > 0.0
          ? dot / std::sqrt(baseline_square_sum * candidate_square_sum)
          : (baseline_square_sum == candidate_square_sum ? 1.0 : 0.0);
  return metrics;
}

[[nodiscard]] DifferenceMetrics compare_bf16(
    const std::vector<std::uint16_t>& baseline,
    const std::vector<std::uint16_t>& candidate) {
  return compare_bf16_range(baseline, candidate, 0U, baseline.size());
}

void print_difference_metrics(const std::string_view suite,
                              const std::string_view region,
                              const DifferenceMetrics& metrics) {
  std::cout << suite
            << " region=" << region
            << " elements=" << metrics.elements
            << " unequal_bf16=" << metrics.unequal_bf16
            << " max_abs_error=" << metrics.maximum_absolute_error
            << " nrmse=" << metrics.nrmse
            << " cosine=" << metrics.cosine
            << " nonfinite=" << metrics.nonfinite << '\n';
}

[[nodiscard]] bool exact_metrics(const DifferenceMetrics& metrics) {
  return metrics.elements != 0U && metrics.unequal_bf16 == 0U &&
         metrics.nonfinite == 0U && metrics.maximum_absolute_error == 0.0 &&
         metrics.nrmse == 0.0 && metrics.cosine == 1.0;
}

[[nodiscard]] std::size_t first_unequal_bf16(
    const std::vector<std::uint16_t>& baseline,
    const std::vector<std::uint16_t>& candidate,
    const std::size_t begin, const std::size_t end) noexcept {
  if (baseline.size() != candidate.size() || end > baseline.size() ||
      begin > end) {
    return std::numeric_limits<std::size_t>::max();
  }
  for (std::size_t index = begin; index < end; ++index) {
    if (baseline[index] != candidate[index]) {
      return index;
    }
  }
  return std::numeric_limits<std::size_t>::max();
}

void destroy_event(cudaEvent_t& event) noexcept {
  if (event != nullptr) {
    (void)cudaEventDestroy(event);
    event = nullptr;
  }
}

[[nodiscard]] bool valid_group_wy_event_sample(
    const Sample& sample) noexcept {
  return sample.semantic_oracle &&
         sample.route_hits == expected_native_route_hits() &&
         std::isfinite(sample.prefix_milliseconds) &&
         sample.prefix_milliseconds > 0.0 &&
         std::isfinite(sample.ttft_milliseconds) &&
         sample.ttft_milliseconds > 0.0;
}

[[nodiscard]] bool run_timed_group_wy_sample(
    runtime::ReferenceEngine& engine, const std::string& prompt,
    const bool packed_baseline, const std::string_view phase, Sample& sample,
    float& group_milliseconds, float& qk_milliseconds) {
  cudaEvent_t begin = nullptr;
  cudaEvent_t after_initial = nullptr;
  cudaEvent_t after_qk = nullptr;
  cudaEvent_t after_final = nullptr;
  bool ready = cudaEventCreate(&begin) == cudaSuccess &&
               cudaEventCreate(&after_initial) == cudaSuccess &&
               cudaEventCreate(&after_qk) == cudaSuccess &&
               cudaEventCreate(&after_final) == cudaSuccess;
  if (ready) {
    const ScopedFusedKktBaseline fused_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(packed_baseline);
    const ScopedResidentStateBaseline resident_route(false);
    const ScopedWyTimingHook timing(begin, after_initial, after_qk,
                                    after_final);
    (void)run_sample(engine, prompt, true, phase, sample);
    // The event gate owns a stricter native-route contract and does not rely
    // on the legacy harness's prefix-execution-count convention.
    ready = valid_group_wy_event_sample(sample);
  }
  if (ready) {
    ready = cudaEventSynchronize(after_final) == cudaSuccess &&
            cudaEventElapsedTime(&group_milliseconds, begin,
                                 after_initial) == cudaSuccess &&
            cudaEventElapsedTime(&qk_milliseconds, after_initial,
                                 after_qk) == cudaSuccess;
  }
  destroy_event(after_final);
  destroy_event(after_qk);
  destroy_event(after_initial);
  destroy_event(begin);
  return ready;
}

[[nodiscard]] bool run_native_group_wy_event_attribution(
    runtime::ReferenceEngine& engine, const std::string& prompt) {
  if (g_prompt_tokens != kDefaultPromptTokens) {
    std::cerr << "group-WY event attribution requires real P513\n";
    return false;
  }
  Sample baseline_warmup;
  Sample candidate_warmup;
  bool ready = false;
  {
    const ScopedFusedKktBaseline fused_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(true);
    const ScopedResidentStateBaseline resident_route(false);
    (void)run_sample(engine, prompt, true, "packless_event_baseline_warmup",
                     baseline_warmup);
    ready = valid_group_wy_event_sample(baseline_warmup);
  }
  {
    const ScopedFusedKktBaseline fused_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(false);
    const ScopedResidentStateBaseline resident_route(false);
    (void)run_sample(engine, prompt, true, "packless_event_candidate_warmup",
                     candidate_warmup);
    ready = valid_group_wy_event_sample(candidate_warmup) && ready;
  }

  Sample baseline;
  Sample candidate;
  float baseline_group = 0.0F;
  float baseline_qk = 0.0F;
  float candidate_group = 0.0F;
  float candidate_qk = 0.0F;
  ready = run_timed_group_wy_sample(
              engine, prompt, true, "packless_event_baseline", baseline,
              baseline_group, baseline_qk) &&
          ready;
  ready = run_timed_group_wy_sample(
              engine, prompt, false, "packless_event_candidate", candidate,
              candidate_group, candidate_qk) &&
          ready;
  const float baseline_total = baseline_group + baseline_qk;
  const float candidate_total = candidate_group + candidate_qk;
  const float saved = baseline_total - candidate_total;
  const bool semantics =
      baseline.semantic_oracle && candidate.semantic_oracle &&
      baseline.generated_token == candidate.generated_token &&
      baseline.generated_text == candidate.generated_text;
  const bool passed = ready && semantics && std::isfinite(baseline_total) &&
                      std::isfinite(candidate_total) &&
                      baseline_total > 0.0F && candidate_total > 0.0F &&
                      saved > 0.0F;
  std::cout << "GDN_CHUNK64_NATIVE_GROUP_WY_EVENT_ATTRIBUTION"
            << " baseline_group_ms=" << baseline_group
            << " baseline_qk_ms=" << baseline_qk
            << " baseline_core_window_ms=" << baseline_total
            << " candidate_group_ms=" << candidate_group
            << " candidate_qk_ms=" << candidate_qk
            << " candidate_core_window_ms=" << candidate_total
            << " saved_ms=" << saved
            << " speedup=" << baseline_total / candidate_total
            << " generation_semantics=" << (semantics ? "PASS" : "FAIL")
            << " gate=" << (passed ? "PASS" : "FAIL")
            << " authority=REAL_WEIGHT_FINAL_LAYER_PACKED_VS_PACKLESS_CORE_WINDOW\n";
  return passed;
}

[[nodiscard]] bool run_native_kkt_equivalence(
    runtime::ReferenceEngine& engine,
    const std::string& prompt) {
  if (g_prompt_tokens != kDefaultPromptTokens) {
    std::cerr << "group-WY equivalence requires the real P513 route\n";
    return false;
  }
  constexpr std::size_t kNativeTokens = kDefaultPromptTokens - 1U;
  constexpr std::size_t kTransformElements =
      (kNativeTokens / 64U) * runtime::kGdnValueHeadCount * 64U * 64U;
  constexpr std::size_t kOutputElements =
      kNativeTokens * runtime::kGdnVElements;

  StateSnapshot baseline_state;
  StateSnapshot candidate_state;
  NativeBoundarySnapshot baseline_boundaries;
  NativeBoundarySnapshot candidate_boundaries;
  try {
    const std::size_t request_state_elements =
        static_cast<std::size_t>(runtime::kRequestGdnStateBytes /
                                 sizeof(std::uint16_t));
    baseline_state.values.resize(request_state_elements);
    candidate_state.values.resize(request_state_elements);
    baseline_boundaries.transform.resize(kTransformElements);
    candidate_boundaries.transform.resize(kTransformElements);
    baseline_boundaries.w.resize(kOutputElements);
    candidate_boundaries.w.resize(kOutputElements);
    baseline_boundaries.u.resize(kOutputElements);
    candidate_boundaries.u.resize(kOutputElements);
    baseline_boundaries.state.resize(runtime::kGdnStateElements);
    candidate_boundaries.state.resize(runtime::kGdnStateElements);
    baseline_boundaries.output.resize(kOutputElements);
    candidate_boundaries.output.resize(kOutputElements);
  } catch (const std::bad_alloc&) {
    std::cerr << "group-WY equivalence host allocation failed\n";
    return false;
  }

  std::size_t baseline_hits = 0U;
  runtime::ReferenceGenerateResult baseline_result;
  {
    const ScopedFusedKktBaseline fused_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(true);
    if (!fused_route.valid() || !split_route.valid()) {
      std::cerr << "failed to select d51 packed-QKV baseline\n";
      return false;
    }
    const ScopedNativeInspectionHook hook(baseline_boundaries);
    baseline_result = run_snapshot_generation(
        engine, prompt, true, baseline_state, baseline_hits);
  }

  std::size_t candidate_hits = 0U;
  runtime::ReferenceGenerateResult candidate_result;
  {
    const ScopedFusedKktBaseline fused_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(false);
    if (!fused_route.valid() || !split_route.valid()) {
      std::cerr << "failed to select compact packless candidate\n";
      return false;
    }
    const ScopedNativeInspectionHook hook(candidate_boundaries);
    candidate_result = run_snapshot_generation(
        engine, prompt, true, candidate_state, candidate_hits);
  }
  if (!baseline_result || !candidate_result) {
    std::cerr << "group-WY equivalence generation failed\n";
    if (!baseline_result) {
      print_diagnostic(baseline_result.diagnostic);
    }
    if (!candidate_result) {
      print_diagnostic(candidate_result.diagnostic);
    }
    return false;
  }

  const DifferenceMetrics transform = compare_bf16(
      baseline_boundaries.transform, candidate_boundaries.transform);
  const DifferenceMetrics w = compare_bf16(
      baseline_boundaries.w, candidate_boundaries.w);
  const DifferenceMetrics u = compare_bf16(
      baseline_boundaries.u, candidate_boundaries.u);
  const DifferenceMetrics final_layer_state = compare_bf16(
      baseline_boundaries.state, candidate_boundaries.state);
  const DifferenceMetrics final_layer_output = compare_bf16(
      baseline_boundaries.output, candidate_boundaries.output);
  const DifferenceMetrics full_request_state = compare_bf16(
      baseline_state.values, candidate_state.values);
  constexpr std::string_view kSuite =
      "GDN_CHUNK64_NATIVE_PACKLESS_EQUIVALENCE";
  print_difference_metrics(kSuite, "transform", transform);
  print_difference_metrics(kSuite, "w", w);
  print_difference_metrics(kSuite, "u", u);
  print_difference_metrics(kSuite, "final_layer_state", final_layer_state);
  print_difference_metrics(kSuite, "final_layer_output", final_layer_output);
  print_difference_metrics(kSuite, "full_request_state", full_request_state);

  const bool boundary_contract =
      !baseline_boundaries.contract_error &&
      !candidate_boundaries.contract_error &&
      baseline_boundaries.cuda_error == static_cast<int>(cudaSuccess) &&
      candidate_boundaries.cuda_error == static_cast<int>(cudaSuccess) &&
      baseline_boundaries.calls == runtime::kRequestLinearLayerCount &&
      candidate_boundaries.calls == runtime::kRequestLinearLayerCount;
  const bool generation_semantics =
      expected_generation(*baseline_result.value) &&
      expected_generation(*candidate_result.value) &&
      baseline_result.value->generated_token_ids ==
          candidate_result.value->generated_token_ids &&
      baseline_result.value->generated_text ==
          candidate_result.value->generated_text;
  const bool passed =
      valid_snapshot(baseline_state) && valid_snapshot(candidate_state) &&
      baseline_hits == expected_native_route_hits() &&
      candidate_hits == expected_native_route_hits() && boundary_contract &&
      generation_semantics && exact_metrics(transform) && exact_metrics(w) &&
      exact_metrics(u) &&
      exact_metrics(final_layer_state) && exact_metrics(final_layer_output) &&
      exact_metrics(full_request_state);
  std::cout << kSuite
            << " baseline_hits=" << baseline_hits
            << " candidate_hits=" << candidate_hits
            << " boundary_contract="
            << (boundary_contract ? "PASS" : "FAIL")
            << " generation_semantics="
            << (generation_semantics ? "PASS" : "FAIL")
            << " gate=" << (passed ? "PASS" : "FAIL")
            << " authority=REAL_WEIGHT_PACKLESS_VS_D51_PACKED\n";
  return passed;
}

[[nodiscard]] bool run_native_resident_state_equivalence(
    runtime::ReferenceEngine& engine,
    const std::string& prompt) {
  if (g_prompt_tokens != kDefaultPromptTokens) {
    std::cerr << "resident-state equivalence requires the real P513 route\n";
    return false;
  }
  constexpr std::size_t kNativeTokens = kDefaultPromptTokens - 1U;
  constexpr std::size_t kTransformElements =
      (kNativeTokens / 64U) * runtime::kGdnValueHeadCount * 64U * 64U;
  constexpr std::size_t kOutputElements =
      kNativeTokens * runtime::kGdnVElements;

  StateSnapshot baseline_state;
  StateSnapshot candidate_state;
  NativeBoundarySnapshot baseline_boundaries;
  NativeBoundarySnapshot candidate_boundaries;
  try {
    const std::size_t request_state_elements =
        static_cast<std::size_t>(runtime::kRequestGdnStateBytes /
                                 sizeof(std::uint16_t));
    baseline_state.values.resize(request_state_elements);
    candidate_state.values.resize(request_state_elements);
    baseline_boundaries.transform.resize(kTransformElements);
    candidate_boundaries.transform.resize(kTransformElements);
    baseline_boundaries.w.resize(kOutputElements);
    candidate_boundaries.w.resize(kOutputElements);
    baseline_boundaries.u.resize(kOutputElements);
    candidate_boundaries.u.resize(kOutputElements);
    baseline_boundaries.state.resize(runtime::kGdnStateElements);
    candidate_boundaries.state.resize(runtime::kGdnStateElements);
    baseline_boundaries.output.resize(kOutputElements);
    candidate_boundaries.output.resize(kOutputElements);
  } catch (const std::bad_alloc&) {
    std::cerr << "resident-state equivalence host allocation failed\n";
    return false;
  }

  std::size_t baseline_hits = 0U;
  runtime::ReferenceGenerateResult baseline_result;
  {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedResidentStateBaseline route(true);
    const ScopedNativeInspectionHook hook(baseline_boundaries);
    baseline_result = run_snapshot_generation(
        engine, prompt, true, baseline_state, baseline_hits);
  }

  std::size_t candidate_hits = 0U;
  runtime::ReferenceGenerateResult candidate_result;
  {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedResidentStateBaseline route(false);
    const ScopedNativeInspectionHook hook(candidate_boundaries);
    candidate_result = run_snapshot_generation(
        engine, prompt, true, candidate_state, candidate_hits);
  }
  if (!baseline_result || !candidate_result) {
    std::cerr << "resident-state equivalence generation failed\n";
    if (!baseline_result) {
      print_diagnostic(baseline_result.diagnostic);
    }
    if (!candidate_result) {
      print_diagnostic(candidate_result.diagnostic);
    }
    return false;
  }

  const DifferenceMetrics transform = compare_bf16(
      baseline_boundaries.transform, candidate_boundaries.transform);
  const DifferenceMetrics w = compare_bf16(
      baseline_boundaries.w, candidate_boundaries.w);
  const DifferenceMetrics u = compare_bf16(
      baseline_boundaries.u, candidate_boundaries.u);
  const DifferenceMetrics final_layer_state = compare_bf16(
      baseline_boundaries.state, candidate_boundaries.state);
  const DifferenceMetrics final_layer_output = compare_bf16(
      baseline_boundaries.output, candidate_boundaries.output);
  const DifferenceMetrics full_request_state = compare_bf16(
      baseline_state.values, candidate_state.values);
  constexpr std::string_view kSuite =
      "GDN_CHUNK64_NATIVE_RESIDENT_STATE_EQUIVALENCE";
  print_difference_metrics(kSuite, "transform", transform);
  print_difference_metrics(kSuite, "w", w);
  print_difference_metrics(kSuite, "u", u);
  print_difference_metrics(kSuite, "final_layer_state", final_layer_state);
  print_difference_metrics(kSuite, "final_layer_output", final_layer_output);
  print_difference_metrics(kSuite, "full_request_state", full_request_state);

  const bool boundary_contract =
      !baseline_boundaries.contract_error &&
      !candidate_boundaries.contract_error &&
      baseline_boundaries.cuda_error == static_cast<int>(cudaSuccess) &&
      candidate_boundaries.cuda_error == static_cast<int>(cudaSuccess) &&
      baseline_boundaries.calls == runtime::kRequestLinearLayerCount &&
      candidate_boundaries.calls == runtime::kRequestLinearLayerCount;
  const bool generation_semantics =
      expected_generation(*baseline_result.value) &&
      expected_generation(*candidate_result.value) &&
      baseline_result.value->generated_token_ids ==
          candidate_result.value->generated_token_ids &&
      baseline_result.value->generated_text ==
          candidate_result.value->generated_text;
  const bool passed =
      valid_snapshot(baseline_state) && valid_snapshot(candidate_state) &&
      baseline_hits == expected_native_route_hits() &&
      candidate_hits == expected_native_route_hits() && boundary_contract &&
      generation_semantics && exact_metrics(transform) && exact_metrics(w) &&
      exact_metrics(u) &&
      exact_metrics(final_layer_state) && exact_metrics(final_layer_output) &&
      exact_metrics(full_request_state);
  std::cout << kSuite
            << " baseline_hits=" << baseline_hits
            << " candidate_hits=" << candidate_hits
            << " boundary_contract="
            << (boundary_contract ? "PASS" : "FAIL")
            << " generation_semantics="
            << (generation_semantics ? "PASS" : "FAIL")
            << " gate=" << (passed ? "PASS" : "FAIL")
            << " authority=REAL_WEIGHT_CANDIDATE_VS_BASELINE\n";
  return passed;
}

// Narrow real-weight diagnostic for a structural recurrence candidate. It
// captures only the first GDN layer invocation and walks the recurrent
// boundaries in execution order: H before chunk i, Vnew for chunk i, then
// final H. This deliberately does not build a full correctness harness.
[[nodiscard]] bool run_vllm_faithful_state_call1_snapshot(
    runtime::ReferenceEngine& engine, const std::string& prompt) {
  if (g_prompt_tokens != kDefaultPromptTokens) {
    std::cerr << "vLLM-faithful snapshot requires the real P513 route\n";
    return false;
  }
  constexpr std::size_t kNativeTokens = kDefaultPromptTokens - 1U;
  constexpr std::size_t kChunkCount = kNativeTokens / 64U;
  constexpr std::size_t kChunkVNewElements =
      64U * runtime::kGdnVElements;
  constexpr std::size_t kVNewElements =
      kChunkCount * kChunkVNewElements;
  constexpr std::size_t kBoundaryElements =
      kChunkCount * runtime::kGdnStateElements;
  constexpr std::size_t kCompactKElements =
      kChunkCount * runtime::kGdnQkHeadCount * 64U * 128U;
  constexpr std::size_t kGammaElements =
      kChunkCount * runtime::kGdnValueHeadCount * 64U;

  NativeBoundarySnapshot baseline;
  NativeBoundarySnapshot candidate;
  baseline.capture_call = 1U;
  candidate.capture_call = 1U;
  try {
    baseline.v_new.resize(kVNewElements);
    candidate.v_new.resize(kVNewElements);
    baseline.boundary_state.resize(kBoundaryElements);
    candidate.boundary_state.resize(kBoundaryElements);
    baseline.state.resize(runtime::kGdnStateElements);
    candidate.state.resize(runtime::kGdnStateElements);
    candidate.compact_k.resize(kCompactKElements);
    candidate.gamma.resize(kGammaElements);
    candidate.diagnostic_k_decay.resize(kVNewElements);
    candidate.diagnostic_post_update_state.resize(
        runtime::kGdnStateElements);
  } catch (const std::bad_alloc&) {
    std::cerr << "vLLM-faithful snapshot host allocation failed\n";
    return false;
  }

  Sample baseline_sample;
  Sample candidate_sample;
  bool baseline_completed = false;
  bool candidate_completed = false;
  {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(false);
    const ScopedResidentStateBaseline resident_route(false);
    const ScopedPacklessResidentStateFallback packless_fallback(true);
    const ScopedNativeInspectionHook hook(baseline);
    baseline_completed = run_sample(
        engine, prompt, true, "faithful_call1_baseline", baseline_sample);
  }
  {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(false);
    const ScopedResidentStateBaseline resident_route(false);
    const ScopedPacklessResidentStateFallback packless_fallback(false);
    const ScopedNativeInspectionHook hook(candidate);
    candidate_completed = run_sample(
        engine, prompt, true, "faithful_call1_candidate", candidate_sample);
  }

  const bool snapshot_contract =
      baseline.calls == runtime::kRequestLinearLayerCount &&
      candidate.calls == runtime::kRequestLinearLayerCount &&
      baseline.captured && candidate.captured &&
      !baseline.contract_error && !candidate.contract_error &&
      baseline.cuda_error == static_cast<int>(cudaSuccess) &&
      candidate.cuda_error == static_cast<int>(cudaSuccess);
  // Full-production prompt tiling can make the older timing-vector shape
  // check in run_sample return false even when generation and the snapshot
  // both completed. The narrow gate is the captured call contract plus the
  // exact native route-hit count; the two booleans remain diagnostic only.
  const bool ready =
      snapshot_contract &&
      baseline_sample.route_hits == expected_native_route_hits() &&
      candidate_sample.route_hits == expected_native_route_hits();

  constexpr std::string_view kSuite =
      "GDN_CHUNK64_VLLM_FAITHFUL_CALL1_SNAPSHOT";
  const DifferenceMetrics v_new =
      compare_bf16(baseline.v_new, candidate.v_new);
  const DifferenceMetrics boundary =
      compare_bf16(baseline.boundary_state, candidate.boundary_state);
  const DifferenceMetrics final_state =
      compare_bf16(baseline.state, candidate.state);
  print_difference_metrics(kSuite, "v_new", v_new);
  print_difference_metrics(kSuite, "boundary_state", boundary);
  print_difference_metrics(kSuite, "final_state", final_state);

  std::vector<std::uint16_t> expected_k_decay;
  std::vector<std::uint16_t> observed_k_decay;
  std::vector<std::uint16_t> baseline_h1;
  std::vector<std::uint16_t> candidate_boundary_h1;
  try {
    expected_k_decay.resize(kChunkVNewElements);
    observed_k_decay.assign(
        candidate.diagnostic_k_decay.begin(),
        candidate.diagnostic_k_decay.begin() + kChunkVNewElements);
    baseline_h1.assign(
        baseline.boundary_state.begin() + runtime::kGdnStateElements,
        baseline.boundary_state.begin() +
            2U * runtime::kGdnStateElements);
    candidate_boundary_h1.assign(
        candidate.boundary_state.begin() + runtime::kGdnStateElements,
        candidate.boundary_state.begin() +
            2U * runtime::kGdnStateElements);
  } catch (const std::bad_alloc&) {
    std::cerr << "vLLM-faithful component comparison allocation failed\n";
    return false;
  }
  for (std::size_t head = 0U;
       head < runtime::kGdnValueHeadCount; ++head) {
    const std::size_t compact_head = head / 3U;
    const float g_last = candidate.gamma[
        head * 64U + 63U];
    for (std::size_t token = 0U; token < 64U; ++token) {
      const float scale = std::exp(
          g_last - candidate.gamma[head * 64U + token]);
      for (std::size_t key = 0U; key < 128U; ++key) {
        const std::size_t source =
            (compact_head * 64U + token) * 128U + key;
        const std::size_t destination =
            (head * 64U + token) * 128U + key;
        expected_k_decay[destination] = encode_bf16(
            decode_bf16(candidate.compact_k[source]) * scale);
      }
    }
  }
  const DifferenceMetrics k_decay =
      compare_bf16(expected_k_decay, observed_k_decay);
  const DifferenceMetrics post_update_oracle = compare_bf16(
      baseline_h1, candidate.diagnostic_post_update_state);
  const DifferenceMetrics post_update_publication = compare_bf16(
      candidate_boundary_h1,
      candidate.diagnostic_post_update_state);
  print_difference_metrics(kSuite, "chunk0_k_decay", k_decay);
  print_difference_metrics(kSuite, "chunk0_post_update_vs_oracle",
                           post_update_oracle);
  print_difference_metrics(kSuite, "chunk0_post_update_vs_publication",
                           post_update_publication);
  for (std::size_t chunk = 0U; chunk < kChunkCount; ++chunk) {
    const std::size_t boundary_begin =
        chunk * runtime::kGdnStateElements;
    const std::size_t v_new_begin = chunk * kChunkVNewElements;
    const DifferenceMetrics chunk_boundary = compare_bf16_range(
        baseline.boundary_state, candidate.boundary_state,
        boundary_begin, boundary_begin + runtime::kGdnStateElements);
    const DifferenceMetrics chunk_v_new = compare_bf16_range(
        baseline.v_new, candidate.v_new, v_new_begin,
        v_new_begin + kChunkVNewElements);
    std::cout << kSuite
              << " chunk=" << chunk
              << " boundary_unequal_bf16="
              << chunk_boundary.unequal_bf16
              << " boundary_nrmse=" << chunk_boundary.nrmse
              << " v_new_unequal_bf16=" << chunk_v_new.unequal_bf16
              << " v_new_nrmse=" << chunk_v_new.nrmse << '\n';
  }

  std::string_view first_stage = "none";
  std::size_t first_chunk = kChunkCount;
  std::size_t first_index = std::numeric_limits<std::size_t>::max();
  std::size_t first_local = std::numeric_limits<std::size_t>::max();
  for (std::size_t chunk = 0U; chunk < kChunkCount; ++chunk) {
    const std::size_t boundary_begin =
        chunk * runtime::kGdnStateElements;
    const std::size_t boundary_end =
        boundary_begin + runtime::kGdnStateElements;
    first_index = first_unequal_bf16(
        baseline.boundary_state, candidate.boundary_state,
        boundary_begin, boundary_end);
    if (first_index != std::numeric_limits<std::size_t>::max()) {
      first_stage = "boundary_state";
      first_chunk = chunk;
      first_local = first_index - boundary_begin;
      break;
    }
    const std::size_t v_new_begin = chunk * kChunkVNewElements;
    const std::size_t v_new_end = v_new_begin + kChunkVNewElements;
    first_index = first_unequal_bf16(
        baseline.v_new, candidate.v_new, v_new_begin, v_new_end);
    if (first_index != std::numeric_limits<std::size_t>::max()) {
      first_stage = "v_new";
      first_chunk = chunk;
      first_local = first_index - v_new_begin;
      break;
    }
  }
  if (first_stage == "none") {
    first_index = first_unequal_bf16(
        baseline.state, candidate.state, 0U, baseline.state.size());
    if (first_index != std::numeric_limits<std::size_t>::max()) {
      first_stage = "final_state";
      first_local = first_index;
    }
  }

  std::cout << kSuite
            << " first_stage=" << first_stage;
  if (first_index != std::numeric_limits<std::size_t>::max()) {
    constexpr std::size_t kStateMatrixElements = 128U * 128U;
    std::size_t head = 0U;
    std::size_t token_or_value = 0U;
    std::size_t value_or_key = 0U;
    if (first_stage == "v_new") {
      head = first_local / (64U * 128U);
      const std::size_t remainder = first_local % (64U * 128U);
      token_or_value = remainder / 128U;
      value_or_key = remainder % 128U;
      std::cout << " chunk=" << first_chunk
                << " head=" << head
                << " token_in_chunk=" << token_or_value
                << " value=" << value_or_key;
    } else {
      head = first_local / kStateMatrixElements;
      const std::size_t remainder = first_local % kStateMatrixElements;
      token_or_value = remainder / 128U;
      value_or_key = remainder % 128U;
      if (first_stage == "boundary_state") {
        std::cout << " chunk=" << first_chunk;
      }
      std::cout << " head=" << head
                << " value=" << token_or_value
                << " key=" << value_or_key;
    }
    const std::vector<std::uint16_t>* baseline_region = &baseline.state;
    const std::vector<std::uint16_t>* candidate_region = &candidate.state;
    if (first_stage == "v_new") {
      baseline_region = &baseline.v_new;
      candidate_region = &candidate.v_new;
    } else if (first_stage == "boundary_state") {
      baseline_region = &baseline.boundary_state;
      candidate_region = &candidate.boundary_state;
    }
    std::cout << " baseline_bf16=" << (*baseline_region)[first_index]
              << " candidate_bf16=" << (*candidate_region)[first_index]
              << " baseline_value="
              << decode_bf16((*baseline_region)[first_index])
              << " candidate_value="
              << decode_bf16((*candidate_region)[first_index]);
  }
  std::cout << " baseline_token=" << baseline_sample.generated_token
            << " candidate_token=" << candidate_sample.generated_token
            << " baseline_sample_shape="
            << (baseline_completed ? "PASS" : "MISMATCH")
            << " candidate_sample_shape="
            << (candidate_completed ? "PASS" : "MISMATCH")
            << " snapshot_contract="
            << (snapshot_contract ? "PASS" : "FAIL")
            << " authority=REAL_WEIGHT_P513_FIRST_GDN_CALL\n";
  return ready;
}

[[nodiscard]] bool run_vllm_faithful_state_equivalence(
    runtime::ReferenceEngine& engine, const std::string& prompt) {
  if (g_prompt_tokens != kDefaultPromptTokens) {
    std::cerr << "vLLM-faithful equivalence requires the real P513 route\n";
    return false;
  }
  constexpr std::size_t kNativeTokens = kDefaultPromptTokens - 1U;
  constexpr std::size_t kOutputElements =
      kNativeTokens * runtime::kGdnVElements;
  const std::size_t request_state_elements =
      static_cast<std::size_t>(runtime::kRequestGdnStateBytes /
                               sizeof(std::uint16_t));

  StateSnapshot baseline_state;
  StateSnapshot candidate_state;
  StateSnapshot baseline_return_state;
  StateSnapshot candidate_return_state;
  NativeBoundarySnapshot baseline_final;
  NativeBoundarySnapshot candidate_final;
  try {
    baseline_state.values.resize(request_state_elements);
    candidate_state.values.resize(request_state_elements);
    baseline_return_state.values.resize(request_state_elements);
    candidate_return_state.values.resize(request_state_elements);
    baseline_final.state.resize(runtime::kGdnStateElements);
    candidate_final.state.resize(runtime::kGdnStateElements);
    baseline_final.output.resize(kOutputElements);
    candidate_final.output.resize(kOutputElements);
  } catch (const std::bad_alloc&) {
    std::cerr << "vLLM-faithful equivalence host allocation failed\n";
    return false;
  }

  std::size_t baseline_hits = 0U;
  runtime::ReferenceGenerateResult baseline_result;
  {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(false);
    const ScopedResidentStateBaseline resident_route(false);
    const ScopedPacklessResidentStateFallback packless_fallback(true);
    const ScopedNativeInspectionHook hook(baseline_final);
    const ScopedGenerateReturnSnapshotHook return_hook(
        baseline_return_state);
    baseline_result = run_snapshot_generation(
        engine, prompt, true, baseline_state, baseline_hits);
  }

  std::size_t candidate_hits = 0U;
  runtime::ReferenceGenerateResult candidate_result;
  {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(false);
    const ScopedResidentStateBaseline resident_route(false);
    const ScopedPacklessResidentStateFallback packless_fallback(false);
    const ScopedNativeInspectionHook hook(candidate_final);
    const ScopedGenerateReturnSnapshotHook return_hook(
        candidate_return_state);
    candidate_result = run_snapshot_generation(
        engine, prompt, true, candidate_state, candidate_hits);
  }
  if (!baseline_result || !candidate_result) {
    std::cerr << "vLLM-faithful equivalence generation failed\n";
    if (!baseline_result) {
      print_diagnostic(baseline_result.diagnostic);
    }
    if (!candidate_result) {
      print_diagnostic(candidate_result.diagnostic);
    }
    return false;
  }

  const DifferenceMetrics final_layer_state = compare_bf16(
      baseline_final.state, candidate_final.state);
  const DifferenceMetrics final_layer_output = compare_bf16(
      baseline_final.output, candidate_final.output);
  const DifferenceMetrics post_c512_request_state = compare_bf16(
      baseline_state.values, candidate_state.values);
  const DifferenceMetrics post_c1_request_state = compare_bf16(
      baseline_return_state.values, candidate_return_state.values);
  constexpr std::string_view kSuite =
      "GDN_CHUNK64_VLLM_FAITHFUL_EQUIVALENCE";
  print_difference_metrics(kSuite, "final_layer_state", final_layer_state);
  print_difference_metrics(kSuite, "final_layer_output", final_layer_output);
  print_difference_metrics(kSuite, "post_c512_request_state",
                           post_c512_request_state);
  print_difference_metrics(kSuite, "post_c1_request_state",
                           post_c1_request_state);

  const bool boundary_contract =
      baseline_final.captured && candidate_final.captured &&
      !baseline_final.contract_error && !candidate_final.contract_error &&
      baseline_final.cuda_error == static_cast<int>(cudaSuccess) &&
      candidate_final.cuda_error == static_cast<int>(cudaSuccess) &&
      baseline_final.calls == runtime::kRequestLinearLayerCount &&
      candidate_final.calls == runtime::kRequestLinearLayerCount;
  const bool generation_semantics =
      expected_generation(*baseline_result.value) &&
      expected_generation(*candidate_result.value) &&
      baseline_result.value->generated_token_ids ==
          candidate_result.value->generated_token_ids &&
      baseline_result.value->generated_text ==
          candidate_result.value->generated_text;
  const bool return_state_contract =
      valid_snapshot_contract(baseline_return_state) &&
      valid_snapshot_contract(candidate_return_state) &&
      baseline_return_state.committed_position == g_prompt_tokens &&
      candidate_return_state.committed_position == g_prompt_tokens;
  const bool passed =
      valid_snapshot(baseline_state) && valid_snapshot(candidate_state) &&
      baseline_hits == expected_native_route_hits() &&
      candidate_hits == expected_native_route_hits() && boundary_contract &&
      return_state_contract && generation_semantics &&
      exact_metrics(final_layer_state) && exact_metrics(final_layer_output) &&
      exact_metrics(post_c512_request_state) &&
      exact_metrics(post_c1_request_state);
  std::cout << kSuite
            << " baseline_hits=" << baseline_hits
            << " candidate_hits=" << candidate_hits
            << " boundary_contract="
            << (boundary_contract ? "PASS" : "FAIL")
            << " return_state_contract="
            << (return_state_contract ? "PASS" : "FAIL")
            << " baseline_return_position="
            << baseline_return_state.committed_position
            << " candidate_return_position="
            << candidate_return_state.committed_position
            << " generation_semantics="
            << (generation_semantics ? "PASS" : "FAIL")
            << " generated_token="
            << candidate_result.value->generated_token_ids.front()
            << " generated_text=" << candidate_result.value->generated_text
            << " gate=" << (passed ? "PASS" : "FAIL")
            << " authority=REAL_WEIGHT_P513_POST_C512_AND_POST_C1_STATE\n";
  return passed;
}

[[nodiscard]] bool run_vllm_faithful_default_smoke(
    runtime::ReferenceEngine& engine, const std::string& prompt) {
  Sample sample;
  {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(false);
    const ScopedResidentStateBaseline resident_route(false);
    // Deliberately install no state-recurrence selector: this is the native
    // C64 production default exactly as a caller reaches it.
    (void)run_sample(engine, prompt, true, "faithful_default_smoke", sample);
  }
  const bool passed =
      sample.route_hits == expected_native_route_hits() &&
      sample.semantic_oracle &&
      sample.generated_token == kExpectedGeneratedToken &&
      sample.generated_text == kExpectedGeneratedText &&
      std::isfinite(sample.prefix_milliseconds) &&
      sample.prefix_milliseconds > 0.0 &&
      std::isfinite(sample.ttft_milliseconds) &&
      sample.ttft_milliseconds > 0.0;
  std::cout << "GDN_CHUNK64_VLLM_FAITHFUL_DEFAULT_SMOKE"
            << " prompt_tokens=" << g_prompt_tokens
            << " prefix_ms=" << sample.prefix_milliseconds
            << " ttft_ms=" << sample.ttft_milliseconds
            << " route_hits=" << sample.route_hits
            << " generated_token=" << sample.generated_token
            << " generated_text=" << sample.generated_text
            << " gate=" << (passed ? "PASS" : "FAIL")
            << " authority=REAL_WEIGHT_PRODUCTION_DEFAULT_NO_SELECTOR\n";
  return passed;
}

// One clean real-weight B->C direction read. Both samples stay on the full
// native production route; only the state-recurrence implementation changes.
// No inspection hook is installed, so C launches the production <false>
// candidate template and performs no diagnostic writes.
[[nodiscard]] bool run_vllm_faithful_state_direction(
    runtime::ReferenceEngine& engine, const std::string& prompt) {
  Sample baseline;
  Sample candidate;
  {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(false);
    const ScopedResidentStateBaseline resident_route(false);
    const ScopedPacklessResidentStateFallback packless_fallback(true);
    (void)run_sample(engine, prompt, true, "faithful_clean_baseline",
                     baseline);
  }
  {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(false);
    const ScopedResidentStateBaseline resident_route(false);
    const ScopedPacklessResidentStateFallback packless_fallback(false);
    (void)run_sample(engine, prompt, true, "faithful_clean_candidate",
                     candidate);
  }
  const std::size_t expected_hits = expected_native_route_hits();
  const bool valid =
      baseline.route_hits == expected_hits &&
      candidate.route_hits == expected_hits &&
      std::isfinite(baseline.prefix_milliseconds) &&
      baseline.prefix_milliseconds > 0.0 &&
      std::isfinite(candidate.prefix_milliseconds) &&
      candidate.prefix_milliseconds > 0.0 &&
      std::isfinite(baseline.ttft_milliseconds) &&
      baseline.ttft_milliseconds > 0.0 &&
      std::isfinite(candidate.ttft_milliseconds) &&
      candidate.ttft_milliseconds > 0.0;
  const bool semantics =
      baseline.semantic_oracle && candidate.semantic_oracle &&
      baseline.generated_token == candidate.generated_token &&
      baseline.generated_text == candidate.generated_text;
  const double prefix_saved =
      baseline.prefix_milliseconds - candidate.prefix_milliseconds;
  const double ttft_saved =
      baseline.ttft_milliseconds - candidate.ttft_milliseconds;
  const bool positive = valid && semantics && prefix_saved > 0.0 &&
                        ttft_saved > 0.0;
  std::cout << "GDN_CHUNK64_VLLM_FAITHFUL_CLEAN_DIRECTION"
            << " prompt_tokens=" << g_prompt_tokens
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
            << " baseline_hits=" << baseline.route_hits
            << " candidate_hits=" << candidate.route_hits
            << " generated_token=" << candidate.generated_token
            << " generated_text=" << candidate.generated_text
            << " direction=" << (positive ? "POSITIVE" : "NEGATIVE")
            << " semantic_oracle=" << (semantics ? "PASS" : "FAIL")
            << " authority=REAL_WEIGHT_SINGLE_CLEAN_B_C"
            << " production_default=faithful\n";
  return positive;
}

[[nodiscard]] bool run_vllm_faithful_state_bccb(
    runtime::ReferenceEngine& engine, const std::string& prompt) {
  auto run = [&](const bool faithful, const std::string_view phase,
                 Sample& sample) {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedSplitWyBaseline split_route(false);
    const ScopedPackedQkvBaseline packed_route(false);
    const ScopedResidentStateBaseline resident_route(false);
    const ScopedPacklessResidentStateFallback packless_fallback(!faithful);
    (void)run_sample(engine, prompt, true, phase, sample);
  };

  Sample warm_baseline;
  Sample warm_candidate;
  Sample baseline_first;
  Sample candidate_first;
  Sample candidate_second;
  Sample baseline_second;
  run(false, "faithful_bccb_warm_baseline", warm_baseline);
  run(true, "faithful_bccb_warm_candidate", warm_candidate);
  run(false, "faithful_bccb_b1", baseline_first);
  run(true, "faithful_bccb_c1", candidate_first);
  run(true, "faithful_bccb_c2", candidate_second);
  run(false, "faithful_bccb_b2", baseline_second);

  const std::size_t expected_hits = expected_native_route_hits();
  const Sample* const samples[] = {
      &warm_baseline, &warm_candidate, &baseline_first,
      &candidate_first, &candidate_second, &baseline_second};
  bool valid = true;
  for (const Sample* const sample : samples) {
    valid = valid && sample->route_hits == expected_hits &&
            sample->semantic_oracle &&
            sample->generated_token == kExpectedGeneratedToken &&
            sample->generated_text == kExpectedGeneratedText &&
            std::isfinite(sample->prefix_milliseconds) &&
            sample->prefix_milliseconds > 0.0 &&
            std::isfinite(sample->ttft_milliseconds) &&
            sample->ttft_milliseconds > 0.0;
  }
  const double pair1_prefix_saved =
      baseline_first.prefix_milliseconds -
      candidate_first.prefix_milliseconds;
  const double pair1_ttft_saved =
      baseline_first.ttft_milliseconds - candidate_first.ttft_milliseconds;
  const double pair2_prefix_saved =
      baseline_second.prefix_milliseconds -
      candidate_second.prefix_milliseconds;
  const double pair2_ttft_saved =
      baseline_second.ttft_milliseconds -
      candidate_second.ttft_milliseconds;
  const bool positive = valid && pair1_prefix_saved > 0.0 &&
                        pair1_ttft_saved > 0.0 &&
                        pair2_prefix_saved > 0.0 &&
                        pair2_ttft_saved > 0.0;
  std::cout << "GDN_CHUNK64_VLLM_FAITHFUL_BCCB"
            << " prompt_tokens=" << g_prompt_tokens
            << " pair1_baseline_prefix_ms="
            << baseline_first.prefix_milliseconds
            << " pair1_candidate_prefix_ms="
            << candidate_first.prefix_milliseconds
            << " pair1_prefix_saved_ms=" << pair1_prefix_saved
            << " pair1_baseline_ttft_ms="
            << baseline_first.ttft_milliseconds
            << " pair1_candidate_ttft_ms="
            << candidate_first.ttft_milliseconds
            << " pair1_ttft_saved_ms=" << pair1_ttft_saved
            << " pair2_baseline_prefix_ms="
            << baseline_second.prefix_milliseconds
            << " pair2_candidate_prefix_ms="
            << candidate_second.prefix_milliseconds
            << " pair2_prefix_saved_ms=" << pair2_prefix_saved
            << " pair2_baseline_ttft_ms="
            << baseline_second.ttft_milliseconds
            << " pair2_candidate_ttft_ms="
            << candidate_second.ttft_milliseconds
            << " pair2_ttft_saved_ms=" << pair2_ttft_saved
            << " route_hits_each=" << expected_hits
            << " generated_token=" << candidate_second.generated_token
            << " generated_text=" << candidate_second.generated_text
            << " gate=" << (positive ? "PASS" : "FAIL")
            << " authority=REAL_WEIGHT_WARM_SAME_PROCESS_B_C_C_B"
            << " production_default=faithful\n";
  return positive;
}

[[nodiscard]] bool run_native_resident_state_direction(
    runtime::ReferenceEngine& engine,
    const std::string& prompt) {
  Sample baseline;
  Sample candidate;
  bool valid = false;
  {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedResidentStateBaseline route(true);
    valid = run_sample(engine, prompt, true, "resident_baseline", baseline);
  }
  {
    const ScopedFusedKktBaseline kkt_route(false);
    const ScopedResidentStateBaseline route(false);
    valid = run_sample(engine, prompt, true, "resident_candidate", candidate) &&
            valid;
  }
  const bool semantics =
      baseline.semantic_oracle && candidate.semantic_oracle &&
      baseline.generated_token == candidate.generated_token &&
      baseline.generated_text == candidate.generated_text;
  const double prefix_saved =
      baseline.prefix_milliseconds - candidate.prefix_milliseconds;
  const double ttft_saved =
      baseline.ttft_milliseconds - candidate.ttft_milliseconds;
  const bool positive = valid && semantics && prefix_saved > 0.0 &&
                        ttft_saved > 0.0;
  std::cout << "GDN_CHUNK64_NATIVE_RESIDENT_STATE_DIRECTION"
            << " prompt_tokens=" << g_prompt_tokens
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
            << " semantic_oracle=" << (semantics ? "PASS" : "FAIL")
            << " authority=REAL_WEIGHT_SINGLE_B_C\n";
  return positive;
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::size_t bytes) noexcept : bytes_(bytes) {
    status_ = cudaMalloc(&data_, bytes_);
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] bool valid() const noexcept {
    return status_ == cudaSuccess && data_ != nullptr;
  }

  [[nodiscard]] bool zero() noexcept {
    return valid() && cudaMemset(data_, 0, bytes_) == cudaSuccess;
  }

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }

  template <typename T>
  [[nodiscard]] T* as() noexcept {
    return static_cast<T*>(data_);
  }

  template <typename T>
  [[nodiscard]] const T* as() const noexcept {
    return static_cast<const T*>(data_);
  }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0U;
  cudaError_t status_ = cudaErrorMemoryAllocation;
};

[[nodiscard]] bool run_native_graph_validation() {
  constexpr std::size_t kTokens = 512U;
  constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);
  DeviceBuffer workspace(
      q3x::runtime::gdn_prefill_chunk64_native_detail::workspace_bytes());
  DeviceBuffer conv_qkv(
      kTokens * runtime::kGdnQkvChannels * kBf16Bytes);
  DeviceBuffer a(kTokens * runtime::kGdnValueHeadCount * kBf16Bytes);
  DeviceBuffer b(kTokens * runtime::kGdnValueHeadCount * kBf16Bytes);
  DeviceBuffer A_log(runtime::kGdnValueHeadCount * kBf16Bytes);
  DeviceBuffer dt_bias(runtime::kGdnValueHeadCount * kBf16Bytes);
  DeviceBuffer state(runtime::kGdnStateElements * kBf16Bytes);
  DeviceBuffer norm(runtime::kGdnHeadDimension * kBf16Bytes);
  DeviceBuffer silu_gate(kTokens * runtime::kGdnVElements * kBf16Bytes);
  DeviceBuffer output(kTokens * runtime::kGdnVElements * kBf16Bytes);
  DeviceBuffer* const buffers[] = {
      &workspace, &conv_qkv, &a,     &b,      &A_log,
      &dt_bias,  &state,    &norm,  &silu_gate, &output};
  bool ready = true;
  for (DeviceBuffer* const buffer : buffers) {
    ready = buffer->zero() && ready;
  }
  if (!ready) {
    std::cerr << "native Graph buffer allocation/initialization failed\n";
    return false;
  }

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  ready = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
          cudaSuccess;
  const ScopedFusedKktBaseline kkt_route(false);
  const ScopedSplitWyBaseline split_route(false);
  const ScopedPackedQkvBaseline packed_route(false);
  const ScopedResidentStateBaseline resident_route(false);
  const ScopedPacklessResidentStateFallback packless_fallback(false);
  ready = kkt_route.valid() && split_route.valid() &&
          resident_route.valid() && ready;
  auto launch = [&]() {
    return q3x::runtime::gdn_prefill_chunk64_native_detail::launch(
        workspace.data(),
        q3x::runtime::gdn_prefill_chunk64_native_detail::workspace_bytes(),
        conv_qkv.as<const std::uint16_t>(), kTokens,
        a.as<const std::uint16_t>(), b.as<const std::uint16_t>(),
        A_log.as<const std::uint16_t>(), dt_bias.as<const std::uint16_t>(),
        state.as<const std::uint16_t>(), state.as<std::uint16_t>(), 1.0e-6F,
        norm.as<const std::uint16_t>(),
        silu_gate.as<const std::uint16_t>(), 1.0e-6F,
        output.as<std::uint16_t>(), stream);
  };
  if (ready) {
    ready = launch() == static_cast<int>(cudaSuccess) &&
            cudaStreamSynchronize(stream) == cudaSuccess;
  }
  if (ready) {
    ready = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) ==
            cudaSuccess;
  }
  if (ready) {
    ready = launch() == static_cast<int>(cudaSuccess);
  }
  if (ready) {
    ready = cudaStreamEndCapture(stream, &graph) == cudaSuccess &&
            graph != nullptr;
  } else if (stream != nullptr) {
    (void)cudaStreamEndCapture(stream, &graph);
  }

  std::size_t node_count = 0U;
  std::size_t kernel_nodes = 0U;
  std::size_t other_nodes = 0U;
  if (ready) {
    ready = cudaGraphGetNodes(graph, nullptr, &node_count) == cudaSuccess;
  }
  std::vector<cudaGraphNode_t> nodes(node_count);
  if (ready && node_count != 0U) {
    std::size_t capacity = node_count;
    ready = cudaGraphGetNodes(graph, nodes.data(), &capacity) == cudaSuccess &&
            capacity == node_count;
  }
  if (ready) {
    for (const cudaGraphNode_t node : nodes) {
      cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
      if (cudaGraphNodeGetType(node, &type) != cudaSuccess) {
        ready = false;
        break;
      }
      if (type == cudaGraphNodeTypeKernel) {
        ++kernel_nodes;
      } else {
        ++other_nodes;
      }
    }
  }
  if (ready) {
    ready = cudaGraphInstantiate(&executable, graph, 0U) == cudaSuccess;
  }
  if (ready) {
    ready = cudaGraphLaunch(executable, stream) == cudaSuccess &&
            cudaGraphLaunch(executable, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess;
  }
  const bool passed =
      ready && node_count == 6U && kernel_nodes == 6U && other_nodes == 0U;
  std::cout << "GDN_CHUNK64_NATIVE_GRAPH"
            << " nodes=" << node_count
            << " kernel_nodes=" << kernel_nodes
            << " other_nodes=" << other_nodes
            << " replays=2"
            << " gate=" << (passed ? "PASS" : "FAIL")
            << " authority=CAPTURE_INSTANTIATE_REPLAY_SMOKE\n";

  if (executable != nullptr) {
    (void)cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }
  if (stream != nullptr) {
    (void)cudaStreamDestroy(stream);
  }
  return passed;
}
#endif

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
                      candidate_hits == expected_native_route_hits() &&
                      finite && semantics && std::isfinite(nrmse);
  std::cout << kRouteMarker << "_STATE_CHARACTERIZATION"
            << " prompt_tokens=" << g_prompt_tokens
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
#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
  if (std::getenv("Q3X_GDN_CHUNK64_VALIDATE_GRAPH_ONLY") != nullptr) {
    return run_native_graph_validation() ? 0 : 8;
  }
#endif
  runtime::ReferenceEngineOptions options;
  const char* const prompt_tokens_environment =
      std::getenv("Q3X_GDN_CHUNK64_PROMPT_TOKENS");
  if (prompt_tokens_environment != nullptr) {
    char* end = nullptr;
    const unsigned long long parsed =
        std::strtoull(prompt_tokens_environment, &end, 10);
    if (end == prompt_tokens_environment || *end != '\0' || parsed < 13U ||
        parsed > 4096U) {
      std::cerr << "invalid Q3X_GDN_CHUNK64_PROMPT_TOKENS\n";
      return 2;
    }
    g_prompt_tokens = static_cast<std::size_t>(parsed);
  }
  options.request_options.prefill_chunk_size = kPrefillChunkTokens;
  options.request_options.max_sequence_length = g_prompt_tokens + 1U;
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
#if defined(Q3X_GDN_CHUNK64_NATIVE_TEST)
  if (std::getenv(
          "Q3X_GDN_CHUNK64_RUN_GROUP_WY_EQUIVALENCE_ONLY") != nullptr) {
    return run_native_kkt_equivalence(*created.value, prompt) ? 0 : 7;
  }
  if (std::getenv(
          "Q3X_GDN_CHUNK64_RUN_GROUP_WY_EVENT_ATTRIBUTION_ONLY") !=
      nullptr) {
    return run_native_group_wy_event_attribution(*created.value, prompt)
               ? 0
               : 10;
  }
  if (std::getenv(
          "Q3X_GDN_CHUNK64_RUN_VLLM_FAITHFUL_CALL1_SNAPSHOT_ONLY") !=
      nullptr) {
    return run_vllm_faithful_state_call1_snapshot(*created.value, prompt)
               ? 0
               : 11;
  }
  if (std::getenv(
          "Q3X_GDN_CHUNK64_RUN_VLLM_FAITHFUL_EQUIVALENCE_ONLY") != nullptr) {
    return run_vllm_faithful_state_equivalence(*created.value, prompt)
               ? 0
               : 9;
  }
  if (std::getenv(
          "Q3X_GDN_CHUNK64_RUN_VLLM_FAITHFUL_DEFAULT_SMOKE_ONLY") !=
      nullptr) {
    return run_vllm_faithful_default_smoke(*created.value, prompt) ? 0 : 3;
  }
  if (std::getenv(
          "Q3X_GDN_CHUNK64_RUN_VLLM_FAITHFUL_DIRECTION_ONLY") != nullptr) {
    return run_vllm_faithful_state_direction(*created.value, prompt)
               ? 0
               : 3;
  }
  if (std::getenv("Q3X_GDN_CHUNK64_RUN_VLLM_FAITHFUL_BCCB_ONLY") !=
      nullptr) {
    return run_vllm_faithful_state_bccb(*created.value, prompt) ? 0 : 3;
  }
  // A structural recurrence rewrite must prove its exact real-model
  // boundaries before its first direction timing is allowed to matter.
  if (std::getenv(
          "Q3X_GDN_CHUNK64_RUN_RESIDENT_STATE_EQUIVALENCE") != nullptr &&
      !run_native_resident_state_equivalence(*created.value, prompt)) {
    return 9;
  }
  if (std::getenv(
          "Q3X_GDN_CHUNK64_RUN_RESIDENT_STATE_DIRECTION_ONLY") != nullptr) {
    return run_native_resident_state_direction(*created.value, prompt)
               ? 0
               : 3;
  }
#endif
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
      baseline.semantic_oracle && candidate.semantic_oracle &&
      baseline.generated_token == candidate.generated_token &&
      baseline.generated_text == candidate.generated_text;
  std::cout << kRouteMarker << "_DIRECTION"
            << " prompt_tokens=" << g_prompt_tokens
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
  if (std::getenv("Q3X_GDN_CHUNK64_RUN_KKT_EQUIVALENCE") != nullptr &&
      !run_native_kkt_equivalence(*created.value, prompt)) {
    return 7;
  }
  if (std::getenv("Q3X_GDN_CHUNK64_VALIDATE_GRAPH") != nullptr &&
      !run_native_graph_validation()) {
    return 8;
  }
  if (std::getenv("Q3X_GDN_CHUNK64_SKIP_STATE_CHARACTERIZATION") == nullptr &&
      !run_state_characterization(*created.value, prompt)) {
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
