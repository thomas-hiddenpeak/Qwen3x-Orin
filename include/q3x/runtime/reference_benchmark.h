#pragma once

#include "q3x/runtime/reference_engine.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime {

inline constexpr std::size_t kReferenceBenchmarkNoPrompt =
    std::numeric_limits<std::size_t>::max();

enum class ReferenceBenchmarkError : std::uint8_t {
  kNone = 0,
  kInvalidArgument,
  kGenerationFailure,
  kRepeatabilityFailure,
  kDeviceMemoryProbeFailure,
  kInvalidTiming,
  kAllocationFailure,
};

struct ReferenceBenchmarkOptions {
  std::uint32_t warmup_rounds = 1U;
  std::uint32_t measured_rounds = 3U;
  std::uint32_t max_new_tokens = 16U;
  std::uint32_t stop_token_id = kQwen36ImEndTokenId;
  std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
  // A persistent free-memory decrease larger than this is called out in the
  // report. It does not discard otherwise valid latency measurements.
  std::uint64_t device_memory_drop_tolerance_bytes =
      64ULL * 1024ULL * 1024ULL;
  // The CLI explicitly selects prediction-only; direct API callers retain
  // the full-statistics generation default unless they opt in here.
  ReferenceLogitsMode logits_mode = ReferenceLogitsMode::kFullStatistics;
  // Emits stable prefill/decode NVTX ranges for an explicitly profiled run.
  bool emit_nvtx_phase_ranges = false;
};

// count==0 means that no values exist for this metric. Median is the middle
// value (or mean of the two middle values). P95 is nearest-rank:
// sorted[ceil(0.95 * count) - 1].
struct ReferenceLatencyStatistics {
  std::size_t count = 0U;
  double minimum_milliseconds = 0.0;
  double median_milliseconds = 0.0;
  double p95_milliseconds = 0.0;
  double maximum_milliseconds = 0.0;
};

struct ReferenceBenchmarkStep {
  std::uint32_t position = 0U;
  std::uint32_t input_token_id = 0U;
  std::optional<std::uint32_t> predicted_token_id;
};

struct ReferenceBenchmarkSample {
  std::size_t prompt_index = 0U;
  std::uint32_t measured_round = 0U;
  ReferenceGenerationTiming timing;
};

struct ReferenceBenchmarkPromptReport {
  std::string prompt;
  std::vector<std::uint32_t> prompt_token_ids;
  std::vector<std::uint32_t> generated_token_ids;
  std::string generated_text;
  ReferenceStopReason stop_reason = ReferenceStopReason::kMaxNewTokens;
  std::vector<ReferenceBenchmarkStep> step_sequence;
  // Each sample contributes the sum of its prefix-execution invocations.
  ReferenceLatencyStatistics prompt_prefix;
  ReferenceLatencyStatistics finish_prefill;
  ReferenceLatencyStatistics prompt_prefill;
  ReferenceLatencyStatistics decode_after_first;
  ReferenceLatencyStatistics time_to_first_token;
  ReferenceLatencyStatistics total_generation;
  // This distribution flattens every post-first-token latency from every
  // measured invocation of this prompt.
  ReferenceLatencyStatistics subsequent_token;
};

struct ReferenceBenchmarkMemory {
  std::uint64_t start_free_bytes = 0U;
  std::uint64_t end_free_bytes = 0U;
  std::uint64_t minimum_free_bytes = 0U;
  std::uint64_t total_bytes = 0U;
  std::uint64_t persistent_drop_bytes = 0U;
  std::uint64_t maximum_observed_drop_bytes = 0U;
  std::uint64_t drop_tolerance_bytes = 0U;
  bool persistent_drop_detected = false;
};

struct ReferenceBenchmarkReport {
  std::uint32_t warmup_rounds = 0U;
  std::uint32_t measured_rounds = 0U;
  std::uint32_t max_new_tokens = 0U;
  std::uint32_t stop_token_id = kQwen36ImEndTokenId;
  std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
  std::vector<ReferenceBenchmarkPromptReport> prompts;
  std::vector<ReferenceBenchmarkSample> samples;
  // Each sample contributes the sum of its prefix-execution invocations.
  ReferenceLatencyStatistics prompt_prefix;
  ReferenceLatencyStatistics finish_prefill;
  ReferenceLatencyStatistics prompt_prefill;
  ReferenceLatencyStatistics decode_after_first;
  ReferenceLatencyStatistics time_to_first_token;
  ReferenceLatencyStatistics total_generation;
  ReferenceLatencyStatistics subsequent_token;
  ReferenceBenchmarkMemory device_memory;
  ReferenceLogitsMode logits_mode = ReferenceLogitsMode::kFullStatistics;
  bool nvtx_phase_ranges_emitted = false;
};

struct ReferenceBenchmarkDiagnostic {
  ReferenceBenchmarkError code = ReferenceBenchmarkError::kNone;
  std::string message;
  std::size_t prompt_index = kReferenceBenchmarkNoPrompt;
  std::uint32_t round = 0U;
  bool warmup = false;
  std::string mismatch_field;
  ReferenceEngineDiagnostic generation;
  int cuda_error = 0;
};

struct ReferenceBenchmarkResult {
  std::optional<ReferenceBenchmarkReport> value;
  ReferenceBenchmarkDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() &&
           diagnostic.code == ReferenceBenchmarkError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Reuses one already-created engine. ReferenceEngine::generate resets all
// request state before every invocation. Warmup and measured invocations are
// both compared against the first result for their prompt; timing statistics
// contain measured invocations only.
[[nodiscard]] ReferenceBenchmarkResult benchmark_reference_engine(
    ReferenceEngine& engine,
    const std::vector<std::string>& prompts,
    const ReferenceBenchmarkOptions& options = {});

[[nodiscard]] std::string_view to_string(ReferenceBenchmarkError error) noexcept;

namespace reference_benchmark_detail {

struct DeviceMemorySnapshot {
  std::uint64_t free_bytes = 0U;
  std::uint64_t total_bytes = 0U;
};

struct DeviceMemoryProbeResult {
  std::optional<DeviceMemorySnapshot> value;
  int cuda_error = 0;
  std::string message;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && cuda_error == 0;
  }
};

using GenerateFunction = ReferenceGenerateResult (*)(
    void* context, std::string_view prompt,
    const ReferenceGenerateOptions& options);
using DeviceMemoryProbeFunction = DeviceMemoryProbeResult (*)(void* context);

// Host-only orchestration seam used to test round ordering, replay checks,
// statistics, and memory accounting without loading a checkpoint.
[[nodiscard]] ReferenceBenchmarkResult run_benchmark_control(
    const std::vector<std::string>& prompts,
    const ReferenceBenchmarkOptions& options,
    void* generate_context,
    GenerateFunction generate,
    void* memory_context,
    DeviceMemoryProbeFunction probe_memory);

[[nodiscard]] std::optional<ReferenceLatencyStatistics>
compute_latency_statistics(const std::vector<double>& milliseconds);

// Returns an empty string when the required deterministic fields match.
[[nodiscard]] std::string generation_mismatch_field(
    const ReferenceGeneration& expected,
    const ReferenceGeneration& actual);

}  // namespace reference_benchmark_detail

}  // namespace q3x::runtime
