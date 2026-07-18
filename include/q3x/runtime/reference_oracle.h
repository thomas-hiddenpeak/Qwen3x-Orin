#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime {

inline constexpr std::size_t kReferenceOracleLayerCount = 64U;
inline constexpr std::size_t kReferenceOraclePhaseCount = 2U;
inline constexpr std::size_t kReferenceOracleSampleCount = 16U;
inline constexpr std::size_t kReferenceOracleHiddenSize = 5120U;
inline constexpr std::size_t kReferenceOracleVocabularySize = 248320U;

struct ReferenceOracleLimits {
  std::size_t max_file_bytes = 4U * 1024U * 1024U;
  std::size_t max_json_nodes = 50'000U;
  std::size_t max_nesting_depth = 32U;
  std::size_t max_container_items = 50'000U;
  std::size_t max_string_bytes = 1U * 1024U * 1024U;
  std::size_t max_array_length = 512U;
};

enum class ReferenceOracleErrorCode : std::uint8_t {
  kNone,
  kInvalidArgument,
  kInvalidLimit,
  kOpenFailed,
  kNotRegularFile,
  kFileTooLarge,
  kIoFailure,
  kJsonRejected,
  kResourceLimit,
  kSchemaMismatch,
  kCrossFileMismatch,
  kNonFiniteValue,
  kInvalidSampleIndex,
  kArithmeticOverflow,
  kAllocationFailure,
};

struct ReferenceOracleDiagnostic {
  ReferenceOracleErrorCode code = ReferenceOracleErrorCode::kNone;
  std::string path;
  std::string message;
  std::string expected;
  std::string actual;
  int system_error = 0;
  std::size_t json_offset = 0U;
};

struct CachePolicyReference {
  std::string kv_cache_dtype;
  std::string mamba_cache_dtype;
  std::string mamba_ssm_cache_dtype;
};

struct GreedyReference {
  std::string fixture;
  std::string source_repository;
  std::string source_revision;
  CachePolicyReference cache_policy;
  std::vector<std::uint32_t> prompt_token_ids;
  std::vector<std::uint32_t> expected_token_ids;
  std::string expected_text;
  std::string finish_reason;
  std::uint32_t stop_token_id = 0U;
  std::vector<double> chosen_logprobs;
};

struct BoundarySummary {
  std::string dtype;
  std::size_t length = 0U;
  std::string sha256_raw;
  double mean = 0.0;
  double rms = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
  std::vector<double> samples;
};

struct LayerBoundaryReference {
  std::size_t index = 0U;
  BoundarySummary hidden;
  BoundarySummary residual;
};

struct TopLogitReference {
  std::uint32_t token_id = 0U;
  double logit = 0.0;
};

struct LogitsReference {
  std::string dtype;
  std::size_t length = 0U;
  double logsumexp = 0.0;
  std::array<TopLogitReference, 20U> top20{};
  double chosen_logprob = 0.0;
};

struct PhaseReference {
  std::string name;
  std::size_t position = 0U;
  std::uint32_t input_token_id = 0U;
  std::uint32_t predicted_token_id = 0U;
  BoundarySummary embedding;
  std::array<LayerBoundaryReference, kReferenceOracleLayerCount> layers{};
  BoundarySummary final_norm;
  LogitsReference logits;
};

struct LayerReference {
  std::string fixture;
  std::string source_greedy_fixture;
  std::string source_revision;
  CachePolicyReference cache_policy;
  std::array<std::size_t, kReferenceOracleSampleCount> sample_indices{};
  std::array<PhaseReference, kReferenceOraclePhaseCount> phases{};
};

struct ReferenceOracle {
  GreedyReference greedy;
  LayerReference layers;
};

struct ReferenceOracleResult {
  std::optional<ReferenceOracle> value;
  ReferenceOracleDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() &&
           diagnostic.code == ReferenceOracleErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Loads and cross-validates the pinned schema-1 BF16-cache oracle pair.
// Files must be regular non-symlink files. Both JSON DOM construction and the
// post-parse string/array walk are bounded by ReferenceOracleLimits.
[[nodiscard]] ReferenceOracleResult load_reference_oracle(
    const std::filesystem::path& greedy_path,
    const std::filesystem::path& layers_path,
    const ReferenceOracleLimits& limits = {});

struct BoundarySummaryResult {
  std::optional<BoundarySummary> value;
  ReferenceOracleDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() &&
           diagnostic.code == ReferenceOracleErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Produces a canonical little-endian SHA-256 over raw BF16 words, FP64
// diagnostic statistics, and samples at the caller-provided indices.
[[nodiscard]] BoundarySummaryResult summarize_bf16_span(
    const std::uint16_t* values, std::size_t length,
    const std::vector<std::size_t>& sample_indices);

struct BoundarySampleMismatch {
  std::size_t sample_ordinal = 0U;
  std::size_t element_index = 0U;
  double expected = 0.0;
  double actual = 0.0;
  double tolerance = 0.0;
};

struct BoundaryComparisonResult {
  bool matches = false;
  std::optional<BoundarySampleMismatch> first_mismatch;
  ReferenceOracleDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return diagnostic.code == ReferenceOracleErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Compares only the canonical diagnostic sample positions. A valid numerical
// mismatch is not an API error: ok()==true, matches==false, and first_mismatch
// identifies the first differing sample.
[[nodiscard]] BoundaryComparisonResult compare_boundary_samples(
    const std::uint16_t* actual, std::size_t actual_length,
    const BoundarySummary& expected,
    const std::vector<std::size_t>& sample_indices, double absolute_tolerance,
    double relative_tolerance);

[[nodiscard]] std::string_view to_string(
    ReferenceOracleErrorCode code) noexcept;

}  // namespace q3x::runtime
