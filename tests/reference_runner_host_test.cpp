#include "q3x/runtime/reference_runner.h"

#include "q3x/runtime/decode_ops.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

namespace runtime = q3x::runtime;
namespace detail = q3x::runtime::reference_runner_detail;

class TestContext {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

[[nodiscard]] bool close(const double left, const double right,
                         const double tolerance = 1.0e-12) {
  return std::abs(left - right) <= tolerance;
}

[[nodiscard]] bool same_bits(const float left, const float right) noexcept {
  return std::memcmp(&left, &right, sizeof(left)) == 0;
}

[[nodiscard]] bool same_bits(const double left, const double right) noexcept {
  return std::memcmp(&left, &right, sizeof(left)) == 0;
}

[[nodiscard]] bool same_analysis_bits(
    const detail::LogitsAnalysis& left,
    const detail::LogitsAnalysis& right) noexcept {
  return left.status == right.status &&
         left.predicted_index == right.predicted_index &&
         same_bits(left.maximum, right.maximum) &&
         same_bits(left.logsumexp, right.logsumexp) &&
         same_bits(left.max_log_probability, right.max_log_probability);
}

[[nodiscard]] detail::LogitsAnalysis scalar_bf16_logits_bits_analysis(
    const std::uint16_t* const logits,
    const std::size_t element_count) noexcept {
  detail::LogitsAnalysis result;
  if (logits == nullptr || element_count == 0U) {
    result.status = detail::LogitsAnalysisStatus::kInvalidArgument;
    return result;
  }

  std::size_t maximum_index = 0U;
  float maximum = detail::bf16_to_float(logits[0]);
  bool all_finite = std::isfinite(maximum);
  for (std::size_t index = 1U; index < element_count; ++index) {
    const float value = detail::bf16_to_float(logits[index]);
    all_finite = all_finite && std::isfinite(value);
    if (value > maximum) {
      maximum = value;
      maximum_index = index;
    }
  }
  if (!all_finite) {
    result.status = detail::LogitsAnalysisStatus::kNonFinite;
    return result;
  }

  double exponential_sum = 0.0;
  for (std::size_t index = 0U; index < element_count; ++index) {
    exponential_sum +=
        std::exp(static_cast<double>(detail::bf16_to_float(logits[index])) -
                 static_cast<double>(maximum));
  }
  const double logsumexp =
      static_cast<double>(maximum) + std::log(exponential_sum);
  result.status = detail::LogitsAnalysisStatus::kSuccess;
  result.predicted_index = maximum_index;
  result.maximum = maximum;
  result.logsumexp = logsumexp;
  result.max_log_probability = static_cast<double>(maximum) - logsumexp;
  return result;
}

class DeterministicRng {
 public:
  [[nodiscard]] std::uint32_t next() noexcept {
    state_ ^= state_ << 13U;
    state_ ^= state_ >> 17U;
    state_ ^= state_ << 5U;
    return state_;
  }

 private:
  std::uint32_t state_ = 0x8f31'a6d5U;
};

void expect_bits_analysis_matches_scalar(
    TestContext& test, const std::vector<std::uint16_t>& logits,
    const std::string_view message) {
  const detail::LogitsAnalysis expected =
      scalar_bf16_logits_bits_analysis(logits.data(), logits.size());
  const detail::LogitsAnalysis actual =
      detail::analyze_bf16_logits_bits(logits.data(), logits.size());
  test.expect(same_analysis_bits(actual, expected), message);
}

void test_bf16_rounding(TestContext& test) {
  test.expect(detail::float_to_bf16_rne(1.00390625F) == 0x3F80U,
              "BF16 halfway tie with even low bit rounds down");
  test.expect(detail::float_to_bf16_rne(1.01171875F) == 0x3F82U,
              "BF16 halfway tie with odd low bit rounds up");
  test.expect(detail::float_to_bf16_rne(-0.0F) == 0x8000U,
              "BF16 conversion preserves negative zero");
  test.expect(detail::round_float_to_bf16(1.00390625F) == 1.0F &&
                  detail::bf16_to_float(0x3F82U) == 1.015625F,
              "BF16 expansion and round-trip are exact");

  const std::uint16_t nan_bits = detail::float_to_bf16_rne(
      std::numeric_limits<float>::quiet_NaN());
  test.expect((nan_bits & 0x7FC0U) == 0x7FC0U &&
                  std::isnan(detail::bf16_to_float(nan_bits)),
              "NaN remains a quiet BF16 NaN");
}

void test_logits_analysis(TestContext& test) {
  std::array<float, 4U> logits{1.00390625F, 2.0F, 2.0F, -4.0F};
  const detail::LogitsAnalysis analysis =
      detail::analyze_bf16_logits_in_place(logits.data(), logits.size());
  const double expected_lse =
      2.0 + std::log(2.0 + std::exp(-1.0) + std::exp(-6.0));
  test.expect(analysis.ok() && analysis.predicted_index == 1U,
              "argmax tie selects the smallest token id");
  test.expect(logits[0] == 1.0F && analysis.maximum == 2.0F,
              "analysis rounds logits to BF16 then expands to FP32");
  test.expect(close(analysis.logsumexp, expected_lse) &&
                  close(analysis.max_log_probability,
                        2.0 - expected_lse),
              "stable logsumexp and maximum log-probability are exact");

  detail::LogitsAnalysis invalid =
      detail::analyze_bf16_logits_in_place(nullptr, 1U);
  test.expect(invalid.status == detail::LogitsAnalysisStatus::kInvalidArgument,
              "null logits are rejected");
  float one = 0.0F;
  invalid = detail::analyze_bf16_logits_in_place(&one, 0U);
  test.expect(invalid.status == detail::LogitsAnalysisStatus::kInvalidArgument,
              "empty logits are rejected");

  for (const float nonfinite :
       {std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    float value = nonfinite;
    const detail::LogitsAnalysis rejected =
        detail::analyze_bf16_logits_in_place(&value, 1U);
    test.expect(rejected.status == detail::LogitsAnalysisStatus::kNonFinite,
                "all nonfinite logits poison the analysis boundary");
  }
}

void test_bf16_argmax_analysis(TestContext& test) {
  std::array<float, 7U> full_values{
      1.00390625F, 2.0F, 2.0F, -4.0F, -0.0F, 0.0F, 1.5F};
  std::array<float, 7U> greedy_values = full_values;
  const detail::LogitsAnalysis full =
      detail::analyze_bf16_logits_in_place(full_values.data(),
                                           full_values.size());
  const detail::LogitsAnalysis greedy =
      detail::analyze_bf16_argmax_in_place(greedy_values.data(),
                                           greedy_values.size());
  test.expect(full.ok() && greedy.ok() &&
                  greedy.predicted_index == full.predicted_index &&
                  same_bits(greedy.maximum, full.maximum) &&
                  std::memcmp(greedy_values.data(), full_values.data(),
                              full_values.size() * sizeof(float)) == 0,
              "prediction-only FP32 analysis preserves BF16 rounding and argmax");

  const std::array<std::uint16_t, 6U> bits{
      0x8000U, 0x0000U, 0xbf80U, 0x8000U, 0x0000U, 0xbf00U};
  const detail::LogitsAnalysis full_bits =
      detail::analyze_bf16_logits_bits(bits.data(), bits.size());
  const detail::LogitsAnalysis greedy_bits =
      detail::analyze_bf16_argmax_bits(bits.data(), bits.size());
  test.expect(full_bits.ok() && greedy_bits.ok() &&
                  greedy_bits.predicted_index == 0U &&
                  greedy_bits.predicted_index == full_bits.predicted_index &&
                  same_bits(greedy_bits.maximum, full_bits.maximum) &&
                  detail::float_to_bf16_rne(greedy_bits.maximum) == 0x8000U,
              "prediction-only BF16 bits preserve the earliest signed-zero tie");

  test.expect(detail::analyze_bf16_argmax_bits(nullptr, 1U).status ==
                      detail::LogitsAnalysisStatus::kInvalidArgument &&
                  detail::analyze_bf16_argmax_in_place(nullptr, 1U).status ==
                      detail::LogitsAnalysisStatus::kInvalidArgument,
              "prediction-only analyzers reject null storage");
  constexpr std::array<std::uint16_t, 4U> kPoisoned{
      0x3f80U, 0x7f80U, 0x4000U, 0x7fc1U};
  test.expect(detail::analyze_bf16_argmax_bits(kPoisoned.data(),
                                               kPoisoned.size())
                      .status == detail::LogitsAnalysisStatus::kNonFinite,
              "prediction-only BF16 analysis retains nonfinite rejection");
}

void test_bf16_logits_bits_analysis(TestContext& test) {
  std::vector<float> rounded_source{1.25F, -2.0F, 3.5F, 3.5F, 0.0F};
  std::vector<std::uint16_t> bits;
  bits.reserve(rounded_source.size());
  for (const float value : rounded_source) {
    bits.push_back(detail::float_to_bf16_rne(value));
  }
  const auto from_bits =
      detail::analyze_bf16_logits_bits(bits.data(), bits.size());
  const auto from_fp32 = detail::analyze_bf16_logits_in_place(
      rounded_source.data(), rounded_source.size());
  test.expect(from_bits.ok() && from_fp32.ok() &&
                  from_bits.predicted_index == from_fp32.predicted_index &&
                  from_bits.predicted_index == 2U &&
                  from_bits.maximum == from_fp32.maximum &&
                  from_bits.logsumexp == from_fp32.logsumexp &&
                  from_bits.max_log_probability ==
                      from_fp32.max_log_probability,
              "packed BF16 logits preserve analysis and earliest ties");

  test.expect(detail::analyze_bf16_logits_bits(nullptr, 1U).status ==
                  detail::LogitsAnalysisStatus::kInvalidArgument &&
                  detail::analyze_bf16_logits_bits(bits.data(), 0U).status ==
                      detail::LogitsAnalysisStatus::kInvalidArgument,
              "packed BF16 logits reject empty storage");
  constexpr std::uint16_t kNonFinite[] = {0x3f80U, 0x7f80U};
  test.expect(detail::analyze_bf16_logits_bits(kNonFinite, 2U).status ==
                  detail::LogitsAnalysisStatus::kNonFinite,
              "packed BF16 logits reject non-finite values");

  const std::vector<std::uint16_t> signed_zero_tie{
      0x8000U, 0x0000U, 0xbf80U, 0x8000U, 0x0000U};
  expect_bits_analysis_matches_scalar(
      test, signed_zero_tie,
      "memoized packed logits preserve negative-zero earliest ties");
  const detail::LogitsAnalysis signed_zero_analysis =
      detail::analyze_bf16_logits_bits(signed_zero_tie.data(),
                                       signed_zero_tie.size());
  test.expect(signed_zero_analysis.predicted_index == 0U &&
                  detail::float_to_bf16_rne(
                      signed_zero_analysis.maximum) == 0x8000U,
              "memoized packed logits retain the first signed-zero maximum");

  DeterministicRng rng;
  std::array<std::uint16_t, 257U> repeated_codes{};
  for (std::size_t index = 0U; index < repeated_codes.size(); ++index) {
    std::uint16_t code = static_cast<std::uint16_t>(rng.next());
    if ((code & 0x7f80U) == 0x7f80U) {
      code ^= 0x0080U;
    }
    repeated_codes[index] = code;
  }
  std::vector<std::uint16_t> repeated_logits(32'769U);
  for (std::uint16_t& code : repeated_logits) {
    code = repeated_codes[rng.next() % repeated_codes.size()];
  }
  repeated_logits[0U] = 0x4200U;
  repeated_logits[17'003U] = 0x4200U;
  expect_bits_analysis_matches_scalar(
      test, repeated_logits,
      "memoized packed logits match scalar order for repeated random codes");

  std::vector<std::uint16_t> all_finite_codes;
  all_finite_codes.reserve(65'280U);
  for (std::uint32_t code = 0U; code <= 0xffffU; ++code) {
    if ((code & 0x7f80U) != 0x7f80U) {
      all_finite_codes.push_back(static_cast<std::uint16_t>(code));
    }
  }
  for (std::size_t remaining = all_finite_codes.size(); remaining > 1U;
       --remaining) {
    const std::size_t selected = rng.next() % remaining;
    const std::uint16_t temporary = all_finite_codes[remaining - 1U];
    all_finite_codes[remaining - 1U] = all_finite_codes[selected];
    all_finite_codes[selected] = temporary;
  }
  test.expect(all_finite_codes.size() == 65'280U,
              "exhaustive finite BF16 fixture covers every finite code");
  expect_bits_analysis_matches_scalar(
      test, all_finite_codes,
      "memoized packed logits match scalar order for every finite BF16 code");

  constexpr std::array<std::uint16_t, 6U> kNonFiniteCodes{
      0x7f80U, 0xff80U, 0x7f81U, 0x7fc0U, 0xff81U, 0xffffU};
  for (const std::uint16_t nonfinite : kNonFiniteCodes) {
    const std::vector<std::uint16_t> poisoned{
        0x3f80U, nonfinite, 0x4000U, 0x3f80U};
    expect_bits_analysis_matches_scalar(
        test, poisoned,
        "memoized packed logits match scalar rejection for non-finite codes");
  }

  // Exercise two complete uint8 generation cycles. Reusing every code while
  // changing both the maximum and its earliest position catches stale memo
  // values if the wrap path fails to invalidate old stamps.
  std::array<std::uint16_t, 255U> wrap_probe_codes{};
  for (std::size_t index = 0U; index < wrap_probe_codes.size(); ++index) {
    // Distinct finite negative codes that never become the maximum.
    wrap_probe_codes[index] =
        static_cast<std::uint16_t>(0xbf80U + index);
  }
  std::vector<std::uint16_t> wrap_logits(7U);
  for (std::size_t iteration = 0U; iteration < 512U; ++iteration) {
    const std::uint16_t maximum = detail::float_to_bf16_rne(
        2.0F + static_cast<float>(iteration % 31U) / 8.0F);
    const std::uint16_t probe =
        wrap_probe_codes[iteration % wrap_probe_codes.size()];
    wrap_logits = {0x3f00U, maximum, probe, 0x8000U,
                   maximum, 0x0000U, 0x3f80U};
    const std::size_t shift = iteration % wrap_logits.size();
    std::rotate(wrap_logits.begin(), wrap_logits.begin() + shift,
                wrap_logits.end());
    expect_bits_analysis_matches_scalar(
        test, wrap_logits,
        "memoized packed logits invalidate stale values across stamp wrap");
  }
}

using BitsAnalyzer = detail::LogitsAnalysis (*)(
    const std::uint16_t*, std::size_t) noexcept;

struct TimedBitsAnalysis {
  detail::LogitsAnalysis analysis{};
  double milliseconds = 0.0;
};

[[nodiscard]] TimedBitsAnalysis time_bits_analysis(
    const BitsAnalyzer analyzer,
    const std::vector<std::uint16_t>& logits) {
  const auto started = std::chrono::steady_clock::now();
  const detail::LogitsAnalysis analysis =
      analyzer(logits.data(), logits.size());
  const auto stopped = std::chrono::steady_clock::now();
  return {analysis,
          std::chrono::duration<double, std::milli>(stopped - started)
              .count()};
}

[[nodiscard]] std::size_t count_unique_codes(
    const std::vector<std::uint16_t>& logits) {
  std::array<bool, 1U << 16U> seen{};
  std::size_t count = 0U;
  for (const std::uint16_t code : logits) {
    if (!seen[code]) {
      seen[code] = true;
      ++count;
    }
  }
  return count;
}

void test_bf16_logits_memo_perf(TestContext& test) {
  const char* const enabled =
      std::getenv("Q3X_RUN_BF16_LOGITS_MEMO_PERF");
  if (enabled == nullptr || enabled[0] == '\0' ||
      std::string_view(enabled) == "0") {
    return;
  }

  constexpr std::size_t kLength = runtime::kReferenceVocabularySize;
  constexpr std::size_t kPoolSize = 4'096U;
  constexpr std::size_t kMeasurementRounds = 3U;
  constexpr double kRequiredSpeedup = 2.0;
  constexpr double kRequiredSavedMilliseconds = 0.75;
  std::array<std::uint16_t, kPoolSize> code_pool{};
  for (std::size_t index = 0U; index < code_pool.size(); ++index) {
    const float value =
        static_cast<float>(static_cast<int>(index) - 2'048) / 64.0F;
    code_pool[index] = detail::float_to_bf16_rne(value);
  }

  DeterministicRng rng;
  std::vector<std::uint16_t> logits(kLength);
  for (std::uint16_t& code : logits) {
    code = code_pool[rng.next() % code_pool.size()];
  }
  logits[0U] = code_pool.back();
  logits[kLength / 2U] = code_pool.back();
  const std::size_t unique_count = count_unique_codes(logits);
  const detail::LogitsAnalysis expected =
      scalar_bf16_logits_bits_analysis(logits.data(), logits.size());
  const detail::LogitsAnalysis warmed =
      detail::analyze_bf16_logits_bits(logits.data(), logits.size());
  test.expect(same_analysis_bits(warmed, expected),
              "memoized packed-logits perf fixture matches scalar warmup");

  double baseline_total = 0.0;
  double candidate_total = 0.0;
  bool exact = true;
  for (std::size_t round = 0U; round < kMeasurementRounds; ++round) {
    const TimedBitsAnalysis baseline_first =
        time_bits_analysis(scalar_bf16_logits_bits_analysis, logits);
    const TimedBitsAnalysis candidate_first =
        time_bits_analysis(detail::analyze_bf16_logits_bits, logits);
    const TimedBitsAnalysis candidate_second =
        time_bits_analysis(detail::analyze_bf16_logits_bits, logits);
    const TimedBitsAnalysis baseline_second =
        time_bits_analysis(scalar_bf16_logits_bits_analysis, logits);
    baseline_total +=
        baseline_first.milliseconds + baseline_second.milliseconds;
    candidate_total +=
        candidate_first.milliseconds + candidate_second.milliseconds;
    exact = exact && same_analysis_bits(baseline_first.analysis, expected) &&
            same_analysis_bits(candidate_first.analysis, expected) &&
            same_analysis_bits(candidate_second.analysis, expected) &&
            same_analysis_bits(baseline_second.analysis, expected);
    std::cout << "PERF_BF16_LOGITS_MEMO_ROUND: round=" << round + 1U
              << " length=" << kLength << " unique=" << unique_count
              << " baseline_pass1_ms=" << baseline_first.milliseconds
              << " candidate_pass1_ms=" << candidate_first.milliseconds
              << " candidate_pass2_ms=" << candidate_second.milliseconds
              << " baseline_pass2_ms=" << baseline_second.milliseconds
              << '\n';
  }

  constexpr double kTimedPasses =
      2.0 * static_cast<double>(kMeasurementRounds);
  const double baseline_milliseconds = baseline_total / kTimedPasses;
  const double candidate_milliseconds = candidate_total / kTimedPasses;
  const double speedup = baseline_milliseconds / candidate_milliseconds;
  const double saved_milliseconds =
      baseline_milliseconds - candidate_milliseconds;
  const bool gate = exact && std::isfinite(speedup) &&
                    baseline_milliseconds > 0.0 &&
                    candidate_milliseconds > 0.0 &&
                    speedup >= kRequiredSpeedup &&
                    saved_milliseconds >= kRequiredSavedMilliseconds;
  std::cout << "PERF_BF16_LOGITS_MEMO: length=" << kLength
            << " unique=" << unique_count
            << " baseline_ms=" << baseline_milliseconds
            << " candidate_ms=" << candidate_milliseconds
            << " saved_ms=" << saved_milliseconds
            << " speedup=" << speedup
            << " required_speedup=" << kRequiredSpeedup
            << " required_saved_ms=" << kRequiredSavedMilliseconds
            << " exact=" << (exact ? "true" : "false")
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate,
              "memoized packed-logits analyzer clears the B-C-C-B gate");
}

void test_gdn_prompt_span_plan(TestContext& test) {
  constexpr detail::GdnPromptSpanWorkspacePlan maximum_workspace =
      detail::gdn_prompt_span_workspace_plan(4'096U);
  static_assert(maximum_workspace.valid());
  static_assert(maximum_workspace.total_bytes == 228'065'280U);
  static_assert(maximum_workspace.compact_q.size_bytes == 16U << 20U);
  static_assert(maximum_workspace.compact_k.size_bytes == 16U << 20U);
  static_assert(maximum_workspace.convolved_v.size_bytes == 48U << 20U);
  static_assert(maximum_workspace.transform.size_bytes == 24U << 20U);
  static_assert(maximum_workspace.w.size_bytes == 48U << 20U);
  static_assert(maximum_workspace.u.size_bytes == 48U << 20U);
  static_assert(maximum_workspace.raw_gram.size_bytes == 16U << 20U);
  static_assert(maximum_workspace.gamma.size_bytes == 786'432U);
  static_assert(maximum_workspace.beta.size_bytes == 786'432U);

  constexpr detail::GdnPromptSpanWorkspacePlan p2048_workspace =
      detail::gdn_prompt_span_workspace_plan(2'048U);
  constexpr detail::GdnPromptSpanWorkspacePlan p544_workspace =
      detail::gdn_prompt_span_workspace_plan(544U);
  test.expect(
      maximum_workspace.valid() && maximum_workspace.chunk_count == 64U &&
          p2048_workspace.valid() &&
          p2048_workspace.total_bytes == 114'032'640U &&
          p544_workspace.valid() && p544_workspace.chunk_count == 9U &&
          p544_workspace.padded_token_capacity == 576U &&
          p544_workspace.total_bytes == 32'071'680U &&
          !detail::gdn_prompt_span_workspace_plan(512U).valid() &&
          !detail::gdn_prompt_span_workspace_plan(4'097U).valid(),
      "prompt-span GDN workspace scales by C64 and caps exactly at 217.5 MiB");

  constexpr detail::GdnPromptSpanNativePlan p544 =
      detail::gdn_prompt_span_native_plan(544U);
  constexpr detail::GdnPromptSpanNativePlan p1024 =
      detail::gdn_prompt_span_native_plan(1'024U);
  constexpr detail::GdnPromptSpanNativePlan p2048 =
      detail::gdn_prompt_span_native_plan(2'048U);
  constexpr detail::GdnPromptSpanNativePlan p4096 =
      detail::gdn_prompt_span_native_plan(4'096U);
  test.expect(
      p544.valid() && p544.chunk_count == 9U &&
          p544.virtual_c512_tile_count == 2U &&
          p544.intermediate_bf16_state_boundaries == 1U &&
          p1024.valid() && p1024.chunk_count == 16U &&
          p1024.intermediate_bf16_state_boundaries == 1U &&
          p2048.valid() && p2048.chunk_count == 32U &&
          p2048.intermediate_bf16_state_boundaries == 3U &&
          p4096.valid() && p4096.chunk_count == 64U &&
          p4096.virtual_c512_tile_count == 8U &&
          p4096.intermediate_bf16_state_boundaries == 7U &&
          !detail::gdn_prompt_span_native_plan(513U).valid() &&
          !detail::gdn_prompt_span_native_plan(1'025U).valid() &&
          detail::gdn_prompt_span_native_plan(1'056U).valid() &&
          !detail::gdn_prompt_span_native_plan(4'097U).valid(),
      "prompt-span GDN plan preserves each intermediate C512 BF16 boundary "
      "and rejects the incumbent exact tail");

  const auto selected = [&](const bool admission_enabled,
                            const bool kernel_available,
                            const runtime::ProjectionBackend backend,
                            const q3x::model::LayerType layer_type,
                            const bool capture_trace,
                            const bool optimized_disabled,
                            const std::size_t workspace_bytes) {
    return detail::use_gdn_prompt_span_native_prefill(
        admission_enabled, kernel_available, backend, layer_type,
        capture_trace, optimized_disabled, 2'048U, workspace_bytes);
  };
  test.expect(
      selected(true, true, runtime::ProjectionBackend::kSm87WeightOnly,
               q3x::model::LayerType::kLinearAttention, false, false,
               p2048.workspace_bytes) &&
          !selected(false, true,
                    runtime::ProjectionBackend::kSm87WeightOnly,
                    q3x::model::LayerType::kLinearAttention, false, false,
                    p2048.workspace_bytes) &&
          !selected(true, false,
                    runtime::ProjectionBackend::kSm87WeightOnly,
                    q3x::model::LayerType::kLinearAttention, false, false,
                    p2048.workspace_bytes) &&
          !selected(true, true, runtime::ProjectionBackend::kReference,
                    q3x::model::LayerType::kLinearAttention, false, false,
                    p2048.workspace_bytes) &&
          !selected(true, true,
                    runtime::ProjectionBackend::kSm87WeightOnly,
                    q3x::model::LayerType::kFullAttention, false, false,
                    p2048.workspace_bytes) &&
          !selected(true, true,
                    runtime::ProjectionBackend::kSm87WeightOnly,
                    q3x::model::LayerType::kLinearAttention, true, false,
                    p2048.workspace_bytes) &&
          !selected(true, true,
                    runtime::ProjectionBackend::kSm87WeightOnly,
                    q3x::model::LayerType::kLinearAttention, false, true,
                    p2048.workspace_bytes) &&
          !selected(true, true,
                    runtime::ProjectionBackend::kSm87WeightOnly,
                    q3x::model::LayerType::kLinearAttention, false, false,
                    p2048.workspace_bytes - 1U),
      "prompt-span GDN selector is independently opt-in and preserves all "
      "fallback contracts");
}

void test_schedule_and_workspace(TestContext& test) {
  std::size_t linear = 0U;
  std::size_t full = 0U;
  for (std::size_t layer = 0U;
       layer < runtime::kReferenceDecoderLayerCount; ++layer) {
    const q3x::model::LayerType type =
        detail::expected_reference_layer_type(layer);
    if (type == q3x::model::LayerType::kLinearAttention) {
      ++linear;
    } else if (type == q3x::model::LayerType::kFullAttention) {
      ++full;
    }
  }
  test.expect(linear == 48U && full == 16U &&
                  detail::expected_reference_layer_type(3U) ==
                      q3x::model::LayerType::kFullAttention &&
                  detail::expected_reference_layer_type(63U) ==
                      q3x::model::LayerType::kFullAttention &&
                  detail::expected_reference_layer_type(64U) ==
                      q3x::model::LayerType::kInvalid,
              "runner schedule is exactly 48 linear plus 16 full layers");

  test.expect(detail::use_fused_gqa_sigmoid_gate_tile(0U, 16U) &&
                  detail::use_fused_gqa_sigmoid_gate_tile(48U, 16U) &&
                  detail::use_fused_gqa_sigmoid_gate_tile(63U, 1U) &&
                  !detail::use_fused_gqa_sigmoid_gate_tile(64U, 1U) &&
                  !detail::use_fused_gqa_sigmoid_gate_tile(60U, 5U) &&
                  !detail::use_fused_gqa_sigmoid_gate_tile(0U, 0U),
              "fused GQA selector accepts only complete tiles ending by sequence 64");
  test.expect(
      detail::fused_gqa_sigmoid_gate_prefix_token_count(0U, 512U) == 64U &&
          detail::fused_gqa_sigmoid_gate_prefix_token_count(31U, 512U) ==
              33U &&
          detail::fused_gqa_sigmoid_gate_prefix_token_count(63U, 512U) ==
              1U &&
          detail::fused_gqa_sigmoid_gate_prefix_token_count(64U, 512U) ==
              0U &&
          detail::fused_gqa_sigmoid_gate_prefix_token_count(0U, 0U) == 0U,
      "large Prefill tiles retain exactly their position-bounded fused GQA prefix");

  test.expect(
      runtime::gqa_attention_splitkv_sigmoid_gate_24_4_256_workspace_elements(
          64U) == 0U &&
          runtime::gqa_attention_splitkv_sigmoid_gate_24_4_256_workspace_elements(
              65U) == 24U * 4U * 258U &&
          runtime::gqa_attention_splitkv_sigmoid_gate_24_4_256_workspace_elements(
              512U) == 24U * 4U * 258U &&
          runtime::gqa_attention_splitkv_sigmoid_gate_24_4_256_workspace_elements(
              513U) == 24U * 8U * 258U &&
          runtime::gqa_attention_splitkv_sigmoid_gate_24_4_256_workspace_elements(
              4'096U) ==
              runtime::kDecodeGqaSplitKvMaximumWorkspaceElements &&
          runtime::gqa_attention_splitkv_sigmoid_gate_24_4_256_workspace_elements(
              4'097U) == 0U,
      "decode split-KV workspace switches from 16 to 32 state CTAs and "
      "rejects positions outside [65,4096]");

  test.expect(
      detail::use_bulk_causal_gqa_sigmoid_gate_prefill(
          runtime::ProjectionBackend::kSm87WeightOnly,
          q3x::model::LayerType::kFullAttention, 0U, 256U) &&
          detail::use_bulk_causal_gqa_sigmoid_gate_prefill(
              runtime::ProjectionBackend::kSm87WeightOnly,
              q3x::model::LayerType::kFullAttention, 17U, 512U) &&
          detail::use_bulk_causal_gqa_sigmoid_gate_prefill(
              runtime::ProjectionBackend::kSm87WeightOnly,
              q3x::model::LayerType::kFullAttention, 0U, 407U) &&
          detail::use_bulk_causal_gqa_sigmoid_gate_prefill(
              runtime::ProjectionBackend::kSm87WeightOnly,
              q3x::model::LayerType::kFullAttention, 0U, 481U) &&
          detail::use_bulk_causal_gqa_sigmoid_gate_prefill(
              runtime::ProjectionBackend::kSm87WeightOnly,
              q3x::model::LayerType::kFullAttention,
              runtime::kBulkCausalGqaMaximumSequenceLength - 512U, 512U),
      "bulk causal GQA/Gate selector accepts arbitrary SM87 full-attention "
      "tiles through the final legal append");
  test.expect(
      !detail::use_bulk_causal_gqa_sigmoid_gate_prefill(
          runtime::ProjectionBackend::kReference,
          q3x::model::LayerType::kFullAttention, 0U, 256U) &&
          !detail::use_bulk_causal_gqa_sigmoid_gate_prefill(
              runtime::ProjectionBackend::kSm87WeightOnly,
              q3x::model::LayerType::kLinearAttention, 0U, 256U) &&
          !detail::use_bulk_causal_gqa_sigmoid_gate_prefill(
              runtime::ProjectionBackend::kSm87WeightOnly,
              q3x::model::LayerType::kFullAttention, 0U, 1U) &&
          !detail::use_bulk_causal_gqa_sigmoid_gate_prefill(
              runtime::ProjectionBackend::kSm87WeightOnly,
              q3x::model::LayerType::kFullAttention, 0U, 513U) &&
          !detail::use_bulk_causal_gqa_sigmoid_gate_prefill(
              runtime::ProjectionBackend::kSm87WeightOnly,
              q3x::model::LayerType::kFullAttention,
              runtime::kBulkCausalGqaMaximumSequenceLength - 511U, 512U),
      "bulk causal GQA/Gate selector preserves reference, non-full-layer, "
      "single-token, over-capacity, and causal-range fallbacks");
  test.expect(
      runtime::use_bulk_causal_gqa_group_q64_prefill(0U, 2U) &&
          runtime::use_bulk_causal_gqa_group_q64_prefill(0U, 407U) &&
          runtime::use_bulk_causal_gqa_group_q64_prefill(0U, 481U) &&
          runtime::use_bulk_causal_gqa_group_q64_prefill(0U, 512U) &&
          runtime::use_bulk_causal_gqa_group_q64_prefill(512U, 2U) &&
          runtime::use_bulk_causal_gqa_group_q64_prefill(512U, 52U) &&
          runtime::use_bulk_causal_gqa_group_q64_prefill(512U, 183U) &&
          runtime::use_bulk_causal_gqa_group_q64_prefill(512U, 201U) &&
          runtime::use_bulk_causal_gqa_group_q64_prefill(512U, 512U) &&
          !runtime::use_bulk_causal_gqa_group_q64_prefill(0U, 1U) &&
          !runtime::use_bulk_causal_gqa_group_q64_prefill(0U, 513U) &&
          !runtime::use_bulk_causal_gqa_group_q64_prefill(1U, 407U) &&
          !runtime::use_bulk_causal_gqa_group_q64_prefill(511U, 512U) &&
          !runtime::use_bulk_causal_gqa_group_q64_prefill(512U, 1U) &&
          !runtime::use_bulk_causal_gqa_group_q64_prefill(512U, 513U),
      "grouped-Q64 Tensor Core selector accepts P0/P512 C2..C512");

  test.expect(
      detail::use_a4w4_full_prefill_tile_route(true, false, false) &&
          !detail::use_a4w4_full_prefill_tile_route(false, false, false) &&
          !detail::use_a4w4_full_prefill_tile_route(true, true, false) &&
          !detail::use_a4w4_full_prefill_tile_route(true, false, true),
      "A4 Prefill selector retains residency but falls back for tracing and "
      "the unified comparator");

  constexpr std::size_t kLongMaximum =
      runtime::kBulkCausalGqaLongContextGroupQ64MaximumSequenceLength;
  constexpr std::size_t kPackedPositionCapacity =
      std::size_t{1U}
      << runtime::kBulkCausalGqaGroupQ64FirstPositionBits;
  test.expect(
      runtime::can_use_bulk_causal_gqa_long_context_group_q64_prefill(
          1'024U, 512U) &&
          runtime::can_use_bulk_causal_gqa_long_context_group_q64_prefill(
              4'096U, 512U) &&
          runtime::can_use_bulk_causal_gqa_long_context_group_q64_prefill(
              kLongMaximum - 512U, 512U) &&
          !runtime::can_use_bulk_causal_gqa_long_context_group_q64_prefill(
              1'023U, 2U) &&
          !runtime::can_use_bulk_causal_gqa_long_context_group_q64_prefill(
              1'024U, 1U) &&
          !runtime::can_use_bulk_causal_gqa_long_context_group_q64_prefill(
              1'024U, 513U) &&
          !runtime::can_use_bulk_causal_gqa_long_context_group_q64_prefill(
              kLongMaximum - 511U, 512U) &&
          !runtime::can_use_bulk_causal_gqa_long_context_group_q64_prefill(
              kPackedPositionCapacity, 2U) &&
          !runtime::can_use_bulk_causal_gqa_long_context_group_q64_prefill(
              std::numeric_limits<std::size_t>::max(), 2U),
      "long-context grouped-Q64 capability enforces P1024+, C2..C512, "
      "40K KV, 18-bit ABI, and overflow boundaries");
  const char* const long_context_environment = std::getenv(
      "Q3X_RUN_FULL_ATTENTION_LONG_CONTEXT_GROUP_Q64_ADMISSION");
  const bool long_context_run_enabled =
      long_context_environment == nullptr ||
      std::strcmp(long_context_environment, "1") == 0;
  const bool long_context_expected =
      runtime::
          bulk_causal_gqa_long_context_group_q64_admission_compiled() &&
      long_context_run_enabled &&
      !runtime::optimized_prefill_dispatch_disabled();
  test.expect(
      runtime::use_bulk_causal_gqa_long_context_group_q64_admission(
          1'024U, 512U) == long_context_expected &&
          runtime::use_bulk_causal_gqa_long_context_group_q64_admission(
              4'096U, 512U) == long_context_expected &&
          !runtime::use_bulk_causal_gqa_long_context_group_q64_admission(
              kLongMaximum - 511U, 512U),
      "long-context grouped-Q64 dispatch defaults on when built, accepts the "
      "legacy RUN selector, and preserves disable/40K fallbacks");

  constexpr std::size_t kMaximum =
      std::numeric_limits<std::size_t>::max();
  test.expect(
      detail::use_qk_rope_tile(0U, 1U) &&
          detail::use_qk_rope_tile(11U,
                                   runtime::kQkRopeTileMaximumTokens) &&
          !detail::use_qk_rope_tile(0U, 0U) &&
          !detail::use_qk_rope_tile(
              0U, runtime::kQkRopeTileMaximumTokens + 1U) &&
          !detail::use_qk_rope_tile(kMaximum, 1U) &&
          !detail::use_qk_rope_tile(
              kMaximum /
                  ((runtime::kQwenRotaryDimension / 2U) * sizeof(float)),
              1U),
      "fused full-attention preprocess selector accepts M=1..16 and rejects "
      "M=0, M=17, and position/table byte-offset overflow");

  test.expect(
      detail::use_full_attention_preprocess_tile(0U, 1U) &&
          detail::use_full_attention_preprocess_tile(
              11U, runtime::kFullAttentionPreprocessMaximumTokens) &&
          !detail::use_full_attention_preprocess_tile(0U, 0U) &&
          !detail::use_full_attention_preprocess_tile(
              0U, runtime::kFullAttentionPreprocessMaximumTokens + 1U) &&
          !detail::use_full_attention_preprocess_tile(kMaximum, 1U) &&
          !detail::use_full_attention_preprocess_tile(
              kMaximum /
                  ((runtime::kQwenRotaryDimension / 2U) * sizeof(float)),
              1U),
      "prompt-wide full-attention preprocess selector accepts M=1..512 and "
      "rejects M=0, M=513, and position/table byte-offset overflow");

  test.expect(
      detail::use_m32_prefill_residual_rms_fusion(
          32U, runtime::kReferenceHiddenSize) &&
          detail::use_m32_prefill_residual_rms_fusion(
              33U, runtime::kReferenceHiddenSize) &&
          detail::use_m32_prefill_residual_rms_fusion(
              63U, runtime::kReferenceHiddenSize) &&
          detail::use_m32_prefill_residual_rms_fusion(
              64U, runtime::kReferenceHiddenSize) &&
          detail::use_m32_prefill_residual_rms_fusion(
              256U, runtime::kReferenceHiddenSize) &&
          detail::use_m32_prefill_residual_rms_fusion(
              407U, runtime::kReferenceHiddenSize) &&
          detail::use_m32_prefill_residual_rms_fusion(
              481U, runtime::kReferenceHiddenSize) &&
          detail::use_m32_prefill_residual_rms_fusion(
              512U, runtime::kReferenceHiddenSize) &&
          !detail::use_m32_prefill_residual_rms_fusion(
              0U, runtime::kReferenceHiddenSize) &&
          !detail::use_m32_prefill_residual_rms_fusion(
              31U, runtime::kReferenceHiddenSize) &&
          !detail::use_m32_prefill_residual_rms_fusion(
              513U, runtime::kReferenceHiddenSize) &&
          !detail::use_m32_prefill_residual_rms_fusion(32U, 5'119U) &&
          !detail::use_m32_prefill_residual_rms_fusion(32U, 5'121U),
      "Prefill residual/RMS schedule accepts arbitrary M=32..512 spans and "
      "preserves smaller, over-capacity, and wrong-hidden fallbacks");

  const auto expect_residual_rms_schedule =
      [&test](const std::size_t token_count,
              const std::size_t expected_prefix,
              const std::size_t expected_tail) {
        const detail::PrefillResidualRmsM32Schedule schedule =
            detail::prefill_residual_rms_m32_schedule(
                token_count, runtime::kReferenceHiddenSize);
        test.expect(schedule.fused_prefix_tokens == expected_prefix &&
                        schedule.fallback_tail_tokens == expected_tail &&
                        schedule.valid() == (expected_prefix != 0U),
                    "Prefill residual/RMS M=" +
                        std::to_string(token_count) + " decomposes into M" +
                        std::to_string(expected_prefix) + "+M" +
                        std::to_string(expected_tail));
      };
  expect_residual_rms_schedule(31U, 0U, 0U);
  expect_residual_rms_schedule(32U, 32U, 0U);
  expect_residual_rms_schedule(33U, 32U, 1U);
  expect_residual_rms_schedule(63U, 32U, 31U);
  expect_residual_rms_schedule(64U, 64U, 0U);
  expect_residual_rms_schedule(407U, 384U, 23U);
  expect_residual_rms_schedule(481U, 480U, 1U);
  expect_residual_rms_schedule(512U, 512U, 0U);

  const runtime::RequestPlanResult built =
      runtime::build_request_memory_plan();
  test.expect(built.ok(), "default request plan builds for runner validation");
  if (!built) {
    return;
  }
  runtime::RequestMemoryPlan plan = *built.value;
  test.expect(detail::validate_reference_workspace_plan(plan) ==
                  runtime::ReferenceRunnerError::kNone,
              "canonical RequestState plan satisfies runner workspace ABI");

  runtime::RequestMemoryOptions chunk_options;
  chunk_options.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  const runtime::RequestPlanResult chunk_built =
      runtime::build_request_memory_plan(chunk_options);
  const runtime::ReferencePrefillTileResult tile_result;
  test.expect(chunk_built &&
                  runtime::kMaximumRequestPrefillChunkSize == 512U &&
                  runtime::kMaximumProjectionTileTokenCount == 64U &&
                  chunk_built.value->prefill_chunk_size == 512U &&
                  tile_result.steps.size() == 512U &&
                  detail::validate_reference_workspace_plan(
                      *chunk_built.value) ==
                      runtime::ReferenceRunnerError::kNone,
              "C512 plan and tile result satisfy the runner workspace ABI");

  runtime::RequestMemoryOptions whole_m_options;
  whole_m_options.max_sequence_length = 4'096U;
  whole_m_options.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  whole_m_options.long_prefill_token_capacity = 4'096U;
  whole_m_options.long_prefill_projection_span_capacity = 4'096U;
  whole_m_options.enable_a4_prefill_workspace = true;
  whole_m_options.max_arena_bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
  const runtime::RequestPlanResult whole_m_built =
      runtime::build_request_memory_plan(whole_m_options);
  test.expect(whole_m_built &&
                  detail::validate_reference_workspace_plan(
                      *whole_m_built.value) ==
                      runtime::ReferenceRunnerError::kNone,
              "S4096 primary, secondary, and A4 workspaces satisfy the "
              "runner ABI");
  if (whole_m_built) {
    const runtime::RequestMemoryPlan whole_m_plan = *whole_m_built.value;
    const auto expect_one_byte_short =
        [&test, &whole_m_plan](
            runtime::RequestRegion runtime::RequestMemoryPlan::* const region,
            const std::string_view message) {
          runtime::RequestMemoryPlan undersized = whole_m_plan;
          --(undersized.*region).byte_size;
          test.expect(detail::validate_reference_workspace_plan(undersized) ==
                          runtime::ReferenceRunnerError::kInvalidRequestState,
                      message);
        };
    expect_one_byte_short(
        &runtime::RequestMemoryPlan::long_prefill_projection_primary_bf16,
        "S4096 primary BF16 span one byte below capacity is rejected");
    expect_one_byte_short(
        &runtime::RequestMemoryPlan::long_prefill_projection_secondary_bf16,
        "S4096 secondary BF16 span one byte below capacity is rejected");
    expect_one_byte_short(
        &runtime::RequestMemoryPlan::prefill_a4_hidden_packed,
        "S4096 A4 hidden packed span one byte below capacity is rejected");
    expect_one_byte_short(
        &runtime::RequestMemoryPlan::prefill_a4_hidden_scales_bf16,
        "S4096 A4 hidden scales span one byte below capacity is rejected");
    expect_one_byte_short(
        &runtime::RequestMemoryPlan::prefill_a4_intermediate_packed,
        "S4096 A4 intermediate packed span one byte below capacity is "
        "rejected");
    expect_one_byte_short(
        &runtime::RequestMemoryPlan::prefill_a4_intermediate_scales_bf16,
        "S4096 A4 intermediate scales span one byte below capacity is "
        "rejected");
    expect_one_byte_short(
        &runtime::RequestMemoryPlan::prefill_a4_gateup_cta_scratch,
        "paired-Gate CTA scratch one byte below capacity is rejected");
  }

  plan.prefill_chunk_size = runtime::kMaximumRequestPrefillChunkSize;
  test.expect(detail::validate_reference_workspace_plan(plan) ==
                  runtime::ReferenceRunnerError::kInvalidRequestState,
              "chunk metadata cannot exceed the allocated workspace spans");
  plan = *chunk_built.value;
  plan.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize + 1U;
  test.expect(detail::validate_reference_workspace_plan(plan) ==
                  runtime::ReferenceRunnerError::kInvalidRequestState,
              "C513 runner metadata is rejected");
  plan = *built.value;
  plan.prefill_chunk_size = 0U;
  test.expect(detail::validate_reference_workspace_plan(plan) ==
                  runtime::ReferenceRunnerError::kInvalidRequestState,
              "zero prefill chunk metadata is rejected");

  plan = *built.value;
  plan.layers[3U].type = q3x::model::LayerType::kLinearAttention;
  test.expect(detail::validate_reference_workspace_plan(plan) ==
                  runtime::ReferenceRunnerError::kInvalidLayerSchedule,
              "wrong attention variant schedule is rejected");
  plan = *built.value;
  plan.hidden_bf16[0U].element_capacity =
      runtime::kReferenceHiddenSize - 1U;
  test.expect(detail::validate_reference_workspace_plan(plan) ==
                  runtime::ReferenceRunnerError::kInvalidRequestState,
              "undersized hidden workspace is rejected");
  plan = *built.value;
  ++plan.gqa_probability_scratch.arena_offset;
  test.expect(detail::validate_reference_workspace_plan(plan) ==
                  runtime::ReferenceRunnerError::kInvalidRequestState,
              "GQA scratch must alias the FP32 projection/logits scratch");
}

void test_fake_linear_weight_validation(TestContext& test) {
  std::uint8_t encoded_weight = 0U;
  float device_weight_scale = 0.0F;
  float device_input_scale = 0.0F;
  runtime::Fp8LinearWeight fp8;
  fp8.weight = &encoded_weight;
  fp8.weight_scale_device = &device_weight_scale;
  fp8.input_scale_device = &device_input_scale;
  fp8.weight_scale = 0.25F;
  fp8.input_scale = 0.5F;
  fp8.output_size = 7U;
  fp8.input_size = 16U;
  runtime::LinearWeight weight = fp8;
  test.expect(detail::valid_reference_linear_weight_contract(weight, 7U,
                                                               16U),
              "finite positive FP8 static input scale passes preflight");

  fp8.input_scale = 0.0F;
  weight = fp8;
  test.expect(!detail::valid_reference_linear_weight_contract(weight, 7U,
                                                                16U),
              "zero FP8 static input scale is rejected");
  fp8.input_scale = std::numeric_limits<float>::infinity();
  weight = fp8;
  test.expect(!detail::valid_reference_linear_weight_contract(weight, 7U,
                                                                16U),
              "nonfinite FP8 static input scale is rejected");
  fp8.input_scale = 0.5F;
  weight = fp8;
  test.expect(!detail::valid_reference_linear_weight_contract(weight, 8U,
                                                                16U),
              "fake linear shape mismatch is rejected");

  std::uint16_t first_bf16 = 0U;
  std::uint16_t second_bf16 = 0U;
  const runtime::LinearWeight first_pair = runtime::Bf16LinearWeight{
      &first_bf16, 48U, runtime::kReferenceHiddenSize};
  const runtime::LinearWeight second_pair = runtime::Bf16LinearWeight{
      &second_bf16, 48U, runtime::kReferenceHiddenSize};
  test.expect(runtime::supports_bf16_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first_pair,
                  second_pair) &&
                  !runtime::supports_bf16_projection_pair(
                      runtime::ProjectionBackend::kReference, first_pair,
                      second_pair),
              "decode A/B production shapes select only the SM87 BF16 pair "
              "path");
  const runtime::LinearWeight near_miss = runtime::Bf16LinearWeight{
      &second_bf16, 47U, runtime::kReferenceHiddenSize};
  test.expect(!runtime::supports_bf16_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first_pair,
                  near_miss),
              "decode A/B pair selector preserves the split fallback for "
              "shape mismatches");

  std::uint8_t second_encoded_weight = 0U;
  float second_device_weight_scale = 0.0F;
  float second_device_input_scale = 0.0F;
  const runtime::LinearWeight first_kv = runtime::Fp8LinearWeight{
      &encoded_weight, &device_weight_scale, &device_input_scale, 0.25F,
      0.5F, 1'024U, runtime::kReferenceHiddenSize};
  const runtime::LinearWeight second_kv = runtime::Fp8LinearWeight{
      &second_encoded_weight, &second_device_weight_scale,
      &second_device_input_scale, 0.125F, 0.5F, 1'024U,
      runtime::kReferenceHiddenSize};
  test.expect(runtime::supports_fp8_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first_kv,
                  second_kv) &&
                  !runtime::supports_fp8_projection_pair(
                      runtime::ProjectionBackend::kReference, first_kv,
                      second_kv),
              "decode K/V production shapes select only the SM87 FP8 M1 "
              "pair path");
  const runtime::LinearWeight near_miss_kv = runtime::Fp8LinearWeight{
      &second_encoded_weight, &second_device_weight_scale,
      &second_device_input_scale, 0.125F, 0.5F, 1'023U,
      runtime::kReferenceHiddenSize};
  test.expect(!runtime::supports_fp8_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first_kv,
                  near_miss_kv),
              "decode K/V pair selector preserves the split fallback for "
              "shape mismatches");

  alignas(16) std::array<std::uint8_t, 16U> attention_output_weight{};
  alignas(8) std::array<std::uint16_t, 8U> attention_input{};
  std::array<std::uint16_t, 8U> attention_output{};
  const runtime::LinearWeight attention_output_projection =
      runtime::Fp8LinearWeight{
          attention_output_weight.data(), &device_weight_scale,
          &device_input_scale, 0.25F, 0.5F, 5'120U, 6'144U};
  test.expect(
      detail::use_fp8_m64_prefill_attention_output_projection(
          runtime::ProjectionBackend::kSm87WeightOnly,
          attention_output_projection, attention_input.data(),
          attention_output.data(), 64U),
      "runner selects exact aligned FP8 C64 attention output for one "
      "dispatcher call");
  const runtime::LinearWeight attention_output_near_miss =
      runtime::Fp8LinearWeight{
          attention_output_weight.data(), &device_weight_scale,
          &device_input_scale, 0.25F, 0.5F, 5'119U, 6'144U};
  const runtime::LinearWeight attention_output_unaligned_weight =
      runtime::Fp8LinearWeight{
          attention_output_weight.data() + 1U, &device_weight_scale,
          &device_input_scale, 0.25F, 0.5F, 5'120U, 6'144U};
  const auto* const unaligned_attention_input =
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<const std::uint8_t*>(attention_input.data()) +
          sizeof(std::uint16_t));
  auto* const odd_attention_output = reinterpret_cast<std::uint16_t*>(
      reinterpret_cast<std::uint8_t*>(attention_output.data()) + 1U);
  test.expect(
      !detail::use_fp8_m64_prefill_attention_output_projection(
          runtime::ProjectionBackend::kReference,
          attention_output_projection, attention_input.data(),
          attention_output.data(), 64U) &&
          !detail::use_fp8_m64_prefill_attention_output_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              attention_output_projection, attention_input.data(),
              attention_output.data(), 1U) &&
          !detail::use_fp8_m64_prefill_attention_output_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              attention_output_projection, attention_input.data(),
              attention_output.data(), 32U) &&
          !detail::use_fp8_m64_prefill_attention_output_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              attention_output_projection, attention_input.data(),
              attention_output.data(), 63U) &&
          !detail::use_fp8_m64_prefill_attention_output_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              attention_output_projection, attention_input.data(),
              attention_output.data(), 512U) &&
          !detail::use_fp8_m64_prefill_attention_output_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              attention_output_near_miss, attention_input.data(),
              attention_output.data(), 64U) &&
          !detail::use_fp8_m64_prefill_attention_output_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              attention_output_unaligned_weight, attention_input.data(),
              attention_output.data(), 64U) &&
          !detail::use_fp8_m64_prefill_attention_output_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              attention_output_projection, unaligned_attention_input,
              attention_output.data(), 64U) &&
          !detail::use_fp8_m64_prefill_attention_output_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              attention_output_projection, attention_input.data(),
              odd_attention_output, 64U),
      "runner FP8 C64 selector rejects decode/C32/C512/near-miss/alignment "
      "cases without inheriting the request maximum");

  const runtime::LinearWeight qkv_whole_chunk = runtime::Fp8LinearWeight{
      attention_output_weight.data(), &device_weight_scale,
      &device_input_scale, 0.25F, 0.5F, 10'240U, 5'120U};
  const runtime::LinearWeight z_whole_chunk = runtime::Fp8LinearWeight{
      attention_output_weight.data(), &device_weight_scale,
      &device_input_scale, 0.25F, 0.5F, 6'144U, 5'120U};
  const runtime::LinearWeight full_q_whole_chunk = runtime::Fp8LinearWeight{
      attention_output_weight.data(), &device_weight_scale,
      &device_input_scale, 0.25F, 0.5F, 12'288U, 5'120U};
  const runtime::LinearWeight full_kv_whole_chunk = runtime::Fp8LinearWeight{
      attention_output_weight.data(), &device_weight_scale,
      &device_input_scale, 0.25F, 0.5F, 1'024U, 5'120U};
  const runtime::LinearWeight missing_whole_chunk_companions =
      runtime::Fp8LinearWeight{attention_output_weight.data(), nullptr,
                               nullptr, 0.25F, 0.5F, 10'240U, 5'120U};
  const auto selects_fp8_whole_chunk =
      [&](const runtime::LinearWeight& weight,
          const std::size_t token_count) noexcept {
        return detail::use_fp8_whole_chunk_prefill_projection(
            runtime::ProjectionBackend::kSm87WeightOnly, weight,
            attention_input.data(), attention_output.data(), token_count);
      };
  test.expect(
      selects_fp8_whole_chunk(qkv_whole_chunk, 256U) &&
          selects_fp8_whole_chunk(qkv_whole_chunk, 512U) &&
          selects_fp8_whole_chunk(z_whole_chunk, 256U) &&
          selects_fp8_whole_chunk(z_whole_chunk, 512U) &&
          selects_fp8_whole_chunk(full_q_whole_chunk, 256U) &&
          selects_fp8_whole_chunk(full_q_whole_chunk, 512U) &&
          selects_fp8_whole_chunk(full_kv_whole_chunk, 256U) &&
          selects_fp8_whole_chunk(full_kv_whole_chunk, 512U) &&
          selects_fp8_whole_chunk(attention_output_projection, 256U) &&
          selects_fp8_whole_chunk(attention_output_projection, 512U),
      "runner selects linear QKV/Z, full Q/K/V, and O exact aligned FP8 "
      "C256/C512 whole chunks");
  test.expect(
      selects_fp8_whole_chunk(missing_whole_chunk_companions, 256U),
      "runner FP8 whole-chunk selector leaves malformed companion scales "
      "to runtime validation");

  const runtime::LinearWeight qkv_whole_chunk_shape_near_miss =
      runtime::Fp8LinearWeight{
          attention_output_weight.data(), &device_weight_scale,
          &device_input_scale, 0.25F, 0.5F, 10'239U, 5'120U};
  const runtime::LinearWeight qkv_whole_chunk_unaligned_weight =
      runtime::Fp8LinearWeight{
          attention_output_weight.data() + 1U, &device_weight_scale,
          &device_input_scale, 0.25F, 0.5F, 10'240U, 5'120U};
  const runtime::LinearWeight other_fp8_checkpoint_shape =
      runtime::Fp8LinearWeight{
          attention_output_weight.data(), &device_weight_scale,
          &device_input_scale, 0.25F, 0.5F, 12'287U, 5'120U};
  const runtime::LinearWeight bf16_whole_chunk_near_miss =
      runtime::Bf16LinearWeight{attention_input.data(), 10'240U, 5'120U};
  test.expect(
      !detail::use_fp8_whole_chunk_prefill_projection(
          runtime::ProjectionBackend::kReference, qkv_whole_chunk,
          attention_input.data(), attention_output.data(), 256U) &&
          !selects_fp8_whole_chunk(qkv_whole_chunk, 64U) &&
          !selects_fp8_whole_chunk(qkv_whole_chunk, 255U) &&
          !selects_fp8_whole_chunk(qkv_whole_chunk, 513U) &&
          !selects_fp8_whole_chunk(qkv_whole_chunk_shape_near_miss, 256U) &&
          !selects_fp8_whole_chunk(other_fp8_checkpoint_shape, 256U) &&
          !selects_fp8_whole_chunk(qkv_whole_chunk_unaligned_weight, 256U) &&
          !detail::use_fp8_whole_chunk_prefill_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              qkv_whole_chunk, unaligned_attention_input,
              attention_output.data(), 256U) &&
          !detail::use_fp8_whole_chunk_prefill_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              qkv_whole_chunk, attention_input.data(), odd_attention_output,
              256U) &&
          !selects_fp8_whole_chunk(bf16_whole_chunk_near_miss, 256U),
      "runner FP8 whole-chunk selector rejects backend/token/shape/type/"
      "alignment near misses");

  alignas(16) std::array<std::uint8_t, 16U> down_packed{};
  alignas(2) std::array<std::uint8_t, 2U> down_scales{};
  alignas(8) std::array<std::uint16_t, 4U> down_input{};
  std::array<std::uint16_t, 4U> down_output{};
  float nvfp4_down_weight_scale = 1.0F / 64.0F;
  float nvfp4_down_input_scale = 1.0F;
  const runtime::LinearWeight down = runtime::NvFp4LinearWeight{
      down_packed.data(), down_scales.data(), &nvfp4_down_weight_scale,
      &nvfp4_down_input_scale, nvfp4_down_weight_scale,
      nvfp4_down_input_scale, runtime::kReferenceHiddenSize,
      runtime::kReferenceIntermediateSize};
  test.expect(
      detail::use_nvfp4_whole_chunk_prefill_down_projection(
          runtime::ProjectionBackend::kSm87WeightOnly, down,
          down_input.data(), down_output.data(), 256U) &&
          detail::use_nvfp4_whole_chunk_prefill_down_projection(
              runtime::ProjectionBackend::kSm87WeightOnly, down,
              down_input.data(), down_output.data(), 512U),
      "runner selects exact aligned NVFP4 Down C256/C512 whole chunks");

  const runtime::LinearWeight down_n_near_miss =
      runtime::NvFp4LinearWeight{
          down_packed.data(), down_scales.data(),
          &nvfp4_down_weight_scale, &nvfp4_down_input_scale,
          nvfp4_down_weight_scale, nvfp4_down_input_scale,
          runtime::kReferenceHiddenSize - 1U,
          runtime::kReferenceIntermediateSize};
  const runtime::LinearWeight down_k_near_miss =
      runtime::NvFp4LinearWeight{
          down_packed.data(), down_scales.data(),
          &nvfp4_down_weight_scale, &nvfp4_down_input_scale,
          nvfp4_down_weight_scale, nvfp4_down_input_scale,
          runtime::kReferenceHiddenSize,
          runtime::kReferenceIntermediateSize - 16U};
  const runtime::LinearWeight gate_shape = runtime::NvFp4LinearWeight{
      down_packed.data(), down_scales.data(), &nvfp4_down_weight_scale,
      &nvfp4_down_input_scale, nvfp4_down_weight_scale,
      nvfp4_down_input_scale, runtime::kReferenceIntermediateSize,
      runtime::kReferenceHiddenSize};
  const runtime::LinearWeight down_unaligned_weight =
      runtime::NvFp4LinearWeight{
          down_packed.data() + 4U, down_scales.data(),
          &nvfp4_down_weight_scale, &nvfp4_down_input_scale,
          nvfp4_down_weight_scale, nvfp4_down_input_scale,
          runtime::kReferenceHiddenSize,
          runtime::kReferenceIntermediateSize};
  const runtime::LinearWeight down_unaligned_scale =
      runtime::NvFp4LinearWeight{
          down_packed.data(), down_scales.data() + 1U,
          &nvfp4_down_weight_scale, &nvfp4_down_input_scale,
          nvfp4_down_weight_scale, nvfp4_down_input_scale,
          runtime::kReferenceHiddenSize,
          runtime::kReferenceIntermediateSize};
  const auto* const down_unaligned_input =
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<const std::uint8_t*>(down_input.data()) + 2U);
  auto* const down_unaligned_output = reinterpret_cast<std::uint16_t*>(
      reinterpret_cast<std::uint8_t*>(down_output.data()) + 1U);
  test.expect(
      !detail::use_nvfp4_whole_chunk_prefill_down_projection(
          runtime::ProjectionBackend::kReference, down, down_input.data(),
          down_output.data(), 256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_down_projection(
              runtime::ProjectionBackend::kSm87WeightOnly, down,
              down_input.data(), down_output.data(), 64U) &&
          !detail::use_nvfp4_whole_chunk_prefill_down_projection(
              runtime::ProjectionBackend::kSm87WeightOnly, down,
              down_input.data(), down_output.data(), 255U) &&
          !detail::use_nvfp4_whole_chunk_prefill_down_projection(
              runtime::ProjectionBackend::kSm87WeightOnly, down,
              down_input.data(), down_output.data(), 513U) &&
          !detail::use_nvfp4_whole_chunk_prefill_down_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              down_n_near_miss, down_input.data(), down_output.data(),
              256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_down_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              down_k_near_miss, down_input.data(), down_output.data(),
              256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_down_projection(
              runtime::ProjectionBackend::kSm87WeightOnly, gate_shape,
              down_input.data(), down_output.data(), 256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_down_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              down_unaligned_weight, down_input.data(), down_output.data(),
              256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_down_projection(
              runtime::ProjectionBackend::kSm87WeightOnly,
              down_unaligned_scale, down_input.data(), down_output.data(),
              256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_down_projection(
              runtime::ProjectionBackend::kSm87WeightOnly, down,
              down_unaligned_input, down_output.data(), 256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_down_projection(
              runtime::ProjectionBackend::kSm87WeightOnly, down,
              down_input.data(), down_unaligned_output, 256U),
      "runner NVFP4 Down whole-chunk selector rejects reference/C64/C255/"
      "C513/Gate/shape/alignment near misses");

  alignas(16) std::array<std::uint8_t, 16U> gate_packed{};
  alignas(16) std::array<std::uint8_t, 16U> up_packed{};
  alignas(2) std::array<std::uint8_t, 2U> gate_scales{};
  alignas(2) std::array<std::uint8_t, 2U> up_scales{};
  alignas(8) std::array<std::uint16_t, 4U> input{};
  constexpr std::size_t kGateUpRows = 17'408U;
  constexpr std::uintptr_t kGateOutputAddress = 0x40'0000'0000ULL;
  constexpr std::uintptr_t kFarUpOutputAddress = 0x60'0000'0000ULL;
  auto* const gate_output =
      reinterpret_cast<std::uint16_t*>(kGateOutputAddress);
  auto* const far_up_output =
      reinterpret_cast<std::uint16_t*>(kFarUpOutputAddress);
  float nvfp4_weight_scale = 1.0F / 64.0F;
  float nvfp4_input_scale = 1.0F;
  const runtime::LinearWeight gate = runtime::NvFp4LinearWeight{
      gate_packed.data(), gate_scales.data(), &nvfp4_weight_scale,
      &nvfp4_input_scale, nvfp4_weight_scale, nvfp4_input_scale, 17'408U,
      runtime::kReferenceHiddenSize};
  const runtime::LinearWeight up = runtime::NvFp4LinearWeight{
      up_packed.data(), up_scales.data(), &nvfp4_weight_scale,
      &nvfp4_input_scale, nvfp4_weight_scale, nvfp4_input_scale, 17'408U,
      runtime::kReferenceHiddenSize};
  const runtime::LinearWeight malformed_up_companion =
      runtime::NvFp4LinearWeight{
          up_packed.data(), up_scales.data(), nullptr, &nvfp4_input_scale,
          nvfp4_weight_scale, nvfp4_input_scale, 17'408U,
          runtime::kReferenceHiddenSize};
  test.expect(
      detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
          runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
          input.data(), gate_output, far_up_output, 256U) &&
          detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, far_up_output, 512U),
      "exact aligned NVFP4 C256/C512 MLP Gate/Up pair selects the "
      "whole-chunk two-stream fork/join");
  test.expect(
      detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
          runtime::ProjectionBackend::kSm87WeightOnly, gate,
          malformed_up_companion, input.data(), gate_output, far_up_output,
          256U),
      "whole-chunk Gate/Up selector leaves malformed companion scales to "
      "the launcher's Invalid validation");
  test.expect(
      detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
          runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
          input.data(), gate_output, far_up_output, 32U) &&
          detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, far_up_output, 64U),
      "exact aligned NVFP4 C32/C64 MLP gate/up pair selects dual-stream prefill");

  const auto output_bytes = [](const std::size_t token_count) constexpr {
    return token_count * kGateUpRows * sizeof(std::uint16_t);
  };
  const auto output_at_byte_offset =
      [](const std::uintptr_t base, const std::size_t offset) noexcept {
        return reinterpret_cast<std::uint16_t*>(base + offset);
      };
  auto* const wrapping_output = reinterpret_cast<std::uint16_t*>(
      std::numeric_limits<std::uintptr_t>::max() &
      ~(static_cast<std::uintptr_t>(alignof(std::uint16_t)) - 1U));
  test.expect(
      detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
          runtime::ProjectionBackend::kSm87WeightOnly, gate, up, input.data(),
          gate_output,
          output_at_byte_offset(kGateOutputAddress, output_bytes(256U)),
          256U) &&
          detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output,
              output_at_byte_offset(kGateOutputAddress, output_bytes(512U)),
              512U),
      "whole-chunk C256/C512 dual-stream selector accepts exactly adjacent "
      "complete output spans");
  test.expect(
      detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
          runtime::ProjectionBackend::kSm87WeightOnly, gate, up, input.data(),
          gate_output,
          output_at_byte_offset(kGateOutputAddress, output_bytes(32U)), 32U) &&
          detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output,
              output_at_byte_offset(kGateOutputAddress, output_bytes(64U)),
              64U),
      "C32/C64 dual-stream selector accepts exactly adjacent complete output "
      "spans");
  auto* const aligned_partial_output =
      output_at_byte_offset(kGateOutputAddress, 128U);
  test.expect(
      !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
          runtime::ProjectionBackend::kSm87WeightOnly, gate, up, input.data(),
          gate_output, gate_output, 256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, aligned_partial_output, 256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), wrapping_output, far_up_output, 512U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, wrapping_output, 512U),
      "whole-chunk dual-stream selector rejects identical, aligned partial, "
      "and wrapping output spans");
  test.expect(
      !detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
          runtime::ProjectionBackend::kSm87WeightOnly, gate, up, input.data(),
          gate_output, gate_output, 32U) &&
          !detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, aligned_partial_output, 64U) &&
          !detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), wrapping_output, far_up_output, 32U) &&
          !detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, wrapping_output, 64U),
      "C32/C64 dual-stream selector rejects identical, aligned partial, and "
      "wrapping output spans");
  test.expect(
      !detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
          runtime::ProjectionBackend::kReference, gate, up, input.data(),
          gate_output, far_up_output, 32U) &&
          !detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, far_up_output, 31U) &&
          !detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, far_up_output, 63U) &&
          !detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, gate_output, 32U),
      "dual-stream prefill selector preserves backend, tile, and output-span "
      "fallbacks");
  const runtime::LinearWeight wrong_shape = runtime::NvFp4LinearWeight{
      up_packed.data(), up_scales.data(), &nvfp4_weight_scale,
      &nvfp4_input_scale, nvfp4_weight_scale, nvfp4_input_scale, 17'407U,
      runtime::kReferenceHiddenSize};
  const runtime::LinearWeight unaligned_gate_weight =
      runtime::NvFp4LinearWeight{
          gate_packed.data() + 4U, gate_scales.data(), &nvfp4_weight_scale,
          &nvfp4_input_scale, nvfp4_weight_scale, nvfp4_input_scale, 17'408U,
          runtime::kReferenceHiddenSize};
  const runtime::LinearWeight unaligned_up_scale = runtime::NvFp4LinearWeight{
      up_packed.data(), up_scales.data() + 1U, &nvfp4_weight_scale,
      &nvfp4_input_scale, nvfp4_weight_scale, nvfp4_input_scale, 17'408U,
      runtime::kReferenceHiddenSize};
  auto* const odd_gate_output = reinterpret_cast<std::uint16_t*>(
      kGateOutputAddress + 1U);
  test.expect(
      !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
          runtime::ProjectionBackend::kReference, gate, up, input.data(),
          gate_output, far_up_output, 256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, far_up_output, 64U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, far_up_output, 255U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, far_up_output, 513U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, wrong_shape,
              input.data(), gate_output, far_up_output, 256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly,
              unaligned_gate_weight, up, input.data(), gate_output,
              far_up_output, 256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate,
              unaligned_up_scale, input.data(), gate_output, far_up_output,
              256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), gate_output, gate_output, 256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              input.data(), odd_gate_output, far_up_output, 256U) &&
          !detail::use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              reinterpret_cast<const std::uint16_t*>(
                  reinterpret_cast<const std::uint8_t*>(input.data()) + 2U),
              gate_output, far_up_output, 256U),
      "whole-chunk Gate/Up selector preserves backend/C64/token/shape/output/"
      "alignment fallbacks");
  test.expect(
      !detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
          runtime::ProjectionBackend::kSm87WeightOnly, gate, wrong_shape,
          input.data(), gate_output, far_up_output, 32U) &&
          !detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              runtime::ProjectionBackend::kSm87WeightOnly, gate, up,
              reinterpret_cast<const std::uint16_t*>(
                  reinterpret_cast<const std::uint8_t*>(input.data()) + 2U),
              gate_output, far_up_output, 32U),
      "dual-stream prefill selector preserves shape and alignment fallbacks");
}

void test_trace_layout_and_factory_error(TestContext& test) {
  static std::array<std::uint16_t, runtime::kReferenceTraceElements> trace{};
  trace[0U] = 11U;
  trace[runtime::kReferenceHiddenSize] = 22U;
  trace[2U * runtime::kReferenceHiddenSize] = 33U;
  trace[(1U + 2U * runtime::kReferenceDecoderLayerCount) *
        runtime::kReferenceHiddenSize] = 44U;
  const runtime::ReferenceTraceView view{
      7U, 9U, trace.data(), trace.size()};
  test.expect(view.raw().size == runtime::kReferenceTraceElements &&
                  view.embedding()[0U] == 11U &&
                  view.layer_hidden(0U)[0U] == 22U &&
                  view.layer_residual(0U)[0U] == 33U &&
                  view.final_norm()[0U] == 44U &&
                  view.layer_hidden(64U).empty(),
              "trace spans expose fixture-aligned embedding/hidden/residual/final layout");

  const runtime::ReferenceRunnerFactoryResult rejected =
      runtime::create_reference_runner(nullptr, nullptr);
  test.expect(!rejected &&
                  rejected.diagnostic.error ==
                      runtime::ReferenceRunnerError::kInvalidDependency &&
                  rejected.diagnostic.cuda_error == 0,
              "factory rejects null dependencies before touching CUDA");
  test.expect(std::string_view(runtime::reference_runner_error_string(
                  runtime::ReferenceRunnerError::kNonFiniteLogits)) ==
                  "nonfinite_logits",
              "runner diagnostic strings are stable");
  test.expect(runtime::is_valid_reference_logits_mode(
                  runtime::ReferenceLogitsMode::kFullStatistics) &&
                  runtime::is_valid_reference_logits_mode(
                      runtime::ReferenceLogitsMode::kPredictedTokenOnly) &&
                  !runtime::is_valid_reference_logits_mode(
                      static_cast<runtime::ReferenceLogitsMode>(255U)) &&
                  std::string_view(runtime::reference_runner_error_string(
                      runtime::ReferenceRunnerError::kInvalidStepOptions)) ==
                      "invalid_step_options",
              "logits modes and invalid-step diagnostics are stable");
}

void test_a4w4_full_prefill_admission_controls(TestContext& test) {
  using Consumer = detail::A4W4PrefillConsumer;
  const Consumer k64 = detail::a4w4_prefill_consumer_from_contract(
      runtime::PrefillSidecarKind::kA4K64, 64U, 64U);
  const Consumer k128 = detail::a4w4_prefill_consumer_from_contract(
      runtime::PrefillSidecarKind::kA4K128, 64U, 128U);
  const Consumer k256 = detail::a4w4_prefill_consumer_from_contract(
      runtime::PrefillSidecarKind::kA4K256, 64U, 256U);
  test.expect(
      detail::authenticated_a4_payload_bytes_for_kind(
          runtime::PrefillSidecarKind::kA4K64) ==
              runtime::kPrefillA4K64SidecarPayloadBytes &&
          detail::authenticated_a4_payload_bytes_for_kind(
              runtime::PrefillSidecarKind::kA4K128) ==
              runtime::kPrefillA4K128SidecarPayloadBytes &&
          detail::authenticated_a4_payload_bytes_for_kind(
              runtime::PrefillSidecarKind::kA4K256) ==
              runtime::kPrefillA4K256SidecarPayloadBytes &&
          detail::authenticated_a4_payload_bytes_for_kind(
              runtime::PrefillSidecarKind::kExact) == 0U,
      "engine A4 payload gate selects all authenticated A4 receipt byte identities");
  test.expect(
      k64 == Consumer::kK64 && k128 == Consumer::kK128 &&
          k256 == Consumer::kK256 &&
          detail::a4w4_prefill_consumer_from_contract(
              runtime::PrefillSidecarKind::kA4K64, 64U, 128U) ==
              Consumer::kUnavailable &&
          detail::a4w4_prefill_consumer_from_contract(
              runtime::PrefillSidecarKind::kA4K128, 128U, 128U) ==
              Consumer::kUnavailable &&
          detail::a4w4_prefill_consumer_from_contract(
              runtime::PrefillSidecarKind::kA4K256, 64U, 128U) ==
              Consumer::kUnavailable,
      "A4 consumer selector binds kind, packed layout, and scale group");
  test.expect(
      detail::a4w4_prefill_inventory_consumers_match(k64, k64) &&
          detail::a4w4_prefill_inventory_consumers_match(k128, k128) &&
          detail::a4w4_prefill_inventory_consumers_match(k256, k256) &&
          !detail::a4w4_prefill_inventory_consumers_match(k64, k128) &&
          !detail::a4w4_prefill_inventory_consumers_match(k128, k256),
      "mixed K64/K128/K256 inventories fail the immutable consumer selector");
  test.expect(
      detail::a4w4_prefill_consumer_supports_token_count(k64, 1U) &&
          detail::a4w4_prefill_consumer_supports_token_count(k64, 513U) &&
          !detail::a4w4_prefill_consumer_supports_token_count(k128, 63U) &&
          detail::a4w4_prefill_consumer_supports_token_count(k128, 64U) &&
          !detail::a4w4_prefill_consumer_supports_token_count(k128, 513U) &&
          detail::a4w4_prefill_consumer_supports_token_count(k128, 4'096U) &&
          detail::a4w4_prefill_consumer_supports_token_count(k128, 40'000U) &&
          !detail::a4w4_prefill_consumer_supports_token_count(k256, 64U) &&
          !detail::a4w4_prefill_consumer_supports_token_count(k256, 128U) &&
          !detail::a4w4_prefill_consumer_supports_token_count(k256, 4'096U),
      "K256 remains projection-span-only while K64/K128 preserve their "
      "complete-M tile contracts");
  const auto p1853 =
      detail::a4w4_projection_span_padding_plan(k128, 1'853U, 4'096U);
  const auto p481 =
      detail::a4w4_projection_span_padding_plan(k128, 481U, 512U);
  const auto p3987 =
      detail::a4w4_projection_span_padding_plan(k128, 3'987U, 4'096U);
  const auto p40959_tail =
      detail::a4w4_projection_span_padding_plan(k128, 4'095U, 4'096U);
  const auto p40960_span =
      detail::a4w4_projection_span_padding_plan(k128, 4'096U, 4'096U);
  const auto p1853_k64 =
      detail::a4w4_projection_span_padding_plan(k64, 1'853U, 4'096U);
  const auto p1853_k256 =
      detail::a4w4_projection_span_padding_plan(k256, 1'853U, 4'096U);
  test.expect(
      p481.valid() && p481.logical_token_count == 481U &&
          p481.projection_token_count == 512U &&
          p481.padding_token_count == 31U && p1853.valid() &&
          p1853.logical_token_count == 1'853U &&
          p1853.projection_token_count == 1'920U &&
          p1853.padding_token_count == 67U &&
          p1853.projection_token_count % 128U == 0U && p3987.valid() &&
          p3987.projection_token_count == 4'096U &&
          p3987.padding_token_count == 109U &&
          p3987.projection_token_count % 128U == 0U &&
          p40959_tail.valid() &&
          p40959_tail.projection_token_count == 4'096U &&
          p40959_tail.padding_token_count == 1U && p40960_span.valid() &&
          p40960_span.padding_token_count == 0U && p1853_k64.valid() &&
          p1853_k64.projection_token_count == 1'853U &&
          p1853_k64.padding_token_count == 0U && p1853_k256.valid() &&
          p1853_k256.projection_token_count == 1'920U &&
          p1853_k256.padding_token_count == 67U,
      "K128/K256 whole-M padding maps natural prompts to M128 while K64 remains "
      "unpadded");
  test.expect(
      detail::a4w4_prefill_consumer_supports_projection_span_prompt(
          k128, 481U, 512U) &&
          detail::a4w4_prefill_consumer_supports_projection_span_prompt(
              k128, 1'853U, 4'096U) &&
          detail::a4w4_prefill_consumer_supports_projection_span_prompt(
              k128, 1'853U, 1'920U) &&
          detail::a4w4_prefill_consumer_supports_projection_span_prompt(
              k128, 3'987U, 4'096U) &&
          detail::a4w4_prefill_consumer_supports_projection_span_prompt(
              k128, 40'959U, 4'096U) &&
          detail::a4w4_prefill_consumer_supports_projection_span_prompt(
              k128, 40'960U, 4'096U) &&
          detail::a4w4_prefill_consumer_supports_projection_span_prompt(
              k64, 40'959U, 4'096U) &&
          !detail::a4w4_prefill_consumer_supports_projection_span_prompt(
              Consumer::kUnavailable, 1'853U, 4'096U) &&
          !detail::a4w4_prefill_consumer_supports_projection_span_prompt(
              k128, 1'853U, 1'919U) &&
          !detail::a4w4_prefill_consumer_supports_projection_span_prompt(
              k128, 4'097U, 4'095U) &&
          !detail::a4w4_projection_span_padding_plan(
               k128, 4'095U, 4'095U).valid(),
      "K128 natural-prompt route proves every full/tail span fits and fails "
      "closed when ceil128 exceeds capacity");

  using CellQuery = detail::A4W4GateUpCompleteCellV2RouteQuery;
  const auto make_cell_query = [k128](
                                   const std::size_t projection_tokens,
                                   const std::size_t workspace_tokens) {
    CellQuery query;
    query.admission_enabled = true;
    query.inventory_consumer = k128;
    query.projection_token_count = projection_tokens;
    query.gate_output_size = runtime::kReferenceIntermediateSize;
    query.gate_input_size = runtime::kReferenceHiddenSize;
    query.up_output_size = runtime::kReferenceIntermediateSize;
    query.up_input_size = runtime::kReferenceHiddenSize;
    query.packed_input_capacity_bytes =
        workspace_tokens * runtime::kReferenceHiddenSize / 2U;
    query.input_scale_capacity_elements =
        workspace_tokens * runtime::kReferenceHiddenSize / 128U;
    query.gate_weight_capacity_bytes =
        runtime::kReferenceIntermediateSize *
        runtime::kReferenceHiddenSize / 2U;
    query.gate_scale_capacity_elements =
        runtime::kReferenceIntermediateSize *
        runtime::kReferenceHiddenSize / 128U;
    query.up_weight_capacity_bytes = query.gate_weight_capacity_bytes;
    query.up_scale_capacity_elements = query.gate_scale_capacity_elements;
    query.packed_output_capacity_bytes =
        workspace_tokens * runtime::kReferenceIntermediateSize / 2U;
    query.output_scale_capacity_elements =
        workspace_tokens * runtime::kReferenceIntermediateSize / 128U;
    return query;
  };
  const CellQuery real_cell = make_cell_query(2'048U, 2'048U);
  const CellQuery padded_cell =
      make_cell_query(p1853.projection_token_count, 4'096U);
  CellQuery disabled_cell = real_cell;
  disabled_cell.admission_enabled = false;
  CellQuery k64_cell = real_cell;
  k64_cell.inventory_consumer = k64;
  CellQuery logical_tail_cell = padded_cell;
  logical_tail_cell.projection_token_count = p1853.logical_token_count;
  CellQuery gate_near_miss = real_cell;
  gate_near_miss.gate_output_size -= 128U;
  CellQuery up_near_miss = real_cell;
  up_near_miss.up_input_size -= 128U;
  test.expect(
      detail::use_a4w4_gateup_complete_cell_v2_route(real_cell) &&
          detail::use_a4w4_gateup_complete_cell_v2_route(padded_cell) &&
          !detail::use_a4w4_gateup_complete_cell_v2_route(disabled_cell) &&
          !detail::use_a4w4_gateup_complete_cell_v2_route(k64_cell) &&
          !detail::use_a4w4_gateup_complete_cell_v2_route(
              logical_tail_cell) &&
          !detail::use_a4w4_gateup_complete_cell_v2_route(gate_near_miss) &&
          !detail::use_a4w4_gateup_complete_cell_v2_route(up_near_miss),
      "complete-cell v2 selector requires its independent gate, authenticated "
      "K128, exact Gate/Up shapes, and the internally padded M128");

  std::array<CellQuery, 8U> short_capacity_queries{};
  short_capacity_queries.fill(real_cell);
  --short_capacity_queries[0U].packed_input_capacity_bytes;
  --short_capacity_queries[1U].input_scale_capacity_elements;
  --short_capacity_queries[2U].gate_weight_capacity_bytes;
  --short_capacity_queries[3U].gate_scale_capacity_elements;
  --short_capacity_queries[4U].up_weight_capacity_bytes;
  --short_capacity_queries[5U].up_scale_capacity_elements;
  --short_capacity_queries[6U].packed_output_capacity_bytes;
  --short_capacity_queries[7U].output_scale_capacity_elements;
  const bool all_short_capacities_rejected = std::all_of(
      short_capacity_queries.begin(), short_capacity_queries.end(),
      [](const CellQuery& query) {
        return !detail::use_a4w4_gateup_complete_cell_v2_route(query);
      });
  test.expect(all_short_capacities_rejected,
              "complete-cell v2 selector rejects every independently short "
              "input/weight/Down-publication capacity");

  using V3Query = detail::A4W4GateUpProjectionV3RouteQuery;
  const auto make_v3_query = [k128](
                                 const std::size_t projection_tokens,
                                 const std::size_t workspace_tokens) {
    V3Query query;
    query.admission_enabled = true;
    query.inventory_consumer = k128;
    query.projection_token_count = projection_tokens;
    query.gate_output_size = runtime::kReferenceIntermediateSize;
    query.gate_input_size = runtime::kReferenceHiddenSize;
    query.up_output_size = runtime::kReferenceIntermediateSize;
    query.up_input_size = runtime::kReferenceHiddenSize;
    query.packed_input_capacity_bytes =
        workspace_tokens * runtime::kReferenceHiddenSize / 2U;
    query.input_scale_capacity_elements =
        workspace_tokens * runtime::kReferenceHiddenSize / 128U;
    query.gate_weight_capacity_bytes =
        runtime::kReferenceIntermediateSize *
        runtime::kReferenceHiddenSize / 2U;
    query.gate_scale_capacity_elements =
        runtime::kReferenceIntermediateSize *
        runtime::kReferenceHiddenSize / 128U;
    query.up_weight_capacity_bytes = query.gate_weight_capacity_bytes;
    query.up_scale_capacity_elements = query.gate_scale_capacity_elements;
    query.packed_output_capacity_bytes =
        workspace_tokens * runtime::kReferenceIntermediateSize / 2U;
    query.output_scale_capacity_elements =
        workspace_tokens * runtime::kReferenceIntermediateSize / 128U;
    return query;
  };
  const V3Query real_v3 = make_v3_query(2'048U, 2'048U);
  const V3Query padded_v3 =
      make_v3_query(p1853.projection_token_count, 4'096U);
  V3Query disabled_v3 = real_v3;
  disabled_v3.admission_enabled = false;
  V3Query k64_v3 = real_v3;
  k64_v3.inventory_consumer = k64;
  V3Query logical_tail_v3 = padded_v3;
  logical_tail_v3.projection_token_count = p1853.logical_token_count;
  V3Query gate_near_miss_v3 = real_v3;
  --gate_near_miss_v3.gate_output_size;
  V3Query up_near_miss_v3 = real_v3;
  --up_near_miss_v3.up_input_size;
  test.expect(
      detail::use_a4w4_gateup_projection_v3_route(real_v3) &&
          detail::use_a4w4_gateup_projection_v3_route(padded_v3) &&
          !detail::use_a4w4_gateup_projection_v3_route(disabled_v3) &&
          !detail::use_a4w4_gateup_projection_v3_route(k64_v3) &&
          !detail::use_a4w4_gateup_projection_v3_route(logical_tail_v3) &&
          !detail::use_a4w4_gateup_projection_v3_route(
              gate_near_miss_v3) &&
          !detail::use_a4w4_gateup_projection_v3_route(up_near_miss_v3),
      "Gate+Up projection v3 requires its independent gate, authenticated "
      "K128, exact model shapes, and the internal ceil128 projection M");

  std::array<V3Query, 8U> short_v3_capacities{};
  short_v3_capacities.fill(real_v3);
  --short_v3_capacities[0U].packed_input_capacity_bytes;
  --short_v3_capacities[1U].input_scale_capacity_elements;
  --short_v3_capacities[2U].gate_weight_capacity_bytes;
  --short_v3_capacities[3U].gate_scale_capacity_elements;
  --short_v3_capacities[4U].up_weight_capacity_bytes;
  --short_v3_capacities[5U].up_scale_capacity_elements;
  --short_v3_capacities[6U].packed_output_capacity_bytes;
  --short_v3_capacities[7U].output_scale_capacity_elements;
  test.expect(
      std::all_of(
          short_v3_capacities.begin(), short_v3_capacities.end(),
          [](const V3Query& query) {
            return !detail::use_a4w4_gateup_projection_v3_route(query);
          }),
      "Gate+Up projection v3 rejects every independently short "
      "input/weight/Down-publication capacity");

  using GateUpRoute = detail::A4W4K128GateUpPrefillRoute;
  test.expect(
      detail::select_a4w4_k128_gateup_prefill_route(
          real_v3, real_cell, true) == GateUpRoute::kProjectionV3 &&
          detail::select_a4w4_k128_gateup_prefill_route(
              disabled_v3, real_cell, true) ==
              GateUpRoute::kCompleteCellV2 &&
          detail::select_a4w4_k128_gateup_prefill_route(
              disabled_v3, disabled_cell, true) ==
              GateUpRoute::kRejectedM128StageMajor &&
          detail::select_a4w4_k128_gateup_prefill_route(
              disabled_v3, disabled_cell, false) ==
              GateUpRoute::kBaseline,
      "Gate+Up v3, complete-cell v2, and rejected M128 retain strict "
      "independent priority without enabling one another");

  using AttentionFamily = detail::A4W4AttentionSupermatrixFamily;
  using AttentionQuery = detail::A4W4AttentionSupermatrixRouteQuery;
  const auto make_attention_query = [k128](
                                        const AttentionFamily family,
                                        const std::size_t projection_tokens,
                                        const std::size_t workspace_tokens) {
    AttentionQuery query;
    query.admission_enabled = true;
    query.inventory_consumer = k128;
    query.family = family;
    query.projection_token_count = projection_tokens;
    const std::size_t input_size =
        family == AttentionFamily::kOutput ? 6'144U
                                           : runtime::kReferenceHiddenSize;
    query.packed_input_capacity_bytes = workspace_tokens * input_size / 2U;
    query.input_scale_capacity_elements =
        workspace_tokens * input_size / 128U;
    std::array<std::size_t, 3U> outputs{};
    std::size_t count = 0U;
    if (family == AttentionFamily::kLinearInput) {
      outputs = {10'240U, 6'144U, 0U};
      count = 2U;
    } else if (family == AttentionFamily::kFullInput) {
      outputs = {12'288U, 1'024U, 1'024U};
      count = 3U;
    } else {
      outputs = {runtime::kReferenceHiddenSize, 0U, 0U};
      count = 1U;
    }
    for (std::size_t index = 0U; index < count; ++index) {
      auto& plane = query.projections[index];
      plane.output_size = outputs[index];
      plane.input_size = input_size;
      plane.weight_capacity_bytes = outputs[index] * input_size / 2U;
      plane.weight_scale_capacity_elements =
          outputs[index] * input_size / 128U;
      plane.output_row_stride_elements = outputs[index];
      plane.output_capacity_elements = workspace_tokens * outputs[index];
    }
    return query;
  };
  const AttentionQuery linear_attention = make_attention_query(
      AttentionFamily::kLinearInput, 2'048U, 2'048U);
  const AttentionQuery full_attention = make_attention_query(
      AttentionFamily::kFullInput, 2'048U, 2'048U);
  const AttentionQuery output_attention = make_attention_query(
      AttentionFamily::kOutput, 2'048U, 2'048U);
  const AttentionQuery natural_p1853_linear_attention = make_attention_query(
      AttentionFamily::kLinearInput, p1853.projection_token_count, 4'096U);
  const AttentionQuery natural_p3987_output_attention = make_attention_query(
      AttentionFamily::kOutput, p3987.projection_token_count, 4'096U);
  const auto make_attention_k256_query =
      [make_attention_query, k256](const AttentionFamily family,
                                   const std::size_t projection_tokens,
                                   const std::size_t workspace_tokens) {
        AttentionQuery query = make_attention_query(
            family, projection_tokens, workspace_tokens);
        query.inventory_consumer = k256;
        query.input_scale_capacity_elements /= 2U;
        for (auto& plane : query.projections) {
          plane.weight_scale_capacity_elements /= 2U;
        }
        return query;
      };
  const AttentionQuery linear_attention_k256 = make_attention_k256_query(
      AttentionFamily::kLinearInput, 2'048U, 2'048U);
  const AttentionQuery full_attention_k256 = make_attention_k256_query(
      AttentionFamily::kFullInput, 2'048U, 2'048U);
  const AttentionQuery output_attention_k256 = make_attention_k256_query(
      AttentionFamily::kOutput, p1853_k256.projection_token_count, 4'096U);
  AttentionQuery disabled_attention = linear_attention;
  disabled_attention.admission_enabled = false;
  AttentionQuery k64_attention = linear_attention;
  k64_attention.inventory_consumer = k64;
  AttentionQuery m64_attention = make_attention_query(
      AttentionFamily::kLinearInput, 64U, 128U);
  AttentionQuery linear_shape_miss = linear_attention;
  --linear_shape_miss.projections[1U].output_size;
  AttentionQuery full_shape_miss = full_attention;
  --full_shape_miss.projections[2U].input_size;
  AttentionQuery output_stride_miss = output_attention;
  --output_stride_miss.projections[0U].output_row_stride_elements;
  AttentionQuery dirty_unused_plane = output_attention;
  dirty_unused_plane.projections[1U].output_size = 64U;
  AttentionQuery invalid_family = linear_attention;
  invalid_family.family = static_cast<AttentionFamily>(255U);
  AttentionQuery wrapped_device_rows = linear_attention;
  wrapped_device_rows.projection_token_count =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) +
      2U;
  test.expect(
      detail::use_a4w4_attention_supermatrix_route(linear_attention) &&
          detail::use_a4w4_attention_supermatrix_route(full_attention) &&
          detail::use_a4w4_attention_supermatrix_route(output_attention) &&
          detail::use_a4w4_attention_supermatrix_route(
              natural_p1853_linear_attention) &&
          detail::use_a4w4_attention_supermatrix_route(
              natural_p3987_output_attention) &&
          !detail::use_a4w4_attention_supermatrix_route(disabled_attention) &&
          !detail::use_a4w4_attention_supermatrix_route(k64_attention) &&
          !detail::use_a4w4_attention_supermatrix_route(m64_attention) &&
          !detail::use_a4w4_attention_supermatrix_route(linear_shape_miss) &&
          !detail::use_a4w4_attention_supermatrix_route(full_shape_miss) &&
          !detail::use_a4w4_attention_supermatrix_route(output_stride_miss) &&
          !detail::use_a4w4_attention_supermatrix_route(dirty_unused_plane) &&
          !detail::use_a4w4_attention_supermatrix_route(invalid_family) &&
          !detail::use_a4w4_attention_supermatrix_route(wrapped_device_rows),
      "Attention supermatrix selector requires its independent gate, complete "
      "authenticated K128 M128 including natural padded prompts, fixed "
      "Qwen3.6 family shapes, clean unused planes, and device-row-safe "
      "coordinates");
  AttentionQuery short_k256_scale = linear_attention_k256;
  --short_k256_scale.projections[0U].weight_scale_capacity_elements;
  test.expect(
      detail::use_a4w4_attention_k256_m128n256_route(
          linear_attention_k256) &&
          detail::use_a4w4_attention_k256_m128n256_route(
              full_attention_k256) &&
          detail::use_a4w4_attention_k256_m128n256_route(
              output_attention_k256) &&
          !detail::use_a4w4_attention_k256_m128n256_route(
              linear_attention) &&
          !detail::use_a4w4_attention_supermatrix_route(
              linear_attention_k256) &&
          !detail::use_a4w4_attention_k256_m128n256_route(
              short_k256_scale),
      "K256 Attention selector is ABI-exclusive and checks K256 scale capacity");

  std::array<AttentionQuery, 8U> short_linear_attention{};
  short_linear_attention.fill(linear_attention);
  --short_linear_attention[0U].packed_input_capacity_bytes;
  --short_linear_attention[1U].input_scale_capacity_elements;
  --short_linear_attention[2U].projections[0U].weight_capacity_bytes;
  --short_linear_attention[3U]
        .projections[0U]
        .weight_scale_capacity_elements;
  --short_linear_attention[4U].projections[0U].output_capacity_elements;
  --short_linear_attention[5U].projections[1U].weight_capacity_bytes;
  --short_linear_attention[6U]
        .projections[1U]
        .weight_scale_capacity_elements;
  --short_linear_attention[7U].projections[1U].output_capacity_elements;
  std::array<AttentionQuery, 11U> short_full_attention{};
  short_full_attention.fill(full_attention);
  --short_full_attention[0U].packed_input_capacity_bytes;
  --short_full_attention[1U].input_scale_capacity_elements;
  for (std::size_t projection = 0U; projection < 3U; ++projection) {
    --short_full_attention[2U + projection * 3U]
          .projections[projection]
          .weight_capacity_bytes;
    --short_full_attention[3U + projection * 3U]
          .projections[projection]
          .weight_scale_capacity_elements;
    --short_full_attention[4U + projection * 3U]
          .projections[projection]
          .output_capacity_elements;
  }
  std::array<AttentionQuery, 5U> short_output_attention{};
  short_output_attention.fill(output_attention);
  --short_output_attention[0U].packed_input_capacity_bytes;
  --short_output_attention[1U].input_scale_capacity_elements;
  --short_output_attention[2U].projections[0U].weight_capacity_bytes;
  --short_output_attention[3U]
        .projections[0U]
        .weight_scale_capacity_elements;
  --short_output_attention[4U].projections[0U].output_capacity_elements;
  const auto rejects_attention_query = [](const AttentionQuery& query) {
    return !detail::use_a4w4_attention_supermatrix_route(query);
  };
  test.expect(
      std::all_of(short_linear_attention.begin(),
                  short_linear_attention.end(), rejects_attention_query) &&
          std::all_of(short_full_attention.begin(), short_full_attention.end(),
                      rejects_attention_query) &&
          std::all_of(short_output_attention.begin(),
                      short_output_attention.end(), rejects_attention_query),
      "Attention supermatrix selector rejects every independently short A, "
      "weight, scale, and BF16 output capacity for Linear, Full, and O");

  using DownCellQuery = detail::A4W4DownCompleteCellV2RouteQuery;
  const auto make_down_cell_query = [k128](
                                          const std::size_t projection_tokens) {
    DownCellQuery query;
    query.admission_enabled = true;
    query.inventory_consumer = k128;
    query.projection_token_count = projection_tokens;
    query.output_size = runtime::kReferenceHiddenSize;
    query.input_size = runtime::kReferenceIntermediateSize;
    query.packed_input_capacity_bytes =
        projection_tokens * runtime::kReferenceIntermediateSize / 2U;
    query.input_scale_capacity_elements =
        projection_tokens * runtime::kReferenceIntermediateSize / 128U;
    query.weight_capacity_bytes =
        runtime::kReferenceHiddenSize *
        runtime::kReferenceIntermediateSize / 2U;
    query.weight_scale_capacity_elements =
        runtime::kReferenceHiddenSize *
        runtime::kReferenceIntermediateSize / 128U;
    query.output_capacity_elements =
        projection_tokens * runtime::kReferenceHiddenSize;
    return query;
  };
  const DownCellQuery real_down_cell = make_down_cell_query(2'048U);
  detail::A4W4DownCompleteCellV3RouteQuery real_down_cell_v3 =
      real_down_cell;
  detail::A4W4DownCompleteCellV3RouteQuery disabled_down_cell_v3 =
      real_down_cell_v3;
  disabled_down_cell_v3.admission_enabled = false;
  DownCellQuery disabled_down_cell = real_down_cell;
  disabled_down_cell.admission_enabled = false;
  DownCellQuery k64_down_cell = real_down_cell;
  k64_down_cell.inventory_consumer = k64;
  DownCellQuery natural_p1853_down_cell = make_down_cell_query(
      p1853.projection_token_count);
  DownCellQuery explicit_m64_not_m128 = make_down_cell_query(1'856U);
  DownCellQuery wrong_down_n = real_down_cell;
  --wrong_down_n.output_size;
  DownCellQuery wrong_down_k = real_down_cell;
  wrong_down_k.input_size -= 128U;
  test.expect(
      detail::use_a4w4_down_complete_cell_v2_route(real_down_cell) &&
          detail::use_a4w4_down_complete_cell_v2_route(
              natural_p1853_down_cell) &&
          !detail::use_a4w4_down_complete_cell_v2_route(
              disabled_down_cell) &&
          !detail::use_a4w4_down_complete_cell_v2_route(k64_down_cell) &&
          !detail::use_a4w4_down_complete_cell_v2_route(
              explicit_m64_not_m128) &&
          !detail::use_a4w4_down_complete_cell_v2_route(wrong_down_n) &&
          !detail::use_a4w4_down_complete_cell_v2_route(wrong_down_k),
      "Down complete-cell v2 requires its independent gate, authenticated "
      "K128, exact N5120/K17408, and a complete internal M128 after ceil128");
  test.expect(
      detail::use_a4w4_down_complete_cell_v3_route(real_down_cell_v3) &&
          !detail::use_a4w4_down_complete_cell_v3_route(
              disabled_down_cell_v3),
      "Down complete-cell v3 retains the authenticated v2 capacity contract "
      "behind an independent admission bit");

  std::array<DownCellQuery, 5U> short_down_capacities{};
  short_down_capacities.fill(real_down_cell);
  --short_down_capacities[0U].packed_input_capacity_bytes;
  --short_down_capacities[1U].input_scale_capacity_elements;
  --short_down_capacities[2U].weight_capacity_bytes;
  --short_down_capacities[3U].weight_scale_capacity_elements;
  --short_down_capacities[4U].output_capacity_elements;
  test.expect(
      std::all_of(
          short_down_capacities.begin(), short_down_capacities.end(),
          [](const DownCellQuery& query) {
            return !detail::use_a4w4_down_complete_cell_v2_route(query);
          }),
      "Down complete-cell v2 rejects each independently short A4 input, "
      "weight, scale, and BF16 output capacity");

  using DownRoute = detail::A4W4K128DownPrefillRoute;
  test.expect(
      detail::select_a4w4_k128_down_prefill_route(
          real_down_cell_v3, real_down_cell, true) ==
              DownRoute::kCompleteCellV3 &&
          detail::select_a4w4_k128_down_prefill_route(
              disabled_down_cell_v3, real_down_cell, true) ==
              DownRoute::kCompleteCellV2 &&
          detail::select_a4w4_k128_down_prefill_route(
              disabled_down_cell, true) ==
              DownRoute::kRejectedM128StageMajor &&
          detail::select_a4w4_k128_down_prefill_route(
              disabled_down_cell, false) == DownRoute::kBaseline,
      "Down complete-cell precedence is v3, v2, then the independently "
      "gated rejected M128 diagnostic route");

  test.expect(
      !detail::a4w4_m128_stage_major_common_route(false, k128, 2'048U) &&
          !detail::a4w4_m128_stage_major_common_route(true, k64, 2'048U) &&
          !detail::a4w4_m128_stage_major_common_route(
              true, Consumer::kUnavailable, 2'048U) &&
          !detail::a4w4_m128_stage_major_common_route(true, k128, 0U) &&
          !detail::a4w4_m128_stage_major_common_route(true, k128, 64U) &&
          detail::a4w4_m128_stage_major_common_route(true, k128, 128U) &&
          detail::a4w4_m128_stage_major_common_route(true, k128, 2'048U) &&
          !detail::a4w4_m128_stage_major_common_route(
              true, k128, 40'000U) &&
          detail::a4w4_m128_stage_major_common_route(
              true, k128, 40'960U),
      "M128 selector requires explicit admission, uniform K128 inventory, "
      "and complete M128 spans");
  using GenericRoute = detail::A4W4K128GenericPrefillRoute;
  test.expect(
      detail::select_a4w4_k128_generic_prefill_route(
          false, true, k128, 2'048U, runtime::kReferenceHiddenSize,
          runtime::kReferenceIntermediateSize) ==
              GenericRoute::kDownM128StageMajor &&
          detail::select_a4w4_k128_generic_prefill_route(
              true, false, k128, 2'048U, runtime::kReferenceHiddenSize,
              runtime::kReferenceIntermediateSize) ==
              GenericRoute::kBaseline &&
          detail::select_a4w4_k128_generic_prefill_route(
              false, false, k128, 2'048U, runtime::kReferenceHiddenSize,
              runtime::kReferenceIntermediateSize) ==
              GenericRoute::kBaseline &&
          detail::select_a4w4_k128_generic_prefill_route(
              false, true, k64, 2'048U, runtime::kReferenceHiddenSize,
              runtime::kReferenceIntermediateSize) ==
              GenericRoute::kBaseline &&
          detail::select_a4w4_k128_generic_prefill_route(
              true, false, k128, 2'048U, 10'240U,
              runtime::kReferenceHiddenSize) ==
              GenericRoute::kM128StageMajor &&
          detail::select_a4w4_k128_generic_prefill_route(
              false, true, k128, 2'048U, 10'240U,
              runtime::kReferenceHiddenSize) == GenericRoute::kBaseline &&
          detail::select_a4w4_k128_generic_prefill_route(
              true, false, k64, 2'048U, 10'240U,
              runtime::kReferenceHiddenSize) == GenericRoute::kBaseline &&
          detail::select_a4w4_k128_generic_prefill_route(
              true, true, k128, 2'048U, 1'024U, 5'121U) ==
              GenericRoute::kBaseline &&
          detail::select_a4w4_k128_generic_prefill_route(
              false, true, k128, 64U, runtime::kReferenceHiddenSize,
              runtime::kReferenceIntermediateSize) ==
              GenericRoute::kBaseline &&
          detail::select_a4w4_k128_generic_prefill_route(
              true, false, k128, 2'048U, runtime::kReferenceHiddenSize,
              runtime::kReferenceIntermediateSize - 128U) ==
              GenericRoute::kM128StageMajor &&
          detail::select_a4w4_k128_generic_prefill_route(
              true, true, k128, p1853.projection_token_count, 10'240U,
              runtime::kReferenceHiddenSize) ==
              GenericRoute::kM128StageMajor &&
          detail::select_a4w4_k128_generic_prefill_route(
              true, true, k128, p3987.projection_token_count, 10'240U,
              runtime::kReferenceHiddenSize) ==
              GenericRoute::kM128StageMajor &&
          detail::select_a4w4_k128_generic_prefill_route(
              true, false, k128, p40959_tail.projection_token_count,
              10'240U, runtime::kReferenceHiddenSize) ==
              GenericRoute::kM128StageMajor,
      "shape-aware M128 selector gives exact Down an independent gate, "
      "admits natural prompts after ceil128 padding, and rejects invalid N/K "
      "shapes");

  const char* const v3_environment = std::getenv(
      "Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION");
  const bool v3_environment_requested =
      v3_environment != nullptr && std::strcmp(v3_environment, "1") == 0 &&
      !runtime::optimized_prefill_dispatch_disabled();
  const bool initial_v3_enabled =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
          false);
  (void)detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
      true);
  const bool v3_admission_is_compiled =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
          false);
  test.expect(
      initial_v3_enabled ==
          (v3_admission_is_compiled && v3_environment_requested),
      "Gate+Up projection v3 environment gate is exact-value, default-off, "
      "and independent from v2 and rejected M128");

  const char* const cell_environment = std::getenv(
      "Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION");
  const bool cell_environment_requested =
      cell_environment != nullptr &&
      std::strcmp(cell_environment, "1") == 0 &&
      !runtime::optimized_prefill_dispatch_disabled();
  const bool initial_cell_enabled =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
          false);
  (void)detail::
      exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(true);
  const bool cell_admission_is_compiled =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
          false);
  test.expect(
      initial_cell_enabled ==
          (cell_admission_is_compiled && cell_environment_requested),
      "complete-cell v2 environment gate is exact-value, default-off, and "
      "independent of code availability");

  const char* const down_cell_environment = std::getenv(
      "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION");
  const bool down_cell_environment_requested =
      down_cell_environment != nullptr &&
      std::strcmp(down_cell_environment, "1") == 0 &&
      !runtime::optimized_prefill_dispatch_disabled();
  const bool initial_down_cell_enabled =
      detail::exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
          false);
  (void)detail::exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
      true);
  const bool down_cell_admission_is_compiled =
      detail::exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
          false);
  test.expect(
      initial_down_cell_enabled ==
          (down_cell_admission_is_compiled &&
           down_cell_environment_requested),
      "Down complete-cell v2 environment gate is exact-value, default-off, "
      "and independent from the rejected Down M128 switch");

  const char* const down_cell_v3_environment = std::getenv(
      "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION");
  const bool down_cell_v3_environment_requested =
      down_cell_v3_environment != nullptr &&
      std::strcmp(down_cell_v3_environment, "1") == 0 &&
      !runtime::optimized_prefill_dispatch_disabled();
  const bool initial_down_cell_v3_enabled =
      detail::exchange_a4w4_down_complete_cell_v3_admission_test_enabled(
          false);
  (void)detail::exchange_a4w4_down_complete_cell_v3_admission_test_enabled(
      true);
  const bool down_cell_v3_admission_is_compiled =
      detail::exchange_a4w4_down_complete_cell_v3_admission_test_enabled(
          false);
  test.expect(
      initial_down_cell_v3_enabled ==
          (down_cell_v3_admission_is_compiled &&
           down_cell_v3_environment_requested),
      "Down complete-cell v3 environment gate is exact-value, default-off, "
      "global-disable aware, and independent from v2");

  const char* const attention_supermatrix_environment = std::getenv(
      "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION");
  const bool attention_supermatrix_environment_requested =
      attention_supermatrix_environment != nullptr &&
      std::strcmp(attention_supermatrix_environment, "1") == 0 &&
      !runtime::optimized_prefill_dispatch_disabled();
  const bool initial_attention_supermatrix_enabled =
      detail::exchange_a4w4_attention_supermatrix_admission_test_enabled(
          false);
  (void)detail::exchange_a4w4_attention_supermatrix_admission_test_enabled(
      true);
  const bool attention_supermatrix_admission_is_compiled =
      detail::exchange_a4w4_attention_supermatrix_admission_test_enabled(
          false);
  test.expect(
      initial_attention_supermatrix_enabled ==
          (attention_supermatrix_admission_is_compiled &&
           attention_supermatrix_environment_requested),
      "Attention supermatrix environment gate is exact-value, default-off, "
      "global-disable aware, and independent from code availability");

  const bool prior_enabled =
      detail::exchange_a4w4_full_prefill_admission_test_enabled(true);
  const bool admission_is_compiled =
      detail::exchange_a4w4_full_prefill_admission_test_enabled(
          prior_enabled);
  const bool prior_m128_enabled =
      detail::exchange_a4w4_m128_stage_major_admission_test_enabled(true);
  const bool m128_admission_is_compiled =
      detail::exchange_a4w4_m128_stage_major_admission_test_enabled(
          prior_m128_enabled);
  const bool prior_down_m128_enabled =
      detail::exchange_a4w4_down_m128_stage_major_admission_test_enabled(true);
  const bool down_m128_admission_is_compiled =
      detail::exchange_a4w4_down_m128_stage_major_admission_test_enabled(
          prior_down_m128_enabled);
  const detail::A4W4FullPrefillAdmissionHits fixture{
      192U, 272U, 64U, 400U, 1U, 208U, 64U, 64U};
  const detail::A4W4FullPrefillAdmissionHits prior_hits =
      detail::exchange_a4w4_full_prefill_admission_test_hits(fixture);
  const detail::A4W4FullPrefillAdmissionHits observed =
      detail::exchange_a4w4_full_prefill_admission_test_hits(prior_hits);
  const std::size_t prior_cell_hits =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_hits(73U);
  const std::size_t observed_cell_hits =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_hits(
          prior_cell_hits);
  const std::size_t prior_v3_hits =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_hits(83U);
  const std::size_t observed_v3_hits =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_hits(
          prior_v3_hits);
  const std::size_t prior_down_cell_hits =
      detail::exchange_a4w4_down_complete_cell_v2_admission_test_hits(41U);
  const std::size_t observed_down_cell_hits =
      detail::exchange_a4w4_down_complete_cell_v2_admission_test_hits(
          prior_down_cell_hits);
  const std::size_t prior_down_cell_v3_hits =
      detail::exchange_a4w4_down_complete_cell_v3_admission_test_hits(43U);
  const std::size_t observed_down_cell_v3_hits =
      detail::exchange_a4w4_down_complete_cell_v3_admission_test_hits(
          prior_down_cell_v3_hits);
  const detail::A4W4AttentionSupermatrixAdmissionHits attention_fixture{
      48U, 16U, 64U, 208U};
  const detail::A4W4AttentionSupermatrixAdmissionHits
      prior_attention_supermatrix_hits =
          detail::exchange_a4w4_attention_supermatrix_admission_test_hits(
              attention_fixture);
  const detail::A4W4AttentionSupermatrixAdmissionHits
      observed_attention_supermatrix_hits =
          detail::exchange_a4w4_attention_supermatrix_admission_test_hits(
              prior_attention_supermatrix_hits);
  if (admission_is_compiled) {
    test.expect(observed.activation_quantize_hits == 192U &&
                    observed.generic_projection_hits == 272U &&
                    observed.paired_gate_up_hits == 64U &&
                    observed.logical_projection_hits == 400U &&
                    observed.complete_model_tile_hits == 1U &&
                    observed.m128_stage_major_generic_projection_hits ==
                        208U &&
                    observed.m128_stage_major_down_projection_hits == 64U &&
                    observed.m128_stage_major_paired_gate_up_hits == 64U &&
                    m128_admission_is_compiled &&
                    down_m128_admission_is_compiled &&
                    cell_admission_is_compiled && observed_cell_hits == 73U,
                "compiled A4W4 admission preserves all route counters");
  } else {
    test.expect(observed.activation_quantize_hits == 0U &&
                    observed.generic_projection_hits == 0U &&
                    observed.paired_gate_up_hits == 0U &&
                    observed.logical_projection_hits == 0U &&
                    observed.complete_model_tile_hits == 0U &&
                    observed.m128_stage_major_generic_projection_hits == 0U &&
                    observed.m128_stage_major_down_projection_hits == 0U &&
                    observed.m128_stage_major_paired_gate_up_hits == 0U &&
                    !m128_admission_is_compiled &&
                    !down_m128_admission_is_compiled &&
                    !cell_admission_is_compiled &&
                    observed_cell_hits == 0U,
                "ordinary build exposes inert A4W4 admission controls");
  }
  test.expect(
      observed_down_cell_hits ==
          (down_cell_admission_is_compiled ? 41U : 0U),
      "Down complete-cell v2 exposes dedicated active or inert accounting "
      "independently from full-A4 code availability");
  test.expect(
      observed_down_cell_v3_hits ==
          (down_cell_v3_admission_is_compiled ? 43U : 0U),
      "Down complete-cell v3 exposes dedicated active or inert success-only "
      "accounting independently from v2");
  test.expect(observed_v3_hits == (v3_admission_is_compiled ? 83U : 0U),
              "Gate+Up projection v3 exposes dedicated active or inert hit "
              "accounting independently from v2 and full-A4 counters");
  test.expect(
      observed_attention_supermatrix_hits.linear_input_launch_hits ==
              (attention_supermatrix_admission_is_compiled ? 48U : 0U) &&
          observed_attention_supermatrix_hits.full_input_launch_hits ==
              (attention_supermatrix_admission_is_compiled ? 16U : 0U) &&
          observed_attention_supermatrix_hits.output_launch_hits ==
              (attention_supermatrix_admission_is_compiled ? 64U : 0U) &&
          observed_attention_supermatrix_hits.logical_projection_hits ==
              (attention_supermatrix_admission_is_compiled ? 208U : 0U),
      "Attention supermatrix exposes independent active or inert success-only "
      "accounting");
  (void)detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
      initial_v3_enabled);
  (void)detail::
      exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
          initial_cell_enabled);
  (void)detail::exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
      initial_down_cell_enabled);
  (void)detail::exchange_a4w4_down_complete_cell_v3_admission_test_enabled(
      initial_down_cell_v3_enabled);
  (void)detail::exchange_a4w4_attention_supermatrix_admission_test_enabled(
      initial_attention_supermatrix_enabled);
}

void test_prefill_admission_gate_orthogonality(TestContext& test) {
  // Discover availability by round-tripping each worker-local switch, then
  // leave both disabled for the cross-gate checks. Ordinary builds expose
  // inert BF16 controls while the production A4 build still exposes M128.
  const bool initial_bf16 =
      detail::exchange_bf16_ab_large_m_prefill_admission_test_enabled(false);
  const bool initial_m128 =
      detail::exchange_a4w4_m128_stage_major_admission_test_enabled(false);
  const bool initial_cell =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
          false);
  const bool initial_v3 =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
          false);
  const bool initial_down_cell =
      detail::exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
          false);
  const bool initial_down_m128 =
      detail::exchange_a4w4_down_m128_stage_major_admission_test_enabled(
          false);
  const bool initial_attention_supermatrix =
      detail::exchange_a4w4_attention_supermatrix_admission_test_enabled(
          false);
  (void)detail::exchange_bf16_ab_large_m_prefill_admission_test_enabled(true);
  const bool bf16_compiled =
      detail::exchange_bf16_ab_large_m_prefill_admission_test_enabled(false);
  (void)detail::exchange_a4w4_m128_stage_major_admission_test_enabled(true);
  const bool m128_compiled =
      detail::exchange_a4w4_m128_stage_major_admission_test_enabled(false);
  (void)detail::
      exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(true);
  const bool cell_compiled =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
          false);
  (void)detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
      true);
  const bool v3_compiled =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
          false);
  (void)detail::exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
      true);
  const bool down_cell_compiled =
      detail::exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
          false);
  (void)detail::exchange_a4w4_down_m128_stage_major_admission_test_enabled(
      true);
  const bool down_m128_compiled =
      detail::exchange_a4w4_down_m128_stage_major_admission_test_enabled(
          false);
  (void)detail::exchange_a4w4_attention_supermatrix_admission_test_enabled(
      true);
  const bool attention_supermatrix_compiled =
      detail::exchange_a4w4_attention_supermatrix_admission_test_enabled(
          false);

  (void)detail::exchange_bf16_ab_large_m_prefill_admission_test_enabled(true);
  const bool m128_after_bf16 =
      detail::exchange_a4w4_m128_stage_major_admission_test_enabled(false);
  const bool bf16_after_bf16 =
      detail::exchange_bf16_ab_large_m_prefill_admission_test_enabled(false);
  test.expect(!m128_after_bf16 && bf16_after_bf16 == bf16_compiled,
              "BF16 whole-span gate does not enable the M128 A4 gate");

  (void)detail::exchange_a4w4_m128_stage_major_admission_test_enabled(true);
  const bool bf16_after_m128 =
      detail::exchange_bf16_ab_large_m_prefill_admission_test_enabled(false);
  const bool m128_after_m128 =
      detail::exchange_a4w4_m128_stage_major_admission_test_enabled(false);
  test.expect(!bf16_after_m128 && m128_after_m128 == m128_compiled,
              "M128 A4 gate does not enable the BF16 whole-span gate");

  (void)detail::
      exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(true);
  const bool m128_after_cell =
      detail::exchange_a4w4_m128_stage_major_admission_test_enabled(false);
  const bool cell_after_cell =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
          false);
  test.expect(!m128_after_cell && cell_after_cell == cell_compiled,
              "complete-cell v2 gate does not enable the rejected M128 gate");

  (void)detail::exchange_a4w4_m128_stage_major_admission_test_enabled(true);
  const bool cell_after_m128 =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
          false);
  const bool m128_after_cell_check =
      detail::exchange_a4w4_m128_stage_major_admission_test_enabled(false);
  test.expect(!cell_after_m128 && m128_after_cell_check == m128_compiled,
              "M128 A4 gate does not enable the complete-cell v2 gate");

  (void)detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
      true);
  const bool cell_after_v3 =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
          false);
  const bool m128_after_v3 =
      detail::exchange_a4w4_m128_stage_major_admission_test_enabled(false);
  const bool v3_after_v3 =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
          false);
  test.expect(!cell_after_v3 && !m128_after_v3 &&
                  v3_after_v3 == v3_compiled,
              "Gate+Up projection v3 gate does not enable v2 or rejected "
              "M128");

  (void)detail::
      exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(true);
  const bool v3_after_cell =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
          false);
  const bool cell_after_v3_check =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
          false);
  test.expect(!v3_after_cell && cell_after_v3_check == cell_compiled,
              "complete-cell v2 gate does not enable Gate+Up projection v3");

  (void)detail::exchange_a4w4_attention_supermatrix_admission_test_enabled(
      true);
  const bool m128_after_attention_supermatrix =
      detail::exchange_a4w4_m128_stage_major_admission_test_enabled(false);
  const bool v3_after_attention_supermatrix =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
          false);
  const bool attention_supermatrix_after_attention_supermatrix =
      detail::exchange_a4w4_attention_supermatrix_admission_test_enabled(
          false);
  test.expect(
      !m128_after_attention_supermatrix && !v3_after_attention_supermatrix &&
          attention_supermatrix_after_attention_supermatrix ==
              attention_supermatrix_compiled,
      "Attention supermatrix gate does not enable the M128 or Gate+Up v3 "
      "experiments");

  (void)detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
      true);
  const bool attention_supermatrix_after_v3 =
      detail::exchange_a4w4_attention_supermatrix_admission_test_enabled(
          false);
  const bool v3_after_attention_supermatrix_check =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
          false);
  test.expect(
      !attention_supermatrix_after_v3 &&
          v3_after_attention_supermatrix_check == v3_compiled,
      "Gate+Up projection v3 does not enable Attention supermatrix");

  (void)detail::exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
      true);
  const bool down_m128_after_down_cell =
      detail::exchange_a4w4_down_m128_stage_major_admission_test_enabled(
          false);
  const bool down_cell_after_down_cell =
      detail::exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
          false);
  test.expect(
      !down_m128_after_down_cell &&
          down_cell_after_down_cell == down_cell_compiled,
      "Down complete-cell v2 gate does not enable rejected Down M128");

  (void)detail::exchange_a4w4_down_m128_stage_major_admission_test_enabled(
      true);
  const bool down_cell_after_down_m128 =
      detail::exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
          false);
  const bool down_m128_after_down_m128 =
      detail::exchange_a4w4_down_m128_stage_major_admission_test_enabled(
          false);
  test.expect(
      !down_cell_after_down_m128 &&
          down_m128_after_down_m128 == down_m128_compiled,
      "rejected Down M128 gate does not enable Down complete-cell v2");

  const std::size_t initial_bf16_hits =
      detail::exchange_bf16_ab_large_m_prefill_admission_test_hits(19U);
  const detail::A4W4FullPrefillAdmissionHits a4_fixture{
      192U, 272U, 64U, 400U, 1U, 208U, 64U, 64U};
  const detail::A4W4FullPrefillAdmissionHits initial_a4_hits =
      detail::exchange_a4w4_full_prefill_admission_test_hits(a4_fixture);
  const std::size_t observed_bf16_hits =
      detail::exchange_bf16_ab_large_m_prefill_admission_test_hits(
          initial_bf16_hits);
  const std::size_t initial_cell_hits =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_hits(23U);
  const std::size_t observed_cell_hits =
      detail::exchange_a4w4_gateup_complete_cell_v2_admission_test_hits(
          initial_cell_hits);
  const std::size_t initial_v3_hits =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_hits(31U);
  const std::size_t observed_v3_hits =
      detail::exchange_a4w4_gateup_projection_v3_admission_test_hits(
          initial_v3_hits);
  const std::size_t initial_down_cell_hits =
      detail::exchange_a4w4_down_complete_cell_v2_admission_test_hits(29U);
  const std::size_t observed_down_cell_hits =
      detail::exchange_a4w4_down_complete_cell_v2_admission_test_hits(
          initial_down_cell_hits);
  const detail::A4W4FullPrefillAdmissionHits observed_a4_hits =
      detail::exchange_a4w4_full_prefill_admission_test_hits(initial_a4_hits);
  const detail::A4W4AttentionSupermatrixAdmissionHits attention_fixture{
      48U, 16U, 64U, 208U};
  const detail::A4W4AttentionSupermatrixAdmissionHits
      initial_attention_supermatrix_hits =
          detail::exchange_a4w4_attention_supermatrix_admission_test_hits(
              attention_fixture);
  const detail::A4W4AttentionSupermatrixAdmissionHits
      observed_attention_supermatrix_hits =
          detail::exchange_a4w4_attention_supermatrix_admission_test_hits(
              initial_attention_supermatrix_hits);
  test.expect(observed_bf16_hits == (bf16_compiled ? 19U : 0U),
              "BF16 whole-span hits use independent storage");
  test.expect(observed_cell_hits == (cell_compiled ? 23U : 0U),
              "complete-cell v2 hits use storage independent from M128/A4");
  test.expect(observed_v3_hits == (v3_compiled ? 31U : 0U),
              "Gate+Up projection v3 hits use storage independent from "
              "v2/M128/A4");
  test.expect(
      observed_down_cell_hits == (down_cell_compiled ? 29U : 0U),
      "Down complete-cell v2 hits use independent storage from all old gates");
  test.expect(
      observed_attention_supermatrix_hits.linear_input_launch_hits ==
              (attention_supermatrix_compiled ? 48U : 0U) &&
          observed_attention_supermatrix_hits.full_input_launch_hits ==
              (attention_supermatrix_compiled ? 16U : 0U) &&
          observed_attention_supermatrix_hits.output_launch_hits ==
              (attention_supermatrix_compiled ? 64U : 0U) &&
          observed_attention_supermatrix_hits.logical_projection_hits ==
              (attention_supermatrix_compiled ? 208U : 0U),
      "Attention supermatrix hits use storage independent from every older "
      "Prefill gate");
  test.expect(
      observed_a4_hits.activation_quantize_hits ==
              (m128_compiled ? 192U : 0U) &&
          observed_a4_hits.generic_projection_hits ==
              (m128_compiled ? 272U : 0U) &&
          observed_a4_hits.paired_gate_up_hits ==
              (m128_compiled ? 64U : 0U) &&
          observed_a4_hits.logical_projection_hits ==
              (m128_compiled ? 400U : 0U) &&
          observed_a4_hits.complete_model_tile_hits ==
              (m128_compiled ? 1U : 0U) &&
          observed_a4_hits.m128_stage_major_generic_projection_hits ==
              (m128_compiled ? 208U : 0U) &&
          observed_a4_hits.m128_stage_major_down_projection_hits ==
              (m128_compiled ? 64U : 0U) &&
          observed_a4_hits.m128_stage_major_paired_gate_up_hits ==
              (m128_compiled ? 64U : 0U),
      "A4/M128 accounting is unchanged by BF16 hit accounting");

  (void)detail::exchange_bf16_ab_large_m_prefill_admission_test_enabled(
      initial_bf16);
  (void)detail::exchange_a4w4_m128_stage_major_admission_test_enabled(
      initial_m128);
  (void)detail::
      exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
          initial_cell);
  (void)detail::exchange_a4w4_gateup_projection_v3_admission_test_enabled(
      initial_v3);
  (void)detail::exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
      initial_down_cell);
  (void)detail::exchange_a4w4_down_m128_stage_major_admission_test_enabled(
      initial_down_m128);
  (void)detail::exchange_a4w4_attention_supermatrix_admission_test_enabled(
      initial_attention_supermatrix);
}

void test_factorized_lane_r4_prefill_route(TestContext& test) {
  using Query = detail::FactorizedLaneR4PrefillRouteQuery;
  using Route = detail::FactorizedLaneR4PrefillRoute;

  Query valid;
  valid.requested = true;
  valid.full_a4_prefill_enabled = true;
  valid.sm87_weight_only = true;
  valid.k256_inventory = true;
  valid.complete_r4_mlp_attached = true;
  test.expect(detail::select_factorized_lane_r4_prefill_route(valid) ==
                  Route::kEnabled,
              "R4 whole-MLP Prefill route admits only its complete base "
              "contract");

  Query disabled = valid;
  disabled.requested = false;
  disabled.complete_r4_mlp_attached = false;
  test.expect(detail::select_factorized_lane_r4_prefill_route(disabled) ==
                  Route::kDisabled,
              "R4 whole-MLP Prefill route remains default-off");

  const auto rejects = [&](Query query) {
    return detail::select_factorized_lane_r4_prefill_route(query) ==
           Route::kInvalid;
  };
  Query conflict = valid;
  conflict.r1_requested = true;
  Query no_full_prefill = valid;
  no_full_prefill.full_a4_prefill_enabled = false;
  Query wrong_backend = valid;
  wrong_backend.sm87_weight_only = false;
  Query trace = valid;
  trace.trace_enabled = true;
  Query globally_disabled = valid;
  globally_disabled.optimized_dispatch_disabled = true;
  Query wrong_inventory = valid;
  wrong_inventory.k256_inventory = false;
  Query incomplete = valid;
  incomplete.complete_r4_mlp_attached = false;
  Query selector_conflict = valid;
  selector_conflict.mlp_selector_conflict = true;
  test.expect(rejects(conflict) && rejects(no_full_prefill) &&
                  rejects(wrong_backend) && rejects(trace) &&
                  rejects(globally_disabled) && rejects(wrong_inventory) &&
                  rejects(incomplete) && rejects(selector_conflict),
              "R4 whole-MLP Prefill route fails closed on R1, backend, "
              "trace, inventory, attachment, and selector conflicts");

  test.expect(
      !detail::factorized_lane_r4_prefill_prompt_supported(
          1'792U, 1'920U, 4'096U) &&
          detail::factorized_lane_r4_prefill_prompt_supported(1'793U,
                                                              1'920U,
                                                              4'096U) &&
          detail::factorized_lane_r4_prefill_prompt_supported(1'853U,
                                                              1'920U,
                                                              2'048U) &&
          detail::factorized_lane_r4_prefill_prompt_supported(1'920U,
                                                              1'920U,
                                                              1'920U) &&
          !detail::factorized_lane_r4_prefill_prompt_supported(1'921U,
                                                               1'920U,
                                                               4'096U) &&
          !detail::factorized_lane_r4_prefill_prompt_supported(1'853U,
                                                               2'048U,
                                                               4'096U) &&
          !detail::factorized_lane_r4_prefill_prompt_supported(
              1'853U, 1'920U, 1'536U),
      "R4 whole-MLP Prefill route is exact to logical P1793..P1920 and "
      "launch P1920 while allowing a larger aligned workspace span");

  detail::A4W4FullPrefillAdmissionHits aggregate_hits;
  aggregate_hits.factorized_lane_r1_package_launch_hits = 7U;
  aggregate_hits.factorized_lane_r4_package_launch_hits = 11U;
  runtime::ReferenceLongPrefillResult request_hits;
  request_hits.factorized_lane_r1_package_launch_hits = 13U;
  request_hits.factorized_lane_r4_package_launch_hits = 17U;
  test.expect(
      aggregate_hits.factorized_lane_r1_package_launch_hits == 7U &&
          aggregate_hits.factorized_lane_r4_package_launch_hits == 11U &&
          request_hits.factorized_lane_r1_package_launch_hits == 13U &&
          request_hits.factorized_lane_r4_package_launch_hits == 17U,
      "R1 and R4 package proofs remain independent in aggregate and "
      "request-local accounting");
}

void test_paired_gateup_canonical_down_selector_and_accounting(
    TestContext& test) {
  using Query = detail::A4W4PairedGateUpCanonicalDownSelectorQuery;
  using Route = detail::A4W4PairedGateUpCanonicalDownRoute;

  Query query;
  query.projection_span = true;
  test.expect(
      detail::select_a4w4_paired_gateup_canonical_down_route(query) ==
          Route::kDisabled,
      "paired GateUp/canonical Down route is default-off");

  query.master_requested = true;
  test.expect(
      detail::select_a4w4_paired_gateup_canonical_down_route(query) ==
          Route::kInvalid,
      "paired GateUp master cannot silently run without its Gate selector");
  query = Query{};
  query.old_gate_requested = true;
  query.projection_span = true;
  test.expect(
      detail::select_a4w4_paired_gateup_canonical_down_route(query) ==
          Route::kInvalid,
      "old paired Gate cannot run without its publication master");
  query = Query{};
  query.new_paired_warp_gate_requested = true;
  query.projection_span = true;
  test.expect(
      detail::select_a4w4_paired_gateup_canonical_down_route(query) ==
          Route::kInvalid,
      "new paired-warp Gate cannot run without its publication master");

  query = Query{};
  query.master_requested = true;
  query.old_gate_requested = true;
  query.projection_span = true;
  test.expect(
      detail::select_a4w4_paired_gateup_canonical_down_route(query) ==
          Route::kOldGateCanonicalDown,
      "master plus old paired Gate selects canonical-v1 Down");
  query.old_ldmatrix_pairring_down_requested = true;
  test.expect(
      detail::select_a4w4_paired_gateup_canonical_down_route(query) ==
          Route::kOldGateLdmatrixPairringDown,
      "old paired Gate accepts only the old LDSM pair-ring Down");

  query = Query{};
  query.master_requested = true;
  query.new_paired_warp_gate_requested = true;
  query.projection_span = true;
  test.expect(
      detail::select_a4w4_paired_gateup_canonical_down_route(query) ==
          Route::kNewPairedWarpGateCanonicalDown,
      "master plus paired-warp Gate selects canonical-v1 Down");
  query.new_16warp_pairring_down_requested = true;
  test.expect(
      detail::select_a4w4_paired_gateup_canonical_down_route(query) ==
          Route::kNewPairedWarpGate16WarpPairringDown,
      "paired-warp Gate accepts the 16-warp pair-ring Down");

  Query cross_pair = query;
  cross_pair.new_16warp_pairring_down_requested = false;
  cross_pair.old_ldmatrix_pairring_down_requested = true;
  const bool cross_pair_rejected =
      detail::select_a4w4_paired_gateup_canonical_down_route(cross_pair) ==
      Route::kInvalid;
  Query both_gates = query;
  both_gates.old_gate_requested = true;
  const bool both_gates_rejected =
      detail::select_a4w4_paired_gateup_canonical_down_route(both_gates) ==
      Route::kInvalid;

  Query conflict = query;
  conflict.legacy_mlp_requested = true;
  const bool legacy_mlp_rejected =
      detail::select_a4w4_paired_gateup_canonical_down_route(conflict) ==
      Route::kInvalid;
  conflict = query;
  conflict.legacy_gate_requested = true;
  const bool legacy_gate_rejected =
      detail::select_a4w4_paired_gateup_canonical_down_route(conflict) ==
      Route::kInvalid;
  conflict = query;
  conflict.legacy_down_requested = true;
  const bool legacy_down_rejected =
      detail::select_a4w4_paired_gateup_canonical_down_route(conflict) ==
      Route::kInvalid;
  conflict = query;
  conflict.projection_span = false;
  const bool non_span_rejected =
      detail::select_a4w4_paired_gateup_canonical_down_route(conflict) ==
      Route::kInvalid;
  test.expect(
      legacy_mlp_rejected && legacy_gate_rejected &&
          legacy_down_rejected && non_span_rejected && cross_pair_rejected &&
          both_gates_rejected,
      "paired route rejects legacy conflicts, crossed Gate/Down pairs, dual Gate selection, and non-span dispatch");

  constexpr std::size_t kOldGate = 11U;
  constexpr std::size_t kNewGate = 13U;
  constexpr std::size_t kOldDown = 7U;
  constexpr std::size_t kNewDown = 17U;
  constexpr std::size_t kTwoSpanHits =
      2U * runtime::kReferenceDecoderLayerCount;
  test.expect(
      detail::a4w4_paired_gateup_canonical_down_accounting_valid(
          Route::kDisabled, 2U, kOldGate, kOldGate, kNewGate, kNewGate,
          kOldDown, kOldDown, kNewDown, kNewDown) &&
          detail::a4w4_paired_gateup_canonical_down_accounting_valid(
              Route::kOldGateCanonicalDown, 2U, kOldGate,
              kOldGate + kTwoSpanHits, kNewGate, kNewGate, kOldDown,
              kOldDown, kNewDown, kNewDown) &&
          detail::a4w4_paired_gateup_canonical_down_accounting_valid(
              Route::kOldGateLdmatrixPairringDown, 2U, kOldGate,
              kOldGate + kTwoSpanHits, kNewGate, kNewGate, kOldDown,
              kOldDown + kTwoSpanHits, kNewDown, kNewDown) &&
          detail::a4w4_paired_gateup_canonical_down_accounting_valid(
              Route::kNewPairedWarpGateCanonicalDown, 2U, kOldGate,
              kOldGate, kNewGate, kNewGate + kTwoSpanHits, kOldDown,
              kOldDown, kNewDown, kNewDown) &&
          detail::a4w4_paired_gateup_canonical_down_accounting_valid(
              Route::kNewPairedWarpGate16WarpPairringDown, 2U, kOldGate,
              kOldGate, kNewGate, kNewGate + kTwoSpanHits, kOldDown,
              kOldDown, kNewDown, kNewDown + kTwoSpanHits),
      "request accounting proves exactly one Gate family and its matching optional Down");
  test.expect(
      !detail::a4w4_paired_gateup_canonical_down_accounting_valid(
          Route::kNewPairedWarpGateCanonicalDown, 2U, kOldGate, kOldGate,
          kNewGate, kNewGate + kTwoSpanHits - 1U, kOldDown, kOldDown,
          kNewDown, kNewDown) &&
          !detail::a4w4_paired_gateup_canonical_down_accounting_valid(
              Route::kNewPairedWarpGateCanonicalDown, 2U, kOldGate,
              kOldGate + 1U, kNewGate, kNewGate + kTwoSpanHits, kOldDown,
              kOldDown, kNewDown, kNewDown) &&
          !detail::a4w4_paired_gateup_canonical_down_accounting_valid(
              Route::kNewPairedWarpGate16WarpPairringDown, 2U, kOldGate,
              kOldGate, kNewGate, kNewGate + kTwoSpanHits, kOldDown,
              kOldDown, kNewDown, kNewDown + kTwoSpanHits - 1U),
      "request accounting rejects partial coverage and any sibling-family hit");

  const runtime::ReferenceLongPrefillResult result;
  test.expect(
          result.gateup_m128n64_same_cta_launch_hits == 0U &&
          result.gateup_m128n512_fused_quantize_launch_hits == 0U &&
          result.gateup_m128n512_paired_ldmatrix_launch_hits == 0U &&
          result.gateup_m64n8_paired_warp_register_pipeline_launch_hits ==
              0U &&
          result.down_m128n128_ldmatrix_pairring_launch_hits == 0U &&
          result.down_m128n128_16warp_pairring_launch_hits == 0U &&
          result.gdn_chunk64_native_launch_hits == 0U &&
          result.gdn_chunk64_native_logical_token_hits == 0U &&
          result.gdn_prompt_span_macro_launch_hits == 0U &&
          result.gdn_prompt_span_macro_logical_token_hits == 0U,
      "new request-local telemetry is zero for every unselected route");
}

void test_projection_major_gateup_canonical_down_selector_and_accounting(
    TestContext& test) {
  using Query =
      detail::A4W4ProjectionMajorGateUpCanonicalDownSelectorQuery;
  using Route = detail::A4W4ProjectionMajorGateUpCanonicalDownRoute;

  Query query;
  query.projection_span = true;
  test.expect(
      detail::select_a4w4_projection_major_gateup_canonical_down_route(
          query) == Route::kDisabled,
      "projection-major GateUp route is default-off");

  query.master_requested = true;
  test.expect(
      detail::select_a4w4_projection_major_gateup_canonical_down_route(
          query) == Route::kInvalid,
      "projection-major master requires its fused Gate selector");
  query = Query{};
  query.gate_requested = true;
  query.projection_span = true;
  test.expect(
      detail::select_a4w4_projection_major_gateup_canonical_down_route(
          query) == Route::kInvalid,
      "projection-major Gate selector requires its publication master");
  query = Query{};
  query.down_requested = true;
  query.projection_span = true;
  test.expect(
      detail::select_a4w4_projection_major_gateup_canonical_down_route(
          query) == Route::kInvalid,
      "projection-major Down selector cannot run without fused GateUp");

  query = Query{};
  query.master_requested = true;
  query.gate_requested = true;
  query.projection_span = true;
  test.expect(
      detail::select_a4w4_projection_major_gateup_canonical_down_route(
          query) == Route::kGateOnly,
      "projection-major Gate route keeps canonical Down independently selectable");
  query.down_requested = true;
  test.expect(
      detail::select_a4w4_projection_major_gateup_canonical_down_route(
          query) == Route::kGateAndDown,
      "projection-major route can select the independent Down successor");

  Query conflict = query;
  conflict.legacy_publication_requested = true;
  const bool publication_rejected =
      detail::select_a4w4_projection_major_gateup_canonical_down_route(
          conflict) == Route::kInvalid;
  conflict = query;
  conflict.legacy_gate_requested = true;
  const bool gate_rejected =
      detail::select_a4w4_projection_major_gateup_canonical_down_route(
          conflict) == Route::kInvalid;
  conflict = query;
  conflict.legacy_down_requested = true;
  const bool down_rejected =
      detail::select_a4w4_projection_major_gateup_canonical_down_route(
          conflict) == Route::kInvalid;
  conflict = query;
  conflict.projection_span = false;
  const bool non_span_rejected =
      detail::select_a4w4_projection_major_gateup_canonical_down_route(
          conflict) == Route::kInvalid;
  test.expect(publication_rejected && gate_rejected && down_rejected &&
                  non_span_rejected,
              "projection-major route rejects every old publication/leaf family and non-span dispatch");

  constexpr std::size_t kBeforeGate = 13U;
  constexpr std::size_t kBeforeDown = 5U;
  constexpr std::size_t kTwoSpanHits =
      2U * runtime::kReferenceDecoderLayerCount;
  test.expect(
      detail::a4w4_projection_major_gateup_canonical_down_accounting_valid(
          Route::kDisabled, 2U, kBeforeGate, kBeforeGate, kBeforeDown,
          kBeforeDown) &&
          detail::a4w4_projection_major_gateup_canonical_down_accounting_valid(
              Route::kGateOnly, 2U, kBeforeGate,
              kBeforeGate + kTwoSpanHits, kBeforeDown, kBeforeDown) &&
          detail::a4w4_projection_major_gateup_canonical_down_accounting_valid(
              Route::kGateAndDown, 2U, kBeforeGate,
              kBeforeGate + kTwoSpanHits, kBeforeDown,
              kBeforeDown + kTwoSpanHits),
      "projection-major accounting requires exactly spans times 64 independent Gate/Down hits");
  test.expect(
      !detail::a4w4_projection_major_gateup_canonical_down_accounting_valid(
          Route::kGateOnly, 2U, kBeforeGate,
          kBeforeGate + kTwoSpanHits - 1U, kBeforeDown, kBeforeDown) &&
          !detail::a4w4_projection_major_gateup_canonical_down_accounting_valid(
              Route::kGateAndDown, 2U, kBeforeGate,
              kBeforeGate + kTwoSpanHits, kBeforeDown,
              kBeforeDown + kTwoSpanHits - 1U),
      "projection-major accounting rejects partial Gate or Down ownership");

  const runtime::ReferenceLongPrefillResult result;
  test.expect(result.gateup_m64n128_register_pipeline_launch_hits == 0U,
              "projection-major request telemetry defaults to zero");
}

void test_down_m128n128_ldmatrix_pairring_v1_selector_and_accounting(
    TestContext& test) {
  using Query =
      detail::A4W4DownK512M128N128LdmatrixPairringV1SelectorQuery;
  using Route =
      detail::A4W4DownK512M128N128LdmatrixPairringV1Route;

  Query query;
  query.projection_span = true;
  test.expect(
      detail::select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
          query) == Route::kDisabled,
      "v1 pair-ring Down route is default-off");

  query.requested = true;
  test.expect(
      detail::select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
          query) == Route::kInvalid,
      "independent pair-ring Down cannot run without v1 K512 MLP");
  query.mlp_k512_v1_requested = true;
  test.expect(
      detail::select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
          query) == Route::kEnabled,
      "pair-ring Down independently composes with a v1 GateUp selector");

  Query conflict = query;
  conflict.fragment_native_requested = true;
  const bool fragment_rejected =
      detail::select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
          conflict) == Route::kInvalid;
  conflict = query;
  conflict.hybrid_requested = true;
  const bool hybrid_rejected =
      detail::select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
          conflict) == Route::kInvalid;
  conflict = query;
  conflict.conflicting_down_requested = true;
  const bool down_conflict_rejected =
      detail::select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
          conflict) == Route::kInvalid;
  conflict = query;
  conflict.projection_span = false;
  const bool non_span_rejected =
      detail::select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
          conflict) == Route::kInvalid;
  test.expect(fragment_rejected && hybrid_rejected &&
                  down_conflict_rejected && non_span_rejected,
              "v1 pair-ring Down rejects other publications, Down "
              "selectors, and non-span dispatch");

  constexpr std::size_t kBefore = 13U;
  constexpr std::size_t kTwoSpanHits =
      2U * runtime::kReferenceDecoderLayerCount;
  test.expect(
      detail::a4w4_down_k512_m128n128_ldmatrix_pairring_v1_accounting_valid(
          Route::kDisabled, 2U, kTwoSpanHits, kBefore, kBefore) &&
          detail::
              a4w4_down_k512_m128n128_ldmatrix_pairring_v1_accounting_valid(
                  Route::kEnabled, 2U, kTwoSpanHits, kBefore,
                  kBefore + kTwoSpanHits),
      "v1 pair-ring accounting accepts exact zero or spans times 64 hits");
  test.expect(
      !detail::
           a4w4_down_k512_m128n128_ldmatrix_pairring_v1_accounting_valid(
               Route::kEnabled, 2U, kTwoSpanHits - 1U, kBefore,
               kBefore + kTwoSpanHits) &&
          !detail::
               a4w4_down_k512_m128n128_ldmatrix_pairring_v1_accounting_valid(
                   Route::kEnabled, 2U, kTwoSpanHits, kBefore,
                   kBefore + kTwoSpanHits - 1U),
      "v1 pair-ring accounting rejects MLP mismatch and partial coverage");
}

void test_down_m128n128_16warp_pairring_v1_selector_and_accounting(
    TestContext& test) {
  using Query =
      detail::A4W4DownK512M128N128Pairring16V1SelectorQuery;
  using Route = detail::A4W4DownK512M128N128Pairring16V1Route;

  Query query;
  query.projection_span = true;
  test.expect(
      detail::select_a4w4_down_k512_m128n128_16warp_pairring_v1_route(
          query) == Route::kDisabled,
      "16-warp pair-ring Down route is default-off");

  query.requested = true;
  test.expect(
      detail::select_a4w4_down_k512_m128n128_16warp_pairring_v1_route(
          query) == Route::kInvalid,
      "16-warp pair-ring Down cannot run without v1 K512 MLP");
  query.mlp_k512_v1_requested = true;
  test.expect(
      detail::select_a4w4_down_k512_m128n128_16warp_pairring_v1_route(
          query) == Route::kEnabled,
      "16-warp pair-ring Down independently composes with v1 GateUp");

  Query conflict = query;
  conflict.fragment_native_requested = true;
  const bool fragment_rejected =
      detail::select_a4w4_down_k512_m128n128_16warp_pairring_v1_route(
          conflict) == Route::kInvalid;
  conflict = query;
  conflict.hybrid_requested = true;
  const bool hybrid_rejected =
      detail::select_a4w4_down_k512_m128n128_16warp_pairring_v1_route(
          conflict) == Route::kInvalid;
  conflict = query;
  conflict.incumbent_pairring_requested = true;
  const bool incumbent_rejected =
      detail::select_a4w4_down_k512_m128n128_16warp_pairring_v1_route(
          conflict) == Route::kInvalid;
  conflict = query;
  conflict.conflicting_down_requested = true;
  const bool down_conflict_rejected =
      detail::select_a4w4_down_k512_m128n128_16warp_pairring_v1_route(
          conflict) == Route::kInvalid;
  conflict = query;
  conflict.projection_span = false;
  const bool non_span_rejected =
      detail::select_a4w4_down_k512_m128n128_16warp_pairring_v1_route(
          conflict) == Route::kInvalid;
  test.expect(fragment_rejected && hybrid_rejected && incumbent_rejected &&
                  down_conflict_rejected && non_span_rejected,
              "16-warp pair-ring Down rejects other publications, the "
              "incumbent, every Down selector, and non-span dispatch");

  constexpr std::size_t kBefore = 17U;
  constexpr std::size_t kTwoSpanHits =
      2U * runtime::kReferenceDecoderLayerCount;
  test.expect(
      detail::
          a4w4_down_k512_m128n128_16warp_pairring_v1_accounting_valid(
              Route::kDisabled, 2U, kTwoSpanHits, kBefore, kBefore) &&
          detail::
              a4w4_down_k512_m128n128_16warp_pairring_v1_accounting_valid(
                  Route::kEnabled, 2U, kTwoSpanHits, kBefore,
                  kBefore + kTwoSpanHits),
      "16-warp pair-ring accounting accepts exact zero or spans times 64 hits");
  test.expect(
      !detail::
           a4w4_down_k512_m128n128_16warp_pairring_v1_accounting_valid(
               Route::kEnabled, 2U, kTwoSpanHits - 1U, kBefore,
               kBefore + kTwoSpanHits) &&
          !detail::
               a4w4_down_k512_m128n128_16warp_pairring_v1_accounting_valid(
                   Route::kEnabled, 2U, kTwoSpanHits, kBefore,
                   kBefore + kTwoSpanHits - 1U),
      "16-warp pair-ring accounting rejects MLP mismatch and partial coverage");
}

void test_attention_k256_a_exchange_b4_selector_and_accounting(
    TestContext& test) {
  using Implementation =
      detail::A4W4AttentionK256M128N256Implementation;
  using Query = detail::A4W4AttentionK256M128N256ImplementationQuery;

  Query query;
  const bool disabled =
      detail::select_a4w4_attention_k256_m128n256_implementation(query) ==
      Implementation::kDisabled;
  query.incumbent_requested = true;
  const bool incumbent =
      detail::select_a4w4_attention_k256_m128n256_implementation(query) ==
      Implementation::kIncumbent;
  query.incumbent_requested = false;
  query.a_exchange_b4_requested = true;
  const bool candidate =
      detail::select_a4w4_attention_k256_m128n256_implementation(query) ==
      Implementation::kAExchangeB4;
  query.incumbent_requested = true;
  const bool conflict =
      detail::select_a4w4_attention_k256_m128n256_implementation(query) ==
      Implementation::kInvalid;
  query = Query{};
  query.a_exchange_b4_l2_macro4x4_requested = true;
  const bool orphan_l2_macro4x4 =
      detail::select_a4w4_attention_k256_m128n256_implementation(query) ==
      Implementation::kInvalid;
  query.a_exchange_b4_requested = true;
  const bool nested_l2_macro4x4 =
      detail::select_a4w4_attention_k256_m128n256_implementation(query) ==
      Implementation::kAExchangeB4;
  query = Query{};
  query.a_exchange_b3_m128n128_requested = true;
  const bool orphan_b3_m128n128 =
      detail::select_a4w4_attention_k256_m128n256_implementation(query) ==
      Implementation::kInvalid;
  query.a_exchange_b4_requested = true;
  const bool nested_b3_m128n128 =
      detail::select_a4w4_attention_k256_m128n256_implementation(query) ==
      Implementation::kAExchangeB4;
  query.a_exchange_b4_l2_macro4x4_requested = true;
  const bool modifier_conflict =
      detail::select_a4w4_attention_k256_m128n256_implementation(query) ==
      Implementation::kInvalid;
  test.expect(disabled && incumbent && candidate && conflict &&
                  orphan_l2_macro4x4 && nested_l2_macro4x4 &&
                  orphan_b3_m128n128 && nested_b3_m128n128 &&
                  modifier_conflict,
              "K256 Attention implementation selector is default-off and "
              "makes incumbent/A-exchange-B4 exactly mutually exclusive; "
              "the L2 macro4x4 and M128N128/B3 modifiers cannot orphan "
              "their parent or coexist");

  const char* const incumbent_environment = std::getenv(
      "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION");
  const bool incumbent_environment_requested =
      incumbent_environment != nullptr &&
      std::strcmp(incumbent_environment, "1") == 0 &&
      !runtime::optimized_prefill_dispatch_disabled();
  const char* const candidate_environment = std::getenv(
      "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION");
  const bool candidate_environment_requested =
      candidate_environment != nullptr &&
      std::strcmp(candidate_environment, "1") == 0 &&
      !runtime::optimized_prefill_dispatch_disabled();
  const char* const b3_environment = std::getenv(
      "Q3X_RUN_A4W4_ATTENTION_K256_M128N128_A_EXCHANGE_B3_ADMISSION");
  const bool b3_environment_requested =
      b3_environment != nullptr && std::strcmp(b3_environment, "1") == 0 &&
      !runtime::optimized_prefill_dispatch_disabled();

  const bool initial_incumbent =
      detail::exchange_a4w4_attention_k256_m128n256_admission_test_enabled(
          false);
  (void)detail::
      exchange_a4w4_attention_k256_m128n256_admission_test_enabled(true);
  const bool incumbent_compiled =
      detail::exchange_a4w4_attention_k256_m128n256_admission_test_enabled(
          false);
  const bool initial_candidate = detail::
      exchange_a4w4_attention_k256_m128n256_a_exchange_b4_admission_test_enabled(
          false);
  (void)detail::
      exchange_a4w4_attention_k256_m128n256_a_exchange_b4_admission_test_enabled(
          true);
  const bool candidate_compiled = detail::
      exchange_a4w4_attention_k256_m128n256_a_exchange_b4_admission_test_enabled(
          false);
  const bool initial_b3 = detail::
      exchange_a4w4_attention_k256_m128n128_a_exchange_b3_admission_test_enabled(
          false);
  (void)detail::
      exchange_a4w4_attention_k256_m128n128_a_exchange_b3_admission_test_enabled(
          true);
  const bool b3_compiled = detail::
      exchange_a4w4_attention_k256_m128n128_a_exchange_b3_admission_test_enabled(
          false);
  test.expect(
      initial_incumbent ==
              (incumbent_compiled && incumbent_environment_requested) &&
          initial_candidate ==
              (candidate_compiled && candidate_environment_requested) &&
          initial_b3 == (b3_compiled && b3_environment_requested),
      "K256 Attention implementation gates are exact-value, default-off, "
      "global-disable aware, and independently initialized");

  const detail::A4W4AttentionSupermatrixAdmissionHits incumbent_fixture{
      48U, 16U, 64U, 208U};
  const detail::A4W4AttentionSupermatrixAdmissionHits candidate_fixture{
      96U, 32U, 128U, 416U};
  const auto incumbent_hits_before =
      detail::exchange_a4w4_attention_k256_m128n256_admission_test_hits(
          incumbent_fixture);
  const auto candidate_hits_before = detail::
      exchange_a4w4_attention_k256_m128n256_a_exchange_b4_admission_test_hits(
          candidate_fixture);
  const auto observed_incumbent =
      detail::exchange_a4w4_attention_k256_m128n256_admission_test_hits(
          incumbent_hits_before);
  const auto observed_candidate = detail::
      exchange_a4w4_attention_k256_m128n256_a_exchange_b4_admission_test_hits(
          candidate_hits_before);
  test.expect(
      observed_incumbent.linear_input_launch_hits ==
              (incumbent_compiled ? 48U : 0U) &&
          observed_incumbent.full_input_launch_hits ==
              (incumbent_compiled ? 16U : 0U) &&
          observed_incumbent.output_launch_hits ==
              (incumbent_compiled ? 64U : 0U) &&
          observed_incumbent.logical_projection_hits ==
              (incumbent_compiled ? 208U : 0U) &&
          observed_candidate.linear_input_launch_hits ==
              (candidate_compiled ? 96U : 0U) &&
          observed_candidate.full_input_launch_hits ==
              (candidate_compiled ? 32U : 0U) &&
          observed_candidate.output_launch_hits ==
              (candidate_compiled ? 128U : 0U) &&
          observed_candidate.logical_projection_hits ==
              (candidate_compiled ? 416U : 0U),
      "incumbent and A-exchange/B4 K256 Attention hit storage is independent");

  (void)detail::
      exchange_a4w4_attention_k256_m128n256_admission_test_enabled(
          initial_incumbent);
  (void)detail::
      exchange_a4w4_attention_k256_m128n256_a_exchange_b4_admission_test_enabled(
          initial_candidate);
  (void)detail::
      exchange_a4w4_attention_k256_m128n128_a_exchange_b3_admission_test_enabled(
          initial_b3);
}

void test_gateup_m32n512_owner_child_selector(TestContext& test) {
  using Implementation =
      detail::A4W4GateUpK512PairfeedImplementation;
  using Query = detail::A4W4GateUpK512PairfeedImplementationQuery;

  Query query;
  query.projection_span = true;
  const bool default_off =
      detail::select_a4w4_gateup_k512_pairfeed_implementation(query) ==
      Implementation::kDisabled;
  query.parent_pairfeed_requested = true;
  const bool parent =
      detail::select_a4w4_gateup_k512_pairfeed_implementation(query) ==
      Implementation::kM64N128K256LdmatrixPairfeed;
  query.m32n512_owner_requested = true;
  const bool child =
      detail::select_a4w4_gateup_k512_pairfeed_implementation(query) ==
      Implementation::kM32N512Owner;

  Query invalid;
  invalid.projection_span = true;
  invalid.m32n512_owner_requested = true;
  const bool orphan_rejected =
      detail::select_a4w4_gateup_k512_pairfeed_implementation(invalid) ==
      Implementation::kInvalid;
  invalid.parent_pairfeed_requested = true;
  invalid.conflicting_gate_sibling_requested = true;
  const bool sibling_rejected =
      detail::select_a4w4_gateup_k512_pairfeed_implementation(invalid) ==
      Implementation::kInvalid;
  invalid.conflicting_gate_sibling_requested = false;
  invalid.conflicting_macro_requested = true;
  const bool macro_rejected =
      detail::select_a4w4_gateup_k512_pairfeed_implementation(invalid) ==
      Implementation::kInvalid;
  invalid.conflicting_macro_requested = false;
  invalid.projection_span = false;
  const bool non_span_child_rejected =
      detail::select_a4w4_gateup_k512_pairfeed_implementation(invalid) ==
      Implementation::kInvalid;
  invalid.m32n512_owner_requested = false;
  const bool non_span_parent_rejected =
      detail::select_a4w4_gateup_k512_pairfeed_implementation(invalid) ==
      Implementation::kInvalid;
  test.expect(default_off && parent && child && orphan_rejected &&
                  sibling_rejected && macro_rejected &&
                  non_span_child_rejected && non_span_parent_rejected,
              "M32N512 owner is a default-off pair-feed child and rejects "
              "orphans, sibling/macro conflicts, and non-span dispatch");

  const char* const environment = std::getenv(
      "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M32N512_OWNER_ADMISSION");
  const bool environment_requested =
      environment != nullptr && std::strcmp(environment, "1") == 0 &&
      !runtime::optimized_prefill_dispatch_disabled();
  const bool initial = detail::
      exchange_a4w4_gateup_down_k512_edge_m32n512_owner_admission_test_enabled(
          false);
  (void)detail::
      exchange_a4w4_gateup_down_k512_edge_m32n512_owner_admission_test_enabled(
          true);
  const bool compiled = detail::
      exchange_a4w4_gateup_down_k512_edge_m32n512_owner_admission_test_enabled(
          false);
  test.expect(initial == (compiled && environment_requested),
              "M32N512 owner gate is exact-value, default-off, global-disable "
              "aware, and compile-time isolated");
  (void)detail::
      exchange_a4w4_gateup_down_k512_edge_m32n512_owner_admission_test_enabled(
          initial);
}

}  // namespace

int main() {
  TestContext test;
  test_bf16_rounding(test);
  test_logits_analysis(test);
  test_bf16_argmax_analysis(test);
  test_bf16_logits_bits_analysis(test);
  test_bf16_logits_memo_perf(test);
  test_gdn_prompt_span_plan(test);
  test_schedule_and_workspace(test);
  test_fake_linear_weight_validation(test);
  test_a4w4_full_prefill_admission_controls(test);
  test_prefill_admission_gate_orthogonality(test);
  test_factorized_lane_r4_prefill_route(test);
  test_paired_gateup_canonical_down_selector_and_accounting(test);
  test_projection_major_gateup_canonical_down_selector_and_accounting(test);
  test_down_m128n128_ldmatrix_pairring_v1_selector_and_accounting(test);
  test_down_m128n128_16warp_pairring_v1_selector_and_accounting(test);
  test_attention_k256_a_exchange_b4_selector_and_accounting(test);
  test_gateup_m32n512_owner_child_selector(test);
  test_trace_layout_and_factory_error(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " reference-runner host test(s) failed\n";
    return 1;
  }
  std::cout << "Reference runner host tests passed\n";
  return 0;
}
