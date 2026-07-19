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
                  chunk_built.value->prefill_chunk_size == 16U &&
                  tile_result.steps.size() == 16U &&
                  detail::validate_reference_workspace_plan(
                      *chunk_built.value) ==
                      runtime::ReferenceRunnerError::kNone,
              "chunk-sixteen plan and tile result satisfy the runner workspace ABI");

  plan.prefill_chunk_size = runtime::kMaximumRequestPrefillChunkSize;
  test.expect(detail::validate_reference_workspace_plan(plan) ==
                  runtime::ReferenceRunnerError::kInvalidRequestState,
              "chunk metadata cannot exceed the allocated workspace spans");
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

}  // namespace

int main() {
  TestContext test;
  test_bf16_rounding(test);
  test_logits_analysis(test);
  test_bf16_argmax_analysis(test);
  test_bf16_logits_bits_analysis(test);
  test_bf16_logits_memo_perf(test);
  test_schedule_and_workspace(test);
  test_fake_linear_weight_validation(test);
  test_trace_layout_and_factory_error(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " reference-runner host test(s) failed\n";
    return 1;
  }
  std::cout << "Reference runner host tests passed\n";
  return 0;
}
