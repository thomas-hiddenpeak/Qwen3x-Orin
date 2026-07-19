#include "q3x/runtime/reference_runner.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

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
}

}  // namespace

int main() {
  TestContext test;
  test_bf16_rounding(test);
  test_logits_analysis(test);
  test_bf16_logits_bits_analysis(test);
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
