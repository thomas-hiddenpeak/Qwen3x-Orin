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
#include <utility>
#include <vector>

namespace q3x::runtime {

struct ReferenceRunnerPrefillControlTestPeer {
  struct Control {
    std::optional<std::uint32_t> first_position_override;
    std::size_t layer_begin = 0U;
    std::size_t layer_end = kReferenceDecoderLayerCount;
    bool gather_embedding = true;
    bool apply_final_norm = true;
    bool synchronize = true;
    bool commit_state = true;
    bool commit_route = true;
    bool allow_scalar_m1_delegate = true;
    bool allow_cross_layer_m32_fusion = true;
    bool emit_commit_hooks = true;
    bool allow_experimental_gdn_b8_admission = false;
    bool allow_experimental_gdn_chunk64_native_admission = false;
    bool allow_experimental_gdn_chunk64_reference_admission = false;
  };

  struct Result {
    ReferenceRunnerStatus status;
    std::uint32_t first_position = 0U;
    std::uint32_t completed_position = 0U;
    bool delegate_scalar_m1 = false;
    bool legacy_control = false;
    Control selected_control;
  };

  struct LayerRouteReducer {
    ReferenceRunner::PrefillLayerRouteReducer value;
  };

  using LayerRouteFragment =
      ReferenceRunner::PrefillLayerSegmentRouteFragment;
  using LayerRouteSlot = ReferenceRunner::PrefillLayerRouteSlot;

  struct CompactViews {
    std::array<std::uint16_t*, 3U> hidden{};
    std::array<std::uint16_t*, 4U> projection{};
    std::uint16_t* linear_a = nullptr;
    std::uint16_t* linear_b = nullptr;
    float* fp32_scratch = nullptr;
    std::size_t fp32_scratch_elements = 0U;
    std::array<std::uint16_t*, kReferenceDecoderLayerCount> conv_state{};
    std::array<std::uint16_t*, kReferenceDecoderLayerCount> gdn_state{};
    std::array<std::uint16_t*, kReferenceDecoderLayerCount> key_cache{};
    std::array<std::uint16_t*, kReferenceDecoderLayerCount> value_cache{};
    const float* rope_cos = nullptr;
    const float* rope_sin = nullptr;
  };

  [[nodiscard]] static ReferenceRunner empty_runner() noexcept {
    return ReferenceRunner{};
  }

  static void set_route_evidence(
      ReferenceRunner& runner,
      const PrefillRouteEvidence& evidence) noexcept {
    runner.prefill_route_evidence_ = evidence;
  }

  [[nodiscard]] static const PrefillRouteEvidence& route_evidence(
      const ReferenceRunner& runner) noexcept {
    return runner.prefill_route_evidence_;
  }

  [[nodiscard]] static ReferenceRunnerStatus bind_layer_major_views(
      ReferenceRunner& runner,
      ReferenceLayerMajorRequestViews candidate) noexcept {
    return runner.bind_layer_major_candidate_views(std::move(candidate));
  }

  [[nodiscard]] static bool has_layer_major_views(
      const ReferenceRunner& runner) noexcept {
    return runner.layer_major_request_views_.has_value();
  }

  [[nodiscard]] static const ReferenceLayerMajorRequestViews*
  layer_major_views(const ReferenceRunner& runner) noexcept {
    return runner.layer_major_request_views_
               ? &*runner.layer_major_request_views_
               : nullptr;
  }

  [[nodiscard]] static CompactViews compact_views(
      const ReferenceRunner& runner) noexcept {
    CompactViews result;
    std::copy(std::begin(runner.views_.hidden),
              std::end(runner.views_.hidden), result.hidden.begin());
    std::copy(std::begin(runner.views_.projection),
              std::end(runner.views_.projection),
              result.projection.begin());
    result.linear_a = runner.views_.linear_a;
    result.linear_b = runner.views_.linear_b;
    result.fp32_scratch = runner.views_.fp32_scratch;
    result.fp32_scratch_elements = runner.views_.fp32_scratch_elements;
    std::copy(std::begin(runner.views_.conv_state),
              std::end(runner.views_.conv_state),
              result.conv_state.begin());
    std::copy(std::begin(runner.views_.gdn_state),
              std::end(runner.views_.gdn_state),
              result.gdn_state.begin());
    std::copy(std::begin(runner.views_.key_cache),
              std::end(runner.views_.key_cache),
              result.key_cache.begin());
    std::copy(std::begin(runner.views_.value_cache),
              std::end(runner.views_.value_cache),
              result.value_cache.begin());
    result.rope_cos = runner.views_.rope_cos;
    result.rope_sin = runner.views_.rope_sin;
    return result;
  }

  static void release(ReferenceRunner& runner) noexcept { runner.release(); }

  [[nodiscard]] static constexpr std::size_t
  lightweight_enqueue_result_bytes() noexcept {
    return sizeof(ReferenceRunner::PrefillLayerSegmentEnqueueResult);
  }

  [[nodiscard]] static constexpr std::size_t
  legacy_tile_outcome_bytes() noexcept {
    return sizeof(ReferencePrefillTileOutcome);
  }

  [[nodiscard]] static constexpr std::size_t
  lightweight_enqueue_payload_bytes() noexcept {
    return sizeof(ReferenceRunnerStatus) +
           sizeof(ReferenceRunner::PrefillEnqueueRouteFragment);
  }

  [[nodiscard]] static constexpr std::size_t
  whole_request_outcome_bytes() noexcept {
    return sizeof(ReferenceWholeRequestPrefillOutcome);
  }

  static void seed_whole_request_stage(
      ReferenceRunner& runner, const std::uint32_t initial_position,
      const std::uint32_t committed_position,
      const std::uint32_t final_input_token_id,
      const std::size_t prompt_token_count,
      const std::size_t panel_count,
      const std::uint16_t* const final_hidden,
      const PrefillExecutionProgress& progress,
      const PrefillRouteEvidence& staged_route,
      const bool awaiting_commit) noexcept {
    ReferenceRunner::WholeRequestPrefillStage stage;
    stage.phase =
        awaiting_commit
            ? ReferenceRunner::WholeRequestPrefillStagePhase::kAwaitingCommit
            : ReferenceRunner::WholeRequestPrefillStagePhase::kAwaitingLogits;
    stage.expected_initial_sequence_length = initial_position;
    stage.committed_sequence_length = committed_position;
    stage.final_position = committed_position - 1U;
    stage.final_input_token_id = final_input_token_id;
    stage.prompt_token_count = prompt_token_count;
    stage.logical_panel_count = panel_count;
    stage.final_normalized_hidden = final_hidden;
    stage.completed_uncommitted_progress = progress;
    stage.route_evidence_after_commit = staged_route;
    runner.whole_request_prefill_stage_ = stage;
  }

  [[nodiscard]] static bool whole_request_stage_active(
      const ReferenceRunner& runner) noexcept {
    return runner.whole_request_prefill_active();
  }

  [[nodiscard]] static bool whole_request_stage_matches(
      const ReferenceRunner& runner,
      const std::uint32_t initial_position,
      const std::uint32_t committed_position,
      const std::uint32_t final_input_token_id,
      const std::size_t prompt_token_count,
      const std::size_t panel_count,
      const std::uint16_t* const final_hidden,
      const PrefillExecutionProgress& progress,
      const PrefillRouteEvidence& staged_route,
      const bool awaiting_commit) noexcept {
    const ReferenceRunner::WholeRequestPrefillStage& stage =
        runner.whole_request_prefill_stage_;
    const ReferenceRunner::WholeRequestPrefillStagePhase expected_phase =
        awaiting_commit
            ? ReferenceRunner::WholeRequestPrefillStagePhase::kAwaitingCommit
            : ReferenceRunner::WholeRequestPrefillStagePhase::kAwaitingLogits;
    return stage.phase == expected_phase &&
           stage.expected_initial_sequence_length == initial_position &&
           stage.committed_sequence_length == committed_position &&
           stage.final_position == committed_position - 1U &&
           stage.final_input_token_id == final_input_token_id &&
           stage.prompt_token_count == prompt_token_count &&
           stage.logical_panel_count == panel_count &&
           stage.final_normalized_hidden == final_hidden &&
           ReferenceRunner::same_prefill_execution_progress(
               stage.completed_uncommitted_progress, progress) &&
           same_prefill_route_evidence_for_peer(
               stage.route_evidence_after_commit, staged_route);
  }

  [[nodiscard]] static bool poisoned(
      const ReferenceRunner& runner) noexcept {
    return runner.poisoned_;
  }

  [[nodiscard]] static LayerRouteFragment production_layer_route(
      const std::size_t layer, const std::uint32_t first_position,
      const std::uint32_t token_count) noexcept {
    LayerRouteFragment fragment;
    fragment.layer = layer;
    fragment.first_position = first_position;
    fragment.token_count = token_count;
    fragment.recorded_slots =
        ReferenceRunner::expected_prefill_layer_route_slots(layer);
    for (std::size_t slot = 0U;
         slot < ReferenceRunner::kPrefillLayerRouteSlotCount; ++slot) {
      fragment.dispositions[slot] = PrefillRouteDisposition::kProduction;
    }
    return fragment;
  }

  static void set_route_disposition(
      LayerRouteFragment& fragment, const LayerRouteSlot slot,
      const PrefillRouteDisposition disposition) noexcept {
    fragment.dispositions[static_cast<std::size_t>(slot)] = disposition;
  }

  static void set_forbidden_boundary(
      LayerRouteFragment& fragment,
      const PrefillForbiddenBoundary boundary) noexcept {
    fragment.forbidden_boundaries = static_cast<std::uint8_t>(
        fragment.forbidden_boundaries |
        static_cast<std::uint8_t>(
            1U << static_cast<std::size_t>(boundary)));
  }

  [[nodiscard]] static ReferenceRunnerStatus reduce_layer_route(
      LayerRouteReducer& reducer,
      const LayerRouteFragment& fragment) noexcept {
    return ReferenceRunner::reduce_prefill_layer_route_fragment(
        fragment, reducer.value);
  }

  [[nodiscard]] static bool route_initialized(
      const LayerRouteReducer& reducer) noexcept {
    return reducer.value.initialized;
  }

  [[nodiscard]] static std::size_t route_layer(
      const LayerRouteReducer& reducer) noexcept {
    return reducer.value.route_fragment.layer;
  }

  [[nodiscard]] static LayerRouteFragment reduced_route(
      const LayerRouteReducer& reducer) noexcept {
    return reducer.value.route_fragment;
  }

  [[nodiscard]] static ReferenceRunnerStatus collapse_layer_route(
      const LayerRouteFragment& fragment,
      PrefillRouteEvidence& layer_pass) noexcept {
    return ReferenceRunner::collapse_prefill_layer_route_fragment(
        fragment, layer_pass);
  }

  [[nodiscard]] static Result select(
      const Control& requested, const std::uint32_t current_position,
      const std::uint32_t max_sequence_length,
      const std::uint32_t workspace_token_capacity,
      const std::size_t token_count,
      const ReferencePrefillTileOptions& options = {}) noexcept {
    ReferenceRunner::PrefillTileExecutionControl control;
    control.first_position_override = requested.first_position_override;
    control.layer_begin = requested.layer_begin;
    control.layer_end = requested.layer_end;
    control.gather_embedding = requested.gather_embedding;
    control.apply_final_norm = requested.apply_final_norm;
    control.synchronize = requested.synchronize;
    control.commit_state = requested.commit_state;
    control.commit_route = requested.commit_route;
    control.allow_scalar_m1_delegate = requested.allow_scalar_m1_delegate;
    control.allow_cross_layer_m32_fusion =
        requested.allow_cross_layer_m32_fusion;
    control.emit_commit_hooks = requested.emit_commit_hooks;
    control.allow_experimental_gdn_b8_admission =
        requested.allow_experimental_gdn_b8_admission;
    control.allow_experimental_gdn_chunk64_native_admission =
        requested.allow_experimental_gdn_chunk64_native_admission;
    control.allow_experimental_gdn_chunk64_reference_admission =
        requested.allow_experimental_gdn_chunk64_reference_admission;
    return select_private(control, current_position, max_sequence_length,
                          workspace_token_capacity, token_count, options);
  }

  [[nodiscard]] static Result legacy_defaults(
      const std::uint32_t current_position,
      const std::uint32_t max_sequence_length,
      const std::uint32_t workspace_token_capacity,
      const std::size_t token_count,
      const ReferencePrefillTileOptions& options = {}) noexcept {
    return select_private(
        ReferenceRunner::legacy_prefill_tile_execution_control(),
        current_position, max_sequence_length, workspace_token_capacity,
        token_count, options);
  }

 private:
  [[nodiscard]] static bool same_prefill_route_evidence_for_peer(
      const PrefillRouteEvidence& left,
      const PrefillRouteEvidence& right) noexcept {
    for (std::size_t index = 0U; index < left.operators.size(); ++index) {
      const PrefillOperatorRouteCounts& left_counts = left.operators[index];
      const PrefillOperatorRouteCounts& right_counts =
          right.operators[index];
      if (left_counts.production_hits != right_counts.production_hits ||
          left_counts.exact_fallback_hits !=
              right_counts.exact_fallback_hits ||
          left_counts.forbidden_hits != right_counts.forbidden_hits) {
        return false;
      }
    }
    return left.forbidden_boundary_hits == right.forbidden_boundary_hits &&
           left.completed_layer_passes == right.completed_layer_passes &&
           left.expected_layer_passes == right.expected_layer_passes &&
           left.request_active == right.request_active &&
           left.complete == right.complete && left.valid == right.valid &&
           left.error == right.error;
  }

  [[nodiscard]] static Result select_private(
      const ReferenceRunner::PrefillTileExecutionControl& control,
      const std::uint32_t current_position,
      const std::uint32_t max_sequence_length,
      const std::uint32_t workspace_token_capacity,
      const std::size_t token_count,
      const ReferencePrefillTileOptions& options) noexcept {
    ReferenceRunner::PrefillTileExecutionSelection selection;
    Result result;
    result.status = ReferenceRunner::select_prefill_tile_execution(
        control, current_position, max_sequence_length,
        workspace_token_capacity, token_count, options, selection);
    result.first_position = selection.first_position;
    result.completed_position = selection.completed_position;
    result.delegate_scalar_m1 = selection.delegate_scalar_m1;
    result.legacy_control =
        ReferenceRunner::is_legacy_prefill_tile_execution_control(control);
    result.selected_control = {
        control.first_position_override,
        control.layer_begin,
        control.layer_end,
        control.gather_embedding,
        control.apply_final_norm,
        control.synchronize,
        control.commit_state,
        control.commit_route,
        control.allow_scalar_m1_delegate,
        control.allow_cross_layer_m32_fusion,
        control.emit_commit_hooks,
        control.allow_experimental_gdn_b8_admission,
        control.allow_experimental_gdn_chunk64_native_admission,
        control.allow_experimental_gdn_chunk64_reference_admission};
    return result;
  }
};

}  // namespace q3x::runtime

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

[[nodiscard]] bool same_prefill_route_evidence(
    const runtime::PrefillRouteEvidence& left,
    const runtime::PrefillRouteEvidence& right) noexcept {
  for (std::size_t index = 0U; index < left.operators.size(); ++index) {
    const runtime::PrefillOperatorRouteCounts& left_counts =
        left.operators[index];
    const runtime::PrefillOperatorRouteCounts& right_counts =
        right.operators[index];
    if (left_counts.production_hits != right_counts.production_hits ||
        left_counts.exact_fallback_hits !=
            right_counts.exact_fallback_hits ||
        left_counts.forbidden_hits != right_counts.forbidden_hits) {
      return false;
    }
  }
  return left.forbidden_boundary_hits == right.forbidden_boundary_hits &&
         left.completed_layer_passes == right.completed_layer_passes &&
         left.expected_layer_passes == right.expected_layer_passes &&
         left.request_active == right.request_active &&
         left.complete == right.complete && left.valid == right.valid &&
         left.error == right.error;
}

[[nodiscard]] runtime::PrefillRouteEvidence route_evidence_fixture(
    const std::uint64_t seed) noexcept {
  runtime::PrefillRouteEvidence evidence;
  for (std::size_t index = 0U; index < evidence.operators.size(); ++index) {
    const std::uint64_t offset = static_cast<std::uint64_t>(index) * 3U;
    evidence.operators[index] = {
        seed + offset + 1U, seed + offset + 2U, seed + offset + 3U};
  }
  for (std::size_t index = 0U;
       index < evidence.forbidden_boundary_hits.size(); ++index) {
    evidence.forbidden_boundary_hits[index] =
        seed + static_cast<std::uint64_t>(index) + 31U;
  }
  evidence.completed_layer_passes = seed + 41U;
  evidence.expected_layer_passes = seed + 43U;
  evidence.request_active = true;
  evidence.complete = false;
  evidence.valid = false;
  evidence.error = runtime::PrefillRouteEvidenceError::kForbiddenRoute;
  return evidence;
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

void test_prefill_tile_execution_control(TestContext& test) {
  using Peer = runtime::ReferenceRunnerPrefillControlTestPeer;
  const auto operation_is = [](const Peer::Result& result,
                               const std::string_view expected) noexcept {
    return result.status.operation != nullptr &&
           std::string_view(result.status.operation) == expected;
  };

  const Peer::Result legacy =
      Peer::legacy_defaults(17U, 100U, 512U, 32U);
  test.expect(
      legacy.status.ok() && legacy.legacy_control &&
          !legacy.selected_control.first_position_override.has_value() &&
          legacy.selected_control.layer_begin == 0U &&
          legacy.selected_control.layer_end ==
              runtime::kReferenceDecoderLayerCount &&
          legacy.selected_control.gather_embedding &&
          legacy.selected_control.apply_final_norm &&
          legacy.selected_control.synchronize &&
          legacy.selected_control.commit_state &&
          legacy.selected_control.commit_route &&
          legacy.selected_control.allow_scalar_m1_delegate &&
          legacy.selected_control.allow_cross_layer_m32_fusion &&
          legacy.selected_control.emit_commit_hooks &&
          legacy.selected_control.allow_experimental_gdn_b8_admission &&
          legacy.selected_control
              .allow_experimental_gdn_chunk64_native_admission &&
          legacy.selected_control
              .allow_experimental_gdn_chunk64_reference_admission &&
          legacy.first_position == 17U &&
          legacy.completed_position == 49U &&
          !legacy.delegate_scalar_m1,
      "public Prefill control defaults preserve the complete legacy path");
  const Peer::Result legacy_m1 =
      Peer::legacy_defaults(17U, 100U, 512U, 1U);
  test.expect(legacy_m1.status.ok() && legacy_m1.delegate_scalar_m1,
              "only the complete legacy M1 path delegates to step");

  Peer::Control candidate;
  candidate.first_position_override = 513U;
  candidate.layer_begin = 17U;
  candidate.layer_end = 18U;
  candidate.gather_embedding = false;
  candidate.apply_final_norm = false;
  candidate.synchronize = false;
  candidate.commit_state = false;
  candidate.commit_route = false;
  candidate.allow_scalar_m1_delegate = false;
  candidate.allow_cross_layer_m32_fusion = false;
  candidate.emit_commit_hooks = false;
  const Peer::Result candidate_m1 =
      Peer::select(candidate, 8'000U, 8'192U, 512U, 1U);
  test.expect(candidate_m1.status.ok() && !candidate_m1.legacy_control &&
                  candidate_m1.first_position == 513U &&
                  candidate_m1.completed_position == 514U &&
                  !candidate_m1.delegate_scalar_m1 &&
                  !candidate_m1.selected_control
                       .allow_experimental_gdn_b8_admission &&
                  !candidate_m1.selected_control
                       .allow_experimental_gdn_chunk64_native_admission &&
                  !candidate_m1.selected_control
                       .allow_experimental_gdn_chunk64_reference_admission,
              "single-layer M1 candidate is compatibility-exact, enqueue-only, "
              "and independent of host state");

  const auto expect_experimental_gdn_rejected =
      [&test](const Peer::Control& control,
              const std::string_view message) {
        const Peer::Result result =
            Peer::select(control, 17U, 8'192U, 512U, 32U);
        test.expect(
            !result.status &&
                result.status.error ==
                    runtime::ReferenceRunnerError::kInvalidStepOptions &&
                result.status.operation != nullptr &&
                std::string_view(result.status.operation) ==
                    "prefill_tile_candidate_experimental_gdn_admission",
            message);
      };
  Peer::Control experimental_gdn = candidate;
  experimental_gdn.allow_experimental_gdn_b8_admission = true;
  expect_experimental_gdn_rejected(
      experimental_gdn,
      "single-layer candidate rejects approximate B8 before enqueue");
  experimental_gdn = candidate;
  experimental_gdn.allow_experimental_gdn_chunk64_native_admission = true;
  expect_experimental_gdn_rejected(
      experimental_gdn,
      "single-layer candidate rejects Chunk64 native before enqueue");
  experimental_gdn = candidate;
  experimental_gdn.allow_experimental_gdn_chunk64_reference_admission = true;
  expect_experimental_gdn_rejected(
      experimental_gdn,
      "single-layer candidate rejects external Chunk64 reference before enqueue");

  Peer::Control synchronized_candidate = candidate;
  synchronized_candidate.synchronize = true;
  Peer::Result rejected = Peer::select(
      synchronized_candidate, 8'000U, 8'192U, 512U, 32U);
  test.expect(
      !rejected.status &&
          operation_is(rejected, "prefill_tile_candidate_not_enqueue_only"),
      "single-layer candidate leaves synchronization to the outer executor");

  Peer::Control invalid = candidate;
  invalid.first_position_override.reset();
  rejected = Peer::select(invalid, 17U, 8'192U, 512U, 32U);
  test.expect(!rejected.status &&
                  operation_is(rejected,
                               "prefill_tile_first_position_override"),
              "candidate first position never depends on uncommitted host state");

  runtime::ReferencePrefillTileOptions candidate_options;
  candidate_options.retain_last_hidden_for_logits = true;
  rejected = Peer::select(candidate, 17U, 8'192U, 512U, 32U,
                          candidate_options);
  test.expect(
      !rejected.status &&
          operation_is(rejected,
                       "prefill_tile_candidate_completion_options"),
      "enqueue-only candidate cannot publish a retained hidden row");
  candidate_options = {};
  candidate_options.measure_timing = true;
  rejected = Peer::select(candidate, 17U, 8'192U, 512U, 32U,
                          candidate_options);
  test.expect(
      !rejected.status &&
          operation_is(rejected,
                       "prefill_tile_candidate_completion_options"),
      "enqueue-only candidate cannot report a host completion timer");

  invalid = candidate;
  invalid.apply_final_norm = true;
  rejected = Peer::select(invalid, 17U, 8'192U, 512U, 32U);
  test.expect(!rejected.status &&
                  operation_is(rejected, "prefill_tile_partial_final_norm"),
              "partial layer execution cannot own final normalization");

  invalid = candidate;
  invalid.commit_state = true;
  rejected = Peer::select(invalid, 17U, 8'192U, 512U, 32U);
  test.expect(
      !rejected.status &&
          operation_is(rejected, "prefill_tile_candidate_not_enqueue_only"),
      "single-layer candidate cannot publish request state");

  invalid = candidate;
  invalid.commit_route = true;
  rejected = Peer::select(invalid, 17U, 8'192U, 512U, 32U);
  test.expect(
      !rejected.status &&
          operation_is(rejected, "prefill_tile_candidate_not_enqueue_only"),
      "single-layer candidate cannot commit incomplete route evidence");

  invalid = candidate;
  invalid.emit_commit_hooks = true;
  rejected = Peer::select(invalid, 17U, 8'192U, 512U, 32U);
  test.expect(
      !rejected.status &&
          operation_is(rejected, "prefill_tile_candidate_not_enqueue_only"),
      "commit hooks cannot observe an uncommitted partial layer");

  invalid = candidate;
  invalid.layer_end = invalid.layer_begin + 2U;
  rejected = Peer::select(invalid, 17U, 8'192U, 512U, 32U);
  test.expect(
      !rejected.status &&
          operation_is(rejected, "prefill_tile_candidate_layer_span"),
      "candidate seam executes exactly one model layer per enqueue call");

  invalid = candidate;
  invalid.layer_end = invalid.layer_begin;
  rejected = Peer::select(invalid, 17U, 8'192U, 512U, 32U);
  test.expect(!rejected.status &&
                  operation_is(rejected, "prefill_tile_layer_range"),
              "empty layer ranges fail closed");
  invalid = candidate;
  invalid.layer_end = runtime::kReferenceDecoderLayerCount + 1U;
  rejected = Peer::select(invalid, 17U, 8'192U, 512U, 32U);
  test.expect(!rejected.status &&
                  operation_is(rejected, "prefill_tile_layer_range"),
              "out-of-range layer endpoints fail closed");

  invalid = candidate;
  invalid.first_position_override = 8'192U;
  rejected = Peer::select(invalid, 0U, 8'192U, 512U, 1U);
  test.expect(!rejected.status &&
                  rejected.status.error ==
                      runtime::ReferenceRunnerError::kCapacityExceeded &&
                  operation_is(rejected, "prefill_tile_capacity"),
              "first-position plus token count is checked without overflow");
}

void test_prefill_layer_route_reducer(TestContext& test) {
  using Peer = runtime::ReferenceRunnerPrefillControlTestPeer;
  using Slot = Peer::LayerRouteSlot;
  const auto operation_is = [](const runtime::ReferenceRunnerStatus& status,
                               const std::string_view expected) noexcept {
    return status.operation != nullptr &&
           std::string_view(status.operation) == expected;
  };
  const auto disposition = [](const Peer::LayerRouteFragment& fragment,
                              const Slot slot) noexcept {
    return fragment.dispositions[static_cast<std::size_t>(slot)];
  };
  const auto boundary_bit = [](
                                const runtime::PrefillForbiddenBoundary boundary)
      noexcept {
        return static_cast<std::uint8_t>(
            1U << static_cast<std::size_t>(boundary));
      };

  test.expect(
      Peer::legacy_tile_outcome_bytes() >= 32U * 1'024U &&
          Peer::lightweight_enqueue_result_bytes() < 1'024U &&
          Peer::lightweight_enqueue_result_bytes() <=
              Peer::lightweight_enqueue_payload_bytes() + 16U &&
          Peer::lightweight_enqueue_result_bytes() * 16U <
              Peer::legacy_tile_outcome_bytes(),
      "candidate enqueue result contains route identity, not a C512 transcript");

  Peer::LayerRouteReducer linear;
  Peer::LayerRouteFragment first =
      Peer::production_layer_route(0U, 0U, 512U);
  Peer::set_route_disposition(
      first, Slot::kNvFp4GateUp,
      runtime::PrefillRouteDisposition::kExactFallback);
  runtime::ReferenceRunnerStatus status =
      Peer::reduce_layer_route(linear, first);
  Peer::LayerRouteFragment second =
      Peer::production_layer_route(0U, 512U, 256U);
  Peer::set_route_disposition(
      second, Slot::kGdn, runtime::PrefillRouteDisposition::kForbidden);
  Peer::set_forbidden_boundary(
      second, runtime::PrefillForbiddenBoundary::kApproximateNumerics);
  status = status.ok() ? Peer::reduce_layer_route(linear, second) : status;
  const Peer::LayerRouteFragment reduced_linear =
      Peer::reduced_route(linear);
  test.expect(
      status.ok() && Peer::route_initialized(linear) &&
          Peer::route_layer(linear) == 0U &&
          reduced_linear.first_position == 0U &&
          reduced_linear.token_count == 768U &&
          disposition(reduced_linear, Slot::kNvFp4GateUp) ==
              runtime::PrefillRouteDisposition::kExactFallback &&
          disposition(reduced_linear, Slot::kGdn) ==
              runtime::PrefillRouteDisposition::kForbidden &&
          disposition(reduced_linear, Slot::kQOrLinearQkv) ==
              runtime::PrefillRouteDisposition::kProduction &&
          (reduced_linear.forbidden_boundaries &
           boundary_bit(
               runtime::PrefillForbiddenBoundary::kApproximateNumerics)) != 0U,
      "same-layer physical segments preserve geometry and weakest typed routes");

  const auto reduce_gate_sequence =
      [&test](const std::array<runtime::PrefillRouteDisposition, 3U>& routes)
      noexcept {
        Peer::LayerRouteReducer reducer;
        runtime::ReferenceRunnerStatus sequence_status;
        for (std::size_t index = 0U; index < routes.size(); ++index) {
          Peer::LayerRouteFragment fragment =
              Peer::production_layer_route(
                  0U, static_cast<std::uint32_t>(index * 32U), 32U);
          Peer::set_route_disposition(
              fragment, Slot::kNvFp4GateUp, routes[index]);
          if (sequence_status.ok()) {
            sequence_status = Peer::reduce_layer_route(reducer, fragment);
          }
        }
        test.expect(sequence_status.ok(),
                    "route algebra fixture remains valid");
        return Peer::reduced_route(reducer);
      };
  const Peer::LayerRouteFragment pff = reduce_gate_sequence(
      {runtime::PrefillRouteDisposition::kProduction,
       runtime::PrefillRouteDisposition::kExactFallback,
       runtime::PrefillRouteDisposition::kForbidden});
  const Peer::LayerRouteFragment fpf = reduce_gate_sequence(
      {runtime::PrefillRouteDisposition::kForbidden,
       runtime::PrefillRouteDisposition::kProduction,
       runtime::PrefillRouteDisposition::kExactFallback});
  const Peer::LayerRouteFragment ffp = reduce_gate_sequence(
      {runtime::PrefillRouteDisposition::kExactFallback,
       runtime::PrefillRouteDisposition::kForbidden,
       runtime::PrefillRouteDisposition::kProduction});
  const Peer::LayerRouteFragment iii = reduce_gate_sequence(
      {runtime::PrefillRouteDisposition::kExactFallback,
       runtime::PrefillRouteDisposition::kExactFallback,
       runtime::PrefillRouteDisposition::kExactFallback});
  test.expect(
      disposition(pff, Slot::kNvFp4GateUp) ==
              runtime::PrefillRouteDisposition::kForbidden &&
          disposition(fpf, Slot::kNvFp4GateUp) ==
              disposition(pff, Slot::kNvFp4GateUp) &&
          disposition(ffp, Slot::kNvFp4GateUp) ==
              disposition(pff, Slot::kNvFp4GateUp) &&
          disposition(iii, Slot::kNvFp4GateUp) ==
              runtime::PrefillRouteDisposition::kExactFallback,
      "explicit weakest-route algebra is commutative, associative, and idempotent");

  Peer::LayerRouteReducer full;
  Peer::LayerRouteFragment full_first =
      Peer::production_layer_route(3U, 1'024U, 512U);
  Peer::set_route_disposition(
      full_first, Slot::kQOrLinearQkv,
      runtime::PrefillRouteDisposition::kExactFallback);
  Peer::LayerRouteFragment full_second =
      Peer::production_layer_route(3U, 1'536U, 512U);
  Peer::set_route_disposition(
      full_second, Slot::kFullK,
      runtime::PrefillRouteDisposition::kExactFallback);
  status = Peer::reduce_layer_route(full, full_first);
  status = status.ok() ? Peer::reduce_layer_route(full, full_second) : status;
  const Peer::LayerRouteFragment reduced_full = Peer::reduced_route(full);
  runtime::PrefillRouteEvidence collapsed_full;
  status = status.ok()
               ? Peer::collapse_layer_route(reduced_full, collapsed_full)
               : status;
  const runtime::PrefillOperatorRouteCounts& collapsed_qkv =
      collapsed_full.operators[static_cast<std::size_t>(
          runtime::PrefillOperatorRole::kFp8Qkv)];
  test.expect(
      status.ok() &&
          disposition(reduced_full, Slot::kQOrLinearQkv) ==
              runtime::PrefillRouteDisposition::kExactFallback &&
          disposition(reduced_full, Slot::kFullK) ==
              runtime::PrefillRouteDisposition::kExactFallback &&
          disposition(reduced_full, Slot::kFullV) ==
              runtime::PrefillRouteDisposition::kProduction &&
          collapsed_qkv.production_hits == 1U &&
          collapsed_qkv.exact_fallback_hits == 2U &&
          collapsed_qkv.forbidden_hits == 0U,
      "full Q/K/V ordinals survive segment reduction before legacy collapse");

  const Peer::LayerRouteFragment before_rejection =
      Peer::reduced_route(linear);
  Peer::LayerRouteFragment mixed_layer =
      Peer::production_layer_route(1U, 768U, 32U);
  status = Peer::reduce_layer_route(linear, mixed_layer);
  test.expect(
      !status && operation_is(status, "prefill_layer_route_mixed_layer") &&
          Peer::reduced_route(linear).token_count ==
              before_rejection.token_count,
      "one reducer cannot combine different model layers");

  Peer::LayerRouteFragment out_of_order =
      Peer::production_layer_route(0U, 800U, 32U);
  status = Peer::reduce_layer_route(linear, out_of_order);
  test.expect(
      !status &&
          operation_is(status, "prefill_layer_route_segment_order") &&
          Peer::reduced_route(linear).token_count ==
              before_rejection.token_count,
      "same-layer segment order is monotonic and transactional");

  Peer::LayerRouteReducer maximum_panel;
  status = {};
  for (std::size_t segment = 0U; segment < 16U; ++segment) {
    const Peer::LayerRouteFragment fragment =
        Peer::production_layer_route(
            0U, static_cast<std::uint32_t>(segment * 512U), 512U);
    if (status.ok()) {
      status = Peer::reduce_layer_route(maximum_panel, fragment);
    }
  }
  test.expect(status.ok() &&
                  Peer::reduced_route(maximum_panel).token_count == 8'192U,
              "same-layer reducer admits exactly one complete C8192 panel");
  const Peer::LayerRouteFragment beyond_panel =
      Peer::production_layer_route(0U, 8'192U, 512U);
  status = Peer::reduce_layer_route(maximum_panel, beyond_panel);
  test.expect(
      !status && operation_is(status, "prefill_layer_route_segment_span") &&
          Peer::reduced_route(maximum_panel).token_count == 8'192U,
      "same-layer reducer rejects an unbounded seventeenth C512 segment");

  Peer::LayerRouteReducer malformed;
  Peer::LayerRouteFragment bad_slots =
      Peer::production_layer_route(0U, 0U, 32U);
  bad_slots.recorded_slots = static_cast<std::uint16_t>(
      bad_slots.recorded_slots &
      ~static_cast<std::uint16_t>(
          1U << static_cast<std::size_t>(Slot::kQOrLinearQkv)));
  status = Peer::reduce_layer_route(malformed, bad_slots);
  test.expect(
      !status &&
          operation_is(status, "prefill_layer_route_fragment_slots") &&
          !Peer::route_initialized(malformed),
      "missing ordinal route identity fails before reducer publication");

  Peer::LayerRouteFragment bad_geometry =
      Peer::production_layer_route(0U, 0U, 0U);
  status = Peer::reduce_layer_route(malformed, bad_geometry);
  test.expect(
      !status &&
          operation_is(status, "prefill_layer_route_fragment_geometry") &&
          !Peer::route_initialized(malformed),
      "zero-length route geometry fails closed");

  Peer::LayerRouteFragment oversized_physical =
      Peer::production_layer_route(0U, 0U, 513U);
  status = Peer::reduce_layer_route(malformed, oversized_physical);
  test.expect(
      !status &&
          operation_is(status, "prefill_layer_route_physical_segment") &&
          !Peer::route_initialized(malformed),
      "legacy fallback reducer retains the C512 physical-segment boundary");

  Peer::LayerRouteFragment bad_disposition =
      Peer::production_layer_route(0U, 0U, 32U);
  Peer::set_route_disposition(
      bad_disposition, Slot::kGdn,
      static_cast<runtime::PrefillRouteDisposition>(255U));
  status = Peer::reduce_layer_route(malformed, bad_disposition);
  test.expect(
      !status &&
          operation_is(status,
                       "prefill_layer_route_fragment_disposition") &&
          !Peer::route_initialized(malformed),
      "unknown route dispositions never participate in weakest selection");

  Peer::LayerRouteFragment bad_boundaries =
      Peer::production_layer_route(0U, 0U, 32U);
  bad_boundaries.forbidden_boundaries = 0x80U;
  status = Peer::reduce_layer_route(malformed, bad_boundaries);
  test.expect(
      !status &&
          operation_is(status,
                       "prefill_layer_route_fragment_boundaries") &&
          !Peer::route_initialized(malformed),
      "unknown forbidden-boundary bits fail closed");

  const Peer::LayerRouteFragment bad_layer =
      Peer::production_layer_route(
          runtime::kReferenceDecoderLayerCount, 0U, 32U);
  status = Peer::reduce_layer_route(malformed, bad_layer);
  test.expect(!status &&
                  operation_is(status, "prefill_layer_route_layer") &&
                  !Peer::route_initialized(malformed),
              "out-of-schedule route reduction fails closed");
}

void test_prefill_route_evidence_runner_lifetime(TestContext& test) {
  using Peer = runtime::ReferenceRunnerPrefillControlTestPeer;
  const runtime::PrefillRouteEvidence empty;
  const runtime::PrefillRouteEvidence first = route_evidence_fixture(100U);
  const runtime::PrefillRouteEvidence second = route_evidence_fixture(200U);

  runtime::ReferenceRunner move_construct_source = Peer::empty_runner();
  Peer::set_route_evidence(move_construct_source, first);
  runtime::ReferenceRunner move_constructed(
      std::move(move_construct_source));
  test.expect(
      same_prefill_route_evidence(Peer::route_evidence(move_constructed),
                                  first) &&
          same_prefill_route_evidence(
              Peer::route_evidence(move_construct_source), empty),
      "runner move construction transfers Prefill route evidence and clears "
      "the source");

  runtime::ReferenceRunner move_assign_source = Peer::empty_runner();
  runtime::ReferenceRunner move_assign_target = Peer::empty_runner();
  Peer::set_route_evidence(move_assign_source, first);
  Peer::set_route_evidence(move_assign_target, second);
  move_assign_target = std::move(move_assign_source);
  test.expect(
      same_prefill_route_evidence(Peer::route_evidence(move_assign_target),
                                  first) &&
          same_prefill_route_evidence(
              Peer::route_evidence(move_assign_source), empty),
      "runner move assignment replaces prior Prefill route evidence and "
      "clears the source");

  runtime::ReferenceRunner self_move = Peer::empty_runner();
  Peer::set_route_evidence(self_move, second);
  runtime::ReferenceRunner* const self_move_source = &self_move;
  self_move = std::move(*self_move_source);
  test.expect(
      same_prefill_route_evidence(Peer::route_evidence(self_move), second),
      "runner self move preserves Prefill route evidence");

  runtime::ReferenceRunner empty_source = Peer::empty_runner();
  runtime::ReferenceRunner nonempty_target = Peer::empty_runner();
  Peer::set_route_evidence(nonempty_target, first);
  nonempty_target = std::move(empty_source);
  test.expect(
      same_prefill_route_evidence(Peer::route_evidence(nonempty_target),
                                  empty) &&
          same_prefill_route_evidence(Peer::route_evidence(empty_source),
                                      empty),
      "moving an empty runner clears stale target Prefill route evidence");

  runtime::ReferenceRunner released = Peer::empty_runner();
  Peer::set_route_evidence(released, first);
  Peer::release(released);
  test.expect(
      same_prefill_route_evidence(Peer::route_evidence(released), empty),
      "runner release clears Prefill route evidence");
}

[[nodiscard]] runtime::ReferenceLayerMajorRequestViews
make_layer_major_runner_view_fixture(
    std::array<std::uint64_t, 192U>& identities) noexcept {
  runtime::ReferenceLayerMajorRequestViews views;
  views.descriptor.profile =
      runtime::RequestMemoryProfile::kLayerMajorC8192;
  views.descriptor.legacy_prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;

  std::size_t identity = 0U;
  const auto next_pointer = [&]() noexcept -> void* {
    return static_cast<void*>(&identities[identity++]);
  };
  views.prompt_residual_bf16.storage.device_data = next_pointer();
  views.panel_token_ids_u32.storage.device_data = next_pointer();
  views.gdn.qkv_bf16.storage.device_data = next_pointer();
  views.attention.raw_q_gate_bf16.storage.device_data = next_pointer();
  views.mlp.gate_bf16.storage.device_data = next_pointer();
  views.final_hidden_bf16.storage.device_data = next_pointer();
  for (runtime::DeviceMatrixView& hidden :
       views.legacy_c512.hidden_bf16) {
    hidden.storage.device_data = next_pointer();
  }
  for (runtime::DeviceMatrixView& projection :
       views.legacy_c512.projection_bf16) {
    projection.storage.device_data = next_pointer();
  }
  views.legacy_c512.linear_a_bf16.storage.device_data = next_pointer();
  views.legacy_c512.linear_b_bf16.storage.device_data = next_pointer();
  views.legacy_c512.fp32_scratch.device_data = next_pointer();
  views.legacy_c512.fp32_scratch.element_capacity =
      runtime::kReferenceVocabularySize;

  std::size_t linear_slot = 0U;
  std::size_t full_slot = 0U;
  for (std::size_t layer = 0U;
       layer < runtime::kReferenceDecoderLayerCount; ++layer) {
    const q3x::model::LayerType type =
        detail::expected_reference_layer_type(layer);
    runtime::RequestLayerSlot& slot = views.descriptor.layers[layer];
    slot.type = type;
    if (type == q3x::model::LayerType::kLinearAttention) {
      slot.slot = linear_slot;
      views.persistent.conv_state_bf16[linear_slot].device_data =
          next_pointer();
      views.persistent.gdn_state_bf16[linear_slot].device_data =
          next_pointer();
      ++linear_slot;
    } else {
      slot.slot = full_slot;
      views.persistent.key_cache_bf16[full_slot].device_data =
          next_pointer();
      views.persistent.value_cache_bf16[full_slot].device_data =
          next_pointer();
      ++full_slot;
    }
  }
  views.persistent.rope_cos_fp32.device_data = next_pointer();
  views.persistent.rope_sin_fp32.device_data = next_pointer();
  return views;
}

void test_layer_major_runner_view_binding_lifetime(TestContext& test) {
  using Peer = runtime::ReferenceRunnerPrefillControlTestPeer;
  std::array<std::uint64_t, 192U> identities{};
  const runtime::ReferenceLayerMajorRequestViews candidate =
      make_layer_major_runner_view_fixture(identities);

  runtime::ReferenceRunner runner = Peer::empty_runner();
  const runtime::ReferenceRunnerStatus status =
      Peer::bind_layer_major_views(runner, candidate);
  const Peer::CompactViews mapped = Peer::compact_views(runner);
  bool schedule_mapped = status.ok();
  for (std::size_t layer = 0U;
       layer < runtime::kReferenceDecoderLayerCount; ++layer) {
    const runtime::RequestLayerSlot slot =
        candidate.descriptor.layers[layer];
    if (slot.type == q3x::model::LayerType::kLinearAttention) {
      schedule_mapped =
          schedule_mapped &&
          mapped.conv_state[layer] == static_cast<std::uint16_t*>(
              candidate.persistent.conv_state_bf16[slot.slot].device_data) &&
          mapped.gdn_state[layer] == static_cast<std::uint16_t*>(
              candidate.persistent.gdn_state_bf16[slot.slot].device_data) &&
          mapped.key_cache[layer] == nullptr &&
          mapped.value_cache[layer] == nullptr;
    } else {
      schedule_mapped =
          schedule_mapped && mapped.conv_state[layer] == nullptr &&
          mapped.gdn_state[layer] == nullptr &&
          mapped.key_cache[layer] == static_cast<std::uint16_t*>(
              candidate.persistent.key_cache_bf16[slot.slot].device_data) &&
          mapped.value_cache[layer] == static_cast<std::uint16_t*>(
              candidate.persistent.value_cache_bf16[slot.slot].device_data);
    }
  }
  const runtime::ReferenceLayerMajorRequestViews* const cached =
      Peer::layer_major_views(runner);
  test.expect(
      schedule_mapped && Peer::has_layer_major_views(runner) &&
          mapped.hidden[0U] == static_cast<std::uint16_t*>(
              candidate.legacy_c512.hidden_bf16[0U].storage.device_data) &&
          mapped.projection[3U] == static_cast<std::uint16_t*>(
              candidate.legacy_c512.projection_bf16[3U].storage.device_data) &&
          mapped.linear_a == static_cast<std::uint16_t*>(
              candidate.legacy_c512.linear_a_bf16.storage.device_data) &&
          mapped.linear_b == static_cast<std::uint16_t*>(
              candidate.legacy_c512.linear_b_bf16.storage.device_data) &&
          mapped.fp32_scratch_elements ==
              runtime::kReferenceVocabularySize &&
          mapped.rope_cos == static_cast<const float*>(
              candidate.persistent.rope_cos_fp32.device_data) &&
          mapped.rope_sin == static_cast<const float*>(
              candidate.persistent.rope_sin_fp32.device_data) &&
          cached != nullptr &&
          cached->prompt_residual_bf16.storage.device_data ==
              candidate.prompt_residual_bf16.storage.device_data &&
          cached->final_hidden_bf16.storage.device_data ==
              candidate.final_hidden_bf16.storage.device_data,
      "layer-major binding caches the complete typed snapshot and maps only "
      "legacy C512 plus descriptor-indexed persistent views");

  runtime::ReferenceRunner moved(std::move(runner));
  test.expect(
      Peer::has_layer_major_views(moved) &&
          !Peer::has_layer_major_views(runner) &&
          Peer::compact_views(moved).hidden[0U] == mapped.hidden[0U] &&
          Peer::compact_views(runner).hidden[0U] == nullptr,
      "runner move transfers the typed layer-major binding and clears source");
  Peer::release(moved);
  test.expect(!Peer::has_layer_major_views(moved) &&
                  Peer::compact_views(moved).hidden[0U] == nullptr &&
                  Peer::compact_views(moved).rope_cos == nullptr,
              "runner release clears typed and compact layer-major views");

  runtime::ReferenceLayerMajorRequestViews malformed = candidate;
  ++malformed.descriptor.layers[3U].slot;
  runtime::ReferenceRunner rejected = Peer::empty_runner();
  const runtime::ReferenceRunnerStatus rejected_status =
      Peer::bind_layer_major_views(rejected, std::move(malformed));
  test.expect(!rejected_status &&
                  rejected_status.error ==
                      runtime::ReferenceRunnerError::kInvalidLayerSchedule &&
                  !Peer::has_layer_major_views(rejected) &&
                  Peer::compact_views(rejected).hidden[0U] == nullptr,
              "malformed layer-major schedule fails transactionally");
}

void test_whole_request_prefill_staging_contract(TestContext& test) {
  using Peer = runtime::ReferenceRunnerPrefillControlTestPeer;

  test.expect(
      Peer::whole_request_outcome_bytes() < 4U * 1'024U &&
          Peer::whole_request_outcome_bytes() * 8U <
              Peer::legacy_tile_outcome_bytes(),
      "whole-request runner result remains fixed-size and transcript-free");

  runtime::PrefillExecutionPlanOptions plan_options;
  plan_options.first_position = 17U;
  plan_options.prompt_token_count = 9'000U;
  plan_options.max_sequence_length = 32'768U;
  const runtime::PrefillExecutionPlanResult plan_result =
      runtime::build_unbound_layer_major_prefill_execution_plan(
          plan_options);
  test.expect(plan_result.ok(),
              "whole-request staging fixture builds a valid topology");
  if (!plan_result) {
    return;
  }
  const runtime::PrefillExecutionPlan& plan = *plan_result.value;

  runtime::PrefillExecutionProgress progress =
      runtime::make_prefill_execution_progress(plan);
  bool progress_ok = true;
  for (std::size_t layer = 0U;
       progress_ok && layer < runtime::kReferenceDecoderLayerCount; ++layer) {
    for (std::size_t panel = 0U;
         progress_ok && panel < plan.panel_count; ++panel) {
      progress_ok = runtime::advance_prefill_progress_after_completion(
                        plan, progress, layer, panel) ==
                    runtime::PrefillExecutionProgressError::kNone;
    }
  }
  progress_ok =
      progress_ok &&
      runtime::mark_prefill_final_hidden_ready(plan, progress) ==
          runtime::PrefillExecutionProgressError::kNone &&
      runtime::prefill_final_commit_ready(plan, progress);
  test.expect(progress_ok,
              "whole-request staging fixture reaches uncommitted final state");

  runtime::PrefillRouteEvidence live_route;
  runtime::reset_prefill_route_request(live_route);
  runtime::PrefillRouteEvidence staged_route = live_route;
  bool route_ok = true;
  for (std::size_t panel = 0U;
       route_ok && panel < plan.panel_count; ++panel) {
    runtime::PrefillRouteEvidence complete_layer_pass;
    for (std::size_t layer = 0U;
         route_ok && layer < runtime::kReferenceDecoderLayerCount; ++layer) {
      const Peer::LayerRouteFragment fragment =
          Peer::production_layer_route(layer, 0U, 512U);
      route_ok =
          Peer::collapse_layer_route(fragment, complete_layer_pass).ok();
    }
    route_ok =
        route_ok && runtime::commit_prefill_route_layer_pass(
                        staged_route, complete_layer_pass);
  }
  test.expect(
      route_ok && staged_route.completed_layer_passes == plan.panel_count,
      "one staged route pass accumulates all 64 layers for every C8192 panel");

  std::array<std::uint16_t, runtime::kReferenceHiddenSize> final_hidden{};
  runtime::ReferenceRunner runner = Peer::empty_runner();
  Peer::set_route_evidence(runner, live_route);
  Peer::seed_whole_request_stage(
      runner, plan.first_position, plan.final_position, 42U,
      plan.prompt_token_count, plan.panel_count, final_hidden.data(),
      progress, staged_route, false);
  test.expect(
      Peer::whole_request_stage_active(runner) &&
          Peer::whole_request_stage_matches(
              runner, plan.first_position, plan.final_position, 42U,
              plan.prompt_token_count, plan.panel_count,
              final_hidden.data(), progress, staged_route, false) &&
          same_prefill_route_evidence(Peer::route_evidence(runner),
                                      live_route),
      "staged progress and route evidence do not publish into live request state");

  runtime::ReferenceRunner moved(std::move(runner));
  test.expect(
      !Peer::whole_request_stage_active(runner) &&
          Peer::whole_request_stage_matches(
              moved, plan.first_position, plan.final_position, 42U,
              plan.prompt_token_count, plan.panel_count,
              final_hidden.data(), progress, staged_route, false) &&
          same_prefill_route_evidence(Peer::route_evidence(moved),
                                      live_route),
      "runner move transfers the whole-request hand-off exactly once");
  Peer::release(moved);
  test.expect(!Peer::whole_request_stage_active(moved),
              "runner release invalidates the whole-request hand-off");

  runtime::ReferenceRunner guarded = Peer::empty_runner();
  Peer::set_route_evidence(guarded, live_route);
  Peer::seed_whole_request_stage(
      guarded, plan.first_position, plan.final_position, 42U,
      plan.prompt_token_count, plan.panel_count, final_hidden.data(),
      progress, staged_route, true);
  const runtime::ReferenceRunnerStatus guard_status =
      guarded.record_scalar_prefill_route_fallback();
  test.expect(
      !guard_status && Peer::poisoned(guarded) &&
          !Peer::whole_request_stage_active(guarded) &&
          same_prefill_route_evidence(Peer::route_evidence(guarded),
                                      live_route),
      "a competing mutating operation fails closed without publishing staged route evidence");

  runtime::ReferenceRunner prematurely_finalized = Peer::empty_runner();
  Peer::set_route_evidence(prematurely_finalized, live_route);
  Peer::seed_whole_request_stage(
      prematurely_finalized, plan.first_position, plan.final_position, 42U,
      plan.prompt_token_count, plan.panel_count, final_hidden.data(),
      progress, staged_route, true);
  const runtime::PrefillRouteEvidence rejected =
      prematurely_finalized.finalize_prefill_route_evidence(
          plan.panel_count);
  test.expect(
      rejected.complete && !rejected.valid && !rejected.request_active &&
          rejected.error ==
              runtime::PrefillRouteEvidenceError::kIncompleteTile &&
          Peer::poisoned(prematurely_finalized) &&
          !Peer::whole_request_stage_active(prematurely_finalized) &&
          same_prefill_route_evidence(
              Peer::route_evidence(prematurely_finalized), live_route),
      "premature route finalization aborts the staged transaction without publishing it");
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
  test_prefill_tile_execution_control(test);
  test_prefill_layer_route_reducer(test);
  test_prefill_route_evidence_runner_lifetime(test);
  test_layer_major_runner_view_binding_lifetime(test);
  test_whole_request_prefill_staging_contract(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " reference-runner host test(s) failed\n";
    return 1;
  }
  std::cout << "Reference runner host tests passed\n";
  return 0;
}
