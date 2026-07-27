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
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace runtime = q3x::runtime;

constexpr std::array<std::uint32_t, 19U> kPromptIds = {
    248045U, 846U,    198U,    95826U,  110827U,
    98682U,  52965U,  220U,    98661U,  1710U,
    248046U, 198U,    248045U, 74455U,  198U,
    248068U, 271U,    248069U, 271U,
};

constexpr std::array<std::uint32_t, 26U> kGeneratedIds = {
    77517U,  220U,    95761U,  32449U,  220U,    97039U,  101692U,
    119548U, 97792U,  125574U, 102027U, 103725U, 3709U,   109238U,
    97327U,  21966U,  220U,    127361U, 130111U, 95860U,  103806U,
    108751U, 97792U,  97995U,  1710U,   248046U,
};

constexpr std::size_t kMeasuredDrafts = kGeneratedIds.size() - 1U;
constexpr double kM2VerifierEstimateMilliseconds = 151.85;
constexpr double kContinuationDraftMilliseconds = 25.0;
constexpr double kAbsoluteDraftCeilingMilliseconds = 48.15;
constexpr int kSkipReturnCode = 77;

std::string model_directory_from(const int argc, char** const argv) {
  if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0' &&
      std::string_view(argv[1]) != "-") {
    return argv[1];
  }
  const char* environment =
      std::getenv("Q3X_MTP_DRAFT_P1_MODEL_DIR");
  if (environment != nullptr && environment[0] != '\0') {
    return environment;
  }
  environment = std::getenv("Q3X_E2E_MODEL_DIR");
  return environment == nullptr ? std::string{} : std::string(environment);
}

void print_bool(const std::string_view key, const bool value) {
  std::cout << key << '=' << (value ? 1 : 0) << '\n';
}

void print_resident_diagnostic(
    const std::string_view stage,
    const runtime::ResidentLoadDiagnostic& diagnostic) {
  std::cerr << stage << " failed: code=" << runtime::to_string(diagnostic.code)
            << " message=" << diagnostic.message
            << " context=" << diagnostic.context
            << " shard=" << diagnostic.shard
            << " cuda_error=" << diagnostic.cuda_error << '\n';
}

void print_bind_diagnostic(
    const std::string_view stage,
    const runtime::WeightBindDiagnostic& diagnostic) {
  std::cerr << stage << " failed: code=" << runtime::to_string(diagnostic.code)
            << " tensor=" << diagnostic.tensor
            << " message=" << diagnostic.message
            << " cuda_error=" << diagnostic.cuda_error << '\n';
}

void print_request_diagnostic(
    const runtime::RequestDiagnostic& diagnostic) {
  std::cerr << "request state failed: code="
            << runtime::to_string(diagnostic.code)
            << " message=" << diagnostic.message
            << " context=" << diagnostic.context
            << " cuda_error=" << diagnostic.cuda_error << '\n';
}

void print_runner_diagnostic(
    const std::string_view stage,
    const runtime::ReferenceRunnerStatus& diagnostic) {
  std::cerr << stage << " failed: code="
            << runtime::reference_runner_error_string(diagnostic.error)
            << " operation="
            << (diagnostic.operation == nullptr ? "" : diagnostic.operation)
            << " layer=" << diagnostic.layer
            << " cuda_error=" << diagnostic.cuda_error << '\n';
}

double percentile(std::vector<double> values, const double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double scaled =
      fraction * static_cast<double>(values.size());
  std::size_t index =
      static_cast<std::size_t>(std::ceil(scaled));
  index = index == 0U ? 0U : index - 1U;
  return values[std::min(index, values.size() - 1U)];
}

}  // namespace

int main(const int argc, char** const argv) {
  const char* const enabled =
      std::getenv("Q3X_RUN_MTP_DRAFT_P1_PERF");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "mtp_draft_p1.status=skip\n"
              << "SKIP: set Q3X_RUN_MTP_DRAFT_P1_PERF=1 to run the "
                 "full-model MTP-1 shadow screen\n";
    return kSkipReturnCode;
  }
  if (argc > 2) {
    std::cerr << "usage: q3x_reference_mtp_draft_p1_perf_test "
                 "[MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cerr << "Q3X_RUN_MTP_DRAFT_P1_PERF=1 requires MODEL_DIR, "
                 "Q3X_MTP_DRAFT_P1_MODEL_DIR, or Q3X_E2E_MODEL_DIR\n";
    return 2;
  }

  int device = 0;
  (void)cudaGetLastError();
  cudaError_t cuda_status = cudaGetDevice(&device);
  cudaDeviceProp properties{};
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaGetDeviceProperties(&properties, device);
  }
  if (cuda_status != cudaSuccess) {
    std::cerr << "failed to inspect CUDA device: "
              << cudaGetErrorString(cuda_status) << '\n';
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "mtp_draft_p1.status=skip\n"
              << "SKIP: MTP draft P1 requires an SM87 device\n";
    return kSkipReturnCode;
  }

  runtime::ResidentLoadResult text_resident =
      runtime::load_pinned_qwen36_27b(model_directory);
  if (!text_resident) {
    print_resident_diagnostic("text resident load",
                              text_resident.diagnostic);
    return 1;
  }
  runtime::ResidentLoadResult mtp_resident =
      runtime::load_pinned_qwen36_27b_mtp(model_directory);
  if (!mtp_resident) {
    print_resident_diagnostic("MTP resident load", mtp_resident.diagnostic);
    return 1;
  }
  runtime::WeightBindResult text_weights =
      runtime::bind_qwen36_27b_weights(*text_resident.value);
  if (!text_weights) {
    print_bind_diagnostic("text weight bind", text_weights.diagnostic);
    return 1;
  }
  runtime::MtpWeightBindResult mtp_weights =
      runtime::bind_qwen36_27b_mtp_weights(*mtp_resident.value);
  if (!mtp_weights) {
    print_bind_diagnostic("MTP weight bind", mtp_weights.diagnostic);
    return 1;
  }

  runtime::RequestMemoryOptions request_options;
  request_options.max_sequence_length = 64U;
  request_options.prefill_chunk_size = 1U;
  runtime::RequestStateResult request =
      runtime::create_request_state(request_options);
  if (!request) {
    print_request_diagnostic(request.diagnostic);
    return 1;
  }
  runtime::ReferenceRunnerOptions runner_options;
  runner_options.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  runtime::ReferenceRunnerFactoryResult runner =
      runtime::create_reference_runner(
          *text_weights.value, *mtp_weights.value, *request.value,
          runner_options);
  if (!runner) {
    print_runner_diagnostic("runner factory", runner.diagnostic);
    return 1;
  }

  runtime::ReferenceStepOptions prefix_options;
  prefix_options.compute_logits = false;
  runtime::ReferenceMtpDraftOptions prompt_draft_options;
  prompt_draft_options.compute_logits = false;
  for (std::size_t index = 0U; index + 1U < kPromptIds.size(); ++index) {
    runtime::ReferenceStepOutcome base =
        runner.value->step(kPromptIds[index], prefix_options);
    if (!base) {
      print_runner_diagnostic("prompt base step", base.status);
      return 1;
    }
    runtime::ReferenceMtpDraftOutcome draft =
        runner.value->shadow_mtp_draft(kPromptIds[index + 1U],
                                       prompt_draft_options);
    if (!draft) {
      print_runner_diagnostic("prompt MTP transition", draft.status);
      return 1;
    }
    if (base.value->position != index || draft.value->position != index) {
      std::cerr << "prompt position mismatch at transition " << index << '\n';
      return 1;
    }
  }

  runtime::ReferenceStepOptions base_decode_options;
  base_decode_options.compute_logits = true;
  base_decode_options.logits_mode =
      runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  runtime::ReferenceStepOutcome final_prompt =
      runner.value->step(kPromptIds.back(), base_decode_options);
  if (!final_prompt) {
    print_runner_diagnostic("final prompt base step", final_prompt.status);
    return 1;
  }
  if (!final_prompt.value->prediction ||
      final_prompt.value->prediction->predicted_token_id !=
          kGeneratedIds.front()) {
    std::cerr << "final prompt base prediction differs from pinned oracle\n";
    return 1;
  }

  runtime::ReferenceMtpDraftOptions draft_options;
  draft_options.compute_logits = true;
  runtime::ReferenceMtpDraftOutcome initial_draft =
      runner.value->shadow_mtp_draft(kGeneratedIds.front(), draft_options);
  if (!initial_draft || !initial_draft.value->prediction) {
    print_runner_diagnostic("initial decode MTP draft", initial_draft.status);
    return 1;
  }

  std::uint32_t pending_draft =
      initial_draft.value->prediction->predicted_token_id;
  std::vector<std::uint32_t> draft_tokens;
  draft_tokens.reserve(kGeneratedIds.size());
  draft_tokens.push_back(pending_draft);
  std::vector<double> draft_milliseconds;
  draft_milliseconds.reserve(kMeasuredDrafts);
  std::size_t accepted = 0U;
  bool base_oracle_exact = true;
  bool positions_exact = true;

  for (std::size_t index = 0U; index < kMeasuredDrafts; ++index) {
    runtime::ReferenceStepOutcome base =
        runner.value->step(kGeneratedIds[index], base_decode_options);
    if (!base || !base.value->prediction) {
      print_runner_diagnostic("decode base step", base.status);
      return 1;
    }
    const std::uint32_t expected_next = kGeneratedIds[index + 1U];
    const std::uint32_t actual_next =
        base.value->prediction->predicted_token_id;
    base_oracle_exact = base_oracle_exact && actual_next == expected_next;
    positions_exact = positions_exact &&
                      base.value->position == kPromptIds.size() + index;
    if (pending_draft == actual_next) {
      ++accepted;
    }

    runtime::ReferenceMtpDraftOptions measured_options;
    measured_options.compute_logits = true;
    measured_options.measure_timing = true;
    runtime::ReferenceMtpDraftOutcome draft =
        runner.value->shadow_mtp_draft(actual_next, measured_options);
    if (!draft || !draft.value->prediction || !draft.value->timing) {
      print_runner_diagnostic("measured MTP draft", draft.status);
      return 1;
    }
    positions_exact = positions_exact &&
                      draft.value->position ==
                          kPromptIds.size() + index;
    const double milliseconds =
        draft.value->timing->elapsed_milliseconds;
    if (!std::isfinite(milliseconds) || milliseconds < 0.0) {
      std::cerr << "non-finite measured MTP latency\n";
      return 1;
    }
    draft_milliseconds.push_back(milliseconds);
    pending_draft = draft.value->prediction->predicted_token_id;
    draft_tokens.push_back(pending_draft);
  }

  const double median_milliseconds = percentile(draft_milliseconds, 0.50);
  const double p95_milliseconds = percentile(draft_milliseconds, 0.95);
  const double acceptance =
      static_cast<double>(accepted) / static_cast<double>(kMeasuredDrafts);
  const double required_acceptance =
      (kM2VerifierEstimateMilliseconds + median_milliseconds) / 100.0 - 1.0;
  const double projected_milliseconds_per_token =
      (kM2VerifierEstimateMilliseconds + median_milliseconds) /
      (1.0 + acceptance);
  const bool state_exact =
      request.value->sequence_length() == 44U &&
      runner.value->current_position() == 44U;
  bool reset_replay_exact = false;
  const runtime::ReferenceRunnerStatus reset_status = runner.value->reset();
  if (!reset_status) {
    print_runner_diagnostic("MTP runner reset", reset_status);
  } else {
    runtime::ReferenceStepOutcome reset_base =
        runner.value->step(kPromptIds[0], prefix_options);
    if (!reset_base) {
      print_runner_diagnostic("post-reset base step", reset_base.status);
    } else {
      runtime::ReferenceMtpDraftOutcome reset_draft =
          runner.value->shadow_mtp_draft(kPromptIds[1],
                                         prompt_draft_options);
      if (!reset_draft) {
        print_runner_diagnostic("post-reset MTP draft", reset_draft.status);
      } else {
        reset_replay_exact = reset_base.value->position == 0U &&
                             reset_draft.value->position == 0U &&
                             request.value->sequence_length() == 1U;
      }
    }
  }
  const bool correctness_gate = base_oracle_exact && positions_exact &&
                                state_exact && reset_replay_exact &&
                                draft_milliseconds.size() == kMeasuredDrafts;
  const bool continuation_gate =
      median_milliseconds <= kContinuationDraftMilliseconds;
  const bool absolute_ceiling_gate =
      median_milliseconds < kAbsoluteDraftCeilingMilliseconds;
  const bool target_projection_gate =
      acceptance >= required_acceptance &&
      projected_milliseconds_per_token <= 100.0;
  const bool passed = correctness_gate && continuation_gate &&
                      absolute_ceiling_gate && target_projection_gate;

  std::cout << std::fixed << std::setprecision(6);
  for (std::size_t index = 0U; index < draft_milliseconds.size(); ++index) {
    std::cout << "mtp_draft_p1.sample." << index
              << ".position=" << kPromptIds.size() + index << '\n'
              << "mtp_draft_p1.sample." << index
              << ".milliseconds=" << draft_milliseconds[index] << '\n';
  }
  for (std::size_t index = 0U; index < draft_tokens.size(); ++index) {
    std::cout << "mtp_draft_p1.draft_token." << index
              << '=' << draft_tokens[index] << '\n';
  }
  std::cout << "mtp_draft_p1.text_arena_bytes="
            << text_resident.value->size_bytes() << '\n'
            << "mtp_draft_p1.mtp_arena_bytes="
            << mtp_resident.value->size_bytes() << '\n'
            << "mtp_draft_p1.samples=" << draft_milliseconds.size() << '\n'
            << "mtp_draft_p1.accepted=" << accepted << '\n'
            << "mtp_draft_p1.acceptance=" << acceptance << '\n'
            << "mtp_draft_p1.draft_median_ms=" << median_milliseconds << '\n'
            << "mtp_draft_p1.draft_p95_ms=" << p95_milliseconds << '\n'
            << "mtp_draft_p1.m2_estimate_ms="
            << kM2VerifierEstimateMilliseconds << '\n'
            << "mtp_draft_p1.required_acceptance="
            << required_acceptance << '\n'
            << "mtp_draft_p1.projected_ms_per_token="
            << projected_milliseconds_per_token << '\n';
  print_bool("mtp_draft_p1.base_oracle_exact", base_oracle_exact);
  print_bool("mtp_draft_p1.positions_exact", positions_exact);
  print_bool("mtp_draft_p1.state_exact", state_exact);
  print_bool("mtp_draft_p1.reset_replay_exact", reset_replay_exact);
  print_bool("mtp_draft_p1.gate.correctness", correctness_gate);
  print_bool("mtp_draft_p1.gate.continuation", continuation_gate);
  print_bool("mtp_draft_p1.gate.absolute_ceiling", absolute_ceiling_gate);
  print_bool("mtp_draft_p1.gate.target_projection", target_projection_gate);
  print_bool("mtp_draft_p1.gate.pass", passed);
  print_bool("mtp_draft_p1.production_admitted", false);
  std::cout << "mtp_draft_p1.status="
            << (passed ? "p1_projected_pass"
                       : correctness_gate ? "stop_loss"
                                          : "correctness_failure")
            << '\n';
  return passed ? 0 : 1;
}
