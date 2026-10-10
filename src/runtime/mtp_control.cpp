#include "mtp_control.h"

#include <algorithm>

namespace q3x::runtime::mtp_detail {

RoundResult run_round(const RoundOptions& options, RoundBackend& backend,
                      const TokenObserver observer,
                      void* const observer_context) noexcept {
  RoundResult result;
  if ((options.draft_length != 2 && options.draft_length != 3) ||
      options.vocabulary_size == 0 ||
      options.seed_token >= options.vocabulary_size ||
      options.stop_token >= options.vocabulary_size ||
      options.seed_token == options.stop_token ||
      options.remaining_output == 0 ||
      options.available_target_rows < options.remaining_output ||
      (observer == nullptr && observer_context != nullptr)) {
    return result;
  }
  // There must be room for the correction/bonus token. This also makes the
  // final one-token tail an ordinary target step, without unnecessary drafting.
  std::uint32_t count =
      std::min(options.draft_length, options.remaining_output - 1);
  std::array<std::uint32_t, kMaximumDraftLength> draft{};
  std::array<std::uint32_t, kMaximumVerifyRows> predictions{};
  const auto fail = [&](const RoundStatus status) noexcept {
    result.status = status;
    result.abort_succeeded = backend.abort();
    return result;
  };
  if (!backend.begin(count + 1)) return fail(RoundStatus::kBackendFailure);
  if (count != 0) {
    const auto maximum = count;
    const bool ok = options.confidence_lookahead
        ? backend.propose_bounded(options.seed_token, maximum, draft.data(), count)
        : backend.propose(options.seed_token, maximum, draft.data());
    if (!ok) return fail(RoundStatus::kBackendFailure);
    if (count == 0 || count > maximum) return fail(RoundStatus::kInvalidPrediction);
  }
  result.proposed_tokens = count;
  for (std::uint32_t i = 0; i < count; ++i) {
    if (draft[i] >= options.vocabulary_size) {
      return fail(RoundStatus::kInvalidPrediction);
    }
  }
  if (!backend.verify(options.seed_token, draft.data(), count,
                      predictions.data())) {
    return fail(RoundStatus::kBackendFailure);
  }
  result.verified_rows = count + 1;
  for (std::uint32_t i = 0; i <= count; ++i) {
    if (predictions[i] >= options.vocabulary_size) {
      return fail(RoundStatus::kInvalidPrediction);
    }
  }

  for (std::uint32_t i = 0; i <= count; ++i) {
    const bool accepted = i < count && draft[i] == predictions[i];
    const std::uint32_t token = predictions[i];
    // At row i, every earlier proposal was accepted. Later predictions become
    // invalid immediately on a mismatch, even if they happen to match drafts.
    if (!backend.commit_prefix(i + 1, token)) {
      return fail(RoundStatus::kBackendFailure);
    }
    result.committed_target_rows = i + 1;
    result.output[result.output_count++] = token;
    result.accepted_tokens += accepted ? 1U : 0U;
    const bool proceed = observer == nullptr || observer(observer_context, token);
    // Match the engine contract: semantic EOS takes precedence over cancel.
    if (token == options.stop_token) {
      result.status = RoundStatus::kStop;
    } else if (!proceed) {
      result.status = RoundStatus::kCancelled;
    } else if (result.output_count == options.remaining_output) {
      result.status = RoundStatus::kLength;
    } else {
      result.status = RoundStatus::kContinue;
    }
    if (result.status != RoundStatus::kContinue || !accepted) break;
  }
  if (!backend.finish()) return fail(RoundStatus::kBackendFailure);
  return result;
}

}  // namespace q3x::runtime::mtp_detail
