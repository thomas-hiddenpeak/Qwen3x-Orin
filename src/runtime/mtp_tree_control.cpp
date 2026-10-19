#include "mtp_tree_control.h"
#include <algorithm>
#include <cmath>

namespace q3x::runtime::mtp_detail {
bool plan_budgeted_proposal(const std::uint32_t* primary, const double* logp,
    std::uint32_t count, std::uint32_t alternate, double alternate_logp,
    std::uint32_t vocabulary, BudgetedProposal& output) noexcept {
  if (!primary || !logp || count == 0 || count > kMaximumDraftLength ||
      vocabulary < 2 || alternate >= vocabulary ||
      !std::isfinite(alternate_logp) || alternate_logp > 0) return false;
  BudgetedProposal next;
  next.count = count;
  double joint = 0;
  for (std::uint32_t i = 0; i < count; ++i) {
    if (primary[i] >= vocabulary || !std::isfinite(logp[i]) || logp[i] > 0)
      return false;
    next.tokens[i] = primary[i]; next.parents[i + 1] = i;
    next.depth[i + 1] = i + 1; joint += logp[i];
  }
  if (!std::isfinite(joint) || alternate == primary[0] || alternate_logp > logp[0])
    return false;
  if (count >= 2 && alternate_logp > joint) {
    next.tokens[count - 1] = alternate; next.parents[count] = 0;
    next.depth[count] = 1; next.alternate_selected = true;
  }
  output = next;
  return true;
}

RoundResult run_tree_round(const RoundOptions& options, TreeRoundBackend& backend,
    TokenObserver observer, void* observer_context) noexcept {
  RoundResult result;
  if ((options.draft_length != 2 && options.draft_length != 3) ||
      options.vocabulary_size == 0 || options.seed_token >= options.vocabulary_size ||
      options.stop_token >= options.vocabulary_size ||
      options.seed_token == options.stop_token || options.remaining_output == 0 ||
      options.available_target_rows < options.remaining_output ||
      (!observer && observer_context)) return result;
  auto count = std::min(options.draft_length, options.remaining_output - 1);
  std::array<std::uint32_t, kMaximumDraftLength> draft{};
  std::array<std::uint32_t, kMaximumVerifyRows> predictions{};
  TreeParents parents{};
  const auto fail = [&](RoundStatus status) noexcept {
    result.status = status; result.abort_succeeded = backend.abort(); return result;
  };
  if (!backend.begin(count + 1)) return fail(RoundStatus::kBackendFailure);
  if (count) {
    const auto maximum = count;
    const bool ok = options.confidence_lookahead
        ? backend.propose_bounded(options.seed_token, maximum, draft.data(), count)
        : backend.propose(options.seed_token, maximum, draft.data());
    if (!ok) return fail(RoundStatus::kBackendFailure);
    if (!count || count > maximum) return fail(RoundStatus::kInvalidPrediction);
  }
  result.proposed_tokens = count;
  for (std::uint32_t i = 0; i < count; ++i)
    if (draft[i] >= options.vocabulary_size) return fail(RoundStatus::kInvalidPrediction);
  if (!backend.proposal_parents(count, parents)) return fail(RoundStatus::kBackendFailure);
  if (parents[0] != kTreeEntryParent) return fail(RoundStatus::kInvalidPrediction);
  for (std::uint32_t i = 1; i <= count; ++i) {
    if (parents[i] != i - 1 && !(i == count && count >= 2 && parents[i] == 0))
      return fail(RoundStatus::kInvalidPrediction);
    for (std::uint32_t j = 1; j < i; ++j)
      if (parents[i] == parents[j] && draft[i - 1] == draft[j - 1])
        return fail(RoundStatus::kInvalidPrediction);
  }
  if (!backend.verify(options.seed_token, draft.data(), count, predictions.data()))
    return fail(RoundStatus::kBackendFailure);
  result.verified_rows = count + 1;
  for (std::uint32_t i = 0; i <= count; ++i)
    if (predictions[i] >= options.vocabulary_size) return fail(RoundStatus::kInvalidPrediction);

  std::uint32_t node = 0;
  for (std::uint32_t depth = 1; depth <= count + 1; ++depth) {
    const auto token = predictions[node];
    auto child = kTreeEntryParent;
    for (std::uint32_t j = node + 1; j <= count; ++j)
      if (parents[j] == node && draft[j - 1] == token) { child = j; break; }
    if (!backend.commit_node(node, depth, token)) return fail(RoundStatus::kBackendFailure);
    result.committed_target_rows = depth;
    result.output[result.output_count++] = token;
    result.accepted_tokens += child != kTreeEntryParent ? 1U : 0U;
    const bool proceed = !observer || observer(observer_context, token);
    if (token == options.stop_token) result.status = RoundStatus::kStop;
    else if (!proceed) result.status = RoundStatus::kCancelled;
    else if (result.output_count == options.remaining_output) result.status = RoundStatus::kLength;
    else result.status = RoundStatus::kContinue;
    if (result.status != RoundStatus::kContinue || child == kTreeEntryParent) break;
    node = child;
  }
  if (!backend.finish()) return fail(RoundStatus::kBackendFailure);
  return result;
}
}  // namespace q3x::runtime::mtp_detail
