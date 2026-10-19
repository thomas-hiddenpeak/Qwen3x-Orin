#pragma once
#include "mtp_control.h"

namespace q3x::runtime::mtp_detail {
inline constexpr std::uint32_t kTreeEntryParent = kMaximumVerifyRows;
using TreeParents = std::array<std::uint32_t, kMaximumVerifyRows>;

struct BudgetedProposal {
  std::uint32_t count = 0;
  std::array<std::uint32_t, kMaximumDraftLength> tokens{};
  TreeParents parents{kTreeEntryParent, kTreeEntryParent, kTreeEntryParent, kTreeEntryParent};
  std::array<std::uint32_t, kMaximumVerifyRows> depth{};
  bool alternate_selected = false;
};
// Fixed rule: replace only the primary tail when the alternate first-token
// log probability exceeds the full primary path log probability. Ties stay
// linear. Failure leaves output unchanged; no allocation or implicit fallback.
bool plan_budgeted_proposal(const std::uint32_t* primary,
    const double* primary_log_probability, std::uint32_t count,
    std::uint32_t alternate, double alternate_log_probability,
    std::uint32_t vocabulary, BudgetedProposal& output) noexcept;

// Same <=4 physical nodes as the linear controller. Only the last proposal
// may instead be a sibling of node1. Parent map is frozen through finish/abort.
class TreeRoundBackend : public RoundBackend {
 public:
  virtual bool proposal_parents(std::uint32_t count, TreeParents&) const noexcept = 0;
  // node is physical; depth is the number of consumed nodes on the committed
  // path. Publish complete target AND reconciled draft state before returning.
  virtual bool commit_node(std::uint32_t node, std::uint32_t depth,
                           std::uint32_t pending_token) noexcept = 0;
};

// Isolated prerequisite: existing run_round and service are unchanged.
[[nodiscard]] RoundResult run_tree_round(const RoundOptions&, TreeRoundBackend&,
    TokenObserver = nullptr, void* observer_context = nullptr) noexcept;
}  // namespace q3x::runtime::mtp_detail
