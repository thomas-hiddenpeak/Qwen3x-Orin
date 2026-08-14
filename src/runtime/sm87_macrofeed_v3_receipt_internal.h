#pragma once

#include "q3x/runtime/sm87_macrofeed_v3_receipt.h"

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::macrofeed_v3_receipt_detail {

// Non-installed issuer boundary.  These helpers exist so the future private
// V3 executor can issue evidence only after consuming owner-backed, real
// constituent receipts and real completion events.  Calling them with
// caller-authored observations, as the T0 host test deliberately does, proves
// only the structural contract and never execution or production authority.
[[nodiscard]] std::uint64_t compute_layer_receipt_identity(
    const Sm87MacroFeedV3LayerReceipt& receipt) noexcept;

// Returns a zero receipt when any layer-local observation is incomplete or
// inconsistent with the fixed V3 physical topology.
[[nodiscard]] Sm87MacroFeedV3LayerReceipt seal_layer_receipt(
    const Sm87MacroFeedV3LayerReceipt& observed) noexcept;

// Accepts real layer receipts in any order, rejects duplicates or omissions,
// and stores them canonically by layer index.  A false request completion
// observation can issue only a coherent incomplete transaction.
[[nodiscard]] bool build_transaction_receipt(
    std::uint64_t request_identity,
    const Sm87MacroFeedV3LayerReceipt* layer_receipts,
    std::size_t layer_receipt_count, bool request_completion_observed,
    Sm87MacroFeedV3TransactionReceipt* transaction) noexcept;

}  // namespace q3x::runtime::macrofeed_v3_receipt_detail
