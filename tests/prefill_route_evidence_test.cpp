#include "q3x/runtime/prefill_route_evidence.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

using q3x::runtime::PrefillForbiddenBoundary;
using q3x::runtime::PrefillOperatorRole;
using q3x::runtime::PrefillRouteDisposition;
using q3x::runtime::PrefillRouteEvidence;
using q3x::runtime::PrefillRouteEvidenceError;

[[nodiscard]] PrefillRouteEvidence complete_tile(
    const PrefillRouteDisposition disposition) {
  PrefillRouteEvidence tile;
  for (std::size_t index = 0U;
       index < q3x::runtime::kPrefillOperatorRoleCount; ++index) {
    if (!q3x::runtime::record_prefill_operator_route(
            tile, static_cast<PrefillOperatorRole>(index), disposition,
            q3x::runtime::kExpectedPrefillLogicalOperatorsPerTile[index])) {
      std::cerr << "failed to construct tile\n";
      return {};
    }
  }
  return tile;
}

}  // namespace

int main() {
  PrefillRouteEvidence request;
  q3x::runtime::reset_prefill_route_request(request);
  const PrefillRouteEvidence production =
      complete_tile(PrefillRouteDisposition::kProduction);
  const PrefillRouteEvidence fallback =
      complete_tile(PrefillRouteDisposition::kExactFallback);
  if (!q3x::runtime::commit_prefill_route_layer_pass(request, production) ||
      !q3x::runtime::commit_prefill_route_layer_pass(request, fallback)) {
    std::cerr << "complete tiles did not merge\n";
    return 1;
  }
  const PrefillRouteEvidence complete =
      q3x::runtime::finalize_prefill_route_request(request, 2U);
  if (!complete.valid || !complete.complete || complete.request_active ||
      complete.completed_layer_passes != 2U ||
      complete.expected_layer_passes != 2U) {
    std::cerr << "valid request did not finalize\n";
    return 1;
  }
  for (std::size_t index = 0U; index < complete.operators.size(); ++index) {
    if (complete.operators[index].production_hits !=
            q3x::runtime::kExpectedPrefillLogicalOperatorsPerTile[index] ||
        complete.operators[index].exact_fallback_hits !=
            q3x::runtime::kExpectedPrefillLogicalOperatorsPerTile[index] ||
        complete.operators[index].forbidden_hits != 0U) {
      std::cerr << "operator totals are wrong\n";
      return 1;
    }
  }

  // A reset is the request-isolation boundary. Nothing from the prior request
  // may survive, including its finalized status.
  q3x::runtime::reset_prefill_route_request(request);
  if (!request.request_active || request.complete || request.valid ||
      request.completed_layer_passes != 0U ||
      request.expected_layer_passes != 0U ||
      request.error != PrefillRouteEvidenceError::kNone) {
    std::cerr << "request reset did not isolate state\n";
    return 1;
  }
  for (const auto& counts : request.operators) {
    if (counts.production_hits != 0U || counts.exact_fallback_hits != 0U ||
        counts.forbidden_hits != 0U) {
      std::cerr << "request reset retained operator hits\n";
      return 1;
    }
  }

  PrefillRouteEvidence incomplete_tile = production;
  --incomplete_tile.operators[0].production_hits;
  if (q3x::runtime::commit_prefill_route_layer_pass(request,
                                                    incomplete_tile) ||
      request.error != PrefillRouteEvidenceError::kIncompleteTile) {
    std::cerr << "incomplete tile did not fail closed\n";
    return 1;
  }

  q3x::runtime::reset_prefill_route_request(request);
  PrefillRouteEvidence forbidden = production;
  forbidden.operators[0].production_hits = 63U;
  forbidden.operators[0].forbidden_hits = 1U;
  if (!q3x::runtime::record_prefill_forbidden_boundary(
          forbidden, PrefillForbiddenBoundary::kApproximateNumerics) ||
      !q3x::runtime::commit_prefill_route_layer_pass(request, forbidden)) {
    std::cerr << "forbidden tile could not be retained for diagnosis\n";
    return 1;
  }
  const PrefillRouteEvidence forbidden_result =
      q3x::runtime::finalize_prefill_route_request(request, 1U);
  if (forbidden_result.valid ||
      forbidden_result.error != PrefillRouteEvidenceError::kForbiddenRoute) {
    std::cerr << "forbidden route did not invalidate evidence\n";
    return 1;
  }

  q3x::runtime::reset_prefill_route_request(request);
  if (!q3x::runtime::commit_prefill_route_layer_pass(request, production)) {
    return 1;
  }
  const PrefillRouteEvidence wrong_count =
      q3x::runtime::finalize_prefill_route_request(request, 2U);
  if (wrong_count.valid || wrong_count.error !=
                               PrefillRouteEvidenceError::
                                   kUnexpectedLayerPassCount) {
    std::cerr << "tile-count mismatch did not fail closed\n";
    return 1;
  }

  PrefillRouteEvidence overflow;
  overflow.operators[0].production_hits =
      std::numeric_limits<std::uint64_t>::max();
  if (q3x::runtime::record_prefill_operator_route(
          overflow, PrefillOperatorRole::kNvFp4GateUp,
          PrefillRouteDisposition::kProduction) ||
      overflow.error != PrefillRouteEvidenceError::kCounterOverflow) {
    std::cerr << "counter overflow did not fail closed\n";
    return 1;
  }

  PrefillRouteEvidence zero_tile_request;
  q3x::runtime::reset_prefill_route_request(zero_tile_request);
  const PrefillRouteEvidence zero =
      q3x::runtime::finalize_prefill_route_request(zero_tile_request, 0U);
  if (!zero.valid || zero.completed_layer_passes != 0U ||
      zero.expected_layer_passes != 0U) {
    std::cerr << "zero-prefix request did not finalize\n";
    return 1;
  }

  return 0;
}
