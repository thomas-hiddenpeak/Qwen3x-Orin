#include "q3x/runtime/prefill_route_evidence.h"

#include <limits>

namespace q3x::runtime {
namespace {

[[nodiscard]] bool checked_add(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

void fail(PrefillRouteEvidence& evidence,
          const PrefillRouteEvidenceError error) noexcept {
  evidence.valid = false;
  if (evidence.error == PrefillRouteEvidenceError::kNone) {
    evidence.error = error;
  }
}

[[nodiscard]] bool total(const PrefillOperatorRouteCounts& counts,
                         std::uint64_t& output) noexcept {
  std::uint64_t partial = 0U;
  return checked_add(counts.production_hits, counts.exact_fallback_hits,
                     partial) &&
         checked_add(partial, counts.forbidden_hits, output);
}

[[nodiscard]] bool layer_pass_is_complete(
    const PrefillRouteEvidence& layer_pass) noexcept {
  if (layer_pass.error != PrefillRouteEvidenceError::kNone ||
      layer_pass.request_active || layer_pass.complete || layer_pass.valid ||
      layer_pass.completed_layer_passes != 0U ||
      layer_pass.expected_layer_passes != 0U) {
    return false;
  }
  for (std::size_t index = 0U;
       index < layer_pass.operators.size(); ++index) {
    std::uint64_t observed = 0U;
    if (!total(layer_pass.operators[index], observed) ||
        observed != kExpectedPrefillLogicalOperatorsPerTile[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool has_forbidden_hits(
    const PrefillRouteEvidence& evidence) noexcept {
  for (const PrefillOperatorRouteCounts& counts : evidence.operators) {
    if (counts.forbidden_hits != 0U) {
      return true;
    }
  }
  for (const std::uint64_t hits : evidence.forbidden_boundary_hits) {
    if (hits != 0U) {
      return true;
    }
  }
  return false;
}

}  // namespace

void reset_prefill_route_request(PrefillRouteEvidence& evidence) noexcept {
  evidence = {};
  evidence.request_active = true;
}

bool record_prefill_operator_route(
    PrefillRouteEvidence& evidence, const PrefillOperatorRole role,
    const PrefillRouteDisposition disposition, const std::uint64_t count) noexcept {
  const std::size_t role_index = static_cast<std::size_t>(role);
  if (role_index >= evidence.operators.size()) {
    fail(evidence, PrefillRouteEvidenceError::kInvalidRole);
    return false;
  }
  std::uint64_t* destination = nullptr;
  switch (disposition) {
    case PrefillRouteDisposition::kProduction:
      destination = &evidence.operators[role_index].production_hits;
      break;
    case PrefillRouteDisposition::kExactFallback:
      destination = &evidence.operators[role_index].exact_fallback_hits;
      break;
    case PrefillRouteDisposition::kForbidden:
      destination = &evidence.operators[role_index].forbidden_hits;
      break;
    default:
      fail(evidence, PrefillRouteEvidenceError::kInvalidDisposition);
      return false;
  }
  std::uint64_t updated = 0U;
  if (!checked_add(*destination, count, updated)) {
    fail(evidence, PrefillRouteEvidenceError::kCounterOverflow);
    return false;
  }
  *destination = updated;
  return true;
}

bool record_prefill_forbidden_boundary(
    PrefillRouteEvidence& evidence, const PrefillForbiddenBoundary boundary,
    const std::uint64_t count) noexcept {
  const std::size_t index = static_cast<std::size_t>(boundary);
  if (index >= evidence.forbidden_boundary_hits.size()) {
    fail(evidence, PrefillRouteEvidenceError::kInvalidBoundary);
    return false;
  }
  std::uint64_t updated = 0U;
  if (!checked_add(evidence.forbidden_boundary_hits[index], count, updated)) {
    fail(evidence, PrefillRouteEvidenceError::kCounterOverflow);
    return false;
  }
  evidence.forbidden_boundary_hits[index] = updated;
  return true;
}

bool commit_prefill_route_layer_pass(
    PrefillRouteEvidence& request,
    const PrefillRouteEvidence& layer_pass) noexcept {
  if (!request.request_active || request.complete) {
    fail(request, PrefillRouteEvidenceError::kInactiveRequest);
    return false;
  }
  if (!layer_pass_is_complete(layer_pass)) {
    fail(request, PrefillRouteEvidenceError::kIncompleteTile);
    return false;
  }

  PrefillRouteEvidence merged = request;
  for (std::size_t index = 0U; index < merged.operators.size(); ++index) {
    const PrefillOperatorRouteCounts& source = layer_pass.operators[index];
    PrefillOperatorRouteCounts& destination = merged.operators[index];
    if (!checked_add(destination.production_hits, source.production_hits,
                     destination.production_hits) ||
        !checked_add(destination.exact_fallback_hits,
                     source.exact_fallback_hits,
                     destination.exact_fallback_hits) ||
        !checked_add(destination.forbidden_hits, source.forbidden_hits,
                     destination.forbidden_hits)) {
      fail(request, PrefillRouteEvidenceError::kCounterOverflow);
      return false;
    }
  }
  for (std::size_t index = 0U;
       index < merged.forbidden_boundary_hits.size(); ++index) {
    if (!checked_add(merged.forbidden_boundary_hits[index],
                     layer_pass.forbidden_boundary_hits[index],
                     merged.forbidden_boundary_hits[index])) {
      fail(request, PrefillRouteEvidenceError::kCounterOverflow);
      return false;
    }
  }
  if (!checked_add(merged.completed_layer_passes, 1U,
                   merged.completed_layer_passes)) {
    fail(request, PrefillRouteEvidenceError::kCounterOverflow);
    return false;
  }
  request = merged;
  if (has_forbidden_hits(request)) {
    fail(request, PrefillRouteEvidenceError::kForbiddenRoute);
  }
  return true;
}

PrefillRouteEvidence finalize_prefill_route_request(
    PrefillRouteEvidence& request,
    const std::uint64_t expected_layer_passes) noexcept {
  if (!request.request_active || request.complete) {
    fail(request, PrefillRouteEvidenceError::kInactiveRequest);
  } else if (request.completed_layer_passes != expected_layer_passes) {
    fail(request, PrefillRouteEvidenceError::kUnexpectedLayerPassCount);
  } else if (has_forbidden_hits(request)) {
    fail(request, PrefillRouteEvidenceError::kForbiddenRoute);
  }
  request.request_active = false;
  request.complete = true;
  request.expected_layer_passes = expected_layer_passes;
  request.valid = request.error == PrefillRouteEvidenceError::kNone;
  return request;
}

const char* to_string(const PrefillOperatorRole role) noexcept {
  switch (role) {
    case PrefillOperatorRole::kNvFp4GateUp:
      return "nvfp4_gate_up";
    case PrefillOperatorRole::kNvFp4Down:
      return "nvfp4_down";
    case PrefillOperatorRole::kFp8Qkv:
      return "fp8_qkv";
    case PrefillOperatorRole::kFp8Z:
      return "fp8_z";
    case PrefillOperatorRole::kFp8O:
      return "fp8_o";
    case PrefillOperatorRole::kAttention:
      return "attention";
    case PrefillOperatorRole::kGdn:
      return "gdn";
    case PrefillOperatorRole::kCount:
      break;
  }
  return "invalid";
}

const char* to_string(const PrefillRouteEvidenceError error) noexcept {
  switch (error) {
    case PrefillRouteEvidenceError::kNone:
      return "none";
    case PrefillRouteEvidenceError::kInactiveRequest:
      return "inactive_request";
    case PrefillRouteEvidenceError::kInvalidRole:
      return "invalid_role";
    case PrefillRouteEvidenceError::kInvalidDisposition:
      return "invalid_disposition";
    case PrefillRouteEvidenceError::kInvalidBoundary:
      return "invalid_boundary";
    case PrefillRouteEvidenceError::kCounterOverflow:
      return "counter_overflow";
    case PrefillRouteEvidenceError::kIncompleteTile:
      return "incomplete_tile";
    case PrefillRouteEvidenceError::kUnexpectedLayerPassCount:
      return "unexpected_layer_pass_count";
    case PrefillRouteEvidenceError::kForbiddenRoute:
      return "forbidden_route";
  }
  return "invalid";
}

}  // namespace q3x::runtime
