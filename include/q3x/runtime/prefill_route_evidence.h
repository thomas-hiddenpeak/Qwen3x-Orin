#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime {

// Request-level logical Prefill roles. A hit represents one successfully
// completed logical operator invocation, not a selector decision or an
// individual CUDA kernel launch. This keeps fused and unfused routes
// comparable without pretending that configured/admitted code executed.
enum class PrefillOperatorRole : std::uint8_t {
  kNvFp4GateUp = 0,
  kNvFp4Down,
  kFp8Qkv,
  kFp8Z,
  kFp8O,
  kAttention,
  kGdn,
  kCount,
};

enum class PrefillRouteDisposition : std::uint8_t {
  kProduction = 0,
  kExactFallback,
  kForbidden,
};

enum class PrefillForbiddenBoundary : std::uint8_t {
  kPrefixCache = 0,
  kMtp,
  kCublasLt,
  kExternalReference,
  kApproximateNumerics,
  kCount,
};

enum class PrefillRouteEvidenceError : std::uint8_t {
  kNone = 0,
  kInactiveRequest,
  kInvalidRole,
  kInvalidDisposition,
  kInvalidBoundary,
  kCounterOverflow,
  kIncompleteTile,
  kUnexpectedLayerPassCount,
  kForbiddenRoute,
};

inline constexpr std::size_t kPrefillOperatorRoleCount =
    static_cast<std::size_t>(PrefillOperatorRole::kCount);
inline constexpr std::size_t kPrefillForbiddenBoundaryCount =
    static_cast<std::size_t>(PrefillForbiddenBoundary::kCount);

// The pinned Qwen3.6 schedule has 48 GDN layers and 16 full-Attention layers.
// These are logical invocations expected from every successfully committed
// 64-layer Prefill pass, independent of tile length or fusion choices.
inline constexpr std::array<std::uint64_t, kPrefillOperatorRoleCount>
    kExpectedPrefillLogicalOperatorsPerTile{{
        64U,  // NVFP4 Gate+Up
        64U,  // NVFP4 Down
        96U,  // linear QKV plus full Q/K/V
        48U,  // linear Z
        64U,  // linear/full output projection
        16U,  // full causal Attention
        48U,  // exact GDN recurrence
    }};

struct PrefillOperatorRouteCounts {
  std::uint64_t production_hits = 0U;
  std::uint64_t exact_fallback_hits = 0U;
  std::uint64_t forbidden_hits = 0U;
};

// A request recorder is reset before runner state, receives only complete
// layer-pass records, and is finalized once generation control has returned.
// A missing/overflowed/forbidden record remains observable but is never valid.
struct PrefillRouteEvidence {
  std::array<PrefillOperatorRouteCounts, kPrefillOperatorRoleCount> operators{};
  std::array<std::uint64_t, kPrefillForbiddenBoundaryCount>
      forbidden_boundary_hits{};
  std::uint64_t completed_layer_passes = 0U;
  std::uint64_t expected_layer_passes = 0U;
  bool request_active = false;
  bool complete = false;
  bool valid = false;
  PrefillRouteEvidenceError error = PrefillRouteEvidenceError::kNone;
};

void reset_prefill_route_request(PrefillRouteEvidence& evidence) noexcept;

[[nodiscard]] bool record_prefill_operator_route(
    PrefillRouteEvidence& evidence, PrefillOperatorRole role,
    PrefillRouteDisposition disposition, std::uint64_t count = 1U) noexcept;

[[nodiscard]] bool record_prefill_forbidden_boundary(
    PrefillRouteEvidence& evidence, PrefillForbiddenBoundary boundary,
    std::uint64_t count = 1U) noexcept;

// Validates one complete 64-layer pass, then atomically merges it into the
// active request. The request is left invalid on any mismatch; partial pass
// counters are never merged.
[[nodiscard]] bool commit_prefill_route_layer_pass(
    PrefillRouteEvidence& request,
    const PrefillRouteEvidence& layer_pass) noexcept;

// Marks the request closed and checks that the runner committed exactly the
// number of Prefix executions reported by generation control.
[[nodiscard]] PrefillRouteEvidence finalize_prefill_route_request(
    PrefillRouteEvidence& request,
    std::uint64_t expected_layer_passes) noexcept;

[[nodiscard]] const char* to_string(PrefillOperatorRole role) noexcept;
[[nodiscard]] const char* to_string(
    PrefillRouteEvidenceError error) noexcept;

}  // namespace q3x::runtime
