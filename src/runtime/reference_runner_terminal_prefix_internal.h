#pragma once

#include "q3x/runtime/prefill_route_evidence.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace q3x::runtime {
class RequestState;
}

namespace q3x::runtime::reference_runner_detail {

// This is a source-private ordinary-generation topology, not a new numerical
// route or a public runner option. All Q/K/V work remains unchanged; only the
// terminal layer's prefix Attention/O/MLP suffix is absent. The already chosen
// scalar final prompt step completes the live last row.
inline constexpr const char* kTerminalPrefixElisionPlanId =
    "q3x.sm87.legacy-c512.terminal-prefix-elision.v1";

[[nodiscard]] constexpr bool is_terminal_prefix_elision_scope(
    const bool sm87_weight_only, const bool legacy_c512,
    const bool whole_request, const bool capture_trace,
    const bool all_prompt_tokens) noexcept {
  return sm87_weight_only && legacy_c512 && !whole_request &&
         !capture_trace && !all_prompt_tokens;
}

[[nodiscard]] constexpr bool terminal_prefix_elides_role(
    const PrefillOperatorRole role) noexcept {
  return role == PrefillOperatorRole::kNvFp4GateUp ||
         role == PrefillOperatorRole::kNvFp4Down ||
         role == PrefillOperatorRole::kFp8O ||
         role == PrefillOperatorRole::kAttention;
}

// Only the private terminal-prefix path may commit this precise four-role
// omission. The public full-pass commit remains strict and unchanged.
[[nodiscard]] bool commit_terminal_prefix_elided_layer_pass(
    PrefillRouteEvidence& request,
    const PrefillRouteEvidence& layer_pass) noexcept;

// Recover the physical omission receipt without enlarging any public object.
// All four deficits must agree, unaffected roles must be complete, and at
// most one suffix per layer pass may be absent. nullopt is malformed evidence.
[[nodiscard]] std::optional<std::uint64_t>
terminal_prefix_elided_layer_passes(
    const PrefillRouteEvidence& evidence) noexcept;

// Defined only in BUILD_TESTING=ON. No environment, CLI, or request selector
// exists, and the installed ABI exports no mutable route authority.
void set_terminal_prefix_elision_enabled_for_test(bool enabled) noexcept;

// BUILD_TESTING-only synchronous observation/injection seam. The first stage
// follows layer 62 enqueue, before layer 63 may read or write any workspace;
// its callback must wait on cuda_stream before reading device bytes. The last
// stage follows successful whole-tile stream/state/route commit, before the
// next tile or scalar step. Mutable state exists solely to permit bounded
// dead-scratch poison tests; persistent state and RoPE are never poison targets.
enum class TerminalPrefixObservationStage : std::uint8_t {
  kBeforeLayer63,
  kAfterTileCommit,
};

struct TerminalPrefixObservationView {
  RequestState* state = nullptr;
  std::uint32_t first_position = 0U;
  std::size_t token_count = 0U;
  TerminalPrefixObservationStage stage =
      TerminalPrefixObservationStage::kBeforeLayer63;
  void* cuda_stream = nullptr;
};

struct TerminalPrefixObservationHook {
  // False propagates through the ordinary synchronous failure/poison path;
  // an observer error must never be silently treated as successful Prefill.
  bool (*callback)(const TerminalPrefixObservationView&, void*) noexcept = nullptr;
  void* context = nullptr;
};

[[nodiscard]] TerminalPrefixObservationHook
exchange_terminal_prefix_observation_hook_for_test(
    TerminalPrefixObservationHook hook) noexcept;

}  // namespace q3x::runtime::reference_runner_detail
