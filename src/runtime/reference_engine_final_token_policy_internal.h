#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::reference_engine_detail {

// Private production authority for the legacy tiled runner's final prompt
// token. Keep this selector source-local: installed ReferenceGenerateOptions
// must not acquire an admission field or an ABI dependency on this policy.
enum class ReferenceEnginePrefillFinalTokenPolicy : std::uint8_t {
  kLegacyScalarFinalStep = 0,
  kExactFullC512Tile,
};

// Promotion changes exactly this private default after the state/logit/Decode
// handoff oracle qualifies the exact-C512 route.
inline constexpr ReferenceEnginePrefillFinalTokenPolicy
    kReferenceEnginePrefillFinalTokenProductionPolicy =
        ReferenceEnginePrefillFinalTokenPolicy::kLegacyScalarFinalStep;

// BUILD_TESTING-only, thread-local A/B authority. kExactFullC512Tile retains
// the production selector matrix: legacy mode, no trace, C512 chunking, and
// P % 512 == 0. kAllPromptTiles and kSingleArbitraryTile are explicit oracle
// routes that replace the former runtime-environment admissions.
enum class ReferenceEnginePrefillFinalTokenPolicyForTest : std::uint8_t {
  kProductionDefault = 0,
  kLegacyScalarFinalStep,
  kExactFullC512Tile,
  kAllPromptTiles,
  kSingleArbitraryTile,
};

struct ReferenceEnginePrefillFinalTokenSelection {
  bool all_prompt_tokens = false;
  bool single_arbitrary_tile = false;
};

[[nodiscard]] constexpr bool
reference_engine_prefill_exact_full_c512_prompt_selected(
    const ReferenceEnginePrefillFinalTokenPolicy policy,
    const bool whole_request_layer_major, const bool capture_trace,
    const std::uint32_t prefill_chunk_size,
    const std::size_t prompt_token_count) noexcept {
  constexpr std::uint32_t kExactTileTokens = 512U;
  return policy ==
             ReferenceEnginePrefillFinalTokenPolicy::kExactFullC512Tile &&
         !whole_request_layer_major && !capture_trace &&
         prefill_chunk_size == kExactTileTokens && prompt_token_count != 0U &&
         (prompt_token_count % kExactTileTokens) == 0U;
}

[[nodiscard]] constexpr ReferenceEnginePrefillFinalTokenPolicy
resolve_reference_engine_prefill_final_token_policy(
    const ReferenceEnginePrefillFinalTokenPolicyForTest policy) noexcept {
  switch (policy) {
    case ReferenceEnginePrefillFinalTokenPolicyForTest::kLegacyScalarFinalStep:
    case ReferenceEnginePrefillFinalTokenPolicyForTest::kAllPromptTiles:
    case ReferenceEnginePrefillFinalTokenPolicyForTest::kSingleArbitraryTile:
      return ReferenceEnginePrefillFinalTokenPolicy::kLegacyScalarFinalStep;
    case ReferenceEnginePrefillFinalTokenPolicyForTest::kExactFullC512Tile:
      return ReferenceEnginePrefillFinalTokenPolicy::kExactFullC512Tile;
    case ReferenceEnginePrefillFinalTokenPolicyForTest::kProductionDefault:
    default:
      return kReferenceEnginePrefillFinalTokenProductionPolicy;
  }
}

[[nodiscard]] constexpr ReferenceEnginePrefillFinalTokenSelection
select_reference_engine_prefill_final_token_policy(
    const ReferenceEnginePrefillFinalTokenPolicyForTest policy,
    const bool whole_request_layer_major, const bool capture_trace,
    const std::uint32_t prefill_chunk_size,
    const std::size_t prompt_token_count) noexcept {
  if (whole_request_layer_major) {
    return {true, false};
  }
  if (capture_trace || prefill_chunk_size <= 1U) {
    return {};
  }
  if (policy ==
      ReferenceEnginePrefillFinalTokenPolicyForTest::kAllPromptTiles) {
    return {true, false};
  }
  if (policy ==
      ReferenceEnginePrefillFinalTokenPolicyForTest::kSingleArbitraryTile) {
    return {true, true};
  }
  return {reference_engine_prefill_exact_full_c512_prompt_selected(
              resolve_reference_engine_prefill_final_token_policy(policy),
              false, false, prefill_chunk_size, prompt_token_count),
          false};
}

[[nodiscard]] ReferenceEnginePrefillFinalTokenPolicyForTest
exchange_reference_engine_prefill_final_token_policy_for_test(
    ReferenceEnginePrefillFinalTokenPolicyForTest policy) noexcept;

}  // namespace q3x::runtime::reference_engine_detail
