#include "reference_engine_final_token_policy_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

namespace detail = q3x::runtime::reference_engine_detail;

using Policy = detail::ReferenceEnginePrefillFinalTokenPolicy;
using TestPolicy = detail::ReferenceEnginePrefillFinalTokenPolicyForTest;

static_assert(detail::kReferenceEnginePrefillFinalTokenProductionPolicy ==
              Policy::kLegacyScalarFinalStep);

[[nodiscard]] bool expect_exact(const std::size_t prompt_tokens,
                                const bool expected) {
  const bool selected =
      detail::reference_engine_prefill_exact_full_c512_prompt_selected(
          Policy::kExactFullC512Tile, false, false, 512U, prompt_tokens);
  if (selected != expected) {
    std::cerr << "P" << prompt_tokens << " selected=" << selected
              << " expected=" << expected << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  constexpr std::array<std::size_t, 9U> prompt_lengths{
      511U, 512U, 513U, 4'095U, 4'096U, 4'097U, 8'176U, 8'177U,
      8'192U};
  constexpr std::array<bool, prompt_lengths.size()> expected{
      false, true, false, false, true, false, false, false, true};

  bool exact = true;
  for (std::size_t index = 0U; index < prompt_lengths.size(); ++index) {
    exact = expect_exact(prompt_lengths[index], expected[index]) && exact;
  }

  exact =
      !detail::reference_engine_prefill_exact_full_c512_prompt_selected(
          Policy::kLegacyScalarFinalStep, false, false, 512U, 4'096U) &&
      !detail::reference_engine_prefill_exact_full_c512_prompt_selected(
          Policy::kExactFullC512Tile, false, false, 256U, 4'096U) &&
      !detail::reference_engine_prefill_exact_full_c512_prompt_selected(
          Policy::kExactFullC512Tile, false, true, 512U, 4'096U) &&
      !detail::reference_engine_prefill_exact_full_c512_prompt_selected(
          Policy::kExactFullC512Tile, true, false, 512U, 4'096U) &&
      exact;

  const auto production =
      detail::select_reference_engine_prefill_final_token_policy(
          TestPolicy::kProductionDefault, false, false, 512U, 4'096U);
  const auto forced_exact =
      detail::select_reference_engine_prefill_final_token_policy(
          TestPolicy::kExactFullC512Tile, false, false, 512U, 4'096U);
  const auto all_prompt =
      detail::select_reference_engine_prefill_final_token_policy(
          TestPolicy::kAllPromptTiles, false, false, 512U, 513U);
  const auto single =
      detail::select_reference_engine_prefill_final_token_policy(
          TestPolicy::kSingleArbitraryTile, false, false, 512U, 513U);
  const auto traced =
      detail::select_reference_engine_prefill_final_token_policy(
          TestPolicy::kExactFullC512Tile, false, true, 512U, 4'096U);
  const auto whole =
      detail::select_reference_engine_prefill_final_token_policy(
          TestPolicy::kExactFullC512Tile, true, false, 512U, 4'096U);
  exact = !production.all_prompt_tokens &&
          !production.single_arbitrary_tile &&
          forced_exact.all_prompt_tokens &&
          !forced_exact.single_arbitrary_tile &&
          all_prompt.all_prompt_tokens &&
          !all_prompt.single_arbitrary_tile && single.all_prompt_tokens &&
          single.single_arbitrary_tile && !traced.all_prompt_tokens &&
          !traced.single_arbitrary_tile && whole.all_prompt_tokens &&
          !whole.single_arbitrary_tile && exact;

  if (!exact) {
    std::cerr << "reference Prefill final-token policy matrix failed\n";
    return 1;
  }
  std::cout << "reference Prefill final-token policy matrix passed\n";
  return 0;
}
