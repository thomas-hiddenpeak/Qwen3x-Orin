#include "reference_runner_layer_wide_p40_mlp_policy_internal.h"

#include <iostream>

namespace {

namespace detail = q3x::runtime::reference_runner_detail;
namespace runtime = q3x::runtime;

using Input = detail::LayerWideP40MlpDescriptorPolicyInput;

[[nodiscard]] constexpr Input baseline_input() noexcept {
  Input input;
  input.mlp_layout = runtime::LayerMajorRequestMlpLayout::
      kLayerWideP40PersistentTwoSpan;
  input.max_sequence_length =
      runtime::kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens;
  input.mlp_capacity_tokens =
      runtime::kLayerMajorPrefillLayerWideMlpP40Tokens;
  return input;
}

static_assert(
    detail::valid_layer_wide_p40_mlp_descriptor_policy(baseline_input()));
static_assert(detail::valid_layer_wide_p40_mlp_residual_capacity(
    baseline_input(),
    runtime::kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens));
static_assert(!detail::valid_layer_wide_p40_mlp_residual_capacity(
    baseline_input(),
    runtime::kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens + 1U));

[[nodiscard]] constexpr Input candidate_input() noexcept {
  Input input = baseline_input();
  input.max_sequence_length = 40'016U;
  return input;
}

static_assert(
    !detail::valid_layer_wide_p40_mlp_descriptor_policy(candidate_input()));
static_assert(!detail::valid_layer_wide_p40_mlp_residual_capacity(
    candidate_input(), 40'016U));

}  // namespace

int main() {
  if (!detail::valid_layer_wide_p40_mlp_descriptor_policy(baseline_input()) ||
      detail::valid_layer_wide_p40_mlp_residual_capacity(
          baseline_input(), 40'000U)) {
    std::cerr << "layer-wide P40 MLP baseline policy failed\n";
    return 1;
  }
  if (detail::valid_layer_wide_p40_mlp_descriptor_policy(candidate_input())) {
    std::cerr << "default-off P40016 isolation failed\n";
    return 1;
  }
  std::cout << "layer-wide P40 MLP descriptor policy passed\n";
  return 0;
}
