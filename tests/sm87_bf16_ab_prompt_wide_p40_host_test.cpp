#include "q3x/kernels/sm87_bf16_ab_prefill.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

namespace {

using q3x::kernels::Sm87Bf16AbPromptWideP40Resources;
using q3x::kernels::make_sm87_bf16_ab_prompt_wide_p40_plan;
using q3x::kernels::sm87_bf16_ab_prompt_wide_p40_tile;

constexpr auto kP40 = make_sm87_bf16_ab_prompt_wide_p40_plan(40'000U);
constexpr auto kP39999 = make_sm87_bf16_ab_prompt_wide_p40_plan(39'999U);
constexpr auto kP40001 = make_sm87_bf16_ab_prompt_wide_p40_plan(40'001U);
constexpr auto kP60 = make_sm87_bf16_ab_prompt_wide_p40_plan(60'000U);
constexpr auto kFirst = sm87_bf16_ab_prompt_wide_p40_tile(kP40, 0U);
constexpr auto kLast = sm87_bf16_ab_prompt_wide_p40_tile(kP40, 624U);
constexpr auto kPast = sm87_bf16_ab_prompt_wide_p40_tile(kP40, 625U);

static_assert(kP40.valid() && kP40.admitted);
static_assert(kP40.requested_token_count == 40'000U &&
              kP40.tile_tokens == 64U && kP40.grid_blocks == 625U &&
              kP40.threads == 256U && kP40.launch_count == 1U &&
              kP40.dynamic_shared_bytes == 46'080U);
static_assert(kP40.input_elements == 204'800'000U &&
              kP40.weight_elements_per_projection == 245'760U &&
              kP40.output_elements_per_projection == 1'920'000U);
static_assert(
    q3x::kernels::kSm87Bf16AbPromptWideP40MaximumWeightIndex == 245'759U &&
    q3x::kernels::kSm87Bf16AbPromptWideP40MaximumInputIndex ==
        204'799'999U &&
    q3x::kernels::kSm87Bf16AbPromptWideP40MaximumOutputIndex == 1'919'999U);
static_assert(!kP39999.valid() && !kP39999.admitted &&
              kP39999.grid_blocks == 0U && kP39999.launch_count == 0U);
static_assert(!kP40001.valid() && !kP40001.admitted &&
              kP40001.grid_blocks == 0U && kP40001.launch_count == 0U);
static_assert(!kP60.valid() && !kP60.admitted &&
              kP60.grid_blocks == 0U && kP60.launch_count == 0U);
static_assert(kFirst.valid && kFirst.block == 0U &&
              kFirst.first_token == 0U && kFirst.token_count == 64U);
static_assert(kLast.valid && kLast.block == 624U &&
              kLast.first_token == 39'936U &&
              kLast.first_token + kLast.token_count == 40'000U);
static_assert(!kPast.valid);

constexpr Sm87Bf16AbPromptWideP40Resources kAdmittedResources{
    8, 7, 87, 64, 0U, 46'080U, 128U, 2, true};
constexpr Sm87Bf16AbPromptWideP40Resources kWrongSm{
    8, 6, 87, 64, 0U, 46'080U, 0U, 2, true};
constexpr Sm87Bf16AbPromptWideP40Resources kWrongBinary{
    8, 7, 86, 64, 0U, 46'080U, 0U, 2, true};
constexpr Sm87Bf16AbPromptWideP40Resources kOneCta{
    8, 7, 87, 64, 0U, 46'080U, 0U, 1, true};
static_assert(kAdmittedResources.valid(),
              "local spill bytes are evidence, not an automatic rejection");
static_assert(!kWrongSm.valid() && !kWrongBinary.valid() &&
              !kOneCta.valid());

using Launcher = int (*)(const std::uint16_t*, const std::uint16_t*,
                         const std::uint16_t*, std::size_t,
                         std::uint16_t*, std::uint16_t*, void*) noexcept;
using ResourceQuery = int (*)(Sm87Bf16AbPromptWideP40Resources*) noexcept;
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_bf16_ab_prompt_wide_p40_cuda),
              Launcher>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           query_sm87_bf16_ab_prompt_wide_p40_resources_cuda),
              ResourceQuery>);

[[nodiscard]] bool exhaustive_grid_is_exact() {
  std::vector<std::uint8_t> visits(kP40.requested_token_count, 0U);
  std::size_t previous_end = 0U;
  for (std::size_t block = 0U; block < kP40.grid_blocks; ++block) {
    const auto tile = sm87_bf16_ab_prompt_wide_p40_tile(kP40, block);
    if (!tile.valid || tile.first_token != previous_end ||
        tile.token_count != 64U ||
        tile.first_token + tile.token_count > visits.size()) {
      return false;
    }
    for (std::size_t token = tile.first_token;
         token < tile.first_token + tile.token_count; ++token) {
      if (visits[token] != 0U) {
        return false;
      }
      visits[token] = 1U;
    }
    previous_end = tile.first_token + tile.token_count;
  }
  if (previous_end != visits.size()) {
    return false;
  }
  for (const std::uint8_t visit : visits) {
    if (visit != 1U) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!exhaustive_grid_is_exact()) {
    std::cerr << "BF16 A/B prompt-wide P40 grid is not a one-to-one token "
                 "partition\n";
    return 1;
  }
  std::cout << "BF16 A/B prompt-wide P40 host contract passed\n";
  return 0;
}
