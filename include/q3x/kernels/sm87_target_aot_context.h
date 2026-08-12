#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Shared capacity identity for every constituent of
// AC-PREFILL-SM87-AOT-SYSTEM-v1. The first implementation admits only the
// three exact target witnesses. The wider minimum/maximum ranges describe the
// eventual non-overlapping product coverage; they do not make an unfinished
// constituent callable for arbitrary M.
enum class Sm87TargetAotCapacityBucket : std::uint8_t {
  kInvalid = 0U,
  kP40,
  kP60,
  kP130,
};

struct Sm87TargetAotCapacityContract {
  Sm87TargetAotCapacityBucket bucket =
      Sm87TargetAotCapacityBucket::kInvalid;
  std::size_t minimum_prompt_tokens = 0U;
  std::size_t maximum_prompt_tokens = 0U;
  std::size_t witness_prompt_tokens = 0U;
  std::size_t request_capacity_tokens = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    switch (bucket) {
      case Sm87TargetAotCapacityBucket::kP40:
        return minimum_prompt_tokens == 1U &&
               maximum_prompt_tokens == 40'000U &&
               witness_prompt_tokens == 40'000U &&
               request_capacity_tokens == 40'001U;
      case Sm87TargetAotCapacityBucket::kP60:
        return minimum_prompt_tokens == 40'001U &&
               maximum_prompt_tokens == 60'000U &&
               witness_prompt_tokens == 60'000U &&
               request_capacity_tokens == 60'001U;
      case Sm87TargetAotCapacityBucket::kP130:
        return minimum_prompt_tokens == 60'001U &&
               maximum_prompt_tokens == 130'000U &&
               witness_prompt_tokens == 130'000U &&
               request_capacity_tokens == 130'001U;
      case Sm87TargetAotCapacityBucket::kInvalid:
      default:
        return false;
    }
  }
};

inline constexpr std::array<Sm87TargetAotCapacityContract, 3U>
    kSm87TargetAotCapacityContracts{{
        {Sm87TargetAotCapacityBucket::kP40, 1U, 40'000U, 40'000U,
         40'001U},
        {Sm87TargetAotCapacityBucket::kP60, 40'001U, 60'000U, 60'000U,
         60'001U},
        {Sm87TargetAotCapacityBucket::kP130, 60'001U, 130'000U, 130'000U,
         130'001U},
    }};

inline constexpr std::array<std::size_t, 3U>
    kSm87TargetAotWitnessTokenCounts{{40'000U, 60'000U, 130'000U}};

[[nodiscard]] constexpr Sm87TargetAotCapacityContract
sm87_target_aot_capacity_contract(
    const Sm87TargetAotCapacityBucket bucket) noexcept {
  for (const auto& contract : kSm87TargetAotCapacityContracts) {
    if (contract.bucket == bucket) {
      return contract;
    }
  }
  return {};
}

[[nodiscard]] constexpr Sm87TargetAotCapacityContract
sm87_target_aot_capacity_for_witness(
    const std::size_t token_count) noexcept {
  for (const auto& contract : kSm87TargetAotCapacityContracts) {
    if (contract.witness_prompt_tokens == token_count) {
      return contract;
    }
  }
  return {};
}

[[nodiscard]] constexpr bool sm87_target_aot_exact_witness_tokens(
    const std::size_t token_count) noexcept {
  return sm87_target_aot_capacity_for_witness(token_count).valid();
}

template <std::size_t kCount>
[[nodiscard]] constexpr bool sm87_target_aot_nonzero_unique_identities(
    const std::array<std::uint64_t, kCount>& identities) noexcept {
  for (std::size_t left = 0U; left < identities.size(); ++left) {
    if (identities[left] == 0U) {
      return false;
    }
    for (std::size_t right = 0U; right < left; ++right) {
      if (identities[left] == identities[right]) {
        return false;
      }
    }
  }
  return true;
}

static_assert(kSm87TargetAotCapacityContracts[0U].valid());
static_assert(kSm87TargetAotCapacityContracts[1U].valid());
static_assert(kSm87TargetAotCapacityContracts[2U].valid());
static_assert(kSm87TargetAotCapacityContracts[0U].maximum_prompt_tokens +
                  1U ==
              kSm87TargetAotCapacityContracts[1U].minimum_prompt_tokens);
static_assert(kSm87TargetAotCapacityContracts[1U].maximum_prompt_tokens +
                  1U ==
              kSm87TargetAotCapacityContracts[2U].minimum_prompt_tokens);

}  // namespace q3x::kernels
