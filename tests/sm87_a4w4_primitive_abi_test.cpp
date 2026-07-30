#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

using q3x::kernels::Sm87A4W4AccumulatorCoordinate;
using q3x::kernels::Sm87A4W4FragmentCoordinate;

[[nodiscard]] bool check_signed_nibble_domain() {
  for (int value = -8; value <= 7; ++value) {
    if (!q3x::kernels::sm87_a4w4_is_signed_nibble(value)) {
      return false;
    }
    const std::uint8_t packed =
        q3x::kernels::sm87_a4w4_pack_signed_pair(value, -value - 1);
    if (q3x::kernels::sm87_a4w4_unpack_signed(packed, 0U) != value ||
        q3x::kernels::sm87_a4w4_unpack_signed(packed, 1U) !=
            -value - 1) {
      return false;
    }
  }
  return !q3x::kernels::sm87_a4w4_is_signed_nibble(-9) &&
         !q3x::kernels::sm87_a4w4_is_signed_nibble(8);
}

[[nodiscard]] bool check_a_fragment_bijection() {
  constexpr std::size_t kM = q3x::kernels::kSm87A4W4MmaM;
  constexpr std::size_t kK = q3x::kernels::kSm87A4W4MmaK;
  std::array<unsigned int, kM * kK> visits{};
  for (std::size_t lane = 0U;
       lane < q3x::kernels::kSm87A4W4WarpThreads; ++lane) {
    for (std::size_t nibble = 0U; nibble < 32U; ++nibble) {
      const Sm87A4W4FragmentCoordinate coordinate =
          q3x::kernels::sm87_a4w4_a_fragment_coordinate(lane, nibble);
      if (coordinate.outer >= kM || coordinate.k >= kK) {
        return false;
      }
      ++visits[static_cast<std::size_t>(coordinate.outer) * kK +
               coordinate.k];
    }
  }
  for (const unsigned int visit_count : visits) {
    if (visit_count != 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool check_b_fragment_bijection() {
  constexpr std::size_t kN = q3x::kernels::kSm87A4W4MmaN;
  constexpr std::size_t kK = q3x::kernels::kSm87A4W4MmaK;
  std::array<unsigned int, kN * kK> visits{};
  for (std::size_t lane = 0U;
       lane < q3x::kernels::kSm87A4W4WarpThreads; ++lane) {
    for (std::size_t nibble = 0U; nibble < 16U; ++nibble) {
      const Sm87A4W4FragmentCoordinate coordinate =
          q3x::kernels::sm87_a4w4_b_fragment_coordinate(lane, nibble);
      if (coordinate.outer >= kN || coordinate.k >= kK) {
        return false;
      }
      ++visits[static_cast<std::size_t>(coordinate.outer) * kK +
               coordinate.k];
    }
  }
  for (const unsigned int visit_count : visits) {
    if (visit_count != 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool check_accumulator_bijection() {
  constexpr std::size_t kM = q3x::kernels::kSm87A4W4MmaM;
  constexpr std::size_t kN = q3x::kernels::kSm87A4W4MmaN;
  std::array<unsigned int, kM * kN> visits{};
  for (std::size_t lane = 0U;
       lane < q3x::kernels::kSm87A4W4WarpThreads; ++lane) {
    for (std::size_t reg = 0U; reg < 4U; ++reg) {
      const Sm87A4W4AccumulatorCoordinate coordinate =
          q3x::kernels::sm87_a4w4_accumulator_coordinate(lane, reg);
      if (coordinate.m >= kM || coordinate.n >= kN) {
        return false;
      }
      ++visits[static_cast<std::size_t>(coordinate.m) * kN + coordinate.n];
    }
  }
  for (const unsigned int visit_count : visits) {
    if (visit_count != 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool check_direct_load_register_abi() {
  constexpr std::size_t kM = q3x::kernels::kSm87A4W4MmaM;
  constexpr std::size_t kN = q3x::kernels::kSm87A4W4MmaN;
  constexpr std::size_t kK = q3x::kernels::kSm87A4W4MmaK;
  constexpr std::size_t kRowBytes = kK / 2U;
  std::array<std::uint8_t, kM * kRowBytes> packed_a{};
  std::array<std::uint8_t, kN * kRowBytes> packed_b{};

  for (std::size_t row = 0U; row < kM; ++row) {
    for (std::size_t byte = 0U; byte < kRowBytes; ++byte) {
      const int even = static_cast<int>((row + 2U * byte) % 16U) - 8;
      const int odd = static_cast<int>((3U * row + 2U * byte + 1U) % 16U) -
                      8;
      packed_a[row * kRowBytes + byte] =
          q3x::kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
  }
  for (std::size_t row = 0U; row < kN; ++row) {
    for (std::size_t byte = 0U; byte < kRowBytes; ++byte) {
      const int even = static_cast<int>((5U * row + 2U * byte) % 16U) - 8;
      const int odd = static_cast<int>((7U * row + 2U * byte + 1U) % 16U) -
                      8;
      packed_b[row * kRowBytes + byte] =
          q3x::kernels::sm87_a4w4_pack_signed_pair(even, odd);
    }
  }

  // A/B register loads in the CUDA helper are four-byte reads.  Reconstruct
  // their byte/nibble positions here and prove that the documented fragment
  // mapping sees the same logical code for every lane-owned value.
  for (std::size_t lane = 0U;
       lane < q3x::kernels::kSm87A4W4WarpThreads; ++lane) {
    for (std::size_t nibble = 0U; nibble < 32U; ++nibble) {
      const std::size_t reg = nibble / 8U;
      const std::size_t nibble_in_register = nibble % 8U;
      const std::size_t row = lane / 4U + 8U * (reg & 1U);
      const std::size_t byte = 4U * (lane % 4U) +
                               16U * (reg / 2U) +
                               nibble_in_register / 2U;
      const int register_value = q3x::kernels::sm87_a4w4_unpack_signed(
          packed_a[row * kRowBytes + byte], nibble_in_register);
      const Sm87A4W4FragmentCoordinate coordinate =
          q3x::kernels::sm87_a4w4_a_fragment_coordinate(lane, nibble);
      const int canonical_value = q3x::kernels::sm87_a4w4_unpack_signed(
          packed_a[static_cast<std::size_t>(coordinate.outer) * kRowBytes +
                   coordinate.k / 2U],
          coordinate.k);
      if (register_value != canonical_value) {
        return false;
      }
    }

    for (std::size_t nibble = 0U; nibble < 16U; ++nibble) {
      const std::size_t reg = nibble / 8U;
      const std::size_t nibble_in_register = nibble % 8U;
      const std::size_t row = lane / 4U;
      const std::size_t byte = 4U * (lane % 4U) + 16U * reg +
                               nibble_in_register / 2U;
      const int register_value = q3x::kernels::sm87_a4w4_unpack_signed(
          packed_b[row * kRowBytes + byte], nibble_in_register);
      const Sm87A4W4FragmentCoordinate coordinate =
          q3x::kernels::sm87_a4w4_b_fragment_coordinate(lane, nibble);
      const int canonical_value = q3x::kernels::sm87_a4w4_unpack_signed(
          packed_b[static_cast<std::size_t>(coordinate.outer) * kRowBytes +
                   coordinate.k / 2U],
          coordinate.k);
      if (register_value != canonical_value) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool check_real_model_k64_scale_contract() {
  constexpr std::size_t kGateK = 5'120U;
  constexpr std::size_t kAttentionOK = 6'144U;
  constexpr std::size_t kDownK = 17'408U;
  constexpr std::size_t kGateGroups = 80U;
  constexpr std::size_t kAttentionOGroups = 96U;
  constexpr std::size_t kDownGroups = 272U;
  return q3x::kernels::sm87_a4w4_packed_row_bytes(kGateK) == 2'560U &&
         q3x::kernels::sm87_a4w4_k64_group_count(kGateK) == kGateGroups &&
         q3x::kernels::sm87_a4w4_k64_group_count(kAttentionOK) ==
             kAttentionOGroups &&
         q3x::kernels::sm87_a4w4_k64_group_count(kDownK) == kDownGroups &&
         q3x::kernels::sm87_a4w4_k64_scale_offset(511U, 79U,
                                                  kGateGroups) ==
             511U * kGateGroups + 79U &&
         q3x::kernels::sm87_a4w4_k64_scale_offset(17'407U, 271U,
                                                  kDownGroups) ==
             17'407U * kDownGroups + 271U &&
         q3x::kernels::sm87_a4w4_k64_group_count(5'121U) == 0U;
}

[[nodiscard]] bool check_link_contract() {
  auto* volatile query =
      &q3x::kernels::query_sm87_a4w4_primitive_resources_cuda;
  auto* volatile launch =
      &q3x::kernels::launch_sm87_a4w4_k64_primitive_smoke_cuda;
  return query != nullptr && launch != nullptr;
}

}  // namespace

int main() {
  if (!check_signed_nibble_domain()) {
    std::cerr << "signed-nibble ABI check failed\n";
    return 1;
  }
  if (!check_a_fragment_bijection()) {
    std::cerr << "A-fragment ownership is not a bijection\n";
    return 1;
  }
  if (!check_b_fragment_bijection()) {
    std::cerr << "B-fragment ownership is not a bijection\n";
    return 1;
  }
  if (!check_accumulator_bijection()) {
    std::cerr << "accumulator ownership is not a bijection\n";
    return 1;
  }
  if (!check_direct_load_register_abi()) {
    std::cerr << "canonical packed bytes do not match fragment registers\n";
    return 1;
  }
  if (!check_real_model_k64_scale_contract()) {
    std::cerr << "real-model K64 scale contract check failed\n";
    return 1;
  }
  if (!check_link_contract()) {
    std::cerr << "admission library link contract check failed\n";
    return 1;
  }
  std::cout << "SM87 A4W4 K64 packed/fragment/scale ABI checks passed\n";
  return 0;
}
