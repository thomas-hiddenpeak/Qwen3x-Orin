#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

static_assert(kernels::kSm87A4W4GateUpK512FragmentNativeThreads == 256U);
static_assert(kernels::kSm87A4W4GateUpK512FragmentNativeWarps == 8U);
static_assert(kernels::kSm87A4W4GateUpK512FragmentNativeTileM == 64U);
static_assert(kernels::kSm87A4W4GateUpK512FragmentNativeTileN == 64U);
static_assert(kernels::kSm87A4W4GateUpK512FragmentNativeAStages == 3U);
static_assert(kernels::kSm87A4W4GateUpK512FragmentNativeSharedBytes ==
              12'288U);
static_assert(kernels::kSm87A4W4GateUpK512FragmentNativePairSlotBytes ==
              16U);
static_assert(kernels::kSm87A4W4GateUpK512FragmentNativeCtasPerSm == 2U);
static_assert(kernels::kSm87A4W4GateUpK512FragmentNativeMaximumRegisters ==
              128U);
static_assert(
    kernels::sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
        64U, 512U) == 32'768U);
static_assert(
    kernels::sm87_a4w4_gateup_k512_fragment_native_scale_capacity_elements(
        64U, 512U) == 128U);
static_assert(
    kernels::sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
        17'408U, 5'120U) == 89'128'960U);
static_assert(
    kernels::sm87_a4w4_gateup_k512_fragment_native_scale_capacity_elements(
        17'408U, 5'120U) == 348'160U);

[[nodiscard]] bool verify_code_bijection() {
  constexpr std::size_t n = 128U;
  constexpr std::size_t k = 1'024U;
  constexpr std::size_t k512_groups = k / 512U;
  constexpr std::size_t k64_groups = k / 64U;
  const std::size_t canonical_one =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(n, k);
  const std::size_t native =
      kernels::sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
          n, k);
  if (native != 2U * canonical_one) {
    std::cerr << "paired code capacity is not equal-byte\n";
    return false;
  }
  std::vector<std::uint8_t> native_seen(native, 0U);
  std::vector<std::uint8_t> gate_seen(canonical_one, 0U);
  std::vector<std::uint8_t> up_seen(canonical_one, 0U);
  for (std::size_t block = 0U; block < n / 64U; ++block) {
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      for (std::size_t phase = 0U; phase < 8U; ++phase) {
        for (std::size_t fragment = 0U; fragment < 8U; ++fragment) {
          const std::size_t fragment_n = block * 64U + fragment * 8U;
          for (std::size_t lane = 0U; lane < 32U; ++lane) {
            const std::size_t slot =
                kernels::sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                    fragment_n, group, phase, lane, k512_groups);
            if (slot % 16U != 0U || slot + 16U > native) {
              std::cerr << "unaligned or out-of-range native slot\n";
              return false;
            }
            const std::size_t row = fragment_n + lane / 4U;
            const std::size_t physical_k64 = group * 8U + phase;
            for (std::size_t projection = 0U; projection < 2U;
                 ++projection) {
              for (std::size_t reg = 0U; reg < 2U; ++reg) {
                for (std::size_t byte = 0U; byte < 4U; ++byte) {
                  const std::size_t canonical =
                      kernels::sm87_a4w4_consumer_packed_offset(
                          row, physical_k64,
                          reg * 16U + 4U * (lane % 4U) + byte,
                          k64_groups);
                  const std::size_t physical =
                      slot + projection * 8U + reg * 4U + byte;
                  std::vector<std::uint8_t>& seen =
                      projection == 0U ? gate_seen : up_seen;
                  if (canonical >= canonical_one ||
                      physical >= native || seen[canonical] != 0U ||
                      native_seen[physical] != 0U) {
                    std::cerr << "code map is not a bijection\n";
                    return false;
                  }
                  seen[canonical] = 1U;
                  native_seen[physical] = 1U;
                }
              }
            }
          }
        }
      }
    }
  }
  for (const std::uint8_t value : native_seen) {
    if (value != 1U) {
      std::cerr << "native code hole\n";
      return false;
    }
  }
  for (const std::uint8_t value : gate_seen) {
    if (value != 1U) {
      std::cerr << "canonical Gate code hole\n";
      return false;
    }
  }
  for (const std::uint8_t value : up_seen) {
    if (value != 1U) {
      std::cerr << "canonical Up code hole\n";
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool verify_scale_bijection() {
  constexpr std::size_t n = 128U;
  constexpr std::size_t k = 1'024U;
  constexpr std::size_t groups = k / 512U;
  const std::size_t capacity =
      kernels::sm87_a4w4_gateup_k512_fragment_native_scale_capacity_elements(
          n, k);
  if (capacity != 2U * n * groups) {
    return false;
  }
  std::vector<std::uint8_t> seen(capacity, 0U);
  for (std::size_t row = 0U; row < n; ++row) {
    for (std::size_t group = 0U; group < groups; ++group) {
      const std::size_t offset =
          kernels::sm87_a4w4_gateup_k512_fragment_native_scale_pair_offset(
              row, group, groups);
      if (offset + 1U >= capacity || seen[offset] != 0U ||
          seen[offset + 1U] != 0U) {
        return false;
      }
      seen[offset] = 1U;
      seen[offset + 1U] = 1U;
    }
  }
  for (const std::uint8_t value : seen) {
    if (value != 1U) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!verify_code_bijection() || !verify_scale_bijection()) {
    return 1;
  }
  const kernels::Sm87A4W4GateUpK512FragmentNativePlan plan =
      kernels::sm87_a4w4_gateup_k512_fragment_native_plan(
          1'920U, 17'408U, 5'120U, 0U, 8'704U);
  if (!kernels::sm87_a4w4_gateup_k512_fragment_native_is_model_plan(plan) ||
      plan.m_tiles != 30U || plan.n_tiles != 136U ||
      plan.k512_groups != 10U || plan.physical_k64_groups != 80U ||
      plan.launch_ctas != 30U) {
    std::cerr << "model plan mismatch\n";
    return 1;
  }
  if (kernels::sm87_a4w4_gateup_k512_fragment_native_plan(
          1'921U, 17'408U, 5'120U, 0U, 8'704U)
          .launch_ctas != 0U ||
      kernels::sm87_a4w4_gateup_k512_fragment_native_plan(
          1'920U, 17'408U, 5'121U, 0U, 8'704U)
          .launch_ctas != 0U) {
    std::cerr << "invalid plan admitted\n";
    return 1;
  }
  std::cout << "PASS: fragment-native paired B ABI is equal-byte and "
               "bijective\n";
  return 0;
}
