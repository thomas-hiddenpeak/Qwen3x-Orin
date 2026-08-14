#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace kernels = q3x::kernels;

namespace {

static_assert(kernels::kSm87BulkV2NvFp4GateUpWholeP40Tokens == 40'000U);
static_assert(kernels::kSm87BulkV2NvFp4GateUpWholeP40MTiles == 625U);
static_assert(kernels::kSm87BulkV2NvFp4GateUpWholeP40NTiles == 272U);
static_assert(kernels::kSm87BulkV2NvFp4GateUpWholeP40KTiles == 80U);
static_assert(kernels::kSm87BulkV2NvFp4GateUpWholeP40MGroups == 157U);
static_assert(kernels::kSm87BulkV2NvFp4GateUpWholeP40NGroups == 34U);
static_assert(kernels::kSm87BulkV2NvFp4GateUpWholeP40Cohorts == 5'338U);
static_assert(kernels::kSm87BulkV2NvFp4GateUpWholeP40LogicalCells ==
              170'000U);
static_assert(kernels::kSm87BulkV2NvFp4GateUpWholeP40ScheduledCells ==
              170'816U);
static_assert(kernels::kSm87BulkV2NvFp4GateUpWholeP40MaskedCells == 816U);
static_assert(
    kernels::kSm87BulkV2NvFp4GateUpWholeP40PipelineSharedBytes == 38'400U);
static_assert(kernels::kSm87BulkV2NvFp4GateUpWholeP40DynamicSharedBytes ==
              38'400U);

class TestContext final {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

void test_frozen_contract(TestContext& test) {
  const auto& contract =
      kernels::kSm87BulkV2NvFp4GateUpWholeP40FrozenContract;
  const auto& traffic = contract.traffic;
  test.expect(
      kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_host_contract_valid(
          contract),
      "the independent whole-P40 Gate+Up contract is internally valid");
  test.expect(
      kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_traffic_valid(traffic),
      "the whole-layer traffic proof retains every frozen exact value");
  test.expect(traffic.physical_launches == 1U &&
                  traffic.legacy_m1024_segments == 0U,
              "the successor is one whole-layer launch, not forty segments");
  test.expect(traffic.grid_syncs_per_launch == 2U &&
                  contract.no_cohort_grid_barrier,
              "only entry and terminal grid synchronization remain");
  test.expect(traffic.theoretical_full_cohort_b_reuse == 4U &&
                  traffic.theoretical_not_measured &&
                  traffic.activation_group_l2_residency_unproven,
              "four-way B reuse is marked as a falsifiable theory");
  test.expect(traffic.activation_group_footprint_bytes == 2'621'440ULL &&
                  traffic.distinct_activation_bytes == 409'600'000ULL,
              "the four-M-tile and whole-P40 A footprints are exact");
  test.expect(contract.dynamic_shared_bytes == 38'400U,
              "the three exact 12800B pipeline stages are the only smem");
  test.expect(traffic.branch_b_bytes_per_n_tile == 184'320ULL &&
                  traffic.paired_b_bytes_per_n_tile == 368'640ULL,
              "Gate and Up packed payload traffic is accounted separately");
  test.expect(traffic.logical_cta_b_request_bytes == 62'668'800'000ULL &&
                  traffic.theoretical_l2_b_service_bytes ==
                      15'742'402'560ULL,
              "logical and ideal L2 B service bytes expose the reuse gap");
  test.expect(contract.deterministic_full_grid_mapping &&
                  contract.snake_n_group_order && contract.no_split_k &&
                  contract.no_global_partial_c &&
                  contract.no_activation_quantization &&
                  contract.no_cublaslt && contract.no_mtp &&
                  contract.no_request_time_jit_repack_query &&
                  contract.independent_default_off_identity &&
                  !contract.production_dispatch_eligible,
              "the candidate remains independent, exact, and default-off");

  auto tampered = contract;
  tampered.traffic.legacy_m1024_segments = 40U;
  test.expect(
      !kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_host_contract_valid(
          tampered),
      "the contract rejects restoration of the old forty-segment shape");
  tampered = contract;
  tampered.no_cohort_grid_barrier = false;
  test.expect(
      !kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_host_contract_valid(
          tampered),
      "the contract rejects a barrier in the 5,338-cohort hot loop");
}

void test_exhaustive_raster(TestContext& test) {
  std::vector<std::uint8_t> seen(
      kernels::kSm87BulkV2NvFp4GateUpWholeP40LogicalCells, 0U);
  std::uint32_t active = 0U;
  std::uint32_t masked = 0U;

  for (std::uint32_t cohort = 0U;
       cohort < kernels::kSm87BulkV2NvFp4GateUpWholeP40Cohorts;
       ++cohort) {
    const std::uint32_t expected_m_group =
        cohort / kernels::kSm87BulkV2NvFp4GateUpWholeP40NGroups;
    const std::uint32_t expected_n_epoch =
        cohort % kernels::kSm87BulkV2NvFp4GateUpWholeP40NGroups;
    const std::uint32_t expected_n_group =
        (expected_m_group & 1U) == 0U
            ? expected_n_epoch
            : kernels::kSm87BulkV2NvFp4GateUpWholeP40NGroups - 1U -
                  expected_n_epoch;
    std::array<std::uint32_t,
               kernels::kSm87BulkV2NvFp4GateUpWholeP40RasterN>
        b_requesters{};
    std::array<std::uint32_t,
               kernels::kSm87BulkV2NvFp4GateUpWholeP40RasterM>
        a_requesters{};

    for (std::uint32_t cta = 0U;
         cta < kernels::kSm87BulkV2NvFp4GateUpWholeP40PersistentCtas;
         ++cta) {
      const auto cell =
          kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_cell(cohort, cta);
      test.expect(cell.cohort == cohort && cell.cta == cta &&
                      cell.m_group == expected_m_group &&
                      cell.n_epoch == expected_n_epoch &&
                      cell.n_group == expected_n_group,
                  "every CTA observes the deterministic M-outer snake");
      test.expect(cell.m_lane ==
                          cta /
                              kernels::
                                  kSm87BulkV2NvFp4GateUpWholeP40RasterN &&
                      cell.n_lane ==
                          cta %
                              kernels::
                                  kSm87BulkV2NvFp4GateUpWholeP40RasterN,
                  "CTA identity is a fixed 4M by 8N Cartesian lane");
      if (!cell.active) {
        ++masked;
        continue;
      }
      ++active;
      ++b_requesters[cell.n_lane];
      ++a_requesters[cell.m_lane];
      test.expect(cell.logical_ordinal < seen.size(),
                  "active cells map inside the exact logical domain");
      if (cell.logical_ordinal < seen.size()) {
        test.expect(seen[cell.logical_ordinal] == 0U,
                    "the whole-layer raster has no duplicate cell");
        ++seen[cell.logical_ordinal];
      }
    }

    if (expected_m_group + 1U <
        kernels::kSm87BulkV2NvFp4GateUpWholeP40MGroups) {
      for (const auto requesters : b_requesters) {
        test.expect(requesters == 4U,
                    "each full-cohort B pair has four M-lane requesters");
      }
      for (const auto requesters : a_requesters) {
        test.expect(requesters == 8U,
                    "each full-cohort A tile has eight N-lane requesters");
      }
    } else {
      test.expect(a_requesters[0U] == 8U && a_requesters[1U] == 0U &&
                      a_requesters[2U] == 0U && a_requesters[3U] == 0U,
                  "the final M group exposes only the exact M64 tail");
      for (const auto requesters : b_requesters) {
        test.expect(requesters == 1U,
                    "the terminal M64 tail has one requester per N lane");
      }
    }
  }

  test.expect(active == 170'000U && masked == 816U,
              "170000 active and 816 masked cells close the schedule");
  for (const auto count : seen) {
    test.expect(count == 1U,
                "all 625 by 272 logical tiles are covered exactly once");
  }

  for (std::uint32_t m_group = 0U;
       m_group < kernels::kSm87BulkV2NvFp4GateUpWholeP40MGroups;
       ++m_group) {
    const std::uint32_t first_cohort =
        m_group * kernels::kSm87BulkV2NvFp4GateUpWholeP40NGroups;
    const std::uint32_t last_cohort =
        first_cohort +
        kernels::kSm87BulkV2NvFp4GateUpWholeP40NGroups - 1U;
    const auto first =
        kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_cell(first_cohort, 0U);
    const auto last =
        kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_cell(last_cohort, 0U);
    test.expect(first.m_group == m_group && last.m_group == m_group,
                "all N cohorts remain inside one M group before advancing");
    test.expect((m_group & 1U) == 0U
                    ? first.n_group == 0U && last.n_group == 33U
                    : first.n_group == 33U && last.n_group == 0U,
                "successive M groups reverse N traversal without a jump");
  }
}

void test_oracle_narrowing_mapping(TestContext& test) {
  std::uint32_t active = 0U;
  for (std::uint32_t cta = 0U;
       cta < kernels::kSm87BulkV2NvFp4GateUpWholeP40PersistentCtas;
       ++cta) {
    const auto cell =
        kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_cell(0U, cta, 1U,
                                                            1U);
    active += cell.active ? 1U : 0U;
    test.expect(cell.active == (cta == 0U),
                "the M64xN64 oracle narrows only CTA zero");
  }
  test.expect(active == 1U,
              "the same-kernel oracle contains one exact logical cell");
  test.expect(
      !kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_cell(1U, 0U, 1U, 1U)
           .active &&
          !kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_cell(0U, 32U)
               .active &&
          !kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_cell(0U, 0U, 0U,
                                                               1U)
               .active,
      "out-of-domain cohort, CTA, and zero dimensions fail closed");
}

void test_public_abi_ranges(TestContext& test) {
  constexpr std::uintptr_t kBase = 0x10'0000'0000ULL;
  constexpr std::uintptr_t kInput = kBase;
  constexpr std::uintptr_t kH =
      kInput + kernels::kSm87BulkV2NvFp4GateUpWholeP40InputBytes;
  constexpr std::uintptr_t kPayload =
      kH + kernels::kSm87BulkV2NvFp4GateUpWholeP40HBytes;
  constexpr std::uintptr_t kPayloadEnd =
      kPayload + kernels::kSm87BulkV2NvFp4GateUpPayloadBytes;
  constexpr std::uintptr_t kControl = kPayloadEnd;
  constexpr std::uintptr_t kCancellation =
      kControl +
      sizeof(kernels::Sm87BulkV2NvFp4GateUpWholeP40DeviceControl);

  const auto ranges = [](const std::uintptr_t input,
                         const std::uintptr_t h,
                         const std::uintptr_t payload,
                         const std::uintptr_t payload_end,
                         const std::uintptr_t control,
                         const std::uintptr_t cancellation) {
    return kernels::
        sm87_bulk_v2_nvfp4_gate_up_whole_p40_public_byte_ranges(
            reinterpret_cast<const std::uint16_t*>(input),
            reinterpret_cast<std::uint16_t*>(h), payload, payload_end, true,
            reinterpret_cast<kernels::
                                 Sm87BulkV2NvFp4GateUpWholeP40DeviceControl*>(
                control),
            reinterpret_cast<const std::uint32_t*>(cancellation));
  };
  const auto valid = [&ranges](const std::uintptr_t input,
                               const std::uintptr_t h,
                               const std::uintptr_t payload,
                               const std::uintptr_t payload_end,
                               const std::uintptr_t control,
                               const std::uintptr_t cancellation) {
    return kernels::
        sm87_bulk_v2_nvfp4_gate_up_whole_p40_public_byte_ranges_valid(
            ranges(input, h, payload, payload_end, control, cancellation));
  };

  const auto adjacent =
      ranges(kInput, kH, kPayload, kPayloadEnd, kControl, kCancellation);
  test.expect(
      kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_public_byte_ranges_valid(
          adjacent),
      "exact adjacent fixed-P40 spans are legal and non-overlapping");
  test.expect(adjacent.input.begin == kInput && adjacent.input.end == kH &&
                  adjacent.h.begin == kH && adjacent.h.end == kPayload &&
                  adjacent.payload.begin == kPayload &&
                  adjacent.payload.end == kPayloadEnd &&
                  adjacent.control.begin == kControl &&
                  adjacent.control.end == kCancellation &&
                  adjacent.cancellation.begin == kCancellation &&
                  adjacent.cancellation.end == kCancellation + 4U,
              "the public ABI computes every fixed endpoint exactly");

  const auto without_cancellation =
      kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_public_byte_ranges(
          reinterpret_cast<const std::uint16_t*>(kInput),
          reinterpret_cast<std::uint16_t*>(kH), kPayload, kPayloadEnd, true,
          reinterpret_cast<
              kernels::Sm87BulkV2NvFp4GateUpWholeP40DeviceControl*>(
              kControl),
          nullptr);
  test.expect(
      kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_public_byte_ranges_valid(
          without_cancellation) &&
          !without_cancellation.cancellation_present,
      "a null optional cancellation word leaves required ranges valid");

  test.expect(!valid(kInput, kInput + 16U, kPayload, kPayloadEnd, kControl,
                     kCancellation),
              "input/H alias fails closed");
  test.expect(!valid(kInput, kH, kInput + 16U,
                     kInput + 16U +
                         kernels::kSm87BulkV2NvFp4GateUpPayloadBytes,
                     kControl, kCancellation),
              "input/payload alias fails closed");
  test.expect(!valid(kInput, kH, kPayload, kPayloadEnd, kInput + 64U,
                     kCancellation),
              "input/control alias fails closed");
  test.expect(!valid(kInput, kH, kPayload, kPayloadEnd, kControl,
                     kInput + 64U),
              "input/cancellation alias fails closed");
  test.expect(!valid(kInput, kH, kH + 16U,
                     kH + 16U +
                         kernels::kSm87BulkV2NvFp4GateUpPayloadBytes,
                     kControl, kCancellation),
              "H/payload alias fails closed");
  test.expect(!valid(kInput, kH, kPayload, kPayloadEnd, kH + 64U,
                     kCancellation),
              "H/control alias fails closed");
  test.expect(!valid(kInput, kH, kPayload, kPayloadEnd, kControl, kH + 64U),
              "H/cancellation alias fails closed");
  test.expect(!valid(kInput, kH, kPayload, kPayloadEnd, kPayload + 64U,
                     kCancellation),
              "payload/control alias fails closed");
  test.expect(!valid(kInput, kH, kPayload, kPayloadEnd, kControl,
                     kPayload + 64U),
              "payload/cancellation alias fails closed");
  test.expect(!valid(kInput, kH, kPayload, kPayloadEnd, kControl,
                     kControl),
              "control/cancellation alias fails closed");

  constexpr std::uintptr_t kMax =
      std::numeric_limits<std::uintptr_t>::max();
  test.expect(!valid(kMax & ~std::uintptr_t{15U}, kH, kPayload, kPayloadEnd,
                     kControl, kCancellation),
              "input end overflow fails closed");
  test.expect(!valid(kInput, kMax & ~std::uintptr_t{15U}, kPayload,
                     kPayloadEnd, kControl, kCancellation),
              "H end overflow fails closed");
  test.expect(!valid(kInput, kH, kPayload, kPayloadEnd,
                     kMax & ~std::uintptr_t{63U}, kCancellation),
              "control end overflow fails closed");
  test.expect(!valid(kInput, kH, kPayload, kPayloadEnd, kControl,
                     kMax & ~std::uintptr_t{3U}),
              "cancellation end overflow fails closed");
  test.expect(!valid(kInput, kH, kPayload, kPayloadEnd - 1U, kControl,
                     kCancellation),
              "a non-exact payload endpoint fails closed");
  test.expect(!valid(kInput + 2U, kH, kPayload, kPayloadEnd, kControl,
                     kCancellation) &&
                  !valid(kInput, kH + 2U, kPayload, kPayloadEnd, kControl,
                         kCancellation) &&
                  !valid(kInput, kH, kPayload + 1U, kPayloadEnd + 1U,
                         kControl, kCancellation) &&
                  !valid(kInput, kH, kPayload, kPayloadEnd, kControl + 4U,
                         kCancellation) &&
                  !valid(kInput, kH, kPayload, kPayloadEnd, kControl,
                         kCancellation + 2U),
              "misaligned input/H/payload/control/cancellation fail closed");
}

}  // namespace

int main() {
  TestContext test;
  test_frozen_contract(test);
  test_exhaustive_raster(test);
  test_oracle_narrowing_mapping(test);
  test_public_abi_ranges(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " Gate+Up whole-P40 host checks failed\n";
    return 1;
  }
  std::cout << "Gate+Up whole-P40 host contract: PASS\n";
  return 0;
}
