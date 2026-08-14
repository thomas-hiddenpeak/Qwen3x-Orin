#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_down_whole_p40.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace kernels = q3x::kernels;

namespace {

struct Test final {
  int failures = 0;

  void expect(const bool condition, const char* const message) {
    if (!condition) {
      std::cerr << "FAIL: " << message << '\n';
      ++failures;
    }
  }
};

[[nodiscard]] constexpr kernels::
    Sm87BulkV2NvFp4DownWholeP40CodeEvidence valid_code_evidence() {
  kernels::Sm87BulkV2NvFp4DownWholeP40CodeEvidence evidence;
  evidence.elf_identity = 1U;
  evidence.canonical_sass_hash = 2U;
  evidence.instruction_rows = 1U;
  evidence.text_bytes = 16U;
  evidence.launch_bounds_256_2 = true;
  evidence.cooperative_grid_sync_present = true;
  evidence.cp_async_cg_present = true;
  evidence.two_stage_s2r_present = true;
  evidence.full_k_accumulator_present = true;
  evidence.split_k_or_partial_c_absent = true;
  return evidence;
}

}  // namespace

int main() {
  Test test;
  const auto traffic =
      kernels::sm87_bulk_v2_nvfp4_down_whole_p40_traffic_contract();
  test.expect(traffic.valid, "traffic contract is valid");
  test.expect(traffic.physical_launches == 1U,
              "whole P40 owns exactly one physical launch");
  test.expect(traffic.useful_output_tiles == 625ULL * 20ULL,
              "P40 output domain contains exactly 625x20 tiles");
  test.expect(traffic.useful_h_requests ==
                  4ULL * traffic.unique_h_services,
              "each useful H service has four N-lane requesters");
  test.expect(traffic.full_cohort_b_requests ==
                  8ULL * traffic.full_cohort_unique_b_services,
              "each full-cohort B service has eight M-lane requesters");
  test.expect(traffic.tail_b_requests == 20ULL * 272ULL,
              "one-row tail performs only useful B requests");
  test.expect(traffic.masked_tail_lanes == 7U,
              "final M cohort masks seven M lanes");
  test.expect(traffic.cooperative_grid_barriers == 2U &&
                  traffic.cooperative_grid_barriers_per_super_wave == 0U,
              "cooperative barriers are launch-terminal only, never per wave");
  test.expect(traffic.exact_unique_coverage && traffic.four_way_h_cohort &&
                  traffic.eight_way_b_cohort &&
                  traffic.tail_publication_safe,
              "coverage, cohort reuse, and tail invariants are frozen");
  test.expect(traffic.theoretical_l2_service_only &&
                  !traffic.measured_cross_cta_residency,
              "cohort address equivalence is not mislabeled measured L2 residency");

  std::array<std::uint8_t,
             kernels::kSm87BulkV2NvFp4DownWholeP40LogicalOutputTiles>
      coverage{};
  std::uint64_t owners = 0U;
  std::uint64_t masked_cta_wave_items = 0U;
  for (std::uint32_t wave = 0U;
       wave < kernels::kSm87BulkV2NvFp4DownWholeP40SuperWaves;
       ++wave) {
    for (std::uint32_t block = 0U;
         block < kernels::kSm87BulkV2NvFp4DownWholeP40PersistentCtas;
         ++block) {
      const auto item =
          kernels::sm87_bulk_v2_nvfp4_down_whole_p40_work_item(wave, block);
      test.expect(item.valid, "every physical wave/CTA map is valid");
      test.expect(item.m_lane == block / 4U && item.n_lane == block % 4U,
                  "block lane is the explicit 8M x 4N map");
      if (item.output_owner) {
        const std::size_t index =
            static_cast<std::size_t>(item.m_tile) *
                kernels::kSm87BulkV2NvFp4DownWholeP40NTiles +
            item.n_tile;
        test.expect(index < coverage.size(),
                    "output owner lies inside P40 tile domain");
        if (index < coverage.size()) {
          ++coverage[index];
        }
        test.expect(item.h_request_valid && item.b_service_participant,
                    "every owner participates in both service families");
        ++owners;
      } else {
        test.expect(!item.h_request_valid && !item.b_service_participant,
                    "masked tail lane issues no H/B request");
        ++masked_cta_wave_items;
      }
    }

    const std::uint32_t m_cohort =
        wave / kernels::kSm87BulkV2NvFp4DownWholeP40NCohorts;
    const bool full_m_cohort =
        m_cohort < kernels::kSm87BulkV2NvFp4DownWholeP40FullMCohorts;
    if (full_m_cohort) {
      for (std::uint32_t m_lane = 0U; m_lane < 8U; ++m_lane) {
        const auto first =
            kernels::sm87_bulk_v2_nvfp4_down_whole_p40_work_item(
                wave, m_lane * 4U);
        for (std::uint32_t n_lane = 1U; n_lane < 4U; ++n_lane) {
          const auto peer =
              kernels::sm87_bulk_v2_nvfp4_down_whole_p40_work_item(
                  wave, m_lane * 4U + n_lane);
          test.expect(
              kernels::sm87_bulk_v2_nvfp4_down_whole_p40_same_h_service(
                  first, peer),
              "four N lanes share one H address service");
        }
      }
      for (std::uint32_t n_lane = 0U; n_lane < 4U; ++n_lane) {
        const auto first =
            kernels::sm87_bulk_v2_nvfp4_down_whole_p40_work_item(
                wave, n_lane);
        for (std::uint32_t m_lane = 1U; m_lane < 8U; ++m_lane) {
          const auto peer =
              kernels::sm87_bulk_v2_nvfp4_down_whole_p40_work_item(
                  wave, m_lane * 4U + n_lane);
          test.expect(
              kernels::sm87_bulk_v2_nvfp4_down_whole_p40_same_b_service(
                  first, peer),
              "eight M lanes share one packed B+scale address service");
        }
      }
    }
  }
  test.expect(owners ==
                  kernels::kSm87BulkV2NvFp4DownWholeP40LogicalOutputTiles,
              "mapping has exactly 12,500 output owners");
  test.expect(masked_cta_wave_items == 7ULL * 4ULL * 5ULL,
              "only the seven tail M lanes are masked across five N waves");
  for (const auto count : coverage) {
    test.expect(count == 1U, "each (M64,N256) output tile has one owner");
  }

  test.expect(
      !kernels::sm87_bulk_v2_nvfp4_down_whole_p40_work_item(
           kernels::kSm87BulkV2NvFp4DownWholeP40SuperWaves, 0U)
           .valid &&
          !kernels::sm87_bulk_v2_nvfp4_down_whole_p40_work_item(
               0U, kernels::kSm87BulkV2NvFp4DownWholeP40PersistentCtas)
               .valid,
      "out-of-range wave/CTA maps fail closed");

  auto code = valid_code_evidence();
  test.expect(
      kernels::sm87_bulk_v2_nvfp4_down_whole_p40_code_evidence_valid(code),
      "zero-stack/zero-spill SASS evidence is structurally valid");
  ++code.spill_load_bytes;
  test.expect(
      !kernels::sm87_bulk_v2_nvfp4_down_whole_p40_code_evidence_valid(code),
      "any spill evidence fails the hard resource gate");

  kernels::Sm87BulkV2NvFp4DownWholeP40Resources resources;
  resources.kernel_symbol_identity =
      kernels::kSm87BulkV2NvFp4DownWholeP40KernelSymbolIdentity;
  resources.device_ordinal = 0;
  resources.binary_version = 87;
  resources.registers_per_thread = 128;
  resources.dynamic_shared_bytes =
      kernels::kSm87BulkV2NvFp4DownWholeP40DynamicSharedBytes;
  resources.maximum_threads_per_block = 256;
  resources.active_blocks_per_sm = 2;
  resources.cooperative_grid_capacity = 32;
  resources.code = valid_code_evidence();
  resources.kernel_compiled = true;
  resources.cooperative_launch_supported = true;
  resources.dynamic_shared_attribute_configured = true;
  resources.resource_gate_passed = false;
  test.expect(
      !kernels::sm87_bulk_v2_nvfp4_down_whole_p40_resources_valid(resources),
      "a structurally valid public resource record cannot self-promote");
  resources.resource_gate_passed = true;
  test.expect(
      kernels::sm87_bulk_v2_nvfp4_down_whole_p40_resources_valid(resources),
      "the structural predicate is diagnostic only, never launch authority");
  resources.registers_per_thread = 129;
  test.expect(
      !kernels::sm87_bulk_v2_nvfp4_down_whole_p40_resources_valid(resources),
      "129 registers fails the hard two-CTA resource contract");
  resources.registers_per_thread = 128;
  resources.dynamic_shared_attribute_configured = false;
  test.expect(
      !kernels::sm87_bulk_v2_nvfp4_down_whole_p40_resources_valid(resources),
      ">48-KiB launch cannot pass without the exact-symbol startup opt-in");
  resources.dynamic_shared_attribute_configured = true;
  ++resources.kernel_symbol_identity;
  test.expect(
      !kernels::sm87_bulk_v2_nvfp4_down_whole_p40_resources_valid(resources),
      "startup capability is bound to the exact kernel symbol identity");

  return test.failures == 0 ? 0 : 1;
}
