#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_whole_p40.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernels = q3x::kernels;

namespace {

using Role = kernels::Sm87TargetAotProjectionRole;

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

[[nodiscard]] float float_from_bf16(const std::uint16_t bits) noexcept {
  const std::uint32_t bits32 = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits32, sizeof(result));
  return result;
}

[[nodiscard]] float exact_marlin_e4m3(const std::uint8_t code) noexcept {
  const int sign = (code & 0x80U) == 0U ? 1 : -1;
  const unsigned int magnitude = code & 0x7fU;
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 7U;
  if (exponent == 0U) {
    return static_cast<float>(sign) *
           std::ldexp(static_cast<float>(mantissa), -9);
  }
  return static_cast<float>(sign) *
         std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                    static_cast<int>(exponent) - 7);
}

void test_raw_fp8_contract(TestContext& test) {
  constexpr std::uint16_t kCompensatedOne =
      kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
          0x3f80'0000U);
  static_assert(kCompensatedOne == 0x7b80U);
  const float compensation = float_from_bf16(kCompensatedOne);
  for (unsigned int raw = 0U; raw <= 0xffU; ++raw) {
    const auto code = static_cast<std::uint8_t>(raw);
    const float observed =
        float_from_bf16(
            kernels::
                sm87_bulk_v2_fp8_whole_p40_raw_code_to_biased_bf16_bits(
                    code)) *
        compensation;
    test.expect(observed == exact_marlin_e4m3(code),
                "all raw FP8 codes preserve exact Marlin semantics");
  }
  test.expect(exact_marlin_e4m3(0x7fU) == 480.0F &&
                  exact_marlin_e4m3(0xffU) == -480.0F,
              "terminal FP8 codes remain signed 480, never canonical NaN");
}

void test_role_plans(TestContext& test) {
  constexpr std::array<Role, 3U> roles{{
      Role::kFp8GdnQkvZ, Role::kFp8FullQkv,
      Role::kFp8AttentionOutput}};
  constexpr std::array<std::uint32_t, 3U> logical{{
      80'000U, 70'000U, 25'000U}};
  constexpr std::array<std::uint32_t, 3U> cohorts{{
      2'512U, 2'198U, 785U}};
  constexpr std::array<std::uint32_t, 3U> masked{{384U, 336U, 120U}};
  for (std::size_t index = 0U; index < roles.size(); ++index) {
    const auto plan =
        kernels::sm87_bulk_v2_fp8_whole_p40_role_plan(roles[index]);
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(roles[index]);
    test.expect(plan.valid && layout.valid(),
                "all three fixed FP8 roles have valid frozen plans");
    test.expect(plan.input_features == layout.input_features &&
                    plan.projected_output_features ==
                        layout.projected_output_features &&
                    plan.partition_count == layout.partition_count &&
                    plan.payload_bytes == layout.payload_bytes,
                "whole-P40 role plans preserve authenticated ABI dimensions");
    test.expect(plan.logical_cells == logical[index] &&
                    plan.cohorts == cohorts[index] &&
                    plan.masked_cells == masked[index],
                "role-specific P40 cell/cohort/tail accounting is exact");
    for (std::size_t partition = 0U;
         partition < plan.partition_count; ++partition) {
      test.expect(plan.partition_n128_tiles[partition] ==
                          2U * layout.partitions[partition].n_tiles &&
                      plan.partition_payload_offsets[partition] ==
                          layout.partitions[partition].payload_offset &&
                      plan.k_tiles ==
                          layout.partitions[partition].k_tiles,
                  "N128 halves address the unchanged N256 payload cells");
    }
  }
}

void test_mapping_for_role(TestContext& test, const Role role) {
  const auto plan =
      kernels::sm87_bulk_v2_fp8_whole_p40_role_plan(role);
  std::vector<std::uint8_t> coverage(plan.logical_cells, 0U);
  std::uint32_t active = 0U;
  std::uint32_t masked = 0U;
  for (std::uint32_t cohort = 0U; cohort < plan.cohorts; ++cohort) {
    for (std::uint32_t cta = 0U;
         cta < kernels::kSm87BulkV2Fp8WholeP40PersistentCtas; ++cta) {
      const auto item = kernels::sm87_bulk_v2_fp8_whole_p40_work_item(
          role, cohort, cta);
      test.expect(item.valid, "every fixed cohort lane has a valid mapping");
      if (!item.active) {
        ++masked;
        test.expect(item.m_tile >=
                            kernels::kSm87BulkV2Fp8WholeP40MTiles ||
                        item.n_tile >= plan.n_tiles,
                    "a masked lane is outside useful M or N before work");
        continue;
      }
      ++active;
      test.expect(item.logical_ordinal < coverage.size(),
                  "active mapping stays in the logical cell domain");
      if (item.logical_ordinal < coverage.size()) {
        ++coverage[item.logical_ordinal];
      }
    }
  }
  test.expect(active == plan.logical_cells && masked == plan.masked_cells,
              "mapping enumerates exact useful and masked cells");
  for (const auto count : coverage) {
    test.expect(count == 1U, "every logical output tile has exactly one CTA");
  }

  // One full cohort has eight A requesters per M tile and four B requesters
  // per N tile.  This proves address equivalence only, not cache residency.
  const auto a0 = kernels::sm87_bulk_v2_fp8_whole_p40_work_item(
      role, 0U, 0U);
  const auto a7 = kernels::sm87_bulk_v2_fp8_whole_p40_work_item(
      role, 0U, 7U);
  const auto b0 = kernels::sm87_bulk_v2_fp8_whole_p40_work_item(
      role, 0U, 0U);
  const auto b24 = kernels::sm87_bulk_v2_fp8_whole_p40_work_item(
      role, 0U, 24U);
  test.expect(a0.active && a7.active && a0.m_tile == a7.m_tile &&
                  a0.n_tile != a7.n_tile,
              "eight N lanes request one identical A tile");
  test.expect(b0.active && b24.active && b0.n_tile == b24.n_tile &&
                  b0.m_tile != b24.m_tile,
              "four M lanes request one identical B tile");
}

void test_mapping_and_traffic(TestContext& test) {
  constexpr std::array<Role, 3U> roles{{
      Role::kFp8GdnQkvZ, Role::kFp8FullQkv,
      Role::kFp8AttentionOutput}};
  for (const auto role : roles) {
    test_mapping_for_role(test, role);
    const auto traffic =
        kernels::sm87_bulk_v2_fp8_whole_p40_traffic(role);
    test.expect(
        kernels::sm87_bulk_v2_fp8_whole_p40_traffic_valid(traffic),
        "every role passes its exact launch and traffic contract");
    test.expect(traffic.physical_launches == 1U &&
                    traffic.legacy_segments == 0U &&
                    traffic.grid_barriers == 2U &&
                    traffic.grid_barriers_per_cohort == 0U,
                "one role is one launch with no hot-loop grid barrier");
    test.expect(traffic.same_address_service_is_theoretical_only &&
                    traffic.complete_role_footprint_exceeds_l2 &&
                    !traffic.measured_cross_cta_residency,
                "same-address arithmetic cannot claim L2 residency");
  }
  const auto output = kernels::sm87_bulk_v2_fp8_whole_p40_traffic(
      Role::kFp8AttentionOutput);
  test.expect(output.stationary_a_group_footprint_bytes == 3'145'728ULL,
              "O keeps the four-row stationary-A footprint below 4 MiB");
  test.expect(output.input_footprint_bytes == 491'520'000ULL &&
                  output.payload_footprint_bytes == 31'457'280ULL &&
                  output.stationary_hypothesis_total_service_bytes ==
                      5'430'312'960ULL,
              "O global stationary hypothesis is explicitly byte-accounted");
}

void test_family_ledger(TestContext& test) {
  const auto& manifest =
      kernels::kSm87BulkV2Fp8WholeP40FrozenFamilyManifest;
  test.expect(
      kernels::sm87_bulk_v2_fp8_whole_p40_family_manifest_valid(manifest),
      "128-role whole-P40 manifest validates");
  test.expect(manifest.role_count == 128U &&
                  manifest.physical_launches == 128U &&
                  manifest.logical_cells == 6'560'000ULL,
              "64 input plus 64 O roles remain exactly 128 launches");
  auto changed = manifest;
  changed.roles[0U].role = Role::kFp8FullQkv;
  test.expect(
      !kernels::sm87_bulk_v2_fp8_whole_p40_family_manifest_valid(changed),
      "an even-ordinal layer-role substitution fails closed");
  changed = manifest;
  --changed.roles[0U].logical_cells;
  ++changed.roles[2U].logical_cells;
  test.expect(
      !kernels::sm87_bulk_v2_fp8_whole_p40_family_manifest_valid(changed),
      "compensated per-role cell mutations cannot preserve validity");
  test.expect(kernels::sm87_bulk_v2_fp8_whole_p40_family_contract_valid(
                  kernels::kSm87BulkV2Fp8WholeP40FrozenFamilyContract),
              "independent default-off family identity is frozen");
}

void test_range_and_shape_guards(TestContext& test) {
  auto* const input =
      reinterpret_cast<const std::uint16_t*>(0x0000'0001'0000'0000ULL);
  constexpr std::uintptr_t kPayloadBegin = 0x0000'0002'0000'0000ULL;
  constexpr std::uintptr_t kPayloadEnd =
      kPayloadBegin + 83'886'080ULL;
  auto* const primary =
      reinterpret_cast<std::uint16_t*>(0x0000'0003'0000'0000ULL);
  auto* const control = reinterpret_cast<void*>(0x0000'0005'0000'0000ULL);
  auto* const cancel = reinterpret_cast<void*>(0x0000'0006'0000'0000ULL);

  test.expect(kernels::sm87_bulk_v2_fp8_whole_p40_ranges_valid(
                  Role::kFp8GdnQkvZ, input, kPayloadBegin, kPayloadEnd,
                  primary, nullptr, nullptr, control, cancel),
              "disjoint complete P40 ranges pass host admission");
  test.expect(!kernels::sm87_bulk_v2_fp8_whole_p40_ranges_valid(
                  Role::kFp8GdnQkvZ, input, kPayloadBegin,
                  kPayloadEnd - 1U, primary, nullptr, nullptr, control,
                  cancel),
              "truncated authenticated payload fails closed");
  test.expect(!kernels::sm87_bulk_v2_fp8_whole_p40_ranges_valid(
                  Role::kFp8GdnQkvZ, input, kPayloadBegin, kPayloadEnd,
                  reinterpret_cast<void*>(
                      0x0000'0001'0000'1000ULL),
                  nullptr, nullptr, control, cancel),
              "input/output overlap fails before enqueue");
  test.expect(!kernels::sm87_bulk_v2_fp8_whole_p40_ranges_valid(
                  Role::kFp8GdnQkvZ, input, kPayloadBegin, kPayloadEnd,
                  primary, nullptr, nullptr,
                  reinterpret_cast<void*>(kPayloadBegin + 64U), cancel),
              "control state may not alias authenticated payload bytes");
  test.expect(kernels::sm87_bulk_v2_fp8_whole_p40_output_shape_valid(
                  Role::kFp8GdnQkvZ, primary, nullptr, nullptr) &&
                  !kernels::sm87_bulk_v2_fp8_whole_p40_output_shape_valid(
                      Role::kFp8GdnQkvZ, primary, primary, nullptr),
              "single-output role rejects hidden extra publications");
  auto* const secondary =
      reinterpret_cast<std::uint16_t*>(0x0000'0007'0000'0000ULL);
  auto* const tertiary =
      reinterpret_cast<std::uint16_t*>(0x0000'0008'0000'0000ULL);
  test.expect(kernels::sm87_bulk_v2_fp8_whole_p40_output_shape_valid(
                  Role::kFp8FullQkv, primary, secondary, tertiary) &&
                  !kernels::sm87_bulk_v2_fp8_whole_p40_output_shape_valid(
                      Role::kFp8FullQkv, primary, primary, tertiary),
              "Full-QKV keeps three partition-private output ranges");
}

[[nodiscard]] kernels::Sm87BulkV2Fp8WholeP40CodeEvidence
valid_code_evidence(const std::uint64_t identity) noexcept {
  kernels::Sm87BulkV2Fp8WholeP40CodeEvidence code;
  code.elf_identity = identity;
  code.sass_identity = identity ^ 0x5133'5832'5034'3046ULL;
  code.launch_bounds_256_2 = true;
  code.contains_cp_async_cg = true;
  code.contains_ldmatrix = true;
  code.contains_bf16_mma = true;
  code.same_kernel_exact_oracle = true;
  code.no_partial_c_symbol = true;
  code.valid = true;
  return code;
}

[[nodiscard]] kernels::Sm87BulkV2Fp8WholeP40KernelResources
valid_resources(const Role role, const std::uint64_t identity) noexcept {
  kernels::Sm87BulkV2Fp8WholeP40KernelResources result;
  result.role = role;
  result.binary_version = 87;
  result.registers_per_thread = 128;
  result.dynamic_shared_bytes =
      kernels::kSm87BulkV2Fp8WholeP40DynamicSharedBytes;
  result.maximum_threads_per_block = 256;
  result.active_blocks_per_sm = 2;
  result.cooperative_grid_capacity = 32;
  result.code = valid_code_evidence(identity);
  result.cooperative_launch_supported = true;
  result.runtime_envelope_observed = true;
  result.external_static_record_consistent = true;
  result.admission_capability_issued = false;
  result.numerical_contract_qualified = false;
  result.production_dispatch_eligible = false;
  return result;
}

void test_fail_closed_resource_observation(TestContext& test) {
  auto resource = valid_resources(Role::kFp8GdnQkvZ, 1U);
  test.expect(
      kernels::
          sm87_bulk_v2_fp8_whole_p40_resource_observation_consistent(
              resource),
      "the exact <=128-register two-CTA envelope validates");
  resource.registers_per_thread = 129;
  test.expect(
      !kernels::
          sm87_bulk_v2_fp8_whole_p40_resource_observation_consistent(
              resource),
      "129 registers fail closed");
  resource = valid_resources(Role::kFp8GdnQkvZ, 1U);
  resource.active_blocks_per_sm = 1;
  resource.cooperative_grid_capacity = 16;
  test.expect(
      !kernels::
          sm87_bulk_v2_fp8_whole_p40_resource_observation_consistent(
              resource),
              "one CTA per SM fails the whole-P40 resource observation");
  resource = valid_resources(Role::kFp8GdnQkvZ, 1U);
  resource.code.spill_store_bytes = 4U;
  test.expect(
      !kernels::
          sm87_bulk_v2_fp8_whole_p40_resource_observation_consistent(
              resource),
      "one spill byte rejects the role");

  kernels::Sm87BulkV2Fp8WholeP40FamilyResources family;
  family.roles = {{
      valid_resources(Role::kFp8GdnQkvZ, 1U),
      valid_resources(Role::kFp8FullQkv, 2U),
      valid_resources(Role::kFp8AttentionOutput, 3U)}};
  family.all_runtime_envelopes_observed = true;
  family.all_external_static_records_consistent = true;
  family.admission_capability_issued = false;
  family.numerical_contract_qualified = false;
  family.production_dispatch_eligible = false;
  test.expect(
      kernels::
          sm87_bulk_v2_fp8_whole_p40_family_resource_observation_consistent(
              family),
      "all three role kernels must pass atomically");
  family.roles[2U].registers_per_thread = 129;
  test.expect(
      !kernels::
          sm87_bulk_v2_fp8_whole_p40_family_resource_observation_consistent(
              family),
      "one failed role rejects the complete family");
}

}  // namespace

int main() {
  static_assert(kernels::kSm87BulkV2Fp8WholeP40TileM == 64U);
  static_assert(kernels::kSm87BulkV2Fp8WholeP40TileN == 128U);
  static_assert(kernels::kSm87BulkV2Fp8WholeP40TileK == 64U);
  static_assert(kernels::kSm87BulkV2Fp8WholeP40DynamicSharedBytes ==
                49'152U);
  static_assert(sizeof(kernels::Sm87BulkV2Fp8WholeP40DeviceControl) == 64U);
  static_assert(
      (kernels::kSm87BulkV2Fp8WholeP40RequiredPolicy &
       kernels::kSm87BulkV2Fp8WholeP40NoProductionSelector) != 0U);
  TestContext test;
  test_raw_fp8_contract(test);
  test_role_plans(test);
  test_mapping_and_traffic(test);
  test_family_ledger(test);
  test_range_and_shape_guards(test);
  test_fail_closed_resource_observation(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " failure(s)\n";
    return 1;
  }
  std::cout << "SM87 bulk-v2 FP8 whole-P40000 host contract passed\n";
  return 0;
}
