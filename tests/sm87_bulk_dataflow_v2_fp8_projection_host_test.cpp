#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_projection.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

namespace kernels = q3x::kernels;

namespace {

using Role = kernels::Sm87TargetAotProjectionRole;

static_assert(kernels::kSm87BulkV2Fp8TileM == 64U);
static_assert(kernels::kSm87BulkV2Fp8TileN == 256U);
static_assert(kernels::kSm87BulkV2Fp8TileK == 64U);
static_assert(kernels::kSm87BulkV2Fp8PipelineStages == 4U);
static_assert(kernels::kSm87BulkV2Fp8RegisterStages == 2U);
static_assert(kernels::kSm87BulkV2Fp8DynamicSharedBytes == 98'304U);
static_assert(kernels::kSm87BulkV2Fp8LogicalRoleCount == 128U);
static_assert(kernels::kSm87BulkV2Fp8PhysicalLaunches == 5'120U);
static_assert(kernels::sm87_bulk_v2_fp8_segment_token_count_valid(1'024U));
static_assert(kernels::sm87_bulk_v2_fp8_segment_token_count_valid(64U));
static_assert(!kernels::sm87_bulk_v2_fp8_segment_token_count_valid(512U));
static_assert(
    (kernels::kSm87BulkV2Fp8RequiredPolicy &
     kernels::kSm87BulkV2Fp8NoProductionSelector) != 0U);
static_assert(
    (kernels::kSm87BulkV2Fp8RequiredPolicy &
     kernels::kSm87BulkV2Fp8NoRequestRepack) != 0U);
static_assert(
    (kernels::kSm87BulkV2Fp8RequiredPolicy &
     kernels::kSm87BulkV2Fp8NoL2ResidencyClaim) != 0U);
static_assert(
    (kernels::kSm87BulkV2Fp8RequiredPolicy &
     kernels::kSm87BulkV2Fp8AXorSharedLayout) != 0U);
static_assert(
    (kernels::kSm87BulkV2Fp8RequiredPolicy &
     kernels::kSm87BulkV2Fp8TwoSlotS2R) != 0U);
static_assert(
    (kernels::kSm87BulkV2Fp8RequiredPolicy &
     kernels::kSm87BulkV2Fp8NMajorM64Cohort) != 0U);
static_assert(kernels::sm87_bulk_v2_fp8_family_manifest_valid(
    kernels::kSm87BulkV2Fp8FrozenFamilyManifest));

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

[[nodiscard]] float float_from_bits(const std::uint32_t bits) noexcept {
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

[[nodiscard]] float float_from_bf16(const std::uint16_t bits) noexcept {
  return float_from_bits(static_cast<std::uint32_t>(bits) << 16U);
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

void test_raw_code_semantics(TestContext& test) {
  constexpr std::uint16_t kCompensatedOne =
      kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
          0x3f80'0000U);
  static_assert(kCompensatedOne == 0x7b80U);
  const float compensation = float_from_bf16(kCompensatedOne);
  for (unsigned int raw = 0U; raw <= 0xffU; ++raw) {
    const auto code = static_cast<std::uint8_t>(raw);
    const std::uint16_t biased =
        kernels::sm87_bulk_v2_fp8_raw_code_to_biased_bf16_bits(code);
    const float observed = float_from_bf16(biased) * compensation;
    const float expected = exact_marlin_e4m3(code);
    test.expect(observed == expected,
                "all 256 raw FP8 codes preserve Marlin bit semantics");
  }
  test.expect(exact_marlin_e4m3(0x7fU) == 480.0F,
              "terminal positive code is +480, never NaN");
  test.expect(exact_marlin_e4m3(0xffU) == -480.0F,
              "terminal negative code is -480, never NaN");
}

void test_role_plans(TestContext& test) {
  constexpr std::array<Role, 3U> roles{{
      Role::kFp8GdnQkvZ, Role::kFp8FullQkv,
      Role::kFp8AttentionOutput}};
  constexpr std::array<std::uint64_t, 3U> expected_tiles{{
      40'000U, 35'000U, 12'500U}};
  for (std::size_t index = 0U; index < roles.size(); ++index) {
    const auto role = roles[index];
    const auto plan = kernels::sm87_bulk_v2_fp8_role_plan(role);
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(role);
    test.expect(plan.valid && layout.valid(),
                "every family role has a frozen valid plan and payload");
    test.expect(plan.input_features == layout.input_features &&
                    plan.projected_output_features ==
                        layout.projected_output_features,
                "v2 execution dimensions match authenticated payload");
    test.expect(plan.partition_count == layout.partition_count,
                "v2 partition count matches authenticated payload");
    test.expect(plan.k_tiles == layout.partitions[0U].k_tiles,
                "v2 full-K ownership matches authenticated payload");
    test.expect(plan.logical_cta_tiles == expected_tiles[index],
                "P40 logical CTA accounting is exact per role");
    for (std::size_t partition = 0U;
         partition < plan.partition_count; ++partition) {
      test.expect(plan.partition_n_tiles[partition] ==
                      layout.partitions[partition].n_tiles &&
                      plan.partition_payload_offsets[partition] ==
                          layout.partitions[partition].payload_offset,
                  "v2 consumes the exact admitted partition bytes");
    }
  }
}

[[nodiscard]] std::uint64_t v2_manual_weight_byte_offset(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const std::size_t partition_index, const std::size_t n,
    const std::size_t k) noexcept {
  const auto& partition = layout.partitions[partition_index];
  const std::size_t n_tile = n / kernels::kSm87BulkV2Fp8TileN;
  const std::size_t k_tile = k / kernels::kSm87BulkV2Fp8TileK;
  const std::size_t n_in_tile = n % kernels::kSm87BulkV2Fp8TileN;
  const std::size_t k_in_tile = k % kernels::kSm87BulkV2Fp8TileK;
  const std::size_t global_n8_panel = n_in_tile / 8U;
  const std::size_t row = n_in_tile % 8U;
  const std::size_t k16 = k_in_tile / 16U;
  const std::size_t k_in_16 = k_in_tile % 16U;
  const std::size_t lane_in_group =
      kernels::sm87_target_aot_projection_packed_lane_in_group_for_k16(
          static_cast<std::uint32_t>(k_in_16));
  const std::size_t component =
      kernels::sm87_target_aot_projection_packed_component_for_k16(
          static_cast<std::uint32_t>(k_in_16));
  const std::size_t lane = row * 4U + lane_in_group;
  const std::size_t fragment = k16 * 32U + global_n8_panel;
  return partition.payload_offset +
         (n_tile * partition.k_tiles + k_tile) *
             kernels::kSm87BulkV2Fp8BBytesPerStage +
         fragment * 128U + lane * 4U + component;
}

void test_v2_addressing_matches_authenticated_layout(TestContext& test) {
  constexpr std::array<Role, 3U> roles{{
      Role::kFp8GdnQkvZ, Role::kFp8FullQkv,
      Role::kFp8AttentionOutput}};
  for (const auto role : roles) {
    const auto plan = kernels::sm87_bulk_v2_fp8_role_plan(role);
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(role);
    for (std::size_t partition_index = 0U;
         partition_index < layout.partition_count; ++partition_index) {
      const auto& partition = layout.partitions[partition_index];
      const std::array<std::size_t, 8U> columns{{
          0U, 31U, 32U, 63U, 64U, 165U, 255U,
          partition.output_features - 1U}};
      const std::array<std::size_t, 6U> ks{{
          0U, 15U, 16U, 63U, 64U,
          static_cast<std::size_t>(plan.input_features - 1U)}};
      for (const auto n : columns) {
        if (n >= partition.output_features) {
          continue;
        }
        for (const auto k : ks) {
          const auto authenticated =
              kernels::sm87_target_aot_projection_packed_weight_address(
                  layout, partition_index, n, k);
          test.expect(
              authenticated.valid &&
                  authenticated.byte_offset ==
                      v2_manual_weight_byte_offset(
                          layout, partition_index, n, k),
              "v2 N32-warp flattening matches the authenticated N64 payload");
        }
      }
    }
    const auto& last_partition =
        layout.partitions[layout.partition_count - 1U];
    const auto last =
        kernels::sm87_target_aot_projection_packed_weight_address(
            layout, layout.partition_count - 1U,
            last_partition.output_features - 1U,
            plan.input_features - 1U);
    test.expect(last.valid && last.byte_offset < layout.payload_bytes,
                "the final K tile remains inside the exact payload range");
  }
}

void test_family_manifest(TestContext& test) {
  const auto& manifest = kernels::kSm87BulkV2Fp8FrozenFamilyManifest;
  std::uint32_t gdn = 0U;
  std::uint32_t full = 0U;
  std::uint32_t output = 0U;
  std::uint32_t launches = 0U;
  std::uint64_t tiles = 0U;
  for (std::size_t ordinal = 0U; ordinal < manifest.role_count; ++ordinal) {
    const auto& descriptor = manifest.roles[ordinal];
    test.expect(descriptor.ordinal == ordinal &&
                    descriptor.layer == ordinal / 2U,
                "family execution order remains layer/role canonical");
    launches += descriptor.physical_launches;
    tiles += descriptor.logical_cta_tiles;
    if (descriptor.role == Role::kFp8GdnQkvZ) {
      ++gdn;
    } else if (descriptor.role == Role::kFp8FullQkv) {
      ++full;
    } else if (descriptor.role == Role::kFp8AttentionOutput) {
      ++output;
    }
  }
  test.expect(gdn == 48U && full == 16U && output == 64U,
              "family covers all 128 model projection roles");
  test.expect(launches == 5'120U && tiles == 3'280'000U,
              "family launch/tile accounting is closed");
  test.expect(!manifest.production_dispatch_eligible &&
                  !manifest.request_time_repack &&
                  manifest.reuses_authenticated_payload_bytes,
              "the base is default-off and reuses only authenticated bytes");
}

[[nodiscard]] kernels::Sm87BulkV2Fp8KernelResources
valid_resources(const Role role) noexcept {
  kernels::Sm87BulkV2Fp8KernelResources result;
  result.role = role;
  result.binary_version = 87;
  result.registers_per_thread = 192;
  result.static_shared_bytes = 0U;
  result.dynamic_shared_bytes =
      kernels::kSm87BulkV2Fp8DynamicSharedBytes;
  result.local_bytes = 0U;
  result.maximum_threads_per_block = 256;
  result.active_blocks_per_sm = 1;
  result.kernel_compiled = true;
  result.resource_gate_passed = true;
  result.numerical_contract_qualified = false;
  result.production_dispatch_eligible = false;
  return result;
}

void test_resource_gate(TestContext& test) {
  auto resources = valid_resources(Role::kFp8GdnQkvZ);
  test.expect(kernels::sm87_bulk_v2_fp8_kernel_resources_valid(resources),
              "the exact SM87 resource envelope validates");
  ++resources.local_bytes;
  test.expect(!kernels::sm87_bulk_v2_fp8_kernel_resources_valid(resources),
              "one local/spill byte is a hard rejection");
  resources = valid_resources(Role::kFp8GdnQkvZ);
  ++resources.static_shared_bytes;
  test.expect(!kernels::sm87_bulk_v2_fp8_kernel_resources_valid(resources),
              "unplanned static shared memory is a hard rejection");
  resources = valid_resources(Role::kFp8GdnQkvZ);
  resources.registers_per_thread = 256;
  test.expect(!kernels::sm87_bulk_v2_fp8_kernel_resources_valid(resources),
              "register allocation cannot cross the SM87 hard limit");
  resources = valid_resources(Role::kFp8GdnQkvZ);
  resources.active_blocks_per_sm = 2;
  test.expect(!kernels::sm87_bulk_v2_fp8_kernel_resources_valid(resources),
              "the M64N256 four-stage base freezes one CTA per SM");
  resources = valid_resources(Role::kFp8GdnQkvZ);
  resources.numerical_contract_qualified = true;
  test.expect(!kernels::sm87_bulk_v2_fp8_kernel_resources_valid(resources),
              "resource evidence cannot self-grant numerical qualification");
  resources = valid_resources(Role::kFp8GdnQkvZ);
  resources.production_dispatch_eligible = true;
  test.expect(!kernels::sm87_bulk_v2_fp8_kernel_resources_valid(resources),
              "resource evidence cannot open a production selector");

  kernels::Sm87BulkV2Fp8FamilyResources family;
  family.roles = {{
      valid_resources(Role::kFp8GdnQkvZ),
      valid_resources(Role::kFp8FullQkv),
      valid_resources(Role::kFp8AttentionOutput)}};
  family.all_compiled = true;
  family.resource_gate_passed = true;
  test.expect(kernels::sm87_bulk_v2_fp8_family_resources_valid(family),
              "all three kernels must pass the family gate together");
  family.roles[2U].local_bytes = 4U;
  test.expect(!kernels::sm87_bulk_v2_fp8_family_resources_valid(family),
              "one spilling role rejects the entire family");
}

void test_argument_boundary(TestContext& test) {
  auto* const primary =
      reinterpret_cast<std::uint16_t*>(0x0000'0020'0000'0000ULL);
  auto* const secondary =
      reinterpret_cast<std::uint16_t*>(0x0000'0030'0000'0000ULL);
  auto* const tertiary =
      reinterpret_cast<std::uint16_t*>(0x0000'0040'0000'0000ULL);
  test.expect(kernels::sm87_bulk_v2_fp8_output_shape_valid(
                  Role::kFp8GdnQkvZ, primary, nullptr, nullptr),
              "GDN QKVZ publishes one fused output interval");
  test.expect(kernels::sm87_bulk_v2_fp8_output_shape_valid(
                  Role::kFp8FullQkv, primary, secondary, tertiary),
              "Full QKV publishes three partition-private intervals");
  test.expect(!kernels::sm87_bulk_v2_fp8_output_shape_valid(
                  Role::kFp8FullQkv, primary, primary, tertiary),
              "Full QKV output intervals must be distinct");
  test.expect(kernels::sm87_bulk_v2_fp8_output_shape_valid(
                  Role::kFp8AttentionOutput, primary, nullptr, nullptr),
              "Attention O publishes one pure projection interval");

  kernels::Sm87BulkV2Fp8SegmentArguments arguments;
  arguments.role = Role::kFp8GdnQkvZ;
  arguments.input =
      reinterpret_cast<const std::uint16_t*>(0x0000'0010'0000'0000ULL);
  arguments.token_count = kernels::kSm87BulkV2Fp8MacroTokens;
  arguments.primary_output = primary;
  arguments.cuda_stream = reinterpret_cast<void*>(0x1000ULL);
  test.expect(!kernels::sm87_bulk_v2_fp8_segment_arguments_valid(arguments),
              "an unauthenticated payload cannot enter the executor");
}

}  // namespace

int main() {
  TestContext test;
  test_raw_code_semantics(test);
  test_role_plans(test);
  test_v2_addressing_matches_authenticated_layout(test);
  test_family_manifest(test);
  test_resource_gate(test);
  test_argument_boundary(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " failure(s)\n";
    return 1;
  }
  std::cout << "SM87 bulk-dataflow v2 FP8 host/static contract passed\n";
  return 0;
}
