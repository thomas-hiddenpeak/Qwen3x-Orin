#include "q3x/kernels/sm87_macrofeed_v3_fp8.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

namespace kernels = q3x::kernels;
using Role = kernels::Sm87TargetAotProjectionRole;

bool expect(const bool condition, const char* const message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

bool expect_partition(const kernels::Sm87MacroFeedV3Fp8Plan& plan,
                      const std::size_t n_tile,
                      const std::size_t partition,
                      const std::size_t local_n_tile) {
  const auto task =
      kernels::sm87_macrofeed_v3_fp8_partition_task(plan, n_tile);
  return expect(task.valid && task.partition == partition &&
                    task.local_n_tile == local_n_tile,
                "role partition boundary changed");
}

}  // namespace

int main() {
  bool ok = true;
  constexpr std::array<Role, 3U> kRoles{{
      Role::kFp8GdnQkvZ,
      Role::kFp8FullQkv,
      Role::kFp8AttentionOutput,
  }};

  ok &= expect(kernels::kSm87MacroFeedV3Fp8WarpM == 128U &&
                   kernels::kSm87MacroFeedV3Fp8WarpN == 32U &&
                   kernels::kSm87MacroFeedV3Fp8Threads == 256U &&
                   kernels::kSm87MacroFeedV3Fp8PersistentCtas == 16U,
               "M128N32 warp/16-CTA ownership changed");
  ok &= expect(kernels::kSm87MacroFeedV3Fp8DynamicSharedBytes == 98'304U,
               "three-stage FP8 shared ring changed");

  for (const Role role : kRoles) {
    const auto plan =
        kernels::sm87_macrofeed_v3_fp8_plan(role, 40'000U);
    ok &= expect(plan.valid(), "P40000 role plan must validate");
    ok &= expect(plan.grid_m == 313U &&
                     plan.logical_tasks == plan.grid_m * plan.grid_n,
                 "P40 M64 tail/task geometry changed");
    ok &= expect(plan.noncooperative_persistent_queue &&
                     plan.role_specific_raster && plan.tail_predicated &&
                     plan.authenticated_asset_zero_copy &&
                     plan.no_request_time_repacking &&
                     plan.no_request_time_jit &&
                     !plan.fallback_permitted &&
                     !plan.cublaslt_permitted && !plan.mtp_permitted &&
                     plan.exact_fp8_marlin_semantics && plan.t0_t1_only &&
                     !plan.production_dispatch_eligible,
                 "default-off exact constituent boundary changed");
    ok &= expect(!kernels::sm87_macrofeed_v3_fp8_plan(role, 39'999U).valid(),
                 "non-P40000 shape must fail closed");
  }

  constexpr auto kGdn = kernels::sm87_macrofeed_v3_fp8_plan(
      Role::kFp8GdnQkvZ, 40'000U);
  constexpr auto kFull = kernels::sm87_macrofeed_v3_fp8_plan(
      Role::kFp8FullQkv, 40'000U);
  constexpr auto kOutput = kernels::sm87_macrofeed_v3_fp8_plan(
      Role::kFp8AttentionOutput, 40'000U);
  ok &= expect(kGdn.grid_n == 64U && kGdn.k_tiles == 80U &&
                   kGdn.raster_group_m == 2U &&
                   kGdn.partition_payload_offsets[1U] == 52'428'800U,
               "GDN QKV/Z geometry changed");
  ok &= expect_partition(kGdn, 39U, 0U, 39U);
  ok &= expect_partition(kGdn, 40U, 1U, 0U);
  ok &= expect_partition(kGdn, 63U, 1U, 23U);
  ok &= expect(!kernels::sm87_macrofeed_v3_fp8_partition_task(kGdn, 64U)
                    .valid,
               "GDN N tile overflow must fail closed");

  ok &= expect(kFull.grid_n == 56U && kFull.k_tiles == 80U &&
                   kFull.raster_group_m == 2U &&
                   kFull.partition_payload_offsets[1U] == 62'914'560U &&
                   kFull.partition_payload_offsets[2U] == 68'157'440U,
               "Full Q/K/V geometry changed");
  ok &= expect_partition(kFull, 47U, 0U, 47U);
  ok &= expect_partition(kFull, 48U, 1U, 0U);
  ok &= expect_partition(kFull, 51U, 1U, 3U);
  ok &= expect_partition(kFull, 52U, 2U, 0U);
  ok &= expect_partition(kFull, 55U, 2U, 3U);

  ok &= expect(kOutput.grid_n == 20U && kOutput.k_tiles == 96U &&
                   kOutput.raster_group_m == 1U,
               "Attention O geometry changed");
  ok &= expect_partition(kOutput, 19U, 0U, 19U);

  for (std::uint32_t code = 0U; code < 256U; ++code) {
    ok &= expect(kernels::sm87_macrofeed_v3_fp8_weight_code_admitted(
                     static_cast<std::uint8_t>(code)),
                 "all raw FP8 weight bytes must be admitted");
    const std::uint16_t expected = static_cast<std::uint16_t>(
        ((code & 0x80U) << 8U) | ((code & 0x7fU) << 4U));
    ok &= expect(kernels::sm87_macrofeed_v3_fp8_bias_shift_bf16_bits(
                     static_cast<std::uint8_t>(code)) == expected,
                 "raw FP8 bias-shift mapping changed");
  }
  ok &= expect(kernels::sm87_macrofeed_v3_fp8_bias_shift_bf16_bits(0x7fU) ==
                   0x07f0U &&
                   kernels::sm87_macrofeed_v3_fp8_bias_shift_bf16_bits(
                       0xffU) == 0x87f0U,
               "terminal FP8 codes must remain signed 480 candidates");
  ok &= expect(
      kernels::sm87_target_aot_projection_mma_b_register_component(0U) ==
              0U &&
          kernels::sm87_target_aot_projection_mma_b_register_component(1U) ==
              2U &&
          kernels::sm87_target_aot_projection_mma_b_register_component(2U) ==
              1U &&
          kernels::sm87_target_aot_projection_mma_b_register_component(3U) ==
              3U,
      "persisted [K0,K8,K1,K9] to MMA [K0,K1,K8,K9] mapping changed");

  kernels::Sm87MacroFeedV3Fp8CudaResources resources{};
  resources.identity =
      kernels::sm87_macrofeed_v3_fp8_identity(Role::kFp8FullQkv);
  resources.role = Role::kFp8FullQkv;
  resources.device_ordinal = 0;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.binary_version = 87;
  resources.registers_per_thread = 228;
  resources.dynamic_shared_bytes = 98'304U;
  resources.maximum_threads_per_block = 1'024;
  resources.active_blocks_per_sm = 1;
  resources.optin_shared_bytes_per_block = 102'400U;
  resources.kernel_compiled = true;
  ok &= expect(kernels::sm87_macrofeed_v3_fp8_resource_gate(resources),
               "valid SM87 resource witness must pass");
  auto invalid_resources = resources;
  invalid_resources.local_bytes = 1U;
  ok &= expect(!kernels::sm87_macrofeed_v3_fp8_resource_gate(
                   invalid_resources),
               "local memory must fail resource admission");

  kernels::Sm87MacroFeedV3Fp8StartupSeal seal{};
  seal.resources = resources;
  seal.resources.static_resource_gate_passed = true;
  seal.dynamic_shared_attribute_set = true;
  seal.tactic_frozen_before_requests = true;
  seal.no_hot_device_queries = true;
  seal.no_hot_function_queries = true;
  seal.no_hot_occupancy_queries = true;
  seal.no_hot_pointer_queries = true;
  seal.no_hot_error_state_clear = true;
  seal.t0_t1_only = true;
  seal.production_dispatch_eligible = false;
  seal.seal_identity =
      kernels::sm87_macrofeed_v3_fp8_compute_startup_seal_identity(seal);
  ok &= expect(kernels::sm87_macrofeed_v3_fp8_startup_seal_valid(seal),
               "complete startup resource/tactic seal must validate");
  auto changed_seal = seal;
  changed_seal.no_hot_pointer_queries = false;
  changed_seal.seal_identity =
      kernels::sm87_macrofeed_v3_fp8_compute_startup_seal_identity(
          changed_seal);
  ok &= expect(!kernels::sm87_macrofeed_v3_fp8_startup_seal_valid(
                   changed_seal),
               "hot pointer query permission must fail closed");

  if (ok) {
    std::cout << "sm87_macrofeed_v3_fp8_host_test: PASS\n";
  }
  return ok ? 0 : 1;
}
