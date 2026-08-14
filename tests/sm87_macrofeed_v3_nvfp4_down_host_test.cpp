#include "q3x/kernels/sm87_macrofeed_v3_nvfp4_down.h"

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

bool expect(const bool condition, const char* const message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

}  // namespace

int main() {
  using namespace q3x::kernels;
  bool ok = true;

  constexpr auto plan = sm87_macrofeed_v3_nvfp4_down_plan(40'000U);
  ok &= expect(plan.valid(), "exact P40000 plan must be valid");
  ok &= expect(!sm87_macrofeed_v3_nvfp4_down_plan(39'999U).valid(),
               "non-P40000 plan must fail closed");
  ok &= expect(!sm87_macrofeed_v3_nvfp4_down_plan(40'001U).valid(),
               "oversized plan must fail closed");
  ok &= expect(plan.grid_m == 313U && plan.grid_n == 20U &&
                   plan.logical_tasks == 6'260U && plan.tail_rows == 64U,
               "P40 task and tail geometry changed");
  ok &= expect(plan.payload_bytes == 50'135'040U,
               "canonical Down payload byte count changed");
  ok &= expect(plan.noncooperative_persistent_queue && plan.n_stationary &&
                   plan.tail_predicated && !plan.fallback_permitted &&
                   plan.t0_t1_only && !plan.production_dispatch_eligible,
               "independent admission boundary changed");
  ok &= expect(kSm87MacroFeedV3NvFp4DownWarpM == 128U &&
                   kSm87MacroFeedV3NvFp4DownWarpN == 32U &&
                   kSm87MacroFeedV3NvFp4DownThreads == 256U &&
                   kSm87MacroFeedV3NvFp4DownPipelineStages == 3U &&
                   kSm87MacroFeedV3NvFp4DownPersistentCtas == 16U,
               "V3 physical dataflow changed");

  Sm87MacroFeedV3NvFp4DownPayloadReceipt payload_receipt{};
  payload_receipt.plan_identity = kSm87MacroFeedV3NvFp4DownIdentity;
  payload_receipt.payload_identity = 0xabcU;
  payload_receipt.device_ordinal = 0;
  payload_receipt.payload_begin = 0x2'0000'0000ULL;
  payload_receipt.payload_bytes = kSm87MacroFeedV3NvFp4DownPayloadBytes;
  payload_receipt.payload_end =
      payload_receipt.payload_begin + payload_receipt.payload_bytes;
  payload_receipt.canonical_consumer_n64_k16_lane_component_v1 = true;
  payload_receipt.host_bytes_authenticated_before_copy = true;
  payload_receipt.device_readback_authenticated = true;
  payload_receipt.allocation_retained_for_launch = true;
  payload_receipt.receipt_identity =
      sm87_macrofeed_v3_nvfp4_down_compute_payload_receipt_identity(
          payload_receipt);
  ok &= expect(sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
                   payload_receipt),
               "complete payload receipt must validate");
  auto mutated_receipt = payload_receipt;
  mutated_receipt.payload_identity += 1U;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
                   mutated_receipt),
               "receipt substitution must be rejected");
  mutated_receipt = payload_receipt;
  mutated_receipt.device_readback_authenticated = false;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
                   mutated_receipt),
               "unverified device payload must be rejected");

  Sm87MacroFeedV3NvFp4DownArguments arguments{};
  arguments.input = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x1'0000'0000ULL));
  arguments.payload = reinterpret_cast<const std::uint8_t*>(
      payload_receipt.payload_begin);
  arguments.payload_bytes = kSm87MacroFeedV3NvFp4DownPayloadBytes;
  arguments.tensor_scale = 0.5F;
  arguments.token_count = kSm87MacroFeedV3NvFp4DownTokens;
  arguments.residual = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x3'0000'0000ULL));
  arguments.payload_receipt = payload_receipt;
  ok &= expect(sm87_macrofeed_v3_nvfp4_down_arguments_valid(arguments),
               "structurally authenticated disjoint arguments must pass");
  auto bad_arguments = arguments;
  bad_arguments.residual = const_cast<std::uint16_t*>(arguments.input);
  ok &= expect(!sm87_macrofeed_v3_nvfp4_down_arguments_valid(bad_arguments),
               "input/residual alias must fail closed");
  bad_arguments = arguments;
  bad_arguments.tensor_scale = 0.0F;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_down_arguments_valid(bad_arguments),
               "non-positive tensor scale must fail closed");

  Sm87MacroFeedV3NvFp4DownCudaResources resources{};
  resources.identity = kSm87MacroFeedV3NvFp4DownIdentity;
  resources.device_ordinal = 0;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.binary_version = 87;
  resources.registers_per_thread = 255;
  resources.static_shared_bytes = 0U;
  resources.dynamic_shared_bytes =
      kSm87MacroFeedV3NvFp4DownDynamicSharedBytes;
  resources.local_bytes = 0U;
  resources.maximum_threads_per_block = 1'024;
  resources.active_blocks_per_sm = 1;
  resources.optin_shared_bytes_per_block = 102'400U;
  resources.kernel_compiled = true;
  ok &= expect(sm87_macrofeed_v3_nvfp4_down_resource_gate(resources),
               "boundary resource observation must pass");
  auto bad_resources = resources;
  bad_resources.registers_per_thread = 256;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_down_resource_gate(bad_resources),
               "register overflow must fail resource admission");
  bad_resources = resources;
  bad_resources.local_bytes = 1U;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_down_resource_gate(bad_resources),
               "local memory must fail resource admission");
  bad_resources = resources;
  bad_resources.active_blocks_per_sm = 0;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_down_resource_gate(bad_resources),
               "zero-residency kernel must fail resource admission");

  Sm87MacroFeedV3NvFp4DownLaunchReceipt launch_receipt{
      kSm87MacroFeedV3NvFp4DownIdentity,
      payload_receipt.payload_identity,
      kSm87MacroFeedV3NvFp4DownTokens,
      kSm87MacroFeedV3NvFp4DownLogicalTasks,
      kSm87MacroFeedV3NvFp4DownTailRows,
      1U,
      0U,
      true,
      false,
      true,
      false};
  ok &= expect(launch_receipt.valid_enqueue_receipt(),
               "zero-fallback enqueue receipt must validate");
  launch_receipt.fallback_launches = 1U;
  ok &= expect(!launch_receipt.valid_enqueue_receipt(),
               "fallback receipt must be impossible");

  if (ok) {
    std::cout << "sm87_macrofeed_v3_nvfp4_down_host_test: PASS\n";
  }
  return ok ? 0 : 1;
}
