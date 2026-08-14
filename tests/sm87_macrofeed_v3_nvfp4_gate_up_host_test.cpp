#include "q3x/kernels/sm87_macrofeed_v3_nvfp4_gate_up.h"

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

  constexpr auto plan = sm87_macrofeed_v3_nvfp4_gate_up_plan(40'000U);
  ok &= expect(plan.valid(), "exact P40000 plan must be valid");
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_plan(39'999U).valid(),
               "non-P40000 plan must fail closed");
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_plan(40'001U).valid(),
               "oversized plan must fail closed");
  ok &= expect(plan.grid_m == 313U && plan.grid_n == 68U &&
                   plan.logical_tasks == 21'284U && plan.tail_rows == 64U,
               "P40 task and tail geometry changed");
  ok &= expect(plan.payload_bytes == 100'270'080U,
               "canonical GateThenUp payload byte count changed");
  ok &= expect(plan.noncooperative_persistent_queue &&
                   plan.shared_a_stages &&
                   plan.independent_branch_payloads &&
                   plan.independent_branch_scales &&
                   plan.independent_branch_accumulators &&
                   plan.canonical_gate_then_up &&
                   plan.cta_private_bf16_epilogue && plan.tail_predicated &&
                   !plan.fallback_permitted && plan.t0_t1_only &&
                   !plan.production_dispatch_eligible,
               "independent Gate+Up admission boundary changed");
  ok &= expect(kSm87MacroFeedV3NvFp4GateUpWarpM == 128U &&
                   kSm87MacroFeedV3NvFp4GateUpWarpN == 32U &&
                   kSm87MacroFeedV3NvFp4GateUpThreads == 256U &&
                   kSm87MacroFeedV3NvFp4GateUpWarpsPerBranch == 4U &&
                   kSm87MacroFeedV3NvFp4GateUpPipelineStages == 3U &&
                   kSm87MacroFeedV3NvFp4GateUpPersistentCtas == 16U &&
                   kSm87MacroFeedV3NvFp4GateUpRasterGroupM == 2U,
               "V3 Gate+Up physical feed changed");

  Sm87MacroFeedV3NvFp4GateUpPayloadReceipt payload_receipt{};
  payload_receipt.plan_identity = kSm87MacroFeedV3NvFp4GateUpIdentity;
  payload_receipt.payload_identity = 0xabcU;
  payload_receipt.gate_source_identity = 0x101U;
  payload_receipt.up_source_identity = 0x202U;
  payload_receipt.device_ordinal = 0;
  payload_receipt.payload_begin = 0x2'0000'0000ULL;
  payload_receipt.payload_bytes =
      kSm87MacroFeedV3NvFp4GateUpPayloadBytes;
  payload_receipt.payload_end =
      payload_receipt.payload_begin + payload_receipt.payload_bytes;
  payload_receipt.gate_partition_bytes =
      kSm87MacroFeedV3NvFp4GateUpPartitionBytes;
  payload_receipt.up_partition_bytes =
      kSm87MacroFeedV3NvFp4GateUpPartitionBytes;
  payload_receipt.canonical_consumer_n64_k16_lane_component_v1 = true;
  payload_receipt.canonical_gate_then_up_partition_order = true;
  payload_receipt.independent_tensor_scales = true;
  payload_receipt.host_bytes_authenticated_before_copy = true;
  payload_receipt.device_readback_authenticated = true;
  payload_receipt.allocation_retained_for_launch = true;
  payload_receipt.receipt_identity =
      sm87_macrofeed_v3_nvfp4_gate_up_compute_payload_receipt_identity(
          payload_receipt);
  ok &= expect(sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
                   payload_receipt),
               "complete paired payload receipt must validate");
  auto mutated_receipt = payload_receipt;
  mutated_receipt.up_source_identity = mutated_receipt.gate_source_identity;
  mutated_receipt.receipt_identity =
      sm87_macrofeed_v3_nvfp4_gate_up_compute_payload_receipt_identity(
          mutated_receipt);
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
                   mutated_receipt),
               "aliased Gate/Up source identities must be rejected");
  mutated_receipt = payload_receipt;
  mutated_receipt.canonical_gate_then_up_partition_order = false;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
                   mutated_receipt),
               "partition-order substitution must be rejected");
  mutated_receipt = payload_receipt;
  mutated_receipt.device_readback_authenticated = false;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
                   mutated_receipt),
               "unverified device payload must be rejected");

  Sm87MacroFeedV3NvFp4GateUpArguments arguments{};
  arguments.input = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x1'0000'0000ULL));
  arguments.payload = reinterpret_cast<const std::uint8_t*>(
      payload_receipt.payload_begin);
  arguments.payload_bytes = kSm87MacroFeedV3NvFp4GateUpPayloadBytes;
  arguments.gate_tensor_scale = 0.5F;
  arguments.up_tensor_scale = 0.25F;
  arguments.token_count = kSm87MacroFeedV3NvFp4GateUpTokens;
  arguments.output = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x3'0000'0000ULL));
  arguments.payload_receipt = payload_receipt;
  ok &= expect(sm87_macrofeed_v3_nvfp4_gate_up_arguments_valid(arguments),
               "authenticated disjoint Gate+Up arguments must pass");
  auto bad_arguments = arguments;
  bad_arguments.output = const_cast<std::uint16_t*>(arguments.input);
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_arguments_valid(
                   bad_arguments),
               "input/output alias must fail closed");
  bad_arguments = arguments;
  bad_arguments.gate_tensor_scale = 0.0F;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_arguments_valid(
                   bad_arguments),
               "non-positive Gate tensor scale must fail closed");
  bad_arguments = arguments;
  bad_arguments.up_tensor_scale = 0.0F;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_arguments_valid(
                   bad_arguments),
               "non-positive Up tensor scale must fail closed");

  Sm87MacroFeedV3NvFp4GateUpCudaResources resources{};
  resources.identity = kSm87MacroFeedV3NvFp4GateUpIdentity;
  resources.device_ordinal = 0;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.binary_version = 87;
  resources.registers_per_thread = 255;
  resources.static_shared_bytes = 0U;
  resources.dynamic_shared_bytes =
      kSm87MacroFeedV3NvFp4GateUpDynamicSharedBytes;
  resources.local_bytes = 0U;
  resources.maximum_threads_per_block = 1'024;
  resources.active_blocks_per_sm = 1;
  resources.optin_shared_bytes_per_block = 102'400U;
  resources.kernel_compiled = true;
  ok &= expect(sm87_macrofeed_v3_nvfp4_gate_up_resource_gate(resources),
               "boundary resource observation must pass");
  auto bad_resources = resources;
  bad_resources.registers_per_thread = 256;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_resource_gate(
                   bad_resources),
               "register overflow must fail resource admission");
  bad_resources = resources;
  bad_resources.local_bytes = 1U;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_resource_gate(
                   bad_resources),
               "local memory must fail resource admission");
  bad_resources = resources;
  bad_resources.active_blocks_per_sm = 0;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_resource_gate(
                   bad_resources),
               "zero-residency kernel must fail resource admission");

  Sm87MacroFeedV3NvFp4GateUpStartupSeal startup_seal{};
  startup_seal.plan_identity = kSm87MacroFeedV3NvFp4GateUpIdentity;
  startup_seal.kernel_symbol_identity =
      kSm87MacroFeedV3NvFp4GateUpKernelSymbolIdentity;
  startup_seal.device_ordinal = resources.device_ordinal;
  startup_seal.compute_major = resources.compute_major;
  startup_seal.compute_minor = resources.compute_minor;
  startup_seal.sm_count = resources.sm_count;
  startup_seal.binary_version = resources.binary_version;
  startup_seal.registers_per_thread = resources.registers_per_thread;
  startup_seal.static_shared_bytes = resources.static_shared_bytes;
  startup_seal.dynamic_shared_bytes = resources.dynamic_shared_bytes;
  startup_seal.local_bytes = resources.local_bytes;
  startup_seal.maximum_threads_per_block =
      resources.maximum_threads_per_block;
  startup_seal.active_blocks_per_sm = resources.active_blocks_per_sm;
  startup_seal.optin_shared_bytes_per_block =
      resources.optin_shared_bytes_per_block;
  startup_seal.dynamic_shared_attribute_configured = true;
  startup_seal.static_resource_gate_passed = true;
  startup_seal.request_hot_static_queries_forbidden = true;
  startup_seal.t0_t1_only = true;
  startup_seal.production_dispatch_eligible = false;
  startup_seal.seal_identity =
      sm87_macrofeed_v3_nvfp4_gate_up_compute_startup_seal_identity(
          startup_seal);
  ok &= expect(sm87_macrofeed_v3_nvfp4_gate_up_startup_seal_valid(
                   startup_seal),
               "complete startup resource seal must validate");
  auto mutated_seal = startup_seal;
  mutated_seal.active_blocks_per_sm = 0;
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_startup_seal_valid(
                   mutated_seal),
               "startup seal substitution must be rejected");
  mutated_seal = startup_seal;
  mutated_seal.request_hot_static_queries_forbidden = false;
  mutated_seal.seal_identity =
      sm87_macrofeed_v3_nvfp4_gate_up_compute_startup_seal_identity(
          mutated_seal);
  ok &= expect(!sm87_macrofeed_v3_nvfp4_gate_up_startup_seal_valid(
                   mutated_seal),
               "hot-query prohibition must be part of the seal");

  Sm87MacroFeedV3NvFp4GateUpLaunchReceipt launch_receipt{
      kSm87MacroFeedV3NvFp4GateUpIdentity,
      payload_receipt.payload_identity,
      payload_receipt.gate_source_identity,
      payload_receipt.up_source_identity,
      kSm87MacroFeedV3NvFp4GateUpTokens,
      kSm87MacroFeedV3NvFp4GateUpLogicalTasks,
      kSm87MacroFeedV3NvFp4GateUpTailRows,
      1U,
      0U,
      true,
      true,
      true,
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
    std::cout << "sm87_macrofeed_v3_nvfp4_gate_up_host_test: PASS\n";
  }
  return ok ? 0 : 1;
}
