#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off T0/T1 constituent for the paired NVFP4 Gate+Up projection in
// AC-PREFILL-SM87-MACROFEED-v3.  It is deliberately disconnected from every
// runtime selector and accepts only the exact P40000 shape.  Gate and Up share
// the CTA's A stages, but retain independent packed partitions, tensor scales,
// FP32 accumulators, BF16-RNE publications, and the canonical SiLU(Gate)*Up
// consumer boundary.  No fallback or production dispatch is represented by
// this interface.
enum class Sm87MacroFeedV3NvFp4GateUpIdentity : std::uint64_t {
  kInvalid = 0U,
  kM128N256K64WarpM128N32PairedPersistent16V1 =
      0x5133'4d46'5633'4701ULL,
};

inline constexpr Sm87MacroFeedV3NvFp4GateUpIdentity
    kSm87MacroFeedV3NvFp4GateUpIdentity =
        Sm87MacroFeedV3NvFp4GateUpIdentity::
            kM128N256K64WarpM128N32PairedPersistent16V1;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpTokens = 40'000U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpInputFeatures =
    5'120U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpOutputFeatures =
    17'408U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpBranches = 2U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpBlockM = 128U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpBlockN = 256U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpBlockK = 64U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpWarpM = 128U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpWarpN = 32U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpThreads = 256U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpWarps = 8U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpWarpsPerBranch = 4U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpPipelineStages = 3U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpPersistentCtas = 16U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpSmCount = 16U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpRasterGroupM = 2U;
inline constexpr std::size_t
    kSm87MacroFeedV3NvFp4GateUpWeightBytesPerCell = 8'192U;
inline constexpr std::size_t
    kSm87MacroFeedV3NvFp4GateUpScaleBytesPerCell = 1'024U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpCellBytes = 9'216U;
inline constexpr std::size_t
    kSm87MacroFeedV3NvFp4GateUpDynamicSharedBytes = 76'800U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpGridM = 313U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpGridN = 68U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpKTiles = 80U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpLogicalTasks =
    21'284U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpTailRows = 64U;
inline constexpr std::size_t
    kSm87MacroFeedV3NvFp4GateUpPartitionBytes = 50'135'040U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpPayloadBytes =
    100'270'080U;
inline constexpr std::size_t
    kSm87MacroFeedV3NvFp4GateUpPayloadAlignment = 256U;
inline constexpr std::uint64_t
    kSm87MacroFeedV3NvFp4GateUpKernelSymbolIdentity =
        0x5133'4755'4b45'5201ULL;

// T1 preserves the canonical two-partition cell format and crosses the
// three-stage ring turnover, while reducing K to four K64 cells and N to one
// logical N256 task per branch.
inline constexpr std::size_t
    kSm87MacroFeedV3NvFp4GateUpTestInputFeatures = 256U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4GateUpTestKTiles = 4U;
inline constexpr std::size_t
    kSm87MacroFeedV3NvFp4GateUpTestPartitionBytes =
        kSm87MacroFeedV3NvFp4GateUpTestKTiles *
        kSm87MacroFeedV3NvFp4GateUpCellBytes;
inline constexpr std::size_t
    kSm87MacroFeedV3NvFp4GateUpTestPayloadBytes =
        kSm87MacroFeedV3NvFp4GateUpBranches *
        kSm87MacroFeedV3NvFp4GateUpTestPartitionBytes;

struct Sm87MacroFeedV3NvFp4GateUpPlan final {
  Sm87MacroFeedV3NvFp4GateUpIdentity identity =
      Sm87MacroFeedV3NvFp4GateUpIdentity::kInvalid;
  std::size_t token_count = 0U;
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t k_tiles = 0U;
  std::size_t logical_tasks = 0U;
  std::size_t tail_rows = 0U;
  std::size_t payload_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t raster_group_m = 0U;
  bool noncooperative_persistent_queue = false;
  bool shared_a_stages = false;
  bool independent_branch_payloads = false;
  bool independent_branch_scales = false;
  bool independent_branch_accumulators = false;
  bool canonical_gate_then_up = false;
  bool cta_private_bf16_epilogue = false;
  bool tail_predicated = false;
  bool fallback_permitted = true;
  bool t0_t1_only = false;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return identity == kSm87MacroFeedV3NvFp4GateUpIdentity &&
           token_count == kSm87MacroFeedV3NvFp4GateUpTokens &&
           grid_m == kSm87MacroFeedV3NvFp4GateUpGridM &&
           grid_n == kSm87MacroFeedV3NvFp4GateUpGridN &&
           k_tiles == kSm87MacroFeedV3NvFp4GateUpKTiles &&
           logical_tasks == kSm87MacroFeedV3NvFp4GateUpLogicalTasks &&
           tail_rows == kSm87MacroFeedV3NvFp4GateUpTailRows &&
           payload_bytes == kSm87MacroFeedV3NvFp4GateUpPayloadBytes &&
           dynamic_shared_bytes ==
               kSm87MacroFeedV3NvFp4GateUpDynamicSharedBytes &&
           raster_group_m == kSm87MacroFeedV3NvFp4GateUpRasterGroupM &&
           noncooperative_persistent_queue && shared_a_stages &&
           independent_branch_payloads && independent_branch_scales &&
           independent_branch_accumulators && canonical_gate_then_up &&
           cta_private_bf16_epilogue && tail_predicated &&
           !fallback_permitted && t0_t1_only &&
           !production_dispatch_eligible;
  }
};

[[nodiscard]] constexpr Sm87MacroFeedV3NvFp4GateUpPlan
sm87_macrofeed_v3_nvfp4_gate_up_plan(
    const std::size_t token_count) noexcept {
  if (token_count != kSm87MacroFeedV3NvFp4GateUpTokens) {
    return {};
  }
  return {kSm87MacroFeedV3NvFp4GateUpIdentity,
          token_count,
          kSm87MacroFeedV3NvFp4GateUpGridM,
          kSm87MacroFeedV3NvFp4GateUpGridN,
          kSm87MacroFeedV3NvFp4GateUpKTiles,
          kSm87MacroFeedV3NvFp4GateUpLogicalTasks,
          kSm87MacroFeedV3NvFp4GateUpTailRows,
          kSm87MacroFeedV3NvFp4GateUpPayloadBytes,
          kSm87MacroFeedV3NvFp4GateUpDynamicSharedBytes,
          kSm87MacroFeedV3NvFp4GateUpRasterGroupM,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          false};
}

struct Sm87MacroFeedV3NvFp4GateUpPayloadReceipt final {
  std::uint64_t receipt_identity = 0U;
  Sm87MacroFeedV3NvFp4GateUpIdentity plan_identity =
      Sm87MacroFeedV3NvFp4GateUpIdentity::kInvalid;
  std::uint64_t payload_identity = 0U;
  std::uint64_t gate_source_identity = 0U;
  std::uint64_t up_source_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::uintptr_t payload_begin = 0U;
  std::uintptr_t payload_end = 0U;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t gate_partition_bytes = 0U;
  std::uint64_t up_partition_bytes = 0U;
  bool canonical_consumer_n64_k16_lane_component_v1 = false;
  bool canonical_gate_then_up_partition_order = false;
  bool independent_tensor_scales = false;
  bool host_bytes_authenticated_before_copy = false;
  bool device_readback_authenticated = false;
  bool allocation_retained_for_launch = false;
};

[[nodiscard]] constexpr std::uint64_t
sm87_macrofeed_v3_gate_up_hash_u64(std::uint64_t hash,
                                  const std::uint64_t value) noexcept {
  for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
    hash ^= static_cast<std::uint8_t>(value >> (byte * 8U));
    hash *= 1'099'511'628'211ULL;
  }
  return hash;
}

[[nodiscard]] constexpr std::uint64_t
sm87_macrofeed_v3_nvfp4_gate_up_compute_payload_receipt_identity(
    const Sm87MacroFeedV3NvFp4GateUpPayloadReceipt& receipt) noexcept {
  std::uint64_t hash = 14'695'981'039'346'656'037ULL;
  const auto add = [&hash](const std::uint64_t value) constexpr {
    hash = sm87_macrofeed_v3_gate_up_hash_u64(hash, value);
  };
  add(static_cast<std::uint64_t>(receipt.plan_identity));
  add(receipt.payload_identity);
  add(receipt.gate_source_identity);
  add(receipt.up_source_identity);
  add(static_cast<std::uint32_t>(receipt.device_ordinal));
  add(receipt.payload_begin);
  add(receipt.payload_end);
  add(receipt.payload_bytes);
  add(receipt.gate_partition_bytes);
  add(receipt.up_partition_bytes);
  add(receipt.canonical_consumer_n64_k16_lane_component_v1);
  add(receipt.canonical_gate_then_up_partition_order);
  add(receipt.independent_tensor_scales);
  add(receipt.host_bytes_authenticated_before_copy);
  add(receipt.device_readback_authenticated);
  add(receipt.allocation_retained_for_launch);
  return hash;
}

[[nodiscard]] constexpr bool
sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
    const Sm87MacroFeedV3NvFp4GateUpPayloadReceipt& receipt) noexcept {
  return receipt.receipt_identity != 0U &&
         receipt.receipt_identity ==
             sm87_macrofeed_v3_nvfp4_gate_up_compute_payload_receipt_identity(
                 receipt) &&
         receipt.plan_identity == kSm87MacroFeedV3NvFp4GateUpIdentity &&
         receipt.payload_identity != 0U &&
         receipt.gate_source_identity != 0U &&
         receipt.up_source_identity != 0U &&
         receipt.gate_source_identity != receipt.up_source_identity &&
         receipt.device_ordinal >= 0 && receipt.payload_begin != 0U &&
         receipt.payload_bytes == kSm87MacroFeedV3NvFp4GateUpPayloadBytes &&
         receipt.payload_begin <=
             std::numeric_limits<std::uintptr_t>::max() -
                 receipt.payload_bytes &&
         receipt.payload_end == receipt.payload_begin + receipt.payload_bytes &&
         receipt.payload_begin %
                 kSm87MacroFeedV3NvFp4GateUpPayloadAlignment ==
             0U &&
         receipt.gate_partition_bytes ==
             kSm87MacroFeedV3NvFp4GateUpPartitionBytes &&
         receipt.up_partition_bytes ==
             kSm87MacroFeedV3NvFp4GateUpPartitionBytes &&
         receipt.canonical_consumer_n64_k16_lane_component_v1 &&
         receipt.canonical_gate_then_up_partition_order &&
         receipt.independent_tensor_scales &&
         receipt.host_bytes_authenticated_before_copy &&
         receipt.device_readback_authenticated &&
         receipt.allocation_retained_for_launch;
}

struct Sm87MacroFeedV3NvFp4GateUpCudaResources final {
  Sm87MacroFeedV3NvFp4GateUpIdentity identity =
      Sm87MacroFeedV3NvFp4GateUpIdentity::kInvalid;
  std::int32_t device_ordinal = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::int32_t sm_count = 0;
  std::int32_t binary_version = 0;
  std::int32_t registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  std::int32_t maximum_threads_per_block = 0;
  std::int32_t active_blocks_per_sm = 0;
  std::size_t optin_shared_bytes_per_block = 0U;
  bool kernel_compiled = false;
  bool static_resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool
sm87_macrofeed_v3_nvfp4_gate_up_resource_gate(
    const Sm87MacroFeedV3NvFp4GateUpCudaResources& resources) noexcept {
  return resources.identity == kSm87MacroFeedV3NvFp4GateUpIdentity &&
         resources.device_ordinal >= 0 && resources.compute_major == 8 &&
         resources.compute_minor == 7 &&
         resources.sm_count ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV3NvFp4GateUpSmCount) &&
         resources.binary_version == 87 && resources.kernel_compiled &&
         resources.registers_per_thread > 0 &&
         resources.registers_per_thread <= 255 &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kSm87MacroFeedV3NvFp4GateUpDynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV3NvFp4GateUpThreads) &&
         resources.active_blocks_per_sm >= 1 &&
         resources.optin_shared_bytes_per_block >=
             kSm87MacroFeedV3NvFp4GateUpDynamicSharedBytes &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

// Startup creates this integrity seal after configuring the exact kernel's
// dynamic-shared attribute and querying its static resources and occupancy.
// A request-hot launch accepts only this sealed observation and performs no
// cudaGetDevice, function-attribute, occupancy, or pointer-attribute query.
// The seal is a coherence checksum, not a secret or production authority.
struct Sm87MacroFeedV3NvFp4GateUpStartupSeal final {
  std::uint64_t seal_identity = 0U;
  Sm87MacroFeedV3NvFp4GateUpIdentity plan_identity =
      Sm87MacroFeedV3NvFp4GateUpIdentity::kInvalid;
  std::uint64_t kernel_symbol_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::int32_t sm_count = 0;
  std::int32_t binary_version = 0;
  std::int32_t registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  std::int32_t maximum_threads_per_block = 0;
  std::int32_t active_blocks_per_sm = 0;
  std::size_t optin_shared_bytes_per_block = 0U;
  bool dynamic_shared_attribute_configured = false;
  bool static_resource_gate_passed = false;
  bool request_hot_static_queries_forbidden = false;
  bool t0_t1_only = false;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr std::uint64_t
sm87_macrofeed_v3_nvfp4_gate_up_compute_startup_seal_identity(
    const Sm87MacroFeedV3NvFp4GateUpStartupSeal& seal) noexcept {
  std::uint64_t hash = 14'695'981'039'346'656'037ULL;
  const auto add = [&hash](const std::uint64_t value) constexpr {
    hash = sm87_macrofeed_v3_gate_up_hash_u64(hash, value);
  };
  add(static_cast<std::uint64_t>(seal.plan_identity));
  add(seal.kernel_symbol_identity);
  add(static_cast<std::uint32_t>(seal.device_ordinal));
  add(static_cast<std::uint32_t>(seal.compute_major));
  add(static_cast<std::uint32_t>(seal.compute_minor));
  add(static_cast<std::uint32_t>(seal.sm_count));
  add(static_cast<std::uint32_t>(seal.binary_version));
  add(static_cast<std::uint32_t>(seal.registers_per_thread));
  add(seal.static_shared_bytes);
  add(seal.dynamic_shared_bytes);
  add(seal.local_bytes);
  add(static_cast<std::uint32_t>(seal.maximum_threads_per_block));
  add(static_cast<std::uint32_t>(seal.active_blocks_per_sm));
  add(seal.optin_shared_bytes_per_block);
  add(seal.dynamic_shared_attribute_configured);
  add(seal.static_resource_gate_passed);
  add(seal.request_hot_static_queries_forbidden);
  add(seal.t0_t1_only);
  add(seal.production_dispatch_eligible);
  return hash;
}

[[nodiscard]] constexpr bool
sm87_macrofeed_v3_nvfp4_gate_up_startup_seal_valid(
    const Sm87MacroFeedV3NvFp4GateUpStartupSeal& seal) noexcept {
  Sm87MacroFeedV3NvFp4GateUpCudaResources resources{};
  resources.identity = seal.plan_identity;
  resources.device_ordinal = seal.device_ordinal;
  resources.compute_major = seal.compute_major;
  resources.compute_minor = seal.compute_minor;
  resources.sm_count = seal.sm_count;
  resources.binary_version = seal.binary_version;
  resources.registers_per_thread = seal.registers_per_thread;
  resources.static_shared_bytes = seal.static_shared_bytes;
  resources.dynamic_shared_bytes = seal.dynamic_shared_bytes;
  resources.local_bytes = seal.local_bytes;
  resources.maximum_threads_per_block = seal.maximum_threads_per_block;
  resources.active_blocks_per_sm = seal.active_blocks_per_sm;
  resources.optin_shared_bytes_per_block =
      seal.optin_shared_bytes_per_block;
  resources.kernel_compiled = true;
  resources.static_resource_gate_passed = seal.static_resource_gate_passed;
  resources.numerical_contract_qualified = false;
  resources.production_dispatch_eligible = false;
  return seal.seal_identity != 0U &&
         seal.seal_identity ==
             sm87_macrofeed_v3_nvfp4_gate_up_compute_startup_seal_identity(
                 seal) &&
         seal.plan_identity == kSm87MacroFeedV3NvFp4GateUpIdentity &&
         seal.kernel_symbol_identity ==
             kSm87MacroFeedV3NvFp4GateUpKernelSymbolIdentity &&
         seal.dynamic_shared_attribute_configured &&
         seal.static_resource_gate_passed &&
         seal.request_hot_static_queries_forbidden && seal.t0_t1_only &&
         !seal.production_dispatch_eligible &&
         sm87_macrofeed_v3_nvfp4_gate_up_resource_gate(resources);
}

struct Sm87MacroFeedV3NvFp4GateUpArguments final {
  const std::uint16_t* input = nullptr;
  const std::uint8_t* payload = nullptr;
  std::size_t payload_bytes = 0U;
  float gate_tensor_scale = 0.0F;
  float up_tensor_scale = 0.0F;
  std::size_t token_count = 0U;
  std::uint16_t* output = nullptr;
  void* cuda_stream = nullptr;
  Sm87MacroFeedV3NvFp4GateUpPayloadReceipt payload_receipt{};
};

struct Sm87MacroFeedV3NvFp4GateUpLaunchReceipt final {
  Sm87MacroFeedV3NvFp4GateUpIdentity identity =
      Sm87MacroFeedV3NvFp4GateUpIdentity::kInvalid;
  std::uint64_t payload_identity = 0U;
  std::uint64_t gate_source_identity = 0U;
  std::uint64_t up_source_identity = 0U;
  std::size_t token_count = 0U;
  std::size_t logical_tasks = 0U;
  std::size_t tail_rows = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  std::uint32_t fallback_launches = 0U;
  bool shared_a_stages = false;
  bool independent_branch_publications = false;
  bool canonical_gate_then_up = false;
  bool launch_enqueued = false;
  bool completion_observed = false;
  bool t0_t1_only = false;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid_enqueue_receipt() const noexcept {
    return identity == kSm87MacroFeedV3NvFp4GateUpIdentity &&
           payload_identity != 0U && gate_source_identity != 0U &&
           up_source_identity != 0U &&
           gate_source_identity != up_source_identity &&
           token_count == kSm87MacroFeedV3NvFp4GateUpTokens &&
           logical_tasks == kSm87MacroFeedV3NvFp4GateUpLogicalTasks &&
           tail_rows == kSm87MacroFeedV3NvFp4GateUpTailRows &&
           physical_kernel_launches == 1U && fallback_launches == 0U &&
           shared_a_stages && independent_branch_publications &&
           canonical_gate_then_up && launch_enqueued &&
           !completion_observed && t0_t1_only &&
           !production_dispatch_eligible;
  }
};

[[nodiscard]] bool sm87_macrofeed_v3_nvfp4_gate_up_arguments_valid(
    const Sm87MacroFeedV3NvFp4GateUpArguments& arguments) noexcept;

[[nodiscard]] int query_sm87_macrofeed_v3_nvfp4_gate_up_cuda_resources(
    Sm87MacroFeedV3NvFp4GateUpCudaResources* resources) noexcept;

[[nodiscard]] int seal_sm87_macrofeed_v3_nvfp4_gate_up_startup(
    Sm87MacroFeedV3NvFp4GateUpStartupSeal* seal) noexcept;

[[nodiscard]] int launch_sm87_macrofeed_v3_nvfp4_gate_up_cuda(
    const Sm87MacroFeedV3NvFp4GateUpArguments& arguments,
    Sm87MacroFeedV3NvFp4GateUpLaunchReceipt* receipt) noexcept;

[[nodiscard]] int launch_sm87_macrofeed_v3_nvfp4_gate_up_sealed_cuda(
    const Sm87MacroFeedV3NvFp4GateUpArguments& arguments,
    const Sm87MacroFeedV3NvFp4GateUpStartupSeal& startup_seal,
    Sm87MacroFeedV3NvFp4GateUpLaunchReceipt* receipt) noexcept;

// T1 numerical helper. It launches one M128 x N256 Gate+Up tile over four
// canonical K64 cells per independent partition. It is not a production or
// fallback interface.
[[nodiscard]] int launch_sm87_macrofeed_v3_nvfp4_gate_up_tile_test_cuda(
    const std::uint16_t* input_m128_k256,
    const std::uint8_t* canonical_gate_then_up_payload,
    float gate_tensor_scale, float up_tensor_scale, std::size_t valid_rows,
    std::uint16_t* output_m128_n256, void* cuda_stream) noexcept;

static_assert(kSm87MacroFeedV3NvFp4GateUpGridM *
                  kSm87MacroFeedV3NvFp4GateUpBlockM ==
              40'064U);
static_assert(kSm87MacroFeedV3NvFp4GateUpGridN *
                  kSm87MacroFeedV3NvFp4GateUpBlockN ==
              kSm87MacroFeedV3NvFp4GateUpOutputFeatures);
static_assert(kSm87MacroFeedV3NvFp4GateUpKTiles *
                  kSm87MacroFeedV3NvFp4GateUpBlockK ==
              kSm87MacroFeedV3NvFp4GateUpInputFeatures);
static_assert(kSm87MacroFeedV3NvFp4GateUpGridM *
                  kSm87MacroFeedV3NvFp4GateUpGridN ==
              kSm87MacroFeedV3NvFp4GateUpLogicalTasks);
static_assert(kSm87MacroFeedV3NvFp4GateUpGridM *
                      kSm87MacroFeedV3NvFp4GateUpBlockM -
                  kSm87MacroFeedV3NvFp4GateUpTokens ==
              kSm87MacroFeedV3NvFp4GateUpTailRows);
static_assert(kSm87MacroFeedV3NvFp4GateUpGridN *
                  kSm87MacroFeedV3NvFp4GateUpKTiles *
                  kSm87MacroFeedV3NvFp4GateUpCellBytes ==
              kSm87MacroFeedV3NvFp4GateUpPartitionBytes);
static_assert(kSm87MacroFeedV3NvFp4GateUpBranches *
                  kSm87MacroFeedV3NvFp4GateUpPartitionBytes ==
              kSm87MacroFeedV3NvFp4GateUpPayloadBytes);
static_assert(sm87_macrofeed_v3_nvfp4_gate_up_plan(40'000U).valid());
static_assert(!sm87_macrofeed_v3_nvfp4_gate_up_plan(39'999U).valid());

}  // namespace q3x::kernels
