#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off T0/T1 architecture slice for the K-heavy Down projection.
// This interface is deliberately independent from the v1/v2 launch receipts
// and from every runtime selector.  Its only accepted production geometry is
// the exact P40000 Down cell and it has no fallback route.
enum class Sm87MacroFeedV3NvFp4DownIdentity : std::uint64_t {
  kInvalid = 0U,
  kM128N256K64WarpM128N32Persistent16V1 =
      0x5133'4d46'5633'4401ULL,
};

inline constexpr Sm87MacroFeedV3NvFp4DownIdentity
    kSm87MacroFeedV3NvFp4DownIdentity =
        Sm87MacroFeedV3NvFp4DownIdentity::
            kM128N256K64WarpM128N32Persistent16V1;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownTokens = 40'000U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownInputFeatures =
    17'408U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownOutputFeatures =
    5'120U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownBlockM = 128U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownBlockN = 256U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownBlockK = 64U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownWarpM = 128U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownWarpN = 32U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownThreads = 256U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownWarps = 8U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownPipelineStages = 3U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownPersistentCtas = 16U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownSmCount = 16U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownWeightBytesPerCell =
    8'192U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownScaleBytesPerCell =
    1'024U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownCellBytes = 9'216U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownDynamicSharedBytes =
    76'800U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownGridM = 313U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownGridN = 20U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownKTiles = 272U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownLogicalTasks = 6'260U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownTailRows = 64U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownPayloadBytes =
    50'135'040U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownPayloadAlignment =
    256U;

// The reduced T1 oracle keeps the real packed cell format and crosses a
// three-stage ring turnover, but reduces N to one tile and K to four cells.
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownTestInputFeatures =
    256U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownTestKTiles = 4U;
inline constexpr std::size_t kSm87MacroFeedV3NvFp4DownTestPayloadBytes =
    kSm87MacroFeedV3NvFp4DownTestKTiles *
    kSm87MacroFeedV3NvFp4DownCellBytes;

struct Sm87MacroFeedV3NvFp4DownPlan final {
  Sm87MacroFeedV3NvFp4DownIdentity identity =
      Sm87MacroFeedV3NvFp4DownIdentity::kInvalid;
  std::size_t token_count = 0U;
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t k_tiles = 0U;
  std::size_t logical_tasks = 0U;
  std::size_t tail_rows = 0U;
  std::size_t payload_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  bool noncooperative_persistent_queue = false;
  bool n_stationary = false;
  bool tail_predicated = false;
  bool canonical_payload_required = false;
  bool fallback_permitted = true;
  bool t0_t1_only = false;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return identity == kSm87MacroFeedV3NvFp4DownIdentity &&
           token_count == kSm87MacroFeedV3NvFp4DownTokens &&
           grid_m == kSm87MacroFeedV3NvFp4DownGridM &&
           grid_n == kSm87MacroFeedV3NvFp4DownGridN &&
           k_tiles == kSm87MacroFeedV3NvFp4DownKTiles &&
           logical_tasks == kSm87MacroFeedV3NvFp4DownLogicalTasks &&
           tail_rows == kSm87MacroFeedV3NvFp4DownTailRows &&
           payload_bytes == kSm87MacroFeedV3NvFp4DownPayloadBytes &&
           dynamic_shared_bytes ==
               kSm87MacroFeedV3NvFp4DownDynamicSharedBytes &&
           noncooperative_persistent_queue && n_stationary &&
           tail_predicated && canonical_payload_required &&
           !fallback_permitted && t0_t1_only &&
           !production_dispatch_eligible;
  }
};

[[nodiscard]] constexpr Sm87MacroFeedV3NvFp4DownPlan
sm87_macrofeed_v3_nvfp4_down_plan(const std::size_t token_count) noexcept {
  if (token_count != kSm87MacroFeedV3NvFp4DownTokens) {
    return {};
  }
  return {kSm87MacroFeedV3NvFp4DownIdentity,
          token_count,
          kSm87MacroFeedV3NvFp4DownGridM,
          kSm87MacroFeedV3NvFp4DownGridN,
          kSm87MacroFeedV3NvFp4DownKTiles,
          kSm87MacroFeedV3NvFp4DownLogicalTasks,
          kSm87MacroFeedV3NvFp4DownTailRows,
          kSm87MacroFeedV3NvFp4DownPayloadBytes,
          kSm87MacroFeedV3NvFp4DownDynamicSharedBytes,
          true,
          true,
          true,
          true,
          false,
          true,
          false};
}

struct Sm87MacroFeedV3NvFp4DownPayloadReceipt final {
  std::uint64_t receipt_identity = 0U;
  Sm87MacroFeedV3NvFp4DownIdentity plan_identity =
      Sm87MacroFeedV3NvFp4DownIdentity::kInvalid;
  std::uint64_t payload_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::uintptr_t payload_begin = 0U;
  std::uintptr_t payload_end = 0U;
  std::uint64_t payload_bytes = 0U;
  bool canonical_consumer_n64_k16_lane_component_v1 = false;
  bool host_bytes_authenticated_before_copy = false;
  bool device_readback_authenticated = false;
  bool allocation_retained_for_launch = false;
};

[[nodiscard]] constexpr std::uint64_t sm87_macrofeed_v3_hash_u64(
    std::uint64_t hash, const std::uint64_t value) noexcept {
  for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
    hash ^= static_cast<std::uint8_t>(value >> (byte * 8U));
    hash *= 1'099'511'628'211ULL;
  }
  return hash;
}

[[nodiscard]] constexpr std::uint64_t
sm87_macrofeed_v3_nvfp4_down_compute_payload_receipt_identity(
    const Sm87MacroFeedV3NvFp4DownPayloadReceipt& receipt) noexcept {
  std::uint64_t hash = 14'695'981'039'346'656'037ULL;
  hash = sm87_macrofeed_v3_hash_u64(
      hash, static_cast<std::uint64_t>(receipt.plan_identity));
  hash = sm87_macrofeed_v3_hash_u64(hash, receipt.payload_identity);
  hash = sm87_macrofeed_v3_hash_u64(
      hash, static_cast<std::uint32_t>(receipt.device_ordinal));
  hash = sm87_macrofeed_v3_hash_u64(hash, receipt.payload_begin);
  hash = sm87_macrofeed_v3_hash_u64(hash, receipt.payload_end);
  hash = sm87_macrofeed_v3_hash_u64(hash, receipt.payload_bytes);
  hash = sm87_macrofeed_v3_hash_u64(
      hash, receipt.canonical_consumer_n64_k16_lane_component_v1);
  hash = sm87_macrofeed_v3_hash_u64(
      hash, receipt.host_bytes_authenticated_before_copy);
  hash = sm87_macrofeed_v3_hash_u64(
      hash, receipt.device_readback_authenticated);
  return sm87_macrofeed_v3_hash_u64(
      hash, receipt.allocation_retained_for_launch);
}

[[nodiscard]] constexpr bool
sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
    const Sm87MacroFeedV3NvFp4DownPayloadReceipt& receipt) noexcept {
  return receipt.receipt_identity != 0U &&
         receipt.receipt_identity ==
             sm87_macrofeed_v3_nvfp4_down_compute_payload_receipt_identity(
                 receipt) &&
         receipt.plan_identity == kSm87MacroFeedV3NvFp4DownIdentity &&
         receipt.payload_identity != 0U && receipt.device_ordinal >= 0 &&
         receipt.payload_begin != 0U &&
         receipt.payload_bytes == kSm87MacroFeedV3NvFp4DownPayloadBytes &&
         receipt.payload_begin <=
             std::numeric_limits<std::uintptr_t>::max() -
                 receipt.payload_bytes &&
         receipt.payload_end == receipt.payload_begin + receipt.payload_bytes &&
         receipt.payload_begin %
                 kSm87MacroFeedV3NvFp4DownPayloadAlignment ==
             0U &&
         receipt.canonical_consumer_n64_k16_lane_component_v1 &&
         receipt.host_bytes_authenticated_before_copy &&
         receipt.device_readback_authenticated &&
         receipt.allocation_retained_for_launch;
}

struct Sm87MacroFeedV3NvFp4DownCudaResources final {
  Sm87MacroFeedV3NvFp4DownIdentity identity =
      Sm87MacroFeedV3NvFp4DownIdentity::kInvalid;
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
sm87_macrofeed_v3_nvfp4_down_resource_gate(
    const Sm87MacroFeedV3NvFp4DownCudaResources& resources) noexcept {
  return resources.identity == kSm87MacroFeedV3NvFp4DownIdentity &&
         resources.device_ordinal >= 0 && resources.compute_major == 8 &&
         resources.compute_minor == 7 &&
         resources.sm_count ==
             static_cast<std::int32_t>(kSm87MacroFeedV3NvFp4DownSmCount) &&
         resources.binary_version == 87 && resources.kernel_compiled &&
         resources.registers_per_thread > 0 &&
         resources.registers_per_thread <= 255 &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kSm87MacroFeedV3NvFp4DownDynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<std::int32_t>(kSm87MacroFeedV3NvFp4DownThreads) &&
         resources.active_blocks_per_sm >= 1 &&
         resources.optin_shared_bytes_per_block >=
             kSm87MacroFeedV3NvFp4DownDynamicSharedBytes &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

struct Sm87MacroFeedV3NvFp4DownArguments final {
  const std::uint16_t* input = nullptr;
  const std::uint8_t* payload = nullptr;
  std::size_t payload_bytes = 0U;
  float tensor_scale = 0.0F;
  std::size_t token_count = 0U;
  std::uint16_t* residual = nullptr;
  void* cuda_stream = nullptr;
  Sm87MacroFeedV3NvFp4DownPayloadReceipt payload_receipt{};
};

struct Sm87MacroFeedV3NvFp4DownLaunchReceipt final {
  Sm87MacroFeedV3NvFp4DownIdentity identity =
      Sm87MacroFeedV3NvFp4DownIdentity::kInvalid;
  std::uint64_t payload_identity = 0U;
  std::size_t token_count = 0U;
  std::size_t logical_tasks = 0U;
  std::size_t tail_rows = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  std::uint32_t fallback_launches = 0U;
  bool launch_enqueued = false;
  bool completion_observed = false;
  bool t0_t1_only = false;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid_enqueue_receipt() const noexcept {
    return identity == kSm87MacroFeedV3NvFp4DownIdentity &&
           payload_identity != 0U &&
           token_count == kSm87MacroFeedV3NvFp4DownTokens &&
           logical_tasks == kSm87MacroFeedV3NvFp4DownLogicalTasks &&
           tail_rows == kSm87MacroFeedV3NvFp4DownTailRows &&
           physical_kernel_launches == 1U && fallback_launches == 0U &&
           launch_enqueued && !completion_observed && t0_t1_only &&
           !production_dispatch_eligible;
  }
};

[[nodiscard]] bool sm87_macrofeed_v3_nvfp4_down_arguments_valid(
    const Sm87MacroFeedV3NvFp4DownArguments& arguments) noexcept;

[[nodiscard]] int query_sm87_macrofeed_v3_nvfp4_down_cuda_resources(
    Sm87MacroFeedV3NvFp4DownCudaResources* resources) noexcept;

[[nodiscard]] int launch_sm87_macrofeed_v3_nvfp4_down_cuda(
    const Sm87MacroFeedV3NvFp4DownArguments& arguments,
    Sm87MacroFeedV3NvFp4DownLaunchReceipt* receipt) noexcept;

// Request-hot form. The caller supplies the immutable startup resource seal;
// no device, function-attribute, occupancy, or pointer-attribute query occurs
// before enqueue. The v10 typed workspace and authenticated payload owner are
// therefore part of the caller's retained capability contract.
[[nodiscard]] int launch_sm87_macrofeed_v3_nvfp4_down_sealed_cuda(
    const Sm87MacroFeedV3NvFp4DownArguments& arguments,
    const Sm87MacroFeedV3NvFp4DownCudaResources& sealed_resources,
    Sm87MacroFeedV3NvFp4DownLaunchReceipt* receipt) noexcept;

// Numerical admission helper. It launches exactly one M128N256 tile over
// four canonical K64 cells and is not a production/fallback interface.
[[nodiscard]] int launch_sm87_macrofeed_v3_nvfp4_down_tile_test_cuda(
    const std::uint16_t* input_m128_k256,
    const std::uint8_t* canonical_payload_four_cells,
    float tensor_scale, std::size_t valid_rows,
    std::uint16_t* residual_m128_n256, void* cuda_stream) noexcept;

static_assert(kSm87MacroFeedV3NvFp4DownGridM *
                  kSm87MacroFeedV3NvFp4DownBlockM ==
              40'064U);
static_assert(kSm87MacroFeedV3NvFp4DownGridN *
                  kSm87MacroFeedV3NvFp4DownBlockN ==
              kSm87MacroFeedV3NvFp4DownOutputFeatures);
static_assert(kSm87MacroFeedV3NvFp4DownKTiles *
                  kSm87MacroFeedV3NvFp4DownBlockK ==
              kSm87MacroFeedV3NvFp4DownInputFeatures);
static_assert(kSm87MacroFeedV3NvFp4DownGridM *
                  kSm87MacroFeedV3NvFp4DownGridN ==
              kSm87MacroFeedV3NvFp4DownLogicalTasks);
static_assert(kSm87MacroFeedV3NvFp4DownGridM *
                      kSm87MacroFeedV3NvFp4DownBlockM -
                  kSm87MacroFeedV3NvFp4DownTokens ==
              kSm87MacroFeedV3NvFp4DownTailRows);
static_assert(kSm87MacroFeedV3NvFp4DownGridN *
                  kSm87MacroFeedV3NvFp4DownKTiles *
                  kSm87MacroFeedV3NvFp4DownCellBytes ==
              kSm87MacroFeedV3NvFp4DownPayloadBytes);
static_assert(sm87_macrofeed_v3_nvfp4_down_plan(40'000U).valid());
static_assert(!sm87_macrofeed_v3_nvfp4_down_plan(39'999U).valid());

}  // namespace q3x::kernels
