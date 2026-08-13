#pragma once

#include "q3x/kernels/sm87_target_aot_attention_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_BULK_V2_ATTENTION_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_BULK_V2_ATTENTION_HOST_DEVICE
#endif

namespace q3x::kernels {

// First executable full-Attention cell for
// AC-PREFILL-SM87-BULK-DATAFLOW-v2.  It changes only CTA ownership: the
// target-AOT Q128/KV32 arithmetic body remains the sole numerical body.  The
// cell is exact-P40000, default-off, test-only, and has no runner selector or
// production route.
inline constexpr std::size_t kSm87BulkV2AttentionTokens = 40'000U;
inline constexpr std::size_t kSm87BulkV2AttentionKvHeads =
    kSm87TargetAotAttentionKvHeads;
inline constexpr std::size_t kSm87BulkV2AttentionFlattenedQueriesPerKvHead =
    kSm87BulkV2AttentionTokens * kSm87TargetAotAttentionQueriesPerKv;
inline constexpr std::size_t kSm87BulkV2AttentionQueryRows =
    kSm87TargetAotAttentionQueryRows;
inline constexpr std::size_t kSm87BulkV2AttentionQueryTilesPerKvHead =
    kSm87BulkV2AttentionFlattenedQueriesPerKvHead /
    kSm87BulkV2AttentionQueryRows;
inline constexpr std::size_t kSm87BulkV2AttentionPersistentLanes = 16U;
inline constexpr std::size_t kSm87BulkV2AttentionSnakeEpochs =
    (kSm87BulkV2AttentionQueryTilesPerKvHead +
     kSm87BulkV2AttentionPersistentLanes - 1U) /
    kSm87BulkV2AttentionPersistentLanes;
inline constexpr std::size_t kSm87BulkV2AttentionLastQueryTile =
    kSm87BulkV2AttentionQueryTilesPerKvHead - 1U;
inline constexpr std::size_t kSm87BulkV2AttentionFinalRepeatLanes = 13U;
inline constexpr std::size_t kSm87BulkV2AttentionScheduledBodiesPerKvHead =
    kSm87BulkV2AttentionPersistentLanes *
    kSm87BulkV2AttentionSnakeEpochs;
inline constexpr std::size_t kSm87BulkV2AttentionStoreDisabledBodies =
    kSm87BulkV2AttentionKvHeads *
    kSm87BulkV2AttentionFinalRepeatLanes;
inline constexpr std::size_t kSm87BulkV2AttentionRealOutputs =
    kSm87BulkV2AttentionKvHeads *
    kSm87BulkV2AttentionQueryTilesPerKvHead;
inline constexpr std::size_t kSm87BulkV2AttentionKernelLaunches =
    kSm87BulkV2AttentionKvHeads;
inline constexpr std::size_t kSm87BulkV2AttentionThreads =
    kSm87TargetAotAttentionThreads;
inline constexpr std::size_t kSm87BulkV2AttentionDynamicSharedBytes =
    kSm87TargetAotAttentionSharedBytes;
inline constexpr std::size_t kSm87BulkV2AttentionPointerAlignment = 16U;
inline constexpr int kSm87BulkV2AttentionMaximumRegistersPerThread = 255;
inline constexpr int kSm87BulkV2AttentionRequiredSmCount = 16;

enum Sm87BulkV2AttentionPolicy : std::uint32_t {
  kSm87BulkV2AttentionExactTargetAotNumericalBody = 1U << 0U,
  kSm87BulkV2AttentionSameReductionTree = 1U << 1U,
  kSm87BulkV2AttentionSameBf16Publication = 1U << 2U,
  kSm87BulkV2AttentionSameKvHeadPerLaunch = 1U << 3U,
  kSm87BulkV2AttentionSnakeEpochOrder = 1U << 4U,
  kSm87BulkV2AttentionFinalRepeatStoreSuppressed = 1U << 5U,
  kSm87BulkV2AttentionNoCooperativeLaunch = 1U << 6U,
  kSm87BulkV2AttentionNoCrossCtaBarrierOrLock = 1U << 7U,
  kSm87BulkV2AttentionNoProductionSelector = 1U << 8U,
  kSm87BulkV2AttentionResourceGateFailClosed = 1U << 9U,
};

inline constexpr std::uint32_t kSm87BulkV2AttentionRequiredPolicy =
    kSm87BulkV2AttentionExactTargetAotNumericalBody |
    kSm87BulkV2AttentionSameReductionTree |
    kSm87BulkV2AttentionSameBf16Publication |
    kSm87BulkV2AttentionSameKvHeadPerLaunch |
    kSm87BulkV2AttentionSnakeEpochOrder |
    kSm87BulkV2AttentionFinalRepeatStoreSuppressed |
    kSm87BulkV2AttentionNoCooperativeLaunch |
    kSm87BulkV2AttentionNoCrossCtaBarrierOrLock |
    kSm87BulkV2AttentionNoProductionSelector |
    kSm87BulkV2AttentionResourceGateFailClosed;

// This is a copied identity, not a new arithmetic contract.  The CUDA source
// enforces the stronger condition by calling the same device body from both
// the target-AOT control and this persistent cohort wrapper.
inline constexpr Sm87TargetAotAttentionNumericalContract
    kSm87BulkV2AttentionNumericalContract =
        sm87_target_aot_attention_plan(kSm87BulkV2AttentionTokens)
            .numerical_execution;

struct Sm87BulkV2AttentionWorkItem final {
  std::size_t kv_head = kSm87BulkV2AttentionKvHeads;
  std::size_t lane = kSm87BulkV2AttentionPersistentLanes;
  std::size_t epoch = kSm87BulkV2AttentionSnakeEpochs;
  std::size_t query_tile = kSm87BulkV2AttentionQueryTilesPerKvHead;
  bool store_enabled = false;
  bool valid = false;
};

// Even epochs walk low-to-high; odd epochs walk high-to-low.  Epoch 117 has
// only three real tiles (1874, 1873, 1872).  Lanes 0..12 execute tile 1874 as
// a read-only cohort repeat so every lane retains the same epoch count, while
// the lane-13 owner is the only writer for tile 1874.
[[nodiscard]] Q3X_SM87_BULK_V2_ATTENTION_HOST_DEVICE constexpr
Sm87BulkV2AttentionWorkItem sm87_bulk_v2_attention_work_item(
    const std::size_t kv_head, const std::size_t lane,
    const std::size_t epoch) noexcept {
  if (kv_head >= kSm87BulkV2AttentionKvHeads ||
      lane >= kSm87BulkV2AttentionPersistentLanes ||
      epoch >= kSm87BulkV2AttentionSnakeEpochs) {
    return {};
  }
  const std::size_t epoch_begin =
      epoch * kSm87BulkV2AttentionPersistentLanes;
  const std::size_t scheduled_tile =
      (epoch & 1U) == 0U
          ? epoch_begin + lane
          : epoch_begin + kSm87BulkV2AttentionPersistentLanes - 1U - lane;
  const bool store_enabled =
      scheduled_tile < kSm87BulkV2AttentionQueryTilesPerKvHead;
  return {kv_head, lane, epoch,
          store_enabled ? scheduled_tile
                        : kSm87BulkV2AttentionLastQueryTile,
          store_enabled, true};
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_attention_mapping_is_bijective() noexcept {
  std::array<std::uint8_t, kSm87BulkV2AttentionRealOutputs> counts{};
  std::size_t stores = 0U;
  std::size_t suppressed = 0U;
  for (std::size_t kv_head = 0U;
       kv_head < kSm87BulkV2AttentionKvHeads; ++kv_head) {
    for (std::size_t epoch = 0U;
         epoch < kSm87BulkV2AttentionSnakeEpochs; ++epoch) {
      for (std::size_t lane = 0U;
           lane < kSm87BulkV2AttentionPersistentLanes; ++lane) {
        const auto item =
            sm87_bulk_v2_attention_work_item(kv_head, lane, epoch);
        if (!item.valid || item.kv_head != kv_head || item.lane != lane ||
            item.epoch != epoch ||
            item.query_tile >= kSm87BulkV2AttentionQueryTilesPerKvHead) {
          return false;
        }
        if (item.store_enabled) {
          const std::size_t output =
              kv_head * kSm87BulkV2AttentionQueryTilesPerKvHead +
              item.query_tile;
          if (++counts[output] != 1U) {
            return false;
          }
          ++stores;
        } else {
          if (epoch + 1U != kSm87BulkV2AttentionSnakeEpochs ||
              lane >= kSm87BulkV2AttentionFinalRepeatLanes ||
              item.query_tile != kSm87BulkV2AttentionLastQueryTile) {
            return false;
          }
          ++suppressed;
        }
      }
    }
  }
  if (stores != kSm87BulkV2AttentionRealOutputs ||
      suppressed != kSm87BulkV2AttentionStoreDisabledBodies) {
    return false;
  }
  for (const auto count : counts) {
    if (count != 1U) {
      return false;
    }
  }
  return true;
}

inline constexpr std::uint64_t kSm87BulkV2AttentionQueryBytes =
    kSm87BulkV2AttentionTokens *
    kSm87TargetAotAttentionQueryTokenStride * sizeof(std::uint16_t);
inline constexpr std::uint64_t kSm87BulkV2AttentionKvBytes =
    kSm87BulkV2AttentionTokens *
    kSm87TargetAotAttentionKvTokenStride * sizeof(std::uint16_t);

struct Sm87BulkV2AttentionArguments final {
  const std::uint16_t* processed_query = nullptr;  // [40000,24,256]
  const std::uint16_t* processed_key = nullptr;    // [40000,4,256]
  const std::uint16_t* processed_value = nullptr;  // [40000,4,256]
  const std::uint16_t* processed_gate = nullptr;   // [40000,24,256]
  std::uint16_t* gated_output = nullptr;           // [40000,24,256]
  std::size_t token_count = 0U;
  std::int32_t device_ordinal = -1;
  void* cuda_stream = nullptr;
};

struct Sm87BulkV2AttentionByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2AttentionByteRange
sm87_bulk_v2_attention_byte_range(const void* const pointer,
                                  const std::uint64_t bytes) noexcept {
  if (pointer == nullptr || bytes == 0U ||
      bytes > std::numeric_limits<std::uintptr_t>::max()) {
    return {};
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() -
                  static_cast<std::uintptr_t>(bytes)) {
    return {};
  }
  return {begin, begin + static_cast<std::uintptr_t>(bytes), true};
}

[[nodiscard]] constexpr bool sm87_bulk_v2_attention_ranges_overlap(
    const Sm87BulkV2AttentionByteRange& left,
    const Sm87BulkV2AttentionByteRange& right) noexcept {
  return !left.valid || !right.valid ||
         (left.begin < right.end && right.begin < left.end);
}

[[nodiscard]] constexpr auto sm87_bulk_v2_attention_argument_ranges(
    const Sm87BulkV2AttentionArguments& arguments) noexcept {
  return std::array<Sm87BulkV2AttentionByteRange, 5U>{{
      sm87_bulk_v2_attention_byte_range(arguments.processed_query,
                                        kSm87BulkV2AttentionQueryBytes),
      sm87_bulk_v2_attention_byte_range(arguments.processed_key,
                                        kSm87BulkV2AttentionKvBytes),
      sm87_bulk_v2_attention_byte_range(arguments.processed_value,
                                        kSm87BulkV2AttentionKvBytes),
      sm87_bulk_v2_attention_byte_range(arguments.processed_gate,
                                        kSm87BulkV2AttentionQueryBytes),
      sm87_bulk_v2_attention_byte_range(arguments.gated_output,
                                        kSm87BulkV2AttentionQueryBytes),
  }};
}

[[nodiscard]] constexpr bool sm87_bulk_v2_attention_arguments_valid(
    const Sm87BulkV2AttentionArguments& arguments) noexcept {
  if (arguments.token_count != kSm87BulkV2AttentionTokens ||
      arguments.device_ordinal < 0 || arguments.cuda_stream == nullptr) {
    return false;
  }
  const auto ranges = sm87_bulk_v2_attention_argument_ranges(arguments);
  for (std::size_t first = 0U; first < ranges.size(); ++first) {
    if (!ranges[first].valid ||
        ranges[first].begin % kSm87BulkV2AttentionPointerAlignment != 0U) {
      return false;
    }
    for (std::size_t second = first + 1U; second < ranges.size(); ++second) {
      if (sm87_bulk_v2_attention_ranges_overlap(ranges[first],
                                                ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

struct Sm87BulkV2AttentionResources final {
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int device_sm_count = 0;
  std::size_t device_optin_shared_bytes = 0U;
  int threads_per_block = 0;
  int physical_grid_ctas_per_launch = 0;
  int physical_launches = 0;
  std::size_t query_tiles_per_kv_head = 0U;
  std::size_t snake_epochs = 0U;
  std::size_t store_disabled_bodies = 0U;
  bool kernel_compiled = false;
  bool exact_p40000_only = false;
  bool same_kv_head_per_launch = false;
  bool mapping_bijective = false;
  bool no_cooperative_launch = false;
  bool no_cross_cta_barrier_or_lock = false;
  // Capacity is necessary for the 16-lane cohort, but a non-cooperative
  // launch does not prove simultaneous scheduling or phase lockstep.
  bool persistent_cta_residency_capacity = false;
  bool resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool sm87_bulk_v2_attention_resources_valid(
    const Sm87BulkV2AttentionResources& resources) noexcept {
  return resources.binary_version == 87 &&
         resources.registers_per_thread > 0 &&
         resources.registers_per_thread <=
             kSm87BulkV2AttentionMaximumRegistersPerThread &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kSm87BulkV2AttentionDynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<int>(kSm87BulkV2AttentionThreads) &&
         resources.active_blocks_per_sm >= 1 &&
         resources.device_sm_count == kSm87BulkV2AttentionRequiredSmCount &&
         resources.device_optin_shared_bytes >=
             kSm87BulkV2AttentionDynamicSharedBytes &&
         resources.threads_per_block ==
             static_cast<int>(kSm87BulkV2AttentionThreads) &&
         resources.physical_grid_ctas_per_launch ==
             static_cast<int>(kSm87BulkV2AttentionPersistentLanes) &&
         resources.physical_launches ==
             static_cast<int>(kSm87BulkV2AttentionKernelLaunches) &&
         resources.query_tiles_per_kv_head ==
             kSm87BulkV2AttentionQueryTilesPerKvHead &&
         resources.snake_epochs == kSm87BulkV2AttentionSnakeEpochs &&
         resources.store_disabled_bodies ==
             kSm87BulkV2AttentionStoreDisabledBodies &&
         resources.kernel_compiled && resources.exact_p40000_only &&
         resources.same_kv_head_per_launch && resources.mapping_bijective &&
         resources.no_cooperative_launch &&
         resources.no_cross_cta_barrier_or_lock &&
         resources.persistent_cta_residency_capacity &&
         resources.active_blocks_per_sm * resources.device_sm_count >=
             static_cast<int>(kSm87BulkV2AttentionPersistentLanes) &&
         resources.resource_gate_passed &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

// The explicit test-only CMake admission compiles the persistent wrapper.
// query() inspects that exact kernel and launch() re-runs the hard resource
// gate before issuing four stream-ordered, same-KV-head 16-CTA launches.
[[nodiscard]] int
query_sm87_bulk_dataflow_v2_attention_l2_cohort_resources_cuda(
    std::int32_t device_ordinal,
    Sm87BulkV2AttentionResources* resources) noexcept;

[[nodiscard]] int launch_sm87_bulk_dataflow_v2_attention_l2_cohort_cuda(
    const Sm87BulkV2AttentionArguments& arguments) noexcept;

static_assert(kSm87BulkV2AttentionFlattenedQueriesPerKvHead == 240'000U);
static_assert(kSm87BulkV2AttentionQueryTilesPerKvHead == 1'875U);
static_assert(kSm87BulkV2AttentionSnakeEpochs == 118U);
static_assert(kSm87BulkV2AttentionScheduledBodiesPerKvHead == 1'888U);
static_assert(kSm87BulkV2AttentionLastQueryTile == 1'874U);
static_assert(kSm87BulkV2AttentionStoreDisabledBodies == 52U);
static_assert(kSm87BulkV2AttentionRealOutputs == 7'500U);
static_assert(kSm87BulkV2AttentionDynamicSharedBytes == 128U * 1024U);
static_assert(kSm87BulkV2AttentionRequiredPolicy == 0x3ffU);
static_assert(sm87_bulk_v2_attention_mapping_is_bijective());

}  // namespace q3x::kernels

#undef Q3X_SM87_BULK_V2_ATTENTION_HOST_DEVICE
