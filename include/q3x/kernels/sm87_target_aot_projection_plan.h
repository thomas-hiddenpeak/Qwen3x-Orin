#pragma once

#include "q3x/kernels/sm87_target_aot_context.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Host-only physical contract for the projection constituent of
// AC-PREFILL-SM87-AOT-SYSTEM-v1. This header deliberately exposes no launcher,
// device pointer, or production selector. It freezes the independently
// derived SM87 dataflow that a later default-off CUDA implementation must
// satisfy without inheriting any rejected P40 packed-kernel skeleton.
inline constexpr std::size_t kSm87TargetAotProjectionHidden = 5'120U;
inline constexpr std::size_t kSm87TargetAotProjectionIntermediate = 17'408U;
inline constexpr std::size_t kSm87TargetAotProjectionMergedGateUp = 34'816U;
inline constexpr std::size_t kSm87TargetAotProjectionBlockM = 128U;
inline constexpr std::size_t kSm87TargetAotProjectionBlockN = 256U;
inline constexpr std::size_t kSm87TargetAotProjectionBlockK = 64U;
inline constexpr std::size_t kSm87TargetAotProjectionWarpM = 64U;
inline constexpr std::size_t kSm87TargetAotProjectionWarpN = 64U;
inline constexpr std::size_t kSm87TargetAotProjectionWarpK = 64U;
inline constexpr std::size_t kSm87TargetAotProjectionWarps = 8U;
inline constexpr std::size_t kSm87TargetAotProjectionThreads = 256U;
inline constexpr std::size_t kSm87TargetAotProjectionPipelineStages = 3U;
inline constexpr std::size_t kSm87TargetAotProjectionPersistentCtas = 16U;
inline constexpr std::size_t kSm87TargetAotProjectionSmCount = 16U;
inline constexpr std::size_t kSm87TargetAotProjectionMaximumPartitions = 3U;

inline constexpr std::size_t kSm87TargetAotNvFp4ABytesPerStage =
    kSm87TargetAotProjectionBlockM * kSm87TargetAotProjectionBlockK *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87TargetAotNvFp4BBytesPerStage =
    kSm87TargetAotProjectionBlockN * kSm87TargetAotProjectionBlockK / 2U;
inline constexpr std::size_t kSm87TargetAotNvFp4ScaleBytesPerStage =
    kSm87TargetAotProjectionBlockN *
    (kSm87TargetAotProjectionBlockK / 16U);
inline constexpr std::size_t kSm87TargetAotNvFp4SharedBytes =
    kSm87TargetAotProjectionPipelineStages *
    (kSm87TargetAotNvFp4ABytesPerStage +
     kSm87TargetAotNvFp4BBytesPerStage +
     kSm87TargetAotNvFp4ScaleBytesPerStage);
inline constexpr std::size_t kSm87TargetAotFp8ABytesPerStage =
    kSm87TargetAotProjectionBlockM * kSm87TargetAotProjectionBlockK *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87TargetAotFp8BBytesPerStage =
    kSm87TargetAotProjectionBlockN * kSm87TargetAotProjectionBlockK;
inline constexpr std::size_t kSm87TargetAotFp8SharedBytes =
    kSm87TargetAotProjectionPipelineStages *
    (kSm87TargetAotFp8ABytesPerStage +
     kSm87TargetAotFp8BBytesPerStage);

enum class Sm87TargetAotProjectionRole : std::uint8_t {
  kInvalid = 0U,
  kNvFp4GateUp,
  kNvFp4Down,
  kFp8GdnQkvZ,
  kFp8FullQkv,
  kFp8AttentionOutput,
  kCount,
};

enum class Sm87TargetAotProjectionTactic : std::uint8_t {
  kInvalid = 0U,
  kNvFp4GateUpM128N256K64MGroup2,
  kNvFp4DownM128N256K64MGroup1,
  kFp8GdnQkvZM128N256K64MGroup2,
  kFp8FullQkvM128N256K64MGroup2,
  kFp8AttentionOutputM128N256K64MGroup1,
};

enum class Sm87TargetAotProjectionEncoding : std::uint8_t {
  kInvalid = 0U,
  kNvFp4E2M1Block16E4M3FnScale,
  kFp8E4M3FnTensorScale,
};

enum class Sm87TargetAotLogicalRole : std::uint8_t {
  kInvalid = 0U,
  kNvFp4Gate,
  kNvFp4Up,
  kNvFp4Down,
  kFp8GdnQkv,
  kFp8GdnZ,
  kFp8FullQGate,
  kFp8FullK,
  kFp8FullV,
  kFp8AttentionOutput,
};

enum class Sm87TargetAotConsumer : std::uint8_t {
  kInvalid = 0U,
  kSiluTimesUp,
  kResidualAdd,
  kGdnCore,
  kFullAttentionCore,
};

enum Sm87TargetAotProjectionPolicy : std::uint32_t {
  kSm87TargetAotAheadOfTimeTactic = 1U << 0U,
  kSm87TargetAotNoRequestJit = 1U << 1U,
  kSm87TargetAotNoFallback = 1U << 2U,
  kSm87TargetAotNoMtp = 1U << 3U,
  kSm87TargetAotNoCuBlasLt = 1U << 4U,
  kSm87TargetAotPackedWeightThroughShared = 1U << 5U,
  kSm87TargetAotRegisterDecodeBeforeMma = 1U << 6U,
  kSm87TargetAotFullKSingleCta = 1U << 7U,
  kSm87TargetAotBf16RoundingBoundary = 1U << 8U,
  kSm87TargetAotAccuracyUnqualified = 1U << 9U,
  kSm87TargetAotCpAsyncCg = 1U << 10U,
  kSm87TargetAotSharedToRegisterDoubleBuffer = 1U << 11U,
  kSm87TargetAotPrefetchBeforeStageDrain = 1U << 12U,
  kSm87TargetAotNoReductionWorkspace = 1U << 13U,
  kSm87TargetAotGateUpSameCtaPartitionPair = 1U << 14U,
  kSm87TargetAotGateBf16ReadyBeforeUpConsumer = 1U << 15U,
};

inline constexpr std::uint32_t kSm87TargetAotProjectionRequiredPolicy =
    kSm87TargetAotAheadOfTimeTactic | kSm87TargetAotNoRequestJit |
    kSm87TargetAotNoFallback | kSm87TargetAotNoMtp |
    kSm87TargetAotNoCuBlasLt | kSm87TargetAotPackedWeightThroughShared |
    kSm87TargetAotRegisterDecodeBeforeMma |
    kSm87TargetAotFullKSingleCta |
    kSm87TargetAotBf16RoundingBoundary |
    kSm87TargetAotAccuracyUnqualified | kSm87TargetAotCpAsyncCg |
    kSm87TargetAotSharedToRegisterDoubleBuffer |
    kSm87TargetAotPrefetchBeforeStageDrain |
    kSm87TargetAotNoReductionWorkspace;

struct Sm87TargetAotProjectionPartition {
  Sm87TargetAotLogicalRole role = Sm87TargetAotLogicalRole::kInvalid;
  std::size_t output_offset = 0U;
  std::size_t output_features = 0U;
  bool independent_tensor_scale = false;
  bool bf16_rounding_boundary = false;
};

struct Sm87TargetAotProjectionPlan {
  Sm87TargetAotProjectionRole role = Sm87TargetAotProjectionRole::kInvalid;
  Sm87TargetAotProjectionTactic tactic =
      Sm87TargetAotProjectionTactic::kInvalid;
  Sm87TargetAotProjectionEncoding encoding =
      Sm87TargetAotProjectionEncoding::kInvalid;
  Sm87TargetAotConsumer consumer = Sm87TargetAotConsumer::kInvalid;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t projected_output_features = 0U;
  std::size_t published_output_features = 0U;
  std::array<Sm87TargetAotProjectionPartition,
             kSm87TargetAotProjectionMaximumPartitions>
      partitions{};
  std::size_t partition_count = 0U;
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t tail_rows = 0U;
  std::size_t k_tiles = 0U;
  std::size_t logical_tasks = 0U;
  std::size_t mma_tile_tasks = 0U;
  std::size_t mma_partitions_per_task = 0U;
  std::size_t kernel_launches = 0U;
  std::size_t physical_ctas = 0U;
  std::size_t ctas_per_sm = 0U;
  std::size_t raster_group_m = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t split_k_workspace_bytes = 0U;
  std::size_t stream_k_lock_bytes = 0U;
  std::uint32_t policy = 0U;
  bool cp_async_a = false;
  bool cp_async_b = false;
  bool cp_async_scale = false;
  bool cp_async_cache_global = false;
  bool register_double_buffer = false;
  bool prefetch_before_stage_drain = false;
  bool same_cta_partition_pair = false;
  bool gate_bf16_ready_before_up_consumer = false;
  bool stream_k = false;
  bool cuda_implementation_present = false;
  bool static_resources_qualified = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;

  [[nodiscard]] constexpr bool valid() const noexcept;
};

struct Sm87TargetAotProjectionTask {
  std::size_t m_tile = 0U;
  std::size_t n_tile = 0U;
  std::size_t first_partition = 0U;
  std::size_t partition_count = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotProjectionPartition
sm87_target_aot_partition(const Sm87TargetAotLogicalRole role,
                          const std::size_t output_offset,
                          const std::size_t output_features) noexcept {
  return {role, output_offset, output_features, true, true};
}

[[nodiscard]] constexpr Sm87TargetAotProjectionPlan
sm87_target_aot_projection_plan(
    const Sm87TargetAotProjectionRole role,
    const std::size_t token_count) noexcept {
  if (!sm87_target_aot_exact_witness_tokens(token_count) ||
      role == Sm87TargetAotProjectionRole::kInvalid ||
      role == Sm87TargetAotProjectionRole::kCount) {
    return {};
  }

  Sm87TargetAotProjectionPlan plan;
  plan.role = role;
  plan.token_count = token_count;
  plan.grid_m =
      (token_count + kSm87TargetAotProjectionBlockM - 1U) /
      kSm87TargetAotProjectionBlockM;
  plan.tail_rows = token_count % kSm87TargetAotProjectionBlockM;
  if (plan.tail_rows == 0U) {
    plan.tail_rows = kSm87TargetAotProjectionBlockM;
  }
  plan.policy = kSm87TargetAotProjectionRequiredPolicy;
  plan.physical_ctas = kSm87TargetAotProjectionPersistentCtas;
  plan.ctas_per_sm = 1U;
  plan.cp_async_a = true;
  plan.cp_async_b = true;
  plan.cp_async_cache_global = true;
  plan.register_double_buffer = true;
  plan.prefetch_before_stage_drain = true;
  plan.mma_partitions_per_task = 1U;
  plan.kernel_launches = 1U;
  plan.stream_k = false;
  plan.production_dispatch_eligible = false;

  if (role == Sm87TargetAotProjectionRole::kNvFp4GateUp) {
    plan.tactic = Sm87TargetAotProjectionTactic::
        kNvFp4GateUpM128N256K64MGroup2;
    plan.encoding =
        Sm87TargetAotProjectionEncoding::kNvFp4E2M1Block16E4M3FnScale;
    plan.consumer = Sm87TargetAotConsumer::kSiluTimesUp;
    plan.input_features = kSm87TargetAotProjectionHidden;
    plan.projected_output_features = kSm87TargetAotProjectionMergedGateUp;
    plan.published_output_features = kSm87TargetAotProjectionIntermediate;
    plan.partitions[0U] = sm87_target_aot_partition(
        Sm87TargetAotLogicalRole::kNvFp4Gate, 0U,
        kSm87TargetAotProjectionIntermediate);
    plan.partitions[1U] = sm87_target_aot_partition(
        Sm87TargetAotLogicalRole::kNvFp4Up,
        kSm87TargetAotProjectionIntermediate,
        kSm87TargetAotProjectionIntermediate);
    plan.partition_count = 2U;
    plan.raster_group_m = 2U;
    plan.dynamic_shared_bytes = kSm87TargetAotNvFp4SharedBytes;
    plan.cp_async_scale = true;
    plan.mma_partitions_per_task = 2U;
    plan.same_cta_partition_pair = true;
    plan.gate_bf16_ready_before_up_consumer = true;
    plan.policy |= kSm87TargetAotGateUpSameCtaPartitionPair |
                   kSm87TargetAotGateBf16ReadyBeforeUpConsumer;
  } else if (role == Sm87TargetAotProjectionRole::kNvFp4Down) {
    plan.tactic =
        Sm87TargetAotProjectionTactic::kNvFp4DownM128N256K64MGroup1;
    plan.encoding =
        Sm87TargetAotProjectionEncoding::kNvFp4E2M1Block16E4M3FnScale;
    plan.consumer = Sm87TargetAotConsumer::kResidualAdd;
    plan.input_features = kSm87TargetAotProjectionIntermediate;
    plan.projected_output_features = kSm87TargetAotProjectionHidden;
    plan.published_output_features = kSm87TargetAotProjectionHidden;
    plan.partitions[0U] = sm87_target_aot_partition(
        Sm87TargetAotLogicalRole::kNvFp4Down, 0U,
        kSm87TargetAotProjectionHidden);
    plan.partition_count = 1U;
    plan.raster_group_m = 1U;
    plan.dynamic_shared_bytes = kSm87TargetAotNvFp4SharedBytes;
    plan.cp_async_scale = true;
  } else if (role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    plan.tactic =
        Sm87TargetAotProjectionTactic::kFp8GdnQkvZM128N256K64MGroup2;
    plan.encoding = Sm87TargetAotProjectionEncoding::kFp8E4M3FnTensorScale;
    plan.consumer = Sm87TargetAotConsumer::kGdnCore;
    plan.input_features = kSm87TargetAotProjectionHidden;
    plan.projected_output_features = 16'384U;
    plan.published_output_features = 16'384U;
    plan.partitions[0U] = sm87_target_aot_partition(
        Sm87TargetAotLogicalRole::kFp8GdnQkv, 0U, 10'240U);
    plan.partitions[1U] = sm87_target_aot_partition(
        Sm87TargetAotLogicalRole::kFp8GdnZ, 10'240U, 6'144U);
    plan.partition_count = 2U;
    plan.raster_group_m = 2U;
    plan.dynamic_shared_bytes = kSm87TargetAotFp8SharedBytes;
  } else if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    plan.tactic =
        Sm87TargetAotProjectionTactic::kFp8FullQkvM128N256K64MGroup2;
    plan.encoding = Sm87TargetAotProjectionEncoding::kFp8E4M3FnTensorScale;
    plan.consumer = Sm87TargetAotConsumer::kFullAttentionCore;
    plan.input_features = kSm87TargetAotProjectionHidden;
    plan.projected_output_features = 14'336U;
    plan.published_output_features = 14'336U;
    plan.partitions[0U] = sm87_target_aot_partition(
        Sm87TargetAotLogicalRole::kFp8FullQGate, 0U, 12'288U);
    plan.partitions[1U] = sm87_target_aot_partition(
        Sm87TargetAotLogicalRole::kFp8FullK, 12'288U, 1'024U);
    plan.partitions[2U] = sm87_target_aot_partition(
        Sm87TargetAotLogicalRole::kFp8FullV, 13'312U, 1'024U);
    plan.partition_count = 3U;
    plan.raster_group_m = 2U;
    plan.dynamic_shared_bytes = kSm87TargetAotFp8SharedBytes;
  } else if (role ==
             Sm87TargetAotProjectionRole::kFp8AttentionOutput) {
    plan.tactic = Sm87TargetAotProjectionTactic::
        kFp8AttentionOutputM128N256K64MGroup1;
    plan.encoding = Sm87TargetAotProjectionEncoding::kFp8E4M3FnTensorScale;
    plan.consumer = Sm87TargetAotConsumer::kResidualAdd;
    plan.input_features = 6'144U;
    plan.projected_output_features = kSm87TargetAotProjectionHidden;
    plan.published_output_features = kSm87TargetAotProjectionHidden;
    plan.partitions[0U] = sm87_target_aot_partition(
        Sm87TargetAotLogicalRole::kFp8AttentionOutput, 0U,
        kSm87TargetAotProjectionHidden);
    plan.partition_count = 1U;
    plan.raster_group_m = 1U;
    plan.dynamic_shared_bytes = kSm87TargetAotFp8SharedBytes;
  } else {
    return {};
  }

  plan.grid_n =
      (role == Sm87TargetAotProjectionRole::kNvFp4GateUp
           ? plan.published_output_features
           : plan.projected_output_features) /
      kSm87TargetAotProjectionBlockN;
  plan.k_tiles = plan.input_features / kSm87TargetAotProjectionBlockK;
  plan.logical_tasks = plan.grid_m * plan.grid_n;
  plan.mma_tile_tasks =
      plan.logical_tasks * plan.mma_partitions_per_task;
  return plan;
}

[[nodiscard]] constexpr bool sm87_target_aot_same_partition(
    const Sm87TargetAotProjectionPartition& left,
    const Sm87TargetAotProjectionPartition& right) noexcept {
  return left.role == right.role &&
         left.output_offset == right.output_offset &&
         left.output_features == right.output_features &&
         left.independent_tensor_scale == right.independent_tensor_scale &&
         left.bf16_rounding_boundary == right.bf16_rounding_boundary;
}

[[nodiscard]] constexpr bool sm87_target_aot_same_plan(
    const Sm87TargetAotProjectionPlan& left,
    const Sm87TargetAotProjectionPlan& right) noexcept {
  if (left.role != right.role || left.tactic != right.tactic ||
      left.encoding != right.encoding || left.consumer != right.consumer ||
      left.token_count != right.token_count ||
      left.input_features != right.input_features ||
      left.projected_output_features != right.projected_output_features ||
      left.published_output_features != right.published_output_features ||
      left.partition_count != right.partition_count ||
      left.grid_m != right.grid_m || left.grid_n != right.grid_n ||
      left.tail_rows != right.tail_rows || left.k_tiles != right.k_tiles ||
      left.logical_tasks != right.logical_tasks ||
      left.mma_tile_tasks != right.mma_tile_tasks ||
      left.mma_partitions_per_task != right.mma_partitions_per_task ||
      left.kernel_launches != right.kernel_launches ||
      left.physical_ctas != right.physical_ctas ||
      left.ctas_per_sm != right.ctas_per_sm ||
      left.raster_group_m != right.raster_group_m ||
      left.dynamic_shared_bytes != right.dynamic_shared_bytes ||
      left.split_k_workspace_bytes != right.split_k_workspace_bytes ||
      left.stream_k_lock_bytes != right.stream_k_lock_bytes ||
      left.policy != right.policy || left.cp_async_a != right.cp_async_a ||
      left.cp_async_b != right.cp_async_b ||
      left.cp_async_scale != right.cp_async_scale ||
      left.cp_async_cache_global != right.cp_async_cache_global ||
      left.register_double_buffer != right.register_double_buffer ||
      left.prefetch_before_stage_drain !=
          right.prefetch_before_stage_drain ||
      left.same_cta_partition_pair != right.same_cta_partition_pair ||
      left.gate_bf16_ready_before_up_consumer !=
          right.gate_bf16_ready_before_up_consumer ||
      left.stream_k != right.stream_k ||
      left.cuda_implementation_present !=
          right.cuda_implementation_present ||
      left.static_resources_qualified != right.static_resources_qualified ||
      left.numerical_contract_qualified !=
          right.numerical_contract_qualified ||
      left.production_dispatch_eligible !=
          right.production_dispatch_eligible) {
    return false;
  }
  for (std::size_t index = 0U;
       index < kSm87TargetAotProjectionMaximumPartitions; ++index) {
    if (!sm87_target_aot_same_partition(left.partitions[index],
                                        right.partitions[index])) {
      return false;
    }
  }
  return true;
}

constexpr bool Sm87TargetAotProjectionPlan::valid() const noexcept {
  if (role == Sm87TargetAotProjectionRole::kInvalid ||
      role == Sm87TargetAotProjectionRole::kCount) {
    return false;
  }
  return sm87_target_aot_same_plan(
      *this, sm87_target_aot_projection_plan(role, token_count));
}

[[nodiscard]] constexpr Sm87TargetAotProjectionTask
sm87_target_aot_projection_task(
    const Sm87TargetAotProjectionPlan& plan,
    const std::size_t linear_task) noexcept {
  if (!plan.valid() || linear_task >= plan.logical_tasks ||
      plan.raster_group_m == 0U) {
    return {};
  }
  const std::size_t group_span = plan.raster_group_m * plan.grid_n;
  const std::size_t group = linear_task / group_span;
  const std::size_t first_m = group * plan.raster_group_m;
  const std::size_t remaining_m = plan.grid_m - first_m;
  const std::size_t active_m =
      remaining_m < plan.raster_group_m ? remaining_m
                                        : plan.raster_group_m;
  const std::size_t group_offset = linear_task % group_span;
  const std::size_t n_tile = group_offset / active_m;
  if (plan.role == Sm87TargetAotProjectionRole::kNvFp4GateUp) {
    return {first_m + group_offset % active_m, n_tile, 0U, 2U, true};
  }
  std::size_t partition_index = 0U;
  const std::size_t global_n = n_tile * kSm87TargetAotProjectionBlockN;
  while (partition_index + 1U < plan.partition_count &&
         global_n >= plan.partitions[partition_index].output_offset +
                         plan.partitions[partition_index].output_features) {
    ++partition_index;
  }
  return {first_m + group_offset % active_m, n_tile, partition_index, 1U,
          true};
}

[[nodiscard]] constexpr Sm87TargetAotProjectionTask
sm87_target_aot_projection_persistent_task(
    const Sm87TargetAotProjectionPlan& plan, const std::size_t cta,
    const std::size_t iteration) noexcept {
  if (!plan.valid() || cta >= kSm87TargetAotProjectionPersistentCtas ||
      iteration >
          (plan.logical_tasks - 1U) /
              kSm87TargetAotProjectionPersistentCtas) {
    return {};
  }
  return sm87_target_aot_projection_task(
      plan, cta + iteration * kSm87TargetAotProjectionPersistentCtas);
}

static_assert(kSm87TargetAotProjectionThreads ==
              32U * kSm87TargetAotProjectionWarps);
static_assert(kSm87TargetAotProjectionPersistentCtas ==
              kSm87TargetAotProjectionSmCount);
static_assert(kSm87TargetAotNvFp4ABytesPerStage == 16'384U);
static_assert(kSm87TargetAotNvFp4BBytesPerStage == 8'192U);
static_assert(kSm87TargetAotNvFp4ScaleBytesPerStage == 1'024U);
static_assert(kSm87TargetAotNvFp4SharedBytes == 76'800U);
static_assert(kSm87TargetAotFp8SharedBytes == 98'304U);

}  // namespace q3x::kernels
