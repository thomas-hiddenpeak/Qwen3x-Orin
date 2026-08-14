#pragma once

#include "q3x/kernels/sm87_target_aot_projection_fp8_cuda.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Independent, default-off T0/T1 constituent of
// AC-PREFILL-SM87-MACROFEED-v3.  The canonical AOT FP8 payload remains owned
// by Sm87TargetAotFp8CudaAssetView; this contract adds no repack, JIT,
// fallback, cuBLASLt, or MTP path and grants no production qualification.
enum class Sm87MacroFeedV3Fp8Identity : std::uint64_t {
  kInvalid = 0U,
  kGdnQkvZM128N256K64Persistent16V1 = 0x5133'4d46'5633'4601ULL,
  kFullQkvM128N256K64Persistent16V1 = 0x5133'4d46'5633'4602ULL,
  kAttentionOutputM128N256K64Persistent16V1 =
      0x5133'4d46'5633'4603ULL,
};

inline constexpr std::size_t kSm87MacroFeedV3Fp8Tokens = 40'000U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8BlockM = 128U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8BlockN = 256U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8BlockK = 64U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8WarpM = 128U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8WarpN = 32U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8Threads = 256U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8Warps = 8U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8PipelineStages = 3U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8PersistentCtas = 16U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8SmCount = 16U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8DynamicSharedBytes =
    98'304U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8TailRows = 64U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8PayloadAlignment = 256U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8TestInputFeatures = 256U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8TestKTiles = 4U;
inline constexpr std::size_t kSm87MacroFeedV3Fp8TestPayloadBytes = 65'536U;

[[nodiscard]] constexpr bool sm87_macrofeed_v3_fp8_role(
    const Sm87TargetAotProjectionRole role) noexcept {
  return role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ ||
         role == Sm87TargetAotProjectionRole::kFp8FullQkv ||
         role == Sm87TargetAotProjectionRole::kFp8AttentionOutput;
}

[[nodiscard]] constexpr Sm87MacroFeedV3Fp8Identity
sm87_macrofeed_v3_fp8_identity(
    const Sm87TargetAotProjectionRole role) noexcept {
  switch (role) {
    case Sm87TargetAotProjectionRole::kFp8GdnQkvZ:
      return Sm87MacroFeedV3Fp8Identity::
          kGdnQkvZM128N256K64Persistent16V1;
    case Sm87TargetAotProjectionRole::kFp8FullQkv:
      return Sm87MacroFeedV3Fp8Identity::
          kFullQkvM128N256K64Persistent16V1;
    case Sm87TargetAotProjectionRole::kFp8AttentionOutput:
      return Sm87MacroFeedV3Fp8Identity::
          kAttentionOutputM128N256K64Persistent16V1;
    default:
      return Sm87MacroFeedV3Fp8Identity::kInvalid;
  }
}

// All 256 bytes are admitted FP8 weight codes.  In particular 0x7f/0xff are
// not canonicalized to NaN: the frozen Marlin bias-shift interpretation
// makes them +480/-480 after the 2^120 scale compensation.
[[nodiscard]] constexpr bool sm87_macrofeed_v3_fp8_weight_code_admitted(
    const std::uint8_t) noexcept {
  return true;
}

[[nodiscard]] constexpr std::uint16_t
sm87_macrofeed_v3_fp8_bias_shift_bf16_bits(
    const std::uint8_t code) noexcept {
  return static_cast<std::uint16_t>(
      (static_cast<std::uint16_t>(code & 0x80U) << 8U) |
      (static_cast<std::uint16_t>(code & 0x7fU) << 4U));
}

struct Sm87MacroFeedV3Fp8Plan final {
  Sm87MacroFeedV3Fp8Identity identity =
      Sm87MacroFeedV3Fp8Identity::kInvalid;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t output_features = 0U;
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t k_tiles = 0U;
  std::size_t logical_tasks = 0U;
  std::size_t raster_group_m = 0U;
  std::size_t partition_count = 0U;
  std::array<std::size_t, 3U> partition_first_n{};
  std::array<std::size_t, 3U> partition_features{};
  std::array<std::size_t, 3U> partition_payload_offsets{};
  std::size_t payload_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  bool noncooperative_persistent_queue = false;
  bool role_specific_raster = false;
  bool tail_predicated = false;
  bool authenticated_asset_zero_copy = false;
  bool no_request_time_repacking = false;
  bool no_request_time_jit = false;
  bool fallback_permitted = true;
  bool cublaslt_permitted = true;
  bool mtp_permitted = true;
  bool exact_fp8_marlin_semantics = false;
  bool t0_t1_only = false;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid() const noexcept;
};

struct Sm87MacroFeedV3Fp8PartitionTask final {
  std::size_t partition = 0U;
  std::size_t local_n_tile = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87MacroFeedV3Fp8PartitionTask
sm87_macrofeed_v3_fp8_partition_task(
    const Sm87MacroFeedV3Fp8Plan& plan,
    const std::size_t n_tile) noexcept {
  if (!plan.valid() || n_tile >= plan.grid_n) {
    return {};
  }
  for (std::size_t index = 0U; index < plan.partition_count; ++index) {
    const std::size_t first =
        plan.partition_first_n[index] / kSm87MacroFeedV3Fp8BlockN;
    const std::size_t tiles =
        plan.partition_features[index] / kSm87MacroFeedV3Fp8BlockN;
    if (n_tile >= first && n_tile < first + tiles) {
      return {index, n_tile - first, true};
    }
  }
  return {};
}

[[nodiscard]] constexpr Sm87MacroFeedV3Fp8Plan
sm87_macrofeed_v3_fp8_plan(const Sm87TargetAotProjectionRole role,
                           const std::size_t token_count) noexcept {
  if (!sm87_macrofeed_v3_fp8_role(role) ||
      token_count != kSm87MacroFeedV3Fp8Tokens) {
    return {};
  }
  Sm87MacroFeedV3Fp8Plan plan;
  plan.identity = sm87_macrofeed_v3_fp8_identity(role);
  plan.role = role;
  plan.token_count = token_count;
  plan.grid_m = 313U;
  plan.dynamic_shared_bytes = kSm87MacroFeedV3Fp8DynamicSharedBytes;
  plan.noncooperative_persistent_queue = true;
  plan.role_specific_raster = true;
  plan.tail_predicated = true;
  plan.authenticated_asset_zero_copy = true;
  plan.no_request_time_repacking = true;
  plan.no_request_time_jit = true;
  plan.fallback_permitted = false;
  plan.cublaslt_permitted = false;
  plan.mtp_permitted = false;
  plan.exact_fp8_marlin_semantics = true;
  plan.t0_t1_only = true;
  plan.production_dispatch_eligible = false;
  if (role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    plan.input_features = 5'120U;
    plan.output_features = 16'384U;
    plan.grid_n = 64U;
    plan.k_tiles = 80U;
    plan.raster_group_m = 2U;
    plan.partition_count = 2U;
    plan.partition_first_n = {0U, 10'240U, 0U};
    plan.partition_features = {10'240U, 6'144U, 0U};
    plan.partition_payload_offsets = {0U, 52'428'800U, 0U};
    plan.payload_bytes = 83'886'080U;
  } else if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    plan.input_features = 5'120U;
    plan.output_features = 14'336U;
    plan.grid_n = 56U;
    plan.k_tiles = 80U;
    plan.raster_group_m = 2U;
    plan.partition_count = 3U;
    plan.partition_first_n = {0U, 12'288U, 13'312U};
    plan.partition_features = {12'288U, 1'024U, 1'024U};
    plan.partition_payload_offsets = {0U, 62'914'560U, 68'157'440U};
    plan.payload_bytes = 73'400'320U;
  } else {
    plan.input_features = 6'144U;
    plan.output_features = 5'120U;
    plan.grid_n = 20U;
    plan.k_tiles = 96U;
    plan.raster_group_m = 1U;
    plan.partition_count = 1U;
    plan.partition_first_n = {0U, 0U, 0U};
    plan.partition_features = {5'120U, 0U, 0U};
    plan.partition_payload_offsets = {0U, 0U, 0U};
    plan.payload_bytes = 31'457'280U;
  }
  plan.logical_tasks = plan.grid_m * plan.grid_n;
  return plan;
}

constexpr bool Sm87MacroFeedV3Fp8Plan::valid() const noexcept {
  if (identity != sm87_macrofeed_v3_fp8_identity(role) ||
      token_count != kSm87MacroFeedV3Fp8Tokens || grid_m != 313U ||
      output_features != grid_n * kSm87MacroFeedV3Fp8BlockN ||
      input_features != k_tiles * kSm87MacroFeedV3Fp8BlockK ||
      logical_tasks != grid_m * grid_n ||
      dynamic_shared_bytes != kSm87MacroFeedV3Fp8DynamicSharedBytes ||
      !noncooperative_persistent_queue || !role_specific_raster ||
      !tail_predicated || !authenticated_asset_zero_copy ||
      !no_request_time_repacking || !no_request_time_jit ||
      fallback_permitted || cublaslt_permitted || mtp_permitted ||
      !exact_fp8_marlin_semantics || !t0_t1_only ||
      production_dispatch_eligible) {
    return false;
  }
  const auto layout = sm87_target_aot_projection_packed_layout(role);
  if (!layout.valid() || layout.payload_bytes != payload_bytes ||
      layout.partition_count != partition_count) {
    return false;
  }
  for (std::size_t index = 0U; index < partition_count; ++index) {
    if (partition_first_n[index] != layout.partitions[index].global_n_offset ||
        partition_features[index] != layout.partitions[index].output_features ||
        partition_payload_offsets[index] !=
            layout.partitions[index].payload_offset) {
      return false;
    }
  }
  return true;
}

struct Sm87MacroFeedV3Fp8CudaResources final {
  Sm87MacroFeedV3Fp8Identity identity =
      Sm87MacroFeedV3Fp8Identity::kInvalid;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
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

[[nodiscard]] constexpr bool sm87_macrofeed_v3_fp8_resource_gate(
    const Sm87MacroFeedV3Fp8CudaResources& resources) noexcept {
  return resources.identity == sm87_macrofeed_v3_fp8_identity(resources.role) &&
         resources.device_ordinal >= 0 && resources.compute_major == 8 &&
         resources.compute_minor == 7 &&
         resources.sm_count ==
             static_cast<std::int32_t>(kSm87MacroFeedV3Fp8SmCount) &&
         resources.binary_version == 87 && resources.kernel_compiled &&
         resources.registers_per_thread > 0 &&
         resources.registers_per_thread <= 255 &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kSm87MacroFeedV3Fp8DynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<std::int32_t>(kSm87MacroFeedV3Fp8Threads) &&
         resources.active_blocks_per_sm >= 1 &&
         resources.optin_shared_bytes_per_block >=
             kSm87MacroFeedV3Fp8DynamicSharedBytes &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

struct Sm87MacroFeedV3Fp8StartupSeal final {
  std::uint64_t seal_identity = 0U;
  Sm87MacroFeedV3Fp8CudaResources resources{};
  bool dynamic_shared_attribute_set = false;
  bool tactic_frozen_before_requests = false;
  bool no_hot_device_queries = false;
  bool no_hot_function_queries = false;
  bool no_hot_occupancy_queries = false;
  bool no_hot_pointer_queries = false;
  bool no_hot_error_state_clear = false;
  bool t0_t1_only = false;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr std::uint64_t sm87_macrofeed_v3_fp8_hash_u64(
    std::uint64_t hash, const std::uint64_t value) noexcept {
  for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
    hash ^= static_cast<std::uint8_t>(value >> (byte * 8U));
    hash *= 1'099'511'628'211ULL;
  }
  return hash;
}

[[nodiscard]] constexpr std::uint64_t
sm87_macrofeed_v3_fp8_compute_startup_seal_identity(
    const Sm87MacroFeedV3Fp8StartupSeal& seal) noexcept {
  std::uint64_t hash = 14'695'981'039'346'656'037ULL;
  hash = sm87_macrofeed_v3_fp8_hash_u64(
      hash, static_cast<std::uint64_t>(seal.resources.identity));
  hash = sm87_macrofeed_v3_fp8_hash_u64(
      hash, static_cast<std::uint64_t>(seal.resources.role));
  hash = sm87_macrofeed_v3_fp8_hash_u64(
      hash, static_cast<std::uint32_t>(seal.resources.device_ordinal));
  hash = sm87_macrofeed_v3_fp8_hash_u64(
      hash, static_cast<std::uint32_t>(seal.resources.registers_per_thread));
  hash = sm87_macrofeed_v3_fp8_hash_u64(
      hash, seal.resources.dynamic_shared_bytes);
  hash = sm87_macrofeed_v3_fp8_hash_u64(
      hash, static_cast<std::uint32_t>(seal.resources.active_blocks_per_sm));
  hash = sm87_macrofeed_v3_fp8_hash_u64(
      hash, seal.resources.static_resource_gate_passed);
  hash = sm87_macrofeed_v3_fp8_hash_u64(
      hash, seal.dynamic_shared_attribute_set);
  hash = sm87_macrofeed_v3_fp8_hash_u64(
      hash, seal.tactic_frozen_before_requests);
  hash = sm87_macrofeed_v3_fp8_hash_u64(hash, seal.no_hot_device_queries);
  hash = sm87_macrofeed_v3_fp8_hash_u64(hash, seal.no_hot_function_queries);
  hash = sm87_macrofeed_v3_fp8_hash_u64(hash, seal.no_hot_occupancy_queries);
  hash = sm87_macrofeed_v3_fp8_hash_u64(hash, seal.no_hot_pointer_queries);
  hash = sm87_macrofeed_v3_fp8_hash_u64(hash, seal.no_hot_error_state_clear);
  hash = sm87_macrofeed_v3_fp8_hash_u64(hash, seal.t0_t1_only);
  return sm87_macrofeed_v3_fp8_hash_u64(
      hash, seal.production_dispatch_eligible);
}

[[nodiscard]] constexpr bool sm87_macrofeed_v3_fp8_startup_seal_valid(
    const Sm87MacroFeedV3Fp8StartupSeal& seal) noexcept {
  return seal.seal_identity != 0U &&
         seal.seal_identity ==
             sm87_macrofeed_v3_fp8_compute_startup_seal_identity(seal) &&
         sm87_macrofeed_v3_fp8_resource_gate(seal.resources) &&
         seal.resources.static_resource_gate_passed &&
         seal.dynamic_shared_attribute_set &&
         seal.tactic_frozen_before_requests && seal.no_hot_device_queries &&
         seal.no_hot_function_queries && seal.no_hot_occupancy_queries &&
         seal.no_hot_pointer_queries && seal.no_hot_error_state_clear &&
         seal.t0_t1_only && !seal.production_dispatch_eligible;
}

struct Sm87MacroFeedV3Fp8Arguments final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  const std::uint16_t* input = nullptr;
  Sm87TargetAotFp8CudaAssetView asset{};
  std::size_t token_count = 0U;
  // Independent tensor scales require independent publication intervals.
  // GDN uses QKV/Z, Full Attention uses QGate/K/V, and O uses slot zero.
  std::array<std::uint16_t*, 3U> partition_outputs{};
  void* cuda_stream = nullptr;
};

struct Sm87MacroFeedV3Fp8LaunchReceipt final {
  Sm87MacroFeedV3Fp8Identity identity =
      Sm87MacroFeedV3Fp8Identity::kInvalid;
  std::uint64_t artifact_identity = 0U;
  std::size_t token_count = 0U;
  std::size_t logical_tasks = 0U;
  std::size_t tail_rows = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  std::uint32_t fallback_launches = 0U;
  bool authenticated_asset_zero_copy = false;
  bool launch_enqueued = false;
  bool completion_observed = false;
  bool t0_t1_only = false;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid_enqueue_receipt(
      const Sm87TargetAotProjectionRole role) const noexcept {
    const auto plan = sm87_macrofeed_v3_fp8_plan(role, token_count);
    return plan.valid() && identity == plan.identity &&
           artifact_identity != 0U && logical_tasks == plan.logical_tasks &&
           tail_rows == kSm87MacroFeedV3Fp8TailRows &&
           physical_kernel_launches == 1U && fallback_launches == 0U &&
           authenticated_asset_zero_copy && launch_enqueued &&
           !completion_observed && t0_t1_only &&
           !production_dispatch_eligible;
  }
};

[[nodiscard]] bool sm87_macrofeed_v3_fp8_arguments_valid(
    const Sm87MacroFeedV3Fp8Arguments& arguments) noexcept;

[[nodiscard]] int query_sm87_macrofeed_v3_fp8_cuda_resources(
    Sm87TargetAotProjectionRole role,
    Sm87MacroFeedV3Fp8CudaResources* resources) noexcept;

[[nodiscard]] int seal_sm87_macrofeed_v3_fp8_startup_cuda(
    Sm87TargetAotProjectionRole role,
    Sm87MacroFeedV3Fp8StartupSeal* seal) noexcept;

[[nodiscard]] int launch_sm87_macrofeed_v3_fp8_cuda(
    const Sm87MacroFeedV3Fp8Arguments& arguments,
    Sm87MacroFeedV3Fp8LaunchReceipt* receipt) noexcept;

// Request-hot form.  It performs no device, function, occupancy, or pointer
// query, never clears CUDA's prior error state, and consumes the existing
// authenticated payload interval without copying or repacking it.
[[nodiscard]] int launch_sm87_macrofeed_v3_fp8_sealed_cuda(
    const Sm87MacroFeedV3Fp8Arguments& arguments,
    const Sm87MacroFeedV3Fp8StartupSeal& startup_seal,
    Sm87MacroFeedV3Fp8LaunchReceipt* receipt) noexcept;

// T1-only canonical-cell oracle.  Exactly one M128N256 tile consumes four
// K64 cells; valid_rows is restricted to 128 or the P40 M64 tail.  The
// partition index authenticates the role-specific semantic fixture only: this
// seam deliberately does not qualify the production partition offsets,
// scatter stores, 16-CTA scheduler, or the complete 80/96-stage K lifetime.
[[nodiscard]] int launch_sm87_macrofeed_v3_fp8_tile_test_cuda(
    Sm87TargetAotProjectionRole role, std::size_t partition_index,
    const std::uint16_t* input_m128_k256,
    const std::uint8_t* canonical_payload_four_cells,
    std::uint16_t compensated_scale_bf16_bits, std::size_t valid_rows,
    std::uint16_t* output_m128_n256, void* cuda_stream) noexcept;

// T1-only bit-domain witness for all 256 raw FP8 codes.  This is not an
// all-code production MMA arithmetic qualification.
[[nodiscard]] int launch_sm87_macrofeed_v3_fp8_code_test_cuda(
    const std::uint8_t* codes_256, std::uint16_t* bias_shift_bits_256,
    void* cuda_stream) noexcept;

static_assert(kSm87MacroFeedV3Fp8DynamicSharedBytes ==
              kSm87TargetAotFp8SharedBytes);
static_assert(kSm87MacroFeedV3Fp8WarpM == 128U &&
              kSm87MacroFeedV3Fp8WarpN == 32U);
static_assert(sm87_macrofeed_v3_fp8_plan(
                  Sm87TargetAotProjectionRole::kFp8GdnQkvZ, 40'000U)
                  .valid());
static_assert(sm87_macrofeed_v3_fp8_plan(
                  Sm87TargetAotProjectionRole::kFp8FullQkv, 40'000U)
                  .valid());
static_assert(sm87_macrofeed_v3_fp8_plan(
                  Sm87TargetAotProjectionRole::kFp8AttentionOutput, 40'000U)
                  .valid());
static_assert(!sm87_macrofeed_v3_fp8_plan(
                   Sm87TargetAotProjectionRole::kFp8GdnQkvZ, 39'999U)
                   .valid());
static_assert(sm87_macrofeed_v3_fp8_bias_shift_bf16_bits(0x7fU) ==
              0x07f0U);
static_assert(sm87_macrofeed_v3_fp8_bias_shift_bf16_bits(0xffU) ==
              0x87f0U);

}  // namespace q3x::kernels
