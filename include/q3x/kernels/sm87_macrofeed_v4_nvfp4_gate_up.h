#pragma once

#include "q3x/kernels/sm87_macrofeed_v3_nvfp4_gate_up.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off MacroFeed-v4 Gate+Up constituent.  It consumes the canonical
// V3 target-AOT payload in place; it has no production selector, fallback,
// repack, JIT, or autotune path.  The M8000 declaration is intentionally
// narrower than the complete P40000 route while this physical feed is being
// qualified independently.
enum class Sm87MacroFeedV4NvFp4GateUpIdentity : std::uint64_t {
  kInvalid = 0U,
  kM64N128K64TwoStagePersistent32V1 = 0x5133'4d46'5634'4701ULL,
};

inline constexpr Sm87MacroFeedV4NvFp4GateUpIdentity
    kSm87MacroFeedV4NvFp4GateUpIdentity =
        Sm87MacroFeedV4NvFp4GateUpIdentity::
            kM64N128K64TwoStagePersistent32V1;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpTokens = 8'000U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpInputFeatures =
    5'120U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpOutputFeatures =
    17'408U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpBranches = 2U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpBlockM = 64U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpBlockN = 128U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpBlockK = 64U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpThreads = 256U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpWarps = 8U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpWarpsPerBranch = 4U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpPipelineStages = 2U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpPersistentCtas = 32U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpSmCount = 16U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpGridM = 125U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpGridN = 136U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpKTiles = 80U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpLogicalTasks =
    17'000U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpActivationBytes =
    8'192U;
inline constexpr std::size_t
    kSm87MacroFeedV4NvFp4GateUpBranchWeightBytes = 4'096U;
inline constexpr std::size_t
    kSm87MacroFeedV4NvFp4GateUpBranchScaleBytes = 512U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpStageBytes =
    17'408U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpDynamicSharedBytes =
    34'816U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpPayloadBytes =
    kSm87MacroFeedV3NvFp4GateUpPayloadBytes;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpPartitionBytes =
    kSm87MacroFeedV3NvFp4GateUpPartitionBytes;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpPayloadAlignment =
    kSm87MacroFeedV3NvFp4GateUpPayloadAlignment;

// T1 keeps four canonical V3 K64 cells so the two-stage ring turns over.
inline constexpr std::size_t
    kSm87MacroFeedV4NvFp4GateUpTestInputFeatures = 256U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpTestKTiles = 4U;
inline constexpr std::size_t
    kSm87MacroFeedV4NvFp4GateUpTestPartitionBytes =
        kSm87MacroFeedV4NvFp4GateUpTestKTiles *
        kSm87MacroFeedV3NvFp4GateUpCellBytes;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4GateUpTestPayloadBytes =
    kSm87MacroFeedV4NvFp4GateUpBranches *
    kSm87MacroFeedV4NvFp4GateUpTestPartitionBytes;

struct Sm87MacroFeedV4NvFp4GateUpPlan final {
  Sm87MacroFeedV4NvFp4GateUpIdentity identity =
      Sm87MacroFeedV4NvFp4GateUpIdentity::kInvalid;
  std::size_t token_count = 0U;
  std::size_t logical_tasks = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  bool canonical_v3_payload_zero_copy = false;
  bool shared_a = false;
  bool independent_gate_up = false;
  bool two_cta_per_sm_required = false;
  bool fallback_permitted = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return identity == kSm87MacroFeedV4NvFp4GateUpIdentity &&
           token_count == kSm87MacroFeedV4NvFp4GateUpTokens &&
           logical_tasks == kSm87MacroFeedV4NvFp4GateUpLogicalTasks &&
           dynamic_shared_bytes ==
               kSm87MacroFeedV4NvFp4GateUpDynamicSharedBytes &&
           canonical_v3_payload_zero_copy && shared_a &&
           independent_gate_up && two_cta_per_sm_required &&
           !fallback_permitted && !production_dispatch_eligible;
  }
};

[[nodiscard]] constexpr Sm87MacroFeedV4NvFp4GateUpPlan
sm87_macrofeed_v4_nvfp4_gate_up_plan(
    const std::size_t token_count) noexcept {
  if (token_count != kSm87MacroFeedV4NvFp4GateUpTokens) {
    return {};
  }
  return {kSm87MacroFeedV4NvFp4GateUpIdentity,
          token_count,
          kSm87MacroFeedV4NvFp4GateUpLogicalTasks,
          kSm87MacroFeedV4NvFp4GateUpDynamicSharedBytes,
          true,
          true,
          true,
          true,
          false,
          false};
}

struct Sm87MacroFeedV4NvFp4GateUpCudaResources final {
  Sm87MacroFeedV4NvFp4GateUpIdentity identity =
      Sm87MacroFeedV4NvFp4GateUpIdentity::kInvalid;
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
  bool kernel_compiled = false;
  bool static_resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool
sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(
    const Sm87MacroFeedV4NvFp4GateUpCudaResources& resources) noexcept {
  return resources.identity == kSm87MacroFeedV4NvFp4GateUpIdentity &&
         resources.device_ordinal >= 0 && resources.compute_major == 8 &&
         resources.compute_minor == 7 &&
         resources.sm_count ==
             static_cast<std::int32_t>(kSm87MacroFeedV4NvFp4GateUpSmCount) &&
         resources.binary_version == 87 && resources.kernel_compiled &&
         resources.registers_per_thread > 0 &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kSm87MacroFeedV4NvFp4GateUpDynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<std::int32_t>(kSm87MacroFeedV4NvFp4GateUpThreads) &&
         resources.active_blocks_per_sm >= 2 &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

struct Sm87MacroFeedV4NvFp4GateUpArguments final {
  const std::uint16_t* input = nullptr;
  const std::uint8_t* payload = nullptr;
  std::size_t payload_bytes = 0U;
  float gate_tensor_scale = 0.0F;
  float up_tensor_scale = 0.0F;
  std::size_t token_count = 0U;
  std::uint16_t* output = nullptr;
  void* cuda_stream = nullptr;
  Sm87MacroFeedV3NvFp4GateUpPayloadReceipt canonical_v3_payload_receipt{};
};

struct Sm87MacroFeedV4NvFp4GateUpLaunchReceipt final {
  Sm87MacroFeedV4NvFp4GateUpIdentity identity =
      Sm87MacroFeedV4NvFp4GateUpIdentity::kInvalid;
  std::uint64_t payload_identity = 0U;
  std::size_t token_count = 0U;
  std::size_t logical_tasks = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  bool shared_a = false;
  bool independent_gate_up = false;
  bool canonical_v3_payload_zero_copy = false;
  bool launch_enqueued = false;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid_enqueue_receipt() const noexcept {
    return identity == kSm87MacroFeedV4NvFp4GateUpIdentity &&
           payload_identity != 0U &&
           token_count == kSm87MacroFeedV4NvFp4GateUpTokens &&
           logical_tasks == kSm87MacroFeedV4NvFp4GateUpLogicalTasks &&
           physical_kernel_launches == 1U && shared_a &&
           independent_gate_up && canonical_v3_payload_zero_copy &&
           launch_enqueued && !production_dispatch_eligible;
  }
};

[[nodiscard]] bool sm87_macrofeed_v4_nvfp4_gate_up_arguments_valid(
    const Sm87MacroFeedV4NvFp4GateUpArguments& arguments) noexcept;

[[nodiscard]] int query_sm87_macrofeed_v4_nvfp4_gate_up_cuda_resources(
    Sm87MacroFeedV4NvFp4GateUpCudaResources* resources) noexcept;

[[nodiscard]] int launch_sm87_macrofeed_v4_nvfp4_gate_up_cuda(
    const Sm87MacroFeedV4NvFp4GateUpArguments& arguments,
    Sm87MacroFeedV4NvFp4GateUpLaunchReceipt* receipt) noexcept;

// Synthetic T1 helper only. valid_rows=64 covers the full M64 cell; any value
// in [1,63] exercises the predicated tail. canonical_n_half selects either
// N128 half of the unchanged canonical V3 N256 cell.
[[nodiscard]] int launch_sm87_macrofeed_v4_nvfp4_gate_up_tile_test_cuda(
    const std::uint16_t* input_m64_k256,
    const std::uint8_t* canonical_gate_then_up_payload,
    float gate_tensor_scale, float up_tensor_scale, std::size_t valid_rows,
    std::size_t canonical_n_half, std::uint16_t* output_m64_n128,
    void* cuda_stream) noexcept;

static_assert(kSm87MacroFeedV4NvFp4GateUpGridM *
                      kSm87MacroFeedV4NvFp4GateUpBlockM ==
                  kSm87MacroFeedV4NvFp4GateUpTokens);
static_assert(kSm87MacroFeedV4NvFp4GateUpGridN *
                      kSm87MacroFeedV4NvFp4GateUpBlockN ==
                  kSm87MacroFeedV4NvFp4GateUpOutputFeatures);
static_assert(kSm87MacroFeedV4NvFp4GateUpKTiles *
                      kSm87MacroFeedV4NvFp4GateUpBlockK ==
                  kSm87MacroFeedV4NvFp4GateUpInputFeatures);
static_assert(kSm87MacroFeedV4NvFp4GateUpGridM *
                      kSm87MacroFeedV4NvFp4GateUpGridN ==
                  kSm87MacroFeedV4NvFp4GateUpLogicalTasks);
static_assert(kSm87MacroFeedV4NvFp4GateUpDynamicSharedBytes ==
              kSm87MacroFeedV4NvFp4GateUpPipelineStages *
                  kSm87MacroFeedV4NvFp4GateUpStageBytes);
static_assert(sm87_macrofeed_v4_nvfp4_gate_up_plan(8'000U).valid());
static_assert(!sm87_macrofeed_v4_nvfp4_gate_up_plan(40'000U).valid());

}  // namespace q3x::kernels
