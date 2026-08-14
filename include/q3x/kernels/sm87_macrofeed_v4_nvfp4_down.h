#pragma once

#include "q3x/kernels/sm87_macrofeed_v3_nvfp4_down.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off C8000 architecture constituent for the K-heavy NVFP4 Down
// projection.  V4 deliberately keeps the authenticated canonical V3 payload
// and the exact Down+residual consumer boundary, but replaces the one-CTA/SM
// M128N256 persistent skeleton with a two-CTA/SM M64N128 ordinary full grid.
enum class Sm87MacroFeedV4NvFp4DownIdentity : std::uint64_t {
  kInvalid = 0U,
  kC8000M64N128K64WarpM64N16TwoStageFullGridV1 =
      0x5133'4d46'5634'4401ULL,
};

inline constexpr Sm87MacroFeedV4NvFp4DownIdentity
    kSm87MacroFeedV4NvFp4DownIdentity =
        Sm87MacroFeedV4NvFp4DownIdentity::
            kC8000M64N128K64WarpM64N16TwoStageFullGridV1;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownTokens = 8'000U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownInputFeatures =
    17'408U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownOutputFeatures =
    5'120U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownBlockM = 64U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownBlockN = 128U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownBlockK = 64U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownWarpM = 64U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownWarpN = 16U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownThreads = 256U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownWarps = 8U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownPipelineStages = 2U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownSmCount = 16U;
inline constexpr std::size_t
    kSm87MacroFeedV4NvFp4DownCanonicalWeightBytesPerCell =
        kSm87MacroFeedV3NvFp4DownWeightBytesPerCell;
inline constexpr std::size_t
    kSm87MacroFeedV4NvFp4DownCanonicalScaleBytesPerCell =
        kSm87MacroFeedV3NvFp4DownScaleBytesPerCell;
inline constexpr std::size_t
    kSm87MacroFeedV4NvFp4DownCanonicalCellBytes =
        kSm87MacroFeedV3NvFp4DownCellBytes;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownWeightBytesPerStage =
    4'096U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownScaleBytesPerStage =
    512U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownDynamicSharedBytes =
    25'600U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownGridM = 125U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownGridN = 40U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownKTiles = 272U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownLogicalTasks = 5'000U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownTailRows = 0U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownPayloadBytes =
    kSm87MacroFeedV3NvFp4DownPayloadBytes;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownPayloadAlignment =
    kSm87MacroFeedV3NvFp4DownPayloadAlignment;

// The numerical oracle crosses both pipeline slots while retaining a complete
// canonical N256 cell.  The helper selects either N128 half without repacking.
inline constexpr std::size_t
    kSm87MacroFeedV4NvFp4DownTestInputFeatures = 256U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownTestKTiles = 4U;
inline constexpr std::size_t kSm87MacroFeedV4NvFp4DownTestPayloadBytes =
    kSm87MacroFeedV4NvFp4DownTestKTiles *
    kSm87MacroFeedV4NvFp4DownCanonicalCellBytes;

struct Sm87MacroFeedV4NvFp4DownPlan final {
  Sm87MacroFeedV4NvFp4DownIdentity identity =
      Sm87MacroFeedV4NvFp4DownIdentity::kInvalid;
  std::size_t token_count = 0U;
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t k_tiles = 0U;
  std::size_t logical_tasks = 0U;
  std::size_t payload_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  bool ordinary_full_grid = false;
  bool m_major_n_adjacent = false;
  bool stream_k = true;
  bool canonical_v3_payload = false;
  bool exact_down_residual = false;
  bool fallback_permitted = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return identity == kSm87MacroFeedV4NvFp4DownIdentity &&
           token_count == kSm87MacroFeedV4NvFp4DownTokens &&
           grid_m == kSm87MacroFeedV4NvFp4DownGridM &&
           grid_n == kSm87MacroFeedV4NvFp4DownGridN &&
           k_tiles == kSm87MacroFeedV4NvFp4DownKTiles &&
           logical_tasks == kSm87MacroFeedV4NvFp4DownLogicalTasks &&
           payload_bytes == kSm87MacroFeedV4NvFp4DownPayloadBytes &&
           dynamic_shared_bytes ==
               kSm87MacroFeedV4NvFp4DownDynamicSharedBytes &&
           ordinary_full_grid && m_major_n_adjacent && !stream_k &&
           canonical_v3_payload && exact_down_residual &&
           !fallback_permitted && !production_dispatch_eligible;
  }
};

[[nodiscard]] constexpr Sm87MacroFeedV4NvFp4DownPlan
sm87_macrofeed_v4_nvfp4_down_plan(const std::size_t token_count) noexcept {
  if (token_count != kSm87MacroFeedV4NvFp4DownTokens) {
    return {};
  }
  return {kSm87MacroFeedV4NvFp4DownIdentity,
          token_count,
          kSm87MacroFeedV4NvFp4DownGridM,
          kSm87MacroFeedV4NvFp4DownGridN,
          kSm87MacroFeedV4NvFp4DownKTiles,
          kSm87MacroFeedV4NvFp4DownLogicalTasks,
          kSm87MacroFeedV4NvFp4DownPayloadBytes,
          kSm87MacroFeedV4NvFp4DownDynamicSharedBytes,
          true,
          true,
          false,
          true,
          true,
          false,
          false};
}

struct Sm87MacroFeedV4NvFp4DownCudaResources final {
  Sm87MacroFeedV4NvFp4DownIdentity identity =
      Sm87MacroFeedV4NvFp4DownIdentity::kInvalid;
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
  std::size_t shared_bytes_per_sm = 0U;
  std::size_t optin_shared_bytes_per_block = 0U;
  bool kernel_compiled = false;
  bool static_resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool sm87_macrofeed_v4_nvfp4_down_resource_gate(
    const Sm87MacroFeedV4NvFp4DownCudaResources& resources) noexcept {
  return resources.identity == kSm87MacroFeedV4NvFp4DownIdentity &&
         resources.device_ordinal >= 0 && resources.compute_major == 8 &&
         resources.compute_minor == 7 &&
         resources.sm_count ==
             static_cast<std::int32_t>(kSm87MacroFeedV4NvFp4DownSmCount) &&
         resources.binary_version == 87 && resources.kernel_compiled &&
         resources.registers_per_thread > 0 &&
         resources.registers_per_thread <= 128 &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kSm87MacroFeedV4NvFp4DownDynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<std::int32_t>(kSm87MacroFeedV4NvFp4DownThreads) &&
         resources.active_blocks_per_sm >= 2 &&
         resources.shared_bytes_per_sm >=
             2U * kSm87MacroFeedV4NvFp4DownDynamicSharedBytes &&
         resources.optin_shared_bytes_per_block >=
             kSm87MacroFeedV4NvFp4DownDynamicSharedBytes &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

struct Sm87MacroFeedV4NvFp4DownArguments final {
  const std::uint16_t* input = nullptr;
  const std::uint8_t* payload = nullptr;
  std::size_t payload_bytes = 0U;
  float tensor_scale = 0.0F;
  std::size_t token_count = 0U;
  std::uint16_t* residual = nullptr;
  void* cuda_stream = nullptr;
  Sm87MacroFeedV3NvFp4DownPayloadReceipt payload_receipt{};
};

struct Sm87MacroFeedV4NvFp4DownLaunchReceipt final {
  Sm87MacroFeedV4NvFp4DownIdentity identity =
      Sm87MacroFeedV4NvFp4DownIdentity::kInvalid;
  std::uint64_t payload_identity = 0U;
  std::size_t token_count = 0U;
  std::size_t logical_tasks = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  std::uint32_t fallback_launches = 0U;
  bool m_major_n_adjacent = false;
  bool stream_k = true;
  bool launch_enqueued = false;
  bool completion_observed = false;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid_enqueue_receipt() const noexcept {
    return identity == kSm87MacroFeedV4NvFp4DownIdentity &&
           payload_identity != 0U &&
           token_count == kSm87MacroFeedV4NvFp4DownTokens &&
           logical_tasks == kSm87MacroFeedV4NvFp4DownLogicalTasks &&
           physical_kernel_launches == 1U && fallback_launches == 0U &&
           m_major_n_adjacent && !stream_k && launch_enqueued &&
           !completion_observed && !production_dispatch_eligible;
  }
};

[[nodiscard]] bool sm87_macrofeed_v4_nvfp4_down_arguments_valid(
    const Sm87MacroFeedV4NvFp4DownArguments& arguments) noexcept;

[[nodiscard]] int query_sm87_macrofeed_v4_nvfp4_down_cuda_resources(
    Sm87MacroFeedV4NvFp4DownCudaResources* resources) noexcept;

[[nodiscard]] int launch_sm87_macrofeed_v4_nvfp4_down_cuda(
    const Sm87MacroFeedV4NvFp4DownArguments& arguments,
    Sm87MacroFeedV4NvFp4DownLaunchReceipt* receipt) noexcept;

// Numerical admission helper. It launches one M64N128 tile over four
// canonical K64 cells. canonical_n_half selects [0,128) or [128,256) from
// each unchanged V3 cell; valid_rows admits M64 and a predicated tail.
[[nodiscard]] int launch_sm87_macrofeed_v4_nvfp4_down_tile_test_cuda(
    const std::uint16_t* input_m64_k256,
    const std::uint8_t* canonical_payload_four_cells, float tensor_scale,
    std::size_t valid_rows, std::size_t canonical_n_half,
    std::uint16_t* residual_m64_n128, void* cuda_stream) noexcept;

static_assert(kSm87MacroFeedV4NvFp4DownGridM *
                  kSm87MacroFeedV4NvFp4DownBlockM ==
              kSm87MacroFeedV4NvFp4DownTokens);
static_assert(kSm87MacroFeedV4NvFp4DownGridN *
                  kSm87MacroFeedV4NvFp4DownBlockN ==
              kSm87MacroFeedV4NvFp4DownOutputFeatures);
static_assert(kSm87MacroFeedV4NvFp4DownKTiles *
                  kSm87MacroFeedV4NvFp4DownBlockK ==
              kSm87MacroFeedV4NvFp4DownInputFeatures);
static_assert(kSm87MacroFeedV4NvFp4DownGridM *
                  kSm87MacroFeedV4NvFp4DownGridN ==
              kSm87MacroFeedV4NvFp4DownLogicalTasks);
static_assert((kSm87MacroFeedV4NvFp4DownGridN / 2U) *
                  kSm87MacroFeedV4NvFp4DownKTiles *
                  kSm87MacroFeedV4NvFp4DownCanonicalCellBytes ==
              kSm87MacroFeedV4NvFp4DownPayloadBytes);
static_assert(2U * (kSm87MacroFeedV4NvFp4DownBlockM *
                        kSm87MacroFeedV4NvFp4DownBlockK *
                        sizeof(std::uint16_t) +
                    kSm87MacroFeedV4NvFp4DownWeightBytesPerStage +
                    kSm87MacroFeedV4NvFp4DownScaleBytesPerStage) ==
              kSm87MacroFeedV4NvFp4DownDynamicSharedBytes);
static_assert(sm87_macrofeed_v4_nvfp4_down_plan(8'000U).valid());
static_assert(!sm87_macrofeed_v4_nvfp4_down_plan(7'999U).valid());

}  // namespace q3x::kernels
