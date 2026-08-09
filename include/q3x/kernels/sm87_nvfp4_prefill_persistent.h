#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// BUILD_TESTING-only physical surface for WP-V2-C1-v2.  It keeps the frozen
// Marlin BF16 x NVFP4 M64N256K64, four-stage compute/decode body and replaces
// only output-tile ownership.  The production/default Marlin routes do not
// call this API.
enum class Sm87NvFp4PersistentPrefillRole : std::uint8_t {
  kGateUpPaired = 0U,
  kDown = 1U,
};

enum class Sm87NvFp4PersistentPrefillRaster : std::uint8_t {
  // A four-CTA cohort consumes one N256 tile for four M64 tiles, then advances
  // N.  A group retains the same four A tiles across the
  // complete Gate/Up N sweep.  The sidecar must interleave Gate/Up columns so
  // the final writer can publish BF16 SiLU(gate) * up without a merged tensor.
  kGateUpGroupedM4NMajor = 0U,
  // All M64 tiles for one Down N256 slab are traversed before the next slab.
  // One 16-CTA wave therefore requests the same packed B/scale tile.
  kDownBStationaryN256 = 1U,
};

enum class Sm87NvFp4PersistentPrefillTailPolicy : std::uint8_t {
  kNone = 0U,
  // The aligned prefix can use the P40 body, but admission remains closed
  // until an exact M32 companion and its ordered publication are implemented.
  kNativeM32CompanionRequired = 1U,
  // Reserved for the balanced M5424/general-tail architecture contract.  No
  // compatibility or legacy fallback is hidden behind this surface.
  kGeneralTailCompanionRequired = 2U,
};

inline constexpr std::size_t kSm87NvFp4PersistentPrefillP40Tokens = 40'000U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillP60Tokens = 60'000U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillBalancedP60TailTokens =
    5'424U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillHidden = 5'120U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillIntermediate = 17'408U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillMergedGateUp = 34'816U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillTileM = 64U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillTileN = 256U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillTileK = 64U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillPipelineStages = 4U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillThreads = 256U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillCtas = 16U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillGateMGroupTiles = 4U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillDynamicSharedBytes =
    166'912U;
// Audited ptxas spill/local-memory ceilings for the two exact-P40 kernels.
// Small fixed compiler spill surfaces are admissible; an unexpected increase
// fails binding before the production request path can launch the candidate.
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillGateMaximumLocalBytes = 32U;
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillDownMaximumLocalBytes = 16U;
inline constexpr std::size_t kSm87NvFp4PersistentPrefillL2BudgetBytes =
    4U * 1'024U * 1'024U;
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillGateBBytesPerN256 =
        kSm87NvFp4PersistentPrefillHidden *
        kSm87NvFp4PersistentPrefillTileN / 2U;
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillGateScaleBytesPerN256 =
        kSm87NvFp4PersistentPrefillHidden *
        kSm87NvFp4PersistentPrefillTileN / 16U;
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillGateABytesPerM64 =
        kSm87NvFp4PersistentPrefillTileM *
        kSm87NvFp4PersistentPrefillHidden * sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillGateOneCohortFullSlabBytes =
        kSm87NvFp4PersistentPrefillGateBBytesPerN256 +
        kSm87NvFp4PersistentPrefillGateScaleBytesPerN256 +
        kSm87NvFp4PersistentPrefillGateMGroupTiles *
            kSm87NvFp4PersistentPrefillGateABytesPerM64;
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillGateConcurrentFullSlabBytes =
        4U * (kSm87NvFp4PersistentPrefillGateBBytesPerN256 +
              kSm87NvFp4PersistentPrefillGateScaleBytesPerN256) +
        kSm87NvFp4PersistentPrefillGateMGroupTiles *
            kSm87NvFp4PersistentPrefillGateABytesPerM64;
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillGateABytesPerM64K64Stage =
        kSm87NvFp4PersistentPrefillTileM *
        kSm87NvFp4PersistentPrefillTileK * sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillGateBBytesPerN256K64Stage =
        kSm87NvFp4PersistentPrefillTileK *
        kSm87NvFp4PersistentPrefillTileN / 2U;
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillGateScaleBytesPerN256K64Stage =
        kSm87NvFp4PersistentPrefillTileK *
        kSm87NvFp4PersistentPrefillTileN / 16U;
// One 16-CTA wave contains four N-slab cohorts.  A is shared logically across
// their four M tiles; each N cohort owns one B/scale stage.  Marlin's A, B and
// scale fetches are cp.async.cg, so L1 is bypassed and the intended reuse is
// temporal L2 reuse of the current K64 stage, not false whole-slab residency.
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillGateConcurrentK64StageL2Bytes =
        kSm87NvFp4PersistentPrefillGateMGroupTiles *
            kSm87NvFp4PersistentPrefillGateABytesPerM64K64Stage +
        4U * (kSm87NvFp4PersistentPrefillGateBBytesPerN256K64Stage +
              kSm87NvFp4PersistentPrefillGateScaleBytesPerN256K64Stage);
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillGateFourStageL2Bytes =
        kSm87NvFp4PersistentPrefillPipelineStages *
        kSm87NvFp4PersistentPrefillGateConcurrentK64StageL2Bytes;
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillDownBBytesPerN256K64Stage =
        kSm87NvFp4PersistentPrefillTileK *
        kSm87NvFp4PersistentPrefillTileN / 2U;
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillDownScaleBytesPerN256K64Stage =
        kSm87NvFp4PersistentPrefillTileK *
        kSm87NvFp4PersistentPrefillTileN / 16U;
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillDownConcurrentK64StageL2Bytes =
        kSm87NvFp4PersistentPrefillCtas *
            (kSm87NvFp4PersistentPrefillTileM *
             kSm87NvFp4PersistentPrefillTileK * sizeof(std::uint16_t)) +
        kSm87NvFp4PersistentPrefillDownBBytesPerN256K64Stage +
        kSm87NvFp4PersistentPrefillDownScaleBytesPerN256K64Stage;
inline constexpr std::size_t
    kSm87NvFp4PersistentPrefillDownFourStageL2Bytes =
        kSm87NvFp4PersistentPrefillPipelineStages *
        kSm87NvFp4PersistentPrefillDownConcurrentK64StageL2Bytes;

static_assert(kSm87NvFp4PersistentPrefillP40Tokens %
                      kSm87NvFp4PersistentPrefillTileM ==
                  0U,
              "P40 must be one exact M64 surface");
static_assert(kSm87NvFp4PersistentPrefillP60Tokens %
                      kSm87NvFp4PersistentPrefillTileM ==
                  32U,
              "P60 freezes an explicit M32 tail contract");
static_assert(kSm87NvFp4PersistentPrefillBalancedP60TailTokens %
                      kSm87NvFp4PersistentPrefillTileM ==
                  48U,
              "balanced P60 panel freezes a distinct general-tail contract");
static_assert(kSm87NvFp4PersistentPrefillMergedGateUp %
                      kSm87NvFp4PersistentPrefillTileN ==
                  0U);
static_assert(kSm87NvFp4PersistentPrefillHidden %
                      kSm87NvFp4PersistentPrefillTileN ==
                  0U);
static_assert(kSm87NvFp4PersistentPrefillGateBBytesPerN256 == 655'360U);
static_assert(kSm87NvFp4PersistentPrefillGateScaleBytesPerN256 == 81'920U);
static_assert(kSm87NvFp4PersistentPrefillGateABytesPerM64 == 655'360U);
static_assert(kSm87NvFp4PersistentPrefillGateOneCohortFullSlabBytes ==
              3'358'720U);
static_assert(kSm87NvFp4PersistentPrefillGateConcurrentFullSlabBytes ==
              5'570'560U);
static_assert(kSm87NvFp4PersistentPrefillGateConcurrentFullSlabBytes >
                  kSm87NvFp4PersistentPrefillL2BudgetBytes,
              "the policy must not claim whole-slab L2 residency");
static_assert(kSm87NvFp4PersistentPrefillGateConcurrentK64StageL2Bytes ==
              69'632U);
static_assert(kSm87NvFp4PersistentPrefillGateFourStageL2Bytes == 278'528U);
static_assert(kSm87NvFp4PersistentPrefillGateFourStageL2Bytes <
              kSm87NvFp4PersistentPrefillL2BudgetBytes);
static_assert(kSm87NvFp4PersistentPrefillDownConcurrentK64StageL2Bytes ==
              140'288U);
static_assert(kSm87NvFp4PersistentPrefillDownFourStageL2Bytes == 561'152U);
static_assert(kSm87NvFp4PersistentPrefillDownFourStageL2Bytes <
              kSm87NvFp4PersistentPrefillL2BudgetBytes);

struct Sm87NvFp4PersistentPrefillShapeContract {
  std::size_t token_count = 0U;
  std::size_t aligned_prefix_tokens = 0U;
  std::size_t tail_tokens = 0U;
  Sm87NvFp4PersistentPrefillTailPolicy tail_policy =
      Sm87NvFp4PersistentPrefillTailPolicy::kGeneralTailCompanionRequired;
  bool admitted = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return token_count != 0U &&
           aligned_prefix_tokens + tail_tokens == token_count &&
           aligned_prefix_tokens % kSm87NvFp4PersistentPrefillTileM == 0U &&
           ((tail_tokens == 0U &&
             tail_policy == Sm87NvFp4PersistentPrefillTailPolicy::kNone &&
             admitted) ||
            (tail_tokens == 32U &&
             tail_policy == Sm87NvFp4PersistentPrefillTailPolicy::
                                kNativeM32CompanionRequired &&
             !admitted) ||
            (tail_tokens != 0U && tail_tokens != 32U &&
             tail_policy == Sm87NvFp4PersistentPrefillTailPolicy::
                                kGeneralTailCompanionRequired &&
             !admitted));
  }
};

[[nodiscard]] constexpr Sm87NvFp4PersistentPrefillShapeContract
sm87_nvfp4_persistent_prefill_shape_contract(
    const std::size_t token_count) noexcept {
  if (token_count == 0U) {
    return {};
  }
  const std::size_t aligned =
      token_count - token_count % kSm87NvFp4PersistentPrefillTileM;
  const std::size_t tail = token_count - aligned;
  if (token_count == kSm87NvFp4PersistentPrefillP40Tokens) {
    return {token_count, aligned, tail,
            Sm87NvFp4PersistentPrefillTailPolicy::kNone, true};
  }
  if (tail == 32U) {
    return {token_count, aligned, tail,
            Sm87NvFp4PersistentPrefillTailPolicy::
                kNativeM32CompanionRequired,
            false};
  }
  return {token_count, aligned, tail,
          Sm87NvFp4PersistentPrefillTailPolicy::
              kGeneralTailCompanionRequired,
          false};
}

struct Sm87NvFp4PersistentPrefillTask {
  std::size_t m_tile = 0U;
  std::size_t n_tile = 0U;
  bool valid = false;
};

struct Sm87NvFp4PersistentPrefillPlan {
  Sm87NvFp4PersistentPrefillRole role =
      Sm87NvFp4PersistentPrefillRole::kGateUpPaired;
  Sm87NvFp4PersistentPrefillRaster raster =
      Sm87NvFp4PersistentPrefillRaster::kGateUpGroupedM4NMajor;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t weight_output_features = 0U;
  std::size_t published_output_features = 0U;
  std::size_t m_tiles = 0U;
  std::size_t n_tiles = 0U;
  std::size_t task_count = 0U;
  std::size_t persistent_ctas = 0U;
  std::size_t tasks_per_cta_upper_bound = 0U;
  std::size_t gate_m_group_tiles = 0U;
  std::size_t tile_m = 0U;
  std::size_t tile_n = 0U;
  std::size_t tile_k = 0U;
  std::size_t threads = 0U;
  std::size_t pipeline_stages = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  bool split_k = true;
  bool stream_k = true;
  bool fp32_accumulation = false;
  bool bf16_publication_boundary = false;
  bool requires_interleaved_gate_up_sidecar = false;
  bool fused_down_residual = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    const bool gate =
        role == Sm87NvFp4PersistentPrefillRole::kGateUpPaired;
    const bool down = role == Sm87NvFp4PersistentPrefillRole::kDown;
    const std::size_t expected_input =
        gate ? kSm87NvFp4PersistentPrefillHidden
             : kSm87NvFp4PersistentPrefillIntermediate;
    const std::size_t expected_weight_output =
        gate ? kSm87NvFp4PersistentPrefillMergedGateUp
             : kSm87NvFp4PersistentPrefillHidden;
    const std::size_t expected_published_output =
        gate ? kSm87NvFp4PersistentPrefillIntermediate
             : kSm87NvFp4PersistentPrefillHidden;
    const std::size_t expected_m_tiles =
        token_count / kSm87NvFp4PersistentPrefillTileM;
    const std::size_t expected_n_tiles =
        expected_weight_output / kSm87NvFp4PersistentPrefillTileN;
    return (gate || down) &&
           sm87_nvfp4_persistent_prefill_shape_contract(token_count)
               .admitted &&
           raster ==
               (gate ? Sm87NvFp4PersistentPrefillRaster::
                           kGateUpGroupedM4NMajor
                     : Sm87NvFp4PersistentPrefillRaster::
                           kDownBStationaryN256) &&
           input_features == expected_input &&
           weight_output_features == expected_weight_output &&
           published_output_features == expected_published_output &&
           m_tiles == expected_m_tiles && n_tiles == expected_n_tiles &&
           task_count == expected_m_tiles * expected_n_tiles &&
           persistent_ctas == kSm87NvFp4PersistentPrefillCtas &&
           tasks_per_cta_upper_bound ==
               (task_count + persistent_ctas - 1U) / persistent_ctas &&
           gate_m_group_tiles ==
               (gate ? kSm87NvFp4PersistentPrefillGateMGroupTiles : 0U) &&
           tile_m == kSm87NvFp4PersistentPrefillTileM &&
           tile_n == kSm87NvFp4PersistentPrefillTileN &&
           tile_k == kSm87NvFp4PersistentPrefillTileK &&
           threads == kSm87NvFp4PersistentPrefillThreads &&
           pipeline_stages == kSm87NvFp4PersistentPrefillPipelineStages &&
           dynamic_shared_bytes ==
               kSm87NvFp4PersistentPrefillDynamicSharedBytes &&
           !split_k && !stream_k && fp32_accumulation &&
           bf16_publication_boundary &&
           requires_interleaved_gate_up_sidecar == gate &&
           fused_down_residual == down;
  }
};

[[nodiscard]] constexpr Sm87NvFp4PersistentPrefillPlan
sm87_nvfp4_persistent_prefill_plan(
    const Sm87NvFp4PersistentPrefillRole role,
    const std::size_t token_count) noexcept {
  const bool gate =
      role == Sm87NvFp4PersistentPrefillRole::kGateUpPaired;
  const bool down = role == Sm87NvFp4PersistentPrefillRole::kDown;
  const auto shape =
      sm87_nvfp4_persistent_prefill_shape_contract(token_count);
  if ((!gate && !down) || !shape.admitted) {
    return {};
  }
  const std::size_t input_features =
      gate ? kSm87NvFp4PersistentPrefillHidden
           : kSm87NvFp4PersistentPrefillIntermediate;
  const std::size_t weight_output_features =
      gate ? kSm87NvFp4PersistentPrefillMergedGateUp
           : kSm87NvFp4PersistentPrefillHidden;
  const std::size_t published_output_features =
      gate ? kSm87NvFp4PersistentPrefillIntermediate
           : kSm87NvFp4PersistentPrefillHidden;
  const std::size_t m_tiles =
      token_count / kSm87NvFp4PersistentPrefillTileM;
  const std::size_t n_tiles =
      weight_output_features / kSm87NvFp4PersistentPrefillTileN;
  const std::size_t task_count = m_tiles * n_tiles;
  return {role,
          gate ? Sm87NvFp4PersistentPrefillRaster::kGateUpGroupedM4NMajor
               : Sm87NvFp4PersistentPrefillRaster::kDownBStationaryN256,
          token_count,
          input_features,
          weight_output_features,
          published_output_features,
          m_tiles,
          n_tiles,
          task_count,
          kSm87NvFp4PersistentPrefillCtas,
          (task_count + kSm87NvFp4PersistentPrefillCtas - 1U) /
              kSm87NvFp4PersistentPrefillCtas,
          gate ? kSm87NvFp4PersistentPrefillGateMGroupTiles : 0U,
          kSm87NvFp4PersistentPrefillTileM,
          kSm87NvFp4PersistentPrefillTileN,
          kSm87NvFp4PersistentPrefillTileK,
          kSm87NvFp4PersistentPrefillThreads,
          kSm87NvFp4PersistentPrefillPipelineStages,
          kSm87NvFp4PersistentPrefillDynamicSharedBytes,
          false,
          false,
          true,
          true,
          gate,
          down};
}

// Host/device-independent mapping used by deployment-plan validation.  The
// CUDA kernel owns tasks by `blockIdx.x + wave * 16` and applies exactly this
// bijection before entering the unchanged Marlin K pipeline.
[[nodiscard]] constexpr Sm87NvFp4PersistentPrefillTask
sm87_nvfp4_persistent_prefill_task(
    const Sm87NvFp4PersistentPrefillPlan& plan,
    const std::size_t linear_task) noexcept {
  if (!plan.valid() || linear_task >= plan.task_count) {
    return {};
  }
  if (plan.role == Sm87NvFp4PersistentPrefillRole::kDown) {
    return {linear_task % plan.m_tiles, linear_task / plan.m_tiles, true};
  }

  const std::size_t complete_group_tasks =
      plan.gate_m_group_tiles * plan.n_tiles;
  const std::size_t group = linear_task / complete_group_tasks;
  const std::size_t group_begin_m = group * plan.gate_m_group_tiles;
  const std::size_t group_rows =
      (plan.m_tiles - group_begin_m) < plan.gate_m_group_tiles
          ? (plan.m_tiles - group_begin_m)
          : plan.gate_m_group_tiles;
  const std::size_t in_group = linear_task - group * complete_group_tasks;
  return {group_begin_m + in_group % group_rows, in_group / group_rows, true};
}

static_assert(
    sm87_nvfp4_persistent_prefill_shape_contract(40'000U).valid());
static_assert(
    sm87_nvfp4_persistent_prefill_shape_contract(40'000U).admitted);
static_assert(
    sm87_nvfp4_persistent_prefill_shape_contract(60'000U).valid());
static_assert(
    !sm87_nvfp4_persistent_prefill_shape_contract(60'000U).admitted);
static_assert(
    sm87_nvfp4_persistent_prefill_shape_contract(60'000U).tail_tokens == 32U);
static_assert(
    sm87_nvfp4_persistent_prefill_shape_contract(5'424U).tail_tokens == 48U);

struct Sm87NvFp4PersistentPrefillCapability {
  Sm87NvFp4PersistentPrefillPlan plan{};
  int device = -1;
  int compute_major = 0;
  int compute_minor = 0;
  int sm_count = 0;
  std::size_t optin_shared_bytes_per_block = 0U;
  bool supported = false;
};

struct Sm87NvFp4PersistentPrefillResources {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

// CUDA functions return cudaError_t encoded as int.  Disabled admission,
// unsupported M, a non-SM87/16-SM device, or a resource mismatch returns
// cudaErrorNotSupported.  Pointer and alignment violations return
// cudaErrorInvalidValue.  GateUp requires the load-time interleaved sidecar.
int query_sm87_nvfp4_persistent_prefill_capability_cuda(
    Sm87NvFp4PersistentPrefillRole role, std::size_t token_count,
    Sm87NvFp4PersistentPrefillCapability* capability) noexcept;

int query_sm87_nvfp4_persistent_prefill_resources_cuda(
    Sm87NvFp4PersistentPrefillRole role, std::size_t token_count,
    Sm87NvFp4PersistentPrefillResources* resources) noexcept;

int launch_sm87_nvfp4_persistent_prefill_gate_up_cuda(
    const std::uint16_t* input,
    const std::uint8_t* interleaved_marlin_weight,
    const std::uint8_t* interleaved_marlin_scales,
    const float* marlin_global_scale, std::size_t token_count,
    std::uint16_t* activated_output, void* cuda_stream) noexcept;

// residual_in_out contains the BF16 residual on entry.  The Marlin epilogue
// first publishes the globally-scaled branch to BF16, adds that rounded value
// to the BF16 residual in FP32, then BF16-RNE publishes in place.
int launch_sm87_nvfp4_persistent_prefill_down_residual_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales,
    const float* marlin_global_scale, std::size_t token_count,
    std::uint16_t* residual_in_out, void* cuda_stream) noexcept;

}  // namespace q3x::kernels
