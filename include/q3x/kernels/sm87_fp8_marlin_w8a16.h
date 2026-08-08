#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Test-admission-only direct port of the BF16 x E4M3FN W8A16 path selected by
// vLLM ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb. This is intentionally a
// Qwen3.6-27B/SM87 surface, not a generic Marlin API. Prefill M=2..8192 is one
// operator panel. It is not one underlying launch: the common fixed-shape
// wrapper follows the frozen vLLM boundary of at most 16 M64 panels (M1024)
// per ordered launch when N>4096, and 128 M64 panels (M8192) when N<=4096.
// Large non-M64-aligned panels end with one masked tail launch because upstream
// Marlin intentionally truncates prob_m to floor(prob_m/64) parallel panels.
inline constexpr std::size_t kSm87Fp8MarlinSmCount = 16U;
inline constexpr std::size_t kSm87Fp8MarlinC32Tokens = 32U;
inline constexpr std::size_t kSm87Fp8MarlinC64Tokens = 64U;
inline constexpr std::size_t kSm87Fp8MarlinC256Tokens = 256U;
inline constexpr std::size_t kSm87Fp8MarlinC512Tokens = 512U;
inline constexpr std::size_t kSm87Fp8MarlinMaximumOperatorPanelTokens =
    8'192U;
inline constexpr std::size_t kSm87Fp8MarlinLargeNMaximumKernelSegmentTokens =
    1'024U;
inline constexpr std::size_t kSm87Fp8MarlinSmallNMaximumKernelSegmentTokens =
    8'192U;
// No-shape planning uses the restrictive N>4096 boundary. Launch planning is
// shape-aware and admits M8192 only for the supported N1024 K/V projection.
inline constexpr std::size_t kSm87Fp8MarlinMaximumKernelSegmentTokens =
    kSm87Fp8MarlinLargeNMaximumKernelSegmentTokens;
inline constexpr std::size_t kSm87Fp8MarlinPipelineStages = 4U;
inline constexpr std::size_t kSm87Fp8MarlinDynamicSharedBytes = 166'912U;
inline constexpr std::size_t kSm87Fp8MarlinMaximumThreadM = 64U;
inline constexpr std::size_t kSm87Fp8MarlinMaximumThreadN = 256U;
inline constexpr std::size_t kSm87Fp8MarlinLockBytes =
    kSm87Fp8MarlinSmCount * sizeof(std::int32_t);
inline constexpr std::size_t kSm87Fp8MarlinReductionElements =
    kSm87Fp8MarlinSmCount * kSm87Fp8MarlinMaximumThreadM *
    kSm87Fp8MarlinMaximumThreadN;
inline constexpr std::size_t kSm87Fp8MarlinReductionBytes =
    kSm87Fp8MarlinReductionElements * sizeof(float);

static_assert(kSm87Fp8MarlinMaximumKernelSegmentTokens %
                  kSm87Fp8MarlinC64Tokens ==
              0U);
static_assert(kSm87Fp8MarlinMaximumKernelSegmentTokens /
                  kSm87Fp8MarlinC64Tokens ==
              kSm87Fp8MarlinSmCount);
static_assert(kSm87Fp8MarlinMaximumOperatorPanelTokens %
                  kSm87Fp8MarlinMaximumKernelSegmentTokens ==
              0U);
static_assert(kSm87Fp8MarlinSmallNMaximumKernelSegmentTokens /
                  kSm87Fp8MarlinC64Tokens ==
              128U);
static_assert(kSm87Fp8MarlinC512Tokens == 512U);

enum class Sm87Fp8MarlinProjection : std::uint8_t {
  kLinearQkv,
  kLinearZ,
  kAttentionOutput,
  kFullQuery,
  kFullKeyOrValue,
};

struct Sm87Fp8MarlinShape {
  std::size_t output_size = 0U;
  std::size_t input_size = 0U;
};

[[nodiscard]] constexpr Sm87Fp8MarlinShape sm87_fp8_marlin_shape(
    const Sm87Fp8MarlinProjection projection) noexcept {
  switch (projection) {
    case Sm87Fp8MarlinProjection::kLinearQkv:
      return {10'240U, 5'120U};
    case Sm87Fp8MarlinProjection::kLinearZ:
      return {6'144U, 5'120U};
    case Sm87Fp8MarlinProjection::kAttentionOutput:
      return {5'120U, 6'144U};
    case Sm87Fp8MarlinProjection::kFullQuery:
      return {12'288U, 5'120U};
    case Sm87Fp8MarlinProjection::kFullKeyOrValue:
      return {1'024U, 5'120U};
  }
  return {};
}

[[nodiscard]] constexpr bool sm87_fp8_marlin_supports_shape(
    const std::size_t output_size, const std::size_t input_size) noexcept {
  return (output_size == 10'240U && input_size == 5'120U) ||
         (output_size == 6'144U && input_size == 5'120U) ||
         (output_size == 5'120U && input_size == 6'144U) ||
         (output_size == 12'288U && input_size == 5'120U) ||
         (output_size == 1'024U && input_size == 5'120U);
}

[[nodiscard]] constexpr bool sm87_fp8_marlin_supports_token_count(
    const std::size_t token_count) noexcept {
  return token_count >= 2U && token_count <= kSm87Fp8MarlinC512Tokens;
}

// Candidate-only panel capability. The current production C512 admission
// continues to call the compatibility predicate above until a complete
// layer-major route is explicitly connected and attested.
[[nodiscard]] constexpr bool
sm87_fp8_marlin_supports_operator_panel_token_count(
    const std::size_t token_count) noexcept {
  return token_count >= 2U &&
         token_count <= kSm87Fp8MarlinMaximumOperatorPanelTokens;
}

struct Sm87Fp8MarlinExecutionPlan {
  // Total prefix covered by one or more M64-aligned launches. For M<=64 this
  // is the single masked launch and remainder_tokens is zero.
  std::size_t primary_tokens = 0U;
  // Final masked launch after the aligned prefix, always in [1,63].
  std::size_t remainder_tokens = 0U;
  // Number of exact maximum_segment_tokens launches at the panel start.
  std::size_t complete_segment_count = 0U;
  // Optional final aligned launch below that maximum, or the sole M<=64.
  std::size_t final_primary_segment_tokens = 0U;
  std::size_t launch_count = 0U;
  // Frozen host-wrapper boundary used to build this shape-specific plan.
  std::size_t maximum_segment_tokens = 0U;
};

struct Sm87Fp8MarlinLaunchSegment {
  std::size_t token_offset = 0U;
  std::size_t token_count = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return token_count != 0U;
  }
};

[[nodiscard]] constexpr std::size_t
sm87_fp8_marlin_maximum_kernel_segment_tokens(
    const std::size_t output_size) noexcept {
  if (output_size == 1'024U) {
    return kSm87Fp8MarlinSmallNMaximumKernelSegmentTokens;
  }
  return output_size == 10'240U || output_size == 6'144U ||
                 output_size == 5'120U || output_size == 12'288U
             ? kSm87Fp8MarlinLargeNMaximumKernelSegmentTokens
             : 0U;
}

[[nodiscard]] constexpr Sm87Fp8MarlinExecutionPlan
sm87_fp8_marlin_execution_plan_with_segment_limit(
    const std::size_t token_count,
    const std::size_t maximum_segment_tokens) noexcept {
  if (!sm87_fp8_marlin_supports_operator_panel_token_count(token_count)) {
    return {};
  }
  if ((maximum_segment_tokens !=
           kSm87Fp8MarlinLargeNMaximumKernelSegmentTokens &&
       maximum_segment_tokens !=
           kSm87Fp8MarlinSmallNMaximumKernelSegmentTokens) ||
      maximum_segment_tokens % kSm87Fp8MarlinC64Tokens != 0U) {
    return {};
  }
  if (token_count <= kSm87Fp8MarlinC64Tokens) {
    return {token_count, 0U, 0U, token_count, 1U,
            maximum_segment_tokens};
  }
  const std::size_t primary =
      token_count - token_count % kSm87Fp8MarlinC64Tokens;
  const std::size_t complete_segments =
      primary / maximum_segment_tokens;
  const std::size_t final_primary =
      primary % maximum_segment_tokens;
  const std::size_t remainder = token_count - primary;
  return {primary, remainder, complete_segments, final_primary,
          complete_segments + (final_primary != 0U ? 1U : 0U) +
              (remainder != 0U ? 1U : 0U),
          maximum_segment_tokens};
}

[[nodiscard]] constexpr Sm87Fp8MarlinExecutionPlan
sm87_fp8_marlin_execution_plan(const std::size_t token_count,
                               const std::size_t output_size) noexcept {
  return sm87_fp8_marlin_execution_plan_with_segment_limit(
      token_count,
      sm87_fp8_marlin_maximum_kernel_segment_tokens(output_size));
}

// Compatibility overload for callers that do not yet carry the projection
// shape. It deliberately uses the restrictive N>4096 boundary.
[[nodiscard]] constexpr Sm87Fp8MarlinExecutionPlan
sm87_fp8_marlin_execution_plan(const std::size_t token_count) noexcept {
  return sm87_fp8_marlin_execution_plan_with_segment_limit(
      token_count, kSm87Fp8MarlinLargeNMaximumKernelSegmentTokens);
}

[[nodiscard]] constexpr Sm87Fp8MarlinLaunchSegment
sm87_fp8_marlin_launch_segment(
    const Sm87Fp8MarlinExecutionPlan& plan,
    const std::size_t launch_index) noexcept {
  if (launch_index >= plan.launch_count) {
    return {};
  }
  if (launch_index < plan.complete_segment_count) {
    return {launch_index * plan.maximum_segment_tokens,
            plan.maximum_segment_tokens};
  }
  std::size_t next_index = plan.complete_segment_count;
  const std::size_t final_primary_offset =
      plan.complete_segment_count * plan.maximum_segment_tokens;
  if (plan.final_primary_segment_tokens != 0U) {
    if (launch_index == next_index) {
      return {final_primary_offset, plan.final_primary_segment_tokens};
    }
    ++next_index;
  }
  if (plan.remainder_tokens != 0U && launch_index == next_index) {
    return {plan.primary_tokens, plan.remainder_tokens};
  }
  return {};
}

struct Sm87Fp8MarlinTileConfig {
  std::size_t thread_m = 0U;
  std::size_t thread_n = 0U;
  std::size_t thread_k = 0U;
  std::size_t threads = 0U;
  std::size_t stages = 0U;
  std::size_t persistent_ctas = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return thread_m != 0U;
  }
};

// Frozen-vLLM selector family. N=1024 K/V uses its N64/K128 low-work override
// for M17..64. Every other Qwen projection uses N128/K128 through M16 and
// N256/K64 above it. M33..48 uses vLLM's ceil(M/16)=3 ThreadMBlocks member,
// while M49..64 uses four blocks. All
// variants use one BF16 channel scale per N, four stages, and 16 CTAs.
[[nodiscard]] constexpr Sm87Fp8MarlinTileConfig
sm87_fp8_marlin_tile_config(const std::size_t token_count,
                            const std::size_t output_size) noexcept {
  const bool supported_output =
      output_size == 10'240U || output_size == 6'144U ||
      output_size == 5'120U || output_size == 12'288U ||
      output_size == 1'024U;
  if (!sm87_fp8_marlin_supports_operator_panel_token_count(token_count) ||
      !supported_output) {
    return {};
  }
  if (token_count <= 8U) {
    return {8U, 128U, 128U, 256U, 4U, 16U};
  }
  if (token_count <= 16U) {
    return {16U, 128U, 128U, 256U, 4U, 16U};
  }
  if (output_size == 1'024U && token_count <= kSm87Fp8MarlinC32Tokens) {
    return {32U, 64U, 128U, 128U, 4U, 16U};
  }
  if (output_size == 1'024U &&
      token_count <= kSm87Fp8MarlinC64Tokens) {
    return {token_count <= 48U ? 48U : 64U, 64U, 128U, 128U, 4U, 16U};
  }
  if (token_count <= kSm87Fp8MarlinC32Tokens) {
    return {32U, 256U, 64U, 256U, 4U, 16U};
  }
  return {token_count <= 48U ? 48U : 64U, 256U, 64U, 256U, 4U, 16U};
}

static_assert(sm87_fp8_marlin_shape(
                  Sm87Fp8MarlinProjection::kLinearQkv)
                      .output_size == 10'240U);
static_assert(sm87_fp8_marlin_tile_config(32U, 10'240U).thread_n == 256U);
static_assert(sm87_fp8_marlin_tile_config(32U, 1'024U).thread_k == 128U);
static_assert(sm87_fp8_marlin_tile_config(64U, 1'024U).threads == 128U);
static_assert(sm87_fp8_marlin_tile_config(256U, 1'024U).thread_n == 256U);
static_assert(sm87_fp8_marlin_tile_config(512U, 5'120U).thread_m == 64U);
static_assert(sm87_fp8_marlin_tile_config(128U, 5'120U).valid());
static_assert(sm87_fp8_marlin_tile_config(7U, 10'240U).thread_m == 8U);
static_assert(sm87_fp8_marlin_tile_config(15U, 1'024U).thread_n == 128U);
static_assert(sm87_fp8_marlin_tile_config(33U, 1'024U).thread_n == 64U);
static_assert(sm87_fp8_marlin_tile_config(33U, 1'024U).thread_m == 48U);
static_assert(sm87_fp8_marlin_tile_config(33U, 10'240U).thread_m == 48U);
static_assert(sm87_fp8_marlin_execution_plan(407U).primary_tokens == 384U);
static_assert(sm87_fp8_marlin_execution_plan(407U).remainder_tokens == 23U);
static_assert(sm87_fp8_marlin_execution_plan(481U).primary_tokens == 448U);
static_assert(sm87_fp8_marlin_execution_plan(481U).remainder_tokens == 33U);
static_assert(sm87_fp8_marlin_execution_plan(512U).launch_count == 1U);
static_assert(sm87_fp8_marlin_execution_plan(513U).launch_count == 2U);
static_assert(
    sm87_fp8_marlin_launch_segment(sm87_fp8_marlin_execution_plan(513U), 0U)
        .token_count == 512U);
static_assert(
    sm87_fp8_marlin_launch_segment(sm87_fp8_marlin_execution_plan(513U), 1U)
        .token_offset == 512U);
static_assert(sm87_fp8_marlin_execution_plan(8'192U).launch_count == 8U);
static_assert(
    sm87_fp8_marlin_launch_segment(sm87_fp8_marlin_execution_plan(8'192U), 7U)
        .token_offset == 7'168U);
static_assert(sm87_fp8_marlin_execution_plan(8'191U).launch_count == 9U);
static_assert(
    sm87_fp8_marlin_launch_segment(sm87_fp8_marlin_execution_plan(8'191U), 8U)
        .token_count == 63U);
static_assert(sm87_fp8_marlin_tile_config(513U, 10'240U).thread_m == 64U);
static_assert(sm87_fp8_marlin_tile_config(8'192U, 5'120U).thread_m == 64U);
static_assert(sm87_fp8_marlin_execution_plan(8'192U, 1'024U)
                  .launch_count == 1U);
static_assert(
    sm87_fp8_marlin_launch_segment(
        sm87_fp8_marlin_execution_plan(8'192U, 1'024U),
        0U)
        .token_count == 8'192U);
static_assert(sm87_fp8_marlin_execution_plan(8'191U, 1'024U)
                  .launch_count == 2U);
static_assert(sm87_fp8_marlin_execution_plan(8'193U, 1'024U)
                  .launch_count == 0U);
static_assert(!sm87_fp8_marlin_tile_config(32U, 2'048U).valid());
static_assert(!sm87_fp8_marlin_tile_config(8'193U, 1'024U).valid());
static_assert(!sm87_fp8_marlin_supports_token_count(513U));
static_assert(sm87_fp8_marlin_supports_operator_panel_token_count(513U));

[[nodiscard]] constexpr std::size_t sm87_fp8_marlin_weight_bytes(
    const std::size_t output_size, const std::size_t input_size) noexcept {
  return output_size * input_size;
}

[[nodiscard]] constexpr std::size_t sm87_fp8_marlin_scale_bytes(
    const std::size_t output_size) noexcept {
  return output_size * sizeof(std::uint16_t);
}

// Load-time transformation of canonical checkpoint E4M3FN [N,K] bytes into
// the exact vLLM GPTQ-Marlin W8 layout.  transpose_scratch is disposable and
// exactly N*K bytes.  canonical_weight_scale_device points to the checkpoint's
// scalar F32 scale; marlin_scales receives vLLM's N channel-wise BF16 scales
// with the E4M3->BF16 exponent-bias delta (2^120) fused into them.
[[nodiscard]] int prepare_sm87_fp8_marlin_projection_cuda(
    const std::uint8_t* canonical_weight,
    const float* canonical_weight_scale_device, std::size_t output_size,
    std::size_t input_size, std::uint8_t* marlin_weight,
    std::uint16_t* marlin_scales, void* transpose_scratch,
    std::size_t transpose_scratch_bytes,
    void* cuda_stream = nullptr) noexcept;

// Locks must be zero-initialized before first use.  The vendored stripe
// scheduler restores them to zero on an ordered stream.  c_tmp is the fixed
// FP32 cross-CTA reduction arena described by kSm87Fp8MarlinReductionBytes.
[[nodiscard]] int launch_sm87_fp8_marlin_projection_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint16_t* marlin_scales, std::size_t token_count,
    std::size_t output_size, std::size_t input_size, std::uint16_t* output,
    float* c_tmp, std::int32_t* locks,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
