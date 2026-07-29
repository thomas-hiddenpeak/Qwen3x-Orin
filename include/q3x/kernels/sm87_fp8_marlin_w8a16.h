#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Test-admission-only direct port of the BF16 x E4M3FN W8A16 path selected by
// vLLM ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb.  This is intentionally a
// Qwen3.6-27B/SM87 surface, not a generic Marlin API.  Production dispatch
// remains on the existing FP8 supermatrix implementation until real-checkpoint
// end-to-end admission is complete.
inline constexpr std::size_t kSm87Fp8MarlinSmCount = 16U;
inline constexpr std::size_t kSm87Fp8MarlinC32Tokens = 32U;
inline constexpr std::size_t kSm87Fp8MarlinC64Tokens = 64U;
inline constexpr std::size_t kSm87Fp8MarlinC256Tokens = 256U;
inline constexpr std::size_t kSm87Fp8MarlinC512Tokens = 512U;
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
  return token_count == kSm87Fp8MarlinC32Tokens ||
         token_count == kSm87Fp8MarlinC64Tokens ||
         token_count == kSm87Fp8MarlinC256Tokens ||
         token_count == kSm87Fp8MarlinC512Tokens;
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

// Exact frozen-vLLM auto-selector result.  N=1024 K/V uses the low-work N64
// override at C32/C64; every other Qwen projection, and K/V at C256/C512,
// uses the large-batch M64N256K64 skeleton.  All variants are group_blocks=-1
// (one BF16 channel scale per N), four pipeline stages, and a 16-CTA grid.
[[nodiscard]] constexpr Sm87Fp8MarlinTileConfig
sm87_fp8_marlin_tile_config(const std::size_t token_count,
                            const std::size_t output_size) noexcept {
  const bool supported_output =
      output_size == 10'240U || output_size == 6'144U ||
      output_size == 5'120U || output_size == 12'288U ||
      output_size == 1'024U;
  if (!sm87_fp8_marlin_supports_token_count(token_count) ||
      !supported_output) {
    return {};
  }
  if (output_size == 1'024U && token_count == kSm87Fp8MarlinC32Tokens) {
    return {32U, 64U, 128U, 128U, 4U, 16U};
  }
  if (output_size == 1'024U && token_count == kSm87Fp8MarlinC64Tokens) {
    return {64U, 64U, 128U, 128U, 4U, 16U};
  }
  if (token_count == kSm87Fp8MarlinC32Tokens) {
    return {32U, 256U, 64U, 256U, 4U, 16U};
  }
  return {64U, 256U, 64U, 256U, 4U, 16U};
}

static_assert(sm87_fp8_marlin_shape(
                  Sm87Fp8MarlinProjection::kLinearQkv)
                      .output_size == 10'240U);
static_assert(sm87_fp8_marlin_tile_config(32U, 10'240U).thread_n == 256U);
static_assert(sm87_fp8_marlin_tile_config(32U, 1'024U).thread_k == 128U);
static_assert(sm87_fp8_marlin_tile_config(64U, 1'024U).threads == 128U);
static_assert(sm87_fp8_marlin_tile_config(256U, 1'024U).thread_n == 256U);
static_assert(sm87_fp8_marlin_tile_config(512U, 5'120U).thread_m == 64U);
static_assert(!sm87_fp8_marlin_tile_config(128U, 5'120U).valid());
static_assert(!sm87_fp8_marlin_tile_config(32U, 2'048U).valid());

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
