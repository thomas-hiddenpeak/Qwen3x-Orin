#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off, BUILD_TESTING-only physical surface for the exact P40000
// AC-PREFILL-PROMPT-WIDE-v2 BF16 A/B slice.  It reuses the established
// M64xN96xK64 arithmetic kernel but submits all 625 M64 tiles in one CUDA
// grid.  No smaller/larger prompt, tail, compatibility path, or production
// selector is implied by this ABI.
inline constexpr std::size_t kSm87Bf16AbPromptWideP40Tokens = 40'000U;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40InputFeatures = 5'120U;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40RowsPerProjection = 48U;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40ProjectionCount = 2U;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40TileTokens = 64U;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40Threads = 256U;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40GridBlocks = 625U;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40LaunchCount = 1U;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40DynamicSharedBytes =
    46'080U;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40WeightElements =
    kSm87Bf16AbPromptWideP40RowsPerProjection *
    kSm87Bf16AbPromptWideP40InputFeatures;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40InputElements =
    kSm87Bf16AbPromptWideP40Tokens *
    kSm87Bf16AbPromptWideP40InputFeatures;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40OutputElements =
    kSm87Bf16AbPromptWideP40Tokens *
    kSm87Bf16AbPromptWideP40RowsPerProjection;
inline constexpr std::size_t kSm87Bf16AbPromptWideP40MaximumWeightIndex =
    (kSm87Bf16AbPromptWideP40RowsPerProjection - 1U) *
        kSm87Bf16AbPromptWideP40InputFeatures +
    (kSm87Bf16AbPromptWideP40InputFeatures - 1U);
inline constexpr std::size_t kSm87Bf16AbPromptWideP40MaximumInputIndex =
    (kSm87Bf16AbPromptWideP40Tokens - 1U) *
        kSm87Bf16AbPromptWideP40InputFeatures +
    (kSm87Bf16AbPromptWideP40InputFeatures - 1U);
inline constexpr std::size_t kSm87Bf16AbPromptWideP40MaximumOutputIndex =
    (kSm87Bf16AbPromptWideP40Tokens - 1U) *
        kSm87Bf16AbPromptWideP40RowsPerProjection +
    (kSm87Bf16AbPromptWideP40RowsPerProjection - 1U);

static_assert(kSm87Bf16AbPromptWideP40Tokens %
                      kSm87Bf16AbPromptWideP40TileTokens ==
                  0U,
              "P40000 must be one exact M64 grid");
static_assert(kSm87Bf16AbPromptWideP40GridBlocks *
                      kSm87Bf16AbPromptWideP40TileTokens ==
                  kSm87Bf16AbPromptWideP40Tokens,
              "the one-grid contract must cover every P40000 token once");
static_assert(kSm87Bf16AbPromptWideP40WeightElements == 245'760U);
static_assert(kSm87Bf16AbPromptWideP40InputElements == 204'800'000U);
static_assert(kSm87Bf16AbPromptWideP40OutputElements == 1'920'000U);
static_assert(kSm87Bf16AbPromptWideP40MaximumWeightIndex + 1U ==
              kSm87Bf16AbPromptWideP40WeightElements);
static_assert(kSm87Bf16AbPromptWideP40MaximumInputIndex + 1U ==
              kSm87Bf16AbPromptWideP40InputElements);
static_assert(kSm87Bf16AbPromptWideP40MaximumOutputIndex + 1U ==
              kSm87Bf16AbPromptWideP40OutputElements);

struct Sm87Bf16AbPromptWideP40Plan final {
  std::size_t requested_token_count = 0U;
  std::size_t tile_tokens = 0U;
  std::size_t grid_blocks = 0U;
  std::size_t threads = 0U;
  std::size_t launch_count = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t input_elements = 0U;
  std::size_t weight_elements_per_projection = 0U;
  std::size_t output_elements_per_projection = 0U;
  bool admitted = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return admitted &&
           requested_token_count == kSm87Bf16AbPromptWideP40Tokens &&
           tile_tokens == kSm87Bf16AbPromptWideP40TileTokens &&
           grid_blocks == kSm87Bf16AbPromptWideP40GridBlocks &&
           grid_blocks * tile_tokens == requested_token_count &&
           threads == kSm87Bf16AbPromptWideP40Threads &&
           launch_count == kSm87Bf16AbPromptWideP40LaunchCount &&
           dynamic_shared_bytes ==
               kSm87Bf16AbPromptWideP40DynamicSharedBytes &&
           input_elements == kSm87Bf16AbPromptWideP40InputElements &&
           weight_elements_per_projection ==
               kSm87Bf16AbPromptWideP40WeightElements &&
           output_elements_per_projection ==
               kSm87Bf16AbPromptWideP40OutputElements;
  }
};

[[nodiscard]] constexpr Sm87Bf16AbPromptWideP40Plan
make_sm87_bf16_ab_prompt_wide_p40_plan(
    const std::size_t token_count) noexcept {
  if (token_count != kSm87Bf16AbPromptWideP40Tokens) {
    return {token_count};
  }
  return {
      token_count,
      kSm87Bf16AbPromptWideP40TileTokens,
      kSm87Bf16AbPromptWideP40GridBlocks,
      kSm87Bf16AbPromptWideP40Threads,
      kSm87Bf16AbPromptWideP40LaunchCount,
      kSm87Bf16AbPromptWideP40DynamicSharedBytes,
      kSm87Bf16AbPromptWideP40InputElements,
      kSm87Bf16AbPromptWideP40WeightElements,
      kSm87Bf16AbPromptWideP40OutputElements,
      true,
  };
}

struct Sm87Bf16AbPromptWideP40Tile final {
  std::size_t block = 0U;
  std::size_t first_token = 0U;
  std::size_t token_count = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87Bf16AbPromptWideP40Tile
sm87_bf16_ab_prompt_wide_p40_tile(
    const Sm87Bf16AbPromptWideP40Plan& plan,
    const std::size_t block) noexcept {
  if (!plan.valid() || block >= plan.grid_blocks) {
    return {};
  }
  return {
      block,
      block * plan.tile_tokens,
      plan.tile_tokens,
      true,
  };
}

struct Sm87Bf16AbPromptWideP40Resources final {
  int compute_major = 0;
  int compute_minor = 0;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int active_blocks_per_sm = 0;
  bool admitted = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return admitted && compute_major == 8 && compute_minor == 7 &&
           binary_version == 87 && registers_per_thread > 0 &&
           dynamic_shared_bytes ==
               kSm87Bf16AbPromptWideP40DynamicSharedBytes &&
           active_blocks_per_sm >= 2;
  }
};

// Returns cudaErrorNotSupported unless the active device is exactly SM87,
// the installed cubin contains an SM87 body, the function accepts the exact
// block/shared-memory launch, and at least two CTAs can reside per SM.  The
// result is intended to be sealed at engine readiness before this default-off
// surface is connected to a runner tactic.
[[nodiscard]] int query_sm87_bf16_ab_prompt_wide_p40_resources_cuda(
    Sm87Bf16AbPromptWideP40Resources* resources) noexcept;

// Enqueues exactly one grid: dim3(625,1,1) x dim3(256,1,1), with one CTA per
// disjoint M64 token interval.  The two canonical BF16 [48,5120] weights and
// token-major BF16 [40000,5120] input produce independent token-major
// [40000,48] outputs.  Read-only payloads require 16-byte alignment; outputs
// require 4-byte alignment and may not overlap any input/weight/output span.
// token_count must be exactly 40000.  There is no tail or fallback launch.
[[nodiscard]] int launch_sm87_bf16_ab_prompt_wide_p40_cuda(
    const std::uint16_t* first_weights,
    const std::uint16_t* second_weights,
    const std::uint16_t* input, std::size_t token_count,
    std::uint16_t* first_output, std::uint16_t* second_output,
    void* cuda_stream = nullptr) noexcept;

// Test-admission-only Qwen3.6 linear-attention A/B projection. The two
// canonical BF16 [48, 5120] row-major matrices are treated as one logical
// N96 matrix, while the public outputs retain their independent token-major
// [M, 48] layouts. M is accepted in [2, 512]. Complete M64 spans use the
// SM87 BF16 Tensor Core pipeline; a final M1..M63 span preserves the existing
// M16/generic BF16 projection pair boundary. No persistent weight transform,
// scratch allocation, cuBLAS, or cuBLASLt dependency is introduced.
[[nodiscard]] int launch_sm87_bf16_ab_large_m_prefill_cuda(
    const std::uint16_t* first_weights,
    const std::uint16_t* second_weights,
    const std::uint16_t* input, std::size_t token_count,
    std::uint16_t* first_output, std::uint16_t* second_output,
    void* cuda_stream = nullptr) noexcept;

// Resource probe for the M64xN96xK64 Tensor Core kernel. The occupancy result
// includes the launcher's two-stage dynamic shared-memory footprint.
[[nodiscard]] int query_sm87_bf16_ab_large_m_prefill_resources_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* active_blocks_per_sm) noexcept;

}  // namespace q3x::kernels
