#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_PREFILL_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_PREFILL_HOST_DEVICE
#endif

namespace q3x::kernels {

// Experimental full-tile A4W4 Prefill GEMM.  This is the first complete
// consumer of the native SM87 m16n8k64 primitive; it deliberately owns only
// the integer projection boundary and does not select itself in the runner.
//
// A and B use the signed-nibble consumer-block ABI shared with the calibrated
// full-model sidecar:
//
//   A: [ceil(M/64), K/64, 64, 32] bytes
//   B: [N/64, K/64, 64, 32] bytes
//   A scales: BF16 [ceil(M/64), K/64, 64]
//   B scales: BF16 [N/64, K/64, 64]
//   D: BF16 [M, N]
//
// M64/N64 are physical blocks only. The CTA combines one M32 half-block and
// two adjacent N64 blocks into M32N128, reducing activation rescans while
// retaining the two-CTA/SM resource contract. Shared packed rows use the
// primitive's XOR half-row swizzle; global consumer blocks stay contiguous.
//
// Every K64 integer partial is dequantized before it is accumulated in FP32;
// scales from distinct groups are never incorrectly factored across K.
inline constexpr std::size_t kSm87A4W4PrefillTileM = 32U;
inline constexpr std::size_t kSm87A4W4PrefillTileN = 128U;
inline constexpr std::size_t kSm87A4W4PrefillTileK = 64U;
inline constexpr std::size_t kSm87A4W4PrefillThreads = 256U;
inline constexpr std::size_t kSm87A4W4PrefillWarps = 8U;
inline constexpr std::size_t kSm87A4W4PrefillPipelineStages = 3U;
inline constexpr std::size_t kSm87A4W4PrefillPersistentCtas = 32U;
inline constexpr std::size_t kSm87A4W4PrefillCtasPerSm = 2U;

// Whole-span large-M candidate.  It preserves the public consumer/output ABI
// and the 256-thread persistent launch contract, but assigns the eight warps
// as 4x2 M16N32 warp tiles.  Relative to M32N128, M64N64 halves packed-B and
// B-scale staging per output while accepting the corresponding A rescan.  It
// is selected for every complete large-M M64 prefix. Any residual 1..63 rows
// stay on the established M32N128 kernel in the same public launch.
inline constexpr std::size_t kSm87A4W4PrefillLargeMTileM = 64U;
inline constexpr std::size_t kSm87A4W4PrefillLargeMTileN = 64U;
inline constexpr std::size_t kSm87A4W4PrefillLargeMMinimumTokens = 1'024U;

// Whole-M wide-N candidate.  Each CTA owns M64N256 and assigns the eight
// warps as 2x4 M32N64 warp tiles.  Two adjacent K64 groups share one logical
// pipeline buffer without changing their accumulation order.  The candidate
// is admitted for a complete M64/N256 prefix at P>=1536. P1024 remains on
// M64N64, while the natural P2K bucket no longer falls back merely because
// its aligned prefix is slightly below 2048. Any residual 1..63 rows remain
// on M32N128.
inline constexpr std::size_t kSm87A4W4PrefillWideTileM = 64U;
inline constexpr std::size_t kSm87A4W4PrefillWideTileN = 256U;
inline constexpr std::size_t kSm87A4W4PrefillWideMinimumTokens = 1'536U;
inline constexpr std::size_t kSm87A4W4PrefillWideLogicalTileK = 128U;
inline constexpr std::size_t kSm87A4W4PrefillWidePipelineStages = 2U;

// Independent K128 shared-scale vertical slice.  Codes remain in two
// adjacent physical K64 consumer blocks, while A/B expose one BF16 scale per
// logical K128 group.  This surface is intentionally not selected by the
// K64 launcher or the runner.
inline constexpr std::size_t kSm87A4W4PrefillK128TileM = 64U;
inline constexpr std::size_t kSm87A4W4PrefillK128TileN = 256U;
inline constexpr std::size_t kSm87A4W4PrefillK128TileK = 128U;
inline constexpr std::size_t kSm87A4W4PrefillK128PipelineSlots = 2U;
inline constexpr std::size_t kSm87A4W4PrefillK128MaximumRegisters = 128U;

// Independent Attention-O activation ABI.  Logical rows are rounded to an
// exact M128 launch surface before quantization so the K512 consumer never
// observes an implicit tail.  Codes retain the physical K64 transport layout
// while one BF16 scale covers each logical K512 group.
inline constexpr std::size_t kSm87A4W4PrefillK512ScaleK = 512U;
inline constexpr std::size_t kSm87A4W4PrefillK512LaunchMAlignment = 128U;
inline constexpr std::size_t
    kSm87A4W4PrefillK512PhysicalK64BlocksPerScale = 8U;

[[nodiscard]] constexpr bool sm87_a4w4_prefill_uses_large_m_candidate(
    const std::size_t token_count) noexcept {
  return token_count >= kSm87A4W4PrefillLargeMMinimumTokens &&
         token_count % kSm87A4W4PrefillLargeMTileM == 0U;
}

[[nodiscard]] constexpr bool sm87_a4w4_prefill_uses_m64n256_candidate(
    const std::size_t token_count,
    const std::size_t output_size) noexcept {
  return token_count >= kSm87A4W4PrefillWideMinimumTokens &&
         token_count % kSm87A4W4PrefillWideTileM == 0U &&
         output_size != 0U &&
         output_size % kSm87A4W4PrefillWideTileN == 0U;
}

struct Sm87A4W4PrefillGemmPlan final {
  std::size_t token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
};

enum class Sm87A4W4PrefillK64CompositePrefixKernel : std::uint8_t {
  kNone = 0U,
  kM64N64K64,
  kM64N256K64,
};

// One public K64 call may compose a complete M64 prefix with one canonical
// M32 fallback launch for the final partial consumer block. prefix_plan and
// tail_plan retain the exact launch cardinalities used by each existing
// kernel; no K128 layout is represented by this plan.
struct Sm87A4W4PrefillK64CompositePlan final {
  bool valid{};
  std::size_t token_count{};
  std::size_t prefix_token_count{};
  std::size_t tail_token_count{};
  Sm87A4W4PrefillK64CompositePrefixKernel prefix_kernel{
      Sm87A4W4PrefillK64CompositePrefixKernel::kNone};
  Sm87A4W4PrefillGemmPlan prefix_plan{};
  Sm87A4W4PrefillGemmPlan tail_plan{};
};

struct Sm87A4W4PrefillK128GemmPlan final {
  std::size_t token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k128_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr std::size_t sm87_a4w4_ceil_div(
    const std::size_t numerator, const std::size_t denominator) noexcept {
  return denominator == 0U || numerator == 0U
             ? 0U
             : 1U + (numerator - 1U) / denominator;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_prefill_k512_launch_token_count(
    const std::size_t logical_token_count) noexcept {
  const std::size_t tiles = sm87_a4w4_ceil_div(
      logical_token_count, kSm87A4W4PrefillK512LaunchMAlignment);
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  return tiles == 0U ||
                 tiles > maximum / kSm87A4W4PrefillK512LaunchMAlignment
             ? 0U
             : tiles * kSm87A4W4PrefillK512LaunchMAlignment;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_prefill_k512_group_count(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4PrefillK512ScaleK == 0U
             ? logical_k / kSm87A4W4PrefillK512ScaleK
             : 0U;
}

// Scale capacity follows [ceil(launch_M/64), K/512, 64].  The launcher
// requires launch_M to equal ceil128(logical_M), but keeping this helper
// overflow-safe makes the public buffer contract independently auditable.
[[nodiscard]] constexpr std::size_t
sm87_a4w4_prefill_k512_scale_capacity_elements(
    const std::size_t launch_token_count,
    const std::size_t logical_k) noexcept {
  const std::size_t outer_blocks =
      sm87_a4w4_ceil_div(launch_token_count, 64U);
  const std::size_t groups =
      sm87_a4w4_prefill_k512_group_count(logical_k);
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  if (outer_blocks == 0U || groups == 0U ||
      outer_blocks > maximum / groups) {
    return 0U;
  }
  const std::size_t block_groups = outer_blocks * groups;
  return block_groups > maximum / 64U ? 0U : block_groups * 64U;
}

[[nodiscard]] Q3X_SM87_A4W4_PREFILL_HOST_DEVICE constexpr std::size_t
sm87_a4w4_prefill_k512_scale_offset(
    const std::size_t outer_coordinate,
    const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  return ((outer_coordinate / 64U) * k512_group_count + k512_group) * 64U +
         outer_coordinate % 64U;
}

[[nodiscard]] constexpr Sm87A4W4PrefillGemmPlan
sm87_a4w4_prefill_gemm_plan(const std::size_t token_count,
                            const std::size_t output_size,
                            const std::size_t input_size) noexcept {
  if (token_count == 0U || output_size == 0U || input_size == 0U ||
      output_size % kSm87A4W4PrefillTileN != 0U ||
      input_size % kSm87A4W4PrefillTileK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      sm87_a4w4_ceil_div(token_count, kSm87A4W4PrefillTileM);
  const std::size_t n_tiles = output_size / kSm87A4W4PrefillTileN;
  if (m_tiles == 0U || n_tiles == 0U ||
      m_tiles > static_cast<std::size_t>(-1) / n_tiles) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4PrefillTileK,
          work_tiles,
          work_tiles < kSm87A4W4PrefillPersistentCtas
              ? work_tiles
              : kSm87A4W4PrefillPersistentCtas};
}

// The wide tile changes the number and ordering of persistent work items, so
// it deliberately owns a separate planner instead of reinterpreting the
// baseline M32N128 plan in the launcher.
[[nodiscard]] constexpr Sm87A4W4PrefillGemmPlan
sm87_a4w4_prefill_gemm_m64n256_plan(
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (!sm87_a4w4_prefill_uses_m64n256_candidate(token_count, output_size) ||
      input_size == 0U || input_size % kSm87A4W4PrefillTileK != 0U) {
    return {};
  }
  const std::size_t m_tiles = token_count / kSm87A4W4PrefillWideTileM;
  const std::size_t n_tiles = output_size / kSm87A4W4PrefillWideTileN;
  if (m_tiles == 0U || n_tiles == 0U ||
      m_tiles > static_cast<std::size_t>(-1) / n_tiles) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4PrefillTileK,
          work_tiles,
          work_tiles < kSm87A4W4PrefillPersistentCtas
              ? work_tiles
              : kSm87A4W4PrefillPersistentCtas};
}

[[nodiscard]] constexpr Sm87A4W4PrefillK64CompositePlan
sm87_a4w4_prefill_gemm_k64_composite_plan(
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size) noexcept {
  Sm87A4W4PrefillK64CompositePlan result{};
  const Sm87A4W4PrefillGemmPlan full_plan =
      sm87_a4w4_prefill_gemm_plan(token_count, output_size, input_size);
  if (full_plan.launch_ctas == 0U) {
    return result;
  }

  result.valid = true;
  result.token_count = token_count;
  const std::size_t maximum_m64_prefix =
      token_count / kSm87A4W4PrefillLargeMTileM *
      kSm87A4W4PrefillLargeMTileM;
  if (maximum_m64_prefix >= kSm87A4W4PrefillWideMinimumTokens &&
      output_size % kSm87A4W4PrefillWideTileN == 0U) {
    result.prefix_plan = sm87_a4w4_prefill_gemm_m64n256_plan(
        maximum_m64_prefix, output_size, input_size);
    if (result.prefix_plan.launch_ctas != 0U) {
      result.prefix_token_count = maximum_m64_prefix;
      result.tail_token_count = token_count - maximum_m64_prefix;
      result.prefix_kernel =
          Sm87A4W4PrefillK64CompositePrefixKernel::kM64N256K64;
    }
  }
  if (result.prefix_token_count == 0U &&
      maximum_m64_prefix >= kSm87A4W4PrefillLargeMMinimumTokens) {
    result.prefix_plan = sm87_a4w4_prefill_gemm_plan(
        maximum_m64_prefix, output_size, input_size);
    if (result.prefix_plan.launch_ctas != 0U) {
      result.prefix_token_count = maximum_m64_prefix;
      result.tail_token_count = token_count - maximum_m64_prefix;
      result.prefix_kernel =
          Sm87A4W4PrefillK64CompositePrefixKernel::kM64N64K64;
    }
  }
  if (result.prefix_token_count == 0U) {
    result.tail_token_count = token_count;
    result.tail_plan = full_plan;
    return result;
  }
  if (result.tail_token_count != 0U) {
    result.tail_plan = sm87_a4w4_prefill_gemm_plan(
        result.tail_token_count, output_size, input_size);
    if (result.tail_plan.launch_ctas == 0U) {
      return {};
    }
  }
  return result;
}

[[nodiscard]] constexpr Sm87A4W4PrefillK128GemmPlan
sm87_a4w4_prefill_gemm_k128_plan(
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4PrefillK128TileM != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4PrefillK128TileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4PrefillK128TileK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4PrefillK128TileM;
  const std::size_t n_tiles =
      output_size / kSm87A4W4PrefillK128TileN;
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  if (m_tiles == 0U || n_tiles == 0U || m_tiles > maximum / n_tiles) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4PrefillK128TileK,
          input_size / kSm87A4W4PrefillTileK,
          work_tiles,
          work_tiles < kSm87A4W4PrefillPersistentCtas
              ? work_tiles
              : kSm87A4W4PrefillPersistentCtas};
}

struct Sm87A4W4PrefillGemmResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

// Bring-up activation producer.  Eight warps independently quantize eight
// [row,K64] groups per CTA. clip_ratio is calibration-controlled and must be
// in (0,1]. The producer writes the consumer-block layout above with an
// addressable final M64 block; unused tail rows are never consumed. Capacity
// arguments cover that padded representation and are checked before launch.
[[nodiscard]] int launch_sm87_a4_quantize_bf16_cuda(
    const std::uint16_t* input_bf16,
    std::size_t input_row_stride_elements,
    std::size_t token_count,
    std::size_t input_size,
    float clip_ratio,
    std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    std::uint16_t* a_k64_scales_bf16,
    std::size_t a_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// K128 activation producer. One warp quantizes one [row,K128] group using a
// shared BF16 scale, then publishes its 64 packed bytes into the unchanged
// pair of physical K64 consumer blocks.
[[nodiscard]] int launch_sm87_a4_quantize_bf16_k128_cuda(
    const std::uint16_t* input_bf16,
    std::size_t input_row_stride_elements,
    std::size_t token_count,
    std::size_t input_size,
    float clip_ratio,
    std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    std::uint16_t* a_k128_scales_bf16,
    std::size_t a_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// K512 activation producer for the independent Attention-O cell.  The caller
// supplies both the number of real input rows and the exact physical launch
// extent, which must be ceil128(logical_token_count).  Every padded row is
// explicitly published as zero packed codes with a BF16 scale of one; the
// kernel never reads input storage for those rows.
[[nodiscard]] int launch_sm87_a4_quantize_bf16_k512_cuda(
    const std::uint16_t* input_bf16,
    std::size_t input_row_stride_elements,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    std::size_t input_size,
    float clip_ratio,
    std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int query_sm87_a4w4_prefill_gemm_resources_cuda(
    Sm87A4W4PrefillGemmResources* resources) noexcept;

// Queries the independently compiled M64N64 candidate.  Keeping this separate
// from the baseline query makes the >=2 CTA/SM and zero-local-spill admission
// auditable without changing the established resource-query ABI.
[[nodiscard]] int query_sm87_a4w4_prefill_gemm_m64n64_resources_cuda(
    Sm87A4W4PrefillGemmResources* resources) noexcept;

// Queries the independently compiled M64N256/K128-logical candidate.
[[nodiscard]] int query_sm87_a4w4_prefill_gemm_m64n256_resources_cuda(
    Sm87A4W4PrefillGemmResources* resources) noexcept;

[[nodiscard]] int query_sm87_a4w4_prefill_gemm_k128_resources_cuda(
    Sm87A4W4PrefillGemmResources* resources) noexcept;

// Launches one persistent projection.  The current admission accepts arbitrary
// positive M, all fixed Qwen3.6 projection N values (multiples of 128), and K
// values divisible by 64.  Tail M rows are zero-filled in the async staging
// path and never written outside D.
[[nodiscard]] int launch_sm87_a4w4_prefill_gemm_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k64_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k64_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    void* cuda_stream = nullptr) noexcept;

// M64N256 K128 shared-scale projection.  Each pair of native
// m16n8k64 instructions accumulates into one S32 fragment; the kernel then
// performs exactly one S32->FP32 conversion and one A_scale*B_scale
// application for that logical K128 contribution.
[[nodiscard]] int launch_sm87_a4w4_prefill_gemm_k128_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k128_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k128_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    void* cuda_stream = nullptr) noexcept;

static_assert(sm87_a4w4_prefill_gemm_plan(512U, 17'408U, 5'120U)
                  .work_tiles == 2'176U);
static_assert(sm87_a4w4_prefill_gemm_plan(512U, 5'120U, 17'408U)
                  .work_tiles == 640U);
static_assert(sm87_a4w4_prefill_gemm_plan(3'847U, 12'288U, 5'120U)
                  .launch_ctas == kSm87A4W4PrefillPersistentCtas);
static_assert(sm87_a4w4_prefill_gemm_plan(1U, 1'024U, 5'120U)
                  .m_tiles == 1U);
static_assert(sm87_a4w4_prefill_gemm_plan(0U, 128U, 64U).launch_ctas == 0U);
static_assert(!sm87_a4w4_prefill_uses_large_m_candidate(512U));
static_assert(!sm87_a4w4_prefill_uses_large_m_candidate(1'023U));
static_assert(sm87_a4w4_prefill_uses_large_m_candidate(1'024U));
static_assert(sm87_a4w4_prefill_uses_large_m_candidate(4'096U));
static_assert(!sm87_a4w4_prefill_uses_large_m_candidate(4'097U));
static_assert(!sm87_a4w4_prefill_uses_m64n256_candidate(1'024U, 5'120U));
static_assert(!sm87_a4w4_prefill_uses_m64n256_candidate(1'472U, 5'120U));
static_assert(sm87_a4w4_prefill_uses_m64n256_candidate(1'536U, 5'120U));
static_assert(sm87_a4w4_prefill_uses_m64n256_candidate(2'048U, 5'120U));
static_assert(!sm87_a4w4_prefill_uses_m64n256_candidate(2'048U, 5'184U));
static_assert(!sm87_a4w4_prefill_uses_m64n256_candidate(2'049U, 5'120U));
static_assert(sm87_a4w4_prefill_gemm_k64_composite_plan(
                  1'023U, 5'120U, 17'408U)
                  .prefix_token_count == 0U);
static_assert(sm87_a4w4_prefill_gemm_k64_composite_plan(
                  1'025U, 5'120U, 17'408U)
                  .prefix_kernel ==
              Sm87A4W4PrefillK64CompositePrefixKernel::kM64N64K64);
static_assert(sm87_a4w4_prefill_gemm_k64_composite_plan(
                  1'025U, 5'120U, 17'408U)
                  .tail_token_count == 1U);
static_assert(sm87_a4w4_prefill_gemm_k64_composite_plan(
                  1'804U, 5'120U, 17'408U)
                  .prefix_kernel ==
              Sm87A4W4PrefillK64CompositePrefixKernel::kM64N256K64);
static_assert(sm87_a4w4_prefill_gemm_k64_composite_plan(
                  2'049U, 5'120U, 17'408U)
                  .prefix_kernel ==
              Sm87A4W4PrefillK64CompositePrefixKernel::kM64N256K64);
static_assert(sm87_a4w4_prefill_gemm_k64_composite_plan(
                  3'987U, 5'120U, 17'408U)
                  .prefix_token_count == 3'968U);
static_assert(sm87_a4w4_prefill_gemm_k64_composite_plan(
                  3'987U, 5'120U, 17'408U)
                  .tail_token_count == 19U);
static_assert(sm87_a4w4_prefill_gemm_m64n256_plan(
                  2'048U, 17'408U, 5'120U)
                  .work_tiles == 2'176U);
static_assert(sm87_a4w4_prefill_gemm_m64n256_plan(
                  4'096U, 5'120U, 17'408U)
                  .work_tiles == 1'280U);
static_assert(sm87_a4w4_prefill_gemm_k128_plan(
                  64U, 256U, 128U)
                  .work_tiles == 1U);
static_assert(sm87_a4w4_prefill_gemm_k128_plan(
                  4'096U, 5'120U, 17'408U)
                  .k128_groups == 136U);
static_assert(sm87_a4w4_prefill_gemm_k128_plan(
                  4'096U, 5'120U, 17'408U)
                  .physical_k64_groups == 272U);
static_assert(sm87_a4w4_prefill_gemm_k128_plan(
                  4'096U, 5'120U, 17'408U)
                  .work_tiles == 1'280U);
static_assert(sm87_a4w4_prefill_gemm_k128_plan(
                  65U, 256U, 128U)
                  .launch_ctas == 0U);
static_assert(sm87_a4w4_prefill_k512_launch_token_count(1U) == 128U);
static_assert(sm87_a4w4_prefill_k512_launch_token_count(128U) == 128U);
static_assert(sm87_a4w4_prefill_k512_launch_token_count(129U) == 256U);
static_assert(sm87_a4w4_prefill_k512_group_count(6'144U) == 12U);
static_assert(sm87_a4w4_prefill_k512_scale_capacity_elements(
                  256U, 6'144U) == 3'072U);
inline constexpr std::size_t kSm87A4W4PrefillK128OverflowProbe =
    static_cast<std::size_t>(-1) & ~static_cast<std::size_t>(255U);
static_assert(sm87_a4w4_prefill_gemm_k128_plan(
                  kSm87A4W4PrefillK128OverflowProbe,
                  kSm87A4W4PrefillK128OverflowProbe, 128U)
                  .launch_ctas == 0U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_PREFILL_HOST_DEVICE
