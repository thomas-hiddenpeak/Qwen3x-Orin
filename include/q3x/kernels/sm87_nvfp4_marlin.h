#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Fixed Qwen3.6-27B/Orin contract. This API intentionally does not expose a
// generic Marlin surface: callers may use any M in [1,512] for the fixed
// BF16 x NVFP4 Gate+Up and Down projections on a 16-SM
// SM87 device. The shape selector is copied from vLLM's frozen Marlin launch:
// M1..8=M8N128K128, M9..16=M16N128K128, M17..32=M32N256K64, and
// M33..512=M64N256K64. A non-M64-aligned large request is one runner tile but
// two ordered kernel launches: its complete M64 prefix followed by the same
// <=32 specialization or one masked M64 remainder. Every specialization uses
// four pipeline stages and a persistent 16-CTA grid.
inline constexpr std::size_t kSm87NvFp4MarlinTokens = 512U;
inline constexpr std::size_t kSm87NvFp4MarlinTailMaximumTokens = 32U;
inline constexpr std::size_t kSm87NvFp4MarlinM64Tokens = 64U;
inline constexpr std::size_t kSm87NvFp4MarlinM256Tokens = 256U;
inline constexpr std::size_t kSm87NvFp4MarlinHidden = 5'120U;
inline constexpr std::size_t kSm87NvFp4MarlinIntermediate = 17'408U;
inline constexpr std::size_t kSm87NvFp4MarlinGateUpOutput = 34'816U;
inline constexpr std::size_t kSm87NvFp4MarlinSmCount = 16U;
inline constexpr std::size_t kSm87NvFp4MarlinThreadM = 64U;
inline constexpr std::size_t kSm87NvFp4MarlinThreadN = 256U;
inline constexpr std::size_t kSm87NvFp4MarlinThreadK = 64U;
inline constexpr std::size_t kSm87NvFp4MarlinPipelineStages = 4U;
inline constexpr std::size_t kSm87NvFp4MarlinDynamicSharedBytes = 166'912U;
inline constexpr std::size_t kSm87NvFp4MarlinLockBytes =
    kSm87NvFp4MarlinSmCount * sizeof(std::int32_t);
inline constexpr std::size_t kSm87NvFp4MarlinReductionElements =
    kSm87NvFp4MarlinSmCount * kSm87NvFp4MarlinThreadM *
    kSm87NvFp4MarlinThreadN;
inline constexpr std::size_t kSm87NvFp4MarlinReductionBytes =
    kSm87NvFp4MarlinReductionElements * sizeof(float);

[[nodiscard]] constexpr bool sm87_nvfp4_marlin_supports_token_count(
    const std::size_t token_count) noexcept {
  return token_count >= 1U && token_count <= kSm87NvFp4MarlinTokens;
}

struct Sm87NvFp4MarlinExecutionPlan {
  std::size_t primary_tokens = 0U;
  std::size_t remainder_tokens = 0U;
  std::size_t launch_count = 0U;
};

// Mirrors the vendored kernel's floor(prob_m/64) parallelization contract.
// Exact multiples and M33..64 need one launch. Larger nonmultiples first pass
// only their complete M64 prefix because passing the full value directly would
// make upstream Marlin intentionally ignore the remainder.
[[nodiscard]] constexpr Sm87NvFp4MarlinExecutionPlan
sm87_nvfp4_marlin_execution_plan(const std::size_t token_count) noexcept {
  if (!sm87_nvfp4_marlin_supports_token_count(token_count)) {
    return {};
  }
  if (token_count <= kSm87NvFp4MarlinM64Tokens ||
      token_count % kSm87NvFp4MarlinM64Tokens == 0U) {
    return {token_count, 0U, 1U};
  }
  const std::size_t primary =
      token_count - token_count % kSm87NvFp4MarlinM64Tokens;
  return {primary, token_count - primary, 2U};
}

struct Sm87NvFp4MarlinTileConfig {
  std::size_t thread_m = 0U;
  std::size_t thread_n = 0U;
  std::size_t thread_k = 0U;
  std::size_t threads = 0U;
  std::size_t stages = 0U;
  std::size_t persistent_ctas = 0U;
  bool m_block_size_8 = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return thread_m != 0U;
  }
};

// Exact specialization selected by vLLM ccd49f682's determine_exec_config
// for both Qwen3.6 Gate+Up (N=34816,K=5120) and Down
// (N=5120,K=17408). Neither shape triggers vLLM's low-work N64 override.
[[nodiscard]] constexpr Sm87NvFp4MarlinTileConfig
sm87_nvfp4_marlin_tile_config(const std::size_t token_count) noexcept {
  if (token_count >= 1U && token_count <= 8U) {
    return {8U, 128U, 128U, 256U, 4U, 16U, true};
  }
  if (token_count <= 16U) {
    return token_count >= 9U
               ? Sm87NvFp4MarlinTileConfig{
                     16U, 128U, 128U, 256U, 4U, 16U, false}
               : Sm87NvFp4MarlinTileConfig{};
  }
  if (token_count <= kSm87NvFp4MarlinTailMaximumTokens) {
    return {32U, 256U, 64U, 256U, 4U, 16U, false};
  }
  if (token_count >= 33U && token_count <= kSm87NvFp4MarlinTokens) {
    return {64U, 256U, 64U, 256U, 4U, 16U, false};
  }
  return {};
}

static_assert(sm87_nvfp4_marlin_tile_config(1U).thread_m == 8U);
static_assert(sm87_nvfp4_marlin_tile_config(9U).thread_k == 128U);
static_assert(sm87_nvfp4_marlin_tile_config(31U).thread_m == 32U);
static_assert(sm87_nvfp4_marlin_tile_config(64U).thread_n == 256U);
static_assert(sm87_nvfp4_marlin_tile_config(256U).thread_m == 64U);
static_assert(sm87_nvfp4_marlin_tile_config(33U).thread_m == 64U);
static_assert(sm87_nvfp4_marlin_tile_config(407U).thread_k == 64U);
static_assert(sm87_nvfp4_marlin_execution_plan(407U).primary_tokens == 384U);
static_assert(sm87_nvfp4_marlin_execution_plan(407U).remainder_tokens == 23U);
static_assert(sm87_nvfp4_marlin_execution_plan(481U).primary_tokens == 448U);
static_assert(sm87_nvfp4_marlin_execution_plan(481U).remainder_tokens == 33U);
static_assert(sm87_nvfp4_marlin_execution_plan(512U).launch_count == 1U);

[[nodiscard]] constexpr std::size_t sm87_nvfp4_marlin_weight_bytes(
    const std::size_t output_size, const std::size_t input_size) noexcept {
  return output_size * input_size / 2U;
}

[[nodiscard]] constexpr std::size_t sm87_nvfp4_marlin_scale_bytes(
    const std::size_t output_size, const std::size_t input_size) noexcept {
  return output_size * input_size / 16U;
}

// Derives the same power-of-two scale factor as vLLM's
// nvfp4_marlin_process_scales. Inputs are authenticated canonical ModelOpt
// E4M3FN scale payloads in host memory. Gate+Up passes both tensors so the
// merged projection uses one common factor. Negative/NaN encodings fail.
[[nodiscard]] bool derive_sm87_nvfp4_marlin_scale_factor(
    const std::uint8_t* first_scales, std::size_t first_scale_bytes,
    const std::uint8_t* second_scales, std::size_t second_scale_bytes,
    float* scale_factor) noexcept;

// Load-time-only transformation from two canonical ModelOpt matrices
// [17408,5120/2] into the single vLLM Marlin N=34816 layout. transpose_scratch
// is a disposable device buffer exactly as large as marlin_weight. The output
// block-scale tensor uses vLLM's permuted S0E5M3 byte representation. The
// supplied scale factor must be derived jointly from both scale payloads.
[[nodiscard]] int prepare_sm87_nvfp4_marlin_gate_up_cuda(
    const std::uint8_t* canonical_gate_weight,
    const std::uint8_t* canonical_up_weight,
    const std::uint8_t* canonical_gate_scales,
    const std::uint8_t* canonical_up_scales,
    const float* canonical_shared_weight_scale_2_device,
    float scale_factor, std::uint8_t* marlin_weight,
    std::uint8_t* marlin_scales, float* marlin_global_scale,
    void* transpose_scratch, std::size_t transpose_scratch_bytes,
    void* cuda_stream = nullptr) noexcept;

// Load-time-only transformation for canonical Down [5120,17408/2].
[[nodiscard]] int prepare_sm87_nvfp4_marlin_down_cuda(
    const std::uint8_t* canonical_weight,
    const std::uint8_t* canonical_scales,
    const float* canonical_weight_scale_2_device, float scale_factor,
    std::uint8_t* marlin_weight, std::uint8_t* marlin_scales,
    float* marlin_global_scale, void* transpose_scratch,
    std::size_t transpose_scratch_bytes,
    void* cuda_stream = nullptr) noexcept;

// Locks must be zero-initialized before their first launch. The vendored
// stripe scheduler returns them to zero, so one workspace is reusable on an
// ordered stream. C_tmp is the fixed FP32 cross-CTA reduction workspace.
[[nodiscard]] int launch_sm87_nvfp4_marlin_gate_up_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales, const float* marlin_global_scale,
    std::size_t token_count, std::uint16_t* merged_gate_up_output,
    float* c_tmp, std::int32_t* locks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_nvfp4_marlin_down_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales, const float* marlin_global_scale,
    std::size_t token_count, std::uint16_t* output, float* c_tmp,
    std::int32_t* locks, void* cuda_stream = nullptr) noexcept;

// Admission-only large-M Down epilogue for the fixed Qwen3.6 shape. Marlin's
// ordinary BF16-rounded Down result is added to the BF16 residual and written
// directly to output, removing the standalone residual-add/intermediate
// writeback. residual and output must be distinct 16-byte-aligned matrices of
// shape [token_count,5120].
[[nodiscard]] int launch_sm87_nvfp4_marlin_down_residual_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales, const float* marlin_global_scale,
    std::size_t token_count, const std::uint16_t* residual,
    std::uint16_t* output, float* c_tmp, std::int32_t* locks,
    void* cuda_stream = nullptr) noexcept;

// Converts row-major merged [M,34816] into canonical BF16
// SiLU(gate)*up [M,17408]. M follows the same C64/C256/C512 contract.
[[nodiscard]] int launch_sm87_nvfp4_marlin_gate_up_silu_cuda(
    const std::uint16_t* merged_gate_up, std::size_t token_count,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// Compatibility aliases retained for the authenticated layer-0 C512 test.
[[nodiscard]] int launch_sm87_nvfp4_marlin_gate_up_m512_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales, const float* marlin_global_scale,
    std::uint16_t* merged_gate_up_output, float* c_tmp,
    std::int32_t* locks, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_nvfp4_marlin_down_m512_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales, const float* marlin_global_scale,
    std::uint16_t* output, float* c_tmp, std::int32_t* locks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_nvfp4_marlin_gate_up_silu_m512_cuda(
    const std::uint16_t* merged_gate_up, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

struct Sm87NvFp4MarlinKernelResources {
  int registers_per_thread = 0;
  int static_shared_bytes = 0;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

[[nodiscard]] int query_sm87_nvfp4_marlin_m512_resources_cuda(
    Sm87NvFp4MarlinKernelResources* resources) noexcept;

}  // namespace q3x::kernels
